//! Targeted flip-seam + back-ref-across-seam correctness net.
//!
//! WHY (campaign danger zone): the copy-free-to-final clean-tail refactor
//! changes how the post-flip clean tail is stored (clean u8 written DIRECTLY
//! into `chunk.data`'s reserved tail; back-refs resolve from that tail; no
//! `output_ring` for the clean phase) and how the flip seam is addressed
//! (`conflate_to_clean_u8`'s `base = ring_pos - w` + `% RING_U8_SIZE` window
//! downcast). The classic break classes are:
//!   (a) an off-by-one in seam addressing (window placed one byte off),
//!   (b) a post-flip back-ref reading UNINITIALIZED reserved tail,
//!   (c) a back-ref at distance up to 32768 that should reach across the seam
//!       into the OLDEST pre-flip window byte but reads the wrong byte,
//!   (d) the >16 MiB reserve-clamp fallback mis-sizing the contiguous tail.
//!
//! This file drives the PRODUCTION FOLD PATH directly —
//! `decode_chunk_window_absent` (Engine M: u16 markers for the window-absent
//! prefix, FLIP to clean at 32 KiB, FOLD in place to the end) then
//! `resolve_and_narrow_markers_in_place` + `merge_resolved_markers_into_data`
//! against the TRUE predecessor window — exactly what a real mid-stream chunk
//! goes through. Unlike `chunk_decode::native_fold_parity` (which needs the
//! silesia file on disk) these are SYNTHETIC + self-contained, so they run in
//! CI and can be aimed at specific seam edge cases.
//!
//! Ground truth: the slice `full_decode[out_off .. out_off + chunk_len]` of a
//! single straight decode of the same raw-deflate stream. A divergence is a
//! seam/fold bug. Each fixture is engineered so the chunk decode actually
//! FLIPS (marker prefix present AND clean tail present) — asserted — otherwise
//! the test would silently pass without touching the seam.
//!
//! Run: `cargo test --release --features pure-rust-inflate seam_crossing`.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::Write;

    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_decode::decode_chunk_window_absent;
    use crate::decompress::parallel::inflate_wrapper::{StoppingPoints, StreamingInflateWrapper};

    const WINDOW: usize = 32 * 1024;

    // ── Raw-deflate helpers ───────────────────────────────────────────────────

    /// Compress `payload` to a RAW deflate stream (no gzip/zlib wrapper).
    fn raw_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    struct Boundary {
        bit: usize,
        out_off: usize,
    }

    /// Single straight decode of the whole raw-deflate member, recording every
    /// END_OF_BLOCK boundary (`bit` = start of next header, `out_off` = decoded
    /// bytes just past the finished block). This is the GROUND TRUTH oracle —
    /// it's gzippy's own wrapper run start-to-end with a known empty window,
    /// which we cross-check against flate2 below so a wrapper bug can't make
    /// this a self-fulfilling oracle.
    fn enumerate(input: &[u8]) -> (Vec<u8>, Vec<Boundary>) {
        let mut wrapper =
            StreamingInflateWrapper::with_until_bits(input, 0, input.len() * 8).expect("init");
        wrapper.set_window(&[]).expect("empty window");
        wrapper.set_stopping_points(
            StoppingPoints::END_OF_BLOCK
                | StoppingPoints::END_OF_BLOCK_HEADER
                | StoppingPoints::END_OF_STREAM_HEADER,
        );
        let mut decoded: Vec<u8> = Vec::new();
        let mut bounds: Vec<Boundary> = Vec::new();
        let mut buf = vec![0u8; 128 * 1024];
        let mut last_bit = 0usize;
        loop {
            let r = wrapper.read_stream(&mut buf).expect("read_stream");
            decoded.extend_from_slice(&buf[..r.bytes_written]);
            if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                bounds.push(Boundary {
                    bit: r.bit_position,
                    out_off: decoded.len(),
                });
            }
            if r.finished {
                break;
            }
            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && r.bit_position == last_bit
            {
                break;
            }
            last_bit = r.bit_position;
        }
        (decoded, bounds)
    }

    /// Run the production fold path on `input` starting at `start_bit` with NO
    /// window (Engine M emits markers, flips to clean at 32 KiB, folds), then
    /// resolve the markers against the true 32 KiB predecessor `window` and
    /// fold them into one contiguous buffer. Returns (full_bytes, markers_len,
    /// clean_len, final_bit).
    fn fold_from(
        input: &[u8],
        start_bit: usize,
        stop_hint: usize,
        window: &[u8],
    ) -> (Vec<u8>, usize, usize, usize) {
        assert_eq!(window.len(), WINDOW, "resolution window must be 32 KiB");
        let cfg = ChunkConfiguration::default();
        let mut chunk = decode_chunk_window_absent(input, start_bit, stop_hint, cfg)
            .expect("window-absent fold decode");
        assert_eq!(chunk.data_prefix_len, 0, "unexpected window-image prefix");
        let final_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let markers_len = chunk.data_with_markers.len();
        let clean_len = chunk.data.len();
        chunk.resolve_and_narrow_markers_in_place(window);
        chunk.merge_resolved_markers_into_data();
        assert!(
            chunk.data_with_markers.is_empty(),
            "markers not consumed by resolve+merge"
        );
        let full = chunk.data.to_contiguous();
        assert_eq!(
            full.len(),
            markers_len + clean_len,
            "resolve length mismatch"
        );
        (full, markers_len, clean_len, final_bit)
    }

    /// Cross-check the wrapper ground-truth against flate2 so the oracle is
    /// independent (a wrapper bug can't silently agree with itself).
    fn flate2_decode_raw(input: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut d = flate2::read::DeflateDecoder::new(input);
        let mut out = Vec::new();
        d.read_to_end(&mut out).expect("flate2 raw decode");
        out
    }

    /// The core driver: for a synthetic raw-deflate `payload`, decode the whole
    /// stream once (ground truth), cross-check vs flate2, then walk start
    /// boundaries that (a) have a full 32 KiB of preceding output for the
    /// predecessor window and (b) have at least `lookahead` boundaries of room
    /// ahead. For each, run the production fold path with a BOUNDED stop hint
    /// (`bounds[start+lookahead].bit`, mirroring `chunk_decode::native_fold_parity`
    /// — an unbounded hint overruns the slice) and assert byte-equality of the
    /// chunk's full payload against the ground-truth slice. Returns the number
    /// of chunks that actually FLIPPED (clean tail extends past the 32 KiB
    /// window — i.e. Engine M crossed the seam and continued in place), so
    /// callers can assert the seam machinery was exercised.
    fn drive_seam(payload: &[u8], level: u32, label: &str, _require_flip: bool) -> usize {
        // Target clean-tail span past the seam (well over the 32 KiB window so
        // the chunk reliably FLIPS regardless of how zlib sized its blocks).
        const SPAN: usize = 256 * 1024;
        let input = raw_deflate(payload, level);
        let (decoded, bounds) = enumerate(&input);
        assert_eq!(
            decoded,
            flate2_decode_raw(&input),
            "{label}: wrapper ground-truth disagrees with flate2 — oracle is suspect"
        );
        assert_eq!(decoded, payload, "{label}: decode != original payload");
        assert!(
            bounds.len() >= 3,
            "{label}: need >=3 deflate boundaries to test seams, got {}",
            bounds.len()
        );

        // Pick the stop boundary for a start at decoded offset `out_off`: the
        // first boundary whose output offset is >= out_off + SPAN (span-based,
        // robust to few-huge vs many-small block layouts), else the last.
        let pick_stop = |out_off: usize| -> Option<usize> {
            bounds
                .iter()
                .position(|b| b.out_off >= out_off + SPAN)
                .or_else(|| Some(bounds.len() - 1))
        };

        let mut flips = 0usize;
        let mut seam_checks = 0usize;
        let step = (bounds.len() / 16).max(1);
        let mut idx = 1usize;
        while idx + 1 < bounds.len() {
            let b = &bounds[idx];
            if b.out_off < WINDOW {
                idx += step;
                continue;
            }
            let Some(stop_idx) = pick_stop(b.out_off) else {
                break;
            };
            if stop_idx <= idx {
                idx += step;
                continue;
            }
            let window = &decoded[b.out_off - WINDOW..b.out_off];
            let stop_hint = bounds[stop_idx].bit;
            let (full, markers_len, clean_len, _final_bit) =
                fold_from(&input, b.bit, stop_hint, window);

            // Ground truth: the chunk's bytes are exactly the decoded slice
            // starting at this boundary's output offset.
            assert!(
                b.out_off + full.len() <= decoded.len(),
                "{label}[bnd {idx}]: chunk extends past ground truth"
            );
            let gt = &decoded[b.out_off..b.out_off + full.len()];
            if full != gt {
                let d = full
                    .iter()
                    .zip(gt.iter())
                    .position(|(a, c)| a != c)
                    .unwrap_or(0);
                panic!(
                    "{label}[bnd {idx}]: SEAM/FOLD DIVERGE at chunk-offset {d} \
                     (got=0x{:02x} truth=0x{:02x}); markers_len={markers_len} clean_len={clean_len} \
                     (offset {d} is {} the flip seam at {markers_len})",
                    full[d],
                    gt[d],
                    if d < markers_len { "BEFORE" } else { "AT/AFTER" }
                );
            }
            seam_checks += 1;
            // A flip means Engine M crossed the 32 KiB seam: the clean tail is
            // strictly longer than the window. (markers_len ~ 32 KiB always.)
            if clean_len > WINDOW {
                flips += 1;
            }
            eprintln!(
                "[seam {label}] bnd {idx}: full={} markers={markers_len} clean={clean_len} flip={}",
                full.len(),
                clean_len > WINDOW
            );
            idx += step;
        }
        assert!(
            seam_checks > 0,
            "{label}: no boundary produced a 32-KiB-preceded chunk; test exercised no seam"
        );
        flips
    }

    // ── Synthetic payload generators aimed at the seam ────────────────────────

    /// Long-period unique block repeated → near-max-distance back-refs that, in
    /// a mid-stream chunk, reach across the flip seam into the oldest window.
    fn periodic_max_distance(raw: usize, period: usize) -> Vec<u8> {
        let mut base = vec![0u8; period];
        let mut rng: u64 = 0xa5a5_5a5a_dead_c0de ^ (period as u64);
        for b in &mut base {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 24) as u8;
        }
        let mut out = Vec::with_capacity(raw);
        while out.len() < raw {
            let take = (raw - out.len()).min(base.len());
            out.extend_from_slice(&base[..take]);
        }
        out.truncate(raw);
        out
    }

    /// Mixed: PRNG literals interleaved with periodic motifs so the stream has
    /// many block boundaries AND back-refs of varied distance (short, medium,
    /// near-32K) — the broadest cross-seam distance coverage.
    fn mixed_distances(raw: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(raw);
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let motif: Vec<u8> = (0..4096).map(|i| (i * 7 + 13) as u8).collect();
        while out.len() < raw {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            match (rng >> 40) % 4 {
                0 => {
                    // PRNG literal burst (forces literals / new blocks).
                    for _ in 0..256 {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        out.push((rng >> 24) as u8);
                        if out.len() >= raw {
                            break;
                        }
                    }
                }
                _ => {
                    // Motif repeat (forms back-refs at the motif period).
                    let take = (raw - out.len()).min(motif.len());
                    out.extend_from_slice(&motif[..take]);
                }
            }
        }
        out.truncate(raw);
        out
    }

    // ── Tests ─────────────────────────────────────────────────────────────────
    //
    // The flip (Engine M crossing the 32 KiB seam and continuing clean in
    // place) only fires once 32 KiB of CONSECUTIVE marker-free output
    // accumulates — i.e. when no back-ref in that span reaches before the
    // chunk's decode start (`maybe_flip`, lut_bulk_inflate.rs:864). Periodic
    // data with period >= 32 KiB reaches that state (after one period every
    // back-ref is self-referential), so those fixtures reliably FLIP and
    // exercise the post-flip clean-tail + cross-seam back-ref path. Periodic
    // data with a SHORT period stays in the marker phase (a back-ref keeps
    // reaching into the predecessor window) — those fixtures don't flip but
    // still give byte-exact coverage of the marker-resolution + merge path on
    // multi-MiB marker zones (`require_flip = false`).

    /// Near-32768-distance back-refs spanning the flip seam — the central case.
    /// Period 32767 forces zlib to emit back-refs at the maximum DEFLATE
    /// distance, which in a mid-stream chunk land across the seam into the
    /// OLDEST pre-flip window byte. MUST flip.
    #[test]
    fn seam_max_distance_backrefs_across_flip() {
        let payload = periodic_max_distance(8 * 1024 * 1024, 32 * 1024 - 1);
        let flips = drive_seam(&payload, 6, "max_distance", true);
        assert!(
            flips > 0,
            "max_distance: no chunk flipped marker→clean; seam fold not exercised"
        );
    }

    /// Period just OVER 32 KiB → max-distance refs AND a clean self-referential
    /// tail past the seam. MUST flip; the post-flip back-refs read the freshly
    /// folded clean tail (the copy-free-to-final danger zone).
    #[test]
    fn seam_period_over_window_flips() {
        let payload = periodic_max_distance(8 * 1024 * 1024, 40 * 1024 + 7);
        let flips = drive_seam(&payload, 6, "period_over_window", true);
        assert!(flips > 0, "period_over_window: seam fold not exercised");
    }

    /// Medium-period back-refs (8 KiB): stays marker-heavy — byte-exact coverage
    /// of marker resolution + merge over large marker zones (no flip required).
    #[test]
    fn seam_medium_distance_marker_resolution() {
        let payload = periodic_max_distance(6 * 1024 * 1024, 8 * 1024 + 123);
        drive_seam(&payload, 6, "medium_distance", false);
    }

    /// Broad mixed-distance coverage: short + medium + near-max back-refs and
    /// many block boundaries (byte-exact; flip not required).
    #[test]
    fn seam_mixed_distances() {
        let payload = mixed_distances(8 * 1024 * 1024);
        drive_seam(&payload, 6, "mixed_distances", false);
    }

    /// L9 maximizes match finding → genuine long matches AND maximal dynamic
    /// table churn over a period-over-window shape that flips. MUST flip.
    #[test]
    fn seam_l9_dynamic_heavy() {
        let payload = periodic_max_distance(7 * 1024 * 1024, 36 * 1024 + 1);
        let flips = drive_seam(&payload, 9, "l9_dynamic", true);
        assert!(flips > 0, "l9_dynamic: seam fold not exercised");
    }

    /// Consecutive-chunk SEAM HANDOFF: decode a chunk, then start the NEXT
    /// fold at the first chunk's final bit with the first chunk's trailing
    /// 32 KiB as the predecessor window. The boundary between the two chunks
    /// must be byte-continuous with the ground truth — this is the exact
    /// flip-in-place seam the refactor must preserve across chunk borders.
    #[test]
    fn seam_consecutive_chunk_handoff() {
        let payload = periodic_max_distance(12 * 1024 * 1024, 32 * 1024 - 1);
        let input = raw_deflate(&payload, 6);
        let (decoded, bounds) = enumerate(&input);
        assert_eq!(decoded, flate2_decode_raw(&input), "oracle mismatch");
        let boundary_bits: std::collections::HashSet<usize> =
            bounds.iter().map(|b| b.bit).collect();
        const SPAN: usize = 256 * 1024;
        assert!(bounds.len() >= 3, "need boundaries, got {}", bounds.len());
        let pick_stop = |out_off: usize| -> usize {
            bounds
                .iter()
                .position(|b| b.out_off >= out_off + SPAN)
                .map(|i| bounds[i].bit)
                .unwrap_or_else(|| bounds[bounds.len() - 1].bit)
        };

        // First chunk: an early boundary with a full preceding window and a
        // span-bounded stop hint so it flips (clean tail > 32 KiB).
        let first_idx = bounds
            .iter()
            .position(|b| b.out_off >= WINDOW)
            .expect("need a 32-KiB-preceded start boundary");
        let first = &bounds[first_idx];
        let w0 = decoded[first.out_off - WINDOW..first.out_off].to_vec();
        let stop1 = pick_stop(first.out_off);
        let (full1, _m1, c1, final_bit) = fold_from(&input, first.bit, stop1, &w0);
        assert!(c1 > WINDOW, "first chunk must flip (clean tail > 32 KiB)");
        let gt1 = &decoded[first.out_off..first.out_off + full1.len()];
        assert_eq!(full1, gt1, "first chunk diverged from ground truth");

        // final_bit must land on a real EOB boundary (the seam handoff point).
        assert!(
            boundary_bits.contains(&final_bit),
            "first chunk final_bit {final_bit} is not a real block boundary"
        );

        // Second chunk: start at the seam, seeded with the FIRST chunk's
        // trailing 32 KiB (the real cross-chunk handoff window). Its bytes must
        // continue the output byte-for-byte — this is the seam the consumer
        // reconciles, and the exact place a copy-free-to-final off-by-one would
        // corrupt the first cross-chunk back-reference.
        let seam_out_off = first.out_off + full1.len();
        if seam_out_off + WINDOW >= decoded.len() {
            eprintln!("seam handoff: first chunk consumed the stream; single-chunk case only");
            return;
        }
        assert!(full1.len() >= WINDOW, "first chunk shorter than a window");
        let w1 = full1[full1.len() - WINDOW..].to_vec();
        let stop2 = pick_stop(seam_out_off);
        let (full2, _m2, _c2, _fb2) = fold_from(&input, final_bit, stop2, &w1);
        let gt2 = &decoded[seam_out_off..seam_out_off + full2.len()];
        if full2 != gt2 {
            let d = full2
                .iter()
                .zip(gt2.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            panic!(
                "SEAM HANDOFF DIVERGE at chunk2-offset {d} (got=0x{:02x} truth=0x{:02x}); \
                 the cross-chunk seam (back-ref from chunk2 into chunk1's tail window) is wrong",
                full2[d], gt2[d]
            );
        }
    }

    // ── (a) Window-extremity distances (advisor item a) ───────────────────────
    //
    // An off-by-one in seam addressing only manifests if a back-ref actually
    // touches the EXTREME window byte. These pin a period of EXACTLY 32768 (the
    // max DEFLATE distance == window length), so the periodic back-ref reads the
    // very oldest pre-flip window byte every period — the one position a +/-1
    // seam-address error reads wrong. Asserts a flip AND byte-exactness.

    #[test]
    fn seam_distance_exactly_window_length() {
        // Period == 32768 (= MAX window): back-ref distance lands on the window
        // extremity. (32768 itself is a legal DEFLATE distance.)
        let payload = periodic_max_distance(8 * 1024 * 1024, 32 * 1024);
        let flips = drive_seam(&payload, 6, "distance_eq_window", true);
        assert!(flips > 0, "distance_eq_window: seam fold not exercised");
    }

    #[test]
    fn seam_distance_one_under_window() {
        // Period == 32767: the largest distance strictly inside the window —
        // the neighbouring extremity to the exact-window case above.
        let payload = periodic_max_distance(8 * 1024 * 1024, 32 * 1024 - 1);
        let flips = drive_seam(&payload, 9, "distance_window_minus_1", true);
        assert!(
            flips > 0,
            "distance_window_minus_1: seam fold not exercised"
        );
    }

    // ── (c) >16 MiB reserve-clamp boundary (advisor item c) ───────────────────
    //
    // `decode_chunk_with_rapidgzip_impl` pre-reserves the clean tail as
    // `compressed*8 + 1 MiB` CLAMPED to `RESERVE_CLAMP = 16 MiB`
    // (chunk_decode.rs:1001). Two failure modes live at that boundary: the clamp
    // DECISION (just-under vs just-over) and the CONTIGUITY of the tail once the
    // clamp under-reserves and the buffer must amortized-regrow (realloc + move)
    // — a post-flip back-ref must still resolve correctly across that realloc.
    // These drive a SINGLE folded chunk whose clean tail straddles 16 MiB, with
    // a periodic back-ref that keeps reading across the growing region, and
    // assert byte-exactness vs the straight-decode slice. The reserved spare is
    // poisoned (0xCD) in test builds, so a read-before-write across the regrow
    // boundary corrupts deterministically.

    fn drive_single_large_chunk(payload: &[u8], level: u32, label: &str, min_clean: usize) {
        let input = raw_deflate(payload, level);
        let (decoded, bounds) = enumerate(&input);
        assert_eq!(
            decoded,
            flate2_decode_raw(&input),
            "{label}: oracle mismatch"
        );
        assert_eq!(decoded, payload, "{label}: decode != payload");

        let start_idx = bounds
            .iter()
            .position(|b| b.out_off >= WINDOW)
            .expect("need a 32-KiB-preceded start boundary");
        let start = &bounds[start_idx];
        let window = decoded[start.out_off - WINDOW..start.out_off].to_vec();
        // Stop hint: the first boundary whose output is >= start + min_clean, so
        // the clean tail spans `min_clean` (straddling the 16 MiB clamp).
        let stop = bounds
            .iter()
            .find(|b| b.out_off >= start.out_off + min_clean)
            .map(|b| b.bit)
            .unwrap_or_else(|| bounds[bounds.len() - 1].bit);

        let (full, _markers_len, clean_len, _final_bit) =
            fold_from(&input, start.bit, stop, &window);
        assert!(
            clean_len > WINDOW,
            "{label}: chunk did not flip (clean_len={clean_len})"
        );
        let gt = &decoded[start.out_off..start.out_off + full.len()];
        if full != gt {
            let d = full
                .iter()
                .zip(gt.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            panic!(
                "{label}: DIVERGE at chunk-offset {d} (got=0x{:02x} truth=0x{:02x}); \
                 clean_len={clean_len} — reserve-clamp / regrow contiguity bug",
                full[d], gt[d]
            );
        }
        eprintln!("[seam {label}] OK full={} clean={clean_len}", full.len());
    }

    #[test]
    fn seam_reserve_clamp_just_under_16mib() {
        // Period >= 32768 so the chunk reliably FLIPS, then a long
        // self-referential clean tail of ~15 MiB (estimate < 16 MiB clamp →
        // clamp inactive, a single reserve covers the tail).
        let payload = periodic_max_distance(20 * 1024 * 1024, 36 * 1024 + 1);
        drive_single_large_chunk(&payload, 6, "clamp_under", 15 * 1024 * 1024);
    }

    #[test]
    fn seam_reserve_clamp_over_16mib() {
        // Clean tail ~24 MiB (> 16 MiB clamp → reserve under-sizes → the tail
        // must amortized-regrow; post-flip back-refs read across the realloc).
        let payload = periodic_max_distance(30 * 1024 * 1024, 36 * 1024 + 1);
        drive_single_large_chunk(&payload, 6, "clamp_over", 24 * 1024 * 1024);
    }
}
