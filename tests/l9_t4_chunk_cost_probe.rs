//! Probe 1 (final-lap lever #364 investigation, agent-16): the L9/T>1
//! config-cost matrix.
//!
//! Times the EXACT production chunk entry (`encode_deflate_splice_chunk_to_sink`)
//! over one 1.8 MB chunk + 32 KiB prefix dictionary, for the variants that
//! discriminate the pigz-L9/T4-loss hypotheses:
//!
//! 1. production: `parallel=true, level=9`  → `params_parallel(9)` ≡
//!    `params_parallel(11)` with the trunk's L9 retune: near-optimal, depth
//!    400-alias knobs, **passes 2** (level.rs `max_optim_passes = 2`).
//! 2. T1 engine:  `parallel=false, level=9` → `params(9)` (Lazy2, depth 600,
//!    nice 258 — the ldx-shaped parse the T1 cells ship).
//! 3. control:    `parallel=true, level=11` (L11 default knobs, **passes 4** —
//!    since the retune this NO LONGER aliases variant 1: wall(1)/wall(3)
//!    now prices the passes 2 vs 4 delta).
//! 4. control:    `parallel=false, level=11`.
//!
//! READS: best-of-k wall via std::time::Instant + output bytes (the size
//! ledger side; the near-opt upgrade was *paid for* in size).
//!
//! DISCRIMINATION (docs/board/sprint-2026-09-25.md):
//!   wall(1)/wall(2) >= 2.5  → the near-opt upgrade is the multiplier
//!                             (agent-16 predicted 3.5-4.5x from 61f0f01d)
//!   wall(1) ~ wall(2)       → REFUTED; escalate to scheduler phase accounting
//!   wall(1)/wall(3)         → the passes-2 retune's own chunk-level price.
//!
//! Measurement only (`#[ignore]`), never a regression gate.
//!
//! The `parallel_flush_probes` module below (feature `near-opt-parallel-flush`,
//! lever ledger row 3) adds the flush-dispatch variants to this matrix:
//! serial-vs-parallel flush timing on the probe corpus, and the
//! whole-chunk byte-identity gate serial-vs-pool across the probe corpus and
//! the four campaign corpora at the L11-T1 and L9-T4 shapes.

use std::time::Instant;

const CHUNK: usize = 1_800_000;
const DICT: usize = 32 * 1024;

fn build_corpus() -> Vec<u8> {
    if let Ok(path) = std::env::var("PROBE_CORPUS") {
        // real-corporus mode: the file must be at least DICT + CHUNK + 4096
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }

    // non-periodic deterministic corpus: interleave the four frozen fixtures
    let mut base = Vec::new();
    for name in gzippy::fixtures::NAMES {
        base.extend(gzippy::fixtures::generate(name));
    }
    let mut cursor = [0usize; 4];
    let target = CHUNK + DICT + 4096;
    while base.len() < target {
        for (i, name) in gzippy::fixtures::NAMES.iter().enumerate() {
            let piece = gzippy::fixtures::generate(name);
            let start = cursor[i].min(piece.len());
            let take = (piece.len() / 8)
                .max(4096)
                .min(piece.len().saturating_sub(start));
            base.extend_from_slice(&piece[start..start + take]);
            cursor[i] = (start + take) % piece.len().max(1);
        }
    }
    base
}

fn run(
    label: &str,
    body: &[u8],
    dict: &[u8],
    level: u8,
    parallel: bool,
    input_total_len: usize,
    runs: usize,
) -> (f64, usize) {
    let mut times = Vec::with_capacity(runs);
    let mut bytes = 0usize;
    for _ in 0..2 {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
    }
    for _ in 0..runs {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let t = Instant::now();
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
        times.push(t.elapsed().as_secs_f64());
        bytes = out.len();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    println!("{label}: best {best:.4}s bytes {bytes}");
    (best, bytes)
}

fn matrix_for(level: u8, parallel: bool, label: &'static str, runs: usize) {
    let data = build_corpus();
    let body = &data[DICT..DICT + CHUNK];
    let dict = &data[..DICT];
    run(label, body, dict, level, parallel, 200 * 1024 * 1024, runs);
}

#[test]
#[ignore]
fn prod_l9t4_depth400() {
    matrix_for(
        9,
        true,
        "1 production (near-opt@L11, depth400, passes2 - the L9 retune)",
        5,
    );
}

#[test]
#[ignore]
fn t1_engine_lazy2_depth600() {
    matrix_for(9, false, "2 T1 engine (Lazy2 depth600)", 5);
}

#[test]
#[ignore]
fn l11_control_passes4() {
    matrix_for(
        11,
        true,
        "3 control (L11 default knobs, passes4) - NOT an alias since the L9 passes2 retune: wall(1)/wall(3) prices the passes delta",
        5,
    );
}

#[test]
#[ignore]
fn depth_split_d100() {
    matrix_for(9, true, "4 nearoptimal:100:150 (depth100 ladder)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p1() {
    matrix_for(9, true, "5 nearoptimal:400:150:1 (depth400, passes1)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p4() {
    matrix_for(
        9,
        true,
        "6 nearoptimal:400:150:4 (L11 default, passes 4)",
        3,
    );
}

// ─────────────────────────────────────────────────────────────────────────
// LEVER #3 (feature `near-opt-parallel-flush`, DEFAULT OFF): the block-
// parallel optimize_and_flush variants. Without the feature this module
// compiles to nothing and the matrix above is the whole file.
// ─────────────────────────────────────────────────────────────────────────
#[cfg(feature = "near-opt-parallel-flush")]
mod parallel_flush_probes {
    use super::*;
    use gzippy::compress::deflate::parse::near_opt_flush_probe;

    /// Deterministic incompressible noise (own LCG, mirroring the bitstream
    /// tests' fixture style). Appended as a small tail to one identity
    /// corpus so the CHUNK's final partial block exercises the STORED
    /// replay path (`BitWriter::append_fragment`'s BTYPE=00 arm) inside the
    /// real pipeline, not just the unit test.
    fn noise(n: usize) -> Vec<u8> {
        let mut s: u32 = 0x5EED_1234;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            out.push((s >> 24) as u8);
        }
        out
    }

    /// The L9 production chunk shape (the exact `matrix_for` entry): chunk
    /// + 32 KiB dictionary, `parallel=true`. `pool` toggles the
    /// near-opt flush pool — with it off this IS the serial reference.
    fn encode_l9t4(
        body: &[u8],
        dict: &[u8],
        input_total_len: usize,
        pool: bool,
    ) -> (Vec<u8>, u8, bool) {
        near_opt_flush_probe::set_force_serial_flush(!pool);
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            9,
            true,
            &mut out,
            true,
            input_total_len,
        );
        (out, meta.pad_bits, meta.needs_alignment)
    }

    /// The L11-T1 whole-buffer shape (the T1 route through the same
    /// near-optimal parser).
    fn encode_l11t1(data: &[u8], pool: bool) -> Vec<u8> {
        near_opt_flush_probe::set_force_serial_flush(!pool);
        gzippy::compress::deflate::encode_deflate_bytes_to_vec(data, 11)
    }

    /// Variant 7 of the matrix: the SAME production chunk through the SAME
    /// entry, serial flush vs block-parallel flush pool, best-of-5 each,
    /// plus the byte-identity verdict between the two arms.
    #[test]
    #[ignore]
    fn flush_serial_vs_parallel() {
        let data = build_corpus();
        let body = &data[DICT..DICT + CHUNK];
        let dict = &data[..DICT];
        // Two warmups so the pool spawn and the flush workers' first-touch
        // allocations land outside the measured window (the same shape the
        // `run` helper below uses).
        let (ref_bytes, ref_pad, ref_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
        let (par_bytes, par_pad, par_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
        let identical = ref_bytes == par_bytes && ref_pad == par_pad && ref_align == par_align;

        let mut serial_times = Vec::new();
        let mut parallel_times = Vec::new();
        let runs = 5;
        for _ in 0..runs {
            let t = Instant::now();
            let _ = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            serial_times.push(t.elapsed().as_secs_f64());
        }
        for _ in 0..runs {
            let t = Instant::now();
            let _ = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            parallel_times.push(t.elapsed().as_secs_f64());
        }
        serial_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        parallel_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (serial, parallel) = (serial_times[0], parallel_times[0]);
        println!(
            "7 parallel-flush serial   : best {serial:.4}s bytes {}",
            ref_bytes.len()
        );
        println!(
            "7 parallel-flush parallel : best {parallel:.4}s bytes {} (ratio {:.3}x)",
            par_bytes.len(),
            parallel / serial
        );
        println!(
            "7 parallel-flush identity : {}",
            if identical {
                "BYTE-IDENTICAL (whole chunk)"
            } else {
                "DIVERGED"
            }
        );
        assert!(
            identical,
            "flush pool must not move a single bit of the chunk stream"
        );
    }

    /// Whole-stream byte identity, serial reference vs flush pool, at the
    /// L11-T1 and L9-T4 shapes, on the probe corpus plus the four mission
    /// corpora (the campaign files fall back to a note when the local
    /// checkout does not carry them — the synthetic probe corpus is always
    /// present). Also proves, per parallel arm, that the pool actually
    /// engaged (`parallel_flush_writes` moves) and the stale-flag guard
    /// never fired, then exercises the guard's fallback half by latching
    /// it and re-checking serial bytes.
    #[test]
    fn chunks_are_byte_identical_to_the_serial_stream() {
        // The four campaign corpora, sliced like the production chunk probe
        // (32 KiB dict + chunk body). silesia.tar / software.archive are
        // campaign-local fixtures and may be absent on other checkouts;
        // text-1MB / alice are tracked.
        let mut corpora: Vec<(String, Vec<u8>)> = Vec::new();
        {
            let data = build_corpus();
            let mut noisy = data.clone();
            // The final partial block of a chunk near the end is small;
            // appending incompressible noise makes its emission price
            // STORED, so the write-back crosses
            // `BitWriter::append_fragment`'s stored replay in the real
            // pipeline.
            noisy.extend_from_slice(&noise(600));
            corpora.push(("probe-corpus+noise-tail".to_string(), noisy));
        }
        for (name, path) in [
            ("test_data/text-1MB.txt", "test_data/text-1MB.txt"),
            ("test_data/alice.txt", "test_data/alice.txt"),
            (
                "benchmark_data/software.archive (2 MiB slice)",
                "benchmark_data/software.archive",
            ),
            (
                "benchmark_data/silesia.tar (2 MiB slice)",
                "benchmark_data/silesia.tar",
            ),
        ] {
            match std::fs::read(path) {
                Ok(bytes) => {
                    let take = bytes.len().min(2 * 1024 * 1024);
                    corpora.push((name.to_string(), bytes[..take].to_vec()));
                }
                Err(_) => println!("corpus unavailable, skipping: {path}"),
            }
        }

        for (name, data) in &corpora {
            // L9-T4 shape: production chunk splice, dict = first 32 KiB (the
            // exact shape `matrix_for` drives; corpora are all >= 32 KiB +
            // 4 KiB of body, so the split is total).
            let (dict, body) = data.split_at(DICT);
            // serial reference
            let (ser, ser_pad, ser_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            let ser_l11 = encode_l11t1(data, false);
            // parallel arms, and PROVE the pool ran them
            let writes_before = near_opt_flush_probe::parallel_flush_writes();
            let (par, par_pad, par_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            let par_l11 = encode_l11t1(data, true);
            let writes_delta = near_opt_flush_probe::parallel_flush_writes() - writes_before;
            assert_eq!(
                writes_delta, 2,
                "{name}: the pool must have engaged for BOTH shapes"
            );
            assert_eq!(ser_pad, par_pad, "{name}: L9-T4 trailing pad_bits diverged");
            assert_eq!(
                ser_align, par_align,
                "{name}: L9-T4 needs_alignment diverged (stored-block detection)"
            );
            if name == "probe-corpus+noise-tail" {
                // The reviewer's positivity fix (agent-29 minor 3): the noise
                // tail exists to price a STORED block into the final
                // fragment; assert the witness rather than trusting
                // equality with itself.
                assert!(
                    ser_align,
                    "{name}: the noise tail must still price a stored block \
                     (else the stored-replay coverage silently stops firing)"
                );
            }
            assert_eq!(
                ser, par,
                "{name}: L9-T4 chunk stream diverged from the serial reference"
            );
            assert_eq!(
                ser_l11, par_l11,
                "{name}: L11-T1 whole-buffer stream diverged"
            );
            assert!(
                !near_opt_flush_probe::stale_flag_fired(),
                "{name}: the stale-flag guard fired on a campaign corpus — \
                 the zero-flip finding no longer holds here"
            );
            println!("identity ok: {name} ({} B)", data.len());
        }
        // The interior-chunk shape (`is_last = false`): the caller appends the
        // sync-flush stored marker onto the SAME writer after `run` returns,
        // exercising the packed stream's trailing pending state that the
        // final-chunk arm above does not.
        {
            let data = &corpora[0].1;
            let (dict, body) = data.split_at(DICT);
            near_opt_flush_probe::set_force_serial_flush(true);
            let mut ser_out = Vec::with_capacity(body.len() / 2 + 1024);
            let ser_meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
                body,
                dict,
                9,
                false,
                &mut ser_out,
                true,
                200 * 1024 * 1024,
            );
            near_opt_flush_probe::set_force_serial_flush(false);
            let writes_before = near_opt_flush_probe::parallel_flush_writes();
            let mut par_out = Vec::with_capacity(body.len() / 2 + 1024);
            let par_meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
                body,
                dict,
                9,
                false,
                &mut par_out,
                true,
                200 * 1024 * 1024,
            );
            assert_eq!(
                near_opt_flush_probe::parallel_flush_writes() - writes_before,
                1,
                "the pool must have engaged for the interior-chunk shape"
            );
            assert_eq!(
                ser_out, par_out,
                "interior chunk (sync-flush seam follows) diverged from serial"
            );
            assert_eq!(
                (ser_meta.pad_bits, ser_meta.needs_alignment),
                (par_meta.pad_bits, par_meta.needs_alignment),
                "interior chunk ChunkMeta diverged"
            );
        }
        // The guard's fallback half: latch the stale flag and prove the next
        // chunk's bytes are the serial ones even with the pool addressable.
        near_opt_flush_probe::set_stale_flag_for_tests(true);
        {
            let data = &corpora[0].1;
            let (dict, body) = data.split_at(DICT);
            let (ser, ..) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            let (par, ..) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            assert_eq!(ser, par, "vetoed runs must produce the serial bytes");
        }
        near_opt_flush_probe::set_stale_flag_for_tests(false);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// LEVER #2 / d3 (feature `near-opt-d3-probe`, DEFAULT OFF): the cheap
// ordered-candidate fill probe. Without the feature this module compiles to
// nothing and the matrix above + the flush block are the whole file.
//
// The probes ride the EXACT production entries (the `matrix_for` shape):
//   * the L9/T4 chunk shape (`encode_deflate_splice_chunk_to_sink`,
//     parallel=true, chunk + 32 KiB dict) through BOTH arms of one binary —
//     the production bt fill (`force-bt` arm, the serial reference) vs the
//     d3 chain gather (the feature-on default);
//   * the L9-T4 whole-FILE path (`compress_with_threads(data, 9, 4)`) —
//     the parallel route end to end, gzip-framed so `gzippy::decompress`
//     can roundtrip it (the chunk-level splice stream is dict-anchored and
//     has no standalone decoder).
//
// Output bytes DIFFER between the arms by design (cheaper candidate set, not
// an identical one): each probe reports exact deltas instead of asserting
// identity, and pins roundtrip validity + run-to-run determinism for both.
// ─────────────────────────────────────────────────────────────────────────
#[cfg(feature = "near-opt-d3-probe")]
mod d3_probes {
    use super::*;
    use gzippy::compress::deflate::matchfinder::near_opt_probe::ChainGather;
    use gzippy::compress::deflate::parse::near_opt_d3_probe;

    /// One production-chunk encode at the `matrix_for` shape (L9, T>1
    /// splice, chunk + 32 KiB dict, `input_total_len` 200 MiB — the exact
    /// entry the matrix drives), through the requested fill arm.
    fn encode_arm(body: &[u8], dict: &[u8], force_bt: bool) -> Vec<u8> {
        near_opt_d3_probe::set_force_bt(force_bt);
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let _meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            9,
            true,
            &mut out,
            true,
            200 * 1024 * 1024,
        );
        near_opt_d3_probe::set_force_bt(false);
        out
    }

    /// One full-file L9-T4 encode (the parallel route end to end), through
    /// the requested fill arm.
    fn encode_full_arm(data: &[u8], force_bt: bool) -> Vec<u8> {
        near_opt_d3_probe::set_force_bt(force_bt);
        let stream = gzippy::compress_with_threads(data, 9, 4).expect("encode should succeed");
        near_opt_d3_probe::set_force_bt(false);
        stream
    }

    /// Bitwise CRC-32 (IEEE) — enough to frame a dict-free chunk stream as
    /// a single-member gzip for the standard decoder (the chunk-level
    /// splice stream itself is dict-anchored and has no standalone decoder).
    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            table[i as usize] = c;
        }
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
        }
        !crc
    }

    /// Variant 10 of the matrix, `matrix_for`'s shape: the production bt
    /// fill vs the d3 chain gather at the L9-T4 chunk shape, best-of-5
    /// each after two warmups, plus the exact deltas. Rows print in the
    /// matrix's format so they paste alongside variants 1-6.
    #[test]
    #[ignore]
    fn d3_probe_wall_and_bytes_l9t4() {
        let data = build_corpus();
        let body = &data[DICT..DICT + CHUNK];
        let dict = &data[..DICT];

        // Two warmups per arm (the shape `run` uses), then best-of-5.
        let _ = encode_arm(body, dict, true);
        let _ = encode_arm(body, dict, false);
        let mut bt_times = Vec::with_capacity(5);
        let mut d3_times = Vec::with_capacity(5);
        let (mut bt_bytes, mut d3_bytes) = (0usize, 0usize);
        for _ in 0..5 {
            let t = Instant::now();
            let out = encode_arm(body, dict, true);
            bt_times.push(t.elapsed().as_secs_f64());
            bt_bytes = out.len();
        }
        for _ in 0..5 {
            let t = Instant::now();
            let out = encode_arm(body, dict, false);
            d3_times.push(t.elapsed().as_secs_f64());
            d3_bytes = out.len();
        }
        bt_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d3_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (bt_best, d3_best) = (bt_times[0], d3_times[0]);

        let delta = d3_bytes as i64 - bt_bytes as i64;
        let pct = delta as f64 / bt_bytes as f64 * 100.0;
        println!(
            "10 d3 serial ref (bt fill, depth400, passes2) : best {bt_best:.4}s bytes {bt_bytes}"
        );
        println!(
            "10 d3 chain gather  (same shape)              : best {d3_best:.4}s bytes {d3_bytes} (wall ratio {:.3}x, bytes {:+} B, {:+.2}%)",
            d3_best / bt_best,
            delta,
            pct
        );

        // Both arms must be deterministic run-to-run, and their
        // dict-free streams must decode (roundtrip validity). Byte
        // identity BETWEEN the arms is not asserted — the candidate sets
        // differ by design, and the exact deltas are the receipt above.
        for (label, force_bt) in [("bt-fill", true), ("d3-chain", false)] {
            let once = encode_arm(body, &[], force_bt);
            let again = encode_arm(body, &[], force_bt);
            assert_eq!(once, again, "{label}: the fill arm must be deterministic");
            // Wrap the dict-free chunk stream as a single-member gzip for
            // the standard decoder. CRC32 is over the UNCOMPRESSED body,
            // ISIZE is the body's length mod 2^32.
            let mut framed = Vec::with_capacity(once.len() + 28);
            framed.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 0x03]);
            framed.extend_from_slice(&once);
            framed.extend_from_slice(&crc32(body).to_le_bytes());
            framed.extend_from_slice(&(body.len() as u32).to_le_bytes());
            assert!(
                matches!(
                    gzippy::decompress(&framed),
                    Ok(back) if back == body
                ),
                "{label}: the dict-free stream must decode to the body"
            );
        }
    }

    /// The shape-confirmation run (mission deliverable 3's "one run at the
    /// L9-T4 shape with the parallel path"): ONE whole-file L9-T4 encode on
    /// each arm, then (a) both streams roundtrip byte-exactly via
    /// `gzippy::decompress`, (b) the exact delta between them is printed
    /// (not asserted equal — the candidate sets differ by design), and (c)
    /// the chain gather's OWN output shape holds at the API level over a
    /// 80k-position sweep of the probe corpus: strictly-increasing lengths,
    /// non-decreasing offsets — the DP's trusted layout, carried through.
    #[test]
    fn d3_probe_shape_lands_at_l9t4_production() {
        let data = build_corpus();
        let body = data[..DICT + CHUNK + 4096].to_vec();

        let bt_stream = encode_full_arm(&body, true);
        let d3_stream = encode_full_arm(&body, false);
        assert!(
            matches!(
                gzippy::decompress(&d3_stream),
                Ok(back) if back == body
            ),
            "the d3 chain gather must produce a valid L9-T4 stream"
        );
        assert!(
            matches!(
                gzippy::decompress(&bt_stream),
                Ok(back) if back == body
            ),
            "the force-bt arm must reproduce the production stream"
        );
        let delta = d3_stream.len() as i64 - bt_stream.len() as i64;
        println!(
            "11 d3 L9-T4 whole-file: force-bt {} B, d3 chain {} B (delta {:+} B, {:+.2}%)",
            bt_stream.len(),
            d3_stream.len(),
            delta,
            delta as f64 / bt_stream.len() as f64 * 100.0
        );

        // The layout contract at the API level (the probe's own
        // side-by-side evidence that the trusted shape carries through).
        let mut padded = body.clone();
        padded.resize(body.len() + 16, 0);
        let mut mf = ChainGather::new();
        let mut next_hashes = [0u32; 2];
        let mut lists = 0usize;
        for pos in 0..80_000 {
            let remaining = padded.len() - pos;
            let max_len = 258u32.min(remaining as u32);
            if max_len < 5 {
                break;
            }
            let mut out = vec![Default::default(); 260];
            let n = mf.get_matches(
                &padded,
                0,
                pos as isize,
                max_len,
                max_len.min(150),
                100,
                #[cfg(feature = "near-opt-bt-probebudget")]
                100, // the chain arm ignores it; signature-identical
                &mut next_hashes,
                &mut out,
            );
            let mut prev_len = 0u16;
            let mut prev_off = 0u16;
            for m in &out[..n] {
                assert!(
                    m.length > prev_len,
                    "pos {pos}: lengths must strictly increase"
                );
                if prev_off != 0 {
                    assert!(m.offset >= prev_off, "pos {pos}: offsets must not decrease");
                }
                prev_len = m.length;
                prev_off = m.offset;
            }
            lists += usize::from(n > 0);
        }
        assert!(
            lists > 0,
            "the chain gather must find matches on the probe corpus"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// LEVER #4 / bt (feature `near-opt-bt-probebudget`, DEFAULT OFF): the
// per-descent probe-budget arms on the LAST un-priced loss class
// (pigz:silesia.tar:L9:T4:wall 1.0840). Without the feature this module
// compiles to nothing.
//
// Same dual-shape drive as levers #2/#3 above:
//   * the L9/T4 chunk shape (`matrix_for`'s exact production entry):
//     the inert production arm (budget >= the fill's depth) vs budget
//     {24, 48, 96, 150}, best-of-5 walls + exact byte deltas;
//   * the L9-T4 whole-FILE path (`compress_with_threads(data, 9, 4)`) per
//     arm, each roundtrip-verified via `gzippy::decompress` — the arms'
//     SHAPES may differ by design (a budget cut shrinks the candidate
//     list; it never reorders it), so the deltas are REPORTED, not asserted
//     equal, while the roundtrip itself is REQUIRED of every arm.
// ─────────────────────────────────────────────────────────────────────────
#[cfg(feature = "near-opt-bt-probebudget")]
mod bt_probebudget_probes {
    use super::*;
    use gzippy::compress::deflate::matchfinder::bt::BtMatchfinder;
    use gzippy::compress::deflate::parse::near_opt_probebudget;

    /// The production L9-T4 fill depth (`params_parallel(9)` ≡
    /// `params_parallel(11)` at depth 400 — the matrix's variant 1). A
    /// budget at this value is the inert production arm.
    const DEPTH: u32 = 400;

    /// One production-chunk encode at the `matrix_for` shape (L9, T>1
    /// splice, chunk + 32 KiB dict, `input_total_len` 200 MiB) through the
    /// requested budget arm.
    fn encode_arm(body: &[u8], dict: &[u8], budget: u32) -> Vec<u8> {
        near_opt_probebudget::set_budget(budget);
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let _meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            9,
            true,
            &mut out,
            true,
            200 * 1024 * 1024,
        );
        out
    }

    /// One full-file L9-T4 encode (the parallel route end to end) through
    /// the requested budget arm, resting the knob back on the inert value.
    fn encode_full_arm(data: &[u8], budget: u32) -> Vec<u8> {
        near_opt_probebudget::set_budget(budget);
        let stream = gzippy::compress_with_threads(data, 9, 4).expect("encode should succeed");
        near_opt_probebudget::set_budget(DEPTH);
        stream
    }

    /// The ARMS constant in arm order: the inert production reference first
    /// (every wall ratio and byte delta reads against it).
    const ARMS: [(&str, u32); 5] = [
        ("full-depth bt (inert budget)", DEPTH),
        ("ABIT 24", 24),
        ("ABIT 48", 48),
        ("ABIT 96", 96),
        ("ABIT 150", 150),
    ];

    /// Variant 12 of the matrix, `matrix_for`'s shape: bt fill at each
    /// budget, best-of-5 after two warmups per arm (the shape the d3 probe
    /// and `run` use). Rows print in the matrix's format so they paste
    /// alongside variants 1-11.
    #[test]
    #[ignore]
    fn bt_probebudget_wall_and_bytes_l9t4() {
        let data = build_corpus();
        let body = &data[DICT..DICT + CHUNK];
        let dict = &data[..DICT];

        let mut rows: Vec<(&str, f64, usize)> = Vec::with_capacity(ARMS.len());
        for (label, budget) in ARMS {
            let _ = encode_arm(body, dict, budget);
            let _ = encode_arm(body, dict, budget);
            let mut times = Vec::with_capacity(5);
            let mut bytes = 0usize;
            for _ in 0..5 {
                let t = Instant::now();
                let out = encode_arm(body, dict, budget);
                times.push(t.elapsed().as_secs_f64());
                bytes = out.len();
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rows.push((label, times[0], bytes));
        }
        near_opt_probebudget::set_budget(DEPTH);

        let (_, ref_wall, ref_bytes) = rows[0];
        println!("-- bt-probebudget L9/T4 chunk (1.8 MB matrix corpus, depth 400, passes 2) --");
        for (label, wall, bytes) in &rows {
            let db = *bytes as i64 - ref_bytes as i64;
            println!(
                "12 bt-probebudget {label:<28}: best {wall:.4}s bytes {bytes} \
                 (wall {:.3}x vs inert, bytes {db:+} B, {:+.2}%)",
                wall / ref_wall,
                db as f64 / ref_bytes as f64 * 100.0
            );
        }
    }

    /// The shape-confirmation run at the L9-T4 production route: per-arm
    /// WHOLE-FILE encodes on the probe corpus plus a 2 MiB silesia.tar
    /// slice, every arm roundtrip-verified (OBSERVABLE roundtrip — the
    /// byte-exact requirement is the decode-matches-input invariant, not
    /// arm-to-arm byte identity, because the arms' candidate sets differ by
    /// design), exact deltas printed, both arms deterministic run-to-run,
    /// THEN the layout receipt at the matchfinder API: per-pair fresh-state
    /// positions, a budget descent's recorded list is a true PREFIX of the
    /// unrestricted walk's list (the trusted strictly-increasing-length /
    /// non-decreasing-offset order the DP reads is only ever CUT, never
    /// reshaped), and every budget candidate is a real back-reference —
    /// so a byte delta is the DP's own choice among well-formed inputs.
    #[test]
    fn bt_probebudget_shape_lands_at_l9t4_production() {
        let mut corpora: Vec<(String, Vec<u8>)> = Vec::new();
        {
            let data = build_corpus();
            corpora.push((
                "probe-corpus".to_string(),
                data[..DICT + CHUNK + 4096].to_vec(),
            ));
        }
        if let Ok(bytes) = std::fs::read("benchmark_data/silesia.tar") {
            let take = bytes.len().min(2 * 1024 * 1024);
            corpora.push((
                "benchmark_data/silesia.tar (2 MiB slice)".to_string(),
                bytes[..take].to_vec(),
            ));
        }

        for (name, data) in &corpora {
            let mut streams = Vec::with_capacity(ARMS.len());
            for (label, budget) in ARMS {
                streams.push((label, encode_full_arm(data, budget)));
            }
            for (label, stream) in &streams {
                assert!(
                    matches!(
                        gzippy::decompress(stream),
                        Ok(back) if back == *data
                    ),
                    "{name}: the {label} arm must roundtrip byte-exactly"
                );
            }
            let ref_len = streams[0].1.len() as i64;
            for (label, stream) in &streams {
                let d = stream.len() as i64 - ref_len;
                println!(
                    "13 bt-probebudget {name:<40} {label:<28}: {} B (delta {d:+} B, {:+.2}%)",
                    stream.len(),
                    d as f64 / ref_len as f64 * 100.0
                );
            }
            // Determinism per arm: re-encode and require the same bytes.
            for idx in 1..streams.len() {
                let (label, budget) = ARMS[idx];
                let again = encode_full_arm(data, budget);
                assert_eq!(
                    streams[idx].1, again,
                    "{name}: the {label} arm must be deterministic"
                );
            }
        }

        // The layout receipt: per pair, FRESH matchfinders on the same
        // corpus prefix share the walk's state evolution to the compared
        // position, so the budget cut is directly observable as a prefix.
        // Windows of 4096 positions: inside one window the first position
        // (identical state) gives the exact-prefix verdict; every position
        // pins the trusted layout and back-reference validity.
        let data = build_corpus();
        let mut padded = data[..128 * 1024].to_vec();
        padded.resize(padded.len() + 16, 0);
        let window = 4096usize;
        let mut budgets_hit = false;
        let mut lists = 0usize;
        for start in (0..padded.len() - window).step_by(window) {
            let mut mf_full = BtMatchfinder::new();
            let mut mf_b = BtMatchfinder::new();
            let mut mf_w = BtMatchfinder::new();
            let mut hashes_full = [0u32; 2];
            let mut hashes_b = [0u32; 2];
            let mut hashes_w = [0u32; 2];
            for pos in start..start + window {
                let remaining = padded.len() - pos;
                let max_len = 258u32.min(remaining as u32);
                if max_len < 5 {
                    break;
                }
                let mut out_full = vec![Default::default(); 260];
                let mut out_b = vec![Default::default(); 260];
                let mut out_w = vec![Default::default(); 260];
                // Signature: (buf, in_base, cur_pos, max_len, nice_len,
                // max_depth, probe_budget, next_hashes, out) — the full arm
                // budgets at DEPTH (inert), the b arm at the lever's
                // ABIT 24. The WITNESS arm at budget 2 exists because a
                // 24-cap cut usually lands AFTER a descent's last recorded
                // candidate (same recorded list, fewer probes spent) — the
                // witness arm's cuts observably shrink SOME recorded list,
                // proving the cut path keeps the LAYOUT (exact prefix of
                // the full walk's), which the 24 arm spells out per window
                // anyway.
                let nf = mf_full.get_matches(
                    &padded,
                    0,
                    pos as isize,
                    max_len,
                    max_len.min(150),
                    DEPTH,
                    DEPTH,
                    &mut hashes_full,
                    &mut out_full,
                );
                let nb = mf_b.get_matches(
                    &padded,
                    0,
                    pos as isize,
                    max_len,
                    max_len.min(150),
                    DEPTH,
                    24,
                    &mut hashes_b,
                    &mut out_b,
                );
                let nw = mf_w.get_matches(
                    &padded,
                    0,
                    pos as isize,
                    max_len,
                    max_len.min(150),
                    DEPTH,
                    2,
                    &mut hashes_w,
                    &mut out_w,
                );
                let mut prev_len = 0u16;
                let mut prev_off = 0u16;
                for m in &out_b[..nb] {
                    assert!(
                        m.length > prev_len,
                        "pos {pos}: lengths must strictly increase (budget arm)"
                    );
                    if prev_off != 0 {
                        assert!(
                            m.offset >= prev_off,
                            "pos {pos}: offsets must not decrease (budget arm)"
                        );
                    }
                    // The candidate must be a REAL back-reference (the bytes
                    // it claims to copy must equal the source bytes).
                    for i in 0..m.length as usize {
                        assert_eq!(
                            padded[pos + i],
                            padded[pos - m.offset as usize + i],
                            "pos {pos}: budget candidate mismatches at byte {i}"
                        );
                    }
                    prev_len = m.length;
                    prev_off = m.offset;
                }
                if pos == start {
                    // Same-state prefix receipt: the budget walk makes the
                    // SAME visit sequence and stops earlier — the ABIT 24
                    // list and the witness list must each be a prefix of
                    // the unrestricted walk's list (the trusted order is
                    // only ever cut, never reshaped).
                    assert!(
                        nb <= nf && out_b[..nb] == out_full[..nb],
                        "pos {pos}: the budget list must be a prefix of the \
                         unrestricted walk's list (nb {nb}, nf {nf})"
                    );
                    assert!(
                        nw <= nf && out_w[..nw] == out_full[..nw],
                        "pos {pos}: the witness list must be a prefix of the \
                         unrestricted walk's list (nw {nw}, nf {nf})"
                    );
                    // Same walk-order prefix: every witness candidate is a
                    // prefix-member so byte deltas are the DP's own choice.
                    assert!(
                        nw <= nb && out_w[..nw] == out_b[..nw],
                        "pos {pos}: the witness list must be a prefix of the \
                         24-arm's list"
                    );
                }
                if nw < nb || nw < nf {
                    budgets_hit = true;
                }
                lists += usize::from(nf > 0);
            }
        }
        assert!(
            lists > 0,
            "the fill must find matches on the probe corpus somewhere"
        );
        assert!(
            budgets_hit,
            "the probe corpus must exercise budget cuts somewhere \
             (24 vs 400 differ) — else the prefix receipt never engaged"
        );
        near_opt_probebudget::set_budget(DEPTH);
    }

    /// The whole-NAMED-LOSS run: every budget arm over the ENTIRE
    /// `benchmark_data/silesia.tar` (202 MiB — the actual corpus of
    /// pigz:silesia.tar:L9:T4:wall 1.0840) through the L9-T4 parallel route
    /// (`compress_with_threads(data, 9, 4)`), best-of-5 per arm after one
    /// warmup (the `run` helper's shape), the FIRST stream of each arm
    /// roundtrip-verified through `gzippy::decompress`, exact byte deltas
    /// reported. This is the cell the chunk-level matrix above is the
    /// cheap proxy for; wall spread on this box made best-of-2 unstable.
    #[test]
    #[ignore]
    fn bt_probebudget_full_silesia_file_l9t4() {
        let data = std::fs::read("benchmark_data/silesia.tar")
            .expect("benchmark_data/silesia.tar must exist for this probe");
        let ref_stream = encode_full_arm(&data, DEPTH);
        let mut rows: Vec<(&str, f64, usize)> = Vec::new();
        for (label, budget) in ARMS {
            let mut times = Vec::new();
            let mut bytes = 0usize;
            for run in 0..5 {
                let t = Instant::now();
                let stream = encode_full_arm(&data, budget);
                times.push(t.elapsed().as_secs_f64());
                bytes = stream.len();
                if run == 0 {
                    assert!(
                        matches!(
                            gzippy::decompress(&stream),
                            Ok(back) if back == data
                        ),
                        "{label}: the full-file arm must roundtrip byte-exactly"
                    );
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rows.push((label, times[0], bytes));
        }

        let (_, ref_wall, ref_bytes) = rows[0];
        println!(
            "-- bt-probebudget FULL silesia.tar ({} B), L9-T4 whole-file --",
            data.len()
        );
        for (label, wall, bytes) in &rows {
            let d = *bytes as i64 - ref_bytes as i64;
            println!(
                "14 bt-probebudget {label:<28}: best {wall:.3}s bytes {bytes} \
                 (wall {:.3}x vs inert, bytes {d:+} B, {:+.3}%)",
                wall / ref_wall,
                d as f64 / ref_bytes as f64 * 100.0
            );
        }
        assert_eq!(
            ref_stream.len(),
            ref_bytes,
            "the inert arm stream is the byte reference"
        );
    }
}
