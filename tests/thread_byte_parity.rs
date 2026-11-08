//! Per-`-p`-N byte-parity gates (grid-by-threads renegotiation of the parity
//! unification, `docs/board/parity-unification-design.md` v3.1 → charter
//! amendment v2).
//!
//! RENEGOTIATION (2026-11-08, charter amendment v2, cross-`-p` bullet): the
//! CONTRACT v3.1 "every `-p N`, N >= 2 emits identical bytes" cross-N byte
//! equality was renegotiated to buy wall: pigz default strips `p` workers on
//! `p` cores and beat the thread-free grid's best `-p` by 12-38% wall at
//! L1-L4 (12% at L5), because the v3.1 anchor (`pipelined_block_size`,
//! `GRID_REF_THREADS`) capped every run at `4 × cpt ≈ 8` chunks — `-p16`
//! bought idle pipes, not parallelism. The grid-by-threads lever reinstates
//! `target_chunks = max(REF, N) × cpt`, so for N > 4 a larger `-p` now splits
//! the input into (weakly) more chunks and MAY emit different bytes than
//! `-p4`. What replaces cross-N equality as the shipped contract:
//!
//!   * same `-p N`, same input, twice ⇒ byte-identical (per-N determinism);
//!   * `-p N` with N <= 4 ⇒ byte-identical ACROSS N, bit-for-bit as before:
//!     the preserved band — the board's dominant T>1 class (T <= 4) and the
//!     T4 fingerprint rows do not move;
//!   * different N MAY differ (asserted nowhere: bytes per N are the size/
//!     wall census instrument's business, 0.1% tie band).
//!
//! T1 is deliberately a distinct stream class: the whole-buffer parse lets
//! matches straddle what would be chunk seams, so its bytes cannot bit-match
//! any chunked stream (measured on the real corpus —
//! `records/2026-09-22-ec2-c7a4xl/FINAL-ADJUDICATION.md:48; repro: T1 vs T4 on
//! this payload). The T1 leg is therefore a size-tie, not a digest tie.
//!
//! Inputs are multi-MB at every level, and the library entry used here
//! (`compress_with_threads` → `compress::compress_bytes`) passes the "big
//! file" sentinel, so the CLI's `optimal_thread_count` halving (a real-file
//! -size rule, `compress/optimization.rs`) never fires through this API — the
//! small-input T-ROUTING escape is CLI-only and does not weaken these gates.
//! (That filer also caps `-p` at the core count; N above the core count
//! collapses deterministically to the core count.)
//!
//! Payload grids derive from `pipelined_block_size` itself — the gates must
//! never hardcode chunk sizes the encoder is free to retune.
//! Charter v2 (2026-11-07, PR #447): this cross--p contract is pre-authorized for in-lever renegotiation — it stays law until a lever deliberately renegotiates it.

use gzippy::compress::pipelined::{pipelined_block_size, GRID_REF_THREADS};

fn encode_at(data: &[u8], level: u8, threads: usize) -> Vec<u8> {
    gzippy::compress_with_threads(data, level, threads).expect("encode should succeed")
}

/// Well past the 102,400-byte routing-escape threshold at every level, and
/// past the L9 grid (1.8 MiB) so T=4 splits into multiple chunks with real
/// seams; periodic + mixed content like the board's text class.
fn seam_payload(len: usize) -> Vec<u8> {
    let unit: &[u8] = b"abracadabra-sing-a-song-of-sixpence-pockets-full-of-rye";
    let mut v = Vec::with_capacity(len + unit.len());
    while v.len() < len {
        v.extend_from_slice(unit);
        v.extend((0u8..=255u8).rev());
        v.extend_from_slice(unit);
    }
    v.truncate(len);
    v
}

/// Unit form of the renegotiated grid contract, every level 1..=12:
///   * N <= GRID_REF_THREADS ⇒ the T4 reference grid bit-for-bit;
///   * N >  GRID_REF_THREADS ⇒ chunk size never GROWS and the chunk count is
///     (weakly) never SMALLER than the T4 reference — scaling up may never
///     merge chunks, the whole lever is "-p > 4 buys chunks";
///   * the MIN/level-cap bounds hold on every arm;
///   * pure function of (len, N, level) — no host, no env, no clock.
#[test]
fn grid_is_reference_anchored_below_4_and_scales_above() {
    let len: usize = 5_400_773;
    let level_cap = |level: u32| {
        if level >= 6 {
            2 * 1024 * 1024
        } else {
            8 * 1024 * 1024
        }
    };
    for level in 1u32..=12 {
        let chunks_of = |n: usize| len.div_ceil(pipelined_block_size(len, n, level));
        let base = pipelined_block_size(len, 4, level);
        assert!(
            chunks_of(4) > 1,
            "L{level}: T4 must split into multiple chunks"
        );
        assert!(
            base >= 128 * 1024,
            "L{level}: grid collapsed below MIN ({base})"
        );
        assert!(
            base <= len.min(level_cap(level)),
            "L{level}: grid above the level cap ({base})"
        );
        for threads in [1usize, 2, 3, 4] {
            assert_eq!(
                pipelined_block_size(len, threads, level),
                base,
                "L{level} T{threads}: grid moved inside the preserved band \
                 (N <= {GRID_REF_THREADS} must be the T4 layout bit-for-bit)"
            );
        }
        for threads in [8usize, 16, 64] {
            let scaled = pipelined_block_size(len, threads, level);
            assert!(
                scaled >= 128 * 1024,
                "L{level} T{threads}: scaled grid collapsed below MIN ({scaled})"
            );
            assert!(
                scaled <= len.min(level_cap(level)),
                "L{level} T{threads}: scaled grid above the level cap ({scaled})"
            );
            assert!(
                scaled <= base,
                "L{level} T{threads}: scaled chunk ({scaled}) is LARGER than the \
                 T4 reference ({base}) — scaling up never grows a chunk"
            );
            assert!(
                chunks_of(threads) >= chunks_of(4),
                "L{level} T{threads}: scaled grid has FEWER chunks ({}) than the \
                 T4 reference ({}) — -p > 4 must buy chunks, not merge them",
                chunks_of(threads),
                chunks_of(4)
            );
            // Purity: the same request derives the identical grid twice.
            assert_eq!(
                pipelined_block_size(len, threads, level),
                scaled,
                "L{level} T{threads}: grid is not a pure function of its args"
            );
        }
    }
}

/// THE PRESERVED BAND: `-p` 2/3/4 remain byte-identical ACROSS `-p` at every
/// level (this is what keeps the T4 fingerprint rows and the board's dominant
/// cells put), with roundtrip protection at T4.
#[test]
fn preserved_band_n_le_4_bytes_are_across_p_identical() {
    let len = 5_400_773;
    let data = seam_payload(len);
    for level in 1u8..=12 {
        let reference = encode_at(&data, level, 4);
        assert!(
            matches!(gzippy::decompress(&reference), Ok(back) if back == data),
            "L{level}: T4 stream must roundtrip"
        );
        for threads in [2usize, 3] {
            let stream = encode_at(&data, level, threads);
            assert_eq!(
                stream, reference,
                "L{level} T{threads}: bytes differ from T4 inside the preserved \
                 band — a thread-dependent byte input moved (grid? params? header \
                 budget?)"
            );
        }
    }
}

/// THE RENEGOTIATED ARM: for `-p` in {4, 8, 16} at the representative levels
/// (L1 = cpt 1, L6 = the 2 MiB cap arm, L9 = standard cpt, L12 = top), output
/// bytes are DETERMINISTIC PER N — same N twice ⇒ identical bytes, every run
/// roundtrip-protected through our decoder. A `-p8`/`-p16` bytes-vs-`-p4`
/// DIFFERENCE is expected and correct under the renegotiation (the grid
/// scales above 4), so no cross-N digest assertion lives here.
#[test]
fn per_n_determinism_and_validity_with_renegotiated_grid() {
    let len = 5_400_773;
    let data = seam_payload(len);
    for level in [1u8, 6, 9, 12] {
        let t4 = encode_at(&data, level, 4);
        assert!(
            matches!(gzippy::decompress(&t4), Ok(back) if back == data),
            "L{level}: T4 stream must roundtrip"
        );
        for threads in [8usize, 16] {
            let a = encode_at(&data, level, threads);
            let b = encode_at(&data, level, threads);
            assert_eq!(
                a, b,
                "L{level} T{threads}: run-to-run output changed — per-N \
                 determinism is non-negotiable after the renegotiation"
            );
            assert!(
                matches!(gzippy::decompress(&a), Ok(back) if back == data),
                "L{level} T{threads}: stream must roundtrip"
            );
        }
    }
}

/// T1 is a distinct stream class (whole-buffer): its BYTES never match a
/// chunked stream, but its size must stay inside the synthetic-payload tie
/// (0.25% cap or 128 B per chunk seam, whichever is smaller) and it must
/// roundtrip. This keeps the T1-distinctness contract pinned without
/// pretending a digest tie is retrievable, and far tighter than the 0.5%
/// first cut so a real seam-tax regression (the class substituting here)
/// cannot hide inside the band. The seam count derives from the UNCHANGED
/// T4 reference grid, so it does not move with the renegotiation.
#[test]
fn t1_stream_is_a_distinct_class_within_the_size_tie() {
    let len = 5_400_773;
    let data = seam_payload(len);
    for level in [1u8, 6, 9] {
        let t1 = encode_at(&data, level, 1);
        assert!(
            matches!(gzippy::decompress(&t1), Ok(back) if back == data),
            "L{level}: T1 stream must roundtrip"
        );
        let t4 = encode_at(&data, level, 4);
        let seams =
            (len / gzippy::compress::pipelined::pipelined_block_size(len, 4, level as u32)).max(1);
        let slack = (len as f64 * 0.0025).min((seams * 128) as f64).max(1024.0);
        let delta = (t1.len() as i64 - t4.len() as i64).abs();
        assert!(
            (delta as f64) <= slack,
            "L{level}: T1 ({}) vs T4 ({}) sizes differ beyond the tie ({slack:.0} B)",
            t1.len(),
            t4.len()
        );
    }
}
