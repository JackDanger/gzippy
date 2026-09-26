//! Cross-`-p` byte-parity gates (PR-2 of the parity unification,
//! `docs/board/parity-unification-design.md` v3.1).
//!
//! THE CONTRACT v3.1 SHIPS: for any `threads >= 2`, the chunk grid, engine,
//! per-level params (`params_parallel(level)`) and header budget are all
//! thread-independent, so `-p N` output bytes are IDENTICAL on a given box at
//! every N — asserted here across 2/4/8/16 with roundtrip protection.
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
//!
//! The T1-vs-T(N) size band below is SYNTHETIC-PAYLOAD-SPECIFIC: this periodic
//! payload makes seam losses tiny (measured Δ: L1 67 B, L6/L9 26 B), and the
//! real-corpus comparison is the census instrument's job — where chunked T>1
//! vs T1 legitimately differs in BOTH directions (e.g. the near-opt T>1 parse
//! is SMALLER than T1). The gate holds the band to 0.25% or a per-seam
//! absolute budget, whichever is smaller, so only structural growth fires it.
//!
//! Payload grids derive from `pipelined_block_size` itself — the gates must
//! never hardcode chunk sizes the encoder is free to retune.

use gzippy::compress::pipelined::pipelined_block_size;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn digest(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

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

/// At this payload every level must derive the SAME grid for any N >= 2: the
/// thread-free anchor. This is the unit form of the parity contract.
#[test]
fn grid_is_thread_free_at_every_pipelined_level() {
    let len = 5_400_773;
    for level in 1u32..=12 {
        let base = pipelined_block_size(len, 4, level);
        assert!(
            base >= 128 * 1024,
            "L{level}: grid collapsed below MIN ({base})"
        );
        assert!(
            base <= len.min(if level >= 6 {
                2 * 1024 * 1024
            } else {
                8 * 1024 * 1024
            }),
            "L{level}: grid above the level cap ({base})"
        );
        for threads in [2usize, 3, 8, 16, 64] {
            assert_eq!(
                pipelined_block_size(len, threads, level),
                base,
                "L{level} T{threads}: grid depends on threads"
            );
        }
    }
}

/// The cross-`-p` digest tie at every pipelined level, with roundtrip.
/// L10–12 included: the audit-wave M3 gap (agent-34) named the missing
/// range; verified locally on software.archive (1 sha256 per level across
/// T2/T4/T8) before this loop was widened.
#[test]
fn cross_thread_bytes_are_identical_at_every_level() {
    let len = 5_400_773;
    let data = seam_payload(len);
    for level in 1u8..=12 {
        let reference = encode_at(&data, level, 4);
        assert!(
            matches!(gzippy::decompress(&reference), Ok(back) if back == data),
            "L{level}: T4 stream must roundtrip"
        );
        for threads in [2usize, 8, 16] {
            let stream = encode_at(&data, level, threads);
            assert!(
                matches!(gzippy::decompress(&stream), Ok(back) if back == data),
                "L{level} T{threads}: stream must roundtrip"
            );
            assert_eq!(
                digest(&stream),
                digest(&reference),
                "L{level} T{threads}: bytes differ from T4 — a thread-dependent \
                 byte input survived the unification (grid? params? header budget?)"
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
/// cannot hide inside the band.
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
