//! Size INVARIANTS over the synthetic fixtures — not values.
//!
//! Project rule (CLAUDE.md non-negotiable 5): test the INVARIANT, not the
//! VALUE. An equality assertion beats a sentence in a doc because only one of
//! them fails closed. These tests pin two structural claims about the encoder
//! that must hold at every commit, in-process, in seconds:
//!
//! 1. **Ladder monotonicity** (`ladder_is_monotone_t1`): asking for more
//!    compression must never produce a LARGER file. For every fixture and
//!    every adjacent level pair N -> N+1, size(N+1) <= size(N). The known
//!    real-corpus defect ("the level ladder sags at L4" — `-4` larger than
//!    `-3` on 10/11 TUNE files) is encoded in [`KNOWN_SAGS`], which fails
//!    closed in BOTH directions: an unlisted sag fails, and a listed sag that
//!    HEALS also fails so the list can only shrink.
//!
//! 2. **Incompressible expansion bound** (`noise_expansion_bounded_t1` /
//!    `_t4`): on 1 MiB of incompressible noise the output must stay within a
//!    measured slack of the OPTIMAL stored framing —
//!    `ceil(n/65535) * 5 + 18` bytes (gzip header 10 + trailer 8, 5 bytes of
//!    stored-block framing per maximal 65535-byte block). The slack consts
//!    ratchet DOWN, never up.
//!
//! Every output that is sized is also ROUNDTRIPPED through an independent
//! decoder (flate2/zlib-ng) — a corrupt-but-small stream must never pass a
//! size test.

use gzippy::fixtures;
use std::io::Read;

const LEVELS: std::ops::RangeInclusive<u32> = 0..=9;

/// (fixture, N) pairs where size(N+1) > size(N) is the CURRENT, MEASURED
/// state of the encoder — the "level ladder sags" defect family
/// (memory: project_the_level_ladder_sags_at_L4.md; on the real corpus `-4`
/// is larger than `-3` on 10/11 TUNE files). Measured on the synthetic
/// fixtures at the commit that introduced this test (T1, in-process).
/// A pair listed here that STOPS sagging fails the test with instructions to
/// delete it, so this list is honest and can only shrink.
const KNOWN_SAGS: &[(&str, u32)] = &[
    // The L3->L4 sag, the synthetic twin of the real-corpus L4 defect:
    ("tabular", 3), // L3 255735 -> L4 271505 (+15770 B)
    ("binary", 3),  // L3 661353 -> L4 663583 (+2230 B)
    // High-level sag on prose: L7 is the best level on `text`.
    ("text", 7), // L7 305775 -> L8 306342 (+567 B)
    ("text", 8), // L8 306342 -> L9 306755 (+413 B)
    // Stored (L0) beats L1 on incompressible input: L0 emits the optimal
    // 17-block stored grid (+0 B slack) while L1 pays issue #266's
    // 32-block 65535+1 alternating grid (+75 B).
    ("noise", 0), // L0 1048679 -> L1 1048754 (+75 B)
];

/// Optimal stored framing for an n-byte input: gzip header (10) + trailer (8)
/// plus 5 bytes of block framing (BFINAL/BTYPE bits rounded up + LEN/NLEN)
/// per maximal 65535-byte stored block.
fn optimal_stored_framing(n: usize) -> usize {
    n.div_ceil(65535) * 5 + 18
}

/// Measured maximum slack over the optimal stored bound, T1, levels 0-9, on
/// the 1 MiB `noise` fixture. The 75 B maximum is at L1 and is issue #266:
/// the stored-block emitter there uses a 65535+1-byte alternating grid,
/// emitting 32 blocks where libdeflate emits the optimal 17 — 15 extra
/// 5-byte headers. (L0 is exactly optimal at 0 B; L2-L9 sit at 5 B, one
/// extra block header.) When #266 is fixed this const ratchets DOWN — the
/// tightening guard in `assert_noise_bounded` fails if the worst measured
/// level ever passes with margin below the pin, so a loose pin cannot linger.
const T1_MAX_SLACK: usize = 75;

/// Same bound at T4 (parallel path): each chunk seam adds framing, so the
/// slack is measured separately — 90 B max at L1 (issue #266's 75 B plus
/// seam framing), 20 B at L2-L9, 0 B at L0. Ratchets down like
/// [`T1_MAX_SLACK`].
const T4_MAX_SLACK: usize = 90;

/// Compress in-process at T1 through the production whole-buffer entry point.
fn compress_t1(data: &[u8], level: u32) -> Vec<u8> {
    gzippy::compress::deflate::encode_gzip_bytes_to_vec(data, level)
}

/// Compress in-process at T>1 through the library pipeline entry point.
fn compress_tn(data: &[u8], level: u32, threads: usize) -> Vec<u8> {
    let mut out = Vec::new();
    gzippy::compress::compress_bytes(data, &mut out, level as u8, threads)
        .expect("in-process parallel compression failed");
    out
}

/// Independent-decoder roundtrip: the stream must decode byte-exactly to the
/// input through flate2 (zlib-ng). MultiGzDecoder because T>1 output may be
/// multi-member; that is valid gzip and the only correctness bar.
fn assert_roundtrip(input: &[u8], stream: &[u8], what: &str) {
    let mut decoded = Vec::with_capacity(input.len());
    flate2::read::MultiGzDecoder::new(stream)
        .read_to_end(&mut decoded)
        .unwrap_or_else(|e| panic!("{what}: output is not valid gzip: {e}"));
    assert!(
        decoded == input,
        "{what}: roundtrip mismatch — decoded {} bytes, input {} bytes. \
         A size number from this stream is void.",
        decoded.len(),
        input.len()
    );
}

#[test]
fn ladder_is_monotone_t1() {
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let sizes: Vec<usize> = LEVELS
            .map(|level| {
                let out = compress_t1(&data, level);
                assert_roundtrip(&data, &out, &format!("{name} L{level} T1"));
                out.len()
            })
            .collect();
        for n in 0..9u32 {
            let (lo, hi) = (sizes[n as usize], sizes[n as usize + 1]);
            let listed = KNOWN_SAGS.contains(&(name, n));
            let sags = hi > lo;
            match (sags, listed) {
                (true, false) => panic!(
                    "ladder monotonicity violated: on fixture '{name}', level {} produced a \
                     LARGER output than level {n} ({hi} > {lo} bytes, +{} bytes). Asking for \
                     more compression must never cost size. If this sag is a knowingly \
                     accepted defect, list (\"{name}\", {n}) in KNOWN_SAGS with a receipt; \
                     otherwise fix the regression.",
                    n + 1,
                    hi - lo,
                ),
                (false, true) => panic!(
                    "known sag healed — remove it from KNOWN_SAGS in this commit: \
                     (\"{name}\", {n}) no longer sags (level {} = {hi} <= level {n} = {lo}).",
                    n + 1,
                ),
                _ => {}
            }
        }
    }
}

fn assert_noise_bounded(threads: usize, max_slack: usize, label: &str) {
    let data = fixtures::generate("noise");
    let bound = data.len() + optimal_stored_framing(data.len());
    let mut worst: i64 = i64::MIN;
    for level in LEVELS {
        let out = if threads == 1 {
            compress_t1(&data, level)
        } else {
            compress_tn(&data, level, threads)
        };
        assert_roundtrip(&data, &out, &format!("noise L{level} {label}"));
        assert!(
            out.len() <= bound + max_slack,
            "incompressible expansion bound violated: noise (1 MiB, incompressible) at \
             L{level} {label} compressed to {} bytes, but input ({}) + optimal stored \
             framing ({}) + pinned slack ({max_slack}) = {} bytes. The encoder is \
             spending more than the pinned framing overhead on data it cannot compress.",
            out.len(),
            data.len(),
            optimal_stored_framing(data.len()),
            bound + max_slack,
        );
        worst = worst.max(out.len() as i64 - bound as i64);
    }
    // The ratchet's other jaw: the pin must sit exactly on the worst measured
    // level. If an encoder improvement (e.g. fixing issue #266's 32-block
    // grid) lowers the worst slack, this fails until the const is tightened,
    // so the bound tracks the encoder instead of rotting loose.
    assert!(
        worst >= max_slack as i64,
        "slack pin is loose — the worst measured slack at {label} is now {worst} bytes but \
         the pin is {max_slack}. Ratchet the const down to {worst} in this commit."
    );
}

#[test]
fn noise_expansion_bounded_t1() {
    assert_noise_bounded(1, T1_MAX_SLACK, "T1");
}

#[test]
fn noise_expansion_bounded_t4() {
    assert_noise_bounded(4, T4_MAX_SLACK, "T4");
}

/// Measurement harness (not a gate): prints the per-pair ladder table and the
/// per-level slack table used to fill KNOWN_SAGS / T1_MAX_SLACK /
/// T4_MAX_SLACK. Run: cargo test --test size_invariants -- --ignored --nocapture
#[test]
#[ignore]
fn measure_tables() {
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let sizes: Vec<usize> = LEVELS.map(|l| compress_t1(&data, l).len()).collect();
        for n in 0..9u32 {
            let (lo, hi) = (sizes[n as usize], sizes[n as usize + 1]);
            let tag = if hi > lo { "  SAG" } else { "" };
            println!(
                "ladder {name:8} L{n}->L{}: {lo:8} -> {hi:8} ({:+}){tag}",
                n + 1,
                hi as i64 - lo as i64
            );
        }
    }
    let data = fixtures::generate("noise");
    let base = data.len() + optimal_stored_framing(data.len());
    for level in LEVELS {
        let t1 = compress_t1(&data, level).len();
        let t4 = compress_tn(&data, level, 4).len();
        println!(
            "slack noise L{level}: T1 {} B (out {t1}), T4 {} B (out {t4})",
            t1 as i64 - base as i64,
            t4 as i64 - base as i64
        );
    }
}
