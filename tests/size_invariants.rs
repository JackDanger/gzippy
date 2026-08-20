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
// ⚠ REBASELINED 2026-08-20 — the T1 CUMULATIVE RATCHET WAS REMOVED, and several
// entries below were healed BY that ratchet rather than by any parse improvement.
// See `deflate_one_shot_t1_ratcheted`'s doc comment for the full reasoning; the
// short version is CLAUDE.md hard stop 10 (a test is not a goal): the ratchet
// bought unconditional ladder monotonicity for a MEASURED 7.3x/8.7x wall cost,
// and a full-corpus one-directional check (23 files x L1-L5 x gzip/pigz/libdeflate,
// 345 cells, every stream sha256-roundtripped) found its entire size benefit was
// margin ABOVE what any rival requires: **0 pass->fail cell flips**, 340 passing
// before and after. It closed nothing; it only made us slower. Removing it costs
// 1.73x-4.50x LESS wall (paired hyperfine, aarch64) and re-exposes the sags below,
// which are therefore honest pre-existing defects, not new regressions.
const KNOWN_SAGS: &[(&str, u32)] = &[
    // ("noise", 0) and ("noise", 1) HEALED for real (not by the ratchet) —
    // the issue #266 fix (stored-span coalescing in `parse::StoredCoalescer`)
    // gave L0/L1 the optimal 17-block stored grid, and L2 mmap pick-min (#332)
    // matched it at L2 (all 1048679 B). Both stay healed without the ratchet.
    //
    // The residual +5 B stored-grid sag is real and predates the ratchet: L3-L9's
    // greedy/lazy paths parse on the SOFT_MAX_BLOCK (300,000 B) grid and emit each
    // block's stored payload uncoalesced (65535x4 + 37860 runt per parse block),
    // costing one extra block header vs L1/L2's coalesced 17-block grid. That grid
    // is byte-identical to libdeflate's own (tie-cage cells), so healing it means
    // deliberately breaking those ties in our favour by extending StoredCoalescer
    // to greedy/lazy — a separate, tie-guard-adjudicated lever, still not built.
    // The ratchet formerly hid this at L3/L4/L5 by substituting L2's bytes, which
    // pushed the identical +5 B to the edge of its scope, ("noise", 5). With the
    // ratchet gone it returns to its true origin at L2->L3.
    ("noise", 2), // L2 1048679 -> L3 1048684 (+5 B)
    // The "level ladder sags at L4" defect family — the SAME defect memory records
    // on the real corpus (`-4` larger than `-3` on 10/11 TUNE files;
    // project_the_level_ladder_sags_at_L4.md). L4 widens the search
    // (`max_search_depth`) and the longer search produces a genuinely larger parse
    // on this fixture — `level.rs`'s own engine.wasm L8 receipt, recurring. The
    // ratchet masked it by handing L4 L3's bytes; it never fixed it. Real fix is a
    // better L4 parse, not a fold.
    ("text", 3), // L3 324800 -> L4 333536 (+8736 B)
    // High-level sag on prose, outside the (former) ratchet's 1..=5 scope entirely
    // and so unaffected by its removal: L7 is simply the best level on `text`.
    ("text", 7), // L7 303038 -> L8 306342 (+3304 B)
    ("text", 8), // L8 306342 -> L9 306755 (+413 B)
                 // ("binary", 1) was listed as HEALED-BY-RATCHET (L2 taking L1's 662,577
                 // instead of its own 666,108). With the ratchet removed L2 returns to
                 // 666,108 — which STILL BEATS libdeflate-gzip -2's 666,112, so the cell
                 // this entry was worried about stays won either way, and no sag appears
                 // here because L1 and L2 both land on the same side of the comparison.
                 // Verified in the 345-cell sweep above, not inferred.
];

/// Optimal stored framing for an n-byte input: gzip header (10) + trailer (8)
/// plus 5 bytes of block framing (BFINAL/BTYPE bits rounded up + LEN/NLEN)
/// per maximal 65535-byte stored block.
fn optimal_stored_framing(n: usize) -> usize {
    n.div_ceil(65535) * 5 + 18
}

/// Measured maximum slack over the optimal stored bound, T1, levels 0-9, on
/// the 1 MiB `noise` fixture. With issue #266 fixed (stored-span coalescing:
/// L0/L1 now emit the optimal 17-block maximal grid, 0 B slack) the worst
/// level is L2-L9's 5 B — one extra block header from the SOFT_MAX_BLOCK
/// (300,000 B) parse grid, whose 300000-mod-65535 runt sub-blocks libdeflate
/// shares. This const ratchets DOWN — the tightening guard in
/// `assert_noise_bounded` fails if the worst measured level ever passes with
/// margin below the pin, so a loose pin cannot linger.
const T1_MAX_SLACK: usize = 5;

/// Same bound at T4 (parallel path): each chunk seam adds framing, so the
/// slack is measured separately. With #266 fixed the in-process pipeline's
/// worst level also sits at 5 B. Ratchets down like [`T1_MAX_SLACK`].
const T4_MAX_SLACK: usize = 5;

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

/// Compress at T1 through the STREAMING/library route — `compress_bytes` with
/// `threads=1`, the same `compress_with_pipeline_sized` -> `encode_gzip_reader_to_writer_sized`
/// path the CLI takes for stdin/pipe input and every library caller takes
/// regardless of input source. Deliberately NOT `compress_t1`
/// (`encode_gzip_bytes_to_vec`, the whole-buffer/mmap entry point) directly —
/// this test exercises a DIFFERENT production entry point with its own dispatch.
///
/// UPDATED 2026-08-18 (streaming-route ladder fix, `encode_gzip_single_pass`):
/// for any input that fits in the first ~4.56 MiB refill (all fixtures here
/// do — 1 MiB), this route now calls `encode_gzip_slack_padded_to_vec`
/// directly and so IS byte-identical to `compress_t1`/the mmap route,
/// including `deflate_one_shot_t1_ratcheted` for L1-5. Only genuinely large
/// (>~4.56 MiB) streamed input still takes the un-ratcheted single-arm path —
/// see `PLAN.md` "PROMOTION PAUSED" for that residual scope.
fn compress_t1_streaming(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::new();
    gzippy::compress::compress_bytes(data, &mut out, level as u8, 1)
        .expect("in-process T1 streaming compression failed");
    out
}

/// Sags SHARED with the mmap route's own [`KNOWN_SAGS`]. Below the ~4.56 MiB
/// refill threshold the streaming route calls `encode_gzip_slack_padded_to_vec`
/// and is therefore byte-IDENTICAL to the mmap route, so for these 1 MiB
/// fixtures the two lists must agree entry-for-entry — they are kept as separate
/// consts because the two routes' accepted-defect sets are not required to
/// coincide in general (above the threshold they genuinely do not).
///
/// ⚠ REBASELINED 2026-08-20 alongside [`KNOWN_SAGS`] when the T1 cumulative
/// ratchet was removed (measured: 0 pass->fail rival cell flips across 345
/// cells, 1.73x-4.50x less wall — see that const's comment and
/// `deflate_one_shot_t1_ratcheted`). `("noise", 5)` was an artefact of the
/// ratchet's own L5/L6 scope cutoff and is GONE; the true stored-grid sag it
/// displaced is back at its origin, `("noise", 2)`. `("text", 3)` is the
/// "ladder sags at L4" defect family the ratchet masked rather than fixed.
/// `("text", 7)`/`("text", 8)` are the older, deeper "L7 is the best level on
/// prose" property of `params(level)`'s high-level table — outside the former
/// ratchet's 1..=5 scope on EITHER route, and confirmed present on
/// `origin/main` before `c8bbde67`/`b4b821c9` ever existed.
///
/// NOTE this list is a weaker guard than [`KNOWN_SAGS`]: the loop below only
/// SKIPS a listed pair, so a listed sag that heals passes silently here,
/// whereas `KNOWN_SAGS` fails closed in both directions and can only shrink.
const STREAMING_SHARED_KNOWN_SAGS: &[(&str, u32)] =
    &[("noise", 2), ("text", 3), ("text", 7), ("text", 8)];

/// **PARTIALLY FIXED 2026-08-18/19** (was a known gap from Codex's `b4b821c9` pre-merge
/// review; partially fixed same day per Fable + cursor-agent's independent streaming-route
/// design reviews — see `PLAN.md`'s CRITICAL section, then "PROMOTION PAUSED", for the
/// full history — status is more complicated than either heading alone suggests).
///
/// `ladder_is_monotone_t1` proves the T1 WHOLE-BUFFER/MMAP route
/// (`encode_gzip_bytes_to_vec`) is ladder-monotone via `deflate_one_shot_t1_ratcheted`.
/// The STREAMING route — CLI stdin/pipe input and every `compress_bytes`/
/// `compress_with_pipeline` library caller regardless of input source — used to
/// dispatch unconditionally to `encode_gzip_reader_to_writer_sized`'s single-arm
/// per-chunk parse, entirely bypassing the ratchet. Fixed in
/// `encode_gzip_single_pass` (`mod.rs`): the function already buffers the ENTIRE
/// input before parsing anything whenever it fits in the first ~4.56 MiB refill
/// (Fable's finding) — for that case (all four `fixtures::NAMES`, 1 MiB each, and
/// the overwhelming majority of real files/pipes BY COUNT), it now routes through
/// `encode_gzip_slack_padded_to_vec` directly, making streaming byte-identical to
/// the mmap route including the ratchet.
///
/// **⚠ CORRECTION 2026-08-19: "zero extra memory or latency" (the original wording
/// here and in the `12cd8094` commit message) was FALSE — measured, not just
/// imprecise.** Hyperfine, paired, 3 MiB incompressible pipe at `-5`: 295ms on this
/// branch vs 40ms on the prior code and 34ms on libdeflate — 7.3x and 8.7x slower,
/// 11x more instructions, 3x peak RSS. The input READ buffer genuinely doesn't grow
/// (that part was correct), but the RATCHET the whole-buffer branch now runs does
/// real, uncosted extra work (up to ~10-12 cumulative arms by L5) — the exact same
/// cost this session then found on the ORIGINAL mmap-route ratchet too, via the
/// wall census that had been running the whole time. See `PLAN.md`'s CRITICAL
/// section: this is now the dominant open question for the whole branch, on BOTH
/// routes, not a streaming-specific residual.
///
/// **Residual, NOT fixed — inputs LARGER than the ~4.56 MiB refill boundary still
/// take the single-arm streaming path with no monotonicity guarantee.** Reproduced
/// directly on a 5 MiB pipe input (2026-08-19): real violations at L2→L3 and L4→L5.
/// This test's fixtures are all 1 MiB and so cannot exercise that residual — it is
/// intentionally out of this test's reach, and has ZERO test coverage anywhere in
/// this suite as of this comment; adding that coverage is next, but is secondary to
/// the wall-cost question above, which affects whether the below-threshold "fix"
/// should even ship as currently built.
#[test]
fn streaming_t1_is_ladder_monotone() {
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let sizes: Vec<usize> = LEVELS
            .map(|level| {
                let out = compress_t1_streaming(&data, level);
                assert_roundtrip(&data, &out, &format!("{name} L{level} T1-streaming"));
                out.len()
            })
            .collect();
        for n in 0..9u32 {
            let (lo, hi) = (sizes[n as usize], sizes[n as usize + 1]);
            if hi > lo && STREAMING_SHARED_KNOWN_SAGS.contains(&(name, n)) {
                continue;
            }
            assert!(
                hi <= lo,
                "STREAMING ladder monotonicity violated: on fixture '{name}', level {} \
                 produced a LARGER output than level {n} via the T1 streaming/library route \
                 ({hi} > {lo} bytes, +{} bytes). The whole-buffer/mmap route already fixed \
                 this via deflate_one_shot_t1_ratcheted (b4b821c9) — this route needs the \
                 same guarantee, by a mechanism compatible with single-pass streaming.",
                n + 1,
                hi - lo,
            );
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
