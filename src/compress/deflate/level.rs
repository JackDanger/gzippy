//! Level → parser-parameters table (single source of truth for L0..L12).
//!
//! Port of the `switch (compression_level)` preset table in libdeflate
//! `vendor/libdeflate/lib/deflate_compress.c` (`deflate_alloc_compressor`,
//! ~:3920-4005). Each level selects a PARSE STRATEGY plus the two tuning knobs
//! the greedy/lazy parsers consume: `max_search_depth` and `nice_match_length`.
//!
//! Increment 2 implements the greedy (L2-4) and lazy/lazy2 (L5-9) strategies;
//! Increment 3 adds the near-optimal (L10-12) strategy; Increment 4 adds the
//! igzip-class one-pass FAST strategy for L1 (chainless single-probe hash table
//! + per-block cheapest-of-{dynamic,static,stored} Huffman coding — a port of
//! igzip `isal_deflate_body_base`). Increment 5 (ratio-hole fix, 2026-07)
//! gives L0 the SAME chainless matchfinder as L1 (`Strategy::Fast0`), but
//! skips the per-block dynamic-Huffman evaluation (always static-or-stored) —
//! cheaper than L1, and a real compressor instead of L0's old pure
//! stored-block passthrough.

use super::tables::DEFLATE_MAX_MATCH_LEN;

/// Parse strategy selected by a compression level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Level 0: igzip-class one-pass fast path, same chainless single-probe
    /// matchfinder as [`Strategy::Fast`], but each block is coded as the
    /// cheaper of a fixed (static) Huffman block or a stored block — the
    /// per-block DYNAMIC Huffman evaluation (canonical code build +
    /// length-limiting) that [`Strategy::Fast`] does is skipped entirely,
    /// which is the ratio/speed trade that makes L0 cheaper than L1. This
    /// replaces the old pure stored-block passthrough (which never
    /// compressed at all — see the L0 fix in the compression-ratio
    /// campaign).
    Fast0,
    /// Level 1: igzip-class one-pass fast path — chainless single-probe
    /// hash-table matchfinder + per-block cheapest-of-{dynamic,static,stored}
    /// Huffman coding. No hash chains, no depth loop
    /// (`vendor/isa-l/igzip/igzip_base.c:isal_deflate_body_base`).
    Fast,
    /// Greedy parse: always take the longest match at each position.
    Greedy,
    /// Lazy parse: defer a match one byte to check for a longer one.
    Lazy,
    /// Lazy2 parse: look ahead two positions.
    Lazy2,
    /// DETECTOR-GATED LAZY-L3 (promoted 2026-07-23 by supervisor adjudication
    /// of the `88cf1b09` gate record — see this module's level-3 arm for the
    /// full record): per-block GREEDY-vs-LAZY dispatch under a two-sided
    /// content detector — see `parse::gated`'s module doc comment. Produced
    /// unconditionally by this module's level-3 arm in EVERY build (the
    /// `l3-tune` Cargo feature now only controls whether `parse::gated`'s
    /// threshold/block-length knobs are env-var-overridable for the harness;
    /// it no longer gates whether L3 uses this strategy at all).
    /// Near-optimal parse: bt matchfinder + iterative min-cost-path DP (L10-12).
    NearOptimal,
}

/// Extra knobs for the near-optimal parser (`deflate_compress_near_optimal`).
#[derive(Clone, Copy, Debug)]
pub struct NearOptimalParams {
    /// `max_optim_passes` — max min-cost-path passes per block.
    pub max_optim_passes: u32,
    /// `min_improvement_to_continue` — stop passes early below this bit gain.
    pub min_improvement_to_continue: u32,
    /// `min_bits_to_use_nonfinal_path` — recover a prior pass's path if the
    /// final pass regressed by at least this many bits.
    pub min_bits_to_use_nonfinal_path: u32,
    /// `max_len_to_optimize_static_block` — block length below which to also
    /// optimize a static-Huffman solution.
    pub max_len_to_optimize_static_block: u32,
}

/// The parser parameters for a compression level.
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    pub strategy: Strategy,
    /// Cap on hash-chain nodes searched per position (`c->max_search_depth`).
    pub max_search_depth: u32,
    /// Stop searching once a match this long is found (`c->nice_match_length`).
    pub nice_match_length: u32,
    /// Near-optimal-only knobs (meaningful iff `strategy == NearOptimal`).
    pub near_optimal: NearOptimalParams,
}

/// Resolve a compression level (clamped to 0..=12) to its parser parameters.
///
/// The `max_search_depth`/`nice_match_length` values transliterate the vendor
/// presets exactly; the strategy mapping substitutes a fallback for the two
/// strategies not yet implemented in this increment (see the module docs).
pub fn params(level: u32) -> LevelParams {
    #[allow(unused_mut)]
    let mut p = params_inner(level);
    // Measurement-only frontier override; compiles to nothing when the feature is off.
    #[cfg(feature = "ladder-tune")]
    ladder_tune::apply(&mut p);
    // Report the knobs that were ACTUALLY RESOLVED by the production path, once
    // per process. `fulcrum explain` reads this and asserts it against observed
    // behaviour; without it, the tool would have to keep its own copy of this
    // table, which would rot silently and would be a source-read rather than an
    // observation. Emitting from inside `params` means what is reported is what
    // executed. Feature-gated (default OFF) and compiles to nothing when off.
    #[cfg(feature = "anatomy-counters")]
    emit_declared_once(level, &p);
    p
}

/// One line per process: `LEVEL_DECLARED={json}`.
#[cfg(feature = "anatomy-counters")]
fn emit_declared_once(level: u32, p: &LevelParams) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "LEVEL_DECLARED={{\"level\":{},\"strategy\":\"{:?}\",\"max_search_depth\":{},\"nice_match_length\":{}}}",
            level, p.strategy, p.max_search_depth, p.nice_match_length
        );
    });
}

/// MEASUREMENT-ONLY level→params override, for deriving our own ladder instead of
/// inheriting libdeflate's (see the `ladder-tune` feature comment in `Cargo.toml`).
///
/// WHY THIS EXISTS. A full T1 size census (`/root/sizeboard-all-12fcd0ed/census.json`,
/// 22 corpus files x L1-9 x 4 rivals) found that against libdeflate we are an EXACT
/// BYTE TIE on 154 of 198 cells — 22/22 files at every one of L2, L4, L5, L6, L7, L8,
/// L9 — because `params_inner` transliterates their preset table. A tie is not a win:
/// it leaves zero size headroom, so all 109 T4 cells that fail vs libdeflate fail by
/// exactly the seam growth, with 0 bytes of slack to absorb it. L3 is the ONLY deep
/// level where we diverge (Lazy where they run Greedy at the same knobs) and it is the
/// only one where we hold a margin: smaller on 20/22 files, median 44 KB.
///
/// The frontier we need — a config that is BOTH smaller and cheaper than their level N
/// — cannot be read off the vendor's 8 points, because every one of those points is a
/// choice they made. It has to be measured.
///
/// This is NOT a production knob. CLAUDE.md non-negotiable #3 forbids env vars changing
/// behaviour in the shipped path; the deliverable here is a STATIC table, and the
/// feature is default-off, marks the binary `+INSTRUMENTED`, and is refused by
/// `scripts/campaign/board-wall.sh`. Resolved exactly ONCE per process via `OnceLock`
/// — reading the environment inside a per-position loop previously cost ~1.3 BILLION
/// instructions and inflated an ablation by 10x.
#[cfg(feature = "ladder-tune")]
pub mod ladder_tune {
    use super::{LevelParams, Strategy};

    /// `GZIPPY_LADDER=<strategy>:<max_search_depth>:<nice_match_length>`,
    /// e.g. `lazy:12:14`. Absent or unparseable => no override.
    fn spec() -> Option<(Strategy, u32, u32)> {
        static S: std::sync::OnceLock<Option<(Strategy, u32, u32)>> = std::sync::OnceLock::new();
        *S.get_or_init(|| {
            let raw = std::env::var("GZIPPY_LADDER").ok()?;
            let mut it = raw.split(':');
            let strategy = match it.next()? {
                "fast0" => Strategy::Fast0,
                "fast" => Strategy::Fast,
                "greedy" => Strategy::Greedy,
                "lazy" => Strategy::Lazy,
                "lazy2" => Strategy::Lazy2,
                "nearoptimal" => Strategy::NearOptimal,
                other => panic!("GZIPPY_LADDER: unknown strategy {other:?}"),
            };
            let depth = it.next()?.parse().ok()?;
            let nice = it.next()?.parse().ok()?;
            Some((strategy, depth, nice))
        })
    }

    pub fn apply(p: &mut LevelParams) {
        if let Some((strategy, depth, nice)) = spec() {
            p.strategy = strategy;
            p.max_search_depth = depth;
            p.nice_match_length = nice;
        }
    }
}

fn params_inner(level: u32) -> LevelParams {
    let max_match = DEFLATE_MAX_MATCH_LEN;
    // Placeholder near-optimal knobs for the non-near-optimal levels (unused).
    const NONE_NO: NearOptimalParams = NearOptimalParams {
        max_optim_passes: 0,
        min_improvement_to_continue: 0,
        min_bits_to_use_nonfinal_path: 0,
        max_len_to_optimize_static_block: 0,
    };
    match level {
        0 => LevelParams {
            strategy: Strategy::Fast0,
            max_search_depth: 0,
            nice_match_length: 32,
            near_optimal: NONE_NO,
        },
        // Native L1 is the igzip-class one-pass FAST path (Increment 4):
        // chainless single-probe hash table + direct-emit static Huffman. The
        // search-depth / nice-len knobs are unused by `Strategy::Fast` (it does
        // exactly one probe per position); they are left at the vendor-ish
        // values only so the struct is populated.
        1 => LevelParams {
            strategy: Strategy::Fast,
            max_search_depth: 1,
            nice_match_length: 32,
            near_optimal: NONE_NO,
        },
        2 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 6,
            nice_match_length: 10,
            near_optimal: NONE_NO,
        },
        // ⚠ STALE BELOW, KEPT FOR THE HISTORY ONLY: `Strategy::LazyGated` and
        // `parse::gated.rs` NO LONGER EXIST (deleted by user order — `CLAUDE.md`
        // non-negotiable #3 forbids content detectors choosing a parser). L3 routes to
        // `Strategy::Lazy` unconditionally; see the `3 => LevelParams` arm below, which
        // is the fact. The original note read:
        // L3 = DETECTOR-GATED LAZY (`Strategy::LazyGated` -> `parse::gated::run`,
        // per-block GREEDY-vs-LAZY dispatch under a two-sided literal-fraction
        // content detector — see `parse::gated`'s module doc comment).
        // Knobs unchanged from the prior plain-Greedy L3 (max_search_depth=12,
        // nice_match_length=14); the gate's own params (two-sided 34/95 pct
        // thresholds, 300KB detection block, initial_lazy=false) are
        // `parse::gated`'s `L3_GATE_*` constants, unconditionally in effect —
        // `l3-tune` no longer selects WHETHER L3 uses this strategy, only
        // whether those constants are env-var-overridable (see below).
        //
        // PROMOTION HISTORY (full campaign in git log; `2c7f9444` plain-lazy
        // -> `992c5837` strict-Pareto FAIL (ecoli.fastq/weights.safetensors
        // regress) -> `2c7f9444`-successor `parse::gated` composition fixes
        // both -> `2b566fcb` self-tax-vs-Greedy wall gate FAILS (wrong
        // rival) -> `88cf1b09` re-gated against the REAL rivals, pigz-3/
        // gzip-3, which is the record this promotion is adjudicated from).
        //
        // ADJUDICATION (2026-07-23, supervisor, promoting `88cf1b09`'s
        // frozen record — AMD EPYC 7282 Zen2 solvency, `/root/gz-l3final` +
        // `/root/l3final/{wall_results,wall_ld_results}.jsonl`, N=15
        // paired-diff/A/A-controlled, `/dev/null` sink, 6-file corpus x
        // T1/4/8/16 x {pigz-3, gzip-3}):
        //   - SIZE: strictly SMALLER than shipped Greedy on every file
        //     (the L3 campaign's original goal). Smaller than libdeflate-gzip-3
        //     on every file including `dd79_bin6` (was byte-EQUAL to ld-3
        //     under Greedy, a faithful-port confirmation, not a routing bug).
        //   - WALL: beats pigz-3/gzip-3 by 12-62% on all 30 (file x T x
        //     rival) cells, every 95% CI excluding 1.0 and clearing the ~1%
        //     A/A spread — INCLUDING `dd79_bin6` (21-45% faster) at every T.
        //   - ZERO class regressions: L2/L4/L6 byte-identical
        //     Greedy-vs-LazyGated builds; roundtrip byte-exact all files x
        //     T1/4/8/16; T4==T16 output byte-identical; `cargo test
        //     --release` green + clippy/fmt clean both feature states.
        //   - The ONE failing sub-leg — `dd79_bin6` size vs pigz-3/gzip-3
        //     (+0.445-0.767%, deterministic, zero-variance, narrowed from
        //     Greedy's 1.339% miss but not closed) — is a conjunctive-rule
        //     miss on a cell that is SPEED-ONLY today (L3 has never been the
        //     ship default on a size-vs-rivals basis; `dd79_bin6` already
        //     lost this same size comparison under Greedy) and remains
        //     SPEED-ONLY after this flip. Its residual is the SAME
        //     `match_diff` parse-quality gap `2c7f9444`'s own fulcrum
        //     diagnosis located (an optimal-frontier match-choice question,
        //     not an accept-vs-defer one lazy/greedy toggling reaches) —
        //     independent of this promotion and not a regression it
        //     introduces. Adjudicated PROMOTE: every gating leg (size vs
        //     shipped default, size vs ld-3, wall vs both real rivals at
        //     every T, zero-regression legs) clears; the recorded miss is
        //     out of this change's causal reach.
        3 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 12,
            nice_match_length: 14,
            near_optimal: NONE_NO,
        },
        4 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 16,
            nice_match_length: 30,
            near_optimal: NONE_NO,
        },
        5 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 16,
            nice_match_length: 30,
            near_optimal: NONE_NO,
        },
        6 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 35,
            nice_match_length: 65,
            near_optimal: NONE_NO,
        },
        7 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 100,
            nice_match_length: 130,
            near_optimal: NONE_NO,
        },
        8 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 300,
            nice_match_length: max_match,
            near_optimal: NONE_NO,
        },
        9 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 600,
            nice_match_length: max_match,
            near_optimal: NONE_NO,
        },
        // Native near-optimal parser (`deflate_compress_near_optimal`,
        // deflate_compress.c:3974-4004).
        10 => LevelParams {
            strategy: Strategy::NearOptimal,
            max_search_depth: 35,
            nice_match_length: 75,
            near_optimal: NearOptimalParams {
                max_optim_passes: 2,
                min_improvement_to_continue: 32,
                min_bits_to_use_nonfinal_path: 32,
                max_len_to_optimize_static_block: 0,
            },
        },
        11 => LevelParams {
            strategy: Strategy::NearOptimal,
            max_search_depth: 100,
            nice_match_length: 150,
            near_optimal: NearOptimalParams {
                max_optim_passes: 4,
                min_improvement_to_continue: 16,
                min_bits_to_use_nonfinal_path: 16,
                max_len_to_optimize_static_block: 1000,
            },
        },
        _ => LevelParams {
            strategy: Strategy::NearOptimal,
            max_search_depth: 300,
            nice_match_length: max_match,
            near_optimal: NearOptimalParams {
                max_optim_passes: 10,
                min_improvement_to_continue: 1,
                min_bits_to_use_nonfinal_path: 1,
                max_len_to_optimize_static_block: 10000,
            },
        },
    }
}

/// `max_passthrough_size` (`deflate_compress.c:3918`): inputs at or below this
/// size are emitted as a stored block without running the parser. `55 - 4*level`
/// for the near-optimal levels (negative/overflow clamps to 0 for lower levels
/// which are handled by their own passthrough).
///
/// Ported (formula pinned by the tests below) but not yet wired into
/// [`super::parse::near_optimal`]'s entry point — no call site skips the
/// parser for tiny near-optimal-level inputs today. Left as a documented,
/// tested residual (Stage E dead-code audit,
/// docs/compressor-architecture.md §5-E) rather than deleted: wiring it in
/// would change L10-12 output for inputs at/below the threshold, which is an
/// algorithmic change out of scope for a polish stage, but deleting a
/// correct, vendor-cited, unit-tested port for no reason would just be
/// throwing the work away. Re-open trigger: wiring this in is a real
/// (untried) small near-optimal-levels win, gated like any other lever.
#[allow(dead_code)]
pub fn max_passthrough_size(level: u32) -> usize {
    (55i64 - 4 * level as i64).max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_mapping_matches_increment_scope() {
        assert_eq!(params(0).strategy, Strategy::Fast0);
        assert_eq!(params(1).strategy, Strategy::Fast); // igzip-class one-pass

        // ⚠ STALE: LazyGated is deleted; L3 is `Strategy::Lazy`. Original note:
        // L3 = LazyGated unconditionally since the 2026-07-23 supervisor
        // adjudication of the frozen cell gate (see the level-3 arm's
        // promotion-history comment); `l3-tune` only makes the gate's
        // constants env-overridable for the search harness. L2/L4 stay
        // Greedy either way.
        assert_eq!(params(2).strategy, Strategy::Greedy, "level 2");
        assert_eq!(params(3).strategy, Strategy::Lazy, "level 3");
        assert_eq!(params(4).strategy, Strategy::Greedy, "level 4");
        for l in 5..=7 {
            assert_eq!(params(l).strategy, Strategy::Lazy, "level {l}");
        }
        for l in 8..=9 {
            assert_eq!(params(l).strategy, Strategy::Lazy2, "level {l}");
        }
        for l in 10..=12 {
            assert_eq!(params(l).strategy, Strategy::NearOptimal, "level {l}");
        }
    }

    // DELETED 2026-07-31: `vendor_knob_values`, which asserted
    //     params(6).max_search_depth == 35   params(9).max_search_depth == 600
    //
    // IT WAS A CAGE, AND IT BEAT THE DOCUMENTATION. `CLAUDE.md` says the level->config
    // map "is currently a copy of libdeflate's, which is why we run their algorithm
    // slower than they do; it is free to change." This test said the opposite, in the
    // only medium that fails closed: touch the map, turn a test red. When a doc and a red
    // test disagree, the test wins every time, and it won for weeks —
    // `.git/logs/HEAD:235-237` records `probe/l5-depth` created and abandoned 102 seconds
    // later with no commit.
    //
    // Raising L5-L9 to zlib-ng's chain depths — the change this test forbade — closed 84
    // failing size cells (2026-07-31, full board, 1,320 measured, 0 VOID).
    //
    // A knob that is DECLARED FREE TO CHANGE must not be pinned by an equality assertion.
    // What is worth testing is the INVARIANT, not the value: depth must not decrease as
    // the level rises, because that is what "higher level = more effort" means and it is
    // the property a typo would actually break.
    #[test]
    fn search_effort_is_monotonic_in_level() {
        for l in 2..=9u32 {
            let prev = params(l - 1);
            let cur = params(l);
            assert!(
                cur.max_search_depth >= prev.max_search_depth,
                "L{l} searches less deeply than L{}: {} < {}",
                l - 1,
                cur.max_search_depth,
                prev.max_search_depth
            );
        }
        // nice_match_length is capped by the format, never exceeds it.
        for l in 0..=9u32 {
            assert!(
                params(l).nice_match_length <= DEFLATE_MAX_MATCH_LEN,
                "level {l}"
            );
        }
    }

    #[test]
    fn near_optimal_effort_is_monotonic_in_level() {
        // DELETED 2026-07-31: `near_optimal_knob_values`, which asserted
        //     params(10).max_search_depth == 35, .nice_match_length == 75,
        //     .max_optim_passes == 2, params(12).max_search_depth == 300, etc.
        //
        // SECOND INSTANCE OF THE CAGE. `vendor_knob_values` was deleted earlier today for
        // pinning L6/L9 depths that `CLAUDE.md` declares free to change; this pinned the
        // L10-L12 knobs the same way, and non-negotiable #5 now forbids it outright:
        // an equality assertion beats a sentence in a doc, because only one fails closed.
        // Found by an adversarial review after the first one was fixed — the pattern
        // repeats, so test the INVARIANT, not the VALUE.
        //
        // The invariant a typo would actually break is that effort rises with level.
        for l in 11..=12u32 {
            let prev = params(l - 1);
            let cur = params(l);
            assert!(
                cur.max_search_depth >= prev.max_search_depth,
                "L{l} searches less deeply than L{}",
                l - 1
            );
            assert!(
                cur.near_optimal.max_optim_passes >= prev.near_optimal.max_optim_passes,
                "L{l} runs fewer optimisation passes than L{}",
                l - 1
            );
        }
        for l in 10..=12u32 {
            let p = params(l);
            assert_eq!(p.strategy, Strategy::NearOptimal, "level {l}");
            assert!(p.nice_match_length <= DEFLATE_MAX_MATCH_LEN, "level {l}");
            assert!(p.near_optimal.max_optim_passes >= 1, "level {l}");
        }
    }
}
