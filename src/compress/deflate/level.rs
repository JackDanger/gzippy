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
    /// DETECTOR-GATED LAZY-L3 (`l3-tune` feature, 2026-07-23 mission): per-block
    /// GREEDY-vs-LAZY dispatch under a two-sided content detector — see
    /// `parse::gated`'s module doc comment. Only ever produced by this
    /// module's level-3 arm under `l3-tune`; a default build never selects
    /// it (dead code path there, kept exhaustive-match-compatible).
    #[cfg_attr(not(feature = "l3-tune"), allow(dead_code))]
    LazyGated,
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
        // L3-STRATEGY experiment (`l3-tune` Cargo feature, OFF by default):
        // routes L3 through the SAME lazy parser (`Strategy::Lazy` ->
        // `lazy::run`, already wired and used unmodified by L5-7) instead of
        // `Strategy::Greedy`, with L3's knobs UNCHANGED
        // (max_search_depth=12, nice_match_length=14) — only the
        // accept-immediately-vs-defer-one-byte decision changes, isolating
        // "parse decision quality" from "finder reach" per the residual
        // diagnosis (dd79_bin6 vs pigz-3: decision headroom 190185B is 3.2x
        // the real 59507B gap; reach saturates by K~=32-48, see
        // ~/www/gzippy-bench/l3_diag/l3_diag_notes.txt §4/§7). Byte-identical
        // to today's shipped L3 (`Strategy::Greedy`) when the feature is off.
        //
        // RE-GATED 2026-07-23 under a STRICT-PARETO promotion rule (size
        // strictly <= current-default on ALL 21 breadth files + 4 fixtures,
        // zero larger) — FAILED leg (a): ecoli.fastq (+0.3146%) and
        // weights.safetensors (+0.0378%) regress vs the Greedy default,
        // reproduced byte-identical on both Apple M1 Pro (arm64) and AMD
        // EPYC 7282 Zen2 (x86_64, solvency). This is a deterministic,
        // zero-variance result (compressed size has no run-to-run noise),
        // so it conclusively blocks promotion regardless of the downstream
        // wall/rival legs — see the re-gate verdict commit for the full
        // record. The plain-lazy config is NOT PROMOTED.
        //
        // DETECTOR-GATED LAZY-L3 (2026-07-23, `l3-tune`, same feature):
        // the re-gate's own failure named its next configuration —
        // content-detector-gate the lazy dispatch per block (the hash3-gate
        // precedent, `parse::gated`) so ecoli/weights-class blocks keep
        // GREEDY while everything else gets LAZY. L3's arm now selects
        // `Strategy::LazyGated` (-> `parse::gated::run`) instead of plain
        // `Strategy::Lazy` under `l3-tune`; plain lazy is still fully
        // reproducible through the SAME entry point via
        // `gated::tune::L3GateTune::enabled = false` (the sweep's control
        // arm — see `gated.rs`'s doc comment). Byte-identical to today's
        // shipped L3 (`Strategy::Greedy`) when `l3-tune` is off.
        //
        // VERDICT (2026-07-23, see [`super::parse::gated::L3_GATE_ENABLED`]'s
        // doc comment for the full numbers): SIZE leg CLEARS — 21/21 strict
        // Pareto on the 21-file breadth corpus at T1/T4/T16, INCLUDING the
        // two files (`ecoli.fastq`, `weights.safetensors`) that blocked the
        // plain-lazy re-gate above. WALL leg FAILS — the pre-registered
        // `self-tax <= +10%` bar is missed by several stays-LAZY middle-band
        // files (`aozora.txt` +25%, `dd79_bin6` +13%, `dickens` +11%, local
        // M1 N>=21 `hyperfine`), traced to `Strategy::Lazy`'s OWN inherent
        // per-position lookahead cost (a control run with the gate DISABLED
        // shows the SAME tax, so the detector/dispatch machinery itself is
        // not the cost). NOT PROMOTED, same disposition as the plain-lazy
        // re-gate above — `l3-tune` stays default-off; the full frozen
        // solvency gate was correctly not run (mission's own "any leg fails
        // -> keep default, record" escape hatch).
        3 => LevelParams {
            #[cfg(not(feature = "l3-tune"))]
            strategy: Strategy::Greedy,
            #[cfg(feature = "l3-tune")]
            strategy: Strategy::LazyGated,
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

        // L3's strategy flips to LazyGated under the `l3-tune` experiment
        // (see level.rs's level-3 arm); L2/L4 stay Greedy either way. Plain
        // Lazy's strict-Pareto promotion re-gate (2026-07-23) FAILED leg (a)
        // (2/21 breadth files regress), so the default stays Greedy; the
        // DETECTOR-GATED composition (`parse::gated`) is the re-gate's named
        // next configuration.
        assert_eq!(params(2).strategy, Strategy::Greedy, "level 2");
        #[cfg(not(feature = "l3-tune"))]
        assert_eq!(params(3).strategy, Strategy::Greedy, "level 3");
        #[cfg(feature = "l3-tune")]
        assert_eq!(params(3).strategy, Strategy::LazyGated, "level 3 (l3-tune)");
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

    #[test]
    fn vendor_knob_values() {
        assert_eq!(params(6).max_search_depth, 35);
        assert_eq!(params(6).nice_match_length, 65);
        assert_eq!(params(9).max_search_depth, 600);
        assert_eq!(params(9).nice_match_length, DEFLATE_MAX_MATCH_LEN);
    }

    #[test]
    fn near_optimal_knob_values() {
        let l10 = params(10);
        assert_eq!(l10.max_search_depth, 35);
        assert_eq!(l10.nice_match_length, 75);
        assert_eq!(l10.near_optimal.max_optim_passes, 2);
        assert_eq!(l10.near_optimal.min_improvement_to_continue, 32);
        assert_eq!(l10.near_optimal.max_len_to_optimize_static_block, 0);

        let l12 = params(12);
        assert_eq!(l12.max_search_depth, 300);
        assert_eq!(l12.nice_match_length, DEFLATE_MAX_MATCH_LEN);
        assert_eq!(l12.near_optimal.max_optim_passes, 10);
        assert_eq!(l12.near_optimal.min_improvement_to_continue, 1);
        assert_eq!(l12.near_optimal.max_len_to_optimize_static_block, 10000);

        assert_eq!(max_passthrough_size(10), 15);
        assert_eq!(max_passthrough_size(12), 7);
    }
}
