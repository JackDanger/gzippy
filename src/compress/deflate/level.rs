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
        3 => LevelParams {
            strategy: Strategy::Greedy,
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
        for l in 2..=4 {
            assert_eq!(params(l).strategy, Strategy::Greedy, "level {l}");
        }
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
