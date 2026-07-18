//! Level → parser-parameters table (single source of truth for L0..L12).
//!
//! Port of the `switch (compression_level)` preset table in libdeflate
//! `vendor/libdeflate/lib/deflate_compress.c` (`deflate_alloc_compressor`,
//! ~:3920-4005). Each level selects a PARSE STRATEGY plus the two tuning knobs
//! the greedy/lazy parsers consume: `max_search_depth` and `nice_match_length`.
//!
//! Increment 2 implements the greedy (L2-4) and lazy/lazy2 (L5-9) strategies.
//! Levels whose native strategy is not yet built fall back to the closest
//! implemented one with an explicit TODO:
//!   * L1  (`deflate_compress_fastest`, ht_matchfinder) → **greedy** (TODO Inc4).
//!   * L10-12 (`deflate_compress_near_optimal`)         → **lazy2** (TODO Inc3).
//! L0 stays a pure stored-block passthrough.

use super::tables::DEFLATE_MAX_MATCH_LEN;

/// Parse strategy selected by a compression level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Level 0: emit stored (uncompressed) blocks only — no match finding.
    Stored,
    /// Greedy parse: always take the longest match at each position.
    Greedy,
    /// Lazy parse: defer a match one byte to check for a longer one.
    Lazy,
    /// Lazy2 parse: look ahead two positions.
    Lazy2,
}

/// The parser parameters for a compression level.
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    pub strategy: Strategy,
    /// Cap on hash-chain nodes searched per position (`c->max_search_depth`).
    pub max_search_depth: u32,
    /// Stop searching once a match this long is found (`c->nice_match_length`).
    pub nice_match_length: u32,
}

/// Resolve a compression level (clamped to 0..=12) to its parser parameters.
///
/// The `max_search_depth`/`nice_match_length` values transliterate the vendor
/// presets exactly; the strategy mapping substitutes a fallback for the two
/// strategies not yet implemented in this increment (see the module docs).
pub fn params(level: u32) -> LevelParams {
    let max_match = DEFLATE_MAX_MATCH_LEN;
    match level {
        0 => LevelParams {
            strategy: Strategy::Stored,
            max_search_depth: 0,
            nice_match_length: 32,
        },
        // TODO(Increment 4): native L1 is `deflate_compress_fastest`
        // (ht_matchfinder, fixed-length blocks). Until it exists, route L1
        // through the greedy parser with light search settings.
        1 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 6,
            nice_match_length: 32,
        },
        2 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 6,
            nice_match_length: 10,
        },
        3 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 12,
            nice_match_length: 14,
        },
        4 => LevelParams {
            strategy: Strategy::Greedy,
            max_search_depth: 16,
            nice_match_length: 30,
        },
        5 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 16,
            nice_match_length: 30,
        },
        6 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 35,
            nice_match_length: 65,
        },
        7 => LevelParams {
            strategy: Strategy::Lazy,
            max_search_depth: 100,
            nice_match_length: 130,
        },
        8 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 300,
            nice_match_length: max_match,
        },
        9 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 600,
            nice_match_length: max_match,
        },
        // TODO(Increment 3): native L10-12 is `deflate_compress_near_optimal`
        // (cost-model minimum-cost parse). Until it exists, route L10-12
        // through lazy2 with the vendor near-optimal search knobs.
        10 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 35,
            nice_match_length: 75,
        },
        11 => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 100,
            nice_match_length: 150,
        },
        _ => LevelParams {
            strategy: Strategy::Lazy2,
            max_search_depth: 300,
            nice_match_length: max_match,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_mapping_matches_increment_scope() {
        assert_eq!(params(0).strategy, Strategy::Stored);
        assert_eq!(params(1).strategy, Strategy::Greedy); // fallback
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
            assert_eq!(params(l).strategy, Strategy::Lazy2, "level {l} fallback");
        }
    }

    #[test]
    fn vendor_knob_values() {
        assert_eq!(params(6).max_search_depth, 35);
        assert_eq!(params(6).nice_match_length, 65);
        assert_eq!(params(9).max_search_depth, 600);
        assert_eq!(params(9).nice_match_length, DEFLATE_MAX_MATCH_LEN);
    }
}
