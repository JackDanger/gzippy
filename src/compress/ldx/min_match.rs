//! C: `vendor/libdeflate/lib/deflate_compress.c:2286-2380` — choosing the minimum
//! match length for the greedy and lazy parsers.
//!
//! By default the minimum match length is 3, which is the smallest length the DEFLATE
//! format allows. However, with greedy and lazy parsing, some data (e.g. DNA
//! sequencing data) benefits greatly from a longer minimum length. Typically this is
//! because literals are very cheap. In general, the near-optimal parser handles this
//! case naturally, but the greedy and lazy parsers need a heuristic to decide when to
//! use short matches.
//!
//! The heuristic is to make the minimum match length depend on the number of different
//! literals that exist in the data. Many different literals => literals are probably
//! expensive => short matches are probably worthwhile. Few => the opposite.
//!
//! # This is content-dependent, and it is NOT the thing CLAUDE.md clause 3 forbids
//!
//! Clause 3 bans env knobs and "detecting which corpus, cell, or archive type we are
//! in". This is neither: it is a fixed function of the data the encoder is already
//! reading, present in the vendor, with no branch on file identity or type. It is a
//! parse parameter, exactly like `max_search_depth`.

use super::codes::DeflateFreqs;
use super::{DEFLATE_MIN_MATCH_LEN, DEFLATE_NUM_LITERALS};

/// C: `choose_min_match_len(u32 num_used_literals, u32 max_search_depth)` (:2299)
///
/// The table maps a literal count to a minimum length. Its shape is the heuristic: 9
/// for a handful of distinct bytes, falling to 3 once ~80 distinct literals are in
/// play. The C notes "the rest is implicitly 3", which the length check reproduces.
pub(crate) fn choose_min_match_len(num_used_literals: u32, max_search_depth: u32) -> u32 {
    // map from num_used_literals to min_len
    #[rustfmt::skip]
    const MIN_LENS: [u8; 80] = [
        9, 9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        /* The rest is implicitly 3. */
    ];

    const _: () = assert!(DEFLATE_MIN_MATCH_LEN <= 3);
    const _: () = assert!(MIN_LENS.len() <= DEFLATE_NUM_LITERALS + 1);

    if num_used_literals as usize >= MIN_LENS.len() {
        return 3;
    }
    let mut min_len = MIN_LENS[num_used_literals as usize] as u32;

    // With a low max_search_depth, it may be too hard to find long matches.
    if max_search_depth < 16 {
        if max_search_depth < 5 {
            min_len = core::cmp::min(min_len, 4);
        } else if max_search_depth < 10 {
            min_len = core::cmp::min(min_len, 5);
        } else {
            min_len = core::cmp::min(min_len, 7);
        }
    }
    min_len
}

/// C: `calculate_min_match_len(const u8 *data, size_t data_len, u32 max_search_depth)`
/// (:2330)
///
/// # Two thresholds that are easy to get wrong, both pinned by tests
///
/// * **`data_len < 512` returns 3 immediately.** For very short inputs the static
///   Huffman code has a good chance of being best, in which case there is no reason to
///   avoid short matches.
/// * **Only the first 4 KiB is scanned**, as an initial approximation.
///   `recalculate_min_match_len` updates it later from the block's real frequencies.
///   Scanning the whole block instead would be a different heuristic — and would be an
///   instance of the trap recorded in `feedback_count_the_shipped_quantity`, where a
///   whole-file literal count "validated" a rule the code applies per block.
pub(crate) fn calculate_min_match_len(data: &[u8], data_len: usize, max_search_depth: u32) -> u32 {
    let mut used = [0u8; 256];
    let mut num_used_literals: u32 = 0;

    if data_len < 512 {
        return DEFLATE_MIN_MATCH_LEN;
    }

    let data_len = core::cmp::min(data_len, 4096);
    for i in 0..data_len {
        used[data[i] as usize] = 1;
    }
    for i in 0..256 {
        num_used_literals += used[i] as u32;
    }
    choose_min_match_len(num_used_literals, max_search_depth)
}

/// C: `recalculate_min_match_len(const struct deflate_freqs *freqs,
/// u32 max_search_depth)` (:2360)
///
/// Recalculate the minimum match length for a block, now that we know the distribution
/// of literals that are actually being used.
///
/// The cutoff is `literal_freq >> 10` — literals used less than ~0.1% of the time are
/// ignored, so a single stray byte does not make the alphabet look wide. Note the
/// comparison is strictly `>`, so with a total frequency under 1024 the cutoff is 0 and
/// every literal that occurs at all counts.
pub(crate) fn recalculate_min_match_len(freqs: &DeflateFreqs, max_search_depth: u32) -> u32 {
    let mut literal_freq: u32 = 0;
    let mut num_used_literals: u32 = 0;

    for i in 0..DEFLATE_NUM_LITERALS {
        literal_freq += freqs.litlen[i];
    }

    let cutoff = literal_freq >> 10; // Ignore literals used very rarely.

    for i in 0..DEFLATE_NUM_LITERALS {
        if freqs.litlen[i] > cutoff {
            num_used_literals += 1;
        }
    }
    choose_min_match_len(num_used_literals, max_search_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's shape IS the heuristic: monotonically non-increasing in the literal
    /// count, 9 at the bottom, 3 past the table. A transposed entry would be invisible
    /// in a round-trip test and would change every L2-L9 output.
    #[test]
    fn min_len_falls_monotonically_with_literal_variety() {
        let deep = 64; // above 16, so the depth clamp never fires
        let mut prev = choose_min_match_len(0, deep);
        assert_eq!(prev, 9, "a 1-literal alphabet must demand long matches");
        for n in 1..300u32 {
            let cur = choose_min_match_len(n, deep);
            assert!(
                cur <= prev,
                "min_len rose from {prev} to {cur} at {n} literals"
            );
            assert!((3..=9).contains(&cur), "min_len {cur} out of range at {n}");
            prev = cur;
        }
        assert_eq!(
            choose_min_match_len(80, deep),
            3,
            "past the table is implicitly 3"
        );
        assert_eq!(choose_min_match_len(255, deep), 3);
    }

    /// The shallow-depth clamp only LOWERS min_len, and only below depth 16. Its three
    /// bands are 5/10/16, and getting a boundary wrong changes L2 and L3 — the two
    /// levels this parser owns with `max_search_depth` 6 and 12.
    #[test]
    fn the_shallow_depth_clamp_only_lowers_and_bands_at_5_10_16() {
        for n in 0..90u32 {
            let unclamped = choose_min_match_len(n, 64);
            for depth in 1..20u32 {
                let got = choose_min_match_len(n, depth);
                assert!(
                    got <= unclamped,
                    "n={n} depth={depth}: clamp raised min_len"
                );
            }
        }
        // A literal count that maps to 9, so every band is visible.
        assert_eq!(choose_min_match_len(0, 4), 4, "depth < 5");
        assert_eq!(choose_min_match_len(0, 5), 5, "5 <= depth < 10");
        assert_eq!(choose_min_match_len(0, 9), 5);
        assert_eq!(choose_min_match_len(0, 10), 7, "10 <= depth < 16");
        assert_eq!(choose_min_match_len(0, 15), 7);
        assert_eq!(choose_min_match_len(0, 16), 9, "depth >= 16: no clamp");

        // libdeflate's own L2 (depth 6) and L3 (depth 12).
        assert_eq!(choose_min_match_len(0, 6), 5);
        assert_eq!(choose_min_match_len(0, 12), 7);
    }

    /// Under 512 bytes the answer is 3 regardless of content; at 512 the scan runs.
    #[test]
    fn short_inputs_always_allow_length_3_matches() {
        let uniform = vec![b'q'; 4096];
        assert_eq!(calculate_min_match_len(&uniform, 511, 64), 3);
        // At 512 the same one-literal data now demands long matches.
        assert_eq!(calculate_min_match_len(&uniform, 512, 64), 9);
    }

    /// Only the first 4 KiB is scanned. Data that is uniform for 4 KiB and then varied
    /// must be judged on the uniform prefix alone.
    #[test]
    fn only_the_first_4_kib_is_scanned() {
        let mut data = vec![b'z'; 4096];
        data.extend((0..=255u8).cycle().take(4096));
        assert_eq!(
            calculate_min_match_len(&data, data.len(), 64),
            9,
            "the varied tail past 4 KiB must not be counted"
        );

        // Move the variety inside the window and the answer changes.
        let mut data2: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        data2.extend(vec![b'z'; 4096]);
        assert_eq!(calculate_min_match_len(&data2, data2.len(), 64), 3);
    }

    /// `recalculate` ignores literals below `total >> 10`, so a stray byte cannot widen
    /// the alphabet. The comparison is strictly `>`, so a literal AT the cutoff is
    /// excluded too.
    #[test]
    fn recalculate_ignores_rare_literals() {
        let mut freqs = DeflateFreqs::new();
        // 200 literals that each appear once, plus one dominant literal. Order matters:
        // the first draft set `litlen[b'a']` FIRST and then clobbered it in the loop
        // (b'a' is 97, inside 0..200), so the "dominant" literal had frequency 1 and
        // the test measured nothing.
        for i in 0..200 {
            freqs.litlen[i] = 1;
        }
        freqs.litlen[250] = 100_000;
        // total ~100,200 -> cutoff 97. Only 'a' clears it.
        assert_eq!(
            recalculate_min_match_len(&freqs, 64),
            9,
            "199 single-occurrence literals must not widen the alphabet"
        );

        // Give them real weight and the answer flips.
        let mut freqs2 = DeflateFreqs::new();
        for i in 0..200 {
            freqs2.litlen[i] = 500;
        }
        assert_eq!(recalculate_min_match_len(&freqs2, 64), 3);
    }

    /// With a total frequency under 1024 the cutoff is 0, so every literal that occurs
    /// at all is counted. The `>` (not `>=`) is what makes that true.
    #[test]
    fn a_small_block_counts_every_literal_that_occurs() {
        let mut freqs = DeflateFreqs::new();
        for i in 0..100 {
            freqs.litlen[i] = 1;
        }
        assert_eq!(
            recalculate_min_match_len(&freqs, 64),
            3,
            "cutoff is 0 here, so all 100 literals count"
        );
    }
}
