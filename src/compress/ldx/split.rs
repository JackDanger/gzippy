//! C: `vendor/libdeflate/lib/deflate_compress.c:2055-2222` — the block splitting
//! algorithm.
//!
//! The problem is to decide when it is worthwhile to start a new block with new
//! Huffman codes. There is a theoretically optimal solution: recursively consider
//! every possible block split, considering the exact cost of each block, and choose
//! the minimum cost approach. But this is far too slow. Instead, as an approximation,
//! we can count symbols and after every N symbols, compare the expected distribution
//! of symbols based on the previous data with the actual distribution. If they differ
//! "by enough", then start a new block.
//!
//! As an optimization and heuristic, we don't distinguish between every symbol but
//! rather we combine many symbols into a single "observation type". For literals we
//! only look at the high bits and low bits, and for matches we only look at whether
//! the match is long or not. The assumption is that for typical "real" data, places
//! that are good block boundaries will tend to be noticeable based only on changes in
//! these aggregate probabilities, without looking for subtle differences in
//! individual symbols. For example, a change from ASCII bytes to non-ASCII bytes, or
//! from few matches (generally less compressible) to many matches (generally more
//! compressible), would be easily noticed based on the aggregates.
//!
//! For determining whether the probability distributions are "different enough" to
//! start a new block, the simple heuristic of splitting when the sum of absolute
//! differences exceeds a constant seems to be good enough. We also add a number
//! proportional to the block length so that the algorithm is more likely to end long
//! blocks than short blocks. This reflects the general expectation that it will
//! become increasingly beneficial to start a new block as the current block grows
//! longer.
//!
//! Finally, for an approximation, it is not strictly necessary that the exact symbols
//! being used are considered. With "near-optimal parsing", for example, the actual
//! symbols that will be used are unknown until after the block boundary is chosen and
//! the block has been optimized. Since the final choices cannot be used, we can use
//! preliminary "greedy" choices instead.
//!
//! # Why every arithmetic width here is load-bearing
//!
//! `examples/blockspans` measured that gzip splits on a fixed ~34,000-symbol cadence
//! (cv=0.023) while our spans run 8x longer at cv=0.373 — so where the boundaries
//! fall is a live question for the campaign, and this heuristic is what answers it.
//! `do_end_block_check` mixes `u32` wrapping arithmetic with ONE deliberate `u64`
//! promotion, and widening the rest "for safety" changes where blocks end.
//! `PORT_STATUS.md` records that our shipping `block_split.rs:192-200` computes the
//! cutoff in `u64` where the C uses `u32`; this module is the C's widths.

use super::{MIN_BLOCK_LENGTH, NUM_OBSERVATIONS_PER_BLOCK_CHECK};

/// C: `#define NUM_LITERAL_OBSERVATION_TYPES 8` (:440)
pub(crate) const NUM_LITERAL_OBSERVATION_TYPES: usize = 8;
/// C: `#define NUM_MATCH_OBSERVATION_TYPES 2` (:441)
pub(crate) const NUM_MATCH_OBSERVATION_TYPES: usize = 2;
/// C: `#define NUM_OBSERVATION_TYPES ...` (:442)
pub(crate) const NUM_OBSERVATION_TYPES: usize =
    NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES;

/// C: `struct block_split_stats` (:445)
#[derive(Clone)]
pub(crate) struct BlockSplitStats {
    pub(crate) new_observations: [u32; NUM_OBSERVATION_TYPES],
    pub(crate) observations: [u32; NUM_OBSERVATION_TYPES],
    pub(crate) num_new_observations: u32,
    pub(crate) num_observations: u32,
}

impl BlockSplitStats {
    pub(crate) const fn new() -> Self {
        Self {
            new_observations: [0; NUM_OBSERVATION_TYPES],
            observations: [0; NUM_OBSERVATION_TYPES],
            num_new_observations: 0,
            num_observations: 0,
        }
    }
}

/// C: `init_block_split_stats(struct block_split_stats *stats)` (:2104)
///
/// Initialize the block split statistics when starting a new block.
pub(crate) fn init_block_split_stats(stats: &mut BlockSplitStats) {
    for i in 0..NUM_OBSERVATION_TYPES {
        stats.new_observations[i] = 0;
        stats.observations[i] = 0;
    }
    stats.num_new_observations = 0;
    stats.num_observations = 0;
}

/// C: `observe_literal(struct block_split_stats *stats, u8 lit)` (:2120)
///
/// Literal observation. Heuristic: use the top 2 bits and low 1 bits of the literal,
/// for 8 possible literal observation types.
///
/// `((lit >> 5) & 0x6) | (lit & 1)` is not a typo for `>> 6`: shifting by 5 and
/// masking with `0x6` puts bits 6 and 7 into positions 1 and 2, leaving position 0
/// for the low bit. So the type is `(bit7, bit6, bit0)` — which is what makes
/// "ASCII vs non-ASCII" a visible aggregate.
#[inline(always)]
pub(crate) fn observe_literal(stats: &mut BlockSplitStats, lit: u8) {
    crate::anatomy_count!(block_split_observations);
    stats.new_observations[(((lit >> 5) & 0x6) | (lit & 1)) as usize] += 1;
    stats.num_new_observations += 1;
}

/// C: `observe_match(struct block_split_stats *stats, u32 length)` (:2131)
///
/// Match observation. Heuristic: use one observation type for "short match" and one
/// observation type for "long match".
#[inline(always)]
pub(crate) fn observe_match(stats: &mut BlockSplitStats, length: u32) {
    crate::anatomy_count!(block_split_observations);
    stats.new_observations[NUM_LITERAL_OBSERVATION_TYPES + (length >= 9) as usize] += 1;
    stats.num_new_observations += 1;
}

/// C: `merge_new_observations(struct block_split_stats *stats)` (:2139)
pub(crate) fn merge_new_observations(stats: &mut BlockSplitStats) {
    for i in 0..NUM_OBSERVATION_TYPES {
        stats.observations[i] += stats.new_observations[i];
        stats.new_observations[i] = 0;
    }
    stats.num_observations += stats.num_new_observations;
    stats.num_new_observations = 0;
}

/// C: `do_end_block_check(struct block_split_stats *stats, u32 block_length)` (:2153)
///
/// # The arithmetic, and why it is spelled with `wrapping_*`
///
/// Every intermediate here is `u32` in the C and therefore wraps rather than trapping.
/// With real inputs nothing actually wraps — `num_new_observations` is ~512 at a
/// check, `observations[i]` is bounded by the block's symbol count (~3e5), so the
/// products reach ~1.5e8 and `total_delta` ~1.5e9, all inside `u32` — but the port
/// must not PANIC where the C would compute. `wrapping_*` says "same value as C, no
/// trap", which is the contract; plain `+` would make a debug build diverge from a
/// release build on an input the C handles silently.
///
/// The ONE place the C deliberately widens is the short-block penalty:
/// `cutoff += (u64)cutoff * (8192 - num_items) / 8192`. That product genuinely
/// overflows `u32` (6e7 * 8192 ~ 5e11), which is why the cast is there and why it is
/// on the multiply and not on `cutoff` itself — the sum is then truncated back to
/// `u32` by the `+=`. Promoting `cutoff` to `u64` throughout would keep bits the C
/// discards and move block boundaries.
pub(crate) fn do_end_block_check(stats: &mut BlockSplitStats, block_length: u32) -> bool {
    if stats.num_observations > 0 {
        // Compute the sum of absolute differences of probabilities. To avoid needing
        // to use floating point arithmetic or do slow divisions, we do all arithmetic
        // with the probabilities multiplied by num_observations *
        // num_new_observations. E.g., for the "old" observations the probabilities
        // would be (double)observations[i] / num_observations, but since we multiply
        // by both num_observations and num_new_observations we really do
        // observations[i] * num_new_observations.
        let mut total_delta: u32 = 0;

        for i in 0..NUM_OBSERVATION_TYPES {
            let expected = stats.observations[i].wrapping_mul(stats.num_new_observations);
            let actual = stats.new_observations[i].wrapping_mul(stats.num_observations);
            // C: `u32 delta = (actual > expected) ? actual - expected
            //                                     : expected - actual;`
            // Clippy suggests `actual.abs_diff(expected)`. Same value, but this
            // module's contract is that it diffs against the C line for line, and the
            // ternary is one of them.
            #[allow(clippy::manual_abs_diff, reason = "C: the ternary is the source line")]
            let delta = if actual > expected {
                actual - expected
            } else {
                expected - actual
            };

            total_delta = total_delta.wrapping_add(delta);
        }

        let num_items = stats
            .num_observations
            .wrapping_add(stats.num_new_observations);

        // Heuristic: the cutoff is when the sum of absolute differences of
        // probabilities becomes at least 200/512. As above, the probability is
        // multiplied by both num_new_observations and num_observations. Be careful to
        // avoid integer overflow.
        //
        // The `/ 512` sits BETWEEN the two multiplies, not after both. Moving it
        // changes the value by the truncation it performs, so the grouping is part of
        // the heuristic, not an optimisation.
        let mut cutoff = stats
            .num_new_observations
            .wrapping_mul(200)
            .wrapping_div(512)
            .wrapping_mul(stats.num_observations);

        // Very short blocks have a lot of overhead for the Huffman codes, so only use
        // them if it clearly seems worthwhile. (This is an additional penalty, which
        // adds to the smaller penalty below which scales more slowly.)
        if block_length < 10000 && num_items < 8192 {
            cutoff = (cutoff as u64)
                .wrapping_add((cutoff as u64) * (8192 - num_items) as u64 / 8192)
                as u32;
        }

        // Ready to end the block?
        if total_delta.wrapping_add((block_length / 4096).wrapping_mul(stats.num_observations))
            >= cutoff
        {
            return true;
        }
    }
    merge_new_observations(stats);
    false
}

/// C: `ready_to_check_block(...)` (:2200)
///
/// The C compares pointers; we compare byte positions in the same input. Identical
/// values, since `in_block_begin <= in_next <= in_end` always holds.
#[inline(always)]
pub(crate) fn ready_to_check_block(
    stats: &BlockSplitStats,
    in_block_begin: usize,
    in_next: usize,
    in_end: usize,
) -> bool {
    stats.num_new_observations >= NUM_OBSERVATIONS_PER_BLOCK_CHECK
        && in_next - in_block_begin >= MIN_BLOCK_LENGTH
        && in_end - in_next >= MIN_BLOCK_LENGTH
}

/// C: `should_end_block(...)` (:2210)
#[inline(always)]
pub(crate) fn should_end_block(
    stats: &mut BlockSplitStats,
    in_block_begin: usize,
    in_next: usize,
    in_end: usize,
) -> bool {
    // Ready to try to end the block (again)?
    if !ready_to_check_block(stats, in_block_begin, in_next, in_end) {
        return false;
    }

    do_end_block_check(stats, (in_next - in_block_begin) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal observation type is `(bit7, bit6, bit0)`, so ASCII and non-ASCII
    /// land in disjoint halves of the table. That separation is the whole reason the
    /// heuristic notices a text-to-binary transition, so pin it.
    #[test]
    fn literal_observation_type_separates_ascii_from_high_bytes() {
        let mut stats = BlockSplitStats::new();

        // 'a' = 0x61: bit7=0, bit6=1, bit0=1 -> ((0x61>>5)&6)|1 = (3&6)|1 = 2|1 = 3.
        observe_literal(&mut stats, b'a');
        assert_eq!(stats.new_observations[3], 1);

        // 0xE1: bit7=1, bit6=1, bit0=1 -> ((0xE1>>5)&6)|1 = (7&6)|1 = 6|1 = 7.
        observe_literal(&mut stats, 0xE1);
        assert_eq!(stats.new_observations[7], 1);

        // Every printable ASCII byte must land in types 0..=3; every byte with the
        // high bit set must land in 4..=7.
        for b in 0u8..=127 {
            let t = ((b >> 5) & 0x6) | (b & 1);
            assert!(t < 4, "ASCII {b} landed in type {t}");
        }
        for b in 128u8..=255 {
            let t = ((b >> 5) & 0x6) | (b & 1);
            assert!((4..8).contains(&t), "high byte {b} landed in type {t}");
        }
    }

    /// Match observations use exactly two types, split at length 9.
    #[test]
    fn match_observation_splits_at_length_nine() {
        let mut stats = BlockSplitStats::new();
        observe_match(&mut stats, 8);
        observe_match(&mut stats, 9);
        assert_eq!(stats.new_observations[NUM_LITERAL_OBSERVATION_TYPES], 1);
        assert_eq!(stats.new_observations[NUM_LITERAL_OBSERVATION_TYPES + 1], 1);
        assert_eq!(stats.num_new_observations, 2);
    }

    /// A stationary distribution must NOT split: the new observations match the old
    /// ones, so `total_delta` stays near zero.
    #[test]
    fn a_stationary_distribution_does_not_split() {
        let mut stats = BlockSplitStats::new();

        // Prime the "old" distribution with 4096 identical-looking literals.
        for _ in 0..4096 {
            observe_literal(&mut stats, b'e');
        }
        merge_new_observations(&mut stats);

        // Now the same distribution again.
        for _ in 0..NUM_OBSERVATIONS_PER_BLOCK_CHECK {
            observe_literal(&mut stats, b'e');
        }
        assert!(
            !do_end_block_check(&mut stats, 20_000),
            "identical distributions must not trigger a split"
        );
        // The check merged instead of splitting.
        assert_eq!(stats.num_new_observations, 0);
        assert_eq!(stats.num_observations, 4096 + 512);
    }

    /// A hard distribution change MUST split: prime with ASCII, then feed high bytes.
    #[test]
    fn a_distribution_change_splits() {
        let mut stats = BlockSplitStats::new();

        for _ in 0..4096 {
            observe_literal(&mut stats, b'e');
        }
        merge_new_observations(&mut stats);

        for _ in 0..NUM_OBSERVATIONS_PER_BLOCK_CHECK {
            observe_literal(&mut stats, 0xF3);
        }
        assert!(
            do_end_block_check(&mut stats, 20_000),
            "ASCII -> high bytes must trigger a split"
        );
    }

    /// The short-block penalty must FLIP a real decision, and the numbers are chosen
    /// so that it does. Everything else is held equal: `block_length / 4096` is 2 for
    /// both 9,000 and 10,000, so the ONLY difference between the two calls is whether
    /// `block_length < 10000` admits the penalty.
    ///
    /// With `observations = [1000, 0, ...]`, `new_observations = [400, 112, 0, ...]`:
    ///
    /// ```text
    ///   total_delta = |400*1000 - 1000*512| + |112*1000 - 0|  = 112,000 + 112,000
    ///               = 224,000
    ///   left        = 224,000 + (block_length/4096) * 1000     = 226,000
    ///   cutoff      = 512 * 200 / 512 * 1000                   = 200,000
    ///   penalised   = 200,000 + 200,000 * (8192-1512) / 8192   = 363,085
    /// ```
    ///
    /// 226,000 clears 200,000 but not 363,085 — so the long block splits and the short
    /// one does not. A width mistake in the `u64` promotion (or dropping the penalty)
    /// changes `penalised` and this test fails.
    #[test]
    fn the_short_block_penalty_flips_a_real_decision() {
        let build = || {
            let mut stats = BlockSplitStats::new();
            stats.observations[0] = 1000;
            stats.num_observations = 1000;
            stats.new_observations[0] = 400;
            stats.new_observations[1] = 112;
            stats.num_new_observations = 512;
            stats
        };

        let mut long = build();
        assert!(
            do_end_block_check(&mut long, 10_000),
            "10,000 is not penalised, and 226,000 >= 200,000, so it must split"
        );

        let mut short = build();
        assert!(
            !do_end_block_check(&mut short, 9_000),
            "9,000 is penalised to 363,085, and 226,000 < that, so it must NOT split"
        );
        // A non-split must have merged the sample.
        assert_eq!(short.num_new_observations, 0);
        assert_eq!(short.num_observations, 1512);
    }

    /// The penalty can only ever RAISE the cutoff, so it can never turn a non-split
    /// into a split. Checked across the whole `num_items` range it applies to.
    #[test]
    fn the_penalty_is_monotone_and_never_causes_a_split() {
        for num_obs in [1u32, 100, 1000, 4000, 7000, 7679] {
            let build = || {
                let mut stats = BlockSplitStats::new();
                stats.observations[0] = num_obs;
                stats.num_observations = num_obs;
                stats.new_observations[0] = 400;
                stats.new_observations[1] = 112;
                stats.num_new_observations = 512;
                stats
            };
            let mut penalised = build();
            let mut plain = build();
            let p = do_end_block_check(&mut penalised, 9_000);
            let q = do_end_block_check(&mut plain, 10_000);
            assert!(
                !(p && !q),
                "num_obs={num_obs}: the penalty made a split MORE likely"
            );
        }
    }

    /// `ready_to_check_block` gates on all three conditions independently.
    #[test]
    fn ready_to_check_requires_samples_and_room_on_both_sides() {
        let mut stats = BlockSplitStats::new();
        stats.num_new_observations = NUM_OBSERVATIONS_PER_BLOCK_CHECK;

        let big = MIN_BLOCK_LENGTH * 4;
        assert!(ready_to_check_block(&stats, 0, big, big * 2));

        // Not enough samples.
        stats.num_new_observations = NUM_OBSERVATIONS_PER_BLOCK_CHECK - 1;
        assert!(!ready_to_check_block(&stats, 0, big, big * 2));
        stats.num_new_observations = NUM_OBSERVATIONS_PER_BLOCK_CHECK;

        // Block so far is too short.
        assert!(!ready_to_check_block(
            &stats,
            0,
            MIN_BLOCK_LENGTH - 1,
            big * 2
        ));

        // Not enough input left to be worth a new block.
        assert!(!ready_to_check_block(
            &stats,
            0,
            big,
            big + MIN_BLOCK_LENGTH - 1
        ));
    }

    /// `init_block_split_stats` must clear both tables and both counters — a stale
    /// observation would leak the previous block's distribution into this one and
    /// move the next boundary.
    #[test]
    fn init_clears_everything() {
        let mut stats = BlockSplitStats::new();
        for b in 0u8..=255 {
            observe_literal(&mut stats, b);
        }
        merge_new_observations(&mut stats);
        observe_match(&mut stats, 100);

        init_block_split_stats(&mut stats);
        assert!(stats.observations.iter().all(|&o| o == 0));
        assert!(stats.new_observations.iter().all(|&o| o == 0));
        assert_eq!(stats.num_observations, 0);
        assert_eq!(stats.num_new_observations, 0);
    }
}
