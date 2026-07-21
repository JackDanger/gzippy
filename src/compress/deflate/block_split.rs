//! Block-splitting statistic (observation-count SAD heuristic).
//!
//! Port of libdeflate `vendor/libdeflate/lib/deflate_compress.c`
//! `block_split_stats` + `init_block_split_stats` / `observe_literal` /
//! `observe_match` / `merge_new_observations` / `do_end_block_check` /
//! `ready_to_check_block` / `should_end_block` (~:2091-2218).
//!
//! It aggregates symbols into a handful of "observation types" (top-2 + low-1
//! bits of a literal; short vs long match), and starts a new block when the sum
//! of absolute differences between the running distribution and the recent
//! distribution grows past a length-scaled cutoff. Landed now with unit tests;
//! wired into the parser in a later increment.

pub const NUM_LITERAL_OBSERVATION_TYPES: usize = 8;
pub const NUM_MATCH_OBSERVATION_TYPES: usize = 2;
pub const NUM_OBSERVATION_TYPES: usize =
    NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES;
pub const NUM_OBSERVATIONS_PER_BLOCK_CHECK: u32 = 512;

/// Minimum input bytes before/around a candidate split (`MIN_BLOCK_LENGTH`).
pub const MIN_BLOCK_LENGTH: usize = 5000;

#[derive(Clone)]
pub struct BlockSplitStats {
    new_observations: [u32; NUM_OBSERVATION_TYPES],
    observations: [u32; NUM_OBSERVATION_TYPES],
    num_new_observations: u32,
    num_observations: u32,
}

impl Default for BlockSplitStats {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockSplitStats {
    /// `init_block_split_stats`.
    pub fn new() -> Self {
        BlockSplitStats {
            new_observations: [0; NUM_OBSERVATION_TYPES],
            observations: [0; NUM_OBSERVATION_TYPES],
            num_new_observations: 0,
            num_observations: 0,
        }
    }

    /// Reset for a new block.
    pub fn reset(&mut self) {
        *self = BlockSplitStats::new();
    }

    /// `observe_literal`: top-2 bits + low-1 bit => one of 8 literal types.
    #[inline]
    pub fn observe_literal(&mut self, lit: u8) {
        let idx = (((lit >> 5) & 0x6) | (lit & 1)) as usize;
        self.new_observations[idx] += 1;
        self.num_new_observations += 1;
    }

    /// `observe_match`: short (<9) vs long (>=9) match.
    #[inline]
    pub fn observe_match(&mut self, length: u32) {
        let idx = NUM_LITERAL_OBSERVATION_TYPES + (length >= 9) as usize;
        self.new_observations[idx] += 1;
        self.num_new_observations += 1;
    }

    /// Running (merged) observation counts, for the cross-block cost blend.
    #[inline]
    pub fn observations(&self) -> &[u32; NUM_OBSERVATION_TYPES] {
        &self.observations
    }

    /// Number of merged observations.
    #[inline]
    pub fn num_observations(&self) -> u32 {
        self.num_observations
    }

    /// Zero the running (merged) observations, keeping the not-yet-merged
    /// `new_observations` (`deflate_near_optimal_clear_old_stats`, obs part).
    #[inline]
    pub fn clear_old_observations(&mut self) {
        self.observations = [0; NUM_OBSERVATION_TYPES];
        self.num_observations = 0;
    }

    /// `merge_new_observations`: fold the recent window into the running totals.
    pub fn merge_new_observations(&mut self) {
        for i in 0..NUM_OBSERVATION_TYPES {
            self.observations[i] += self.new_observations[i];
            self.new_observations[i] = 0;
        }
        self.num_observations += self.num_new_observations;
        self.num_new_observations = 0;
    }

    /// `do_end_block_check`: SAD-of-probabilities test. Returns true (end the
    /// block) or merges the recent window and returns false. `block_length` is
    /// the number of input bytes in the current block so far.
    pub fn do_end_block_check(&mut self, block_length: u32) -> bool {
        if self.num_observations > 0 {
            let mut total_delta: u32 = 0;
            for i in 0..NUM_OBSERVATION_TYPES {
                let expected = self.observations[i].wrapping_mul(self.num_new_observations);
                let actual = self.new_observations[i].wrapping_mul(self.num_observations);
                let delta = actual.abs_diff(expected);
                total_delta = total_delta.wrapping_add(delta);
            }

            let num_items = self.num_observations + self.num_new_observations;
            // cutoff = num_new_observations * 200/512 * num_observations.
            let mut cutoff: u64 =
                self.num_new_observations as u64 * 200 / 512 * self.num_observations as u64;
            if block_length < 10000 && num_items < 8192 {
                cutoff += cutoff * (8192 - num_items as u64) / 8192;
            }

            if total_delta as u64 + (block_length as u64 / 4096) * self.num_observations as u64
                >= cutoff
            {
                return true;
            }
        }
        self.merge_new_observations();
        false
    }

    /// `ready_to_check_block`: enough recent observations and enough surrounding
    /// input to justify testing a split.
    #[inline]
    pub fn ready_to_check_block(&self, bytes_in_block: usize, bytes_remaining: usize) -> bool {
        self.num_new_observations >= NUM_OBSERVATIONS_PER_BLOCK_CHECK
            && bytes_in_block >= MIN_BLOCK_LENGTH
            && bytes_remaining >= MIN_BLOCK_LENGTH
    }

    /// `should_end_block`.
    pub fn should_end_block(&mut self, bytes_in_block: usize, bytes_remaining: usize) -> bool {
        if !self.ready_to_check_block(bytes_in_block, bytes_remaining) {
            return false;
        }
        self.do_end_block_check(bytes_in_block as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_observation_bucketing() {
        let mut s = BlockSplitStats::new();
        // lit=0 => ((0>>5)&6)|(0&1) = 0
        s.observe_literal(0);
        // lit=0xFF => ((0xFF>>5)&6)|(1) = (7&6)|1 = 6|1 = 7
        s.observe_literal(0xFF);
        assert_eq!(s.num_new_observations, 2);
        assert_eq!(s.new_observations[0], 1);
        assert_eq!(s.new_observations[7], 1);
    }

    #[test]
    fn match_observation_short_long() {
        let mut s = BlockSplitStats::new();
        s.observe_match(3); // short
        s.observe_match(200); // long
        assert_eq!(s.new_observations[NUM_LITERAL_OBSERVATION_TYPES], 1);
        assert_eq!(s.new_observations[NUM_LITERAL_OBSERVATION_TYPES + 1], 1);
    }

    #[test]
    fn homogeneous_stream_does_not_split() {
        // A perfectly stationary distribution should never trigger a split:
        // expected == actual for every type, so total_delta stays 0.
        let mut s = BlockSplitStats::new();
        // Establish a baseline, then keep feeding the identical mix.
        for _ in 0..(NUM_OBSERVATIONS_PER_BLOCK_CHECK) {
            s.observe_literal(b'a');
        }
        // First check merges the baseline.
        assert!(!s.do_end_block_check(6000));
        for _ in 0..(NUM_OBSERVATIONS_PER_BLOCK_CHECK) {
            s.observe_literal(b'a');
        }
        // Same distribution => total_delta 0 => no split.
        assert!(!s.do_end_block_check(6000));
    }

    #[test]
    fn distribution_shift_triggers_split() {
        let mut s = BlockSplitStats::new();
        // Baseline: all one literal bucket.
        for _ in 0..NUM_OBSERVATIONS_PER_BLOCK_CHECK {
            s.observe_literal(0x00); // bucket 0
        }
        assert!(!s.do_end_block_check(20000)); // merges baseline
                                               // Sharp change: all a very different bucket.
        for _ in 0..NUM_OBSERVATIONS_PER_BLOCK_CHECK {
            s.observe_literal(0xFF); // bucket 7
        }
        // Large block_length keeps the small-block penalty off; the maximal SAD
        // should exceed the cutoff.
        assert!(s.do_end_block_check(20000));
    }

    #[test]
    fn not_ready_without_enough_observations() {
        let s = BlockSplitStats::new();
        assert!(!s.ready_to_check_block(100_000, 100_000));
    }
}
