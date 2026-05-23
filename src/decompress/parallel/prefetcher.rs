#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::FetchingStrategy`
//! (vendor/.../core/Prefetcher.hpp).
//!
//! Strategies for deciding which indexes to speculatively prefetch
//! ahead of the consumer. Three flavors:
//! - [`FetchNextFixed`]: prefetch the next N indexes after the last
//!   access. Used by tests and minimal configurations.
//! - [`FetchNextAdaptive`]: scale prefetch count by the consecutive-
//!   access ratio (the production default; exponential interpolation
//!   between 1 and `max_to_prefetch`).
//! - [`FetchMultiStream`]: extend `FetchNextAdaptive` with detection
//!   of multiple interleaved sequential access streams.

use std::collections::VecDeque;

/// Trait mirroring `FetchingStrategy::FetchingStrategy`
/// (Prefetcher.hpp:18-54).
pub trait FetchingStrategy {
    /// Record that index was accessed. Derived strategies override but
    /// must call this base behavior to update `last_fetched()`.
    fn fetch(&mut self, index: usize);

    fn last_fetched(&self) -> Option<usize>;

    /// Whether all of the memorized last accesses were sequential.
    /// True enables more aggressive prefetching downstream.
    fn is_sequential(&self) -> bool {
        false
    }

    /// Decide which indexes to prefetch next, up to
    /// `max_amount_to_prefetch`. Returned vector is in prefetch-order.
    fn prefetch(&self, max_amount_to_prefetch: usize) -> Vec<usize>;
}

/// Prefetch the next N indexes after the last access. Mirror of
/// `FetchNextFixed` (Prefetcher.hpp:60-74).
#[derive(Debug, Default)]
pub struct FetchNextFixed {
    last_fetched: Option<usize>,
}

impl FetchingStrategy for FetchNextFixed {
    fn fetch(&mut self, index: usize) {
        self.last_fetched = Some(index);
    }
    fn last_fetched(&self) -> Option<usize> {
        self.last_fetched
    }
    fn prefetch(&self, max_amount_to_prefetch: usize) -> Vec<usize> {
        match self.last_fetched {
            Some(last) => (last + 1..last + 1 + max_amount_to_prefetch).collect(),
            None => Vec::new(),
        }
    }
}

/// Scale prefetch count by the consecutive-access ratio. Mirror of
/// `FetchNextAdaptive` (Prefetcher.hpp:88-225).
#[derive(Debug)]
pub struct FetchNextAdaptive {
    last_fetched: Option<usize>,
    memory_size: usize,
    /// Most-recent at front (mirrors rapidgzip's `push_front`).
    previous_indexes: VecDeque<usize>,
}

impl FetchNextAdaptive {
    pub fn new(memory_size: usize) -> Self {
        Self {
            last_fetched: None,
            memory_size,
            previous_indexes: VecDeque::new(),
        }
    }

    pub fn memory_size(&self) -> usize {
        self.memory_size
    }

    pub fn previous_indexes(&self) -> &VecDeque<usize> {
        &self.previous_indexes
    }

    /// Per `FetchNextAdaptive::extrapolateForward` (Prefetcher.hpp:126-146).
    /// Exponential interpolation: amount = 2^(consecutive_ratio * log2(max_extrapolation)).
    pub fn extrapolate_forward(
        highest_value: usize,
        consecutive_values: usize,
        saturation_count: usize,
        max_extrapolation: usize,
    ) -> Vec<usize> {
        if max_extrapolation == 0 {
            return Vec::new();
        }
        let consecutive_ratio = if saturation_count == 0 {
            1.0
        } else {
            consecutive_values.min(saturation_count) as f64 / saturation_count as f64
        };
        let amount = (consecutive_ratio * (max_extrapolation as f64).log2())
            .exp2()
            .round();
        let amount = amount.max(0.0) as usize;
        let amount = amount.min(max_extrapolation);
        (highest_value + 1..highest_value + 1 + amount).collect()
    }

    /// Per `FetchNextAdaptive::extrapolate` (Prefetcher.hpp:148-185).
    /// `previous` is most-recent-at-front (matches `previous_indexes`).
    pub fn extrapolate(previous: &VecDeque<usize>, max_amount_to_prefetch: usize) -> Vec<usize> {
        let size = previous.len();
        if size == 0 || max_amount_to_prefetch == 0 {
            return Vec::new();
        }
        if size == 1 {
            let highest = *previous.front().unwrap();
            return (highest + 1..highest + 1 + max_amount_to_prefetch).collect();
        }
        // Count adjacent decreasing pairs (most recent at left).
        let mut any_adjacent = false;
        for (a, b) in previous.iter().zip(previous.iter().skip(1)) {
            if *a == *b + 1 {
                any_adjacent = true;
                break;
            }
        }
        if !any_adjacent {
            return Vec::new();
        }
        let mut last_consecutive_count: usize = 0;
        for (a, b) in previous.iter().zip(previous.iter().skip(1)) {
            if *a == *b + 1 {
                last_consecutive_count = if last_consecutive_count == 0 {
                    2
                } else {
                    last_consecutive_count + 1
                };
            } else {
                break;
            }
        }
        let highest = *previous.front().unwrap();
        Self::extrapolate_forward(
            highest,
            last_consecutive_count,
            size,
            max_amount_to_prefetch,
        )
    }

    /// Per `FetchNextAdaptive::splitIndex` (Prefetcher.hpp:197-219).
    /// Reindex `previous_indexes` such that one index is interpreted
    /// as `split_count` separate sequential indexes.
    pub fn split_index(&mut self, index_to_split: usize, split_count: usize) {
        if split_count <= 1 {
            return;
        }
        let mut new_indexes = VecDeque::new();
        for &index in &self.previous_indexes {
            if index == index_to_split {
                for i in 0..split_count {
                    new_indexes.push_back(index + split_count - 1 - i);
                }
            } else if index > index_to_split {
                new_indexes.push_back(index + split_count - 1);
            } else {
                new_indexes.push_back(index);
            }
        }
        self.previous_indexes = new_indexes;
    }
}

impl FetchingStrategy for FetchNextAdaptive {
    fn fetch(&mut self, index: usize) {
        self.last_fetched = Some(index);
        // Ignore duplicate accesses (rapidgzip Prefetcher.hpp:104-106).
        if self.previous_indexes.front() == Some(&index) {
            return;
        }
        self.previous_indexes.push_front(index);
        while self.previous_indexes.len() > self.memory_size {
            self.previous_indexes.pop_back();
        }
    }

    fn last_fetched(&self) -> Option<usize> {
        self.last_fetched
    }

    fn is_sequential(&self) -> bool {
        // True for empty / single-element memory (rapidgzip's choice).
        for (a, b) in self
            .previous_indexes
            .iter()
            .zip(self.previous_indexes.iter().skip(1))
        {
            if *b + 1 != *a {
                return false;
            }
        }
        true
    }

    fn prefetch(&self, max_amount_to_prefetch: usize) -> Vec<usize> {
        Self::extrapolate(&self.previous_indexes, max_amount_to_prefetch)
    }
}

/// Round-robin merge of same-index elements across subsequences.
/// Mirror of `interleave` (vendor/.../core/common.hpp:496-512).
fn interleave(subsequences: &[Vec<usize>]) -> Vec<usize> {
    let total: usize = subsequences.iter().map(|s| s.len()).sum();
    let max_len = subsequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = Vec::with_capacity(total);
    for i in 0..max_len {
        for sub in subsequences {
            if i < sub.len() {
                result.push(sub[i]);
            }
        }
    }
    result
}

/// Extend [`FetchNextAdaptive`] with detection of multiple interleaved
/// sequential access streams. Mirror of `FetchMultiStream`
/// (Prefetcher.hpp:234-336).
#[derive(Debug)]
pub struct FetchMultiStream {
    adaptive: FetchNextAdaptive,
    memory_size_per_stream: usize,
}

impl FetchMultiStream {
    pub fn new(memory_size_per_stream: usize, max_stream_count: usize) -> Self {
        Self {
            adaptive: FetchNextAdaptive::new(max_stream_count * memory_size_per_stream),
            memory_size_per_stream,
        }
    }

    pub fn memory_size_per_stream(&self) -> usize {
        self.memory_size_per_stream
    }

    fn memory_full(&self) -> bool {
        self.adaptive.previous_indexes().len() >= self.adaptive.memory_size()
    }
}

impl FetchingStrategy for FetchMultiStream {
    fn fetch(&mut self, index: usize) {
        self.adaptive.fetch(index);
    }

    fn last_fetched(&self) -> Option<usize> {
        self.adaptive.last_fetched()
    }

    fn is_sequential(&self) -> bool {
        self.adaptive.is_sequential()
    }

    fn prefetch(&self, max_amount_to_prefetch: usize) -> Vec<usize> {
        let previous_indexes = self.adaptive.previous_indexes();
        if previous_indexes.is_empty() {
            return Vec::new();
        }

        if previous_indexes.len() == 1 {
            let start = *previous_indexes.front().unwrap() + 1;
            return (start..start + max_amount_to_prefetch).collect();
        }

        let mut sorted: Vec<usize> = previous_indexes.iter().copied().collect();
        sorted.sort_unstable();

        let mut subsequence_prefetches: Vec<Vec<usize>> = Vec::new();

        let extrapolate_subsequence = |begin: usize, end: usize, out: &mut Vec<Vec<usize>>| {
            if begin >= end {
                return;
            }
            let highest_value = sorted[end - 1];
            let mut sequence_length = 0usize;
            let mut indexes_begin = 0usize;
            for current_highest in (begin..end).rev() {
                let target = sorted[current_highest];
                if let Some(pos) = previous_indexes
                    .iter()
                    .skip(indexes_begin)
                    .position(|&v| v == target)
                {
                    indexes_begin += pos + 1;
                    sequence_length += 1;
                } else {
                    break;
                }
            }

            if self.memory_full() && sequence_length == 1 {
                return;
            }

            let consecutive_values = if sequence_length <= 1 {
                0
            } else {
                sequence_length
            };
            let saturation_count = if !self.memory_full() && consecutive_values > 0 {
                consecutive_values
            } else {
                self.memory_size_per_stream
            };
            out.push(FetchNextAdaptive::extrapolate_forward(
                highest_value,
                consecutive_values,
                saturation_count,
                max_amount_to_prefetch,
            ));
        };

        let mut last_range_begin = 0usize;
        for i in 0..sorted.len() {
            let at_end = i + 1 == sorted.len();
            let consecutive_break = !at_end && sorted[i] + 1 != sorted[i + 1];
            if at_end || consecutive_break {
                let range_end = if at_end { sorted.len() } else { i + 1 };
                extrapolate_subsequence(last_range_begin, range_end, &mut subsequence_prefetches);
                last_range_begin = i + 1;
            }
        }

        let mut result = interleave(&subsequence_prefetches);
        result.retain(|value| !previous_indexes.contains(value));
        result.truncate(max_amount_to_prefetch);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_returns_next_n_after_last() {
        let mut s = FetchNextFixed::default();
        s.fetch(10);
        assert_eq!(s.prefetch(3), vec![11, 12, 13]);
    }

    #[test]
    fn fixed_empty_when_nothing_fetched() {
        let s = FetchNextFixed::default();
        assert_eq!(s.prefetch(3), Vec::<usize>::new());
    }

    #[test]
    fn adaptive_single_access_extrapolates_fully() {
        let mut s = FetchNextAdaptive::new(3);
        s.fetch(5);
        assert_eq!(s.prefetch(4), vec![6, 7, 8, 9]);
    }

    #[test]
    fn adaptive_random_access_returns_empty() {
        let mut s = FetchNextAdaptive::new(3);
        s.fetch(5);
        s.fetch(100);
        s.fetch(50);
        // No consecutive pair → no prefetch.
        assert!(s.prefetch(4).is_empty());
    }

    #[test]
    fn adaptive_sequential_returns_full() {
        let mut s = FetchNextAdaptive::new(3);
        s.fetch(1);
        s.fetch(2);
        s.fetch(3);
        assert!(s.is_sequential());
        // 3 consecutive pairs → max amount.
        let p = s.prefetch(4);
        assert_eq!(p.first(), Some(&4));
    }

    #[test]
    fn adaptive_duplicate_fetches_ignored() {
        let mut s = FetchNextAdaptive::new(3);
        s.fetch(5);
        s.fetch(5);
        s.fetch(5);
        assert_eq!(s.previous_indexes().len(), 1);
    }

    #[test]
    fn split_index_reindexes_previous() {
        let mut s = FetchNextAdaptive::new(5);
        s.fetch(0);
        s.fetch(1);
        s.fetch(2);
        // Split index 1 into 3 indexes (so 1 → 1,2,3 and 2 shifts to 4).
        s.split_index(1, 3);
        let v: Vec<usize> = s.previous_indexes().iter().copied().collect();
        // Original previous (front→back) = [2, 1, 0]. After split:
        //   2 > 1 → 2 + (3-1) = 4
        //   1 == 1 → push 3, 2, 1 in that order
        //   0 < 1 → 0
        assert_eq!(v, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn multi_stream_single_access_extrapolates_fully() {
        let mut s = FetchMultiStream::new(3, 16);
        s.fetch(5);
        assert_eq!(s.prefetch(4), vec![6, 7, 8, 9]);
    }

    #[test]
    fn multi_stream_two_interleaved_sequences() {
        let mut s = FetchMultiStream::new(3, 16);
        // Two sequential streams interleaved: 10,11 and 20,21
        s.fetch(11);
        s.fetch(21);
        s.fetch(10);
        s.fetch(20);
        let p = s.prefetch(6);
        assert!(
            !p.is_empty(),
            "interleaved sequential streams should prefetch"
        );
        assert!(p.contains(&12) || p.contains(&22));
    }

    #[test]
    fn interleave_round_robin() {
        assert_eq!(
            interleave(&[vec![1, 2], vec![10, 20, 30]]),
            vec![1, 10, 2, 20, 30]
        );
    }
}
