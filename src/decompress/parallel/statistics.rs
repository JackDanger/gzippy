//! Literal port of rapidgzip's fetcher statistics types
//! (vendor/.../core/BlockFetcher.hpp:34-200 +
//! vendor/.../GzipChunkFetcher.hpp:55-75).
//!
//! Two-layer structure that matches rapidgzip's layout:
//! - [`FetcherStatistics`] — base counters tracked by `BlockFetcher`:
//!   cache hits/misses, prefetch counts, per-stage durations, pool
//!   utilization.
//! - [`ChunkFetcherStatistics`] — extends with chunk-level counters
//!   `preemptive_stop_count` + `queue_post_processing_duration`.
//!
//! Both are thread-safe via internal mutex (rapidgzip uses
//! `std::scoped_lock` on a mutex member).

#![allow(dead_code)]

use std::fmt;
use std::sync::Mutex;
use std::time::Instant;

/// Snapshot taken by `FetcherStatistics::print`. Fields mirror
/// rapidgzip's `--verbose` output (BlockFetcher.hpp:73-124).
#[derive(Debug, Clone, Default)]
pub struct FetcherStatsSnapshot {
    pub block_count: usize,
    pub block_count_finalized: bool,
    pub gets: u64,
    pub sequential_reads: u64,
    pub on_demand_fetch_count: u64,
    pub prefetch_count: u64,
    pub prefetch_direct_hits: u64,
    pub prefetch_cache_hits: u64,
    pub prefetch_cache_misses: u64,
    pub prefetch_cache_unused_entries: u64,
    pub wait_on_block_finder_count: u64,
    /// Total CPU time spent in chunk decode (sum across workers).
    pub decode_block_total_time: f64,
    /// Total wall time spent waiting on `std::future::get` in rapidgzip.
    pub future_wait_total_time: f64,
    /// Total wall time spent inside the `get` (consumer) entry point.
    pub get_total_time: f64,
    /// First chunk-decode start; last chunk-decode end. Used to
    /// compute wall-clock duration for pool efficiency.
    pub decode_block_start: Option<f64>,
    pub decode_block_end: Option<f64>,
    pub parallelization: usize,
}

impl FetcherStatsSnapshot {
    /// Cache miss rate vs total fetches. Mirror of
    /// `Statistics::cacheUnusedFraction` (BlockFetcher.hpp:62-71).
    pub fn cache_unused_fraction(&self) -> f64 {
        let total = self.prefetch_count + self.on_demand_fetch_count;
        if total == 0 {
            return 0.0;
        }
        self.prefetch_cache_unused_entries as f64 / total as f64
    }

    /// Pool efficiency = (decode CPU total / parallelization) /
    /// decode wall duration. Mirror of BlockFetcher.hpp:83-84.
    pub fn pool_efficiency(&self) -> f64 {
        let decode_duration = match (self.decode_block_start, self.decode_block_end) {
            (Some(s), Some(e)) if e > s => e - s,
            _ => return 0.0,
        };
        if decode_duration <= 0.0 || self.parallelization == 0 {
            return 0.0;
        }
        let optimal = self.decode_block_total_time / self.parallelization as f64;
        optimal / decode_duration
    }
}

impl fmt::Display for FetcherStatsSnapshot {
    /// Mirrors rapidgzip's `Statistics::print` output format
    /// (BlockFetcher.hpp:73-124).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.block_count_finalized { "" } else { ">=" };
        writeln!(
            f,
            "        Total Existing                : {}{}",
            prefix, self.block_count
        )?;
        writeln!(
            f,
            "        Total Fetched                 : {}",
            self.prefetch_count + self.on_demand_fetch_count
        )?;
        writeln!(
            f,
            "        Prefetched                    : {}",
            self.prefetch_count
        )?;
        writeln!(
            f,
            "        Fetched On-demand             : {}",
            self.on_demand_fetch_count
        )?;
        writeln!(
            f,
            "    Prefetch Stall by BlockFinder     : {}",
            self.wait_on_block_finder_count
        )?;
        writeln!(
            f,
            "    Time spent in:\n        decodeBlock                   : {:.6} s\n        std::future::get              : {:.6} s\n        get                           : {:.6} s",
            self.decode_block_total_time, self.future_wait_total_time, self.get_total_time
        )?;
        let decode_duration = match (self.decode_block_start, self.decode_block_end) {
            (Some(s), Some(e)) => e - s,
            _ => 0.0,
        };
        let optimal = if self.parallelization > 0 {
            self.decode_block_total_time / self.parallelization as f64
        } else {
            0.0
        };
        writeln!(
            f,
            "    Thread Pool Utilization:\n        Total Real Decode Duration    : {:.6} s\n        Theoretical Optimal Duration  : {:.6} s\n        Pool Efficiency (Fill Factor) : {:.2} %",
            decode_duration, optimal, self.pool_efficiency() * 100.0
        )?;
        // Cache stats — mirror of vendor's BlockFetcher.hpp:97-118
        // (the section vendor labels "Prefetch Cache" + "Cache Hit
        // Rate" + "Useless Prefetches"). Populated by the prefetch-
        // miss + cache-unused-entry counters (commit ef15b4a).
        writeln!(
            f,
            "    Prefetch Cache:\n        Hits                          : {}\n        Misses                        : {}\n        Unused Entries                : {}",
            self.prefetch_cache_hits, self.prefetch_cache_misses, self.prefetch_cache_unused_entries
        )?;
        let total_fetches = self.prefetch_count + self.on_demand_fetch_count;
        let cache_hit_rate = if total_fetches > 0 {
            self.prefetch_cache_hits as f64 / total_fetches as f64 * 100.0
        } else {
            0.0
        };
        let useless_prefetches = if self.prefetch_count > 0 {
            self.prefetch_cache_unused_entries as f64 / self.prefetch_count as f64 * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "    Cache Hit Rate                    : {:.2} %\n    Useless Prefetches                : {:.2} %",
            cache_hit_rate, useless_prefetches
        )?;
        Ok(())
    }
}

/// Thread-safe accumulator for `FetcherStatsSnapshot`. Mirror of
/// `BlockFetcher::Statistics` (BlockFetcher.hpp:34-200).
#[derive(Default)]
pub struct FetcherStatistics {
    inner: Mutex<FetcherStatsSnapshot>,
}

impl FetcherStatistics {
    pub fn new(parallelization: usize) -> Self {
        Self {
            inner: Mutex::new(FetcherStatsSnapshot {
                parallelization,
                ..Default::default()
            }),
        }
    }

    pub fn record_get(&self) {
        let mut g = self.inner.lock().unwrap();
        g.gets += 1;
    }

    pub fn record_sequential_read(&self) {
        let mut g = self.inner.lock().unwrap();
        g.sequential_reads += 1;
    }

    pub fn record_prefetch(&self) {
        let mut g = self.inner.lock().unwrap();
        g.prefetch_count += 1;
    }

    pub fn record_on_demand_fetch(&self) {
        let mut g = self.inner.lock().unwrap();
        g.on_demand_fetch_count += 1;
    }

    pub fn record_prefetch_cache_hit(&self, direct: bool) {
        let mut g = self.inner.lock().unwrap();
        g.prefetch_cache_hits += 1;
        if direct {
            g.prefetch_direct_hits += 1;
        }
    }

    pub fn record_prefetch_cache_miss(&self) {
        let mut g = self.inner.lock().unwrap();
        g.prefetch_cache_misses += 1;
    }

    pub fn record_cache_unused_entry(&self) {
        let mut g = self.inner.lock().unwrap();
        g.prefetch_cache_unused_entries += 1;
    }

    pub fn record_wait_on_block_finder(&self) {
        let mut g = self.inner.lock().unwrap();
        g.wait_on_block_finder_count += 1;
    }

    pub fn add_decode_block_time(&self, seconds: f64) {
        let mut g = self.inner.lock().unwrap();
        g.decode_block_total_time += seconds;
    }

    pub fn add_future_wait_time(&self, seconds: f64) {
        let mut g = self.inner.lock().unwrap();
        g.future_wait_total_time += seconds;
    }

    pub fn add_get_time(&self, seconds: f64) {
        let mut g = self.inner.lock().unwrap();
        g.get_total_time += seconds;
    }

    pub fn note_decode_block_start(&self, t: f64) {
        let mut g = self.inner.lock().unwrap();
        if g.decode_block_start.is_none() {
            g.decode_block_start = Some(t);
        }
    }

    pub fn note_decode_block_end(&self, t: f64) {
        let mut g = self.inner.lock().unwrap();
        g.decode_block_end = Some(t);
    }

    pub fn set_block_count(&self, count: usize, finalized: bool) {
        let mut g = self.inner.lock().unwrap();
        g.block_count = count;
        g.block_count_finalized = finalized;
    }

    pub fn snapshot(&self) -> FetcherStatsSnapshot {
        self.inner.lock().unwrap().clone()
    }
}

/// `GzipChunkFetcher::Statistics` extension (GzipChunkFetcher.hpp:55-75).
/// Tracks chunk-level counters that don't make sense on the generic
/// BlockFetcher base.
#[derive(Default)]
pub struct ChunkFetcherStatistics {
    pub base: FetcherStatistics,
    inner: Mutex<ChunkExtraSnapshot>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkExtraSnapshot {
    pub preemptive_stop_count: u64,
    pub queue_post_processing_duration: f64,
}

impl ChunkFetcherStatistics {
    pub fn new(parallelization: usize) -> Self {
        Self {
            base: FetcherStatistics::new(parallelization),
            inner: Mutex::new(ChunkExtraSnapshot::default()),
        }
    }

    pub fn record_preemptive_stop(&self) {
        let mut g = self.inner.lock().unwrap();
        g.preemptive_stop_count += 1;
    }

    pub fn add_queue_post_processing_duration(&self, seconds: f64) {
        let mut g = self.inner.lock().unwrap();
        g.queue_post_processing_duration += seconds;
    }

    pub fn extra_snapshot(&self) -> ChunkExtraSnapshot {
        *self.inner.lock().unwrap()
    }
}

/// Helper to time a region of code and add to a counter. Returned guard
/// records elapsed seconds when dropped.
pub struct TimerGuard<'a, F: FnMut(f64)> {
    start: Instant,
    record: F,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, F: FnMut(f64)> TimerGuard<'a, F> {
    pub fn new(record: F) -> Self {
        Self {
            start: Instant::now(),
            record,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: FnMut(f64)> Drop for TimerGuard<'_, F> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        (self.record)(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_thread_safely() {
        let s = FetcherStatistics::new(4);
        s.record_get();
        s.record_get();
        s.record_prefetch();
        s.record_on_demand_fetch();
        s.record_prefetch_cache_hit(true);
        let snap = s.snapshot();
        assert_eq!(snap.gets, 2);
        assert_eq!(snap.prefetch_count, 1);
        assert_eq!(snap.on_demand_fetch_count, 1);
        assert_eq!(snap.prefetch_cache_hits, 1);
        assert_eq!(snap.prefetch_direct_hits, 1);
    }

    #[test]
    fn cache_unused_fraction_zero_when_empty() {
        let s = FetcherStatistics::new(4);
        let snap = s.snapshot();
        assert_eq!(snap.cache_unused_fraction(), 0.0);
    }

    #[test]
    fn pool_efficiency_computes_from_decode_window() {
        let s = FetcherStatistics::new(4);
        s.note_decode_block_start(0.0);
        s.note_decode_block_end(1.0); // 1 second wall
        s.add_decode_block_time(2.0); // 2 seconds of CPU across 4 threads
        let snap = s.snapshot();
        // optimal = 2.0 / 4 = 0.5; pool_efficiency = 0.5 / 1.0 = 0.5
        assert!((snap.pool_efficiency() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn snapshot_display_includes_all_sections() {
        let s = FetcherStatistics::new(16);
        s.record_prefetch();
        s.record_on_demand_fetch();
        s.set_block_count(39, true);
        let snap = s.snapshot();
        let printed = format!("{}", snap);
        assert!(printed.contains("Total Existing"));
        assert!(printed.contains("Total Fetched"));
        assert!(printed.contains("Thread Pool Utilization"));
    }

    #[test]
    fn chunk_extra_counts_preemptive_stops() {
        let s = ChunkFetcherStatistics::new(4);
        s.record_preemptive_stop();
        s.record_preemptive_stop();
        s.add_queue_post_processing_duration(0.1);
        let extra = s.extra_snapshot();
        assert_eq!(extra.preemptive_stop_count, 2);
        assert!((extra.queue_post_processing_duration - 0.1).abs() < 1e-9);
    }
}
