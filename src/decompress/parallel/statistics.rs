#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of rapidgzip's fetcher statistics types
//! (vendor/.../core/BlockFetcher.hpp:34-200 +
//! vendor/.../GzipChunkFetcher.hpp:55-75).
//!
//! Two-layer structure that matches rapidgzip's layout:
//! - [`FetcherStatistics`] — base counters tracked by `BlockFetcher`:
//!   cache hits/misses, prefetch counts.
//! - [`ChunkFetcherStatistics`] — extends with the chunk-level counter
//!   `preemptive_stop_count`.
//!
//! Both are thread-safe via internal mutex (rapidgzip uses
//! `std::scoped_lock` on a mutex member).

use std::fmt;
use std::sync::Mutex;

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
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FetcherStatsSnapshot::default()),
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
}

impl ChunkFetcherStatistics {
    pub fn new() -> Self {
        Self {
            base: FetcherStatistics::new(),
            inner: Mutex::new(ChunkExtraSnapshot::default()),
        }
    }

    pub fn record_preemptive_stop(&self) {
        let mut g = self.inner.lock().unwrap();
        g.preemptive_stop_count += 1;
    }

    pub fn extra_snapshot(&self) -> ChunkExtraSnapshot {
        *self.inner.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_thread_safely() {
        let s = FetcherStatistics::new();
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
        let s = FetcherStatistics::new();
        let snap = s.snapshot();
        assert_eq!(snap.cache_unused_fraction(), 0.0);
    }

    #[test]
    fn snapshot_display_includes_all_sections() {
        let s = FetcherStatistics::new();
        s.record_prefetch();
        s.record_on_demand_fetch();
        s.set_block_count(39, true);
        let snap = s.snapshot();
        let printed = format!("{}", snap);
        assert!(printed.contains("Total Existing"));
        assert!(printed.contains("Total Fetched"));
        assert!(printed.contains("Prefetch Cache"));
    }

    #[test]
    fn snapshot_display_mirrors_vendor_verbose_sections() {
        // Lock in vendor-parity for the full Display output. Regression
        // catcher for the --verbose port (commits 52b398a..8e77404).
        // If a future commit drops or renames any of these sections,
        // gzippy's --verbose stops being directly comparable to
        // rapidgzip --verbose — which advisor passes rely on for
        // cross-tool diagnosis (used to find the chunk-size mismatch
        // and chunk-finalize divergence this session).
        let s = FetcherStatistics::new();
        s.record_prefetch();
        s.record_prefetch();
        s.record_on_demand_fetch();
        s.record_prefetch_cache_hit(true);
        s.record_prefetch_cache_miss();
        s.record_cache_unused_entry();
        s.set_block_count(42, true);
        let printed = format!("{}", s.snapshot());
        // vendor BlockFetcher.hpp:73-124 count sections (timing sections removed).
        for needle in [
            "Total Existing",
            "Total Fetched",
            "Prefetched",
            "Fetched On-demand",
            "Prefetch Stall by BlockFinder",
            "Prefetch Cache:",
            "Hits",
            "Misses",
            "Unused Entries",
            "Cache Hit Rate",
            "Useless Prefetches",
        ] {
            assert!(
                printed.contains(needle),
                "vendor-verbose Display missing section: '{needle}'\n=== printed ===\n{printed}",
            );
        }
    }

    #[test]
    fn chunk_extra_counts_preemptive_stops() {
        let s = ChunkFetcherStatistics::new();
        s.record_preemptive_stop();
        s.record_preemptive_stop();
        let extra = s.extra_snapshot();
        assert_eq!(extra.preemptive_stop_count, 2);
    }
}
