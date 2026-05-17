//! Literal port of `rapidgzip::BlockFetcher`
//! (vendor/.../core/BlockFetcher.hpp:38-688).
//!
//! Composes:
//! - [`Cache`](super::cache::Cache) for resolved block data (main cache).
//! - [`Cache`](super::cache::Cache) for prefetched block data
//!   (`prefetch_cache`), separated so prefetch hit/miss statistics can be
//!   tracked independently.
//! - [`FetchingStrategy`](super::prefetcher::FetchingStrategy) to decide
//!   which indexes to speculatively prefetch.
//! - [`ChunkFetcherStatistics`](super::statistics::ChunkFetcherStatistics)
//!   aggregating per-cache and per-fetcher counters.
//!
//! The async machinery (rapidgzip's `BS::thread_pool` + futures-based
//! `submitOnDemandTask` / `takeFromPrefetchQueue` / `wait_for` flow) is
//! NOT included here — gzippy uses `chunk_fetcher::drive`'s mpsc-channel
//! worker pool for that. This module is the cache-composition core; the
//! integration glue lives elsewhere.

#![allow(dead_code)]

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Mutex;

use super::cache::{Cache, CacheStrategy, LeastRecentlyUsed};
use super::prefetcher::FetchingStrategy;
use super::statistics::ChunkFetcherStatistics;

/// Generic BlockFetcher orchestration. `Key` is the block's compressed-bit
/// offset; `Value` is whatever the caller stores per-block.
///
/// `Strategy` defaults to LRU; `Prefetch` defaults to nothing — caller
/// picks a `FetchingStrategy` implementor.
pub struct BlockFetcher<
    Key: Hash + Eq + Clone + Ord,
    Value: Clone,
    Prefetch: FetchingStrategy,
    Strategy: CacheStrategy<Key> = LeastRecentlyUsed<Key>,
> {
    /// Main cache for resolved blocks.
    cache: Mutex<Cache<Key, Value, Strategy>>,
    /// Separate cache for prefetched blocks (rapidgzip BlockFetcher.hpp:46
    /// uses `prefetchCache` to keep prefetch stats independent of main
    /// cache hit/miss).
    prefetch_cache: Mutex<Cache<Key, Value>>,
    /// Set of block offsets currently being prefetched (in-flight tasks).
    /// Mirror of rapidgzip's `m_prefetching` (BlockFetcher.hpp:229).
    prefetching: Mutex<HashSet<Key>>,
    /// Strategy decides which indexes to prefetch.
    fetching_strategy: Mutex<Prefetch>,
    /// Aggregated statistics. Direct access; rapidgzip's struct uses a
    /// scoped lock on a member mutex per recording.
    pub statistics: ChunkFetcherStatistics,
    /// Block offsets where prefetch failed (e.g., from a phantom
    /// boundary). Subsequent get() calls skip these to avoid re-issuing
    /// failing work. Mirror of rapidgzip's `m_failedPrefetchCache`
    /// (BlockFetcher.hpp:369-374).
    failed_prefetch: Mutex<HashSet<Key>>,
}

impl<Key, Value, Prefetch> BlockFetcher<Key, Value, Prefetch>
where
    Key: Hash + Eq + Clone + Ord,
    Value: Clone,
    Prefetch: FetchingStrategy,
{
    pub fn new(
        cache_capacity: usize,
        prefetch_capacity: usize,
        fetching_strategy: Prefetch,
        parallelization: usize,
    ) -> Self {
        Self {
            cache: Mutex::new(Cache::new(cache_capacity)),
            prefetch_cache: Mutex::new(Cache::new(prefetch_capacity)),
            prefetching: Mutex::new(HashSet::new()),
            fetching_strategy: Mutex::new(fetching_strategy),
            statistics: ChunkFetcherStatistics::new(parallelization),
            failed_prefetch: Mutex::new(HashSet::new()),
        }
    }

    /// True iff `block_offset` is currently in any cache or in-flight
    /// prefetch set. Mirror of `BlockFetcher::test`
    /// (BlockFetcher.hpp:227-232).
    pub fn test(&self, block_offset: &Key) -> bool {
        self.prefetching.lock().unwrap().contains(block_offset)
            || self.cache.lock().unwrap().test(block_offset)
            || self.prefetch_cache.lock().unwrap().test(block_offset)
    }

    /// Synchronous dispatch primitive — literal port of rapidgzip's
    /// `BlockFetcher::get(blockOffset, blockIndex, getPartitionOffset)`
    /// at vendor/.../core/BlockFetcher.hpp:245-329.
    ///
    /// Returns the block data for `block_offset`. On cache or
    /// prefetch-cache hit, returns immediately (mirror of
    /// BlockFetcher.hpp:302-309). On miss, invokes `dispatch` to
    /// produce the value, on success inserts into the main cache and
    /// returns the value (mirror of BlockFetcher.hpp:317-328); on error
    /// the cache is left untouched and the error is propagated.
    /// Rapidgzip raises a C++ exception in the error path
    /// (BlockFetcher.hpp:656-661) — we use a typed Result.
    ///
    /// `Value` is typically `Arc<ChunkData>` for the production
    /// pipeline, matching rapidgzip's `std::shared_ptr<ChunkData>` at
    /// BlockFetcher.hpp:46. The Arc semantics are identical: cache and
    /// caller share the same allocation; consumer-side mutation goes
    /// through `Arc::make_mut` or a deliberate `.clone()` per the
    /// rapidgzip shared_ptr-aliasing model.
    ///
    /// **Threading-model deviation (§B5)**: rapidgzip submits a future
    /// to its internal `ThreadPool` (BlockFetcher.hpp:580) and waits on
    /// `std::future::get`. gzippy's `dispatch` closure is the unit of
    /// work — the caller is responsible for arranging actual parallelism
    /// (e.g. by having `dispatch` `submit_job(...).recv()` into the
    /// mpsc worker pool from `chunk_fetcher::drive`). The CALLER pattern
    /// is the same: one call returns the resolved data. Underlying
    /// thread-pool unification is deferred (see
    /// `docs/rapidgzip-port-reference.md` §B5).
    pub fn get<F, E>(&self, block_offset: Key, dispatch: F) -> Result<Value, E>
    where
        F: FnOnce() -> Result<Value, E>,
    {
        // BlockFetcher.hpp:263 — getFromCaches (prefetch-cache promote,
        // then main cache lookup). `get_if_available` already records
        // hit stats and promotes prefetched entries into the main cache.
        if let Some(v) = self.get_if_available(&block_offset) {
            return Ok(v);
        }
        // BlockFetcher.hpp:274-277 — submit on-demand and block.
        self.record_on_demand_fetch();
        let value = dispatch()?;
        // BlockFetcher.hpp:320 — insertIntoCache after future resolves.
        self.insert(block_offset.clone(), value.clone());
        Ok(value)
    }

    /// Pure cache lookup (no fetch). Checks prefetch cache first; on
    /// hit, MOVES the entry into the main cache (mirrors rapidgzip's
    /// `takeFromPrefetchQueue` pattern at BlockFetcher.hpp:385-410 —
    /// "take" semantically removes from prefetch queue).
    pub fn get_if_available(&self, block_offset: &Key) -> Option<Value> {
        // Try the prefetch cache first.
        {
            let mut pc = self.prefetch_cache.lock().unwrap();
            if let Some(v) = pc.get(block_offset) {
                self.statistics.base.record_prefetch_cache_hit(true);
                // Remove from prefetch cache (the "take" semantic) and
                // promote to main cache so subsequent gets are direct
                // main-cache hits. Mirror BlockFetcher.hpp:392-410.
                pc.evict(block_offset);
                drop(pc);
                self.cache
                    .lock()
                    .unwrap()
                    .insert(block_offset.clone(), v.clone());
                return Some(v);
            }
        }
        // Fall through to the main cache.
        let mut c = self.cache.lock().unwrap();
        match c.get(block_offset) {
            Some(v) => {
                self.statistics.base.record_get();
                Some(v)
            }
            None => None,
        }
    }

    /// Insert a freshly-decoded block into the main cache. If the
    /// fetching strategy reports sequential access, the cache is
    /// cleared first (rapidgzip BlockFetcher.hpp:355-358: avoids
    /// retaining no-longer-needed entries during streaming reads).
    pub fn insert(&self, block_offset: Key, block_data: Value) {
        let sequential = self.fetching_strategy.lock().unwrap().is_sequential();
        let mut c = self.cache.lock().unwrap();
        if sequential {
            c.clear();
        }
        c.insert(block_offset, block_data);
    }

    /// Insert a prefetched block into the prefetch cache. Stats:
    /// records the prefetch (rapidgzip's BlockFetcher tracks this via
    /// `prefetchCount` increment on prefetch submission).
    pub fn insert_prefetched(&self, block_offset: Key, block_data: Value) {
        self.prefetch_cache
            .lock()
            .unwrap()
            .insert(block_offset, block_data);
        self.statistics.base.record_prefetch();
        // The prefetching-set entry is removed by `complete_prefetch`
        // once the worker finishes.
    }

    /// Record that an on-demand fetch was issued. Bumps the on-demand
    /// counter. Mirror of `BlockFetcher::submitOnDemandTask`'s stats
    /// path (around BlockFetcher.hpp:600).
    pub fn record_on_demand_fetch(&self) {
        self.statistics.base.record_on_demand_fetch();
    }

    /// Record that a prefetch task started. Adds the offset to the
    /// in-flight set so duplicate prefetches are skipped.
    pub fn note_prefetch_started(&self, block_offset: Key) {
        self.prefetching.lock().unwrap().insert(block_offset);
    }

    /// Record that a prefetch task completed. Removes from in-flight set.
    pub fn note_prefetch_completed(&self, block_offset: &Key) {
        self.prefetching.lock().unwrap().remove(block_offset);
    }

    /// Mark a block offset as a failed prefetch (rapidgzip
    /// `m_failedPrefetchCache`). Future `is_failed_prefetch` lookups
    /// return true so the caller doesn't re-issue.
    pub fn mark_failed_prefetch(&self, block_offset: Key) {
        self.failed_prefetch.lock().unwrap().insert(block_offset);
    }

    pub fn is_failed_prefetch(&self, block_offset: &Key) -> bool {
        self.failed_prefetch.lock().unwrap().contains(block_offset)
    }

    /// Update the fetching strategy with a freshly-observed access.
    pub fn record_fetch(&self, data_block_index: usize) {
        self.fetching_strategy
            .lock()
            .unwrap()
            .fetch(data_block_index);
    }

    /// Ask the fetching strategy which indexes to prefetch.
    pub fn prefetch_indexes(&self, max_amount_to_prefetch: usize) -> Vec<usize> {
        self.fetching_strategy
            .lock()
            .unwrap()
            .prefetch(max_amount_to_prefetch)
    }

    /// Last accessed index (informational; matches FetchingStrategy::last_fetched).
    pub fn last_fetched(&self) -> Option<usize> {
        self.fetching_strategy.lock().unwrap().last_fetched()
    }

    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn clear_prefetch_cache(&self) {
        self.prefetch_cache.lock().unwrap().clear();
    }

    /// Lock + snapshot both caches' statistics + the chunk-extra
    /// counters. Mirror of `BlockFetcher::statistics`
    /// (BlockFetcher.hpp:337-348).
    pub fn cache_statistics(
        &self,
    ) -> (super::cache::CacheStatistics, super::cache::CacheStatistics) {
        (
            self.cache.lock().unwrap().statistics(),
            self.prefetch_cache.lock().unwrap().statistics(),
        )
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().size()
    }

    pub fn prefetch_cache_size(&self) -> usize {
        self.prefetch_cache.lock().unwrap().size()
    }

    pub fn in_flight_count(&self) -> usize {
        self.prefetching.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::prefetcher::FetchNextAdaptive;
    use super::*;

    fn new_fetcher() -> BlockFetcher<u64, String, FetchNextAdaptive> {
        BlockFetcher::new(4, 4, FetchNextAdaptive::new(3), 4)
    }

    #[test]
    fn get_if_available_returns_none_when_empty() {
        let bf = new_fetcher();
        assert_eq!(bf.get_if_available(&100), None);
    }

    #[test]
    fn insert_then_get_returns_value() {
        let bf = new_fetcher();
        bf.insert(100, "block-100".into());
        assert_eq!(bf.get_if_available(&100), Some("block-100".into()));
    }

    #[test]
    fn prefetched_block_promotes_to_main_cache_on_get() {
        let bf = new_fetcher();
        bf.insert_prefetched(200, "pre-200".into());
        assert_eq!(bf.prefetch_cache_size(), 1);
        let v = bf.get_if_available(&200);
        assert_eq!(v, Some("pre-200".into()));
        // After promotion, main cache has it (prefetch_cache still has it
        // too — Cache::get does NOT evict; promotion just inserts into main).
        assert_eq!(bf.cache_size(), 1);
    }

    #[test]
    fn test_returns_true_for_in_flight_prefetch() {
        let bf = new_fetcher();
        bf.note_prefetch_started(300);
        assert!(bf.test(&300));
        bf.note_prefetch_completed(&300);
        assert!(!bf.test(&300));
    }

    #[test]
    fn failed_prefetch_flag_persists() {
        let bf = new_fetcher();
        assert!(!bf.is_failed_prefetch(&400));
        bf.mark_failed_prefetch(400);
        assert!(bf.is_failed_prefetch(&400));
    }

    #[test]
    fn record_fetch_drives_prefetch_indexes() {
        let bf = new_fetcher();
        bf.record_fetch(10);
        // Single-fetch case: FetchNextAdaptive extrapolates fully.
        let p = bf.prefetch_indexes(3);
        assert_eq!(p, vec![11, 12, 13]);
    }

    #[test]
    fn statistics_track_hits_and_prefetch_count() {
        let bf = new_fetcher();
        bf.insert_prefetched(500, "p".into());
        let _ = bf.get_if_available(&500); // prefetch hit
        let _ = bf.get_if_available(&500); // main cache hit (promoted)
        let snap = bf.statistics.base.snapshot();
        assert!(snap.prefetch_count >= 1);
        assert!(snap.prefetch_cache_hits >= 1);
        assert!(snap.gets >= 1);
    }

    #[test]
    fn get_returns_cached_value_without_dispatch() {
        let bf = new_fetcher();
        bf.insert(700, "cached".into());
        let mut dispatched = false;
        let v = bf
            .get(700, || -> Result<String, ()> {
                dispatched = true;
                Ok("dispatched".into())
            })
            .unwrap();
        assert_eq!(v, "cached");
        assert!(!dispatched, "dispatch should not run on cache hit");
    }

    #[test]
    fn get_invokes_dispatch_on_miss_and_caches_result() {
        let bf = new_fetcher();
        let v = bf
            .get(800, || -> Result<String, ()> { Ok("from-dispatch".into()) })
            .unwrap();
        assert_eq!(v, "from-dispatch");
        // Mirror of rapidgzip BlockFetcher.hpp:320 (`insertIntoCache`
        // after dispatch): the freshly-dispatched value is now in the
        // main cache. A subsequent `get` for the same key must short-
        // circuit without invoking the dispatch closure.
        let mut dispatched_again = false;
        let v2 = bf
            .get(800, || -> Result<String, ()> {
                dispatched_again = true;
                Ok("re-dispatched".into())
            })
            .unwrap();
        assert_eq!(v2, "from-dispatch");
        assert!(
            !dispatched_again,
            "second get must hit cache, not re-dispatch — mirror of \
             rapidgzip's insertIntoCache at BlockFetcher.hpp:320"
        );
    }

    #[test]
    fn get_promotes_prefetched_value_on_hit() {
        let bf = new_fetcher();
        bf.insert_prefetched(900, "prefetched".into());
        let mut dispatched = false;
        let v = bf
            .get(900, || -> Result<String, ()> {
                dispatched = true;
                Ok("should-not-run".into())
            })
            .unwrap();
        assert_eq!(v, "prefetched");
        assert!(!dispatched);
        // After get, prefetched entry has been promoted to main cache.
        assert_eq!(bf.cache_size(), 1);
    }

    #[test]
    fn get_records_on_demand_fetch_on_miss() {
        let bf = new_fetcher();
        let before = bf.statistics.base.snapshot().on_demand_fetch_count;
        let _v = bf
            .get(1000, || -> Result<String, ()> { Ok("x".into()) })
            .unwrap();
        let after = bf.statistics.base.snapshot().on_demand_fetch_count;
        assert!(after > before);
    }

    #[test]
    fn get_propagates_dispatch_error_without_caching() {
        let bf = new_fetcher();
        let err = bf
            .get(1100, || -> Result<String, &'static str> { Err("boom") })
            .unwrap_err();
        assert_eq!(err, "boom");
        // Error must NOT be cached: a second get with a successful
        // dispatch should run it.
        let ok = bf
            .get(1100, || -> Result<String, &'static str> {
                Ok("recovered".into())
            })
            .unwrap();
        assert_eq!(ok, "recovered");
    }

    #[test]
    fn sequential_insert_clears_main_cache() {
        let bf = new_fetcher();
        // Drive the strategy to is_sequential() = true.
        bf.record_fetch(10);
        bf.record_fetch(11);
        bf.record_fetch(12);
        // Pre-populate main cache.
        bf.insert(99, "older".into());
        assert_eq!(bf.cache_size(), 1);
        // Sequential insert should clear before inserting.
        bf.insert(13, "newer".into());
        assert_eq!(bf.cache_size(), 1);
        assert_eq!(bf.get_if_available(&99), None);
        assert_eq!(bf.get_if_available(&13), Some("newer".into()));
    }
}
