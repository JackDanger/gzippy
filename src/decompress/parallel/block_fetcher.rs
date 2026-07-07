#![cfg(parallel_sm)]

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
//! - A `prefetching: HashMap<Key, Receiver<Result<Value, E>>>` mirror of
//!   vendor's `m_prefetching: std::map<size_t, std::future<BlockData>>`
//!   at vendor/.../core/BlockFetcher.hpp:131. Receivers are produced by
//!   the caller-supplied `submit` closure (a stand-in for vendor's
//!   `m_threadPool.submit(...)` at BlockFetcher.hpp:554-558) and stored
//!   here so the consumer's `get()` can do an exact-match take from the
//!   prefetch queue (BlockFetcher.hpp:385-410) instead of an inline
//!   ad-hoc ring.
//!
//! The thread pool is external: callers pass a `submit_decode` closure
//! that dispatches a `DecodeJob` to the ported `ThreadPool`, and
//! `BlockFetcher` holds the `Receiver` for the result. This is the
//! rapidgzip `BS::thread_pool::submit` -> `future` mapping with a
//! stdlib `mpsc` channel as the closest equivalent.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::mpsc::Receiver;
use std::sync::Mutex;

use super::cache::{Cache, CacheStrategy, LeastRecentlyUsed};
use super::prefetcher::FetchingStrategy;
use super::statistics::ChunkFetcherStatistics;

/// Generic BlockFetcher orchestration. `Key` is the block's compressed-bit
/// offset; `Value` is whatever the caller stores per-block; `Err` is the
/// error type produced by a dispatched fetch task (vendor uses C++
/// exceptions thrown out of the future).
///
/// `Strategy` defaults to LRU; `Prefetch` defaults to nothing — caller
/// picks a `FetchingStrategy` implementor.
pub struct BlockFetcher<
    Key: Hash + Eq + Clone + Ord + std::fmt::Debug,
    Value: Clone,
    Prefetch: FetchingStrategy,
    Err = (),
    Strategy: CacheStrategy<Key> = LeastRecentlyUsed<Key>,
> {
    /// Main cache for resolved blocks.
    cache: Mutex<Cache<Key, Value, Strategy>>,
    /// Separate cache for prefetched blocks (rapidgzip BlockFetcher.hpp:46
    /// uses `prefetchCache` to keep prefetch stats independent of main
    /// cache hit/miss).
    prefetch_cache: Mutex<Cache<Key, Value>>,
    /// In-flight prefetch tasks. Mirror of vendor's
    /// `m_prefetching: std::map<size_t /* blockOffset */, std::future<BlockData>>`
    /// at vendor/.../core/BlockFetcher.hpp:131. Receivers replace the
    /// old `HashSet<Key>` whose only role was tracking in-flight
    /// membership — now we both track membership AND hold the future so
    /// `get()` can do an exact-match take + wait, mirroring vendor's
    /// `takeFromPrefetchQueue` (BlockFetcher.hpp:385-410).
    prefetching: Mutex<HashMap<Key, Receiver<Result<Value, Err>>>>,
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
    /// Soft cap on the simultaneous in-flight prefetches, used by
    /// `prefetch_new_blocks` to decide when to stop. Mirror of vendor's
    /// `m_threadPool.capacity()` check at BlockFetcher.hpp:467 (vendor
    /// caps prefetches at `parallelization - 1`).
    parallelization: usize,
}

impl<Key, Value, Prefetch, Err> BlockFetcher<Key, Value, Prefetch, Err>
where
    Key: Hash + Eq + Clone + Ord + std::fmt::Debug,
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
            prefetching: Mutex::new(HashMap::new()),
            fetching_strategy: Mutex::new(fetching_strategy),
            statistics: ChunkFetcherStatistics::new(parallelization),
            failed_prefetch: Mutex::new(HashSet::new()),
            parallelization,
        }
    }

    /// True iff `block_offset` is currently in any cache or in-flight
    /// prefetch set. Mirror of `BlockFetcher::test`
    /// (BlockFetcher.hpp:227-232).
    pub fn test(&self, block_offset: &Key) -> bool {
        self.prefetching.lock().unwrap().contains_key(block_offset)
            || self.cache.lock().unwrap().test(block_offset)
            || self.prefetch_cache.lock().unwrap().test(block_offset)
    }

    /// Synchronous dispatch primitive — literal port of rapidgzip's
    /// `BlockFetcher::get(blockOffset, blockIndex, getPartitionOffset)`
    /// at vendor/.../core/BlockFetcher.hpp:245-329.
    ///
    /// Returns the block data for `block_offset`. Vendor flow, body
    /// mirrored line-by-line:
    ///
    ///   1. `getFromCaches(blockOffset)` (vendor L263) — cache lookup,
    ///      with prefetch-cache promote-on-hit.
    ///   2. If not in caches, check `m_prefetching` for an in-flight
    ///      task at the same offset (vendor's `queuedResult` at L265).
    ///      Take it if present.
    ///   3. If neither cache nor prefetch queue has it, call `submit`
    ///      to dispatch a new on-demand task (vendor's
    ///      `submitOnDemandTask` at L276).
    ///   4. Wait on the receiver (vendor's `queuedResult.get()` at L317).
    ///   5. `insertIntoCache` on success (vendor L320). Errors are
    ///      propagated without caching (vendor raises in this branch
    ///      via `decodeAndMeasureBlock` at L656-661).
    ///
    /// The `submit` closure is the gzip-specific dispatch (sends a
    /// `DecodeJob` to the worker pool and returns the reply `Receiver`).
    /// Mirror of vendor's `m_threadPool.submit(...)` lambda body at
    /// BlockFetcher.hpp:554-558, factored out so this generic core
    /// doesn't depend on the gzip job type.
    ///
    /// **Threading-model deviation**: rapidgzip uses
    /// `BS::thread_pool` + `std::future`. We use stdlib `mpsc::Receiver`
    /// as the closest equivalent. The CALLER pattern is identical:
    /// `get()` does cache lookup + prefetch take + dispatch + wait +
    /// cache-insert in a single call. Underlying thread-pool unification
    /// is a follow-up.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn get<S>(&self, block_offset: Key, submit: S) -> Result<Value, Err>
    where
        S: FnOnce(Key) -> Receiver<Result<Value, Err>>,
    {
        // BlockFetcher.hpp:263 — getFromCaches. `get_if_available`
        // already records hit stats and promotes prefetched entries
        // into the main cache.
        if let Some(v) = self.get_if_available(&block_offset) {
            return Ok(v);
        }

        // BlockFetcher.hpp:265 — `queuedResult = ... takeFromPrefetchQueue`.
        // Vendor's exact-match path on `m_prefetching`: if an in-flight
        // prefetch matches our key, take the receiver and wait on it
        // instead of issuing a duplicate dispatch.
        let rx = match self.take_prefetch(&block_offset) {
            Some(existing) => existing,
            None => {
                // BlockFetcher.hpp:274-277 — submit on-demand task.
                self.record_on_demand_fetch();
                submit(block_offset.clone())
            }
        };

        // BlockFetcher.hpp:317 — `queuedResult.get()`. Wait on the
        // receiver. Sender-dropped is treated as a broken promise
        // (vendor's `std::future::get` raises in this case); we panic
        // since the worker pool dropping its reply mid-decode means
        // the pool itself is in a broken state.
        let value = match rx.recv() {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(e),
            Err(_) => panic!("block_fetcher::get: dispatch worker dropped reply (broken promise)"),
        };
        // BlockFetcher.hpp:320 — `insertIntoCache(blockOffset, result)`.
        self.insert(block_offset.clone(), value.clone());
        Ok(value)
    }

    /// Try to satisfy `block_offset` from caches or the in-flight
    /// prefetch queue WITHOUT issuing an on-demand submit. Returns
    /// `Some(Ok(value))` on cache hit, `Some(Ok(value))` after waiting
    /// on a matched prefetch receiver, `Some(Err(e))` if the matched
    /// prefetch task errored, or `None` if neither cache nor prefetch
    /// queue had anything to serve.
    ///
    /// Mirror of vendor's `BaseType::test(partitionOffset) &&
    /// BaseType::get(partitionOffset, ...)` pattern inside
    /// `GzipChunkFetcher::getBlock` at
    /// vendor/.../rapidgzip/GzipChunkFetcher.hpp:600-609. Vendor uses
    /// this to first try the prefetched future keyed under the partition
    /// offset (where prefetch was submitted) before falling back to
    /// dispatching a fresh on-demand task at the real `blockOffset`.
    ///
    /// On success, the value is inserted into the main cache under the
    /// original `block_offset` (the partition offset) — matching vendor's
    /// `BaseType::get(partitionOffset, ...)` at
    /// GzipChunkFetcher.hpp:602, whose `insertIntoCache(blockOffset, ...)`
    /// at BlockFetcher.hpp:320 keys under the SAME argument it was called
    /// with (i.e. partitionOffset). The consumer is responsible for
    /// running `matchesEncodedOffset(realOffset)` and falling back to
    /// `BaseType::get(realOffset, ...)` on mismatch
    /// (GzipChunkFetcher.hpp:646-654). Re-keying under the real offset
    /// here would pollute the cache: a wrong-range chunk would short-
    /// circuit the next `get_if_available(realOffset)` check.
    /// `pump_prefetch` is called once per 1ms wait tick while `rx.recv`
    /// is blocking — the caller threads in `prefetch_new_blocks` with its
    /// closures captured. Vendor parity: BlockFetcher.hpp:314-316
    /// `while ( queuedResult.wait_for(1ms) == timeout ) prefetchNewBlocks(...)`.
    /// Previously gzippy did a bare `rx.recv()` here, which parked the
    /// consumer thread without dispatching new prefetches — `pool.pick`
    /// stayed at 2.3s of idle worker capacity while consumer waited
    /// 89-171 ms per session on in-flight prefetches that took 20-25 ms
    /// each (`ttp.rx_recv_block` trace bucket).
    pub fn try_take_prefetched_pumping<F>(
        &self,
        block_offset: &Key,
        mut pump_prefetch: F,
    ) -> Option<Result<Value, Err>>
    where
        F: FnMut(),
    {
        if let Some(v) = {
            let _tv2 =
                crate::decompress::parallel::trace_v2::SpanGuard::begin("ttp.get_if_available");
            self.get_if_available(block_offset)
        } {
            return Some(Ok(v));
        }
        let rx = {
            let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin("ttp.take_prefetch");
            self.take_prefetch(block_offset)?
        };
        // Record the AWAITED chunk's encoded offset so the consumer wait can be
        // decomposed (Fulcrum) against that exact chunk's decode span — robust
        // keyed correlation, not overlap-heuristic blame. This is THE wait that
        // gates the in-order wall (~97% of it), so knowing which chunk it waits
        // on, and what that chunk was doing, is the indisputable measurement.
        let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
            "ttp.rx_recv_block",
            &format!(r#""awaited_offset":{block_offset:?}"#),
        );
        // Lever H: pump prefetch on 1ms ticks while waiting (vendor
        // BlockFetcher.hpp:314-316). Keeps the prefetch horizon
        // advancing so by the time chunk N arrives, chunks N+1..N+k
        // are already in flight or done — reducing subsequent
        // `wait.block_fetcher_get` and `ttp.rx_recv_block` events.
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(Ok(v)) => return Some(Ok(v)),
                Ok(Err(e)) => return Some(Err(e)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    pump_prefetch();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!(
                    "block_fetcher::try_take_prefetched: dispatch worker dropped reply \
                     (broken promise)"
                ),
            }
        }
    }

    /// Back-compat wrapper for callers that don't want to thread a pump
    /// closure (tests). Production callers must use the pumping variant.
    #[allow(dead_code)]
    pub fn try_take_prefetched(&self, block_offset: &Key) -> Option<Result<Value, Err>> {
        self.try_take_prefetched_pumping(block_offset, || {})
    }

    /// Like `get`, plus the `prefetchNewBlocks` body interleaved
    /// before and after the on-demand wait. This is the literal
    /// shape of vendor's `BlockFetcher::get` at BlockFetcher.hpp:245-329:
    ///
    ///   1. getFromCaches / takeFromPrefetchQueue
    ///   2. If miss: submitOnDemandTask
    ///   3. m_fetchingStrategy.fetch(validDataBlockIndex)
    ///   4. If index changed: prefetchNewBlocks (line 297-299)
    ///   5. Wait on queuedResult, calling prefetchNewBlocks during each
    ///      timeout cycle (line 314-316)
    ///   6. insertIntoCache
    ///
    /// We provide a single `should_drive_prefetch` here that captures
    /// the index-changed condition; the caller still does the
    /// `record_fetch(idx)` to update the strategy. The prefetch
    /// closures match `prefetch_new_blocks` above.
    ///
    /// `partition_offset_for` mirrors vendor's `getPartitionOffsetFromOffset`
    /// parameter (BlockFetcher.hpp:240-248). It is consulted in
    /// `prefetch_new_blocks` to double-key the prefetch map under both
    /// real and partition offsets (BlockFetcher.hpp:485-489), matching
    /// vendor exactly.
    ///
    /// Returns `(value, prefetches_submitted_count)` so the caller can
    /// log / stat-record without re-querying.
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_prefetch<S, L, LN, P, F, PO>(
        &self,
        block_offset: Key,
        submit: S,
        lookup_block_offset: L,
        lookup_next_block_offset: LN,
        submit_for_prefetch: P,
        is_finalized_and_index_too_high: F,
        partition_offset_for: PO,
        should_drive_prefetch: bool,
    ) -> (Result<Value, Err>, usize)
    where
        S: FnOnce(Key) -> Receiver<Result<Value, Err>>,
        L: Fn(usize) -> Option<Key>,
        // `lookup_next_block_offset` — see `prefetch_new_blocks` doc.
        // Permits `file_size_in_bits` for the last chunk's stop hint
        // (vendor BlockFetcher.hpp:533-535 asymmetry).
        LN: Fn(usize) -> Option<Key>,
        // Vendor's prefetch task captures both offset and nextOffset
        // (BlockFetcher.hpp:555-557). See `prefetch_new_blocks`.
        P: Fn(Key, Key) -> Receiver<Result<Value, Err>>,
        F: Fn(usize) -> bool,
        PO: Fn(&Key) -> Key,
    {
        // Vendor's `BlockFetcher::get` timing at BlockFetcher.hpp:280-322:
        //   tGetStart = now();                            // line 280
        //   ... wait on future ...
        //   tFutureGetStart = now();                      // line 311
        //   queuedResult.wait_for(1ms)*                   // line 314
        //   futureGetDuration = duration(tFutureGetStart);
        //   m_statistics.futureWaitTotalTime += futureGetDuration;  // line 324
        //   m_statistics.getTotalTime += duration(tGetStart);       // line 325
        let t_get_start = std::time::Instant::now();

        if let Some(v) = self.get_if_available(&block_offset) {
            self.statistics
                .base
                .add_get_time(t_get_start.elapsed().as_secs_f64());
            return (Ok(v), 0);
        }

        let rx = match self.take_prefetch(&block_offset) {
            Some(existing) => existing,
            None => {
                self.record_on_demand_fetch();
                submit(block_offset.clone())
            }
        };

        // BlockFetcher.hpp:297-299 — `if ( !lastFetchedIndex || ... )
        // prefetchNewBlocks(...)`. The caller has already invoked
        // `record_fetch(idx)` before this method, so the "index
        // changed" condition is computed there and passed as
        // `should_drive_prefetch`.
        let mut prefetched = 0usize;
        if should_drive_prefetch {
            prefetched += self.prefetch_new_blocks(
                &lookup_block_offset,
                &lookup_next_block_offset,
                &submit_for_prefetch,
                &is_finalized_and_index_too_high,
                &partition_offset_for,
            );
        }

        // BlockFetcher.hpp:312-316 — `while ( queuedResult.wait_for(1ms)
        // == timeout ) prefetchNewBlocks(...)`. We approximate with
        // mpsc::Receiver::recv_timeout in a short loop so the prefetch
        // queue keeps filling while we wait. The 1ms tick matches
        // vendor exactly.
        let t_future_wait_start = std::time::Instant::now();
        let value = loop {
            match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(Ok(v)) => break v,
                Ok(Err(e)) => {
                    self.statistics
                        .base
                        .add_future_wait_time(t_future_wait_start.elapsed().as_secs_f64());
                    self.statistics
                        .base
                        .add_get_time(t_get_start.elapsed().as_secs_f64());
                    return (Err(e), prefetched);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if should_drive_prefetch {
                        prefetched += self.prefetch_new_blocks(
                            &lookup_block_offset,
                            &lookup_next_block_offset,
                            &submit_for_prefetch,
                            &is_finalized_and_index_too_high,
                            &partition_offset_for,
                        );
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    panic!(
                        "block_fetcher::get_with_prefetch: dispatch worker dropped reply \
                         (broken promise)"
                    );
                }
            }
        };
        self.statistics
            .base
            .add_future_wait_time(t_future_wait_start.elapsed().as_secs_f64());
        // Lever G: do NOT insert into the cache after on-demand fetch.
        // See note at `try_take_prefetched` — the cache-insert held a
        // second Arc ref forcing the consumer's `Arc::try_unwrap` to
        // fail and deep-clone the ~MB-sized ChunkData. Vendor's cache
        // insert is only used to satisfy CONCURRENT lookups against
        // the same key, which the single-pass consumer never issues.
        self.statistics
            .base
            .add_get_time(t_get_start.elapsed().as_secs_f64());
        (Ok(value), prefetched)
    }

    /// Pure cache lookup (no fetch). Checks prefetch cache first; on
    /// hit, MOVES the entry into the main cache (mirrors rapidgzip's
    /// `takeFromPrefetchQueue` pattern at BlockFetcher.hpp:385-410 —
    /// "take" semantically removes from prefetch queue).
    pub fn get_if_available(&self, block_offset: &Key) -> Option<Value> {
        // Vendor BlockFetcher.hpp:280 — `processReadyPrefetches()` is
        // called first thing in `get`. Drains completed prefetches
        // from the in-flight map into the prefetch_cache so the
        // cache lookup below has a chance of hitting.
        self.process_ready_prefetches();
        // Try the prefetch cache first.
        {
            let mut pc = self.prefetch_cache.lock().unwrap();
            if let Some(v) = pc.get(block_offset) {
                self.statistics.base.record_prefetch_cache_hit(true);
                // cache.get_outcome: per-lookup result. Pairs against
                // each `coord.prefetch_emit` span so the diff can
                // attribute every emitted prefetch to either a
                // consume ("source":"prefetch") or a wasted decode
                // (no matching cache.get_outcome → discarded by
                // clear_prefetch_cache at end-of-decode).
                crate::decompress::parallel::trace_v2::emit_instant(
                    "cache.get_outcome",
                    &format!(r#""source":"prefetch","offset":{:?}"#, block_offset),
                    "t",
                );
                // Lever G: previously this evicted from prefetch_cache
                // and PROMOTED a clone into self.cache for "subsequent
                // gets". The single-pass forward consumer never re-gets
                // the same key, so the promotion held a redundant Arc
                // ref that forced the consumer's `Arc::try_unwrap` to
                // deep-clone (~7ms × 24 chunks). Drop the promote —
                // just evict from prefetch_cache and return.
                pc.evict(block_offset);
                crate::decompress::parallel::chunk_data::lc_set(
                    &crate::decompress::parallel::chunk_data::LC_G_PREFETCH,
                    pc.size(),
                );
                return Some(v);
            }
        }
        // Fall through to the main cache.
        let mut c = self.cache.lock().unwrap();
        match c.get(block_offset) {
            Some(v) => {
                self.statistics.base.record_get();
                crate::decompress::parallel::trace_v2::emit_instant(
                    "cache.get_outcome",
                    &format!(r#""source":"main","offset":{:?}"#, block_offset),
                    "t",
                );
                Some(v)
            }
            None => {
                // Vendor BlockFetcher.hpp:263-277 — `getFromCaches`
                // returns no cached value; the on-demand path will
                // dispatch a fresh decode. Records the miss for stats
                // visibility (cache hit rate = hits / (hits + misses)).
                self.statistics.base.record_prefetch_cache_miss();
                crate::decompress::parallel::trace_v2::emit_instant(
                    "cache.get_outcome",
                    &format!(r#""source":"miss","offset":{:?}"#, block_offset),
                    "t",
                );
                None
            }
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
        crate::decompress::parallel::chunk_data::lc_set(
            &crate::decompress::parallel::chunk_data::LC_G_MAIN,
            c.size(),
        );
    }

    /// Insert a prefetched block into the prefetch cache. Stats:
    /// records the prefetch (rapidgzip's BlockFetcher tracks this via
    /// `prefetchCount` increment on prefetch submission).
    #[allow(dead_code)] // vendor parity or unit-test surface
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

    /// Take the in-flight prefetch receiver for `block_offset` if one
    /// exists. Mirror of vendor's `takeFromPrefetchQueue` exact-match
    /// branch at vendor/.../core/BlockFetcher.hpp:385-410.
    pub fn take_prefetch(&self, block_offset: &Key) -> Option<Receiver<Result<Value, Err>>> {
        self.prefetching.lock().unwrap().remove(block_offset)
    }

    /// True iff a prefetch task at `block_offset` is in flight.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn prefetch_in_flight(&self, block_offset: &Key) -> bool {
        self.prefetching.lock().unwrap().contains_key(block_offset)
    }

    /// Record an in-flight prefetch. The receiver lives in the prefetch
    /// queue and is consumed by `take_prefetch` on the consumer's
    /// `BlockFetcher::get` call. Mirror of vendor's
    /// `m_prefetching.emplace(*prefetchBlockOffset, std::move(prefetchedFuture))`
    /// at BlockFetcher.hpp:558.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn submit_prefetch(&self, block_offset: Key, rx: Receiver<Result<Value, Err>>) {
        self.prefetching.lock().unwrap().insert(block_offset, rx);
    }

    /// Drop any in-flight prefetch receivers that match the predicate.
    /// Mirror of vendor's `processReadyPrefetches` cleanup
    /// (BlockFetcher.hpp:431-450) — receivers whose consumer no longer
    /// cares are dropped, the worker's `reply.send` then fails silently.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn drop_prefetches_matching<F: FnMut(&Key) -> bool>(&self, mut pred: F) {
        self.prefetching.lock().unwrap().retain(|k, _| !pred(k));
    }

    /// Number of in-flight prefetches. Used by callers' own prefetch
    /// throttle to mirror vendor's `m_prefetching.size()` reads.
    pub fn prefetching_len(&self) -> usize {
        self.prefetching.lock().unwrap().len()
    }

    /// Snapshot of the in-flight prefetch keys.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn prefetching_keys(&self) -> Vec<Key> {
        self.prefetching.lock().unwrap().keys().cloned().collect()
    }

    /// Snapshot of `(key, value)` pairs in the prefetch cache, sorted by
    /// key. Does NOT touch LRU/stats and does NOT evict — the consumer's
    /// later `get_if_available` still finds the entries. Mirror of
    /// vendor's `prefetchCache().contents()` consumed by
    /// `queuePrefetchedChunkPostProcessing` (GzipChunkFetcher.hpp:524-528).
    #[cfg_attr(not(parallel_sm), allow(dead_code))]
    pub fn prefetch_cache_contents_sorted(&self) -> Vec<(Key, Value)> {
        let mut v = self.prefetch_cache.lock().unwrap().contents_snapshot();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Vendor's `m_parallelization` cap on simultaneous prefetches
    /// (BlockFetcher.hpp:467: `m_threadPool.capacity()`).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn parallelization(&self) -> usize {
        self.parallelization
    }

    /// Mark a block offset as a failed prefetch (rapidgzip
    /// `m_failedPrefetchCache`). Future `is_failed_prefetch` lookups
    /// return true so the caller doesn't re-issue.
    #[allow(dead_code)] // vendor parity or unit-test surface
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

    /// Notify the fetching strategy that a chunk at `index_to_split`
    /// has expanded into `split_count` sequential subchunk indexes.
    /// Mirror of vendor's call at
    /// `GzipChunkFetcher::appendSubchunksToIndexes`
    /// (`vendor/.../rapidgzip/GzipChunkFetcher.hpp:382`):
    ///
    /// ```cpp
    /// if ( subchunks.size() > 1 ) {
    ///     BaseType::m_fetchingStrategy.splitIndex(
    ///         m_nextUnprocessedBlockIndex, subchunks.size() );
    /// }
    /// ```
    ///
    /// Without this call the strategy's `previous_indexes` /
    /// `last_fetched` accounting stays in CHUNK units while the
    /// BlockFinder accumulates SUBCHUNK indexes, and the next
    /// `prefetch_new_blocks` call queries indexes that are already
    /// in `block_offsets` — returning confirmed sub-partition
    /// offsets that get emitted as wasted sub-partition prefetches.
    /// Falsification commit aba6b59 showed ~50 ms wall savings on
    /// silesia-large 16T from suppressing those emits via env-gated
    /// skip; this is the upstream fix that makes the emits never
    /// generated in the first place.
    pub fn split_index(&self, index_to_split: usize, split_count: usize) {
        self.fetching_strategy
            .lock()
            .unwrap()
            .split_index(index_to_split, split_count);
    }

    /// Fill the prefetch queue (`m_prefetching`) with up to
    /// `parallelization - 1` futures for the next expected block
    /// indices. Literal port of `BlockFetcher::prefetchNewBlocks`
    /// at vendor/.../core/BlockFetcher.hpp:458-572.
    ///
    /// `lookup_block_offset(index) -> Option<Key>` mirrors vendor's
    /// `m_blockFinder->get(blockIndexToPrefetch, /* timeout */ 0)` at
    /// BlockFetcher.hpp:479 — returns the block-offset for the given
    /// data block index, or `None` if the BlockFinder hasn't confirmed
    /// it yet (we treat that as "skip this index").
    ///
    /// `submit_for(offset) -> Receiver<Result<Value, Err>>` mirrors
    /// vendor's `m_threadPool.submit([this, offset, nextOffset] () {
    /// return decodeAndMeasureBlock(offset, nextOffset); }, /* priority */ 0)`
    /// at BlockFetcher.hpp:554-557. The caller wraps the gzip-specific
    /// task body.
    ///
    /// `is_finalized_and_index_too_high(idx)` mirrors vendor's
    /// `m_blockFinder->finalized() && (idx >= m_blockFinder->size())`
    /// check at BlockFetcher.hpp:504 — once the block finder has been
    /// finalized, indices beyond `size()` are dropped.
    ///
    /// The `stop_prefetching` predicate (vendor's `stopPrefetching` at
    /// BlockFetcher.hpp:455) is omitted because gzippy's BlockFinder
    /// `get` is non-blocking by construction (no `wait_for_timeout`
    /// equivalent today) — the vendor uses it only to time-bound the
    /// inner `m_blockFinder->get(idx, 0.0001s)` poll loop.
    ///
    /// Returns the number of new prefetch tasks submitted (for stats /
    /// trace purposes).
    pub fn prefetch_new_blocks<L, LN, S, F, PO>(
        &self,
        lookup_block_offset: L,
        lookup_next_block_offset: LN,
        submit_for: S,
        is_finalized_and_index_too_high: F,
        partition_offset_for: PO,
    ) -> usize
    where
        L: Fn(usize) -> Option<Key>,
        // Vendor BlockFetcher.hpp:533-535 differentiates between the
        // CURRENT index's offset lookup (SUCCESS-only — failure means
        // skip) and the NEXT index's offset lookup (the worker's stop
        // hint, which may be `file_size_in_bits` for the last chunk).
        // gzippy's prior single-`lookup_block_offset` collapsed both
        // behind the SUCCESS-only check, so the LAST prefetch in a
        // file was always skipped because `lookup(last_idx + 1)`
        // returned `GetReturnCode::Failure` even though the offset
        // value (`file_size_in_bits`) was a perfectly usable stop hint.
        // Asymmetric split lets `lookup_next_block_offset` accept the
        // file-size sentinel.
        LN: Fn(usize) -> Option<Key>,
        // Mirror of vendor's `decodeAndMeasureBlock(offset, nextOffset)`
        // capture at BlockFetcher.hpp:555-557 — both the prefetched
        // block's offset AND the next block's offset are captured in
        // the worker task closure so the worker knows where to stop.
        S: Fn(Key, Key) -> Receiver<Result<Value, Err>>,
        F: Fn(usize) -> bool,
        PO: Fn(&Key) -> Key,
    {
        PREFETCH_NEW_BLOCKS_CALLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        use crate::decompress::parallel::trace_v2;

        // coord.prefetch_call span: wraps the entire prefetch decision
        // loop. End-of-span outcome is emitted as a separate instant
        // event ("coord.prefetch_call.outcome") so the SpanGuard name
        // can stay 'static while args carry per-call detail.
        let _tv2_call = trace_v2::SpanGuard::begin("coord.prefetch_call");
        let prefetching_len_at_entry = self.prefetching_len();

        // BlockFetcher.hpp:463 — processReadyPrefetches() before new dispatch.
        self.process_ready_prefetches();

        // BlockFetcher.hpp:465-472 — threadPoolSaturated() gate.
        // Vendor: `m_prefetching.size() + 1 >= m_threadPool.capacity()`.
        // The "+1" accounts for the on-demand task that consumer is
        // currently waiting on. We use the configured parallelization
        // as the capacity (same as `m_threadPool.capacity()` since the
        // pool is sized to that).
        let thread_pool_saturated = || self.prefetching_len() + 1 >= self.parallelization;

        if thread_pool_saturated() {
            PREFETCH_RETURN_SATURATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            trace_v2::emit_instant(
                "coord.prefetch_skip",
                &format!(
                    r#""reason":"saturated_entry","prefetching_len":{prefetching_len_at_entry},"parallelization":{}"#,
                    self.parallelization
                ),
                "t",
            );
            trace_v2::emit_instant(
                "coord.prefetch_call.outcome",
                r#""submitted":0,"early_exit":"saturated_entry""#,
                "t",
            );
            return 0;
        }

        // BlockFetcher.hpp:474 — `m_fetchingStrategy.prefetch(m_prefetchCache.capacity())`.
        let block_indexes_to_prefetch = {
            let strategy = self.fetching_strategy.lock().unwrap();
            // Vendor uses `m_prefetchCache.capacity()` as the upper
            // bound; the prefetch_cache cap equals the parallelization
            // setting in our constructor sites.
            let cap = self.prefetch_cache.lock().unwrap().capacity();
            strategy.prefetch(cap)
        };
        let strategy_candidates = block_indexes_to_prefetch.len();
        trace_v2::emit_instant(
            "coord.prefetch_strategy",
            &format!(
                r#""candidates":{strategy_candidates},"prefetching_len":{prefetching_len_at_entry}"#
            ),
            "t",
        );

        // BlockFetcher.hpp:476-491 — resolve the prefetch INDEXES to a
        // `blockOffsetsToPrefetch` set up-front (including partition
        // offsets) so we can (a) protect them from eviction via touch
        // below and (b) reference them in the cache-pollution stop in
        // the loop. Vendor builds this with the timeout-0 lookup; we use
        // the same `lookup_block_offset` the loop uses.
        let mut block_offsets_to_prefetch: Vec<Key> =
            Vec::with_capacity(block_indexes_to_prefetch.len());
        for &index in &block_indexes_to_prefetch {
            if let Some(off) = lookup_block_offset(index) {
                let partition_offset = partition_offset_for(&off);
                if partition_offset != off {
                    block_offsets_to_prefetch.push(partition_offset);
                }
                block_offsets_to_prefetch.push(off);
            }
        }

        // BlockFetcher.hpp:493-497 — touch all to-be-prefetched offsets
        // (reverse order, vendor) in BOTH caches so that prefetching one
        // block cannot evict another block we also intend to prefetch.
        // The touch makes the to-be-prefetched set most-recently-used, so
        // `next_nth_eviction` below returns a genuinely-stale candidate.
        // Lock order is cache-then-prefetch_cache to match `test()`
        // (block_fetcher.rs:111-113) and avoid a deadlock.
        {
            let mut c = self.cache.lock().unwrap();
            let mut pc = self.prefetch_cache.lock().unwrap();
            for off in block_offsets_to_prefetch.iter().rev() {
                c.touch(off);
                pc.touch(off);
            }
        }

        let mut submitted = 0usize;
        for index in block_indexes_to_prefetch {
            // BlockFetcher.hpp:500-502 — stop when the pool is full.
            if thread_pool_saturated() {
                trace_v2::emit_instant(
                    "coord.prefetch_skip",
                    &format!(
                        r#""reason":"saturated_loop","index":{index},"prefetching_len":{}"#,
                        self.prefetching_len()
                    ),
                    "t",
                );
                break;
            }

            // BlockFetcher.hpp:504-506 — drop indices past finalize.
            if is_finalized_and_index_too_high(index) {
                trace_v2::emit_instant(
                    "coord.prefetch_skip",
                    &format!(r#""reason":"finalized_too_high","index":{index}"#),
                    "t",
                );
                continue;
            }

            // BlockFetcher.hpp:479 — `m_blockFinder->get(idx, /* timeout */ 0)`.
            let prefetch_block_offset = match lookup_block_offset(index) {
                Some(o) => o,
                None => {
                    // Vendor BlockFetcher.hpp:533 — skip when no offset.
                    trace_v2::emit_instant(
                        "coord.prefetch_skip",
                        &format!(r#""reason":"lookup_miss","index":{index}"#),
                        "t",
                    );
                    continue;
                }
            };

            // BlockFetcher.hpp:536-539 — skip if already cached, in
            // prefetch queue, or in failed-prefetch list. Vendor also
            // checks `isInCacheOrQueue(getPartitionOffsetFromOffset(*prefetchBlockOffset))`
            // at line 537-538 to avoid double-submitting a prefetch
            // that was already submitted under its partition offset.
            if self.test(&prefetch_block_offset) || self.is_failed_prefetch(&prefetch_block_offset)
            {
                trace_v2::emit_instant(
                    "coord.prefetch_skip",
                    r#""reason":"in_cache_or_queue""#,
                    "t",
                );
                continue;
            }
            let partition_offset = partition_offset_for(&prefetch_block_offset);
            if partition_offset != prefetch_block_offset && self.test(&partition_offset) {
                trace_v2::emit_instant(
                    "coord.prefetch_skip",
                    r#""reason":"partition_in_cache""#,
                    "t",
                );
                continue;
            }

            // BlockFetcher.hpp:519-520 — `m_blockFinder->get(idx + 1, ...)`
            // to compute nextOffset (the worker's stop hint).
            // BlockFetcher.hpp:535 — drop if nextOffset is missing.
            //
            // Uses `lookup_next_block_offset` (NOT `lookup_block_offset`)
            // because vendor accepts `file_size_in_bits` here as the
            // stop hint for the last chunk in the file. The current-
            // index lookup above remains SUCCESS-only.
            let next_prefetch_block_offset = match lookup_next_block_offset(index + 1) {
                Some(o) => o,
                None => {
                    trace_v2::emit_instant(
                        "coord.prefetch_skip",
                        &format!(r#""reason":"next_offset_missing","index":{index}"#),
                        "t",
                    );
                    continue;
                }
            };

            // BlockFetcher.hpp:544-551 — avoid cache pollution: if
            // submitting this prefetch would (on the (prefetching+1)-th
            // hypothetical insert — the "+1" for the on-demand task the
            // consumer is waiting on, plus the in-flight prefetches that
            // also insert before our eviction of interest) evict a block
            // we ourselves intend to prefetch, STOP. There is no point
            // prefetching block X if doing so evicts block Y we also want.
            if let Some(offset_to_be_evicted) = self
                .prefetch_cache
                .lock()
                .unwrap()
                .next_nth_eviction(self.prefetching_len() + 1)
            {
                if block_offsets_to_prefetch.contains(&offset_to_be_evicted) {
                    PREFETCH_CACHE_POLLUTION_STOPS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    trace_v2::emit_instant(
                        "coord.prefetch_skip",
                        r#""reason":"cache_pollution_stop""#,
                        "t",
                    );
                    break;
                }
            }

            // BlockFetcher.hpp:553-557 — submit prefetch task.
            // coord.prefetch_emit span: per-emission, one B/E pair with
            // rich args. The vendor patch (patch_vendor.sh) wraps the
            // equivalent `m_threadPool.submit([..]decodeAndMeasureBlock..)`
            // at BlockFetcher.hpp:554-557 with the same span name and
            // matching arg keys, so timeline_analyze.py can diff them
            // emission-for-emission.
            //
            // `offset_eq_partition`: true means this is a
            // partition-aligned prefetch (the "main" one for a
            // partition); false means a sub-partition speculative
            // prefetch — those are the candidates for the missing
            // vendor `nextNthEviction` cache-pollution guard
            // (BlockFetcher.hpp:544-551).
            let _tv2_emit = {
                use std::fmt::Write as _;
                let mut args = String::with_capacity(160);
                let _ = write!(
                    args,
                    r#""index":{index},"offset":{:?},"partition_offset":{:?},"next_offset":{:?},"offset_eq_partition":{}"#,
                    prefetch_block_offset,
                    partition_offset,
                    next_prefetch_block_offset,
                    partition_offset == prefetch_block_offset
                );
                trace_v2::SpanGuard::begin_with("coord.prefetch_emit", &args)
            };
            self.statistics.base.record_prefetch();
            let rx = submit_for(prefetch_block_offset.clone(), next_prefetch_block_offset);
            // BlockFetcher.hpp:558 — `m_prefetching.emplace(offset, std::move(future))`.
            // We bypass `submit_prefetch` here to avoid double-counting
            // record_prefetch (already done above).
            {
                let mut p = self.prefetching.lock().unwrap();
                p.insert(prefetch_block_offset, rx);
                crate::decompress::parallel::chunk_data::lc_set(
                    &crate::decompress::parallel::chunk_data::LC_G_PREFETCHING,
                    p.len(),
                );
            }
            submitted += 1;
        }

        if submitted == 0 {
            PREFETCH_RETURN_ZERO_SUBMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            PREFETCH_TOTAL_SUBMITTED
                .fetch_add(submitted as u64, std::sync::atomic::Ordering::Relaxed);
            PREFETCH_RETURN_SUBMITTED_ANY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        trace_v2::emit_instant(
            "coord.prefetch_call.outcome",
            &format!(
                r#""submitted":{submitted},"candidates":{strategy_candidates},"prefetching_len_after":{}"#,
                self.prefetching_len()
            ),
            "t",
        );
        submitted
    }

    /// Ask the fetching strategy which indexes to prefetch.
    #[allow(dead_code)] // vendor parity or unit-test surface
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

    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Vendor's `processReadyPrefetches` (BlockFetcher.hpp:463) ported.
    /// Polls every in-flight prefetch receiver non-blockingly via
    /// `try_recv`; for each that has a ready value, moves it from the
    /// `prefetching` map into the `prefetch_cache`. After this call,
    /// `get_if_available` can return ready prefetches via the
    /// prefetch-cache fast path without blocking.
    ///
    /// Before this method existed, gzippy's prefetch results stayed in
    /// the `prefetching: HashMap<Key, Receiver>` map until the
    /// consumer called `try_take_prefetched` (which blocked on
    /// `recv()` even if the result was ready) and then inserted
    /// directly into the MAIN cache. The prefetch_cache was always
    /// empty → `Cache Hit Rate` in --verbose always reported 0%. The
    /// underlying chunk was still usable but the stats were
    /// misleading.
    pub fn process_ready_prefetches(&self) -> usize {
        let mut moved = 0usize;
        let mut prefetching = self.prefetching.lock().unwrap();
        let keys: Vec<Key> = prefetching.keys().cloned().collect();
        for key in keys {
            if let Some(rx) = prefetching.get(&key) {
                match rx.try_recv() {
                    Ok(Ok(value)) => {
                        prefetching.remove(&key);
                        // Insert into prefetch cache.
                        drop(prefetching);
                        {
                            let mut pc = self.prefetch_cache.lock().unwrap();
                            pc.insert(key, value);
                            crate::decompress::parallel::chunk_data::lc_set(
                                &crate::decompress::parallel::chunk_data::LC_G_PREFETCH,
                                pc.size(),
                            );
                        }
                        prefetching = self.prefetching.lock().unwrap();
                        moved += 1;
                    }
                    Ok(Err(_)) => {
                        // Worker reported failure; drop the receiver
                        // and let downstream re-dispatch on-demand.
                        prefetching.remove(&key);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => { /* still in flight */ }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        // Worker pool dropped the sender without
                        // replying — treat as failure.
                        prefetching.remove(&key);
                    }
                }
            }
        }
        crate::decompress::parallel::chunk_data::lc_set(
            &crate::decompress::parallel::chunk_data::LC_G_PREFETCHING,
            prefetching.len(),
        );
        moved
    }

    pub fn clear_prefetch_cache(&self) {
        let mut pc = self.prefetch_cache.lock().unwrap();
        // Vendor BlockFetcher.hpp:199-201 — entries still in the
        // prefetch cache at destruction time are "never used"
        // (`get_if_available` removes promoted entries on hit), so
        // each remaining one counts as a cache_unused_entry. Mirrors
        // the LRU eviction-callback path vendor uses.
        let unused = pc.size();
        // cache.discard_unused: emit per-entry so the diff can show
        // exactly which offsets in `coord.prefetch_emit` were never
        // consumed. Sum equals the (count of `coord.prefetch_emit`
        // spans) - (count of `cache.get_outcome source=prefetch`
        // events) modulo any still-in-flight in `prefetching`.
        crate::decompress::parallel::trace_v2::emit_instant(
            "cache.discard_unused.summary",
            &format!(r#""unused_count":{unused}"#),
            "t",
        );
        for _ in 0..unused {
            self.statistics.base.record_cache_unused_entry();
        }
        pc.clear();
    }

    /// Lock + snapshot both caches' statistics + the chunk-extra
    /// counters. Mirror of `BlockFetcher::statistics`
    /// (BlockFetcher.hpp:337-348).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn cache_statistics(
        &self,
    ) -> (super::cache::CacheStatistics, super::cache::CacheStatistics) {
        (
            self.cache.lock().unwrap().statistics(),
            self.prefetch_cache.lock().unwrap().statistics(),
        )
    }

    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().size()
    }

    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn prefetch_cache_size(&self) -> usize {
        self.prefetch_cache.lock().unwrap().size()
    }

    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn in_flight_count(&self) -> usize {
        self.prefetching.lock().unwrap().len()
    }
}

/// Counters for `prefetch_new_blocks` outcomes — surfaced in
/// `--verbose` stats. Used to disambiguate "workers idle because
/// dispatch wasn't called" vs "dispatch was called but no-op'd because
/// the pool was already saturated / no indexes to submit".
pub static PREFETCH_NEW_BLOCKS_CALLED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PREFETCH_RETURN_SATURATED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PREFETCH_RETURN_ZERO_SUBMITTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PREFETCH_RETURN_SUBMITTED_ANY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PREFETCH_TOTAL_SUBMITTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Counts firings of the vendor cache-pollution stop (BlockFetcher.hpp:544-551):
/// a prefetch refused because submitting it would evict a block we intend to
/// prefetch. Non-zero ⇒ the admission control is live (falsifier check for
/// Divergence #4). See git history (campaign plan, removed).
pub static PREFETCH_CACHE_POLLUTION_STOPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use super::super::prefetcher::FetchNextAdaptive;
    use super::*;
    use std::sync::mpsc;

    fn new_fetcher() -> BlockFetcher<u64, String, FetchNextAdaptive, &'static str> {
        BlockFetcher::new(4, 4, FetchNextAdaptive::new(3), 4)
    }

    /// Helper: build a `Receiver` already populated with the given
    /// `Result`. Used to simulate a worker that has already run by the
    /// time `get` consults the prefetch queue (the dispatch closure runs
    /// inline in these tests since there's no real worker pool).
    fn ready_rx<T, E>(result: Result<T, E>) -> Receiver<Result<T, E>>
    where
        T: Send + 'static,
        E: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(result);
        rx
    }

    #[test]
    fn get_if_available_returns_none_when_empty() {
        let bf = new_fetcher();
        assert_eq!(bf.get_if_available(&100), None);
    }

    #[test]
    fn get_if_available_records_cache_miss() {
        // Vendor BlockFetcher.hpp:263-277: when getFromCaches returns
        // no cached value, the miss is recorded for cache-hit-rate
        // visibility in --verbose. Asserts `record_prefetch_cache_miss`
        // is wired at the `None` arm of `get_if_available`.
        let bf = new_fetcher();
        let _ = bf.get_if_available(&42);
        let _ = bf.get_if_available(&99);
        let snap = bf.statistics.base.snapshot();
        assert_eq!(snap.prefetch_cache_misses, 2);
    }

    #[test]
    fn clear_prefetch_cache_counts_unused_entries() {
        // Vendor BlockFetcher.hpp:199-201: entries left in the prefetch
        // cache at destruction count as `record_cache_unused_entry`.
        // Asserts `clear_prefetch_cache` drains and counts.
        let bf = new_fetcher();
        bf.insert_prefetched(100, "pre-100".into());
        bf.insert_prefetched(200, "pre-200".into());
        bf.insert_prefetched(300, "pre-300".into());
        // Promote one (simulates the consumer using it).
        let _ = bf.get_if_available(&200);
        assert_eq!(bf.prefetch_cache_size(), 2);
        bf.clear_prefetch_cache();
        let snap = bf.statistics.base.snapshot();
        assert_eq!(snap.prefetch_cache_unused_entries, 2);
        assert_eq!(bf.prefetch_cache_size(), 0);
    }

    #[test]
    fn insert_then_get_returns_value() {
        let bf = new_fetcher();
        bf.insert(100, "block-100".into());
        assert_eq!(bf.get_if_available(&100), Some("block-100".into()));
    }

    #[test]
    fn prefetched_block_is_evicted_on_get_without_promotion() {
        // Lever G (commit 4890e81): the single-pass forward consumer never
        // re-gets the same key, so the old prefetch→main-cache PROMOTE held
        // a redundant Arc ref that forced the consumer's `Arc::try_unwrap`
        // to deep-clone (~7ms × 24 chunks). Behavior changed: `get_if_available`
        // on a prefetch hit now EVICTS from prefetch_cache and returns the
        // value WITHOUT promoting to the main cache.
        let bf = new_fetcher();
        bf.insert_prefetched(200, "pre-200".into());
        assert_eq!(bf.prefetch_cache_size(), 1);
        let v = bf.get_if_available(&200);
        assert_eq!(v, Some("pre-200".into()));
        // Post-Lever-G: prefetch evicted, main cache NOT populated.
        assert_eq!(bf.cache_size(), 0);
    }

    #[test]
    fn test_returns_true_for_in_flight_prefetch() {
        let bf = new_fetcher();
        bf.submit_prefetch(300, ready_rx::<String, &'static str>(Ok("p300".into())));
        assert!(bf.test(&300));
        let _ = bf.take_prefetch(&300);
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
        // Lever G (4890e81): no promotion. The second `get_if_available`
        // after the first one drained the prefetch_cache returns None
        // — it's a miss, not a main-cache hit. Statistics still record
        // the original prefetch insertion + the one prefetch hit.
        let bf = new_fetcher();
        bf.insert_prefetched(500, "p".into());
        let _ = bf.get_if_available(&500); // prefetch hit, evicts
        let _ = bf.get_if_available(&500); // miss (no promotion under Lever G)
        let snap = bf.statistics.base.snapshot();
        assert!(snap.prefetch_count >= 1);
        assert!(snap.prefetch_cache_hits >= 1);
        // Prefetch hits do not touch the main cache (`record_get` is main-cache only).
    }

    #[test]
    fn get_returns_cached_value_without_dispatch() {
        let bf = new_fetcher();
        bf.insert(700, "cached".into());
        let mut dispatched = false;
        let v = bf
            .get(700, |_k: u64| -> Receiver<Result<String, &'static str>> {
                dispatched = true;
                ready_rx(Ok("dispatched".into()))
            })
            .unwrap();
        assert_eq!(v, "cached");
        assert!(!dispatched, "dispatch should not run on cache hit");
    }

    #[test]
    fn get_invokes_dispatch_on_miss_and_caches_result() {
        let bf = new_fetcher();
        let v = bf
            .get(800, |_k: u64| -> Receiver<Result<String, &'static str>> {
                ready_rx(Ok("from-dispatch".into()))
            })
            .unwrap();
        assert_eq!(v, "from-dispatch");
        // Mirror of rapidgzip BlockFetcher.hpp:320 (`insertIntoCache`
        // after dispatch): the freshly-dispatched value is now in the
        // main cache. A subsequent `get` for the same key must short-
        // circuit without invoking the dispatch closure.
        let mut dispatched_again = false;
        let v2 = bf
            .get(800, |_k: u64| -> Receiver<Result<String, &'static str>> {
                dispatched_again = true;
                ready_rx(Ok("re-dispatched".into()))
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
    fn get_takes_prefetched_future_on_hit() {
        // Mirror of vendor's `takeFromPrefetchQueue` exact-match branch
        // at vendor/.../core/BlockFetcher.hpp:385-410.
        let bf = new_fetcher();
        bf.submit_prefetch(900, ready_rx(Ok("prefetched".into())));
        let mut dispatched = false;
        let v = bf
            .get(900, |_k: u64| -> Receiver<Result<String, &'static str>> {
                dispatched = true;
                ready_rx(Ok("should-not-run".into()))
            })
            .unwrap();
        assert_eq!(v, "prefetched");
        assert!(!dispatched);
        // Prefetch receiver was consumed. Lever G (4890e81): no promotion
        // to main cache — single-pass forward consumer never re-gets the
        // same key, so the prefetch→main-cache promote was holding a
        // redundant Arc ref. Post-Lever-G, the in-flight receiver is
        // drained and the result is returned WITHOUT entering main cache.
        assert!(!bf.prefetch_in_flight(&900));
        assert_eq!(bf.cache_size(), 0);
    }

    #[test]
    fn get_records_on_demand_fetch_on_miss() {
        let bf = new_fetcher();
        let before = bf.statistics.base.snapshot().on_demand_fetch_count;
        let _v = bf
            .get(1000, |_k: u64| -> Receiver<Result<String, &'static str>> {
                ready_rx(Ok("x".into()))
            })
            .unwrap();
        let after = bf.statistics.base.snapshot().on_demand_fetch_count;
        assert!(after > before);
    }

    #[test]
    fn get_propagates_dispatch_error_without_caching() {
        let bf = new_fetcher();
        let err = bf
            .get(1100, |_k: u64| -> Receiver<Result<String, &'static str>> {
                ready_rx(Err("boom"))
            })
            .unwrap_err();
        assert_eq!(err, "boom");
        // Error must NOT be cached: a second get with a successful
        // dispatch should run it.
        let ok = bf
            .get(1100, |_k: u64| -> Receiver<Result<String, &'static str>> {
                ready_rx(Ok("recovered".into()))
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
