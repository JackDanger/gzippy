# Vendor pacing map: how rapidgzip keeps its consumer ~0–17ms behind its prefetcher

Read-only source analysis. Vendor = `vendor/rapidgzip/librapidarchive/src/`.
gzippy = `src/decompress/parallel/`. Every structural claim cites `file:line`.

Context being explained: gzippy's consumer reaches a partition chunk ~318ms after
it was prefetched (chunk evicted / cold re-decoded); rapidgzip's consumer stays
~0–17ms behind. Goal: map rapidgzip's pacing structure and locate the gzippy delta.

---

## 1. PREFETCH DEPTH / WINDOW

**rapidgzip prefetch depth is bounded by the thread pool, NOT a fixed window, and the
COUNT it asks the strategy for is the prefetch-cache capacity.**

- The depth cap on simultaneously in-flight prefetches is `parallelization - 1`:
  `prefetchNewBlocks` stops when `m_prefetching.size() + 1 >= m_threadPool.capacity()`
  (`core/BlockFetcher.hpp:465-468`, re-checked in the loop at `:500-502`). The "+1"
  reserves a slot for the on-demand chunk the consumer is currently waiting on.
  Pool capacity == `m_parallelization` (`:185`, `m_threadPool( ... m_parallelization )`).
- The *number of indexes requested from the strategy* per pump is
  `m_prefetchCache.capacity()` = `2 * m_parallelization` (`:474` calls
  `m_fetchingStrategy.prefetch( m_prefetchCache.capacity() )`; capacity set at `:182`).
- Depth IS adaptive in the *index* dimension: `FetchNextAdaptive::prefetch` returns
  `2^(consecutiveRatio * log2(maxExtrapolation))` indexes — exponential ramp from 1
  (random access) to the full `maxAmountToPrefetch` (fully sequential)
  (`core/Prefetcher.hpp:126-146`, `:187-191`). For the perfectly-sequential single-
  member streaming case, `extrapolate` special-cases `size==1` to return the full
  `maxAmountToPrefetch` immediately (`Prefetcher.hpp:160-166`) — one cache miss then
  full-depth prefetch. The production strategy is `FetchNextAdaptive`/`FetchMultiStream`
  (`Prefetcher.hpp:88, 234`).

So the *effective look-ahead distance* is: up to `parallelization-1` chunks decoding
in flight at any instant, plus up to `2*parallelization` completed-but-unconsumed
chunks parked in `prefetchCache`. At T8 that is ~7 in flight + ~16 parked ≈ a ~16-chunk
ceiling on how far the frontier can run ahead of the consumer.

## 2. CONSUMER-JOINS-IN-FLIGHT

**The consumer never cold-starts a chunk that is already being decoded; it takes the
in-flight future and waits only for its remainder.**

- `m_prefetching` is a `std::map<blockOffset, std::future<BlockData>>` separate from
  both caches (`core/BlockFetcher.hpp:685`).
- `get()` first calls `getFromCaches` (`:263`), which calls `takeFromPrefetchQueue`
  (`:385`): an exact-offset match in `m_prefetching` MOVES the future out
  (`:410-414`, `m_prefetching.erase`) and returns it as `queuedResult`.
- The new-task dispatch at `:275-277` runs **only if** `!cachedResult.has_value() &&
  !queuedResult.valid()` — i.e. only when neither a cache hit nor an in-flight future
  exists. A matched in-flight future therefore suppresses the cold start.
- The wait is the *remainder* of that future: `while (queuedResult.wait_for(1ms) ==
  timeout) prefetchNewBlocks(...)` then `queuedResult.get()` (`:314-317`). If the
  prefetch is already done the wait is ~0; if it's mid-decode the consumer waits only
  the time left, never re-decoding from scratch.
- A COLD start happens only when the offset is in neither cache nor `m_prefetching`
  (`submitOnDemandTask`, `:573-589`). That is exactly the case gzippy hits at +318ms:
  the chunk was prefetched, *completed*, parked in `prefetchCache`, then **evicted**
  before the consumer's `get` arrived → cache miss → cold `submitOnDemandTask`.

## 3. RETAIN / EVICT POLICY

**Two structural guarantees keep a soon-needed chunk alive: (a) in-flight chunks live
in a map that is never evicted, and (b) an explicit cache-pollution stop refuses to
prefetch a block if doing so would evict a block we still intend to prefetch.**

- Cache caps: main `m_cache( max(16, m_parallelization) )` and prefetch
  `m_prefetchCache( 2 * m_parallelization )` (`core/BlockFetcher.hpp:181-182`).
- The in-flight `m_prefetching` map is SEPARATE from both caches (`:685`). A future in
  it cannot be evicted; `processReadyPrefetches` (`:426-450`) only moves it into
  `m_prefetchCache` *after* the future is ready (`:438`). So "in-flight" is structurally
  un-evictable; eviction risk begins only once a chunk has COMPLETED and entered the
  (capacity-`2*par`) prefetch cache.
- Anti-eviction touch: before prefetching, `prefetchNewBlocks` touches every
  to-be-prefetched offset in BOTH caches so prefetching block X cannot evict block Y
  also slated for prefetch (`:493-497`).
- Cache-pollution STOP: `if (offsetToBeEvicted = m_prefetchCache.nextNthEviction(
  m_prefetching.size()+1); contains(blockOffsetsToPrefetch, offsetToBeEvicted)) break;`
  (`:544-551`). It refuses to add a prefetch when the (prefetching+1)-th hypothetical
  insert would evict a block in the intended-prefetch set.
- BUT note: for SEQUENTIAL access, `insertIntoCache` CLEARS the main cache on every
  insert (`:355-358`). So for streaming single-member decode, the main cache is
  essentially a scratchpad; look-ahead retention lives almost entirely in
  `m_prefetching` (un-evictable) + `m_prefetchCache` (cap `2*par`, pollution-guarded).

Net: rapidgzip does NOT guarantee a *completed* prefetched chunk survives arbitrarily
long — only up to `2*par` completed chunks are retained. The guarantee is weaker than
"survives until consumed"; it is "survives as long as the consumer stays within ~`2*par`
chunks of the frontier." The consumer keeping pace is what makes that bound sufficient.

## 4. IN-ORDER vs OFF-PATH CONSUMER WORK

**The orchestrator (in-order) thread does only lightweight bookkeeping + ONE window
publish; the heavy per-chunk work (marker replacement / applyWindow / CRC) runs on the
pool. The consumer blocks on the *current* chunk's apply-future but overlaps every
other chunk's post-processing during that block.**

- `processNextChunk` (`rapidgzip/GzipChunkFetcher.hpp:311-362`): get block (`:329`),
  fetch last window (`:334-341`), `postProcessChunk` (`:343`), `setEncodedOffset`
  (`:349`), `appendSubchunksToIndexes` (`:357`).
- `postProcessChunk` → `waitForReplacedMarkers` (`:467-518`). The heavy `applyWindow`
  is NOT run inline; it is submitted to the pool with high priority via
  `queueChunkForPostProcessing` → `submitTaskWithHighPriority([chunkData, window]{
  chunkData->applyWindow(...); })` (`:577-582`, pool submit at
  `core/BlockFetcher.hpp:606-611`).
- The CRITICAL lean step: `queueChunkForPostProcessing` emplaces the chunk's LAST
  window into the WindowMap on the orchestrator thread BEFORE submitting the
  apply-future (`GzipChunkFetcher.hpp:557-575`, comment at `:559-561`: "The last window
  is always inserted into the window map by the main thread ... This is the critical
  path that cannot be parallelized. Therefore, do not compress the last window to save
  time."). Publishing the successor's required window first is what lets the NEXT
  chunk's apply-future start on the pool without serializing behind this chunk.
- While blocked, the consumer overlaps work: it harvests other ready marker futures
  (`:498-511`) and calls `queuePrefetchedChunkPostProcessing` (`:513`, body `:520-551`)
  which scans the whole prefetch cache and submits apply-futures for every ready
  successor — so successors' heavy work is already running by the time the consumer
  reaches them. Only THEN does it block on the current chunk: `markerReplaceFuture->
  second.get()` (`:516`).
- `appendSubchunksToIndexes` (`:364-465`) is pure index bookkeeping (BlockMap/BlockFinder
  push, WindowMap emplace of already-computed subchunk windows, fetching-strategy
  splitIndex). The indexing callbacks run on the orchestrator and are explicitly
  required to be cheap (`:243-245`: "As this is run on the orchestrator thread, it
  should not be compute-intensive").

So the in-order per-chunk cost ≈ one uncompressed 32KiB window publish + map inserts +
a future-get that overlaps all other post-processing. That is light enough that the
consumer drains roughly as fast as the pool produces.

## 5. THE CORE PACING INVARIANT (synthesis)

The ~0–17ms is the conjunction of all four, but the load-bearing property is **(b): a
lean in-order consumer whose per-chunk cost is small enough to keep pace with the pool's
per-chunk production rate** — which keeps the consumer inside the ~`2*par`-chunk
retention window, which is what makes (a) and (c) sufficient:

- (1)+(3) bound the look-ahead to ~`2*par` chunks and make in-flight chunks
  un-evictable. This gives a *retention budget*, not an unconditional guarantee.
- (2) join-in-flight means even a "miss" on an in-flight chunk costs only the remainder
  of a decode, never a full re-decode.
- (4) the lean in-order path (publish-window-then-offload, overlap-while-blocked) keeps
  the consumer's drain rate ≈ the pool's fill rate, so the consumer never drifts past
  the `2*par` retention budget, so completed prefetches are still resident when the
  consumer arrives → cache hit, not cold start.

If the consumer's per-chunk in-order cost rises (or per-chunk decode slows so the pool
can't stay ahead), the consumer drifts beyond `2*par` chunks; completed prefetches get
evicted from the `2*par`-capacity prefetch cache before consumption → cache miss → cold
`submitOnDemandTask`. **That drift is exactly gzippy's +318ms.**

---

## 6. THE GZIPPY DELTA (per item)

gzippy is a faithful structural port; the pacing *machinery* is present. Item by item:

| Vendor structure | gzippy counterpart | Match? |
|---|---|---|
| Prefetch depth cap `prefetching+1 >= par` (`BlockFetcher.hpp:467`) | `block_fetcher.rs:737` `prefetching_len()+1 >= parallelization` | **MATCH** |
| Strategy asked for `prefetchCache.capacity()` indexes (`:474`) | `block_fetcher.rs:758-765` asks `prefetch_cache.capacity()` | **MATCH** |
| Adaptive exp ramp (`Prefetcher.hpp:126-146`) | `prefetcher.rs:109-129` `extrapolate_forward`; `:138-141` size==1 full | **MATCH** |
| Separate in-flight futures map `m_prefetching` (`:685`) | `block_fetcher.rs:66` `prefetching: HashMap<Key, Receiver>` | **MATCH** |
| Join-in-flight via `takeFromPrefetchQueue` (`:385-410`) | `take_prefetch` (`block_fetcher.rs:536-538`) + `try_take_prefetched_pumping` (`:223-269`) | **MATCH** |
| Cache caps `max(16,par)` / `2*par` (`:181-182`) | `chunk_fetcher.rs:528-529` `max(16,pool)` / `pool*2` | **MATCH** |
| Pollution stop `nextNthEviction` (`:544-551`) | `block_fetcher.rs:899-915` `next_nth_eviction(prefetching_len+1)` | **MATCH** |
| Anti-evict touch (`:493-497`) | `block_fetcher.rs:800-807` | **MATCH** |
| Sequential → clear main cache (`:355-358`) | `block_fetcher.rs:491-496` `insert` | **MATCH** |
| Publish last window on consumer BEFORE offload (`:557-575`) | `chunk_fetcher.rs:1542 / 1658` `publish_end_window_before_post_process` | **MATCH** |
| apply/marker work on pool, overlap while blocked (`:467-518`) | `chunk_fetcher.rs:1561/1671` `queue_prefetched_marker_postprocess`; `:1732-1740` pool submit + overlap | **MATCH (structurally)** |
| Pump prefetch on 1ms tick while waiting (`:314-316`) | `chunk_fetcher.rs:1289-1301` pump closure; `block_fetcher.rs:256-268` | **MATCH** |

### Where gzippy's in-order path is HEAVIER than vendor (the real divergences)

These are present *inside* the matching structure and are what make the consumer's
per-chunk cost larger than vendor's lean path:

1. **Arc ownership gymnastics on every consumed chunk** (`chunk_fetcher.rs:1457-1491`):
   `Arc::try_unwrap` then, on failure, `SharedChunkData::take_or_clone` — a potential
   deep clone of a ~MB ChunkData. Vendor moves a `shared_ptr` (`GzipChunkFetcher.hpp:
   308, 361`) with no equivalent copy. Lever-G comments (`block_fetcher.rs:409-414,
   447-456`) document removing cache-insert/promote precisely because the extra Arc ref
   forced this deep clone (~7ms × 24 chunks). The clone path still exists when an eager
   post-process holds a second ref.

2. **Blocking recv on the CURRENT marker chunk's apply** (`chunk_fetcher.rs:1718-1740`,
   `recv_post_process_blocking`). Vendor also blocks here (`:516`), so this is parity —
   BUT it is parity *in structure* only; the cost is downstream of how fast the pool
   `applyWindow` runs (engine), see judgment.

3. **Window-publish materialization** uses owned-None 32KiB buffers
   (`window_map.rs:128-142` `insert_owned_none`, `chunk_fetcher.rs:1657`
   `materialize_window`) with a per-publish alloc+copy. Vendor publishes the last window
   uncompressed too (`:559-561`) — roughly parity, slightly heavier on gzippy's copy.

4. **Drain/recycle bookkeeping**: `pending` queue, `recycle_deferral` (depth 2,
   `chunk_fetcher.rs:1065-1067`), `post_process_inflight_cap` (`:1053`), and repeated
   `drain_ready_pending_heads` / `drain_one_pending` scans per iteration
   (`:1775-1809`). Vendor has no per-iteration FIFO-drain scan of this shape; its writes
   are inline after `postProcessChunk` returns. This is extra in-order work with no
   vendor counterpart and is a candidate structural lever.

### Is gzippy's prefetch depth shallower? Does its cache evict in-flight?

- **Depth: NO, not shallower.** Same `par-1` in-flight cap, same `2*par` index request,
  same adaptive ramp. (`block_fetcher.rs:737, 758-765`; `prefetcher.rs:109-172`.)
- **In-flight eviction: NO.** gzippy has the SAME separate `prefetching` map
  (`block_fetcher.rs:66`); in-flight receivers are not in the evictable caches.
  `process_ready_prefetches` (`:1008-1038`) moves them to `prefetch_cache` only after
  `try_recv` succeeds — same as vendor `:438`.
- **Completed-prefetch eviction: YES, same `2*par` budget as vendor** (`:528-529`). This
  is the budget that the +318ms drift overruns — but it overruns it in *both* impls if
  the consumer falls behind; gzippy's consumer falls behind, vendor's doesn't.

---

## CRITICAL DISTINCTION (labeled JUDGMENT)

**Is rapidgzip's ~0–17ms pacing a STRUCTURE gzippy lacks, or is it downstream of the
fast igzip engine?**

**Judgment: gzippy does NOT lack the pacing structure. Every load-bearing pacing
mechanism — `par-1` prefetch depth, the separate un-evictable in-flight futures map,
join-in-flight, `max(16,par)`/`2*par` cache caps, the `nextNthEviction` pollution stop,
publish-window-before-offload, and overlap-while-blocked — is present and cites the
matching vendor `file:line`. Therefore the +318ms is NOT explained by a missing
structure. It is the predicted symptom of the consumer drifting past the (correctly
sized) `~2*par`-chunk retention budget, and that drift has TWO coupled causes:**

(i) **Engine throughput (dominant, per `project_pregate_placement_is_dominant_lever`'s
2.3x clean-rate gap).** rapidgzip's `par-1` workers each finish a chunk fast enough that
the pool fill rate ≥ the consumer drain rate, so the consumer stays within `2*par`. With
a 2.3x-slower inner loop, each worker holds a slot ~2.3x longer; the frontier cannot get
far ahead AND the consumer's own apply-wait (item 2) lengthens — the consumer drifts,
and completed prefetches evict before consumption.

(ii) **A heavier in-order consumer path (secondary but structurally fixable
independent of the engine):** the Arc unwrap/deep-clone (delta #1), the per-iteration
FIFO drain/recycle bookkeeping (delta #4), and per-publish window copies (delta #3) add
in-order cost per chunk that vendor does not pay. These raise the consumer's drain time
regardless of engine speed and shrink the engine headroom needed to keep pace.

**Implication for the supervisor's lever question:** placement is NOT a cleanly
*separable* faithful-port lever in the sense of "port a missing scheduler structure" —
the scheduler structure is already ported. The remaining placement work is (a) shaving
the gzippy-specific in-order overheads in deltas #1, #3, #4 so the consumer's drain rate
rises (engine-independent, worth doing), and (b) the engine clean-rate gap, which sets
the pool fill rate. Because the retention budget is fixed at `~2*par`, **(a) only buys
margin; it cannot fully close the gap while (b) keeps the fill rate below the drain
rate.** This is consistent with the memo that PLACEMENT and ENGINE are CO-PRIMARY: a
faithful scheduler alone (already ~achieved) does not produce 0–17ms pacing unless the
per-chunk decode is also fast enough to keep the frontier supplied.

*Confidence: high on "the structure is present and matches vendor" (direct line-by-line
cites). Medium on the relative weighting of (i) vs (ii) — that split needs a causal
perturbation (e.g., neutralize delta #1's clone and re-measure consumer drift), not just
this source read.*

---

## BOTTOM LINE

1. rapidgzip's 0–17ms pacing = `par-1` in-flight prefetch (un-evictable map) + `2*par`
   pollution-guarded retention + join-in-flight + a LEAN in-order consumer that publishes
   the successor window then offloads all heavy work to the pool and overlaps it.
2. The load-bearing property is the lean consumer keeping its drain rate ≈ the pool's
   fill rate, so it never drifts past the fixed `~2*par`-chunk retention budget.
3. gzippy has ALL of this machinery, line-for-line (`block_fetcher.rs`, `prefetcher.rs`,
   `chunk_fetcher.rs`, `window_map.rs`, `cache.rs` all cite the matching vendor lines).
4. gzippy's +318ms is the consumer drifting past that retention budget → completed
   prefetch evicted → cold `submitOnDemandTask`; caused by (i) the 2.3x slower engine
   lowering pool fill rate and (ii) gzippy-only in-order overheads (Arc deep-clone, FIFO
   drain/recycle scans, window copies) raising consumer drain time.
5. JUDGMENT: placement is not a *missing* structure — the scheduler is faithfully ported.
   It is downstream of per-chunk cost; PLACEMENT and ENGINE are co-primary, and lean-up
   of the gzippy-only consumer overheads buys margin but cannot tie 0–17ms while the
   engine keeps the fill rate below the drain rate.
