# rapidgzip ↔ gzippy parallel-machinery map (SUPPORT deliverable)

Read-only source map for the full-curve leader's H2/H3 discrimination. Every
row cites vendor `file:line` and gzippy `file:line`, verified first-hand. No
box, no build, no measurement — this is the vendor SIDE + convergence target
only.

Vendor root: `vendor/rapidgzip/librapidarchive/src/`
gzippy root: `src/decompress/parallel/`

Bottom line up front: **the four structures the leader asked about are FAITHFUL
ports — pool, priorities, prefetch depth, CRC combine, and the (absent) wave
barrier all match vendor.** The only structures with NO vendor counterpart are
three gzippy-specific *backpressure / buffer-recycle* knobs in the consumer loop
(`post_process_inflight_cap`, `RECYCLE_DEFER_DEPTH`, lone-Ready drain rules).
Those are the only plausible machinery origin of a T4-specific trough; the depth
and pool-priority shapes are identical and so do NOT support H3 and only weakly
support H2 (see each section).

---

## 1. THREAD POOL — one pool, two priorities (decode 0 / post-process −1)

| Aspect | rapidgzip | gzippy |
|---|---|---|
| Pool count | ONE `ThreadPool m_threadPool` per BlockFetcher (`core/BlockFetcher.hpp:686`) | ONE `ThreadPool` (`thread_pool.rs:175`), held by the fetcher run |
| Task ordering | `std::map<int priority, std::deque<...>>`, worker serves `m_tasks.begin()` first → **lower int = higher priority** (`core/ThreadPool.hpp:237`, dispatch `:213-220`) | `BTreeMap<priority, VecDeque<..>>`, same ordered dispatch (`thread_pool.rs:15,150-152`) — documented faithful port |
| Decode priority | `submit(..., /* priority */ 0)` for on-demand (`BlockFetcher.hpp:586`) and prefetch (`:557`) | `submit_decode_to_pool(..)` → `submit(task, /* priority */ 0)` (`chunk_fetcher.rs:2295,2298`; both speculative + on-demand branches are 0) |
| Post-process priority | `submitTaskWithHighPriority` → `submit(task, /* priority */ -1)` (`BlockFetcher.hpp:606-611`), called from `queueChunkForPostProcessing` (`GzipChunkFetcher.hpp:579`) | `apply_window` submit at `/* priority */ -1` (`chunk_fetcher.rs:2389,2399`; doc `:48-51`) |
| On-demand spawn | spawn until `m_threadCount`, gated by `m_idleThreadCount==0` (`ThreadPool.hpp:157`) | same lazy spawn + idle gate (`thread_pool.rs:28-31`) |

**H2 reading (post-process stealing decode parallelism at intermediate T):**
rapidgzip's structure does **NOT** avoid this — both use ONE pool and run
post-process (marker-replace / `applyWindow`) at priority −1, i.e. STRICTLY
ahead of decode at priority 0. When a post-process task is queued an idle worker
grabs it before any pending decode. So the "post-process preempts decode" effect
is INHERENT to the faithful design and is present in vendor too. It is *correct*
there because `applyWindow` is the terminal step that unblocks the in-order
consumer, so prioritizing it shortens the serial critical path. → A T4 trough
caused purely by "post-process steals a decode slot" would also afflict vendor;
on this axis gzippy is already converged. The discriminator must be a knob that
gzippy has and vendor does NOT (Section 5), not the priority itself.

Faithful target: KEEP as-is (one pool, decode 0 / post-process −1). No change.

---

## 2. PREFETCH DEPTH / admission / saturation

| Aspect | rapidgzip (`BlockFetcher.hpp`) | gzippy (`block_fetcher.rs`) |
|---|---|---|
| In-flight cap | `m_prefetching.size() + 1 >= m_threadPool.capacity()` → stop (`:465-468`); capacity == parallelization | `prefetching_len() + 1 >= self.parallelization` → stop (`:737`) — identical formula |
| Max prefetched | "only `m_parallelization-1` blocks prefetched" (`:564-566` invariant + throw `:568`) | same `parallelization-1` intent, doc `:79-81,601-605` |
| `m_cache` capacity | `max(16, parallelization)` (`:181`) | `max(16, pool_size)` (`chunk_fetcher.rs:578`) |
| `m_prefetchCache` capacity | `2 * parallelization` (`:182`) | `pool_size * 2` (`chunk_fetcher.rs:579`) |
| Prefetch count requested | `m_fetchingStrategy.prefetch(m_prefetchCache.capacity())` (`:474`) | `prefetch_cache.capacity()` upper bound (`block_fetcher.rs:757-763`) |
| Cache-pollution stop | `nextNthEviction(m_prefetching.size()+1)` → break if usable evicted (`:546-551`) | `next_nth_eviction(prefetching_len()+1)` (`block_fetcher.rs:903`) |
| Ready-prefetch promotion | `processReadyPrefetches()` at top of `prefetchNewBlocks` + in `get` wait loop (`:427,463,314-316`) | `process_ready_prefetches()` at top of every consumer iter (`chunk_fetcher.rs:1208-1214`) |

**H3 reading (does rg keep MORE in-flight at intermediate T?):** The
effective-depth formulas are **byte-for-byte identical** — in-flight prefetch
caps at `parallelization` (= T), prefetch_cache at `2·T`, cache at `max(16,T)`,
same saturation predicate, same pollution-eviction guard. There is **no
structural basis for rg holding more chunks in flight at T4** than gzippy. So
the SOURCE does NOT support H3 as a machinery defect — if the leader measures rg
with deeper effective in-flight at T4, the cause is downstream (e.g. faster
per-chunk decode → the pipeline drains and refills faster, so depth *utilization*
differs even though the *cap* is identical), not a different admission policy.
Recommend the leader treat H3 as "depth-utilization", not "depth-cap", and look
to engine clean-rate (the known 2.3× inner-loop gap) rather than the fetcher.

Faithful target: KEEP. Already converged.

---

## 3. SERIAL FLOOR (S) — what runs on the in-order consumer

rapidgzip's serial consumer is `ParallelGzipReader::read` (`ParallelGzipReader.hpp:553-646`).
Per consumed chunk, on the single consumer thread:

1. `chunkFetcher().get(m_currentPosition)` — blocks on the current chunk's
   future only (`:581`).
2. `processCRC32(chunkData, ...)` — `:613-617`, body `:1453-1503`. This is a
   **per-stream COMBINE, not a recompute**: `m_crc32.append(chunkData->crc32s.front())`
   then folds each footer's stream crc (`:1490-1502`). Workers compute the
   crc32s during decode; the consumer only chains them.
3. `writeFunctor(...)` → `writeAll` (`:619-625`, functor `:511-544`) — the only
   mandatory output work.
4. `releaseUpTo` on the file reader / window map (`:637,641`).

Window publication that is on the **critical path** is done by the consumer/main
thread, NOT a worker: the last (end-of-chunk) window is emplaced by the main
thread in `queueChunkForPostProcessing` (`GzipChunkFetcher.hpp:558-575`, see the
`:559-561` comment: *"The last window is always inserted into the window map by
the main thread … This is the critical path that cannot be parallelized"*). The
actual `applyWindow` (marker→byte resolution) runs on the pool at prio −1.

gzippy serial consumer is `chunk_fetcher::run_consumer_loop` / `drive`
(`chunk_fetcher.rs:1205` loop):

1. `block_fetcher.process_ready_prefetches()` + `harvest_ready_postprocess`
   (`:1213,1218`).
2. `block_finder.get(next_unprocessed_block_index)` (`:1223`).
3. `BlockFetcher::get` (waits current chunk future).
4. CRC combine: `total_crc.append(stream_crc)` per `chunk.crc32s`
   (`chunk_fetcher.rs:462-463` oracle path, `:644` production) — **same
   per-stream fold as vendor, not a recompute**.
5. Window publish on consumer: `publish_end_window_before_post_process`
   (`:1696,1812`) — vendor parity, publish stays on consumer (doc `:49`).
6. `drain_one_pending` → write to `writer`/`out_fd` (`:1766,1944,1980`).

**Interpreting "S_gz < S_rg":** the serial COMPONENTS are the same set (get-wait,
crc-combine, last-window-publish, write). gzippy's per-chunk serial CRC is an
O(streams) `append`, identical algorithmic cost to vendor — so a genuinely
*smaller* S in gzippy is plausible only from: (a) gzippy publishing the end
window via the eager full-scan ahead of the consumer so the consumer's
`get_last_window` recompute is skipped (`chunk_fetcher.rs:1802-1813` —
documented redundant-recompute elision, byte-exact), and/or (b) gzippy's writev
/ `out_fd` zero-copy path vs vendor's `writeAll`. Neither is a defect; both are
legitimately-lower serial work. If the leader's model attributes the gzippy
*advantage* to S, that is consistent with these two elisions and is NOT a place
to "converge back" to vendor.

---

## 4. CONSUMER in-order collection — wave/batch barrier?

| | rapidgzip | gzippy |
|---|---|---|
| Collection shape | strictly pipelined: one `get(position)` per loop turn, waits on that chunk's future only; later chunks' `applyWindow` keep running on the pool (`ParallelGzipReader.hpp:575-643`; `waitForReplacedMarkers` waits ONLY on current `markerReplaceFuture->second.get()`, `GzipChunkFetcher.hpp:516`) | one chunk per loop turn (`chunk_fetcher.rs:1205` loop), waits on predecessor window presence then current future (`:1756`, `drain_one_pending`) |
| Wave / batch barrier | **NONE** | **NONE** — atomic-index work queue, the prior wave-of-pool_size design was explicitly removed (`chunk_fetcher.rs:410-412` comment "NO wave barriers") |
| Opportunistic harvest | `waitForReplacedMarkers` drains other ready futures (`GzipChunkFetcher.hpp:497-511`) + `queuePrefetchedChunkPostProcessing` (`:520-551`) full sorted scan of prefetch cache, queues every not-yet-processed chunk whose predecessor window exists | `harvest_ready_postprocess` (`:1218`) + `queue_prefetched_marker_postprocess(.., &[], &[])` full-scan = vendor's `None` robust path (`:1705-1722`) |

Neither side has a barrier; both are single-chunk-at-a-time pipelines with
opportunistic post-process draining. Converged.

---

## 5. The ONLY divergences with no vendor counterpart — T4-trough candidates

These three live in the gzippy consumer loop and have NO line in vendor. They
are the only machinery that could manufacture a T4-specific trough:

1. **`post_process_inflight_cap = pool_size - 1`** (`chunk_fetcher.rs:1175`),
   enforced by `while pending.len() > post_process_inflight_cap { drain_one_pending }`
   (`:1944`). This makes the consumer BLOCK-drain (synchronously wait + write)
   whenever queued post-process/output exceeds `pool-1`. **At T4 the cap is 3.**
   Vendor has no such hard cap: its in-order `read` waits only on the *current*
   chunk's future (`GzipChunkFetcher.hpp:516`) and bounds in-flight solely via
   the prefetch-cache capacity (`2·T`) — it never forces the consumer to block
   on the (N−pool)-th pending write. At low T this cap is the tightest at T4
   (cap=3 vs e.g. T8 cap=7, T16 cap=15 where the cap rarely binds relative to
   the ~21-chunk fixture) — i.e. its throttling effect is *largest at
   intermediate T and washes out at high T*, which is the exact signature of a
   T4 trough.
   - **Convergence target:** drop the hard `pending.len() > pool-1` block-drain;
     mirror vendor `waitForReplacedMarkers` — block only on the predecessor's
     post-process future, harvest the rest non-blocking, and bound in-flight via
     the prefetch-cache capacity (`2·T`) as vendor does. (`GzipChunkFetcher.hpp:478-518`.)

2. **`RECYCLE_DEFER_DEPTH = 2`** (`chunk_fetcher.rs:1187-1189`, applied
   `:3722,3761`) — keeps the last 2 drained chunk buffers off the pool to dodge
   a lone-Ready-drain CRC race. No vendor counterpart (vendor recycles via
   `shared_ptr`/cache eviction). At T4 holding 2 buffers back is a larger
   fraction of the 4-deep buffer set than at T8/T16, so it can shave effective
   buffer supply most at intermediate T.
   - **Convergence target:** vendor returns chunk memory through `shared_ptr`
     refcount + `m_cache`/`m_prefetchCache` eviction (`BlockFetcher.hpp:352-359,
     680-682`); converge buffer lifetime onto cache eviction rather than a
     fixed-depth defer ring once the lone-Ready CRC bug is fixed structurally.

3. **lone-Ready drain rule** (`chunk_fetcher.rs:1172-1175,1926` "CRC-unsafe at
   len==1" / `:3761`) — gzippy-specific drain-timing rule with no vendor analog.
   Low risk on its own; flagged for completeness because it shares the same
   `pending`-queue machinery as (1).

---

## Hand-off summary for the leader

- **H2 (post-process steals decode parallelism at intermediate T):** the pool +
  priority shape that *would* cause this is IDENTICAL in vendor, so the bare
  priority is not the defect. The real gzippy-only throttle is
  `post_process_inflight_cap = pool-1` (`chunk_fetcher.rs:1175,1944`), which is
  tightest at T4 and absent in vendor. If H2 survives perturbation, this is the
  faithful thing to converge (delete the hard cap, mirror
  `waitForReplacedMarkers`).
- **H3 (rg keeps more in-flight at intermediate T):** NOT supported by source —
  prefetch depth caps, prefetch-cache (`2·T`), saturation predicate, and
  pollution guard are byte-identical. Any measured depth gap is utilization
  (drain/refill rate, downstream of engine clean-rate), not admission policy.
- **S_gz < S_rg:** consistent with two legitimate gzippy serial-work elisions
  (eager end-window publish skipping the consumer recompute,
  `chunk_fetcher.rs:1802-1813`; and `out_fd` writev). Not a defect, not a
  converge-back target. CRC is a per-stream combine on both sides.
- **No wave/batch barrier on either side.**
