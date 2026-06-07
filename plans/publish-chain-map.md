# GATE 2 — per-chunk WINDOW PUBLISH transliteration map (rapidgzip ↔ gzippy)

Read-only. HEAD `cbfb256`, branch `reimplement-isa-l`. Every claim cites a
`file:line` I opened.
`vendor` = `vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`,
`cf` = `src/decompress/parallel/chunk_fetcher.rs`,
`sm` = `src/decompress/parallel/segmented_markers.rs`.

Advisor-reviewed (Opus, subagent_type=claude): verdict CONFIRMED, no hole found
in (a) cross-chunk getLastWindow overlap, (b) consumer/pool step placement, or
(c) the 217µs term gating the wall.

---

## VERDICT (one line)

**SERIAL-IN-BOTH — NO DIVERGENCE.** rapidgzip's per-chunk window-publish is
serial on the main/consumer thread, and gzippy serializes *exactly the same step,
no more*. The only consumer-serial term in BOTH is the cheap O(32 KiB)
`getLastWindow` end-window build (= 217 µs total/run); the heavy marker→byte
`applyWindow` resolve runs off-path on the pool in BOTH; successor dispatch
overlaps the wait in BOTH. There is no serialization rapidgzip avoids that gzippy
imposes. GATE 2 is **not** a transliteration lever.

---

## 1. The two-column map (each step of the per-chunk window publish)

| # | role | vendor `file:line` | gzippy `file:line` | thread |
|---|------|--------------------|--------------------|--------|
| 1 | In-order consumer pulls next chunk | `processNextChunk` vendor:312, `getBlock` vendor:329 | consumer loop, `into_chunk_data` cf:1598 | consumer |
| 2 | **Predecessor window lookup** (window required by THIS chunk) | `m_windowMap->get(*nextBlockOffset)` vendor:334 (asserts present, vendor:335-339) | spin `window_map.contains(handoff_bit)` cf:1510 ("near-0× after chunk 0" cf:1517); `confirmed_predecessor_window` cf:1546→2382→2386 | consumer |
| 3 | Overlap: queue successor post-processing **during** the wait | `queuePrefetchedChunkPostProcessing` vendor:513 **before** the `.get()` | `drain_one_pending` cf:1520 inside `consumer.wait_replaced_markers` | consumer |
| 4 | **Publish THIS chunk's END window** (window provided to chunk i+1) — the serial critical-path term | `queueChunkForPostProcessing`→`getLastWindow(*previousWindow)` vendor:572 at `encodedOffsetInBits+encodedSizeInBits`; footer→empty window vendor:562-570; deliberately NOT compressed (vendor:560-561) | `publish_end_window_before_post_process` cf:1566→2426; `get_last_window_vec` cf:2445; footer→empty cf:2442-2443; span `consumer.get_last_window` cf:1564 | consumer |
| 4b | Vendor's *reason* this is serial-on-main | vendor:559-561 verbatim: "The last window is always inserted into the window map by the main thread because else it wouldn't be able queue the next chunk for post-processing in parallel. **This is the critical path that cannot be parallelized.**" | mirrored — comment cf:1556-1562 cites vendor:558 | — |
| 5 | Eager full sorted prefetch-cache scan → dispatch ready successors' resolve | `queuePrefetchedChunkPostProcessing` vendor:520-551 (sorted offsets vendor:525-528; `hasBeenPostProcessed` gate vendor:539; predecessor `get` vendor:544; `queueChunkForPostProcessing` vendor:549) | `queue_prefetched_marker_postprocess` cf:1579→2456 (`prefetch_cache_contents_sorted` cf:2468; `chunk_may_resolve_markers_early`/`has_been_post_processed` cf:2403-2408; predecessor cf:2504; publishes successor end-window on consumer cf:2513) | consumer |
| 6 | **Heavy marker→byte resolve** `applyWindow` — submitted to pool, high priority | `submitTaskWithHighPriority([chunkData,window]{ chunkData->applyWindow(...) })` vendor:577-582 | `submit_post_process_void`/`_task` `thread_pool.submit(task, priority -1)` cf:2094/2104 → `run_post_process_in_place` cf:2540 → `apply_window`/`resolve_and_narrow_in_place` cf:2554→sm:461 | **pool** |
| 7 | Consumer blocks on THIS chunk's resolve future (after overlapping successors) | `markerReplaceFuture->second.get()` vendor:516 | `wait.future_recv` cf:1626; `recv_post_process_blocking` cf:1638/1647 with `overlap` queueing during the wait cf:1630-1636 | consumer (WAIT) |
| 8 | After resolve: fix encoded offset, append subchunks/index, emplace subchunk windows | `setEncodedOffset` vendor:349, `appendSubchunksToIndexes` vendor:357/365 (subchunk window emplace vendor:429-458) | `set_encoded_offset` cf:1656, `consumer_append_subchunks_vendor` cf:1658 | consumer |

Note the symmetry of step 4: gzippy publishes the end-window on the consumer in
**both** call sites (the in-order consumer cf:1566 and the eager resolve-ahead
scan cf:2513) — exactly as vendor calls `getLastWindow` from both
`waitForReplacedMarkers` (vendor:494, main thread) and
`queuePrefetchedChunkPostProcessing` (vendor:549, main thread). Neither tool ever
puts the end-window build on the pool.

---

## 2. ANSWER to the key question

**SERIAL-IN-BOTH (no divergence found).** Walking the three holes the lever would
have to live in:

- **(a) Does vendor overlap chunk i+1's `getLastWindow` with chunk i's
  `applyWindow` in a way gzippy can't?** No. Both `getLastWindow` calls are serial
  on the *same* main thread by vendor construction (vendor:559-561 declares it the
  unparallelizable critical path). gzippy puts `get_last_window` on the same
  consumer thread (cf:1564). There is no i+1/applyWindow overlap vendor achieves
  and gzippy forgoes — the placement is symmetric. The end-window build is the
  intentionally-cheap, uncompressed O(32 KiB) tail in BOTH (vendor:560-561 ↔
  gzippy `get_last_window_vec` cf:2445).

- **(b) Any step gzippy puts on the consumer that vendor puts on the pool (or
  vice-versa)?** No asymmetry. Predecessor-get (step 2), end-window insert
  (step 4), and eager successor dispatch (steps 3/5) are consumer-side in both;
  the heavy `applyWindow` resolve (step 6) is pool-side at high priority in both
  (vendor:579-582 ↔ cf:2094 priority −1 → cf:2540). Resolve-ahead publishes the
  end-window on the consumer *before* the pool submit in both (vendor:572 then
  vendor:577-582 ↔ cf:2513 then cf:2514).

- **(c) Could the cheap 217 µs serial term secretly gate the wall?** No. 217 µs
  total/run (the `consumer.get_last_window` span, `structural-gap-analysis.md` §2
  row 6) is ≪ the 4.89 ms per-link "L_resolve" interval, and `apply_window`
  measured **0 ms wall-critical** in both the baseline and the decode≈0 floor
  (`leverB-ceiling.md` §CORRECTED, lines 122). Even fully on the wall, 217 µs is a
  second-order term; the wall lever is decode rate, not publish.

---

## 3. (No divergence — section 3 is the explicit "no lever" statement + reconcile)

There is **no** minimal faithful delete-divergent / create-match change here,
because the dependency structure already matches the vendor byte-for-byte:

- The serial dependency `window(i+1) needs end-window(i)` EXISTS in both and is
  walked identically — on the consumer thread, carrying only the cheap
  `getLastWindow` (vendor:572 ↔ cf:1566/2513).
- The heavy resolve is off the consumer path on the pool in both
  (vendor:579-582 ↔ cf:2540, priority −1).
- Successor dispatch overlaps the wait in both (vendor:513 ↔ cf:1520/1579 + the
  cf:1630-1647 overlap during `wait.future_recv`).

### Reconciliation with the 217 µs-compute claim
Publish is **both cheap AND structurally faithful** — these are not in tension.
`L_resolve-investigation.md` §1.3/§5 separates two distinct quantities the
"L_resolve" name has conflated:

1. **Publish COMPUTE** = the `consumer.window_publish_marker` /
   `consumer.get_last_window` span (cf:1542/1564) = **217 µs total/run**. This is
   the serial dependency term, and it is the faithful counterpart of vendor's
   uncompressed main-thread `getLastWindow` (vendor:572, deliberately cheap per
   vendor:560-561). Cheap in vendor by design ⇒ cheap in gzippy by faithful port.

2. **Inter-publish GAP** = `t_publish(i) − t_publish(i−1)` ≈ 4.89 ms
   (`parallel-sm-model.md:70`), which telescopes to ≈ wall/N and is dominated by
   `wait.future_recv` (cf:1626) — the consumer **blocking** on the pool's
   window-absent **decode**+applyWindow future, gated by the 1.77×-slower decode
   (`d_w` 125.5 ms vs 70.95 ms). This is a WAIT on decode, **not** the publish
   dependency.

So GATE 2 confirms the publish chain is already a faithful transliteration; the
4.89 ms is decode-rate WAIT, not a serialization rapidgzip avoids. The faithful
lever lives in the window-absent inner decode loop (Lever B), exactly as
`L_resolve-investigation.md` §3 and `leverB-ceiling.md` §DIRECTIONAL conclude —
NOT in the publish/resolve dependency structure mapped above.

GATE2_DONE
