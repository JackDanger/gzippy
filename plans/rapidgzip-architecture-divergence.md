# Where gzippy's architecture diverges from rapidgzip's (2026-06-04)

User directive: *"Where does our architecture not match rapidgzip's? That's the gap I'm
trying to close. I don't want to measure performance until we close it."*

This maps the parallel-SM decode pipeline of gzippy against the vendor
(`vendor/rapidgzip/librapidarchive/src/rapidgzip/`), grounded in source citations.
Measured context (locked frozen-clock, byte-exact): gzippy is 2.0–2.6× slower than
rapidgzip at every thread count; the fulcrum MODEL says the wall is bound by the serial
per-chunk marker-resolution publish-chain (`N·L_resolve`), NOT decode rate (decode engine
`isal_lut_bulk` ≈ `ResumableInflate2`, a TIE). That symptom is explained by Divergence #1.

## Divergence #1 (PRIMARY — explains the publish-chain serialization): parallel marker resolution via EAGER window-chain propagation

**rapidgzip** (`GzipChunkFetcher.hpp`):
- `waitForReplacedMarkers` (479) runs on EVERY consumed chunk and ALWAYS calls
  `queuePrefetchedChunkPostProcessing()` (513).
- `queuePrefetchedChunkPostProcessing` (520) iterates ALL prefetched cached chunks sorted
  by offset; for each whose predecessor window exists, calls `queueChunkForPostProcessing`.
- `queueChunkForPostProcessing` (553) does the critical split:
  1. **Cheap SERIAL part on the main/consumer thread:** compute + publish the chunk's END
     window via `getLastWindow(previousWindow)` into the WindowMap (557-575). Comment
     (559-561): *"The last window is always inserted by the main thread because else it
     wouldn't be able to queue the next chunk for post-processing in parallel. This is the
     critical path that cannot be parallelized."*
  2. **Expensive PARALLEL part on the pool:** submit `chunkData->applyWindow(window)` as a
     future (577-582), tracked in `m_markersBeingReplaced`.
- Because step 1 publishes chunk K's END window (= K+1's predecessor) cheaply, the loop
  PROPAGATES THE WINDOW CHAIN forward in one pass, so K, K+1, K+2 … all get queued for
  parallel `applyWindow` ahead of the consumer. The only serial cost is `getLastWindow`
  (last-32 KiB extraction), NOT the full marker resolution.

**gzippy** (`chunk_fetcher.rs`):
- `consumer_loop` (942) is a faithful port of `processNextChunk`, BUT the comment at 971-973
  states: *"production uses handoff-triggered resolve-ahead only"* — the full-cache scan
  equivalent of `queuePrefetchedChunkPostProcessing` is gated OFF behind
  `GZIPPY_EAGER_POSTPROC` (`eager_postproc_enabled()`).
- `queue_prefetched_marker_postprocess` (2520) only resolves a chunk if
  `confirmed_predecessor_window(window_map, resolve_anchor)` ALREADY exists (2568), and
  follows the handoff CHAIN one chunk at a time (2581-2593). It does NOT eagerly publish
  each chunk's end window on the consumer to unblock the next — so the chain does not
  propagate ahead and only the chunk whose predecessor is already confirmed resolves.
- Window publication is coupled to resolution: `resolve_chunk_markers_on_chunk` (2457) runs
  `populate_subchunk_windows` AFTER `apply_window` (full resolution). rapidgzip publishes
  the end window INDEPENDENT of (and before) the full resolution.
- DIAGNOSTIC (GZIPPY_VERBOSE, silesia p4): `Eager post-process: runs_nonempty=0 submitted=0`;
  `Worker resolve-ahead: ok=5 / ~12 marker chunks`. → most marker chunks fall to the
  consumer's `Some(window)` submit-and-wait arm (1671) → serial `N·L_resolve`.

**Primitives gzippy already has:** `ChunkData::get_last_window(predecessor)` (chunk_data.rs:969),
`window_map.insert_owned_none` (consumer publish). The machinery exists; the TIGHT
eager-publish-then-queue-all loop is what diverges.

**Fix (faithful port):** mirror `queueChunkForPostProcessing` — on the consumer, for each
prefetched chunk in offset order, EAGERLY publish its end window (`get_last_window`) into the
WindowMap (cheap, serial), THEN submit `apply_window` to the pool; this unblocks the next
chunk so the whole prefetched set resolves in parallel. Make this the production path (not
gated behind GZIPPY_EAGER_POSTPROC). CAUTION: the resolve-ahead path has a documented history
of CRC regressions / races (the ~50-commit churn `8291adb..HEAD` is full of "revert CRC
regression", "revert racy handoff"); the port must preserve byte-exactness + CRC32 ordering.

## Divergence #2 (the user's original /goal): one decoder vs gzippy's dual engine

rapidgzip has ONE `deflate::Block` with `read<containsMarkerBytes>` (`deflate.hpp`) — a single
const-templated decoder that emits u16 markers when window-absent and clean bytes otherwise,
on one bit cursor, flipping mid-stream. gzippy split this into TWO engines: `deflate_block::Block`
(marker bootstrap, ring) + `isal_lut_bulk` (clean tail, flat), joined by a `MarkerStep::Handoff`
seam. NOTE: this is perf-marginal (marker phase ≤0.4% of bytes; both use the same ISA-L LUT;
decode is a TIE) — it is a FAITHFULNESS/simplicity divergence, not the wall lever. Worth closing
for structural convergence but it will not move the wall on its own.

## Divergence #3 (suspected regression, lower priority): buffer layout cost
The FOOTPRINT-ALIGN segment-native conversion (`U8`/`Vec<u16>` → `SegmentedU8`/`SegmentedU16`)
matches rapidgzip's segmented `DecodedData` in SHAPE but cost ~20-30% wall (per-access
indirection) for ~25% RSS. rapidgzip's `DecodedData` is also segmented, so the divergence is in
the ACCESS PATTERNS (per-element vs per-segment-slice) in gzippy's hot loops, not the segmentation
itself. Re-examine after Divergence #1 (the publish-chain dominates).

## Method to find more divergences
Pair-read each vendor file against its gzippy port and diff the STRUCTURE (not the bytes):
GzipChunkFetcher.hpp↔chunk_fetcher.rs (consumer/post-process/window — done above),
GzipChunk.hpp↔gzip_chunk.rs (decode loop), deflate.hpp↔deflate_block.rs (Block),
ChunkData.hpp↔chunk_data.rs (applyWindow/getLastWindow/footers), WindowMap.hpp↔window_map.rs,
BlockFetcher.hpp↔block_fetcher.rs (prefetch). Priority = #1 (publish-chain) since the model
says it binds the wall.

## STATUS 2026-06-05: Divergence #1 PORTED (the win). Next divergence relocated by measurement.
Divergence #1 (eager window-chain) CLOSED — ported vendor queuePrefetchedChunkPostProcessing
(full sorted scan; Some(chunk_end_bit)->None at the 2 post-publish sites; commits f7868ab/1351909).
Locked-Fulcrum A/B: T8 -16.5%, T16 -21.1%, T4 -7.1% (T1 +15% benchmark-only). See wall-progress.md.
POST-FIX (new T8 trace): consumer head-of-line wait 737->416ms (4 cold re-decodes remain),
consumer.iter 1011->855ms; consumer serial "other" still ~3x rapidgzip.
- The 4 cold re-decodes: PRE-REGISTERED probe FALSIFIED interior-accept (advisor's secondary
  divergence: gzippy exact `==max` reuse guard vs vendor matchesEncodedOffset interior-range
  ChunkData.hpp:402 + setEncodedOffset). All 4 stalls show containing_chunk=FALSE — the partition
  chunk is EVICTED before the lagging consumer arrives, so there is NOTHING to interior-accept.
  These are an EVICTION consequence of consumer lag, NOT a reuse-guard gap. DO NOT port
  interior-accept for them (dead — measured).
- NEXT DIVERGENCE (= Divergence #3 data-plane): consumer serial cost drives the lag that drives
  eviction. Dominant: dispatch_post_process ~144ms = gzippy DEEP-CLONES the ~16MB ChunkData when
  `Arc::try_unwrap` fails (ConsumerChunkHold `(*arc).clone()`), where vendor SHARES via shared_ptr
  and mutates applyWindow on the pool. The eager-chain INCREASED Arc refs -> more clone cost (also
  the T1 regression). Plus window_publish_marker ~122ms (get_last_window on SegmentedU8 vs vendor
  contiguous FasterVector). Correctness-sensitive (the clone avoids a mutation race) — a real
  vendor-cited cycle, not a tail-of-session edit.

## CORRECTION 2026-06-05 (measurement falsified the Arc-clone hypothesis — discipline, not assumption)
Measured `Arc::try_unwrap hits/misses = 39 / 0`: the consumer-side deep-clone (chunk_fetcher.rs:1401)
NEVER fires, and the eager-resolve clone (submit_post_process_from_prefetch:2005) runs INSIDE the move
closure = on the POOL, not the consumer. So Divergence #3-as-"consumer Arc deep-clone" is DEAD (the
T1 regression and dispatch_post_process cost are NOT the clone). Do NOT port a shared-mutate data
model for this — measured, no clone on the critical path.
ACTUAL remaining consumer serial cost (T8, post-eager-chain, self-times): window_publish_marker
~122ms (= get_last_window extracting the last 32KiB on SegmentedU8 — ~3ms/chunk, far slower than a
contiguous 32KiB copy; THIS is the real Divergence #3 candidate = SEGMENTED access pattern vs vendor
contiguous FasterVector), dispatch_post_process ~144ms (into_chunk_data move + match + submit; NOT a
clone since misses=0 — needs decomposition), queue_prefetched_postproc ~116ms (the eager full-scan
itself — matches vendor structure; cost is the per-chunk get_last_window inside it).
NEXT: decompose get_last_window on SegmentedU8 (chunk_data.rs:969) — why ~3ms for a 32KiB tail?
Segment walk? materialize? If it's the segmented access pattern, that is the vendor-cited Divergence
#3 to port (contiguous tail access). Pre-register and MEASURE before porting (2 hypotheses already
falsified this session: interior-accept [no containing chunk], Arc-clone [misses=0]).

## GATE + ADVISOR 2026-06-05: decode no longer slack, but #3 NOT yet greenlit; eviction is NOT a vendor divergence
Ran the pre-registered slow-injection gate (GZIPPY_SLOW_BOOTSTRAP=N% spins N% of each decode's own
time AFTER decode, byte-exact, harness-plumbed). +100% decode => T8 wall +61% (0.988->1.591s),
T16 +48%. => decode is NO LONGER fully slack (the prior page-walk-REMOVE-oracle "0% wall / slack"
finding is OBSOLETE post-eager-chain). BUT advisor (correctly) flagged the slope is CONFOUNDED:
spinning ALL decodes (incl. prefetch) holds pool slots => the on-demand frontier decode queues
behind spinning prefetches = SCHEDULING contention, inflating the slope above true decode-criticality;
also +100% shifts eviction timing. And slope!=ceiling (CLAUDE.md): a slow-DOWN slope cannot greenlight
a speed-UP campaign. So #3 is NOT greenlit yet.
Advisor's strongest lead — "the 416ms cold re-decodes are a vendor CACHE-RETENTION divergence
(rapidgzip protects not-yet-consumed chunks from eviction)" — FALSIFIED by reading vendor Cache.hpp:
it is plain LeastRecentlyUsed, NO retention protection, identical to gzippy. Vendor avoids eviction
because its faster decode keeps the consumer in pace, not via policy. So the eviction is decode-speed-
driven (consumer pace), NOT a port-fidelity bug; a "pin soonest-needed" fix is a T3 non-vendor lever.
=> The 416ms residual path IS #3 (faster pure-Rust decode -> consumer keeps pace -> fewer evictions),
but it is gated, per advisor, by: (1) ON-DEMAND-ONLY slow-injection (slow only frontier decodes, not
prefetch) to separate the scheduling confound from true decode-criticality; (2) cold-re-decode COUNT
baseline vs perturbation (eviction-timing confound check); (3) a decode-REMOVAL oracle (decode~=0, CRC
gated off) to BOUND the speed-up ceiling. Only if the oracle ceiling is worth it does the deep #3
data-plane cycle start. Temp gate reverted (commit 7e7c00f). These 3 measurements are the next cycle's
disciplined gate before any #3 port.
