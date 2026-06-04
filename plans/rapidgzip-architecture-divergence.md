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
