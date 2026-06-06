# Structural gap: gzippy vs rapidgzip (parallel single-member)

**Branch:** `reimplement-isa-l` (pure-rust-inflate production path)  
**Ground truth:** locked Fulcrum on neurotic, silesia-large, frozen host, byte-exact  
**Wall today:** gzippy ~2.0–2.6× slower than rapidgzip at T8–T16 (trace ~955 ms vs ~463 ms; interleaved min ~1.2 s vs ~0.48 s)

This document maps **structure** (vendor `vendor/rapidgzip/librapidarchive/src/rapidgzip/` ↔ gzippy `src/decompress/parallel/`), not attribution guesses. Fulcrum proposes; **causal perturbation** (slow-inject, oracle removal) disposes.

---

## 1. What already matches (control plane)

Phases 1-3 from `plans/structural-gap-closure.md` are landed: decode dispatch,
block-finder seek semantics, and window-map cleanup now match the vendor control
plane. The remaining gaps are data-plane cost, scheduler parity, and instrument
gating.

| Subsystem | Vendor | gzippy | Status |
|-----------|--------|--------|--------|
| Prefetcher loop | `BlockFetcher.hpp` | `block_fetcher.rs` | Aligned (+ cache-pollution guard ported; **inert** on silesia — see §5) |
| In-order consumer | `GzipChunkFetcher::processNextChunk` | `consumer_loop` | Aligned |
| Window map keying | `WindowMap::get(offset)` at chunk **start** | `insert` at chunk **end** (= next start) | Equivalent when chain exact |
| Eager marker resolve chain | `queuePrefetchedChunkPostProcessing` | `queue_prefetched_marker_postprocess` (full scan) | Aligned |
| Speculative boundary search | `GzipChunk::decodeChunk` + `tryToDecode` | `speculative_decode_find_boundary` | Aligned shape |
| Post-process | `applyWindow` on pool | `apply_window` / fused LUT | Aligned |
| CRC / ISIZE | worker CRC + consumer combine | same | Aligned |
| Chunk metadata | `encoded` / `max` / `tryToDecode` rewrite | `encoded_offset_bits` / `max_acceptable_start_bit` / partition-seed rewrite | Aligned |

---

## 2. Worker lookup and decode dispatch

### rapidgzip

`GzipChunkFetcher::decodeBlock` (`GzipChunkFetcher.hpp:712`):

```cpp
auto sharedWindow = m_windowMap->get( blockOffset );  // exact key only
return GzipChunk::decodeChunk( ..., blockOffset, untilOffset, sharedWindow, ... );
```

- **Window hit:** `decodeChunkWithRapidgzip` with 32 KiB dict.
- **Window miss:** `decodeChunk` → `tryToDecode({blockOffset, blockOffset})` then
  8 KiB block-finder walk (`GzipChunk.hpp:712-841`).

### gzippy

`run_decode_task` (`chunk_fetcher.rs`):

1. `window_map.get(start_bit)` — exact lookup only, same keying rule as vendor.
2. Else `speculative_decode_find_boundary` → `try_speculative_decode_candidate`
   using the partition seed / seek-bit pair rules from the block finder.
3. No worker-side speculative publish, `get_at_worker`, `pred`, or handoff-only
   lookup path remains in the production control plane.

### Structural delta

| Mechanism | rapidgzip | gzippy |
|-----------|-----------|--------|
| Predecessor lookup | `get(blockOffset)` exact | `get(start_bit)` exact |
| Miss path | Block finder + `tryToDecode` | `speculative_decode_find_boundary` + `try_speculative_decode_candidate` |
| Remaining open cost | clean/windowed tail + `applyWindow` chain | `deflate_block` bootstrap + segmented-tail publish |

**June 2026 alignment:** the worker lookup table is now the vendor-shaped exact
`get(start)` path only. The earlier `get_at_worker` / worker-publish / H' /
pred exploration is not part of the current production design and should not be
read as an active gap.

**Still open for wall parity:** bootstrap speed (`deflate_block` vs vendor
`deflate::Block`), consumer `get_last_window` on segmented storage, and
post-process / publish-chain cost.

---

## 3. Decode engine (secondary — large busy, not always wall-critical)

| Phase | rapidgzip | gzippy | Fulcrum signal |
|-------|-----------|--------|----------------|
| Unified chunk loop | `deflate::Block` + `decodeChunkWithRapidgzip` | `decode_chunk_unified_marker` (ported) | Structure aligned |
| Marker bootstrap | Inside unified loop | `deflate_block::Block` pure Rust | `worker.block_body` **+4992 ms** busy vs rg **0** |
| Clean tail | ISA-L `IsalInflateWrapper` | `ResumableInflate2` / pure bulk | `worker.isal_stream_inflate` rg **+1303 ms** busy; gzippy often **0** (stuck in bootstrap) |
| Window-present fast path | `setInitialWindow` | `window_seed_enabled` + `WINDOW_SEEDED_CHUNKS` | Only **~3** chunks/trace hit clean-at-seed |

**Insight:** gzippy spends worker time in **bootstrap** because windows are missing at decode start; rapidgzip spends it in **ISA-L bulk**. Unifying engines (P2) helps fidelity; **window hit rate** is the wall lever.

Causal: `GZIPPY_SLOW_BOOTSTRAP` +100% → wall +48–61% (decode on critical path post-eager-chain).

---

## 4. Publish-chain / marker resolve

| | rapidgzip | gzippy (post-eager-chain) |
|--|-----------|---------------------------|
| Serial cost | `getLastWindow` on consumer (~32 KiB) | `get_last_window` on `SegmentedU8` ~**3 ms/chunk** |
| Parallel cost | `applyWindow` on pool | fused LUT `post_process` ~**298 ms wc** vs rg **~215 ms** busy |
| Model binding | `N·L_resolve` when chain slow | Fulcrum model: publish-chain bound at T8 |

**Divergence #3 (data plane):** `ChunkData::getLastWindow` on segmented storage vs vendor `FasterVector` contiguous tail — suspected **~122 ms** `window_publish_marker` serial tax. Arc deep-clone hypothesis **falsified** (`try_unwrap` misses=0).

---

## 5. Prefetch / scheduling (not the silesia wall lever)

| Item | Vendor | gzippy | Measured |
|------|--------|--------|----------|
| Cache pollution guard | `nextNthEviction` + touch (`BlockFetcher.hpp:474–551`) | Ported | `pollution_stops=0`, **inert** |
| Prefetch saturation | — | `saturated=102+` | 4 cold on-demand decodes = **missed prefetch**, not eviction (`Useless Prefetches=0%`) |
| Placement vs rate | — | Fulcrum schedule | **100% RATE** — frontier decode speed, not idle workers |

---

## 6. Consumer accept / chain invariant

| | rapidgzip | gzippy |
|--|-----------|--------|
| Offset match | `matchesEncodedOffset`: `encoded ≤ offset ≤ max` (`ChunkData.hpp:397–402`) | `matches_encoded_offset` range OK |
| Prefetch accept | Range + chain | **Strict:** `max_acceptable_start_bit == decode_start` only (`chunk_fetcher.rs:1251`) |
| Risk | Chain enforced by construction | Guard-rejects when spacing overshoots; safe but more on-demand re-decodes |

Porting vendor range-accept without chain invariant risks corruption (`plans/wall-progress.md` § overshoot-prefix).

---

## 7. Data-plane output (bounded ceiling)

| | rapidgzip | gzippy |
|--|-----------|--------|
| Output | `DecodedData` segmented `FasterVector` | `SegmentedU8` + `writev` (shipped `0a448d1`) |
| Fulcrum | — | `consumer.writev` ~**0.6 ms** self — **slack**, not wall lever |
| Lone mid-stream emit | vendor pair drain | **DISPROVED** — CRC at chunk boundary |

---

## 8. Priority order (structural, measurement-backed)

1. **Window-present decode rate** — faster publish-chain so exact `get(start)`
   hits sooner and reduces bootstrap share.
2. **Bootstrap / inner loop speed** — `deflate_block` toward ISA-L-class; causal bootstrap perturbation confirms ceiling work.
3. **`get_last_window` / segmented tail** — contiguous 32 KiB extraction for consumer serial publish (~122 ms).
4. **Marker resolve** — pool post-process parity (partial; eager chain done).
5. **Prefetch prediction** — saturation / confirmed-offset gap (4 cold decodes); secondary on silesia.
6. **Chain accept policy** — only after chain invariant proven end-to-end.

**Explicitly not levers (disproved):** writev, pair-drain lone emit, consumer Arc clone, cache-pollution guard on silesia, interior-accept for evicted chunks, Fulcrum Δbusy without perturbation.

---

## 9. References

- `plans/rapidgzip-architecture-divergence.md` — divergence #1–#4 history
- `plans/sm-parity-gap-matrix.md` — row-level closure designs A–R
- `plans/wall-progress.md` — causal verdicts only
- `docs/fulcrum-sota.md` — instrument semantics
- Vendor anchors: `GzipChunkFetcher.hpp`, `GzipChunk.hpp:712–841`, `WindowMap.hpp:79–90`, `ChunkData.hpp:397–402`
