# resolve-ahead fix — read-only investigation @ HEAD 85ad00a

All citations are `file:line` I opened. `cf` = `src/decompress/parallel/chunk_fetcher.rs`,
`vendor` = `vendor/rapidgzip/librapidarchive/src/rapidgzip/`.

---

## 0. HEADLINE — the F1 falsifier reads DEAD counters

`handoff_key` and `RESOLVE_AHEAD_OK`/`RESOLVE_AHEAD_ATTEMPTS` (the §5 F1 gating
falsifier) are **declared but never incremented anywhere in the tree**:

- `handoff_key=` in `--verbose` prints `HANDOFF_WINDOW_PUBLISHED` (cf:784).
- `Worker resolve-ahead: ok=/attempts=` prints `RESOLVE_AHEAD_OK`/`RESOLVE_AHEAD_ATTEMPTS` (cf:942-943).
- All three are defined at cf:2738-2744. `grep` for `HANDOFF_WINDOW_PUBLISHED.fetch`,
  `RESOLVE_AHEAD_OK.fetch`, `RESOLVE_AHEAD_ATTEMPTS.fetch` (and `.store`) returns **zero hits**.

They are guaranteed to read `0` whether or not resolve-ahead fires. **The premise
"handoff_key still 0 ⇒ 85ad00a's fix did not take effect" is unfounded — it is the
broken-instrument trap CLAUDE.md rule 4 warns about** (cf. the frozen clean-window oracle).
F1 as written is unfalsifiable.

The **live** resolve-ahead signal already exists and is printed but was not used in the
captured verdict:
- `Eager post-process: runs= runs_nonempty= inspected= submitted= max/run= reused=` (cf:668-676):
  `EAGER_PROBE_SUBMITTED` (cf:2450) = chunks resolve-ahead actually queued;
  `EAGER_PROBE_REUSED` (cf:1614) = consumer reused ahead-work (the payoff).
- `EAGER_PROBE_REUSE_PRED_MISMATCH` (cf:1618) / `EAGER_PROBE_REUSE_KEY_ABSENT` (cf:1622) = reuse failures.

**Before any code change, re-read these from the SAME report.** `submitted=0` ⇒ resolve-ahead
truly idle (key/scan bug below); `submitted>0, reused≈0` with high `pred_mismatch` ⇒ the key
divergence below; `submitted>0, reused>0` ⇒ resolve-ahead already works and the wall gap is
Lever B (decode rate), not this.

---

## 1. Central hypothesis — CONFIRMED on keys, REFUTED on its evidence

### (a) Publish key — `encoded_offset_bits + encoded_size_bits` (real chunk end)
`publish_end_window_before_post_process`: `window_offset = chunk.encoded_offset_bits +
chunk.encoded_size_bits` (cf:2378), inserted via `window_map.insert_owned_none(window_offset, …)`
(cf:2383/2386). For a range-speculative chunk the metadata was rewritten in
`try_speculative_decode_candidate`: `encoded_offset_bits = partition_seed` (=offset.first, cf:3149),
`encoded_size_bits = encoded_end - partition_seed` (cf:3150) ⇒ `window_offset = partition_seed +
(encoded_end - partition_seed) = encoded_end` = the **real bit where the chunk's decode ended**.
Subchunk windows likewise publish at `sc.encoded_offset_bits + sc.encoded_size_bits` (cf:2681,
`publish_subchunk_windows`; consumer side cf:2646) — all **real boundaries**.
Mirrors vendor `queueChunkForPostProcessing`: `windowOffset = chunkData->encodedOffsetInBits +
chunkData->encodedSizeInBits` (vendor `GzipChunkFetcher.hpp:557`).

### (b) Resolve-ahead lookup key — `max_acceptable_start_bit` (offset.second, speculative start)
`chunk_may_resolve_markers_early` → `window_map.get(chunk_consumer_handoff_bit(chunk))` (cf:2348),
and the submit path `handoff_bit = chunk_consumer_handoff_bit(arc)` →
`confirmed_predecessor_window(window_map, handoff_bit)` (cf:2428-2430).
`chunk_consumer_handoff_bit(chunk) = chunk.max_acceptable_start_bit` (cf:2335-2337).
`max_acceptable_start_bit = decode_start` (=offset.second, the actual block-aligned bit the worker
seeked to and decoded byte 0 from) — set at cf:3151.

### (c) Range-speculative chunk N with predecessor P — keys
- P publishes at `P.encoded_offset_bits + P.encoded_size_bits = P.encodedEnd` (P's real end).
- N's resolve-ahead looks up at `N.max_acceptable_start_bit` (=N's offset.second, N's speculative
  decode-start).
- N's `encoded_offset_bits` = N's partition **seed** (offset.first) — generally ≠ P.encodedEnd.
- **They coincide only when N's speculation landed exactly on the true boundary**
  (`N.offset.second == P.encodedEnd`, contiguous + correctly-speculated). For a mis-speculated or
  deep-prefetch chunk they differ ⇒ lookup misses. So the "NEVER coincide" claim is too strong:
  *coincide iff speculation was correct; miss otherwise.*

### (d) Consumer's WORKING key — block-finder-confirmed real boundary, not a chunk field
Consumer `handoff_bit = decode_start` (cf:1410); for a confirmed block
`decode_start = next_block_offset = block_finder.get(next_unprocessed_block_index)` (cf:1049,1071-1072).
Real boundaries were inserted into the block finder by the predecessor's
`consumer_append_subchunks_vendor`: `block_finder.insert(chunk_end_bit)` (cf:2643) /
`block_finder.insert(sc.encoded_offset_bits + sc.encoded_size_bits)` (cf:2646) — mirror of vendor
`m_blockFinder->insert(subchunk.encodedOffset + subchunk.encodedSize)` (`GzipChunkFetcher.hpp:374`).
So `decode_start` == predecessor's real end == the published key (a). The consumer then resolves
the window at exactly that key: `confirmed_predecessor_window(window_map, handoff_bit)` (cf:1448,1546).
**The consumer never keys on a stored chunk offset field; it keys on the block-finder's confirmed
real boundary, which by construction equals the publish key.** That is why the consumer always
matches and output is correct.

**Verdict:** central key observation CONFIRMED (publish=real-end; resolve-ahead=speculative-start;
the two stored chunk fields — seed and speculative-start — are not guaranteed to equal the
predecessor's published real-end). The *evidence offered for it* (`handoff_key=0`) is REFUTED:
that counter is dead.

---

## 2. Root cause (one sentence)

Resolve-ahead keys the predecessor-window lookup on a **stored field of the chunk being resolved**
(`max_acceptable_start_bit` at HEAD; `encoded_offset_bits` in vendor) which equals the
predecessor's *published* real-end key (`pred.encoded_offset_bits + pred.encoded_size_bits`) only
when speculation was exactly correct — whereas the consumer succeeds because it keys on the
block-finder-**confirmed** real boundary; the gap is hidden behind a dead `handoff_key` counter
that makes the failure unobservable.

---

## 3. Precise minimal faithful fix

### 3.0 (BLOCKING, do first) Make the instrument live
Either re-read `EAGER_PROBE_SUBMITTED`/`EAGER_PROBE_REUSED`/`EAGER_PROBE_REUSE_PRED_MISMATCH`
(cf:668-676, already printed) as the F1 signal, **or** wire the dead counters:
- `EARLY_WINDOW_PUBLISHED`/`PUBLISH_AHEAD_WINDOWS` already bump at cf:2388-2389; add
  `HANDOFF_WINDOW_PUBLISHED.fetch_add(1, …)` beside the resolve-ahead submit at cf:2447, and
  `RESOLVE_AHEAD_ATTEMPTS`/`RESOLVE_AHEAD_OK` around the `confirmed_predecessor_window` call at
  cf:2429-2433 (attempt = entered the lookup; ok = `Some`).
No key change can be evaluated until F1 measures something real.

### 3.1 Key alignment (apply only if §3.0 re-measure shows `submitted≈0` or `pred_mismatch` high)
Make resolve-ahead look up the **predecessor's published key**, which is exactly what predecessors
emplace under (cf:2378) and what the consumer's reuse comparison expects.

Vendor mechanism mirrored: `queuePrefetchedChunkPostProcessing` scans the prefetch cache **sorted
by offset** (`GzipChunkFetcher.hpp:524-529`) and, for each chunk, looks up the window that its
immediate predecessor emplaced at `encodedOffsetInBits + encodedSizeInBits`
(`GzipChunkFetcher.hpp:557`). In gzippy the sorted scan already exists
(`prefetch_cache_contents_sorted`, block_fetcher.rs:583-587, sorts ascending by cache key); thread
the previous element through and key off it.

`queue_prefetched_marker_postprocess` (cf:2396-2456). The predecessor handle **is available** — it
is the previous element of the offset-sorted `contents` loop (cf:2411). Track it:

Before (cf:2411, 2419, 2428-2430):
```rust
for (cache_key, arc) in contents {
    …
    if !chunk_may_resolve_markers_early(arc.as_ref(), window_map) { continue; }   // get(max_acceptable_start_bit)
    …
    let handoff_bit = chunk_consumer_handoff_bit(arc.as_ref());                   // = arc.max_acceptable_start_bit
    let Some((pred_key, predecessor_window)) =
        confirmed_predecessor_window(window_map, handoff_bit) else { continue; };
```
After:
```rust
let mut prev: Option<ChunkArc> = None;
for (cache_key, arc) in contents {
    // predecessor's published key == the bit the predecessor emplaced its tail window under (cf:2378)
    let handoff_bit = match prev.as_ref() {
        Some(p) => p.encoded_offset_bits + p.encoded_size_bits,   // vendor windowOffset, GzipChunkFetcher.hpp:557
        None => chunk_consumer_handoff_bit(arc.as_ref()),         // frontier seed: keep current behavior
    };
    prev = Some(Arc::clone(&arc));
    …
    if window_map.get(handoff_bit).is_none() { continue; }        // replaces chunk_may_resolve_markers_early window-probe
    …
    let Some((pred_key, predecessor_window)) =
        confirmed_predecessor_window(window_map, handoff_bit) else { continue; };
```
(`chunk_consumer_handoff_bit` / `chunk_may_resolve_markers_early`'s `has_been_post_processed` guard
at cf:2345 stay; only the *window key* changes.)

Why this is the faithful key and not a new scheme: the resulting `handoff_bit` equals the publish
key (cf:2378 ↔ vendor:557), and therefore equals the consumer's `decode_start` for the same link
(§1d). The `pred_key` then stored in `in_flight` (cf:2446) becomes the **real boundary**, so the
consumer's reuse test `consumer_pred_key == Some(eager_pred_key)` (cf:1612, where
`consumer_pred_key` = `decode_start` real boundary, cf:1546-1552) can succeed —
which `max_acceptable_start_bit` only achieved on correct speculation.

Note this is *more robust* than literal vendor (`get(chunkData->encodedOffsetInBits)`,
`GzipChunkFetcher.hpp:544`): gzippy keeps `encoded_offset_bits` pinned at the partition seed for
cached chunks (cf:3149; never re-anchored until consumer `set_encoded_offset`), so the literal
vendor field would key on the seed and miss. Vendor gets away with `encodedOffsetInBits` because
its BlockFinder yields the real offset for confirmed indices; gzippy reproduces that same
coincidence by reading it off the predecessor's published key directly.

Do **not** switch workers to clean-decode; this changes only the resolve-ahead lookup key.

---

## 4. Falsifiable prediction (after §3.0, and §3.1 if applied)

With the instrument live:
- `EAGER_PROBE_SUBMITTED > 0` and `EAGER_PROBE_REUSED > 0` (and, if wired, `handoff_key`/
  `RESOLVE_AHEAD_OK > 0`); `EAGER_PROBE_REUSE_PRED_MISMATCH` drops toward 0.
- `consumer.dispatch_recv` / `wait.future_recv` wall-critical drops from ~278/231 ms toward
  rapidgzip's ~3 ms; `model` `L_resolve` **median** drops from ~2.67 ms toward ~0.065 ms.
- **GUARD (reject if violated):** causal `window_present:false` must stay ~90% (was 38/42). The
  fix touches only post-process scheduling, not the worker decode path, so the window-absent
  fraction must NOT drift toward the 31% clean fraction — that would mean workers turned into
  clean-decoders (the forbidden divergence).
- If `submitted>0, reused>0` already at HEAD (instrument was the only thing broken), then the
  residual wall gap is decode-rate (Lever B, `d_w` 1.77×), to be confirmed by a
  `GZIPPY_SLOW_BOOTSTRAP` perturbation, not by this lever.

---

## 5. Caveats / disproof attempts I could not close read-only

- **Cannot run the re-measure** (read-only), so I cannot state whether HEAD's
  `EAGER_PROBE_SUBMITTED` is already >0. The whole §3.1 key change is conditional on that number;
  if resolve-ahead already fires and reuses, §3.1 is unnecessary and the story is Lever B.
- **`prev` = true predecessor assumption.** The fix treats the offset-sorted-scan previous element
  as the contiguous predecessor. This holds for a dense contiguous partition (matches vendor's
  `sortedOffsets` walk) but a gap (failed/absent prefetch, an overshoot chunk that swallowed a
  partition index, cf:1064-1067) would make `prev` non-adjacent and the `handoff_bit` wrong → that
  chunk simply fails the `window_map.get` guard and is skipped (safe: no incorrect resolve, just a
  missed ahead-opportunity, same as today). I could not enumerate gap frequency read-only.
- **Vendor literal vs robust.** I assert §3.1 (predecessor-published-key) is more robust than the
  literal vendor field, based on gzippy pinning `encoded_offset_bits` to the seed (cf:3149) where
  vendor's BlockFinder would supply the real offset. I confirmed gzippy never re-anchors a cached
  prefetched chunk before consumption (only consumer `set_encoded_offset` at drain, cf:1656/3447),
  but did not exhaustively prove vendor's `encodedOffsetInBits` is the real offset at its line-544
  call — that inference rests on `appendSubchunksToIndexes`/BlockFinder insert (vendor:374) and the
  partition-offset `getBlock` path (vendor:597-654), read but not executed.
- **`published=39` is ambiguous.** `EARLY_WINDOW_PUBLISHED` (cf:2389) bumps in BOTH the consumer
  branches (cf:1450,1566) and resolve-ahead (cf:2437), so the captured `published=39` does not by
  itself prove resolve-ahead published any window. Only `EAGER_PROBE_SUBMITTED` disambiguates.

INVESTIGATION COMPLETE
