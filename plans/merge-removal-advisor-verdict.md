# Verdict — merge-removal (view-based applyWindow), independent disproof review

Read-only. Every premise source-verified first-hand against the cited file:line.
I tried to BREAK C1/C2/C3; below is what survived and what didn't.

## Vendor citation check (foundation for all three claims) — ACCURATE
`DecodedData.hpp:325-388` confirmed:
- 325-363: narrow markers IN PLACE — `target[i] = fullWindow[chunk[i]]`, writing u8
  into the u16 backing. No output-size copy.
- 368 `std::swap(reusedDataBuffers, dataWithMarkers)`; 371-386 build `dataViews` as
  a `VectorView<uint8_t>` per reused buffer (the narrowed bytes, reinterpret_cast),
  387 move existing `data` views AFTER them, 388 `std::swap(data, dataViews)`.
- **390 `dataWithMarkers.clear()`** — so vendor `containsMarkers` (which inspects
  `dataWithMarkers`) IS false post-`applyWindow`. The brief's premise is correct.

So the change is genuinely MORE faithful to vendor, not a divergence: gzippy's prior
`merge_resolved_markers_into_data` did a copy vendor never does. Narrowed‖data emit
order (narrowed iovecs first, then `data` payload) mirrors vendor's reused-views-then-
data-views order. ✓

---

## C1 BYTE-EXACT — **UPHELD**
- `contains_markers` narrowed-guard (chunk_data.rs:578-591) is REQUIRED and correct:
  after `resolve_and_narrow_in_place` the u16 backing's HIGH bytes are stale, so
  `all_resolved()` (segmented_markers.rs:551, reads u16) would misread → the
  `narrowed_len == 0 &&` short-circuit is the right fix and matches vendor's cleared
  `dataWithMarkers`.
- `copy_window_at_chunk_offset` (chunk_data.rs:1230-1275) branches on `narrowed_len>0`
  and reads u8 via `copy_narrowed_u8_range_into` (segmented_markers.rs:501, u8 read
  from u16 low bytes) — `getWindowAt` correct merged OR un-merged.
- `append_narrowed_iovecs` (segmented_markers.rs:532) reads u8 from the same low
  bytes the in-place narrow wrote — consistent.
- Emit order narrowed‖data == CRC append order (`narrowed_crc` then `crc32s`,
  chunk_fetcher.rs:3669-3672 / 3735-3738), and == the old merge order even with
  `data_prefix_len>0` (`insert_logical_at(data_prefix_len, …)` produced the same
  byte order). I could not re-run tests (read-only), but the design is byte-faithful.

## C2 WALL MOVED — **UPHELD-WITH-CAVEATS**
The removed work is real and on the critical path. `prepend_narrowed_from_markers`
(segmented_buffer.rs:356-379) is NOT a marker-sized copy: it allocates
`n + self.buf.len()` and `nb.extend_from_slice(&self.buf)` copies the **entire clean
payload** of the chunk. So "redundant full-output memcpy" is literally accurate — it
was an O(whole-chunk) alloc+copy per marker chunk, executed on the post-process
worker that the consumer blocks on (`recv_post_process_blocking`) for head-of-line,
un-eagerly-resolved marker chunks. Removing it plausibly yields +12%; 3 stable
interleaved sha-verified runs + load-invariance rule out a turbo artifact, and
sha-identical output rules out a wrong-but-fast artifact.
- CAVEAT (not a refutation): the change removes the full-chunk **copy** AND the
  per-chunk **allocation** together; the verdict does not isolate
  consumer-blocking-wait reduction from allocator-pressure reduction. The brief's
  Q4 mechanism is the most likely, but remove-and-measure (which was done) is the
  honest verdict, not the attribution.

## C3 CORRECTNESS of the un-merged state — **UPHELD-WITH-CAVEATS**
No use-after-recycle found across the three emit paths:
- **Pipe (Linux):** `write_chunk_payload_to_fd` boxes the whole chunk into the
  `SpliceVault` owner (fd_vectored_write.rs:531-534, 442-446), kept alive until the
  pipe drains; `recycle_decoded_buffers` frees BOTH `data` and `data_with_markers`
  (chunk_data.rs:1642-1651), so the owner covers the new narrowed-in-`data_with_markers`
  iovecs. ✓
- **Non-pipe writev / non-Linux:** writev is synchronous (kernel copies before
  return); `iovs` are raw `libc::iovec` (no borrow), the `parts` borrow ends before
  the move; `defer_chunk_recycle` runs AFTER the write. ✓
- **Buffered fallback:** `write_all` synchronous, then defer. ✓
- Re-resolution gates hold WITHOUT relying on emptiness: consumer path guarded by
  `!chunk.markers_resolved` (chunk_fetcher.rs:1729); eager path by
  `chunk_may_resolve_markers_early`→`!has_been_post_processed` (2512), which is now
  true post-narrow via the `contains_markers` guard.

### SINGLE MOST IMPORTANT CORRECTION (applies to C3)
The change **silently defeated two defense-in-depth guards** against double-resolution
that previously worked only because `merge` left `data_with_markers` EMPTY:
1. the self-guard `if dwm_len_pre > 0` at chunk_fetcher.rs:2458-2466 inside
   `resolve_chunk_markers_on_chunk`, and
2. `if arc.data_with_markers.is_empty() { continue; }` at chunk_fetcher.rs:2592 in
   `queue_prefetched_marker_postprocess`.
Both are now permanently bypassed (`data_with_markers` stays non-empty). A SECOND
`resolve_and_narrow_in_place` on already-narrowed bytes would feed stale u16 high
bytes through the 64 KiB LUT → **silent byte corruption**, and `all_resolved()` no
longer catches it either. Correctness now rests SOLELY on the `markers_resolved` /
`has_been_post_processed` gating, which currently appears sufficient — but the
tripwire is gone. Recommend restoring one: add
`debug_assert!(self.narrowed_len == 0 && !self.markers_resolved)` at the top of
`resolve_and_narrow_markers_in_place` (chunk_data.rs:1576) so any future caller that
resolves twice fails loudly in test/debug instead of emitting corrupt-but-CRC-... no,
CRC would catch it in prod, but only after wasted work and as a terminal Err with
partial output already written.

Secondary note (not blocking): marker u16 buffers now live longer (until consumer
writev + DEPTH=2 defer) instead of being recycled eagerly on the worker — more
resident memory for marker-heavy inputs. It measured faster, so not a regression,
but it is a real residency tradeoff worth recording.
