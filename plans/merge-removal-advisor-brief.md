# Advisor brief — merge-removal (view-based applyWindow) landed + measured

You are an INDEPENDENT DISPROOF advisor (read-only). Your job is to BREAK the
claims below, not ratify them. Source-verify every premise first-hand against the
cited file:line. Reply with a verdict per claim (UPHELD / UPHELD-WITH-CAVEATS /
REFUTED) and a single most-important correction.

## THE CHANGE (committed this turn, branch reimplement-isa-l)
Faithful port of rapidgzip's view-based `applyWindow` (vendor DecodedData.hpp:365-388
= narrow → std::swap + in-place VectorViews, NO output-size copy). gzippy previously
did a redundant full-output memcpy in `merge_resolved_markers_into_data`
(chunk_data.rs:1589 → segmented_buffer.rs:356 `prepend_narrowed_from_markers`).

Diff (2 files):
1. `resolve_chunk_markers_on_chunk` (chunk_fetcher.rs:2453-2475): REMOVED the calls
   `chunk.merge_resolved_markers_into_data()` and `chunk.recycle_markers_after_resolution()`.
   Order is now: fused resolve+narrow → update_narrowed_crc → populate_subchunk_windows
   → markers_resolved=true. The narrowed marker bytes STAY in `data_with_markers`
   (u8 view of the u16 backing) with `narrowed_len` set; the consumer emits them
   zero-copy via `append_output_iovecs`→`append_narrowed_iovecs` (chunk_data.rs:1609,
   already supports narrowed_len>0). Marker-segment recycle is now DEFERRED behind
   the consumer writev via the existing `defer_chunk_recycle`→`recycle_decoded_buffers`
   (chunk_fetcher.rs:3486 / chunk_data.rs:1621, which frees BOTH data and data_with_markers).
2. `contains_markers` (chunk_data.rs:577): now returns false when `narrowed_len>0`
   (resolved-but-unmerged), because after narrow the u16 elements hold stale high
   bytes so `all_resolved()` would misread them. `has_been_post_processed` depends
   on `!contains_markers()`.
3. `populate_subchunk_windows`: relaxed the `narrowed_len==0` debug_assert (the
   un-merged state is now legal; copy_window_at_chunk_offset already branches on
   narrowed_len>0 at chunk_data.rs:1220).

## CLAIMS TO BREAK
- C1 BYTE-EXACT: gzippy-isal native, silesia T1+T8 path=ParallelSM both sha
  028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f (guest + local
  gzippy-native arm64 too). 856 lib tests pass (the 1 failure is the pre-existing
  flaky timing micro-test diff_ratio, fails identically on unmodified 507d6ecb).
  New test populate_subchunk_windows_unmerged_view_based_apply_window locks the
  un-merged path. Adversarial seam test + native_fold_parity green.
- C2 WALL MOVED (remove-and-measure, NOT the SUM): locked guest REDACTED_IP double-ssh,
  16c gov=performance turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved
  N=11, RAW=68229982, sha-verified every run. base(with merge) vs mergefix(removed):
  run1 0.2291→0.2045 (+12.0%); run2 0.2128→0.1900 (+12.0%); run3 0.2006→0.1765 (+13.7%).
  rg ratio: base ~0.65× → mergefix ~0.73×. Sign stable across 3 runs; load-invariant
  (1.64/2.80/1.86) ⇒ not a turbo artifact (interleaved = freq-neutral by construction).
- C3 CORRECTNESS of the un-merged state: every consumer (writev pipe + non-pipe,
  buffered fallback, getWindowAt, CRC, has_been_post_processed gates) handles
  narrowed_len>0; the marker pages outlive the writev via defer_chunk_recycle; no
  use-after-recycle. The eager/resolve-ahead re-resolution gates (chunk_fetcher.rs
  1727 !markers_resolved, 2512 has_been_post_processed, 2586/2592) don't double-resolve.

## WHAT TO CHECK HARDEST
- Is there ANY path where a resolved-but-unmerged chunk's narrowed bytes are freed
  (recycle_decoded_buffers / recycle_markers / Drop) BEFORE the writev iovecs that
  borrow them complete? (pipe vmsplice keeps Box<chunk> alive; non-pipe writev is
  synchronous then recycles; buffered path defers.)
- Does contains_markers()=false-on-narrowed break any place that NEEDS to know
  markers physically remain in data_with_markers (e.g. a re-resolution, a window
  walk that assumes empty)?
- Is the +12% wall delta plausibly the merge memcpy landing on the consumer's
  blocking recv for un-pre-resolved head-of-line marker chunks (advisor Q4 mechanism),
  or could it be an unrelated allocation/scheduling artifact?
