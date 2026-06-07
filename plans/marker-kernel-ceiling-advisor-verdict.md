# Disproof-advisor verdict — marker-kernel ceiling + apply_window parity (2026-06-07)

Read-only. Every premise below was source-verified first-hand against the cited
file:line (vendor deflate.hpp / DecodedData.hpp / ChunkData.hpp; gzippy
chunk_fetcher.rs / chunk_data.rs / segmented_buffer.rs / segmented_markers.rs).
No build/measure. The job was to BREAK the two findings, not ratify them.

## Source confirmations (what checked out)
- **deflate.hpp dispatch (:1428, :1451-1456).** With `LIBRAPIDARCHIVE_WITH_ISAL`
  the lit/len path is `readInternalCompressedMultiCached` for BOTH window types —
  confirmed; dispatch is by `CompressionType`/coding-type, NOT marker-vs-clean.
- **One loop, constexpr-gated marker arms.** `containsMarkerBytes` is
  `constexpr` from the element type (:1600, also :1523 in the non-multi variant);
  the only marker-specific work is the `m_distanceToLastMarkerByte` counter
  (:1311-1317), the post-memcpy back-scan (:1379-1389), and the *inverse* skip of
  the window-range check (:1652-1655). Confirmed verbatim.
- **resolveBackreference fast arm is `std::memcpy` for both (:1376).** Confirmed.
- **applyWindow is in-place narrow + swap + views, NO output-size copy.**
  `target[i]=fullWindow[chunk[i]]` over `reinterpret_cast<uint8_t*>(chunk.data())`
  (DecodedData.hpp:335-337), then `std::swap(reusedDataBuffers, dataWithMarkers)`
  (:368) and VectorViews built into the reused buffers (:371-388). Confirmed.
- **gzippy gather == rg gather, byte-for-byte algorithm.**
  `base.add(i).write(lut[v])` (segmented_markers.rs:489-494) ↔ rg
  `target[i]=fullWindow[chunk[i]]`; LUT build (identity 0..256 + window at
  MARKER_BASE, :470-476) ↔ rg fullWindow (iota + copy at MAX_WINDOW_SIZE).
  Confirmed identical.
- **The merge IS a full-output-size copy.** `prepend_narrowed_from_markers`
  (segmented_buffer.rs:356-379) allocates `n + self.buf.len()`, `extend_from_slice`s
  the narrowed marker bytes AND re-copies the existing clean `buf`. So it is even
  broader than the brief states: it recopies the clean tail too. rg's analogue is
  an O(1) `std::swap`. Confirmed structural divergence.
- **The iovec write path already supports the un-merged state.** Consumer at
  chunk_fetcher.rs:3674-3708 computes `payload_bytes = chunk.narrowed_len +
  decoded_data_len` and calls `append_output_iovecs` (chunk_data.rs:1609), which
  emits `append_narrowed_iovecs` when `narrowed_len>0` THEN data. Confirmed — the
  writer does not need unified `data`.
- **The window-extraction path already supports the un-merged state.**
  `copy_window_at_chunk_offset` (chunk_data.rs:1211-1278) explicitly branches on
  `narrowed_len>0` and reads `predecessor ‖ markers ‖ data` directly
  (`copy_narrowed_u8_range_into` / `resolve_range_into_buf`). Confirmed — getWindowAt
  does NOT need merge.

---

## Q1 — Is the rg deflate.hpp read fair (marker loop == clean loop, gap is engine not a separate path)? — **UPHELD-WITH-CAVEATS**

Fair in substance. rg runs the *same* `readInternalCompressedMultiCached` over a
u16 window for markers; the only deltas are cheap constexpr bookkeeping. There is
no separate slow marker algorithm in rg, so the measured 2× (decodeBlock 1.69×
this turn) is gzippy's engine, not a structural rg shortcut a faithful port can't
reach. Caveats that the brief overstates:

1. **"~as fast as clean" is loose.** rg's marker decode operates on `uint16_t`:
   appendToWindow stores 2-byte elements and `resolveBackreference` memcpys
   `length * sizeof(uint16_t)` = 2× the bytes (:1376). So rg's marker path is
   inherently ~2× the memory traffic of its *u8* clean path. It is "as fast as a
   u16 clean decode would be," not as fast as the real u8 clean decode. This does
   not change the conclusion (the gap is engine, and the faithful target is the
   multi-cached u16 loop), but a faithful port will still be heavier than the u8
   clean path by construction — markers are u16.
2. **m_window16 in-place vs gzippy's separate ring is NOT a structural rg
   advantage.** rg's "window" passed to readInternal *is* the growing
   dataWithMarkers buffer; backrefs resolve in-place within it via memcpy. gzippy's
   `data_with_markers` (SegmentedU16) is the same idea. The faithful port closes
   the gap by adopting rg's multi-cached decode + memcpy backref + constexpr
   bookkeeping, not by changing where the buffer lives. The named gzippy costs
   (per-symbol ring stores, reversed-bits distance decode, scalar back-ref tail)
   are implementation, not architecture — closable.

Net: the read is fair; the engine port is the right target. Just don't promise
"marker == u8-clean speed" — promise "marker == rg's u16 multi-cached loop."

## Q2 — Is the apply_window comparison apples-to-apples? — **UPHELD-WITH-CAVEATS**

Same denominator in the ways that matter: both SUM-across-chunks, both pool-side,
both timing the in-place u16→u8 narrow. rg's `applyWindowDuration`
(ChunkData.hpp:159, accumulated in merge) wraps gather + swap + view-build; gzippy's
gather sub-step wraps the narrow alone. Caveats:

1. **rg's 0.032s applyWindow already INCLUDES rg's merge-analogue** (the
   swap+views, which is ≈0 because there is no copy). gzippy's *gather* sub-step
   EXCLUDES merge. So the honest, fully apples-to-apples comparison is:
   rg applyWindow 0.032s  vs  gzippy gather+merge = 0.044-0.064 + 0.116-0.134.
   Under that framing the gap is dominated by merge exactly as the brief argues —
   so the conclusion stands, but the clean "gather 1.5-2×" line is comparing
   gzippy-gather to rg-(gather+swap), which flatters gzippy slightly (rg's swap is
   ~0, so it's nearly fair, but not exactly).
2. **CRC is correctly separated.** rg has `computeChecksumDuration` (ChunkData.hpp:160,
   checksum 0.0096s) distinct from applyWindow; gzippy's `update_narrowed_crc`
   (chunk_data.rs:1569) is the gzippy CRC sub-step (0.013-0.019s). Apples-to-apples,
   ~1.5×, small.
3. **Residual gather 1.5-2× is real and NOT closed by either named port.** If the
   algorithm is "identical," the residual is the SegmentedU16 multi-segment walk
   (pointer-chasing across segments) and per-chunk 64KiB LUT rebuild vs rg's single
   contiguous chunk + hoisted fullWindow. This is a third term (below).

## Q3 — Is the merge-copy truly REDUNDANT / removable byte-exactly? — **UPHELD-WITH-CAVEATS** (the strongest finding, but not a one-line delete)

Every consumer of the merged buffer was traced; none *fundamentally* requires
unified `data`:
- **Writer:** `append_output_iovecs` already emits narrowed-segments + data when
  `narrowed_len>0` (chunk_data.rs:1609-1615; consumer uses it at 3686/3736). ✓
- **Window extraction (getWindowAt / get_last_window / subchunk windows):**
  `copy_window_at_chunk_offset` already reads the 3-part view with a `narrowed_len>0`
  branch (chunk_data.rs:1220-1276). ✓
- **CRC:** `update_narrowed_crc` walks the marker segments directly
  (chunk_data.rs:1569-1583); the consumer combines `narrowed_crc` + per-stream
  `crc32s` (chunk_fetcher.rs:3688-3691). No unified walk. ✓
- **data_prefix_len>0:** the iovec path handles it (`append_narrowed_iovecs` then
  `append_payload_iovecs(data_prefix_len)`); ordering matches the merge's
  `insert_logical_at(data_prefix_len, …)` (narrowed then data-after-prefix). ✓

So the merge is genuinely redundant and removal is byte-exact in principle. BUT it
is NOT a deletion — removing it requires the rest of rg's view model:
1. **Lifetime/recycling.** Today `merge_resolved_markers_into_data` copies bytes
   out, then `recycle_markers_after_resolution` (chunk_fetcher.rs:2500) can free the
   u16 marker pages. Without merge, the narrowed bytes LIVE in those marker pages
   and must survive until the consumer's writev completes — i.e. marker-segment
   recycling must move behind `defer_chunk_recycle` (3700/3706), exactly mirroring
   rg's "the narrowed buffers ARE the output views, recycled on writeAll." Miss this
   and you get a use-after-recycle / data corruption, not a clean win.
2. **populate_subchunk_windows assert.** It hard-asserts `narrowed_len==0`
   (chunk_data.rs:1291-1294) — a *convention*, not a requirement, since
   copy_window_at_chunk_offset already supports `narrowed_len>0`. Must be relaxed
   and the call re-ordered before/independent of any merge.
3. **Order of operations in resolve_chunk_markers_on_chunk** (chunk_fetcher.rs:2484-2500):
   gather → crc → merge → subwin. Dropping merge means subwin runs against markers,
   and `narrowed_len` must stay set through consumer write.

Verdict: removable byte-exactly, faithful-to-rg (it converts gzippy to rg's
swap+views), but it is a small structured change across recycle + assert + ordering,
not `delete one call`.

## Q4 — Does fixing merge alone (−0.12s SUM) plausibly move the WALL, or is it slack-masked? — **UPHELD-WITH-CAVEATS (do NOT trust the SUM as the wall delta)**

Merge is partly on the serial path, so it is NOT purely slack-masked like the
Fill-90% engine — but the wall contribution is NOT the 0.12s SUM. Mechanism, traced:
- Post-process (gather+crc+merge+subwin) runs on the POOL — either ahead of time
  via resolve-ahead/eager (`submit_post_process_from_prefetch` →
  `run_post_process_in_place`, 2148/2666) or on demand
  (`submit_post_process_to_pool` → `run_post_process_task`, 1762/2131).
- For a head-of-line marker chunk that was NOT pre-resolved, the consumer thread
  BLOCKS on `recv_post_process_blocking` (chunk_fetcher.rs:1769) until the whole
  post-process — including merge — completes. There the per-chunk merge time is on
  the critical path.
- **But two things cap the upside:** (a) when resolve-ahead/eager HITS
  (`resolved_pred_matches` / `eager_completed` / `prefetch_post_inflight`, 1496-1514,
  1725-1730), merge already ran on the pool and is fully hidden; (b) even the blocked
  `recv_post_process_blocking` carries an `overlap` (1752-1758) that does useful
  pool-feeding while waiting, so blocked time is not pure stall.

So whether merge is wall-critical depends entirely on the resolve-ahead HIT RATE
for marker chunks — which is bounded by the serial dependency (a marker chunk needs
its predecessor's freshly published tail window; for the FIFO head that window is
often only just available, defeating ahead-of-time post-process — cf.
[[project_confirmed_offset_prefetch_gap]]). The brief's own Q4 caveat is correct and
must be honored: the SUM/parallelism is an upper bound only for overlapped chunks;
for serial head-of-line chunks the per-chunk merge is on the wall. **This is exactly
the case where the project's measurement rule binds: the −0.12s claim is a
hypothesis, provable ONLY by removing merge and measuring the interleaved wall
response (with a frequency-neutral control), never by citing the SUM.** Given merge
is the largest sub-step AND lands on the consumer's blocking-recv for un-pre-resolved
marker chunks, it is the single most likely apply_window lever to move the wall — but
"likely" is not "measured."

## Q5 — Overall: is "T8 TIE reachable in pure-Rust via (i) rg marker loop + (ii) rg view-based applyWindow" SOUND? — **UPHELD-WITH-CAVEATS**

The two-port decomposition is correct in shape and both ports are genuinely
faithful-to-rg (not gzippy-only optimizations), which is exactly what the charter
demands. The ceiling is plausibly sound. The caveats that keep it from being proven:

1. **A third, un-named term survives both ports: gather residual ~1.5-2×** (Q2.3) +
   **crc ~1.5×** (small). If gather's algorithm is identical, the residual is the
   SegmentedU16 multi-segment walk + per-chunk LUT rebuild vs rg's contiguous chunk +
   hoisted fullWindow. Neither "rg marker loop" nor "rg view-based applyWindow"
   addresses it. A faithful TIE may additionally require (iii) hoisting the
   fullWindow LUT build out of the per-chunk loop and/or a contiguous narrow target.
   This is small but non-zero and currently uncounted in the two-port ceiling.
2. **Merge removal's wall payoff is unproven** (Q4) and is gated on resolve-ahead
   hit rate, which is itself bounded by the serial predecessor-window dependency.
   The SUM is not the wall.
3. **Engine port upside is bounded by an oracle, not the slow-down slope**
   (CLAUDE.md rule 3). The 1.69× decodeBlock gap says the engine is heavy; it does
   NOT say closing it yields a TIE until the bootstrap-removed oracle sets the
   ceiling. Same for merge: remove-and-measure, don't extrapolate.

**Overall verdict: the bounded ceiling is DIRECTIONALLY SOUND and the two ports are
the right, faithful levers — but it is NOT yet a proven TIE.** It is a well-located
hypothesis with two genuine structural divergences identified (decodeBlock engine;
merge copy) plus one smaller residual the brief under-counts (gather/LUT). To
convert it to a TIE claim per the project's own rules: (a) land the merge removal as
the rg swap+views model — *including the deferred marker-recycle and the relaxed
subchunk assert* — and measure the interleaved T8 wall with a frequency-neutral
control and sha verification; (b) land the multi-cached marker loop and measure;
(c) re-check the gather/LUT residual once merge is gone. Until (a)-(c) each survive a
causal perturbation, the ceiling is a strong lead, not a result. No finding here is
refuted; the brief's two findings stand, with the three caveats above attached.

## Single most important correction to the brief
Stop quoting "−0.12s SUM" as the merge wall delta. The merge is on the pool; its
wall cost is only the un-overlapped fraction landing on the consumer's
`recv_post_process_blocking` for un-pre-resolved head-of-line marker chunks. That
fraction is real (so merge is the best apply_window lever) but it is unknown until
measured — and it is the thing to measure FIRST, because it's the cheapest of the
two ports to land and the one whose payoff is most uncertain.
