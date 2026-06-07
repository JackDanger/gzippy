# Disproof-advisor brief — marker-kernel ceiling + apply_window parity (2026-06-07)

Read-only. Source-verify against the cited file:line. No build/measure/edit. All numbers
below are first-hand from the locked guest (ssh neurotic → ssh 10.30.0.199), 16c gov=perf,
turbo on, load ~1.0, taskset -c 0,2,4,6,8,10,12,14, T8, /root/silesia.gz (RAW=68229982),
binary /tmp/gzbuild-isal/release/gzippy (gzippy-isal, target-cpu=native), sha
028bd002…cb410f verified EVERY run. rapidgzip 0.16.0 --verbose same host.

## Claim A — rapidgzip's u16 MARKER decode and CLEAN decode are the SAME fast loop (faithful target found)
Source (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp):
- `readInternal` (:1428) dispatches by Huffman-coding TYPE, NOT by marker-vs-clean. With
  LIBRAPIDARCHIVE_WITH_ISAL the lit/len path is `readInternalCompressedMultiCached`
  (:1453) for BOTH u16 and u8 windows (templated on `Window`).
- `readInternalCompressedMultiCached` (:1589) is one loop; `containsMarkerBytes` is a
  `constexpr` from the window element type (:1600). Marker (u16) vs clean (u8) differ ONLY
  in `if constexpr (containsMarkerBytes)` arms: appendToWindow tracks
  m_distanceToLastMarkerByte (:1311-1317, a counter), resolveBackreference back-scans for
  the last marker after the SAME memcpy (:1379-1389), and the window-range check is skipped
  for markers (:1652-1655 inverse).
- resolveBackreference (:1349): the fast non-overlap arm is a plain `std::memcpy` for BOTH
  u16 and u8 (:1376). So rg's marker decode is ~as fast as clean.

⇒ rg's marker decode is fast because it (a) runs the multi-cached fast loop on the u16
window too, and (b) only differs by cheap constexpr-gated marker bookkeeping. The 2× gap is
NOT a separate slow marker path in rg — it is gzippy's marker loop being slower than rg's
SAME-shaped loop (per-symbol u16 ring stores + reversed-bits dist decode + scalar back-ref
tail), the documented engine gap (decodeBlock 1.69× this turn, see Claim C).

## Claim B (the OWED measurement) — apply_window is a SECOND term; gzippy's marker-resolution SUM ≈ 0.19-0.27s vs rg's applyWindow 0.032s, BUT the gap is NOT the LUT gather
- rg --verbose (first-hand THIS turn): "Time spent applying the last window" = **0.0322 s**
  (NOT the charter's cached 0.113s — that number is WRONG/superseded). Checksum 0.0096s.
  decodeBlock 0.497s. Markers 34.4981%.
- gzippy total apply_window_us SUM (post_process_span timer wrapping
  resolve_chunk_markers_on_chunk, chunk_fetcher.rs:2648-2655): **0.19-0.27s**.
- DECOMPOSED that timer (measurement-only sub-step timers added this turn, byte-exact, sha
  unchanged) into its four sub-steps (SUM across 15 marker chunks, 3 runs):
    | sub-step | gzippy SUM | rg equivalent | ratio |
    | gather (LUT resolve+narrow, segmented_markers.rs:481-497) | 0.044-0.064s | applyWindow 0.032s | ~1.5-2× |
    | crc (update_narrowed_crc) | 0.013-0.019s | checksum 0.0096s | ~1.5× |
    | **merge_resolved_markers_into_data** | **0.116-0.134s** | std::swap (~0s) | **∞** |
    | subwin (populate_subchunk_windows) | 0.010-0.012s | window export (separate) | — |
- The gather (rg's actual applyWindow analogue) is only ~1.5-2× — algorithm is IDENTICAL
  (gzippy resolve_and_narrow_segments_in_place `base[i]=lut[v]` ↔ rg applyWindow
  `target[i]=fullWindow[chunk[i]]`, DecodedData.hpp:335-337; same 64KB LUT, same in-place
  u16→u8 downcast, same anti-reorder per-element loop). The dominant cost is **merge**.

## Claim C — merge_resolved_markers_into_data is a STRUCTURAL DIVERGENCE from rg (a full output-size memcpy rg does not do)
Source:
- gzippy: merge_resolved_markers_into_data (chunk_data.rs:1589) → when data_prefix_len==0
  (the common window-absent case) calls prepend_narrowed_from_markers
  (segmented_buffer.rs:356) which allocates a fresh n-byte buffer and `extend_from_slice`
  COPIES every narrowed byte out of the marker segments into `data`. n = full chunk decoded
  size ⇒ a ~68MB total memcpy of the whole output (0.12-0.13s SUM).
- rg: DecodedData::applyWindow (DecodedData.hpp:368) `std::swap(reusedDataBuffers,
  dataWithMarkers)` then builds VectorViews INTO the marker buffers in place (:371-388). NO
  output-size copy. The resolved u8 already lives in the low half of the marker pages.
- gzippy ALREADY HAS a zero-copy emit path: append_output_iovecs (chunk_data.rs:1609) →
  append_narrowed_iovecs when narrowed_len>0 (segmented_markers.rs:532) emits the narrowed
  marker segments directly as &[u8]. ⇒ the merge-copy is REDUNDANT for the iovec writer.

## The two findings I want disproved
1. CEILING: porting rg's marker decode loop closes the decodeBlock 2× term, but the T8 TIE
   ALSO requires fixing the apply_window second term — and that term is dominated NOT by the
   LUT gather (which is ~1.5-2× and algorithmically identical) but by the redundant
   merge memcpy (Claim C), a structural divergence from rg's std::swap+views. So the scoped
   fix is TWO faithful ports: (i) rg's multi-cached marker loop (decodeBlock), (ii) rg's
   view-based applyWindow that skips the output-size copy (merge).
2. apply_window PARITY: gzippy's marker-resolution is materially WORSE than rg's, but the
   excess is a removable copy (faithful to rg = a swap), not an irreducible algorithmic gap.

## Disproof questions
Q1. Is the rg deflate.hpp read is fair (marker loop == clean loop, gap is engine not a
    separate path)? Any vendor detail that makes rg's marker path structurally cheaper than
    a faithful gzippy port could be (e.g. m_window16 in-place vs gzippy's separate ring)?
Q2. Is the apply_window comparison apples-to-apples? rg's "applying the last window" 0.032s
    vs gzippy's GATHER 0.044-0.064s — same denominator (both SUM-across-chunks, both pool-
    parallel; rg ChunkData.hpp:159 applyWindowDuration += ...; gzippy per-chunk SUM)? Does
    rg's number exclude what gzippy's gather includes (narrowing)?
Q3. Is the merge-copy truly REDUNDANT / removable byte-exactly? append_narrowed_iovecs
    exists, but does any consumer (CRC walk, subchunk windows, multi-member, partial-clean-
    prefix chunks data_prefix_len>0) REQUIRE data to be unified? Is the data_prefix_len>0
    branch (insert_logical_at) a separate case that still needs a copy?
Q4. Does fixing merge alone (−0.12s SUM) plausibly move the WALL, or is it slack-masked
    like the engine was at Fill 90%? (It is a SUM; wall contribution ≈ SUM/parallelism only
    if it overlaps; the post-process runs on the pool AND partly on the consumer head —
    chunk_fetcher.rs:2678 run_post_process_task is consumer-thread serial for the head
    chunk. Is merge on the serial consumer path ⇒ un-overlapped ⇒ full wall cost?)
Q5. Overall: is the bounded ceiling "T8 TIE reachable in pure-Rust, via (i) rg marker loop
    + (ii) rg view-based applyWindow" sound, or does some term remain that neither closes?

Write verdict to plans/marker-kernel-ceiling-advisor-verdict.md.
