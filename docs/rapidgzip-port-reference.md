# rapidgzip → gzippy port: living reference

**Purpose**: persistent ground-truth artifact for the parallel single-member
decoder port. Whenever a change to `src/decompress/parallel/*` produces an
unexpected result or forces a judgment call, consult this document first —
it captures (1) what rapidgzip actually does (with C++ line citations),
(2) where gzippy currently differs, and (3) what tests/probes reveal each
gap.

If a change appears to "work" but contradicts this document, the change is
suspect — either the document is stale (in which case update it with the
new ground truth) or the change is masking a problem behind correct output
on a too-easy fixture.

---

## 1. rapidgzip target architecture

Authoritative sources, all under `vendor/rapidgzip/src/rapidgzip/`:

| Component | File | Key entry point |
|-----------|------|-----------------|
| Parallel dispatcher | `GzipChunkFetcher.hpp` | `processNextChunk` (~L311), `getBlock` (~L591) |
| Prefetcher / LRU cache | `BlockFetcher.hpp` | `get` (~L246), prefetch loop (~L314) |
| Per-chunk decoder | `chunkdecoding/GzipChunk.hpp` | `decodeChunk` (L661), `decodeChunkWithRapidgzip` (L413), `finishDecodeChunkWithInexactOffset` (L280-410) |
| Chunk data model | `ChunkData.hpp` | `applyWindow` (L247), `matchesEncodedOffset` (L396-403), `getLastWindow` (subchunk loop L335-366) |
| Window storage | `WindowMap.hpp` | mutex + `BTreeMap`, `emplace_hint(end(), ...)` (~L47-77) |
| ISA-L wrapper | `gzip/isal.hpp` | `IsalInflateWrapper::readStream` (L253-385), `setWindow`/`setStoppingPoints` (L52-79) |
| Block finder | `blockfinder/DynamicHuffman.hpp` | `seekToNonFinalDynamicDeflateBlock` (L167-298), bfinal=1 filter (L47-49), EOB-symbol check (L243-245) |
| Block-finder dispatch | `blockfinder/GzipBlockFinder.hpp` | `get(blockIndex)` (~L137) |

### Pipeline shape (per `BlockFetcher::get`)

```
caller (consumer) ──► getBlock(blockOffset)
                       ├─ check LRU cache → hit? return
                       ├─ check prefetch cache → hit? promote → return
                       ├─ check in-flight future → wait
                       └─ submit on-demand work → wait
                              │
                              └─ during wait: prefetch loop (1ms poll):
                                   prefetchNewBlocks() — fills T-1
                                   prefetch slots predicted by
                                   FetchingStrategy::prefetch(N+1..N+T-1)

post-processing (marker-apply): submitted to pool at priority -1 (higher than
prefetch) so multiple successors' applyWindow runs concurrently with new
decodes. Window for next chunk emplaced BEFORE post-processing returns so
prefetches can launch immediately (GzipChunkFetcher.hpp L572).
```

### Per-chunk decode (`finishDecodeChunkWithInexactOffset`)

```
1. IsalInflateWrapper wrapper(bitReader);
2. wrapper.setFileType(GZIP)
3. wrapper.setWindow(initialWindow)            // isal_inflate_set_dict
4. wrapper.setStoppingPoints(END_OF_BLOCK | END_OF_BLOCK_HEADER
                              | END_OF_STREAM_HEADER)
5. while !stoppingPointReached:
     buffer = vec(128 KiB)
     while nBytesRead < buffer.size() && !footer:
       (n, footer) = wrapper.readStream(buffer.data + nBytesRead,
                                        buffer.size - nBytesRead)
       // wrapper.readStream resets m_stream.stopped_at=NONE at entry
       // (isal.hpp:261) then loops isal_inflate until stop fires OR
       // buffer fills OR finished. Single-call read_stream cannot
       // deliver stops because OUT_OVERFLOW always wins on a small
       // avail_out.
       nBytesRead += n
       switch wrapper.stoppedAt():
         END_OF_BLOCK:        if !isFinal: appendBlockBoundary
         END_OF_BLOCK_HEADER: if nextBlockOffset >= untilOffset
                                 && !isFinal
                                 && btype != FIXED_HUFFMAN:
                                stoppingPointReached = true
         END_OF_STREAM_HEADER: isBlockStart = true
         NONE:                if n == 0 && !footer:
                                stoppingPointReached = true
     result.append(move(buffer))
6. finalizeChunk(...)
```

### Block finder semantics

Pure header validation, **zero trial decoding** (DynamicHuffman.hpp L257-263):
- 15-bit LUT skips invalid positions in 1-15 bit jumps.
- Checks HCLEN, precode leaf count (Kraft), lit/dist Huffman validity.
- Filters bfinal=1 (L47-49) — not useful as chunk split points.
- Requires `literalCL[END_OF_BLOCK_SYMBOL] != 0` (L243-245) — every block
  must terminate.
- Reported throughput: ~8.7 MB/s of compressed input. 512-KiB-per-partition
  scan budget caps cost at ~60ms/chunk.

### Consumer reconciliation

`matchesEncodedOffset(off)` (ChunkData.hpp L396-403):
```cpp
encoded_offset_bits <= off && off <= max_encoded_offset_bits
```
A chunk is a **range** of acceptable start positions, not a point. The
consumer uses this to accept speculative starts within tolerance; on miss
it submits re-decode to the **thread pool** (`BaseType::get`), not inline.

---

## 2. gzippy current state (HEAD `0df6c1a`)

| File | Lines | Role |
|------|-------|------|
| `single_member.rs` | 4144 | Entry point + dead v0.6 code (slated for deletion). `decompress_parallel_via_fetcher` is the live entry (L417-501). |
| `chunk_fetcher.rs` | 553 | Pass-A + Pass-B dispatcher; consumer with synchronous re-dispatch on mismatch. **Both passes run to completion before consumer reads chunk 0**. |
| `gzip_chunk.rs` | 240 | `finish_decode_chunk_with_inexact_offset`: marker-bootstrap → patched-ISA-L via raw bindings (bypasses `IsalInflateWrapper`). |
| `inflate_wrapper.rs` | 430 | Exists, used by `gzip_chunk.rs` tests. Production path uses raw bindings. |
| `chunk_data.rs` | 388 | `ChunkData` + `Subchunk` + `ChunkConfiguration`. No `max_encoded_offset_bits` field yet. |
| `apply_window.rs` | 162 | `replace_markers` + CRC prepend. Allocates fresh u8 Vec from u16s (extra alloc). |
| `window_map.rs` | 107 | **Single-threaded** `BTreeMap`, no mutex. Consumer-only access. |
| `block_finder.rs` | 1153 | 13-bit LUT + Huffman validation; **accepts bfinal=0 AND bfinal=1**; EOB-symbol check IS present (L771-774). |
| `fast_marker_inflate.rs` | ~2100 | Pure-Rust marker decoder used by chunk_fetcher's `decode_chunk_bootstrap`. |
| `trace.rs` | 124 | JSON-lines tracing gated on `GZIPPY_LOG_FILE`. |

---

## 3. Gap matrix — current status

Classifications: **CORRECTNESS** wrong-output if not closed; **PERFORMANCE** correct but slow; **STRUCTURAL** affects shape only.

| ID | Gap | Class | rapidgzip ref | gzippy ref | Status | Impact |
|----|-----|-------|---------------|------------|--------|--------|
| G1 | No async prefetch / pipelined dispatcher. Both passes run to completion before consumer reads chunk 0. | PERF / CRIT | BlockFetcher.hpp L314 (1ms poll + prefetchNewBlocks) | chunk_fetcher.rs `dispatch_all_blocking` L170 | **OPEN** | DOMINANT — dispatcher and consumer can't overlap; consumer-side re-dispatch is single-threaded. |
| G2 | Synchronous re-dispatch on consumer thread when speculative boundary mismatches. | PERF / CRIT | GzipChunkFetcher.hpp L591-687 `getBlock` re-submits to pool | chunk_fetcher.rs `redispatch_chunk` L292 (inline) | **OPEN** | ~33/39 chunks re-dispatched serially per recent trace. |
| G3 | Block-finder admits more false positives than rapidgzip's. | PERF | DynamicHuffman.hpp L47-49 bfinal=1 filter + L243-245 EOB check | block_finder.rs `is_valid_candidate_13` L57-70 (no bfinal filter — see decision log #1) | **OPEN** | Attempted bfinal filter (`fbfff82`) caused OOM, reverted (`0df6c1a`). Needs investigation. |
| G4 | `validate_boundary` did a 32 KiB marker decode per candidate. | PERF | DynamicHuffman.hpp L257-263 (no trial decode) | chunk_fetcher.rs `find_first_boundary_candidate` L132-146 | **CLOSED** (`58d56ee`) | Was 50s aggregated CPU; now 2.9s. Single biggest win so far. |
| G5 | Per-chunk 80 MiB output buffer allocation. | PERF / HIGH | GzipChunk.hpp L213 persistent 128 KiB buffer | gzip_chunk.rs L218-223 `Vec::with_capacity(max_decoded_chunk_size)` | **OPEN** | 38 chunks × 80 MiB; allocator pressure across 16 threads. |
| G6 | Every chunk uses pure-Rust marker bootstrap before ISA-L. | PERF / HIGH | GzipChunk.hpp L432-444 (skip bootstrap when initialWindow known) | gzip_chunk.rs L142 `decode_chunk_bootstrap` (always) | **OPEN** | Worker per-chunk fixed cost ~30ms even when window IS known. |
| G7 | `WindowMap` is consumer-only, not shared across workers. | STRUCTURAL → PERF | WindowMap.hpp L19 `mutable std::mutex` | window_map.rs L19-22 no mutex | **OPEN** | Precludes G6's fix — workers can't read predecessor's window. |
| G8 | `IsalInflateWrapper` bypassed by production path. | STRUCTURAL | isal.hpp L253-385 full wrapper | gzip_chunk.rs L176-301 raw bindings | **OPEN** | Less granular stop control than rapidgzip. |
| G9 | CRC built by re-narrowing u16→u8 instead of prepend on u8 view. | PERF / LOW | ChunkData.hpp L325 `CRC32Calculator::prepend` | apply_window.rs L52-55 | **OPEN** | Modest. |
| G10 | "Next partition seed" used as `until_bits` in redispatch is a fixed grid, not a confirmed boundary. | STRUCTURAL | uses confirmed boundaries | chunk_fetcher.rs L324-330 | **OPEN** | Contributes to G2 mismatch rate. |
| G11 | `std::thread::scope` spawns N threads per call (twice). | PERF / LOW | persistent ThreadPool (BlockFetcher.hpp L186) | chunk_fetcher.rs L190+ | **OPEN** | < 5ms per call. |
| G12 | Chunks have subchunks recorded but consumer treats whole chunk as one. | PERF | rapidgzip splits + windows per subchunk | chunk_data.rs has Subchunk but consumer doesn't use it | **OPEN** | Larger blocking time during `apply_window`. |
| G13 | Chunk i>0 always has `data_with_markers` prefix that must be resolved. | OK (inefficient) | rapidgzip clean-only when window known | follows from G6+G7 | **OPEN** | Closes when G6+G7 close. |

### Decision log

1. **Bfinal=1 filter caused OOM** (commit `fbfff82` reverted by `0df6c1a`).
   Rejecting bfinal=1 candidates made `find_first_candidate` scan further
   on average; consequence not yet traced. **Next step**: re-instrument
   to see whether (a) the scan reaches the 8 MiB radius and returns None
   more often (raising re-dispatch count), or (b) the chunk-decoder
   accepts a different candidate that decodes a much larger range.
   Verify with `GZIPPY_LOG_FILE=...` + `scripts/parallel_sm_log_summary.py`.

---

## 4. Measurement infrastructure

### Trace events emitted by `trace.rs`

| Event | Fields | When |
|-------|--------|------|
| `boundary_done` | partition_idx, seed_bit, found_bit (null if none), duration_us | Per-partition boundary search completes |
| `speculative_start` | partition_idx, start_bit, until_bit | Per-worker decode begins |
| `speculative_ok` | partition_idx, start_bit, end_bit, decoded, markers, clean, preemptive, duration_us | Worker decode succeeds |
| `speculative_err` | partition_idx, start_bit, until_bit, err, duration_us | Worker decode fails |
| `speculative_skip` | partition_idx, reason | Worker skipped (no boundary etc.) |
| `speculative_accept` | partition_idx, start_bit, end_bit | Consumer accepts speculative result |
| `speculative_mismatch` | partition_idx, speculative_start, expected_start | Consumer rejects: position mismatch |
| `speculative_missing` | partition_idx, expected_start | Consumer found no speculative result |
| `redispatch_start` | partition_idx, expected_start, until_bit | Consumer-side re-decode begins |
| `redispatch_ok` | partition_idx, expected_start, end_bit, decoded, duration_us | Re-decode succeeds |
| `redispatch_err` | partition_idx, expected_start, err, duration_us | Re-decode fails |
| `apply_window_done` | partition_idx, marker_bytes, duration_us | `apply_window` completes |
| `consume_done` | partition_idx, end_bit, decoded | Consumer finished a chunk |

### How to capture + summarize a trace

```bash
# On neurotic (x86_64 + ISA-L):
ssh -J neurotic root@10.30.0.199 \
  'cd gzippy && rm -f /tmp/sm.log && \
   GZIPPY_LOG_FILE=/tmp/sm.log \
   ./target/release/gzippy -d -p16 < benchmark_data/silesia-large.gz > /dev/null'

# Pull + summarize locally:
scp -J neurotic root@10.30.0.199:/tmp/sm.log /tmp/sm-current.log
python3 scripts/parallel_sm_log_summary.py /tmp/sm-current.log

# Drill into one partition:
python3 scripts/parallel_sm_log_summary.py /tmp/sm-current.log --partition 5

# Filter:
scripts/parallel_sm_log_grep.sh /tmp/sm-current.log ev=speculative_mismatch
scripts/parallel_sm_log_grep.sh /tmp/sm-current.log partition_idx=12
```

### Baseline benchmarks (Silesia gzip -9, T=16, neurotic homelab)

| HEAD | gzippy MB/s | vs rapidgzip | Notes |
|------|-------------|--------------|-------|
| pre-port (v0.6) | ~270 | ~0.18× | Old marker pipeline (different perf characteristics) |
| `6ac952c` | 120 | 0.08× | Initial cutover; everything sequential through re-dispatch |
| `58d56ee` | 205 | 0.19× | Dropped validate_boundary; boundary search 50s → 2.9s CPU |
| `0df6c1a` | (TBD) | (TBD) | Reverted bfinal=1 filter (caused OOM) |

Rapidgzip on same fixture: ~1500 MB/s.
Goal: ≥0.99× rapidgzip.

---

## 5. Judgment-call checklist

Before committing a perf-targeted change to `parallel/*`:

1. **Which gap does this close?** Map to a G# above. If none, you're either fixing something this doc doesn't know about (add it) or doing speculative work (stop).
2. **Did you capture a before-trace?** `make bench-sm` + the trace capture in §4 BEFORE the change.
3. **Did you run the after-trace?** Same.
4. **Compare summaries.** If event counts moved in unexpected directions (e.g. more re-dispatches), don't ship.
5. **Did `cmp` say CORRECT?** Always.
6. **Did `make bench-sm` say PASSED?** If FAILED, what's the new gzippy MB/s number? (FAILED can mean correctness or routing-trace heuristic regression — the heuristic is stale; trust the number if cmp matches.)
7. **Is the change consistent with §1?** If you're doing something rapidgzip doesn't, you need a reason that names a measured gzippy-specific constraint.

---

## 6. Open structural choices to decide before further perf work

- **Shared WindowMap design** (G7): `Arc<(Mutex<BTreeMap<...>>, Condvar)>` vs lock-free skip-list. Mutex is simpler and matches rapidgzip (its mutex is uncontended in practice). Default: mutex.
- **Async re-dispatch** (G2): persistent rayon pool vs `std::thread::scope`. Persistent pool simpler given workers need to live across `decompress_parallel` calls. Default: rayon.
- **Fast-path API** (G6): new `decode_chunk_with_window(input, start, until, window)` in `gzip_chunk.rs` using the existing `IsalInflateWrapper` (closes G8 simultaneously). Default: yes.

---

## 7. References to keep alongside this doc

- `docs/rapidgzip-port-design.md` (existing) — original 10-step design from
  earlier in the port. Some sections outdated; treat as historical record.
- `vendor/rapidgzip/` — the C++ ground truth. Always cite line numbers when
  this doc references it.
- `src/tests/routing.rs::test_single_member_routing_multithread` — the
  byte-correctness regression test. Must always pass.
- `make bench-sm` — measured perf. Authoritative for the ≥0.99× rapidgzip
  goal. Run before claiming a perf change worked.
