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

## 3. Gap matrix — current status (HEAD `ab98aca`)

Rewrite landed: shared WindowMap, persistent worker pool with prefetch,
fast/slow worker paths, async authoritative re-dispatch, bfinal=1
filter, chunked CRC prepend, v0.6 deletion. Status table:

Classifications: **CORRECTNESS** wrong-output if not closed; **PERFORMANCE** correct but slow; **STRUCTURAL** affects shape only.

| ID | Gap | Class | rapidgzip ref | gzippy ref | Status | Impact |
|----|-----|-------|---------------|------------|--------|--------|
| G1 | No async prefetch / pipelined dispatcher. | PERF / CRIT | BlockFetcher.hpp L314 | chunk_fetcher.rs `consumer_loop` prefetch_window + worker pool | **CLOSED** (`ab98aca`) | Pending bench verification. |
| G2 | Synchronous re-dispatch on consumer thread. | PERF / CRIT | GzipChunkFetcher.hpp L591-687 | chunk_fetcher.rs `authoritative_dispatch` submits to pool | **CLOSED** (`ab98aca`) | Pending bench verification. |
| G3 | Block-finder accepts bfinal=1 candidates. | PERF | DynamicHuffman.hpp L47-49 | block_finder.rs `is_valid_candidate_13` bfinal filter | **CLOSED** (`a889e78`) | Safe now: rejections route through async pool, not the synchronous chain that OOMed in `fbfff82`. |
| G4 | `validate_boundary` did a 32 KiB marker decode per candidate. | PERF | DynamicHuffman.hpp L257-263 | chunk_fetcher.rs `find_first_boundary_candidate` (removed; now inline in `compute_speculative_boundaries`) | **CLOSED** (`58d56ee`) | Was 50s aggregated CPU; now 2.9s. |
| G5 | Per-chunk 80 MiB output buffer allocation. | PERF / HIGH | GzipChunk.hpp L213 persistent 128 KiB | gzip_chunk.rs slow-path retains 80 MiB cap; fast path uses 128 KiB per `read_stream` iter | **PARTIAL** | Fast-path closes G5. Slow path (chunk 0 + cold misses) still allocates large; revisit if measurement shows it matters. |
| G6 | Every chunk uses pure-Rust marker bootstrap. | PERF / HIGH | GzipChunk.hpp L432-444 | gzip_chunk.rs `decode_chunk_with_window` (fast path) | **CLOSED** (`ac06755`) | Workers take fast path when WindowMap has predecessor window. |
| G7 | `WindowMap` consumer-only. | STRUCTURAL → PERF | WindowMap.hpp L19 mutex | window_map.rs `Arc<(Mutex<BTreeMap>, Condvar)>` | **CLOSED** (`ac06755`) | Shared across worker pool + consumer. |
| G8 | `IsalInflateWrapper` bypassed. | STRUCTURAL | isal.hpp L253-385 | gzip_chunk.rs `decode_chunk_with_window` uses wrapper | **CLOSED** (`ac06755`) | Fast path uses wrapper. Slow path still uses raw bindings (kept; works). |
| G9 | CRC built by re-narrowing u16→u8. | PERF / LOW | ChunkData.hpp L325 | apply_window.rs 4 KiB stack chunks | **CLOSED** (`a889e78`) | Minor improvement. |
| G10 | `until_bits` in redispatch is a fixed grid. | STRUCTURAL | rapidgzip uses confirmed boundaries | chunk_fetcher.rs `authoritative_dispatch` uses next partition seed | **CLOSED** (`ab98aca`) | Range-match acceptance via `matches_encoded_offset` makes exact alignment optional. |
| G11 | `std::thread::scope` spawns N threads per call. | PERF / LOW | persistent ThreadPool | chunk_fetcher.rs scope persists across all decode work in one `drive()` call | **CLOSED** (`ab98aca`) | Per-call instead of per-decode. |
| G12 | Subchunk-level emission not used. | PERF | rapidgzip splits per subchunk | unchanged; chunks still emitted as whole units | **OPEN** | Would let consumer yield earlier bytes. Defer until measurement shows it matters. |
| G13 | Chunk i>0 always has data_with_markers prefix. | follows G6+G7 | rapidgzip clean-only when window known | closes via fast-path adoption | **CLOSED** (`ac06755`) | Fast path emits clean bytes only. |

### Decision log

1. **Bfinal=1 filter caused OOM** (commit `fbfff82` reverted by `0df6c1a`).
   Root cause now understood: rejecting bfinal=1 raised the rate of
   "no candidate found" → consumer's synchronous redispatch chain
   processed every chunk inline, each allocating a fresh 80 MiB output
   buffer. The new chunk_fetcher routes rejections through the async
   pool instead, so the same filter is now safe. Re-landed in `a889e78`.

2. **One-piece rewrite over staged commits** (`ab98aca`). After the
   first attempt to land changes phase-by-phase (G4 closed alone moved
   bench from 0.08× → 0.19×, then G3-alone OOMed), the next attempt
   was a single coherent rewrite that closes G1, G2, G6, G7, G8, G10,
   G11 together — none of them are useful in isolation. The seams
   between partial fixes were the problem.

3. **Slow-path bootstrap hang** (`d78bf9d` trace, fixed in `ff47485`).
   First post-rewrite run on Silesia OOMed at byte 161755235 after
   producing 11 chunks. RSS-instrumented trace showed worker-26
   started slow_path at bit 872417560 and never emitted decode_err
   for 24s. Root cause measured (not guessed): `fast_marker_inflate::
   decode_loop` was passing `max_output=0` ("no limit") to
   `decode_fixed` / `decode_dynamic`, so a phantom-boundary block whose
   Huffman codes happened to validate would emit unbounded garbage
   symbols. Fix: cap per-block output at 256 MiB. Bench: 205 → 340 MB/s.

   Process lesson: the trace instrumentation (rss_kib + per-event
   timestamps + per-partition lookups) made the cause findable in two
   trace captures. Without it, this would have been hours of guessing.

### Decision log entry #4 — phantom boundaries are fundamental

After two more rounds of measurement (HEAD `4263f00` non-fixed-only
stop + pre-header reporting; HEAD `685fee8` per-block subchunks +
range-match trim; HEAD `7bb1dd6` per-subchunk window inserts), the
authoritative-dispatch rate stayed at 38/39. Measured diagnosis:

- Speculative chunk N's `encoded_offset_bits` (= BlockFinder candidate
  at-or-past partition_seed[N]) and the predecessor's `actual_end`
  (= where chunk N-1's worker stopped) DIFFER by median 28 Kbit
  (3.5 KB), max 357 Kbit (44 KB). 100% within 64 KiB.
- Both are valid non-fixed dynamic-Huffman block starts per the
  precode + lit/dist Huffman validation in `block_finder.rs::
  validate_huffman_codes`.
- They differ because they're block starts ON DIFFERENT DECODE
  PATHS through the same compressed stream. BlockFinder finds
  the first position past the seed that LOOKS like a valid block
  header. Predecessor's natural decode goes through a specific
  block sequence determined by where its decode started.
- Per-block subchunks + range-match trim don't help because
  speculative-chunk subchunks are at speculative-decode positions,
  not at natural-decode positions.
- Per-block window inserts don't help for the same reason.

Rapidgzip's identical precode validation (verified at
`vendor/rapidgzip/.../blockfinder/precodecheck/CountAllocatedLeaves.hpp`)
should produce the same candidates. Their reported "1 false positive"
on this fixture has to mean something different — probably "1
candidate that didn't decode", not "1 candidate that mismatched
predecessor's natural decode." Their 97% cache hit rate from
`--verbose` likely comes from sequential consumer-driven dispatch
hitting the prefetch cache, not from speculative starts aligning
with natural-decode endpoints.

### Next attempt: depth-2 authoritative pipelining

Acceptance criteria (before/after `make bench-sm` on neurotic):
- Byte-correct on Silesia gzip -9 (no regression).
- gzippy MB/s ≥ 600 (~2× current 340).
- Trace shows ≥1 chunk where chunk N+1's authoritative was already
  in flight when chunk N consumed (new event: `authoritative_prefetch`).

Change scope (single commit):
- In `consumer_loop`, after a successful consume of chunk N, submit
  chunk N+1's authoritative with expected_start = chunk N's
  actual_end (just computed). Store its receiver in
  `pending_auth[N+1]`.
- When the loop advances to consume chunk N+1, check pending_auth
  first. If present, wait on that receiver. Else fall through to the
  existing speculative-then-authoritative logic.
- This is a "look-ahead by 1" prefetch: while the consumer applies
  window / writes bytes / inserts windowmap for chunk N (~5–10ms),
  chunk N+1's worker is already decoding (~20ms fast path). Pipeline
  depth = 2.

Expected mechanism:
- consume[N] + decode[N+1] overlap.
- Without pipelining: each chunk takes (consume_time + decode_time).
- With pipelining: each chunk takes max(consume_time, decode_time).
- Decode dominates (~20ms vs consume ~5ms) → ~25% wall saving.

What this does NOT close:
- Speculative-mismatch rate stays at ~37/39 (the architectural ceiling).
- Authoritative chain is still serial (one in-flight authoritative,
  not many). Depth > 2 would require predictions, which we measured
  at ~21% accuracy.

If this doesn't hit ≥600 MB/s, the next attempt is:
- Run a single-threaded marker-decoder scan ONCE upfront to enumerate
  every natural block boundary. Use as authoritative starts for full
  parallel decode. Cost: ~3s CPU scan, possibly overlapped with
  decode. Bench target conditional on scan/decode overlap.

Discipline: capture before-trace, implement, capture after-trace,
compare authoritative_submit + new authoritative_prefetch counts,
verify bench-sm number. Update this doc with results.

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
| `ff47485` | 340 | 0.17× | One-piece rewrite: shared WindowMap + worker pool + fast/slow paths + bfinal filter + bootstrap per-block cap. Correct ✓; rapidgzip variance is high (1080-2000 MB/s across runs). |
| `4263f00` | 320 | 0.16× | Tried non-fixed-only stop with pre-header end_bit reporting. Mismatch rate unchanged (37/39). |
| `685fee8` | 340 | 0.17× | Added per-block subchunks + range-match consumer trimming. Trim path never fires (speculative subchunks aren't on natural decode path). |
| `7bb1dd6` (reverted) | 290 | 0.14× | Per-block window inserts. Net loss — overhead without hits. |
| `44047f2` | (≈340) | (≈0.17×) | Revert to baseline post-rewrite. |

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
