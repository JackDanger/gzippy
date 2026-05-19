# rapidgzip → gzippy parallel single-member: structural reference

## AUDIT (2026-05-17): cognitive bias review + real gap analysis

> This section is an external second-opinion review by an Opus advisor
> after ~60 commits on `feat/cross-chunk-retry` failed to close the
> Silesia gap (gzippy 489 MB/s vs rapidgzip 1077–1374 MB/s, i.e.
> 0.36–0.45×). The user's words: *"you've run afoul of the same bias
> that has gotten you wedged a bunch of times in the past, where you
> hyper-focus on one thing and then you actually just back yourself
> into a shitty new architecture rather than keeping an eye on the
> actual goal architecture… one correct, complete implementation, which
> is a direct port, not an invention, of an existing bit of code that
> is right there on disk."*  The user is correct. This audit names the
> biases, the real (still-present) architectural deviations, and a
> straight-line plan that stops inventing and starts copying.

### A1. Behavioral / cognitive-bias findings

**Bias #1 — "Port files" as a substitute for porting the architecture.**
Between `87cbc54` and `53f1a0f` we landed ~30 commits whose subject
lines start with `port(parallel):` — `ThreadPool`, `BlockFetcher`,
`BlockMap`, `Cache`, `Prefetcher`, `Statistics`, `CompressedVector`,
`deflate::Block`, every Huffman variant. Per §H of this doc the
**total unused-port LOC is ~2900**. The production path
(`chunk_fetcher::drive`, `consumer_loop`, `fast_marker_inflate.rs`,
`gzip_chunk::finish_decode_chunk_with_inexact_offset`) was barely
touched while these files accumulated alongside it. The mental model
that produced each commit was "this file looks like vendor's, so the
port is progressing." The execution model is that none of these files
are on the hot path. We have a museum of rapidgzip headers and a
gzippy-shaped engine that the museum decorates.

Evidence: `block_fetcher.rs`, `block_map.rs`, `cache.rs`,
`prefetcher.rs`, `statistics.rs`, `compressed_vector.rs`,
`deflate_block.rs` — all marked **Unused in prod** in §A. Commits
`306c1e7` and `422da1f` claim to "wire BlockFetcher pipeline into
chunk_fetcher::drive" but the diff shows `block_fetcher.get` invoked
as a synchronous wrapper around our pre-existing `mpsc` worker pool
plus our pre-existing `spec` HashMap (chunk_fetcher.rs:767, 944) —
the BlockFetcher is decoration, not dispatch.

**Bias #2 — Defensive patches over root-cause fixes.** Two consecutive
commits (`f902c06`, `9f76440`) were merged within hours of each other
and BOTH later reverted in this branch's history because they were
*defensive skips* in `BlockMap::push` and
`append_subchunks_to_block_map` to "tolerate phantom zero-size
subchunks." The root cause was upstream: the consumer was advancing
`expected_start` before EOF termination (fixed in `d089aa9` —
"terminate consumer on EOF (vendor parity)") and `next_block_index`
was tracking partition count instead of subchunk count (fixed in
`435184b` — "track block index by subchunk count (vendor parity)").
The commit-message body for `9f76440` literally describes its own
patch as moving "the failure from panic to gzip ISIZE mismatch" —
i.e. the author noticed the patch was wrong while writing the message
and shipped it anyway. The vendor parity fix in `d089aa9` made both
defensive patches unnecessary.

Evidence: `git log --oneline -80` shows `d089aa9 fix(parallel-sm):
terminate consumer on EOF (vendor parity)` followed shortly by
`435184b fix(parallel-sm): track block index by subchunk count
(vendor parity)`. If those two fixes had been the FIRST response
("the vendor does X, what does X look like?"), `f902c06` and
`9f76440` would never have been written.

**Bias #3 — Confirmation bias on "advisor signed off" / "vendor
parity" branding.** Subject lines like `port(parallel-sm): align
append_block_boundary dedup with rapidgzip` (`4612252`) and `feat:
Step 7b — port deflate::Block read + readInternalCompressed`
(`04e518d`) read as victories but the deliverables were file-local:
the function bodies match vendor C++ but the CALL SITES still go to
gzippy-original code. `deflate_block.rs::read` is faithful to
`deflate::Block::read` (§A row "Faithful; unused in prod") and the
production slow path runs `fast_marker_inflate::decode_chunk_bootstrap`
(2356 lines of gzippy-original code). The "advisor signed off"
pattern from previous sessions reinforces this — a sub-agent reports
"faithful port landed, tests green" and the chain continues, but the
production routing never moves to the new port.

Evidence: §A rows for `deflate::Block<>`, `Cache`, `Prefetcher`,
`BlockFetcher`, `Statistics`, `CompressedVector` all marked
**Unused** alongside Faithful or Deviated. The doc itself records
the divergence; we kept landing port commits anyway.

**Bias #4 — Local hill-climbing without a global view.** Commits like
`ec9bee7` (key spec ring by blockOffset, refill at real boundaries),
`194ddf8` (split apply_window onto worker pool), `77b2f1c` (bulk-copy
subchunk windows), `8970f71` (drop zero-init from inner-loop
scratch), `d0ec938` (gate trace::rss_kib syscalls behind is_enabled)
are each individually defensible perf knobs that nibble the gap. None
of them changes the fact that the consumer is a 700-line
`consumer_loop` with a hand-rolled HashMap-keyed `spec` ring while
vendor is a 50-line `processNextChunk` that just calls
`BlockFetcher::get`. We have spent weeks tuning the wrong engine.

**Bias #5 — Implementer chains producing N commits with no coherent
diff against vendor.** 373 commits in 14 days. The history shows
clusters: 25-30 "port(parallel): land X primitive" commits that built
the museum, followed by 15-20 "perf(parallel-sm): ..." commits trying
to make our pre-existing architecture catch up, followed by ~10
revert commits ("Revert: perf(...): cap bootstrap output at 1 MB
without handoff"; "Revert: feat(parallel-sm): wire IsalLitLenCode
into bootstrap"). Each implementer ran in isolation and shipped a
locally-coherent change. No one held the question "does our consumer
match `processNextChunk`?" the whole time. The CUTOVER POLICY in this
very doc ("PORT, DON'T INNOVATE … crossbeam-channel, custom thread
pools, unique algorithms: all out of scope") was authored on
2026-05-?? and we have since shipped `7aba6db perf(parallel-sm):
lock-free job dispatch via crossbeam-channel` (immediately reverted
in `5bd16b9`) and the current `consumer_loop`-with-`HashMap<usize,
SpecSlot>` ad-hoc ring (which IS the kind of "custom" the policy
forbade). The policy is correct; we are not following it.

### A2. Real architectural gap (what is still actually different)

Walk both production sequences side by side, classify each step.

#### Vendor production hot path

1. **Top-level driver**:
   `ParallelGzipReader<ChunkFetcher>::read(...)` →
   `processNextChunk()` (vendor `GzipChunkFetcher.hpp:311-362`).
   ~50 lines. Each iteration:
   a. `m_blockFinder->get(m_nextUnprocessedBlockIndex)` — get next
      block offset.
   b. `getBlock(*nextBlockOffset, m_nextUnprocessedBlockIndex)` →
      `BaseType::get(...)` (= `BlockFetcher::get`, `core/BlockFetcher.hpp:245-329`).
      This is THE work primitive: it consults caches, dispatches an
      on-demand task via `ThreadPool` if needed, runs
      `prefetchNewBlocks` during the wait, returns a
      `shared_ptr<ChunkData>`.
   c. `m_windowMap->get(*nextBlockOffset)->decompress()` — get
      predecessor's window from a `CompressedVector`-backed map.
   d. `postProcessChunk(...)` → `waitForReplacedMarkers(...)` → if
      not enqueued, `queueChunkForPostProcessing(...)` →
      `submitTaskWithHighPriority(applyWindow)` on the SAME
      `ThreadPool`. Marker resolution runs on workers, not consumer.
   e. `chunkData->setEncodedOffset(*nextBlockOffset)` — re-anchor.
   f. `appendSubchunksToIndexes(...)` — push into `BlockMap`,
      `BlockFinder::insert`, emplace per-subchunk windows.
   g. Return chunk. Caller does the write.
2. **Single decode primitive**: `decodeBlock(blockOffset,
   nextBlockOffset) const override` (vendor `GzipChunkFetcher.hpp:692-729`).
   Calls `decodeChunk(...)` (vendor `GzipChunk.hpp:660-852`). Inside,
   `decodeChunkWithRapidgzip` instantiates a `deflate::Block<true>`
   (i.e. enableAnalysis=true), seeds it with `setInitialWindow` from
   the previous chunk's last 32 KiB if available (else starts in
   marker mode), reads blocks one at a time. When cumulative
   `cleanDataCount` ≥ `MAX_WINDOW_SIZE`, hands off to
   `IsalInflateWrapper` via `setWindow`.
3. **Caches + prefetch**: `Cache<K,V,LeastRecentlyUsed>` for main +
   prefetch; `FetchNextAdaptive` strategy.
4. **Threading**: `BS::thread_pool` + `std::future<BlockData>`.
   Tasks: decode tasks and `applyWindow` tasks share the same pool.

#### Gzippy production hot path (what runs at `feat/cross-chunk-retry` HEAD)

1. **Top-level driver**: `decompress_single_member` →
   `parallel::single_member::decompress_parallel` →
   `chunk_fetcher::drive` (1661-line file). `drive` spawns a
   `std::thread::scope` worker pool sharing a
   `Mutex<mpsc::Receiver<Job>>`. The consumer is
   `consumer_loop`, ~700 lines (chunk_fetcher.rs:670-1373).
2. **Consumer**: hand-rolled HashMap-keyed `spec: HashMap<usize,
   SpecSlot>` ring (chunk_fetcher.rs:767) with an `expected_start`
   walk, an `auth_result = block_fetcher.get(expected_start, ||
   {...spec drain... or submit_job...})` synchronous wrapper
   (chunk_fetcher.rs:944-1066). The closure inside `block_fetcher.get`
   is gzippy-original logic; the call is decoration so a deletion-
   trap counter (`BLOCK_FETCHER_GET_CALLS_OBSERVED`) can fire.
3. **Decode primitive**: `worker_loop` (chunk_fetcher.rs:487-667)
   branches on `window_map.get_or_wait` into FAST PATH
   (`decode_chunk_with_window`, ISA-L with known dict) or SLOW PATH
   (`decode_or_iterate` → `finish_decode_chunk_with_inexact_offset`
   → `fast_marker_inflate::decode_chunk_bootstrap`, 2356 lines of
   gzippy-original code, NOT `deflate::Block`).
4. **Caches**: `BlockFetcher` IS constructed (chunk_fetcher.rs:274)
   but its main cache and prefetch cache are only inserted into for
   bookkeeping (chunk_fetcher.rs:1492). Reads from the cache happen
   only inside `block_fetcher.get`'s wrapper closure, which we
   control. The vendor's `getFromCaches` → `takeFromPrefetchQueue` →
   `m_cache.get(blockOffset)` chain is not the dispatch primitive
   here; the spec HashMap is.
5. **Post-process**: `queueChunkForPostProcessing` → `submit_post_process`
   on the SAME mpsc pool (chunk_fetcher.rs:814-839). This part is
   actually parallel and matches vendor in structure since `194ddf8`.
6. **Threading**: `std::thread::scope` + `mpsc::channel` + per-job
   `mpsc::channel` reply.

#### Architectural deviation matrix

| Step | Vendor | Gzippy | Class |
|------|--------|--------|-------|
| Top-level read | `ParallelGzipReader::read` → `processNextChunk` (~50 LOC) | `decompress_parallel` → `drive` → `consumer_loop` (~700 LOC) | **Architectural deviation** |
| Block lookup | `m_blockFinder->get(idx)` | `seed_for(block_index)` wrapping `block_finder.get(idx)` | Faithful |
| Dispatch | `BlockFetcher::get(blockOffset, blockIndex, partitionOffsetFn)` (`core/BlockFetcher.hpp:245-329`) | `block_fetcher.get(expected_start, closure)` where the closure is `spec.remove(&expected_start)`-or-`submit_job(...).recv()` (chunk_fetcher.rs:944-1066) | **Architectural deviation** — wrapper around our ring, not the BlockFetcher dispatch |
| Threading | `BS::thread_pool` + `std::future<BlockData>` | `std::thread::scope` + `Mutex<mpsc::Receiver<Job>>` + per-job `mpsc::channel` reply | **Architectural deviation** (deferred in §B5; never closed) |
| Prefetch queue | `m_prefetching: std::map<size_t, std::future<BlockData>>` (BlockFetcher.hpp:131) advanced in `prefetchNewBlocks` during the future wait | `spec: HashMap<usize, SpecSlot>` (chunk_fetcher.rs:767) advanced once per consumer iter at `seed_for(next_block_index)` (chunk_fetcher.rs:1285-1329) | **Architectural deviation** — we explicitly chose ONE immediate-next-seed per iter; vendor fills `parallelization` prefetches via `FetchingStrategy::prefetch` |
| Cache | `Cache<K,V,LRU>` + LRU eviction (`core/Cache.hpp`) | `BlockFetcher`'s internal `Cache` + `PrefetchCache` (block_fetcher.rs:31-49); inserts happen but reads do not drive dispatch | **Architectural deviation** — cache exists; routing doesn't query it |
| Decode core (slow path) | `deflate::Block<true>` (vendor `gzip/deflate.hpp:513-2005`) seeded by `setInitialWindow`, marker mode→clean handoff via `cleanDataCount` | `fast_marker_inflate::decode_chunk_bootstrap` (2356 LOC, gzippy-original) + `CleanTailTracker` heuristic | **Architectural deviation** — `deflate_block.rs` is the faithful port and is **NOT CALLED** |
| Marker encoding | `MapMarkers` (oldest-byte-indexed, `MarkerReplacement.hpp:15-46`) | `MARKER_BASE=32768`, newest-byte-indexed (`replace_markers.rs:24,49-58`) | **Architectural deviation** (B1; documented since the doc was first written) |
| Window storage | `Cache<size_t, SharedWindow>` where `SharedWindow = CompressedVector` | `BTreeMap<usize, Arc<[u8;32768]>>` + `Mutex` + `Condvar` (`window_map.rs:23-96`) | **Architectural deviation** (B12) + Extra (Condvar) |
| Post-process | `submitTaskWithHighPriority([chunkData, window]() { chunkData->applyWindow(...); })` on the shared pool | `submit_post_process(...)` on the same mpsc pool | **Faithful** (since `194ddf8`, May 2026) |
| Re-anchor on speculative miss | `chunkData->setEncodedOffset(*nextBlockOffset)` inside `processNextChunk` always | `chunk.set_encoded_offset(expected_start)` inside the spec-hit branch of `block_fetcher.get`'s closure (chunk_fetcher.rs:1004) | Cosmetic deviation |
| EOF | `if ( chunkData->encodedSizeInBits == 0 ) { m_blockMap->finalize(); m_blockFinder->finalize(); return {}; }` | `if chunk.encoded_size_bits == 0 { break; }` then `block_map.finalize()` / `block_finder_par.finalize()` after `drive` returns | Cosmetic deviation |
| Routing layer fallback | None — vendor throws | None at HEAD — the routing-layer fallback was already removed in `decompress_single_member_for` (mod.rs:281-294). | **Faithful** (rare bright spot) |
| Multi-stream gzip | `decodeChunkWithRapidgzip` loops over streams within a chunk via `readGzipFooter` / `readHeader` | `gzip_chunk::decode_chunk_with_window` was updated for the multi-stream loop (Step 13 ✅) but `decompress_parallel` reads the trailer inline assuming single-member (single_member.rs:95-99) | Mixed: decoder loop faithful; driver assumes single-member |
| Direct-try-at-guessed-offset | `tryToDecode` first attempts decode at `blockOffset` (GzipChunk.hpp:736-741) | Deleted (decode_or_iterate iterates BlockFinder candidates only) | Architectural deviation (B10), deliberate |
| `appendDeflateBlockBoundary` semantics | `(encoded, decoded)` dedup + `splitChunkSize` gate (ChunkData.hpp:455-467) | `(encoded)` dedup only, always push new subchunk (chunk_data.rs:264-285) — partially fixed in `4612252`; split-size gate still missing | Architectural deviation (B2) |

#### The single observation that explains the perf gap

Vendor's `processNextChunk` body is ~50 lines because EVERY non-trivial
operation is a method call into a primitive that vendor wrote ONCE.
The consumer is a coordination layer over a few well-tested
algorithms. Gzippy's `consumer_loop` is 700 lines because it
**reimplements those algorithms inline** as it goes:

- vendor's `BlockFetcher::get` does prefetch-during-wait, cache
  lookup, future timeout polling, on-demand submission, insert-into-
  cache on completion, statistics — all in 84 lines. Gzippy's
  `consumer_loop` does each of these AS INLINE LOGIC inside the
  closure passed to `block_fetcher.get` — which is itself a
  one-shot wrapper that calls the closure (block_fetcher.rs `get`
  method). The vendor's primitive is bypassed.
- vendor's `deflate::Block<true>` IS the decoder loop. Gzippy
  ported it as `deflate_block.rs` (964 lines, faithful, tests pass)
  AND wrote `fast_marker_inflate.rs` (2356 lines, gzippy-original,
  in production). Two decoders, the wrong one wired.
- vendor's `Cache<K,V,LRU>` IS the chunk store. Gzippy ported it as
  `cache.rs` (289 lines, faithful, tests pass) AND has an in-flight
  `spec` HashMap that does the equivalent job for the one queue that
  matters. Cache is unused.

The user is right. We have one correct, complete implementation on
disk: vendor's. We did not port it. We built a parallel structure
next to it.

### A3. Straight-line plan (stop hill-climbing; start copying)

The remaining work is NOT a sequence of perf knobs. It is a single
structural cutover that replaces `chunk_fetcher::drive` /
`consumer_loop` / `worker_loop` / `fast_marker_inflate` with a literal
port of `ParallelGzipReader::read` + `processNextChunk` +
`decodeBlock`. The doc's §I "Single-commit cutover plan" got the spec
right; we have been ignoring it. Below is the straight-line action
list. NO sub-agent should be allowed to ship a change that isn't on
this list, and NO "perf" commit is acceptable until step 9.

1. **Delete the consumer's spec ring.** Remove `spec:
   HashMap<usize, SpecSlot>` (chunk_fetcher.rs:767) and every line
   that touches it. Remove the inline drain logic in the
   `block_fetcher.get` closure (chunk_fetcher.rs:944-1066). The
   `submit_job` helper goes too. The 700-line `consumer_loop` shrinks
   to ~80 lines: a loop that calls `block_fetcher.get(blockOffset,
   blockIndex, partitionOffsetFn)`, gets back an `Arc<ChunkData>`,
   does `postProcessChunk` + `setEncodedOffset` +
   `appendSubchunksToIndexes`, hands the chunk to the writer.

2. **Make `BlockFetcher::get` the dispatch primitive.** Inside
   `block_fetcher.rs`, port the body of `core/BlockFetcher.hpp:245-329`
   literally: `getFromCaches` → `takeFromPrefetchQueue` → submit
   on-demand task if needed → `prefetchNewBlocks` during the wait →
   `insertIntoCache` on completion. This means BlockFetcher OWNS the
   thread pool (or holds a reference to it), the prefetch map, and
   the decision logic. Stop calling it from a closure as if it were
   passive.

3. **Replace `std::thread::scope + Mutex<mpsc::Receiver<Job>>` with a
   real `ThreadPool`.** Wire `parallel/thread_pool.rs` (already
   ported, `87cbc54`) and the `JoiningThread` / `AtomicMutex`
   primitives to provide `submit_task(F) -> Future<R>`. BlockFetcher
   submits to this pool. PostProcess submits to this pool. No mpsc
   channels in the production path. This eliminates the `spec`
   HashMap as a category — futures live in `m_prefetching` keyed by
   `blockOffset`, exactly mirroring vendor.

4. **Stop calling `fast_marker_inflate`. Wire `deflate_block.rs`
   into `finish_decode_chunk_with_inexact_offset`.** Replace the
   `decode_chunk_bootstrap` call in
   `gzip_chunk::finish_decode_chunk_with_inexact_offset`
   (gzip_chunk.rs:234-462) with a `deflate_block::Block` driven in
   the same loop shape as `decodeChunkWithRapidgzip`
   (GzipChunk.hpp:413-657): instantiate Block, `read_header`, `read`
   in a loop, hand off to `IsalInflateWrapper::set_window` when
   `cleanDataCount >= MAX_WINDOW_SIZE`. Then **delete
   `fast_marker_inflate.rs` and `replace_markers.rs`'s newest-byte
   encoding** — flip `MARKER_BASE` to oldest-byte-indexed (B1) and
   delete the 2356 lines of gzippy-original marker code.

5. **Replace `WindowMap` with `Arc<CompressedVector>`-valued
   storage.** Remove the Condvar (`window_map.rs:23-96`). The
   `BlockFetcher::get` flow already waits via the future returned by
   `submit_task` — workers don't need to block on a Condvar because
   they will be SCHEDULED with the predecessor's window in hand by
   the consumer (vendor pattern). Use `CompressedVector` for storage.

6. **Drive `Prefetcher` and `Cache` from `BlockFetcher`.** They're
   already ported (`prefetcher.rs`, `cache.rs`). Plug them into
   `block_fetcher.rs::get` per the vendor body. Stop wrapping the
   cache as a passive bookkeeping store outside the dispatch.

7. **Move multi-stream loop into the driver, not the chunk decoder.**
   `decompress_parallel` (single_member.rs:68-148) currently reads
   the trailer inline assuming single-member; the chunk decoder
   handles the loop. Vendor handles multi-stream end-to-end via
   `processNextChunk` returning chunks across stream boundaries. Port
   that shape: `decompress_parallel` calls
   `parallel_gzip_reader::read(...)` which loops `processNextChunk`
   until EOF — the function that returns the running CRC and total
   size, not the wrapper that calls a 700-line `drive`.

8. **Delete deletion-trap counters that exist only to prove `port(...)`
   commits "wired" something.** `BLOCK_FETCHER_GETS_OBSERVED`,
   `BLOCK_FETCHER_GET_CALLS_OBSERVED`,
   `GZIP_BLOCK_FINDER_INSERTS_OBSERVED`,
   `MARKER_PIPELINE_RUNS` (chunk_fetcher.rs:107-130, single_member.rs:44).
   They mask "is it wired?" with "is a counter incrementing?". Once
   the architecture is the port, the counters are noise. The single
   no-fallback test (`test_marker_pipeline_actually_runs_on_x86_64_isal`)
   becomes a `path == DecodePath::IsalParallelSM` assertion at the
   routing layer.

9. **THEN measure.** After steps 1-8, `make ship` runs against the
   structural port and compares to vendor on the same hardware. If
   there is still a gap, it is a TUNING gap (cache sizes, prefetch
   depth, thread count) and the perf optimization work begins — with
   an architecture that matches vendor's, so each tuning change is a
   parameter sweep, not a rewrite. **No perf commit is acceptable
   before step 9 completes.**

10. **Pre-commit invariant for future work.** Add a
    `scripts/diff_against_vendor.sh` that prints the LOC ratio of
    `src/decompress/parallel/{chunk_fetcher,single_member,gzip_chunk}.rs`
    vs `vendor/.../GzipChunkFetcher.hpp + ParallelGzipReader.hpp +
    GzipChunk.hpp`. If gzippy's production-hot LOC exceeds 1.5×
    vendor's, the structural port has regressed.

### A4. Sanity checks before any of this lands

- This audit does NOT delete the existing prior §0–§J content below.
  That content is the working state map; the audit adds intent on top.
- The §I "Single-commit cutover plan" further down was already
  correct. Steps 1–10 above are an ordering + commitment to it. If a
  future audit finds §I contradicts the steps above, §I is
  authoritative for spec; the steps above are authoritative for the
  user-imposed "stop hill-climbing" tone.
- The single test that prevents future regression of THIS audit's
  recommendation: `scripts/diff_against_vendor.sh` invoked from
  `make` and asserting the LOC ratio. Code review can rubber-stamp;
  the ratio cannot.

---

> **Status at the head of branch `feat/cross-chunk-retry`.** Reset of the
> previous reference; the prior "Step 1..16 closure plan" is gone — most
> of those steps landed as *port files* but were never wired into the
> production decode path. This doc reflects what is actually executed
> today (the marker pipeline plus an ad-hoc fetcher) versus what sits
> next to it as reference code (the new rapidgzip-shaped ports).

## CUTOVER POLICY (May 2026, user directive)

**SCOPE: ALL GZIP-RELEVANT RAPIDGZIP, not just parallel single-member.** The
goal is rapidgzip's *algorithmic speed* for gzip-family streams (gzip,
multi-member gzip, BGZF), not rapidgzip's *format breadth* (BZIP2, ZLIB —
out of scope per user 2026-05-17). GNU gzip supports gzip-family only;
gzippy mirrors that scope. Port every gzip-relevant primitive, decoder,
block finder, huffman variant, reader, index, and analyzer from
`vendor/rapidgzip/librapidarchive/` into gzippy with vendor file:line
citations. See CLAUDE.md "Cutover Goal" for the full inventory.

**Explicitly OUT OF SCOPE:** `chunkdecoding/Bzip2Chunk.hpp`,
`indexed_bzip2/bzip2.hpp` (BZIP2 decoder), `gzip/zlib.hpp`'s
`ZlibInflateWrapper` (the format header parsed at `zlib_format.rs` landed
incidentally and is harmless to keep).

This doc focused on the parallel single-member path because that was the
first cutover area. As ports of BZIP2, ZLIB, IndexFileFormat, ParallelGzipReader,
ThreadPool, blockfinders, and remaining Huffman variants land, this doc
should grow to cover them too (per-area component map, deviations, missing
pieces, extra pieces).

The goal until the rapidgzip port is structurally complete is **faithful
correctness, not performance**. Intermediate states will be slower than
steady state. That is expected and accepted.

**Rules for the cutover period:**

- **DO NOT revert perf regressions.** Ship every faithful port commit
  even when `make bench-sm` shows a perf loss. Performance is optimized
  LATER, after the structural port is complete.
- **PORT, DON'T INNOVATE.** Every change must mirror a specific rapidgzip
  C++ region. Cite `vendor/.../file:line` in code comments AND commit
  messages. If no rapidgzip counterpart exists, do not write the code.
  Crossbeam-channel, custom thread pools, unique algorithms: all out of
  scope during the cutover.
- **CORRECTNESS still matters.** Output bytes, CRC32, ISIZE must always
  verify. Tests under `src/tests/routing.rs` must stay green.
- **NO FALLBACKS.** Failure is `Err(GzippyError::Decompression(_))`.
  No silent libdeflate or ISA-L retries.

Done when an Opus advisor agrees gzippy structurally and calculationally
matches the FULL rapidgzip surface (not a subset). Then — and only then —
does performance optimization start.

Citations are `file:line` against:

- gzippy: `src/decompress/parallel/*.rs`, `src/decompress/mod.rs`,
  `src/tests/routing.rs`, `src/backends/{libdeflate,isal_decompress}.rs`
- rapidgzip: `vendor/rapidgzip/librapidarchive/src/rapidgzip/*.hpp` and
  `vendor/rapidgzip/librapidarchive/src/core/*.hpp`

Status legend:

- **Faithful** — semantics + control flow match modulo language idioms.
- **Deviated** — deliberate semantic difference (documented per item).
- **Missing** — rapidgzip has it; gzippy does not.
- **Extra** — gzippy has it; rapidgzip does not.
- **Unused** — gzippy has a faithful (or close) port file, but no
  production code path calls it. Tests do.

---

## 0. TL;DR — what runs in production today

`src/decompress/mod.rs:207-260` — `decompress_single_member` routes to
the parallel path only when **all** of:

1. `crate::backends::isal_decompress::is_available()` is true (x86_64
   + `isal-compression` feature)
2. `num_threads > 1`
3. `data.len() > 10 MiB`

When that gate passes, control enters
`src/decompress/parallel/single_member.rs::decompress_parallel`
which calls `chunk_fetcher::drive`
(`src/decompress/parallel/chunk_fetcher.rs:114-177`). `drive` spawns a
mpsc-based worker pool, statically partitions the input by
`TARGET_COMPRESSED_CHUNK_BYTES` (4 MiB), and runs `worker_loop` +
`consumer_loop`. Workers either take the FAST PATH
(`decode_chunk_with_window` — ISA-L with a known dict) or the SLOW PATH
(`decode_or_iterate` → `finish_decode_chunk_with_inexact_offset`, which
delegates the actual decode to
`fast_marker_inflate.rs::decode_chunk_bootstrap` — our **non-rapidgzip**
marker decoder).

Everything else under `src/decompress/parallel/` — `block_fetcher.rs`,
`block_map.rs`, `cache.rs`, `prefetcher.rs`, `statistics.rs`,
`compressed_vector.rs`, `deflate_block.rs`, `gzip_format.rs` (except
its `read_header` thin call) — is **unused by the production path**.
They are reference ports next to the live code.

The libdeflate fallback at `src/decompress/mod.rs:240-259` fires
silently when (i) the parallel gate doesn't trigger, or (ii) the
parallel path returns `ParallelError::TooSmall`. This is the violation
of the no-fallback invariant. See §F.

---

## A. Component map

| rapidgzip file / class | Rust counterpart | Status |
|------------------------|------------------|--------|
| `chunkdecoding/GzipChunk.hpp` — `GzipChunk<T>::decodeChunk` (entry, L660–L852) | `parallel/gzip_chunk.rs` + `parallel/chunk_fetcher.rs::decode_or_iterate` (L187-244) | Deviated (used in prod) |
| `GzipChunk::decodeChunkWithInflateWrapper` (L190-L268) | `gzip_chunk.rs::decode_chunk_with_inflate_wrapper` (L182-219) | Faithful subset; **no production caller** |
| `GzipChunk::decodeChunkWithRapidgzip` (L413-L657) | (no full counterpart) — partial behavior in `fast_marker_inflate::decode_chunk_bootstrap` + `gzip_chunk::finish_decode_chunk_with_inexact_offset` | Deviated (used in prod via marker bootstrap, not via `deflate::Block`) |
| `GzipChunk::finishDecodeChunkWithInexactOffset` (L280-L410) | `gzip_chunk.rs::finish_decode_chunk_with_inexact_offset` (L234-462) + `decode_chunk_with_window` (L86-165) | Deviated (used in prod) |
| `GzipChunk::tryToDecode` lambda (L712-L734) | `chunk_fetcher.rs::decode_or_iterate` (L187-244) | Deviated; direct-try-at-start deleted (B10) |
| `GzipChunk::appendDeflateBlockBoundary` (L161-L183) | `chunk_data.rs::ChunkData::append_block_boundary` (L264-285) | Deviated (no split-size gating) |
| `GzipChunk::startNewSubchunk` (L46-L58) | `chunk_data.rs::ChunkData::new` first-subchunk init (L135-156) | Faithful |
| `GzipChunk::finalizeChunk` (L135-L159) | `chunk_data.rs::ChunkData::finalize` (L293-301) | Deviated (no merge, no window finalization) |
| `GzipChunk::finalizeWindowForLastSubchunk` (L99-L133) | (none) | Missing |
| `GzipChunk::determineUsedWindowSymbolsForLastSubchunk` (L60-L97) | (none) | Missing |
| `GzipChunkFetcher.hpp` — `GzipChunkFetcher<T,S>::get` (L206-L225) | `chunk_fetcher.rs::drive` + `consumer_loop` (L114-698) | Deviated (used in prod) |
| `GzipChunkFetcher::processNextChunk` (L311-L362) | `chunk_fetcher.rs::consumer_loop` inner per-iteration body (L470-695) | Deviated (used in prod, no BlockFetcher) |
| `GzipChunkFetcher::decodeBlock` override (L692-L729) | `chunk_fetcher.rs::worker_loop` (L246-373) | Deviated (used in prod) |
| `GzipChunkFetcher::waitForReplacedMarkers` (L478-L518) | `consumer_loop` synchronous `apply_window` block (L596-639) | Deviated (single-threaded, single chunk) |
| `GzipChunkFetcher::queueChunkForPostProcessing` (L553-L583) | (inlined synchronously into `consumer_loop`) | Deviated |
| `GzipChunkFetcher::appendSubchunksToIndexes` (L364-L465) | `block_map::append_subchunks_to_block_map` (L205-213) | **Unused** in prod; only test calls it |
| `GzipChunkFetcher::Statistics` (L55-L75) | `statistics::FetcherStatistics` (L131-300+) | **Unused** in prod |
| `core/BlockFetcher.hpp` — `BlockFetcher<F,D,S>` (L41-end) | `parallel/block_fetcher.rs::BlockFetcher` (L35-235) | **Unused** in prod (drive() uses ad-hoc mpsc + spec[] ring instead, L422-668) |
| `BlockFetcher::Statistics` (L53-L155) | `statistics::ChunkFetcherStatistics` (composed from cache stats + FetcherStatistics) | **Unused** in prod |
| `core/Cache.hpp` — `Cache<K,V,Strategy>` | `parallel/cache.rs::Cache` + `LeastRecentlyUsed` (L39-219) | **Unused** in prod |
| `core/Prefetcher.hpp` — `FetchingStrategy`/`FetchNextFixed`/`FetchNextAdaptive` | `parallel/prefetcher.rs::FetchingStrategy` + `FetchNextFixed`/`FetchNextAdaptive` (L1-282) | **Unused** in prod |
| `core/ThreadPool.hpp` | `std::thread::scope` + `std::sync::mpsc` in `chunk_fetcher.rs` | Deviated (no `ThreadPool` abstraction) |
| `GzipBlockFinder.hpp` — partitioning + `get`/`insert`/`finalize` (L34-end) | (uniform static partition in `chunk_fetcher.rs::drive` L120-123) | Deviated (no GzipBlockFinder class at all) |
| `blockfinder/DynamicHuffman.hpp` — `seekToNonFinalDynamicDeflateBlock` (L166-298) | `block_finder.rs::BlockFinder::find_dynamic_blocks` (L686-795) | Faithful (with deviations B7) |
| `blockfinder/DynamicHuffman.hpp` — LUT generator + `nextDeflateCandidate` (L39-154) | `block_finder.rs::is_deflate_candidate_n` / `next_deflate_candidate` / `generate_deflate_lut` (L62-123) | Faithful |
| `blockfinder/precodecheck/CountAllocatedLeaves.hpp` — `checkPrecode` (L95-213) | `block_finder.rs::validate_precode` (L164-216) | Deviated (different LUT layout) |
| `blockfinder/Uncompressed.hpp` — `seekToNonFinalUncompressedDeflateBlock` (L21-95) | `block_finder.rs::BlockFinder::find_uncompressed_blocks` (L562-665) | Faithful |
| `blockfinder/Bgzf.hpp` | (handled outside the parallel SM module in `decompress/bgzf.rs`) | Out of scope here |
| `huffman/HuffmanCodingISAL.hpp` — lit/len wrapper (L21-188) | `isal_huffman.rs::IsalLitLenCode` (L80-226) | Faithful |
| `huffman/HuffmanCodingDistanceISAL.hpp` — distance wrapper (L20-124) | `isal_huffman.rs::IsalDistCode` (L243-336) | Faithful |
| `huffman/HuffmanCodingDoubleLiteralCached.hpp` | (none) | Missing |
| `huffman/HuffmanCodingReversedBitsCached*.hpp` | `block_finder.rs::build_huffman_table` (L956-997) | Deviated (block-finder only; not in decoder hot loop) |
| `huffman/HuffmanCodingShortBitsCached*.hpp` | (none) | Missing |
| `MarkerReplacement.hpp` — `MapMarkers<FULL_WINDOW>` (L15-60) | `replace_markers.rs::replace_markers{,_avx2,_neon,_scalar}` (L30-156) | Deviated (different encoding — B1) |
| `WindowMap.hpp` (L19-end) | `window_map.rs::WindowMap` (L23-96) | Deviated (BTreeMap+Condvar, no compression) |
| `ChunkData.hpp` — `ChunkData` (L88-573) | `chunk_data.rs::ChunkData` (L86-317) | Deviated (smaller surface) |
| `ChunkData::Configuration` (L97-113) | `chunk_data.rs::ChunkConfiguration` (L55-79) | Deviated (fewer fields) |
| `ChunkData::Subchunk` (L115-145) | `chunk_data.rs::Subchunk` (L29-37) | Deviated (no `newlineCount`, `usedWindowSymbols`) |
| `ChunkData::Statistics` (L147-180) | `chunk_data.rs::ChunkStatistics` (L43-50) | Deviated (smaller) |
| `ChunkData::applyWindow` (L246-394) | `apply_window.rs::apply_window` (L17-65) | Deviated (no per-subchunk windowing, no compression) |
| `ChunkData::appendDeflateBlockBoundary` (L455-467) | `chunk_data.rs::append_block_boundary` (L264-285) | Deviated |
| `ChunkData::matchesEncodedOffset` (L396-403) | `chunk_data.rs::matches_encoded_offset` (L163-165) | Faithful |
| `ChunkData::setEncodedOffset` (L601-629) | `chunk_data.rs::set_encoded_offset` (used by consumer `consumer_loop` L527) | Faithful (recent landing) |
| `ChunkData::split` (L632-754) | (none — `append_block_boundary` emits subchunks inline) | Missing semantically |
| `ChunkData::getWindowAt` / `getLastWindow` (DecodedData L394-488) | `chunk_data.rs::last_32kib_window` (L173-207) | Deviated (returns `Option`, no MapMarkers) |
| `ChunkData::cleanUnmarkedData` (DecodedData L491-516) | `chunk_data.rs::clean_unmarked_data` (port landed; **unused** in prod path) | Unused |
| `ChunkData::appendFooter` / `footers` / `crc32s` (L472-489, L559-561) | `chunk_data.rs::Footer` + `crc32s: Vec<crc32fast::Hasher>` (struct fields landed; **single-stream only** in prod) | Deviated + partially unused |
| `CompressedVector.hpp` | `parallel/compressed_vector.rs::CompressedVector` (L39-219) | **Unused** in prod (WindowMap holds `Arc<[u8;32768]>`) |
| `DecodedData.hpp` — base class | folded into `chunk_data.rs::ChunkData` | Deviated |
| `DecodedDataView.hpp` | (none) | Missing |
| `gzip/isal.hpp` — `IsalInflateWrapper` (L26-212) | `inflate_wrapper.rs::IsalInflateWrapper` (L102-311) | Deviated (B4) |
| `gzip/isal.hpp` — `readGzipFooter`/`readZlibFooter`/`readDeflateFooter`/`readHeader`/`readIsalHeader` (L429-560) | `inflate_wrapper.rs::read_footer_at_current` + `reset_for_next_stream` (L243-287) + `gzip_format::read_header` (L47+) | Partially ported; **multi-stream loop not wired** into worker |
| `gzip/deflate.hpp` — `deflate::Block<>` (L513-2005) | `parallel/deflate_block.rs::Block` (L98-495) | **Unused** in prod (production calls `fast_marker_inflate.rs` for the bootstrap instead) |
| `gzip/deflate.hpp` — `deflate::Block::readHeader` / `readDynamicHuffmanCoding` (L579, L587) | `deflate_block.rs::read_header` (L224) + `read_dynamic_huffman_coding` (L296) | Faithful; **unused** in prod |
| `gzip/deflate.hpp` — `deflate::Block::read` + `read_internal_compressed` (L1514-1582) | `deflate_block.rs::read` (L343) + `read_internal_compressed` (L399) | Faithful; **unused** in prod |
| `gzip/deflate.hpp` — `deflate::Block::setInitialWindow` (L607) | (none — gzippy uses `IsalInflateWrapper::set_window` directly) | Missing |
| `gzip/deflate.hpp` — circular `m_window16` (2×32 KiB u16 buffer) | (none — `fast_marker_inflate` grows a single `Vec<u16>`) | Missing |
| `gzip/deflate.hpp` — `BlockStatistics` (L456-503) | (none) | Missing |
| `gzip/format.hpp` — `determineFileTypeAndOffset` (L17-58) | (none; `gzip_format::read_header` does gzip only) | Missing |
| `gzip/definitions.hpp` — `CompressionType`, `StoppingPoint`, `BlockBoundary`, `MAX_WINDOW_SIZE` | `inflate_wrapper.rs::DeflateCompressionType`, `StoppingPoints`; constants scattered | Deviated (names) |
| `gzip/gzip.hpp` — `readHeader`, `readFooter`, `Footer` | `gzip_format.rs::{read_header, read_footer, Header, Footer}` (L23-end) | Faithful (read_header used by `skip_gzip_header`; read_footer **unused**) |
| `gzip/GzipReader.hpp` | (none) | Out of scope |
| `gzip/GzipAnalyzer.hpp` | (none) | Out of scope |
| `ParallelGzipReader.hpp` | (driver fragmented across `decompress/mod.rs` + `single_member.rs`; not a port) | Deviated |
| `IndexFileFormat.hpp` | (none) | Missing |
| `chunkdecoding/Bzip2Chunk.hpp` | (none) | Missing (gzippy is gzip-only) |
| `chunkdecoding/DecompressionError.hpp` — `NoBlockInRange` | `gzip_chunk.rs::ChunkDecodeError` (L51-68) | Deviated |
| (none in rapidgzip) | `parallel/trace.rs` — JSON-lines structured trace | Extra |
| (none in rapidgzip) | `fast_marker_inflate.rs` — entire 2356-line marker decoder | **Extra (production)** |
| (none in rapidgzip) | `fast_marker_inflate::validate_boundary` | Extra |
| (none in rapidgzip) | `replace_markers_avx2` / `replace_markers_neon` | Extra |
| (none in rapidgzip) | `block_finder::validate_fixed_block_prefix` | Extra |
| (none in rapidgzip) | `block_finder::find_blocks_parallel` | Extra |
| (none in rapidgzip) | `single_member::MARKER_PIPELINE_RUNS` / `MARKER_PIPELINE_TEST_LOCK` | Extra (test infra) |
| (none in rapidgzip) | `decompress::mod.rs::map_parallel_single_member_error` (L262-287) | Extra (the fallback wire) |

---

## B. Concrete deviations

### B1. Marker encoding (`MapMarkers` vs `replace_markers`)

**rapidgzip** (`MarkerReplacement.hpp:15-46`, `DecodedData.hpp:305-391`):
`v <= 0xFF` → literal. `v < MAX_WINDOW_SIZE` (and > 0xFF) → throws.
`v >= MAX_WINDOW_SIZE` → `window[v - MAX_WINDOW_SIZE]` (index from
**oldest** byte).

**gzippy** (`replace_markers.rs:24,49-58`): `MARKER_BASE = 32768`. `v <
32768` is treated as literal. `v >= 32768` → `window[window.len() - 1 -
(v - 32768)]` (index from **newest** byte).

**Impact**: high — incompatible with any rapidgzip code that uses
`MapMarkers`. Earlier docs claimed Step 1 "flipped encoding to
MapMarkers" but the file at HEAD still has the newest-byte-indexed
encoding. Anything ported that depends on `MapMarkers` (e.g. our
unused `deflate_block::read_internal_compressed` calling
`emit_backref` — `deflate_block.rs:640-675`) uses a different marker
convention than `replace_markers.rs`. Two incompatible marker codes
live in the same crate.

### B2. `appendDeflateBlockBoundary` semantics

rapidgzip (`ChunkData.hpp:455-467`, `GzipChunk.hpp:161-183`) dedups on
`(encodedOffset, decodedOffset)` and gates new-subchunk creation on
`splitChunkSize`. `ChunkData::split` partitions the boundary list after
the fact.

gzippy (`chunk_data.rs::append_block_boundary` L264-285) dedups on
encoded_offset only and **always** pushes a new subchunk. No
`ChunkData::split`. Consumer locates subchunks via `decoded_offset_for`
at L308-316.

**Impact**: behaviourally fine for single-pass write; gzippy keeps more
subchunks than rapidgzip. Missing rapidgzip's "merge small trailing
subchunk" semantic.

### B3. `applyWindow`

rapidgzip (`ChunkData.hpp:246-394`) flips `dataWithMarkers` into
`reusedDataBuffers`, walks subchunks computing per-subchunk windows via
`getWindowAt`, stores them as `SharedWindow` (CompressedVector),
optionally counts newlines, and updates the first `crc32s` entry.

gzippy (`apply_window.rs:17-65`) calls `replace_markers` in place on
`data_with_markers` (left as `Vec<u16>` whose values are all `<= 255`
after the resolve), updates `chunk.crc` by streaming 4 KiB scratch
chunks. Per-subchunk windows are populated via
`chunk_data::populate_subchunk_windows` (called from
`chunk_fetcher.rs:614`) but stored uncompressed.

**Impact**: behaviourally fine for our single-pass write. The
per-subchunk window populate is wired in production (good). Per-window
CompressedVector storage is now wired via `WindowMap` (commit
`17fd9b2`); single-member production uses `CompressionType::None`
since each window is consumed once. The `Window = Arc<[u8; 32768]>`
type signature on `WindowMap::get` is the remaining structural
divergence (task #79).

### B4. `IsalInflateWrapper::readStream`

rapidgzip (`gzip/isal.hpp:253-385`) handles `END_OF_STREAM` and
emits `END_OF_STREAM_HEADER`, returns `pair<size_t,
optional<Footer>>`, refills its buffer 128 KiB at a time from a
`BitReader`, and throws on negative ISA-L returns with a hex dump.

gzippy (`inflate_wrapper.rs:249-310`) assumes the entire slice is in
memory (no buffer refill), returns `ReadStreamResult { bytes_written,
stopped_at, bit_position, finished }` with **no Footer**. The
`read_footer_at_current` (L243-272) and `reset_for_next_stream`
(L277-287) methods exist (Step 11 landed) but **`worker_loop` does not
loop on multi-stream input**. Single-stream only in practice.

### B5. Consumer / fetcher architecture

rapidgzip (`GzipChunkFetcher.hpp:311-362` + `core/BlockFetcher.hpp`):
`processNextChunk` consults `GzipBlockFinder` for the next offset,
calls `BlockFetcher::get(partitionOffset, blockIndex)` to pull from
`Cache`, await a prefetch via `takeFromPrefetchQueue` (L385-410), or
trigger a fresh `decodeBlock`. `Prefetcher` strategy decides which
indexes to prefetch. `Cache` is LRU. `ThreadPool` runs the futures.

gzippy (`chunk_fetcher.rs:114-698`): `drive` spawns a fixed pool of
workers sharing a single `Mutex<mpsc::Receiver<DecodeJob>>`
(L129-130). `consumer_loop` keeps a `Vec<Option<Receiver>>` of in-flight
speculative jobs per partition (L422-423), pre-fills `pool_size * 2`
speculatives (L463-465), drains in order. On miss, re-dispatches
authoritative at the predecessor's actual end (L573).

**Partial-port status (May 2026, post commit `306c1e7` and successor
`feat/cross-chunk-retry`)**: `BlockFetcher` + `BlockMap` +
`GzipBlockFinder` all wired. The consumer calls
`block_fetcher.record_fetch`, `get_if_available`,
`record_on_demand_fetch`, `record_prefetch_cache_*`,
`block_fetcher.get` (synchronous dispatch primitive — closure submits
to mpsc worker pool, blocks on reply, returns resolved chunk; on
miss, caches `Arc<ChunkData>` under the block offset exactly as
rapidgzip's `insertIntoCache` at BlockFetcher.hpp:320), and
`append_subchunks_to_block_map(block_map, &chunk)` +
`block_finder.insert(subchunk_end)` per subchunk at
`chunk_fetcher.rs::consumer_loop`.

`ChunkData` is `Clone` (chunk_data.rs:106), so the cache-aliased
`Arc<ChunkData>` returned by `get` follows rapidgzip's
`shared_ptr<ChunkData>` aliasing model: consumer obtains an owned
mutation copy via `Arc::try_unwrap` (sole holder) or `(*arc).clone()`
(cache also holds it). The `crc32fast::Hasher` embedded in
`ChunkData` is itself `Clone` (serializes its rolling state).

All chunk acquisition flows through `block_fetcher.get(expected_start,
dispatch)` — the dispatch closure first drains a pre-filled
`SpecSlot` (validates the speculative hit), and only on miss submits
authoritative to the worker pool. This matches rapidgzip's
`processNextChunk` single-call shape at GzipChunkFetcher.hpp:293
(`getBlock` → `BaseType::get`).

What's still divergent (deferred):
1. **Thread pool unification.** Rapidgzip uses `BS::thread_pool` +
   `std::future`; gzippy keeps `std::thread::scope` + mpsc. The
   closure passed to `block_fetcher.get` does the
   `submit_job(...).recv()` dance to integrate. Caller-side API shape
   is identical (single synchronous call returns resolved `ChunkData`);
   the threading idiom is the deviation.
2. **Spec-ring pre-fill is caller-managed.** Rapidgzip's
   `BlockFetcher::get` triggers `prefetchNewBlocks` during the wait on
   the on-demand task (BlockFetcher.hpp:297-299, 314-316), overlapping
   prefetches with the blocking dispatch. Gzippy maintains the
   `SpecSlot` pre-fill ring in `consumer_loop` (pre-filled to
   `pool_size * 2` ahead of the consumer; refilled after each
   processed chunk). The DRAIN of a slot is inside the
   `block_fetcher.get` dispatch closure (the speculative-hit branch);
   the PRE-FILL is still caller-side. Function: equivalent. Shape:
   pre-fill orchestration is the remaining deviation.

`Cache` (LRU from `cache.rs`) and `FetchingStrategy` (from
`prefetcher.rs`) are now consumed by `BlockFetcher` directly
(`block_fetcher.rs:38-39` imports `Cache`, `CacheStrategy`,
`LeastRecentlyUsed`, `FetchingStrategy`); the prefetch dispatcher
plugged in at `chunk_fetcher.rs:229` is `FetchNextAdaptive`. The
spec-ring lives alongside (consumer-side pre-fill) until the prefetch
trigger is moved inside `BlockFetcher::get` per #2.

### B6. Block finder partitioning

Wired (May 2026, branch `feat/cross-chunk-retry`):
`GzipBlockFinder` in `gzip_block_finder.rs` is constructed in
`chunk_fetcher::drive` with the first confirmed offset (`0`) and
spacing = `configuration.split_chunk_size`. Consumer queries
`block_finder.get(idx)` for partition seeds (literal port of
GzipChunkFetcher.hpp:318), and post-process calls
`block_finder.insert(subchunk_end)` per subchunk (literal port of
GzipChunkFetcher.hpp:374). The partitioner-feedback loop is closed:
later partitions' seeds reflect earlier partitions' actual ends.
`block_finder.finalize()` runs at end of `drive`.

### B7. Block-finder validation strictness

rapidgzip maintains two bit buffers (LUT-sized + 61-bit) and a signed
`int8_t` LUT (negative magnitude = partial verify needed). gzippy
(`block_finder.rs:686-795`) uses a single `BitReader`, an `i8` LUT with
no negative-encoding branch, and re-seeks on each candidate. Same
acceptance criteria, different LUT.

### B8. BTYPE handling in block finder

rapidgzip emits dynamic OR uncompressed candidates only (fixed Huffman
has no header redundancy). gzippy
(`block_finder.rs:449-501,674-680`) ALSO has a
`validate_fixed_block_prefix` prefilter that emits BTYPE=01
candidates. The decoder hot path
(`fast_marker_inflate::validate_boundary`) rejects them via
`require_non_fixed_stop`.

### B9. `finishDecodeChunkWithInexactOffset` END_OF_BLOCK_HEADER guard

rapidgzip stops at `bit_position` (post-header). gzippy
(`gzip_chunk.rs:139-148,412-422`) stops at `last_eob_pos` (pre-header,
i.e. the previous END_OF_BLOCK boundary). Logically equivalent;
gzippy's variant is so the next chunk's worker resumes at the start of
a new block header.

### B10. `decode_or_iterate` vs `decodeChunkWithRapidgzip` direct-try

rapidgzip (`GzipChunk.hpp:736-741`) attempts a direct decode at the
guessed `blockOffset` first, then iterates BlockFinder candidates.
gzippy (`chunk_fetcher.rs:194-203`) deletes the direct-try-at-start
entirely — the comment explains that our marker bootstrap is too
lenient on malformed headers and false-positives silently.

### B11. Chunk handoff to ISA-L

rapidgzip (`GzipChunk.hpp:520-526`) tracks cumulative `cleanDataCount`
and hands off to ISA-L when it crosses `MAX_WINDOW_SIZE`. gzippy
(`fast_marker_inflate::decode_chunk_bootstrap` L460-551,
`gzip_chunk::finish_decode_chunk_with_inexact_offset` L240-268) uses a
`CleanTailTracker` that only counts the **trailing** clean bytes.
Stricter; produces more marker-only chunks.

### B12. `WindowMap` blocking and storage — RESOLVED (2026-05-17 / 2026-05-18)

Closed by audit step 5 (Condvar removed; vendor has no condvar) and
the 2026-05-18 commits:
- `7c524da` zero-alloc presence check (`contains`) + None-fast
  `decompress_to_window`.
- `84d2ad6` overwrite semantic on `insert` (was first-wins, vendor
  uses `insert_or_assign`).
- `ba6540a` `pub type Window = Arc<CompressedVector>` — matches
  vendor `SharedWindow = shared_ptr<const CompressedVector>`
  exactly. `get()` returns the shared pointer directly (zero alloc);
  callers materialize bytes via the new `materialize_window` helper
  (`Cow<[u8]>`, zero-alloc borrow for None compression).

### B13. Worker decode pathways

rapidgzip has one entry: `decodeBlock` → `decodeChunk` → 3 internal
modes depending on (initialWindow, untilOffsetIsExact). gzippy has two
worker paths selected by `WindowMap::get_or_wait`
(`chunk_fetcher.rs:269-326`): fast (window known →
`decode_chunk_with_window`) vs slow (no window →
`decode_or_iterate` → marker bootstrap). The exact-stop
`decode_chunk_with_inflate_wrapper` (L182-219 of gzip_chunk.rs) is
**not called from production**.

### B14. Authoritative re-dispatch / `setEncodedOffset`

rapidgzip's `processNextChunk` calls `chunkData->setEncodedOffset(...)`
to re-anchor the chunk to the real offset after decode. gzippy now
calls `chunk.set_encoded_offset(expected_start)`
(`chunk_fetcher.rs:526-528`) on speculative hits where `encoded !=
expected`; on **miss** gzippy discards the chunk and re-dispatches
authoritative (`chunk_fetcher.rs:561-583`). Rapidgzip's setEncodedOffset
trims the leading subchunk; gzippy's `set_encoded_offset` does
something similar (see `chunk_data.rs::set_encoded_offset` for
specifics).

### B15. Per-chunk Footer / multi-stream support

rapidgzip threads a `std::vector<Footer> footers` + `std::vector<CRC32Calculator>
crc32s` through ChunkData and loops over multiple streams per chunk in
`decodeChunkWithRapidgzip`. gzippy's `chunk_data::ChunkData` carries
`Vec<Footer>` and `Vec<crc32fast::Hasher>` since Step 2 landed, BUT the
production driver (`single_member.rs:65-140`) reads the trailer
**inline at the end of `decompress_parallel`**, assumes single-stream,
and would not handle a multi-member input through this path. (Routing
in `decompress/mod.rs:76-82` sends multi-member to a different code path
entirely.)

### B16. **Silent libdeflate fallback** — RESOLVED

`decompress_single_member` (`src/decompress/mod.rs:230-294`) is now a
pure dispatcher: it classifies once and hands off to exactly one
backend. Each backend either succeeds or returns
`Err(GzippyError::Decompression(_))`. The function-level doc reads
"**No fallback.**" The deletion-trap killer test
`test_marker_pipeline_actually_runs_on_x86_64_isal`
(`src/tests/routing.rs`) continues to gate the parallel path's
invocation. `LIBDEFLATE_SM_CALLS` and `ISAL_STREAM_SM_CALLS`
counters (mod.rs:262, 270) provide test-only fences confirming
the routing decision matched the gate.

---

## C. Missing pieces (rapidgzip has, gzippy doesn't)

Status legend: ✅ DONE · 🟡 PARTIAL · ❌ NOT STARTED · ⏭ DEFERRED-by-design

1. ❌ **`HuffmanCodingDoubleLiteralCached`** — multi-symbol pure-C++ decoder.
2. ❌ **`HuffmanCodingShortBitsCached*` family** — fallback Huffman decoders.
3. 🟡 **`deflate::Block<>` driving production decode** — the port exists
   in `deflate_block.rs` and is wired into the SLOW PATH bootstrap
   (`gzip_chunk.rs::bootstrap_with_deflate_block`, post commit `d973907`).
   The FAST PATH still uses ISA-L's `set_dict` rather than
   `deflate::Block` seeded via `setInitialWindow` — that swap is a
   deeper refactor, deferred.
4. ✅ **`deflate::Block::setInitialWindow`** — landed as
   `deflate_block::Block::set_initial_window` (commit `0417e98`,
   May 2026). API is callable; fast-path wiring deferred (see #3).
5. ⏭ **`getLastWindow({})` raising on markers** — rapidgzip's mechanism
   to detect "trailing 32 KiB straddles markers, can't hand off yet."
   gzippy substitutes `CleanTailTracker`. Documented in §B3; functionally
   equivalent; will not be re-ported.
6. ✅ **`BlockFetcher` wired into the production driver** — cache /
   prefetch wiring landed at `306c1e7`. `BlockFetcher::get`
   (synchronous dispatch primitive, BlockFetcher.hpp:245-329) ported in
   `block_fetcher.rs::get` and consumed by `chunk_fetcher::drive` for
   authoritative re-dispatch (deletion-trap counter
   `BLOCK_FETCHER_GET_CALLS_OBSERVED` proves wiring). Threading-pool
   unification is the §B5 deviation — closure runs on the existing
   mpsc worker pool; caller pattern (one synchronous call returns
   resolved data) is the literal port. Insert-on-miss is the caller's
   responsibility because `ChunkData` is not `Clone` — semantic-equivalent
   deviation documented in the method.
7. ✅ **`GzipBlockFinder` (the partitioner class)** — port file at
   `gzip_block_finder.rs`. Wired into `chunk_fetcher::drive`: workers
   call `block_finder.insert(actual_end)` per subchunk (literal port of
   GzipChunkFetcher.hpp:374); consumer queries `block_finder.get(idx)`
   for partition seeds and `get(idx+1)` for `until_bit`
   (literal port of GzipChunkFetcher.hpp:318 + BlockFetcher.hpp:479).
   `block_finder.finalize()` called at end of `drive` (mirror of
   GzipChunkFetcher.hpp:325). Deletion-trap counter
   `GZIP_BLOCK_FINDER_INSERTS_OBSERVED` proves the insert wiring.
8. ✅ **`BlockMap` wired into the production driver** —
   `append_subchunks_to_block_map(block_map, &chunk)` at
   `chunk_fetcher.rs:784`, post commit `306c1e7`.
9. ❌ **`IndexFileFormat`** — seekable index export. Optional.
10. ✅ **Parallel marker post-processing** — the consumer submits each
    chunk's marker resolution to the thread pool via
    `submit_post_process_to_pool` (`chunk_fetcher.rs:598`,
    `run_post_process_task` at `:823`). Mirror of vendor's
    `queueChunkForPostProcessing` (GzipChunkFetcher.hpp:579-582). The
    consumer drains FIFO via `drain_one_pending`. Open subitem: vendor
    submits with priority -1 (`submitTaskWithHighPriority` at
    BlockFetcher.hpp:606-611); gzippy's `ThreadPool::submit` has no
    priority queue, so the apply-window task competes equally with
    decode tasks. Documented behavioral deviation.
11. ✅ **Per-subchunk window publishing into a `BlockMap`-indexed lookup**
    — `populate_subchunk_windows` runs in prod (chunk_fetcher.rs:752),
    windows are inserted into `WindowMap` at chunk_fetcher.rs:769-776,
    and the BlockMap insertion is at chunk_fetcher.rs:784. The lookup
    chain is now closed.
12. ✅ **Window sparsity / window compression in the live path** —
    `CompressedVector` backs `WindowMap`'s entries (commit `17fd9b2`)
    and `pub type Window = Arc<CompressedVector>` matches vendor's
    `SharedWindow` type exactly (commit `ba6540a`). `WindowMap::get`
    returns a zero-alloc shared pointer; consumers materialize bytes
    via `materialize_window` (`Cow<[u8]>`). Single-member production
    uses `CompressionType::None` (chunk_fetcher.rs:208) to avoid
    compress/decompress overhead on the single-pass path. The
    Zlib-default lives on `parallel_gzip_reader.rs:184` (seekable
    reader) where windows accumulate.
13. ✅ **`Footer` + multi-stream loop in `worker_loop`** — landed in
    `gzip_chunk.rs::decode_chunk_with_window` outer-loop reset cycle
    (L259-302), post commit `306c1e7`.
14. ⏭ **`gzip/format.hpp::determineFileTypeAndOffset`** — gzippy detects
    BGZF / multi-member at routing layer
    (`decompress/mod.rs::classify_gzip`), not via a unified detector
    inside the parallel SM module. Functionally equivalent; routing-layer
    placement is a gzippy-specific organizational choice.
15. ❌ **Statistics & `--verbose` output** — `FetcherStatistics` is now
    populated by `chunk_fetcher.rs` (record_get, record_prefetch, etc.)
    but never exported/printed at the end of a run.
16. ⏭ **`NoBlockInRange` exception** — gzippy returns a generic
    `ExactStopMissed` and handles re-dispatch through `consumer_loop`.
    Naming-only deviation.
17. ⏭ **`finalizeChunk` subchunk merge** — rapidgzip has NO explicit
    "merge" method; the merge semantic is implicit in `split(spacing)`
    partitioning. See #18 for the deferral rationale.
18. ⏭ **`ChunkData::split` in production** — `chunk_data.rs::split` IS
    ported (L361-474). NOT called from production: gzippy's consumer
    uses `decoded_offset_for(expected_start)` to find an exact-encoded
    subchunk for speculative-trim. If `finalize` called `split(spacing)`,
    real block boundaries that don't fall on the spacing grid would be
    lost from `subchunks` and the speculation hit rate would collapse.
    Documented in §B2 as a behavioral deviation that gzippy
    intentionally retains for the speculative-trim path.
19. ✅ **`appendSubchunksToIndexes` cascade** —
    `block_map::append_subchunks_to_block_map` called at
    `chunk_fetcher.rs:784`, followed by
    `block_finder.insert(subchunk_end)` per subchunk (literal port of
    GzipChunkFetcher.hpp:374). `m_unsplitBlocks` map is still pending
    (only consulted on backward-seek replay; unused on streaming).
20. ⏭ **Direct-try-at-guessed-offset on the slow path** — deleted in
    gzippy because the marker bootstrap is too lenient (B10).
    Documented deviation, not a regression.
21. ✅ **`appendDeflateBlockBoundary` (encoded, decoded) dedup** —
    landed in commit `4612252` (May 2026). Was encoded-only; now
    matches rapidgzip's pair-dedup semantic.

---

## D. Extra pieces (gzippy has, rapidgzip doesn't)

1. **`fast_marker_inflate.rs`** — 2356-line from-scratch marker
   decoder. Production-critical (drives every slow-path bootstrap).
   Marker encoding does not match `MapMarkers` (B1). Public API is
   procedural rather than stateful.
2. **`fast_marker_inflate::validate_boundary`** — trial-decode probe
   that confirms a candidate by decoding ≥ min_blocks before
   committing. Exists because the bootstrap is unreliable on false
   positives.
3. **`fast_marker_inflate::decode_chunk_bootstrap` + `BootstrapResult`**
   — explicit "decode until 32 KiB clean tail, then hand off to ISA-L"
   contract. Substitute for rapidgzip's cleanDataCount heuristic +
   getLastWindow throw flow.
4. **`replace_markers_avx2` and `replace_markers_neon`** — SIMD
   marker resolution. Rapidgzip's MapMarkers is scalar.
5. **`block_finder::validate_fixed_block_prefix`** — ~50 ns prefilter
   for fixed-Huffman candidates.
6. **`block_finder::find_blocks_parallel`** — multi-thread block
   finder.
7. **`parallel/trace.rs`** — JSON-lines per-event trace under
   `GZIPPY_LOG_FILE`.
8. **`MARKER_PIPELINE_RUNS` + `MARKER_PIPELINE_TEST_LOCK`** — test
   counter to catch silent fallback. Catches case 2 of §F only.
9. ~~`WindowMap::get_or_wait` with Condvar~~ — REMOVED (audit
   step 5, 2026-05-17). gzippy's `WindowMap` is now plain `Mutex` +
   `BTreeMap`, vendor-faithful. Workers do non-blocking `get`;
   ordering is enforced by the per-block future dispatch from
   `BlockFetcher::get`.
10. **`ChunkDecodeError::ExactStopMissed { requested, actual }`** —
    explicit variant.
11. **The `gzippy-parallel` "GZ" FEXTRA path** (in `decompress/bgzf.rs`,
    out of scope).
12. ~~Routing-layer fallback to libdeflate~~ — REMOVED.
    `decompress_single_member` (mod.rs:230) is now a pure dispatcher
    with explicit "No fallback" contract. See B16.
13. ~~Reference ports of rapidgzip primitives sitting next to live
    code~~ — REWIRED (2026-05-18). `block_fetcher`, `block_map`,
    `cache`, `prefetcher`, `statistics`, `compressed_vector`,
    `deflate_block` are all now reachable from the production
    consumer (see §E per-file status). The remaining genuinely
    unwired modules are the in-scope vendor-port surface
    (`huffman_reversed_*`, `huffman_double_literal_cached`,
    `huffman_short_bits_*`, `index_file_format`) — kept per
    CLAUDE.md "Active port" scope listing.

---

## E. Per-file deep dive

### `src/decompress/parallel/single_member.rs` (215 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `MIN_PARALLEL_SIZE = 4 MiB`, `MIN_THREADS_FOR_PARALLEL = 2`, `TARGET_COMPRESSED_CHUNK_BYTES = 4 MiB` | L31-33 | gzippy-original; rapidgzip uses 4 MiB as `splitChunkSize` per `GzipChunkFetcher.hpp:706` |
| `MARKER_PIPELINE_RUNS` / `MARKER_PIPELINE_TEST_LOCK` | L41-47 | gzippy-original (test instrumentation) |
| `skip_gzip_header` | L61-63 | Thin wrapper around `gzip_format::read_header` |
| `decompress_parallel` driver | L65-146 | gzippy-original (rapidgzip has `ParallelGzipReader` for the same role) |
| Trailer parsing (CRC + ISIZE) | L83-98 | gzippy-original — assumes single stream |
| `ParallelError` enum | L150-183 | gzippy-original |

### `src/decompress/parallel/chunk_fetcher.rs` (758 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `DecodeJob` + mpsc work queue | L95-130 | gzippy-original (no `ThreadPool` abstraction) |
| `BOUNDARY_SEARCH_RADIUS_BYTES = 512 KiB` | L80-81 | port of cap in `GzipChunk.hpp:811` |
| `WINDOW_WAIT_TIMEOUT = 50ms` | L88-89 | gzippy-original |
| `drive()` | L114-177 | gzippy-original |
| `decode_or_iterate` | L187-244 | port of `tryToDecode` + BlockFinder loop, **minus** direct-try-at-start |
| `worker_loop` | L246-373 | gzippy-original fast/slow split via `WindowMap::get_or_wait` |
| `consumer_loop` speculative ring | L375-698 | port of `processNextChunk` + parts of `BlockFetcher::get`, **but using ad-hoc spec[] instead of BlockFetcher** |
| `set_encoded_offset` call on hit | L527 | port of rapidgzip's setEncodedOffset (B14) |
| Authoritative re-dispatch | L561-583 | port of getBlock retry in `GzipChunkFetcher.hpp:646-662` |
| Synchronous `apply_window` + populate_subchunk_windows + WindowMap insert | L596-639 | deviation from `queueChunkForPostProcessing` |
| Output write + CRC combine | L641-682 | gzippy-original |

### `src/decompress/parallel/gzip_chunk.rs` (588 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `ChunkDecodeError` | L51-68 | gzippy-original; rapidgzip throws |
| `decode_chunk_with_window` (fast path) | L86-165 | port of `finishDecodeChunkWithInexactOffset` when window is known |
| `decode_chunk_with_inflate_wrapper` (exact stop) | L182-219 | faithful port — **NOT CALLED IN PROD** |
| `finish_decode_chunk_with_inexact_offset` (slow path) | L234-462 | port of `decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset` combination; uses `fast_marker_inflate::decode_chunk_bootstrap` instead of `deflate::Block` |

### `src/decompress/parallel/chunk_data.rs` (855 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `Subchunk` | L29-37 | port of `ChunkData::Subchunk` |
| `ChunkStatistics` | L43-50 | port of `ChunkData::Statistics` |
| `ChunkConfiguration` | L55-79 | port of `ChunkData::Configuration` |
| `ChunkData` | L86-128 | port of `ChunkData` + `DecodedData` flattened |
| `matches_encoded_offset` | L163-165 | faithful port |
| `last_32kib_window` | L173-207 | port of `getLastWindow` (returns `Option`, no MapMarkers) |
| `append_markered` / `append_clean` | L222-249 | port of `DecodedData::append` variants |
| `append_block_boundary` | L264-285 | deviates from `appendDeflateBlockBoundary` (B2) |
| `finalize` | L293-301 | port of `ChunkData::finalize` (no `cleanUnmarkedData`, no `split`, no merge) |
| `decoded_offset_for` | L308-316 | gzippy-original (rapidgzip uses `BlockMap::findDataOffset`) |
| `set_encoded_offset` (Step 4) | (later lines) | port of `ChunkData::setEncodedOffset` |
| `populate_subchunk_windows` (Step 3) | (later lines) | port of subchunk-window emplacement in `appendSubchunksToIndexes` |
| `clean_unmarked_data` (Step 8) | (later lines) | port — **not called in prod** |

### `src/decompress/parallel/apply_window.rs` (159 lines)

Marker-resolve in place via `replace_markers`. Updates `chunk.crc` from
scratch 4 KiB chunks. No per-subchunk windowing in this function — that
moved to `populate_subchunk_windows` called separately by the consumer.

### `src/decompress/parallel/window_map.rs` (172 lines)

`BTreeMap<usize, Arc<[u8;32768]>>` + Mutex + Condvar. `get_or_wait`
blocks until window appears or timeout. Missing `releaseUpTo`,
`operator==`, `data()` export, `CompressedVector` storage.

### `src/decompress/parallel/block_finder.rs` (1318 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `LUT_BITS = 15` | L42 | port of `OPTIMAL_NEXT_DEFLATE_LUT_SIZE = 15` |
| `is_deflate_candidate_n` | L62-88 | faithful port |
| `next_deflate_candidate` | L95-103 | faithful port |
| `generate_deflate_lut` | L109-115 | deviates from rapidgzip's signed-LUT encoding (B7) |
| `validate_precode` | L164-216 | gzippy-original LUT layout (4-precode-at-a-time differs) |
| `BitReader` | L222-311 | gzippy-original |
| `BlockBoundary` | L317-329 | port with extra fields |
| `validate_fixed_block_prefix` | L449-501 | **Extra** |
| `find_uncompressed_blocks` | L562-665 | faithful port |
| `find_dynamic_blocks` | L686-795 | port with bitreader/LUT diffs |
| `parse_precode` / `is_valid_huffman_lengths` / `validate_huffman_codes` | L818-948 | own impl matching same checks |
| `find_blocks_parallel` | L1029-1081 | **Extra** |

### `src/decompress/parallel/inflate_wrapper.rs` (489 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `StoppingPoints` | L33-60 | port of `StoppingPoint` enum |
| `ReadStreamResult` | L62-77 | gzippy-original (rapidgzip returns `pair<size_t, optional<Footer>>`) |
| `DeflateCompressionType` | L82-86 | port |
| `InflateError` | L89-97 | gzippy-original (rapidgzip throws) |
| `IsalInflateWrapper::new` | L116-139 | port of ctor + `initStream` |
| `set_window` | L145-163 | port of `setWindow` |
| `set_stopping_points` etc. | L170-215 | port |
| `tell_compressed` | L222-232 | port |
| `read_stream` | L249-310 | port of `readStream`, **no footer/multi-stream**, no diagnostic dump |
| `read_footer_at_current` | L243-272 | port of `readGzipFooter` — Step 11, **unused in worker_loop** |
| `reset_for_next_stream` | L277-287 | port, **unused in worker_loop** |

### `src/decompress/parallel/isal_huffman.rs` (412 lines)

Faithful port of `HuffmanCodingISAL` (L80-226) and
`HuffmanCodingDistanceISAL` (L243-336). `with_thread_litlen` and
`with_thread_litlen_dist` are gzippy-original (thread-local reuse) —
rapidgzip stack-allocates these inside `deflate::Block`.

### `src/decompress/parallel/fast_marker_inflate.rs` (2356 lines)

**Not a port of `deflate::Block<>`.** A from-scratch implementation
of the same abstract idea (marker-emitting deflate decode) with
gzippy's own marker encoding (B1) and a procedural API. Production-
critical for the slow-path bootstrap.

| Item | Lines | Origin |
|------|-------|--------|
| RFC 1951 tables | L68-89 | port of `gzip/RFCTables.hpp` |
| `validate_boundary` | L174-262 | **Extra** |
| `decode_chunk_markers_bounded` | L274-347 | gzippy-original |
| `decode_chunk_bootstrap` + `BootstrapResult` | L361-551 | **Extra** |
| `CleanTailTracker` | L387-437 | gzippy-original substitute for rapidgzip's `cleanDataCount` |
| `decode_chunk_markers_continuing` | L571-615 | **Extra** |
| `decode_loop` | L622-748 | gzippy-original |
| `decode_stored` / `decode_fixed` / `decode_dynamic` | L767-998 | own implementations of RFC 1951 §3.2.4/§3.2.6/§3.2.7 |
| `decode_huffman_block_isal` | L1005-1083 | port of `Block::readInternalCompressed` using ISAL Huffman + ConsumeFirstTable distance |
| `decode_dynamic_block_full_isal` | L1093+ | port using ISA-L for both lit/len and distance |
| `decode_huffman_block` | L1207-1293 | pure-Rust fallback |
| `emit_match` | L1294+ | gzippy-original; implements the marker convention from B1 |

### `src/decompress/parallel/replace_markers.rs` (249 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `MARKER_BASE = 32768` | L24 | gzippy-original encoding (B1) |
| `replace_markers` dispatcher | L30-46 | gzippy-original |
| `replace_markers_scalar` | L49-58 | newest-byte-indexed (differs from `MapMarkers`) |
| `replace_markers_avx2` | L62-109 | **Extra** |
| `replace_markers_neon` | L112-156 | **Extra** |
| `u16_to_u8` | L161-166 | gzippy-original |

### `src/decompress/parallel/trace.rs` (143 lines)

Entirely gzippy-original. JSON-lines event log keyed off `GZIPPY_LOG_FILE`.

### `src/decompress/parallel/block_fetcher.rs` (323 lines) — **WIRED** (post 2026-05-18)

Literal port of `core/BlockFetcher.hpp:41-688`. Used by
`chunk_fetcher.rs:225` as `Arc<BlockFetcher<usize, Arc<ChunkData>,
FetchNextAdaptive, ChunkDecodeError>>` to back the consumer-side
cache + prefetch dispatch.

### `src/decompress/parallel/block_map.rs` (317 lines) — **WIRED** (post 2026-05-18)

`append_subchunks_to_block_map(map, chunk)` is called inside the
consumer loop at `chunk_fetcher.rs:579` after each chunk's
post-process completes. Mirrors `GzipChunkFetcher.hpp:373` (push per
subchunk into the block map).

### `src/decompress/parallel/cache.rs` (289 lines) — **WIRED** (via block_fetcher)

LRU `Cache<K,V,Strategy>` + `LeastRecentlyUsed` + `CacheStatistics`.
Reachable from production via `BlockFetcher::new` composing the
main + prefetch caches.

### `src/decompress/parallel/prefetcher.rs` (283 lines) — **WIRED** (via block_fetcher)

`FetchNextAdaptive` is the strategy plugged into `BlockFetcher` at
`chunk_fetcher.rs:229`. Faithful to `core/Prefetcher.hpp`.

### `src/decompress/parallel/statistics.rs` (350 lines) — **WIRED** (record-only)

`ChunkFetcherStatistics` lives on `BlockFetcher` and is updated from
the consumer (e.g. `chunk_fetcher.rs:588` `record_get`). Not yet
exported via `--verbose` output — open task (C15).

### `src/decompress/parallel/compressed_vector.rs` (227 lines) — **WIRED** (post 2026-05-18)

`CompressedVector` now backs `WindowMap`'s entry storage
(`window_map.rs:39`). Single-member production uses
`CompressionType::None` (`chunk_fetcher.rs:208-210`) for the
single-pass case where compression is pure overhead. `raw_bytes()`
accessor (added in commit `7c524da`) lets `WindowMap::get` skip the
`decompress()` Vec clone on the None path.

### `src/decompress/parallel/deflate_block.rs` (~2400 lines, post-consolidation) — **WIRED** (slow-path bootstrap)

Faithful port of `deflate::Block` with the full vendor ring buffer
(marker pre-init, contains_marker_bytes flag, mid-decode mode switch,
const-generic split, AVX2 narrow path, emit_backref via copy_within,
RLE memset fast path). Wired into `gzip_chunk.rs::
bootstrap_with_deflate_block` via a thread-local recycled `Block`
(commit `17fd9b2`). The fast path (chunk with predecessor window) still
uses ISA-L set_dict (substitution of a faster primitive per CLAUDE.md
Rule 6).

### `src/decompress/parallel/gzip_format.rs` (260 lines)

Port of `gzip/gzip.hpp` header + footer parsing.
`read_header` is called from `single_member::skip_gzip_header`
(used in prod). `read_footer` is **not called from prod** — the
inline trailer parse at `single_member.rs:85-98` reads raw bytes.

### `crates/isal-sys-patched/`

Vendored ISA-L with rapidgzip-style patches exposing
`isal_internals::{huff_code, make_inflate_huff_code_lit_len, ...}`
through FFI — matches rapidgzip's `<igzip_lib.h>` patches.

---

## F. The libdeflate-fallback surface

This is the invariant violation. Every fallback that bypasses or
silently abandons the parallel marker pipeline.

### F1. Gate-bypass — `decompress/mod.rs:218-222`

```
if crate::backends::isal_decompress::is_available()
    && num_threads > 1
    && data.len() > MIN_PARALLEL_COMPRESSED   // 10 MiB
{
```

If the gate is false (no ISA-L, T==1, or compressed ≤ 10 MiB), the
parallel path is never entered and control falls through to
`isal_decompress::decompress_gzip_stream` (L241) or
`decompress_single_member_libdeflate` (L259). **No log, no error, no
counter.** This is the case the bench warning catches because no
`MARKER_PIPELINE_RUNS` increment happens.

**Violation**: the user's directive is that failure must be explicit.
Today, "gate didn't fire" is indistinguishable from "ran and succeeded
via libdeflate". The benchmark numbers reflect libdeflate, not the
parallel path, and there is nothing in-process that flags it.

### F2. `ParallelError::TooSmall` re-fallback — `decompress/mod.rs:232-237,262-273`

```rust
Err(e) => {
    if let Some(err) = map_parallel_single_member_error(e) {
        return Err(err);
    }
    // map returned None → falls through to libdeflate
}
```

`map_parallel_single_member_error` returns `None` for `TooSmall`,
which means the parallel-path call **silently continues** to the
sequential block at L240-259. The `TooSmall` error is emitted by
`decompress_parallel` at `single_member.rs:75,80` whenever the
deflate region or thread count don't meet the parallel preconditions.
Inside the production path this should be impossible because the
outer gate already checked size and thread count — but the safety net
exists and the wire forwards to libdeflate.

**Violation**: same as F1. Soft-routing decision, not a hard error.

### F3. ISA-L sequential → libdeflate — `decompress/mod.rs:240-252`

```rust
if crate::backends::isal_decompress::is_available() {
    if let Some(bytes) = crate::backends::isal_decompress::decompress_gzip_stream(...) {
        ...
        return Ok(bytes);
    }
    // ISA-L returned None → falls through to libdeflate
}
```

`isal_decompress::decompress_gzip_stream` returning `None`
(decompression failed for any reason) silently moves to libdeflate at
L259. Emits a debug eprintln only.

**Violation**: same pattern. Implicit failover masks bugs.

### F4. `decompress_gzip_to_vec` parallel multi-member fallback — `decompress/mod.rs:188-200`

```rust
Err(_) => {
    let mut out = Vec::new();
    decompress_multi_member_sequential(data, &mut out)?;
    ...
}
```

If `decompress_multi_member_parallel_to_vec` errors, it silently falls
back to the sequential multi-member path. Different code path (not the
single-member parallel SM path under review), but same anti-pattern.

### F5. `decompress_gzip_libdeflate` `MultiMemberPar` failover — `decompress/mod.rs:162-167`

Same pattern as F4 for the writer-output multi-member case.

### Summary

Production paths under `decompress::decompress_single_member` that
must die per the no-fallback invariant: **F1, F2, F3**. Multi-member
fallbacks (F4, F5) are out of scope for the parallel-single-member
cutover but follow the same anti-pattern and should be addressed
similarly later.

The right answer in every case: convert silent fallback into a hard
`Err(GzippyError::Decompression(_))` and let the caller decide. The
parallel pipeline either runs and verifies CRC + ISIZE, or it
returns an error. No third path.

---

## G. The bench / test routing-trap

### Tests that exist

`src/tests/routing.rs` has:

- **`test_marker_pipeline_actually_runs_on_x86_64_isal`** (L298-344)
  — snapshots `MARKER_PIPELINE_RUNS` around a
  `decompress_single_member(T=4)` call on a 24 MiB low-entropy
  fixture. Asserts counter incremented.
  **Catches F2.** Does NOT catch F1 (gate bypass — fixture is
  engineered to clear the gate). Does NOT catch F3 (ISA-L did not
  fail on the fixture).
- **`test_marker_pipeline_runs_on_btype01_heavy_input`** (L492-525)
  — same shape against a BTYPE=01-heavy fixture. Same coverage as
  above.
- **`test_single_member_routing_multithread`** (L259-276) — asserts
  byte-correctness of decompress at T=4, no counter check.
  **Does not detect silent fallback** (libdeflate produces the same
  bytes).
- **`test_single_member_parallel_not_slower_than_sequential`** (L365-423)
  — best-of-3 wall-clock. Two-tier threshold (1.5× on ≥4 physical
  cores, 3.0× on <4). Would CATCH F1 implicitly on a small enough
  fixture that hits the gate, because it gates by size and thread
  count just like prod.

### Test that the user wants

A test that fires on **every** silent fallback path:

> "When `decompress_single_member` is called with conditions that
> SHOULD route to the parallel path (per the documented routing
> table), the parallel path actually runs end-to-end, AND any failure
> is a hard error visible to the caller."

This test must:
1. Construct an input that satisfies the gate.
2. Snapshot `MARKER_PIPELINE_RUNS`.
3. Run `decompress_single_member`.
4. Assert (counter increased) **AND** (no error).
5. Repeat for a deliberately-broken input and assert the error is
   `GzippyError::Decompression(_)`, not silently `Ok(...)` from
   libdeflate.

Step 5 is the missing rung. It would have caught the silent fallback
the day the gate logic was added.

---

## H. The unused-port inventory (sober count)

Lines of "rapidgzip ports that no production code calls":

| File | Lines | Production caller |
|------|-------|-------------------|
| `block_fetcher.rs` | 323 | none |
| `block_map.rs` | 317 | none |
| `cache.rs` | 289 | none |
| `prefetcher.rs` | 283 | none |
| `statistics.rs` | 350 | none |
| `compressed_vector.rs` | 219 | none |
| `deflate_block.rs` | 964 | none |
| `gzip_format.rs::read_footer` (~50 lines) | ~50 | none |
| `inflate_wrapper.rs::read_footer_at_current` / `reset_for_next_stream` (~40 lines) | ~40 | none |
| `chunk_data.rs::clean_unmarked_data` + `set_encoded_offset` (~80 lines) | ~80 | `set_encoded_offset` IS called from `chunk_fetcher.rs:527`; `clean_unmarked_data` is unused |
| `gzip_chunk.rs::decode_chunk_with_inflate_wrapper` (38 lines) | 38 | none (tests only) |

**Total unused port code: ~2900 LOC.** Living next to ~2900 LOC of
production code (`fast_marker_inflate.rs` + `chunk_fetcher.rs` body +
`gzip_chunk.rs::finish_decode_chunk_with_inexact_offset`) that
implements the same functionality with different shape and a
silent-fallback safety net underneath.

---

## I. Single-commit cutover plan

The user's directive: **one commit, no phases, no half-measures.**
This section is the implementation spec.

### Goal

After this commit:

- `decompress_single_member` routes single-stream gzip (>some
  minimum) to the rapidgzip-shaped parallel path **only**. No
  ISA-L-stream fallback, no libdeflate fallback. Failure = hard error.
- `fast_marker_inflate.rs` is deleted. The bootstrap is `deflate_block.rs`.
- `chunk_fetcher.rs::drive`'s ad-hoc spec[] + mpsc ring is deleted.
  Dispatch goes through `block_fetcher.rs` + `block_map.rs` +
  `prefetcher.rs` + `cache.rs` + `statistics.rs`.
- `worker_loop` reads gzip headers + footers via
  `IsalInflateWrapper::read_footer_at_current` +
  `reset_for_next_stream` (the multi-stream Footer loop, Step 11
  wired).
- Window storage in `WindowMap` is `Arc<CompressedVector>`.
- Marker encoding flipped to match `MapMarkers`
  (`replace_markers.rs::MARKER_BASE` = `MAX_WINDOW_SIZE` = 32768,
  indexing from **oldest** byte; SIMD kernels updated).
- `chunk_data::clean_unmarked_data` runs in `finalize`.
- Chunk re-anchoring uses `ChunkData::split` + `set_encoded_offset`
  semantics on the consumer.
- Header bracketing is `gzip_format::read_header` /
  `read_footer`. No raw trailer slicing in `single_member.rs`.
- A new test fails if libdeflate is invoked from the SM path
  for any reason.

### Files deleted

- `src/decompress/parallel/fast_marker_inflate.rs` (2356 lines)
- `src/decompress/parallel/inflate_wrapper.rs` `// no buffer refill`
  variant of `read_stream` if a BitReader-backed variant lands —
  otherwise unchanged but bound into the multi-stream loop.

### Files rewritten

- `src/decompress/mod.rs` — `decompress_single_member` becomes a
  single dispatch: classify → parallel-SM. Remove the ISA-L stream
  branch (L240-252) and the libdeflate branch (L259) **from
  single-member**. Remove `map_parallel_single_member_error`
  (L262-287); propagate the error directly. Streaming `>1 GiB`
  branch stays only for non-x86_64 (where ISA-L is absent) as a
  documented Tmax=1 path — that decision still belongs to the
  classifier, not to a silent fallback in the body.
- `src/decompress/parallel/single_member.rs` —
  `decompress_parallel` reads header via `gzip_format::read_header`
  (already does), but stops slicing the trailer inline; instead
  passes the whole gzip input to `chunk_fetcher::drive`, which
  reads footer(s) via the inflate wrapper. Remove `ParallelError::TooSmall`
  as a non-error variant; replace with the actual reason (`InvalidGzipFormat`,
  `InputTooShortForGzip`, etc.).
- `src/decompress/parallel/chunk_fetcher.rs` — `drive`
  constructs `BlockMap`, `BlockFetcher<usize, Arc<ChunkData>,
  FetchNextAdaptive>`, `WindowMap`, `ChunkFetcherStatistics`.
  Replace `consumer_loop`'s `spec: Vec<Option<Receiver>>` with
  `BlockFetcher::get_if_available` / `record_fetch` /
  `prefetch_indexes` flow. `worker_loop` invokes `BlockFetcher::insert_prefetched`
  on completion; `consumer_loop` calls `BlockFetcher::insert` for
  on-demand fetches.
- `src/decompress/parallel/gzip_chunk.rs` —
  `finish_decode_chunk_with_inexact_offset` replaces the
  `fast_marker_inflate::decode_chunk_bootstrap` call site with a
  `deflate_block::Block` instance: instantiate, `read_header`,
  `read` in a loop driving a 2×32 KiB `m_window16`-equivalent
  circular buffer (port the buffer if not already in
  `deflate_block.rs`). On hand-off threshold (32 KiB cumulative
  clean output via the rapidgzip cleanDataCount rule), seed
  `IsalInflateWrapper::set_window` and continue with ISA-L.
- `src/decompress/parallel/replace_markers.rs` — flip
  `MARKER_BASE` to `32768` matching MapMarkers, change indexing
  to `window[v - MARKER_BASE]` (oldest byte), update AVX2 + NEON
  kernels, update `fast_marker_inflate::emit_match` (but
  `fast_marker_inflate.rs` is being deleted, so update
  `deflate_block::emit_backref` instead).
- `src/decompress/parallel/window_map.rs` — `Arc<[u8;32768]>` →
  `Arc<CompressedVector>`. `get_or_wait` decompresses on
  retrieval. Default compression type: `CompressionType::Zlib`
  (rapidgzip default).
- `src/decompress/parallel/chunk_data.rs` — call
  `clean_unmarked_data` from `finalize`. Use `split` semantics for
  subchunk emission rather than always-push.
- `src/decompress/parallel/inflate_wrapper.rs` — wire
  `read_footer_at_current` + `reset_for_next_stream` into the
  `read_stream` multi-stream loop (or expose them so
  `worker_loop` can drive the loop). Stream end produces an
  `Option<Footer>` per stream, accumulated into the chunk.

### New wiring

- `chunk_fetcher::drive` constructs:
  ```
  let stats = ChunkFetcherStatistics::new(pool_size);
  let strategy = FetchNextAdaptive::new(64);
  let fetcher: BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive> =
      BlockFetcher::new(cache_capacity, prefetch_capacity, strategy, pool_size);
  let block_map = BlockMap::new();
  let window_map = WindowMap::new();
  ```
- The consumer loop becomes a port of
  `GzipChunkFetcher::processNextChunk`
  (`vendor/.../GzipChunkFetcher.hpp:311-362`): consult
  `BlockMap::get_block_info(decoded_offset)` (random-access not
  needed yet, but the structure is in place), or use
  `partition_offsets[idx]` for the next prefetch; ask
  `BlockFetcher::get_if_available(block_offset)`; if miss,
  dispatch on-demand; on completion insert via
  `BlockFetcher::insert`, push subchunks via
  `block_map::append_subchunks_to_block_map`, publish per-subchunk
  windows into `WindowMap` (already wired).

### Integration order within the commit

The implementer should write the commit in this order so they can
unit-test as they go, but **everything ships in one commit**:

1. **State machine first** — flip marker encoding in `replace_markers.rs`
   (and AVX2 + NEON kernels), update `deflate_block::emit_backref` to
   match. Verify `deflate_block.rs`'s in-file tests still pass.
2. **Decoder loop next** — port the missing pieces of `deflate::Block`
   (`setInitialWindow`, the `m_window16` circular buffer if missing,
   block-level statistics shells). Update `gzip_chunk::finish_decode_chunk_with_inexact_offset`
   to drive `deflate_block::Block` through a multi-block loop with
   cumulative-clean handoff to `IsalInflateWrapper::set_window`. Add
   the `set_initial_window` path so once we have ≥ 32 KiB clean
   output, future blocks emit real bytes (not markers).
3. **Multi-stream Footer** — wire `IsalInflateWrapper::read_footer_at_current`
   + `reset_for_next_stream` into the wrapper's `read_stream` (or into
   `gzip_chunk`'s decode functions). On END_OF_STREAM, accumulate
   footer to `ChunkData::footers`, push a fresh CRC32 to
   `ChunkData::crc32s`, call `gzip_format::read_header` on the next
   bytes, continue.
4. **Consumer rewrite last** — gut `chunk_fetcher::consumer_loop` and
   replace with the `BlockFetcher`-backed flow. Wire `BlockMap::push`
   via `append_subchunks_to_block_map`. Wrap windows in
   `Arc<CompressedVector>` end-to-end (`window_map`, the publish
   sites in `worker_loop` L353 and `consumer_loop` L625 / L631).
   Delete `fast_marker_inflate.rs`. Delete the
   `chunk_fetcher.rs::decode_or_iterate` BlockFinder loop (it stays
   inside `BlockFetcher::get`'s on-demand path now).
5. **Routing cleanup** — in `decompress::decompress_single_member`,
   delete L240-260 entirely. Replace with:
   ```rust
   // Sole single-member path. The parallel pipeline runs and verifies
   // CRC/ISIZE — or returns an error. No fallback.
   parallel::single_member::decompress_parallel(data, writer, num_threads)
       .map_err(|e| GzippyError::decompression(format!("parallel SM: {e}")))
   ```
   Delete `map_parallel_single_member_error`. Delete
   `ParallelError::TooSmall` (or repurpose to "input not single-member
   gzip" — but that case is unreachable here because
   `classify_gzip` already filtered).
6. **Tests added** — see next section.

### Tests the cutover must keep passing (existing)

These tests currently exist and must remain green after the cutover:

- `src/tests/routing.rs::test_file_oracle_roundtrip` (L110)
- `src/tests/routing.rs::test_detect_bgzf` (L138)
- `src/tests/routing.rs::test_detect_multi_member` (L157)
- `src/tests/routing.rs::test_bgzf_path_correctness` (L178)
- `src/tests/routing.rs::test_multi_member_path_correctness` (L201)
- `src/tests/routing.rs::test_single_member_path_correctness` (L231)
- `src/tests/routing.rs::test_single_member_routing_multithread` (L258)
- `src/tests/routing.rs::test_marker_pipeline_actually_runs_on_x86_64_isal` (L298)
- `src/tests/routing.rs::test_single_member_parallel_not_slower_than_sequential` (L365)
- `src/tests/routing.rs::test_marker_pipeline_runs_on_btype01_heavy_input` (L492)
- `src/tests/routing.rs::test_bgzf_thread_independence` (L546)
- `src/tests/routing.rs::test_cross_format_output_identity` (L579)
- `src/tests/routing.rs::test_classify_*` (L622-670)
- `src/decompress/mod.rs::tests::test_parallel_single_member_only_too_small_is_routing` (L450)
  — this test must be deleted or rewritten since `TooSmall` no
  longer flows through `map_parallel_single_member_error` (that
  function is being deleted).
- `src/decompress/mod.rs::tests::test_decompress_multi_member_file` (L478)
- `src/decompress/parallel/single_member.rs::tests::*` (L191-213) —
  the `single_thread_returns_too_small` test must be updated since
  the gate decision moves to the routing layer.
- `src/decompress/parallel/chunk_fetcher.rs::tests::drive_round_trips_2mb_level6` (L726)
- `src/decompress/parallel/chunk_fetcher.rs::tests::drive_round_trips_8mb_level9` (L741)
- All in-file tests for `block_finder.rs`, `block_fetcher.rs`,
  `block_map.rs`, `cache.rs`, `prefetcher.rs`, `chunk_data.rs`,
  `deflate_block.rs`, `compressed_vector.rs`, `gzip_format.rs`,
  `inflate_wrapper.rs`, `replace_markers.rs` (post-encoding-flip).

### Tests required (new)

1. **`test_no_libdeflate_fallback_ever_fires_from_sm_path`** — a
   test-only `AtomicU64` counter wrapping calls to
   `decompress_single_member_libdeflate` and
   `isal_decompress::decompress_gzip_stream` from the SM path.
   Snapshot before / after a series of single-member decodes that
   should route to parallel. Assert delta == 0. After the cutover,
   the only way for the counter to increment from the SM path is a
   bug.

2. **`test_parallel_sm_propagates_errors_not_fallbacks`** —
   construct a single-member gzip whose CRC trailer is wrong.
   Assert `decompress_single_member` returns
   `Err(GzippyError::Decompression(_))`. Today, depending on which
   variant of corruption, gzippy might silently libdeflate-decode
   (which detects CRC) and return Err — but it might also produce
   inconsistent output. Lock it down.

3. **`test_parallel_sm_routes_below_10mib`** — after the cutover,
   the 10 MiB gate at L221 either stays (justified by perf) or
   leaves (no gate). If it stays, write a test for an 8 MiB
   compressed gzip that asserts a specific known-routing decision
   — either it goes parallel anyway, or it goes through the
   single-thread variant of the parallel pipeline (T=1). Document
   the choice; don't keep "below 10 MiB → libdeflate" as a
   silent fork.

4. **`test_block_fetcher_in_drive`** — assert the production
   `drive` path constructs and uses a `BlockFetcher`. Smoke test
   that `ChunkFetcherStatistics::record_get` was called at least
   once after a decode.

5. **`test_window_map_uses_compressed_vector`** — type-level: the
   `WindowMap` field type is `Arc<CompressedVector>` not
   `Arc<[u8;32768]>`. Compile-time fence.

6. **`test_multi_stream_in_parallel_sm`** — feed a multi-member gzip
   directly to the parallel SM path (bypass routing) and assert
   byte-correct output. Today this can't work; after the cutover
   the multi-stream Footer loop should handle it.

### Acceptance

The cutover is complete when:

- `make` is green.
- `make ship` shows non-trivial throughput on
  `single-member-large.gz` and the bench script does NOT emit the
  `[SILENT FALLBACK]` warning.
- `MARKER_PIPELINE_RUNS` (or a renamed `PARALLEL_SM_RUNS`) is the
  ONLY way `decompress_single_member` produces output on
  parallel-eligible input.
- `fast_marker_inflate.rs` is gone from `git ls-files`.
- The unused-port LOC tally in §H drops to zero on
  `cache.rs` / `block_fetcher.rs` / `block_map.rs` / `prefetcher.rs`
  / `statistics.rs` / `compressed_vector.rs` / `deflate_block.rs`.

### Pre-commit judgment-call checklist

Before the implementer commits:

- [ ] No file in `src/decompress/parallel/` is `pub mod`-exported but
      uncalled.
- [ ] `grep -rn 'libdeflate' src/decompress/parallel/` returns only
      doc comments, never code paths.
- [ ] `decompress_single_member` body has no `if-let-Some-else-fall-through`
      pattern.
- [ ] No `Err(_) => decompress_*` arms.
- [ ] All new tests above are present and green.
- [ ] `GZIPPY_DEBUG=1 gzippy -d -c large.gz > /dev/null` shows
      `path=IsalSingle` or the new equivalent, and the per-chunk
      trace lines fire.

---

## J. Cross-cutting notes

- Multi-member gzip currently bypasses the parallel SM module
  entirely via `classify_gzip` → `MultiMemberPar`/`MultiMemberSeq`
  (`decompress/mod.rs:76-82`). After Footer-loop wiring lands in the
  cutover, the parallel SM path *could* handle multi-member —
  consolidating the routes is a follow-up, NOT part of this
  one-commit cutover. The cutover only owns the single-member path.

- CRC32 / ISIZE mismatch at the trailer remains terminal corruption.
  The cutover does not change that; it only removes the silent
  libdeflate retry above the trailer-check level.

- Streaming-write trade-off: bytes flow to the writer as each chunk
  resolves, so a late CRC/ISIZE mismatch leaves partial bytes
  written. This is documented at `single_member.rs:17-21`. Out of
  scope for the cutover; the rapidgzip parent path has the same
  property.

- The `block_finder.rs` deviations (B7, B8) are not part of the
  cutover. They predate it and don't affect the
  `BlockFetcher`/`BlockMap` wiring.

- Naming: rapidgzip uses `encodedOffsetInBits` / `decodedOffsetInBytes`;
  gzippy uses `encoded_offset_bits` / `decoded_offset`. Style only.
