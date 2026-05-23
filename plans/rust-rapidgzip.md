# Rapidgzip → gzippy parallel single-member: architectural completion

*Last verified against working tree as of May 2026 — uncommitted parallel-SM slice.*

**Scope.** `src/decompress/parallel/` and the surfaces it touches. One
structural port of rapidgzip's `ParallelGzipReader → GzipChunkFetcher →
BlockFetcher → GzipChunk → IsalInflateWrapper` chain. GNU-gzip family
only; seekable / BGZF / multi-member entry points are out of scope.

**Frame.** Not a list of optimisations. The completion of a system the
C++ already finished. Each item is a missing-or-divergent vendor file
with a named gzippy module that owns its closure. Items stage across
commits; the **north-star validation gate** (ratio `< 0.5`, gate 5
`perf stat`) fires when the full shape lands. The May 2026 slice is an
**interim milestone**: production wiring + deletion traps + `< 1.0`
perf wall — not a claim that gates (1a)–(5) are all met.

**Motivating signal.** `tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
historically reported ratio = 1.72 on `neurotic` (16 physical cores,
x86_64, ISA-L): T=4 was *slower* than T=1 on a 24 MiB low-entropy
fixture. Perf-guard gates 1a/1b in `routing.rs` are now tightened to
`ratio < 1.0` at T = `min(16, num_cpus::get_physical())` (x86_64 +
`isal-compression` only). North-star target stays `ratio < 0.5`;
neurotic re-measurement pending (gates 1a/1b final tightening + gate 5).

**Incompleteness marker.** Every parallel module carries
`#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are
pending consumer-port`. When the system lands the allows go; every
exported symbol has a production caller; CI's `-D warnings` succeeds
without them.

## What is already faithfully ported (do not touch)

- **ISA-L stopping-point wrapper.** `inflate_wrapper.rs` = vendor
  `gzip/isal.hpp:26-322` (`IsalInflateWrapper`). Patched `mxmlnkn/isa-l`
  vendored byte-identically. Hot path on x86_64 already lands here.
- **`ThreadPool`.** `thread_pool.rs` = `core/ThreadPool.hpp:33-248`:
  on-demand spawn, `BTreeMap<priority, VecDeque<Task>>` mirroring C++
  `std::map`, `m_idleThreadCount` gating, `core_affinity::set_for_current`
  pinning hook, `Future<T>` over `mpsc::sync_channel(1)` =
  `std::future`/`std::packaged_task`. `submit(closure, priority)` is
  the only dispatch primitive in use.
- **`FetchNextFixed` + `FetchNextAdaptive`.** `prefetcher.rs:38-211` =
  `core/Prefetcher.hpp:60-225` line-for-line, including
  `extrapolateForward` (vendor 127–146) and `splitIndex` (vendor
  198–219).
- **`BlockFetcher` skeleton.** `block_fetcher.rs` = `core/BlockFetcher.hpp:38-687`:
  two-cache (primary + prefetch) with hit-promote; `m_prefetching`
  holding `mpsc::Receiver<Result<V, E>>` in place of `std::map<size_t,
  std::future<BlockData>>` (declared at vendor 685); `prefetchNewBlocks`
  (vendor 459) driven by the consumer's `prefetch_submit` closure
  (`chunk_fetcher.rs:576-594`).
- **`GzipBlockFinder`.** `gzip_block_finder.rs` = `GzipBlockFinder.hpp:34-307`
  minus the FileReader/BGZF branches. Partition guesses via `get(idx)`,
  confirmed boundaries via `insert(off)`.
- **Dynamic-Huffman finder.** `block_finder.rs` ports
  `blockfinder/DynamicHuffman.hpp:39-225` (15-bit LUT at vendor 145,
  `OPTIMAL_NEXT_DEFLATE_LUT_SIZE`) plus
  `blockfinder/precodecheck/CountAllocatedLeaves.hpp`.
- **Marker bootstrap then ISA-L.** `gzip_chunk.rs:288
  decode_chunk_marker_bootstrap_then_isal` = tail of
  `chunkdecoding/GzipChunk.hpp:468-654` (handoff at vendor 520–525).
  The "every candidate fails InvalidHuffmanCode" claim in older
  doc-comments is stale; `SLOW_PATH_FIRST_CANDIDATE_OK`
  (`chunk_fetcher.rs:1263`) increments on real fixtures.
- **Post-process priority.** `chunk_fetcher.rs:1059
  submit_post_process_to_pool` (callsite at `:922`) runs `applyWindow`
  via `ThreadPool::submit(..., -1)`, mirroring
  `GzipChunkFetcher.hpp:554-583 queueChunkForPostProcessing` and
  `BlockFetcher.hpp:608-611 submitTaskWithHighPriority`.

The diff against current code does not move any of the above.

## Routing and sizing facts

Three size thresholds frame the motivating signal and the gap items
below.

| Gate | Location | Value |
|---|---|---|
| Classifier parallel-SM eligibility | `decompress/mod.rs:81 MIN_PARALLEL_COMPRESSED` | compressed ≥ **10 MiB** |
| Internal `decompress_parallel` floor | `single_member.rs:32 MIN_PARALLEL_SIZE` | deflate body ≥ **4 MiB** |
| Perf-guard fixture | `routing.rs:453` | 24 MiB raw → ~12 MiB gzip |

`adjusted_chunk_size_bytes` (`single_member.rs:66-79`) shrinks the
default 4 MiB chunk when `file_size < 2 × default × parallelization`;
on the 24 MiB perf fixture at T = 4 this still yields only ~3 chunks,
so the motivating ratio = 1.72 is partly under-partition pressure on
top of any page-fault cost. *Per-thread allocator arena* (below) targets
the page-fault gap; slow-path boundary search now overlaps finder + trial
decode via `StreamedResults` (see Completed).

## Completed (May 2026)

This slice ports four vendor primitives and re-wires the production slow
path. Deletion-trap tests pass; 186 parallel tests green; clippy clean.

### `StreamedResults` substrate — DONE

`streamed_results.rs` ports `core/StreamedResults.hpp:27-158`:
`VecDeque<V>` + `Mutex` + `Condvar`, `push`/`finalize` notify_all,
`get(idx, timeout)` with `wait_timeout`. The `(Some(value), Failure)`
finalised-past-EOF asymmetry is preserved (vendor `GzipBlockFinder.hpp:144-157`).

### Async raw-finder coordinator — DONE (scoped variant)

`raw_block_finder.rs` ports `core/BlockFinder<RawFinder>`
(`BlockFinder.hpp:202`) as `RawBlockFinderCoordinator`. Production entry:
`with_scoped_boundary_search` — `thread::scope` zero-copy `&[u8]`,
`AtomicBool` cancel flag, 8 KiB-bit sequential scan windows feeding
`StreamedResults<usize>`. First successful trial-decode cancels the
producer. *Residual:* vendor keeps one long-lived finder thread; we
spawn/join per slow-path call until profiling says otherwise.

### Production slow-path wiring — DONE

`speculative_decode_find_boundary` (`chunk_fetcher.rs`) drives
`RawBlockFinderCoordinator::with_scoped_boundary_search` (100 µs poll,
matching `BlockFetcher.hpp:518`). Fan-out `find_blocks_parallel` was
**deliberately not wired** — see Architectural decisions. Deletion trap:
`COORDINATOR_BOUNDARY_SEARCH_RUNS` +
`test_coordinator_boundary_search_runs_on_x86_64_isal`.

### Thread affinity — DONE

`ThreadPool::with_pinning_for_capacity(n)` (`thread_pool.rs`) +
`chunk_fetcher.rs:324` call site. `pool_size.min(num_cpus::get_physical())`.

### Per-worker LIFO buffer pool — PARTIAL (not the full arena)

`chunk_buffer_pool.rs` keys u8/u16 reuse per worker (`bind_worker_pool_index`
in `worker_main` after pinning), `MAX_POOLED = 8` per worker. Closes
cross-thread pool thrash; pool misses still hit `std::alloc::System`.
True `allocator-api2` arena remains open.

### Opportunistic prefetch promotion — DONE

`process_ready_prefetches` (`block_fetcher.rs:696`) called from
`prefetch_new_blocks` and `consumer_loop` outer iteration.

### `FetchMultiStream` — PORTED, NOT WIRED

`prefetcher.rs` implements vendor `Prefetcher.hpp:234-336`. Production
still uses `FetchNextAdaptive`.

### Block-finder test fixtures — DONE

Parallel tests use random deflate payloads, not zeros.

### Routing perf gates — TIGHTENED (interim)

Gates 1a/1b assert `ratio < 1.0` at T = `min(16, physical)`, x86-only.
North-star `< 0.5` pending neurotic re-measure.

### Architectural decisions (recorded)

1. **Slow path = single sequential finder + first-candidate-wins
   trial-decode.** Not bulk `find_blocks_parallel` on `ThreadPool`.
2. **No separate `uncompressed_block_finder.rs`.** Stored blocks handled
   in `block_finder.rs:668 find_blocks`.
3. **Per-worker LIFO ≠ rpmalloc arena.** Gate (5) still needs
   `allocator-api2`.
4. **`thread::scope` per slow-path call**, not long-lived finder thread.

## The gap

### Per-thread allocator arena for chunk buffers

**Vendor.** `core/FasterVector.hpp:120-128` aliases `FasterVector<T>` to
`std::vector<T, RpmallocAllocator<T>>`. `RpmallocAllocator` (vendor
72–113) calls `rpmalloc` / `rpfree`; per-thread init is lazy via
`static thread_local RpmallocThreadInit` (vendor 64).
The two `FasterVector` instances that matter are `ChunkData::data`
(`Vec<u8>`) and `ChunkData::data_with_markers` (`Vec<u16>`).

**Now.** `chunk_buffer_pool.rs` keys u8/u16 LIFO pools per worker via
`bind_worker_pool_index` (called from `worker_main` after pinning), 8
buffers max per worker. **Partial step:** removes global Mutex contention
and keeps freed buffers warm on the allocating worker. Does **not**
replace `std::alloc::System` — pool misses still hit glibc /
`libsystem_malloc`, which `munmap`s large frees and remaps fresh pages.
Per the module comment (`chunk_buffer_pool.rs:67-90`): on the silesia
profile neurotic spent ~40% CPU in page faults; rapidgzip ~17%. Per-worker
LIFO partially closes the warm-page gap; the headroom past pool capacity
still needs the allocator work below.

**Work.** Mirror `RpmallocAllocator` via `allocator-api2`
(stable polyfill; std `Vec<T, A>` is unstable). The two `chunk_data.rs`
fields:

```rust
use allocator_api2::vec::Vec as ArenaVec;

pub struct ChunkData {
    pub data_with_markers: ArenaVec<u16, ThreadArenaAlloc>,
    pub data:              ArenaVec<u8,  ThreadArenaAlloc>,
    // …
}
```

`ThreadArenaAlloc` is a `Copy + Send` ZST wrapping `mimalloc` (mature;
preferred) or `rpmalloc-rs` (closer vendor analogue but less
maintained). Per-thread init is lazy, matching vendor 64. The
motivating metric is rapidgzip + rpmalloc
(`LIBRAPIDARCHIVE_WITH_RPMALLOC`); mimalloc is fine for first
implementation, but if validation gate (5)'s `minor-faults` 1.5×
bound is missed, switch to `rpmalloc-rs` before declaring this
pillar complete.
`chunk_buffer_pool.rs` per-worker take/return is already keyed by worker
index (see Completed). The take path should become lock-free per worker
once buffers use `ArenaVec`; cross-thread returns stay supported via
fallback enqueue.

**Modules touched.** `chunk_data.rs` (two fields, ~30 constructor /
accessor call sites), `chunk_buffer_pool.rs` (per-worker-index pools),
every test that constructs `ChunkData` directly. `Cargo.toml` adds
`allocator-api2 = "0.2"` and `mimalloc` behind a feature gate
`arena-allocator`.

**Risks.** `allocator-api2::Vec` does not expose every `std::Vec`
method. The production `extend_from_slice` callsites on the two
buffers are `chunk_data.rs:343` (`append_markered` →
`data_with_markers`), `:361` (`append_clean` → `data`), and `:432`
(`append_owned_buffer` → `data`). Audit those plus every other
`extend` on either field. Do not pre-warm pages at process start — that
experiment was −50% throughput on the bench (`chunk_buffer_pool.rs:68-72`).
The per-thread arena pays its init cost once per worker, amortising
across all chunks that worker handles.

### Thread affinity — DONE

Moved to **Completed (May 2026)**. `ThreadPool::with_pinning_for_capacity`
+ `pool_size.min(num_cpus::get_physical())` at `chunk_fetcher.rs:324`.

### `StreamedResults` + async raw-finder coordinator — DONE

Moved to **Completed (May 2026)**. Residual: long-lived finder thread
(vendor shape) vs scoped per-call — optional follow-up.

### Parallel block-finding on the production slow path — SUPERSEDED

**Status.** Original plan: wire `find_blocks_parallel` fan-out into
`speculative_decode_find_boundary`. **Rejected** May 2026 in favour of
single sequential finder streaming through `StreamedResults`, with
trial-decode overlap and first-candidate-wins cancel.

**Why fan-out was rejected.** Under-partition (~3 chunks at T=4 on 24 MiB)
means a fan-out finder competes with decode workers. Vendor overlaps
finder with decoder, not finder-with-finder. `find_blocks_parallel` stays
as a tested helper with no production caller.

**Optional second layer.** `ParallelBitStringFinder.hpp:35-265` — defer
until gate (5) identifies LUT scan as bottleneck.

### Missing block-finder variants

**`Uncompressed.hpp`** — folded into `block_finder.rs:668 find_blocks`.
No separate module.

**`blockfinder/PigzStringView.hpp:30-179`** — `PigzStringView` detects
pigz's `00 00 ff ff` sync-flush via `std::string_view::find` (vendor
77); 8 GB/s on the boundary search (vendor 22–26 comment).
Pigz-produced inputs are common in the wild; on those, every bit
position without `PigzStringView` runs through the dynamic-Huffman
LUT — ~95% wasted. Port as `pigz_string_view.rs` (~200 lines),
dispatch ahead of the dynamic-Huffman scan; hits short-circuit the
slow path (no trial decode needed).

### Cached-bits Huffman for the marker-bootstrap inner loop

**Vendor.** Three cached variants in `huffman/`:
`HuffmanCodingShortBitsCachedDeflate.hpp:23-280` (deflate
specialisation, 8–12 bit LUT), `HuffmanCodingShortBitsMultiCached.hpp:25-269`
(two symbols per lookup when both fit the cached bits), and
`HuffmanCodingReversedBitsCached.hpp` — used by
`gzip/deflate.hpp:195-196`'s `FixedHuffmanCoding`, which is the table
for BTYPE=01 blocks. The dynamic-Huffman bootstrap uses the ShortBits
variants; the fixed-Huffman bootstrap path (which
`deflate_block.rs:836-857` explicitly bypasses ISA-L for) uses the
Reversed variant.

**Now.** `deflate_block.rs:223` uses `IsalLitLenCode` on the ISA-L
fast path (fine). Non-ISA-L Huffman decode falls through to
`huffman_symbols_per_length.rs` =
`huffman/HuffmanCodingSymbolsPerLength.hpp:30-142`, which walks one
bit per symbol. The marker bootstrap
(`gzip_chunk.rs:409 bootstrap_with_deflate_block`, callsite at
`:311`) runs this *even on x86_64* — it precedes the ISA-L handoff
and decides where the boundary is. The cached variants rapidgzip
uses for the same job are absent.

**Work.** Two new modules:

- `huffman_short_bits_cached.rs` ← vendor 23–280. Const-generic
  `LUT_BITS_COUNT`:

  ```rust
  pub struct HuffmanCodingShortBitsCachedDeflate<const LUT_BITS_COUNT: u8> {
      base: HuffmanCodingSymbolsPerLength<MAX_LITERAL_HUFFMAN_CODE_COUNT>,
      code_cache: [CacheEntry; 1 << LUT_BITS_COUNT],
      bits_to_read_at_once: u8,
  }
  #[derive(Default, Copy, Clone)]
  struct CacheEntry { bits_to_skip: u8, symbol_or_length: u8, distance: u16 }
  ```

  `initialize_from_lengths` = vendor 43–100; `decode(bit_reader)` =
  vendor 120–142.

- `huffman_short_bits_multi_cached.rs` ← vendor 25–269. Same shape,
  cache entries hold two symbols.

- `huffman_reversed_bits_cached.rs` ← `HuffmanCodingReversedBitsCached`
  for the fixed-Huffman BTYPE=01 path. Vendor `deflate.hpp:195-196`
  declares `FixedHuffmanCoding = HuffmanCodingReversedBitsCached<…>`;
  `deflate_block.rs:836-857` currently bypasses ISA-L for fixed
  blocks via the slow `SymbolsPerLength` path, which this closes.

Swap the import in `deflate_block.rs:222-228`. Retain
`huffman_symbols_per_length.rs` as the long-code fallback (codes
exceeding `LUT_BITS_COUNT`).

**Bootstrap decode matrix:**

| Block type | x86 today | Target |
|---|---|---|
| Dynamic (BTYPE=10) | `IsalLitLenCode` / ISA-L | keep |
| Fixed (BTYPE=01) | `HuffmanCodingSymbolsPerLength` (one bit per symbol) | `HuffmanCodingReversedBitsCached` |
| Stored (BTYPE=00) | dynamic finder rejects → tail byte walk | `Uncompressed` finder (above) |

### Opportunistic prefetch promotion — DONE

Moved to **Completed (May 2026)**.

### `FetchMultiStream` — PORTED, NOT WIRED

Moved to **Completed (May 2026)**. Type exists; production uses
`FetchNextAdaptive`.

## Why these land as a unit

Each pair is "X is structurally impossible without Y."

- **Arena ↔ affinity.** Per-thread arenas need worker → physical-core
  stability. Affinity landed May 2026; arena still open.
- **`StreamedResults` ↔ slow-path overlap.** Landed May 2026 — consumer
  trial-decodes while finder streams candidates. Fan-out rejected.
- **Arena ↔ post-process on the worker pool.** `applyWindow` touches
  `data_with_markers`; arena makes pages warm. Still open.
- **Cached-bits Huffman ↔ PigzStringView.** Either alone helps a fraction;
  both together cheapen the slow path.

`FetchMultiStream` ports cleanly on its own — done, not wired.

## Out of scope

- `IndexFileFormat.hpp` and the seekable-reader path. Supporting
  scaffolding — `chunk_data.rs::subchunks`, `block_map.rs`,
  `UnsplitBlocks` at `chunk_fetcher.rs:118` — is plumbed through
  `consumer_loop` (line 462) but currently surfaces only via the
  `GZIPPY_VERBOSE` stats dump (`UNSPLIT_BLOCKS_EMPLACED` at line 408).
  The seekable reader that would consume them is out of scope.
- Python bindings, `rapidgzip.hpp` public API.
- `GzipReader.hpp`, `GzipAnalyzer.hpp` (single-threaded paths route
  through `decompress::mod`, not `parallel/`).
- BZIP2, ZLIB-format decoders (project scope).
- `FasterVector.hpp` custom-storage variant (vendor 159–416, behind
  `#if 1 … #else`) — the active branch is the allocator variant; only
  that needs porting.
- `AlignedAllocator.hpp` — `replace_markers.rs:146` uses unaligned
  AVX loads (`_mm256_loadu_si256`), faithful to `MarkerReplacement.hpp`.
  Re-evaluate only if `dTLB-load-misses` stays elevated after the
  arena lands.

## Remaining work — priority order

With the May 2026 slice landed, open items reduce to five:

1. **Per-thread allocator arena** (`allocator-api2` + mimalloc on
   `ChunkData::data` / `data_with_markers`). Largest commit; unlocks
   gate (5). Switch to `rpmalloc-rs` if gate (5) `minor-faults ≤ 1.5×`
   misses.
2. **Cached-bits Huffman** (`huffman_short_bits_cached.rs`, multi,
   reversed). Closes marker-bootstrap inner loop.
3. **`PigzStringView` finder** — remaining block-finder variant.
4. **Tighten gates 1a/1b to north-star `< 0.5`** after (1)–(3) and
   neurotic re-measure. Lift interim `< 1.0` wall.
5. **Strip `#![allow(dead_code)]`** from hot-path modules (gate 4) +
   capture gate-5 `perf stat` numbers.

Optional follow-ups: long-lived finder thread; wire `FetchMultiStream`
if multi-stream workload appears; `ParallelBitStringFinder` if profiling
identifies LUT scan as bottleneck.

The unit gate is now (4)+(5) — dead-code purge and perf stat. Everything
else either feeds them or is optional.

## Validation gate

All on `neurotic` (16 physical x86_64, ISA-L), driven by
`make test-x86_64`. Local `make` is insufficient for perf gates.

| # | Gate | Status |
|---|---|---|
| 1a | Synthetic perf — `ratio < 0.5` at T = `min(16, physical)` on 24 MiB low-entropy fixture | **PARTIAL** — interim `< 1.0` in `routing.rs`; north-star pending neurotic |
| 1b | Silesia-class perf — same ratio bar | **PARTIAL** — `test_single_member_parallel_silesia_class_not_slower_than_sequential` uses **synthetic** 24 MiB random data + `Compression::best()` (high-entropy class), not `silesia.tar.gz`. Interim `< 1.0`; real corpus + `< 0.5` pending neurotic |
| 2  | `test_single_member_routing_multithread` + `MARKER_PIPELINE_RUNS` | **DONE** |
| 2b | `test_coordinator_boundary_search_runs_on_x86_64_isal` | **DONE** |
| 3  | `test_parallel_sm_handles_fname_header` | **DONE** |
| 4  | Remove `#![allow(dead_code)]` from production SM hot-path modules | **PARTIAL** — `streamed_results`, `raw_block_finder`, `chunk_buffer_pool` module allows removed; modules cfg-gated `x86_64 + isal-compression`. Remaining allows on other hot-path modules until arena/Huffman land |
| 5  | `perf stat` Silesia vs rapidgzip: minor-faults ≤ 1.5×; wall-time ≤ 1.2× | **OPEN** — needs arena |
| 6  | `GZIPPY_VERBOSE` diagnostics sane | **DONE** |

If any of (1a)–(5) miss the **north-star** target, the unit hasn't fully
landed. The May 2026 slice holds the interim `< 1.0` wall on gates
1a/1b only.

## Reading order for picking this up cold

1. `src/decompress/parallel/single_member.rs` — entry,
   `MARKER_PIPELINE_RUNS` deletion trap.
2. `chunk_fetcher.rs:253 drive` → `consumer_loop` →
   `submit_decode_to_pool` → `submit_post_process_to_pool`.
3. Vendor `GzipChunkFetcher.hpp:312 processNextChunk` and
   `BlockFetcher.hpp:245 get`, side-by-side with the Rust.
4. Vendor `ParallelGzipReader.hpp:495 read` for the outer-loop shape.
5. This document, item by item, with the cited vendor line ranges in
   a second window. Every claim above is at vendor `file:line`. Verify
   before changing the corresponding gzippy module.

The vendor citations are the spec.
