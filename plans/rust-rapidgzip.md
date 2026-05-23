# Rapidgzip → gzippy parallel single-member: architectural completion

**Scope.** `src/decompress/parallel/` and the surfaces it touches. One
structural port of rapidgzip's `ParallelGzipReader → GzipChunkFetcher →
BlockFetcher → GzipChunk → IsalInflateWrapper` chain. GNU-gzip family
only; seekable / BGZF / multi-member entry points are out of scope.

**Frame.** Not a list of optimisations. The completion of a system the
C++ already finished. Each item is a missing-or-divergent vendor file
with a named gzippy module that owns its closure. Items may stage
across commits; the validation gate fires once the whole shape lands.

**Motivating signal.** `tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
reports ratio = 1.72 on `neurotic` (16 physical cores, x86_64, ISA-L):
T=4 is *slower* than T=1 on a 24 MiB low-entropy fixture. The test
applies a tight 1.5× threshold on ≥4-core hosts and a relaxed 3.0×
threshold on <4-core hosts (`routing.rs:507`); GHA CI runners are
typically <4 cores, so the failure surfaces only on neurotic via
`make test-x86_64`. End state: ratio < 0.5 on the same host.

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

## The gap

### Per-thread allocator arena for chunk buffers

**Vendor.** `core/FasterVector.hpp:120-128` aliases `FasterVector<T>` to
`std::vector<T, RpmallocAllocator<T>>`. `RpmallocAllocator` (vendor
72–113) calls `rpmalloc` / `rpfree`; per-thread init is lazy via
`static thread_local RpmallocThreadInit` (vendor 64).
The two `FasterVector` instances that matter are `ChunkData::data`
(`Vec<u8>`) and `ChunkData::data_with_markers` (`Vec<u16>`).

**Now.** `chunk_buffer_pool.rs:88-89` is one process-global
`static U8_POOL: Mutex<Vec<Vec<u8>>>` and a `U16_POOL`, capped at 64
(line 86). Misses fall through to `Vec::with_capacity(cap)` →
`std::alloc::System`. glibc / `libsystem_malloc` `munmap` large frees
and `mmap` fresh on next alloc. Per the module comment block
(`chunk_buffer_pool.rs:60-67`): on the silesia profile neurotic spends
~40% CPU in `asm_exc_page_fault` + `clear_page_erms`; rapidgzip spends
~17%. The global `Mutex` LIFO doesn't close it: pages migrate cores
across workers, and pool-miss allocations bypass the arena entirely.

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
maintained). Per-thread init is lazy, matching vendor 64.
`chunk_buffer_pool.rs` is keyed per worker index — `take_u8(worker_id)`
/ `return_u8(worker_id, buf)`. The take path becomes lock-free per
worker; cross-thread returns (the `Arc<ChunkData>` `Drop` may run on a
post-process worker, not the decoder that allocated) stay supported
via a fallback enqueue.

**Modules touched.** `chunk_data.rs` (two fields, ~30 constructor /
accessor call sites), `chunk_buffer_pool.rs` (thread-local), every
test that constructs `ChunkData` directly. `Cargo.toml` adds
`allocator-api2 = "0.2"` and `mimalloc` behind a feature gate
`arena-allocator`.

**Risks.** `allocator-api2::Vec` does not expose every `std::Vec`
method; `extend_from_slice` / `extend` callsites at
`gzip_chunk.rs:650`, `:665`, `:677`, `:690` are supported, but audit
every `extend`. Do not pre-warm pages at process start — that
experiment was −50% throughput on the bench (`chunk_buffer_pool.rs:68-72`).
The per-thread arena pays its init cost once per worker, amortising
across all chunks that worker handles.

### Thread affinity, populated by gzippy itself

**Vendor.** `core/AffinityHelpers.hpp:11` declares
`pinThreadToLogicalCore`; the Linux body is vendor 76–101 (raw
`sched_setaffinity(0, cpuSetSize, pCpuSet)`).
`ThreadPool.hpp:198-200` consults `m_threadPinning` once per worker on
`workerMain` entry.

**Now.** `thread_pool.rs:191` accepts a `ThreadPinning` parameter;
`worker_main:386-395` pins via `core_affinity::set_for_current` when
given one. The sole production construction at
`chunk_fetcher.rs:318` passes `ThreadPinning::new()` — the empty map.
No pin map is built.

**Work.** Add `ThreadPool::with_pinning_for_capacity(n)` in
`thread_pool.rs` that assigns worker `i` to
`core_affinity::get_core_ids()[i % len]`, mirroring vendor's typical
embedder pattern. Switch the `chunk_fetcher.rs:318` callsite. Clamp
`pool_size` to `num_cpus::get_physical()` to match vendor
`availableCores()` semantics (`AffinityHelpers.hpp:18-21` / 104–124)
and avoid SMT-sibling collisions.

**Cross-platform.** Linux: `sched_setaffinity` (vendor parity).
macOS: `thread_policy_set(THREAD_AFFINITY_POLICY)` — Apple's policy
is advisory, no-op on Apple Silicon; benign. The `core_affinity` crate
is already a dep and no-ops where unsupported.

### `StreamedResults` for both block-finder output and ordered drain

**Vendor.** `core/StreamedResults.hpp:27-158`:
`std::deque<Value>` + mutex + condvar. Push-side never blocks; read
blocks on `get(idx, timeout)` until the index is available or
`finalised`. The vendor callsite is `core/BlockFinder<RawFinder>`
(`BlockFinder.hpp:202`): the async raw-finder coordinator queues
candidate offsets through it, and `BlockFetcher::prefetchNewBlocks`
waits on `get(idx, 0.0001 s)` (`BlockFetcher.hpp:518`). (Ordered-drain
post-processing uses a different primitive — vendor's
`std::map<size_t, std::future<void>> PostProcessingFutures` at
`GzipChunkFetcher.hpp:48` — and is *not* a `StreamedResults`
consumer.)

**Now.** gzippy has no `core/BlockFinder<RawFinder>` async-coordinator
port; the raw block-finding (`block_finder.rs::find_blocks`) is
invoked synchronously from `chunk_fetcher.rs:1344`. There is no
condvar, no wait-with-timeout, no async candidate queue. That
missing primitive is what serialises parallel block-finding (next
item) onto the consumer thread. (`gzip_block_finder.rs:65 Inner`'s
`Vec<usize>` in `Mutex<Inner>` is the synchronous `GzipBlockFinder`
shape — faithful to vendor `GzipBlockFinder.hpp:120-157`'s raw deque
+ mutex — and is not the gap.)

**Work.** Add `streamed_results.rs` (~250 lines):

```rust
pub struct StreamedResults<V: Clone> { inner: Arc<Mutex<Inner<V>>>, cv: Arc<Condvar> }
struct Inner<V> { values: VecDeque<V>, finalised: bool }

impl<V: Clone> StreamedResults<V> {
    pub fn push(&self, v: V);                       // notify_all
    pub fn finalise(&self);                         // notify_all
    pub fn get(&self, idx: usize, timeout: Duration)
        -> (Option<V>, GetReturnCode);              // wait_timeout
    pub fn results_view(&self) -> ResultsView<'_, V>; // RAII lock holder, vendor 39–60
}
```

The new consumer is the async raw-finder coordinator that wraps
`block_finder.rs::find_blocks` and emits offsets through
`StreamedResults<usize>`. This is the substrate for parallel
block-finding (next item) and the prefetch wait-with-timeout
(`BlockFetcher.hpp:518`). Preserve the `(Some(offset), Failure)`
"finalised past EOF" asymmetry that `chunk_fetcher.rs:617-621`
(`PREFETCH_NEXT_FILESIZE_ACCEPT`) depends on — same shape as vendor
`GzipBlockFinder.hpp:144-157`.

### Parallel block-finding on the production slow path

**Vendor.** `core/ParallelBitStringFinder.hpp:35-265` parallelises the
bit-string scan via a worker-pool fan-out. `chunkSize` at vendor
91–115 picks the per-task range; offsets stream out through a
`StreamedResults`-shaped queue.

**Now.** `block_finder.rs:1034 find_blocks_parallel` exists — vendored
in shape — but its callers are only tests (lines 1274, 1302, 1314).
The production slow path at `chunk_fetcher.rs:1344` calls
`finder.find_blocks(chunk_begin, chunk_end)` from
`block_finder.rs:668` — single-threaded, on the consumer, inside a
64-iteration loop (`chunk_fetcher.rs:1321-1322`; constants are in
bits: `CHUNK_SIZE_BITS = 65 536` ≈ 8 KiB scanned per iteration,
`MAX_SCAN_BITS = 4 Mib` ≈ 512 KiB total window).

**Work.** Pass `thread_pool: &Arc<ThreadPool>` into
`speculative_decode_find_boundary` (`chunk_fetcher.rs:1315`), fan out
per-sub-chunk scans via `ThreadPool::submit`, join via
`Future::wait`. Activation threshold: `chunkSize` rule from vendor 91
— inputs below ~16 KiB stay single-threaded.

**Why it's part of the unit.** Without the per-thread arena, the
fan-out's N concurrent allocations of LUT scratch fault fresh pages —
the arena work caps that cost. Without `StreamedResults`, the join
collapses to a barrier and the consumer cannot interleave decode of
chunk N with finder work on chunk N+K.

### Missing block-finder variants

**`blockfinder/Uncompressed.hpp:21-95`** —
`seekToNonFinalUncompressedDeflateBlock` finds stored (BTYPE=00) blocks
via the `size ^ (size >> 16) == 0xFFFF` invariant (vendor 56) plus the
preceding 3-magic-bits check (vendor 70). Without it, files with
stored regions (gzip `--stored`, pipelined L9 mid-stream stored
blocks) reject in the dynamic finder, fall through to on-demand
serial decode. Port as `uncompressed_block_finder.rs` (~120 lines),
dispatch in parallel with the dynamic-Huffman scan at
`chunk_fetcher.rs:1315`; first hit wins.

**`blockfinder/PigzStringView.hpp:30-179`** — `PigzStringView` detects
pigz's `00 00 ff ff` sync-flush via `std::string_view::find` (vendor
77); 8 GB/s on the boundary search (vendor 22–26 comment).
Pigz-produced inputs are common in the wild; on those, every bit
position without `PigzStringView` runs through the dynamic-Huffman
LUT — ~95% wasted. Port as `pigz_string_view.rs` (~200 lines),
dispatch ahead of the dynamic-Huffman scan; hits short-circuit the
slow path (no trial decode needed).

Both are necessary because the *rate* of slow-path triggering is high
on input classes gzippy itself emits. The marker-bootstrap bug is
fixed; the *finder coverage* gap remains.

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

### `FetchMultiStream`

**Vendor.** `core/Prefetcher.hpp:234-336` — a `FetchNextAdaptive`
subclass that sorts access history, extrapolates per detected
sub-sequence (`extrapolateSubsequence` at vendor 270–303), interleaves
via `common.hpp::interleave` (vendor 316).

**Now.** `prefetcher.rs:13` claims `FetchMultiStream` exists in the
module preamble, but the type is not defined. The lie is the
deliverable: port vendor 234–336 into `prefetcher.rs` (~150 lines).
Wire is behind a `BlockFetcher` config option. Single-member streaming
keeps `FetchNextAdaptive`; this is the one item that doesn't unlock
another, but the preamble's claim of "Prefetcher port complete" must
be honest.

## Why these land as a unit

Each pair is "X is structurally impossible without Y."

- **Arena ↔ affinity.** Per-thread arenas need worker → physical-core
  stability. Unpinned workers migrate; mimalloc's per-thread heap
  travels with the OS thread but the physical pages don't.
- **`StreamedResults` ↔ parallel block-finding.** The fan-out's join
  needs append-and-wait with timeout; a barrier serialises decode and
  finder work.
- **Arena ↔ post-process on the worker pool.** `applyWindow` (already
  on the pool) touches the entire `data_with_markers` Vec — the arena
  is what makes its pages warm. Without the arena, post-process-on-pool
  buys nothing.
- **Cached-bits Huffman ↔ block-finder variants.** Parallel
  block-finding amortises *finding* candidates; cached-bits Huffman
  amortises *validating* them. Either alone improves a fraction; both
  together make the slow path cheap enough that the consumer never
  goes synchronous.

`FetchMultiStream` is the exception — it ports cleanly on its own. It
is in the unit to keep the module preamble honest, not because of a
structural dependency.

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

## Smallest viable work item

Commits stage independently. The smallest is "land
`streamed_results.rs` plus a tiny async raw-finder coordinator that
wraps `block_finder.rs::find_blocks` and emits offsets through it" —
no perf change on its own, but it is the substrate the
parallel-block-finding item rests on. The largest single commit is
"wire per-thread arena through `chunk_data.rs`" because every
`Vec<u8>`/`Vec<u16>` field changes type. Both are fine. The unit is
the gate, not the merge order.

## Validation gate

All on `neurotic` (16 physical x86_64, ISA-L). Local `make` is
insufficient — the ratio=1.72 regression triggers only at ≥4 physical
cores.

1. `cargo test --release -- test_single_member_parallel_not_slower_than_sequential`
   reports `ratio < 0.5`. Capture printed `seq_mbps`, `par_mbps`,
   `ratio`. Floor for `par_mbps`: ≥2× sequential libdeflate throughput
   on the same fixture (~600 MB/s baseline on neurotic).

2. `cargo test --release -- test_single_member_routing_multithread`
   passes byte-perfect, and `MARKER_PIPELINE_RUNS`
   (`single_member.rs:95`) increments — proves the parallel path took
   the input (no silent libdeflate fall-through).

3. `cargo test --release -- test_parallel_sm_handles_fname_header`
   passes (header-parse / partition-offset interaction).

4. Every `#![allow(dead_code)] // vendor-faithful rapidgzip port`
   removed from `src/decompress/parallel/*.rs`.
   `cargo build --release -- -D warnings` succeeds. The allows are
   the structural signal that exported symbols lack callers; removing
   them is the closure proof.

5. `perf stat -e dTLB-load-misses,major-faults,minor-faults,cycles
   ./target/release/gzippy -d -c silesia.tar.gz > /dev/null`
   (1.4 GB Silesia, `https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia`).
   Compare to `rapidgzip -d -P 16 -c silesia.tar.gz > /dev/null` on
   the same host. Required: gzippy `minor-faults` within 1.5× of
   rapidgzip's; gzippy wall-time within 1.2× of rapidgzip's. Capture
   exact baseline numbers when the first item lands (rapidgzip is
   already installable on neurotic via `pip install rapidgzip`); the
   1.2× / 1.5× bounds are relative to that capture.

6. `GZIPPY_VERBOSE=1 ./target/release/gzippy -d -c <fixture>` (stats
   dump at `chunk_fetcher.rs:384-438`): `Slow-path decode: ok > 0
   and fail < 0.1 × ok`; `Buffer pool u8: misses < 0.05 × hits`;
   `Prefetch guard-rejects` does not grow with input size. (6) is
   diagnostic — use to localise which item regressed if (1)–(5) miss.

If any of (1)–(5) miss, the unit didn't land — a single item's gap
broke a cross-pillar invariant.

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

---

## Appendix: review corrections (integrate into body above)

Second-pass review against vendor sources and current gzippy tree. Opus
should merge these into the main sections and delete this appendix once
done.

### A1. `StreamedResults` — wrong dual callsite (critical)

**Main doc claim (§ StreamedResults):** vendor uses `StreamedResults` for
both ordered drain (`m_markersBeingReplaced`) and `BlockFinder` output.

**Correction:** only one vendor callsite.

| Mechanism | Vendor type | gzippy analogue |
|---|---|---|
| Post-process ordering | `PostProcessingFutures` = `std::map<size_t, std::future<void>>` (`GzipChunkFetcher.hpp:48`, member at `:783`) | `VecDeque<PendingWrite>` + `mpsc::Receiver` (`chunk_fetcher.rs:475`) — **already correct** |
| Async raw-finder scan DB | `StreamedResults<size_t>` inside `core/BlockFinder<RawFinder>` (`BlockFinder.hpp:202`) | **missing** — not `GzipBlockFinder` |
| Partition spacing | `GzipBlockFinder` = `std::deque` + mutex; `get(idx, timeout)` ignores timeout (`GzipBlockFinder.hpp:120-157`) | `gzip_block_finder.rs` — **already correct shape** |

**Do not:** replace `pending: VecDeque<PendingWrite>` with
`StreamedResults<PendingWrite>`. That is not vendor-faithful.

**Do:** port `streamed_results.rs` for a future `core_block_finder.rs`
(async wrapper around pigz/dynamic/uncompressed raw finders) OR for the
slow-path scan coordinator join inside `speculative_decode_find_boundary`.
Wire `get(idx, timeout)` blocking there so prefetch/decode can overlap
finder fan-out — mirror `BlockFetcher.hpp:518` (`0.0001` s timeout on
`m_blockFinder->get`), which applies when the finder is the async
`core/BlockFinder` wrapper, not `GzipBlockFinder`.

**Remove or rewrite** “Smallest viable work item: land `streamed_results.rs`
and consume it in `gzip_block_finder.rs`” — wrong target module.

**Coherence fix:** “StreamedResults ↔ parallel block-finding” is valid for
**async raw-finder join**, not for `GzipBlockFinder` or post-process ordering.

### A2. Parallel scan — reuse `find_blocks_parallel`, don't greenfield

**Main doc claim:** port `ParallelBitStringFinder` into production slow path.

**Correction:** `block_finder.rs:1034` **`find_blocks_parallel`** already
exists (fan-out + merge, `+1024`-bit overlap at `:1065`). Callers today
are **tests only** (`:1244`, `:1273`, `:1298`). Production slow path at
`chunk_fetcher.rs:1344` calls single-threaded `finder.find_blocks`.

**Work should say:** wire `speculative_decode_find_boundary` through existing
`find_blocks_parallel`, rehosted on `Arc<ThreadPool>` (not
`std::thread::scope`) + per-worker buffer pool id. Evaluate whether
`ParallelBitStringFinder.hpp` adds anything beyond what
`find_blocks_parallel` + pigz/uncompressed finders already provide before
porting another scan layer.

### A3. Fixed-Huffman decoder missing from pillar 3

**Main doc:** ports `HuffmanCodingShortBitsCachedDeflate` +
`HuffmanCodingShortBitsMultiCached` only.

**Correction:** on x86, **dynamic** bootstrap already uses ISA-L tables
(`deflate_block.rs:864-878`). **Fixed** blocks explicitly bypass ISA-L and
use canonical decode because ISA-L's fixed-table LUT is incomplete
(`deflate_block.rs:836-857`, cites vendor `deflate.hpp:195-196`). Vendor
uses **`HuffmanCodingReversedBitsCached`** for fixed Huffman, not
ShortBitsCached.

Slow-path tail walk (`chunk_fetcher.rs:1369-1374`) targets fixed/stored
tails. Add to pillar 3:

- `huffman_reversed_bits_cached.rs` ← vendor fixed path
- Wire in `read_internal_compressed_canonical_specialized` before or
  alongside ShortBitsCached

**Bootstrap decode matrix (add to doc):**

| Block type | x86 today | Target |
|---|---|---|
| Dynamic | `IsalLitLenCode` / ISA-L | keep |
| Fixed | `HuffmanCodingSymbolsPerLength` (one bit/symbol) | ReversedBitsCached |
| Stored | dynamic finder rejects → tail byte walk | Uncompressed finder |

### A4. `PigzStringView` motivation — soften PipelinedGzEncoder claim

**Main doc:** “`PipelinedGzEncoder` produces this shape, so gzippy frequently
decodes pigz-shaped input.”

**Correction:** pigz's `00 00 ff ff` sync-flush is **pigz output**, a major
**fleet input class** (silesia-pigz losses). No verified emission of that
pattern from `src/compress/pipelined.rs`. Rephrase: “pigz-compressed inputs
and pigz-shaped single-member streams”; add a reproducer test if claiming
gzippy-generated output.

### A5. Per-thread arena — worker-index lifecycle, not TLS alone

**Main doc:** collapse `chunk_buffer_pool` to thread-local storage; remove
cross-thread `Mutex`.

**Correction:** buffers are returned on **`Arc<ChunkData>` drop**, often on
the post-process worker, while the next decode may run on a **different**
pool worker. Vendor rpmalloc amortises **malloc/free on the same OS thread**
(`FasterVector.hpp:64` `thread_local RpmallocThreadInit`).

**Add invariant:** `take_u8(worker_id)` / `return_u8(worker_id, buf)` from
decode **and** post-process closures using the same `worker_id` obtained
from `ThreadPool` TLS in `worker_main`. Debug-assert
`pool_id == current_worker()` on return. “Thread-local pool” without
worker-index keyed take/return will not close the 40% → 17% gap.

### A6. mimalloc vs rpmalloc — tie to perf gate (5)

**Main doc:** prefer mimalloc for maintainability.

**Correction:** the motivating metric compares to **rapidgzip + rpmalloc**
(`LIBRAPIDARCHIVE_WITH_RPMALLOC`). mimalloc is fine for first implementation
velocity; **gate (5) is authoritative**. If `minor-faults` miss 1.5×
rapidgzip after mimalloc, switch to `rpmalloc-rs` before declaring pillar 1
complete. Document both behind `feature = "arena-allocator"`.

### A7. Validation gate fixes

**Gate (1) — thread count:** motivating ratio 1.72 is at **T=4** on 16
physical cores; perf gate (5) compares rapidgzip **-P 16**. North-star
gate (1) should require ratio < 0.5 at
**T = min(16, num_cpus::get_physical())**, not T=4 alone. Keep T=4 as an
optional regression canary.

**Gate (1) — fixture split:** current test uses **24 MiB low-entropy
synthetic** (friendly to speculation). Gate (5) uses **silesia.tar.gz**
(~1.4 GB, hard). State explicitly:

- Gate (1a): synthetic, ratio < 0.5 at T=max — algorithmic parallel win.
- Gate (1b): add `test_single_member_parallel_silesia_*` on neurotic —
  same ratio bar on real corpus (or gate (5) wall-time subsumes this).

**Gate (4) — dead_code allows:** “remove every `#![allow(dead_code)]` from
`parallel/*.rs`” conflicts with **out-of-scope seekable scaffolding**
(`UnsplitBlocks`, `block_map`, subchunk index — listed in Out of scope).
Scope gate (4) to **production SM hot-path modules** only, with an explicit
allowlist or `#[cfg(feature = "seekable-index")]` for index/scaffolding
files.

**Homelab entrypoint:** add `make test-x86_64` (pushes branch, runs
`routing` suite on neurotic) as the authoritative CI path alongside raw
`cargo test` on the box.

### A8. Routing / sizing facts to document

| Gate | Location | Value |
|---|---|---|
| Classifier parallel SM | `decompress/mod.rs:81` `MIN_PARALLEL_COMPRESSED` | compressed ≥ **10 MiB** |
| Internal `decompress_parallel` | `single_member.rs:32` `MIN_PARALLEL_SIZE` | deflate body ≥ **4 MiB** |
| Perf test fixture | `routing.rs:453` | 24 MiB raw → ~12 MiB gzip |

`adjusted_chunk_size_bytes` (`single_member.rs:66-78`) affects chunk count
on the 24 MiB fixture — mention when interpreting ratio 1.72 (may be
partly under-partitioned at T=4, not purely page-fault cost).

### A9. Missing work items (add to gap list)

| Item | Location | Why |
|---|---|---|
| `processReadyPrefetches` | `block_fetcher.rs:575-584` | Vendor moves ready prefetches into cache opportunistically; gzippy skips — workers may idle while prefetches sit in-flight |
| Trace triage | `GZIPPY_LOG_FILE` + `scripts/parallel_sm_log_summary.py` | Gate (6) uses `GZIPPY_VERBOSE`; add log summary for partition-level regressions |
| `FetchMultiStream` | `prefetcher.rs:13` preamble lie | Keep as honesty fix; safe to defer entirely for SM-only completion |

### A10. Suggested integration order for Opus

When merging this appendix into the body:

1. Rewrite **§ StreamedResults** (A1) — highest impact; fixes smallest-work-item and coherence bullets.
2. Expand **§ Cached-bits Huffman** with fixed reversed-bits (A3).
3. Expand **§ Per-thread allocator** with worker-index lifecycle (A5).
4. Fix **§ Parallel block-finding** to reference `find_blocks_parallel` (A2).
5. Update **§ Validation gate** (A7) and **§ Out of scope / gate (4)** (A7).
6. Soften pigz motivation (A4); add bootstrap matrix table (A3).
7. Delete this appendix.
