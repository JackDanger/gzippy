# Rapidgzip → gzippy parallel single-member: architectural completion

*Last verified against commit `2345dbc`.*

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
`make test-x86_64`. End state: ratio < 0.5 on the same host. (If `make test-x86_64`
on neurotic currently shows ratio ≤ 1.5, re-measure — the 1.72
figure may be stale.)

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
top of any page-fault cost. *Parallel block-finding* (below) targets
the under-partition; *Per-thread allocator arena* (below) targets the
faults.

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
maintained). Per-thread init is lazy, matching vendor 64. The
motivating metric is rapidgzip + rpmalloc
(`LIBRAPIDARCHIVE_WITH_RPMALLOC`); mimalloc is fine for first
implementation, but if validation gate (5)'s `minor-faults` 1.5×
bound is missed, switch to `rpmalloc-rs` before declaring this
pillar complete.
`chunk_buffer_pool.rs` is keyed per worker index — `take_u8(worker_id)`
/ `return_u8(worker_id, buf)`. The take path becomes lock-free per
worker; cross-thread returns (the `Arc<ChunkData>` `Drop` may run on a
post-process worker, not the decoder that allocated) stay supported
via a fallback enqueue.

**Modules touched.** `chunk_data.rs` (two fields, ~30 constructor /
accessor call sites), `chunk_buffer_pool.rs` (per-worker-index pools),
every test that constructs `ChunkData` directly. `Cargo.toml` adds
`allocator-api2 = "0.2"` and `mimalloc` behind a feature gate
`arena-allocator`.

**Risks.** `allocator-api2::Vec` does not expose every `std::Vec`
method. The production `extend_from_slice` callsites on the two
buffers are `chunk_data.rs:337` (`append_markered` →
`data_with_markers`), `:355` (`append_clean` → `data`), and `:426`
(`append_owned_buffer` → `data`). Audit those plus every other
`extend` on either field. Do not pre-warm pages at process start — that
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

### `StreamedResults` for async raw-finder coordination

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

**Now.** `block_finder.rs:1034 find_blocks_parallel` already exists,
already does fan-out + merge with a `+1024`-bit boundary overlap at
`:1065` — but its only callers are tests (lines 1274, 1302, 1314).
The production slow path at `chunk_fetcher.rs:1344` calls
single-threaded `finder.find_blocks(chunk_begin, chunk_end)` from
`block_finder.rs:668` on the consumer, inside a 64-iteration loop
(`chunk_fetcher.rs:1321-1322`; constants are in bits:
`CHUNK_SIZE_BITS = 65 536` ≈ 8 KiB scanned per iteration,
`MAX_SCAN_BITS = 4 Mib` ≈ 512 KiB total window).

**Work.** Wire `speculative_decode_find_boundary`
(`chunk_fetcher.rs:1315`) through the existing `find_blocks_parallel`,
rehosted on `Arc<ThreadPool>` (not `std::thread::scope`) and the
per-worker buffer pool. Activation threshold: vendor's `chunkSize`
rule (`core/ParallelBitStringFinder.hpp:91-115`) — inputs below
~16 KiB stay single-threaded.

**Optional second layer.** `core/ParallelBitStringFinder.hpp:35-265`
(vendor's general parallelised bit-string scan, offsets streamed
through a `StreamedResults`-shaped queue) is worth porting *only* if
profiling after `find_blocks_parallel` is wired shows the LUT scan
itself is still the bottleneck on real fixtures. Evaluate after the
production rehost, not before.

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
on pigz-shaped and stored-block inputs — both common in the wild as
fleet input classes. The marker-bootstrap bug is fixed; the
*finder coverage* gap remains.

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

### Opportunistic prefetch promotion

**Vendor.** `core/BlockFetcher.hpp:427 processReadyPrefetches` is
called from `prefetchNewBlocks` (vendor 463) before issuing a new
prefetch. It walks `m_prefetching` for receivers whose
`future.wait_for(0s) == future_status::ready`, takes them, and inserts
the result into `prefetch_cache` — so workers don't idle on
prefetches that completed while the consumer was elsewhere.

**Now.** `block_fetcher.rs:575-588` documents the deliberate omission:
"we don't yet poll-collect ready prefetches into the prefetch_cache
because the worker pool's mpsc::Receiver doesn't expose a
non-blocking `wait_for(0s)` equivalent…". The premise is stale —
`mpsc::Receiver::try_recv()` is exactly that non-blocking poll. The
comment's fallback reasoning ("consumer will take them via
getFromCaches on the next iteration") only holds for a *hot*
consumer; on finder-bottlenecked decodes the consumer waits on a
*different* key while a ready prefetch sits in `prefetching` unused.

**Work.** Add `process_ready_prefetches(&self)` to `block_fetcher.rs`:
iterate `prefetching: HashMap<Key, Receiver<…>>`, call `try_recv` on
each, on `Ok(value)` insert into `prefetch_cache` and remove from
`prefetching`. Call it from `prefetch_new_blocks` before each new
dispatch (mirroring vendor 463) and from `consumer_loop` once per
outer iteration. Strip the now-stale comment block at `:575-588`.

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

All on `neurotic` (16 physical x86_64, ISA-L), driven by
`make test-x86_64` (pushes the branch, runs the `routing` suite over
SSH). Local `make` is insufficient — the ratio=1.72 regression
triggers only at ≥4 physical cores.

1a. **Synthetic perf guard.**
    `cargo test --release -- test_single_member_parallel_not_slower_than_sequential`
    reports `ratio < 0.5` at T = `min(16, num_cpus::get_physical())` on
    the 24 MiB low-entropy synthetic fixture. Capture printed
    `seq_mbps`, `par_mbps`, `ratio`. Floor for `par_mbps`: ≥ 2×
    sequential libdeflate throughput on the same fixture (~600 MB/s
    baseline on neurotic). T = 4 stays as an optional regression
    canary.

1b. **Real-corpus perf guard.** Add
    `test_single_member_parallel_silesia` (or rely on gate (5)
    wall-time, which subsumes it): same ratio < 0.5 bar at the same
    T on the real Silesia corpus. Synthetic-only is friendly to
    speculation; the real corpus is the honest gate.

2. `cargo test --release -- test_single_member_routing_multithread`
   passes byte-perfect, and `MARKER_PIPELINE_RUNS`
   (`single_member.rs:95`) increments — proves the parallel path took
   the input (no silent libdeflate fall-through).

3. `cargo test --release -- test_parallel_sm_handles_fname_header`
   passes (header-parse / partition-offset interaction).

4. Every `#![allow(dead_code)] // vendor-faithful rapidgzip port`
   removed from the **production SM hot-path modules** in
   `src/decompress/parallel/*.rs`. Out-of-scope seekable scaffolding
   (`UnsplitBlocks`, `block_map`, the subchunk index — listed in
   *Out of scope*) keeps its allow behind
   `#[cfg(feature = "seekable-index")]`; those modules stay dead
   until the seekable reader lands, which is not in this unit.
   `cargo build --release -- -D warnings` succeeds for the hot-path
   set.

5. `perf stat -e dTLB-load-misses,major-faults,minor-faults,cycles
   ./target/release/gzippy -d -c silesia.tar.gz > /dev/null`
   (1.4 GB Silesia, `https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia`).
   Compare to `rapidgzip -d -P 16 -c silesia.tar.gz > /dev/null` on
   the same host. Required: gzippy `minor-faults` within 1.5× of
   rapidgzip's; gzippy wall-time within 1.2× of rapidgzip's. Capture
   exact baseline numbers when the first item lands (rapidgzip is
   installable on neurotic via `pip install rapidgzip`); the 1.2× /
   1.5× bounds are relative to that capture.

6. **Diagnostic only.** `GZIPPY_VERBOSE=1
   ./target/release/gzippy -d -c <fixture>` (stats dump at
   `chunk_fetcher.rs:384-438`): `Slow-path decode: ok > 0 and
   fail < 0.1 × ok`; `Buffer pool u8: misses < 0.05 × hits`;
   `Prefetch guard-rejects` does not grow with input size. For
   partition-level regressions add `GZIPPY_LOG_FILE=/tmp/sm.log
   ./target/release/gzippy -d -c <fixture>` and post-process with
   `scripts/parallel_sm_log_summary.py`. Use (6) to localise which
   item regressed if (1a)–(5) miss.

If any of (1a)–(5) miss, the unit didn't land — a single item's gap
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
