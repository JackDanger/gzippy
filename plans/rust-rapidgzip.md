# Rapidgzip → gzippy parallel single-member: unported primitives

`src/decompress/parallel/` is a structural port of rapidgzip's
`ParallelGzipReader → GzipChunkFetcher → BlockFetcher → GzipChunk →
IsalInflateWrapper` chain. This document lists the rapidgzip primitives
still unported. Each named item is a literal vendor class or header,
with the gzippy module and call site to change. The unit is "done"
when the validation gate at the bottom passes on `neurotic` (16
physical x86_64, ISA-L).

## What is already ported (do not touch)

| Vendor | gzippy |
|---|---|
| `core/ThreadPool.hpp:33-248` | `thread_pool.rs` |
| `core/AffinityHelpers.hpp:11,76-101` | `thread_pool.rs::pinning_for_capacity`, `with_pinning_for_capacity`; wired at `chunk_fetcher.rs:324` |
| `core/StreamedResults.hpp:27-158` | `streamed_results.rs` |
| `core/BlockFinder.hpp:35-218` | `raw_block_finder.rs::RawBlockFinderCoordinator` |
| `core/BlockFetcher.hpp:38-687` | `block_fetcher.rs`; `process_ready_prefetches` at `:696` (vendor 427/463) |
| `core/Prefetcher.hpp:60-225` | `prefetcher.rs:41-211` (`FetchNextFixed`, `FetchNextAdaptive`) |
| `core/Prefetcher.hpp:234-336` | `prefetcher.rs:232-343` (`FetchMultiStream` — **not yet wired**, see #5) |
| `core/BlockMap.hpp` | `block_map.rs` |
| `core/Cache.hpp` | `cache.rs` |
| `core/MarkerReplacement.hpp` | `replace_markers.rs:146` (unaligned AVX) |
| `core/common.hpp:498-512 interleave` | `prefetcher.rs:215-227` |
| `rapidgzip/WindowMap.hpp` | `window_map.rs` |
| `rapidgzip/GzipBlockFinder.hpp:34-307` (no seekable branches) | `gzip_block_finder.rs` |
| `rapidgzip/blockfinder/DynamicHuffman.hpp:39-225` + `precodecheck/CountAllocatedLeaves.hpp` | `block_finder.rs` (stored-block from `blockfinder/Uncompressed.hpp` folded in at `:668 find_blocks`) |
| `rapidgzip/chunkdecoding/GzipChunk.hpp:468-654` (handoff at vendor 520-525) | `gzip_chunk.rs:288 decode_chunk_marker_bootstrap_then_isal` |
| `rapidgzip/GzipChunkFetcher.hpp:554-583 queueChunkForPostProcessing` + `applyWindow` | `chunk_fetcher.rs:1069 submit_post_process_to_pool` (callsite `:932`); marker resolution in `apply_window.rs` + `replace_markers.rs` |
| `rapidgzip/gzip/isal.hpp:26-212` | `inflate_wrapper.rs`; patched `mxmlnkn/isa-l` vendored byte-identically |
| `rapidgzip/gzip/deflate.hpp:175` (`HuffmanCodingISAL`) + `:38-39` (`HuffmanCodingDistanceISAL`) | `isal_huffman.rs` |
| `rapidgzip/huffman/HuffmanCodingBase.hpp`, `HuffmanCodingSymbolsPerLength.hpp:30-142` | `huffman_base.rs`, `huffman_symbols_per_length.rs` |

## Unported primitives

Listed in dependency order.

### 1. `RpmallocAllocator` / `FasterVector`

**Vendor:** `core/FasterVector.hpp:73-113` (`RpmallocAllocator`),
`:124` (`FasterVector` alias).

```cpp
template<typename T>
using FasterVector = std::vector<T, RpmallocAllocator<T>>;
```

`RpmallocAllocator` calls `rpmalloc` / `rpfree`; per-thread heap init
via `static thread_local RpmallocThreadInit` (vendor `:64`). Freed
pages return to rpmalloc's per-thread free list, so the next allocation
on the same thread reuses warm pages without faulting. The two
`FasterVector` instances on the SM hot path are `ChunkData::data`
(`DecodedVector = FasterVector<uint8_t>`) and `data_with_markers`
(`MarkerVector = FasterVector<uint16_t>`), cited at
`rapidgzip/DecodedData.hpp:23-24`.

**gzippy now.** `chunk_data.rs:133` `pub data_with_markers: Vec<u16>`,
`:137` `pub data: Vec<u8>` — system allocator. Recycled by a per-worker
LIFO pool in `chunk_buffer_pool.rs` (preserves Vec capacity, eliminates
cross-thread Mutex contention; pool misses and drops past
`MAX_POOLED = 8` still hit `std::alloc::System`'s `munmap`).
`chunk_buffer_pool.rs:67-90` records the ~40% / ~17% silesia
page-fault gap this leaves open.

**Delta.** Add `allocator-api2 = "0.2"` + `rpmalloc-rs` to
`Cargo.toml`. Define `unsafe impl allocator_api2::alloc::Allocator for
RpmallocAlloc` — `Copy + Send` ZST. Change the two `ChunkData` fields
to `allocator_api2::vec::Vec<T, RpmallocAlloc>`. Audit the three
production `extend_from_slice` callsites: `chunk_data.rs:343`
(`append_markered` → `data_with_markers`), `:361` (`append_clean` →
`data`), `:432` (`append_owned_buffer` → `data`). Do not pre-warm
pages at process start — that experiment was −50% throughput
(`chunk_buffer_pool.rs:78-83`).

**Wiring.** `ChunkData::new_with_buffers` at `chunk_data.rs:236-253`
and callers in `gzip_chunk.rs` / `chunk_fetcher.rs`. The per-worker
buffer-pool keys stay; the wrapped Vec then allocates via the
per-thread arena.

**Dependency.** None. Worker → physical-core stability is already in
place via `with_pinning_for_capacity`.

### 2. `HuffmanCodingReversedBitsCached`

**Vendor:** `huffman/HuffmanCodingReversedBitsCached.hpp:32-136`.
Generic LUT-cached canonical decoder, `1 << MAX_CODE_LENGTH` entries of
`(length, symbol)`. `gzip/deflate.hpp:196` aliases it as
`FixedHuffmanCoding` — every BTYPE=01 block. Also the distance HC for
the deflate-specific decoder in #3.

**gzippy now.** No port. `deflate_block.rs:1110
read_internal_compressed_canonical_specialized` builds tables via
`HuffmanCodingSymbolsPerLength` (one-bit-per-symbol tree walk) for
both DYNAMIC and FIXED. RFC 1951 length arrays for FIXED come from
`fixed_huffman_code_lengths()` at `:1315`. x86_64 production routes
through ISA-L (`inflate_wrapper.rs`) and bypasses this; the
marker bootstrap (`gzip_chunk.rs:409 bootstrap_with_deflate_block`)
still hits the slow path on every arch.

**Delta.** Add `huffman_reversed_bits_cached.rs`. Wraps
`HuffmanCodingSymbolsPerLength` plus
`code_cache: [(u8 /*length*/, u16 /*symbol*/); 1 << MAX_LEN]`.
`initialize_from_lengths` fills the cache by reversing each canonical
code via `bit_manipulation::reverse_bits`. `decode(bit_reader)` peeks
`MAX_LEN` and returns the cache entry directly.

**Wiring.** `deflate_block.rs:1126` instantiates the litlen HC and
`:1132` the distance HC. For `CompressionType::FixedHuffman` the
litlen HC swaps to `HuffmanCodingReversedBitsCached`; the distance HC
stays — FIXED blocks have no separate distance table (RFC 1951
3.2.6).

**Dependency.** None — base `HuffmanCodingSymbolsPerLength`
(`huffman_symbols_per_length.rs:228`) is already ported.

### 3. `HuffmanCodingShortBitsCachedDeflate<11>`

**Vendor:** `huffman/HuffmanCodingShortBitsCachedDeflate.hpp:22-280`.
Deflate-specialized 11-bit LUT for DYNAMIC blocks. Cache entry:

```cpp
struct CacheEntry { uint8_t bitsToSkip; uint8_t symbolOrLength; uint16_t distance; };
```

The distance code is fused into the literal-LUT when both fit one
peek. Selected by `gzip/deflate.hpp:177 LiteralOrLengthHuffmanCoding`
when `WITH_DEFLATE_SPECIFIC_HUFFMAN_DECODER` is defined. The LUT fill
at vendor `:84-109` reads `distanceHC.codeCache()` directly.

**gzippy now.** No port. Same call site as #2 uses
`HuffmanCodingSymbolsPerLength` for the `CompressionType::DynamicHuffman`
arm.

**Delta.** Add `huffman_short_bits_cached_deflate.rs`. Const-generic
`LUT_BITS`. `code_cache: [CacheEntry; 1 << LUT_BITS]` with the layout
above. `decode` peeks `LUT_BITS` and returns the cache entry; falls
back to base canonical decode for codes longer than `LUT_BITS`.

**Wiring.** `deflate_block.rs:1126` litlen HC for the
`CompressionType::DynamicHuffman` arm (non-ISA-L fallback) and the
marker-bootstrap path at `gzip_chunk.rs:409`.

**Dependency.** Requires #2 — the LUT fill reads the distance HC's
`code_cache`.

### 4. `HuffmanCodingShortBitsMultiCached<11>`

**Vendor:** `huffman/HuffmanCodingShortBitsMultiCached.hpp:24-269`.
Two-symbols-per-lookup variant of #3. Cache entry packs
`needToReadDistanceBits:1, bitsToSkip:6, symbolCount:2, symbols:18`
into one `u32`. `decode` returns `(packed_symbols: u32, count: u32)`.
Selected by `gzip/deflate.hpp:179` when
`WITH_MULTI_CACHED_HUFFMAN_DECODER`.

**gzippy now.** No port.

**Delta.** Add `huffman_short_bits_multi_cached.rs`. Packed-u32 cache
entry; `decode` returns the two-symbol pair.

**Wiring.** Same site as #3. The appender loop in
`gzip_chunk.rs:409 bootstrap_with_deflate_block` unpacks the
two-symbol return.

**Dependency.** Requires #2 for the distance HC at fill time.

### 5. `FetchMultiStream` production wiring

**Vendor:** `rapidgzip/ParallelGzipReader.hpp:85`:

```cpp
using ChunkFetcher = rapidgzip::GzipChunkFetcher<
    FetchingStrategy::FetchMultiStream, ChunkData>;
```

This is the prefetch strategy vendor ships in production.

**gzippy now.** `FetchMultiStream` is ported at `prefetcher.rs:232-343`.
`chunk_fetcher.rs:303` constructs `FetchNextAdaptive::new(FETCH_STRATEGY_MEMORY)`
(`FETCH_STRATEGY_MEMORY = 32` at `:175`).

**Delta.**

- Swap the `FetchNextAdaptive` type parameter to `FetchMultiStream` at
  the five `BlockFetcher<…, FetchNextAdaptive, …>` occurrences:
  `chunk_fetcher.rs:299, 465, 1037, 1098, 1459`.
- Replace the constructor at `chunk_fetcher.rs:303` with
  `FetchMultiStream::new(memory_size_per_stream = 3, max_stream_count = 16)`
  (vendor defaults).

**Dependency.** None — the `FetchingStrategy` trait
(`prefetcher.rs:20-36`) covers both, and `BlockFetcher` is generic
over it.

## Not part of this port

Vendor primitives or gzippy code intentionally excluded.

- **`blockfinder/PigzStringView.hpp:30-179`** — referenced only from
  `src/benchmarks/benchmarkGzipBlockFinder.cpp`; vendor's production
  `ParallelGzipReader.hpp:85` uses `GzipBlockFinder` alone.
- **`core/ParallelBitStringFinder.hpp:35-265`** — used only by
  `tools/ibzip2.cpp:111` (bzip2; out of project scope).
- **`huffman/HuffmanCodingDoubleLiteralCached.hpp`** —
  `gzip/deflate.hpp:45,182` shows it commented out.
- **`IndexFileFormat.hpp` and the seekable reader.** Subchunk indexing
  at `chunk_data.rs::subchunks` and `UnsplitBlocks` at
  `chunk_fetcher.rs:118` are plumbed but gated behind
  `#[cfg(feature = "seekable-index")]` until the seekable reader
  lands.
- BZIP2, ZLIB-format, BGZF parallel encoders, multi-member parallel
  encoders, Python bindings, `GzipReader.hpp`, `GzipAnalyzer.hpp`.
- **`AlignedAllocator.hpp`** — `replace_markers.rs:146` uses unaligned
  AVX (`_mm256_loadu_si256`); re-evaluate only if `dTLB-load-misses`
  stays elevated after #1.
- **`src/decompress/deflate64.rs`** — gzippy-original Deflate64; no
  rapidgzip counterpart; no consumer in the SM path. Delete.
- **`src/decompress/ultra_fast_inflate.rs`** — gzippy-original;
  consumed by `bgzf.rs`, not the SM path. Keep for BGZF.

## Validation gate

On `neurotic` (16 physical x86_64, ISA-L), via `make test-x86_64`.
The unit is the gate.

1. **Synthetic perf.**
   `cargo test --release -- test_single_member_parallel_not_slower_than_sequential`
   reports `ratio < 0.5` at `T = min(16, num_cpus::get_physical())` on
   the 24 MiB low-entropy fixture in `routing.rs`.
2. **Silesia perf.** `test_single_member_parallel_silesia` reports
   `ratio < 0.5` on real `silesia.tar.gz`
   (`https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia`). The
   synthetic-class proxy in `routing.rs` is not this gate.
3. **Routing / FNAME / coordinator traps.**
   `test_single_member_routing_multithread`,
   `test_coordinator_boundary_search_runs_on_x86_64_isal`, and
   `test_parallel_sm_handles_fname_header` pass byte-perfect.
   `MARKER_PIPELINE_RUNS` (`single_member.rs:95`) and
   `COORDINATOR_BOUNDARY_SEARCH_RUNS` (`chunk_fetcher.rs:1281`)
   increment.
4. **Dead-code allows gone.** Every
   `#![allow(dead_code)] // vendor-faithful rapidgzip port` removed
   from production SM hot-path modules. `cargo build --release -- -D warnings`
   succeeds. Seekable scaffolding stays behind
   `#[cfg(feature = "seekable-index")]`.
5. **`perf stat` vs vendor.**

   ```
   perf stat -e dTLB-load-misses,major-faults,minor-faults,cycles \
       ./target/release/gzippy -d -c silesia.tar.gz > /dev/null
   ```

   compared against `rapidgzip -P 16 -c silesia.tar.gz > /dev/null` on
   the same host (`pip install rapidgzip`). Required: gzippy
   `minor-faults` within 1.5× of rapidgzip's; gzippy wall-time within
   1.2×. The arena in #1 is what closes the `minor-faults` gap;
   vendor's rpmalloc per-thread free list keeps freed pages off
   `munmap`.

If any of (1)–(5) miss, the unit hasn't landed.

## Reading order

- `src/decompress/parallel/single_member.rs` — entry,
  `MARKER_PIPELINE_RUNS` deletion trap.
- `chunk_fetcher.rs:257 drive` → `consumer_loop` →
  `submit_decode_to_pool` → `submit_post_process_to_pool`.
- Vendor `rapidgzip/ParallelGzipReader.hpp:495 read` and
  `rapidgzip/GzipChunkFetcher.hpp:312 processNextChunk`, side-by-side
  with the Rust.
