# Parallel single-member decode — architecture & gz↔rapidgzip role map

**Scope:** the pure-Rust parallel single-member decode pipeline under
`src/decompress/parallel/` (production path for `pure-rust-inflate` /
`gzippy-isal` builds when the classifier returns `DecodePath::ParallelSM`).

**Purpose of this doc:** it is the single canonical place the *"which gzippy
module ports which rapidgzip source"* mapping lives, so a future session does
**not** re-derive the structure from a cold read. The mandate (CLAUDE.md) is a
**faithful structural port of rapidgzip** — `vendor/rapidgzip/` is the
blueprint, mirror its `file:line`. When a gz module "works but looks
structurally off", the cited vendor file is the reference.

The same table is mirrored in the module-doc at the top of
`src/decompress/parallel/mod.rs` (keep the two in sync). Per-module doc
comments carry the precise `file:line` anchors; this doc is the index.

Vendor source root: `vendor/rapidgzip/librapidarchive/src/` (paths below are
relative to that, e.g. `rapidgzip/GzipChunkFetcher.hpp`, `core/BlockFetcher.hpp`).

---

## Call flow (production)

```
single_member::decompress_parallel        (classifier-routed thin wrapper)
  └─ sm_driver::read_parallel_sm           (parse gzip envelope; CRC32+ISIZE verify)
       └─ chunk_fetcher::drive             (GzipChunkFetcher::processNextChunk + BlockFetcher dispatch)
            ├─ block_fetcher / prefetcher  (prefetch + LRU cache coordinator)
            ├─ {async,gzip}_block_finder + blockfinder_validation  (block-boundary discovery)
            └─ chunk_decode::decode_chunk_with_rapidgzip           (per-chunk decode driver)
                 ├─ marker_inflate::Block   (window-absent u16-marker decode)
                 ├─ lut_huffman / huffman_* (Huffman coding tables)
                 ├─ apply_window / replace_markers  (window application + marker resolution)
                 └─ chunk_data / segmented_*        (decoded-data + marker buffers)
       └─ output_writer / fd_vectored_write (streamed in-order writev to the sink)
```

---

## Role map (gzippy module → rapidgzip counterpart)

### Orchestration / consumer

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `single_member` | `ParallelGzipReader` (entry / orchestration) | `ParallelGzipReader.hpp` |
| `sm_driver` | `ParallelGzipReader::read*` driver loop | `ParallelGzipReader.hpp:1320-1330` |
| `chunk_fetcher` | `GzipChunkFetcher::processNextChunk` + `BlockFetcher` consumer | `GzipChunkFetcher.hpp:311-362`, `core/BlockFetcher.hpp:245-329` |
| `chunk_decode` | `GzipChunkFetcher::decodeChunkWithRapidgzip` / `decodeChunk` (per-chunk decode driver) | `GzipChunkFetcher.hpp` |
| `chunk_handle` | `std::shared_ptr<ChunkData>` prefetch-cache alias semantics | `GzipChunkFetcher.hpp:579-582` |
| `output_writer` / `fd_vectored_write` | `writeFunctor` / vectored `writeAll` → `toIoVec` / splice | `ParallelGzipReader.hpp:521`, `ChunkData.hpp:794`, `DecodedData.hpp:529` |

### Prefetch / scheduling

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `block_fetcher` | `BlockFetcher` (prefetch/cache coordinator) | `core/BlockFetcher.hpp:38-688` |
| `prefetcher` | `FetchingStrategy` | `core/Prefetcher.hpp:18-54` |
| `cache` | `Cache` + `LeastRecentlyUsed` strategy | `core/Cache.hpp` |
| `thread_pool` | `ThreadPool` | `core/ThreadPool.hpp:33-248` |
| `statistics` | fetcher statistics types | `core/BlockFetcher.hpp:34-200`, `GzipChunkFetcher.hpp:55-75` |
| `streamed_results` | `StreamedResults` | `core/StreamedResults.hpp:27-158` |

### Block finding (three distinct finders — do not confuse)

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `blockfinder_validation` (type `DeflateBlockValidator`) | `blockfinder/{DynamicHuffman,Uncompressed}.hpp` per-position validators | `blockfinder/` |
| `gzip_block_finder` | `GzipBlockFinder` (offset partitioner) | `GzipBlockFinder.hpp:34-307` |
| `async_block_finder` | `core/BlockFinder<RawFinder>` async coordinator | `core/BlockFinder.hpp:35-218` |
| `bit_reader` | `core/BitReader.hpp` (shared LSB bit reader) | `core/BitReader.hpp` |
| `bit_manipulation` | bit-manipulation primitives (`byteSwap` etc.) | `core/BitManipulation.hpp` |

### Decode engine

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `marker_inflate` (type `Block`) | `deflate::Block` — u16 marker-ring decode (formerly `deflate_block`) | `gzip/deflate.hpp:513-1156` |
| `width_ring` | window pair as one object (width reinterpretation = the "flip") | `gzip/deflate.hpp:805-806,890-894,936-939` |
| `used_window_symbols` | `deflate::getUsedWindowSymbols` | `gzip/deflate.hpp:1846-1988` |
| `inflate_wrapper` | `IsalInflateWrapper` (clean-tail; ISA-L on x86_64) | `gzip/isal.hpp` |
| `lut_huffman` / `lut_bulk_inflate` | bulk LUT inflate (ISA-L `make_inflate_huff_code` lineage) | `vendor/isa-l/igzip/igzip_inflate.c:46-599` |
| `huffman_base` | `HuffmanCodingBase` | `huffman/HuffmanCodingBase.hpp` |
| `huffman_reversed_bits_cached` | `HuffmanCodingReversedBitsCached` | `huffman/HuffmanCodingReversedBitsCached.hpp:32-136` |
| `huffman_short_bits_cached` | `HuffmanCodingShortBitsCached` | `huffman/HuffmanCodingShortBitsCached.hpp:1-172` |
| `huffman_symbols_per_length` | `HuffmanCodingSymbolsPerLength` | `huffman/HuffmanCodingSymbolsPerLength.hpp` |
| `asm_kernel` | (gz-specific) `asm!` clean-loop kernel for `run_contig` (feature `asm-kernel`, x86_64) | — |

`marker_inflate`'s `read_internal_compressed_specialized<const CONTAINS_MARKERS>`
dispatcher is split (commit 87b82265, byte-exact) into three methods mirroring
rg's decode decomposition:

| gzippy method | rapidgzip counterpart | anchor |
|---|---|---|
| `decode_clean_fast_loop` | `readInternalCompressedMultiCached<false>` | `gzip/deflate.hpp:1589-1666` |
| `decode_marker_fast_loop` | `readInternalCompressedMultiCached<true>` | `gzip/deflate.hpp:1585-1666` |
| `decode_careful_tail<M>` | per-symbol `readInternalCompressed` fallback (wrap-straddle / resumable boundary / block tail) | `gzip/deflate.hpp:1612-1661` |

### Window / marker resolution & data plane

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `apply_window` | `ChunkData::applyWindow` | `ChunkData.hpp:302` |
| `replace_markers` | `MapMarkers` semantics | `MarkerReplacement.hpp:24-42` |
| `window_map` | `WindowMap` (`std::map` + mutex; `SharedWindow`) | `WindowMap.hpp` |
| `block_map` | `BlockMap` | `core/BlockMap.hpp` |
| `chunk_data` | `ChunkData` (+ `Subchunk`) | `ChunkData.hpp:80-400,138-145` |
| `segmented_buffer` | `DecodedData::append` clean storage | `rapidgzip/DecodedData.hpp:278-289` |
| `segmented_markers` | `DecodedData::dataWithMarkers` | `rapidgzip/DecodedData.hpp:238-275` |
| `compressed_vector` | `CompressedVector` | `CompressedVector.hpp:113-246` |
| `chunk_buffer_pool` / `rpmalloc_alloc` | `FasterVector` + `RpmallocAllocator` | `core/FasterVector.hpp:46-128` |

### Format / utility

| gzippy module | rapidgzip counterpart | anchor |
|---|---|---|
| `gzip_format` | gzip header+footer parsing | `gzip/gzip.hpp:77-309` |
| `gzip_definitions` | `gzip` + `deflate` constants | `gzip/definitions.hpp` |
| `crc32` | `CRC32Calculator` + carryless `combineCRC32` | `gzip/crc32.hpp` |
| `error` | `Error` | `core/Error.hpp` |
| `stored_split` | (gz-specific) non-speculative parallel decode for stored-block-dominated streams; portable, not `parallel_sm`-gated | — |

### Instruments (NO rapidgzip counterpart)

`parallel::{perturb, phase_timing, storeprobe}` are campaign measurement
instruments with NO rapidgzip counterpart. Each is a Cargo-feature-gated module
(`perturb`, `phase-timing`, `storeprobe` — all OFF by default) whose every item
compiles to a true no-op on the production build, so they have ZERO effect on
the production bytes/timing (`storeprobe` is byte-INEXACT when its feature is
enabled — a removal oracle, never shipped). They live as flat modules alongside
the pipeline rather than in a subdirectory.

---

## Multi-member routing (`MultiMemberChunked` — 2026-07-05)

`DecodePath::MultiMemberChunked` → `sm_driver::read_parallel_sm_multi` walks each
member and inflates it with the FULL within-member parallel engine
(`read_parallel_sm` per member, per-member CRC32 + ISIZE verified), streaming
output in member order. It is the **deterministic route for a MIXED "GZ" ++
plain concatenation** (`bgzf::gz_coverage_is_pure` returns false → not pure GZ),
which the BGZF fast path (`decompress_bgzf_parallel`) would truncate.

**Plain multi-member streams stay on `MultiMemberPar`** (the member-per-worker
split): routing dominant/few-member distributions to this member-walk was
**MEASURED to REGRESS on M1** — the walk spins a full pipeline per member and
oversubscribes threads at high T (few-large T8: 156 ms chunked vs 45 ms
member-per-worker; the member-per-worker baseline was ~90 ms flat / scaling).
The located dominant-member plateau requires the rapidgzip-faithful whole-file
block-finder cross-member continuation (one pool, one chunk grid ignoring member
boundaries + a single decode walking footer→header→empty-window-reset, vendor
`GzipChunk.hpp:468-654`, CRC segmentation per `ChunkData.hpp:559-561` +
`ParallelGzipReader.hpp:1454-1502`), **not** this member-walk shortcut — the
gate-phase core (see `scratchpad/MM-PARALLELSM-DESIGN.md`).

The false-single re-entry (`single_member.rs` `trailing_member_after_first` +
`sm_driver::read_parallel_sm_resume_multi`, counter `MISROUTE_REENTRY_APPLIED`)
handles multi-member streams that mis-detect as single (empty-first member; first
member compressed past the 16 MiB detection window).

---

## Related docs

- `docs/structural-gap-rapidgzip.md` — measured structural-gap analysis (where the wall is lost vs rg).
- `docs/production-decode-callgraph.md` / `docs/production-paths.md` — full routing/call-graph.
- `docs/fulcrum-sota.md` — measurement instrument semantics.

Historical row-level closure designs and divergence history formerly lived under
`plans/` (deleted as stale-prone interpretation) — recover from git history if
needed, but re-derive any perf verdict with Fulcrum before citing it.
