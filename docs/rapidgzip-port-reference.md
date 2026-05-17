# rapidgzip → gzippy parallel single-member: structural gap analysis

This document is the structural reference for the rapidgzip → gzippy port of
the parallel single-member decoder. Performance is intentionally ignored.
The goal of this file is: **for every component on either side, identify its
counterpart and characterize what differs structurally.**

All citations are `file:line` against:

- gzippy: `src/decompress/parallel/*.rs` on branch `feat/cross-chunk-retry`.
- rapidgzip: `vendor/rapidgzip/librapidarchive/src/rapidgzip/*.hpp` and
  `vendor/rapidgzip/librapidarchive/src/core/*.hpp`.

Status legend used in the component map:

- **Faithful**: semantics + control flow match modulo language idioms; can
  be re-derived from rapidgzip by translating Rust back to C++.
- **Deviated**: deliberate semantic difference (documented per item below).
- **Missing**: rapidgzip has it; gzippy does not.
- **Extra**: gzippy has it; rapidgzip does not.

---

## A. Component map

| rapidgzip file / class | Rust counterpart | Status |
|------------------------|------------------|--------|
| `chunkdecoding/GzipChunk.hpp` — `GzipChunk<T>::decodeChunk` (entry, L661–L852) | `parallel/gzip_chunk.rs` + `parallel/chunk_fetcher.rs::decode_or_iterate` | Deviated |
| `GzipChunk::decodeChunkWithInflateWrapper` (L190–L268) | `gzip_chunk.rs::decode_chunk_with_inflate_wrapper` (L182–L219) | Faithful (subset) |
| `GzipChunk::decodeChunkWithRapidgzip` (L413–L657) | (no full counterpart) — partial behavior in `fast_marker_inflate::decode_chunk_bootstrap` + `gzip_chunk::finish_decode_chunk_with_inexact_offset` | Deviated |
| `GzipChunk::finishDecodeChunkWithInexactOffset` (L280–L410) | `gzip_chunk.rs::finish_decode_chunk_with_inexact_offset` (L234–L462) + `decode_chunk_with_window` (L86–L165) | Deviated |
| `GzipChunk::tryToDecode` lambda (L712–L734) | `chunk_fetcher.rs::decode_or_iterate` (L187–L244) | Deviated |
| `GzipChunk::appendDeflateBlockBoundary` (L161–L183) | `chunk_data.rs::ChunkData::append_block_boundary` (L264–L285) | Deviated (no split gating) |
| `GzipChunk::startNewSubchunk` (L46–L58) | `chunk_data.rs::ChunkData::new` first-subchunk init (L135–L156) | Faithful |
| `GzipChunk::finalizeChunk` (L135–L159) | `chunk_data.rs::ChunkData::finalize` (L293–L301) | Deviated (no subchunk merge, no window finalization) |
| `GzipChunk::finalizeWindowForLastSubchunk` (L99–L133) | (none) | Missing |
| `GzipChunk::determineUsedWindowSymbolsForLastSubchunk` (L60–L97) | (none) | Missing |
| `GzipChunkFetcher.hpp` — `GzipChunkFetcher<T,S>::get` (L206–L225) | `chunk_fetcher.rs::drive` + `consumer_loop` (L114–L671) | Deviated |
| `GzipChunkFetcher::processNextChunk` (L311–L362) | `chunk_fetcher.rs::consumer_loop` inner per-iteration body | Deviated |
| `GzipChunkFetcher::decodeBlock` (override, L692–L729) | `chunk_fetcher.rs::worker_loop` (L246–L373) | Deviated |
| `GzipChunkFetcher::waitForReplacedMarkers` (L478–L518) | `consumer_loop` apply_window block (L596–L612) | Deviated (sync, single chunk) |
| `GzipChunkFetcher::queueChunkForPostProcessing` (L553–L583) | (inlined into `consumer_loop`) | Deviated |
| `GzipChunkFetcher::appendSubchunksToIndexes` (L364–L465) | (none) | Missing |
| `GzipChunkFetcher::Statistics` (L55–L75) | `chunk_data.rs::ChunkStatistics` | Deviated (smaller surface) |
| `core/BlockFetcher.hpp` — `BlockFetcher<F,D,S>` cache + prefetch (L41–end) | inlined `spec[]` ring + `submit_job` in `chunk_fetcher.rs::consumer_loop` (L422–L668) | Deviated |
| `BlockFetcher::Statistics` (L53–L155) | (none) | Missing |
| `core/Cache.hpp`, `core/Prefetcher.hpp`, `core/ThreadPool.hpp` | `std::thread::scope` + `std::sync::mpsc` ad hoc | Deviated |
| `GzipBlockFinder.hpp` — `GzipBlockFinder` partitioning + `get`/`insert`/`finalize` (L34–end) | (uniform partition by `TARGET_COMPRESSED_CHUNK_BYTES` in `chunk_fetcher.rs::drive` L120–L123) | Deviated (no on-line confirmation) |
| `blockfinder/DynamicHuffman.hpp` — `seekToNonFinalDynamicDeflateBlock` (L166–L298) | `block_finder.rs::BlockFinder::find_dynamic_blocks` (L686–L795) | Faithful (with deviations) |
| `blockfinder/DynamicHuffman.hpp` — LUT generator + `nextDeflateCandidate` (L39–L154) | `block_finder.rs::is_deflate_candidate_n` / `next_deflate_candidate` / `generate_deflate_lut` (L62–L123) | Faithful |
| `blockfinder/precodecheck/CountAllocatedLeaves.hpp` — `checkPrecode` (L95–L213) | `block_finder.rs::validate_precode` (L164–L216) | Deviated (different LUT layout) |
| `blockfinder/Uncompressed.hpp` — `seekToNonFinalUncompressedDeflateBlock` (L21–L95) | `block_finder.rs::BlockFinder::find_uncompressed_blocks` (L562–L665) | Faithful |
| `blockfinder/Bgzf.hpp` | (none; gzippy's `gzippy-parallel` "GZ" format handled elsewhere in `decompress/bgzf.rs`) | Missing (in parallel SM) |
| `huffman/HuffmanCodingISAL.hpp` — lit/len wrapper (L21–L188) | `isal_huffman.rs::IsalLitLenCode` (L80–L226) | Faithful |
| `huffman/HuffmanCodingDistanceISAL.hpp` — distance wrapper (L20–L124) | `isal_huffman.rs::IsalDistCode` (L243–L336) | Faithful |
| `huffman/HuffmanCodingDoubleLiteralCached.hpp` — multi-symbol decoder | (none; gated off in rapidgzip anyway when ISA-L is present) | Missing |
| `huffman/HuffmanCodingReversedBitsCached*.hpp` (precode / fixed Huffman) | `block_finder.rs::build_huffman_table` (L956–L997) | Deviated (only used by block finder, not by decoder hot loop) |
| `huffman/HuffmanCodingShortBitsCached*.hpp` | (none; not used by gzippy's decoder path) | Missing |
| `MarkerReplacement.hpp` — `MapMarkers<FULL_WINDOW>` + `replaceMarkerBytes` (L15–L60) | `replace_markers.rs::replace_markers{,_avx2,_neon,_scalar}` (L30–L156) | Deviated (different encoding) |
| `WindowMap.hpp` — `WindowMap` mutex-protected `std::map`, no condvar (L19–end) | `window_map.rs::WindowMap` (BTreeMap + mutex + **condvar**, no compression) (L23–L96) | Deviated |
| `ChunkData.hpp` — `ChunkData` (L88–L573) | `chunk_data.rs::ChunkData` (L86–L317) | Deviated (smaller surface; subchunk semantics differ) |
| `ChunkData::Configuration` (L97–L113) | `chunk_data.rs::ChunkConfiguration` (L55–L79) | Deviated (fields trimmed) |
| `ChunkData::Subchunk` (L115–L145) | `chunk_data.rs::Subchunk` (L29–L37) | Deviated (no `newlineCount`, `usedWindowSymbols`) |
| `ChunkData::Statistics` (L147–L180) | `chunk_data.rs::ChunkStatistics` (L43–L50) | Deviated (smaller) |
| `ChunkData::applyWindow` (L246–L394) | `apply_window.rs::apply_window` (L17–L65) | Deviated (single window only, no per-subchunk windowing, no compression, no sparsity) |
| `ChunkData::appendDeflateBlockBoundary` (L455–L467) | `chunk_data.rs::ChunkData::append_block_boundary` (L264–L285) | Deviated (always emits subchunk) |
| `ChunkData::matchesEncodedOffset` (L396–L403) | `chunk_data.rs::ChunkData::matches_encoded_offset` (L163–L165) | Faithful |
| `ChunkData::setEncodedOffset` (L601–L629) | (none; consumer handles trimming via `decoded_offset_for`) | Missing |
| `ChunkData::split` (L632–L754) | (none — `append_block_boundary` emits subchunks on-line) | Missing (semantically replaced) |
| `ChunkData::getWindowAt` / `getLastWindow` (DecodedData L394–L488) | `chunk_data.rs::ChunkData::last_32kib_window` (L173–L207) | Deviated (returns `Option<[u8;32768]>`, no `MapMarkers` integration; throws-via-None instead of exception) |
| `ChunkData::cleanUnmarkedData` (DecodedData L491–L516) | (none; two-segment layout enforced by `append_markered` vs `append_clean`) | Missing |
| `ChunkData::appendFooter` / `footers` / `crc32s` vector (L472–L489, L559–L561) | (none — single per-chunk `crc32fast::Hasher`, no `Footer` type) | Missing |
| `CompressedVector.hpp` | (none — windows stored uncompressed as `Arc<[u8;32768]>`) | Missing |
| `DecodedData.hpp` — base class with two-vec layout, `Iterator`, `append`, `applyWindow`, `getLastWindow`, `cleanUnmarkedData` | folded into `chunk_data.rs::ChunkData` (no inheritance) | Deviated |
| `DecodedDataView.hpp` | (none) | Missing |
| `gzip/isal.hpp` — `IsalInflateWrapper` (L26–L212) | `inflate_wrapper.rs::IsalInflateWrapper` (L102–L311) | Deviated (no gzip header parsing, no `Footer`, no multi-stream) |
| `gzip/isal.hpp` — `readGzipFooter`, `readZlibFooter`, `readDeflateFooter`, `readHeader`, `readIsalHeader` (L429–L560) | (none) | Missing |
| `gzip/isal.hpp` — `inflateWithIsal`, `compressWithIsal` (L587–L689) | other backends in `src/backends/`, not part of parallel SM | (out of scope) |
| `gzip/deflate.hpp` — `deflate::Block<>` (L513–L2005) | `fast_marker_inflate.rs` (entire file) | Deviated (separate code, not a port; uses ISA-L huffman wrapper for hot loop) |
| `gzip/deflate.hpp` — `deflate::Block::readHeader` / `readDynamicHuffmanCoding` | `fast_marker_inflate::decode_dynamic` (L869–L998) | Deviated (own implementation) |
| `gzip/deflate.hpp` — `deflate::Block::read` + `readInternalCompressed` (the hot loop) | `fast_marker_inflate::decode_dynamic_block_full_isal` (L1093+) | Deviated (own loop) |
| `gzip/deflate.hpp` — `deflate::Block::setInitialWindow` | (none — gzippy hands window to ISA-L via `IsalInflateWrapper::set_window` only) | Missing |
| `gzip/deflate.hpp` — circular 64 KiB window buffer (`m_window16`, `PreDecodedBuffer`) | (none — `fast_marker_inflate` keeps decoded chunk in a single growing `Vec<u16>`) | Missing/Different |
| `gzip/deflate.hpp` — `BlockStatistics` (L456–L503) | (none — counters folded into `ChunkStatistics`) | Missing |
| `gzip/format.hpp` — `determineFileTypeAndOffset` (L17–L58) | `single_member.rs::skip_gzip_header` (L60–L105) | Deviated (gzip only, no zlib / bzip2 / deflate / BGZF detection) |
| `gzip/definitions.hpp` — `CompressionType`, `StoppingPoint`, `BlockBoundary`, `MAX_WINDOW_SIZE`, etc. | `inflate_wrapper.rs::DeflateCompressionType`, `StoppingPoints`; constants scattered (e.g. `WINDOW_SIZE` in `fast_marker_inflate.rs:59`) | Deviated (names) |
| `gzip/gzip.hpp` — `readHeader`, `readFooter`, `Footer` | (none in parallel SM; `single_member.rs` reads trailer inline at L127–L141) | Missing |
| `gzip/GzipReader.hpp` | (none) | Missing (out of scope: parallel SM only) |
| `gzip/GzipAnalyzer.hpp` | (none) | Missing (out of scope) |
| `ParallelGzipReader.hpp` | (driver lives in `decompress/mod.rs` + `single_member.rs`; not a port of this class) | Deviated |
| `IndexFileFormat.hpp` (index export) | (none) | Missing |
| `chunkdecoding/Bzip2Chunk.hpp` | (none) | Missing (gzippy is gzip-only) |
| `chunkdecoding/DecompressionError.hpp` — `NoBlockInRange` | `gzip_chunk.rs::ChunkDecodeError` (L51–L56) | Deviated (single enum, no `NoBlockInRange` semantics) |
| (none in rapidgzip) | `parallel/trace.rs` — JSON-lines structured trace under `GZIPPY_LOG_FILE` | Extra |
| (none in rapidgzip) | `parallel/fast_marker_inflate.rs::decode_chunk_bootstrap` (bootstrap-handoff mode) | Extra |
| (none in rapidgzip) | `parallel/fast_marker_inflate.rs::validate_boundary` (start-position probe) | Extra |
| (none in rapidgzip) | `parallel/replace_markers.rs::{replace_markers_avx2, replace_markers_neon}` | Extra |
| (none in rapidgzip) | `parallel/block_finder.rs::validate_fixed_block_prefix` (fixed-Huffman prefilter) | Extra |
| (none in rapidgzip) | `parallel/block_finder.rs::find_blocks_parallel` (multi-threaded block finder) | Extra |
| (none in rapidgzip) | `parallel/single_member.rs::MARKER_PIPELINE_RUNS` + `MARKER_PIPELINE_TEST_LOCK` (routing telemetry) | Extra |

---

## B. Concrete deviations

### B1. Marker encoding (`MapMarkers` vs `replace_markers`)

**rapidgzip** (`MarkerReplacement.hpp:15–46`, `DecodedData.hpp:305–391`)

- `MapMarkers<FULL_WINDOW>::operator()(uint16_t)` returns the literal byte when
  `value <= 0xFF`, **throws** `std::invalid_argument` on `value < MAX_WINDOW_SIZE`
  (a 2-byte code that lies in the "ambiguous" range), and indexes
  `m_window[value - MAX_WINDOW_SIZE]` for `value >= MAX_WINDOW_SIZE`.
- Encoding: index `0` → `window[0]` (i.e. the **oldest** byte of the window).
- `DecodedData::applyWindow` walks `dataWithMarkers` chunk-by-chunk and either
  uses an embedded `MapMarkers` (small chunks) or a precomputed `fullWindow`
  array `[0..256, then window]` for chunks ≥ 128 KiB (fused literal+marker
  table lookup).

**gzippy** (`src/decompress/parallel/replace_markers.rs:7–58`)

- `MARKER_BASE = 32768`. Values `< MARKER_BASE` are literals
  (`*val as u8` directly).
- Values `>= MARKER_BASE` are markers: `offset = val - MARKER_BASE`,
  `byte = window[window.len() - 1 - offset]` — i.e. index `0` → **newest** byte
  of the window. **This is the opposite end of the window from rapidgzip.**
- Out-of-range markers are silently left unresolved (the CRC check catches
  it at the trailer). Rapidgzip throws and the caller catches.

**Correctness impact**: high. The bootstrap decoder
(`fast_marker_inflate::emit_match`) and the resolver
(`replace_markers::replace_markers`) are paired and self-consistent, but the
encoding is incompatible with rapidgzip's `MapMarkers`. Porting any
rapidgzip code that depends on `MapMarkers` semantics (e.g. `getWindowAt`
or `getLastWindow`) cannot be done bit-for-bit without flipping our
encoding to match rapidgzip first.

### B2. `ChunkData::appendDeflateBlockBoundary` semantics

**rapidgzip** (`ChunkData.hpp:455–467`, `GzipChunk.hpp:161–183`):

- `appendDeflateBlockBoundary` is idempotent (dedup on `(encodedOffset,
  decodedOffset)`).
- It updates a flat `blockBoundaries` vector; subchunks are **synthesized
  later** in `ChunkData::split` (`ChunkData.hpp:632–754`) by partitioning
  the boundary list to evenly cover `decodedSize / spacing`.
- `GzipChunk::appendDeflateBlockBoundary` adds a wrapper that **also**
  finalizes the trailing subchunk and starts a new one **only when**
  `subchunks.back().decodedSize >= configuration.splitChunkSize`.

**gzippy** (`chunk_data.rs::append_block_boundary`, L264–L285):

- Dedups on encoded_offset only.
- **Always** pushes a new subchunk — no split-size gating. Comment at L259–L263:
  "Trade-off: per-chunk memory grows by ~50 bytes per block boundary
  crossed."
- `ChunkData::split` does not exist; subchunks are emitted on-line and the
  consumer uses `decoded_offset_for` to find the trim point.

**Correctness impact**: none (consumer sums correctly). Behavioural impact:
gzippy keeps many more subchunks; consumer can match at any boundary.
Porting `split` faithfully would also require porting `setEncodedOffset`
and the boundary deduplication semantics.

### B3. `ChunkData::applyWindow`

**rapidgzip** (`ChunkData.hpp:246–394`):

- Inherits and calls `DecodedData::applyWindow(window)` which (a) flips
  `dataWithMarkers` into `reusedDataBuffers` and (b) inserts `VectorView`s
  into the head of `data`, so the chunk's iteration order is preserved.
- Then loops over every **subchunk**, computing the *per-subchunk* window
  via `getWindowAt(window, decodedOffsetInBlock)` and storing it
  compressed (`SharedWindow`) on `subchunk.window`.
- Also walks subchunks to count `newlineCount` if `configuration.newlineCharacter`
  is set.
- Updates `crc32s.front()` by prepending a fresh `CRC32Calculator` covering
  the bytes that were resolved out of `dataWithMarkers` (and any bytes
  cleaned out by `cleanUnmarkedData` in `finalize`).

**gzippy** (`apply_window.rs:17–65`):

- Calls `replace_markers(&mut chunk.data_with_markers, window)` in place,
  leaving `data_with_markers` as `Vec<u16>` whose values are all `<= 255`
  (verified by `debug_assert`).
- Updates `chunk.crc` by computing the CRC of the resolved bytes in 4 KiB
  scratch chunks (no `Vec<u8>` allocation) and calling
  `resolved_crc.combine(&chunk.crc)`.
- Does **not** touch subchunks, does **not** compute per-subchunk windows,
  does **not** compute newline counts, does **not** convert
  `data_with_markers` into `data` (consumer reads it as `u16 as u8` later).

**Correctness impact**: none for our single-pass write-and-discard usage.
Functional gap: any caller that needs per-subchunk windows for indexing
or random access is unsupported.

### B4. `IsalInflateWrapper::readStream`

**rapidgzip** (`gzip/isal.hpp:253–385`):

- Multi-loop internal iteration; resets `stopped_at` at entry; refills
  buffer 128 KiB at a time from a `BitReader`; handles gzip footer / next
  header in-line when stream ends; returns `pair<size_t, optional<Footer>>`.
- Handles `m_needToReadHeader` flag and emits `END_OF_STREAM_HEADER` stop
  when configured.
- Throws on negative ISA-L errors with a verbose message (file offset,
  bit position, set-window state, hex dump of first 128 bytes).

**gzippy** (`inflate_wrapper.rs:249–311`):

- Multi-loop internal iteration; resets `stopped_at` at entry. **No
  buffer refill**: assumes the entire input slice is materialized in
  memory and seen by ISA-L through `state.next_in` / `state.avail_in`
  set at construction.
- Returns `ReadStreamResult { bytes_written, stopped_at, bit_position,
  finished }`. **No `Footer` return**; multi-member handling is out of
  scope.
- Negative ISA-L returns become `InflateError::{InvalidBlock, InvalidSymbol,
  InvalidLookback, Internal(other)}` — no diagnostic context.

**Correctness impact**: none for parallel SM (single member, raw deflate
only). Functional gap: cannot decode multi-member gzip; cannot resume from
mid-stream with a BitReader.

### B5. Consumer / fetcher architecture

**rapidgzip** (`GzipChunkFetcher.hpp:311–362` + `core/BlockFetcher.hpp`):

- `processNextChunk()` is the orchestrator: it asks `GzipBlockFinder` for
  the next block offset (which may be a known-exact or a partition guess),
  calls `BlockFetcher::get(partitionOffset, blockIndex)` to either pull
  from cache, wait on a prefetched future, or trigger a fresh decode.
- `BlockFetcher` (in `core`) owns: a `Cache` (LRU of completed chunks),
  a `prefetchCache`, a `Prefetcher` (fetching strategy that predicts
  the next N block indexes from access pattern), and a `ThreadPool` that
  runs `decodeBlock(blockOffset, nextBlockOffset)` jobs.
- A chunk decoded at a guessed partition offset has `matchesEncodedOffset`
  invariant tested by `processNextChunk`; if the offset is outside the
  range, the consumer re-`get(blockOffset)` — that's an authoritative
  decode at the exact offset.
- Marker post-processing is enqueued via `submitTaskWithHighPriority` so
  multiple chunks can have their markers resolved in parallel
  (`waitForReplacedMarkers` + `queuePrefetchedChunkPostProcessing`).
- After a chunk resolves, `appendSubchunksToIndexes` updates `m_blockMap`
  (output offset → encoded offset), `m_blockFinder` (insert exact next
  offset), `m_unsplitBlocks`, and the `WindowMap` (per-subchunk).

**gzippy** (`chunk_fetcher.rs:114–671`):

- `drive()` spawns `pool_size` worker threads sharing a single MPSC
  receiver wrapped in a `Mutex` (work queue).
- `consumer_loop()`: maintains `spec[0..n_partitions]: Vec<Option<Receiver>>`,
  pre-submits `prefetch_count` speculative jobs (default `2 * pool_size`),
  drains each partition in order. On speculative miss
  (`matches_encoded_offset` fails OR `decoded_offset_for` is None),
  re-dispatches an authoritative job at the consumer's `expected_start`
  and discards the speculative result.
- `apply_window` is called **synchronously** by the consumer thread for
  the in-order chunk — no parallel marker post-processing.
- No `BlockMap`, no `Prefetcher`, no `Cache`, no `ThreadPool` abstraction,
  no per-subchunk indexing.

**Correctness impact**: none. Functional gap: random-access seeks
unsupported; only sequential decode works. Multi-chunk parallel
marker-resolve unsupported.

### B6. Block finder partitioning

**rapidgzip** (`GzipBlockFinder.hpp:34–end`):

- `GzipBlockFinder` is stateful. Constructor calls
  `determineFileTypeAndOffset` to find the very first block.
- It hands out **partition offsets** (`spacingInBits * idx`) as guesses
  for indexes past the last confirmed boundary.
- `insert(blockOffset)` is called by `appendSubchunksToIndexes` with the
  **actual** end-of-chunk offset, so subsequent `get` calls return exact
  offsets — partition guesses are only used until a real offset is known.
- File-type detection covers GZIP, BGZF (uses `blockfinder::Bgzf`), ZLIB,
  BZIP2, DEFLATE.

**gzippy** (`chunk_fetcher.rs::drive`, L120–L123):

- Fully static partitioning: `partition_offsets[i] = i * split_chunk_size *
  8`, computed once in `drive()`. No on-line confirmation; the consumer
  never tells the partitioner that chunk N actually ended at bit X.
- Speculative dispatch uses `partition_offsets[idx]` directly; on miss,
  the consumer re-dispatches authoritative at the **predecessor's actual
  end** (`expected_start`).

**Correctness impact**: none (the iteration converges via re-dispatch).
Functional gap: no `m_blockMap` or `m_blockFinder` indexes, no support
for resuming an index, no BGZF or ZLIB or BZIP2.

### B7. Block-finder validation strictness

**rapidgzip** (`blockfinder/DynamicHuffman.hpp:166–298`):

- Maintains two bit buffers (`bitBufferForLUT` of `OPTIMAL_NEXT_DEFLATE_LUT_SIZE=15`
  bits, `bitBufferPrecodeBits` of 61 bits) so the LUT skip and the precode check
  can share state without re-reading.
- LUT entries are signed `int8_t`: positive = guaranteed skip to next
  candidate, negative magnitude = skip but partial-verify needed
  (`nextDeflateCandidate` at one bit fewer was 0).
- On positive candidate: calls `checkPrecode(next4Bits, next57Bits)`
  (`CountAllocatedLeaves::checkPrecode`). If passes, builds a
  `PrecodeHuffmanCoding` (`HuffmanCodingReversedBitsCachedCompressed`) and
  decodes the lit+dist code lengths via
  `readDistanceAndLiteralCodeLengths`. EOB-must-be-nonzero is checked, then
  `checkHuffmanCodeLengths<MAX_CODE_LENGTH>` on both lit and dist arrays.

**gzippy** (`block_finder.rs:686–795`):

- Single `BitReader` (no shadow buffers); manual re-seek on each candidate
  (`reader.seek_to_bit(bit_offset + 1)` after rejection).
- LUT entries are unsigned-`i8`: `0` = candidate, positive = skip; the
  negative-sign branch of rapidgzip's encoding is dropped (comment at
  L107–L115: "minus the negative-encoding for lut[i] == 0 case").
- Precode check uses an own `validate_precode` (L164–L216) and an own
  `parse_precode` + `validate_huffman_codes` flow that **fully decodes**
  the lit/dist code lengths and **fully runs** the precode + Kraft checks.
  Acceptance is the same; the LUT layout for leaf counting differs.

**Correctness impact**: should be equivalent (both reject the same
bitstreams). LUT semantics differ; the negative-encoding path is missing
and would matter only for boundary candidates that the 15-bit LUT
partially-but-not-fully rejects — for those, rapidgzip can still derive
a useful skip count from one bit fewer.

### B8. BTYPE handling in block finder

**rapidgzip** `seekToNonFinalDynamicDeflateBlock` only emits **dynamic**
candidates. The `seekToNonFinalUncompressedDeflateBlock` separately emits
**uncompressed** candidates. Fixed Huffman (`BTYPE=01`) is intentionally
not found because the header has no redundancy. Caller alternates between
the two finders inside `decodeChunk` (`GzipChunk.hpp:803–846`).

**gzippy** `BlockFinder::find_blocks` (L674–L680) calls both finders and
merges results — same architecture as rapidgzip's alternating loop, but
done in one shot. Additionally, `validate_fixed_block_prefix` (L449–L501)
exists as a **separate cheap prefilter for BTYPE=01**, which rapidgzip
does not have. The decoder hot path (`fast_marker_inflate::validate_boundary`)
also rejects BTYPE=01 stops by default
(`fast_marker_inflate.rs:174–262` `require_non_fixed_stop`).

**Correctness impact**: none. Behavioral: gzippy emits more candidates
(including BTYPE=01 in some paths); rapidgzip is strictly dynamic-or-uncompressed.

### B9. `finishDecodeChunkWithInexactOffset` non-fixed stop guard

**rapidgzip** (`GzipChunk.hpp:338–345`):

```
case StoppingPoint::END_OF_BLOCK_HEADER:
    if ( ( ( nextBlockOffset >= untilOffset )
           && !inflateWrapper.isFinalBlock()
           && ( inflateWrapper.compressionType() != FIXED_HUFFMAN ) )
         || ( nextBlockOffset == untilOffset ) ) {
        stoppingPointReached = true;
    }
```

**gzippy** (`gzip_chunk.rs:139–148` for `decode_chunk_with_window`, and
`gzip_chunk.rs:412–422` for `finish_decode_chunk_with_inexact_offset`):

```
} else if state.stopped_at == ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER {
    let not_final = state.bfinal == 0;
    let not_fixed = state.btype != 1;
    if last_eob_pos >= until_bits && not_final && not_fixed {
        chunk_end_override = Some(last_eob_pos);
        break;
    }
    ...
}
```

**Difference**: rapidgzip stops at `bit_position` (post-header). gzippy
stops at `last_eob_pos` (pre-header, i.e. the previous END_OF_BLOCK
boundary). gzippy's variant exists because its consumer wants the next
chunk's worker to resume *at the start of a new block's header*; rapidgzip
internally seeks back to the boundary later (the `BitReader` is mutable).

Also: rapidgzip's `nextBlockOffset` is the **last END_OF_BLOCK boundary**
tracked inside the loop (via `inflateWrapper.tellCompressed()` when
`isBlockStart`); gzippy tracks the same value as `last_eob_pos`. Logic
is equivalent.

Also: rapidgzip's `|| (nextBlockOffset == untilOffset)` clause (which
allows stopping exactly at `untilOffset` even at a final or fixed-Huffman
block boundary) is **missing** in gzippy. In practice this rarely fires
because partitions are speculative.

### B10. `decode_or_iterate` vs `decodeChunkWithRapidgzip` direct-try

**rapidgzip** (`GzipChunk.hpp:736–741`):

```
if ( auto result = tryToDecode( { blockOffset, blockOffset } ); result ) {
    return *std::move( result );
}
```

A direct decode at the guessed `blockOffset` is attempted **first**, before
the BlockFinder iteration begins. Rapidgzip then iterates dynamic + uncompressed
candidates in 8 KiB chunks up to 512 KiB.

**gzippy** (`chunk_fetcher.rs:187–244`):

The direct-try-at-start is **deleted** (comment L194–L203 explains: our
bootstrap is too lenient about malformed headers and produced silent
false positives). Instead we go straight to `BlockFinder::find_blocks`
candidates within 512 KiB.

**Correctness impact**: rapidgzip's first guess is essentially free; the
deflate decoder's own validation catches false positives. Gzippy is
stricter and slower-to-first-byte but more robust against bootstrap
false positives.

### B11. Chunk handoff to ISA-L (`cleanDataCount >= MAX_WINDOW_SIZE`)

**rapidgzip** (`GzipChunk.hpp:520–526`):

Inside `decodeChunkWithRapidgzip` the per-chunk `cleanDataCount` is summed
across all decoded blocks. When `cleanDataCount >= deflate::MAX_WINDOW_SIZE`
(32 KiB), control jumps to `finishDecodeChunkWithInexactOffset` (the
ISA-L fast path), seeding the wrapper with `result.getLastWindow({})` (the
chunk's own last 32 KiB).

If `getLastWindow({})` is called and the window still contains marker
bytes (because the trailing 32 KiB straddles markers), `MapMarkers`
**throws** inside the iteration; `tryToDecode` catches the exception and
moves to the next BlockFinder candidate.

**gzippy** (`fast_marker_inflate.rs::decode_chunk_bootstrap`, L460–L551,
and `gzip_chunk.rs::finish_decode_chunk_with_inexact_offset`, L240–L268):

Uses a `CleanTailTracker` that tracks only the **trailing** clean bytes
(rapidgzip uses cumulative `cleanDataCount`). Handoff happens when
`tracker.trailing_clean_bytes >= WINDOW_SIZE` AT a block boundary; the
last 32 KiB is then cast to `Vec<u8>` (with `assert!(v < MARKER_BASE)`)
and used as the ISA-L dict.

When no such window accumulates, the chunk returns marker-only
(`clean_window = None`) and the entire output is `data_with_markers`;
`apply_window` runs over the whole chunk in the consumer.

**Correctness impact**: none in isolation. Behavioural: rapidgzip's
"cumulative clean" handoff fires earlier (any 32 KiB of clean output,
not necessarily contiguous at the tail), so a wider variety of streams
get the ISA-L bulk path. Gzippy's "trailing 32 KiB clean" requirement
is stricter and forces marker-only chunks more often.

### B12. `WindowMap` blocking and storage

**rapidgzip** (`WindowMap.hpp:19–186`):

- `std::map<size_t, SharedWindow>` protected by `std::mutex`; **no condvar**.
- Insertions try `emplace_hint(end())` first (O(1) for ordered inserts).
- Windows are **`CompressedVector`** (`emplace(offset, window, compressionType)`),
  saving memory at the cost of decompress-on-access.
- Supports `operator==`, `releaseUpTo(offset)`, exposes `data()` for
  external lock acquisition.

**gzippy** (`window_map.rs:23–96`):

- `BTreeMap<usize, Arc<[u8; 32768]>>` protected by `Mutex` **and a `Condvar`**.
- `get_or_wait(offset, timeout)` blocks until the window appears or the
  deadline expires. Used by workers to wait for the predecessor's window.
- Windows are uncompressed `Arc<[u8; 32768]>` (constant 32 KiB).
- No `releaseUpTo`; entries live for the duration of `drive()`.

**Correctness impact**: none. Behavioural: gzippy's condvar enables a
"fast path" worker (`worker_loop:269–280`) that blocks until the
predecessor's window is published. Rapidgzip's consumer is the
synchronization point instead.

### B13. Worker decode pathways

**rapidgzip** has effectively **one** worker entry: `decodeBlock` (the
override), which routes through `GzipChunk<T>::decodeChunk`. That function
internally picks between three modes based on whether `initialWindow`
exists and whether `untilOffsetIsExact`:

1. `(initialWindow, untilOffsetIsExact)` → `decodeChunkWithInflateWrapper`
   (ISA-L, exact stop).
2. `(initialWindow, !exact)` → `decodeChunkWithRapidgzip` (which internally
   delegates immediately to `finishDecodeChunkWithInexactOffset` if ISA-L
   is enabled).
3. `(no window, ...)` → `decodeChunkWithRapidgzip` calls `tryToDecode` at
   the guessed offset, then iterates BlockFinder candidates.

**gzippy** has **two** worker code paths chosen by `worker_loop` based on
whether `window_map.get_or_wait` returns Some/None:

1. Fast path (window known) → `decode_chunk_with_window` (ISA-L only,
   inexact stop). Equivalent to rapidgzip case 2.
2. Slow path (no window) → `decode_or_iterate` →
   `finish_decode_chunk_with_inexact_offset` per candidate.
   Equivalent to rapidgzip case 3 but with two structural differences:
   - The marker bootstrap (`fast_marker_inflate::decode_chunk_bootstrap`)
     is a separate Rust decoder, not a `deflate::Block` call.
   - The direct-try-at-guessed-offset step is deleted (see B10).

The "exact stop" variant (rapidgzip case 1) is exposed in
`gzip_chunk.rs::decode_chunk_with_inflate_wrapper` (L182–L219) but
**no production call site uses it** — only the tests. The driver
always treats partition offsets as inexact.

### B14. Authoritative re-dispatch / `setEncodedOffset`

**rapidgzip**: after a chunk completes, `processNextChunk` calls
`chunkData->setEncodedOffset(*nextBlockOffset)` to **correct** the chunk's
`encodedOffsetInBits` from the speculative guess to the real offset. The
chunk's `setEncodedOffset` (`ChunkData.hpp:601–629`) re-anchors the
`encodedOffsetInBits` and trims the first subchunk's
`encodedOffset`/`encodedSize` accordingly.

**gzippy**: never corrects the chunk after the fact. Instead, on mismatch
(`!c.matches_encoded_offset(expected_start) || c.decoded_offset_for(expected_start).is_none()`),
the consumer **discards** the speculative chunk and submits an
authoritative job at `expected_start` (`chunk_fetcher.rs:550–574`). On a
**hit**, the consumer uses `decoded_offset_for(expected_start)` to compute
`trim_bytes` and skips that many leading bytes from `data_with_markers`
or `data` (`chunk_fetcher.rs:576–644`).

**Correctness impact**: none. Behavioural: gzippy may re-decode chunks
whose start happens to be near (but not exactly) the partition seed.
Rapidgzip avoids that re-decode by trusting `matchesEncodedOffset`'s
range check.

### B15. Per-chunk Footer / multi-stream support

**rapidgzip**: `ChunkData` carries `std::vector<Footer> footers` and a
`std::vector<CRC32Calculator> crc32s` (one per stream segment). The
`appendFooter` machinery (`ChunkData.hpp:472–489`) is invoked when the
inflater hits an `END_OF_STREAM` stopping point. `decodeChunkWithRapidgzip`
loops over potentially multiple gzip streams per chunk
(`GzipChunk.hpp:468–654`).

**gzippy**: single per-chunk `crc32fast::Hasher`. No `Footer` type, no
multi-stream support. The driver assumes the input is exactly one gzip
stream and strips the trailer at `single_member.rs:127–141`.

**Correctness impact**: cannot decode multi-stream gzips through the
parallel SM path. (Routing in `decompress/mod.rs` sends multi-member
input elsewhere.) Functional gap.

---

## C. Missing pieces (rapidgzip has, gzippy doesn't)

1. **`HuffmanCodingDoubleLiteralCached`** (`huffman/HuffmanCodingDoubleLiteralCached.hpp`):
   multi-symbol pure-C++ decoder. rapidgzip uses it only when ISA-L is
   unavailable; we don't have a fallback that does multi-symbol decode at
   all.

2. **`HuffmanCodingShortBitsCached*` family** (4 variants): rapidgzip's
   fallback Huffman decoders when ISA-L is off. Gzippy uses
   `ConsumeFirstTable` (from `inflate/consume_first_table.rs`) for both
   precode and lit/dist in the non-ISAL slow path.

3. **`deflate::Block<>` (`gzip/deflate.hpp:513–end`)**: the ~1500-line
   class encapsulating a single block decode. Has `m_window16` (the
   2×32 KiB circular pre-decode buffer that holds u16s when the initial
   window is unknown — exactly the "markers in band" mechanism), `read`
   that decodes up to `nMaxToDecode` bytes, `setInitialWindow` to seed
   the dict, `eob()` / `eos()` / `eof()` block-state queries,
   `BlockStatistics` counters, `Backreference` tracking. Gzippy's
   `fast_marker_inflate.rs` is a separate implementation of similar
   functionality but **not** a port — it lacks the circular window, the
   marker convention is different (B1), the public API is procedural
   rather than stateful, and many features (BFINAL multi-stream loop,
   block-statistics, backreference recording) are absent.

4. **`setInitialWindow(window)`** on the deflate block decoder
   (`deflate.hpp:606`): allows resuming decode of a chunk with an
   already-known 32 KiB window so that the chunk emits real bytes
   directly instead of markers. Gzippy's marker decoder cannot be seeded
   with a window — when a window is known, gzippy hands off to ISA-L
   via `IsalInflateWrapper::set_window` (in `decode_chunk_with_window`).

5. **`getLastWindow({})` raising on markers** (`DecodedData.hpp:394–488`
   via `MapMarkers::operator()`): the mechanism rapidgzip uses to detect
   "trailing 32 KiB has markers, can't hand off yet" — `MapMarkers` throws,
   `tryToDecode` catches, the iterator moves to the next candidate.
   Gzippy substitutes a `CleanTailTracker` check that returns
   `clean_window: None` without throwing; the chunk is then marker-only
   and the entire decoder bootstrap runs.

6. **`BlockFetcher` with `Cache` + `Prefetcher` + `ThreadPool`**
   (`core/BlockFetcher.hpp`). Gzippy has ad-hoc `mpsc::channel` +
   `spec[]` ring + `std::thread::scope`. Functional consequences:
   - No LRU cache; speculative chunks are dropped after one consumer
     pass.
   - No fetching strategy (linear access pattern only).
   - No splitting of chunks into cache entries.
   - No retry / backoff / waiting on block finder.

7. **`GzipBlockFinder` (the partitioner)**
   (`GzipBlockFinder.hpp`). Gzippy's static partition table can't be
   updated mid-decode and can't take advantage of known exact boundaries
   after the first chunk completes.

8. **`BlockMap` (output-offset → encoded-offset index)**. Required for
   any random-access seek operation. Gzippy doesn't index decoded output
   at all.

9. **`IndexFileFormat`**: rapidgzip can export a seekable index that
   another process can use to start decoding at any block. Gzippy can't.

10. **Parallel marker post-processing**
    (`GzipChunkFetcher::queueChunkForPostProcessing` +
    `waitForReplacedMarkers`): rapidgzip can resolve markers on multiple
    chunks in parallel using the thread pool. Gzippy's consumer is the
    only thread that calls `apply_window`.

11. **Per-subchunk window** (`ChunkData::Subchunk::window`,
    `appendSubchunksToIndexes` L429–L458): rapidgzip stores a window at
    every subchunk boundary so seeks can land mid-chunk. Gzippy stores
    only the chunk-end window in the WindowMap.

12. **Window sparsity / window compression**
    (`ChunkData::windowCompressionType`, `CompressedVector`, the
    `determineUsedWindowSymbolsForLastSubchunk` flow): rapidgzip can
    detect that many bytes of a window are never referenced and replaces
    them with zeros before compressing the window. Gzippy stores raw
    32 KiB.

13. **`Footer`, `appendFooter`, `crc32s` vector, multi-stream support**
    (see B15).

14. **File-format detection** (`gzip/format.hpp::determineFileTypeAndOffset`):
    GZIP, BGZF, ZLIB, BZIP2, DEFLATE. Gzippy's `skip_gzip_header` handles
    GZIP only (BGZF detection lives elsewhere in `decompress/bgzf.rs`).

15. **Footer reading in `IsalInflateWrapper`**
    (`gzip/isal.hpp:429–470`, `readGzipFooter`/`readZlibFooter`/
    `readDeflateFooter`): gzippy's wrapper exits at the deflate stream
    end without consuming a footer.

16. **Statistics + `--verbose` profile output**
    (`GzipChunkFetcher.hpp:114–197`, `BlockFetcher::Statistics`):
    cache hit rate, prefetch efficiency, decode duration per stage, pool
    efficiency, false positive count, etc. Gzippy has only the JSON
    trace (`trace.rs`), which is unstructured for aggregation.

17. **`NoBlockInRange` exception** (`chunkdecoding/DecompressionError.hpp`):
    distinct error type that drives `GzipChunkFetcher::getBlock` to retry
    with the exact offset. Gzippy returns the generic
    `ChunkDecodeError::ExactStopMissed` and the consumer handles
    re-dispatch through a different path.

18. **`finalizeChunk` subchunk merge** (`GzipChunk.hpp:142–153`): when
    the trailing subchunk is smaller than `minimumSplitChunkSize()`
    (= `splitChunkSize / 4`), rapidgzip merges it back into its
    predecessor. Gzippy never merges.

19. **`ChunkData::split` and `setEncodedOffset`** (see B2, B14).

20. **`appendSubchunksToIndexes` cascade**: after a chunk resolves,
    rapidgzip writes per-subchunk into `BlockMap`, `BlockFinder`
    (`insert`), `WindowMap`, `m_unsplitBlocks`, and runs all
    `indexFirstSeenChunkCallbacks`. Gzippy writes one window into
    `WindowMap` and discards the chunk.

---

## D. Extra pieces (gzippy has, rapidgzip doesn't)

1. **`fast_marker_inflate.rs`** is a from-scratch Rust marker decoder.
   It is **not** a port of `deflate::Block<>`; it shares the abstract
   idea of emitting "markers" for out-of-window back-references, but the
   marker encoding (B1), the storage (single `Vec<u16>` rather than a
   circular `m_window16`), the public API (procedural functions, not a
   stateful block class), and the bootstrap-handoff strategy
   (`CleanTailTracker`) are all original. Production decoders should
   eventually replace this with a `deflate::Block` port if a faithful
   port is desired.

2. **`fast_marker_inflate::validate_boundary`** (L174–L262): a
   trial-decode probe that confirms a candidate bit offset by decoding
   at least `min_blocks` blocks and `min_output_bytes` bytes without
   error. Rapidgzip doesn't validate up-front; it instead lets the
   decoder fail naturally (`tryToDecode`'s catch). The probe exists in
   gzippy because the marker bootstrap is unreliable on false positives
   (over-emits garbage).

3. **`fast_marker_inflate::decode_chunk_bootstrap`** + `BootstrapResult`
   (L460–L551): the explicit "decode until 32 KiB clean tail at a block
   boundary, then return so the caller can hand off to ISA-L" return
   contract. Rapidgzip's equivalent is the `cleanDataCount` heuristic
   wired inside `decodeChunkWithRapidgzip` and the `getLastWindow({})`
   throw-on-marker control flow — there is no separate function for it.

4. **`replace_markers_avx2` and `replace_markers_neon`**: SIMD
   marker-resolution kernels. Rapidgzip's `MapMarkers` is scalar (with a
   fused `fullWindow` table for the large-chunk branch but no SIMD).

5. **`block_finder::validate_fixed_block_prefix`** (L449–L501): a
   ~50 ns prefilter that decodes the first 2 fixed-Huffman symbols and
   rejects obviously-bad starts. Rapidgzip has no equivalent (it
   doesn't emit fixed-Huffman candidates).

6. **`block_finder::find_blocks_parallel`** (L1029–L1081): runs the
   block finder across N threads on disjoint slices. Rapidgzip's
   `seekToNonFinalDynamicDeflateBlock` is invoked single-threaded from
   the worker.

7. **`parallel/trace.rs`** (L1–L143): JSON-lines structured tracing,
   keyed by `GZIPPY_LOG_FILE`. Rapidgzip prints aggregate statistics
   on `~GzipChunkFetcher()` (when `m_showProfileOnDestruction`) but
   has no per-event log.

8. **`MARKER_PIPELINE_RUNS` + `MARKER_PIPELINE_TEST_LOCK`**
   (`single_member.rs:41–47`): test-only routing telemetry to catch
   silent fallback to libdeflate. Rapidgzip-specific concern: the C++
   code has no "fall back to a different decoder" logic, so doesn't
   need this.

9. **`WindowMap::get_or_wait`** with `Condvar`: rapidgzip's `WindowMap`
   has no blocking variant; the consumer thread is the only reader, and
   it is the producer too.

10. **`ChunkDecodeError::ExactStopMissed { requested, actual }`** —
    explicit "wrapper stopped at the wrong bit position" variant.
    Rapidgzip throws a `runtime_error` with a string.

11. **The `gzippy-parallel` "GZ" FEXTRA path** (out of scope here;
    lives in `decompress/bgzf.rs`) is a gzippy-only format.

12. **Routing-layer fallback** (`decompress/mod.rs` not shown here):
    when the parallel SM path returns `ParallelError::TooSmall` or
    `ParallelError::DecodeFailed`, the caller silently falls back to
    libdeflate / multi-member / scan-inflate. Rapidgzip has no such
    fallback; it commits to the parallel path.

---

## E. Per-file deep dive

### `src/decompress/parallel/single_member.rs` (256 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `MIN_PARALLEL_SIZE = 4 MiB`, `TARGET_COMPRESSED_CHUNK_BYTES = 4 MiB` | L31–L33 | gzippy-original (rapidgzip uses 4 MiB as `splitChunkSize` per `GzipChunkFetcher.hpp:706`) |
| `MARKER_PIPELINE_RUNS` / `MARKER_PIPELINE_TEST_LOCK` | L41–L47 | gzippy-original (test infrastructure) |
| `skip_gzip_header` | L60–L105 | gzippy-original; rapidgzip uses `gzip::readHeader` in `gzip/gzip.hpp` |
| `decompress_parallel` driver | L107–L188 | gzippy-original (rapidgzip has `ParallelGzipReader` for the same role) |
| Trailer parsing (CRC + ISIZE) | L127–L141 | gzippy-original |
| `ParallelError` enum | L192–L225 | gzippy-original |

### `src/decompress/parallel/chunk_fetcher.rs` (730 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `DecodeJob` struct + worker pool | L96–L172 | gzippy-original (no equivalent struct in rapidgzip's `ThreadPool`) |
| `BOUNDARY_SEARCH_RADIUS_BYTES = 512 KiB` | L80–L81 | port of `512_Ki * BYTE_SIZE` cap in `GzipChunk.hpp:811` |
| `WINDOW_WAIT_TIMEOUT = 50ms` | L88–L89 | gzippy-original (rapidgzip blocks indefinitely on prefetch futures) |
| `drive()` (driver entry) | L114–L177 | gzippy-original |
| `decode_or_iterate` | L187–L244 | port of `tryToDecode` + the BlockFinder loop in `GzipChunk.hpp:712–852`, **minus** the direct-try-at-start (see B10) |
| `worker_loop` | L246–L373 | gzippy-original architecture; fast/slow split via `WindowMap::get_or_wait` is not present in rapidgzip |
| `consumer_loop` speculative ring | L375–L671 | port of `GzipChunkFetcher::processNextChunk` + parts of `BlockFetcher::get` |
| Authoritative re-dispatch | L550–L574 | port of the `if (!chunkData || !matchesEncodedOffset(...))` retry in `GzipChunkFetcher::getBlock:646–662` |
| Synchronous `apply_window` | L596–L612 | deviation from `queueChunkForPostProcessing` (B5) |
| Output write + CRC combine | L614–L645 | gzippy-original (rapidgzip uses `writeAll` / `vmsplice` in `ChunkData.hpp:794–825`) |

### `src/decompress/parallel/gzip_chunk.rs` (588 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `ChunkDecodeError` | L51–L68 | gzippy-original; rapidgzip throws |
| `ALLOCATION_CHUNK_SIZE = 128 KiB` | L72 | port of `ChunkData.hpp:65` |
| `decode_chunk_with_window` (fast path) | L86–L165 | port of `finishDecodeChunkWithInexactOffset` `GzipChunk.hpp:280–410` when window is known; deviates per B9 |
| `decode_chunk_with_inflate_wrapper` (exact stop) | L182–L219 | faithful port of `decodeChunkWithInflateWrapper` `GzipChunk.hpp:190–268` (subset: no decoded-size check, no `Footer`) |
| `finish_decode_chunk_with_inexact_offset` (slow path) | L234–L462 | port of the `decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset` combination, with the bootstrap done in `fast_marker_inflate::decode_chunk_bootstrap` rather than `deflate::Block`. Uses raw `isal_sys` directly instead of `IsalInflateWrapper` — comment at L276–L281 explains why (preserves exact known-good shape). |

### `src/decompress/parallel/chunk_data.rs` (474 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `Subchunk` | L29–L37 | port of `ChunkData::Subchunk` `ChunkData.hpp:115–145` (no `newlineCount`, `usedWindowSymbols`) |
| `ChunkStatistics` | L43–L50 | port of `ChunkData::Statistics` `ChunkData.hpp:147–180` (reduced to 6 fields) |
| `ChunkConfiguration` | L55–L79 | port of `ChunkData::Configuration` `ChunkData.hpp:97–113` (no `fileType`, `windowCompressionType`, `windowSparsity`, `newlineCharacter`) |
| `ChunkData` struct | L86–L128 | port of `ChunkData` + `DecodedData` flattened |
| `matches_encoded_offset` | L163–L165 | faithful port of `ChunkData::matchesEncodedOffset` L396–L403 |
| `last_32kib_window` | L173–L207 | port of `DecodedData::getLastWindow` L394–L488 with deviations: returns `Option<[u8;32768]>` instead of throwing; doesn't use `MapMarkers`; doesn't combine with `previousWindow` |
| `append_markered` / `append_clean` | L222–L249 | port of `DecodedData::append(MarkerVector)` and `append(DecodedDataView)` (L109–L290) |
| `append_block_boundary` (always emits subchunk) | L264–L285 | deviates from `appendDeflateBlockBoundary` (B2) |
| `finalize` | L293–L301 | port of `ChunkData::finalize` L417–L449 (no `cleanUnmarkedData`, no `split`, no subchunk merge) |
| `decoded_offset_for` | L308–L316 | gzippy-original (rapidgzip uses `BlockMap::findDataOffset` for the analogous lookup) |

### `src/decompress/parallel/apply_window.rs` (164 lines)

Implements `chunk_data.rs::apply_window`. Single-window only; see B3. Faithful
to the *idea* of `DecodedData::applyWindow` but does not invoke `MapMarkers`
and does not handle per-subchunk windows.

### `src/decompress/parallel/window_map.rs` (172 lines)

Single-class file. The class is structurally close to rapidgzip's `WindowMap`
(`std::map<size_t, SharedWindow>` ⇔ `BTreeMap<usize, Arc<[u8;32768]>>`) but
adds a `Condvar` and `get_or_wait`. See B12. Missing `releaseUpTo`,
`operator==`, `data()` lock-export, `CompressedVector`.

### `src/decompress/parallel/block_finder.rs` (1318 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `LUT_BITS = 15` | L42 | port of `OPTIMAL_NEXT_DEFLATE_LUT_SIZE = 15` (`DynamicHuffman.hpp:145`) |
| `is_deflate_candidate_n` | L62–L88 | faithful port of `isDeflateCandidate<bitCount>` (`DynamicHuffman.hpp:39–79`) |
| `next_deflate_candidate` | L95–L103 | faithful port of `nextDeflateCandidate<bitCount>` (`DynamicHuffman.hpp:85–98`) |
| `generate_deflate_lut` | L109–L115 | deviates from `NEXT_DYNAMIC_DEFLATE_CANDIDATE_LUT` (`DynamicHuffman.hpp:113–124`): missing negative-encoding branch (see B7) |
| `validate_precode` | L164–L216 | gzippy-original implementation; `CountAllocatedLeaves::checkPrecode` uses a 4-precode-at-a-time 4 KiB LUT, gzippy uses a 4-precode 12-bit LUT differently laid out |
| `BitReader` | L222–L311 | gzippy-original |
| `BlockBoundary` | L317–L329 | port of `BlockBoundary` (`gzip/definitions.hpp:121–131`) with extra fields (`valid`, `hlit`, `hdist`, `hclen`) |
| `validate_fixed_block_prefix` | L449–L501 | **Extra** — no rapidgzip counterpart |
| `find_uncompressed_blocks` | L562–L665 | faithful port of `seekToNonFinalUncompressedDeflateBlock` (`Uncompressed.hpp:21–95`) |
| `find_dynamic_blocks` | L686–L795 | port of `seekToNonFinalDynamicDeflateBlock` (`DynamicHuffman.hpp:166–298`); uses different bit reader; LUT signedness differs |
| `parse_precode` + `is_valid_huffman_lengths` + `validate_huffman_codes` | L818–L948 | own implementation matching the same checks rapidgzip runs inline (precode → readDistanceAndLiteralCodeLengths → `checkHuffmanCodeLengths`) |
| `find_blocks_parallel` | L1029–L1081 | **Extra** |

### `src/decompress/parallel/inflate_wrapper.rs` (430 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `StoppingPoints` | L33–L60 | port of `StoppingPoint` enum (`gzip/definitions.hpp:92–100`) — values match (1, 2, 4, 8) |
| `ReadStreamResult` | L62–L77 | gzippy-original (rapidgzip returns `pair<size_t, optional<Footer>>`) |
| `DeflateCompressionType` | L80–L86 | port of `CompressionType` enum (`gzip/definitions.hpp:61–67`) |
| `InflateError` | L88–L97 | gzippy-original (rapidgzip throws) |
| `IsalInflateWrapper::new` | L116–L139 | port of `IsalInflateWrapper` ctor + `initStream` (`isal.hpp:32–43, 215–225`); deviates by not handling `BitReader` — assumes direct slice ownership |
| `set_window` | L145–L163 | port of `setWindow` (`isal.hpp:52–59`) — defaults to all-zero on empty |
| `set_stopping_points` / `stopped_at` / `is_final_block` / `btype` | L170–L215 | faithful port of `setStoppingPoints` / `stoppedAt` / `isFinalBlock` / `compressionType` (`isal.hpp:75–109`) |
| `tell_compressed` | L222–L232 | port of `tellCompressed` (`isal.hpp:69–74`) but computed directly from `avail_in`/`read_in_length` instead of subtracting from the `BitReader`'s tell |
| `read_stream` | L249–L310 | port of `readStream` (`isal.hpp:253–385`) with: no refillBuffer (entire input is one slice), no `Footer` handling, no header re-reading, no `setStartWithHeader`, no diagnostic dump on error |

### `src/decompress/parallel/isal_huffman.rs` (412 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `LIT_LEN_ELEMS = 514`, `MAX_LIT_LEN_COUNT = 23` | L41–L44 | port of constants in `HuffmanCodingISAL.hpp:24–28` |
| `LEN_EXTRA_BIT_COUNT` table | L62–L64 | port of `len_extra_bit_count` (`HuffmanCodingISAL.hpp:30–35`) |
| `IsalLitLenCode` struct + `from_lengths` + `rebuild_from` | L80–L168 | port of `HuffmanCodingISAL::initializeFromLengths` (`HuffmanCodingISAL.hpp:38–74`); allocations heap-boxed (rapidgzip stack-allocates `std::array<huff_code, 514>` since it lives in a `Block`) |
| `IsalLitLenCode::decode` | L181–L221 | faithful port of `HuffmanCodingISAL::decode` (`HuffmanCodingISAL.hpp:94–183`) — bit layout constants match |
| `IsalDistCode` + `rebuild_from` + `decode` | L243–L336 | faithful port of `HuffmanCodingDistanceISAL` (`HuffmanCodingDistanceISAL.hpp:20–118`) |
| `with_thread_litlen` / `with_thread_litlen_dist` | L347–L379 | gzippy-original (thread-local table reuse). Rapidgzip's equivalent is the `deflate::Block` instance which is heap-allocated once and reused across blocks. |

### `src/decompress/parallel/fast_marker_inflate.rs` (2351 lines)

This file is **not a port of `deflate::Block<>`**, and the file's
module-doc says so explicitly (`L36–L48`: "Reuse: Bit buffer borrowed
from `consume_first_decode::Bits`. Canonical Huffman table build is
implemented locally"). It is a gzippy-original implementation of the
same abstract idea (marker-emitting deflate decode), with a separate
encoding (B1) and a separate API.

Components:

| Item | Lines | Origin |
|------|-------|--------|
| `LENGTH_BASE` / `LENGTH_EXTRA` / `DISTANCE_BASE` / `DISTANCE_EXTRA` / `CL_ORDER` | L68–L89 | port of RFC 1951 tables (also at `gzip/RFCTables.hpp` in rapidgzip) |
| `decode_chunk_markers` | L142–L143 | gzippy-original entry |
| `validate_boundary` | L174–L262 | **Extra** (B7, D2) |
| `decode_chunk_markers_bounded` | L274–L347 | gzippy-original (rapidgzip's `decodeChunkWithRapidgzip` uses `untilOffset` for the same role) |
| `decode_chunk_bootstrap` + `BootstrapResult` | L361–L551 | **Extra** (D3); semantically replaces rapidgzip's `cleanDataCount` + `getLastWindow({})` flow |
| `CleanTailTracker` | L387–L437 | gzippy-original; substitute for rapidgzip's cumulative `cleanDataCount` |
| `decode_chunk_markers_continuing` | L571–L615 | **Extra** (used by the slow path when ISA-L handoff fails) |
| `decode_loop` | L622–L748 | gzippy-original (rapidgzip's analogue is `decodeChunkWithRapidgzip`'s while loop) |
| `decode_stored` / `decode_fixed` / `decode_dynamic` | L767–L998 | own implementations of RFC 1951 §3.2.4 / §3.2.6 / §3.2.7; rapidgzip's are in `deflate::Block::readInternalUncompressed`, the inlined fixed-Huffman path, and `readDynamicHuffmanCoding` |
| `decode_huffman_block_isal` | L1005–L1083 | port of `deflate::Block::readInternalCompressed` (`deflate.hpp:1514–1582`) using `IsalLitLenCode` for lit/len and our `ConsumeFirstTable` for distance |
| `decode_dynamic_block_full_isal` | L1093+ | **literal port** of the same inner loop using ISA-L for both lit/len and distance (matches `HuffmanCodingISAL` + `HuffmanCodingDistanceISAL` paired use in rapidgzip) |
| `decode_huffman_block` | L1207–L1293 | pure-Rust fallback when ISA-L is unavailable |
| `emit_match` | L1294+ | gzippy-original; implements the marker convention from B1 |

### `src/decompress/parallel/replace_markers.rs` (250 lines)

| Item | Lines | Origin |
|------|-------|--------|
| `MARKER_BASE = 32768` | L24 | gzippy-original encoding (B1) |
| `replace_markers` dispatcher | L30–L46 | gzippy-original |
| `replace_markers_scalar` | L49–L58 | logical equivalent of `MapMarkers<true>::operator()` but with the opposite-end encoding (B1) |
| `replace_markers_avx2` | L62–L109 | **Extra** (D4) |
| `replace_markers_neon` | L112–L156 | **Extra** (D4) |
| `u16_to_u8` | L161–L166 | gzippy-original |

### `src/decompress/parallel/trace.rs` (143 lines)

Entirely gzippy-original. Rapidgzip has aggregate statistics in
`~GzipChunkFetcher`, no per-event trace.

### `crates/isal-sys-patched/`

Vendored ISA-L with rapidgzip-style patches that expose
`isal_internals::{huff_code, make_inflate_huff_code_lit_len,
set_and_expand_lit_len_huffcode, make_inflate_huff_code_dist, set_codes}`
through FFI. These are the same internal ISA-L functions that rapidgzip
includes via its own `<igzip_lib.h>` patches (see `gzip/isal.hpp` and
`huffman/HuffmanCodingISAL.hpp` for the `HuffmanCodingISAL.hpp` direct
calls). Faithful to rapidgzip's patched-ISA-L approach.

---

## F. Closure plan

The plan below is in **priority order** for restoring structural fidelity
to rapidgzip. Each step is sized to land as one PR. Per-step LOC estimates
exclude tests.

**Status snapshot** (after the structural-gap-closure session):
- ✅ Step 1 (e335a28): marker encoding flipped to MapMarkers
- ✅ Step 2 (78ba3d8): Footer + crc32s vector
- ✅ Step 3 (d3ae688): per-subchunk window emplacement
- ✅ Step 4 (08e5224): ChunkData::setEncodedOffset
- ✅ Step 5 (c808396): BlockMap
- ◐ Step 6: BlockFetcher cache + Prefetcher (~600 LOC, partial)
  - ✅ 6a (45cb6db): Cache + LeastRecentlyUsed + CacheStatistics
  - ✅ 6b (b3bf3dc): FetchingStrategy + FetchNextFixed + FetchNextAdaptive
  - ⏳ 6c: BlockFetcher orchestration (cache + prefetch_cache + strategy
    + statistics integration, including unsplit_blocks remap, thread
    pool, get/getIfAvailable APIs)
- ⏳ Step 7: deflate::Block port (~1500 LOC, not started — biggest)
- ✅ Step 8 (f2f2ae6): DecodedData::cleanUnmarkedData
- ✅ Step 9 (bc8c50c): appendSubchunksToIndexes BlockMap insertion
- ✅ Step 10 (0728eca): gzip readHeader + readFooter
- ✅ Step 11 (9f5b291): IsalInflateWrapper footer + multi-stream methods
- ✅ Step 12 (f6c1e4f): fetcher statistics
- ✅ Step 13 (87fba04): CompressedVector
- ⏳ Step 14: IndexFileFormat (~400 LOC, optional)
- ✅ Step 15 (f6c1e4f): ChunkData::split
- ⏳ Step 16: HuffmanCodingDoubleLiteralCached (~350 LOC, optional)

12 of 16 steps landed (75%). Remaining: Step 6 + 7 (the two largest)
and Steps 14 + 16 (both marked optional).

### Step 1. Align marker encoding with rapidgzip's `MapMarkers` (~150 LOC)

**Why**: B1 / D4. The current encoding (`MARKER_BASE + offset`, indexing
from the **newest** byte) is incompatible with any directly-ported
rapidgzip code that uses `MapMarkers`. Until this flips,
`getLastWindow({})`, `applyWindow`'s per-subchunk path,
`cleanUnmarkedData`, and many other rapidgzip primitives cannot be
ported without manual re-derivation.

Change:

- In `replace_markers.rs`, change the encoding so a u16 value `v`:
  - `v <= 255` → literal byte `v as u8`.
  - `v >= 32768` → `window[v - 32768]` (i.e. index from **oldest** byte).
  - `256 <= v < 32768` → invalid (rapidgzip throws; gzippy can debug_assert).
- In `fast_marker_inflate::emit_match` (L1294+), update the marker
  emission to use the new encoding. Pure rewrite of one function.
- Update SIMD kernels in `replace_markers.rs` (the indexing changes:
  `window[offset]` instead of `window[window.len() - 1 - offset]`).
- Update tests in `replace_markers.rs`, `apply_window.rs`,
  `fast_marker_inflate.rs`.

After this, future steps can re-use the `MapMarkers` semantics directly.

### Step 2. Port `gzip::Footer`, multi-stream loop, `appendFooter`, `crc32s` vector (~250 LOC)

**Why**: B15 / C13. Required for multi-member input through the parallel
SM path. Touches:

- New `chunk_data.rs::Footer` struct mirroring `gzip::Footer` (crc32 +
  uncompressedSize + blockBoundary).
- `chunk_data.rs::ChunkData`: replace single `crc: crc32fast::Hasher`
  with `crc32s: Vec<crc32fast::Hasher>` and a `footers: Vec<Footer>`.
- `apply_window.rs::apply_window`: use only `crc32s.front()` for the
  marker-resolved CRC.
- `chunk_fetcher.rs::consumer_loop`: combine all `crc32s` from each
  chunk into the driver's total CRC.
- `gzip_chunk.rs::finish_decode_chunk_with_inexact_offset`: detect
  `END_OF_STREAM`, read footer, append, restart on the next header
  (mirrors `GzipChunk.hpp:602–653`).
- `single_member.rs::decompress_parallel`: stop assuming a single
  trailer.

### Step 3. Port `ChunkData::Subchunk::window` + `appendSubchunksToIndexes` window emplacement (~150 LOC)

**Why**: C11. Required for any random-access seek and for
`finalizeWindowForLastSubchunk`. Touches `chunk_data.rs`, `apply_window.rs`
(populate per-subchunk windows from the resolved chunk output), and
`chunk_fetcher.rs::consumer_loop` (publish each subchunk window into
`WindowMap`).

### Step 4. Port `ChunkData::setEncodedOffset` and re-anchor speculative chunks (~80 LOC)

**Why**: B14. After this, the consumer can accept a chunk whose
`encoded_offset_bits` was a partition seed and correct it to the real
offset post-decode, instead of always re-dispatching authoritatively.
Touches `chunk_data.rs` (new method) and `chunk_fetcher.rs::consumer_loop`
(call `chunk.set_encoded_offset(expected_start)` on speculative hits and
recompute `trim_bytes` as the difference, not by subchunk lookup).

### Step 5. Port `BlockMap` (output-offset → encoded-offset) and `GzipBlockFinder` (~400 LOC)

**Why**: C7, C8. Required for any future random-access seek path and
for a faithful `GzipChunkFetcher::processNextChunk`. Two new files:

- `parallel/block_map.rs` — port of `core/BlockMap.hpp`.
- `parallel/gzip_block_finder.rs` — port of `GzipBlockFinder.hpp`
  partitioning + `insert`/`get`/`finalize`.

Replace `chunk_fetcher.rs::drive`'s static partition table with a
`GzipBlockFinder` instance and an `insert(actual_end)` call after each
chunk resolves.

### Step 6. Port `BlockFetcher` cache + `Prefetcher` strategy (~600 LOC)

**Why**: C6. Replace the ad hoc `spec[]` + `mpsc::channel` ring with a
proper `Cache<size_t, Arc<ChunkData>>` + `Prefetcher<FetchingStrategy>` +
`ThreadPool`. Three new files in `parallel/`:

- `parallel/cache.rs` — LRU cache.
- `parallel/prefetcher.rs` — fetching strategy (sequential first).
- `parallel/block_fetcher.rs` — the dispatch class wiring cache, prefetcher,
  thread pool, and `decodeBlock`.

Replace `chunk_fetcher.rs::consumer_loop` with a port of
`GzipChunkFetcher::processNextChunk` that calls
`BlockFetcher::get(partitionOffset, blockIndex)`.

### Step 7. Replace `fast_marker_inflate.rs` with a port of `deflate::Block<>` (~1500 LOC, largest)

**Why**: D1 / C3, C4. Establishes the same shape rapidgzip has for its
non-ISAL bootstrap. The current `fast_marker_inflate` would be deleted
(or kept for the non-x86_64 path). Touches:

- New `parallel/deflate_block.rs` — port of `deflate::Block<>` with
  `m_window16` (2×32 KiB circular buffer of u16), `setInitialWindow`,
  `read`, `eob`/`eos`/`eof`, `readDynamicHuffmanCoding`, etc.
- Update `gzip_chunk.rs::finish_decode_chunk_with_inexact_offset` to
  drive a `deflate::Block` instance over multiple blocks rather than
  calling `decode_chunk_bootstrap`.

Pre-req: Step 1 (marker encoding).

### Step 8. Port `DecodedData::cleanUnmarkedData` and `getLastWindow` (~120 LOC)

**Why**: C5. Once Step 1 is done, port these directly from
`DecodedData.hpp:394–516` so `ChunkData` matches rapidgzip's
"during-finalize cleanup + window extraction" semantics. The
`getLastWindow({})` flow with `MapMarkers` exception is what powers
rapidgzip's "trailing 32 KiB has markers, retry" recovery — port it
to enable that recovery in gzippy too.

### Step 9. Port `appendSubchunksToIndexes` cascade and parallel marker post-processing (~250 LOC)

**Why**: C10, C20. After this, marker resolution can happen on multiple
chunks in parallel through the thread pool, and per-subchunk indexes
are populated. Touches `parallel/block_fetcher.rs` (`waitForReplacedMarkers`,
`queuePrefetchedChunkPostProcessing`, `queueChunkForPostProcessing`)
and the new `BlockMap` / `WindowMap` integration.

### Step 10. Port `gzip::readHeader`/`readFooter`, file-type detection (~200 LOC)

**Why**: C14. Move gzip header / trailer parsing into a `gzip/` module
that mirrors `vendor/rapidgzip/.../gzip/gzip.hpp`. Replace
`single_member.rs::skip_gzip_header` and the inline trailer parse.
Add `parallel/format.rs` mirroring `gzip/format.hpp`.

### Step 11. Port `IsalInflateWrapper` footer reading + multi-stream + header re-reading (~150 LOC)

**Why**: C15. Extends `inflate_wrapper.rs::IsalInflateWrapper` with
`m_need_to_read_header`, `read_header`, `read_footer`, and
`END_OF_STREAM` / `END_OF_STREAM_HEADER` stop emission. Folds together
with Step 2's `Footer` type.

### Step 12. Port `BlockFetcher::Statistics` and `GzipChunkFetcher::Statistics` (~200 LOC)

**Why**: C16. Replace `parallel/trace.rs` (or keep it alongside) with
aggregated counters that mirror rapidgzip's `--verbose` output. Exposes
cache hit rate, prefetch efficiency, false-positive count, decode
duration per stage, pool efficiency.

### Step 13. Port `CompressedVector` and window compression (~250 LOC)

**Why**: C12. Add per-window compression with `CompressionType::ZLIB`
default for highly-compressible chunks (per rapidgzip's heuristic at
`ChunkData.hpp:200–207`). Touches `parallel/window_map.rs` (windows
become `Arc<CompressedVector>` instead of `Arc<[u8;32768]>`),
`apply_window.rs` (decompress on lookup), and a new
`parallel/compressed_vector.rs`.

### Step 14. Port `IndexFileFormat` (~400 LOC, optional)

**Why**: C9. Once Steps 5 (BlockMap) and 11 (windows) are done, this is
mostly serialization. Provides a future seekable-index export feature.

### Step 15. Port `ChunkData::split` and `setEncodedOffset` (~200 LOC)

**Why**: C19, B2. After Steps 3, 4, 5 land, replace
`append_block_boundary`'s "always emit subchunk" with rapidgzip's
boundary-list + post-decode `split` flow. This restores the merge-small-trailing-subchunk
behavior (B2).

### Step 16. Port `HuffmanCodingDoubleLiteralCached` for the non-ISA-L path (~350 LOC, optional)

**Why**: C1. Restores the multi-symbol decode capability when
`isal-compression` feature is off (arm64 builds). Currently
`fast_marker_inflate` uses single-symbol `ConsumeFirstTable` only.

---

## G. Cross-cutting notes

- gzippy's parallel SM path **never** participates in multi-stream gzip
  decoding. Multi-member input is routed elsewhere
  (`decompress::decompress_gzip_libdeflate` → multi-member parallel path)
  before reaching `decompress_parallel`. Steps 2, 10, 11 above would
  unify these paths.

- gzippy treats CRC32 / ISIZE mismatch at the trailer as terminal
  corruption, never falling back to libdeflate
  (`single_member.rs:18–21` doc comment). Rapidgzip throws and the
  outer driver decides.

- Naming: rapidgzip uses `encodedOffsetInBits` / `decodedOffsetInBytes`;
  gzippy uses `encoded_offset_bits` / `decoded_offset`. Pure style.

- Bit-position arithmetic: both sides compute "absolute bit position"
  the same way (`input_len * 8 - avail_in * 8 - read_in_length`), but
  rapidgzip routes this through `BitReader::tell()` while gzippy
  computes it manually inside `IsalInflateWrapper::tell_compressed` and
  inside `finish_decode_chunk_with_inexact_offset`. No semantic
  difference.

- Test policy: every Rust file in `parallel/` has unit tests, and the
  whole pipeline is covered by `src/tests/routing.rs` (the
  "deletion-trap killer test"). Rapidgzip's tests live separately
  under `vendor/rapidgzip/tests/`; we do not run them.
