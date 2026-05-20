# Rapidgzip-in-Rust Port: Full Design

> **Status (May 2026):** The unwired rapidgzip port files (“museum”) were deleted.
> Production SM decode is `sm_driver::read_parallel_sm` → `chunk_fetcher::drive` →
> `gzip_chunk::decode_chunk_isal_inexact` (ISA-L via `inflate_wrapper`). Modules
> such as `fast_marker_inflate`, `deflate_block`, and `parallel_gzip_reader` no
> longer exist — see git history. The spec below is kept for type/layout reference;
> the **Files** tables are outdated.

This is the complete design of the rewrite. Every type, every function
signature, every piece of data flow, and every concurrency interaction is
specified before any of it gets implemented. Per the user directive: hold
the whole in mind, increment toward the whole at all times, do not write
any of it until the entire thing is reasoned about.

## Goal

Replace gzippy's v0.6 marker-pipeline + reconcile + splitter design with
a port of rapidgzip's parallel-decode architecture. End state: gzippy's
parallel single-member path is structurally rapidgzip with the
ISA-L-stopping-point patch we already vendored at `crates/isal-sys-patched/`.

## Non-goals

- Replacing the v0.6 multi-member or BGZF paths. These already work; out
  of scope.
- Matching rapidgzip's compress side. Only decompress.
- Matching their CLI flags, indexing format, or random-access API.

## Reference

All citations of the form `[path:lines]` are to the vendored rapidgzip
source at `vendor/rapidgzip/librapidarchive/src/rapidgzip/`.

## Files (as of May 2026 — production tree)

### Kept (hot path)

| Path | Role |
|---|---|
| `sm_driver.rs` | Gzip envelope + CRC/ISIZE verify; calls `chunk_fetcher::drive` |
| `chunk_fetcher.rs` | Consumer/worker orchestration, prefetch, window propagation |
| `gzip_chunk.rs` | `decode_chunk_isal_inexact` — ISA-L chunk decode (speculative + on-demand) |
| `inflate_wrapper.rs` | `IsalInflateWrapper` — patched ISA-L stopping points |
| `chunk_data.rs`, `window_map.rs`, `apply_window.rs`, `replace_markers.rs` | Chunk model + marker resolution |
| `block_finder.rs`, `block_fetcher.rs`, `block_map.rs` | Boundary candidates + block map |
| `single_member.rs` | Router entry `decompress_parallel` |

### Removed (museum — recover via git)

Unwired rapidgzip ports deleted to reduce confusion: `deflate_block`, `parallel_gzip_reader`,
`fast_marker_inflate`, Huffman table variants, split `blockfinder_*`, `gzip_reader`,
`gzip_analyzer`, `index_file_format`, and ~20 similar files.

## Type system (complete)

### `chunk_data.rs`

```rust
/// One deflate-block-aligned slice of a chunk's decoded output. Ports
/// `rapidgzip::ChunkData::Subchunk` (ChunkData.hpp:138-145).
#[derive(Debug, Clone)]
pub struct Subchunk {
    pub encoded_offset_bits: usize,
    pub encoded_size_bits: usize,
    pub decoded_offset: usize,
    pub decoded_size: usize,
    /// Last 32 KiB output BEFORE this subchunk started decoding. Set
    /// after sequential window propagation completes; consumed by
    /// `apply_window` to resolve any markers in this subchunk's range.
    pub window: Option<std::sync::Arc<[u8; 32768]>>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ChunkStatistics {
    pub decode_duration_ns: u64,
    pub apply_window_duration_ns: u64,
    pub marker_count: u64,
    pub non_marker_count: u64,
    pub preemptive_stop_count: u64,
}

#[derive(Debug, Clone)]
pub struct ChunkConfiguration {
    pub split_chunk_size: usize,         // ~512 KiB; subchunk emission threshold
    pub max_decoded_chunk_size: usize,   // ~4 MiB; preemptive stop threshold
    pub crc32_enabled: bool,
}

/// Ported from `rapidgzip::ChunkData` (ChunkData.hpp ~lines 80-400).
#[derive(Debug)]
pub struct ChunkData {
    pub encoded_offset_bits: usize,
    pub encoded_size_bits: usize,
    pub data_with_markers: Vec<u16>,   // prefix; values >= MARKER_BASE need apply_window
    pub data: Vec<u8>,                  // suffix; clean bytes
    pub subchunks: Vec<Subchunk>,
    pub crc: crc32fast::Hasher,
    pub stopped_preemptively: bool,
    pub statistics: ChunkStatistics,
    pub configuration: ChunkConfiguration,
}

impl ChunkData {
    pub fn new(encoded_offset_bits: usize, configuration: ChunkConfiguration) -> Self;
    pub fn decoded_size(&self) -> usize;
    pub fn is_empty(&self) -> bool;

    /// Ports `ChunkData::append(DecodedVector&&)` (ChunkData.hpp:209-).
    /// Appends to either data_with_markers (if any u16 has the marker bit)
    /// or data (otherwise). Updates crc.
    pub fn append_markered(&mut self, values: &[u16]);
    pub fn append_clean(&mut self, bytes: &[u8]);

    /// Ports `ChunkData::appendDeflateBlockBoundary` (ChunkData.hpp:159-180).
    /// Called by the decoder when it crosses a real deflate block boundary
    /// AND the accumulated decoded size since the last subchunk meets the
    /// `split_chunk_size` threshold. Starts a new subchunk at the current
    /// (encoded_offset_bits, decoded_size).
    pub fn append_block_boundary(&mut self, encoded_offset_bits: usize);

    /// Ports `ChunkData::finalizeChunk` (ChunkData.hpp:136-159).
    /// Called at end of decode. Merges undersized trailing subchunks back
    /// into their predecessor, sets the final encoded_size_bits.
    pub fn finalize(&mut self, end_encoded_offset_bits: usize);
}

pub use crate::decompress::parallel::replace_markers::MARKER_BASE;
```

### `inflate_wrapper.rs`

```rust
/// Bitfield matching `ISAL_STOPPING_POINT_*` constants in the patched
/// ISA-L. Port of `enum StoppingPoint` in `gzip/isal.hpp`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoppingPoint {
    None                  = 0,
    EndOfStreamHeader     = 1,
    EndOfStream           = 2,
    EndOfBlockHeader      = 4,
    EndOfBlock            = 8,
}

bitflags! {
    pub struct StoppingPoints: u32 {
        const NONE                  = 0;
        const END_OF_STREAM_HEADER  = 1;
        const END_OF_STREAM         = 2;
        const END_OF_BLOCK_HEADER   = 4;
        const END_OF_BLOCK          = 8;
    }
}

/// Port of `rapidgzip::IsalInflateWrapper` (gzip/isal.hpp). Single
/// stateful inflate. Holds the patched `inflate_state` and a reference
/// to the compressed input slice. Caller sets stopping points + a window,
/// then calls `read_stream` to pump output until a stop fires.
pub struct IsalInflateWrapper<'a> {
    state: isal_sys::igzip_lib::inflate_state,
    input: &'a [u8],
    encoded_start_offset_bits: usize,
    needs_to_read_header: bool,
}

#[derive(Debug)]
pub struct ReadStreamResult {
    pub bytes_written: usize,
    pub footer: Option<GzipFooter>,
}

#[derive(Debug, Clone, Copy)]
pub struct GzipFooter {
    pub crc32: u32,
    pub uncompressed_size: u32,
    pub block_boundary_decoded_offset: usize,
}

impl<'a> IsalInflateWrapper<'a> {
    pub fn new(input: &'a [u8], encoded_offset_bits: usize) -> Self;
    pub fn set_window(&mut self, window: &[u8]);
    pub fn set_stopping_points(&mut self, points: StoppingPoints);
    pub fn stopped_at(&self) -> StoppingPoints;
    pub fn is_final_block(&self) -> bool;
    pub fn btype(&self) -> Option<DeflateCompressionType>; // Stored/Fixed/Dynamic
    pub fn tell_compressed(&self) -> usize;
    pub fn read_stream(&mut self, output: &mut [u8]) -> Result<ReadStreamResult, InflateError>;
}
```

### `apply_window.rs`

```rust
/// Resolve markers in `chunk.data_with_markers` against the given
/// window. Mirrors `ChunkData::applyWindow` (ChunkData.hpp:302).
/// Uses the existing SIMD `replace_markers` kernel.
///
/// After this call, `chunk.data_with_markers` values are all < 256
/// (literal bytes). The caller is expected to narrow it to u8 and
/// concatenate with `chunk.data` to produce the final output.
pub fn apply_window(chunk: &mut ChunkData, window: &[u8; 32768]);
```

### `gzip_chunk.rs`

```rust
/// Ports `GzipChunk::decodeChunkWithInflateWrapper` (GzipChunk.hpp:190-268).
/// Used when the caller has a verified initial window AND a verified
/// exact stopping offset. Errors if `tell_compressed() != exact_until_bits`
/// at the end. Caller dispatches this when window propagation has
/// established the real window for this chunk's start.
pub fn decode_chunk_with_inflate_wrapper(
    input: &[u8],
    encoded_offset_bits: usize,
    exact_until_bits: usize,
    initial_window: &[u8],
    decoded_size_hint: Option<usize>,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError>;

/// Ports `GzipChunk::finishDecodeChunkWithInexactOffset` (GzipChunk.hpp:280-410).
/// Inexact-stop discovery path. Caller passes the wrapper at any state
/// (typically just after seeking to a partition_offset) plus the
/// initial window (may be empty for speculative decode). Stops at
/// END_OF_BLOCK / END_OF_BLOCK_HEADER / END_OF_STREAM events. Emits
/// subchunks at boundaries when `decoded_size >= split_chunk_size`.
/// Sets `stopped_preemptively` if `max_decoded_chunk_size` was hit
/// before reaching `until_bits`.
pub fn decode_chunk_isal_inexact(
    wrapper: &mut IsalInflateWrapper,
    until_bits: usize,
    initial_window: &[u8],
    max_decoded_chunk_size: usize,
    partial: ChunkData,
) -> Result<ChunkData, ChunkDecodeError>;

#[derive(Debug)]
pub enum ChunkDecodeError {
    InflateFailed(InflateError),
    ExactStopMissed { requested: usize, actual: usize },
    InvalidDeflateBlock,
}
```

### `window_map.rs`

```rust
/// Ports `rapidgzip::WindowMap` (WindowMap.hpp). Stores propagated
/// windows keyed by the compressed-bit-offset they're meant to seed.
/// Append-only, monotonically-increasing-key. Concurrent reader-friendly
/// (`Arc<[u8; 32768]>` values).
#[derive(Default)]
pub struct WindowMap {
    entries: std::collections::BTreeMap<usize, std::sync::Arc<[u8; 32768]>>,
}

impl WindowMap {
    pub fn new() -> Self;
    pub fn get(&self, encoded_offset_bits: usize) -> Option<std::sync::Arc<[u8; 32768]>>;
    pub fn insert(&mut self, encoded_offset_bits: usize, window: std::sync::Arc<[u8; 32768]>);
}
```

### `chunk_fetcher.rs`

```rust
/// Ports `rapidgzip::GzipChunkFetcher` and the parts of `BlockFetcher`
/// we need (just enough to drive sequential read with prefetching). The
/// in-process equivalent of rapidgzip's parallel-decompress loop.
///
/// Owned by the dispatcher (called from `single_member::decompress_parallel`).
/// Workers dispatched via `std::thread::scope`. Window propagation +
/// `apply_window` dispatch happens on the main thread between worker
/// completions.
pub struct GzipChunkFetcher<'a> {
    input: &'a [u8],
    chunk_size_bits: usize,           // partition spacing (e.g. 4 MiB * 8)
    parallelization: usize,           // num threads
    window_map: WindowMap,
    block_map: BlockMap,              // BTreeMap encoded_offset → (encoded_size, decoded_size)
    prefetch_cache: PrefetchCache,    // LRU keyed by partition_offset
    configuration: ChunkConfiguration,
}

impl<'a> GzipChunkFetcher<'a> {
    pub fn new(
        input: &'a [u8],
        parallelization: usize,
        configuration: ChunkConfiguration,
    ) -> Self;

    /// Returns the next chunk in order. Drives both prefetching and
    /// on-demand fetch. Ports `GzipChunkFetcher::get` (GzipChunkFetcher.hpp).
    pub fn get_next_chunk(&mut self) -> Result<std::sync::Arc<ChunkData>, ParallelError>;

    pub fn has_more(&self) -> bool;
}

struct BlockMap {
    /// Ordered (encoded_offset_bits → (encoded_size_bits, decoded_size_bytes))
    /// entries. Populated as chunks complete. The dispatcher uses this to
    /// know how many chunks are needed and to seed prefetching.
    entries: std::collections::BTreeMap<usize, (usize, usize)>,
}

struct PrefetchCache {
    /// Maps partition_offset → in-flight task or completed ChunkData.
    /// Bounded to `2 * parallelization` per rapidgzip's
    /// `BlockFetcher::m_prefetchCache` (BlockFetcher.hpp:182).
    inflight: std::collections::HashMap<usize, ChunkSlot>,
    capacity: usize,
}

enum ChunkSlot {
    InFlight,                            // a worker thread is decoding
    Ready(std::sync::Arc<ChunkData>),    // decoded; awaiting consumer pickup
}
```

### `single_member.rs` (rewritten)

```rust
pub fn decompress_parallel<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    // 1. Header validation + trailer extraction (unchanged from current)
    let deflate_data = ...;
    let (expected_crc, expected_size) = ...;

    // 2. Configure
    // Rapidgzip defaults (verified by grep against vendored source):
    //   ParallelGzipReader.hpp:280  chunkSizeInBytes = 4_Mi
    //   ParallelGzipReader.hpp:292  setMaxDecompressedChunkSize( 20U * chunkSizeInBytes )
    //   GzipChunkFetcher.hpp:706    chunkDataConfiguration.splitChunkSize = spacingInBits()/8U
    let chunk_size_bytes = 4 * 1024 * 1024;
    let configuration = ChunkConfiguration {
        split_chunk_size: chunk_size_bytes,
        max_decoded_chunk_size: 20 * chunk_size_bytes,
        crc32_enabled: true,
    };

    // 3. Create fetcher + thread pool
    let mut fetcher = GzipChunkFetcher::new(deflate_data, num_threads, configuration);

    // 4. Consume chunks in order; apply_window in parallel; write
    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size = 0;
    while fetcher.has_more() {
        let chunk = fetcher.get_next_chunk()?;
        // chunk is fully decoded AND already has apply_window done in parallel
        // (fetcher drives apply_window before returning a chunk).
        writer.write_all(&chunk.data_with_markers_as_u8())?;
        writer.write_all(&chunk.data)?;
        total_crc.combine(&chunk.crc);
        total_size += chunk.decoded_size();
    }

    // 5. Verify trailer
    if total_size != expected_size { return Err(SizeMismatch); }
    if total_crc.finalize() != expected_crc { return Err(CrcMismatch); }

    MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
    Ok(total_size as u64)
}
```

## Data flow (end-to-end)

```
decompress_parallel(deflate_data, writer, T=16)
  │
  ├─► GzipChunkFetcher::new(input=deflate_data, parallelization=16, ...)
  │     creates: window_map (with WindowMap[0] = empty), block_map, prefetch_cache(cap=32)
  │
  ├─► loop while fetcher.has_more():
  │     │
  │     ├─► fetcher.get_next_chunk():
  │     │     │
  │     │     ├─► current encoded_offset = (last completed end_bit) or 0
  │     │     │
  │     │     ├─► partition_offset = (current_offset / chunk_size_bits) * chunk_size_bits
  │     │     │
  │     │     ├─► IF prefetch_cache[partition_offset] = Ready(chunk):
  │     │     │     check chunk.matches(current_offset) — if not, discard
  │     │     │
  │     │     ├─► IF cache miss OR mismatch:
  │     │     │     dispatch worker for current_offset (on-demand)
  │     │     │     spawn workers for next prefetch slots up to cache capacity
  │     │     │     workers run decode_chunk_isal_inexact(...)
  │     │     │     each worker produces a ChunkData with subchunks
  │     │     │
  │     │     ├─► WAIT for current chunk's worker (with timeout-based prefetch poking)
  │     │     │
  │     │     ├─► Once chunk ready:
  │     │     │     window = window_map.get(current_offset) — must be Some()
  │     │     │     apply_window(chunk, &window) — runs on a worker thread
  │     │     │     extract chunk's last 32 KiB → window_map.insert(end_bit, ...)
  │     │     │     block_map.insert(current_offset, encoded_size, decoded_size)
  │     │     │     return Arc<ChunkData>
  │     │
  │     ├─► writer.write_all(chunk.data_with_markers_as_u8 + chunk.data)
  │     ├─► total_crc.combine(chunk.crc)
  │     ├─► total_size += chunk.decoded_size()
  │
  └─► verify trailer, return total_size
```

## Concurrency model

- **One scoped thread pool** of size T. Workers blocked on a work queue
  fed by the fetcher.
- **Worker tasks**: `decode_chunk_isal_inexact` (most
  common) or `decode_chunk_with_inflate_wrapper` (rare, on exact retry).
  Produce ChunkData. No shared mutable state.
- **Apply-window tasks**: dispatched after window_map.get() succeeds.
  Run on the same thread pool. Take `&mut ChunkData` (via `Arc<Mutex<>>`
  or by transferring ownership through a channel).
- **Main thread**: drives `get_next_chunk()`. Sequential. Holds the
  window_map and block_map (no contention).
- **Sequential dependency**: chunk N+1's window depends on chunk N's
  output. Therefore chunk N's apply_window must complete before chunk
  N+1's apply_window can dispatch. This is the bottleneck rapidgzip has
  too.

## Error handling

- `InflateError`: ISA-L returned non-zero ret or unexpected state.
  Surface up as `ParallelError::DecodeFailed`. Routing-layer fallback
  to single-thread libdeflate.
- `ChunkDecodeError::ExactStopMissed`: exact-stop decode landed at
  wrong offset. Should be unreachable; bug. Panic in debug,
  `DecodeFailed` in release.
- `ChunkDecodeError::InvalidDeflateBlock`: corrupt input. Same handling.
- `CrcMismatch` / `SizeMismatch`: after trailer verify. Same handling.

## Test plan

Each new module gets its own unit tests. Key end-to-end tests:

1. **`apply_window` correctness**: synthetic ChunkData with known
   markers + known window → expected byte output. Compares against
   per-byte expected values.

2. **`IsalInflateWrapper` oracle**: decode 5 random fixtures with
   stopping points enabled vs disabled. Byte-for-byte equality.
   Verifies the patched ISA-L is correct.

3. **`decode_chunk_with_inflate_wrapper` oracle**: decode a known
   chunk from a verified boundary. Resulting ChunkData's
   `data_with_markers` MUST be empty (because we had the real
   window); `data` MUST equal expected bytes; `subchunks` must
   match `record_block_starts` boundaries inside the range.

4. **`decode_chunk_isal_inexact` oracle**: decode
   from bit 0 with empty window. Resulting ChunkData's
   `data_with_markers` may be non-empty (cross-chunk back-refs are
   zeros from empty window). After `apply_window` with the right
   (empty) window, MUST byte-match the gzip oracle.

5. **`GzipChunkFetcher` integration**: drive a 24 MiB synthetic
   fixture through the fetcher. Byte-for-byte equality with oracle.

6. **`decompress_parallel` end-to-end**: replaces all existing
   `roundtrip_*` tests in `single_member.rs::tests`. Must produce
   identical output to before.

7. **Routing tests** (`tests/routing.rs`): unchanged. They test
   that the parallel path is taken; the path internals are
   different but the contract is the same (correct output + CRC).

## Migration order (execute in this order)

Each step is a separate commit. The codebase compiles and passes tests
at every step. No "wire it later" placeholders.

1. **Land design doc** (this file). [DOC ONLY]
2. **Complete `chunk_data.rs`** with full ChunkData / Subchunk /
   ChunkConfiguration + impl. Unit tests for `append_*` and
   `append_block_boundary`. [TYPES + TESTS]
3. **Add `apply_window.rs`** with the resolver + tests. [PURE FUNCTION]
4. **Add `inflate_wrapper.rs`** with IsalInflateWrapper + oracle test
   (test #2). [ISA-L PRIMITIVE]
5. **Add `gzip_chunk.rs`** with both decode functions + oracle tests
   (tests #3 and #4). [CHUNK DECODER]
6. **Add `window_map.rs`**. [STORAGE]
7. **Add `chunk_fetcher.rs`** with GzipChunkFetcher + integration
   test (test #5). [DISPATCHER]
8. **Rewrite `decompress_parallel`** to use GzipChunkFetcher. All
   existing roundtrip tests pass. [WIRING] — at this point the new
   architecture is live and the old code is still there but unused.
9. **Delete v0.6 scaffolding**. [DELETIONS]
10. **Re-bench**. Target ≥0.99× rapidgzip. [VALIDATION]

## Disproof prompts (for use after implementation)

- "Does `apply_window` produce byte-identical output to the legacy
  `replace_markers` + `narrow_and_append` path on every fixture?"
- "Does `GzipChunkFetcher::get_next_chunk` ever return chunks out of
  order? If so, the writer's bytes are corrupt."
- "Does `IsalInflateWrapper::tell_compressed` after a stopping-point
  pause match the bit position the patched ISA-L recorded? If not,
  the block_map entries point at wrong offsets."
- "Can two workers see the same `partition_offset` and produce two
  ChunkDatas? If yes, the prefetch_cache locking is wrong."
- "What happens if `chunk_size_bits > deflate_data.len() * 8`? Single
  chunk; the fetcher should handle this without crashing."

## Estimated effort

~2000 lines of new Rust (mostly mechanical port). ~3000 lines of
deletion. 6-10 commits. Several hours of focused work. The risk is
concentrated in `chunk_fetcher.rs` (the dispatcher state machine) and
`inflate_wrapper.rs` (correct use of the patched ISA-L state across
pause/resume cycles).
