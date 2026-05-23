# Rapidgzip → gzippy parallel single-member: unported primitives

`src/decompress/parallel/` is a structural port of rapidgzip's
`ParallelGzipReader → GzipChunkFetcher → BlockFetcher → GzipChunk →
IsalInflateWrapper` chain. The destination is rapidgzip's parallel
architecture with no external compiled deflate: pure Rust for both the
orchestration (rapidgzip's C++ templates) and the inner inflate
(patched ISA-L). This document lists what still has to land — the
unported rapidgzip primitives (§§1–4) and the pure-Rust replacement
for ISA-L (§5). Each named item is a literal vendor class, header, or
patch. The unit is "done" when the validation gate at the bottom passes
on `neurotic` (16 physical x86_64).

## What is already ported (do not touch)

| Vendor | gzippy |
|---|---|
| `core/ThreadPool.hpp:33-248` | `thread_pool.rs` |
| `core/AffinityHelpers.hpp:11,76-101` | `thread_pool.rs::pinning_for_capacity`, `with_pinning_for_capacity`; wired at `chunk_fetcher.rs:324` |
| `core/StreamedResults.hpp:27-158` | `streamed_results.rs` |
| `core/BlockFinder.hpp:35-218` | `raw_block_finder.rs::RawBlockFinderCoordinator` |
| `core/BlockFetcher.hpp:38-687` | `block_fetcher.rs`; `process_ready_prefetches` at `:696` (vendor 427/463) |
| `core/Prefetcher.hpp:60-225` | `prefetcher.rs:41-211` (`FetchNextFixed`, `FetchNextAdaptive`) |
| `core/Prefetcher.hpp:234-336` | `prefetcher.rs:232-343` (`FetchMultiStream` — **not yet wired**, see §4) |
| `core/FasterVector.hpp:73-113` (`RpmallocAllocator`), `:124` (`FasterVector` alias) | `rpmalloc_alloc.rs` (`unsafe impl allocator_api2::alloc::Allocator for RpmallocAlloc` on `rpmalloc-sys`); `chunk_data.rs:24` `ChunkData::data` / `data_with_markers` typed `allocator_api2::vec::Vec<T, RpmallocAlloc>`; `Cargo.toml` features `arena-allocator = ["dep:allocator-api2", "dep:rpmalloc-sys"]`, transitively enabled by `isal-compression` |
| `core/BlockMap.hpp` | `block_map.rs` |
| `core/Cache.hpp` | `cache.rs` |
| `core/MarkerReplacement.hpp` | `replace_markers.rs:146` (unaligned AVX) |
| `core/common.hpp:498-512 interleave` | `prefetcher.rs:215-227` |
| `rapidgzip/WindowMap.hpp` | `window_map.rs` |
| `rapidgzip/GzipBlockFinder.hpp:34-307` (no seekable branches) | `gzip_block_finder.rs` |
| `rapidgzip/blockfinder/DynamicHuffman.hpp:39-225` + `precodecheck/CountAllocatedLeaves.hpp` | `block_finder.rs` (stored-block from `blockfinder/Uncompressed.hpp` folded in at `:668 find_blocks`) |
| `rapidgzip/chunkdecoding/GzipChunk.hpp:468-654` (handoff at vendor 520-525) | `gzip_chunk.rs:288 decode_chunk_marker_bootstrap_then_isal` |
| `rapidgzip/GzipChunkFetcher.hpp:554-583 queueChunkForPostProcessing` + `applyWindow` | `chunk_fetcher.rs:1069 submit_post_process_to_pool` (callsite `:932`); marker resolution in `apply_window.rs` + `replace_markers.rs` |
| `rapidgzip/gzip/isal.hpp:26-212` | `inflate_wrapper.rs`: FFI shim over patched `mxmlnkn/isa-l`. **Interim production path; removed at §5 Tier 3.** |
| `rapidgzip/gzip/deflate.hpp:175` (`HuffmanCodingISAL`) + `:38-39` (`HuffmanCodingDistanceISAL`) | `isal_huffman.rs`. **Interim production path on the post-bootstrap leg; removed at §5 Tier 3.** |
| `rapidgzip/huffman/HuffmanCodingBase.hpp`, `HuffmanCodingSymbolsPerLength.hpp:30-142` | `huffman_base.rs`, `huffman_symbols_per_length.rs` |

## Unported primitives

Listed in dependency order.

### 1. `HuffmanCodingReversedBitsCached`

**Vendor:** `huffman/HuffmanCodingReversedBitsCached.hpp:32-136`.
Generic LUT-cached canonical decoder, `1 << MAX_CODE_LENGTH` entries of
`(length, symbol)`. `gzip/deflate.hpp:196` aliases it as
`FixedHuffmanCoding` — every BTYPE=01 block. Also the distance HC for
the deflate-specific decoder in §2.

**gzippy now.** No port.
`deflate_block.rs:1095 read_internal_compressed_canonical_specialized`
builds tables via `HuffmanCodingSymbolsPerLength` (one-bit-per-symbol
tree walk) for both DYNAMIC and FIXED, with the litlen call site at
`:1126` and the distance call site at `:1132`. RFC 1951 length arrays
for FIXED come from `fixed_huffman_code_lengths()` at `:1315`. x86_64
production routes through ISA-L (`inflate_wrapper.rs`) for the
post-bootstrap leg and bypasses this; the marker-bootstrap leg
(`gzip_chunk.rs:411 bootstrap_with_deflate_block`) hits this slow path
on every arch.

**Delta.** Add `huffman_reversed_bits_cached.rs`. Wraps
`HuffmanCodingSymbolsPerLength` plus
`code_cache: [(u8 /*length*/, u16 /*symbol*/); 1 << MAX_LEN]`.
`initialize_from_lengths` fills the cache by reversing each canonical
code via `bit_manipulation::reverse_bits`. `decode(bit_reader)` peeks
`MAX_LEN` and returns the cache entry directly.

**Wiring.** `deflate_block.rs:1126` litlen HC: for
`CompressionType::FixedHuffman` the litlen HC swaps to
`HuffmanCodingReversedBitsCached`; the `:1132` distance HC stays —
FIXED blocks have no separate distance table (RFC 1951 3.2.6).

**Dependency.** None — base `HuffmanCodingSymbolsPerLength`
(`huffman_symbols_per_length.rs:228`) is already ported.

### 2. `HuffmanCodingShortBitsCachedDeflate<11>`

**Vendor:** `huffman/HuffmanCodingShortBitsCachedDeflate.hpp:22-280`.
Deflate-specialized 11-bit LUT for DYNAMIC blocks. Cache entry:

```cpp
struct CacheEntry { uint8_t bitsToSkip; uint8_t symbolOrLength; uint16_t distance; };
```

The distance code is fused into the literal-LUT when both fit one
peek. Selected by `gzip/deflate.hpp:177 LiteralOrLengthHuffmanCoding`
when `WITH_DEFLATE_SPECIFIC_HUFFMAN_DECODER` is defined. The LUT fill
at vendor `:84-109` reads `distanceHC.codeCache()` directly.

**gzippy now.** No port. Same call site as §1 uses
`HuffmanCodingSymbolsPerLength` for the
`CompressionType::DynamicHuffman` arm.

**Delta.** Add `huffman_short_bits_cached_deflate.rs`. Const-generic
`LUT_BITS`. `code_cache: [CacheEntry; 1 << LUT_BITS]` with the layout
above. `decode` peeks `LUT_BITS` and returns the cache entry; falls
back to base canonical decode for codes longer than `LUT_BITS`.

**Wiring.** `deflate_block.rs:1126` litlen HC for the
`CompressionType::DynamicHuffman` arm (non-ISA-L fallback) and the
marker-bootstrap path at `gzip_chunk.rs:411`.

**Dependency.** Requires §1 — the LUT fill reads the distance HC's
`code_cache`.

### 3. `HuffmanCodingShortBitsMultiCached<11>`

**Vendor:** `huffman/HuffmanCodingShortBitsMultiCached.hpp:24-269`.
Two-symbols-per-lookup variant of §2. Cache entry packs
`needToReadDistanceBits:1, bitsToSkip:6, symbolCount:2, symbols:18`
into one `u32`. `decode` returns `(packed_symbols: u32, count: u32)`.
Selected by `gzip/deflate.hpp:179` when
`WITH_MULTI_CACHED_HUFFMAN_DECODER`.

**gzippy now.** No port.

**Delta.** Add `huffman_short_bits_multi_cached.rs`. Packed-u32 cache
entry; `decode` returns the two-symbol pair.

**Wiring.** Same site as §2. The appender loop in
`gzip_chunk.rs:411 bootstrap_with_deflate_block` unpacks the
two-symbol return.

**Dependency.** Requires §1 for the distance HC at fill time.

### 4. `FetchMultiStream` production wiring

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

### 5. Pure-Rust DEFLATE inflate with stopping points

The only C dependency on the parallel SM hot path is patched ISA-L
(rapidgzip itself relies on it). gzippy's destination is no compiled-C
deflate anywhere in production. Two-phase scope:

- **§5 itself** replaces the parallel SM post-bootstrap inflate
  (`IsalInflateWrapper`'s `read_stream`).
- **§5 Tier 3** (submodule deletion) additionally requires pure-Rust
  replacements for the non-SM ISA-L users — see Tier 3 below.

The marker-bootstrap leg (`gzip_chunk.rs:411 bootstrap_with_deflate_block`)
already uses pure-Rust `deflate_block::Block` and is **not** what §5
swaps; the cached-Huffman items §§1–3 are what speed up that leg.

**Vendor + patches.** Rapidgzip's inner inflate is
`vendor/isa-l/igzip/igzip_inflate.c`. The stopping-point capability is
gzippy's own patch on top of intel/isa-l commit `496255c`, maintained
at `packaging/isal-patches/`:

- `igzip_lib.h-stopping-points.patch` (41 lines, ~22 additions) — adds
  the `isal_stopping_point` bitfield enum
  (`END_OF_STREAM_HEADER`, `END_OF_STREAM`, `END_OF_BLOCK_HEADER`,
  `END_OF_BLOCK`) and the `points_to_stop_at`, `stopped_at`,
  `tmp_out_stopped_at` fields on `isal_inflate_state`.
- `igzip_inflate.c-stopping-points.patch` (395 lines) — state-machine
  checks for those four points; early-return from `isal_inflate()` with
  state preserved so the caller resumes from saved state.

**gzippy now.** Two inflates live side by side:

- **C path on the parallel SM post-bootstrap leg.**
  `inflate_wrapper.rs` = `IsalInflateWrapper` FFI shim. Owns stopping
  points today.
- **Pure-Rust paths already in production (non-SM).** ~14.8k lines
  across `src/decompress/inflate/` plus standalone modules; production
  callers today:
  - `bgzf.rs:71,1149,1170,1422` — `TwoLevelTable` / `CombinedLUT`
    inflate path (BGZF parallel decompress).
  - `backends/isal_decompress.rs:1054` — sequential x86 decompress
    falls into the `_pub` family at the slow-path boundary.
  - `src/decompress/index.rs:87,417,424` — gzip-index building uses
    `scan_inflate::scan_deflate_fast` → `consume_first_decode::{
    decode_stored_pub, decode_fixed_pub, decode_dynamic_pub}`.
  - `parallel/gzip_chunk.rs:411 bootstrap_with_deflate_block` — uses
    only the `Bits` reader from `consume_first_decode`; the actual
    block decoder is `deflate_block::Block` (a separate engine).
    Not a `consume_first_decode` inflate-engine caller.

  | Module | Lines | Role |
  |---|---|---|
  | `inflate/consume_first_decode.rs` | 3562 | DEFLATE inflate engine + `Bits` reader |
  | `inflate/consume_first_table.rs` | 898 | Packed Huffman LUT layout |
  | `inflate/libdeflate_decode.rs` | 1226 | libdeflate-compatible table format |
  | `inflate/libdeflate_entry.rs` | 977 | Fixed-table builders |
  | `inflate/vector_huffman.rs` | 966 | SIMD multi-symbol decode (AVX2 / AVX-512) |
  | `inflate/specialized_decode.rs` | 602 | Per-block code-length-specialized decoders |
  | `inflate/jit_decode.rs` | 527 | Table-fingerprint cache key |
  | `inflate/double_literal.rs` | 289 | Two-symbol literal cache |
  | `inflate/bmi2.rs` | 151 | BMI2 bit-extraction |
  | `simd_huffman.rs` | 772 | Separate SIMD Huffman path |
  | `two_level_table.rs` | 805 | Two-level Huffman table (bgzf + simd_huffman) |
  | `inflate_tables.rs` | 670 | Shared LUT primitives |
  | `combined_lut.rs` | 292 | Combined literal/length LUT |
  | `packed_lut.rs` | 340 | Packed LUT entry layout |

  **None expose stopping-point semantics.** That's the missing piece.

**Delta.**

(a) **Add stopping points to `consume_first_decode`.** Mirror the C
patch field-for-field. `StoppingPoints` already exists at
`inflate_wrapper.rs:50` — reuse it. Add resumable state on the
inflate driver, shaped like vendor's pattern of storing the stop result
on the state struct:

```rust
// in src/decompress/inflate/consume_first_decode.rs

pub struct InflateState {            // resumable; new
    points_to_stop_at:    u32,        // mirror of patched isal_inflate_state
    stopped_at:           u32,        // ditto
    tmp_out_stopped_at:   u32,        // ditto
    // … existing fields …
}

impl InflateState {
    pub fn set_points_to_stop_at(&mut self, points: StoppingPoints);
    pub fn stopped_at(&self) -> StoppingPoints;
    pub fn clear_stop(&mut self);
}

pub fn inflate_resumable(
    state:  &mut InflateState,
    input:  &[u8],
    output: &mut [u8],
) -> Result<usize /*bytes_written*/, Error>;
```

State-machine checks at the four patched sites: after BFINAL/BTYPE
decode → `END_OF_STREAM_HEADER`/`END_OF_BLOCK_HEADER`; after BFINAL=1's
last symbol → `END_OF_STREAM`; after any block's last symbol →
`END_OF_BLOCK`. On a fired stop, return early with `stopped_at` set.
Output is raw bytes (matching `IsalInflateWrapper::read_stream`); the
16-bit `MARKER_BASE` form is `deflate_block::Block`'s concern and is
not part of this delta.

(b) **Pure-Rust `inflate_wrapper.rs` body, same module path.**
Keep the file name `inflate_wrapper.rs`. Add a body alternative behind
`#[cfg(feature = "pure-rust-inflate")]` that hosts the existing public
surface — every method on `IsalInflateWrapper`
(`inflate_wrapper.rs:60-540`):

```
new, with_until_bits, set_window, set_stopping_points,
stopped_at, clear_stop, is_final_block, btype, tell_compressed,
read_stream, read_footer_at_current, reset_for_next_stream,
remaining_input, advance_input, at_end_of_stream, encoded_until_bits,
debug_points_to_stop_at, debug_stopped_at_raw,
debug_tmp_out_stopped_at, debug_block_state
```

Return types (`ReadStreamResult` at `inflate_wrapper.rs:79`,
`StoppingPoints` at `:50`) stay. One-for-one swap target.

(c) **Bench, tiered.** Add `benches/inflate_isal_vs_pure_rust.rs`
calling the wrapper API (NOT raw inflate against a flat buffer — the
production hot path runs through `IsalInflateWrapper::read_stream` from
within the marker-bootstrap + chunk-decode loop). Silesia + the 24 MiB
low-entropy fixture, AVX2 + BMI2 enabled, three trials each.
`ChunkData::data` is now on `RpmallocAlloc` (already ported), so the
bench compares like-for-like allocator behaviour.

- **Tier 1 — opt-in:** pure-Rust wall-time within **1.5×** of ISA-L
  on silesia. Build with `--features pure-rust-inflate`.
- **Tier 2 — default behind feature:** within **1.2×** on silesia
  AND no regression on the 24 MiB fixture vs Tier-1 numbers. Flip the
  default; ISA-L still selectable.
- **Tier 3 — submodule deletion:** within **1.05×** on silesia, AND
  pure-Rust replacements landed for the non-SM ISA-L users:

  - `backends/isal_decompress.rs` (T1 x86 sequential decompress) →
    swap to the pure-Rust inflate already used at
    `:1054`.
  - `backends/isal_compress.rs` (L0–L3 x86 compress) → pure-Rust
    replacement is out of scope of the rapidgzip port; either land a
    separate pure-Rust fast-compression module or accept that Tier 3
    removes the L0–L3 fast path until a Rust replacement ships.
  - `Cargo.toml`: detach `arena-allocator` from
    `isal-compression` so the arena survives the feature drop.

  When the above land: drop the `isal-compression` feature, remove
  `vendor/isa-l`, `vendor/isal-rs`, `packaging/isal-patches/`.

(d) **Production swap (Tier 2 onward).** Replace
`IsalInflateWrapper::*` call sites on the parallel SM path:
`gzip_chunk.rs`, `chunk_fetcher.rs`, `single_member.rs`,
`block_fetcher.rs`.

**Wiring.** Above.

**Dependency.**

- §§1–3 (cached Huffman) speed up the **bootstrap** leg; §5 swaps the
  **post-bootstrap** leg. Independent in code terms — neither blocks
  the other on swap-correctness. Bench results (c) are easier to
  attribute once §§1–3 land because bootstrap stops dominating the
  hot-path profile.
- Tier 3 requires the non-SM ISA-L migrations listed above.

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
  `chunk_fetcher.rs:122` (the type alias) are plumbed but the
  consumer-side seekable reader is unbuilt. Today the supporting code
  is gated on `#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]`
  (same as the rest of the parallel SM path); a dedicated
  `seekable-index` feature flag should land when the reader does.
- BZIP2, ZLIB-format, BGZF parallel encoders, multi-member parallel
  encoders, Python bindings, `GzipReader.hpp`, `GzipAnalyzer.hpp`.
- **`AlignedAllocator.hpp`** — `replace_markers.rs:146` uses unaligned
  AVX (`_mm256_loadu_si256`); re-evaluate only if `dTLB-load-misses`
  stays elevated.
- **`src/decompress/deflate64.rs`** — gzippy-original Deflate64; no
  rapidgzip counterpart; no consumer in the SM path. Published API at
  `src/lib.rs:267,277` (`decompress_deflate64`,
  `decompress_deflate64_to_writer`); keep until that API is removed in
  a separate breaking-change cycle.
- **`src/decompress/ultra_fast_inflate.rs`** — gzippy-original;
  consumed by `bgzf.rs`, not the SM path. Keep for BGZF.

## Validation gate

On `neurotic` (16 physical x86_64, ISA-L), via `make test-x86_64`.
The unit is the gate.

1. **Synthetic perf.**
   `cargo test --release -- test_single_member_parallel_not_slower_than_sequential`
   reports `ratio < 0.5` at `T = min(16, num_cpus::get_physical())` on
   the 24 MiB low-entropy fixture in `routing.rs`.
2. **Silesia perf (real corpus, not yet implemented).** Add
   `test_single_member_parallel_silesia` against real `silesia.tar.gz`
   (`https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia`); report
   `ratio < 0.5` at the same T. The existing
   `test_single_member_parallel_silesia_class_not_slower_than_sequential`
   in `routing.rs:521` is a synthetic-high-entropy proxy and does not
   discharge this gate.
3. **Routing / FNAME / coordinator traps.**
   `test_single_member_routing_multithread`,
   `test_coordinator_boundary_search_runs_on_x86_64_isal`, and
   `test_parallel_sm_handles_fname_header` (nested under
   `#[cfg(all(target_arch = "x86_64", feature = "isal-compression"))] mod fname_header_parallel_sm`)
   pass byte-perfect. `MARKER_PIPELINE_RUNS` (`single_member.rs:95`)
   and `COORDINATOR_BOUNDARY_SEARCH_RUNS` (`chunk_fetcher.rs:1282`)
   both increment.
4. **Dead-code allows gone.** Every
   `#![allow(dead_code)] // vendor-faithful rapidgzip port` removed
   from production SM hot-path modules.
   `cargo clippy --release --all-targets --features isal-compression -- -D warnings`
   succeeds. Pre-seekable-reader scaffolding stays under the existing
   `cfg(all(feature = "isal-compression", target_arch = "x86_64"))`
   gating until the seekable reader lands.
5. **`perf stat` vs vendor.**

   ```
   perf stat -e dTLB-load-misses,major-faults,minor-faults,cycles \
       ./target/release/gzippy -d -c silesia.tar.gz > /dev/null
   ```

   compared against `rapidgzip -P 16 -c silesia.tar.gz > /dev/null` on
   the same host (`pip install rapidgzip`). Required: gzippy
   `minor-faults` within 1.5× of rapidgzip's; gzippy wall-time within
   1.2×. (The `RpmallocAllocator` port already landed; this gate
   measures whether it closes the page-fault gap as expected.)
6. **No external compiled deflate.** `cargo tree --release` shows no
   `isal-sys` / `isal-rs` in the production dependency graph; the
   `vendor/isa-l` and `vendor/isal-rs` submodules and
   `packaging/isal-patches/` are removed. §§1–4 mark parity with
   rapidgzip's *architecture*; this gate marks parity with no C
   dependency anywhere in production (parallel SM, sequential
   decompress, fast-tier compress).

If any of (1)–(6) miss, the unit hasn't landed.

## Reading order

- `src/decompress/parallel/single_member.rs` — entry,
  `MARKER_PIPELINE_RUNS` deletion trap.
- `chunk_fetcher.rs:257 drive` → `consumer_loop` →
  `submit_decode_to_pool` → `submit_post_process_to_pool`.
- Vendor `rapidgzip/ParallelGzipReader.hpp:495 read` and
  `rapidgzip/GzipChunkFetcher.hpp:312 processNextChunk`, side-by-side
  with the Rust.
