# Rapidgzip → gzippy parallel single-member: unported primitives

`src/decompress/parallel/` is a structural port of rapidgzip's
`ParallelGzipReader → GzipChunkFetcher → BlockFetcher → GzipChunk →
IsalInflateWrapper` chain. The destination is rapidgzip's parallel
architecture with no external compiled deflate: pure Rust for both the
orchestration (rapidgzip's C++ templates) and the inner inflate
(patched ISA-L). This document lists what still has to land — the
unported rapidgzip primitives (§§1–4) and the pure-Rust replacement
for ISA-L (Track B). Each named item is a literal vendor class, header, or
patch. The unit is "done" when the **performance validation gate** (Track A)
passes on `neurotic` (16 physical x86_64). Track B (C-free production) is
a separate stretch goal.

## What is already ported (do not touch)

| Vendor | gzippy |
|---|---|
| `core/ThreadPool.hpp:33-248` | `thread_pool.rs` |
| `core/AffinityHelpers.hpp:11,76-101` | `thread_pool.rs::pinning_for_capacity`, `with_pinning_for_capacity`; wired at `chunk_fetcher.rs:324` |
| `core/StreamedResults.hpp:27-158` | `streamed_results.rs` |
| `core/BlockFinder.hpp:35-218` | `raw_block_finder.rs::RawBlockFinderCoordinator` |
| `core/BlockFetcher.hpp:38-687` | `block_fetcher.rs`; `process_ready_prefetches` at `:696` (vendor 427/463) |
| `rapidgzip/huffman/HuffmanCodingReversedBitsCached.hpp:32-136` | `huffman_reversed_bits_cached.rs`; wired for FIXED litlen at `deflate_block.rs` canonical path |
| `rapidgzip/huffman/HuffmanCodingShortBitsCachedDeflate.hpp:22-280` | `huffman_short_bits_cached_deflate.rs` (+ `rfc_tables.rs` helpers); distance HC uses §1 `code_cache()` at LUT fill |
| `rapidgzip/huffman/HuffmanCodingShortBitsMultiCached.hpp:24-269` | `huffman_short_bits_multi_cached.rs`; wired for DYNAMIC canonical decode in `deflate_block.rs` (bootstrap leg) |
| `core/Prefetcher.hpp:234-336` | `prefetcher.rs:232-343` (`FetchMultiStream`); **wired** at `chunk_fetcher.rs:304` (`3×16` vendor defaults) |
| `core/Prefetcher.hpp:60-225` | `prefetcher.rs:41-211` (`FetchNextFixed`, `FetchNextAdaptive`) |
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

## Ported primitives — honest status (May 2026)

Listed in dependency order. Status reflects **x86_64+isal production routing**,
which is the only platform that runs the parallel SM pipeline.

### §1. `HuffmanCodingReversedBitsCached` — WIRED, MARGINAL

`huffman_reversed_bits_cached.rs`; called from
`deflate_block.rs:1217` inside `read_internal_compressed_canonical_specialized`
for **FIXED (BTYPE=01) blocks only** during marker bootstrap. On x86_64,
DYNAMIC blocks dispatch to ISA-L's `IsalLitLenCode` at
`deflate_block.rs:874`, bypassing this module entirely.

Impact: FIXED blocks are rare in gzip -6…-9 data (the dominant
real-world case). This module runs in production but on a cold path.

### §2. `HuffmanCodingShortBitsCachedDeflate` — DEAD CODE

`huffman_short_bits_cached_deflate.rs` + `rfc_tables.rs`. Declared at
`mod.rs:27`; **imported by zero modules**. No `use` statement, no call
site, no unit test. Ported for vendor fidelity; superseded by §3 for the
canonical DYNAMIC path. Candidate for deletion or future non-ISA-L platform.

### §3. `HuffmanCodingShortBitsMultiCached` — DEAD ON x86_64

`huffman_short_bits_multi_cached.rs`; wired at `deflate_block.rs:1226`
inside canonical DYNAMIC decode. On x86_64+isal, canonical DYNAMIC is
**unreachable**: `read_internal_compressed_specialized` (line 831) routes
DYNAMIC to ISA-L at line 864 and routes only FIXED to canonical at
line 852. The non-ISA-L entry point (`deflate_block.rs:1340`,
`#[cfg(not(all(...)))]`) calls canonical for everything — but on that
platform, `decompress_parallel` returns `UnsupportedPlatform`.

Impact: zero on x86_64. This module would activate if/when the parallel
pipeline runs without ISA-L (post-Track-B Tier 3).

### §4. `FetchMultiStream` — WIRED ✅

`chunk_fetcher.rs:304` uses `FetchMultiStream::new(3, 16)`. Genuinely
production-active orchestration improvement.

### §§1–3 hidden dependency on ISA-L

The bootstrap leg's DYNAMIC block decode on x86_64 goes through
`isal_huffman.rs` (module-level `#![cfg(all(feature = "isal-compression",
target_arch = "x86_64"))]`), which wraps ISA-L's C FFI
(`set_and_expand_lit_len_huffcode`, `make_inflate_huff_code_lit_len`)
for table building. **The bootstrap is not pure Rust on x86_64.** Track B
Tier 3 must also replace `isal_huffman.rs`, not just `inflate_wrapper.rs`.

---

## Remaining work

Two independent tracks. **Track A (performance) is prerequisite** for
declaring the parallel SM pipeline production-ready. Track B
(C-dependency removal) is a separate goal that can follow Track A.

### Track A — Performance gate closure

The validation gate requires `parallel_elapsed < 0.5 × sequential_elapsed`
on neurotic. This is a **parallelism efficiency** question. ISA-L stays
as the inflate engine. The bottleneck is unknown — no profiling has been
done on neurotic since the Huffman/prefetch ports.

#### A1. Profile the parallel pipeline on neurotic

**Files:** `src/decompress/parallel/single_member.rs`,
`chunk_fetcher.rs`, `gzip_chunk.rs`.

**Action:** Run the parallel SM pipeline on neurotic with
`GZIPPY_LOG_FILE=/tmp/sm.log` on silesia.tar.gz (211 MiB uncompressed,
~68 MiB gzip-9). Analyze with `scripts/parallel_sm_log_summary.py`.
Separately, run `perf record` + `perf report` on the same input.

**Deliverable:** A breakdown of wall-time into:
- Block finding (`RawBlockFinderCoordinator` + `find_blocks`)
- Bootstrap (`bootstrap_with_deflate_block` in `gzip_chunk.rs:411`)
- Post-bootstrap ISA-L inflate (`decode_chunk_isal_impl`)
- Marker resolution (`apply_window` + `replace_markers`)
- Thread coordination (pool dispatch, result ordering in `StreamedResults`)
- Sequential confirmation path (chunks that fail speculation)

**Proof obligation:** The profile must account for ≥90% of wall time.
If no single component dominates, the bottleneck is coordination overhead
and the architecture itself may need work.

**Why it will work:** rapidgzip achieves similar parallelism on the same
architecture with the same ISA-L inflate engine. If gzippy's overhead
exceeds rapidgzip's, the profile will show where.

**OUT:** No code changes in this step. Measurement only.

#### A2. Run existing perf gates on neurotic

**Files:** `src/tests/routing.rs:456-515`
(`test_single_member_parallel_not_slower_than_sequential`),
`routing.rs:521-572`
(`test_single_member_parallel_silesia_class_not_slower_than_sequential`).

**Action:**

```bash
cargo test --release --features isal-compression -- \
  test_single_member_parallel_not_slower_than_sequential --ignored --nocapture
```

Record ratio. If `ratio ≥ 1.0` on 16 cores, parallel is currently
slower than sequential and the profile from A1 determines why.

**Proof obligation:** Both tests run without panic. Ratio is recorded.

**Why it will work:** The tests exist and are `#[ignore]` specifically
for neurotic runs.

**OUT:** No new tests. No code changes.

#### A3. Fix the measured bottleneck

Depends on A1/A2 results. Likely candidates and their fixes:

| Bottleneck | Fix | Files |
|---|---|---|
| Block finder too slow | Tune `find_blocks` parallelism; check false-positive rate on silesia | `block_finder.rs`, `gzip_block_finder.rs` |
| Bootstrap dominates chunk time | Bootstrap is O(32 KiB) — if slow, profile into Huffman table build vs decode; ISA-L table build may be the cause | `gzip_chunk.rs:411`, `isal_huffman.rs` |
| Marker resolution overhead | Profile `replace_markers.rs:146` AVX path; check alignment penalty | `replace_markers.rs`, `apply_window.rs` |
| Speculation failure rate | Too many chunks fail and enter sequential confirmation; tune chunk size via `adjusted_chunk_size_bytes` (`single_member.rs:65`) | `single_member.rs`, `chunk_fetcher.rs` |
| Thread pool overhead | Check pinning, rpmalloc arena per-thread init cost | `thread_pool.rs`, `rpmalloc_alloc.rs` |

**Proof obligation:** After fix,
`test_single_member_parallel_not_slower_than_sequential` reports
`ratio < 0.5` on neurotic with the 24 MiB synthetic fixture.

**Why it will work:** rapidgzip demonstrates this ratio is achievable
with ISA-L on the same input class. The fix targets the measured delta
between gzippy and vendor.

**OUT:** No pure-Rust inflate work. No ISA-L removal.

#### A4. Add real silesia perf test

**Files:** `src/tests/routing.rs` (new test), `src/tests/datasets.rs`.

**Action:** Add `test_single_member_parallel_silesia` that downloads
or skip-if-missing `silesia.tar.gz`, compresses to gzip-9 via gzippy,
and runs parallel vs sequential. Assert `ratio < 0.5` on ≥4 cores.

**Proof obligation:** Test passes on neurotic. Silesia is the standard
benchmark corpus; synthetic fixtures may not expose real-world
compression-ratio-dependent bottlenecks (e.g. high-entropy data has
fewer DEFLATE blocks, changing the speculation dynamics).

**Why it will work:** The existing `test_single_member_routing_multithread`
framework (lines 259-276) already does byte-exact parallel SM verification
on synthetic data. This adds a realistic corpus.

**OUT:** No `#[ignore]` removal for CI — silesia download would be
fragile in GHA. Keep `#[ignore]` with `perf gate` annotation.

#### A5. Remove `#![allow(dead_code)]` from hot-path modules

**Files:** All modules under `src/decompress/parallel/` with
`#![allow(dead_code)]`.

**Action:** For each module on the production hot path
(`single_member.rs`, `sm_driver.rs`, `chunk_fetcher.rs`,
`block_fetcher.rs`, `gzip_chunk.rs`, `deflate_block.rs`,
`inflate_wrapper.rs`, `isal_huffman.rs`, `replace_markers.rs`,
`apply_window.rs`, `thread_pool.rs`, `chunk_data.rs`,
`streamed_results.rs`, `block_map.rs`):
remove module-level `#![allow(dead_code)]`, fix or delete unused items.

Non-hot-path modules (`huffman_reversed_bits_cached.rs`,
`huffman_short_bits_cached_deflate.rs`,
`huffman_short_bits_multi_cached.rs`, `rfc_tables.rs`,
`huffman_symbols_per_length.rs`, `huffman_base.rs`,
`compressed_vector.rs`, `crc32.rs`, `statistics.rs`, `trace.rs`,
`gzip_definitions.rs`, `gzip_format.rs`, `window_map.rs`,
`prefetcher.rs`, `cache.rs`, `error.rs`, `bit_manipulation.rs`,
`block_finder.rs`, `raw_block_finder.rs`, `gzip_block_finder.rs`) may
retain the annotation if they have genuine pending consumers (seekable
reader scaffolding, non-ISA-L path activation after Track B Tier 3).

**Proof obligation:**
`cargo clippy --release --all-targets --features isal-compression -- -D warnings`
passes.

**Why it will work:** Hot-path modules are exercised by 8+ active
end-to-end tests. Dead items are either genuinely unused (delete) or
test-only (add `#[cfg(test)]`).

**OUT:** Non-hot-path modules with legitimate pending consumers keep
their `#![allow(dead_code)]`.

---

### Track B — C-dependency removal

Goal: no ISA-L (no compiled C) anywhere in the parallel SM hot path.
Independent of Track A. Track A proves the pipeline works at speed
with ISA-L; Track B replaces ISA-L with pure Rust without regressing.

#### B1. Pure-Rust inflate with stopping points

**Files:** `src/decompress/inflate/consume_first_decode.rs` (3562 lines).

**Action:** Add resumable inflate state and stopping-point checks,
mirroring the 4 patched ISA-L stopping points in
`packaging/isal-patches/igzip_inflate.c-stopping-points.patch` (395
lines). Reuse `StoppingPoints` from `inflate_wrapper.rs:50`.

```rust
// New in consume_first_decode.rs
pub struct InflateState {
    points_to_stop_at: StoppingPoints,
    stopped_at: StoppingPoints,
    // ... existing Bits reader fields, window, etc.
}

pub fn inflate_resumable(
    state: &mut InflateState,
    input: &[u8],
    output: &mut [u8],
) -> Result<usize, Error>;
```

State-machine checks at 4 sites:
1. After BFINAL/BTYPE decode → `END_OF_BLOCK_HEADER`
2. After BFINAL=0 block's last symbol → `END_OF_BLOCK`
3. After BFINAL=1 block's last symbol → `END_OF_STREAM`
4. After gzip stream header consumed → `END_OF_STREAM_HEADER`

**Proof obligation:** Byte-exact cross-test: for each deflate block
boundary in silesia.tar.gz (gzip-9), run ISA-L wrapper and pure-Rust
wrapper from the same bit offset with the same window, assert identical
output bytes and identical `stopped_at` / `bit_position` values in
`ReadStreamResult`. This is a **differential oracle test** — ISA-L is
the oracle.

**Why it will work:** The stopping-point patch to ISA-L is 395 lines of
state-machine checks at 4 well-defined sites. `consume_first_decode.rs`
already has the inflate loop structure; the delta is adding early-return
checks and a resumable state struct. rapidgzip's own ISA-L patch
demonstrates the 4 sites are sufficient.

**OUT:** No table-format changes to `consume_first_table.rs` or
`libdeflate_decode.rs`. No SIMD changes. Only control-flow additions
to the existing inflate driver.

#### B2. Pure-Rust `isal_huffman.rs` replacement

**Files:** `src/decompress/parallel/isal_huffman.rs` (ISA-L FFI for
Huffman table building), `deflate_block.rs:874` (call site).

**Action:** Replace `IsalLitLenCode::rebuild_from` and
`IsalDistCode::rebuild_from` with pure-Rust table builders that produce
the same triple-symbol-packed LUT layout. The table format is documented
at `isal_huffman.rs:54-59` (`LARGE_SHORT_SYM_LEN`, `LARGE_FLAG_BIT`,
etc.). The builder logic is in ISA-L's
`igzip_inflate.c:285-383` (`set_and_expand_lit_len_huffcode`) and
`:384-420` (`make_inflate_huff_code_lit_len`).

Alternatively: if §3 `HuffmanCodingShortBitsMultiCached` performs
within 1.2× of ISA-L's LUT on the bootstrap path, replace the ISA-L
table path with §3 directly (wiring the existing dead code). Measure
first.

**Proof obligation:** For every DYNAMIC block in silesia.tar.gz, the
pure-Rust table builder produces identical LUT entries as ISA-L's
builder. Differential test: build both tables from the same code-length
arrays, assert `short_code_lookup` and `long_code_lookup` arrays match
entry-for-entry.

**Why it will work:** The table format is fully documented in
`isal_huffman.rs` and the ISA-L source. The builder is ~100 lines of C.
Alternatively, §3 already implements a complete Huffman decoder for the
same symbol set — the question is performance, not correctness.

**OUT:** No changes to ISA-L's inflate engine itself (that's B1). No
changes to FIXED-block path (§1 already handles it).

**Dependency:** B2 is independent of B1. Can be done in either order.

#### B3. Pure-Rust `inflate_wrapper.rs` body

**Files:** `src/decompress/parallel/inflate_wrapper.rs` (540 lines on
x86_64+isal).

**Action:** Add `#[cfg(feature = "pure-rust-inflate")]` body that
implements the full `IsalInflateWrapper` public surface using the
resumable inflate from B1:

```
new, with_until_bits, set_window, set_stopping_points,
stopped_at, clear_stop, is_final_block, btype, tell_compressed,
read_stream, read_footer_at_current, reset_for_next_stream,
remaining_input, advance_input, at_end_of_stream, encoded_until_bits,
debug_points_to_stop_at, debug_stopped_at_raw,
debug_tmp_out_stopped_at, debug_block_state
```

Return types (`ReadStreamResult` at `:79`, `StoppingPoints` at `:50`)
stay unchanged. The 128 KiB staging buffer (`buffer` field) may be
unnecessary (pure-Rust inflate can read directly from the slice), but
retain it initially for behavioral equivalence with the ISA-L path.

**Proof obligation:** All existing end-to-end tests
(`test_single_member_routing_multithread`,
`test_marker_pipeline_actually_runs_on_x86_64_isal`, etc.) pass with
`--features pure-rust-inflate`. Byte-exact output on silesia.tar.gz.

**Why it will work:** B1 provides the resumable inflate; B2 provides
pure-Rust table building. B3 wires them into the existing wrapper API.
The wrapper's complexity is in the `refill_buffer` + `read_stream`
loop, which is a 130-line state machine already documented line-by-line
against the vendor.

**Dependency:** Requires B1 and B2.

#### B4. Benchmark and tier up

**Files:** New `benches/inflate_isal_vs_pure_rust.rs`.

**Action:** Benchmark `IsalInflateWrapper::read_stream` (ISA-L) vs
pure-Rust wrapper on silesia.tar.gz chunks. Measure per-chunk decode
throughput, not raw inflate speed — the production hot path runs through
the wrapper API with stopping points, refill buffer, and window management.

**Tiers:**
- **Tier 1:** pure-Rust within 1.5× of ISA-L on silesia. `--features pure-rust-inflate`.
- **Tier 2:** within 1.2×. Flip default; ISA-L still selectable.
- **Tier 3:** within 1.05×. Delete `vendor/isa-l`, `vendor/isal-rs`,
  `packaging/isal-patches/`. Also requires pure-Rust replacements for
  non-SM ISA-L users:
  - `backends/isal_decompress.rs` (T1 x86 sequential)
  - `backends/isal_compress.rs` (L0–L3 x86 compress) — **out of scope
    for this plan**; accept L0–L3 fast-path loss or land separately.
  - Detach `arena-allocator` from `isal-compression` in `Cargo.toml`.

**Proof obligation:** Tier 1 must not regress the validation gate from
Track A (ratio < 0.5 on neurotic). If it does, the pure-Rust inflate is
too slow for production and needs optimization before proceeding.

**Why it will work:** gzippy's pure-Rust inflate already runs at
competitive speeds for BGZF (`src/decompress/inflate/` powers BGZF
parallel decompress in production). The missing piece is stopping
points, not raw inflate throughput.

**OUT:** Tier 3's compress-path replacement (`isal_compress.rs`) is a
separate project. Tier 3 may ship with L0–L3 compress regression
acknowledged.

---

## Validation gate (revised)

On `neurotic` (16 physical x86_64, ISA-L available), via `make test-x86_64`.

### Performance gate (Track A — ISA-L in place)

All criteria must pass BEFORE Track B begins:

1. **Synthetic perf.**
   `test_single_member_parallel_not_slower_than_sequential` reports
   `ratio < 0.5` at T = min(16, num_cpus::get_physical()) on the 24 MiB
   low-entropy fixture.

2. **Silesia perf.**
   `test_single_member_parallel_silesia` (new, from A4) reports
   `ratio < 0.5` on real `silesia.tar.gz` at the same T.

3. **Routing / coordinator traps.**
   `test_single_member_routing_multithread`,
   `test_coordinator_boundary_search_runs_on_x86_64_isal`, and
   `test_parallel_sm_handles_fname_header` pass byte-perfect.
   `MARKER_PIPELINE_RUNS` and `COORDINATOR_BOUNDARY_SEARCH_RUNS`
   both increment.

4. **Hot-path dead-code allows gone** (A5).
   `cargo clippy --release --all-targets --features isal-compression -- -D warnings`
   succeeds.

5. **`perf stat` vs vendor.**
   ```
   perf stat -e dTLB-load-misses,major-faults,minor-faults,cycles \
       ./target/release/gzippy -d -c silesia.tar.gz > /dev/null
   ```
   compared against `rapidgzip -P 16 -c silesia.tar.gz > /dev/null`.
   Required: gzippy `minor-faults` within 1.5× of rapidgzip's; gzippy
   wall-time within 1.2×.

### Purity gate (Track B — ISA-L removed)

Separate from the performance gate. May ship in a follow-up PR:

6. **No external compiled deflate.**
   `cargo tree --release` shows no `isal-sys` / `isal-rs` in the
   production dependency graph; `vendor/isa-l`, `vendor/isal-rs`, and
   `packaging/isal-patches/` are removed. Performance gate criteria 1–5
   still pass with the pure-Rust inflate.

If (1)–(5) pass, the parallel SM pipeline is production-ready with ISA-L.
(6) is the stretch goal for C-free production.

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

## Reading order

- `src/decompress/parallel/single_member.rs` — entry,
  `MARKER_PIPELINE_RUNS` deletion trap.
- `chunk_fetcher.rs:257 drive` → `consumer_loop` →
  `submit_decode_to_pool` → `submit_post_process_to_pool`.
- Vendor `rapidgzip/ParallelGzipReader.hpp:495 read` and
  `rapidgzip/GzipChunkFetcher.hpp:312 processNextChunk`, side-by-side
  with the Rust.
