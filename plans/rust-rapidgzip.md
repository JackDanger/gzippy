# Rapidgzip → gzippy parallel single-member port

Structural port of rapidgzip's `ParallelGzipReader → GzipChunkFetcher → BlockFetcher → GzipChunk → IsalInflateWrapper` chain into
`src/decompress/parallel/`. GNU gzip formats only (single-member,
multi-member, BGZF).

## End-state architecture (implemented)

```
classify_gzip (sm_cfg::PARALLEL_SM)
  → single_member::decompress_parallel
  → sm_driver::read_parallel_sm
  → chunk_fetcher::drive
  → gzip_chunk (marker bootstrap + IsalInflateWrapper read_stream)
  → apply_window / replace_markers
```

**Feature matrix (x86_64):**


| Build                           | Bootstrap DYNAMIC                      | Post-bootstrap inflate                      |
| ------------------------------- | -------------------------------------- | ------------------------------------------- |
| `isal-compression` (default SM) | `isal_huffman.rs` (C FFI)              | patched ISA-L via `inflate_wrapper.rs`      |
| `pure-rust-inflate`             | §3 `HuffmanCodingShortBitsMultiCached` | `ResumableInflate` via `inflate_wrapper.rs` |
| both features enabled           | §3 canonical (pure-Rust wins)          | `ResumableInflate` (pure-Rust wins)         |


Cfg gates live in `src/decompress/parallel/sm_cfg.rs`:

- `PARALLEL_SM` — orchestration compiles
- `USE_ISAL_INFLATE` — C inflate + C Huffman table build

## Completed items

### Track B — C-free SM hot path ✅


| Item                                    | Status        | Proof                                                                                                                     |
| --------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| B1 ResumableInflate + 4 stopping points | ✅             | `consume_first_decode.rs`; differential oracle in `inflate_wrapper.rs`; `test_resumable_end_of_stream_header_after_reset` |
| B2 Bootstrap without C                  | ✅ (§3 bypass) | `deflate_block.rs` canonical path when `!USE_ISAL_INFLATE`; no LUT entry-for-entry test (§3 is end-to-end equivalent)     |
| B3 Pure-Rust wrapper body               | ✅             | `inflate_wrapper.rs` pure backend; `test_pure_rust_parallel_sm_e2e` (pure-rust-inflate only)                              |
| B3a Block scratch sizing                | ✅             | `consume_first_decode.rs:3174` — session = `decode_start + max_new + PER_BLOCK_HEADROOM` so one block can overflow caller buffer by up to PER_BLOCK_HEADROOM, surplus drains on next `read_stream` (matches ISA-L tmp_out) |
| B4 Throughput bench                     | ✅             | `benches/inflate_isal_vs_pure_rust.rs` + inline `test_isal_vs_pure_rust_silesia_throughput`                               |


### Track A — Infrastructure ✅ (perf gates deferred)


| Item                  | Status | Notes                                                                                                                       |
| --------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| A1 Trace events       | ✅      | `chunk_fetcher.rs` drive_begin/drive_end/speculative_*                                                                      |
| A4 Silesia perf test  | ✅      | `test_single_member_parallel_silesia` (#[ignore])                                                                           |
| A5 Hot-path dead_code | ✅      | Module-level `#![allow(dead_code)]` removed from 14 hot-path modules; item-level allows for seekable-index scaffolding only |


### §§1–4 Huffman + prefetch


| §   | Module                              | x86+isal                | x86+pure-rust         |
| --- | ----------------------------------- | ----------------------- | --------------------- |
| 1   | `huffman_reversed_bits_cached`      | FIXED bootstrap         | FIXED bootstrap       |
| 2   | `huffman_short_bits_cached_deflate` | unused                  | unused                |
| 3   | `huffman_short_bits_multi_cached`   | unreachable (isal path) | **DYNAMIC bootstrap** |
| 4   | `FetchMultiStream`                  | wired                   | wired                 |


## Remaining proof on neurotic (not code)

Run this yourself (don't wait for the user) on 16-core x86_64 homelab via 'ssh -J neurotic root@10.30.0.199'

```bash
# Routing + deletion traps (both feature sets)
cargo test --release --features isal-compression -- routing
cargo test --release --no-default-features --features pure-rust-inflate -- routing

# Perf gates (#[ignore])
cargo test --release --features isal-compression -- \
  test_single_member_parallel_not_slower_than_sequential \
  test_single_member_parallel_silesia --ignored --nocapture

# B4 bench
cargo bench --release --features isal-compression -- \
  --bench inflate_isal_vs_pure_rust -- --nocapture
```

Three pure-rust-inflate routing tests known to fail before the B3a
headroom fix (commit 2eff70f); re-verify against HEAD on neurotic:

- `test_marker_pipeline_runs_on_btype01_heavy_input` — directly fixed
  by B3a (match overflow on chunk_size+window boundary).
- `test_prefetch_next_filesize_accept_fires` — likely downstream
  (decode aborted before reaching last-chunk prefetch); re-test.
- `test_coordinator_boundary_search_runs_on_x86_64_isal` — name is
  historical (gated for both backends since pure-rust-inflate landed);
  re-test, and rename to drop the `_isal` suffix once it stays green.

## Out of scope (Tier 3 / separate projects)

- Remove `vendor/isa-l`, `isal-rs`, `packaging/isal-patches/` (requires sequential compress/decompress replacements)
- `backends/isal_decompress.rs` / `isal_compress.rs` pure-Rust replacements
- Seekable index reader (`IndexFileFormat.hpp`)
- BZIP2, ZLIB-format decoders

## Reading order

- `single_member.rs` — entry, `MARKER_PIPELINE_RUNS` deletion trap
- `chunk_fetcher.rs:257 drive` → consumer_loop → submit_decode_to_pool
- Vendor `GzipChunkFetcher.hpp:312 processNextChunk` side-by-side with Rust

