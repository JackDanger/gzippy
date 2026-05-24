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


| Item                                    | Status | Proof                                                                                                                  |
| --------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| B1 ResumableInflate + 4 stopping points | ✅     | `inflate/resumable.rs`; 25 unit tests; differential oracle in `inflate_wrapper.rs:1276+` (`ResumableInflate2` vs ISA-L) |
| B2 Bootstrap without C                  | ✅     | `deflate_block.rs` canonical path when `!USE_ISAL_INFLATE`; §3 cached-Huffman bootstrap exercises end-to-end           |
| B3 Pure-Rust wrapper body               | ✅     | `inflate_wrapper.rs` pure backend uses `ResumableInflate2`; all 33 pure-rust-inflate routing tests green on neurotic   |
| B3a Block scratch sizing                | —      | Obsoleted by §5: `ResumableInflate2` writes directly into caller's output with a 32 KiB sliding window. No per-block scratch exists. B3a band-aid (commit 2eff70f) deleted in §5 step 6. |
| B4 Throughput bench                     | ✅     | `benches/inflate_isal_vs_pure_rust.rs` + inline `test_isal_vs_pure_rust_silesia_throughput`                            |


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

Three pure-rust-inflate routing tests fail on neurotic against
HEAD (`60f411c`):

- `test_marker_pipeline_runs_on_btype01_heavy_input` — panics with
  `ResumableInflate("Generic match overflow: out_pos=8552442 length=8
  output.len=8552448")`. Math: session = `decode_start (32K from
  set_window) + max_new (128K) + PER_BLOCK_HEADROOM (8M) = 8552448`;
  the single zlib L1 block emits ~8.13 MiB. Root cause is the
  run-to-completion block decoder, not session accumulation.
- `test_coordinator_boundary_search_runs_on_x86_64_isal` — same panic,
  same fixture. (Name is historical; cfg-gates both backends.)
- `test_prefetch_next_filesize_accept_fires` — `PREFETCH_NEXT_FILESIZE_ACCEPT
  did not increment (2 -> 2)`. Independent of the match overflow;
  needs root-causing once §5 lands and the first two go green.

## §5 — Pure-Rust DEFLATE inflate with stopping points (option 2)

**Architecture decision** (May 2026): two decoders per BTYPE.

| Caller                                   | Decoder                                | Yield mid-block? |
| ---------------------------------------- | -------------------------------------- | ---------------- |
| BGZF, scan_inflate, sequential decompress | existing `decode_huffman_*` (fast)     | no               |
| `ResumableInflate::read_stream`          | new `decode_huffman_*_resumable`       | yes              |

Faithfulness to vendor (`vendor/.../gzip/isal.hpp:254-356`): ISA-L writes
incrementally into the caller's `output` with an internal sliding window
(~32 KiB via `tmp_out_buffer`). Our resumable decoders do the same; the
non-resumable decoders stay untouched so BGZF/sequential pay no
yield-check tax. This is what the band-aid `session` buffer (B3/B3a) was
faking.

**Files**:

- `src/decompress/inflate/resumable.rs` (new) — `ResumableInflate2`
  holding a `[u8; 32768]` ring buffer + pending-match state. Replaces
  the `session: Vec<u8>` accumulator.
- `src/decompress/inflate/resumable_decoders.rs` (new) — `decode_stored_resumable`,
  `decode_fixed_resumable`, `decode_dynamic_resumable`. Each yields by
  returning `Ok(YieldedMidBlock { pending_match })` instead of erroring
  with `WriteZero`. Resume = re-enter with same `(litlen, dist, pending_match)`.
- `src/decompress/parallel/inflate_wrapper.rs` — pure-rust backend
  switches `inner` from `ResumableInflate` to `ResumableInflate2`.
- `src/decompress/inflate/consume_first_decode.rs` — `ResumableInflate`
  marked deprecated; deleted once `inflate_wrapper.rs` is the only
  caller and migrates over.

**Match-copy when distance reaches past `output[0]`**: the new module's
hot path is `copy_match_windowed(output, out_pos, distance, length,
window: &[u8; 32768], window_head: usize)`. Branch on
`distance <= out_pos` — fast path (existing logic) vs window-stitched
path. Window is updated after each `read_stream` from the trailing
≤32 KiB of bytes just emitted to `output`.

**Tiered bench gates** (preserved from earlier draft):

- **Tier 1** — feature `pure-rust-inflate` is opt-in; throughput ≥ 1/1.5 ×
  ISA-L acceptable. Gate: `benches/inflate_isal_vs_pure_rust.rs` green
  on neurotic.
- **Tier 2** — `pure-rust-inflate` becomes the default; throughput ≥ 1/1.2 ×
  ISA-L. Same bench, stricter threshold.
- **Tier 3** — `vendor/isa-l` + `isal-rs` + `packaging/isal-patches/`
  deleted; throughput ≥ 1/1.05 × ISA-L. Detaches
  `isal-compression`/`arena-allocator` coupling in `Cargo.toml`.
  Migrates `backends/isal_decompress.rs` (T1 x86 sequential),
  `backends/isal_compress.rs` (L0-L3 fast compress; out of port scope),
  and any other ISA-L call site.

**Implementation order**:

1. Scaffold `resumable.rs` + `resumable_decoders.rs` with stubs returning
   `Err(NotImplemented)`. Wire `inflate_wrapper.rs` behind a
   `cfg(feature = "resumable-decoders")` flag so both backends coexist
   during the cut.
2. Land `decode_stored_resumable` first (simplest — no Huffman). Validate
   via existing oracles.
3. Land `decode_fixed_resumable` (static tables, just yield logic).
4. Land `decode_dynamic_resumable` (full path).
5. Flip the feature default to on; re-run the 3 red routing tests.
6. Delete `ResumableInflate` + `session` field from `consume_first_decode.rs`.
7. Delete B3a band-aid (commit 2eff70f).

## Beyond parity — path to exceed rapidgzip

Once §5 ships (parity within ~5%), the throughput-exceed path is six
phases, each gated by `make ship` on neurotic and abandoned if its
measurement doesn't beat the prior:

- **Phase A — Close the page-fault gap.** `chunk_buffer_pool.rs:73-82`
  notes gzippy spends ~40% of CPU in `asm_exc_page_fault`/`clear_page_erms`
  vs rapidgzip's ~17%. Try (in order): `Vec<T, RpmallocAlloc>` for chunk
  buffers (per-Vec, faithful to vendor `FasterVector<u8, RpmallocAllocator>`),
  then `MADV_HUGEPAGE` on output, then `#[global_allocator] = RpMalloc`.
  Largest expected win; independent of §5.
- **Phase B — Bring gzippy's SIMD inflate primitives to the parallel-SM
  resumable path.** `vector_huffman`, `simd_huffman`, `two_level_table`,
  `packed_lut`, `combined_lut`, `bmi2` already ship for BGZF and
  sequential decompress; extend `decode_*_resumable` to dispatch through
  them. This is where pure-Rust beats ISA-L's general inflate on the
  code-length distributions gzip(1) produces.
- **Phase C — Architecture-specific dispatch.** `target_feature` +
  CPUID runtime dispatch (`multiversion` crate). AVX2 + AVX-512 + NEON
  variants of `decode_*_resumable`'s inner loops. Bench each ISA flavor
  against ISA-L's equivalent.
- **Phase D — Pipeline overlap.** (1) CRC32 via hardware CLMUL
  interleaved with decode; today computed per-chunk after decode
  completes. (2) Move `apply_window` from consumer thread to worker as
  post-decode step.
- **Phase E — Speculation depth + SIMD BlockFinder.** `RawBlockFinderCoordinator`'s
  scan loop replaced by 4-byte SIMD pattern match on block-header
  candidates. Speculate two-three boundaries ahead per worker instead
  of one.
- **Phase F — Memory bandwidth.** Non-temporal stores for outputs
  >500 MiB. NUMA-aware worker pinning. `MADV_POPULATE_WRITE` on
  output. Marginal individually, multiplicative with A-E.

Each phase is its own branch + PR with its own neurotic bench measurement.
Don't pre-commit to all six; abandon any that doesn't beat its prior
measurement.

## Out of scope (Tier 3 / separate projects)

- Remove `vendor/isa-l`, `isal-rs`, `packaging/isal-patches/` (requires sequential compress/decompress replacements)
- `backends/isal_decompress.rs` / `isal_compress.rs` pure-Rust replacements
- Seekable index reader (`IndexFileFormat.hpp`)
- BZIP2, ZLIB-format decoders

## Reading order

- `single_member.rs` — entry, `MARKER_PIPELINE_RUNS` deletion trap
- `chunk_fetcher.rs:257 drive` → consumer_loop → submit_decode_to_pool
- Vendor `GzipChunkFetcher.hpp:312 processNextChunk` side-by-side with Rust

