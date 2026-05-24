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
| B4 Throughput bench                     | ✅     | `benches/inflate_isal_vs_pure_rust.rs` (the inline `test_isal_vs_pure_rust_silesia_throughput` mentioned in earlier drafts was never landed — the bench harness alone is the gate) |


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

The 3 routing tests that motivated §5 are now ✅ on neurotic (verified
post-step-5/6: 33/33 pure-rust-inflate routing + 32/32 isal-compression).

### Known pre-existing failures (NOT introduced by §5; opening as
separate follow-ups; release-non-blocking per CLAUDE.md rules 4-5)

After the §5 step-6 cleanup landed (`a72d533`), `cargo test --release
--lib` on neurotic surfaces 5 failures that were also failing before
the §5 sequence (verified pre-existing at `b8f901d` via
`git stash && cargo test` by an Opus advisor on Apple-silicon Rosetta,
and re-classified here after a broader differential):

1. `decompress::tests::test_parallel_sm_propagates_errors_not_fallbacks`
   — pre-existing. Corrupt-input test asserts `Err(Decompression(_))`;
   actually gets `Err(InvalidArgument)` because corruption at certain
   offsets makes `is_likely_multi_member` (`format.rs:44+`) false-
   positive on the corrupt bytes, routing to multi-member parallel,
   which fails and falls through to `decompress_multi_member_sequential`
   (`mod.rs:188`), then libdeflate emits `BadData → InvalidArgument`.
   The corruption IS surfaced as `Err`, satisfying CLAUDE.md rule 5's
   spirit; the variant mismatch is a separate cleanup.

2. `decompress::parallel::inflate_wrapper::tests::with_until_bits_resume_non_byte_aligned_with_dict`
   — pre-existing. Synthetic flate2 fixture; `tell_compressed()` lands
   9 bits before `resume_at` on the resumable backend. Suspected
   subtable-entry `total_bits` accounting in `decode_huffman_body_resumable`
   for non-byte-aligned EOB. Production silesia routing (covered by
   `test_single_member_routing_multithread`, green) is unaffected.

3. `decompress::parallel::gzip_chunk::tests::cross_chunk_resume_silesia_gzip9_chunk0_handoff`
   — pre-existing class (same family as commit `03c8f48` "prime
   non-byte-aligned bit offset before set_dict"). zlib-ng resume at
   chunk0's reported end_bit fails. Production parallel-SM silesia
   integration is green (`make ship`); this synthetic test exercises a
   stricter contract than production uses.

4. `decompress::parallel::inflate_wrapper::tests::resumable_isal_oracle::stopping_points_match_at_every_block_boundary`
   — pre-existing. Fixture bug in `make_multi_block_deflate` at
   `inflate_wrapper.rs:841-889`: with `vec![0xAB; 300_000]` and flate2 1.x
   the encoder emits a single dynamic block + END_OF_STREAM rather
   than multi-block, so the ISA-L probe never observes
   `END_OF_BLOCK` and `ends.len() == 0`.

5. `decompress::parallel::inflate_wrapper::tests::resumable_isal_oracle::resume_with_window_matches_isal`
   — pre-existing. Same root cause as (4); panics with
   `index out of bounds: the len is 0 but the index is 0`.

The 5 tests above MUST eventually be cleaned up (especially (1) — the
multi→sequential fallback at `mod.rs:188` does violate CLAUDE.md rule
5 in letter), but they predate this branch and are orthogonal to the
§5 port. They are deferred to a separate "step 6 follow-up" cleanup PR.

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

**Implementation order (completed; retained for history)**:

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

§5 ships (parity); the throughput-exceed path is six phases, each
gated by `make ship` on neurotic. **Order**: fix failure #2 → Phase A →
Phase B → rest as measurements dictate. Failure #2
(`with_until_bits_resume_non_byte_aligned_with_dict`) is suspected
subtable `total_bits` accounting at `resumable.rs:812-818` — **must
land before Phase B** (Phase A is decoder-independent and safe to land
first).

- **Phase A — Close the page-fault gap. CLOSED 2026-05-24.**
  Fresh `perf record` on neurotic against real silesia.tar.gz shows
  `asm_exc_page_fault + clear_page_erms` at **17.2%** of weighted
  cycles — matches rapidgzip's documented 17%. The arena-allocator
  (`Cargo.toml:39,50-51` → `Vec<_, RpmallocAlloc>` for `ChunkData::data`
  / `data_with_markers`) + per-worker LIFO pool
  (`chunk_buffer_pool.rs:108-204`) shipping on `main` already closed
  this gap. The 2.76× pure-Rust-parallel vs T=1-libdeflate ratio on
  real silesia is NOT page-fault bound — see Phase B "Instrument
  before optimizing." If a future regression brings page-faults back,
  the levers (in `plans/pure-rust-perf.md`) are: worker-local
  pre-touch → MADV_HUGEPAGE → `#[global_allocator] = RpMalloc`.

  **Try in order (decreasing safety):**
    1. **Worker-local pre-touch of pool buffers.** The
       consumer-thread prewarm regressed −50%
       (`chunk_buffer_pool.rs:84-88`); worker-local has not been
       tried. After `bind_worker_pool_index`
       (`chunk_buffer_pool.rs:124`), write a zero to each pooled
       Vec's first byte per 4 KiB page.
    2. `MADV_HUGEPAGE` on chunk buffers ≥ 2 MiB. New code in
       `rpmalloc_alloc.rs::types::u8_with_capacity` or
       `chunk_buffer_pool.rs::take_u8`.
    3. `#[global_allocator] = rpmalloc::RpMalloc`. Highest risk
       (`chunk_buffer_pool.rs:94-96` flags it "unproven"; prior
       mimalloc/jemalloc tries regressed).

  Largest expected win; independent of §5.
- **Phase B — Instrument the per-chunk path THEN optimize.** Real
  silesia per-chunk decode is p50=47ms; the inflate-only bench says
  pure-Rust does ~5-8ms for a 1.6 MB chunk. ~85% of wall time is
  NOT inner inflate. Step zero: add per-chunk timing spans
  (`bootstrap`, `inflate_to_markers`, `replace_markers`,
  `window_publish`, `output_copy`) and emit p50/p95 per span via
  the existing GZIPPY_LOG_FILE trace. Then SIMD inflate primitives
  (`vector_huffman`, `simd_huffman`, `two_level_table`, `packed_lut`,
  `combined_lut`, `bmi2` — already production in
  `decode_huffman_cf_vector` at `consume_first_decode.rs:571+`) get
  wired into `decode_huffman_body_resumable` (`resumable.rs:793-852`)
  **only if** measurement shows inflate dominates. **Pre-req for
  SIMD path: failure #2 fixed.** Otherwise Phase B work goes to
  bootstrap amortization or speculation depth (closer to Phase E)
  per the measurement. See `plans/pure-rust-perf.md` "Phase B —
  Instrument before optimizing."
- **Phase C — Architecture-specific dispatch.** `target_feature` +
  CPUID runtime dispatch (`multiversion` crate). AVX2 + AVX-512 +
  NEON variants of `decode_*_resumable`'s inner loops. Bench each ISA
  flavor against ISA-L's equivalent. **Cross-cutting note**: lifting
  the arm64 gate requires undoing arch gates on every parallel-SM
  module (`chunk_buffer_pool.rs:1-4`, `rpmalloc_alloc.rs:1-4`, etc.),
  not just `sm_cfg.rs`.
- **Phase D — Pipeline overlap.** (1) CRC32 via hardware CLMUL
  interleaved with decode; today computed per-chunk after decode
  completes. (2) **NOTE: `apply_window` already runs on the worker**
  via `run_post_process_task` (`chunk_fetcher.rs:1405-1412`). What
  remains on the consumer is `publish_subchunk_windows` — the
  WindowMap index write. Moving that earlier is the remaining D
  opportunity *if* profiling shows the consumer stalls on it.
- **Phase E — Speculation depth + SIMD BlockFinder.**
  `RawBlockFinderCoordinator`'s scan loop replaced by 4-byte SIMD
  pattern match on block-header candidates. Speculate two-three
  boundaries ahead per worker instead of one.
- **Phase F — Memory bandwidth.** Non-temporal stores for outputs
  >500 MiB. NUMA-aware worker pinning. `MADV_POPULATE_WRITE` on
  output. Marginal individually, multiplicative with A-E.

Each phase is its own branch + PR with its own neurotic bench
measurement. Don't pre-commit to all six; abandon any that doesn't
beat its prior measurement.

## Out of scope (Tier 3 / separate projects)

- Remove `vendor/isa-l`, `isal-rs`, `packaging/isal-patches/` (requires sequential compress/decompress replacements)
- `backends/isal_decompress.rs` / `isal_compress.rs` pure-Rust replacements
- Seekable index reader (`IndexFileFormat.hpp`)
- BZIP2, ZLIB-format decoders

## Reading order

- `single_member.rs` — entry, `MARKER_PIPELINE_RUNS` deletion trap
- `chunk_fetcher.rs:350 drive` → consumer_loop → submit_decode_to_pool
- Vendor `GzipChunkFetcher.hpp:312 processNextChunk` side-by-side with Rust

## Troubleshooting (Beyond parity work)

### If Phase A doesn't move the bench number — first 3 things to check

1. **Is `arena-allocator` actually on?** `cargo tree --features
   pure-rust-inflate | grep rpmalloc-sys` must list it. If missing,
   the feature graph broke.
2. **Is the pool being hit?** Add `eprintln!` of `TAKE_U8_HITS` /
   `MISSES` (`chunk_buffer_pool.rs:206-207`) at end of
   `test_single_member_parallel_silesia_class_not_slower_than_sequential`.
   Hit-rate < 50% on iters 2-3 means workers aren't returning buffers
   — check `chunk_data.rs:1079-1085` Drop is firing.
3. **Is the silesia-class fixture re-compressing on every iter?** The
   test (`routing.rs:639+`) loops 3× and the fixture build runs each
   time the test starts. Check the bench is measuring decode only.

### If Phase B's SIMD primitives produce wrong output — first 3 things to check

1. **Failure #2 status.** If
   `with_until_bits_resume_non_byte_aligned_with_dict` is still red,
   the subtable `total_bits` bug at `resumable.rs:812-818`
   (`consume(TABLE_BITS=11)` then `consume_entry(subtable.raw())`) is
   biasing every bit-position downstream. Fix first or the SIMD path
   inherits the bug.
2. **Stopping-point bookkeeping at literal clusters.**
   `decode_huffman_cf_vector` (`consume_first_decode.rs:582+`) has no
   yield mid-block; `decode_huffman_body_resumable`
   (`resumable.rs:800+`) must check `bit_position() >=
   encoded_until_bits` after every batched cluster, not every
   literal.
3. **`copy_match_windowed` window precondition.** The window-stitched
   copy at `resumable.rs::copy_match_windowed` is per-byte. SIMD bulk
   replacement requires `distance <= out_pos - window_head`.
   Off-by-one reads uninitialized window bytes — the bug surfaces
   only on cross-chunk boundaries
   (`cross_chunk_resume_silesia_gzip9_chunk0_handoff` would re-red).

### How to capture a flame-graph on neurotic

```bash
ssh -J neurotic root@10.30.0.199
cd ~/gzippy-dev   # the worktree the bench job uses
cargo build --release --features pure-rust-inflate
perf record -F 999 -g --call-graph dwarf -- \
    ./target/release/gzippy -d -c benchmark_data/silesia-gzip.tar.gz > /dev/null
perf script | inferno-flamegraph > /tmp/flame-pure-rust.svg
# Target: asm_exc_page_fault + clear_page_erms < 25% combined CPU
```

### How to verify isal-compression production path stays green after any pure-rust change

```bash
# Local (30s):
cargo test --release --features isal-compression -- routing

# Authoritative (neurotic):
ssh -J neurotic root@10.30.0.199 \
  'cd ~/gzippy-dev && cargo test --release --features isal-compression -- \
     test_single_member_parallel_silesia_class_not_slower_than_sequential \
     --ignored --nocapture'
# Ratio must stay < 0.5 on the 16-core homelab box.
```

