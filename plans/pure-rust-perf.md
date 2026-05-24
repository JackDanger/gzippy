# Pure-Rust parallel SM — performance plan

§5 of `rust-rapidgzip.md` is correctness-complete: pure-Rust resumable
inflate + vendor-faithful parallel SM ports, both backends green on
neurotic (33/33 pure-rust-inflate routing + 32/32 isal-compression).
This document is the *performance* follow-up: drive the
`--features pure-rust-inflate` path from "works" to "matches and then
exceeds rapidgzip's patched-ISA-L parallel SM."

## Baseline measurements (2026-05-24, neurotic, 16-core x86_64)

### Inner inflate — full-stream silesia deflate body, single-threaded

`benches/inflate_isal_vs_pure_rust.rs` (best of 3 runs):

| Decoder                                 | Throughput |
| --------------------------------------- | ---------- |
| Patched ISA-L via `IsalInflateWrapper`  | 799 MB/s   |
| Pure-Rust `ResumableInflate2`           | 334 MB/s   |
| **Ratio (ISA-L / pure-Rust)**           | **2.39×**  |

Plan §5 Tier 1 gate: ≤ 1.5× ISA-L. **Currently failing by ~0.9×.**

### End-to-end parallel SM — silesia-class 24 MiB, T=16

`test_single_member_parallel_silesia_class_not_slower_than_sequential`
(`--features pure-rust-inflate`):

- Parallel/sequential ratio: **4.04** (parallel is 4× SLOWER than
  sequential). Test gate wants < 0.5.

The e2e gap is bigger than the inner-inflate gap because per-chunk
overhead (allocator, page faults, cross-worker coordination) doesn't
amortize when the inner decode is slow.

### vs rapidgzip parallel SM on x86_64

gzippy's `--features isal-compression` parallel SM ≈ rapidgzip
parallel: same patches, same ISA-L underneath, same chunked-decode
architecture. That path is covered by `make ship`'s authoritative
bench. The **pure-Rust** path is currently slower than gzippy's own
sequential decoder; therefore much slower than gzippy-isal-parallel,
and hence much slower than rapidgzip parallel — easily 4–8× on
typical fixtures.

## Six phases to exceed rapidgzip (from `rust-rapidgzip.md §"Beyond parity"`)

Each phase is its own branch + PR + bench-on-neurotic. Abandon any
that doesn't beat its prior measurement.

### Phase A — Close the page-fault gap (highest leverage)

`src/decompress/parallel/chunk_buffer_pool.rs:73-82` documents the
profile: gzippy spends ~40% of CPU in `asm_exc_page_fault +
clear_page_erms`; rapidgzip spends ~17%. Largest expected win;
independent of decoder tuning.

Try in order:
1. `Vec<T, RpmallocAlloc>` for chunk buffers (per-Vec, faithful to
   vendor `FasterVector<u8, RpmallocAllocator>`). Lowest risk.
2. `MADV_HUGEPAGE` on the output Vec.
3. `#[global_allocator] = rpmalloc::RpMalloc`. Highest risk (prior
   mimalloc/jemalloc tries regressed).

Bench gate: `make ship` parallel-SM throughput improves measurably on
silesia. Re-measure the perf-profile flame-graph to confirm the
page-fault CPU% comes down.

### Phase B — SIMD inflate primitives on the resumable path

gzippy already has `vector_huffman`, `simd_huffman`, `two_level_table`,
`packed_lut`, `combined_lut`, `bmi2` (`src/decompress/inflate/`,
`src/decompress/`). They're production for BGZF + sequential
decompress. They're NOT on the parallel SM resumable path because
`ResumableInflate2`'s body is `decode_huffman_body_resumable` →
generic libdeflate-style table lookup with no SIMD batching.

Action: extend `decode_huffman_body_resumable` to dispatch literal-
batch and match-copy through the SIMD primitives. Where these
primitives beat ISA-L on the specific code-length distributions
gzip(1) produces, this is where pure-Rust wins against ISA-L's
general inflate.

Sub-tasks:
- Wire `vector_huffman` multi-symbol decode for short codes
  (≤ TABLE_BITS).
- Bring `double_literal` two-symbol cache for back-to-back short
  literals.
- Specialize `copy_match_windowed` for `distance > 32` (bulk
  `copy_match_safe`-style) and for `distance > out_pos` (window
  branch) separately.
- Drop the per-byte loop in `copy_match_windowed` for the common
  case where `distance > out_pos` is false.

Bench gate: inner-inflate ratio ≤ 1.2× ISA-L (Tier 2). E2E
parallel-SM ≥ 2× faster than sequential (the existing test gate).

### Phase C — Architecture-specific dispatch

`target_feature` + CPUID runtime dispatch (`multiversion` crate or
hand-rolled `is_x86_feature_detected!`). AVX-512, AVX2, NEON variants
of the inner-loop hot paths.

- AVX-512 BMI2 `pext`/`pdep` for bit-shuffling — speeds up Huffman
  decode and bit extraction.
- AVX2 256-bit copies for `length >= 32` matches.
- NEON dispatch for arm64. Today the parallel SM path is x86-only
  (gated by `sm_cfg::PARALLEL_SM`); arm64 falls back to libdeflate.
  Lift that gate once NEON variants exist.

Bench gate: per-ISA flavor compared against ISA-L's equivalent ISA
flavor.

### Phase D — Pipeline overlap

Two rapidgzip doesn't fully exploit:

1. **CRC32 in flight with decode.** Hardware CLMUL CRC32 issues at
   ~1 byte/cycle. Today gzippy + rapidgzip compute CRC32 after the
   chunk decode completes. Interleaving CRC compute with literal/
   match emission hides it almost entirely.

2. **`apply_window` on worker, not consumer.** `apply_window` resolves
   cross-chunk back-references via `replace_markers`. Currently runs
   on the consumer thread between chunks. Move it onto the worker as
   a post-decode step so the consumer immediately accepts the next
   chunk.

Bench gate: per-chunk wall time → not pipeline-stalled by serialized
post-decode work.

### Phase E — SIMD BlockFinder + deeper speculation

`src/decompress/parallel/raw_block_finder.rs` scans 8 KiB-bit windows
sequentially looking for valid block starts. Vendor's
`BlockFinder.hpp` + `blockfinder/DynamicHuffman.hpp` do the same.

- Multi-byte SIMD pattern match for `0b10`/`0b01` followed by HLIT/
  HDIST ranges that gate valid block headers. Probably 4-8× faster
  boundary scan.
- Speculate two-three boundaries ahead per worker instead of one.

Bench gate: time-to-first-output drops; T=16 utilization stays
> 14 cores.

### Phase F — Memory bandwidth tricks

After A-E, throughput approaches DRAM bandwidth. Then:
- Non-temporal stores for the output writer on very large files
  (> 500 MiB).
- NUMA-aware worker pinning. On 2-socket boxes, pin workers to the
  socket whose memory holds the input mmap.
- Pre-fault output with `MADV_POPULATE_WRITE`.

Marginal returns (5-15%) individually, multiplicative atop A-E.

## Discipline

Per the §5 retrospective:

1. **Bench before code.** Every phase opens with a flame-graph and a
   throughput measurement on neurotic. No phase commits until that
   measurement repeats green at the PR's bench gate.
2. **Get adversarial Opus reviews on judgment calls.** §5 dodged
   two infinite-loop bugs only because we paused for advisor
   diagnosis instead of patching blindly. Same rule here.
3. **Don't break the isal-compression path.** It's the production
   default. Every PR's CI must keep `isal-compression` green on
   neurotic.
4. **Monitor every long-running job.** 10-min health-check monitors
   for any `cargo bench` / neurotic ssh that might hang.

## Known issues to clean up while we're here (orthogonal to perf)

From `rust-rapidgzip.md "Known pre-existing failures"`:

1. `test_parallel_sm_propagates_errors_not_fallbacks` — silent
   multi→sequential fallback at `mod.rs:188` violates CLAUDE.md
   "no fallbacks" rule. Fix scope: gzip-format classifier + routing.
2. `with_until_bits_resume_non_byte_aligned_with_dict` — synthetic
   fixture; suspected subtable `total_bits` accounting in
   `decode_huffman_body_resumable`. Worth chasing because it might
   point to a real bit-accounting issue that limits Phase B.
3. `cross_chunk_resume_silesia_gzip9_chunk0_handoff` — zlib-ng
   resume at chunk-end-bit (same class as `03c8f48`).
4-5. `resumable_isal_oracle::*` — `make_multi_block_deflate` fixture
   bug on flate2 1.x. Trivial: replace fixture builder, repoint.

Pick these up opportunistically when the active phase touches the
relevant area, not as scheduled work.
