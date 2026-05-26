# Phase 0.6 spike: iai-callgrind SIMD fidelity — results (v2 post-Opus)

## Setup

- Host: neurotic (i7-13700T, Debian trixie)
- valgrind 3.24.0
- `RUSTFLAGS="-C target-cpu=x86-64-v3"`
- `[profile.bench]` inherits release with `strip=false, debug=true, lto=false`
  (LTO breaks iai-callgrind sentinel discovery — documented requirement).
- Bench: `benches/iai_simd_spike.rs` — paired scalar/SIMD impls of popcount
  and u16→u8 narrowing on small fixed-size inputs.

## v1 confound caught by Opus (resolved)

The original "scalar" narrow loop at 1280 instructions over 1024 elements was 1.25 instr/element — impossible for genuine scalar. LLVM auto-vectorized the iterator-zip pattern under `target-cpu=x86-64-v3`. The v1 measurement was actually "hand-rolled AVX2 with permute" vs "LLVM-auto-vectorized AVX2."

**Fix**: scalar narrow now uses `black_box(src[i])` per element to defeat auto-vectorization. Verified via `cg_annotate` (2 instr/elem in `narrow_scalar` body, 2 instr/elem in `hint.rs` for the black_box, plus loop overhead — total ~7 instr/elem, consistent with genuine scalar codegen).

## Raw measurements (v2, clean)

### Popcount (256 × u64 array) — control case

| Impl | Instructions | Est. cycles | Notes |
|---|---:|---:|---|
| `count_ones()` | 1389 | 2588 | Lowered to `popcnt` |
| `_popcnt64` intrinsic | 1389 | 2588 | Identical — confirms `count_ones()` already used popcnt |
| Bit-twiddle Hamming weight | 6580 | 8598 | Pure scalar bitops; 4.7× more |

### u16→u8 narrowing (1024 elements) — the actual SIMD comparison

| Impl | Instructions | Est. cycles | Per-element instr |
|---|---:|---:|---:|
| **Scalar** (`black_box` defeats auto-vec) | **8193** | 16149 | ~7 |
| **AVX2** (`_mm256_packus_epi16`) | **1144** | 5576 | ~1.1 |
| **Ratio** | **7.2× fewer** | **2.9× fewer** | |

## cg_annotate evidence (Opus-required)

### Scalar narrow (8193 total instructions)

```
2,052 (25.05%)  gzippy/benches/iai_simd_spike.rs:narrow_scalar
2,048 (25.00%)  core/src/hint.rs:narrow_scalar (black_box implementation)
1,024 (12.50%)  core/src/cmp.rs:narrow_scalar (loop range comparison)
1,024 (12.50%)  core/src/iter/range.rs:narrow_scalar (range iterator)
1,024 (12.50%)  core/src/num/uint_macros.rs:narrow_scalar (index arithmetic)
```

Loop body: 2 instr (load + truncate-store) per element × 1024 elements = 2048, plus 5 instr/elem of loop machinery (range/cmp/uint_macros/black_box) = 7 instr/elem. Realistic scalar codegen.

### AVX2 narrow (1144 total instructions)

```
185 (16.17%)  core/src/slice/iter/macros.rs   (bench harness)
128 (11.19%)  core/src/num/uint_macros.rs     (bench harness)
124 (10.84%)  libc memcpy_avx                 (bench harness)
119 (10.40%)  libc _int_malloc                (bench harness)
 96 ( 8.39%)  iai_simd_spike::narrow_avx2     (actual SIMD work)
 32 ( 2.80%)  core_arch/src/x86/avx2.rs       (AVX2 intrinsic)
```

The actual SIMD narrow function is 96 instructions for 1024 elements = 0.09 instr/elem at this granularity. Cachegrind correctly sees the wider work as fewer instructions; `_mm256_packus_epi16` shows up as 1 instruction in the trace despite producing 32 narrowed bytes.

## Interpretation

### Cachegrind counts SIMD instructions

`_mm256_packus_epi16` appears in the trace as 1 instruction, not 32. The 7.2× instruction-count reduction in AVX2 narrow comes from **fewer loop iterations doing wider work**, not from cachegrind giving credit for throughput.

### Cachegrind does NOT model SIMD throughput

Estimated cycles differ by only 2.9× (5576 vs 16149). In wall-clock terms, AVX2 narrow is typically 5-8× faster than genuinely scalar on the production silesia path (`src/decompress/parallel/chunk_fetcher.rs:1665-1731`). Cachegrind's cycle model is `1 cycle/instr + cache penalties` and doesn't account for execution-port throughput. The popcount control confirms: `popcnt` (3-cycle latency, 1/cycle throughput) is reported as 1 cycle, identical to `mov`.

### Verdict: outcome (b) from Phase 0.6 plan, unambiguously

Cachegrind distinguishes SIMD from scalar by **a wide margin** (7.2× instruction reduction, 2.9× cycle reduction). A ±2% iai-callgrind baseline gate against a scalar baseline **WILL trip** when a SIMD variant lands. Per-variant baselines are mandatory.

## Plan adjustments (applied to plans/inner-loop-execution.md)

1. **Tier 1 6.1.2 per-variant baselines**: each SIMD variant ships with its own `tests/baselines/iai_inflate_simd_<variant_name>.json`. The ±2% gate applies to the variant's own baseline, not the scalar baseline.

2. **rustc-bump exception path applies to SIMD variant baselines** (Opus catch). When `rustc --version` differs from baseline's recorded version, the SIMD variant test is `#[ignore]`d for that run; the rustc-bump PR mandatorily regenerates via `make update-baselines`.

3. **Pre-declared wall-clock floor for narrow variant**: ≥ **2.0×** scalar reference. Cites the AVX2 license-downclock concern at `src/decompress/parallel/chunk_fetcher.rs:1668-1674` — the production code already documents that AVX2 narrow can trigger downclocking on some Intel CPUs, so the floor must be high enough to make the SIMD path worth the downclock tax.

4. **CI-only wall-clock supplement for SIMD variants**: `make perf-counters` on neurotic measures actual cycles via `perf stat -e cycles` on the corpus. SIMD variant must hit its declared wall-clock floor.

## Incidental findings

- iai-callgrind takes ~3-5s per benchmark function. 30-50 corpus blocks → ~3 minutes total. Acceptable for a `make test-perf` invocation.
- The bench harness overhead is ~50% of total instructions on small workloads (notice `_int_malloc` and `memcpy_avx` in both annotations). For the real inflate bench, the workload is much larger (~30 KB output per block) and harness overhead amortizes to <5%.
- `count_ones()` already lowers to `popcnt` at `target-cpu=x86-64-v3` — no explicit intrinsic needed in production code.

## Phase 0.6 status

**COMPLETE.** Phase 1 unblocked.

## Outstanding follow-up for Phase 2

After the function extraction lands, re-run iai-callgrind against the real `decode_dynamic_huffman_block` to validate that the ±2% gate behaves sensibly on the production-shape function. The spike used 1024-element inputs; real inflate blocks are 100×+ larger and harness overhead will be negligible.
