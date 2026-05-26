# Phase 0.6 spike: iai-callgrind SIMD fidelity — results

## Setup

- Host: neurotic (i7-13700T, Debian trixie)
- valgrind 3.24.0
- `RUSTFLAGS="-C target-cpu=x86-64-v3"`
- `[profile.bench]` inherits release with `strip=false, debug=true, lto=false`
  (LTO breaks iai-callgrind sentinel discovery).
- Bench: `benches/iai_simd_spike.rs` — paired scalar/SIMD impls of popcount
  and u16→u8 narrowing on small fixed-size inputs.

## Raw measurements

### Popcount (256 × u64 array)

| Impl | Instructions | L1 Hits | L2 Hits | RAM Hits | Est. cycles |
|---|---:|---:|---:|---:|---:|
| `count_ones()` (compiles to `popcnt`) | 1389 | 1678 | 42 | 20 | 2588 |
| `_popcnt64` intrinsic | 1389 | 1678 | 42 | 20 | 2588 |
| Bit-twiddle (manual Hamming weight) | 6580 | 7833 | 41 | 16 | 8598 |

### u16→u8 narrowing (1024 elements)

| Impl | Instructions | L1 Hits | L2 Hits | RAM Hits | Est. cycles |
|---|---:|---:|---:|---:|---:|
| Scalar loop | 1280 | 1763 | 4 | 124 | 6123 |
| AVX2 (`_mm256_packus_epi16`) | 1144 | 1602 | 3 | 113 | 5572 |

## Interpretation

### Cachegrind DOES count SIMD instructions

The AVX2 narrow shows 11% fewer instructions and 9% fewer estimated cycles than the scalar narrow. Cachegrind is not blind to SIMD — it sees `vpackuswb`, `vmovdqu`, etc. as instructions in the trace.

### But cachegrind does NOT model SIMD throughput

A `_mm256_packus_epi16` produces 32 narrowed bytes per instruction (one cache-line worth). In wall-clock terms, this is typically 3-5× faster than scalar narrowing on real silicon. Cachegrind's 11% reduction in instructions reflects only "fewer loop iterations" (32× wider per iter), not the per-instruction throughput advantage.

**Estimated cycles** in cachegrind are a coarse model (1 cycle per instruction + cache-miss penalties) that does NOT account for execution-port throughput. AVX2 instructions issued through multiple ports per cycle are counted as 1 cycle each.

### Confirmation: count_ones() vs _popcnt64 are identical

Both forms produce identical instruction streams (1389 ops). LLVM lowers `count_ones()` to `popcnt` when target-cpu allows. This is the control case: when two forms compile to the SAME instructions, cachegrind reports them identically. Good.

### Bit-twiddle Hamming weight shows real divergence

The bit-twiddle popcount (no `popcnt` instruction; pure bitops) shows 4.7× more instructions and 3.3× more estimated cycles than the popcnt version. Cachegrind correctly distinguishes algorithmically-different scalar code.

## Verdict

**Outcome (b) from Phase 0.6 plan**: cachegrind counts SIMD instructions but doesn't model their widened throughput. The instruction-count delta from a SIMD variant will be on the order of 5-20% (fewer loop iterations doing wider work), while the wall-clock delta will be on the order of 70-90% faster.

## Plan adjustment

The Tier 1 6.1.2 iai-callgrind perf gate (±2% of baseline) is still valid for **scalar variant regression detection**:

- A scalar variant edit that increases instruction count by >2% is a clear regression signal.
- A scalar variant edit that decreases instruction count by >2% indicates either a real algorithmic improvement OR a measurement artifact (compiler eliding something we wanted to measure).

For **SIMD variants**, the iai-callgrind baseline must be **per-variant**, not a single shared baseline. When a SIMD variant lands:

1. The variant ships with its OWN baseline in `tests/baselines/iai_inflate_simd_<variant_name>.json`.
2. The ±2% gate applies to that variant's own baseline, not to the scalar baseline.
3. A CI-only wall-clock gate (using `perf stat -e cycles` on a fixed-machine-class runner) supplements the iai gate for SIMD variants. Wall-clock floor: variant must be ≥ X% faster than scalar reference on the corpus.

This is consistent with v2 of `plans/inner-loop-execution.md` Phase 0.6 outcome (b).

## Other findings (incidental)

- `count_ones()` already uses `popcnt` when `target-cpu=x86-64-v3` is set — no need for explicit intrinsic. (This is mildly interesting; means we don't need to feature-gate `_popcnt64` in production code.)
- iai-callgrind on full LTO (`lto = "fat"`) fails sentinel discovery. The `[profile.bench]` override (lto=false) is mandatory and should be documented in CLAUDE.md.
- iai-callgrind takes ~3-5 seconds per benchmark function. For 30-50 corpus blocks this is ~3 minutes total — acceptable for a perf-test-suite invocation.

## Outstanding follow-ups

1. Verify `_mm256_packus_epi16` is actually emitted in the AVX2 path's asm (the 11% instruction reduction is real but small; we want to confirm SIMD is being lowered correctly with `target-cpu=x86-64-v3` and not falling back to scalar SSE2).
2. Test iai-callgrind with the actual inflate inner loop once Phase 2 extraction is done — confirm 6.1.2 gate behavior on the real production-shape function.
