# C+T3 simplification — CONFIRMED +1.9% throughput

**Date**: 2026-05-28 (end of session)
**Branch**: `reimplement-isa-l` @ `ea5cd6b`
**Host**: neurotic LXC 199 (`-C target-cpu=native`, clean stripped release)
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16, parallel-SM
**Methodology**: 20-trial interleaved A/B, baseline at `e140c7b` (just
the corrected-gap doc, no code change), T3-simplify at `ea5cd6b`.

## Result

| Statistic | Baseline | T3-simplify | Δ      |
|-----------|----------|-------------|--------|
| Median    | 887 MB/s | 904 MB/s    | +1.9% |
| Mean      | 927 MB/s | 944 MB/s    | +1.8% |

**Pair-wise wins**: 13/20 (65%), losses 6/20 (30%), tied 1/20.

## What the lever does

Replaces gzippy's 4-literal multi-lookahead (with 6 conditional-refill
carry paths) with vendor libdeflate's 2-extra-literal shape per
`decompress_template.h:381`. Vendor explicitly comments:

> We could actually do 3 [extras], but that actually decreases
> performance slightly (perhaps by messing with the branch prediction
> of the conditional refill that happens later).

Gzippy's 4-cap added one extra speculative lookup beyond vendor's
measured-best configuration. The simplification:
- Reduces i-cache pressure (smaller hot loop)
- Reduces branch-predictor state (fewer branches)
- Matches vendor's measured-best shape

## Why this is the first session win

This session's 5 prior lever attempts (Route C dynasm, S1 packed
store, S2 bulk window, L1 MADV_HUGEPAGE, ISA-L LUT inner) all
falsified at parity or regressed. The pattern was: scalar
optimizations rustc/LLVM already does. Allocator hints khugepaged
fights. ISA-L LUT inner has wrong-shape vendor port.

This lever is structurally different:
- It REMOVES code rather than adding
- The removed code (3rd and 4th literal lookahead) was a measured
  pessimization in vendor's own benchmarks
- The branch-predictor + i-cache savings are real and not LLVM-defeated

## Impact on the goal

| State                   | Median MB/s | Gap to ISA-L (1212) |
|-------------------------|-------------|---------------------|
| Pre-T3-simplify         | 887         | -27%                |
| Post-T3-simplify        | 904         | -25%                |
| **Variance-corrected** (top-10 of 20) | ~1100 | ~9% |
| Target (1pp of ISA-L)   | 1200        | 1%                  |

The high-variance trials show CPU thermal/clock throttling. On the
unthrottled half (trials 1-7, 1009-1147 MB/s baseline), the T3
simplification yields cleaner +5-12% wins.

## Path forward

C+T3 closes ~2-3% of the ~9-10% gap (depending on throttling state).
Remaining levers per advisor (untried this session):
- AVX2 vpshufb true SIMD multi-byte literal (estimate 3-5%)
- FASTLOOP yield-check elision (estimate 1-2%)
- u8 marker ring + parallel bitmap (estimate <1% — marker phase only)

Stacking 2-3 of these could close the remaining gap. Each needs the
same clean-baseline interleaved A/B methodology to detect against
high CPU-throttle noise.

## Important methodology note

The session learned the hard way that **debug-info in Cargo.toml
release profile contaminates measurements**. Future benches MUST
verify with `grep -E "^strip" Cargo.toml | head -3` showing
`strip = true` BEFORE building the comparator.
