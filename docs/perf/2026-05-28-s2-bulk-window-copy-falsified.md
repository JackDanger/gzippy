# S2 — bulk window-copy + memmove in slow path: FALSIFIED at parity

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `cea4d7c`
**Host**: neurotic LXC 199 (`-C target-cpu=native`)
**Build**: `cargo build --release --features pure-rust-inflate`
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16, parallel-SM

## Implementation

Replaced the per-byte loop in `copy_match_windowed`'s slow path with three vectorizable phases:

1. **Phase 1**: source entirely in window → bulk memcpy via new
   `SlidingWindow::copy_logical_range_into` (LLVM lowers
   `copy_from_slice` to libc memcpy, SIMD-vectorized on neurotic).
2. **Phase 2**: source entirely in output, `dist >= remaining` →
   `copy_within` (memmove).
3. **Phase 3**: LZ77 overlap (`dist < remaining`) → per-byte loop (semantic preservation).

Output byte-identical to baseline on full silesia: md5
`b0ef8cdaac65b6f49fa045b84b8d7460`. All 1371 tests pass.

## Result

Interleaved 10-trial A/B on neurotic, internal `parallel_sm:v0.6` timer:

| Trial | Baseline | S2  |
|-------|----------|-----|
| 1     | 592      | 553 |
| 2     | 606      | 567 |
| 3     | 640      | 594 |
| 4     | 656      | 589 |
| 5     | 573      | 647 |
| 6     | 654      | 708 |
| 7     | 679      | 689 |
| 8     | 717      | 770 |
| 9     | 710      | 713 |
| 10    | 730      | 748 |

Units: MB/s.

- Baseline mean: **655.7 MB/s**
- S2 mean: **657.8 MB/s**
- **Δ = +0.3% — at parity.**

## Why falsified

The attribution doc (2026-05-28) attributed 4.14% absolute CPU to
`copy_match_windowed`. I assumed that was the slow-path per-byte loop.
It wasn't.

In parallel-SM at gzip-9, after the first 32 KiB of output of any 1 MiB
chunk is decoded, all subsequent matches have source entirely in
output (`dist <= 32K <= out_pos`). The FAST PATH handles those via
`copy_match_fast`. The slow path only fires:
- First 32 KiB of each chunk (window-touch matches)
- Yield-mid-copy (rare on 1 MiB output buffers)
- Insufficient FAST PATH margin at chunk tail

That's ~3% of chunk bytes. The 4.14% CPU was almost entirely FAST PATH
(`copy_match_fast`), which my change didn't touch.

## What would actually help

The dominant code is `copy_match_fast` (`consume_first_decode.rs:395+`),
which is already a heavily optimized 40-byte unrolled SIMD-shaped
loop. Beating that requires algorithmic changes (e.g. shorter unroll
for small matches, AVX2 wider stores) — risky given the existing
optimization.

Alternative: attack the bigger picture. Pure-Rust at ~655 MB/s vs
ISA-L FFI at ~1130 MB/s (~72% gap) suggests the lever is NOT
micro-optimization of any single function. Either:
- The full ISA-L bulk-decoder port (multi-day) — see memory note
  `project_isal_lut_port_landed.md` ("remaining gap requires
  bulk-phase port").
- Profiling on neurotic with fresh `perf record` to identify which
  function actually dominates today.

Three falsifications in a row (Route C v3.7/v3.9, S1 packed store, S2
bulk window copy) all targeted code rustc/LLVM was already
near-optimal at. The next move needs fresh perf data, not another
guess.

## Decision

**Keeping S2 in main, not reverting.** Throughput is at parity (no
regression) and the new code is structurally cleaner:
- Explicit Phase 1/2/3 split documents the three cases.
- `SlidingWindow::copy_logical_range_into` is a reusable helper.
- 1371 tests pass; silesia round-trip byte-identical.

Leaving an in-source comment pointing to this falsification doc so
future sessions know the perf hypothesis was wrong, but the code
structure is intentional.
