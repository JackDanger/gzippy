# Session 2026-05-28 — perf optimization attempts, all falsified

## Goal

Pure-rust inflate within 1pp of ISA-L FFI on neurotic silesia-large at T=16.

## Current state (this session's measurements)

| Path                                             | Median MB/s |
|--------------------------------------------------|-------------|
| Pure-rust libdeflate-LUT (production default)    | ~670        |
| Pure-rust ISA-L LUT inner (GZIPPY_RESUMABLE_ISAL_INNER=1) | ~630 (-6%)  |
| ISA-L FFI (isal-compression build)               | ~1130       |

Gap: **41-44% behind ISA-L FFI**. Goal (1pp = ~1119 MB/s) not met.

## Levers attempted and falsified

1. **Route C v3.7 / v3.9** (dynasm-emit x86_64 asm literal decode): at
   parity with rustc on 3350-block silesia (369/405 vs 364/400 MB/s).
   Refill strategy doesn't matter — rustc emits equivalent codegen.

2. **S1 u32 packed multi-literal store**: 10-trial neurotic A/B at +0.4%
   (within noise). LLVM + x86 store-buffer coalescing already merges
   the 4 adjacent byte stores.

3. **S2 bulk window-copy in slow path**: at parity. The slow path is
   too rare on parallel-SM (only first 32 KiB of each 1 MiB chunk hits
   it). Kept the new code for structural clarity.

4. **L1 MADV_HUGEPAGE**: **-38% regression**. khugepaged contention +
   madvise call latency exceeded the savings from fewer faults.

5. **GZIPPY_RESUMABLE_ISAL_INNER**: **-6% regression** on production
   silesia. The May 27 memory note claiming +4.7pp gain was on the
   marker-phase decoder, not the production bulk decoder. Memory
   updated.

## Why everything failed

Fresh `perf record` on neurotic shows:
- `__memmove_avx_unaligned_erms` (libc): **18.07% CPU**
- `clear_page_erms` (kernel): **13.26% CPU**
- top inflate symbol: 8.21% (~15% total inflate)
- Allocator total: ~33.8% absolute CPU

**The 41-44% throughput gap to ISA-L FFI lives in the allocator, not
in the inflate inner loop.** Single-lever attempts on inflate can save
at most ~15% × 50% = 7.5% absolute CPU — invisible on a 30%+
allocator-dominated workload.

The kernel zeros every page on first touch for security (CVE
mitigation). For a 212 MB output buffer split across 17 chunks of
12 MiB each, that's ~51K page-fault round trips into the kernel.

## What would actually work (multi-day, beyond this session)

1. **Daemon-mode CLI**: amortize the page-zeroing cost across many
   decompressions in a long-lived process. The current one-shot CLI
   pays the full allocator tax on every invocation.

2. **Parallel pre-warm on worker threads**: the prior Z-allocator
   attempt failed because the prewarm was on the consumer thread
   (single-threaded). A correct implementation would have workers
   prewarm their own buffer pools in parallel, hiding the cost
   under existing worker startup.

3. **Full ISA-L bulk-decoder port** (multi-week per CLAUDE.md): port
   `decode_huffman_code_block_stateless` to Rust. ISA-L's bulk
   decode reportedly handles ~252 MB to gzippy's 200 MB per worker —
   a per-byte cycle delta that BMI2 PEXT + AVX2 in Rust intrinsics
   should be able to match.

4. **Memmove tracing**: the 18.07% `__memmove_avx_unaligned_erms`
   needs symbolized flamegraph (build with `debug = true`) to
   identify which gzippy function owns it. Likely candidates: the
   chunk-reorder buffer copy, `ChunkData::data` final-buffer copy,
   or `absorb_isal_tail`.

## Session deliverables (8 commits)

- `e642d7f` `8774609` `107ee19` `50d2930` `976a656`: Route C v3.7-v3.9
  + falsification docs.
- `dd86cc3` `dbb9831`: S1 attempt + falsification.
- `cea4d7c` `e794e70`: S2 attempt (kept for structural clarity) +
  falsification.
- `94f0710`: fresh perf data + the pivot to allocator.
- `9421664` `f6eba0f`: L1 attempt + falsification.

Plus this summary doc and memory updates
(`feedback_scalar_inflate_at_parity.md`,
`project_inflate_not_bottleneck.md`, refreshed
`project_isal_lut_port_landed.md`).

## Bottom line

The goal as stated **cannot be achieved by inflate-inner-loop scalar
optimization alone**. The session confirmed this empirically with 5
falsifications across the inflate-inner-loop dimension.

## Late session update: symbolized memmove

Built with `force-frame-pointers + debuginfo=1`, captured fresh perf
record with `--call-graph dwarf`. The 19.10% memmove decomposes:

  - 2.84% `emit_backref_ring` (marker u16 ring)
  - 1.61% `Vec::extend_from_slice` from `drain_to_output`
  - 1.82% `copy_within` in `clean_unmarked_data`
  - All three: **marker-bootstrap phase only**

The bulk inflate (`decode_huffman_body_resumable`, `copy_match_windowed`)
contributes essentially zero memmove. This means:
- The 13.26% `clear_page_erms` is the only TRULY allocator-side cost.
- The 18% memmove is **shared between pure-rust and ISA-L builds**
  (same bootstrap code).
- The 41% throughput gap really IS in the bulk inflate inner loop,
  even though scalar levers all measure parity.

The remaining bulk-inflate lever is **structural** rather than scalar:
- Speculative-parallel LUT lookups (vector_huffman re-attempt;
  CLAUDE.md 2026-05-27 authorizes)
- AVX2 multi-byte literal output via vpshufb (different shape than
  the falsified S1 u32 store)
- Yield-check elision restructure in FASTLOOP (CLAUDE.md authorizes)

Plus a ~5% marker-phase lever from shrinking the u16 ring to u8.

See `docs/perf/2026-05-28-memmove-symbolized.md` for full call stacks.

## FINAL UPDATE: C+T3 simplification ships +1.9%

After advisor consultation (the consult required by the goal directive),
the advisor recommended **Lever C (FASTLOOP yield-check elision) +
T3-simplify (vendor 2-extra-literal shape)** as the single change with
highest measured probability of paying back. The advisor explicitly
ruled OUT levers A (speculative-parallel LUT — speculation-recovery
trap), B (vpshufb — same trap as A), and D (u8 marker ring — caps at
0.8%).

Implemented the T3-simplify portion of the advisor's bundled
recommendation (replaces gzippy's 4-literal cap + 6 carry paths with
vendor libdeflate's 2-extra-literal shape per
`decompress_template.h:381`).

**Measured result on neurotic 20-trial interleaved A/B (clean release):**

| Statistic | Pre-T3 | Post-T3 | Δ      |
|-----------|--------|---------|--------|
| Median    | 887    | 904     | +1.9% |
| Mean      | 927    | 944     | +1.8% |

13/20 wins (65%), 6 losses, 1 tied. **First non-falsified lever this
session.** It works because it REMOVES code (3rd + 4th literal
lookahead) that vendor explicitly measured as a pessimization, rather
than ADDING optimization that LLVM already does.

See `docs/perf/2026-05-28-c-t3-confirmed-1-9-percent.md`.

## Session ledger (final)

| Outcome      | Count |
|--------------|-------|
| Falsified at parity / regress | 6 |
| Confirmed wins | 1 |
| Correction of contaminated measurement | 1 |
| Symbolized perf findings | 1 |
| Advisor consultations (per process rule) | 2 |

21 commits shipped. Goal not met but credible path established with
advisor sign-off and one demonstrated positive lever.

## Late update: C-fastloop iter-count batch — INFINITE LOOP BUG, REVERTED

Attempted the C part of the advisor's bundled C+T3 recommendation
(commit `762ac0a`). Replaced inner fastloop `while in_pos < in_fl_end
&& out_pos < out_fl_end { decode_one_symbol!(); }` with iter-count
`for _ in 0..safe_iters`.

**Bug**: when `safe_iters = 0` near the chunk tail (e.g.,
`in_fl_end - in_pos < 8` causing `safe_iters_in = 0`), the for loop
ran zero iterations, `continue`d to outer loop, which re-entered the
fastloop with same bounds (still `< in_fl_end` and `< out_fl_end`
because those thresholds are looser than the safe_iters
thresholds), looping forever.

Caught by `cargo test resumable` hanging for 4+ hours
(235 minutes CPU per stuck process before kill). Reverted in commit
`6a7b72e`.

**Final advisor consultation**: ship `6a7b72e` as session deliverable;
remaining levers (vpshufb / u8 marker ring / fixed C-fastloop) share
the same hung-test failure mode; expected value is negative after
the 4-hour debug tax. Open next session with fresh clean-build
re-measure of the gap before picking the next lever.
