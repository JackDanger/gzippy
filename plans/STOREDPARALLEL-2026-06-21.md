# StoredParallel T≥2 loss vs rapidgzip — localization (2026-06-21)

**STAMP: NOT-YET-LAW.** Intel i7-13700T LXC (guest 199), FROZEN (no_turbo=1,
gov=performance, BENCH_LOCK=quiet runnable_avg=1.25). Single arch — **AMD/Zen2
owed** before LAW. Subject sha 1ee49fc7 (origin/kernel-converge-A HEAD at start).
Build: gzippy-native (`--no-default-features --features gzippy-native`,
`RUSTFLAGS=-C target-cpu=native`), build-flavor=parallel-sm+pure (C-FFI OFF).
Sink `/dev/null` all arms; even-P-core pin (0,2,4,…); N=15 interleaved; sha==zcat
Gate-0 PASS for every arm.

## STEP 0 — routing reality: only `pure_stored` uses the StoredParallel COPY path
GZIPPY_DEBUG on the three "stored" corpora:
- **pure_stored** (100 MB, 100% stored blocks) → `StoredParallel` → the parallel
  copy path (`stored_split::fill_and_crc`). THE path under test.
- **storedheavy** → classified `StoredParallel` but at runtime the stored prefix is
  only **8.6%** of output (`prefix_out=8617627 < expected/2`), so the demotion gate
  fires: `StoredParallel demote → ParallelSM`. It decodes via the **ParallelSM marker
  pipeline**, NOT the copy path.
- **storedmix** → classified `ParallelSM` directly (first block not stored).

So storedheavy/storedmix losses are the **ParallelSM pipeline tax on cheap-to-decode
data** (the known PARALLEL-SCALING-2026-06-20 finding), a DIFFERENT path. The actual
StoredParallel copy path is exercised only by pure_stored.

## STEP 1 — frozen reproduction (gz BASE vs rg, /dev/null, N=15, min wall ms)
| corpus | T1 | T2 | T4 | T8 | path |
|--------|----|----|----|----|------|
| pure_stored gz | 123.9 | 79.2 | 66.8 | 50.5 | StoredParallel copy |
| pure_stored rg | 71.7 | 50.4 | 45.7 | 31.1 | |
| **gz/rg** | **1.73** | **1.57** | **1.46** | **1.62** | gz LOSES every T |
| storedheavy gz/rg | 0.78 | 1.23 | 1.33 | 1.27 | ParallelSM (demoted) |
| storedmix gz/rg | 0.72 | 1.13 | 1.19 | 1.31 | ParallelSM |

Loss REPRODUCES frozen. pure_stored loses at EVERY T including T1 (1.73×) — it is
NOT a T≥2-only / pipeline-overhead signature for THIS path. storedheavy/storedmix
match the matrix: T1 win, T≥2 loss (the ParallelSM cheap-corpus tax, out of scope).
Box spread was high at T≥2 (spr 16–40%) — the rg ratios are robust (≫spread) but the
small fused-CRC delta below is NOT.

## STEP 2 — LOCALIZED mechanism (Gate-0 self-validated; attribution = HYPOTHESIS tier)

### (a) Phase timing (`GZIPPY_STORED_PHASE_TIMING=1`, pure_stored)
The entire wall is `fill_and_crc` (parallel copy + CRC). `alloc_zero` ≈18 µs
(`vec![0u8;100MB]` is lazy/calloc — NOT a memset cost) and `verify_write` ≈4 µs
(write_all to /dev/null is free). T1 109 ms → T8 30 ms.

### (b) perf counters gz BASE vs rg, pure_stored (the decisive signal)
| metric (T1) | gz | rg | ratio |
|-------------|----|----|-------|
| wall ms | 137 | 72 | 1.90 |
| **page-faults** | **27471** | **3345** | **8.2×** |
| cpu_core cycles | 181 M | 96 M | 1.89× |
| instructions | 212 M | 92 M | 2.31× |
| cache-misses | 4.58 M | 1.82 M | 2.52× |

| metric (T8) | gz | rg |
|-------------|----|----|
| wall ms | 54 | 43 |
| page-faults | 27642 | 22173 |
| task-clock ms | 170 | 164 |
| CPUs utilized | 3.20 | 3.82 |

**Reading (HYPOTHESIS-tier attribution, NOT a causal verdict):** gz materializes the
WHOLE 100 MB output as one fresh `vec![0u8; expected_size]` and first-touches every
page during the copy → ~25 600 minor faults (== 100 MB / 4 KiB), single-threaded at
T1. rg streams chunked output with reused buffers → 3 345 faults (~8×). gz also runs
~2× the cycles/instructions for the same memcpy. At T8 the gz faults parallelize
across the copy threads and the gap collapses (task-clock 170≈164, gap 1.27×): the
T1 fault-in is a SERIAL bottleneck that parallelizes — consistent with faults being a
real lever, but this is monotonic-with-T evidence, not an isolated removal oracle.

## STEP 3 — fix landed (byte-exact micro-fix; TIE) + recommendation

### Landed: FUSED copy+CRC in `fill_and_crc` / `copy_runs_fused_crc`
Before: copy the whole partition, THEN a SECOND full pass `crc32(out_slice)`
(re-reads multi-MB output from DRAM, cache-cold). After: hash each run's bytes while
hot in cache from the copy (one pass). `GZIPPY_STORED_SPLIT_CRC=1` restores the old
split for A/B. Byte-exact (runs are contiguous+ordered → incremental hash == whole-
slice CRC); 12/12 stored_split tests pass; sha==zcat on silesia/monorepo/nasa/
pure_stored/storedheavy/storedmix × T1/T4/T8.

**Gated result (N=15, frozen):** FIX/BASE min-wall = 0.962–0.973 on ALL FOUR
pure_stored cells (~3–4% faster, directionally consistent) — but **Δ < box spread
(9–40%) ⇒ TIE, NOT a gated win** (CLAUDE.md Gate-1). It does NOT close the rg gap
(FIX still 1.41–1.68× rg). KEPT as a correctness-neutral memory-traffic reduction
(byte-identical, no regression), labelled a TIE — not a finding about the cause.
The separate-CRC pass is NOT the dominant lever; the page-faults + copy are.

### Recommended follow-on (the actual StoredParallel lever — NOT an obvious byte-exact one-liner, so NOT forced this turn)
**Stream the stored copy in chunks to the writer with a REUSED buffer instead of one
monolithic 100 MB `vec![0u8; expected_size]`** — mirrors rg's chunked output and
should cut page-faults ~8× (27k→~3k) at T1. This changes StoredParallel's
"verify-CRC-before-writing-any-byte" safety contract (it currently buffers the whole
output, verifies, then writes once — stricter than gzip(1)); streaming would write as
it goes (partial output on a late CRC mismatch, like gzip(1) and the existing
ParallelSM path) and fold the CRC incrementally. That is a deliberate redesign with a
correctness-contract change — pre-register the falsifier and verify byte-exact +
gated before landing. Secondary: investigate the 2× instruction count for the copy
itself (gz 212 M vs rg 92 M instr at T1) — possible non-SIMD / per-run-overhead copy.

## CAVEATS
- NOT-YET-LAW: Intel frozen only; AMD/Zen2 owed.
- perf-counter attribution is HYPOTHESIS-tier — no removal oracle isolated the
  page-fault lever; the T1→T8 gap-collapse is supporting (monotonic) evidence only.
- pure_stored is a synthetic CORNER (100% stored). Real archives rarely hit this
  path (storedheavy at 8.6% stored already demotes away from it).
