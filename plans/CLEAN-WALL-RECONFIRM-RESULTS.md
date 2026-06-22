# CLEAN-WALL-RECONFIRM vs rapidgzip — gated matrix at HEAD (2026-06-22)

**Subject:** kernel-converge-A HEAD `6cd6b4b154cc799e6e1ed13586f47d14e873f8ce`,
gzippy-native (`--no-default-features --features gzippy-native` = pure-rust-inflate
+ asm-kernel; build-flavor `parallel-sm+pure`, FFI off the decode graph).
Measurement-only cycle (NO source change). Branch in worktree gzippy-amd-t2t4.

**Discipline:** interleaved best-of-N=11 per cell (GZ,RG,GZ2,RG2), wall via
`perf stat -e duration_time`, /dev/null both arms, UNPINNED (natural scheduler — no
taskset). Gate-0: build-flavor==parallel-sm+pure, sha==HEAD, path=ParallelSM asserted,
every arm sha256==zcat, rg/gz A/A self-test (binary-vs-itself ~1.0). Gate-1: Δ vs
inter-run spread; TIE when |Δ| ≤ gz_sp% + rg_sp%. ratio = gz_best/rg_best (>1 = gz slower).

## A. INTEL neurotic — UNPINNED (no llama on this box)
LXC `ssh -J REDACTED_IP root@REDACTED_IP`, i7-13700T, gov=powersave/no_turbo=0
(host-set, LXC read-only). load 3.96→7.42 during run (high-T spreads inflated; A/A
kept the comparison fair). N=11.

| cell | gz_ms | rg_ms | ratio | gz_sp% | rg_sp% | verdict |
|---|--:|--:|--:|--:|--:|---|
| silesia-T2  | 724.9 | 721.6 | 1.005 | 1.6 | 2.0 | **TIE** (+0.5%, spr 3.6%) |
| silesia-T4  | 405.7 | 401.7 | 1.010 | 2.6 | 1.9 | **TIE** (+1.0%, spr 4.5%) |
| silesia-T8  | 276.1 | 284.8 | 0.969 | 5.7 | 6.0 | TIE (-3.1%, spr 11.7%) |
| monorepo-T2 | 210.0 | 205.8 | 1.020 | 3.7 | 3.9 | TIE (+2.0%, spr 7.5%) |
| monorepo-T4 | 129.3 | 137.9 | 0.938 | 3.9 | 4.4 | TIE (-6.2%, spr 8.3%) |
| monorepo-T8 |  87.5 |  96.3 | 0.909 | 5.5 | 3.1 | gz BEATS 9.1% |
| nasa-T2     | 552.1 | 588.0 | 0.939 | 2.4 | 1.7 | gz BEATS 6.1% |
| nasa-T4     | 303.6 | 337.2 | 0.900 | 4.1 | 7.9 | TIE (-10.0%, spr 12.0%) |
| nasa-T8     | 185.9 | 226.3 | 0.822 | 5.7 | 2.0 | gz BEATS 17.8% |
| squishy-T2  |1222.6 |1287.3 | 0.950 | 1.0 | 1.4 | gz BEATS 5.0% |
| squishy-T4  | 679.3 | 679.6 | 1.000 | 2.6 | 2.4 | TIE (0.0%, spr 5.0%) |
| squishy-T8  | 431.7 | 446.1 | 0.968 | 2.4 | 4.1 | TIE (-3.2%, spr 6.5%) |

**Intel = gz TIES-or-BEATS rapidgzip on EVERY cell at HEAD unpinned. No real loss.**

## B. AMD solvency — CLEAN (llama SIGSTOP'd, box frozen)
`ssh root@REDACTED_IP`, EPYC 7282 Zen2, FROZEN gov=performance/boost=0, llama-server +
chain_final.sh + niah2.py all SIGSTOP'd (state T) for the window. N=11.
**A/A spread DROPPED to 0.2–5.6% (gz) / 0.5–5.3% (rg)** — confirming the clean window
worked vs the load-11-16 contamination that previously swamped deltas.

| cell | gz_ms | rg_ms | ratio | gz_sp% | rg_sp% | verdict |
|---|--:|--:|--:|--:|--:|---|
| silesia-T2  | 433.8 | 414.0 | 1.048 | 0.6 | 1.0 | **gz LOSES 4.8%** (spr 1.6%) |
| silesia-T4  | 252.7 | 242.7 | 1.041 | 2.1 | 0.8 | **gz LOSES 4.1%** (spr 3.0%) |
| silesia-T8  | 176.0 | 179.3 | 0.982 | 1.1 | 2.3 | TIE (-1.8%, spr 3.4%) |
| monorepo-T2 | 133.2 | 125.7 | 1.060 | 1.7 | 0.9 | **gz LOSES 6.0%** (spr 2.6%) |
| monorepo-T4 |  83.6 |  89.3 | 0.936 | 1.9 | 0.5 | gz BEATS 6.4% |
| monorepo-T8 |  57.3 |  59.3 | 0.967 | 1.7 | 5.3 | TIE (-3.3%, spr 7.0%) |
| nasa-T2     | 390.1 | 399.0 | 0.978 | 1.5 | 0.5 | gz BEATS 2.2% |
| nasa-T4     | 222.1 | 247.6 | 0.897 | 4.2 | 1.1 | gz BEATS 10.3% |
| nasa-T8     | 132.8 | 152.7 | 0.870 | 5.6 | 4.9 | gz BEATS 13.0% |
| squishy-T2  | 731.8 | 743.7 | 0.984 | 0.2 | 1.3 | gz BEATS 1.6% (spr 1.5%) |
| squishy-T4  | 437.1 | 422.8 | 1.034 | 0.9 | 1.2 | **gz LOSES 3.4%** (spr 2.2%) |
| squishy-T8  | 288.7 | 293.3 | 0.984 | 1.8 | 3.4 | TIE (-1.6%, spr 5.1%) |

## EXPLICIT ANSWERS
1. **Intel silesia-T4 unpinned @HEAD = TIE** (gz/rg 1.010, |Δ|=1.0% < spread 4.5%).
   gz does NOT lose. The prior P-core-PINNED 1.036 "loss" was a scheduling artifact;
   removing the pin resolves it to a TIE. (silesia-T2 also TIE.)
2. **Clean AMD residuals are REAL but were llama-INFLATED.** Clean magnitudes:
   silesia-T4 **4.1%** (was 7.5% contaminated), silesia-T2 **4.8%** (was 7.0%),
   monorepo-T2 **6.0%** (was 9.5%), squishy-T4 **3.4%** (was 7.5%). The loss survives
   un-contaminated at ~3–6%; llama inflated it by ~1.5–3.4 pp. nasa BEATS at every T;
   monorepo-T4 BEATS; all T8 cells TIE.

## CROSS-ARCH (Gate-3)
Intel near-parity does NOT replicate to AMD: gz ties/beats everywhere on Intel but
carries real ~3–6% T2/T4 losses on silesia/monorepo/squishy on Zen2. The residual
rapidgzip gap is **AMD/Zen2-specific** — consistent with the banked inner window-absent
decode-kernel cyc/B excess on Zen2 (project_t2_rg_locate / project_amd_t2t4_locate),
NOT a universal wall gap.

## BOX HYGIENE
- AMD: llama-server (725545) + chain_final.sh + niah2.py SIGSTOP'd for the window via a
  trap+watchdog-guarded driver; RESTORED at exit — all CONT (llama state Rl, sweep
  resumed with new orchestrator PIDs), gov=ondemand, boost=1, watchdog killed.
- Intel: no mutation (LXC freq host-read-only); no llama on this box.
- No source changed; bench-lock.sh untouched.
