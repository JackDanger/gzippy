# STEP-0 BLOCKING GATE — FROZEN re-confirm of silesia-T4 +16% loss + util-deficit

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz sha:** 8081516b
(decode binary sha256[:16] = `b5eec9eed5feeb45` — byte-identical to the STEP-1 binary)
**rg:** `/root/oracle_c/rapidgzip-native` v0.16.0 sha[:16]=`41baa20fdfbdea24` (NATIVE ELF, not a wheel)
**Box:** Intel i7-13700T LXC (neurotic guest REDACTED_IP), **FROZEN** via host
`REDACTED_IP:/root/bench-lock.sh acquire 2400` (no_turbo=1, governor=performance pinned,
8 noisy neighbors frozen, DNS/VPN/proxy kept). **Stamp:** NOT-YET-LAW — single-arch Intel; AMD/Zen2 owed.
**Method:** MEASUREMENT ONLY (no decode-path src change). Instrument =
`scripts/bench/kernel-ab/intel_gz_rg_cycbyte_guest.sh` (+ `_analyze.py`), driven across
3 corpora by `step0_frozen_driver.sh`. N=13 interleaved (gzA,gzB,rgA,rgB / rep), /dev/null
both arms, perf-stat cpu_core PMU (instructions/cycles/LLC-miss/task-clock/duration_time).

## FREEZE LIFECYCLE (the #1 safety requirement)
- ACQUIRED: `acquire 2400` → watchdog ARMED, no_turbo←1, governor=performance.
  Printed **`BENCH_LOCK=quiet runnable_avg=1.50 (<=2.0)`** at acquire.
  (One non-fatal `wrmsr 0x620` segfault = the uncore-freq MSR pin failing; frequency is
  still pinned by no_turbo=1 + performance governor, and the measured GHz spread gate
  PASSED ≤0.15% on every cell — so frequency was stable regardless.)
- RELEASED + VERIFIED: `release` → **`RESTORE VERIFIED: no_turbo=0 uncore=0x guests-thawed`**,
  `BENCH_LOCK=released no_turbo=0 frozen_now=[] wd=inactive`. Box is thawed, neighbors back.
  (The post-release `verify` correctly prints `loaded` because neighbors have resumed — that
  is the restored state, not a frozen one.)

## GATE-0 (all reported; silesia loss cell)
- flavor=parallel-sm+pure, **path=ParallelSM** PASS; same /dev/null sink both arms.
- byte-exact: ref(zcat)=`028bd002c89c9a90` == gz == rg ✓ (211,968,000 B out).
- A/A wall self-test T4: gz 1.0003, rg 1.0033 — both ≤1.02 PASS (licenses the ratio).
- GHz: T4 gz 1.3940 / rg 1.3952, spreads gz 0.115% rg 0.063% (gate ≤1%) PASS; gz-vs-rg
  0.08% apart (frequency-fair). LLC-miss/B tiny (gz 0.0020 rg 0.0038) — not bandwidth-bound.
- instr inter-run spread T4: gz 0.700% / **rg 1.257% WARN** (rapidgzip's per-thread split is
  nondeterministic at T4 — same ~1% wobble seen loaded; the T4 instr gap is soft to ±1% but
  unambiguously SMALL, nowhere near the wall gap).
- procs_running during the silesia cell = 2–3 (our own decode threads); neighbors stayed frozen.

## RESULT — silesia T4 (the loss cell), FROZEN
| metric | gz(native) | rapidgzip | gz/rg |
|---|---|---|---|
| instr/B (mean) | 22.799 | 22.265 | 1.024 (+2.4%) [LOAD-IMMUNE, rg WARN] |
| cyc/B (best-of-N) | 9.561 | 9.250 | 1.034 (+3.4%) |
| IPC (best) | 2.378 | 2.392 | 0.994 (−0.6% ≈ PARITY) |
| GHz (mean) | 1.3940 | 1.3952 | 0.999 |
| task-clock ms | 1467.8 | 1423.4 | 1.031 |
| **CPUs-utilized** | **2.458** | **2.710** | **0.907** |
| **wall ms (best)** | **579.2** | **507.0** | **1.1424 (+14.2%)** A/A-licensed |

**CONSERVATION CLOSES:** wall_ratio **1.1424** ≈ cyc/B_ratio **1.0336** × util-deficit(rg/gz)
**1.1023** = **1.1393** (within 0.3% → no dimensional error). WALL gap share: per-byte
CPU-work **~24%** | parallel-utilization **~72–76%**. util-deficit(rg/gz) = **+10.2%**
(gz keeps 2.458 of 4 cores busy vs rapidgzip's 2.710) — Δ ≫ the ~0.3% A/A residual ⇒ real.

### Frozen vs loaded (STEP-1) — the deficit SURVIVES freezing
| | wall gz/rg | cyc/B gz/rg | util gz | util rg | util-deficit |
|---|---|---|---|---|---|
| LOADED (STEP-1) | +16.9% | +4.0% | 2.468 | 2.736 | +10.8% |
| **FROZEN (this)** | **+14.2%** | **+3.4%** | **2.458** | **2.710** | **+10.2%** |

The wall loss stays **well above 1.10** (1.1424). The util-deficit is **essentially unchanged**
(+10.8% → +10.2%); the absolute cores-busy magnitudes (2.46 / 2.71) held nearly identical
frozen vs loaded. The util-deficit is **NOT a contention artifact**.

## NEGATIVE CONTROLS — T4, FROZEN (must stay ≈TIE, util-deficit≈0)
| corpus | wall gz/rg | util gz | util rg | util-deficit | instr/B gz/rg |
|---|---|---|---|---|---|
| **nasa** | 0.9815 (gz WIN −1.8%) | 2.525 | 2.577 | +2.1% (≈0) | 0.745 (gz lower) |
| **monorepo** | 1.0138 (≈TIE +1.4%) | 2.555 | 2.563 | +0.3% (≈0) | 0.847 (gz lower) |

Both controls: wall ≈TIE/win AND util-deficit ≈0. (nasa is the small/noisy corpus — A/A wall
1.0266/1.0169 WARN, instr spread gz 8.3% WARN — so its conservation product is loose, but its
verdict (gz wins the wall, util-deficit≈0) is unambiguous.) ⇒ the +10.2% util-deficit is
**SILESIA-T4-SPECIFIC**, not an instrument bias.

## THE BLOCKING-GATE VERDICT → **BRANCH 1: utilization direction CONFIRMED real (→ STEP 1)**
- silesia-T4 wall ratio frozen = **1.1424 ≥ ~1.10** ✓ (the loss did NOT collapse on a quiet box).
- util-deficit **SURVIVES freezing**: gz 2.458 < rg 2.710, **+10.2%**, Δ ≫ spread ✓.
- nasa & monorepo stay ≈TIE with util-deficit ≈0 ✓.
⇒ The utilization-deficit decomposition is **NOT a contention artifact** — it is a real,
silesia-T4-specific structural property of gz's chunked-decode scaffold. Recommend **STEP 1:
repair/build the per-thread busy/idle instrument** to localize WHERE gz leaves ~0.25 cores
idle vs rapidgzip (chunk granularity / prefetch depth / consumer pacing / sync points in
`single_member` driver + `block_fetcher`). NOT yet a located cause — that is STEP-3's perturbation.

## OWED (NOT-YET-LAW)
- AMD/Zen2 replication of the +14% wall AND the util-vs-cyc decomposition.
- rg's T4 instr-count nondeterminism (1.26%) — inherent to its work-split; not gz's.
- The cause is NOT concluded here (STEP-0 confirms the deficit is real + survives freezing only).

## Reproduce
```
ssh root@REDACTED_IP /root/bench-lock.sh acquire 2400 && ssh root@REDACTED_IP /root/bench-lock.sh status
# build native gz on guest @8081516b (RUSTFLAGS=-C target-cpu=native, --no-default-features --features pure-rust-inflate)
ssh -J REDACTED_IP root@REDACTED_IP 'GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native N=13 PIN_T1=3 PIN_T4=2,4,6,8 bash /root/step0_frozen_driver.sh'
ssh root@REDACTED_IP /root/bench-lock.sh release && ssh root@REDACTED_IP /root/bench-lock.sh verify
```
