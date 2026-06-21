# CONSUMER-CONFIRM — resolve the consumer-thread instrument contradiction + test the real lever (FROZEN)

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz built @38dbbb81**
(decode path identical to origin 3e78cd51 + the env-gated `GZIPPY_PHYS_PIN` affinity
probe; `git diff` of the decode path vs 3e78cd51 is the affinity-only change).
RUSTFLAGS=`-C target-cpu=native`, `--no-default-features --features pure-rust-inflate`,
flavor=parallel-sm+pure, path=ParallelSM. Binary on guest `/dev/shm/gzrg-target/release/gzippy`
sha[:16]=`685eaba4dfb1d958`. **rg:** `/root/oracle_c/rapidgzip-native` sha[:16]=`41baa20fdfbdea24`.
**Box:** Intel i7-13700T LXC (guest REDACTED_IP) **FROZEN** via `REDACTED_IP:/root/bench-lock.sh
acquire 3600` (`BENCH_LOCK=quiet runnable_avg=1.00`, no_turbo=1, governor=performance,
procs_running=1 during the cells). **RELEASED + RESTORE VERIFIED** (`no_turbo=0 uncore=0x
guests-thawed`, frozen=[], wd=inactive). **Stamp: NOT-YET-LAW — single-arch Intel; AMD/Zen2 owed.**
Corpus `/root/silesia.gz` (68,229,982 B → 211,968,000 B; sha `028bd002c89c9a90`).

## SMT topology (guest cpuset = `0,2-5,8,10,12-17,19-21`)
`thread_siblings_list`: cpu0={0,1}, cpu2={2,3}, cpu4={4,5}, cpu8={8,9}, cpu10={10,11},
cpu12={12,13}, cpu14={14,15}; E-cores 16,17,19,20,21 are solo. **Even logical ids 0,2,4,8,10,…
are distinct physical cores; odd ids are their SMT siblings.**

## THE BUG IN THE DEFAULT WORKER PINNING (deterministic, code+/proc)
`thread_pool::pinning_for_capacity` cycles `core_affinity::get_core_ids()` in order. On this
cpuset that order is `[0,2,3,4,5,8,…]`, so for 4 workers it pins **worker0→cpu0, worker1→cpu2,
worker2→cpu3, worker3→cpu4** — and **cpu2/cpu3 are SMT siblings of ONE physical core.** Confirmed
by `/proc/<tid>/stat` field-39 placement (`_placement_probe.py`): baseline workers run on
`{0,2,3,4}` (two workers share physical core 1); the consumer (main thread) floats.
`GZIPPY_PHYS_PIN=1` selects one logical id per SMT sibling group → workers `{0,2,4,8}` (4 distinct
physical) + consumer→cpu10.

## STEP 0 — P1 baseline reproduced on a FRESH freeze (N=13)
| regime | gz/rg best | gz/rg med | gz cyc/B | note |
|---|---|---|---|---|
| UNPIN | **1.1803 (+18.0%)** | 1.1640 | ~11.0 | reproduces P1 +19.4%/+16.3% within spread |
| PIN5  | 1.0112 (+1.1%) | 1.0164 | 9.594 | ≈parity |
| PIN4  | 1.1500 (+15.0%) | 1.1464 | 9.600 | exact reproduction; util-deficit +11.0% |
GATE-0 all PASS (byte-exact gz==rg==zcat; /dev/null both; A/A ≤1.02; GHz spread ≤0.07%,
gz-vs-rg ≤0.07% apart; conservation closes per regime). **Baseline confirmed — did not drift.**

## STEP 1 [BLOCKING CONFIRM] — per-thread OS-CPU DUTY (UNPINNED, N=9) → CONSUMER IS **COLD**
`perthread_cpu_duty.py` (/proc/<tid>/stat utime+stime ÷ wall; consumer tid == pid since
`chunk_fetcher::drive` runs `consumer_loop` inline on main; workers are ThreadPool children):
| thread | duty (median core) |
|---|---|
| worker rank 0 | 0.897 |
| worker rank 1 | 0.877 |
| worker rank 2 | 0.854 |
| worker rank 3 | 0.832 |
| **CONSUMER (tid==pid)** | **0.043** (per-run 0.043–0.083, very stable) |

- **CONSERVATION PASS:** Σ per-tid core-sec = 1.650 s vs **clean** (un-polled) perf task-clock
  1.6945 s → **2.6%** (gate ≤5%). The poller depresses each worker ~3% (effcores 3.525 polled vs
  3.616 clean), explaining workers reading ~0.83–0.90 rather than ~0.92; conservation confirms
  they are the real near-saturated decode threads.
- **FORK = COLD (consumer duty 0.043 ≤ 0.2).** **STEP-1's span trace (~98% blocked consumer) was
  RIGHT; P1's /proc `max_running=5` was a SAMPLING OVER-READ** — a 1ms-pump consumer intermittently
  shows state R, and `max` caught one coincident sample; that is NOT a CPU duty cycle.

### FALSIFY — "gz runs a 5th HOT consumer thread"
- **premise:** gz's in-order consumer runs HOT (P1 topology probe).
- **disproof:** frozen per-tid OS-CPU duty = **0.043 core** (≤0.2), conservation-gated (2.6%).
- **scope:** frozen-Intel silesia-T4 @38dbbb81. **re-open:** AMD/Zen2 (SMT/scheduling differ).
- ⇒ The +18% unpinned loss is **NOT** a hot 5th thread; "fold the consumer cold / blocking-recv"
  is DEAD (and was always unfaithful — the wait is a literal rg 1ms-pump, no spin to remove).

## STEP 3 [Gate-2, byte-exact, zero decode-path] — INTERNAL DISTINCT-PHYSICAL AFFINITY (UNPINNED, N=11)
`GZIPPY_PHYS_PIN=1` (affinity only; sha gzbase==gzphys==rg==zcat). Workers pinned to distinct
physical cores escape the inherited 1-core mask via a direct `set_for_current` (the fix in
38dbbb81; the original guard skipped when the worker inherited the consumer's narrowed mask).
| arm | best ms | med ms | cyc/B | effcores | gz/rg best |
|---|---|---|---|---|---|
| **gzbase** (default pin → SMT-collide on cpu2/3) | 464.4 | 468.9 | **11.145** | 3.609 | **1.2066** |
| **gzphys** (workers {0,2,4,8} + consumer 10) | 406.2 | 411.4 | **9.606** | 3.549 | **1.0555** |
| rg | 384.8 | 398.8 | 9.427 | 3.589 | — |

- A/A all PASS (gzbase 1.0057, gzphys 1.0038, rg 0.9870 ≤1.02).
- **gzphys/gzbase = 0.8748 best (cyc/B 0.8619)** — affinity recovers ~12.5% of wall, ~14% of cyc/B,
  Δ ≫ spread.
- **cyc/B 11.145 → 9.606** (≈ rg 9.427; residual **+1.9%** = the known PIN5 ~2.8% kernel front).
- **gz/rg 1.2066 → 1.0555 best / 1.0315 med.** cyc/B ratio +1.9%.

⇒ **SMT-CO-LOCATION CONFIRMED as the dominant mechanism of the +18% unpinned silT4 loss.** It is
the WORKERS (not the cold consumer) being packed onto SMT siblings by the default `get_core_ids()`
cycling pin. The mechanism metric (cyc/B inflation) is essentially fully closed; the ~3–5% wall
residual is the pre-existing kernel cyc/B front (same as PIN5).

## THE VERDICT
- **What the +18–19% unpinned silesia-T4 loss IS:** SMT co-location of the 4 decode WORKERS caused
  by `pinning_for_capacity` cycling logical ids `{0,2,3,4}` on this cpuset (cpu2/cpu3 = one
  physical core). NOT a hot 5th consumer thread (consumer duty 0.043c, COLD).
- **Shippable byte-exact fix candidate:** pin the decode workers to DISTINCT PHYSICAL cores (one
  logical id per SMT sibling group) instead of raw `get_core_ids()` order. `GZIPPY_PHYS_PIN=1`
  closes UNPIN gz/rg from 1.2066 → 1.0555 best (cyc/B 11.145 → 9.606), byte-exact. Making this the
  DEFAULT pinning is a strategic (R3) + cross-arch decision (the cpuset layout and SMT behavior are
  box-specific; AMD/Zen2 owed) — landed env-gated for now.
- **Residual after the fix:** ~+1.9% cyc/B / ~+3–5% wall vs rg = the pre-existing kernel
  instruction-surplus front (PIN5 territory), a separate front.

## OWED (NOT-YET-LAW)
- **AMD/Zen2 replication** of STEP-1 (consumer duty) and STEP-3 (affinity) — SMT topology and the
  cpuset ordering differ; the default-pin SMT collision may or may not reproduce.
- Cross-corpus (monorepo/nasa) replication of the affinity win.
- Promote distinct-physical pinning to default only after AMD confirms + a portable
  non-SMT-collide selection is validated across cpuset shapes (R3 decision).

## Reproduce
```
ssh root@REDACTED_IP /root/bench-lock.sh acquire 3600   # require BENCH_LOCK=quiet
ssh -J REDACTED_IP root@REDACTED_IP 'cd /root/gzippy && \
  GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native CORP=/root/silesia.gz \
  N=13 bash scripts/bench/kernel-ab/intel_gz_rg_pinregime_guest.sh'                 # STEP 0
ssh -J REDACTED_IP root@REDACTED_IP 'cd /root/gzippy && \
  GZ=/dev/shm/gzrg-target/release/gzippy CORP=/root/silesia.gz N=9 \
  bash scripts/bench/kernel-ab/perthread_cpu_duty_guest.sh'                          # STEP 1
ssh -J REDACTED_IP root@REDACTED_IP 'cd /root/gzippy && \
  GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native CORP=/root/silesia.gz \
  N=11 bash scripts/bench/kernel-ab/step3_affinity_guest.sh'                         # STEP 3
ssh root@REDACTED_IP /root/bench-lock.sh release && ssh root@REDACTED_IP /root/bench-lock.sh verify
```
