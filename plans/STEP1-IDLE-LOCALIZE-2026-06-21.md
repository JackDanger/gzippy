# STEP-1 — repair the utilization instrument + LOCATE the silesia-T4 idle (FROZEN)

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz decode sha[:16]=`b5eec9eed5feeb45`**
(built @8081516b, RUSTFLAGS=`-C target-cpu=native`, `--no-default-features --features
pure-rust-inflate`, build-flavor=parallel-sm+pure, path=ParallelSM verified — BYTE-IDENTICAL
to STEP-0's frozen binary). **Box:** Intel i7-13700T LXC (guest 10.30.0.199) **FROZEN** via
`10.0.0.100:/root/bench-lock.sh acquire 1800` (`BENCH_LOCK=quiet runnable_avg=1.50`,
no_turbo=1, governor=performance). **Released + RESTORE VERIFIED** (`no_turbo=0 uncore=0x
guests-thawed`). **Stamp: NOT-YET-LAW — single-arch Intel; AMD/Zen2 owed.**
**Scope:** INSTRUMENT REPAIR + LOCALIZATION only — no decode-path src change, no CAUSE/fix
(that is STEP 2/3). Corpus `/root/silesia.gz` (68,229,982 B → 211,968,000 B out).

## WHAT WAS REPAIRED
`scripts/parallel_sm_tail_metric.py` previously emitted only `effcores = Σ(decode_chunk
SPAN)/wall` (~3.25/4). It is now a **CONSERVATION-GATED per-thread busy/idle timeline**:

- **Leaf/innermost-span attribution via a B/E stack per tid** → every microsecond of every
  thread's life is split into BUSY (on-CPU work spans) / WAIT (off-CPU marked blocks:
  `wait.future_recv`, `wait.block_fetcher_get`(+`ttp.rx_recv_block`), `pool.pick.wait`) /
  GAP (untraced). **No double-count** (nested `worker.block_header/body` etc. fold into
  their parent — the "62ms nested-span" trap is structurally impossible here).
- **CONSERVATION GATE (blocking):** per-tid `Σ(busy+wait+gap) == life`; worst-thread GAP
  ≤ tol (else refuse); decode_chunk non-overlap per worker tid; thread-time closes to
  `wall × n_active_threads` (here 5 = 4 workers + 1 consumer).
- **NON-INERT GATE:** decode_chunk span count **== 17 == ceil(68.23 MB / 4 MiB)** ✓
  (matches the rg-style partition; `--expected-chunks 17` passes).
- **RECONCILIATION GATE:** new `effcores_CPU = Σbusy/wall` compared to perf
  `CPUs-utilized` of the **SAME traced run** (`scripts/bench/kernel-ab/step1_idle_trace_guest.sh`
  perf-stats the identical `GZIPPY_TIMELINE` command).

## THE RECONCILIATION — RESOLVED (the 3.3-vs-2.458 paradox)
The paradox is **NOT** a double-count and **NOT** in-span marked wait
(`in-decode MARKED-wait = 0.000 cores`). It is **OFF-CPU deschedule from
OVERSUBSCRIPTION**: STEP-0 pinned gz's **5 active threads (4 decode workers + 1 in-order
consumer) onto 4 P-cores** (`taskset -c 2,4,6,8`). Proven by running the SAME binary/trace
both PINNED and UNPINNED on the frozen box (N=3 each, same-run perf):

| mode (frozen) | perf CPUs-util (clean, N=3) | perf CPUs-util (traced) | trace effcores_CPU | wall |
|---|---|---|---|---|
| **PINNED** 2,4,6,8 | 2.413 / 2.498 / 2.414 (≈**2.44 ≈ STEP-0 2.458**) | 2.404 / 2.458 / 2.516 | **3.512** | ~600 ms |
| **UNPINNED** (16c) | 3.631 / 3.639 / 3.618 (≈**3.63**) | 3.642 / 3.586 / 3.640 | **3.791** | ~468 ms |

- **UNPINNED → RIG HONEST:** trace `effcores_CPU 3.791` reconciles to perf `3.642` within
  **+4.1%** (gate PASS). When there is no oversubscription, the wall-span trace's busy
  occupancy reproduces perf task-clock. The instrument is trustworthy.
- **PINNED → diagnostic:** trace busy `3.512` vs perf `2.458` = **+42.9% = 1.054 cores of
  off-CPU deschedule INSIDE busy spans** (5 threads / 4 pinned cores). A wall-span trace
  cannot localize this to a span (it is deschedule within a busy span, GAP still ~0), so the
  reducer reports it explicitly as OVERSUBSCRIPTION rather than as a rig failure.
- **The STEP-0 "util deficit 2.458/4" is therefore largely a 4-core-pin artifact**, not gz
  leaving cores idle: unpinned, gz drives **3.63 real cores and finishes ~130 ms faster.**

## THE LOCATED IDLE (genuine, oversubscription-free = UNPINNED, FROZEN)
- **Workers are ~saturated:** per-tid busy 92–98% (tids 2–5: 429.9/418.5/435.8/407.9 ms of
  ~444 ms) → 3.79 effcores of 4.
- **The in-order CONSUMER (5th thread) is ~98% blocked** (busy 8.3 ms / 448 ms), almost
  entirely on `wait.block_fetcher_get` (**0.855 cores**) + `wait.future_recv` (**0.118
  cores**). Expected for an in-order drain thread — it costs nothing UNPINNED, but under the
  4-core pin it steals a P-core from a worker (the deficit mechanism).
- **Genuine worker idle is small:** `pool.pick.wait` (cross-chunk dispatch starvation) =
  **0.187 cores (~4.7% of T=4)**.
- **Modest tail imbalance:** last-wave/global = **1.23**, per-chunk **cv 0.35** (max 159 ms
  vs mean 95.6 ms, max/min 5.56), **3** completion-order inversions. Under the pin this
  worsens to last-wave 1.29, cv 0.56, **14** inversions (oversubscription jitter).

### CLASSIFICATION (the brief's fork)
- **in-decode_chunk stall** (memory/contention inside the kernel): **0.000 cores marked**;
  unpinned reconciles cleanly ⇒ no hidden in-decode off-CPU. **NOT the idle.**
- **wait.future_recv cross-chunk drain straggler:** only **0.118 cores** ⇒ **NOT
  tail-future_recv-concentrated.**
- **DOMINANT located idle = OVERSUBSCRIPTION** of gz's 4-workers-+-1-consumer topology onto
  STEP-0's 4 pinned cores (**~1.05 cores**), plus a small **worker cross-chunk dispatch gap
  `pool.pick.wait` 0.187 cores** and a **1.23× heterogeneous tail**. The ~0.25-core gz-vs-rg
  deficit (gz 2.458 vs rg 2.710, both pinned) lives in how gz's 5-thread topology packs onto
  4 cores, NOT in decode-kernel stall and NOT in the drain tail.

## GATE LEDGER
- GATE-0 self-validation: conservation closes (gap ≤0.9%, unaccounted ≤0.04 core); decode
  non-overlap ✓; non-inert chunk-count 17==ceil(comp/4MiB) ✓; rig honesty proven by the
  UNPINNED trace↔perf +4.1% reconciliation ✓.
- GATE-1: N=3 per mode, tight (pinned 2.41–2.52, unpinned 3.62–3.64); tracing perturbs perf
  CPUs-utilized <2% (clean vs traced) — the trace is load/overhead-light.
- GATE-4: path=ParallelSM, build-flavor=parallel-sm+pure, decode sha b5eec9ee == STEP-0,
  out=211,968,000 B.
- NOT-YET-LAW: single-arch Intel; **AMD/Zen2 owed**.

## RECOMMENDED STEP 2 (rg-vs-gz util on the frozen box + addressable-vs-irreducible discriminator)
1. **Repeat the pinned/unpinned matrix for rapidgzip** on the frozen box (same harness).
   Hypothesis to test (not concluded): rg's pinned 2.710 also rises unpinned; the real
   comparison is the UNPINNED wall, where gz already hits 3.63 cores. Count rg's active
   thread topology (does it also run a separate in-order consumer? `nproc`-vs-`-P` thread
   audit) to test the oversubscription mechanism.
2. **Addressable-vs-irreducible discriminator:** pin gz to **5 cores (= n_active)** and to
   **all 16**, re-measure. If the gz deficit closes at ≥5 pinned cores, it is **addressable**
   (pin/topology, e.g. fold the consumer onto a worker / don't over-pin) not an intrinsic
   decode-parallelism loss. Then a Gate-2 perturbation (STEP 3) confirms the lever.

## REPRODUCE
```
ssh root@10.0.0.100 /root/bench-lock.sh acquire 1800   # require BENCH_LOCK=quiet
# build native gz on guest @8081516b (RUSTFLAGS=-C target-cpu=native,
#   --no-default-features --features pure-rust-inflate, CARGO_TARGET_DIR=/dev/shm/gzrg-target)
ssh -J 10.0.0.100 root@10.30.0.199 'GZ=/dev/shm/gzrg-target/release/gzippy \
  CORP=/root/silesia.gz OUT=/dev/shm/step1 PINSET=2,4,6,8 N=3 \
  bash scripts/bench/kernel-ab/step1_idle_trace_guest.sh'
# then per trace:
python3 scripts/parallel_sm_tail_metric.py TRACE.json --expected-chunks 17 --perf-cpus <same-run CPUs-utilized>
ssh root@10.0.0.100 /root/bench-lock.sh release && ssh root@10.0.0.100 /root/bench-lock.sh verify
```
