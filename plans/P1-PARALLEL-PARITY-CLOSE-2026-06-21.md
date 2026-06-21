# P1 — CLOSE the "gz at-parity with rapidgzip silesia-T4" question (PIN-REGIME, FROZEN)

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz decode sha[:16]=`b5eec9eed5feeb45`**
(built @8081516b == HEAD decode path: `git diff 8081516b..HEAD -- src build.rs Cargo.toml`
is EMPTY — only docs/scripts changed; binary on guest at `/dev/shm/gzrg-target/release/gzippy`,
RUSTFLAGS=`-C target-cpu=native`, `--no-default-features --features pure-rust-inflate`,
flavor=parallel-sm+pure, path=ParallelSM). **rg:** `/root/oracle_c/rapidgzip-native` v0.16.0
sha[:16]=`41baa20fdfbdea24`. **Box:** Intel i7-13700T LXC (neurotic guest 10.30.0.199),
**FROZEN** via `10.0.0.100:/root/bench-lock.sh acquire 3600` (`BENCH_LOCK=quiet
runnable_avg=1.00`, no_turbo=1, governor=performance, procs_running=1 during the cells).
**RELEASED + RESTORE VERIFIED** (`no_turbo=0 uncore=0x guests-thawed`, frozen=[], wd=inactive).
**Stamp: NOT-YET-LAW — single-arch Intel; AMD/Zen2 owed.** **Scope:** MEASUREMENT ONLY (no
decode-path src change; new files = rig + analyzer + probe + this note).
Corpus `/root/silesia.gz` (68,229,982 B → 211,968,000 B out).

## THE QUESTION (closed)
Every prior gz-vs-rapidgzip silesia-T4 wall ratio (+14-17%: STEP-0 frozen +14.2%, AXIS +16.9%,
WRITEV +14.4%) was measured **PINNED to a 4-core cpuset** (`taskset 2,4,6,8`) — the regime that
oversubscribes gz's 5-thread topology. The PRODUCTION comparison (`gzippy -p4`, OS schedules
freely) is **UNPINNED**, and was **never run**. So: is gz at-parity UNPINNED, or is the loss real?

## RESULT — silesia T4, gz(-p4) vs rapidgzip(-P4), 3 pin regimes, N=13 interleaved, /dev/null both, FROZEN
| regime | gz best ms | rg best ms | **gz/rg best** | gz/rg med | gz cyc/B | rg cyc/B | **cyc/B ratio** | gz util | rg util | **util-deficit(rg/gz)** |
|---|---|---|---|---|---|---|---|---|---|---|
| **UNPIN** (16 logical / 8 P-cores) | 466.4 | 390.7 | **1.1938 (+19.4%)** | 1.1627 (+16.3%) | 11.194 | 9.360 | **1.196 (+19.6%)** | 3.625 | 3.568 | 0.984 (**−1.6%**) |
| **PIN5** (2,4,6,8,10) | 396.0 | 387.2 | **1.0225 (+2.3%)** | 1.0172 (+1.7%) | 9.575 | 9.314 | 1.028 (+2.8%) | 3.641 | 3.549 | 0.975 (−2.5%) |
| **PIN4** (2,4,6,8) | 578.0 | 502.4 | **1.1503 (+15.0%)** | 1.1460 (+14.6%) | 9.553 | 9.322 | 1.025 (+2.5%) | 2.465 | 2.747 | 1.115 (**+11.5%**) |

**CONSERVATION (wall_ratio ≈ cyc/B_ratio × util-deficit) closes per regime:** UNPIN
1.1938≈1.1959×0.9843=1.1772; PIN5 1.0225≈1.0279×0.9746=1.0018; PIN4 1.1503≈1.0248×1.1147=1.1423.

## GATE-0 (all PASS, per regime)
- flavor=parallel-sm+pure, path=ParallelSM; same /dev/null sink both arms.
- byte-exact ref(zcat)=`028bd002c89c9a90` == gz == rg (211,968,000 B), all regimes.
- A/A wall ratio (license ≤1.02): UNPIN gz 1.0035/rg 1.0074; PIN5 gz 1.0000/rg 1.0065;
  PIN4 gz 0.9993/rg 0.9836 — all PASS (licenses the ratios).
- GHz spread ≤0.14% every cell (gate ≤1%) PASS; gz-vs-rg GHz ≤0.08% apart (frequency-fair).
- procs_running=1 during the freeze; runnable_avg=1.00 at acquire.

## THE rapidgzip THREAD-TOPOLOGY FINDING (the discriminator, gated)
`thread_topology_probe.py` (samples `/proc/<pid>/task` state every 3 ms):
- **gz -p4: max_total_threads=5, max_running(stateR)=5** — 4 decode workers + 1 in-order
  consumer, ALL FIVE run HOT (5 simultaneously runnable).
- **rg -P4: max_total_threads=5, max_running(stateR)=4** — rg ALSO spawns 5 threads, but its
  5th (reader/consumer) is mostly BLOCKED; only 4 are simultaneously runnable.
⇒ rg does **NOT fold** to 4 threads — it spawns 5 like gz — but rg's consumer stays COLD while
gz's runs HOT. **So gz demands 5 hot cores; rg demands 4.** This is the mechanism behind both
the PIN4 util-deficit AND the UNPIN cyc/B inflation (below).

## THE VERDICT (decided)
**UNPINNED gz/rg = 1.1938 best / 1.1627 med ≥ ~1.10 (Δ ≫ A/A spread) ⇒ the loss is REAL beyond
pinning. The premature "parallel at-parity with rapidgzip" claim is REFUTED on the production
(unpinned) scenario, gated frozen-Intel.** BUT the mechanism is NOT what the PIN4 history showed:

1. **PIN4 (the prior regime) reproduces +15.0%** and conserves as **util-deficit +11.5%**
   (gz 2.465 vs rg 2.747 cores; cyc/B at ~parity +2.5%). This IS the 4-core oversubscription
   artifact: gz's **5 hot threads** packed onto 4 cores deschedule (~1.05 cores off-CPU, per
   STEP-1) while rg's 4 hot threads fit. **The +14-17% pinned history was the pin artifact —
   CONFIRMED.**
2. **PIN5 (gz's natural n_active = 5 cores) is ≈PARITY: +2.3% best / +1.7% med**, util slightly
   favors gz (3.641 vs 3.549). **The deficit CLOSES at ≥5 cores ⇒ the PIN4 util-deficit is
   ADDRESSABLE** (it is the 5th-hot-thread topology, not intrinsic decode-parallelism loss).
   The PIN5 residual is just the **~2.8% kernel cyc/B** (the T1 instruction-surplus front).
3. **UNPINNED (production) loses +19.4% via a DIFFERENT mechanism — cyc/B, NOT util.** Util is
   at parity (gz 3.625 vs rg 3.568; gz keeps MORE cores busy), but gz's **cyc/B INFLATES from
   9.575 (PIN5) → 11.194 (UNPIN), +17%**, while rg's barely moves (9.314 → 9.360). With gz's
   **5 hot threads** scheduled freely across **8 physical P-cores (16 logical, SMT)**, the
   scheduler frequently **co-locates two gz threads on SMT siblings of one physical core** →
   IPC drop → cyc/B inflation. rg's **4 hot threads** more easily land on 4 distinct physical
   cores → no SMT contention → stable cyc/B. **The unpinned loss is SMT co-location driven by
   the SAME 5th-hot-thread topology** that drove the PIN4 util-deficit.

**SINGLE ROOT, TWO FACES:** gz runs a 5th HOT consumer thread; rg keeps its consumer COLD.
- pinned-4 → that 5th hot thread oversubscribes 4 cores (util-deficit +11.5%).
- unpinned → that 5th hot thread SMT-co-locates on 8 P-cores (cyc/B +17%).
- pinned-5 → that 5th hot thread gets its own core → ≈parity (residual = ~2.8% kernel).

## RECOMMENDED NEXT (a Gate-2 perturbation, NOT a concluded cause)
The located lever is the **CONSUMER-THREAD TOPOLOGY** (gz's in-order consumer runs hot; rg's is
cold/blocking). HYPOTHESIS (unvalidated): folding gz's consumer to a cold/blocking drain (like
rg's) — or onto a worker — removes the 5th hot thread, closing BOTH the unpinned cyc/B inflation
AND the pin4 util-deficit. **Gate-2:** perturb the consumer thread (e.g. make it block-drain, or
pin gz internally to n_workers distinct physical cores) and re-measure UNPIN + PIN4 — does the
unpinned cyc/B fall toward 9.6 and the pin4 util-deficit collapse? (NOT the STEP-1 `pool.pick.wait`
0.187c / 1.23× tail — those are sub-dominant; and NOT consumer-output writev, already FALSIFIED.)

## DOES THE DEFICIT CLOSE AT ≥5 CORES?
**YES** — PIN5 best 1.0225 (+2.3%) vs PIN4 best 1.1503 (+15.0%). Giving gz its 5th core
recovers ~12.7 pts of the 15-pt pin4 loss. The deficit is **ADDRESSABLE** (topology/scheduling),
not an irreducible decode-parallelism loss. The irreducible residual is the ~2.8% kernel cyc/B.

## OWED (NOT-YET-LAW)
- **AMD/Zen2 replication** of all three regimes + the topology finding (SMT behavior differs;
  Zen2 PEXT/PDEP microcode may shift cyc/B).
- The unpinned cyc/B-inflation = SMT-co-location is a HYPOTHESIS consistent with the conserved
  numbers; the Gate-2 consumer-topology perturbation is what would confirm it causally.

## Reproduce
```
ssh root@10.0.0.100 /root/bench-lock.sh acquire 3600   # require BENCH_LOCK=quiet
ssh -J 10.0.0.100 root@10.30.0.199 'cd /root/gzippy && \
  GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
  CORP=/root/silesia.gz PIN5=2,4,6,8,10 PIN4=2,4,6,8 N=13 \
  bash scripts/bench/kernel-ab/intel_gz_rg_pinregime_guest.sh'
ssh root@10.0.0.100 /root/bench-lock.sh release && ssh root@10.0.0.100 /root/bench-lock.sh verify
```
