# STEP-1 AXIS — is the silesia-T4 +16% gz-vs-RAPIDGZIP gap INSTRUCTION-COUNT, IPC/CODEGEN, or something else?

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz sha:** 5526585e
(decode path == W3 binary; `git diff` since W3 is docs/scripts only)
**Box:** Intel i7-13700T LXC (neurotic), LOADED (load 4.0–5.1), single pinned
P-core (T1, cpu3) and a 4 distinct-physical-P-core cpuset (T4, cpus {2,4,6,8}).
**Stamp:** NOT-YET-LAW — single-arch Intel x86; AMD/Zen2 owed.
**Method:** MEASUREMENT ONLY (no decode-path src change; new files = harness +
analyzer + this note). rapidgzip = the RIGHT reference (chunked decoder WITH a
scaffold + ISA-L kernel, like gz) — NOT the bare ISA-L `_04` STEP-0 used.

## The instrument (committed)
- `scripts/bench/kernel-ab/intel_gz_rg_cycbyte_guest.sh` + `…_analyze.py`.
- Combines the single-P-core taskset isolation + measured GHz-stability gate
  (`_distpreload_cycbyte_guest.sh`) with the gz-vs-rapidgzip both-arms + A/A
  self-test (`standing/_cleankernel_silt4_guest.sh`).
- Runs `perf stat` (cpu_core PMU) on the REAL decode of each whole binary:
  `instructions` (LOAD-IMMUNE), `cycles`, `LLC-load-misses`, `task-clock`,
  `duration_time` (wall). T1 = 1 pinned P-core; T4 = 4-P-core cpuset, perf
  aggregates instructions/cycles across ALL threads ⇒ instr/B(T4) =
  Σ_all_threads_instructions / output_bytes (the load-immune 4-thread compare).
- N=15 interleaved (gzA,gzB,rgA,rgB per rep). instr/B reported as mean+spread
  (load-immune); cyc/B best-of-N (min-cycles run); IPC=instr/cyc; GHz=cyc/task-clock.
- gz built ON guest @5526585e, `RUSTFLAGS=-C target-cpu=native`,
  `--no-default-features --features pure-rust-inflate`, bin sha256[:16]=`b5eec9eed5feeb45`.
  rapidgzip = `/root/oracle_c/rapidgzip-native` v0.16.0, sha[:16]=`41baa20fdfbdea24`.

## GATE-0 (all PASS / reported; silesia)
- byte-exact: ref(zcat)=`028bd002c89c9a90` == gz == rg ✓ (211 968 000 B out).
- /dev/null both arms; gz flavor=parallel-sm+pure, path=ParallelSM ✓.
- A/A wall: T1 gz 0.9973 rg 1.0042; T4 gz 1.0050 rg 0.9935 — both ≤1.02 PASS
  (licenses the loaded-box wall ratio).
- GHz: T1 1.3955/1.3954, T4 1.3931/1.3942; spreads ≤0.11% (gate ≤1%) PASS;
  gz-vs-rg GHz 0.01–0.08% apart (frequency-fair).
- LOAD-IMMUNE PRIMITIVE stability: instr inter-run spread T1 gz 0.140% rg 0.123%
  PASS(<0.5%); T4 gz 0.403% PASS, **rg 1.067% WARN** (rapidgzip's per-thread
  work-split is nondeterministic at T4 → its aggregate instr count wobbles ~1%;
  gz is deterministic. Consequence: the T4 instr/B gap is soft to ~±1%, but it is
  unambiguously SMALL, nowhere near +16%).
- LLC-load-miss/B tiny for both (T1 gz 0.0009 rg 0.0033; T4 gz 0.0023 rg 0.0038)
  — not memory-bandwidth bound. Box load 4.0–5.1 reported.

## RESULTS — silesia (the loss cell)

### T1 (single pinned P-core)
| metric | gz(native) | rapidgzip | gz/rg |
|---|---|---|---|
| **instr/B (mean)** | **13.650** | **11.942** | **1.143 (+14.3%)** [LOAD-IMMUNE] |
| cyc/B (best-of-N) | 5.454 | 5.112 | 1.067 (+6.7%) |
| IPC (best) | 2.504 | 2.337 | 1.072 (+7.2%) |
| GHz (mean) | 1.3955 | 1.3954 | 1.000 |
| wall ms (best) | 831.3 | 780.9 | 1.065 (+6.5%) A/A-licensed |
| CPUs-utilized | 0.995 | 0.994 | — |

WALL decomposition: per-byte CPU-work **104%** | parallel-util **−2%**.
cyc/B decomposition: instructions **206%** | IPC(1/ipc) **−107%**.
⇒ **T1 = INSTRUCTION-COUNT AXIS.** gz executes **+14.3% more instr/B** at **+7.2%
HIGHER IPC**; the higher IPC partly offsets but gz still loses +6.5% wall. (If gz
matched rg's instr/B at its current IPC, gz would BEAT rg.) IPC is NOT the problem.

### T4 (4-P-core cpuset {2,4,6,8})
| metric | gz(native) | rapidgzip | gz/rg |
|---|---|---|---|
| instr/B (mean) | 22.789 | 22.189 | **1.027 (+2.7%)** [LOAD-IMMUNE, rg WARN] |
| cyc/B (best-of-N) | 9.644 | 9.274 | 1.040 (+4.0%) |
| IPC (best) | 2.359 | 2.381 | **0.991 (−0.9% ≈ PARITY)** |
| GHz (mean) | 1.3931 | 1.3942 | 0.999 |
| task-clock ms | 1477.5 | 1427.8 | 1.035 |
| **CPUs-utilized** | **2.468** | **2.736** | **0.902** |
| **wall ms (best)** | **586.4** | **501.8** | **1.169 (+16.9%)** A/A-licensed |

**WALL decomposition (CONSERVED): wall_ratio 1.169 ≈ cyc/B_ratio 1.040 ×
util-deficit(rg/gz) 1.108 = 1.152.** WALL gap share: per-byte CPU-work **24%** |
**parallel-utilization 64%.**

⇒ At T4 gz ≈ rg on BOTH instr/B (+2.7%, within rg's ~1% noise) AND IPC (parity).
The +16.9% wall gap is **DOMINATED (~64%) by PARALLEL UTILIZATION**: gz keeps only
**2.47 of 4 cores busy vs rapidgzip's 2.74**. Neither tool reaches 4× on this
loaded box, but gz's chunked-decode scaffold leaves MORE cores idle than rg's.

## THE VERDICT (the branch the advisor asked for — answered, plus the third axis)
- **T1 axis = INSTRUCTION-COUNT** (gz +14.3% instr/B at +7.2% IPC). Clean, gated.
- **T4 axis (the actual silesia loss cell) = NEITHER instruction-count NOR
  IPC/codegen — it is PARALLEL UTILIZATION (scaffold/scheduling).** At T4 gz is at
  ~parity on instr/B AND on IPC; the +16.9% wall is mostly gz keeping fewer cores
  busy (2.47 vs 2.74). Per-byte CPU-work explains only ~24% of the T4 wall gap.
- **T1 and T4 DISAGREE** — and that disagreement is the finding: the gap's
  character CHANGES with thread count. The single-thread instruction surplus
  (+14.3%) largely COLLAPSES at T4 (+2.7%); the T4 loss is a different mechanism.

### This CONTRADICTS STEP-0's reconciliation (now measured directly)
STEP-0 claimed the ISOLATED single-thread emission instruction gap (+3.22 instr/B
vs the bare `_04`) "reconciles with / accounts for" the production silesia-T4 +16%
("same order of magnitude"). Measured DIRECTLY at T4 with the right reference
(rapidgzip, load-immune aggregate instr/B), the instruction gap is only **+2.7%**,
not +16% — and the +16.9% is ~64% parallel-utilization. **STEP-0's "same order of
magnitude" reconciliation was COINCIDENTAL** (it compared a 1-thread isolated slice
vs a 4-thread wall; the instruction gap shrinks at T4 and the wall gap is mostly a
utilization/scaffold phenomenon). This MEASURES the "DEEPER RECONCILIATION TENSION"
the bias-forensics note flagged: gz-scaffold-vs-rg-scaffold, not the emission kernel.

### RECOMMENDATION to the supervisor/advisor
NEITHER pre-decided branch is the right next move for the silesia-T4 +16% loss cell:
- STEP-2 "per-phase emission instruction localize" targets the T1 surplus, which is
  largely GONE at T4 — it would, by these numbers, recover at most ~24% of the T4
  wall gap (≈4 of the 16.9 pts).
- "BANK IPC/codegen" is moot — IPC is at PARITY at T4 (and gz LEADS at T1).
- **The lever THIS measurement points at is the PARALLEL-UTILIZATION / scheduling
  deficit in the chunked-decode SCAFFOLD** (gz 2.47 vs rg 2.74 effective cores on a
  4-core cpuset). The right STEP-2 is to localize WHY gz leaves more cores idle
  (chunk granularity / prefetch depth / consumer pacing / sync points in
  `single_member` driver + `block_fetcher`), with the conservation Gate-0 above.
  This is a HYPOTHESIS for the next perturbation — not yet a located cause.

## OWED (NOT-YET-LAW)
- **AMD/Zen2 replication** (both the +16% T4 wall and the utilization-vs-instruction
  decomposition; PEXT/PDEP microcoded on Zen2 may shift gz's instr/B).
- **Quiet/frozen Intel re-run** — the absolute CPUs-utilized (2.47/2.74 of 4) is
  depressed by box load; the loaded box may AMPLIFY a utilization deficit (a blocked
  thread waits longer to be rescheduled). The util-deficit SIGN + rough dominance are
  A/A-licensed, but its MAGNITUDE needs a quiet box.
- rg's T4 instr-count noise (1.067%) — tighten on a quiet box to firm the +2.7%.
- **Controls (N=15 nasa, N=2 monorepo) — confirm the instrument is not gz-biased
  and silesia is the loss outlier:** nasa T4 gz wall **−2.3%** (gz at parity/win),
  util-deficit **+2.5%** (≈0), gz instr/B actually LOWER than rg; monorepo T4 gz
  wall −1.4%, gz instr/B lower. So the **+10.8% util-deficit is SILESIA-T4-SPECIFIC**
  (the loss cell), not a universal artifact — nasa shows ~parity utilization.

## Reproduce
```
# build native gz on the guest
cd /root/gzippy && CARGO_TARGET_DIR=/dev/shm/gzrg-target RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --no-default-features --features pure-rust-inflate
# run the axis instrument (silesia)
GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
  CORP=/root/silesia.gz PIN_T1=3 PIN_T4=2,4,6,8 N=15 \
  bash scripts/bench/kernel-ab/intel_gz_rg_cycbyte_guest.sh
```
