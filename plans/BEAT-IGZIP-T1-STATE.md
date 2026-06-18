# BEAT-IGZIP-T1 — DURABLE STATE

Mission: make gzippy-native (pure-Rust, FFI-off) **T1 single-member gzip DECODE**
measurably FASTER than igzip (ISA-L), byte-exact, gated. Single-arch Intel = NOT LAW
(AMD/Zen2 replication owed). T1 single-core only — no T4/T8 extrapolation.

Branch: `perf/igzip-full-rewrite`. Mac worktree (edit/commit/push only — aarch64,
CANNOT run the asm): `/home/user/www/gzippy/.claude/worktrees/agent-a8069a92d914fcef3`.
Guest (ONLY x86_64/BMI2 measure box): `ssh -J REDACTED_IP root@REDACTED_IP`.
Guest worktrees: gzippy(B)=/root/gz-fullrewrite (kernel 2b10aa48 dist-preload),
baseline=/root/gz-baseline (8383a2eb). igzip=/usr/bin/igzip (ISA-L 2.31.1).
Harness on guest: /root/distpreload-harness/.

## COMMITS THIS MISSION
- 2c135d07 — deliverable #0: commit orphaned paired harness (analyzer+memstress+driver) to scripts/bench.
- 2e01dd4f — deliverable #1: add igzip arm `scripts/bench/_gzippy_vs_igzip_paired_guest.sh`.

## THE INSTRUMENT (deliverable #1) — Gate-0 SELF-VALIDATED, PASS
`scripts/bench/_gzippy_vs_igzip_paired_guest.sh` (reuses committed `_distpreload_paired_analyze.py`).
arm A(A1,A2)=igzip, arm B=gzippy-native (ParallelSM @ T1). medΔ=(B-A1) cyc/byte;
NEGATIVE => gzippy faster. Gate-0 verified live: gzippy run_contig KERN entries>0
(monorepo 8299, nasa 25399), igzip --version printed, BOTH arms sha==zcat ref==each
other, same /dev/null sink + same pin(cpu4), GHz spread <0.07%, A2-A1 self-test ~0.
Run: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`

## THE GAP TO CLOSE — gzippy-native T1 vs igzip cyc/byte (whole-process, /dev/null sink)
SMOKE (N=2, magnitude only — p not yet gated):
| corpus   | igzip cyc/B | gzippy cyc/B | Δ (B-A1)        | Δinstr/byte | note |
|----------|-------------|--------------|-----------------|-------------|------|
| monorepo | 2.96        | 5.15         | +2.19 (+73.9%)  | +2.95       | gzippy ~1.7x slower |
| nasa     | 1.57        | 3.38         | +1.81 (+115.7%) | +2.13       | gzippy ~2.2x slower |

REFRAME (HYPOTHESIS, unvalidated): the gap is LARGE and INSTRUCTION-DOMINATED
(gzippy retires ~2-3x more instr/byte than igzip), not a small IPC tie-margin.
This is whole-process T1 (FORCE_PARALLEL_SM single chunk) so it includes gzippy's
parallel-SM scaffold (block finder, u16 marker machinery, apply_window,
replace_markers, CRC) around run_contig — NOT just the inner kernel. Whether the
excess instr/byte lives in run_contig or in the scaffold is the FIRST thing to
localize before any loop-pipelining tweak. The prior mission framing ("close a
cyc/byte tie via loop_block pipelining") is NOT supported by this measurement — a
2x instruction gap will not close with a preload tweak.

## NEXT (planned, in priority order)
1. Run FULL gated gap (N=21, silesia+monorepo+nasa, both stress phases) — record real number. [IN PROGRESS]
2. LOCALIZE the instr/byte excess: is it run_contig or the SM scaffold? Build a
   self-time/instr attribution (perf record on the gzippy arm, or a run_contig-only
   instr counter via GZIPPY_ASM_STATS) so kernel changes are measurable in isolation.
   If the excess is mostly scaffold, the mission's run_contig focus is mis-aimed and
   the lever is reducing per-chunk scaffold instr at T1 (single-chunk fast path).
3. Only then iterate igzip loop_block techniques on run_contig, one per commit,
   each byte-exact gated (cargo test + proptest≥60k + tri-oracle) + paired cyc/byte A/B.

## DONE-CRITERION (do NOT self-bless; report as gated-HYPOTHESIS + re-verify cmd)
gzippy T1 cyc/byte ≤ igzip with paired p<0.01 + bootstrap CI excluding 0 + margin
surviving the bandwidth stressor, on silesia AND ≥1 more corpus; + wall-time
confirmation if a quiet window appears. AMD/Zen2 replication owed for LAW.
