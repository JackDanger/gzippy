# RING / FOLD-DRAIN perturbation — RESULTS (Gate-2)

Finding `F-8b5f9f0b0e33`. 2026-06-22, solvency EPYC 7282 Zen2, FROZEN
(gov=performance boost=0) + llama SIGSTOP'd safely (watchdog+trap; SIGCONT'd +
verified Running after; box restored gov=ondemand boost=1). gz = kernel-converge-A
39acc213 + ring-knob instrument (parallel-sm + pure-rust, FFI off), bin
/dev/shm/ring-target. Raw sweep: plans/amd-gap-data/ring-perturb-sweep/ +
ring_perturb_capture.sh. Pre-registered falsifier: plans/RING-PERTURB-FALSIFIER.md.

## QUESTION
Is the per-worker RING/FOLD-DRAIN copy (ring_other region) a WALL lever for the
AMD/Zen2 gz-vs-rapidgzip gap — confirm BEFORE committing to the faithful-rg
fold-drain→append convergence (decision from finding F-9c5ca01d020d, which
flagged ring_other as the largest raw gz>rg gap but labeled it INTRINSIC, with
"does it move the wall" UNPROVEN).

## METHOD (cursor-agent reviewed)
- SLOW-KNOB `GZIPPY_RING_INJECT_NS` = per-chunk rdtsc-bounded busy spin in the
  chunk_fetcher worker wrapper, INSIDE the R_WORKER rdtsc window but OUTSIDE the
  Block-method R_TABLE/R_DECODE spans → injected cycles land in `ring_other`.
  `GZIPPY_SLOW_KIND=sleep` = frequency-neutral control (real nanosleep).
- REMOVAL-ORACLE `GZIPPY_FOLD_NODRAIN=1` (pre-existing) skips the literal
  ring→data drain memcpy.
- N=9 interleaved, /dev/null both arms, silesia-T4 + monorepo-T2.
- fulcrum `perturb` is the verdict oracle.

## RESULT — fulcrum perturb: LEVER (worker_residual) on BOTH cells
| cell | criticality (busy) | sleep-control | Δwall(30%) | 2×spread | A/A drift |
|------|--------------------|---------------|-----------|----------|-----------|
| silesia-T4   | 0.823 (CI≥0.470) | 0.80 reproduces | +32.3 ms | 13.9 ms | ~0.2 ms |
| monorepo-T2  | 1.533 (CI≥1.123) | 0.93 reproduces | +32.1 ms |  8.6 ms | tight |

Raw silesia-T4 dose-response (s): baseline ~0.263 → spin t10 ~0.278 → t30 ~0.299;
sleep t30 ~0.295 (matches spin). Monotonic, proportional, sleep-confirmed →
`ring_other` is ON the critical path (NOT off-critical-path slack). This REFUTES
the "ring is intrinsic-but-slack, hidden by pipeline overlap" possibility.

## REMOVAL-ORACLE — INVALID at T>1 (dropped from the verdict)
`GZIPPY_FOLD_NODRAIN=1` produced a 2.7× SLOWDOWN, not a speedup:
- silesia T1: baseline 0.384s vs nodrain 0.387s → EQUAL (no window consumption).
- silesia T4: baseline 0.248s vs nodrain 0.666s → 2.7× SLOWER.
Mechanism: skipping the drain leaves UNINITIALIZED window bytes; successor
speculative chunks consume the corrupt windows → re-decode cascade (the
documented all-marker failure mode). FOLD_NODRAIN is therefore an INVALID
copy-removal oracle at the T>1 loss cells — the speed-up CEILING is UNBOUNDED by
it. (Raw kept as oracle_removed.INVALID.txt.)

## INTERPRETATION — what this does and does NOT license (anti-bias)
CONFIRMED (gated, Gate-2): the `ring_other` worker-residual region causally gates
the wall on both loss cells; it is serial critical-path work, not slack.

NOT confirmed / caveats on the fold-drain→append convergence specifically:
1. The slow-knob tests the WHOLE residual (per-chunk SCAFFOLD = ChunkData setup +
   the per-block loop in `finish_decode_chunk_contig_native` + finalize + the
   ~1% ContigFoldSink copy), NOT the fold-drain copy in isolation. Verdict scope
   = `LEVER(worker_residual)`, NOT "the fold-drain copy is the lever".
2. Per source (chunk_decode.rs:1326-1333) the literal fold-drain COPY is the ~1%
   marker-dribble path; the BULK clean tail already does rg-style DIRECT APPEND
   via `decode_clean_into_contig` (no ring, no drain). So gz is ALREADY appending
   directly on the hot path — the convergence target may largely already exist.
3. "Slow-down slope ≠ speed-up ceiling" (CLAUDE Gate-2). The slow-knob proves
   on-critical-path; on a decode-bound worker (R_WORKER=94% cyc) any per-chunk
   worker region shows criticality~1, so this is necessary-not-sufficient for a
   convergence ROI. The speed-up ceiling needs a VALID removal oracle, which we
   do not have (FOLD_NODRAIN invalid at T>1).

## VERDICT for the convergence decision
ring_other is a CONFIRMED critical-path region (worth further work), but the
fold-drain→append convergence is **NOT YET JUSTIFIED**: the lever is the broad
scaffold not the copy, the copy is ~1% (bulk already direct-appends), and no
valid removal oracle bounds the recoverable wall. NEXT before building:
(a) sub-localize ring_other (setup vs per-block-loop vs finalize vs copy) with a
matched gz/rg partition + fulcrum excess; (b) build a WINDOW-SAFE copy-removal
oracle (one that writes correct-but-cheap bytes so successor windows stay valid)
to bound the ceiling.

## GATES
Gate-0 PASS (non-inert RING hits == chunk count 17/5; baseline sha==zcat both
cells; rg self-test sha==zcat; sleep control present; /dev/null both arms; A/A
drift << Δ). Gate-1 PASS (N=9 interleaved, Δwall > 2×spread both). Gate-2 = the
perturbation IS the verdict (fulcrum LEVER). Gate-4 PASS (path=ParallelSM, bin
from 39acc213 + ring knob). **Gate-3 OWED: Intel replication → NOT-YET-LAW
(Zen2 single-arch).** Minor: BIMODAL note on the sample sets (N=9 met; widen N if
re-run). Removal-oracle ceiling NOT available (invalid instrument).
