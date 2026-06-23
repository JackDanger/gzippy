# RING / FOLD-DRAIN perturbation — pre-registered FALSIFIER

Pre-registered BEFORE running (CLAUDE.md Gate-2). Decides whether the per-worker
RING/FOLD-DRAIN region (`ring_other` in region_prof) is a WALL lever for the
AMD/Zen2 gz-vs-rapidgzip gap — i.e. whether the faithful-rg `fold-drain→append`
convergence is worth building.

## Context / prior (do NOT cite as current verdict)
F-9c5ca01d020d (fulcrum excess, Zen2, gated): R_WORKER sub-decomp →
`DECODE_TOTAL` near-parity, `table_build` acquitted, and `ring_other`
(= R_WORKER − R_TABLE − R_DECODE) is the LARGEST raw gz>rg gap (gz 0.786 vs rg
0.118 cyc/B silesia) but labeled INTRINSIC (gz heavier on nasa control too).
"INTRINSIC" ≠ "off critical path" — whether it MOVES THE WALL is UNPROVEN. That
is what this perturbation tests.

## SOURCE-READ CAVEAT (must be weighed; flagged to cursor-agent)
`ring_other` is an ARITHMETIC RESIDUAL, not a tagged span. region_prof's
R_TABLE / R_DECODE spans wrap the `Block` methods (`read_header`,
`decode_clean_into_contig`, `read`, `ensure_dist_*`), which ARE called on BOTH
the bulk contig clean path and the marker path. So `ring_other` =
per-chunk SCAFFOLD (ChunkData setup + the per-block loop machinery in
`finish_decode_chunk_contig_native` + finalize), NOT a single memcpy. The
literal "fold-drain copy" (`ContigFoldSink::push_clean_u8`'s `extend_from_slice`)
is, per the in-code note (chunk_decode.rs:1326-1333), the ~1% marker-dribble
path; the BULK clean tail is u8-DIRECT via `decode_clean_into_contig` (NO ring,
NO drain — already rg-style append). FOLD_NODRAIN was previously measured at
~0-1ms (frozen N=21). So the "ring/fold-drain copy is the gap" premise is
PARTLY contradicted by source: gz already appends directly on the hot path.
This perturbation tests BOTH (a) the whole ring_other residual and (b) the
literal copy, to resolve it empirically rather than by source-read.

## Perturbation design
TWO independent perturbations of the ring_other region, /dev/null both arms,
N≥9 interleaved, frozen box (gov=performance boost=0), llama paused-safely:

1. SLOW-KNOB (critical-path test of the whole ring_other residual).
   `GZIPPY_RING_INJECT_NS=<ns>` injects `<ns>` ns of known work PER CHUNK in the
   ring_other region (chunk_fetcher worker wrapper, inside R_WORKER, OUTSIDE
   R_TABLE/R_DECODE → lands in ring_other). Levels t10/t20/t30 = 10/20/30% of
   the measured ring_other wall-ms. `GZIPPY_SLOW_KIND=sleep` = frequency-neutral
   control (nanosleep) vs busy rdtsc-bounded spin. Non-inert: RING_INJECT_HITS
   (== chunk count) + RING_INJECT_NS_TOTAL > 0.

2. REMOVAL-ORACLE (bounds the fold-drain→append convergence ceiling).
   `GZIPPY_FOLD_NODRAIN=1` (pre-existing) SKIPS the literal ring→data drain
   memcpy (decode still runs; output bytes WRONG → oracle only, no sha). Ceiling
   = baseline_min − oracle_min. Non-inert proof: the knob's accounting path runs
   (banner prints); report whether it is cold on silesia-T4 (it removes only the
   ~1% marker-dribble copy).

## Cells
silesia-T4 (primary LOSS cell) + monorepo-T2 (2nd loss cell).

## PRE-REGISTERED VERDICTS
- **LEVER** iff: slow-knob busy arm RESPONDS (monotonic + proportional,
  slope_lo>0, Δt30 > SIGMA_K·spread) AND the sleep control also RESPONDS
  (rules out turbo artifact) — fulcrum `perturb` → `Verdict::Lever`. THEN the
  recoverable wall is bounded by the removal-oracle ceiling; report what %
  of the ~3-6% gap the literal-copy removal recovers (likely small if the copy
  is cold → "the ring region is on the critical path but the recoverable part is
  the SCAFFOLD, not the copy").
- **NOT-A-LEVER (SLACK)** iff: both slow-knob arms FLAT (fulcrum
  `Verdict::Slack`) → ring_other cycles are off-critical-path slack at T4
  (overlapped by the parallel pipeline / consumer). The AMD gap is NOT
  recoverable via the ring; the `fold-drain→append` convergence is NOT worth
  building. Report honestly — this SAVES the convergence effort.
- **INCONCLUSIVE / VOID**: underpowered (Δt30 ≤ 2·spread), non-monotonic
  (instrument inconsistency), or control bracket swung > spread (box not quiet)
  → re-capture; NOT a verdict either way.

## Gates owed
Gate-0 (non-inert + sleep control + /dev/null both + baseline sha==zcat + rg
self-test ~1.0 + A/A << Δ); Gate-1 (N≥9, Δ vs spread); Gate-2 (perturbation IS
the verdict); Gate-4 (path=ParallelSM, HEAD sha). Gate-3 OWED (Zen2-only →
NOT-YET-LAW; Intel replication owed).
