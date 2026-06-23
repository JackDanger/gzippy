# T≥2 Tri-Falsifier — Intel results (NOT-YET-LAW; AMD owed)

**2026-06-23, neurotic Intel i7-13700T LXC (un-frozen; taskset core-pin +
paired-interleaved so turbo/drift cancels in the paired Δ). Binary: gzippy-native
@ `4495d3d4` (reimplement-isa-l), `target-cpu=native`, `/dev/shm/tri-target`,
build-flavor=parallel-sm+pure, path=ParallelSM. /dev/null sink both arms.**

Tools (this commit): `scripts/bench/_tri_falsifier_guest.sh` +
`scripts/bench/tri_falsifier_analyze.py` (orthogonality + ceilings + peak RSS),
`scripts/bench/_teardown_split_guest.sh` (pure-teardown isolation). Data:
`plans/tri-falsifier-data/{tri_intel.csv,teardown_intel.csv,tri_intel_analysis.txt}`.

## Mission
cursor-agent-directed prize-bounding + fixation-exposer for the three Zen2-gated
T≥2 findings (project_amd_t2_phase_and_t4_floor / project_ring_perturb): replicate
on a 2nd arch, bound each prize BEFORE any architectural port, and test the lead's
"one unifying decoupled-marker port fixes T2-RSS AND T4-marker" synthesis.

## Gate-0 (instrument self-validation — all PASS)
- sha==zcat both corpora (silesia, monorepo).
- MARKER perturbation non-inert: `GZIPPY_SLOW_MARKER_MODE` marker-loop inject
  hits = 14,227,247 (>0). (The Gate-0 banner's hits-regex initially misparsed
  `hits = N`; the instrument fires — proven directly.)
- RSS-inflate non-inert: peak RSS rose +19.8/+39.7/+59.6 MiB for N=20/40/60
  (exactly proportional). NOTE: the FIRST cut was INERT (+0.3 MiB) — plain
  `buf[i]=v` stores were dead-eliminated (buf leaked/never read); fixed with
  volatile write+read-back (`4495d3d4`). Caught by Gate-0, as designed.
- phase_timing conservation PASS (gap=0, monotonic).

## RESULT 1 — TWO INDEPENDENT LEVERS (the lead's one-port synthesis FALSIFIED)
Paired Δ% vs same-rep baseline, N=15 (noise floor = BASE inter-rep spread):

| cell (noise)     | MARKER@100 spin | MARKER@100 sleep-ctl | RSS@60 mmap |
|------------------|-----------------|----------------------|-------------|
| T4 silesia (2.1%)| **+55.3%** MOVES| +3.2%                | +4.5% MOVES |
| T2 monorepo (4.0%)| **+43.2%** MOVES| +8.7%               | **+16.5%** MOVES |

A *pure-RSS* perturbation (touches NO decode) moves the wall (T2 +16.5%@60MiB),
and a *decode-compute* perturbation (touches NO allocation) moves the wall
(+34–52% real-compute). ⇒ at least TWO independent mechanisms: (1) the marker
fast-loop decode and (2) RSS lifecycle. A single "decoupled-marker port" cannot be
ASSUMED to capture both. The synthesis was OVER-CLAIMED (cursor's self-bias check
vindicated). NOT orthogonal-by-cell (both T≥2 cells decode markers) — orthogonal
by MECHANISM.

## RESULT 2 — marker-decode is the DOMINANT lever (criticality, NOT a ceiling)
real-compute (spin − sleep) on the critical path: **+52.1% (T4)**, **+34.5% (T2)**.
GATE-2 CAVEAT: a slow-down slope is CRITICALITY, not a speed-up ceiling. To BOUND
the marker speed-up prize, build the byte-exact removal oracle (route window-absent
chunks through the fast clean decode + marker emission, skip
`decode_marker_fast_loop`) — do NOT extrapolate this slope.

## RESULT 3 — RSS/teardown is REAL but SMALL (de-conflated)
The tri-falsifier `run()` timed the WHOLE process, conflating first-touch +
teardown. The `_teardown_split` rig isolates pure teardown = external − internal
(phase main_start→main_end), N=12:

| ΔRSS | internal/touch | PURE teardown |
|------|----------------|---------------|
| +40 MiB | +25.6 ms (0.640 ms/MiB) | +3.27 ms (0.082 ms/MiB) |
| +60 MiB | +38.9 ms (0.648 ms/MiB) | +5.02 ms (0.084 ms/MiB) |

- **Pure-teardown slope ≈ 0.084 ms/MiB** — REPLICATES AMD's 0.054 ms/MB
  (same order, proportional, monotonic). Teardown∝RSS confirmed on a 2nd arch.
- Peak RSS T2: gz 97.1 / rg 68.6 = **1.42×** (AMD was 1.65×). ΔRSS = 28.5 MiB.
- **RECOVERABLE teardown at the gz−rg gap ≈ 2.4 ms = 0.88% of the T2 wall.**
  The earlier conflated "7.7%" was first-touch of ARTIFICIAL held memory — NOT a
  clean gz-RSS-reduction prize. This REFINES the AMD-T2 finding: the *recoverable*
  teardown is ~1% of wall, not the headline magnitude. (RSS remains its own
  scoreboard the user values separately — but on the WALL it is a ~1% lever.)

## VERDICT / NEXT
- Focus wall effort on the MARKER-DECODE lever (dominant); RSS/teardown is a ~1%
  wall sideshow (pursue for the RSS scoreboard, not the wall).
- NEXT deterministic step (cursor option 3): build the byte-exact marker-decode
  removal oracle to BOUND the marker speed-up prize on BOTH arches before any port.
- Gate-3 OWED: AMD replication of RESULTS 1–3 (rg-free legs runnable on solvency;
  needs clone+build + guaranteed-resume llama-pause). Single-arch ⇒ NOT-YET-LAW.
