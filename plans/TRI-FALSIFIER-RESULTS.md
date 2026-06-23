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

## RESULT 4 — PREMISE BREAKER: gz already BEATS rapidgzip AND pigz at T≥2 on Intel
Direct cross-tool wall, interleaved N=11, /dev/null all arms, sha==zcat, fair
thread counts (rg/gz both spawn 5 threads at 4 workers — verified via /proc/task):

| cell (corpus)      | gz ms | rg ms | pigz ms | gz/rg | gz/pigz | verdict |
|--------------------|-------|-------|---------|-------|---------|---------|
| T2 monorepo        | 264.4 | 331.4 | 335.2   | **0.798** | 0.789 | gz beats rg +20% |
| T4 silesia         |1004.4 |1095.5 |1812.9   | **0.917** | 0.554 | gz beats rg +8% |
| T8 silesia         | 485.9 | 559.5 |1759.7   | **0.868** | 0.276 | gz beats rg +13% |

**The T≥2-vs-rapidgzip goal is ALREADY MET on Intel** — gz wins every cell. The
A0 marker-ceiling sweep was INVALID (ceiling arms slower than baseline: U16/U16W
inject a phantom 201MB resolve pass = extra work; u8 cross-cell-inconsistent —
confirms cursor's "wrong-routing" warning), so it gives no marker-prize bound;
but it does not matter, because RESULT 4 shows there is no gz-vs-rg gap to close
on Intel. The marker-loop's ~50% gz-internal criticality (RESULT 2) is HEADROOM,
not a competitive deficit. FINDING-DECAY: the 2026-06-16 two-binary matrix's
"native loses at T≥2" and the AMD T2/T4 (~3-6% loss) BOTH predate the B2/B3/RANK-2
kernel wins (matrix @pre-wins; AMD findings @8cad4f6b/39acc213, before f5f827b6).
The kernel wins appear to have CLOSED the T≥2-vs-rg gap.

## RESULT 5 — CROSS-ARCH CONFIRMED: gz beats rapidgzip+pigz at T≥2 on AMD too (@HEAD)
AMD/Zen2 EPYC 7282 (solvency), gz-native @eea9e445 built on box, rapidgzip 0.16.0
(venv), interleaved N=11, /dev/null, cores 24-31 (load-robust ratio; box under
llama load so spreads high but the paired ratio + direction are decisive). silesia
corpus byte-identical to Intel (decode sha 028bd002…).

| cell | gz ms | rg ms | pigz ms | gz/rg | gz/pigz | verdict |
|------|-------|-------|---------|-------|---------|---------|
| T2 monorepo | 131.0 | 154.6 | 150.4 | **0.848** | 0.871 | gz beats rg +18% |
| T4 silesia  | 243.8 | 260.1 | 767.0 | **0.937** | 0.318 | gz beats rg +7% |
| T8 silesia  | 184.6 | 214.1 | 736.7 | **0.863** | 0.251 | gz beats rg +16% |

**gz BEATS rapidgzip AND pigz at EVERY T≥2 cell on BOTH Intel and AMD at HEAD.**
The AMD ~3-6% rg loss ([[project_amd_t2_phase_and_t4_floor_2026_06_22]],
@pre-kernel-wins) is GONE — B2/B3/RANK-2 closed it on both arches. The T≥2-vs-rg
GOAL IS MET CROSS-ARCH. (AMD load-noisy ⇒ a frozen+llama-paused reconfirm would
tighten the exact ratio for LAW, but the direction is robust across all 3 cells.)

## VERDICT / NEXT (REVISED after RESULT 4)
- **On Intel, T≥2-vs-rg+pigz is WON** — no decode port needed; the marker-port
  front is MOOT here. Do NOT build the A1 marker oracle for an Intel gz-vs-rg gap
  that does not exist.
- The only place gz reportedly loses to rg at T≥2 is AMD/Zen2 (~3-6%, AMD finding
  @pre-kernel-wins). DECISIVE NEXT MEASUREMENT: re-run gz-vs-rg at T≥2 on AMD AT
  HEAD (eea9e445, with the kernel wins). If gz now ties/beats rg on AMD too, the
  T≥2-vs-rg goal is MET cross-arch and this front CLOSES. (AMD has no rg binary —
  pip install rapidgzip; clone+build gz-native; guaranteed-resume llama-pause.)
- RSS/teardown is a ~0.9% wall sideshow (RESULT 3) — pursue only for the RSS
  scoreboard, not the wall.
- The marker port is only justified IF AMD still loses at HEAD AND an A1 removal
  oracle brackets a real gz-vs-rg prize there. Otherwise it solves a non-problem.
