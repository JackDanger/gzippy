# ZEN2 STEP-1 CEILING ORACLE — CORRECTED U16-PRESERVING RESULTS (2026-06-22)

Branch `kernel-converge-A` @ `08895e9b`. Box solvency AMD EPYC 7282 Zen2
(`root@10.0.2.240`), FROZEN gov=performance/boost=0, **llama-server SIGSTOP'd for
the timed window** (restored CLEAN at exit: gov=ondemand, boost=1, llama state `Rl`,
600s watchdog did NOT fire). Build `/dev/shm/zen-tgt` `--no-default-features
--features gzippy-native`, `RUSTFLAGS=-C target-cpu=native`, flavor `parallel-sm+pure`.
Rig: `scripts/bench/zen_ceiling_{gate0,measure,driver}.sh` + `zen_ceiling_report.py`.

## What was built (cursor-agent design-reviewed)
The pre-existing `GZIPPY_MARKER_CEILING` (u8) seeds a zeroed 32 KiB window so the
speculative chunk decodes "clean" — but leaves `data_with_markers` EMPTY, so the
consumer's `resolve_and_narrow_markers_in_place` is a NO-OP. cursor-agent's design
review (correct) flagged that this deletes BOTH the u16 marker write traffic AND the
apply-window resolve gather — costs a marker-decode asm CANNOT remove. The CORRECTED
U16-preserving arms keep that traffic, crediting ONLY clean decode speed, with two
resolve-location bracket ends:
* `GZIPPY_MARKER_CEILING_U16W` — phantom resolve in the decode WORKER (parallel; optimistic).
* `GZIPPY_MARKER_CEILING_U16`  — phantom resolve INLINE on the CONSUMER thread, once
  per chunk (serial; pessimistic).
Gate-0 PASSED unpaused (all 3 arms non-inert: hits>0, resolve_bytes ≈ full decoded
volume, sha≠baseline; base+rg sha==zcat; path=ParallelSM).

## BRACKET (interleaved best-of-N=9, /dev/null both arms, A/A spread tight)
ratio = arm_best / best(rg). gz=baseline, u8=GZIPPY_MARKER_CEILING, u16c=consumer-serial,
u16w=worker-parallel.

| cell        | gz ms | u8 ms | u16c ms | u16w ms | rg ms | r_base | r_u8  | r_u16c | r_u16w | AAg% | AAr% |
|-------------|------:|------:|--------:|--------:|------:|-------:|------:|-------:|-------:|-----:|-----:|
| silesia T4  | 254.7 | 655.2 | 1034.6  | 799.8   | 240.9 | 1.057  | 2.719 | 4.294  | 3.319  | 0.1  | 0.5  |
| silesia T2  | 436.7 | 770.1 | 1084.6  | 1040.2  | 415.5 | 1.051  | 1.853 | 2.610  | 2.504  | 0.7  | 0.0  |
| squishy T4  | 437.5 |1318.6 | 1943.2  | 1577.5  | 417.1 | 1.049  | 3.162 | 4.659  | 3.782  | 0.9  | 2.0  |
| monorepo T2 | 136.1 | 140.6 | 202.4   | 198.1   | 127.2 | 1.070  | 1.105 | 1.591  | 1.557  | 0.3  | 0.4  |
| nasa T4     | 216.8 | 274.2 | 668.8   | 450.7   | 244.3 | 0.888  | 1.123 | 2.738  | 1.845  | 3.4  | 0.2  |

## CRITICAL FINDING — the seeded-window ceiling is INVALID (perturbation moves the wall the WRONG way)
The baseline reproduces the known clean AMD loss (r_base 1.05–1.07 on the losing cells,
A/A < 3.4% → clean). BUT the "speed ceiling" arms are all SLOWER than baseline, not
faster:
* u8 ceiling / gz baseline = **2.5–3.0×** on the decode-bound cells (silesia-T4 655 vs
  254; squishy-T4 1319 vs 438). A valid decode-speed ceiling must be FASTER than the
  production decode (clean asm ~4.7 cyc/B vs marker loop ~11.7). It is 2.5–3× slower.
* Therefore the pre-existing u8 STEP-1 ceiling (commit 981575f4) was INVALID — not
  "over-generous/too-fast" as the design review hypothesized, but the OPPOSITE: it
  INFLATES the wall, so it cannot bound a speed-UP (CLAUDE.md Gate-2: a perturbation
  that does not move the wall in the analyzable direction is not a measurement).

MECHANISM (code-read + empirical, HYPOTHESIS-tier): seeding a 32 KiB window routes the
chunk through the seeded-window branch in `decode_chunk_with_rapidgzip_impl`
(`finish_decode_chunk_with_inexact_offset` / `finish_decode_chunk_seeded_block_native`
— the StreamingInflateWrapper / `unified::Inflate` "second clean engine"), NOT the fast
`run_contig` asm kernel the production marker path flips to. So the u8 ceiling measured
a SLOW clean engine, never the fast asm. (Confirming this would need a route-disable
A/B; banked as the mechanism hypothesis.)

## The ONE valid signal: u16 − u8 delta (both arms share the same seeded decode)
The difference between the u16 arms and u8 isolates the u16-write + resolve traffic the
corrected arms add (same seeded decode underneath):
* silesia-T4: u16w−u8 = **+145 ms** (parallel resolve+u16); u16c−u8 = **+380 ms** (serial).
* squishy-T4: u16w−u8 = +259 ms; u16c−u8 = +625 ms.
These are COMPARABLE TO OR LARGER THAN the entire ~250–440 ms baseline wall. Even
discounting (a) the conservative full-chunk resolve volume (real marker volume is a
fraction, ~31% of symbols) and (b) the separate-pass / serial penalty, the u16-write +
apply-window resolve traffic is clearly NOT a negligible tail — it is heavy. This points
AWAY from "decode speed is the dominant remaining cost".

## VERDICT — DO NOT BUILD THE MARKER ASM (the confirm-before-code SAVE); clean CONFIRM is BLOCKED
* A clean **CONFIRM is impossible**: the decode-speed ceiling instrument is INVALID
  (seeded decode is slower than baseline), so it cannot demonstrate decode-dominance.
* The pre-registered **REFUTE direction is supported** by the only valid signal (u16−u8
  shows the u16/resolve traffic is heavy) AND by the prior banked bound
  (project_amd_t2t4_phase_and_t4_floor: NET prize ~3–10% total, "bounded by the small
  total gap"). The baseline gap itself is only ~14 ms on silesia-T4 (254.7 vs 240.9).
* ACTION: do NOT start the multi-session marker-asm effort on this evidence. To get a
  VALID decode-speed ceiling one would need a DIFFERENT instrument — e.g. route the
  window-absent decode through the FAST `run_contig`/asm kernel directly (not the slow
  seeded engine) while keeping u16+resolve, or a frequency-pinned cyc/B microbench of
  `decode_marker_fast_loop` in isolation. That is the re-open trigger.

## SCOPE / TIER
STRONG (gated, clean AMD, llama-paused, A/A-tight, N=9) for: baseline gap reproduction
AND the instrument-invalidity finding (u8/u16 ceilings slower than baseline). The
u16−u8 "resolve traffic is heavy" reading is MEDIUM (confounded by full-chunk volume +
separate-pass). AMD/Zen2 only; Intel not run. Re-verify after any change to the
seeded-window route or the marker/resolve path.
