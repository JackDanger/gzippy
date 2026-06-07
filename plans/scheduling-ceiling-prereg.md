# Pre-registration — scheduling/serial removal-oracle ceiling (owner turn, HEAD f1aceee1)

## Question
Does removing the SCHEDULING/SERIAL overlap loss (pool-fill gap + in-order
consumer head-of-line `future::get` wait) — while keeping the REAL window-absent
marker engine at its current (slack-masked) compute rate — bring the T8 whole-system
wall to a TIE with rapidgzip (~0.13s, ~1.0×)?

## First-hand baseline (locked guest 10.30.0.199, gzippy-isal native, sha 028bd002…cb410f, T8)
| metric | gzippy HEAD | rapidgzip | ratio |
| decodeBlock SUM | 0.803s | 0.502s | 1.60× |
| Theoretical Optimal (÷8) | 0.100s | 0.068s | 1.47× |
| Total Real Decode Duration | 0.116s | 0.084s | 1.38× |
| std::future::get (consumer wait) | 0.089s | 0.064s | 1.39× |
| serial tail (wall − Real Decode) | 0.058s | 0.046s | 1.26× |
| WALL (interleaved best, measure.sh) | 0.174-0.177s | 0.130s | 0.736-0.755× |

seedfull cell (windows seeded → CLEAN engine, 0 spec-failures, but ENGINE PREMIUM
ALSO removed — NOT the targeted isolation): Real Decode 0.086, future::get 0.083,
wall 0.128 = 1.029× rg TIE. Confirms a tie is reachable with everything removed;
does NOT isolate scheduling from engine.

## Oracle (targeted, keeps the REAL marker engine)
`GZIPPY_PERFECT_OVERLAP=1` (measurement-only, byte-transparent on the output):
PRE-DECODE + PRE-RESOLVE every chunk to completion in a fully-parallel warm phase
(Fill→100%, every post-process future resolved) BEFORE the in-order consumer runs;
then the consumer drains a cache where every `get`/`recv` returns immediately (no
head-of-line block, no fill gap). Wall = parallel-warm time + serial drain time.

NOTE this oracle's wall is a STRICT UPPER BOUND on the true perfect-overlap wall
(warm and drain run sequentially, not overlapped). If even this upper bound ties rg,
the tie is firmly reachable in the scheduling/serial term. If it does NOT tie, the
engine premium (Theoretical Optimal 1.47×) is implicated and a pure-scheduling fix
lands SHORT — escalate to the engine.

## Pre-registered falsifiers (decide BEFORE seeing the number)
- F1 (tie lives in scheduling): GZIPPY_PERFECT_OVERLAP wall ≤ ~1.05× rg (≤ ~0.137s)
  ⇒ removing the overlap loss reaches the tie; the scheduling/serial term is the binder.
- F2 (engine still binds): GZIPPY_PERFECT_OVERLAP wall ≥ ~1.15× rg (≥ ~0.150s)
  ⇒ the engine Theoretical-Optimal premium (1.47×) survives perfect overlap; a
  pure-scheduling fix is INSUFFICIENT. Re-perceive: the binder is BOTH.
- F3 (between, 1.05–1.15×): partial — scheduling helps but does not fully close;
  report the residual and the engine's share.

## Arithmetic prediction (to be confirmed/refuted by the oracle, NOT trusted)
If wall→Real Decode under perfect overlap: gzippy 0.116 vs rg WALL 0.130 ⇒ ~0.89× (TIE-or-better).
BUT floor-to-floor (both keep an irreducible serial tail): gzippy Real Decode 0.116
is itself 1.38× rg's 0.084, so a symmetric serial tail leaves gzippy ~1.1-1.2× rg.
The oracle resolves which holds. Pre-registered expectation: F3 (partial) — most
likely lands ~0.130-0.140s = ~1.0-1.07× (the engine premium is real but small at the
wall because Real Decode 0.116 < rg wall 0.130).

## Frequency-neutral control
measure.sh interleaved (A/B/A/B) is freq-neutral by construction. Re-run if load > 2.

---

## RESOLUTION (oracle-grounded, advisor-corrected — verdict plans/scheduling-ceiling-advisor-verdict.md)

The advisor REFUTED my arithmetic F2 conclusion as a forbidden Rule-3 extrapolation
(the 0.116+0.043=0.159 sum is the prereg's OWN strict UPPER BOUND, double-counting the
overlapping tail). Correct status: **F3 (partial) / both terms live.** Resolved with
ACTUAL oracles, not arithmetic:

1. **The faithful "perfect window-overlap" oracle IS seedfull.** gzippy's dispatch is
   architecturally COUPLED: a present 32KiB window routes to the CLEAN engine
   (gzip_chunk.rs:790 `len()==MAX_WINDOW_SIZE`) vs the MARKER bootstrap otherwise
   (:826). So the ONLY way to give the in-order consumer pre-resolved windows (remove
   head-of-line wait) is to seed windows — which ALSO flips the engine clean. A
   pure-scheduling oracle that keeps the marker engine is IMPOSSIBLE in-architecture.
   seedfull (the oracle) = T8 0.128s = **1.029× rg = TIE = F1** (WIN at T16, 1.121×).
   Its `future::get` (0.083s) ≈ HEAD's (0.089s) yet it ties ⇒ the head-of-line wait
   is NOT the sole binder, AND the tie IS reachable once windows are overlapped.

2. **Scheduling IS firmly on the critical path** (negative control): `GZIPPY_NO_PREFETCH`
   = T8 0.523s = **0.253× rg (3× slower)**. Removing the prefetch overlap is catastrophic.
   `future::get` HALVES T8→T16 (0.089→0.046) = signature of criticality, not slack
   (I had read this backwards; advisor corrected C2).

3. **The engine premium ALSO reaches the wall** (HEAD→seedfull A/B): wall moved 0.046s
   while future::get held ~fixed; ~0.040s moved on the engine-mode axis.

**BOUNDED CEILING:** the T8 TIE IS reachable (seedfull proves it, F1) but BOTH the
scheduling overlap AND the window-absent marker-engine rate are live, COUPLED terms.
The faithful path to the tie WITHOUT seeding (rg ties unseeded at 34.5% markers) is:
make the window-absent decode cheaper at the wall — EITHER (a) a faster window-absent
u16 MARKER engine (rg decodeBlock 0.502 vs gzippy 0.803 = 1.6×, both window-absent), OR
(b) publish/resolve predecessor windows EARLIER so MORE chunks hit the clean path at
high T (closing project_confirmed_offset_prefetch_gap — dispatch TIMING, not horizon
DEPTH which is vendor-identical). Both are advisor-flagged as UNCONFIRMED-by-perturbation
(C4) and must be confirmed by a causal perturbation before a work-stretch — do NOT enter
on attribution.
