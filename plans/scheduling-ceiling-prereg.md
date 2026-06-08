# Pre-registration — scheduling/serial removal-oracle ceiling (owner turn, HEAD f1aceee1)

## Question
Does removing the SCHEDULING/SERIAL overlap loss (pool-fill gap + in-order
consumer head-of-line `future::get` wait) — while keeping the REAL window-absent
marker engine at its current (slack-masked) compute rate — bring the T8 whole-system
wall to a TIE with rapidgzip (~0.13s, ~1.0×)?

## First-hand baseline (locked guest REDACTED_IP, gzippy-isal native, sha 028bd002…cb410f, T8)
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

## RESOLUTION #2 (2026-06-07, owner turn — GZIPPY_PERFECT_OVERLAP FINALLY BUILT + RUN; advisor REFUTED my read)

The registered decider oracle was built (src/decompress/parallel/perfect_overlap.rs),
self-tested (Rule 4: sha 028bd002…cb410f byte-identical w/ and w/o the oracle on
arm64 + x86_64; warm_hit_frac 0.88-0.96, the 2 misses are offset-0 stream-start
chunks), and run on the locked guest. Advisor verdict: plans/perfect-overlap-advisor-verdict.md.

**THE ORACLE AS BUILT IS WRONG (advisor, load-bearing):** it runs warm (decode-ALL)
fully THEN drain (resolve-chain+write) — it SERIALIZES the two phases production
already OVERLAPS. So its wall is a pessimistic SUM and comes out **SLOWER than
production** (oracle 0.183-0.225s > HEAD 0.177s). A "perfect overlap" config slower
than the un-optimized scheduler is an ANTI-overlap; its LOSS verdict says NOTHING
about F1 (an upper bound built by DESTROYING overlap cannot falsify a TIE claim
about the BEST overlap — the symmetric of last turn's upper-bound-can't-fire-F2).

**THE GENUINE FINDING (advisor-corrected, the real lower bound):** the warm phase
alone = **0.117s** is a TRUE LOWER bound on any schedule's wall (every chunk must
decode; drain 0.066 < warm so it hides under decode). And **0.117 < rg WALL 0.131 <
tie threshold 0.138** — the lower bound is INSIDE the tie zone. I reported "lower
bound above the tie" only by mis-comparing 0.117 to rg's decode FLOOR (0.085)
instead of rg's WALL (0.131). Read correctly with the right denominator: **the
decode floor says the T8 TIE IS REACHABLE by better decode↔drain overlap** — the
scheduling/overlap direction is NOT refuted; this oracle FAILED TO TEST it.

NUMBERS (locked guest, T8, measure.sh interleaved N=11, sha-OK, 2 runs):
rg 0.131-0.132s (1.000) | HEAD 0.177s (0.740-0.745× LOSS) | perfovl 0.225-0.227s
(0.581-0.583× LOSS). Warm/drain split: warm (full-parallel decode-all, real marker
engine, Fill→100%) 0.117s; drain 0.066s. rg --verbose: Theoretical-Optimal 0.085s,
Real Decode 0.104s. Matched floor-to-floor = warm 0.117 vs rg Real Decode 0.104 =
**1.13×** (NOT my brief's denominator-mismatched 1.38×).

**STILL OPEN (the decider question this oracle did NOT answer):** can a REAL
OVERLAPPED schedule (decode overlapping resolve+write, NOT serialized) collapse
production's 0.177s toward the 0.117-0.13 floor? That needs a CORRECTED oracle —
one that overlaps warm with drain (e.g. pipeline the drain to start as soon as each
chunk's predecessor window is ready, while later chunks still decode), NOT
warm-all-then-drain. My implementation was backwards. F1 remains UNDECIDED; do NOT
declare STOP/TIE (Rule 3 + PROCESS FIX #3).

---
## RESOLUTION #1 (SUPERSEDED by #2 — its seedfull/coupling argument stands; its "ceiling bounded" framing was premature, the decider was unrun)

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
