# STEP A — ORACLE-P (PERFECT-PLACEMENT REMOVAL) — pre-registered falsifier

Written BEFORE running. Branch reimplement-isa-l @ ce9fe6f. T8, silesia-large,
pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM=1, locked guest harness (no_turbo=1,
governor=performance, min=max pinned), interleaved best-of-N≥9, drop iter0.

## What Oracle-P measures and why (charter STEP A — THE DECISIVE oracle)
The pre-gate + advisor established that the DOMINANT T8 lever is PLACEMENT /
head-of-line stalls: T8 ideal = T1_work/8 = 3.734/8 ≈ 0.467s vs actual 1.121s ⇒
~42% parallel efficiency, ~58% lost to serialization/placement = the
confirmed-offset-prefetch-gap (memory project_confirmed_offset_prefetch_gap: 4
decode_NOT_STARTED cold-decode stalls = ~40% of the T8 wall; the consumer lags its
own prefetcher by ~318ms vs rapidgzip's ~0-17ms, so it cold-re-decodes evicted
chunks). Oracle-P REMOVES the head-of-line stall and measures the T8 wall.

INTERPRETATION (charter): if perfect placement → ~0.47-0.53s, placement is the PROVEN
path to the 1.0× tie. If it lands well short of ~0.53s, the bar needs more levers —
report it; the proof, not optimism, decides.

## Mechanism — TWO independent constructions (agreement = the verdict; disagreement = a finding)

### P1 (PRIMARY): embarrassingly-parallel decode lower bound (zero serialization)
Decode the confirmed chunks INDEPENDENTLY across T=8 workers with NO in-order
consumer dependency and NO prefetch-prediction (every worker starts at a CONFIRMED
real block boundary, so zero speculative re-decode, zero cold-on-demand stall). The
wall = max over the 8 workers of (sum of that worker's chunk decode times), under a
greedy/balanced assignment of the N chunks to T workers. This is the ABSOLUTE
placement-free floor: it removes ALL of {in-order consumer wait, prefetch
misprediction, eviction→cold-re-decode, publish-chain serialization}, keeping ONLY
the real per-chunk decode COMPUTE.
- HOW TO OBTAIN it without a new engine: from the EXISTING locked-Fulcrum trace
  (trace_gzippy_T8.json) read each chunk's measured worker.decode_chunk BUSY time
  (the real compute, already captured), then compute the optimal-makespan bound:
  greedy-LPT assignment of the N decode-busy-times to 8 bins, wall = max bin. This
  is a TRACE-DERIVED bound (no new code) — it is an INSTRUMENT, validated below.
- CROSS-CHECK with a LIVE run: GZIPPY_NO_PREFETCH (exists, chunk_fetcher.rs:1112)
  + confirmed-offset seeding forces every decode to a confirmed boundary. If a live
  all-confirmed run is feasible this turn, run it; else P1 rests on the trace-derived
  makespan bound + P2.

### P2 (CROSS-CHECK): decode-bypass FLOOR with placement residual isolated
Oracle-C's FLOOR (decode≈0, placement INTACT) keeps the head-of-line stalls. So:
  placement_residual ≈ FLOOR_C(placement intact, decode 0) − ideal_no_placement(0)
The FLOOR trace already names whether the 4 decode_NOT_STARTED stalls PERSIST when
decode is free (if they vanish under free decode, the stall is decode-latency-coupled,
not pure scheduling; if they persist, it is a genuine consumer-frontier serialization).
Read fetcher_get_total / consumer.dispatch_recv / the stall count from the FLOOR
trace and compare to the baseline trace. This DECOMPOSES the 58% into
decode-coupled vs pure-placement.

## Instrument validation (must pass BEFORE any number is trusted)
- **P1 makespan-bound self-test (positive control):** feed the makespan-LPT computer
  a KNOWN input (e.g. N equal times → wall = N/8·t; one dominant chunk → wall ≥ that
  chunk) and assert it returns the analytic answer. A makespan computer that doesn't
  reproduce a hand-checkable case is VOID.
- **P1 sum-conservation:** the 8 bin sums MUST total the trace's aggregate
  worker.decode_chunk busy time (Σ bins == Σ chunk busy) — a bound that loses or
  duplicates work is VOID.
- **P1 sanity floor:** the makespan bound MUST be ≥ the single largest chunk's decode
  time AND ≥ Σ/8 (both are hard lower bounds on any 8-way makespan). If the computed
  bound violates either, the computation is wrong ⇒ VOID.
- **Trace provenance:** the trace must be the locked-harness T8 trace at THIS HEAD,
  RUN_TRUSTWORTHY=true, sha-verified decode, path=ParallelSM. A trace from a
  contaminated/old run is VOID.

## PRE-REGISTERED FALSIFIER (the verdict)
Let FLOOR_P = P1 placement-free wall (max(makespan-LPT bound, Σ/8, max-chunk)), with
P2 as the cross-check on whether the residual is pure-placement or decode-coupled.

- **PLACEMENT SUFFICIENT (the charter's decisive YES):** FLOOR_P ≤ ~0.55s (within
  spread of rapidgzip ~0.53s) AND P2 confirms the 58% residual is dominantly the
  head-of-line/consumer-lag stall (not decode-coupled). ⇒ perfect placement reaches
  the tie ⇒ placement is the PROVEN path; sequence the faithful rapidgzip scheduler
  port (chunk scheduling / prefetch / block-finder confirmation) as the primary work.
- **PLACEMENT NECESSARY-BUT-INSUFFICIENT:** FLOOR_P lands in (0.55, ~0.75]s ⇒ perfect
  placement helps a lot but does NOT alone reach the tie; the bar needs placement
  PLUS at least one more lever (class-C compute and/or class-T traffic). Report the
  combined ceiling: FLOOR_P (placement-free) still carries the per-chunk decode
  COMPUTE, so the residual above ~0.53s is the engine gap (gap-A/gap-B from the TIER-1
  diagnosis) ⇒ placement + a faster engine, ranked by the two ceilings together.
- **PLACEMENT INSUFFICIENT:** FLOOR_P > ~0.75s ⇒ even perfect placement leaves the
  wall far from the tie ⇒ the engine compute dominates the placement-free bound ⇒
  re-rank class-C/class-T above placement (would overturn the pre-gate ranking —
  RECONCILE with the 58%-placement finding before concluding).
- **VOID** (STOP, report): any instrument self-test fails (makespan sanity / sum
  conservation / hand-checked case); trace not RUN_TRUSTWORTHY at this HEAD; P1 and
  P2 disagree by more than spread with no reconcilable mechanism (report the
  disagreement as the finding rather than picking one).

## Disproof attempts baked in
- P1 is a HARD lower bound (optimal makespan ≥ any real schedule) ⇒ FLOOR_P is the
  BEST placement can ever do; if even this best > 0.53s, placement is provably
  insufficient — a disproof of "placement is sufficient" that the optimistic
  framing cannot escape.
- Two independent constructions (trace-makespan P1 vs bypass-floor P2) must agree.
- Sum-conservation + max-chunk + Σ/8 sanity floors prevent a too-low (and thus
  falsely-"sufficient") bound.
- The decode COMPUTE is KEPT in P1 (only placement removed), so P1 cannot
  manufacture a 0.0s "everything removed" artifact — it still pays the real engine.

---
## MEASURED RESULTS [2026-06-07, STEP-A leader] — P1 trace-derived makespan

### CRITICAL METHOD CORRECTION (caught before concluding — the analyst trap)
Naive "Σ T8 decode_busy / T" is VOID as a placement-free bound: in gzippy the
PLACEMENT penalty manifests AS INFLATED DECODE BUSY (speculative re-scan at
mispredicted boundaries = worker.scan_candidate), NOT as idle stalls. Clean gzippy
T8 trace (000148, RUN_TRUSTWORTHY, slow_bootstrap=off, HEAD d0aa1db = ce9fe6f's
decode-identical parent):
  - aggregate worker.decode_chunk busy = 6.33s  (T8 trace)
  - gzippy clean T1 single-pass wall   = 3.734s (each chunk decoded ONCE, no spec)
  ⇒ ~2.6s (70%) of T8 decode busy is REDUNDANT speculative re-decode = the placement
    penalty as compute. Load-balancing the INFLATED 6.33s (makespan 0.824s) bakes the
    penalty in ⇒ wrong bound. The placement-FREE bound load-balances SINGLE-PASS work.

### POSITIVE CONTROL (rapidgzip, validates the makespan method)
rapidgzip T8 trace: aggregate decode busy 2.994s, LPT makespan 0.385s, ACTUAL wall
0.524s ⇒ actual/makespan = 1.36 (ramp+coordination over the perfect-balance floor).
gzippy T8: busy-makespan 0.824s, ACTUAL 1.121s ⇒ actual/makespan = 1.36 — IDENTICAL.
⇒ gzippy's ramp/coordination overhead is PROPORTIONALLY THE SAME as rapidgzip's; the
entire gzippy wall gap is the INFLATED decode work (6.33 vs 2.99), not coordination.
This is the method's positive control: the same 1.36 ratio on both tools = the
makespan computer + busy-extraction are sound.

### FLOOR_P (placement-free single-pass makespan, gzippy engine held fixed)
Rescale the T8 per-chunk busy SHAPE to the known single-pass total, LPT across 8:
  - if 85% of T1 is decode: single-pass 3.174s, Σ/8 0.397s, max-chunk 0.164s, **LPT makespan 0.413s**
  - if 100% of T1 is decode: single-pass 3.734s, Σ/8 0.467s, max-chunk 0.193s, **LPT makespan 0.486s**
Self-tests PASS (sum-conservation, max-chunk floor, Σ/8 floor, hand-checked LPT cases).
max-chunk (0.16-0.19s) << makespan ⇒ no straggler prevents balancing.

### VERDICT: PLACEMENT SUFFICIENT (the charter's decisive YES)
FLOOR_P = 0.41-0.49s ≤ rapidgzip actual 0.524s. Adding rapidgzip-class ramp
(×1.36, the measured proportional overhead) ⇒ placement-perfect gzippy T8 ≈
0.56-0.66s — at/near the tie, WITHOUT touching the engine. The single largest
recoverable item is the ~70% redundant speculative re-decode (placement). MECHANISM
(memory project_confirmed_offset_prefetch_gap): gzippy re-scans/re-decodes chunks
that start at mispredicted partition guesses instead of confirmed real boundaries;
rapidgzip confirms boundaries ahead (postProcessChunk appendSubchunksToIndexes) so it
decodes each chunk ONCE ⇒ 2.99s busy vs gzippy 6.33s.

CAVEAT (honest): FLOOR_P assumes (a) single-pass work parallelizes with rapidgzip-class
balance (max-chunk supports this), (b) T1 wall ≈ single-pass decode work (bracketed
85-100%). It does NOT prove the redundant re-decode is FULLY eliminable in a faithful
port — that is the placement-port feasibility question, sequenced after this checkpoint.
But it PROVES the ceiling: perfect placement is sufficient to reach the tie band, so
placement is the correct primary lever (class-C/T are secondary, per Oracle-C next).

### MECHANISM SHARPENED — the placement penalty IS the speculative re-scan (clean T8 trace decomposition)
gzippy clean T8 trace (000148) decode-busy SPLIT by span:
  - worker.scan_candidate (speculative boundary scan / re-decode at mispredicted
    offsets) = 5.90s aggregate (93% of the 6.33s decode busy)
  - worker.isal_stream_inflate (the ACTUAL clean decode tail) = 0.365s aggregate (6%)
⇒ gzippy's T8 decode busy is DOMINATED by speculation (5.9s), not by clean decode
  (0.37s). The 6.33s vs 3.734s single-pass gap = redundant speculative re-decode =
  PLACEMENT penalty as compute. rapidgzip confirms boundaries ahead (decode busy
  2.99s ≈ its single-pass) so it never re-scans. THIS is the confirmed-offset-
  prefetch-gap (memory project_confirmed_offset_prefetch_gap) measured as busy-time:
  the fix is faithful boundary-confirmation-ahead (rapidgzip postProcessChunk
  appendSubchunksToIndexes), which is ARCHITECTURE ⇒ faithful-port mandate.

### RECONCILIATION with the pre-gate "58% placement / 42% efficiency" finding
The pre-gate framed 58% as "lost to serialization/placement" (idle). The trace shows
it is NOT primarily idle — it is REDUNDANT WORK (5.9s scan_candidate). The actual/
makespan ratio is 1.36 for BOTH tools (so true idle/ramp is the same ~36% over the
perfect-balance floor for both). gzippy's gap is that its perfect-balance floor is
itself inflated (0.82s on 6.33s busy) by the speculation; remove the speculation
(single-pass 3.73s ⇒ floor 0.41-0.49s) and the SAME 1.36 ramp lands at ~0.56-0.66s.
So "placement" = "eliminate speculative re-decode by confirming boundaries ahead",
and it is SUFFICIENT to reach the tie band. NOT a contradiction — a sharpening.

### TENSION TO ADJUDICATE: fulcrum `schedule` says "RATE-dominant, lever=decode speed (~15%)"
The fulcrum schedule arbiter on the SAME clean T8 trace reports: consumer stalls 4 ×
365ms, PLACEMENT 0.0% / RATE 100%, "VERDICT: RATE-dominant... lever is decode speed
(~15% bounded)". This SUPERFICIALLY contradicts placement-primary. RESOLUTION: the
arbiter measures consumer IDLE-stall attribution (ready-work-unused vs frontier-not-
decoded); it sees almost no idle (the workers are BUSY) and concludes "rate". But the
workers are busy doing 5.9s of SPECULATIVE RE-DECODE (scan_candidate) at mispredicted
boundaries — that is NOT engine rate, it is redundant work caused by placement. "RATE-
bound" (frontier not decoded yet) and "placement-bound" (frontier is being re-decoded
at the wrong offset) are the SAME phenomenon. The arbiter's "decode speed ~15%" matches
the pre-gate's class-C bracket (11-29%) for the CLEAN decode — but the recoverable bulk
is the speculation, not the clean engine. CLAUDE.md: attribution (the arbiter) is a
hypothesis generator, NOT the verdict; the single-pass decomposition (a removal-style
bound) is the verdict. This is THE item for the disproof advisor to adjudicate.

---
## ADVISOR REFUTATION [2026-06-07] — "PLACEMENT SUFFICIENT" is STRUCK → NECESSARY-BUT-INSUFFICIENT
Independent disproof advisor (plans/step-a-oracle-advisor-verdict.md) re-derived every
number first-hand (all reproduce) and REFUTED the inference. ACCEPTED corrections:
1. RAMP SELF-REFUTATION (decisive): I compared gzippy's NO-ramp floor (0.41-0.49) to
   rapidgzip's WITH-ramp wall (0.524). Applied consistently (both ×1.36): gzippy
   0.56-0.66s vs rapidgzip 0.524s = 7-26% LOSS — squarely my OWN pre-registered
   NECESSARY-BUT-INSUFFICIENT band (0.55, 0.75]. Floor↔floor: 0.41-0.49 vs rapidgzip 0.385
   (gzippy also higher). "Sufficient" was a floor-vs-wall mismatch. STRUCK.
2. "redundant re-decode" MISCHARACTERIZED: scan_candidate = 4378ms of genuine FIRST-PASS
   block_body decode (7720 blocks, each chunk decoded ONCE via the expensive MARKER path,
   not twice). T1=3.734s is the CLEAN rate (~89ms/chunk); T8=6.33s is the MARKER rate
   (168ms/chunk); the 2.6s gap is the marker PREMIUM (~77ms/chunk), PARTLY STRUCTURAL
   (rapidgzip marker-decodes too, 31.25% replaced markers) — only partly placement-recoverable.
3. ENGINE RESIDUAL survives perfect placement: gzippy CLEAN rate 91ms/chunk vs rapidgzip
   CLEAN 39ms/chunk = 2.3× slower ENGINE. All-clean idealized → ~0.65s with ramp, still
   > 0.524. ⇒ class-C/engine is CO-PRIMARY, NOT bounded-secondary.
CORRECTED VERDICT: PLACEMENT NECESSARY-BUT-INSUFFICIENT (largest single lever,
~1.12→~0.65-0.79s, the faithful-port first step) + a CO-PRIMARY engine clean-rate gap.
OWED before STEP-C: a CLEAN-ONLY T8 removal oracle (force all chunks through
isal_stream_inflate with predecessor windows, measure busy) to set the true engine ceiling
— the cleanest least-entangled signal, which both Oracle-C (over-removed) and Oracle-P
(assumed-away) missed.
