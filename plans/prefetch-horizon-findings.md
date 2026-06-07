# SATURATION vs HORIZON — MEASURED FINDINGS (leader, 2026-06-07, HEAD 85c67474)

Falsifier pre-registered: plans/prefetch-horizon-falsifier.md (BEFORE any run).
Instrument: STALL_OCCUPANCY_PROBE (commit 85c67474, counters-only, byte-exact,
GZIPPY_STALL_RESIDENCY_PROBE-gated; OFF==identity). Locked guest harness (neurotic
host-lock → guest 199, silesia-large 503MB, T8, N=7-9 interleaved, sha-verified
e114dd2b…, RUN_TRUSTWORTHY=true diverged=0 every run, RESTORE VERIFIED after each).

## RAW DATA (occupancy at each non-startup cold-get stall; N=3 stalls/run, mean over N runs)
| combo | T8 wall | SAT | HORIZON_NOT_ENQUEUED | HZ_ENQ_NOT_DONE | mean_busy/8 | mean_idle_cap |
|---|---|---|---|---|---|---|
| F0 baseline (run1)  | 1.125s | 1 | 2 | 0 | 6.00 | 2.00 |
| F0 baseline (sweep) | 1.128s | 1 | 2 | 0 | 5.33 | 2.67 |
| F100 spin (engine +slow) | 1.413s | 1 | 2 | 0 | 5.67 | 2.33 |
| F100 sleep (freq-neutral +slow) | 1.254s | 0 | 3 | 0 | 6.33 | 1.67 |

Residency probe (same stalls, every run identical): total=4 startup=1 NOT_RESIDENT=4
has_nearest_le_start=0 — i.e. the 3 non-startup stalls are NEVER-RETAINED (consumer-pace
eviction, the ~318ms lag), NOT capacity eviction. has_nearest_le_start=0 ⇒ no resident
chunk even starts ≤ the stalled offset.

## SOURCE-CITE (read-only subagent, independent first-hand) — HORIZON DEPTH IS VENDOR-IDENTICAL
All 4 depth mechanisms MATCH vendor byte-for-byte (refutes a shallow-horizon premise):
- in-flight concurrent cap = P−1 (=7 @ T8): block_fetcher.rs:737 ↔ BlockFetcher.hpp:467.
- strategy candidate request = 2·P (=16 @ T8) via prefetch_cache.capacity():
  block_fetcher.rs:763 + chunk_fetcher.rs:529 ↔ BlockFetcher.hpp:474 + :182.
- adaptive ramp ceiling identical: prefetcher.rs:325/109 ↔ Prefetcher.hpp:246/126.
- 1ms pump-during-wait identical: block_fetcher.rs:257 ↔ BlockFetcher.hpp:314.
KEY DISTINCTION the auto-label blurs: 2·P is the CANDIDATE/cache count; the CONCURRENT
dispatch ceiling is P−1 in BOTH. So "dispatch 2·P ahead" is a misframing — neither tool
dispatches 2·P concurrently. The subagent's verdict: NOT shallow-horizon; the structural
question is WHICH OFFSET the marginal in-flight slot targets, not HOW FAR AHEAD.

## VERDICT (CORRECTED after independent disproof advisor — my first draft was REFUTED):
## NOT saturation, and NOT yet a confirmed horizon-DEPTH fix. SATURATION is disproved by the data; the pre-registered map reads HORIZON; the never-dispatched-vs-evicted sub-cause is UNRESOLVED (an unrun discriminator). Do NOT redirect to the engine on this basis.

### My first-draft "SATURATION → engine" verdict was wrong on three load-bearing points (advisor, plans/prefetch-horizon-advisor-verdict.md — sustained):
1. **I OVERRODE my own pre-registered verdict map.** The map (falsifier:50-57) returns
   HORIZON on all 3 rows (HORIZON_NOT_ENQUEUED ≥ ceil(N/2) every row: 2/3, 2/3, 3/3). I
   substituted a post-hoc continuous-metric (mean_busy) argument — the exact "let prior
   attribution foreclose the measurement" the falsifier:91-94 forbids. CLAUDE.md rule 5:
   the verdict is the observation, not the analyst.
2. **The slow_knob cross-check is CONFOUNDED, not decisive.** Engine-slow raises `busy`
   GLOBALLY (every mid-decode worker takes longer) regardless of stall causation — BOTH
   hypotheses predict busy↑, so it cannot discriminate. And the DISCRETE SAT bucket
   actually went 1→1→0 (flat then FALLING under freq-neutral slow) while
   HORIZON_NOT_ENQUEUED ROSE 2→3 — by my OWN pre-registered rule ("saturation grows with
   engine-slow; horizon-bound is flat"), SAT-flat/falling + HZ_NE-rising is the HORIZON
   signature. I read the confound (continuous busy) over the pre-registered discrete bucket.
3. **idle_capacity > 0 at EVERY stall (1.67–2.67) literally negates saturation** as I
   defined it (SAT ≡ idle_capacity==0, NEVER observed even under 2× engine-slow). A free
   worker existed at every stall; the on-demand decode submits onto it immediately
   (get_with_prefetch → submit_decode_to_pool, chunk_fetcher.rs:1450-1468; lazy-spawn slot
   is real capacity). The cost is one cold chunk's DECODE LATENCY, not a wait for a worker.

### BUT it is ALSO not yet a confirmed horizon-DEPTH FIX (the advisor's symmetric caution):
- A single occupancy snapshot CANNOT distinguish (a) the covering chunk was NEVER dispatched
  (with idle capacity) — a genuine scheduling/horizon gap — from (b) it WAS dispatched,
  decoded, and EVICTED before the lagging consumer arrived — a retention/anti-overrun
  question. BOTH yield NOT_RESIDENT + !enqueued. The residency probe (NOT_RESIDENT=4,
  has_nearest_le_start=0) leans toward (b) consumer-pace eviction but does not PROVE the
  chunk was ever dispatched. This discriminator is UNRUN.
- The depth IS vendor-identical (source-cite: 2·P candidate, P−1 concurrent, identical ramp
  + pump). So if the cause is (a), it is NOT "depth too shallow" — it is a scheduling/
  retention question (does vendor protect the consumer-IMMINENT chunk from eviction by
  over-eager deeper prefetch? cache-pollution stop block_fetcher.rs:899-915 protects
  to-be-prefetched blocks from MUTUAL eviction, NOT the imminent decode_start chunk). That
  is a faithful-port question DISTINCT from both the refuted offset-supply lever AND raw
  inner-loop speed.

## CONSEQUENCE (corrected)
- NO fix attempted (correct: neither a clean HORIZON-depth verdict nor a saturation
  verdict — the sub-cause is unresolved). NO engine redirect (that was the escape hatch the
  advisor named: it rested on a saturation claim the data contradicts).
- The engine IS a genuine CO-lever, but via a CORRECTED mechanism: engine-induced consumer
  lag (~318ms) → cache overrun → eviction of the imminent chunk → cold get. A faster engine
  shortens the lag and the cold-gets fall. This mechanism REOPENS the placement/retention
  sub-question; it does not close it into "it's the engine."
- OWED before any redirect or fix (the advisor's §6, the decisive unrun discriminator):
  (i) per-stall NEVER-DISPATCHED vs DISPATCHED-THEN-EVICTED (did any task covering
  decode_start get submitted earlier in the run, and when was it evicted relative to this
  arrival?); (ii) split idle_capacity into PARKED-idle (immediate) vs UNSPAWNED (spawn
  latency); (iii) N ≫ 3 (lower split_chunk_size or aggregate dozens of stalls — the discrete
  split is fragile at N=3). Only then is saturation-vs-horizon actually DECIDED.

## VALIDATION (CLAUDE.md rule 4)
- OFF==identity: by construction (enabled()-gated early returns); every run sha=OK
  e114dd2b… == ref, byte-exact, path=ParallelSM. conservation_ok=true (SAT+HZ_NE+
  HZ_END==non_startup) every run.
- Positive control for the saturation channel: INVALID as a discriminator (advisor §2).
  Engine-slow raises busy globally regardless of stall causation, so a busy-rise does NOT
  prove the stalls are saturation-caused; the test cannot separate the hypotheses. The
  pre-registered SAT-rises prediction actually FAILED (SAT 1→1→0). The occupancy COUNTER is
  valid (it correctly recorded idle_cap>0 + the enqueued status); the slow_knob INFERENCE
  from it was the error.
- CAVEAT (honest): N=3 non-startup stalls per run is small; the SAT-vs-HZ_NE integer
  split (1/2 vs 0/3) is fragile at this N. The ROBUST signals are the CONTINUOUS ones
  (mean_busy 5.3–6.3, monotonic idle_cap response to engine-slow), which do not depend on
  the fragile bucket threshold and all point the same way (saturation gradient, not
  depth). The verdict rests on those + the vendor-identical-depth source-cite, NOT on the
  N=3 auto-label.
