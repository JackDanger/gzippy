# STEP C — REVISED DESIGN (supervisor, 2026-06-07): the gate to TIER-3

TIER-2 PROVE is complete. All lever ceilings measured + advisor-vetted. This step
SYNTHESIZES them into the revised design that, once supervisor+advisor ratify, unlocks
TIER-3 implementation. (Memory: project_pregate_placement_is_dominant_lever.)

## Proven measurement inputs (locked guest, T8, silesia-large, all advisor-corroborated)
- Baseline gzippy T8 = 1.124s vs rapidgzip 0.540s (~2.08×).
- **Placement ceiling:** placement-perfect floor ~0.56–0.66s (ramp-consistent) = 7–26% over
  rapidgzip → necessary but INSUFFICIENT.
- **Engine ceiling:** clean-only (publish-chain intact, byte-exact) = 0.6134s = +13.7%;
  per-chunk clean 92.7 ms vs 39 ms = 2.38×. Conservative bound (gzippy had a coordination
  advantage and still lost) → engine must reach ~igzip-class clean rate.
- **Structural coupling:** clean decode requires CONFIRMED boundaries = the placement fix.
  Scheduler port is a PREREQUISITE for the engine to run clean. Order is forced.
- **Traffic (1b):** resolve-traffic component already subsumed by the engine oracle; only
  the u8-vs-u16 WRITE sub-lever remains untested (now an engine-rate sub-technique).

## THE DELIVERABLE (primary): the revised TIER-1 design, rewritten with the above
A single coherent architecture for the 1.0× tie, with both CO-PRIMARY levers and the
forced order. Must specify:
1. **PLACEMENT (first, faithful port — ARCHITECTURE, no innovation):** transliterate
   rapidgzip's chunk scheduler / prefetch / confirmed-boundary dispatch so workers decode
   at confirmed deflate boundaries (no partition-guess misalignment), recovering the ~42%→
   ideal parallel efficiency AND enabling clean engine decode. Cite vendor file:line for
   each ported piece. This is the [[project_confirmed_offset_prefetch_gap]] fix.
2. **ENGINE (co-primary, inner-loop OPEN TERRITORY — pure-Rust+inline-ASM authorized):**
   close the 2.38× per-chunk clean gap to igzip-class. Name the concrete techniques
   (igzip AVX2 clean inner loop: wide back-ref SIMD copy, packed multi-literal write,
   wide refill; u8-direct clean write = the 1b sub-lever) and how each will be isolation-
   benchmarked (PROOF-1-style, with the in-bench ISA-L oracle as positive control) before
   integration. Inline ASM where Rust codegen lags.
3. **The reachability arithmetic:** show how placement-perfect + engine-igzip-class
   combine (NOT naive-additive — they're coupled) to a projected ~0.54s tie, and state the
   residual risk honestly (the engine front is the high-risk part: matching igzip AVX2 in
   pure-Rust). If the math says the tie is unreachable even with both maxed, SAY SO.
4. **Structure mandate** (still owed, do during TIER-3): gzippy-isal/gzippy-native subdir
   split (the native=marker_inflate vs isal=resumable confusion that mis-sited the slow_knob
   is exactly why); dead-code removal (unified.rs HAS_BMI2; specialized_decode/SPEC_CACHE;
   stale guest_fulcrum_capture.sh:69-71 comment). Names describe behavior.

## SECONDARY (only if quickly tractable this turn): run experiment 1b
u8-write clean A/B to measure the write sub-lever's effect on the 92.7 ms/chunk engine rate.
If it needs real instrument work, DEFER it into TIER-3 engine implementation and note it —
do not let it block the design deliverable.

## CHECKPOINT (STOP — this is the TIER-3 gate)
Deliver the revised design. Route through an independent disproof advisor (synchronous,
read-only, verdict to plans/step-c-design-advisor-verdict.md) attacking: the reachability
arithmetic, the faithful-port-vs-innovation boundary for placement, and whether the engine
techniques can plausibly hit igzip-class. Then STOP for supervisor ratification. Do NOT
start TIER-3 implementation until the supervisor authorizes it.

## DISCIPLINES (enforced)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; guest runs from
a Bash task that HOLDS the ssh (bare claude -p SIGHUPs its ssh → orphans guest run); verify
guest idle before + restore host after each measurement; serialize builds via cargo-lock.sh;
leave NO orphaned processes; reject a lever only with a mechanism; numbers only from the
locked harness. Update plans/orchestrator-status.md.
