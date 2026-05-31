# Closing a diffuse performance gap, provably — design

## Why single-lever A/B rabbit-holes (the failure we keep hitting)
The wall of a parallel decode is the length of its **critical path** — the longest chain of
dependency-linked spans. Three facts make per-lever wall-A/B the wrong instrument for a diffuse gap:
1. **Off-critical-path work is free.** Speeding a span not on the critical path = 0 wall change
   (every "copies wall-neutral", "decode TIE" corpse this session).
2. **The wait moves.** Shorten one critical-path span and the critical path *recomputes* to the
   next-longest chain — so a local win shows TIE (a4960635: −178ms straggler → +6ms wall).
3. **Diffuse = no single span dominates.** If the gap is spread over N spans, each lever is ~gap/N;
   on a noisy box that's below the spread → every honest measurement reads TIE. You can run N
   build-and-measure cycles and close nothing, because you never address the *set*.

## The principle: conservation of wall
**wall ≡ Σ(spans on the critical path).** This is an identity, not a model — the critical path is
*defined* as the chain whose lengths sum to the wall. That gives a provable closure:

> **Claim.** If, for a fixed input and thread count, gzippy's critical path is ≤ rapidgzip's in
> *every span-category*, then gzippy's wall ≤ rapidgzip's wall.
>
> **Proof.** wall(gzippy) = Σ_c time_c(gzippy critical path) ≤ Σ_c time_c(rapidgzip critical path)
> = wall(rapidgzip), provided the category decomposition is **complete** (Σ_c = wall, no unaccounted
> time) and the *same* taxonomy is used for both. ∎

The proof's load-bearing precondition is **completeness** (conservation): if any time is unaccounted,
the "wait" can hide there and the bound fails. So the method is only as valid as its conservation check.

## The method (the closure loop — handles "the wait moves" by construction)
1. **Instrument both decoders** with one shared span taxonomy covering EVERY category that can sit on
   the critical path: `decode`, `boundary-scan`, `marker-resolve`, `window-publish`, `copy/drain`,
   `alloc/first-touch`, `sync-wait` (channels/locks/condvars), `consumer-write`, `schedule/dispatch`.
2. **Reconstruct the critical path** (longest path through the wait-for DAG) for each. 
3. **CONSERVATION GATE:** assert Σ(critical-path spans) = wall ± ε (≈2%). If it fails, there is
   unaccounted time on the path → an un-instrumented edge exists → add it and re-run. *This gate is
   what makes the result provable rather than hand-wavy; do not proceed past a failing conservation check.*
4. **Differential:** per category, Δ_c = time_c(gzippy) − time_c(rapidgzip). ΣΔ_c = the whole gap,
   decomposed completely. No category is "missed" — that's the point.
5. **Elasticity, computed once, no fixes built (the anti-rabbit-hole):** Coz-style virtual speedup —
   while running, for each category insert proportional delays into *all other* threads to simulate
   THIS category being k× faster, and measure ∂wall. This yields, in ONE profiling campaign, the true
   ∂wall/∂speed for every category — INCLUDING the wait-moves effect (an overlapped category measures
   ∂wall/∂speed ≈ 0). This is exactly the signal the N build-and-measure cycles were paying for, but
   for all N at once and without writing any fix.
6. **Closure loop:** rank categories by Δ_c weighted by elasticity; port rapidgzip's approach for the
   top one (the *differential* tells you the target shape; `asm_compare.sh` tells you the instruction
   diff); **re-reconstruct the critical path** (the wait moved — that's expected and now visible);
   repeat. Terminate when the conservation-gated critical path is ≤ rapidgzip's in every category, OR
   when the residual categories are provably fundamental (Δ_c is irreducible, e.g. a hardware floor a
   port can't change — and you can prove it, not assume it).
7. **Verdict once, at the end:** `measure.sh` interleaved production A/B. By conservation, if the
   critical path matched rapidgzip category-by-category, this WILL show — there is no hidden span left.

## The tool to build (extend Fulcrum — this is its purpose)
Fulcrum already has trace_v2 + critical-path reconstruction + a Coz module + a rapidgzip trace-patch
(same schema). What's missing — and what we keep paying for by not having — is the **system-wide,
conservation-checked, differential** assembly of them:
- **(T1) Conservation-checked critical-path** — emit the critical path AND the Σ=wall residual; FAIL
  loudly if unaccounted time > ε. Without this we've been trusting incomplete traces (the broken-oracle
  class of bug). This is the single highest-value addition.
- **(T2) Two-sided differential** — ingest gzippy + rapidgzip traces in the shared taxonomy, output the
  per-category Δ table that IS the diffuse gap. (Needs the rapidgzip-side spans at the same boundaries.)
- **(T3) Whole-program Coz campaign** — virtual-speedup every category, output the elasticity vector;
  the dot-product (Δ · elasticity) is the *predicted closeable gain BEFORE any fix is built*. If that
  predicted gain is small, the gap is genuinely fundamental and we STOP — provably, not by fatigue.
- **(T4) Closure ledger** — each category: Δ_c, elasticity_c, verdict {ported→parity | fundamental+proof}.
  The gap is closed iff every row is parity-or-proven-fundamental and Σ residual ≈ 0.

## Why this is the right shape for a DIFFUSE gap specifically
Single-lever asks "does fixing X help?" N times and dies on overlap. This asks "what is the complete,
conserved decomposition of the gap, and which slice of it is *causally* closeable?" — ONCE — then closes
the whole causally-closeable set as a unit, re-checking conservation so the wait can't hide. The diffuse
gap is the *sum*; you close the sum, not a term.

## The honest "very hard" parts (the user already granted this is hard)
- **Complete instrumentation + a passing conservation check** on a 16-thread decoder is the crux; an
  un-instrumented condvar/channel wait silently breaks the proof. T1's loud FAIL is the mitigation.
- **rapidgzip-side taxonomy parity** (C++): the differential needs spans at the *same* semantic
  boundaries, or Δ_c is apples-to-oranges. This is real porting of the trace patch.
- **Coz virtual-speedup in a real decoder** (delay injection at category boundaries without perturbing
  correctness or the very overlap you're measuring) — the Coz paper's mechanism, applied here.
- **Proving a residual category is *fundamental*** (not just "we couldn't close it") — requires showing
  Δ_c is a hardware floor (e.g. DRAM bandwidth at measured traffic), via the microarch tools, not assertion.
