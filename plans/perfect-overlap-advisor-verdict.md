# Disproof-advisor VERDICT — GZIPPY_PERFECT_OVERLAP oracle

Independent, read-only. Branch reimplement-isa-l, HEAD 7aae6c4a + the oracle
overlay. Source-verified first-hand (perfect_overlap.rs; chunk_fetcher.rs
warm/wire/counters/resolve-chain; gzip_chunk.rs:777-829; chunk_data.rs:1195;
run_decode_task window lookup). Numbers as reported in the brief (builds/guest
not re-run, per the read-only mandate).

Convention: **wall ratio = rg_wall / gzippy_wall.** 1.00 = tie. <1.00 = gzippy slower.
Tie threshold F1 (≤1.05× rg): gzippy_wall ≤ 1.05 × 0.131 = **0.138s**.

---

## TL;DR — the oracle does not measure what its name claims

The oracle runs the **warm (decode-all) phase fully, THEN the drain phase** —
i.e. it **serializes** the pipeline's two big phases (decode and resolve+write)
that production already **overlaps**. So its wall is a *pessimistic sum*
(warm 0.117 + drain 0.066 = 0.183 single-run; 0.225 interleaved), the **opposite**
of perfect overlap. The proof is in the brief's own numbers:

> oracle total 0.183s **> production HEAD 0.177s.**

A "perfect overlap" config that comes out **slower than the un-optimized
production scheduler** is not perfect overlap — it is an *anti-overlap* of
decode↔drain. It removed intra-decode head-of-line wait (term 2) but paid for it
by destroying the decode↔drain overlap production already has. Its LOSS verdict
therefore says nothing about F1.

The genuine lower bound on a perfectly-overlapped wall is **max(warm, serial-chain)
≈ warm = 0.117s** (decode-all must happen; drain 0.066 < warm so it hides under
decode). **0.117 < rg wall 0.131 < tie threshold 0.138.** The lower bound sits
*inside the tie zone.* The brief's C3 rescue compared that 0.117 to rg's decode
**FLOOR (0.085)** instead of rg's **WALL (0.131)** and so reported "a lower bound
above the tie" — it is below the tie. That single mis-denominator flips the whole
conclusion.

---

## C1 — "removing scheduling/overlap does NOT reach the tie; F1 FALSIFIED" → **REFUTED**

- The oracle wall (0.225 / 0.183) is a **strict UPPER BOUND** (the brief concedes
  this in C3) and is **inflated by the warm↔drain serialization**, which is the
  inverse of overlap. The prior advisor's binding correction ("an upper-bound LOSS
  cannot fire F2") applies symmetrically to **F1**: an upper bound that is *slower
  than production* cannot falsify a *tie* claim, because the construction added a
  penalty (zero decode↔drain overlap) that a real scheduling fix would never have.
- Disproof angle 5 argues "a config slower than production cannot be the
  tie-reaching config." True for *this* config — but **this config is not the
  scheduling fix.** It is a deliberately pessimal schedule (decode-all-then-drain).
  Its slowness is the serialization artifact, not evidence about the achievable
  overlap floor. F1 asks whether the *best* overlap reaches the tie; a *bad*
  overlap being slow is silent on that.
- Direct contradiction: oracle 0.183 **> production 0.177.** The oracle schedules
  *worse* than production. You cannot read "scheduling can't help" off a config
  that schedules worse than the baseline.

**F1 is NOT falsified.** The oracle fails to decide it in either direction.

## C2 — "perfectly-parallel decode floor (warm 0.117) is 1.38× rg's floor / 0.89× rg's whole wall, reaching the wall even under perfect overlap" → **UPHELD-WITH-CAVEATS on the fact, REFUTED on the inference**

- Narrow fact (gzippy decode floor > rg decode floor): plausibly true and is the
  one real signal here. But the **1.38×** is **denominator-mismatched**: warm
  (0.117) is a *measured parallel wall that includes pool spin-up* (the pool uses
  lazy thread spawn — chunk_fetcher.rs ~1471 `spawned_threads`/`capacity`; warm is
  the first thing timed at drive_t0), whereas rg's "Theoretical-Optimal" 0.085 is
  an *arithmetic* decodeBlock_SUM/parallelism floor with no spin-up. The
  measured-to-measured comparison is warm 0.117 vs rg **Real Decode 0.104 ≈ 1.13×**,
  not 1.38×.
- Inference REFUTED: "reaching the wall even under perfect overlap" is false.
  gzippy's decode floor **0.117 < rg's whole WALL 0.131.** gzippy can decode
  everything in less time than rg's entire run — so the engine floor does **not**
  bind the wall tie. What is left between 0.117 and a 0.131 tie is ~0.014s of
  drain/overlap headroom, i.e. a *scheduling* budget, not an engine wall.
- The "0.89× of rg's entire wall" phrasing is self-undermining: 0.117/0.131 = 0.89
  means warm is **89% of rg's wall = faster than rg's wall**, which argues *for*
  reachability, not against it.

## C3 — "the warm-alone LOWER bound (0.117) is itself above the tie, so the engine floor is implicated by a lower bound, not just the upper-bound sum" → **REFUTED**

This is the load-bearing error. The lower bound is correctly identified
(warm 0.117 is a true floor on any schedule's wall — you must decode all chunks,
and drain 0.066 < warm hides under it). But it is compared to the **wrong rg
quantity**: the tie is defined on the **WALL** (rg/gzippy wall ≤1.05), and
- warm 0.117 **< rg wall 0.131**
- warm 0.117 **< tie threshold 0.138**

So the lower bound is **inside the tie zone**, not "above the tie." The engine
floor is *not* implicated by the lower bound; the lower bound is *consistent with a
tie.* The brief reached the opposite by silently substituting rg's decode floor
(0.085) for rg's wall. Symmetric-care check (requested last turn): read both ways,
the lower bound says the tie is *reachable*, the upper bound (serialized) is
*uninformative* — neither supports an engine-floor-binds conclusion.

## C4 — "next binder = window-absent marker-engine compute rate, NOT a scheduling fix" → **REFUTED as stated (UPHELD only as 'faster marker decode wouldn't hurt')**

The evidence points the *opposite* way to C4's claim. The recoverable gap is
production 0.177 → ~0.12-0.13 achievable, and that distance is decode↔drain
**overlap** (the oracle proved you can make it *worse* by serializing; production
already does better than the oracle). The decode floor (0.117) is *below* rg's
wall, so a faster marker engine is **neither necessary nor sufficient** for the
wall tie — it would shave the ~1.13× floor gap but the wall is gated by overlap.
C4's own caveat (the 04fda86d marker fast-loop was a TIE) is consistent with
"engine is not the wall lever." So: keep the marker-engine-speed idea as a
secondary T1-flavored lever, but the **anti-scheduling conclusion is inverted** —
this oracle does not refute the scheduling direction; it failed to test it.

---

## Disproof angles

1. **Faithful all-marker warm?** UPHELD. `perfect_overlap_warm` never publishes
   windows or runs the resolve chain; `run_decode_task` only clean-flips when
   `window_map.get(start)` hits (gzip_chunk.rs:790 vs :826), which is impossible
   during warm except `start_bit==0`. `is_speculative_prefetch=true` at partition
   guesses (chunk_fetcher.rs:2154) matches the prefetch path. Marker premium kept.
   *Minor:* warm decodes EVERY non-zero chunk markered, whereas production decodes
   the speculative-MISS chunks clean on the on-demand re-issue (the published
   window now hits at the real offset, 2490-2507). So warm slightly *over*-charges
   per-chunk vs production — making warm a (mildly loose) *upper* bound on the
   all-marker decode cost and a conservative lower bound on the wall. Does not
   change any verdict.
2. **Does warm/drain serialization inflate the wall?** YES — decisively. This is
   the core flaw (see TL;DR). The brief's own C3 lower-bound is the correct way to
   rescue a real conclusion — but it was then *mis-compared* to rg's floor.
3. **Floor-to-floor denominator-matched?** NO. warm (measured wall, incl. pool
   spin-up) vs rg Theoretical-Optimal (arithmetic floor, no spin-up) is apples-to-
   oranges. Matched comparison ≈ warm 0.117 vs rg Real Decode 0.104 = 1.13×.
4. **warm_hit_frac 0.88-0.96 residual?** Acceptable. The misses are offset-0
   stream-start chunk(s) (start_bit==0, clean, no predecessor) which are on the
   critical path regardless. Not material to the verdicts.
5. **Is F1 safe to falsify given the upper bound?** NO — see C1. A config slower
   than production, built by *destroying* decode↔drain overlap, cannot falsify a
   tie claim about the *best* overlap.

---

## SINGLE MOST LOAD-BEARING CORRECTION

The oracle runs **decode-all THEN drain (serialized)** — the opposite of overlap —
so its wall is a pessimistic SUM and comes out *slower than production itself*
(0.183 > 0.177); it therefore cannot falsify F1. The honest lower bound is the
warm phase alone, **0.117s, which is BELOW rg's WALL (0.131s) and inside the tie
threshold (0.138s).** The brief flipped this to "engine floor exceeds the tie" only
by comparing 0.117 to rg's decode **floor (0.085)** instead of rg's **wall**. Read
correctly, the oracle does not refute the scheduling/overlap direction — its
decode-floor number says the tie is *reachable* by better decode↔drain overlap, and
the unresolved question (which this oracle did **not** answer) is whether a real
overlapped schedule can collapse production's 0.177 toward that ~0.117-0.13 floor.
