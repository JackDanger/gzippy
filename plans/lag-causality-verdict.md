# LAG-CAUSALITY VERDICT + RE-SCOPED PLACEMENT DESIGN (placement-rescope leader, 2026-06-07)

Charter: plans/placement-rescope-diagnosis.md. Falsifier (pre-registered BEFORE any run):
plans/lag-causality-falsifier.md. HEAD cb60842d. All numbers from the LOCKED guest harness
(silesia-large 162MB corpus, T8, N=7 interleaved, sha-verified, host RESTORE VERIFIED).

## THE PIVOTAL QUESTION
Is the consumer-prefetcher lag (the ~318ms / cold-re-decode head-of-line stalls) **STRUCTURAL**
(prefetch depth/scheduling/in-order serialization rapidgzip avoids) or an **EFFECT of the 2.38×
slow engine** (consumer can't keep pace because decode is slow)?

## THE PERTURBATION DATA (slow_knob sweep, native clean arm, T8, locked guest)

| combo       | T8 wall | per-blk busy (ms) | **STALL COUNT** | wait.block_fetcher_get (ms) | **wait ÷ per-blk-busy** |
|-------------|---------|-------------------|-----------------|------------------------------|--------------------------|
| F0          | 1.1105  | 0.29963           | **3**           | 369.1                        | 1232                     |
| F50 spin    | 1.2055  | 0.35335 (+18%)    | **3**           | 445.0                        | 1259                     |
| F100 spin   | 1.3704  | 0.44310 (+48%)    | **3**           | 646.3                        | 1459                     |
| F50 sleep   | 1.1827  | 0.32327 (+8%)     | **3**           | 418.5                        | 1295                     |
| F100 sleep  | 1.2340  | 0.36314 (+21%)    | **3**           | 507.5                        | 1398                     |

- STALL COUNT = NOT_RESIDENT non-startup cold-gets (GZIPPY_STALL_RESIDENCY_PROBE,
  has_nearest_le_start=0, conservation_ok=true on EVERY run). Byte-exact (NO DIVERGENCE
  any run). Injection fired (wall rose monotonically with F; per-blk busy +48% at F100 spin).
- Wall response is MONOTONIC under spin (1.11→1.21→1.37) and SURVIVES the frequency-neutral
  SLEEP control (1.11→1.18→1.23) — confirming the clean-loop compute is on the T8 critical
  path (consistent with the prior pre-gate; not the question here).

## VERDICT (advisor-corrected): MIXED — existence STRUCTURAL, magnitude materially ENGINE-COUPLED, separability UNPROVEN

> **CORRECTION (independent disproof advisor, plans/placement-rescope-advisor-verdict.md, folded
> in 2026-06-07).** My first-draft headline "STRUCTURAL / engine-invariant, small residual" was
> OVERSTATED and the re-scoped design rested on a VENDOR SOURCE MISREAD. The corrected verdict
> below is MIXED; the corrected design is in the next section. The advisor's two load-bearing
> corrections (both re-verified first-hand by the leader): (1) the COUNT is a saturated, low-N (3),
> WRONG-DIRECTION (CLAUDE.md rule 3: slow-down can't bound speed-up) proxy — the load-bearing
> structural signal is `has_nearest_le_start=0`, NOT the flat count; the normalized-wait rise
> (+18% spin / +13.5% sleep, ≈24% of cold-wait growth after dividing out per-block cost, surviving
> the sleep control) shows the lag MAGNITUDE is materially engine-coupled. (2) vendor's
> `appendSubchunksToIndexes` (the block-finder insert) runs IN-ORDER on the orchestrator
> (GzipChunkFetcher.hpp:357), NOT pool-side — gzippy ALREADY matches this
> (consumer_append_subchunks_vendor, chunk_fetcher.rs:1750/2790). "Move the insert pool-side" is
> the OPPOSITE of vendor; REFUTED.

### Corrected verdict (what the evidence actually supports)
- **EXISTENCE of the 3 overshoot-tail cold-gets is STRUCTURAL.** Load-bearing signal:
  `has_nearest_le_start=0` at every stall and every cache cap (STEP-0(a) + this run) — NO resident
  chunk even STARTS before the stalled offset, so the parent is never-retained, not evicted-by-drift.
  These 3 stalls exist at F0 and cost real wall NOW (~369ms cold-wait); removing them is worth doing.
  The stale-partition-guess mechanism (overshoot chunk → that partition's guess-prefetch is stale →
  real start never prefetched, [[project_confirmed_offset_prefetch_gap]]) is the structural cause.
- **The COUNT being flat (3) across F0→F100 is NOT strong evidence the lag is engine-invariant.**
  The count is a threshold metric (fires only when drift exceeds the ~2×par≈16-chunk retention
  budget), saturated at N=3, and the slow-direction perturbation cannot bound a speed-up (rule 3) —
  a +48% slowdown need not push drift past 16 chunks, so "count stayed 3" is consistent with BOTH
  structural AND sub-threshold engine-induced. Do NOT elevate count-flatness to the verdict.
- **The lag MAGNITUDE is materially ENGINE-COUPLED.** Normalized wait/blk-busy rose +18.4% (spin)
  and +13.5% (sleep, frequency-neutral — so real, not a turbo artifact). ≈24% of the cold-wait
  growth under slowdown is genuine drift (the consumer arrives "colder" behind a slower engine; each
  cold re-decode reaches further back). The cost the campaign optimizes (wall) is partly engine-set.
- **SEPARABILITY is UNPROVEN.** A faster engine shrinks the 318ms COST while the count stays 3, so
  "placement is a cleanly-separable co-primary that pays before the engine" is only weakly supported.
  Placement's payoff is partly entangled with engine speed. It does NOT fully dissolve into the
  engine (the 3 stalls exist at F0 and cost wall now), but it is NOT cleanly independent either.

### Re-rank consequence
Placement does NOT collapse into the engine (existence is structural; the stalls cost wall today).
But it is NOT the cleanly-separable lever the design-v2 co-primary framing assumed — its MAGNITUDE
payoff is partly engine-coupled, so the honest expected placement-alone win is SMALLER than the full
369ms cold-wait (some of that cold-wait shrinks for free when the engine speeds up). Both levers stay
co-primary; the engine remains the +13.7% residual that survives perfect placement (A.2, unchanged).

---

## (SUPERSEDED FIRST-DRAFT REASONING — kept for the record; read the corrected verdict above)
Per the pre-registered falsifier this is the **STRUCTURAL** branch on the PRIMARY channel and
MIXED on the secondary, resolving to STRUCTURAL:

1. **PRIMARY CHANNEL (stall COUNT) is DEAD FLAT at 3** across a 0→+48% engine slowdown, under
   BOTH spin and sleep. The pre-registered rule: *"STALL COUNT ~flat ⇒ STRUCTURAL — the lag
   (count of cold-gets) is invariant to engine speed; it is set by prefetch depth/scheduling/
   in-order serialization, not by how fast each chunk decodes."* The cold-get points are the
   3 FIXED overshoot-tail boundaries ([[project_confirmed_offset_prefetch_gap]]): a chunk
   whose decode overshot the next partition boundary, so that partition's guess-prefetch is
   stale and the real next-start was never prefetched. **Slowing the engine 1.5× does NOT
   create a single additional eviction/cold-get.** If the lag were engine-INDUCED (consumer
   drifting further past the retention budget as decode slows), the count would CLIMB — it
   does not move at all.

2. **SECONDARY CHANNEL (normalized wait) shows a MILD engine-coupled rise** (1232→1459 spin
   +18%, 1232→1398 sleep +13%) — NOT flat. Mechanism: the raw wait (369→646) rises mostly by
   the pre-registered CONFOUND (each cold re-decode costs ∝ engine speed), but the residual
   after dividing out per-block cost means each of the 3 fixed cold-waits blocks for slightly
   MORE than per-block-busy as the engine slows — the consumer drifts a bit further behind, so
   the cold chunk is "colder" (more of its decode remains to be done on demand). This is a
   small ENGINE-COUPLED component, but it is a magnitude effect on a FIXED set of 3 stalls,
   not a count effect. It cannot manufacture or remove stalls.

**Net:** the lag is **predominantly STRUCTURAL** — its existence and count are fixed by the
scheduling structure (overshoot-tail boundaries that were never prefetched), invariant to
engine speed. There is a small engine-coupled magnitude residual (the consumer drifts a bit
more behind a slower engine), but it modulates the depth of 3 fixed stalls, it does not cause
them. **Placement is a GENUINE SEPARATE faithful-port lever**, not a sub-effect that dissolves
when the engine speeds up. (Even at infinite engine speed the 3 cold-gets would still exist:
the overshoot-tail boundaries are still never prefetched; what shrinks is only their per-stall
depth.)

### Reconciliation with the vendor-pacing-map (plans/vendor-pacing-map.md) — IMPORTANT
The read-only vendor-pacing subagent found rapidgzip's pacing STRUCTURE (prefetch depth ≈2×par,
separate un-evictable in-flight map, join-in-flight, lean off-path consumer post-processing) is
**faithfully ported in gzippy LINE-FOR-LINE** (block_fetcher.rs:737/758/66/536,
chunk_fetcher.rs:528-529/1542/1561). The subagent's source-read JUDGMENT leaned engine-induced.
**The causal perturbation OVERRULES the source-read judgment (exactly why the charter mandated
perturbation over attribution).** The reconciliation: the generic pacing machinery IS faithful
and is NOT the defect — the defect is the SPECIFIC, narrow [[project_confirmed_offset_prefetch_gap]]
mechanism the generic machinery does not cover: an OVERSHOOT chunk that decodes past the next
partition boundary leaves that partition's guess-prefetch stale AND cannot be reused at the
confirmed interior boundary. That is a structural GAP in gzippy's reuse path — present
regardless of engine speed (count flat) — NOT a pace-drift caused by the slow engine. So both
findings agree: the GENERIC structure matches vendor; the SPECIFIC overshoot-reuse path is the
structural lever, and it is engine-speed-invariant (3 stalls at any F).

## RE-SCOPED PLACEMENT DESIGN — faithful port of rapidgzip's OVERSHOOT-BOUNDARY handling

Since the verdict is STRUCTURAL, placement is a separate lever and the design is a FAITHFUL
PORT (not a gzippy-invented scheduler). The re-scope CORRECTS the design-v2 §1.2 framing:

### What STEP-0 already ruled OUT (do not rebuild)
- design-v2 §1.2's "interior-reuse / getIndexedChunk port reads the PARENT from cache()" is
  NOT the fix AS WRITTEN: STEP-0(a) proved has_nearest_le_start=0 at ALL cache caps (1/16/256)
  — there is NO resident parent to reuse at any cache size (the overshoot parent was consumed
  +passed, never-retained). Porting getIndexedChunk's cache-read verbatim reuses a chunk that
  is not there. CONFIRMED again this run: NOT_RESIDENT=4 at every F.

### The faithful structural target — the block-finder SPECULATIVE OFFSET SUPPLY (advisor-corrected)
The 3 stalls are overshoot tails: gzippy's prior chunk decoded ~21KB PAST the next partition
boundary, the guess-prefetch at that partition is stale, and the confirmed real start was never
prefetched. **The insert TIMING/LOCATION is already faithful** — gzippy's
`consumer_append_subchunks_vendor` (chunk_fetcher.rs:1750→2778, insert at
`sc.encoded_offset_bits + sc.encoded_size_bits` :2790) runs IN-ORDER on the consumer/orchestrator,
EXACTLY matching vendor's `appendSubchunksToIndexes` → `m_blockFinder->insert` at
GzipChunkFetcher.hpp:357,371 (the doc at chunk_fetcher.rs:1682-1686 even cites :343-357 and notes
gzippy was deliberately changed to this in-order ordering). So the first-draft "move the insert
pool-side" was WRONG (vendor inserts in-order too; pool-side would DIVERGE).

The real divergence is in the **block-finder's offset SUPPLY to the prefetcher**, NOT the insert.
Vendor `GzipBlockFinder::get(blockIndex)` (GzipBlockFinder.hpp:117-158) returns the EXACT confirmed
offset when `blockIndex < m_blockOffsets.size()` (a real boundary was inserted for it) and only
falls back to the partition GUESS (`partitionIndex * m_spacingInBits`, :138) for not-yet-known
indices. So once a chunk's real end boundary is inserted, vendor's NEXT `get` for the overshot
index returns the CONFIRMED offset and the prefetcher targets the real start — with whatever lead
that index's prefetch has. gzippy's side ([[project_confirmed_offset_prefetch_gap]]): the
FetchingStrategy only ever offers an index at its partition GUESS and never RE-OFFERS an index once
prefetched (`needs_confirmed_offset never fires`), so even after the in-order insert records the
real boundary, the stale guess-prefetch is never superseded by a prefetch at the confirmed offset.

**OPEN MECHANISM QUESTION (the advisor's mandated re-derivation — answer BEFORE any build):** since
vendor ALSO inserts in-order with the same ~1-chunk lead the 3 prior gzippy attempts had, how does
vendor avoid the overshoot cold-get that gzippy hits? Candidate answers to establish from vendor
source: (a) vendor's block finder supplies the confirmed offset on the NEXT get so the prefetcher
naturally re-targets it within its existing 2×par look-ahead window (no extra lead needed — the
fix is making gzippy's strategy re-offer/re-target, not gaining lead); (b) vendor's
getIndexedChunk/unsplitBlocks interior reuse covers the overshoot case via the parent (but STEP-0
ruled the parent not-resident in gzippy — does vendor keep it resident via a different cache/lead
discipline?); (c) vendor simply doesn't overshoot as far (different chunk-finalize/coalesce
behavior). This must be re-derived first-hand; the leader's earlier mechanistic story was tied to
the refuted pool-side premise and is NOT established.

### Two-column port map (CORRECTED — the TIER-3 placement work, advisor-gated before build)
| mechanism | vendor file:line | gzippy today | STATUS / port action |
|---|---|---|---|
| in-order block-finder insert of confirmed boundary | GzipChunkFetcher.hpp:357,371 appendSubchunksToIndexes → m_blockFinder->insert | chunk_fetcher.rs:1750→2778/2790 consumer_append_subchunks_vendor (in-order) | **ALREADY FAITHFUL — do NOT change** (first-draft pool-side move REFUTED) |
| block finder SUPPLIES confirmed offset to prefetcher (re-target the overshot index) | GzipBlockFinder.hpp:117-158 get (returns confirmed offset for known index; guess only for unknown) | gzip_block_finder.rs:147-205 + prefetcher.rs: strategy offers ONLY partition GUESS, never RE-OFFERS once prefetched (needs_confirmed_offset never fires) | **THE candidate lever** — make the strategy re-offer / the prefetcher re-target the overshot index at the confirmed offset once the in-order insert records it. Re-derive the vendor lead mechanism (open question above) FIRST. |

### INJECTION-SYMMETRY CHECK (pre-empts the "asymmetric injection artifact" attack)
The slow_knob injects at the NATIVE clean decode arm (marker_inflate.rs:1307,1546) — reached
by BOTH prefetch-path decode AND on-demand/cold-get decode (all chunks route through
decode_chunk → finish_decode_chunk_impl / marker engine). So the perturbation slows ALL clean
decodes UNIFORMLY (prefetch and on-demand alike). The count-flat result is therefore NOT an
artifact of slowing only the consumer path: if the lag were engine-induced, uniformly slowing
both prefetch and consumer would still let the consumer drift behind a slower prefetch frontier
(producing more evictions) — and the count would climb. It does not. (resumable.rs:1199 is the
gzippy-isal Engine-C control site, dead on native post-fold; the live native sites are the two
marker_inflate ones.)

### CAVEATS (honest — advisor-corrected)
1. The insert TIMING is already faithful (in-order, matching vendor :357); the candidate lever is
   the block-finder offset SUPPLY (re-offer/re-target the overshot index at the confirmed offset).
   But 3 prior consumer-confirmation-prefetch attempts ALL regressed (flooding; ~1-chunk lead) —
   TIER-3 must FIRST re-derive how vendor avoids the overshoot cold-get with the SAME in-order
   ~1-chunk lead (the OPEN MECHANISM QUESTION above), THEN gate any port on the STALL probe (3→≤1)
   + a no-flooding Fulcrum A/B (the prior attempts flooded the pool; the gate must catch that).
2. The engine-coupled magnitude residual (≈24% of cold-wait growth) means placement-alone recovers
   LESS than the full 369ms cold-wait — part of that cost shrinks for free when the engine speeds
   up. The COUNT going to ≤1 (eliminating 2 of 3 cold-gets) is still the dominant placement win.
   Engine stays co-primary (A.2 +13.7% survives perfect placement) — UNCHANGED.
3. Structural change with corruption risk (boundary confirmation must be byte-exact); design
   carefully, do NOT guess (the [[project_confirmed_offset_prefetch_gap]] discipline — 3 reverts).

## RE-RANK / REACHABILITY (advisor-corrected)
Placement does NOT collapse into the engine (the 3 stalls exist at F0 and cost wall today) but is
NOT a cleanly-separable co-primary either (its magnitude payoff is partly engine-coupled,
separability unproven by the slow-direction sweep — rule 3). Both levers stay co-primary. Engine
stays the +13.7% residual that survives perfect placement (A.2). The design-v2 §3 coupled
arithmetic STANDS as an UPPER bound on the placement win: placement → ~0.61s clean-rate operating
point is the optimistic edge (some cold-wait is engine-coupled, so placement-alone may land above
0.61s); placement+engine-at-igzip-class → ~0.54s tie, conditional on the two floors (engine
≤39-45ms/chunk; non-decode floor ≤0.54s, STEP-0 measured ~0.015s+output).

## DELIVERABLE STATUS (advisor-corrected)
1. Verdict: MIXED — existence STRUCTURAL (`has_nearest_le_start=0`), magnitude materially
   ENGINE-COUPLED (normalized wait +18%/+13.5%, survives sleep), separability UNPROVEN. ✓
2. Re-scoped placement design = faithful re-targeting via the block-finder offset SUPPLY
   (GzipBlockFinder.hpp:117-158), NOT pool-side insert (REFUTED — vendor inserts in-order, gzippy
   already matches) and NOT cache-read getIndexedChunk (ruled out by STEP-0). The vendor lead
   mechanism for the overshoot case must be re-derived first-hand BEFORE any TIER-3 port. ✓
3. Independent disproof advisor: DONE → plans/placement-rescope-advisor-verdict.md (REFUTED the
   first-draft headline + design; corrections folded in above). ✓
Then STOP for supervisor ratification. NO placement port, NO engine build started.
