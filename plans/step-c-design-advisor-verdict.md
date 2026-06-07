# STEP-C TIER-1 design v2 — independent disproof advisor verdict (READ-ONLY)

Independent, adversarial, disproof-driven Opus advisor. Target: `plans/tier1-design-v2.md`.
HEAD b8a38e64, branch reimplement-isa-l. Every source claim below was re-derived
FIRST-HAND from the files (line numbers cited), not taken from the design's own prose.
Default posture: try to BREAK each load-bearing claim; endorse only what survives.

---

## TARGET 1 — REACHABILITY ARITHMETIC (§3): **CORROBORATE-WITH-CAVEATS** (the caveat is load-bearing on the *conclusion*)

### What survives disproof (give the design its due)
- **The coupled-not-additive framing is sound and does NOT double-count.** §3 does not
  multiply two independent speedups; it converts the operating point (marker-rate →
  clean-rate via placement) and then shrinks the clean-rate (engine). I tried to find a
  stacked optimistic removal and could not — the design explicitly refuses additive
  stacking ("Neither alone reaches it"). Target 1(c) **fails**: there is no hidden
  double-count.
- **The placement-alone number (0.61s) is on FIRMER ground than the step-a advisor's
  Oracle-P critique.** v2 does NOT reuse Oracle-P's rescaled floor (the "3 unproven
  removals" target). It uses the A.2 clean-only oracle's **measured** 0.6134s wall —
  every chunk forced clean with the real publish chain running (step-a2 verdict,
  CORROBORATE-WITH-CAVEATS, 3.47σ). So "placement-perfect ⇒ ~0.61s" is anchored to a
  measurement, not an extrapolation. Target 1(b) (is 1.36 validly reused?) **largely
  fails**: A.2 independently re-derived the clean-only ramp at **1.356** (step-a2 Target
  3), so reusing 1.36 at the clean operating point is empirical, not assumed. And §3
  correctly *abandons* the ramp at the engine-maxed step (it does not multiply 0.26s by
  anything) — so 1(b) finds no abuse there either.

### Where it breaks — the 0.61 → 0.54 "TIE" step (Target 1(a) and 1(d) SURVIVE as real defects)
The fragile move is: "decode stops binding ⇒ the wall re-binds on **rapidgzip's OWN
floor ~0.54s**." This silently assumes **gzippy's non-decode pipeline floor EQUALS
rapidgzip's 0.54s.** That equality is unproven and there is measured evidence against it:

1. **gzippy carries consumer-serial bookkeeping that rapidgzip does off-critical-path.**
   `[[project_confirmed_offset_prefetch_gap]]` ROOT-CAUSE section measured
   `window_publish_marker/get_last_window_vec + dispatch_post_process +
   queue_prefetched_postproc ≈ 225ms` of in-order consumer work, and names that
   rapidgzip's consumer runs apply_window/window-publish OFF the in-order path. 225ms of
   serial consumer work does **not** shrink when the engine speeds up — it is a floor
   term that survives both levers.
2. **A.2's own trace shows the consumer block is ~0.5s.** step-a2 Target 2:
   `wait.block_fetcher_get = 0.497s` on the single consumer thread at the clean operating
   point. The A.2 advisor charitably called it "normal pipeline fill, present in
   rapidgzip too" — but it was measured WITH free pre-seeded windows (zero window-arrival
   latency). In a real run the consumer block is ≥ that, and any serial fraction of it is
   a hard floor.

So when the engine reaches 39ms/chunk (decode optimal ≈ 39×39/8 ≈ 0.19s), the wall
re-binds on **gzippy's** non-decode floor, which the evidence puts materially **above**
0.54s — not on rapidgzip's 0.54s. Target 1(d) (a term re-binds before 0.54s) is therefore
not merely a "residual risk" — there is a concrete, *already-measured* mechanism (the
~225ms consumer-serial term + the 0.497s consumer block) that the §3 headline treats as
shared-and-equal when it is gzippy-specific-and-higher. The design DOES list this under
residual risk #2/#3, but it is buried as "the ramp may not hold," not surfaced as the
thing that converts the headline "TIE" into "≥0.54s, plausibly 0.54–0.62s, TIE only at
the optimistic edge."

**Verdict 1: CORROBORATE-WITH-CAVEATS.** The arithmetic's *structure* (coupled, no
double-count, measured anchors) is honest and survives. Its *conclusion* ("⇒ ~0.54s =
the TIE") rests on an unproven floor-equality contradicted by gzippy's own traces. Read
§3 as proving "placement+engine-maxed ⇒ **≥0.54s**," not "= 0.54s."

---

## TARGET 2 — FAITHFUL-PORT-vs-INNOVATION for placement (§1): **CORROBORATE-WITH-CAVEATS, leaning REFUTE on the "just wire dead code" framing**

### What survives (the design's factual corrections are RIGHT)
- **The accept predicate IS a pure range check** — verified `chunk_data.rs:466-468`
  (`encoded_offset_bits ≤ expected ≤ max_acceptable_start_bit`) and the consumer call
  site `chunk_fetcher.rs:1289` (`arc.matches_encoded_offset(decode_start)` with **no**
  `&& max == decode_start` clause). The design's "the memory `== decode_start` is STALE"
  is **correct**; the blocker is not the predicate.
- **The interior byte offset DOES exist.** `Subchunk` (`chunk_data.rs:34-46`) records
  `encoded_offset_bits`, `decoded_offset`, `decoded_size` — so a parent chunk that
  overshot CAN, in principle, emit `[confirmed, end]`: the data needed for the emit is
  present, not absent. This refutes the strongest form of the memory's "data-layout makes
  reuse impossible" worry **for multi-subchunk overshoot parents**.

### Where the "faithful, just activate dead structures" claim breaks (three concrete defects)
1. **The two named dead functions are in the WRONG coordinate system for gzippy's
   consumer.** §1.2 step 1 says "calls `block_map.find_data_offset(offset)`." But
   `find_data_offset` (`block_map.rs:135`) is keyed by **decoded BYTE offset**
   (`binary_search_by(|&(_, d)| d.cmp(&data_offset))`), serving vendor's
   `get(decodedOffset)` consumer. gzippy's `consumer_loop` requests in **encoded BIT**
   space — it tracks `furthest_decoded_bit` (`chunk_fetcher.rs:1029`, an encoded bit
   position) and `next_block_offset`, never a running decoded-byte output position used as
   a key. So `find_data_offset(next_block_offset)` is a coordinate-type mismatch. Vendor
   reaches `getIndexedChunk` *through* its decoded-offset consumer; gzippy's consumer is
   encoded-keyed. Faithfully porting the **entry** (`get(decodedOffset)`) implies changing
   gzippy's consumer request model to decoded-offset — that is a **consumer redesign
   (architecture)**, which the faithful-port mandate forbids ([[feedback_bias_guardrails]]).
   The alternative (build a new encoded-range-*containing* index) is **not** "activating
   dead code." `get_encoded_offset` (`block_map.rs:152`) only does *exact* match, not
   containing, so it does not fill the gap either. (`find_data_offset`/`get_encoded_offset`
   confirmed production-dead — only test callers, `block_map.rs:278-323`.)
2. **`unsplit_blocks` is populated ONLY when `subchunks.len() > 1`**
   (`chunk_fetcher.rs:2736-2758`). A single-subchunk overshoot parent writes NO
   `unsplit_blocks` entry, so the encoded-keyed reuse path the design proposes has no map
   entry for exactly the common overshoot case. The design's "the map is already
   populated" is only true for multi-subchunk parents.
3. **DEEPEST — getIndexedChunk presupposes the parent is still CACHED, which the cited
   memory MEASURED to be false and left as an UNANSWERED discriminator.**
   `[[project_confirmed_offset_prefetch_gap]]` ROOT-CAUSE: gzippy's consumer lags its own
   prefetcher by **~318ms**; the containing chunk "P decoded early, finished, **EVICTED**
   … ⇒ consumer COLD RE-DECODES." Cache is small: `cache_capacity = max(16, pool_size)`,
   `prefetch_capacity = pool_size*2` = 16 at T8 (`chunk_fetcher.rs:527-528`). getIndexedChunk
   fetches the parent from `cache()`; if the 318ms lag evicted it, the lookup misses and
   falls through to the same cold decode. The memory's **pre-registered discriminator** —
   "is the containing partition chunk in-flight/cached at the gzippy stall? YES ⇒ fix =
   interior reuse; NO ⇒ deeper gap (gzippy discards what rapidgzip keeps)" — was
   explicitly **NOT answered** ("Do NOT patch until confirmed"). The design asserts the
   root cause is "now precisely located" and that interior-EMIT is the sole missing piece;
   that is **more certainty than the memory established.** If the answer is NO (consistent
   with the measured eviction), there is a chicken-and-egg: reuse needs the parent cached,
   but the parent is evicted *because* the consumer lags, and the lag is what reuse is
   meant to cure.

### Why this is a caveat, not a flat REFUTE
The design's §1.3 **pre-registers the right falsifier** ("if the port does NOT drop the
stall count AND the wall does NOT move toward 0.66s, the defect is deeper than
getIndexedChunk — re-open; do not patch-and-pray") and keeps a byte-exact tie even on a
partial wall move (CLAUDE.md rule 7a). That is methodologically correct and protects
against shipping a non-fix. So the *plan* is safe. What is overstated is the *confidence*:
the port recipe names the wrong-coordinate function, omits the single-subchunk gap, and
assumes the parent-cached precondition the memory flagged and never confirmed.

**Verdict 2: CORROBORATE-WITH-CAVEATS (leaning REFUTE on "small faithful diff that just
wires dead code").** The factual corrections are right and the falsifier is sound, but
"activates dead-but-built structures, NO redesign" is too glib: the dead structures are
decoded-offset-keyed for a consumer gzippy doesn't have, the map is conditionally
populated, and the whole mechanism is hostage to a cache-residency precondition that was
*measured to fail* and left as an open discriminator. **Answer that discriminator (is the
parent cached at the stall?) BEFORE building the port** — it is the difference between
"faithful wiring" and "deeper consumer-throughput gap."

---

## TARGET 3 — CAN E1–E4 HIT IGZIP-CLASS IN PURE-RUST+ASM (§2): **CORROBORATE-WITH-CAVEATS**

### What survives
- **E1 and E3 are genuinely not-yet-done — re-attempting is legitimate.** Verified the
  ring is still u16 (`marker_inflate.rs:290` `output_ring: Box<[u16; RING_SIZE]>`) with a
  u16→u8 narrow at drain (`:82` `v as u8`, `push_clean_u8` `:42`). The post-flip ring
  store was never made u8-direct — matches `[[project_faithful_unified_decoder_over_perf]]`
  ("u8-direct post-flip path was NEVER ported"). The 2-/3-literal chain exists
  (`marker_inflate.rs:~1799-1841`) but emits per-literal into the u16 ring, not one packed
  wide store — so E3 (packed multi-literal store) is genuinely unbuilt. Calling ca52389
  non-binding (Target 3(b)) is **fair**: that regression predates both the PRELOAD loop
  and the u16-ring fold; re-measure is sound.
- **§2.3's isolation bench with a GUEST-RATIO ISA-L positive control is a SOUND engine
  gate** and directly answers CLAUDE.md rule 3 ("slow-down slope ≠ speed-up ceiling") and
  the step-a2 RULE-3 flag. The design correctly insists the control read the **guest
  ratio** (≈2× scalar single-thread), NOT the illegitimate Frankensystem absolutes
  337/720 — that is exactly the step-a2 caveat, honored.

### Where the risk is real (and the design says so, to its credit)
- **The 2.1× pure-decoder ceiling is a hard physical headwind.** rapidgzip's own bench:
  ISA-L is 2.1× its BEST pure decoder single-thread. gzippy must close a **2.38×**
  per-chunk clean gap (92.7 → ~39ms) in pure-Rust+ASM. The design's residual risk #1 is
  **honest, not wishful**: it states plainly that if E1–E4 plateau near the pure ceiling
  the clean rate lands ~50–60ms, decode-bound wall ~0.34–0.40s, total ~0.54–0.60s — "a
  NARROW miss, possibly a TIE within spread, possibly +5–10%." That is the correct
  framing; it does not claim the tie, it claims the bench decides.
- **CAVEAT (3(c)): the §2.3 bench is an ENGINE gate, NOT a WALL gate.** Benching one
  known-window clean chunk is the *right* way to isolate engine compute (free windows +
  zero marker-resolve is the POINT of isolation, not a bias here as it was for the wall
  claim). But the design then feeds the bench number "into the §3 model" to project the
  T8 wall — so the wall projection inherits **all of Target 1's floor uncertainty.** A
  passing isolation bench proves the *engine* can reach igzip-class; it does NOT prove the
  *wall* ties, because the wall re-binds on the non-decode floor Target 1 flags. The
  design should state the §2.3 PASS criterion as "engine reaches X" and keep the wall
  conclusion explicitly contingent on the Target-1 floor being ≤0.54s.

**Verdict 3: CORROBORATE-WITH-CAVEATS.** The techniques are real and unbuilt (re-attempt
justified), the gate is methodologically correct with the right positive control and the
right guest ratio, and the design is admirably honest that the engine front is HIGH-RISK
with a plausible narrow-miss even on success. The caveat: a passing engine bench is
necessary but not sufficient for the wall tie — it is hostage to the §3 non-decode floor.

---

## TL;DR (6 lines + bottom line)

1. **§3 reachability: CORROBORATE-WITH-CAVEATS** — coupled-not-additive is sound and
   double-count-free; placement-alone 0.61s is *measured* (A.2), not rescaled; BUT the
   0.61→0.54 "TIE" step assumes gzippy's non-decode floor = rg's 0.54s, contradicted by a
   *measured* ~225ms consumer-serial term + 0.497s consumer block. Read it as **≥0.54s**.
2. **§1 placement port: CORROBORATE-WITH-CAVEATS (leaning REFUTE on "just wire dead
   code")** — the range-check correction is right and `Subchunk.decoded_offset` exists,
   but `find_data_offset` is decoded-byte-keyed for a consumer gzippy doesn't have
   (gzippy is encoded-bit-keyed), `unsplit_blocks` is only built for subchunks>1, and the
   whole thing presupposes a parent-cached precondition the cited memory *measured to fail*
   and left as an unanswered discriminator.
3. **§2 engine E1–E4: CORROBORATE-WITH-CAVEATS** — E1/E3 genuinely unbuilt (u16 ring,
   per-literal emit) so re-attempt is justified; §2.3's guest-ratio ISA-L control is a
   sound ENGINE gate; but the 2.1× pure ceiling is a hard headwind (design's own narrow-
   miss math) and the bench gates the engine, not the wall.

**BOTTOM LINE — ratify into TIER-3?** Ratify the **method**, not the **conclusion**. The
plan is correctly sequenced and falsifier-gated (placement first → measure → engine-bench-
before-build), so it is safe to authorize TIER-3 *as a measurement-gated investigation*.
Do NOT ratify the headline claim "the 1.0× TIE is reachable by this design" — that rests
on two unproven floor-equalities. **The ONE thing most likely to break it:** gzippy's
non-decode consumer-serial floor (~225ms measured) is structurally higher than
rapidgzip's, so even placement+engine-maxed re-binds *above* 0.54s — a residual neither
lever touches. **Add a third pre-registered measurement before/with TIER-3:** decompose
the placement-perfect 0.61s consumer block into decode-wait vs serial-bookkeeping (the
memory's own owed discriminator) to set the true non-decode floor. If that floor is
>0.54s, the engine front is chasing a tie the consumer structurally forbids — and that is
a supervisor-level finding (revisit the bar or move the serial work off the in-order
path), not a silent miss.
