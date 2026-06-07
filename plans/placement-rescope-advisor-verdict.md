# PLACEMENT-RESCOPE — INDEPENDENT ADVERSARIAL VERDICT (read-only disproof advisor, 2026-06-07)

HEAD cb60842d. Posture: assume the leader is wrong; re-derived every claim first-hand from
`/tmp/lag-causality-run1/*`, `src/decompress/parallel/{slow_knob,stall_residency,chunk_fetcher}.rs`,
and `vendor/.../GzipChunkFetcher.hpp`. Attacking `plans/lag-causality-verdict.md`.

**TL;DR — the STRUCTURAL *existence* claim survives weakly; the STRUCTURAL *confidence* and the
"small residual" framing are overstated; and the headline DESIGN ("move the block-finder insert
POOL-side, faithful to vendor") is REFUTED by vendor source — vendor inserts on the in-order
orchestrator, exactly where gzippy already does. The recommended port is a DIVERGENCE, not a port.**

---

## POINT 1 — Is the STALL COUNT the right lag proxy? → CORROBORATE-WITH-CAVEATS (count is near-unfalsifiable here)

The probe reports are byte-identical across all five combos (verified, not trusting prose):
`total=4 startup=1 NOT_RESIDENT=4 has_nearest_le_start=0 conservation_ok=true` at F0/F50/F100 ×
{spin,sleep}. So the count truly is dead flat at 3 non-startup.

But "flat count ⇒ structural" is much weaker than the verdict claims, for three independent reasons:

1. **The count is a THRESHOLD metric, saturated in the tested regime.** `classify_stall` fires only
   in the `None` branch at `chunk_fetcher.rs:1357-1399` — when the partition lookup yields no usable
   prefetch. The 3 non-startup stalls are the corpus's fixed overshoot-tail boundaries. An
   *engine-induced* extra stall requires the consumer to drift past the `2*par`≈16-chunk retention
   budget so a *completed* prefetch gets evicted before consumption. A +48% per-block slowdown (the
   max tested) need not push drift past 16 chunks. So "count stayed 3" is equally consistent with
   "structural" AND "engine-induced but sub-threshold drift." The count cannot discriminate in this
   range — it is a saturated, low-N (3) instrument.

2. **The perturbation direction is mismatched to the hypothesis (CLAUDE.md PROCESS rule 3).** The
   pivotal question is whether *speeding* the engine dissolves the lag. The leader answered it by
   *slowing* the engine. Rule 3: "slow-down slope ≠ speed-up ceiling." Worse, the slow-knob slows
   **decode work only** (the inner Huffman loop), not the consumer's fixed in-order overhead (Arc
   clone, drain/recycle scans, window publish — vendor-pacing-map deltas #1/#3/#4). Slowing the pool's
   decode while the consumer's fixed overhead stays constant makes the consumer keep pace *easier*
   (its fixed overhead becomes a smaller share of the cycle), i.e. the perturbation pushes drift in
   the WRONG direction to manufacture evictions. A "count climbs" result was therefore nearly
   impossible by construction → a passing "flat" result is near-unfalsifiable, not strong evidence.

3. **The genuinely structural signal is `has_nearest_le_start=0`, not the count.** That says at every
   stall NO resident chunk even *starts* before the offset — the parent was never retained, not
   evicted-by-drift. That is a real layout/scheduling property and is the load-bearing evidence for
   "these 3 stalls exist independent of decode speed." The leader leaned on the wrong channel (count)
   when its own STEP-0 channel (`has_nearest_le_start`) carries the argument better.

**Verdict P1:** the *existence* of 3 overshoot stalls is plausibly structural (via `has_nearest_le_start=0`
+ the stale-partition-guess mechanism), but the COUNT-flatness the verdict elevates to "PRIMARY
CHANNEL ... DEAD FLAT ⇒ STRUCTURAL" is a saturated, wrong-direction, low-N proxy and should not carry
the weight assigned to it.

## POINT 2 — Does the normalized-wait rise refute "structural"? → CORROBORATE the rise is real; the leader UNDERSTATES it

Re-derived the normalizer arithmetic from the verdict's own table (and confirmed the raw wait against
the Fulcrum trace: `F0_spin.log: wait.block_fetcher_get 4 363.69ms 369.09ms 35.6% WAIT` — count and
369ms both match the probe, so the numbers are Fulcrum-sourced, not hand-rolled):

| combo | raw wait | ÷per-blk-busy | vs F0 |
|---|---|---|---|
| F0 | 369.1 | 1232 | — |
| F50 spin | 445.0 | 1259 | +2.2% |
| F100 spin | 646.3 | 1459 | **+18.4%** |
| F50 sleep | 418.5 | 1295 | +5.1% |
| F100 sleep | 507.5 | 1398 | **+13.5%** |

Decomposition at F100 spin: raw wait grew **+75%** while per-block decode cost grew **+48%**. So of the
total cold-wait growth, ≈48/75 is the mechanical confound (each cold decode costs more) and the
remaining **≈24%** of the growth is the engine-coupled DRIFT residual (1.75/1.48 = +18.2%). It survives
the frequency-neutral sleep control (+13.5%), so it is not a turbo artifact — it is real engine
coupling of the lag *magnitude*. The normalizer (per-block-busy) is a reasonable divisor and, if
anything, is *generous to the leader*: the cold-get re-decode spans ~410 block-equivalents (123ms /
0.3ms), so the residual reflects the re-decode reaching *further back* as the engine slows — i.e. the
consumer arriving "colder." That is exactly the drift mechanism.

**Verdict P2:** the normalized channel is the more sensitive instrument and it points engine-coupled.
Calling it a "small magnitude residual" is not defensible: ~a quarter of the cold-wait growth under
slowdown is genuine drift. Since the *cost* (wall), not the *count*, is what the campaign optimizes,
the thing that matters is partly engine-induced. **MIXED, leaning more engine-coupled than the verdict
admits.**

## POINT 3 — Injection symmetry / confound direction → CORROBORATE (symmetry holds) but it cuts AGAINST the leader

Checked the injection sites: `marker_inflate.rs:1307` and `:1546` (the `CONTAINS_MARKERS=false` clean
arm) and `resumable.rs:1199` (the dead isal control). All chunk decodes — prefetch-path and
on-demand/cold-get alike — route through the same `submit_decode_to_pool` → clean decode arm, so the
knob slows BOTH uniformly. The leader's INJECTION-SYMMETRY CHECK is correct: this is NOT an
"asymmetric injection" artifact. So the specific attack "it slows only the consumer" is REFUTED.

**But the symmetry is double-edged and the leader stops one step short.** Uniform decode slowdown means
the pool's fill rate AND the consumer's cold-get both slow by ~F, while the consumer's *fixed*
non-decode in-order overhead does not. The net effect of uniform slowdown is to *reduce* the
frontier-vs-consumer gap, not enlarge it — confirming P1's point that the test is biased against
producing engine-induced evictions. The clean symmetry the leader cites as a strength is exactly why
the count could not move.

## POINT 4 — Is the re-scoped design a FAITHFUL PORT? → **REFUTE.** Vendor inserts in-order; the design's premise is a source misread.

This is the decisive finding. The verdict (lines 107-118) proposes: *"validated boundaries are appended
from the POOL-SIDE postProcessChunk (GzipChunkFetcher.hpp:553-575 queueChunkForPostProcessing →
appendSubchunksToIndexes), which runs as each chunk completes, NOT at the in-order consumer frontier."*

Read directly from `vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`:

- `processNextChunk()` (the in-order/orchestrator path, invoked at `:219`) calls
  `postProcessChunk(...)` at **`:343`** then `appendSubchunksToIndexes(...)` at **`:357`**. Both run on
  the orchestrator. The header even annotates `:242-244`: *"As this is run on the orchestrator thread,
  it should not be compute-intensive."*
- `appendSubchunksToIndexes` is defined at `:365` and is called from exactly ONE site — `:357`, on the
  orchestrator. It is the block-finder/index insert.
- `queueChunkForPostProcessing` (`:554-582`) is a SEPARATE method. What it submits to the pool
  (`submitTaskWithHighPriority` at `:579-581`) is `chunkData->applyWindow(...)` — the heavy marker
  replacement — and it inserts the **last window** into the window map on the **main thread** (`:559`).
  It does **not** call `appendSubchunksToIndexes`.

So the leader **conflated two distinct vendor methods**: it cited the line range of
`queueChunkForPostProcessing` (553-575, which pool-submits `applyWindow`) AS IF it were
`appendSubchunksToIndexes` (357, the orchestrator/in-order block-finder insert). They are different
functions on different threads.

gzippy's existing code already matches vendor correctly: `consumer_append_subchunks_vendor`
(`chunk_fetcher.rs:2778`, calling `block_finder.insert(chunk_end_bit)` at `:2790`) runs **in-order in
the consumer loop** at `:1750`, and the doc at `:1682-1686` explicitly cites
`GzipChunkFetcher.hpp:343-357` and notes gzippy was *deliberately changed* to append AFTER
post-process to MATCH vendor's in-order ordering.

**Therefore the design's headline action — "move the confirmed-boundary insert to the pool-side" — is
the OPPOSITE of what vendor does.** Vendor confirms boundaries on the in-order orchestrator, with the
SAME ~1-chunk lead the leader calls "DEAD per prior attempts." Implementing the leader's design would
make gzippy UNLIKE rapidgzip — a direct violation of the governing transliteration guardrail
([[feedback_bias_guardrails]]: "a change that makes gzippy UNLIKE rapidgzip is forbidden even if it
helps the wall").

Corollary: the leader's mechanistic story for HOW vendor avoids the overshoot stall is therefore also
unestablished. Since vendor ALSO inserts in-order, its lead cannot come from insert *location*. It must
come from the **block finder's speculative offset supply** (`GzipBlockFinder.hpp`) — the second row of
the leader's own port map (`prefetcher targets confirmed offset ... needs_confirmed_offset never
fires`). That row points at the real mechanism; the PRIMARY recommended action contradicts it.

## POINT 5 — Can the verdict FLIP to engine-induced? → A credible partial-dissolve path EXISTS

Construct it from the surviving evidence: (a) the lag *magnitude* is demonstrably engine-coupled (P2:
~24% of cold-wait growth is drift, surviving the sleep control); (b) rule 3 says the slow-down sweep
cannot bound the speed-up; (c) a faster engine decodes the overshoot chunk sooner (its confirmed
boundary lands earlier relative to the consumer) AND makes each cold re-decode cheaper. The COUNT can
stay 3 while the 318ms COST shrinks substantially with engine speed. So "placement pays *before* the
engine, as a cleanly separable co-primary lever" is only weakly supported — placement's payoff is
partly entangled with the engine, more than the "small residual" framing admits. This is precisely the
engine-leaning conclusion the vendor-pacing-map reached on source-read; the leader claimed to "overrule"
it via the count channel, but that channel is the weak/saturated one (P1). The overrule is not earned.

This does NOT fully flip placement into the engine — the 3 stalls exist at F0 and cost real wall now,
so removing them is worth doing. But it does refute "separable co-primary, count is engine-invariant
so placement is independent."

---

## BOTTOM LINE

**Is the STRUCTURAL verdict safe for the supervisor to act on? — NO, not as written.** The narrow,
defensible claim is: *three overshoot-tail cold-gets exist at the current operating point, their parent
is never retained (`has_nearest_le_start=0`), and eliminating them is worth ~the 369ms cold-wait.* That
much is supported. But the verdict's headline — "lag is STRUCTURAL / engine-invariant because the count
is DEAD FLAT, with only a small engine residual" — overstates confidence: the count is a saturated,
low-N, wrong-direction (rule-3) proxy, and the verdict's own secondary channel shows ~a quarter of the
lag-magnitude growth is engine-coupled drift surviving the frequency control. The honest verdict is
MIXED → existence structural, magnitude materially engine-coupled, separability unproven.

**Is the re-scoped design a faithful port worth authorizing for TIER-3? — NO as written; YES only as a
re-derivation investigation.** The design's load-bearing premise (vendor appends confirmed boundaries
pool-side) is false: `appendSubchunksToIndexes` runs on the in-order orchestrator at
`GzipChunkFetcher.hpp:357`; `queueChunkForPostProcessing` (553-582) pool-submits only `applyWindow`.
gzippy already mirrors vendor (in-order `consumer_append_subchunks_vendor`). "Move the insert pool-side"
would DIVERGE from vendor. Authorize TIER-3 ONLY to first correctly establish how vendor supplies the
prefetch *lead offset* for the overshoot case (block-finder speculative supply — the leader's port-map
row 2, `GzipBlockFinder.hpp:105-158` / `needs_confirmed_offset never fires`), gated on the STALL probe
(3→≤1) + a no-flooding Fulcrum A/B. Do NOT authorize the insert-relocation.

**The ONE thing most likely to break it:** the design rests on a vendor source misread —
`appendSubchunksToIndexes` (orchestrator/in-order, `:357`) was conflated with
`queueChunkForPostProcessing` (pool `applyWindow`, `:554-582`). Vendor confirms boundaries in-order
with the same ~1-chunk lead the leader already declared dead, so the proposed "faithful pool-side port"
is neither faithful nor obviously a source of new lead. Re-derive the vendor lead mechanism before any
build.
