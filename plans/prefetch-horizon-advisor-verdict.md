# PREFETCH-HORIZON vs SATURATION — INDEPENDENT DISPROOF ADVISOR VERDICT (read-only, 2026-06-07)

HEAD 85c67474. Charter: plans/prefetch-horizon-diagnosis.md + plans/prefetch-horizon-falsifier.md.
Tasked to ATTACK the leader's STOP ("NOT horizon; the auto-label is an artifact; redirect to
the ENGINE; no horizon fix attempted") assuming the named campaign bias toward the escape-hatch
"it's the engine, placement is done." All claims below are checked first-hand against the code.

## VERDICT: **REFUTE the STOP AS WRITTEN — the saturation conclusion is contradicted by the leader's own data, and the engine-redirect is an ESCAPE HATCH from a pre-registered HORIZON verdict the leader overrode.** Not yet a confirmed horizon *fix* either; the decisive discriminator was never run. The STOP must NOT be ratified; one specific measurement is owed first.

I could NOT confirm "it's saturation." I found the opposite: every load-bearing datum the leader
cites points AGAINST saturation, and the one cross-check the leader calls "decisive" is confounded.

---

## 1. The leader OVERRODE their own pre-registered verdict map. (Process violation — load-bearing.)

The falsifier (prefetch-horizon-falsifier.md:50-57) pre-registered, BEFORE any run:

> **HORIZON verdict iff (HORIZON_NOT_ENQUEUED) >= ceil(N/2)** (majority had idle capacity AND
> the index was never enqueued).

Observed (findings.md:11-15), N=3 non-startup stalls:
- F0 baseline run1: SAT=1, **HORIZON_NOT_ENQUEUED=2** → 2 ≥ ceil(3/2)=2 → **HORIZON by the map.**
- F0 baseline sweep: SAT=1, **HORIZON_NOT_ENQUEUED=2** → **HORIZON by the map.**
- F100 sleep (freq-neutral slow): SAT=0, **HORIZON_NOT_ENQUEUED=3** → **HORIZON, stronger.**

The pre-registered map returns **HORIZON on every row.** The leader then concluded SATURATION,
overriding the pre-registration with a post-hoc continuous-metric argument (mean_busy). The
falsifier's OWN closing instruction (prefetch-horizon-falsifier.md:91-94) forbids exactly this:

> "**don't let the prior MATCH map foreclose the measurement; the measurement overrules
> attribution.**"

The leader did the inverted, forbidden thing: let the prior source-MATCH *attribution* ("depth is
vendor-identical, so HORIZON is a priori unlikely") **foreclose the measurement** (the buckets that
said HORIZON). Per CLAUDE.md rule 5 ("the verdict is decided by the observation, not the analyst")
this STOP is not a clean read of the pre-registered instrument — it is a re-derivation AROUND it.

This alone means the STOP cannot be ratified as a falsifier outcome. It is an analyst override.

---

## 2. The "decisive" slow-knob cross-check is CONFOUNDED — it cannot discriminate saturation from a horizon gap whose cold-decode latency is engine-set.

The leader's central inference (findings.md:44-49): engine-slow MONOTONICALLY raises busy / lowers
idle_cap ⇒ saturation. **This inference is invalid.** Slowing the per-chunk engine raises `busy`
*globally and trivially* — every chunk that is mid-decode at any sampled instant now takes longer,
so the instantaneous count of mid-decode workers rises **regardless of whether the stall is
saturation-caused.** Both hypotheses predict busy↑ under engine-slow:
- SATURATION predicts busy↑ (no free worker, more queueing).
- HORIZON-with-engine-latency ALSO predicts busy↑ (the cold on-demand decode + all other decodes
  each take longer; more decode-time in flight at every instant).

A test that both hypotheses pass is not a discriminator. The busy-rise is a property of decode
*duration*, not of *stall causation*. The leader read a confound as the decisive falsifier.

Worse — the pre-registered SAT prediction FAILED for the leader's conclusion. The falsifier
(prefetch-horizon-falsifier.md:71-75) predicted "a slower engine MUST increase busy/**decrease
idle_capacity** at stalls (**more SAT-classified**)." The actual **SAT** bucket went **1 → 1 → 0**
(flat, then FALLING under the freq-neutral slow); the bucket that ROSE was HORIZON_NOT_ENQUEUED
(2 → 3). By the leader's own stated logic ("saturation grows with engine-slow; a horizon-bound
count is flat regardless of engine speed"), SAT flat/falling + HZ_NE rising is the **HORIZON**
signature. The leader substituted the confounded continuous metric for the discrete SAT bucket
precisely because the discrete bucket pointed the wrong way for the conclusion.

---

## 3. idle_capacity > 0 at EVERY stall is the literal disproof of saturation — and the code proves the on-demand decode starts immediately on a free worker.

Saturation, as the leader DEFINED it (prefetch-horizon-falsifier.md:42): `SAT : idle_capacity == 0`
(all workers busy, **no slot to start it on**). In the data, `mean_idle_cap` is **2.00, 2.67, 2.33,
1.67** — it **never reaches 0**, not even under a 2× engine slowdown. There was always a free worker.

This is not a soft signal; the code makes it dispositive. At the cold-get `None` stall
(chunk_fetcher.rs:1357), the consumer submits the on-demand decode and it runs **on the pool now**:

- `block_fetcher.get_with_prefetch(next_block_offset, |_| submit_decode_to_pool(thread_pool, …), …)`
  — chunk_fetcher.rs:1450-1468. With `idle_capacity ≈ 2 > 0`, that submit lands on a parked/unspawned
  slot and **starts immediately** (thread_pool.rs:259-265 defines `busy = spawned − idle`,
  `idle_capacity = idle + (cap − spawned)`; lazy-spawn slots are real capacity).

So at the stall there IS a free worker AND the needed decode IS dispatched onto it without queueing.
That is the exact negation of "all workers busy decoding the slow engine ⇒ no free slot to start the
marginal index." The cost the consumer pays is the **decode latency of one cold chunk that should
have been ready**, not a wait for a worker to free up. Saturation is absent at the stall instant.

---

## 4. SOURCE-CHECK of the load-bearing claim "the confirmed offset is never a dispatch candidate at any depth" — IMPRECISE, and it does NOT carry the weight the leader puts on it.

First-hand (gzip_block_finder.rs:176-207, `get`):
- index `< block_offsets.len()` (confirmed) → returns the **exact confirmed offset** (:180-181).
- index beyond the confirmed frontier → returns a partition-spacing **GUESS** (:184-206).

The strategy generates the upcoming *indexes* (sequential extrapolation, prefetcher.rs:128/140/261);
`prefetch_new_blocks` resolves each via `lookup_block_offset(index) = block_finder.get(idx)`
(chunk_fetcher.rs:1229-1234, block_fetcher.rs:835). So:

- The marginal index **IS** a strategy candidate (sequential extrapolation reaches it).
- It is dispatched at the **GUESS** offset while its predecessor is still in flight, and at the
  **CONFIRMED** offset once the predecessor decodes and inserts it (the in-order
  `consumer_append_subchunks_vendor` → block_finder `insert`). The confirmed offset is therefore
  *not* "never a candidate" — it becomes one as soon as confirmation propagates.

The accurate statement is narrower: **confirmation lags dispatch.** That is a *timing/scheduling*
property, not "the offset can never be offered." And it does NOT establish saturation — it is
orthogonal to whether a free worker existed (one did).

Crucially, the bucket reads `HORIZON_NOT_ENQUEUED`, **not** `HORIZON_ENQUEUED_NOT_DONE`
(prefetch-horizon-falsifier.md:44-47; counter at chunk_fetcher.rs:1413-1416). `enqueued` is true iff
some in-flight key K satisfies `K ≤ decode_start < K + partition_span`. The result is `!enqueued`:
**no in-flight chunk — not even a guess — covered decode_start at the stall.** The residency probe
agrees: NOT_RESIDENT=4, `has_nearest_le_start=0` (findings.md:17-20) — "no resident chunk even
starts ≤ the stalled offset." So the covering chunk was **neither in flight nor resident** while a
free worker sat idle. That is a placement/horizon/retention failure with headroom present — the
exact regime the prior offset-supply refutation (placement-port-advisor-verdict.md) did **not**
cover. The prior 3 failures were `in-flight-not-done` (lead too short); THIS is `not-in-flight-at-
all-despite-idle-capacity`. Different failure mode, different fix. Folding it into the refuted
offset-supply lever is the conflation that powers the escape hatch.

---

## 5. Where the engine genuinely IS implicated — and why that still does NOT earn the leader's STOP.

Honesty cuts both ways. There is a coherent engine story, but it is **not** the leader's "saturation":
the cache and depth are vendor-identical (`m_cache = max(16, P)`, `m_prefetchCache = 2P`,
chunk_fetcher.rs:520-529 ↔ BlockFetcher.hpp:181-183; depth/ramp/pump byte-identical per the
source-cite). Under identical sizing, the residency probe's "never-retained" finding most plausibly
reads: the prefetcher races ahead, decodes a chunk covering decode_start, and the **engine-induced
consumer lag (~318ms)** lets the cache overrun and EVICT it before the lagging consumer arrives →
cold get. A faster engine shortens that lag and the cold-gets fall. So the engine is a **real
co-lever** — via lag→eviction, NOT via "no free worker."

But that mechanism **refutes the STOP's conclusion**, it does not support it:
- "placement has no SEPARATE horizon headroom" (findings.md:75-79) is **false**: idle workers exist
  at every stall and are being spent decoding even-further-ahead chunks that will themselves be
  evicted (wasted speculative work), instead of the about-to-be-consumed chunk being protected.
- The cache-pollution stop (block_fetcher.rs:899-915) protects *to-be-prefetched* blocks from
  mutual eviction; it does **NOT** protect the consumer's imminent `decode_start` chunk from being
  evicted by over-eager deeper prefetch. Whether vendor protects the consumer-imminent chunk under
  the same lag is an OPEN, faithful-port question (anti-overrun / retention), distinct from both
  offset-supply and from raw inner-loop speed. The leader did not look; the data can't yet say.

---

## 6. The unrun discriminator (this is what the STOP skipped, and it is cheap).

A single occupancy snapshot cannot tell "covering chunk NEVER dispatched" from "dispatched →
decoded → EVICTED before arrival." Both yield NOT_RESIDENT + !enqueued. The verdict hinges entirely
on which it is:
- **never dispatched (with idle capacity)** ⇒ a genuine scheduling/horizon gap; faithful lever live.
- **dispatched-decoded-evicted** ⇒ retention/anti-overrun question (still placement, still faithful,
  but a different fix), with the engine shortening the lag as a co-lever.

Instrument owed (counters, byte-exact, same gating): for each stall, did ANY task covering
decode_start get submitted earlier in the run, and if so, at what wall-time was it evicted relative
to this consumer arrival? Plus the parked-idle vs unspawned split (the leader reports only aggregate
idle_capacity; a parked worker is unambiguously immediate, an unspawned slot has spawn latency —
report them separately). And gather **N ≫ 3** (the leader concedes findings.md:88-94 the discrete
split is "fragile at this N"; the honest move is to lower split_chunk_size or aggregate dozens of
stalls, not to fall back on a confounded continuous metric).

---

## ANSWERS TO THE CHARGED QUESTIONS

- **Is the leader's reading of the occupancy data sound?** No. idle_capacity > 0 at every stall is
  the leader's own definition of NOT-saturation; the leader explained away a real horizon/dispatch
  signal that their pre-registered map scored as HORIZON.
- **Does idle_cap>0 prove a dispatch failure rather than saturation?** It disproves saturation
  (saturation ≡ idle_capacity==0, never observed). It establishes that a free worker existed and the
  covering decode was not running on it — a dispatch/retention failure. Whether the precise sub-cause
  is "never dispatched" or "evicted" is the unrun discriminator (§6). "busy is high" does NOT
  dominate "idle_cap>0": 5.3/8 busy is irrelevant when 2/8 are free and the needed work isn't on them.
- **Is the slow-knob a valid Method-3 application?** No — confounded (§2). Engine-slow raises busy
  for a reason orthogonal to stall causation; the test cannot separate the hypotheses, and the
  pre-registered SAT-rises prediction actually FAILED (SAT fell, HZ_NE rose).
- **Is the marginal confirmed offset truly never a candidate at any depth?** No (§4). It is a
  candidate (guess pre-confirmation, confirmed offset post-confirmation). The real property is
  "confirmation lags dispatch," which is a scheduling fact, not a saturation fact.
- **Is the engine-redirect earned or an escape hatch?** As written, an escape hatch: it rests on a
  saturation claim the data contradicts and on overriding the pre-registered HORIZON verdict. The
  engine is nonetheless a real co-lever via lag→eviction — but that is a CORRECTED mechanism that
  reopens a placement/retention question, not a closure of it.
- **Is N=3 too small?** Yes, decisively — and compounded by using the wrong (confounded)
  discriminator. Gather N≫3 AND run the never-dispatched-vs-evicted discriminator before any redirect.

## BOTTOM LINE
Do NOT ratify the STOP and do NOT start the engine inline-ASM build on this basis. The leader's
saturation verdict is contradicted by its own data (idle_capacity>0 everywhere; SAT bucket flat/
falling under engine-slow; pre-registered map scored HORIZON 3/3) and rests on a confounded
cross-check. The engine is a genuine co-lever, but only via engine-induced consumer lag → cache
overrun → eviction — a mechanism that **reopens** the placement/retention sub-question (does vendor
protect the consumer-imminent chunk from over-eager prefetch eviction?), it does not close it. Run
the one cheap discriminator in §6 (never-dispatched vs decoded-then-evicted, parked-vs-unspawned
idle, N≫3) FIRST. Only then is the saturation-vs-horizon question actually decided rather than
attributed.
