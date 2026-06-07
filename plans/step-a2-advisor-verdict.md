# STEP-A.2 CLEAN-ONLY ENGINE ORACLE — adversarial disproof verdict (READ-ONLY)

Independent disproof-driven Opus advisor. HEAD e89006b0, branch reimplement-isa-l.
Every number below was re-derived FIRST-HAND from the artifacts in
`/tmp/a2-artifacts/` (host-guest.log, st_seed.err, st_normal.err, trace_seed_T8.json,
trace_normal_T8.json) with my own line-tolerant parser (`/tmp/a2-artifacts/parse.py`),
NOT taken from the leader's summary. Trace `ts` unit confirmed = **microseconds**
(trace isal busy 3.6145s == BlockFetcher `decodeBlock` stat 3.6196s).

## VERDICT: **CORROBORATE-WITH-CAVEATS**

The narrow claim **survives a genuine disproof attempt**: forcing every chunk down
the clean (window-present) decode path, with real Huffman compute preserved and the
production consumer/publish chain running, yields a T8 wall of **0.6134s median vs
rapidgzip 0.5396s = +13.7%**, a margin **≫ inter-run spread (3.47σ)** — so the engine
gap is REAL and SURVIVES placement-perfection. class-C/engine is a genuine co-primary
lever, not negligible. This is the cleanest engine signal in the campaign and it is
**conservative** (the oracle's defects all bias the wall DOWNWARD, in gzippy's favour).

But two of the leader's framings overstate what was measured. Both are caveats, not
refutations — they make the engine-residual conclusion *stronger*, not weaker.

---

## Target 1 — DID IT ISOLATE THE ENGINE? **YES (clean pass).**

Re-derived from `trace_seed_T8.json` + `st_seed.err`:
- `worker.decode_mode` instants: **{clean: 39, window_absent: 0}**. Every chunk clean.
- `window_seeded=39`, `finished_no_flip=0`, `fused_lut=0`, `finish_decode=39`,
  `Slow-path decode ok=0`, `Speculation failure modes` all 0, `BlockFinder coordinator
  spawns=0`, `Body-fail forensics count=0`, bootstrap `post_flip_u16_bytes=0`.
  `scan_candidate` / `block_body` / `block_header` spans are **entirely ABSENT** from
  the seed trace (normal: scan_candidate 5.94s/n=35, block_body 4.44s/n=7720). There
  is **zero marker decode and zero marker resolution**. SEED replay hits=38 misses=0
  (38 seeded + chunk-0 zero-window = 39). sha e114dd2b… byte-exact every iteration.
- The work is REAL and representative, not a shortcut: clean decode busy 3.6145s
  produces all 503 MB (sha-OK) at 139 MB/s/core vs the marker path's 105 MB/s — the
  clean path is faster *per byte* exactly because it resolves back-refs against the
  known window instead of emitting/resolving u16 markers. No byte is skipped.

No hidden marker work; no decode shortcut. Engine isolation is genuine.

## Target 2 — PUBLISH-CHAIN PRESERVED, or DEGENERATE like Oracle-C? **Not degenerate, but free-window — a LOWER bound.**

This is the load-bearing caveat. Oracle-C's defect (decode≈0 ⇒ windows free ⇒
L_resolve collapses to 162µs) is genuinely AVOIDED here: decode is real (92.7ms/chunk),
`consumer.window_publish_clean` runs all **39** times. So the publish path is not
collapsed.

BUT pre-seeding ALL 38 predecessor windows up front removes **window-arrival latency**.
`seed_window_for` is consulted whenever `window_map.get()==None` (chunk_fetcher.rs:2184),
so every worker ahead of the consumer frontier gets its predecessor window *for free
from a prior run* instead of waiting for predecessor N−1 to finish. First-hand from the
trace: `pool.pick.wait` (worker idle) = 0.406s over 8 threads while decode busy = 3.616s
→ workers ~90% busy, **no window stall** (BlockFetcher: Pool Efficiency 91.19%, Prefetch
Stall by BlockFinder 0). The 0.497s `wait.block_fetcher_get` is the single consumer
thread blocking on the in-order chunk future (normal pipeline fill, present in rapidgzip
too) — NOT a worker window stall.

Consequence (the leader states this correctly in the falsifier's DISPROOF section): a
**real** all-clean parallel pipeline is non-physical — worker N's clean window IS worker
N−1's output, so insisting on clean windows without a prior run serializes the pipeline.
The seed breaks that dependency, so **0.6134 is a LOWER BOUND** on any realistic engine-
bound wall, not "the" engine ceiling. It also pays **no marker-resolve pass** (gzippy
publish 2.2ms total vs rapidgzip's ~0.113s apply-window pass). So gzippy clean-only runs
at a **coordination ADVANTAGE** over rapidgzip's real run and STILL loses by 13.7%. The
contamination is downward → it can only UNDERSTATE the engine gap → the conclusion is
conservative and survives. ✔ (Caveat: report 0.61 as "≥ engine floor under free windows
+ zero marker-resolve," not "= the engine ceiling.")

## Target 3 — RAMP CONSISTENT (the STEP-A error)? **YES.**

Headline is **wall-to-wall, both measured** (0.6134 median vs 0.5396 median) — no
floor-vs-wall mismatch (the STEP-A defect). First-hand: seed decode busy 3.616s/8 =
0.4518s == BlockFetcher "Theoretical Optimal" 0.4525s; wall/optimal ramp = **1.356**.
rapidgzip ramp (prior trace, LPT makespan 0.385) = 1.40. Both ~1.36–1.40; the small
asymmetry FAVOURS gzippy (lower ramp), so it does not manufacture the gap. The 1.36 is
descriptive, not load-bearing — the headline does not depend on rescaling. ✔

## Target 4 — STRUCTURAL COUPLING FINDING. **SOUND; it SUPPORTS co-primary.**

The claim (clean decode requires BOTH a real boundary AND its window; production
dispatch uses spacing GUESSES whose WindowMap keys are misaligned — first naive snapshot
capture hit 0% / diverged) is directly visible in the instrument: `seedable_chunk_starts()`
pre-seeds `block_finder.insert()` with real boundaries (chunk_fetcher.rs:498-503) AND
`seed_window_for` supplies windows. You cannot isolate the engine without ALSO solving
placement. This is correct and it *reinforces* co-primary: the two levers are structurally
coupled, both required for the tie. It does not undermine the conclusion. (It is also a
second reason 0.61 is a best-case bound — engine isolation is only reachable atop perfect
placement.)

## Target 5 — IS +13.7% REAL given the spreads? **YES, not a TIE.**

clean-only sd 1.9% → 0.0117s; rapidgzip sd 3.3% → 0.0178s; combined σ = 0.0213s.
Δ = 0.0738s = **3.47σ** (6.3× the clean sd). Min-vs-min gap = 0.0549s — the ranges do
not overlap (worst-case gzippy still slower than best-case rapidgzip). Δ ≫ spread ⇒
**NOT a TIE** (CLAUDE.md). ✔

---

## The one overstatement to STRIKE: "2.38× clean-rate gap confirmed AT THE WALL"

The 92.7ms/chunk clean busy is real and corroborates the prior 91ms (365/4) derivation.
But "2.38× confirmed at the wall" conflates two baselines:
- **2.38×** = gzippy clean per-chunk (92.7ms) vs rapidgzip **clean** per-chunk (39ms).
  A clean-to-clean microbenchmark gap. rapidgzip rarely decodes clean — its real run is
  mostly marker decode.
- **At the wall / busy level** the honest first-hand gap is gzippy-clean busy (3.616s)
  vs rapidgzip **full** decode busy (2.994s, prior trace) = **1.21×**, giving the 1.137×
  (13.7%) wall gap.

So what is "confirmed at the wall" is a **13.7% engine residual under gzippy-favourable
conditions** (free windows, zero marker-resolve), NOT a 2.38× wall gap. The 2.38× is a
separate, real per-chunk clean-rate fact. Recommend the status line read: *"engine
residual ≥13.7% survives all-clean; per-chunk clean rate 92.7ms vs rapidgzip-clean 39ms
(2.38×) — but rapidgzip's wall pays marker, not clean, rate, so the wall gap is 13.7%,
not 2.38×."* This makes the engine-is-residual claim MORE defensible (gzippy loses even
when advantaged), not less.

## Minor note (non-disqualifying)
The traced seed run's inner drive span (0.5065s) is below the benched min (0.5896s);
the ~0.09s difference is process startup + mmap + output-write + sha that the bench
end-to-end wall includes and the trace's drive-span excludes. rapidgzip's 0.5396 is the
same end-to-end CLI wall, so the comparison stays wall-to-wall fair; the decode-only
portion is even more engine-dominated.

---

## BOTTOM LINE
- **The narrow claim survives disproof.** Engine compute is genuinely isolated (zero
  marker work, sha-exact), the wall gap is statistically real (3.47σ), wall-to-wall and
  ramp-consistent, and the oracle is NOT degenerate like Oracle-C.
- **It is a conservative LOWER bound:** free windows + zero marker-resolve bias the wall
  down, so the true engine-bound wall is ≥0.61s. This STRENGTHENS engine-is-residual.
- **Strike "2.38× at the wall"** → the wall confirms a 13.7% residual; 2.38× is the
  per-chunk clean-to-clean rate.
- **co-primary CONFIRMED:** placement & engine are both real, both ~13% gaps, and the
  oracle proves they are structurally coupled (engine isolation requires confirmed-
  boundary dispatch). The honest sequence remains: port the scheduler (placement) FIRST,
  re-measure, expect a residual that then requires the inner-loop engine to reach 1.0×.
