# PLACEMENT PORT — INDEPENDENT DISPROOF ADVISOR VERDICT (read-only, synchronous, 2026-06-07)

HEAD e52b0fc2. Charter: plans/placement-port-authorization.md. The advisor was tasked to
ATTACK the leader's STOP recommendation (assume the leader is biased toward an escape-hatch
"it's already faithful / it's the engine" conclusion) and hunt for the faithful, distinct,
lead-lengthening vendor mechanism the leader is giving up on.

## VERDICT: **GATE FAIL — the leader's STOP is CORRECT, and for a STRONGER reason than the leader gave.**

The advisor could not find a faithful, distinct, lead-lengthening mechanism. The STOP is
earned by source, not an escape hatch. But the recorded *reason* is corrected (hardened).

### 1. FINDING A (vendor reaches the same overshoot cold-get; absorbs it via fill/pace) — CONFIRMED, leader UNDER-claimed it.
Verified first-hand:
- Block finder byte-faithful: `gzip_block_finder.rs:176-207` ↔ `GzipBlockFinder.hpp:120-158`
  (confirmed idx→confirmed offset `:180-182`↔`:130-132`; unconfirmed→spacing guess).
- Consumer getBlock faithful: `chunk_fetcher.rs:1268-1452` ↔ `GzipChunkFetcher.hpp:591-687`
  — try partition-keyed prefetch, accept on `matches_encoded_offset` (`:1306`↔`:646-648`),
  else fall through to on-demand get AT THE CONFIRMED OFFSET via `get_with_prefetch`
  (`:1431`↔`:654`).
- The confirmed offset is born at the in-order insert in BOTH:
  `consumer_append_subchunks_vendor` (`chunk_fetcher.rs:2788-2795`) ↔ `appendSubchunksToIndexes`
  (`GzipChunkFetcher.hpp:357-375`). There is NO earlier confirmation/supply path in vendor —
  not in the block finder, not in the prefetcher, not in any adaptive ramp. Vendor's long lead
  comes ENTIRELY from the 2·P guess-prefetch horizon (`BlockFetcher.hpp:182,474`); the overshoot
  cold-get is a rare guess-vs-confirmed-mismatch FALLBACK, pumped during the wait
  (`:312-316`, mirrored at `chunk_fetcher.rs:1289-1301`). **No distinct lead-lengthening
  block-finder mechanism to port.**

### 2. FINDING B — the leader's "live divergence" was PARTLY CONFABULATED; the truth makes the GATE FAIL HARDER.
The leader claimed gzippy re-maps every offset to the partition offset
(`block_fetcher.rs:785-787,862-863`) so a confirmed offset never becomes a prefetch candidate,
and that `needs_confirmed_offset` never fires. **False against the code:**
- `needs_confirmed_offset` does NOT exist in `src/` (zero grep hits).
- `block_fetcher.rs:784-790` pushes BOTH the real/confirmed `off` AND the partition offset
  (partition secondary) — exactly vendor `BlockFetcher.hpp:485-490`. `:862-863` is the
  dedup-skip identical to vendor `:537-538`.
- The actual prefetch submit (`block_fetcher.rs:945` `submit_for(prefetch_block_offset,…)`)
  uses `lookup_block_offset(index) = block_finder.get(index)` (`:835`,
  `chunk_fetcher.rs:1229-1234`) = the CONFIRMED offset for confirmed indexes, faithful to
  vendor `:479`. gzippy does NOT collapse to partition.

⇒ gzippy ALREADY re-targets the overshot index at its confirmed offset at all three sites.
**The authorization's premise — "gzippy never re-targets an overshot index at its CONFIRMED
offset; rapidgzip does" — is FACTUALLY WRONG.** The lever targets a NON-DIVERGENCE.

### 3. Gate questions resolved
1. HOW vendor avoids the overshoot cold-get: it does NOT avoid it via a distinct mechanism —
   it ABSORBS it via the 2·P guess-prefetch depth + pump-during-wait; the cold-get fallthrough
   exists in vendor too (`:646-654`). gzippy mirrors all of this.
2. WHY the prior 3 failed: they supplied the confirmed offset at the frontier (≤1-chunk lead);
   Attempt 3 proved that is "in-flight-not-done" (submitted EXACT correct offset, fetcher_get
   UNCHANGED T8 449ms / WORSE T16 303→936ms).
3. Is the new port distinct in its LOAD-BEARING constraint: **NO.** It targets a non-divergence
   (already faithful), AND its confirmed offset is still born at the frontier (same ≤1-chunk
   lead Attempt 3 disproved). Attempt #4 with the same load-bearing property — and nothing to
   actually change.

**GATE FAIL. Do not code attempt #4.** Corrected recorded reason: NOT
"distinct-in-impl-but-inherits-1-chunk-lead" — but **"the authorized offset-supply lever
targets a non-divergence; gzippy is already a faithful port at `gzip_block_finder.rs:180-182`,
`chunk_fetcher.rs:1306`+`:1431`, `block_fetcher.rs:945`."**

### 4. Anti-escape-hatch guardrails the leader must obey with this STOP
- **Do NOT bundle this into "therefore it's the engine, placement is done."** That WOULD be the
  named bias. The 318ms lag (structural vs engine-induced) is still open.
- **The genuinely-distinct structural sub-question the re-derivation did NOT close:** the stalls
  are recorded "all decode_NOT_STARTED" ([[project_confirmed_offset_prefetch_gap]]). Decode
  not-even-started ⇒ either (a) workers saturated (engine), or (b) the GUESS-prefetch for that
  index was never dispatched DEEP ENOUGH AHEAD (prefetch-horizon/scheduling — STRUCTURAL, and
  DISTINCT from offset supply). That is the real faithful-port question; it is exactly what the
  pre-registered slow-knob perturbation in placement-rescope-diagnosis.md targets. **Run THAT
  next — not the offset-supply port, not the engine build.**
- **Cheap loose-end on FINDING A:** verify whether gzippy's consumer ever enters the
  `!block_is_confirmed` branch (`chunk_fetcher.rs:1113-1129`) in production at T8 — vendor's
  invariant (`GzipChunkFetcher.hpp:411-427`) forbids the consumer from reaching an unconfirmed
  index. If it fires, it's a real (opposite-direction) divergence but still cannot lengthen the
  lead (does not reopen the gate); if it never fires, FINDING A is fully airtight.

## BOTTOM LINE
GATE FAIL. The STOP stands. Strengthen its stated reason (non-divergence, not
1-chunk-lead-inheritance). Redirect the next step to the decode_NOT_STARTED / prefetch-horizon
perturbation, NOT to the offset-supply port and NOT to the engine concession.
