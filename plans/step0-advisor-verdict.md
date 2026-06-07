# STEP-0 DISCRIMINATORS — INDEPENDENT DISPROOF ADVISOR VERDICT (READ-ONLY)

Independent, adversarial, disproof-driven Opus advisor. Targets: the two TIER-3 STEP-0
results in `plans/orchestrator-status.md` (a) parent-cached-at-stall = NO, and
(b) consumer-block decompose, non-decode floor ≈0.015s ≤ 0.54s = PASS. HEAD ~a34c9c61,
branch reimplement-isa-l. Every claim below re-derived first-hand from source (file:line),
not from the result prose. Posture: try to BREAK each load-bearing claim.

---

## (a) PARENT-CACHED-AT-STALL — VERDICT: **CORROBORATE-WITH-CAVEATS**
### The headline counter (CONTAINING_CACHED) is an ARTIFACT, but the conclusion (NO) survives on an INDEPENDENT signal.

**The prime suspect is CONFIRMED.** The probe records each resident chunk as
`(encoded_offset_bits, max_acceptable_start_bit)` (`chunk_fetcher.rs:1369,1372`) and tests
strict-interior containment with `enc <= decode_start && decode_start <= max &&
decode_start > enc` (`stall_residency.rs:68-74`). But `[encoded_offset_bits,
max_acceptable_start_bit]` is **NOT the chunk's decoded-content span — it is the
speculative START-TOLERANCE window** (`matches_encoded_offset`, `chunk_data.rs:466-469`).
The decoded bytes BEGIN at `max_acceptable_start_bit` (doc `chunk_data.rs:143-145`,
`decode_origin_bit` set `= decode_start` at decode, `chunk_fetcher.rs:3281`); the content
span is `[max_acceptable_start_bit, encoded_offset_bits + encoded_size_bits]`.

Consequences, both fatal to the CONTAINING_CACHED bucket:
- A **re-anchored / consumed** chunk has `set_encoded_offset` collapse `enc == max == offset`
  (`chunk_data.rs:1445-1447`). Then `decode_start > enc && decode_start <= max` is
  `decode_start > enc && decode_start <= enc` — **impossible by construction.** This is
  exactly the trace's `cached_ranges [235000298,235000298]` (enc==max).
- A genuine **overshoot parent** P (the chunk whose CONTENT spans the stalled offset) has
  `decode_start ∈ (P.max, P.end)`, i.e. `decode_start > P.max`, which fails `decode_start <=
  max`. So even if P were resident, it would be misclassified NOT_RESIDENT.

⇒ **CONTAINING_CACHED can never fire for a true containing parent.** Its `0` is vacuous, and
the pre-registered "≥50% NOT_RESIDENT ⇒ re-scope" rule is rigged toward NO. The positive
control (cap=1 → NOT_RESIDENT 4→9; cap=256 → stays 4 at 0%) validates **stall COUNT**
response to cache size, NOT that the residency bucket can observe a resident parent — both
caps can only ever produce NOT_RESIDENT, so the control does not exonerate the predicate.

**Why the NO conclusion nevertheless SURVIVES.** Two channels with REAL discriminating power
both say "nothing at/below the stalled offset is resident":
1. The auxiliary `nearest_le_start` (`stall_residency.rs:109-119`) is the **correct** test:
   the largest resident `encoded_offset_bits ≤ decode_start`. A containing parent MUST have
   `enc < decode_start` (it starts before the offset it contains), so it WOULD be found here
   even though CONTAINING_CACHED misses it. The trace reports `nearest_le_start:-1` on EVERY
   non-startup stall ⇒ no resident chunk (either cache) starts at/below the offset.
2. CONTAINING_IN_FLIGHT is keyed (`k <= decode_start < k+span`, `stall_residency.rs:85-87`),
   independent of the `[enc,max]` bug; it read 0.

The probe reads BOTH `prefetch_cache_contents_sorted` and `cache_contents_sorted` plus
`prefetching_keys` (`chunk_fetcher.rs:1367-1374`; `block_fetcher.rs:573,583,595`), so the
snapshot is complete. The design ACTION gated on (a) — "build getIndexedChunk interior reuse
that reads the parent from `cache()`" — requires precisely a resident chunk with `enc ≤
decode_start` spanning the offset; `nearest_le_start:-1` proves that precondition is absent.
**So re-scoping placement away from cache-resident interior reuse is SAFE.** This also
matches the bilateral finding in `project_confirmed_offset_prefetch_gap` (consumer lags its
prefetcher ~318ms; resident chunks are all AHEAD; the containing parent is gone/never-kept).

**Single most likely thing that breaks it:** the verdict rests entirely on the auxiliary
`nearest_le_start`, which was reported `-1` only at the DEFAULT cap. The "not a capacity
problem" sub-claim (cap=256 → still 0% resident) is supported only by the vacuous bucket. To
distinguish "parent was decoded then evicted" (capacity — a big cache would retain it) from
"parent was never retained/decoded" (consumer-pace/overshoot — a big cache cannot help), the
team must report **`nearest_le_start` at cap=256**: if still `-1`, the consumer-pace re-scope
is clean; if it flips ≥0 there, capacity was masking a retainable parent and the re-scope
framing needs adjustment. Either way the as-specified cache-read port is not the fix — the
re-scope direction holds.

---

## (b) CONSUMER-BLOCK DECOMPOSE — VERDICT: **CORROBORATE-WITH-CAVEATS**
### The floor conclusion (non-decode serial ≪ 0.54s ⇒ continue) is SAFE and robust across operating points.

**The WAIT/SERIAL classification is HONEST in the dominant term, and the source dissolves the
~225ms fear.** The advisor's worry was that `apply_window`/window-publish is consumer-serial
CPU rapidgzip runs off-path. Source shows gzippy ALSO runs `applyWindow` on the POOL: the
consumer SUBMITS post-process and then BLOCKS on a future, and that block is wrapped as
`wait.future_recv` NESTED inside `consumer.dispatch_post_process`
(`chunk_fetcher.rs:1671-1713`). `wait.future_recv` is in `WAIT_SPANS`
(`consumer_block_decompose.py:27-32`), so the apply-window blocking is correctly counted as
DECODE-WAIT (shrinks with engine speed), not as floor. The marker-resolution work is
therefore NOT a serial floor term — the 225ms was `wait.future_recv` + `dispatch_recv`
mis-bucketed as serial, as the result claims.

**Robust across operating points (the strongest part of (b)).** The clean-only oracle
(`GZIPPY_SEED_WINDOWS`) removes markers BY CONSTRUCTION, so it could be accused of hiding the
floor. But the NORMAL (full-marker) trace independently reports SERIAL = 0.069s — still ≪
0.54s — because the marker work is on the pool (wait.future_recv), not consumer CPU. So the
floor conclusion does NOT depend on placement achieving all-clean (which (a) just showed is
not yet in hand): even at today's marker-heavy operating point the consumer's own serial
bookkeeping is small. `get_last_window_vec`/`window_publish` are ~39 × 32 KiB copies
(`chunk_data.rs:527-565`; `chunk_fetcher.rs:1520,1618,1639`), measured 2.2ms clean — trivial.

**Self-time math is sound.** Stack-based child subtraction, no double-count
(`consumer_block_decompose.py:60-89`); RAII `SpanGuard` spans are well-nested by construction;
conservation gap 0.46%. The SLOW positive control passed the pre-registered falsifier
(decode +100% → DECODE-WAIT 0.49→5.10s, SERIAL flat 0.013→0.045s), which is the right
direction and confirms the buckets don't leak decode time into serial.

**Caveats (none flip the gate):**
1. **`/dev/null` excludes production output-write serial work — the "36× margin" is
   illusory for a real sink.** `ttp.rx_recv_block` and the writev path are real consumer
   work; the interim mktemp run measured `consumer.writev ≈ 0.245–0.267s`. To `/dev/null`
   that vanishes. The production floor is `~0.015s + output-write`, plausibly 0.05–0.27s
   depending on sink. This STILL passes 0.54s, and the comparison is fair **only if
   rapidgzip's 0.54s was also measured to `/dev/null`** (output write is on both critical
   paths). Flag: real margin is materially smaller than 36×, and gzippy-specific writev cost
   (segmented iovec vs contiguous) is a separate, off-critical-path-able consumer lever not
   measured here — it does not gate the engine build.
2. **Minor: `ttp.rx_recv_block` (`block_fetcher.rs:247-268`) wraps the pump loop**, so its
   inclusive time includes `pump_prefetch()` = `prefetch_new_blocks` CPU. If that CPU is not
   in child spans, it is bucketed as WAIT and under-counts SERIAL. Magnitude is bounded
   (≲ tens of ms over a 0.5s wait) and cannot flip a 0.015–0.07s vs 0.54s gate.

**Single most likely thing that breaks it:** not the gate (it passes with room even adding
production writev), but the FRAMING — the 0.015s "floor" is the clean-only, /dev/null,
all-marker-work-on-pool number; the production-relevant floor is `~serial + output-write`,
and any future move of apply_window OFF the pool back onto the consumer would re-inflate it.
As measured today, the consumer does NOT structurally forbid the tie.

---

## BOTTOM LINE — are BOTH conclusions safe for the supervisor to act on?

- **(a) "re-scope placement away from interior-reuse": YES, safe.** The headline
  CONTAINING_CACHED counter is a measurement artifact (it tests the start-tolerance window,
  which is `enc==max` for re-anchored chunks and pre-content for speculative ones, so it can
  never see a containing parent), and the positive control validated only stall count. BUT
  the conclusion is independently established by `nearest_le_start:-1` (the correct test) and
  CONTAINING_IN_FLIGHT=0: the parent the as-specified cache-read port needs is not resident.
  Re-scope to consumer-pace / parent-retention is correct. **Owed before building the
  re-scope:** report `nearest_le_start` at cap=256 to separate "evicted (capacity)" from
  "never retained (pace)" — it changes the shape of the re-scoped fix, not the decision to
  re-scope.
- **(b) "non-decode floor passes, continue": YES, safe.** Confirmed by source that
  apply_window runs on the pool (wait.future_recv), so the ~225ms was mis-bucketed; the
  serial floor is small at BOTH the clean (0.015s) and normal (0.069s) operating points.
  Caveat: this is a `/dev/null` floor — add production output-write before quoting margin,
  and ensure rapidgzip's 0.54s is the same-sink comparison. The engine front may proceed to
  its §2.3 isolation bench; the wall tie remains contingent on the engine reaching
  igzip-class AND a same-sink floor staying ≤0.54s.

**Net:** both STEP-0 verdicts are directionally correct and safe to act on. The one process
correction: stop citing CONTAINING_CACHED=0 / "0% resident" as the evidence for (a) — it is
vacuous; cite `nearest_le_start:-1` (+ in-flight=0) instead, and close the cap=256
`nearest_le_start` gap. For (b), re-state the floor with production output-write included so
the margin quoted to the supervisor is the real one.
