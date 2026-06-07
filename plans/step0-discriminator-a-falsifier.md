# STEP-0 DISCRIMINATOR (a) — PARENT-CACHED-AT-STALL — PRE-REGISTERED FALSIFIER

Status: PRE-REGISTERED BEFORE ANY RUN. Leader (TIER-3 STEP-0), HEAD b8a38e64.
Charter: plans/tier3-step0-authorization.md §THIS-TURN (a). Design ref: tier1-design-v2.md §1.2 precond 3.
This is the literal UNANSWERED discriminator from [[project_confirmed_offset_prefetch_gap]]
("BILATERAL STRUCTURAL FINDING" + "ROOT CAUSE 2026-06-04"): the memory MEASURED a ~318ms
consumer lag → eviction and left the question "is the containing partition chunk in-flight/
cached at the gzippy stall?" explicitly NOT answered ("Do NOT patch until confirmed").

## THE QUESTION
When the consumer stalls at a CONFIRMED offset and falls through to a synchronous COLD
get (`get_with_prefetch`, chunk_fetcher.rs:1374, the `None` branch at :1342 = no usable
partition-keyed prefetch), is the chunk that CONTAINS that confirmed offset in its decoded
range — i.e. the OVERSHOOT PARENT whose `[encoded_offset_bits, max_acceptable_start_bit]`
range covers `decode_start` — currently IN-FLIGHT or CACHED (recoverable), or EVICTED/absent?

- YES (containing parent resident) ⇒ interior-reuse (getIndexedChunk port) IS the fix:
  the data is present, the consumer just lacks a path to emit `[confirmed, end]` from the
  parent's interior. Placement-port direction VALID.
- NO (containing parent evicted/absent) ⇒ the gap is cache-residency / consumer-pace
  (gzippy DISCARDS what rapidgzip keeps in-flight). The placement direction must be
  RE-SCOPED: the fix is upstream of interior reuse (the consumer can't reuse what is gone).
  Surface to supervisor as a re-scope, NOT a "wire dead code" port.

## INSTRUMENT (byte-exact, env-gated; counters only, OFF==identity)
New env GZIPPY_STALL_RESIDENCY_PROBE=1. Hooked ONLY at the cold-get `None` branch
(chunk_fetcher.rs:1342, the genuine head-of-line stall — the same site the prior
breakthrough probe used). At each such stall, BEFORE calling get_with_prefetch, classify
the residency of the CONTAINING chunk for `decode_start`:
  1. Scan block_fetcher's prefetch_cache + main cache + in-flight `prefetching` set.
  2. For each resident/in-flight chunk whose Arc is available, test
     `arc.matches_encoded_offset(decode_start)` (the TRUE range check, chunk_data.rs:461-469)
     i.e. encoded_offset_bits ≤ decode_start ≤ max_acceptable_start_bit, AND
     decode_start STRICTLY INSIDE (decode_start > encoded_offset_bits — an interior, not
     the exact start, which would already have been accepted at :1289).
  3. Classify each stall into exactly one bucket:
     - CONTAINING_IN_FLIGHT  (an in-flight prefetch's eventual range will contain it — best
       case for join; note in-flight Arc not yet decoded so range may be unknowable →
       record by KEY containment: an in-flight key ≤ decode_start whose partition spans it)
     - CONTAINING_CACHED     (a decoded chunk in prefetch_cache/cache contains decode_start)
     - NOT_RESIDENT          (no resident/in-flight chunk contains decode_start = EVICTED/absent)
  4. Emit per-stall trace line + maintain three atomic counters. Print the tally at the
     consumer-loop summary (alongside the existing PREFETCH_REJECT_BY_GUARD / fetcher_get
     prints, chunk_fetcher.rs:959).

Probe is read-only over the fetcher state (locks, snapshots, no mutation of decode flow).
OFF path is an inlined early-return guard (enabled() OnceLock<bool>) ⇒ OFF==identity proven
by dual-sha (env-set vs env-unset both 028bd002…cb410f / guest e114dd2b…).

## INSTRUMENT VALIDATION (CLAUDE.md rule 4 — must pass BEFORE the result counts)
- POSITIVE CONTROL: shrink the prefetch_cache capacity via an env knob
  (GZIPPY_PREFETCH_CACHE_CAP=N, default = today's 2*pool) to a TINY value (e.g. 1). With a
  pathologically small cache the containing parent MUST be evicted before the lagging
  consumer reaches it ⇒ NOT_RESIDENT count must RISE toward the stall count. If shrinking
  the cache does NOT increase NOT_RESIDENT, the probe is not actually observing residency —
  FIX before trusting. (Conversely, a LARGE cache should DROP NOT_RESIDENT — both directions
  checked = the probe tracks the variable it claims to.)
- SELF/NEGATIVE: probe ON vs OFF on the same input = byte-identical output (dual-sha). The
  counters must sum to the stall count (every cold-get stall classified into exactly one
  bucket — conservation check; a leak ⇒ probe miscounts).
- The stall set must REPRODUCE the memory's signature on silesia-large T8 (≈4 head-of-line
  cold-get stalls, the prior breakthrough count) — if the cold-get stall count is wildly
  different, the routing/operating-point changed and the probe is measuring something else.

## PRE-REGISTERED VERDICT RULE
On the LOCKED GUEST harness (silesia-large, T8, GZIPPY_FORCE_PARALLEL_SM, path=ParallelSM,
sha-verified), after the positive control passes:
- If ≥ majority of the non-startup cold-get stalls classify CONTAINING_CACHED or
  CONTAINING_IN_FLIGHT ⇒ **YES, parent resident ⇒ placement port (interior reuse) is the
  fix.** (Threshold pre-registered: ≥50% of non-startup stalls resident.)
- If ≥ majority classify NOT_RESIDENT ⇒ **NO, parent evicted ⇒ re-scope: the gap is
  cache-residency/consumer-pace; interior reuse needs a resident parent it does not have.**
  Surface to supervisor as a placement-direction re-scope.
- A SPLIT (no majority) is itself a finding: interior reuse helps the resident fraction only;
  report the fraction and flag the re-scope for the non-resident fraction.

DISCIPLINE: this probe answers ONLY the residency question. It does NOT license writing the
port. A YES means "placement port direction valid"; building it is gated by the supervisor.

## POST-RUN CORRECTION (advisor-caught, applied commit d764734c)
The original CONTAINING_CACHED test used `[encoded_offset_bits, max_acceptable_start_bit]`,
but that is the speculative START-tolerance window (decoded bytes BEGIN at
`max_acceptable_start_bit`, chunk_data.rs:143-145; a re-anchored chunk collapses to
enc==max) — so that counter could NEVER fire for a real containing parent (vacuous). FIX:
snapshot the encoded END (`enc + encoded_size_bits`) and test `enc < decode_start <
encoded_end`; ADD the bug-free necessary-condition channel `has_nearest_le_start` (does any
resident chunk START below the stalled offset — the minimum a containing parent must satisfy).
The NO verdict held on the corrected, cap-swept evidence: has_nearest_le_start=0 at default,
cap1, AND cap256 (huge cache) ⇒ never-retained / consumer-pace, NOT capacity-eviction.
