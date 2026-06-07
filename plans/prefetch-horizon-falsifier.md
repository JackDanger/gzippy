# SATURATION vs HORIZON falsifier — PRE-REGISTERED (leader, 2026-06-07, HEAD e52b0fc2)

Charter: plans/prefetch-horizon-diagnosis.md. The placement offset-supply lever is
REFUTED (GATE FAIL, advisor-confirmed: gzippy ALREADY re-targets the overshot index
at its confirmed offset; vendor absorbs the same cold-get via a 2·P guess-prefetch
HORIZON + pump-during-wait). The OPEN question (anti-escape-hatch — do NOT conclude
"it's all engine" without proving it): at each `decode_NOT_STARTED` head-of-line stall
(the `None` cold-get branch, chunk_fetcher.rs:1357), why hasn't the stalled index's
decode STARTED?

Two distinct causes, OPPOSITE fixes. This falsifier states — BEFORE any run — which
observation maps to which verdict. The verdict is decided by the observation, not the
analyst.

## THE TWO HYPOTHESES
- **SATURATION (engine):** all workers are busy decoding (the 2.38× slow engine), so
  the marginal index's decode hasn't started — there is no free worker to start it on.
  ⇒ the lever is the ENGINE; placement has no separate horizon headroom here.
- **HORIZON (structural):** the prefetcher never DISPATCHED that index far enough
  ahead (the stalled offset is not in the in-flight set) AND there was idle worker
  capacity to have decoded it — the depth/scheduling, not the engine, left it cold.
  ⇒ a genuine faithful-port lever: deepen the guess-prefetch horizon + pump-during-wait
  (mirror vendor 2·P) so the index's decode is DONE (not merely in-flight) at arrival.

## THE INSTRUMENT (counters only, env-gated, byte-exact — OFF==identity)
At the cold-get `None` stall site (chunk_fetcher.rs:1357, the SAME genuine
head-of-line stall the residency probe already classifies), snapshot ATOMICALLY at the
stall instant:
1. **Worker occupancy** — from ThreadPool: `capacity` (= configured T), `spawned`
   (`spawned_threads()`), `idle` (`idle_thread_count`, the count of workers parked in
   the condvar wait). Derived:
   - `busy = spawned - idle` (workers actively running a task)
   - `idle_capacity = idle + (capacity - spawned)` (parked workers + not-yet-spawned
     slots; lazy-spawn means an unspawned slot IS available capacity).
2. **Was the stalled index enqueued?** — `block_fetcher.prefetching_keys()`: is there
   any in-flight key K with K <= decode_start < K + partition_span (the in-flight decode
   that could cover decode_start), OR an exact in-flight key == decode_start? Reuses the
   stall_residency CONTAINING_IN_FLIGHT logic. Call this `enqueued`.

Classify each NON-STARTUP stall (decode_start != 0) into exactly one bucket:
- **SAT** : `idle_capacity == 0` (all workers busy) — saturation signal, regardless of
  enqueued (no slot to start it on).
- **HORIZON_NOT_ENQUEUED** : `idle_capacity > 0` AND `!enqueued` — idle worker(s)
  existed and the index was never dispatched ⇒ horizon-too-shallow signal.
- **HORIZON_ENQUEUED_NOT_DONE** : `idle_capacity > 0` AND `enqueued` — a worker was
  free AND the index WAS in-flight but not finished (the "in-flight-not-done" the prior
  3 attempts hit; lead too short, not depth too shallow).

## PRE-REGISTERED VERDICT MAP (decided by the buckets, not the analyst)
Let N = non-startup stalls.
- **SATURATION verdict** iff SAT >= ceil(N/2) (majority of stalls have NO idle slot).
  ⇒ report: placement has no separate horizon headroom; the path to the tie is the
  ENGINE (next gate = §2.3 isolation bench). Do NOT attempt a horizon deepening.
- **HORIZON verdict** iff (HORIZON_NOT_ENQUEUED) >= ceil(N/2) (majority had idle
  capacity AND the index was never enqueued). ⇒ a faithful guess-prefetch horizon
  deepening + pump-during-wait is a distinct, source-justified lever; attempt the SMALL
  byte-exact deepening; falsifier for the FIX = stall count drops AND wall → ~0.61s.
- **MIXED / IN-FLIGHT-NOT-DONE** iff HORIZON_ENQUEUED_NOT_DONE is the plurality (idle
  capacity existed but the index was already in-flight, just not done). ⇒ this is NOT a
  horizon-depth lever (it WAS dispatched); it is the lead-length / engine-speed problem
  the prior 3 attempts proved (decode in-flight-not-done). Report as engine-coupled, do
  NOT deepen depth (depth is already reaching it).
- Ambiguous (no bucket reaches majority): report the distribution; default to NO fix
  (anti-escape-hatch: do not invent a lever the data doesn't support).

## INSTRUMENT VALIDATION (CLAUDE.md rule 4 — before trusting any number)
- **OFF == identity:** env unset ⇒ inlined early-return; dual-sha (gzippy-native sha
  028bd002…cb410f on silesia via path=ParallelSM) must be byte-identical ON vs OFF.
- **POSITIVE CONTROL for the occupancy counter (saturation channel):** run with the
  SLOW knob (GZIPPY_SLOW_BOOTSTRAP/clean-loop inject) — a slower engine MUST increase
  `busy`/decrease `idle_capacity` at stalls (more SAT-classified). If the SAT count
  does NOT rise under a 2× engine slowdown, the occupancy counter is not tracking real
  saturation — fix the instrument before trusting it. (This is also Method step 3: the
  slow_knob cross-check — saturation grows with engine-slow; a horizon-bound count is
  flat regardless of engine speed.)
- **POSITIVE CONTROL for the enqueued counter:** with a TINY prefetch cap (force a
  shallow horizon, e.g. GZIPPY_STALL_RESIDENCY_PROBE cap analogue or a reduced
  prefetch_capacity) ⇒ HORIZON_NOT_ENQUEUED must rise. With a deep horizon ⇒ fall.
- **CONSERVATION:** SAT + HORIZON_NOT_ENQUEUED + HORIZON_ENQUEUED_NOT_DONE == N
  (non-startup); startup excluded. Assert in the report line.

## SOURCE-CITE STEP (delegate, read-only)
Independently source-cite gzippy's EFFECTIVE prefetch horizon/depth vs vendor 2·P +
pump-during-wait:
- gzippy: block_fetcher.rs prefetch cap (`prefetch_capacity` = `pool*2`,
  chunk_fetcher.rs:528-529), in-flight cap (`prefetching_len()+1 >= parallelization`,
  :737), strategy request count (`prefetch_cache.capacity()`, :758-765), adaptive ramp
  (prefetcher.rs:109-141), pump-during-wait (chunk_fetcher.rs:1289-1301).
- vendor: BlockFetcher.hpp:181-182 (caps), :465-468/:474 (depth+2·P request),
  Prefetcher.hpp:126-166 (ramp), GzipChunkFetcher.hpp:312-316 (pump).
The vendor-pacing-map.md ALREADY found these MATCH line-for-line. So if the horizon is
NOT shallower, the HORIZON verdict is structurally unlikely a priori — which makes the
occupancy/enqueued buckets the deciding evidence (don't let the prior "MATCH" map
foreclose the measurement; the measurement overrules attribution).

## DISCIPLINE
Subagents synchronous + killed; no detached sentinel; guest run from a Bash task
holding the ssh; verify guest idle before + restore host after; builds via
cargo-lock.sh; numbers only from the locked harness; diagnose the first error before
retrying. STOP at the checkpoint; spawn ONE independent disproof advisor; do NOT start
the engine inline-ASM build.
