# PREFETCH-HORIZON vs WORKER-SATURATION DIAGNOSIS (supervisor, 2026-06-07)

The offset-supply placement lever is REFUTED (HEAD e52b0fc2, GATE FAIL, advisor-confirmed):
gzippy ALREADY re-targets the overshot index at its confirmed offset (nothing to change);
vendor absorbs the same cold-get via a 2·P guess-prefetch HORIZON + pump-during-wait, not a
block-finder trick; the 3 prior failures all had the correct offset at the same ≤1-chunk lead.

## THE OPEN QUESTION (anti-escape-hatch: do NOT conclude "it's all engine" without this)
The head-of-line stalls are all `decode_NOT_STARTED`. Two distinct causes, opposite fixes:
- **WORKER SATURATION (engine):** all workers are busy decoding (slowly, the 2.38× engine),
  so the marginal index's decode hasn't started. ⇒ the lever is the ENGINE, placement has no
  separate headroom here.
- **PREFETCH HORIZON too shallow (structural):** the prefetcher never DISPATCHED that index
  far enough ahead (vendor dispatches ~2·P ahead + pumps during consumer wait). ⇒ a genuine
  faithful-port lever: deepen the horizon so the index's decode is DONE (not just in-flight)
  when the consumer arrives.

## METHOD (diagnose which; pre-register the falsifier BEFORE running)
At each `decode_NOT_STARTED` head-of-line stall on the locked guest (T8), determine:
1. **Were all worker slots busy at that instant?** (saturation signal) — vs idle workers
   with the index simply not yet queued (horizon signal). Instrument the worker-pool
   occupancy + whether the stalled index had been ENQUEUED to the prefetcher at stall time.
2. **Causal cross-checks (use existing knobs, no premature build):**
   - Compare gzippy's effective prefetch horizon/depth to vendor's 2·P, source-cited (how
     far ahead does gzippy's prefetcher dispatch vs vendor — chunk_fetcher.rs prefetch_capacity
     / pool_size vs GzipChunkFetcher 2·P + pump-during-wait). A SHALLOWER horizon ⇒ structural.
   - slow_knob: if engine-slow makes MORE stalls NOT_STARTED (saturation grows) that points
     engine; if NOT_STARTED count is horizon-bound regardless of engine speed, structural.
   Falsifier: pre-state which observation ⇒ which verdict.
3. If the verdict is HORIZON: a small, byte-exact, FAITHFUL deepening of the guess-prefetch
   horizon + pump-during-wait (mirror vendor 2·P), falsifier = stall count drops AND wall
   moves toward ~0.61s. (Only if clearly faithful + distinct from the 3 prior failures —
   those were offset-supply, not horizon, so horizon IS distinct.)
   If the verdict is SATURATION: report it — placement has no separate headroom; the path to
   the tie is the ENGINE (gate the §2.3 isolation bench next).

## CHECKPOINT (STOP)
Report the verdict (saturation vs horizon) with the perturbation/occupancy numbers; if
horizon and a faithful deepening was attempted, the byte-exact + stall + wall deltas. Route
through an independent disproof advisor (verdict to plans/prefetch-horizon-advisor-verdict.md).
Then STOP for supervisor gate. Do NOT start the engine inline-ASM build.

## DISCIPLINES (enforced — and a NEW one)
- SOURCE-VERIFY a lever's premise FIRST-HAND before treating it as actionable (the
  offset-supply premise was wrong and wasted a turn — the gate caught it; don't repeat).
- Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel (the held
  pid-56181 lock orphaned and had to be cleared); guest runs from a Bash task HOLDING the
  ssh; verify guest idle before + restore host after; leave NO orphaned processes (2 orphaned
  advisors had to be killed this round); serialize builds via cargo-lock.sh; numbers only from
  the locked harness; don't run multi-line python via Bash (write a .py file); diagnose the
  FIRST error before retrying. Update plans/orchestrator-status.md.
