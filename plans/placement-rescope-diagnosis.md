# PLACEMENT RE-SCOPE + LAG-CAUSALITY DIAGNOSIS (supervisor, 2026-06-07)

STEP-0 (HEAD 598e22a4, advisor-vetted) resolved both discriminators:
- Non-decode serial floor ~0.015s ≪ 0.54s ⇒ the tie is NOT structurally forbidden by the
  consumer (the ~225ms fear was mis-bucketed decode-wait). NO user escalation. (Caveat:
  real file-sink writev ~0.245s excluded — fold into the FINAL same-sink floor check.)
- Parent-cached = NO, not eviction (held at cap=256). Consumer lags its prefetcher ~318ms;
  the containing chunk is never retained. ⇒ interior-reuse/getIndexedChunk (design-v2 §1.2)
  is NOT the fix; placement RE-SCOPES to consumer-pace.

## THE PIVOTAL OPEN QUESTION (answer FIRST — gates whether placement is a lever at all)
Is the **318ms consumer-prefetcher lag STRUCTURAL** (prefetch depth / scheduling / an
in-order serialization rapidgzip avoids) **or an EFFECT of the 2.38× slow engine** (the
consumer simply can't keep up because each chunk decodes slowly)?
- If **engine-induced**: placement partly DISSOLVES into the engine lever (fixing the engine
  speeds the consumer, shrinking the lag) — there may be ~ONE lever, not two. Re-rank.
- If **structural**: placement is a genuine separate faithful-port lever (mirror rapidgzip's
  pacing) that pays even before the engine.

### Method (causal perturbation with the EXISTING slow_knob — no new instrument)
Use the committed clean-loop slow-injection (GZIPPY_SLOW_MODE/KIND, marker_inflate native
arm) to vary engine speed and measure the consumer-prefetcher lag RESPONSE on the locked
guest, T8:
- Slow the engine by F ∈ {0,50,100} and measure the lag (consumer-behind-prefetcher ms)
  AND the head-of-line stall count. If the lag grows ~proportionally with engine slowdown
  ⇒ ENGINE-INDUCED. If the lag is ~flat ⇒ STRUCTURAL. Run the frequency-neutral sleep
  control. Pre-register the falsifier (which way = which verdict) BEFORE running.
- Cross-check: at the clean-only operating point (A.2 oracle, faster effective per-chunk),
  is the lag already smaller? (corroborating signal, not the verdict.)

## THEN — map rapidgzip's pacing (read-only subagent, source-cited)
Why is rapidgzip's consumer ~0-17ms behind its prefetcher? What structure achieves it
(prefetch depth, when it evicts/retains, in-order vs off-path consumer work)? Cite vendor
file:line. This is the faithful-port reference for the structural case.

## DELIVERABLE (checkpoint, STOP)
1. The lag-causality verdict (structural vs engine-induced) with the perturbation numbers.
2. IF structural: a re-scoped placement design = faithful port of rapidgzip's pacing
   (source-cited, within the faithful-port mandate — NOT a gzippy-invented scheduler).
   IF engine-induced: a re-ranked plan folding placement into the engine lever + the revised
   reachability arithmetic.
3. Route through an independent disproof advisor (verdict to plans/placement-rescope-advisor-verdict.md).
Then STOP for supervisor ratification. Do NOT implement the placement change or start the
engine build this turn — diagnosis + design only.

## DISCIPLINES (enforced)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; guest runs from
a Bash task that HOLDS the ssh; verify guest idle before + restore host after; leave NO
orphaned processes; serialize builds via cargo-lock.sh; numbers only from the locked harness;
don't run multi-line python via Bash (write a .py file); diagnose the FIRST error before
retrying. Update plans/orchestrator-status.md.
