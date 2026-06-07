# PLACEMENT PORT AUTHORIZATION — the first BUILD (supervisor, 2026-06-07)

Lag-causality (HEAD e52b0fc2, advisor-vetted) = MIXED: the 3 overshoot-tail cold-gets are
STRUCTURAL (real placement lever, doesn't dissolve into engine); magnitude is engine-coupled;
both co-primary. Precise lever located: block-finder offset SUPPLY — gzippy never re-targets
an overshot index at its CONFIRMED offset; rapidgzip does (GzipBlockFinder.hpp:117-158).

## Supervisor RATIFIES proceeding to the placement port (the first real build), GATED.
This is byte-exact + falsifier-gated + reversible, so it is within the measurement-gated
TIER-3 authorization (the multi-week ENGINE build remains separately gated behind the §2.3
isolation bench — NOT authorized here).

## HARD GATE before writing any port code: the 3-prior-failures re-derivation
3 prior gzippy attempts at this FAILED. Do NOT write attempt #4 until you have, first-hand
and source-cited:
1. **HOW vendor avoids the overshoot cold-get** with the same in-order ~1-chunk consumer
   lead gzippy had — the exact mechanism (GzipBlockFinder offset supply + how the consumer
   re-requests the overshot index at the confirmed offset), vendor file:line.
2. **WHY the 3 prior gzippy attempts failed** (find them in git history / plans / status).
   The new port MUST be mechanistically DIFFERENT from the failed three, or it's a blind
   retry. State the difference explicitly.
3. Have this re-derivation pass its OWN advisor mini-check (is the proposed re-target a
   FAITHFUL port of vendor, and genuinely distinct from the failed attempts?) BEFORE coding.
If the re-derivation does NOT yield a clear faithful mechanism distinct from the failures,
STOP and checkpoint — do not code a fourth blind attempt.

## THEN implement (only if the gate passes)
- Faithful re-target of the overshot index at its confirmed offset. Byte-exact every step
  (dual-sha 028bd002…cb410f both features); keep all lib tests green.
- PRE-REGISTER the falsifier (design-v2 §1.3): the port must DROP the head-of-line stall
  count (3→fewer) AND move the wall toward ~0.61s. Per CLAUDE.md rule 7a, a byte-exact
  change is KEPT even on a TIE — but report honestly whether the stall count dropped.
- Re-measure on the locked guest harness (verify idle first, restore after, N≥9 interleaved,
  sha-verified). Report stall-count delta + wall delta.

## CHECKPOINT (STOP)
Report: the re-derivation (vendor mechanism + why prior 3 failed + how this differs); whether
the port landed byte-exact; the stall-count + wall deltas vs the falsifier. Route through an
independent disproof advisor (verdict to plans/placement-port-advisor-verdict.md). Then STOP
for supervisor gate. Do NOT start the engine isolation bench or engine build.

## CONTEXT (not this turn): after placement lands + re-measures, next gate is the §2.3 ENGINE
isolation bench (with in-bench ISA-L positive control) to bound the engine speed-up ceiling
BEFORE the inline-ASM engine build. The tie needs BOTH levers; placement alone → ~0.61s.

## DISCIPLINES (enforced)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; guest runs from a
Bash task that HOLDS the ssh; verify guest idle before + restore host after; leave NO orphaned
processes; serialize builds via cargo-lock.sh (df -h around builds — disk at 29%, fine);
don't run multi-line python via Bash (write a .py file); diagnose the FIRST error before
retrying; numbers only from the locked harness. Update plans/orchestrator-status.md.
