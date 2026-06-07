# TIER-3 AUTHORIZATION — STEP 0 discriminators (supervisor, 2026-06-07)

Supervisor RATIFIES the STEP-C design (plans/tier1-design-v2.md) as the ADVISOR did
(plans/step-c-design-advisor-verdict.md): **ratify the METHOD, not the conclusion.**
TIER-3 is authorized ONLY as a MEASUREMENT-GATED INVESTIGATION. The headline "1.0× tie
reachable" is NOT ratified — it rests on two unproven floor-equalities. Nothing expensive
(placement port, multi-week engine build) is authorized yet; each is gated by a measurement.

## Ratified
- The SEQUENCE (§5): step-0 discriminators → placement (conditional) → engine bench →
  engine build (conditional). Correctly falsifier-gated.
- The governing-tension resolution (one no-FFI engine, igzip-class inner loop via pure-Rust+
  inline-ASM, inner-loop divergence accepted while ARCHITECTURE stays faithful) — consistent
  with the user's explicit prior authorizations (inner Huffman loop = open territory; inline
  ASM allowed; 1.0× bar; no-FFI native).

## NOT ratified (advisor caveats, now BINDING constraints on TIER-3)
1. **Reachability is ≥0.54s, not =0.54s.** The 0.61→0.54 step assumes gzippy's non-decode
   floor = rapidgzip's; ~225ms in-order consumer-serial bookkeeping + a 0.497s consumer
   block measure AGAINST it. The wall conclusion is CONTINGENT on the non-decode floor ≤0.54s.
2. **Placement is NOT "just wire dead code."** `find_data_offset` is decoded-BYTE-keyed but
   gzippy's consumer is encoded-BIT-keyed → porting the entry implies a CONSUMER REDESIGN
   (architecture — forbidden by the faithful-port mandate); `unsplit_blocks` is built only
   for subchunks>1; and interior-reuse presupposes the parent is still CACHED, which
   project_confirmed_offset_prefetch_gap MEASURED to FAIL (318ms lag → eviction) and left
   UNANSWERED. Do NOT build the placement port until the parent-cached discriminator says YES.
3. **Engine bench gates the ENGINE, not the WALL.** A passing §2.3 isolation bench proves
   the engine can reach igzip-class; it does NOT prove the wall ties (the wall re-binds on
   caveat-1's floor). State §2.3 PASS as "engine reaches X clean rate," wall contingent.

## THIS TURN — STEP 0 ONLY (cheap, byte-exact, then STOP)
Run the two pre-registered discriminators; do NOT proceed to placement port or engine work.

(a) **Parent-cached-at-stall probe** (§1.2 precond 3 / the confirmed-offset memory's
   unanswered discriminator): when the consumer stalls at a confirmed offset, IS the
   containing partition chunk in-flight/cached, or evicted? YES ⇒ interior reuse is the fix
   (placement port is on-track). NO ⇒ the gap is cache-residency/consumer-pace (gzippy
   discards what rapidgzip keeps) and the placement direction must be re-scoped — surface it.

(b) **Consumer-block decompose** (§3): split the placement-perfect ~0.61s consumer block
   into DECODE-WAIT vs SERIAL-BOOKKEEPING. Sets gzippy's true NON-DECODE floor.
   **HARD GATE / ESCALATION:** if the non-decode serial floor measures **>0.54s**, STOP and
   ESCALATE TO THE SUPERVISOR (→ user-level fork: revisit the bar, add a 4th off-critical-
   path consumer lever faithfully mirroring rapidgzip, or accept FFI). Do NOT start the
   engine build chasing a tie the consumer structurally forbids.

Each discriminator: pre-register its falsifier BEFORE running; validate the instrument with
a positive control; numbers ONLY from the locked guest harness (verify guest idle first,
restore host after); byte-exact (these are probes, keep them off the production bytes or
prove OFF==identity).

## CHECKPOINT (STOP)
Report both discriminator results + their implication:
- parent-cached YES/NO (→ is placement-port the fix?).
- non-decode floor value (→ ≤0.54s continue, >0.54s ESCALATE).
Route through an independent disproof advisor (verdict to plans/step0-advisor-verdict.md).
Then STOP for supervisor gate. Do NOT start the placement port or engine work.

## DISCIPLINES (enforced — every prior leader needed reminding)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; guest runs from
a Bash task that HOLDS the ssh (bare claude -p SIGHUPs its ssh → orphans guest run); leave
NO orphaned processes (kill subagent guest runs explicitly); serialize builds via
cargo-lock.sh; don't run multi-line python via Bash (write a .py file); diagnose the FIRST
error before retrying. Update plans/orchestrator-status.md.
