# FULL igzip ASM-KERNEL PORT — FEASIBILITY SCOPE (supervisor, user-chosen 2026-06-07)

User decision (after the PLATEAU go/no-go, memory project_engine_plateau_pure_rust): SCOPE a
full igzip AVX2 decode-kernel port — feasibility-first, honoring BOTH the 1.0× tie bar AND
no-C-FFI-in-native. This step is ANALYSIS/DESIGN ONLY — NO multi-session build. Output a
feasibility verdict + port plan for supervisor + user ratification before any build.

## Why this step exists
The engine bench proved: grafting E2-E4 onto the faithful pure-Rust **u16-ring** engine
plateaus at +13% (0.41× ISA-L; tie needs 0.85×). The only identified remaining pure path is
transliterating igzip's whole AVX2 decode kernel as our own inline-Rust-asm (our asm = NOT
C-FFI). The risk is NOT the asm itself (it's igzip's own code) — it's whether it can be
INTEGRATED into gzippy's architecture and reach the rate. Scope that before committing.

## THE THREE QUESTIONS THE SCOPE MUST ANSWER (source-cited, advisor-vetted)
1. **What actually makes igzip fast** (beyond the E2-E4 grafts we already tried)? Map
   igzip_decode_block_stateless_01/04.asm + igzip_inflate.c: the decode-table layout, the
   bulk multi-symbol loop, the state machine, the u8 output model. Cite vendor file:line.
   Delegate to a read-only subagent (SYNCHRONOUS).
2. **Can it be INTEGRATED faithfully, and at what architectural cost?** igzip writes **u8**;
   gzippy's faithful engine uses the **u16 marker-ring + in-place flip** (the one-engine
   governing memory project_faithful_unified_decoder_over_perf). Determine:
   - Is igzip-class achievable WITHIN the one-engine u16-flip arch, or does it REQUIRE a u8
     clean path?
   - CRITICAL FAITHFULNESS QUESTION: rapidgzip ITSELF reaches igzip-class by handing its
     CLEAN TAIL to a u8 ISA-L path (deflate::Block for the ≤32KiB markered prefix → ISA-L u8
     for the bulk). So "our igzip-asm-port on a u8 clean tail" may be the MOST FAITHFUL port
     of rapidgzip's real structure — BUT it is a TWO-PHASE handoff, which the 2026-06-05
     governing memory forbade as "Divergence #2." Lay out this tension EXPLICITLY: faithful-
     to-rapidgzip-structure (u8 clean tail, our asm) vs the gzippy one-engine-flip invariant.
     This is a USER-LEVEL fork — surface it, do NOT pre-decide.
3. **Will it PROJECT to the tie?** Estimate the ported kernel's rate (it's igzip's code → ~
     igzip-class ~283-388 MB/s expected) and project via §3 (same-sink 0.604s bar) whether it
     closes the 0.41→0.85 gap. If the integration forces overheads that keep it < tie, say so.

## DELIVERABLE (checkpoint, STOP — no build)
A feasibility report: (1) the igzip-kernel map; (2) the integration verdict + the faithfulness
fork (u16-one-engine vs u8-faithful-to-rapidgzip-with-our-asm) with a recommendation; (3) the
projected rate/tie. Route through an independent disproof advisor (verdict to
plans/asm-kernel-feasibility-advisor-verdict.md) attacking the rate projection + the
faithfulness analysis + whether the asm is realistically portable+maintainable. Then STOP for
supervisor + USER ratification before any build.

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- This is ANALYSIS: delegate read-only vendor study to subagents but run them SYNCHRONOUSLY
  (block in-turn); do NOT background-and-yield (two leaders died doing that). NO auto-reinvoke.
- NO detached sleep sentinel; leave NO orphaned processes (kill all claude -p + timeout sleeps
  before finishing; the supervisor has cleaned orphans every round).
- SOURCE-VERIFY every premise first-hand (a wrong premise wasted a whole turn).
- No build needed; if any sanity build, serialize via cargo-lock.sh. Don't run multi-line
  python via Bash. Update plans/orchestrator-status.md.
