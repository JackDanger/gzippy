# Phase-2 inline-asm igzip hot-loop prototype — brief for the disproof advisor

## Charter context
USER DECISION 2026-06-08: close the ~36ms native→ocl_cf engine gap by transliterating
igzip's AVX2 clean-decode hot loop as inline Rust asm (pure-Rust, no-FFI). EXECUTE
TIERED: prove the asm closes the 36ms IN ISOLATION (Phase-2 gate) before any production
integration (Phase-3). All high-level techniques are maxed: VAR_VI (pure-Rust speculative
flat-u8 loop + igzip packed-u32 multi-symbol table + BMI2 BZHI/SHRX + AVX2 MOVDQU copy)
is the LLVM-compiled ceiling; ISA-L (VAR_III) is the oracle ceiling (ocl_cf 0.945× whole-
system). Pre-registered falsifier: PASS = inline asm closes a material fraction of the
36ms toward igzip-class (approaches ISA-L clean rate); PLATEAU = asm ≈ LLVM (no headroom)
⇒ even hand-asm-in-Rust can't beat LLVM here → STOP and escalate (ceiling ~0.86×).

## Phase-1 SOURCE-MAP (read-only subagent, cited)
igzip_decode_block_stateless.asm:507-627 is the fast loop. What the hand-asm does that
LLVM does NOT emit from idiomatic Rust:
- F1 one-iteration-ahead literal-table gather across the back-edge (asm:540): next
  symbol's short_code_lookup[12-bit] load issued before the current symbol is retired.
- F2 speculative dist-table gather before the lit-vs-len branch (asm:550-552).
- F3 unconditional flag-free SHLX/SHRX refill+consume; IN_BUFFER_SLOP=8 makes the
  "enough bits?" branch unnecessary (asm:528-531,543-547,370).
- F4 all loop-carried state pinned in callee-saved GPRs, no spill (asm:108-136).
- C  8-byte speculative packed-literal store + advance-by-count (asm:518-519).
- D  16-byte MOVDQU back-ref copy w/ distance-doubling overlap (asm:603-627).
- E  one slop-adjusted two-compare gate per multi-symbol iteration (asm:488-489,508-512).
VAR_V/VAR_VI already implement C, D, E, and a Rust-level PRELOAD; the things LLVM cannot
synthesize and VAR_VI lacks are F1 (gather hoist across back-edge), F2, F3 (branchless
refill), F4 (register pinning). VAR_VII targets F1/F3/F4/C in inline asm.

## Phase-2 PROTOTYPE (VAR_VII)
A core::arch::asm! literal-run hot loop: keeps read_in(bitbuf)/read_in_length(bitsleft)/
next_in_pos(pos)/next_out pinned in registers; does the unconditional SHLX refill, the
12-bit table gather, the SHRX consume, and the 8-byte speculative packed-literal store
entirely in asm; ONE slop-adjusted two-compare gate per iteration. It EXITS to the
validated Rust careful tail on any length code / long code / boundary (silesia is ~31%
back-refs, so the asm covers the literal RUNS between back-refs; each back-ref is an
asm→Rust handoff that writes the 4 loop regs back to `bits`, resolves the dist+copy, then
re-enters the asm). Byte-exact by construction (refill mirrors gzippy's Bits convention
bitsleft=len|56, NOT igzip's own accounting).

## BYTE-EXACTNESS (the absolute gate)
GUEST native x86_64, frozen host: SHA_ALL_EQUAL=yes on ALL 5 swept clean silesia chunks
(VAR_VII byte-identical to VAR_I scalar AND VAR_III ISA-L). SELFTEST=PASS. The
GZIPPY_VII_CAREFUL_ONLY knob proved the careful machinery is exact in isolation from the asm.

## MEASUREMENT (authoritative: guest native x86_64, frozen host no_turbo=1 perf,
## taskset core 0, interleaved best-of-11, vs the ISA-L oracle VAR_III)
Per-chunk vs_iii (variant rate / ISA-L rate):
| chunk | VAR_VI (LLVM) | VAR_VII (asm) |
|---|---|---|
| 67145126  | 0.597 | 0.547 |
| 167879427 | 0.589 | 0.540 |
| 268518043 | 0.572 | 0.530 |
| 369261644 | 0.561 | 0.520 |
| 469882459 | 0.593 | 0.559 |
AGGREGATE med-of-med MB/s: ISA-L 283 | VAR_VI 169 (0.597×) | VAR_VII 155 (0.547×).

## RESULT
VAR_VII is systematically 0.04-0.05× BELOW VAR_VI on EVERY chunk, and both sit at
~0.53-0.60× ISA-L — far from the 0.945× ocl_cf / ISA-L-class ceiling. The inline-asm
transliteration does NOT close the 36ms; it is marginally WORSE than the LLVM-compiled
loop. Pre-registered falsifier ⇒ PLATEAU.

## CLAIMED VERDICT (for disproof): NO-GO for Phase-3.
The partial-loop transliteration (asm literal-run + Rust back-ref handoff) cannot match
ISA-L because ISA-L keeps the WHOLE loop — including the frequent (~31%) back-refs — in
one register-pinned asm machine; gzippy's per-back-ref asm↔Rust handoff overhead erases
the literal-run gains. A full-kernel asm rewrite (including F2 dist-gather + D SIMD copy
inside the asm, no Rust handoff) is the only thing this prototype did NOT test.

## QUESTIONS FOR THE ADVISOR (disprove or sharpen)
1. Is the PLATEAU verdict EARNED, or is the NO-GO an artifact of the PARTIAL scope (asm
   literal-run only, Rust back-ref handoff)? Specifically: does the per-back-ref handoff
   confound invalidate the conclusion that "inline asm can't beat LLVM here," OR is the
   handoff itself the honest finding that a faithful partial transliteration plateaus?
2. Should the gate require a FULL-kernel asm prototype (back-ref + dist-gather in asm, no
   handoff) before declaring NO-GO — i.e. is the falsifier under-powered as run?
3. Is VAR_VI a valid LLVM-ceiling baseline (it already has BMI2/AVX/preload)? Is the
   ~0.55-0.60× ISA-L ratio in the bench consistent with the whole-system 0.945× ocl_cf
   ceiling (the bench is single-thread pure clean decode, no scheduler/output)?
4. Byte-exactness: is SHA_ALL_EQUAL=yes over 5 chunks vs TWO independent oracles (scalar +
   ISA-L) sufficient to trust the rate, or is more coverage owed?

Be a genuine disproof: try to BREAK the NO-GO. If the partial scope is a fatal confound,
say so and name the full-kernel prototype as owed before any escalation.

---

## UPDATE 2 — re-entry version (advisor pass-1 fix) MEASURED on the guest

The advisor (pass 1) correctly broke the first cut: the asm ran once per block over
only the leading literal run (coverage 0.89%), so the deficit was careful-tail-vs-
fast-loop, not asm-vs-LLVM. FIX: wrap the asm in an 'asm_reentry loop — on a length
code resolve ONE back-ref in Rust then re-enter; on a long code decode ONE symbol in
Rust then re-enter; only a boundary ends the asm phase. NEW coverage counter
(GZIPPY_VII_COVERAGE) confirms the asm now emits 57-99% of bytes (median ~0.65-0.74,
per-chunk: 0.57/0.65/0.74/0.99/0.62). Still SHA_ALL_EQUAL=yes (byte-exact vs scalar
AND ISA-L on all 5 chunks).

GUEST (native x86_64, frozen host no_turbo=1, taskset core 0, interleaved best-of-11):
| chunk | ISA-L | VAR_VI (LLVM) vs_iii | VAR_VII (asm-reentry) vs_iii | VAR_VII vs SCALAR |
|---|---|---|---|---|
| 67145126  | 283 | 0.596 | 0.275 | 0.758 |
| 167879427 | 266 | 0.591 | 0.279 | 0.745 |
| 268518043 | 324 | 0.573 | 0.295 | 0.825 |
| 369261644 | 173 | 0.568 | 0.335 | 0.803 |
| 469882459 | 386 | 0.591 | 0.203 | 0.722 |
AGGREGATE med-of-med: ISA-L 283 | VAR_VI 168 (0.594×) | VAR_VII 78 (0.276×).

RESULT: with the asm now carrying the dominant byte share, VAR_VII is DRAMATICALLY
WORSE — 0.28× ISA-L, and ~0.75× of even the NAIVE SCALAR baseline. The per-symbol
asm↔Rust re-entry (300-460K re-entries/chunk, each spilling the 4 loop regs to `bits`
and re-reading) dominates. Two bracketing data points: leading-run-only 0.55×
(asm barely runs ≈ careful tail) and re-enter-per-symbol 0.28× (asm runs, handoff
dominates). Neither approaches ISA-L; both are below the LLVM VAR_VI ceiling.

## CLAIMED VERDICT v2 (for disproof): NO-GO for the partial-handoff inline-asm approach.
A faithful partial transliteration plateaus-or-regresses: any dist-decode/long-code
left in Rust forces a per-back-ref register spill, and silesia is ~31% back-refs, so
the handoff is paid every ~2-3 symbols. ISA-L's advantage is keeping the WHOLE loop
(back-ref copy + dist gather + long code) in one register-pinned asm machine. The ONLY
untested path is a FULL-kernel asm (the entire loop in asm, zero Rust handoff) = hand-
re-writing ISA-L, a multi-session high-risk effort — AND its whole-system payoff is
known-small (the engine rate is slack-masked at T8 per the charter's Phase-0; ocl_cf
whole-system ceiling is only 0.945× and partly a shared output floor). So the user-fork
IS now correctly triggered: accept ~0.86× pure-Rust, OR authorize the full-kernel asm
knowing the small bounded payoff.

## QUESTIONS v2
1. Is the NO-GO for the PARTIAL-HANDOFF approach now EARNED (asm coverage 57-99%,
   measured worse than both LLVM and scalar)? Or is there still a confound?
2. Is "the only remaining path is a full-kernel asm = re-write ISA-L by hand" correct,
   and is the user-fork (accept 0.86× vs authorize multi-session full-kernel asm with a
   known-small T8 payoff) now correctly triggered?
3. Anything that would make a full-kernel asm prototype worth the multi-session cost
   GIVEN the engine is slack-masked at T8?
