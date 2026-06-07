# INDEPENDENT DISPROOF VERDICT — igzip ASM-KERNEL PORT FEASIBILITY REPORT

Read-only advisor. Every vendor `file:line` below was opened and read first-hand this
session (not trusted from the report's summary). Target under attack:
`plans/asm-kernel-feasibility-report.md`. Charter: `plans/asm-kernel-feasibility-scope.md`.

**BOTTOM LINE FIRST:** the report's core conclusion is **SOUND — no FATAL error**.
Conditional-feasible; Arm A is the only tie-projecting path and only with placement also
closed; the fork is a genuine user-level call. The faithfulness analysis is honest (it does
NOT evade the governing memory — it surfaces the contradiction explicitly). Two material
**CORRECTIONS** the supervisor must hear: (3) the "prefer intrinsics over `asm!`"
recommendation is under-supported and is actively contradicted by igzip's OWN
C(slow)/asm(fast) two-implementation split — treat the route as an OPEN codegen risk, not a
settled recommendation; and (1b/2) **Arm B is probably UNDER-rated** — gzippy already
amortizes the ring modulo to a physical-pointer bump (`marker_inflate.rs:1751`), so the "ring
tax" is wrap-boundary handling every 64 KiB, not a `% RING_SIZE` per copy; Arm B deserves a
real isolation bench, not a projected dismissal.

---

## ATTACK 1 — THE RATE PROJECTION — **UPHELD-WITH-CORRECTION**

### 1a. Arm A ~250–283 MB/s — optimistic anchor, but the verdict is ROBUST to the optimism.
The 283 anchor is **ISA-L's OWN measured rate** (the FFI oracle in the round-2 bench,
`project_engine_plateau_pure_rust`: ISA-L=283, scalar=104, E234=118). Projecting "our
transliteration ≈ 250–283" is the report taking the TOP of the range = the vendor's own
codegen as achievable by our port. That is circular and the report discounts it only verbally
("DISCOUNTED for our-asm-vs-vendor-codegen", report:184). **However — and this is the report's
strongest, slightly-under-stated point — the §3 re-bind makes the Arm A TIE verdict robust to
a large engine-rate miss.** Worked through the §3 model (`tier1-design-v2.md:228-251`):
decode_wall = 39 × per-chunk / 8 × 1.36; per-chunk = 92.7 ms × 104/rate.
- at 283 MB/s → 34 ms/chunk → decode_wall **0.225 s**
- at 220 MB/s (intrinsics under-shoot) → 44 ms → **0.290 s**
- at 200 MB/s → 48 ms → **0.318 s**

All three are well below the ~0.54 s shared floor, so decode STOPS binding at any igzip-ish
rate (>~120 MB/s) and the wall re-binds on the floor regardless. So the Arm A tie does not
actually depend on hitting 283 — it depends on the floor, not the ceiling. The report should
have SAID this (it would have strengthened its own case); instead it leaves the 283 looking
load-bearing when it is not.

### 1b. Arm B 283→150–200 discount — **GROUNDED in direction, SOFT in magnitude, possibly PESSIMISTIC.**
The discount rests on two real datapoints, both verified:
- E1 round-1 "+6% output-only" (`project_pregate`, round-1 bench: u8-direct output 125 vs
  scalar 118 = +6%) — confirms output width alone is minor; the tax is elsewhere. ✓
- the 05a3835 ring-overshoot hazard (`project_faithful_unified_decoder_over_perf` item #3:
  rounded word-copy corrupted `corpus_large_repetitive` via circular-ring overshoot). ✓

BUT the report frames the ring tax as "a `% RING_SIZE` on every copy-source address
… exactly the per-symbol cost igzip avoids" (report:138-141). **First-hand code reading
contradicts the "every copy" framing.** gzippy already has a no-per-byte-modulo fast path:
`marker_inflate.rs:1751` ("pointer bump (no `% RING_SIZE` … on the write-address)") and the
`run_multi_cached_loop` region bumps `phys = phys.wrapping_add(1)` (`:1789,:1957`), computing
a physical pointer once and advancing it, handling wrap only at the 64 KiB boundary — exactly
how igzip runs linearly until its `end_out - OUT_BUFFER_SLOP` boundary (asm:488,509-512). So
the ring tax is **wrap-handling once per 64 KiB of output**, NOT a modulo per copy. Within a
64 KiB span the kernel CAN run linearly. The genuine residual cost is real but narrower than
stated: a `MOVDQU` back-ref read of `next_out − dist` can straddle the 64 KiB wrap (the
05a3835 hazard, now on u8), needing a per-byte tail only near wraps. **Consequence:** the
150–200 band is a soft guess that may be pessimistic; Arm B could land closer to Arm A. This
does not overturn "Arm B is the harder integration" — but it does mean **Arm B should be
isolation-benched, not projected-and-dismissed as a narrow miss.**

### 1c. Does a NON-decode term re-bind FIRST? — the report's handling is SOUND.
Checked against STEP-0 (`project_pregate`): the consumer-block decompose found the
non-decode SERIAL floor = **~0.015 s** (the feared ~225 ms was decode-wait mis-bucketed;
decode-wait ≈ 0.49 s = 97%); SAME-SINK file writev adds **~0.245 s**. So after BOTH levers
fix, gzippy's non-decode floor ≈ 0.015 + 0.245 ≈ **0.26 s** — BELOW 0.54 s. The report's
"placement-laden consumer floor ≈ 0.54–0.66 s" (report:204) is therefore **partly
decode/placement-COUPLED, not an independent hard floor** — much of that 0.54 s is itself
the decode-wait and the cold-get lag that BOTH levers shrink. The binding term at the tie is
**rapidgzip's OWN shared same-sink floor (~0.54–0.60 s, DRAM + writev)** — which gzippy
matches but cannot beat. Net: the report's NUMBER (0.54–0.66) is right for the wrong reason,
and its conclusion ("Arm A = TIE-to-+10%, ONLY if placement also closed") is correct and
matches the co-primary finding (`project_pregate`: placement-perfect alone = 0.56–0.66 s =
+7–26%; engine residual survives perfect placement). **The "engine alone ≠ tie" is upheld.**

### 1d. §3 PASS criterion is correctly STRUCTURAL, not the numeric coincidence.
Verified the trap the report flags (report:191-194). At E234's 118 MB/s, decode_wall = 39 ×
81.7 ms / 8 × 1.36 = **0.541 s** — numerically brushing the 0.54 floor. The report (and the
round-2 advisor before it) correctly REJECTS this as "a 13%-engine-bump artifact while decode
still binds," requiring instead (ii)/(iii) ≥ 0.85 so decode drops BELOW the floor and
re-binds OFF. This is why Arm A (ratio ~0.9–1.0) passes and Arm B (ratio 0.53–0.71) is rated
a miss even though Arm B's projected wall numerically brushes the bar. **This is rigorous and
internally consistent** — the report does not fall for its own arithmetic coincidence.

---

## ATTACK 2 — THE FAITHFULNESS ANALYSIS — **UPHELD** (with a framing caution)

### 2a. "u16 ring forbids the igzip kernel" (Q2a) — **SOUND, source-confirmed.**
Verified all four tricks are defined over a flat-u8 LINEAR buffer:
- speculative 8-byte packed literal store: `mov [next_out], next_sym` / `add next_out,
  next_sym_num` (asm:518-519) — a single u8-wide store of up to 3 packed literals
  (table packs them: `short_code_lookup[code] = sym1 | (sym2<<8) | … | (2<<LARGE_SYM_COUNT_OFFSET)`,
  igzip_inflate.c make_inflate_huff_code_lit_len). Cannot be one store into u16 slots. ✓
- `MOVDQU xmm1, [copy_start]` (asm:591,605,611,618,621) reads a flat u8 source
  `next_out − dist`. Cannot read a u16 ring. ✓
- slop headroom: `sub end_out, OUT_BUFFER_SLOP` (asm:488), `cmp next_out, end_out`
  (asm:511-512) — unchecked over-write within the slop of a LINEAR buffer. ✓
- window lower bound = `start_out` (`cmp copy_start, [rsp + start_out_mem_offset]`, asm:589;
  C mirror `state->next_out - look_back_dist < start_out`, igzip_inflate.c:1693) — the window
  IS the tail of the same linear buffer. ✓

So "igzip-CLASS is unachievable on the u16 ring" is a **code-level structural fact**, and it
is the same wall the round-2 bench measured (E2 = SIMD copy on the still-u16 ring → +13%
plateau). UPHELD.

### 2b. rapidgzip REALLY delegates its clean tail to ISA-L (Arm A's premise) — **CONFIRMED.**
Verified in vendor (not trusted from the memory): `GzipChunk.hpp:521-522`
`if ( cleanDataCount >= deflate::MAX_WINDOW_SIZE ) return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(…)`;
default coding `deflate.hpp:175 using LiteralOrLengthHuffmanCoding = HuffmanCodingISAL` under
`#ifdef LIBRAPIDARCHIVE_WITH_ISAL`. So rapidgzip's REAL fast path, when ISA-L is present, IS a
two-phase markered-prefix→ISA-L-clean-tail handoff. The report's Arm A premise is factually
correct.

### 2c. Does the report EVADE the governing memory? — **NO. It engages honestly.**
The governing memory (`project_faithful_unified_decoder_over_perf`) is explicit: the CHOSEN
blueprint is the **no-ISA-L** build = ONE `deflate::Block` decoding the whole chunk; and
"Reverting the clean tail to a 2nd engine … for speed IS the 600-commit Divergence #2 — do
not." The report does NOT hide behind its "our-asm not C-FFI" re-reading. It states plainly
(report:116-121): "BUT it is a TWO-PHASE HANDOFF — exactly what the 2026-06-05 governing
memory named 'Divergence #2' … Arm A picks the with-ISA-L rapidgzip as the blueprint instead …
it directly contradicts the memory's chosen blueprint and its explicit prohibition," and the
recommendation (report:246-248) correctly says ratifying Arm A means **OVERRIDING** that
memory. That is the honest framing the charter demanded.

**Framing caution (not a flaw, but the supervisor should not skim past it):** Arm A's header
calls it "the MOST FAITHFUL port of rapidgzip's REAL structure." It is faithful to the
*with-ISA-L* rapidgzip — a DIFFERENT blueprint than the one the user/memory chose. The
load-bearing fact the "our-asm" wording risks obscuring: **an our-asm clean-tail kernel IS a
second engine on the clean tail.** The memory's prohibition is about a SECOND ENGINE for the
clean tail, and whether that engine is `decode_block`/`Inflate` or our-igzip-asm is
immaterial to the prohibition — Arm A is textbook Divergence #2. The report DOES say this; the
caution is only that "MOST FAITHFUL" could mislead a skimming reader into the opposite.

### 2d. Is Arm B under-rated / is Arm A really the harder-to-argue path? — **partially, see 1b.**
Arm B is the memory-faithful path (its own next step: 128 KB dual-view store,
setInitialWindow rotate+downcast). The report rates it narrow-miss largely on the
"% RING_SIZE per copy" cost, which §1b shows is overstated (gzippy already bumps a physical
pointer). Arm B's TRUE residual is wrap-handling at the 64 KiB boundary — bounded, benchable.
**Recommendation to the report's recommendation:** do not present Arm B's narrow-miss as
settled; it is the only path that keeps BOTH no-FFI AND the one-engine memory, and its actual
rate is unmeasured. If the user values the memory, Arm B earns an isolation bench before the
fork is resolved.

---

## ATTACK 3 — PORTABILITY + MAINTAINABILITY — **UPHELD (attack lands); the report's INTRINSICS RECOMMENDATION is OVERTURNED-TO-OPEN-RISK**

### 3a. Hot-loop + macro size — report is accurate.
Verified: hot loop asm:507-627 ≈ 120 lines; macros = `inflate_in_load` (235),
`inflate_in_small_load` (257), `CLEAR_HIGH_BITS` (299), `decode_next_lit_len` (322),
`decode_next_lit_len_with_load` (376), `decode_next_dist` (+ its `_with_load`) — ~6 macros,
no stack frame in the hot path, SSE `MOVDQU` (xmm, NOT AVX2 ymm — confirmed every vector op
is xmm). The report's "~80–100 hot instructions, ~6 macros, SSE not ymm" is correct and a
genuinely useful finding (the "AVX2 build" differs mainly in BMI2 `SHLX/SHRX/BZHI`, not vector
width).

### 3b. Can idiomatic Rust + std::arch intrinsics reproduce igzip's speed? — **DOUBTFUL; the report's own Q1 + igzip's source argue NO.**
This is the sharpest landing of the attack. The report recommends the intrinsics route
(report:236-240) on the theory that "igzip's speed lives in the table format + output model +
pipelining, NOT in hand-scheduled instructions, so a faithful intrinsics port should reach the
same class." **igzip's own source is the counter-existence-proof.** igzip ships TWO decoders:
- the C inflate (`igzip_inflate.c:1641-1642`): `*state->next_out = next_lit; state->next_out++;`
  — ONE literal at a time, scalar `memcpy`/`byte_copy` back-ref (igzip_inflate.c:1697-1701,
  byte_copy:126-132).
- the hand-written asm (`igzip_decode_block_stateless.asm`): the speculative 8-byte packed
  store (asm:518), branchless preload of next lit/len AND next dist before the type is known
  (asm:524-552), `MOVDQU` overlap-doubling copy (asm:591-627).

Both consume the SAME packed multisym table. If a compiler could emit the asm's speculative
software-pipeline from the high-level form, igzip would not maintain a separate 22 KB asm file
— the C path would already be igzip-class. It is not (it is the fallback). The specific
patterns at risk under LLVM-from-intrinsics: (a) the UNCONDITIONAL 8-byte over-store advanced
by the actual count — expressible with raw-pointer `write_unaligned` (no bounds check) but
LLVM may refuse to keep it branch-free; (b) the cross-branch speculative preload (load the
NEXT symbol before the EOB test, asm:539-552) — this is a manual software-pipeline LLVM is not
obligated to reconstruct. **Verdict: the intrinsics route is a real OPEN codegen risk, not a
maintainability win to be banked.** The honest statement for the supervisor: the `asm!` route
is the safer bet for actually reaching igzip-class (it is the only thing PROVEN to hit the
rate — it is igzip's own choice); the intrinsics route MIGHT match with far less unsafe
surface but must be de-risked by a small codegen spike (write the hot loop both ways, diff the
emitted asm + microbench) BEFORE committing the route. The report states this as a settled
recommendation; it should be a gated experiment.

### 3c. Resumability bridge — adequately costed for Arm A, correctly flagged harder for Arm B.
Verified igzip is not per-symbol resumable: the fast loop bails to a careful tail
(asm:637-703, `rep movsb`) and uses `write_overflow_*`/`copy_overflow_*` spill slots
(asm:711-712; C copy_overflow at igzip_inflate.c). The report's mitigation — "confine the
kernel to the post-flip clean tail where a chunk-at-a-time call suffices" (report:160-161) —
is SOUND for Arm A: the whole chunk's input+output are present, so resumability is needed only
at the chunk boundary, which the careful tail + spill slots already handle. For Arm B (one
ring engine that callers resume mid-chunk) it is genuinely harder — consistent with the report
rating Arm B the harder integration. Not under-costed.

### 3d. Table-build "~300 LOC clean" — right ballpark, real but bounded complexity.
Verified `make_inflate_huff_code_lit_len` spans igzip_inflate.c:387-599 = 213 lines (plus the
dist builder + shared helpers ≈ the report's ~300). It is pure scalar C and transliterates,
BUT it carries non-trivial logic: single/pair/TRIPLE symbol packing (nested
`count_total`-indexed loops), the `multisym` flag gating, the long-code split
(`short_code_lookup[first_bits] = long_code_lookup_length | … | LARGE_FLAG_BIT`), and the
`index_to_sym(513)→512` quirk. The report's "transliterates directly to safe Rust … New
tables, new build, new tests" is fair but mildly under-sells the test surface (the packed
format is the load-bearing speed source per Q1, so its correctness is co-critical with the
kernel). Minor.

### 3e. "asm is the easy ~20%, integration the hard ~80%" — **CORRECT direction.**
Given (3b) puts the kernel's RATE realization itself at risk (route choice unresolved) and
(2)/(1b) put the host-buffer model (Arm A handoff seam vs Arm B ring rewrite) as the dominant
uncertainty, the report's split is right: the asm transcription is bounded and well-understood;
the integration (table format, resumability boundary, the handoff seam or ring rewrite,
byte-exact flip-seam tests) is where the risk and effort live.

---

## SUMMARY TABLE

| Attack | Verdict | Deciding citation |
|---|---|---|
| 1 — rate projection | **UPHELD-WITH-CORRECTION** | §3 re-bind robust to engine-rate miss (decode stops binding at any rate >~120 MB/s, `tier1-design-v2.md:242-251`); but Arm A's 283 anchor is ISA-L's OWN rate (circular) and the "0.54–0.66 consumer floor" is decode/placement-COUPLED per STEP-0 (`project_pregate`, serial floor 0.015 s + writev 0.245 s). Conclusion stands. |
| 1b — Arm B discount | **UPHOLD-DIRECTION / SOFT MAGNITUDE** | ring is NOT `% RING_SIZE` per copy — gzippy already bumps a physical pointer (`marker_inflate.rs:1751,1789`); tax is wrap-handling per 64 KiB. Arm B possibly under-rated; bench it. |
| 2 — faithfulness | **UPHELD** | u16-forbids-kernel is a code fact (asm:518/591/589); rapidgzip-with-ISAL really delegates (`GzipChunk.hpp:521-522`, `deflate.hpp:175`); report engages the memory honestly (report:116-121). Caution: Arm A = a second clean-tail engine = Divergence #2 regardless of "our-asm." |
| 3 — portability | **UPHELD; intrinsics recommendation → OPEN RISK** | igzip ships C(slow, igzip_inflate.c:1641 one-literal) AND asm(fast, asm:518 packed store) on the SAME table ⇒ the compiler does NOT auto-produce igzip-class; the route must be de-risked by a codegen spike, not recommended outright. |

## IS THE CORE CONCLUSION SOUND? — **YES.**
Conditional-feasible; **Arm A is the only path that projects to a tie, and only if placement
is ALSO closed** (engine alone ≠ tie — co-primary, `project_pregate`); the three-way fork
(1.0× tie vs no-C-FFI vs one-engine memory — faithful pick-two) is a real, correctly-surfaced
user-level decision. **No FATAL error** that would mislead the supervisor or user.

## THE TWO THINGS TO FIX BEFORE RATIFICATION (neither overturns the verdict)
1. **Demote the "prefer intrinsics" recommendation to a GATED codegen spike.** igzip's own
   C/asm split is direct evidence the compiler won't auto-emit the speculative pipeline; the
   `asm!` route is the only PROVEN path to the rate. Decide the route by measuring both on the
   hot loop, not by a maintainability preference (report:236-240).
2. **Do not dismiss Arm B by projection.** Its "ring tax" is wrap-boundary handling per 64 KiB
   (`marker_inflate.rs:1751`), not per-copy modulo; it is the ONLY arm that keeps both no-FFI
   AND the one-engine memory. If the user weights the memory, Arm B earns an isolation bench
   before the fork is resolved — the 150–200 MB/s band is presently an un-measured guess.

Add to the report's honesty ledger: the Arm A tie verdict is actually MORE robust than the
report claims — it survives a 30% engine-rate shortfall because the ~0.54 s floor sits far
above the decode-optimal (~0.19–0.32 s). The report buried its own best argument.

— Independent disproof advisor, read-only, all vendor file:line verified first-hand.
