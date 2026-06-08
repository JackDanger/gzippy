# Phase-2 inline-asm prototype — INDEPENDENT DISPROOF VERDICT

**Role:** independent disproof advisor. Mandate: try to BREAK the owner's NO-GO, not rubber-stamp it.
**Inputs read first-hand:** the brief; `benches/engine_isolation.rs` (VAR_VI 742-997, VAR_VII 1119-1412,
`run_careful_tail` 1053-1115, `emit_one_backref` 1004-1051); `vendor/isa-l/igzip/igzip_decode_block_stateless.asm`
(`loop_block` 507-560, back-ref copy 595-630); `plans/CAMPAIGN-CHARTER.md` (USER DECISION 2026-06-08 @835-858,
USER DECISION 2026-06-07 @727-733, PHASE-0 RESULT @748-762).

## BOTTOM LINE: **OWED-MORE.** The NO-GO is NOT earned. The falsifier is under-powered and the prototype is mis-described.

The escalation fork (accept ~0.86× pure-Rust **or** authorize a multi-session full-kernel asm) is **NOT correctly
triggered.** What is owed is the cheaper, in-scope thing the charter's Phase-2 actually names — a **full-kernel asm
isolation prototype** — which was never built. Until that runs, "even hand-asm can't beat LLVM" is unproven.

---

## The kill-shot: VAR_VII does not measure asm-vs-LLVM. It measures (tiny asm prefix + careful per-symbol tail) vs (VAR_VI full speculative fast loop).

The brief describes VAR_VII as: *"the asm covers the literal RUNS between back-refs; each back-ref is an asm→Rust
handoff that … re-enters the asm loop"* (brief lines 36-38). **The code does not do this.** Traced through
`decode_var_vii` (engine_isolation.rs:1173-1408):

1. Per DEFLATE block, the asm loop (1245-1336) runs **once**, from the block start, over the leading short-literal
   run. It exits on the **first** length/EOB/long code or boundary (exits 7f/8f/9f → exit_code 1/0/2).
2. On exit_code 0 the Rust block (1357-1387) emits the packet's leading literals and resolves **one** back-ref via
   `emit_one_backref`.
3. Control then falls into `run_careful_tail` (1391-1403), which decodes **the entire remainder of the block** with
   the per-symbol careful loop (`litlen.decode` at 1067/1113) until EOB or `target_end`.
4. **There is no re-entry into the asm within a block.** After `run_careful_tail` returns (1403), the only paths are
   `break 'blocks` or looping to `read_header` for the **next** block (1405-1407). The asm is re-entered only at the
   *start of the next block*.

So the asm covers the leading-literal prefix of each block — on silesia (~31% back-refs) that is a geometric run with
mean ≈ 0.69/0.31 ≈ **2–3 symbols** — and the Rust **careful per-symbol tail** carries essentially the whole block.
The brief's "re-enters the asm" handoff **never happens**; the real fallback is "careful-tail-eats-the-rest-of-the-block."

Now compare the baseline. VAR_VI (`decode_var_vi`, 742-997) runs its **`'fast` speculative loop** over the *whole*
block — packed u64 literal store (831), BMI2 dist decode (864-879), `avx_backref_copy` (888), back-refs handled
**in-loop** with `continue`, dropping to its careful tail only at boundaries (`break 'fast`, 857). VAR_VI never
abandons the fast loop mid-block.

**Therefore the measured contrast is:**
- VAR_VI = LLVM **full speculative fast loop**, whole block.
- VAR_VII = **asm for ~2–3 leading symbols** + **careful per-symbol tail** for ~99% of every block.

VAR_VII being uniformly ~0.04-0.05× *below* VAR_VI is fully and mechanically explained by the careful-tail penalty:
VAR_VII replaces VAR_VI's fast loop with a slower per-symbol loop over nearly all bytes, plus a negligible asm prefix.
**This says nothing about asm-vs-LLVM and everything about the fallback design.** The asm was barely on the measured
path; the 155 MB/s number is essentially `run_careful_tail`'s rate, not the asm's.

This is a Rule-4 instrument-validation failure (CLAUDE.md PROCESS #4 / charter §NON-NEGOTIABLE): byte-exactness was
validated, but **the instrument never validated that the asm carried the load.** There is no coverage counter for
"output bytes emitted by the asm vs by `run_careful_tail`." Given the control flow, that fraction is tiny — so the
falsifier is measuring the wrong code.

---

## What a FAITHFUL transliteration is — and igzip proves the back-ref belongs in the asm

USER DECISION 2026-06-07 (charter:731): *"transliterating igzip's ACTUAL assembly instruction-for-instruction."*
igzip's hot loop is `loop_block` (asm:507-560), and the back-ref machinery is **inside it**:
- The next **distance** code is gathered **speculatively before the lit-vs-len branch** (asm:550-552 `next_sym3`) — F2.
- A length code falls through to `decode_len_dist` (asm:559+) → the **MOVDQU overlap copy** (asm:600-627) — D — whose
  back-edges are `jle loop_block` (602) and `jmp loop_block` (627): **the copy returns into the same asm loop.**

So in igzip there is **no handoff and no per-symbol careful path** — dist decode, the SIMD copy, and the literal run
are one register-pinned machine with the back-ref copy looping straight back to `loop_block`. A faithful Phase-2
prototype of "igzip's hot loop" must keep F2 + D **in asm** with the back-edge returning to the asm loop. VAR_VII
implements **none** of that for the bulk: it omits F2/D from asm and routes every block's remainder to Rust. It is not
a faithful partial transliteration of the hot loop; it is a transliteration of only each block's opening literal run.

A faithful *partial* transliteration plateauing would be a legitimate finding **only if the partial scope were itself
the faithful/intended unit.** It is not. The intended unit (per the charter and per igzip's actual code) is the full
`loop_block` including back-refs. The falsifier is therefore **under-powered**, and the verdict owed is
"full-kernel prototype," not "NO-GO, escalate."

---

## Answers to the four brief questions

**Q1 — Is the PLATEAU/NO-GO earned, or a partial-scope confound?**
A **confound — and worse than the brief states.** It is not merely "asm literal-run only with per-back-ref handoff"; the
code hands **the entire rest of each block** to a careful per-symbol loop and never re-enters the asm intra-block
(1391-1403; no re-entry). The measured deficit is the careful-tail penalty, not an asm property. NO-GO is not earned.
This is *not* "the handoff is the honest finding" — because the honest faithful unit (igzip `loop_block`) keeps
back-refs in asm, so the handoff is a prototype shortcut, not a faithful structure that plateaus.

**Q2 — Should the gate require a full-kernel asm prototype before NO-GO? Is the falsifier under-powered as run?**
**Yes and yes.** The charter Phase-2 target is "prototype the inline-asm hot loop" (charter:848-853); igzip's hot loop
is the whole `loop_block` (F2 dist-gather + D MOVDQU copy + back-edge into the loop). A full-kernel **isolation** bench
prototype (a VAR_VIII in `engine_isolation.rs`, *not* production integration — that stays Phase-3) is the correct,
still-cheap gate. The falsifier as run cannot reject the full kernel because (a) the asm's literal-run contribution was
never isolated, and (b) F2/D — igzip's largest divergences from LLVM codegen (speculative cross-branch dist gather,
branchless overlap MOVDQU) — were never in asm at all.

**Q3 — Is "hand-asm slower than LLVM" an indictment of asm, or of the partial-handoff design?**
**The design.** The slowdown is mechanically attributable to substituting `run_careful_tail` (per-symbol) for VAR_VI's
fast loop over ~99% of bytes. "VAR_VII < VAR_VI" is the *expected* output of that substitution and carries **zero**
information about whether a register-pinned full-kernel asm beats LLVM. It specifically does **not** prove "asm can't
help."

**Q3b — Is VAR_VI a valid LLVM ceiling, and is ~0.55-0.60× ISA-L consistent with the 0.945× whole-system ocl_cf?**
VAR_VI is a valid LLVM ceiling baseline (BMI2 BZHI/SHRX, AVX2 MOVDQU copy, Rust-level preload all present —
742-997). The ~0.55-0.60× isolation ratio is **consistent** with the 0.945× whole-system ocl_cf, because they are
different bases: the bench is single-thread pure clean symbol-rate with no scheduler/output, whereas the whole-system
wall amortizes the engine across ~8 threads where the charter's own PHASE-0 RESULT (charter:748-755) found the engine
rate is **slack-masked at T8** (seeded pure-Rust ties). A 0.55× isolation engine rate coexisting with a 0.945×
whole-system wall is not a contradiction. **Caveat (flagged, not load-bearing for this verdict):** that same Phase-0
finding means even a *successful* full-kernel asm has a known-small whole-system payoff (it is the T1 lever, not the
T8 binder; T8 is the window-absent marker path + output floor). The full-kernel prototype should therefore be gated on
a whole-system payoff hypothesis, not merely "match igzip's isolation rate."

**Q4 — Is SHA_ALL_EQUAL over 5 chunks vs 2 oracles + SELFTEST enough to trust the rate?**
Byte-exactness is **sufficient for the correctness gate** (it proves the bytes VAR_VII emitted are correct, vs both
scalar and ISA-L, and SELFTEST/`GZIPPY_VII_CAREFUL_ONLY` isolates the careful machinery). But it is **irrelevant to the
rate verdict**: correct bytes do not change the fact that the asm wasn't the dominant code path. The owed coverage is
**not** more SHA oracles — it is an **asm-coverage counter** (bytes emitted in-asm vs in `run_careful_tail`) to confirm
the obvious-from-the-source conclusion that the asm carried a negligible share. That instrument gap is the real
deficiency.

---

## OWED before any NO-GO or escalation (pre-register falsifiers)

1. **Full-kernel asm isolation prototype (VAR_VIII).** Faithful to igzip `loop_block`: F1 preload, F3 branchless
   refill, F4 register pinning, C packed store, **F2 speculative dist gather in asm**, **D MOVDQU overlap copy in
   asm**, with the copy's back-edge returning into the asm loop (asm:602,627) — **no Rust handoff, no careful tail for
   the bulk**; careful tail only at the genuine boundary slop, exactly as igzip exits to `end_loop_block`. Byte-exact
   gate unchanged. Falsifier: does VAR_VIII close a material fraction of the 36ms toward ISA-L (≥ ~0.75× as a
   pre-registered material bar)? PLATEAU only if VAR_VIII ≈ VAR_VI.
2. **Cheap attribution check (do first):** add an asm-vs-careful-tail output-byte coverage counter to VAR_VII and
   re-run; this confirms (in minutes) that the current 155 MB/s is `run_careful_tail`'s rate, retiring the current
   NO-GO without waiting for #1.

If VAR_VIII (full kernel, faithful) **also** ties/loses VAR_VI on the matched comparator, *then* the PLATEAU is earned
and the user-fork (accept ~0.86× or authorize multi-session production asm) is correctly triggered. **Not before.**

## Final: **OWED-MORE — full-kernel asm isolation prototype (VAR_VIII), + an asm-coverage counter on VAR_VII.** NO-GO is a confound of a mis-described partial scope; escalation is premature.
