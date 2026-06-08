# Phase-2 inline-asm prototype — INDEPENDENT DISPROOF VERDICT, PASS 2

**Role:** independent disproof advisor, pass 2. Mandate: try to BREAK the owner's NO-GO v2, not rubber-stamp it.
**Inputs read first-hand:** the updated brief incl. UPDATE 2; my pass-1 verdict; `benches/engine_isolation.rs`
(VAR_VII re-entry 1287-1532, `decode_one_symbol` 1062-1112, `run_careful_tail` 1119-1176, `emit_one_backref`,
the `GZIPPY_VII_COVERAGE` counter 1291-1546); `plans/CAMPAIGN-CHARTER.md` (ocl_cf removal-oracle @91-98, the
2026-06-08 localization @72-81).

## BOTTOM LINE: **NO-GO for the partial-handoff inline-asm approach is now EARNED.** The user-fork escalation is **correctly triggered.** No VAR_VIII prototype is owed to make the decision — the full-kernel's whole-system upside is *already bounded by a validated removal oracle (ocl_cf)*, not a slope.

This **reverses the "build VAR_VIII before you may NO-GO/escalate" half of my pass-1 verdict** — and I am reversing it
on new evidence the owner produced, exactly as PROCESS #7 requires (a rejection needs a mechanism, and the owner
supplied two: the coverage-vs-speed anti-correlation, and the back-edge structural argument). What survives from pass-1:
the *broad* claim "even hand-asm can't beat LLVM" is still unproven, because the full kernel was never built. What
changed: that no longer blocks the verdict, because the full kernel's payoff is bounded *above* by the ISA-L oracle we
already have, so building it cannot change the escalation decision.

---

## I tried to break the NO-GO three ways. All three failed.

### Break-attempt 1: "the per-symbol re-entry is a NEW confound — a smarter partial design avoids it." → REFUTED.

This was the live confound. The re-entry tax is real and it is NOT "asm vs LLVM": each `core::arch::asm!` block is an
LLVM optimization barrier. On every back-ref the code writes the 4 loop-carried regs back to the `bits` struct in memory
(engine_isolation.rs:1411-1414) and re-reads them on re-entry (1295-1298); at 300-460K invocations/chunk the fixed
per-invocation materialize/spill cost swamps the 2-3 symbols of literal work the asm does between back-refs. So in the
narrow sense the brief is right to worry — VAR_VII re-entry does not cleanly isolate the register-pinned fast loop.

**But it is not a *rescuable* confound, and that is the kill-shot for the partial family.** The owner's coverage counter
delivers the decisive datum my pass-1 demanded, and it points the other way from a rescue:

- **Bracket A (leading-run-only):** coverage 0.9%, rate **0.55× ISA-L** — asm barely runs, this is ~`run_careful_tail`'s rate.
- **Bracket B (re-enter-per-symbol):** coverage 57-99% (median ~0.65-0.74), rate **0.28× ISA-L** — asm carries the bytes, pays the tax.

The rate **falls monotonically as coverage rises** (0.9%→0.55×, ~70%→0.28×). That is a *causal* signature, not an
attribution: every additional symbol you push into the asm costs one more re-entry, so within the partial-handoff
family the objective is monotone-decreasing in coverage. **The two brackets sandwich the entire interior of the
partial design space and both lose.** There is no intermediate partial split that wins.

The brief's two hypothetical rescues both collapse under inspection:
- **"batch back-refs to amortize the asm-block tax"** — *not viable.* A back-ref copy must land in the output stream
  *before* the subsequent literals (output cursor + overlap/window dependency). You cannot defer copies while continuing
  to decode literals into the same region without reordering the output — which breaks the in-order output contract.
  Batching back-refs is therefore not a legal transformation here.
- **"keep dist in asm but long-codes in Rust"** — to keep the *dist decode* in asm AND avoid the re-entry tax you must
  keep the *copy* (D, the MOVDQU overlap) in asm too, so the loop back-edge stays *inside* the `asm!` block (igzip's
  `jle loop_block`/`jmp loop_block`, asm:602,627). The moment the copy is in asm and the back-edge stays in asm, it is
  **no longer a partial-handoff design — it is the full kernel.** "Dist-in-asm without the tax" is a synonym for the
  full kernel, not an intermediate.

So: the re-entry confound is honest, general (an `asm!`-barrier property, not a pass-1-style instrument bug), and the
coverage-vs-speed anti-correlation proves no interior partial design escapes it. **NO-GO for the partial-handoff family
is EARNED.** Note the brief scopes its v2 NO-GO correctly this time — "NO-GO for the partial-handoff inline-asm
approach," not the unprovable "asm can't beat LLVM."

### Break-attempt 2: "the only remaining path is a full-kernel asm = re-writing ISA-L by hand" — is that mis-characterized? → It is CORRECT.

I looked for a cheaper untested design between VAR_VII and a full ISA-L re-derivation. There isn't one. The remaining
design is precisely igzip `loop_block` in inline asm: F1 preload + F3 branchless refill + F4 register pinning (VAR_VII
has these) **plus F2 speculative dist gather in asm + D MOVDQU overlap-copy in asm with the back-edge returning into the
asm loop**, exiting to the Rust careful path only on the *rare* long-code / boundary / EOB. Structurally that is
re-deriving igzip's hot loop by hand in Rust inline-asm — multi-session, and correctness-hostile (the distance-doubling
overlap copy is the classic deflate-decoder bug site). The brief's characterization is accurate.

### Break-attempt 3: "the falsifier is still under-powered — VAR_VIII is owed before any NO-GO/escalate" (my own pass-1 position). → REFUTED by the ocl_cf removal oracle.

This is where I overturn my pass-1 demand. In pass-1 I said you may not declare NO-GO or escalate until a full-kernel
VAR_VIII is built and measured. **That demand is now moot, because the full kernel's *whole-system* upside is already
measured by a validated removal oracle — and a removal oracle is exactly what PROCESS #3 requires to bound a speed-up
(slope is not enough; you must REMOVE the region).** The charter already did the removal:

- **`ocl_cf` = GZIPPY_ISAL_ENGINE_ORACLE** swaps *real ISA-L FFI* into gzippy's clean engine. It is the engine
  fully removed/replaced. Charter:94: *"ocl_cf is the VALIDATED speed-UP ceiling (removal not slope)."* Result:
  **0.945× rg at T8** (charter:91).
- A full-kernel hand-asm's *best possible* outcome is to match ISA-L's engine rate. Its whole-system ceiling is
  therefore **≤ ocl_cf = 0.945×** — it cannot exceed the actual ISA-L engine, because ISA-L *is* the target it is
  transliterating.
- The 2026-06-08 localization tightens this further: swapping the clean engine to ISA-L "barely moves consumer_wall
  (0.3899→0.3851, **~5ms**) and does NOT shrink DECODE_WAIT" (charter:72), and `ocl_cf+seedfull = 0.983-0.985×`
  **still not 1.0×** — a ~6ms output-floor residual survives even full bootstrap removal (charter:77), a floor
  rapidgzip also pays.

**Therefore VAR_VIII is not owed to decide the fork.** Building it could only answer the narrower question "can a
hand-rolled Rust-inline-asm full kernel actually *reach* ISA-L's isolation rate, or does LLVM's register allocator
around the `asm!` block leave it stuck at VAR_VI's 168 MB/s?" — and the answer to *that* only matters *after* someone has
decided the bounded ~5ms / 0.945× T8 upside is worth capturing in pure-Rust. The upside itself needs no new prototype:
ISA-L (VAR_III isolation = 283 MB/s; ocl_cf whole-system = 0.945×) already is the measured ceiling for any engine,
hand-asm or FFI.

---

## Answers to UPDATE-2's three questions

**Q1 — Is the partial-handoff NO-GO now EARNED, or is per-symbol re-entry a confound a smarter partial design avoids?**
**EARNED.** The re-entry overhead is honest and general (an `asm!`-barrier tax, ~300-460K barriers/chunk), and the
owner's coverage counter shows rate *falling monotonically as asm coverage rises* (0.55× @0.9% → 0.28× @~70%) — the two
brackets sandwich the whole partial design space and both lose. The brief's candidate rescues don't exist: back-ref
batching is illegal here (breaks in-order output), and "dist-in-asm-without-the-tax" *is* the full kernel, not an
intermediate. This is no longer a pass-1-style instrument artifact; it is the genuine finding.

**Q2 — Is "the only remaining path is full-kernel asm = re-writing ISA-L by hand" correct, and is the user-fork triggered?**
**Both correct.** The remaining design is igzip `loop_block` in asm (F2 dist gather + D MOVDQU copy + back-edge into the
loop, Rust only on rare long-code/boundary) = a hand re-derivation of ISA-L's hot loop, multi-session and
correctness-hostile. The user-fork (accept ~0.86× pure-Rust vs authorize the full-kernel asm) is correctly triggered —
**and it is a cleaner trigger than the brief claims**, because the full kernel's payoff is already bounded by the ocl_cf
removal oracle, so the fork is a pure value judgment for the user, not a measurement gap.

**Q3 — Anything that makes a full-kernel asm worth the multi-session cost given the engine is slack-masked at T8?**
**At T8: essentially nothing.** The charter's own removal oracles cap it: ocl_cf 0.945× (engine fully ISA-L),
ocl_cf+seedfull 0.983-0.985× (engine + bootstrap removed), and a ~6ms shared output floor survives even that. Localized
T8 engine-swap delta ≈ 5ms — at/below the harness spread the overlap-writer already operates in. The engine is the **T1
binder, slack-masked ~8× at T8** (charter:72, and prior PHASE-0). So if the production target is T8-dominated, the
multi-session asm is **not** justified — accept ~0.86-0.945×.

**The one scenario that flips it — and the brief misses it:** the campaign goal is parity across archive × *thread-count*,
**including low T**, and goal #1 forbids FFI in the production decode graph. At **T1 the engine is NOT slack-masked** —
there the pure-Rust engine runs at 0.55-0.60× ISA-L and the engine rate is on the critical path. The ocl_cf escape hatch
(real ISA-L FFI) is *forbidden in production by goal #1*, so the **only pure-Rust route to ISA-L-class engine rate at low
T is the full-kernel hand-asm.** Hence the full-kernel asm is justified **iff** the goal matrix has an *open, closable,
engine-bound, low-T cell where gzippy loses to rapidgzip.* That — not the small T8 payoff — is the real decision input.

## The one thing owed before the fork RESOLVES (cheap, not a prototype)

Before the user picks "accept" vs "authorize asm," the owner must state, from the existing goal matrix (no new build):
**is there an open, closable, low-T (≈T1) cell where gzippy's pure-Rust wall loses to rapidgzip and the binder is the
clean engine rate (not output/scheduling)?**
- **If NO** (all low-T cells already TIE, or are output/scheduler-bound, or rapidgzip isn't ISA-L-class there): the fork
  resolves to **ACCEPT ~0.86-0.945× pure-Rust.** The full-kernel asm buys a removal-oracle-bounded ~5ms at T8, inside
  the spread — not worth a multi-session, correctness-hostile re-derivation of ISA-L.
- **If YES**: goal #1 (no FFI) makes the full-kernel hand-asm the **only** pure-Rust path to close that cell, and the
  multi-session cost is then justified *by the goal*, despite the small T8 payoff. In that case VAR_VIII is the right
  next build — but build it as the *production-bound* engine for the low-T path, gated on a pre-registered low-T wall
  falsifier vs rapidgzip (TIE-or-better on that cell), not on "match ISA-L isolation rate."

This is a one-paragraph answer from data the owner already has, not a new prototype.

---

## FINAL

- **VERDICT: NO-GO — EARNED — for the partial-handoff inline-asm approach (VAR_VII, both brackets).** The re-entry
  result is the honest disproof; coverage-vs-speed is monotone-adverse; no legal interior partial design exists.
- **User-fork escalation: CORRECTLY TRIGGERED.** And cleaner than the brief frames it: the full kernel's whole-system
  upside is already bounded by the ocl_cf removal oracle (≤0.945× at T8, ~5ms), so the fork is a value judgment, not a
  measurement gap.
- **NOT OWED-MORE on a prototype.** I withdraw my pass-1 "build VAR_VIII before you may escalate" — that demand is
  retired by the ocl_cf removal oracle (PROCESS #3 satisfied without VAR_VIII). VAR_VIII is owed **only if** the user
  authorizes capturing the bounded low-T upside in pure-Rust.
- **OWED before the fork RESOLVES (cheap):** the owner names whether an open, closable, engine-bound, *low-T* parity
  cell vs rapidgzip exists. That single answer decides accept-vs-authorize; absent such a cell, my recommendation is
  **ACCEPT ~0.86-0.945× pure-Rust** and close Phase-2.
