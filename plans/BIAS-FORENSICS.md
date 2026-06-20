# BIAS-FORENSICS — what the Claude transcripts ACTUALLY show this campaign did

Mined from the on-disk Claude conversation record (not training-data priors):
`/Users/jackdanger/.claude/projects/*gzippy*/` — 1,509 `.jsonl`, 764 MB. This report
draws on the 7 "spine" main-thread sessions (2026-05-27 → 2026-06-20), 6,186 assistant
text turns. Method: jq/python extraction of assistant prose, regex bias signatures,
then hand-verification that each flagged claim was later reversed (the reversal is the
proof it was a bias, not a finding). Quotes are real, with file-prefix + timestamp.

## HEADLINE NUMBERS (the empirical fingerprint of the bias)

Across the 7 spine files (6,186 assistant turns):

| Signal | Count | What it means |
|---|---|---|
| Conclusion-language sentences ("the lever is / this means / the cause is / settled / dead / floor / proves / confirms / definitively") | **533** | rate of narrated certainty |
| Reversal-language sentences ("falsified / disproven / phantom / actually / I was wrong / TIE / not a win / doesn't reproduce") | **1,127** | rate of *undoing* prior certainty |
| Overclaim words ("breakthrough / finally / nailed it / massive / decisive") | **155** | hope-as-evidence |
| The word **"finally"** alone | **108** | premature-closure tic (see §2) |
| Distinct **"the lever is X"** subjects asserted | **74** | mutually-incompatible root causes |

**The single most damning stat: reversals (1,127) outnumber conclusions (533) ~2:1.**
The campaign spent more sentences un-saying findings than saying them. And **74
different things were each named "the lever"** — they cannot all be the lever; most were
inference-as-conclusion that the next measurement killed.

---

## 1. THE EMPIRICAL BIAS PROFILE (ranked by cost, with receipts)

### BIAS #1 — INFERENCE-AS-CONCLUSION (the master bias; cost: most of the campaign)

The assistant repeatedly converted a *reasoned attribution* into a stated *finding*,
then had it killed by the next run. This is the one CLAUDE.md's preamble names — and
the transcripts prove it fired dozens of times, with contradictory verdicts:

- **74 distinct "the lever is X"** subjects across the spine. A sampling of the
  mutually-exclusive ones actually asserted as fact:
  - "the lever is **buffer warmth**, and lever A is dead" — `52ac30bc` 2026-05-29T20:53
  - "the lever is the **allocator/chunk pipeline, not the inflate inner loop**" — `33517091` 2026-05-28T17:29
  - "the lever is **BMI2/AVX2 SIMD, not scalar codegen**" — `33517091` 2026-05-28T07:37
  - "dominant lever **finally correctly located**: the **inner Huffman decode loop**" — `a7df0aa6` 2026-06-05T06:13
  - "the lever is **window-key matching**" — `c0f5ffa8` 2026-06-06T16:17
  - "the lever is **unambiguously the publish chain**" vs "the lever is **structural placement**" vs "the lever is **decode speed**" — `c0f5ffa8` (same session, competing)
  - "the lever is the **opposite — make the bootstrap fast**" — `52ac30bc` 2026-05-30T00:41 (directly reverses the earlier "eliminate the bootstrap")
- The assistant eventually *named its own pattern* in later turns — the regex caught
  lever-subjects like *'"the lever is X" conclusions that were untested inferences'* and
  *'"the lever is X" is literally unreachable without a perturbation'* (`37b843ea`,
  2026-06-13). That self-recognition is real progress, but note it took ~74 assertions
  to arrive.

**Proof these were biases, not findings:** each was contradicted within the same or next
session. E.g. "buffer warmth"/segmentation was declared the master lever 2026-05-29,
then: *"**Segmentation is FALSIFIED** … SEG 1332 vs BASE 1653 MB/s = −19.4%"*
(`52ac30bc` 2026-05-29T23:15) — within hours.

### BIAS #2 — DISPROOF AMNESIA / CONTEXT-LOSS ACROSS RESTARTS (cost: near-repeat of multi-hour work)

The most concrete, expensive instance in the whole record — the assistant caught itself
about to rebuild something it had already measured dead:

- *"I was about to rebuild a falsified lever."* — `52ac30bc` 2026-05-30T00:51
- *"I was about to spend hours rebuilding something I'd already disproven."* — `52ac30bc` 2026-05-30T00:54

And the cold-subagent version of the same disease:
- *"**~20 cold subagent spawns were the engine of the disaster** — each re-derives
  context cold, doesn't know the tracer flag is dead, repeats the setup error (**two
  agents got opposite signs on the same oracle**)."* — `37b843ea` 2026-06-13T23:18

This bias is NOT cleanly in the generic CLAUDE.md list (which frames over-banking as the
hazard). The transcripts show the *opposite-and-equal* hazard too: hard-won **disproofs
evaporate** on restart, so dead levers get re-attempted. 291 turns touch
spawn/subagent/budget/orphan — the multi-agent fan-out amplified amnesia.

### BIAS #3 — PREMATURE-CLOSURE / HOPE-AS-EVIDENCE (the "finally" tic; cost: chronic)

108 uses of "finally" and 155 overclaim words. The pattern: narrating that *this* run
is the decisive one BEFORE it returns — emotional closure substituting for a gated
result. Receipts:

- *"This single A/B — **finally** runnable trustworthily — ends the d_w-vs-chain
  ambiguity."* — `d5aa0eb3` 2026-06-01T04:07
- *"the **dominant lever correctly and finally located**"* — `a7df0aa6` 2026-06-05T06:12
- *"This is it — grounded, confirmed, textbook."* — `52ac30bc` 2026-05-31T02:16
- *"**Breakthrough** … overturns 5 prior A/Bs"* — `d5aa0eb3` 2026-05-31T16:14

Each "finally" was followed by more sessions on the same question. The d_w-vs-chain
("decode speed vs publish chain") ambiguity declared "ended" on 2026-06-01 was STILL
being adjudicated on 2026-06-06 (`c0f5ffa8`: "this resolves the central question the
whole campaign has circled"). The tic is the giveaway: when you feel "finally," that is
the bias firing, not the evidence arriving.

### BIAS #4 — "FLOOR / IRREDUCIBLE / DEAD" DECLARED TOO EARLY (cost: nearly abandoned real wins)

Confident terminal verdicts that a later measurement reopened:

- *"Multiple optimization attempts **confirm this is the floor** without much deeper
  restructuring."* — `33517091` 2026-05-27T23:26 — yet the campaign kept finding wins for
  three more weeks.
- *"the remaining 14% faults are **irreducible** first-touch"* / *"lever B (allocator) is
  **largely exhausted**"* — `3ed8f25e` 2026-05-28T20:18 — later reopened: *"This
  **reopens lever B** with a new, un-falsified angle"* (same file, 2026-05-28T20:20).
- *"cheap levers exhausted"* declared (per MEMORY) **3×**, advisor each time named a
  paying lever — corroborated in the tail: N38/N39 "cheap removals exhausted" preceded
  NIGHT40's clean +1.4–1.5% win (`37b843ea` 2026-06-20).

CLAUDE.md's rule "a TIE is not a refutation of the direction" exists precisely because
this bias kept declaring directions dead on one cell.

### BIAS #5 — UN-SELF-VALIDATED INSTRUMENTS TRUSTED AS ORACLES (cost: a whole session of phantom signal)

Gate 0 violations that the record shows actually happened and went unnoticed:

- The **inert-oracle** disaster, root-caused only after it had misled the run:
  *"The irreproducibility is root-caused, definitively: `GZIPPY_SEED_WINDOWS` is a **file
  path**, not a boolean — the old script set it to "1", so `File::open("1")` failed →
  `hits=0 misses=50`, the oracle **never fired**."* — `37b843ea` 2026-06-13T23:27. The
  instrument was silently measuring the normal path the whole time.
- **Fulcrum contradicting itself** and being believed on both sides until an
  orchestrator caught it: *"Fulcrum contradicts itself — `fulcrum causal` says gzippy is
  90.5% window-absent … `fulcrum model` says gzippy is 0% window-absent and rapidgzip is
  the 97.4% one."* — `c0f5ffa8` 2026-06-06T16:54.
- **Mislabeled corpus** producing a fake closure: *"The 'bignasa closed at 1.018x' claim
  is dead: forensics showed the sched worker measured **silesia mislabeled as
  bignasa**."* — `37b843ea` 2026-06-12T00:24.
- **File-sink phantom sign-flip** caught: *"With a free sink (/dev/null):
  `consumer.write_data` collapses 137ms → 0.1ms … kills the output-write lever, saving
  the 16th TIE."* — `d5aa0eb3` 2026-05-31T07:42 (the file-sink had manufactured a
  phantom lever — the "16th TIE").

### BIAS #6 — OVER-CORRECTION (flipping to the opposite conclusion just as confidently)

Not just being wrong, but swinging hard the other way with equal certainty:

- *"the lever is the **opposite** — make the bootstrap **fast** … not eliminate it"* —
  `52ac30bc` 2026-05-30T00:41, after a session arguing to *eliminate* the bootstrap.
- *"**Breakthrough — and it overturns the prior conclusion.** Slowing only the bootstrap
  by 2× made the wall +30% slower … This **overturns 5 prior A/Bs**"* — `d5aa0eb3`
  2026-05-31T16:14. Five prior gated A/Bs reversed by one new one, immediately believed.
  (To its credit the next turn says "let me get an advisor before I over-claim it" — the
  antidote firing.)

### BIAS #7 — ATTRIBUTION-AS-VERDICT / DECOMPOSE-AND-BLAME instead of perturb (cost: the "decompose-a-slice" loop)

Naming a region as "the gap" from a decomposition rather than a removal-oracle:

- *"the gap is the clean-decode rate, and my span analysis **pinned it** to the per-block
  re-entry"* — `a7df0aa6` 2026-06-04T20:24, followed by more competing pins.
- *"the high-T wall is **not** an architectural ceiling … the diagnosis is a real
  breakthrough and it stands"* — `a7df0aa6` 2026-06-04T22:25 — a decomposition claim
  asserted as a standing verdict.
- The phrase "decompose-a-slice-and-shave-it loop" is the campaign's own name (CLAUDE.md)
  for where these went — repeated TIEs.

---

## Which GENERIC biases ACTUALLY fired (with proof) vs which are BLIND SPOTS

**Generic ones in CLAUDE.md/MEMORY that the receipts CONFIRM fired:**
inference-as-conclusion (#1, 74 levers), premature-closure (#3, 108 "finally"),
over-correction (#6), attribution-as-verdict (#7), un-self-validated instruments (#5,
SEED_WINDOWS/Fulcrum/mislabel/file-sink — all four named gate-0 traps observed),
single-arch over-generalization (the tail shows "Intel-only, AMD owed for LAW" on every
recent win — confirmed live).

**BLIND SPOTS — observed patterns NOT well-covered by the standard list:**
1. **Disproof amnesia across restarts (#2)** — the standard list warns about *over*-banking
   disproven claims; the receipts show the symmetric, costlier failure: disproofs
   *vanishing* on context reset, so dead levers get re-built ("I was about to rebuild a
   falsified lever"). The fix isn't "bank less," it's "make disproofs survive a cold
   start as executable falsifiers."
2. **Multi-agent fan-out as a bias *amplifier*** — 291 spawn-related turns; "~20 cold
   subagent spawns were the engine of the disaster"; "two agents got opposite signs on
   the same oracle." Spawning more cold agents multiplied #1, #2, #5 rather than
   diluting them. (MEMORY's later "ONE leader, supervisor doesn't do hands-on" rule is
   the scar tissue from this — and it is correct.)
3. **The "finally" emotional tell** — a *linguistic* early-warning the rules don't name.
   108 instances. It is the cheapest tripwire available: grep your own draft for
   "finally / this is it / breakthrough / nailed" before sending.

---

## 2. THE META-PATTERN and the antidote the record shows actually worked

**Underlying habit:** the assistant is fluent and *narrates a coherent story in real
time*. Coherence feels like truth, so a plausible attribution gets written as a finding,
emotional closure ("finally") gets written as evidence, and on restart the *story*
survives in prose while the *disproofs* (which lived in scattered measurements) are lost.
The 2:1 reversal:conclusion ratio is the signature: a narrator generating verdicts faster
than reality can validate them, then spending the next turn retracting.

**The antidote the transcripts show genuinely working** (these turns are followed by
*correct* outcomes, not reversals):

1. **The empirical arm beats every code-read and every guess.** *"Both my guess and the
   vet's code-inference were wrong; only the empirical isal arm settled it — which is
   precisely the gate's value."* (`37b843ea` 2026-06-13T18:24). Whenever an *A/B with a
   confirmed-firing perturbation* was the source, the verdict held. Whenever a
   span-decomposition or source-read was the source, it got reversed.
2. **Adversarial advisor BEFORE banking** caught real errors: *"the disproof-ledger
   methodology … caught three misdiagnoses today (a phantom seeding bug, a backwards LUT
   root-cause, a serialized-apply hypothesis killed by its own trace)"* (`37b843ea`
   2026-06-10T02:13); and in the tail the advisor *"caught me circling (8 leaders on the
   same fork) and killed my planned chunk-sweep as confounded"* (2026-06-20T12:42).
3. **The /dev/null sink law + interleaved best-of-N** dissolved the phantom sign-flips.
4. **Writing the falsifier's MECHANISM into the commit message** (not just prose) is the
   only form of disproof that survived a cold restart.

The pattern: **the bias is killed only by handing the verdict to deterministic software
(a firing perturbation), never by better reasoning.** Every time the record substitutes
reasoning, it reverses; every time it substitutes a self-validated A/B, it holds.

---

## 3. THE CORRECTED CLEAN VIEW — where the project ACTUALLY stands

Reading past the prose to only the gated results (most recent first):

### Genuinely GATED / KNOWN (trust, after a quick re-verify against current sha)
- **gzippy-isal T1 BEATS rapidgzip 15–48%** on silesia (two-binary matrix, Intel,
  2026-06-16). The T1 goal is *met on the isal build.*
- **gzippy-native T1 loses ~12–24%** to igzip/libdeflate; the residual is **inner-kernel
  instruction count, NOT architecture.** This is the best-supported current verdict:
  N33 measured gz retires **+2.6 instr/B more than igzip at near-equal IPC (2.53 vs
  2.60)** — so it is genuinely an instruction-surplus problem, and the surplus is gz's
  *resumable/reclass speculative-parallel kernel machinery* (+3.40 instr/B vs igzip's
  `_04` monolithic serial loop). Streaming-parity and table-build were both removal-tested
  and ruled OUT as the lever (N38/N39 table-build = dead code; gating it was byte-exact
  but *regressed* the wall).
- **NIGHT40 (2026-06-20): a real, gated WIN** — hoisting the d0 half of the D-1 anchor
  in the inner kernel: byte-exact, production-wall CI-disjoint on BOTH corpora, p<0.01,
  ~1.4–1.5% of the gap closed. The byte-exact differential caught a flags-clobber bug
  mid-attempt (the test infra worked). This is the only fully-modern gated win and it
  validates the "instruction-surplus in the resumable kernel" direction by *moving the
  wall when reduced.*
- At **T≥2 both native and isal LOSE** (~0.77 squishy@T4); kernel-swap moved the
  multi-thread wall ~0 and P_gz≈P_rg — so the T≥2 deficit is the **SHARED parallel
  pipeline, not the clean kernel.** This is a HYPOTHESIS (gated on one un-run
  discriminator), not yet a located lever.

### Still NARRATIVE (do not trust as state — re-measure if live)
- Any single "the lever is X" from the 2026-05-27 → 06-06 era (allocator, buffer-warmth,
  window-key, publish-chain, placement, prefetch-depth, etc.). All are in MEMORY's
  CAUTIONARY DISPROVEN section for good reason — 74 competing claims, mostly self-refuted.
- "Cheap levers exhausted" — asserted ≥3× and falsified ≥3×. Treat any future instance as
  the bias firing until an advisor sweep disagrees.
- The "true cause is instruction-count" is *by elimination* + N33 IPC data — strong, but
  note "by elimination" is still partly inferential; the NIGHT40 win is what upgrades it
  toward a perturbation-backed verdict.

### PROVEN DEAD-ENDS — do NOT re-attempt (each has a gated falsifier)
- **Segmentation / SegmentedU8 128KiB memory model as a speed lever** — −19.4%, faults
  did not drop (`52ac30bc` 2026-05-29). (Already nearly-rebuilt once; do not rebuild.)
- **rpmalloc/global-allocator + Z-prewarm/MADV_POPULATE_WRITE** — serializes
  previously-parallel faults; rpmalloc hard-caps ~3.94 MiB (`3ed8f25e`).
- **BMI2 PEXT/PDEP hand-dispatch & scalar hand-asm inner loop** — rustc already emits
  BZHI; at parity (`3ed8f25e`/`33517091`). (Zen2 caveat: PEXT microcoded — separate.)
- **SIMD multi-literal extending packed store to N>3** — icache+mispredict cost >
  literals saved (`33517091` 2026-05-28).
- **Table-build path** as the T1 lever — byte-for-byte already igzip's; the one delta is
  dead code; gating it regressed the wall (N38/N39, 2026-06-20).
- **Replay/known-window/clean-window oracles as a ceiling tool** — 3× VOID; the original
  clean-window oracle silently re-ran the bootstrap (the broken instrument that licensed
  a whole wrong strategic phase).

### THE 1–3 HIGHEST-VALUE NEXT ACTIONS (bias-corrected)
1. **Continue the NIGHT40 inner-kernel restructure — it is the ONLY direction with a
   live perturbation-backed win.** The p0 half of the D-1 anchor is blocked by the N32
   trap (refill destroys `bc`), needing an exit-reclass contract change. This is the
   funded "heroic restructure." Keep each step byte-exact + production-wall gated +
   advisor-checked-for-circling. ROI is bounded but *real and demonstrated.*
2. **Build the one un-run discriminator for the T≥2 deficit** (ISA-L vs pure-Rust clean
   cyc/byte in the shared pipeline). The T≥2 loss is the largest *un-located* gap and the
   matrix says it lives in the shared pipeline, not the kernel — but it is HYPOTHESIS
   until that discriminator runs. Highest VoI for opening new ground.
3. **Pay the AMD/Zen2 replication debt on the kept Intel wins** (dist-preload,
   NIGHT40, the two-binary matrix). Every recent win is "Intel-only, NOT-YET-LAW." One
   AMD run converts a stack of HYPOTHESES into LAW — cheap, and it is the Gate-3 the
   campaign keeps deferring.

**The discipline that the receipts say matters most:** stop writing "finally / the lever
is / this proves" — when you feel that sentence coming, it is bias #1/#3 firing. Hand the
verdict to a firing perturbation, write the mechanism into the commit, and let the
advisor check for circling before banking. That combination is the only thing in 6,186
turns that consistently produced verdicts that did NOT reverse.
