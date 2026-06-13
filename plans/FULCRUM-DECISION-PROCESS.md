# The Fulcrum Decision Process

How we turn Fulcrum/score numbers into work — without (a) picking a number and
optimizing it, (b) jumping to a conclusion, or (c) drowning in
coverage-completionism. This is the governing procedure for the gzippy
decode-parity campaign. It exists because every scar this campaign took came from
skipping a stage: a phantom 2105ms rg figure (no provenance check), a wheel-tax
comparator (no comparability check), a double-counted ledger and a clean-window
oracle that silently re-ran bootstrap (no instrument self-test), weeks of
lever-grinding (no causal-perturbation gate).

The shape is a **funnel with gates**, plus a **fast lane** for strictly-less-work
fixes and a **prioritization rule** so the funnel is entered in goal order, not
analyst order. A finding moves to the next stage only by passing that stage's
gate. Most candidate "levers" die at a gate — that is the point.

---

## The failure modes this prevents

1. **Premature optimization** — "crc_write is 54% of the wall, let's speed up
   CRC32." Attribution is not causation. Forbidden before Stage 4 *unless* it
   qualifies for the fast lane (strictly-less-work, no attribution belief needed).
2. **Jumping to a conclusion** — reading one ratio, an incomplete row, or a
   bimodal cell's *mean* and declaring a mechanism. Forbidden before Stage 2.
3. **Coverage-completionism** — filling the whole matrix / building tools for
   their own sake. Gaps and tools are pursued only when they block the current
   goal-deficit cell (see Prioritization).

---

## PRIORITIZATION — which gap/anomaly next (so the goal picks, not the analyst)

The prohibitions above ("biggest number forbidden") are not a selection rule.
Use this **lexicographic order**; the goal definition supplies the entry point,
which is what makes it principled rather than arbitrary:

1. **Goal-deficit entry.** The TIE bar (≥0.99× rapidgzip at *every* threadcount,
   and — for the T1 mandate — fastest vs libdeflate/zlib-ng/ISA-L) defines the
   worst cell objectively. Entry = the **lowest-ratio goal cell that is
   trustworthy** (or whose trust gap is cheap to close). You don't pick the
   biggest cost; the goal picks the most-broken cell.
2. **Blast radius, not magnitude.** Among hypotheses, rank by *how many failing
   goal cells the mechanism would explain if confirmed* × confidence-it's-real. A
   flat per-byte tax touching every cell outranks a large single-cell spike.
   (This is why the 1.56×-instructions finding ranks #1 — breadth, not size.)
3. **Value-of-information per probe.** Prefer the anomaly whose one diagnostic
   probe most cleanly *separates rival hypotheses* (kills the most hypotheses
   whichever way it lands). A probe that nudges a posterior loses to one that
   bisects the hypothesis space.
4. **Cost — final tie-break only.** Cheapest probe wins only when information is
   equal. Never promote cheap-but-uninformative.

**Re-rank after every finding** (the funnel mutates the matrix). **Anti-thrash:**
do not abandon a mid-funnel finding for a shinier S1 anomaly unless the new one
*strictly dominates* on (1)→(4). No anomaly-of-the-week churn.

---

## FAST LANE — for strictly-less-work, obvious-in-hindsight fixes

The full funnel is engineered for **attribution risk** — the cost of S2–S4 is
justified only when you're about to spend real implementation effort *betting* a
region gates the wall. The campaign's actual wins were "obvious in hindsight":
correctness-shaped, strictly-less-work fixes (a wrong constant, a double-decode, a
redundant copy, an accidental O(n²)). For those the funnel is upside-down.

**Discriminator (applied at S0): does justifying this change require BELIEVING AN
ATTRIBUTION?**

- **No** — the change is (i) byte-identical output, (ii) *demonstrably* fewer
  ops/allocs/bytes-moved (provably less work, not "should be faster"), and (iii)
  implementation cost ≤ one probe → **FAST LANE: enter at Stage 5.** Implement,
  re-measure on the matrix (interleaved, sha-verified), keep if TIE-or-better,
  revert if it regresses. No perturbation: you're not betting effort on an
  attribution; a strictly-less-work byte-identical change can't break correctness
  and is self-bounded.
- **Yes** — you'd invest effort betting a region gates the wall, OR the change
  *trades* (faster here / unknown there), OR it needs a perf model to believe it
  helps → **full funnel.** Perturbation mandatory.

Fast-lane changes still pay the faithfulness gate + matrix re-measure.

**FAST-LANE IS A BIAS-AMPLIFIER — two hard limits (added 2026-06-13 after the
antagonist caught it).** The fast lane skips S4 (causal perturbation), the one gate
that has ever produced truth in this campaign. Without limits it becomes a machine
for: form an interpretation → implement it without the gate that detects phantom
levers → keep it because it's byte-identical-and-TIE. So:
- **A kept-on-TIE change is BOOKKEEPING (correctness/cleanliness), NOT a
  performance advance.** Do not report a wall-TIE as a win; by the campaign's
  governing truth (1000 commits: a TIE is not an advance) it is a guess we kept.
  Keeping it is fine; *crediting* it as progress, or as "we already fixed region
  X," is the trap (it makes the real structural lever look addressed).
- **"Fewer instructions/ops" is NOT a fast-lane qualifier when the wall is
  frontend/memory/serialization/scheduling-bound** (which our oracles say it is).
  Reducing a retired-instruction count is an *attribution* about the wall, and
  attributions go through the funnel. The fast lane is ONLY for changes that
  remove a STRUCTURAL divergence from rapidgzip (e.g. a wrong data structure rg
  doesn't use) whose wall effect the matrix then confirms — not micro-opts *inside*
  a divergence (renaming the cost without removing it: the crc_write/push_slice
  pattern).

---

## Stage 0 — COVERAGE, TRUST & VALIDITY MAP (is the data even readable?)

Before reading ANY number for a decision, establish the numbers are complete,
trustworthy, and *mean what they claim*. This is the "gaps in our numbers" half of
the job. Classify every gap:

| Gap type | Definition | Example |
|---|---|---|
| **Coverage hole** | missing (arch, T, corpus, **comparator**) cell — comparator/definitional are sub-axes of coverage | no t2 column; T1 goal lacks vs libdeflate/zlib-ng/ISA-L |
| **Trust hole** | cell exists but NOISY / BIMODAL / low-N / stale / host-drift | t8/model BIMODAL — its *mean* is meaningless |
| **Validity hole** | the *instrument* may not measure what it claims (founding scar) | clean-window oracle silently re-ran bootstrap; empty-output instrument |
| **Provenance hole** | number lacks commit / host / flavor / fulcrum-version / freeze context to be reproduced | the phantom 2105ms that reproduced on no binary |
| **Decomposition hole** | a span conflates sub-costs; can't act until split | `crc_write` = CRC32 + write() + buffer-copy |
| **Mechanism hole** | an attribution exists but no causal perturbation | "serial consumer = 69% of wall" (attribution only) |

**CONSULT THE MEASURED LEDGER FIRST (added 2026-06-13 — the session's deepest
bias).** Before forming ANY new hypothesis, read `plans/disproof-ledger.md` and the
prior removal-oracle findings. This campaign's advances come from MEASUREMENT, and
many levers are ALREADY located by past oracles (e.g. DIS-15: T1 deficit is the
247ms ParallelSM per-chunk serialization, not the inner loop; DIS-16: T4 gap is
parallel-scheduling, not the consumer lifecycle; DIS-22: isal-T1→single-shot
shipped). Generating a fresh interpretation that re-treads or contradicts a measured
oracle is the relapse. A new hypothesis must either cite why the prior oracle
doesn't apply, or build on it — never ignore it.

**Gate 0:** (a) the instrument used to produce a cell must have passed a self-test
on this host/commit (see Instrument Self-Validation) — an unvalidated instrument
is a Validity hole, full stop; (b) a bimodal/noisy cell is reported as *modes +
the discriminator* (what flips the outcome), never averaged; (c) you FILL only the
gap that **blocks the current goal-deficit cell** — filling the whole matrix is
coverage-completionism. Scope the decision to the trustworthy subset and say so.

---

## Stage 0.5 — NEGATIVE-SPACE CHECK (no failing cell left unexplained)

The funnel hunts *shapes* (S1) and is therefore structurally blind to a gap that
is **flat and everywhere** — uniform across arch/T/corpus generates no shape. Yet
uniform per-byte overhead is this campaign's dominant suspect (the 1.56× tax was a
residual, not a shape; it nearly didn't surface).

**Gate 0.5:** every goal cell below the bar must map to **either** an open
hypothesis **or** an explicit `unexplained-flat — owes a shape-finding probe`
ticket. An agent may not chase three shapes while silently leaving a fourth
failing cell unaccounted. Coverage of *failing cells by explanations* is tracked,
not just coverage of the matrix by measurements.

---

## Stage 1 — ANOMALY READ (generate hypotheses, not conclusions)

Read the complete+trusted matrix for **structural shapes**, not single large
numbers:

- a value that **diverges across arch**,
- a **monotonic trend** with T or file-size,
- a **build-to-build convergence/divergence** (native↔isal collapsing as T rises
  ⇒ the differing code stopped mattering ⇒ gap is in shared machinery — a
  *build-independent* inference that doesn't trust ratios),
- a **regime split** (small-file overhead-bound vs large-file compute-bound),
- a **self-speedup gap** S(t)=wall(t1)/wall(tN) for *both* tools (separates "we
  scale badly" from "they scale badly too" — a raw ratio cannot).

**Forbidden:** "X is the biggest number ⇒ X is the lever." Output is a written,
**falsifiable hypothesis with a named mechanism** (plus any `unexplained-flat`
tickets from S0.5).

---

## Stage 2 — ADVISOR DISPROOF (mandatory; emit RIVAL hypotheses)

A heterogeneous Opus advisor tries to BREAK the hypothesis *before* a probe is
spent. It must check: deceiving framing (a ratio baking in the comparator's
scaling); **distinct mechanisms merged** under one label (fixed overhead vs
too-few-chunks vs SMT contention are three things); confounds (file-size regime,
SMT vs physical cores, comparator pinning/mask fairness, bimodality).

**Gate 2:** the advisor returns **≥2 co-equal competing hypotheses** that fit the
same data, each with a **pre-registered CONFIRM and FALSIFY signature**, and names
the **single probe chosen to DISCRIMINATE between them** (not to confirm the
favorite). Pre-registration + a real rival is what stops S2 from laundering S1's
conclusion into a sharpened version of itself.

**SHARED-NESS GATE (added 2026-06-13 after this bias was caught TWICE in one
session).** A profile/trace of ONE build (or one cell) does NOT establish that a
cost is paid by the OTHER build / generalizes. The native and isal builds FORK at
`gzip_chunk.rs` (`cfg(isal_clean_tail)`), so a native-only profile can be
native-heavy even when it "looks shared." Before claiming a finding is shared
(lifts both builds) or is the master cause of a cross-build deficit, **verify with
the apples-to-apples arm**: re-run the same probe on the OTHER build (both forced
`GZIPPY_FORCE_PARALLEL_SM=1`) and read whether it pays the same cost. native-vs-isal
at the same T isolates the inner loop from shared machinery (isal's pass with the
identical consumer is what proved the T1 deficit is inner-loop, not consumer).
"Assume shared / one master key" is this campaign's most recurrent bias — the gate
exists to break it.

---

## Stage 3 — CONFIRM-OR-KILL PROBE (attribution; narrows WHERE)

Run the one discriminating probe on the chosen cell/mode, on the right box
(perf-based modes ⇒ bare-metal perf box; LXC for wall-only). Compare to the
pre-registered signatures.

Mechanical requirements (every probe, non-negotiable): the **instrument passed its
self-test this host/commit**; production path asserted (`GZIPPY_DEBUG=1 →
path=ParallelSM`); build flavor (`--version` + `nm`); input sha pinned (STRIKE-5)
or abort; sink to a real file + sha-verify output; frozen host; interleaved N≥7
best-of-N; report inter-run spread.

**Gate 3:** a clean result selects among the rival hypotheses and narrows to a
region (now a *Decomposition hole* if it conflates sub-costs — split it). The
losing hypotheses are recorded as FALSIFIED in the disproof ledger. Δ < inter-run
spread ⇒ TIE, full stop. S3 and S4 may be the **same instrumented harness run** —
the gates are evidence standards, not five separate host trips.

---

## Stage 4 — CAUSAL PERTURBATION (the verdict — promotes to "lever")

Attribution, even a perfect trace, is NEVER the verdict. To prove region R gates
the wall, **perturb R's cost and measure the interleaved wall response** with a
**frequency-neutral control** (sleep, not busy-spin, so depressed turbo can't
masquerade as criticality):

- Use **≥2 injection magnitudes** (e.g. 25/50/100% of R's own measured time) and
  check the **slope** — one factor can't tell a proportional (critical-path)
  response from a threshold/step artifact. Proportional ⇒ on the critical path;
  flat ⇒ slack (rejected, however big it looked).
- Slow-down slope ≠ speed-up ceiling: to BOUND the win, REMOVE R (oracle) and
  measure — never extrapolate the slow-down slope through an unlocated knee.

**Gate 4:** only a survived perturbation promotes "candidate inefficiency" →
**confirmed lever**. This is the only gate (besides the fast lane) that authorizes
implementation work.

---

## Stage 5 — BOUND, DECIDE, IMPLEMENT

1. **Bound the ceiling** with the remove-region oracle; compute expected wall gain
   across *every matrix cell the region touches* (a fix helping t1/silesia may be
   neutral on t16/monorepo — quantify it).
2. **Strategic gate (Rule 3):** major effort or scope/ship decisions go to the
   user with the bounded gain BEFORE committing.
3. **Faithfulness gate:** no divergence from the rapidgzip port unless byte-exact
   + causally-won + ledgered.
4. Implement, then **re-measure on the matrix** (interleaved, sha-verified). A
   correct (byte-identical) change is KEPT and layered even on a wall-TIE; a
   rejection needs a *mechanism*, not a narrow miss.

---

## Instrument Self-Validation (hard gate for S3/S4)

Two instruments were silently broken this campaign. No fulcrum mode or
perturbation harness is trusted in S3/S4 until, **on this host/commit**, it passes:
binary-vs-itself reads 1.0 ± spread; positive and negative controls behave; and
(for the perturbation harness) a known-critical *synthetic* region responds
monotonically before the harness is believed on a real region.

---

## Finding Lifecycle (decay / re-open — so we don't re-cache phantoms)

A long campaign re-caches conclusions whose premise has died (the exact scar:
broken-oracle conclusions outliving their premise). Therefore:

- Every finding **stamps the commit it was measured at**.
- A confirmed lever **auto-demotes to "stale — re-measure"** when files it touches
  change.
- Every FALSIFY is recorded in the **disproof ledger with its probe AND the
  structural premise that killed it**, plus a **re-open trigger** (re-open if that
  premise changes). A FALSIFY is a verdict on that run, not an eternal law.
- **Baseline drift:** rg version / host turbo / corpus drift across weeks; run a
  periodic comparator re-baseline self-test (neurotic-drift history warrants it).
- **Stopping rule:** a cell is DONE and retired from the active set when it meets
  the bar (≥0.99× at every T; T1 also fastest vs the FFI tools it must replace).
  State it so agents stop funneling a closed cell.

---

## Worked example (the loop that produced this doc, 2026-06-13)

- **S0:** AMD+Intel × {t1,t4,t8,t12,t16} × 5 corpora × 3 builds, all `fulcrum
  score` (6 provenance invariants — provenance + comparator holes closed; the
  wheel tax that corrupted earlier ratios was a Stage-0 failure). Trust holes:
  t4/storedheavy NOISY, t8/model BIMODAL. *Open coverage hole:* T1 mandate needs
  libdeflate/zlib-ng/ISA-L comparators.
- **S0.5:** every failing cell mapped to the convergence hypothesis or an
  `unexplained-flat` ticket — which is what kept the per-byte tax in view.
- **S1:** native↔isal **converge** as T rises ⇒ high-T gap is shared machinery,
  not the clean tail.
- **S2:** advisor broke the ratio framing, demanded self-speedup for both tools,
  and pre-registered rivals: "shared scheduler" vs "more work per byte" (both
  build-independent), discriminating probe = **AMD silesia t4 `fulcrum flow`**.
- **S3:** FALSIFIED "shared machinery" (starvation 0%; workers run ahead; native &
  rg self-speedup identical). Selected the rival: serial consumer (`crc_write`
  54%, `apply_window` 15%) + **1.56× instructions/byte, constant across T**
  (a flat-everywhere finding S0.5 protected). `crc_write` → decomposition hole.
- **S4 (OWED):** not yet a lever. Decompose `crc_write`, then perturb CRC32 /
  apply_window at ≥2 magnitudes with a frequency-neutral control. The unification
  ("same per-byte tax loses T1") is a Stage-1 hypothesis awaiting its own probe.

---

## Tooling backlog (R1b — PULLED by a blocked finding, never pushed)

Build a tool only when a current goal-deficit finding is blocked on it (else
tool-building is just failure-mode #3 with a different object):

- `score/regen-index.sh`: add validity / provenance / decomposition / mechanism
  views (it surfaces coverage/trust/staleness today).
- `fulcrum`: self-speedup emitter (done: `score/derived_views.py`); **insn-by-region**
  breakdown (pulled now — needed to split the `crc_write` decomposition hole); a
  **perturbation harness** with its own self-test (pulled when S4 on the consumer
  finding starts).
