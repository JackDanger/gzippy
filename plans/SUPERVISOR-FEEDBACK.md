# SUPERVISOR-FEEDBACK.md — standing coach's report

Independent, read-only review for the SUPERVISOR (human). Candid by design. Extend
this doc on each review; keep the BOTTOM LINE at the top current.

Convention used throughout: **wall ratio = rapidgzip_wall / gzippy_wall.** 1.00× =
tie (the goal). <1.00× = gzippy is slower (LOSS). 0.73× ⇒ gzippy ~37% slower than rg.

Review #1 — 2026-06-07, by review-coach agent. Evidence: `git log reimplement-isa-l`
(1367 commits), plans/*.md (esp. CAMPAIGN-CHARTER.md @04fda86d, all *-advisor-verdict.md),
memory/*.md. Transcripts sampled only to fill gaps.

---

## BOTTOM LINE — 3 things the supervisor should change NOW

1. **RUN `GZIPPY_PERFECT_OVERLAP` BEFORE authorizing one more scheduling work-stretch.**
   It is the *registered* decider for the current direction (scheduling/serial = T8
   binder), and today's own advisor (`plans/scheduling-ceiling-advisor-verdict.md`)
   REFUTED the owner's C2+C3 and said in plain words: the pure-scheduling ceiling is
   **"UNBOUNDED by the evidence presented — run GZIPPY_PERFECT_OVERLAP."** It has
   *never been run* (appears only in the prereg/brief/verdict of today + the old
   diffuse-gap-closer). The whole current strategy rests on an unmeasured premise.
   This is a Rule-3 violation (slow-down slope ≠ speed-up ceiling) being committed
   right now. **Make the next turn produce that one oracle number, with a sha-
   transparent self-test (Rule 4), and nothing else.**

2. **Forbid producer-side attribution as a verdict — enforce it, don't just restate it.**
   The campaign's #1 recurring defect is *counter/attribution inversions read as
   binders*. In the last ~10 turns alone: the owner read `BOOTSTRAP_POST_FLIP_U16_BYTES`
   **backwards** (charter line 74 — "the exact counter-inversion the charter's
   u16-ceiling correction warns about — I repeated it"); C2/C3 of the scheduling brief
   were built on `future::get` *attribution* and refuted; the earlier "1.18× gap =
   diffuse coordination, no recoverable lever" STOP was an attribution conclusion that
   got overturned. The charter already says "perturb, don't attribute" — but it keeps
   happening because attribution numbers are *allowed into the brief at all*. Process
   fix: **every binder claim in a brief must lead with the causal perturbation result;
   any SUM/ratio/Fill% is footnoted as hypothesis-only.** Reject briefs that invert.

3. **Stop re-opening "stop"/"tie" conclusions; gate them on the oracle, not the narrative.**
   We have at least TWO declared-victory events that were reversed: the 2026-06-02
   "STOP EARNED / Goal achieved" arc (`8291adb9`, `ff65819f`, `ca9a8f90`) and the
   broken clean-window oracle that licensed the 2026-05-29 rescind (`64eb6df`). Each
   reversal cost the campaign a re-derivation of the same map. The lesson the owner
   has *written down* but not *operationalized*: a STOP or a TIE is only real if the
   deciding **removal oracle** was run and validated. You (supervisor) should refuse
   to ratify any "done/stop/at-parity" claim that isn't backed by an oracle number,
   the way you'd refuse a merge without tests.

---

## Q1 — CONVERGENCE / RATCHET: is the T8 wall ratcheting to 1.0×?

**Short answer: directionally yes within the *current* (faithful pure-Rust) basis,
but the campaign has crossed a measurement-basis boundary that hides a real regression
in the headline number, and the within-basis ratchet is shallow (one confirmed tooth).**

Reconstructed T8 ratio timeline (rg/gzippy, higher = better):

| date | basis | T8 ratio | source |
|------|-------|----------|--------|
| 2026-06-02 | **shipped ISA-L product** vs rg | ~0.85 (gap "1.18×") | `8291adb9`/`ff65819f` "1.18x T8 gap fully mapped" |
| 2026-06-07 (pre-merge-removal) | pure-Rust faithful | ~0.65 | charter line 89/431 |
| 2026-06-07 (merge-removal landed) | pure-Rust faithful | ~0.73 | charter line 86–89, `77a02f5f`/merge fix |
| 2026-06-07 (marker fast loop) | pure-Rust faithful | ~0.73 (TIE, no move) | charter line 68, `04fda86d` |

Reference points that bound the ceiling (same harness, 2026-06-07, charter 147–152):
- rg = 0.134s (1.00×); **`pure` engine SEEDED = 0.134s (1.002×) — already TIES;**
  ISA-L seeded = 0.148s (0.90×, TIE); **prod (no seed) = 0.194s (0.69×) LOSS.**

Verdict: **It is a genuine ratchet, not pure churn — but a shallow one and the
headline is basis-confounded.**
- *Real teeth banked:* merge-removal (+~12%, 0.65→0.73, byte-exact, advisor-upheld,
  remove-and-measure with a freq-neutral control). That is the gold-standard kind of
  progress and it is the ONLY clean wall-moving tooth in the recent record.
- *Not progress:* the marker fast-loop (TIE), the entire engine-bench/VAR_V/BMI2/asm
  thread (all PLATEAU/TIE), placement (oracle TIE). Many turns, ~one tooth.
- *The hidden regression:* the best-EVER mapped ratio was ~0.85 (June 2, ISA-L
  product). We are now at 0.73 on a *different* path. This is defensible (the goal
  legitimately changed to a no-FFI pure-Rust port, and the June-2 "stop" was
  overturned as premature) — but the supervisor should be explicit that **0.73 < 0.85
  is not a regression of the same artifact; it is the cost of switching to the harder,
  correct target.** Do not let anyone narrate 0.73 as "best ever."
- *The good news the owner under-weights:* the seeded `pure` TIE (1.002×) is a hard
  existence proof that **the remaining 0.73→1.00 gap is ENTIRELY in the window-absent /
  scheduling path, not the engine.** That sharply narrows the search. The campaign
  spent many turns (engine-bench, VAR_V, asm, BMI2, plateau) *re-confirming* the engine
  is not the T8 lever — which the seeded-TIE already proved. That is the churn.

---

## Q2 — RECURRING FAILURE MODES (and the fix)

| failure mode | still happening? | evidence | fix |
|---|---|---|---|
| **Counter/attribution INVERSION read as a binder** | **YES — last turn** | charter 74 ("read `BOOTSTRAP_POST_FLIP_U16_BYTES` BACKWARDS… repeated it"); scheduling C2/C3 refuted on `future::get` attribution | **#2 in BOTTOM LINE.** Briefs must lead with perturbation; name counters for what they count; advisor must check the counter's *sign* first. |
| **Premature STOP / declared victory, later overturned** | recurring | 2026-06-02 "STOP EARNED/Goal achieved" reversed; 2026-05-29 rescind on broken oracle `64eb6df` | **#3 in BOTTOM LINE.** No stop without a validated removal oracle. |
| **The decider oracle keeps getting deferred** | **YES** | `GZIPPY_PERFECT_OVERLAP` registered but never run; the 2x2 decompose was DEGENERATE (onlywin self-test FAILED, charter 122/125); seedfull over-removes (advisor) | **#1 in BOTTOM LINE.** Make the oracle the *first* artifact of the turn, not the last. |
| **Leaders dying mid-flight (background-and-yield, no auto-reinvoke)** | **mostly FIXED** | charter "HOW YOU DELEGATE": "Run subagents SYNCHRONOUSLY… There is NO auto-reinvoke — do NOT background and yield; you will die"; singleton-leader lock added (status 1005) | Hold the line — this was clearly a painful, now-codified lesson. |
| **Orphaned processes** | **FIXED** | every recent status turn ends "NO orphan processes" (status 44/79/130/173/215/247/253/343) | Keep the pgrep-clean discipline. |
| **Duplicate orchestrators** | FIXED | `scripts/leader-lock.sh` mkdir-mutex + stale-pid reclaim, self-tested (status 1005) | Fine. |
| **Multi-line python / heredoc wedging the Bash channel; full-disk builds** | codified, watch | memory `feedback_when_tool_errors_find_out_why`; charter delegate rules | Standing rule; no recent recurrence seen. |
| **Unit errors / turbo confounds** | recurred earlier, now controlled | `43e3c63b` "1.42x was a TURBO CONFOUND (real ~1.20x)"; interleaved harness now freq-neutral by construction | Interleaved-N≥11 + sha-every-run is the right guard; keep it mandatory. |

**Single highest-leverage process fix:** make the *registered decider oracle* the
gating deliverable of each loop, run and self-validated, BEFORE any port/optimization
work and BEFORE any binder claim. Every one of the top-3 failure modes is a symptom of
acting on attribution/inference because the oracle was deferred.

---

## Q3 — DROPPED / RE-DERIVED THREADS

**Abandoned with a real mechanism (correctly closed — do NOT revive):**
- *Engine / inner-Huffman / VAR_V / BMI2 / asm-port* — PLATEAU at ~0.6–0.68× ISA-L,
  pre-registered falsifier FIRED (charter 200–223; `3895a23c`; round-2 verdict). AND
  the seeded-`pure` TIE proves the engine isn't the T8 binder anyway. Closed twice over.
  **Caveat the owner flagged honestly:** the plateau was measured on the *u16* ring
  (memory `project_engine_plateau_pure_rust`), which rg never uses for the bulk; the
  *faithful u8-direct native* clean engine ceiling is arguably still un-measured. This
  is the one engine thread that is *legitimately* open — but it is a T1 lever, not T8.
- *Footprint / page-walk / THP / allocator reuse* — falsified 6× (`52e16e01`,
  `8291adb9`); demand-miss ≠ page-fault; closed.
- *placement / frontier-prefetch / prefetch-horizon* — oracle TIE (`3ac55b7f` "gzippy
  already prefetches frontier 94%"; `249f25b5`). BUT see below — the *head-of-line*
  variant is NOT the same as frontier-placement and is still open.
- *window-discard* — source-verified then FALSIFIED (status 348; gzippy already
  re-targets, `needs_confirmed_offset` has zero hits). Correctly closed.

**The genuine engine↔scheduling OSCILLATION (real migration vs churn):** counted
~10 "engine is the binder" framings vs ~12 "scheduling/consumer/head-of-line is the
binder" framings across orchestrator-status. This is the charter's named risk. My read:
it is **~60% real migration, ~40% wasted re-derivation.** Real: each clean tooth
(merge-removal) genuinely moved the binder. Wasted: the engine side keeps getting
*re-tested* (engine-bench round 1, round 2, VAR_V, BMI2, asm-feasibility, marker fast
loop) long after the seeded-TIE settled that the engine is not the T8 lever. The
charter's 2026-06-07 "IT'S A RATCHET" addendum is the right framing — but it is being
used to *license* re-testing the engine rather than to *forbid* it.

**Dropped possibly-prematurely — REVIVE:**
- **`project_confirmed_offset_prefetch_gap`** (memory): "4 head-of-line stalls at
  confirmed offsets ≠ partition guess, ~40% of T8 wall, **fixable, NOT architectural.**"
  This is the most specific, highest-value, *named* lever in the whole record, it is
  directly the current "scheduling/serial binder," and it has been "long-deferred"
  (charter's own word, line 75). The current scheduling direction should be *this
  specific fix*, validated by `GZIPPY_PERFECT_OVERLAP` — not a fresh diffuse re-survey.

**Kept that should be cut:** nothing dead is on `main`/branch by policy ("no dead code"),
and that discipline is being honored (`70ec5ff9`/`edcd8631`/`03f592ee` delete dead
decode modules). Good. The thing to "cut" is *attention*, not code: stop re-opening the
engine ceiling.

---

## Q4 — TRUST AUDIT: how often does a finding get refuted?

**Refutation rate is HIGH and is the dominant texture of the campaign.** Mechanical
count of refutation language in advisor verdicts: 22 of 22 verdict files contain
≥1 "refuted/overstated/inverted/overturned"; today's `scheduling-ceiling-advisor-verdict`
contains **6**, `tier1-advisor-verdict` 7, `placement-rescope` 7, `step-c-design` 6,
`engine-bench-round2` 6, `u16-ceiling` 5, `prefetch-horizon` 5. The owner's *own*
brief claims are refuted by the advisor in a large fraction of turns.

**Is this a pattern of over-claiming then walking back? Yes — but it is being CAUGHT,
which is the system working.** The synchronous disproof advisor is doing real work
(the merge-removal tooth survived; the marker-loop *mechanism* did not and was struck;
C2/C3 today were caught before a work-stretch). The danger is not that findings get
refuted — it's the *volume* of confident briefs that need refuting, which means turns
are spent generating-then-demolishing rather than measuring.

**Is the CURRENT "scheduling/serial is the T8 binder" attribution trustworthy?**
**Partially — the LOCALIZATION is sound, the MECHANISM is not yet earned.**
- TRUSTWORTHY: seeded-`pure` TIE (1.002×) + prod LOSS (0.69×) is a clean remove-and-
  measure proving the gap lives on the window-absent/scheduling path, not the engine.
  That is the firmest result in the campaign.
- NOT YET EARNED: that the binder is specifically *in-order consumer head-of-line wait*
  vs *pool-fill* vs *the window/clean coupling* is still attribution. Today's advisor
  refuted the owner's specific scheduling mechanism (C2 "future::get mostly overlapped"
  REFUTED — it HALVES T8→T16, the signature of a critical-path term; C3 ceiling
  double-counts the serial tail, refuted). **The direction is right; the specific
  mechanism is shaky and the deciding oracle is unrun.** Given the oscillation history,
  treat "scheduling is THE binder" as a high-prior hypothesis, not a finding, until
  `GZIPPY_PERFECT_OVERLAP` lands.

---

## Q5 — SUPERVISION EFFICIENCY

**Fraction of turns producing a real wall RATCHET vs measurement/diagnosis-only:**
eyeballing the last ~30 commits and status turns, **~1 in 10 turns banked a confirmed
wall-moving tooth** (merge-removal). The rest were diagnosis, oracles, falsifiers,
advisor rounds, plateau confirmations, and TIEs. For a bottleneck-following campaign at
the hard end of the curve some of this is irreducible — but a 1:10 ratchet:diagnosis
ratio with a high refutation rate says **too much effort is going into measuring/re-
measuring things already settled** (the engine), and not enough into running the ONE
decider oracle and then fixing the named head-of-line lever.

**Are YOU (supervisor) still over-doing hands-on?** The user has told you twice you do
too much hands-on, and the charter now *literally says* "Supervisor = thin relay/cleanup
only" and "You (the OWNER) fully own this campaign." Yet:
- The owner re-uses *your* exact test invocation as canon (status 1029 "the supervisor's
  invocation… the supervisor's valid invocation") — meaning you were hands-on enough in
  test mechanics that it's encoded as the reference. That's a tell you were in the loop
  at a level the owner should own.
- Specific places to be THINNER: (a) do not hand the owner harness/test invocations —
  make the owner own and document them (it now has, in the fold-gate); (b) do not ratify
  per-turn TIE/keep decisions — those are rule-7a mechanical, the owner can self-apply;
  (c) reserve your intervention for exactly the charter's named escalation: **a genuine
  FORK trading off a user constraint (1.0× vs no-FFI vs faithfulness).** Everything else
  is the owner's.
- Where you SHOULD stay hands-on (and it's high-leverage, not over-doing): **enforcing
  the three BOTTOM-LINE gates** (run-the-oracle-first, no-attribution-verdicts, no-
  stop-without-oracle). That is supervision, not hands-on work.

Net: you're thinner than before (orphan/leader/duplicate discipline is now the owner's,
codified), but you're still being pulled into measurement mechanics. Push those down.

---

## Q6 — STRATEGIC: is the current direction the highest-confidence path to 1.0×?

**Direction (scheduling/serial ceiling → faithful port of rg's pacing/scheduling): YES,
it is the right place to look — the seeded-TIE proves the gap is there and nowhere
else. But it is NOT yet executable as stated, because the decider oracle is unrun and
the specific mechanism was just refuted.** Recommended sequencing:

1. **Run `GZIPPY_PERFECT_OVERLAP`** (sha-transparent, self-tested). Three outcomes:
   - ties rg (≤~0.137s) ⇒ scheduling reaches the tie; the path is *exactly* the named
     `project_confirmed_offset_prefetch_gap` head-of-line fix. Highest-confidence win.
   - lands in F2 (strict-improve-but-not-tie) ⇒ scheduling is *a* term not *the* term;
     then perturb the engine clean-rate and the backward marker scan (C4) to find the
     co-binder. This matches memory `project_pregate_placement_is_dominant_lever`'s
     "PLACEMENT + ENGINE are CO-PRIMARY" — which the campaign has half-forgotten.
   - flat ⇒ the scheduling premise is wrong; re-perceive. (Cheap to learn.)
2. **Then** fix the named head-of-line stall faithfully against rg's pacing (the one
   specific, "fixable not architectural," ~40%-of-wall lever), not a diffuse re-survey.
3. **Hold the engine closed at T8** (seeded-TIE settles it); the only engine work worth
   funding is the *faithful u8-direct native clean* ceiling AS A T1 lever, and only if
   T1 becomes the binding cell.

**Anything to redirect/revive:** revive `project_confirmed_offset_prefetch_gap` as the
*concrete* target of the scheduling direction (it's the same thing, stated precisely).
Re-surface the CO-PRIMARY (placement+engine) memory so the next loop doesn't re-derive
"engine isn't the *sole* binder" a third time. De-fund further generic engine-ceiling
benchmarking.

**Confidence:** the path can reach a tie *if* the head-of-line stall is as fixable as
the memory claims and the perfect-overlap oracle confirms it. The biggest risk to 1.0×
is not the technical lever — it's the campaign's habit of acting on attribution and re-
deriving settled results, which burns the turns this needs.

---

## APPENDIX — key evidence pointers (for next reviewer)
- Freshest verdict (today, 6 refutations, decider unrun): `plans/scheduling-ceiling-advisor-verdict.md`
- The seeded-TIE existence proof (engine not T8 binder): `plans/CAMPAIGN-CHARTER.md` lines 143–160
- Confirmed tooth (merge-removal +12%): charter 86–89; commits around `77a02f5f`
- Counter-inversion repeated last turn: charter line 74 (`04fda86d`)
- Premature-STOP arc later overturned: `8291adb9`, `ff65819f`, `ca9a8f90` (2026-06-02)
- Broken-oracle rescind: `64eb6df` (referenced in CLAUDE.md goal block)
- The named, fixable, deferred lever: memory `project_confirmed_offset_prefetch_gap.md`
- CO-PRIMARY framing the campaign half-forgot: memory `project_pregate_placement_is_dominant_lever.md`
- Process discipline now codified (good): charter "HOW YOU DELEGATE"; status `leader-lock.sh` @1005
