# DISPROOF VERDICT — standing-specialists.md charter (Opus advisor, 2026-06-08)

Read first-hand: plans/standing-specialists.md, plans/orchestrator-status.md (JOB 1 + JOB 2
+ cache-residency entry), CLAUDE.md Measurement PROCESS rules 1–8, memory
project_tie_bar_99pct_all_threadcounts.md.

Bottom line: the charter is **well-written but premature**. It is a governance layer
provisioned before the work it governs exists. Three of five axes are FIX-NEEDED, one is
REJECT, one SOUND. The single biggest problem is **sequencing (axis 5)**: running three
"first deliverables" before the next owner lever turn delays the only thing the user's
binding bar actually requires — closing the <=0.101x non-engine residual at T4. The charter
risks reproducing the exact failure it names: spending sessions producing artifacts ABOUT
the work instead of doing it.

---

## AXIS 1 — OVERLAP / AUTHORITY COLLISION — **FIX-NEEDED**

The three-way split (Steward=NUMBERS, Cartographer=FIDELITY, Red-team=JUDGEMENT-OVER-TIME)
is clean on paper, but two real collisions exist:

1. **Red-team vs Steward overlap on "attribution-as-verdict."** Rule 1 (no producer-side
   attribution as a verdict) is in the Steward's REFUSE-TO-BANK criteria ("NO producer-side
   attribution presented as a verdict") AND is core Red-team territory ("levers concluded
   from attribution"). Two roles can flag the same defect and — worse — one can *refuse to
   bank* while the other only *advises*. The owner then gets a hard block from one watcher
   and a soft caveat from another on the *same* number. Fix: give the Steward sole authority
   over whether a number is *bankable* (mechanical: did it pass the protocol). Give the
   Red-team sole authority over whether a *conclusion drawn from a bankable number* is a
   lever or a TIE. A number can be bankable (Steward OK) yet its lever-claim disproven
   (Red-team REJECT). Write that boundary into the charter explicitly: **Steward gates the
   measurement; Red-team gates the inference.** They never rule on the same proposition.

2. **"advisory, no veto" vs "refuse" is incoherent as written.** The Steward "may REFUSE to
   bank a number" and "a refused number does not gate a decision" — that IS a veto over
   measurement-driven decisions. The Cartographer is "advisory" but "the owner must address
   or justify each [divergence] before a change lands" — that is also a soft veto. Calling
   one "refuse" and two "advisory" obscures that all three can *block forward motion*. Fix:
   state the real asymmetry honestly — **Steward can block a number from banking (hard);
   Cartographer/Red-team can block a claim from being *recorded as survived-disproof* (the
   plans/wall-progress.md gate), but cannot block the owner from *trying* a lever.** That
   matches CLAUDE.md: levers are tried via causal perturbation, not pre-authorized.

**Deadlock risk on REFUSE-TO-BANK: REAL and under-scoped.** The criteria require ALL of:
quiet box (instantaneous procs_running), matched same-sink, interleaved best-of-N>=9,
sha-verified, path asserted, fallbacks==0, instrument self-test 1.0±spread + OFF==identity +
controls, spread reported. On a campaign whose own JOB 1 entry says the box is shared with
"Plex+7 noisy LXCs" that must be *frozen* per-run, and where N>=9 interleaved is slow, the
Steward can legitimately refuse nearly every number the owner produces under time pressure —
**halting the campaign with no number ever "good enough."** Fix: add an **escape valve** —
a refused number is recorded as PROVISIONAL with the specific failed criterion named, and
the owner may proceed on a provisional number for *exploration* (not for banking a verdict
or reporting completion). Banking a *completion/verdict* requires a clean number; banking a
*direction to keep digging* does not. Without this, REFUSE-TO-BANK is a campaign deadlock.

## AXIS 2 — WHO WATCHES THE WATCHER — **FIX-NEEDED**

The charter has no answer to this. Three concrete drift vectors, none guarded:

1. **The Steward's own instrument-registry can rot.** The registry validates other tools but
   nothing re-validates the registry. The campaign LITERALLY just hit this: the
   `isal_oracle_chunks=` grep bug silently hard-failed engine-isolation on *every prior
   attempt* (JOB 1 entry) — a stale self-test that *read as passing*. A registry entry marked
   "validated 2026-06-01" is trusted forever unless something forces re-validation. Fix:
   every registry entry carries (a) the binary/script content hash it was validated against
   and (b) a re-validate trigger — **if the tool's hash changes OR > N days OR the host
   changes, the entry reverts to UNVALIDATED and the Steward must re-run the self-test before
   that tool gates anything.** Hash-pinning is the only defense against "validated-once,
   trusted-stale."

2. **The warm Red-team's ledger drifting from current truth.** A persistent ledger is exactly
   where a re-measured TIE gets recorded as "disproven, stop measuring" and then the
   underlying *binary changes* (e.g. the dist-cache shrink, the ShortBitsCached port) and the
   old disproof no longer applies — but the ledger still says "don't measure this." This is
   the inverse failure: a *valid* lever frozen out by a *stale disproof*. Fix: every
   disproof-ledger entry pins the **commit/HEAD it was established against**; a disproof is
   only binding while the cited code is unchanged in the relevant region. The Cartographer's
   map (which tracks exactly those regions) is the natural staleness trigger — wire them:
   when the Cartographer marks a region changed, the Red-team's disproofs touching that region
   revert to OPEN.

3. **The Cartographer's map drifting from vendor + from gzippy.** It cites vendor file:line
   and gzippy code; both move (vendor is pinned, gzippy is not). Same fix: pin gzippy-side
   citations to HEAD; flag rows stale when the cited lines move.

**General fix (the watcher-of-watchers):** the SUPERVISOR (or a cold per-turn Opus spawn, as
is already done for owner claims) audits the three artifacts for staleness at the START of
each turn — cheap, cold, unbiased — rather than trusting the warm roles to catch their own
drift. Warmth is the strength (pattern memory) AND the failure mode (anchored to its own
prior conclusions). A cold periodic check is the only thing that catches a warm role's
self-anchor.

## AXIS 3 — NAMED-VS-PREVENTED FAILURE MODES — **FIX-NEEDED**

The charter NAMES every recurring failure but PREVENTS only some. Scorecard:

- **Broken instruments** (clean-window re-ran bootstrap; empty output; isal_oracle_chunks
  grep): PARTIALLY prevented. The Steward's "self-test reads 1.0±spread AND OFF==identity AND
  has ± controls" is a genuine, specific guard — this is the charter's strongest content. BUT
  it doesn't catch the *grep-bug* class (a self-test can pass while a downstream assert silently
  no-ops on a renamed field). Fix: the registry must record, per tool, the *exact assert
  strings it greps for* and verify those strings still appear in current binary output — the
  isal_oracle_chunks bug is precisely a grep-target that drifted out of the binary.

- **Contaminated numbers** (loaded box): prevented by the instantaneous-procs_running gate.
  SOUND.

- **Attribution-as-verdict**: NAMED in both Steward and Red-team but the charter gives no
  *mechanism* to enforce rule 1 — there's no requirement that every banked lever-claim cite
  the *causal perturbation* (slow-injection + frequency-neutral control) that CLAUDE.md rule
  1–3 demands. Fix: the Red-team's ledger must record, per "lever" claim, the perturbation
  that confirmed it (slope + frequency-neutral control result). A lever with only attribution
  evidence is logged as UNCONFIRMED, not as a lever. This is the single most important
  prevention to add given the campaign's history of phantom levers.

- **Cold re-derivation / re-measured TIEs**: the warm Red-team + ledger is the right shape.
  But see axis 2.2 — without commit-pinning, the ledger *causes* the inverse failure.

- **Orphans**: the charter says "no orphan processes / NO detached sleep sentinel" — naming,
  not prevention. The mechanism that actually prevents orphans is the harness (background-job
  lifecycle), not a sentence in a charter. ACCEPTABLE (out of charter's control) but don't
  claim it as a guard.

## AXIS 4 — MISSING / WASTED ROLE — **REJECT (the Cartographer is mis-prioritized for the frontier)**

Given the frontier the user's binding bar actually defines — **the <=0.101x non-engine
residual at T4** (the JOB 1 verdict's lower-risk lever that "helps BOTH builds") plus the
SYNC_FLUSH/until_exact gap (JOB 2) — the role values are NOT equal:

- **Steward: HIGH value.** Every fork rides on one number; the campaign has been reversed by
  bad numbers. Keep.
- **Red-team: HIGH value.** The campaign's dominant failure is re-measured TIEs and
  attribution-levers across sessions. Keep — but harden per axis 3 (perturbation-citation).
- **Cartographer: LOW value AT THIS FRONTIER, and partly REDUNDANT.** Two problems. (1) The
  non-engine residual lever (scheduling/bootstrap/prefetch/confirmed-offset partition) is
  *exactly* the region where CLAUDE.md says the structure is ALREADY a faithful vendor port
  (clean-window arming is "byte-for-byte"; the confirmed-offset gap is "fixable, NOT
  architectural"). A fidelity map of an already-faithful region produces low marginal signal.
  (2) The Cartographer's JOB-2 row (until_exact vs readStream) is *already fully mapped* in the
  orchestrator-status JOB 2 entry, with vendor file:line, by the owner — the Cartographer's
  first deliverable (a) is re-typing work already done and advisor-vetted x2. That is a WASTED
  deliverable.

**The 4th role that would help more than the Cartographer: a NON-ENGINE RESIDUAL DECOMPOSER /
PERTURBATION ENGINEER.** The JOB 1 verdict explicitly leaves the campaign blind: the
<=0.101x residual "CANNOT be sized... until the marker-prefix-engine + FFI buckets are
separated (advisor-owed: an oracle that also ISA-Ls the marker prefix, or an FFI-overhead
null run)." That owed oracle is the single highest-leverage missing artifact in the whole
campaign — it's what tells the owner whether the lower-risk residual lever is worth turning at
all. A role that *builds and validates the perturbation oracles* (the marker-prefix ISA-L
oracle, the FFI-null run, the slow-injection controls) directly unblocks the frontier. The
Cartographer guards against a divergence that, in the in-play region, the codebase isn't at
risk of. Fix: **replace the Cartographer with the Residual Decomposer**, OR narrow the
Cartographer strictly to JOB 2's until_exact change (the one genuinely
correctness-sensitive, divergence-prone seed-path edit) and fold general map-keeping into the
owner's existing "write the map before any change" discipline.

## AXIS 5 — SEQUENCING — **REJECT (run the residual lever, not three deliverables, first)**

The plan: run all three first deliverables BEFORE the next owner lever turn. This is wrong
for the binding bar.

- The user's bar is >=0.99x at EVERY T. Both builds LOSE at T4. The JOB 1 gate already
  delivered the decision input: **PARTIAL — engine share >=0.159x + non-engine residual
  <=0.101x.** The asm engine lever is GATED on the user's explicit call. So the ONLY lever
  the owner can turn *without waiting on the user* is the **non-engine residual** — and the
  JOB 1 entry says that lever is BLOCKED on an owed oracle (marker-prefix-ISA-L or FFI-null).

- Therefore the correct first action is **build the owed decomposition oracle and turn (or
  size-and-reject) the non-engine residual lever** — the real work. Running three
  documentation deliverables first *delays exactly the thing the bar requires*, and it
  reproduces the campaign's named failure mode (sessions spent measuring/mapping TIEs instead
  of moving the wall).

- Two of the three deliverables ARE worth doing, but as a *thin* first pass concurrent with
  the real work, not as a gate before it:
  - Steward's registry retro-validation of the JUST-USED T4 gate (the 0.899x/0.740x numbers
    the whole verdict rests on) — YES, do this first; if those numbers are contaminated the
    entire JOB 1 verdict is void. This is the one deliverable that genuinely must precede the
    next lever. **Highest priority, do immediately.**
  - Red-team's ledger seed — useful but small; seed it in <1 turn, don't gate on
    completeness.
  - Cartographer's full vendor-map — DEFER (axis 4); the JOB-2 row is already mapped, the
    in-play regions are already faithful.

**Correct sequencing:** (1) Steward retro-validates the 0.899x/0.740x gate numbers [must
precede anything that builds on them]. (2) In parallel, owner builds the owed marker-prefix /
FFI-null decomposition oracle — the actual frontier. (3) Red-team + Cartographer seed thin
artifacts as the work generates claims, not before it. The deliverables should *ride* the
real work, not block it.

---

## VERDICT SUMMARY
- Axis 1 (overlap/authority): **FIX-NEEDED** — separate "gates the measurement" (Steward)
  from "gates the inference" (Red-team); make REFUSE-TO-BANK non-deadlocking via a PROVISIONAL
  escape valve for exploration.
- Axis 2 (watcher-of-watchers): **FIX-NEEDED** — hash-pin + commit-pin every artifact entry;
  stale = reverts to UNVALIDATED/OPEN; cold periodic supervisor audit catches warm self-anchor.
- Axis 3 (named vs prevented): **FIX-NEEDED** — registry must verify grep-target assert
  strings still exist in binary output; ledger must cite the causal perturbation per lever
  (attribution-only = UNCONFIRMED).
- Axis 4 (missing/wasted role): **REJECT** — Cartographer is low-value/redundant at this
  frontier; replace with (or add) a Residual Decomposer that builds the owed
  marker-prefix-ISA-L / FFI-null oracle.
- Axis 5 (sequencing): **REJECT** — do NOT gate the next lever on three deliverables; only the
  Steward's retro-validation of the 0.899x/0.740x gate must precede the work. The non-engine
  residual oracle IS the real work and should start immediately, not after the paperwork.

The charter's instinct (durable artifacts, warm continuity, strict anti-overlap, instrument
self-tests) is correct. Its defect is that it provisions a full standing bureaucracy in front
of a campaign whose next move is already known and singular, and it leaves the watchers
themselves unwatched. Trim to the two high-value roles (Steward, Red-team), add staleness
pins and the residual-decomposition oracle, and let the artifacts trail the work.
