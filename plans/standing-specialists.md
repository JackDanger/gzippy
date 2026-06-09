# STANDING SPECIALISTS — campaign support team (supervisor, 2026-06-08; advisor-revised)

Persistent support roles for the gzippy parallel-single-member decode campaign, launched
at the user's direction (2026-06-08). They SUPPORT the owner; they do NOT do the owner's
implementation work or pick levers. Each is CONTINUED across turns via SendMessage (warm
context) and writes a DURABLE, PINNED artifact under plans/ so its knowledge survives
session boundaries. Revised per plans/standing-specialists-advisor-verdict.md (the advisor
REJECTed the standing Cartographer and the gate-on-three-deliverables sequencing; both fixed).

## ROLE BOUNDARIES — the two gates rule on DIFFERENT propositions (advisor fix #1)
- **Steward gates the MEASUREMENT**: "is this number BANKABLE?" (quiet box, matched sink,
  validated instrument, spread reported). Owns NUMBERS.
- **Red-team gates the INFERENCE**: "does this bankable number SUPPORT the claim/lever?"
  (causal perturbation present? attribution-as-verdict? re-measured TIE? moved goalpost?).
  Owns JUDGEMENT-OVER-TIME.
- They NEVER rule on the same proposition: Steward never opines on what a number means;
  Red-team never re-checks box hygiene. A claim needs BOTH gates (measurement bankable AND
  inference sound).
- **Owner owns IMPLEMENTATION + LEVER CHOICE.** Supervisor owns sequencing + user calls.
- **Cartographer is ON-DEMAND** (owner-invoked before a faithfulness-sensitive change), NOT
  a standing daemon — see below.

## PROVISIONAL ESCAPE VALVE (advisor fix #1b — prevents refuse-to-bank deadlock)
A number the Steward REFUSES may still be used for EXPLORATION (hypothesis generation,
direction-finding) — it just cannot gate a VERDICT or a COMPLETION claim. Refusal blocks
banking, not motion. The Steward labels each number BANKABLE or EXPLORATORY-ONLY, never
"stop."

## STALENESS PINS (advisor fix #2 — who watches the watcher)
Warmth is the strength AND the self-anchor failure (a stale ledger can freeze out a valid
lever after the binary changed; a stale self-test can read as passing — exactly the
isal_oracle_chunks= grep bug).
- Every instrument-registry entry is HASH-PINNED to the binary it validated AND asserts its
  grep/assert target strings STILL EXIST in current binary output. Hash mismatch OR missing
  target string => entry reverts to UNVALIDATED.
- Every disproof-ledger / vendor-map entry is COMMIT-PINNED. If the cited code commit moved,
  the entry reverts to OPEN/UNMAPPED until re-confirmed.
- The SUPERVISOR runs a periodic COLD audit (fresh spawn, no warm context) to catch warm
  drift the warm agents can't see in themselves.

---

## 1. MEASUREMENT-INTEGRITY STEWARD  [LAUNCH NOW]
MANDATE: no number BANKS into a verdict/completion unless it passes the Steward's gate.
AUTHORITY: labels each number BANKABLE / EXPLORATORY-ONLY (never "stop"). Owns scripts/bench/*
protocol + host bench-lock + the validated-instrument registry.
BANKABLE CRITERIA (ALL must hold): quiet box (instantaneous procs_running gate, host frozen);
matched same-sink (regular file /dev/shm); interleaved best-of-N>=9; sha-verified;
path=ParallelSM asserted; in-script fallbacks==0 where applicable; instrument self-test
1.0 +/- spread AND OFF==identity AND +/- controls AND its grep/assert targets exist in
current binary output; spread reported; NO producer-side attribution presented AS a verdict.
FIRST DELIVERABLE: (a) retro-validate the JUST-USED T4 gate instrument and CONFIRM the
0.899x (ocl_cf) / 0.740x (native) gate numbers are BANKABLE or flag caveats — this is the
ONE deliverable that must precede further work (the whole JOB-1 verdict rests on these);
(b) audit the harness for more of the isal_oracle_chunks= grep-bug class; (c) produce
plans/instrument-registry.md (each tool + hash-pin + grep-target-exists check + validation
status + known confounds). Does NOT pick levers or write production code (may land
measurement-only harness fixes like the grep label, isolated).

## 2. WARM RED-TEAM / CONTINUITY ADVISOR  [LAUNCH NOW]
MANDATE: gate the INFERENCE — catch cross-session judgement drift: re-measured TIEs, moved
goalposts, levers concluded from ATTRIBUTION (rule 1: a lever is only confirmed by a causal
perturbation + frequency-neutral control; attribution-only = UNCONFIRMED), claims not
survived-disproof.
AUTHORITY: advisory; every consequential claim/completion routes through it. Continued via
SendMessage per claim.
FIRST DELIVERABLE: plans/disproof-ledger.md (commit-pinned entries) — what's DISPROVEN
(+mechanism), what's a re-measured TIE (stop re-measuring), where the bar/goal shifted (incl.
the BINDING >=0.99x-at-EVERY-T tie bar, user-set 2026-06-08), open falsifiers, and for each
"the lever is X" line whether a causal perturbation confirmed it or it's attribution-only.
Seed from plans/orchestrator-status.md + the memory index. Does NOT pick levers or write code.

## 3. VENDOR-FIDELITY CARTOGRAPHER  [ON-DEMAND — owner-invoked, NOT launched as a daemon]
MANDATE: when the owner is about to make a faithfulness-sensitive change, map the
rapidgzip<->gzippy two columns for THAT region and flag divergence (vendor file:line). The
bias-guardrail "write the map before any change" — invoked at the change, not pre-produced.
SCOPE NOTE (advisor): the in-play regions (until_exact, the FFI handoff) are already
faithful + advisor-vetted; do NOT re-type those maps. The UNMAPPED region worth mapping is
the low-T scheduling / bootstrap / window-absent path (region (c)) — map it WHEN the owner
attacks the non-engine residual there, as part of that lever, not before.

---

## NEXT OWNER LEVER (advisor fix #4/#5 — the real work, start NOW not after artifacts)
The only owner-turnable lever (asm is the user's gated call; both builds lose T4) is the
NON-ENGINE RESIDUAL (<=0.101x / ~55ms at T4), today only an UPPER BOUND bundling: (i)
marker-prefix pure-Rust engine compute (the <=32KiB markered prefix that does NOT go through
ISA-L even in ocl_cf), (ii) FFI-handoff cost, (iii) scheduling/bootstrap/window-absent
placement. It cannot be turned into a lever until DECOMPOSED into those buckets. The owner's
immediate task = the RESIDUAL-DECOMPOSITION ORACLE (see plans/residual-decomposition.md).
The Steward gate-number validation runs in parallel; the Red-team ledger TRAILS the work.

## OPERATING MODEL / DISCIPLINES (all roles)
Read-only on production code (Steward may land isolated measurement-only harness fixes);
SOURCE-VERIFY first-hand (cite vendor file:line / actual gzippy code, never infer); pinned
durable artifact + short summary back to supervisor; NO orphan processes, NO detached sleep
sentinel.
