# FULCRUM — product charter (committed BEFORE the restructure; this is the contract)

USER DIRECTIVE (2026-06-11): evolve Fulcrum from gzippy's in-repo measurement tool
into an elegant, extremely powerful, open-source-ready **performance-decision
instrument**, usable across the user's other concurrent performance projects — and
keep raising its sophistication until ONE RUN reveals precisely what to do next.

## Product thesis

**Fulcrum is a causal performance-decision engine.** It is NOT a profiler (profilers
attribute; attribution has repeatedly manufactured phantom levers) and NOT a benchmark
harness (harnesses produce numbers; numbers without provenance produce phantom
comparisons). Fulcrum is the thing that closes the **attribution → causal** gap and
outputs **ranked, re-verifiable actions**:

- Every component row either carries a **tool-executed causal A/B verdict**
  (a same-binary kill-switch knob, effect-verified) or is explicitly labeled
  **HYPOTHESIS** with the exact pre-registered perturbation that would test it.
- Every stored number carries a **measurement fingerprint** (sink, mask, freeze,
  binary sha, corpus sha, protocol version). The tool **refuses** to form a ratio
  across incompatible fingerprints — phantom deltas die at the type level.
- Every run is checked against an **append-only results ledger**; a live number
  that contradicts a banked number with a *compatible* fingerprint is flagged
  CONTRADICTS-LEDGER instead of silently ranked (either the tool or the bank is
  wrong, and the report says so).
- The output is not a table the analyst interprets; it is a **decision brief**:
  top action + its causal evidence + its preconditions + the exact command +
  what result would falsify it.

The one-sentence pitch: *a profiler tells you where the time went; Fulcrum tells
you what to do next, and what observation would prove it wrong.*

## Architecture

```
tools/fulcrum/                     # standalone python package (stdlib-only core)
  fulcrum/
    core/                          # project-agnostic engine
      stats.py                     #   sample stats, bimodality, SPREAD-RESOLUTION
      trace.py                     #   Chrome-trace engine: pairing, leaf attribution,
                                   #   self-time, busy+idle==span (the trust asserts)
      causal.py                    #   knob A/B verdicts (CAUSAL-VERIFIED/NULL)
      fingerprint.py               #   measurement fingerprints + compatibility law
      ledger.py                    #   append-only results ledger + contradiction scan
      invariants.py                #   the INVARIANT SET, first-class, scar-named
      decide.py                    #   artifact loading + ranked decision engine
      report.py                    #   table + DECISION BRIEF rendering
    adapters/
      base.py                      #   ProjectAdapter interface (the plug surface)
      gzippy.py                    #   the first adapter: taxonomy, knob registry,
                                   #   effect predicates, routing guard, bank rows
    cli.py                         #   fulcrum decide/analyze/total/selftest/...
    selftests/                     #   every self-test old + new (incl. named
                                   #   regression cases for the 2026-06-11 phantoms)
scripts/fulcrum_total.py           # thin shim -> fulcrum.* (byte-compatible CLI)
scripts/fulcrum_decide.py          # thin shim -> fulcrum.* (byte-compatible CLI)
scripts/fulcrum                    # front door (decide/analyze/total/selftest/...)
scripts/bench/decide.sh            # gzippy launch policy (freeze/ship/pull) — the
                                   # adapter's environment-control plugin, unchanged CLI
```

**Core ↔ adapter split.** The core knows nothing about gzippy. A PROJECT ADAPTER
supplies, behind `fulcrum.adapters.base.ProjectAdapter`:

- **binary/launch matrix** — how to invoke the tool-under-test and its comparators
  for a (corpus, threads) cell; the expected production routing per cell.
- **corpora/workloads with integrity pins** — corpus ids + the decompressed-content
  sha pins every measured run is verified against (SHA-OR-VOID).
- **knob registry with effect predicates** — same-binary kill-switches (env-only,
  byte-exact in both arms) + the counter predicate proving each switch actually
  disabled the feature (EFFECT-VERIFIED-OR-FLAGGED).
- **comparator tools** — the parity targets (for gzippy: rapidgzip) and bank rows.
- **environment-control policies** — freeze/mask/sink as pluggable policy: the
  host-freeze acquisition, canonical pin masks, regular-file sink rule. For gzippy
  this is `decide.sh`/`_decide_guest.sh`/`lib_decide_guest.sh` (the hash-pinned
  parity-spine primitives); another project supplies its own.
- **trace taxonomy** — span-name prefixes for wait/compute/output/overhead and the
  wall-critical-thread ownership frames.
- **routing/contamination guard** — counter-sidecar rules that certify a run as
  production vs oracle-contaminated.

## THE INVARIANT SET (first-class, enforced, each named for its scar)

These are not documentation; each is a rule the tool *executes* (refusal or label),
with a self-test proving the enforcement fires. `fulcrum invariants` lists them with
their scars.

1. **SINK-LAW** — *scar: the 2026-06-11 HALF-PHANTOM matrix (rg re-based to
   file-sink while gz kept /dev/null numbers; T1 "0.973" was a phantom; the
   anchor's "gzippy is sink-insensitive" claim falsified — native pays ~110ms@T1
   for real output too).* Both arms of ANY comparison must use identical
   regular-file sinks. Every measurement is tagged with its sink class in the
   fingerprint; the tool REFUSES mixed-sink or half-rebased comparisons. A FIFO
   or /dev/null sink is itself flagged (the writev-phantom class).
2. **FROZEN-OR-LABELED** — *scar: ocl_cf's 0.945↔0.989 drift from a thawed box; a
   bench-lock TTL lapse caught only by absolute-level sanity.* A wall number from
   a thawed/loaded/readback-failed box is REFUSED for ranking; `--allow-thaw`
   downgrades refusal to an UNFROZEN label on every affected row. Freeze state is
   part of the fingerprint.
3. **SHA-OR-VOID** — *scar: "a speed win with wrong bytes is a loss"; the false
   SHA-DIVERGENCE from the read-slurp bug shows the check must be structural.*
   Every measured run's output is sha-verified against the corpus pin; any
   mismatch VOIDS the cell (and a knob arm with wrong bytes is recorded as its own
   finding — the switch is not byte-transparent — never ranked).
4. **SPREAD-RESOLUTION** — *scar: sessions spent measuring TIEs; the N=21
   silesia-T16 lesson (rg distributions go bimodal/quantized; a median can sit on
   either mode).* Every verdict carries RESOLVED/UNRESOLVED with N-needed; a
   sub-spread delta is NEVER presented as a finding; bimodality is detected and
   flagged on every sample set.
5. **CAUSAL-OR-HYPOTHESIS** — *scar: the 377ms pair-drain phantom, the per-EOB
   stop cost, the KEY-MISMATCH re-key lever — attribution that never converted at
   the wall; "contig_prof cycle-shares do NOT translate to wall share".* No row is
   ranked as actionable without a tool-executed causal A/B; everything else is
   HYPOTHESIS + the exact pre-registered perturbation command.
6. **EFFECT-VERIFIED-OR-FLAGGED** — *scar: the rpmalloc stats line that printed in
   BOTH arms (line presence proves nothing — the predicate had to read the
   slab-specific counters); the oracle.sh duplicate-env bug (env last-wins ⇒ ZERO
   injection, silently).* A kill-switch A/B is causal only if a counter predicate
   proves the switch engaged/disengaged; knobs without an in-tree counter are
   labeled EFFECT-UNVERIFIED, and a failed predicate voids the A/B
   (EFFECT-CHECK-FAILED), never silently trusted.
7. **SELF-TEST-OR-NO-TRUST** — *scar: two instruments were silently broken (a
   clean-window oracle that re-ran the bootstrap; another that emitted EMPTY
   output); the busy+idle==span check was once a tautology.* The analyzers carry
   synthetic-trace self-tests with positive AND negative controls and
   assertion-fires-on-corruption tests; `fulcrum decide`/`analyze` refuse (or
   label, with `--allow-untested`) when the installed engine's self-test stamp is
   missing or stale for the current code version.
8. **FINGERPRINT-OR-NO-COMPARE** (the generalization that subsumes 1, 2 and the
   clock confound) — *scar: the cyc/iter "regression" that was a TSC
   frequency-state mismatch between captures; the stale rg-anchor (the "0.98x"
   claim measured against a banked 926.6 when the live co-located rg ran 810).*
   Every stored number carries {sink, mask, freeze, binary sha, corpus sha,
   protocol version}; ratios/deltas across incompatible fingerprints are REFUSED.
   Ledger contradiction checks compare ONLY fingerprint-compatible rows.

## The sophistication raise (this phase)

a. **Measurement fingerprints** on every stored number, with
   refusal-to-compare across fingerprints (`fulcrum.core.fingerprint`). The guest
   manifest now records sink class + protocol version; the analyzer constructs the
   per-cell fingerprint and enforces SINK-LAW/FINGERPRINT-OR-NO-COMPARE before any
   ratio is formed.
b. **Results ledger** (`artifacts/fulcrum/ledger.jsonl`, append-only): every
   analyzed cell/knob verdict is banked with its fingerprint. On each run the tool
   scans prior compatible rows and emits CONTRADICTS-LEDGER anomalies when a live
   number diverges beyond tolerance — the bank-drift detection that caught the
   cyc/iter clock confound, generalized and automatic.
c. **Decision brief**: DO-THIS-NEXT upgraded from one line to a brief —
   ACTION / WHY (causal evidence + distribution health) / PRECONDITIONS
   (fingerprint, freeze, effect-verification state) / COMMAND (exact, re-runnable)
   / FALSIFIER (the observation that would disprove the recommendation).

## Open-source plan

- **Repo layout**: `tools/fulcrum/` is extraction-ready — stdlib-only core, no
  gzippy import in `fulcrum/core/*`, adapter + launch policy at the edges. To
  publish: lift `tools/fulcrum/` to its own repo root, keep `adapters/gzippy.py`
  as the reference adapter (its shell launch policy stays in the host project).
- **Naming**: package `fulcrum` (CLI `fulcrum`); PyPI candidate name
  `fulcrum-perf` if `fulcrum` is taken (note in pyproject).
- **License note**: MIT OR Apache-2.0 (match the host repo's posture; final choice
  is the user's call at publication — recorded in tools/fulcrum/README.md).
- **README skeleton**: shipped at `tools/fulcrum/README.md` (thesis, the invariant
  set with scar stories as design rationale, quickstart, adapter how-to).
- **Compatibility**: the gzippy scripts remain byte-compatible shims so every
  in-repo workflow (`decide.sh`, `fulcrum decide`, instrument-registry rows)
  survives the extraction unchanged.

## Acceptance (this phase)

- All existing self-tests pass relocated; new invariant self-tests pass, including
  a **mixed-sink refusal test** and a **cross-fingerprint ratio refusal test**
  (the 2026-06-11 phantoms as named regression cases) and a **ledger
  contradiction test** (the stale-anchor class).
- `scripts/bench/decide.sh` works unchanged; `scripts/fulcrum_*.py` CLIs are
  byte-compatible.
- One end-to-end `fulcrum decide` on the box at this HEAD with fingerprinting
  active, whose scoreboard encodes the current truth (the 7634a1a8 matrix:
  isal 6/14 / native 3/14 on the canonical cells measured) and whose decision
  brief is defensible.
