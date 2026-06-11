# fulcrum

**A causal performance-decision engine.** Not a profiler, not a benchmark
harness — the thing that closes the attribution → causal gap and outputs
ranked, re-verifiable actions.

A profiler tells you where the time went. Fulcrum tells you what to do next —
and what observation would prove it wrong.

## Why

Performance campaigns are routinely misled by their own instruments:

- attribution that never converts at the wall (a region can dominate every
  busy/latency/critical-path metric and still be wall-neutral);
- comparisons across silently different protocols (one arm writing to
  /dev/null, the other to a real file; a frozen box vs a thawed one; a stale
  banked comparator number vs a live one; a different comparator version or
  a different host);
- kill-switches that silently didn't switch (env typos, last-wins duplicates,
  predicates that read a log line that prints in both arms);
- instruments that were themselves broken (an oracle that re-ran the work it
  claimed to remove; a capture that emitted empty output and got trusted);
- manifests that self-report what should be observed (a hardcoded
  "sink=regular-file" line is a claim, not a measurement).

Fulcrum's design position: **every one of those failures becomes a rule the
tool enforces** — a refusal or a label, with a self-test proving the
enforcement fires. See `fulcrum invariants` for the full set; each rule is
named for the scar that made it law, and
[docs/CASE-STUDIES.md](docs/CASE-STUDIES.md) tells the original stories:

| invariant | rule (short form) |
|---|---|
| SINK-LAW | both comparison arms use identical regular-file sinks; mixed-sink/half-rebased comparisons are refused |
| FROZEN-OR-LABELED | thawed/loaded-box wall numbers are refused for ranking, or loudly labeled |
| SHA-OR-VOID | every measured run's output is content-verified; a mismatch voids the cell |
| SPREAD-RESOLUTION | verdicts carry RESOLVED/UNRESOLVED + N-needed; sub-spread deltas are never findings; bimodality flagged |
| CAUSAL-OR-HYPOTHESIS | only tool-executed causal A/Bs rank as actionable; all else is HYPOTHESIS + the exact perturbation |
| EFFECT-VERIFIED-OR-FLAGGED | a kill-switch A/B is causal only if a counter proves the switch flipped |
| SELF-TEST-OR-NO-TRUST | the engine refuses trust until its self-tests (incl. corruption-fires tests) pass at this source version |
| FINGERPRINT-OR-NO-COMPARE | every number carries {sink, mask, freeze, binary sha, corpus sha, protocol, comparator version, host identity}; cross-fingerprint ratios are refused |

Supporting disciplines the tool executes:

- **Derived, never self-reported, environment fields** — sink class via
  stat, pin mask via kernel readback, freeze via sysfs, comparator version
  via probe, host identity from the box; a manifest self-report
  contradicting its derivation is flagged `DERIVED-MISMATCH` and the derived
  value governs.
- **Append-only results ledger with resolution semantics** — a live number
  contradicting a compatible banked number is flagged CONTRADICTS-LEDGER and
  banked `pending-reconcile` (never as an anchor) until an explicit
  `supersede` or `invalid` record resolves which side was wrong. Append-only
  is a convention the tooling upholds; a hash-chain field per record makes
  tampering with the file evident (`fulcrum ledger` verifies it).

## What one run produces

- a **cell scoreboard**: tool vs comparator wall per (workload × threads)
  cell, interleaved, sha-verified, fingerprinted, PASS/FAIL against the
  project's tie bar, RESOLVED/UNRESOLVED with N-needed, bimodality flags;
- a **ranked component table**: tier 1 = causally verified
  shipped-default-COSTS rows (same-binary kill-switch A/Bs, effect-verified),
  tier 2 = bounded hypotheses (trace decomposition + engine micro-profile),
  tier 3/4 = causal confirmations and nulls — each row with distribution
  health, RSS, and the exact re-verify command;
- **anomalies**: ledger contradictions, derived-vs-claimed mismatches, bank
  divergences, refused captures — verbatim, never silently dropped;
- a **DECISION BRIEF**: top action + causal evidence + preconditions + the
  exact command + the result that would falsify it.

## What is contract vs what is gzippy's

Honesty about the package's current shape:

**The contract (project-agnostic, stdlib-only):** everything under
`fulcrum/core/` — stats, the Chrome-trace engine, causal verdicts,
fingerprints, the ledger, the invariant registry, the decision engine and
report — plus `adapters/base.ProjectAdapter` (the plug surface) and the
documented artifact schema ([docs/SCHEMA.md](docs/SCHEMA.md)). The core never
imports project code; a toy second project drives the whole pipeline in
`selftests/test_adapter.py`.

**The reference project (gzippy):** `adapters/gzippy.py` — its span
taxonomy, counter patterns, knob registry, effect predicates, routing
guards, banked profile comparators, and re-verify command strings are all
specific to the gzippy↔rapidgzip parity campaign. The measurement *policy*
(host freeze, canonical pin masks, sink discipline, sha pins) lives in the
host repo's shell scripts (`scripts/bench/decide.sh` + guest scripts), not
in this package. The case studies are that campaign's real failures.

If you adopt fulcrum for another project you write: one adapter module
(taxonomy, knobs + effect predicates, guards, optionally `load_run` for your
own artifact layout) and your own measurement policy that emits the
documented schema (or any layout your `load_run` reads).

## Architecture

```
fulcrum/
  core/        stats, trace engine, causal verdicts, fingerprints, ledger
               (supersede/pending-reconcile + hash chain), invariants,
               decision engine, report   (stdlib-only, no project knowledge)
  adapters/    base.ProjectAdapter + one module per project
  selftests/   4 suites: trace engine, decision engine, invariant
               enforcement (named regression cases for the historical
               phantoms), adapter pluggability (the toy second project)
  cli.py       fulcrum analyze/total/selftest/invariants/ledger
docs/
  SCHEMA.md        the artifact-dir + run-dict + row contracts
  CASE-STUDIES.md  the measurement failures behind the invariants
```

## Quickstart (inside the gzippy host repo)

```bash
scripts/fulcrum selftest            # must pass before any number is trusted
scripts/fulcrum decide              # one run on the bench box -> decision brief
                                    #   (host-repo policy: freeze/ship/pull)
scripts/fulcrum analyze <art-dir>   # re-render from pulled artifacts
scripts/fulcrum invariants          # the enforced rule set, with scars
scripts/fulcrum ledger              # banked results + fingerprints + chain check
scripts/fulcrum ledger supersede --key K --retire RUNID --promote RUNID --reason "..."
```

`fulcrum decide` is host-repo policy (it needs a bench box and freeze
infrastructure); `analyze`, `total`, `selftest`, `invariants` and `ledger`
are pure package functionality and run anywhere.

## Writing an adapter

Subclass `fulcrum.adapters.base.ProjectAdapter`; supply a `Taxonomy`
(span-name prefixes for wait/compute/output and the wall-critical-thread
frames), a knob registry (`{name: Knob(env, pred, desc, reverted=False)}`),
`effect_check` predicates that prove each switch actually flipped,
`parse_counters` + `routing_guard` for contamination detection,
`comparator_version` for the fingerprint, and re-verify command surfaces.
Override `load_run` if your artifacts aren't in the documented layout.
Your measurement policy (how cells are launched under freeze/mask/sink
discipline) stays in your repo; fulcrum consumes what it produces. The full
contract is [docs/SCHEMA.md](docs/SCHEMA.md).

## Status / license

Extraction-ready in-repo package; stdlib-only, Python >= 3.9. Dual-licensed
MIT OR Apache-2.0 (see LICENSE-MIT and LICENSE-APACHE; choose either). PyPI
candidate name: `fulcrum-perf` if `fulcrum` is taken.
