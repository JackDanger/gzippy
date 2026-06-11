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
  banked comparator number vs a live one);
- kill-switches that silently didn't switch (env typos, last-wins duplicates,
  predicates that read a log line that prints in both arms);
- instruments that were themselves broken (an oracle that re-ran the work it
  claimed to remove; a capture that emitted empty output and got trusted).

Fulcrum's design position: **every one of those failures becomes a rule the
tool enforces** — a refusal or a label, with a self-test proving the
enforcement fires. See `fulcrum invariants` for the full set; each rule is
named for the scar that made it law:

| invariant | rule (short form) |
|---|---|
| SINK-LAW | both comparison arms use identical regular-file sinks; mixed-sink/half-rebased comparisons are refused |
| FROZEN-OR-LABELED | thawed/loaded-box wall numbers are refused for ranking, or loudly labeled |
| SHA-OR-VOID | every measured run's output is content-verified; a mismatch voids the cell |
| SPREAD-RESOLUTION | verdicts carry RESOLVED/UNRESOLVED + N-needed; sub-spread deltas are never findings; bimodality flagged |
| CAUSAL-OR-HYPOTHESIS | only tool-executed causal A/Bs rank as actionable; all else is HYPOTHESIS + the exact perturbation |
| EFFECT-VERIFIED-OR-FLAGGED | a kill-switch A/B is causal only if a counter proves the switch flipped |
| SELF-TEST-OR-NO-TRUST | the engine refuses trust until its self-tests (incl. corruption-fires tests) pass at this source version |
| FINGERPRINT-OR-NO-COMPARE | every number carries {sink, mask, freeze, binary sha, corpus sha, protocol}; cross-fingerprint ratios are refused |

## What one run produces

- a **cell scoreboard**: tool vs comparator wall per (workload × threads)
  cell, interleaved, sha-verified, fingerprinted, PASS/FAIL against the
  project's tie bar, RESOLVED/UNRESOLVED with N-needed, bimodality flags;
- a **ranked component table**: tier 1 = causally verified
  shipped-default-COSTS rows (same-binary kill-switch A/Bs, effect-verified),
  tier 2 = bounded hypotheses (trace decomposition + engine micro-profile),
  tier 3/4 = causal confirmations and nulls — each row with distribution
  health, RSS, and the exact re-verify command;
- **anomalies**: ledger contradictions (a live number diverging from a banked
  one under a *compatible* fingerprint), bank divergences, refused captures —
  verbatim, never silently dropped;
- a **DECISION BRIEF**: top action + causal evidence + preconditions + the
  exact command + the result that would falsify it.

## Architecture

```
fulcrum/
  core/        stats, trace engine, causal verdicts, fingerprints, ledger,
               invariants, decision engine, report   (stdlib-only, no project
               knowledge)
  adapters/    base.ProjectAdapter + one module per project (taxonomy, knob
               registry + effect predicates, routing guards, banks, re-verify
               command surfaces)
  selftests/   every guarantee tested, incl. named regression cases for the
               historical phantoms
  cli.py       fulcrum analyze/total/selftest/invariants/ledger
```

A **project adapter** supplies: the binary/launch matrix, corpora with
integrity pins, the knob registry with effect predicates, comparator tools,
environment-control policies (freeze/mask/sink), the trace taxonomy, and the
routing/contamination guard. The reference adapter is `adapters/gzippy.py`
(its shell launch policy lives in the host repo: `scripts/bench/decide.sh`).

## Quickstart (gzippy host repo)

```bash
scripts/fulcrum selftest            # must pass before any number is trusted
scripts/fulcrum decide              # one run on the bench box -> decision brief
scripts/fulcrum analyze <art-dir>   # re-render from pulled artifacts
scripts/fulcrum invariants          # the enforced rule set, with scars
scripts/fulcrum ledger              # banked results + fingerprints
```

## Writing an adapter

Subclass `fulcrum.adapters.base.ProjectAdapter`; supply a `Taxonomy` (span-name
prefixes for wait/compute/output and the wall-critical-thread frames), a knob
registry (`{name: Knob(env, pred, desc)}`), `effect_check` predicates that
prove each switch actually flipped, `parse_counters` + `routing_guard` for
contamination detection, and re-verify command surfaces. Your measurement
policy (how cells are launched under freeze/mask/sink discipline) stays in
your repo; fulcrum consumes the artifact dir it produces (`manifest.txt` +
per-cell sample files — see `core/decide.py` docstrings for the schema).

## Status / license

Extraction-ready in-repo package; stdlib-only, Python >= 3.9. License: MIT OR
Apache-2.0 (final choice at publication). PyPI candidate name: `fulcrum-perf`
if `fulcrum` is taken.
