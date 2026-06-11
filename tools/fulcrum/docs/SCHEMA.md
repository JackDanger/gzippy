# Fulcrum run-artifact schema (the documented loader contract)

This is the contract between a project's **measurement policy** (the shell/CI
side that runs the tool under freeze/mask/sink discipline) and the fulcrum
**decision engine** (`fulcrum analyze <art-dir>` / `fulcrum.core.decide`).

Two ways to satisfy it:

1. **Produce this directory layout** and use the default loader
   (`ProjectAdapter.load_run` → `load_run_documented`). The gzippy reference
   policy (`scripts/bench/_decide_guest.sh` in the host repo) does this.
2. **Keep your own artifact layout** and override `ProjectAdapter.load_run`
   to return the run-dict shape in the last section. The decision engine
   consumes only that shape; it never touches the filesystem layout itself.

## Directory layout (documented schema)

```
<art-dir>/
  manifest.txt                       # required — provenance, key=value lines
  cell_<corpus>_T<threads>/          # one per measured (workload × threads)
    wall_gz.txt                      # tool-under-test wall samples, seconds,
                                     #   whitespace-separated (interleaved capture)
    wall_rg.txt                      # comparator wall samples, seconds
    prof.txt                         # optional — engine micro-profile capture
                                     #   (opaque to core; adapter.parse_microprofile)
    trace.json                       # optional — Chrome-trace (B/E events)
    verbose.txt                      # optional — counter sidecar for the
                                     #   routing/contamination guard
    knob_<name>/                     # one per same-binary kill-switch A/B
      base.txt                       # baseline-arm wall samples, seconds
      knob.txt                       # feature-ALTERED-arm wall samples
      meta.txt                       # key=value: knob, env, pred, cell, mask,
                                     #   sha_ok, rss_base_mb, rss_knob_mb
  knob_effects_<corpus>_T<threads>/  # optional — effect-predicate captures
    effect_base_<name>.txt           # counter output, baseline arm
    effect_knob_<name>.txt           # counter output, altered arm
```

Cell directory names must match `cell_([a-z0-9]+)_T(\d+)`; corpus ids are
lowercase alphanumeric.

## manifest.txt keys

Required for ranking:

| key | meaning |
|---|---|
| `runid` | unique id for this measurement run (ledger idempotence key) |
| `bin` / `bin_sha` | tool-under-test path + sha256 (fingerprint `bin_sha`) |
| `freeze_state` | `frozen` \| `acknowledged` \| anything else (refused unless `--allow-thaw`) — must come from a sysfs/equivalent READBACK, not a claim |
| `quiet_state` | `quiet` \| `loaded-acked` \| ... (same gate) |
| `cell_done=<corpus>:<T>:mask=<M>[:maskd=<D>]:sha_ok=<0\|1>` | one per completed cell; `sha_ok=1` is the SHA-OR-VOID gate; `maskd` is the DERIVED taskset readback (governs the fingerprint mask) |

Fingerprint fields (FINGERPRINT-OR-NO-COMPARE; missing ⇒ `unknown` ⇒ the
cell is labeled FP-INCOMPLETE and never banked):

| key | fingerprint field |
|---|---|
| `protocol` | measurement-protocol version (e.g. `fulcrum-v3`) |
| `sink_gz` / `sink_rg` | sink class per arm; `sink_gz_derived` / `sink_rg_derived` are the stat-DERIVED duplicates — derived governs, a contradicting self-report is flagged DERIVED-MISMATCH |
| `corpus_<corpus>_sha` | decompressed-content pin per corpus (SHA-OR-VOID) |
| `host_cpu_model`, `host_kernel`, `host_id` | DERIVED host identity → fingerprint `host` = `cpu|kernel|id`; all three or `unknown` |
| comparator version | adapter probe (`comparator_version(manifest)`); the gzippy adapter normalizes `rg_version=` — supply your comparator's version under a key your adapter reads (default key: `comparator_version`) |

Context (rendered in the header, cross-checked where derivable): `feature`,
`rg_version`, `governor`, `no_turbo`, `runnable_avg`, `n`, `knob_n`,
`started`, `finished`. Optional records: `knob_done=...`,
`knob_sha_fail=<corpus>:<T>:<name>` (a knob arm with wrong bytes — its own
finding, never ranked).

## Sample-file format

Wall samples are plain decimal **seconds**, whitespace/newline separated.
Convention: the capture interleaves arms (gz/rg or base/knob) and drops the
warm-up iteration before writing.

## The run-dict shape (what a custom `load_run` must return)

```python
{
  "dir":      art_dir,
  "manifest": {                      # parse_manifest() shape
     "runid": str, "bin_sha": str, "freeze_state": str, "quiet_state": str,
     "cells_done": [str], "knobs_done": [str], "knob_sha_fail": [str],
     "cell_meta": {(corpus, T): {"mask": str, "maskd": str, "sha_ok": "1"}},
     ...every other manifest key verbatim...
  },
  "cells": {
    (corpus: str, T: int): {
      "dir": str,
      "gz":  [float, ...],           # wall seconds, tool under test
      "rg":  [float, ...],           # wall seconds, comparator
      "prof": object|None,           # adapter.parse_microprofile output
      "trace": path,                 # may not exist (row skipped, anomaly)
      "verbose": path,               # may not exist
      "knobs": {
        name: {"base": [float], "knob": [float], "meta": {k: v}},
      },
    },
  },
  "effects": {knob_name: {"base": str, "knob": str}},   # counter text per arm
}
```

## Decision-table row contract (adapter-supplied rows)

`microprofile_rows` (and any custom row source) must emit dicts with:
`component, cells, attrib, status, dist, verify, tier, rank_ms` plus the
**structured fields** the brief builder keys on (never string-matched):

- `kind`: `"knob"` | `"engine"` | `"pipeline"`
- `perturb_cmd`: the exact pre-registered perturbation (tier-2 HYPOTHESIS
  rows; the brief falls back to `adapter.perturbations["compute"]`, then to
  `verify`)
- `reverted`: bool (knob rows; set via `Knob(..., reverted=True)`)

## Ledger

Analyzed cells are banked to an append-only jsonl ledger (default
`artifacts/fulcrum/ledger.jsonl`, override `FULCRUM_LEDGER`). Record kinds:
`cell`/`knob` measurements (optional `status: pending-reconcile`),
`supersede` (retires an anchor, may promote a pending row), `invalid`
(retires a measurement error). **Append-only is a convention the tooling
upholds, not an OS guarantee** — see `core/ledger.py` for the tamper-evidence
hash chain.
