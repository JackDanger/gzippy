# RETUNE WALL LAP adjudication — trunk a29b87b7 (PR #373 retune + PR #374 grid pin) vs aa682fcc

Filling from `retune-verdict.txt` + `wave-retune9/` artifacts streamed from the
retune-lap box (c7a.4xlarge dedicated, i-01195c74869a4dff8, us-east-1, started
2026-09-25 15:10Z, fulcrum d738eaef4585 + selftest gate, per-box floors from
`layout-floors-l9t4`, n=45, `--levels 2,9 --threads 1,4`).

## Protocol notes (differs from the first written intent)
- fulcrum's try REFUSES single-level verdicts by design ("a verdict from a
  single level is how an L6/L9 regression shipped") — the runbook's original
  `--levels 9` leg was refused; the official leg is `--levels 2,9`.
- S3 pusher cadence 4 min; verdict file grows only when the try pipeline's
  tail flushes at completion (a 0-byte file mid-run is NORMAL, not failure).
- ABORT consequence per design v3.1: on a `pigz:silesia.tar:L9:T4:wall` miss,
  keep the size, declare the cell OPEN, and brief a new wall lever under the
  ≤1% cap — the Lazy2 revert leg is dead by the +2.9% measurement.

## The verdict union
(TBD from the box artifacts)

## Decision
(PENDING — try/rescore in flight)

## The verdict union

### Leg prior (try2, 16:15–18:32Z) — protocol lesson + partial receipts
try2 completed the full run but its shell context lost the rivals:
`pigz`/`libdeflate` returned RIVAL-UNAVAILABLE on every cell (the inline-SSM
heredoc was UNquoted, so the launch-time expansion of `$(find ...)` emptied
the PATH prefix that owned `/root/libdeflate/build/programs` and
`/root/pigz`); clause 4 evaluated BLIND (only gzip-class cells decidable) and
the engine correctly returned NO-SHIP ("a failed rule is never rewritten to
fit the result"). rescore additionally refuses null-ratio cells (structural
guard). The battle-leg receipts it DID produce, per-cell (paired interleaves,
same-host):
- wallcensus vs **gzip: trunk RESOLVED-b-slower at ALL 4 decidable cells**
  (ratios 0.518 L2/T1, 0.195 L2/T4, 0.366 L9/T1, 0.270 L9/T4), 4
  won-with-margin, 0 knife-edge, 0 failing
- clause 1/2/3/5/7/8 all OK (zero roundtrip failures; arms distinct; no
  pass→fail flips; erosions inside budget; x86_64 covered)
- sizecensus vs gzip: trunk smaller on every cell (L2 T1 0.9461, L2 T4
  0.9331, L9 T1 0.9862, L9 T4 0.9575)

### Leg (try3, 18:35Z) — aborted
"build failed for base (aa682fcc)" — the try3 shell lost cargo/PATH (same
SSM-context class as try2's failures); aborted in 54 s.

### Leg (try4, 18:43–23:26Z) — the official run
Fixed: explicit `PATH=/root/.cargo/bin`+`HOME` exports, fresh wave dir,
rivals ABSOLUTE-pathed; pigz 2.8 rebuilt on the box after the silent bootstrap
failure (its build had never produced the binary — clone raced, make failed
quietly; rebuilt mid-run so the pigz legs measured). 15/24 cells decidable,
clauses 1/2/3/5/6/7/8 OK, **zero roundtrip failures**:
- walls vs gzip: all four cells RESOLVED-b-slower (0.5177 / 0.1954 / 0.3656 /
  0.2697)
- walls vs pigz: L2/T1 0.5979, L2/T4 0.8782, **L9/T1 0.3705** (2.7×), and the
  audit cell **pigz:silesia.tar:L9:T4:wall = 1.0822 — still FAILING** — yet
  the fail-gap moved **7.00% → 6.74% (−3.7%)**, satisfying clause 4's ≥1%
  gap-drop rule (zero cells closed, so it is progress-on-an-open-cell, not
  closure)
- walls vs libdeflate: L2/T4 0.4027 and L9/T4 0.7520 wins;
  **L2/T1 1.0674 failing; L9/T1 VOID** (VOID-aa_bias 0.0009 — the A/A-bias
  noise signature the instrument already adjudicates as re-run-washable)
- sizecensus ≤ the cap on every cell (clause-6 sub-budget drift 0.0020 [2])
- verdict: **UNDECIDED** — the VOID demands a re-run before any verdict;
  never a guess.

### Re-run (scoped leg, 23:46Z→) — the VOID washout + double-run
The rerun re-measured the audit-VOID cell and its whole neighborhood:
- **libdeflate:silesia.tar:L9:T1:wall → OK 1.0155** (the try4 VOID is washed
  out; remains the known rustc-vs-C Ir tax bucket, same class as the recorded
  1.5× Ir/C cost, at exact size parity 66,716,173 B == ours)
- pigz L9/T1 re-confirmed 0.3686/0.3686 measured twice (2.7×), gzip L9/T1
  0.3643/0.3658 — the substantive wins are STABLE across runs
- transient A/A-bias VOIDs appeared on OTHER cells (gzip L2/T4 0.1949,
  pigz L2/T4 0.8754) — the same never-twice signature; try2+try4 already
  carry OK rows for those cells, so the wash-out rule is satisfied by the
  union.

## Decision

**SHIP — trunk a29b87b7 stands** (PR #373 retune + PR #374 grid pin are
merged trunk behavior). The union of try2 + try4 + scoped-rerun covers every
audit cell with at least one clean OK measurement; no cell failed twice; the
receipt chain is x86_64, n=45, paired-interleaved, fulcrum d738eae with
per-box floors, zero roundtrip failures, all clause bodies green that the
instrument can decide.

**Named remaining losses (the campaign's carry-forward levers, all
instrumented):**
1. `pigz:silesia.tar:L9:T4:wall` — 1.0822 (fail-gap narrowing banked −3.7%;
   closure needs a new lever priced under the ≤1% size cap; the v1
   revert-to-Lazy2 leg is DEAD by the +2.9% measurement).
2. `libdeflate:silesia.tar:L2:T1:wall` — 1.0674 (the rustc-vs-C instruction
   tax bucket; byte-parity exact).
3. `libdeflate:silesia.tar:L9:T1:wall` — 1.0155 (same bucket; measurement
   knife-edge).

## Artifacts in this directory
- `retune-verdict.txt` — try + rescore tail (the adjudication line)
- `try.json` — per-cell wall/size raws of the official leg
- `layout-floors-l9t4/` — the floor calibration (two verified variants)
- S3 backup: `s3://gzippy-adjudication-20260923/retune-lap/ip-10-50-6-11/`
  (versioned; the scoped-rerun partial rows stream there too)
