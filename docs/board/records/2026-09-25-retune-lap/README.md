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

## Artifacts in this directory
- `retune-verdict.txt` — try + rescore tail (the adjudication line)
- `try.json` / `try-rescore.json` — per-cell wall/size raws
- `layout-floors-l9t4/` — the floor calibration (two verified variants)
- S3 backup: `s3://gzippy-adjudication-20260923/retune-lap/ip-10-50-6-11/`
