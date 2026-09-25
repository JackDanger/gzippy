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

### Leg (try4, 18:43Z, in flight) — the official adjudication leg
Both failure classes fixed: explicit `PATH=/root/.cargo/bin:...` +
`HOME=/root` exports, fresh `wave-retune9`, rivals ABSOLUTE-pathed
(`/root/pigz/pigz`, `/root/libdeflate/build/programs/libdeflate-gzip`),
marker-quoted heredoc. This leg's union decides the campaign.

## Decision
(PENDING try4 rescore — the v3.1 rules apply: pigz L9/T4 closes → SHIP with
the win; miss → keep size, cell OPEN, new lever under the ≤1% cap.)

## Artifacts in this directory
- `retune-verdict.txt` — try + rescore tail (the adjudication line)
- `try.json` / `try-rescore.json` — per-cell wall/size raws
- `layout-floors-l9t4/` — the floor calibration (two verified variants)
- S3 backup: `s3://gzippy-adjudication-20260923/retune-lap/ip-10-50-6-11/`
