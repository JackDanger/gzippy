# FINAL ADJUDICATION — gzippy PR #367 (s2-fixes/review-1 @ 4cd5deca) vs origin/main (8748c20f)

## The verdict union (this box: i-043ccd7298e2df783, c7a.4xlarge dedicated, fulcrum d738eae, floors 0.0001)

**Substantive result, stable across every artifact produced (full run + scoped re-measures):**
- clause 1 verify: zero roundtrip failures — every run
- clause 3: zero pass→fail flips — every run
- clause 4: BOTH previously-failing cells closed (pigz L2/T1 wall, libdeflate L9/T4 wall); scoped l9t1 additionally narrowed the fail-gap −19.8%
- clause 5: wall erosions inside the margin floor; size spends ≤ ceiling, explicitly `AUTHORIZED-<=1%-SIZE-SPEND` per the owner's 2026-09-05 directive
- clause 6: **improvement 3.2462× vs residual harm 0.0000**
- clause 7/8: archs + paired-interleaved method clean

**The noise signature (does not touch the verdict):** the A/A-bias gate (0.0005) voids one *random* cell per run (0.0006–0.0009); the same cell never voids twice in a row, never on both arms of the same run pair, and the full-grid census NEVER voided a substantive receipt. Re-measures wash it out:

| original full-try demand | receipt | status |
| --- | --- | --- |
| gzip L2/T1 wall | scoped-l2t1: `OK ratio=0.5188` | CLOSED |
| gzip L9/T1 wall | scoped-l9t1: `OK ratio=0.3643` | CLOSED |
| libdeflate L6/T4 wall | l2t1 sentinel re-measure `OK 0.5831` | CLOSED |

**scoped-l6t4 status:** the third scoped run (l6t4's own in-scope duplicate) was **cut at ~3h50 per the owner's wrap-up directive** — the demand it would redundantly re-cover already stands closed via the l2t1 sentinel receipt above. Its partial census at the cut (40 cell JSONs, base-arm cells green so far) is banked as `scoped-l6t4-partial/`, and the box pushed both bundles to S3 before termination.

## Recommendation

**SHIP** — merge gzippy PR #367. Evidence: one full-grid run + two completed scoped runs + attempt-1's independent census, all clause-green; three single-cell stochastic VOIDs explained, washed out by re-measurement on the same box, and never reproduced on the same coordinate twice. The "UNDECIDED" verdicts are the protocol's decidability demands, not failures; each demanded cell has a fresh OK receipt.

Post-merge (Playbook A): rebase the lever branches #363/#364, run `board-size.sh all` + the nightly census on the new trunk, re-pin `ir_budget.tsv` from the paired-Ir rows.

## Artifacts in this directory
- `README.md` — per-box reproduction recipe + timeline
- `attempt-1/` — salvaged census tapes of the killed first box (full grid)
- `attempt-2/layout-floors-l6t4/` — the per-box floors (0.0001 max |ln|)
- `attempt-2/wave-s2-final/` — the full try verdict artifact + per-cell JSONs (36 cells)
- `attempt-2/scoped-l2t1/`, `scoped-l9t1/`, `scoped-l6t4/` — scoped re-measure verdicts + try.json + rescore.json
- S3 backup: `s3://gzippy-adjudication-20260923/attempt2/i-043ccd7298e2df783/` (versioned, incremental every 4 min — survives any box death)

## LEVER VERDICTS (final lap, box i-01efd2cbe0d160203, dedicated c7a.4xlarge, 2026-09-24/25)

Both lever PRs were scoped-adjudicated on this box against the re-pinned trunk (aa682fcc), with per-box floors, n=45, sentinels, and the fulcrum d738eae rules. **Both are NO-SHIP by their own clause-4 rule** — the promised wall progress against the instrumented rivals did not materialize; the L6/L7 and L3 respective fail-gaps did not close by the required >=1% (lever1: 0.6897 -> 0.6867, -0.3%; lever2: 0.1604 -> 0.2564, worse). Artifacts: `final-lap/wave-out/{lever1,lever2}-verdict.txt` + both `wave-lever*/try{,-rescore}.json` (rescores reproduce the stored verdicts bit-for-bit).

- **#363 (good_match port)**: mechanism fully faithful — 28/28 decidable cells byte-identical between trunk and the port; the wall win the PR promised is absent at this level of the instrumented rival. NOT MERGED.
- **#364 (far_len3 port)**: on the rebase the byte-identity packaging was already void (14/23 representative files shift at L3); its scoped leg additionally found the targeted libdeflate-wall cells FURTHER away (gap up to 0.2564) plus two local floor gaps making two erosion cells UNDECIDED (a floor is never borrowed). NOT MERGED.

The levers stay parked on their branches with their artifacts and their PR threads carrying the verdicts; the trunk's composition stands as PR #367 + #371 (the S2 SHIP + the Ir re-pin).
