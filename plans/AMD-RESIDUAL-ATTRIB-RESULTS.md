# AMD/Zen2 residual re-attribution — RESULTS (live)
Branch kernel-converge-A @ 43923e84. Box solvency (Zen2, root@10.0.2.240). gzippy-native
(parallel-sm+pure), rg /root/rg-build-src (WA_PROF-patched, identity when off).
Method: taskset -c 0-3, interleaved gz,rg,gzAA, perf `cycles` (work metric, freq-invariant),
NO freeze, NO llama pause (load-robust). cyc is reproducible: inter-run spread ~1.4%, A/A ~0.1%.

## S0 — PREMISE RE-CONFIRM (perf cycles, N=9, T2+T4) — gz_total vs rg_total
gate: best-of-9 (min cycles); ratio gz/rg (>1 = gz more work). Stability via within-tool
rep spread + A/A (NOT the gz-vs-rg "GHz" which differs by tool occupancy — cosmetic mislabel fixed).

| corpus   | T | gz_cyc(best)  | rg_cyc(best)  | gz/rg | gzspr% | rgspr% | A/A% | verdict        |
|----------|---|---------------|---------------|-------|--------|--------|------|----------------|
| monorepo | 2 | 617,741,087   | 607,518,117   | 1.017 | 1.4    | 1.4    | 0.1  | weak +1.7% ~tie|
| monorepo | 4 | 676,490,586   | 722,477,048   | 0.936 | 2.1    | 6.7    | 0.1  | gz BEATS 6.4%  |
| silesia  | 2 | 2,256,731,503 | 2,181,934,029 | 1.034 | 1.0    | 1.4    | 0.5  | gz +3.4% (real)|
| silesia  | 4 | 2,458,589,991 | 2,274,973,812 | 1.081 | 1.4    | 0.8    | 0.1  | CONFIRM +8.1%  |
| squishy  | 2 | 3,820,616,938 | 3,875,382,564 | 0.986 | 1.4    | 1.5    | 0.4  | gz beats 1.4%  |
| squishy  | 4 | 4,425,518,238 | 4,068,165,343 | 1.088 | 0.9    | 1.4    | 0.1  | CONFIRM +8.8%  |

PREMISE HOLDS at HEAD: WORK-bound gz>rg on silesia-T4 (+8.1%, +184M cyc), squishy-T4
(+8.8%, +357M cyc), silesia-T2 (+3.4%). gz also retires MORE instructions (smoke: silesia
+5.4%, squishy +7.3% insn) => genuinely more WORK (extra computation), not pure stalls —
points to an extra non-decode computational region, not a cache/latency artifact.
ATTRIBUTION CELLS = silesia-T4, squishy-T4 (T4 losers). monorepo-T4 / squishy-T2 gz beats.
