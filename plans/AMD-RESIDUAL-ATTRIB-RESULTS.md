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

## S1 — gz EXCLUSIVE-TSC REGION PARTITION (FROZEN gov=perf+boost0 so TSC~=core-cyc; N=9, T4)
GZIPPY_REGION_PROF=1, taskset 0-3, no llama pause. Gate-0: non-inert (W=17,M=16,O=17 all>0);
OFF==identity sha==zcat; conservation sum(regions)~=perf_total (R_OTHER within +-0.5%).
Medians:

| cell      | gz_total      | rg_total      | Dtotal (+%)      | R_WORKER(decode+commit+crc) | R_MARKERPP(resolve+narrow+crc) | R_OUTPUT | R_OTHER |
|-----------|---------------|---------------|------------------|------------------------------|---------------------------------|----------|---------|
| silesia-T4| 2,499,108,520 | 2,263,735,361 | +235,373,159 (+10.4%) | 2,320,336,900 (92.8%) | 184,851,044 (7.4%) | 826,840 (0.03%) | -6.9M (-0.28%) |
| squishy-T4| 4,508,097,589 | 4,048,853,377 | +459,244,212 (+11.3%) | 4,208,645,504 (93.4%) | 317,401,812 (7.0%) | 1,447,292 (0.03%) | -19.4M (-0.43%) |

KEY: **R_OTHER ~= 0 (-0.3%/-0.4%)** — the pipeline/blockfinder/windowmap/pool/consumer-
coordination/alloc bucket is NEGLIGIBLE. **R_OUTPUT ~= 0.03%** (writev to /dev/null).
=> The entire +10-11% T4 work excess lives in the two COMPUTE regions only:
R_WORKER (decode+commit+clean-crc, ~93%) + R_MARKERPP (resolve+narrow+marker-crc, ~7%).
RULES OUT: pipeline/consumer coordination, block-finder, window-map, allocation, AND output
copy as the fat region. (Note frozen Dtotal +10-11% > S0 unfrozen +8% — boost-off base-clock
makes the work gap cleaner.) NEXT: split Dtotal between {R_MARKERPP} and {R_WORKER bucket}
via symmetric rg applyWindow counter (rg_total - rg_applyWindow = rg worker-bucket).
