# AMD/Zen2 residual re-attribution — PRE-REGISTERED FALSIFIER
Branch kernel-converge-A @ 43923e84. Box solvency (Zen2). Pre-registered BEFORE measuring.

## PREMISE (S0) — re-confirm or report decayed
H0: gz_total_cyc(T4) > rg_total_cyc(T4) by >= ~4% on >=2 of {silesia,monorepo,squishy},
freq-robust (taskset 0-3, interleaved N>=9, GHz-spread<1%, perf cycles same metric both arms).
- CONFIRM premise: gz/rg cyc ratio >= 1.04 on >=2 cells, Δ > inter-run spread.
- DECAYED: ratio < 1.04 / within spread → report "premise decayed at HEAD; no work excess to attribute".

## ATTRIBUTION VERDICT (S1-S3)
Units: thread-summed EXCLUSIVE TSC (rdtsc), symmetric counters gz+rg. Verdict on ABSOLUTE
cycle deltas D_r = gz_abs_cyc(region) - rg_abs_cyc(region). Dtotal = gz_total - rg_total.
Regions: R_dec, R_crc, R_resolve(+narrow), R_output, R_materialize, R_other.

- LOCATED (a fat region): some non-decode region R has D_r >= 0.40*Dtotal AND gz_r > rg_r,
  replicated on >=2 cells, D_r > A/A spread. -> name it the next lever.
- DISTRIBUTED/INTRINSIC: no single non-decode region reaches 0.40*Dtotal (excess spread
  across >=3 regions or buried in R_other with no instrumentable fat) -> report honestly
  (NIGHT35 pattern; AMD residual near a floor).
- DECODE-DOMINATED (unexpected, would reopen acquittal): R_dec D_r >= 0.40*Dtotal with
  gz_dec > rg_dec -> contradicts the acquittal; flag for re-measure, do NOT bank.

## GATE-0 (all required, else number void)
- non-inert: every region counter >0 on BOTH arms (rg R_materialize may be ~0 BY DESIGN —
  must be proven structurally absent, not silently zero).
- OFF==identity: GZIPPY_* / RAPIDGZIP_* unset -> sha(out)==zcat on BOTH; counters compiled
  out / branch-off.
- A/A: gzAA region cyc within inter-run spread of gz.
- conservation: sum(regions)+R_other == TSC total; R_other>=0; report R_other fraction.
- matched denominators per region (region-correct, NOT total-output for resolve/decode).
- Gate-4: path=ParallelSM, HEAD sha 43923e84, gzippy-native.

## T2 (separate)
Re-examine whether AMD-T2 residual is teardown/RSS (prior finding: peak RSS gz 97 vs rg 59
=1.65x, teardown ∝ RSS +0.054ms/MB) vs a serial-wrapper phase. Confirm via phase_timing
instrument-wall vs process-wall gap, llama-free-ish (taskset, GHz-gate).
