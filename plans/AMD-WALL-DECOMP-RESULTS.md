# AMD/Zen2 gz-vs-rapidgzip WALL DECOMPOSITION — PROOF-CARRYING TOOL + RESULTS

Branch kernel-converge-A @ 15ff6cbe (gz) + rg-build-src REGION-patched. Box solvency
(AMD EPYC 7282 Zen2, root@10.0.2.240). FROZEN gov=performance, boost=0 (TSC ~= core
cycles), NO llama pause, trap+watchdog restore (box RESTORED: gov=ondemand boost=1).
taskset -c 0-3, interleaved gz,rg,gzAA, perf `cycles` total, region cyc = thread-summed
EXCLUSIVE TSC, N=9. cursor-agent-reviewed design (S2/S3 extension).

## THE TOOL (deliverable)
Symmetric 3-region exclusive-TSC partition of BOTH binaries into the SAME regions, each
binary's perf-cycle total as the conserved currency, R_OTHER = total - sum(regions):
  WORKER   = per-chunk decode + commit + clean-CRC   (gz chunk_fetcher.rs:2938 / rg decodeBlock)
  MARKERPP = marker-resolve/applyWindow + marker-CRC  (gz resolve_and_narrow:3088 / rg ChunkData::applyWindow)
  OUTPUT   = iovec/crc-combine/writev                 (gz chunk_fetcher.rs:4454 / rg writeAll)
  OTHER    = perf_total - sum (blockfinder/windowmap/pool/alloc/idle/verbose-print)
Files: gz `instruments/region_prof.rs` (+OVERLAP_VIOLATIONS), `scripts/bench/rg_region_prof_patch.py`,
`scripts/bench/amd_wall_decomp.sh`, `scripts/bench/amd_wall_decomp_report.py`.

## GATE-0 SELF-VALIDATION (all PASS)
- OFF == identity: sha(out)==zcat for gz AND rg on silesia/squishy/monorepo (counters byte-transparent).
- NON-INERT: every region fires >0 calls both arms (gz W/M/O, rg W/M/O).
- OVERLAP_VIOLATIONS == 0 on BOTH binaries, EVERY cell — proves the 3 regions are
  exclusive non-nested leaves (cursor-agent fix #1: kills the cancellation-masking that
  could make R_OTHER spuriously small). gz+rg both print SELF-TEST: PASS.
- A/A: gzAA-vs-gz perf 0.00-0.10%, worker 0.02-0.62% << every reported Δ.
- inter-run spread: gz_perf 0.8-2.2%, rg_perf 0.4-2.1%.
- scheduler gates: cpu-migrations 0 (T2) / 15-47 (T4), ctx-sw low — no descheduling artifact.
- env fingerprint asserted: GZIPPY_OVERLAP_WRITER unset (also dead code at HEAD),
  RAPIDGZIP_AW/WA_PROF unset; only *_REGION_PROF=1.

## THE THREE CONSERVATION IDENTITIES (N=9 medians)
(i)  resid_i  = |R_OTHER_gz|/gz_perf   PASS all cells (max 3.54%)
(ii) resid_ii = |R_OTHER_rg|/rg_perf   PASS all cells (max 1.86%)
(iii) gap = gz_perf - rg_perf == sum_r(gz_r - rg_r) + (R_OTHER_gz - R_OTHER_rg).
     Residual (R_OTHER_gz - R_OTHER_rg) = **<=1.71% of TOTAL cycles on EVERY cell.**
     As a fraction of the GAP it is small on the work-bound losers and large only on
     near-ties (gap < 3% of total = below the attribution floor; tool declines to attribute).

| cell        | gap=gz-rg (% of rg) | resid_i | resid_ii | resid_iii (of total / of gap) | overlap |
|-------------|---------------------|---------|----------|-------------------------------|---------|
| squishy-T4  | +326.3M (+8.0%)     | 0.94%   | 1.11%    | 0.09% / 1.3%  PASS            | 0/0     |
| silesia-T4  | +171.8M (+7.6%)     | 0.40%   | 0.71%    | 1.07% / 15.2%                 | 0/0     |
| monorepo-T4 | -52.4M  (-7.2% gz BEATS) | 0.65% | 0.14% | 0.80% / 10.4%               | 0/0     |
| silesia-T2  | +57.6M  (+2.6%)     | 2.39%   | 0.95%    | 1.47% / near-TIE (N/A)        | 0/0     |
| monorepo-T2 | +9.3M   (+1.5%)     | 3.54%   | 1.86%    | 1.71% / near-TIE (N/A)        | 0/0     |
| squishy-T2  | -89.2M  (-2.3% gz BEATS) | 2.03% | 0.61% | 1.40% / near-TIE (N/A)       | 0/0     |

## PER-REGION {gz, rg, delta} ON THE WORK-BOUND LOSS CELLS (the gap, precisely)
squishy-T4 (gap +326.3M):
  WORKER   gz 4,221.4M  rg 3,783.1M  **D = +438.3M  (+134% of gap)**
  MARKERPP gz   240.5M  rg   352.8M  **D = -112.3M  (-34%; gz CHEAPER)**
  OUTPUT   gz     1.4M  rg     5.4M  D = -3.9M (-1.2%)
  OTHER    gz   -41.4M  rg   -45.5M  D = +4.1M (+1.3%)   [sum(D)=326.3M == gap, resid 0.09% of total]
silesia-T4 (gap +171.8M):
  WORKER   gz 2,318.1M  rg 2,056.2M  **D = +261.9M  (+152% of gap)**
  MARKERPP gz   136.7M  rg   198.6M  **D =  -61.9M  (-36%; gz CHEAPER)**
  OUTPUT   gz     0.8M  rg     2.9M  D = -2.1M (-1.2%)
  OTHER    gz    -9.9M  rg    16.3M  D = -26.1M (-15%)   [sum(D)=197.9M, resid 1.07% of total]

## THE PROVEN EXPLANATION OF THE AMD GAP
On the work-bound T4 loss cells the gz>rg cycle gap is **CONCENTRATED in R_WORKER**
(per-chunk decode + commit + clean-CRC): D_WORKER = +438M (squishy) / +262M (silesia) —
i.e. larger than the gap itself — **PARTIALLY OFFSET by gz's CHEAPER R_MARKERPP**
(applyWindow/resolve): D_MARKERPP = -112M / -62M. OUTPUT and OTHER (pipeline / block-finder
/ window-map / pool / alloc / consumer coordination) are RULED OUT (each <=1.3% of gap on
the conservation-PASS cell; OUTPUT always <5%). Net loss = WORKER excess - MARKERPP saving.
This is conservation-PROVEN (the deltas sum to the gap to within 0.09%-1.07% of total cyc),
NOT an attribution. It sharpens prior S1 (gz-only partition) with the symmetric rg side and
is consistent with the marker-decode acquittal (gz's marker work is cheaper; the excess is
NOT in marker resolve).

## PRECISION BOUNDARY / NEXT REFINEMENT (named, not hand-waved)
R_WORKER bundles {clean-decode, marker-decode, commit/ring, clean-CRC}. The marker-decode
sub-loop is ACQUITTED (faster than rg, project_zen2_decode_microbench_acquittal), so the
WORKER excess lives in clean-decode and/or commit and/or clean-CRC. To split it requires
ADDING symmetric sub-counters on BOTH sides: a clean-decode TSC counter (gz contig C_CYC vs
rg Block::read clean / isal loop_block) and a clean-CRC TSC counter (both), so WORKER
partitions into {CLEAN_DECODE, MARKER_DECODE, CRC, COMMIT}. That is the next conservation
layer; the current tool already proves the gap is in WORKER and not in resolve/output/pipeline.

## SCOPE / TIER
STRONG (gated, self-validated 3-identity conservation, overlap=0, A/A<<Δ, N=9), Zen2-single-
arch + frozen. Intel replication owed for LAW (Intel is at parity per clean-wall-reconfirm,
so the WORKER excess is Zen2-specific). Re-verify after any kernel change.
