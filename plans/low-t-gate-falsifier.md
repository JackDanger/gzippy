# JOB 1 — LOW-T (T4) GATE: pre-registered falsifier (OWNER, 2026-06-08, HEAD d56cb0f5)

PRE-REGISTERED BEFORE MEASURING (Measurement PROCESS rule 5, charter rule). The
verdict below is bound to the falsifier as written; it does not get rewritten after
seeing the number.

## What is measured
ONE tight (<=5% inter-run spread) quiet-box (Plex+noisy-LXC FROZEN, INSTANTANEOUS
procs_running gate <=2.0) measurement at **T4** of:
- `ocl_cf-T4`   = gzippy-isal build, `GZIPPY_ISAL_ENGINE_ORACLE=1` engine-isolation
                  kind (ISA-L clean engine, off the production decode graph),
                  fallbacks==0 asserted in-script, sha-verified vs corpus.
- `native-T4`   = gzippy-native build, same-sink kind (PRODUCTION), sha-verified.
- `rapidgzip-T4`= rapidgzip 0.16.0, interleaved against BOTH above (consistent ref).

Both gzippy runs interleave with rapidgzip on the SAME frozen box → ratios are
directly comparable. Matched same-sink: both -> regular file on /dev/shm. path=ParallelSM
asserted in-script. best-of-N>=9 interleaved.

Ratio = rg_min / gzippy_min (so >1.0 means gzippy faster; the bar is gzippy >= 0.99x rg).

## Falsifier (binding)
- **F-ENGINE-CLOSABLE**: `ocl_cf-T4 >= 0.99x` AND `native-T4 < 0.99x`
  => the low-T gap is ENGINE-CLOSABLE => a full-kernel hand-asm rewrite is JUSTIFIED
  at low-T. Report it as the next gated build; do NOT start it (user's call).
- **F-NON-ENGINE**: `ocl_cf-T4 ~ native-T4` (BOTH < 0.99x, delta < inter-run spread)
  => the low-T gap is NON-ENGINE => close the engine chapter at low-T. The lever is
  scheduling/bootstrap/pipeline. Name the binder from the per-stage trace
  (fulcrum_total / consumer decompose), NOT from producer-side attribution. The
  located candidate is project_confirmed_offset_prefetch_gap (head-of-line stalls at
  confirmed offsets).
- **BLOCKED**: if the box cannot be made quiet (procs_running gate fails / runner
  hard-fails host-loaded) => report BLOCKED, do NOT bank a loaded number (charter
  rule 8 / past ocl_cf drift). RATIO-only (ALLOW_LOAD) is NOT a substitute for the
  <=5%-spread gate this turn — the whole point is the tight number the prior turns
  could not get.

## Tie-break detail
- If `ocl_cf-T4 >= 0.99x` AND `native-T4 >= 0.99x`: low-T already at parity on BOTH;
  no engine fork needed; report TIE.
- If `ocl_cf-T4 < 0.99x` AND `native-T4 < 0.99x` AND `ocl_cf - native >= spread`:
  PARTIAL — engine closes SOME of the gap but a non-engine residual remains; the fork
  captures the engine share only, the residual stays non-engine (report both shares).

## MEASURED (2026-06-08, frozen quiet box 10.30.0.199, runnable_avg 1.00-1.50, T4, interleaved best-of-N)
| build | ratio rg/gz | gz min | rg min | gz spread | rg spread | sha | coverage |
|-------|------------|--------|--------|-----------|-----------|-----|----------|
| ocl_cf (ISA-L engine oracle, bin b9eb0a73) | **0.899x** | 545ms | 490ms | 3% | 4% | OK | isal_chunks=14 fallbacks=0 |
| native (pure-Rust prod, bin 710a6dc) | **0.740x** | 652ms | 482ms | 4% | 6% | OK | n/a (production) |

Both TIGHT (<=5%). Same-sink (->/dev/shm regular file), path=ParallelSM asserted,
N=11/13 interleaved, host bench-locked QUIET (Plex+7 noisy LXCs frozen).

## VERDICT (PARTIAL, pre-registered tie-break #2)
- ocl_cf-T4 0.899x < 0.99x => F-ENGINE-CLOSABLE NOT met (the engine alone does NOT
  reach parity at T4; even REAL ISA-L, the fastest engine, loses 0.899x).
- ocl_cf (0.899) - native (0.740) = 0.159x >> spread (3-4%) => NOT F-NON-ENGINE
  either (the two are NOT equal).
- => PARTIAL: the engine closes a large (>>spread, ~5x spread) share of native's T4
  deficit (pure-Rust->ISA-L moves 0.740->0.899, +0.159x of the 170ms native deficit).
  ADVISOR-CORRECTED SPLIT (the ocl_cf "ISA-L ceiling" is a BLEND, source-confirmed
  gzip_chunk.rs:128-131,196-223: ONLY the clean 32KiB-window continuation goes through
  ISA-L FFI; the markered prefix + chunk-0 marker bootstrap STAY pure-Rust):
    * engine share >= 0.159x  (UNDERESTIMATE — ocl_cf still runs pure-Rust marker work)
    * non-engine residual <= 0.101x / ~55ms (UPPER BOUND — contains marker-prefix
      pure-Rust engine + per-chunk ISA-L FFI/handoff overhead, NOT pure scheduling)
  ZERO-MARGIN RISK (advisor): an asm engine's BEST case == ISA-L == 0.899x at T4 (never
  0.99 alone); parity needs BOTH levers fully realized to land at ~1.0 with no slack.
- NAMED NEXT LEVER: TWO co-primary levers, both gated, neither started:
  (1) ENGINE at low-T: a full-kernel hand-asm rewrite is justified to CAPTURE the
      ~0.159x engine share (NOT to reach 1.0x alone). User's call (do NOT start).
  (2) NON-ENGINE residual ~55ms: survives real ISA-L => scheduling/bootstrap/pipeline.
      Name the binder from the per-stage trace (fulcrum_total/consumer decompose);
      located candidate = project_confirmed_offset_prefetch_gap (head-of-line stalls
      at confirmed offsets). This is the LOWER-RISK lever and helps BOTH builds.

## Disciplines
Quiet via host bench-lock (freezes Plex+noisy LXCs); engine-isolation asserts
fallbacks==0; sha vs 028bd002...cb410f; do NOT start the asm rewrite.
