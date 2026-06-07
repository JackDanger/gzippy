# Lever B CEILING ORACLE — pre-registered falsifier (Stage-5b)

Written BEFORE running. Branch reimplement-isa-l @ cbfb256. T8, silesia-large,
pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM=1, path=IsalParallelSM asserted,
interleaved best-of-N≥9, sha-verified vs `gzip -dc`.

## Context (what is already established — NOT re-litigated here)
- Lever B (decode rate) is CONFIRMED on the critical path by the slow-injection
  perturbation matrix (`/tmp/leverB-matrix.txt`): baseline 0.9171s; ondemand
  spin+50% 0.9722s; sleep+50% control 0.9765s (delta survives freq-neutral);
  spin+100% 1.1029s. Monotonic/proportional ⇒ decode region gates the wall.
- BUT slow-down slope ≠ speed-up ceiling (CLAUDE.md Measurement rule 3). To BOUND
  the speed-up we must REMOVE the decode region (oracle) and measure the
  interleaved wall — never extrapolate the slope through the unlocated knee.
- The oracle already exists: `src/decompress/parallel/decode_bypass.rs`
  (GZIPPY_BYPASS_CAPTURE / GZIPPY_BYPASS_DECODE / GZIPPY_SLEEP_DECODE_NS),
  wired in `chunk_fetcher.rs:2144-2162`. Replay returns captured ChunkData via
  memcpy (~0 inner-Huffman CPU) keeping the FULL downstream coordination chain
  (apply_window, marker resolve, publish, consumer). Cache miss → real decode
  (keeps output byte-exact).

## Three measured points  (ROLES CORRECTED after advisor review a62eab35)
- (A) **THE FLOOR — decode≈0, BYTE-EXACT, the verdict.** gzippy_bypass = replay of
      a FULL capture. The replayed form-B chunks STILL carry markers ⇒
      apply_window / marker-resolve (the L_resolve term the model says BINDS)
      STILL runs. So (A) is the floor that PRESERVES the binding publish-chain
      term. MUST sha-match `gzip -dc`. Replay misses are STRUCTURAL (run-to-run
      speculative key drift; validated locally ~89% on an 18-chunk corpus); they
      fall back to REAL decode ⇒ keep bytes correct AND only INFLATE the floor
      (upper bound). So VOID only on SEVERE contamination (hit% <90%); report hit%
      either way. Gate sd% ≤5% (swap-thrash ⇒ bimodal wall ⇒ INVALID). Confounds: instrument load (660MB parse + 39× ChunkData rebuild +
      500MB CRC, all INSIDE the timed wall) and any swap INFLATE (A) ⇒ upper-bound
      lean; the turbo/frequency DEFLATING confound the advisor flags is CONTROLLED
      by the host lock (no_turbo=1, governor=performance, min=max pinned — verify
      in the run's host log).
- (A2) **DECOMPOSITION PROBE, NOT a floor.** GZIPPY_SLEEP_DECODE_NS=0 + meta
      capture returns CLEAN ZEROED chunks (no markers) ⇒ it ELIDES L_resolve /
      apply_window. So A2 = floor MINUS resolve (a SMALLER, different pipeline);
      it must NOT carry the verdict. Value: (A) − (A2) ≈ L_resolve + load, to
      cross-check the model's L_resolve ≈ 0.78s and to confirm harness overhead
      isn't itself the cost. Full coverage (start_bit fallback), no swap.
- (B) RAPIDGZIP-RATE projection (WEAK — markers ELIDED). SLEEP_DECODE_NS=66_420_000
      + meta capture ALSO returns clean zeroed chunks ⇒ also drops L_resolve, so it
      UNDER-states by ~0.78s. The real rapidgzip-rate projection comes from the
      MODEL: at d_w→66.42ms worker-bound=frontier+(N/T)·66.42=405ms but
      publish-chain (L_resolve unchanged) still BINDS at 838ms ⇒ projected wall
      still ≈0.84s. B's measured number is reported only as a resolve-elided lower
      bound that corroborates the model.

## Model cross-check (analytic, from /tmp/leverB-R1-baseline, residual +0.0% CONFIRMED)
gzippy T8 model: N=39, d_c=84.04ms, d_w=122.28ms, L_resolve(mean)=19.93ms,
frontier=81.40ms, tail=0.85ms.
  worker-bound  = frontier + (N/T)·d_w_eff = 491.10ms
  publish-chain = frontier + N·L_resolve   = 838.86ms  [BINDS]
  wall_pred = max(.,.) + tail = 839.71ms (binding: publish-chain).
PREDICTION for the floor: decode→0 collapses d_w and shrinks `frontier` toward
header time, but L_resolve (marker resolve / apply_window) is PRESERVED by the
oracle (captured form-B chunks still carry markers). So publish-chain ≈
(small frontier) + 39·19.93ms ≈ 0.78–0.84s STILL BINDS. Model ⇒ expect FLOOR
materially > 0.6s ⇒ CEILING CAPPED. The oracle is the verdict; if oracle and
model disagree, that disagreement is itself the finding.

## PRE-REGISTERED FALSIFIER  (the model is the pre-registered prediction: (A)≈0.78–0.84s ⇒ CAPPED)
The floor is **(A)** (it preserves the binding L_resolve term); A2/B are probes.
- **CEILING CAPPED ABOVE rapidgzip** (model's prediction ⇒ inner-loop decode alone
  insufficient; another component binds): FLOOR wall (A) stays materially > 0.6s
  with decode≈0 (expected band [0.78,0.84]s from the model). CAPPED soundness does
  NOT rely on (A)'s absolute magnitude (which load may inflate): it is anchored by
  the decomposition — if (A) − (A2) ≈ model L_resolve ≈ 0.78s, the marker-resolve
  term ALONE exceeds 0.6s and lives in the floor, so true floor > 0.6s regardless
  of load. THEN name the binding component from the FLOOR-pass fulcrum trace
  (apply_window / marker-resolve / publish-chain L_resolve / consumer.dispatch_recv
  / writev) with file:line.
- **CEILING REACHES TIE** (⇒ inner-loop decode IS the right next arc AND the model
  is WRONG — must reconcile, not cherry-pick): FLOOR wall (A) ≤ ~0.55s, sha-EXACT,
  hit% ≥90%, sd% ≤5%. Because the host lock pins frequency (deflating confound
  controlled) and load only inflates, (A) ≤ 0.55 ⇒ true floor ≤ 0.55. This
  CONTRADICTS the confirmed model (predicted 0.78–0.84) ⇒ investigate which is
  wrong (e.g. L_resolve overlaps more than the serial publish-chain model assumes)
  before declaring inner-loop the arc.
- **VOID**: gzippy_bypass sha-DIVERGES from `gzip -dc`; or path≠IsalParallelSM; or
  replay hit% <90% on (A) (floor heavily contaminated by real decode); or (A) sd% >5%
  (swap-thrash bimodal wall); or capture wrote 0 chunks; or host log lacks
  no_turbo=1 (frequency not pinned ⇒ deflation uncontrolled). STOP and report.

## Disproof attempts baked in
- Decomposition (A)−(A2) vs model L_resolve(0.78s): agreement ⇒ load small, (A) is
  the true floor; large gap ⇒ load inflated (A), fall back to (A2 clean-coord +
  model L_resolve) as the floor estimate. Either way CAPPED is anchored.
- Frequency: host lock pins turbo OFF + governor=performance (verify no_turbo=1 in
  host log) — neutralizes the bypass-lowers-occupancy turbo-headroom deflation.
- Hit% gated ≥90% (misses are structural & inflating) and swap-in logged; sd% ≤5% gate kills a bimodal/swap-thrashed
  floor run rather than reporting a lucky min.
- gzippy_normal interleaved as the baseline anchor (must reproduce ~0.917s ±spread,
  else the harness itself drifted and nothing is comparable).
- Floor-pass fulcrum trace captured (decode≈0) so the binding component under a
  free decode is NAMED, not attributed.

## Status: COMPLETE (CEILING_DONE rc=0, host restored no_turbo=0).

## MEASURED RESULTS (T8, /tmp/leverB-ceiling.txt; host pinned no_turbo=1 governor=performance)
| point | min | note |
|---|---|---|
| CEIL_BASELINE_NORMAL | 0.9287s | anchor (reproduces ~0.917s ✓) |
| CEIL_FLOOR_A (decode≈0, byte-exact, sha=OK, hit%=95.2) | **3.6670s** sd%=0.8 | **LOAD-CONTAMINATED — DO NOT read magnitude** |
| CEIL_RAPIDGZIP | 0.5375s | TIE target |
| CEIL_A2_SLEEP0 (resolve-ELIDED, garbage) | 0.4558s | probe |
| CEIL_B_SLEEP66 (rapidgzip-rate, resolve-ELIDED, garbage) | 0.7446s | weak lower bound |
| CEIL_DECOMP A−A2 | 3.2112s | dominated by LOAD, not L_resolve |

## CORRECTED INTERPRETATION (floor-trace wall-critical decomposition — supersedes the raw CEIL_VERDICT=CAPPED headline)
The script's `CEIL_VERDICT=CAPPED` is mechanically true (3.667s > 0.6s with decode≈0) but
**rests on a load-contaminated magnitude and is NOT the real ceiling verdict.** Pass A replays a
656MB on-disk capture (memcpy + 41× ChunkData rebuild + 500MB CRC, all IN-WALL; swap_in=112).
The 3.667s is the replay-instrument cost (`pool.run_task` umbrella 3137ms wall-crit,
`worker.decode_chunk` 25200ms busy doing memcpy+CRC), NOT the pipeline.

Looking PAST the load to the wall-critical DECOMPOSITION of the floor trace
(/tmp/leverB-ceiling-art/artifacts-ceiling/trace_floor_T8.json, fulcrum):
- Baseline's dominant binder = **inner clean decode 548.7ms (65% wall-crit)**. Under free decode
  it **VANISHED and nothing comparable replaced it.**
- `apply_window` stayed **0ms wall-crit** (parallel on the pool) in BOTH runs — it never became the binder.
- `consumer.dispatch_recv` barely moved 169→204ms. `writev` ~1ms.
- Reconstructed genuine coordination residue ≈ **~400ms** (bootstrap ~75ms + serial publish-chain
  N·L_resolve ~307ms + tail/dispatch ~30ms) — which sits **BELOW** rapidgzip's ~0.52s.

⇒ **DIRECTIONAL VERDICT: inner Huffman decode IS the arc** (the 548ms binder is the decode itself;
no coordination span approaches 0.55s). This is a STRONG HYPOTHESIS, not a confirmed ceiling,
held back by two honest caveats:
- (a) L_resolve is window-wait-coupled: floor median collapsed 6.72ms→162µs because free decode
  delivered windows instantly. At merely-fast (not free) decode, L_resolve interpolates between
  7.88 and 19.93ms ⇒ the ~307ms publish-chain term is the decode=0 asymptote, not the
  rapidgzip-speed value. (Counter-evidence the arc is real: floor median 162µs ≈ rapidgzip's
  73µs, and rapidgzip reaches 0.52s with the SAME resolve machinery ⇒ resolve stays cheap when
  windows arrive promptly.)
- (b) The ~400ms is reconstructed/spliced, not directly measured (fulcrum never measured a clean
  coordination wall; frontier under fast decode scales with decode speed).

## OUTSTANDING DISPROOF (to convert hypothesis → survived-disproof finding)
Re-run the floor with a **fast-but-REAL decode oracle** — replay from RAM with CRC stripped, NOT a
656MB on-disk capture+CRC — and check L_resolve: stays ~7.88ms ⇒ contention story holds, inner
decode is a clean arc to ~0.52s; climbs toward 19.93ms ⇒ L_resolve is window-wait-coupled and
inner decode does NOT cleanly reach TIE. Routed to the Stage-5c advisor for adjudication before
committing the next arc.

## Status: detached guest run COMPLETE. Completion: `grep CEILING_DONE /tmp/leverB-ceiling.txt`
