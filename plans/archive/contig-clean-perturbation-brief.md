# DISPROOF BRIEF — contig clean symbol-decode causal perturbation (where the ≤0.11× is)

## CLAIM TO DISPROVE
The gzippy-native FOLD **contig clean symbol-decode** loop (`decode_clean_into_contig`,
marker_inflate.rs) IS on the T8 critical path, so the residual native_fold→ocl_cf gap
(~0.155× at the wall) lives in inner-Huffman symbol-decode COMPUTE — NOT in a
ring-write or drain (Stage 2 removed the clean-path ring-write). Therefore inner-Huffman
rate techniques (BMI2, wider multi-literal store, packed-u32 LUT) applied to this loop
will move the wall, bounded by ocl_cf 0.925× (engine-removed ISA-L ceiling), never the
slope.

## WHY THIS PERTURBATION (vs prior knobs)
The pre-existing `GZIPPY_SLOW_MODE` knob lived ONLY on the ring-based
`read_internal_compressed_specialized::<false>` path. After Stage-2 copy-free-to-final,
the PRODUCTION FOLD clean tail decodes via `decode_clean_into_contig` (no ring), which
the old knob did NOT touch. So no prior perturbation actually tested the production
clean loop. This turn I wired the SAME clean knob (`GZIPPY_SLOW_MODE`/`GZIPPY_SLOW_KIND`)
into `decode_clean_into_contig` (per-decode-event inject, snapshot once before loop).
Byte-transparent: OFF==ON==028bd002…cb410f on guest x86_64 + arm64. Knob fires:
~23.6M/27M clean-loop inject hits confirm the contig loop is exercised.

## METHOD (Measurement PROCESS rules 1,2,5)
Locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved measure.sh
N=11 best-of-11/pass, RAW=211968000, output-verified=OK every run, T8 (-p 8).
Contenders: off / spin50 (+50%) / spin100 (+100%) busy ALU / sleep100 (+100%
frequency-NEUTRAL control, deschedules core) / rg.

## RESULT (two interleaved passes)
PASS 1 (load 0.99): off 0.1652 | spin50 0.1829 | spin100 0.2154 | sleep100 0.2278 | rg 0.1271
  off/rg = 0.77×; spin50 0.903× off; spin100 0.767× off; sleep100 0.725× off.
PASS 2 (load 1.46): off 0.1684 | spin50 0.1800 | spin100 0.2181 | sleep100 0.2247 | rg 0.1293
  off/rg = 0.768×; spin50 0.936× off; spin100 0.772× off; sleep100 0.749× off.

MONOTONIC both passes: off < spin50 < spin100 < sleep100. Sign-stable. The
frequency-neutral SLEEP control preserves (in fact exceeds) the spin delta ⇒ NOT a
turbo/frequency artifact (rule 2). off/rg sign-stable at ~0.77× (consistent with the
~0.79× banked native_fold, allowing for load).

## CONTEXT
fulcrum_total (native T8, byte-exact, routing-guard REFUSED on natural window_seeded=2,
read descriptively): worker.block_body 572ms SELF dominant compute; consumer 57% WAIT
(starved on workers) ⇒ engine-rate-bound. block_body now folds in the contig clean
decode (Stage 2 removed the separate drain/ring-write spans).

## DISPROOF QUESTIONS
1. Is the monotonic spin50<spin100 + sleep-survives-spin sufficient to conclude the
   contig clean loop is ON the critical path (not slack-masked)? Any confound?
2. Does the per-decode-event inject site (once per outer `while emitted < local_cap`
   iteration, BEFORE refill) fairly proxy "symbol-decode compute time", or does it
   over/under-weight (e.g. a back-ref of length 258 = one inject vs 258 literals = one
   inject each across iterations)? Does this bias the slope but NOT the on-path verdict?
3. Is ocl_cf 0.925× still the correct speed-up CEILING for THIS loop (engine-removed),
   and the slope (slow-down) correctly NOT used as the payoff bound (rule 3)?
4. Strongest reason this could be a phantom: anything that makes the wall move under
   inject WITHOUT the symbol-rate being the real lever?
