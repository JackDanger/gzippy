# Clean engine-rate CEILING — matched-comparator measurement (owner turn 2026-06-08)

## What was measured (parity.sh + oracle.sh spine — contamination-proof)

Frozen host (neurotic clock_freeze: no_turbo=1, gov=performance, cpu pinned
min==max 1.4 GHz, noisy-neighbor LXCs 111/105 cgroup-frozen). Guest 10.30.0.199,
taskset 0,2,4,6,8,10,12,14 (T8). Corpus /root/silesia.gz raw=211,968,000,
decompressed sha == pin 028bd002…cb410f. Both tools → identical regular-file sink
on /dev/shm (the MATCHED comparator; NOT /dev/null — /dev/null's write_null skips
copy_from_user and is NOT a matched sink per the prior output-reconciliation
verdict). Interleaved best-of-N, sha-verified EVERY run, window-absent-preserving
(no seeding). A stale orphan test binary (PID 587818, /tmp/gz-ft-src, 970MB RSS)
was killed before measuring — it had inflated gzippy spread to 42%.

| run | wall (min) | ratio vs rg | spread | what it is |
|---|---|---|---|---|
| rapidgzip 0.16.0 | 383 ms | 1.000× | 7% | the target (rg WITH_ISAL) |
| **ocl_cf** (gzippy + real ISA-L clean engine, copy-free) | **404 ms** | **0.945×** | 7% | clean engine-rate CEILING |
| production native (pure-Rust) | 440 ms | 0.870× | 8% | current production |

- ocl_cf is `GZIPPY_ISAL_ENGINE_ORACLE=1` on the gzippy-isal build: the post-flip
  clean tail decodes via REAL ISA-L FFI straight into chunk.data (copy-free,
  writable_tail_reserve), the marker bootstrap + ring + window-publish + consumer
  + CRC all UNCHANGED. byte-exact (sha=OK, full N).
- COVERAGE VERIFIED (GZIPPY_VERBOSE): isal_oracle_chunks=14, isal_oracle_fallbacks=0
  ⇒ all 14 clean-tail chunks ran ISA-L, ZERO fell back to pure-Rust. Routing
  flip_to_clean=12 finished_no_flip=4 window_seeded=2 = natural propagation
  (NOT the seed oracle) ⇒ the window-absent bootstrap is preserved (production
  shape). The number is NOT contaminated by pure-Rust decode (the contamination
  that made the OLD ocl oracle inconclusive — it paid a per-chunk 64MiB alloc+copy;
  this copy-free version does not).

## The claims for disproof

C1. The clean engine-rate gap (pure-Rust vs ISA-L) is **36ms = 440→404ms
    (0.870×→0.945×)** — recovering it (matching ISA-L's clean rate in pure-Rust)
    is the largest single remaining matched-wall lever.

C2. Even with the IDEAL clean engine (real ISA-L), gzippy is **0.945× rg = 21ms
    short of parity (404 vs 383)**. That 21ms residual is OUTSIDE the clean engine
    (window-absent bootstrap structure + scheduling/placement). ⇒ The clean engine
    is NOT the whole gap; the no-FFI 1.0× bar is bounded by BOTH the engine rate
    AND this ~21ms non-engine residual. This corroborates the prior ocl 0.925×
    "≥0.075× outside the engine" finding, now MATCHED-paired and copy-free.

C3. The pure-Rust clean engine is the binder we are authorized to attack (STEP 2):
    table-load-latency mitigation (table _mm_prefetch ahead of the dependent load;
    single-level L1-resident decode-table geometry; static-Huffman specialization;
    FASTLOOP yield-elision). ocl_cf 404ms is the CEILING — no pure-Rust engine
    technique can take production below 404ms (only ISA-L's clean RATE is being
    isolated; the 21ms residual survives). A technique that "measures" below 404ms
    is measuring noise or a non-engine side effect.

C4. The 1.4 GHz frequency pin shifts ratios vs the prior campaign's turbo-on 0.13s
    numbers (the memory-bandwidth floor is frequency-invariant, so at low freq the
    compute-bound engine gap shrinks as a fraction of wall). The ratios here
    (0.870× prod, 0.945× ceiling) are the FROZEN-HOST truth; the prior campaign's
    1.17× matched / 2.3× clean-rate numbers were turbo-on and not directly
    comparable in absolute ratio, but the STRUCTURE (engine gap + non-engine
    residual) is consistent.

## Question to the advisor
Is the 404ms ocl_cf a TRUSTWORTHY clean-engine-rate ceiling for bounding STEP-2
pure-Rust techniques? Specifically: (a) does copy-free + coverage=14/0 + matched
sink make it a valid speed-UP ceiling (Rule 3 — removal, not slope)? (b) is C2's
21ms non-engine residual real or an artifact of the oracle still paying SOME
gzippy-specific structure ISA-L-in-rg doesn't? (c) does the 1.4 GHz pin invalidate
any of this?
