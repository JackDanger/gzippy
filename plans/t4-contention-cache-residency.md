# T4/T8 CONTENTION + CACHE-RESIDENCY INVESTIGATION (the binding BAR-1 blocker) — owner charter 2026-06-09

## WHY (the real BAR-1 blocker, and it's on the user's NORTH STAR)
isal T1 is recoverable (single-shot beats rg); T8 ties; the binding gap is T4 0.911x = DEEPER
parallel-scheduling (consumer-lean refuted NULL, DIS-16). NEW SIGNAL: gzippy T4 run-to-run
variance = 17-36% vs rapidgzip 8-10%. High parallel variance = CONTENTION/nondeterminism —
which is exactly the user's core goal: "the same SHARED, HOT-IN-CACHE, small amount of memory
for all parts of the decoding path so 8 or 16 (or more) threads use such a small amount of
memory it's all in cache a lot of the time." This is NOT the DIS-6-refuted offset/prefetch
placement fixes — it is the memory/cache/contention story, unexplored.

## STEP 0 — proof-of-binary + clear the owed disasm
Build gzippy-isal at HEAD, assert isal_chunks>=14, env-unset, path=ParallelSM. THEN run the owed
disasm confirmation (scripts/analysis/disasm_extract.sh + disasm_diff.py from the ISA-L audit) on
the gzippy-isal binary vs rapidgzip 0.16.0 — confirm gzippy linked the AVX2 nasm igzip kernel
(symbol *_stateless_04, AVX2 instr>0). Closes the ISA-L equivalence Level-2 empirical.

## STEP 1 — LOCATE the contention (perf, gzippy-isal vs rg, T4 AND T8, frozen box)
Characterize the variance source — which of these is it? (do NOT assume; measure):
- MEMORY/CACHE CONTENTION: perf stat -e LLC-load-misses,LLC-store-misses,cache-misses,
  L1-dcache-load-misses,dTLB-load-misses at T4+T8, gzippy vs rg. Is gzippy's MPKI / LLC-miss
  rate higher (per-thread working set spilling cache / cross-thread eviction)?
- FALSE SHARING: perf c2c (if available on the guest) on the shared structures (window-map,
  prefetch cache, consumer queue, the chunk ring) — are threads ping-ponging cache lines?
- LOCK CONTENTION: context-switches, the shared Mutex/condvar hot spots (the consumer's
  process_ready_prefetches lock, the window-publish lock) — perf record on futex/lock symbols.
- ALLOCATION CONTENTION: rpmalloc cross-thread frees (the recycle_deferral / cross-thread chunk
  free the campaign flagged) — per-thread vs cross-thread alloc/free.
Also measure the PER-THREAD working set + the SHARED read/write footprint at T4/T8 (the campaign
banked 158.8 KiB/thread; re-confirm + compare to rg's footprint). The user's goal is small-shared-
hot-in-cache — quantify gzippy vs rg on that axis.

## STEP 2 — CONVERGE the located contention (faithful, byte-transparent), measure WALL + VARIANCE
If a specific contention source is located (false-sharing line -> pad/align to cache line;
cross-thread free -> per-thread free; oversized shared structure -> shrink/shard; lock hot-spot ->
match rg's lock-free/leaner structure), CONVERGE it (faithful to rg where rg is the existence
proof; cite vendor file:line). Byte-exact (OFF==identity, dual-sha, full suite). Measure: the T4+T8
WALL delta AND whether the run-to-run VARIANCE tightens toward rg's 8-10% (the variance IS the
signal — closing it is the deliverable). NO WORK-DISPLACEMENT.

PRE-REGISTER falsifier per hypothesis (e.g. "if false sharing: perf c2c shows HITM on a named
line; padding it cuts HITM + tightens variance"). A located-but-unfixable contention => report it
as the bound. gzippy-NATIVE shares this pipeline (+ its 0.667x engine floor) — the fix helps both.

## GATES + DISCIPLINES
git WORKTREE; box is FREE — bench-lock freeze (re-acquire until runnable_avg<=2.0), release clean.
Numbers ONLY from the bench-locked quiet guest; matched same-sink; interleaved N>=15 (the T4
variance is large — use enough samples + report the spread, not just min); sha-verified;
path=ParallelSM + isal_chunks>=14 asserted. The Agent/advisor tool is UNAVAILABLE to you — run
rigorous self-disproof (pre-registered falsifiers) and hand ME (supervisor) RAW numbers +
provenance + isal_chunks readback for my Opus gate; do NOT claim "advisor-vetted." Run
measurements YOURSELF holding the ssh. SOURCE-VERIFY first-hand. Serialize builds via
cargo-lock.sh, df -h around builds. No multi-line python via Bash (write a .py). Wrap hang-prone
cmds in timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels —
pgrep clean on local + guest + neurotic before finishing (a prior agent left two 100%-CPU orphans
for 3.5h). Update plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and
report: the disasm confirmation, the LOCATED contention source (with perf evidence), any converged
fix's wall+variance delta, and raw numbers for my gate.
