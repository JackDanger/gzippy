# IMPLEMENT T1 SINGLE-SHOT ROUTING (gzippy-isal) — owner charter 2026-06-09 (supervisor-DECIDED)

## DECISION (supervisor, per user "make all decisions yourself")
Route single-threaded gzippy-isal single-member decode to SINGLE-SHOT ISA-L instead of the
16-chunk ParallelSM pipeline. This is the gzippy-isal charter LITERALLY ("hand off to ISA-L via
FFI at the right spot" — at T1 the right spot is one ISA-L call). PROVEN win: DIS-15 single-shot
T1 = 1.197x rg (beats it) vs ParallelSM 0.90x; the 247ms chunking overhead buys zero parallelism
at 1 thread. Byte-exact (single-shot decompress_gzip_stream verifies CRC32+ISIZE; DIS-15 sha-OK).

## THE JOB
1. SOURCE-VERIFY the routing: classify_gzip (mod.rs:170-188) returns ParallelSM for single-member
   with NO thread floor. The single-shot path isal_decompress::decompress_gzip_stream still exists
   (the oracle turns used it via try_isal_singleshot_oracle). Add a PRODUCTION route: on the
   gzippy-isal build (cfg isal_clean_tail), single-member, num_threads==1 (and NOT
   gzippy-parallel/multi-member — those route as today) -> single-shot ISA-L. Keep it a CLEAN
   thread-count route (preserve ONE-PRODUCTION-PATH clarity: document T1-isal->single-shot,
   T>1-isal->ParallelSM, native->ParallelSM-always). gzippy-NATIVE UNCHANGED (no ISA-L; stays
   ParallelSM at every T).
2. CORRECTNESS: byte-exact — dual-sha 028bd002…cb410f BOTH features at T1 (single-shot) AND T4/T8
   (still ParallelSM, must be unchanged); full lib suite; the multi-member + gzippy-parallel +
   routing traps must still pass (T1 single-member single-shot must NOT swallow multi-member —
   verify multi-member at T1 still routes correctly). The deletion-trap / routing regression tests
   must stay green.
3. MEASURE: T1 wall before/after (expect ~0.90 -> ~1.2x rg, beats rg), and CONFIRM T4/T8 unchanged
   (no regression — they stay ParallelSM). Frozen quiet box, interleaved N>=11, sha-verified.

## GATES + DISCIPLINES
git WORKTREE; box FREE — bench-lock freeze, release clean. Numbers ONLY from the bench-locked quiet
guest; matched same-sink; interleaved N>=11; sha-verified; assert routing (GZIPPY_DEBUG path: T1
isal -> single-shot, T4 -> ParallelSM). The Agent/advisor tool is UNAVAILABLE to you — run rigorous
self-disproof + hand ME (supervisor) the byte-exact proof + T1/T4/T8 wall + routing readback for my
Opus gate; do NOT claim "advisor-vetted." This is a PRODUCTION routing change — extra correctness
care (multi-member-at-T1, the routing/deletion-trap tests). Source-verify first-hand. Serialize
builds via cargo-lock.sh, df -h around builds. No multi-line python via Bash. Wrap hang-prone cmds
in timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels — pgrep
clean on local+guest+neurotic (prior turns leaked find/ + timeout-sleep orphans — sweep). Update
plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and report: the routing
change, the byte-exact proof (both features, T1+T4+T8), the T1 wall win + T4/T8 no-regression, the
multi-member-at-T1 correctness, and raw numbers for my gate.
