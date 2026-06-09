# JOB-2 SYNC_FLUSH RESYNC PORT (box-INDEPENDENT, Plex window) — owner charter (supervisor, 2026-06-08)

## CONSTRAINT (the user needs Plex/neurotic for a few hours — HARD)
Do NOT freeze neurotic. Do NOT run ANY bench-locked / wall measurement. This is a CORRECTNESS
+ COVERAGE turn ONLY, tested LOCALLY: the gzippy-isal x86_64 path builds+runs locally via
Rosetta 2 (see memory reference_local_x86_tests_rosetta) — use that + the lib suite + the
adversarial fixture. The WALL parity-regression check is DEFERRED to the next box window;
do not attempt it.

## WHY (user-directed "close that gap")
gzippy-isal degrades to byte-exact native (ZERO ISA-L coverage) on SYNC_FLUSH / flush-dense
(and stored/fixed-block-heavy) streams: gzip_chunk's `until_exact` accept requires an EXACT
stop-hint match (the `end_bit<=stop_hint` genuine-BFINAL-only accept added in 19add96c), and
on dense small blocks the stop_hint rarely coincides with a boundary bit, so ISA-L declines.
rapidgzip's `readStream` instead coalesces to the NEAREST clean EOB. Faithful port = relax the
accept to coalesce to the nearest clean END_OF_BLOCK at/before the boundary. The cited
isal.hpp:392-405 was the WRONG mechanism (readBytes = footer reader) — find readStream
first-hand. The adversarial fixture (20,480 SYNC_FLUSH blocks) + the gap diagnosis are already
on branch isal-resync-stored-fixed (commits 8c87cc24 + 7695463d).

## THE JOB
1. SOURCE-VERIFY first-hand: rapidgzip's readStream nearest-clean-EOB coalesce (vendor
   file:line) vs gzippy's `finish_decode_chunk_impl` until_exact accept (the end_bit<=stop_hint
   BFINAL-only path from 19add96c). Write the two-column map.
2. PORT the coalesce faithfully so gzippy-isal keeps ISA-L coverage across SYNC_FLUSH /
   stored / fixed blocks. CRITICAL HAZARD: do NOT re-introduce the 19add96c OVER-DECODE
   mis-seed (the reason the accept is BFINAL-only is that ISA-L's EOB-stop doesn't fire on
   stored/fixed → nbounds=0 → a 2MB over-decode past stop_hint mis-seeds the next chunk). The
   coalesce must land on a CLEAN EOB at/before the boundary and DECLINE (byte-exact pure-Rust)
   rather than over-decode when no clean EOB exists. Mirror rapidgzip's readStream exactly.
3. PROVE byte-exact + coverage LOCALLY: dual-sha 028bd002...cb410f BOTH features (gzippy-native
   UNCHANGED, byte-identical, isal_chunks unaffected on it); full lib suite green; the
   btype01-heavy + multi-subchunk routing traps + the isal_tail_parity differential gate green;
   and PROVE ISA-L coverage > 0 on the adversarial SYNC_FLUSH fixture (the coverage gate — show
   isal_chunks rises from 0). Ship the silesia differential in the same commit.

## HAND-OFF (owners can't spawn advisor/Steward in this env)
You likely CANNOT spawn a synchronous Opus advisor (Agent tool absent in owner env). Run
rigorous self-disproof, then hand ME (supervisor) the byte-exact proof (sha digests both
features), the coverage delta on the fixture, the two-column map, and the diff — I run the
Opus advisor + correctness gate at the supervisor level before this is considered done. Do NOT
claim "advisor-vetted."

## GATES + DISCIPLINES
Work in the git WORKTREE / branch isal-resync-stored-fixed (extend it). Source-verify first-
hand (the cite was wrong once). No multi-line python via Bash (write a .py). Wrap hang-prone
cmds in timeout. Diagnose the FIRST error before retrying. NO neurotic freeze, NO wall runs.
NO orphan processes / sleep sentinels — pgrep clean before finishing. Update
plans/orchestrator-status.md. STOP at the checkpoint and report to me: the map, the port, the
byte-exact proof, the coverage delta (isal_chunks 0 -> N on the fixture), and the diff for my
gate.
