# gzippy-ISAL PER-BYTE OVERHEAD CONVERGENCE — leader charter (2026-06-09)

## GOVERNING PRINCIPLE (user): a performance difference IS a compiled-code difference; converge it.
The difference is IDENTIFIED (capstone gate, plans/structural-residual-capstone-verdict.md): the
gzippy-ISAL low-T gap is a ~13% PER-BYTE rate gap on the IDENTICAL ISA-L kernel (gz 226 vs rg 261
MB/s @T1, engine-matched, output+marker removed). Two named compiled-code causes:
1. NON-LTO'd FFI GLUE: rg builds ISA-L IN-TREE with LTO/inlining (dispatcher inlinable into the
   caller); gzippy links a separate autotools `.a` via opaque Rust->C FFI, NO cross-language LTO
   (ISA-L audit Level-3, plans/isal-equivalence-verdict.md). Per-byte invocation overhead.
2. D1 OUTPUT OVER-RESERVE 8x: gzippy reserves ~8x compressed_span per chunk vs vendor's
   incremental FasterVector growth -> cache/page-fault pressure (the long-standing gz-faults-2x-rg
   gap). gzip_chunk.rs:263-274 -> segmented_buffer.rs:243.
This is gzippy-ISAL ONLY (it already uses FFI; rg is the existence proof). gzippy-NATIVE is
SEPARATE — walled by the 0.667x VAR_VIII engine ceiling; do NOT touch native here.

## STEP 0 (premise + binary)
(a) VERIFY rg 0.16.0 actually uses ISA-L at T1 (the capstone's one unverified premise) — check rg
build/symbols or its T1 rate vs the ISA-L kernel rate. (b) Proof-of-binary: build gzippy-isal at
HEAD, assert isal_chunks>=14@T4/16@T1, env-unset, path=ParallelSM (isal_chunks increments only in
real-ISA-L cfg gzip_chunk.rs:386).

## THE JOB (do the EASY win first; the wall is the verdict)
1. D1 OVER-RESERVE FIX (easier): change the ISA-L output reserve from ~8x-compressed-span to
   vendor's INCREMENTAL growth (match FasterVector). Byte-exact (OFF==identity, dual-sha both
   features, full suite). REMOVE-AND-MEASURE: the interleaved WHOLE-SYSTEM wall (isal T1+T4) AND
   the page-fault count (perf stat) before/after. NO WORK-DISPLACEMENT (don't push the alloc cost
   elsewhere). NOTE: branch isal-resync-stored-fixed (JOB-2, unmerged) already touched a DIFFERENT
   aspect of this reserve (an under-reserve CORRECTNESS fix); base on HEAD, keep your reserve-SIZE
   change independent + note it for merge reconciliation.
2. PER-BYTE GLUE (harder — investigate, attempt the tractable): is the per-byte excess the
   non-inlined dispatcher (LTO/inline), or the avail_out feed (rg refills in BitReader chunks; gz
   hands the whole slice — gate said avail_out-amortization is available), or both? Source+disasm-
   verify (reuse scripts/analysis/disasm_*). Attempt the TRACTABLE convergence: cross-language LTO
   on the isal-rs link (-C linker-plugin-lto + matching clang -flto on ISA-L), OR build ISA-L
   in-tree LTO like rg, OR an avail_out amortization change — whichever is tractable WITHOUT a
   multi-hour build-system yak-shave. If LTO is a yak-shave, STOP and report feasibility + scope;
   do NOT burn hours on a build-system rabbit hole. Measure each attempt's wall.

## OUTPUT
The wall delta for the D1 fix (isal T1/T4, + page-fault delta) and for any glue convergence
attempted; how much of the ~13% per-byte gap each closes; the remaining gap. VERDICT: does
converging the identified differences move isal T1/T4 toward 0.99 (lever real) or not (re-examine)?

## GATES + DISCIPLINES
git WORKTREE; box is FREE — bench-lock freeze, release clean. Numbers ONLY from the bench-locked
quiet guest (procs_running gate); matched same-sink; interleaved N>=11; sha-verified;
path=ParallelSM + isal_chunks>=14 asserted. The Agent/advisor tool is UNAVAILABLE to you — run
rigorous self-disproof (pre-register falsifiers; freq-neutral controls) and hand ME (supervisor)
RAW numbers + provenance + isal_chunks readback for my Opus gate; do NOT claim "advisor-vetted."
Run measurements YOURSELF holding the ssh. SOURCE-VERIFY first-hand. Serialize builds via
cargo-lock.sh, df -h around builds. No multi-line python via Bash (write a .py). Wrap hang-prone
cmds in timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels —
pgrep clean on local + guest + neurotic before finishing (a prior agent left two 100%-CPU orphans
for 3.5h). Update plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and
report the wall deltas + the lever-real-or-not verdict + raw numbers for my gate.
