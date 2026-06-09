# OPEN-1: DISENTANGLE THE FAITHFUL-PLACEMENT SLICE FROM THE GATED ASM — owner charter (2026-06-08)

## WHY (advisor-gated, plans/lowt-residual-gate-verdict.md)
The seed-all T4 gain (231ms, 0.902x->1.55x) is a CONFOUNDED upper bound. It bundles THREE
things and we cannot authorize a lever until they're separated:
1. ~3 chunks getting a FREE pure-Rust->ISA-L engine upgrade (proven: isal_chunks 14->17 under
   seed-all; a seeded hit forces the clean arm `decode_chunk_with_until_exact`
   chunk_fetcher.rs:2582 = real ISA-L FFI on the isal build) = the USER-GATED clean-tail asm
   (LEV-4) in disguise.
2. FREE precomputed placement from an UNCOUNTED p=1 pre-pass (work rg pays at RUNTIME).
3. The genuinely-faithful runtime placement gain — the ONLY owner-turnable, non-gated slice.
The faithful slice (3) is currently UNSIZED. "T4 reaches 0.99x by closing the bootstrap" is
UNPROVEN. This turn SIZES (3) cleanly.

## THE OBJECTIVE — an oracle that isolates faithful placement
Build a measurement oracle (byte-transparent, OFF==identity) that:
- grants rg-grade chunk placement/window-availability AT RUNTIME COST — the window-map build /
  pre-pass MUST be COUNTED in the measured wall (NOT a free precomputed hand-off; contrast the
  seed-all capture/replay which hides the pre-pass), AND
- keeps the window-absent prefix on the PRODUCTION pure-Rust engine — NO free ISA-L upgrade.
  HARD INVARIANT + ASSERT: isal_chunks MUST stay == 14 (NOT rise to 17). If isal_chunks rises,
  the oracle is leaking the gated engine swap and the number is VOID.
This isolates slice (3): the faithful runtime-placement gain, apart from the gated asm (1) and
the uncounted pre-pass (2).

## MEASURE + FALSIFIER (pre-register)
T4 primary (+ T1 sanity), frozen box, interleaved N>=11, sha-verified, isal_chunks==14 asserted:
- If the faithful-placement slice (production wall - this-oracle wall) is < inter-run spread =>
  placement is NOT a faithful low-T lever at T4; the 231ms was the gated asm + uncounted
  pre-pass. CONCLUSION: the low-T lever IS the (gated) asm; report that.
- If it's a real chunk (> spread) => there IS a faithful runtime-placement lever; identify the
  rg vendor mechanism (window-map runtime build, BlockFetcher/window propagation) that gzippy
  diverges from, and report its size + the convergence target (NEXT turn ports it).

## STEP 0 (MANDATORY proof-of-binary — the mislabel failure mode)
Build gzippy-isal at HEAD; assert isal_chunks>=14 @T4/T8 on env-unset silesia BEFORE any wall
number (isal_chunks increments only in real-ISA-L cfg gzip_chunk.rs:386; native stub :390-400
never does). GZIPPY_ISAL_ENGINE_ORACLE unset, path=ParallelSM. If 0 you built native — rebuild.

## GATES + DISCIPLINES
git WORKTREE (do NOT collide with the concurrent box-free JOB-2 owner on branch
isal-resync-stored-fixed). The box is FREE; bench-lock freeze for measurements, release clean.
Numbers ONLY from the bench-locked quiet guest (procs_running gate); matched same-sink;
interleaved N>=11 (N>=15 any T8 cell). The Agent/advisor/Steward tool is UNAVAILABLE to you —
run rigorous self-disproof and hand ME (supervisor) RAW numbers + provenance + the isal_chunks
invariant readback for my gate; do NOT claim "advisor-vetted." Run measurements YOURSELF
holding the ssh. SOURCE-VERIFY first-hand. Serialize builds via cargo-lock.sh, df -h around
builds. No multi-line python via Bash (write a .py). Wrap hang-prone cmds in timeout. Diagnose
the FIRST error before retrying. NO orphan processes / sleep sentinels — pgrep clean on local +
guest + neurotic before finishing. Do NOT start the inner-loop asm (user-gated). Update
plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and report:
proof-of-binary, the faithful-placement slice size (with isal_chunks==14 readback proving no
engine leak), the falsifier verdict, and raw numbers + provenance for my gate.
