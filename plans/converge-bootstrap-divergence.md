# CONVERGE THE WINDOW-ABSENT BOOTSTRAP DIVERGENCE — leader charter (supervisor, 2026-06-08)

## GOVERNING FRAME (user-set 2026-06-08 — read FIRST)
Stop thinking in "levers" / "individual wins." Matching rapidgzip's ACTUAL RUNTIME BEHAVIOR
= 100% parity BY CONSTRUCTION. Any wall delta vs rg (UNDER or OVER) means gzippy's CPU
instructions differ materially from rg's somewhere. Your job: USE THE NUMBERS to LOCATE
where gzippy's instructions/behavior diverge from rg, then CONVERGE (port rg's behavior;
delete the divergent path / create the matching one). NOT "pick a lever." See
plans/standing-specialists.md + the governing memory feedback_wall_delta_is_instruction_delta
+ feedback_bias_guardrails.

## WHERE THE NUMBERS POINT (the located symptom — not a lever, a divergence site)
T4 native 0.740x rg. Decomposed + advisor-vetted: clean-rate divergence ~0.159x + a 56ms
window-absent-bootstrap residual. STEP-0 ruled OUT the window-present-fraction hypothesis:
gzippy 33.62% window-absent marker bytes vs rg 34.50% — gzippy marginally AHEAD, the
flip-to-clean threshold is a byte-for-byte vendor port. So the 56ms is NOT "more
window-absent chunks." It is gzippy's window-absent bootstrap executing MATERIALLY DIFFERENT
INSTRUCTIONS than rg's bootstrap on the SAME ~16/18 chunks. FFI handoff bounded OUT.

## THE JOB — find the instruction divergence, converge to rg (find-and-fix, in this turn)
1. LOCATE: profile BOTH gzippy-native and rapidgzip on the window-absent bootstrap chunks at
   the instruction level (perf stat/record, instruction counts, cache/branch/syscall/lock
   behavior, the actual code path each takes). Find the MATERIAL divergence: what does
   gzippy's marker bootstrap DO that rg's does not, or differently? The numbers must
   DISCRIMINATE (do NOT assume) among: (a) marker-decode per-symbol instruction count
   (gzippy pure-Rust marker decode vs rg's marker decode — both non-ISA-L on window-absent);
   (b) prefetch-horizon / head-of-line scheduling (chunks decode_NOT_STARTED when they could
   run — OPEN-2, DISTINCT from the DIS-6-refuted offset-supply; do NOT replay DIS-6);
   (c) memory/alloc/page behavior. Cite the perf evidence.
2. MAP (cartographer-on-demand): two-column rg-behavior (vendor file:line) vs gzippy for the
   divergent region. The convergence TARGET is rg's actual instructions/structure.
3. CONVERGE: port rg's behavior so gzippy emits matching instructions there. Delete the
   divergent path / create the matching one. Byte-exact (OFF==identity where gated, dual-sha
   028bd002...cb410f BOTH features, full lib suite). The VERDICT is the interleaved WALL
   moving toward rg on the bench-locked quiet box (NOT a producer-side count). A change that
   makes gzippy UNLIKE rg is forbidden even if it helps the wall.
   - USER-GATED BOUNDARY: if the located divergence is the INNER-LOOP engine instruction rate
     (gzippy's pure-Rust symbol decode emitting more instructions than rg's ISA-L-backed
     decode), converging there = the inner-loop asm rewrite, which is the USER'S GATED CALL —
     STOP and report "the divergence is the engine-instruction gap," with the key strategic
     read: is the 56ms residual ALSO engine-instruction (=> a full-engine asm port could be
     SUFFICIENT for T4 parity, not merely necessary)? Converge anything NON-inner-loop
     (scheduling / structure / alloc / page behavior) faithfully THIS turn.

## NO WORK-DISPLACEMENT (user-set 2026-06-08, BINDING)
"I don't mind winning where we can win, but don't make one part super fast if it actually
pushes the work somewhere else where it doesn't belong." Winning is fine when the
instructions are GENUINELY fewer (a true convergence to rg's behavior). It is NOT a win — it
is forbidden — to make region X look fast by DISPLACING its work into region Y where it does
not belong: that is (1) NON-FAITHFUL (rg doesn't do it) AND (2) a PHANTOM (the work just
moved, the whole-system wall doesn't improve — or improves by an artifact). The campaign has
hit this exact trap: the OverlapWriter "won" the consumer by pushing output onto a background
thread rg LACKS (refuted, non-faithful); copy-free tricks "won" decode by pushing cost into
writev / page-faults. ENFORCEMENT: every converged change must be verified to NOT grow the
regions it could displace into — report the whole-system wall AND the would-be displaced-to
stages (output/writev, page-faults, alloc, the next chunk's start) before/after. A
region-local speedup with a flat or worse whole-system wall = displacement = REVERT.

## GATES + DISCIPLINES
Work in a git WORKTREE. Numbers ONLY from the bench-locked quiet guest (procs_running gate);
matched same-sink, interleaved N>=9, sha-verified, path=ParallelSM. Route numbers through the
Measurement-Integrity Steward (bankability) and the inference through a synchronous Opus
disproof advisor (verdict to plans/converge-bootstrap-advisor-verdict.md). Cartographer
on-demand for the map. RUN SUBAGENTS/ADVISORS SYNCHRONOUSLY (no background-and-yield). Run
measurements YOURSELF holding the ssh. Source-verify first-hand. Serialize builds via
cargo-lock.sh, df -h around builds. No multi-line python via Bash. Wrap hang-prone cmds in
timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels —
pgrep clean (a prior agent left two 100%-CPU test orphans for 3.5h; verify yours are reaped).
Update plans/orchestrator-status.md. STOP at the checkpoint (converged fix + wall delta, OR
the user-gated inner-loop finding) for the supervisor gate. Do NOT start the inner-loop asm.

## PARKED (not this turn): JOB-2 SYNC_FLUSH gap (branch isal-resync-stored-fixed) — its own
gated correctness-sensitive turn.
