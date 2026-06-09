# RESIDUAL ATTRIBUTION + OPEN-2 DISCRIMINATOR — owner charter (supervisor, 2026-06-08)

## GOVERNING FRAMES (user-set 2026-06-08, BINDING)
- Match rapidgzip's ACTUAL RUNTIME BEHAVIOR = 100% by construction; any delta = CPU
  instructions differ. USE THE NUMBERS to LOCATE the divergence, CONVERGE to rg. Not levers.
  (memory feedback_wall_delta_is_instruction_delta)
- NO WORK-DISPLACEMENT: a region-local speedup that pushes work into another stage
  (output/writev, page-faults, alloc, next-chunk-start) is forbidden — non-faithful AND a
  phantom. Verify the whole-system wall AND displaced-to stages before/after; flat/worse
  whole-system = REVERT. (user-set 2026-06-08)

## CORRECTED PICTURE (advisor-vetted plans/converge-bootstrap-advisor-verdict.md — the prior
## leader trusted a STALE WRONG build.rs comment; advisor refuted it source-first-hand)
- gzippy-ISAL: production clean tail = REAL ISA-L FFI at HEAD (build.rs:110 isal_clean_tail ->
  gzip_chunk.rs:161 isal_engine_oracle_enabled -> :205 FFI body -> :669 production gate -> :275
  decompress_deflate_from_bit_into). So isal already MATCHES rg WITH_ISAL on the clean tail;
  its low-T deficit (T4 0.885x) is the RESIDUAL ONLY. The pure-Rust StreamingInflateWrapper is
  a rare FALLBACK (fallbacks=0 on the parity corpus).
- gzippy-NATIVE: clean tail = pure-Rust decode_clean_into_contig (gzip_chunk.rs:1453/1224/1303,
  marker_inflate.rs:2071) = the ~2.3x divergence vs rg's ISA-L. Faithful convergence there =
  full-kernel asm = USER-GATED, NOT this turn.
- The <=0.101x/56ms T4 residual is UNATTRIBUTED — an upper-bound bucket of {marker-prefix
  pure-Rust engine, FFI-handoff, placement}. OPEN-2 (consumer-imminent eviction) is a
  HYPOTHESIS; the dispatched-then-EVICTED vs never-dispatched discriminator is UNRUN.

## THE JOB
1. FIX the lying comment build.rs:98-110 (byte-transparent: it wrongly says both topologies are
   pure-Rust / the ISA-L claim is stale — FALSE at HEAD). Record the reconciliation in
   plans/disproof-ledger.md: isal clean tail IS ISA-L FFI at HEAD; GOAL #2 (19add96c)
   source-correct; the "pure-Rust isal tail" was a stale-comment error. Cite file:line.
2. FREEZE THE BOX (host bench-lock; prior turn found it thawed, procs_running=3). All numbers
   bench-locked + Steward-bankable, else labeled EXPLORATORY. No number from a thawed box.
3. ATTRIBUTE the 56ms residual into {marker-prefix engine / FFI-handoff / placement} via
   REMOVAL oracles (rule 3, pre-register each falsifier). Which sub-term DOMINATES? Needed for
   BOTH builds (isal's ENTIRE low-T deficit is this residual; native's is this + clean-tail
   engine).
4. OPEN-2 DISCRIMINATOR (only if placement is a non-trivial share): is the covering chunk
   NEVER-DISPATCHED or DISPATCHED-THEN-EVICTED? Build the unrun instrument (in-flight vs
   resident vs evicted at each stall; parked-vs-unspawned idle; N>>3). Determines the faithful
   fix (retention-protect imminent chunk vs deepen horizon — different vendor-port questions).
   Do NOT replay the DIS-6-refuted offset-supply.
5. CONVERGE only a FAITHFUL non-asm fix (matches a rg vendor mechanism file:line, no
   work-displacement): byte-exact, remove-and-measure WHOLE-SYSTEM wall + displaced-to-stage
   check. If the residual is dominated by marker-prefix ENGINE (also pure-Rust-vs-ISA-L), report
   that — it folds into the asm question, not separately closable.

## THE PIVOTAL OUTPUT
The residual attribution + the OPEN-2 discriminator result + CRITICALLY: can gzippy-ISAL reach
0.99x at T4 by closing the residual ALONE (no asm), since its clean tail is already ISA-L? That
sharpens the native asm decision (the user's gated call).

## GATES + DISCIPLINES
git WORKTREE; numbers ONLY from the bench-locked quiet guest; matched same-sink, interleaved
N>=9, sha-verified, path=ParallelSM. Route numbers through the Measurement-Integrity Steward
(bankability) + the inference through a synchronous Opus disproof advisor (verdict to
plans/residual-attribution-advisor-verdict.md). Cartographer on-demand for any vendor map.
RUN SUBAGENTS/ADVISORS SYNCHRONOUSLY (no background-and-yield). Run measurements YOURSELF
holding the ssh. Source-verify first-hand. Serialize builds via cargo-lock.sh, df -h around
builds. No multi-line python via Bash. Wrap hang-prone cmds in timeout. Diagnose the FIRST
error before retrying. NO orphan processes / sleep sentinels — pgrep clean before finishing
(verify on local + guest + neurotic; a prior agent left two 100%-CPU orphans for 3.5h).
Update plans/orchestrator-status.md. Do NOT start the inner-loop asm (user-gated). STOP at the
checkpoint for the supervisor gate.

## PARKED (not this turn): JOB-2 SYNC_FLUSH gap (branch isal-resync-stored-fixed).
