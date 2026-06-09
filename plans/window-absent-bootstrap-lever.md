# WINDOW-ABSENT BOOTSTRAP LEVER — owner charter (supervisor, 2026-06-08)

## WHY (the now-complete T4 gap map, advisor-vetted ×2)
Native T4 deficit 0.740->1.0 = 0.260x decomposes into: ENGINE rate 0.159x (asm lever,
user-gated, SEPARATE) + NON-ENGINE residual 0.101x/56ms. The residual decomposition
(plans/residual-decomposition-advisor-verdict.md) proved: FFI handoff is bounded OUT (cell
A routes EVERY chunk through ISA-L FFI and still beats rg 1.555x); the per-bucket
placement/marker split is NON-REALIZABLE (windows are boundary-relative — cell C seed probe
hits=0/16); the 56ms is dominated by the WINDOW-ABSENT MARKER-BOOTSTRAP as an INSEPARABLE
UNIT. Arithmetic: closing ONLY this -> ~0.82x T4; closing it AND the engine -> ~491ms ≈ rg
= 1.0x. So this lever is REQUIRED (with the engine lever) for the >=0.99x-at-every-T bar.
Ceiling = 56ms (NOT the 259ms gross free-placement cost — rapidgzip ALSO pays a window-absent
bootstrap: CLAUDE.md 31.25%-replaced-markers / 0.113s apply-window).

## STEP 0 (advisor-OWED budget gate — do FIRST, may stop here)
Measure the WINDOW-PRESENT chunk fraction in baseline production native T4 (cell D) vs
rapidgzip. Convert "window-absent dominates" into a concrete budget: "rg runs P_rg% of
chunks window-present vs gzippy P_gz%; closing the gap is N chunks x M ms toward the 56ms
ceiling." If gzippy's window-present fraction is ALREADY ≈ rg's, the lever is small -> report
and STOP (the residual is then intrinsic/shared, not placement-closable). Bankable numbers
only (Steward criteria); pre-register this falsifier.

## STEP 1 (faithful map — VENDOR-PORT region, cartographer-on-demand FIRST)
The block-finder / window-map is FAITHFUL-PORT territory (vendor-port rule, NOT the open
inner-loop). Before ANY change, write the two-column rapidgzip<->gzippy map (bias guardrail)
for: rapidgzip's window-map + block-finder placement — HOW it raises the window-present
fraction / lowers window-absent bootstraps (setInitialWindow, the window map, block-finder
boundary alignment, prefetch horizon) vs gzippy's. CRITICAL (red-team DIS-6): prior gzippy
placement FIXES were REFUTED — offset-supply was a no-op, consumer-confirmation prefetch was
dead by measurement. So source-verify what rapidgzip does DIFFERENTLY that gzippy lacks; do
NOT replay the refuted attempts. Cite vendor file:line. Map row each: faithful / divergent /
unmapped.

## STEP 2 (port — only if STEP 0 budget + STEP 1 map justify, gated)
Port the faithful placement mechanism to raise gzippy's window-present chunk fraction.
Byte-exact (OFF==identity where gated, dual-sha 028bd002...cb410f BOTH features, full lib
suite). REMOVE-AND-MEASURE the wall vs the 56ms ceiling on the bench-locked quiet box
(matched same-sink, interleaved N>=9, sha-verified, path=ParallelSM). Do NOT touch the asm
rewrite (user's gated call). Do NOT start STEP 2 if STEP 0 says the budget is small.

## GATES + DISCIPLINES
Work in a git WORKTREE. Route STEP 0 numbers through the Measurement-Integrity Steward
(bankability) and the inference through a synchronous Opus disproof advisor (verdict to
plans/window-absent-bootstrap-advisor-verdict.md). Invoke the Cartographer (on-demand) for
STEP 1. RUN SUBAGENTS/ADVISORS SYNCHRONOUSLY (no background-and-yield). Run measurements
YOURSELF holding the ssh. Source-verify first-hand. Serialize builds via cargo-lock.sh, df -h
around builds. No multi-line python via Bash. Wrap hang-prone cmds in timeout. Diagnose the
FIRST error before retrying. NO orphan processes / sleep sentinels — pgrep clean before
finishing (a prior agent left two 100%-CPU adv_trailing_garbage_terminates orphans for 3.5h;
do not repeat). Update plans/orchestrator-status.md. STOP at each checkpoint for the
supervisor gate.

## PARKED (do NOT lose, NOT this turn): the JOB-2 SYNC_FLUSH gap (user-directed "close that
gap") — relax gzip_chunk until_exact to coalesce to nearest clean EOB like rapidgzip
readStream; correctness-sensitive seed-path change, its own gated turn with the committed
adversarial fixture (branch isal-resync-stored-fixed) as the coverage gate.
