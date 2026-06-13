# ISA-L DORMANCY RECONCILIATION — owner charter (supervisor, 2026-06-08)

## THE CONTRADICTION (resolve which banked picture is real at HEAD, and WHY)
Two measurements of the SAME thing (gzippy-isal production, env-unset, silesia, bench-locked)
disagree by a wide margin — this is the broken-instrument / mislabeled-oracle class
(CLAUDE.md: two instruments here were already silently broken). Resolve it FIRST-HAND.

- BANKED (orchestrator-status GOAL #2, commit 19add96c; + the residual-decomposition ocl_cf
  gate): ISA-L coverage 14/14 fallbacks=0 @ T4/T8; gzippy-isal T4 0.885x, T8 1.030x TIE.
  CAUTION: the ocl_cf gate used GZIPPY_ISAL_ENGINE_ORACLE=1 (FORCED ISA-L) — that is the
  ENGINE ORACLE, NOT natural production routing. Whether GOAL #2's "14/14 production coverage"
  was ALSO oracle-forced or genuinely env-unset production is the crux.
- FRESH, UNBANKED (residual-attribution owner, HEAD d56cb0f5, env-unset, bench-locked N=9):
  isal_chunks=0 fallbacks=2; window_present=2/18; clean_flipped=2% of body; u16 marker
  bootstrap=98% @ 85-87 MB/s; gzippy-isal T4 654ms=0.757x (==native within spread), T8
  406ms=0.919x; the ocl_cf engine-isolation ABORTS coverage-zero (isal_chunks 14->0).
- CORROBORATION: the STEP-0 window-present-budget owner ALSO measured window_present=2/18 at
  HEAD. So 2/18 window-present is independently seen twice.

## RESOLVE (source-verify + measure first-hand, frozen box, verified binary)
1. Build the gzippy-isal PRODUCTION binary at HEAD d56cb0f5 and VERIFY: isal-compression
   feature compiled in; the binary is the isal build (not native); GZIPPY_ISAL_ENGINE_ORACLE
   UNSET. Confirm via GZIPPY_DEBUG/VERBOSE.
2. On silesia, bench-locked (procs_running gate), env-unset PRODUCTION routing: measure the
   REAL isal_chunks coverage, window_present fraction, clean_flipped fraction, fallbacks, and
   T1/T4/T8 wall vs rg. Is coverage 0 or 14? Is isal T4 0.757x (dormant, ==native) or 0.885x
   (active)?
3. DISAMBIGUATE the counters (do NOT assume): what does `isal_chunks` count vs `window_present`
   vs `clean_flipped`? Source-trace each emitter. The "14 coverage" vs "2/18 window-present"
   gap must be explained (does ISA-L fire on inexact-offset clean CONTINUATION after a flip,
   not just window-present-from-start? if clean_flipped=2%, how could coverage be 14?).
4. EXPLAIN the discrepancy with GOAL #2. Candidates: (a) GOAL #2's "14/14" was the engine
   ORACLE (GZIPPY_ISAL_ENGINE_ORACLE=1) mislabeled as production; (b) window-seeding REGRESSED
   between 19add96c and d56cb0f5 (bisect candidates: f7970c99 multi-member, 8d4f20f7 dist-cache
   shrink) dropping coverage 14->0; (c) a measurement misconfig in one. Bisect 19add96c..HEAD
   on the isal coverage counter if (b) is plausible.
5. RE-ESTABLISH the TRUE scorecard (native + isal, T1/T4/T8, coverage), Steward-banked +
   advisor-vetted.

## WHY IT MATTERS (the strategic stake)
- If ISA-L is runtime-dormant in production (fresh number right): gzippy-isal ≈ native on
  silesia, "GOAL #2 gzippy-isal ties rg in PRODUCTION" is OVERSTATED (it ties only with FORCED
  ISA-L), and the asm TARGET is the u16 MARKER BOOTSTRAP (98% of body) for BOTH builds — NOT
  the clean tail. The recent clean-tail framing would be misdirected.
- If ISA-L genuinely covers 14 in production (banked right): the residual owner mismeasured;
  the clean-tail framing holds. Find the mismeasurement.
This determines the asm target before the user's asm decision. Do NOT start any asm.

## GATES + DISCIPLINES
git WORKTREE; numbers ONLY from the bench-locked quiet guest; matched same-sink, interleaved
N>=9, sha-verified, path=ParallelSM. Route numbers through the Measurement-Integrity Steward
(bankability) + the inference through a synchronous Opus disproof advisor (verdict to
plans/isal-dormancy-advisor-verdict.md). SOURCE-VERIFY the binary feature-set + counter
semantics + env state first-hand (this is the mislabeled-oracle class — verify, don't trust a
label). RUN SUBAGENTS/ADVISORS SYNCHRONOUSLY. Run measurements YOURSELF holding the ssh.
Serialize builds via cargo-lock.sh, df -h around builds. No multi-line python via Bash. Wrap
hang-prone cmds in timeout. Diagnose the FIRST error before retrying. NO orphan processes /
sleep sentinels — pgrep clean on local + guest + neurotic before finishing. Update
plans/orchestrator-status.md + the disproof-ledger (DIS-12/OPEN-5). STOP at the checkpoint and
report: the TRUE coverage + scorecard + the explanation of the GOAL #2 discrepancy + the
resulting asm target (marker bootstrap vs clean tail).
