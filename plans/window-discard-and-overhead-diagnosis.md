# WINDOW-DISCARD + OVERHEAD DIAGNOSIS — leader charter (supervisor, 2026-06-07)

VAR_V integrated byte-exact but is a TIE: the +48% isolation gain was ABSORBED by production
clean-path overheads (ring %U8_RING_SIZE, wrap-straddle, resumable cap, CRC). Decode still binds
(~1.70×, Pool Fill ~83%). The inner ARITHMETIC is now igzip-class in isolation; the binder is the
production OVERHEADS + possibly window-handling. Advisor-upheld.

## STEP 0 — COMMIT VAR_V (rule 7a: byte-exact change is KEPT even on a TIE)
Commit the VAR_V integration (gain is latent, unlocked when the binding overhead is removed).
Concise message; note it's a byte-exact TIE kept per rule 7a.

## STEP 1 — PRIORITY LEVER: the window-discard (potential faithfulness BUG + possibly largest)
Advisor flagged `reset(None, window_opt)` (gzip_chunk.rs:1107) appears to DISCARD the supplied
predecessor window, forcing marker-mode (slow u16) bootstrap per chunk even when a window WAS
available. Investigate first-hand:
- Does that call discard an available window? Source-verify the path.
- Quantify: of the ~89% window-ABSENT (marker-mode) production chunks, how many had a predecessor
  window AVAILABLE but discarded (= the bug/lever) vs genuinely unavailable (= faithful, rapidgzip
  is also ~97% window-absent at runtime — see parity-final.md / project_faithful_unified...)?
- Source-verify vs vendor: rapidgzip seeds the window (setInitialWindow, deflate.hpp:693) → clean
  from block 0 when a window is known. If gzippy discards an available window, that is a divergence
  to fix. The faithful-unified memory says window-seed was DONE (+8.7%, gzip_chunk skips phase 1
  when a full 32KiB window is present) — confirm whether that regressed or reset(None) bypasses it.
If this is real and significant: it's a faithful fix (seed the window) with a quantified ceiling.
Pre-register the falsifier (fraction-of-chunks-rescued × marker-vs-clean per-chunk delta → wall).

## STEP 2 — SECONDARY: overhead-absorption (only if window-discard isn't the big lever)
Causal-perturb each production clean-path overhead to find WHICH absorbed VAR_V's gain:
ring modulo (%U8_RING_SIZE per store) vs wrap-straddle handling vs resumable n_max cap checks vs
CRC. Add a fired-fraction / clean-self-time counter (the advisor's owed proof that the gain is
"absorbed" not "loop-not-firing" — though the fast loop processes ~98% of bytes). Bound each
overhead's ceiling (remove it, measure the wall). Identify the binding overhead → the next fix.

## CHECKPOINT (STOP)
Report: VAR_V committed?; the window-discard finding (is a window discarded? what fraction? wall
ceiling if fixed?); if reached, the binding overhead. Route through an independent disproof advisor
(SYNCHRONOUS, read-only, verdict to plans/window-discard-advisor-verdict.md). Then STOP for
supervisor gate. Do NOT start a large build before the ceiling is bounded.

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- RUN SUBAGENTS SYNCHRONOUSLY (block with timeout, collect in-turn). Do NOT background-and-yield —
  NO auto-reinvoke; multiple leaders died this way. Run measurements YOURSELF via Bash holding the
  ssh; only the advisor is a delegated SYNCHRONOUS call.
- NO detached sleep sentinel. Before finishing, pgrep MUST show none of your claude -p subagents
  and no orphaned timeout `sleep` procs — kill them.
- SOURCE-VERIFY every premise first-hand (the offset-supply premise was wrong once; verify the
  window-discard claim against the actual code before acting). Serialize builds via cargo-lock.sh;
  don't run multi-line python via Bash; numbers only from the locked guest. Update
  plans/orchestrator-status.md.
