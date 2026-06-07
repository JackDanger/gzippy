# STEP A.2 — CLEAN-ONLY ENGINE ORACLE (supervisor, 2026-06-07)

The STEP-A oracles (advisor-vetted, plans/step-a-oracle-advisor-verdict.md) CORRECTED
the lever ranking. Recorded in memory project_pregate_placement_is_dominant_lever.

## Corrected picture (advisor-vetted)
- **Placement and ENGINE are CO-PRIMARY; the 1.0× tie needs BOTH.**
- Oracle-P: placement-perfect gzippy = 0.56–0.66s vs rapidgzip 0.524s = **7–26% LOSS**
  → placement is NECESSARY-BUT-INSUFFICIENT (largest single lever, correct FIRST step,
  faithful scheduler port).
- Engine residual survives perfect placement: gzippy clean decode **91 ms/chunk vs
  rapidgzip 39 ms = 2.3×**. The inner-loop pure-Rust+inline-ASM work is now JUSTIFIED as
  co-primary (not the local-optimum trap).
- Oracle-C was DEGENERATE (free decode also frees windows → publish-chain collapses);
  its 0.4–0.7s is GREY and cannot bound class-C.

## THIS STEP (the advisor's explicitly-owed measurement, before any STEP-C design)
Build + run a **clean-only T8 removal oracle**: force ALL chunks through the CLEAN decode
path (seed each with its predecessor window so none is window-absent/markered), and measure
the engine busy / T8 wall — the LEAST-ENTANGLED engine signal that both prior oracles
missed (it preserves the publish-chain that Oracle-C collapsed). This cleanly bounds the
ENGINE (class-C) ceiling and resolves the grey 0.4–0.7s.
- Pre-register the falsifier BEFORE running (plans/). Positive control / self-test: prove
  it actually forced the clean path (e.g. window-absent fraction → ~0, marker decode → ~0
  in the trace) AND that it preserved the publish-chain (unlike degenerate Oracle-C).
- Numbers ONLY from the locked guest harness; verify guest IDLE first; interleaved N≥9.

## SECONDARY (if guest free after the oracle): STEP B 1b traffic A/B
u8-clean-write + drop-clean-byte resolve over the ~20MiB u16 buffer — rank class-T as a
possible THIRD co-lever. Bound its ceiling. Lower priority than the clean-only oracle.

## CHECKPOINT (STOP)
Report the cleanly-bounded ENGINE ceiling (and 1b if run). Route through an independent
disproof advisor (verdict to plans/). Do NOT start STEP-C design revision or TIER-3 —
supervisor + advisor gate that. After this, the measurement inputs for the design are
complete: placement ceiling (0.56–0.66 floor), engine ceiling (this step), traffic (1b).

## NEXT (future, after corroboration) — for context, NOT this turn
STEP C: revise the TIER-1 design with placement + engine as CO-PRIMARY (placement =
faithful port of rapidgzip's scheduler/prefetch FIRST; engine = inner-loop ASM to close
the 2.3× clean-rate gap). STEP D: TIER-3 align, placement first, dual-sha byte-exact.

## DISCIPLINES (unchanged, enforced)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; ONE of YOUR
subagent's guest runs must be driven by a Bash task that HOLDS the ssh (a bare `claude -p`
that prints-and-exits SIGHUPs its ssh and orphans the guest run — this already bit STEP A);
serialize builds via cargo-lock.sh; verify guest idle before each measurement AND restore
host after (no_turbo=0, thaw guests); reject a lever only with a mechanism.
