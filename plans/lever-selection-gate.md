# Lever-selection gate (Amdahl gate) — read BEFORE attacking any perf lever

The 2026-05-30 post-mortem's root cause was NOT bad measurement (per-lever refutation was
rigorous). It was **lever SELECTION**: ~14 levers were spent in a slice the project's own
*good* instrument had already sized at 14% of the gap, while the 86% slice with a proven
+28% lever sat un-attacked. This gate exists so that never happens again.

## RULE
Before attacking a lever, fill the template. **If the lever's ceiling (max fraction of the
gap it can touch, per the standing decomposition) is below the gap you need to close, DO NOT
ATTACK IT** — no benchmark, no agent. Re-derive the decomposition only if you can show the
prior one was measured on a broken/contaminated instrument (cite which).

## STANDING DECOMPOSITION of the x86 high-thread single-member gap (gzippy ~1.5-1.65× behind native rapidgzip)
Source instruments + their validity stamp (a claim is only usable if its instrument has a positive-control):
- **M0.3 (interleaved best-of-15, May 29 — VALID):** gap ≈ **85.9% structural pipeline / buffer-lifecycle, 14.1% decoder.** Even gzippy-with-ISA-L's-C loses on this cell ⇒ the decoder is NOT the gap.
- **Frozen window-export (2026-05-30 — VALID, sha-verified):** bootstrap/marker machinery ≈1.66× of production (STRUCTURAL); clean residual ≈1.17-1.41× (inner-loop dispatch — the 14% decoder slice).
- **Work-stealing driver e44bf0b (VALID, sha-verified):** replacing a wave-barrier with work-stealing lifted clean decode **2065→2654 (+28%)** — a STRUCTURAL/scheduling lever, UN-applied to production. THE biggest gain found all session.
- **clean-window oracle pre-b757038 (BROKEN — do not use): its "pipeline at parity 2035≈2067" corrupted CLAUDE.md's decision to rescind the structural port.**

## CEILINGS (use these to veto levers)
| Slice | Max fraction of the gap | ⇒ Levers it can/can't justify |
|---|---|---|
| Inner-loop / decoder (markers decode-speed, BMI2, multi-literal, branch-mispredict, distance-LUT) | **~14%** | CANNOT close a 1.5× gap alone. Attack ONLY after the structural slice is exhausted. (We did the opposite.) |
| Bootstrap MARKER decode rate | bandwidth-bound floor (proven) | DEAD for speed; only volume-reduction could help, and that's 0% reducible (kill-test). |
| **Structural pipeline: scheduling / load-balance / buffer-lifecycle / work-stealing** | **~86%** | THE slice. Work-stealing already proved +28% here. ATTACK THIS FIRST. |

## PER-LEVER PRE-REGISTRATION TEMPLATE (fill before attacking)
```
Lever: <name>
Slice it touches: <pipeline | decoder | bootstrap | ...>
Ceiling (max % of gap, per standing decomposition + which validated instrument): <%>
Gate: is ceiling >= the gap to close?  <YES → proceed | NO → VETO, do not attack>
Wall-elasticity hypothesis (on critical path? how measured): <...>
Kill-criterion (abandon if): <e.g. measure.sh shows TIE after N=11 interleaved>
Verdict (measure.sh, output-verified, interleaved): <WIN x% | TIE | LOSS>
```

## The next lever, gated (worked example — the one we should have done 10 levers ago)
- Lever: **port rapidgzip's scheduling/buffer-lifecycle to the production driver (work-stealing, bounded buffer reuse)**
- Slice: structural pipeline (~86% of the gap)
- Ceiling: up to ~86% — and work-stealing already showed +28% on the clean path. Gate: **YES, proceed.**
- Kill-criterion: if production `measure.sh` shows TIE vs baseline after the scheduling port, the production driver lacked the wave-barrier and the +28% was import-only — report and move to buffer-lifecycle.
