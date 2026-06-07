# T8 SEEDING-BUNDLE DECOMPOSITION — RESULT (2026-06-07, owner turn)

Pre-reg: plans/t8-decompose-prereg.md. Locked guest REDACTED_IP (double-ssh), 16c
gov=performance turbo-on load 1.3-2.0, measure.sh interleaved N=11 CPUS=0,2,4,6,8,10,12,14
RAW=68229982, output-verified=OK every run (sha 028bd002…cb410f), 2 independent runs.
Binary /tmp/gzbuild-isal/release/gzippy (gzippy-isal, target-cpu=native) + the 2 decompose
knobs added this turn (seed_no_windows / seed_no_boundaries, OFF==identity, byte-exact).

## The 4 production-pipeline cells (+ rg anchor)
| cell | what is seeded | run1 wall | run2 wall | vs rg |
|------|----------------|-----------|-----------|-------|
| rg (rapidgzip 0.16.0) | — | 0.1327s | 0.1322s | 1.000 |
| **seedfull** (windows + boundaries) | both | 0.1264s | 0.1343s | **~1.00× TIE** |
| onlywin (GZIPPY_SEED_NO_BOUNDARIES) | windows only | 0.1999s | 0.1986s | 0.66× LOSS |
| onlybnd (GZIPPY_SEED_NO_WINDOWS) | boundaries only | 0.1984s | 0.2051s | 0.66× LOSS |
| prod (no seeding) | nothing | 0.1976s | 0.2030s | 0.66× LOSS |

**onlywin ≈ onlybnd ≈ prod, all ~0.66× (Δ < spread). ONLY seedfull (both) ties.**

## Pre-registered formula → verdict
G = W_prod − W_full ≈ 0.197 − 0.130 = 0.067s.
  f_windows  = (W_prod − W_onlyW)/G ≈ (0.197 − 0.199)/0.067 ≈ ~0  (windows alone: NOTHING)
  f_boundary = (W_prod − W_onlyB)/G ≈ (0.197 − 0.198)/0.067 ≈ ~0  (boundaries alone: NOTHING)
NEITHER isolated knob recovers ≥0.3 of G, yet seed-full ties ⇒ **SUPER-ADDITIVE / COUPLED**
per the pre-registered rule. The gain needs BOTH windows AND boundaries together.

## The MECHANISM of the coupling (per-cell counters, GZIPPY_VERBOSE, first-hand)
| cell | window_seeded | seed_hits/misses | spec-fail header | Fill | decodeBlock | Theo-Opt | body_rate |
|------|--------------:|-----------------:|-----------------:|-----:|------------:|---------:|----------:|
| seedfull | **17** | 15/0 | **0** | **90.84%** | **0.846s** | 0.106s | (all clean) |
| onlywin  | 2  | **0/16** | 13 | 76.48% | 1.061s | 0.133s | 171 MB/s |
| onlybnd  | 1  | 0/15 | **0** | 78.03% | 1.106s | 0.138s | 170 MB/s |
| prod     | 2  | — | 13 | 80.41% | 1.067s | 0.133s | 169 MB/s |

1. **Windows are UNUSABLE without seeded boundaries** (onlywin: seed_hits=0). The seed store
   is keyed by REAL boundaries; with prod partition-GUESS dispatch, `window_map.get(start_bit)`
   /`seed_window_for(start_bit)` never hit those keys ⇒ onlywin is byte-for-counter IDENTICAL to
   prod (window_seeded=2, spec-fail=13, decodeBlock 1.06s). This is WHY windows+boundaries are
   coupled, not a measurement artifact.
2. **Real boundaries alone (onlybnd) ELIMINATE the 13 header spec-failures** (13→0) — a genuine,
   isolated effect of boundary-alignment — **but do NOT make chunks clean** (window_seeded=1,
   body still 170 MB/s u16 marker decode, decodeBlock 1.106s ≈ prod, actually marginally WORSE).
   ⇒ Removing spec-failures is wall-NEUTRAL here (onlybnd wall = prod wall). Spec-failure
   re-decodes (sub-lever c) are NOT the dominant T8 cost.
3. **Only seedfull converts marker→clean** (window_seeded 2→17, finished_no_flip 4→0,
   body→clean). decodeBlock collapses 1.07→0.85s, Fill 80→91%, wall 0.197→0.130s. The whole
   wall gain is the CLEAN-vs-u16-MARKER decode swap.

## APPLES-TO-APPLES with rapidgzip (--verbose, window-absent, SAME 34.5% marker workload)
| metric | gzippy prod (window-absent) | rapidgzip (window-absent) | gzippy seedfull (clean) |
|--------|----------------------------:|--------------------------:|------------------------:|
| Replaced-marker fraction | ~34.5% | 34.4981% | 0% (seeded clean) |
| decodeBlock (SUM workers) | 1.067s | **0.542s** | 0.846s |
| Theoretical-Optimal | 0.133s | **0.068-0.074s** | 0.106s |
| Fill | 80% | 76-84% | 91% |

**rapidgzip's MARKER decodeBlock (0.54s) is ~2× FASTER than gzippy's window-absent
decodeBlock (1.07s) — on the identical 34.5%-marker workload.** rapidgzip ties WITHOUT
seeding because its u16 marker-phase decode is ~2× faster per byte. gzippy only ties by
cheating (seedfull) — replacing the slow u16 marker decode with clean decode. Even
gzippy's CLEAN decode (0.846s) is 1.57× slower than rg's MARKER decode (0.54s); it ties
only because it skips applyWindow and reaches Fill 91%.

## PINPOINTED T8 SUB-LEVER
**marker-COMPUTE: gzippy's window-absent u16 marker decode path (~170 MB/s body, 1.07s
decodeBlock) is ~2× slower than rapidgzip's u16 marker decode (0.54s) on the same
window-absent 34.5%-marker workload.** This is the dominant ~1.5× T8 prod gap.
- Boundary-ALIGNMENT (sub-lever b) is a real but SECONDARY effect: it removes the 13
  spec-failures (onlybnd 13→0) yet is WALL-NEUTRAL (onlybnd wall ≈ prod wall) and is a
  PRECONDITION for window-application, not the dominant cost itself.
- Speculation-FAILURE re-decodes (sub-lever c) are NOT dominant (onlybnd removes them with
  zero wall change).
- The Phase-0 ISA-L oracle could NOT see this lever: ISA-L cannot emit u16 markers, so it
  only ever replaced the CLEAN tail. The u16 marker-COMPUTE path was never tested by it.
  ⇒ the asm / igzip-class inner-kernel work IS in scope HERE, adapted to u16 marker output.

## BOUNDED CEILING
The seedfull cell IS the ceiling oracle (Measurement PROCESS #3 — region removed by making
the marker path clean): converting marker→clean (gzippy's own clean decode) takes the T8
wall 0.197→0.130s = ties rg. So:
- **CEILING of fixing the marker-COMPUTE path = the T8 1.0× TIE with rapidgzip** (seedfull
  achieved 0.126-0.134s vs rg 0.132s). The full ~1.5× prod gap is recoverable on this lever.
- Two equivalent routes to that ceiling, both pure-Rust no-FFI:
  (i) make the window-absent path decode at rapidgzip's marker RATE (~2× faster u16 decode —
      igzip-class kernel adapted to u16 marker output), OR
  (ii) get correct windows onto more chunks earlier so they decode CLEAN in production (close
      the worker-vs-consumer publish race so window_seeded rises from 2/18 toward 17/18 — but
      rapidgzip is also ~window-absent and does NOT seed, so (i) is the faithful route; (ii)
      would diverge from how rapidgzip actually ties).
- The faithful lever is (i): **make gzippy's u16 marker decode ~2× faster to match rg's**,
  bounded to the T8 TIE by the seedfull oracle.

## ADVISOR REFINEMENTS (plans/t8-decompose-advisor-verdict.md — folded in)
1. **The 2×2 knobs CANNOT separate (a) marker-COMPUTE from (b) boundary-ALIGNMENT** —
   `onlywin` is DEGENERATE (windows unusable without seeded boundaries by construction ⇒
   ≡ prod; its pre-reg self-test seed_hits>0 FAILED ⇒ the cell is void as a windows-only
   isolation, not a wiring bug — it is the pre-reg branch-4 SUPER-ADDITIVE/COUPLED outcome).
   The verdict's real backbone is **onlybnd (the one clean isolation) + the rg cross-tool
   decodeBlock comparison**, NOT "the decomposition recovered f_windows/f_boundary."
2. **The 2× rate gap is the strongest pillar and is FAIR** (Q3 UPHELD): rg's decodeBlock is
   `decodeBlockTotalTime` summed over all decode invocations, Theo-Opt = /parallelization,
   Fill = optimal/wall — gzippy reproduces the formula 1:1 (0.133=1.067/8). applyWindow is a
   SEPARATE pass in BOTH tools ⇒ decodeBlock-to-decodeBlock isolates the marker-DECODE rate.
   The 2× survives zeroing spec-failures (onlybnd decodeBlock 1.106 ≈ 2× rg 0.542) ⇒ NOT a
   re-decode artifact.
3. **The CEILING is OPTIMISTIC; seedfull is the WRONG oracle for the faithful fix** (Q4,
   the most-important disproof). seedfull removes TWO things — (a) the marker-decode premium
   AND (d) the applyWindow serial pass (clean chunks skip marker-resolution entirely). The
   faithful route (i) "fast u16 marker decode" KEEPS applyWindow (must, to stay window-absent
   like rg). seedfull never exercises (fast-marker + applyWindow), so it bounds route (ii)
   (more clean windows), NOT route (i). The route-(i) ceiling rests on the **rapidgzip
   EXISTENCE PROOF** (rg does 0.54 decode + ~0.113s applyWindow → 0.13 wall), not seedfull.
   ⇒ State the ceiling as **"≤ TIE, upper bound, CONDITIONAL on gzippy's applyWindow /
   marker-resolution pass ≈ rg's ~0.113s."**
4. **OWED MEASUREMENT before claiming TIE-recoverable:** gzippy's applyWindow /
   marker-resolution pass cost vs rg's ~0.113s, on a cell that KEEPS applyWindow (a
   fast-marker prototype or a direct apply_window timer). No existing cell isolates it
   (seedfull eliminates it; onlybnd/onlywin/prod bury it in the slow marker path).

## NET (supervisor gate)
- **Pinpointed T8 sub-lever: marker-COMPUTE** — gzippy's window-absent u16 marker decode is
  ~2× slower per byte than rapidgzip's (decodeBlock 1.07s vs 0.54s, same window-absent 34.5%
  workload, denominator-matched, survives spec-failure removal). UPHELD-WITH-CAVEATS.
- **NOT the lever:** spec-failure re-decodes (c, wall-neutral); boundary-ALIGNMENT (b) is a
  secondary precondition, not the cost.
- **Bounded ceiling:** ≤ T8 1.0× TIE (rapidgzip existence proof), CONDITIONAL on applyWindow
  parity. The clean seedfull oracle achieved 0.126-0.134s vs rg 0.132s but over-removes
  applyWindow ⇒ optimistic; the conditional bound is the honest one.
- **Scoped fix for next loop (do NOT start this turn — bound-ceiling-first):** an igzip-class
  u16 marker-decode kernel (the asm/inner-kernel techniques adapted to u16 marker output —
  in scope HERE because the Phase-0 ISA-L oracle could never test the marker path). PLUS the
  owed prerequisite measurement: time gzippy's apply_window/marker-resolution vs rg's ~0.113s
  to confirm the ceiling is reachable before the kernel build.
