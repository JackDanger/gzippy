# DISPROOF-ADVISOR VERDICT — T8 seeding-bundle decomposition (2026-06-07)

Read-only. Source-verified against `src/decompress/parallel/seed_windows.rs`,
`src/decompress/parallel/chunk_fetcher.rs`, `vendor/rapidgzip/.../BlockFetcher.hpp`,
and the owner's `t8-decompose-findings.md` / `t8-decompose-prereg.md`. No build, no
measure, no edit.

## Headline
**Owner's core claim "marker-COMPUTE dominates the T8 prod gap" — UPHELD-WITH-CAVEATS.**
The *conclusion* survives, but its evidentiary backbone is NOT the decomposition knobs
(those only establish COUPLING + that spec-failures are not the cost). The backbone is
the cross-tool rapidgzip decodeBlock comparison, which I verify IS semantically fair —
so the verdict stands, but it must be re-attributed to "rg-comparison + onlybnd," not
"the 2×2 decomposition." Two arms of the decomposition are degenerate.

---

## Q1 — "windows unusable without seeded boundaries (seed_hits=0 in onlywin)": **REFUTED as an isolation (claim-of-fact is correct, but onlywin is not a valid windows-only cell).**

The mechanism is **code-correct**: `onlywin` sets `GZIPPY_SEED_NO_BOUNDARIES=1`, which
skips the block_finder pre-seed (`chunk_fetcher.rs:503` guard → loop 504–509 not run).
The seed store is keyed by REAL boundaries (`seedable_chunk_starts()` = `windows.keys()`,
`seed_windows.rs:260-273`); `seed_window_for(params.start_bit)` (2275) does
`store.windows.get(&start_bit)` against partition-GUESS dispatch offsets ⇒ structural
miss ⇒ `seed_hits=0`. Verified.

**Disproof finding:** because the miss is forced *by construction*, `onlywin` is
byte-for-counter IDENTICAL to `prod` (findings confirm: window_seeded=2, spec-fail=13,
decodeBlock≈1.06s). It therefore **does not isolate the windows sub-lever** — it is an
unrealizable cell that collapses onto prod. `f_windows≈0` does **not** mean "windows
don't help"; it means "windows cannot be applied without boundaries." The pre-registered
self-test (`prereg` line "seed-only-windows must show seed_hits>0") **FAILED** for this
cell, and the owner reinterpreted the failed self-test as a mechanism finding rather than
voiding the cell. That reinterpretation is *honest and code-justified* (the failure is
real coupling, not a wiring bug), but the consequence is that the 2×2 decomposition is
really a **3-cell story (seedfull / onlybnd / prod≡onlywin)**. The knobs **cannot
separate sub-lever (a) marker-COMPUTE from (b) boundary-ALIGNMENT** — they are
structurally coupled, exactly the pre-reg branch-4 (SUPER-ADDITIVE/COUPLED) outcome.

## Q2 — "onlybnd removes spec-failures yet is wall-neutral ⇒ boundary-alignment not dominant": **UPHELD-WITH-CAVEATS.**

`onlybnd` (`GZIPPY_SEED_NO_WINDOWS=1`) is the **one clean isolation**: block_finder is
pre-seeded (503 guard passes), `seed_window_for` forced to None (286-289). It passes its
pre-registered self-test (seed_hits=0/misses>0, block_finder seeded). Result spec-fail
13→0 with wall ≈ prod is genuine and the wall-neutrality is the PROCESS-correct verdict:
**spec-failure re-decodes (sub-lever c) are NOT the dominant cost.** Solid.

**Caveat:** onlybnd's decodeBlock is *marginally WORSE* than prod (1.106 vs 1.067) and
window_seeded *dropped* (2→1). So onlybnd is not "prod + free boundaries" — real-boundary
dispatch also perturbed the live window-publish race and yielded one fewer clean chunk,
adding marker bytes. This does not threaten the wall-neutral conclusion, but it means
onlybnd slightly *under*-counts any boundary benefit by suppressing a live clean hit. The
honest framing the owner already uses — boundary-alignment is a real but SECONDARY
*precondition*, not the cost — is the correct read.

## Q3 — rg marker decodeBlock 0.54s vs gzippy window-absent 1.07s, apples-to-apples: **UPHELD (semantics are fair).**

Verified in `BlockFetcher.hpp`: rapidgzip's `decodeBlock` line = `decodeBlockTotalTime`,
a **SUM over every decode invocation** (`+= duration(tDecodeStart,tDecodeEnd)`, :651);
`Theoretical Optimal = decodeBlockTotalTime / parallelization` (:82); `Fill =
optimal/wall` (:84). gzippy's findings label its column "SUM workers" and its
Theo-Opt/Fill columns reproduce rg's formula 1:1 (0.133=1.067/8, etc.) — strong
corroboration the statistic is a faithful port, **same denominator**. Both cells are
window-absent at the SAME ~34.5% marker density (rg --verbose 34.4981% vs gzippy ~34.5%).
applyWindow is a *separate* pass in both tools (not inside decodeBlock), so the
decodeBlock-to-decodeBlock comparison cleanly isolates the marker-DECODE rate. **The 2×
is a genuine per-byte rate gap.** Crucially, onlybnd (spec-fail=0) still shows
decodeBlock 1.106s ≈ 2× rg's 0.542 — so the 2× is NOT a re-decode artifact; it survives
removing every speculation failure. This is the single strongest pillar under the
owner's verdict and it holds.

## Q4 — "fixing marker-COMPUTE ⇒ T8 TIE, bounded by the seedfull oracle": **UPHELD-WITH-CAVEATS — the ceiling is OPTIMISTIC and the cited oracle is the wrong one. (Most important disproof finding.)**

`seedfull` makes every non-zero chunk take the CLEAN path (`decode_mode_clean`,
`chunk_fetcher.rs:2282-2283`; clean arm 2334-2351 / 2352-2363). A clean chunk emits final
bytes — it **skips the u16 marker-resolution / applyWindow pass entirely**. So seedfull
removes **TWO** things: (a) the marker-decode premium AND (d) the applyWindow serial pass.
rapidgzip PAYS applyWindow (campaign notes its ~0.113s apply-window pass; it is
window-absent on this workload). The owner's own findings admit seedfull "ties only
because it skips applyWindow and reaches Fill 91%," and that gzippy's CLEAN decode (0.846)
is still 1.57× slower than rg's MARKER decode (0.542).

**Therefore seedfull is NOT a PROCESS#3 removal oracle for the proposed faithful fix.**
The proposed route (i) — "make gzippy's u16 marker decode ~2× faster to match rg" —
*keeps* applyWindow (it must, to stay window-absent like rg). seedfull's configuration
(clean + no applyWindow) never exercises route (i)'s configuration (fast-marker +
applyWindow). seedfull bounds route (ii) (get more clean windows), not route (i). The
route-(i) ceiling actually rests on the **rapidgzip existence proof** (rg does
0.54 decode + applyWindow → 0.13 wall), not on seedfull.

**Optimism residual:** route (i) ties ONLY if gzippy's applyWindow ≈ rg's. That cost is
**untested by any of these four cells** (seedfull eliminates it; onlybnd/onlywin/prod
pay it inside the slow marker path, not separable). If gzippy's marker-resolution /
apply_window pass exceeds rg's ~0.113s, a marker-COMPUTE-only fix lands SHORT of TIE.
The owner should state the ceiling as "≤ TIE, upper-bound, conditional on applyWindow
parity," not "the full 1.5× gap is recoverable on this lever."

## Q5 — overall: which sub-lever does the data support? **"marker-COMPUTE dominates" UPHELD-WITH-CAVEATS.**

What the data robustly supports:
1. The wall gain is the **clean-vs-u16-marker decode swap** (seedfull internal counters:
   decodeBlock 1.07→0.85, finished_no_flip 4→0, Fill 80→91 — no cross-tool dependency).
2. **Spec-failure re-decodes (c) are NOT the cost** (onlybnd, clean isolation, wall-neutral).
3. The u16 marker decode is **~2× slower per byte than rg's**, and this survives zeroing
   spec-failures (onlybnd) and is measured on a denominator-matched statistic (Q3).

What the data does NOT support / must be down-weighted:
- The 2×2 decomposition does **not** itself separate (a) from (b) — onlywin is degenerate
  (Q1); the knobs land on the COUPLED branch. Re-attribute the verdict to "onlybnd +
  rg-comparison," not "the decomposition recovered f_windows / f_boundary."
- The ceiling is an **upper bound conditional on applyWindow parity**, not a guarantee
  (Q4). The cited seedfull oracle over-removes; the real warrant is the rg existence proof.

**Recommendation:** proceed with the marker-COMPUTE (igzip-class u16 kernel) lever — it
is the best-supported direction — but before claiming "TIE recoverable," add the missing
measurement: **gzippy's applyWindow / marker-resolution pass cost vs rg's ~0.113s**, on a
cell that keeps applyWindow (i.e., a fast-marker prototype or a direct apply_window
timer), since no existing cell isolates it. Without it, the ceiling is the optimistic
half of the claim.
