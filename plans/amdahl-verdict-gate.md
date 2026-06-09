# Amdahl-Crossover Verdict — Independent Disproof Gate

Read-only Opus gate on DIS-18 (owner self-verified, no advisor tool). Subject:
"the T-curve trough is an Amdahl crossover (gz W worse, gz S better), NOT a
machinery defect." Data: disproof-ledger.md DIS-18 (@ d56cb0f5, bin
b9eb0a733b4ccb6d, frozen guest, N=13, sha-OK, path=ParallelSM every cell).

## Data reconciliation (independent recompute) — PASSES

Re-fit S+W/T against the published curve, T3–T8:
- gz residuals (pred−meas): −0.9/+2.7/−0.1/−1.7/+0.3/+0.1 % — within r²=0.996.
- rg residuals: −1.7/+4.2/+0.6/−1.7/−0.8/+0.1 % — T4 worst (+4.2%), consistent
  with rg's lower r²=0.986.
- Crossover from S/W coefficients = **7.482**, matches the reported 7.49.
The fit, coefficients, and crossover are arithmetically self-consistent. The
data does reconcile with the model. The attack is therefore on the *premises*,
not the arithmetic.

---

## CLAIM 1 — Amdahl-fit validity (fixed work across T): **FIX-NEEDED**

The fit wall=S+W/T is only meaningful if W (parallelizable work) is T-INVARIANT
in the fitted regime. Verification status:

- **Chunk count fixed: SOUND.** rg `--verbose` Total Fetched = 17, CONSTANT
  T2=T4=T8 (16 prefetch + 1 on-demand); gz uses the same 4 MiB chunking. H4
  (granularity) correctly refuted. No contamination from chunk-count drift in
  T3–T8.
- **fb=0 across T3–T8: PARTIALLY verified — GAP.** isal_fallbacks=0 is BANKED at
  T4 and T8 only (DIS-13/BAR-2). T3/T5/T6/T7 fb-counters are NOT in the ledger.
  A single late-fired fallback (one silesia chunk re-decodes ~7.5×, per DIS-14)
  at any unmeasured T would inject a large local W spike that the 6-point fit
  would silently absorb into W. Low prior (fixed chunking, all-dynamic corpus),
  but UNMEASURED.
- **Marker-decoded fraction T-invariant: REFUTED as stated — this is the real
  soft spot.** The ledger's OWN decompositions show the marker fraction is
  T-DEPENDENT: T1 forced-SM = flip_to_clean=0, markers ZERO, window_seeded=16
  (DIS-15); T4 = flip_to_clean=12/14 (line 496). So the marker/engine work per
  unit output RISES from ~0 (T1, fully serial → every window pre-seeded) toward
  ~all-but-first (T4, speculative). W is NOT fixed across the FULL range.

  **Why this does NOT sink the verdict (but must be stated):** the rise is the
  T1→T3 serial-startup transition — exactly the regime the fit DROPS (T1/T2,
  r²=0.82/0.86, near-flat anti-Amdahl). By T3–T8, with 17 chunks and ≥3-way
  parallelism, speculation already saturates marker-decode (you cannot
  marker-decode more than ~all-but-one chunk), so flip_to_clean plausibly
  plateaus ≈12–14 across T3–T8 and W is approximately fixed there. That is the
  load-bearing assumption and it is **inferred, not measured** — flip_to_clean
  is banked at T4 only, never at T3/T5/T6/T7/T8.

VERDICT: the fit is valid IF marker-fraction plateaus and fb=0 in T3–T8; both are
plausible but UNVERIFIED at 4 of the 6 fitted points. This does not refute
engine-W (a T-rising marker fraction is MORE engine work, reinforcing the
direction), but it leaves the fit's "fixed W" premise resting on one measured
point. FIX: re-emit isal_chunks / isal_fallbacks / flip_to_clean at T3,T5,T6,T7
(free — same harness) and confirm the plateau before banking the coefficients.

## CLAIM 2 — Busy-fraction refutes starvation (H2/H3): **SOUND (with a refinement)**

The objection "busy ≠ productive — workers could be 91% busy SPINNING on a
window dependency (a head-of-line stall = a machinery defect masquerading as
busy)" is the right attack. It is defeated by the ledger's own data:

- **Spin artifact ruled out by the T8 point.** If gz workers spin-waited when
  starved, utilization would pin near 100% at every T. Instead gz busy-fraction
  FALLS to 72% at T8 — the measurement DOES capture real idle, so gz threads
  BLOCK (not spin) when starved. Therefore 91% at T4 genuinely means low idle
  ⇒ NOT starved ⇒ H2/H3 (starvation as the trough mechanism) soundly refuted.
- **Re-decode (wasted busy) ruled out:** fb=0 at T4 (banked) ⇒ no fallback
  re-decode inflating busy.
- **Refinement (does not change the verdict, sharpens the LEVER):** "busy" is
  NOT decomposed into productive-decode vs gz-specific marker-RESOLVE SCAFFOLD.
  DIS-19 attributes the +1.54e9-instr excess as ~71% marker inner-loop / ~25%
  resolution scaffold. So ~1/4 of the +35% W is SCAFFOLD (resolve/replace
  2-pass, u16→u8 narrow), which is machinery-class and potentially closable
  WITHOUT asm. The lever is therefore "~71% asm symbol-rate + ~25%
  scaffold-reducible," not purely "asm-bounded symbol rate." Busy-but-on-scaffold
  is still WORK, not starvation — H2/H3 stay refuted.

## CLAIM 3 — Verdict scope: **RESOLVED (2026-06-09, DIS-24): the high-T regime was measured and it is a LOSS — the curve TURNS OVER at T8, the S-floor story INVERTS.**

> RESOLUTION (DIS-24, frozen guest, interleaved N=9, sha-OK, path=ParallelSM, topology-controlled):
> the goal's 16+-thread regime is a **LOSS** for gzippy-isal. The T-curve PEAKS at T7/T8 (≈1.00-1.01)
> and gz loses EVERY cell T9..T32 (0.94 → trough 0.77 @ T12-14 → 0.86-0.91 @ T16/T24/T32). The high-T
> binder is **gz's T-PROPORTIONAL CHUNK-COUNT GROWTH** (finish_decode 14→34 as T 8→32) vs rg's CONSTANT
> ~17: every added chunk raises flip_to_clean (engine-W, 12→31), the serial publish-chain (floor S), and
> fallback risk (isal_fallbacks 0→1→2). rg holds a FLAT wall (336-378ms, saturated). The fixed-W premise
> (CLAIM 1) is now MEASURED-CONTAMINATED above T8 (fb≠0, flip_to_clean does not plateau). The T9 dip is
> DISENTANGLED: a real machinery knee (T9-E=0.938 with NO SMT spill) PLUS an additional SMT-spill penalty
> (T9-SMT=0.890); the dip survives clean physical placement, so it is the chunk-count machinery, not the
> topology. See plans/disproof-ledger.md DIS-24 (full table + counters). The original (now-falsified)
> CLAIM-3 text is preserved below.

### (original FIX-NEEDED text, FALSIFIED by DIS-24)

- **(a) High-T (T16/T32) — the GOAL's regime — is unmeasured, and T9 is a
  counterexample inside the measured range.** The S-floor story predicts gz
  KEEPS winning as T→∞ (smaller S). But T9 REGRESSES (0.873) while rg keeps
  scaling, and the owner's OWN mechanism is fatal to the unconditional verdict:
  "more/smaller chunks = more serial marker-resolve/window-publish chain links =
  a RAISED floor S." That means **S_gz is NOT a constant — it grows with chunk
  count, which grows at T≥9.** The "-26% better S" is a property of the FIXED-17-
  chunk T3–T8 regime, not of the architecture. At T16/T32 the chunk count rises
  further and gz's S advantage may erode or invert. The SMT-spill confound at T9
  is real but does not rescue this — it is co-present, not exonerating. The
  high-T win is the goal's headline and it is NOT banked; the only high-T data
  point trends AGAINST it.
- **(b) Corpus scope.** The fixed-work premise rests on silesia being
  all-dynamic with fb=0. On flush-dense / fallback-prone corpora a fallback
  storm balloons W (7.5× re-decodes) and the S+W/T model breaks. The engine-W
  verdict is **silesia-specific** until a second corpus is run.

VERDICT: bankable for **silesia, T3–T8, fixed-17-chunk regime ONLY.** Does NOT
cover T≥9 / T16 / T32 (where S grows with chunk count and the sole data point
regresses) nor non-all-dynamic corpora.

## CLAIM 4 — "Machinery S is BETTER than vendor (−26%)": **SOUND with attribution caveat**

S_gz<S_rg is REAL serial work gz genuinely does not do, but it is NOT generic
machinery superiority. The rg-machinery-map (lines 116–121, 192–194) attributes
it to TWO specific, byte-exact, NON-FAITHFUL-but-legitimate elisions: (i) eager
end-window publish that skips the consumer recompute
(`chunk_fetcher.rs:1802-1813`), and (ii) `out_fd` writev vs vendor `writeAll`.
So the correct framing is "gz does LESS serial work via two named, correct
elisions," not "gz's scheduler is intrinsically faster." This does not change the
lever (still W) and the elisions are correctly KEPT (rule 7a). Caveat for the
faithful-port goal: both elisions are deliberate DIVERGENCES from rg — they help
S precisely BECAUSE they are unfaithful, a standing tension with the port mandate
(flag, not a defect).

---

## BOTTOM LINE

The engine-W verdict is **BANKABLE, but scoped tighter than stated**: sound for
**silesia × T3–T8 × fixed-17-chunk regime**. Within that box: H2/H3 (starvation)
are soundly refuted (the T8 idle drop proves blocking-not-spinning, so 91% busy
at T4 is real low-idle), the fit reconciles arithmetically, and W_gz>W_rg /
S_gz<S_rg is the correct dominant structure. The LEVER (close W = inner-loop /
marker symbol rate) is correctly identified, with the refinement that ~25% of the
W-excess is marker-resolve SCAFFOLD (machinery-class, asm-free reducible) not raw
symbol rate.

The verdict does NOT cover, and must not be quoted as covering:
1. **The goal's own target regime (T16/T32).** The single high-T point measured
   (T9) REGRESSES, and the owner's mechanism (S grows with chunk count at T≥9)
   predicts the S-advantage can erode there. The "S is already at/below rg" claim
   is a fixed-chunk-regime fact, not an architectural one.
2. **Non-all-dynamic corpora** (fallback storms break fixed-W).
3. Two of the six fitted points' fixed-work premise (fb/flip_to_clean banked at
   T4/T8 only).

## Highest-value next measurements (ranked)

1. **T9–T16(–T32) curve, SMT-spill controlled** (pin one thread/P-core, no
   sibling spill; if cores exhausted, say so and bound). This directly tests
   whether the S-floor advantage survives the chunk-count growth that T9 already
   shows eroding it. It is the goal's target regime AND contains the only
   existing counterexample — top priority over a second corpus.
2. **Re-emit isal_chunks/fb/flip_to_clean at T3,T5,T6,T7** (free, same harness):
   confirm fb=0 and the marker-fraction plateau, converting the fixed-W premise
   from 1-point-inferred to 6-point-measured. Cheapest gap-closer.
3. **Second corpus** (a flush-dense / fallback-prone one): bounds the corpus
   scope of the engine-W verdict.

No edits made to source; this gate file is the only artifact.
