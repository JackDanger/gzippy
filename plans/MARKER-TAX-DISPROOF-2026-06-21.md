# MARKER-TAX-DISPROOF — does gz's marker loop being "1.5-2.3× heavier" gate a gz-vs-rg gap?

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz git:** d364eca2
**gz bin sha:** d20bc9558660…  **rg:** /tmp/rgvenv/bin/rapidgzip 0.16.0
**Box:** macOS Apple-Silicon (M1 Pro), quiet/deterministic.
**Stamp:** NOT-YET-LAW (mac aarch64 absolutes + Intel standing-matrix wall ratios;
AMD/Zen2 and a quiet-Intel absolute-instr replication OWED).

## CLAIM UNDER TEST (disproof attempt — actively attacked, not defended)
> "gzippy's marker decode loop is 1.5-2.3× heavier than rapidgzip's → a real
> convergence gap; porting rg's leaner marker loop will close a gz-vs-rg multi-thread
> gap."

The "1.5-2.3×" = PIPE-tax = gz_inflation / rg_inflation (each = instr_T4 / instr_T1).
SUSPECTED ERROR: it is a **ratio-of-ratios** inflated by gz's very light T1 base, not a
heavier absolute marker cost. Disproof predicts: if gz's ABSOLUTE marker-added
instr/B ≤ rg's, AND the gz-vs-rg gap ANTI-correlates with marker fraction, the claim
is FALSE.

## Gate-0 (self-validation, this session)
- **gz** rig `scripts/bench/standing/mac_pipeline_components.sh` N=7: path=ParallelSM,
  build-flavor=parallel-sm+pure, byte-exact gz==gzip all arms, /dev/null both arms,
  sig/spread 19-65×. **Non-inert PROVEN:** `GZIPPY_VERBOSE` — base4(-p4) routes 97.1%
  of body through the marker loop (body_bytes=41.8M, in-flight depth 10, fused_lut=9);
  `GZIPPY_NO_PREFETCH=1` COLLAPSES it to body_bytes=0, fused_lut=0, depth 4 (marker
  machinery provably eliminated). base4−noprefetch4 is therefore a CAUSAL removal of
  the marker machinery.
- **rg** rig `scripts/bench/standing/rg_marker_added.py` N=7: rg==gzip byte-exact,
  /dev/null sink, instr-spread ≤0.65% (TRUSTED). rg has no NO_PREFETCH oracle (it
  always speculates) so rg's marker-added = (instr_P4 − instr_P1)/B — a **clean
  absolute subtraction of two measured quantities** (NO ratio×ratio / share×wall).
- **CAVEAT (scope):** mac rg runs its aarch64 portable inflate (NO ISA-L). Valid for
  the marker/pipeline structural compare; the x86 ISA-L clean-kernel cells stay owed.

## D1 — THE ABSOLUTE COMPARISON (corpora ordered by DESCENDING marker fraction)

```
corpus     mkr% |  gzT1/B  gzT4/B  rgT1/B  rgT4/B | gz_infl rg_infl PIPEtax | gz_tax/B gz_mkrMach/B rg_add/B | gz-rg gap/B
nasa       89.8 |    3.42   13.45   13.93   24.09 |    3.94    1.73    2.28 |    10.04         9.54    10.15 |       -0.12
monorepo   80.9 |    6.84   21.58   22.79   34.70 |    3.15    1.52    2.07 |    14.73        14.25    11.91 |       +2.82
silesia    34.5 |    9.90   18.93   26.94   31.98 |    1.91    1.19    1.61 |     9.03         8.59     5.04 |       +3.99
squishy    31.6 |    9.53   17.15   26.64   31.24 |    1.80    1.17    1.53 |     7.62         6.97     4.60 |       +3.01
```
- `gz_tax/B` = (base4−base1)/B = gz's T1→T4 added instr/B (incl gz coordination) —
  apples-to-apples with rg's `rg_add/B` = (P4−P1)/B (incl rg coordination).
- `gz_mkrMach/B` = (base4−noprefetch4)/B = gz's CAUSAL marker-only added instr/B
  (coordination removed → even smaller than gz_tax).
- `gz-rg gap/B` = gz_tax/B − rg_add/B (positive = gz heavier in absolute instr/B).
- mkr% = rg `--verbose` replaced-marker fraction.

## D2 — MEASUREMENT 1 (absolute): the "uniformly 1.5-2.3× heavier" claim is FALSE
gz's absolute marker-added instr/B is NOT uniformly above rg's:
- **nasa (marker-HEAVIEST, 89.8%): gz_tax 10.04 ≈ rg_add 10.15 (gz −1.2%); causal
  gz_mkrMach 9.54 < rg 10.15 → gz's marker work is LIGHTER where markers dominate.**
- low-marker corpora: gz heavier in absolute terms (silesia +79%, squishy +66%,
  monorepo +24%) — but that added work is mostly NOT the marker loop (those streams
  are 65-68% CLEAN; the delta there is coordination + the smaller marker share).

So the marker LOOP is not "1.5-2.3× heavier" in any absolute sense; on the corpus
where the marker loop is ~the entire body (nasa) gz is at-or-below rg.

## D3 — MEASUREMENT 3 (ratio-of-ratios): the "2.3×" is a dimensional artifact
nasa is the clean demonstration: gz inflation 3.94× vs rg 1.73× → PIPEtax **2.28×**,
which the claim reads as "gz's marker machinery is 2.28× heavier." But the ABSOLUTE
added marker instr/B are gz 10.04 ≤ rg 10.15 — **gz adds LESS**. The 2.28× is
manufactured entirely by dividing by gz's tiny T1 base (3.42 instr/B vs rg's 13.93,
because gz's clean T1 is 3.6× lighter / has the inline T1 path). A −1% absolute
difference is reported as a 2.28× "gap." This is exactly the ratio-of-ratios /
light-base artifact CLAUDE.md's protocol forbids treating as a finding.

## D4 — MEASUREMENT 2 (correlation): the gz-vs-rg gap ANTI-correlates with markers
The claim REQUIRES the gz-vs-rg disadvantage to GROW with marker fraction (a heavier
marker loop hurts more where there are more markers). Measured on BOTH data sets:
- **mac absolute** (gz_tax − rg_add) instr/B vs marker%: **Pearson r = −0.767** (N=4).
  Gap is −0.12 at 89.8% markers, rising to +3.0…+4.0 at 31-35% markers.
- **Intel wall** gz/rg @T4 (STANDING-MATRIX-2026-06-20) vs marker%: **r = −0.999**
  (N=3): nasa 89.8%→0.960 (gz WINS), monorepo 80.9%→1.002 (TIE), silesia 34.5%→1.161
  (gz LOSES worst). Highest marker fraction = gz wins; lowest = gz loses most.

Both arches: **r < 0.** The claim predicts r > 0. The data shows the OPPOSITE sign.

## VERDICT — CLAIM DISPROVEN
The "marker decode loop is a gz-vs-rg convergence gap; port rg's leaner marker loop to
close a multi-thread gap" claim is **DISPROVEN**:
1. The "1.5-2.3× heavier" is a ratio-of-ratios artifact of gz's light T1 base, not a
   heavier absolute marker cost (D3); on the marker-dominated corpus gz's absolute
   marker work is ≤ rg's (D2).
2. The gz-vs-rg gap **anti-correlates** with marker fraction on both the mac absolute
   instr/B and the Intel wall ratios (D4, r=−0.77 / −0.999). A heavier marker loop
   would make the loss grow with marker fraction; it shrinks. Where gz actually LOSES
   to rg (Intel silesia-T4 +16%) the marker loop is LEAST active (~35% markers), so a
   leaner marker loop cannot be the lever there; where the marker loop dominates
   (nasa ~90%) gz already WINS/TIES rg.

### What IS the real multi-thread gz-vs-rg gap (where gz loses, Intel)
The loss tracks the **CLEAN fraction = 1 − marker** — exactly the portion rg decodes
with **ISA-L on x86**. silesia (65% clean) → rg's ISA-L clean kernel covers most of the
stream → gz loses (+16% T4). nasa (~10% clean) → ISA-L barely applies → gz wins. The
gz-vs-rg multi-thread gap is the **x86 ISA-L CLEAN inner-kernel**, NOT the marker loop
— consistent with STANDING-MATRIX "silesia is the standing loss" and the owed-to-Intel
x86 ISA-L kernel cells. This is a kernel/codegen target, not a marker-machinery port.

### Honestly held residual (do NOT over-correct)
gz's parallel marker MACHINERY does add MORE absolute instr/B than rg's on the
low-marker corpora (silesia +79%, squishy +66%, monorepo +24%). That instruction
difference is real and a generic-efficiency target. But it is NOT the gz-vs-rg
CONVERGENCE gap the claim asserts: it doesn't track where gz loses (anti-correlated),
and on the mac — where rg's clean kernel is portable-heavy — gz wins silesia-T4
*despite* the heavier marker machinery. Porting rg's marker loop would be a generic
efficiency change of unproven sign at the wall, not a fix for the silesia/Intel loss.

## Owed (NOT-YET-LAW)
- Quiet-Intel ABSOLUTE instr/B for gz & rg at T1/T4 (the Intel cells here are wall
  ratios from the standing matrix; the absolute-instr comparison is mac-aarch64 only).
- AMD/Zen2 replication.
- A Gate-2 perturbation directly on the x86 clean ISA-L kernel cell to causally
  confirm the clean kernel (not markers) carries the silesia-T4 loss.

## Reproduce
```
scripts/bench/standing/mac_pipeline_components.sh           # gz N=7, base1/base4/noprefetch4
python3 scripts/bench/standing/rg_marker_added.py 7         # rg P1/P4 absolute instr/B
python3 /tmp/disproof_calc.py                               # assemble table + Pearson r
GZIPPY_VERBOSE=1 target/release/gzippy -dc -p4 /tmp/monorepo.gz >/dev/null  # 97.1% marker
GZIPPY_VERBOSE=1 GZIPPY_NO_PREFETCH=1 target/release/gzippy -dc -p4 /tmp/monorepo.gz >/dev/null  # 0 marker (non-inert)
```
