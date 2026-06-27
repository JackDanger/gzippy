# Serial-clean cost-model selector — erases the low-T monotonicity regression (2026-06-26)

Branch `feat/serial-clean-floor-selector` @ `7879176` (NOT merged).
Scope stamp: **Intel** i7-13700T (16 logical cores), gzippy-native+isal-compression,
N=11 interleaved, `/dev/null` both arms, sha-verified, production `path=ParallelSM`.
**AMD owed** (llama untouched/SACRED — not measured here). storedheavy corpus not
present on this box (owed); silesia + monorepo are the demonstrated regression cells.

## The selector (how it predicts serial vs parallel, conservatively)

One-line change in scope: extend `effective_parallel_threads`
(`src/decompress/parallel/single_member.rs`) — the EXISTING ratio-cap machinery —
with a cost-model crossover. When it returns 1, the driver already routes the whole
decode (chunk sizing + driver) to the proven serial-clean thin-T1 path via the
existing `parallelization<=1` branch in `read_parallel_sm`. No new driver; byte-exact
by construction.

Mechanism (gated, T2-MECHANISM-2026-06-26): the parallel pipeline does
**W ≈ (ISIZE/deflate ratio)× more total CPU work** than the serial-clean driver —
marker-RESOLUTION + apply_window + per-chunk SCAFFOLD are OUTPUT-proportional and
under-amortized at low T (decode kernel K≈1.0). Predicted parallel speedup ≈ T/W, so:

```
hard ratio cap (pre-existing): ratio >= 8           -> 1 thread (parallel hurts at all T)
NEW cost-model crossover:      T < ceil(ratio*margin) -> 1 thread (serial-clean floor)
otherwise                                            -> requested T threads
```

`margin` defaults to 1.0 (env `GZIPPY_PARALLEL_CROSSOVER_MARGIN`; `0` disables the
selector for A/B). `ceil()` is conservative-by-construction (rounds the crossover UP →
stays on the safe serial floor one notch longer). The asymmetry licenses the
conservatism: misrouting a cell to PARALLEL below its crossover REGRESSES it below T1
(loses to single-thread igzip); misrouting to SERIAL at worst ties gz-T1 — and gz-T1
already BEATS single-thread igzip, so the floor is a WIN, not a tie.

Per-corpus crossover the model picks (ratio measured on-box):
silesia 3.11→4, monorepo 5.18→6, squishy 2.76→3, nasa 9.93→hard-cap (always serial).

## Byte-exactness (BLOCKING) — PASS

- Routes to the existing proven serial driver (`drive_thin_t1_oracle`) via
  `parallelization<=1`; same path `GZIPPY_THIN_T1_ORACLE` already validated at any T.
- sha == gzip on the x86 production ParallelSM path: silesia T2/T3/T4/T8 all OK.
- Local arm64 ParallelSM: sha-identical across T1/T2/T3/T4/T8 on silesia-shaped 30 MiB
  and 430 MiB corpora.
- `fulcrum abmeasure` sha-verifies every arm of every cell against the gzip oracle —
  all 26 cells × 2 gz arms passed (it aborts on mismatch).
- 947 lib tests pass (incl. new selector unit tests), clippy clean.

## BEFORE/AFTER all-T scaling curve (wall ms; base = selector OFF = old parallel-always)

`after_over_base` < 1 ⇒ selector faster than old. `after_over_comp` < 1 ⇒ gz dominant.

### vs single-thread igzip
```
corpus    T   base(old)  after(sel)  igzip   after/base   after/igzip
silesia   1   681        684         688     1.004        0.994
silesia   2   707*       676         689     0.956        0.981   <- REGRESSION ERASED
silesia   3   529        677         686     1.28         0.987   (conservative: serial)
silesia   4   396        399         690     1.008        0.578
silesia   8   292        288         689     0.986        0.418
silesia   16  275        269         691     0.978        0.389
monorepo  1   107        109         113     1.019        0.965
monorepo  2   202*       107         112     0.53         0.955   <- 1.89x REGRESSION ERASED
monorepo  3   136*       107         113     0.787        0.947   <- ERASED
monorepo  4   128*       107         113     0.836        0.947   <- ERASED
monorepo  6   110        111         111     1.009        1.0
monorepo  8   89         91          112     1.022        0.812
monorepo  16  87         87          112     1.0          0.777
squishy   1   1200       1204        1215    1.003        0.991
squishy   2   1115       1206        1222    1.082        0.987   (conservative; no regression)
squishy   4   626        629         1216    1.005        0.517
squishy   16  397        407         1217    1.025        0.334
nasa      1   216        215         237     0.995        0.907   (hard-capped serial; flat)
nasa      8   217        217         241     1.0          0.9
```
`*` = a cell where the OLD behavior was SLOWER than its own T1 (monotonicity violation,
loses to single-thread igzip). All erased by the selector.

### vs parallel rapidgzip (the only place a residual appears)
```
corpus    T   after(sel)  rapidgzip   after/rapidgzip
silesia   2   679         816         0.832
silesia   3   677         605         1.119   <-- ONLY non-dominant cell
silesia   4   392         469         0.836
monorepo  2   109         262         0.416
monorepo  8   91          148         0.615
squishy   2   1200        1292        0.929   (serial, still beats rg)
nasa      2   217         651         0.333
```
Every cell across all 4 corpora: gz-after beats rapidgzip — EXCEPT **silesia T3**.

## Verdict: (a) with one documented residual

- **Low-T regression ERASED.** Every monotonicity violation in the old behavior
  (monorepo T2 1.89×, T3, T4; silesia T2 1.05×) is gone. The after-curve is
  non-increasing within spread up to physical cores on every corpus (the T16 / SMT
  boundary bumps are within ±5% spread — tolerated per the relaxed gate).
- **Dominant vs every single-thread competitor at EVERY T.** gz-after ≤ igzip at all
  26 cells (monorepo T6 = 1.000 tie); igzip < libdeflate (734 vs 776 ms silesia) so gz
  beats libdeflate everywhere too. The regressed cells now BEAT single-thread igzip
  (the exact defect: old monorepo T2 lost to igzip 1.80×, now wins 0.955).
- **Dominant vs parallel rapidgzip at every cell EXCEPT silesia T3** (1.119).

### The one residual (silesia T3), and why it is NOT chased here
silesia's true crossover is T3 — the old parallel path runs it in 526 ms (beats
rapidgzip 605, 0.87) — but `ceil(3.11)=4` conservatively keeps T3 on the serial floor
(677 ms). It is NOT a regression (677 ≈ gz-T1, still beats igzip); it just declines a
parallel win that rapidgzip captures. The conservatism is load-bearing: the ratio→
crossover map is super-linear across corpora (silesia 3.11→wants 3; monorepo 5.18→wants
~7-8), so NO single linear `margin` fixes silesia without re-regressing monorepo T5
(margin that gives silesia crossover 3 gives monorepo crossover 5 → T5 parallel ≈118 ms
> serial 107 = a fresh monotonicity violation). This is exactly the workload-dependent-
constants caveat the reviews raised. Per the conservative mandate (misrouting-to-
parallel is the cardinal sin; ties are acceptable), the selector errs to the safe floor.
Closing silesia T3 is the **win-more follow-on**, not an overfit second constant.

## Follow-ons (selector leaves room for; NOT built here)
- **T-adaptive, work-proportional chunking** (chunk by OUTPUT, keep the ratio-cap):
  finer balance at any T; can lower the effective crossover so silesia T3 (and squishy
  T2) go parallel profitably — closing the residual without re-regressing high-W corpora.
- **Cheapen the parallel low-T overhead at its source** (marker-resolution multi-pass +
  per-chunk scaffold + apply_window amortization): lowers W → lowers the real crossover →
  more cells profitably parallel. This is the faithful "#1" (cheaper speculation, NOT
  window-flow-by-waiting which serializes).
- Replace the ratio proxy for W with a measured/block-density signal to fix the
  super-linear ratio→crossover mismatch (silesia vs monorepo).

## Reproduce
`scripts/bench/_serial_clean_curve.sh` (on the trainer): per (corpus,T)
`fulcrum abmeasure`, base arm `GZIPPY_PARALLEL_CROSSOVER_MARGIN=0` (selector off) vs
after arm (on), `--core 0-15`, n=11, `--no-gate`, parses `ABMEASURE-WALL`.
`COMPETITOR=igzip|rapidgzip`.

## Status of claims
GATED (Intel, this session): regression-erasure, monotonicity, igzip/libdeflate
dominance, byte-exactness. The silesia-T3-vs-rapidgzip residual is GATED. All AMD
claims are **owed** (HYPOTHESIS until replicated). storedheavy corpus owed.
