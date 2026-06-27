# BEAT silesia-T3 — output-size crossover bonus (2026-06-26)

Branch `feat/silesia-t3-output-crossover` @ `0c6bca87` (off the floor selector
`feat/serial-clean-floor-selector` e78c870a; **NOT merged**).
Scope stamp: **Intel** i7-13700T (16 logical cores), gzippy-native+isal-compression
(`build-flavor=parallel-sm+pure`, `path=ParallelSM`, `GZIPPY_FORCE_PARALLEL_SM=1`),
`fulcrum abmeasure` interleaved, `/dev/null` both arms, sha-verified every arm
against `gzip -dc` (no mismatch). **AMD owed** (llama untouched/SACRED).

## DONE-CRITERION — MET

gz **BEATS** rapidgzip at silesia-T3: **gz 504 ms vs rapidgzip 521 ms = 0.967**
(N=15, interleaved, /dev/null, sha-verified). Replicated across THREE independent
runs: Stage-1 N=15 515/530=0.973, crossover-sweep N=11 516/525=0.983,
Stage-3 gate N=15 504/521=0.967 — same sign every time (gz wins ~2-3%).

## Stage 1 — the decisive measurement (selector-fix vs overhead-reduction?)

Forced gz's PARALLEL path at silesia-T3 (`GZIPPY_PARALLEL_CROSSOVER_MARGIN=0`
disables the selector decline) vs rapidgzip-T3, N=15:

```
gz-parallel-T3 515ms   gz-serial-T3 (default selector) 672ms   rapidgzip-T3 530ms
base/rapidgzip 0.9728   after(serial)/rapidgzip 1.2690
```

VERDICT: **SELECTOR FIX (Stage 2A).** gz's parallel path already beats rapidgzip at
silesia-T3 (515<530). The serial-clean floor selector was conservatively *declining*
an available parallel win — not a structural/overhead problem.

## Root cause — the ISIZE-ratio proxy can't separate compressibility from fixed overhead

Measured TRUE per-corpus crossover (T where parallel first beats serial, N=11,
parallel=`MARGIN=0` vs serial=`MARGIN=100`):

```
corpus   ratio  output  true-xover  proxy ceil(ratio)  declined win?
silesia  3.1    212MiB  T3 (0.762)  4                  YES — T3 lost to rg (ser/rg 1.29)
squishy  2.76   480MiB  T2 (0.879)  3                  YES — T2 lost to rg (ser/rg 1.02)
monorepo 5.18    51MiB  T7 (0.889)  6                  no  — proxy ~correct
nasa     9.93   205MiB  hard-capped (ratio>=8)         no
```

The crossover ≈ W / (1 − B/S): W (work-inflation) ≈ 0.6·ratio, but B/S (the
parallel pipeline's FIXED overhead as a fraction of serial work) is LARGE for a
SMALL output (monorepo B/S≈0.5 → crossover pushed up to 7) and ~0 for a LARGE
output (silesia/squishy → crossover ≈ W, ~1 notch below the proxy). A single
linear `margin` can't serve both (squishy needs ≤0.725, monorepo needs >0.965 —
incompatible). **Output size is the missing variable.**

## The fix (surgical, one notch, large outputs only)

`effective_parallel_threads` (`src/decompress/parallel/single_member.rs`): after
computing `crossover = ceil(ratio·margin)`, subtract ONE when the output (ISIZE)
is ≥ 128 MiB (`GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES`, `0` disables). This lowers the
crossover for large outputs (where fixed overhead is amortized) without touching
small outputs (where the proxy is right). Effect is provably surgical — it can
only flip cells where output≥128MiB AND T==proxy_crossover−1:
  silesia crossover 4→3 (T3 flips serial→parallel),
  squishy crossover 3→2 (T2 flips serial→parallel),
  monorepo (51MiB) UNCHANGED, nasa hard-capped UNCHANGED.
Byte-exact (routes through the already-validated serial/parallel drivers).
947 lib tests + new `serial_clean_selector_large_output_bonus` unit test, clippy clean.

## Stage 3 — gated full-curve, no-regression proof (A/B: bonus OFF vs ON)

`old` = `GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES=0` (floor selector, pre-fix);
`new` = default (fix on). `new/old`<1 ⇒ my change improved it; ≈1 ⇒ untouched.

### vs rapidgzip (N=15) — gz now DOMINANT at EVERY cell
```
corpus    T   old   new   rg    new/old  new/rg
silesia   1   676   676   784   1.000    0.862
silesia   2   674   676   726   1.003    0.931
silesia   3   674   504   521   0.748    0.967   <- TARGET: now BEATS rg
silesia   4   376   375   398   0.997    0.942
silesia   6   315   315   335   1.000    0.940
silesia   8   273   277   290   1.015    0.955
silesia   12  250   247   266   0.988    0.929
silesia   16  261   266   279   1.019    0.953
monorepo  2   107   107   210   1.000    0.510
monorepo  3   107   106   139   0.991    0.763
monorepo  4   107   107   139   1.000    0.770
monorepo  5   107   107   117   1.000    0.915
monorepo  6   106   108   110   1.019    0.982
monorepo  8    89    88    96   0.989    0.917
monorepo  16   87    87    90   1.000    0.967
squishy   2  1345  1164  1275   0.865    0.913   <- bonus: was an rg LOSS (1.055), now win
squishy   3   848   851   859   1.004    0.991
squishy   4   656   655   685   0.998    0.956
nasa      2   217   217   590   1.000    0.368
nasa      4   215   216   354   1.005    0.610
nasa      8   217   217   227   1.000    0.956
```

### vs igzip (single-thread dominance, N=11) — no regression vs the floor competitor
```
corpus    T   old   new   igzip new/old  new/igzip
silesia   1   675   674   672   0.999    1.003   (tie, within spread; serial, untouched)
silesia   2   674   674   672   1.000    1.003   (tie; untouched)
silesia   3   676   502   673   0.743    0.746   <- now BEATS igzip too
silesia   4   374   379   671   1.013    0.565
silesia   8   271   272   672   1.004    0.405
silesia   16  263   266   674   1.011    0.395
monorepo  2-16  ~107..87        1.0±.01  0.79-0.97
squishy   2  1339  1159  1330   0.866    0.871
squishy   3   850   849  1327   0.999    0.640
squishy   4   660   663  1338   1.005    0.496
nasa      2   217   215   236   0.991    0.911
nasa      4-8                   ~1.0     0.915-0.919
```

## Verdict

- **silesia-T3 BEATS rapidgzip (0.967), GATED** (N=15, interleaved, /dev/null,
  sha-verified, 3× replicated). The last non-dominant cell is closed.
- **No regression anywhere**: `new/old` ∈ [0.743, 1.019] across all 22 cells × 2
  comparators; every value >1 is within inter-run spread (~1-2%); the only
  out-of-spread moves are the two intended improvements (silesia-T3 0.748,
  squishy-T2 0.865). monorepo + nasa provably untouched (below threshold / capped).
- **gz DOMINANT vs rapidgzip at EVERY cell × corpus** (was: all except silesia-T3).
- **gz DOMINANT vs igzip at every cell** (silesia T1/T2 = 1.003 tie within spread,
  pre-existing serial-floor near-tie, NOT caused by this change: new/old 0.999/1.0).
- **Bonus**: squishy-T2 was a *hidden* rg loss (serial 1345 vs rg 1275 = 1.055);
  the same fix flips it to a win (1164/1275 = 0.913).

## Reproduce (on the Intel trainer)
- Build: `CARGO_TARGET_DIR=/dev/shm/sel-target RUSTFLAGS="-C target-cpu=native"
  cargo build --release --no-default-features --features gzippy-native`
- Stage-3 gate: `/dev/shm/gate.sh` (COMP=rapidgzip|igzip N=15) — per (corpus,T)
  `fulcrum abmeasure` base=`GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES=0` vs after=default,
  `--core 0-15`, sha-verified.

## Status of claims
GATED (Intel, this session): silesia-T3-beats-rapidgzip, no-regression curve,
igzip/rapidgzip dominance, byte-exactness, unit tests. **AMD owed** (HYPOTHESIS
until replicated cross-arch). storedheavy corpus not on the box (owed).
