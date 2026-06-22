# T1-MONOLITH — PRE-REGISTERED FALSIFIER (commit BEFORE building; do not move goalposts)

Branch `t1-monolith` off `kernel-converge-A` @ f378c1cd. Author: x86 T1 driver-convergence
leader. This file is committed and pushed BEFORE the monolith path is built, so the verdict
is judged against a frozen threshold.

## Context (gated starting point — LAW-grade, Intel+AMD agree)

Per `plans/PROD-PATH-LOCATE-RESULTS.md` / `project_x86_prod_path_locate_2026_06_21`:
the REAL x86 production T1 driver (`decompress_parallel(...,1)` → `drive_thin_t1_oracle`)
is **+28.8..42.8% slower than the igzip monolith on all 8 cells**, holding the inner
kernel constant. The gap is per-chunk FIXED driver cost (per-chunk output alloc +
ratio-reserve with NO recycling pool; per-chunk window clone+re-seed; per-chunk ISA-L
lifecycle; per-block boundary record/subchunk split — all PURE cost at T1) + CRC
second-touch (2–12%) + a suboptimal 1MiB T1 chunk default. The KERNEL is NOT the lever
on the real path (native = parity-or-faster than ISA-L under the same driver). The lever
is the DRIVER. gzippy-NATIVE prod/igzip = **+15.8..40.3%** (1.158..1.403), almost
entirely driver.

## What is being built

A T1-GATED monolith decode path faithful to the igzip serial monolith shape
(`vendor/isa-l/igzip/igzip_inflate.c isal_inflate :2239`): ONE reused output buffer
reserved upfront to the whole-member ISIZE (NOT compressed×factor capped at 64 MiB —
that cap is the `prodbig` reserve-balloon confound), ZERO per-chunk alloc, ZERO per-chunk
window clone+re-seed, ZERO per-block boundary record / subchunk split, implicit history
(the single contiguous buffer IS the window). Gated strictly on `T==1`; `T>1` keeps the
faithful rapidgzip chunk pipeline untouched (chunk granularity is T>1-LOAD-BEARING).
gzippy-native drives the pure-Rust `decode_clean_into_contig` kernel over the whole
stream; gzippy-isal drives one full-stream ISA-L call (≈ igzip).

## PRE-REGISTERED VERDICT THRESHOLDS (frozen; the measurement is judged against these)

Metric = `prod_ms / igzip_ms` (interleaved best-of-N≥7, /dev/null both arms, decode-only
timed, sha==zcat all arms, A/A ≪ Δ), where `prod` is the gzippy-NATIVE monolith and
`igzip` is the ISA-L monolith bar WITH CRC. Cells = {silesia, nasa, monorepo, squishy} ×
{Intel neurotic, AMD solvency} = 8 cells.

- **CONFIRMED** iff after the T1-monolith path gzippy-NATIVE prod/igzip drops from the
  current **1.158..1.403** to **<= 1.10 on ALL 8 cells** (Intel+AMD), byte-exact (sha==zcat),
  AND T4/T8 do NOT regress vs the current build (and vs rapidgzip) — i.e. the T>1 chunk
  pipeline is provably untouched.
- **FALSIFIED** iff residual prod/igzip stays **> ~1.10** on any cell after shedding the
  located per-chunk bookkeeping → the cost is INTRINSIC TO THE CHUNK-DECODE STRUCTURE,
  not the alloc/window/boundary glue. Report it as the gated finding; the next lever is
  the structure (e.g. the inline-vs-second-touch CRC, the per-symbol kernel codegen),
  not the glue. Do NOT narrate a partial drop (e.g. 1.40 → 1.18) as "closed" — state it
  against this 1.10 line.

## ALSO PRE-REGISTERED — post-shed kernel re-measurement

After the driver is shed, re-measure native-vs-igzip KERNEL ceiling on the monolith path:
`(native_monolith − igzip)/igzip` and `(native_monolith − isal_monolith)/isal_monolith`.
The bias-check predicts the kernel may RE-EMERGE as the dominant residual term once the
driver overhead is gone. If `native_monolith/igzip` stays >1.10 while `isal_monolith/igzip`
≈ 1.00, the residual is the pure-Rust per-symbol KERNEL (the NEXT real lever, per
no-phases). Report this regardless of the headline verdict — do not hide it.

## STRATEGIC FACT (gated, for the supervisor → user)

Measure `rapidgzip_ms / igzip_ms` at T1 on all corpora, both arches. If rapidgzip ITSELF
loses to igzip at T1, then "faithful-to-rg" and ">=0.99x vs igzip at T1" genuinely
conflict, which JUSTIFIES the T1-monolith divergence as necessary, not optional. Report
the gated rg/igzip T1 ratios.

## Measurement discipline (gates that make a number exist)

- Gate-0: sha==zcat all arms; A/A `|prod − prod2|` ≪ Δ; /dev/null both arms; igzip bar
  self-tests (igzip-vs-itself ≈ 1.0 ± spread); routing non-inert (GZIPPY_DEBUG shows the
  monolith path + native kernel fingerprint, MONOLITH_T1_RUNS fired).
- Gate-1: interleaved best-of-N≥7 (N=15 on noisy LXC), Δ vs inter-run spread, label TIEs.
- Gate-3: Intel (neurotic, taskset cpu4) AND AMD (solvency, frozen gov=performance
  boost=0, thawed+verified after); T1 AND T4/T8 (no cross-cell regression).
- Gate-4: GZIPPY_DEBUG routing + feature-set fingerprint (native vs isal build).
