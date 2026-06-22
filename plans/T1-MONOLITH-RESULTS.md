# T1-MONOLITH — GATED RESULTS (judged against the pre-registered falsifier)

Branch `t1-monolith` off `kernel-converge-A`. Instrument: `examples/streaming_thin`
(`prod` mode = `decompress_parallel(&[u8],…,1)`, now routed to the T1-monolith) +
`scripts/bench/_t1_monolith_falsifier.sh`. Bar = `igzip` mode (ISA-L monolith one-shot
WITH CRC, via `decompress_gzip_stream` — note: STREAMS through a small reused buffer).
/dev/null both arms, decode-only timed, interleaved best-of-15, taskset cpu4.

## VERDICT: **FALSIFIED** (with mechanism)

The pre-registered CONFIRMED threshold was `prodN/igzip <= 1.10 on all 8 cells`. The
gzippy-NATIVE monolith did NOT meet it — it **REGRESSED past the legacy thin-T1 driver**
on every cell. Per the pre-registered rule, FALSIFIED ⇒ the cost is NOT the per-chunk
alloc/window/boundary glue: shedding that glue to a single ISIZE buffer INTRODUCED a
larger cost (whole-output first-touch page faults) than it removed.

## INTEL (neurotic i7-13700T LXC, taskset cpu4, best-of-15, load ~2.3; A/A ≤ 1.1 ms)

| corpus   | igzip ms | prodN (monolith) | prodNthin (legacy T1) | prodI (isal monolith) | **FALSIFIER prodN/igzip** | BEFORE thin/igzip | KERNEL (nat−isal) |
|----------|----------|------------------|-----------------------|-----------------------|---------------------------|-------------------|-------------------|
| silesia  | 656.3    | 914.2            | 796.2                 | 837.6                 | **1.393**                 | 1.213             | +9.15%            |
| nasa     | 233.0    | 416.1            | 293.6                 | 376.2                 | **1.786**                 | 1.260             | +10.61%           |
| monorepo | 105.6    | 162.0            | 142.9                 | 142.5                 | **1.533**                 | 1.353             | +13.67%           |
| squishy  | 1288.3   | 1765.0           | 1517.7                | 1645.1                | **1.370**                 | 1.178             | +7.29%            |

Gate-0 bytes==zcat PASS all arms; A/A |prodN−prodN2| ≤ 1.06 ms ≪ every Δ. Routing
(Gate-4): prodN fired `T1-MONOLITH` (MONOLITH_NATIVE), prodNthin fired `thin-T1`,
prodI fired `T1-MONOLITH` (MONOLITH_ISAL).

**The monolith is SLOWER than the legacy thin-T1 driver on every Intel cell** (prodN >
prodNthin), and even the isal-monolith (one full-stream ISA-L call) is +28..61% over
igzip — far above the ≈0 a true monolith bar would predict.

## MECHANISM (Gate-2 causal, perf minor-faults — software counter reliable on LXC)

silesia, taskset cpu4:

| arm                | wall ms | minor-faults |
|--------------------|---------|--------------|
| igzip (bar)        | 661     | 17,095       |
| prodN (monolith)   | 922     | **68,958**   |
| prodNthin (thin T1)| 796     | 23,046       |

The monolith faults **~4× igzip / ~3× thin-T1** (≈46k EXTRA faults). silesia output is
212 MB = 53k pages: the single ISIZE buffer first-touches nearly all of them fresh during
decode. The bar `igzip` (`decompress_gzip_stream`) and the legacy `thin-T1` (1 MiB
recycled chunks) both STREAM through a small cache-warm buffer and never materialize the
whole output — so they pay ~17–23k faults. **igzip's T1 advantage is streaming through a
small reused buffer, NOT a hold-everything monolith.** This is the extreme bad end of the
prior cycle's documented chunk-size U-curve (2 MiB optimum; giant buffer slower).

## STRATEGIC FACT (gated) — rapidgzip ITSELF loses to igzip at T1

CLI wall, taskset cpu4, interleaved best-of-7/9, /dev/null (Intel neurotic):

| corpus   | rapidgzip −P1 ms | igzip ms | **rg/igzip** |
|----------|------------------|----------|--------------|
| silesia  | 825.5            | 670.7    | **1.231**    |
| nasa     | 441.7            | 237.6    | **1.859**    |
| monorepo | 207.2            | 114.3    | **1.813**    |
| squishy  | 1578.1           | 1318.5   | **1.197**    |

rapidgzip at −P1 loses to igzip by **+20..86%** on all 4 corpora. So "faithful-to-rg"
and ">=0.99x vs igzip at T1" genuinely CONFLICT at T1 — rapidgzip's chunked architecture
is not built to win single-threaded; igzip's serial monolith is. A T1-specialization
divergence FROM rg is therefore NECESSARY, not optional. (Notably, gzippy-native's legacy
thin-T1 already BEATS rapidgzip −P1 at T1: nasa 1.260 vs 1.859, monorepo 1.353 vs 1.813,
silesia 1.213 vs 1.231 — the residual gap is purely vs igzip, the single-stream SOTA.)

## POST-SHED KERNEL re-measurement (pre-registered)

Under the monolith driver, KERNEL (native − isal)/isal = +7.3..13.7% (Intel). This is
CONFOUNDED by the monolith's fault storm (both arms fault the giant buffer), so it is NOT
a clean kernel ceiling. The clean prior-cycle measurement (native = ISA-L parity-or-faster
on the real chunked path) stands. The kernel did NOT cleanly re-emerge as a separable
lever here because the monolith's page-fault cost dominates both arms.

## AMD (solvency EPYC 7282 Zen2, FROZEN gov=performance boost=0, taskset cpu4, best-of-15)
Box thawed + verified after: governor=ondemand, boost=1.

| corpus   | igzip ms | prodN (monolith) | prodNthin (legacy T1) | prodI (isal monolith) | **FALSIFIER prodN/igzip** | BEFORE thin/igzip | KERNEL (nat−isal) |
|----------|----------|------------------|-----------------------|-----------------------|---------------------------|-------------------|-------------------|
| silesia  | 357.2    | 543.8            | 429.0                 | 503.3                 | **1.522**                 | 1.201             | +8.03%            |
| nasa     | 131.4    | 285.4            | 163.9                 | 264.0                 | **2.173**                 | 1.248             | +8.12%            |
| monorepo | 58.0     | 101.1            | 80.7                  | 91.3                  | **1.744**                 | 1.392             | +10.73%           |
| squishy  | 704.6    | 1050.5           | 815.2                 | 987.9                 | **1.491**                 | 1.157             | +6.34%            |

Gate-0 bytes==zcat PASS all; A/A ≤ 2.6 ms ≪ every Δ. Same verdict as Intel: the monolith
is SLOWER than the legacy thin-T1 driver on every cell.

### AMD mechanism (bare-metal HW counters, silesia, taskset cpu4)

| arm                | minor-faults | instructions | cycles  |
|--------------------|--------------|--------------|---------|
| igzip (bar)        | 17,092       | 2.517 B      | 1.188 B |
| prodN (monolith)   | **68,950**   | 3.171 B      | 1.738 B |
| prodNthin (thin T1)| 23,040       | 2.954 B      | 1.383 B |

Identical fault profile to Intel (68,950 vs 17,092 vs 23,040). The monolith also runs MORE
instructions (3.17 B vs thin's 2.95 B) and cycles (1.74 B vs 1.38 B) than thin despite
shedding the per-chunk glue — the giant-buffer fault handling + cold CRC/write passes
exceed the glue it removed. **Cross-arch LAW (Intel + AMD agree): the monolith faults ~4×
igzip and is the wrong shape for T1.**

## CROSS-ARCH VERDICT (Intel + AMD AGREE → LAW-grade): FALSIFIED on all 8 cells

prodN/igzip = 1.37–1.79 (Intel) / 1.49–2.17 (AMD) — far above the pre-registered 1.10
CONFIRMED line, AND worse than the legacy thin-T1 driver (1.18–1.39) on every cell.

## CONCLUSION + REDIRECT (gated)

- The monolith hypothesis is **FALSIFIED**: a single ISIZE-sized output buffer is the
  WRONG shape for T1. It does not converge to igzip; it regresses past gzippy's own
  thin-T1 driver because it materializes the whole output and pays the first-touch page
  faults that igzip's streaming small-buffer avoids.
- The real igzip T1 advantage is **streaming decode through a small cache-warm reused
  buffer** (igzip's tmp_out double-buffer + incremental write). gzippy's thin-T1 already
  approximates this (1 MiB recycled chunks) and already BEATS rapidgzip −P1.
- The next lever toward igzip parity is therefore NOT a monolith but: (a) the prior
  cycle's gated 2 MiB chunk optimum (recovers 4–10%), (b) shedding the per-chunk
  bookkeeping (boundary record / marker setup / window-clone) FROM the chunked streaming
  driver while keeping its small recycled buffer, and (c) the per-symbol kernel codegen
  residual. The `record_boundaries=false` shed built here is correct and reusable, but
  must be applied to the CHUNKED streaming path, not to a hold-everything monolith.
