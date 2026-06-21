# PARALLEL-SCALING DISENTANGLEMENT — why low-T gzippy loses to igzip-1-thread (2026-06-20)

**STAMP: NOT-YET-LAW.** macOS-aarch64 (M1 Pro, quiet) + Intel i7-13700T LXC
(load ~2.5–4, relatively quiet, UNFROZEN). AMD/Zen2 + a FROZEN box owed for LAW
and for the precise igzip-vs-gz per-T crossover. igzip/rapidgzip absent on the mac;
the full 3-tool quiet per-T answer is AMD/frozen-box-blocked — noted, not faked.

Subject: origin/kernel-converge-A. Mac binary build-flavor=parallel-sm+pure (FFI off,
pure-Rust engine A), sha d20bc955…. Intel binary /root/gzippy @6dc67b1a same line,
build-flavor=parallel-sm+pure (pure-Rust → apples-to-apples with the mac engine).

Harness: `scripts/bench/standing/parallel_scaling_mac.py` (mac) +
`scripts/bench/standing/parallel_scaling_intel.py` (guest). Wall = `time.perf_counter()` around each subprocess (microsecond
resolution — NOT the 10 ms `real` clock), interleaved best-of-N=11, report
min(floor)+median+inter-run spread. CPU = `/usr/bin/time -l` instr/cyc per
decompressed byte. /dev/null BOTH arms. Startup (process+pool spawn) measured via a
near-empty .gz at each -pN and subtracted (≈3–4 ms, negligible vs 200–820 ms decodes).

## GATE-0 (passed)
- build-flavor=parallel-sm+pure, path=ParallelSM asserted (both boxes).
- byte-exact: gz sha == `gzip -d` sha for every corpus, every box.
- /dev/null both arms.
- GZIPPY_CHUNK_KIB perturbation PROVEN NON-INERT via "Total Fetched" chunk count:
  silesia3 T1@1MiB=196, T1@4MiB=49, T2@1MiB=196, T2@4MiB=49 (forcing the size moves
  the chunk count to match the other thread's default → the knob bites, both directions).
- inter-run spread reported per cell; Δ<spread ⇒ TIE.

---

## 1. CLEAN SCALING CURVE (quiet mac, best-of-11, startup-subtracted decode-only)

| corpus (clean cyc/B) | T | wall ms | MB/s | speedup | cyc/B | spr% | monotonic? |
|----------------------|---|---------|------|---------|-------|------|------------|
| silesia3 real (4.16) | 1 | 821 | 775 | 1.00x | 4.16 | 0.9 | |
| silesia3 real        | 2 | 736 | 864 | 1.11x | 6.84 | 2.0 | YES T2>T1 |
| silesia3 real        | 4 | 395 | 1609 | 2.08x | 7.08 | 0.7 | YES |
| silesia3 real        | 8 | 224 | 2842 | 3.67x | 7.25 | 7.4 | YES |
| big2 SYNTH (1.38)    | 1 | 237 | 2337 | 1.00x | 1.38 | 3.4 | |
| big2 SYNTH           | 2 | 254 | 2179 | **0.93x** | 2.21 | 5.2 | **NO T2<T1** |
| big2 SYNTH           | 4 | 229 | 2421 | 1.04x | 2.38 | 5.8 | (recovers) |
| big2 SYNTH           | 8 | 300 | 1848 | **0.79x** | 3.15 | 4.7 | **NO T8<T1** |

**On a QUIET box, silesia (expensive clean decode) scales MONOTONICALLY (T2>T1>…).
big2 (cheap clean decode) does NOT — T2<T1 and T8<T1 reproduce on the quiet mac.**
So T2<T1 is NOT a contention artifact; it is corpus-dependent and present uncontended.

The discriminator is the CLEAN per-byte decode cost: silesia3 4.16 cyc/B vs big2
1.38 cyc/B. cyc/B jumps once at T1→T2 (the pipeline tax) then is ~flat T2→T8.

## 2. CHUNK-FIXED A/B — isolate the T1-special-path (cause 2) — quiet mac, N=11

| config (silesia3) | chunks | wall ms | MB/s | cyc/B | spr% |
|-------------------|--------|---------|------|-------|------|
| T1 default(1MiB)  | 196 | 826 | 770 | 4.162 | 1.6 |
| T1 @1MiB          | 196 | 829 | 767 | 4.166 | 0.8 |
| T1 @4MiB          |  49 | 824 | 772 | 4.147 | 0.9 |
| T2 default(4MiB)  |  49 | 738 | 861 | 6.844 | 1.6 |
| T2 @4MiB          |  49 | 736 | 864 | 6.850 | 1.0 |
| T2 @1MiB          | 196 |1101 | 577 | 9.494 | 0.5 |

- **T1 is chunk-size INSENSITIVE: 1MiB 4.166 vs 4MiB 4.147 cyc/B, Δ=−0.018** (≪ spread).
  At T1 the pipeline is inline depth-1, so chunk size only changes the output-buffer
  size, not work-per-byte. ⇒ **the T1-special-path (1 MiB) contributes ~0 to T2<T1.**
- **Forcing EQUAL chunk size does NOT remove the T2 overhead:** at identical 4 MiB,
  T2 still pays +2.70 cyc/B over T1; at identical 1 MiB it pays +5.33. The default
  (T1=1MiB, T2=4MiB) actually hands T2 its BEST granularity — 1 MiB at T2 is +38%
  wall. So the chunk-size discontinuity, if anything, HELPS T2, not hurts it.

## 3. PIPELINE FIXED-OVERHEAD (cause 1) — the inherent parallel-decode tax

T2-pipeline minus T1-inline at the SAME chunk size (deterministic cyc/B primitive):

| chunk size | T1 cyc/B | T2 cyc/B | Δ (pipeline tax) |
|------------|----------|----------|------------------|
| 4 MiB (prod)| 4.147 | 6.850 | **+2.70 cyc/B** |
| 1 MiB       | 4.166 | 9.494 | **+5.33 cyc/B** |

The pipeline (block-find / worker dispatch / prefetch / marker / window-apply) adds a
FIXED per-byte CPU tax of ~+2.70 cyc/B at the production 4 MiB granularity (blows up to
+5.33 at 1 MiB — more/smaller chunks = more per-chunk fixed cost per byte). This is the
inherent parallel-decode tax rapidgzip also pays.

**Why this makes T2<T1 on CHEAP corpora:** wall ≈ total_cyc/B ÷ cores. T2 doubles the
cores but adds the fixed tax. When clean cost is small the tax dominates:
- nasa-like (clean ~2.1 cyc/B): T1 ∝ 2.1/1=2.1; T2 ∝ (2.1+~4.7)/2=3.4 → T2 ~1.6× SLOWER.
- silesia (clean ~4.2–5.5): T1 ∝ 4.2; T2 ∝ (4.2+2.7)/2=3.4 → T2 ~1.2× FASTER.
The standing-matrix Intel cyc/B corroborate exactly: nasa T1 2.13→T2 6.87 (tax +4.74),
silesia T1 5.55→T2 9.33 (tax +3.78). The tax is ~flat T2→T8, so T4/T8 amortize it and
recover (nasa T8 1.66×, silesia T8 2.49–3.67×).

## 4. INTEL QUIET-WINDOW CURVE (load ~2.5–4 ≪ standing-matrix ~8.8; best-of-11)

| corpus (cheap/expensive) | T1 | T2 | T4 | T8 |
|--------------------------|----|----|----|----|
| nasa MB/s (cheap)        | 636 (1.00x) | 363 (**0.57x**) | 627 (0.99x) | 1058 (1.66x) |
| monorepo MB/s (cheap)    | 326 (1.00x) | 236 (**0.72x**) | 356 (1.09x) | 530 (1.62x) |
| silesia MB/s (expensive) | 254 (1.00x) | 289 (1.14x) | 452 (1.78x) | 633 (2.49x) |

**T2<T1 REPRODUCES on the relatively-quiet Intel (load ~2.5–4) for the cheap corpora
(nasa 0.57×, monorepo 0.72×) and silesia scales monotonically** — the SAME pattern as
the mac, and matching the standing matrix (nasa 651→360, monorepo 328→235, silesia
250→284) that was taken at load ~8.8. Since the dip survives at 1/3 the load, contention
is not its driver; contention only WIDENS the T≥4 spread (here ≤9% vs standing-matrix
12–18%).

---

## DISENTANGLED VERDICT (gated, cross-arch mac+Intel, NOT-YET-LAW)

The "igzip-1-thread beats gzippy-multi-thread at T1/T2/T4, and gzippy T2<T1" observation
decomposes as:

1. **Inherent pipeline fixed-overhead — DOMINANT.** +2.70 cyc/B (mac silesia, 4 MiB),
   corroborated by Intel cyc/B (+3.8–4.7). On corpora whose clean decode is cheap
   (nasa/monorepo/big2) this fixed tax exceeds the 1→2-core saving, so T2 wall > T1 wall;
   on expensive corpora (silesia) it amortizes and T2>T1. Reproduces on BOTH a quiet mac
   and a quiet Intel ⇒ this is the cause of T2<T1. igzip-1-thread is a SOTA serial kernel
   that pays ZERO pipeline tax, so at low T (before enough cores amortize the tax) gzippy
   loses to it; gzippy overtakes at T4/T8.

2. **T1-special-path discontinuity (1 MiB inline vs 4 MiB pipeline) — ~0.** T1 cyc/B is
   chunk-size-insensitive (Δ=−0.018), and forcing equal chunk sizes does not remove the
   regression. The 1 MiB-at-T1 default is not the lever.

3. **Intel LXC contention — NOT the driver, only a variance inflator.** The T2<T1 dip
   reproduces uncontended on both boxes; load only widens the T≥4 spread.

**AMD-owed (for LAW + the full picture):** AMD/Zen2 replication (BMI2 microcode caveat);
a FROZEN/bare-metal box to resolve absolute cyc/B and the exact igzip-vs-gz per-T
crossover; igzip/rapidgzip on a quiet box for the 3-tool low-T comparison (absent on mac,
Intel unfrozen). The pipeline-tax MECHANISM is cross-ISA (mac aarch64 + Intel x86) → STRONG.

**Actionable consequence (HYPOTHESIS, not acted on here):** the low-T loss is the
fixed per-chunk pipeline cost, not the kernel and not chunk-sizing. Reducing the
per-chunk fixed overhead (block-find/marker/window setup amortized over fewer chunks at
low T) is the only lever that would move the T2/T4 cheap-corpus dip — to be tested with a
Gate-2 perturbation, not assumed.
