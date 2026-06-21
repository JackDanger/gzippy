# PIPELINE-TAX-LOCALIZE — which component carries gz's 1.5-2.3× tax differential

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz git:** 04ef912d
**gz bin sha:** d20bc9558660…  **Box:** macOS Apple-Silicon (M1 Pro), quiet.
**Stamp:** NOT-YET-LAW (single-arch aarch64). x86/AMD owed for LAW.
**Builds:** `cargo build --release --no-default-features --features gzippy-native`
(FFI off, build-flavor=parallel-sm+pure, path=ParallelSM at p1 & p4 — Gate-0(b) asserted).

## What this answers
#2 (MAC-RG-GAP-2026-06-21) found gz's parallel pipeline inflates instructions
T1→T by 1.8-4.1× vs rapidgzip's 1.2-1.8× = a 1.5-2.3× PIPELINE-TAX DIFFERENTIAL
(arch-independent CONVERGENCE gap). THIS run LOCALIZES which gz pipeline component
carries it, and maps it to rapidgzip's leaner source (the port target → see
`plans/PIPELINE-PORT-DESIGN.md`).

## Method (Gate-0 self-validated)
- **Primitive:** `/usr/bin/time -l` "instructions retired" (deterministic ~0.04% warm).
  Component cost = instruction delta of a byte-exact / correct-output REMOVAL ORACLE
  vs the full pipeline (Gate-2 causal where a removal exists; attribution otherwise).
- **Rig (committed):** `scripts/bench/standing/mac_pipeline_components.sh` +
  `mac_pipeline_components_report.py`. Interleaved best-of-N=7, /dev/null sink BOTH
  arms, byte-exact sha gate per arm/corpus, A/A spread reported.
- **Arms (all byte-EXACT vs gzip -d; nocrc4 exempt):**
  - `base1` = -p1 (clean in-order decode, NO markers) = the no-tax floor
  - `base4` = -p4 (full speculative marker pipeline)
  - `noprefetch4` = -p4 `GZIPPY_NO_PREFETCH=1` (no speculation → on-demand CLEAN
    decode, no markers). **base4 − noprefetch4 = CAUSAL marker-machinery tax.**
    (Parallelism is ~instruction-neutral — splitting the same work over threads
    does not change total retired instructions — so the delta is the marker work,
    not the thread split. Confirmed: thin1 ≈ noprefetch4 ≈ base1, see below.)
  - `thin1` = -p4 `GZIPPY_THIN_T1_ORACLE=1` (clean SERIAL one-pass) — cross-checks base1.
  - `nocrc4` = -p4 `GZIPPY_FOLD_NOCRC=1` — DIAGNOSTIC only; NON-ISOLATING (it also
    skips the drain / changes the post-path → delta came out NEGATIVE, i.e. heavier;
    discarded). CRC is not a tax component anyway (it runs at T1 too).
- **Non-inert proof:** base1/thin1 print `marker_chunks=0` (clean serial); base4
  routes 97-100% of body through the marker loop (`GZIPPY_VERBOSE`); NO_PREFETCH
  consumer at chunk_fetcher.rs:1461 gates speculation off (instr collapses to ~base1).
- **Reconciliation (validates the rig):** base1 instr/byte == the #2 report's T1
  numbers to 2 d.p. — silesia 9.92≈9.90, monorepo 6.84, nasa 3.42, squishy 9.53;
  base4/base1 inflation == report infl (silesia 1.91, monorepo 3.15, nasa 3.94,
  squishy 1.80). Same machine, same finding, independently re-derived.

## D1 — THE COMPONENT BREAKDOWN

### (1) CAUSAL: 91-97% of the tax is the speculative window-absent MARKER MACHINERY
Removing speculation (NO_PREFETCH → clean on-demand decode) collapses base4 back to
~base1. N=7, instructions retired (M = ×1e6), sig/spread = signal ÷ A/A spread.

```
corpus    base1   base4   noPF4   thin1 |  tax    mkrMach  mkr/tax infl  sig/spread
silesia  2103.2  4017.0  2196.9  2181.7 | 1913.8  1820.1   95.1%  1.91     13×
monorepo  348.3  1097.6   371.2   361.3 |  749.4   726.5   96.9%  3.15     20×
nasa      701.2  2760.6   802.3   792.2 | 2059.5  1958.3   95.1%  3.94     33×
squishy  3816.6  6856.9  4089.9  4076.8 | 3040.3  2767.0   91.0%  1.80     61×
```
- `tax` = base4 − base1 (the T1→T instruction inflation).
- `mkrMach` = base4 − noprefetch4 (CAUSAL: the speculation→u16-marker machinery).
- The **residual coordination** (tax − mkrMach) is SMALL: silesia 94M, monorepo 23M,
  nasa 101M, squishy 273M = prefetch + block-find + dispatch + consumer. Not the lever.
- thin1 ≈ noprefetch4 ≈ base1 every corpus → confirms base1 is the genuine clean floor
  and parallelism adds ~no instructions; the tax is the MARKER WORK, not the threads.
- **VERDICT (Gate-2 causal, sig/spread 13-61×):** the pipeline tax IS the speculative
  window-absent marker machinery (decode-into-u16-markers + apply_window/resolve +
  narrow). Coordination/prefetch/block-find is ≤9% of the tax.

### (2) ATTRIBUTION: within the marker machinery, the marker-DECODE loop dominates
gz self-time (exclusive, conservation-respecting) from a `GZIPPY_TIMELINE` trace
(`scripts`-side `selftime.py`; TIME, instrumented → attribution-tier, not causal):

```
corpus   block_body(marker decode)  apply_window  scan(block-find)  header   [rest=waits]
silesia        63.9%                    5.0%           2.6%           2.5%
monorepo       52.3%                    9.1%           3.9%           0.7%
nasa           46.3%                   17.9%           1.7%           0.4%
squishy        65.0%                    4.8%           2.8%           2.9%
```
- The marker **decode loop** (`worker.block_body` = decode_marker_fast_loop) is the
  dominant work component everywhere; `apply_window` (resolve+narrow) is secondary
  and highest on nasa (the ~100%-marker corpus); block-finding/scan is minor.
- (An earlier raw-sum read showed `scan_candidate`≈100ms — that was the NESTED
  successful trial-decode double-counting `block_body`; exclusive self-time corrects
  it: real scan overhead is ~3-9ms. Block-finding is NOT the heavy component.)

### (3) THE LEAN REFERENCE: rapidgzip --verbose phase breakdown (T4, CPU-seconds)
```
corpus   block-find  decode(custom-inflate)  alloc/copy  apply-window  checksum  marker%
silesia   0.0038        0.653 (84.6%)          0.021       0.034 (4.4%)  0.061    34.5%
monorepo  0.0035        0.144 (74.2%)          0.0085      0.0226(11.6%) 0.0149   80.9%
nasa      0.0038        0.272 (60.5%)          0.031       0.086 (19.1%) 0.060    89.8%
squishy   0.010         1.156 (84.8%)          0.034       0.060 (4.4%)  0.114    31.6%
```
- rg is ALSO decode-dominated (custom-inflate 60-85%); apply-window 4-19% (highest on
  nasa); block-finder ~0.5-1.8%. **Same component SHAPE as gz** → the differential is
  per-symbol MAGNITUDE inside the marker path, not a different component distribution.

### (4) ARCHIVE-TYPE VARIATION — the tax tracks the MARKER FRACTION
rg `marker%` (replaced marker symbols) correlates 1:1 with gz's tax infl:
nasa 89.8% mkr → infl 3.94 (worst); monorepo 80.9% → 3.15; silesia 34.5% → 1.91;
squishy 31.6% → 1.80 (mildest). gz marks even MORE: monorepo 97.1%, nasa ~100.0%
(`GZIPPY_VERBOSE` clean_flipped 2.9% / 0.0%). **The differential is worst where the
stream relies most on cross-window backrefs (high-ratio / long-match corpora).** The
detectable archive-type signal is the marker fraction ≈ compression ratio / backref
distance — see the port doc for whether a gate is needed.

### Absolute-vs-ratio HONESTY STAMP (do not over-read the "differential")
On aarch64 gz is LIGHTER in ABSOLUTE instructions than rg at EVERY cell (#2: gz
silesia-T4 19.0 vs rg 32.2 instr/B). The "1.5-2.3× tax differential" is a
RATIO-OF-RATIOS: gz's very cheap CLEAN T1 (3.4-9.9 instr/B) scales up more steeply
because its marker machinery ADDS 6.9-14.3 instr/B on top. So the actionable,
arch-independent lever is **reducing the marker machinery's ABSOLUTE added
instructions** (helps every arch, and is exactly the term that would close the x86
gap where rg's ISA-L makes rg's absolute lower). It is NOT "gz's pipeline does more
total work than rg's" — on aarch64 it does less.

## Evidence tier
- mkrMach = THE tax: **CAUSAL** (byte-exact removal oracle, sig/spread 13-61×, N=7).
- decode-loop dominates within mkrMach: **ATTRIBUTION** (self-time + rg --verbose,
  instrumented/perturbed, single-arch aarch64). Needs a marker-decode-isolating oracle
  to become causal (specified in the port doc).

## Owed (NOT-YET-LAW)
- x86 (rg ISA-L) + AMD/Zen2 replication of the component ranking.
- A Gate-2 marker-decode-ONLY removal oracle to causally split "gz marks more symbols"
  vs "gz's per-marker decode is heavier" (PIPELINE-PORT-DESIGN §next-instrument).

## Reproduce
```
scripts/bench/standing/mac_pipeline_components.sh           # N=7, 4 corpora, instr deltas
GZIPPY_TIMELINE=/tmp/tl.json target/release/gzippy -dc -p4 /tmp/<c>.gz >/dev/null
python3 scripts/bench/standing/selftime.py /tmp/tl.json     # exclusive self-time split
/tmp/rgvenv/bin/rapidgzip -dc -P4 --verbose /tmp/<c>.gz >/dev/null   # rg phase reference
```
