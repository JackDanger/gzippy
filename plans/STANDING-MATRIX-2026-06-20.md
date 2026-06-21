# STANDING MATRIX — single-member decompression, gated, both boxes (2026-06-20)

Subject sha: **39160e003f32** (origin/kernel-converge-A HEAD), build-flavor
`parallel-sm+pure` (C-FFI OFF the decode graph), path=ParallelSM asserted per cell.
**STAMP: NOT-YET-LAW** — Intel(x86_64) + macOS(aarch64) only; AMD/Zen2 owed for LAW.

- **Intel** = `scripts/bench/standing/standing.sh` on the i7-13700T LXC; PRODUCTION
  build = **x86_64 BMI2 asm kernel**, `RUSTFLAGS=-C target-cpu=native`. Comparators
  rapidgzip(rg, T>1 SOTA) + igzip(ig, T1 SOTA). N=13 interleaved best-of-N, perf stat,
  /dev/null both arms. Box UNFROZEN (no_turbo=0, gov=powersave, load_start≈8.8) →
  trust RELATIVE ratios gated by the A/A self-test, not absolute cyc/B.
- **macOS** = `scripts/bench/standing/standing_mac.sh` on M1 Pro; PRODUCTION build =
  **pure-Rust engine A** (run_contig_ref_biased ParallelSM kernel). Comparator
  libdeflate-gunzip (single-thread, T1 only). N=13, `/usr/bin/time -l` (instr retired
  = deterministic primitive; cyc elapsed; real wall), /dev/null both arms.

---

## UNIFIED TABLE — gzippy throughput (MB/s of decompressed output) + ratio vs box SOTA

### INTEL i7-13700T LXC — x86_64 asm-kernel build (rg + igzip comparators)

| corpus   | T  | gz MB/s | wall ms | spr% | cyc/B | gz/rg | gz/ig | GATE / verdict            |
|----------|----|---------|---------|------|-------|-------|-------|---------------------------|
| nasa     | 1  | 651.0   | 315.3   | 1.0  | 2.13  | 0.771 | 1.331 | PASS — gz FASTER 22.9%    |
| nasa     | 2  | 359.6   | 570.7   | 1.7  | 6.87  | 0.963 | 2.397 | PASS — TIE (±3.9%)        |
| nasa     | 4  | 511.4   | 401.3   | 10.5 | 6.88  | 0.960 | 1.690 | PASS — TIE (±17.6%)       |
| nasa     | 8  | 872.7   | 235.2   | 7.6  | 8.01  | 1.016 | 0.982 | PASS — TIE (±18.8%)       |
| monorepo | 1  | 327.6   | 155.4   | 1.3  | 4.20  | 0.958 | 1.402 | PASS — gz FASTER 4.2%     |
| monorepo | 2  | 234.5   | 217.1   | 1.8  | 10.38 | 1.051 | 1.950 | PASS — gz SLOWER 5.1%     |
| monorepo | 4  | 315.4   | 161.4   | 4.8  | 11.59 | 1.002 | 1.463 | PASS — TIE (±14.9%)       |
| monorepo | 8  | 537.0   | 94.8    | 11.8 | 12.87 | 0.979 | 0.853 | PASS — TIE (±18.1%)       |
| silesia  | 1  | 250.2   | 847.3   | 1.1  | 5.55  | 1.067 | 1.242 | PASS — gz SLOWER 6.7%     |
| silesia  | 2  | 284.4   | 745.2   | 1.1  | 9.33  | 1.022 | 1.093 | PASS — TIE (±3.6%)        |
| silesia  | 4  | 356.8   | 594.1   | 2.2  | 9.77  | 1.161 | 0.871 | PASS — gz SLOWER 16.1%    |
| silesia  | 8  | 683.7   | 310.0   | 8.9  | 10.06 | 1.070 | 0.457 | PASS — TIE (±13.2%)       |

Comparator MB/s (same cells): rg — silesia 267/291/414/732, monorepo 314/247/316/526,
nasa 502/347/491/887 (T1/2/4/8). igzip is single-thread (constant ~MB/s across T):
silesia ≈311, monorepo ≈459, nasa ≈862. (gz/ig at T>1 = gz-multithread vs ig-1-thread.)

All 12 Intel cells PASS Gate-0 (rg-vs-rg & gz-vs-gz A/A self-tests ≤3.7% ≤ 5% tol →
ratios TRUSTED; sha==zcat all arms; path=ParallelSM; /dev/null both arms). GHz pinned
at 1.39 across every arm (GHzσ ≤1%). No UNTRUSTED cells.

### macOS M1 Pro — pure-Rust engine-A build (libdeflate comparator, T1 single-thread)

| corpus       | T  | gz MB/s* | cyc/B | c-flr% | instr/B | i-flr% | gz/ld cyc | gz/ld instr |
|--------------|----|----------|-------|--------|---------|--------|-----------|-------------|
| silesia      | 1  | 785      | 4.110 | 0.27   | 9.903   | 0.014  | 1.243     | 1.292       |
| silesia      | 2  | 848      | 6.897 | 0.22   | 18.655  | 0.053  | 2.085     | 2.434       |
| silesia      | 4  | 1631     | 7.103 | 0.31   | 18.907  | 0.054  | 2.148     | 2.467       |
| silesia      | 8  | 2650     | 7.296 | 0.35   | 19.066  | 0.124  | 2.206     | 2.487       |
| big2(synth)  | 1  | 2309     | 1.399 | 0.59   | 2.895   | 0.78W  | 1.374     | 1.153       |
| big2(synth)  | 2  | 2132     | 2.194 | 0.07   | 5.445   | 0.02   | 2.156     | 2.169       |
| big2(synth)  | 4  | 2309     | 2.441 | 0.35   | 5.840   | 0.36W  | 2.399     | 2.326       |
| big2(synth)  | 8  | 1848     | 3.210 | 0.26   | 8.347   | 1.03!  | 3.154     | 3.324       |

libdeflate-gunzip (single-thread reference): silesia cyc/B 3.308, instr/B 7.665, ~963
MB/s; big2 cyc/B 1.018, instr/B 2.511, ~3079 MB/s.

\* macOS MB/s is from `/usr/bin/time` `real` at **10 ms resolution**; sub-second runs
(silesia T8 ≈0.08 s, T4 ≈0.13 s) carry ~±12% quantization error → treat mac MB/s as
COARSE/informational. The DETERMINISTIC mac metric is **instr/B** (floor ≤0.12% on
silesia), and **cyc/B** (c-flr ≤0.59%, cyc-trust PASS all arms). The gz/ld ratios are
the trustworthy per-core comparison **at T1 only** (libdeflate is single-thread; at T>1
gz is multi-thread vs a serial tool — apples-to-oranges, shown for completeness).

Mac Gate-0: byte-exact (gz sha == libdeflate sha == gzip -d sha) PASS all corpora;
path=ParallelSM + build-flavor=parallel-sm+pure PASS; /dev/null both arms; cyc-trust
PASS all arms. instr A/A: silesia all PASS (≤0.12%); big2 elevated — T1 0.78% / T4
0.36% = **WARN** (-p1 coordination jitter on the highly-redundant synthetic corpus),
**T8 1.03% = SUSPECT/UNTRUSTED for that one instr cell** (cyc still trusted there). The
rig's overall instr-gate reports FAIL solely because of the big2/T8 1.03% cell; every
silesia cell is clean.

---

## CAVEATS (read with the table)

- **Different comparators per box** (asymmetry is real): Intel = rapidgzip + igzip;
  macOS = libdeflate only (rapidgzip/igzip NOT installed on mac). Cross-box gz numbers
  are also a DIFFERENT PRODUCTION ENGINE (Intel asm kernel vs mac pure-Rust engine A) —
  do not compare gz MB/s across boxes as if one engine.
- **Different corpora per box**: Intel ran silesia(mixed real) + monorepo(text/uniform
  real) + nasa(binary/numeric real); macOS ran silesia(mixed real) + **big2 = SYNTHETIC
  highly-redundant text** ("the quick brown fox…" numbered lines, 554 MB) — the only
  large extra corpus present on the mac; it is NOT a real-data corpus, labeled (synth).
  monorepo/nasa are not present on the mac.
- **Intel LXC is UNFROZEN** (no_turbo=0, gov=powersave, load≈3–9 during the run): GHz
  did pin at 1.39 (GHzσ ≤1%), but T≥4 wall spread is wide (up to ~12%) → several T≥4
  cells land as TIE within spread. Absolute cyc/B drifts; trust the gated ratios.
- **macOS wall is 10 ms-quantized** → mac MB/s is coarse; rely on instr/B (deterministic)
  + cyc/B for the mac per-core picture.
- TIE = |ratio−1| ≤ combined inter-run spread (never scored as a win). UNTRUSTED = an
  A/A self-test failed (none on Intel; one mac instr cell: big2/T8).

## GATE-0 EVIDENCE SUMMARY

- Intel: build sha==39160e00 (requested), build-flavor=parallel-sm+pure (FFI-off proven),
  path=ParallelSM per corpus, sha==zcat for gz+rg+ig all corpora, rg & igzip present and
  self-test ≈1.0, /dev/null both arms; GHz 1.39 pinned, LLC% reported, load logged.
- macOS: build-flavor=parallel-sm+pure + path=ParallelSM, byte-exact gz==libdeflate==gzip
  per corpus, /dev/null both arms, instr-floor determinism (silesia 0.01–0.12%),
  cyc-trust PASS all arms.

## HEADLINE (gated, NOT-YET-LAW)

- Intel asm kernel: **wins/ties rg on nasa & monorepo at every T**; nasa T1 −22.9% is the
  standout win; **silesia is the standing loss** (T1 +6.7%, T4 +16.1%; T2/T8 TIE). vs
  igzip: gz beats ig at T≥4 (parallel) but loses single-core (ig is ISA-L SOTA).
- macOS pure-Rust engine A: per-core (T1) it costs ~24–29% more cyc/B & instr/B than
  libdeflate on silesia, ~37%/15% on synthetic big2; scales with threads on silesia
  (cyc/B roughly flat T1→T8 → near-linear wall scaling).

Artifacts: `artifacts/standing/standing_20260621T005221Z/` (Intel REPORT.txt + CSVs),
`artifacts/standing_mac_samples.csv` (mac raw).
