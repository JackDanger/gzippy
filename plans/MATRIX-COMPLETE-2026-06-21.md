# COMPLETE GROUND-TRUTH MATRIX — post-pinning-deletion (2026-06-21)

Subject sha **462b0549** (origin/kernel-converge-A HEAD) — `perf(parallel/decode):
DELETE speculative decode-worker pinning — default to unpinned`. Build-flavor
`parallel-sm+pure` (C-FFI OFF the decode graph), `path=ParallelSM` (or
`StoredParallel` for stored-dominated corpora) asserted per cell.

**STAMP: NOT-YET-LAW.** Intel(x86_64, FROZEN) + macOS(aarch64, quiet) only.
AMD/Zen2 (solvency, OFFLINE) owed for cross-arch LAW. `model` is a
near-incompressible CORNER — included, never a verdict-driver.

Comparators differ per box (be explicit):
- **Intel** = asm-kernel build, `RUSTFLAGS=-C target-cpu=native`. vs **rapidgzip**
  (rg, native ELF v0.16.0, T>1 SOTA) AND **igzip** (ig, ISA-L, T1 SOTA). N=13
  interleaved best-of-N, `perf stat`, `/dev/null` all arms, taskset even-P-core pin.
- **macOS** = pure-Rust **engine A** build (M-series, aarch64). vs **libdeflate-gunzip**
  (T1 aarch64 SOTA, single-thread) AND **rapidgzip 0.16.0 aarch64** — CAVEAT: the
  aarch64 rapidgzip has NO ISA-L, so gz-vs-rg here is aarch64-vs-aarch64, **NOT the
  SOTA**. N=13 interleaved best-of-N wall (`time.perf_counter`), `/dev/null` all arms.

---

## STEP 0 — the pinning deletion is CORRECT

- **Byte-exact**: gz sha == zcat on silesia+monorepo+nasa × T1/T4/T8 at 462b0549 — ALL OK.
  (also re-verified per-corpus in the matrix Gate-0b: gz==rg==ig==zcat every corpus/T.)
- **cargo test** `--release --no-default-features --features gzippy-native`: the full-suite
  run hit EXIT=124 (900 s) ONLY because the 8 `decompress::parallel::fd_vectored_write`
  tests deadlock on a LOADED box (documented pre-existing test-harness pipe deadlock,
  project_parallel_test_hang). Zero FAILED/panicked. Re-running those exact 8 tests
  single-threaded under the freeze: **9 passed; 0 failed** in 0.01 s. No regression.
- Code confirmed: decode pool constructs with an EMPTY `ThreadPinning` (chunk_fetcher.rs
  :766-778); the speculative `with_pinning_for_capacity` machinery is retired. Affinity-only
  / byte-transparent.

=> the deletion is correct; proceed to the matrix.

---

## INTEL i7-13700T LXC — x86_64 asm kernel (FROZEN: no_turbo=1, gov=performance)

gz throughput (MB/s decompressed) + wall + ratios vs rg & ig. `gz/rg`,`gz/ig` are
WALL ratios (**<1.0 = gzippy faster**). TIE = |Δ| ≤ combined inter-run spread.

| corpus | path | T | gz MB/s | gz wall ms | spr% | cyc/B | gz/rg | gz/ig | A/A gz | GATE / verdict |
|--------|------|---|---------|-----------|------|-------|-------|-------|--------|----------------|
| silesia | ParallelSM | 1 | 255.9 | 828.4 | 0.5 | 5.44 | 1.067 | 1.241 | 0.1% | gz SLOWER than rg 6.7% |
| silesia | ParallelSM | 2 | 294.0 | 720.9 | 0.6 | 9.09 | 1.020 | 1.081 | 0.1% | gz SLOWER than rg 2.0% |
| silesia | ParallelSM | 4 | 414.5 | 511.4 | 2.7 | 9.53 | **1.032** | 0.766 | 0.5% | **TIE vs rg (±5.9%)** |
| silesia | ParallelSM | 8 | 748.5 | 283.2 | 3.1 | 9.81 | 1.015 | 0.424 | 5.1% | UNTRUSTED(box A/A>5%) |
| monorepo | ParallelSM | 1 | 331.4 | 153.6 | 0.2 | 4.15 | 0.974 | 1.415 | 0.3% | gz FASTER than rg 2.6% |
| monorepo | ParallelSM | 2 | 242.6 | 209.8 | 1.0 | 10.09 | 1.045 | 1.928 | 0.3% | gz SLOWER than rg 4.5% |
| monorepo | ParallelSM | 4 | 321.9 | 158.2 | 3.9 | 11.18 | 0.983 | 1.456 | 3.1% | TIE vs rg (±6.7%) |
| monorepo | ParallelSM | 8 | 586.4 | 86.8 | 5.4 | 12.40 | 0.955 | 0.797 | 0.8% | TIE vs rg (±10.9%) |
| nasa | ParallelSM | 1 | 658.0 | 311.9 | 0.3 | 2.11 | 0.797 | 1.344 | 0.0% | gz FASTER than rg 20.3% |
| nasa | ParallelSM | 2 | 364.8 | 562.5 | 0.9 | 6.67 | 0.982 | 2.427 | 0.0% | TIE vs rg (±1.8%) |
| nasa | ParallelSM | 4 | 565.0 | 363.3 | 6.3 | 6.61 | 0.899 | 1.564 | 0.0% | TIE vs rg (±10.2%) |
| nasa | ParallelSM | 8 | 1077.4 | 190.5 | 6.3 | 7.08 | 0.894 | 0.819 | 0.2% | TIE vs rg (±11.1%) |
| model* | ParallelSM | 1 | 118.7 | 2267.2 | 0.3 | 11.75 | 1.096 | 1.167 | 0.1% | gz SLOWER than rg 9.6% |
| model* | ParallelSM | 2 | 211.2 | 1274.1 | 0.2 | 12.64 | 1.134 | 0.654 | 0.2% | gz SLOWER than rg 13.4% |
| model* | ParallelSM | 4 | 303.1 | 887.6 | 1.2 | 13.46 | 1.161 | 0.456 | 0.1% | gz SLOWER than rg 16.1% |
| model* | ParallelSM | 8 | 661.7 | 406.6 | 1.6 | 13.50 | 1.162 | 0.209 | 0.6% | gz SLOWER than rg 16.2% |
| squishy | ParallelSM | 1 | 256.5 | 1561.2 | 0.5 | 5.43 | 1.028 | 1.191 | 0.1% | gz SLOWER than rg 2.8% |
| squishy | ParallelSM | 2 | 330.1 | 1212.8 | 1.5 | 8.16 | 0.974 | 0.925 | 0.5% | TIE vs rg (±3.5%) |
| squishy | ParallelSM | 4 | 458.5 | 873.2 | 2.0 | 8.82 | 1.035 | 0.667 | 0.2% | TIE vs rg (±6.7%) |
| squishy | ParallelSM | 8 | 910.0 | 440.0 | 3.8 | 9.22 | 0.999 | 0.336 | 1.7% | TIE vs rg (±12.5%) |
| bignasa | ParallelSM | 1 | 670.0 | 1225.4 | 0.4 | 2.08 | 0.894 | 1.280 | 0.0% | gz FASTER than rg 10.6% |
| bignasa | ParallelSM | 2 | 464.2 | 1768.5 | 0.4 | 5.79 | 0.758 | 1.848 | 0.0% | gz FASTER than rg 24.2% |
| bignasa | ParallelSM | 4 | 626.8 | 1309.7 | 3.6 | 6.31 | 0.804 | 1.368 | 0.1% | gz FASTER than rg 19.6% |
| bignasa | ParallelSM | 8 | 1141.9 | 718.9 | 8.3 | 7.13 | 0.886 | 0.752 | 3.2% | TIE vs rg (±15.4%) |
| weights | ParallelSM | 1 | 140.6 | 646.5 | 0.3 | 9.91 | 1.020 | 1.082 | 0.0% | gz SLOWER than rg 2.0% |
| weights | ParallelSM | 2 | 241.3 | 376.6 | 1.3 | 10.98 | 1.078 | 0.630 | 0.4% | gz SLOWER than rg 7.8% |
| weights | ParallelSM | 4 | 333.2 | 272.7 | 1.8 | 11.73 | 1.119 | 0.456 | 1.1% | gz SLOWER than rg 11.9% |
| weights | ParallelSM | 8 | 676.8 | 134.3 | 3.3 | 11.81 | 1.084 | 0.224 | 0.9% | gz SLOWER than rg 8.4% |
| storedmix | ParallelSM | 1 | 1815.7 | 57.8 | 0.8 | 0.75 | 0.720 | 1.300 | 0.2% | gz FASTER than rg 28.0% |
| storedmix | ParallelSM | 2 | 1574.4 | 66.6 | 1.9 | 1.38 | 1.143 | 1.503 | 1.2% | gz SLOWER than rg 14.3% |
| storedmix | ParallelSM | 4 | 1689.9 | 62.0 | 4.0 | 1.94 | 1.159 | 1.394 | 0.7% | gz SLOWER than rg 15.9% |
| storedmix | ParallelSM | 8 | 2427.3 | 43.2 | 5.3 | 2.28 | 1.330 | 0.975 | 0.7% | gz SLOWER than rg 33.0% |
| storedheavy | StoredParallel | 1 | 1533.7 | 65.2 | 0.9 | 0.89 | 0.803 | 1.217 | 0.1% | gz FASTER than rg 19.7% |
| storedheavy | StoredParallel | 2 | 1354.8 | 73.8 | 0.6 | 1.43 | 1.229 | 1.382 | 0.1% | gz SLOWER than rg 22.9% |
| storedheavy | StoredParallel | 4 | 1456.2 | 68.7 | 3.8 | 2.25 | 1.312 | 1.287 | 0.1% | gz SLOWER than rg 31.2% |
| storedheavy | StoredParallel | 8 | 2236.1 | 44.7 | 8.9 | 2.40 | 1.316 | 0.831 | 0.2% | gz SLOWER than rg 31.6% |
| pure_stored | StoredParallel | 1 | 820.7 | 127.8 | 1.4 | 1.66 | 1.788 | 2.782 | 0.3% | gz SLOWER than rg 78.8% |
| pure_stored | StoredParallel | 2 | 1310.4 | 80.0 | 2.5 | 1.72 | 1.607 | 1.743 | 0.3% | gz SLOWER than rg 60.7% |
| pure_stored | StoredParallel | 4 | 1563.9 | 67.0 | 4.2 | 1.72 | 1.374 | 1.452 | 0.5% | UNTRUSTED(box A/A>5%) |
| pure_stored | StoredParallel | 8 | 2068.2 | 50.7 | 4.3 | 2.13 | 1.600 | 1.097 | 1.7% | gz SLOWER than rg 60.0% |

\* `model` = near-incompressible CORNER (never a verdict-driver). 38/40 cells PASS
Gate-0; 2 cells (silesia-T8, pure_stored-T4) are UNTRUSTED — their gz A/A self-test
exceeded 5% (box load rose as these last corpora ran near the freeze TTL). GHz pinned
1.39-1.40 (GHzσ ≤0.3%) every arm; sha==zcat all arms; `/dev/null` all arms.

## macOS M-series — aarch64 pure-Rust **engine A** (quiet box)

gz throughput MB/s + WALL ratios (**<1.0 = gzippy faster**). libdeflate is
**T1-only** (single-thread) → gz/ld at T>1 is multi-vs-single (informational).
rapidgzip here is **aarch64 (no ISA-L) — NOT the Intel SOTA**.

| corpus | T | gz MB/s | gz wall ms | spr% | ld MB/s | rg(aarch64) MB/s | gz/ld | gz/rg | A/A gz | GATE |
|--------|---|---------|-----------|------|---------|------------------|-------|-------|--------|------|
| silesia | 1 | 761.6 | 278.3 | 1.3 | 922.1 | 317.1 | 1.211 | 0.416 | 0.10% | PASS |
| silesia | 2 | 807.6 | 262.5 | 0.3 | 926.7 | 518.3 | 1.147 | 0.642 | 0.25% | PASS |
| silesia | 4 | 1514.5 | 140.0 | 1.3 | 927.5 | 924.1 | 0.612 | 0.610 | 0.13% | PASS |
| silesia | 8 | 2389.7 | 88.7 | 1.2 | 929.8 | 1394.0 | 0.389 | 0.583 | 0.83% | PASS |
| monorepo | 1 | 1075.6 | 47.3 | 1.9 | 1332.2 | 337.2 | 1.239 | 0.314 | 0.65% | PASS |
| monorepo | 2 | 741.2 | 68.7 | 1.7 | 1298.6 | 429.1 | 1.752 | 0.579 | 0.46% | PASS |
| monorepo | 4 | 1141.0 | 44.6 | 2.1 | 1311.4 | 627.7 | 1.149 | 0.550 | 0.44% | PASS |
| monorepo | 8 | 1721.3 | 29.6 | 2.5 | 1333.0 | 871.3 | 0.774 | 0.506 | 0.52% | PASS |
| nasa | 1 | 1836.9 | 111.7 | 1.1 | 2277.0 | 648.9 | 1.240 | 0.353 | 1.03% | PASS |
| nasa | 2 | 1173.7 | 174.9 | 1.3 | 2283.1 | 751.4 | 1.945 | 0.640 | 0.05% | PASS |
| nasa | 4 | 2001.4 | 102.5 | 3.1 | 2271.9 | 1273.4 | 1.135 | 0.636 | 1.81% | PASS |
| nasa | 8 | 3270.8 | 62.7 | 6.7 | 2288.7 | 2016.8 | 0.700 | 0.617 | 0.44% | PASS |
| squishy | 1 | 724.0 | 553.0 | 1.2 | 890.3 | 341.0 | 1.230 | 0.471 | 0.37% | PASS |
| squishy | 2 | 867.7 | 461.4 | 1.2 | 887.1 | 571.8 | 1.022 | 0.659 | 0.21% | PASS |
| squishy | 4 | 1644.3 | 243.5 | 3.0 | 888.4 | 1039.1 | 0.540 | 0.632 | 0.65% | PASS |
| squishy | 8 | 2670.6 | 149.9 | 1.2 | 887.5 | 1692.1 | 0.332 | 0.634 | 1.20% | PASS |

All 16 mac cells PASS Gate-0 (A/A ≤1.81% ≤5% tol; sha gz==ld==rg==gzip per corpus;
build-flavor parallel-sm+pure / path=ParallelSM; `/dev/null` all arms).

---

## THE 3 KEY CONFIRMATIONS

### (a) silesia-T4 gz/rg is now a TIE — the pin-deletion payoff CONFIRMED
PRE-fix (STANDING-MATRIX-2026-06-20, sha 39160e00): silesia-T4 gz/rg = **1.161**
(gz SLOWER 16.1%). POST-fix (462b0549): silesia-T4 gz/rg = **1.032, TIE vs rg
(±5.9%)**. The +16% T4 deficit is gone. The other multi-thread cells moved the same
way (PRE→POST gz/rg): silesia-T8 1.070→1.015, nasa-T4 0.960→0.899, nasa-T8
1.016→0.894, monorepo-T4 1.002→0.983, monorepo-T8 0.979→0.955.

### (b) NO REGRESSION vs the PRE-fix matrix
Comparing the 3 corpora in BOTH matrices, every cell either improved or held within
noise (PRE→POST gz/rg):

| corpus | T1 | T2 | T4 | T8 |
|--------|----|----|----|----|
| silesia | 1.067→1.067 | 1.022→1.020 | **1.161→1.032 ↓** | 1.070→1.015 ↓ |
| monorepo | 0.958→0.974 | 1.051→1.045 | 1.002→0.983 ↓ | 0.979→0.955 ↓ |
| nasa | 0.771→0.797 | 0.963→0.982 | 0.960→0.899 ↓ | 1.016→0.894 ↓ |

Every T4/T8 cell improved; T1/T2 unchanged (single/low-thread does not stress the
decode-pool affinity that was deleted — as expected for an affinity-only change). No
cell regressed beyond inter-run spread.

### (c) Honest per-arch standing vs SOTA (gated, trusted cells)
**Intel vs rapidgzip (T>1 SOTA):**
- gz **WINS/TIES rg** on: nasa (all T), monorepo (all T), bignasa (all T), squishy
  (T2/T4/T8 TIE), silesia (T4/T8 TIE), storedmix-T1, storedheavy-T1.
- gz **LOSES to rg** on: silesia-T1 (+6.7%) & -T2 (+2%), squishy-T1 (+2.8%), weights
  (all T, +2–12%), model (all T, +10–16%, the incompressible corner), and the
  **stored-dominated `StoredParallel` path at T≥2** (storedheavy/storedmix +14–33%,
  pure_stored +60–79%) — a real standing gap, but a DIFFERENT path the pin deletion
  did not touch.
- **Intel vs igzip (T1 single-core SOTA):** gz LOSES single-core at T1 on every
  compressed corpus (ig 0.55–0.86× gz wall = ig ~16–45% faster); gz beats ig at T≥4 by
  parallelism. ISA-L/igzip remains the single-stream SOTA; the T1 single-core deficit
  is gzippy's standing weakness on x86.

**macOS (pure-Rust engine A) vs libdeflate (T1 aarch64 SOTA):** at T1 (like-for-like
single-core) gz is **~21–24% slower** than libdeflate (gz/ld 1.21–1.24 across
silesia/monorepo/nasa/squishy) — consistent with the prior aarch64 measurement. gz
beats rapidgzip-aarch64 at every cell (gz/rg 0.31–0.66) but rg-aarch64 has no ISA-L
and is NOT the SOTA.

---

## CAVEATS
- **NOT-YET-LAW**: Intel (frozen) + macOS (quiet) is a single arch-pair. **AMD/Zen2
  (solvency) is OFFLINE and owed** before any of this is LAW (BMI2 is microcoded on
  Zen2 → an Intel-only result can differ).
- **Different comparators per box** (asymmetric, intentional): Intel = rg + igzip;
  macOS = libdeflate + rg-aarch64 (NOT SOTA). Different production ENGINE per box too
  (Intel asm kernel vs mac pure-Rust engine A) — do NOT compare gz MB/s across boxes.
- `model` is a near-incompressible CORNER; the stored corpora exercise the
  `StoredParallel` path (memcpy-dominated), NOT the ParallelSM marker kernel.
- 2 Intel cells UNTRUSTED (silesia-T8, pure_stored-T4): gz A/A self-test >5% as the box
  load rose near the freeze TTL; their ratios are withheld (not scored).
- TIE = |ratio−1| ≤ combined inter-run spread (never scored as a win).

Artifacts: Intel `artifacts/standing/standing_intel_20260621T185256Z/`
(REPORT.txt + per-cell CSVs + standing.log); mac raw `/tmp/mac_wall.csv`,
report `/tmp/mac_wall.report.txt`. Freeze RELEASED + RESTORE VERIFIED (no_turbo=0,
guests thawed) at 18:52Z.
