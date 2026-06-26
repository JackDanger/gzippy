# CRC KERNEL — VPCLMULQDQ 256-bit fold (DOMINANCE lever, Intel) — 2026-06-26

**VERDICT (a): DOMINANCE LEVER on Intel x86.** gz's CRC kernel (`crc32fast` crate,
128-bit SSE fold-by-4) is replaced on the x86_64 production decode path by a pure-Rust
**256-bit VEX-VPCLMULQDQ fold-by-4 (128 B/iter)** — a faithful steal of libdeflate's
wider-fold technique. Byte-EXACT (949 lib tests + the `crc32_fold` differential through
the new kernel + 5-corpus sha gate, BOTH arms). Wall: **+6.0% on logs-T1 (contention-
invariant CERTIFIED)**, +3.6% nasa (CERTIFIED), +1.6% silesia, +1.9% monorepo; instr/B
RESOLVED-reduced on ALL four. Moves logs-T1 from ~1.111× → **1.048× igzip**; gz now
BEATS igzip on silesia/nasa/monorepo. Committed `feat/crc-vpclmul`, **NOT merged** —
NOT-YET-LAW (AMD + aarch64 owed).

## STAGE 1 — GO/NO-GO microbench (the premise gate) — GO
`/dev/shm/crcbench` (standalone, `target-cpu=native`), 3 real gzip-CRC kernels over the
SAME buffer, GATE-0 = all produce the identical CRC32. crc32fast 1.5.0's x86 path is a
128-bit SSE fold-by-4 (64 B/iter); the box (i7-13700T raptorlake) has VPCLMULQDQ+VAES
but NO AVX-512.

GB/s (best-of-15, taskset core 0):

| buffer | crc32fast | isal `crc32_gzip_refl` | libdeflate | isal/cf | libdeflate/cf |
|---|---:|---:|---:|---:|---:|
| 32 KiB (L1) | 10.96 | 11.04 | 21.90 | 1.007 | **1.998** |
| 256 KiB (L2) | 10.98 | 11.05 | 21.98 | 1.006 | **2.001** |
| 1 MiB | 11.00 | 11.02 | 22.04 | 1.002 | **2.004** |
| 4 MiB | 9.55 | 10.62 | 14.02 | 1.112 | 1.469 |
| 221 MiB (RAM) | 8.05 | 8.80 | 10.57 | 1.094 | 1.313 |

**The decisive regime is cache-resident (≤1 MiB)** — gz CRCs per-deflate-block on hot
data (the CRC-FOLD-2026-06-26 read-split proved the second touch is ≤0.5%, i.e. hot).
There, **libdeflate is a clean 2.0×** crc32fast; **ISA-L is TIED** with crc32fast
(igzip-centric thinking would have missed this — the STACKING-AUDIT's "libdeflate is
leanest" reframe is corroborated). At 221 MiB everything is memory-BW-bound (mission's
literal "221 MB buffer" understates the lever by 1.5×). GO: real ~2× throughput gap.

## STAGE 1b — pure-Rust kernel reaches the ceiling (built the tool, then gated it)
A Rust `crc32_vpclmul` using `_mm256_clmulepi64_epi128` (256-bit, 4 YMM accumulators,
128 B/iter) was added to the microbench. GATE-0: byte-exact vs crc32fast across
sizes×alignments×seeds. Throughput: **2.0× crc32fast on hot data (21.4–21.8 GB/s ≈
libdeflate's 22)**, 1.41× at 4 MiB, 1.26× at 221 MiB. So the portable Rust kernel hits
libdeflate's wall — no FFI needed.

Fold constants are NOT magic numbers: derived at COMPILE TIME from gz's own reflected
GF(2) routine (`polynomial_multiply_modulo`/`x_power_modulo`) and `const`-asserted to
reproduce crc32fast's published keys (K1_64=x^544, K2_64=x^480, K3=x^160, K4=x^96,
K5=x^64) before yielding the new 128 B-stride keys (K1_128=x^1056, K2_128=x^992). The
128→64 + Barrett tail is verbatim from crc32fast 1.5.0. cursor-agent (gpt-5.5) drafted
the kernel; its first constant derivation was WRONG and the compile-time assert CAUGHT
it — the convention (`key(e)=x_pow_mod(e)<<1`, high/low at x^(8S∓32)) was then pinned
empirically (`/tmp/findkeys.py`).

## STAGE 2 — production integration (byte-exact, BLOCKING) — PASS
`src/decompress/parallel/crc32.rs`: new `mod vpclmul` (the kernel) + an x86_64 branch in
`crc32_fold` (the single source for `CRC32Calculator::update`, the T1 ParallelSM CRC
path). Runtime-dispatched on `is_x86_feature_detected!("vpclmulqdq"/"avx2"/…)`;
crc32fast remains the fallback for short inputs (<128 B), non-VPCLMUL x86, and other
arches. `GZIPPY_CRC_LEGACY=1` forces crc32fast (now wired on x86 too) so the A/B runs on
ONE binary. Build: `--no-default-features --features pure-rust-inflate`,
`RUSTFLAGS=-C target-cpu=native`, path=ParallelSM verified.

Byte-exact gate (ALL PASS):
- 949 lib tests + 14 crc32 tests; `crc32_fold_matches_crc32fast_all_sizes_seeds_alignments`
  now routes THROUGH the new x86 kernel and passes.
- 5-corpus sha (logs, logsbig, silesia, nasa, monorepo): vpclmul arm == legacy arm ==
  `gzip -dc`, path=ParallelSM.

## STAGE 3 — cross-corpus paired wall gate (`fulcrum abmeasure`, N=15, core 0, /dev/null, sha)
base = `GZIPPY_CRC_LEGACY=1` (crc32fast), after = default (vpclmul), SAME binary.

| corpus | base cyc/B | after cyc/B | wall after/base | paired | instr/B base→after | after/igzip | cyc/B verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **logs** | 1.137 | 1.069 | **+6.0%** | 15+/0− p=6.1e-5 | 3.57→3.42 | **1.048** (was ~1.111) | **CONTENTION-INVARIANT CERTIFIED** |
| **nasa** | 1.519 | 1.465 | **+3.6%** | 14+/1− p=9.8e-4 | 3.85→3.71 | 0.897 | **CONTENTION-INVARIANT CERTIFIED** |
| silesia | 4.506 | 4.444 | +1.6% | 15+/0− p=6.1e-5 | 12.34→12.19 | 0.978 | VOID-QUIET (run-queue gate; effect small vs load) |
| monorepo | 2.970 | 2.913 | +1.9% | 15+/0− p=6.1e-5 | 7.55→7.38 | 0.958 | VOID-QUIET (small effect) |

- instr/B RESOLVED-reduced on ALL four (load-immune): the wider fold does strictly fewer
  instructions per CRC — the certain, contention-proof core of the win.
- logs + nasa CERTIFIED contention-invariant (FLAT across run-queue strata, A/A
  symmetric) despite a contended box (two long `find`s held run-queue 3–6). silesia/
  monorepo VOIDed ONLY on fulcrum's quietness gate, NOT a regression — their paired
  wall ratios (15+/0−, 14+/1−) consistently favor vpclmul.
- gz now BEATS igzip on silesia/nasa/monorepo; logs improved to 1.048× igzip.

## Does it flip logs-T1 to a WIN vs igzip? — NO (alone); CLOSE (stacked, HYPOTHESIS)
The CRC-OFF removal-oracle (CRC-FOLD-2026-06-26) bounded full CRC removal at +11.6%
→ 0.986× igzip. The 2× kernel captures ~half the CRC compute → measured +6.0% → **1.048×
igzip** (consistent: 1.111×0.945≈1.05). It does NOT by itself beat igzip on logs.
HYPOTHESIS (unvalidated, NOT measured together): stacking the table-build cache
(PR #134, +2.5%) → ~1.048×0.975 ≈ **1.022× igzip** — very close but still a slight loss;
a clean logs-T1 flip likely needs the copy lever (STACKING-AUDIT #1, CEILING −8.9%) too.
Report honestly: this is a large DOMINANCE step + an outright cross-corpus win, NOT a
standalone logs-T1 flip-to-beat-igzip.

## TIER / SCOPE / OWED
- STRONG/gated (Intel i7-13700T, load-immune): instr/B reduction all corpora; logs +6.0%
  & nasa +3.6% contention-invariant CERTIFIED wall wins; Stage-1 2× throughput;
  byte-exactness (tests + 5-corpus sha both arms).
- SCOPE STAMP: **Intel x86 VPCLMULQDQ only.** aarch64 uses its own `crc32x` fold3 (this
  change is x86-cfg'd, no aarch64 effect) — a PMULL-wide aarch64 analogue is OWED.
  **AMD (Zen) replication OWED for LAW** (runtime-detect falls back if no VPCLMULQDQ).
- bin: feat/crc-vpclmul off reimplement-isa-l @8e2eb801, gzippy-native, target-cpu=native.
- HYPOTHESIS (label): the cache+kernel→logs-flip stack; the copy lever needed for a clean
  logs win. Next: AMD abmeasure (same env-toggle A/B) → bank as LAW; then aarch64 PMULL.
