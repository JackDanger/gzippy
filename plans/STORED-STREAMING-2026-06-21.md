# STORED-STREAMING results — SURPASS Target 1 (pure-stored monolithic-buffer kill)

**Date:** 2026-06-21. **Branch:** `kernel-converge-A`. **Subject sha:** `105efc90`
(fix commit `21562a28` + debug-witness `105efc90`). **Box:** neurotic (Intel
i7-13700T), FROZEN (`BENCH_LOCK=quiet`, runnable_avg=1.25, no_turbo=1,
governor=performance, guest 199 in keep_allowlist). Guest = trainer (LXC, 16c).
**Status:** Intel-frozen BOX-VALID → **NOT-YET-LAW** (AMD/Zen2 owed).

## What changed (file:line)
`src/decompress/parallel/stored_split.rs` — the pure-stored arm
`decompress_stored_parallel` → `WalkEnd::Final`:
- OLD: `vec![0u8; total]` (100 MB zero-init → first-touch page-fault storm) +
  `fill_and_crc` copy-runs-into-buffer + `verify_and_write` `write_all` the whole
  buffer = a 100 MB alloc-fault + 2 full copy passes.
- NEW: NO intermediate buffer. The output bytes ARE the verbatim input run
  slices (`StoredRun{src_off,len}` index into `deflate`), so:
  - `crc_runs` (new) computes the whole-output CRC32 straight from the input run
    slices — parallel partitioned (reusing `partition_runs`) + `combine_crc32`
    fold, output order == run order (`out_off` ascends); inline for T≤1 / small.
  - CRC + size verified against the trailer BEFORE any write (input is fully
    buffered → verify-before-write / no-partial-output-on-corruption PRESERVED
    exactly, identical contract to the old path).
  - `write_runs` (new) streams the input run slices directly to the writer
    (rg-style chunk-by-chunk writeAll, `DecodedData.hpp` + `GzipChunkFetcher`).
- A/B kill-switch `GZIPPY_STORED_MONOLITHIC=1` restores the old buffer path
  (same-binary measurement). Gate-0 witness counter `STORED_STREAM_RUNS` dumped
  by `GZIPPY_DEBUG=1` on the StoredParallel ok-path.
- Stored-prefix+Huffman-tail path UNCHANGED (tail has no explicit length).

## Correctness (HARD gate — all PASS)
- **sha==zcat**: pure_stored_100mb, storedheavy, storedmix, silesia × T1/T4/T8 =
  12/12 PASS (== `zcat`).
- **cargo test (x86_64, gzippy-native, AVX2)**: `stored_split` 13/13 pass (incl.
  new `crc_runs_matches_whole_crc` = parallel-combined CRC == serial
  crc32(whole) at every T; corrupt-NLEN + CRC-mismatch terminal-Err); `routing`
  45/45 pass (4 pre-existing ignores).
- **corrupt-trailer still Errs**: flipping a CRC byte → terminal
  `stored CRC32 mismatch: expected 62726bff, got 62726b82`, exit 1, NO bytes
  written (crc_runs caught it before write_runs — verify-before-write proven).

## Mechanism witness
- **RSS drop** (pure_stored_100mb T4, /usr/bin/time -v, same binary):
  streaming = **106608 KB** vs monolithic = **207476 KB** → **−100.9 MB**
  (exactly the eliminated 100 MB zero-init buffer).
- **Path/counter** (`GZIPPY_DEBUG=1`): pure_stored_100mb →
  `path=StoredParallel` + `chunked-streaming runs (no monolithic buffer)=1`
  (non-inert). storedheavy → `StoredParallel declined → pure-Rust SM` (demoted;
  my path NOT taken). storedmix → `path=ParallelSM` (my path NOT taken).

## Frozen wall (interleaved best-of-N=13, /dev/null, A/A self-test ≤5%, sha==zcat)
gz/rg = gz_wall / rg_wall (<1.0 = gz wins). All pure_stored cells A/A-TRUSTED.

| cell | gz cyc/B | gz wall | rg wall | gz/rg | before (gated) | verdict |
|---|---|---|---|---|---|---|
| pure_stored_100mb T1 | 0.63 | 50.99 | 80.04 | **0.637** | ~1.5 LOSS | **WIN −36%** (also beats igzip 0.935) |
| pure_stored_100mb T2 | 0.63 | 44.16 | 54.87 | **0.805** | ~1.6 LOSS | **WIN −19.5%** (beats igzip 0.803) |
| pure_stored_100mb T4 | 0.65 | 44.48 | 50.92 | **0.874** | ~1.7 LOSS | **WIN −12.6%** (beats igzip 0.809) |
| pure_stored_100mb T8 | 0.78 | 42.29 | 33.42 | 1.265 | ~1.7 LOSS | residual LOSS (improved) |

- **Headline:** pure_stored flipped from all-T LOSS (gated baseline 1.46–1.79) to
  **T1/T2/T4 WINS**; T8 improved from ~1.7 to 1.265.
- gz cyc/B stays ~0.63–0.78 (flat, bandwidth-bound); rg cyc/B rises 1.03→2.15
  with T. At T8 rg still wins the *wall* (33.42 vs 42.29 ms) by throwing more
  parallel memory bandwidth at the copy — gz is far more cycle-efficient but its
  single-writer stream floors at ~42 ms (100 MB read+write). T8 residual is a
  write/CRC bandwidth-parallelism gap, not the old alloc-fault tax.

## No compressible regression (silesia — ParallelSM path, NOT touched by this fix)
silesia T1 1.068 / T2 1.034 / T4 1.044 / T8 0.940 — consistent with the
pre-fix MATRIX-COMPLETE (silesia-T4 ~1.032 TIE). My change does not touch the
compressible path; no regression attributable to it.

## storedheavy / storedmix — OUT OF SCOPE for Target 1 (separate front)
Both have Huffman tails → they DECLINE/demote off the pure-stored path
(storedheavy → ParallelSM; storedmix → ParallelSM). Their T≥2 losses
(storedheavy T2 1.243/T8 1.367; storedmix T2 1.149/T4 1.230/T8 1.412) are the
PRE-EXISTING ParallelSM-pipeline-on-stored gap — unchanged by this fix and a
distinct target. storedheavy/storedmix T1 WIN (0.724 / 0.679). storedheavy T4
was A/A-UNTRUSTED (loaded-box, withheld).

## Verdict
**KEEP.** Byte-exact (12/12 sha==zcat + corrupt-trailer Errs + 58 unit tests);
the one path the fix touches (pure-stored) flipped from all-T LOSS to T1–T4
WIN with a clean −100 MB RSS mechanism witness; no compressible regression.
pure_stored T8 residual (1.265, bandwidth-parallelism not alloc-fault) and the
storedheavy/storedmix T≥2 ParallelSM-on-stored losses are reported as separate,
remaining fronts.

**Stamp:** Intel-frozen BOX-VALID, NOT-YET-LAW; AMD/Zen2 replication owed
(`standing.sh --box solvency`). Freeze acquired+released+RESTORE-VERIFIED
(no_turbo=0, guests-thawed).
