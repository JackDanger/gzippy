# DIVERGENCE LEDGER — T1-MONOLITH path (deliberate, T1-gated divergence from rapidgzip)

Per `feedback_fable_reflection_2026_06_12` rule 1: a divergence from rapidgzip is allowed
when (a) byte-exact output, (b) causally verified at the wall on ≥1 cell with no
regression cell, (c) recorded in THIS ledger with its vendor counterpart. rg remains the
blueprint + diagnostic oracle, NOT the ceiling.

## What this is (framing correction, honored)

This is **NOT** "convergence" and **NOT** a faithful rapidgzip port. gzippy's per-chunk
window/boundary/subchunk machinery is FAITHFUL TO RAPIDGZIP (`GzipChunk.hpp`
`finishDecodeChunkWithInexactOffset` :282-410: per-readStream `DecodedVector(128KiB)`
alloc, `appendDeflateBlockBoundary` per block-start, subchunk split, `getLastWindow`
per chunk) but DIVERGENT FROM the igzip serial monolith. This is a DELIBERATE, LEDGERED
**T1-SPECIALIZATION DIVERGENCE** from the rg chunk path TOWARD the igzip monolith.

## Why (mechanism, gated)

At T1 there is no parallelism to amortize per-chunk machinery: the decode is strictly
sequential front-to-back, so EVERY block already has its true 32 KiB predecessor window,
and the parallel block-finder / WindowMap / marker arming / prefetch / per-chunk boundary
index / subchunk split are pure overhead. The gated PROD-PATH-LOCATE measurement
(Intel+AMD LAW-grade, `plans/PROD-PATH-LOCATE-RESULTS.md`) shows this overhead is
+28.8..42.8% of the T1 wall vs the igzip monolith, dominated by per-chunk fixed cost.

## The blueprint for the T1 path = the igzip monolith (NOT rapidgzip)

`vendor/isa-l/igzip/igzip_inflate.c` `isal_inflate` (:2239-~2560): ONE call over the whole
stream; state inited ONCE; 32 KiB history kept implicitly in a `tmp_out` double-buffer
(`2*ISAL_DEF_HIST_SIZE`, :2342); decodes into `tmp_out` then `memcpy tmp_out→user out`
(:2436-2447); NO chunk lifecycle / NO block-boundary recording / NO subchunk split / NO
per-chunk alloc / NO per-chunk window handoff; CRC folded inline (`update_checksum`,
:2453). The T1-monolith path is a faithful port of THIS shape.

## The divergences (each with vendor counterpart)

| # | gzippy chunk path (faithful-to-rg) — vendor cite | T1-monolith divergence (toward igzip) | igzip counterpart |
|---|---|---|---|
| 1 | per-chunk `ChunkData::new` + `compute_initial_reserve` (≥4MiB/≤64MiB cap) per chunk, NO T1 recycler — `GzipChunk.hpp:310` `DecodedVector(128KiB)` per readStream | ONE `ChunkData`, ONE uncapped reserve = whole-member ISIZE, faulted once | `tmp_out` reused + caller `out` for the whole stream :2342-2447 |
| 2 | per-chunk rolling-window clone+re-seed: `vec![0u8;32768]`+`copy_last_into`+`set_initial_window` per chunk — `getLastWindow` GzipChunk.hpp:523 | ZERO per-chunk window handoff; single contiguous buffer IS the history (one zero-window seed for chunk 0) | implicit history in `tmp_out` (no analog) |
| 3 | per-chunk ISA-L re-slice/truncate/commit/boundary-replay OR per-chunk seeded-Block reset — `GzipChunk.hpp:282-410` | ONE engine pass over the whole stream (native: `decode_clean_into_contig` loop; isal: one `decompress_deflate_from_bit_into_growable`) | one `isal_inflate` :2239 |
| 4 | per-block `append_block_boundary_at` + subchunk split — `appendDeflateBlockBoundary` GzipChunk.hpp:364,561 | SUPPRESSED at T1 (`record_boundaries=false`); never consumed (no block map / prefetch / T>1 split at T1) | no boundary recording in igzip |

CRC second-touch (#5, per-byte re-read of just-decoded bytes) is KEPT for now (native
inline-during-loop CRC over `decoded_range`, cache-hot). igzip folds it truly inline
(:2453); shedding the second-touch is a separately-pre-registered residual lever if the
falsifier needs it.

## Scope guard (T1-gated; T>1 untouched)

The monolith fires ONLY at `T==1` (verified `GZIPPY_DEBUG` routing + `MONOLITH_T1_RUNS`).
`T>1` keeps `chunk_fetcher::drive` (the faithful rapidgzip chunk pipeline) byte-for-byte
unchanged — chunk granularity is T>1-LOAD-BEARING (`PROD-PATH-LOCATE-RESULTS.md` §T>1).
Kill-switch `GZIPPY_NO_MONOLITH=1` restores the legacy `drive_thin_t1_oracle` T1 path for
A/B re-verification.

## Verification (filled by the gated run — see plans/T1-MONOLITH-RESULTS.md)

- (a) byte-exact: sha==zcat on all corpora/arches (Gate-0).
- (b) causal-win at the wall, no regression cell: T1 prod/igzip drop vs the falsifier
  threshold; T4/T8 non-regression vs current build + rapidgzip.
- (c) this ledger + the falsifier doc, committed before the build.
