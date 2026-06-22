# RSS / MEMORY-MODEL CONVERGENCE — RESULTS

Branch: kernel-converge-A (worktree gzippy-amd-t2t4). Base: 6b88251c.
Primary metric: PEAK RSS (load-immune). macOS aarch64 `/usr/bin/time -l` maximum
resident set size; Linux x86_64 `/usr/bin/time -v` Maximum resident set size.
All RSS readings were deterministic (run-to-run spread ≤ 0.02 MiB).

## LEVER 2 — T1 right-size resident reserve → **FALSIFIED (reverted)**

HYPOTHESIS (from the mission's bias-flagged premise): the shipped 64 MiB T1 resident
pool PINS ~64 MiB peak RSS for EVERY T1 single-member regardless of input size, so
sizing it to `min(64 MiB, ISIZE)` would drop small-file RSS toward input-proportional
while leaving large files at 64 MiB.

GATE-0 NON-INERT METRIC SELF-TEST refuted the premise. The 64 MiB reserve is **virtual
capacity** (`Vec::with_capacity` / `reserve`), never made resident — pages fault only on
first WRITE, so peak RSS already scales with the actual decoded chunk size, NOT with the
reserve. Baseline (no change) peak RSS at T1:

| corpus (T1)            | arch          | BASELINE peak RSS |
|------------------------|---------------|-------------------|
| 4 MiB synth (small)    | macOS aarch64 | **8.48 MiB**      |
| 120 MiB synth (large)  | macOS aarch64 | 28.42 MiB         |
| engine.wasm (~1.5 MiB) | x86_64 native | **6.5 MiB**       |
| monorepo.tar           | x86_64 native | 23.0 MiB          |
| silesia (~210 MiB out) | x86_64 native | 80.5 MiB (input-mmap dominated) |

Small-file T1 RSS is 6.5–8.5 MiB on BOTH arches — there is NO 64 MiB pin to reclaim.

EFFECT OF THE LEVER (min(64 MiB, ISIZE) reserve), macOS aarch64, deterministic N=5:

| corpus (T1) | BASELINE | LEVER 2 | verdict |
|-------------|----------|---------|---------|
| small 4 MiB | 8.48 MiB | **12.53 MiB** | REGRESSION (+4.05 MiB) |
| large 120 MiB | 28.42 MiB | 28.41 MiB | TIE (reserve still 64 MiB) |

Shrinking the reserve to ISIZE makes the decode's contiguous output Vec grow by
realloc-doubling DURING decode (e.g. 4→8 MiB), so the old and new buffers are both
resident across the realloc memcpy → a HIGHER transient peak than writing into the
already-reserved (but virtual) 64 MiB buffer with no realloc. Per the pre-registered
falsifier ("FALSIFIED iff: no small-file RSS drop, OR any throughput regression"),
Lever 2 is FALSIFIED and REVERTED. The bias-flagged premise it rested on is dead on
both arches (load-immune metric).

## LEVER 1 — T2 ChunkData slab recycle + window eviction → target CONFIRMED real

The T2 RSS gap is REAL and reproduces (x86_64 native, deterministic):

| corpus       | gz T2 peak RSS | gz T4 peak RSS |
|--------------|----------------|----------------|
| monorepo.tar | **96.5 MiB**   | 83.8 MiB       |
| silesia      | 246.2 MiB      | 211.4 MiB      |

monorepo-T2 96.5 MiB matches the gated 97 MB vs rapidgzip 59 MB (1.65×). Unlike T1, this
is real resident memory.

### Gate-2 LOCATE (causal perturbation, knob = chunk size) — CONFIRMED
monorepo decoded = 50.9 MiB. Peak RSS at T2 is MONOTONIC+PROPORTIONAL to the per-chunk
decoded byte size (deterministic, x86_64 native):

| GZIPPY_CHUNK_KIB | monorepo-T2 peak RSS |
|------------------|----------------------|
| 256              | 38.5 MiB             |
| 512              | 49.5 MiB             |
| 1024             | 72.8 MiB             |
| 4096 (default)   | 96.8 MiB             |

=> The T2 peak RSS is governed by IN-FLIGHT DECODED CHUNK BYTES (chunk size × concurrent
in-flight/cached chunk count). At T2 the file is only ~4 chunks while `cache_capacity =
max(16, pool) = 16` (vendor-faithful, BlockFetcher.hpp:181-183), so essentially the WHOLE
output plus per-chunk overhead is resident at peak. (Same-chunk-size T1 vs T2: 1024 KiB
→ T1 23 MiB vs T2 72.8 MiB — the +50 MiB is the parallel in-flight set + marker overhead.)

### Candidate faithful-rg mechanisms for the ~37 MiB gz-vs-rg gap (HYPOTHESIS — unisolated)
NOT yet isolated to a single cause; the only Gate-2-CONFIRMED fact is "RSS ∝ in-flight
decoded chunk bytes". Candidates, each with a vendor anchor, to isolate BEFORE any code:
1. **u16 marker buffer never freed.** `apply_window` (apply_window.rs:22-44) resolves
   markers IN PLACE in `chunk.data_with_markers` (SegmentedU16, 2 bytes/symbol) but never
   frees/down-converts it; a speculatively-decoded (window-absent) chunk therefore holds a
   2×-size u16 buffer for its whole lifetime. rapidgzip frees/recycles the marker storage
   after `applyWindow` (vendor `ChunkData::applyWindow` + `MarkerReplacement`). At T2,
   chunks 1..N are window-absent ⇒ markered ⇒ 2× memory. Isolation: a fully clean-window
   corpus, or instrument resident `data_with_markers` bytes at peak. (The wired
   GZIPPY_CLEAN_WINDOW_ORACLE is NOT a usable perturbation — it errored mid-decode here.)
2. **No ChunkData slab recycling at T>1.** `manual_buffer_pool_enabled()` is false at T>1,
   so `take_u8`/`return_u8_to_worker` allocate-fresh/drop (chunk_buffer_pool.rs:321-382);
   vendor recycles via `FasterVector`/`RpmallocAllocator` (FasterVector.hpp:120-128). This
   is primarily a CHURN/fault mechanism, not obviously a PEAK-RSS one — lower priority for
   the RSS goal, higher for the teardown/throughput goal.
3. **WindowMap not evicted during decode.** windows (32 KiB each) stay in `window_map`
   (chunk_fetcher.rs:749) for the member lifetime; rg evicts resolved windows. Minor at
   monorepo scale (few chunks) but scales with chunk count on large corpora.

### PRE-REGISTERED next measurement (before writing Lever-1 code)
Isolate candidate #1 with a peak-resident instrument over `data_with_markers` vs `data`
(byte-transparent, env-gated) on monorepo-T2; if the u16 marker buffer is ≳ the gz-vs-rg
gap, the faithful fix is to free/down-convert `data_with_markers` immediately after
`apply_window` (mirroring rg's post-applyWindow marker-storage release) — CORRECTNESS-
GATED (the consumer currently reads resolved bytes from `data_with_markers`, so the write
path must read from the down-converted `data` instead; byte-exact + silesia differential
in the same commit). NOT shipped this session: it is a correctness-sensitive change to the
parallel consumer/marker path and the governing rule forbids code from an unconfirmed
model — the isolation measurement above is the gating prerequisite.

## STATUS
- Lever 2: COMPLETE — gated FALSIFIED + reverted (premise dead both arches).
- Lever 1: target + driver gated-CONFIRMED; specific faithful mechanism HYPOTHESIS,
  isolation measurement pre-registered. No production code shipped (anti-bias discipline).
