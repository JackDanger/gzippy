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
is real resident memory: the parallel path allocates per-chunk ChunkData buffers and (with
the manual pool OFF at T>1) drops them, while several decoded-but-unconsumed chunks are
in flight at once. Their TOUCHED (resident) output bytes + windows accumulate to the peak.
This is the load-bearing RSS target. (See continuation below as the lever is built.)
