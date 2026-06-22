# RSS / MEMORY-MODEL CONVERGENCE — PRE-REGISTERED FALSIFIER

**Branch:** kernel-converge-A (worktree gzippy-amd-t2t4). Committed BEFORE building.
**Primary verdict metric:** PEAK RSS (LOAD-IMMUNE — valid under llama / contended box).
  - macOS aarch64: `/usr/bin/time -l` → "maximum resident set size" (bytes).
  - Linux x86_64: `/usr/bin/time -v` → "Maximum resident set size" (KiB), or
    `/proc/<pid>/status` `VmHWM`.
**Secondary (WALL, only when box quiet):** interleaved best-of-N≥7, Gate-1 Δ vs spread.

## GATED STARTING POINT (load-independent facts being converged)
- gz peak RSS = 97 MB vs rapidgzip 59 MB (1.65×) on monorepo-T2.
- AMD-T2 wall excess (~13–17 ms) is kernel address-space TEARDOWN at process `_exit`,
  MONOTONIC+PROPORTIONAL to RSS (+0.054 ms/MB; Gate-2 knob = RSS).
- T1: shipped 64 MiB resident pool pins 64 MiB RSS for EVERY T1 single-member regardless
  of input size (RSS regression on small files; large-file wall win is the gated keeper).

## GATE-0 (every commit, BLOCKING)
- sha == zcat of every arm (byte-exact).
- flate2 + libdeflate silesia differential, MULTIPLE chunk sizes, IN THE SAME COMMIT.
- RSS-metric self-test: the metric must reflect the change NON-INERTLY (a knob that
  changes the reserve must move peak RSS in the expected direction; an inert metric
  reading is not a measurement).
- Gate-4: `GZIPPY_DEBUG=1` → path=ParallelSM; grep the changed symbol IS in the built
  binary before trusting any delta (the push.default=tracking incident: verify the box
  built THIS code).

## PER-LEVER PASS/FAIL

### Lever 2 — T1 right-size resident reserve (do NOT pin flat 64 MiB for tiny inputs)
- **CONFIRMED iff:** byte-exact AND, on a SMALL single-member input (output ≪ 64 MiB),
  peak RSS drops toward input-proportional (target: small-file peak RSS no longer pinned
  at ~64 MiB; falls toward output size + fixed overhead), measured LOAD-IMMUNE on BOTH
  macOS aarch64 AND a Linux x86_64 box, with **NO throughput/wall regression on the LARGE
  T1 inputs** (silesia/monorepo) — the resident-pool cache-residency wall win preserved.
  Design intent: for inputs with ISIZE ≥ 64 MiB the reserve stays EXACTLY 64 MiB
  (byte-identical behavior), so the large-file win cannot regress.
- **FALSIFIED iff:** no small-file RSS drop, OR any throughput regression on large T1
  (silesia/monorepo) or any other cell → revert.

### Lever 1 — T2 recycle ChunkData slabs + evict resolved windows
- **CONFIRMED iff:** byte-exact AND peak RSS at T2 drops from ~97 MB toward rg's ~59 MB,
  measured LOAD-IMMUNE on BOTH arches, with NO throughput/wall regression on T2/T4/T8 or
  any cell. Faithful to rg BlockFetcher/ChunkData lifecycle (cite vendor file:line).
- **BONUS (box quiet):** T2 teardown wall tax drops proportional to the RSS reduction
  (Gate-2 already established +0.054 ms/MB).
- **FALSIFIED iff:** no RSS drop / a throughput regression → revert that lever.

## SCOPE STAMP
TIME/CODE-STAMPED: kernel-converge-A @ 6b88251c. Verdicts hold only at this commit/corpus/
arch/T. RSS is the load-immune primary; any WALL number carries its own Gate-1 accounting
and box-state (frozen+restored+verified, llama flagged) note. Each lever is a SEPARATE
gated commit. No ROI/phase language; HYPOTHESIS/TIE/PARTIAL labels mandatory.
