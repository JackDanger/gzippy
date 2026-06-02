# footprint-align findings (2026-06-02) — VERDICT: theory's specific lever REFUTED, real levers LOCATED

## What was tested

Theory: gzippy's gap to rapidgzip at T8 is aggregate working-set / DRAM
contention — gzippy keeps ~2.4-3× the resident footprint per worker, fine at
T1 but at T8 the combined working set overruns LLC + saturates DRAM → IPC
collapses. Proposed whole-system fix: shrink the resident footprint toward
rapidgzip's (windows-compressed + buffer right-sizing + pool-retention).

## Grounding (memlife instrument, pure-rust, silesia-large T8, guest199)

`GZIPPY_MEMLIFE` per-component byte attribution + getrusage:

| component         | alloc bytes | written | notes |
|-------------------|-------------|---------|-------|
| data_with_markers | 485 MB / 38 | 312 MB  | u16, on **glibc**, 62% of decoded bytes go through markers |
| data              | 8.64 GB /106| (bulk)  | rpmalloc-huge 3.27 GB/39 + pool-hit |
| narrowed          | 1.88 GB / 32| 149 MB  | **separate u8 buffer rapidgzip does NOT have** (it resolves in place) |
| window            | 1.3 MB / 40 | —       | **already compressed (Zlib)** — premise stale, no lever |

maxrss T8: **gzippy 1044 MB** vs **rapidgzip 347 MB** (3.0×); T16 1182 vs 538 (2.2×).
gzippy marker fraction **62%** vs rapidgzip **31.46%** (rapidgzip `-v`).

## What was tried + the frozen A/B (N=11, interleaved, sha-verified, file+pipe, T1/8/16)

Change: right-size `data` initial cap 80 MiB → split_chunk_size (4 MiB);
MAX_POOLED 8 → 3 (then reverted to 8 — see below).

Result (wall ratio vs rapidgzip, T8 file): **base 1.297 → fp 1.510 (WORSE)**;
T16 file 1.677 → 1.922 (WORSE). minflt 232k → 363k (UP). maxrss 1025 → 1010 MB
(barely moved). IPC T8 base 1.95, fp 1.86, rapidgzip 2.42.

allocator TRAFFIC fell 55% (data alloc 8.64 GB→1.10 GB) but **peak maxrss did
not drop and the wall regressed**.

## VERDICT

**The footprint theory's specific lever (over-RESERVATION) is REFUTED.** The
80 MiB `with_capacity` reserves address space but the kernel only faults the
~12 MiB actually written, so the reservation was never the resident driver.
Shrinking it FORCED the contiguous `data` Vec to realloc+memcpy+RE-FAULT as it
grows past 4 MiB → +131k minor faults → wall regression. The change is a net
regression and was NOT shipped (branch reverts to baseline behavior).

## The REAL, mechanism-backed levers (located, not yet built)

1. **Segmented `data` buffer (the architectural divergence).** rapidgzip's
   `DecodedData::data` is `std::vector<DecodedVector>` — a LIST of fixed
   128 KiB chunks, never reallocated/copied (DecodedData.hpp:231-289). gzippy
   uses ONE contiguous growing Vec, so it either over-reserves (resident
   waste) or reallocs+refaults (the regression we just saw). gzippy already
   HAS `segmented_buffer.rs` (128 KiB ALLOCATION_CHUNK_SIZE) — wiring `data`
   to it would match vendor and remove BOTH the over-reservation AND the
   grow-realloc faults. This is the faithful fix.

2. **In-place marker resolution (eliminate `narrowed`, 1.88 GB alloc).**
   rapidgzip resolves the u16 `dataWithMarkers` IN PLACE into its own backing
   memory reinterpreted as u8 (DecodedData.hpp:325-388: `target[i] =
   fullWindow[chunk[i]]` via `reinterpret_cast<uint8_t*>(chunk.data())`, then
   swap+VectorView). gzippy resolves in the u16 buffer THEN copies down into a
   SEPARATE `narrowed` u8 buffer — a buffer rapidgzip does not have, held
   co-resident with `data_with_markers`. Narrowing in place into the u16
   buffer's own memory (safe: write byte i never clobbers u16[≥i] since i<2i)
   removes the `narrowed` allocation and the double-residency.

3. **62% vs 31% marker fraction (the deepest lever).** gzippy marker-decodes
   2× as many bytes as rapidgzip → 2× the resident u16 `data_with_markers`
   working set AND 2× the apply_window/narrow CPU. This is window-propagation
   / speculation timing: workers decode before the predecessor window is
   published, emitting markers. Closing this toward 31% is the largest single
   reducer of both footprint and the marker-resolve CPU, but it is an
   architectural change to the prefetch/window-publish ordering, not a
   buffer-sizing knob.

## Why IPC-collapse-from-contention is also weakened as the story

IPC at T8: gzippy 1.95 vs rapidgzip 2.42. gzippy's lower IPC is real, but
halving allocator traffic (the fp arm) did NOT raise IPC (1.86) — so the IPC
gap is not driven by allocator/fault pressure from over-reservation. It tracks
the marker-resolve overhead (62% marker bytes = a big u16→u8 pass rapidgzip
does at half the volume + in place) and the contiguous-buffer realloc traffic.

## Box

All runs frozen via host_lock_and_bench.sh (watchdog TTL 2700s); box restored
to baseline (no_turbo=0, paranoid=4, uncore=0x82b, guests thawed) + lock
released after each run.
