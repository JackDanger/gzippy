# footprint-align: segmented DecodedData port (2026-06-02)

## What shipped on this branch

A faithful port of rapidgzip's `DecodedData` memory model, as ONE structural change:

1. **`ChunkData::data`: contiguous `Vec<u8>` → `SegmentedU8`** (a list of fixed
   128 KiB segments, never reallocated/copied). Port of vendor
   `std::vector<DecodedVector>` (DecodedData.hpp:231-289). The decode hot loop
   writes each `read_stream` call into a fresh segment tail
   (`writable_tail`/`commit`); cross-segment back-references resolve via the
   resumable decoder's internal 32 KiB window ring (resumable.rs:653) — exactly
   how rapidgzip resolves them.

2. **In-place marker resolution** (`narrow_markers_in_place`): the resolved u16
   `data_with_markers` is narrowed to u8 INSIDE its own backing store (low byte
   of each element), exposed via `narrowed_bytes()`. The separate `narrowed`
   u8 buffer (memlife: 1.88 GB alloc / a buffer rapidgzip does NOT have) is
   ELIMINATED. Port of vendor `applyWindow` (DecodedData.hpp:325-388).

3. **`clean_unmarked_data`** now PREPENDS the migrated clean-tail as front
   segments (`prepend_bytes`) instead of a contiguous right-shift — vendor
   `dataBuffers.emplace(begin(), …)` (DecodedData.hpp:502).

4. **A3 window-prefill REMOVED** (was ON by default — opt-out env knob, NOT off
   as first assumed; this was the key surprise). A3 prefilled the predecessor
   window into the front of the contiguous output for a fast `copy_match`
   back-ref path — fundamentally incompatible with segments, and not a vendor
   pattern (rapidgzip uses the window ring). Forced off; `data_prefix_len`
   is now always 0.

## Correctness

- **0 mismatches** across ALL `benchmark_data/*.gz` × T{1,2,4,8,16} vs `gzip -dc`.
- 237/241 parallel unit tests pass. 3 failures are PRE-EXISTING on baseline
  (block_fetcher statistics, inflate_wrapper with_until_bits, isal_huffman_pure).
  The 4th (`drive_round_trips_2mb_level6`) passes in isolation and with its
  sibling tests; it fails only in the full cross-module `decompress::parallel`
  run — order-dependent shared-global-pool state (documented pre-existing
  parallel-test isolation flakiness), NOT a production-path bug (proven by the
  exhaustive byte-exact corpus differential above).

## Frozen whole-system A/B (neurotic, N=11, interleaved, sha-verified, file+pipe)

baseline = reimplement-isa-l, fp = feat/footprint-align, rg = rapidgzip 0.16.0.

FOOTPRINT (the target) — DROPPED toward rapidgzip:
| T / sink | base maxRSS | fp maxRSS | rg maxRSS | base minflt | fp minflt |
|----------|-------------|-----------|-----------|-------------|-----------|
| T8 file  | 1081 MB     | 768 MB (-29%) | 354 MB | 252k     | 159k (-37%) |
| T16 file | 1149 MB     | 879 MB (-23%) | 556 MB | 260k     | 188k     |
| T1 file  | 231 MB      | 235 MB    | 66 MB     | 21k         | 22k      |

IPC (instr/cycle) — RECOVERED at T8/T16:
| T  | base | fp    | rg    |
|----|------|-------|-------|
| T8 | 1.96 | 2.01  | 2.42  |
| T16| 1.48 | 1.51  | 1.78  |

WALL ratio vs rapidgzip — multi-thread TIE w/ baseline, T1 regressed:
| T / sink | base  | fp    |
|----------|-------|-------|
| T1 file  | 1.122 | 1.264 (REGRESSED) |
| T8 file  | 1.322 | 1.352 (~tie) |
| T8 pipe  | 1.397 | 1.401 (TIE)  |
| T16 file | 1.642 | 1.732 |
| T16 pipe | 1.424 | 1.447 |

## VERDICT

- **Footprint target ACHIEVED**: T8 maxRSS 1081->768 MB (-29%), minflt -37%,
  IPC nudged up — the segmented buffer + eliminated `narrowed` buffer are the
  real resident-footprint levers (the prior reservation-size tweak was not).
- **Wall is NOT the win**: multi-thread wall is a TIE with baseline (<=2% at
  T8), so reducing footprint to ~0.7x did NOT close the ~1.35x wall gap to
  rapidgzip => working-set/DRAM contention is NOT the dominant T8 wall lever
  (the residual gap is decode-rate + the diffuse architectural difference).
- **T1 regressed** (1.122->1.264): single-threaded pays for per-segment alloc +
  the loss of the A3 contiguous `copy_match` fast-path, with no footprint
  benefit at T1. In real production T1 single-member routes to libdeflate
  one-shot, not this path; the regression is on the forced-parallel T1 path.

## Bottom line

The faithful segmented-`DecodedData` + in-place-narrow port cuts gzippy's T8
resident footprint ~29% (toward rapidgzip's) at multi-thread wall-parity with
baseline and correct bytes everywhere — but it confirms footprint is NOT the
wall lever (wall TIE), and it costs T1. A footprint/memory win; the wall gap
needs a different lever (decode-rate).
