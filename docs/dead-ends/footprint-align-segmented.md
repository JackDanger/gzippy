# Dead End (HELD, not fully refuted): footprint-align SegmentedU8 DecodedData port

## Hypothesis

Port rapidgzip's full `DecodedData` layout faithfully:
- `SegmentedU8` at 128 KiB granule for `ChunkData::data` (the clean u8 suffix)
- Segmented 128 KiB granule for `data_with_markers` (the u16 marker prefix),
  moved from glibc to rpmalloc
- In-place-resolve (eliminate the separate `narrowed` buffer)
- Remove the A3 prefill optimization (not present in vendor)

Together, this achieves footprint –29% toward rapidgzip's ~350 MB RSS (from gzippy's
~1040 MB) while matching the vendor buffer lifecycle precisely.

## How Measured

Branch `feat/footprint-align`, commit `2b8bfae`. Measured on the clean frozen-clock
neurotic harness (`scripts/bench/clean_bench.sh`, N≥9 interleaved, sha-verified,
T4/T8/T16).

- Footprint (maxrss): 1040 MB → 738 MB = **–29%** — HIT the target trajectory toward
  rapidgzip's ~350 MB.
- T4 / T8 wall: **TIE** (within frozen-clock spread, sd <3%).
- T16 wall: **REGRESSED +5.5%** vs the pre-segmented baseline.

The T16 regression mechanism is **confirmed**: the SegmentedU8 port removed the A3
prefill at `gzip_chunk.rs:178`, which is a measured **+4.2% T16 production win**
(separately A/B'd). When A3 is absent, T16 workers stall slightly more because the
next chunk's decode buffer is not prefaulted, and the faults land on the critical
path at high thread counts.

## Verdict: RE-ENTERED (2026-06-02) — segment-native A3 landed on main

The segmented port is wired into production `ChunkData::data` (`SegmentedU8` at
128 KiB granule), in-place marker narrow (`narrowed_len`), and **segment-native A3**:
`prefill_window_prefix` writes the predecessor 32 KiB window into segment 0;
`read_stream_starting_at` uses `first_segment_a3_output()` while
`all_in_first_segment()`; consumer skips prefix via `write_payload_skipping_prefix`.

The T16 +5.5% regression from A3-removal on the held branch should be addressed;
**re-measure** T4/T8/T16 + RSS on frozen harness before calling this shipped.

## Code Location

Branch: `feat/footprint-align`, commit `2b8bfae`.

The SegmentedU8 type is scaffolded at:
- `src/decompress/parallel/segmented_buffer.rs` — full implementation
- `src/decompress/parallel/chunk_data.rs` — `data` is `SegmentedU8`;
  `data_with_markers` is `segmented_markers::SegmentedU16`

## Re-Entry Conditions

**segment-native A3**: prefill segment 0 of the new SegmentedU8 buffer at construction
time (matching the original A3 behavior). This is the single missing piece that blocked
production landing. Branch `feat/footprint-align` commit `2b8bfae` is the base.
See `docs/open-candidates.md` for the full ship-gate.

## Related Entries

- `docs/dead-ends/footprint-bandwidth.md` — broader DRAM/footprint theory (6 refutations)
- `docs/dead-ends/data-plane-2touch.md` — the aggregate port (granule + in-place + writev)
  that REGRESSED on file output; the segmented port alone without writev is still open
- `segmented_buffer.rs` — the SegmentedU8 implementation itself
