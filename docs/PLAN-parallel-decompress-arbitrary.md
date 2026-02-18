# Parallel Decompression of Arbitrary Gzip Files

## Goal
Make gzippy the fastest decompressor for large gzip files from ANY tool (gzip, pigz, igzip, etc.),
not just our own BGZF-marked output.

## Current state
- BGZF-marked files (our output): parallel decompression works, 148% of libdeflate with 8 threads
- Multi-member gzip (pigz output): parallel per-member decompression works
- Single-member gzip (gzip output): single-threaded only, 99% of libdeflate

Single-member files are the gap. A 10GB gzip file from `gzip` decompresses on one core.

## Existing pieces (all on main or beat-all-decompression)

### block_finder_lut.rs (336 lines, on main)
- 15-bit LUT (32KB) for candidate rejection (~99% of positions eliminated)
- `is_deflate_candidate()`: checks BFINAL=0, BTYPE=2, HLIT<=29, HDIST<=29
- `compute_skip()`: multi-bit advancement for invalid positions
- `validate_precode()`: Kraft inequality check on precode Huffman tree

### block_finder.rs (697 lines, on main)
- Full 5-level validation pipeline:
  1. 13-bit LUT rejection
  2. Header field range check
  3. Precode leaf count (Kraft inequality)
  4. Full precode Huffman table build
  5. Full literal/distance code validation
- `BlockFinder::find_blocks(start_bit, end_bit) -> Vec<BlockBoundary>`
- Complete and functional

### marker_turbo.rs (539 lines, on beat-all-decompression branch)
- Fast inflate with u16 marker output instead of u8 bytes
- Uses the Bits struct from consume_first_decode.rs
- 2129 MB/s throughput (16x faster than old MarkerDecoder)
- Handles all block types (dynamic, fixed, stored)

## What's missing: the pipeline glue (~500-800 lines)

### Algorithm (rapidgzip approach)
1. **Scan phase**: Use block_finder to find candidate block boundaries in parallel
   - Divide file into N chunks (one per core)
   - Each thread scans its chunk for block boundaries
   - Block finder already handles false positive filtering

2. **Speculative decode phase**: Each thread decodes from its found boundary
   - Problem: deflate backreferences can point up to 32KB before the block start
   - Solution: decode speculatively with unknown backreference bytes as markers
   - marker_turbo already does this at 2129 MB/s

3. **Marker resolution phase**: Resolve markers using previous block's output
   - Once the previous block's output is known, replace markers with actual bytes
   - This is a simple byte-replacement pass (already exists in hyper_parallel.rs)

4. **Ordered output**: Write blocks in order to output
   - Same pattern as existing BGZF parallel output in bgzf.rs

### Key implementation decisions
- Minimum file size for parallel: ~4MB (below this, single-thread is faster)
- Chunk overlap: 32KB (maximum deflate backreference distance)
- Block boundary search: scan at byte-aligned positions first (most tools byte-align blocks),
  fall back to bit-level scan only if needed
- Thread count: match existing `available_parallelism()` pattern

## What the git history tells us
- The block finder is complete and tested (on main)
- marker_turbo works at high speed (2129 MB/s on beat-all-decompression)
- The HYPERION attempt (beat-all-decompression) built the pieces but never connected them
  into a working pipeline — the branch was abandoned at the routing/dispatch level
- The "glue" is the only missing piece

## Validation
- Decompress arbitrary gzip files from gzip, pigz, igzip and verify byte-for-byte match
- Benchmark on SILESIA single-member (211 MB) — target: linear speedup with cores
- Benchmark on small files (<1MB) — verify no regression from parallel overhead
