---
name: gzippy-parallel-decompress
description: Development workflow for parallel single-member gzip decompression. Use when working on block finding, marker decoding, parallel pipelines, or any single-member Tmax decompression work.
---

# Parallel Single-Member Decompress Development

## Read First

Before making ANY changes to parallel decompression code, read:
1. `.cursor/rules/parallel-decompress.mdc` — architecture, proven failures, invariants
2. `src/pipeline_tests.rs` — layered oracle test harness (the source of truth)
3. `src/parallel_single_member.rs` — production pipeline + regression tests

## The 7 Invariants

Every parallel implementation must satisfy ALL of these. The test suite checks them.

1. **Sum of chunk output sizes ≈ ISIZE** (within 2x — closer is better)
2. **Each chunk's output ≤ max_output_for_chunk()** (8x compressed size)
3. **Chunk 0 has 0 markers** (starts from beginning of stream)
4. **Chunk 0 output matches sequential decode** (byte-exact)
5. **Data slices are bounded**: `&data[start..end]`, never `&data[start..]`
6. **Block finder candidates verified by try-decode before use**
7. **Boundaries are sorted and within data bounds**

## Development Workflow

### Step 1: Prove with Oracle

Use `DeflateOracle` boundaries + windows first. This isolates pipeline logic
from block finding. Run:

```bash
cargo test --release test_layer4_pipeline_oracle_source -- --nocapture
```

If oracle pipeline doesn't produce byte-exact output, the pipeline is broken.
Fix the pipeline before touching the block finder.

### Step 2: Test Invariants

Run the regression test suite:

```bash
cargo test --release parallel_single_member -- --nocapture
```

All 11 tests must pass. Key ones:
- `test_try_decode_rejects_random_positions` — false positive rate <20%
- `test_try_decode_accepts_oracle_boundaries` — must be 100%
- `test_chunk_output_bounded` — no chunk exceeds its limit
- `test_chunk0_no_markers_matches_sequential` — chunk 0 is correct
- `test_e2e_roundtrip_strict` — either correct output or clean fallback

### Step 3: Test on Real Data

```bash
cargo test --release test_parallel_silesia -- --nocapture
```

If this falls back, check the debug output:
```bash
GZIPPY_DEBUG=1 cargo test --release test_parallel_silesia -- --nocapture
```

### Step 4: Benchmark (Only After Correctness)

```bash
cargo test --release bench_cf_silesia -- --nocapture  # single-thread baseline
cargo test --release bench_production_inflate -- --nocapture  # production
```

## Known Failure Modes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Chunk output 10x+ expected | `decode_until` doesn't check mid-block | Use `output_limit` field |
| Total output 5x ISIZE | Chunks overlap due to no size cap | `max_output_for_chunk()` |
| All chunks fail to decode | Block finder false positives | Try-decode validation |
| -86% regression | `&data[start..]` tail passed to decoders | Bound slice to `[start..end]` |
| Sequential slower than expected | Parallel path falling back silently | `GZIPPY_DEBUG=1` to trace |
| 0 markers everywhere | Decoder starts from real beginning | Expected for chunk 0 |
| Simulation ≠ reality | Micro-benchmark doesn't model full loop | Always bench on silesia |

## Key Modules

| Module | Purpose | Hot Path? |
|--------|---------|-----------|
| `marker_decode.rs` | MarkerDecoder: decode with markers for unresolved refs | Yes |
| `block_finder.rs` | LUT + precode + Huffman block candidate finding | No (setup) |
| `parallel_single_member.rs` | Production pipeline + regression tests | Yes |
| `pipeline_tests.rs` | 4-layer oracle test harness | Tests only |
| `scan_inflate.rs` | Sequential scanner (ground truth) | Tests only |
| `consume_first_decode.rs` | Pure Rust inflate (used by chunk 0 alternative) | Yes |

## What Still Needs Work

The block finder has 0% recall at finding real boundaries within the search
radius. The pipeline falls back to sequential correctly but doesn't provide
speedup yet. Two paths forward:

1. **Improve block finder recall** — current LUT only finds dynamic blocks,
   and even those have ~38% recall at exact positions. Need better LUT or
   different search strategy.

2. **Use scan-only approach** — a lightweight scanner that tracks bit positions
   and 32KB windows without writing full output. Could be 2x faster than full
   decode, making two-pass viable at 4 threads.

Both approaches should be tested against the invariant suite before wiring
into production.
