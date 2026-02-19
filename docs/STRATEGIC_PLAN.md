# Strategic Plan: Beat Every Tool in Every Scenario

## Current State (Feb 2026)

**71 wins, 52 gaps** across 123 performance comparisons (2 platforms × datasets × thread counts × competitors).

### What We Already Dominate
- **ALL compression**: 1.3-2.8x faster than pigz at every level, thread count, and platform
- **x86 T1 decompression**: Within 0-2% of all tools on software and logs
- **x86 Tmax decompression**: Within 0-3% on software, logs (non-silesia)

### What We Must Fix

Three distinct performance problems, in priority order:

---

## Problem 1: ARM64 T1 Sequential Inflate (24% gap)

**The gap**: gzippy is 20-25% slower than pigz and rapidgzip at T1 on ARM64 across ALL archive types.

**Why this is #1**: This is the largest *systemic* gap. It affects every single ARM64 T1 scenario. Neither pigz nor rapidgzip parallelize at T1, so they're just running faster sequential inflate than we are.

**Root cause hypothesis**: libdeflate FFI call overhead and/or buffer management. On ARM64, the overhead of:
- ISIZE trailer parsing and pre-allocation
- BGZF header scanning (even for non-BGZF files)
- FFI boundary crossing to libdeflate

...is proportionally larger than on x86 because the CI ARM64 runners have different memory subsystem characteristics.

**Investigation plan**:
1. Instrument `decompress_single_member_libdeflate` and `decompress_bgzf_parallel` with timing breakdowns (header parse, buffer alloc, inflate, CRC check)
2. Compare with a minimal "raw libdeflate inflate" baseline that skips all gzippy wrapper logic
3. Profile on actual ARM64 hardware

**Fix approaches (try in order)**:
1. Fast-path for small files: skip multi-member scanning for files < 10MB
2. Pre-allocate from ISIZE without scanning: go straight to inflate with ISIZE-sized buffer
3. Inline the critical path: avoid function call overhead in the hot decompression path

**Expected impact**: 15-20% improvement on ARM64 T1, closing the gap to <5%.

**Measurement**: `./gzippy-dev ci gaps` — track all ARM64 T1 rows

---

## Problem 2: Single-Member Tmax Parallel (19-21% gap)

**The gap**: rapidgzip achieves 1.2-1.3x speedup on Tmax for single-member files via speculative parallel decoding. We run sequentially.

**Affected scenarios**: silesia-gzip Tmax (both platforms), silesia-pigz Tmax (both platforms)

**Why this is #2**: Only affects Tmax on large files. Fewer scenarios than Problem 1, but the individual gaps are large (19-21%).

**What's been tried and failed**:
- Two-pass parallel: scan pass alone costs more than sequential libdeflate
- Prefix-overlap: window convergence fails at high thread counts
- Speculative parallel (current attempt): false-positive block finding on compressed data causes crashes

**New approach: Hybrid libdeflate+parallel**

Instead of pure-Rust speculative decode, use a strategy that leverages our fast libdeflate FFI:

1. **Sequential scan with libdeflate**: Decompress the file sequentially with libdeflate FFI (our fastest path), but record block boundary positions and 32KB windows at fixed output intervals
2. **Parallel re-verify**: On subsequent decompressions of the same file (cache the boundary map), use the boundaries for parallel libdeflate inflate
3. **OR**: Accept the libdeflate FFI sequential path is fast enough and focus resources on Problem 1

**Alternative: Window-propagation parallelism**
- Split compressed data into N chunks at byte boundaries
- Chunk 0 starts at the beginning (known state)
- Each chunk decodes sequentially with libdeflate FFI
- Window from chunk K propagates to chunk K+1 for back-reference resolution
- This only works if we can find valid deflate block starts, which is the hard part

**Expected impact**: 10-15% improvement on Tmax silesia, closing the gap to <5%

**Measurement**: `./gzippy-dev ci gaps` — track silesia-gzip/pigz Tmax rows

---

## Problem 3: BGZF/Multi-member T1 and Tmax (3-7% gap)

**The gap**: 3-7% behind rapidgzip on BGZF Tmax (x86), 17-22% on ARM64 T1

**Root cause**: The ARM64 T1 gap is really Problem 1 (general ARM64 T1 overhead). The x86 Tmax gap is thread scheduling and decompressor allocation overhead.

**Fix approaches**:
1. Solve Problem 1 first (fixes the ARM64 T1 component)
2. For x86 Tmax: reduce per-block overhead in `decompress_bgzf_parallel`
   - Pre-allocate decompressor pool instead of creating per-thread
   - Batch small BGZF blocks for single-call inflate

**Expected impact**: Once Problem 1 is solved, most remaining gaps should be <3%

---

## Execution Plan

### Phase 1: ARM64 T1 (Week 1)
1. Create branch `perf/arm64-t1-fast-path`
2. Instrument decompression timing
3. Push, `./gzippy-dev ci watch`, analyze
4. Implement fast-path, push, watch
5. Repeat until ARM64 T1 gaps < 5%

### Phase 2: Single-Member Parallel (Week 2-3)
1. Decide: speculative vs. hybrid vs. accept sequential
2. Implement chosen approach
3. Verify correctness on all datasets
4. Benchmark on Tmax

### Phase 3: Polish (Week 4)
1. Close remaining BGZF Tmax gap
2. Verify all 123 comparisons are wins or within noise (<3%)
3. Update CI thresholds to enforce wins

### Ongoing: Use `gzippy-dev` in Every PR

```bash
# Before starting work
./gzippy-dev ci gaps --branch main

# After pushing
./gzippy-dev ci watch

# Verify gaps closed
./gzippy-dev ci gaps
```

---

## Success Criteria

**Goal**: Zero gaps >5% across all 123 comparisons.

| Metric | Current | Target |
|--------|---------|--------|
| Total gaps | 52 | <10 |
| Gaps >10% | 14 | 0 |
| Gaps >5% | 24 | 0 |
| Gaps >3% | 29 | <5 |
| Total wins | 71 | >110 |
