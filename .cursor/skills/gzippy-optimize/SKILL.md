---
name: gzippy-optimize
description: Systematic optimization workflow for gzippy decompression and compression performance. Use when the user asks to close performance gaps, beat a competitor tool, improve throughput, or analyze why gzippy is slower in a specific scenario.
---

# Optimization Workflow

## Step 1: Quantify the Gap

Always start with data, never assumptions.

```bash
# CI gaps (production-like environment)
./gzippy-dev ci gaps --branch main

# Local measurement
./gzippy-dev bench --dataset silesia
```

## Step 2: Classify the Gap

| Gap Type | Signature | Solution Path |
|----------|-----------|---------------|
| T1 decompression | Slower at T1, any archive | Optimize decode loop in `consume_first_decode.rs` |
| Tmax decompression, BGZF | Slower at Tmax, bgzf archives | Optimize `bgzf.rs` parallel inflate |
| Tmax decompression, gzip/pigz | Slower at Tmax, single/multi-member | Implement speculative parallel (`speculative_parallel.rs`) |
| T1 compression | Slower at T1, any level | Optimize deflate (level-dependent) |
| Tmax compression | Slower at Tmax | Thread scheduling in compression pipeline |
| ARM-specific | Gap on arm64 only | Check NEON codegen, branch patterns |
| x86-specific | Gap on x86_64 only | Check BMI2, AVX2 usage |

## Step 3: Identify the Code Path

```bash
./gzippy-dev path benchmark_data/target-file.gz
```

Then read the identified source file to understand current implementation.

## Step 4: Change ONE Thing

**Rules from hard experience:**
- ONE change at a time (never batch)
- Benchmark BEFORE and AFTER on the same dataset
- If it regresses, revert immediately — don't try to "fix forward"
- Simulation benchmarks lie — only full-file benchmarks count

## Step 5: Verify

```bash
# Correctness
timeout 120 cargo test --release 2>&1 | tail -5

# Local performance
./gzippy-dev bench --dataset <relevant>

# Push and verify in CI
git push && ./gzippy-dev ci watch
```

## Historical Lessons (Don't Repeat These)

### Optimizations That HURT Performance
- DoubleLitCache per block (-73%)
- Table-free fixed Huffman (-325%)
- Unconditional refill (-12%)
- x86 2-3 literal batching (-20% on SOFTWARE)
- x86 5-word loop unroll (-10%)
- `#[cold]` on error paths (-4%)

### Optimizations That HELPED
- 8-literal batching (+significant)
- Packed writes (u16/u32/u64 stores)
- saved_bitbuf pattern
- bitsleft -= entry trick
- Branchless refill
- AVX2 match copy for large matches
- Unsafe unchecked table lookups (+5%)

### Key Architectural Insight
The speculative parallel decoder is the ONLY way to match rapidgzip on single-member files. Two-pass (sequential scan then parallel decode) is provably slower than just running libdeflate sequentially — the scan IS a full decode.

## Current Priority Gaps (from latest CI)

1. **Single-member Tmax**: No parallel decompression for gzip/pigz archives → `speculative_parallel.rs`
2. **ARM T1 decompression**: ~20% behind pigz/rapidgzip on software/logs → decode loop tuning
3. **BGZF T1**: Parallel detection overhead hurts T1 → fast-path bypass
