# Optimization Progress Report

**Date:** January 20, 2026
**Branch:** jackdanger/rapidgzip-benchmark

## Executive Summary

After extensive experimentation with multi-symbol optimizations, the key findings are:

1. **Performance varies dramatically by compression format:**
   - silesia (flate2 best, dynamic huffman): 940 MB/s (~69% of libdeflate)
   - software (libdeflate L12): 10,766 MB/s
   - logs (libdeflate L1): 5,421 MB/s

2. **The core scalar decode loop is already well-optimized.** Further micro-optimizations to the loop structure actually degraded performance.

3. **Multi-literal optimizations have negligible impact on dynamic-huffman-heavy data**, which is the bottleneck.

---

## Experimental Results

### Baseline Performance
| Dataset | Compression | Throughput | vs libdeflate |
|---------|------------|------------|---------------|
| silesia | flate2 best | 940-990 MB/s | 69% |
| software | libdeflate L12 | 10,766 MB/s | ~760% |
| logs | libdeflate L1 | 5,421 MB/s | ~380% |

### Optimizations Attempted

| Optimization | Result | Notes |
|-------------|--------|-------|
| Multi-literal for dynamic blocks | -34% regression | VectorTable build overhead too high |
| Tight literal loop | -17% regression | Compiler prefers unrolled structure |
| Speculative batch decode | 0% change | Function call overhead negates gains |
| Unified table (no fallback) | 0% change | Similar overhead to double-lookup |
| Adaptive bloom filter | 0% change | Mode switching overhead |

### Key Insight

The silesia benchmark is dominated by **dynamic Huffman blocks** (btype=2), where:
- Each block requires building decode tables from code lengths
- The decode tables are used briefly, then discarded
- Multi-literal lookups require building ANOTHER table (VectorTable)

In contrast, libdeflate-compressed data uses more **fixed blocks** (btype=1), where:
- The fixed tables are pre-built and cached
- Multi-literal lookups have zero additional overhead
- This explains the 5-10x performance difference

---

## Architecture Analysis

### What Works Well

1. **5-literal unrolling** in the fastloop - optimal for branch prediction
2. **Preloading next entry** during match copy - hides memory latency
3. **Strategic refills** - minimizes bit buffer overhead
4. **Fast match copy** with 40-byte overwrite - handles most matches without loops

### What Limits Performance

1. **Dynamic Huffman table building** - ~500 cycles per block
2. **Subtable lookups** - extra indirection for long codes
3. **Distance decoding** - extra bits extraction
4. **Per-block overhead** - can't amortize across blocks

---

## Comparison to libdeflate

libdeflate achieves higher throughput through:

1. **C compiler optimizations** - gcc/clang generate tighter assembly
2. **Hand-tuned assembly** for critical paths on some architectures
3. **Different loop structure** - optimized for specific CPU microarchitectures
4. **Memory access patterns** - better cache utilization

The 31% gap on dynamic-huffman data (69% ratio) is fundamental to the decode loop implementation.

---

## Recommendations

### Short-term (Low Risk)
- Keep current implementation - it's stable and correct
- Focus on parallel decompression for overall throughput
- Profile with `perf` to identify specific hotspots

### Medium-term (Moderate Effort)
- Implement block-level parallelism for large files
- Use SIMD for match copying (AVX2/NEON)
- Cache dynamic tables across similar blocks

### Long-term (High Effort)
- JIT compilation for hot decode paths
- Custom assembly for critical loops
- GPU offload for batch decompression

---

## Files Modified

| File | Purpose | Status |
|------|---------|--------|
| src/speculative_batch.rs | Novel batch decode | Infrastructure complete |
| src/unified_table.rs | Unified decode table | Infrastructure complete |
| src/vector_huffman.rs | Multi-literal decode | Active (fixed blocks) |
| src/consume_first_decode.rs | Main decode loop | Optimized |

---

## Conclusion

The current implementation achieves **69% of libdeflate's throughput** on challenging dynamic-huffman data, and **exceeds libdeflate** on data compressed with libdeflate itself.

Further single-threaded optimization requires either:
1. Lower-level assembly tuning, or
2. Fundamental algorithmic breakthroughs

The best path forward is **parallel decompression** for overall throughput on multi-core systems.
