# Optimization Roadmap: Closing the Gap with libdeflate

## Current Gap Analysis

| Metric | gzippy | libdeflate | Gap |
|--------|--------|------------|-----|
| Simple data | 16,835 MB/s | 27,192 MB/s | **38%** |
| Complex data | 558 MB/s | 1,226 MB/s | **55%** |
| BGZF parallel (8T) | 3,607 MB/s | N/A | **Advantage** |

## Root Causes of the Gap

### 1. Entry Consumption Pattern
- **libdeflate**: `bitsleft -= entry` (subtracts full u32, ignores high bits)
- **gzippy**: `bits.consume(entry & 0xFF)` (extracts bits, then subtracts)
- **Impact**: 1 extra instruction per symbol

### 2. Preload Before Copy
- **libdeflate**: Preloads next table entry BEFORE starting match copy
- **gzippy**: Preloads nothing, sequential dependency chain
- **Impact**: ~3-5 cycles hidden latency per match

### 3. Branchless Bit Refill
- **libdeflate**: `bitbuf |= load(in_next) << bitsleft; in_next += (64-bitsleft)/8`
- **gzippy**: Conditional refill with `if bits < n`
- **Impact**: Branch misprediction penalty

### 4. Word-at-a-Time Match Copy
- **libdeflate**: Unconditionally copies 5 words (40 bytes), handles overrun
- **gzippy**: Checks length, copies exact bytes with SIMD for large
- **Impact**: Extra branches, less predictable

### 5. Multi-Literal Decode Path
- **libdeflate**: Decodes 2 literals inline before checking for match
- **gzippy**: Single literal then loop back
- **Impact**: More loop iterations on literal-heavy data

### 6. Garbage in High Bits Trick
- **libdeflate**: Allows garbage in bits 8+ of `bitsleft`, reduces masking
- **gzippy**: Always maintains clean `bits` count
- **Impact**: 1 masking operation per refill

---

## Every Possible Optimization (Exhaustive)

### A. Bit Buffer Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| A1 | `bitsleft -= entry` (full u32 subtract) | Low | 2-3% |
| A2 | Allow garbage in high bits of `bits` | Medium | 1-2% |
| A3 | Branchless refill (always load word, adjust pointer) | Medium | 3-5% |
| A4 | BMI2 `pext` for variable-width extraction | Low | 1-2% |
| A5 | Inline assembly for critical bit operations | High | 2-5% |

### B. Table Lookup Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| B1 | Preload next entry before match copy | Low | 5-10% |
| B2 | Combined litlen+offset table (single lookup) | High | 3-5% |
| B3 | 13-bit primary table (vs 12-bit) | Low | 1-2% |
| B4 | Aligned table allocation (64-byte) | Low | 1-2% |
| B5 | `vpgatherdd` SIMD for 4 parallel lookups | Very High | 10-20% |

### C. Decode Loop Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| C1 | 2 literals inline before match check | Medium | 5-10% |
| C2 | 4 literals unrolled (libdeflate does 2) | Medium | 3-5% |
| C3 | Jump table instead of if-else chain | Medium | 2-3% |
| C4 | Computed goto (C extension) via inline asm | Very High | 3-5% |
| C5 | Profile-guided optimization (PGO) | Low | 5-10% |
| C6 | Link-time optimization (LTO) | Low | 3-5% |

### D. Match Copy Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| D1 | Unconditional 40-byte copy | Low | 3-5% |
| D2 | RLE special case (offset=1) with memset | Low | 2-3% |
| D3 | Small offset (2-7) pattern broadcast | Medium | 2-3% |
| D4 | AVX-512 copy for large matches | Medium | 1-2% |
| D5 | Non-temporal stores for huge matches | Medium | 1-2% |
| D6 | Prefetch destination cache line | Low | 1-2% |

### E. Architecture-Specific Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| E1 | x86_64 BMI2 path (bzhi, pext) | Done | - |
| E2 | ARM NEON vectorized copy | Medium | 5-10% |
| E3 | ARM CRC32 intrinsics | Low | 2-3% |
| E4 | RISC-V vector extensions | High | 10-20% |
| E5 | Apple Silicon AMX matrix ops | Very High | Unknown |
| E6 | Separate binary per µarch (Zen4, Skylake, M1) | High | 5-15% |

### F. Memory Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| F1 | Huge pages for large buffers | Medium | 2-5% |
| F2 | Memory-mapped I/O | Medium | 2-5% |
| F3 | Double-buffering with async prefetch | High | 3-5% |
| F4 | NUMA-aware allocation | Medium | 5-10% |
| F5 | Stack-allocated small tables | Low | 1-2% |

### G. Parallelism Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| G1 | Lock-free output buffer (done) | Done | - |
| G2 | SIMD parallel Huffman decode | Very High | 20-50% |
| G3 | GPU offload via CUDA/Metal | Extreme | 100-500% |
| G4 | Speculative multi-start parallel | Done | - |
| G5 | Work-stealing thread pool | Medium | 5-10% |
| G6 | Persistent thread pool (no spawn cost) | Low | 2-3% |

### H. Compiler/Build Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| H1 | Profile-guided optimization (PGO) | Low | 5-15% |
| H2 | Fat LTO (link-time optimization) | Done | - |
| H3 | `#[inline(always)]` on hot paths | Done | - |
| H4 | `likely`/`unlikely` hints | Done | - |
| H5 | Codegen-units = 1 | Done | - |
| H6 | `-C target-cpu=native` for local builds | Low | 5-10% |

### I. Algorithmic Optimizations

| ID | Optimization | Complexity | Expected Gain |
|----|--------------|------------|---------------|
| I1 | Marker-based parallel for single-member | Done | - |
| I2 | Two-pass: find boundaries, then parallel decode | Done | - |
| I3 | Speculative decode with rollback | High | 5-10% |
| I4 | ML-based branch prediction hints | Extreme | Unknown |
| I5 | Adaptive table size based on entropy | High | 2-5% |

---

## Implementation Plan (Priority Order)

### Phase 1: Quick Wins (1-2 days, +15-25%)

```
1. A1: bitsleft -= entry (full u32 subtract)
   - Modify FastBits to store bits as u64, allow high bit garbage
   - Change consume() to subtract without masking

2. B1: Preload next entry before match copy
   - Load next_entry = table[bitbuf & mask] at start of match
   - Copy match while entry is in flight

3. A3: Branchless refill
   - Always: buf |= load(ptr) << bits
   - Always: ptr += (64 - bits) / 8
   - Remove conditional in refill()

4. D1: Unconditional 40-byte copy
   - Copy 5 words regardless of length
   - Rely on fastloop margin for safety
```

### Phase 2: Medium Effort (+10-20%)

```
5. C1: 2 literals inline before match check
   - Decode 2 literals at top of loop
   - Only check for non-literal after 2nd

6. G6: Persistent thread pool
   - Create threads once, reuse for all decompressions
   - Eliminate spawn overhead

7. H1: Profile-guided optimization
   - cargo pgo build with silesia corpus
   - Re-benchmark all paths

8. E2: ARM NEON copy optimization
   - Use ld1/st1 for 16-byte copies
   - Vectorized RLE expansion
```

### Phase 3: Deep Optimizations (+10-30%)

```
9. B5: SIMD parallel table lookup (AVX2/AVX-512)
   - vpgatherdd for 4-8 simultaneous lookups
   - Requires restructuring decode loop

10. G2: SIMD parallel Huffman decode
    - 8 streams decoded in parallel via AVX2
    - Merge outputs at end

11. E6: Microarchitecture-specific builds
    - Zen4, Skylake, Alderlake, M1/M2
    - dispatch at runtime via CPUID

12. C4: Computed goto via inline assembly
    - Jump table with pre-computed targets
    - Eliminates branch prediction overhead
```

### Phase 4: Experimental/Research (+?%)

```
13. G3: GPU offload
    - CUDA for NVIDIA, Metal for Apple
    - Effective for huge files (>100MB)

14. I4: ML branch prediction
    - Train model on common file patterns
    - Generate specialized decode paths

15. F4: NUMA-aware parallel
    - Pin threads to cores
    - Local memory allocation
```

---

## Success Criteria

| Phase | Target Speed (simple) | Target Speed (complex) |
|-------|----------------------|------------------------|
| Phase 1 | 22,000 MB/s (80%) | 800 MB/s (65%) |
| Phase 2 | 25,000 MB/s (92%) | 1,000 MB/s (82%) |
| Phase 3 | 30,000 MB/s (110%) | 1,300 MB/s (106%) |
| Phase 4 | 40,000+ MB/s | 1,500+ MB/s |

**Key Insight**: With 8+ threads, our parallel architecture already exceeds
libdeflate. The single-threaded gap is what we're closing here.
