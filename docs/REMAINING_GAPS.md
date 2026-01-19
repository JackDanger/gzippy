# Exhaustive Remaining Optimization Gaps

**Date**: January 2026  
**Status**: After P0/P1 fixes, ultra_fast_inflate is 4% faster than libdeflate (19,421 MB/s vs 18,661 MB/s)

---

## Summary

| Category | Status | Gap | Priority |
|----------|--------|-----|----------|
| Single-member decompression | ✅ **FASTER than libdeflate** | None | Done |
| BGZF parallel decompression | ✅ 3400+ MB/s | None | Done |
| Multi-member parallel | ⚠️ Sequential fallback | ~2x vs rapidgzip | P1 |
| Arbitrary gzip parallel | ⚠️ Partial | ~3x vs rapidgzip | P1 |
| Compression | ✅ Beats pigz | None | Done |
| Memory usage | ⚠️ Some overhead | 5-10% | P3 |

---

## Decompression Gaps

### Gap 1: Multi-Member Parallel Decompression (P1)
**Current**: Falls back to sequential for multi-member gzip (pigz output)  
**Target**: Parallel per-member decompression like rapidgzip

**What's Missing**:
- Member boundary detection is in `is_multi_member_quick()` but only scans 256KB
- `ultra_decompress.rs` has parallel member decompression but often falls back
- Need to find all member boundaries first, then decompress each in parallel

**Expected Gain**: 2-4x on pigz-compressed files (14 threads)

**Files**:
- `src/decompression.rs` - main dispatcher
- `src/ultra_decompress.rs` - parallel member logic

---

### Gap 2: Arbitrary Gzip Speculative Parallel (P1)
**Current**: `marker_decode.rs` and `rapidgzip_decoder.rs` exist but aren't fully integrated  
**Target**: Match rapidgzip on single-member gzip files

**What's Missing**:
- Chunk spacing strategy (guess positions, not find blocks)
- Window propagation through chunk chain
- Parallel marker replacement after windows are known
- Full integration into main decompression path

**Expected Gain**: 2-3x on gzip (not pigz, not gzippy) files

**Files**:
- `src/marker_decode.rs` - core marker algorithm (implemented)
- `src/rapidgzip_decoder.rs` - speculative decoder (implemented)
- `src/block_finder.rs` - block finding (may not be needed)
- `src/decompression.rs` - not calling these paths

---

### Gap 3: Dynamic Block Multi-Symbol Decode (P2)
**Current**: Single symbol per lookup in dynamic blocks  
**Target**: 2-3 symbols per lookup when codes are short (3-5 bits)

**What's Missing**:
- Runtime multi-symbol table generation for dynamic blocks
- `turbo_inflate.rs` has MULTI_SYM_LIT_TABLE for fixed, not dynamic

**Expected Gain**: 15-25% on files with many dynamic blocks

**Files**:
- `src/turbo_inflate.rs` - has multi-sym for fixed only
- `src/ultra_fast_inflate.rs` - uses two-level, no multi-sym

---

### Gap 4: Table Construction Optimization (P2)
**Current**: Heap-allocate Vec for each table, zero-fill 1024+ entries  
**Target**: Stack-allocate L1, lazy L2

**What's Missing**:
```rust
// Current (in two_level_table.rs)
let mut table = Self::new();  // 2KB zero-fill
table.l2 = Vec::new();        // heap allocation

// Optimal
let mut l1 = [0u16; 1024];  // stack, no heap
// Only allocate L2 if max_len > 10
```

**Expected Gain**: 3-5% on files with many dynamic blocks

**Files**:
- `src/two_level_table.rs` - `TwoLevelTable::build()`

---

### Gap 5: Prefetching (P3)
**Current**: Basic prefetching in `simd_inflate.rs`  
**Target**: Aggressive prefetching like ISA-L

**What's Missing**:
```rust
// Prefetch next input bytes during decode
std::arch::x86_64::_mm_prefetch(input.add(128), _MM_HINT_T0);

// Prefetch output during LZ77 copy
std::arch::x86_64::_mm_prefetch(output.add(64), _MM_HINT_T0);
```

**Expected Gain**: 2-5% on large files

**Files**:
- `src/simd_copy.rs` - LZ77 copy paths
- `src/ultra_fast_inflate.rs` - main decode loop

---

### Gap 6: AVX-512 Support (P3)
**Current**: AVX2 (32-byte) and NEON (16-byte)  
**Target**: AVX-512 (64-byte) for newer Intel/AMD CPUs

**What's Missing**:
```rust
#[cfg(target_feature = "avx512f")]
unsafe fn copy_64_avx512(src: *const u8, dst: *mut u8) {
    let data = _mm512_loadu_si512(src as *const __m512i);
    _mm512_storeu_si512(dst as *mut __m512i, data);
}
```

**Expected Gain**: 5-10% on AVX-512 capable CPUs

**Files**:
- `src/simd_copy.rs` - needs avx512 module

---

### Gap 7: Branch Hints (P3)
**Current**: No explicit branch hints  
**Target**: Use likely/unlikely for hot paths

**What's Missing**:
```rust
#[cold]
fn handle_error() { ... }

if std::intrinsics::likely(symbol < 256) {
    output.push(symbol as u8);
} else if std::intrinsics::unlikely(symbol > 285) {
    return handle_error();
}
```

**Expected Gain**: 1-3%

---

## Compression Gaps

### Gap 8: Ultra Compression L10-L12 Parallelism (P2)
**Current**: L10-L12 use libdeflate but compression is slower  
**Target**: Parallel compression for L10-L12 with independent blocks

**What's Missing**:
- Current L10-L12 compresses sequentially
- Should use same parallel block strategy as L1-L5

**Expected Gain**: 4-8x compression speed for L10-L12

**Files**:
- `src/parallel_compress.rs` - parallel compression engine
- `src/compression.rs` - level dispatcher

---

### Gap 9: Compression Memory Pool (P3)
**Current**: Each compression block allocates its own buffers  
**Target**: Pre-allocated buffer pool to reduce allocations

**What's Missing**:
```rust
struct BufferPool {
    buffers: Vec<Vec<u8>>,
    available: AtomicBitmap,
}
```

**Expected Gain**: 2-5% compression throughput

---

## Memory/Resource Gaps

### Gap 10: Memory Mapping Optimization (P3)
**Current**: `memmap2` for large files  
**Target**: Huge page support, MAP_POPULATE hints

**What's Missing**:
```rust
// Use huge pages when available
let mmap = MmapOptions::new()
    .huge(Some(HugePages::Size2MB))
    .map(&file)?;
```

**Expected Gain**: 5-10% on very large files

---

### Gap 11: Thread Affinity (P3)
**Current**: No thread pinning  
**Target**: Pin threads to cores for cache locality

**What's Missing**:
```rust
core_affinity::set_for_current(core_id);
```

**Expected Gain**: 2-5% on NUMA systems

**Files**:
- `src/parallel_compress.rs`
- `src/ultra_decompress.rs`

---

## Integration Gaps

### Gap 12: Streaming Decompression (P2)
**Current**: Memory-maps entire file  
**Target**: True streaming for stdin and pipes

**What's Missing**:
- Current stdin path uses flate2 streaming, not ultra_fast_inflate
- Need chunked streaming path using ultra_fast_inflate

**Expected Gain**: Better latency for pipes

**Files**:
- `src/decompression.rs` - `decompress_stdin()`

---

### Gap 13: Index Caching (P3)
**Current**: No persistent index  
**Target**: Cache block indexes for repeated decompression

**What's Missing**:
- `index_cache.rs` was deleted (was dead code)
- rapidgzip saves `.gzidx` files for fast re-decompression

**Expected Gain**: Near-instant startup for repeated decompressions

---

### Gap 14: CRC32 SIMD (P3)
**Current**: Uses flate2/zlib-ng for CRC  
**Target**: Ensure SIMD CRC32 is used

**Status**: Likely already optimized via zlib-ng, need to verify

**What to Check**:
- Is zlib-ng compiled with SIMD CRC?
- Could use `crc32fast` crate as alternative

---

## CLI/UX Gaps

### Gap 15: Progress Reporting (P3)
**Current**: Simple output  
**Target**: Progress bar for large files

**What's Missing**:
```rust
[gzippy] Decompressing: [████████████░░░░░░░░] 60% (600MB/1GB) 2.5 GB/s
```

---

### Gap 16: Rsyncable Output (P3)
**Current**: Not implemented  
**Target**: Match pigz --rsyncable

**What's Missing**:
- Reset block boundaries on content-defined checkpoints
- Allows rsync to efficiently sync compressed files

---

## Priority Summary

| Priority | Gap | Expected Gain | Effort |
|----------|-----|---------------|--------|
| **P1** | Multi-member parallel | 2-4x | 1 week |
| **P1** | Arbitrary gzip parallel | 2-3x | 2 weeks |
| **P2** | Dynamic multi-symbol | 15-25% | 3 days |
| **P2** | Table construction | 3-5% | 1 day |
| **P2** | L10-L12 parallel | 4-8x compress | 2 days |
| **P2** | Streaming decompress | Better UX | 2 days |
| **P3** | Prefetching | 2-5% | 1 day |
| **P3** | AVX-512 | 5-10% | 2 days |
| **P3** | Branch hints | 1-3% | 0.5 days |
| **P3** | Memory pool | 2-5% | 1 day |
| **P3** | Huge pages | 5-10% | 1 day |
| **P3** | Thread affinity | 2-5% | 0.5 days |
| **P3** | Index caching | Startup time | 2 days |
| **P3** | CRC32 verify | Verify | 0.5 days |
| **P3** | Progress bar | UX | 1 day |
| **P3** | Rsyncable | Feature | 3 days |

---

## What's Done (Completed Optimizations)

| Optimization | Status | Location |
|--------------|--------|----------|
| Two-level Huffman tables | ✅ Complete | `two_level_table.rs` |
| SIMD pattern expansion | ✅ Complete | `simd_copy.rs` |
| SIMD overlapping copy | ✅ Complete | `simd_copy.rs` |
| AVX2/NEON copy | ✅ Complete | `simd_copy.rs` |
| 64-bit bit buffer | ✅ Complete | `two_level_table.rs` |
| BGZF parallel | ✅ Complete | `ultra_inflate.rs` |
| Parallel compression | ✅ Complete | `parallel_compress.rs` |
| libdeflate integration | ✅ Complete | `libdeflate_ext.rs` |
| Fixed Huffman multi-sym | ✅ Complete | `turbo_inflate.rs` |
| Cache-aligned buffers | ✅ Complete | `decompression.rs` |
| Thread-local decompressor | ✅ Complete | `decompression.rs` |

---

## Recommended Next Steps

### Immediate (This Week)
1. Integrate speculative parallel for arbitrary gzip files
2. Enable parallel member decompression for pigz files

### Short Term (This Month)
3. Dynamic block multi-symbol decode
4. Table construction optimization
5. L10-L12 parallel compression

### Long Term
6. All P3 items as time permits
