# Optimization Implementation Checklist

## Phase 1: Decode Loop (Priority: Critical)

### 1.1 Multi-Symbol Huffman Decode
- [ ] Analyze ISA-L's `decode_next_lit_len` function
- [ ] Update `convert_isal_tables.py` to generate multi-symbol entries
  - [ ] Identify literal pairs that fit in 12-bit lookup
  - [ ] Pack: `[lit1:8][lit2:8][lit3:8][count:2][len:4]`
  - [ ] Handle edge cases (end of block, length codes)
- [ ] Regenerate `inflate_tables.rs` with multi-symbol entries
- [ ] Implement multi-symbol decode in `simd_inflate.rs`
  - [ ] Fast path: 3 literals in one lookup
  - [ ] Medium path: 2 literals
  - [ ] Slow path: single symbol (length code or long code)
- [ ] Add tests for multi-symbol decode correctness
- [ ] Benchmark improvement

### 1.2 Bitstream Optimization
- [ ] Change refill strategy: ensure 57+ bits before decode loop
- [ ] Batch multiple decodes between refills
- [ ] Use unaligned 64-bit reads where safe
- [ ] Profile: count refill calls per 1MB

### 1.3 Static Huffman Fast Path
- [ ] Pre-compute all 286 literal/length codes
- [ ] Pre-compute all 30 distance codes
- [ ] Direct lookup without length/distance table indirection
- [ ] Specialize decode loop for static blocks

## Phase 2: LZ77 Copy (Priority: High)

### 2.1 x86_64 AVX2 Copies
- [ ] Create `src/simd_copy.rs` module
- [ ] Implement `copy_32_avx2(src, dst)` with inline asm
- [ ] Implement `copy_64_avx2(src, dst)` for large copies
- [ ] Add runtime CPU feature detection
- [ ] Fallback to SSE2 if AVX2 not available
- [ ] Benchmark on different copy sizes

### 2.2 Pattern Expansion
- [ ] Distance 1: byte broadcast (`_mm256_set1_epi8`)
- [ ] Distance 2: word broadcast (`_mm256_set1_epi16`)
- [ ] Distance 4: dword broadcast (`_mm256_set1_epi32`)
- [ ] Distance 8: qword broadcast (`_mm256_set1_epi64x`)
- [ ] Distances 3,5,6,7: pattern shuffle
- [ ] Benchmark on repetitive data

### 2.3 Overlapping Copy Optimization
- [ ] For `distance < length && distance >= 32`: use overlapping AVX2
- [ ] For `distance < 32`: use byte-by-byte with unrolling
- [ ] Avoid branches in inner loop

### 2.4 ARM NEON Implementation
- [ ] `copy_16_neon(src, dst)` using `vld1q_u8`/`vst1q_u8`
- [ ] Pattern expansion for distances 1-8
- [ ] Benchmark on Apple Silicon

## Phase 3: Table Optimization (Priority: Medium)

### 3.1 Two-Level Tables
- [ ] Analyze code length distribution in typical files
- [ ] Implement Level 1: 10-bit direct lookup (1024 entries)
- [ ] Implement Level 2: overflow table for codes >10 bits
- [ ] Benchmark cache miss rate

### 3.2 Table Alignment
- [ ] Add `#[repr(C, align(64))]` to table structs
- [ ] Ensure hot tables fit in L1 cache (32KB)
- [ ] Profile cache behavior with `perf stat`

### 3.3 Table Packing
- [ ] Pack symbol + length into single u16 where possible
- [ ] Reduce table size by 50%
- [ ] Measure memory bandwidth improvement

## Phase 4: Memory Optimization (Priority: Medium)

### 4.1 Prefetching
- [ ] Add input prefetch 256 bytes ahead
- [ ] Add output prefetch 256 bytes ahead
- [ ] Prefetch lookup tables on first use
- [ ] Profile L1/L2 cache hit rate

### 4.2 Output Buffer
- [ ] Pre-allocate from ISIZE trailer
- [ ] Use `Vec::with_capacity` instead of `Vec::new`
- [ ] Avoid reallocation during decompression
- [ ] Measure allocation overhead

### 4.3 Branch Optimization
- [ ] Add `#[cold]` to error paths
- [ ] Add `likely`/`unlikely` hints (nightly)
- [ ] Reorder match arms by frequency
- [ ] Profile branch misprediction rate

## Phase 5: Parallel Decompression (Priority: High)

### 5.1 BGZF Optimization (Already done)
- [x] Parse BGZF blocks
- [x] Parallel decompress with work stealing
- [x] Write output in order
- [ ] Optimize block size for parallelism

### 5.2 Multi-Member (pigz output)
- [ ] Detect gzip member boundaries
- [ ] Parallel per-member decompression
- [ ] Handle false positives (gzip magic in data)

### 5.3 Single-Member Speculative
- [ ] Implement chunk spacing (every 4MB)
- [ ] Speculative decode with markers
- [ ] Window propagation between chunks
- [ ] Parallel marker resolution
- [ ] Benchmark vs rapidgzip

### 5.4 Block Index Caching
- [ ] Save block boundaries to `.gzidx` file
- [ ] Load index on subsequent decompressions
- [ ] Parallel decompress using cached boundaries

## Phase 6: Compression (Priority: Lower)

### 6.1 Match Finding
- [ ] 4-byte hash chains
- [ ] SIMD string comparison
- [ ] Optimal chain length per level

### 6.2 Lazy Matching
- [ ] Implement for levels 4-6
- [ ] Tune thresholds per level

### 6.3 Block Splitting
- [ ] Analyze entropy per block
- [ ] Split when entropy changes significantly

## Phase 7: Testing & Benchmarking (Priority: High)

### 7.1 Correctness Tests
- [ ] Round-trip all compression levels
- [ ] Cross-tool compatibility (gzip, pigz, zlib)
- [ ] Edge cases (empty, 1 byte, max size)
- [ ] Fuzzing with libFuzzer/AFL

### 7.2 Performance Tests
- [ ] Micro-benchmarks per function
- [ ] End-to-end benchmarks vs competitors
- [ ] Memory usage profiling
- [ ] CI performance regression detection

### 7.3 Test Data
- [ ] Calgary corpus
- [ ] Silesia corpus
- [ ] Synthetic patterns (zeros, random, text)
- [ ] Real-world samples (logs, DNA, binaries)

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| 1MB decompress | 1.11ms | <0.60ms | ❌ |
| BGZF decompress | ~3000 MB/s | 4000 MB/s | 🔄 |
| Arbitrary gzip (14T) | ~1000 MB/s | 2500 MB/s | ❌ |
| L1 compression ratio | matches pigz | matches | ✅ |
| L9 compression ratio | matches pigz | matches | ✅ |
| L12 compression ratio | near zopfli | near zopfli | ✅ |

## Weekly Milestones

**Week 1**: Multi-symbol decode working, 1.5x improvement
**Week 2**: AVX2 copies working, 1.7x improvement
**Week 3**: Pattern expansion + prefetch, 1.9x improvement
**Week 4**: Table optimization + branch hints, 2.0x improvement
**Week 5**: Parallel arbitrary gzip, 2500 MB/s multi-thread
**Week 6**: Compression improvements
**Week 7**: Polish and edge cases
**Week 8**: Final benchmarking and release
