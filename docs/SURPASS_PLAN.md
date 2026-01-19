# Plan to Surpass libdeflate and rapidgzip

## Executive Summary

**Goal**: Pure Rust gzippy that beats:
- libdeflate single-threaded performance (~1232 MB/s → 1400+ MB/s)
- rapidgzip parallel performance on ALL file types and thread counts

## Analysis of Competitors

### libdeflate Key Optimizations (Single-Threaded)

From `decompress_template.h`:

1. **Packed Decode Table Entry** (Critical)
   - Entry format: bits 0-7 = code len, bits 8-15 = flags/extra, bits 16-31 = symbol/length
   - Single u32 contains everything needed for decode
   - Flag bits: `HUFFDEC_LITERAL`, `HUFFDEC_END_OF_BLOCK`, `HUFFDEC_EXCEPTIONAL`, `HUFFDEC_SUBTABLE_POINTER`

2. **Multi-Literal Fast Path** (Critical)
   - Decodes up to 3 literals per loop iteration
   - `entry >> 16` extracts literal, same entry tells code length
   - Reduces loop overhead by 3x for literal-heavy data

3. **Preloading** (Important)
   - Next table entry preloaded BEFORE consuming current
   - `entry = d->litlen_decode_table[bitbuf & litlen_tablemask]` happens early
   - Hides memory latency

4. **Saved Bitbuf for Extra Bits** (Important)
   - `saved_bitbuf = bitbuf` before consuming code
   - Extra bits extracted from saved value: `EXTRACT_VARBITS8(saved_bitbuf, entry)`
   - Avoids separate read operation

5. **Subtable for Long Codes** (Important)
   - Main table: 11 bits (2048 entries, 8KB)
   - Subtables for codes > 11 bits
   - `entry = d->litlen_decode_table[(entry >> 16) + EXTRACT_VARBITS(bitbuf, (entry >> 8) & 0x3F)]`

6. **Fast LZ77 Copy** (Important)
   - Word-at-a-time copies instead of byte-at-a-time
   - Uses `memcpy` for large distances
   - Special case for overlapping copies

7. **BMI2 on x86_64** (Architecture-specific)
   - Uses `pdep`/`pext` for faster bit extraction
   - Compiled with multiple dispatch

### rapidgzip Key Optimizations (Parallel)

From `deflate.hpp` and `HuffmanCodingShortBitsCachedDeflate.hpp`:

1. **CacheEntry with Pre-decoded Length+Distance** (Critical)
   ```cpp
   struct CacheEntry {
       uint8_t bitsToSkip;     // Total bits consumed
       uint8_t symbolOrLength; // For lengths: length-3
       uint16_t distance;      // 0=literal, 0xFFFF=EOB, 0xFFFE=slow path
   };
   ```
   - Single lookup decodes literal/length/distance/extra bits
   - 11-bit LUT is optimal (from their benchmarks)

2. **Marker-Based Speculative Decoding** (Critical for Parallel)
   - Uses `uint16_t` output buffer (values 0-255 = bytes, 256+ = markers)
   - Markers encode unresolved back-references
   - Allows decoding from arbitrary positions without waiting for window

3. **Chunk-Based Parallelism** (Critical for Parallel)
   - Divides compressed stream into chunks
   - Each chunk decoded speculatively with markers
   - Window propagated sequentially, marker replacement in parallel

4. **Block Finder with False Positive Tolerance** (Important)
   - Tests deflate block candidates at many positions
   - Validates via precode/Huffman table construction
   - Falls back gracefully on false positives

5. **ISA-L Integration** (for comparison baseline)
   - Uses ISA-L when available for fastest single-threaded decode
   - Pure Rust fallback without ISA-L is 330 MB/s vs ISA-L's 720 MB/s

## Our Implementation Plan

### Phase 1: Single-Threaded Parity with libdeflate (Target: 1400 MB/s)

#### 1.1 Packed Decode Table Entry (Priority: CRITICAL)

Implement libdeflate-style packed entries:

```rust
// Entry format (32 bits):
// Bits 0-7:   Code length to consume
// Bits 8-9:   Entry type (0=literal, 1=length, 2=EOB, 3=subtable)
// Bits 10-15: Extra bits count (for lengths) OR subtable index (for subtables)
// Bits 16-31: Symbol value OR base length

pub struct PackedEntry(u32);

impl PackedEntry {
    #[inline(always)]
    fn code_len(self) -> u32 { self.0 & 0xFF }
    
    #[inline(always)]
    fn is_literal(self) -> bool { (self.0 & 0x300) == 0 }
    
    #[inline(always)]
    fn literal_byte(self) -> u8 { (self.0 >> 16) as u8 }
    
    #[inline(always)]
    fn length_base(self) -> u16 { (self.0 >> 16) as u16 }
    
    #[inline(always)]
    fn extra_bits(self) -> u32 { (self.0 >> 10) & 0x3F }
}
```

**Estimated Impact**: 20-30% speedup

#### 1.2 Multi-Literal Decode Loop (Priority: CRITICAL)

```rust
loop {
    bits.ensure(48);  // Enough for 3 literals + length + offset
    
    let entry = table[bits.peek() & TABLE_MASK];
    bits.consume(entry.code_len());
    
    if entry.is_literal() {
        *out_ptr = entry.literal_byte();
        out_ptr = out_ptr.add(1);
        
        // Try 2nd literal
        let entry2 = table[bits.peek() & TABLE_MASK];
        if entry2.is_literal() {
            bits.consume(entry2.code_len());
            *out_ptr = entry2.literal_byte();
            out_ptr = out_ptr.add(1);
            
            // Try 3rd literal
            let entry3 = table[bits.peek() & TABLE_MASK];
            if entry3.is_literal() {
                bits.consume(entry3.code_len());
                *out_ptr = entry3.literal_byte();
                out_ptr = out_ptr.add(1);
            }
        }
        continue;
    }
    // ... handle length/EOB
}
```

**Estimated Impact**: 30-40% speedup for literal-heavy data

#### 1.3 Combined Length+Distance Lookup (Priority: HIGH)

Like rapidgzip's `CacheEntry`, pre-compute full LZ77 matches:

```rust
// 12-bit LUT (4096 entries) for combined decode
// For lengths with short distance codes, entire match is in one lookup
pub struct CombinedEntry {
    bits_to_skip: u8,   // Total bits consumed
    length: u8,         // Length - 3 (0-255 maps to 3-258)
    distance: u16,      // 0 = literal, 0xFFFF = EOB, 0xFFFE = slow path
}
```

Build table with distance codes inlined where they fit:

```rust
// For length code 257 (length 3) with 5-bit distance code:
// If code_len + dist_code_len + dist_extra_bits <= 12, inline it
for len_code in 257..=285 {
    for dist_code in 0..30 {
        let total_bits = lit_len_code_len + len_extra + dist_code_len + dist_extra;
        if total_bits <= 12 {
            // Pre-compute and store in table
        }
    }
}
```

**Estimated Impact**: 50% speedup by eliminating distance table lookup

#### 1.4 Subtable for Long Codes (Priority: MEDIUM)

Handle codes > 11 bits with subtables instead of fallback:

```rust
const MAIN_BITS: usize = 11;
const MAX_SUBTABLE_BITS: usize = 4;  // Codes up to 15 bits

// Main table entry points to subtable for long codes
// Subtable index stored in entry when EXCEPTIONAL flag set
```

**Estimated Impact**: 5-10% speedup for dynamic blocks with long codes

#### 1.5 Fast LZ77 Copy with Architecture-Specific SIMD (Priority: HIGH)

Already implemented, but needs tuning:

```rust
// For distance >= 8, use 8-byte unaligned loads/stores
#[cfg(target_arch = "x86_64")]
unsafe fn copy_fast(dst: *mut u8, src: *const u8, len: usize) {
    if len <= 16 {
        // Single SIMD load/store
        let v = _mm_loadu_si128(src as *const __m128i);
        _mm_storeu_si128(dst as *mut __m128i, v);
    } else {
        // Loop with 32-byte chunks
    }
}
```

**Estimated Impact**: Already good, minor tuning possible

### Phase 2: Parallel Decompression to Beat rapidgzip (Target: 3500+ MB/s)

#### 2.1 Marker-Based Speculative Decoder (Priority: CRITICAL)

Already started in `marker_decode.rs`, needs completion:

```rust
pub struct MarkerDecoder<'a> {
    bits: FastBits<'a>,
    output: Vec<u16>,  // u16: 0-255 = byte, 256+ = marker
    output_bytes: usize,
    markers_pending: Vec<PendingMarker>,
}

struct PendingMarker {
    output_pos: usize,
    distance: u16,
    length: u16,
}

impl MarkerDecoder {
    /// Decode until max_output bytes (with markers counting as resolved)
    pub fn decode_chunk(&mut self, max_output: usize) -> Result<ChunkResult>;
    
    /// Replace markers given the 32KB window from previous chunk
    pub fn resolve_markers(&mut self, window: &[u8]) -> Vec<u8>;
}
```

**Key insight**: rapidgzip uses `uint16_t` for output initially, then converts to `uint8_t` after marker replacement. This doubles memory but enables parallelism.

#### 2.2 Chunk Partitioning Strategy (Priority: HIGH)

```rust
const CHUNK_SIZE: usize = 4 * 1024 * 1024;  // 4 MB chunks

fn partition_for_parallel(data: &[u8], num_threads: usize) -> Vec<ChunkInfo> {
    let mut chunks = Vec::new();
    let chunk_spacing = data.len() / num_threads;
    
    for i in 0..num_threads {
        let start_bit = i * chunk_spacing * 8;
        chunks.push(ChunkInfo {
            start_bit_offset: start_bit,
            estimated_output_size: CHUNK_SIZE,
        });
    }
    chunks
}
```

#### 2.3 Parallel Decode Pipeline (Priority: CRITICAL)

```rust
fn decompress_parallel(data: &[u8], num_threads: usize) -> Result<Vec<u8>> {
    let chunks = partition_for_parallel(data, num_threads);
    
    // Stage 1: Parallel speculative decode with markers
    let chunk_results: Vec<ChunkResult> = chunks
        .par_iter()
        .map(|chunk| {
            let mut decoder = MarkerDecoder::new(&data[chunk.start..]);
            decoder.decode_chunk(CHUNK_SIZE)
        })
        .collect();
    
    // Stage 2: Sequential window propagation
    let mut windows = Vec::new();
    let mut prev_window = [0u8; 32768];
    for result in &chunk_results {
        let resolved = result.resolve_markers(&prev_window);
        prev_window.copy_from_slice(&resolved[resolved.len()-32768..]);
        windows.push(resolved);
    }
    
    // Stage 3: Parallel final output assembly (already done in Stage 2)
    let total_len: usize = windows.iter().map(|w| w.len()).sum();
    let mut output = Vec::with_capacity(total_len);
    for window in windows {
        output.extend_from_slice(&window);
    }
    
    Ok(output)
}
```

#### 2.4 BGZF Fast Path (Priority: HIGH)

BGZF files have embedded block markers - exploit these:

```rust
fn decompress_bgzf_parallel(data: &[u8]) -> Result<Vec<u8>> {
    // Parse BGZF headers to find block boundaries
    let blocks = find_bgzf_blocks(data);
    
    // Each block is independent - no markers needed!
    let results: Vec<Vec<u8>> = blocks
        .par_iter()
        .map(|block| decompress_block_fast(block))
        .collect();
    
    // Concatenate results
    results.into_iter().flatten().collect()
}
```

**Target**: 3500+ MB/s (each block decompresses independently)

### Phase 3: Optimizations Beyond rapidgzip

#### 3.1 AVX-512 LZ77 Copy (x86_64 only)

```rust
#[cfg(target_feature = "avx512f")]
unsafe fn copy_avx512(dst: *mut u8, src: *const u8, len: usize) {
    // 64-byte copies for large lengths
}
```

#### 3.2 Prefetching in Decode Loop

```rust
// Prefetch next table entries while processing current
#[cfg(target_arch = "x86_64")]
unsafe fn decode_with_prefetch(...) {
    let entry = table[bits.peek() & MASK];
    std::arch::x86_64::_mm_prefetch(
        &table[(bits.peek() >> entry.code_len()) & MASK] as *const _ as *const i8,
        std::arch::x86_64::_MM_HINT_T0
    );
    // ... process entry
}
```

#### 3.3 Lock-Free Chunk Output

```rust
// Pre-allocate output buffer, write chunks in place
fn decompress_parallel_lockfree(data: &[u8]) -> Result<Vec<u8>> {
    let output = Arc::new(UnsafeCell::new(vec![0u8; estimated_size]));
    
    chunks.par_iter().for_each(|chunk| {
        let out_slice = unsafe { 
            &mut (*output.get())[chunk.output_offset..] 
        };
        decode_into(chunk, out_slice);
    });
    
    Arc::try_unwrap(output).unwrap().into_inner()
}
```

## Implementation Order

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | Combined length+distance LUT | +50% | Medium |
| 2 | Packed decode table entry | +30% | Medium |
| 3 | Multi-literal decode loop | +30% | Low |
| 4 | Marker-based parallel decoder | Parallel | High |
| 5 | BGZF fast path | 3500 MB/s | Medium |
| 6 | Subtable for long codes | +10% | Low |
| 7 | AVX-512 / prefetching | +5% | Medium |

## Success Metrics

| Metric | Current | Target | Baseline |
|--------|---------|--------|----------|
| Single-threaded inflate | 596 MB/s | 1400 MB/s | libdeflate: 1232 MB/s |
| Multi-member parallel | 580 MB/s | 1200 MB/s | rapidgzip: 1000 MB/s |
| Single-member parallel (8 threads) | N/A | 2500 MB/s | rapidgzip: 2200 MB/s |
| BGZF parallel (8 threads) | N/A | 3500 MB/s | rapidgzip: 3168 MB/s |

## Files to Modify

1. `src/ultra_fast_inflate.rs` - Main decode loop optimization
2. `src/two_level_table.rs` - Packed entry format + subtables
3. `src/combined_lut.rs` - Combined length+distance LUT
4. `src/turbo_decode.rs` - New optimized decode loop
5. `src/marker_decode.rs` - Parallel marker-based decoder
6. `src/parallel_decompress.rs` - Parallel pipeline orchestration

## Testing Strategy

1. Unit tests for each optimization
2. Byte-by-byte verification against libdeflate output
3. Performance microbenchmarks after each change
4. Full regression tests on Silesia corpus
