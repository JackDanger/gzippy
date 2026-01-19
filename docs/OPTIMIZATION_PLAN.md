# gzippy Optimization Plan: Matching and Surpassing libdeflate/ISA-L

This document outlines a comprehensive plan to fully match and exceed the performance of libdeflate and ISA-L in both compression and decompression.

## Current Status

| Metric | gzippy | libdeflate | ISA-L | Gap |
|--------|--------|------------|-------|-----|
| 1MB decompress | 1.11ms | 0.63ms | ~0.60ms | 1.76x slower |
| Multi-symbol decode | ❌ | ❌ | ✅ (2-3 symbols) | Missing |
| AVX2 copy | Partial | ✅ | ✅ | Partial |
| ARM NEON | Partial | ✅ | ✅ | Partial |
| Parallel BGZF | ✅ | ❌ | ❌ | Advantage |

## Phase 1: Decode Loop Optimizations (Week 1-2)

The decode loop is the #1 bottleneck. We need to match ISA-L's techniques.

### 1.1 Multi-Symbol Huffman Decode

**Problem**: We decode 1 symbol per lookup. ISA-L decodes 2-3 literals per lookup.

**Implementation**:
```rust
// Current: Single symbol per lookup
let entry = LUT[bits & 0xFFF];
let symbol = entry & 0x1FF;
let len = entry >> 28;

// Target: 2-3 symbols per lookup
let entry = LUT[bits & 0xFFF];
if entry & FLAG_MULTI != 0 {
    // Packed: [sym3:8][sym2:8][sym1:8][count:2][len:4]
    let count = (entry >> 26) & 0x3;
    let len = entry >> 28;
    output.extend(&[
        (entry & 0xFF) as u8,
        ((entry >> 8) & 0xFF) as u8,
        ((entry >> 16) & 0xFF) as u8,
    ][..count]);
    bits.consume(len);
}
```

**Effort**: 3-4 days
**Expected gain**: 20-30%

### 1.2 Table Generation for Multi-Symbol

The conversion script needs to generate multi-symbol entries:
- For each 12-bit lookup prefix
- Find all combinations of 2-3 literal codes that fit
- Pack them into the entry

**Files to modify**:
- `scripts/convert_isal_tables.py`
- `src/inflate_tables.rs` (auto-generated)

**Effort**: 2 days

### 1.3 Bitstream Optimization

**Current issues**:
1. We refill too often (every symbol)
2. We use 64-bit buffer but only peek 12 bits

**ISA-L approach**:
```c
// Load 57+ bits at once, decode multiple symbols before refill
while (state->read_in_length < 57 && state->avail_in > 0) {
    state->read_in |= (uint64_t)*state->next_in << state->read_in_length;
    state->read_in_length += 8;
    state->next_in++;
    state->avail_in--;
}
```

**Effort**: 1 day
**Expected gain**: 5-10%

## Phase 2: LZ77 Copy Optimizations (Week 2-3)

### 2.1 Inline Assembly for x86_64

For the hot copy path, inline assembly is faster than intrinsics:

```rust
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn copy_32_bytes(src: *const u8, dst: *mut u8) {
    core::arch::asm!(
        "vmovdqu ymm0, [{src}]",
        "vmovdqu [{dst}], ymm0",
        src = in(reg) src,
        dst = in(reg) dst,
        options(nostack)
    );
}
```

**Effort**: 2 days
**Expected gain**: 10-15% on copy-heavy data

### 2.2 Pattern Expansion for Small Distances

For distances 1-8, expand the pattern:

```rust
match distance {
    1 => {
        // RLE: fill with single byte using SIMD broadcast
        let byte = *src;
        let pattern = _mm256_set1_epi8(byte);
        // Store in 32-byte chunks
    }
    2 => {
        // 2-byte pattern: broadcast 16-bit word
        let word = *(src as *const u16);
        let pattern = _mm256_set1_epi16(word);
    }
    // ... up to 8
}
```

**Effort**: 2 days
**Expected gain**: 15-20% on repetitive data

### 2.3 ARM NEON Implementation

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn copy_16_bytes_neon(src: *const u8, dst: *mut u8) {
    use std::arch::aarch64::*;
    let data = vld1q_u8(src);
    vst1q_u8(dst, data);
}
```

**Effort**: 2 days

## Phase 3: Table Structure Optimization (Week 3)

### 3.1 Two-Level Lookup Tables

For codes > 12 bits, use two-level lookup:

```
Level 1 (10 bits): Direct decode for short codes
Level 2 (5 bits): Secondary lookup for long codes
```

**Current**: Single 4096-entry table
**Target**: 1024-entry primary + 32-entry secondary tables

**Effort**: 3 days
**Expected gain**: 5% (fewer cache misses)

### 3.2 Table Alignment

Ensure lookup tables are cache-line aligned:

```rust
#[repr(C, align(64))]
struct AlignedTable {
    entries: [u32; 4096],
}
```

**Effort**: 1 day

## Phase 4: Memory Access Optimization (Week 4)

### 4.1 Prefetching Strategy

```rust
// Prefetch next input chunk while processing current
if i % 64 == 0 {
    prefetch_read(input.as_ptr().add(i + 256));
    prefetch_write(output.as_mut_ptr().add(output.len() + 256));
}
```

**Effort**: 1 day

### 4.2 Output Buffer Pre-allocation

Estimate output size from gzip ISIZE trailer:

```rust
fn estimate_output_size(data: &[u8]) -> usize {
    if data.len() >= 4 {
        // ISIZE in last 4 bytes (mod 2^32)
        let isize = u32::from_le_bytes([
            data[data.len()-4], data[data.len()-3],
            data[data.len()-2], data[data.len()-1]
        ]);
        isize as usize
    } else {
        data.len() * 4 // Conservative estimate
    }
}
```

**Effort**: 0.5 days

### 4.3 Branch Prediction Hints

Use likely/unlikely hints for common paths:

```rust
#[cold]
fn handle_error() { ... }

if std::intrinsics::likely(symbol < 256) {
    output.push(symbol as u8);
} else if std::intrinsics::unlikely(symbol == 256) {
    break;
}
```

**Effort**: 0.5 days

## Phase 5: Parallel Decompression (Week 5)

### 5.1 Speculative Block Finding

For arbitrary gzip files, speculatively find deflate block boundaries:

```rust
// Try to decode from position, track if it produces valid output
fn try_decode_from(data: &[u8], pos: usize) -> Option<DecodedChunk> {
    let mut decoder = SpeculativeDecoder::new(&data[pos..]);
    match decoder.try_decode() {
        Ok(chunk) => Some(chunk),
        Err(_) => None,
    }
}
```

**Effort**: 5 days

### 5.2 Window Propagation Pipeline

```
Thread 1: Decode chunk 0 → window_0
Thread 2: Wait for window_0, decode chunk 1 → window_1
Thread 3: Wait for window_1, decode chunk 2 → window_2
```

**Effort**: 3 days

### 5.3 Marker-Based Decompression

For back-references crossing chunk boundaries:

```rust
// Output buffer uses u16: 0-255 = literal, 256+ = marker
let marker = 32768 + (distance - decoded_so_far - 1);
output.push(marker);

// Later: resolve markers when window is known
fn resolve_markers(output: &mut [u16], window: &[u8]) {
    for val in output.iter_mut() {
        if *val >= 32768 {
            let offset = (*val - 32768) as usize;
            *val = window[window.len() - offset - 1] as u16;
        }
    }
}
```

**Already implemented** in `marker_decode.rs`

## Phase 6: Compression Improvements (Week 6-7)

### 6.1 Better Match Finding

Use hash chains with 4-byte hashes:

```rust
let hash = hash4(input[pos..pos+4]);
let chain_head = hash_table[hash];
let mut best_match = Match::none();

for candidate in chain_iter(chain_head) {
    let len = compare(input, pos, candidate);
    if len > best_match.len {
        best_match = Match { len, dist: pos - candidate };
    }
}
```

### 6.2 Lazy Matching

Check if next position has better match:

```rust
let match1 = find_match(pos);
let match2 = find_match(pos + 1);
if match2.len > match1.len + 1 {
    emit_literal(input[pos]);
    pos += 1;
    emit_match(match2);
} else {
    emit_match(match1);
}
```

### 6.3 Optimal Parsing (L10-L12)

Already implemented via libdeflate L10-L12.

## Phase 7: Testing & Benchmarking (Week 8)

### 7.1 Micro-benchmarks

```rust
#[bench]
fn bench_huffman_decode_1mb(b: &mut Bencher) {
    let data = generate_test_data(1_000_000);
    let compressed = compress(&data);
    b.iter(|| decompress(&compressed));
}
```

### 7.2 Comparison Suite

```bash
./bench.sh --tools gzippy,pigz,gzip,rapidgzip \
           --sizes 1,10,100,1000 \
           --levels 1,6,9,12 \
           --content text,binary,random
```

### 7.3 CI Performance Regression

Alert if performance drops >5% from baseline.

## Summary: Expected Gains

| Optimization | Expected Gain | Cumulative |
|--------------|---------------|------------|
| Multi-symbol decode | 20-30% | 1.35-1.50x |
| Inline asm copies | 10-15% | 1.48-1.72x |
| Pattern expansion | 15-20% | 1.70-2.07x |
| Table structure | 5% | 1.78-2.17x |
| Prefetching | 5% | 1.87-2.28x |
| Branch hints | 3% | 1.93-2.35x |

**Target**: 2x faster than current, matching or exceeding libdeflate.

## Timeline

| Week | Focus |
|------|-------|
| 1-2 | Decode loop (multi-symbol, bitstream) |
| 2-3 | LZ77 copies (asm, SIMD, patterns) |
| 3 | Table structure optimization |
| 4 | Memory access (prefetch, allocation) |
| 5 | Parallel decompression |
| 6-7 | Compression improvements |
| 8 | Testing & benchmarking |

## Files to Modify/Create

| File | Changes |
|------|---------|
| `scripts/convert_isal_tables.py` | Multi-symbol table generation |
| `src/inflate_tables.rs` | Regenerated with multi-symbol |
| `src/simd_inflate.rs` | Multi-symbol decode loop |
| `src/simd_copy.rs` | NEW: Inline asm copy routines |
| `src/parallel_inflate.rs` | Speculative block finding |
| `benches/inflate_bench.rs` | NEW: Micro-benchmarks |

## Success Criteria

1. **Decompress 1MB in <0.6ms** (matching libdeflate)
2. **BGZF decompress at 4000+ MB/s** (beating rapidgzip)
3. **Arbitrary gzip at 2500+ MB/s** on multi-core
4. **All tests pass** with correct output
5. **No regressions** in compression ratio
