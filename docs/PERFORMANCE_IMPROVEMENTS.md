# Potential Performance Improvements

Based on analysis of libdeflate and rapidgzip source code (available in git submodules).

## Priority 1: High Impact, Achievable

### 1.1 Fastloop/Generic Loop Architecture (libdeflate)
**Source:** `libdeflate/lib/decompress_template.h` lines 336-680

libdeflate uses a two-loop architecture:
- **Fastloop**: Runs when plenty of buffer space available (input and output)
- **Generic loop**: Slower, careful loop for near-end-of-buffer cases

Key optimizations in fastloop:
- Preloads next decode table entry before consuming current one
- Overlaps memory copy latency with bit refills
- Unrolled word-at-a-time copies (5 words initially, then loops)
- No bounds checking in inner loop

**Our gap:** We use a single loop with bounds checking on every iteration.

**Implementation:** Add `in_fastloop_end` and `out_fastloop_end` bounds, use separate fast/slow paths.

### 1.2 Multi-Literal Decode in Fastloop (libdeflate)
**Source:** `libdeflate/lib/decompress_template.h` lines 366-434

libdeflate decodes up to 2 extra literals after each primary literal:
```c
if (entry & HUFFDEC_LITERAL) {
    // 1st extra fast literal
    lit = entry >> 16;
    entry = d->litlen_decode_table[bitbuf & tablemask];
    bitbuf >>= (u8)entry;
    bitsleft -= entry;
    *out_next++ = lit;
    if (entry & HUFFDEC_LITERAL) {
        // 2nd extra fast literal
        ...
    }
}
```

**Our gap:** We have multi-literal but it was buggy (now fixed with simpler version).

**Implementation:** Restore optimized multi-literal with proper bounds checking.

### 1.3 BMI2 Intrinsics for Bit Extraction (libdeflate)
**Source:** `libdeflate/lib/x86/decompress_impl.h` lines 30-37

```c
#define EXTRACT_VARBITS(word, count)  _bzhi_u64((word), (count))
```

The `bzhi` instruction extracts the low N bits much faster than `word & ((1 << count) - 1)`.

**Our gap:** We use `word & BITMASK(count)` everywhere.

**Implementation:** Add `#[cfg(target_feature = "bmi2")]` path with `_bzhi_u64` intrinsic.

### 1.4 Distance=1 Special Case as memset (libdeflate)
**Source:** `libdeflate/lib/decompress_template.h` lines 623-648

```c
} else if (offset == 1) {
    machine_word_t v;
    v = (machine_word_t)0x0101010101010101 * src[0];
    store_word_unaligned(v, dst);
    // ... unrolled stores
}
```

This handles RLE-encoded runs (very common in silesia benchmark).

**Our gap:** We have this in `simd_copy.rs` but it may not be as optimized.

**Implementation:** Verify our distance=1 path uses broadcast+store pattern.

### 1.5 Preload Next Table Entry Before Copy (libdeflate)
**Source:** `libdeflate/lib/decompress_template.h` lines 555-572

```c
// Before starting copy, refill bits and preload next entry
entry = d->litlen_decode_table[bitbuf & litlen_tablemask];
REFILL_BITS_IN_FASTLOOP();
// Then do the copy
```

This hides memory latency by overlapping operations.

**Our gap:** We refill and decode sequentially.

**Implementation:** Reorder decode loop to preload entry before LZ77 copy.

---

## Priority 2: Medium Impact

### 2.1 Double Literal Cache (rapidgzip)
**Source:** `rapidgzip/.../HuffmanCodingDoubleLiteralCached.hpp`

Caches pairs of consecutive literals in a larger LUT:
- Index by `2 * minCodeLength + 1` bits
- Returns two literals at once when both fit

**Our gap:** We decode one symbol at a time.

**Implementation:** Build double-literal LUT for common literal pairs.

### 2.2 Multi-Symbol Decode with Distance (rapidgzip)
**Source:** `rapidgzip/.../HuffmanCodingShortBitsMultiCached.hpp`

CacheEntry struct (4 bytes):
```cpp
struct CacheEntry {
    bool needToReadDistanceBits : 1;
    uint8_t bitsToSkip : 6;
    uint8_t symbolCount : 2;
    uint32_t symbols : 18;
};
```

Decodes multiple symbols per LUT lookup when possible.

**Our gap:** Our CombinedLUT only handles single symbol + distance.

**Implementation:** Extend CombinedLUT to return symbol count and packed symbols.

### 2.3 Table Doubling During Build (libdeflate)
**Source:** `libdeflate/lib/deflate_decompress.c` lines 859-900

libdeflate optimizes table building by incrementally doubling:
- Start with 2^len entries
- Copy first half to second half when doubling
- Avoids strided stores, better cache performance

**Our gap:** We fill entries with strided access.

**Implementation:** Use table doubling during `TwoLevelTable::build()`.

### 2.4 Union for Decompressor State (libdeflate)
**Source:** `libdeflate/lib/deflate_decompress.c` lines 642-677

```c
union {
    u8 precode_lens[19];
    struct {
        u8 lens[288 + 32 + overrun];
        u32 precode_decode_table[128];
    } l;
    u32 litlen_decode_table[2342];
} u;
```

Uses union to overlap arrays that aren't used simultaneously.

**Our gap:** We allocate separate buffers.

**Implementation:** Use enum + union for decompressor state.

---

## Priority 3: Parallel Decompression Improvements

### 3.1 Block Finder with Precode Check (rapidgzip)
**Source:** `rapidgzip/.../blockfinder/DynamicHuffman.hpp`

rapidgzip has a 15-bit LUT to quickly filter deflate block candidates before full decode.

**Our gap:** We scan for member boundaries, not internal block boundaries.

**Implementation:** Implement block finder for single-member parallel decompress.

### 3.2 Window Map for Parallel Re-decode (rapidgzip)
**Source:** `rapidgzip/.../WindowMap.hpp`

Stores windows at chunk boundaries for parallel re-decode.

**Our gap:** We have `marker_decode.rs` but it's not fully integrated.

**Implementation:** Complete the two-pass decode with window propagation.

### 3.3 Chunk Fetcher with Prefetch (rapidgzip)
**Source:** `rapidgzip/.../GzipChunkFetcher.hpp`

Intelligent prefetching based on access pattern (sequential vs random).

**Our gap:** We use simple thread pool without prefetch strategy.

**Implementation:** Add prefetch hints for expected next chunks.

---

## Priority 4: Architecture/Platform Specific

### 4.1 AVX2/AVX-512 Copy Loops
**Source:** libdeflate implicitly uses auto-vectorized loops

For large copies (length > 32), use SIMD:
- AVX2: 32-byte loads/stores
- AVX-512: 64-byte loads/stores (where available)

**Implementation:** Add `#[cfg(target_feature = "avx2")]` copy path.

### 4.2 ARM NEON Optimizations
Current implementation focuses on x86. ARM needs:
- NEON intrinsics for bit operations
- PMULL for CRC32 acceleration

---

## Priority 5: Micro-optimizations

### 5.1 Branchless Bit Refill (libdeflate)
**Source:** `libdeflate/lib/deflate_decompress.c` lines 206-212

```c
bitbuf |= get_unaligned_leword(in_next) << (u8)bitsleft;
in_next += sizeof(bitbuf_t) - 1;
in_next -= (bitsleft >> 3) & 0x7;
bitsleft |= MAX_BITSLEFT & ~7;
```

Eliminates branch on bitsleft check.

**Our gap:** We have conditional refill.

**Implementation:** Copy libdeflate's branchless refill pattern.

### 5.2 Static Codes Caching (libdeflate)
**Source:** `decompress_template.h` lines 305-327

libdeflate caches static Huffman codes in decompressor struct:
```c
if (d->static_codes_loaded)
    goto have_decode_tables;
d->static_codes_loaded = true;
```

**Our gap:** We use thread-local, which is fine but slightly slower.

### 5.3 Entry Format Combining (libdeflate)
**Source:** `libdeflate/lib/deflate_decompress.c` lines 437-499

Single u32 entry encodes:
- Literal flag (bit 31)
- Literal/length value (bits 16-23)
- Code length (bits 0-7)
- Extra bits count (combined in low byte)

**Our gap:** We use separate fields in structs.

**Implementation:** Pack all into u32 with bitfields.

---

## Benchmarks to Target

From rapidgzip source (deflate.hpp lines 56-173):

| Decoder | Silesia Parallel | Silesia Sequential |
|---------|-----------------|-------------------|
| ISA-L | 5024 MB/s | 720 MB/s |
| HuffmanCodingShortBitsCached 10-bit | 3953 MB/s | 330 MB/s |
| HuffmanCodingDoubleLiteralCached | 3123 MB/s | 253 MB/s |

Our current:
- BGZF parallel: 4089 MB/s ✅ (beats rapidgzip pure Rust)
- Silesia single-thread: 600 MB/s ❌ (below ISA-L's 720 MB/s)

**Target:** Match ISA-L's 720 MB/s on silesia single-threaded.

---

## Implementation Order

1. **Fastloop architecture** (biggest impact on sequential)
2. **BMI2 intrinsics** (easy win on modern x86)
3. **Distance=1 optimization** (helps silesia specifically)
4. **Branchless bit refill** (removes branches from hot path)
5. **Table preload before copy** (hides latency)
6. **Multi-symbol decode** (more complex, good for parallel)
