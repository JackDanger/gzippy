# Performance Gap Analysis: gzippy vs. Established Tools

## 1. OPTIMIZATION INVENTORY

### libdeflate (Eric Biggers) - Single-Thread King

| Category | Optimization | Impact |
|----------|-------------|--------|
| **Bitstream** | 64-bit bitbuffer, refill every ~56 bits | ~15% faster |
| **Bitstream** | Branchless refill: `in_next -= (bitsleft >> 3) & 0x7` | ~5% fewer branches |
| **Bitstream** | `bitsleft -= entry` (not `(u8)entry`) - garbage in high bits OK | µop reduction |
| **Table** | LITLEN_TABLEBITS=11, OFFSET_TABLEBITS=8 | Optimal cache use |
| **Table** | Packed entry: literal/length/extrabits/codelen in u32 | 1 lookup = all info |
| **Table** | Subtables allocated inline after main table | Cache locality |
| **Fastloop** | Process 2 literals + 1 match per iteration | Amortize loop overhead |
| **Fastloop** | Exit when near output/input boundaries | Avoid per-iteration checks |
| **Fastloop** | FASTLOOP_MAX_BYTES_WRITTEN = 2+258+39 = 299 | Fixed bounds |
| **Copy** | Word-at-a-time copy (not byte-at-a-time) | 8x fewer stores |
| **Copy** | Overlapping copy for distance >= 8: 8-byte chunks | Fast memcpy |
| **BMI2** | Runtime dispatch to BMI2 version on x86_64 | 5-10% on Intel |
| **Safety** | Fixed output buffer size (no realloc) | Predictable perf |
| **Safety** | `SAFETY_CHECK(expr)` macro with early return | Branch hints |

### rapidgzip (Maximilian Knespel) - Parallel King

| Category | Optimization | Impact |
|----------|-------------|--------|
| **Parallel** | 4MB chunk spacing (not block boundaries) | Simple partitioning |
| **Parallel** | Speculative decode with 16-bit markers | No blocking on window |
| **Parallel** | Marker values encode window offset: `val - 32768` | Direct index |
| **Parallel** | `REPLACE_MARKERS_IN_PARALLEL = true` | Each chunk in parallel |
| **Parallel** | Window propagation: only last 32KB per chunk | 128x less serial work |
| **Table** | LUT_BITS_COUNT=11 optimal (documented benchmarks) | 4KB fits L1 |
| **Table** | HuffmanCodingShortBitsCached: cache hit priority | Minimal fallback |
| **Table** | Pre-compute length+distance in single entry | 1 lookup per match |
| **Memory** | `FasterVector<uint8_t>` - custom allocation | Reduce fragmentation |
| **Memory** | WindowMap with SharedWindow for deduplication | Less memory pressure |
| **I/O** | BlockFetcher with prefetch strategy | Hide I/O latency |
| **I/O** | SharedFileReader for parallel reads | No fd contention |

### ISA-L igzip (Intel) - Assembly King

| Category | Optimization | Impact |
|----------|-------------|--------|
| **Assembly** | Hand-tuned x86-64 assembly decode loops | ~50% vs C |
| **Assembly** | NASM macros: `LARGE_SHORT_SYM_LEN=25` | Packed entry decode |
| **Assembly** | Multiple specialized decode functions (01, 04, default) | CPU-specific |
| **Table** | ISAL_DECODE_LONG_BITS=12, SHORT_BITS=10 | Two-tier table |
| **Table** | `TRIPLE_SYM_FLAG`, `DOUBLE_SYM_FLAG`, `SINGLE_SYM_FLAG` | Multi-symbol decode |
| **Table** | Dynamic switch based on block size (<2KB→single, <4KB→double, else→triple) | Adaptive |
| **Copy** | `COPY_SIZE=16`, `COPY_LEN_MAX=258` | SIMD-aligned |
| **Copy** | 16-byte overlapping copy for all distances | Vectorized |
| **CRC** | AVX2/AVX-512 CRC32 with `adler32_avx2_4.asm` | Parallel CRC |
| **Intrinsics** | BMI2 `_bzhi_u64`, AVX2 scatter/gather | Bit extraction |

### pigz (Mark Adler) - Dictionary Threading Model

| Category | Optimization | Impact |
|----------|-------------|--------|
| **Threading** | Dedicated writer thread + N compress workers | I/O overlapped |
| **Threading** | 32KB dictionary chains between blocks | Compression ratio |
| **Threading** | Pool of reusable buffers (Space) | Zero malloc in loop |
| **Threading** | use-count based buffer recycling | Automatic GC |
| **Threading** | Sequence numbers for ordered output | Lock-free ordering |
| **Memory** | `INPOOL` limit on concurrent buffers | Bounded memory |
| **Memory** | Dictionary copy overlapped with write | Hide latency |
| **Compress** | zlib with preset dictionary | Standard zlib perf |

---

## 2. GZIPPY CURRENT STATE

| Area | What We Have | Status |
|------|-------------|--------|
| **Bitstream** | 64-bit FastBits with branchless refill | ✅ Matches libdeflate |
| **Table** | TwoLevelTable (10-bit L1 + 5-bit L2) | ✅ Good |
| **Table** | CombinedLUT (length+distance in single entry) | ✅ Like rapidgzip |
| **Parallel** | BGZF: parse headers, pre-alloc, parallel write | ✅ 3.2x vs pigz |
| **Parallel** | Multi-member: find boundaries, parallel decode | ⚠️ 1.0x of pigz |
| **Parallel** | Single-member: falls back to sequential | ❌ No parallel |
| **Copy** | memset for dist=1, 8-byte chunks for dist≥8 | ✅ Good |
| **Copy** | AVX-512 for large non-overlapping | ✅ Where available |
| **Threading** | rayon with atomic counter | ⚠️ Basic |
| **Memory** | Vec::with_capacity from ISIZE | ⚠️ Reallocs possible |

---

## 3. GAP ANALYSIS

### Critical Gaps (Must Fix for 3x)

| Gap | Their Solution | Our Gap | Fix Effort |
|-----|---------------|---------|------------|
| **Multi-symbol decode** | ISA-L: decode 2-3 literals per iteration | We do 1 at a time | Medium |
| **Entry format** | libdeflate: `bitsleft -= entry` (whole u32) | We extract fields | Small |
| **Single-member parallel** | rapidgzip: marker-based speculative | We use sequential | Large |
| **Buffer pool** | pigz: reusable Space structs | We malloc/free | Medium |
| **Multi-member detection** | We scan byte-by-byte | Rapidgzip: chunk spacing | Small |

### Secondary Gaps (Nice to Have)

| Gap | Their Solution | Our Gap | Fix Effort |
|-----|---------------|---------|------------|
| **BMI2 dispatch** | libdeflate: runtime CPU feature detection | We use compile-time | Small |
| **Assembly decode** | ISA-L: NASM hand-tuned | We use Rust | N/A (out of scope) |
| **Dictionary chaining** | pigz: 32KB from prev block | Our L1-5 are independent | By design |
| **CRC32 SIMD** | ISA-L: AVX-512 parallel CRC | We use zlib-ng | Small |

### Architecture-Specific Gaps

| Arch | Gap | Fix |
|------|-----|-----|
| **x86_64** | No BMI2 `_bzhi_u64` dispatch | Add runtime feature check |
| **x86_64** | No AVX-512 CRC32 | Use `crc32fast` crate with AVX-512 |
| **ARM64** | No NEON copy in inner loop | Add `vld1q_u8`/`vst1q_u8` |
| **ARM64** | No hardware CRC | Already using zlib-ng |

### Thread-Count Specific Gaps

| Threads | Gap | Cause | Fix |
|---------|-----|-------|-----|
| **1** | 30% slower than libdeflate | No multi-sym, no BMI2 dispatch | Multi-sym decode |
| **2-4** | BGZF good, multi-member at parity | Chunk overhead | Reduce sync points |
| **8+** | Memory bandwidth limited | Too many cache misses | Better prefetch |

---

## 4. PRIORITY FIXES FOR 3X

### Phase 1: Multi-Symbol Decode (Expected: +30-50% single-thread)

```rust
// Current: 1 symbol per iteration
let (sym, bits) = table.decode(peek);
consume(bits);

// Target: 2-3 symbols per iteration (ISA-L style)
let entry = table.lookup(peek);
if entry.is_triple_literal() {
    output[pos..pos+3].copy_from_slice(&entry.literals);
    pos += 3;
    consume(entry.total_bits);
} else if entry.is_double_literal() {
    // ...
}
```

### Phase 2: Packed Entry Format (Expected: +5-10%)

```rust
// Current: extract fields with masks
let symbol = entry & 0x1FF;
let bits = (entry >> 9) & 0xF;
consume(bits);

// Target: libdeflate style
// entry format: bits[0:8]=bits_to_consume, bits[8:15]=flags, bits[16:31]=payload
self.bitsleft -= entry;  // Consume directly from entry
```

### Phase 3: Marker-Based Single-Member Parallel (Expected: 2-5x for large files)

```rust
// rapidgzip approach:
// 1. Partition input at 4MB intervals
// 2. Each chunk decodes with uint16_t output buffer
// 3. Unknown back-refs become markers: value = 32768 + window_offset
// 4. After window propagation, replace markers in parallel
```

### Phase 4: Buffer Pool (Expected: -5-10% allocation overhead)

```rust
// pigz approach: reusable buffers with use-count
struct BufferPool {
    available: Vec<Vec<u8>>,
    max_size: usize,
}

impl BufferPool {
    fn acquire(&mut self) -> Vec<u8> {
        self.available.pop().unwrap_or_else(|| Vec::with_capacity(self.max_size))
    }
    fn release(&mut self, buf: Vec<u8>) {
        if self.available.len() < MAX_POOLED { self.available.push(buf); }
    }
}
```

---

## 5. EXPECTED OUTCOME

| Metric | Current | After Phase 1-4 | vs pigz |
|--------|---------|-----------------|---------|
| Single-thread | ~700 MB/s | ~1000 MB/s | 2x |
| BGZF 8-thread | 3.2x of pigz | 3.5x of pigz | 3.5x |
| Multi-member 8-thread | 1.0x of pigz | 2.5x of pigz | 2.5x |
| Single-member 8-thread | 1.0x of gzip | 3.0x of gzip | 3x |
