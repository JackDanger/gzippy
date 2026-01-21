# Gap Analysis & Optimization Plan

**Current: 970 MB/s (69.7%) | Target: 1840+ MB/s (130%+)**

---

## Detailed Gap Analysis

### libdeflate vs Us (decompress_template.h)

| Feature | libdeflate | Us | Gap |
|---------|-----------|-----|-----|
| `CAN_CONSUME_AND_THEN_PRELOAD` macro | ✓ Compile-time bit budget | ✗ Runtime checks | 5-10% |
| Triple-literal in fastloop | ✓ 3 fast lits + preload | ✓ 5 lits but nested | ~same |
| `bitsleft -= entry` (full u32) | ✓ No mask | ✓ Done | ~same |
| `REFILL_BITS_IN_FASTLOOP` | ✓ Branchless | ✓ Done | ~same |
| `EXTRACT_VARBITS8` macro | ✓ 8-bit cast optimization | ✗ Not used | 1-2% |
| `out_fastloop_end` pre-calc | ✓ Bounds in registers | ✓ Done | ~same |
| `unlikely()` branch hints | ✓ Everywhere | ✗ Missing | 2-5% |
| Subtable inline in main array | ✓ Cache locality | ✓ Done | ~same |
| BMI2 EXTRACT_VARBITS | ✓ `_bzhi_u64` | ✓ Ready, N/A ARM | N/A |
| Copy with 5-word overwrite | ✓ 40 bytes | ✓ Done | ~same |

### rapidgzip vs Us (HuffmanCodingShortBitsMultiCached.hpp)

| Feature | rapidgzip | Us | Gap |
|---------|----------|-----|-----|
| `CacheEntry` with symbolCount | ✓ 1-2 symbols per lookup | ✗ 1 symbol | 10-20% |
| `needToReadDistanceBits` flag | ✓ Avoid branch | ✗ Check symbol value | 2-5% |
| Pre-computed length in entry | ✓ `readLength()` embedded | ✗ Separate decode | 5-10% |
| `DISTANCE_OFFSET` in symbols | ✓ Length + offset combined | ✗ Separate tables | 5-10% |
| Marker-based parallel | ✓ 16-bit markers | ✓ Similar approach | ~same |
| Shared window deduplication | ✓ WindowMap | ✗ Full copy per chunk | 10-20% parallel |

### What We Have That They Don't

| Feature | Source |
|---------|--------|
| JIT table cache (fingerprint) | Novel |
| Static fixed table (OnceLock) | Novel |
| Pure Rust (no C/assembly deps) | Design |

---

## Optimization Tiers

### Tier 1: Micro-Optimizations (Est: +5-15%)

1. **`#[cold]`/`#[inline(never)]` on error paths** — Move error handling out of hot loop
2. **`likely()`/`unlikely()` intrinsics** — `core::intrinsics::likely` on literal check
3. **`EXTRACT_VARBITS8` pattern** — Cast to u8 before shift to hint 8-bit ops
4. **Register pressure reduction** — Fewer locals in fastloop, use `black_box`
5. **Prefetch next cache line** — `_mm_prefetch` on input + output

### Tier 2: Algorithmic (Est: +15-30%)

6. **Multi-symbol CacheEntry** — Pack 2 literals in single lookup (rapidgzip style)
7. **Length+distance combined entry** — Pre-compute for common short matches  
8. **Compile-time bit budget** — `CAN_CONSUME_AND_THEN_PRELOAD` as const fn
9. **Inline subtables** — Subtable entries immediately follow main entry
10. **Lazy refill** — Only refill when `bitsleft < threshold`, not every iter

### Tier 3: Novel/Exotic (Est: +20-50%)

11. **JIT-compiled decode loop** — Generate machine code for specific Huffman table
    - Fixed table = fixed control flow = no branches
    - Each code length maps to a specific instruction sequence
    - Cranelift/LLVM backend for codegen

12. **SIMD parallel decode** — Decode 4-8 streams in parallel using AVX2/NEON
    - Split input into lanes, decode independently, merge
    - Works best on fixed Huffman (same table per lane)

13. **Speculative decode with rollback** — Assume literal, rollback if wrong
    - Most symbols are literals (70-90%)
    - Write speculatively, revert on misprediction

14. **Huffman → FSM transformation** — Convert code tree to finite state machine
    - Each state = partial code, transitions = bit values
    - Vectorizable state transitions

15. **Table-free fixed Huffman** — Hard-code decode logic for RFC 1951 fixed codes
    - 0-143: 8 bits, 144-255: 9 bits, 256-279: 7 bits, 280-287: 8 bits
    - No table lookup for fixed blocks

16. **Batch output buffering** — Accumulate literals in SIMD register, flush as vector
    - 16-32 byte aligned stores vs byte-at-a-time

17. **Predictive table switching** — Detect block boundary early, pre-build next table
    - Overlap table construction with decode of current block

18. **Memory-mapped I/O** — `mmap` input for zero-copy
    - Kernel prefetch, lazy loading, huge pages

19. **Profile-guided table layout** — Reorder table entries by frequency
    - Most common codes → lowest indices → better cache

20. **Hardware CRC offload** — ARM CRC32 or Intel CRC32C for validation
    - Parallel with decode, not after

---

## Implementation Priority

| # | Optimization | Expected | Effort | Result |
|---|--------------|----------|--------|--------|
| 1 | Multi-symbol CacheEntry | +15% | Medium | Pending - exists, needs integration |
| 6 | `#[cold]` on error paths | +5% | Easy | **REGRESSED 4%** |
| 15 | Table-free fixed Huffman | +20% | Medium | **3.25x SLOWER** |
| - | Unconditional refill | +5% | Easy | **REGRESSED 12%** |
| 11 | JIT decode loop | +30% | Hard | Pending |
| 12 | SIMD parallel decode | +40% | Very Hard | Pending |

### Key Finding

**Simple micro-optimizations REGRESS performance.** The current code is already well-tuned.
The remaining 30% gap requires novel approaches:
- Compiler-level optimizations (target-cpu=native)
- JIT code generation for specific Huffman tables
- SIMD parallel decoding (multiple streams)

---

## Validation

```bash
# Baseline
for i in {1..3}; do cargo test --release bench_cf_silesia -- --nocapture 2>&1 | grep "Our throughput"; done

# After each change
cargo test --release  # Must pass 282 tests
# Re-run baseline
```

---

## Reference Files

| Tool | File | Key Technique |
|------|------|---------------|
| libdeflate | `lib/decompress_template.h:350-500` | Fastloop structure |
| rapidgzip | `huffman/HuffmanCodingShortBitsMultiCached.hpp` | Multi-symbol cache |
| ISA-L | `igzip/igzip_decode_block_stateless.asm` | Assembly decode |
