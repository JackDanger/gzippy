# Performance Gap Analysis: gzippy vs libdeflate

## Current State

| Metric | gzippy turbo | libdeflate | Gap |
|--------|--------------|------------|-----|
| 1MB decompress | 175µs (5.7 GB/s) | 85µs (11.8 GB/s) | **2.07x slower** |

## Root Cause Analysis

### 1. L1 Cache Misses from Large Tables (Primary Bottleneck)

**Problem**: Our 15-bit lookup tables are 128KB each (32K entries × 4 bytes)
- L1 cache is typically 32KB
- Every table lookup likely misses L1, hits L2 (12+ cycles vs 3 cycles)

**Evidence**: 
```
15-bit table: 128KB (L2 hit = 12 cycles)
10-bit table:   4KB (L1 hit =  3 cycles)
```

**Expected speedup from fixing**: **2.26x** (matches our 2.07x gap!)

### 2. No Multi-Symbol Decode for Dynamic Blocks

**Problem**: Dynamic Huffman blocks can have 3-5 bit codes
- With 12-bit lookup, could pack 3-4 symbols per entry
- We decode one symbol at a time

**Expected additional speedup**: **1.2x**

### 3. Table Construction Overhead

**Problem**: Building Huffman tables per-block is expensive
- Allocating Vec (heap allocation)
- Filling 32K entries

**Expected impact**: ~5% overhead

### 4. Bit Buffer Refill Overhead

**Problem**: We refill aggressively
- Current: refill when bits < 24
- Optimal: refill when bits < 16 (less frequent)

**Expected impact**: ~3% overhead

## Optimization Priorities

### Priority 1: Two-Level Huffman Tables (Biggest Win)

**Approach**:
- Level 1: 10-bit table (1024 entries, 4KB) - fits in L1
- Level 2: Secondary table for codes > 10 bits

**Format**:
```
L1 entry (16 bits):
  Bit 15: 0 = direct decode, 1 = use L2
  Bits 0-8: Symbol (0-285)
  Bits 9-13: Code length (1-15)
  
L2 entry: Same as L1 but for remaining bits
```

**Implementation**:
```rust
// Level 1 lookup
let entry = table[bits.peek(10) as usize];
if entry & 0x8000 == 0 {
    // Direct decode (fast path)
    let sym = entry & 0x1FF;
    let len = (entry >> 9) & 0x1F;
    bits.consume(len);
} else {
    // Need L2 lookup
    let l2_idx = (entry & 0x7FFF) as usize;
    let l2_entry = l2_table[l2_idx + bits.peek_extra(5)];
    ...
}
```

**Estimated effort**: 2-3 days
**Estimated speedup**: 2.0-2.3x

### Priority 2: Multi-Symbol Decode for Dynamic Blocks

**Approach**:
When building dynamic Huffman tables, identify literal pairs/triples that fit in 10 bits.

**Key insight**: Dynamic blocks often have:
- Very common symbols with 3-4 bit codes
- 3+3+4 = 10 bits → 3 symbols per lookup!

**Format**:
```
Multi-symbol entry (32 bits):
  Bits 0-7:   First symbol
  Bits 8-15:  Second symbol
  Bits 16-23: Third symbol
  Bits 24-25: Symbol count - 1
  Bits 26-31: Total code length
```

**Estimated effort**: 2-3 days
**Estimated speedup**: 1.2-1.4x (on top of two-level)

### Priority 3: Optimize Table Construction

**Current**:
```rust
let mut table = vec![(0u16, 0u8); 32768];  // Heap alloc + zero fill
for symbol in 0..num_symbols { ... }       // Fill entries
```

**Optimized**:
```rust
// Stack-allocated L1 table
let mut l1_table = [0u16; 1024];  // 2KB on stack

// Only allocate L2 if needed
let l2_table = if max_len > 10 {
    Some(build_l2_table(...))
} else {
    None
};
```

**Estimated effort**: 1 day
**Estimated speedup**: 1.05x

### Priority 4: Reduce Bit Buffer Overhead

**Change refill threshold**:
```rust
// Current
const REFILL_THRESHOLD: u32 = 24;

// Optimal (refill less often)
const REFILL_THRESHOLD: u32 = 16;
```

Also: Use unrolled 8-byte loads when possible.

**Estimated effort**: 0.5 days
**Estimated speedup**: 1.03x

## Implementation Plan

### Week 1: Two-Level Tables (Critical Path)

**Day 1-2**: Implement two-level table structure
- Create `TwoLevelTable` struct
- Implement `build_two_level_table(lens: &[u8]) -> TwoLevelTable`
- Test correctness

**Day 3**: Integrate into turbo_inflate
- Replace `build_huffman_table` with two-level version
- Update decode loops to use new structure
- Benchmark

**Day 4-5**: Optimize and polish
- Tune L1 bits (try 9, 10, 11)
- Profile cache behavior
- Fix any edge cases

**Expected outcome**: 1.8-2.2x speedup, matching or beating libdeflate

### Week 2: Multi-Symbol for Dynamic Blocks

**Day 1-2**: Multi-symbol table generation
- Identify pairs/triples during table build
- Pack into entries

**Day 3-4**: Multi-symbol decode loop
- Unpack and write multiple literals
- Handle mixed literal/length cases

**Day 5**: Benchmark and tune
- Test on various file types
- Profile hit rates

**Expected outcome**: Additional 1.2x speedup, exceeding libdeflate

### Week 3: Polish and Edge Cases

- Table construction optimization
- Bit buffer tuning
- ARM NEON two-level path
- CI benchmarks

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| 1MB decompress | 175µs | < 85µs |
| MB/s throughput | 5,700 | > 12,000 |
| vs libdeflate | 2.07x slower | ≥ 1.0x (match or beat) |

## Risks and Mitigations

1. **Risk**: Two-level tables add branch
   **Mitigation**: Most codes are < 10 bits, so L2 is rarely needed

2. **Risk**: Multi-symbol adds complexity
   **Mitigation**: Keep single-symbol path for length codes

3. **Risk**: ARM performance differs
   **Mitigation**: Test on both x86_64 and ARM64 throughout
