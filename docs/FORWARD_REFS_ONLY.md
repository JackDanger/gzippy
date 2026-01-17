# Forward References Only: Analysis and Conclusion

> **STATUS: ABANDONED** - Empirical testing proves this approach provides
> 0% compression benefit. Hash priming does NOT improve output size.

## Overview

This document describes our investigation into "forward refs only" mode,
which would allow dictionary-primed compression while producing independently
decompressible output.

## The Goal

```
Standard dictionary compression:
  - set_dictionary(prev_block_data)
  - compress(current_block)
  - Output MAY reference dictionary → requires dictionary for decompression

Forward-refs-only mode:
  - set_dictionary(prev_block_data)  ← primes hash tables
  - compress(current_block)
  - Output ONLY references current block → independently decompressible
```

## Implementation Strategy

### Phase 1: Pure Rust Proof of Concept

Before modifying zlib-ng, we'll implement a proof-of-concept in pure Rust:

1. Use `miniz_oxide` (pure Rust DEFLATE) as a starting point
2. Modify the match emission to skip dictionary references
3. Benchmark to validate the approach

### Phase 2: zlib-ng Fork

1. Fork zlib-ng repository
2. Add a new flag: `Z_FORWARD_REFS_ONLY`
3. Modify match emission in `deflate_slow.c` and `deflate_fast.c`

### Key Code Changes (zlib-ng)

```c
// deflate.c - Add new flag
#define Z_FORWARD_REFS_ONLY 0x8000  // New flag

// In deflateSetDictionary:
if (strm->flags & Z_FORWARD_REFS_ONLY) {
    strm->forward_refs_base = strm->strstart;  // Mark where block starts
}

// In emit_match (deflate_slow.c, deflate_fast.c):
static void emit_match_or_literal(deflate_state *s, int dist, int len) {
    // Check if reference points before block start
    if (s->flags & Z_FORWARD_REFS_ONLY) {
        unsigned int match_pos = s->strstart - dist;
        if (match_pos < s->forward_refs_base) {
            // Reference points into dictionary - emit literals instead
            for (int i = 0; i < len; i++) {
                emit_literal(s, s->window[s->strstart + i]);
            }
            return;
        }
    }
    // Normal match emission
    emit_match(s, dist, len);
}
```

### Phase 3: Rust Integration

1. Build modified zlib-ng as a static library
2. Create `zlib-ng-forward-refs` crate with custom bindings
3. Use this instead of standard zlib-ng in rigz

## Expected Benefits

| Metric | Standard rigz | Forward-refs-only |
|--------|---------------|-------------------|
| Hash priming | ❌ | ✅ |
| Dictionary refs | ❌ | ❌ |
| Parallel decompress | ✅ | ✅ |
| Compression ratio | 95% of pigz | 97-98% of pigz |

The improvement comes from better hash chain coverage:
- Dictionary bytes hash to chains
- These chains help find matches in current block
- Even without referencing dictionary, we find more matches

## Alternative: miniz_oxide Modification

If zlib-ng modification is too complex, we can modify `miniz_oxide`:

```rust
// In miniz_oxide's LZ77 encoder:

fn find_match(&self, pos: usize) -> Option<Match> {
    let match = self.hash_table.find_longest_match(pos);
    
    if let Some(m) = match {
        // Check if match is within current block
        if m.distance as usize > pos - self.block_start {
            // Match points before block - skip it
            return None;
        }
    }
    
    match
}
```

## Validation

1. Compress with forward-refs-only mode
2. Verify output is valid DEFLATE
3. Decompress WITHOUT dictionary - must succeed
4. Compare sizes: should be smaller than no-dictionary, larger than with-refs

## Timeline

- Phase 1 (miniz_oxide PoC): 2-3 days
- Phase 2 (zlib-ng fork): 1 week
- Phase 3 (Rust integration): 2-3 days
- Testing & benchmarking: 1 week

Total: ~3 weeks for full implementation

## Risks

1. **Hash priming benefit may be smaller than expected**
   - The improvement comes from finding intra-block matches that hash-collide with dictionary
   - This is a second-order effect, may be <1%

2. **zlib-ng internals are complex**
   - Match finding is highly optimized with SIMD
   - Modifications may break performance

3. **Maintenance burden**
   - Keeping fork in sync with upstream zlib-ng

## Experimental Results

We ran experiments to measure the actual benefit of hash priming.

### Methodology

1. Take 15 consecutive 64KB blocks from Proust text (Project Gutenberg)
2. For each block, use the previous 32KB as "dictionary"
3. Measure compression with Python zlib (single-stream, allows cross-refs)

### Results

| Level | Block Alone | Dict+Block Together | Cross-Ref Savings |
|-------|-------------|---------------------|-------------------|
| L1 | 459,015 B | 683,921 B | 8,531 B (1.9%) |
| L6 | 398,882 B | 588,490 B | 18,463 B (4.6%) |
| L9 | 397,957 B | 586,678 B | 19,133 B (4.8%) |

### Key Finding

**Cross-ref savings = Total dictionary benefit!**

At L9:
- Full dictionary benefit (from earlier tests): ~4.8%
- Cross-reference savings: ~4.8%
- Hash priming benefit: **~0%**

### Why Hash Priming Provides Zero Benefit

1. `deflateSetDictionary` does two things:
   - **Primes hash tables**: Builds hash chains for dictionary bytes
   - **Fills sliding window**: Enables backreferences to dictionary

2. Hash priming helps find matches *faster* but doesn't change *which* matches exist

3. The compression benefit comes entirely from **referencing dictionary bytes**

4. Without emitting those references, output size is identical to no-dictionary

### Conclusion

**Forward-refs-only is useless.** The approach would provide:
- 0% compression improvement
- Same output as no-dictionary compression
- All complexity for no benefit

The only way to get dictionary-quality compression is to actually reference
the dictionary bytes, which requires the decompressor to have the same dictionary.

## Decision Point

~~Before investing in Phase 2+, we should:~~
~~1. Implement Phase 1 (miniz_oxide PoC)~~
~~2. Measure actual compression improvement~~
~~3. If <1% benefit, this isn't worth the complexity~~

**DECISION: ABANDONED** - Experimental results prove 0% benefit.
