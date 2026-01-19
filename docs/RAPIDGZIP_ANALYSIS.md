# Rapidgzip Optimization Analysis

## Key Insight: Combined Length+Distance LUT

Rapidgzip's primary performance advantage comes from pre-computing **entire LZ77 matches** (length + distance) in a single lookup table entry.

### CacheEntry Structure (4 bytes)
```cpp
struct CacheEntry {
    uint8_t bitsToSkip;      // Total bits consumed
    uint8_t symbolOrLength;  // Literal value OR length-3
    uint16_t distance;       // Distance value OR special marker
};
```

### Special Distance Values
| Value | Meaning |
|-------|---------|
| 0 | Literal byte (symbolOrLength is the value) |
| 0xFFFE | Length code needs slow path |
| 0xFFFF | END_OF_BLOCK |
| 1-32768 | Actual distance value |

### How It Works

**For literals (symbol 0-255)**:
- Single lookup returns the byte
- `bitsToSkip` = code length

**For short LZ77 matches** (when length+distance fit in LUT bits):
- Pre-compute: length code + length extra bits + distance code + distance extra bits
- Single lookup returns: total bits, length-3, and distance
- No separate distance decode needed!

**For long codes** (don't fit in LUT):
- Use `distance = 0xFFFE` as marker
- Fall back to separate length/distance decode

### Why This Is Fast

1. **Single lookup for LZ77**: Most matches decoded in one table access
2. **No distance table lookup in hot path**: Distance is pre-computed
3. **Reduced branches**: Fewer conditionals per symbol
4. **Better pipelining**: CPU can predict next operations

### Implementation Requirements

1. **Combined table structure**: 4-byte entries with length+distance
2. **Build phase**: Enumerate all (length code, length extra, dist code, dist extra) combinations
3. **Decode loop**: Single lookup, branch on distance value (0=literal, 0xFFFF=EOB, else LZ77)

## Performance Impact

This optimization provides ~2x speedup over separate literal/distance tables because:
- Most real-world data has many short LZ77 matches
- Distance codes 0-3 (distance 1-4) have no extra bits
- Many length codes have 0-1 extra bits

With 12-bit LUT, we can fit:
- 7-bit length code + 5 extra bits for length/distance = most common cases
- 8-bit length code + 4 extra bits = still covers many cases

## Gaps in Our Implementation

| Feature | rapidgzip | gzippy |
|---------|-----------|--------|
| Combined length+distance LUT | ✅ | ❌ |
| Single-lookup LZ77 decode | ✅ | ❌ |
| Pre-computed extra bits | ✅ | ❌ |
| 4-byte packed entries | ✅ | ❌ |

## Implementation Plan

1. Create `CombinedLUT` struct with 4-byte entries
2. Build phase: enumerate all length+distance combinations
3. Decode loop: single lookup, branch on distance special values
4. Test with real data to verify speedup
