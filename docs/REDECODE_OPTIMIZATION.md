# Re-decode Optimization: 13.6x Speedup

## The Problem

Current single-member gzip decompression:
- MarkerDecoder: 100 MB/s (slow, but can start at any bit position)
- libdeflate: 1361 MB/s (fast, but needs known window)
- **Gap: 13.6x**

For single-member files, we currently fall back to flate2 sequential (~750 MB/s).
We're leaving performance on the table.

## The Solution: Two-Phase Decode

### Phase 1: Boundary Finding (MarkerDecoder)
Use MarkerDecoder to decode chunks and discover:
- Exact block boundaries (start_bit, end_bit)
- Required window data (last 32KB of previous chunk)
- Output size for each chunk

### Phase 2: Re-decode (libdeflate)
Once boundaries are known, re-decode each chunk with libdeflate:
- Set dictionary to the window from Phase 1
- Decompress from start_bit to end_bit
- Output directly to final buffer (no marker replacement needed)

## Implementation

### Step 1: Add dictionary support to libdeflate wrapper

```rust
// src/libdeflate_ext.rs
pub fn decompress_deflate_with_dict(
    data: &[u8],           // Deflate stream (not gzip!)
    start_bit: usize,      // Bit offset to start
    end_bit: usize,        // Bit offset to end (exclusive)
    window: &[u8],         // 32KB dictionary
    output: &mut [u8],     // Pre-allocated output buffer
) -> io::Result<usize>
```

### Step 2: Modify parallel_decompress to use two-phase approach

```rust
pub fn decompress_parallel<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Phase 1: Sequential boundary finding with MarkerDecoder
    // (One thread decodes until TARGET_OUTPUT, records boundaries)
    let boundaries = find_chunk_boundaries(data)?;
    
    // Phase 2: Parallel re-decode with libdeflate
    // (N threads decompress verified chunks with known windows)
    let outputs = redecode_chunks_parallel(&boundaries, data, num_threads)?;
    
    // Phase 3: Write outputs in order
    for output in outputs {
        writer.write_all(&output)?;
    }
}
```

### Step 3: Benchmark target

| Phase | Current | Target | Method |
|-------|---------|--------|--------|
| Find boundaries | 2.1s (sequential) | 2.1s | MarkerDecoder |
| Re-decode | N/A | 0.16s (8 threads) | libdeflate |
| **Total** | 2.1s | 0.28s | **7.5x speedup** |

Single-member target: **750 MB/s → 2500+ MB/s**

## Technical Details

### libdeflate Dictionary API

libdeflate supports setting a dictionary via:
```c
void libdeflate_set_dict(struct libdeflate_decompressor *d,
                         const void *dict, size_t dict_len);
```

The Rust wrapper `libdeflater` exposes this as:
```rust
impl Decompressor {
    pub fn set_dict(&mut self, dict: &[u8]);
}
```

### Bit-Aligned Decoding

Challenge: libdeflate expects byte-aligned input, but deflate blocks can start at any bit.

Solutions:
1. **Align to block boundaries**: Always decode to byte-aligned block boundaries
2. **Bit extraction**: Extract the bit range into a new buffer, padding as needed
3. **Use our own inflate**: For non-byte-aligned chunks, use ultra_fast_inflate with dictionary

Recommendation: Start with option 1 (only re-decode byte-aligned chunks), fall back to MarkerDecoder for non-aligned.

## Expected Results

| File Type | Current | After | Speedup |
|-----------|---------|-------|---------|
| BGZF | 4930 MB/s | 4930 MB/s | 1.0x (already optimal) |
| Multi-member | 3636 MB/s | 3636 MB/s | 1.0x (already optimal) |
| Single-member | 750 MB/s | 2500+ MB/s | 3.3x |

This makes gzippy competitive with rapidgzip on ALL file types.
