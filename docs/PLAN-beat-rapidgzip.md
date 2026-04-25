# Plan: Beat rapidgzip on single-member decompress

## The gap

x86_64 silesia Tmax: gzippy 240 MB/s, rapidgzip 314 MB/s (**-24%**).
All other losses are on arm64 where speculation is fundamentally wrong (16× slower,
block boundaries too sparse).

This plan targets x86_64 only. arm64 stays on sequential libdeflate.

## Why every previous attempt failed

Five attempts, same root cause: the pure-Rust marker decoder runs at **~22 MB/s per
thread** and is the bottleneck regardless of what we do after it.

| Attempt | What was tried | Why it failed |
|---------|---------------|---------------|
| #1 Feb 2026 | Basic parallel pipeline | Pure-Rust inflate + overhead > sequential |
| #2 Feb 2026 | Wire rapidgzip_decoder module | Reverted in 42 min; same speed problem |
| #3 Feb 2026 | Prefix-overlap two-pass | Pure-Rust inflate still bottleneck |
| #4 Feb 2026 | Speculative parallel w/ markers | Working, but 88–148 MB/s total |
| #5 (current) | parallel_single_member.rs | Same, better instrumented |

None of the attempts used fast SIMD inflate (ISA-L) for anything except the
first chunk. The `redecode_chunk_with_isal` scaffold in `marker_decode.rs` always
returned `None` because it requires **byte-aligned** chunk starts — and deflate
block boundaries are almost never byte-aligned.

## The new weapon: `inflatePrime`

rapidgzip's `IsalInflateWrapper` (gzip/isal.hpp) handles non-byte-aligned starts via:

```cpp
void inflatePrime(size_t nBitsToPrime, uint64_t bits) {
    m_stream.read_in |= bits << m_stream.read_in_length;
    m_stream.read_in_length += nBitsToPrime;
}
```

ISA-L's `inflate_state` has a 64-bit bit buffer (`read_in`) and a bit count
(`read_in_length`). Pre-loading the partial first byte's bits into this buffer
lets ISA-L decode from **any bit position** at full AVX2/AVX-512 speed.

Our isal-sys 0.5.3 bindings expose everything we need:
- `inflate_state.read_in: u64` and `inflate_state.read_in_length: i32`
- `isal_inflate_set_dict` (igzip_lib.rs:2072) — sets the 32KB back-reference window

We have never used this combination. It is the mechanism rapidgzip uses to get
ISA-L speed on non-byte-aligned chunk boundaries.

## Architecture

```
Phase 1 — Parallel (N threads):
  Chunk 0:   ISA-L directly from bit 0 (always byte-aligned, no markers)
  Chunk 1–N: BlockFinder + marker decode in parallel with chunk 0
             (existing parallel_single_member.rs logic)

Phase 2 — Pipelined serial:
  Chunk 0 ISA-L completes → window known for chunk 1
  ISA-L re-decode chunk 1:  isal_inflate_set_dict(window) + inflatePrime(bit_skip)
  Chunk 1 completes → window known for chunk 2
  ISA-L re-decode chunk 2: ...
```

The marker decode of chunks 1..N happens in parallel with chunk 0's ISA-L decode.
The serial re-decode chain uses ISA-L (≈1500 MB/s) instead of marker resolution.

### Critical path model (silesia 200MB, 8 threads)

| Step | Time |
|------|------|
| Chunks 1–7 marker decode (8 threads parallel) | ~140ms |
| Chunk 0 ISA-L | ~17ms (in parallel with above) |
| ISA-L re-decode chunks 1–7, serial | 7 × 17ms = 119ms |
| **Total** | **140 + 119 = 259ms → ~770 MB/s theoretical** |

At 50% efficiency for overhead: ~385 MB/s, beats rapidgzip's 314 MB/s.

The key ratio: marker decode now drives only 140ms instead of the full 570ms
(8 threads × 1/4 file vs. sequential). The ISA-L chain is fast enough that it
doesn't add much.

## Implementation

### Step 1 — `decompress_deflate_from_bit` in `isal_decompress.rs`

New function, x86_64 + isal-compression feature only:

```rust
pub fn decompress_deflate_from_bit(
    data: &[u8],         // full deflate data (from gzip header to trailer)
    bit_offset: usize,   // where this chunk starts (any bit alignment)
    dict: &[u8],         // 32KB window from previous chunk's last 32KB output
    out: &mut Vec<u8>,
) -> Option<usize>
```

Implementation:
```
byte_idx = bit_offset / 8
bit_skip = bit_offset % 8

isal_inflate_init(&mut state)
state.crc_flag = IGZIP_DEFLATE   // raw deflate, no gzip wrapper

if bit_skip > 0:
    state.read_in = (data[byte_idx] >> bit_skip) as u64
    state.read_in_length = (8 - bit_skip) as i32
    state.next_in = data[byte_idx + 1..].as_ptr()
    state.avail_in = data.len() - byte_idx - 1
else:
    state.next_in = data[byte_idx..].as_ptr()
    state.avail_in = data.len() - byte_idx

isal_inflate_set_dict(&mut state, dict.as_ptr(), dict.len() as u32)

loop isal_inflate until ISAL_BLOCK_FINISH or error
```

Unit test: compress a known blob, scan ISA-L to find a block boundary at a
known bit offset, call this function, compare output to reference.

### Step 2 — Replace marker resolution with ISA-L in `confirm_resolve_write`

In `parallel_single_member.rs::confirm_resolve_write`, for each confirmed chunk:

```rust
// Try fast path first
if let Some(decoded) = isal_decompress::decompress_deflate_from_bit(
    deflate_data, chunk.start_bit, &window, &mut tmp_buf
) {
    writer.write_all(&decoded)?;
    update_window(&mut window, &decoded);
    continue;
}
// Fall back to marker resolution (existing code)
```

This keeps correctness: ISA-L failure → existing marker path. Chunk 0 (bit 0)
is always byte-aligned and already works.

### Step 3 — Production gate in `decompression.rs`

Add inside `decompress_single_member`:

```rust
if isal_decompress::is_available()   // x86_64 only
    && num_threads > 1
    && deflate_data.len() > 10 * 1024 * 1024  // 10MB compressed
{
    use crate::experiments::parallel_single_member::decompress_parallel;
    if let Ok(n) = decompress_parallel(gzip_data, writer, num_threads) {
        return Ok(n);
    }
    // fall through to sequential ISA-L
}
```

Verify with `make route-check`, then `make quick`, then cloud fleet.

## What this is NOT

- Not changing arm64 (block boundaries too sparse, speculation loses badly)
- Not improving marker decoder speed (do this if Phase 1 benchmarks show we still lose)
- Not adding an index/cache system
- Not changing compression paths

## Rollback

The production gate falls through to sequential ISA-L on any error. If `make`
shows regression, remove the gate — zero risk to existing paths.

## Files touched

| File | Change |
|------|--------|
| `src/backends/isal_decompress.rs` | Add `decompress_deflate_from_bit` |
| `src/experiments/parallel_single_member.rs` | Use ISA-L in confirm phase |
| `src/decompression.rs` | Wire gate into `decompress_single_member` |
| `src/experiments/marker_decode.rs` | Fix `decode_with_isal` to call new fn |
