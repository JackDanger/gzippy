# Parallel Single-Member Decompression — Architecture Bug and Redesign

**Document purpose**: Technical write-up for Claude Opus to design a correct replacement
for `src/decompress/parallel/single_member.rs`. The current implementation is slower than
single-threaded ISA-L decompression. This document traces the exact code paths that cause
the regression and specifies two viable redesign options.

**Audience**: Expert in deflate/gzip internals and parallel algorithms. Has not seen this
codebase before. The document is self-sufficient.

---

## ⚠️ Outcome note (v0.5.1, May 2026)

The bug analysis in §§1–6 of this document is correct, but the proposed designs in
§§7–8 are **not what shipped**. The "Option B: 32 KB prefix correction in phase 2"
plan is **wrong on a subtle but fatal point**:

> §9 claims: "After the first 32,768 bytes of a chunk are decoded (regardless of
> correctness), all subsequent bytes in the chunk are produced using only bytes
> within the chunk itself — no cross-boundary dependencies."

The first clause (no cross-boundary deps after 32 KB) is true. The implied
conclusion (so phase-1's `decoded[32 KB..]` is correct) is **false**. Cross-chunk
back-references resolve to zeros in phase 1, producing wrong bytes near the chunk
start. Those wrong bytes then act as the *source* of later chunk-local back-references
within `decoded[0..32 KB]` — and writes to `decoded[32 KB..]` from chunk-local
back-references can reach into the wrong prefix (RFC 1951 max distance = 32,768).
Error propagation continues arbitrarily far through chains of near-max-distance
back-references. The 32 KB prefix re-decode in phase 2 doesn't unwind that.

**What shipped (v0.5.1):** a two-pass algorithm that **fully re-decodes** chunks
1..T-1 in phase 2 with each predecessor's phase-1 last 32 KB as a *speculative*
window. Total parallel work is 2N (vs. N for the buggy 32 KB plan, vs. N+O(32KB)
the plan thought it would achieve). Total elapsed = 2N/T, giving speedup ≈ T/2.
At T=2 this ties sequential ISA-L (the cost of unknown-token-stream correctness);
at T=4 → 2×; T=8 → 4×; scaling linearly with T until memory bandwidth saturates.

Speculation can fail on pathological inputs (long chains of near-max-distance
back-references reaching the predecessor's last 32 KB). The final phase combines
per-chunk CRC32s and compares to the gzip trailer **before** writing any bytes to
the output stream, so a failed speculation returns `Err(CrcMismatch)` without
corrupting the writer — the caller falls back to sequential ISA-L.

The actual module documentation lives in `src/decompress/parallel/single_member.rs`.
The §§1–6 trace of the v0.3.0–v0.5.0 bug is still accurate and is preserved here
for historical context.

---

## 1. Project and routing context

gzippy is a Rust gzip tool targeting maximum decompression speed. The production routing
entry point is `decompress_gzip_libdeflate` in `src/decompress/mod.rs`.

### 1.1 `classify_gzip` (mod.rs lines 71–89)

```rust
pub fn classify_gzip(data: &[u8], num_threads: usize) -> DecodePath {
    if has_bgzf_markers(data) {
        return DecodePath::GzippyParallel;
    }
    if is_likely_multi_member(data) {
        return if num_threads > 1 {
            DecodePath::MultiMemberPar
        } else {
            DecodePath::MultiMemberSeq
        };
    }
    if crate::backends::isal_decompress::is_available() {
        return DecodePath::IsalSingle;
    }
    if data.len() > 1024 * 1024 * 1024 {
        return DecodePath::StreamingSingle;
    }
    DecodePath::LibdeflateSingle
}
```

The `IsalSingle` path (and also `StreamingSingle` and `LibdeflateSingle`) all dispatch to
`decompress_single_member`. The parallel upgrade is wired inside that function, not in
`classify_gzip`.

### 1.2 `decompress_single_member` (mod.rs lines 206–256)

```rust
pub(crate) fn decompress_single_member<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    const MIN_PARALLEL_COMPRESSED: usize = 10 * 1024 * 1024;
    if crate::backends::isal_decompress::is_available()
        && num_threads > 1
        && data.len() > MIN_PARALLEL_COMPRESSED
    {
        match crate::decompress::parallel::single_member::decompress_parallel(
            data, writer, num_threads,
        ) {
            Ok(n) => { writer.flush()?; return Ok(n); }
            Err(_) => { /* fall through to sequential */ }
        }
    }
    // Falls to sequential ISA-L, streaming, or libdeflate
    ...
}
```

**Gate conditions for the parallel path:**
- `isal_decompress::is_available()` — true on x86_64 with feature `isal-compression`, false on arm64
- `num_threads > 1` — CI (ubuntu-latest, 2 vCPUs) presents T=2
- `data.len() > 10 MiB` — gzip compressed size, not uncompressed

The parallel path must succeed or the call falls back to single-threaded ISA-L
(`decompress_gzip_stream`), which is the correct sequential baseline.

---

## 2. Claimed architecture vs. actual code

### 2.1 Module-level comment claims (single_member.rs lines 1–15)

```
Architecture (no sequential pre-scan):

1. Partition the compressed stream at regular intervals.
2. Each chunk searches forward for a deflate block boundary using BlockFinder,
   then decodes from there until the next partition point in parallel.
3. Confirmed chunks re-decode via ISA-L inflatePrime (non-byte-aligned restart
   plus 32 KiB sliding-window dictionary). v0.3.0 made this the path that beats
   rapidgzip at scale.
4. CRC32 + ISIZE verification on the assembled output.
```

This claims T threads decode in parallel in phase 1, then phase 2 does fast ISA-L
re-decodes of the confirmed chunks using correct windows. **The actual code does something
fundamentally different, as traced below.**

---

## 3. Detailed code trace

### 3.1 Entry: `decompress_parallel` (lines 52–155)

Parses gzip header, extracts `deflate_data`, reads ISIZE and CRC32 from trailer.
Sets `num_chunks = num_threads`. Calls `speculative_decode_parallel` (Phase 1) then
`confirm_resolve_write` (Phase 2).

### 3.2 `max_output_for_chunk` (lines 162–170)

```rust
fn max_output_for_chunk(isize_total: usize, num_chunks: usize, compressed_bytes: usize) -> usize {
    let isize_based = if num_chunks > 0 && isize_total > 0 {
        (isize_total / num_chunks) * 2
    } else { 0 };
    let ratio_based = (compressed_bytes * 8).max(64 * 1024);
    ratio_based.max(isize_based)
}
```

For Silesia corpus (~211 MB uncompressed, ~80 MB compressed) at T=2:

| Variable | Value |
|---|---|
| `compressed_bytes` | 80 MB / 2 = 40 MB |
| `isize_based` | (211 MB / 2) × 2 = 211 MB |
| `ratio_based` | 40 MB × 8 = 320 MB |
| `max_output` | max(320 MB, 211 MB) = **320 MB** |

**320 MB exceeds the entire uncompressed output.** This cap effectively means "decode until
BFINAL block is reached" because the true output (211 MB) is always under 320 MB. The cap
was intended to prevent runaway allocation for pathological chunk boundaries, but at 8×
compression ratio it is not tight enough.

### 3.3 Phase 1: `speculative_decode_parallel` (lines 181–224)

```rust
fn speculative_decode_parallel(
    deflate_data: &[u8],
    num_chunks: usize,
    spacing_bits: usize,
    isize_total: usize,
) -> Vec<Option<SpeculativeChunk>> {
    let results: Vec<Mutex<Option<SpeculativeChunk>>> = ...;
    let task_idx = AtomicUsize::new(0);
    let compressed_bytes = deflate_data.len() / num_chunks;
    let max_output = max_output_for_chunk(isize_total, num_chunks, compressed_bytes);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = task_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks { break; }

                let chunk = if idx == 0 {
                    find_chunk_end(deflate_data, 0, max_output)
                        .map(|end_bit| SpeculativeChunk { start_bit: 0, end_bit })
                } else {
                    search_and_find(deflate_data, partition_bit, max_output)
                };
                // store in results[idx]
            });
        }
    });
    ...
}
```

**What each thread does:**

- **Thread 0** (`idx == 0`): calls `find_chunk_end(deflate_data, 0, 320_MB)`. This calls
  `decompress_deflate_from_bit_with_end(deflate_data, 0, &[], 320_MB)`. ISA-L decodes from
  the very beginning with an empty window. It decodes the entire stream (211 MB output)
  because the 320 MB cap is never hit. Returns `end_bit = EOF`.

- **Thread 1** (`idx == 1`): calls `search_and_find(deflate_data, midpoint_bit, 320_MB)`.
  `search_boundary_forward` uses BlockFinder + ISA-L validation to find a valid deflate
  block boundary near the midpoint. Call that `mid_bit`. Then calls
  `find_chunk_end(deflate_data, mid_bit, 320_MB)`, which decodes from `mid_bit` to
  BFINAL — approximately half the file's worth of output.

**The `SpeculativeChunk` struct** (lines 157–160):

```rust
struct SpeculativeChunk {
    start_bit: usize,
    end_bit: usize,
}
```

**This stores only bit positions. No decoded output bytes are retained.**
Both threads do real ISA-L decode work but throw away all decompressed bytes. Only the
start and end bit positions of each chunk are kept.

**Total decode work in phase 1 (T=2, Silesia):**

- Thread 0: decodes 211 MB of output (entire file, from bit 0)
- Thread 1: decodes ~105 MB of output (from midpoint to EOF)
- Combined: ~316 MB decoded in parallel, all discarded

### 3.4 Phase 2: `confirm_resolve_write` (lines 326–425)

```rust
fn confirm_resolve_write<W: Write>(
    deflate_data: &[u8],
    speculative: &[Option<SpeculativeChunk>],
    expected_size: usize,
    expected_crc: u32,
    writer: &mut W,
) -> Result<usize, ParallelError> {
    let mut buffer = Vec::with_capacity(expected_size);
    let mut window = Vec::<u8>::new();
    let mut confirmed_bit: usize = 0;

    // Build HashMap: start_bit → chunk_index
    let mut spec_by_start: HashMap<usize, usize> = HashMap::new();
    for (i, spec) in speculative.iter().enumerate() {
        if let Some(chunk) = spec { spec_by_start.insert(chunk.start_bit, i); }
    }

    loop {
        if buffer.len() >= expected_size || confirmed_bit >= total_bits { break; }

        if let Some(&idx) = spec_by_start.get(&confirmed_bit) {
            let chunk = speculative[idx].as_ref().unwrap();
            let chunk_end_byte = chunk.end_bit.div_ceil(8).min(deflate_data.len());
            let chunk_input = &deflate_data[..chunk_end_byte];
            match decompress_deflate_from_bit_with_end(
                chunk_input, chunk.start_bit, &window, expected_size - buffer.len(),
            ) {
                Some((decoded, actual_end_bit)) => {
                    update_window(&mut window, &decoded);
                    buffer.extend_from_slice(&decoded);
                    confirmed_bit = snap_to_nearest_spec(actual_end_bit, &spec_by_start);
                }
                None => return Err(ParallelError::DecodeFailed),
            }
        } else {
            let (bytes, end_bit) = decode_sequential_to_spec(...)?;
            buffer.extend_from_slice(&bytes);
            update_window(&mut window, &bytes);
            confirmed_bit = end_bit;
        }
    }
    ...
}
```

**Trace for T=2, Silesia:**

1. `confirmed_bit = 0`
2. `spec_by_start.get(&0)` → HIT (chunk 0 recorded `start_bit = 0`)
3. `chunk_end_byte = EOF`; `chunk_input = deflate_data` (entire compressed stream)
4. Calls `decompress_deflate_from_bit_with_end(entire_deflate, 0, empty_window, 211_MB)`
   — this is a **full sequential ISA-L decode of the entire file**
5. `buffer` now contains 211 MB, `confirmed_bit = EOF`
6. Loop exits because `buffer.len() >= expected_size`

The spec HIT for chunk 0 at `start_bit=0` causes phase 2 to re-decode the entire file
from scratch, sequentially. This is the only output-producing decode. Phase 1 work is
entirely wasted.

### 3.5 `decode_sequential_to_spec` (lines 430–471)

```rust
fn decode_sequential_to_spec(
    deflate_data: &[u8],
    start_bit: usize,
    window: &[u8],
    max_output: usize,
    _spec_by_start: &HashMap<usize, usize>,  // NOTE: unused, prefixed with _
) -> Result<(Vec<u8>, usize), ParallelError> {
    // Fast path: decode sequential gaps at full inflate speed.
    if crate::backends::inflate_bit::is_available() {
        match decompress_deflate_from_bit_with_end(deflate_data, start_bit, window, max_output) {
            Some((decoded, end_bit)) => return Ok((decoded, end_bit)),
            None => return Err(ParallelError::DecodeFailed),
        }
    }
    Err(ParallelError::DecodeFailed)
}
```

The parameter `_spec_by_start` is explicitly unused (note the leading underscore). This
function does not stop at spec boundaries — it decodes all the way to max_output or BFINAL.
This confirms that even the sequential fallback path in phase 2 performs a complete decode
without using the speculative boundary information.

---

## 4. Why the result is slower than sequential ISA-L

### 4.1 Work accounting

For T=2 on Silesia (211 MB output, 80 MB compressed):

| Phase | Description | Work (decode-equivalent MB) | Parallelism |
|---|---|---|---|
| Phase 1 | Speculation — output discarded | ~316 MB | T=2 parallel |
| Phase 2 | Re-decode entire file | 211 MB | T=1 sequential |
| Total | | ~527 MB equivalent | — |

Phase 1 parallel work reduces elapsed time by ~T=2, but adds overhead (thread spawn,
AtomicUsize, Mutex). Phase 2 is fully sequential and is the bottleneck.

### 4.2 Timing model (CI: ubuntu-latest, 2 vCPUs)

ISA-L on this hardware: ~370 MB/s (memory-bandwidth-limited at ~2×370 = 740 MB/s
aggregate, but decode is single-threaded in phase 2).

- Phase 1 elapsed: 316 MB / (2 × 370 MB/s) ≈ **427 ms**
- Phase 2 elapsed: 211 MB / 370 MB/s ≈ **571 ms**
- Total: ≈ **998 ms**
- Effective throughput: 211 MB / 998 ms ≈ **211 MB/s**

Single-threaded ISA-L baseline:

- 211 MB / 370 MB/s ≈ **571 ms → 370 MB/s**

The parallel path is **1.75× slower** than sequential on the same machine. The observed
CI benchmark result (~185 MB/s) matches this model, with additional overhead from thread
spawn and memory allocation.

### 4.3 Benchmark comparisons (CI, ubuntu-latest, 2 vCPUs, Silesia)

| Tool | Throughput |
|---|---|
| gzippy parallel single-member (current) | ~185 MB/s |
| unpigz (reference parallel) | ~199 MB/s |
| rapidgzip | ~314 MB/s |
| gzippy sequential ISA-L (fallback) | ~370 MB/s |

The parallel path underperforms even unpigz and is far behind single-threaded ISA-L.

---

## 5. The architectural flaw

### 5.1 Why phase 2 must be sequential

Each chunk's correct decode requires the 32 KB sliding window from the end of the
previous chunk's output. The window is produced by decoding, so:

- Chunk 0's output produces the window for chunk 1
- Chunk 1's output produces the window for chunk 2
- ...

This chain is inherently sequential. Phase 2 cannot be parallelized without restructuring
how the window dependency is resolved.

### 5.2 Why the current algorithm cannot beat sequential

The current algorithm:

1. **Phase 1 (parallel):** T threads × (N/T) bytes each = N bytes total speculation work.
   Work is discarded. Only bit positions are retained.
2. **Phase 2 (sequential):** One big ISA-L decode of N bytes from the first confirmed
   boundary (always bit 0).

For any T, the total time is:
```
T_total = T_phase1 + T_phase2
        = (N / (T × speed)) + (N / speed)
        = N/speed × (1/T + 1)
```

This is always greater than `N/speed` (pure sequential). At T=2 it is 1.5× sequential; at
T=4 it is 1.25× sequential. The algorithm structurally cannot beat sequential at any thread
count because phase 2 is the output-producing step and it is always sequential.

### 5.3 What v0.3.0 broke

From CLAUDE.md:

> **v0.3.0 — parallel single-member:** ISA-L `inflatePrime` re-decodes confirmed chunks
> at non-byte-aligned bit offsets at full ISA-L SIMD speed (~1500 MB/s/thread), replacing
> the prior pure-Rust marker decoder (22 MB/s/thread).

The original architecture (pre-v0.3.0) used the `MarkerDecoder` (still present in
`src/decompress/parallel/marker_decode.rs`). In that design:

- **Phase 1 (parallel):** Each thread calls `MarkerDecoder` with an empty window.
  `MarkerDecoder` produces `Vec<u16>` output where values 0–255 are literal bytes and
  values ≥ `MARKER_BASE` (32768) are sentinel markers encoding unresolved back-references.
  **The decoded bytes are retained in memory**, not discarded.
- **Phase 2 (sequential):** For each chunk in order, compute the correct 32 KB window from
  the prior chunk's output. Replace markers in the current chunk's buffer using the window.
  Since back-references can reach back at most 32 KB, after the first 32 KB of a chunk is
  resolved, the rest is self-consistent.
- **Phase 3:** Concatenate and write.

v0.3.0 replaced the slow pure-Rust marker fix (22 MB/s) with ISA-L re-decode (~1500 MB/s),
but in doing so changed the architecture to **discard phase-1 output and re-decode
everything in phase 2** — inadvertently eliminating all parallelism.

---

## 6. The `MarkerDecoder` — what already exists

`src/decompress/parallel/marker_decode.rs` contains a complete pure-Rust Huffman decoder
that produces `Vec<u16>` with marker encoding:

```rust
pub const MARKER_BASE: u16 = 32768; // = WINDOW_SIZE as u16
// values 0–255: literal bytes
// values 32768–65535: markers where (value - MARKER_BASE) = window offset

pub struct MarkerChunk {
    pub start_bit: usize,
    pub end_bit: usize,
    pub data: Vec<u16>,        // decoded output, with markers
    pub marker_count: usize,
    pub final_window: Vec<u8>, // last 32 KB of output (markers converted to 0)
    pub success: bool,
    ...
}
```

This decoder is `#[allow(dead_code)]` — currently unused by the production path. It runs
at ~22 MB/s, which is why v0.3.0 abandoned it. The marker representation is correct and
the replacement logic (`replace_markers`, `to_bytes`) is implemented.

---

## 7. Correct algorithm: two viable designs

Both designs share the same key invariant:

> **The deflate back-reference window is at most 32,768 bytes (32 KB).** After a chunk
> has produced 32 KB of correct output, all subsequent output within that chunk is
> self-consistent regardless of what came before the chunk's start.

This means only the **first 32 KB of each chunk** can contain errors from missing window
data. The rest is always correct.

### 7.1 Option A: Stored-output parallel decode with O(32 KB) correction per chunk

This is the correct generalization of the v0.3.0 intention.

**Phase 1 (parallel, T threads):**

For each chunk i:
1. Find a valid deflate block boundary near partition point i using BlockFinder (same as
   current `search_boundary_forward`).
2. Call `decompress_deflate_from_bit(deflate_data_up_to_next_partition, found_bit, &ZERO_WINDOW, estimated_chunk_output_size)`.
   This produces a `Vec<u8>` with errors only in the regions where back-references cross
   the chunk boundary. Those errors are concentrated in the first ≤32 KB of output.
3. **Store the entire decoded output**, not just the bit positions.
4. Record `(start_bit, end_bit, decoded: Vec<u8>)`.

Cost: N / (T × speed) elapsed. Each thread stores ~N/T bytes.

**Phase 2 (sequential, fast):**

Process chunks in order, maintaining a 32 KB window:

```
window = []  // empty for chunk 0
for i in 0..num_chunks:
    chunk = phase1_results[i]
    if i == 0:
        // Chunk 0 decoded with empty window — correct by definition
        window = last 32 KB of chunk.decoded
        output[0] = chunk.decoded
        continue

    // Re-decode only the first min(32 KB output, chunk_size) bytes with correct window.
    // At most 32 KB of correction needed because back-refs cannot reach further.
    correction_input = deflate_data[chunk.start_bit/8 .. chunk.start_bit/8 + correction_input_limit]
    corrected_prefix = decompress_deflate_from_bit(correction_input, chunk.start_bit, window, 32*1024 + safety)
    
    // Splice: replace first len(corrected_prefix) bytes of chunk.decoded with corrected bytes.
    // The rest of chunk.decoded is already correct (no cross-boundary back-refs beyond 32 KB).
    output[i] = corrected_prefix + chunk.decoded[len(corrected_prefix)..]
    window = last 32 KB of output[i]
```

Cost per chunk: O(32 KB / speed). For T chunks: T × O(32 KB / speed) = negligible.

**Phase 3:**
Concatenate `output[0..T]` and write. Can be done during phase 2 to avoid a second pass.

**Key question**: how many bytes of re-decode does each chunk actually need?

The answer depends on how far back-references reach from the chunk start. In the worst
case (a back-reference at the very start of the chunk reaching back 32 KB), the first 32 KB
of output could all be wrong. In practice (high-entropy data like Silesia), markers are
sparse near chunk boundaries.

An efficient implementation:
- Phase 1 also records the index of the last marker byte within the first 32 KB of each chunk.
- Phase 2 only re-decodes up to `last_marker_in_first_32KB + max_match_length` bytes.
- For many chunks this will be much less than 32 KB.

### 7.2 Option B: ISA-L for both phases (pure stored-output, simpler)

This avoids the `MarkerDecoder` entirely by decoding with a zeroed window in phase 1 and
re-decoding just the correction prefix in phase 2:

**Phase 1 (parallel):**
Each thread i:
1. `search_boundary_forward` → `found_bit`
2. Call `decompress_deflate_from_bit_with_end(deflate_data, found_bit, &ZERO_WINDOW, max_chunk_output)` via ISA-L
3. Store the full decoded `Vec<u8>` and the `(start_bit, end_bit)` pair

Output is ~correct everywhere except back-references that cross the chunk boundary.

**Phase 2 (sequential):**
For each chunk i > 0:
1. The window is the last 32 KB of chunk i-1's (now-correct) output.
2. Estimate how many output bytes need correction. Two strategies:
   - **Pessimistic**: always re-decode the first 32 KB of output per chunk. ISA-L does this
     in ~(32 KB / 1500 MB/s) ≈ 21 µs. Negligible.
   - **Optimistic**: scan the first 32 KB of phase-1 output for ISA-L errors (the output
     produced with a zeroed window may have ISAL_INVALID_LOOKBACK at positions where
     back-refs cross the boundary). Re-decode from the chunk start up to the last error
     position + 258 bytes (max match length).
3. Write corrected bytes over the phase-1 buffer in-place.

**Implementability**: ISA-L's `isal_inflate_set_dict` sets the 32 KB window and
`isal_inflate_init` + bit-priming handle non-byte-aligned starts. The
`decompress_deflate_from_bit_with_end` function in `src/backends/isal_decompress.rs`
already implements this correctly. The only new requirement is an input-length limit:
pass `&deflate_data[..chunk_end_byte]` to constrain ISA-L's input to this chunk.
This already works: the current code does `let chunk_input = &deflate_data[..chunk_end_byte]`
in phase 2 (line 369), but it only bounds the phase-2 re-decode, not a stored phase-1 decode.

### 7.3 Recommended design: Option B (stored ISA-L output)

Option B requires fewer code changes and avoids reinstating the slow `MarkerDecoder`.
The `MarkerDecoder` at 22 MB/s would bottleneck phase 1 back to slower than sequential.
Option B uses ISA-L for all decode work.

**Pseudocode for the redesign:**

```rust
struct StoredChunk {
    start_bit: usize,
    end_bit: usize,
    decoded: Vec<u8>,  // output with possible errors in first <= 32 KB
}

fn decompress_parallel(gzip_data, writer, num_threads) -> Result {
    // [header/trailer parsing unchanged]
    let num_chunks = num_threads;

    // Phase 1: parallel decode with zeroed window, store output
    let phase1_results: Vec<Option<StoredChunk>> = thread::scope(|s| {
        (0..num_chunks).map(|idx| s.spawn(move || {
            let partition_bit = idx * spacing_bits;
            let found_bit = if idx == 0 { 0 } else { search_boundary_forward(deflate_data, partition_bit)? };
            let chunk_end_byte = if idx < num_chunks - 1 {
                ((idx + 1) * spacing_bits / 8).min(deflate_data.len())
            } else {
                deflate_data.len()
            };
            let (decoded, end_bit) = decompress_deflate_from_bit_with_end(
                &deflate_data[..chunk_end_byte],
                found_bit,
                &[],                  // zeroed window — errors possible in first 32 KB
                estimated_chunk_output,
            )?;
            Some(StoredChunk { start_bit: found_bit, end_bit, decoded })
        })).collect()
    });

    // Phase 2: sequential correction, O(32 KB) per chunk
    let mut window: Vec<u8> = Vec::new();
    let mut output = Vec::with_capacity(expected_output);
    for (idx, chunk_opt) in phase1_results.iter().enumerate() {
        let chunk = match chunk_opt { Some(c) => c, None => /* sequential fallback */ };
        if idx == 0 {
            // First chunk: decoded with empty window = ISA-L default = correct
            update_window(&mut window, &chunk.decoded);
            output.extend_from_slice(&chunk.decoded);
        } else {
            // Re-decode first min(32 KB output, chunk size) with correct window
            let correction_input_end = estimate_correction_end(chunk);
            let corrected = decompress_deflate_from_bit_with_end(
                &deflate_data[..correction_input_end],
                chunk.start_bit,
                &window,
                WINDOW_SIZE + max_match_length,
            )?;
            let corrected_len = corrected.0.len();
            // Write corrected prefix, then rest of phase-1 output
            output.extend_from_slice(&corrected.0);
            output.extend_from_slice(&chunk.decoded[corrected_len..]);
            update_window(&mut window, &output[output.len() - WINDOW_SIZE.min(output.len())..]);
        }
    }

    // Phase 3: verify and write
    verify_output(&output, expected_size, expected_crc)?;
    writer.write_all(&output)?;
    Ok(output.len() as u64)
}
```

---

## 8. Expected performance with correct design

### 8.1 CI (ubuntu-latest, 2 vCPUs, ISA-L ~370 MB/s per core)

Using Silesia (211 MB uncompressed, 80 MB compressed):

- Phase 1: 2 threads × (211 MB / 2) / 370 MB/s ≈ **285 ms elapsed**
- Phase 2: 2 × (32 KB / 370 MB/s) ≈ **0.17 ms** (negligible)
- Total: ≈ **285 ms**
- Effective throughput: 211 MB / 285 ms ≈ **740 MB/s**

vs. current:
- rapidgzip: 314 MB/s
- unpigz: 199 MB/s
- gzippy sequential ISA-L: 370 MB/s
- gzippy parallel current: ~185 MB/s

The redesign should deliver **740 MB/s** on CI at T=2, beating rapidgzip by 2.4×.

### 8.2 Homelab (neurotic, 8 cores, ISA-L ~1500 MB/s per core)

- Phase 1: 8 threads × (211 MB / 8) / 1500 MB/s ≈ **17.6 ms elapsed**
- Phase 2: 8 × (32 KB / 1500 MB/s) ≈ **0.17 ms** (negligible)
- Total: ≈ **17.8 ms**
- Effective throughput: 211 MB / 17.8 ms ≈ **11,900 MB/s**

This assumes no memory bandwidth bottleneck. At this thread count, aggregate bandwidth
(8 cores × 1500 MB/s = 12,000 MB/s) may saturate memory bandwidth (~50–100 GB/s on a
modern server), so the actual result may be memory-bound. However, it will substantially
exceed single-threaded performance.

---

## 9. Boundary guarantee and window invariant

A deflate stream reader that begins at a **stored block boundary** (BFINAL bit + BTYPE=00
+ LEN + ~LEN) or a **dynamic Huffman block boundary** (BFINAL + BTYPE=10 + header) with
an **empty (zeroed) window** will:

1. Correctly decode all literals in the block (no window dependency).
2. For back-references (length-distance pairs), produce **correct output** for any reference
   to bytes already decoded within this chunk, but **wrong output** for references to bytes
   before the chunk's start (which are from the prior chunk's window).

The `ZERO_WINDOW` used by both ISA-L and zlib-ng paths substitutes zeros for the 32 KB
window at the chunk start. Any back-reference that points before the chunk start produces
zeros in the output instead of the correct prior bytes.

**Key facts:**
- RFC 1951 §3.2.4: maximum distance in a back-reference is 32,768 bytes.
- Therefore, after the first 32,768 bytes of a chunk are decoded (regardless of correctness),
  all subsequent bytes in the chunk are produced using only bytes within the chunk itself —
  no cross-boundary dependencies.
- The correction in phase 2 only needs to re-decode until we are 32,768 bytes past the
  chunk start.

**Important caveat for fixed Huffman blocks:** Fixed Huffman blocks (BTYPE=01) have no
header to validate and cannot be reliably detected as block boundaries without full stream
context. The current BlockFinder skips BTYPE=01 candidates (`btype=1` returns false in
`is_valid_candidate_13`). This is correct: the phase 2 correction handles any residual
errors, but a boundary detected as the start of a fixed Huffman block is more likely to
be a false positive from scanning.

**Stored blocks (BTYPE=00):** Stored blocks contain literal bytes with no compression.
A stored block boundary guarantees no back-references within the block — the correction
for a chunk starting at a stored block boundary is zero bytes.

---

## 10. ISA-L API details

The relevant ISA-L functions used in `src/backends/isal_decompress.rs`:

```c
void isal_inflate_init(struct inflate_state *state);
int isal_inflate_set_dict(struct inflate_state *state, uint8_t *dict, uint32_t dict_len);
int isal_inflate(struct inflate_state *state);
// state.read_in       — 64-bit bit buffer (pre-load partial first byte here)
// state.read_in_length — number of valid bits in read_in
// state.crc_flag      — ISAL_DEFLATE for raw deflate (no gzip wrapper)
```

To start at a non-byte-aligned bit offset `bit_offset`:

```rust
let byte_idx = bit_offset / 8;
let bit_skip = bit_offset % 8;
if bit_skip > 0 {
    state.read_in = (data[byte_idx] as u64) >> bit_skip;
    state.read_in_length = (8 - bit_skip) as i32;
    state.next_in = data[byte_idx + 1..].as_ptr();
    state.avail_in = (data.len() - byte_idx - 1) as u32;
} else {
    state.next_in = data[byte_idx..].as_ptr();
    state.avail_in = (data.len() - byte_idx) as u32;
}
```

This is already implemented and correct in `decompress_deflate_from_bit_with_end`
(lines 397–494 of `src/backends/isal_decompress.rs`).

To prime the 32 KB window: `isal_inflate_set_dict(&mut state, window_ptr, window_len)`.
This call must happen **after** `isal_inflate_init` and **before** the first `isal_inflate`.
Already implemented.

To limit ISA-L to only one chunk's input: pass a sub-slice `&deflate_data[..chunk_end_byte]`
as the `data` parameter. ISA-L will stop when `avail_in` reaches zero. The `end_bit`
returned by `decompress_deflate_from_bit_with_end` uses `data.len() * 8` as the base,
so callers in the redesign should use the full `deflate_data` reference and pass
`chunk_end_byte` as an explicit bound, or accept that `end_bit` is relative to the slice.
The current implementation already handles this at line 368–369:
```rust
let chunk_end_byte = chunk.end_bit.div_ceil(8).min(deflate_data.len());
let chunk_input = &deflate_data[..chunk_end_byte];
```

**There is no ISA-L API to perform partial-window correction** (i.e., "apply this 32 KB
window but only to the first N bytes of output"). The only mechanism is `isal_inflate_set_dict`
which sets the window before decoding starts. The correction in phase 2 works by re-decoding
from the chunk's `start_bit` with the correct window; ISA-L then produces correct output
from the start. The caller splices in corrected bytes over the phase-1 output.

---

## 11. Design questions for the implementer

**Q1: How to estimate `correction_len` per chunk?**

Options in order of implementation complexity:

A. Always re-decode exactly 32 KB of output per chunk. Simple, correct, bounded cost.
   ISA-L does 32 KB in ~21 µs — negligible for any file > 100 MB.

B. Phase 1 decodes with zeroed window and records the **output index of the last zero that
   was produced by a back-reference vs. a literal zero** (a heuristic). This is hard to
   distinguish without tracking reference provenance.

C. Compare first 32 KB of phase-1 chunk output with the ISA-L correction output byte by
   byte; stop at the first byte where they agree for > some distance. This adds a scan
   over 32 KB but avoids needing ISA-L to decode more than necessary.

Option A is recommended for the initial implementation. It is provably correct
(32 KB is the maximum possible correction length by RFC 1951) and the cost is negligible.

**Q2: How to handle chunks where `search_boundary_forward` fails?**

If no valid block boundary is found within `SEARCH_RADIUS` (512 KB), the chunk currently
returns `None`. With stored output, the fallback is: set `chunk_start_bit = confirmed_bit`
(the bit position after the previous chunk) and decode from there using the correct window
sequentially. This chunk contributes to the sequential portion. Since most real streams have
block boundaries within 512 KB, fallback should be rare.

**Q3: Memory pressure with stored output?**

At T threads, peak working set = T × (N/T bytes of output) = N bytes = uncompressed file
size. This is the same as the current sequential approach (which allocates one full output
buffer). No additional memory is required compared to the current implementation.

At T=32 on a large file, all 32 chunks are live simultaneously in phase 1, but their total
size is still N bytes. This is acceptable.

**Q4: Thread pool vs. `thread::scope`?**

The current `thread::scope` re-spawns T threads on every call. For small files this is
measurable (Linux thread spawn ~50 µs × T). For the 10 MB threshold, a 10 MB file at
370 MB/s takes ~27 ms, so T=4 spawns (200 µs) are ~0.7% overhead. Acceptable for the
initial redesign. A persistent rayon thread pool can be added later if profiling shows it
matters.

**Q5: Output writing strategy?**

Current phase 2 builds a single `buffer: Vec<u8>` then calls `writer.write_all(&buffer)`.
The redesign stores per-chunk `Vec<u8>` in phase 1 then writes them in order during phase 2.
This means peak memory = 2 × N bytes (phase-1 chunks + final write buffer). To avoid this,
phase 2 can write each corrected chunk directly to `writer` rather than assembling a single
buffer. However, the current CRC32 verification (`verify_output`) requires the full buffer.
Options:
- Compute CRC32 incrementally during phase 2 writes (using `crc32fast::Hasher::update`)
- Keep the single-buffer approach (acceptable for files gzippy targets: < a few GB)

**Q6: Correctness test**

The existing correctness regression test is:
`tests::routing::tests::test_single_member_routing_multithread` — runs
`decompress_single_member(T=4)` on a 24 MiB input and asserts byte-perfect output.
This test must pass unchanged with the redesign.

---

## 12. Files to modify

| File | Role | Changes needed |
|---|---|---|
| `src/decompress/parallel/single_member.rs` | Primary redesign target | Replace `SpeculativeChunk` (bit-only) with `StoredChunk` (bits + decoded bytes); restructure phase 1 to store output; add phase-2 correction logic |
| `src/backends/isal_decompress.rs` | ISA-L inflatePrime backend | No changes needed — `decompress_deflate_from_bit_with_end` already supports all required operations |
| `src/backends/inflate_bit.rs` | Universal inflate-from-bit (delegates to isal_decompress or zlib-ng) | No changes needed |
| `src/decompress/parallel/marker_decode.rs` | Pre-v0.3.0 marker decoder | No changes needed — keep as-is (`#[allow(dead_code)]`); may be useful as reference but is not used in Option B |
| `src/decompress/parallel/block_finder.rs` | BlockFinder for boundary search | No changes needed |
| `tests/` | Correctness test suite | `test_single_member_routing_multithread` must pass unchanged |

---

## 13. Non-goals

- **Do not change the routing thresholds** (10 MiB, T>1, ISA-L check). These are correct.
- **Do not add arm64 support** for the parallel path. CLAUDE.md explicitly documents that
  arm64 falls through to libdeflate: "Arm64 single-member: currently falls through to
  libdeflate one-shot (fast enough)." ISA-L is unavailable on arm64 and the parallel
  path is already gated on `isal_decompress::is_available()`.
- **Do not change benchmark scripts or CI thresholds**. The threshold changes (0.50/0.80
  floors) already committed are the correct CI floors.
- **Do not change the gzip header/trailer parsing**. The current `skip_gzip_header` and
  ISIZE/CRC32 extraction logic is correct.
- **Do not change the compression path**. Only `src/decompress/parallel/single_member.rs`
  is in scope.

---

## 14. Summary of the bug

The bug is in `SpeculativeChunk`: it stores only `(start_bit, end_bit)` and discards the
bytes decoded during phase 1. Phase 2 re-decodes the entire stream sequentially using the
spec boundary at bit 0 as a cache hit, which causes a full sequential re-decode. All phase
1 parallel work is wasted. The fix is to store the decoded bytes in phase 1 and perform
only an O(32 KB) correction per chunk boundary in phase 2.
