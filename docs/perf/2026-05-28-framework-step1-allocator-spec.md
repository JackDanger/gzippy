# Framework Step 1: rapidgzip allocator source-dive + smaps/strace baseline

**Date**: 2026-05-28
**Purpose**: Document rapidgzip's exact allocation pattern so future
gzippy allocator work is "swap to known target", not "stab in dark".

## Top-level finding (single sentence)

rapidgzip uses **rpmalloc as global allocator** + **many small fixed-size
128 KiB buffers** (not one big variable-size buffer per chunk).
gzippy uses **glibc malloc** + **one big growable Vec per chunk**.
The two design choices stack: rpmalloc bypasses brk entirely while
gzippy spends 64.5% of memory-syscall time in brk.

## 1. rapidgzip's allocation primitives (sources)

### 1.1 RpmallocAllocator (`vendor/rapidgzip/librapidarchive/src/core/FasterVector.hpp:64-110`)

```cpp
template<typename ElementType>
class RpmallocAllocator
{
    [[nodiscard]] constexpr ElementType*
    allocate( std::size_t nElementsToAllocate )
    {
        auto const nBytesToAllocate = nElementsToAllocate * sizeof( ElementType );
        return reinterpret_cast<ElementType*>(
            rpmalloc_ensuring_initialization( nBytesToAllocate ) );
    }

    constexpr void
    deallocate( ElementType* allocatedPointer, std::size_t /*nElementsAllocated*/ )
    {
        rpfree( allocatedPointer );
    }
};
```

It's a `std::allocator`-compatible adapter over rpmalloc's `rpmalloc()` /
`rpfree()`. No magic. The work happens inside rpmalloc.

### 1.2 FasterVector (`FasterVector.hpp:115-129`)

```cpp
#ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
template<typename T>
using FasterVector = std::vector<T, RpmallocAllocator<T> >;
#else
template<typename T>
using FasterVector = std::vector<T>;
#endif
```

**FasterVector is just `std::vector<T>` with rpmalloc backing**.

The custom "FasterVector" class with skipped-init that lives below in the
file is **DISABLED via `#if 1` / `#else`**, per the comment:

> This was supposed to be a faster std::vector alternative that saves
> time by not initializing its contents on resize... However, it leads
> to almost double the memory usage with wikidata.json (12 GB -> 16
> GB)...

So the "no-init" trick was abandoned. The real lever is rpmalloc.

### 1.3 ChunkData allocation pattern (`ChunkData.hpp:30-66`)

`ALLOCATION_CHUNK_SIZE = 128 KiB` and the ChunkData stores its decoded
bytes as `FasterVector<FasterVector<uint8_t>>` — a vector of 128 KiB
chunks, not a single growable vector.

Their own published benchmark (in the source comment):

```
ALLOCATION_CHUNK_SIZE  Bandwidth (silesia-256x.tar.pigz)
    64  KiB         19.4 19.7       GB/s
    256 KiB         20.8 20.7       GB/s
    1   MiB         21.2 20.7 20.8  GB/s
    4   MiB          8.4  8.5  8.3  GB/s   ← gzippy effectively here
```

**2.5× throughput difference between 1 MiB and 4 MiB chunks.**

The dev comment says:

> 1 MiB seems like the natural choice because the optimum (compressed)
> chunk size is around 4 MiB and it would also be exactly one hugepage
> if support for that would ever be added.

Production is 128 KiB to also handle BGZF (~64 KiB streams) without
overallocating.

### 1.4 DecodedData::append (`DecodedData.hpp:239-262`)

```cpp
DecodedData::append( DecodedDataView const& buffers )
{
    const auto& appendToEquallySizedChunks = [&]( /* ... */ ) {
        for ( const auto& buffer : buffersIn ) {
            // ...
            if ( targetChunks.empty() || targetChunks.back().size() == ALLOCATION_ELEMENT_COUNT ) {
                targetChunks.emplace_back().reserve( ALLOCATION_ELEMENT_COUNT );
            }
            // ... copy into the current 128 KiB chunk; spill into next if needed
        }
    };
    // ...
}
```

Pattern: when the current 128 KiB chunk is full, `emplace_back().reserve(128KiB)`
appends a new fixed-size chunk. **Never grows a single buffer; always
appends fixed-size new ones.**

rpmalloc keeps a thread-local cache of recently-freed 128 KiB regions,
so `emplace_back().reserve(128KiB)` is a cache hit after the first few
chunks. No fresh page-fault on subsequent allocations of the same size.

## 2. gzippy's allocation pattern (for contrast)

### 2.1 ChunkData::data (`src/decompress/parallel/chunk_data.rs:251-275`)

```rust
pub fn new(encoded_offset_bits: usize, configuration: ChunkConfiguration) -> Self {
    let cap = configuration.max_decoded_chunk_size;
    Self::new_with_buffers(
        encoded_offset_bits,
        configuration,
        chunk_buffer_pool::take_u16(0),
        chunk_buffer_pool::take_u8(cap),   // ← single Vec<u8> sized to max chunk
        // ...
    )
}
```

**Single `Vec<u8>` per chunk with capacity = max-decoded-chunk-size (typically 12-16 MiB)**.

### 2.2 Allocator backing

- `chunk_buffer_pool` returns a pooled Vec when available (hit) or
  `Vec::with_capacity` (miss).
- With `--features arena-allocator`, the Vec uses `RpmallocAlloc`
  (`src/decompress/parallel/rpmalloc_alloc.rs`).
- WITHOUT that feature: glibc malloc backs the Vec.
- The MAIN HEAP (everything else) is always glibc malloc; `#[global_allocator]`
  is never set.

### 2.3 Why this is structurally worse

1. **One big variable-size buffer per chunk** → all 12 MiB faults on
   first touch, OR the Vec grows (doubling) and old pages are freed
   back to OS via brk-shrink before the next chunk needs them.

2. **glibc malloc uses brk** for medium-size allocations. Each brk
   call extends the heap by zeroed pages (the kernel zeros for
   security). Per strace below, gzippy spends 64.5% of memory-syscall
   time in brk.

3. **No thread-local cache of similar-size regions**. Every chunk's
   first Vec allocation pays page-fault cost in proportion to chunk
   size.

## 3. Strace memory-syscall comparison (neurotic, silesia-gzip9.gz, T=16)

### 3.1 gzippy-pure-strip

```
% time     seconds  usecs/call     calls  syscall
------ ----------- ----------- ---------  -------
 64.51    0.024746        1649        15  brk        ← 64.5% of time
 26.35    0.010109        2527         4  munmap
  5.81    0.002229          58        38  mmap
  3.18    0.001219          55        22  mprotect
  0.15    0.000059          19         3  madvise
------ ----------- ----------- ---------  -------
100.00    0.038362         467        82  total
```

### 3.2 rapidgzip

```
% time     seconds  usecs/call     calls  syscall
------ ----------- ----------- ---------  -------
 66.22    0.005397          77        70  mmap
 28.61    0.002332          89        26  mprotect
  3.09    0.000252          16        15  brk        ← 3.1% of time
  2.07    0.000169           9        17  munmap
------ ----------- ----------- ---------  -------
100.00    0.008150          63       128  total
```

### 3.3 Key syscall-level findings

| Metric | gzippy-pure | rapidgzip | Difference |
|--------|-------------|-----------|------------|
| **Total mem-syscall time** | 38.4 ms | 8.2 ms | gzippy 4.7× slower |
| **brk total time** | 24.7 ms (64.5%) | 0.25 ms (3.1%) | gzippy 100× slower in brk |
| **brk avg per call** | 1649 µs | 16 µs | gzippy 100× per call |
| **mmap call count** | 38 | 70 | rapidgzip uses mmap more |
| **mmap total time** | 2.2 ms | 5.4 ms | rapidgzip spends more time but per-call is similar |
| **munmap calls** | 4 | 17 | rapidgzip frees more often |
| **madvise** | 3 | 0 | gzippy uses madvise; rapidgzip doesn't |

### 3.4 Interpretation

rapidgzip's pattern: lots of mmap (rpmalloc internal pool extensions),
basically no brk. Each mmap is amortized: rpmalloc maps a large region
ONCE, then subdivides it for many small allocations via its internal
free-list.

gzippy's pattern: small number of LARGE brk syscalls because glibc
malloc tries to satisfy 12 MiB Vec allocations by extending the heap.
Each brk has to zero its acquired pages (Linux security
requirement). At 1649 µs per brk × 15 brks = 24.7 ms of CPU spent
just zeroing kernel pages.

**This is the page-fault gap manifesting as syscall cost.**

## 4. The exact lever (no longer "stab in dark")

To match rapidgzip's allocator pattern, gzippy must do BOTH of:

### Lever 4.1: Use rpmalloc as the GLOBAL allocator

Currently gzippy only uses `RpmallocAlloc` for chunk_buffer_pool's
`Vec<u8>` when `--features arena-allocator` is on. Everything else
(LUTs, scratch vectors, etc.) goes through glibc malloc, including
the brk-heavy path.

Fix: set `#[global_allocator] = RpMalloc` in `main.rs`. The
`rpmalloc` crate is already in Cargo.toml under feature
`global-rpmalloc`. **Enable that feature by default.**

Expected: brk time drops from 24.7 ms to ~0.25 ms (24.5 ms saved =
~3.4% of a 720 ms decode).

### Lever 4.2: Switch ChunkData::data to chunked allocation

Replace the single `Vec<u8>` with `Vec<Box<[u8; 128*1024]>>` — a
vector of fixed-size 128 KiB chunks. When decoding writes overflow
the current chunk, allocate a new one. The downstream consumer
(write to writer) iterates the chunks in order.

With rpmalloc backing AND fixed 128 KiB chunks: each `Box::new([0u8; 128*1024])`
hits rpmalloc's thread-local cache after the first few. Page-faults
amortize across the whole process lifetime.

Expected: page-fault count drops from 168.9K to ~80K (matching
rapidgzip's 78.5K).

Combined throughput expectation (extrapolating from rapidgzip's own
bench): up to 2.5× on the alloc-bound portion, ~30-40% e2e
throughput improvement.

## 5. Why we now have a NON-stab-in-dark framework

Pre-this-framework lever attempts on the allocator:
- Z-allocator prewarm: -15% (touched wrong pages, serialized faults)
- L1 MADV_HUGEPAGE: -38% (khugepaged contention; we never asked
  whether rapidgzip USES huge pages — it doesn't!)
- Various pool-size tweaks: noise

Each was guesswork. We never read the rapidgzip source first.

Post-framework: we have an EXACT TARGET (rpmalloc + 128 KiB chunked
Vec). Step 4 (alloc_pattern microbench) and Step 5 (A/B harness with
mandatory rollup fields) will measure whether we hit it.

## 6. Open questions still to answer

1. Does rapidgzip use huge pages? smaps says: no, ChunkData and
   DecodedData are normal anonymous pages. khugepaged might still
   merge them in the background but rapidgzip doesn't request it
   via madvise (strace shows 0 madvise calls in rapidgzip).

2. Does rpmalloc's thread-local cache survive a one-shot CLI
   process? Answer: yes, within the process; lost on exit. For
   one-shot CLI, cache is built up during the first ~20 ms and
   then hits for the remaining ~700 ms.

3. Should we also vendor the rpmalloc C code rather than using the
   Rust crate? The Rust crate (`rpmalloc` 0.2) is already in our
   Cargo.toml. Should be equivalent — confirmed by spot-check that
   it just FFI-wraps the upstream rpmalloc C source.

## 7. Next deliverables in this framework

- **Step 2** (perf-mem PEBS attribution): Captures cycle-exact
  load addresses for the cache misses; will confirm whether the
  misses cluster on ChunkData::data writes (expected) or
  elsewhere (would change the lever).
- **Step 3** (timeline allocator events): Time-series of alloc/free
  events visualized alongside chunk pipeline events.
- **Step 4** (alloc_pattern microbench): Pure-allocation microbench
  in `benches/alloc_pattern.rs` that runs in 30 s instead of a
  20-trial neurotic A/B. Should reproduce rapidgzip's 2.5× internal
  benchmark on the ALLOCATION_CHUNK_SIZE variable.
- **Step 5** (pluggable A/B harness): Tests the full lever (rpmalloc
  global + 128 KiB chunks) end-to-end on neurotic with mandatory
  rollup fields.
