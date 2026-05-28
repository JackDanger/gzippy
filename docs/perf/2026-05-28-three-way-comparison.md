# Three-way perf comparison: gzippy pure-rust vs gzippy ISA-L vs rapidgzip

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `9102459` (T3-simplify in main)
**Host**: neurotic LXC 199 (i7-13700T, kernel 6.12)
**Build (gzippy)**: `cargo build --release --features pure-rust-inflate`
            and `--features isal-compression`, `-C target-cpu=native`,
            `force-frame-pointers=yes` for symbolization, `debuginfo=1`
**rapidgzip**: 0.16.0 (`/usr/local/bin/rapidgzip`)
**Fixture**: `benchmark_data/silesia-gzip9.gz` (212 MB uncompressed)
**Parallelism**: T=16 in all three

## A. Single-run perf-stat counters (robust to system load)

These are EVENT COUNTS per workload — they don't depend on system load or
contention. Definitive comparison.

| Counter | gzippy-pure | gzippy-isal | rapidgzip | gzippy-pure ÷ rapidgzip |
|---------|-------------|-------------|-----------|------------------------|
| **Cycles (core)** | 5.04B | 3.71B | **2.32B** | **2.17×** more |
| **Instructions (core)** | 5.11B | 3.69B | **3.78B** | 1.35× more |
| **IPC (core)** | 1.01 | 1.00 | **1.63** | 0.62× (worse) |
| **Branches (core)** | 822M | 590M | **678M** | 1.21× more |
| **Branch-misses** | 23.0M (2.80%) | 15.9M (2.70%) | 18.7M (2.75%) | similar % |
| **L1-dcache misses (core)** | 42.6M | 20.6M | **15.0M** | **2.84×** more |
| **LLC misses (core)** | 1.65M | 1.90M | **573K** | **2.88×** more |
| **Page faults** | 179,505 | 174,051 | **78,623** | **2.28×** more |
| **Context switches** | 6,144 | 2,259 | **1,876** | 3.27× more |

### Top empirical findings

1. **gzippy-pure takes 2.17× the cycles rapidgzip does** for the same
   workload. That's the throughput gap proper.

2. **gzippy-pure takes 1.35× the instructions rapidgzip does**. So
   gzippy is doing MORE WORK per byte, not just doing the same work
   slower.

3. **gzippy-pure has 2.84× the L1 cache misses and 2.88× the LLC
   misses**. Cache locality is materially worse.

4. **gzippy-pure has 2.28× the page faults**. The output-buffer
   allocator pattern triggers more kernel page-zeroing trips than
   rapidgzip's.

5. **gzippy-pure's IPC is 1.01 vs rapidgzip's 1.63**. Even with more
   instructions, gzippy retires fewer per cycle — significant
   memory-subsystem stall.

6. **gzippy-pure has 3.27× the context switches**. Worker
   synchronization is more expensive than rapidgzip's.

## B. Symbolized perf-record top-15

### B.1 gzippy-pure

```
21.09%  std::thread::local::LocalKey<T>::with        ← marker bootstrap
16.86%  __memmove_avx_unaligned_erms (libc)
 9.54%  clear_page_erms (kernel)
 8.31%  submit_post_process_to_pool::closure         ← consumer dispatch
 8.16%  decode_huffman_body_resumable                ← BULK INFLATE
 5.11%  copy_match_windowed                          ← BULK MATCH COPY
 2.28%  HuffmanCodingReversedBitsCached::decode      ← dist decode
 1.78%  ResumableInflate2::read_stream_inner
 1.64%  get_distance_dynamic
 1.64%  submit_decode_to_pool::closure
 1.52%  LitLenTable::build
 1.35%  IsalLitLenCodePure::rebuild_from
 1.06%  try_charge_memcg (kernel memcg)
 0.94%  memchr searcher_kind_avx2
 0.88%  __rmqueue_pcplist (kernel allocator)
```

### B.2 rapidgzip

```
35.75%  Block::read (the actual deflate decode, all-in-one)
11.74%  DecodedData::applyWindow                     ← marker → clean
 7.72%  clear_page_erms (kernel)
 6.09%  __memmove_avx_unaligned_erms (libc)
 3.45%  ..@37.end                                    ← ISA-L asm symbol
 3.25%  GzipChunk::decodeChunkWithRapidgzip          ← chunk dispatcher
 2.97%  BitReader::peek2
 2.73%  crc32_gzip_refl_by8_02.fold_128_B_loop       ← CRC SIMD
 2.56%  _copy_to_iter (kernel write syscall)
 2.01%  ..@42.end                                    ← ISA-L asm symbol
 1.72%  loop_block                                   ← ISA-L asm symbol
 1.48%  blockfinder::seekToNonFinalDynamicDeflateBlock
 1.41%  __memset_avx2_unaligned_erms (libc)
 0.98%  make_inflate_huff_code_lit_len
 0.96%  large_byte_copy
```

### Side-by-side structural comparison

| Functional layer | gzippy-pure | rapidgzip | Δ |
|------------------|-------------|-----------|---|
| Inflate inner loop | 8.16 + 5.11 + 2.28 + 1.78 = **17.3%** | 35.75% (all-in-one) | rapidgzip's inner is BIGGER %, but that's because total cycles are lower — absolute time spent is less |
| Marker / window swap | 21.09 (LocalKey marker) | 11.74 (applyWindow) | gzippy 1.8× more |
| memmove + clear_page | 16.86 + 9.54 = **26.4%** | 6.09 + 7.72 = **13.8%** | gzippy 1.9× more |
| Consumer/dispatcher | 8.31 + 1.64 = **9.95%** | 3.25 = **3.25%** | gzippy 3.1× more |
| CRC | not in top-15 | 2.73% (SIMD VPCLMULQDQ) | gzippy doesn't use the SIMD variant |

## C. Where the gap lives — operational interpretation

### C.1 The inflate inner loop is NOT the dominant lever

Combined inflate-inner-loop time (decode_huffman_body_resumable +
copy_match_windowed + HuffmanCodingReversedBitsCached + read_stream_inner):
**17.3% of gzippy-pure CPU**.

Even if gzippy's inner loop matched ISA-L's speed exactly, the savings
would be ~7-10% absolute CPU (the gap to rapidgzip's 35.75% all-in-one
inner is misleading — rapidgzip's inner is a bigger % because its
total runtime is lower).

The session's 5 inflate-inner-loop falsifications + 1 win (T3-simplify
+1.9%) are consistent with this attribution.

### C.2 The structural levers worth ~10-15% combined

1. **Marker bootstrap path (21.09% gzippy vs 11.74% rapidgzip = 9.35pp gap)**
   - gzippy uses `LocalKey<T>::with` and `RefCell::borrow_mut` for the
     thread-local Block; rapidgzip's `applyWindow` is a direct memory
     op without RefCell pessimism
   - Lever: replace RefCell with an unsafe-Send static or per-worker
     owned Block

2. **memmove + clear_page (26.4% vs 13.8% = 12.6pp gap)**
   - gzippy has 2.28× the page faults
   - Rapidgzip likely uses MAP_HUGETLB on its chunk staging buffer
     OR rotates a smaller fixed buffer pool — needs source dive
   - Lever: replace gzippy's chunk_buffer_pool with mmap-backed
     huge-page buffer

3. **Consumer/dispatcher (9.95% vs 3.25% = 6.7pp gap)**
   - gzippy's `submit_post_process_to_pool` closure body is 8.31%
   - Rapidgzip dispatches more efficiently
   - Lever: profile the closure body, find the costly subop

4. **CRC SIMD (gzippy not in top-15 vs rapidgzip 2.73%)**
   - rapidgzip uses CRC32 with VPCLMULQDQ AVX2 SIMD (~3× faster than
     scalar)
   - gzippy currently uses crc32fast crate; check if AVX2 path is hit

### C.3 The inflate inner loop is the LAST 5-10%

Even if all four structural levers above land:
- Marker bootstrap: gzippy 21.09% → rapidgzip 11.74% = save 9.35pp
- Allocator: gzippy 26.4% → rapidgzip 13.8% = save 12.6pp
- Dispatcher: gzippy 9.95% → rapidgzip 3.25% = save 6.7pp
- CRC SIMD: save ~3pp

Combined potential savings: ~31pp absolute CPU. That's WAY more than
the current gap suggests we need.

But: 31pp absolute CPU saved doesn't translate 1:1 to throughput.
Other work fills the saved cycles. Realistic throughput gain from
closing all four structural gaps: ~30-50% improvement, bringing
gzippy-pure from 2.17× rapidgzip cycles to maybe 1.3-1.4×.

To match rapidgzip 1:1, ALSO need the inner inflate at parity.
That's the structural SIMD work (BMI2 PEXT, AVX2 vpshufb,
speculative-parallel lookups) the prior advisor identified — multi-
week each.

## D. Wall-time measurements (high variance under load 47)

Skipped — system load average is 47 right now. Wall times ranged
1.7s to 182s for the same workload. Counter data above (per-workload
event counts) is robust to load and is what should drive decisions.

A clean wall-time bench under load < 5 will land in a future session
following the methodology in `docs/perf/2026-05-28-corrected-gap-measurement.md`.

## E. Raw perf data on disk

```
/tmp/perfdata/pure.stat   — perf-stat output for gzippy-pure
/tmp/perfdata/isal.stat   — perf-stat output for gzippy-isal
/tmp/perfdata/rg.stat     — perf-stat output for rapidgzip
/tmp/perfdata/pure.data   — perf.data for gzippy-pure (DWARF)
/tmp/perfdata/rg.data     — perf.data for rapidgzip (DWARF)
/tmp/perfdata/pure.report — perf-report flat top-30 for gzippy-pure
/tmp/perfdata/rg.report   — perf-report flat top-30 for rapidgzip
```

## F. Key actionable findings (priority-sorted)

1. **STOP attacking the inflate inner loop in isolation.** It's
   17.3% absolute CPU — even at 0% it can't close the 2.17× cycle
   gap to rapidgzip.

2. **Attack the marker-bootstrap LocalKey path** — replace
   `LocalKey<T>::with` + `RefCell<Block>` with `OnceCell<Block>` or
   per-worker owned Block. Expected: -8 to -10pp absolute CPU.

3. **Profile the allocator pattern.** rapidgzip allocates 2.28× less
   page-fault traffic. Source dive at rapidgzip's `ChunkData`
   allocation to identify the pattern (MAP_HUGETLB? pool size?).
   Expected: -10 to -13pp absolute CPU.

4. **Verify CRC SIMD path.** gzippy uses `crc32fast`. Confirm it's
   selecting the AVX2 VPCLMULQDQ variant on neurotic. Expected:
   ~2-3pp if it's not.

5. **Audit consumer/dispatcher closure** for hot suboperations.
   Expected: ~3-5pp.

6. **THEN** revisit the inflate inner loop with the SIMD techniques
   (BMI2 PEXT for bit extract, AVX2 vpshufb for literal output,
   speculative-parallel LUT lookups). Expected: the LAST 5-10%.
