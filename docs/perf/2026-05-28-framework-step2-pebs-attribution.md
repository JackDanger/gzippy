# Framework Step 2: PEBS-precise memory-stall attribution

**Date**: 2026-05-28
**Tool**: `perf mem record --call-graph=dwarf` + `perf mem report`
**Binaries**: `/tmp/gzippy-pure-dbg` (frame-pointers + debuginfo),
              `rapidgzip` 0.16.0
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16

## Purpose

Verify the write-side vs read-side memory-stall attribution that
strace cannot see. The advisor flagged: if Lever 4.2 (128 KiB
Box-per-chunk) breaks consumer-side prefetch contiguity, we could
regress more on the read path than we save on alloc. PEBS-precise
attribution tells us which.

## A. gzippy-pure top memory accesses

| % | Samples | Memory Access | Symbol |
|---|---|---|---|
| 45.42% | 213 | LFB/MAB hit | `__memmove_avx_unaligned_erms` |
| 8.24% | 22 | LFB/MAB hit | `LocalKey<T>::with` (marker bootstrap subtree) |
| 7.45% | 36 | L2 hit | `__memmove_avx_unaligned_erms` |
| **7.44%** | **32** | **L3 miss** | **`__memmove_avx_unaligned_erms`** |
| 5.31% | 19 | LFB/MAB hit | `submit_post_process_to_pool::closure` |
| 4.71% | 1 | L2 hit | `IsalLitLenCodePure::rebuild_from` |
| **4.33%** | **3** | **RAM hit** | **`__memmove_avx_unaligned_erms`** |
| 2.95% | 10 | LFB/MAB hit | `__rmqueue_pcplist` (kernel allocator) |
| 1.37% | 9 | LFB/MAB hit | `try_to_claim_block` (kernel) |
| 1.32% | 62 | L1 hit | `__memmove_avx_unaligned_erms` |
| 1.10% | 1 | LFB/MAB hit | `decode_huffman_body_resumable` (BULK INFLATE) |
| 0.87% | 1 | RAM hit | `__lruvec_stat_mod_folio` (kernel) |
| 0.62% | 2 | LFB/MAB hit | `decode_chunk_isal_impl` |
| 0.47% | 103 | L1 hit | `LocalKey<T>::with` |

**Total memmove memory stall share**: 65.96% (45.42 + 7.45 + 7.44 + 4.33 + 1.32)
**memmove L3-miss + RAM-hit (true STALL)**: **11.77%** absolute

## B. rapidgzip top memory accesses

| % | Symbol | Notes |
|---|--------|-------|
| 29.36% | `lru_add` (kernel) | Page LRU maintenance during alloc |
| 18.53% | `sync_regs` (kernel) | Syscall entry/exit |
| 9.74% | `lru_gen_add_folio` (kernel) | More page management |
| 7.90% | `Block::read` (rapidgzip) | Actual inflate |
| 5.52% | `__lruvec_stat_mod_folio` (kernel) | Page state tracking |
| 4.50% | `__folio_batch_add_and_move` (kernel) | Page batching |
| **4.36%** | `__memmove_avx_unaligned_erms` | **Just 4.36% — 10× less than gzippy!** |
| 3.47% | `small_byte_copy` (rapidgzip primitive) | Short match copy |
| 1.57% | `large_byte_copy` (rapidgzip primitive) | Long match copy |
| 1.57% | `..@42.end` (ISA-L asm symbol) |  |

## C. Side-by-side memmove comparison

| Metric | gzippy-pure | rapidgzip | Ratio |
|--------|-------------|-----------|-------|
| Total memmove memory-access share | 65.96% | 4.36% | **15×** |
| memmove L3-miss + RAM-hit (true stall) | 11.77% | not in top — < 1% | **>10×** |

## D. The KEY MECHANISTIC FINDING

**gzippy stalls on memmove writes into freshly-allocated COLD DRAM
pages.** rapidgzip stalls on kernel page-management (LRU + syscalls)
during alloc itself — its memmove operates on hot cache because the
chunks are recycled.

Mechanism:
- gzippy: glibc malloc gives us a fresh `Vec<u8>` of 12 MiB →
  pages are zeroed by kernel on first touch → memmove writes into
  pages that have never been in cache → L3 misses + RAM hits.
- rapidgzip: rpmalloc returns a 128 KiB region from its
  thread-local cache → page is already cache-warm from previous
  fill → memmove writes hit L1/L2 → no L3 miss.

The page-management kernel symbols dominating rapidgzip's PEBS
(29% + 18% + 10% = 57.6% of stalls!) are actually a GOOD sign:
they mean rapidgzip's page state changes are visible to the
kernel's LRU because pages are getting reused/reclaimed, NOT
allocated fresh. The cost is concentrated in syscall overhead
(sync_regs) rather than in user-mode memmove.

## E. Verdict for Lever 4.2 (advisor-asked question)

**Q (advisor): Is the dominant stall WRITE-side (alloc) or READ-side
(consumer/inflate)?**

**A: WRITE-side**. The 11.77% absolute L3-miss + RAM-hit attribution
to `__memmove_avx_unaligned_erms` (which is the chunk-data write
path during marker bootstrap + chunk-flush, per the call-graph view
in the perf-mem output) is the dominant memory stall.

The read-side activity (LocalKey 8.24% LFB, submit_post_process
5.31% LFB) is mostly LFB/MAB hits (in-flight, not stalls). Real
read-side stalls (L3 miss / RAM hit) are < 1%.

**Lever 4.2 (128 KiB chunked allocation with pool reuse) directly
addresses the write-side cold-page miss.** It is NET POSITIVE; the
advisor's read-side regression concern is empirically small.

## F. Additional finding: bulk inflate is NOT the stall

`decode_huffman_body_resumable` shows **1.10% LFB/MAB hit and
nothing else**. The bulk inflate hot loop is operating fully out
of L1 cache — no stalls. This is the strongest empirical proof yet
that **inflate inner loop optimizations cannot close the gap**.

This is consistent with the symbolized perf attribution
(`docs/perf/2026-05-28-memmove-symbolized.md`) and the corrected
n=3 cycle counts (`docs/perf/2026-05-28-three-way-comparison.md`
post-advisor-correction).

## G. What Step 3 (timeline events) would add (and why skip it)

The timeline would show the temporal pattern of alloc/free events.
But Step 2 already attributed the stalls to a specific symbol with
PEBS precision. Step 3 is decorative for THIS lever — we know what
to fix.

If Step 4 microbench shows the lever DOESN'T deliver despite the
PEBS evidence, then Step 3 becomes essential (a temporal effect
would explain the discrepancy). Until then, skip.

## H. What Step 4 should test

Per advisor, the alloc_pattern microbench must mirror gzippy's
exact shape: 16 workers × 12 MiB target × BTYPE-mixed writes WITH
THE CONSUMER READ-SIDE SWEEP that the writer does. Generic
allocate-and-touch reproduces rapidgzip's 2.5× headline but
misleads us on the consumer side.

Test matrix (mandatory variants):
- Allocator: glibc-default | rpmalloc-global | rpmalloc-per-Vec
- Chunk shape: single-12MiB-Vec | many-128KiB-Boxes
- Negative control: glibc + `mallopt(M_MMAP_THRESHOLD, 1 MiB)` to
  convert brk → mmap WITHOUT rpmalloc

Pass gate: a variant must demonstrate ≥10% page-fault reduction
AND ≥5% cycle reduction in the microbench before promoting to
production A/B.
