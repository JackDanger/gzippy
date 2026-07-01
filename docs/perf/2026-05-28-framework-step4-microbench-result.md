# Framework Step 4: alloc_pattern microbench result

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `e1242e3`
**Bench**: `cargo bench --bench alloc_pattern`
**Shape**: 16 workers × 12 MiB target × BTYPE-mixed writes
**n**: 10 iters per variant

## A. Neurotic (Linux x86_64, target-cpu=native)

### A.1 glibc allocator (gzippy production default today)

```
Baseline (singleVec+readON):              ~222.8 ms  (implied)
manyBoxes+readON:                          269.54 ms  +21.0%  ← WORSE
singleVec+readOFF (write-only):            122.28 ms  -45.1%
manyBoxes+readOFF (write-only):             35.62 ms  -84.0%  ← chunked
                                                              wins write-only
```

### A.2 rpmalloc global allocator

```
Baseline (singleVec+readON):              102.76 ms  ← 2.17× faster than
                                                       glibc baseline
manyBoxes+readON:                          79.04 ms  -23.1%  ← WIN
singleVec+readOFF (write-only):           144.30 ms  +40.4%
manyBoxes+readOFF (write-only):            53.50 ms  -47.9%
```

### A.3 Combined deltas vs glibc+singleVec+readON

| Variant | wall (ms) | Δ from glibc baseline |
|---------|-----------|----------------------|
| glibc + singleVec + readON | ~222.8 | (baseline) |
| **glibc + manyBoxes + readON** | 269.5 | +21% WORSE |
| **rpmalloc + singleVec + readON** | 102.8 | **-54% FASTER** |
| **rpmalloc + manyBoxes + readON** | 79.0 | **-64% FASTER** ← target |

## B. Local arm64 (Apple Silicon, glibc-on-Mac/system allocator)

For contrast — shows the OS/allocator matters:

```
Baseline (glibc+singleVec+readON):              3.79 ms
manyBoxes+readON:                              +52.2%  ← OPPOSITE of Linux
singleVec+readOFF (write-only):                -51.5%
manyBoxes+readOFF (write-only):                -10.3%
```

On arm64 macOS the chunked pattern is +52% WORSE for full workload.
On Linux with rpmalloc it's -23% BETTER. **Allocator + OS prefetcher
make a categorical difference**; conclusions from one platform do not
transfer.

## C. Interpretation

### C.1 Lever 4.1 (rpmalloc global) is the dominant win

Switching from glibc to rpmalloc global allocator drops wall time by
**54%** on the baseline (singleVec+readON) workload. This alone is
larger than any inflate-inner-loop lever we attempted this session.

### C.2 Lever 4.2 (128 KiB chunks) is additive

On top of rpmalloc, switching from single 12 MiB Vec to many 128 KiB
Box<[u8; 128K]>> drops wall an additional **23%**. Total combined
improvement vs glibc baseline: **64%**.

### C.3 The advisor's read-side concern was platform-specific

On macOS arm64 with glibc/system allocator, chunked Boxes are +52%
WORSE because the OS prefetcher loses contiguity at chunk boundaries.
On Linux x86_64 with rpmalloc, chunked Boxes are -23% BETTER because
rpmalloc's thread-local cache keeps recently-allocated 128 KiB regions
hot and the Linux prefetcher handles the chunk boundaries fine.

The advisor was right to flag the concern. The microbench caught the
platform difference in 30 seconds — exactly what the framework was
designed for.

### C.4 Page-fault deltas are 0 in PF mode

`GZIPPY_ALLOC_PF=1` shows minor=major=0 for all variants. The
`/proc/self/stat` parsing reads the values at the END of each variant
but the worker threads' faults aren't visible to the main thread's
stat counter at exit. To capture per-variant faults we'd need
`getrusage(RUSAGE_THREAD)` or `perf stat -e page-faults` per
variant. Earlier perf-stat already gave us the absolute counts
(~168K gzippy vs ~79K rapidgzip).

## D. ADVERSARIAL ADVISOR REVIEW — DOWNGRADED PREDICTION

The advisor caught **six substantive overestimation paths** in the
production extrapolation above. Corrected projection:

### D.1 Read pattern is wrong (most damning)

Production consumer (`chunk_fetcher.rs:2234-2273`) does
`writer.write_all(payload)` — one big contiguous memcpy to BufWriter.
The microbench's `for chunk in buf.chunks(4096) { sum literal bytes
}` is a **scalar load-and-reduce per byte** that has nothing in
common with `memcpy(stdout_buf, src, 12MiB)`.

The readON variant's "rpmalloc -54%" gain mostly measures hot-cache
scalar reduce cost — irrelevant to the production read path.

**readOFF numbers are closer to production-relevant**: -47.9% for
rpmalloc+manyBoxes vs glibc+singleVec is more honest, but the +84%
glibc+manyBoxes readOFF win is also irrelevant (it's measuring
scalar pointer-chase across Box headers vs Vec linear, not memcpy).

### D.2 Write shape is wrong

Real `ChunkData::data` writes via `extend_from_slice(bytes)` where
`bytes` is hundreds-of-bytes to KiB-sized inflate-output. Marker
bootstrap (`LocalKey<T>::with` at 21.09% per perf) writes u16 values
one-at-a-time. My fixed 64KiB+512B extends pre-amortize allocator
overhead over much larger writes than production.

### D.3 Extrapolation double-counts

My "~70% alloc share" was wrong. Honest decomposition of the
symbolized perf:
- memmove + clear_page = 26.4% (touched by allocator change)
- LocalKey 21.09% bootstrap subtree (PARTIALLY touched — only the
  alloc inside; the marker decode work isn't)
- submit_post_process 5.31% (touched)
- decode_huffman_body + copy_match_windowed = 13.3% **NOT touched**

Realistic allocator-touchable share: **~35-45%, not 70%**.

With production-shape microbench gain probably ~30% (not 64%):
`0.40 × 0.30 = 12% e2e`. That **exactly matches the 10-12% gap to
rapidgzip — closing it but NOT leapfrogging**.

### D.4 n=10 mean masks rpmalloc warmup

rpmalloc's TLS cache is empty on iteration 1; iterations 2-10 reuse
warm regions. Production gzippy spawns workers FRESH per
`decompress_parallel` invocation — production sees iteration-1 cost,
not iteration-2-10 cost. The honest stat is per-iter median, not
n=10 mean.

### D.5 arm64 read-regression may extend to Linux

When stdout is a real sink (write_all triggers kernel-side prefetch
on the SOURCE buffer), chunked Boxes break that prefetch on Linux
too. Step 5 must test BOTH `-o /dev/null` AND output-to-real-file.

### D.6 Don't stack with falsified prewarm

The Z-allocator prewarm was falsified at -15%. Ship Lever 4.1
standalone (or with explicit `GZIPPY_PREWARM_POOL=0`) before any
combined experiment.

## D'. Corrected expected production impact

| Original claim | Corrected estimate |
|---------------|--------------------|
| 30-35% e2e throughput | **~10-15% e2e** |
| Leapfrog rapidgzip | **Match rapidgzip (close the 10-12% gap)** |

This is still a major win — closes the entire measured gap — but it
does NOT exceed rapidgzip. The "fastest gzip ever" claim from
CLAUDE.md still requires either the inflate-inner SIMD work later
OR a fundamentally different architecture.

## E. Mandatory Step 5 tests (per advisor)

The production A/B harness MUST include:

1. **n=20 with per-iter median** (NOT mean) AND a trial-1-vs-trial-N
   split to expose rpmalloc warmup amortization.
2. **Two output sinks**: `-o /dev/null` AND `-o <real_file>`. The
   real-file path triggers kernel-side prefetch on the source
   buffer; chunked Boxes might regress here.
3. **Full perf-stat rollup**: task-clock, page-faults (minor+major
   separately), L1/LLC/dTLB misses, IPC, cycles, instructions —
   page-fault delta is the load-bearing signal, not wall.
4. **Correctness hash** in rollup (CLAUDE.md rule 4) — rpmalloc +
   chunked alloc interacting with `allocator_api2::vec::Vec` is a
   real soundness surface.
5. **Multi-corpus**: silesia AND a low-redundancy file where marker
   bootstrap (LocalKey<T>::with) dominates. Silesia A/B may not
   exercise the worst-case allocator path.
6. **Standalone Lever 4.1 first**: ship rpmalloc-global with
   `GZIPPY_PREWARM_POOL=0` and confirm it doesn't regress before
   stacking Lever 4.2.

If Step 5 measures only silesia-to-/dev/null with n=20 median wall,
we'll get an inflated number that production (file output, mixed
corpora) won't reproduce.

## E. Mandatory next step (per advisor build order)

Step 5: pluggable A/B harness that runs the LEVER on the actual
gzippy binary on neurotic with full perf-stat counter rollup
(median over n=20, load avg < 4, no multiplex, full mandatory
rollup fields including correctness hash). The microbench
predicts -64%; Step 5 measures the actual production impact.
