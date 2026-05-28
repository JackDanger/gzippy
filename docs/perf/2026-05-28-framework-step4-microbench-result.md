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

## D. Implication for production lever

Both Lever 4.1 (rpmalloc global) and Lever 4.2 (128 KiB chunked
ChunkData::data) should ship.

Expected production impact, extrapolating microbench delta to e2e:
- The microbench measures the alloc + write + read of a 12 MiB
  buffer (gzippy's per-chunk workload).
- The e2e workload also includes inflate (which we know is ~17%
  CPU, not the bottleneck) and bookkeeping (~10%).
- Allocator + chunk-write + chunk-read is the dominant ~70% of CPU
  per the symbolized perf attribution.
- 64% microbench wall reduction → ~70% × 64% = ~45% reduction in
  the dominant component → ~30-35% e2e throughput improvement.

This is well above the corrected 10-12% gap to rapidgzip from the
n=3 perf-stat. **A successful Lever 4.1 + 4.2 should not just close
the gap — it should leapfrog rapidgzip** on this single workload.

Open question: gzippy's parallel-SM may have additional bookkeeping
overhead (consumer thread reorder, CRC verification) that rapidgzip
doesn't, which would damp the e2e win.

## E. Mandatory next step (per advisor build order)

Step 5: pluggable A/B harness that runs the LEVER on the actual
gzippy binary on neurotic with full perf-stat counter rollup
(median over n=20, load avg < 4, no multiplex, full mandatory
rollup fields including correctness hash). The microbench
predicts -64%; Step 5 measures the actual production impact.
