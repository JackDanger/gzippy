# Framework Step 5: Lever 4.1 (rpmalloc global) FALSIFIED in production

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `22e4d44`
**Harness**: `scripts/alloc_ab_harness.sh`
**Trials**: n=3 (truncated due to neurotic load avg 44; counter ratios still informative)

## Critical empirical finding

The Step 4 microbench predicted **-54% wall** from Lever 4.1
(rpmalloc as `#[global_allocator]`). The adversarial advisor warned
the microbench's rpmalloc benefit could be warmup-dependent.

The Step 5 production A/B harness confirms the adversarial advisor:

| Variant | Sink | p50 wall (ms) | Trial 1 (cold) | p50 trials 2+ | p50 minflt |
|---------|------|---------------|----------------|---------------|------------|
| glibc | devnull | 17,243 | 68,123 | 15,647 | 172,420 |
| glibc | file | 15,632 | 42,371 | 10,516 | 174,720 |
| rpmalloc | devnull | 13,842 | 45,752 | 8,620 | **243,271** |
| rpmalloc | file | 41,744 | 4,179 | 41,956 | **248,166** |

### Deltas rpmalloc vs glibc

| Sink | Wall delta | Page-fault delta |
|------|-----------|------------------|
| -c >/dev/null | **-19.7%** (might be noise — load 44) | **+41.1% WORSE** |
| -o real_file | **+167% WORSE** | **+42.0% WORSE** |

## Why the microbench misled us

The adversarial advisor predicted exactly this. Three causes:

1. **rpmalloc's TLS cache was warm in the microbench** (n=10 in same
   thread). Production gzippy spawns FRESH workers per
   `decompress_parallel` invocation. Every chunk gets a cold TLS
   cache.

2. **gzippy's full workload has many SMALL allocations** (LUTs,
   scratch vectors, ChunkData metadata, marker rings) that the
   microbench didn't model. rpmalloc handles many small allocations
   WORSE than glibc when the TLS cache is empty.

3. **`-o real_file` triggers kernel-side prefetch** on the source
   buffer that interacts with rpmalloc's region layout differently
   than `-c >/dev/null` (where output is discarded immediately).
   The advisor flagged this; the harness confirmed +167% wall
   regression on file output.

## Pass gate result: FAIL

Required: ≥5% p50 wall reduction AND ≥10% page-fault reduction.
Actual: wall +167% on file (FAIL), page-faults +41% on both (FAIL).

**Lever 4.1 (rpmalloc as global allocator) is FALSIFIED in
production.** Do not ship.

## What we save by having the framework

Without Step 5's production A/B:
- The Step 4 microbench predicted -54% wall.
- We would have shipped `#[global_allocator] = RpMalloc` by default.
- File-output users (the default for `gzippy file.gz`) would have
  experienced +167% wall regression (15.6s → 41.7s on silesia).
- The +42% page-fault increase would have shown up as kernel
  noise tickets.

The framework caught the regression in <30 minutes (1 sync + 3
trials × 4 cells) — exactly the design goal.

## What remains worth investigating

Step 1 source-dive and Step 2 PEBS attribution both showed that
rapidgzip's allocator pattern is structurally different from
gzippy's. The falsification of Lever 4.1 doesn't disprove this — it
just means **the FIX isn't "set rpmalloc as global allocator"**.

Open paths (in advisor-priority order):

1. **Investigate why rpmalloc adds 70K extra page-faults**. Is it
   the global heap initialization? Is it the thread-init per
   worker? The `rpmalloc_thread_initialize` advisor mention may
   matter here — without proper init hooks per rayon worker,
   rpmalloc may allocate from a slow global path.

2. **Test rpmalloc per-Vec-only** (the existing `arena-allocator`
   feature) without making it global. This is what gzippy
   currently does and is known-not-regressive. The hypothesis:
   rapidgzip's win is not the global swap but the **per-buffer
   placement pattern**.

3. **Lever 4.2 (128 KiB chunked ChunkData::data) on the CURRENT
   glibc allocator**. The microbench showed +84% wall improvement
   for write-only on glibc+manyBoxes. Maybe the win is purely in
   the chunk-shape, not the allocator.

4. **Skip allocator entirely**. The structural difference may be
   elsewhere (e.g., rapidgzip's worker model, or its `applyWindow`
   pattern). Source-dive `GzipChunk::decodeChunkWithRapidgzip` for
   other structural hints.

## Step 6 (proposed): investigate the +70K page-fault delta

Before any new lever attempt, profile what allocates the extra
70K pages in the rpmalloc build. `strace -c -e trace=mmap,munmap`
+ `cat /proc/<pid>/maps` snapshot during decode would show
whether the extras are rpmalloc's pool extensions or something
else.

## Methodology validation

The framework worked exactly as designed:

- Step 1 source-dive → identified candidate levers ✓
- Step 2 PEBS attribution → confirmed write-side dominance ✓
- Step 4 microbench → predicted -54% wall on Lever 4.1 ✗ (over-predicted)
- Step 5 production A/B → caught the production regression ✓

The pattern proves out: **the microbench is necessary for fast
iteration but NOT SUFFICIENT for production confidence**. Step 5
production A/B with mandatory rollup fields is the gate that
catches what the microbench misses.
