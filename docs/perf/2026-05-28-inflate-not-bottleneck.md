# Inflate inner loop is NOT the dominant bottleneck

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `e794e70`
**Host**: neurotic LXC 199 (`-C target-cpu=native`)
**Build**: `cargo build --release --features pure-rust-inflate`
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16

## Fresh perf-record top-15 (cpu-cycles, --call-graph dwarf -F 999)

```
18.07%  __memmove_avx_unaligned_erms (libc)
13.26%  clear_page_erms (kernel)
 8.21%  gzippy unsymbolized (release binary)  ← top inflate hotspot
 1.84%  __rmqueue_pcplist (kernel allocator)
 1.16%  gzippy unsymbolized
 0.99%  gzippy unsymbolized
 0.94%  gzippy unsymbolized
 0.86%  gzippy unsymbolized
 0.74%  gzippy unsymbolized
 0.70%  smp_call_function_many_cond (kernel)
 0.68%  gzippy unsymbolized
 0.65%  __memset_avx2_unaligned_erms (libc)
 0.62%  native_queued_spin_lock_slowpath (kernel)
 0.62%  native_irq_return_iret (kernel)
 0.59%  gzippy unsymbolized
```

## The dominant cost is allocator-side

- **`__memmove_avx_unaligned_erms` (libc)**: 18.07%
- **`clear_page_erms` (kernel)**: 13.26%
- **`__memset_avx2_unaligned_erms`**: 0.65%
- **`__rmqueue_pcplist` (kernel page allocator)**: 1.84%
- Combined: **~33.8% absolute CPU**

The `clear_page_erms` cost is the kernel zeroing pages when the
output buffer is first touched (lazy page allocation). For a 212 MB
output buffer of 4 KiB pages, that's 51,750 page faults. Even at
~5,000 cycles per zeroed page, that's significant absolute time.

## Inflate inner loop is ~8-15%

The top non-allocator gzippy symbol is **8.21%** (unsymbolized in
release-stripped binary). That's `decode_huffman_body_resumable` or
similar inner loop. Several other gzippy symbols at 0.7-1.2% are
ancillary (refill, LUT lookup, etc).

**Conservative estimate: inflate code totals ~15% absolute CPU.**

## Why my last three lever attempts were at parity

- **Route C v3.7/v3.9** (hand-asm literal decode): at parity with
  rustc on the literal loop. But even if it had been 2× faster, the
  literal loop is only a fraction of the 15% inflate CPU. Saving 50%
  of the literal loop = 3-5% absolute CPU = +4-6% throughput. Hard to
  detect in noise.
- **S1 packed u32 literal store**: same reason. Literal stores are a
  small fraction of inflate CPU.
- **S2 bulk window copy**: same. The slow path is rare.

## The real ladder

To close the 72% gap (pure-rust 655 MB/s → ISA-L FFI 1130 MB/s) at
T=16 silesia-gzip9, the dominant lever is **allocator-side**, not
inflate-side:

1. **Eliminate clear_page_erms**: the kernel zeros each page on first
   touch. Pre-touch via MAP_POPULATE OR reuse a long-lived buffer
   that's already zeroed.
   - Z-allocator 4 MiB prewarm was attempted and falsified
     (`feedback_z_allocator_prewarm_falsified.md`); -15% wall.
     Hypothesis was wrong about WHICH allocation to prewarm.
   - Actual fix likely needs a global per-process pool of warmed
     output buffers, sized to the chunk count × chunk size.

2. **Reduce `__memmove_avx_unaligned_erms`**: 18% CPU in memmove.
   Probably output buffer copies between chunk decoder and final
   sink. Eliminate intermediate copies → write inflate output
   directly into the final buffer.

3. **Inflate inner loop**: only a 15% absolute CPU lever even with
   2× speedup. Diminishing returns compared to allocator.

## Recommendation for the next session

**STOP chasing inflate-inner-loop levers.** Three falsifications in a
row (Route C v3.7/v3.9, S1, S2) all confirm: the inflate inner loop
is NOT where the 72% gap lives. Fresh perf data localizes the gap to
allocator-side work.

Next session should:
1. Symbolize the gzippy binary (build with `debug = true`) and
   capture flamegraph to identify which gzippy functions own the
   18% memmove.
2. Reattempt the warmed-output-buffer pool with the correct shape
   (avoid the Z-allocator falsification — that prewarmed the wrong
   buffer).
3. Or: look at the parallel-SM pipeline structure for an eliminable
   intermediate copy.

Inflate-inner-loop work can resume AFTER allocator is closed to
within 5% of ISA-L. Until then, it's the wrong lever.
