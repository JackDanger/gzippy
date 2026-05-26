# SIMD marker scan — ABANDONED after re-measurement

## Status

**Lever rejected after corrected measurement.** Original plan claimed marker scan was ~7% of cycles based on a bootstrap-only bench. On silesia full-pipeline, the marker scan is ≤1.5% of cycles, putting the maximum SIMD win below the noise floor.

## What the corrected measurement showed

`perf record -F 999` on `taskset -c 1,3,4,5,6,7,10,13,15 gzippy -d -c -p 9 silesia-gzip.tar.gz` with symbols enabled (`-Cstrip=none -Cdebuginfo=2`):

```
17.12%  __rust_begin_short_backtrace (main consumer thread)
         |--5.71%  BlockFinder::BitReader::refill
         |--4.06%  BlockFinder::find_dynamic_blocks
         |--3.25%  BlockFinder::BitReader::skip
         |--1.64%  BlockFinder::BitReader::seek_to_bit
         |--1.33%  BlockFinder::validate_precode
         └ ... (block-finder helpers)

10.80%  bootstrap_with_deflate_block (worker bootstrap entry)
         |--5.69%  Block::read_internal_compressed_specialized (Huffman inner loop)
         |--2.84%  emit_backref_ring (back-ref copy + marker scan)
         └--1.02%  Bits::consume

 9.76%  libc memcpy
 3.14%  kernel clear_page (page fault on first-touch alloc)
```

**emit_backref_ring is 2.84% of total cycles.** Of that, the marker scan is roughly half (the other half is the actual back-ref copy via `copy_nonoverlapping` and the branch/RLE-special logic).

**Maximum theoretical win from SIMD marker scan**: 1.5% × (1 − 1/8) ≈ 1.3% cycles ≈ 4.6 ms on a 356 ms baseline. Below noise floor.

## Errors in original plan (per Opus critique)

1. **Wrong build-flag assumption**: claimed `target-cpu=x86-64-v3` was default; it's not. AVX2 cfg-gates would silently dead-code the SIMD path. Would need runtime dispatch via `is_x86_feature_detected!`.

2. **Wrong cycle-share estimate**: extrapolated from bootstrap-only bench (`bench_corpus_aggregate/bootstrap_path` Criterion test, all-marker-mode synthetic) to silesia full-pipeline. The real share is ~5× smaller because the mid-decode mode switch (`deflate_block.rs:738-744`) flips `contains_marker_bytes` to false after 32 KiB clean output, dead-stripping the marker scan for most cycles.

3. **Inconsistent share→wall arithmetic**: predicted "-2.84pp share drop → 2-4% wall" while assuming "~7% marker scan share". Either prediction or assumption was wrong; both turn out to be.

## What surfaced as a real lever instead

**BlockFinder bit reader: 17.12% of total cycles on silesia full-pipeline.** Running on the main/consumer thread (per the `__rust_begin_short_backtrace` parent in the perf graph), so it's on the critical path that defines wall.

The BlockFinder scans the compressed stream looking for deflate block boundaries during prefetch dispatch (to know where to start speculative worker decodes). It's a bit-by-bit scanner that:
- Reads bits via `BitReader::refill` / `read` / `skip` / `seek_to_bit`
- Validates precode candidates via `validate_precode`
- Walks until a valid dynamic-Huffman block header is found

Per the cycle distribution:
- 5.71% in `refill` alone (loading bit chunks)
- 4.06% in `find_dynamic_blocks` (the outer scan)
- 3.25% in `skip` (bit-position advancement)
- 1.64% in `seek_to_bit` (absolute repositioning)

This is a much bigger lever (17% vs 1.5%) AND likely SIMD-friendly (the precode validation is essentially a Huffman code-lengths-consistency check; the bit reader is uniform bit consumption).

**Recommendation**: redirect to BlockFinder optimization. Pending a fresh plan.

## What this episode taught us

- **Measurement matters more than estimates.** The 13.84% from `bootstrap_path` Criterion bench was a useful number for that bench, but wrongly applied to silesia full-pipeline. A 10-minute perf-record on the actual workload would have prevented a full plan + critique cycle.
- **Always measure on the production workload before optimizing.** This is exactly what Opus's profile-driven-exhaustion sign-off criterion meant by "instrumentation before refactor" — but applied here, it'd say "measurement before plan."

## Plan abandoned

No code changes from this plan. Next plan: BlockFinder optimization.
