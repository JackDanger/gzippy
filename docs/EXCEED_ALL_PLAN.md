# EXCEED ALL: The Only Optimization Plan

**Goal: 130%+ of libdeflate single-threaded (1840+ MB/s), beat rapidgzip parallel**

## Current Baseline (Jan 2026)

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Single-thread silesia | **970 MB/s (69.7%)** | 1840+ MB/s (130%) | +90% needed |
| libdeflate reference | 1390 MB/s (100%) | Beat it | - |
| Parallel BGZF (8 threads) | ~3000 MB/s | 4000+ MB/s | +33% needed |

---

## What's ACTUALLY Implemented ✓

| Optimization | Status | Verified |
|--------------|--------|----------|
| Consume-first pattern (`bits.consume_entry(entry)`) | ✅ Done | Line 100 |
| Signed literal check `(entry as i32) < 0` | ✅ Done | Line 291 |
| Unsafe unchecked table lookups (`get_unchecked`) | ✅ Done | Multiple |
| 5-literal unrolling in fastloop | ✅ Done | Lines 293-347 |
| Preload-before-write (entry lookup before copy) | ✅ Done | Lines 425, 460 |
| Unsafe match copy with 40-byte overwrite | ✅ Done | `copy_match_fast` |
| 11-bit litlen table, 8-bit distance table | ✅ Done | libdeflate_entry.rs |
| JIT table cache (fingerprint + HashMap) | ✅ Done | consume_first_table.rs:529 |
| Static fixed table caching (OnceLock) | ✅ Done | libdeflate_decode.rs |
| Branchless 8-byte refill | ✅ Done | bits.refill() |
| Distance=1 memset optimization | ✅ Done | copy_match_fast |
| 8-byte overlapping copy for dist≥8 | ✅ Done | copy_match_fast |

---

## What's NOT Implemented Yet (Real Gaps)

### High Impact (10%+ each) — Must Do

| # | Optimization | Source | Expected | Status |
|---|--------------|--------|----------|--------|
| 1 | **Double-literal cache** | rapidgzip | +15-25% | ⚠️ Only helps fixed blocks; silesia is dynamic |
| 2 | **BMI2 `_bzhi_u64` intrinsic** | libdeflate | +5-10% | ⚠️ N/A on Apple Silicon |
| 3 | **Multi-literal decode (3 at once)** | ISA-L | +10-15% | Pending |
| 4 | **Inline assembly inner loop** | ISA-L | +10-20% | Pending |

### Medium Impact (5-10% each) — Should Do

| # | Optimization | Source | Expected | Effort |
|---|--------------|--------|----------|--------|
| 5 | SIMD literal batching (16 at once) | Novel | +5-10% | Medium |
| 6 | Prefetch next input chunk | general | +2-5% | Easy |
| 7 | 12-bit litlen table (reduce subtables) | libdeflate | +3-5% | Easy |
| 8 | Combined length+distance LUT | rapidgzip | +5-8% | Medium |

### Parallel Improvements

| # | Optimization | Source | Expected | Effort |
|---|--------------|--------|----------|--------|
| 9 | Work-stealing thread pool | pigz | +10-20% | Medium |
| 10 | NUMA-aware allocation | general | +20-40% (NUMA) | Hard |
| 11 | Marker-based single-member parallel | rapidgzip | 2-5x | Hard |

---

## Implementation Order

### Phase 1: Architecture-Independent Optimizations

1. ~~**BMI2 intrinsics**~~ — N/A on Apple Silicon (ARM64), module ready for x86_64
2. ~~**12-bit litlen table**~~ — Broke correctness, needs more investigation  
3. **Prefetch** — Add `prefetch_read` after each refill (TODO)

### Phase 2: REVISITED - Novel Optimizations Needed

The "standard" optimizations from libdeflate either:
- Aren't applicable (BMI2 on ARM)
- Break correctness (12-bit table)  
- Only help fixed blocks (DoubleLitCache)

**New approach**: Focus on what we CAN do:
- **Inline assembly for ARM64** (NEON, not BMI2)
- **Restructure decode loop** to match libdeflate's control flow exactly
- **Profile-guided optimization** to find actual bottlenecks

### Phase 3: Multi-Literal Decode (Target: 110% → 1556 MB/s)

5. **ISA-L style triple-literal decode** — Decode 3 literals per iteration
6. **SIMD literal batching** — Buffer and write 16 literals at once

### Phase 4: Assembly Polish (Target: 130% → 1840 MB/s)

7. **Inline assembly fastloop** — Hand-tune the critical ~50 lines
8. **Combined length+distance LUT** — Single lookup for common matches

---

## Validation Protocol

For EVERY change:
```bash
# 1. Baseline (run 3x, take median)
cargo test --release consume_first_decode::tests::bench_cf_silesia -- --nocapture

# 2. Make change

# 3. Test correctness
cargo test --release

# 4. Measure (run 3x, take median)
cargo test --release consume_first_decode::tests::bench_cf_silesia -- --nocapture

# 5. Commit if improvement > 2%, revert otherwise
```

---

## Reference Implementations

| Tool | File | Key Insight |
|------|------|-------------|
| libdeflate | `libdeflate/lib/decompress_template.h` | Fastloop structure, BMI2 |
| rapidgzip | `rapidgzip/.../HuffmanCodingDoubleLiteralCached.hpp` | Double-literal cache |
| ISA-L | `isa-l/igzip/igzip_decode_block_stateless.asm` | Triple-symbol decode |

---

## Expected Trajectory

| Phase | MB/s | % of libdeflate | Cumulative Gain |
|-------|------|-----------------|-----------------|
| Current | 1016 | 71.8% | - |
| Phase 1 | 1132 | 80% | +11% |
| Phase 2 | 1345 | 95% | +32% |
| Phase 3 | 1556 | 110% | +53% |
| Phase 4 | **1840** | **130%** | +81% |

---

## Lessons Learned (Don't Repeat)

1. **Simulation benchmarks lie** — Always validate on real silesia data
2. **Never trust "% of libdeflate" from old docs** — Re-measure before each change
3. **consume_first worked** — Don't revert to check-first
4. **Preload-before-write works** — Already implemented correctly
5. **JIT cache exists** — No need to build it again
6. **DoubleLitCache exists** — Just needs integration

---

## Success Criteria

| Milestone | Throughput | Status |
|-----------|------------|--------|
| 80% of libdeflate | 1132 MB/s | ❌ |
| 90% of libdeflate | 1274 MB/s | ❌ |
| 100% of libdeflate | 1415 MB/s | ❌ |
| **130% of libdeflate** | **1840 MB/s** | ❌ TARGET |
