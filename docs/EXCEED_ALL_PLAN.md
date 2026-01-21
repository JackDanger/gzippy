# EXCEED ALL: The Only Optimization Plan

**Goal: 130%+ of libdeflate single-threaded (1840+ MB/s), beat rapidgzip parallel**

## Current Baseline (Jan 2026)

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Single-thread silesia | **1016 MB/s (71.8%)** | 1840+ MB/s (130%) | +81% needed |
| libdeflate reference | 1415 MB/s (100%) | Beat it | - |
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

| # | Optimization | Source | Expected | Effort |
|---|--------------|--------|----------|--------|
| 1 | **Double-literal cache in hot path** | rapidgzip | +15-25% | Medium |
| 2 | **BMI2 `_bzhi_u64` intrinsic** | libdeflate | +5-10% | Easy |
| 3 | **Multi-literal decode (3 at once)** | ISA-L | +10-15% | Medium |
| 4 | **Inline assembly inner loop** | ISA-L | +10-20% | Hard |

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

### Phase 1: Low-Hanging Fruit (Target: 80% → 1132 MB/s)

1. **BMI2 intrinsics** — Add `#[target_feature(enable = "bmi2")]` and use `_bzhi_u64`
2. **12-bit litlen table** — Reduce subtable lookups from ~5% to ~1%
3. **Prefetch** — Add `prefetch_read` after each refill

### Phase 2: Double-Literal Integration (Target: 95% → 1345 MB/s)

4. **Integrate DoubleLitCache** — Use in fastloop for literal runs
   - Already built in `src/double_literal.rs`
   - Need to wire into `decode_huffman_cf`

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
