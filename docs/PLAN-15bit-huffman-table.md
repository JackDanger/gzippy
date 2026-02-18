# 15-bit Primary Huffman Table

## Goal
Exceed libdeflate's single-stream decompression speed by 5-10% through wider primary tables.

## Background

### Current approach (libdeflate-style, 11-bit primary)
- Primary table: 2048 entries (2^11), each 32 bits = 8KB
- Subtable: variable-size overflow for codes >11 bits
- ~5-15% of lookups hit the subtable (extra memory access + branch)

### Proposed approach (15-bit primary)
- Primary table: 32768 entries (2^15), each 32 bits = 128KB
- No subtable needed — deflate litlen codes are max 15 bits, distance codes max 15 bits
- Every lookup resolves in a single table access — zero branches, zero overflow

### Why this hasn't been tried
The beat-all-decompression branch (25+ commits, 22K lines) focused on:
- Hand-written ASM (failed: LLVM beats inline asm)
- JIT compilation (failed: can't beat O(1) table lookup)
- ISA-L algorithm port (failed: O(n^2) table build cost)
- Multi-symbol packing (failed: build cost > decode savings)

Nobody tried simply making the table wider. The 11-bit size was copied from libdeflate
without questioning whether a larger table would be faster on modern CPUs.

### Why it might work
- 128KB fits in L1 data cache on Apple M-series (192KB L1d) and recent Intel (48-80KB L1d)
- Eliminates ALL subtable branches — the subtable path is the main source of branch mispredictions
- Simpler decode loop means LLVM can optimize more aggressively
- Table build cost increases from O(2^11) to O(2^15) — 16x more entries, but table build
  is <5% of total time (per the profiling on beat-all-decompression)

### Why it might not work
- 128KB may thrash L1 cache on x86 CPUs with 32-48KB L1d
- The litlen table and distance table together would be 256KB
- Distance codes rarely exceed 11 bits, so the subtable path is rare for distances
- If subtable hits are already rare (<5%), eliminating them saves <5%

## Approach

### Phase 1: Litlen table only (safest)
- Widen litlen primary table from 11-bit to 15-bit (8KB → 128KB)
- Keep distance table at 11-bit (subtable hits are very rare for distances)
- Modify `consume_first_table.rs`: change `LITLEN_TABLEBITS` from 11 to 15
- Modify `consume_first_decode.rs`: remove subtable fallback in litlen path
- Benchmark on SILESIA, SOFTWARE, LOGS

### Phase 2: If Phase 1 wins on ARM, try x86
- ARM has larger L1 (192KB M-series), more likely to benefit
- x86 has smaller L1 (32-48KB), more likely to regress
- May need architecture-specific table sizes

### Phase 3: If both win, consider distance table too
- Only if litlen improvement is significant (>3%)

## Measurement plan
- `cargo test --release bench_cf_silesia -- --nocapture` (before and after)
- `cargo test --release bench_diversity -- --nocapture` (all datasets)
- Run 50+ iterations to control for variance
- Compare: table build time, decode throughput, cache miss rate (perf stat)

## Risk
**Low-medium.** The change is ~50 lines in table building + ~20 lines removing subtable fallback.
Easy to revert if it regresses. The git history shows that EVERY other micro-optimization
regressed, so skepticism is warranted — but this is the one structural change nobody tried.
