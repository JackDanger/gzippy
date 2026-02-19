# Plan to Beat Every Tool in Every Scenario

**Written**: Feb 2026  
**Status**: Active — work through phases in order  
**Goal**: gzippy wins every benchmark, every platform, every thread count, no exceptions

---

## The Fundamental Problem: We Test Shadows, Not Substance

There are **three different inflate implementations** in the codebase, each benchmarked
separately, none of them consistently identified as THE production path:

```
inflate_consume_first()         ← pure Rust (bench_cf_silesia)
inflate_into_pub()              ← libdeflate C FFI (bench_decompress)
decompress_gzip_to_writer()     ← full CLI path (CI benchmark_decompression.py)
```

The internal benchmarks report numbers that are never what the CLI ships:

| Test | What it measures | What CLAUDE.md says | What CI reports |
|------|-----------------|--------------------|--------------------|
| bench_cf_silesia | pure Rust inflate_consume_first, raw deflate | "1400 MB/s, 99% parity" | — |
| bench_decompress | libdeflate FFI inflate_into_pub, raw deflate | — | — |
| CI gzippy -d | full CLI: mmap + routing + FFI + write | — | 129-251 MB/s |

**The CLI is 5-11x slower than what we benchmark internally**, not because of a 
performance bug, but because we compare raw in-memory deflate decoding to end-to-end
file I/O. This is fine — but we call the internal benchmarks "our" performance and
conflate them with the CLI numbers. This is the source of confusion.

**Rule going forward**: Every internal `bench_*` function that measures raw inflate speed
must explicitly label whether it is testing (a) the production path or (b) an experimental
path. If it tests (b), it must show the current production path as the reference.

---

## Current State: Where We Win and Lose

### Wins (36/48 CLI matchups)
- **Compression L6/L9**: Every platform, every thread count — libdeflate quality wins
- **Compression L1 tmax**: x86 silesia, software (with ratio probe fix merged)
- **Decompression arm64 silesia**: 129-130 MB/s vs rapidgzip 120-121 (+8%)

### Losses (12/48, from CI data Feb 2026)

| Scenario | gzippy | Best | Tool | Gap | Root cause |
|----------|--------|------|------|-----|-----------|
| comp x86 software L1 t1 | 1694 | 2603 | igzip | **-35%** | igzip AVX-512 L1 encodes 3 symbols/cycle |
| comp arm64 logs L1 tmax | 761 | 1076 | pigz | **-29%** | ratio probe exits to single-stream |
| decomp x86 silesia tmax | 240 | 314 | rapidgzip | **-24%** | no parallel single-member decode |
| decomp arm64 software t1 | 120 | 152 | rapidgzip | **-21%** | sequential libdeflate, no parallelism |
| decomp arm64 logs tmax | 123 | 154 | rapidgzip | **-25%** | sequential libdeflate, no parallelism |
| comp arm64 software L1 tmax | 1347 | 1552 | pigz | **-13%** | ratio probe fallback not parallelized |
| decomp arm64 logs t1 | 124 | 137 | rapidgzip | **-10%** | sequential libdeflate |
| comp x86 software L9 tmax | 278 | 303 | pigz | **-8%** | likely measurement noise |
| comp x86 logs L1 t1 | 966 | 1030 | igzip | **-6%** | igzip SIMD advantage |
| decomp arm64 software tmax | 131 | 139 | rapidgzip | **-6%** | sequential libdeflate |
| decomp x86 silesia t1 | 250 | 258 | rapidgzip | **-3%** | sequential vs single-pass |
| comp x86 silesia L1 t1 | 327 | 335 | igzip | **-2%** | statistical noise |

---

## Phase 1: Fix Code Path Honesty (immediate, 1-2 days)

**Rule**: One production decompression path. One production compression path. Internal
benchmarks must call production code. No exceptions.

### 1a. Identify and lock the production inflate

Production inflate is `inflate_into_pub()` → `inflate_into_libdeflate()` (libdeflate C FFI).
This handles all per-block decoding in BGZF parallel, multi-member parallel, and single-member.

**Action**: Rename `bench_decompress` → `bench_production_inflate`. Update the benchmark
to call `decompress_gzip_to_writer()` (full gzip path) instead of raw deflate slices.
This makes the internal benchmark match exactly what the CLI does.

**Action**: Rename `bench_cf_silesia` → `bench_experimental_pure_rust`. Add a comment:
"This tests the experimental pure-Rust inflate which is NOT the production path.
See bench_production_inflate for production numbers."
Do not remove it — the experimental bench is useful for optimization work —
but stop reporting it as THE gzippy number.

**Action**: Update CLAUDE.md and .cursorrules to state clearly:
- Production inflate: libdeflate C FFI  
- Experimental inflate: inflate_consume_first (99% ARM, 90% x86)
- CLI decompression throughput on silesia: ~250 MB/s x86, ~130 MB/s arm64 (gzip format)
- BGZF parallel throughput: ~3500 MB/s (8 threads)
- These are different numbers for different workloads — both are true

### 1b. Fix the `num_threads` pass-through

The decompression CLI path ignores `args.processes` (`-p` flag) and always reads from
`available_parallelism()`. The compression path respects it.

**Action**: Thread `num_threads` from `args.processes` through `decompress_gzip_libdeflate`.
(This was implemented in `384f419` but reverted with the rapidgzip regression.
Re-add just the threading fix, without the rapidgzip decoder wiring.)

---

## Phase 2: Parallel Single-Member Decompression (week 1-2)

This is the biggest decompression gap: rapidgzip beats us by 24% on x86 silesia tmax
and 6-25% on arm64, entirely because it parallelizes single-member gzip files.

The previous attempt (`384f419`) caused -86% regression because `rapidgzip_decoder.rs`
spends multiple seconds failing on complex data (silesia) before giving up.

### Why the current rapidgzip_decoder is slow to fail

The speculative decoder tries all 8 bit offsets × N chunks. For silesia (dense,
unpredictable LZ77), almost no speculative start positions produce valid deflate blocks.
Each failed attempt decodes tens of MB of garbage data before detecting the failure.
Cost: ~2-5 seconds wasted before falling back to sequential.

### The correct parallel decompression strategy

Don't use speculative start positions. Instead, use **two-pass sequential boundary finding**:

```
Pass 1 (sequential):  Find all deflate block boundaries by decoding forward.
                      Take O(n/speed) time. For 160MB silesia: ~0.6s
                      Record [block_start_bit, block_end_bit, block_type] for each block.

Pass 2 (parallel):    Divide blocks into N groups. Each thread decodes its group.
                      Thread 0 starts with known window = last 32KB of block before its first block.
                      Threads 1..N start with unknown window (propagated from Thread 0 after Thread 0 finishes).

Result:               Overall time ≈ max(pass1_time, pass2_time/N + propagation)
```

For silesia (160MB compressed, 500MB output, 8 threads):
- Pass 1 (sequential): ~0.6s at 250 MB/s
- Pass 2 (parallel): ~0.5s at 500MB/8threads = ~0.06s each but propagation overhead
- Total: ~0.7s → ~700 MB/s effective throughput (2.8x sequential)

For single-member files under 4MB, fall through directly to sequential (overhead not worth it).

**Why this beats rapidgzip's approach**: rapidgzip uses speculative starting which succeeds
on compressible data (short blocks, known patterns) but fails badly on complex data.
Two-pass always succeeds — the only question is pass 1 speed, which is our sequential
libdeflate FFI speed (250 MB/s x86, 130 MB/s arm64).

### Implementation plan

```
src/two_pass_parallel.rs (new):
  - find_block_boundaries(data: &[u8]) -> Vec<BlockBoundary>
      Parse deflate stream sequentially, record block start positions.
      Uses libdeflate's block type detection (stored/fixed/dynamic headers).
      Must handle the gzip header/trailer correctly.
  
  - decompress_two_pass_parallel<W>(data: &[u8], writer: &mut W, n_threads: usize)
      Phase 1: find_block_boundaries → Vec<BlockBoundary>  (sequential)
      Phase 2: split boundaries into n_threads groups
               Thread 0 gets window = zeros (it's the first chunk, no back-refs cross it)
               Thread i gets window propagated from thread i-1
               Each thread calls inflate_into_libdeflate for its group of blocks
               Write results in order to output

decompression.rs routing change:
  - For single-member files > 4MB and num_threads > 1:
      Try two_pass_parallel first (never fails, just might be slow)
      Fall through to sequential only if two_pass returns error
```

**Test before wiring in**: The two-pass approach must be benchmarked internally (as
`bench_two_pass_parallel` calling `decompress_two_pass_parallel`) AND validated on all
three datasets at multiple thread counts BEFORE being wired into the CLI path.
Regression gate: must not be slower than sequential on any dataset at T1.

---

## Phase 3: Compression Parity at L1 (week 2-3)

### 3a. Merge ratio probe fix immediately

Branch `perf/fix-ratio-probe-fallback` shows these gains (vs baseline):
- x86 logs L1 tmax: +96% (753 → 1477 MB/s, beats igzip 1034)
- arm64 silesia L1 tmax: +31% (249 → 326 MB/s)
- arm64 software L1 tmax: +102% (668 → 1347 MB/s)

L6/L9 show minor regressions (likely measurement noise on shared CI runners, ±17%).
The net is clearly positive. **Merge this branch now.**

### 3b. x86 software L1 tmax: close the igzip gap

Current: gzippy 1694 vs igzip 2603 (-35%) at L1 tmax on x86.
After ratio probe fix: gzippy falls back to ISA-L when ratio < 10% for the full file.
igzip at 2603 MB/s is a single-threaded ISA-L invocation on the full 211MB file.

Root cause: our ISA-L fallback in compress_parallel processes blocks sequentially after
the ratio probe fires on block 0. So we compress block 0 twice (probe + real) and the
rest of the blocks through ISA-L sequentially. This adds ~25% overhead vs igzip's direct
full-file call.

**Fix**: When ratio probe fires in compress_parallel, use parallel ISA-L blocks (spawn
N threads, each compresses a slice of the input with ISA-L, write BGZF blocks in order).
This should achieve N × ISA-L_speed throughput for highly compressible data.

Expected result: ~4 × 2603 = ~10 GB/s theoretical → practical ~4 GB/s (memory-bound)

### 3c. arm64 software/logs L1 tmax: beat pigz

After ratio probe fix: arm64 software L1 tmax gzippy 1347 vs pigz 1552 (-13%).
pigz uses all threads for compression. Our ratio probe falls back to single-stream libdeflate.

**Same fix as 3b**: parallel libdeflate blocks when ratio < 10%, no fallback to single-stream.
On arm64, use libdeflate (no ISA-L). Expected: ~4 × 1238 MB/s → ~3 GB/s practical.

---

## Phase 4: Decompression Parity at L1 t1 (week 3-4)

### 4a. x86 software/logs L1 t1: match igzip

Current gaps (from CI):
- software L1 t1: gzippy 405, igzip 405 — **essentially tied**
- logs L1 t1: gzippy 966, igzip 1030 (-6%)

The logs gap is because highly repetitive data (dist=1 RLE) favors igzip's SIMD fill.
Our AVX2 fill_byte_avx2 (added in d258473) helps on x86. This should close after CI
re-runs with the AVX2 code active.

**Action**: Run CI on a build that exercises the AVX2 path (need an x86 CI run that
verifies AVX2 is being detected and used). Add an assertion test:
```rust
#[test]
fn test_avx2_detected_on_x86() {
    #[cfg(target_arch = "x86_64")]
    assert!(is_x86_feature_detected!("avx2"), "AVX2 not detected in CI");
}
```

### 4b. arm64 decompression: close rapidgzip gap

After Phase 2 (two-pass parallel), single-member decompression at tmax should scale.
The arm64 sequential gap at T1 (gzippy 120-129 vs rapidgzip 121-152) is because
rapidgzip uses a faster inflate on arm64.

Long-term: switch production inflate on ARM to `inflate_consume_first` once it matches
libdeflate speed on ARM (currently at 91% = 1402 vs 1543 MB/s). Required improvement: 9%.

**Known optimization with high probability**: multi-symbol table entries (ISA-L style).
`src/multi_symbol.rs` already implements this but is not integrated. This was identified
as HIGHEST priority in the hyperion branch's GAP_ANALYSIS.md. Expected: 10-20% gain on
literal-heavy data.

---

## Phase 5: Beat igzip L1 Single-Thread on x86 (weeks 4-6)

igzip at L1 t1 software: 2603 MB/s. gzippy after Phase 3b (parallel ISA-L): ~1694 MB/s
single-threaded equivalent. The single-thread gap is pure igzip AVX-512 advantage.

To beat igzip at L1 t1, we need:
1. **ISA-L AVX-512 single-thread compression** — this means calling ISA-L directly on x86
   with the correct vector size. Check if our ISA-L build enables AVX-512.
2. **Or**: Use ISA-L with multiple threads for L1 even at T1 (spawn background threads
   automatically). This violates the spirit of T1 benchmarks.
3. **Or**: Accept that igzip beats us at L1 t1 single-thread on x86 due to hardware SIMD
   and focus on winning everywhere else (we dominate at L6/L9 and tmax).

Reality check: igzip is Intel's hand-tuned assembly specifically optimized for fast L1
compression with AVX-512. Matching it without ISA-L means implementing the equivalent
AVX-512 Huffman encoder in Rust. This is weeks of work with uncertain outcome.

**Decision point**: After Phase 3, evaluate whether L1 T1 matters enough vs other gaps.
If we win 47/48 matchups, chasing the last AVX-512 gap may not be worth it vs improving
other areas. Document this gap explicitly.

---

## Phase 6: CI Coverage Completeness

### 6a. The BGZF decompression advantage is currently invisible

Current CI only benchmarks single-member gzip decompression. For BGZF (gzippy's own
format), we achieve 3500-4000 MB/s — 10-16x faster than single-member. This is our
biggest competitive advantage and it doesn't appear in CI.

PR #41 (`ci/multi-archive-decompress-benchmarks`) adds BGZF and multi-member benchmarks.
**Merge PR #41.**

After merge, the decompression matrix will show:
- BGZF: gzippy ~3500 MB/s vs rapidgzip/pigz ~0 MB/s (they can't decode BGZF)
- Multi-member: gzippy parallel vs competitors

### 6b. Ensure `-p` flag is tested in CI

Current CI passes `-p{threads}` to gzippy for compression but the decompression path
ignores `args.processes` (it always uses `available_parallelism()`). After Phase 1b fix,
add a CI assertion: `gzippy -d -p1 < silesia.gz` must give same output as `-pmax` but
may differ in speed.

---

## Execution Order

```
Week 1:
  [ ] 1a: Fix bench naming (bench_production_inflate, bench_experimental_pure_rust)
  [ ] 1b: Thread num_threads through decompression (without rapidgzip decoder)
  [ ] Merge perf/fix-ratio-probe-fallback
  [ ] Merge PR #41 (BGZF benchmarks)

Week 2:
  [ ] 2: Implement two_pass_parallel.rs (find_block_boundaries + parallel decode)
  [ ] Validate internally with bench_two_pass_parallel on all 3 datasets
  [ ] Wire into CLI with regression gate

Week 3:
  [ ] 3b/3c: Parallel ISA-L/libdeflate blocks for highly-compressible L1
  [ ] Update ratio probe in compress_parallel to use parallel blocks
  [ ] Test compression benchmarks

Week 4:
  [ ] 4b: Attempt multi-symbol table integration for pure Rust inflate on arm64
  [ ] Benchmark inflate_consume_first with multi-symbol vs without
  [ ] If >100% of libdeflate on ARM: switch production inflate to pure Rust on ARM

Week 5-6:
  [ ] Evaluate remaining gaps after all above
  [ ] Decide on L1 T1 x86 igzip gap
  [ ] Update all documentation
```

---

## Hard Rules for All Optimization Work

1. **Benchmark BEFORE touching anything.** Record baseline on silesia, software, logs.

2. **Internal bench must call production function.** If you add a new optimization to
   the hot path, the internal bench must call the same hot path the CLI uses.

3. **Test with the actual CLI** before merging:
   ```bash
   ./target/release/gzippy -d -p4 < benchmark_data/silesia-large.gz > /dev/null
   ```
   This is what CI measures. If the CLI is slower than the internal bench, find out why.

4. **No silently-unused implementations.** Every module in src/ must either:
   - Be in the production call graph
   - Be labeled `#[deprecated]` or have a `// EXPERIMENTAL: not in production path` comment

5. **Revert within 24h if regression.** If CI shows any matchup getting worse, revert
   and understand before trying again.

6. **Never claim parity you don't ship.** If `inflate_consume_first` is experimental,
   CLAUDE.md must say "experimental pure Rust: 1400 MB/s, production CLI: 250 MB/s."

---

## Why This Plan Works

The prior "hyperion" attempt (55 commits, Jan 2026) failed because it optimized code that
wasn't the production path. It achieved 91% of libdeflate in a non-production benchmark
but the CLI never changed.

This plan identifies every production code path, makes internal benchmarks mirror those
paths exactly, and closes gaps in the paths that actually ship.

The two-pass parallel decompressor is architecturally different from the speculative
approach — it always succeeds (never needs a fallback), is simple to reason about, and
scales predictably. The speculative approach (rapidgzip_decoder) should be preserved as
an alternative but must pass a correctness+performance gate before being used in production.
