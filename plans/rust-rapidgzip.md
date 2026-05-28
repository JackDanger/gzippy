# Closing the gap to rapidgzip — current state & action queue

**FRAMING (read this first; everything else follows):**
**gzippy-pure vs rapidgzip is 1.30× cycles, 0.98× instructions, IPC
1.08 vs 1.42.** Same algorithm, ~same instruction count. The gap is
**memory-subsystem stalls in the chunk pipeline + allocator + marker
bootstrap**, NOT the inflate inner loop. Do not propose Huffman
inner-loop optimizations until §4 falsifications are addressed; six
of them on this branch already prove rustc + LLVM is at parity with
hand-tuned scalar code there.

For instructions on how to work on this plan, see §3 (the 5-step
framework) and §4 (methodology guard rails). For what to actually
do next, see §2 (action queue).

---

## 1. Current gap

### 1.1 Measured baseline [PROVISIONAL n=3 — re-measure at n=20 before acting]

Neurotic i7-13700T, silesia-gzip9.gz at T=16, clean release builds
(`strip = true` verified). 10-trial interleaved A/B for wall;
n=3 perf-stat for counters (below §4.3's own ≥20 rule — the n=1
"2.17× cycles" prior cherry-pick proves this matters).
**Before committing to any lever based on these numbers, re-run
at n=20 with load<4.**

| Tool | Median wall MB/s | Cycles | Instructions | IPC |
|------|------------------|--------|--------------|-----|
| gzippy-pure (libdeflate-LUT default + T3 simplify) | 1092 | 4.28B | 4.62B | 1.08 |
| gzippy-isal (`isal-compression` feature) | ~1212 | 3.71B | 3.69B | 1.00 |
| rapidgzip 0.16.0 (`-P 16 -o /dev/null`) | ~1212 | 3.29B | 4.72B | 1.42 |

**Gap to rapidgzip: ~10% throughput.** Page-faults 2.15× more
(168.9K gzippy vs 78.5K rapidgzip — verified n=3, this is the
single most reproducible signal).

### 1.2 KEY EVIDENCE — gap is memory stalls

The instruction-count parity (0.98×) with rapidgzip combined with
the cycle gap (1.30×) and IPC drop (1.08 vs 1.42) proves the issue
is **memory subsystem**, not algorithm. gzippy executes
near-identical work — it just stalls 30% more cycles waiting on
memory.

PEBS-precise attribution
(`docs/perf/2026-05-28-framework-step2-pebs-attribution.md`):

- `__memmove_avx_unaligned_erms`: 45.42% memory accesses, 11.77%
  L3-miss + RAM-hit (true stall). The **write-side** of chunk-data
  extend.
- `decode_huffman_body_resumable` (bulk inflate): 1.10% LFB/MAB hit
  only. **Fully L1-cached. Not a stall site.**
- Marker bootstrap subtree under `LocalKey<T>::with`: 21.09% of CPU
  (rolled-up DWARF), but this is the bootstrap *work* + 128 KiB
  Block reset, not RefCell overhead.

### 1.3 What's structurally locked (do not touch)

- Chunk pipeline shape (parallel-SM with predecessor 32 KiB window
  at chunk boundaries).
- Lever G / Lever H / FetchingStrategy::splitIndex wiring (May 2026
  -29% wall, vendor parity on emit count).
- T3 simplification to vendor's 2-extra-literal shape (this branch,
  +1.9% confirmed 20-trial neurotic A/B).
- The libdeflate-LUT inner loop (default). ISA-L LUT inner is opt-in
  only; it falsified at -6% on production silesia.
- **Speculative prefetch in chunk_fetcher**. `GZIPPY_NO_PREFETCH=1`
  regressed bench-sm by +42%. Speculation is net-positive despite the
  bootstrap cost on chunks whose `contains_marker_bytes` never flips.
  Do not propose disabling it.

---

## 2. Action queue

Ranked by expected wall impact, each with hypothesis + how to
falsify + prior attempts.

### 2.1 Allocator: pthread-init the rpmalloc binding [PRIORITY]

**Hypothesis**: Rust `rpmalloc 0.2.2` crate's `RpMalloc` global
allocator wrapper does NOT call `rpmalloc_thread_initialize()` per
thread. Workers fall back to a global-mutex'd heap. Rapidgzip's
bundled rpmalloc auto-installs pthread hooks (CHANGELOG line 231).

**Evidence**: Step 5 production A/B (this branch) falsified Lever 4.1
at +167% wall on `-o real_file`, +41% page-faults. Source-dive of
`~/.cargo/registry/.../rpmalloc-0.2.2/src/lib.rs:145-155` confirmed
no thread-init.

**How to falsify**: Vendor rapidgzip's bundled rpmalloc into
`vendor/rpmalloc-vendored/` and write a `#[global_allocator]`
wrapper that installs pthread keys with destructors calling
`rpmalloc_thread_finalize`. Re-run `scripts/alloc_ab_harness.sh`
at n=20 with load<4. Pass gate: ≥5% p50 wall reduction AND ≥10%
page-fault reduction on silesia/-o real_file.

**Expected signal**: page-faults drop from 248K → ~80K (matching
rapidgzip). Wall on file output recovers from +167% regression to
neutral or better.

**Prior attempts**: Lever 4.1 falsified
(`docs/perf/2026-05-28-framework-step5-lever4-1-falsified.md`),
Z-allocator prewarm falsified (-15%, single-thread serialized
faults), L1 MADV_HUGEPAGE -38% (khugepaged contention).

**If multi-day vendor work feels heavy, run §2.2 FIRST** as a
1-afternoon falsification — but be aware §2.2 can only refute the
allocator class, not positively confirm rpmalloc specifically.

### 2.2 Negative control: mimalloc + jemalloc as allocators

**Hypothesis**: If allocator-of-the-month is the lever, any
production-quality binding (mimalloc-rs, jemallocator) should
deliver. If both ALSO regress, rpmalloc isn't specially magical
and the win is structural (chunk shape, marker phase).

**How to falsify**: Add `--features global-mimalloc` and
`--features global-jemalloc`. Re-run harness. Expected runtime: 1
afternoon. Cheaper than vendoring rpmalloc properly.

**Why this is queued ahead of any chunk-shape change**: it tests
the hypothesis class without committing to vendoring work.

### 2.3 Chunk-shape: 128 KiB segmented ChunkData

**Hypothesis**: rapidgzip's `ALLOCATION_CHUNK_SIZE = 128 KiB` +
many small Boxes vs gzippy's single 12 MiB Vec causes the page-fault
delta. Their own bench (in source comment): 21.2 GB/s at 1 MiB vs
8.4 GB/s at 4 MiB allocation chunk size = 2.5× difference.

**Evidence**: Step 4 microbench showed glibc+manyBoxes+readOFF = -84%
wall vs baseline. On read-sweep (with current allocator), manyBoxes
regressed +21% wall (advisor caught: prefetcher loses contiguity at
chunk boundaries).

**How to falsify**: Add a feature-gated variant of `ChunkData::data`
that uses `Vec<Box<[u8; 128*1024]>>`. Validate silesia byte-perfect.
Run harness at n=20, load<4. Must include `-o real_file` (where the
arm64 read-regression surfaced locally).

**Risk**: The chunk-shape and allocator levers are coupled. Per
Step 4 microbench, rpmalloc+manyBoxes was -64% but glibc+manyBoxes
was +21% worse. Don't stack with 2.1 until 2.1 is independently
validated.

**Prior attempts**: S2 bulk-window-copy ported the chunk-shape
philosophy to `copy_match_windowed`'s slow path; at parity (slow
path too rare to matter).

### 2.4 Marker bootstrap: shrink u16 ring → u8 + parallel bitmap

**Hypothesis**: marker bootstrap u16 ring writes 2 bytes per output
byte. Replace with u8 ring + 1-bit selector bitmap. rapidgzip's own
ChunkData.hpp comment block (lines 247-300) explicitly plans this
("would reduce allocation overhead by ~80%") but hasn't shipped it.

**Evidence**: symbolized perf shows `emit_backref_ring` at 2.84%,
`Vec::extend_from_slice` from `drain_to_output` at 1.61%,
`clean_unmarked_data` at 1.82% — combined ~6.3% of CPU in marker
data-movement.

**How to falsify**: Multi-day surgery touching `deflate_block.rs`,
`chunk_data.rs`. Defer until 2.1-2.3 resolved.

**Expected ceiling**: ~5% e2e per advisor (marker-phase share of
CPU). Not the dominant lever.

### 2.5 Inflate inner loop (LAST resort, only after 2.1-2.4)

The bulk inflate is **1.10% LFB hit** in PEBS — fully L1-cached.
**Inflate inner-loop work cannot close the gap on its own.**

If 2.1-2.4 all fail to close the remaining 10%, then and only then
consider: BMI2 PEXT, AVX2 vpshufb literal output, speculative-
parallel LUT lookups. These are the multi-week structural
deviations CLAUDE.md 2026-05-27 explicitly authorized — but they
are LAST in the queue, not first.

---

## 3. The 5-step instrumented framework

Used at every step in 2.1-2.5 to avoid the stab-in-dark pattern
that produced 13 falsifications and 1 confirmed win in 2026.

1. **Source-dive target tool first.** Read rapidgzip/libdeflate
   source for the exact pattern. Don't guess. See
   `docs/perf/2026-05-28-framework-step1-allocator-spec.md`.

2. **PEBS attribution.** `perf mem record --call-graph=dwarf` then
   `perf mem report --sort=mem,sym`. Find which load addresses
   stall, not which functions look hot. See
   `docs/perf/2026-05-28-framework-step2-pebs-attribution.md`.

3. **Timeline events** (`GZIPPY_TIMELINE=/tmp/out.json`) — deferred;
   wire allocator events here only when Steps 1+2 don't yield a clear
   flat-profile attribution.

4. **Microbench mirroring production shape.** `benches/alloc_pattern.rs`
   is the template: 16 workers × actual write/read sizes × dual sinks.
   30s iteration. **Not sufficient for production confidence** — the
   microbench predicted Lever 4.1 = -54%, production was +167%.

5. **Production A/B harness.** `scripts/alloc_ab_harness.sh` runs
   n=20 with the mandatory rollup fields (§4.4). Pass gate: ≥5%
   p50 wall reduction AND ≥10% page-fault reduction on the harder
   of the two sinks.

The pattern microbench-says-yes-production-says-no caught the
broken rpmalloc binding in <30 minutes. Use it.

---

## 4. Methodology guard rails

### 4.1 Clean-build verification (every bench)

```
grep -E "^strip" Cargo.toml | head -3   # must show: strip = true
```

The 2026-05-28 session lost 6 hours to a contaminated build where
`debug = "line-tables-only"` had been left in the `release` profile.
The "41% gap" headline that drove most of that session's work was
the contamination, not real.

### 4.2 System load gate

Bench at **load avg < 4**, not 47. neurotic is shared; check
`uptime` before benching. Under high load, wall times swing 10×.
Counter ratios (page-faults, branch-misses) are load-independent
and remain valid; wall is not.

### 4.3 Trial count

**n ≥ 20 with per-iter MEDIAN, not mean.** n=3 already produced a
cherry-pick in this session (advisor proved it: the n=1 "2.17×
cycles" headline was wrong — real ratio was 1.30×). n=10 mean can
mask warmup effects (rpmalloc TLS cache warms in iter 2+).

**Trial 1 (cold) reported separately** from trials 2-N. Production
gzippy spawns FRESH workers per `decompress_parallel` call; the
trial-1 number is closer to user experience than the steady-state.

### 4.4 Mandatory rollup fields for every A/B

Reject any rollup missing any of: wall_ms (per-iter median),
cycles/instructions/IPC (no multiplexing, enable rate ≥95%),
page-faults (minor+major separately), dTLB+L1+LLC misses,
context-switches, task-clock, peak RSS, smaps AnonHugePages +
Private_Dirty, mmap+munmap+brk counts (`strace -c`), correctness
hash (silesia SHA256 byte-identical), build hash (Cargo.lock +
RUSTFLAGS + strip=true verified).

### 4.5 Adversarial advisor on every claimed win

Per user process rule (2026-05-27): every judgment call AND every
claimed task completion goes to an Opus advisor (`Agent` tool,
`subagent_type=claude`) BEFORE finalizing. Frame the request as
"try to disprove this" — adversarial is the high-signal mode.

This session: 3 advisor consultations caught the n=1 cherry-pick,
3 missed sub-levers in the allocator design, and the read-pattern
over-claim in the microbench result. Every catch was empirically
confirmed at the next measurement step.

**Update from May-26 plan:** advisors are now used for both
synthesis (lever ranking, design review) AND adversarial review.
The "sanity-check only" rule from the May-26 plan was wrong.

---

## 5. System map (orientation only — see code for details)

### 5.1 Pipeline

```
gzippy -d <file.gz>
 → src/decompress/mod.rs: route by header
 → src/decompress/parallel/single_member.rs: parallel-SM dispatch
 → src/decompress/parallel/sm_driver.rs: chunk_fetcher::drive
 → workers (T=16):
    bootstrap_with_deflate_block_inner (marker phase, ~21% CPU)
    → decode_chunk_isal_impl (calls IsalInflateWrapper → ResumableInflate2)
    → ChunkData::data via chunk_buffer_pool::take_u8(cap)
 → consumer thread:
    reorder buffer → CRC → writer.write_all
```

### 5.2 Files that own the gap

- `src/decompress/parallel/chunk_data.rs` — the single-Vec
  allocation pattern (Lever 2.3 target).
- `src/decompress/parallel/chunk_buffer_pool.rs` — pool with
  thread-local indices (rpmalloc wiring point for Lever 2.1).
- `src/decompress/parallel/gzip_chunk.rs:1431+` — marker bootstrap
  (Lever 2.4 surgery zone).
- `src/decompress/parallel/deflate_block.rs:1626` — `emit_backref_ring`
  (the 2.84% memmove site).
- `src/decompress/inflate/resumable.rs:1003+` — `decode_huffman_body_resumable`
  inner loop. **Do NOT optimize this** until §2.1-2.4 resolved.

### 5.3 Routing facts that surprise people

- `gzippy -d` at T=16 on silesia routes through parallel-SM, NOT
  the single-member fallback. The classifier gates this on
  `data.len() > MIN_PARALLEL_COMPRESSED` AND `num_threads > 1`.
- `IsalInflateWrapper` in `--features pure-rust-inflate` is a
  wrapper over `unified::Inflate` over `ResumableInflate2`. The
  ISA-L name is historical; the pure-rust path runs through it.
- Marker bootstrap is SAME code in `pure-rust-inflate` and
  `isal-compression` builds. Allocator/page-fault gap is therefore
  NOT what differentiates the two — the inflate bulk does.

---

## 6. Falsification index (don't re-walk)

17 entries. Grep this before proposing any lever. Each row links to
the falsification doc with the hypothesized cause + future-retry
guard.

| # | Date | Lever | Result | Cause | Doc |
|---|------|-------|--------|-------|-----|
| 1 | May-26 | B2 PRELOAD naive | -10% | Wasted lookups on yield paths | `archive/rust-rapidgzip-2026-05-26.md` |
| 2 | May-26 | 2-unroll FASTLOOP | -10% | I-cache pressure | same |
| 3 | May-26 | FASTLOOP PRELOAD with cached entry | no change | OoO already overlaps | same |
| 4 | May-26 | Single-literal fast path | no change | Compiler already branch-predicts | same |
| 5 | May-26 | Writeback-skip around copy_match | -9pp | Borrow-checker spills | same |
| 6 | May-26 | LTO=fat + cgu=1 + native | no change | Flags alone insufficient | same |
| 7 | May-26 | target-cpu=native alone | no change | same | same |
| 8 | May-28 | Route C v3.7/v3.9 dynasm asm | at parity | rustc + target-cpu=native already optimal scalar | `docs/perf/2026-05-28-v3.8-hybrid-silesia-bench.md` |
| 9 | May-28 | S1 u32 packed literal store | +0.4% noise | LLVM + x86 store-buffer coalesces | `docs/perf/2026-05-28-s1-packed-lit-store-falsified.md` |
| 10 | May-28 | S2 bulk window-copy slow path | at parity | Slow path fires on ~3% of bytes | `docs/perf/2026-05-28-s2-bulk-window-copy-falsified.md` |
| 11 | May-28 | L1 MADV_HUGEPAGE on chunk buffers | -38% | khugepaged contention + madvise latency | `docs/perf/2026-05-28-framework-step1-allocator-spec.md` |
| 12 | May-28 | `GZIPPY_RESUMABLE_ISAL_INNER=1` as prod default | -6% | ISA-L LUT multi-pack emit has more overhead on production silesia than libdeflate-LUT | `project_isal_lut_port_landed.md` |
| 13 | May-28 | C-fastloop iter-count batch (`for _ in 0..safe_iters`) | INFINITE LOOP | `safe_iters=0` near chunk tail re-enters with same bounds; hung cargo test 4hr | git commit 762ac0a → reverted 6a7b72e |
| 14 | May-28 | Lever 4.1 rpmalloc global allocator | +167% wall on `-o file`, +41% page-faults | Rust rpmalloc 0.2.2 crate doesn't call `rpmalloc_thread_initialize` per thread | `docs/perf/2026-05-28-framework-step5-lever4-1-falsified.md` |
| 15 | May-26 | Z-allocator pool prewarm | -15% | Serialized 20K faults that were previously parallel across 16 workers | `feedback_z_allocator_prewarm_falsified.md` |
| 16 | May-28 | BZHI/BMI2 explicit wrapper around `(src & ((1<<n)-1))` | regress | rustc with `-C target-cpu=native` already lowers the inline pattern to BZHI on Raptor Lake; explicit wrapper's indirection regressed | `crates/gzippy-inflate/src/route_c_dynamic.rs:93-99` |
| 17 | May-28 | Unchecked raw-pointer multi-literal writes + upfront 4-byte headroom check | -9% (395→360 MB/s) | rustc was already eliding the bounds check via range analysis; the explicit unsafe shape disrupted register allocation | `crates/gzippy-inflate/src/route_c_dynamic.rs:611-619` |

### Confirmed wins (alongside the falsifications)

| # | Date | Lever | Result | Mechanism |
|---|------|-------|--------|-----------|
| W1 | May-28 | C+T3 simplify (4-literal → 2-extra-literal cap) | +1.9% wall, 20-trial neurotic A/B | Vendor `decompress_template.h:370` explicitly measured 3+ extras as a pessimization; removes i-cache + branch-pred pressure |

---

## 7. Done-when criteria

ALL of:

1. **Perf**: pure-rust within 1pp of rapidgzip on neurotic
   silesia-gzip9.gz at T=16, n=20 paired-bench median, load<4,
   strip=true verified. Confirmed on AT LEAST silesia + a
   low-redundancy file (where marker bootstrap dominates) + a
   high-redundancy file (4 GiB-base64-like, where inflate inner
   loop dominates).
2. **No regression on `-o real_file`** (the 2026-05-28 Lever 4.1
   regression mode). Test both `-c >/dev/null` AND `-o real_file`
   sinks in every A/B.
3. **Correctness**: silesia SHA256 byte-identical; 3-oracle
   differential fuzz (flate2 + libdeflate-sys + zlib-ng-sys) ≥ 72h
   per release with zero disagreements.
4. **Adversarial advisor sign-off** on the final neurotic
   measurement.

CLAUDE.md prime directive ("fastest gzip ever") is a stretch goal:
beating rapidgzip on representative workloads. This plan closes the
gap to within 1pp. Beating rapidgzip likely requires the marker
u8-ring + BMI2 PEXT path (§2.4 + §2.5).

---

*Plan last updated: 2026-05-28. Pre-update version preserved at
`plans/archive/rust-rapidgzip-2026-05-26.md`. If you are picking
this up cold: read §1 first, then §2.1 for the highest-priority
lever, then §3+§4 for how to work without re-falsifying.*
