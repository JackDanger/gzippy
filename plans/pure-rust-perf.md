# Pure-Rust perf — measurement log + active work-item status

This document tracks the live measurements that prove (or refute)
progress on the **one deliverable** stated in `plans/rust-rapidgzip.md`.

It is not a phase plan. There is exactly one deliverable; this file
is its dashboard.

## Apples-to-apples comparison on equal hardware (2026-05-24+, neurotic)

The full-population number that should drive decisions:

| Tool | T=1 (silesia, P-cores only) | T=9 (P-cores) | T=16/T=1 ratio | T=9 output rate |
| --- | ---:| ---:| ---:| ---:|
| gzippy (pure-rust-inflate) | 258 ms (libdeflate path) | 506 ms | 2.78 (par SLOWER) | 282 MB/s |
| gzippy (isal-compression)  | 271 ms                   | 537 ms | similar          | similar    |
| **rapidgzip 0.16.0**       | **273 ms**               | **147 ms** | **0.42 (par FASTER)** | **1100 MB/s** |

- **gzippy T=1 (libdeflate) is FASTER than rapidgzip T=1.** Sequential isn't the gap.
- **gzippy T=9 is 3.4× SLOWER than rapidgzip T=9** on equal hardware (P-core pinned).
- **Gate 2's <0.5 target IS achievable** — rapidgzip hits 0.42 on this fixture.
- Switching gzippy from pure-rust-inflate to isal-compression doesn't close the gap
  (537 vs 506 ms). **Inflate engine choice is NOT the lever.**
- Hybrid CPU confound: gzippy T=16 mixed P+E cores = 753 ms; gzippy T=9 P-only =
  506 ms. ~25% of measured gap was thread placement.

## Cross-tool instrumentation infrastructure (Phase 1/2/3, in tree)

The plan and its decisions are now driven by data emitted by **matching
Chrome-trace JSON emitters** on both sides:

- **gzippy**: `src/decompress/parallel/trace_v2.rs` (`SpanGuard` RAII +
  `emit_alloc` + `LockSpan` macros). pid=1. Activated by
  `GZIPPY_TIMELINE=/tmp/gz.tl.json`.
- **rapidgzip**: `scripts/rapidgzip_trace_patch/TraceV2.hpp` +
  `patch_vendor*.sh`. pid=2. Same env var. Built from
  `vendor/rapidgzip/librapidarchive/` via included patch scripts.
- **Analyzer**: `scripts/timeline_analyze.py` reads either file or
  both (diff mode). Top spans, per-thread busy/wait, critical-path
  contributors.

Build/run recipes:

```bash
# gzippy
ssh -J neurotic root@REDACTED_IP 'cd ~/gzippy &&
  cargo build --release --features pure-rust-inflate'
ssh -J neurotic root@REDACTED_IP '
  rm -f /tmp/gz.tl.json &&
  taskset -c 1,3,4,5,6,7,10,13,15 \
    env GZIPPY_TIMELINE=/tmp/gz.tl.json \
    /root/gzippy/target/release/gzippy -d -c -p 9 \
    /root/benchmark_data/silesia-gzip.tar.gz > /dev/null'

# rapidgzip (one-time vendor build)
ssh -J neurotic root@REDACTED_IP '
  cp scripts/rapidgzip_trace_patch/TraceV2.hpp \
     /root/gzippy/vendor/rapidgzip/librapidarchive/src/core/ &&
  bash scripts/rapidgzip_trace_patch/patch_vendor.sh &&
  bash scripts/rapidgzip_trace_patch/patch_vendor_phase3b.sh &&
  cd /root/gzippy/vendor/rapidgzip/librapidarchive/build &&
  cmake --build . --target rapidgzip -j 9'
ssh -J neurotic root@REDACTED_IP '
  rm -f /tmp/rg.tl.json &&
  taskset -c 1,3,4,5,6,7,10,13,15 \
    env GZIPPY_TIMELINE=/tmp/rg.tl.json \
    /root/gzippy/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip -d -c -P 9 \
    /root/benchmark_data/silesia-gzip.tar.gz > /dev/null'

# Diff
ssh -J neurotic root@REDACTED_IP \
  'python3 /tmp/timeline_analyze.py /tmp/gz.tl.json /tmp/rg.tl.json'
```

Span vocabulary (semantic, matched across both tools):

| Span | Meaning | Both tools? |
| --- | --- | --- |
| `worker.decode_chunk` | One chunk's full decode | both |
| `worker.bootstrap` | gzippy phase-1 marker bootstrap | gzippy only (no vendor counterpart — unified inflate) |
| `worker.block_header` / `worker.block_body` | per-block decode in bootstrap | gzippy only |
| `consumer.iter` | One outer-loop iteration | both |
| `consumer.drain` | gzippy pending-write drain | gzippy only |
| `post_process.task` / `post_process.apply_window` | window apply / narrow | both |
| `wait.future_recv` | mpsc::Receiver::recv | gzippy only (vendor uses BlockFetcher::get path) |
| `wait.block_fetcher_get` | block_fetcher.get_with_prefetch wait | both |
| `pool.submit` | thread pool task enqueue | both |
| `pool.pick` | worker dequeue + condvar wait | both |
| `pool.run_task` | task body execution | both |
| `block_finder.scan` | BlockFinder scan thread | both (vendor wired but path may differ) |
| `drive` | entire parallel-SM run | gzippy only |

## VERIFIED FINDINGS (anti-mistake rule: cite the span data)

### 1. Cache-miss culprit on consumer thread

`wait.block_fetcher_get` on the consumer thread:

- gzippy: **4 cache misses** at chunk_id (0, 201, 1855, 2545) for **210 ms total wait** = 43% of wall
- rapidgzip: 26 small waits (~3 ms avg) for 84 ms total, no big misses

These are SPECIFIC, NAMED chunks. Chunk 0 = cold start (~40 ms unavoidable);
chunks 201/1855/2545 = consumer outran the prefetch window mid-stream
(~50-75 ms each).

### 2. Worker load imbalance (more diffuse than the cache-miss data)

Per-thread busy times across 9 workers:

- rapidgzip: 70-92 ms range, σ ≈ 7 ms (TIGHT)
- gzippy: 124-555 ms range, σ ≈ 130 ms (WIDE)

Per Phase 3 dispatch data, this imbalance is largely a **consequence**
of the cache misses: while consumer is blocked, workers idle in
`pool.pick`. Fix the misses → idle drops → imbalance tightens.

### 3. Pool efficiency gap

(useful decode work / (cores × wall)):

- gzippy: 2670 ms / (9 × 666 ms) = **45%**
- rapidgzip: 879 ms / (9 × 162 ms) = **60%**
- Gap: 15 percentage points.

Same root cause as (1) — consumer blocking propagates to pool idle.

### 4. Per-chunk decode rate 1.9× slower

`worker.decode_chunk` averages:

- gzippy: 47 ms / chunk (33 chunks)
- rapidgzip: 25 ms / chunk (26 chunks)

This is real but SEPARATE from the cache-miss + load-balance story.
Ceiling test: even closing all cache misses leaves wall at ~290 ms
vs rapidgzip's 147 ms — per-chunk rate must come down too.

### 5. Bootstrap body rate in production: 98 MB/s (vs bench 242 MB/s)

`worker.block_body` sum (gzippy): 1.03 s of decode for 101 MB output =
98 MB/s per worker. Bench (`bootstrap_marker_overhead.rs`) shows the
same code path at 242 MB/s on a fresh deflate stream — the 2.5×
production-vs-bench gap is from surrounding pipeline overhead
(allocator, narrow, apply_window).

## VERIFIED REFUTATIONS (do NOT pursue)

Each killed by direct measurement:

- **40% page-fault wall premise** — actual is 17% on real silesia
  (matches rapidgzip's 17%). Phase A closed (commit `dd5c64f`).
- **Body-failure speculation accuracy** as the dominant cost — 39 body
  failures total cost 608 bytes wasted = ~4 ms work. Not a lever.
- **CountAllocatedLeaves precode pre-pass** would buy 25% wall — V1
  showed only 29% (17/58) of failures are precode-catchable. Ceiling
  is ~4% wall, not 25%.
- **Cache-key mismatch hypothesis (H2)** — gzippy mirrors vendor's
  two-key dance at `chunk_fetcher.rs:663-666`. Architecture identical.
- **Prefetch-fires-before-confirm race (H1)** — `GZIPPY_NO_PREFETCH=1`
  makes wall 1.55× slower (727 → 1131 ms). Prefetch is helping.
- **Inflate engine (ISA-L vs pure-Rust)** — `--features isal-compression`
  build measures 537 ms vs pure-rust 506 ms. ISA-L doesn't help.
- **Global allocator swap** — `#[global_allocator] = rpmalloc::RpMalloc`
  regressed 12% (492 → 552 ms). Matches mimalloc / jemalloc history.
  Vendor explicitly avoids global rpmalloc per `ChunkData.hpp:24`.
- **Pool the `bootstrap_with_deflate_block` Vec** — advisor-suggested
  fix; advisor implemented and measured. Delivered 30 ms, not the
  predicted 150-200 ms. Page-fault count unchanged.
- **`prefetch_capacity` 2x → 4x** — regressed wall 486 → 510 ms.
  Cache misses dropped 4 → 3 but overhead elsewhere ate it.
- **Marker bookkeeping in bootstrap** — bench showed markers ON/OFF =
  0.99× ratio. Not the cost.
- **Larger LitLenTable::TABLE_BITS (13)** — regressed (32 KB table out of
  L1 territory). Sweet spot is **12** (kept). DistTable sweet spot **9** (kept).
- **SIMD multi-literal fastloop** (vector_huffman ported into
  `decode_huffman_body_resumable`) — regressed bench 334 → 284 MB/s,
  silesia 2.79 → 3.06. Reverted at commit `d08732f`. Silesia data has
  too few literal clusters for the lookahead to pay back its overhead.

## Confirmed wins kept

| Commit | Change | Win |
| --- | --- | --- |
| `41076a0` | Worker-side scalar `narrow_u16_to_u8` | wall 998 → 881 ms (12%) |
| `ee05316` | LitLenTable::TABLE_BITS 11 → 12 | wall 895 → 863 ms (4%) |
| `9a34f32` | DistTable::TABLE_BITS 8 → 9 | wall 863 → 809 ms (6%) |
| (cumulative) | All session wins | wall 998 → 506 ms = 49% (silesia gate 2 ratio 2.76 → 2.78 post-P-core baseline correction) |

## Live baseline (2026-05-24, neurotic, 16-core x86_64)

All numbers via `--features pure-rust-inflate` unless noted.

### Gate 1 — inner-inflate parity

`benches/inflate_isal_vs_pure_rust.rs`, best of 3 runs:

| Decoder                                       | Throughput |
| --------------------------------------------- | ---------- |
| Patched ISA-L via `IsalInflateWrapper`        | 799 MB/s   |
| Pure-Rust `ResumableInflate2`                 | 334 MB/s   |
| **Ratio (ISA-L / pure-Rust)**                 | **2.39×**  |

Deliverable target: ≤ 1.05×. Currently failing by 1.34×.

⚠️ **This bench measures `ResumableInflate2` (parallel-SM phase 2 /
work item 1(a)).** It does NOT measure `deflate_block::Block::read_
internal_compressed_canonical_specialized` (parallel-SM phase 1 /
work item 1(b)). The 334 MB/s number was treated as the
single-thread inflate rate for the whole parallel-SM pipeline. It
isn't — see "Phase B step-zero finding" below.

### Gate 2 — E2E silesia parallel beats sequential

`test_single_member_parallel_silesia` (real silesia.tar.gz, 67 MiB
→ 211 MiB output):

| Path                                          | Wall time | Output rate |
| --------------------------------------------- | --------- | ----------- |
| T=1 sequential (libdeflate one-shot)          | 361 ms    | 586 MB/s    |
| T=16 parallel SM (pure-Rust resumable inflate) | 998 ms    | 212 MB/s    |
| **Ratio (par/seq)**                           | **2.76× SLOWER parallel** |

Deliverable target: < 0.5. Currently failing by 5.5×.

### Gate 3 — `make ship` vs rapidgzip

Not yet measured for pure-rust-inflate (would be performative — Gate 2
is the dominant signal, and `make ship` requires Gate 2 to pass first).

### Gate 4 — tree cleanliness

Not yet attempted. `vendor/isa-l`, `vendor/isal-rs`,
`packaging/isal-patches/` present. `src/backends/isal_decompress.rs`
and `isal_compress.rs` present.

## Phase A — page-faults (CLOSED 2026-05-24)

`perf record -F 999 --call-graph dwarf` on neurotic against real
silesia:

| Component                                    | % of weighted cycles |
| -------------------------------------------- | -------------------- |
| `asm_exc_page_fault` ∪ `clear_page_erms` (all stacks) | **17.2%**     |
| `clear_page_erms` alone                      | 0.7%                 |

This matches rapidgzip's documented 17%. The arena-allocator
(`Cargo.toml:39,50-51` → `Vec<_, RpmallocAlloc>` for `ChunkData::data`)
+ per-worker LIFO pool (`chunk_buffer_pool.rs:108-204`) closed the gap.

The 40% number cited in prior plan revisions was measured **before**
arena-allocator landed on main. Stale prior, caught by re-measurement.
**Do not re-do.**

If a future regression brings page-faults back, the untried levers
(in decreasing safety) are: worker-local pre-touch of pool buffers
after `bind_worker_pool_index` → `MADV_HUGEPAGE` on buffers ≥ 2 MiB →
`#[global_allocator] = rpmalloc::RpMalloc`. Each carries its own
unproven-tier risk; consumer-thread prewarm regressed −50%
previously (`chunk_buffer_pool.rs:84-88`).

## Phase B step-zero finding — bootstrap dominates

Instrumented run on real silesia (commit `6788651` added
`decode_span` + `post_process_span` events; commit `77eda45`
recorded the numbers below):

| span                            | sum (ms) | p50    | p95    | max    |
| ------------------------------- | --------:| ------:| ------:| ------:|
| **bootstrap**                   | **1060** | 14.2ms | 56.3ms | 62.4ms |
| inflate (phase 2)               | 499      | 0us    | 48.6ms | 52.2ms |
| apply_window (replace_markers AVX2) | 17   | 459us  | 1.8ms  | 2.1ms  |
| populate_subchunk_windows       | 52       | 1.3ms  | 3.1ms  | 20.7ms |

Phase distribution among 49 worker decodes:
- `bootstrap_only`: **29** (chunk fully decoded in phase 1)
- `bootstrap+inflate`: 19
- `bootstrap_terminal`: 1 (BFINAL fired)

Speculation outcomes: 62.9% accepted, 28.6% missing, 2.9%
mismatched.

**Interpretation**:

Bootstrap is the bottleneck, not inflate. Bootstrap runs
`deflate_block::Block::read_internal_compressed_canonical_specialized`
(work item 1(b)). The 334 MB/s pure-Rust inflate bench measures
`ResumableInflate2` (work item 1(a)) — a different code path. Phase
1 has never been benched in isolation.

Vendor's `GzipChunk.hpp::decodeChunkWithRapidgzip` ALSO has a
bootstrap phase using `deflate::Block::read()`. The slowdown is not
architectural — gzippy's per-symbol scalar decoder is just slower
than vendor's SIMD-batched one. Work item #1 is the vendor-faithful
fix; the architectural alternatives (parent-thread BlockFinder
pre-pass, unified phase 1 / phase 2 decoder via marker emission) are
not vendor patterns and are not on the critical path.

## Work-item status

Mirrors `plans/rust-rapidgzip.md` six items. Update this table as
items land.

| # | Item | Status | Bench moved? |
|---|---|---|---|
| 1 | SIMD-ify Huffman hot loops (a)+(b) | ACTIVE — confirmed as the lever; one SIMD attempt reverted | gate 1 currently 2.43× (323/785 MB/s); gate 2 currently 2.72× |
| 2 | Subtable bit-accounting (failure #2) | not started | `with_until_bits_resume_non_byte_aligned_with_dict` red |
| 3 | Proactive `RawBlockFinderCoordinator` | not started | `speculative_missing` still 28.6% |
| 4 | Retire phase-1 bootstrap | blocked on #1, #2 | bootstrap span still dominant |
| 5 | CRC32 interleave + worker-side window publish | partial (worker-side narrow landed 41076a0) | wall 998ms → 881ms (12% improvement) |
| 6 | `Cargo.toml` + ISA-L deletion | blocked on #1, #4 | gate 4 not attempted |

## Pipeline serialization is vendor-faithful — the lever is per-chunk decode time

Diagnostic trace from commit `79ff796` (`consumer_loop_summary` event)
on real silesia (727ms wall):

| consumer-thread span                  | sum    | % of wall |
| ------------------------------------- | ------:| ---------:|
| `iter_sum` (outer-loop iterations)    | 706 ms | 97%       |
| `fetcher_get_us` (`block_fetcher.get_with_prefetch`) | 500 ms | 70% |
| `prefetch_us` (`process_ready_prefetches`) | 0 ms |           |
| `finder_us` (`block_finder.get`)      | 0 ms   |           |
| `total_us` (consumer drain — narrow/CRC/write/publish) | 38 ms | 5% |
| `recv_us` (waiting for post_process)  | 0 ms   |           |
| Worker decode work sum                | 3000 ms (T=16 ideal: **187 ms**) |
| Worker queue wait sum                 | 12 ms  |           |

**Interpretation**: the consumer outer loop spends 70% of wall time
inside `block_fetcher.get_with_prefetch`, waiting for the NEXT chunk
in encoded order to finish decoding. This is structurally serial —
the chunk's stitched window must be applied in order. Vendor
`rapidgzip::GzipChunkFetcher::processNextChunk` has the same shape;
this is the rapidgzip-faithful design.

Because workers ARE saturated (3000 ms work / 16 cores = 187 ms
ideal wall) but the consumer can only ADVANCE one chunk at a time,
the only way to reduce wall is to reduce the per-chunk wait the
consumer spends inside `block_fetcher.get_with_prefetch`. That wait
equals **per-chunk decode time minus parallelism overlap**.

At ~14ms wait per chunk × 35 iterations = 500ms `fetcher_get_us`.
Reducing per-chunk decode time from 40ms → 13ms (3× speedup via
SIMD inner inflate) drops the wait to ~5ms × 35 = 175ms. Total
wall would land near 250-300ms (2.5-3× speedup). This is the
**only** lever that moves wall.

What does NOT move wall:
- Increasing prefetch depth (workers already saturated).
- Larger chunks (reduces iter count but proportionally raises per-chunk wait).
- Making the consumer's outer loop "parallel" (vendor doesn't, and
  the in-order stitching is a correctness constraint).

## SIMD attempt log

| commit | attempt | result |
| --- | --- | --- |
| `48c8cf4` | Worker-side narrow + AVX2 `_mm256_packus_epi16` | regressed (silesia 2.76→3.00) — AVX2 narrow gated behind env in `41076a0` |
| `41076a0` | Worker-side **scalar** narrow only | **kept** — silesia 998→881 ms (12% wall improvement) |
| `ca52389` | Multi-literal FASTLOOP in `decode_huffman_body_resumable` mirroring `decode_huffman_cf_vector` | regressed bench 334→284 MB/s, silesia 2.79→3.06 — reverted in `d08732f` |
| `d08732f` | Revert multi-literal fastloop | back to 323 MB/s bench, 2.72× silesia |
| `ee05316` | LitLenTable::TABLE_BITS 11 → 12 (16 KB main table) | **kept** — silesia 895→863 ms, ratio 2.72→2.60 (4% wall) |
| `1bda635` | LitLenTable::TABLE_BITS 12 → 13 (32 KB main table) | regressed bench 322→309, silesia 2.60→2.76 — reverted in `b61b970` |
| `9a34f32` | DistTable::TABLE_BITS 8 → 9 (512 entries) | **kept** — silesia 863→809 ms, ratio 2.60→2.41 (6% wall) |
| `160b8c6` | DistTable::TABLE_BITS 9 → 10 (1024 entries) | regressed silesia 2.41→2.85 — reverted in `6e1e322` |

**Cumulative session wins**: silesia E2E ratio **2.76 → 2.41** (wall **998 → 809 ms**, 19% reduction). Inner-inflate bench still at 318-322 MB/s (~unchanged) — the table-size wins are NOT per-symbol throughput; they reduce subtable lookups for common codes, which shows up in real-data E2E but not in the bench's whole-stream rate.

The remaining 2.41× gap to gate-2's < 0.5 target is the per-chunk decode rate (322 MB/s pure-Rust vs 779 MB/s ISA-L). To close it requires real algorithmic changes — multi-symbol packed table (vendor's TRIPLE_SYM pattern, already in `HuffmanCodingShortBitsMultiCached` used by bootstrap), BMI2 pext, or libdeflate-specific tricks. The 4-candidate plan stays; A (BMI2) and D (libdeflate scalar) are now next.

**Why the multi-literal fastloop regressed**: `vector_huffman::decode_multi_literals` (4-symbol lookahead) wins when literal clusters are common in the data. Silesia is binary/mixed-content (tar of various corpus files); short literal runs are interrupted by length codes, so the lookahead returns `count=0` on most calls and the per-iteration overhead isn't recovered. The advisor's hypothesis #4 (low hit rate on length-heavy data) was correct.

**Known infrastructure issue**: The `pub static` counters in `resumable.rs` are duplicated between the lib and bin compilations of the same crate name (addresses differ by 160 bytes). main.rs `use gzippy::...::COUNTER` reads a different instance than `decode_huffman_body_resumable` increments. Workaround: read counters from a unit-test in the lib, or move main.rs to a separate crate. Both feasible; track when next diagnostic needs counter visibility from the CLI.

**What to try next for item #1**:
- BMI2 `pext` bit extraction in the per-symbol decode (no table reshape, just faster bit shifts). Independent of cluster patterns.
- FIXED-Huffman static-table specialization in `decode_huffman_body_resumable` (FIXED blocks have known code lengths, table doesn't need rebuilding).
- Larger TABLE_BITS in LitLenTable (12 or 13 bit main table → fewer subtable hits on common codes).
- libdeflate's specific scalar tricks (it's faster per-call than `decode_huffman_cf_vector` minus multi-literal because of subtler optimizations in `lookup`/`consume_entry`).

Each one independent of cluster-pattern assumptions, so each one's win is measurable in isolation.

## How to capture the next measurement

### Flame-graph (page-faults + user-space hot spots)

```bash
ssh -J neurotic root@REDACTED_IP
cd ~/gzippy
git fetch origin && git checkout perf/pure-rust-inflate
cargo build --release --features pure-rust-inflate
perf record -F 999 -g --call-graph dwarf -o /tmp/perf.data -- \
  ./target/release/gzippy -d -c /root/benchmark_data/silesia-gzip.tar.gz > /dev/null
perf script -i /tmp/perf.data | inferno-collapse-perf > /tmp/perf.collapsed
# Page-fault percentage:
awk -F' ' '{s+=$NF} END {print "total=" s}' /tmp/perf.collapsed
grep -E 'asm_exc_page_fault|clear_page_erms' /tmp/perf.collapsed | \
  awk -F' ' '{s+=$NF} END {print "page_fault=" s}'
# SVG:
inferno-flamegraph < /tmp/perf.collapsed > /tmp/flame.svg
```

For user-space Rust symbols to demangle cleanly in flame-graphs,
build with `RUSTFLAGS='-C force-frame-pointers=yes'` and
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`.

### Per-chunk span trace (where time goes inside a chunk)

```bash
ssh -J neurotic root@REDACTED_IP
cd ~/gzippy
cargo build --release --features pure-rust-inflate
rm -f /tmp/sm.log
GZIPPY_LOG_FILE=/tmp/sm.log ./target/release/gzippy -d -c \
  /root/benchmark_data/silesia-gzip.tar.gz > /dev/null
python3 - <<'PY'
import json, statistics as st
boot, infl, app, pop = [], [], [], []
phases = {}
for line in open('/tmp/sm.log'):
    d = json.loads(line)
    if d.get('ev') == 'decode_span':
        boot.append(d['bootstrap_us'])
        infl.append(d['inflate_us'])
        phases[d['phase']] = phases.get(d['phase'], 0) + 1
    elif d.get('ev') == 'post_process_span':
        app.append(d['apply_window_us'])
        pop.append(d['populate_subchunk_windows_us'])
def stats(v, label):
    if not v: return f'{label}: empty'
    v = sorted(v)
    return f'{label}: n={len(v)} sum={sum(v)/1000:.0f}ms p50={v[len(v)//2]}us p95={v[int(len(v)*0.95)]}us max={v[-1]}us'
print(stats(boot, 'bootstrap'))
print(stats(infl, 'inflate'))
print(stats(app, 'apply_window'))
print(stats(pop, 'populate_subchunk_windows'))
print('phases:', phases)
PY
```

Also: `scripts/parallel_sm_log_summary.py /tmp/sm.log` for the
higher-level event summary (per-chunk decode_ok, speculation
outcomes, BlockFinder counts).

### Bench gates

```bash
# Gate 1 — inner-inflate
cargo bench --features pure-rust-inflate --bench inflate_isal_vs_pure_rust

# Gate 2 — E2E silesia (the deliverable's primary gate)
cargo test --release --features pure-rust-inflate -- \
  test_single_member_parallel_silesia --ignored --nocapture

# Sanity — isal-compression production path stays green
cargo test --release --features isal-compression -- routing
cargo test --release --features isal-compression -- \
  test_single_member_parallel_silesia --ignored --nocapture
```

### CI-friendly local checks (30 sec)

```bash
make                                # local CI battery
cargo test --release -- routing     # routing-table assertions
cargo test --release --features pure-rust-inflate -- routing
```

## Anti-mistake rules (mirror of `plans/rust-rapidgzip.md`)

Before touching any work item:

1. **Re-measure on neurotic.** The number quoted in the
   "Live baseline" section above may be stale; the page-fault
   measurement was stale for weeks.

2. **Confirm via vendor citation.** Open the cited
   `vendor/rapidgzip/.../...hpp:line` and read it. If the proposed
   change doesn't have a vendor counterpart, reject it.

3. **Confirm not already shipped.** `cargo tree`, `grep -rn`. The
   `arena-allocator` Vec story was instructive.

4. **Mind which bench measures which path.** The 334 MB/s number is
   `ResumableInflate2`, not `deflate_block::Block`. Don't conflate.

5. **Real silesia, not synthetic PRNG.** Use
   `test_single_member_parallel_silesia` for perf gates, not
   `..._silesia_class_...` (that's an adversarial degenerate-data
   check; expected RED on pure-rust-inflate until item #1 lands).

6. **Re-measure after landing.** Items interact via the chunk
   pipeline. Even after item #1's gate 1 passes, re-run gates 2-3.

## Known pre-existing failures (NOT release-blocking, deferred per CLAUDE.md)

These are summarized in `plans/rust-rapidgzip.md`'s §5 retrospective.
Status, in this measurement-doc's terms:

1. `test_parallel_sm_propagates_errors_not_fallbacks` — not on any
   work-item path; cleanup-only.
2. `with_until_bits_resume_non_byte_aligned_with_dict` — **work item
   #2.** Blocks SIMD landing in item 1(a) `decode_huffman_body_resumable`.
3. `cross_chunk_resume_silesia_gzip9_chunk0_handoff` — re-reds if
   item 1's SIMD path has off-by-one window stitching.
4-5. `resumable_isal_oracle::*` — fixture bug, trivial cleanup.

## Trace event reference

These events ship in the tree (commit `6788651` for the new spans).
Listed for grep-ability when extending the analysis.

| event | thread | file:fn | purpose |
|---|---|---|---|
| `drive_begin` / `drive_end` | consumer | chunk_fetcher.rs `drive` | overall wall time |
| `submit_decode` | consumer | chunk_fetcher.rs | partition dispatch |
| `decode_ok` / `decode_err` | worker | chunk_fetcher.rs | per-chunk decode wall (fast/slow path) |
| `decode_span` | worker | gzip_chunk.rs `decode_chunk_marker_bootstrap_then_isal` | bootstrap_us vs inflate_us |
| `submit_post_process` | consumer | chunk_fetcher.rs | post-process pool dispatch |
| `post_process_span` | post_process | chunk_fetcher.rs `run_post_process_task` | materialize_us / apply_window_us / populate_subchunk_windows_us |
| `consume_done` | consumer | chunk_fetcher.rs | bytes flowed to writer |
| `speculative_accept` / `_missing` / `_mismatch` | consumer | chunk_fetcher.rs | speculation outcomes |
