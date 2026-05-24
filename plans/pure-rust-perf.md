# Pure-Rust perf — measurement log + active work-item status

This document tracks the live measurements that prove (or refute)
progress on the **one deliverable** stated in `plans/rust-rapidgzip.md`.

It is not a phase plan. There is exactly one deliverable; this file
is its dashboard.

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
ssh -J neurotic root@10.30.0.199
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
ssh -J neurotic root@10.30.0.199
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
