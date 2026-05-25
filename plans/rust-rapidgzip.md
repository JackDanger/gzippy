# The deliverable

**ONE deliverable, exactly one:**

> A pure-Rust parallel single-member gzip decoder whose end-to-end
> throughput on real silesia matches or exceeds rapidgzip-with-patched-
> ISA-L, with `vendor/isa-l` + `vendor/isal-rs` + `packaging/isal-patches/`
> + `backends/isal_*.rs` all deleted from the tree and CI green.

Everything in this document is in service of that one outcome.
Anything that does not move a measurement toward it is not on the
critical path.

## Done when (all four)

1. `cargo bench --features pure-rust-inflate --bench inflate_isal_vs_pure_rust`
   reports **pure-Rust ≥ 1/1.05× ISA-L** on neurotic. (Currently 1/2.39×.)
2. `cargo test --release --features pure-rust-inflate -- \
   test_single_member_parallel_silesia --ignored --nocapture` reports
   **ratio < 0.5** (T=16 parallel < ½ T=1 sequential) on neurotic.
   (Currently 2.76× the wrong direction.)
3. `make ship` reports pure-rust-inflate throughput **≥ rapidgzip**
   throughput on the silesia corpus, same hardware.
4. `vendor/isa-l`, `vendor/isal-rs`, `packaging/isal-patches/`,
   `backends/isal_decompress.rs`, `backends/isal_compress.rs` are
   deleted; `arena-allocator` is decoupled from `isal-compression` in
   `Cargo.toml`; CI is green with `pure-rust-inflate` as the only
   inflate backend.

If you're reading this in 2027 and trying to figure out whether the
work is finished: run those four. If all four are green, it's done.
If any one is red, it isn't, and the work items below say what's left.

## The six work items

In dependency order. Each one names files, vendor citations, and the
measurement that proves it's done. **Re-measure before AND after every
item.** Stale priors are this plan's #1 historical failure mode (see
"Mistakes to not repeat" below).

### 1. SIMD-ify every Huffman hot loop

Pure-Rust decode today has **three Huffman-decoding hot loops**, only
one of which is SIMD:

| Loop | File:fn | SIMD today? | Used by |
|---|---|---|---|
| (a) `decode_huffman_body_resumable` | `src/decompress/inflate/resumable.rs:793-852` | no — generic libdeflate-style table lookup | parallel-SM phase 2; `inflate_isal_vs_pure_rust` bench (this IS what 334 MB/s measures) |
| (b) `read_internal_compressed_canonical_specialized` | `src/decompress/parallel/deflate_block.rs:1131-1240` | no — per-symbol scalar | parallel-SM phase 1 bootstrap (the **bottleneck** at 1060ms / chunk per `plans/pure-rust-perf.md`) |
| (c) `decode_huffman_cf_vector` | `src/decompress/inflate/consume_first_decode.rs:571+` | **yes** | BGZF + sequential decompress (production) |

The deliverable requires (a) and (b) on the same SIMD primitives as (c):

- `vector_huffman` — multi-symbol decode for short codes (≤ TABLE_BITS).
- `two_level_table` — main-table + subtable.
- `packed_lut` / `combined_lut` — fused literal/length.
- `bmi2` `pext` — bit-shuffling on AVX2+BMI2 hosts.
- `double_literal` — back-to-back short-code cache.

Plus the CONTAINS_MARKERS specialization in (b) preserved.

**Vendor reference**: `vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp::Block::readInternalCompressedMultiCached`
(deflate.hpp:1585-1666) — the SIMD-friendly shape we are porting.
gzippy's (b) cites this vendor function already; today it specializes on
ISA-L LUTs for `isal-compression` builds and falls through to the
scalar canonical decoder for `pure-rust-inflate`. The pure-Rust
specialization gets the SIMD primitives.

**Bench gate**:

- Inner-inflate (`inflate_isal_vs_pure_rust`): pure-Rust ≥ 600 MB/s
  (Tier 2: 1/1.2× ISA-L), then ≥ 760 MB/s (Tier 3: 1/1.05× ISA-L).
- Bootstrap span p50 in `decode_span` trace: 14.2ms → ≤ 6ms.

### 2. Fix subtable bit-accounting (failure #2)

`with_until_bits_resume_non_byte_aligned_with_dict` is red because
`decode_huffman_body_resumable` at `src/decompress/inflate/resumable.rs:812-818`
biases `total_bits` by 9 bits across the subtable boundary. Until
this is fixed, the SIMD batched-decode in item 1's loop (a) will
**inherit the bias** and silently produce wrong bit-position
bookkeeping on cross-chunk boundaries — and the failure surfaces
only on specific fixtures (`cross_chunk_resume_silesia_gzip9_chunk0_handoff`
would re-red).

**Blocks**: SIMD landing in (a) requires this fix first. (b)'s SIMD
work is decoder-independent and can land before #2; we don't need a
hard ordering between #1(b) and #2.

**Done when**: `cargo test --release --features pure-rust-inflate -- \
with_until_bits_resume_non_byte_aligned_with_dict` is green.

### 3. Make `RawBlockFinderCoordinator` proactive

Today gzippy's `RawBlockFinderCoordinator` (`src/decompress/parallel/raw_block_finder.rs`)
is invoked **reactively** — only after a chunk's start_bit fails
trial-decode in `speculative_decode_find_boundary`
(`chunk_fetcher.rs:1512`). Each worker spawns its own finder
thread on miss.

Vendor's `BlockFinder<RawFinder>` (`vendor/rapidgzip/librapidarchive/src/core/BlockFinder.hpp`)
is **proactive**: single background `JoiningThread` running
`blockFinderMain()` that prefetches block offsets ahead of when
workers need them, populating `m_blockOffsets`. Workers consume
from the prefetch queue.

**Effect on bench**: today's trace shows **28.6% missing speculation**
on real silesia. Proactive prefetch should drop that toward < 10%,
meaning more chunks hit phase 2 (fast SIMD path from item 1) with a
known boundary instead of paying the slow-path candidate walk.

**Files to touch**:
- `chunk_fetcher.rs` — `drive`, `consumer_loop`, `submit_decode_to_pool`:
  inject prefetched offsets into BlockFetcher's offset cache when
  partitions are created, BEFORE worker dispatch.
- `raw_block_finder.rs` — extend coordinator to run as a
  long-lived single thread (one per `drive` call), not per-chunk
  scoped.

**Vendor side-by-side**: `BlockFinder.hpp:60-90` startThreads /
stopThreads, `BlockFinder.hpp:130-180` blockFinderMain pattern.

**Bench gate**: trace summary `speculative_missing` drops from 28.6%
to ≤ 10% on real silesia. E2E silesia ratio drops accordingly.

### 4. Retire phase-1 bootstrap

Once #1 puts the bootstrap decoder (b) on the same SIMD primitives
as phase 2 (a), and #2 ensures bit-accounting is correct, the two
decoders become functionally identical except for "have we seen 32
KiB of clean bytes yet?" — which is a single branch over one shared
loop body.

**Action**: delete `bootstrap_with_deflate_block` in `gzip_chunk.rs:511`.
Replace its callers with a single SIMD decoder that takes either a
window OR an empty window + marker-emission mode. The `deflate_block`
module shrinks to its `read_header` + Huffman-table-build surface
(needed for the candidate trial-decode); the per-symbol read path
deletes.

**Vendor reference**: vendor's `Block::read()` is one function with
the same template specialization (`CONTAINS_MARKERS`) gzippy already
has; vendor doesn't have a separate "bootstrap" code path. Today's
gzippy split is a porting artifact.

**Bench gate**: `decode_span` phase distribution: today 59% of chunks
are `bootstrap_only`. Post-#4, the `bootstrap_only` phase tag
disappears (the trace shows phase 2 spans only, with markers when
applicable).

### 5. CRC32 interleave + worker-side window publish

Phase D leftovers, mechanical but real:

- **CRC32**: hardware CLMUL CRC32 issues at ~1 byte/cycle.
  Today gzippy + rapidgzip both compute CRC32 *after* the chunk
  decode completes. Interleaving CRC computation with literal/match
  emission hides it almost entirely.
  Vendor reference: `vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/GzipChunk.hpp`'s
  `m_crc32`. Files to touch: `chunk_data.rs` (incremental CRC), the
  SIMD decode loop from item 1 (CRC update on literal emission).

- **`publish_subchunk_windows`**: today the consumer thread writes
  the WindowMap index after each chunk's apply_window completes
  (`chunk_fetcher.rs::publish_subchunk_windows`). `apply_window` itself
  already runs on the post-process worker pool (verified
  `chunk_fetcher.rs:1405-1412`). Moving the index publish onto the
  same worker task removes a consumer-thread serialization point.

**Bench gate**: per-chunk consumer-thread wall span (currently not
traced — add a trace event around `consume_done` for this gate). The
bench is "no consumer-side stalls on serialized post-decode work."

### 6. `Cargo.toml` cleanup + ISA-L deletion

The mechanical end-state. After items 1-5 put pure-Rust at parity:

- `Cargo.toml`: `arena-allocator` becomes a default (no longer
  transitively activated by `isal-compression`). `pure-rust-inflate`
  becomes the only inflate feature; `isal-compression` deletes.
- `vendor/isa-l`, `vendor/isal-rs`, `packaging/isal-patches/` deleted
  as submodules.
- `src/backends/isal_decompress.rs`, `src/backends/isal_compress.rs`
  deleted. Routing in `src/decompress/mod.rs` simplifies: the
  IsalSingle path collapses into the pure-Rust single-member path
  (T=1 also goes through `ResumableInflate2`).
- CI matrix simplifies from `isal-compression × pure-rust-inflate ×
  default` to one config.
- `make ship` becomes the authoritative bench on neurotic.

**Bench gate**: this IS the deliverable's gate. All four "Done when"
items above must be green simultaneously.

## Bench gates that lock the deliverable

These are the four measurements that prove the deliverable is real,
restated for ease of running:

```bash
# Gate 1 — inner-inflate parity (Tier 3)
cargo bench --features pure-rust-inflate --bench inflate_isal_vs_pure_rust
# Expect: pure-Rust ≥ 1/1.05× ISA-L

# Gate 2 — E2E silesia parallel beats sequential
cargo test --release --features pure-rust-inflate -- \
  test_single_member_parallel_silesia --ignored --nocapture
# Expect: ratio < 0.5

# Gate 3 — make ship matches rapidgzip
make ship
# Expect: pure-rust-inflate throughput ≥ rapidgzip on silesia

# Gate 4 — tree is clean
test ! -d vendor/isa-l && \
test ! -d vendor/isal-rs && \
test ! -f src/backends/isal_decompress.rs && \
test ! -f src/backends/isal_compress.rs && \
cargo build --release --features pure-rust-inflate
```

Run them on neurotic (`ssh -J neurotic root@10.30.0.199`), not on
GHA. The synthetic test
`test_single_member_parallel_silesia_class_not_slower_than_sequential`
is **NOT a deliverable gate**; it's an adversarial-PRNG
graceful-degradation check that encodes the production
(isal-compression) bar and is expected RED on pure-rust-inflate
until item #1+#4 land (see `src/tests/routing.rs:632-638` docstring).

## How to read this plan (anti-mistake rules)

The plan and the advisors have a prior. The prior has been wrong
**every time** measurement contradicted it (see "Mistakes to not
repeat" below). Until this directory has earned trust:

1. **Measure first, every time.** Before starting any work item,
   capture a fresh measurement on neurotic. The plan's claimed
   bottleneck may be stale. Do not optimize on the basis of a
   stated bottleneck without re-confirming it.

2. **Vendor citation required.** If a proposed change has no
   `vendor/rapidgzip/...:line` reference, reject it. This catches
   "advisor says vendor does X" when vendor actually doesn't.
   `find vendor/rapidgzip -name "*.hpp"` is your first move when
   verifying.

3. **"Already shipped?" check.** Before listing an action as "try
   this," grep the tree:
   - `cargo tree --features <feature> | grep <crate>` for deps.
   - `grep -rn '<symbol>' src/` for code.
   The 40% page-fault plan instructed us to "try `Vec<T, RpmallocAlloc>`
   first" — that had shipped months earlier via `arena-allocator`.

4. **Which bench measures which code path?** Document next to every
   bench *what code path it exercises*. The inner-inflate bench at
   334 MB/s measures `ResumableInflate2` (parallel-SM phase 2 / item
   1(a)). It does NOT measure `deflate_block::Block::read_internal_compressed_canonical_specialized`
   (parallel-SM phase 1 / item 1(b) — the actual bottleneck).
   Confusing these wasted half an analysis session.

5. **Real silesia is the gate.** Synthetic PRNG is incompressible
   under `flate2::best()` and produces speculation pathology (29 of
   49 chunks all-marker, no block boundaries). It's the right
   adversarial check; it is NOT the right perf gate. Use
   `test_single_member_parallel_silesia` for perf, not
   `..._silesia_class_...`.

6. **Re-measure after every item lands.** Even if item 1's bench
   gate passes, re-run gates 2-3 before declaring item 1 done.
   Items interact via the chunk pipeline; a 600 MB/s inner-inflate
   win can be eaten by a regression in BlockFinder dispatch.

## Mistakes to not repeat (caught by advisor pushback)

These almost shipped. Each one is captured here so future-us doesn't
re-make the same wrong turn.

1. **"Try `Vec<T, RpmallocAlloc>` first."** The plan listed this as
   Phase A step 1. It had already shipped via the `arena-allocator`
   feature (`Cargo.toml:39,50-51`). Would have wasted half a day.
   **Caught by**: pre-implementation audit; advisor flagged on
   plan-tuning pass.

2. **"40% page-fault is the bottleneck."** Re-measurement on
   2026-05-24 showed 17%. The 40% number was from a measurement
   taken **before** arena-allocator landed on main. The plan
   propagated the stale number for weeks.
   **Caught by**: re-measuring before writing code (anti-mistake
   rule #1).

3. **"Loosen the synthetic test gate to 6×."** First advisor
   recommendation. Wrong: the test encodes the production
   (isal-compression) bar; loosening to 6× would mask a 12×
   regression in the production path. It's `#[ignore]`-gated so it
   doesn't run in CI; expected-red on pure-rust-inflate is the
   correct state.
   **Caught by**: pushing back on the advisor with the production-vs-
   feature distinction.

4. **"Build a parent-thread SIMD BlockFinder pre-pass (option C)."**
   First advisor recommendation. Wrong: vendor's BlockFinder is
   an **async single-thread prefetcher**, NOT a parent-thread
   pre-pass. Building option C would have been a multi-week refactor
   AWAY from the vendor pattern.
   **Caught by**: checking `vendor/rapidgzip/librapidarchive/src/core/BlockFinder.hpp`
   before implementing. The advisor reversed himself when shown the
   vendor file.

5. **"Optimize replace_markers (apply_window) for AVX-512."** A
   tempting prior given marker-replacement sounds expensive. Wrong:
   `apply_window` takes 17 ms total across the whole silesia run.
   It's already AVX2 and is not a bottleneck.
   **Caught by**: instrumenting the per-chunk spans (commit `6788651`)
   before optimizing.

6. **"Per-chunk allocator overhead is the bottleneck."** An advisor
   pointed at `bootstrap_with_deflate_block`'s unpooled `Vec::with_capacity(128*1024)`,
   predicted pooling it would save 150-200 ms wall. Another advisor
   implemented the fix and measured — actual savings: **30 ms**. Page-fault
   count basically unchanged. CPU-fraction reasoning (17% of CPU in
   page-fault paths) doesn't translate to wall on a 9-thread system;
   parallelism dilutes it.
   **Caught by**: insisting an advisor actually implement and measure
   before recommending. Saved several days of allocator spelunking.

7. **"Body-failure speculation accuracy is the real lever."** V1
   instrumentation showed gzippy has 39 body failures per silesia run.
   Forensic measurement (commit `ddc3a3c`): 608 bytes wasted total,
   ~4 ms of work. Body failures are CHEAP — they abort during table
   construction before any decode. NOT the cost. 34 of 39 are
   `InvalidCodeLengths` (precode decoded into bad lengths), which a
   deeper precode validator could catch but with bounded ~4 ms savings.
   **Caught by**: per-failure structured logging
   (`GZIPPY_BODY_FAIL_LOG`).

8. **"Inflate engine choice (ISA-L vs pure-Rust) is the lever."**
   Multiple advisors suggested. Built gzippy with `--features
   isal-compression --no-default-features`: 537 ms vs pure-rust 506 ms.
   ISA-L makes it slightly SLOWER. Engine isn't the gap; the parallel
   pipeline structure around it is.

9. **"Global rpmalloc allocator will close the gap (vendor uses it)."**
   Vendor uses rpmalloc as a PER-VEC custom allocator in specific
   places (`FasterVector.hpp:38-42`), NOT as global allocator. Vendor
   explicitly avoids `rpnew.h` global override per `ChunkData.hpp:24`
   ("memory slab reuse issues"). gzippy already has per-Vec
   `RpmallocAlloc` via `allocator-api2`. Tried global anyway: regressed
   12% (matches mimalloc and jemalloc history).

10. **"Just bump prefetch_capacity from 2x to 4x."** Cheap-looking
    tunable. Cache misses dropped 4 → 3 but wall regressed 486 → 510 ms
    — overhead from larger cache ate the savings.

11. **"SIMD multi-literal lookahead (vector_huffman) ported into the
    resumable inflate body."** Sounded straightforward — port the
    production cf_vector pattern into `decode_huffman_body_resumable`.
    Regressed bench 334 → 284 MB/s and silesia 2.79 → 3.06. Reverted at
    commit `d08732f`. Silesia's mixed-content data has too few literal
    clusters for the 4-symbol lookahead to pay back its overhead.

Pattern: the plan / advisor had a prior; the prior was wrong; only
the measurement was right. **Trust the measurement, not the prior.**

## Verified findings — cited evidence (refer to plans/pure-rust-perf.md for full data)

The cross-tool Chrome-trace instrumentation (`trace_v2.rs` for gzippy,
`scripts/rapidgzip_trace_patch/` for vendor) emits matching spans so
both tools' decisions can be made from identical data shapes:

1. **The 200 ms pipeline idle = 4 specific cache misses.** Consumer
   thread's `wait.block_fetcher_get` totals 210 ms across cached chunks
   0, 201, 1855, 2545. Chunk 0 is cold-start (~40 ms unavoidable);
   the other 3 are consumer outrunning the prefetch window mid-stream.
2. **Worker load imbalance is a CONSEQUENCE of (1), not a separate
   bug.** Per-thread busy times σ=130 ms (gzippy) vs σ=7 ms (rapidgzip).
   `pool.pick` time is 6.7× higher in gzippy. Both track the same
   root cause: when consumer blocks on cache misses, no new work
   submits, workers idle in condvar wait.
3. **Per-chunk decode rate is 1.9× slower** (47 vs 25 ms avg). This
   IS separate from the cache-miss story and remains a follow-up
   lever after (1) is fixed.
4. **Equal-hardware gap is 3.4×, NOT the originally-feared 6.1×.**
   The earlier 6.1× included an unmeasured P/E hybrid CPU confound
   that closed 25% of the gap once gzippy was P-core-pinned.

## Anti-mistake rule (in addition to the six already documented)

**Rule 7 (newly added 2026-05-25): "Run any candidate fix end-to-end
on neurotic and report wall delta BEFORE recommending."** Advisors
who recommended fixes based on CPU-fraction reasoning (page-fault
%, allocator share, body-failure share) repeatedly overestimated
wall impact by 5-10×. The single advisor who actually implemented +
measured the bootstrap-Vec pool delivered the correct prediction
(30 ms, not 150-200 ms) and saved the team from a multi-day pursuit
of a phantom. **Implement + measure beats theorize, every time.**

## §5 retrospective — pure-Rust resumable inflate (correctness)

§5 (May 2026) replaced the band-aid `session: Vec<u8>` accumulator
with `ResumableInflate2` — a 32-KiB-ring resumable inflater
faithful to vendor's `gzip/isal.hpp:254-356`. Architecture decision
was "two decoders per BTYPE":

| Caller                                   | Decoder                            | Yield mid-block? |
| ---------------------------------------- | ---------------------------------- | ---------------- |
| BGZF, scan_inflate, sequential decompress | existing `decode_huffman_*`        | no |
| `ResumableInflate2::read_stream`         | new `decode_huffman_*_resumable`   | yes |

Done at commit `30839e8`. 33/33 pure-rust-inflate routing tests
green on neurotic; 32/32 isal-compression routing tests green.

**Known pre-existing failures** (NOT introduced by §5, deferred per
CLAUDE.md rules):

1. `decompress::tests::test_parallel_sm_propagates_errors_not_fallbacks`
   — silent multi→sequential fallback at `mod.rs:188` violates
   CLAUDE.md "no fallbacks." Variant mismatch (Decompression vs
   InvalidArgument); error IS surfaced. Cleanup: gzip-format
   classifier + routing.
2. `with_until_bits_resume_non_byte_aligned_with_dict` — **this is
   work item #2 above.** Subtable `total_bits` accounting at
   `resumable.rs:812-818`. Blocks SIMD in item 1(a).
3. `cross_chunk_resume_silesia_gzip9_chunk0_handoff` — zlib-ng
   resume at chunk-end-bit (same class as commit `03c8f48`).
4-5. `resumable_isal_oracle::*` — `make_multi_block_deflate`
   fixture bug on flate2 1.x. Replace fixture builder, repoint.

§5's port surface (Track A infrastructure + Track B C-free SM hot
path) is complete. The remaining work is performance, captured in
the six items above.

## Out of scope (not part of this deliverable)

- Seekable index reader (`IndexFileFormat.hpp`).
- BZIP2, ZLIB-format decoders (per CLAUDE.md scope: gzip-family only).
- Compress-side ISA-L removal (`backends/isal_compress.rs` deletion
  is gated on compress-side parity work; item #6 names the file but
  the compress replacement is a separate deliverable).

## Reading order for a fresh agent

1. `CLAUDE.md` — gzippy's goal, routing table, hard-won lessons.
2. This file — what's left, in priority order, with anti-mistake rules.
3. `plans/pure-rust-perf.md` — live measurements, active work-item
   status, how to capture the next measurement.
4. `vendor/rapidgzip/librapidarchive/src/rapidgzip/chunkdecoding/GzipChunk.hpp:468-654`
   — the vendor decoder this all ports.
