# Pure-Rust parallel SM — performance plan

§5 of `rust-rapidgzip.md` is correctness-complete: pure-Rust resumable
inflate + vendor-faithful parallel SM ports, both backends green on
neurotic (33/33 pure-rust-inflate routing + 32/32 isal-compression).
This document is the *performance* follow-up: drive the
`--features pure-rust-inflate` path from "works" to "matches and then
exceeds rapidgzip's patched-ISA-L parallel SM."

## Baseline measurements (2026-05-24, neurotic, 16-core x86_64)

### Inner inflate — full-stream silesia deflate body, single-threaded

`benches/inflate_isal_vs_pure_rust.rs` (best of 3 runs):

| Decoder                                 | Throughput |
| --------------------------------------- | ---------- |
| Patched ISA-L via `IsalInflateWrapper`  | 799 MB/s   |
| Pure-Rust `ResumableInflate2`           | 334 MB/s   |
| **Ratio (ISA-L / pure-Rust)**           | **2.39×**  |

Plan §5 Tier 1 gate: ≤ 1.5× ISA-L. **Currently failing by ~0.9×.**

### End-to-end parallel SM — real silesia.tar.gz (67 MiB → 211 MiB), T=16

`test_single_member_parallel_silesia` with
`--features pure-rust-inflate` (Phase A's real gate per advisor audit):

| Path                                | Wall time | Output rate |
| ----------------------------------- | --------- | ----------- |
| T=1 sequential (libdeflate one-shot) | 361 ms    | 586 MB/s    |
| T=16 parallel SM (pure-Rust resumable inflate) | 998 ms | 212 MB/s |
| **Ratio (par/seq)**                 | **2.76× SLOWER parallel** |

Per-chunk decode trace from `GZIPPY_LOG_FILE`:

- 75 chunks decoded, p50=39ms, p95=109ms, max=172ms per chunk
- For ~1.6 MB output per chunk @ 334 MB/s pure-Rust inflate rate, expected ≈ 5-8ms
- **Per-chunk decode is ~5-10× slower than the inflate-only bench predicts**
- Speculation outcomes: 62.9% accepted, 28.6% missing (slow-path
  boundary search), 2.9% mismatched

### Page-fault CPU% (2026-05-24 fresh measurement, neurotic, real silesia)

`perf record -F 999 --call-graph dwarf` on
`./target/release/gzippy -d -c silesia-gzip.tar.gz --features pure-rust-inflate`:

| Component                                    | % of weighted cycles |
| -------------------------------------------- | -------------------- |
| `asm_exc_page_fault` ∪ `clear_page_erms` (all stacks) | **17.2%**     |
| `clear_page_erms` alone                      | 0.7%                 |

**Phase A's premise was stale.** The 40% page-fault number cited in
prior plan text was measured BEFORE arena-allocator
(`Cargo.toml:39,50-51`) + per-worker LIFO pool
(`chunk_buffer_pool.rs:108-204`) landed. Current measurement matches
rapidgzip's 17% — Phase A's page-fault gap is already closed. The
2.76× slowdown is NOT page-fault-bound.

### End-to-end parallel SM — synthetic PRNG fixture, T=16 (adversarial)

`test_single_member_parallel_silesia_class_not_slower_than_sequential`
runs a 24 MiB **PRNG-generated** fixture compressed with
`flate2::Compression::best()`. PRNG is incompressible; the gzip
stream is dense literals with few back-refs and few block boundaries
— the speculation-pathology corner case CLAUDE.md flags. Ratios
(parallel/seq): 3.5×–7.7× across runs. **This fixture is NOT the
Phase A gate**; it's a graceful-degradation check. Loosen its
threshold to 6× (catch catastrophe, not require parity).

### vs rapidgzip parallel SM on x86_64

gzippy's `--features isal-compression` parallel SM ≈ rapidgzip
parallel: same patches, same ISA-L underneath, same chunked-decode
architecture. That path is covered by `make ship`'s authoritative
bench. The **pure-Rust** path is currently slower than gzippy's own
sequential libdeflate decoder; the per-chunk-decode penalty (≥5×
the inflate-rate prediction) means inner-inflate SIMD alone won't
close the gap. Phase B must first locate WHERE the per-chunk
penalty lives — see "Phase B — instrument before optimizing."

## Six phases to exceed rapidgzip (from `rust-rapidgzip.md §"Beyond parity"`)

Each phase is its own branch + PR + bench-on-neurotic. Abandon any
that doesn't beat its prior measurement.

### Phase A — Close the page-fault gap (CLOSED 2026-05-24)

**STATUS: COMPLETE.** Re-measurement on neurotic against real
silesia.tar.gz shows `asm_exc_page_fault + clear_page_erms` at
**17.2%** (matches rapidgzip's documented 17%). The arena-allocator
+ per-worker LIFO pool already in `main` closed this gap. The
remaining slowdown of pure-Rust parallel SM vs sequential libdeflate
(2.76× ratio on real silesia) is NOT page-fault-bound. See "Phase B
— instrument before optimizing."

**Historical premise (kept for context):**
`src/decompress/parallel/chunk_buffer_pool.rs:73-82` originally
documented ~40% page-fault CPU% vs rapidgzip's 17%. That measurement
predated the `arena-allocator` (`Cargo.toml:39,50-51`) + per-worker
LIFO pool (`chunk_buffer_pool.rs:108-204`) shipping on `main`. The
2.76× E2E slowdown is from a different cause — per-chunk decode is
5-10× slower than the inflate-only bench would predict, meaning
~85% of per-chunk wall time is something OTHER than inflate
(suspected: marker bootstrap + BlockFinder + speculation overhead).

**If a future regression brings page-faults back, the levers (in
decreasing safety) are:**

1. **Worker-local pre-touch of pool buffers.** The consumer-thread
   prewarm regressed −50% (`chunk_buffer_pool.rs:84-88`); worker-local
   has not been tried. After `bind_worker_pool_index`
   (`chunk_buffer_pool.rs:124`), iterate the pool and write a zero
   to each pooled Vec's first byte per 4 KiB page. One fault per page
   *once per worker*, not once per chunk.
2. **`MADV_HUGEPAGE` on chunk buffers ≥ 2 MiB.** New code in
   `rpmalloc_alloc.rs::types::u8_with_capacity`:
   ```rust
   #[cfg(target_os = "linux")]
   if cap >= 2 * 1024 * 1024 {
       unsafe {
           libc::madvise(v.as_mut_ptr().cast(), v.capacity(),
                         libc::MADV_HUGEPAGE);
       }
   }
   ```
3. **`#[global_allocator] = rpmalloc::RpMalloc`.** Highest risk
   (`chunk_buffer_pool.rs:94-96` flags it "unproven"; prior
   mimalloc/jemalloc tries regressed). Requires adding the `rpmalloc`
   crate (not `rpmalloc-sys`) and a `static GLOBAL: RpMalloc =
   RpMalloc;` in `main.rs`.

**Exact commands** (run on neurotic over `ssh -J neurotic root@10.30.0.199`):

```bash
# Build (pure-rust feature, release):
cargo build --release --features pure-rust-inflate

# Correctness gate (must stay green):
cargo test --release --features pure-rust-inflate -- routing

# Bench gate (the failing test today):
cargo test --release --features pure-rust-inflate -- \
  test_single_member_parallel_silesia_class_not_slower_than_sequential \
  --ignored --nocapture

# Inner-inflate bench (the 799 / 334 MB/s baseline):
cargo bench --features isal-compression \
  --bench inflate_isal_vs_pure_rust -- --nocapture
```

**Bench gate (pass/fail) — superseded.** The 4.04 / page-fault gates
above were Phase A measurements. After 2026-05-24's re-measurement
(17% page-fault — already at rapidgzip parity), Phase A's exit gate
collapses to: re-measure annually to detect regression. **The active
gate is now Phase B's bench** (real silesia ratio < 1.0 vs T=1
libdeflate, then < 0.5).

**Flame-graph capture (on neurotic):**

```bash
ssh -J neurotic root@10.30.0.199
cd ~/gzippy-dev
cargo build --release --features pure-rust-inflate
perf record -F 999 -g --call-graph dwarf -- \
    ./target/release/gzippy -d -c benchmark_data/silesia-gzip.tar.gz > /dev/null
perf script | inferno-flamegraph > /tmp/flame-pure-rust.svg
```

### Phase B — Instrument before optimizing (CURRENT FOCUS)

**Step zero**: locate the per-chunk wall-time penalty. On real
silesia, per-chunk decode is ~47ms p50 (109ms p95); the inflate-only
bench says pure-Rust does 334 MB/s → ~5-8ms for a 1.6 MB chunk. So
85% of per-chunk wall time is something OTHER than inner inflate.
Optimizing inflate alone (doubling to 600 MB/s) would only shave
~3ms — leaves the gap effectively unchanged. **Find the missing 40ms
before writing SIMD code.**

Per-chunk timing spans to add (worker thread, gated on `trace::is_enabled()`):

1. **`bootstrap`** — `decode_chunk_marker_bootstrap_then_isal` from
   stream entry to first inflate byte (Huffman-table parse for
   DYNAMIC blocks, or BlockFinder candidate-walk if start_bit ≠
   block boundary).
2. **`inflate_to_markers`** — pure-Rust resumable inflate emitting
   u16 marker stream (`ResumableInflate2::read_stream`).
3. **`replace_markers`** — `apply_window` resolving u16 markers vs
   the 32 KiB predecessor window (AVX2 path already in
   `replace_markers.rs:147-181`).
4. **`window_publish`** — building per-subchunk tail windows
   (`populate_subchunk_windows`) + consumer-side
   `publish_subchunk_windows`.
5. **`output_copy`** — consumer-thread narrow + write to user's
   writer.

Emit p50/p95/max for each span. `scripts/parallel_sm_log_summary.py`
already parses span events; extend its summary table with the new
labels.

**Likely outcomes** (in decreasing prior):
- Marker bootstrap dominates on slow-path chunks (no window → must
  re-bootstrap DYNAMIC table per chunk). Fix: amortize boundary search
  → fewer "slow" decodes. See Phase E (speculation depth).
- Marker replace is small (already AVX2) — verify, not assume.
- Inner inflate is the predicted 5-8ms per chunk; SIMD primitives buy
  ~3ms per chunk = 5-10% of wall time, not the order-of-magnitude
  fix.
- Output copy is small (consumer is single-thread but bytes are
  already resolved).

**Then**, once measurements show the hot span:

#### B.SIMD — SIMD inflate primitives on the resumable path

Only if Step zero shows inflate is ≥ 30% of per-chunk wall time.
gzippy has `vector_huffman`, `simd_huffman`, `two_level_table`,
`packed_lut`, `combined_lut`, `bmi2` (`src/decompress/inflate/`,
`src/decompress/`). They're production for BGZF + sequential
decompress. NOT on the parallel SM resumable path because
`ResumableInflate2`'s body is `decode_huffman_body_resumable` →
generic libdeflate-style table lookup with no SIMD batching.

Sub-tasks:
- Wire `vector_huffman` multi-symbol decode for short codes
  (≤ TABLE_BITS).
- Bring `double_literal` two-symbol cache for back-to-back short
  literals.
- Specialize `copy_match_windowed` for `distance > 32` (bulk
  `copy_match_safe`-style) and for `distance > out_pos` (window
  branch) separately.
- Drop the per-byte loop in `copy_match_windowed` for the common
  case where `distance > out_pos` is false.

Bench gate: inner-inflate ratio ≤ 1.2× ISA-L (Tier 2). E2E
parallel-SM ≥ 2× faster than sequential (the silesia test gate).

#### B.BOOT — bootstrap amortization

If Step zero shows DYNAMIC bootstrap dominates on slow-path chunks:
- Cache the Huffman-table parse for block boundaries already visited
  (`block_map.rs` already keys by start_bit).
- Tighten BlockFinder pre-scan so workers receive a known-good
  boundary rather than scanning per-chunk
  (`raw_block_finder.rs`).

This is closer to Phase E (SIMD BlockFinder + speculation depth).

### Phase C — Architecture-specific dispatch

`target_feature` + CPUID runtime dispatch (`multiversion` crate or
hand-rolled `is_x86_feature_detected!`). AVX-512, AVX2, NEON variants
of the inner-loop hot paths.

- AVX-512 BMI2 `pext`/`pdep` for bit-shuffling — speeds up Huffman
  decode and bit extraction.
- AVX2 256-bit copies for `length >= 32` matches.
- NEON dispatch for arm64. Today the parallel SM path is x86-only
  (gated by `sm_cfg::PARALLEL_SM`); arm64 falls back to libdeflate.
  Lift that gate once NEON variants exist.

Bench gate: per-ISA flavor compared against ISA-L's equivalent ISA
flavor.

### Phase D — Pipeline overlap

One rapidgzip doesn't fully exploit, one already shipped:

1. **CRC32 in flight with decode.** Hardware CLMUL CRC32 issues at
   ~1 byte/cycle. Today gzippy + rapidgzip compute CRC32 after the
   chunk decode completes. Interleaving CRC compute with literal/
   match emission hides it almost entirely.

2. **(DONE on this branch — verify before re-doing.)** `apply_window`
   already runs on the worker via `run_post_process_task`
   (`chunk_fetcher.rs:1405-1412`). What remains on the consumer is
   `publish_subchunk_windows` — the WindowMap index write. Moving
   that earlier is the remaining D opportunity if profiling shows
   the consumer stalls on it.

Bench gate: per-chunk wall time → not pipeline-stalled by serialized
post-decode work.

### Phase E — SIMD BlockFinder + deeper speculation

`src/decompress/parallel/raw_block_finder.rs` scans 8 KiB-bit windows
sequentially looking for valid block starts. Vendor's
`BlockFinder.hpp` + `blockfinder/DynamicHuffman.hpp` do the same.

- Multi-byte SIMD pattern match for `0b10`/`0b01` followed by HLIT/
  HDIST ranges that gate valid block headers. Probably 4-8× faster
  boundary scan.
- Speculate two-three boundaries ahead per worker instead of one.

Bench gate: time-to-first-output drops; T=16 utilization stays
> 14 cores.

### Phase F — Memory bandwidth tricks

After A-E, throughput approaches DRAM bandwidth. Then:
- Non-temporal stores for the output writer on very large files
  (> 500 MiB).
- NUMA-aware worker pinning. On 2-socket boxes, pin workers to the
  socket whose memory holds the input mmap.
- Pre-fault output with `MADV_POPULATE_WRITE`.

Marginal returns (5-15%) individually, multiplicative atop A-E.

## Discipline

Per the §5 retrospective:

1. **Bench before code.** Every phase opens with a flame-graph and a
   throughput measurement on neurotic. No phase commits until that
   measurement repeats green at the PR's bench gate.
2. **Get adversarial Opus reviews on judgment calls.** §5 dodged
   two infinite-loop bugs only because we paused for advisor
   diagnosis instead of patching blindly. Same rule here.
3. **Don't break the isal-compression path.** It's the production
   default. Every PR's CI must keep `isal-compression` green on
   neurotic.
4. **Monitor every long-running job.** 10-min health-check monitors
   for any `cargo bench` / neurotic ssh that might hang.

## Troubleshooting

### If Phase A doesn't move the bench number — first 3 things to check

1. **`arena-allocator` actually on?** `cargo tree --features
   pure-rust-inflate | grep rpmalloc-sys`. If missing, feature graph
   broke.
2. **Pool being hit?** Add `eprintln!` of `TAKE_U8_HITS` / `MISSES`
   (`chunk_buffer_pool.rs:206-207`) at end of the silesia-class test.
   Hit-rate < 50% on iters 2-3 means workers aren't returning buffers
   — check `chunk_data.rs:1079-1085` Drop is firing.
3. **Fixture re-compresses on every iter?** `routing.rs:651-657` builds
   the compressed fixture inside the test fn. The 3-iter loop
   (`routing.rs:666`) is decode-only, but if you moved any compression
   into the loop you'd double-count.

### If Phase B's SIMD primitives produce wrong output — first 3 things to check

1. **Failure #2 status.** If
   `with_until_bits_resume_non_byte_aligned_with_dict` is still red,
   the subtable `total_bits` bug at `resumable.rs:812-818`
   (`consume(TABLE_BITS=11)` then `consume_entry(subtable.raw())`) is
   biasing every bit-position downstream. Fix first or the SIMD path
   inherits the bug.
2. **Stopping-point bookkeeping at literal clusters.**
   `decode_huffman_cf_vector` (`consume_first_decode.rs:582+`) has no
   yield mid-block; `decode_huffman_body_resumable` must check
   `bit_position() >= encoded_until_bits` after every batched cluster,
   not just every iteration of the outer loop.
3. **`copy_match_windowed` window precondition.** The window-stitched
   copy is per-byte; SIMD bulk replacement requires
   `distance <= out_pos - window_head`. Off-by-one reads
   uninitialized window bytes — the bug surfaces only on cross-chunk
   boundaries (`cross_chunk_resume_silesia_gzip9_chunk0_handoff` would
   re-red).

### Verify isal-compression production path stays green after any pure-rust change

```bash
# Local (30s):
cargo test --release --features isal-compression -- routing

# Authoritative (neurotic):
ssh -J neurotic root@10.30.0.199 \
  'cd ~/gzippy-dev && cargo test --release --features isal-compression -- \
     test_single_member_parallel_silesia_class_not_slower_than_sequential \
     --ignored --nocapture'
# Ratio must stay < 0.5 on the 16-core homelab box.
```

### Confidence ratings (calibrated 2026-05-24 post-fresh-measurement)

| Target | Confidence | Notes |
|---|---|---|
| Phase A — closed | HIGH | 17.2% page-fault on real silesia matches rapidgzip's 17% |
| Phase B Step zero (find missing 40ms/chunk) | HIGH | Pure measurement work; new spans land in trace.rs |
| Phase B fix-on-first-try (closing 47ms → 15ms p50) | LOW-MEDIUM | Depends what Step zero finds — bootstrap or replace_markers are the prior suspects |
| Phase B reaches Tier 2 (1.2× ISA-L inner inflate) | MEDIUM | Integration risk: `decode_huffman_body_resumable` has stopping-point bookkeeping that `decode_huffman_cf_vector` lacks |
| Phase B reaches Tier 3 (1.05× ISA-L; delete ISA-L submodule) | LOW-MEDIUM | Requires breadth-of-tree deletion |
| Phase C / D | MEDIUM (C) / MEDIUM-LOW (D — half already done) | |
| Phase E | MEDIUM | Speculation pathology (28.6% missing) is real; SIMD BlockFinder + deeper prefetch hits this |
| Phase F | LOW | Most relevant on > 500 MiB files |

## Known issues to clean up while we're here (orthogonal to perf)

From `rust-rapidgzip.md "Known pre-existing failures"`:

1. `test_parallel_sm_propagates_errors_not_fallbacks` — silent
   multi→sequential fallback at `mod.rs:188` violates CLAUDE.md
   "no fallbacks" rule. Fix scope: gzip-format classifier + routing.
2. `with_until_bits_resume_non_byte_aligned_with_dict` — synthetic
   fixture; suspected subtable `total_bits` accounting in
   `decode_huffman_body_resumable`. Worth chasing because it might
   point to a real bit-accounting issue that limits Phase B.
3. `cross_chunk_resume_silesia_gzip9_chunk0_handoff` — zlib-ng
   resume at chunk-end-bit (same class as `03c8f48`).
4-5. `resumable_isal_oracle::*` — `make_multi_block_deflate` fixture
   bug on flate2 1.x. Trivial: replace fixture builder, repoint.

Pick these up opportunistically when the active phase touches the
relevant area, not as scheduled work.
