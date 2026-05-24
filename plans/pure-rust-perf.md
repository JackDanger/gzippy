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

### End-to-end parallel SM — silesia-class 24 MiB, T=16

`test_single_member_parallel_silesia_class_not_slower_than_sequential`
(`--features pure-rust-inflate`):

- Parallel/sequential ratio: **4.04** (parallel is 4× SLOWER than
  sequential). Test gate wants < 0.5.

The e2e gap is bigger than the inner-inflate gap because per-chunk
overhead (allocator, page faults, cross-worker coordination) doesn't
amortize when the inner decode is slow.

### vs rapidgzip parallel SM on x86_64

gzippy's `--features isal-compression` parallel SM ≈ rapidgzip
parallel: same patches, same ISA-L underneath, same chunked-decode
architecture. That path is covered by `make ship`'s authoritative
bench. The **pure-Rust** path is currently slower than gzippy's own
sequential decoder; therefore much slower than gzippy-isal-parallel,
and hence much slower than rapidgzip parallel — easily 4–8× on
typical fixtures.

## Six phases to exceed rapidgzip (from `rust-rapidgzip.md §"Beyond parity"`)

Each phase is its own branch + PR + bench-on-neurotic. Abandon any
that doesn't beat its prior measurement.

### Phase A — Close the page-fault gap (highest leverage)

`src/decompress/parallel/chunk_buffer_pool.rs:73-82` documents the
profile: gzippy spends ~40% of CPU in `asm_exc_page_fault +
clear_page_erms`; rapidgzip spends ~17%. Largest expected win;
independent of decoder tuning.

**Already in the build (do NOT re-do):** `Cargo.toml:39,50-51` forces
`arena-allocator`, which makes `ChunkData::data` / `data_with_markers`
`Vec<_, RpmallocAlloc>` via `rpmalloc_alloc.rs:14-89`. The per-worker
LIFO pool (`chunk_buffer_pool.rs:108-204`) recycles them via
`chunk_data.rs:1079-1085` Drop. The bench being regressed at 4.04×
WITH this in place means Phase A's remaining levers are below, not
the rpmalloc-Vec switch.

**Step zero:** re-confirm the 40% page-fault CPU% on neurotic with the
current build (see "Flame-graph" below). If it's already < 25%, Phase
A is largely done; pivot to Phase B.

**Try in order (decreasing safety):**

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

**Bench gate (pass/fail):**

- `silesia_class_not_slower_than_sequential` ratio drops from baseline
  4.04 to < 2.0 (test threshold is 0.5; intermediate Phase A wins
  land between those numbers).
- Flame-graph: `asm_exc_page_fault + clear_page_erms` total CPU%
  drops by at least 10 absolute points (40% → ≤ 30%).
- `TAKE_U8_HITS / (TAKE_U8_HITS + TAKE_U8_MISSES)` > 0.8 on iters 2-3
  of the 3-iter warm-up.

**Flame-graph capture (on neurotic):**

```bash
ssh -J neurotic root@10.30.0.199
cd ~/gzippy-dev
cargo build --release --features pure-rust-inflate
perf record -F 999 -g --call-graph dwarf -- \
    ./target/release/gzippy -d -c benchmark_data/silesia-gzip.tar.gz > /dev/null
perf script | inferno-flamegraph > /tmp/flame-pure-rust.svg
```

### Phase B — SIMD inflate primitives on the resumable path

gzippy already has `vector_huffman`, `simd_huffman`, `two_level_table`,
`packed_lut`, `combined_lut`, `bmi2` (`src/decompress/inflate/`,
`src/decompress/`). They're production for BGZF + sequential
decompress. They're NOT on the parallel SM resumable path because
`ResumableInflate2`'s body is `decode_huffman_body_resumable` →
generic libdeflate-style table lookup with no SIMD batching.

Action: extend `decode_huffman_body_resumable` to dispatch literal-
batch and match-copy through the SIMD primitives. Where these
primitives beat ISA-L on the specific code-length distributions
gzip(1) produces, this is where pure-Rust wins against ISA-L's
general inflate.

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
parallel-SM ≥ 2× faster than sequential (the existing test gate).

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

### Confidence ratings (calibrated post-advisor-audit)

| Target | Confidence | Notes |
|---|---|---|
| Phase A diagnosis | MEDIUM-HIGH | 40%/17% number from one-off flame-graph; re-measure first |
| Phase A fix-on-first-try | LOW-MEDIUM | Vec<T,RpmallocAlloc> already shipped; real fix is MADV_HUGEPAGE or global rpmalloc — both flagged "unproven" |
| Phase B reaches Tier 2 (1.2× ISA-L) | MEDIUM | Integration risk: `decode_huffman_body_resumable` has stopping-point bookkeeping that `decode_huffman_cf_vector` lacks |
| Phase B reaches Tier 3 (1.05× ISA-L; delete ISA-L submodule) | LOW-MEDIUM | Requires breadth-of-tree deletion |
| Phase C / D | MEDIUM (C) / MEDIUM-LOW (D — half already done) | |
| Phase E | MEDIUM-LOW | |
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
