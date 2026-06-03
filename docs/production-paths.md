# gzippy Production Paths

Authoritative map of which code path runs for which input and which flag activates
it. Every routing detail is read directly from `src/`; file:line citations are
canonical. Do not trust memory or CLAUDE.md over this file — this file has been
cross-checked against source.

---

## 1. Decompression routing tree

Entry point: `decompress_gzip_libdeflate` (`src/decompress/mod.rs:238`).
The single source-of-truth classifier is `classify_gzip`
(`src/decompress/mod.rs:159`). It returns a `DecodePath` variant; the dispatcher
arms are all terminal — no silent in-body fallbacks.

```
Input (mmap'd slice, num_threads)
│
├─ has_bgzf_markers(data)?    [src/decompress/format.rs, called at mod.rs:160]
│   YES → DecodePath::GzippyParallel
│           → bgzf::decompress_bgzf_parallel(data, writer, num_threads)
│             [src/decompress/bgzf.rs]
│           ("GZ" FEXTRA subfield: gzippy's own multi-block parallel format)
│
├─ is_likely_multi_member(data)?   [src/decompress/format.rs, called at mod.rs:163]
│   YES, num_threads > 1 → DecodePath::MultiMemberPar
│     → bgzf::decompress_multi_member_parallel(data, writer, num_threads)
│       [src/decompress/bgzf.rs]
│       On Err (false gzip header sequence in stored-block data):
│         → decompress_multi_member_sequential(data, writer)
│           [src/decompress/mod.rs:547 — libdeflate, member-by-member]
│
│   YES, num_threads == 1 → DecodePath::MultiMemberSeq
│     → decompress_multi_member_sequential(data, writer)
│       [src/decompress/mod.rs:547 — libdeflate, member-by-member]
│
└─ single-member
    │
    ├─ PARALLEL_SM [cfg(parallel_sm), set by build.rs:80]
    │  AND num_threads >= parallel_sm_min_threads() [mod.rs:113 — 4 in prod, 0 w/ GZIPPY_FORCE_PARALLEL_SM]
    │  AND data.len() > MIN_PARALLEL_COMPRESSED [mod.rs:92 — 10 MiB]
    │   │
    │   ├─ NOT parallel_sm_unprofitable(data)?  [mod.rs:144 — ratio < 1.15 → unprofitable]
    │   │   YES (compressible) → DecodePath::IsalParallelSM
    │   │     → parallel::single_member::decompress_parallel(data, writer, num_threads)
    │   │       [src/decompress/parallel/single_member.rs:126]
    │   │         → sm_driver::read_parallel_sm [src/decompress/parallel/sm_driver.rs]
    │   │           → chunk_fetcher::drive [src/decompress/parallel/chunk_fetcher.rs]
    │   │             (speculative marker pipeline — see §3)
    │   │
    │   └─ unprofitable (stored-dominated, ratio ~1.0)
    │       ├─ first_block_is_stored(data)?  [stored_split.rs, called at mod.rs:186]
    │       │   YES → DecodePath::StoredParallel
    │       │     → stored_split::decompress_stored_parallel(data, writer, num_threads)
    │       │       [src/decompress/parallel/stored_split.rs]
    │       │       On NotStoredDominated → decompress_single_member_one_shot
    │       │
    │       └─ NO → falls through to one-shot paths below
    │
    ├─ isal_decompress::is_available()?  [src/backends/isal_decompress.rs]
    │   YES → DecodePath::IsalSingle
    │     → isal_decompress::decompress_gzip_stream(data, writer)
    │       [src/backends/isal_decompress.rs]
    │       (x86_64 only; ISA-L streaming inflate, one-shot)
    │
    ├─ data.len() > 1 GiB?   [mod.rs:193]
    │   YES → DecodePath::StreamingSingle
    │     → decompress_single_member_streaming(data, writer)
    │       [src/decompress/mod.rs:471 — flate2/zlib-ng, 1 MB streaming buffer]
    │
    └─ default → DecodePath::LibdeflateSingle
          → decompress_single_member_libdeflate(data, writer)
            [src/decompress/mod.rs:494 — libdeflate one-shot, ISIZE-hint sizing]
```

### Key constants (all in `src/decompress/mod.rs`)

| Constant | Value | Line |
|---|---|---|
| `MIN_PARALLEL_COMPRESSED` | 10 MiB | 92 |
| `MIN_PARALLEL_SM_THREADS` | 4 | 109 |
| `PARALLEL_SM_MIN_RATIO_NUM/DEN` | 1.15 | 133–134 |
| `STREAM_BUFFER_SIZE` | 1 MiB | 31 |

### Notes on CLAUDE.md vs code

- CLAUDE.md does not document `DecodePath::StoredParallel`. It exists
  (`src/decompress/mod.rs:80`) and handles stored-block-dominated (incompressible)
  single-member streams via a non-speculative parallel split
  (`src/decompress/parallel/stored_split.rs`). It is gated by the same
  `parallel_sm` cfg and `MIN_PARALLEL_COMPRESSED` floor as `IsalParallelSM` but
  is reached only after the `parallel_sm_unprofitable` ratio guard rejects the
  speculative pipeline.

- CLAUDE.md says `data > 1 GiB (no ISA-L) → StreamingSingle`. The code confirms
  exactly this: ISA-L check fires first (`mod.rs:190`); the `> 1 GiB` check fires
  only if `is_available()` returns false (`mod.rs:193`).

- CLAUDE.md says `MultiMemberPar` falls back on error; the code confirms this at
  `mod.rs:271–273`. The fallback is a deliberate router correction (false gzip
  header sequences in stored-block data), not a silent backend retry.

- `decompress_single_member` (`mod.rs:320`) is a second dispatcher used by the I/O
  layer for the single-member path. It mirrors `classify_gzip` and handles the
  case where a multi-member or BGZF file reaches it at T=1 (or on small files)
  by routing to `decompress_multi_member_sequential` rather than erroring. See
  `mod.rs:344`.

---

## 2. Compression routing tree

Entry point: `compress_with_pipeline` (`src/compress/mod.rs:30`).
Multi-threaded path dispatches through `SimpleOptimizer`
(`src/compress/simple.rs:40`).

```
Input (reader, args, opt_config)
│
├─ args.use_zopfli()?   [L11 or --zopfli flags]
│   YES → ZopfliGzEncoder::compress [src/compress/zopfli.rs]
│         (pure-Rust zopfli; single-member output)
│
├─ T=1 AND level ≤ 9
│   │
│   ├─ level ≤ 3 AND ISA-L available AND NOT --huffman/--rle
│   │   → isal_compress::compress_gzip_stream_direct(reader, writer, level)
│   │     [src/backends/isal_compress.rs]
│   │     (x86_64 + AVX2; single-member output)
│   │
│   ├─ level 1–5 AND NOT --huffman/--rle
│   │   ratio probe on first 64 KiB:
│   │   ├─ compressible (ratio ≥ 0.10) → libdeflate one-shot
│   │   │     compress::parallel::compress_single_member [src/compress/parallel.rs]
│   │   │     (standard single-member gzip output)
│   │   └─ highly compressible (< 0.10) → flate2/zlib-ng streaming
│   │         [src/compress/mod.rs:128–133; standard single-member gzip]
│   │
│   └─ level 6–9 (or --huffman/--rle)
│       → flate2/zlib-ng streaming encoder  [src/compress/mod.rs:135–160]
│         (standard single-member gzip output)
│
└─ T>1 → SimpleOptimizer::compress [src/compress/simple.rs:40]
    │
    ├─ level ≥ 10  → ParallelGzEncoder [src/compress/parallel.rs]
    │                ("GZ" FEXTRA subfield multi-block format)
    │
    ├─ level 6–9   → PipelinedGzEncoder [src/compress/pipelined.rs]
    │                (single-member gzip-compatible output, dictionary sharing)
    │
    └─ level 1–5   → ParallelGzEncoder [src/compress/parallel.rs]
                     ("GZ" FEXTRA subfield multi-block format)
```

### Format notes

- `ParallelGzEncoder` output carries a `"GZ"` FEXTRA subfield with per-block size
  info (`src/compress/parallel.rs`). Decompression routes this to
  `bgzf::decompress_bgzf_parallel` via `has_bgzf_markers`.
- `PipelinedGzEncoder` and all T=1 paths emit plain single-member gzip — standard
  format, no FEXTRA modification.
- CLAUDE.md describes T>1 routing as `L6–9 → Pipelined`, `L0–5 → Parallel`. The
  code confirms this but shows `SimpleOptimizer` checks `>= 6` not `>= 7`
  (`simple.rs:75`). Level 6 is included in the Pipelined branch for T>1.

---

## 3. The production decode path — IsalParallelSM

### What runs

```
decompress_parallel          single_member.rs:126
  → read_parallel_sm         sm_driver.rs:1
    → chunk_fetcher::drive   chunk_fetcher.rs (default)
       or
      chunk_fetcher::drive_clean_window_oracle   (GZIPPY_CLEAN_WINDOW_ORACLE only)
```

Inner decoder per build:
- `cfg(pure_inflate_decode)` (feature `pure-rust-inflate` on x86_64 or aarch64):
  `inflate_wrapper::IsalInflateWrapper` → `ResumableInflate2`
  (`src/decompress/parallel/inflate_wrapper.rs`)
- `cfg(all(feature="isal-compression", not(feature="pure-rust-inflate"), target_arch="x86_64"))`:
  ISA-L C FFI path (patched ISA-L dynamic-table decode)

### Build flags

| Flag | Effect |
|---|---|
| `--no-default-features --features pure-rust-inflate` | Pure-Rust inflate, no ISA-L C in decode graph; enables `parallel_sm` + `pure_inflate_decode` on x86_64/aarch64 |
| `--features isal-compression` | ISA-L C library in the decode graph; enables `parallel_sm` on x86_64 |
| `--features pure-rust-inflate` (on aarch64) | Enables the parallel-SM pipeline on arm64; ISA-L C not required |

`cfg(parallel_sm)` is emitted by `build.rs:80` when:
- x86_64 + (`isal-compression` OR `pure-rust-inflate`), OR
- aarch64 + `pure-rust-inflate`

`cfg(pure_inflate_decode)` is emitted by `build.rs:83` when:
- (x86_64 OR aarch64) + `pure-rust-inflate`

Source: `build.rs:61–85`.

### Asserting the production path

```bash
# Confirm the routing
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null
# → stderr: [gzippy] path=IsalParallelSM threads=N bytes=M

# Exercise at every thread count (bypasses the T≥4 floor)
GZIPPY_FORCE_PARALLEL_SM=1 gzippy -d -c -p1 testfile.gz > /dev/null

# Confirm the counter incremented (proves parallel pipeline ran, not libdeflate)
# MARKER_PIPELINE_RUNS: src/decompress/parallel/single_member.rs:100
```

### Production thread floor

`MIN_PARALLEL_SM_THREADS = 4` (`src/decompress/mod.rs:109`).

T=1 single-member always routes to `IsalSingle` or `LibdeflateSingle` in production.
The parallel-SM engine beats the libdeflate one-shot only at T≥4
(2026-05-29: T1 one-shot 1074 MB/s vs T2 parallel 863 MB/s — a loss).

`GZIPPY_FORCE_PARALLEL_SM=1` sets the effective floor to 0
(`src/decompress/mod.rs:114`) for measurement only; it is not a production setting.

---

## 4. Flag table

Sourced by grepping `env::var` across `src/`. All 33 flags listed; cite file:line
is the first `env::var*` read site. All flags are byte-identical to their unset
state unless noted otherwise.

| FLAG | EFFECT | CLASS |
|---|---|---|
| `GZIPPY_DEBUG` | Print `[gzippy] path=… threads=… bytes=…` and other routing decisions to stderr. Cached once via `utils::debug_enabled()`. | PRODUCTION-AFFECTING (routing visibility) |
| `GZIPPY_FORCE_PARALLEL_SM` | Drop the `MIN_PARALLEL_SM_THREADS=4` floor to 0, making the speculative parallel-SM pipeline engage at any thread count including T=1. Measurement aid; regresses T1–T3. | PRODUCTION-AFFECTING (routing) |
| `GZIPPY_CHUNK_KIB` | Override the default 4 MiB compressed-chunk target with `N KiB`. Allows granularity sweeps without rebuilding (T=16 straggler vs HT regression discrimination). | PRODUCTION-AFFECTING (throughput tuning) |
| `GZIPPY_MMAP_OUTPUT` | When decompressing to a file, pre-allocate the output file at ISIZE bytes and decode directly into a mmap'd region; kernel writeback handles disk I/O. Falls back to BufWriter if ISIZE is not extractable. Disabled for multi-member. | PRODUCTION-AFFECTING (I/O mode) |
| `GZIPPY_LEGACY_INFLATE` | Roll back the unified `Inflate<>` hot loop to the prior `ResumableInflate2` path (kill switch for sub-second rollback). When set, `legacy_kill_switch_active()` returns true and the new monomorphised path is skipped. | PRODUCTION-AFFECTING (decode path) |
| `GZIPPY_ISAL_PURE_BULK` | Route post-marker clean finish through `finish_decode_chunk_bulk_lut` (`isal_lut_bulk::decode_block`) instead of `IsalInflateWrapper`. Only with `cfg(pure_inflate_decode)`. Default **ON**; `=0` disables. | PRODUCTION-AFFECTING (inner decoder) |
| `GZIPPY_OPTION_A_PREFILL` | Toggle Option A3+A4 pre-fill (predecessor window image written into `chunk.data[0..32K]` so back-references hit the AVX2 fast path). `=1` enables, `=0` disables. Measured +4.2% T=16 silesia. | PRODUCTION-AFFECTING (throughput) |
| `GZIPPY_HUGEPAGE` | Apply `MADV_HUGEPAGE` to freshly-allocated chunk buffers (Linux only). Checked once via OnceLock at first allocation miss. | PRODUCTION-AFFECTING (memory) |
| `GZIPPY_SLAB_ALLOC` | Route huge allocations (≥ 3 MiB) through `SlabAlloc` — a resident free-list that reuses blocks instead of munmapping, avoiding first-touch page faults on every reuse. The T4–T16 page-fault fix lever. | PRODUCTION-AFFECTING (allocator) |
| `GZIPPY_SLAB_CAP` | Max resident free blocks retained by `SlabAlloc` (integer). Controls the retain–release tradeoff (over-retaining caused TLB regression when set to 32). | PRODUCTION-AFFECTING (allocator tuning) |
| `GZIPPY_NO_PREFETCH` | Disable the speculative prefetch of the in-order frontier chunk. When absent (the default), prefetching is enabled. Measurement lever. | PRODUCTION-AFFECTING (prefetch) |
| `GZIPPY_BURST_PREFETCH` | Raise the in-flight prefetch gate from `pool_size` to `pool_size * 2` (doubles the max parallel prefetch depth). | PRODUCTION-AFFECTING (prefetch) |
| `GZIPPY_EAGER_POSTPROC` | Extra full-cache post-process scan on every marker wait (diagnostic duplicate of production resolve-ahead). Off by default. | PRODUCTION-AFFECTING (scheduling probe) |
| `GZIPPY_VERBOSE` | Print `BlockFetcher` statistics (cache hits/misses, speculation failures, chunk sizes, timing counters) to stderr after decode. Mirror of rapidgzip's `--verbose` destructor dump. | INSTRUMENTATION |
| `GZIPPY_DEBUG` (also in bgzf/parallel) | Same flag read in multiple modules (`bgzf.rs:3431`, `parallel/single_member.rs:114`, `compress/parallel.rs:555`). Effect is additional per-operation debug prints. | INSTRUMENTATION |
| `GZIPPY_TRACE` | Enable bgzf decode tracing. Cached once in `bgzf.rs` via `OnceLock`. | INSTRUMENTATION |
| `GZIPPY_LOG_FILE` | Write the parallel-SM structured event log (v1 trace) to `<path>`. Consumed by `scripts/parallel_sm_log_summary.py`. | INSTRUMENTATION |
| `GZIPPY_TIMELINE` | Write the parallel-SM v2 timeline trace to `<path>`. | INSTRUMENTATION |
| `GZIPPY_RPMALLOC_STATS` | Print rpmalloc global statistics (mapped_total / unmapped_total / cached / huge_alloc_peak) after decode. Requires `--features rpmalloc-stats`. | INSTRUMENTATION |
| `GZIPPY_BODY_FAIL_LOG` | When a speculative chunk body fails to decode, append the failure record to `<path>` for post-mortem analysis. | INSTRUMENTATION |
| `GZIPPY_ORACLE_TRACE` | Print per-chunk span summary lines (`ORACLE_SPAN i= start_bit= …`) to stderr during `drive_clean_window_oracle`. | INSTRUMENTATION |
| `GZIPPY_STORED_PHASE_TIMING` | Print per-phase wall times for the stored-split parallel decoder to stderr. | INSTRUMENTATION |
| `GZIPPY_SLOW_BOOTSTRAP` | Causal probe: spin `N%` of the bootstrap duration after bootstrap completes, making it appear `(1 + N/100)×` slower. Used to test whether the window-absent bootstrap is on the wall critical path. Value = integer percent. Byte-identical output (pure delay). | ORACLE / MEASUREMENT |
| `GZIPPY_SLOW_BOOTSTRAP_SLEEP` | Frequency-neutral variant of `GZIPPY_SLOW_BOOTSTRAP`: yields the core via `sleep` instead of a busy-spin. Eliminates the all-core turbo-depression confound from the spin probe. Requires `GZIPPY_SLOW_BOOTSTRAP=N` to set the delay quantum. | ORACLE / MEASUREMENT |
| `GZIPPY_CLEAN_WINDOW_ORACLE` | Replace `chunk_fetcher::drive` with `drive_clean_window_oracle`: decode every chunk with its true predecessor window (no speculation, no marker bootstrap, no marker pipeline). CRC/ISIZE still verified. Sizes the marker pipeline's contribution to the rapidgzip gap. **CAUTION: this oracle was found broken in commit `64eb6df` (silently re-ran the full bootstrap); validate the instrument before trusting its numbers.** | ORACLE / MEASUREMENT |
| `GZIPPY_BYPASS_CAPTURE` | Path to a file. Run a normal decode and serialize every decode result (`(start_bit, stop_bit) → ChunkData`) to this file for later replay. | ORACLE / MEASUREMENT |
| `GZIPPY_BYPASS_DECODE` | Path to a file written by `GZIPPY_BYPASS_CAPTURE`. Workers look up precomputed `ChunkData` by `(start_bit, stop_hint_bit)` and reconstruct via memcpy instead of running Huffman decode. Isolates coordination overhead from inner-decode CPU. Output is byte-identical when all chunks hit the map. | ORACLE / MEASUREMENT |
| `GZIPPY_BYPASS_FORCE_CLEAN` | Requires `GZIPPY_BYPASS_DECODE`. Strip `data_with_markers` from every replayed chunk (fold into clean `data`), bypassing marker/apply_window/narrow coordination. Delta vs default replay = marker-coordination cost. | ORACLE / MEASUREMENT |
| `GZIPPY_BYPASS_META_ONLY` | Requires `GZIPPY_BYPASS_DECODE`. Return only metadata (size/boundaries) without reconstructing `data_with_markers`; worker output is a zero-filled buffer. Isolates metadata-only coordination overhead. | ORACLE / MEASUREMENT |
| `GZIPPY_BYPASS_REBUILD` | Requires `GZIPPY_BYPASS_DECODE`. Force per-call `ChunkData` reconstruction (no table re-use across calls). | ORACLE / MEASUREMENT |
| `GZIPPY_SLEEP_DECODE_NS` | Requires `GZIPPY_BYPASS_DECODE`. Replace the Huffman decode with a fixed sleep of `N` nanoseconds and return a correct-SIZE, zero-filled, fully-clean `ChunkData`. Matches the rapidgzip sleep-patch for apples-to-apples coordination measurement. CRC/ISIZE verification is skipped in `sm_driver` when this is set. Output is garbage (zeros). | ORACLE / MEASUREMENT |
| `GZIPPY_STORED_NO_OVERLAP` | Force the stored-split parallel decoder to use sequential prefix→body order instead of overlapped execution. Control lever for the overlap A/B. | ORACLE / MEASUREMENT |
| `GZIPPY_STORED_INLINE_COPY` | Force the stored-split decoder to use the inline copy path at any thread count. | ORACLE / MEASUREMENT |
| `GZIPPY_REGEN_FIXTURES` | (Test-only) Regenerate zopfli golden fixtures instead of checking them. Read in `src/backends/zopfli_pure/tests.rs:58`. | TEST-ONLY |

**Total: 33 flags** (GZIPPY_DEBUG appears in multiple modules but is one logical flag).

---

## 5. What is NOT a production path

These modules or modes are compiled into the binary but are unreachable without an
explicit env-var or feature gate. They are measurement instruments and oracles, not
production paths.

- **`GZIPPY_CLEAN_WINDOW_ORACLE`** — `sm_driver.rs:45`: replaces the speculative
  pipeline with a known-window oracle; broken in one campaign run (commit `64eb6df`),
  retained for future validated use. Not a production decode path.

- **`GZIPPY_BYPASS_CAPTURE / _DECODE / _FORCE_CLEAN / _META_ONLY / _REBUILD`** —
  `src/decompress/parallel/decode_bypass.rs`: decode-bypass harness that captures
  and replays `ChunkData` to isolate coordination overhead from Huffman CPU. All
  five flags together form the bypass experiment harness; none are production.

- **`GZIPPY_SLEEP_DECODE_NS`** — `decode_bypass.rs:296`: fixed-sleep coordination
  isolation mode; produces garbage output (zeros); CRC verification disabled. Not
  a production path.

- **`GZIPPY_SLOW_BOOTSTRAP` / `_SLEEP`** — `gzip_chunk.rs:890/900`: causal probe
  injecting artificial delay into the window-absent bootstrap. Pure measurement;
  byte-identical but adds wall time by design.

- **`GZIPPY_ISAL_PURE_BULK`** — `gzip_chunk.rs:169`: the stateless ISA-L-LUT bulk
  decoder (`src/decompress/parallel/isal_lut_bulk.rs`). Pending neurotic
  confirmation before becoming the production default.

- **`GZIPPY_STORED_NO_OVERLAP` / `_INLINE_COPY`** — `stored_split.rs`: overlap and
  copy-strategy control levers for the stored-split parallel decoder; measurement
  only.

- **`GZIPPY_MMAP_OUTPUT`** — `src/decompress/io.rs:160`: experimental mmap output
  path for single-member file decode. Not the default I/O mode.

- **`GZIPPY_OPTION_A_PREFILL`** — `gzip_chunk.rs:188`: Option A3+A4 pre-fill;
  default state depends on the `OnceLock` logic and may be ON by default in some
  builds (measured +4.2% T=16). Check `gzip_chunk.rs:188–195` for the exact
  current default before treating this as inactive.

- **`global-rpmalloc` feature** — `#[global_allocator] = rpmalloc::RpMalloc` in
  `src/main.rs`. FALSIFIED at +167% wall (+41% faults) on file output
  (campaign 2026-05-28). Not in default feature set.

- **`GZIPPY_LEGACY_INFLATE`** — kill switch back to `ResumableInflate2`; the new
  `Inflate<>` path is the production default when the feature is active.
