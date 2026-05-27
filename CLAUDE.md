# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Cutover Goal (May 2026)

**Port the rapidgzip algorithmic core into Rust, faithfully — but ONLY for
formats GNU gzip supports.** We want rapidgzip's *speed*, not its *format
breadth*. GNU gzip handles gzip-family streams (single-member, multi-member,
BGZF — all valid gzip). BZIP2 and ZLIB are NOT gzip and are out of scope.

The goal is **every gzip-relevant primitive, decoder, block finder, huffman
variant, reader, index, and analyzer in `vendor/rapidgzip/librapidarchive/`
ported into gzippy with vendor file:line citations.** That includes:

- **Formats** (IN SCOPE): gzip single-member, multi-member gzip, BGZF, raw
  DEFLATE (as the inner codec).
- **Formats** (OUT OF SCOPE — not part of GNU gzip): BZIP2 (`Bzip2Chunk.hpp`,
  `indexed_bzip2/bzip2.hpp`), ZLIB stream format (`gzip/zlib.hpp`). A small
  ZLIB-format header parser landed (commit `db19347`) as a side-effect of the
  port pass; keep it for now since it's harmless, but no Bzip2 or Zlib
  decoder is in scope.
- **Decoders** (gzip-only): `chunkdecoding/GzipChunk`, `gzip/GzipReader`,
  `gzip/GzipAnalyzer`, `gzip/InflateWrapper`, `gzip/isal`. NOT
  `chunkdecoding/Bzip2Chunk`.
- **Block finders**: `blockfinder/Bgzf`, `blockfinder/DynamicHuffman`,
  `blockfinder/Uncompressed`, `blockfinder/PigzStringView`,
  `blockfinder/precodecheck/CountAllocatedLeaves` (all gzip-family).
- **Huffman variants**: `HuffmanCodingDoubleLiteralCached`, `HuffmanCodingISAL`,
  `HuffmanCodingReversedBitsCached*`, `HuffmanCodingShortBitsMultiCached*`,
  `HuffmanCodingDistanceISAL` (gzip Deflate decoders).
- **Core primitives**: `ThreadPool` (the real one), `BitStringFinder`,
  `ParallelBitStringFinder`, `StreamedResults`, `SimpleRunLengthEncoding`,
  `AtomicMutex`, `AffinityHelpers`, `AlignedAllocator`, `FasterVector`,
  `FileRanges`, `JoiningThread`, etc.
- **High-level**: `ParallelGzipReader`, `IndexFileFormat` (seekable indexes
  for gzip), gzip-relevant `rapidgzip.hpp` public API surface.

Done when an Opus advisor agrees gzippy structurally and calculationally
matches the gzip-relevant surface of rapidgzip (not the BZIP2/ZLIB-decoder
surface), AND the inner Huffman decode primitives are vendor-competitive
on neurotic. Structural port and perf parity are now BOTH required —
"performance optimization later" is rescinded as a guiding principle
when it blocks vendor-competitive throughput.

## Permission to fully reimplement the inner inflate

The "port faithfully, don't innovate" rule is **scoped to architecture
and high-level shape** (chunk pipeline, prefetcher, block finder, etc.).
For the inner Huffman decode loop — `decode_huffman_body_resumable` and
the `LitLenTable` / `DistTable` / `Bits` primitives — full
re-implementation of every libdeflate / ISA-L technique is in scope and
explicitly authorized, including:

- Multi-literal lookahead (2-/3-/4-literal packed-write paths).
- Fixed-Huffman static-table specialization.
- BMI2 PEXT / BZHI runtime dispatch.
- Table prefetch (`_mm_prefetch`) ahead of dependent loads.
- Inline-asm hot loops if needed to match vendor codegen.
- Reorganized state-machine that elides the resumable yield-check tax
  when output has FASTLOOP_OUTPUT_MARGIN bytes of headroom.

Prior falsifications (e.g. commit `ca52389` SIMD multi-literal regression)
are **NOT binding** — they were measured against the pre-PRELOAD,
pre-BMI2 hot loop. Re-attempt any of them with fresh measurement.

### Update 2026-05-27 (later same day): innovation allowed in the inner loop

The "every change must have a vendor counterpart" requirement is
RESCINDED for the inner Huffman decode loop. Novel techniques that
have no vendor counterpart are now in scope provided they:

- Preserve correctness (all 635+ lib tests + corpus differential pass).
- Show measured win on the bench harness (no negative-results-allowed).
- Document the deviation in the commit message so the falsification
  record stays honest.

This still doesn't extend to architecture (chunk pipeline /
prefetcher / block finder / consumer) — those keep the vendor-port
rule. Only the inner Huffman loop and supporting primitives
(`LitLenTable`, `DistTable`, `Bits`, `bmi2.rs`) are open territory.

### Update 2026-05-27 (final): build the fastest possible raw Huffman decoder

Sharpening: the GOAL is *"build a provably-correct, fastest-possible
raw Huffman decoder"* (user directive). Implications:

- **Provably correct.** Strong testing — corpus differential against
  multiple independent oracles (flate2 + libdeflate), property-based
  testing with proptest, fuzzing if needed, plus the existing 635 lib
  tests. The decoder must produce byte-for-byte output identical to
  vendor on every legal input.
- **Fastest possible.** Beat ISA-L, libdeflate, AND flate2/zlib-ng on
  representative workloads. Not just "vendor-competitive" — strictly
  faster.
- **Arch-specific compilation is in scope.** Use `RUSTFLAGS="-C
  target-cpu=native"` for benches; build per-arch variants (BMI2 on
  x86_64, NEON on aarch64); per-CPU dispatch where it pays. Portable
  binaries can ship later via runtime dispatch; for the perf claim,
  arch-specific is the target.

The decoder is now decoupled (in principle) from the rest of the
parallel-SM machinery — it lives in `src/decompress/inflate/` and can
be evaluated as a standalone primitive. The resumable contract still
applies because callers in `parallel/` need it, but the contract's
cost should be amortized into the FASTLOOP and irrelevant to the
fastest-possible claim.

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before `make ship`, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — `make ship` (homelab bench on `neurotic`) is authoritative; local `make` is for iteration.
4. **NEVER COMPROMISE CORRECTNESS** — output bytes, CRC32, ISIZE must always verify.
5. **NO FALLBACKS** — failure is an explicit `Err(GzippyError::Decompression(_))`. No silent libdeflate or ISA-L retries from the SM body. `decompress_single_member` either succeeds via the parallel pipeline or returns an error.
6. **THE GOAL IS SPEED, NOT FIDELITY.** gzippy aims to be the fastest gzip implementation ever created. Rapidgzip is the current reference for parallel decompression because its architecture is the fastest known — port it as a *means*. Where gzippy's own SIMD inflate primitives (in `src/decompress/inflate/`, `src/decompress/{combined_lut,packed_lut,simd_huffman,vector_huffman,two_level_table}.rs`, `src/backends/inflate_bit.rs`) beat rapidgzip's inner inflate, use them. The deliverable is throughput, not vendor parity.

## Production Routing (Apr 2026)

### Decompression

```
Input → decompress::mod: decompress_gzip_libdeflate
  ├─ gzippy-parallel? ("GZ" subfield in FEXTRA)
  │     → bgzf::decompress_bgzf_parallel (libdeflate FFI, T1 or Tmax internally)
  ├─ Multi-member? (trailing gzip headers detected)
  │     T1  → decompress_multi_member_sequential (libdeflate, member-by-member)
  │     Tmax → bgzf::decompress_multi_member_parallel (libdeflate FFI)
  └─ Single-member?
        ISA-L + T>1 + compressed > 10 MiB
            → parallel::single_member::decompress_parallel
              (parallel chunk pipeline — see `src/decompress/parallel/`.
               Output STREAMS: bytes flow to the writer as each chunk
               resolves; CRC32 + ISIZE are verified against the gzip
               trailer AFTER the final chunk is written, so a mismatch
               is a terminal Err with partial output already on the
               writer — same as gzip(1); for file output
               `io::decompress_file` then deletes the dest file.
               Counter `MARKER_PIPELINE_RUNS` proves production routing
               called us; see the deletion-trap test in
               src/tests/routing.rs)
        x86_64 (ISA-L available)        → isal_decompress::decompress_gzip_stream
        any arch, data > 1 GiB (no ISA-L) → decompress_single_member_streaming (zlib-ng)
        default                          → decompress_single_member_libdeflate
```

### Compression

```
T1 L0–L3, ISA-L available → backends::isal_compress::compress_gzip_{to_writer,stream_direct}
T1 L1–L5                  → libdeflate one-shot (ratio probe) or flate2 streaming (zlib-ng)
T1 L6–L9                  → flate2 streaming (zlib-ng)
T>1 L6–L9                 → compress::pipelined::PipelinedGzEncoder → single-member output
T>1 L0–L5                 → compress::parallel::ParallelGzEncoder  → "GZ" subfield multi-block
```

**"GZ" subfield**: gzippy's own parallel format (not standard BGZF). Files produced by
`ParallelGzEncoder` carry a "GZ" FEXTRA subfield with per-block size info; decompression
routes them to `bgzf::decompress_bgzf_parallel`. `PipelinedGzEncoder` output is plain
single-member — decompresses on the single-member path.

## Optimization Branches

There is no `src/experiments/`. Every module on `main` is reachable from a
production code path or is a test fixture / supportive script.

To prototype a new path: add the module under the relevant subsystem
(`src/decompress/`, `src/compress/`, etc.), wire a feature-gated or size-gated
call site in the routing table above, and add a strict correctness test (no
silent fallback). When `make ship` confirms the win, lift the gate. When
abandoned, delete the module — `main` does not host dead code.

Two regression tests lock the parallel single-member wiring:

1. `tests::routing::tests::test_single_member_routing_multithread` — runs
   `decompress_single_member(T=4)` on a 24 MiB input and asserts byte-perfect
   output. Covers "parallel path takes the input" and "falls back correctly
   when speculation fails on adversarial chunks."

2. `tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
   — runs the same fixture at T=1 (sequential) and T=4 (parallel) and asserts
   `parallel_elapsed` stays within a small multiple of `sequential_elapsed`.
   This is the local-CI guard against a parallel-slower-than-sequential
   regression.

## Active port: rapidgzip → gzippy parallel single-member

The parallel single-member path under `src/decompress/parallel/` is an
in-progress port of rapidgzip's chunked-decode architecture. Vendor
source is in `vendor/rapidgzip/librapidarchive/`; port modules cite
vendor `file:line` in their doc comments — treat those citations as the
reference when a change appears to "work" but looks structurally off.

Capture a trace with `GZIPPY_LOG_FILE=/tmp/sm.log` and analyze via
`scripts/parallel_sm_log_summary.py` before claiming a perf change worked.

## Hard-Won Lessons

**What works**: mmap stdin for multi-threaded (zero-copy, +44%), BufWriter for
stdout, direct FFI, BGZF parallel, 1MB streaming buffer, lock-free parallel.

**What doesn't**: mmap for single-threaded (4x slower from page faults!),
larger blocks for L1 (no help), **speculative parallel decode on arm64**
(16x slower on low-redundancy data — block boundaries are rare, most chunks
become all-marker forcing huge sequential re-decodes),
two-pass scan-then-decode, large pre-allocations.

**arm64 single-member**: currently falls through to libdeflate one-shot (fast enough).
Streaming path only for files > 1 GiB. ISA-L is unavailable on arm64; the parallel
single-member path is gated on `isal_decompress::is_available()` so arm64 never
takes it.

## Branch and PR Workflow

**main is protected** — no direct pushes. Every change goes through a PR.
The pre-push hook (auto-installed by `cargo build`) enforces this locally.

```bash
# Start work
git checkout -b fix/my-change     # or feat/, refactor/, chore/, etc.

# Iterate (same as before)
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null
make
cargo test --release

# Ship
git push origin fix/my-change
gh pr create --fill               # title + body from commit messages
# CI runs → merge when green
```

Emergency bypass (never for performance-affecting changes):
```bash
git push --no-verify origin fix/my-change   # skip local hook only
```

Releasing:
```bash
git tag v0.X.Y && git push origin v0.X.Y   # triggers Release workflow
# Release workflow creates a formula-update PR and auto-merges it
```

## Iteration Loop

```bash
# 1. Make one focused change
# 2. Check routing
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null   # Shows which path is taken

# 3. Local sanity (30s) — catches catastrophic regressions
make

# 4. Correctness
cargo test --release

# 5. Authoritative numbers — only after make passes
make ship   # SSH to neurotic homelab, runs gzippy-dev bench
```

`make route-check` — generates 1MB+10MB test files and shows routing + timing
vs pigz for all four combos (T1/T4 × 1MB/10MB). Use this before ANY decompression change.

## Key Files

| File | Role |
|------|------|
| `src/decompress/mod.rs` | Decompression entry, format detect, routing |
| `src/decompress/bgzf.rs` | gzippy-parallel + multi-member parallel (core engine) |
| `src/decompress/scan_inflate.rs` | Streaming scan-and-inflate path |
| `src/decompress/parallel/single_member.rs` | Parallel single-member decode — entry point |
| `src/decompress/parallel/{block_finder,marker_decode,ultra_fast_inflate}.rs` | Speculation supporting primitives |
| `src/decompress/inflate/consume_first_decode.rs` | Pure-Rust inflate (production helpers used by `bgzf`, `scan_inflate`) |
| `src/decompress/inflate/{consume_first_table,jit_decode,libdeflate_decode,libdeflate_entry,specialized_decode,vector_huffman,double_literal,bmi2}.rs` | Huffman/inflate building blocks |
| `src/decompress/{combined_lut,inflate_tables,packed_lut,simd_copy,simd_huffman,two_level_table}.rs` | SIMD + LUT primitives shared with bgzf |
| `src/backends/isal_decompress.rs` | ISA-L streaming inflate (x86_64 production path) |
| `src/backends/inflate_bit.rs` | Universal inflate-from-bit (ISA-L on x86_64, libz-ng elsewhere) |
| `src/backends/libdeflate.rs` | libdeflate FFI — `gzip_decompress_ex` |
| `src/compress/mod.rs` | Compression entry, routing |
| `src/compress/parallel.rs` | ParallelGzEncoder — T>1 L0–L5, "GZ" multi-block output |
| `src/compress/pipelined.rs` | PipelinedGzEncoder — T>1 L6–L9, single-member output |
| `src/backends/isal_compress.rs` | ISA-L compression (x86_64 T1 L0–L3) |
