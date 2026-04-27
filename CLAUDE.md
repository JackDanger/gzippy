# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before `make ship`, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — `make ship` (homelab bench on `neurotic`) is authoritative; local `make` is for iteration.
4. **REVERT REGRESSIONS** — if `make` or `make ship` shows a loss, revert immediately.
5. **NEVER COMPROMISE PERFORMANCE** — clippy, style, readability: none justify slower code.

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
              (rapidgzip-style speculation + ISA-L inflatePrime; v0.3.0)
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

Regression tests that lock in the parallel single-member wiring:
`decompress::parallel::single_member::tests::test_parallel_path_no_silent_fallback`
and `tests::routing::tests::test_single_member_routing_multithread`.

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

**v0.3.0 — parallel single-member**: ISA-L `inflatePrime` (rapidgzip's pattern from
`isal.hpp`) re-decodes confirmed chunks at non-byte-aligned bit offsets at full
ISA-L SIMD speed (~1500 MB/s/thread), replacing the prior pure-Rust marker decoder
(22 MB/s/thread). Wired into `decompress::decompress_single_member` behind
`isal_decompress::is_available() && num_threads > 1 && data.len() > 10 MiB`.

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
| `src/decompress/parallel/single_member.rs` | v0.3.0 parallel SM — ISA-L `inflatePrime` |
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
