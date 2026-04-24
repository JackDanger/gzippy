# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before cloud fleet, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — cloud fleet is authoritative, local `make` is for iteration.
4. **REVERT REGRESSIONS** — if `make` or cloud fleet shows a loss, revert immediately.
5. **NEVER COMPROMISE PERFORMANCE** — clippy, style, readability: none justify slower code.

## Production Routing (Apr 2026)

### Decompression

```
Input → decompression.rs: decompress_gzip_libdeflate
  ├─ gzippy-parallel? ("GZ" subfield in FEXTRA)
  │     → bgzf::decompress_bgzf_parallel (libdeflate FFI, T1 or Tmax internally)
  ├─ Multi-member? (trailing gzip headers detected)
  │     T1  → decompress_multi_member_sequential (libdeflate, member-by-member)
  │     Tmax → bgzf::decompress_multi_member_parallel (libdeflate FFI)
  └─ Single-member?
        x86_64 (ISA-L available) → isal_decompress::decompress_gzip_stream
        any arch, data > 1GB (no ISA-L) → decompress_single_member_streaming (zlib-ng)
        default → decompress_single_member_libdeflate
```

### Compression

```
T1, L0–L3, ISA-L available → isal_compress::compress_gzip_to_writer
T>1, L6–L9               → pipelined_compress::PipelinedGzEncoder → single-member output
T>1, L0–L5               → parallel_compress::ParallelGzEncoder  → "GZ" subfield multi-block
T1, all other            → flate2 single-threaded
```

**"GZ" subfield**: gzippy's own parallel format (not standard BGZF). Files produced by
`ParallelGzEncoder` carry a "GZ" FEXTRA subfield with per-block size info; decompression
routes them to `bgzf::decompress_bgzf_parallel`. `PipelinedGzEncoder` output is plain
single-member — decompresses on the single-member path.

## Experiments (not yet wired into production)

- `parallel_single_member.rs` — speculative parallel for single-member files.
  Measured at 88–148 MB/s vs 600–2000 MB/s sequential. **Not wired in.**
- `speculative_parallel.rs` — RapidGzip-style u16 marker approach.
  Not declared in `main.rs`; not compiled. Preserved for future experimentation.

To wire in an experiment: add it to `decompress_single_member` in `decompression.rs`
behind a size gate, verify with `make route-check`, then run `make quick`.

## Score: 47W / 13L (Feb 21 2026, cloud fleet)

| Category | Losses | Gap | Actionability |
|----------|--------|-----|---------------|
| T1 decompress near-parity | 2 | <2% | Noise |
| Tmax single-member parallel | 8 | -23% to -40% | Pipeline architecture |
| L1 T1 compress vs igzip | 2 | -63% to -74% | AVX-512 assembly |
| arm64 L1 Tmax compress | 1 | -3.3% | madvise or block tuning |

## Hard-Won Lessons

**What works**: mmap stdin for multi-threaded (zero-copy, +44%), BufWriter for
stdout, direct FFI, BGZF parallel, 1MB streaming buffer, lock-free parallel.

**What doesn't**: mmap for single-threaded (4x slower from page faults!),
larger blocks for L1 (no help), **speculative parallel decode on arm64**
(16x slower on low-redundancy data — block boundaries are rare, most chunks
become all-marker forcing huge sequential re-decodes),
two-pass scan-then-decode, large pre-allocations.

**arm64 single-member**: currently falls through to libdeflate one-shot (fast enough).
Streaming path only for files > 1GB. ISA-L is unavailable on arm64.

**Parallel speculation guard** (future work): if `parallel_single_member` is ever wired in,
it must be gated on `isal_decompress::is_available()` — x86_64 only. Speculative parallel
on arm64 was 16× slower on low-redundancy data (block boundaries rare, most chunks become
all-marker forcing huge sequential re-decodes).

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
source .env && make ship
```

`make route-check` — generates 1MB+10MB test files and shows routing + timing
vs pigz for all four combos (T1/T4 × 1MB/10MB). Use this before ANY decompression change.

## Key Files

| File | Role |
|------|------|
| `src/decompression.rs` | Decompression entry, format detect, routing |
| `src/bgzf.rs` | gzippy-parallel + multi-member parallel (~8400 lines, core engine) |
| `src/isal_decompress.rs` | ISA-L streaming inflate (x86_64 production path) |
| `src/libdeflate_ext.rs` | libdeflate FFI — gzip_decompress_ex (arm64 + fallback) |
| `src/compression.rs` | Compression entry, routing |
| `src/parallel_compress.rs` | ParallelGzEncoder — T>1 L0–L5, "GZ" multi-block output |
| `src/pipelined_compress.rs` | PipelinedGzEncoder — T>1 L6–L9, single-member output |
| `src/isal_compress.rs` | ISA-L compression (x86_64 T1 L0–L3) |
| `src/parallel_single_member.rs` | EXPERIMENT: speculative parallel (not wired in, 88–148 MB/s) |
| `src/speculative_parallel.rs` | EXPERIMENT: RapidGzip u16-marker approach (not in main.rs) |
| `src/consume_first_decode.rs` | EXPERIMENT: pure Rust inflate (not production) |
