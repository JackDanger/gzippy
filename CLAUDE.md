# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before cloud fleet, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — cloud fleet is authoritative, local `make` is for iteration.
4. **REVERT REGRESSIONS** — if `make` or cloud fleet shows a loss, revert immediately.
5. **NEVER COMPROMISE PERFORMANCE** — clippy, style, readability: none justify slower code.

## Production Decompression Routing (Feb 2026)

```
Input → decompression.rs: decompress_file / decompress_stdin
  ├─ BGZF? ("GZ" extra field)
  │     → bgzf::decompress_bgzf_parallel (libdeflate FFI, T1 or Tmax)
  ├─ Multi-member? (trailing gzip headers detected)
  │     T1  → decompress_multi_member_sequential (libdeflate, member-by-member)
  │     Tmax → bgzf::decompress_multi_member_parallel (libdeflate FFI, parallel)
  └─ Single-member?
        x86_64 Tmax (ISA-L available, data ≥ 4MB) → parallel_single_member::decompress_parallel
        x86_64 T1  → isal_decompress::decompress_gzip_stream (ISA-L direct FFI)
        x86_64 fallback → decompress_single_member_libdeflate
        arm64 compressible (ISIZE/len ≥ 2.0) → parallel_single_member (MarkerDecoder)
        arm64 incompressible (ISIZE/len < 2.0) → decompress_single_member_libdeflate
        arm64 huge (data > 1GB) → decompress_single_member_streaming (avoid huge alloc)
```

**Compression (L6 with ≥2 threads)**: `PipelinedGzEncoder` → produces **single-member** output
(not BGZF). Decompress routes to single-member path, not BGZF.

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

**arm64 single-member routing** (compressibility check via ISIZE/len ratio):
- ISIZE/len ≥ 2.0 (compressible): parallel_single_member (MarkerDecoder, 1500+ MB/s on real workloads)
- ISIZE/len < 2.0 (incompressible/random): libdeflate sequential (near-parity, safe)
- data > 1GB: streaming zlib-ng (avoids huge allocation only for pathological files)
ISA-L is unavailable on arm64. Do NOT remove the compressibility check.

**Parallel speculation guard**: `parallel_single_member` is gated on
`isal_decompress::is_available()` — x86_64 only. Never remove this guard
without measuring on both architectures.

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
| `src/bgzf.rs` | BGZF/multi-member parallel (8400 lines, core engine) |
| `src/isal_decompress.rs` | ISA-L streaming inflate (x86_64) |
| `src/parallel_single_member.rs` | Speculative parallel (x86_64 only, gated on ISA-L) |
| `src/compression.rs` | Compression entry |
| `src/parallel_compress.rs` | Parallel BGZF compression |
| `src/pipelined_compress.rs` | L6-L9 Tmax → single-member output |
| `src/consume_first_decode.rs` | Experimental pure Rust inflate (NOT production) |
