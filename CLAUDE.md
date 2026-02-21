# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **BENCHMARK EVERYTHING** — cloud fleet is authoritative, local is for iteration.
3. **REVERT REGRESSIONS** — if performance drops, revert immediately.
4. **NEVER COMPROMISE PERFORMANCE** — clippy, style, readability: none justify slower code.

## Production Decompression Routing (Feb 2026)

```
Input → decompression.rs
  ├─ BGZF?          → bgzf::decompress_bgzf_parallel (libdeflate FFI, parallel)
  ├─ Multi-member?   → bgzf::decompress_multi_member_parallel (libdeflate FFI)
  ├─ Single x86_64?  → isal_decompress::decompress_gzip_stream (ISA-L direct FFI)
  └─ Single arm64?   → decompress_single_member_libdeflate (libdeflate FFI)
```

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
larger blocks for L1 (no help), speculative parallel decode,
two-pass scan-then-decode, large pre-allocations.

## Workflow

```bash
cargo test --release                      # Correctness (always use timeouts)
gzippy-dev cloud bench                    # Authoritative cloud fleet numbers
```

Never claim a win/loss based on local results. Cloud fleet on both x86_64 and
arm64 with RAM-backed I/O is the only source of truth.

## Key Files

| File | Role |
|------|------|
| `src/decompression.rs` | Decompression entry, format detect, routing |
| `src/bgzf.rs` | BGZF/multi-member parallel (8400 lines, core engine) |
| `src/isal_decompress.rs` | ISA-L streaming inflate (x86_64) |
| `src/compression.rs` | Compression entry |
| `src/parallel_compress.rs` | Parallel BGZF compression |
| `src/consume_first_decode.rs` | Experimental pure Rust inflate (NOT production) |
