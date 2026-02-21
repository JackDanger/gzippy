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

## Score: 41W / 19L (Feb 2026, cloud fleet)

| Category | Losses | Gap | Actionability |
|----------|--------|-----|---------------|
| x86 T1 decompress | 4 | <1% | Noise — effectively won |
| Tmax single-member parallel | 8 | -18% to -41% | Major architecture needed |
| L1 T1 compress vs igzip | 2 | -62% to -73% | AVX-512 assembly |
| L1/L6 Tmax compress scaling | 5 | -10% to -39% | Thread scaling |

## Hard-Won Lessons

**What works**: Direct FFI (not wrapper crates), BGZF parallel, 1MB streaming
output buffer, ISIZE-based pre-allocation, lock-free parallel writes.

**What doesn't**: Speculative parallel decode, two-pass scan-then-decode,
deflate block finding, large pre-allocations (page faults), simulation benchmarks.

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
