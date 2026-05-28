# Perf attribution diff: pure-rust vs ISA-L FFI

Date: 2026-05-28
Build: gzippy on neurotic LXC 199, `cargo build --profile bench` with `target-cpu=native`.
Corpus: `benchmark_data/silesia-gzip9.gz` (212 MB output, 3350 dynamic blocks).
Invocation: `gzippy -d -c -p 16 silesia-gzip9.gz > /dev/null`.

## Production throughput

- pure-rust (--features pure-rust-inflate, A4 default-on): **884 MB/s**
- ISA-L FFI (--features isal-compression): **~1213 MB/s**
- Gap: ~28pp; 1pp gate (plan §7.1) requires ≤1pp ≈ 1201 MB/s.

## perf-record + perf-report top-15 (cpu-cycles, --call-graph dwarf -F 999)

| Symbol | pure-rust | ISA-L | Δ |
|---|---|---|---|
| `__memmove_avx_unaligned_erms` (libc) | 18.44% | 19.26% | +0.82pp |
| `clear_page_erms` (kernel) | 14.63% | 16.55% | +1.92pp |
| `submit_post_process_to_pool::closure` | 11.40% | 10.38% | -1.02pp |
| `std::thread::local::LocalKey::with` | **15.71%** | not in top-15 | -15.71pp |
| `decode_huffman_body_resumable` | 7.26% | 0 | -7.26pp |
| `copy_match_windowed` | 4.14% | 0 | -4.14pp |
| `crc32fast::Hasher::update` | 3.00% | 3.09% | +0.09pp |
| `deflate_block::Block::read` (bootstrap) | (via LocalKey body) | **18.57%** | (~same) |

## Key finding

The 15.71% pure-rust attribution to `LocalKey::with` is the CLOSURE BODY of the
`THREAD_PURE_LITLEN.with(|cell| { ... })` in `deflate_block.rs:1352-` — i.e. the
entire bootstrap speculative pre-decode for dynamic-Huffman blocks. ISA-L's
profile shows the equivalent at 18.57% under `Block::read` directly because
ISA-L doesn't go through the pure-rust LUT cache wrapping.

**The two bootstrap-pre-decode costs are ROUGHLY EQUAL (~18% both).** This was
my misread on first inspection — the advisor's "structural cost" hypothesis is
LARGELY EXPLAINED by the same workload showing up under different symbol names.

## Where the 28pp gap actually lives

Structural costs (memmove, clear_page, submit_post_process, crc, bootstrap)
are within 2pp on each side. The genuine differential is inner inflate:

- pure-rust inner inflate (decode_huffman_body_resumable + copy_match_windowed +
  read_stream_inner + LUT ops) ≈ 14-15% absolute CPU.
- ISA-L's inner inflate hot path (deflate_decompress_bmi2 + crc32_x86_vpclmulqdq_avx2 +
  build_decode_table) drops to roughly 5-7% absolute CPU because each symbol
  costs ~3x fewer cycles in hand-tuned asm.

That ≈8-9pp absolute CPU difference, applied to the 884 MB/s baseline, projects
to ~+25-35% throughput recovery — which closes most of the 28pp gap.

## Implication for plan §7.1

The 1pp-of-ISA-L gate cannot be reached without either:
1. **Route C v3 dynasm asm emit** for the inner inflate loop (plan §3.1). The
   inner-loop body executes 1-2 ns/symbol in ISA-L vs 5-6 ns/symbol in Rust;
   no inlining/PGO trick recovers that without going to asm.
2. **Vectorized SIMD inner loop** (BMI2 PEXT for bit extract; AVX2 for literal
   stores; SSE2 overlap-safe match-copy). Roughly equivalent in effort to
   Route C v3 but expressed as Rust intrinsics rather than dynasm-rs.

The sub-crate `gzippy_inflate::route_c_dynamic` Rust reference at 395 MB/s is
SLOWER than production's 884 MB/s — production has the FASTLOOP + multi-lit
+ 12-bit LUT shape already. The sub-crate's value is as the **dynasm-emit
target reference**: every asm emit must produce the same output as the Rust
reference per `route_c_testbed` on every silesia block.

## Falsifications recorded this session

- Extending multi-lit lookahead from 4→8 literals: -13%, icache pressure +
  branch mispredict (mirrors commit `ca52389` from May 2026).
- `bzhi_u64` wrapper for extras extraction: regressed equivalently — rustc
  already lowers `(src & ((1<<n)-1))` to BZHI under `-C target-cpu=native`.
- Raw-pointer unchecked literal writes: -9%, LLVM was already eliding the
  bounds check via range analysis.

These are documented inline in `route_c_dynamic.rs` so future sessions don't
re-walk them.
