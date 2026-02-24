# Production Code Paths (Feb 2026)

Every function listed here is reachable from the CLI binary. If a module is
not listed, it's R&D/benchmark-only code.

## Compression (stdin → stdout: `gzippy -N -c -pT < in > out`)

```
compress_stdin (compression.rs)
├── Tmax L6-L9 → PipelinedGzEncoder::compress_buffer → zlib-ng/flate2
├── Tmax L1-L5,L10-L12 → ParallelGzEncoder::compress_buffer
│   ├── L1-L3 + ISA-L → compress_block_bgzf_isal → isal-rs
│   └── L4-L5,L10-L12 → compress_block_bgzf_libdeflate → libdeflate
├── T1 L0-L3 + ISA-L → compress_gzip_to_writer → isal-rs
└── T1 fallback → compress_with_pipeline
    ├── L1-L3 + ISA-L → compress_gzip_stream → isal-rs
    ├── L1-L5 → compress_single_member → libdeflate or ISA-L
    ├── Ratio < 10% → flate2 streaming → zlib-ng
    └── L6-L9 → flate2 streaming → zlib-ng

compress_file (compression.rs, file path)
├── mmap + SimpleOptimizer → ParallelGzEncoder or PipelinedGzEncoder
└── rsyncable → compress_rsyncable → libdeflate
```

## Decompression (stdin → stdout: `gzippy -d -pT < in > out`)

```
decompress_stdin / decompress_file (decompression.rs)
├── BGZF detected → bgzf::decompress_bgzf_parallel
│   ├── T1 → decompress_bgzf_streaming → libdeflate_deflate_decompress (FFI)
│   └── Tmax → decompress_bgzf_pipelined → libdeflate_deflate_decompress (FFI)
├── Multi-member + Tmax → decompress_gzip_to_vec
│   └── → bgzf::decompress_multi_member_parallel → libdeflate gzip_decompress_ex (FFI)
├── Single-member + ≥8 threads → parallel_single_member::decompress_parallel
│   ├── Pass 1: scan_deflate_fast → consume_first_decode (pure Rust)
│   └── Pass 2: parallel chunk decode → consume_first_decode (pure Rust)
├── Single-member fallback → decompress_single_member
│   ├── ISA-L available → isal_decompress::decompress_gzip_stream → ISA-L (FFI)
│   └── → decompress_single_member_libdeflate → libdeflate gzip_decompress_ex (FFI)
└── Fallback → decompress_multi_member_sequential → libdeflate gzip_decompress_ex (FFI)

decompress_file (!stdout, write to file)
└── decompress_mmap_libdeflate → decompress_gzip_libdeflate
    ├── BGZF → bgzf::decompress_bgzf_parallel
    ├── Single → decompress_single_member → ISA-L or libdeflate
    └── Multi → decompress_multi_member_parallel or sequential
```

## The Gap: Single-Member Tmax

8 benchmark losses (-20% to -40% vs rapidgzip) come from this path.
Two-pass parallel (scan + re-decode) is implemented but gated behind
MIN_THREADS_FOR_PARALLEL=8 because the scan costs ~100% of a sequential
decode. At 4 threads (CI), sequential is always faster.

To close: need pipeline architecture (block-finder thread feeding decoder
threads concurrently) rather than two-pass sequential scan.

## Modules: Production vs R&D

| Module | Status | Used by |
|--------|--------|---------|
| bgzf.rs | PRODUCTION | BGZF + multi-member decompress |
| compression.rs | PRODUCTION | All compression |
| decompression.rs | PRODUCTION | Decompress routing |
| isal_compress.rs | PRODUCTION | ISA-L L0-L3 compress |
| isal_decompress.rs | PRODUCTION | Single-member decompress (x86) |
| parallel_compress.rs | PRODUCTION | Tmax BGZF compress |
| pipelined_compress.rs | PRODUCTION | Tmax L6-L9 compress |
| simple_optimizations.rs | PRODUCTION | File compress routing |
| libdeflate_ext.rs | PRODUCTION | libdeflate FFI wrappers |
| consume_first_decode.rs | R&D | Pure-Rust inflate (benchmarks only) |
| consume_first_table.rs | R&D | Table for consume_first |
| libdeflate_decode.rs | R&D | Pure-Rust libdeflate-style decode |
| libdeflate_entry.rs | R&D | Entry format for libdeflate_decode |
| parallel_single_member.rs | PRODUCTION | Single-member Tmax (≥8 threads) |
| scan_inflate.rs | PRODUCTION | Block scanning for parallel_single_member |
| two_pass_parallel.rs | R&D | Old two-pass parallel (tests only) |
| marker_decode.rs | R&D | Marker-based decode (tests only) |
| ultra_fast_inflate.rs | R&D | Pure-Rust inflate (tests only) |
| ultra_inflate.rs | R&D | Experimental inflate (tests only) |
| speculative_parallel.rs | DEAD | Not declared in main.rs |
| vector_huffman.rs | R&D | SIMD infrastructure |
| bmi2.rs | R&D | BMI2 infrastructure |
| jit_decode.rs | R&D | JIT infrastructure |
| simd_huffman.rs | R&D | SIMD Huffman infrastructure |
| specialized_decode.rs | R&D | Specialized decoder |
| double_literal.rs | R&D | Double-literal optimization |
| packed_lut.rs | R&D | Packed LUT format |
| combined_lut.rs | R&D | Combined LUT format |
| two_level_table.rs | R&D | Two-level Huffman tables |
| block_finder.rs | R&D | Block boundary finder |
| block_finder_lut.rs | R&D | LUT for block finder |
