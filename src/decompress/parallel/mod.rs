//! Parallel single-member decompression (production path).
//!
//! Entry: [`single_member::decompress_parallel`] → [`sm_driver::read_parallel_sm`]
//! → [`chunk_fetcher::drive`] → [`gzip_chunk::decode_chunk_with_rapidgzip`].
//!
//! # gzippy → rapidgzip ROLE MAP (which gz module ports which rg source)
//!
//! Faithful structural port of rapidgzip's chunked single-member decode. Vendor
//! source: `vendor/rapidgzip/src/rapidgzip/`. When a gz module "works but looks
//! structurally off", the cited vendor `file` is the reference.
//!
//! | gzippy module          | rapidgzip counterpart                         |
//! |------------------------|-----------------------------------------------|
//! | `single_member`        | `ParallelGzipReader` (entry / orchestration)  |
//! | `sm_driver`            | `ParallelGzipReader::read*` driver loop       |
//! | `chunk_fetcher`        | `GzipChunkFetcher` + `BlockFetcher` consumer  |
//! | `block_fetcher`        | `BlockFetcher` (prefetch/cache coordinator)   |
//! | `block_finder`/`gzip_block_finder`/`raw_block_finder` | `blockfinder/*` (deflate/gzip block boundary scan) |
//! | `gzip_chunk`           | `GzipChunk::decodeBlock` (per-chunk decode)   |
//! | `marker_inflate`       | `deflate::Block` (u16 marker-ring decode)     |
//! | `apply_window`/`replace_markers` | window application / marker resolution |
//! | `window_map`/`block_map` | `WindowMap` / `BlockMap`                    |
//! | `huffman_*`/`lut_*`    | `huffman/*` coding tables                      |
//! | `chunk_data`/`segmented_*` | `ChunkData` + `MarkerReplacement` buffers  |
//! | `crc32`                | `gzip/crc32.hpp`                              |
//! | `thread_pool`          | `ThreadPool`                                  |
//! | `instruments/*`        | NONE — campaign measurement instruments (env-gated, byte-transparent) |

#[cfg(parallel_sm)]
pub mod apply_window;
/// ASM-campaign rung (c): the contig clean fast loop's `asm!` kernel
/// (feature `asm-kernel`, x86_64-only; pure-Rust path always compiled).
#[cfg(parallel_sm)]
pub mod asm_kernel;
pub mod bit_manipulation;
#[cfg(parallel_sm)]
pub mod block_fetcher;
pub mod block_finder;
#[cfg(parallel_sm)]
pub mod block_map;
pub mod cache;
#[cfg(parallel_sm)]
pub mod chunk_buffer_pool;
#[cfg(parallel_sm)]
pub mod chunk_data;
#[cfg(parallel_sm)]
pub mod chunk_fetcher;
#[cfg(parallel_sm)]
pub mod chunk_handle;
pub mod compressed_vector;
pub mod crc32;
pub mod error;
#[cfg(parallel_sm)]
pub mod fd_vectored_write;
pub mod gzip_block_finder;
#[cfg(parallel_sm)]
pub mod gzip_chunk;
pub mod gzip_definitions;
pub mod gzip_format;
pub mod huffman_base;
pub mod huffman_reversed_bits_cached;
pub mod huffman_short_bits_cached;
pub mod huffman_symbols_per_length;
#[cfg(parallel_sm)]
pub mod inflate_wrapper;
#[cfg(parallel_sm)]
pub mod lut_bulk_inflate;
#[cfg(parallel_sm)]
pub mod lut_huffman;
/// Window-absent marker decoder: literal port of `rapidgzip::deflate::Block`
/// (vendor/.../gzip/deflate.hpp:513-1156) — the u16-marker ring + pre-seeded
/// marker zone decode loops shared by the unified single-member path. (Formerly
/// `deflate_block`; renamed to retire the "deflate_block bootstrap" name.)
#[cfg(parallel_sm)]
pub mod marker_inflate;
#[cfg(all(unix, parallel_sm))]
pub mod output_writer;
pub mod prefetcher;
#[cfg(parallel_sm)]
pub mod raw_block_finder;
#[cfg(parallel_sm)]
pub mod replace_markers;
#[cfg(parallel_sm)]
pub mod rpmalloc_alloc;
pub mod segmented_buffer;
#[cfg(parallel_sm)]
pub mod segmented_markers;
pub mod single_member;
pub mod sm_cfg;
#[cfg(parallel_sm)]
pub mod sm_driver;
pub mod statistics;
/// Non-speculative parallel decode for stored-block-dominated (incompressible)
/// single-member streams. Portable (depends only on crc32 + gzip_format), so it
/// is NOT cfg-gated; routing to it lives in [`crate::decompress::classify_gzip`].
pub mod stored_split;
#[cfg(parallel_sm)]
pub mod streamed_results;
#[cfg(parallel_sm)]
pub mod thread_pool;
#[cfg(parallel_sm)]
pub mod used_window_symbols;
#[cfg(parallel_sm)]
pub mod width_ring;
pub mod window_map;

/// Campaign measurement instruments (env-gated, byte-transparent, NO vendor
/// counterpart). Grouped out of the production modules above so the decode
/// pipeline reads as a clean structural mirror of rapidgzip. The re-exports
/// below preserve the historical `parallel::<name>` paths so every hot-path
/// hook call site is byte-transparent — only the file location changed.
pub mod instruments;
// `contig_prof`/`slow_knob` are consumed by the always-compiled inner loop
// (`inflate::resumable`) + `main`, so they are not parallel_sm-gated.
pub use instruments::{contig_prof, slow_knob};
// The rest are consumed only by parallel_sm modules; gate the re-export so the
// non-parallel_sm (legacy serial) build does not see them as unused imports.
#[cfg(parallel_sm)]
pub use instruments::{
    decode_bypass, memlife, perfect_overlap, removal_oracle, seed_windows, stall_residency,
    trace_jsonl as trace, trace_timeline as trace_v2,
};
