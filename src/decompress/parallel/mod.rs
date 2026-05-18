//! Parallel single-member decompression and its supporting primitives.
//!
//! Top-level entry: [`single_member::decompress_parallel`], wired into
//! [`crate::decompress::decompress_single_member`] when ISA-L is available,
//! `num_threads > 1`, the compressed stream exceeds 10 MiB, and the host has
//! enough physical cores for the algorithm's 2N compute work to amortize.
//!
//! - `single_member` — speculative-window two-pass design (production).
//! - `block_finder` — deflate block-boundary scanner (precode + Huffman validation).
//! - `replace_markers` — SIMD-accelerated marker resolution (AVX2 on x86_64,
//!   NEON on aarch64). Phase 2 of the marker pipeline; resolves cross-chunk
//!   back-references using MapMarkers semantics (`MARKER_BASE = 32768`,
//!   index from the OLDEST window byte) — see
//!   `vendor/rapidgzip/.../MarkerReplacement.hpp::MapMarkers`.
//! - `deflate_block` — rapidgzip-faithful port of `deflate::Block<>`
//!   (`vendor/rapidgzip/.../gzip/deflate.hpp`). Drives the bootstrap
//!   phase of the parallel single-member pipeline in
//!   `gzip_chunk::finish_decode_chunk_with_inexact_offset`: decodes
//!   deflate blocks one-by-one emitting MapMarkers cross-chunk
//!   back-references until ≥32 KiB of clean output accumulates, then
//!   hands off to patched ISA-L for the bulk decode.
//! - `gzip_chunk` — chunk-level decoder entry points (fast path with
//!   known dict, slow path that bootstraps via `deflate_block`).
//! - `inflate_wrapper` — patched-ISA-L wrapper used by the fast path
//!   and the post-bootstrap bulk decode.

pub mod affinity_helpers;
pub mod aligned_allocator;
pub mod apply_window;
pub mod atomic_mutex;
pub mod bit_manipulation;
pub mod bit_string_finder;
pub mod block_fetcher;
pub mod block_finder;
pub mod block_map;
pub mod blockfinder_bgzf;
pub mod blockfinder_dynamic_huffman;
pub mod blockfinder_interface;
pub mod blockfinder_pigz;
pub mod blockfinder_precode_check;
pub mod blockfinder_uncompressed;
pub mod cache;
pub mod chunk_data;
pub mod chunk_fetcher;
pub mod compressed_vector;
pub mod crc32;
pub mod decoded_data;
pub mod deflate_block;
pub mod error;
pub mod faster_vector;
pub mod file_type;
pub mod gzip_analyzer;
pub mod gzip_block_finder;
pub mod gzip_chunk;
pub mod gzip_definitions;
pub mod gzip_format;
pub mod gzip_reader;
pub mod huffman_base;
pub mod huffman_double_literal_cached;
pub mod huffman_reversed_bits_cached;
pub mod huffman_reversed_bits_cached_compressed;
pub mod huffman_reversed_bits_cached_separate_lengths;
pub mod huffman_reversed_codes_per_length;
pub mod huffman_short_bits_cached_deflate;
pub mod huffman_short_bits_multi_cached;
pub mod huffman_symbols_per_length;
pub mod index_file_format;
pub mod inflate_wrapper;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub mod isal_huffman;
pub mod joining_thread;
pub mod parallel_bit_string_finder;
pub mod precode;
pub mod prefetcher;
pub mod replace_markers;
pub mod rfc_tables;
pub mod simple_run_length_encoding;
pub mod single_member;
pub mod statistics;
pub mod streamed_results;
pub mod thread_pool;
pub mod trace;
pub mod window_map;
pub mod zlib_format;
