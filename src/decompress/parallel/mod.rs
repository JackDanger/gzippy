//! Parallel single-member decompression (production path).
//!
//! Entry: [`single_member::decompress_parallel`] → [`sm_driver::read_parallel_sm`]
//! → [`chunk_fetcher::drive`] → [`gzip_chunk::decode_chunk_isal_inexact`]
//! or [`gzip_chunk::decode_chunk_marker_bootstrap_then_isal`] (prefetch).

pub mod apply_window;
pub mod bit_manipulation;
pub mod block_fetcher;
pub mod block_finder;
pub mod block_map;
pub mod cache;
pub mod chunk_buffer_pool;
pub mod chunk_data;
pub mod chunk_fetcher;
pub mod compressed_vector;
pub mod crc32;
// Bootstrap-only (speculative prefetch): marker emit via `deflate_block::Block`.
pub mod deflate_block;
pub mod error;
pub mod gzip_block_finder;
pub mod gzip_chunk;
pub mod gzip_definitions;
pub mod gzip_format;
pub mod huffman_base;
pub mod huffman_symbols_per_length;
pub mod inflate_wrapper;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub mod isal_huffman;
pub mod prefetcher;
pub mod replace_markers;
pub mod single_member;
pub mod sm_driver;
pub mod statistics;
pub mod thread_pool;
pub mod trace;
pub mod window_map;
