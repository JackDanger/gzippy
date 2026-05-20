//! Parallel single-member decompression (production path).
//!
//! Entry: [`single_member::decompress_parallel`] → [`sm_driver::read_parallel_sm`]
//! → [`chunk_fetcher::drive`] → [`gzip_chunk::decode_chunk_isal_inexact`].

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
pub mod gzip_block_finder;
pub mod gzip_chunk;
pub mod gzip_format;
pub mod inflate_wrapper;
pub mod prefetcher;
pub mod replace_markers;
pub mod single_member;
pub mod sm_driver;
pub mod statistics;
pub mod thread_pool;
pub mod trace;
pub mod window_map;
