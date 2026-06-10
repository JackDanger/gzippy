//! Parallel single-member decompression (production path).
//!
//! Entry: [`single_member::decompress_parallel`] → [`sm_driver::read_parallel_sm`]
//! → [`chunk_fetcher::drive`] → [`gzip_chunk::decode_chunk_with_rapidgzip`].

#[cfg(parallel_sm)]
pub mod apply_window;
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
#[cfg(parallel_sm)]
pub mod decode_bypass;
// Window-absent marker decoder (u16 ring + pre-seeded marker zone), shared with
// the unified single-member decode path. Formerly `deflate_block`; renamed to
// retire the "deflate_block bootstrap" name from production entirely.
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
#[cfg(parallel_sm)]
pub mod marker_inflate;
#[cfg(parallel_sm)]
pub mod memlife;
#[cfg(all(unix, parallel_sm))]
pub mod output_writer;
pub mod perfect_overlap;
pub mod prefetcher;
#[cfg(parallel_sm)]
pub mod raw_block_finder;
#[cfg(parallel_sm)]
pub mod replace_markers;
pub mod rfc_tables;
#[cfg(parallel_sm)]
pub mod rpmalloc_alloc;
/// CLEAN-ONLY ENGINE ORACLE — force every chunk through the clean decode path
/// via captured predecessor windows (campaign instrument). See `seed_windows.rs`.
#[cfg(parallel_sm)]
pub mod seed_windows;
pub mod segmented_buffer;
#[cfg(parallel_sm)]
pub mod segmented_markers;
pub mod single_member;
/// Byte-transparent env-gated slow-injection knob for the clean-mode inner
/// decode loop (causal-perturbation pre-gate). See `slow_knob.rs`.
pub mod slow_knob;
pub mod sm_cfg;
#[cfg(parallel_sm)]
pub mod sm_driver;
/// STEP-0 discriminator (a): byte-exact env-gated parent-cached-at-stall probe +
/// the SATURATION-vs-HORIZON occupancy probe (plans/prefetch-horizon-falsifier.md).
/// See `stall_residency.rs` + plans/step0-discriminator-a-falsifier.md. Gated on
/// `parallel_sm` because its only consumer (chunk_fetcher consumer loop) is — under
/// default features it would be dead code (clippy -D warnings).
#[cfg(parallel_sm)]
pub mod stall_residency;
pub mod statistics;
/// Non-speculative parallel decode for stored-block-dominated (incompressible)
/// single-member streams. Portable (depends only on crc32 + gzip_format), so it
/// is NOT cfg-gated; routing to it lives in [`crate::decompress::classify_gzip`].
pub mod stored_split;
#[cfg(parallel_sm)]
pub mod streamed_results;
#[cfg(parallel_sm)]
pub mod thread_pool;
pub mod trace;
pub mod trace_v2;
#[cfg(parallel_sm)]
pub mod used_window_symbols;
#[cfg(parallel_sm)]
pub mod width_ring;
pub mod window_map;
