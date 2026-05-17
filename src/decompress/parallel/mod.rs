//! Parallel single-member decompression and its supporting primitives.
//!
//! Top-level entry: [`single_member::decompress_parallel`], wired into
//! [`crate::decompress::decompress_single_member`] when ISA-L is available,
//! `num_threads > 1`, the compressed stream exceeds 10 MiB, and the host has
//! enough physical cores for the algorithm's 2N compute work to amortize.
//!
//! - `single_member` — v0.5.1 speculative-window two-pass design (production).
//! - `block_finder` — deflate block-boundary scanner (precode + Huffman validation).
//! - `marker_decode` — legacy pure-Rust marker decoder (~22 MB/s). Used only
//!   for `skip_gzip_header` today; the rest of the marker pipeline below
//!   supersedes it. Slated for deletion once the v0.6 marker pipeline lands
//!   in production routing.
//! - `replace_markers` — SIMD-accelerated marker resolution (AVX2 on x86_64,
//!   NEON on aarch64). Phase 2 of the v0.6 marker pipeline.
//! - `fast_marker_inflate` — pure-Rust marker-emitting deflate decoder.
//!   Phase 1 of the v0.6 marker pipeline. Tens of × faster than the legacy
//!   `marker_decode::MarkerDecoder` because it uses a 64-bit bit buffer and
//!   canonical-Huffman lookup tables instead of bit-by-bit reads.
//! - `ultra_fast_inflate` — pre-allocated full-buffer inflate helper used by
//!   `marker_decode` and a handful of bgzf benches.

pub mod apply_window;
pub mod block_finder;
pub mod chunk_data;
pub mod fast_marker_inflate;
pub mod marker_decode;
pub mod replace_markers;
pub mod single_member;
pub mod ultra_fast_inflate;
