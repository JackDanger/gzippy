//! Parallel single-member decompression and its supporting primitives.
//!
//! Top-level entry: [`single_member::decompress_parallel`], wired into
//! [`crate::decompress::decompress_single_member`] when ISA-L is available,
//! `num_threads > 1`, and the compressed stream exceeds 10 MiB.
//!
//! - `single_member` — rapidgzip-style speculation + ISA-L `inflatePrime` (v0.3.0).
//! - `block_finder` — deflate block-boundary scanner (precode + Huffman validation).
//! - `marker_decode` — marker-based pre-decode used during boundary confirmation.
//! - `ultra_fast_inflate` — pre-allocated full-buffer inflate helper used by
//!   `marker_decode` and a handful of bgzf benches.

pub mod block_finder;
pub mod marker_decode;
pub mod single_member;
pub mod ultra_fast_inflate;
