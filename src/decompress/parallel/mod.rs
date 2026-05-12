//! Parallel single-member decompression and its supporting primitives.
//!
//! Top-level entry: [`single_member::decompress_parallel`], wired into
//! [`crate::decompress::decompress_single_member`] when ISA-L is available,
//! `num_threads > 1`, the compressed stream exceeds 10 MiB, and the host has
//! enough physical cores for the algorithm's 2N compute work to amortize.
//!
//! - `single_member` — v0.5.1 speculative-window two-pass design (production).
//! - `block_finder` — deflate block-boundary scanner (precode + Huffman validation).
//! - `marker_decode` — pure-Rust marker-emitting deflate decoder (slow ~22 MB/s,
//!   used for boundary confirmation; the v0.6 marker pipeline below depends on
//!   a faster replacement landing first).
//! - `replace_markers` — SIMD-accelerated marker resolution (AVX2 on x86_64,
//!   NEON on aarch64). Phase-2 of the planned v0.6 marker pipeline that will
//!   replace `single_member`'s speculative-window design with rapidgzip-style
//!   marker-based parallel decoding. The fast marker-emitting decoder that
//!   feeds it is in-progress.
//! - `ultra_fast_inflate` — pre-allocated full-buffer inflate helper used by
//!   `marker_decode` and a handful of bgzf benches.

pub mod block_finder;
pub mod marker_decode;
pub mod replace_markers;
pub mod single_member;
pub mod ultra_fast_inflate;
