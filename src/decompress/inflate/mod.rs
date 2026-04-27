//! Pure-Rust inflate primitives shared across the decompression engine.
//!
//! These are the "consume-first" Huffman decoder, its lookup tables, and the
//! Huffman-table format compatible with libdeflate's internal layout. Used by
//! `decompress::scan_inflate`, `decompress::bgzf`, and the parallel speculation
//! pipeline.
//!
//! - `consume_first_decode` — consume-first decoder + per-block code-path entry.
//! - `consume_first_table` — packed Huffman lookup table layout.
//! - `libdeflate_entry` / `libdeflate_decode` — fixed-table builders and entry packing
//!   matching libdeflate's `decompress_template.h` layout.
//! - `jit_decode` — table-fingerprint cache key for specialized decoders.
//! - `specialized_decode` — code-length-specialized decoder (built per dynamic block).
//! - `vector_huffman` — SIMD multi-symbol decode for short codes.
//! - `double_literal` — two-symbol literal cache for back-to-back short codes.
//! - `bmi2` — x86_64 BMI2 bit-extraction intrinsics (with portable fallback).

pub mod bmi2;
pub mod consume_first_decode;
pub mod consume_first_table;
pub mod double_literal;
pub mod jit_decode;
pub mod libdeflate_decode;
pub mod libdeflate_entry;
pub mod specialized_decode;
pub mod vector_huffman;
