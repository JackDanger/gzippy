//! Fixed Huffman table provider (libdeflate-compatible)
//!
//! Historically this module carried a full libdeflate-style decode loop
//! (`decode_libdeflate` / `inflate_libdeflate` / `decode_huffman*` and a local
//! `LibdeflateBits` reader). Those were never reached by any production decode
//! path (the parallel-SM clean tail and the scan_inflate→index path both use
//! `resumable`/`consume_first_decode`), so they were removed.
//!
//! The one live export is [`get_fixed_tables`], the cached fixed
//! literal/length + distance tables consumed by `resumable.rs` (clean tail) and
//! `consume_first_decode.rs`.

use crate::decompress::inflate::libdeflate_entry::{DistTable, LitLenTable};
use std::sync::OnceLock;

/// Cached fixed Huffman tables.
static FIXED_TABLES: OnceLock<(LitLenTable, DistTable)> = OnceLock::new();

/// Build (once) and return the fixed Huffman literal/length + distance tables.
pub fn get_fixed_tables() -> &'static (LitLenTable, DistTable) {
    FIXED_TABLES.get_or_init(|| {
        // Fixed literal/length table
        let mut litlen_lengths = [0u8; 288];
        for (i, len) in litlen_lengths.iter_mut().enumerate() {
            *len = match i {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
        }

        // Fixed distance table (all 5-bit codes)
        let dist_lengths = [5u8; 32];

        let litlen = LitLenTable::build(&litlen_lengths).unwrap();
        let dist = DistTable::build(&dist_lengths).unwrap();

        (litlen, dist)
    })
}
