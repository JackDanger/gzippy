//! Rust port of rapidgzip's `HuffmanCodingISAL`
//! (vendor/rapidgzip/.../huffman/HuffmanCodingISAL.hpp).
//!
//! Wraps ISA-L's internal `inflate_huff_code_large` table + the
//! `set_and_expand_lit_len_huffcode` / `make_inflate_huff_code_lit_len`
//! builders to provide a literal/length Huffman decoder that runs at
//! ISA-L's table-lookup speed (~340 MB/s/thread) instead of our pure-Rust
//! `ConsumeFirstTable` (~14 MB/s/thread).
//!
//! Used by the parallel single-member bootstrap inside
//! `fast_marker_inflate` to accelerate the dominant cost (Huffman decode
//! of literal/length symbols). Distance codes still use the existing
//! pure-Rust path — they're a small fraction of decode work.
//!
//! Architecture notes for the literal port:
//! - `LIT_LEN_ELEMS = LIT_LEN + 228` = 514 (matches ISA-L's internal
//!   buffer sizing for lit/len + expansion entries).
//! - `MAX_LIT_LEN_COUNT = MAX_LIT_LEN_CODE_LEN + 2` = 23 (one slot per
//!   possible code length 0..=21, plus a guard).
//! - `multisym = 0` matches rapidgzip's call — no multi-symbol decode
//!   compression (each decode returns a single symbol).
//! - Decode constants are bit-packed into the LUT entries the same way
//!   ISA-L lays them out; see `Self::decode` for the exact layout.

#![allow(dead_code)]
#![cfg(all(feature = "isal-compression", target_arch = "x86_64"))]

use crate::decompress::inflate::consume_first_decode::Bits;
use isal::isal_sys::igzip_lib::inflate_huff_code_large;
use isal::isal_sys::isal_internals::{
    huff_code, make_inflate_huff_code_lit_len, set_and_expand_lit_len_huffcode,
};

/// ISAL_DEF_LIT_LEN_SYMBOLS (igzip_lib.h). The number of literal/length
/// symbols in the deflate alphabet (256 literals + EOB + 29 length codes).
pub const LIT_LEN: u32 = 286;

/// Buffer size for `lit_and_dist_huff`. Per rapidgzip's port comment:
/// LIT_LEN + 228 = 514. ISA-L uses the tail for expansion bookkeeping.
pub const LIT_LEN_ELEMS: u32 = 514;

/// Max number of distinct code lengths (MAX_LIT_LEN_CODE_LEN + 2 = 23).
const MAX_LIT_LEN_COUNT: usize = 23;

/// ISA-L's short lookup uses the first 12 bits (ISAL_DECODE_LONG_BITS).
const ISAL_DECODE_LONG_BITS: u32 = 12;

/// Bit-layout constants of the short_code_lookup entry.
/// Match `vendor/rapidgzip/.../huffman/HuffmanCodingISAL.hpp:97-107`.
const LARGE_SHORT_SYM_LEN: u32 = 25;
const LARGE_SHORT_SYM_MASK: u32 = (1u32 << LARGE_SHORT_SYM_LEN) - 1;
const LARGE_LONG_SYM_LEN: u32 = 10;
const LARGE_LONG_SYM_MASK: u32 = (1u32 << LARGE_LONG_SYM_LEN) - 1;
const LARGE_FLAG_BIT: u32 = 1u32 << 25;
const LARGE_SHORT_CODE_LEN_OFFSET: u32 = 28;
const LARGE_SHORT_MAX_LEN_OFFSET: u32 = 26;
const LARGE_LONG_CODE_LEN_OFFSET: u32 = 10;

/// Bit-extra counts for length codes 257..286, per RFC 1951.
/// (Mirror of `len_extra_bit_count` at HuffmanCodingISAL.hpp:30-35.)
const LEN_EXTRA_BIT_COUNT: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0,
];

/// Decoded result. `sym_count` mirrors rapidgzip's multisym field; with
/// `multisym = 0` at build time, sym_count is always 1 — kept here for
/// fidelity with the reference implementation.
#[derive(Copy, Clone, Debug)]
pub struct DecodedSymbol {
    pub symbol: u32,
    pub sym_count: u32,
    pub bit_count: u32,
}

/// Wrapper around ISA-L's `inflate_huff_code_large` table. Built once
/// per dynamic-Huffman block from the precode-resolved code lengths.
/// Heap-allocated to keep the 19 KB table off the stack; reused
/// across blocks via thread-local storage in `with_thread_local`.
pub struct IsalLitLenCode {
    pub table: Box<inflate_huff_code_large>,
    lit_and_dist_huff: Box<[huff_code; LIT_LEN_ELEMS as usize]>,
    code_list: Box<[u32; (LIT_LEN_ELEMS as usize) + 2]>,
    valid: bool,
}

impl IsalLitLenCode {
    pub fn from_lengths(code_lengths: &[u8]) -> Option<Self> {
        let mut c = Self::new_empty();
        if c.rebuild_from(code_lengths) {
            Some(c)
        } else {
            None
        }
    }

    pub fn new_empty() -> Self {
        Self {
            table: Box::new(inflate_huff_code_large {
                short_code_lookup: [0u32; 4096],
                long_code_lookup: [0u16; 1264],
            }),
            lit_and_dist_huff: Box::new([huff_code::default(); LIT_LEN_ELEMS as usize]),
            code_list: Box::new([0u32; (LIT_LEN_ELEMS as usize) + 2]),
            valid: false,
        }
    }

    /// Rebuild the table from a slice of code lengths IN-PLACE — reuses
    /// the table/buffer allocations from this struct. Returns true on
    /// success; on failure, `is_valid()` returns false.
    pub fn rebuild_from(&mut self, code_lengths: &[u8]) -> bool {
        self.valid = false;
        if code_lengths.len() > LIT_LEN as usize {
            return false;
        }

        // Reset scratch buffers we mutate.
        for h in self.lit_and_dist_huff.iter_mut() {
            h.code_and_length = 0;
        }
        let mut lit_count: [u16; MAX_LIT_LEN_COUNT] = [0; MAX_LIT_LEN_COUNT];
        let mut lit_expand_count: [u16; MAX_LIT_LEN_COUNT] = [0; MAX_LIT_LEN_COUNT];

        for (i, &length) in code_lengths.iter().enumerate() {
            if (length as usize) >= MAX_LIT_LEN_COUNT {
                return false;
            }
            lit_count[length as usize] += 1;
            self.lit_and_dist_huff[i].code_and_length = (length as u32) << 24;
            if length != 0 && i >= 264 {
                let extra_count = LEN_EXTRA_BIT_COUNT[i - 257] as usize;
                lit_expand_count[length as usize] =
                    lit_expand_count[length as usize].wrapping_sub(1);
                let target = (length as usize) + extra_count;
                if target < MAX_LIT_LEN_COUNT {
                    lit_expand_count[target] =
                        lit_expand_count[target].wrapping_add(1u16 << extra_count);
                }
            }
        }

        let rc = unsafe {
            set_and_expand_lit_len_huffcode(
                self.lit_and_dist_huff.as_mut_ptr(),
                LIT_LEN,
                lit_count.as_mut_ptr(),
                lit_expand_count.as_mut_ptr(),
                self.code_list.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return false;
        }

        unsafe {
            make_inflate_huff_code_lit_len(
                self.table.as_mut(),
                self.lit_and_dist_huff.as_mut_ptr(),
                LIT_LEN_ELEMS,
                lit_count.as_ptr(),
                self.code_list.as_mut_ptr(),
                0,
            );
        }
        self.valid = true;
        true
    }

    /// Decode the next symbol from `bits`. Returns the symbol value
    /// and the bit count consumed. Caller must consume `bit_count` bits
    /// after this call — we don't mutate the BitReader internally for
    /// inlinability, matching rapidgzip's `seekAfterPeek` pattern only
    /// when caller indicates.
    ///
    /// On invalid code: symbol = 0x1FFF (INVALID_SYMBOL).
    ///
    /// LITERAL port of `HuffmanCodingISAL::decode` at
    /// `vendor/rapidgzip/.../huffman/HuffmanCodingISAL.hpp:94-183`.
    #[inline]
    pub fn decode(&self, bits: &mut Bits) -> DecodedSymbol {
        const INVALID_SYMBOL: u32 = 0x1FFF;
        if bits.available() < 32 {
            bits.refill();
        }
        let next_bits = bits.peek();
        let next_12_bits = (next_bits & ((1u64 << ISAL_DECODE_LONG_BITS) - 1)) as usize;
        let mut next_sym = self.table.short_code_lookup[next_12_bits];
        if (next_sym & LARGE_FLAG_BIT) == 0 {
            let bit_count = next_sym >> LARGE_SHORT_CODE_LEN_OFFSET;
            let mut symbol = next_sym & LARGE_SHORT_SYM_MASK;
            if bit_count == 0 {
                symbol = INVALID_SYMBOL;
            }
            return DecodedSymbol {
                symbol,
                sym_count: (next_sym >> 26) & 0b11, // LARGE_SYM_COUNT_OFFSET=26, mask 0b11
                bit_count,
            };
        }
        // Long code path.
        let long_max_len = next_sym >> LARGE_SHORT_MAX_LEN_OFFSET;
        let used_bits = if long_max_len <= 32 {
            next_bits & ((1u64 << long_max_len) - 1)
        } else {
            next_bits
        };
        let long_idx = ((next_sym & LARGE_SHORT_SYM_MASK)
            + ((used_bits >> ISAL_DECODE_LONG_BITS) as u32)) as usize;
        next_sym = self.table.long_code_lookup[long_idx] as u32;
        let bit_count = next_sym >> LARGE_LONG_CODE_LEN_OFFSET;
        let mut symbol = next_sym & LARGE_LONG_SYM_MASK;
        if bit_count == 0 {
            symbol = INVALID_SYMBOL;
        }
        DecodedSymbol {
            symbol,
            sym_count: 1,
            bit_count,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

thread_local! {
    static THREAD_LITLEN_TABLE: std::cell::RefCell<IsalLitLenCode> =
        std::cell::RefCell::new(IsalLitLenCode::new_empty());
}

/// Run `f` with a thread-local IsalLitLenCode rebuilt from `code_lengths`.
/// Avoids per-call allocation of the 19 KB table.
pub fn with_thread_litlen<R>(
    code_lengths: &[u8],
    f: impl FnOnce(&IsalLitLenCode) -> R,
) -> Option<R> {
    THREAD_LITLEN_TABLE.with(|cell| {
        let mut t = cell.borrow_mut();
        if !t.rebuild_from(code_lengths) {
            return None;
        }
        Some(f(&t))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build a trivial code table from RFC 1951's fixed
    /// Huffman code lengths. Confirms FFI symbol resolution + linkage.
    #[test]
    fn isal_internals_link_and_run() {
        // RFC 1951 fixed lit/len code lengths:
        // 0-143 → 8; 144-255 → 9; 256-279 → 7; 280-287 → 8
        let mut lens = vec![0u8; 288];
        for v in &mut lens[0..144] {
            *v = 8;
        }
        for v in &mut lens[144..256] {
            *v = 9;
        }
        for v in &mut lens[256..280] {
            *v = 7;
        }
        for v in &mut lens[280..288] {
            *v = 8;
        }
        // 288 > LIT_LEN (286). Trim to LIT_LEN.
        lens.truncate(LIT_LEN as usize);
        let code = IsalLitLenCode::from_lengths(&lens);
        assert!(code.is_some(), "fixed-Huffman lit/len table should build");
        let code = code.unwrap();
        assert!(code.is_valid());
        assert_eq!(code.table.short_code_lookup.len(), 4096);
    }
}
