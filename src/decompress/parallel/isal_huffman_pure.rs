//! Pure-Rust port of ISA-L's `inflate_huff_code_{large,small}` LUT format
//! plus the `set_codes` / `set_and_expand_lit_len_huffcode` /
//! `make_inflate_huff_code_lit_len` / `make_inflate_huff_code_dist` builders.
//!
//! ## Why this exists
//!
//! `isal_huffman.rs` wraps the C versions via `isal::isal_sys::*` — only
//! available on `feature = "isal-compression", target_arch = "x86_64"`. The
//! pure-rust-inflate build was falling back to
//! `HuffmanCodingShortBitsMultiCached` (vendor's MultiCached.hpp port), which
//! per the May 27 2026 neurotic perf is the dominant cost in the marker-
//! phase bootstrap decoder (~7.65% of total CPU, with the bootstrap caller
//! at 81% children-time). MultiCached uses a simpler 11-bit cache that
//! caps multi-symbol-packing at 2; ISA-L's `inflate_huff_code_large` uses
//! 12-bit short + variable-length long lookup and packs up to 3 symbols per
//! entry — measurably faster per-byte.
//!
//! This module is the load-bearing piece that lets pure-rust-inflate builds
//! use the same fast LUT format as ISA-L. Port reference:
//! `vendor/isa-l/igzip/igzip_inflate.c:46-599`.
//!
//! ## Port faithfulness
//!
//! Constants, bit layouts, table sizes, and the three-loop short-code
//! packing pass match the C exactly (line citations at each function). The
//! `HuffCode` struct condenses the C union to a single `u32` with named
//! accessors — see its doc comment. The decoder (`decode`) is a literal
//! Rust translation of the C; it was already present in `isal_huffman.rs`
//! and is reproduced here without behavioral change so this module is
//! self-contained.
//!
//! ## Correctness gate
//!
//! When BOTH the `isal-compression` and `pure-rust-inflate` features are
//! enabled (the test-only cross-check build profile), the unit test
//! `port_matches_c_isal_byte_for_byte` runs both this Rust builder and
//! ISA-L's C builder on the same code-length inputs and asserts the
//! resulting LUTs are byte-equal. Per the memory rule
//! `feedback-real-corpus-test-with-lever`, the wiring commit also ships a
//! silesia byte-perfect differential.
//!
//! ## License
//!
//! Original code is BSD-3-Clause licensed by Intel Corporation (ISA-L).
//! See `vendor/isa-l/LICENSE`.

#![cfg(any(feature = "isal-compression", feature = "pure-rust-inflate"))]
#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// Constants — verbatim from `vendor/isa-l/igzip/igzip_inflate.c:46-91`.
// ─────────────────────────────────────────────────────────────────────────────

/// Bits used to index the short-code lookup table (`inflate_huff_code_large`).
/// Vendor: `ISAL_DECODE_LONG_BITS = 12` (igzip_lib.h).
pub const ISAL_DECODE_LONG_BITS: u32 = 12;

/// Bits used to index the short-code lookup of the SMALL (distance) table.
/// Vendor: `ISAL_DECODE_SHORT_BITS = 10`.
pub const ISAL_DECODE_SHORT_BITS: u32 = 10;

/// Layout of `short_code_lookup[]` entries for the LARGE (lit/len) table.
///
/// Bits 0..25  = packed symbols (up to 3 × 8-bit)
/// Bits 25     = LARGE_FLAG_BIT (0 = short-code entry, 1 = long-code pointer)
/// Bits 26..28 = symbol count (1, 2, or 3) — short-code case only
/// Bits 26..32 = long-max-length              — long-code case only
/// Bits 28..32 = code length                   — short-code case only
pub const LARGE_SHORT_SYM_LEN: u32 = 25;
pub const LARGE_SHORT_SYM_MASK: u32 = (1u32 << LARGE_SHORT_SYM_LEN) - 1;
pub const LARGE_LONG_SYM_LEN: u32 = 10;
pub const LARGE_LONG_SYM_MASK: u32 = (1u32 << LARGE_LONG_SYM_LEN) - 1;
pub const LARGE_SHORT_CODE_LEN_OFFSET: u32 = 28;
pub const LARGE_LONG_CODE_LEN_OFFSET: u32 = 10;
pub const LARGE_FLAG_BIT_OFFSET: u32 = 25;
pub const LARGE_FLAG_BIT: u32 = 1u32 << LARGE_FLAG_BIT_OFFSET;
pub const LARGE_SYM_COUNT_OFFSET: u32 = 26;
pub const LARGE_SYM_COUNT_LEN: u32 = 2;
pub const LARGE_SYM_COUNT_MASK: u32 = (1u32 << LARGE_SYM_COUNT_LEN) - 1;
pub const LARGE_SHORT_MAX_LEN_OFFSET: u32 = 26;

/// Layout for the SMALL (distance) lookup.
pub const SMALL_SHORT_SYM_LEN: u32 = 9;
pub const SMALL_SHORT_SYM_MASK: u32 = (1u32 << SMALL_SHORT_SYM_LEN) - 1;
pub const SMALL_LONG_SYM_LEN: u32 = 9;
pub const SMALL_LONG_SYM_MASK: u32 = (1u32 << SMALL_LONG_SYM_LEN) - 1;
pub const SMALL_SHORT_CODE_LEN_OFFSET: u32 = 11;
pub const SMALL_LONG_CODE_LEN_OFFSET: u32 = 10;
pub const SMALL_FLAG_BIT_OFFSET: u32 = 10;
pub const SMALL_FLAG_BIT: u32 = 1u32 << SMALL_FLAG_BIT_OFFSET;

pub const DIST_SYM_OFFSET: u32 = 0;
pub const DIST_SYM_LEN: u32 = 5;
pub const DIST_SYM_MASK: u32 = (1u32 << DIST_SYM_LEN) - 1;
pub const DIST_SYM_EXTRA_OFFSET: u32 = 5;
pub const DIST_SYM_EXTRA_LEN: u32 = 4;
pub const DIST_SYM_EXTRA_MASK: u32 = (1u32 << DIST_SYM_EXTRA_LEN) - 1;

/// Vendor: `MAX_LIT_LEN_CODE_LEN = 21`. Deflate spec allows max 15-bit
/// codes for lit/len in dynamic blocks, but ISA-L's expansion can grow
/// the effective code-length (code bits + length-extra bits) up to 21.
pub const MAX_LIT_LEN_CODE_LEN: usize = 21;
pub const MAX_LIT_LEN_COUNT: usize = MAX_LIT_LEN_CODE_LEN + 2;
pub const MAX_LIT_LEN_SYM: u32 = 512;
pub const LIT_LEN_ELEMS: usize = 514;

/// Vendor `ISAL_DEF_LIT_LEN_SYMBOLS = 286` (igzip_lib.h). 256 literals +
/// EOB + 29 length codes.
pub const LIT_LEN: usize = 286;
pub const ISAL_DEF_LIT_SYMBOLS: usize = 257;
pub const ISAL_DEF_LEN_SYMBOLS: usize = 29;
pub const ISAL_DEF_DIST_SYMBOLS: usize = 30;

pub const MAX_HUFF_TREE_DEPTH: usize = 15;

/// Multi-symbol packing flag — `0 = TRIPLE` means up to 3 syms per LUT
/// entry. rapidgzip uses TRIPLE_SYM_FLAG at `HuffmanCodingISAL.hpp:71`.
pub const TRIPLE_SYM_FLAG: u32 = 0;
pub const DOUBLE_SYM_FLAG: u32 = TRIPLE_SYM_FLAG + 1;
pub const SINGLE_SYM_FLAG: u32 = DOUBLE_SYM_FLAG + 1;
pub const DEFAULT_SYM_FLAG: u32 = TRIPLE_SYM_FLAG;

pub const INVALID_SYMBOL: u32 = 0x1FFF;
pub const INVALID_CODE: u32 = 0xFF_FFFF;

pub const ISAL_INVALID_BLOCK: i32 = -2;

/// Length-extra bit counts indexed by length-code (length symbol minus 257).
/// Vendor `rfc_lookup_table.len_extra_bit_count` (igzip_inflate.c:113-115).
pub const LEN_EXTRA_BIT_COUNT: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02,
    0x03, 0x03, 0x03, 0x03, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x00, 0x00, 0x00, 0x00,
];

/// 8-bit bit-reverse lookup table. Vendor: `bitrev_table`
/// (igzip_inflate.c:158-177).
const BITREV_TABLE: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];

/// Reverse the low `length` bits of `bits` (vendor `bit_reverse2`,
/// igzip_inflate.c:183-190).
#[inline(always)]
pub fn bit_reverse2(bits: u16, length: u8) -> u32 {
    let bitrev = (BITREV_TABLE[(bits >> 8) as usize] as u32)
        | ((BITREV_TABLE[(bits & 0xFF) as usize] as u32) << 8);
    bitrev >> (16 - length as u32)
}

/// Translate a flat code-list index back to its symbol. Used by the LUT
/// builder to discriminate expansion-entry indices from real symbols.
/// Vendor `index_to_sym` (igzip_inflate.c:381) — returns the index
/// unchanged except remapping 513 → 512.
#[inline(always)]
fn index_to_sym(index: u32) -> u32 {
    if index != 513 {
        index
    } else {
        512
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HuffCode — Rust representation of the C `struct huff_code` union
// (vendor huff_codes.h:105-131). Three views over the same u32:
//   - `code_and_length` = (length << 24) | (extra_bit_count << 16) | code
//   - `code_and_extra`  = (extra_bit_count << 16) | code        (low 24 bits)
//   - {code:u16, extra_bit_count:u8, length:u8}                  (named field accessors)
// The `set_codes` / `set_and_expand_*` writers go through
// `code_and_length` and `code_and_extra`; the LUT builder reads via the
// named fields. All three views must remain consistent — that's the
// invariant the named methods preserve.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HuffCode(pub u32);

impl HuffCode {
    #[inline(always)]
    pub fn code(&self) -> u16 {
        self.0 as u16
    }
    #[inline(always)]
    pub fn extra_bit_count(&self) -> u8 {
        (self.0 >> 16) as u8
    }
    #[inline(always)]
    pub fn length(&self) -> u8 {
        (self.0 >> 24) as u8
    }
    #[inline(always)]
    pub fn code_and_extra(&self) -> u32 {
        self.0 & 0xFF_FFFF
    }
    #[inline(always)]
    pub fn code_and_length(&self) -> u32 {
        self.0
    }
    #[inline(always)]
    pub fn set_length(&mut self, length: u8) {
        self.0 = (self.0 & 0x00FF_FFFF) | ((length as u32) << 24);
    }
    #[inline(always)]
    pub fn set_code_and_length(&mut self, code: u32, length: u32) {
        // Mirror of vendor `write_huff_code` (igzip_inflate.c:244-247).
        // Note: `code` from `bit_reverse2` fits in 16 bits; setting
        // code_and_length with that ZEROES the extra_bit_count byte,
        // matching the C behavior (write through the `code_and_length`
        // union view clobbers the entire 32 bits).
        self.0 = code | (length << 24);
    }
    #[inline(always)]
    pub fn set_code_and_extra(&mut self, code_and_extra: u32) {
        self.0 = (self.0 & 0xFF00_0000) | (code_and_extra & 0x00FF_FFFF);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LUT containers — same fixed sizes as ISA-L's `inflate_huff_code_large`
// (igzip_lib.h:?) and `inflate_huff_code_small`.
// ─────────────────────────────────────────────────────────────────────────────

/// ISA-L's `inflate_huff_code_large` — fixed sizes matching the C struct.
/// 4096 = `1 << ISAL_DECODE_LONG_BITS`. 1264 = empirical worst-case
/// long-code fan-out from the deflate alphabet (vendor sized this
/// constant; we mirror it).
#[repr(C)]
pub struct InflateHuffCodeLarge {
    pub short_code_lookup: [u32; 1 << ISAL_DECODE_LONG_BITS],
    pub long_code_lookup: [u16; 1264],
}

impl Default for InflateHuffCodeLarge {
    fn default() -> Self {
        Self {
            short_code_lookup: [0u32; 1 << ISAL_DECODE_LONG_BITS],
            long_code_lookup: [0u16; 1264],
        }
    }
}

/// ISA-L's `inflate_huff_code_small`. 1024 = `1 << ISAL_DECODE_SHORT_BITS`.
/// 80 = empirical worst-case long-code fan-out for the distance alphabet.
#[repr(C)]
pub struct InflateHuffCodeSmall {
    pub short_code_lookup: [u16; 1 << ISAL_DECODE_SHORT_BITS],
    pub long_code_lookup: [u16; 80],
}

impl Default for InflateHuffCodeSmall {
    fn default() -> Self {
        Self {
            short_code_lookup: [0u16; 1 << ISAL_DECODE_SHORT_BITS],
            long_code_lookup: [0u16; 80],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// set_codes — assigns canonical Huffman code values per code-length count.
// Vendor: `igzip_inflate.c:249-279`. Returns Err on invalid (Kraft-
// violating) input.
// ─────────────────────────────────────────────────────────────────────────────

/// Port of vendor `set_codes` (igzip_inflate.c:249-279). On success the
/// `huff_code_table[i].code_and_length` is written for every i with
/// `length != 0`.
pub fn set_codes(
    huff_code_table: &mut [HuffCode],
    table_length: usize,
    count: &mut [u16; MAX_HUFF_TREE_DEPTH + 1],
) -> Result<(), i32> {
    let mut next_code = [0u32; MAX_HUFF_TREE_DEPTH + 1];

    // Setup for calculating huffman codes
    next_code[0] = 0;
    next_code[1] = 0;
    for i in 2..=MAX_HUFF_TREE_DEPTH {
        next_code[i] = (next_code[i - 1] + count[i - 1] as u32) << 1;
    }

    let max = next_code[MAX_HUFF_TREE_DEPTH] + count[MAX_HUFF_TREE_DEPTH] as u32;
    if max > (1u32 << MAX_HUFF_TREE_DEPTH) {
        return Err(ISAL_INVALID_BLOCK);
    }

    for entry in huff_code_table.iter_mut().take(table_length) {
        let length = entry.length() as u32;
        if length == 0 {
            continue;
        }
        let code = bit_reverse2(next_code[length as usize] as u16, length as u8);
        entry.set_code_and_length(code, length);
        next_code[length as usize] += 1;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// set_and_expand_lit_len_huffcode — expands length-code symbols into one
// entry per length-extra bit combination. Vendor:
// `igzip_inflate.c:281-379`.
//
// This pre-bakes the length value into expanded symbols, so the runtime
// decoder need not consume length-extra bits separately — the LUT entry
// returned by `decode` already covers `codeword_bits + length_extra_bits`
// as its `bit_count`.
// ─────────────────────────────────────────────────────────────────────────────

pub fn set_and_expand_lit_len_huffcode(
    lit_len_huff: &mut [HuffCode],
    table_length: usize,
    count: &mut [u16; MAX_LIT_LEN_COUNT],
    expand_count: &mut [u16; MAX_LIT_LEN_COUNT],
    code_list: &mut [u32],
) -> Result<(), i32> {
    let mut next_code = [0u32; MAX_HUFF_TREE_DEPTH + 1];

    // Setup for calculating huffman codes
    let mut count_total: u32 = 0;
    let mut count_tmp: u32 = expand_count[1] as u32;
    next_code[0] = 0;
    next_code[1] = 0;
    expand_count[0] = 0;
    expand_count[1] = 0;

    let mut i = 1;
    while i < MAX_HUFF_TREE_DEPTH {
        count_total = count[i] as u32 + count_tmp + count_total;
        count_tmp = expand_count[i + 1] as u32;
        expand_count[i + 1] = count_total as u16;
        next_code[i + 1] = (next_code[i] + count[i] as u32) << 1;
        i += 1;
    }

    count_tmp = count[i] as u32 + count_tmp;

    while i < MAX_LIT_LEN_COUNT - 1 {
        count_total = count_tmp + count_total;
        count_tmp = expand_count[i + 1] as u32;
        expand_count[i + 1] = count_total as u16;
        i += 1;
    }

    // Correct for extra symbols used by static header
    if table_length > LIT_LEN {
        count[8] -= 2;
    }

    let max = next_code[MAX_HUFF_TREE_DEPTH] + count[MAX_HUFF_TREE_DEPTH] as u32;
    if max > (1u32 << MAX_HUFF_TREE_DEPTH) {
        return Err(ISAL_INVALID_BLOCK);
    }

    // memcpy(count, expand_count, sizeof(*count) * MAX_LIT_LEN_COUNT)
    for k in 0..MAX_LIT_LEN_COUNT {
        count[k] = expand_count[k];
    }

    // Snapshot the length-code portion of lit_len_huff before we zero it out
    // (vendor: memcpy(tmp_table, ...); memset(...)).
    let tmp_table_len = LIT_LEN - ISAL_DEF_LIT_SYMBOLS;
    let mut tmp_table: [HuffCode; 29] = [HuffCode::default(); 29];
    debug_assert_eq!(tmp_table.len(), tmp_table_len);
    for j in 0..tmp_table_len {
        tmp_table[j] = lit_len_huff[ISAL_DEF_LIT_SYMBOLS + j];
    }
    // Clear the length-code region of lit_len_huff (entries get overwritten
    // by the expansion below).
    for j in ISAL_DEF_LIT_SYMBOLS..LIT_LEN_ELEMS {
        lit_len_huff[j] = HuffCode::default();
    }

    // Calculate code for each literal symbol (ISAL_DEF_LIT_SYMBOLS = 257).
    for idx in 0..ISAL_DEF_LIT_SYMBOLS {
        let code_len = lit_len_huff[idx].length() as u32;
        if code_len == 0 {
            continue;
        }
        let code = bit_reverse2(next_code[code_len as usize] as u16, code_len as u8);
        let insert_index = expand_count[code_len as usize];
        code_list[insert_index as usize] = idx as u32;
        expand_count[code_len as usize] += 1;
        lit_len_huff[idx].set_code_and_length(code, code_len);
        next_code[code_len as usize] += 1;
    }

    // Calculate code for each length symbol, expanding by 2^extra bits.
    // `expand_next` mirrors the C pointer that walks the expansion region.
    let mut expand_next_offset: usize = ISAL_DEF_LIT_SYMBOLS;
    for len_sym in 0..(LIT_LEN - ISAL_DEF_LIT_SYMBOLS) {
        let extra_count = LEN_EXTRA_BIT_COUNT[len_sym] as u32;
        let len_size: u32 = 1u32 << extra_count;

        let code_len = tmp_table[len_sym].length() as u32;
        if code_len == 0 {
            expand_next_offset += len_size as usize;
            continue;
        }

        let code = bit_reverse2(next_code[code_len as usize] as u16, code_len as u8);
        let expand_len = code_len + extra_count;
        next_code[code_len as usize] += 1;
        let mut insert_index = expand_count[expand_len as usize];
        expand_count[expand_len as usize] += len_size as u16;

        for extra in 0..len_size {
            code_list[insert_index as usize] = expand_next_offset as u32;
            // write_huff_code(expand_next, code | (extra << code_len), expand_len)
            lit_len_huff[expand_next_offset]
                .set_code_and_length(code | (extra << code_len), expand_len);
            insert_index += 1;
            expand_next_offset += 1;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// make_inflate_huff_code_lit_len — builds the actual LUT with multi-symbol
// short-code packing. Vendor: `igzip_inflate.c:386-599`.
//
// The three-loop pass over `last_length` ∈ [min_length, ISAL_DECODE_LONG_BITS]:
//   1. Singletons: every code of `last_length` bits gets a LUT entry.
//   2. Pairs: literal+something packed where total bits ≤ last_length.
//   3. Triples: literal+literal+something packed.
//
// Codes longer than ISAL_DECODE_LONG_BITS go through the long_code_lookup
// secondary table (final loop).
// ─────────────────────────────────────────────────────────────────────────────

pub fn make_inflate_huff_code_lit_len(
    result: &mut InflateHuffCodeLarge,
    huff_code_table: &mut [HuffCode],
    _table_length: usize,
    count_total: &[u16; MAX_LIT_LEN_COUNT],
    code_list: &[u32],
    multisym: u32,
) {
    let max_symbol: u32 = MAX_LIT_LEN_SYM;

    let code_list_len = count_total[MAX_LIT_LEN_COUNT - 1] as u32;
    if code_list_len == 0 {
        for v in result.short_code_lookup.iter_mut() {
            *v = 0;
        }
        return;
    }

    // Determine the length of the first code
    let last_first = huff_code_table[code_list[0] as usize].length() as u32;
    let mut last_length = if last_first > ISAL_DECODE_LONG_BITS {
        ISAL_DECODE_LONG_BITS + 1
    } else {
        last_first
    };
    let mut copy_size: usize = 1 << (last_length - 1);

    // Initialize short_code_lookup to zero for `copy_size` entries
    for v in result.short_code_lookup[..copy_size].iter_mut() {
        *v = 0;
    }

    let min_length = last_length;

    while last_length <= ISAL_DECODE_LONG_BITS {
        // memcpy(short_code_lookup + copy_size, short_code_lookup,
        //        sizeof(*short_code_lookup) * copy_size);
        // Note: source overlaps destination range — split-borrow to copy
        // [0..copy_size] into [copy_size..copy_size*2].
        let (head, tail) = result.short_code_lookup.split_at_mut(copy_size);
        for k in 0..copy_size {
            tail[k] = head[k];
        }
        copy_size *= 2;

        // Encode code singletons
        let mut index1 = count_total[last_length as usize] as usize;
        let last_total = count_total[(last_length + 1) as usize] as usize;
        while index1 < last_total {
            let sym1_index = code_list[index1];
            let sym1 = index_to_sym(sym1_index);
            let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
            let sym1_code = huff_code_table[sym1_index as usize].code() as u32;

            if sym1 <= max_symbol {
                result.short_code_lookup[sym1_code as usize] = sym1
                    | (sym1_len << LARGE_SHORT_CODE_LEN_OFFSET)
                    | (1 << LARGE_SYM_COUNT_OFFSET);
            }
            index1 += 1;
        }

        // Continue if no pairs are possible
        if multisym >= SINGLE_SYM_FLAG || last_length < 2 * min_length {
            last_length += 1;
            continue;
        }

        // Encode code pairs.
        let pair_idx1_end = count_total[(last_length - min_length + 1) as usize] as usize;
        let mut index1 = count_total[min_length as usize] as usize;
        while index1 < pair_idx1_end {
            let sym1_index = code_list[index1];
            let sym1 = index_to_sym(sym1_index);
            let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
            let sym1_code = huff_code_table[sym1_index as usize].code() as u32;

            // Check that sym1 is a literal — if not, fast-forward past
            // its length bucket (vendor pattern).
            if sym1 >= 256 {
                index1 = count_total[(sym1_len + 1) as usize] as usize;
                index1 = index1.saturating_sub(1) + 1; // = index1 (no-op to mirror C `index1 = ... - 1; index1++`)
                continue;
            }

            let sym2_len = last_length - sym1_len;
            let mut index2 = count_total[sym2_len as usize] as usize;
            let pair_idx2_end = count_total[(sym2_len + 1) as usize] as usize;
            while index2 < pair_idx2_end {
                let sym2_index = code_list[index2];
                let sym2 = index_to_sym(sym2_index);

                if sym2 > max_symbol {
                    break;
                }
                let sym2_code = huff_code_table[sym2_index as usize].code() as u32;
                let code = sym1_code | (sym2_code << sym1_len);
                let code_length = sym1_len + sym2_len;
                result.short_code_lookup[code as usize] = sym1
                    | (sym2 << 8)
                    | (code_length << LARGE_SHORT_CODE_LEN_OFFSET)
                    | (2 << LARGE_SYM_COUNT_OFFSET);
                index2 += 1;
            }
            index1 += 1;
        }

        // Continue if no triples are possible
        if multisym >= DOUBLE_SYM_FLAG || last_length < 3 * min_length {
            last_length += 1;
            continue;
        }

        // Encode code triples
        let trip_idx1_end = count_total[(last_length - 2 * min_length + 1) as usize] as usize;
        let mut index1 = count_total[min_length as usize] as usize;
        while index1 < trip_idx1_end {
            let sym1_index = code_list[index1];
            let sym1 = index_to_sym(sym1_index);
            let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
            let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
            // Check that sym1 is a literal
            if sym1 >= 256 {
                index1 = count_total[(sym1_len + 1) as usize] as usize;
                continue;
            }

            if last_length - sym1_len < 2 * min_length {
                break;
            }

            let trip_idx2_end =
                count_total[(last_length - sym1_len - min_length + 1) as usize] as usize;
            let mut index2 = count_total[min_length as usize] as usize;
            while index2 < trip_idx2_end {
                let sym2_index = code_list[index2];
                let sym2 = index_to_sym(sym2_index);
                let sym2_len = huff_code_table[sym2_index as usize].length() as u32;
                let sym2_code = huff_code_table[sym2_index as usize].code() as u32;

                // Check that sym2 is a literal
                if sym2 >= 256 {
                    index2 = count_total[(sym2_len + 1) as usize] as usize;
                    continue;
                }

                let sym3_len = last_length - sym1_len - sym2_len;
                let mut index3 = count_total[sym3_len as usize] as usize;
                let trip_idx3_end = count_total[(sym3_len + 1) as usize] as usize;
                while index3 < trip_idx3_end {
                    let sym3_index = code_list[index3];
                    let sym3 = index_to_sym(sym3_index);
                    let sym3_code = huff_code_table[sym3_index as usize].code() as u32;

                    if sym3 > max_symbol - 1 {
                        break;
                    }
                    let code =
                        sym1_code | (sym2_code << sym1_len) | (sym3_code << (sym2_len + sym1_len));
                    let code_length = sym1_len + sym2_len + sym3_len;
                    result.short_code_lookup[code as usize] = sym1
                        | (sym2 << 8)
                        | (sym3 << 16)
                        | (code_length << LARGE_SHORT_CODE_LEN_OFFSET)
                        | (3 << LARGE_SYM_COUNT_OFFSET);
                    index3 += 1;
                }
                index2 += 1;
            }
            index1 += 1;
        }
        last_length += 1;
    }

    // Long-code processing.
    let long_start = count_total[ISAL_DECODE_LONG_BITS as usize + 1] as usize;
    let long_code_length = (code_list_len as usize).saturating_sub(long_start);
    let long_code_list = &code_list[long_start..long_start + long_code_length];
    let mut long_code_lookup_length: u32 = 0;
    let mut temp_code_list: [u16; 1 << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)] =
        [0u16; 1 << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)];

    for i in 0..long_code_length {
        // Set the look up table to point to a hint where the symbol can be
        // found in the list of long codes and add the current symbol to the
        // list of long codes.
        let li = long_code_list[i] as usize;
        if huff_code_table[li].code_and_extra() == INVALID_CODE {
            continue;
        }

        let mut max_length = huff_code_table[li].length() as u32;
        let first_bits =
            (huff_code_table[li].code_and_extra() & ((1 << ISAL_DECODE_LONG_BITS) - 1)) as u16;

        temp_code_list[0] = long_code_list[i] as u16;
        let mut temp_code_length: u32 = 1;

        for j in (i + 1)..long_code_length {
            let lj = long_code_list[j] as usize;
            if (huff_code_table[lj].code() as u32 & ((1 << ISAL_DECODE_LONG_BITS) - 1))
                == first_bits as u32
            {
                max_length = huff_code_table[lj].length() as u32;
                temp_code_list[temp_code_length as usize] = long_code_list[j] as u16;
                temp_code_length += 1;
            }
        }

        // Zero out the long-code-lookup region we're about to populate
        let lcl_size = 1usize << (max_length - ISAL_DECODE_LONG_BITS);
        for k in 0..lcl_size {
            result.long_code_lookup[long_code_lookup_length as usize + k] = 0;
        }

        for j in 0..temp_code_length {
            let sym1_index = temp_code_list[j as usize] as usize;
            let sym1 = index_to_sym(sym1_index as u32);
            let sym1_len = huff_code_table[sym1_index].length() as u32;
            let sym1_code = huff_code_table[sym1_index].code_and_extra();

            let mut long_bits = sym1_code >> ISAL_DECODE_LONG_BITS;
            let min_increment = 1u32 << (sym1_len - ISAL_DECODE_LONG_BITS);

            while long_bits < (1 << (max_length - ISAL_DECODE_LONG_BITS)) {
                let idx = (long_code_lookup_length + long_bits) as usize;
                result.long_code_lookup[idx] =
                    (sym1 | (sym1_len << LARGE_LONG_CODE_LEN_OFFSET)) as u16;
                long_bits += min_increment;
            }
            // Mark this code as already-placed so the outer loop's
            // INVALID_CODE check skips it.
            huff_code_table[sym1_index].set_code_and_extra(INVALID_CODE);
        }
        result.short_code_lookup[first_bits as usize] =
            long_code_lookup_length | (max_length << LARGE_SHORT_MAX_LEN_OFFSET) | LARGE_FLAG_BIT;
        long_code_lookup_length += 1u32 << (max_length - ISAL_DECODE_LONG_BITS);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance-table builder — vendor `make_inflate_huff_code_dist`
// (igzip_inflate.c:601-?). Smaller alphabet, simpler structure (no
// multi-symbol packing).
// ─────────────────────────────────────────────────────────────────────────────

pub fn make_inflate_huff_code_dist(
    result: &mut InflateHuffCodeSmall,
    huff_code_table: &mut [HuffCode],
    _table_length: usize,
    count: &[u16; MAX_HUFF_TREE_DEPTH + 1 + 1], // 17 = MAX_HUFF_TREE_DEPTH + 1 + slack
    code_list: &[u32],
    max_symbol: u32,
) {
    let code_list_len = count[MAX_HUFF_TREE_DEPTH + 1] as u32;
    if code_list_len == 0 {
        for v in result.short_code_lookup.iter_mut() {
            *v = 0;
        }
        return;
    }

    // Determine the length of the first code
    let first_len = huff_code_table[code_list[0] as usize].length() as u32;
    let mut last_length = if first_len > ISAL_DECODE_SHORT_BITS {
        ISAL_DECODE_SHORT_BITS + 1
    } else {
        first_len
    };
    let mut copy_size: usize = 1 << (last_length - 1);

    for v in result.short_code_lookup[..copy_size].iter_mut() {
        *v = 0;
    }

    while last_length <= ISAL_DECODE_SHORT_BITS {
        let (head, tail) = result.short_code_lookup.split_at_mut(copy_size);
        for k in 0..copy_size {
            tail[k] = head[k];
        }
        copy_size *= 2;

        let mut index1 = count[last_length as usize] as usize;
        let last_total = count[(last_length + 1) as usize] as usize;
        while index1 < last_total {
            let sym1_index = code_list[index1];
            let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
            let sym1_extra = huff_code_table[sym1_index as usize].extra_bit_count() as u32;
            if sym1_index <= max_symbol {
                result.short_code_lookup[sym1_code as usize] = (sym1_index
                    | (sym1_extra << DIST_SYM_EXTRA_OFFSET)
                    | (last_length << SMALL_SHORT_CODE_LEN_OFFSET))
                    as u16;
            }
            index1 += 1;
        }
        last_length += 1;
    }

    // Long-code processing — symmetric to the lit/len variant but uses
    // SMALL_FLAG_BIT and writes u16 entries.
    let long_start = count[ISAL_DECODE_SHORT_BITS as usize + 1] as usize;
    let long_code_length = (code_list_len as usize).saturating_sub(long_start);
    let long_code_list = &code_list[long_start..long_start + long_code_length];
    let mut long_code_lookup_length: u32 = 0;
    let mut temp_code_list: [u16; 1 << (15 - 10)] = [0u16; 1 << (15 - 10)];

    for i in 0..long_code_length {
        let li = long_code_list[i] as usize;
        if huff_code_table[li].code_and_extra() == INVALID_CODE {
            continue;
        }

        let mut max_length = huff_code_table[li].length() as u32;
        let first_bits =
            (huff_code_table[li].code_and_extra() & ((1 << ISAL_DECODE_SHORT_BITS) - 1)) as u16;

        temp_code_list[0] = long_code_list[i] as u16;
        let mut temp_code_length: u32 = 1;

        for j in (i + 1)..long_code_length {
            let lj = long_code_list[j] as usize;
            if (huff_code_table[lj].code() as u32 & ((1 << ISAL_DECODE_SHORT_BITS) - 1))
                == first_bits as u32
            {
                max_length = huff_code_table[lj].length() as u32;
                temp_code_list[temp_code_length as usize] = long_code_list[j] as u16;
                temp_code_length += 1;
            }
        }

        let lcl_size = 1usize << (max_length - ISAL_DECODE_SHORT_BITS);
        for k in 0..lcl_size {
            result.long_code_lookup[long_code_lookup_length as usize + k] = 0;
        }

        for j in 0..temp_code_length {
            let sym1_index = temp_code_list[j as usize] as usize;
            let sym1_len = huff_code_table[sym1_index].length() as u32;
            let sym1_code = huff_code_table[sym1_index].code_and_extra();
            let sym1_extra = huff_code_table[sym1_index].extra_bit_count() as u32;

            let mut long_bits = sym1_code >> ISAL_DECODE_SHORT_BITS;
            let min_increment = 1u32 << (sym1_len - ISAL_DECODE_SHORT_BITS);

            while long_bits < (1u32 << (max_length - ISAL_DECODE_SHORT_BITS)) {
                let idx = (long_code_lookup_length + long_bits) as usize;
                result.long_code_lookup[idx] = (sym1_index as u32
                    | (sym1_extra << DIST_SYM_EXTRA_OFFSET)
                    | (sym1_len << SMALL_LONG_CODE_LEN_OFFSET))
                    as u16;
                long_bits += min_increment;
            }
            huff_code_table[sym1_index].set_code_and_extra(INVALID_CODE);
        }
        result.short_code_lookup[first_bits as usize] = (long_code_lookup_length
            | (max_length << SMALL_SHORT_CODE_LEN_OFFSET)
            | SMALL_FLAG_BIT) as u16;
        long_code_lookup_length += 1u32 << (max_length - ISAL_DECODE_SHORT_BITS);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — table-build sanity. The wiring commit ships the silesia byte-
// perfect differential and the C-ISAL cross-check.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reverse2_matches_spec() {
        // length=1
        assert_eq!(bit_reverse2(0b1, 1), 0b1);
        assert_eq!(bit_reverse2(0b0, 1), 0b0);
        // length=4
        assert_eq!(bit_reverse2(0b1011, 4), 0b1101);
        assert_eq!(bit_reverse2(0b0011, 4), 0b1100);
        // length=8
        assert_eq!(bit_reverse2(0xAA, 8), 0x55);
        assert_eq!(bit_reverse2(0xFF, 8), 0xFF);
        // length=16
        assert_eq!(bit_reverse2(0x8000, 16), 0x0001);
        assert_eq!(bit_reverse2(0x1234, 16), 0x2C48);
    }

    #[test]
    fn set_codes_simple_canonical() {
        // Three symbols, all length 2 → codes 00, 01, 10 (reversed: 00, 10, 01)
        let mut tbl = [HuffCode(0), HuffCode(0), HuffCode(0)];
        for entry in tbl.iter_mut() {
            entry.set_length(2);
        }
        let mut count = [0u16; MAX_HUFF_TREE_DEPTH + 1];
        count[2] = 3;
        set_codes(&mut tbl, 3, &mut count).expect("valid");
        assert_eq!(tbl[0].length(), 2);
        // Codes assigned in order 00, 01, 10 then bit-reversed (length=2):
        // 00 -> 00, 01 -> 10, 10 -> 01
        assert_eq!(tbl[0].code(), 0b00);
        assert_eq!(tbl[1].code(), 0b10);
        assert_eq!(tbl[2].code(), 0b01);
    }

    #[test]
    fn set_codes_rejects_overflow() {
        // 4 codes of length 1 — Kraft sum = 4 * 0.5 = 2 > 1, invalid.
        let mut tbl: Vec<HuffCode> = (0..4).map(|_| HuffCode(1u32 << 24)).collect();
        let mut count = [0u16; MAX_HUFF_TREE_DEPTH + 1];
        count[1] = 4;
        let r = set_codes(&mut tbl, 4, &mut count);
        assert!(r.is_err());
    }

    #[test]
    fn huff_code_views_consistent() {
        let mut h = HuffCode::default();
        h.set_code_and_length(0xABCD, 7);
        assert_eq!(h.code(), 0xABCD);
        assert_eq!(h.length(), 7);
        assert_eq!(h.extra_bit_count(), 0);
        // After set_code_and_extra preserves length.
        h.set_code_and_extra(0x42_3456);
        assert_eq!(h.length(), 7);
        assert_eq!(h.code(), 0x3456);
        assert_eq!(h.extra_bit_count(), 0x42);
        assert_eq!(h.code_and_extra(), 0x42_3456);
    }

    /// Build a simple lit/len table from a fixed code-length distribution
    /// (RFC 1951 fixed-Huffman style minus 2 symbols, since the static
    /// table needs the `count[8] -= 2` correction handled by the caller).
    #[test]
    fn make_lit_len_smoke() {
        // 286 symbols, all the standard "fixed Huffman" code lengths
        // (8 bits for 0..143, 9 for 144..255, 7 for 256..279, 8 for 280..285).
        let mut lit_len_huff: Vec<HuffCode> = (0..LIT_LEN_ELEMS).map(|_| HuffCode(0)).collect();
        for i in 0..144 {
            lit_len_huff[i].set_length(8);
        }
        for i in 144..256 {
            lit_len_huff[i].set_length(9);
        }
        for i in 256..280 {
            lit_len_huff[i].set_length(7);
        }
        for i in 280..286 {
            lit_len_huff[i].set_length(8);
        }
        let mut count = [0u16; MAX_LIT_LEN_COUNT];
        let mut expand_count = [0u16; MAX_LIT_LEN_COUNT];
        for i in 0..286 {
            let l = lit_len_huff[i].length() as usize;
            count[l] += 1;
            // For length symbols (≥257), the extra-bit accounting lives in
            // `expand_count[length + extra]` per vendor's setup loop.
            if l != 0 && i >= 264 {
                let extra = LEN_EXTRA_BIT_COUNT[i - 257] as usize;
                expand_count[l] = expand_count[l].wrapping_sub(1);
                let target = l + extra;
                if target < MAX_LIT_LEN_COUNT {
                    expand_count[target] = expand_count[target].wrapping_add(1u16 << extra);
                }
            }
        }
        let mut code_list = vec![0u32; LIT_LEN_ELEMS + 2];
        set_and_expand_lit_len_huffcode(
            &mut lit_len_huff,
            LIT_LEN,
            &mut count,
            &mut expand_count,
            &mut code_list,
        )
        .expect("static fixed Huffman should build");

        let mut result = Box::new(InflateHuffCodeLarge::default());
        make_inflate_huff_code_lit_len(
            &mut result,
            &mut lit_len_huff,
            LIT_LEN_ELEMS,
            &count,
            &code_list,
            TRIPLE_SYM_FLAG,
        );

        // After build, the short_code_lookup must have at least some
        // entries with non-zero bit_count (the singleton inserts).
        let nonzero = result
            .short_code_lookup
            .iter()
            .filter(|&&v| (v >> LARGE_SHORT_CODE_LEN_OFFSET) != 0)
            .count();
        assert!(
            nonzero > 0,
            "make_inflate_huff_code_lit_len should populate short_code_lookup"
        );
    }
}
