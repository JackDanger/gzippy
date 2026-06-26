//! Pure-Rust port of ISA-L's `inflate_huff_code_{large,small}` LUT format
//! plus the `set_codes` / `set_and_expand_lit_len_huffcode` /
//! `make_inflate_huff_code_lit_len` / `make_inflate_huff_code_dist` builders.
//!
//! ## Why this exists
//!
//! `isal_huffman.rs` wraps the C versions via `isal::isal_sys::*` — only
//! available on `feature = "isal-compression", target_arch = "x86_64"`. The
//! pure-rust-inflate build was falling back to a vendor MultiCached.hpp-style
//! port, which per the May 27 2026 neurotic perf was the dominant cost in the
//! marker-phase bootstrap decoder (~7.65% of total CPU, with the bootstrap
//! caller at 81% children-time). That cache used a simpler 11-bit format that
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
/// Bits 0..25  = packed symbols (up to 3 × 8-bit; bit 24 is a triple's sym3
///              bit 8 — pure DATA, no build-time class flag, igzip shape)
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

// NOTE (kernel-converge, faithful-to-igzip): there is NO build-time
// trailing-class flag. igzip's `make_inflate_huff_code_lit_len`
// (igzip_inflate.c:387-599) sets no such bit; its decoder classifies the
// trailing packed symbol at RUNTIME (`cmp next_sym2, 256`). gz previously
// carried a `LARGE_TRAILING_NONLIT_FLAG` in bit 24, dead since the kernel
// moved to the late `cmp {t5}, 256` discriminator (asm_kernel.rs run_contig);
// it has been removed to match igzip's table build exactly (the per-entry OR
// igzip does not do). Bit 24 is now pure DATA — a triple's sym3 bit 8 — and
// the decoders' `& 0xFFFF` trailing recovery is byte-identical to before.
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

// ─────────────────────────────────────────────────────────────────────────────
// PROFILING-ONLY sub-step timers (feature = "profile-rebuild"). RDTSC cycle
// accumulators that locate the expensive per-block build sub-step. Never
// compiled into the gate binary. Relative shares are load-immune within one run.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "profile-rebuild")]
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static N_BLOCKS: AtomicU64 = AtomicU64::new(0);
    pub static C_ZERO: AtomicU64 = AtomicU64::new(0); // rebuild zeroing + count loop
    pub static C_EXPAND: AtomicU64 = AtomicU64::new(0); // set_and_expand_lit_len_huffcode
    pub static C_MAKE_DOUBLE: AtomicU64 = AtomicU64::new(0); // doubling memcpy + singletons
    pub static C_MAKE_PAIR: AtomicU64 = AtomicU64::new(0); // pair fill
    pub static C_MAKE_TRIPLE: AtomicU64 = AtomicU64::new(0); // triple fill
    pub static C_MAKE_LONG: AtomicU64 = AtomicU64::new(0); // long-code phase
    pub static C_LONG_SCAN: AtomicU64 = AtomicU64::new(0); // O(n^2) prefix-group scan within long
    pub static N_LONG_CODES: AtomicU64 = AtomicU64::new(0); // sum of long_code_length

    #[inline(always)]
    pub fn rdtsc() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[inline(always)]
    pub fn add(c: &AtomicU64, d: u64) {
        c.fetch_add(d, Ordering::Relaxed);
    }
}

/// Dump the profile-rebuild sub-step cycle shares to stderr. No-op unless built
/// with `--features profile-rebuild`.
pub fn dump_rebuild_profile() {
    #[cfg(feature = "profile-rebuild")]
    {
        use prof::*;
        use std::sync::atomic::Ordering::Relaxed;
        let n = N_BLOCKS.load(Relaxed).max(1);
        let z = C_ZERO.load(Relaxed);
        let e = C_EXPAND.load(Relaxed);
        let d = C_MAKE_DOUBLE.load(Relaxed);
        let p = C_MAKE_PAIR.load(Relaxed);
        let t = C_MAKE_TRIPLE.load(Relaxed);
        let l = C_MAKE_LONG.load(Relaxed);
        let tot = (z + e + d + p + t + l).max(1);
        eprintln!("REBUILD_PROFILE blocks={n}");
        let row = |name: &str, c: u64| {
            eprintln!(
                "  {name:<14} cyc/blk={:>8.1}  share={:>5.1}%",
                c as f64 / n as f64,
                100.0 * c as f64 / tot as f64
            );
        };
        row("zero+count", z);
        row("set_expand", e);
        row("make_double", d);
        row("make_pair", p);
        row("make_triple", t);
        row("make_long", l);
        let scan = C_LONG_SCAN.load(Relaxed);
        let nlong = N_LONG_CODES.load(Relaxed);
        eprintln!(
            "    (long_scan    cyc/blk={:>8.1}  share={:>5.1}%  avg_long_codes/blk={:.1})",
            scan as f64 / n as f64,
            100.0 * scan as f64 / tot as f64,
            nlong as f64 / n as f64
        );
        eprintln!("  TOTAL          cyc/blk={:>8.1}", tot as f64 / n as f64);
    }
}
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
        count_total += count[i] as u32 + count_tmp;
        count_tmp = expand_count[i + 1] as u32;
        expand_count[i + 1] = count_total as u16;
        next_code[i + 1] = (next_code[i] + count[i] as u32) << 1;
        i += 1;
    }

    count_tmp += count[i] as u32;

    while i < MAX_LIT_LEN_COUNT - 1 {
        count_total += count_tmp;
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
//
// LONG-CODE GROUPING (perf, byte-exact divergence from the vendor's O(n²) shape):
// igzip groups long codes that share the low-12-bit prefix and emits one
// long_code_lookup sub-table per group, in the prefix's first-appearance order.
// The faithful port did this with a quadratic inner re-scan (for each group
// leader, scan all later long codes) — measured (RDTSC sub-step profile, Intel
// i7-13700T) as the #1 per-block build cost: ~4.6 kcyc/blk = 32% of the build on
// pigz (~69 long codes/blk) and 28% on silesia. We replace it with an O(n)
// single-pass bucketed grouping using a thread-local generation-stamped 4096-slot
// table (gen tag in the high 20 bits, last-member list-index in the low 12 — no
// per-block init, reset only on the ~1M-block gen wrap). The emitted groups, the
// per-group member order (ascending list order), the max-length, and the
// allocation order are all identical to the quadratic version → the produced
// short_code_lookup / long_code_lookup are byte-for-byte identical, so the
// triple-pack DECODE format is unchanged.
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-local scratch for the O(n) long-code prefix grouping in
/// [`make_inflate_huff_code_lit_len`]. The 4096-slot table is generation-stamped
/// so it never needs a per-block memset; `lc[fb] = (gen << 12) | last_list_index`.
struct LongPrefixScratch {
    lc: Box<[u32; 1 << ISAL_DECODE_LONG_BITS]>,
    gen: u32,
}
impl LongPrefixScratch {
    fn new() -> Self {
        Self {
            lc: Box::new([0u32; 1 << ISAL_DECODE_LONG_BITS]),
            gen: 0,
        }
    }
}
thread_local! {
    static LONG_PREFIX_SCRATCH: std::cell::RefCell<LongPrefixScratch> =
        std::cell::RefCell::new(LongPrefixScratch::new());
}

/// Returns `false` if the (already over-subscription-screened) table still
/// drives a LUT write out of range. A well-formed canonical code never does —
/// proven by every real block decoding correctly — so a `false` here can only
/// be an exotic bad speculative seed whose length set passes the cheap Kraft
/// screen yet corrupts ISA-L's expanded code/bucket arithmetic. The caller
/// rejects the table and re-syncs via `resumable_resync`. (Vendor igzip omits
/// these checks because it only runs on headers `read_header` already
/// validated; the speculative single-member path decodes from *guessed* offsets
/// so the builder must self-guard.)
#[must_use]
pub fn make_inflate_huff_code_lit_len(
    result: &mut InflateHuffCodeLarge,
    huff_code_table: &mut [HuffCode],
    _table_length: usize,
    count_total: &[u16; MAX_LIT_LEN_COUNT],
    code_list: &[u32],
    multisym: u32,
) -> bool {
    let max_symbol: u32 = MAX_LIT_LEN_SYM;
    let short_len = result.short_code_lookup.len();
    let long_len = result.long_code_lookup.len();

    let code_list_len = count_total[MAX_LIT_LEN_COUNT - 1] as u32;
    if code_list_len == 0 {
        // igzip: memset(short_code_lookup, 0, sizeof(short_code_lookup))
        result.short_code_lookup.fill(0);
        return true;
    }

    // Shortest length with at least one code (`count_total` is cumulative).
    let mut last_length = 0u32;
    for l in 1..MAX_LIT_LEN_COUNT as u32 {
        if count_total[(l + 1) as usize] > count_total[l as usize] {
            last_length = l;
            break;
        }
    }
    if last_length == 0 {
        return true;
    }
    if last_length > ISAL_DECODE_LONG_BITS {
        last_length = ISAL_DECODE_LONG_BITS + 1;
    }
    let mut copy_size: usize = 1 << (last_length - 1);

    // Initialize short_code_lookup to zero for `copy_size` entries
    // igzip: memset(short_code_lookup, 0x00, copy_size * sizeof(*short_code_lookup))
    result.short_code_lookup[..copy_size].fill(0);

    let min_length = last_length;

    while last_length <= ISAL_DECODE_LONG_BITS {
        #[cfg(feature = "profile-rebuild")]
        let _ma = prof::rdtsc();
        // memcpy(short_code_lookup + copy_size, short_code_lookup,
        //        sizeof(*short_code_lookup) * copy_size);
        // Note: source overlaps destination range — split-borrow to copy
        // [0..copy_size] into [copy_size..copy_size*2].
        let (head, tail) = result.short_code_lookup.split_at_mut(copy_size);
        // igzip: memcpy(short_code_lookup + copy_size, short_code_lookup,
        //               sizeof(*short_code_lookup) * copy_size)
        tail[..copy_size].copy_from_slice(&head[..copy_size]);
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
                if sym1_code as usize >= short_len {
                    return false;
                }
                result.short_code_lookup[sym1_code as usize] = sym1
                    | (sym1_len << LARGE_SHORT_CODE_LEN_OFFSET)
                    | (1 << LARGE_SYM_COUNT_OFFSET);
            }
            index1 += 1;
        }

        #[cfg(feature = "profile-rebuild")]
        let _mb = prof::rdtsc();
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::C_MAKE_DOUBLE, _mb.wrapping_sub(_ma));

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
                if code as usize >= short_len {
                    return false;
                }
                result.short_code_lookup[code as usize] = sym1
                    | (sym2 << 8)
                    | (code_length << LARGE_SHORT_CODE_LEN_OFFSET)
                    | (2 << LARGE_SYM_COUNT_OFFSET);
                index2 += 1;
            }
            index1 += 1;
        }

        #[cfg(feature = "profile-rebuild")]
        let _mc = prof::rdtsc();
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::C_MAKE_PAIR, _mc.wrapping_sub(_mb));

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
                    if code as usize >= short_len {
                        return false;
                    }
                    // For a triple, sym3 (bits 16..25) occupies the top of the
                    // 25-bit packed-symbol field; a `sym3 >= 256` trailing sets
                    // bit 24 NATURALLY via `sym3 << 16`, so it survives the
                    // decoders' `& 0xFFFF` trailing recovery (igzip same shape).
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
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::C_MAKE_TRIPLE, prof::rdtsc().wrapping_sub(_mc));
        last_length += 1;
    }

    #[cfg(feature = "profile-rebuild")]
    let _ml = prof::rdtsc();
    // Long-code processing.
    let long_start = count_total[ISAL_DECODE_LONG_BITS as usize + 1] as usize;
    let long_code_length = (code_list_len as usize).saturating_sub(long_start);
    #[cfg(feature = "profile-rebuild")]
    prof::add(&prof::N_LONG_CODES, long_code_length as u64);
    let long_code_list = &code_list[long_start..long_start + long_code_length];
    let mut long_code_lookup_length: u32 = 0;
    let mut temp_code_list: [u16; 1 << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)] =
        [0u16; 1 << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)];

    // O(n) prefix-group construction (byte-exact replacement of the original
    // O(n²) inner re-scan — see module note above). `chain[i]` links the next
    // member (in ascending long_code_list order) sharing i's 12-bit prefix;
    // `group_head[g]` is the head list-index of group g in first-appearance order.
    const CHAIN_END: u32 = u32::MAX;
    const LC_LAST_MASK: u32 = (1 << ISAL_DECODE_LONG_BITS) - 1; // low 12 bits = list index
    let mut chain = [CHAIN_END; LIT_LEN_ELEMS + 2];
    let mut group_head = [0u32; LIT_LEN_ELEMS + 2];
    let mut n_groups = 0usize;

    #[cfg(feature = "profile-rebuild")]
    let _ls = prof::rdtsc();
    // Pass 1: bucket every long code into its prefix chain in ONE pass.
    LONG_PREFIX_SCRATCH.with(|cell| {
        let s = &mut *cell.borrow_mut();
        s.gen = s.gen.wrapping_add(1);
        if s.gen >= (1u32 << (32 - ISAL_DECODE_LONG_BITS)) {
            // gen no longer fits above the 12-bit last-index field → reset.
            s.lc.fill(0);
            s.gen = 1;
        }
        let gen_tag = s.gen << ISAL_DECODE_LONG_BITS;
        let lc = &mut *s.lc;
        for i in 0..long_code_length {
            let li = long_code_list[i] as usize;
            let fb = (huff_code_table[li].code() as u32) & LC_LAST_MASK;
            let slot = &mut lc[fb as usize];
            if (*slot & !LC_LAST_MASK) != gen_tag {
                // First member of this prefix this block → new group.
                group_head[n_groups] = i as u32;
                n_groups += 1;
            } else {
                let prev = (*slot & LC_LAST_MASK) as usize;
                chain[prev] = i as u32;
            }
            *slot = gen_tag | (i as u32);
        }
    });
    #[cfg(feature = "profile-rebuild")]
    prof::add(&prof::C_LONG_SCAN, prof::rdtsc().wrapping_sub(_ls));

    // Pass 2: emit groups in first-appearance order (byte-identical layout).
    for g in 0..n_groups {
        let head = group_head[g] as usize;
        let head_li = long_code_list[head] as usize;
        let first_bits =
            (huff_code_table[head_li].code_and_extra() & ((1 << ISAL_DECODE_LONG_BITS) - 1)) as u16;

        // Walk the chain: members in ascending list order; max_length = last
        // (the chain is sorted ascending so the final member is the longest).
        let mut max_length;
        let mut temp_code_length: u32 = 0;
        let mut walk = head;
        loop {
            if temp_code_length as usize >= temp_code_list.len() {
                return false;
            }
            let m_li = long_code_list[walk] as usize;
            temp_code_list[temp_code_length as usize] = m_li as u16;
            max_length = huff_code_table[m_li].length() as u32;
            temp_code_length += 1;
            let nxt = chain[walk];
            if nxt == CHAIN_END {
                break;
            }
            walk = nxt as usize;
        }

        // Zero out the long-code-lookup region we're about to populate
        let lcl_size = 1usize << (max_length - ISAL_DECODE_LONG_BITS);
        if long_code_lookup_length as usize + lcl_size > long_len {
            return false;
        }
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
                if idx >= long_len {
                    return false;
                }
                result.long_code_lookup[idx] =
                    (sym1 | (sym1_len << LARGE_LONG_CODE_LEN_OFFSET)) as u16;
                long_bits += min_increment;
            }
            // Mark this code as already-placed (preserves the vendor post-state;
            // the chain dedup means it is never re-read this call).
            huff_code_table[sym1_index].set_code_and_extra(INVALID_CODE);
        }
        if first_bits as usize >= short_len {
            return false;
        }
        result.short_code_lookup[first_bits as usize] =
            long_code_lookup_length | (max_length << LARGE_SHORT_MAX_LEN_OFFSET) | LARGE_FLAG_BIT;
        long_code_lookup_length += 1u32 << (max_length - ISAL_DECODE_LONG_BITS);
    }
    #[cfg(feature = "profile-rebuild")]
    prof::add(&prof::C_MAKE_LONG, prof::rdtsc().wrapping_sub(_ml));
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance-table builder — vendor `make_inflate_huff_code_dist`
// (igzip_inflate.c:601-?). Smaller alphabet, simpler structure (no
// multi-symbol packing).
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `false` if a (Kraft-screened) table still drives a LUT write out of
/// range — same self-guard as [`make_inflate_huff_code_lit_len`], required for
/// the same reason: a bad speculative seed can pass the cheap over-subscription
/// screen yet overflow ISA-L's code arithmetic, here into the much smaller
/// distance `short_code_lookup` / `long_code_lookup`. A valid canonical code
/// never trips these (every real block decodes correctly); a trip means garbage
/// → reject → re-sync via `resumable_resync`.
#[must_use]
pub fn make_inflate_huff_code_dist(
    result: &mut InflateHuffCodeSmall,
    huff_code_table: &mut [HuffCode],
    _table_length: usize,
    count: &[u16; MAX_HUFF_TREE_DEPTH + 1 + 1], // 17 = MAX_HUFF_TREE_DEPTH + 1 + slack
    code_list: &[u32],
    max_symbol: u32,
) -> bool {
    let short_len = result.short_code_lookup.len();
    let long_len = result.long_code_lookup.len();
    let code_list_len = count[MAX_HUFF_TREE_DEPTH + 1] as u32;
    if code_list_len == 0 {
        // igzip: memset(short_code_lookup, 0, sizeof(short_code_lookup))
        result.short_code_lookup.fill(0);
        return true;
    }

    // Shortest length with at least one code. `count` is cumulative (start
    // index per length), so `count[l]` is often 0 even when length `l` is
    // present — use `count[l + 1] > count[l]`, not `count[l] > 0`.
    let mut first_len = 0u32;
    for l in 1..=MAX_HUFF_TREE_DEPTH as u32 {
        if count[(l + 1) as usize] > count[l as usize] {
            first_len = l;
            break;
        }
    }
    if first_len == 0 {
        return true;
    }
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
        // igzip: memcpy(short_code_lookup + copy_size, short_code_lookup,
        //               sizeof(*short_code_lookup) * copy_size)
        tail[..copy_size].copy_from_slice(&head[..copy_size]);
        copy_size *= 2;

        let mut index1 = count[last_length as usize] as usize;
        let last_total = count[(last_length + 1) as usize] as usize;
        while index1 < last_total {
            let sym1_index = code_list[index1];
            let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
            let sym1_extra = huff_code_table[sym1_index as usize].extra_bit_count() as u32;
            if sym1_index <= max_symbol {
                if sym1_code as usize >= short_len {
                    return false;
                }
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
                if temp_code_length as usize >= temp_code_list.len() {
                    return false;
                }
                temp_code_list[temp_code_length as usize] = long_code_list[j] as u16;
                temp_code_length += 1;
            }
        }

        let lcl_size = 1usize << (max_length - ISAL_DECODE_SHORT_BITS);
        if long_code_lookup_length as usize + lcl_size > long_len {
            return false;
        }
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
                if idx >= long_len {
                    return false;
                }
                result.long_code_lookup[idx] = (sym1_index as u32
                    | (sym1_extra << DIST_SYM_EXTRA_OFFSET)
                    | (sym1_len << SMALL_LONG_CODE_LEN_OFFSET))
                    as u16;
                long_bits += min_increment;
            }
            huff_code_table[sym1_index].set_code_and_extra(INVALID_CODE);
        }
        if first_bits as usize >= short_len {
            return false;
        }
        result.short_code_lookup[first_bits as usize] = (long_code_lookup_length
            | (max_length << SMALL_SHORT_CODE_LEN_OFFSET)
            | SMALL_FLAG_BIT) as u16;
        long_code_lookup_length += 1u32 << (max_length - ISAL_DECODE_SHORT_BITS);
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder types — pure-rust analogs of `IsalLitLenCode` / `IsalDistCode`
// from `isal_huffman.rs`. The `decode` bodies are literal Rust ports of
// `HuffmanCodingISAL::decode` and `HuffmanCodingDistanceISAL::decode`
// (vendor `HuffmanCodingISAL.hpp:94-183` /
// `HuffmanCodingDistanceISAL.hpp:?-?`). The TABLE BUILDERS use the pure-
// Rust functions above instead of FFI, so this type is available on
// builds that don't link ISA-L.
// ─────────────────────────────────────────────────────────────────────────────

/// Decode result — symbol, packed sym count (1..=3), bits consumed.
/// Matches the layout of `isal_huffman::DecodedSymbol`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedSymbol {
    pub symbol: u32,
    pub sym_count: u32,
    pub bit_count: u32,
}

/// Pure-rust analog of `IsalLitLenCode` (isal_huffman.rs:86-243). The
/// decode body is identical (already pure Rust); the rebuild path uses
/// pure-rust `set_and_expand_lit_len_huffcode` +
/// `make_inflate_huff_code_lit_len` instead of ISA-L FFI.
/// `#[repr(C)]` + `table` FIRST: element A (igzip single-state-base register
/// discipline) co-locates the litlen short+long LUT INLINE inside the boxed
/// `AsmState` (asm_kernel.rs) so the asm addresses it `[{ctx}+LIT_OFF+idx*4]`
/// off the ONE `ctx` base (igzip `[state+_lit_huff_code+...]`,
/// igzip_lib.h:515-524). The table is UN-BOXED (was `Box<InflateHuffCodeLarge>`)
/// so its bytes are physically inline in the owning `AsmState`; `table` is the
/// first field at offset 0 so `offset_of!(LutLitLenCode, table) == 0`.
#[repr(C)]
pub struct LutLitLenCode {
    pub table: InflateHuffCodeLarge,
    lit_and_dist_huff: Box<[HuffCode; LIT_LEN_ELEMS]>,
    code_list: Box<[u32; LIT_LEN_ELEMS + 2]>,
    valid: bool,
    /// Per-block table-build cache: the code-length key of the LAST
    /// successfully built table. When the next block presents a byte-identical
    /// key (+ same `multisym`), `self.table` is already correct and the rebuild
    /// is skipped (see `tbuild_cache` in marker_inflate.rs). `cache_key_len ==
    /// usize::MAX` means "no cached table" (forces a miss).
    cache_key: Box<[u8; 290]>,
    cache_key_len: usize,
    cache_multisym: u32,
}

impl LutLitLenCode {
    /// Resident heap footprint (boxed LUT + code-length buffers). Used by the
    /// debug-only `mem_stats` instrument for the cache-residency mandate.
    /// Counters only; never mutates decode state.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<InflateHuffCodeLarge>()
            + LIT_LEN_ELEMS * std::mem::size_of::<HuffCode>()
            + (LIT_LEN_ELEMS + 2) * std::mem::size_of::<u32>()
    }

    pub fn new_empty() -> Self {
        Self {
            table: InflateHuffCodeLarge::default(),
            lit_and_dist_huff: Box::new([HuffCode::default(); LIT_LEN_ELEMS]),
            code_list: Box::new([0u32; LIT_LEN_ELEMS + 2]),
            valid: false,
            cache_key: Box::new([0u8; 290]),
            cache_key_len: usize::MAX,
            cache_multisym: 0,
        }
    }

    /// Rebuild from a slice of code lengths IN-PLACE — reuses the
    /// table/buffer allocations. Returns `true` on success. On failure
    /// `is_valid() == false`. Mirror of
    /// `isal_huffman.rs::IsalLitLenCode::rebuild_from` (line 119-175) but
    /// dispatches to the pure-rust builders defined above.
    pub fn rebuild_from(&mut self, code_lengths: &[u8]) -> bool {
        self.rebuild_from_multisym(code_lengths, TRIPLE_SYM_FLAG)
    }

    /// As `rebuild_from`, but with a caller-chosen multi-symbol packing flag.
    /// Production decode always uses `rebuild_from` (TRIPLE) — this is
    /// byte-identical for `multisym == TRIPLE_SYM_FLAG`. The bare-kernel
    /// removal-oracle (examples/streaming_thin.rs `igzipbare`) uses
    /// `SINGLE_SYM_FLAG` so igzip's `_04` reads unambiguous single-symbol
    /// entries (its asm speculative-literal path mis-handles gz's TRIPLE pack).
    pub fn rebuild_from_multisym(&mut self, code_lengths: &[u8], multisym: u32) -> bool {
        #[cfg(feature = "profile-rebuild")]
        let _t0 = prof::rdtsc();
        // Per-block table-build cache: if the previous successfully built table
        // had a byte-identical code-length key and the same multisym packing,
        // `self.table` is already the exact table this call would rebuild
        // (rebuild is a deterministic function of (code_lengths, multisym), and
        // decode only reads the table). Skip the rebuild — byte-identical, and
        // ~9.5% of the logs-T1 wall per the table-build perturbation gate.
        #[cfg(pure_inflate_decode)]
        {
            use crate::decompress::parallel::marker_inflate::tbuild_cache;
            use std::sync::atomic::Ordering;
            tbuild_cache::note_key(code_lengths, multisym);
            if tbuild_cache::cache_enabled() {
                if self.valid
                    && multisym == self.cache_multisym
                    && self.cache_key_len == code_lengths.len()
                    && self.cache_key[..code_lengths.len()] == *code_lengths
                {
                    tbuild_cache::HITS.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                tbuild_cache::MISSES.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.valid = false;
        // Allow up to 288 entries: 286 for dynamic-Huffman (LIT_LEN) plus
        // symbols 286 and 287 for fixed-Huffman participation (RFC 1951
        // §3.2.6 says these "should never actually appear in compressed
        // data, but participate in the code construction"). Capping at
        // LIT_LEN=286 caused fixed-Huffman tables to omit 2 length-8
        // codes, shifting next_code[9] by 2 → every 9-bit literal symbol
        // (144..255) decoded with an off-by-4 byte value.
        if code_lengths.len() > 288 {
            return false;
        }

        // Reset only the entries we'll touch (mirror of vendor's
        // per-call `lit_and_dist_huff[i].code_and_length = 0` loop).
        //
        // CONVERGE toward igzip (NIGHT38): igzip never separately clears the
        // huff table here — `set_and_expand_lit_len_huffcode` memsets ONLY the
        // length/expansion region `[ISAL_DEF_LIT_SYMBOLS..LIT_LEN_ELEMS]`
        // (igzip_inflate.c:333-334) and relies on the literal read-loop
        // (igzip_inflate.c:338) + the `[257..286]` length snapshot
        // (igzip_inflate.c:331). We must keep zeroing `[0..LIT_LEN]` because
        // (a) the literal read-loop (`set_and_expand`, this file :390) scans
        //     `[0..ISAL_DEF_LIT_SYMBOLS]` and a speculative seed can leave
        //     `split < 257`, and (b) the length snapshot reads `[257..LIT_LEN]`
        //     BEFORE `set_and_expand` re-clears it (:385). But `[LIT_LEN..
        //     LIT_LEN_ELEMS]` is REDUNDANT: on the success path `set_and_expand`
        //     unconditionally clears `[257..LIT_LEN_ELEMS]` (:385) before the
        //     expansion writes it and before `make_inflate` reads it; on the
        //     over-subscription failure path `set_and_expand` returns Err and
        //     `make_inflate` is never called, so the tail is never observed.
        //     ⇒ clearing `[0..LIT_LEN]` is byte-identical to clearing all
        //     LIT_LEN_ELEMS on EVERY input, and drops 228 redundant per-block
        //     writes off the shared table-build (NIGHT28/35: on the T1 wall).
        for h in self.lit_and_dist_huff[..LIT_LEN].iter_mut() {
            h.0 = 0;
        }
        // BUGFIX: `code_list` is a PERSISTENT, reused box. Fixed-Huffman
        // blocks populate fewer entries than a prior dynamic block left
        // behind, so stale tail slots (e.g. the 2 length-8 fixed slots)
        // can hold an out-of-range prior index → `make_inflate_huff_code_*`
        // rejects a valid stream as `InvalidCodeLengths`. The C buffer is
        // freshly built each call; replicate that by clearing it here.
        for v in self.code_list.iter_mut() {
            *v = 0;
        }
        let mut lit_count: [u16; MAX_LIT_LEN_COUNT] = [0; MAX_LIT_LEN_COUNT];
        let mut lit_expand_count: [u16; MAX_LIT_LEN_COUNT] = [0; MAX_LIT_LEN_COUNT];

        for (i, &length) in code_lengths.iter().enumerate() {
            if (length as usize) >= MAX_LIT_LEN_COUNT {
                return false;
            }
            lit_count[length as usize] += 1;
            self.lit_and_dist_huff[i].set_length(length);
            // Length-extra accounting for symbols ≥ 264 (length codes
            // with extra bits per RFC 1951 §3.2.5). The wrapping_sub /
            // wrapping_add mirror the C's `expand_count[l] -= 1;
            // expand_count[target] += 1 << extra` pattern where the
            // initial value is 0 (so -1 wraps to 0xFFFF and the +N
            // later wraps back). This is intentional in the C, not a
            // bug — `set_and_expand_lit_len_huffcode` consumes the
            // wrapping algebra in its setup loop.
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

        let table_len = code_lengths.len();
        #[cfg(feature = "profile-rebuild")]
        let _t1 = prof::rdtsc();
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::C_ZERO, _t1.wrapping_sub(_t0));
        if set_and_expand_lit_len_huffcode(
            &mut self.lit_and_dist_huff[..],
            table_len,
            &mut lit_count,
            &mut lit_expand_count,
            &mut self.code_list[..],
        )
        .is_err()
        {
            return false;
        }
        #[cfg(feature = "profile-rebuild")]
        let _t2 = prof::rdtsc();
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::C_EXPAND, _t2.wrapping_sub(_t1));

        if !make_inflate_huff_code_lit_len(
            &mut self.table,
            &mut self.lit_and_dist_huff[..],
            LIT_LEN_ELEMS,
            &lit_count,
            &self.code_list[..],
            multisym,
        ) {
            return false;
        }
        #[cfg(feature = "profile-rebuild")]
        prof::add(&prof::N_BLOCKS, 1);

        self.valid = true;
        // Store the cache key for the next block (byte-identical headers skip
        // the rebuild). Bounded by the `code_lengths.len() > 288` guard above.
        #[cfg(pure_inflate_decode)]
        {
            let n = code_lengths.len();
            if n <= self.cache_key.len() {
                self.cache_key[..n].copy_from_slice(code_lengths);
                self.cache_key_len = n;
                self.cache_multisym = multisym;
            } else {
                self.cache_key_len = usize::MAX;
            }
        }
        true
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Decode one symbol packet (up to 3 packed literals or 1 length).
    /// LITERAL port of `isal_huffman.rs::IsalLitLenCode::decode`
    /// (line 196-237) — same algorithm, same constants, no behavioral
    /// change. The table is layout-equivalent (`InflateHuffCodeLarge`).
    ///
    /// `#[inline(always)]`: the C++ vendor wrapper inlines through this
    /// to fuse with the marker hot loop. Plain `#[inline]` left the
    /// function as a separate symbol consuming ~3.25% of cycles via
    /// call overhead on the perf profile that motivated this port.
    #[inline(always)]
    pub fn decode(
        &self,
        bits: &mut crate::decompress::inflate::consume_first_decode::Bits<'_>,
    ) -> DecodedSymbol {
        if bits.available() < 32 {
            bits.refill();
        }
        self.decode_prefilled(bits)
    }

    /// P3.5 c4: backstop-free decode for call sites that sit IMMEDIATELY
    /// after a refill (unconditional or `< 48`-threshold). Removes the
    /// `available() < 32` load+branch from the per-symbol critical chain.
    ///
    /// Byte-exact proof (why eliding the backstop cannot change a decode):
    /// at such a site either (a) the threshold refill was NOT taken, so
    /// `bitsleft >= 48 >= 32` and the backstop is dead; or (b) a refill just
    /// ran — fast path leaves `bitsleft >= 56`; slow path
    /// (`refill_slow_with_bits`) fills until `bits > 56` or input is
    /// exhausted, so `available() < 32` afterwards implies `pos == data.len()`
    /// and a second (backstop) refill appends NOTHING — fast path needs
    /// `pos + 8 <= len` (false, pos only grew) and the slow loop re-terminates
    /// immediately. In every case the backstop is a no-op; the lookup reads
    /// the identical `bitbuf`.
    #[inline(always)]
    pub fn decode_prefilled(
        &self,
        bits: &crate::decompress::inflate::consume_first_decode::Bits<'_>,
    ) -> DecodedSymbol {
        // PRECONDITION (gate hardening): callers must invoke this only
        // immediately after a refill — post-refill, available() < 32 implies
        // pos == data.len() (a backstop refill would append nothing). A call
        // site violating this reads short bits silently; catch it in debug.
        debug_assert!(
            bits.available() >= 32 || bits.pos == bits.data.len(),
            "decode_prefilled called without the post-refill precondition"
        );
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
                sym_count: (next_sym >> LARGE_SYM_COUNT_OFFSET) & LARGE_SYM_COUNT_MASK,
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
}

/// Pure-rust analog of `IsalDistCode`. `dist_huff` is sized at
/// LIT_LEN_ELEMS to match the C, which uses the same buffer for both
/// lit/len and dist phases (the dist phase uses only the first 30
/// entries — `ISAL_DEF_DIST_SYMBOLS`).
pub struct LutDistCode {
    pub table: Box<InflateHuffCodeSmall>,
    dist_huff: Box<[HuffCode; LIT_LEN_ELEMS]>,
    valid: bool,
}

impl LutDistCode {
    pub fn new_empty() -> Self {
        Self {
            table: Box::new(InflateHuffCodeSmall::default()),
            dist_huff: Box::new([HuffCode::default(); LIT_LEN_ELEMS]),
            valid: false,
        }
    }

    /// Mirror of `isal_huffman.rs::IsalDistCode::rebuild_from`
    /// (line 279-316). Distance codes have no length-extra expansion,
    /// so this is simpler than the lit/len builder.
    pub fn rebuild_from(&mut self, code_lengths: &[u8]) -> bool {
        self.valid = false;
        if code_lengths.len() > LIT_LEN {
            return false;
        }
        for h in self.dist_huff.iter_mut() {
            h.0 = 0;
        }
        // `set_codes` count array is [u16; MAX_HUFF_TREE_DEPTH + 1] (16);
        // the dist alphabet has codes up to length 15 so [0..=15] is sized.
        let mut dist_count: [u16; MAX_HUFF_TREE_DEPTH + 1] = [0; MAX_HUFF_TREE_DEPTH + 1];
        for (i, &length) in code_lengths.iter().enumerate() {
            if length as usize >= 16 {
                return false;
            }
            dist_count[length as usize] += 1;
            self.dist_huff[i].set_length(length);
        }
        // `set_codes` writes back canonical codes for the dist table.
        if set_codes(&mut self.dist_huff[..], LIT_LEN, &mut dist_count).is_err() {
            return false;
        }
        // Build the LUT. `make_inflate_huff_code_dist` wants a [u16; 17]
        // count array (sized via MAX_HUFF_TREE_DEPTH + 1 + 1 in its
        // signature); we build that from the cumulative dist_count.
        let mut count_cumulative: [u16; MAX_HUFF_TREE_DEPTH + 1 + 1] =
            [0u16; MAX_HUFF_TREE_DEPTH + 1 + 1];
        // Build code_list ordered by code-length the way the LUT builder
        // expects: indices grouped by length, with `count_cumulative[k]`
        // giving the start of length-k group.
        // First pass: cumulative offsets.
        let mut acc: u16 = 0;
        for k in 0..=MAX_HUFF_TREE_DEPTH {
            count_cumulative[k] = acc;
            acc = acc.wrapping_add(dist_count[k]);
        }
        count_cumulative[MAX_HUFF_TREE_DEPTH + 1] = acc; // total count
                                                         // Second pass: write code_list grouped by length.
        let mut offsets = count_cumulative;
        let mut code_list = [0u32; 32];
        for (i, &length) in code_lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let li = length as usize;
            code_list[offsets[li] as usize] = i as u32;
            offsets[li] += 1;
        }
        if !make_inflate_huff_code_dist(
            &mut self.table,
            &mut self.dist_huff[..],
            LIT_LEN,
            &count_cumulative,
            &code_list,
            ISAL_DEF_DIST_SYMBOLS as u32,
        ) {
            return false;
        }
        self.valid = true;
        true
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// LITERAL port of `isal_huffman.rs::IsalDistCode::decode`
    /// (line 329-357).
    #[inline(always)]
    pub fn decode(
        &self,
        bits: &mut crate::decompress::inflate::consume_first_decode::Bits<'_>,
    ) -> Option<(u32, u32)> {
        if bits.available() < 32 {
            bits.refill();
        }
        let next_bits = bits.peek();
        let next_10 = (next_bits & ((1u64 << ISAL_DECODE_SHORT_BITS) - 1)) as usize;
        let mut next_sym = self.table.short_code_lookup[next_10] as u32;
        let bit_count;
        if (next_sym & SMALL_FLAG_BIT) == 0 {
            bit_count = next_sym >> SMALL_SHORT_CODE_LEN_OFFSET;
        } else {
            let bit_len = (next_sym - SMALL_FLAG_BIT) >> SMALL_SHORT_CODE_LEN_OFFSET;
            let long_next_bits = if bit_len <= 32 {
                next_bits & ((1u64 << bit_len) - 1)
            } else {
                next_bits
            };
            let long_idx = ((next_sym & SMALL_SHORT_SYM_MASK)
                + ((long_next_bits >> ISAL_DECODE_SHORT_BITS) as u32))
                as usize;
            next_sym = self.table.long_code_lookup[long_idx] as u32;
            bit_count = next_sym >> SMALL_LONG_CODE_LEN_OFFSET;
        }
        if bit_count == 0 {
            return None;
        }
        Some((next_sym & DIST_SYM_MASK, bit_count))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — table-build sanity. The wiring commit ships the silesia byte-
// perfect differential and the C-ISAL cross-check.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    /// Build a lit/len decoder from a real dynamic-block code-length set
    /// and round-trip decode a few hand-crafted symbol bit patterns.
    ///
    /// Code-length set is the fixed-Huffman table per RFC 1951 §3.2.6:
    /// literals 0..143 → 8 bits; 144..255 → 9 bits; 256..279 → 7 bits;
    /// 280..285 → 8 bits. Symbol 256 has the unique 7-bit code
    /// 0b0000000 (which after bit-reversal is also 0b0000000). The
    /// rebuilt decoder must return symbol=256 for that input pattern.
    #[test]
    fn decode_round_trips_fixed_huffman_eob() {
        let mut code_lengths = vec![0u8; LIT_LEN];
        for sym in 0..144 {
            code_lengths[sym] = 8;
        }
        for sym in 144..256 {
            code_lengths[sym] = 9;
        }
        for sym in 256..280 {
            code_lengths[sym] = 7;
        }
        for sym in 280..286 {
            code_lengths[sym] = 8;
        }

        let mut decoder = LutLitLenCode::new_empty();
        assert!(
            decoder.rebuild_from(&code_lengths),
            "fixed-Huffman code lengths must build"
        );
        assert!(decoder.is_valid());

        // The fixed table guarantees symbol 256 has code 0b0000000 (7
        // bits). Bit-reversed that's still 0b0000000.
        // Stream the bits little-endian into a Bits reader.
        let bytes = [0u8; 16];
        let mut bits = crate::decompress::inflate::consume_first_decode::Bits::new(&bytes);
        bits.refill();
        let result = decoder.decode(&mut bits);
        // Consumers recover the trailing symbol via `& 0xFFFF` (no build-time
        // class flag — faithful to igzip's runtime `cmp 256` classification).
        assert_eq!(result.symbol & 0xFFFF, 256, "EOB symbol decode failed");
        assert_eq!(result.bit_count, 7);
        assert_eq!(result.sym_count, 1);
    }

    /// Regression test for the INVALID_SYMBOL false-positive bug fixed
    /// 2026-05-27. ISA-L's packed-pair LUT entries can have
    /// `symbol & LARGE_SHORT_SYM_MASK == 0x1FFF` for VALID literal pairs
    /// — the caller MUST NOT use `symbol == INVALID_SYMBOL` as the
    /// invalid-code sentinel; only `bit_count == 0` distinguishes
    /// invalid. The bug was caught via reduced-bench data showing my
    /// port produced different bootstrap byte counts on silesia even
    /// though byte-perfect; tracing the LUT entries showed packed pairs
    /// where (sym1 | sym2 << 8) coincidentally equaled 0x1FFF.
    #[test]
    fn decoded_symbol_can_equal_invalid_sentinel_for_valid_pair() {
        // Build a Kraft-valid code-length set with short codes that
        // pack into pairs. We don't need a specific pair to coincide
        // with 0x1FFF for the test to be load-bearing — the assertion
        // is that decode returns bit_count > 0 for EVERY in-LUT entry
        // (so a buggy `symbol == INVALID_SYMBOL` check would catch a
        // valid entry and error).
        // Fixed-Huffman lit/len lengths (RFC 1951 §3.2.6) — Kraft-valid.
        let mut code_lengths = vec![0u8; LIT_LEN];
        for sym in 0..144 {
            code_lengths[sym] = 8;
        }
        for sym in 144..256 {
            code_lengths[sym] = 9;
        }
        for sym in 256..280 {
            code_lengths[sym] = 7;
        }
        for sym in 280..286 {
            code_lengths[sym] = 8;
        }
        let mut decoder = LutLitLenCode::new_empty();
        assert!(
            decoder.rebuild_from(&code_lengths),
            "fixed-Huffman code lengths must build"
        );

        // Count LUT entries with symbol bits == INVALID_SYMBOL AND
        // bit_count > 0 — these are valid entries that the legacy bug
        // would have erroneously rejected. We assert the COUNT is > 0
        // (proving the bug surface is reachable) only if such entries
        // exist; the unconditional assertion below — that no in-LUT
        // entry has bit_count==0 with a non-INVALID symbol field —
        // ensures the decoder never returns mis-flagged invalid.
        let mut dangerous_entries = 0usize;
        for &entry in decoder.table.short_code_lookup.iter() {
            if entry & LARGE_FLAG_BIT != 0 {
                continue;
            }
            let bit_count = entry >> LARGE_SHORT_CODE_LEN_OFFSET;
            let symbol = entry & LARGE_SHORT_SYM_MASK;
            if bit_count > 0 && symbol == INVALID_SYMBOL {
                dangerous_entries += 1;
            }
        }
        eprintln!(
            "[invalid-sentinel regression] LUT entries with symbol-bits == 0x1FFF AND bit_count > 0: {}",
            dangerous_entries
        );
        // Demonstrative — the count may be 0 for this particular
        // code-length set, but the bug class is real (see marker_inflate.rs
        // wiring comment).
    }

    /// Invariant lock that the builder carries NO bit-24 trailing-class flag
    /// (faithful to igzip's `make_inflate_huff_code_lit_len`, which sets none —
    /// the decoder classifies at RUNTIME via `cmp 256`). For EVERY valid short
    /// lit/len entry: (a) bit 24 is set ONLY as the natural sym3 bit-8 of a
    /// triple (`cnt == 3`) — never for cnt ∈ {1, 2}, even when their trailing
    /// is non-literal; and (b) the trailing symbol the decoders recover via the
    /// cnt-shift + `& 0xFFFF` is still byte-correct. A regression that re-adds
    /// a class flag for cnt ∈ {1, 2} would trip (a); a packing/shift error
    /// would trip (b).
    #[test]
    fn no_build_time_trailing_class_flag() {
        let mut fixed = vec![0u8; LIT_LEN];
        for sym in 0..144 {
            fixed[sym] = 8;
        }
        for sym in 144..256 {
            fixed[sym] = 9;
        }
        for sym in 256..280 {
            fixed[sym] = 7;
        }
        for sym in 280..286 {
            fixed[sym] = 8;
        }

        for code_lengths in [fixed, build_kraft_complete()] {
            let mut decoder = LutLitLenCode::new_empty();
            if !decoder.rebuild_from(&code_lengths) {
                continue; // skip sets that aren't Kraft-valid as written
            }
            for &entry in decoder.table.short_code_lookup.iter() {
                if entry & LARGE_FLAG_BIT != 0 {
                    continue; // long-code pointer
                }
                let bit_count = entry >> LARGE_SHORT_CODE_LEN_OFFSET;
                if bit_count == 0 {
                    continue; // unfilled / invalid slot
                }
                let cnt = (entry >> LARGE_SYM_COUNT_OFFSET) & LARGE_SYM_COUNT_MASK;
                debug_assert!((1..=3).contains(&cnt));
                let bit24 = (entry >> 24) & 1;
                if cnt < 3 {
                    // No build-time class flag: bit 24 is free for cnt ∈ {1, 2}.
                    assert_eq!(
                        bit24, 0,
                        "cnt={cnt} entry must not set bit 24 (no class flag): entry={entry:#010x}"
                    );
                }
                // Trailing still recovered byte-correctly by the decoders.
                let trailing = ((entry & LARGE_SHORT_SYM_MASK) >> (8 * (cnt - 1))) & 0xFFFF;
                assert!(
                    trailing <= MAX_LIT_LEN_SYM,
                    "trailing out of range: entry={entry:#010x} cnt={cnt} trailing={trailing}"
                );
            }
        }
    }

    /// A Kraft-complete fixed-Huffman length set (288 syms, the real one),
    /// used by the no-flag invariant test as a second, packing-heavy table.
    #[cfg(test)]
    fn build_kraft_complete() -> Vec<u8> {
        let mut v = vec![0u8; 288];
        for sym in 0..144 {
            v[sym] = 8;
        }
        for sym in 144..256 {
            v[sym] = 9;
        }
        for sym in 256..280 {
            v[sym] = 7;
        }
        for sym in 280..288 {
            v[sym] = 8;
        }
        v
    }

    /// LOCKING regression for the leaked-state `InvalidCodeLengths` bug:
    /// `LutLitLenCode.code_list` is a PERSISTENT reused box. A prior build
    /// (or any stale content) leaves indices in the tail slots; a fresh
    /// fixed-Huffman rebuild populates fewer entries and, before the fix,
    /// the stale slots fed an out-of-range index into the LUT builder →
    /// `rebuild_from` returned false (surfacing as `InvalidCodeLengths` on
    /// a perfectly valid fixed-Huffman block that followed a dynamic one).
    /// This test dirties `code_list` between two identical valid builds and
    /// asserts the second still succeeds. Decode-free and fast.
    #[test]
    fn bughunt_reuse_dynamic_then_fixed() {
        // Full RFC 1951 §3.2.6 fixed-Huffman lit/len lengths (288 syms;
        // symbols 286/287 are the length-8 phantom slots at the heart of
        // the bug).
        let mut fixed = vec![0u8; 288];
        for sym in 0..144 {
            fixed[sym] = 8;
        }
        for sym in 144..256 {
            fixed[sym] = 9;
        }
        for sym in 256..280 {
            fixed[sym] = 7;
        }
        for sym in 280..288 {
            fixed[sym] = 8;
        }

        let mut decoder = LutLitLenCode::new_empty();
        assert!(
            decoder.rebuild_from(&fixed),
            "first fixed-Huffman build must succeed"
        );

        // Simulate leaked state from a prior (e.g. dynamic) block: dirty
        // every code_list slot with an out-of-range index.
        for v in decoder.code_list.iter_mut() {
            *v = 500;
        }

        assert!(
            decoder.rebuild_from(&fixed),
            "reused build after dirtying code_list must still succeed (leaked-state regression)"
        );
        assert!(decoder.is_valid());
    }

    /// Fixed-Huffman distance LUT (32× length-5) must match C igzip byte-for-byte.
    /// The pure-rust dist-table decoder round-trips a single-bit input
    /// when the dist code-length set assigns length 1 to one symbol.
    #[test]
    fn dist_decode_round_trips() {
        // 2 dist symbols both with length 1 — codes 0 and 1.
        let mut code_lengths = vec![0u8; 30];
        code_lengths[0] = 1;
        code_lengths[1] = 1;

        let mut decoder = LutDistCode::new_empty();
        assert!(decoder.rebuild_from(&code_lengths));
        // Bit 0 -> symbol 0.
        let bytes = [0u8; 16];
        let mut bits = crate::decompress::inflate::consume_first_decode::Bits::new(&bytes);
        bits.refill();
        let r = decoder.decode(&mut bits).expect("valid code");
        assert_eq!(r.0, 0, "dist symbol 0 for bit-0 input");
        assert_eq!(r.1, 1, "1 bit consumed");
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
        assert!(
            make_inflate_huff_code_lit_len(
                &mut result,
                &mut lit_len_huff,
                LIT_LEN_ELEMS,
                &count,
                &code_list,
                TRIPLE_SYM_FLAG,
            ),
            "static fixed Huffman LUT must build in range"
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

    // ─────────────────────────────────────────────────────────────────────
    // MERGE-BLOCKER GATE: O(n) long-code grouping ≡ original O(n²) grouping.
    //
    // The production `make_inflate_huff_code_lit_len` replaced an O(n²)
    // long-code prefix-grouping inner re-scan with an O(n) single-pass
    // bucketed grouping (commit 74b41505). The win is byte-identical by
    // source-reasoning + a 3-corpus sha; this gate PROVES byte-identity of
    // the produced LUTs across weird-but-valid canonical Huffman lit/len
    // code-length distributions — a wrong table build = wrong decompressed
    // output on some valid input, so it must hold on ALL distributions, not
    // just the three sampled corpora.
    //
    // `make_inflate_huff_code_lit_len_oldref` below is a VERBATIM copy of the
    // pre-rewrite function (git 38f44528) — the original O(n²) long-code
    // grouping. Kept test-only. The differential builds the LUT both ways
    // from identical post-`set_and_expand` inputs and asserts byte-equality
    // of short_code_lookup, long_code_lookup, the return value, and the
    // huff-table post-state.
    // ─────────────────────────────────────────────────────────────────────

    /// VERBATIM port of the pre-74b41505 `make_inflate_huff_code_lit_len`
    /// (git 38f44528) — the ORIGINAL O(n²) long-code prefix-grouping (for
    /// each not-yet-placed long code, inner-rescan all later long codes
    /// sharing its 12-bit prefix). Test-only reference oracle for the O(n)
    /// rewrite. The short-code (singleton/pair/triple) section is identical
    /// to production (verified by diff) modulo the `profile-rebuild` timers;
    /// only the long-code section differs.
    #[cfg(test)]
    #[must_use]
    fn make_inflate_huff_code_lit_len_oldref(
        result: &mut InflateHuffCodeLarge,
        huff_code_table: &mut [HuffCode],
        _table_length: usize,
        count_total: &[u16; MAX_LIT_LEN_COUNT],
        code_list: &[u32],
        multisym: u32,
    ) -> bool {
        let max_symbol: u32 = MAX_LIT_LEN_SYM;
        let short_len = result.short_code_lookup.len();
        let long_len = result.long_code_lookup.len();

        let code_list_len = count_total[MAX_LIT_LEN_COUNT - 1] as u32;
        if code_list_len == 0 {
            result.short_code_lookup.fill(0);
            return true;
        }

        let mut last_length = 0u32;
        for l in 1..MAX_LIT_LEN_COUNT as u32 {
            if count_total[(l + 1) as usize] > count_total[l as usize] {
                last_length = l;
                break;
            }
        }
        if last_length == 0 {
            return true;
        }
        if last_length > ISAL_DECODE_LONG_BITS {
            last_length = ISAL_DECODE_LONG_BITS + 1;
        }
        let mut copy_size: usize = 1 << (last_length - 1);
        result.short_code_lookup[..copy_size].fill(0);
        let min_length = last_length;

        while last_length <= ISAL_DECODE_LONG_BITS {
            let (head, tail) = result.short_code_lookup.split_at_mut(copy_size);
            tail[..copy_size].copy_from_slice(&head[..copy_size]);
            copy_size *= 2;

            // singletons
            let mut index1 = count_total[last_length as usize] as usize;
            let last_total = count_total[(last_length + 1) as usize] as usize;
            while index1 < last_total {
                let sym1_index = code_list[index1];
                let sym1 = index_to_sym(sym1_index);
                let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
                let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
                if sym1 <= max_symbol {
                    if sym1_code as usize >= short_len {
                        return false;
                    }
                    result.short_code_lookup[sym1_code as usize] = sym1
                        | (sym1_len << LARGE_SHORT_CODE_LEN_OFFSET)
                        | (1 << LARGE_SYM_COUNT_OFFSET);
                }
                index1 += 1;
            }

            if multisym >= SINGLE_SYM_FLAG || last_length < 2 * min_length {
                last_length += 1;
                continue;
            }

            // pairs
            let pair_idx1_end = count_total[(last_length - min_length + 1) as usize] as usize;
            let mut index1 = count_total[min_length as usize] as usize;
            while index1 < pair_idx1_end {
                let sym1_index = code_list[index1];
                let sym1 = index_to_sym(sym1_index);
                let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
                let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
                if sym1 >= 256 {
                    index1 = count_total[(sym1_len + 1) as usize] as usize;
                    index1 = index1.saturating_sub(1) + 1;
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
                    if code as usize >= short_len {
                        return false;
                    }
                    result.short_code_lookup[code as usize] = sym1
                        | (sym2 << 8)
                        | (code_length << LARGE_SHORT_CODE_LEN_OFFSET)
                        | (2 << LARGE_SYM_COUNT_OFFSET);
                    index2 += 1;
                }
                index1 += 1;
            }

            if multisym >= DOUBLE_SYM_FLAG || last_length < 3 * min_length {
                last_length += 1;
                continue;
            }

            // triples
            let trip_idx1_end = count_total[(last_length - 2 * min_length + 1) as usize] as usize;
            let mut index1 = count_total[min_length as usize] as usize;
            while index1 < trip_idx1_end {
                let sym1_index = code_list[index1];
                let sym1 = index_to_sym(sym1_index);
                let sym1_len = huff_code_table[sym1_index as usize].length() as u32;
                let sym1_code = huff_code_table[sym1_index as usize].code() as u32;
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
                        let code = sym1_code
                            | (sym2_code << sym1_len)
                            | (sym3_code << (sym2_len + sym1_len));
                        let code_length = sym1_len + sym2_len + sym3_len;
                        if code as usize >= short_len {
                            return false;
                        }
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

        // ── ORIGINAL O(n²) long-code grouping (38f44528) ──
        let long_start = count_total[ISAL_DECODE_LONG_BITS as usize + 1] as usize;
        let long_code_length = (code_list_len as usize).saturating_sub(long_start);
        let long_code_list = &code_list[long_start..long_start + long_code_length];
        let mut long_code_lookup_length: u32 = 0;
        let mut temp_code_list: [u16; 1
            << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)] =
            [0u16; 1 << (MAX_LIT_LEN_CODE_LEN - ISAL_DECODE_LONG_BITS as usize)];

        for i in 0..long_code_length {
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
                    if temp_code_length as usize >= temp_code_list.len() {
                        return false;
                    }
                    temp_code_list[temp_code_length as usize] = long_code_list[j] as u16;
                    temp_code_length += 1;
                }
            }
            let lcl_size = 1usize << (max_length - ISAL_DECODE_LONG_BITS);
            if long_code_lookup_length as usize + lcl_size > long_len {
                return false;
            }
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
                    if idx >= long_len {
                        return false;
                    }
                    result.long_code_lookup[idx] =
                        (sym1 | (sym1_len << LARGE_LONG_CODE_LEN_OFFSET)) as u16;
                    long_bits += min_increment;
                }
                huff_code_table[sym1_index].set_code_and_extra(INVALID_CODE);
            }
            if first_bits as usize >= short_len {
                return false;
            }
            result.short_code_lookup[first_bits as usize] = long_code_lookup_length
                | (max_length << LARGE_SHORT_MAX_LEN_OFFSET)
                | LARGE_FLAG_BIT;
            long_code_lookup_length += 1u32 << (max_length - ISAL_DECODE_LONG_BITS);
        }
        true
    }

    /// Replicate the `LutLitLenCode::rebuild_from_multisym` prep that runs
    /// BEFORE `make_inflate_huff_code_lit_len` (zero + count + length-extra
    /// accounting + `set_and_expand_lit_len_huffcode`). Returns the
    /// post-expand `(huff_table, lit_count, code_list)` — the IDENTICAL
    /// inputs both make-variants consume — or `None` when the prep itself
    /// rejects the distribution (so `make` would never run).
    #[cfg(test)]
    fn prep_lit_len(
        code_lengths: &[u8],
    ) -> Option<(Vec<HuffCode>, [u16; MAX_LIT_LEN_COUNT], Vec<u32>)> {
        if code_lengths.len() > 288 {
            return None;
        }
        let mut huff = vec![HuffCode::default(); LIT_LEN_ELEMS];
        let mut code_list = vec![0u32; LIT_LEN_ELEMS + 2];
        let mut lit_count = [0u16; MAX_LIT_LEN_COUNT];
        let mut lit_expand_count = [0u16; MAX_LIT_LEN_COUNT];
        for (i, &length) in code_lengths.iter().enumerate() {
            if (length as usize) >= MAX_LIT_LEN_COUNT {
                return None;
            }
            lit_count[length as usize] += 1;
            huff[i].set_length(length);
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
        let table_len = code_lengths.len();
        if set_and_expand_lit_len_huffcode(
            &mut huff[..],
            table_len,
            &mut lit_count,
            &mut lit_expand_count,
            &mut code_list[..],
        )
        .is_err()
        {
            return None;
        }
        Some((huff, lit_count, code_list))
    }

    /// Decode one symbol packet through a built LUT exactly as the production
    /// `decode_prefilled` does (short flag → long-pointer indirection).
    /// Returns `(symbol, sym_count, bit_count)`. Used to compare the O(n) and
    /// O(n²) tables for DECODE-EQUIVALENCE (the property that actually governs
    /// correct decompressed output — see `build_both`).
    #[cfg(test)]
    fn decode_lut(t: &InflateHuffCodeLarge, bits: u64) -> (u32, u32, u32) {
        let n12 = (bits & ((1u64 << ISAL_DECODE_LONG_BITS) - 1)) as usize;
        let s = t.short_code_lookup[n12];
        if s & LARGE_FLAG_BIT == 0 {
            let bc = s >> LARGE_SHORT_CODE_LEN_OFFSET;
            let mut sym = s & LARGE_SHORT_SYM_MASK;
            if bc == 0 {
                sym = INVALID_SYMBOL;
            }
            return (
                sym,
                (s >> LARGE_SYM_COUNT_OFFSET) & LARGE_SYM_COUNT_MASK,
                bc,
            );
        }
        let ml = s >> LARGE_SHORT_MAX_LEN_OFFSET;
        let used = if ml <= 32 {
            bits & ((1u64 << ml) - 1)
        } else {
            bits
        };
        let idx = ((s & LARGE_SHORT_SYM_MASK) + ((used >> ISAL_DECODE_LONG_BITS) as u32)) as usize;
        let ls = t.long_code_lookup[idx] as u32;
        let bc = ls >> LARGE_LONG_CODE_LEN_OFFSET;
        let mut sym = ls & LARGE_LONG_SYM_MASK;
        if bc == 0 {
            sym = INVALID_SYMBOL;
        }
        (sym, 1, bc)
    }

    /// Exhaustive decode-equivalence over the ENTIRE distinguishable input
    /// space: for every 12-bit short index, if both tables hold a non-long
    /// (singleton/pair/triple) entry they must be BYTE-IDENTICAL (multi-symbol
    /// packing is unaffected by the long-code rewrite); for any prefix where
    /// either table holds a long pointer, decode every high-bit extension up
    /// to the larger `max_length` and require both tables to return the same
    /// `(symbol, sym_count, bit_count)`. Bits above `max_length` are
    /// don't-cares, so this covers all reachable bit patterns.
    #[cfg(test)]
    fn decode_equivalent(tn: &InflateHuffCodeLarge, to: &InflateHuffCodeLarge) -> bool {
        for p in 0u64..(1u64 << ISAL_DECODE_LONG_BITS) {
            let en = tn.short_code_lookup[p as usize];
            let eo = to.short_code_lookup[p as usize];
            let ln = en & LARGE_FLAG_BIT != 0;
            let lo = eo & LARGE_FLAG_BIT != 0;
            if !ln && !lo {
                // Pure short (packed) entry — must be byte-identical.
                if en != eo {
                    return false;
                }
                continue;
            }
            let mln = if ln {
                en >> LARGE_SHORT_MAX_LEN_OFFSET
            } else {
                ISAL_DECODE_LONG_BITS
            };
            let mlo = if lo {
                eo >> LARGE_SHORT_MAX_LEN_OFFSET
            } else {
                ISAL_DECODE_LONG_BITS
            };
            // Real max code length ≤ MAX_LIT_LEN_CODE_LEN (21); cap the
            // enumeration there (anything above is a don't-care bit anyway).
            let ml = mln.max(mlo).min(MAX_LIT_LEN_CODE_LEN as u32);
            let ext_bits = ml.saturating_sub(ISAL_DECODE_LONG_BITS);
            for ext in 0u64..(1u64 << ext_bits) {
                let bits = p | (ext << ISAL_DECODE_LONG_BITS);
                if decode_lut(tn, bits) != decode_lut(to, bits) {
                    return false;
                }
            }
        }
        true
    }

    /// Outcome of building the LUT both ways from identical prep inputs.
    #[cfg(test)]
    struct BuildPair {
        /// O(n) and O(n²) are DECODE-EQUIVALENT (identical decompressed
        /// output) AND short packed entries are byte-identical.
        decode_equiv: bool,
        /// O(n) and O(n²) produced byte-identical LUTs (strictly stronger;
        /// fails on the all-ones-prefix layout quirk — see module finding).
        byte_identical: bool,
        /// `set_and_expand` accepted the distribution (so `make` actually ran).
        prepped: bool,
        /// This distribution exercised the rewritten long-code path.
        has_long: bool,
        /// O(n) accepted but O(n²) rejected (the safe direction — the O(n²)
        /// 0xFFF over-match can over-allocate and overflow `long_len`).
        n_accepts_o_rejects: bool,
    }

    /// Build the lit/len LUT with BOTH the production O(n) grouping and the
    /// `*_oldref` O(n²) grouping from the SAME post-`set_and_expand` inputs,
    /// then compare for DECODE-EQUIVALENCE (the safety property) and, as a
    /// stronger diagnostic, raw byte-identity.
    #[cfg(test)]
    fn build_both(code_lengths: &[u8], multisym: u32) -> BuildPair {
        let Some((huff, lit_count, code_list)) = prep_lit_len(code_lengths) else {
            return BuildPair {
                decode_equiv: true,
                byte_identical: true,
                prepped: false,
                has_long: false,
                n_accepts_o_rejects: false,
            };
        };

        // Does this distribution exercise the long-code path at all? Long
        // codes are code_list[count_total[13]..code_list_len] (post-expand).
        let code_list_len = lit_count[MAX_LIT_LEN_COUNT - 1] as usize;
        let long_start = lit_count[ISAL_DECODE_LONG_BITS as usize + 1] as usize;
        let has_long = code_list_len > long_start;

        let mut huff_new = huff.clone();
        let mut table_new = InflateHuffCodeLarge::default();
        let ok_new = make_inflate_huff_code_lit_len(
            &mut table_new,
            &mut huff_new[..],
            LIT_LEN_ELEMS,
            &lit_count,
            &code_list[..],
            multisym,
        );

        let mut huff_old = huff;
        let mut table_old = InflateHuffCodeLarge::default();
        let ok_old = make_inflate_huff_code_lit_len_oldref(
            &mut table_old,
            &mut huff_old[..],
            LIT_LEN_ELEMS,
            &lit_count,
            &code_list[..],
            multisym,
        );

        let byte_identical = ok_new == ok_old
            && table_new.short_code_lookup[..] == table_old.short_code_lookup[..]
            && table_new.long_code_lookup[..] == table_old.long_code_lookup[..]
            && huff_new == huff_old;

        // Decode-equivalence: only meaningful when BOTH builds succeeded
        // (a rejected build leaves a partial table). When they disagree on
        // accept/reject, the only legal direction is O(n) accepts where the
        // O(n²) quirk over-allocates and rejects.
        let decode_equiv = if ok_new && ok_old {
            decode_equivalent(&table_new, &table_old)
        } else {
            true
        };
        let n_accepts_o_rejects = ok_new && !ok_old;

        BuildPair {
            decode_equiv,
            byte_identical,
            prepped: true,
            has_long,
            n_accepts_o_rejects,
        }
    }

    /// Result of checking a distribution across all three packing modes.
    #[cfg(test)]
    #[derive(Default)]
    struct ModeSummary {
        prepped: bool,
        has_long: bool,
        any_byte_divergence: bool,
        n_over_o: bool,
    }

    /// Assert DECODE-EQUIVALENCE (and that O(n) never rejects what O(n²)
    /// accepts) across all three multi-symbol packing modes.
    #[cfg(test)]
    fn assert_decode_equiv_all_modes(code_lengths: &[u8]) -> ModeSummary {
        let mut s = ModeSummary::default();
        for &ms in &[TRIPLE_SYM_FLAG, DOUBLE_SYM_FLAG, SINGLE_SYM_FLAG] {
            let r = build_both(code_lengths, ms);
            assert!(
                r.decode_equiv,
                "O(n) grouping is NOT decode-equivalent to the O(n²) reference \
                 (multisym={ms}) for distribution {:?}",
                code_lengths
            );
            assert!(
                !(r.prepped && !r.byte_identical && r.n_accepts_o_rejects && !r.decode_equiv),
                "unexpected combination"
            );
            s.prepped |= r.prepped;
            s.has_long |= r.prepped && r.has_long;
            s.any_byte_divergence |= r.prepped && !r.byte_identical;
            s.n_over_o |= r.n_accepts_o_rejects;
        }
        s
    }

    // ── Valid canonical-Huffman code-length distribution generator ──
    //
    // A canonical Huffman code must satisfy the Kraft equality (complete
    // tree) for the LUT prep to accept it — `set_and_expand` rejects any
    // length set whose `next_code` overflows 2^15. We build a COMPLETE tree
    // by repeated random leaf-splitting (always Kraft-exact, max depth ≤ 15),
    // then scatter the resulting lengths over a chosen alphabet so that
    // length-codes (≥257, with extra bits) participate and the
    // expansion-driven long path is exercised.

    /// Tiny deterministic splitmix64 PRNG (no external rand dep).
    #[cfg(test)]
    struct Rng(u64);
    #[cfg(test)]
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Build `k` code-lengths forming a COMPLETE canonical code (Kraft == 1),
    /// max length ≤ `max_len`. Returns fewer than `k` only if the depth cap
    /// is hit (still complete). `k == 1` → a single length-1 code.
    #[cfg(test)]
    fn complete_lengths(rng: &mut Rng, k: usize, max_len: u8) -> Vec<u8> {
        if k <= 1 {
            return vec![1];
        }
        let mut depths = vec![1u8, 1u8];
        while depths.len() < k {
            let splittable: Vec<usize> = depths
                .iter()
                .enumerate()
                .filter(|(_, d)| **d < max_len)
                .map(|(i, _)| i)
                .collect();
            if splittable.is_empty() {
                break;
            }
            let pick = splittable[rng.below(splittable.len())];
            let d = depths[pick];
            depths[pick] = d + 1;
            depths.push(d + 1);
        }
        depths
    }

    /// Scatter a complete length set across an `alphabet`-sized code-length
    /// vector at random distinct positions (the rest length 0). This puts
    /// lengths on length-code symbols (257..286) so the length-extra
    /// expansion — and its long-code interaction — is exercised.
    #[cfg(test)]
    fn scatter(rng: &mut Rng, lengths: &[u8], alphabet: usize) -> Vec<u8> {
        let alphabet = alphabet.max(lengths.len());
        let mut positions: Vec<usize> = (0..alphabet).collect();
        // Fisher–Yates partial shuffle for the first lengths.len() slots.
        for i in 0..lengths.len() {
            let j = i + rng.below(alphabet - i);
            positions.swap(i, j);
        }
        let mut out = vec![0u8; alphabet];
        for (idx, &len) in lengths.iter().enumerate() {
            out[positions[idx]] = len;
        }
        out
    }

    /// Generate one valid distribution from a seed, varying symbol count,
    /// alphabet size, and max code length.
    #[cfg(test)]
    fn gen_distribution(seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        // Symbol count: bias across the whole range incl. tiny + near-full.
        let k = match rng.below(10) {
            0 => 1,
            1 => 2,
            2 => 3 + rng.below(6),
            3..=5 => 10 + rng.below(120),
            _ => 100 + rng.below(187), // up to ~286
        };
        let max_len = 9 + (rng.below(7) as u8); // 9..=15
        let lengths = complete_lengths(&mut rng, k, max_len);
        // Alphabet ≤ LIT_LEN (286) — the production INPUT DOMAIN for arbitrary
        // (dynamic-block) content. table_length > 286 is reached ONLY by the
        // genuine fixed-Huffman 288 table (which has the exact 286/287 length-8
        // codes the `count[8] -= 2` correction in `set_and_expand` requires);
        // that case is covered explicitly in `perblock_oldref_edge_cases`.
        // Feeding arbitrary 287/288-symbol sets would underflow that
        // correction — a state the production header path never produces.
        let alphabet = match rng.below(3) {
            0 => 286,
            1 => lengths.len().max(1),
            _ => 30 + rng.below(256), // 30..=285
        };
        scatter(&mut rng, &lengths, alphabet.min(286))
    }

    /// ⛔ MERGE-BLOCKER GATE — Gate-0 + breadth.
    ///
    /// Deterministic sweep over many generated VALID canonical lit/len
    /// code-length distributions. For each, builds the LUT with the production
    /// O(n) grouping and the O(n²) reference and asserts DECODE-EQUIVALENCE on
    /// EVERY one (the property that governs correct decompressed output) AND
    /// that a meaningful fraction actually exercised the rewritten long-code
    /// path (else the gate would be inert — Gate-0).
    ///
    /// FINDING (deterministic, reproduced by `perblock_byte_layout_divergence`
    /// below): raw LUT byte-identity does NOT hold universally — the O(n²)
    /// reference's inner re-scan keys on `code() & 0xFFF`, and an
    /// already-placed member's `code()` is `INVALID_CODE` (low 16 bits
    /// `0xFFFF` → `& 0xFFF == 4095`), so it spuriously re-adds invalidated
    /// members to the ALL-ONES (0xFFF) prefix group, inflating that group's
    /// `max_length` and over-allocating `long_code_lookup`. The O(n) rewrite
    /// buckets by the real prefix BEFORE any invalidation and is immune. Both
    /// remain DECODE-EQUIVALENT (the over-allocated slots are unreachable
    /// zero-fill), so decompressed output is identical — the rewrite is SAFE,
    /// and is in fact the cleaner of the two.
    #[test]
    fn perblock_oldref_decode_equiv_gate() {
        const N: u64 = 20_000;
        let mut prepped_n = 0u64;
        let mut long_n = 0u64;
        let mut byte_div_n = 0u64;
        for seed in 0..N {
            let dist = gen_distribution(seed);
            let s = assert_decode_equiv_all_modes(&dist);
            if s.prepped {
                prepped_n += 1;
            }
            if s.has_long {
                long_n += 1;
            }
            if s.any_byte_divergence {
                byte_div_n += 1;
            }
        }
        // Gate-0 (non-inert): a substantial fraction of accepted distributions
        // must drive the long-code path that the O(n) rewrite touches.
        assert!(
            prepped_n > N / 2,
            "too many distributions rejected by prep: {prepped_n}/{N}"
        );
        let frac = long_n as f64 / prepped_n as f64;
        assert!(
            frac > 0.10,
            "long-code path under-exercised: {long_n}/{prepped_n} = {frac:.3} (<0.10 ⇒ inert gate)"
        );
        eprintln!(
            "perblock_oldref_decode_equiv_gate: {N} distributions, prepped={prepped_n}, \
             long-code-hit={long_n} ({:.1}% of prepped), byte-layout-divergent={byte_div_n}",
            100.0 * frac
        );
    }

    /// Named edge cases the advisor called out — explicit, deterministic.
    /// Each asserts DECODE-EQUIVALENCE between the O(n) win and the O(n²)
    /// reference across all three packing modes.
    #[test]
    fn perblock_oldref_edge_cases() {
        // 1. Fixed-Huffman 288-entry table (symbols 286/287 participate).
        let mut fixed = vec![0u8; 288];
        for s in 0..144 {
            fixed[s] = 8;
        }
        for s in 144..256 {
            fixed[s] = 9;
        }
        for s in 256..280 {
            fixed[s] = 7;
        }
        for s in 280..288 {
            fixed[s] = 8;
        }
        let s = assert_decode_equiv_all_modes(&fixed);
        assert!(s.prepped, "fixed-288 must prep");

        // 2. Single symbol (length 1) at a literal and at a length-code slot.
        let mut single_lit = vec![0u8; 286];
        single_lit[65] = 1;
        assert!(assert_decode_equiv_all_modes(&single_lit).prepped);
        let mut single_len = vec![0u8; 286];
        single_len[270] = 1; // a length-code with extra bits
        assert!(assert_decode_equiv_all_modes(&single_len).prepped);

        // 3. Two symbols (length 1 each = complete).
        let mut two = vec![0u8; 286];
        two[0] = 1;
        two[257] = 1;
        assert!(assert_decode_equiv_all_modes(&two).prepped);

        // 4. All-same-length complete codes: 2^L symbols of length L
        //    (L ≤ 8 so 2^L ≤ 256 fits the alphabet; L=12,13,14,15 covered by
        //    the `deep` case below). Exercises the no-long-code path.
        for l in 1u8..=8 {
            let n = 1usize << l;
            let alphabet = n.max(20);
            let mut all = vec![0u8; alphabet];
            for s in all.iter_mut().take(n) {
                *s = l;
            }
            assert!(
                assert_decode_equiv_all_modes(&all).prepped,
                "all-same-length L={l} must prep"
            );
        }

        // 5. Max base code length 15 present (deep complete tree:
        //    lengths 1,2,3,...,14,15,15 = Kraft 1) — long codes guaranteed.
        let mut deep = vec![0u8; 286];
        for (i, len) in (1u8..=14).enumerate() {
            deep[i] = len;
        }
        deep[14] = 15;
        deep[15] = 15;
        let s = assert_decode_equiv_all_modes(&deep);
        assert!(
            s.has_long,
            "deep 1..15 tree must exercise the long-code path"
        );

        // 6. Length-codes driving the length-extra EXPANSION past 12 bits:
        //    a complete code whose mass sits on high length-codes (each with
        //    several extra bits) so `set_and_expand` produces expanded codes
        //    > 12 bits → the long path via expansion.
        //    Lengths: literals 0,1 @ 2; literal 2 @ 2; length-codes
        //    280..285 @ 4 (extra bits 4..5) → expanded 8..9; deepen them so
        //    some expand past 12: put a length-13 length-code with 5 extra.
        let mut expand = vec![0u8; 286];
        // Complete: 0,1 @ 1? Build a valid complete set with long length-codes.
        // Tree: {a:2, b:2, c:2, d:2} is complete on 4 leaves. Assign two
        // literals + two high length-codes so the length-codes expand long.
        expand[0] = 2;
        expand[1] = 2;
        expand[284] = 2; // length-code 284: 4 extra bits → expanded 6
        expand[285] = 2; // length-code 285: 0 extra bits
                         // To force expansion > 12, give a length-code a deep base code.
                         // Rebuild as: complete tree lengths 1,2,3,3 over {lit0, lit1, len284, len285}
        let mut expand2 = vec![0u8; 286];
        expand2[0] = 1;
        expand2[1] = 2;
        expand2[284] = 3; // 4 extra → expanded 7
        expand2[285] = 3; // 0 extra → expanded 3
        assert!(assert_decode_equiv_all_modes(&expand).prepped);
        assert!(assert_decode_equiv_all_modes(&expand2).prepped);
    }

    /// FINDING (deterministic, banked): the O(n) rewrite is NOT raw
    /// byte-identical to the O(n²) original on every distribution — the
    /// original spuriously re-adds already-invalidated members (whose
    /// `code()` masks to the all-ones 12-bit prefix `0xFFF`) to the 0xFFF
    /// group, over-inflating its `max_length` / `long_code_lookup`
    /// allocation. This test reproduces a concrete divergent distribution,
    /// proves the byte-divergence exists, pins it to the all-ones prefix
    /// group (the short-pointer at index 0xFFF), confirms the diverging
    /// short entries are ALL long pointers (singleton/pair/triple packing
    /// untouched) and the huff post-state is identical, and confirms the two
    /// tables remain DECODE-EQUIVALENT (so output is unaffected → the win is
    /// SAFE; the O(n) version is the cleaner of the two).
    #[test]
    fn perblock_byte_layout_divergence_is_benign() {
        // Scan generated distributions for the first that exhibits the known
        // all-ones-prefix byte-layout divergence (a long code on prefix 0xFFF
        // alongside earlier, already-invalidated long codes). Robust to
        // generator tweaks.
        let mut found: Option<Vec<u8>> = None;
        for seed in 0..50_000u64 {
            let dist = gen_distribution(seed);
            let r = build_both(&dist, TRIPLE_SYM_FLAG);
            if r.prepped && !r.byte_identical {
                found = Some(dist);
                break;
            }
        }
        let dist = found.expect(
            "no byte-layout divergence found in 50k distributions — the O(n²) \
             0xFFF over-match quirk did not reproduce",
        );
        let (huff, lit_count, code_list) = prep_lit_len(&dist).expect("dist must prep");

        let mut hn = huff.clone();
        let mut tn = InflateHuffCodeLarge::default();
        assert!(make_inflate_huff_code_lit_len(
            &mut tn,
            &mut hn[..],
            LIT_LEN_ELEMS,
            &lit_count,
            &code_list[..],
            TRIPLE_SYM_FLAG
        ));
        let mut ho = huff;
        let mut to = InflateHuffCodeLarge::default();
        assert!(make_inflate_huff_code_lit_len_oldref(
            &mut to,
            &mut ho[..],
            LIT_LEN_ELEMS,
            &lit_count,
            &code_list[..],
            TRIPLE_SYM_FLAG
        ));

        // 1. A raw byte-divergence in the LUT exists.
        let byte_identical = tn.short_code_lookup[..] == to.short_code_lookup[..]
            && tn.long_code_lookup[..] == to.long_code_lookup[..];
        assert!(
            !byte_identical,
            "expected the known all-ones-prefix layout divergence to reproduce"
        );

        // 2. Every divergent short entry is a LONG POINTER (packing untouched).
        let all_ones = (1usize << ISAL_DECODE_LONG_BITS) - 1; // 0xFFF
        let mut all_ones_diverged = false;
        for i in 0..tn.short_code_lookup.len() {
            if tn.short_code_lookup[i] != to.short_code_lookup[i] {
                assert!(
                    tn.short_code_lookup[i] & LARGE_FLAG_BIT != 0
                        || to.short_code_lookup[i] & LARGE_FLAG_BIT != 0,
                    "a NON-long short entry diverged at {i:#x} — packing changed!"
                );
                if i == all_ones {
                    all_ones_diverged = true;
                }
            }
        }
        // 3. The root-cause group (the all-ones 0xFFF prefix) is involved:
        //    its short pointer's max_length differs (over-inflated by O(n²)).
        let n_ml = tn.short_code_lookup[all_ones] >> LARGE_SHORT_MAX_LEN_OFFSET;
        let o_ml = to.short_code_lookup[all_ones] >> LARGE_SHORT_MAX_LEN_OFFSET;
        assert!(
            all_ones_diverged && o_ml > n_ml,
            "root cause not confirmed: 0xFFF group n_ml={n_ml} o_ml={o_ml}"
        );

        // 4. huff post-state identical (same real members invalidated).
        assert_eq!(hn, ho, "huff post-state diverged");

        // 5. DECODE-EQUIVALENT — output is unaffected (the win is SAFE).
        assert!(
            decode_equivalent(&tn, &to),
            "tables are NOT decode-equivalent — would change decompressed output"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 3000, ..ProptestConfig::default() })]

        /// Property: for any seed-generated VALID canonical lit/len
        /// code-length distribution and any multisym mode, the O(n) long-code
        /// grouping is DECODE-EQUIVALENT to the O(n²) reference (identical
        /// decompressed output). Shrinks the seed on failure.
        #[test]
        fn perblock_oldref_decode_equivalent(seed in any::<u64>(), mode in 0u32..3u32) {
            let dist = gen_distribution(seed);
            let r = build_both(&dist, mode);
            prop_assert!(
                r.decode_equiv,
                "O(n) vs O(n²) DECODE divergence (seed={seed}, mode={mode}) dist={:?}",
                dist
            );
            // O(n) must never reject what O(n²) accepts (the over-allocation
            // quirk can only make O(n²) the stricter one).
            prop_assert!(
                !(! r.byte_identical && r.prepped) || r.decode_equiv,
                "byte-divergence without decode-equivalence (seed={seed})"
            );
        }

        /// Property over RAW arbitrary length vectors (incl. invalid /
        /// over-subscribed / under-subscribed) — both impls must agree on the
        /// DECODE result and never panic. Catches divergence on ill-formed
        /// seeds the speculative single-member path can hand the builder.
        #[test]
        fn perblock_oldref_raw_lengths(
            // ≤ 286 (LIT_LEN): the production input domain for dynamic blocks
            // (HLIT ≤ 286). 287/288 is reached only by the genuine fixed table
            // — see the note in `gen_distribution`.
            lens in prop::collection::vec(0u8..16u8, 0..=286usize),
            mode in 0u32..3u32,
        ) {
            let r = build_both(&lens, mode);
            prop_assert!(
                r.decode_equiv,
                "DECODE divergence on raw lengths mode={mode}: {:?}",
                lens
            );
        }
    }
}
