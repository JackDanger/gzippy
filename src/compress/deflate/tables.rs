//! DEFLATE constant tables.
//!
//! Transliterated from libdeflate `vendor/libdeflate/lib/deflate_compress.c`
//! (`deflate_length_slot_base`, `deflate_extra_length_bits`,
//! `deflate_offset_slot_base`, `deflate_extra_offset_bits`,
//! `deflate_length_slot`, `deflate_offset_slot`,
//! `deflate_precode_lens_permutation`, `deflate_extra_precode_bits`,
//! `deflate_get_offset_slot`, `deflate_init_static_codes`) and the format
//! definitions in `vendor/libdeflate/lib/deflate_constants.h`. Length/offset
//! bases and extra-bit counts follow RFC 1951 §3.2.5/§3.2.6.

// ---- Format-level constants (deflate_constants.h) ----

pub const DEFLATE_MIN_MATCH_LEN: u32 = 3;
pub const DEFLATE_MAX_MATCH_LEN: u32 = 258;

pub const DEFLATE_NUM_PRECODE_SYMS: usize = 19;
pub const DEFLATE_NUM_LITLEN_SYMS: usize = 288;
pub const DEFLATE_NUM_OFFSET_SYMS: usize = 32;

/// First litlen symbol that represents a match length (symbol 257 => slot 0).
pub const DEFLATE_FIRST_LEN_SYM: usize = 257;
/// End-of-block symbol.
pub const DEFLATE_END_OF_BLOCK: usize = 256;

pub const DEFLATE_MAX_PRE_CODEWORD_LEN: u32 = 7;
pub const DEFLATE_MAX_OFFSET_CODEWORD_LEN: u32 = 15;

// libdeflate uses a slightly-tighter litlen limit than the format max; it is
// still a valid, decodable code (2^14 >= 288).
pub const MAX_LITLEN_CODEWORD_LEN: u32 = 14;
pub const MAX_OFFSET_CODEWORD_LEN: u32 = DEFLATE_MAX_OFFSET_CODEWORD_LEN;
pub const MAX_PRE_CODEWORD_LEN: u32 = DEFLATE_MAX_PRE_CODEWORD_LEN;

// Block type field (BTYPE) values.
pub const DEFLATE_BLOCKTYPE_UNCOMPRESSED: u32 = 0;
pub const DEFLATE_BLOCKTYPE_STATIC_HUFFMAN: u32 = 1;
pub const DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN: u32 = 2;

// ---- Length/offset base + extra-bit tables ----

/// Length slot => base match length (slots 0..=28, litlen symbols 257..=285).
pub const LENGTH_SLOT_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Length slot => number of extra length bits.
pub const LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Offset slot => base offset value (slots 0..=29).
pub const OFFSET_SLOT_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Offset slot => number of extra offset bits.
pub const OFFSET_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Table: length (0..=258) => length slot. Equivalent to libdeflate's
/// `deflate_length_slot[]`; generated from [`LENGTH_SLOT_BASE`] at const-eval so
/// it is provably consistent with the slot bases (lengths < 3 never occur and
/// map to slot 0, matching the vendor filler).
pub const LENGTH_SLOT: [u8; (DEFLATE_MAX_MATCH_LEN + 1) as usize] = build_length_slot();

const fn build_length_slot() -> [u8; (DEFLATE_MAX_MATCH_LEN + 1) as usize] {
    let mut t = [0u8; (DEFLATE_MAX_MATCH_LEN + 1) as usize];
    let mut len = DEFLATE_MIN_MATCH_LEN as usize;
    while len <= DEFLATE_MAX_MATCH_LEN as usize {
        let mut s = 0usize;
        while s + 1 < 29 && LENGTH_SLOT_BASE[s + 1] <= len as u32 {
            s += 1;
        }
        t[len] = s as u8;
        len += 1;
    }
    t
}

/// Table: `offset - 1` (offset <= 256) => offset slot. Equivalent to
/// libdeflate's `deflate_offset_slot[]`; generated from [`OFFSET_SLOT_BASE`].
pub const OFFSET_SLOT: [u8; 256] = build_offset_slot();

const fn build_offset_slot() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut off = 1usize;
    while off <= 256 {
        let mut s = 0usize;
        while s + 1 < 30 && OFFSET_SLOT_BASE[s + 1] <= off as u32 {
            s += 1;
        }
        t[off - 1] = s as u8;
        off += 1;
    }
    t
}

/// Order in which precode codeword lengths are written.
/// `deflate_precode_lens_permutation[]`.
pub const PRECODE_LENS_PERMUTATION: [u8; DEFLATE_NUM_PRECODE_SYMS] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Precode symbol => number of extra bits. `deflate_extra_precode_bits[]`.
pub const PRECODE_EXTRA_BITS: [u8; DEFLATE_NUM_PRECODE_SYMS] =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];

/// Return the length slot for a match length in `3..=258`.
#[inline]
pub fn length_slot(len: u32) -> u8 {
    LENGTH_SLOT[len as usize]
}

/// Return the offset slot for a match offset in `1..=32768`.
///
/// Port of `deflate_get_offset_slot()` — uses the 256-entry small map plus the
/// "each slot [16..30) is 128x a slot [2..16)" identity for larger offsets.
#[inline]
pub fn offset_slot(offset: u32) -> u8 {
    debug_assert!((1..=32768).contains(&offset));
    // n = (offset <= 256) ? 0 : 7, expressed branchlessly. In C this is an
    // unsigned wrapping subtraction; mirror it with wrapping_sub on u32.
    let n = (256u32.wrapping_sub(offset)) >> 29;
    OFFSET_SLOT[((offset - 1) >> n) as usize] + ((n as u8) << 1)
}

/// Frequencies that reproduce the RFC 1951 §3.2.6 fixed literal/length code.
///
/// Port of `deflate_init_static_codes()`: feeding these frequencies through the
/// canonical Huffman builder yields the fixed code (lens 8/9/7/8 in the four
/// ranges).
pub fn static_litlen_freqs() -> [u32; DEFLATE_NUM_LITLEN_SYMS] {
    // Vendor uses `1 << (9 - len)` for the four ranges (len 8/9/7/8), i.e.
    // weights 2 / 1 / 4 / 2 — spelled out here as literals to keep clippy happy.
    let mut f = [0u32; DEFLATE_NUM_LITLEN_SYMS];
    for x in f.iter_mut().take(144) {
        *x = 2; // len 8
    }
    for x in f.iter_mut().take(256).skip(144) {
        *x = 1; // len 9
    }
    for x in f.iter_mut().take(280).skip(256) {
        *x = 4; // len 7
    }
    for x in f.iter_mut().take(288).skip(280) {
        *x = 2; // len 8
    }
    f
}

/// Frequencies that reproduce the RFC 1951 fixed offset code (all length 5).
pub fn static_offset_freqs() -> [u32; DEFLATE_NUM_OFFSET_SYMS] {
    // Vendor: `1 << (5 - 5)` == 1 for every offset symbol.
    [1u32; DEFLATE_NUM_OFFSET_SYMS]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_slot_spot_checks() {
        // RFC 1951 §3.2.5: length 3 => symbol 257 (slot 0), 258 => 285 (slot 28).
        assert_eq!(length_slot(3), 0);
        assert_eq!(length_slot(4), 1);
        assert_eq!(length_slot(10), 7);
        assert_eq!(length_slot(11), 8); // first slot with 1 extra bit
        assert_eq!(length_slot(258), 28);
        // Base + extra must bracket every length in the slot.
        for len in 3..=258u32 {
            let slot = length_slot(len) as usize;
            let base = LENGTH_SLOT_BASE[slot];
            let extra = LENGTH_EXTRA_BITS[slot] as u32;
            assert!(len >= base, "len {len} < base {base}");
            assert!(
                len < base + (1 << extra) || slot == 28,
                "len {len} slot {slot}"
            );
        }
    }

    #[test]
    fn offset_slot_spot_checks() {
        assert_eq!(offset_slot(1), 0);
        assert_eq!(offset_slot(2), 1);
        assert_eq!(offset_slot(4), 3);
        assert_eq!(offset_slot(5), 4); // first slot with 1 extra bit
        assert_eq!(offset_slot(256), 15);
        assert_eq!(offset_slot(257), 16);
        assert_eq!(offset_slot(32768), 29);
        for off in 1..=32768u32 {
            let slot = offset_slot(off) as usize;
            let base = OFFSET_SLOT_BASE[slot];
            let extra = OFFSET_EXTRA_BITS[slot] as u32;
            assert!(off >= base, "off {off} < base {base}");
            assert!(off < base + (1 << extra), "off {off} slot {slot}");
        }
    }

    #[test]
    fn slot_bases_are_contiguous() {
        // Each length slot's range starts exactly where the previous ended,
        // EXCEPT the irregular tail: slot 27 (base 227, 5 extra bits) is capped
        // at length 257, and length 258 gets its own zero-extra-bit slot 28.
        for slot in 0..27usize {
            let end = LENGTH_SLOT_BASE[slot] + (1 << LENGTH_EXTRA_BITS[slot]);
            assert_eq!(end, LENGTH_SLOT_BASE[slot + 1], "length slot {slot}");
        }
        assert_eq!(LENGTH_SLOT_BASE[27], 227);
        assert_eq!(LENGTH_SLOT_BASE[28], 258);
        assert_eq!(LENGTH_SLOT[257], 27);
        assert_eq!(LENGTH_SLOT[258], 28);

        // Offset slots are fully regular.
        for slot in 0..29usize {
            let end = OFFSET_SLOT_BASE[slot] + (1 << OFFSET_EXTRA_BITS[slot]);
            assert_eq!(end, OFFSET_SLOT_BASE[slot + 1], "offset slot {slot}");
        }
    }
}
