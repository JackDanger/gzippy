//! C: `vendor/libdeflate/lib/deflate_compress.c:325-350` — the Huffman code
//! containers, plus `deflate_make_huffman_codes` (:1416) and
//! `deflate_init_static_codes` (:1433).
//!
//! # Why `DeflateLens` is `#[repr(C)]` and exposes a flat byte view
//!
//! `deflate_precompute_huffman_header` (:1572) does this:
//!
//! ```c
//! STATIC_ASSERT(offsetof(struct deflate_lens, offset) == DEFLATE_NUM_LITLEN_SYMS);
//! memmove((u8 *)&c->codes.lens + c->o.precode.num_litlen_syms,
//!         (u8 *)&c->codes.lens + DEFLATE_NUM_LITLEN_SYMS,
//!         c->o.precode.num_offset_syms);
//! ```
//!
//! It slides the offset lengths DOWN over the tail of the litlen lengths so that the
//! two alphabets are contiguous, runs the precode over the joined array, then slides
//! them back. That is not an implementation detail we may abstract away: the joined
//! array is precisely what gets RLE-encoded, and the RLE crosses the boundary — a run
//! of zeroes ending the litlen lengths and beginning the offset lengths is coded as
//! ONE run. Splitting the alphabets into two independent arrays would emit a
//! different (larger, but still valid) header, and would break byte-identity on the
//! 154 tied cells.
//!
//! So the layout is load-bearing, and `as_flat_mut()` reproduces the C's cast. A
//! test asserts the offset of the `offset` field, matching the C's STATIC_ASSERT.

use super::huffman::deflate_make_huffman_code;
use super::{
    DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS, MAX_LITLEN_CODEWORD_LEN,
    MAX_OFFSET_CODEWORD_LEN,
};

/// Total length of the flat litlen+offset lengths array. Not a C constant; the C
/// spells it `DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS` at each use.
pub(crate) const NUM_LITLEN_PLUS_OFFSET_SYMS: usize =
    DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS;

/// C: `struct deflate_codewords` (:326) — codewords for the DEFLATE Huffman codes.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct DeflateCodewords {
    pub(crate) litlen: [u32; DEFLATE_NUM_LITLEN_SYMS],
    pub(crate) offset: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

/// C: `struct deflate_lens` (:331) — codeword lengths (in bits) for the DEFLATE
/// Huffman codes. A zero length means the corresponding symbol had zero frequency.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct DeflateLens {
    pub(crate) litlen: [u8; DEFLATE_NUM_LITLEN_SYMS],
    pub(crate) offset: [u8; DEFLATE_NUM_OFFSET_SYMS],
}

/// C: `struct deflate_codes` (:337) — codewords and lengths together.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct DeflateCodes {
    pub(crate) codewords: DeflateCodewords,
    pub(crate) lens: DeflateLens,
}

/// C: `struct deflate_freqs` (:343) — symbol frequency counters.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct DeflateFreqs {
    pub(crate) litlen: [u32; DEFLATE_NUM_LITLEN_SYMS],
    pub(crate) offset: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl DeflateCodewords {
    pub(crate) const fn new() -> Self {
        Self {
            litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }
}

impl DeflateLens {
    pub(crate) const fn new() -> Self {
        Self {
            litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }

    /// C: `(u8 *)&c->codes.lens` — the whole struct viewed as one flat byte array of
    /// `DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS` entries.
    ///
    /// # Safety of the reinterpretation
    ///
    /// `DeflateLens` is `#[repr(C)]` and holds exactly two `u8` arrays, so it has
    /// alignment 1, size 320, and no padding: `offset` starts at byte 288, which is
    /// the C's own `STATIC_ASSERT` (:1593) and is checked by
    /// `lens_layout_matches_the_c_static_assert` below. Reading or writing any byte
    /// of it as `u8` is therefore in-bounds and correctly typed.
    pub(crate) fn as_flat_mut(&mut self) -> &mut [u8; NUM_LITLEN_PLUS_OFFSET_SYMS] {
        // SAFETY: see the doc comment — repr(C), align 1, size 320, no padding.
        unsafe { &mut *(self as *mut DeflateLens as *mut [u8; NUM_LITLEN_PLUS_OFFSET_SYMS]) }
    }

    /// Read-only counterpart of [`Self::as_flat_mut`].
    pub(crate) fn as_flat(&self) -> &[u8; NUM_LITLEN_PLUS_OFFSET_SYMS] {
        // SAFETY: see [`Self::as_flat_mut`].
        unsafe { &*(self as *const DeflateLens as *const [u8; NUM_LITLEN_PLUS_OFFSET_SYMS]) }
    }
}

impl DeflateCodes {
    pub(crate) const fn new() -> Self {
        Self {
            codewords: DeflateCodewords::new(),
            lens: DeflateLens::new(),
        }
    }
}

impl DeflateFreqs {
    pub(crate) const fn new() -> Self {
        Self {
            litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }

    /// C: `deflate_reset_symbol_frequencies` (:1404) — `memset(&c->freqs, 0, ...)`.
    /// Must be called when starting a new DEFLATE block.
    pub(crate) fn reset(&mut self) {
        self.litlen = [0; DEFLATE_NUM_LITLEN_SYMS];
        self.offset = [0; DEFLATE_NUM_OFFSET_SYMS];
    }
}

/// C: `deflate_make_huffman_codes(const struct deflate_freqs *freqs,
/// struct deflate_codes *codes)` (:1416)
///
/// Build the literal/length and offset Huffman codes for a DEFLATE block.
///
/// This takes as input the frequency tables for each alphabet and produces as output
/// a set of tables that map symbols to codewords and codeword lengths.
///
/// Note the asymmetric length limits: litlen codewords are capped at
/// `MAX_LITLEN_CODEWORD_LEN` = **14**, not the format's 15. That is libdeflate's
/// choice (`:118`), and it is deliberate — a 14-bit cap lets the decoder's
/// `ADD_BITS`/`FLUSH_BITS` budget hold a litlen codeword plus its extra length bits
/// plus an offset codeword plus its extra offset bits inside one 64-bit bitbuffer.
/// Raising it to 15 would be a legal DEFLATE stream and a different one.
pub(crate) fn deflate_make_huffman_codes(freqs: &DeflateFreqs, codes: &mut DeflateCodes) {
    deflate_make_huffman_code(
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &freqs.litlen,
        &mut codes.lens.litlen,
        &mut codes.codewords.litlen,
    );

    deflate_make_huffman_code(
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        &freqs.offset,
        &mut codes.lens.offset,
        &mut codes.codewords.offset,
    );
}

/// C: `deflate_init_static_codes(struct libdeflate_compressor *c)` (:1433)
///
/// Initialize the static Huffman codes defined by the DEFLATE format, by feeding
/// `deflate_make_huffman_codes` a frequency table whose optimal code IS the static
/// code. The frequencies are `1 << (9 - len)` for each symbol's mandated length,
/// which makes the Huffman construction reproduce RFC 1951 section 3.2.6 exactly.
///
/// The C writes these frequencies into `c->freqs` (the live per-block table) and
/// relies on `deflate_reset_symbol_frequencies` clearing it before the first real
/// block. We take the scratch table as a parameter instead of hiding it — the
/// aliasing is a C storage economy, not behaviour, and nothing observes it.
///
/// Clippy flags `1 << (9 - 9)` and `1 << (5 - 5)` as `eq_op`. They are kept because
/// the `9 - len` / `5 - len` shape is what makes the four ranges READABLE as "these
/// symbols get an 8-bit codeword, these a 9-bit" — collapsing them to `1` and `1`
/// hides the only thing the reader needs to check against RFC 1951, and diverges from
/// the C line for line.
#[allow(clippy::eq_op)]
pub(crate) fn deflate_init_static_codes(freqs: &mut DeflateFreqs, static_codes: &mut DeflateCodes) {
    let mut i = 0usize;

    while i < 144 {
        freqs.litlen[i] = 1 << (9 - 8);
        i += 1;
    }
    while i < 256 {
        freqs.litlen[i] = 1 << (9 - 9);
        i += 1;
    }
    while i < 280 {
        freqs.litlen[i] = 1 << (9 - 7);
        i += 1;
    }
    while i < 288 {
        freqs.litlen[i] = 1 << (9 - 8);
        i += 1;
    }

    for i in 0..32 {
        freqs.offset[i] = 1 << (5 - 5);
    }

    deflate_make_huffman_codes(freqs, static_codes);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C: `STATIC_ASSERT(offsetof(struct deflate_lens, offset) ==
    /// DEFLATE_NUM_LITLEN_SYMS)` (:1593). The precode's memmove is only correct if
    /// this holds; if a future edit adds a field or reorders the struct, the flat
    /// view silently starts reading the wrong bytes.
    #[test]
    fn lens_layout_matches_the_c_static_assert() {
        let lens = DeflateLens::new();
        let base = &lens as *const DeflateLens as usize;
        let off = lens.offset.as_ptr() as usize;
        assert_eq!(
            off - base,
            DEFLATE_NUM_LITLEN_SYMS,
            "offsetof(lens, offset)"
        );
        assert_eq!(
            core::mem::size_of::<DeflateLens>(),
            NUM_LITLEN_PLUS_OFFSET_SYMS,
            "no padding"
        );
        assert_eq!(core::mem::align_of::<DeflateLens>(), 1);
    }

    /// The flat view must alias the fields, in both directions.
    #[test]
    fn flat_view_aliases_the_named_fields() {
        let mut lens = DeflateLens::new();
        lens.litlen[287] = 9;
        lens.offset[0] = 4;
        lens.offset[31] = 5;

        assert_eq!(lens.as_flat()[287], 9);
        assert_eq!(lens.as_flat()[288], 4);
        assert_eq!(lens.as_flat()[319], 5);

        lens.as_flat_mut()[288] = 7;
        assert_eq!(lens.offset[0], 7);
    }

    /// The static code must be RFC 1951 section 3.2.6 exactly: litlen lengths of
    /// 8/9/7/8 over the four ranges, offset lengths all 5. This is the real check on
    /// `deflate_init_static_codes` — if the frequency weighting were off, the
    /// Huffman construction would produce a *valid* code with different lengths and
    /// every static block we emit would be non-conformant.
    #[test]
    fn static_codes_match_rfc1951_section_3_2_6() {
        let mut freqs = DeflateFreqs::new();
        let mut codes = DeflateCodes::new();
        deflate_init_static_codes(&mut freqs, &mut codes);

        for i in 0..144 {
            assert_eq!(codes.lens.litlen[i], 8, "litlen {i}");
        }
        for i in 144..256 {
            assert_eq!(codes.lens.litlen[i], 9, "litlen {i}");
        }
        for i in 256..280 {
            assert_eq!(codes.lens.litlen[i], 7, "litlen {i}");
        }
        for i in 280..288 {
            assert_eq!(codes.lens.litlen[i], 8, "litlen {i}");
        }
        for i in 0..32 {
            assert_eq!(codes.lens.offset[i], 5, "offset {i}");
        }
    }

    /// RFC 1951 also pins the static codewords themselves: literal 0 is `00110000`
    /// (0x30) in 8 bits, literal 144 is `110010000` in 9 bits, symbol 256 is
    /// `0000000` in 7 bits, symbol 280 is `11000000` in 8 bits. Our codewords are
    /// stored BIT-REVERSED (LSB-first, ready for the bitstream), so compare against
    /// the reversal of the RFC's MSB-first values.
    #[test]
    fn static_codewords_match_rfc1951_bit_reversed() {
        let mut freqs = DeflateFreqs::new();
        let mut codes = DeflateCodes::new();
        deflate_init_static_codes(&mut freqs, &mut codes);

        let rev = |v: u32, len: u32| (v as u16).reverse_bits() as u32 >> (16 - len);

        assert_eq!(codes.codewords.litlen[0], rev(0b0011_0000, 8), "literal 0");
        assert_eq!(
            codes.codewords.litlen[143],
            rev(0b1011_1111, 8),
            "literal 143"
        );
        assert_eq!(
            codes.codewords.litlen[144],
            rev(0b1_1001_0000, 9),
            "literal 144"
        );
        assert_eq!(
            codes.codewords.litlen[255],
            rev(0b1_1111_1111, 9),
            "literal 255"
        );
        assert_eq!(
            codes.codewords.litlen[256],
            rev(0b000_0000, 7),
            "end-of-block"
        );
        assert_eq!(codes.codewords.litlen[279], rev(0b001_0111, 7), "sym 279");
        assert_eq!(codes.codewords.litlen[280], rev(0b1100_0000, 8), "sym 280");
        assert_eq!(codes.codewords.litlen[287], rev(0b1100_0111, 8), "sym 287");

        for i in 0..32u32 {
            assert_eq!(codes.codewords.offset[i as usize], rev(i, 5), "offset {i}");
        }
    }

    /// `reset` must clear both alphabets — a stale offset frequency would leak the
    /// previous block's statistics into this block's code.
    #[test]
    fn reset_clears_both_alphabets() {
        let mut freqs = DeflateFreqs::new();
        freqs.litlen[100] = 5;
        freqs.offset[7] = 9;
        freqs.reset();
        assert!(freqs.litlen.iter().all(|&f| f == 0));
        assert!(freqs.offset.iter().all(|&f| f == 0));
    }
}
