//! C: `vendor/libdeflate/lib/deflate_compress.c:1640` —
//! `deflate_compute_full_len_codewords`.
//!
//! To make it faster to output matches, compute the "full" match length codewords,
//! i.e. the concatenation of the litlen codeword and the extra bits for each possible
//! match length. `WRITE_MATCH` (:1662) then emits a match length with a single
//! `ADD_BITS(o.length.codewords[len], o.length.lens[len])` instead of a table lookup,
//! a subtraction and two adds.

use super::codes::DeflateCodes;
use super::tables::{DEFLATE_EXTRA_LENGTH_BITS, DEFLATE_LENGTH_SLOT, DEFLATE_LENGTH_SLOT_BASE};
use super::{
    DEFLATE_FIRST_LEN_SYM, DEFLATE_MAX_EXTRA_LENGTH_BITS, DEFLATE_MAX_MATCH_LEN,
    DEFLATE_MIN_MATCH_LEN, MAX_LITLEN_CODEWORD_LEN,
};

/// C: the `length` arm of the `union o` in `struct libdeflate_compressor` (:511).
///
/// The C unions this with [`super::precode::Precode`] because the precode information
/// is dead by the time the full length codewords are needed. We keep them separate —
/// see the note on `Precode` for why the union is storage economy, not behaviour.
pub(crate) struct FullLengthCodewords {
    pub(crate) codewords: [u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
    pub(crate) lens: [u8; DEFLATE_MAX_MATCH_LEN as usize + 1],
}

impl FullLengthCodewords {
    pub(crate) const fn new() -> Self {
        Self {
            codewords: [0; DEFLATE_MAX_MATCH_LEN as usize + 1],
            lens: [0; DEFLATE_MAX_MATCH_LEN as usize + 1],
        }
    }
}

/// C: `deflate_compute_full_len_codewords(struct libdeflate_compressor *c,
/// const struct deflate_codes *codes)` (:1640)
///
/// # Why the extra bits shift LEFT by the codeword length
///
/// Codewords are stored bit-reversed, LSB-first, ready to be shifted straight into
/// the bit buffer. The litlen codeword goes out first and the extra length bits
/// follow, so in an LSB-first buffer the extra bits occupy the positions ABOVE the
/// codeword: `codeword | (extra_bits << codeword_len)`. Note the extra bits are NOT
/// reversed — RFC 1951 says extra bits are written LSB-first as an integer, unlike
/// Huffman codewords which are written MSB-first. Reversing them, or shifting by the
/// wrong length, produces a stream that decodes to different match lengths.
///
/// The C's `STATIC_ASSERT(MAX_LITLEN_CODEWORD_LEN + DEFLATE_MAX_EXTRA_LENGTH_BITS
/// <= 32)` is what makes the packed `u32` safe: 14 + 5 = 19 bits.
#[inline(always)]
pub(crate) fn deflate_compute_full_len_codewords(
    out: &mut FullLengthCodewords,
    codes: &DeflateCodes,
) {
    const _: () = assert!(MAX_LITLEN_CODEWORD_LEN + DEFLATE_MAX_EXTRA_LENGTH_BITS as usize <= 32);

    for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
        let slot = DEFLATE_LENGTH_SLOT[len as usize] as usize;
        let litlen_sym = (DEFLATE_FIRST_LEN_SYM + slot as u32) as usize;
        let extra_bits = len - DEFLATE_LENGTH_SLOT_BASE[slot];

        out.codewords[len as usize] =
            codes.codewords.litlen[litlen_sym] | (extra_bits << codes.lens.litlen[litlen_sym]);
        out.lens[len as usize] = codes.lens.litlen[litlen_sym] + DEFLATE_EXTRA_LENGTH_BITS[slot];
    }
}

#[cfg(test)]
mod tests {
    use super::super::codes::{deflate_init_static_codes, DeflateFreqs};
    use super::*;

    /// Decode a packed full-length codeword the way the bitstream reader would:
    /// strip the litlen codeword from the low bits (reversing it back to MSB-first),
    /// then read the extra bits as a plain integer above it.
    ///
    /// This is the oracle — it reconstructs the match length from the emitted bits
    /// using only RFC 1951, with no reference to how the packing was built.
    fn decode_full_len(packed: u32, total_len: u8, codes: &DeflateCodes) -> u32 {
        // Find the litlen symbol whose (reversed) codeword is a prefix of `packed`.
        for sym in DEFLATE_FIRST_LEN_SYM as usize..DEFLATE_FIRST_LEN_SYM as usize + 29 {
            let cl = codes.lens.litlen[sym];
            if cl == 0 {
                continue;
            }
            let mask = (1u32 << cl) - 1;
            if packed & mask != codes.codewords.litlen[sym] {
                continue;
            }
            let slot = sym - DEFLATE_FIRST_LEN_SYM as usize;
            let extra_width = DEFLATE_EXTRA_LENGTH_BITS[slot];
            if cl + extra_width != total_len {
                continue;
            }
            let extra = packed >> cl;
            assert!(
                extra_width == 0 || extra < (1u32 << extra_width),
                "extra {extra} exceeds {extra_width} bits"
            );
            return DEFLATE_LENGTH_SLOT_BASE[slot] + extra;
        }
        panic!("no litlen symbol matches packed={packed:#x} len={total_len}");
    }

    /// Every match length 3..=258 must round-trip through the packed representation
    /// under the STATIC Huffman code, where all litlen codeword lengths are known
    /// from RFC 1951.
    #[test]
    fn full_len_codewords_round_trip_under_the_static_code() {
        let mut freqs = DeflateFreqs::new();
        let mut codes = super::super::codes::DeflateCodes::new();
        deflate_init_static_codes(&mut freqs, &mut codes);

        let mut out = FullLengthCodewords::new();
        deflate_compute_full_len_codewords(&mut out, &codes);

        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let got = decode_full_len(out.codewords[len as usize], out.lens[len as usize], &codes);
            assert_eq!(got, len, "length {len} did not round-trip");
        }
    }

    /// The same, under a lopsided DYNAMIC code — the static code gives every length
    /// symbol a 7- or 8-bit codeword, which would hide a shift-by-the-wrong-length
    /// bug in the lengths that share a width.
    #[test]
    fn full_len_codewords_round_trip_under_a_skewed_dynamic_code() {
        let mut freqs = DeflateFreqs::new();
        // Wildly unequal length-symbol frequencies force codeword lengths to spread
        // across the whole 1..=14 range.
        //
        // The total must stay under `1 << NUM_FREQ_BITS` (2^22): frequencies are
        // packed into the high 22 bits of a u32 alongside the symbol, and `build_tree`
        // ADDS packed frequencies. The C guarantees this with
        // `STATIC_ASSERT(MAX_BLOCK_LENGTH <= ...)`; a test that exceeds it panics in
        // debug Rust (and would silently wrap in C). This geometric spread sums to
        // ~65 K, four orders of magnitude clear of the limit.
        freqs.litlen[256] = 1;
        for slot in 0..29u32 {
            let sym = (DEFLATE_FIRST_LEN_SYM + slot) as usize;
            freqs.litlen[sym] = 1u32 << (14 - slot / 2);
        }
        freqs.litlen[b'a' as usize] = 1 << 13;

        let mut codes = super::super::codes::DeflateCodes::new();
        super::super::codes::deflate_make_huffman_codes(&freqs, &mut codes);

        // The skew must actually have produced a spread, or the test proves nothing.
        let widths: Vec<u8> = (0..29)
            .map(|s| codes.lens.litlen[(DEFLATE_FIRST_LEN_SYM + s) as usize])
            .collect();
        let min = *widths.iter().filter(|&&w| w != 0).min().unwrap();
        let max = *widths.iter().max().unwrap();
        assert!(
            max - min >= 4,
            "codeword widths {min}..{max} are not spread"
        );

        let mut out = FullLengthCodewords::new();
        deflate_compute_full_len_codewords(&mut out, &codes);

        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let got = decode_full_len(out.codewords[len as usize], out.lens[len as usize], &codes);
            assert_eq!(got, len, "length {len} did not round-trip");
        }
    }

    /// The packed value must never exceed 19 bits (14-bit litlen cap + 5 extra), and
    /// the total length must never exceed 19 either. This is the C's STATIC_ASSERT
    /// turned into a runtime check over the whole domain.
    #[test]
    fn packed_codewords_stay_within_19_bits() {
        let mut freqs = DeflateFreqs::new();
        freqs.litlen[256] = 1;
        for slot in 0..29u32 {
            freqs.litlen[(DEFLATE_FIRST_LEN_SYM + slot) as usize] = 1u32 << (14 - slot / 2);
        }
        let mut codes = super::super::codes::DeflateCodes::new();
        super::super::codes::deflate_make_huffman_codes(&freqs, &mut codes);

        let mut out = FullLengthCodewords::new();
        deflate_compute_full_len_codewords(&mut out, &codes);

        let limit = MAX_LITLEN_CODEWORD_LEN as u8 + DEFLATE_MAX_EXTRA_LENGTH_BITS as u8;
        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let l = out.lens[len as usize];
            assert!(l <= limit, "length {len}: {l} bits exceeds {limit}");
            assert!(
                out.codewords[len as usize] < (1u32 << l),
                "length {len}: codeword has bits above its own length"
            );
        }
    }

    /// Length 258 is the format's irregular slot: slot 28, base 258, zero extra bits.
    /// Its packed form must be the bare litlen codeword.
    #[test]
    fn length_258_carries_no_extra_bits() {
        let mut freqs = DeflateFreqs::new();
        let mut codes = super::super::codes::DeflateCodes::new();
        deflate_init_static_codes(&mut freqs, &mut codes);

        let mut out = FullLengthCodewords::new();
        deflate_compute_full_len_codewords(&mut out, &codes);

        let sym = (DEFLATE_FIRST_LEN_SYM + 28) as usize;
        assert_eq!(out.codewords[258], codes.codewords.litlen[sym]);
        assert_eq!(out.lens[258], codes.lens.litlen[sym]);
    }
}
