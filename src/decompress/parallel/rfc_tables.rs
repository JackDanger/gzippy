//! Literal port of `rapidgzip::deflate` RFC 1951 distance/length tables
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/RFCTables.hpp).
//!
//! Pure-data + pure-function module: distance LUT, length LUT, and the
//! helpers used to look up a base distance/length from a Huffman
//! symbol. These don't depend on the rest of the parallel stack and
//! intentionally avoid pulling in a BitReader — the vendor versions
//! that take a bit reader are split here into a "table lookup" half
//! and a "consume extra bits" half so callers can wire whichever
//! bit-reader they already own.
//!
//! Today, `block_finder.rs` and `deflate_block.rs` carry their own
//! ad-hoc length/distance lookup logic; this module is the rapidgzip-
//! faithful home for the lookup tables. A follow-up will reroute those
//! callers through here.

#![allow(dead_code)]

use super::gzip_definitions::MAX_DISTANCE_SYMBOL_COUNT;

// =====================================================================
// Distance table (RFCTables.hpp:11-65).
// =====================================================================

/// Mirror of `rapidgzip::deflate::calculateDistance` —
/// the three-argument overload (RFCTables.hpp:16-23).
///
/// Computes the actual back-reference distance from a distance code
/// (>= 4), its extra-bit count, and the extra-bit value bits. For
/// codes 0..=3 the distance is `code + 1` directly; this function
/// covers the general case.
#[inline]
pub const fn calculate_distance_with_extra(
    distance_code: u16,
    extra_bits_count: u8,
    extra_bits: u16,
) -> u16 {
    debug_assert!(distance_code >= 4);
    1u16 + (1u16 << (extra_bits_count + 1)) + ((distance_code % 2) << extra_bits_count) + extra_bits
}

/// Mirror of `rapidgzip::deflate::calculateDistanceExtraBits`
/// (RFCTables.hpp:26-30). For distance codes 0..=3 returns 0,
/// otherwise `(distance_code - 2) / 2`.
#[inline]
pub const fn calculate_distance_extra_bits(distance_code: u16) -> u16 {
    if distance_code <= 3 {
        0
    } else {
        (distance_code - 2) / 2
    }
}

/// Mirror of `rapidgzip::deflate::calculateDistance` — the
/// LUT-creation overload (RFCTables.hpp:39-45). Returns the base
/// distance that needs `extra_bits` *added* to yield the final value.
#[inline]
const fn calculate_distance_base(distance_code: u16) -> u16 {
    debug_assert!(distance_code >= 4);
    let extra_bits_count = calculate_distance_extra_bits(distance_code);
    1u16 + (1u16 << (extra_bits_count + 1)) + ((distance_code % 2) << extra_bits_count)
}

/// Mirror of `rapidgzip::deflate::DistanceLUT` (RFCTables.hpp:48) —
/// `std::array<uint16_t, 30>` indexed by distance code 0..29.
pub type DistanceLUT = [u16; MAX_DISTANCE_SYMBOL_COUNT as usize];

/// Mirror of `rapidgzip::deflate::createDistanceLUT`
/// (RFCTables.hpp:50-61). Codes 0..=3 store `code + 1` directly; the
/// remainder store the base value (extra bits added at decode time).
#[inline]
const fn create_distance_lut() -> DistanceLUT {
    let mut result = [0u16; MAX_DISTANCE_SYMBOL_COUNT as usize];
    let mut i: u16 = 0;
    while i < 4 {
        result[i as usize] = i + 1;
        i += 1;
    }
    while i < MAX_DISTANCE_SYMBOL_COUNT as u16 {
        result[i as usize] = calculate_distance_base(i);
        i += 1;
    }
    result
}

/// Mirror of `rapidgzip::deflate::distanceLUT` (RFCTables.hpp:64-65) —
/// `alignas(8) static constexpr`.
pub static DISTANCE_LUT: DistanceLUT = create_distance_lut();

// =====================================================================
// Length table (RFCTables.hpp:68-94).
// =====================================================================

/// Mirror of `rapidgzip::deflate::calculateLength` (RFCTables.hpp:71-77).
///
/// The input `code` is the *offset* into the length table, i.e.
/// `length_code - 261` for length codes 265..284. The vendor asserts
/// `code < 285 - 261 == 24`.
#[inline]
pub const fn calculate_length(code: u16) -> u16 {
    debug_assert!(code < 285 - 261);
    let extra_bits = code / 4;
    3u16 + (1u16 << (extra_bits + 2)) + ((code % 4) << extra_bits)
}

/// Mirror of `rapidgzip::deflate::LengthLUT` (RFCTables.hpp:80) —
/// `std::array<uint16_t, 285 - 261>` indexed by `length_code - 261`.
pub type LengthLUT = [u16; 285 - 261];

/// Mirror of `rapidgzip::deflate::createLengthLUT` (RFCTables.hpp:82-90).
#[inline]
const fn create_length_lut() -> LengthLUT {
    let mut result = [0u16; 285 - 261];
    let mut i: u16 = 0;
    while (i as usize) < result.len() {
        result[i as usize] = calculate_length(i);
        i += 1;
    }
    result
}

/// Mirror of `rapidgzip::deflate::lengthLUT` (RFCTables.hpp:93-94) —
/// `alignas(8) static constexpr`.
pub static LENGTH_LUT: LengthLUT = create_length_lut();

// =====================================================================
// Length lookup helpers (RFCTables.hpp:97-123).
// =====================================================================

/// Result of looking up a length code: base length and the number of
/// extra bits that still need to be consumed from the bit-stream and
/// added to `base_length`.
///
/// Vendor counterpart `getLength` (RFCTables.hpp:97-114) combines the
/// lookup and the bit-reader consume; gzippy splits them so callers
/// can use whichever BitReader they already own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthLookup {
    pub base_length: u16,
    pub extra_bits_count: u8,
}

/// "Pure" half of `rapidgzip::deflate::getLength` (RFCTables.hpp:97-114).
///
/// Returns `Some(LengthLookup)` for valid length codes 257..=285 and
/// `None` for invalid codes (vendor throws `std::invalid_argument`).
/// Codes 257..=264 have `extra_bits_count == 0` (literal lengths 3..=10).
/// Code 285 is the maximum length (258) with no extra bits.
#[inline]
pub const fn length_lookup(length_code: u16) -> Option<LengthLookup> {
    if length_code < 257 {
        return None;
    }
    if length_code <= 264 {
        // Lengths 3..=10 stored directly.
        return Some(LengthLookup {
            base_length: length_code - 257 + 3,
            extra_bits_count: 0,
        });
    }
    if length_code < 285 {
        let code = length_code - 261;
        let extra_bits = (code / 4) as u8;
        return Some(LengthLookup {
            base_length: calculate_length(code),
            extra_bits_count: extra_bits,
        });
    }
    if length_code == 285 {
        return Some(LengthLookup {
            base_length: 258,
            extra_bits_count: 0,
        });
    }
    None
}

// =====================================================================
// Distance lookup helpers (RFCTables.hpp:126-157).
// =====================================================================

/// Result of looking up a distance code: base value and how many extra
/// bits to read and add. Codes 0..=3 have `extra_bits_count == 0`,
/// codes 4..=29 have `(code - 2) / 2` extra bits.
///
/// Vendor counterpart `getDistance` (RFCTables.hpp:126-157) combines
/// the lookup with the bit-reader consume and a Huffman-coding lookup;
/// gzippy splits those concerns the same way as for lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistanceLookup {
    pub base_distance: u16,
    pub extra_bits_count: u8,
}

/// "Pure" half of `rapidgzip::deflate::getDistance` (RFCTables.hpp:146-153)
/// — the distance-code-to-(base, extra_count) lookup *after* the
/// Huffman / fixed-distance bit-read has produced the code.
///
/// Returns `None` if `distance_code >= MAX_DISTANCE_SYMBOL_COUNT` (30).
/// Codes 0..=3 are stored as `code + 1` per the LUT. The vendor's
/// `>= 30` branch throws `std::logic_error("Invalid distance codes")`;
/// gzippy surfaces it as `None`.
#[inline]
pub const fn distance_lookup(distance_code: u16) -> Option<DistanceLookup> {
    if distance_code >= MAX_DISTANCE_SYMBOL_COUNT as u16 {
        return None;
    }
    if distance_code <= 3 {
        return Some(DistanceLookup {
            base_distance: distance_code + 1,
            extra_bits_count: 0,
        });
    }
    let extra_bits_count = ((distance_code - 2) / 2) as u8;
    Some(DistanceLookup {
        base_distance: DISTANCE_LUT[distance_code as usize],
        extra_bits_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot check `distanceLUT` against a few hand-computed values.
    /// Codes 0..=3 are stored as code+1; code 4 (extra=1, base=5..=6),
    /// code 28 (extra=13) is the largest distance category before 29.
    #[test]
    fn distance_lut_matches_rfc1951() {
        assert_eq!(DISTANCE_LUT[0], 1);
        assert_eq!(DISTANCE_LUT[1], 2);
        assert_eq!(DISTANCE_LUT[2], 3);
        assert_eq!(DISTANCE_LUT[3], 4);
        // code 4: extra_bits=1, distance = 1 + 4 + 0 = 5
        assert_eq!(DISTANCE_LUT[4], 5);
        // code 5: extra_bits=1, distance = 1 + 4 + 2 = 7
        assert_eq!(DISTANCE_LUT[5], 7);
        // code 29: extra_bits=13, distance = 1 + 2^14 + 2^13 = 24577
        assert_eq!(DISTANCE_LUT[29], 24577);
    }

    /// Spot check `lengthLUT`. Code 0 here means RFC code 261
    /// (length range 7..10). The first few codes have extra=0.
    #[test]
    fn length_lut_matches_rfc1951() {
        // code 0 == RFC 261, length 7. calculateLength: 3 + 4 + 0 = 7
        assert_eq!(LENGTH_LUT[0], 7);
        // code 4 == RFC 265, length 11. calculateLength(4):
        //   extra=1, 3 + 8 + 0 = 11
        assert_eq!(LENGTH_LUT[4], 11);
    }

    /// `length_lookup(257..=264)` returns lengths 3..=10 with no extra.
    #[test]
    fn length_lookup_for_direct_codes() {
        for code in 257u16..=264 {
            let l = length_lookup(code).unwrap();
            assert_eq!(l.base_length, code - 257 + 3);
            assert_eq!(l.extra_bits_count, 0);
        }
    }

    /// `length_lookup(285)` is the special maximum length (258) with
    /// no extra bits.
    #[test]
    fn length_lookup_for_max_code() {
        let l = length_lookup(285).unwrap();
        assert_eq!(l.base_length, 258);
        assert_eq!(l.extra_bits_count, 0);
    }

    /// `length_lookup(265..285)` matches LUT + extra-bit count.
    #[test]
    fn length_lookup_for_lut_range() {
        for code in 265u16..285 {
            let l = length_lookup(code).unwrap();
            let i = code - 261;
            assert_eq!(l.base_length, LENGTH_LUT[i as usize]);
            assert_eq!(l.extra_bits_count, (i / 4) as u8);
        }
    }

    #[test]
    fn length_lookup_rejects_out_of_range() {
        assert_eq!(length_lookup(0), None);
        assert_eq!(length_lookup(256), None);
        assert_eq!(length_lookup(286), None);
    }

    #[test]
    fn distance_lookup_for_direct_codes() {
        for code in 0u16..=3 {
            let d = distance_lookup(code).unwrap();
            assert_eq!(d.base_distance, code + 1);
            assert_eq!(d.extra_bits_count, 0);
        }
    }

    #[test]
    fn distance_lookup_for_lut_range() {
        for code in 4u16..MAX_DISTANCE_SYMBOL_COUNT as u16 {
            let d = distance_lookup(code).unwrap();
            assert_eq!(d.base_distance, DISTANCE_LUT[code as usize]);
            assert_eq!(d.extra_bits_count, ((code - 2) / 2) as u8);
        }
    }

    #[test]
    fn distance_lookup_rejects_codes_30_and_above() {
        assert_eq!(distance_lookup(30), None);
        assert_eq!(distance_lookup(31), None);
    }

    /// Cross-check: calculateDistance with extra=0 equals the LUT value.
    #[test]
    fn calculate_distance_with_zero_extra_matches_lut() {
        for code in 4u16..MAX_DISTANCE_SYMBOL_COUNT as u16 {
            let extra_count = calculate_distance_extra_bits(code) as u8;
            let calc = calculate_distance_with_extra(code, extra_count, 0);
            assert_eq!(calc, DISTANCE_LUT[code as usize]);
        }
    }

    /// LUT sizes match the vendor's array typedefs.
    #[test]
    fn lut_sizes_match_vendor() {
        assert_eq!(DISTANCE_LUT.len(), 30);
        assert_eq!(LENGTH_LUT.len(), 24);
    }
}
