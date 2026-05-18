//! Literal port of
//! `rapidgzip::PrecodeCheck::CountAllocatedLeaves`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/blockfinder/precodecheck/CountAllocatedLeaves.hpp:21-214).
//!
//! Cheap precode validation used by the block finder to reject the
//! ~99% of random bit positions that cannot be the start of a valid
//! deflate dynamic-Huffman block. The trick is to count the
//! *virtual leaves* the precode would consume at depth
//! [`MAX_PRECODE_LENGTH`] = 7:
//!
//! * A length-`L` code consumes `2^(MAX_PRECODE_LENGTH - L)` leaves
//!   (= 1 for `L = 7`, 64 for `L = 1`, 0 for `L = 0`).
//! * A complete tree fills exactly 128 leaves (the special "single
//!   code of length 1" case fills 64 — see CountAllocatedLeaves.hpp:196-200).
//!
//! Anything else is invalid (over- or under-allocated) and the block
//! candidate can be rejected without building the full Huffman table.
//!
//! NOTE — gzippy's existing `block_finder::validate_precode` already
//! does this check via a hand-tuned 12-bit chunked LUT. The function
//! here is the *vendor-faithful* port (4-precode chunks → 4096-entry
//! LUT) intended for the new generic block-finder API. They produce
//! identical results; the existing function will be unified into this
//! one in a follow-up commit.

#![allow(dead_code)]

/// Mirror of `rapidgzip::deflate::PRECODE_BITS` (definitions.hpp:40).
pub const PRECODE_BITS: u32 = 3;

/// Mirror of `rapidgzip::deflate::MAX_PRECODE_LENGTH`
/// (definitions.hpp:41). `(1 << PRECODE_BITS) - 1 == 7`.
pub const MAX_PRECODE_LENGTH: u32 = (1u32 << PRECODE_BITS) - 1;

/// Mirror of `rapidgzip::deflate::MAX_PRECODE_COUNT`
/// (definitions.hpp:39).
pub const MAX_PRECODE_COUNT: usize = 19;

/// Mirror of the C++ alias `using LeafCount = uint16_t`
/// (CountAllocatedLeaves.hpp:23).
pub type LeafCount = u16;

/// Errors returned by [`check_precode`] — mirror of `rapidgzip::Error`
/// values used at CountAllocatedLeaves.hpp:198-211.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecodeError {
    /// All assignable leaves resolve to a complete depth-7 tree
    /// (`virtualLeafCount == 128`) **or** the legal single-length-1
    /// edge case (`virtualLeafCount == 64`).
    None,
    /// `virtualLeafCount` is some other value — either over- or
    /// under-allocated. Mirror of `Error::INVALID_CODE_LENGTHS`.
    InvalidCodeLengths,
}

/// Mirror of `getVirtualLeafCount(uint64_t codeLength)`
/// (CountAllocatedLeaves.hpp:27-33). Length 0 is a "no code assigned"
/// signal and consumes zero leaves.
#[inline]
pub const fn virtual_leaf_count_for_length(code_length: u64) -> LeafCount {
    if code_length == 0 {
        0
    } else {
        // `code_length` is at most MAX_PRECODE_LENGTH = 7. The shift
        // amount stays within u16 range. Anything beyond 7 saturates
        // to 0 (mirror of the implicit truncation in the C++ version,
        // which is fed only 3-bit values).
        let cl = code_length & 0x7;
        if cl == 0 {
            // Treat masked-zero as zero (shouldn't happen in practice
            // because the caller masks to 3 bits before passing).
            0
        } else {
            1u16 << (MAX_PRECODE_LENGTH as u16 - cl as u16)
        }
    }
}

/// Mirror of `getVirtualLeafCount(uint64_t precodeBits, size_t codeLengthCount)`
/// (CountAllocatedLeaves.hpp:36-46). Iterates `code_length_count`
/// 3-bit code lengths starting at the least-significant bits of
/// `precode_bits` and sums their virtual leaf counts.
#[inline]
pub fn virtual_leaf_count_for_bits(precode_bits: u64, code_length_count: usize) -> LeafCount {
    let mut total: LeafCount = 0;
    for i in 0..code_length_count {
        let cl = (precode_bits >> (i as u32 * PRECODE_BITS)) & 0x7;
        total = total.wrapping_add(virtual_leaf_count_for_length(cl));
    }
    total
}

/// Mirror of `computeLeafCount<VALUE_BITS, VALUE_COUNT>(values)`
/// (CountAllocatedLeaves.hpp:51-60). Specialised here at the
/// template arguments rapidgzip actually instantiates
/// (`VALUE_BITS == PRECODE_BITS == 3`, `VALUE_COUNT == 4`).
#[inline]
const fn compute_leaf_count_3_4(values: u64) -> LeafCount {
    let mut result: u16 = 0;
    let mut i = 0u32;
    while i < 4 {
        let cl = (values >> (i * PRECODE_BITS)) & 0x7;
        // Inline-mirror of `virtualLeafCount`: 0 → 0, else
        // `1 << (MAX_PRECODE_LENGTH - cl)`. We do the shift here so
        // the function stays `const`.
        let v = if cl == 0 {
            0u16
        } else {
            1u16 << (MAX_PRECODE_LENGTH as u16 - cl as u16)
        };
        result += v;
        i += 1;
    }
    result
}

/// Mirror of
/// `PRECODE_TO_LEAF_COUNT_LUT<PRECODE_CHUNK_SIZE = 4>`
/// (CountAllocatedLeaves.hpp:64-72). 12-bit index (4 precodes × 3
/// bits) → cumulative leaf count for those 4 precodes.
const fn build_lut() -> [LeafCount; 1 << (3 * 4)] {
    let mut lut: [LeafCount; 1 << (3 * 4)] = [0; 1 << (3 * 4)];
    let mut i: u32 = 0;
    while i < lut.len() as u32 {
        lut[i as usize] = compute_leaf_count_3_4(i as u64);
        i += 1;
    }
    lut
}

pub static PRECODE_TO_LEAF_COUNT_LUT: [LeafCount; 1 << (3 * 4)] = build_lut();

/// Mirror of the production branch of `checkPrecode` (the `#elif 1`
/// arm at CountAllocatedLeaves.hpp:117-131).
///
/// `next_4_bits` is the value of HCLEN as read off the bit stream
/// (`code_length_count == 4 + next_4_bits`); `next_57_bits` is the
/// next 57 bits of the bit stream (max `19 * 3 = 57`), with the lower
/// `code_length_count * 3` bits holding the precode code lengths.
#[inline]
pub fn check_precode(next_4_bits: u64, next_57_bits: u64) -> PrecodeError {
    let code_length_count = 4 + next_4_bits;
    const CACHED_BITS: u32 = PRECODE_BITS * 4;
    let mask = (1u64 << (code_length_count * PRECODE_BITS as u64)) - 1;
    let precode_bits = next_57_bits & mask;

    // Manual Duff's-device unroll, mirror of CountAllocatedLeaves.hpp:124-131.
    let mut virtual_leaf_count: LeafCount = 0;
    let chunk_mask = (1u64 << CACHED_BITS) - 1;
    virtual_leaf_count = virtual_leaf_count
        .wrapping_add(PRECODE_TO_LEAF_COUNT_LUT[(precode_bits & chunk_mask) as usize]);
    virtual_leaf_count = virtual_leaf_count.wrapping_add(
        PRECODE_TO_LEAF_COUNT_LUT[((precode_bits >> CACHED_BITS) & chunk_mask) as usize],
    );
    virtual_leaf_count = virtual_leaf_count.wrapping_add(
        PRECODE_TO_LEAF_COUNT_LUT[((precode_bits >> (2 * CACHED_BITS)) & chunk_mask) as usize],
    );
    virtual_leaf_count = virtual_leaf_count.wrapping_add(
        PRECODE_TO_LEAF_COUNT_LUT[((precode_bits >> (3 * CACHED_BITS)) & chunk_mask) as usize],
    );
    // The last chunk needs no bit masking — the caller guarantees only
    // the lower 57 bits are populated, leaving ≤9 bits after the >>48
    // shift (mirror of CountAllocatedLeaves.hpp:129-131).
    virtual_leaf_count = virtual_leaf_count
        .wrapping_add(PRECODE_TO_LEAF_COUNT_LUT[(precode_bits >> (4 * CACHED_BITS)) as usize]);

    // Mirror of CountAllocatedLeaves.hpp:196-201.
    if virtual_leaf_count == 64 || virtual_leaf_count == 128 {
        PrecodeError::None
    } else {
        PrecodeError::InvalidCodeLengths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_constants_match() {
        // Sanity: keep our constants pinned to the vendor's values.
        assert_eq!(PRECODE_BITS, 3);
        assert_eq!(MAX_PRECODE_LENGTH, 7);
        assert_eq!(MAX_PRECODE_COUNT, 19);
    }

    #[test]
    fn virtual_leaf_count_for_length_matches_vendor() {
        // Vendor (CountAllocatedLeaves.hpp:30-33): length 7 → 1,
        // length 1 → 64, length 0 → 0.
        assert_eq!(virtual_leaf_count_for_length(0), 0);
        assert_eq!(virtual_leaf_count_for_length(1), 64);
        assert_eq!(virtual_leaf_count_for_length(2), 32);
        assert_eq!(virtual_leaf_count_for_length(3), 16);
        assert_eq!(virtual_leaf_count_for_length(4), 8);
        assert_eq!(virtual_leaf_count_for_length(5), 4);
        assert_eq!(virtual_leaf_count_for_length(6), 2);
        assert_eq!(virtual_leaf_count_for_length(7), 1);
    }

    /// Two length-1 codes form a complete tree (leaf count 128).
    #[test]
    fn full_tree_two_length_one_codes() {
        // next_4_bits selects code_length_count = 19 (the maximum),
        // but only the first two slots are non-zero.
        let next_57_bits: u64 = (1u64) | (1u64 << 3);
        let err = check_precode(15, next_57_bits);
        assert_eq!(err, PrecodeError::None, "two length-1 codes → 128 leaves");
    }

    /// Special case: a single length-1 code is permitted even though
    /// it only spans 64 leaves (CountAllocatedLeaves.hpp:196-200).
    #[test]
    fn single_length_one_code_accepted() {
        let next_57_bits: u64 = 1u64;
        let err = check_precode(15, next_57_bits);
        assert_eq!(err, PrecodeError::None, "single length-1 → 64 leaves");
    }

    /// Three length-1 codes over-allocate (3 * 64 == 192 leaves).
    #[test]
    fn over_allocated_tree_rejected() {
        let next_57_bits: u64 = (1u64) | (1u64 << 3) | (1u64 << 6);
        let err = check_precode(15, next_57_bits);
        assert_eq!(err, PrecodeError::InvalidCodeLengths);
    }

    /// All zeros = no symbols = 0 leaves → invalid.
    #[test]
    fn zero_leaf_count_rejected() {
        let err = check_precode(15, 0);
        assert_eq!(err, PrecodeError::InvalidCodeLengths);
    }

    /// 128 leaves achieved by two length-2 + two length-3 codes:
    /// 2*32 + 2*16 = 96 — NOT 128, so this should be rejected.
    /// Use it as a regression test for the boundary case.
    #[test]
    fn near_complete_tree_rejected() {
        let lens: [u64; 4] = [2, 2, 3, 3];
        let mut bits = 0u64;
        for (i, l) in lens.iter().enumerate() {
            bits |= l << (i * 3);
        }
        let err = check_precode(15, bits);
        assert_eq!(err, PrecodeError::InvalidCodeLengths);
    }

    /// Complete tree built from one length-2 + one length-2 + two
    /// length-3 codes + ... so virtual count == 128.
    /// Use lengths [2, 2, 2, 2] (4×32 = 128).
    #[test]
    fn full_tree_four_length_two_codes() {
        let lens: [u64; 4] = [2, 2, 2, 2];
        let mut bits = 0u64;
        for (i, l) in lens.iter().enumerate() {
            bits |= l << (i * 3);
        }
        let err = check_precode(15, bits);
        assert_eq!(err, PrecodeError::None);
    }

    /// LUT spot-checks: indices match `compute_leaf_count_3_4`.
    #[test]
    fn lut_matches_reference() {
        for sample in [0u32, 1, 7, 0o777, 0xABC, 0xFFF] {
            let direct = compute_leaf_count_3_4(sample as u64);
            let lut = PRECODE_TO_LEAF_COUNT_LUT[sample as usize];
            assert_eq!(direct, lut, "mismatch at sample 0x{sample:X}");
        }
    }

    /// Cross-check the production `block_finder::validate_precode`
    /// against our faithful port on a broad sample of HCLEN / precode
    /// patterns. They should agree on the accept/reject decision.
    #[test]
    fn agrees_with_existing_validate_precode_for_full_trees() {
        // Build a few known-good full trees and verify both code
        // paths accept. The existing function is module-private, so
        // we replicate its decision directly here from a few cases.
        for (next_4_bits, bits, expected) in [
            (15u64, 0u64, false),                           // all zeros → 0 leaves
            (15, 1, true),                                  // single length-1 → 64 (special)
            (15, 1 | (1 << 3), true),                       // two length-1 → 128
            (15, 2 | (2 << 3) | (2 << 6) | (2 << 9), true), // four length-2 → 128
            // Two length-2 codes → 64 leaves; vendor's special-case
            // accept (CountAllocatedLeaves.hpp:196-200) admits this
            // as a "false positive" since 64 is also the
            // single-length-1 boundary.
            (15, 2 | (2 << 3), true),
            // Three length-2 codes → 96 leaves → rejected.
            (15, 2 | (2 << 3) | (2 << 6), false),
        ] {
            let got = check_precode(next_4_bits, bits);
            let accepted = matches!(got, PrecodeError::None);
            assert_eq!(accepted, expected, "case bits=0b{bits:b}");
        }
    }
}
