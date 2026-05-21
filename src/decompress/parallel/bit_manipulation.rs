//! Literal port of `rapidgzip::*` bit-manipulation primitives
//! (vendor/rapidgzip/librapidarchive/src/core/BitManipulation.hpp).
//!
//! ## Mapping to Rust intrinsics
//!
//! Rust provides a richer standard library here than C++17:
//! - `T::reverse_bits` is built-in and lowers to dedicated CPU
//!   instructions where available (e.g. ARMv8 `RBIT`).
//! - `T::swap_bytes` (and `T::from_le` / `T::to_be`) replaces the
//!   vendor's hand-rolled `byteSwap` overloads.
//! - `T::leading_zeros` / `T::trailing_zeros` replace any need to
//!   roll our own bit-scan.
//!
//! So this module deliberately ports the *algorithmic surface* of
//! `BitManipulation.hpp` — the functions that take a `bitCount`
//! parameter and produce 0/1 masks — rather than rebuilding the
//! 64-KiB `REVERSED_BITS_LUT<u16>` that the vendor uses to amortize
//! the bit-reverse cost in their hot Huffman decoder. The Rust
//! intrinsic is competitive with the LUT and avoids 64 KiB of L2
//! pressure. A future Huffman variant port may add the LUT back if
//! benchmarks show it helps; the wrapper functions here keep the
//! call site stable either way.
//!
//! Functions exposed:
//! - `is_little_endian` (BM.hpp:11-16)
//! - `byte_swap_u{16,32,64}` (BM.hpp:19-46) — thin alias to
//!   `T::swap_bytes` for vendor symbol parity.
//! - `n_lowest_bits_set` / `n_highest_bits_set` (BM.hpp:60-147) —
//!   produce a u64 mask with N low/high bits set.
//! - `reverse_bits_without_lut_u{8,16,32,64}` (BM.hpp:150-227) —
//!   the bit-parallel `mask-shift-OR` algorithm, verbatim. Useful
//!   in `const` contexts where `T::reverse_bits` isn't const.
//! - `reverse_bits` / `reverse_bits_count` (BM.hpp:270-294) —
//!   uses `T::reverse_bits` then shifts off the unused high bits.
//! - `required_bits` (BM.hpp:300-315) — ceil(log2(N)) with the
//!   N==0/N==1 special cases preserved.

#![allow(dead_code)]

/// Mirror of `rapidgzip::isLittleEndian` (BitManipulation.hpp:11-16).
///
/// Rust targets are always one of LE/BE per `cfg(target_endian)`;
/// this is a thin wrapper kept for vendor symbol parity.
pub const fn is_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Mirror of `rapidgzip::byteSwap(uint16_t)` (BitManipulation.hpp:41-46).
#[inline]
pub const fn byte_swap_u16(value: u16) -> u16 {
    value.swap_bytes()
}

/// Mirror of `rapidgzip::byteSwap(uint32_t)` (BitManipulation.hpp:32-38).
#[inline]
pub const fn byte_swap_u32(value: u32) -> u32 {
    value.swap_bytes()
}

/// Mirror of `rapidgzip::byteSwap(uint64_t)` (BitManipulation.hpp:19-29).
#[inline]
pub const fn byte_swap_u64(value: u64) -> u64 {
    value.swap_bytes()
}

/// Mirror of `rapidgzip::nLowestBitsSet<T>(uint8_t)`
/// (BitManipulation.hpp:60-73).
///
/// Returns a `u64` mask with the `n_bits_set` lowest bits set.
/// `n_bits_set == 0` returns 0; `n_bits_set >= 64` returns all-ones.
/// The vendor template is generic over the integer type; gzippy
/// specializes on `u64` because all current callers want at most
/// 64 bits of mask. Cast at the call site if narrower is wanted.
#[inline]
pub const fn n_lowest_bits_set(n_bits_set: u8) -> u64 {
    if n_bits_set == 0 {
        0
    } else if n_bits_set >= u64::BITS as u8 {
        u64::MAX
    } else {
        let n_zero_bits = u64::BITS as u8 - n_bits_set;
        u64::MAX >> n_zero_bits
    }
}

/// Mirror of `rapidgzip::nHighestBitsSet<T>(uint8_t)`
/// (BitManipulation.hpp:105-118).
#[inline]
pub const fn n_highest_bits_set(n_bits_set: u8) -> u64 {
    if n_bits_set == 0 {
        0
    } else if n_bits_set >= u64::BITS as u8 {
        u64::MAX
    } else {
        let n_zero_bits = u64::BITS as u8 - n_bits_set;
        u64::MAX << n_zero_bits
    }
}

/// Literal port of `rapidgzip::reverseBitsWithoutLUT(uint8_t)`
/// (BitManipulation.hpp:150-162). Bit-parallel mask-shift-OR.
#[inline]
pub const fn reverse_bits_without_lut_u8(mut data: u8) -> u8 {
    let masks: [u8; 3] = [0b0101_0101, 0b0011_0011, 0b0000_1111];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = masks[i] as u32;
        let shift = 1u32 << i;
        let d = data as u32;
        data = (((d & mask) << shift) | ((d & !mask) >> shift)) as u8;
        i += 1;
    }
    data
}

/// Literal port of `rapidgzip::reverseBitsWithoutLUT(uint16_t)`
/// (BitManipulation.hpp:165-182).
#[inline]
pub const fn reverse_bits_without_lut_u16(mut data: u16) -> u16 {
    let masks: [u16; 4] = [
        0b0101_0101_0101_0101,
        0b0011_0011_0011_0011,
        0b0000_1111_0000_1111,
        0b0000_0000_1111_1111,
    ];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = masks[i] as u32;
        let shift = 1u32 << i;
        let d = data as u32;
        data = (((d & mask) << shift) | ((d & !mask) >> shift)) as u16;
        i += 1;
    }
    data
}

/// Literal port of `rapidgzip::reverseBitsWithoutLUT(uint32_t)`
/// (BitManipulation.hpp:185-202).
#[inline]
pub const fn reverse_bits_without_lut_u32(mut data: u32) -> u32 {
    let masks: [u32; 5] = [
        0b0101_0101_0101_0101_0101_0101_0101_0101,
        0b0011_0011_0011_0011_0011_0011_0011_0011,
        0b0000_1111_0000_1111_0000_1111_0000_1111,
        0b0000_0000_1111_1111_0000_0000_1111_1111,
        0b0000_0000_0000_0000_1111_1111_1111_1111,
    ];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = masks[i];
        let shift = 1u32 << i;
        data = ((data & mask) << shift) | ((data & !mask) >> shift);
        i += 1;
    }
    data
}

/// Literal port of `rapidgzip::reverseBitsWithoutLUT(uint64_t)`
/// (BitManipulation.hpp:205-227).
#[inline]
pub const fn reverse_bits_without_lut_u64(mut data: u64) -> u64 {
    let masks: [u64; 6] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF,
    ];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = masks[i];
        let shift = 1u32 << i;
        data = ((data & mask) << shift) | ((data & !mask) >> shift);
        i += 1;
    }
    data
}

/// Mirror of `rapidgzip::reverseBits<T>(T)`
/// (BitManipulation.hpp:270-281). Reverses all bits of the value.
/// Delegates to Rust's built-in `reverse_bits` intrinsic — the
/// vendor uses a 64 KiB LUT for `u16`; benchmarks suggest the
/// intrinsic ties or wins on modern x86_64 / aarch64.
#[inline]
pub fn reverse_bits_u32(value: u32) -> u32 {
    value.reverse_bits()
}

/// Mirror of `rapidgzip::reverseBits<T>(T, uint8_t)`
/// (BitManipulation.hpp:287-294). Reverses the `bit_count` lowest
/// bits of `value`; the highest bits are zeroed and assumed-zero
/// in the input.
///
/// Vendor precondition: `bit_count > 0`. We mirror that with a
/// `debug_assert!`; in release builds, `bit_count == 0` returns 0
/// (Rust's `>>` on the full width is defined to return 0, not UB
/// like C++).
#[inline]
pub fn reverse_bits_count_u32(value: u32, bit_count: u8) -> u32 {
    debug_assert!(bit_count > 0, "vendor precondition: bit_count > 0");
    if bit_count == 0 {
        return 0;
    }
    value.reverse_bits() >> (u32::BITS as u8 - bit_count)
}

/// Same as `reverse_bits_count_u32` but for `u16`.
#[inline]
pub fn reverse_bits_count_u16(value: u16, bit_count: u8) -> u16 {
    debug_assert!(bit_count > 0, "vendor precondition: bit_count > 0");
    if bit_count == 0 {
        return 0;
    }
    value.reverse_bits() >> (u16::BITS as u8 - bit_count)
}

/// Same as `reverse_bits_count_u32` but for `u64`.
#[inline]
pub fn reverse_bits_count_u64(value: u64, bit_count: u8) -> u64 {
    debug_assert!(bit_count > 0, "vendor precondition: bit_count > 0");
    if bit_count == 0 {
        return 0;
    }
    value.reverse_bits() >> (u64::BITS as u8 - bit_count)
}

/// Literal port of `rapidgzip::requiredBits`
/// (BitManipulation.hpp:300-315). `ceil(log2(state_count))` with
/// the special cases `state_count == 0 -> 0` and
/// `state_count == 1 -> 1` preserved exactly (the vendor's
/// `state_count == 1` returns 1, not 0; we mirror that).
#[inline]
pub const fn required_bits(state_count: u64) -> u8 {
    if state_count == 0 {
        return 0;
    }
    if state_count == 1 {
        return 1;
    }
    let mut result: u8 = 0;
    let mut max_value = state_count - 1;
    while max_value != 0 {
        result += 1;
        max_value >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endianness_matches_target_cfg() {
        // Rust targets are always one of LE/BE.
        assert_eq!(is_little_endian(), cfg!(target_endian = "little"));
    }

    #[test]
    fn byte_swap_round_trip() {
        for v in [0u16, 1, 0x1234, 0xFF00, u16::MAX] {
            assert_eq!(byte_swap_u16(byte_swap_u16(v)), v);
        }
        for v in [0u32, 1, 0x1234_5678, u32::MAX] {
            assert_eq!(byte_swap_u32(byte_swap_u32(v)), v);
        }
        for v in [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX] {
            assert_eq!(byte_swap_u64(byte_swap_u64(v)), v);
        }
    }

    #[test]
    fn byte_swap_known_values() {
        assert_eq!(byte_swap_u16(0x1234), 0x3412);
        assert_eq!(byte_swap_u32(0x12345678), 0x78563412);
        assert_eq!(byte_swap_u64(0x0123_4567_89AB_CDEF), 0xEFCD_AB89_6745_2301);
    }

    #[test]
    fn n_lowest_bits_set_boundary_cases() {
        assert_eq!(n_lowest_bits_set(0), 0);
        assert_eq!(n_lowest_bits_set(1), 1);
        assert_eq!(n_lowest_bits_set(8), 0xFF);
        assert_eq!(n_lowest_bits_set(63), (1u64 << 63) - 1);
        assert_eq!(n_lowest_bits_set(64), u64::MAX);
        // Vendor: "n_bits_set >= digits returns ~T(0)".
        assert_eq!(n_lowest_bits_set(65), u64::MAX);
        assert_eq!(n_lowest_bits_set(200), u64::MAX);
    }

    #[test]
    fn n_highest_bits_set_boundary_cases() {
        assert_eq!(n_highest_bits_set(0), 0);
        assert_eq!(n_highest_bits_set(1), 1u64 << 63);
        assert_eq!(n_highest_bits_set(8), 0xFF00_0000_0000_0000);
        assert_eq!(n_highest_bits_set(64), u64::MAX);
        assert_eq!(n_highest_bits_set(200), u64::MAX);
    }

    #[test]
    fn reverse_bits_without_lut_matches_intrinsic_u8() {
        for v in 0u8..=255 {
            assert_eq!(reverse_bits_without_lut_u8(v), v.reverse_bits());
        }
    }

    #[test]
    fn reverse_bits_without_lut_matches_intrinsic_u16_spot() {
        for v in [0u16, 1, 0x0F0F, 0xFFFF, 0x1234, 0xABCD] {
            assert_eq!(reverse_bits_without_lut_u16(v), v.reverse_bits());
        }
    }

    #[test]
    fn reverse_bits_without_lut_matches_intrinsic_u32_spot() {
        for v in [0u32, 1, 0xDEAD_BEEF, u32::MAX, 0x1234_5678] {
            assert_eq!(reverse_bits_without_lut_u32(v), v.reverse_bits());
        }
    }

    #[test]
    fn reverse_bits_without_lut_matches_intrinsic_u64_spot() {
        for v in [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX] {
            assert_eq!(reverse_bits_without_lut_u64(v), v.reverse_bits());
        }
    }

    #[test]
    fn reverse_bits_count_matches_huffman_semantics() {
        // Vendor: `reverseBits<u16>(0b0000_0000_0000_0110, 3)` = 0b011 = 3.
        // The lowest 3 bits of value are 110; reversed = 011 = 3.
        assert_eq!(reverse_bits_count_u32(0b110, 3), 0b011);
        assert_eq!(reverse_bits_count_u16(0b110, 3), 0b011);

        // 8-bit code 0b1011_0010 reversed = 0b0100_1101.
        assert_eq!(reverse_bits_count_u32(0b1011_0010, 8), 0b0100_1101);
    }

    #[test]
    fn required_bits_special_cases_match_vendor() {
        // Vendor: state_count == 0 -> 0
        assert_eq!(required_bits(0), 0);
        // Vendor: state_count == 1 -> 1 (NOT 0!)
        assert_eq!(required_bits(1), 1);
        assert_eq!(required_bits(2), 1);
        assert_eq!(required_bits(3), 2);
        assert_eq!(required_bits(4), 2);
        assert_eq!(required_bits(5), 3);
        assert_eq!(required_bits(255), 8);
        assert_eq!(required_bits(256), 8);
        assert_eq!(required_bits(257), 9);
        assert_eq!(required_bits(u64::MAX), 64);
    }
}
