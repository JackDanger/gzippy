//! Literal port of `rapidgzip::gzip` + `rapidgzip::deflate` constants
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/definitions.hpp).
//!
//! The centralized, rapidgzip-shaped home for the deflate format
//! constants.

#![allow(dead_code)]

// =====================================================================
// rapidgzip top-level (definitions.hpp:25).
// =====================================================================

/// Mirror of `rapidgzip::BYTE_SIZE` (definitions.hpp:25).
///
/// > Note that this describes bytes in the data format not on the host
/// > system, which is `CHAR_BIT` and might differ.
pub const BYTE_SIZE: u32 = 8;

// =====================================================================
// rapidgzip::deflate namespace (definitions.hpp:29-86).
// =====================================================================

/// Mirror of `rapidgzip::deflate::MAX_WINDOW_SIZE` (definitions.hpp:31).
/// 32 KiB sliding window, per RFC 1951.
pub const MAX_WINDOW_SIZE: usize = 32 * 1024;

/// Mirror of `rapidgzip::deflate::MAX_UNCOMPRESSED_SIZE`
/// (definitions.hpp:33). `std::numeric_limits<uint16_t>::max()` —
/// the length of an uncompressed block is a 16-bit number.
pub const MAX_UNCOMPRESSED_SIZE: usize = u16::MAX as usize;

/// Mirror of `rapidgzip::deflate::MAX_CODE_LENGTH` (definitions.hpp:35).
/// > This is because the code length alphabet can't encode any higher
/// > value and because length 0 is ignored!
pub const MAX_CODE_LENGTH: u8 = 15;

// ---------- Precode constants (definitions.hpp:37-45) ----------

/// Mirror of `rapidgzip::deflate::PRECODE_COUNT_BITS`
/// (definitions.hpp:38). Number of bits to encode the precode count.
pub const PRECODE_COUNT_BITS: u32 = 4;

/// Mirror of `rapidgzip::deflate::MAX_PRECODE_COUNT`
/// (definitions.hpp:39). The maximum precode count (19, also the size
/// of the precode alphabet).
pub const MAX_PRECODE_COUNT: u32 = 19;

/// Mirror of `rapidgzip::deflate::PRECODE_BITS` (definitions.hpp:40).
/// Number of bits per precode (code length).
pub const PRECODE_BITS: u32 = 3;

/// Mirror of `rapidgzip::deflate::MAX_PRECODE_LENGTH`
/// (definitions.hpp:41-42). `( 1 << PRECODE_BITS ) - 1 == 7`.
pub const MAX_PRECODE_LENGTH: u32 = (1u32 << PRECODE_BITS) - 1;
const _: () = assert!(MAX_PRECODE_LENGTH == 7);

/// Mirror of `rapidgzip::deflate::PRECODE_ALPHABET`
/// (definitions.hpp:43-45). The RFC 1951 precode-symbol permutation
/// order.
#[repr(align(8))]
pub struct PrecodeAlphabet(pub [u8; MAX_PRECODE_COUNT as usize]);

pub static PRECODE_ALPHABET: PrecodeAlphabet = PrecodeAlphabet([
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
]);

/// Mirror of `rapidgzip::deflate::MAX_LITERAL_OR_LENGTH_SYMBOLS`
/// (definitions.hpp:47).
pub const MAX_LITERAL_OR_LENGTH_SYMBOLS: usize = 286;

/// Mirror of `rapidgzip::deflate::MAX_DISTANCE_SYMBOL_COUNT`
/// (definitions.hpp:53). RFC 1951 §3.2.7 lists distance codes 1-32,
/// but §3.2.6 states "distance codes 30-31 will never actually occur
/// in the compressed data" — hence 30, not 32.
pub const MAX_DISTANCE_SYMBOL_COUNT: u8 = 30;

/// Mirror of `rapidgzip::deflate::MAX_LITERAL_HUFFMAN_CODE_COUNT`
/// (definitions.hpp:56). Next power of two of
/// `MAX_LITERAL_OR_LENGTH_SYMBOLS`, assuming all symbols are equally
/// likely (9-bit codes).
pub const MAX_LITERAL_HUFFMAN_CODE_COUNT: usize = 512;

/// Mirror of `rapidgzip::deflate::MAX_RUN_LENGTH` (definitions.hpp:57).
pub const MAX_RUN_LENGTH: usize = 258;

/// Mirror of `rapidgzip::deflate::END_OF_BLOCK_SYMBOL`
/// (definitions.hpp:59). Symbol 256 in the literal/length alphabet
/// signals end-of-block.
pub const END_OF_BLOCK_SYMBOL: u16 = 256;

/// Mirror of `rapidgzip::deflate::CompressionType` (definitions.hpp:61-67).
/// 2-bit type field in deflate block headers (BTYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CompressionType {
    Uncompressed = 0b00,
    FixedHuffman = 0b01,
    DynamicHuffman = 0b10,
    Reserved = 0b11,
}

impl CompressionType {
    /// Mirror of `rapidgzip::deflate::toString(CompressionType)`
    /// (definitions.hpp:70-85).
    pub const fn as_str(self) -> &'static str {
        match self {
            CompressionType::Uncompressed => "Uncompressed",
            CompressionType::FixedHuffman => "Fixed Huffman",
            CompressionType::DynamicHuffman => "Dynamic Huffman",
            CompressionType::Reserved => "Reserved",
        }
    }

    /// Parse the 2-bit BTYPE field. Returns `None` for the (unreachable
    /// at the bit level) case of bits outside `0..=3`.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0b11 {
            0b00 => Some(CompressionType::Uncompressed),
            0b01 => Some(CompressionType::FixedHuffman),
            0b10 => Some(CompressionType::DynamicHuffman),
            0b11 => Some(CompressionType::Reserved),
            _ => None,
        }
    }
}

// =====================================================================
// StoppingPoint (definitions.hpp:92-118).
// =====================================================================

/// Mirror of `rapidgzip::StoppingPoint` (definitions.hpp:92-100).
///
/// Bit-flag enum: callers OR variants together to request multiple
/// preemptive stopping points from the decoder.  Modeled here as
/// associated constants on a wrapper struct so consumers can write
/// `StoppingPoint::END_OF_BLOCK | StoppingPoint::END_OF_STREAM`
/// exactly like the C++ side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StoppingPoint(pub u32);

impl StoppingPoint {
    pub const NONE: StoppingPoint = StoppingPoint(0);
    pub const END_OF_STREAM_HEADER: StoppingPoint = StoppingPoint(1 << 0);
    /// `END_OF_STREAM = 1 << 1` — after the gzip footer has been read.
    pub const END_OF_STREAM: StoppingPoint = StoppingPoint(1 << 1);
    pub const END_OF_BLOCK_HEADER: StoppingPoint = StoppingPoint(1 << 2);
    pub const END_OF_BLOCK: StoppingPoint = StoppingPoint(1 << 3);
    pub const ALL: StoppingPoint = StoppingPoint(0xFFFF_FFFF);

    /// Mirror of `rapidgzip::toString(StoppingPoint)` (definitions.hpp:103-118).
    /// Returns the singular form only when `self` exactly matches a
    /// named variant; combinations return "Unknown" (vendor behavior).
    pub const fn as_str(self) -> &'static str {
        match self.0 {
            0 => "None",
            x if x == Self::END_OF_STREAM_HEADER.0 => "End of Stream Header",
            x if x == Self::END_OF_STREAM.0 => "End of Stream",
            x if x == Self::END_OF_BLOCK_HEADER.0 => "End of Block Header",
            x if x == Self::END_OF_BLOCK.0 => "End of Block",
            x if x == Self::ALL.0 => "All",
            _ => "Unknown",
        }
    }

    pub const fn contains(self, other: StoppingPoint) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for StoppingPoint {
    type Output = StoppingPoint;
    fn bitor(self, rhs: StoppingPoint) -> StoppingPoint {
        StoppingPoint(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for StoppingPoint {
    type Output = StoppingPoint;
    fn bitand(self, rhs: StoppingPoint) -> StoppingPoint {
        StoppingPoint(self.0 & rhs.0)
    }
}

// =====================================================================
// BlockBoundary (definitions.hpp:121-131).
// =====================================================================

/// Mirror of `rapidgzip::BlockBoundary` (definitions.hpp:121-131).
///
/// Pair of `(encodedOffset, decodedOffset)` describing a deflate block
/// boundary. The existing `chunk_data.rs::ChunkData::append_block_boundary`
/// uses a Rust-native `(u64, u64)` pair today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockBoundary {
    pub encoded_offset: usize,
    pub decoded_offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_constants_match() {
        assert_eq!(BYTE_SIZE, 8);
        assert_eq!(MAX_WINDOW_SIZE, 32 * 1024);
        assert_eq!(MAX_UNCOMPRESSED_SIZE, 65535);
        assert_eq!(MAX_CODE_LENGTH, 15);
        assert_eq!(PRECODE_COUNT_BITS, 4);
        assert_eq!(MAX_PRECODE_COUNT, 19);
        assert_eq!(PRECODE_BITS, 3);
        assert_eq!(MAX_PRECODE_LENGTH, 7);
        assert_eq!(MAX_LITERAL_OR_LENGTH_SYMBOLS, 286);
        assert_eq!(MAX_DISTANCE_SYMBOL_COUNT, 30);
        assert_eq!(MAX_LITERAL_HUFFMAN_CODE_COUNT, 512);
        assert_eq!(MAX_RUN_LENGTH, 258);
        assert_eq!(END_OF_BLOCK_SYMBOL, 256);
    }

    #[test]
    fn precode_alphabet_matches_rfc1951() {
        // RFC 1951 §3.2.7: the 19-symbol permutation order.
        let expected: [u8; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        assert_eq!(PRECODE_ALPHABET.0, expected);
        assert_eq!(PRECODE_ALPHABET.0.len(), MAX_PRECODE_COUNT as usize);
    }

    #[test]
    fn compression_type_bits_round_trip() {
        for bits in 0u8..=3 {
            let ct = CompressionType::from_bits(bits).unwrap();
            assert_eq!(ct as u8, bits);
        }
        assert_eq!(CompressionType::Uncompressed.as_str(), "Uncompressed");
        assert_eq!(CompressionType::FixedHuffman.as_str(), "Fixed Huffman");
        assert_eq!(CompressionType::DynamicHuffman.as_str(), "Dynamic Huffman");
        assert_eq!(CompressionType::Reserved.as_str(), "Reserved");
    }

    #[test]
    fn stopping_point_flags_or_and_label() {
        assert_eq!(StoppingPoint::NONE.0, 0);
        assert_eq!(StoppingPoint::END_OF_STREAM_HEADER.0, 1);
        assert_eq!(StoppingPoint::END_OF_STREAM.0, 2);
        assert_eq!(StoppingPoint::END_OF_BLOCK_HEADER.0, 4);
        assert_eq!(StoppingPoint::END_OF_BLOCK.0, 8);
        assert_eq!(StoppingPoint::ALL.0, 0xFFFF_FFFF);

        let combo = StoppingPoint::END_OF_BLOCK | StoppingPoint::END_OF_STREAM;
        assert!(combo.contains(StoppingPoint::END_OF_BLOCK));
        assert!(combo.contains(StoppingPoint::END_OF_STREAM));
        assert!(!combo.contains(StoppingPoint::END_OF_BLOCK_HEADER));

        // Vendor: singleton labels for exact matches, "Unknown" for combos.
        assert_eq!(StoppingPoint::NONE.as_str(), "None");
        assert_eq!(StoppingPoint::END_OF_STREAM.as_str(), "End of Stream");
        assert_eq!(combo.as_str(), "Unknown");
        assert_eq!(StoppingPoint::ALL.as_str(), "All");
    }

    #[test]
    fn block_boundary_default_and_equality() {
        let a = BlockBoundary::default();
        assert_eq!(a.encoded_offset, 0);
        assert_eq!(a.decoded_offset, 0);
        let b = BlockBoundary {
            encoded_offset: 100,
            decoded_offset: 200,
        };
        assert_ne!(a, b);
        let c = BlockBoundary {
            encoded_offset: 100,
            decoded_offset: 200,
        };
        assert_eq!(b, c);
    }
}
