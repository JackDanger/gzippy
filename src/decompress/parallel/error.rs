//! Literal port of `rapidgzip::Error`
//! (vendor/rapidgzip/librapidarchive/src/core/Error.hpp).
//!
//! The C++ enum is `enum class [[nodiscard]] Error : (unspecified)` with
//! explicit hex discriminants used as a flat namespace of error codes
//! returned across the rapidgzip codebase (header parsers, Huffman
//! decoders, deflate block reader, etc.). We mirror the discriminants
//! exactly so any future port that compares against these numeric
//! values continues to work bit-for-bit.
//!
//! Rust callers should prefer pattern-matching the variants, but the
//! `as u8` representation matches the vendor's `static_cast<int>(Error)`
//! used in some debug prints / logging.

#![allow(dead_code)]

/// Mirror of `rapidgzip::Error` (Error.hpp:9-41).
///
/// Variant order and discriminants are preserved verbatim from the
/// vendor enum. New variants must NEVER be inserted before existing
/// ones — append with a new discriminant only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Error {
    /// `Error::NONE = 0x00`.
    None = 0x00,
    /// `Error::END_OF_FILE = 0x01`. Not an error per se — there simply
    /// is no data at all for, e.g., reading the next gzip header,
    /// indicating a valid end-of-file.
    EndOfFile = 0x01,

    /// `Error::EOF_ZERO_STRING = 0x10`.
    EofZeroString = 0x10,
    /// `Error::EOF_UNCOMPRESSED = 0x11`.
    EofUncompressed = 0x11,

    /// `Error::EXCEEDED_CL_LIMIT = 0x20`.
    ExceededClLimit = 0x20,
    /// `Error::EXCEEDED_SYMBOL_RANGE = 0x21`.
    ExceededSymbolRange = 0x21,
    /// `Error::EXCEEDED_LITERAL_RANGE = 0x22`.
    ExceededLiteralRange = 0x22,
    /// `Error::EXCEEDED_DISTANCE_RANGE = 0x23`.
    ExceededDistanceRange = 0x23,
    /// `Error::EXCEEDED_WINDOW_RANGE = 0x24`.
    ExceededWindowRange = 0x24,

    /// `Error::EMPTY_INPUT = 0x30`.
    EmptyInput = 0x30,

    /// `Error::INVALID_HUFFMAN_CODE = 0x40`.
    InvalidHuffmanCode = 0x40,
    /// `Error::NON_ZERO_PADDING = 0x41`.
    NonZeroPadding = 0x41,
    /// `Error::LENGTH_CHECKSUM_MISMATCH = 0x42`.
    LengthChecksumMismatch = 0x42,
    /// `Error::INVALID_COMPRESSION = 0x43`.
    InvalidCompression = 0x43,
    /// `Error::INVALID_CL_BACKREFERENCE = 0x44`.
    InvalidClBackreference = 0x44,
    /// `Error::INVALID_BACKREFERENCE = 0x45`.
    InvalidBackreference = 0x45,
    /// `Error::EMPTY_ALPHABET = 0x46`.
    EmptyAlphabet = 0x46,
    /// `Error::INVALID_CODE_LENGTHS = 0x47`.
    InvalidCodeLengths = 0x47,
    /// `Error::BLOATING_HUFFMAN_CODING = 0x48`.
    BloatingHuffmanCoding = 0x48,

    /// `Error::INVALID_GZIP_HEADER = 0x60`.
    InvalidGzipHeader = 0x60,
    /// `Error::INCOMPLETE_GZIP_HEADER = 0x61`.
    IncompleteGzipHeader = 0x61,

    /// `Error::UNEXPECTED_LAST_BLOCK = 0x80`.
    UnexpectedLastBlock = 0x80,
}

impl Error {
    /// Literal port of `rapidgzip::toString( Error )` (Error.hpp:44-95).
    /// Strings match the vendor exactly including punctuation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Error::EofZeroString => {
                "End of file encountered when trying to read zero-terminated string!"
            }
            Error::EofUncompressed => {
                "End of file encountered when trying to copy uncompressed block from file!"
            }
            Error::EmptyAlphabet => "All code lengths are zero!",
            Error::ExceededClLimit => {
                "The number of code lengths may not exceed the maximum possible value!"
            }
            Error::EmptyInput => "Container must not be empty!",
            Error::ExceededSymbolRange => {
                "The range of the symbol type cannot represent the implied alphabet!"
            }
            Error::InvalidHuffmanCode => "Failed to decode Huffman bits!",
            Error::NonZeroPadding => "Assumed padding seems to contain some kind of data!",
            Error::LengthChecksumMismatch => {
                "Integrity check for length of uncompressed deflate block failed!"
            }
            Error::InvalidCompression => "Invalid block compression type!",
            Error::ExceededLiteralRange => "Invalid number of literal/length codes!",
            Error::ExceededDistanceRange => "Invalid number of distance codes!",
            Error::InvalidClBackreference => {
                "Cannot copy last length because this is the first one!"
            }
            Error::InvalidBackreference => "Backreferenced data does not exist!",
            Error::InvalidGzipHeader => "Invalid gzip magic bytes!",
            Error::IncompleteGzipHeader => "Incomplete gzip header!",
            Error::InvalidCodeLengths => {
                "Constructing a Huffman coding from the given code length sequence failed!"
            }
            Error::BloatingHuffmanCoding => "The Huffman coding is not optimal!",
            Error::UnexpectedLastBlock => {
                "The block is the last of the stream even though it should not be!"
            }
            Error::ExceededWindowRange => {
                "The backreferenced distance lies outside the window buffer!"
            }
            Error::None => "No error.",
            Error::EndOfFile => "End of file reached.",
        }
    }

    /// The numeric discriminant matching `static_cast<uint8_t>(Error)`
    /// in the vendor source. Useful for log parity with rapidgzip.
    pub const fn code(self) -> u8 {
        self as u8
    }
}

impl core::fmt::Display for Error {
    /// Mirror of `operator<<( std::ostream&, Error )` (Error.hpp:98-104).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the vendor discriminants. If any of these change, downstream
    /// numeric comparisons will silently break — the test catches the
    /// regression before it ships.
    #[test]
    fn discriminants_match_vendor_hex_values() {
        assert_eq!(Error::None.code(), 0x00);
        assert_eq!(Error::EndOfFile.code(), 0x01);
        assert_eq!(Error::EofZeroString.code(), 0x10);
        assert_eq!(Error::EofUncompressed.code(), 0x11);
        assert_eq!(Error::ExceededClLimit.code(), 0x20);
        assert_eq!(Error::ExceededSymbolRange.code(), 0x21);
        assert_eq!(Error::ExceededLiteralRange.code(), 0x22);
        assert_eq!(Error::ExceededDistanceRange.code(), 0x23);
        assert_eq!(Error::ExceededWindowRange.code(), 0x24);
        assert_eq!(Error::EmptyInput.code(), 0x30);
        assert_eq!(Error::InvalidHuffmanCode.code(), 0x40);
        assert_eq!(Error::NonZeroPadding.code(), 0x41);
        assert_eq!(Error::LengthChecksumMismatch.code(), 0x42);
        assert_eq!(Error::InvalidCompression.code(), 0x43);
        assert_eq!(Error::InvalidClBackreference.code(), 0x44);
        assert_eq!(Error::InvalidBackreference.code(), 0x45);
        assert_eq!(Error::EmptyAlphabet.code(), 0x46);
        assert_eq!(Error::InvalidCodeLengths.code(), 0x47);
        assert_eq!(Error::BloatingHuffmanCoding.code(), 0x48);
        assert_eq!(Error::InvalidGzipHeader.code(), 0x60);
        assert_eq!(Error::IncompleteGzipHeader.code(), 0x61);
        assert_eq!(Error::UnexpectedLastBlock.code(), 0x80);
    }

    /// Spot-check a few message strings to lock them down. The full set
    /// mirrors the vendor's `toString` switch — preserving punctuation
    /// is intentional (some tests in the C++ codebase grep for it).
    #[test]
    fn as_str_matches_vendor_messages() {
        assert_eq!(Error::None.as_str(), "No error.");
        assert_eq!(Error::EndOfFile.as_str(), "End of file reached.");
        assert_eq!(
            Error::InvalidHuffmanCode.as_str(),
            "Failed to decode Huffman bits!"
        );
        assert_eq!(Error::EmptyAlphabet.as_str(), "All code lengths are zero!");
        assert_eq!(
            Error::ExceededWindowRange.as_str(),
            "The backreferenced distance lies outside the window buffer!"
        );
        assert_eq!(
            Error::UnexpectedLastBlock.as_str(),
            "The block is the last of the stream even though it should not be!"
        );
    }

    #[test]
    fn display_uses_to_string() {
        assert_eq!(format!("{}", Error::None), "No error.");
        assert_eq!(
            format!("{}", Error::NonZeroPadding),
            "Assumed padding seems to contain some kind of data!"
        );
    }

    #[test]
    fn is_std_error() {
        let e: &dyn std::error::Error = &Error::InvalidGzipHeader;
        assert_eq!(e.to_string(), "Invalid gzip magic bytes!");
    }
}
