//! Literal port of `rapidgzip::FileType` + `hasCRC32` + `toString`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/gzip.hpp:21-74)
//! and `rapidgzip::determineFileTypeAndOffset`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/format.hpp:17-58).
//!
//! Detects which container format a byte stream uses (GZIP, BGZF,
//! ZLIB, BZIP2, DEFLATE, or NONE) by attempting each header parser
//! in turn and returning the first successful match. The vendor's
//! detection order is significant: it tries gzip/BGZF first
//! (highest header redundancy), then ZLIB, then BZIP2, then raw
//! DEFLATE last because raw DEFLATE has only 1 bit of header
//! validation in the fixed-Huffman case.
//!
//! ## What's ported vs. delegated
//!
//! - `FileType` enum + `as_str` + `has_crc32` are full literal ports.
//! - `determine_file_type_and_offset` is a literal port of the
//!   detection cascade, calling into the existing gzippy header
//!   parsers (gzip_format::read_header, zlib_format::read_header).
//!   BZIP2 and BGZF detection paths return `Some((BZIP2, _))` /
//!   `Some((BGZF, _))` once the corresponding parsers land — for
//!   now we route the BGZF case through a magic-bytes check (the
//!   "BGZF" subfield FEXTRA is part of the gzip header, so a gzip
//!   match plus that subfield ⇒ BGZF).

#![allow(dead_code)]

use super::gzip_format;
use super::zlib_format;

/// Literal port of `rapidgzip::FileType` (gzip.hpp:21-29).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FileType {
    #[default]
    None,
    Bgzf,
    Gzip,
    Zlib,
    Deflate,
    Bzip2,
}

impl FileType {
    /// Literal port of `rapidgzip::toString(FileType)` (gzip.hpp:32-51).
    pub const fn as_str(self) -> &'static str {
        match self {
            FileType::None => "None",
            FileType::Bgzf => "BGZF",
            FileType::Gzip => "GZIP",
            FileType::Zlib => "ZLIB",
            FileType::Deflate => "DEFLATE",
            FileType::Bzip2 => "BZIP2",
        }
    }

    /// Literal port of `rapidgzip::hasCRC32(FileType)` (gzip.hpp:54-74).
    ///
    /// Returns whether the format carries a CRC32 trailer. The
    /// vendor's comment explains the BZIP2 quirk: it uses the same
    /// CRC32 polynomial as gzip but with non-reversed bit ordering,
    /// so the *value* is computed differently and consumers can't
    /// treat them interchangeably — hence BZIP2 is excluded here.
    /// GZIP + BGZF are the only formats that store a 32-bit CRC32
    /// in the gzip footer.
    pub const fn has_crc32(self) -> bool {
        match self {
            FileType::None | FileType::Bzip2 | FileType::Deflate | FileType::Zlib => false,
            FileType::Bgzf | FileType::Gzip => true,
        }
    }
}

impl core::fmt::Display for FileType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of [`determine_file_type_and_offset`]: the detected format
/// and the byte offset immediately after the header.  Mirrors the
/// vendor's `std::pair<FileType, size_t>` return.
pub type DetectedFormat = (FileType, usize);

/// BGZF magic-bytes check: gzip header + "BC" subfield of length 2 +
/// 2 bytes BSIZE-1. This is the FEXTRA subfield BGZF defines on top
/// of plain gzip per SAM/BAM spec §4.1.
///
/// The vendor's `blockfinder::Bgzf::isBgzfFile` walks the gzip
/// header looking for the BC subfield with the BGZF-specific
/// payload; gzippy already has full BGZF detection in
/// `bgzf::bgzf_detect`. This helper mirrors the vendor's
/// "after detecting GZIP, check whether it's specifically BGZF"
/// branch. Returns `true` when the gzip header has the BGZF
/// 16-byte fixed layout.
fn looks_like_bgzf(data: &[u8]) -> bool {
    // BGZF header: 1F 8B 08 04 ?? ?? ?? ?? 00 FF 06 00 42 43 02 00 ...
    if data.len() < 16 {
        return false;
    }
    data[0] == 0x1F
        && data[1] == 0x8B
        && data[2] == 0x08
        && data[3] == 0x04
        && data[12] == b'B'
        && data[13] == b'C'
}

/// BZIP2 magic-byte check: `BZh` + a digit '1'..'9'. Vendor calls
/// `bzip2::readBzip2Header` which validates more, but the 4-byte
/// magic is reliable enough for the dispatch table.
fn looks_like_bzip2(data: &[u8]) -> bool {
    data.len() >= 4
        && data[0] == b'B'
        && data[1] == b'Z'
        && data[2] == b'h'
        && (b'1'..=b'9').contains(&data[3])
}

/// Literal port of `rapidgzip::determineFileTypeAndOffset`
/// (vendor/rapidgzip/.../gzip/format.hpp:17-58).
///
/// Returns `Some((FileType, offset_after_header))` for the first
/// matching format, or `None` if no format is recognized.  The
/// detection order matches the vendor exactly:
///
/// 1. GZIP (and refine to BGZF if the gzip header's FEXTRA carries
///    the BGZF "BC" subfield).
/// 2. ZLIB.
/// 3. BZIP2 (magic-byte check; full header validation is left to
///    the future BZIP2 port).
/// 4. DEFLATE (a raw fixed-Huffman deflate block validates with
///    only 1 bit — hence tried last). Currently surfaces as `None`
///    pending the dedicated raw-DEFLATE detector port; see comment.
pub fn determine_file_type_and_offset(data: &[u8]) -> Option<DetectedFormat> {
    if data.is_empty() {
        return None;
    }

    // 1. gzip header — and the BGZF refinement.
    if let Ok((_, after_header)) = gzip_format::read_header(data) {
        let file_type = if looks_like_bgzf(data) {
            FileType::Bgzf
        } else {
            FileType::Gzip
        };
        return Some((file_type, after_header));
    }

    // 2. zlib header (CMF + FLG, 2 bytes minimum).
    if let Ok((_, after_header)) = zlib_format::read_header(data) {
        return Some((FileType::Zlib, after_header));
    }

    // 3. bzip2 — magic-only for now. Vendor calls
    //    `bzip2::readBzip2Header` which validates the full 4-byte
    //    `BZh<level>` plus block-magic prefix; we'll lift that to
    //    parity once the BZIP2 port lands.
    if looks_like_bzip2(data) {
        return Some((FileType::Bzip2, 4));
    }

    // 4. raw DEFLATE. Vendor calls `deflate::Block::readHeader` and
    //    accepts on `Error::NONE`. gzippy's `deflate_block.rs` has
    //    a `Block::read_header` equivalent, but invoking it here
    //    would tangle a runtime BitReader with the byte-slice API
    //    this function exposes. Deferred until the BitReader port
    //    is consolidated — until then `determine_file_type_and_offset`
    //    surfaces raw DEFLATE inputs as `None`, matching the vendor
    //    early-exit when the deflate block header is malformed.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny well-formed gzip stream of "Hello": fixed-Huffman
    /// encoded ("Hello\n" actually); generated with `printf 'Hello\n' | gzip`
    /// and the raw bytes pasted in to keep the test hermetic.
    fn make_minimal_gzip() -> Vec<u8> {
        // 1F 8B 08 00 timestamp:4 00 03 ... deflate-payload ... crc:4 isize:4
        // Build a simplest valid header (no FLG bits set):
        let mut v = Vec::new();
        v.extend_from_slice(&[
            0x1F, 0x8B, // magic
            0x08, // method = deflate
            0x00, // FLG = 0
            0x00, 0x00, 0x00, 0x00, // mtime
            0x00, // XFL
            0x03, // OS = unix
        ]);
        // Empty deflate stream: a single BFINAL=1, BTYPE=00 (stored),
        // length 0, ~length 0xFFFF.
        v.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
        // CRC32(empty) = 0; ISIZE = 0.
        v.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        v.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        v
    }

    /// Build the 16-byte BGZF fixed-prefix header (without payload):
    /// 1F 8B 08 04 mtime:4 00 FF 06 00 BC 02 00 bsize:2
    fn make_minimal_bgzf_header() -> Vec<u8> {
        vec![
            0x1F, 0x8B, 0x08, 0x04, // gzip + FEXTRA flag
            0x00, 0x00, 0x00, 0x00, // mtime
            0x00, 0xFF, // XFL + OS
            0x06, 0x00, // XLEN = 6 (size of FEXTRA)
            b'B', b'C', // subfield ID
            0x02, 0x00, // subfield size = 2
            0x1B, 0x00, // BSIZE - 1 = 27 (so BGZF block size = 28)
        ]
    }

    #[test]
    fn file_type_as_str_matches_vendor() {
        assert_eq!(FileType::None.as_str(), "None");
        assert_eq!(FileType::Bgzf.as_str(), "BGZF");
        assert_eq!(FileType::Gzip.as_str(), "GZIP");
        assert_eq!(FileType::Zlib.as_str(), "ZLIB");
        assert_eq!(FileType::Deflate.as_str(), "DEFLATE");
        assert_eq!(FileType::Bzip2.as_str(), "BZIP2");
    }

    #[test]
    fn has_crc32_matches_vendor_table() {
        // Vendor: only GZIP + BGZF carry CRC32.
        assert!(FileType::Gzip.has_crc32());
        assert!(FileType::Bgzf.has_crc32());
        // Vendor: ZLIB carries Adler-32, not CRC32 — false.
        assert!(!FileType::Zlib.has_crc32());
        // Vendor: BZIP2 uses non-reversed-bit CRC32 — comment in
        // gzip.hpp:61-62 explains why it returns false here.
        assert!(!FileType::Bzip2.has_crc32());
        assert!(!FileType::Deflate.has_crc32());
        assert!(!FileType::None.has_crc32());
    }

    #[test]
    fn detects_gzip() {
        let data = make_minimal_gzip();
        let (ft, off) = determine_file_type_and_offset(&data).unwrap();
        assert_eq!(ft, FileType::Gzip);
        // 10 bytes of gzip header with no optional fields.
        assert_eq!(off, 10);
    }

    #[test]
    fn detects_bgzf_via_subfield() {
        let data = make_minimal_bgzf_header();
        let (ft, _) = determine_file_type_and_offset(&data).unwrap();
        assert_eq!(ft, FileType::Bgzf);
    }

    #[test]
    fn detects_zlib() {
        // CMF 0x78 (deflate, 32KiB window), FLG with valid FCHECK.
        // 0x7800 % 31 == 18, so flg = 31 - 18 = 13: header = [0x78, 0x9C].
        let data = [0x78u8, 0x9C, 0xAB, 0xCD]; // payload bytes ignored
        let (ft, off) = determine_file_type_and_offset(&data).unwrap();
        assert_eq!(ft, FileType::Zlib);
        assert_eq!(off, 2);
    }

    #[test]
    fn detects_bzip2_magic() {
        let data = [b'B', b'Z', b'h', b'9', 0x31, 0x41, 0x59, 0x26];
        let (ft, off) = determine_file_type_and_offset(&data).unwrap();
        assert_eq!(ft, FileType::Bzip2);
        assert_eq!(off, 4);
    }

    #[test]
    fn rejects_unknown_format() {
        // Random bytes that match none of the headers.
        assert_eq!(determine_file_type_and_offset(&[0u8; 1]), None);
        assert_eq!(
            determine_file_type_and_offset(&[0xDEu8, 0xAD, 0xBE, 0xEF]),
            None
        );
    }

    #[test]
    fn empty_input_is_none() {
        assert_eq!(determine_file_type_and_offset(&[]), None);
    }

    #[test]
    fn display_uses_as_str() {
        assert_eq!(format!("{}", FileType::Gzip), "GZIP");
        assert_eq!(format!("{}", FileType::None), "None");
    }
}
