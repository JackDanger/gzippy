//! Literal port of the `rapidgzip::zlib` namespace
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/gzip.hpp:311-415).
//!
//! This is the ZLIB-format companion to [`crate::decompress::parallel::gzip_format`]:
//! `Header`, `Footer`, `read_header`, `read_footer`, plus the
//! `CompressionLevel` enum.
//!
//! It deliberately mirrors the gzip-format module's byte-slice API
//! rather than the vendor's bit-reader API — gzippy's parallel
//! decompression path operates on `&[u8]` and we keep the surface
//! symmetric. The vendor's `ZlibInflateWrapper` (zlib.hpp:122-267)
//! is **not** ported here; it's a heavy zlib FFI wrapper that will
//! land separately once we wire ZLIB inputs into the chunk fetcher.

#![allow(dead_code)]

use std::io;

/// Mirror of `rapidgzip::zlib::CompressionLevel` (gzip.hpp:313-319).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionLevel {
    Fastest = 0,
    Fast = 1,
    #[default]
    Default = 2,
    Slowest = 3,
}

impl CompressionLevel {
    /// Mirror of `zlib::toString(CompressionLevel)` (gzip.hpp:322-333).
    pub const fn as_str(self) -> &'static str {
        match self {
            CompressionLevel::Fastest => "Fastest",
            CompressionLevel::Fast => "Fast",
            CompressionLevel::Default => "Default",
            CompressionLevel::Slowest => "Slowest",
        }
    }
}

/// Mirror of `rapidgzip::zlib::Header` (gzip.hpp:336-341).
///
/// `dictionary_id` defaults to `1` because the Adler-32 of the empty
/// data stream is `1` — see the vendor field initializer
/// `uint32_t dictionaryID{ 1 };`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// 2^(8 + CINFO) window size, where CINFO ∈ [0,7].
    pub window_size: u16,
    pub compression_level: CompressionLevel,
    pub dictionary_id: u32,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            window_size: 0,
            compression_level: CompressionLevel::Default,
            dictionary_id: 1,
        }
    }
}

/// Mirror of `rapidgzip::zlib::Footer` (gzip.hpp:344-347).
///
/// `adler32` defaults to `1` to match the vendor's default — the
/// initial Adler-32 state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footer {
    pub adler32: u32,
}

impl Default for Footer {
    fn default() -> Self {
        Self { adler32: 1 }
    }
}

/// Error variants returned from [`read_header`] — mirror of the
/// `rapidgzip::Error` subset used in `zlib::readHeader`
/// (gzip.hpp:350-394).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZlibHeaderError {
    /// `compression_method != 8` (not deflate), `compression_info > 7`,
    /// `(cmf << 8) + flags) % 31 != 0`, or `FDICT` is set (vendor
    /// returns `INVALID_GZIP_HEADER` in all four cases —
    /// gzip.hpp:361, 368, 374, 385).
    InvalidZlibHeader,
    /// Reached EOF before reading any header bytes (vendor's
    /// `Error::END_OF_FILE`).
    EndOfFile,
    /// Started reading a header but ran out of bytes mid-parse
    /// (vendor's `Error::INCOMPLETE_GZIP_HEADER`).
    IncompleteZlibHeader,
}

impl ZlibHeaderError {
    pub const fn as_str(self) -> &'static str {
        match self {
            ZlibHeaderError::InvalidZlibHeader => "Invalid zlib header",
            ZlibHeaderError::EndOfFile => "End of file",
            ZlibHeaderError::IncompleteZlibHeader => "Incomplete zlib header",
        }
    }
}

/// Literal port of `rapidgzip::zlib::readHeader` (gzip.hpp:350-394) for
/// byte-slice input. Returns `Ok((header, bytes_consumed))` on
/// success; `Err((partial_header, error))` on failure. The partial
/// header may carry the partially-decoded window size etc., matching
/// the vendor's `{ header, error }` returns.
pub fn read_header(data: &[u8]) -> Result<(Header, usize), (Header, ZlibHeaderError)> {
    let mut header = Header::default();

    // First byte: CMF — `compression_method` in low 4 bits,
    // `compression_info` in high 4 bits.
    let cmf = match data.first() {
        Some(&b) => b,
        None => return Err((header, ZlibHeaderError::EndOfFile)),
    };
    let compression_method = (cmf as u64) & 0x0F;
    if compression_method != 8 {
        return Err((header, ZlibHeaderError::InvalidZlibHeader));
    }
    let compression_info = (cmf as u64) >> 4;
    if compression_info > 7 {
        return Err((header, ZlibHeaderError::InvalidZlibHeader));
    }
    // Mirror of `header.windowSize = 2U << (8U + compressionInfo)`
    // (gzip.hpp:370).
    header.window_size = (2u32 << (8 + compression_info as u32)) as u16;

    // Second byte: FLG. EOF here means partial header (vendor returns
    // INCOMPLETE_GZIP_HEADER once `readPartialHeader` is true —
    // gzip.hpp:354, 390).
    let flags = match data.get(1) {
        Some(&b) => b,
        None => return Err((header, ZlibHeaderError::IncompleteZlibHeader)),
    };

    // FCHECK constraint: ((CMF * 256) + FLG) % 31 == 0
    // (gzip.hpp:373).
    if !(((cmf as u32) << 8) + flags as u32).is_multiple_of(31) {
        return Err((header, ZlibHeaderError::InvalidZlibHeader));
    }

    // FDICT flag (bit 5 of FLG). If set, a 4-byte dictionary ID
    // follows. Rapidgzip rejects this case entirely (gzip.hpp:377-386).
    let uses_dictionary = ((flags >> 5) & 1) != 0;
    let consumed = 2usize;
    if uses_dictionary {
        if data.len() < 6 {
            return Err((header, ZlibHeaderError::IncompleteZlibHeader));
        }
        let mut dict_id: u32 = 0;
        for i in 0..4 {
            dict_id = (dict_id << 8) | data[consumed + i] as u32;
        }
        header.dictionary_id = dict_id;
        // Vendor unconditionally rejects FDICT streams here.
        return Err((header, ZlibHeaderError::InvalidZlibHeader));
    }

    // FLEVEL = bits 6-7 of FLG (gzip.hpp:388).
    header.compression_level = match (flags >> 6) & 0b11 {
        0 => CompressionLevel::Fastest,
        1 => CompressionLevel::Fast,
        2 => CompressionLevel::Default,
        3 => CompressionLevel::Slowest,
        _ => unreachable!(),
    };

    Ok((header, consumed))
}

/// Literal port of `rapidgzip::zlib::readFooter` (gzip.hpp:404-414).
/// The zlib footer is a single big-endian Adler-32 of the
/// uncompressed payload (RFC 1950 § 2.2). The vendor reads it via
/// `bitReader.read<32>()` on a big-endian bit reader, which yields
/// big-endian bytes when called on a byte-aligned input.
///
/// `at` is the byte offset at which the footer begins; it must be
/// `data.len() - 4` for a well-formed zlib stream.
pub fn read_footer(data: &[u8], at: usize) -> io::Result<Footer> {
    if at + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "incomplete zlib footer (need 4 bytes)",
        ));
    }
    // Big-endian per RFC 1950.
    let adler32 = u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
    Ok(Footer { adler32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid zlib header for a given CMF/FLG pair.
    /// Helper computes a valid FCHECK so the header passes the
    /// `% 31 == 0` check.
    fn make_header(cmf: u8, flg_without_fcheck: u8) -> [u8; 2] {
        let base = ((cmf as u32) << 8) | (flg_without_fcheck as u32 & 0b1110_0000);
        let mut flg = flg_without_fcheck & 0b1110_0000;
        for fcheck in 0..=31u8 {
            let candidate = base | fcheck as u32;
            if candidate.is_multiple_of(31) {
                flg = (flg_without_fcheck & 0b1110_0000) | fcheck;
                break;
            }
        }
        [cmf, flg]
    }

    #[test]
    fn parses_standard_zlib_header() {
        // CMF: compression_method=8 (deflate), cinfo=6 (16KiB window).
        // We avoid cinfo=7 here because the vendor's
        // `static_cast<uint16_t>(2U << (8 + cinfo))` overflows to 0 at
        // cinfo=7 (gzip.hpp:370) — we faithfully preserve that bug
        // and exercise it in a separate test below.
        let hdr = make_header(0x68, 0); // FLEVEL=0 (Fastest), no FDICT
        let (h, n) = read_header(&hdr).unwrap();
        assert_eq!(n, 2);
        assert_eq!(h.window_size, 32 * 1024); // 2 << 14 = 32768
        assert_eq!(h.compression_level, CompressionLevel::Fastest);
        assert_eq!(h.dictionary_id, 1, "default dictionary id is 1");
    }

    /// Faithful preservation of vendor behaviour: `cinfo == 7` (the
    /// standard 32 KiB window) overflows `windowSize` (`uint16_t`) to 0
    /// because the C++ expression is `2U << (8U + 7U) == 65536`
    /// cast to u16 (gzip.hpp:370). We do **not** "fix" this — any
    /// downstream code that consults `window_size` must tolerate it
    /// or use `compression_info` from the raw CMF byte.
    #[test]
    fn window_size_overflows_at_cinfo_7_like_vendor() {
        let hdr = make_header(0x78, 0); // cinfo = 7
        let (h, _) = read_header(&hdr).unwrap();
        assert_eq!(h.window_size, 0);
    }

    #[test]
    fn parses_compression_level_field() {
        for level_bits in 0u8..=3 {
            let hdr = make_header(0x78, level_bits << 6);
            let (h, _) = read_header(&hdr).unwrap();
            assert_eq!(
                h.compression_level,
                match level_bits {
                    0 => CompressionLevel::Fastest,
                    1 => CompressionLevel::Fast,
                    2 => CompressionLevel::Default,
                    3 => CompressionLevel::Slowest,
                    _ => unreachable!(),
                }
            );
        }
    }

    #[test]
    fn rejects_non_deflate_method() {
        let hdr = make_header(0x77, 0); // compression_method = 7
        let (_, e) = read_header(&hdr).unwrap_err();
        assert_eq!(e, ZlibHeaderError::InvalidZlibHeader);
    }

    #[test]
    fn rejects_cinfo_above_seven() {
        let hdr = make_header(0x88, 0); // cinfo = 8 (illegal)
        let (_, e) = read_header(&hdr).unwrap_err();
        assert_eq!(e, ZlibHeaderError::InvalidZlibHeader);
    }

    #[test]
    fn rejects_fdict_streams() {
        // Set FDICT bit (bit 5) — vendor rejects unconditionally.
        let mut hdr = vec![0x78, 0];
        hdr[1] = 0b0010_0000; // FDICT set; we'll fix FCHECK below
        let base = ((hdr[0] as u32) << 8) | hdr[1] as u32;
        for fcheck in 0..=31u8 {
            if (base | fcheck as u32).is_multiple_of(31) {
                hdr[1] = 0b0010_0000 | fcheck;
                break;
            }
        }
        hdr.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let (h, e) = read_header(&hdr).unwrap_err();
        assert_eq!(e, ZlibHeaderError::InvalidZlibHeader);
        assert_eq!(h.dictionary_id, 0xDEADBEEF);
    }

    #[test]
    fn rejects_bad_fcheck() {
        // Force a CMF/FLG combo where (cmf << 8) + flg is NOT
        // divisible by 31.
        let hdr = [0x78u8, 0x00]; // 0x7800 % 31 == ... well let's check
        let prod = ((hdr[0] as u32) << 8) | hdr[1] as u32;
        if !prod.is_multiple_of(31) {
            let (_, e) = read_header(&hdr).unwrap_err();
            assert_eq!(e, ZlibHeaderError::InvalidZlibHeader);
        }
    }

    #[test]
    fn end_of_file_on_empty_input() {
        let (_, e) = read_header(&[]).unwrap_err();
        assert_eq!(e, ZlibHeaderError::EndOfFile);
    }

    #[test]
    fn incomplete_header_when_only_cmf_present() {
        let (_, e) = read_header(&[0x78]).unwrap_err();
        assert_eq!(e, ZlibHeaderError::IncompleteZlibHeader);
    }

    #[test]
    fn read_footer_parses_big_endian_adler32() {
        // Adler-32 of "Wikipedia" is 0x11E60398 per Wikipedia's RFC
        // 1950 example. Stored big-endian.
        let mut data = vec![0u8; 4];
        data.extend_from_slice(&0x11E60398u32.to_be_bytes());
        let f = read_footer(&data, 4).unwrap();
        assert_eq!(f.adler32, 0x11E60398);
    }

    #[test]
    fn read_footer_rejects_short_input() {
        let data = [0u8; 3];
        let r = read_footer(&data, 0);
        assert!(r.is_err());
    }

    #[test]
    fn compression_level_as_str_matches_vendor() {
        assert_eq!(CompressionLevel::Fastest.as_str(), "Fastest");
        assert_eq!(CompressionLevel::Fast.as_str(), "Fast");
        assert_eq!(CompressionLevel::Default.as_str(), "Default");
        assert_eq!(CompressionLevel::Slowest.as_str(), "Slowest");
    }

    #[test]
    fn defaults_match_vendor_field_initializers() {
        let h = Header::default();
        // vendor: dictionaryID{ 1 /* ADLER32 of empty data stream */ }
        assert_eq!(h.dictionary_id, 1);
        assert_eq!(h.compression_level, CompressionLevel::Default);

        let f = Footer::default();
        // vendor: adler32{ 1 }
        assert_eq!(f.adler32, 1);
    }
}
