//! Literal port of selected diagnostic primitives from
//! `rapidgzip::deflate::GzipAnalyzer`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/GzipAnalyzer.hpp:34-829).
//!
//! The vendor `GzipAnalyzer` is a monolithic `analyze()` function that
//! reads a stream, builds a [`deflate::Block<Statistics=true>`] over
//! every block, and prints per-block statistics. That code path pulls in
//! roughly half of rapidgzip's runtime types (BitReader,
//! Block<Statistics>, Statistics, Histogram, CRC32Calculator, format
//! detection, etc.) — a faithful port is a large multi-commit project.
//!
//! This commit lands the most reusable structural primitive from the
//! header: [`analyze_extra_string`], the FEXTRA-subfield identifier
//! used at the top of the analyzer's per-stream print loop. Given a
//! gzip `FEXTRA` payload, it reports which writer produced the stream
//! (BGZF, indexed-gzip / pgzip / mgzip, MiGz, QATzip, PGZF, dictzip) and
//! the writer-specific block-size / index hints they encode.
//!
//! The recognized subfields, IDs, and layouts mirror the dispatch ladder
//! at GzipAnalyzer.hpp:36-209.
//!
//! The full analyzer (`analyze()` at GzipAnalyzer.hpp:212+) is out of
//! scope for this commit; it will land as separate ports of the
//! statistics-tracking variants of `Block`, `Statistics<T>`, and
//! `Histogram<T>` first.

#![allow(dead_code)]

/// Mirror of the writer-identification result produced by
/// `analyzeExtraString` (GzipAnalyzer.hpp:36-209).
///
/// Each variant carries the writer-specific metadata that the vendor
/// prints to `std::cout`. Callers replace the `std::cout` side effect
/// with whatever rendering they need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtraStringHint {
    /// `BC` subfield ID — BGZF block.
    /// `compressed_block_size = ((extra[4] << 8) + extra[5]) + 1`
    /// (GzipAnalyzer.hpp:44-53).
    Bgzf { compressed_block_size: u32 },
    /// `IG` subfield ID — indexed gzip (pgzip / mgzip variants).
    /// `compressed_block_size` is little-endian u32 at bytes 4..8
    /// (GzipAnalyzer.hpp:55-66).
    IndexedGzip { compressed_block_size: u32 },
    /// `MZ` subfield ID — MiGz. Stores the *deflate stream* size
    /// (gzip header/footer excluded) as a little-endian u32 at bytes 4..8
    /// (GzipAnalyzer.hpp:83-95).
    MiGz { compressed_deflate_stream_size: u32 },
    /// `QZ` subfield ID — QATzip.
    /// `decompressed_stream_size` is at bytes 4..8 (LE u32), then
    /// `compressed_deflate_stream_size` at bytes 8..12 (LE u32)
    /// (GzipAnalyzer.hpp:97-115).
    QatZip {
        decompressed_stream_size: u32,
        compressed_deflate_stream_size: u32,
    },
    /// `ZC` subfield ID — PGZF. Always carries a deflate-stream size at
    /// bytes 4..8; may carry follow-on `GC` (group compressed size) or
    /// `IX` (index) sub-fields (GzipAnalyzer.hpp:132-170).
    Pgzf {
        compressed_deflate_stream_size: u32,
        compressed_group_size: Option<u64>,
        has_index: bool,
    },
    /// `RA` subfield ID — dictzip random access (GzipAnalyzer.hpp:203-208).
    Dictzip,
    /// FEXTRA was empty or unrecognized.
    Unknown,
}

/// Faithful port of `analyzeExtraString`
/// (GzipAnalyzer.hpp:36-209), with side effects replaced by a returned
/// [`ExtraStringHint`]. Each branch mirrors the corresponding `if`
/// ladder in the C++ source, and unknown / malformed inputs return
/// [`ExtraStringHint::Unknown`].
pub fn analyze_extra_string(extra: &[u8]) -> ExtraStringHint {
    if extra.is_empty() {
        return ExtraStringHint::Unknown;
    }

    // BGZF: 6 bytes, "BC" + 0x02 0x00 + u16 BE-encoded? — vendor reads
    // (extra[4] << 8) + extra[5] + 1 (GzipAnalyzer.hpp:50-51). NB this
    // is *big-endian* for this field only, unlike most LE encodings.
    if extra.len() == 6
        && extra[0] == b'B'
        && extra[1] == b'C'
        && extra[2] == 0x02
        && extra[3] == 0x00
    {
        let block_size = ((extra[4] as u32) << 8) + extra[5] as u32 + 1;
        return ExtraStringHint::Bgzf {
            compressed_block_size: block_size,
        };
    }

    // Indexed Gzip (pgzip / mgzip): 8 bytes, "IG" + 0x04 0x00 + LE u32.
    if extra.len() == 8
        && extra[0] == b'I'
        && extra[1] == b'G'
        && extra[2] == 0x04
        && extra[3] == 0x00
    {
        let block_size = u32_le_at(extra, 4);
        return ExtraStringHint::IndexedGzip {
            compressed_block_size: block_size,
        };
    }

    // MiGz: 8 bytes, "MZ" + 0x04 0x00 + LE u32 deflate stream size.
    if extra.len() == 8
        && extra[0] == b'M'
        && extra[1] == b'Z'
        && extra[2] == 0x04
        && extra[3] == 0x00
    {
        let stream_size = u32_le_at(extra, 4);
        return ExtraStringHint::MiGz {
            compressed_deflate_stream_size: stream_size,
        };
    }

    // QATzip: 12 bytes, "QZ" + 0x08 0x00 + LE u32 chunk size + LE u32 block size.
    if extra.len() == 12
        && extra[0] == b'Q'
        && extra[1] == b'Z'
        && extra[2] == 0x08
        && extra[3] == 0x00
    {
        let chunk_size = u32_le_at(extra, 4);
        let block_size = u32_le_at(extra, 8);
        return ExtraStringHint::QatZip {
            decompressed_stream_size: chunk_size,
            compressed_deflate_stream_size: block_size,
        };
    }

    // PGZF: ≥ 8 bytes, "ZC" + 0x04 0x00 + LE u32 deflate stream size.
    if extra.len() >= 8
        && extra[0] == b'Z'
        && extra[1] == b'C'
        && extra[2] == 0x04
        && extra[3] == 0x00
    {
        let stream_size = u32_le_at(extra, 4);

        // Optional GC follow-on (compressed group size).
        let mut group_size: Option<u64> = None;
        if extra.len() == 20
            && extra[8] == b'G'
            && extra[9] == b'C'
            && extra[10] == 0x08
            && extra[11] == 0x00
        {
            group_size = Some(u64_le_at(extra, 12));
        }

        // Optional IX follow-on (index marker).
        let has_index = extra.len() >= 20
            && extra[8] == b'I'
            && extra[9] == b'X'
            && extra[10] == 0x08
            && extra[11] == 0x00;

        return ExtraStringHint::Pgzf {
            compressed_deflate_stream_size: stream_size,
            compressed_group_size: group_size,
            has_index,
        };
    }

    // Dictzip: ≥ 10 bytes, "RA" prefix (vendor doesn't validate
    // subfield length here — GzipAnalyzer.hpp:203-207).
    if extra.len() >= 10 && extra[0] == b'R' && extra[1] == b'A' {
        return ExtraStringHint::Dictzip;
    }

    ExtraStringHint::Unknown
}

#[inline]
fn u32_le_at(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

#[inline]
fn u64_le_at(buf: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_extra_is_unknown() {
        assert_eq!(analyze_extra_string(&[]), ExtraStringHint::Unknown);
    }

    #[test]
    fn detects_bgzf() {
        // BGZF: "BC" + 0x02 0x00 + 2-byte BE block size; the actual
        // compressed_block_size returned is (size + 1).
        let extra = [b'B', b'C', 0x02, 0x00, 0x12, 0x34];
        let result = analyze_extra_string(&extra);
        assert_eq!(
            result,
            ExtraStringHint::Bgzf {
                // (0x12 << 8) + 0x34 + 1 = 0x1235
                compressed_block_size: 0x1235
            }
        );
    }

    #[test]
    fn detects_indexed_gzip() {
        let mut extra = vec![b'I', b'G', 0x04, 0x00];
        extra.extend_from_slice(&0x0010_0000u32.to_le_bytes());
        let result = analyze_extra_string(&extra);
        assert_eq!(
            result,
            ExtraStringHint::IndexedGzip {
                compressed_block_size: 0x0010_0000
            }
        );
    }

    #[test]
    fn detects_migz() {
        let mut extra = vec![b'M', b'Z', 0x04, 0x00];
        extra.extend_from_slice(&0x0080_0000u32.to_le_bytes());
        assert_eq!(
            analyze_extra_string(&extra),
            ExtraStringHint::MiGz {
                compressed_deflate_stream_size: 0x0080_0000
            }
        );
    }

    #[test]
    fn detects_qatzip() {
        let mut extra = vec![b'Q', b'Z', 0x08, 0x00];
        extra.extend_from_slice(&0xAAAAu32.to_le_bytes()); // chunk
        extra.extend_from_slice(&0xBBBBu32.to_le_bytes()); // block
        assert_eq!(
            analyze_extra_string(&extra),
            ExtraStringHint::QatZip {
                decompressed_stream_size: 0xAAAA,
                compressed_deflate_stream_size: 0xBBBB,
            }
        );
    }

    #[test]
    fn detects_pgzf_basic() {
        let mut extra = vec![b'Z', b'C', 0x04, 0x00];
        extra.extend_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            analyze_extra_string(&extra),
            ExtraStringHint::Pgzf {
                compressed_deflate_stream_size: 0x1000,
                compressed_group_size: None,
                has_index: false,
            }
        );
    }

    #[test]
    fn detects_pgzf_with_group_size() {
        let mut extra = vec![b'Z', b'C', 0x04, 0x00];
        extra.extend_from_slice(&0x1000u32.to_le_bytes());
        extra.extend(b"GC\x08\x00");
        extra.extend_from_slice(&0xDEAD_BEEF_CAFEu64.to_le_bytes());
        assert_eq!(
            analyze_extra_string(&extra),
            ExtraStringHint::Pgzf {
                compressed_deflate_stream_size: 0x1000,
                compressed_group_size: Some(0xDEAD_BEEF_CAFE),
                has_index: false,
            }
        );
    }

    #[test]
    fn detects_dictzip() {
        let extra = [b'R', b'A', 0x06, 0x00, 0, 0, 0, 0, 0, 0];
        assert_eq!(analyze_extra_string(&extra), ExtraStringHint::Dictzip);
    }

    #[test]
    fn unknown_extra() {
        let extra = [b'X', b'Y', 0x01, 0x00, 0xFF];
        assert_eq!(analyze_extra_string(&extra), ExtraStringHint::Unknown);
    }
}
