//! Literal port of `rapidgzip::blockfinder::Bgzf`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/blockfinder/Bgzf.hpp:83-238).
//!
//! Finds BGZF deflate-block boundaries by walking the gzip header of
//! each BGZF member and adding the encoded block size to the running
//! cursor. Each member is exactly 18 bytes of header + `BSIZE` bytes of
//! payload (deflate + gzip footer); the `BSIZE - 1` field lives at
//! header bytes 16..18 (little-endian u16) — see SAMv1 spec linked
//! from the vendor header comment (Bgzf.hpp:64-81).
//!
//! NOTE — gzippy already has `src/decompress/bgzf.rs`, which is the
//! production BGZF/gzippy-parallel decoder and was authored from
//! scratch (not a port). This file lives alongside it as the faithful
//! port of the rapidgzip block-finder so the parallel-fetcher port
//! can wire it in via [`BlockFinderInterface`] without disturbing the
//! production decoder.

#![allow(dead_code)]

use std::io::{self, Read, Seek, SeekFrom};

use crate::decompress::parallel::blockfinder_interface::{BlockFinderInterface, NO_MORE_BLOCKS};

/// Mirror of `Bgzf::HeaderBytes` (Bgzf.hpp:87) — the fixed 18-byte
/// BGZF gzip header (10 byte gzip header + 8 byte FEXTRA).
pub type HeaderBytes = [u8; 18];

/// Mirror of `Bgzf::FooterBytes` (Bgzf.hpp:88) — the fixed 28-byte
/// BGZF end-of-file marker block.
pub type FooterBytes = [u8; 28];

/// Mirror of `Bgzf::BGZF_FOOTER` (Bgzf.hpp:90-103). The empty BGZF
/// block that conforming writers append as an EOF marker.
pub const BGZF_FOOTER: FooterBytes = [
    0x1F, 0x8B, 0x08, // gzip magic bytes
    0x04, // Flags with FEXTRA set
    0x00, 0x00, 0x00, 0x00, // Modification time (dummy)
    0x00, // Extra flags
    0xFF, // Unknown OS
    0x06, 0x00, // Length of extra field
    0x42, 0x43, 0x02, 0x00, 0x1B, 0x00, // Extra field with subfield ID "BC" = 0x42 0x43
    0x03, // Fixed Huffman compressed deflate block with final bit set + EOB
    0x00, // Part of EOB (257 == 0b000'0000 (7 bits)) plus byte padding
    0x00, 0x00, 0x00, 0x00, // gzip footer CRC32
    0x00, 0x00, 0x00, 0x00, // gzip footer uncompressed size
];

/// Mirror of `Bgzf::isBgzfHeader` (Bgzf.hpp:165-178). Returns true
/// when the 18-byte header matches the BGZF magic + FEXTRA layout.
#[inline]
pub fn is_bgzf_header(header: &HeaderBytes) -> bool {
    header[0] == 0x1F
        && header[1] == 0x8B
        && header[2] == 0x08
        && (header[3] & (1u8 << 2)) != 0
        && header[10] == 0x06   // length of extra field is 6B
        && header[11] == 0x00
        && header[12] == b'B'   // subfield ID "BC"
        && header[13] == b'C'
        && header[14] == 0x02   // subfield length is 2B
        && header[15] == 0x00
}

/// Mirror of `Bgzf::getBgzfCompressedSize` (Bgzf.hpp:184-192). Decodes
/// the little-endian `BSIZE - 1` field at header bytes [16..18].
/// Returns `None` if the header is not a BGZF header.
#[inline]
pub fn get_bgzf_compressed_size(header: &HeaderBytes) -> Option<u16> {
    if !is_bgzf_header(header) {
        return None;
    }
    Some(((header[17] as u16) << 8) | header[16] as u16)
}

/// Faithful port of `rapidgzip::blockfinder::Bgzf` (Bgzf.hpp:83-238).
///
/// Wraps an arbitrary [`Read`] + [`Seek`] source. Construction reads
/// the first 18 bytes to confirm the magic header (Bgzf.hpp:111-119)
/// and optionally verifies the trailing 28-byte EOF marker when the
/// source size is known (Bgzf.hpp:122-135).
pub struct BgzfBlockFinder<R> {
    file: R,
    /// Byte offset of the next member's gzip header within the file —
    /// mirror of `m_currentBlockOffset` (Bgzf.hpp:236-237). The
    /// sentinel value [`NO_MORE_BLOCKS`] signals EOF (matches C++'s
    /// `std::numeric_limits<size_t>::max()`).
    current_block_offset: usize,
    /// Cached file size in bytes when known. Mirror of the
    /// `m_fileReader->size()` reads in C++ (Bgzf.hpp:213-215).
    file_size: Option<u64>,
}

impl<R: Read + Seek> BgzfBlockFinder<R> {
    /// Mirror of `Bgzf::Bgzf` (Bgzf.hpp:106-136). Reads the first 18
    /// bytes, verifies the BGZF magic, and (when size is known) checks
    /// the EOF marker block.
    pub fn new(mut file: R) -> io::Result<Self> {
        // Determine current offset (mirror of `m_fileReader->tell()`).
        let current_block_offset = file.stream_position()? as usize;

        // Determine file size if seekable. Rapidgzip uses
        // `m_fileReader->size()`; the closest stable Rust equivalent
        // for an arbitrary `R: Seek` is seek-to-end then restore.
        let file_size = {
            let end = file.seek(SeekFrom::End(0))?;
            file.seek(SeekFrom::Start(current_block_offset as u64))?;
            Some(end)
        };

        // Read 18-byte header.
        let mut header: HeaderBytes = [0; 18];
        file.read_exact(&mut header).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Could not read enough data from given file: {e}"),
            )
        })?;

        if !is_bgzf_header(&header) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Given file does not start with a BGZF header!",
            ));
        }

        // Footer check (Bgzf.hpp:122-135) — only when size is known
        // (otherwise we'd have to buffer the whole stream).
        if let Some(size) = file_size {
            if size >= BGZF_FOOTER.len() as u64 {
                file.seek(SeekFrom::End(-(BGZF_FOOTER.len() as i64)))?;
                let mut footer: FooterBytes = [0; 28];
                file.read_exact(&mut footer).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Could not read enough data from given file for BGZF footer: {e}"),
                    )
                })?;
                if footer != BGZF_FOOTER {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Given file does not end with a BGZF footer!",
                    ));
                }
            }
            file.seek(SeekFrom::Start(current_block_offset as u64))?;
        }

        Ok(Self {
            file,
            current_block_offset,
            file_size,
        })
    }

    /// Mirror of `Bgzf::isBgzfFile` (Bgzf.hpp:138-163) — non-mutating
    /// probe used by the format router. Restores the source position
    /// before returning. `None` is returned on I/O failure rather than
    /// the C++ false return so the caller can distinguish.
    pub fn is_bgzf_file(file: &mut R) -> io::Result<bool> {
        let old_pos = file.stream_position()?;
        let mut header: HeaderBytes = [0; 18];
        let header_ok = match file.read_exact(&mut header) {
            Ok(()) => is_bgzf_header(&header),
            Err(_) => false,
        };
        if !header_ok {
            file.seek(SeekFrom::Start(old_pos))?;
            return Ok(false);
        }

        // Footer probe (mirror of Bgzf.hpp:151-159) — try size; if
        // unavailable just skip and return true on header match alone.
        let end = file.seek(SeekFrom::End(0))?;
        if end >= BGZF_FOOTER.len() as u64 {
            file.seek(SeekFrom::End(-(BGZF_FOOTER.len() as i64)))?;
            let mut footer: FooterBytes = [0; 28];
            let footer_ok = match file.read_exact(&mut footer) {
                Ok(()) => footer == BGZF_FOOTER,
                Err(_) => false,
            };
            file.seek(SeekFrom::Start(old_pos))?;
            return Ok(footer_ok);
        }

        file.seek(SeekFrom::Start(old_pos))?;
        Ok(true)
    }
}

impl<R: Read + Seek> BlockFinderInterface for BgzfBlockFinder<R> {
    /// Mirror of `Bgzf::find()` (Bgzf.hpp:197-233). Returns the bit
    /// offset of the next deflate block (= byte offset of the gzip
    /// header + 18 header bytes, in bits). Advances
    /// `current_block_offset` by `BSIZE - 1 + 1` = `BSIZE` and stops
    /// when the cursor reaches end-of-file.
    fn find(&mut self) -> usize {
        if self.current_block_offset == NO_MORE_BLOCKS {
            return self.current_block_offset;
        }

        let result = (self.current_block_offset + std::mem::size_of::<HeaderBytes>()) * 8;

        // Seek back to current block start, read header, advance.
        if self
            .file
            .seek(SeekFrom::Start(self.current_block_offset as u64))
            .is_err()
        {
            self.current_block_offset = NO_MORE_BLOCKS;
            return result;
        }

        let mut header: HeaderBytes = [0; 18];
        match self.file.read_exact(&mut header) {
            Ok(()) => match get_bgzf_compressed_size(&header) {
                Some(block_size) => {
                    // Mirror of `m_currentBlockOffset += *blockSize + 1`
                    // (Bgzf.hpp:212).
                    self.current_block_offset = self
                        .current_block_offset
                        .saturating_add(block_size as usize + 1);

                    if let Some(size) = self.file_size {
                        if self.current_block_offset as u64 >= size {
                            self.current_block_offset = NO_MORE_BLOCKS;
                        }
                    }
                }
                None => {
                    // Invalid header — mirror of Bgzf.hpp:218-224.
                    self.current_block_offset = NO_MORE_BLOCKS;
                }
            },
            Err(_) => {
                // Partial / failed read — mirror of Bgzf.hpp:225-230.
                self.current_block_offset = NO_MORE_BLOCKS;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal BGZF member with a given fixed payload size.
    /// `payload_size_minus_one` is what the BGZF header encodes.
    fn make_member(payload_size_minus_one: u16, payload: &[u8]) -> Vec<u8> {
        let mut m = vec![
            0x1F, 0x8B, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, // gzip header
            0x06, 0x00, // XLEN = 6
            b'B', b'C', 0x02, 0x00, // FEXTRA subfield "BC", len=2
        ];
        m.push(payload_size_minus_one as u8);
        m.push((payload_size_minus_one >> 8) as u8);
        m.extend_from_slice(payload);
        m
    }

    #[test]
    fn is_bgzf_header_recognises_magic() {
        let m = make_member(0x1B, &[]);
        let mut hdr: HeaderBytes = [0; 18];
        hdr.copy_from_slice(&m[..18]);
        assert!(is_bgzf_header(&hdr));
        // Negative case: flip a magic byte.
        hdr[0] = 0;
        assert!(!is_bgzf_header(&hdr));
    }

    #[test]
    fn get_bgzf_compressed_size_decodes_little_endian() {
        let m = make_member(0x1B, &[]);
        let mut hdr: HeaderBytes = [0; 18];
        hdr.copy_from_slice(&m[..18]);
        assert_eq!(get_bgzf_compressed_size(&hdr), Some(0x1B));
    }

    #[test]
    fn rejects_non_bgzf_input() {
        let bad = vec![0u8; 18];
        let r = BgzfBlockFinder::new(Cursor::new(bad));
        assert!(r.is_err(), "must reject non-BGZF magic");
    }

    #[test]
    fn find_advances_through_eof_marker_only() {
        // Construct a minimal BGZF file consisting of just the EOF
        // marker block. The marker block has BSIZE - 1 = 0x1B = 27
        // so total length 28, matching the BGZF footer size.
        let mut file = BGZF_FOOTER.to_vec();
        // Sanity: the EOF marker IS a valid BGZF header.
        let mut hdr: HeaderBytes = [0; 18];
        hdr.copy_from_slice(&file[..18]);
        assert!(is_bgzf_header(&hdr));

        let mut bf = BgzfBlockFinder::new(Cursor::new(&mut file)).unwrap();
        // First find: returns bit offset (0 + 18) * 8 = 144 and
        // advances cursor to 28 (= file size), then next call returns
        // sentinel.
        assert_eq!(bf.find(), 144);
        assert_eq!(bf.find(), NO_MORE_BLOCKS);
        assert_eq!(bf.find(), NO_MORE_BLOCKS);
    }

    #[test]
    fn find_walks_multiple_members() {
        // Three small BGZF members (each 18B header + 10B body), then
        // EOF marker. Each body is `BSIZE - 1 + 1 - 18 = body_len` →
        // body_len=10 ⇒ BSIZE - 1 = 27, so we pass 27.
        let body = vec![0u8; 10];
        let m1 = make_member(27, &body);
        let m2 = make_member(27, &body);
        let m3 = make_member(27, &body);
        assert_eq!(m1.len(), 28);

        let mut file = Vec::new();
        file.extend_from_slice(&m1);
        file.extend_from_slice(&m2);
        file.extend_from_slice(&m3);
        file.extend_from_slice(&BGZF_FOOTER);

        let mut bf = BgzfBlockFinder::new(Cursor::new(&mut file)).unwrap();
        // Members start at 0, 28, 56, 84 — bit offsets of deflate
        // block (post 18-byte header) are (offset + 18) * 8.
        let mut offsets = Vec::new();
        loop {
            let o = bf.find();
            if o == NO_MORE_BLOCKS {
                break;
            }
            offsets.push(o);
            if offsets.len() > 100 {
                panic!("find did not terminate");
            }
        }
        // Each BGZF member is 28 bytes (18 hdr + 10 body); the
        // deflate block starts at byte offset +18, in bits *8.
        assert_eq!(
            offsets,
            vec![18 * 8, (28 + 18) * 8, (56 + 18) * 8, (84 + 18) * 8]
        );
    }

    #[test]
    fn is_bgzf_file_probe_restores_position() {
        // The probe reads from the CURRENT position (mirror of C++'s
        // `oldPos = file->tell()` at Bgzf.hpp:141). When called at the
        // file start the EOF marker IS a valid BGZF block, so the
        // probe returns true and restores position 0.
        let mut file = BGZF_FOOTER.to_vec();
        let mut cursor = Cursor::new(&mut file);
        assert_eq!(cursor.stream_position().unwrap(), 0);
        let ok = BgzfBlockFinder::is_bgzf_file(&mut cursor).unwrap();
        assert!(ok);
        assert_eq!(cursor.stream_position().unwrap(), 0);

        // Probing from a mid-file offset (where the magic isn't
        // present) should return false and still restore.
        cursor.seek(SeekFrom::Start(5)).unwrap();
        let ok2 = BgzfBlockFinder::is_bgzf_file(&mut cursor).unwrap();
        assert!(!ok2);
        assert_eq!(cursor.stream_position().unwrap(), 5);
    }
}
