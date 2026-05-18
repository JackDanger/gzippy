//! Literal port of `rapidgzip::blockfinder::PigzStringView`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/blockfinder/PigzStringView.hpp:30-178).
//!
//! Finds pigz-style empty-deflate-block "sync" markers (`\0\0\xFF\xFF`)
//! in the compressed stream and reports the bit offset of the deflate
//! block that **starts immediately after** each marker. pigz emits
//! such an empty block at every chunk boundary so multi-threaded
//! decoders can chop the file along those boundaries.
//!
//! The vendor implementation chases ~8 GB/s using `std::string_view::find`
//! over a 16 KiB rolling buffer. The Rust port uses
//! [`memchr::memmem::Finder`] for the same algorithmic shape — a
//! sub-string search over an aligned chunk plus a small boundary
//! buffer that spans the last 4 bytes of the previous chunk and the
//! first 4 bytes of the next chunk (PigzStringView.hpp:97-110).

#![allow(dead_code)]

use std::io::{self, Read, Seek, SeekFrom};

use memchr::memmem;

use crate::decompress::parallel::blockfinder_interface::{BlockFinderInterface, NO_MORE_BLOCKS};
use crate::decompress::parallel::gzip_format;

/// Mirror of `PigzStringView::BUFFER_SIZE` (PigzStringView.hpp:39).
/// "Should probably be larger than the I/O block size of 4096 B and
/// smaller than most L1 cache sizes."
pub const BUFFER_SIZE: usize = 16 * 1024;

/// Mirror of `PigzStringView::MAGIC_BIT_STRING_SIZE` (PigzStringView.hpp:40).
/// The 35-bit empty-deflate-block trailer pattern: bfinal=0, btype=00
/// (stored), then 32 bits of `LEN=0 / NLEN=0xFFFF`, i.e.
/// `\x00\x00\xFF\xFF` aligned at a 3-bit boundary in the byte before.
pub const MAGIC_BIT_STRING_SIZE: u8 = 35;

/// Mirror of `EMPTY_DEFLATE_BLOCK` (PigzStringView.hpp:175-177).
pub const EMPTY_DEFLATE_BLOCK: &[u8; 4] = b"\x00\x00\xFF\xFF";

/// Number of bytes we have to carry over between adjacent reads to
/// catch matches that straddle a 16 KiB boundary. Mirror of
/// `nBytesToRetain = ceilDiv(MAGIC_BIT_STRING_SIZE, CHAR_BIT) - 1 = 4`
/// (PigzStringView.hpp:97-98).
const N_BYTES_TO_RETAIN: usize = (MAGIC_BIT_STRING_SIZE as usize).div_ceil(8) - 1;

/// Faithful port of `rapidgzip::blockfinder::PigzStringView`.
pub struct PigzStringViewBlockFinder<R> {
    /// Source reader. Mirror of `m_fileReader` (PigzStringView.hpp:165).
    file: R,
    /// Cached file size in bytes (mirror of `m_fileSize`,
    /// PigzStringView.hpp:166).
    file_size: Option<u64>,
    /// Rolling 16 KiB read buffer (mirror of `m_buffer`,
    /// PigzStringView.hpp:168).
    buffer: Vec<u8>,
    /// Number of valid bytes in `buffer` (mirror of `m_bufferSize`,
    /// PigzStringView.hpp:169).
    buffer_size: usize,
    /// Whether we've already located the first deflate block via
    /// gzip-header parsing (mirror of `foundFirstBlock`,
    /// PigzStringView.hpp:171).
    found_first_block: bool,
    /// Pending block offsets (in **bytes**, post-marker). Mirror of
    /// `m_blockOffsets` (PigzStringView.hpp:172). They are converted
    /// to bit offsets when popped in `find()`.
    block_offsets: Vec<usize>,
    /// Last bit offset returned — mirror of
    /// `m_lastReturnedBlockOffset` (PigzStringView.hpp:173). Kept for
    /// parity with the vendor class even though it is not used to
    /// influence behaviour.
    last_returned_block_offset: usize,
    /// Whether the source has been observed at EOF / failed. Mirror
    /// of the various `eof() / fail() / closed()` checks scattered
    /// through PigzStringView.hpp.
    exhausted: bool,
    /// Memmem finder for `EMPTY_DEFLATE_BLOCK`. Lifted to a field so
    /// we don't rebuild the SIMD table on every chunk.
    finder: memmem::Finder<'static>,
}

impl<R: Read + Seek> PigzStringViewBlockFinder<R> {
    /// Mirror of the public constructor (PigzStringView.hpp:43-47).
    /// Captures the source size when known.
    pub fn new(mut file: R) -> io::Result<Self> {
        let size_opt = match file.seek(SeekFrom::End(0)) {
            Ok(end) => {
                file.seek(SeekFrom::Start(0))?;
                Some(end)
            }
            Err(_) => None,
        };
        Ok(Self {
            file,
            file_size: size_opt,
            buffer: vec![0u8; BUFFER_SIZE],
            buffer_size: 0,
            found_first_block: false,
            block_offsets: Vec::new(),
            last_returned_block_offset: 0,
            exhausted: false,
            finder: memmem::Finder::new(EMPTY_DEFLATE_BLOCK).into_owned(),
        })
    }

    /// Mirror of `findBlockOffsets` (PigzStringView.hpp:73-92).
    /// Scans `view` for `EMPTY_DEFLATE_BLOCK` patterns and pushes the
    /// **byte offset of the first byte after the marker** to
    /// `block_offsets` — but only if the byte preceding the marker
    /// has its top three bits clear (i.e. the deflate padding zeros
    /// are correctly aligned with the bfinal=0/btype=00 prefix that
    /// pigz emits at sync points).
    fn find_block_offsets(&mut self, view: &[u8], view_offset: usize) {
        for position in self.finder.find_iter(view) {
            if position >= 1 {
                // Only matches if the three bits above the marker are
                // zero — i.e. the byte preceding the four-byte run
                // has its top three bits clear.
                if (view[position - 1] & 0b1110_0000) == 0 {
                    let total_offset = view_offset + position + EMPTY_DEFLATE_BLOCK.len();
                    let within_file = match self.file_size {
                        Some(size) => (total_offset as u64) < size,
                        None => true,
                    };
                    if within_file {
                        self.block_offsets.push(total_offset);
                    }
                }
            }
        }
    }

    /// Mirror of `analyzeNextChunk` (PigzStringView.hpp:94-127). Reads
    /// the next 16 KiB chunk, then scans the chunk plus an 8-byte
    /// boundary buffer that overlaps the tail of the previous chunk.
    fn analyze_next_chunk(&mut self) -> io::Result<()> {
        let check_boundary = self.buffer_size > 0;
        let mut boundary_buffer: [u8; 2 * N_BYTES_TO_RETAIN] = [0; 2 * N_BYTES_TO_RETAIN];
        let mut boundary_buffer_size = 0usize;

        if check_boundary {
            boundary_buffer_size = self.buffer_size.min(N_BYTES_TO_RETAIN);
            let start = self.buffer_size - boundary_buffer_size;
            boundary_buffer[..boundary_buffer_size]
                .copy_from_slice(&self.buffer[start..start + boundary_buffer_size]);
        }

        // Always read BUFFER_SIZE bytes for aligned I/O (mirror of
        // PigzStringView.hpp:112-115).
        let buffer_offset = self.file.stream_position()? as usize;
        let boundary_buffer_offset = buffer_offset.saturating_sub(boundary_buffer_size);

        self.buffer_size = read_full(&mut self.file, &mut self.buffer)?;

        if self.buffer_size == 0 {
            self.exhausted = true;
            return Ok(());
        }

        if check_boundary {
            let add = N_BYTES_TO_RETAIN.min(self.buffer_size);
            boundary_buffer[N_BYTES_TO_RETAIN..N_BYTES_TO_RETAIN + add]
                .copy_from_slice(&self.buffer[..add]);
            boundary_buffer_size += add;

            // Mirror of the findBlockOffsets call on the boundary view
            // (PigzStringView.hpp:123).
            let slice_owned = boundary_buffer[..boundary_buffer_size].to_vec();
            self.find_block_offsets(&slice_owned, boundary_buffer_offset);
        }

        // Mirror of findBlockOffsets on the main buffer
        // (PigzStringView.hpp:126).
        let slice_owned = self.buffer[..self.buffer_size].to_vec();
        self.find_block_offsets(&slice_owned, buffer_offset);

        Ok(())
    }

    /// Mirror of `findFirstBlock` (PigzStringView.hpp:129-162). Parses
    /// the gzip header to locate the first deflate block boundary, or
    /// declares the input exhausted if the header is invalid.
    fn find_first_block(&mut self) -> io::Result<()> {
        // Read up to BUFFER_SIZE bytes from the start of the file.
        self.file.seek(SeekFrom::Start(0))?;
        let mut probe = vec![0u8; BUFFER_SIZE];
        let n = read_full(&mut self.file, &mut probe)?;
        probe.truncate(n);

        match gzip_format::read_header(&probe) {
            Ok((_, header_end)) => {
                // The vendor implementation requires the bit reader to
                // be byte-aligned post-header (PigzStringView.hpp:149-150);
                // our read_header always returns a byte boundary.
                self.block_offsets.push(header_end);
                // Mirror of `seekTo(0)` + `m_bufferSize = 0`
                // (PigzStringView.hpp:152-154).
                self.file.seek(SeekFrom::Start(0))?;
                self.buffer_size = 0;
                self.found_first_block = true;
            }
            Err(_) => {
                // "If the first block couldn't be found, then don't
                // even try to search for the others because the first
                // block would be missing." (PigzStringView.hpp:158-161)
                self.exhausted = true;
            }
        }
        Ok(())
    }
}

/// Helper: try to read until the buffer is full or EOF.
fn read_full<R: Read>(file: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        match file.read(&mut buf[total..])? {
            0 => break,
            n => total += n,
        }
    }
    Ok(total)
}

impl<R: Read + Seek> BlockFinderInterface for PigzStringViewBlockFinder<R> {
    /// Mirror of `find()` (PigzStringView.hpp:52-70). Lazily fills
    /// `block_offsets` and returns the most recently appended offset
    /// in **bits**; returns [`NO_MORE_BLOCKS`] when the input is
    /// exhausted with no offsets left.
    fn find(&mut self) -> usize {
        while self.block_offsets.is_empty() && !self.exhausted {
            let step = if self.found_first_block {
                self.analyze_next_chunk()
            } else {
                self.find_first_block()
            };
            if step.is_err() {
                self.exhausted = true;
            }
        }

        if let Some(byte_offset) = self.block_offsets.pop() {
            // Mirror of `m_lastReturnedBlockOffset = ... * CHAR_BIT`
            // (PigzStringView.hpp:67-69).
            self.last_returned_block_offset = byte_offset * 8;
            self.last_returned_block_offset
        } else {
            NO_MORE_BLOCKS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Make a minimal pigz-style file: gzip header + a "marker" body
    /// constructed so we can find a synthetic sync run inside.
    fn make_with_marker(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![
            0x1F, 0x8B, 0x08, 0x00, // ID1, ID2, CM, FLG=0
            0x00, 0x00, 0x00, 0x00, // mtime
            0x00, // XFL
            0x03, // OS = Unix
        ];
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn returns_first_block_after_header_then_sentinel_for_empty_payload() {
        let data = make_with_marker(&[]);
        let mut bf = PigzStringViewBlockFinder::new(Cursor::new(data)).unwrap();
        // First find triggers `findFirstBlock` which produces one
        // byte-offset (10) → bit offset 80.
        assert_eq!(bf.find(), 80);
        // No further sync markers → sentinel.
        assert_eq!(bf.find(), NO_MORE_BLOCKS);
    }

    #[test]
    fn finds_sync_markers_in_payload() {
        // Layout: gzip header (10B) + 5 padding bytes + sync marker
        // (`00 00 FF FF`) + 5 trailing bytes. The byte preceding the
        // marker is at offset 14 — must have top three bits clear.
        let mut payload = vec![0xAAu8; 5]; // 0xAA top bits not clear ⇒ no match
        payload.extend_from_slice(EMPTY_DEFLATE_BLOCK);
        payload.extend_from_slice(&[0xBBu8; 5]);
        let data = make_with_marker(&payload);
        let mut bf = PigzStringViewBlockFinder::new(Cursor::new(data)).unwrap();
        // Even though the marker bytes match, the prefix byte is 0xAA
        // (top three bits set) ⇒ rejected. Only the gzip-header first
        // block remains.
        assert_eq!(bf.find(), 80);
        assert_eq!(bf.find(), NO_MORE_BLOCKS);
    }

    #[test]
    fn accepts_marker_with_clean_prefix_byte() {
        // Prefix byte 0x00 has top three bits clear → match accepted.
        let mut payload = vec![0u8; 5];
        payload.extend_from_slice(EMPTY_DEFLATE_BLOCK);
        payload.extend_from_slice(&[0xBBu8; 5]);
        let data = make_with_marker(&payload);
        let total_len = data.len();
        let mut bf = PigzStringViewBlockFinder::new(Cursor::new(data)).unwrap();
        // The marker sits at offset 15 in the file (10 header + 5
        // padding) and its end+1 byte position is 15 + 4 = 19. The
        // post-marker byte offset = 19 → 152 bits.
        // First call: find_first_block populates [10], then loop
        // continues only if found_first_block && empty; but here we
        // also need analyze_next_chunk to find the sync. After we
        // pop the first block (10), subsequent calls drive the chunk
        // scanner.
        let first = bf.find();
        let second = bf.find();
        let third = bf.find();
        let mut offsets = vec![first, second, third];
        offsets.retain(|&o| o != NO_MORE_BLOCKS);
        assert!(
            offsets.contains(&80),
            "should report first deflate block: {offsets:?}"
        );
        assert!(
            offsets.contains(&(19 * 8)),
            "should report post-marker block at bit 152: {offsets:?}"
        );
        // Sanity: post-marker offset is within the file.
        assert!(19 < total_len);
    }

    #[test]
    fn rejects_input_without_valid_gzip_header() {
        // 10 zero bytes — not a gzip magic.
        let data = vec![0u8; 32];
        let mut bf = PigzStringViewBlockFinder::new(Cursor::new(data)).unwrap();
        assert_eq!(bf.find(), NO_MORE_BLOCKS);
    }

    #[test]
    fn n_bytes_to_retain_matches_vendor_constant() {
        // PigzStringView.hpp:97-98 hardcodes the expected value of 4.
        assert_eq!(N_BYTES_TO_RETAIN, 4);
    }
}
