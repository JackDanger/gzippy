//! Literal structural port of `rapidgzip::GzipReader`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/GzipReader.hpp:40-756).
//!
//! A strictly sequential gzip interface that can iterate over multiple
//! gzip streams and deflate blocks. **Not** parallelized — used as the
//! reference for `ParallelGzipReader` behavior (GzipReader.hpp:37-38).
//!
//! gzippy production decompression already routes single- and
//! multi-member gzip through the optimized paths in
//! [`crate::decompress::decompress_gzip_libdeflate`] and friends; this
//! module is the rapidgzip-shaped sequential reader API for the new
//! `ParallelGzipReader` port to compare against / fall back to.
//!
//! State machine (mirror of `read()` at GzipReader.hpp:282-360):
//!
//! ```text
//!  NONE / END_OF_STREAM      → readHeader
//!  END_OF_STREAM_HEADER / END_OF_BLOCK → readBlockHeader (or readFooter
//!                                                          if last block)
//!  END_OF_BLOCK_HEADER       → readBlock (decode + flush)
//! ```
//!
//! `m_currentPoint` cycles through the [`StoppingPoint`] values until
//! `read()` has consumed the user's requested `n_bytes_to_read` *or*
//! reached EOF, hitting the optional caller-supplied `stopping_points`
//! mask along the way (mirror of `testFlags(*m_currentPoint, stoppingPoints)`
//! at GzipReader.hpp:349).
//!
//! Inner decoder: the vendor uses `deflate::Block<>` for block-by-block
//! decode (GzipReader.hpp:44). gzippy hasn't ported `Block<>` in its
//! reader-driving form (we have a marker-emitting bootstrap version
//! inside `parallel::deflate_block`). This port uses libdeflate via the
//! existing `backends::libdeflate` FFI for full-member decoding and
//! emits `END_OF_STREAM` once per member, which matches the only
//! observable behavior callers (the upcoming `ParallelGzipReader` port)
//! depend on for sequential reference. Block-granular boundaries
//! (`END_OF_BLOCK_HEADER` / `END_OF_BLOCK`) are reported but coalesced
//! to one event per member — a full Block<> port will refine this.

#![allow(dead_code)]

use std::io::{self, Write};

use crate::backends::libdeflate::{DecompressError, DecompressorEx};
use crate::decompress::parallel::gzip_format::{read_footer, read_header, Footer, Header};

/// Mirror of `rapidgzip::StoppingPoint` (definitions.hpp:92-100).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoppingPoint {
    None = 0,
    /// `END_OF_STREAM_HEADER` (1 << 0). Set after the gzip header has
    /// been parsed but before the first deflate block.
    EndOfStreamHeader = 1 << 0,
    /// `END_OF_STREAM` (1 << 1). Set after the gzip footer has been
    /// verified; the next call to `read()` will try to read another
    /// member's header.
    EndOfStream = 1 << 1,
    /// `END_OF_BLOCK_HEADER` (1 << 2). Set after `readBlockHeader` has
    /// finished parsing a deflate block's header. The coalesced port
    /// reports this once per member, after the header parse but before
    /// the libdeflate full decode.
    EndOfBlockHeader = 1 << 2,
    /// `END_OF_BLOCK` (1 << 3). Set after the inner decode has finished
    /// consuming a deflate block; coalesced to one event per member.
    EndOfBlock = 1 << 3,
}

impl StoppingPoint {
    /// Mirror of the bitmask test at GzipReader.hpp:349
    /// (`testFlags(*m_currentPoint, stoppingPoints)`).
    pub fn matches_mask(self, mask: u32) -> bool {
        mask == u32::MAX || (mask & self as u32) != 0
    }
}

/// Convenience mask value for `read()`'s `stopping_points` parameter —
/// matches `StoppingPoint::ALL = 0xFFFFFFFF` at definitions.hpp:99.
pub const STOPPING_POINTS_ALL: u32 = u32::MAX;
/// "No early stop" — matches `StoppingPoint::NONE = 0` at
/// definitions.hpp:94.
pub const STOPPING_POINTS_NONE: u32 = 0;

/// Errors surfaced by [`GzipReader`].
#[derive(Debug)]
pub enum GzipReaderError {
    /// Gzip header parse error. Mirror of GzipReader.hpp:432-436.
    InvalidHeader(io::Error),
    /// Gzip footer parse / size verify error. Mirror of GzipReader.hpp:702-708.
    InvalidFooter(String),
    /// Inner deflate decode error. Mirror of GzipReader.hpp:660-663.
    DeflateError(String),
    /// Underlying I/O error.
    Io(io::Error),
}

impl std::fmt::Display for GzipReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GzipReaderError::InvalidHeader(e) => write!(f, "invalid gzip header: {e}"),
            GzipReaderError::InvalidFooter(s) => write!(f, "invalid gzip footer: {s}"),
            GzipReaderError::DeflateError(s) => write!(f, "deflate decode error: {s}"),
            GzipReaderError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for GzipReaderError {}

impl From<io::Error> for GzipReaderError {
    fn from(e: io::Error) -> Self {
        GzipReaderError::Io(e)
    }
}

/// Faithful structural port of `rapidgzip::GzipReader`
/// (GzipReader.hpp:40-585), restricted to the in-memory variant
/// (the vendor accepts an `UniqueFileReader`; gzippy operates on a
/// byte slice because that matches all current call sites: mmap-backed
/// stdin or in-memory test fixtures).
pub struct GzipReader<'a> {
    /// `m_fileReader` collapsed to the underlying byte buffer.
    buffer: &'a [u8],
    /// `m_bitReader.tell() / 8` — current byte cursor into `buffer`.
    /// gzippy's libdeflate-based per-member decode runs at byte
    /// granularity, so this is sufficient for the public surface even
    /// though the vendor's `m_bitReader` tracks bit offsets.
    byte_cursor: usize,
    /// `m_currentPosition` (GzipReader.hpp:557).
    decoded_position: u64,
    /// `m_atEndOfFile` (GzipReader.hpp:558).
    at_end_of_file: bool,
    /// `m_currentPoint` (GzipReader.hpp:571). Initial value `END_OF_STREAM`
    /// matches the C++ default-initialization comment.
    current_point: Option<StoppingPoint>,
    /// `m_streamBytesCount` (GzipReader.hpp:573).
    stream_bytes_count: u64,
    /// `m_didReadHeader` (GzipReader.hpp:584).
    did_read_header: bool,
    /// Last-parsed gzip header (held while we're between header and
    /// footer). gzippy collapses the vendor's `m_currentDeflateBlock`
    /// state into "we have a header → can decode → produce a footer".
    current_header: Option<Header>,
    /// Byte offset where the current member's gzip header started. The
    /// libdeflate FFI wants the full member (header+deflate+footer); we
    /// rewind to this position before calling it.
    member_start_byte: usize,
    /// Per-stream verifying CRC32. Mirror of `m_crc32Calculator`
    /// (GzipReader.hpp:580). Updated by `read_block`, verified by
    /// `read_footer_inner`.
    crc32: u32,
}

impl<'a> GzipReader<'a> {
    /// Construct over a borrowed buffer. Mirror of the in-memory test
    /// constructor at GzipReader.hpp:49-58 (plus the in-memory
    /// FileReader collapse).
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            byte_cursor: 0,
            decoded_position: 0,
            at_end_of_file: buffer.is_empty(),
            current_point: Some(StoppingPoint::EndOfStream),
            stream_bytes_count: 0,
            did_read_header: false,
            current_header: None,
            member_start_byte: 0,
            crc32: 0,
        }
    }

    /// Mirror of `tell()` (GzipReader.hpp:133-137).
    pub fn tell(&self) -> u64 {
        self.decoded_position
    }

    /// Mirror of `tellCompressed()` (GzipReader.hpp:228-232) — returns
    /// bits, like the C++ method.
    pub fn tell_compressed_bits(&self) -> u64 {
        self.byte_cursor as u64 * 8
    }

    /// Mirror of `eof()` (GzipReader.hpp:121-125).
    pub fn eof(&self) -> bool {
        self.at_end_of_file
    }

    /// Mirror of `currentPoint()` (GzipReader.hpp:234-238).
    pub fn current_point(&self) -> Option<StoppingPoint> {
        self.current_point
    }

    /// Mirror of `setCRC32Enabled` (GzipReader.hpp:362-366). We always
    /// compute CRC32 in this port (cheap with crc32fast); the setter
    /// exists for API parity.
    pub fn set_crc32_enabled(&mut self, _enabled: bool) {}

    /// Faithful port of the state-machine `read()` loop
    /// (GzipReader.hpp:282-360).
    ///
    /// Drives the gzip reader until at most `n_bytes_to_read` bytes have
    /// been emitted (the WriteFunctor closure inlines to the caller's
    /// `writer`), EOF is reached, or one of the bits in
    /// `stopping_points` matches `current_point` after an internal step.
    ///
    /// Returns the number of *decoded* bytes flushed to `writer`.
    pub fn read<W: Write>(
        &mut self,
        writer: &mut W,
        n_bytes_to_read: usize,
        stopping_points: u32,
    ) -> Result<usize, GzipReaderError> {
        let mut n_bytes_decoded = 0;

        while !self.eof() {
            // The vendor branches on `m_currentPoint`:
            //   None | END_OF_BLOCK_HEADER → readBlock
            //   END_OF_STREAM_HEADER | END_OF_BLOCK → readBlockHeader
            //                                          (or readFooter if eos)
            //   NONE | END_OF_STREAM → readHeader
            // (GzipReader.hpp:293-343).
            //
            // gzippy coalesces readBlockHeader + readBlock into a single
            // per-member libdeflate call, but preserves the StoppingPoint
            // emissions in the documented order. The state transitions:
            //   EndOfStream                → readHeader → EndOfStreamHeader
            //   EndOfStreamHeader          → readBlockHeader (no-op for
            //                                 our coalesced port) →
            //                                 EndOfBlockHeader
            //   EndOfBlockHeader / None    → readMemberDecode → EndOfBlock
            //   EndOfBlock                 → readFooter → EndOfStream
            match self.current_point {
                None | Some(StoppingPoint::EndOfBlockHeader) => {
                    let decoded =
                        self.read_member_decode(writer, n_bytes_to_read - n_bytes_decoded)?;
                    n_bytes_decoded += decoded;
                    self.stream_bytes_count += decoded as u64;
                    self.current_point = Some(StoppingPoint::EndOfBlock);
                }
                Some(StoppingPoint::None) | Some(StoppingPoint::EndOfStream) => {
                    if self.byte_cursor >= self.buffer.len() {
                        self.at_end_of_file = true;
                        break;
                    }
                    self.read_header_inner()?;
                }
                Some(StoppingPoint::EndOfStreamHeader) => {
                    // Mirror of `readBlockHeader` (GzipReader.hpp:399-412).
                    // gzippy's libdeflate-based coalesced decode doesn't
                    // pre-parse block headers, so this is a state-only
                    // transition that satisfies the vendor's
                    // emit-order contract.
                    self.current_point = Some(StoppingPoint::EndOfBlockHeader);
                }
                Some(StoppingPoint::EndOfBlock) => {
                    // After all blocks in the member finish, read footer.
                    // libdeflate already consumed the whole member, so
                    // there is always exactly one EndOfBlock per member
                    // before the footer.
                    self.read_footer_inner()?;
                }
            }

            if let Some(cp) = self.current_point {
                if cp != StoppingPoint::None && cp.matches_mask(stopping_points) {
                    break;
                }
            }
            if n_bytes_decoded >= n_bytes_to_read {
                break;
            }
        }

        self.decoded_position += n_bytes_decoded as u64;
        if self.byte_cursor >= self.buffer.len() {
            self.at_end_of_file = true;
        }
        Ok(n_bytes_decoded)
    }

    /// Mirror of `readHeader()` (GzipReader.hpp:422-461). The C++ branches
    /// on `m_fileType` (GZIP / BGZF / ZLIB / DEFLATE / NONE); we only
    /// handle gzip-family streams per the cutover scope.
    fn read_header_inner(&mut self) -> Result<(), GzipReaderError> {
        self.member_start_byte = self.byte_cursor;
        let (header, consumed) = read_header(&self.buffer[self.byte_cursor..])
            .map_err(GzipReaderError::InvalidHeader)?;
        self.byte_cursor += consumed;
        self.current_header = Some(header);
        self.did_read_header = true;
        self.stream_bytes_count = 0;
        self.crc32 = 0;
        self.current_point = Some(StoppingPoint::EndOfStreamHeader);
        Ok(())
    }

    /// Coalesced member decode. Mirror of the read loop at
    /// GzipReader.hpp:645-687, but using libdeflate one-shot instead of
    /// `deflate::Block<>::read` per-block. The vendor's intermediate
    /// `m_currentDeflateBlock` / `m_lastBlockData` state machine
    /// collapses to a single libdeflate call here.
    fn read_member_decode<W: Write>(
        &mut self,
        writer: &mut W,
        max_bytes: usize,
    ) -> Result<usize, GzipReaderError> {
        if self.current_header.is_none() {
            // Caller jumped straight into a block decode without reading
            // a header first — mirror of GzipReader.hpp:647-648.
            return Err(GzipReaderError::DeflateError(
                "Call readHeader and readBlockHeader before calling readBlock!".to_string(),
            ));
        }
        // Try to decompress the member starting at member_start_byte
        // (we already advanced byte_cursor past the gzip header in
        // read_header_inner for state-machine bookkeeping, but
        // libdeflate's FFI consumes the full gzip member — header,
        // deflate stream, and footer — so we pass the raw bytes from
        // the member's start).
        let remaining = &self.buffer[self.member_start_byte..];
        let mut decompressor = DecompressorEx::new();
        // The vendor caps individual reads via `nMaxBytesToDecode`; we
        // pick an initial allocation cap of 1 MiB and double on
        // InsufficientSpace. A `usize::MAX` request from a caller (used
        // as "read everything") would overflow if we tried to allocate
        // max_bytes*2 up front.
        let initial_cap = max_bytes.clamp(64 * 1024, 16 * 1024 * 1024);
        let mut cap = initial_cap;
        loop {
            let mut out = vec![0u8; cap];
            match decompressor.gzip_decompress_ex(remaining, &mut out) {
                Ok(result) => {
                    let written = result.output_size.min(max_bytes);
                    if written > 0 {
                        writer.write_all(&out[..written])?;
                    }
                    // Compute CRC32 from the decoded bytes for the
                    // footer verification step (libdeflate verifies
                    // internally too, but read_footer_inner replays the
                    // check at this level to mirror the C++ contract).
                    self.crc32 = crc32fast::hash(&out[..result.output_size]);
                    // libdeflate consumed the full member including its
                    // gzip header & footer; advance from the member
                    // start by the total consumed bytes.
                    self.byte_cursor = self.member_start_byte + result.input_consumed;
                    return Ok(written);
                }
                Err(DecompressError::InsufficientSpace) => {
                    cap *= 2;
                    if cap > 1 << 30 {
                        return Err(GzipReaderError::DeflateError(
                            "decoded buffer exceeds 1 GiB".to_string(),
                        ));
                    }
                }
                Err(e) => {
                    return Err(GzipReaderError::DeflateError(format!("libdeflate: {e:?}")));
                }
            }
        }
    }

    /// Mirror of `readFooter()` (GzipReader.hpp:691-755), gzip branch only.
    ///
    /// libdeflate already consumed the gzip footer as part of
    /// [`Self::read_member_decode`]; we walk back 8 bytes and verify
    /// the size + CRC32 the vendor would have read from the bit stream.
    /// This satisfies the structural invariant
    /// `crc32Calculator.verify(footer.crc32)` at GzipReader.hpp:715
    /// even though our calc is computed inside libdeflate.
    fn read_footer_inner(&mut self) -> Result<(), GzipReaderError> {
        if self.byte_cursor < 8 {
            return Err(GzipReaderError::InvalidFooter(
                "byte cursor at less than 8 — footer cannot exist".to_string(),
            ));
        }
        let footer_at = self.byte_cursor - 8;
        let footer: Footer = read_footer(self.buffer, footer_at)
            .map_err(|e| GzipReaderError::InvalidFooter(e.to_string()))?;

        // size check (GzipReader.hpp:702-707).
        if self.did_read_header && (self.stream_bytes_count as u32) != footer.uncompressed_size {
            return Err(GzipReaderError::InvalidFooter(format!(
                "size mismatch: stream {} vs footer {}",
                self.stream_bytes_count as u32, footer.uncompressed_size
            )));
        }
        // CRC check (GzipReader.hpp:714-716).
        if self.did_read_header && self.crc32 != footer.crc32 {
            return Err(GzipReaderError::InvalidFooter(format!(
                "crc32 mismatch: computed {:08x} vs footer {:08x}",
                self.crc32, footer.crc32
            )));
        }

        if self.byte_cursor >= self.buffer.len() {
            self.at_end_of_file = true;
        }
        self.current_point = Some(StoppingPoint::EndOfStream);
        self.did_read_header = false;
        self.current_header = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    fn make_member(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn reads_single_member() {
        let payload = b"hello, gzip!";
        let stream = make_member(payload);
        let mut reader = GzipReader::new(&stream);
        let mut out = Vec::new();
        let n = reader
            .read(&mut out, usize::MAX, STOPPING_POINTS_NONE)
            .unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(&out[..], payload);
        assert!(reader.eof());
    }

    #[test]
    fn reads_multi_member() {
        let mut stream = make_member(b"first");
        stream.extend(make_member(b"second"));
        stream.extend(make_member(b"third"));
        let mut reader = GzipReader::new(&stream);
        let mut out = Vec::new();
        // Drive multiple read() calls — each must continue through the
        // member boundaries.
        loop {
            let n = reader.read(&mut out, 1024, STOPPING_POINTS_NONE).unwrap();
            if n == 0 && reader.eof() {
                break;
            }
            if n == 0 {
                break;
            }
        }
        assert_eq!(&out[..], b"firstsecondthird");
        assert!(reader.eof());
    }

    #[test]
    fn early_stop_on_end_of_block() {
        let stream = make_member(b"abcdef");
        let mut reader = GzipReader::new(&stream);
        let mut out = Vec::new();
        let _ = reader
            .read(&mut out, usize::MAX, StoppingPoint::EndOfBlock as u32)
            .unwrap();
        // We expect to have decoded the payload and stopped at the
        // end-of-block marker (single-block member).
        assert_eq!(&out[..], b"abcdef");
        assert_eq!(reader.current_point(), Some(StoppingPoint::EndOfBlock));
    }

    #[test]
    fn bad_footer_is_rejected() {
        let mut stream = make_member(b"hello");
        // Corrupt CRC32 in footer (footer is last 8 bytes).
        let len = stream.len();
        stream[len - 8] ^= 0xFF;
        let mut reader = GzipReader::new(&stream);
        let mut out = Vec::new();
        let err = reader
            .read(&mut out, usize::MAX, STOPPING_POINTS_NONE)
            .unwrap_err();
        match err {
            GzipReaderError::InvalidFooter(_) | GzipReaderError::DeflateError(_) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
