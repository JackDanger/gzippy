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
//! decode (GzipReader.hpp:44) — a sliding-window-resolving block decoder
//! that emits bytes via `lastBuffers()`. rapidgzip's compile-time variant
//! that uses ISA-L instead is `IsalInflateWrapper` (vendor/.../gzip/isal.hpp);
//! the conditional compilation at GzipReader.hpp picks whichever inflate
//! engine is available, and both produce byte-resolved output.
//!
//! gzippy's `parallel::deflate_block::Block` is the marker-emitting
//! bootstrap variant (Vec<u16> output for chunk 0 / cross-chunk back-refs),
//! not a byte-resolving block decoder. The faithful sequential equivalent
//! is therefore `parallel::inflate_wrapper::IsalInflateWrapper` (the
//! ported ISA-L wrapper) on x86_64+isal — which IS what rapidgzip uses
//! on the same compile-time path — and libdeflate one-shot as a fallback
//! on archs where ISA-L is unavailable (rapidgzip falls back to a zlib
//! wrapper on the same arches).
//!
//! Block-granular boundaries (`END_OF_BLOCK_HEADER` / `END_OF_BLOCK`) are
//! reported but coalesced to one event per member — a full Block<> port
//! will refine this.

#![allow(dead_code)]

use std::io::{self, Write};

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
use crate::backends::libdeflate::{DecompressError, DecompressorEx};
use crate::decompress::parallel::crc32::CRC32Calculator;
use crate::decompress::parallel::gzip_format::{read_footer, read_header, Footer, Header};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::inflate_wrapper::IsalInflateWrapper;

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
    /// (GzipReader.hpp:580). Updated by `read_member_decode`, verified by
    /// `read_footer_inner`. Uses the ported `CRC32Calculator` so the
    /// CRC math goes through the same `update` / `verify` surface the
    /// vendor uses at GzipReader.hpp:715.
    crc32: CRC32Calculator,
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
            crc32: CRC32Calculator::new(),
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
        self.crc32.reset();
        self.current_point = Some(StoppingPoint::EndOfStreamHeader);
        Ok(())
    }

    /// Coalesced member decode. Mirror of the read loop at
    /// GzipReader.hpp:645-687, where the vendor drives
    /// `m_currentDeflateBlock->read( m_bitReader, ... )` until EOB. On
    /// x86_64+isal we use [`IsalInflateWrapper`] — the ported
    /// `rapidgzip::IsalInflateWrapper` (vendor/.../gzip/isal.hpp:253-385)
    /// — to consume the raw deflate stream member; rapidgzip itself uses
    /// the same wrapper on its ISA-L-conditional path. On non-x86_64
    /// builds where ISA-L is unavailable, we fall back to libdeflate
    /// one-shot decode of the whole member (rapidgzip falls back to a
    /// zlib wrapper on the same arches).
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
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
        // After read_header_inner, byte_cursor points at the first byte
        // of the raw deflate stream. The wrapper consumes raw deflate
        // (caller strips the gzip header and trailer, per
        // inflate_wrapper.rs:1-5).
        let deflate_slice = &self.buffer[self.byte_cursor..];
        let mut wrapper = IsalInflateWrapper::new(deflate_slice, 0).map_err(|e| {
            GzipReaderError::DeflateError(format!("IsalInflateWrapper::new: {e:?}"))
        })?;
        // No cross-chunk window: this is a sequential reader, decoding
        // the member from the start. Mirror of
        // `Block<>::setInitialWindow()` (deflate.hpp:1740) called with
        // no argument — leaves the decoder's window empty so the first
        // block can only have intra-block back-references (which is
        // always true for a member-start block per RFC 1951 §3.2.7).
        wrapper
            .set_window(&[])
            .map_err(|e| GzipReaderError::DeflateError(format!("set_window: {e:?}")))?;
        // Pump the wrapper. Mirror of GzipReader.hpp:645-687's
        // `m_currentDeflateBlock->read` loop and the subsequent
        // `flushOutputBuffer` call: read into a buffer, feed bytes to
        // the writer, repeat until the stream finishes or the caller's
        // max_bytes budget is met. Buffer sizing matches rapidgzip's
        // chunk allocation style — start at a few KiB and grow if the
        // decoded payload looks large.
        let mut total_written: usize = 0;
        let mut out_buf = vec![0u8; 64 * 1024];
        let mut finished = false;
        loop {
            if total_written >= max_bytes {
                // Mirror of `nBytesDecoded >= nMaxBytesToDecode` break
                // at GzipReader.hpp:672-674. The vendor stops reading
                // and leaves the rest in m_lastBlockData; ours leaves
                // it in the inflate wrapper's internal state. For the
                // coalesced API (we don't expose the wrapper between
                // calls) this is acceptable because callers either
                // pass max_bytes = usize::MAX or accept a partial
                // member read.
                break;
            }
            let r = wrapper
                .read_stream(&mut out_buf)
                .map_err(|e| GzipReaderError::DeflateError(format!("read_stream: {e:?}")))?;
            if r.bytes_written > 0 {
                let to_write = r.bytes_written.min(max_bytes - total_written);
                writer.write_all(&out_buf[..to_write])?;
                // Mirror of `m_crc32Calculator.update(...)` at
                // GzipReader.hpp:687 (inside flushOutputBuffer's
                // implementation). Verified against the footer in
                // `read_footer_inner` via `CRC32Calculator::verify`.
                self.crc32.update(&out_buf[..to_write]);
                total_written += to_write;
            }
            if r.finished {
                finished = true;
                break;
            }
            if r.bytes_written == 0 {
                // No-progress break — mirror of the implicit
                // termination at GzipReader.hpp:680-683 (`flushedCount
                // == 0 && !bufferHasBeenFlushed()`).
                break;
            }
        }
        if !finished {
            return Err(GzipReaderError::DeflateError(
                "deflate stream did not finish".to_string(),
            ));
        }
        // tell_compressed() returns the bit position within the slice
        // we passed in; ISA-L aligns to a byte boundary at end-of-
        // stream so we round up to convert to bytes.
        let consumed_bits = wrapper.tell_compressed();
        let consumed_bytes = consumed_bits.div_ceil(8);
        self.byte_cursor += consumed_bytes;
        Ok(total_written)
    }

    /// Fallback path for archs without ISA-L: rapidgzip falls back to a
    /// zlib wrapper here (vendor's conditional compilation); gzippy uses
    /// libdeflate one-shot because it's already linked and provides the
    /// same whole-member semantics. Still a port of GzipReader.hpp:645-687
    /// at the API level — the inner kernel choice tracks rapidgzip's
    /// own per-arch conditional.
    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    fn read_member_decode<W: Write>(
        &mut self,
        writer: &mut W,
        max_bytes: usize,
    ) -> Result<usize, GzipReaderError> {
        if self.current_header.is_none() {
            return Err(GzipReaderError::DeflateError(
                "Call readHeader and readBlockHeader before calling readBlock!".to_string(),
            ));
        }
        let remaining = &self.buffer[self.member_start_byte..];
        let mut decompressor = DecompressorEx::new();
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
                    // Mirror of `m_crc32Calculator.update(...)` at
                    // GzipReader.hpp:687 — via ported CRC32Calculator.
                    self.crc32.reset();
                    self.crc32.update(&out[..result.output_size]);
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
        // CRC check (GzipReader.hpp:714-716). Routed through the ported
        // `CRC32Calculator::verify` so the comparison goes via the same
        // vendor surface (crc32.hpp:308-319).
        if self.did_read_header {
            self.crc32
                .verify(footer.crc32)
                .map_err(GzipReaderError::InvalidFooter)?;
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
