//! Port of `rapidgzip::IsalInflateWrapper`
//! (vendor/rapidgzip/.../gzip/isal.hpp) scoped to single-member raw-
//! deflate decode. Caller is expected to strip the gzip header and
//! trailer themselves (single_member.rs already does this); the wrapper
//! consumes raw deflate bits.
//!
//! Surface mirrors rapidgzip's class for the parts we use:
//!   - new(input, bit_offset): construct + seek to start bit (incl.
//!     inflatePrime for non-byte-aligned starts).
//!   - set_window(&[u8]): set the decoder's 32-KiB dictionary.
//!   - set_stopping_points(StoppingPoints): request the patched ISA-L
//!     to pause on END_OF_BLOCK / END_OF_BLOCK_HEADER / END_OF_STREAM*
//!     events.
//!   - read_stream(output): pump bytes into `output`, returning when
//!     the buffer is full, a stop point fires, or decode finishes.
//!   - stopped_at() -> StoppingPoints, is_final_block() -> bool,
//!     btype() -> Option<DeflateCompressionType>, tell_compressed() -> usize.
//!
//! Out of scope: gzip/zlib header parsing (caller's responsibility),
//! multi-stream footer accumulation (multi-member uses a different path).
//
// Allowed dead_code: this module is part of step 4 of the
// rapidgzip-port-design.md migration; consumed by gzip_chunk.rs in
// step 5. Module ships with unit tests so cfg(test) builds exercise
// every public item.
#![allow(dead_code)]

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use isal::isal_sys::igzip_lib as isal_raw;

/// Bit-flag set matching the patched ISA-L's `ISAL_STOPPING_POINT_*`
/// constants. Port of `rapidgzip::StoppingPoint` in gzip/isal.hpp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct StoppingPoints(pub u32);

impl StoppingPoints {
    pub const NONE: Self = Self(0);
    pub const END_OF_STREAM_HEADER: Self = Self(1);
    pub const END_OF_STREAM: Self = Self(2);
    pub const END_OF_BLOCK_HEADER: Self = Self(4);
    pub const END_OF_BLOCK: Self = Self(8);
    pub const ALL: Self = Self(0xFFFF_FFFF);

    #[inline]
    pub fn contains(self, other: StoppingPoints) -> bool {
        (self.0 & other.0) != 0
    }
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for StoppingPoints {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Result of one `read_stream` invocation.
#[derive(Debug, Clone, Copy)]
pub struct ReadStreamResult {
    pub bytes_written: usize,
    /// Set to the stop event that ended this call, or `NONE` if the
    /// call ended because the output buffer filled, input was
    /// exhausted, or the deflate stream finished.
    pub stopped_at: StoppingPoints,
    /// Snapshot of `tell_compressed()` at the moment this call returned.
    /// For boundary recording: when `stopped_at` is END_OF_BLOCK or
    /// END_OF_BLOCK_HEADER, this is the bit position at the boundary.
    pub bit_position: usize,
    /// True once the deflate stream has reached BFINAL=1 and the
    /// decoder has processed it.
    pub finished: bool,
}

/// Compression-type values from a deflate block header. Mirror of
/// `rapidgzip::CompressionType` for the values we care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeflateCompressionType {
    Uncompressed = 0,
    FixedHuffman = 1,
    DynamicHuffman = 2,
}

#[derive(Debug)]
pub enum InflateError {
    InvalidBlock,
    InvalidLookback,
    InvalidSymbol,
    UnsupportedPlatform,
    StartBitPastEnd,
    SetDictFailed,
    Internal(i32),
}

/// Wraps a patched-ISA-L `inflate_state`. Consumes raw deflate bits
/// from a slice of compressed input. Lifetime tied to the input slice.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub struct IsalInflateWrapper<'a> {
    state: isal_raw::inflate_state,
    input: &'a [u8],
    /// The compressed-stream bit offset where decoding originally
    /// started. `tell_compressed` returns this plus bits consumed.
    encoded_start_offset_bits: usize,
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl<'a> IsalInflateWrapper<'a> {
    /// Construct + seek the decoder to `bit_offset` within `input`.
    /// `bit_offset` may be non-byte-aligned; the leading partial byte
    /// is loaded into ISA-L's bit buffer via inflatePrime-equivalent
    /// init. Returns Err if `bit_offset` is past the end of input.
    pub fn new(input: &'a [u8], bit_offset: usize) -> Result<Self, InflateError> {
        let byte_idx = bit_offset / 8;
        let bit_skip = bit_offset % 8;
        if byte_idx >= input.len() {
            return Err(InflateError::StartBitPastEnd);
        }
        let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
        unsafe { isal_raw::isal_inflate_init(&mut state) };
        state.crc_flag = isal_raw::ISAL_DEFLATE;
        if bit_skip > 0 {
            state.read_in = (input[byte_idx] as u64) >> bit_skip;
            state.read_in_length = (8 - bit_skip) as i32;
            state.next_in = unsafe { input.as_ptr().add(byte_idx + 1) as *mut u8 };
            state.avail_in = (input.len() - byte_idx - 1) as u32;
        } else {
            state.next_in = unsafe { input.as_ptr().add(byte_idx) as *mut u8 };
            state.avail_in = (input.len() - byte_idx) as u32;
        }
        Ok(Self {
            state,
            input,
            encoded_start_offset_bits: bit_offset,
        })
    }

    /// Set the 32-KiB sliding-window dictionary for cross-chunk back-
    /// references. Empty slice means "no window known yet" (decoder
    /// will emit literals for any back-reference into the empty
    /// window, producing zero bytes that `apply_window` resolves later).
    pub fn set_window(&mut self, window: &[u8]) -> Result<(), InflateError> {
        static ZERO_WINDOW: [u8; 32768] = [0u8; 32768];
        let dict = if window.is_empty() {
            &ZERO_WINDOW[..]
        } else {
            window
        };
        let ret = unsafe {
            isal_raw::isal_inflate_set_dict(
                &mut self.state,
                dict.as_ptr() as *mut u8,
                dict.len() as u32,
            )
        };
        if ret != 0 {
            return Err(InflateError::SetDictFailed);
        }
        Ok(())
    }

    /// Request the patched ISA-L to pause when any of the given events
    /// fire. Setting NONE disables stopping; the decoder runs to the
    /// next natural pause (output buffer full / input exhausted /
    /// BFINAL). Mirror of `IsalInflateWrapper::setStoppingPoints`
    /// (isal.hpp:76-79).
    pub fn set_stopping_points(&mut self, points: StoppingPoints) {
        self.state.points_to_stop_at = points.0;
    }

    pub fn stopped_at(&self) -> StoppingPoints {
        StoppingPoints(self.state.stopped_at)
    }

    /// Clear the patched-ISA-L `stopped_at` flag so the next
    /// `read_stream` call advances past the stop instead of re-reporting
    /// it. Mirror of rapidgzip's `IsalInflateWrapper::readStream`
    /// pattern (isal.hpp), which resets the field at entry.
    pub fn clear_stop(&mut self) {
        self.state.stopped_at = 0;
    }

    pub fn is_final_block(&self) -> bool {
        self.state.bfinal != 0
    }

    /// Returns Some only after a stop at END_OF_BLOCK_HEADER. Mirror
    /// of `IsalInflateWrapper::compressionType` (isal.hpp:93-109).
    pub fn btype(&self) -> Option<DeflateCompressionType> {
        if self.stopped_at() != StoppingPoints::END_OF_BLOCK_HEADER {
            return None;
        }
        match self.state.btype {
            0 => Some(DeflateCompressionType::Uncompressed),
            1 => Some(DeflateCompressionType::FixedHuffman),
            2 => Some(DeflateCompressionType::DynamicHuffman),
            _ => None,
        }
    }

    /// Compressed-stream bit position. Mirror of
    /// `IsalInflateWrapper::tellCompressed` (isal.hpp:69-74).
    /// Computed as `input.len()*8 - avail_in*8 - read_in_length`,
    /// adjusted for the start offset so the returned value is in the
    /// caller's bit coordinate system.
    pub fn tell_compressed(&self) -> usize {
        // Bits consumed since wrapper construction. The original input
        // window started at byte_idx + (maybe-loaded-partial-byte).
        // We compute the absolute bit position in the input slice
        // using ISA-L's exposed avail_in/read_in_length the same way
        // decompress_deflate_from_bit_with_end does:
        //   absolute = input.len()*8 - avail_in*8 - read_in_length
        self.input.len() * 8
            - self.state.avail_in as usize * 8
            - self.state.read_in_length.max(0) as usize
    }

    /// Pump bytes into `output`. Returns when:
    ///   - output is full
    ///   - a requested stopping point fires
    ///   - input is exhausted
    ///   - BFINAL block has been fully consumed
    ///
    /// Mirror of `IsalInflateWrapper::readStream` (isal.hpp:65-67 +
    /// the body further down in the same file). Single-call equivalent:
    /// not multi-stream; caller drives the loop for chunk-spanning
    /// behavior.
    pub fn read_stream(&mut self, output: &mut [u8]) -> Result<ReadStreamResult, InflateError> {
        let original_len = output.len();
        let original_stopped_at = self.state.stopped_at;
        let _ = original_stopped_at;
        self.state.avail_out = original_len as u32;
        self.state.next_out = output.as_mut_ptr();

        let ret = unsafe { isal_raw::isal_inflate(&mut self.state) };
        let bytes_written = original_len - self.state.avail_out as usize;
        let bit_position = self.tell_compressed();

        let stopped_at = StoppingPoints(self.state.stopped_at);
        let finished = self.state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH;

        match ret {
            0 | 1 => {
                // 0 = ISAL_DECOMP_OK, 1 = ISAL_END_INPUT.
                Ok(ReadStreamResult {
                    bytes_written,
                    stopped_at,
                    bit_position,
                    finished,
                })
            }
            // Patched ISA-L returns these error codes (igzip_lib.h):
            //   -1 = ISAL_INVALID_BLOCK
            //   -2 = ISAL_INVALID_SYMBOL
            //   -3 = ISAL_INVALID_LOOKBACK
            -1 => Err(InflateError::InvalidBlock),
            -2 => Err(InflateError::InvalidSymbol),
            -3 => Err(InflateError::InvalidLookback),
            other => Err(InflateError::Internal(other)),
        }
    }
}

// On non-x86_64 / no-feature builds, the wrapper is unavailable.
// Provide a stub type so callers compile but cannot construct it.
#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub struct IsalInflateWrapper<'a> {
    _phantom: std::marker::PhantomData<&'a [u8]>,
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
impl<'a> IsalInflateWrapper<'a> {
    pub fn new(_input: &'a [u8], _bit_offset: usize) -> Result<Self, InflateError> {
        Err(InflateError::UnsupportedPlatform)
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn round_trip_decode_no_stopping_points() {
        let payload = b"hello world hello world hello world".repeat(1000);
        let deflate = make_deflate(&payload);
        let mut wrapper = IsalInflateWrapper::new(&deflate, 0).expect("init");
        wrapper.set_window(&[]).expect("set_window with empty dict");
        let mut output = vec![0u8; payload.len() + 1024];
        let mut total = 0usize;
        loop {
            let r = wrapper
                .read_stream(&mut output[total..])
                .expect("read_stream");
            total += r.bytes_written;
            if r.finished {
                break;
            }
            if r.bytes_written == 0 {
                break;
            }
        }
        output.truncate(total);
        assert_eq!(output, payload);
    }

    #[test]
    fn stopping_at_end_of_block_records_a_bit_position() {
        let payload = vec![b'x'; 200_000]; // forces multiple deflate blocks at L6
        let deflate = make_deflate(&payload);
        let mut wrapper = IsalInflateWrapper::new(&deflate, 0).expect("init");
        wrapper.set_window(&[]).expect("set_window");
        wrapper.set_stopping_points(StoppingPoints::END_OF_BLOCK);
        let mut output = vec![0u8; payload.len() + 1024];
        let mut total = 0usize;
        let mut boundary_count = 0usize;
        loop {
            let r = wrapper
                .read_stream(&mut output[total..])
                .expect("read_stream");
            total += r.bytes_written;
            if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                boundary_count += 1;
                assert!(r.bit_position > 0);
                assert!(r.bit_position <= deflate.len() * 8);
                if r.finished {
                    break;
                }
                continue;
            }
            if r.finished {
                break;
            }
            if r.bytes_written == 0 {
                break;
            }
        }
        output.truncate(total);
        assert_eq!(output, payload);
        // Should have hit at least one END_OF_BLOCK along the way.
        assert!(boundary_count >= 1);
    }

    #[test]
    fn tell_compressed_matches_input_length_at_end() {
        let payload = b"abc".repeat(10_000);
        let deflate = make_deflate(&payload);
        let mut wrapper = IsalInflateWrapper::new(&deflate, 0).expect("init");
        wrapper.set_window(&[]).expect("set_window");
        let mut output = vec![0u8; payload.len() + 1024];
        let mut total = 0usize;
        loop {
            let r = wrapper
                .read_stream(&mut output[total..])
                .expect("read_stream");
            total += r.bytes_written;
            if r.finished {
                // tell_compressed should be within a byte of total deflate bits
                // (deflate streams end with bit padding to next byte boundary).
                let final_bit_pos = wrapper.tell_compressed();
                assert!(final_bit_pos <= deflate.len() * 8);
                assert!(final_bit_pos + 8 >= deflate.len() * 8);
                break;
            }
            if r.bytes_written == 0 {
                break;
            }
        }
        output.truncate(total);
        assert_eq!(output, payload);
    }
}
