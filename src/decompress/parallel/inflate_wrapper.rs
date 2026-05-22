//! Port of `rapidgzip::IsalInflateWrapper`
//! (vendor/rapidgzip/.../gzip/isal.hpp) scoped to single-member raw-
//! deflate decode. Caller is expected to strip the gzip header and
//! trailer themselves (single_member.rs already does this); the wrapper
//! consumes raw deflate bits.
//!
//! Surface mirrors rapidgzip's class for the parts we use:
//!   - new(input, bit_offset): construct + seek to start bit.
//!   - with_until_bits(input, bit_offset, until_bits): mirror of
//!     vendor's `IsalInflateWrapper(BitReader, untilOffset)` ctor at
//!     `gzip/isal.hpp:32-44`. The `until_bits` value caps how far the
//!     refillBuffer step is allowed to advance the bit reader — i.e.
//!     ISA-L physically cannot read input past this bit position.
//!   - set_window(&[u8]): set the decoder's 32-KiB dictionary.
//!   - set_stopping_points(StoppingPoints): request the patched ISA-L
//!     to pause on END_OF_BLOCK / END_OF_BLOCK_HEADER / END_OF_STREAM*.
//!   - read_stream(output): pump bytes into `output`, returning when
//!     the buffer is full, a stop point fires, or decode finishes.
//!   - stopped_at(), is_final_block(), btype(), tell_compressed().
//!
//! Class shape (matches vendor isal.hpp:198-212 field-for-field where
//! Rust permits):
//!   - `input` is the underlying bytes (vendor stores a `BitReader`,
//!     which is conceptually a `(slice, bit_cursor)` pair).
//!   - `bit_reader_tell` is the vendor `m_bitReader.tell()` value.
//!   - `encoded_start_offset_bits`, `encoded_until_bits` mirror
//!     `m_encodedStartOffset`, `m_encodedUntilOffset`.
//!   - `buffer` is the 128 KiB staging area `m_buffer` at isal.hpp:207.
//!     Vendor's comment there: "Loading the whole encoded data
//!     (multiple MiB) into memory first and then decoding it in one go
//!     is 4x slower than processing it in chunks of 128 KiB!"
//!   - `refill_buffer()` is the literal port of `refillBuffer()` at
//!     isal.hpp:228-250.
//
// Allowed dead_code: this module is part of step 4 of the
// rapidgzip-port-design.md migration; consumed by gzip_chunk.rs in
// step 5. Module ships with unit tests so cfg(test) builds exercise
// every public item.
#![allow(dead_code)]

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use isal::isal_sys::igzip_lib as isal_raw;

/// Vendor `m_buffer` capacity at `gzip/isal.hpp:207` (`std::array<char, 128_Ki>`).
const STAGING_BUFFER_BYTES: usize = 128 * 1024;

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
///
/// **Class shape mirrors vendor `IsalInflateWrapper` (isal.hpp:198-212):**
/// the wrapper owns a 128 KiB staging buffer that ISA-L sees via
/// `next_in`/`avail_in`; the bit reader (`input` + `bit_reader_tell`)
/// is the canonical "where in the input are we" cursor, and
/// `refill_buffer` is what bridges the two.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub struct IsalInflateWrapper<'a> {
    state: isal_raw::inflate_state,
    /// Underlying compressed input. Vendor's `m_bitReader` wraps the
    /// equivalent file/slice; gzippy holds the slice directly +
    /// `bit_reader_tell` as the bit cursor.
    input: &'a [u8],
    /// Vendor `m_encodedStartOffset` (isal.hpp:200).
    encoded_start_offset_bits: usize,
    /// Vendor `m_encodedUntilOffset` (isal.hpp:201). The cap that
    /// `refill_buffer` will not advance past.
    encoded_until_bits: usize,
    /// Vendor `m_bitReader.tell()`. Advances ONLY in `refill_buffer`,
    /// matching vendor's pattern at isal.hpp:228-250 where every
    /// bit-reader read happens inside the refill.
    bit_reader_tell: usize,
    /// Vendor `m_buffer` (isal.hpp:207): `std::array<char, 128_Ki>`
    /// stored INLINE in the wrapper struct. We mirror that exactly:
    /// no heap allocation per wrapper construction, no extra allocator
    /// frame in the flamegraph that vendor doesn't have.
    ///
    /// The struct is ~128 KiB. Workers construct one per chunk, hold
    /// it on the stack inside `decode_chunk_isal_inexact` etc. — that
    /// is vendor's pattern (the `IsalInflateWrapper` object lives in
    /// `decodeChunkWithInflateWrapper`'s stack frame at GzipChunk.hpp:206).
    /// If a future call site has a frame-size problem, that caller
    /// can `Box::new(IsalInflateWrapper::with_until_bits(...))`; the
    /// wrapper itself stays inline.
    buffer: [u8; STAGING_BUFFER_BYTES],
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl<'a> IsalInflateWrapper<'a> {
    /// Convenience constructor: `until_bits = input.len() * 8` (no cap
    /// beyond the slice end). Existing call sites that don't know a
    /// real chunk-end offset use this.
    pub fn new(input: &'a [u8], bit_offset: usize) -> Result<Self, InflateError> {
        Self::with_until_bits(input, bit_offset, input.len() * 8)
    }

    /// Construct + seek the decoder to `bit_offset`, capping the reader
    /// at `until_bits`. Mirror of vendor constructor
    /// `IsalInflateWrapper(BitReader&&, untilOffset)` at
    /// `gzip/isal.hpp:32-44`. The cap is enforced inside `refill_buffer`
    /// — the wrapper physically cannot read input bytes past
    /// `byte_floor(until_bits / 8)`. Returns Err if `bit_offset` is
    /// past the end of input.
    pub fn with_until_bits(
        input: &'a [u8],
        bit_offset: usize,
        until_bits: usize,
    ) -> Result<Self, InflateError> {
        let byte_idx = bit_offset / 8;
        if byte_idx >= input.len() {
            return Err(InflateError::StartBitPastEnd);
        }
        // Vendor caps `m_encodedUntilOffset` at `min(untilOffset, file_size_in_bits)`
        // (isal.hpp:37-41); mirror that exactly.
        let capped_until = until_bits.min(input.len() * 8);
        let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
        // Vendor `initStream()` (isal.hpp:215-225): isal_inflate_init,
        // crc_flag = ISAL_DEFLATE, next_in/avail_in/read_in/read_in_length
        // = 0. No input is staged here — refill_buffer does that on the
        // first read_stream iteration.
        unsafe { isal_raw::isal_inflate_init(&mut state) };
        state.crc_flag = isal_raw::ISAL_DEFLATE;
        state.next_in = std::ptr::null_mut();
        state.avail_in = 0;
        state.read_in = 0;
        state.read_in_length = 0;
        // Prime the partial leading byte before any `set_dict` call.
        // Matches `isal_decompress::decompress_deflate_from_bit` and vendor
        // `refillBuffer` on first refill (isal.hpp:235-239) — must happen
        // before `isal_inflate_set_dict` or cross-chunk resume at non-byte-
        // aligned boundaries returns ISAL_INVALID_BLOCK.
        let tell_mod = bit_offset % 8;
        let mut bit_reader_tell = bit_offset;
        if tell_mod != 0 {
            let bit_skip = tell_mod;
            let byte = input[byte_idx];
            state.read_in = (byte as u64) >> bit_skip;
            state.read_in_length = (8 - bit_skip) as i32;
            bit_reader_tell = bit_offset + (8 - bit_skip);
        }
        // Inline buffer: matches vendor's `std::array<char, 128_Ki>`
        // stack-stored layout (isal.hpp:207). ~128 KiB stack frame per
        // wrapper, same as vendor. `refill_buffer` overwrites these
        // bytes before ISA-L reads them.
        Ok(Self {
            state,
            input,
            encoded_start_offset_bits: bit_offset,
            encoded_until_bits: capped_until,
            bit_reader_tell,
            buffer: [0u8; STAGING_BUFFER_BYTES],
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
    /// fire. Setting NONE disables stopping. Mirror of
    /// `IsalInflateWrapper::setStoppingPoints` (isal.hpp:76-79).
    pub fn set_stopping_points(&mut self, points: StoppingPoints) {
        self.state.points_to_stop_at = points.0;
    }

    pub fn stopped_at(&self) -> StoppingPoints {
        StoppingPoints(self.state.stopped_at)
    }

    /// Clear `stopped_at` so the next `read_stream` advances past the
    /// stop instead of re-reporting it.
    pub fn clear_stop(&mut self) {
        self.state.stopped_at = 0;
    }

    pub fn debug_points_to_stop_at(&self) -> u32 {
        self.state.points_to_stop_at
    }
    pub fn debug_stopped_at_raw(&self) -> u32 {
        self.state.stopped_at
    }
    pub fn debug_tmp_out_stopped_at(&self) -> u32 {
        self.state.tmp_out_stopped_at
    }
    pub fn debug_block_state(&self) -> u32 {
        self.state.block_state
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
    /// `IsalInflateWrapper::tellCompressed` (isal.hpp:69-74):
    ///
    ///     m_bitReader.tell() - m_stream.avail_in * BYTE_SIZE
    ///                        - m_stream.read_in_length
    pub fn tell_compressed(&self) -> usize {
        self.bit_reader_tell
            .saturating_sub(self.state.avail_in as usize * 8)
            .saturating_sub(self.state.read_in_length.max(0) as usize)
    }

    /// Literal port of `IsalInflateWrapper::refillBuffer`
    /// (vendor/.../gzip/isal.hpp:228-250). Stages up to 128 KiB of
    /// input bytes into `self.buffer` so ISA-L's `isal_inflate` has
    /// `next_in`/`avail_in` to consume on the next call. Caps strictly
    /// at `encoded_until_bits` — once `bit_reader_tell` reaches the
    /// cap, this is a no-op and ISA-L will eventually return because
    /// it has nothing more to read.
    fn refill_buffer(&mut self) {
        // Vendor isal.hpp:231 — guard.
        if self.state.avail_in > 0 || self.bit_reader_tell >= self.encoded_until_bits {
            return;
        }

        // Vendor isal.hpp:235-239 — first refill when start is not
        // byte-aligned. Prime the partial leading byte into ISA-L's
        // bit buffer (read_in / read_in_length), advance bit reader to
        // next byte boundary, fall through to byte-aligned load.
        let tell_mod = self.bit_reader_tell % 8;
        if tell_mod != 0 {
            // Non-byte-aligned tell after construction should only happen
            // when `with_until_bits` did not prime (legacy `new()` path).
            let n_bits_to_prime = (8 - tell_mod) as u32;
            let byte_idx = self.bit_reader_tell / 8;
            debug_assert!(byte_idx < self.input.len());
            let byte = self.input[byte_idx];
            let value = (byte as u64) >> tell_mod;
            self.state.read_in |= value << self.state.read_in_length;
            self.state.read_in_length += n_bits_to_prime as i32;
            self.bit_reader_tell += n_bits_to_prime as usize;
            debug_assert!(self.bit_reader_tell.is_multiple_of(8));
        }

        // Vendor isal.hpp:240-244 — last refill where < 8 bits remain.
        let remaining_bits = self.encoded_until_bits - self.bit_reader_tell;
        if remaining_bits < 8 {
            if remaining_bits > 0 {
                let byte_idx = self.bit_reader_tell / 8;
                debug_assert!(byte_idx < self.input.len());
                let byte = self.input[byte_idx];
                let mask = (1u64 << remaining_bits) - 1;
                let value = (byte as u64) & mask;
                self.state.read_in |= value << self.state.read_in_length;
                self.state.read_in_length += remaining_bits as i32;
                self.bit_reader_tell += remaining_bits;
            }
            return;
        }

        // Vendor isal.hpp:246-249 — byte-aligned bulk load.
        let byte_idx = self.bit_reader_tell / 8;
        let max_bytes_by_cap = (self.encoded_until_bits - self.bit_reader_tell) / 8;
        let max_bytes_by_input = self.input.len().saturating_sub(byte_idx);
        let n = max_bytes_by_cap
            .min(max_bytes_by_input)
            .min(self.buffer.len());
        if n == 0 {
            return;
        }
        self.buffer[..n].copy_from_slice(&self.input[byte_idx..byte_idx + n]);
        self.state.next_in = self.buffer.as_mut_ptr();
        self.state.avail_in = n as u32;
        self.bit_reader_tell += n * 8;
    }

    /// Read the 8-byte gzip footer at the current decoder position
    /// (assumed to be at end-of-stream, byte-aligned). Advances the
    /// decoder's cursor past the footer. Returns `(crc32, isize)`.
    ///
    /// Vendor equivalent: `IsalInflateWrapper::readGzipFooter`
    /// (gzip/isal.hpp:429-450). Vendor uses `readBytes<8>()`
    /// (isal.hpp:388-426) which pulls from the bit buffer, then from
    /// the staging buffer, refilling as needed.
    pub fn read_footer_at_current(&mut self) -> Result<(u32, u32), InflateError> {
        let mut footer = [0u8; 8];
        self.read_bytes(&mut footer)?;
        let crc32 = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let isize_field = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
        Ok((crc32, isize_field))
    }

    /// Pull `out.len()` bytes from the wrapper, draining the ISA-L bit
    /// buffer first, then the staging buffer, refilling as needed.
    /// Mirror of vendor `readBytes<SIZE>()` (isal.hpp:388-426).
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<(), InflateError> {
        // Vendor isal.hpp:392-394 — align the bit buffer to a byte boundary
        // (`read_in_length %= BYTE_SIZE` semantics: drop sub-byte remainder).
        let remaining_bits = (self.state.read_in_length % 8).max(0) as u32;
        if remaining_bits > 0 {
            self.state.read_in >>= remaining_bits;
            self.state.read_in_length -= remaining_bits as i32;
        }

        let mut written = 0usize;
        while written < out.len() {
            // Vendor isal.hpp:399-406 — drain whole bytes from the
            // bit buffer first.
            if self.state.read_in_length >= 8 {
                out[written] = (self.state.read_in & 0xFF) as u8;
                self.state.read_in >>= 8;
                self.state.read_in_length -= 8;
                written += 1;
                continue;
            }

            // Vendor isal.hpp:407-411 — then drain avail_in.
            let need = out.len() - written;
            if self.state.avail_in as usize >= need {
                // Safety: next_in is either null (avail_in==0, handled
                // above) or points into our staging buffer with at
                // least `avail_in` valid bytes.
                let src =
                    unsafe { std::slice::from_raw_parts(self.state.next_in as *const u8, need) };
                out[written..written + need].copy_from_slice(src);
                self.state.avail_in -= need as u32;
                unsafe {
                    self.state.next_in = self.state.next_in.add(need);
                }
                written += need;
                continue;
            }

            // Vendor isal.hpp:412-421 — partial drain + refill.
            if self.state.avail_in > 0 {
                let n = self.state.avail_in as usize;
                let src = unsafe { std::slice::from_raw_parts(self.state.next_in as *const u8, n) };
                out[written..written + n].copy_from_slice(src);
                written += n;
                self.state.avail_in = 0;
            }
            self.refill_buffer();
            if self.state.avail_in == 0 && self.state.read_in_length < 8 {
                // Vendor throws EndOfFileReached; map to internal error.
                return Err(InflateError::Internal(-1));
            }
        }
        Ok(())
    }

    /// Reset the wrapper's internal state to begin decoding a fresh
    /// gzip stream from the current input position. Mirror of
    /// `IsalInflateWrapper::setNeedToReadHeader` + subsequent
    /// `readHeader` paths. Preserves `bit_reader_tell` and any pending
    /// bit-buffer state; only resets the ISA-L decode state.
    pub fn reset_for_next_stream(&mut self) {
        // Snapshot the bit-buffer and input-staging state since
        // `isal_inflate_init` zeroes the whole struct.
        let read_in = self.state.read_in;
        let read_in_length = self.state.read_in_length;
        let next_in = self.state.next_in;
        let avail_in = self.state.avail_in;
        unsafe { isal_raw::isal_inflate_init(&mut self.state) };
        self.state.crc_flag = isal_raw::ISAL_DEFLATE;
        self.state.points_to_stop_at = 0;
        self.state.stopped_at = 0;
        self.state.read_in = read_in;
        self.state.read_in_length = read_in_length;
        self.state.next_in = next_in;
        self.state.avail_in = avail_in;
    }

    /// Bytes remaining in the input from the decoder's current
    /// position. Used by the multi-stream loop to parse the next
    /// gzip header. After END_OF_STREAM, ISA-L is byte-aligned and
    /// `tell_compressed()` is the position right after the deflate
    /// body — i.e. the start of the footer.
    pub fn remaining_input(&self) -> &'a [u8] {
        let byte_pos = self.tell_compressed() / 8;
        if byte_pos >= self.input.len() {
            return &[];
        }
        &self.input[byte_pos..]
    }

    /// Advance the decoder's input cursor by `n` bytes. Used by the
    /// multi-stream loop after `gzip_format::read_header` parses the
    /// next gzip header — the wrapper's cursor must then skip past
    /// the parsed header bytes so a subsequent `read_stream` resumes
    /// at the new deflate body.
    ///
    /// Updates `bit_reader_tell` directly + clears the ISA-L
    /// avail_in/read_in_length since they're stale w.r.t. the new
    /// position. The next `read_stream` call's first `refill_buffer`
    /// will repopulate from the input slice at the new cursor.
    pub fn advance_input(&mut self, n: usize) {
        let bits = n.saturating_mul(8);
        // Effective compressed position right now:
        let cur = self.tell_compressed();
        let target = (cur + bits).min(self.encoded_until_bits);
        // Discard whatever staged input we have; refill_buffer will
        // re-load from `target` on demand.
        self.state.avail_in = 0;
        self.state.read_in = 0;
        self.state.read_in_length = 0;
        self.state.next_in = std::ptr::null_mut();
        self.bit_reader_tell = target;
        // Allow advance_input to grow the cap when the caller pushes
        // past the previously-known until offset. This matches vendor
        // multi-stream behavior: the next-stream header isn't part of
        // the just-finished stream's `untilOffset`.
        if self.bit_reader_tell > self.encoded_until_bits {
            self.encoded_until_bits = self.input.len() * 8;
        }
    }

    /// Returns true iff the current decode has hit a stream end
    /// (END_OF_STREAM) and not yet been advanced.
    pub fn at_end_of_stream(&self) -> bool {
        self.stopped_at() == StoppingPoints::END_OF_STREAM
    }

    /// The currently-effective cap. Updated only by callers via
    /// `set_encoded_until_bits` or by `advance_input` widening past
    /// the original cap. Exposed for tests and diagnostics; mirrors
    /// vendor `m_encodedUntilOffset` field at isal.hpp:201.
    pub fn encoded_until_bits(&self) -> usize {
        self.encoded_until_bits
    }

    /// Pump bytes into `output` until any of:
    ///   - output buffer is full
    ///   - a requested stopping point fires
    ///   - input is exhausted (refill returns nothing AND avail_in == 0)
    ///   - BFINAL block has been fully consumed
    ///
    /// Mirror of `rapidgzip::IsalInflateWrapper::readStream`
    /// (vendor/.../gzip/isal.hpp:253-385). The inner loop calls
    /// `refill_buffer()` at the top to mirror the vendor pattern at
    /// isal.hpp:277, AND it resets `state.stopped_at` at entry
    /// (isal.hpp:261).
    pub fn read_stream(&mut self, output: &mut [u8]) -> Result<ReadStreamResult, InflateError> {
        let original_len = output.len();
        self.state.next_out = output.as_mut_ptr();
        self.state.avail_out = original_len as u32;
        self.state.stopped_at = 0;

        let mut finished;
        loop {
            // Mirror of vendor refillBuffer() call at isal.hpp:277.
            self.refill_buffer();

            let prev_avail_in = self.state.avail_in;
            let prev_read_in_length = self.state.read_in_length;
            let prev_avail_out = self.state.avail_out;

            let ret = unsafe { isal_raw::isal_inflate(&mut self.state) };

            finished = self.state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH;

            match ret {
                0 | 1 => {}
                -1 => return Err(InflateError::InvalidBlock),
                -2 => return Err(InflateError::InvalidSymbol),
                -3 => return Err(InflateError::InvalidLookback),
                other => return Err(InflateError::Internal(other)),
            }

            if self.state.stopped_at != 0 {
                break;
            }
            if finished {
                break;
            }
            if self.state.avail_out == 0 {
                break;
            }
            let made_progress = self.state.avail_in != prev_avail_in
                || self.state.read_in_length != prev_read_in_length
                || self.state.avail_out != prev_avail_out;
            if !made_progress {
                // Input genuinely exhausted: refill staged nothing AND
                // ISA-L produced nothing AND bit buffer didn't move.
                break;
            }
        }

        Ok(ReadStreamResult {
            bytes_written: original_len - self.state.avail_out as usize,
            stopped_at: StoppingPoints(self.state.stopped_at),
            bit_position: self.tell_compressed(),
            finished,
        })
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
    pub fn with_until_bits(
        _input: &'a [u8],
        _bit_offset: usize,
        _until_bits: usize,
    ) -> Result<Self, InflateError> {
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

    fn make_multi_block_deflate(payload: &[u8]) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress, Status};
        let mut compress = Compress::new(Compression::new(6), false);
        let mut out = Vec::new();
        let mut scratch = vec![0u8; 64 * 1024];
        for piece in payload.chunks(32 * 1024) {
            let mut block_data = piece;
            loop {
                let before_in = compress.total_in();
                let before_out = compress.total_out();
                let status = compress
                    .compress(block_data, &mut scratch, FlushCompress::None)
                    .unwrap();
                let consumed = (compress.total_in() - before_in) as usize;
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                block_data = &block_data[consumed..];
                if block_data.is_empty() {
                    break;
                }
                if matches!(status, Status::BufError) && produced == 0 {
                    break;
                }
            }
            loop {
                let before_out = compress.total_out();
                let status = compress
                    .compress(&[], &mut scratch, FlushCompress::Sync)
                    .unwrap();
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                if produced == 0 || matches!(status, Status::StreamEnd) {
                    break;
                }
            }
        }
        loop {
            let before_out = compress.total_out();
            let status = compress
                .compress(&[], &mut scratch, FlushCompress::Finish)
                .unwrap();
            let produced = (compress.total_out() - before_out) as usize;
            out.extend_from_slice(&scratch[..produced]);
            if matches!(status, Status::StreamEnd) || produced == 0 {
                break;
            }
        }
        out
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
        let payload = vec![b'x'; 200_000];
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

    /// Structural-port correctness test for `with_until_bits` + the
    /// refill cap. Decodes the same deflate stream three times, each
    /// time with `until_bits` set just past a known block boundary,
    /// and asserts:
    ///   - `tell_compressed()` at decode end equals `until_bits` (the
    ///     cap is enforced exactly).
    ///   - The output bytes equal the libdeflate one-shot decode
    ///     truncated to the same byte count.
    ///
    /// This test exists because before the port, no `with_until_bits`
    /// existed. After the port, it gates regressions in the refill
    /// cap: any drift in the cap accounting makes `tell_compressed`
    /// drift from `until_bits`.
    #[test]
    fn with_until_bits_stops_exactly_at_cap() {
        // Build a payload large enough to span multiple deflate blocks
        // at L6 with diverse content (so block boundaries aren't all
        // at the same fixed-Huffman pattern position).
        let mut payload = Vec::with_capacity(512 * 1024);
        for i in 0..(512 * 1024) {
            payload.push((i as u8).wrapping_mul(31).wrapping_add(7));
        }
        let deflate = make_multi_block_deflate(&payload);

        // Enumerate end-of-block positions by decoding once with
        // END_OF_BLOCK stops and recording bit_position at each stop.
        let mut block_ends: Vec<usize> = Vec::new();
        {
            let mut probe = IsalInflateWrapper::new(&deflate, 0).expect("probe init");
            probe.set_window(&[]).expect("probe set_window");
            probe.set_stopping_points(StoppingPoints::END_OF_BLOCK);
            let mut buf = vec![0u8; payload.len() + 1024];
            let mut total = 0usize;
            loop {
                let r = probe.read_stream(&mut buf[total..]).expect("probe read");
                total += r.bytes_written;
                if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                    block_ends.push(r.bit_position);
                    if r.finished {
                        break;
                    }
                    continue;
                }
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
        }
        assert!(
            block_ends.len() >= 2,
            "fixture should produce multiple deflate blocks; got {}",
            block_ends.len()
        );

        // Pick three boundaries: first, middle, last-but-one.
        let idx_choices = [0usize, block_ends.len() / 2, block_ends.len() - 1];
        for &i in &idx_choices {
            let until = block_ends[i];
            let mut w = IsalInflateWrapper::with_until_bits(&deflate, 0, until)
                .expect("with_until_bits init");
            w.set_window(&[]).expect("set_window");
            let mut buf = vec![0u8; payload.len() + 1024];
            let mut total = 0usize;
            loop {
                let r = w.read_stream(&mut buf[total..]).expect("read_stream");
                total += r.bytes_written;
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
            buf.truncate(total);

            // The cap was set to a real EOB position, so vendor's
            // structural invariant is that tell_compressed() == until
            // at decode end (`gzip/isal.hpp` chain invariant). gzippy
            // post-port: the same.
            assert_eq!(
                w.tell_compressed(),
                until,
                "tell_compressed must equal until_bits for cap at EOB #{i} ({until})",
            );

            // Output bytes must match libdeflate one-shot truncated to
            // the same byte count. We compare against a sequential
            // decode that goes the full distance and truncate.
            let reference = {
                let mut d = flate2::Decompress::new(false);
                let mut out = vec![0u8; payload.len()];
                d.decompress(&deflate, &mut out, flate2::FlushDecompress::Finish)
                    .expect("flate2 ref decode");
                out.truncate(d.total_out() as usize);
                out
            };
            assert_eq!(
                buf,
                reference[..buf.len()],
                "output bytes must match reference[..{}] for cap at EOB #{i} ({until})",
                buf.len()
            );
        }
    }

    /// Cap of `0` (start_bit == until_bits) decodes nothing and reports
    /// tell_compressed == 0. Edge case for the refill-loop entry guard
    /// at vendor isal.hpp:231.
    #[test]
    fn with_until_bits_zero_cap_decodes_nothing() {
        let payload = b"abcdefg".repeat(1000);
        let deflate = make_deflate(&payload);
        let mut w = IsalInflateWrapper::with_until_bits(&deflate, 0, 0).expect("init");
        w.set_window(&[]).expect("set_window");
        let mut buf = vec![0u8; 4096];
        let r = w.read_stream(&mut buf).expect("read_stream");
        assert_eq!(r.bytes_written, 0);
        assert_eq!(w.tell_compressed(), 0);
    }

    /// Cross-chunk resume: decode to a non-byte-aligned boundary, take the
    /// last 32 KiB as dict, resume with `with_until_bits` + `set_window`.
    #[test]
    fn with_until_bits_resume_non_byte_aligned_with_dict() {
        let payload = vec![b'x'; 400_000];
        let deflate = make_deflate(&payload);

        let mut block_ends: Vec<usize> = Vec::new();
        {
            let mut probe = IsalInflateWrapper::new(&deflate, 0).expect("probe");
            probe.set_window(&[]).expect("set_window");
            probe.set_stopping_points(StoppingPoints::END_OF_BLOCK);
            let mut buf = vec![0u8; payload.len() + 1024];
            let mut total = 0usize;
            loop {
                let r = probe.read_stream(&mut buf[total..]).expect("read");
                total += r.bytes_written;
                if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                    block_ends.push(r.bit_position);
                    if r.finished {
                        break;
                    }
                    continue;
                }
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
        }
        let resume_at = block_ends
            .iter()
            .copied()
            .find(|p| *p % 8 != 0)
            .unwrap_or_else(|| block_ends[block_ends.len() / 2]);

        let mut first = IsalInflateWrapper::with_until_bits(&deflate, 0, resume_at).expect("init");
        first.set_window(&[]).expect("set_window");
        let mut out1 = vec![0u8; payload.len()];
        let mut n1 = 0usize;
        loop {
            let r = first.read_stream(&mut out1[n1..]).expect("read");
            n1 += r.bytes_written;
            if r.bytes_written == 0 {
                break;
            }
        }
        out1.truncate(n1);
        assert_eq!(first.tell_compressed(), resume_at);

        let window_start = n1.saturating_sub(32768);
        let window = &out1[window_start..n1];

        let mut second =
            IsalInflateWrapper::with_until_bits(&deflate, resume_at, deflate.len() * 8)
                .expect("resume init");
        second.set_window(window).expect("set_window");
        let mut out2 = vec![0u8; payload.len()];
        let mut n2 = 0usize;
        loop {
            let r = second.read_stream(&mut out2[n2..]).expect("read");
            n2 += r.bytes_written;
            if r.finished || r.bytes_written == 0 {
                break;
            }
        }
        out2.truncate(n2);

        let mut full = out1;
        full.extend_from_slice(&out2);
        let reference = {
            let mut d = flate2::Decompress::new(false);
            let mut out = vec![0u8; payload.len()];
            d.decompress(&deflate, &mut out, flate2::FlushDecompress::Finish)
                .expect("flate2");
            out.truncate(d.total_out() as usize);
            out
        };
        assert_eq!(full, reference);
    }
}
