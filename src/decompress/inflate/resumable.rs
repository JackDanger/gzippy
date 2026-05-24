//! Vendor-faithful resumable DEFLATE inflate (plans/rust-rapidgzip.md §5,
//! option 2).
//!
//! Mirrors `vendor/.../gzip/isal.hpp:254-356` `IsalInflateWrapper::readStream`:
//! the decoder writes directly into the caller's `output` slice and maintains
//! an internal 32 KiB sliding window for back-references that reach past
//! `output[0]`. When `output` fills mid-block, the decoder saves enough state
//! (`PendingMatch` + bit-reader position + active tables) to resume on the
//! next call.
//!
//! This replaces the `session: Vec<u8>` accumulator in
//! [`crate::decompress::inflate::consume_first_decode::ResumableInflate`].
//! The old accumulator was a band-aid for the fact that the existing
//! `decode_dynamic` / `decode_fixed` / `decode_stored` (used by BGZF and
//! sequential decompress) run a block to completion in one shot. Those
//! decoders stay untouched for their non-resumable callers; this module
//! contains the resumable variants used only by `ResumableInflate2` and,
//! via [`crate::decompress::parallel::inflate_wrapper::IsalInflateWrapper`],
//! the parallel-SM hot path.
//!
//! Fill-in status: step 2 (stored block) landed; fixed + dynamic + the
//! window-stitched match copy are still `unimplemented!`. See
//! `plans/rust-rapidgzip.md §5 — Implementation order`.

#![allow(dead_code)] // step 2: dispatch methods called only by step-3 wiring

use std::io::{Error, ErrorKind, Result};

use super::consume_first_decode::Bits;
use super::stopping_point::StoppingPoint;

/// Max DEFLATE back-reference distance; size of the sliding window the
/// resumable decoder must keep for matches that reach past `output[0]`.
pub const WINDOW_SIZE: usize = 32 * 1024;

/// Saved state for a match that was emitted partially when `output` filled.
/// On resume, the decoder must finish this copy before reading the next
/// litlen symbol from the bit stream.
#[derive(Debug, Clone, Copy, Default)]
pub struct PendingMatch {
    pub distance: u32,
    /// Bytes of the match still to be copied.
    pub length_remaining: u32,
}

/// Block-decode state machine position. Persists across `read_stream` calls
/// so the decoder can resume mid-block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockState {
    /// Not currently inside a block — the next call reads the 3-bit
    /// `(BFINAL, BTYPE)` header.
    #[default]
    AwaitingHeader,
    /// Inside a BTYPE=00 stored block; `bytes_remaining` literal bytes still
    /// to copy from the input.
    InStored { bytes_remaining: u32 },
    /// Inside a BTYPE=01 fixed-Huffman block; tables are static.
    InFixed,
    /// Inside a BTYPE=02 dynamic-Huffman block; tables live on the
    /// `ResumableInflate2` (rebuilt from the block header at entry).
    InDynamic,
    /// Stream is finished (final block's BFINAL=1 completed).
    Finished,
}

/// 32 KiB ring buffer holding the last `WINDOW_SIZE` decoded output bytes.
/// Fed from `output` after each `read_stream` call; consulted by the
/// match-copy when `distance > out_pos`.
#[derive(Clone)]
pub struct SlidingWindow {
    buf: Box<[u8; WINDOW_SIZE]>,
    /// Next write position; wraps mod `WINDOW_SIZE`. After the first
    /// `WINDOW_SIZE` bytes have been written, `head` no longer corresponds
    /// to "logical length" — use `written` instead.
    head: usize,
    /// Total bytes ever written, saturating at `usize::MAX`. Used by
    /// `lookback` to know how many bytes the window actually contains.
    written: usize,
}

impl Default for SlidingWindow {
    fn default() -> Self {
        Self {
            buf: Box::new([0; WINDOW_SIZE]),
            head: 0,
            written: 0,
        }
    }
}

impl SlidingWindow {
    /// Number of bytes currently retrievable from the window
    /// (= `min(written, WINDOW_SIZE)`).
    pub fn len(&self) -> usize {
        self.written.min(WINDOW_SIZE)
    }

    pub fn is_empty(&self) -> bool {
        self.written == 0
    }

    /// Append `src` to the window, advancing `head` (mod `WINDOW_SIZE`).
    /// Only the trailing `WINDOW_SIZE` bytes are retained.
    pub fn extend(&mut self, src: &[u8]) {
        if src.is_empty() {
            return;
        }
        // Only the trailing WINDOW_SIZE bytes can possibly matter.
        let drop = src.len().saturating_sub(WINDOW_SIZE);
        let tail = &src[drop..];
        let n = tail.len();
        // Copy in up to two segments (`head..head+n` may wrap past WINDOW_SIZE).
        let first = n.min(WINDOW_SIZE - self.head);
        self.buf[self.head..self.head + first].copy_from_slice(&tail[..first]);
        let rest = n - first;
        if rest > 0 {
            self.buf[..rest].copy_from_slice(&tail[first..]);
        }
        self.head = (self.head + n) % WINDOW_SIZE;
        // `src.len()` (not `n`) so callers that flushed >WINDOW_SIZE bytes
        // still mark the window as full.
        self.written = self.written.saturating_add(src.len());
    }

    /// Copy the last `n` bytes of the window into `dst[..n]`. `n` must be
    /// ≤ `self.len()` (call `len()` first to check). Handles the
    /// ring-buffer wrap with up to two memcpys.
    pub fn copy_last_n_into(&self, n: usize, dst: &mut [u8]) {
        debug_assert!(n <= self.len(), "lookback {n} > window len {}", self.len());
        debug_assert!(dst.len() >= n, "dst too small: {} < {n}", dst.len());
        if n == 0 {
            return;
        }
        // The byte that is `i` positions back from the most recent write
        // sits at `(head + WINDOW_SIZE - 1 - i) mod WINDOW_SIZE`. The "last
        // n" range begins at `(head + WINDOW_SIZE - n) mod WINDOW_SIZE`.
        let start = (self.head + WINDOW_SIZE - n) % WINDOW_SIZE;
        let first = n.min(WINDOW_SIZE - start);
        dst[..first].copy_from_slice(&self.buf[start..start + first]);
        let rest = n - first;
        if rest > 0 {
            dst[first..first + rest].copy_from_slice(&self.buf[..rest]);
        }
    }

    /// Seed the window with up to 32 KiB of bytes (the chunk's predecessor
    /// window from `WindowMap`). Replaces any prior contents.
    pub fn seed(&mut self, bytes: &[u8]) {
        self.head = 0;
        self.written = 0;
        // `extend` will correctly take the trailing 32 KiB.
        self.extend(bytes);
    }
}

/// Result of one `read_stream` call. Mirrors
/// [`crate::decompress::inflate::consume_first_decode::InflateStreamResult`]
/// but with no `session` semantics — `bytes_written` is exactly what landed
/// in the caller's `output`.
#[derive(Debug, Clone, Copy)]
pub struct InflateStreamResult {
    pub bytes_written: usize,
    pub stopped_at: StoppingPoint,
    pub bit_position: usize,
    pub finished: bool,
}

/// Vendor-faithful resumable inflate.
///
/// Owns the input bit reader and a [`SlidingWindow`]. Replaces the
/// `session: Vec<u8>` accumulator on
/// [`crate::decompress::inflate::consume_first_decode::ResumableInflate`].
pub struct ResumableInflate2<'a> {
    pub(crate) bits: Bits<'a>,
    pub(crate) window: SlidingWindow,
    pub(crate) block_state: BlockState,
    pub(crate) pending_match: Option<PendingMatch>,
    pub(crate) last_bfinal: bool,
    pub(crate) points_to_stop_at: StoppingPoint,
    pub(crate) stopped_at: StoppingPoint,
    pub(crate) encoded_until_bits: usize,
    // TODO(§5 step 3+): persistent Huffman tables for `BlockState::InDynamic`.
}

impl<'a> ResumableInflate2<'a> {
    /// Construct positioned at `bit_offset` into `input`, with no window
    /// (caller seeds via [`Self::set_window`]).
    pub fn with_until_bits(input: &'a [u8], bit_offset: usize, until_bits: usize) -> Result<Self> {
        let input_bits = input.len().saturating_mul(8);
        if bit_offset > input_bits {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "bit_offset past end of input",
            ));
        }
        Ok(Self {
            bits: Bits::at_bit_offset(input, bit_offset),
            window: SlidingWindow::default(),
            block_state: BlockState::default(),
            pending_match: None,
            last_bfinal: false,
            points_to_stop_at: StoppingPoint::NONE,
            stopped_at: StoppingPoint::NONE,
            encoded_until_bits: until_bits.min(input_bits),
        })
    }

    /// Seed the sliding window from a chunk's predecessor 32 KiB. Replaces
    /// any prior window contents and clears block state.
    pub fn set_window(&mut self, window: &[u8]) -> Result<()> {
        self.window.seed(window);
        self.block_state = BlockState::AwaitingHeader;
        self.pending_match = None;
        self.last_bfinal = false;
        self.stopped_at = StoppingPoint::NONE;
        Ok(())
    }

    /// Configure which `StoppingPoint`s cause an early return from
    /// `read_stream`.
    pub fn set_stopping_points(&mut self, points: StoppingPoint) {
        self.points_to_stop_at = if points == StoppingPoint::NONE {
            StoppingPoint::NONE
        } else {
            StoppingPoint(points.0 & StoppingPoint::ALL.0)
        };
    }

    pub fn stopped_at(&self) -> StoppingPoint {
        self.stopped_at
    }

    pub fn bit_position(&self) -> usize {
        self.bits.bit_position()
    }

    pub fn is_final_block(&self) -> bool {
        self.last_bfinal
    }

    /// Drive the decoder into `output`, stopping when:
    ///   (1) `output` is full,
    ///   (2) a requested `StoppingPoint` fires,
    ///   (3) `encoded_until_bits` is reached,
    ///   (4) the final block's BFINAL completes.
    pub fn read_stream(&mut self, output: &mut [u8]) -> Result<InflateStreamResult> {
        self.stopped_at = StoppingPoint::NONE;
        let out_pos_start = 0usize;
        let mut out_pos = out_pos_start;

        loop {
            if out_pos >= output.len() {
                break;
            }

            // Resume any pending match from a prior yield BEFORE reading new
            // bits — the match is logically part of the prior block.
            if let Some(pending) = self.pending_match {
                let new_pos = copy_match_windowed(
                    self,
                    output,
                    out_pos,
                    pending.distance,
                    pending.length_remaining,
                )?;
                out_pos = new_pos;
                if self.pending_match.is_some() {
                    // copy_match_windowed yielded again — output is full.
                    debug_assert_eq!(out_pos, output.len());
                    break;
                }
            }

            match self.block_state {
                BlockState::Finished => break,
                BlockState::AwaitingHeader => {
                    if !self.try_enter_next_block()? {
                        // Out of input bits.
                        break;
                    }
                    if self
                        .points_to_stop_at
                        .contains(StoppingPoint::END_OF_BLOCK_HEADER)
                    {
                        self.stopped_at = StoppingPoint::END_OF_BLOCK_HEADER;
                        break;
                    }
                }
                BlockState::InStored { .. } => {
                    out_pos = resume_decode_stored_resumable(self, output, out_pos)?;
                }
                BlockState::InFixed | BlockState::InDynamic => {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "fixed/dynamic resumable decoders pending §5 step 3/4",
                    ));
                }
            }

            if matches!(self.block_state, BlockState::Finished) {
                if self
                    .points_to_stop_at
                    .contains(StoppingPoint::END_OF_STREAM)
                {
                    self.stopped_at = StoppingPoint::END_OF_STREAM;
                }
                break;
            }

            if matches!(self.block_state, BlockState::AwaitingHeader)
                && self.points_to_stop_at.contains(StoppingPoint::END_OF_BLOCK)
            {
                self.stopped_at = StoppingPoint::END_OF_BLOCK;
                break;
            }
        }

        // Feed the bytes just emitted to the user into the sliding window so
        // future match-copies can reach them after they leave `output`.
        if out_pos > out_pos_start {
            self.window.extend(&output[out_pos_start..out_pos]);
        }

        let finished =
            matches!(self.block_state, BlockState::Finished) && self.pending_match.is_none();
        Ok(InflateStreamResult {
            bytes_written: out_pos - out_pos_start,
            stopped_at: self.stopped_at,
            bit_position: self.bits.bit_position(),
            finished,
        })
    }

    /// Read the 3-bit `(BFINAL, BTYPE)` header and parse any block-type-specific
    /// header bits (LEN/NLEN for stored). Returns `Ok(false)` if no header bits
    /// are available (end of input). On entry must have `block_state ==
    /// AwaitingHeader`.
    fn try_enter_next_block(&mut self) -> Result<bool> {
        debug_assert!(matches!(self.block_state, BlockState::AwaitingHeader));

        // Need at least 3 bits to read (BFINAL, BTYPE).
        if self.bits.available() < 3 {
            self.bits.refill();
        }
        if self.bits.available() < 3 {
            return Ok(false);
        }
        if self.bits.bit_position() >= self.encoded_until_bits {
            return Ok(false);
        }

        let header = self.bits.bitbuf & 0b111;
        self.bits.consume(3);
        self.last_bfinal = (header & 1) != 0;
        let btype = (header >> 1) & 0b11;
        match btype {
            0 => {
                let len = self.bits.read_u16();
                let nlen = self.bits.read_u16();
                if len != !nlen {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "Stored block: len={len:#x} nlen={nlen:#x} (~nlen={:#x})",
                            !nlen
                        ),
                    ));
                }
                self.block_state = BlockState::InStored {
                    bytes_remaining: len as u32,
                };
                Ok(true)
            }
            1 => {
                self.block_state = BlockState::InFixed;
                Ok(true)
            }
            2 => {
                self.block_state = BlockState::InDynamic;
                Ok(true)
            }
            _ => Err(Error::new(
                ErrorKind::InvalidData,
                "Reserved DEFLATE block type (BTYPE=11)",
            )),
        }
    }

    /// Mark the current block complete: transition back to `AwaitingHeader`,
    /// or to `Finished` if the most recent block's BFINAL=1. After-block
    /// hook used by every `resume_decode_*_resumable` to keep that bit of
    /// state machine logic out of each individual decoder.
    pub(crate) fn finish_current_block(&mut self) {
        self.block_state = if self.last_bfinal {
            BlockState::Finished
        } else {
            BlockState::AwaitingHeader
        };
    }
}

// =============================================================================
// Resumable block decoders (one per BTYPE)
// =============================================================================
//
// Each yields by saving `PendingMatch` + `BlockState` + bit position on the
// `ResumableInflate2` and returning. The non-resumable counterparts live in
// `consume_first_decode.rs` and stay untouched for BGZF / sequential
// decompress (they need no yield-check tax).

/// Resumable BTYPE=00 (uncompressed): copy `bytes_remaining` literal bytes
/// from input to output, yielding when `output` fills. On entry, the bit
/// reader is positioned immediately after LEN/NLEN (see
/// `ResumableInflate2::try_enter_next_block`); `block_state` is
/// `InStored { bytes_remaining }` with `bytes_remaining > 0` (or the block
/// is empty and we transition straight out).
pub fn resume_decode_stored_resumable(
    state: &mut ResumableInflate2<'_>,
    output: &mut [u8],
    mut out_pos: usize,
) -> Result<usize> {
    let BlockState::InStored {
        mut bytes_remaining,
    } = state.block_state
    else {
        debug_assert!(
            false,
            "resume_decode_stored_resumable called outside InStored"
        );
        return Ok(out_pos);
    };

    if bytes_remaining == 0 {
        state.finish_current_block();
        return Ok(out_pos);
    }

    // After try_enter_next_block parsed LEN/NLEN, the bit cursor is
    // byte-aligned (read_u16 calls align_to_byte). We could drain the
    // remaining buffered bits into output one byte at a time — but that
    // leaves `bitsleft = 0` while `bitbuf` still holds the bytes the
    // libdeflate-style refill speculatively loaded past the LEN/NLEN
    // word. The next `refill()` (`bitbuf |= word << bits_u8` with
    // `bits_u8 = 0`) ORs new bytes onto those stale ones and corrupts
    // the next block's header. Avoid the trap entirely: locate the
    // first data byte via `bit_position`, memcpy directly from
    // `bits.data`, and reset the bit reader so the next refill starts
    // clean.
    debug_assert_eq!(
        state.bits.bit_position() % 8,
        0,
        "stored block body must start byte-aligned"
    );
    let data_start_byte = state.bits.bit_position() / 8;
    let copy_n = (bytes_remaining as usize).min(output.len() - out_pos);
    if data_start_byte + copy_n > state.bits.data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "Truncated stored block: need {copy_n} bytes from offset {data_start_byte}, have {}",
                state.bits.data.len().saturating_sub(data_start_byte)
            ),
        ));
    }
    output[out_pos..out_pos + copy_n]
        .copy_from_slice(&state.bits.data[data_start_byte..data_start_byte + copy_n]);
    state.bits.pos = data_start_byte + copy_n;
    state.bits.bitbuf = 0;
    state.bits.bitsleft = 0;
    out_pos += copy_n;
    bytes_remaining -= copy_n as u32;

    if bytes_remaining == 0 {
        state.finish_current_block();
    } else {
        // Yield mid-block: stash remaining count so the next call resumes.
        state.block_state = BlockState::InStored { bytes_remaining };
    }
    Ok(out_pos)
}

/// Resumable BTYPE=01 (fixed Huffman): static tables, no header parse.
/// Pending — §5 step 3.
pub fn resume_decode_fixed_resumable(
    _state: &mut ResumableInflate2<'_>,
    _output: &mut [u8],
    _out_pos: usize,
) -> Result<usize> {
    unimplemented!("§5 step 3 fill-in")
}

/// Resumable BTYPE=02 (dynamic Huffman). On first entry the block header is
/// read and tables are built onto `state`; subsequent re-entries resume
/// mid-block using those same tables. Pending — §5 step 4.
pub fn resume_decode_dynamic_resumable(
    _state: &mut ResumableInflate2<'_>,
    _output: &mut [u8],
    _out_pos: usize,
) -> Result<usize> {
    unimplemented!("§5 step 4 fill-in")
}

// =============================================================================
// Window-stitched match copy
// =============================================================================

/// LZ77 back-reference copy honoring both `output` bytes already written
/// this call AND the [`SlidingWindow`] of bytes that have already been
/// flushed to prior `read_stream` callers.
///
/// Two cases:
///   1. `distance <= out_pos` — match source is in `output[out_pos -
///      distance..]`. Existing fast `copy_match_*` logic applies.
///   2. `distance > out_pos` — match source is in the window. Read
///      `(distance - out_pos)` bytes from the window tail, then continue
///      into `output` if `length` exceeds that.
///
/// Returns the new `out_pos` (advanced by `length`), or yields a
/// [`PendingMatch`] on `state` and returns the (truncated) `out_pos`
/// (= `output.len()`) if `output` filled mid-copy.
pub fn copy_match_windowed(
    _state: &mut ResumableInflate2<'_>,
    _output: &mut [u8],
    _out_pos: usize,
    _distance: u32,
    _length: u32,
) -> Result<usize> {
    // §5 step 3+: implementation once the first match-emitting Huffman
    // decoder lands. Until then, only stored blocks (which never emit
    // matches) reach `read_stream`. Returning `Unsupported` instead of
    // `unimplemented!()` means a future bug that stashes a `PendingMatch`
    // before step 3 lands surfaces as a recoverable `Err` rather than a
    // process panic — important since this runs on worker threads in
    // the parallel-SM path. (Advisor review of commit f296be1.)
    Err(Error::new(
        ErrorKind::Unsupported,
        "copy_match_windowed pending §5 step 3 — no decoder should be \
         emitting PendingMatch yet",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_stored_block(data: &[u8], bfinal: bool) -> Vec<u8> {
        // 3-bit header packed into byte 0: low bit = BFINAL, next 2 = BTYPE=0.
        // After byte-align, LEN (2 bytes LE), NLEN (~LEN, 2 bytes LE), then data.
        let mut out = Vec::with_capacity(5 + data.len());
        out.push(if bfinal { 0b001 } else { 0b000 });
        let len = data.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn sliding_window_roundtrip() {
        let mut w = SlidingWindow::default();
        w.extend(b"hello world");
        assert_eq!(w.len(), 11);
        let mut buf = [0u8; 5];
        w.copy_last_n_into(5, &mut buf);
        assert_eq!(&buf, b"world");

        // Force a wrap: write more than WINDOW_SIZE bytes total.
        let big = vec![0xAAu8; WINDOW_SIZE + 100];
        w.extend(&big);
        assert_eq!(w.len(), WINDOW_SIZE);
        let mut tail = vec![0u8; 50];
        w.copy_last_n_into(50, &mut tail);
        assert!(tail.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn sliding_window_seed_replaces_contents() {
        let mut w = SlidingWindow::default();
        w.extend(b"oldoldold");
        w.seed(b"NEW");
        assert_eq!(w.len(), 3);
        let mut buf = [0u8; 3];
        w.copy_last_n_into(3, &mut buf);
        assert_eq!(&buf, b"NEW");
    }

    #[test]
    fn sliding_window_seed_trims_to_window_size() {
        let mut w = SlidingWindow::default();
        let big = vec![0xCDu8; WINDOW_SIZE * 3 + 17];
        w.seed(&big);
        // `seed` truncates to keep only the trailing WINDOW_SIZE bytes.
        assert_eq!(w.len(), WINDOW_SIZE);
        let mut tail = [0u8; 16];
        w.copy_last_n_into(16, &mut tail);
        assert!(tail.iter().all(|&b| b == 0xCD));
    }

    #[test]
    fn stored_block_full_output_buffer() {
        let payload = b"abcdefghijklmnopqrstuvwxyz0123456789".repeat(10);
        let stream = raw_stored_block(&payload, true);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = vec![0u8; payload.len()];
        let r = inflate.read_stream(&mut output).unwrap();
        assert_eq!(r.bytes_written, payload.len());
        assert!(r.finished);
        assert_eq!(output, payload);
    }

    #[test]
    fn stored_block_small_output_yields_then_resumes() {
        let payload = b"abcdefghijklmnopqrstuvwxyz0123456789".repeat(20); // 720 bytes
        let stream = raw_stored_block(&payload, true);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut collected = Vec::with_capacity(payload.len());
        let chunk = 37usize; // intentionally awkward
        loop {
            let mut output = vec![0u8; chunk];
            let r = inflate.read_stream(&mut output).unwrap();
            if r.bytes_written == 0 && r.finished {
                break;
            }
            collected.extend_from_slice(&output[..r.bytes_written]);
            if r.finished {
                break;
            }
            assert!(r.bytes_written > 0, "no progress on chunk of {chunk}");
        }
        assert_eq!(collected, payload);
    }

    #[test]
    fn stored_block_empty_payload_finishes_immediately() {
        let stream = raw_stored_block(b"", true);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = [0u8; 16];
        let r = inflate.read_stream(&mut output).unwrap();
        assert_eq!(r.bytes_written, 0);
        assert!(r.finished);
    }

    #[test]
    fn multi_stored_blocks_concatenate() {
        let mut stream = raw_stored_block(b"first chunk ", false);
        stream.extend(raw_stored_block(b"second chunk ", false));
        stream.extend(raw_stored_block(b"third final chunk", true));
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = vec![0u8; 256];
        let r = inflate.read_stream(&mut output).unwrap();
        assert!(
            r.finished,
            "all 3 blocks should decode in one call given ample output"
        );
        assert_eq!(
            &output[..r.bytes_written],
            b"first chunk second chunk third final chunk"
        );
    }

    #[test]
    fn stored_block_corrupt_nlen_errors() {
        let mut stream = raw_stored_block(b"data", true);
        // Flip a bit in NLEN (bytes 3..5) so len != !nlen.
        stream[3] ^= 0x01;
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = [0u8; 16];
        let err = inflate.read_stream(&mut output).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    // Advisor-requested H1: tiny output buffer (1 byte) forces every iteration
    // to yield with bytes_remaining > 0. Catches off-by-one in the
    // bytes_remaining -= copy_n math, and proves resume-after-yield works.
    #[test]
    fn stored_block_one_byte_chunks_yield_every_iteration() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(3);
        let stream = raw_stored_block(&payload, true);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut collected = Vec::with_capacity(payload.len());
        let mut iterations = 0usize;
        loop {
            let mut output = [0u8; 1];
            let r = inflate.read_stream(&mut output).unwrap();
            iterations += 1;
            if r.bytes_written == 0 && r.finished {
                break;
            }
            collected.extend_from_slice(&output[..r.bytes_written]);
            if r.finished {
                break;
            }
            assert!(iterations < payload.len() + 10, "runaway loop");
        }
        assert_eq!(collected, payload);
        assert!(
            iterations >= payload.len(),
            "expected ≥ payload.len() yields, got {iterations}"
        );
    }

    // Advisor-requested H2: stored block whose LEN claims more bytes than
    // the input has. The truncated-stored branch in
    // resume_decode_stored_resumable returns UnexpectedEof — until this
    // test landed it was uncovered.
    #[test]
    fn stored_block_truncated_input_errors() {
        let payload = b"ten bytes!"; // LEN = 10
        let mut stream = raw_stored_block(payload, true);
        // Drop the last 4 bytes so the stored body is shorter than LEN claims.
        stream.truncate(stream.len() - 4);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = [0u8; 32];
        let err = inflate.read_stream(&mut output).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    }

    // Advisor-requested H3: stored block followed by a fixed-Huffman block
    // currently routes to the `Unsupported` arm in read_stream. Locks in
    // "stored-then-fixed yields Unsupported, not silent corruption or a
    // panic" so step 3 wiring can't regress this contract by accident.
    #[test]
    fn stored_then_fixed_returns_unsupported_until_step3() {
        // Block 1: stored "hi" (non-final).
        let mut stream = raw_stored_block(b"hi", false);
        // Block 2: minimal fixed-Huffman BFINAL=1 block — header byte's
        // low 3 bits are 0b011 (BFINAL=1, BTYPE=01). Body content doesn't
        // matter; we only need to reach `InFixed` dispatch.
        stream.push(0b011);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = vec![0u8; 64];
        let err = inflate.read_stream(&mut output).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        let msg = err.to_string();
        assert!(
            msg.contains("step 3") || msg.contains("step 4"),
            "Unsupported error should reference the fill-in step: {msg}"
        );
    }

    // Advisor-requested H4: bit_position after a stored block lands exactly
    // on the byte after the stored payload — locks down the explicit
    // bitbuf=0/bitsleft=0 reset pattern so future "optimizations" can't
    // silently break the byte alignment.
    #[test]
    fn stored_block_bit_position_lands_at_payload_end() {
        let payload = b"ABCDEFGH"; // 8 bytes
        let stream = raw_stored_block(payload, true);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = vec![0u8; payload.len()];
        let r = inflate.read_stream(&mut output).unwrap();
        assert_eq!(r.bytes_written, payload.len());
        // Block layout: 1 header byte + 2 LEN + 2 NLEN + payload.
        // bit_position should be the bit-index of the byte immediately after.
        let expected_bit_position = (5 + payload.len()) * 8;
        assert_eq!(
            inflate.bit_position(),
            expected_bit_position,
            "bit_position must land on the byte after the stored payload"
        );
        assert_eq!(r.bit_position, expected_bit_position);
    }
}
