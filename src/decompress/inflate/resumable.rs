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
//! Skeleton only — see `Beyond parity` and `§5 — Implementation order` in
//! `plans/rust-rapidgzip.md` for the fill-in plan.

#![allow(dead_code)] // skeleton; methods unused until decoders land

use std::io::Result;

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
}

/// 32 KiB ring buffer holding the last `WINDOW_SIZE` decoded output bytes.
/// Fed from `output` after each `read_stream` call; consulted by the
/// match-copy when `distance > out_pos`.
#[derive(Clone)]
pub struct SlidingWindow {
    buf: Box<[u8; WINDOW_SIZE]>,
    /// Next write position; wraps. `len < WINDOW_SIZE` until the first wrap.
    head: usize,
    /// Total bytes ever written (saturating); used to know whether `head`
    /// has wrapped.
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
    /// Returns the last `n` bytes ever fed into the window (or `None` if
    /// fewer than `n` bytes have ever been written). `n` must be ≤
    /// [`WINDOW_SIZE`].
    pub fn lookback(&self, _n: usize) -> Option<&[u8]> {
        // TODO(§5): wrap-aware lookup returning a slice (may span the wrap
        // boundary; caller must handle two-segment reads via `lookback_split`).
        unimplemented!("§5 fill-in")
    }

    /// Append `src` to the window, advancing `head` (mod `WINDOW_SIZE`).
    /// Only the trailing `WINDOW_SIZE` bytes are retained.
    pub fn extend(&mut self, _src: &[u8]) {
        // TODO(§5): memcpy with potential wrap split.
        unimplemented!("§5 fill-in")
    }

    /// Seed the window with up to 32 KiB of bytes (the chunk's predecessor
    /// window from `WindowMap`). Replaces any prior contents.
    pub fn seed(&mut self, _bytes: &[u8]) {
        // TODO(§5): take `bytes[bytes.len().saturating_sub(WINDOW_SIZE)..]`.
        unimplemented!("§5 fill-in")
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
    // TODO(§5): persistent Huffman tables for `BlockState::InDynamic`.
    // These are built once per dynamic block at entry and must survive
    // across mid-block yields.
}

impl<'a> ResumableInflate2<'a> {
    /// Construct positioned at `bit_offset` into `input`, with no window
    /// (caller seeds via [`Self::set_window`]).
    pub fn with_until_bits(
        _input: &'a [u8],
        _bit_offset: usize,
        _until_bits: usize,
    ) -> Result<Self> {
        // TODO(§5): mirror `consume_first_decode::ResumableInflate::with_until_bits`.
        unimplemented!("§5 fill-in")
    }

    /// Seed the sliding window from a chunk's predecessor 32 KiB.
    pub fn set_window(&mut self, _window: &[u8]) -> Result<()> {
        unimplemented!("§5 fill-in")
    }

    /// Configure which `StoppingPoint`s cause an early return from
    /// `read_stream`.
    pub fn set_stopping_points(&mut self, _points: StoppingPoint) {
        unimplemented!("§5 fill-in")
    }

    /// Drive the decoder into `output`, stopping when:
    ///   (1) `output` is full,
    ///   (2) a requested `StoppingPoint` fires,
    ///   (3) `encoded_until_bits` is reached,
    ///   (4) the final block's BFINAL completes and footer is past.
    ///
    /// On (1), the decoder saves [`BlockState`] + [`PendingMatch`] + bit
    /// reader state so the next call resumes mid-block.
    pub fn read_stream(&mut self, _output: &mut [u8]) -> Result<InflateStreamResult> {
        // TODO(§5): dispatch on `self.block_state`:
        //   AwaitingHeader → read (BFINAL, BTYPE); enter Stored/Fixed/Dynamic.
        //   InStored      → resume_decode_stored_resumable(self, output)
        //   InFixed       → resume_decode_fixed_resumable(self, output)
        //   InDynamic     → resume_decode_dynamic_resumable(self, output)
        // Match-copy uses `copy_match_windowed` (see below).
        unimplemented!("§5 fill-in")
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
/// from input to output, yielding when `output` fills.
pub fn resume_decode_stored_resumable<'a>(
    _state: &mut ResumableInflate2<'a>,
    _output: &mut [u8],
    _out_pos: usize,
) -> Result<usize> {
    // TODO(§5): straight memcpy with byte-aligned bit reader; trivial yield.
    unimplemented!("§5 fill-in")
}

/// Resumable BTYPE=01 (fixed Huffman): static tables, no header parse.
pub fn resume_decode_fixed_resumable<'a>(
    _state: &mut ResumableInflate2<'a>,
    _output: &mut [u8],
    _out_pos: usize,
) -> Result<usize> {
    // TODO(§5): inner loop based on `decode_huffman_libdeflate_style` but
    // with yield-on-output-full + window-stitched match-copy. Tables are
    // the static fixed-Huffman tables from `libdeflate_entry`.
    unimplemented!("§5 fill-in")
}

/// Resumable BTYPE=02 (dynamic Huffman). On first entry the block header is
/// read and tables are built onto `state`; subsequent re-entries resume
/// mid-block using those same tables.
pub fn resume_decode_dynamic_resumable<'a>(
    _state: &mut ResumableInflate2<'a>,
    _output: &mut [u8],
    _out_pos: usize,
) -> Result<usize> {
    // TODO(§5): table build on first entry; yield-aware inner loop on resume.
    unimplemented!("§5 fill-in")
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
/// [`PendingMatch`] on `state` and returns the unchanged `out_pos` if
/// `output` filled mid-copy.
pub fn copy_match_windowed<'a>(
    _state: &mut ResumableInflate2<'a>,
    _output: &mut [u8],
    _out_pos: usize,
    _distance: u32,
    _length: u32,
) -> Result<usize> {
    // TODO(§5):
    //   if distance <= out_pos: existing copy_match_safe / copy_match_fast.
    //   else: split = (distance - out_pos) as usize;
    //         read `split` bytes from `state.window.lookback(split)` into
    //         output[out_pos..out_pos+split], then loop into output for
    //         the remaining `length - split`.
    //   On output.len() boundary mid-copy: save PendingMatch + return.
    unimplemented!("§5 fill-in")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_compiles() {
        // The fill-in plan lives in `plans/rust-rapidgzip.md §5`. This test
        // exists only so the skeleton is exercised by `cargo test` and any
        // drift between this module's public API and the wrapper's
        // expectations surfaces immediately.
        assert_eq!(WINDOW_SIZE, 32 * 1024);
        let w = SlidingWindow::default();
        assert_eq!(w.head, 0);
        assert_eq!(w.written, 0);
        assert_eq!(BlockState::default(), BlockState::AwaitingHeader);
        let p = PendingMatch::default();
        assert_eq!(p.length_remaining, 0);
    }
}
