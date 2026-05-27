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
//! This is the production resumable inflate for the parallel-SM hot
//! path. The non-resumable `decode_dynamic` / `decode_fixed` /
//! `decode_stored` in `consume_first_decode.rs` (used by BGZF and
//! sequential decompress) run a block to completion in one shot and
//! pay no yield-check tax; this module's resumable variants exist
//! ONLY for `ResumableInflate2` and, via
//! [`crate::decompress::parallel::inflate_wrapper::IsalInflateWrapper`],
//! the parallel-SM hot path.
//!
//! Plan progress (`plans/rust-rapidgzip.md §5`): all 7 steps complete.
//! The old `ResumableInflate` + `session: Vec<u8>` accumulator and the
//! B3a headroom band-aid (commit 2eff70f) are deleted as of §5 step 6.

#![allow(dead_code)] // some vendor-parity methods are wrapper-only callers

use std::io::{Error, ErrorKind, Result};
use std::sync::atomic::AtomicU64;

use super::consume_first_decode::{build_code_length_table, Bits};

/// FASTLOOP multi-literal hit/miss counters. Inspect via test
/// harness or end-of-run print to verify the SIMD lookahead pays off
/// on the actual workload.
pub static MULTI_LITERAL_HITS: AtomicU64 = AtomicU64::new(0);
pub static MULTI_LITERAL_MISSES: AtomicU64 = AtomicU64::new(0);
pub static MULTI_LITERAL_SYMBOLS: AtomicU64 = AtomicU64::new(0);
pub static BODY_RESUMABLE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static BODY_RESUMABLE_FASTLOOP_ENTERS: AtomicU64 = AtomicU64::new(0);
/// Total bytes produced across all `read_stream` calls. Divide by
/// `READ_STREAM_CALLS` to get bytes/call — exposes the chunked-vs-mono
/// gap in production. Step 2.5 (advisor third-pass post-C1 verdict):
/// if production's bytes/call is small (e.g. ~64 KiB) we have the same
/// chunked-mode problem the bench shows; if it's large (multi-MiB) the
/// bench gap is bench-only and production is fine.
pub static READ_STREAM_CALLS: AtomicU64 = AtomicU64::new(0);
pub static READ_STREAM_BYTES_OUT: AtomicU64 = AtomicU64::new(0);
/// Sum across all calls. Divide by `READ_STREAM_CALLS` for average
/// per-call. High avg = caller passes small buffers (chunked); low
/// avg = caller passes big buffers (monolithic-ish).
pub static READ_STREAM_OUTPUT_BUF_BYTES: AtomicU64 = AtomicU64::new(0);
use super::libdeflate_entry::{DistTable, LitLenTable, HUFFDEC_END_OF_BLOCK, HUFFDEC_EXCEPTIONAL};
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

    /// Return the byte at *logical* position `i` (0 = oldest, `len()-1` =
    /// newest). Used by [`copy_match_windowed`] when a match's source
    /// reaches past `output[0]`. `i` must be < `self.len()`.
    #[inline(always)]
    pub fn byte_at_logical(&self, i: usize) -> u8 {
        debug_assert!(i < self.len(), "logical index {i} >= len {}", self.len());
        // When written ≥ WINDOW_SIZE the oldest byte sits at `buf[head]`
        // (the next-write slot just before the wrap overwrites it). When
        // written < WINDOW_SIZE the oldest is at `buf[0]`. Both cases
        // collapse to: logical i → buf[(head + WINDOW_SIZE - len() + i) %
        // WINDOW_SIZE].
        let idx = (self.head + WINDOW_SIZE - self.len() + i) % WINDOW_SIZE;
        self.buf[idx]
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
/// Owns the input bit reader and a [`SlidingWindow`]. This is the
/// production resumable inflate for the parallel-SM path (the old
/// `session: Vec<u8>`-accumulator `ResumableInflate` was deleted in
/// §5 step 6).
pub struct ResumableInflate2<'a> {
    pub(crate) bits: Bits<'a>,
    pub(crate) window: SlidingWindow,
    pub(crate) block_state: BlockState,
    pub(crate) pending_match: Option<PendingMatch>,
    pub(crate) last_bfinal: bool,
    /// Last `BTYPE` read in `try_enter_next_block`. Exposed via
    /// `btype()` only when `stopped_at == END_OF_BLOCK_HEADER` to match
    /// vendor semantics (`consume_first_decode.rs:2992-2998`).
    pub(crate) last_btype: u8,
    pub(crate) points_to_stop_at: StoppingPoint,
    pub(crate) stopped_at: StoppingPoint,
    pub(crate) encoded_until_bits: usize,
    /// Built when a dynamic-Huffman block is entered (BTYPE=10); reused
    /// across mid-block yields; dropped on EOB / `set_window` / entry
    /// into a non-dynamic block. Invariant: `Some(_)` iff
    /// `block_state == InDynamic`.
    pub(crate) dynamic_tables: Option<(LitLenTable, DistTable)>,
    /// Parallel ISA-L LUT tables for the same dynamic block. Built
    /// alongside `dynamic_tables` so the inner-Huffman migration
    /// (advisor 2026-05-27, "Option D") can drop a new inner loop
    /// using the multi-symbol-pack LUT format without restructuring
    /// data flow. Same lifecycle as `dynamic_tables` — built in
    /// `parse_dynamic_header`, dropped in `finish_current_block`.
    /// Behind cfg(feature = "pure-rust-inflate") only; on
    /// `isal-compression` builds the field is omitted to avoid
    /// pulling the LUT struct into builds that don't need it.
    #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
    pub(crate) dynamic_tables_isal: Option<(
        crate::decompress::parallel::isal_huffman_pure::IsalLitLenCodePure,
        crate::decompress::parallel::isal_lut_bulk::BulkDistDecoder,
    )>,
    /// Set by `reset_for_next_stream` (used between gzip members); the
    /// next `read_stream` call fires `END_OF_STREAM_HEADER` if that
    /// stopping point is configured. Mirrors vendor pattern
    /// (`consume_first_decode.rs:3091-3100`).
    pub(crate) pending_stream_header_stop: bool,
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
            last_btype: 0,
            points_to_stop_at: StoppingPoint::NONE,
            stopped_at: StoppingPoint::NONE,
            encoded_until_bits: until_bits.min(input_bits),
            dynamic_tables: None,
            #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
            dynamic_tables_isal: None,
            pending_stream_header_stop: false,
        })
    }

    /// Seed the sliding window from a chunk's predecessor 32 KiB. Replaces
    /// any prior window contents and clears block state.
    pub fn set_window(&mut self, window: &[u8]) -> Result<()> {
        self.window.seed(window);
        self.block_state = BlockState::AwaitingHeader;
        self.pending_match = None;
        self.last_bfinal = false;
        self.last_btype = 0;
        self.stopped_at = StoppingPoint::NONE;
        self.dynamic_tables = None;
        #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
        {
            self.dynamic_tables_isal = None;
        }
        self.pending_stream_header_stop = false;
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

    /// Reset `stopped_at` so the caller can re-poll without ambiguity.
    /// Vendor parity with `IsalInflateWrapper::clearStop` (isal.hpp).
    pub fn clear_stop(&mut self) {
        self.stopped_at = StoppingPoint::NONE;
    }

    pub fn points_to_stop_at(&self) -> StoppingPoint {
        self.points_to_stop_at
    }

    pub fn bit_position(&self) -> usize {
        self.bits.bit_position()
    }

    /// Vendor parity alias for `bit_position`. Vendor `IsalInflateWrapper`
    /// exposes `tellCompressed()` (isal.hpp:69-74); existing call sites
    /// in `inflate_wrapper.rs` and `gzip_chunk.rs` use this name.
    pub fn tell_compressed(&self) -> usize {
        self.bits.bit_position()
    }

    pub fn encoded_until_bits(&self) -> usize {
        self.encoded_until_bits
    }

    pub fn is_final_block(&self) -> bool {
        self.last_bfinal
    }

    /// Returns `Some(last_btype)` ONLY when stopped at
    /// `END_OF_BLOCK_HEADER`. Matches vendor semantics
    /// (`consume_first_decode.rs:2992-2998`). Used by `gzip_chunk.rs:217`
    /// to detect fixed-Huffman blocks where the chunk can't safely stop.
    pub fn btype(&self) -> Option<u8> {
        if self.stopped_at == StoppingPoint::END_OF_BLOCK_HEADER {
            Some(self.last_btype)
        } else {
            None
        }
    }

    /// True when the deflate stream is fully decoded (BFINAL=1 block's
    /// EOB consumed). Matches vendor
    /// `consume_first_decode.rs:3008-3010`.
    pub fn at_end_of_stream(&self) -> bool {
        matches!(self.block_state, BlockState::Finished)
    }

    /// Bytes of input not yet consumed (`&data[bit_position/8..]`,
    /// byte-aligned). Caller must ensure the bit cursor is byte-aligned
    /// before relying on this — `read_footer_at_current` (in the
    /// wrapper) asserts that. Matches vendor
    /// `consume_first_decode.rs:3012-3019`.
    pub fn remaining_input(&self) -> &'a [u8] {
        let start_byte = self.bits.bit_position() / 8;
        &self.bits.data[start_byte.min(self.bits.data.len())..]
    }

    /// Advance the bit cursor by `n` bytes; widen `encoded_until_bits`
    /// if needed so the next member's body isn't capped by the prior
    /// member's `until_bits`. Mirrors
    /// `consume_first_decode.rs:3021-3027`. Clears `bitbuf`/`bitsleft`
    /// so the next refill loads cleanly (matches the bitbuf-stale
    /// trap pattern from §5 step 2).
    pub fn advance_input(&mut self, n: usize) {
        let cur = self.bits.bit_position() / 8;
        let new_pos = (cur + n).min(self.bits.data.len());
        self.bits.pos = new_pos;
        self.bits.bitbuf = 0;
        self.bits.bitsleft = 0;
        let input_bits = self.bits.data.len() * 8;
        if input_bits > self.encoded_until_bits {
            self.encoded_until_bits = input_bits;
        }
    }

    /// Reset per-member state for the next gzip member in a multi-member
    /// stream. Window is NOT reset (caller will `set_window(&[])` if
    /// they want to discard the predecessor window). Sets
    /// `pending_stream_header_stop` so the next `read_stream` fires
    /// `END_OF_STREAM_HEADER` if requested — matches vendor
    /// `consume_first_decode.rs:3029-3036`.
    pub fn reset_for_next_stream(&mut self) {
        self.block_state = BlockState::AwaitingHeader;
        self.pending_match = None;
        self.last_bfinal = false;
        self.last_btype = 0;
        self.stopped_at = StoppingPoint::NONE;
        self.dynamic_tables = None;
        #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
        {
            self.dynamic_tables_isal = None;
        }
        self.pending_stream_header_stop = true;
    }

    /// `ResumableInflate2` writes directly into the caller's `output`
    /// buffer — there is no internal session accumulator. Returning
    /// `false` here is correct and matches the new architecture's
    /// vendor-faithful sliding-window design. The wrapper API still
    /// exposes this for source compatibility with the old backend
    /// (`session: Vec<u8>` accumulator).
    pub fn session_pending(&self) -> bool {
        false
    }

    /// Drive the decoder into `output`, stopping when:
    ///   (1) `output` is full,
    ///   (2) a requested `StoppingPoint` fires,
    ///   (3) `encoded_until_bits` is reached,
    ///   (4) the final block's BFINAL completes.
    pub fn read_stream(&mut self, output: &mut [u8]) -> Result<InflateStreamResult> {
        // Step 2.5 instrumentation: record per-call shape so production
        // hot-path analysis can distinguish "small-buffer chunked
        // caller" from "large-buffer monolithic caller". Counters are
        // single atomic-add per call (not per iteration) so the hot
        // path pays no per-symbol tax.
        READ_STREAM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        READ_STREAM_OUTPUT_BUF_BYTES
            .fetch_add(output.len() as u64, std::sync::atomic::Ordering::Relaxed);

        self.stopped_at = StoppingPoint::NONE;
        let out_pos_start = 0usize;
        let mut out_pos = out_pos_start;

        // Vendor multi-stream pattern: after `reset_for_next_stream`, the
        // next `read_stream` fires `END_OF_STREAM_HEADER` immediately if
        // requested (zero bytes consumed, zero output written). Mirror
        // of `consume_first_decode.rs:3091-3100`.
        if self.pending_stream_header_stop {
            self.pending_stream_header_stop = false;
            if self
                .points_to_stop_at
                .contains(StoppingPoint::END_OF_STREAM_HEADER)
            {
                self.stopped_at = StoppingPoint::END_OF_STREAM_HEADER;
                return Ok(InflateStreamResult {
                    bytes_written: 0,
                    stopped_at: self.stopped_at,
                    bit_position: self.bits.bit_position(),
                    finished: false,
                });
            }
        }

        loop {
            if out_pos >= output.len() {
                break;
            }

            // No-progress guard: snapshot (out_pos, bit_position,
            // block_state discriminant, pending_match.is_some()) at the
            // top of every iteration. If nothing changes by the bottom,
            // we'd loop forever — break out instead. Catches the class
            // of bug where a sub-decoder yields 0 progress without
            // transitioning state (e.g. body decoder hitting
            // encoded_until_bits while block_state still says InDynamic).
            // Vendor `isal.hpp` doesn't need this because ISA-L's state
            // machine never returns "no progress, no error"; we add it
            // as defensive plumbing per the Opus advisor review of step 4.
            let snap_out_pos = out_pos;
            let snap_bit_pos = self.bits.bit_position();
            let snap_state_disc = std::mem::discriminant(&self.block_state);
            let snap_pending = self.pending_match.is_some();

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
                BlockState::InFixed => {
                    out_pos = resume_decode_fixed_resumable(self, output, out_pos)?;
                }
                BlockState::InDynamic => {
                    out_pos = resume_decode_dynamic_resumable(self, output, out_pos)?;
                }
            }

            if matches!(self.block_state, BlockState::Finished) {
                // Match vendor / OLD wrapper semantics
                // (consume_first_decode.rs:3198-3211): when both
                // END_OF_BLOCK and END_OF_STREAM are requested, the
                // BFINAL block's EOB fires END_OF_BLOCK FIRST. Callers
                // that drive a chunk-by-chunk decoder
                // (`gzip_chunk.rs::decode_chunk_isal_impl`) treat
                // END_OF_BLOCK+`is_final_block=true` as the natural exit
                // for the last block of a single-member stream — they
                // never want to read a footer from the chunk's input
                // slice (which contains only the deflate body, no
                // trailer; the trailer is parsed by `sm_driver` from
                // the outer gzip envelope). The new wrapper firing
                // END_OF_STREAM here would route execution into the
                // footer-read branch of `decode_chunk_isal_impl` and
                // `read_footer_at_current` would return Internal(-1)
                // because the input slice ends at the deflate body.
                // END_OF_STREAM still fires when ONLY END_OF_STREAM is
                // requested (e.g. multi-stream `gzip_chunk` config).
                if self.points_to_stop_at.contains(StoppingPoint::END_OF_BLOCK) {
                    self.stopped_at = StoppingPoint::END_OF_BLOCK;
                } else if self
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

            // No-progress check at loop bottom.
            if out_pos == snap_out_pos
                && self.bits.bit_position() == snap_bit_pos
                && std::mem::discriminant(&self.block_state) == snap_state_disc
                && self.pending_match.is_some() == snap_pending
            {
                // Nothing advanced. Two legitimate cases:
                //   (a) a body decoder observed bit_position >=
                //       encoded_until_bits at top of its loop and
                //       returned without state transition — a CLEAN
                //       stop at a chunk's `until_bits` cap.
                //   (b) bit_position < encoded_until_bits but no body
                //       progress AND no stopping point fired — this is
                //       a contract violation (advisor item D4): a
                //       future change to chunk-boundary logic that
                //       lands `encoded_until_bits` mid-symbol would
                //       silently truncate output here. Convert to a
                //       loud Err.
                //
                // (a) is "we ran out of input under the cap" and the
                // caller's CRC32 verification will surface any
                // truncation downstream.
                let bit_position_under_cap = self.bits.bit_position() < self.encoded_until_bits;
                let mid_block = !matches!(
                    self.block_state,
                    BlockState::AwaitingHeader | BlockState::Finished
                );
                let no_stop_fired = self.stopped_at == StoppingPoint::NONE;
                if bit_position_under_cap && mid_block && no_stop_fired {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "ResumableInflate2 no-progress mid-body: \
                             bit_position={} cap={} block_state={:?} — \
                             a sub-decoder yielded without transitioning \
                             state; refusing to silently truncate (advisor D4)",
                            self.bits.bit_position(),
                            self.encoded_until_bits,
                            self.block_state
                        ),
                    ));
                }
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
        let bytes_written = out_pos - out_pos_start;
        READ_STREAM_BYTES_OUT.fetch_add(bytes_written as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(InflateStreamResult {
            bytes_written,
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
        let btype = ((header >> 1) & 0b11) as u8;
        self.last_btype = btype;
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
                // Parse the dynamic-Huffman header atomically and stash
                // the built tables on `self`. Vendor `isal.hpp:263-272`
                // treats `readHeader` as atomic.
                let tables = parse_dynamic_header(&mut self.bits)?;
                // CLAUDE.md "no fallbacks" + vendor GzipReader contract:
                // parallel-SM chunk boundaries land at block boundaries,
                // so a dynamic header that straddles `encoded_until_bits`
                // is a contract violation. Surface loudly rather than
                // leaving the body decoder in a state where it sees
                // bit_position past cap and returns Ok(0) forever (the
                // infinite-loop bug Opus advisor caught).
                if self.bits.bit_position() > self.encoded_until_bits {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "Dynamic header straddled encoded_until_bits cap: \
                             bit_position={} > cap={}. Chunk boundary contract \
                             expects boundaries at block boundaries.",
                            self.bits.bit_position(),
                            self.encoded_until_bits
                        ),
                    ));
                }
                self.dynamic_tables = Some(tables);
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
        // Invariant: dynamic_tables Some iff block_state == InDynamic.
        // Block is ending — drop tables. If the NEXT block is also
        // dynamic, `try_enter_next_block`'s BTYPE=2 arm rebuilds them.
        self.dynamic_tables = None;
        #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
        {
            self.dynamic_tables_isal = None;
        }
        if self.last_bfinal {
            // Per RFC 1952 the gzip footer is byte-aligned; the encoder
            // pads the last DEFLATE block's bit stream with zero bits up
            // to the next byte boundary. EOB symbols are Huffman codes
            // and rarely land on a byte boundary themselves, so without
            // this alignment ~7 of every 8 streams would read a footer
            // 1-7 bits early — Opus advisor flagged this as the highest-
            // risk silent-corruption path during the wrapper swap audit.
            // Matches vendor `consume_first_decode.rs:3204` semantics.
            self.bits.align_to_byte();
            self.block_state = BlockState::Finished;
        } else {
            self.block_state = BlockState::AwaitingHeader;
        }
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

/// Resumable BTYPE=01 (fixed Huffman): static tables, mid-block yield.
///
/// Mirrors the generic loop of `decode_huffman_libdeflate_style`
/// (`consume_first_decode.rs:1764-1841`) but with two differences:
///
/// 1. **Yield instead of error on output overflow.** When `out_pos`
///    fills mid-block, return `Ok(out_pos)` (no `WriteZero`). The bit
///    reader has already consumed everything up to the LAST completed
///    symbol; the next call re-enters here at the next iteration. For a
///    match that didn't fully fit, [`copy_match_windowed`] stores a
///    [`PendingMatch`] on `state` and the outer `read_stream` finishes
///    that match before re-entering this function.
///
/// 2. **Match copies go through [`copy_match_windowed`].** Matches
///    whose distance reaches past `output[0]` source bytes from the
///    [`SlidingWindow`].
///
/// No fastloop here — the loop adds a yield check at the top of every
/// iteration, which precludes the FASTLOOP_MARGIN bounds-skipping trick.
/// The plan's §5 Tier 2/3 bench gates may motivate adding a fast inner
/// loop later when output has > 256 bytes of headroom and no pending
/// match.
pub fn resume_decode_fixed_resumable(
    state: &mut ResumableInflate2<'_>,
    output: &mut [u8],
    out_pos: usize,
) -> Result<usize> {
    let (litlen, dist) = crate::decompress::inflate::libdeflate_decode::get_fixed_tables();
    decode_huffman_body_resumable(state, output, out_pos, litlen, dist)
}

/// Resumable BTYPE=02 (dynamic Huffman). The block header was parsed
/// atomically in `try_enter_next_block` (BTYPE=2 arm); tables live on
/// `state.dynamic_tables` and persist across mid-block yields. On EOB,
/// `finish_current_block` drops the tables.
pub fn resume_decode_dynamic_resumable(
    state: &mut ResumableInflate2<'_>,
    output: &mut [u8],
    out_pos: usize,
) -> Result<usize> {
    debug_assert!(
        state.dynamic_tables.is_some(),
        "dynamic_tables invariant: must be Some(_) iff block_state == InDynamic"
    );
    // Standard "borrow one field while mutating another" pattern:
    // take the tables out of `state`, drive the body decoder (which
    // mutably borrows `state` for the window/bit reader/pending_match),
    // then put them back on return. Avoids the borrow checker fight
    // around aliased mutable access to `state` while holding references
    // to its `dynamic_tables` field. If the body decoder transitions
    // out of InDynamic (EOB), `finish_current_block` has already
    // dropped tables to None — restoring our `take`d copy would
    // violate the invariant, so check before restoring.
    let tables = state.dynamic_tables.take().ok_or_else(|| {
        Error::other("dynamic_tables invariant violated: None at entry to InDynamic body")
    })?;
    let result = decode_huffman_body_resumable(state, output, out_pos, &tables.0, &tables.1);
    // Restore only if still in InDynamic (i.e., didn't complete the
    // block this call). `finish_current_block` runs on EOB and sets
    // dynamic_tables = None already.
    if matches!(state.block_state, BlockState::InDynamic) {
        state.dynamic_tables = Some(tables);
    }
    result
}

/// Shared inner loop for fixed + dynamic Huffman blocks. Identical to
/// the generic loop of `decode_huffman_libdeflate_style` (vendor
/// `consume_first_decode.rs:1764-1841`) except:
///   - yield at top of loop when output is full (no `WriteZero` error),
///   - matches go through `copy_match_windowed` (window-aware + yield),
///   - EOB transitions the parent state machine via `finish_current_block`.
///
/// No fastloop: every iteration has a yield check, which precludes
/// FASTLOOP_MARGIN bounds-skipping. Documented at the call site as the
/// §5 Tier-2/3 perf gate target.
fn decode_huffman_body_resumable(
    state: &mut ResumableInflate2<'_>,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    // SIMD multi-literal fastloop was tried at commit ca52389 and
    // regressed the inflate bench from 334 → 284 MB/s (-15%) plus
    // the silesia E2E test 2.79× → 3.06×. Suspected cause: silesia
    // data has too few literal clusters for `vector_huffman::
    // decode_multi_literals` lookahead to pay back the per-iteration
    // overhead. Reverted to the original scalar loop below; the
    // counters left in place (`MULTI_LITERAL_*`, `BODY_RESUMABLE_*`)
    // remain reachable for future microbenches.
    //
    // The vendor-faithful win is elsewhere — likely BMI2 pext bit
    // extraction or specializing on FIXED-Huffman static tables.
    // See `plans/pure-rust-perf.md` Phase B work-item #1(a).
    // Vendor refill threshold pattern (mirror of
    // `consume_first_decode::decode_huffman_libdeflate_style:852-863`):
    // a single match consumes at most 48 bits — 20 for litlen (12 table +
    // 3 subtable + 5 length-extra) + 28 for dist (9 table + 6 subtable +
    // 13 dist-extra). One refill brings `bitsleft` into [56, 63], which
    // is enough for one full match. Refill only when below the threshold;
    // the prior unconditional per-iteration refill (and the extra refill
    // before the dist lookup) was wasted work on the hot path.
    const REFILL_THRESHOLD: u8 = 48;

    // Lever B1 (advisor sign-off review): lift bits-reader fields into
    // stack locals so LLVM can promote them into registers across loop
    // iterations. Vendor counterpart: `consume_first_decode.rs:743-747`
    // (`let mut bitbuf = bits.bitbuf;` etc) with writebacks at every
    // function exit (`:1036-1039` and parallel sites).
    //
    // Without this, every `state.bits.bitbuf` access through the
    // `&mut state` reference re-reads from memory because the compiler
    // can't prove `output[i] = ...` writes don't alias `state.bits`.
    // Locals decouple them. `copy_match_windowed` doesn't touch
    // `state.bits` (it only reads `state.window` and writes
    // `state.pending_match`), so we don't need to write back / re-lift
    // around that call — only at return sites.
    let mut bitbuf: u64 = state.bits.bitbuf;
    let mut bitsleft: u32 = state.bits.bitsleft;
    let mut in_pos: usize = state.bits.pos;
    let in_data: &[u8] = state.bits.data;
    let in_data_len: usize = in_data.len();
    let in_data_ptr: *const u8 = in_data.as_ptr();
    let encoded_until_bits: usize = state.encoded_until_bits;

    // Lever T0 (tight-Huffman-decoder plan): lift output base pointer
    // and length to locals so literal writes go through raw-pointer
    // arithmetic instead of slice-indexing. The top-of-loop yield
    // check `out_pos >= output.len()` is the safety boundary that
    // makes the unsafe writes sound — when control enters the body,
    // `out_pos < output_len` is invariant. Vendor counterpart:
    // `consume_first_decode.rs:738-739` lifts `out_ptr`/`out_end`,
    // all literal writes use `(out_ptr.add(out_pos) as *mut u8).write(...)`
    // (`:1021-1024, :1057-1058`).
    let output_ptr: *mut u8 = output.as_mut_ptr();
    let output_len: usize = output.len();

    // Mirror of `Bits::bit_position` semantics
    // (`consume_first_decode.rs:237-240`): bit position of the NEXT bit
    // to consume = (in_pos in bytes × 8) − unread-bits-in-buffer.
    #[inline(always)]
    fn bit_position_of(bitsleft: u32, in_pos: usize) -> usize {
        let unread = (bitsleft as u8).min(64) as usize;
        in_pos.saturating_mul(8).saturating_sub(unread)
    }

    // Mirror of `Bits::refill`
    // (`consume_first_decode.rs:245-264`): branchless 8-byte unaligned
    // load when room, byte-by-byte slow path otherwise. Identical
    // arithmetic and underflow handling.
    macro_rules! refill_local {
        () => {{
            let mut bits_u8 = bitsleft as u8;
            if bits_u8 > 64 {
                bitbuf = 0;
                bits_u8 = 0;
            }
            if in_pos + 8 <= in_data_len {
                let word = unsafe { (in_data_ptr.add(in_pos) as *const u64).read_unaligned() };
                let word = u64::from_le(word);
                bitbuf |= word << bits_u8;
                in_pos += (7 - ((bits_u8 >> 3) & 7)) as usize;
                bitsleft = (bits_u8 as u32) | 56;
            } else {
                // Slow path: byte-by-byte until 56 bits filled or EOF.
                let mut b = bits_u8;
                while b <= 56 && in_pos < in_data_len {
                    bitbuf |= (in_data[in_pos] as u64) << b;
                    in_pos += 1;
                    b = b.wrapping_add(8);
                }
                bitsleft = b as u32;
            }
        }};
    }

    // Sync locals back into state.bits. Call before any return that
    // expects subsequent calls to observe the consumed bits.
    macro_rules! writeback_bits {
        () => {{
            state.bits.bitbuf = bitbuf;
            state.bits.bitsleft = bitsleft;
            state.bits.pos = in_pos;
        }};
    }

    // Lever B2 (advisor second-pass): PRELOAD the first entry before
    // entering the loop, and re-load (`huffdec!`-style) at the END of
    // each iteration's branch. This overlaps the table-lookup latency
    // with the literal emit / match copy. Vendor counterpart:
    // `consume_first_decode.rs:842-844` (preload), `:829-840`
    // (`huffdec!` macro), `:877-879, :889, :903, :909, :922` (per-branch
    // re-lookup at tail).
    //
    // Yield-contract preservation: on yield (output full / cap reached),
    // we return BEFORE consuming the preloaded entry's bits. On resume,
    // the function re-runs the PRELOAD step at the top, so any preloaded
    // entry from the prior call is discarded. The bit reader state is
    // writeback'd before every return, so the re-preload reads the same
    // `bitbuf` value the prior preload did.
    if (bitsleft as u8) < REFILL_THRESHOLD {
        refill_local!();
    }
    let mut entry = litlen.lookup(bitbuf);

    // Lever T4 (tight-Huffman-decoder plan): FASTLOOP yield-elide.
    // When both buffers have safety margin, run a tight inner loop that
    // omits the per-iteration yield checks. Vendor pattern:
    // `consume_first_decode.rs:752` (`in_fastloop_end = in_data.len() - 32`),
    // `:858-859` (`while in_pos < in_fastloop_end && out_pos +
    // FASTLOOP_MARGIN <= out_end`).
    //
    // FASTLOOP_MARGIN bounds the max bytes a single iteration may write
    // to output: max match length is 258, plus a small safety margin.
    // 320 matches vendor.
    const FASTLOOP_MARGIN: usize = 320;
    // Refill_local! reads up to 8 bytes ahead of in_pos; we leave 32
    // bytes (4 refills' worth) headroom so the fastloop body never
    // checks bounds.
    const FASTLOOP_INPUT_TAIL: usize = 32;

    // The symbol decode body. Used inside the fastloop (no yield checks)
    // and inside the safe loop (yield checks above). `continue` advances
    // to the enclosing loop top; `return` exits the function. The body
    // assumes `entry` is preloaded and corresponds to the current `bitbuf`.
    macro_rules! decode_one_symbol {
        () => {{
            // saved_bitbuf for length / dist-extra extraction. Captured
            // BEFORE consume so the same bits the entry was decoded from
            // are available for extra-bit reads.
            let mut saved_bitbuf = bitbuf;
            let mut raw = entry.raw();

            bitbuf >>= raw as u8;
            bitsleft = bitsleft.wrapping_sub(raw & 0x1F);

            // LITERAL FAST PATH (bit 31). Vendor: `:870-1030`.
            if (raw as i32) < 0 {
                // Lever T3 (tight-Huffman-decoder plan): multi-literal
                // lookahead. After emitting the first literal, peek at
                // the preloaded `bitbuf` to see if the NEXT entry is
                // also a literal. If yes, consume+emit it inline. Try
                // up to 3 extra literals (4 total) when bit budget
                // allows. Vendor pattern: `:870-1030` packs up to 8.
                //
                // Bit budget: each main-table lookup consumes up to
                // TABLE_BITS=12 bits. Starting from post-refill bitsleft
                // ∈ [56, 63] (T1's literal-tail refill ensured this on
                // the previous iter), we can absorb 4 lookups (48 bits)
                // before needing another refill. We don't multi-batch
                // subtable literals — those go through the EXCEPTIONAL
                // path with a different consume size.
                //
                // Falsification record: commit `ca52389` regressed -15%
                // at the pre-PRELOAD loop shape. CLAUDE.md update
                // 2026-05-27 explicitly allows the re-attempt; this
                // implementation is structurally different (PRELOAD +
                // register locals + raw-pointer writes are all in place).
                unsafe {
                    output_ptr.add(out_pos).write(entry.literal_value());
                }
                out_pos += 1;

                // Speculative lookup for the next entry. Two outcomes:
                //   (A) Next is literal → consume + emit it; try to
                //       chain up to 4 literals. After the chain, refill
                //       + lookup once for the iter-after-batch entry.
                //   (B) Next is NOT literal → use this entry as the
                //       NEXT iteration's `entry`. No re-lookup needed
                //       (this is the key vendor trick that avoids the
                //       wasted-lookup regression: the speculative
                //       lookup ALWAYS yields the next iter's entry,
                //       whether or not we batched).
                //
                // Output bound is critical for the SAFE-loop path: the
                // FASTLOOP guarantees `out_pos + 320 <= output_len` so
                // these checks are redundant when fastloop fires (and
                // LLVM proves it on inlining). In the safe loop (1-byte
                // output buffers exist in unit tests), the check
                // prevents UB.
                //
                // T3 BUG FIX 2026-05-27: when we carry forward an entry
                // (entry = e_next; continue), the next iter's consume
                // requires up to 45 bits (length codeword + extras +
                // dist codeword + extras). Post-batch bitsleft can be
                // as low as 8 (after 4 literals from a 56-bit start).
                // We MUST refill before carrying, else the next consume
                // reads garbage bits past the valid window. Each carry
                // site below does `if (bitsleft as u8) < 56 { refill }`
                // before the carry to guarantee sufficient bits.
                //
                // Caught by the silesia unit test (real-world deflate);
                // missed by the 729-case synthetic differential because
                // synthetic fixtures don't produce the multi-literal-
                // followed-by-length pattern frequently enough to trip
                // the bug deterministically.
                if (bitsleft as u8) >= REFILL_THRESHOLD {
                    let mut e_next = litlen.lookup(bitbuf);
                    let mut r_next = e_next.raw();

                    // 2nd literal
                    if (r_next as i32) < 0 && out_pos < output_len {
                        bitbuf >>= r_next as u8;
                        bitsleft = bitsleft.wrapping_sub(r_next & 0x1F);
                        unsafe {
                            output_ptr.add(out_pos).write(e_next.literal_value());
                        }
                        out_pos += 1;

                        // 3rd literal
                        if (bitsleft as u8) >= 24 {
                            e_next = litlen.lookup(bitbuf);
                            r_next = e_next.raw();
                            if (r_next as i32) < 0 && out_pos < output_len {
                                bitbuf >>= r_next as u8;
                                bitsleft = bitsleft.wrapping_sub(r_next & 0x1F);
                                unsafe {
                                    output_ptr.add(out_pos).write(e_next.literal_value());
                                }
                                out_pos += 1;

                                // 4th literal
                                if (bitsleft as u8) >= 12 {
                                    e_next = litlen.lookup(bitbuf);
                                    r_next = e_next.raw();
                                    if (r_next as i32) < 0 && out_pos < output_len {
                                        bitbuf >>= r_next as u8;
                                        bitsleft = bitsleft.wrapping_sub(r_next & 0x1F);
                                        unsafe {
                                            output_ptr.add(out_pos).write(e_next.literal_value());
                                        }
                                        out_pos += 1;
                                        // Hit 4-literal cap. Refill +
                                        // lookup for next iter; don't
                                        // carry e_next.
                                        if (bitsleft as u8) < REFILL_THRESHOLD {
                                            refill_local!();
                                        }
                                        entry = litlen.lookup(bitbuf);
                                        continue;
                                    }
                                    // 4th was non-literal — carry as
                                    // next iter's entry. REFILL FIRST
                                    // so next iter has bits to consume
                                    // (T3 bug fix).
                                    if (bitsleft as u8) < REFILL_THRESHOLD {
                                        refill_local!();
                                    }
                                    entry = e_next;
                                    continue;
                                }
                                // 3rd was non-literal — carry. Refill.
                                if (bitsleft as u8) < REFILL_THRESHOLD {
                                    refill_local!();
                                }
                                entry = e_next;
                                continue;
                            }
                            // 3rd-position lookup was non-literal —
                            // carry. Refill.
                            if (bitsleft as u8) < REFILL_THRESHOLD {
                                refill_local!();
                            }
                            entry = e_next;
                            continue;
                        }
                        // bitsleft too low for 3rd lookup. Refill +
                        // lookup for next iter.
                        if (bitsleft as u8) < REFILL_THRESHOLD {
                            refill_local!();
                        }
                        entry = litlen.lookup(bitbuf);
                        continue;
                    }
                    // 2nd was non-literal (or no output room). Carry as
                    // next iter's entry — saves a lookup vs the
                    // pre-T3 path. Refill first.
                    if (bitsleft as u8) < REFILL_THRESHOLD {
                        refill_local!();
                    }
                    entry = e_next;
                    continue;
                }

                // bitsleft too low even for speculative lookup. Refill,
                // then preload as usual. Same shape as single-literal
                // path pre-T3.
                if (bitsleft as u8) < REFILL_THRESHOLD {
                    refill_local!();
                }
                entry = litlen.lookup(bitbuf);
                continue;
            }

            // EXCEPTIONAL PATH (bit 15: subtable_ptr OR end_of_block).
            // Vendor: `:1032-1068`.
            if (raw & HUFFDEC_EXCEPTIONAL) != 0 {
                if (raw & HUFFDEC_END_OF_BLOCK) != 0 {
                    writeback_bits!();
                    state.finish_current_block();
                    return Ok(out_pos);
                }
                // SUBTABLE
                entry = litlen.lookup_subtable_direct(entry, bitbuf);
                saved_bitbuf = bitbuf;
                raw = entry.raw();
                bitbuf >>= raw as u8;
                bitsleft = bitsleft.wrapping_sub(raw & 0x1F);

                if (raw as i32) < 0 {
                    unsafe {
                        output_ptr.add(out_pos).write(entry.literal_value());
                    }
                    out_pos += 1;
                    if (bitsleft as u8) < REFILL_THRESHOLD {
                        refill_local!();
                    }
                    entry = litlen.lookup(bitbuf);
                    continue;
                }
                if (raw & HUFFDEC_END_OF_BLOCK) != 0 {
                    writeback_bits!();
                    state.finish_current_block();
                    return Ok(out_pos);
                }
                // Fall through to LENGTH path.
            }

            // LENGTH+DISTANCE path.
            let length = entry.decode_length(saved_bitbuf);
            let dist_saved = bitbuf;
            let mut dist_entry = dist.lookup(dist_saved);
            if dist_entry.is_subtable_ptr() {
                bitbuf >>= DistTable::TABLE_BITS;
                bitsleft = bitsleft.wrapping_sub(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable_direct(dist_entry, bitbuf);
            }
            let dist_extra_saved = bitbuf;
            let dist_raw = dist_entry.raw();
            bitbuf >>= dist_raw as u8;
            bitsleft = bitsleft.wrapping_sub(dist_raw & 0x1F);
            let distance = dist_entry.decode_distance(dist_extra_saved);

            // copy_match_windowed reads `state.window` and writes
            // `state.pending_match`. It does NOT touch `state.bits`.
            let bits_value_snapshot = if cfg!(debug_assertions) {
                Some((state.bits.bitbuf, state.bits.bitsleft, state.bits.pos))
            } else {
                None
            };
            out_pos = copy_match_windowed(state, output, out_pos, distance, length)?;
            debug_assert_eq!(
                bits_value_snapshot,
                Some((state.bits.bitbuf, state.bits.bitsleft, state.bits.pos)),
                "decode_huffman_body_resumable B1 contract: copy_match_windowed \
                 must not mutate state.bits.{{bitbuf,bitsleft,pos}}"
            );
            if state.pending_match.is_some() {
                debug_assert_eq!(out_pos, output.len());
                writeback_bits!();
                return Ok(out_pos);
            }

            // Refill + preload next entry.
            if (bitsleft as u8) < REFILL_THRESHOLD {
                refill_local!();
            }
            entry = litlen.lookup(bitbuf);
        }};
    }

    // Outer loop alternates: try the fastloop, then run safe iterations
    // until either we can re-enter the fastloop OR a yield/EOB fires.
    loop {
        // FASTLOOP entry condition. Bounds:
        //   - in_pos < in_data_len - 32: refill_local! 8-byte loads safe
        //   - in_pos * 8 < encoded_until_bits: won't decode past cap
        //   - out_pos < output_len - FASTLOOP_MARGIN: any single iter's
        //     write fits (max match 258 + literals)
        //
        // `cap_byte_floor = encoded_until_bits / 8` is a conservative
        // floor: if in_pos < cap_byte_floor, in_pos*8 < cap_byte_floor*8
        // ≤ encoded_until_bits, so bit_position_of(...) ≤ in_pos*8 <
        // cap. The bound is strict enough that the fastloop body can
        // omit the cap check entirely.
        let cap_byte_floor = encoded_until_bits / 8;
        let in_fl_end = in_data_len
            .saturating_sub(FASTLOOP_INPUT_TAIL)
            .min(cap_byte_floor);
        let out_fl_end = output_len.saturating_sub(FASTLOOP_MARGIN);

        if in_pos < in_fl_end && out_pos < out_fl_end {
            BODY_RESUMABLE_FASTLOOP_ENTERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            while in_pos < in_fl_end && out_pos < out_fl_end {
                decode_one_symbol!();
            }
            // Fastloop exited because bounds broke. Re-evaluate yield
            // conditions at the safe-loop top.
            continue;
        }

        // SAFE LOOP iteration: per-iter yield checks then ONE symbol.
        // SAFETY boundary: `out_pos < output_len` after the next check;
        // the unsafe literal writes inside `decode_one_symbol!` rely on
        // this invariant.
        if out_pos >= output_len {
            writeback_bits!();
            return Ok(out_pos);
        }
        if bit_position_of(bitsleft, in_pos) >= encoded_until_bits {
            writeback_bits!();
            return Ok(out_pos);
        }

        // One safe iteration via the shared body macro. The macro's
        // `continue` returns to the outer `loop` top, which re-checks
        // the FASTLOOP entry conditions before running another safe
        // iteration.
        decode_one_symbol!();
    }
}

/// Parse a dynamic-Huffman block header atomically and return the built
/// litlen + distance tables. Mirror of the header-parse portion of the
/// existing non-resumable `decode_dynamic`
/// (`consume_first_decode.rs:2095-2188`). Atomicity matches vendor
/// `isal.hpp:263-272`: on input exhaustion we'd ideally return a
/// "defer" signal, but the parallel-SM chunk contract (boundaries at
/// block boundaries) guarantees the entire header always fits. A
/// truncated header surfaces as `Err(UnexpectedEof)`.
fn parse_dynamic_header(bits: &mut Bits) -> Result<(LitLenTable, DistTable)> {
    // HLIT (5) + HDIST (5) + HCLEN (4) = 14 bits.
    if bits.available() < 14 {
        bits.refill();
    }
    let hlit = (bits.peek() & 0x1F) as usize + 257;
    bits.consume(5);
    let hdist = (bits.peek() & 0x1F) as usize + 1;
    bits.consume(5);
    let hclen = (bits.peek() & 0xF) as usize + 4;
    bits.consume(4);

    // Code-length code lengths in the canonical permutation order.
    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut code_length_lengths = [0u8; 19];
    for i in 0..hclen {
        if bits.available() < 3 {
            bits.refill();
        }
        code_length_lengths[CODE_LENGTH_ORDER[i]] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }

    let cl_table = build_code_length_table(&code_length_lengths)?;

    // Decode the literal/length+distance code lengths via the code-length
    // Huffman code. Symbols 0..15 are literal lengths; 16/17/18 are
    // run-length markers (repeat prev / zero-run-short / zero-run-long).
    //
    // Stack buffer to avoid the heap allocation the non-resumable path
    // does on every block (`vec![0u8; hlit + hdist]`,
    // consume_first_decode.rs:2126). 286 + 30 = 316 is the RFC max.
    let mut all_lengths = [0u8; 320];
    let total_lengths = hlit + hdist;
    if total_lengths > all_lengths.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Dynamic header: hlit+hdist {total_lengths} exceeds RFC max 316"),
        ));
    }
    let mut i = 0usize;
    while i < total_lengths {
        // Need ≤7 bits for code + up to 7 bits of run-length extra = 14 max.
        if bits.available() < 15 {
            bits.refill();
        }
        let entry = cl_table[(bits.peek() & 0x7F) as usize];
        let symbol = (entry >> 8) as u8;
        let len = (entry & 0xFF) as u8;
        if len == 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Dynamic header: undefined code-length code",
            ));
        }
        bits.consume(len as u32);
        match symbol {
            0..=15 => {
                all_lengths[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "Dynamic header: repeat-prev at position 0",
                    ));
                }
                let repeat = 3 + (bits.peek() & 0x3) as usize;
                bits.consume(2);
                let val = all_lengths[i - 1];
                let take = repeat.min(total_lengths - i);
                for _ in 0..take {
                    all_lengths[i] = val;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + (bits.peek() & 0x7) as usize;
                bits.consume(3);
                let take = repeat.min(total_lengths - i);
                for _ in 0..take {
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                let take = repeat.min(total_lengths - i);
                for _ in 0..take {
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Dynamic header: invalid code-length symbol",
                ));
            }
        }
    }

    let litlen_lengths = &all_lengths[..hlit];
    let dist_lengths = &all_lengths[hlit..total_lengths];

    let litlen_table = LitLenTable::build(litlen_lengths).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            "Dynamic header: failed to build litlen table",
        )
    })?;
    let dist_table = DistTable::build(dist_lengths).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            "Dynamic header: failed to build dist table",
        )
    })?;
    Ok((litlen_table, dist_table))
}

/// Build IsalLitLenCodePure + BulkDistDecoder from the raw code lengths of
/// a dynamic-Huffman block. The lens are the same arrays parse_dynamic_header
/// uses internally; this helper accepts them directly so the caller doesn't
/// have to re-parse the header. Used by the Option-D inner-Huffman migration
/// (advisor 2026-05-27): when the new inner loop is active, every dynamic
/// block builds both LUT formats. The libdeflate format will be removed
/// in a follow-up commit once the ISA-L inner loop is verified on neurotic.
#[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
pub(crate) fn build_isal_dynamic_tables(
    litlen_lengths: &[u8],
    dist_lengths: &[u8],
) -> Result<(
    crate::decompress::parallel::isal_huffman_pure::IsalLitLenCodePure,
    crate::decompress::parallel::isal_lut_bulk::BulkDistDecoder,
)> {
    use crate::decompress::parallel::isal_huffman_pure::{IsalLitLenCodePure, LIT_LEN};
    use crate::decompress::parallel::isal_lut_bulk::BulkDistDecoder;

    let mut litlen = IsalLitLenCodePure::new_empty();
    // The ISA-L LUT builder accepts up to 288 entries (RFC reserved 286/287
    // included). Dynamic blocks have hlit ≤ 286; pad with zeros.
    let mut litlen_padded = [0u8; LIT_LEN];
    litlen_padded[..litlen_lengths.len()].copy_from_slice(litlen_lengths);
    if !litlen.rebuild_from(&litlen_padded) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "ISA-L dynamic header: invalid litlen code lengths",
        ));
    }
    let mut dist = BulkDistDecoder::new();
    // BulkDistDecoder is sized 32 (RFC reserved 30/31); dist_lengths ≤ 30.
    let mut dist_padded = [0u8; 32];
    dist_padded[..dist_lengths.len()].copy_from_slice(dist_lengths);
    let err = dist.initialize_from_lengths(&dist_padded, false);
    if err != crate::decompress::parallel::error::Error::None {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "ISA-L dynamic header: invalid dist code lengths",
        ));
    }
    Ok((litlen, dist))
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
    state: &mut ResumableInflate2<'_>,
    output: &mut [u8],
    out_pos: usize,
    distance: u32,
    length: u32,
) -> Result<usize> {
    let dist = distance as usize;
    let len = length as usize;
    let window_len = state.window.len();
    let total_history = window_len + out_pos;

    if dist == 0 || dist > total_history {
        // Clear any pending state so a recover-and-retry doesn't loop.
        state.pending_match = None;
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "Invalid distance {distance} at out_pos {out_pos} (history bytes: {total_history})"
            ),
        ));
    }

    let remaining_in_buf = output.len() - out_pos;
    let copy_n = len.min(remaining_in_buf);

    // Fast path: the entire match source lives in `output[out_pos - dist..]`
    // (no window touch) AND there's enough output headroom to absorb
    // `copy_match_fast`'s unconditional 40-byte unrolled write block. Both
    // conditions are necessary:
    //   - `dist <= out_pos` means the source bytes are all in `output`.
    //   - `copy_n == len` means we're not yielding mid-copy; if we yield
    //     we'd need to record bytes-written correctly which complicates
    //     the fast path (the slow path's `PendingMatch` yield is fine).
    //   - `output.len() - out_pos >= len + COPY_FAST_MARGIN` guarantees
    //     `copy_match_fast`'s unconditional past-the-end writes don't
    //     run off the slice. Tight bound is 40 bytes (5×8-byte unrolled
    //     words for the dist>=8 path) + 8 bytes for the final unaligned
    //     u64 store = 48. Vendor `consume_first_decode.rs:395-397`
    //     documents the "up to 40 bytes beyond length" overshoot.
    //
    // This is a vendor-faithful port: libdeflate's FASTLOOP_OUTPUT_MARGIN
    // serves the same purpose; rapidgzip / ISA-L use the same pattern in
    // their bulk-decode hot loops. The slow per-byte path below remains
    // for window-touch cases and tail cases — same shape as
    // `consume_first_decode::copy_match_fast` vs `copy_match_safe`.
    const COPY_FAST_MARGIN: usize = 48;
    if dist <= out_pos && copy_n == len && output.len() - out_pos >= len + COPY_FAST_MARGIN {
        let new_pos = crate::decompress::inflate::consume_first_decode::copy_match_fast(
            output, out_pos, distance, length,
        );
        // copy_match_fast returns `out_pos + len`; the caller's contract
        // is the same so we mirror PendingMatch=None.
        state.pending_match = None;
        return Ok(new_pos);
    }

    // Slow path: window touch (`dist > out_pos`), insufficient margin, or
    // yielding mid-copy. Per-byte loop is the safe baseline; correctness
    // first.
    //
    // Byte at *logical* history position `(total_history - dist + i)`:
    //   if pos < window_len: window.byte_at_logical(pos)
    //   else                : output[pos - window_len]
    //
    // LZ77 overlap (dist < len) is handled naturally by the per-byte
    // loop because once we write to `output[out_pos + j]`, a subsequent
    // read at logical position (window_len + out_pos + j) sees the
    // just-written byte via the `output` branch.
    let start_logical = total_history - dist;
    for i in 0..copy_n {
        let src_logical = start_logical + i;
        let byte = if src_logical < window_len {
            state.window.byte_at_logical(src_logical)
        } else {
            output[src_logical - window_len]
        };
        output[out_pos + i] = byte;
    }

    if copy_n < len {
        // Output filled mid-copy — stash the rest as PendingMatch. The
        // distance stays the same; `length_remaining` is the bytes still
        // unwritten. The outer `read_stream` calls us again on resume
        // before re-entering the block decoder.
        state.pending_match = Some(PendingMatch {
            distance,
            length_remaining: (len - copy_n) as u32,
        });
    } else {
        state.pending_match = None;
    }
    Ok(out_pos + copy_n)
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

    // (The step-4-pending sentinel test was deleted now that dynamic
    // Huffman is wired. Tests below cover real dynamic-block behavior.)

    // ---- Step 3: fixed-Huffman + window-stitched match copy ---------------

    /// Compress `payload` as raw DEFLATE at the requested level. Used as
    /// an oracle for fixed-Huffman blocks (flate2 picks fixed for short
    /// inputs and dynamic for longer ones; we craft inputs that hit
    /// fixed via small-payload + level=1 + Z_FIXED-like behavior).
    fn raw_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    fn decode_via_resumable_in_chunks(stream: &[u8], chunk: usize) -> Vec<u8> {
        let mut inflate = ResumableInflate2::with_until_bits(stream, 0, stream.len() * 8).unwrap();
        let mut collected = Vec::new();
        loop {
            let mut output = vec![0u8; chunk];
            let r = inflate.read_stream(&mut output).unwrap();
            collected.extend_from_slice(&output[..r.bytes_written]);
            if r.finished {
                break;
            }
            assert!(
                r.bytes_written > 0,
                "no progress: chunk={chunk}, collected={}",
                collected.len()
            );
        }
        collected
    }

    #[test]
    fn fixed_block_roundtrip_with_large_output() {
        let payload = b"hello world hello world hello world";
        let stream = raw_deflate(payload, 9);
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream.len() * 8).unwrap();
        let mut output = vec![0u8; 256];
        let r = inflate.read_stream(&mut output).unwrap();
        assert!(r.finished);
        assert_eq!(&output[..r.bytes_written], payload);
    }

    #[test]
    fn fixed_block_yields_with_tiny_output_chunks() {
        // Repetitive payload → lots of matches → exercises copy_match_windowed
        // mid-buffer yield AND the per-byte yield at top of loop.
        let payload = b"abracadabra abracadabra abracadabra abracadabra abracadabra".repeat(8);
        let stream = raw_deflate(&payload, 9);
        // Sanity: this should fit in one DEFLATE stream; we don't assert
        // BTYPE because flate2's choice is implementation-dependent, but
        // either way both code paths must roundtrip.
        let got = decode_via_resumable_in_chunks(&stream, 7);
        assert_eq!(got, payload);
    }

    #[test]
    fn fixed_block_yields_with_chunk_equal_to_match_length() {
        // Output buffer is intentionally small (5 bytes) so a long
        // match (which can be up to 258 bytes) MUST be split via
        // PendingMatch across multiple read_stream calls. Catches a
        // PendingMatch save/restore bug.
        let payload = b"ABCDEFGHIJ".repeat(50); // 500 bytes; many matches.
        let stream = raw_deflate(&payload, 9);
        let got = decode_via_resumable_in_chunks(&stream, 5);
        assert_eq!(got, payload);
    }

    #[test]
    fn multi_block_deflate_roundtrip() {
        // A long-enough payload forces flate2 to emit multiple DEFLATE
        // blocks (the encoder splits when its internal buffer fills),
        // exercising both the AwaitingHeader → InFixed AND → InDynamic
        // transitions end-to-end. Stitching blocks manually is dangerous
        // because DEFLATE blocks aren't byte-aligned at their boundaries
        // (only stored blocks force alignment) — let the encoder produce
        // a valid bit-stream. Strict roundtrip now that step 4 is wired.
        let mut payload = Vec::with_capacity(64 * 1024);
        let mut state: u64 = 0xc0ffee;
        for _ in 0..64 * 1024 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            payload.push((state >> 24) as u8);
        }
        let stream = raw_deflate(&payload, 1);
        let got = decode_via_resumable_in_chunks(&stream, 4096);
        assert_eq!(got, payload);
    }

    // ---- Step 4: dynamic-Huffman + advisor-suggested edge cases ----------

    /// Build a payload that strongly biases flate2 toward emitting a
    /// dynamic-Huffman block (variable entropy = dynamic Huffman wins
    /// vs fixed/stored). Returns the raw-deflate bytes.
    fn deflate_dynamic_payload(seed: u64, size: usize, level: u32) -> (Vec<u8>, Vec<u8>) {
        // English-like prose with skewed letter frequency forces dynamic
        // Huffman in practice (much smaller than fixed for skewed
        // distributions).
        let words: &[&[u8]] = &[
            b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"a ", b"lazy ", b"dog ",
            b"and ", b"then ", b"runs ",
        ];
        let mut state = seed;
        let mut payload = Vec::with_capacity(size);
        while payload.len() < size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let w = words[(state as usize) % words.len()];
            let take = w.len().min(size - payload.len());
            payload.extend_from_slice(&w[..take]);
        }
        let stream = raw_deflate(&payload, level);
        (payload, stream)
    }

    // Advisor item 1: dynamic block roundtrip through tiny output chunks.
    // Catches table-persistence-across-yield bugs (the `take`/restore
    // dance around `state.dynamic_tables`).
    #[test]
    fn dynamic_block_yields_with_one_byte_chunks() {
        let (payload, stream) = deflate_dynamic_payload(0xa11ce, 8 * 1024, 9);
        let got = decode_via_resumable_in_chunks(&stream, 1);
        assert_eq!(got, payload);
    }

    // Advisor item 2: dynamic block forcing matches longer than a single
    // output chunk → PendingMatch save/restore plus dynamic-table
    // persistence on the same call boundary.
    #[test]
    fn dynamic_block_match_crosses_yield_boundary() {
        let (payload, stream) = deflate_dynamic_payload(0xb0b, 64 * 1024, 9);
        // Chunk size 7 is intentionally smaller than the DEFLATE
        // max-match (258), so most matches yield mid-copy.
        let got = decode_via_resumable_in_chunks(&stream, 7);
        assert_eq!(got, payload);
    }

    // Advisor item 3: two consecutive dynamic blocks with DIFFERENT
    // tables. Catches "stale dynamic_tables leak across block
    // boundaries." Build by concatenating two independent
    // raw_deflate streams via a stored-block bridge to force byte
    // alignment between them. (We can't concatenate two DEFLATE
    // streams directly — both have BFINAL=1.)
    //
    // Construction: stored_block(b"X", false) acts as a single-bit
    // BTYPE=00 sentinel; raw_deflate(..., 9) is then guaranteed to
    // start byte-aligned because stored blocks force alignment.
    // After the first dynamic block completes (its EOB in arbitrary
    // bit position), we can't just concatenate another raw_deflate —
    // so use a different strategy: ONE raw_deflate of a payload
    // big enough that flate2 emits >=2 dynamic blocks naturally.
    #[test]
    fn two_consecutive_dynamic_blocks_with_different_tables() {
        // 256 KiB of varied prose — at level 9 with this size flate2
        // typically emits multiple blocks; the encoder picks per-block
        // Huffman tables based on local statistics, so adjacent
        // dynamic blocks usually differ.
        let (payload, stream) = deflate_dynamic_payload(0xfeed, 256 * 1024, 9);
        // Large chunk so blocks process in one read_stream — exercises
        // the EOB → finish_current_block → dynamic_tables.None pattern
        // mid-call, then re-entry to a new dynamic block on the next
        // loop iteration. Catches "tables leaked from block N into
        // block N+1."
        let got = decode_via_resumable_in_chunks(&stream, 16 * 1024);
        assert_eq!(got, payload);
    }

    // Advisor item 4: dynamic block straddling `encoded_until_bits`.
    // Per CLAUDE.md "no fallbacks" + vendor GzipReader contract
    // (parallel-SM boundaries land at block boundaries), this MUST be
    // a loud Err — not silent stop and not infinite loop. The
    // post-parse check in try_enter_next_block + the no-progress guard
    // in read_stream both defend this; this test asserts the Err
    // surfaces with the expected kind.
    #[test]
    fn dynamic_header_straddling_encoded_until_bits_errors_loudly() {
        let (_payload, stream) = deflate_dynamic_payload(0xc0de, 16 * 1024, 9);
        // 8 bytes = 64 bits is well inside the dynamic header parse window
        // (the header alone is typically 50-200 bits).
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, 64).unwrap();
        let mut output = vec![0u8; 1024];
        let err = inflate
            .read_stream(&mut output)
            .expect_err("straddling header must Err, not Ok or hang");
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("straddled encoded_until_bits"),
            "expected straddling-cap message; got: {err}"
        );
    }

    // Advisor item 5 (highest-value): RFC 1951 §3.2.7 special case —
    // "If only one distance code is used, it is encoded using one bit,
    // not zero bits". A block with all distance code lengths = 0 except
    // a single code of length 1 should decode correctly. If
    // `DistTable::build` doesn't handle the degenerate-tree case, this
    // test will surface it.
    //
    // We can't easily synthesize this manually — instead, force flate2
    // into a low-match regime where only one distance code gets used,
    // by feeding short, highly-random data.
    #[test]
    fn dynamic_block_with_single_distance_code_decodes() {
        // Tiny PRNG payload — flate2 may or may not emit a one-distance
        // dynamic block, but if it does, our decoder must handle it.
        // Even if it doesn't, the test still passes (it's a roundtrip
        // assertion regardless of the encoded block structure).
        let mut payload = Vec::with_capacity(48);
        let mut state: u64 = 0xdeadbeef;
        for _ in 0..48 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            payload.push((state >> 24) as u8);
        }
        let stream = raw_deflate(&payload, 9);
        let got = decode_via_resumable_in_chunks(&stream, 4);
        assert_eq!(got, payload);

        // Also verify `DistTable::build` accepts a degenerate
        // single-code tree directly (one code of length 1 + 1 unused
        // code, as RFC §3.2.7 prescribes).
        let mut lengths = [0u8; 30];
        lengths[0] = 1;
        lengths[1] = 1;
        assert!(
            DistTable::build(&lengths).is_some(),
            "DistTable::build must accept the degenerate single-distance-code tree"
        );
    }

    // Advisor item 6: truncate the dynamic header at every byte offset
    // and ensure the result is Err, never panic or silent corruption.
    #[test]
    fn dynamic_header_truncated_at_every_offset_does_not_panic() {
        let (_payload, stream) = deflate_dynamic_payload(0xface, 8 * 1024, 9);
        // First N bytes — there's a deflate header inside that.
        for cut in 1..stream.len().min(40) {
            let truncated = &stream[..cut];
            let mut inflate =
                ResumableInflate2::with_until_bits(truncated, 0, truncated.len() * 8).unwrap();
            let mut output = vec![0u8; 8 * 1024];
            // Must not panic. May return Err (preferred) or Ok with
            // partial output that is byte-correct for what it produced.
            let _ = inflate.read_stream(&mut output);
        }
    }

    #[test]
    fn set_window_seeds_back_reference_history() {
        // Match `copy_match_windowed`'s window branch directly. Set a
        // 16-byte "predecessor window" and emit a match referencing it.
        let mut state = ResumableInflate2::with_until_bits(&[0u8; 0], 0, 0).unwrap();
        state.set_window(b"0123456789abcdef").unwrap();
        let mut output = [0u8; 8];
        // distance=10 length=5 starting at out_pos=0 → bytes "6789a" from
        // the window (logical positions 6..11; window has 16 bytes).
        let new_pos = copy_match_windowed(&mut state, &mut output, 0, 10, 5).unwrap();
        assert_eq!(new_pos, 5);
        assert_eq!(&output[..5], b"6789a");
        assert!(state.pending_match.is_none());
    }

    #[test]
    fn copy_match_windowed_yields_when_output_full() {
        let mut state = ResumableInflate2::with_until_bits(&[0u8; 0], 0, 0).unwrap();
        state.set_window(b"0123456789abcdef").unwrap();
        let mut output = [0u8; 3]; // smaller than length
        let new_pos = copy_match_windowed(&mut state, &mut output, 0, 10, 5).unwrap();
        assert_eq!(new_pos, 3);
        assert_eq!(&output, b"678");
        let pm = state.pending_match.expect("should have stashed");
        assert_eq!(pm.distance, 10);
        assert_eq!(pm.length_remaining, 2);
    }

    #[test]
    fn copy_match_windowed_invalid_distance_errors() {
        let mut state = ResumableInflate2::with_until_bits(&[0u8; 0], 0, 0).unwrap();
        // window empty + out_pos = 0 + distance = 1 → invalid.
        let mut output = [0u8; 8];
        let err = copy_match_windowed(&mut state, &mut output, 0, 1, 4).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(state.pending_match.is_none());
    }

    #[test]
    fn copy_match_windowed_lz77_overlap_inside_output() {
        // dist=1 length=5 with output starting with one literal: classic
        // LZ77 run-length pattern ("Xxxxxx" from a single literal then
        // 5 repetitions).
        let mut state = ResumableInflate2::with_until_bits(&[0u8; 0], 0, 0).unwrap();
        let mut output = [0u8; 8];
        output[0] = b'X';
        let new_pos = copy_match_windowed(&mut state, &mut output, 1, 1, 5).unwrap();
        assert_eq!(new_pos, 6);
        assert_eq!(&output[..6], b"XXXXXX");
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

    // Correctness gate for the `copy_match_windowed` fast-path port.
    //
    // The fast path delegates to `copy_match_fast` from
    // `consume_first_decode.rs` (the vendor-shape SIMD/u64-unrolled match
    // copy) when `dist <= out_pos` and there's enough output headroom.
    // The slow path is the per-byte loop. This test exercises both with
    // identical inputs and asserts byte-equality, plus exercises the
    // fall-through cases (window touch, insufficient margin) to confirm
    // the gate keeps them on the slow path.
    #[test]
    fn copy_match_windowed_fast_path_matches_slow_path() {
        // Deterministic PRNG without an extra dep — LCG (same constants
        // as `make_low_entropy_data` in routing.rs).
        let mut hash: u64 = 0xdead_beef_cafe_f00d;
        let mut next_byte = || {
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1);
            (hash >> 24) as u8
        };

        // For each scenario, run via copy_match_windowed (the production
        // function under test) AND via a hand-rolled reference per-byte
        // copy. The two must agree.

        struct Case {
            label: &'static str,
            // Bytes already in output before the match.
            prefix: Vec<u8>,
            // (distance, length) of the back-reference.
            distance: u32,
            length: u32,
            // Window seed (32 KiB or shorter).
            window: Vec<u8>,
            // Total output slice length (so headroom can be controlled).
            output_len: usize,
        }

        // Helper: hand-rolled byte-by-byte LZ77 with window. Authoritative
        // reference. Total "history" is logically `[window, output[..out_pos]]`;
        // a match `dist` back starts at logical position
        // `window_len + out_pos - dist`. Indices < window_len read from
        // `window[src_logical]` directly (assumes the input window slice
        // matches what `SlidingWindow::seed` stores, i.e. the trailing
        // WINDOW_SIZE bytes if the slice is longer than WINDOW_SIZE).
        fn ref_copy(
            prefix: &[u8],
            window: &[u8],
            distance: u32,
            length: u32,
            output_len: usize,
        ) -> Vec<u8> {
            let mut out = vec![0u8; output_len];
            out[..prefix.len()].copy_from_slice(prefix);
            let out_pos = prefix.len();
            // Mirror SlidingWindow::seed: it keeps only the trailing
            // WINDOW_SIZE (= 32 KiB) bytes of any longer input.
            let effective_window: &[u8] = if window.len() > WINDOW_SIZE {
                &window[window.len() - WINDOW_SIZE..]
            } else {
                window
            };
            let window_len = effective_window.len();
            let total_history = window_len + out_pos;
            let dist = distance as usize;
            let len = length as usize;
            let copy_n = len.min(output_len - out_pos);
            let start_logical = total_history - dist;
            for i in 0..copy_n {
                let src_logical = start_logical + i;
                let byte = if src_logical < window_len {
                    effective_window[src_logical]
                } else {
                    out[src_logical - window_len]
                };
                out[out_pos + i] = byte;
            }
            out
        }

        // Generate enough cases to exercise: fast-path eligible, fast-path
        // ineligible by margin, fast-path ineligible by window touch.
        let mut prefix_a = Vec::with_capacity(256);
        for _ in 0..256 {
            prefix_a.push(next_byte());
        }
        let mut window_a = Vec::with_capacity(32 * 1024);
        for _ in 0..32 * 1024 {
            window_a.push(next_byte());
        }

        let cases: Vec<Case> = vec![
            // Fast-path eligible (dist 16, len 200, plenty of headroom).
            Case {
                label: "fast: dist=16 len=200 margin=2KB",
                prefix: prefix_a.clone(),
                distance: 16,
                length: 200,
                window: Vec::new(),
                output_len: prefix_a.len() + 2048,
            },
            // Fast-path eligible (large match, max SIMD path).
            Case {
                label: "fast: dist=64 len=2000",
                prefix: {
                    let mut p = prefix_a.clone();
                    while p.len() < 4096 {
                        p.push(next_byte());
                    }
                    p
                },
                distance: 64,
                length: 2000,
                window: Vec::new(),
                output_len: 8192,
            },
            // Fast-path eligible: short match (must still go through fast
            // path's 5-word unconditional store).
            Case {
                label: "fast: dist=8 len=4",
                prefix: prefix_a.clone(),
                distance: 8,
                length: 4,
                window: Vec::new(),
                output_len: prefix_a.len() + 512,
            },
            // Fast-path eligible: RLE (dist=1).
            Case {
                label: "fast: dist=1 len=100",
                prefix: prefix_a.clone(),
                distance: 1,
                length: 100,
                window: Vec::new(),
                output_len: prefix_a.len() + 512,
            },
            // Fast-path eligible: small distance (2-7) overlap.
            Case {
                label: "fast: dist=3 len=30",
                prefix: prefix_a.clone(),
                distance: 3,
                length: 30,
                window: Vec::new(),
                output_len: prefix_a.len() + 512,
            },
            // Slow-path forced: insufficient headroom (would overrun
            // fast path's 40-byte unrolled write). Must still produce
            // correct output.
            Case {
                label: "slow: tight margin (output ends right after match)",
                prefix: prefix_a.clone(),
                distance: 16,
                length: 100,
                window: Vec::new(),
                output_len: prefix_a.len() + 100, // exactly len of match
            },
            // Slow-path forced: window touch (dist > out_pos).
            Case {
                label: "slow: window touch (dist > out_pos)",
                prefix: vec![0u8; 10], // out_pos = 10
                distance: 100,         // > out_pos → must read from window
                length: 50,
                window: window_a.clone(),
                output_len: 1024,
            },
            // Slow-path forced: window-then-output span.
            Case {
                label: "slow: window-then-output span",
                prefix: prefix_a[..50].to_vec(), // out_pos = 50
                distance: 80,                    // 30 bytes from window, 50 from output
                length: 100,
                window: window_a.clone(),
                output_len: 1024,
            },
        ];

        for case in &cases {
            // Reference output.
            let want = ref_copy(
                &case.prefix,
                &case.window,
                case.distance,
                case.length,
                case.output_len,
            );

            // Production output via copy_match_windowed.
            let mut state =
                ResumableInflate2::with_until_bits(&[0u8; 0], 0, 0).expect("dummy init");
            state.set_window(&case.window).expect("set_window");
            let mut got = vec![0u8; case.output_len];
            got[..case.prefix.len()].copy_from_slice(&case.prefix);
            let out_pos = case.prefix.len();
            let new_pos =
                copy_match_windowed(&mut state, &mut got, out_pos, case.distance, case.length)
                    .expect("copy_match_windowed");
            assert_eq!(
                new_pos,
                out_pos + (case.length as usize).min(case.output_len - out_pos),
                "{}: new_pos must equal out_pos + min(length, remaining)",
                case.label
            );
            assert_eq!(
                &got[..new_pos],
                &want[..new_pos],
                "{}: copy_match_windowed output disagrees with reference",
                case.label
            );
        }
    }

    // §2 divergence 2 (pure-rust-isa-l plan): the no-progress guard at
    // `resumable.rs:512-522` silently breaks the inner loop when a
    // `read_stream` iteration makes zero progress. This catches the case
    // where a body decoder observes `bit_position >= encoded_until_bits`
    // at the top of its loop and returns without a state transition.
    //
    // This test locks the current contract:
    //   1. The first `read_stream` call with a mid-block `encoded_until_bits`
    //      must return cleanly (no panic, no infinite loop), with
    //      `finished=false`.
    //   2. The bytes written so far must be byte-correct (a prefix of the
    //      full decoded output).
    //   3. After widening `encoded_until_bits` to cover the full input
    //      (via `advance_input(0)` post-fact won't widen — we instead
    //      construct a fresh wrapper at the same start with the full
    //      `until_bits`), the same input decodes to the full payload.
    //
    // The "silent truncation" risk in the plan is: a production caller
    // sees `finished=false, bytes_written < expected` and treats the
    // partial output as authoritative without checking the surface-level
    // contract. The parallel-SM consumer DOES check (`sm_driver.rs`
    // verifies the final CRC32 + ISIZE; a truncated chunk fails CRC).
    // This test guarantees that the wrapper's own contract is
    // truncation-as-non-error, NOT panic, NOT infinite loop.
    #[test]
    fn no_progress_guard_breaks_cleanly_on_mid_block_until_bits() {
        // 16 KiB dynamic-block payload — produces a single dynamic block
        // whose body is hundreds of bytes long; cap mid-body.
        let (payload, stream) = deflate_dynamic_payload(0xfeed_face_c0de, 16 * 1024, 9);
        let stream_bits = stream.len() * 8;

        // First, decode fully so we know the legitimate output.
        let full: Vec<u8> = {
            let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, stream_bits).unwrap();
            let mut out = vec![0u8; payload.len() + 64];
            let r = inflate.read_stream(&mut out).expect("full decode");
            assert_eq!(r.bytes_written, payload.len(), "fixture sanity");
            out.truncate(r.bytes_written);
            out
        };
        assert_eq!(full, payload);

        // Now cap `encoded_until_bits` at 75 % of the stream — well past
        // any dynamic header (~50-200 bits) and squarely inside the body.
        let mid_cap_bits = (stream_bits * 3) / 4;
        let mut inflate = ResumableInflate2::with_until_bits(&stream, 0, mid_cap_bits).unwrap();
        let mut out = vec![0u8; payload.len() + 64];

        // Drive the wrapper to exhaustion under the cap. Loop terminates
        // because the no-progress guard fires on the iteration that hits
        // `bit_position >= encoded_until_bits` mid-body.
        let mut total = 0usize;
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            assert!(
                iterations < 1000,
                "no-progress guard MUST break the loop; \
                 hitting 1000 iterations means the guard regressed"
            );
            let r = inflate.read_stream(&mut out[total..]).expect("read");
            total += r.bytes_written;
            if r.finished {
                // The cap is mid-block, so the stream cannot have finished.
                panic!(
                    "stream cannot be `finished` while encoded_until_bits is \
                     mid-block; finished=true here means the cap was ignored"
                );
            }
            if r.bytes_written == 0 {
                // No more progress possible under the cap. Clean exit.
                break;
            }
        }

        // The decoder produced SOME prefix of the legitimate output.
        // (It might be zero if the cap landed before any body bytes could
        // be emitted — that's still a valid outcome of the guard.)
        assert!(
            total <= payload.len(),
            "wrote {total} bytes; full payload is {} bytes — wrote more than fits",
            payload.len()
        );
        assert_eq!(
            &out[..total],
            &payload[..total],
            "the bytes written must be a byte-correct prefix of the full payload"
        );
    }
}
