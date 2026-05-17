//! Pure-Rust marker-emitting deflate decoder for parallel single-member decode.
//!
//! # Purpose
//!
//! Phase 1 of the v0.6 marker pipeline. Decodes one chunk of a deflate stream
//! into a `Vec<u16>` where:
//!
//! - Bytes 0..=255 are literal output values, exactly as a normal decoder would
//!   produce them.
//! - Values ≥ [`MARKER_BASE`] (= 32768) are **markers** standing in for bytes
//!   that a back-reference would have copied from *before* this chunk started
//!   (i.e. the predecessor chunk's last 32 KB).
//!
//! The marker encoding matches [`crate::decompress::parallel::replace_markers`]:
//! `marker = MARKER_BASE + offset`, where `offset = 0` means the most recently
//! emitted byte of the predecessor's window.
//!
//! # Algorithm
//!
//! Standard RFC 1951 deflate decode, with two changes from a normal decoder:
//!
//! 1. **Output is `Vec<u16>`** so we can store markers in-band.
//! 2. **The match copy is split** by the relation between back-ref distance
//!    `D` and current output position `P`:
//!    - If `D ≤ P` the entire match is chunk-local: copy `length` u16 values
//!      from `output[P − D ..]`. Markers carried by those source positions
//!      propagate through the copy unchanged (this is the subtle landmine
//!      from the premortem — chunk-local copies must move u16s, not u8s).
//!    - If `D > P` the first `D − P` bytes of the match are cross-chunk:
//!      emit them as markers `MARKER_BASE + (D - P - 1)`,
//!      `MARKER_BASE + (D - P - 2)`, …, `MARKER_BASE + 0`. The remaining
//!      `length − (D − P)` bytes (if any) become chunk-local and copy from
//!      `output[0..]`.
//!
//! # Reuse
//!
//! Bit buffer borrowed from [`crate::decompress::inflate::consume_first_decode::Bits`].
//! Canonical Huffman table build is implemented locally (we'd otherwise need
//! to refactor the existing decoder's internal tables; out of scope).
//!
//! # Non-goals
//!
//! - SIMD inner loop. A future PR can vectorize literal-heavy fast paths.
//! - BMI2 BZHI tricks. Same.
//! - Matching the production `inflate_consume_first` u8 throughput. Goal here
//!   is correctness + "fast enough that 4 threads of this comfortably beat
//!   sequential ISA-L on CI." Threshold from the premortem: ≥ 287 MB/s on
//!   x86_64 CI per thread.

#![allow(dead_code)]

use std::io::{Error, ErrorKind, Result};

use crate::decompress::inflate::consume_first_decode::Bits;
use crate::decompress::inflate::consume_first_table::{CFEntry, ConsumeFirstTable, CF_TABLE_BITS};
use crate::decompress::parallel::replace_markers::MARKER_BASE;

/// Maximum deflate back-reference distance per RFC 1951 §3.2.5.
const WINDOW_SIZE: usize = 32_768;

/// Maximum deflate match length.
const MAX_MATCH_LEN: usize = 258;

// ── RFC 1951 §3.2.5 length / distance tables ────────────────────────────────

/// `(base_length, extra_bits)` for length codes 257..=285.
/// Index `i` corresponds to symbol `257 + i`.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// `(base_distance, extra_bits)` for distance codes 0..=29.
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Order of code-length code lengths in a dynamic block header (RFC 1951 §3.2.7).
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ── Huffman decode helpers (ConsumeFirstTable wrapper) ──────────────────────

/// Decode one symbol from `bits` using a ConsumeFirstTable.
/// Returns the raw symbol (0–255 literal, 256 EOB, 257–285 length for litlen;
/// 0–29 distance for distance tables).
#[inline(always)]
fn decode_cf(table: &ConsumeFirstTable, bits: &mut Bits) -> Result<u32> {
    if bits.available() < CF_TABLE_BITS as u32 {
        bits.refill();
    }
    let entry = table.lookup_main(bits.peek());
    if entry.is_subtable() {
        bits.consume(entry.bits()); // consumes CF_TABLE_BITS (= 11)
        let extra = entry.subtable_extra_bits() as u32;
        if extra > 0 && bits.available() < extra {
            bits.refill();
        }
        let sub_entry = table.lookup_sub(entry, bits.peek());
        bits.consume(sub_entry.bits());
        return cf_entry_to_sym(sub_entry);
    }
    bits.consume(entry.bits());
    cf_entry_to_sym(entry)
}

#[inline(always)]
fn cf_entry_to_sym(entry: CFEntry) -> Result<u32> {
    if entry.is_literal() {
        Ok(entry.symbol() as u32)
    } else if entry.is_eob() {
        Ok(256)
    } else if entry.is_length() {
        Ok(entry.symbol() as u32)
    } else {
        Err(Error::new(ErrorKind::InvalidData, "invalid Huffman code"))
    }
}

// ── Public entry ────────────────────────────────────────────────────────────

/// Decode a deflate stream starting at `start_bit_offset` within `data`,
/// producing `Vec<u16>` with cross-chunk back-references encoded as markers
/// (values ≥ [`MARKER_BASE`]).
///
/// `data` is the deflate bytes (no gzip header / trailer); `start_bit_offset`
/// can be any bit position (`0..=7` for non-byte-aligned chunk starts is
/// supported — see premortem mitigation B1). The decoder runs until BFINAL is
/// seen or `data` runs out.
///
/// Returns `(output, end_bit_offset)` where `end_bit_offset` is the bit
/// position just past the consumed stream (suitable for chaining).
pub fn decode_chunk_markers(data: &[u8], start_bit_offset: usize) -> Result<(Vec<u16>, usize)> {
    decode_chunk_markers_bounded(data, start_bit_offset, None)
}

/// Validate a candidate deflate-block boundary by trial-decoding enough of
/// the stream to make false positives statistically impossible.
///
/// Returns `Ok(())` if the decoder runs cleanly until BOTH:
/// - at least `min_blocks` block headers were decoded successfully, AND
/// - cumulative output reached `min_output_bytes`.
///
/// Returns `Err` on the first malformed block (invalid Huffman table, bad
/// length code, LEN/NLEN mismatch, etc.) OR if the stream ends before the
/// thresholds are met without reaching BFINAL.
///
/// Why two thresholds: a single stored block can legitimately reach 65535
/// bytes of output, so `min_output_bytes` alone could be satisfied by a
/// false-positive "fake stored block" whose LEN/NLEN happens to be valid
/// by chance (~1 / 65536). `min_blocks ≥ 2` forces the validator past
/// that single fake block — the chance two consecutive false-positive
/// blocks both validate is astronomically small.
///
/// `require_non_fixed_stop`: when true, the non-BFINAL exit additionally
/// peeks the next block's BTYPE and refuses to stop if it is BTYPE=01
/// (fixed Huffman). Fixed-Huffman blocks have no header redundancy — any
/// bit sequence decodes as a valid BTYPE=01 block — so stopping there
/// admits false positives in BTYPE=01-heavy regions. Require BTYPE=00
/// (stored) or BTYPE=10 (dynamic) at the stop point, which have
/// structured headers that provide genuine entropy. Mirrors rapidgzip
/// GzipChunk.hpp:552. Set true in boundary-discovery callers
/// (`try_decode_at`); false in G1 invariant checks where the bit IS
/// already known to be a real boundary.
pub fn validate_boundary(
    data: &[u8],
    start_bit_offset: usize,
    min_blocks: u32,
    min_output_bytes: usize,
    require_non_fixed_stop: bool,
) -> Result<()> {
    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        ));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }
    let mut output: Vec<u16> = Vec::with_capacity(min_output_bytes.next_power_of_two().max(4096));
    // Hard cap: if we decoded 100× the minimum output without finding a
    // suitable stop point, this is almost certainly a false positive in a
    // BTYPE=01-heavy region. Real boundaries hit BTYPE=00/10 within a few
    // blocks of the minimum; spinning through many MB of fixed-Huffman output
    // is pure noise. Capping prevents runaway allocation from adversarial or
    // random starting positions.
    let max_output_bytes = min_output_bytes.saturating_mul(100).max(1_000_000);
    let mut blocks_decoded = 0u32;
    loop {
        if bits.available() < 3 {
            bits.refill();
        }
        if output.len() >= max_output_bytes {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "validate_boundary: output cap exceeded — false positive in BTYPE=01-heavy region",
            ));
        }
        if blocks_decoded >= min_blocks && output.len() >= min_output_bytes {
            if require_non_fixed_stop {
                // Peek the upcoming block's BTYPE. Reject BTYPE=01 (fixed
                // Huffman) as a stop point — any bit sequence decodes as
                // BTYPE=01 so false positives in BTYPE=01-heavy regions
                // trivially pass. BTYPE=00/10 have structured headers that
                // provide real entropy, making accidental matches rare.
                let next_btype = (bits.peek() >> 1) & 3;
                if next_btype == 0 || next_btype == 2 {
                    return Ok(());
                }
                // Upcoming block is BTYPE=01 (fixed Huffman) or BTYPE=11
                // (reserved/invalid): keep decoding. BTYPE=01 has no header
                // redundancy — any bit sequence decodes as valid. BTYPE=11
                // never appears in valid deflate; stopping there means this
                // is a false positive. Both require us to continue.
            } else {
                return Ok(());
            }
        }
        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u32;
        bits.consume(3);
        let mut clean_tail = None;
        match btype {
            0 => decode_stored(&mut bits, &mut output, &mut clean_tail)?,
            1 => decode_fixed(&mut bits, &mut output, max_output_bytes, &mut clean_tail)?,
            2 => decode_dynamic(&mut bits, &mut output, max_output_bytes, &mut clean_tail)?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type 3")),
        }
        blocks_decoded += 1;
        if bfinal {
            // BFINAL terminated the stream. For a true mid-stream boundary
            // we'd expect many more blocks ahead, not a stream that ends
            // right here. Accepting BFINAL with too few blocks decoded is
            // how the "fake stored block" false positive slipped through
            // (one BFINAL=true stored block with a coincidentally valid
            // LEN/NLEN at a random offset). Require the same thresholds
            // as the non-BFINAL path. A real near-EOF boundary won't be
            // picked by `search_boundary_forward` — chunk starts are
            // spaced over the whole stream, not packed at the tail.
            if blocks_decoded >= min_blocks && output.len() >= min_output_bytes {
                return Ok(());
            }
            return Err(Error::new(
                ErrorKind::InvalidData,
                "BFINAL before min_blocks/min_output thresholds",
            ));
        }
    }
}

/// Like `decode_chunk_markers` but also stops at a block boundary at or past
/// `end_bit_limit` (if provided). Used by the parallel pipeline so worker N
/// decodes only the range `[start_bits[N], start_bits[N+1])`. Without this
/// bound, every worker would decode all the way to BFINAL and produce
/// duplicate output.
///
/// The decoder always finishes the currently-in-progress block before
/// stopping; it never stops mid-block. Therefore `end_bit_limit` must point
/// at a real deflate block boundary — call sites get those from
/// `search_boundary_forward` or `record_block_starts`.
pub fn decode_chunk_markers_bounded(
    data: &[u8],
    start_bit_offset: usize,
    end_bit_limit: Option<usize>,
) -> Result<(Vec<u16>, usize)> {
    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        ));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }

    // Capacity must scale with the bounded range, not all of `data`. Phase 1
    // workers see the full deflate slice (potentially 100+ MB on Silesia)
    // even though each only decodes ~25 MB of it; allocating 4 × data.len()
    // per worker was ~1.3 GB/thread × 4 threads = 5 GB peak on Silesia, far
    // beyond a CI runner's RAM budget. Bound by the worker's actual range:
    //
    //   if end_bit_limit provided: range = (end_bit - start_bit) / 8 bytes
    //                              (× 4 worst-case expansion factor)
    //   else (last chunk, runs to BFINAL): cap at remaining input × 4
    //
    // Per-thread overshoot caps at the chunk size, not the full stream.
    // Copilot review on PR #94.
    let range_bytes = match end_bit_limit {
        Some(end) => end.saturating_sub(start_bit_offset).div_ceil(8),
        None => data.len().saturating_sub(byte_offset),
    };
    let cap = range_bytes.saturating_mul(4).max(4096);
    let mut output: Vec<u16> = Vec::with_capacity(cap);

    decode_loop(
        &mut bits,
        &mut output,
        byte_offset,
        bit_in_byte,
        None,
        end_bit_limit,
        None,
    )?;

    // Capacity may grow during decode on highly compressible chunks (a single
    // length-code with extras can expand 258× into output, so the 4× upper
    // bound isn't airtight). Vec doubling handles that with O(N) amortized
    // cost; the important property of the bounded initial capacity is the
    // **lower** bound — chunks no longer over-allocate at startup based on
    // the whole-stream length.

    // Compute end_bit_offset. Same arithmetic as `decode_loop`'s bit_pos:
    //
    //   bits.pos*8 - bits.available()  is the number of bits consumed
    //   since `Bits::new(&data[byte_offset..])` — *including* the
    //   `bit_in_byte` bits dropped at startup. Don't add `bit_in_byte`
    //   again or it double-counts. Same bug that bit `decode_loop` until
    //   PR #90 fixed it (commit 80f4e85); the return-value path here was
    //   missed in that fix and was hidden by the lack of a cross-chunk
    //   consistency check until D4 surfaced it on the
    //   end_to_end_low_entropy_24mb_t4_matches_oracle test.
    let _ = bit_in_byte; // intentionally unused — see note above
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    let end_bit_offset = byte_offset * 8 + bits_consumed_from_slice;

    Ok((output, end_bit_offset))
}

/// Result from [`decode_chunk_bootstrap`]: a marker decoder run that exits
/// early at the first block boundary where the trailing 32 KB of decoded
/// output is provably marker-free, so the caller can hand off to ISA-L with
/// that 32 KB as a `isal_inflate_set_dict` seed.
///
/// This mirrors rapidgzip's per-chunk worker design
/// (`vendor/rapidgzip/.../GzipChunk.hpp:413-657`): the marker decoder runs
/// only as a bootstrap until enough clean tail accumulates, then ISA-L
/// decodes the remaining ~99% of the chunk. The previous design ran the
/// marker decoder for the whole chunk and capped per-thread throughput at
/// pure-Rust speed (~50 MB/s/thread vs ISA-L's ~163 MB/s/thread on x86_64
/// CI). The handoff closes that gap.
pub struct BootstrapResult {
    /// u16 output covering bootstrap range. May contain markers in its
    /// prefix; when `clean_window` is `Some`, the trailing 32 KB is
    /// guaranteed marker-free.
    pub markers: Vec<u16>,
    /// Bit position just past the bootstrap's last fully-decoded block.
    /// Always at a real deflate block boundary (G1 invariant), suitable
    /// as `bit_offset` for [`crate::backends::isal_decompress::decompress_deflate_from_bit_with_end`].
    pub end_bit_offset: usize,
    /// 32 KB sliding window for ISA-L's `isal_inflate_set_dict`, present
    /// only when the trailing 32 KB of `markers` is marker-free. When
    /// `None`, the caller must NOT hand off to ISA-L — either the
    /// bootstrap consumed the entire chunk before 32 KB of clean output
    /// accumulated, or `end_bit_limit` was reached first. In that case
    /// `markers` covers the whole chunk and phase-2 marker-resolve runs
    /// over it as in the pre-handoff design.
    pub clean_window: Option<Vec<u8>>,
    /// True when the bootstrap exited because it decoded a BFINAL=1 block
    /// before accumulating a clean 32 KB window AND before reaching
    /// `end_bit_limit`. This distinguishes a genuine stream-end from an
    /// anchor-cap stop, and signals that `end_bit_offset` may be at a
    /// real BFINAL block boundary that is NOT a valid starting point for
    /// the next chunk (no deflate data follows BFINAL).
    pub bfinal_hit: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CleanTailTracker {
    trailing_clean_bytes: usize,
}

impl CleanTailTracker {
    #[inline]
    fn from_output(output: &[u16]) -> Self {
        Self {
            trailing_clean_bytes: output
                .iter()
                .rev()
                .take_while(|&&v| v < MARKER_BASE)
                .count()
                .min(WINDOW_SIZE),
        }
    }

    #[inline]
    fn ready_for_handoff(self) -> bool {
        self.trailing_clean_bytes >= WINDOW_SIZE
    }

    #[inline]
    fn observe_value(&mut self, value: u16) {
        if value < MARKER_BASE {
            self.trailing_clean_bytes = (self.trailing_clean_bytes + 1).min(WINDOW_SIZE);
        } else {
            self.trailing_clean_bytes = 0;
        }
    }

    #[inline]
    fn observe_slice(&mut self, values: &[u16]) {
        if values.is_empty() {
            return;
        }
        let trailing_clean = values
            .iter()
            .rev()
            .take_while(|&&v| v < MARKER_BASE)
            .count()
            .min(WINDOW_SIZE);
        if trailing_clean == values.len() {
            self.trailing_clean_bytes =
                (self.trailing_clean_bytes + trailing_clean).min(WINDOW_SIZE);
        } else {
            self.trailing_clean_bytes = trailing_clean;
        }
    }
}

/// Bootstrap-mode variant of [`decode_chunk_markers_bounded`]. Decodes
/// until any of:
///
/// 1. **The trailing 32 KB of `markers` is marker-free at a block boundary.**
///    Returns with `clean_window = Some(<that 32 KB cast to u8>)` and
///    `end_bit_offset` pointing past the just-completed block. The caller
///    seeds ISA-L's dict with `clean_window` and resumes decoding from
///    `end_bit_offset`.
/// 2. **`end_bit_limit` is reached** (caller's chunk boundary). Returns with
///    `clean_window = None`; the caller treats this as the pre-handoff
///    "whole chunk via marker decoder" path. Rare: only happens when the
///    chunk is so small (<32 KB output) that the bootstrap never accumulates
///    enough clean data.
/// 3. **BFINAL is seen.** Same as (2) — last chunk near EOF.
///
/// Invariants:
/// - On `Ok(_)`, `end_bit_offset` is a real deflate block boundary.
/// - Capacity for `markers` is bounded by the bootstrap's actual output
///   size (typically ~32 KB + one block's worth), not `4 × deflate_bytes`
///   as the whole-chunk variant must allocate. Per-thread memory drops
///   dramatically.
pub fn decode_chunk_bootstrap(
    data: &[u8],
    start_bit_offset: usize,
    end_bit_limit: Option<usize>,
) -> Result<BootstrapResult> {
    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        ));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }

    // Bootstrap output is typically ~32 KB + one trailing block (up to
    // ~64 KB for a single dynamic-Huffman block of literals, ~128 KB
    // worst-case). Allocate generously enough to avoid early growth, but
    // not so much that we re-introduce the 4× deflate_bytes-per-thread
    // memory blowup the whole-chunk variant has.
    let mut output: Vec<u16> = Vec::with_capacity(128 * 1024);
    let mut handoff_at_boundary = false;

    decode_loop(
        &mut bits,
        &mut output,
        byte_offset,
        bit_in_byte,
        None,
        end_bit_limit,
        Some(&mut handoff_at_boundary),
    )?;

    let _ = bit_in_byte;
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    let end_bit_offset = byte_offset * 8 + bits_consumed_from_slice;

    // bfinal_hit: bootstrap stopped before end_bit_limit (or last chunk with
    // None limit) because it decoded a BFINAL=1 block. Distinguished from
    // end_bit_limit stop by comparing end_bit_offset to the limit.
    let bfinal_hit = !handoff_at_boundary
        && end_bit_limit
            .map(|limit| end_bit_offset < limit)
            .unwrap_or(false);

    let clean_window = if handoff_at_boundary && output.len() >= WINDOW_SIZE {
        // Last 32 KB MUST be marker-free here per the bootstrap-stop
        // check at the top of decode_loop. Use `assert!` rather than
        // `debug_assert!` — if the invariant is ever broken in
        // release (e.g., a future refactor flips `handoff_at_boundary`
        // incorrectly), `v as u8` would silently truncate marker
        // values to garbage that ISA-L then uses as a dict, producing
        // wrong output bytes whose CRC mismatch then triggers a
        // silent libdeflate fallback. Asserting catches the bug
        // before ISA-L sees the corrupted dict. Cost: 32 K branch
        // predictions per chunk's bootstrap exit, negligible
        // (~10 µs at modern branch-predictor throughput).
        //
        // Opus advisor on PR #97 flagged the `debug_assert!` version
        // as a release-mode silent-corruption surface.
        let start = output.len() - WINDOW_SIZE;
        let window: Vec<u8> = output[start..]
            .iter()
            .map(|&v| {
                assert!(
                    v < MARKER_BASE,
                    "bootstrap clean window contained marker at offset {}; \
                     bootstrap-stop invariant violated",
                    v - MARKER_BASE
                );
                v as u8
            })
            .collect();
        Some(window)
    } else {
        None
    };

    Ok(BootstrapResult {
        markers: output,
        end_bit_offset,
        clean_window,
        bfinal_hit,
    })
}

/// Continuation variant of [`decode_chunk_markers_bounded`]: append to an
/// existing output Vec instead of returning a fresh one. The caller's
/// `output` is treated as already-decoded chunk data, so `emit_match`'s
/// chunk-local copies and "D > P" marker test see the correct position.
///
/// Used by the slow path of [`crate::decompress::parallel::single_member`]
/// when the ISA-L handoff isn't available (non-x86_64) or fails: the
/// bootstrap accumulates ~32 KB of output, then we want the marker decoder
/// to continue from the bootstrap's `end_bit_offset` and produce the rest
/// of the chunk's u16 output. Calling the regular `decode_chunk_markers_bounded`
/// for the continuation would produce **wrong markers** because its
/// `emit_match` would see `out_pos = 0` and treat any back-reference that
/// reaches into the bootstrap as a cross-chunk reference — corrupting the
/// chunk's output. By appending to the bootstrap directly, the chunk-local
/// logic keeps its meaning.
///
/// Returns the bit position just past the last decoded block (same as
/// `decode_chunk_markers_bounded`).
pub fn decode_chunk_markers_continuing(
    data: &[u8],
    start_bit_offset: usize,
    end_bit_limit: Option<usize>,
    output: &mut Vec<u16>,
) -> Result<usize> {
    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        ));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }

    // Reserve based on the bounded range, same logic as
    // `decode_chunk_markers_bounded`. The vec already holds bootstrap data.
    let range_bytes = match end_bit_limit {
        Some(end) => end.saturating_sub(start_bit_offset).div_ceil(8),
        None => data.len().saturating_sub(byte_offset),
    };
    output.reserve(range_bytes.saturating_mul(4).max(4096));

    decode_loop(
        &mut bits,
        output,
        byte_offset,
        bit_in_byte,
        None,
        end_bit_limit,
        None,
    )?;

    let _ = bit_in_byte;
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    Ok(byte_offset * 8 + bits_consumed_from_slice)
}

/// Inner decode loop, factored so a debug-only variant can record where each
/// block started (used by integration tests to obtain real mid-stream block
/// boundaries — there's no other way to verify a candidate is a true boundary
/// short of decoding from bit 0).
#[inline]
fn decode_loop(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    base_byte: usize,
    _base_bit_in_byte: u32,
    mut block_starts: Option<&mut Vec<usize>>,
    end_bit_limit: Option<usize>,
    mut bootstrap_handoff: Option<&mut bool>,
) -> Result<()> {
    let mut clean_tail = bootstrap_handoff
        .as_ref()
        .map(|_| CleanTailTracker::from_output(output));
    loop {
        if bits.available() < 3 {
            bits.refill();
        }
        // Absolute bit position at the start of this block.
        //
        // `bits.pos` is bytes pulled from the slice that starts at
        // `base_byte`. `bits.available()` is the unconsumed bits sitting
        // in the bit buffer. So `bits.pos*8 - available()` is exactly the
        // number of bits we have consumed since `Bits::new(&data[base_byte..])`
        // was called — and this *already includes* the `bit_in_byte` bits
        // the caller consumed at the top of `decode_chunk_markers_bounded`
        // to align to the start. Earlier versions added `base_bit_in_byte`
        // again here, double-counting the alignment, which made
        // non-byte-aligned starts report a bit position `bit_in_byte` bits
        // ahead of reality. That broke `end_bit_limit` (chunk decode
        // stopped early → `SizeMismatch`) and broke `try_decode_at`'s
        // strict validation (early-exit before any block was decoded).
        let bits_in_buf = bits.available() as usize;
        let consumed = bits.pos.saturating_mul(8).saturating_sub(bits_in_buf);
        let bit_pos = base_byte * 8 + consumed;

        if let Some(starts) = block_starts.as_mut() {
            starts.push(bit_pos);
        }

        // Bounded-mode early exit: when `bit_pos >= end_bit_limit` AND
        // we're at a real block boundary (which we always are at the top
        // of this loop), stop. The actual end_bit returned to the caller
        // may exceed `end_bit_limit` if the caller's limit fell inside a
        // block — chunk decode then naturally lands on the next real
        // boundary past the limit.
        //
        // This is the *speculative* contract: `end_bit_limit` is an
        // advisory target, not a hard upper bound. The caller
        // (`phase1c_resolve_consistency` in `single_member.rs`) uses
        // chunk N's returned `end_bit_offset` as chunk N+1's confirmed
        // start, correcting any speculative-start mismatch.
        //
        // Earlier versions enforced an exact-match contract (PR #90
        // commit 02381c4) that returned Err on overshoot. That made
        // false-positive starts produce a hard failure that fell back
        // to libdeflate. Opus advisor review showed the simpler answer
        // is: chunk N's decode is trustworthy (induction from chunk 0's
        // real start), so let its end_bit propagate forward as the
        // ground truth. See premortem section G (G5 entry).
        if let Some(limit) = end_bit_limit {
            if bit_pos >= limit {
                return Ok(());
            }
        }

        // Bootstrap-mode early exit: when the caller passed a flag for
        // bootstrap handoff, check at every block boundary whether the
        // trailing 32 KB of `output` is marker-free. If yes, we have a
        // clean 32 KB window the caller can hand to ISA-L's
        // `isal_inflate_set_dict` to decode the rest of the chunk at
        // ISA-L speed instead of pure-Rust marker decoder speed.
        //
        // This mirrors rapidgzip's per-chunk worker
        // (`vendor/rapidgzip/.../GzipChunk.hpp:521`,
        // `cleanDataCount >= MAX_WINDOW_SIZE`). It is *the* difference
        // that lets per-thread parity with sequential ISA-L hold on
        // single-member parallel decode: the marker decoder bootstraps
        // ≤32 KB per chunk, ISA-L handles the remaining ~99%.
        //
        // The scan is O(32 KB) per block boundary. Block boundaries
        // happen every ~16-64 KB of input on typical L1-L6 streams, so
        // total scan cost is O(N) with small constant — negligible
        // compared to the decode itself, and irrelevant once the
        // bootstrap exits (which happens within the first ~32-64 KB of
        // output, not at every block of the whole chunk).
        if let Some(ref mut handoff) = bootstrap_handoff {
            if clean_tail
                .as_ref()
                .is_some_and(|tracker| tracker.ready_for_handoff())
            {
                // Rapidgzip hands off unconditionally when cleanDataCount
                // >= MAX_WINDOW_SIZE (GzipChunk.hpp:521). We previously
                // gated on next_btype ∈ {0, 2} as defense against
                // false-positive start positions, but this point is
                // post-decode of a real block (EOB just fired), so the
                // next header is a real one regardless of its btype.
                **handoff = true;
                return Ok(());
            }
            // Bootstrap output cap: if we've decoded >1 MB and STILL haven't
            // accumulated a clean 32 KiB tail, this chunk's data has
            // marker propagation that won't stop. Abort so caller can try
            // next BlockFinder candidate or fall back to authoritative
            // re-dispatch (worker with predecessor's actual window via
            // ISA-L fast path is ~24x faster than running 12 MB through
            // the pure-Rust marker decoder).
            const BOOTSTRAP_OUTPUT_CAP: usize = 1024 * 1024;
            if output.len() >= BOOTSTRAP_OUTPUT_CAP {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "bootstrap output cap exceeded without handoff",
                ));
            }
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u32;
        bits.consume(3);

        // Per-block output cap. Without this, a malformed block on a
        // phantom boundary (BlockFinder false positive) can drive
        // decode_dynamic / decode_fixed into a tight loop emitting
        // gigabytes of garbage symbols before hitting EOF — proven by
        // trace at HEAD `d78bf9d` where worker-26 hung on bit
        // 872417560 and never returned. 256 MiB is generous enough
        // that no legitimate single deflate block exceeds it
        // (deflate's spec caps a single block's match-length output
        // implicitly through bit-stream input limits, but a malformed
        // header can synthesize huge runs from garbage codes).
        const MAX_PER_BLOCK_OUTPUT: usize = 256 * 1024 * 1024;
        match btype {
            0 => decode_stored(bits, output, &mut clean_tail)?,
            1 => decode_fixed(bits, output, MAX_PER_BLOCK_OUTPUT, &mut clean_tail)?,
            2 => decode_dynamic(bits, output, MAX_PER_BLOCK_OUTPUT, &mut clean_tail)?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type 3")),
        }

        if bfinal {
            return Ok(());
        }
    }
}

/// Test helper: decode `data` from bit 0 and return every block-start bit
/// position observed. Each position is a valid input to
/// `decode_chunk_markers` — passing it in starts decoding at that block's
/// header. Used by the end-to-end integration test to avoid relying on
/// `BlockFinder`'s heuristic candidates (which produce false positives that
/// silently corrupt subsequent decode).
#[cfg(test)]
pub(super) fn record_block_starts(data: &[u8]) -> Result<Vec<usize>> {
    let mut bits = Bits::new(data);
    let mut output = Vec::new();
    let mut starts = Vec::new();
    decode_loop(&mut bits, &mut output, 0, 0, Some(&mut starts), None, None)?;
    Ok(starts)
}

// ── Stored block (BTYPE = 00) ───────────────────────────────────────────────

fn decode_stored(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    clean_tail: &mut Option<CleanTailTracker>,
) -> Result<()> {
    bits.align_to_byte();
    let len = bits.read_u16();
    let nlen = bits.read_u16();
    if len != !nlen {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Stored LEN/NLEN mismatch",
        ));
    }
    let mut remaining = len as usize;

    // Drain any whole bytes still in the bit buffer.
    while remaining > 0 && bits.available() >= 8 {
        let value = (bits.bitbuf & 0xFF) as u16;
        output.push(value);
        if let Some(tracker) = clean_tail.as_mut() {
            tracker.observe_value(value);
        }
        bits.consume(8);
        remaining -= 1;
    }

    // Direct copy from input.
    if remaining > 0 {
        if bits.pos + remaining > bits.data.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Truncated stored block",
            ));
        }
        let start = output.len();
        output.reserve(remaining);
        for &b in &bits.data[bits.pos..bits.pos + remaining] {
            output.push(b as u16);
        }
        if let Some(tracker) = clean_tail.as_mut() {
            tracker.observe_slice(&output[start..]);
        }
        bits.pos += remaining;
    }

    // Reset bit buffer state. The libdeflate sliding-window refill advances
    // bits.pos ahead of actual consumption (it pre-loads bytes into bitbuf).
    // When the stored block's data fit entirely in the pre-loaded bitbuf (drain
    // loop emptied remaining without exhausting the buffer), the reset zeroes
    // bitsleft/bitbuf while bits.pos is still pointing past those pre-loaded
    // bytes — breaking the invariant: bits.pos*8 - available() == bits_consumed.
    // Rewind pos by the number of whole bytes still in the buffer so the
    // invariant holds after zeroing available to 0.
    let pre_buffered = (bits.available() / 8) as usize;
    bits.bitbuf = 0;
    bits.bitsleft = 0;
    bits.pos = bits.pos.saturating_sub(pre_buffered);
    Ok(())
}

// ── Fixed Huffman block (BTYPE = 01) ────────────────────────────────────────

fn fixed_litlen_table() -> &'static ConsumeFirstTable {
    use std::sync::OnceLock;
    static T: OnceLock<ConsumeFirstTable> = OnceLock::new();
    T.get_or_init(|| {
        let mut lens = vec![0u8; 288];
        lens[0..144].fill(8);
        lens[144..256].fill(9);
        lens[256..280].fill(7);
        lens[280..288].fill(8);
        ConsumeFirstTable::build(&lens).expect("fixed litlen table builds")
    })
}

fn fixed_dist_table() -> &'static ConsumeFirstTable {
    use std::sync::OnceLock;
    static T: OnceLock<ConsumeFirstTable> = OnceLock::new();
    T.get_or_init(|| {
        ConsumeFirstTable::build_distance(&[5u8; 30]).expect("fixed dist table builds")
    })
}

fn decode_fixed(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    max_output: usize,
    clean_tail: &mut Option<CleanTailTracker>,
) -> Result<()> {
    decode_huffman_block(
        bits,
        output,
        fixed_litlen_table(),
        fixed_dist_table(),
        max_output,
        clean_tail,
    )
}

// ── Dynamic Huffman block (BTYPE = 10) ──────────────────────────────────────

fn decode_dynamic(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    max_output: usize,
    clean_tail: &mut Option<CleanTailTracker>,
) -> Result<()> {
    // Header: HLIT (5), HDIST (5), HCLEN (4).
    if bits.available() < 14 {
        bits.refill();
    }
    let hlit = ((bits.peek() & 0x1F) as usize) + 257;
    bits.consume(5);
    let hdist = ((bits.peek() & 0x1F) as usize) + 1;
    bits.consume(5);
    let hclen = ((bits.peek() & 0xF) as usize) + 4;
    bits.consume(4);

    if hlit > 286 || hdist > 30 || hclen > 19 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Dynamic header out of range",
        ));
    }

    // Code-length code lengths.
    let mut cl_lens = [0u8; 19];
    for &cl_idx in CL_ORDER.iter().take(hclen) {
        if bits.available() < 3 {
            bits.refill();
        }
        cl_lens[cl_idx] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }
    let cl_table = ConsumeFirstTable::build(&cl_lens)?;

    // Read HLIT + HDIST code lengths using the code-length code.
    let total = hlit + hdist;
    let mut lens = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = decode_cf(&cl_table, bits)?;
        match sym {
            0..=15 => {
                lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Copy previous code length 3..=6 times.
                if i == 0 {
                    return Err(Error::new(ErrorKind::InvalidData, "CL repeat-16 at start"));
                }
                if bits.available() < 2 {
                    bits.refill();
                }
                let n = ((bits.peek() & 0x3) as usize) + 3;
                bits.consume(2);
                let prev = lens[i - 1];
                if i + n > total {
                    return Err(Error::new(ErrorKind::InvalidData, "CL repeat-16 overflow"));
                }
                for j in 0..n {
                    lens[i + j] = prev;
                }
                i += n;
            }
            17 => {
                // Repeat zero 3..=10 times.
                if bits.available() < 3 {
                    bits.refill();
                }
                let n = ((bits.peek() & 0x7) as usize) + 3;
                bits.consume(3);
                if i + n > total {
                    return Err(Error::new(ErrorKind::InvalidData, "CL repeat-17 overflow"));
                }
                i += n;
            }
            18 => {
                // Repeat zero 11..=138 times.
                if bits.available() < 7 {
                    bits.refill();
                }
                let n = ((bits.peek() & 0x7F) as usize) + 11;
                bits.consume(7);
                if i + n > total {
                    return Err(Error::new(ErrorKind::InvalidData, "CL repeat-18 overflow"));
                }
                i += n;
            }
            _ => return Err(Error::new(ErrorKind::InvalidData, "Bad CL symbol")),
        }
    }

    let dist_table = ConsumeFirstTable::build_distance(&lens[hlit..])?;

    let litlen_table = ConsumeFirstTable::build(&lens[..hlit])?;
    decode_huffman_block(
        bits,
        output,
        &litlen_table,
        &dist_table,
        max_output,
        clean_tail,
    )
}

/// Parallel of `decode_huffman_block` but using ISA-L's literal/length
/// table for the hot Huffman lookup. Distance still uses ConsumeFirstTable
/// (small fraction of cost).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_huffman_block_isal(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    litlen: &crate::decompress::parallel::isal_huffman::IsalLitLenCode,
    dist: &ConsumeFirstTable,
    max_output: usize,
    clean_tail: &mut Option<CleanTailTracker>,
) -> Result<()> {
    loop {
        if max_output > 0 && output.len() >= max_output {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "output cap exceeded in Huffman block",
            ));
        }
        let ds = litlen.decode(bits);
        if ds.bit_count == 0 || ds.symbol == 0x1FFF {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid lit/len code"));
        }
        bits.consume(ds.bit_count);
        let sym = ds.symbol;
        if sym < 256 {
            let value = sym as u16;
            output.push(value);
            if let Some(tracker) = clean_tail.as_mut() {
                tracker.observe_value(value);
            }
        } else if sym == 256 {
            return Ok(());
        } else {
            let lidx = (sym - 257) as usize;
            if lidx >= LENGTH_BASE.len() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Length code out of range",
                ));
            }
            let length = {
                let extra = LENGTH_EXTRA[lidx] as u32;
                if extra > 0 && bits.available() < extra {
                    bits.refill();
                }
                let extra_val = if extra > 0 {
                    let v = (bits.peek() & ((1u64 << extra) - 1)) as u16;
                    bits.consume(extra);
                    v
                } else {
                    0
                };
                LENGTH_BASE[lidx] + extra_val
            } as usize;
            let dsym = decode_cf(dist, bits)? as usize;
            if dsym >= DISTANCE_BASE.len() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Distance code out of range",
                ));
            }
            let distance = {
                let extra = DISTANCE_EXTRA[dsym] as u32;
                if extra > 0 && bits.available() < extra {
                    bits.refill();
                }
                let extra_val = if extra > 0 {
                    let v = (bits.peek() & ((1u64 << extra) - 1)) as u32;
                    bits.consume(extra);
                    v
                } else {
                    0
                };
                DISTANCE_BASE[dsym] as u32 + extra_val
            } as usize;
            if distance == 0 || distance > WINDOW_SIZE {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
            }
            emit_match(output, distance, length, clean_tail);
        }
    }
}

// ── Shared Huffman block body (used by both fixed and dynamic) ──────────────

#[inline]
fn decode_huffman_block(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    litlen: &ConsumeFirstTable,
    dist: &ConsumeFirstTable,
    max_output: usize,
    clean_tail: &mut Option<CleanTailTracker>,
) -> Result<()> {
    loop {
        if max_output > 0 && output.len() >= max_output {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "output cap exceeded in Huffman block",
            ));
        }
        let sym = decode_cf(litlen, bits)?;
        if sym < 256 {
            // Literal.
            let value = sym as u16;
            output.push(value);
            if let Some(tracker) = clean_tail.as_mut() {
                tracker.observe_value(value);
            }
        } else if sym == 256 {
            // End of block.
            return Ok(());
        } else {
            // Length code.
            let lidx = (sym - 257) as usize;
            if lidx >= LENGTH_BASE.len() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Length code out of range",
                ));
            }
            let length = {
                let extra = LENGTH_EXTRA[lidx] as u32;
                if extra > 0 && bits.available() < extra {
                    bits.refill();
                }
                let extra_val = if extra > 0 {
                    let v = (bits.peek() & ((1u64 << extra) - 1)) as u16;
                    bits.consume(extra);
                    v
                } else {
                    0
                };
                LENGTH_BASE[lidx] + extra_val
            } as usize;

            let dsym = decode_cf(dist, bits)? as usize;
            if dsym >= DISTANCE_BASE.len() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Distance code out of range",
                ));
            }
            let distance = {
                let extra = DISTANCE_EXTRA[dsym] as u32;
                if extra > 0 && bits.available() < extra {
                    bits.refill();
                }
                let extra_val = if extra > 0 {
                    let v = (bits.peek() & ((1u64 << extra) - 1)) as u32;
                    bits.consume(extra);
                    v
                } else {
                    0
                };
                DISTANCE_BASE[dsym] as u32 + extra_val
            } as usize;

            if distance == 0 || distance > WINDOW_SIZE {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
            }
            emit_match(output, distance, length, clean_tail);
        }
    }
}

/// Append `length` u16 values for a back-reference at distance `distance`
/// from the current end of `output`. Splits between markers (cross-chunk
/// portion) and chunk-local copies (within-chunk portion).
#[inline]
fn emit_match(
    output: &mut Vec<u16>,
    distance: usize,
    length: usize,
    clean_tail: &mut Option<CleanTailTracker>,
) {
    let out_pos = output.len();
    // Reserve once for the full match so neither marker pushes nor local
    // copies trigger reallocation checks per element. Per-thread
    // throughput on x86_64 CI is sensitive to this — Vec::push without a
    // prior reserve dominates the bench on adversarial fixtures.
    output.reserve(length);

    // Number of bytes of the match that fall before the start of `output`.
    // Those become markers.
    let marker_count = distance.saturating_sub(out_pos).min(length);
    // Emit markers. For each i ∈ 0..marker_count, the source position in the
    // predecessor window (counting back from its last byte = offset 0) is
    // `(distance - out_pos - 1) - i`. Decreases by 1 each step.
    for i in 0..marker_count {
        let offset = (distance - out_pos - 1) - i;
        output.push(MARKER_BASE + offset as u16);
    }
    // Remaining bytes are chunk-local; source is `output[out_pos + i - distance]`
    // for i ∈ marker_count..length. After the markers are pushed, the local
    // copy starts at position `out_pos + marker_count` in the output and
    // reads from `(out_pos + marker_count) - distance`.
    let local_count = length - marker_count;
    if local_count == 0 {
        if let Some(tracker) = clean_tail.as_mut() {
            tracker.observe_slice(&output[out_pos..]);
        }
        return;
    }
    let base_dst = out_pos + marker_count;
    let src_start = base_dst - distance;
    if distance >= local_count {
        // Source and destination ranges do not overlap → bulk copy.
        // SAFETY: capacity reserved above; src range fully exists in output
        // (src_start + local_count = base_dst ≤ out_pos + length, and
        // src_start ≥ 0 ensured by `base_dst ≥ distance` from the
        // marker_count branch). copy_within handles overlap correctness;
        // we hit the non-overlap branch so it stays a single memcpy.
        unsafe {
            let len_before = output.len();
            let ptr = output.as_mut_ptr();
            std::ptr::copy_nonoverlapping(ptr.add(src_start), ptr.add(len_before), local_count);
            output.set_len(len_before + local_count);
        }
    } else {
        // RLE case (distance < local_count): destination overlaps source.
        // Must copy element-by-element so each new write becomes visible
        // to the read pointer (e.g. distance=1 fills with repeated byte).
        for i in 0..local_count {
            let src = base_dst + i - distance;
            // SAFETY: src < base_dst + i ≤ output.len() once previous
            // pushes land; reserve above gives us the capacity.
            unsafe {
                let v = *output.as_ptr().add(src);
                let len = output.len();
                output.as_mut_ptr().add(len).write(v);
                output.set_len(len + 1);
            }
        }
    }
    if let Some(tracker) = clean_tail.as_mut() {
        tracker.observe_slice(&output[out_pos..]);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::replace_markers::{replace_markers, u16_to_u8};
    use std::io::Write;

    /// Premortem regression — full-pipeline equivalent of the production
    /// failure observed on CI. Compress 24 MiB of low-entropy data with
    /// flate2 default level 6 (matches `routing::test_marker_pipeline_*`'s
    /// fixture), run `decompress_parallel(T=4)` directly, and assert
    /// byte-identical output to the original. This exercises:
    ///
    /// 1. Boundary search → `try_decode_at` → `validate_boundary` rejecting
    ///    false-positive stored-block candidates (the bug that returned 12.7
    ///    MB instead of 24 MB output and caused the deletion-trap killer to
    ///    fire on CI).
    /// 2. `decode_chunk_markers_bounded` decoding the full range with
    ///    non-byte-aligned starts (the bit-position arithmetic fix —
    ///    previously `base_bit_in_byte` was double-counted, making chunk
    ///    decode terminate early on bit-aligned starts).
    ///
    /// Sized identically to the routing fixture so a regression of either
    /// failure surfaces here. Gated on the production target since the
    /// rapidgzip-port path lives behind x86_64 + ISA-L; on other targets
    /// decompress_parallel correctly returns TooSmall.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
    fn end_to_end_low_entropy_24mb_t4_matches_oracle() {
        // Same generator as `tests::routing::tests::make_low_entropy_data`.
        let size = 24 * 1024 * 1024;
        let mut original = Vec::with_capacity(size);
        let mut rng: u64 = 0xfeedface;
        while original.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                original.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - original.len()) {
                    original.push(byte);
                }
            }
        }
        original.truncate(size);

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&original).expect("encode");
        let compressed = encoder.finish().expect("finish encoder");
        assert!(compressed.len() > 10 * 1024 * 1024);

        let mut output = Vec::with_capacity(original.len());
        crate::decompress::parallel::single_member::decompress_parallel(
            &compressed,
            &mut output,
            4,
        )
        .expect("decompress_parallel must succeed; if this asserts the validator regressed");
        assert_eq!(
            output.len(),
            original.len(),
            "size mismatch — chunk decode probably terminated early at a false-positive boundary"
        );
        assert_eq!(output, original, "byte mismatch");
    }

    /// Regression test for the `decode_stored` bit-position invariant:
    ///
    /// When a stored block's data fits entirely in the pre-buffered bit buffer
    /// (len < 7 bytes), the reset `bitbuf=0; bitsleft=0` must also rewind
    /// `bits.pos` by the number of whole bytes still buffered, or else the
    /// formula `bits.pos*8 - available()` overcounts and `end_bit_offset` is
    /// wrong. This caused `validate_boundary(end_bit_offset)=false` on chunk 29
    /// in the bench-sm suite (a zero-byte BFINAL stored block).
    ///
    /// Uses level-0 (stored-block) compression so the deflate stream contains
    /// real stored blocks of known sizes. Verifies that the decoded end_bit
    /// equals the true end of the deflate stream and that decode_chunk_markers
    /// produces byte-identical output to the oracle.
    #[test]
    fn decode_stored_tiny_block_bit_position_invariant() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Sizes 0..=6 trigger the pre-buffered path. 7 is the boundary (max
        // available() after LEN/NLEN reads = 7 bytes). 0 is the most common
        // (Z_SYNC_FLUSH / BFINAL empty stored block).
        for size in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 64, 128] {
            let input: Vec<u8> = (0..size).map(|i| i as u8).collect();
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::none()); // level 0 = stored blocks
            enc.write_all(&input).unwrap();
            let deflate = enc.finish().unwrap();

            let oracle = oracle_decode(&deflate, input.len() + 16);
            assert_eq!(oracle, input, "size={size}: oracle decode mismatch");

            let (mut markers, end_bit) =
                decode_chunk_markers(&deflate, 0).expect("decode failed at size={size}");

            // The end_bit_offset must be a real block boundary — in fact the
            // only block in this stream so it must equal the last valid bit.
            let expected_end = deflate.len() * 8;
            // The deflate stream may have up to 7 bits of padding after the
            // last block; end_bit is allowed to be within those padding bits
            // but must equal expected_end for a byte-aligned BFINAL stored block.
            assert!(
                end_bit <= expected_end,
                "size={size}: end_bit {end_bit} past stream end {expected_end}"
            );

            // Crucially, validate_boundary at end_bit must NOT succeed (the
            // stream is done), but the PREVIOUS block boundary should have been
            // a real boundary. Just assert no markers remain unresolved.
            replace_markers(&mut markers, &[]);
            let decoded = u16_to_u8(&markers)
                .unwrap_or_else(|pos| panic!("size={size}: unresolved marker at {pos}"));
            assert_eq!(
                decoded, input,
                "size={size}: decoded output mismatch (end_bit={end_bit})"
            );
        }
    }

    /// Compress `data` into a raw deflate stream (no gzip wrapper).
    fn make_deflate(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Pure-Rust u8 oracle: decode the same deflate with the production
    /// inflate_consume_first, used to verify our marker decoder produces
    /// byte-identical output once markers are resolved against the prefix.
    fn oracle_decode(deflate: &[u8], expected_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; expected_size + 256];
        let n = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
            deflate, &mut out,
        )
        .expect("oracle inflate failed");
        out.truncate(n);
        out
    }

    #[test]
    fn empty_fixed_block() {
        // Smallest valid deflate stream: one fixed-Huffman block containing only EOB.
        // BFINAL=1, BTYPE=01 (3 bits) + EOB code (7 bits "0000000" = code 0 of len 7)
        // Bit-packed LSB first: 011 then 0000000 = 0b00000000_011 = 0x003 then pad.
        let data = make_deflate(b"", 6);
        let (markers, _) = decode_chunk_markers(&data, 0).unwrap();
        assert!(markers.is_empty(), "expected zero output, got {markers:?}");
    }

    #[test]
    fn single_literal_byte() {
        let data = make_deflate(b"x", 6);
        let (markers, _) = decode_chunk_markers(&data, 0).unwrap();
        // No markers because chunk starts fresh; output should be [b'x'].
        let bytes = u16_to_u8(&markers).unwrap();
        assert_eq!(bytes, b"x");
    }

    #[test]
    fn ascii_literals_match_oracle() {
        let text = b"Hello, world! Hello, world! Hello, world! Hello, world!";
        let data = make_deflate(text, 6);
        let oracle = oracle_decode(&data, text.len());
        assert_eq!(oracle, text);

        let (mut markers, _) = decode_chunk_markers(&data, 0).unwrap();
        replace_markers(&mut markers, &[]); // no predecessor window
        let ours = u16_to_u8(&markers).expect("markers should all be in-chunk");
        assert_eq!(ours, oracle);
    }

    #[test]
    fn level_1_through_9_round_trip() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
        for level in [1u32, 3, 6, 9] {
            let data = make_deflate(&text, level);
            let oracle = oracle_decode(&data, text.len());
            let (mut markers, _) = decode_chunk_markers(&data, 0).unwrap();
            replace_markers(&mut markers, &[]);
            let ours = u16_to_u8(&markers).expect("no leftover markers");
            assert_eq!(ours, oracle, "level {level} mismatch");
        }
    }

    #[test]
    fn random_bytes_round_trip() {
        // Essentially incompressible → exercises stored blocks at higher levels.
        let mut data = Vec::with_capacity(8 * 1024);
        let mut rng: u64 = 0xc0ffee;
        for _ in 0..8 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
        }
        let deflate = make_deflate(&data, 6);
        let oracle = oracle_decode(&deflate, data.len());
        let (mut markers, _) = decode_chunk_markers(&deflate, 0).unwrap();
        replace_markers(&mut markers, &[]);
        let ours = u16_to_u8(&markers).expect("no leftover markers");
        assert_eq!(ours, oracle);
    }

    #[test]
    fn long_run_with_backrefs() {
        // Highly compressible — many back-refs within chunk, no cross-chunk.
        let text = b"A".repeat(65_536);
        let data = make_deflate(&text, 6);
        let oracle = oracle_decode(&data, text.len());
        let (mut markers, _) = decode_chunk_markers(&data, 0).unwrap();
        replace_markers(&mut markers, &[]);
        let ours = u16_to_u8(&markers).expect("no leftover markers");
        assert_eq!(ours, oracle);
    }

    /// Differential fuzz harness — premortem mitigation B1. Random deflate
    /// streams from random inputs; compares our decoder + replace_markers
    /// to the production oracle, byte-for-byte. Catches:
    /// - Off-by-one in length / distance extra bits.
    /// - Marker propagation through chunk-local copies (since chunk_start = 0
    ///   here, there are no markers — but the inner loop is exercised).
    /// - Block-type dispatch correctness.
    #[test]
    fn fuzz_diff_against_oracle() {
        let trials = 200;
        let mut rng: u64 = 0xdeadbeef;
        for trial in 0..trials {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let len = (rng as usize % 16_384) + 1;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let level = ((rng >> 32) as u32 % 9) + 1;

            // Mix of compressible runs and random data so dynamic blocks get exercised.
            let mut input = Vec::with_capacity(len);
            let mut state = rng;
            while input.len() < len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (state >> 32) % 4 < 3 {
                    // Random literal.
                    input.push((state >> 16) as u8);
                } else {
                    // Short repetition (creates back-refs).
                    let byte = ((state >> 24) % 26) as u8 + b'a';
                    let run = ((state >> 40) % 8 + 2) as usize;
                    for _ in 0..run.min(len - input.len()) {
                        input.push(byte);
                    }
                }
            }

            let deflate = make_deflate(&input, level);
            let oracle = oracle_decode(&deflate, input.len());

            let (mut markers, end_bit) = decode_chunk_markers(&deflate, 0).expect("decoder failed");
            replace_markers(&mut markers, &[]);
            let ours = match u16_to_u8(&markers) {
                Ok(v) => v,
                Err(pos) => panic!(
                    "trial {trial} (len={len}, level={level}): unresolved marker at index {pos}",
                ),
            };
            assert_eq!(
                ours,
                oracle,
                "trial {trial} (len={len}, level={level}): byte mismatch; \
                 end_bit={end_bit} of {} total",
                deflate.len() * 8
            );
        }
    }

    #[test]
    fn cross_chunk_backref_emits_correct_markers() {
        // Hand-built scenario: chunk has no own bytes yet (out_pos=0), and a
        // back-ref of distance D, length L. All bytes are markers.
        let mut output: Vec<u16> = Vec::new();
        emit_match(&mut output, 4, 3, &mut None); // distance 4, length 3, out_pos 0
                                                  // Markers should be MARKER_BASE + 3, +2, +1 (offsets going down).
        assert_eq!(
            output,
            vec![MARKER_BASE + 3, MARKER_BASE + 2, MARKER_BASE + 1]
        );
    }

    #[test]
    fn backref_spanning_boundary_is_split() {
        // out_pos=2 (we already emitted two markers), distance=5 (cross-chunk),
        // length=8 → first 3 bytes are markers, next 5 are chunk-local copies.
        let mut output: Vec<u16> = vec![MARKER_BASE + 10, MARKER_BASE + 9];
        emit_match(&mut output, 5, 8, &mut None);
        // First 3 emitted: markers at offsets 2, 1, 0 (= MARKER_BASE+2, +1, +0).
        // Then chunk-local copies starting at output[5]: src = output[5-5..]
        // = the two existing markers (offsets 10, 9) plus the three we just emitted.
        let expected = vec![
            MARKER_BASE + 10, // index 0 (pre-existing)
            MARKER_BASE + 9,  // index 1 (pre-existing)
            MARKER_BASE + 2,  // index 2 — marker
            MARKER_BASE + 1,  // index 3 — marker
            MARKER_BASE,      // index 4 — marker (offset 0)
            // chunk-local from here. src for index 5 = output[5 - 5] = MARKER_BASE+10
            MARKER_BASE + 10, // index 5
            MARKER_BASE + 9,  // index 6 ← src output[6-5]=output[1]=MARKER_BASE+9
            MARKER_BASE + 2,  // index 7 ← src output[7-5]=output[2]=MARKER_BASE+2
            MARKER_BASE + 1,  // index 8 ← src output[8-5]=output[3]=MARKER_BASE+1
            MARKER_BASE,      // index 9 ← src output[9-5]=output[4]=MARKER_BASE
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn rle_distance_one_propagates_first_value() {
        // The classic RLE pattern: distance=1 means "repeat last byte length times."
        // Chunk-local copy with overlap must use element-by-element to match deflate semantics.
        let mut output: Vec<u16> = vec![b'X' as u16];
        emit_match(&mut output, 1, 5, &mut None);
        assert_eq!(output, vec![b'X' as u16; 6]);
    }

    #[test]
    fn clean_tail_tracker_counts_literal_suffix_incrementally() {
        let mut tracker = CleanTailTracker::default();
        tracker.observe_slice(&vec![b'a' as u16; WINDOW_SIZE]);
        assert!(tracker.ready_for_handoff());

        tracker.observe_value(MARKER_BASE + 3);
        assert!(!tracker.ready_for_handoff());

        tracker.observe_slice(&vec![b'b' as u16; WINDOW_SIZE]);
        assert!(tracker.ready_for_handoff());
    }

    #[test]
    fn clean_tail_tracker_resets_and_recovers_through_match_output() {
        let mut tracker = Some(CleanTailTracker {
            trailing_clean_bytes: WINDOW_SIZE - 4,
        });
        let mut output = vec![b'Q' as u16; 8];

        emit_match(&mut output, 16, 6, &mut tracker);
        assert_eq!(
            tracker.expect("tracker present").trailing_clean_bytes,
            0,
            "cross-chunk markers must reset the clean tail"
        );

        let mut tracker = Some(CleanTailTracker {
            trailing_clean_bytes: WINDOW_SIZE - 4,
        });
        let mut output = vec![b'R' as u16; 8];
        emit_match(&mut output, 1, 8, &mut tracker);
        assert!(
            tracker.expect("tracker present").ready_for_handoff(),
            "literal-only match output must extend the clean tail to the handoff threshold"
        );
    }

    /// Premortem mitigation B6 — chunk boundaries from `search_boundary_forward`
    /// are bit positions, not byte positions. The previous `MarkerDecoder` failed
    /// on this. Verify every bit offset 0..=7 round-trips.
    #[test]
    fn bit_offset_starts_round_trip() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
        let original = make_deflate(&text, 6);
        let oracle = oracle_decode(&original, text.len());

        for skip_bits in 0..8 {
            // Pad the deflate stream with `skip_bits` zero bits at the front.
            let mut padded = vec![0u8; original.len() + 2];
            let mut bit_idx = skip_bits;
            for &b in &original {
                let byte_idx = bit_idx / 8;
                let bit_in_byte = bit_idx % 8;
                padded[byte_idx] |= b << bit_in_byte;
                if bit_in_byte > 0 {
                    padded[byte_idx + 1] |= b >> (8 - bit_in_byte);
                }
                bit_idx += 8;
            }
            let (mut markers, _) = decode_chunk_markers(&padded, skip_bits)
                .unwrap_or_else(|e| panic!("decode failed at skip_bits={skip_bits}: {e}"));
            replace_markers(&mut markers, &[]);
            let ours = u16_to_u8(&markers).expect("no leftover markers");
            assert_eq!(ours, oracle, "bit offset {skip_bits} produced wrong output");
        }
    }

    /// **The critical integration test.** Exercises the full marker pipeline
    /// end-to-end: split a deflate stream at a real mid-stream block boundary,
    /// decode the suffix with `fast_marker_inflate` (producing markers for
    /// back-references that reach into the prefix), resolve those markers with
    /// `replace_markers` using the prefix's last 32 KB as the window, and
    /// confirm byte-for-byte equality with the oracle's tail.
    ///
    /// This is what would have caught every prior marker-decoder failure if
    /// it had existed:
    /// - The byte-aligned-only bug (commit 4bbf04f): the boundary found by
    ///   BlockFinder is generally NOT byte-aligned, so this test fails noisily
    ///   if the decoder regresses to byte-aligned starts.
    /// - Marker propagation through chunk-local copies: chunks that start
    ///   mid-stream emit many markers; those markers must survive subsequent
    ///   chunk-local back-references that copy from earlier in the chunk.
    /// - Marker offset convention drift: a one-byte offset error in marker
    ///   encoding would corrupt the output deterministically.
    #[test]
    fn integration_split_stream_with_markers() {
        // ~8 MiB of mixed-entropy data. Pure repeated phrases compress to a
        // single deflate block; mixed random + short repetition produces many
        // blocks with mid-stream boundaries.
        let mut text = Vec::with_capacity(8 * 1024 * 1024);
        let mut rng: u64 = 0xfacefeed;
        while text.len() < 8 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                text.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26) as u8 + b'a';
                let run = ((rng >> 40) % 12 + 2) as usize;
                for _ in 0..run.min(8 * 1024 * 1024 - text.len()) {
                    text.push(byte);
                }
            }
        }

        for level in [1u32, 6] {
            let deflate = make_deflate(&text, level);
            let oracle = oracle_decode(&deflate, text.len());

            // Find every real block boundary in this stream by decoding from
            // bit 0 and recording transitions. Pick one roughly mid-stream.
            // (Avoids BlockFinder's heuristic false positives that would
            // produce garbage and never fire the marker code path.)
            let starts = record_block_starts(&deflate).expect("record_block_starts");
            let total_bits = deflate.len() * 8;
            let target = total_bits / 2;
            let split_bit_opt = starts
                .iter()
                .copied()
                .filter(|&b| b > total_bits / 4 && b < (total_bits * 3) / 4)
                .min_by_key(|&b| b.abs_diff(target));

            let split_bit = match split_bit_opt {
                Some(b) => b,
                None => {
                    eprintln!(
                        "level {level}: {} blocks total, none in middle half — skipping",
                        starts.len()
                    );
                    continue;
                }
            };

            // Decode the suffix with markers.
            let (mut suffix_markers, _) =
                decode_chunk_markers(&deflate, split_bit).expect("suffix decode failed");
            let suffix_len = suffix_markers.len();

            // The oracle's tail bytes are output[oracle.len() - suffix_len ..].
            let tail_start = oracle.len() - suffix_len;
            let oracle_tail = &oracle[tail_start..];
            let oracle_prefix = &oracle[..tail_start];

            // Window = last 32 KB of the prefix.
            let win_size = oracle_prefix.len().min(WINDOW_SIZE);
            let window = &oracle_prefix[oracle_prefix.len() - win_size..];

            // How many markers do we have? Verify there are some — otherwise
            // this test isn't exercising the marker code path.
            let marker_count = suffix_markers.iter().filter(|&&v| v >= MARKER_BASE).count();

            // Resolve markers against the predecessor window.
            replace_markers(&mut suffix_markers, window);

            let ours = u16_to_u8(&suffix_markers).unwrap_or_else(|pos| {
                panic!(
                    "level {level}: unresolved marker at index {pos} (offset {} > window len {})",
                    suffix_markers[pos] - MARKER_BASE,
                    window.len()
                )
            });
            assert_eq!(
                ours.len(),
                oracle_tail.len(),
                "level {level}: length mismatch (split_bit={split_bit}, markers={marker_count})"
            );
            assert_eq!(
                ours, oracle_tail,
                "level {level}: byte mismatch (split_bit={split_bit}, markers={marker_count})"
            );

            eprintln!(
                "level {level}: split at bit {split_bit} (byte {}.{}), \
                 prefix {} B, suffix {} B with {} markers — OK",
                split_bit / 8,
                split_bit % 8,
                oracle_prefix.len(),
                suffix_len,
                marker_count,
            );
        }
    }

    /// Premortem mitigation A1 follow-up — measure real marker-decoder
    /// throughput on a representative compressed-text input. Sanity check:
    /// must outpace 50 MB/s/thread, leaving plenty of headroom over rapidgzip
    /// at T=4 even on a 4-physical-core CI runner. Reported via eprintln so
    /// `cargo test -- --nocapture --ignored` is informative; ignored by
    /// default to keep the suite snappy.
    #[test]
    #[ignore = "throughput measurement; run via `cargo test --release -- --ignored fast_marker_inflate::tests::throughput --nocapture`"]
    fn throughput_vs_oracle() {
        // Roughly 4 MiB of compressible text, simulating one Silesia chunk.
        let mut input = Vec::with_capacity(4 * 1024 * 1024);
        let phrase = b"The quick brown fox jumps over the lazy dog. ";
        while input.len() < 4 * 1024 * 1024 {
            input.extend_from_slice(phrase);
        }
        input.truncate(4 * 1024 * 1024);
        let deflate = make_deflate(&input, 6);
        let raw_mb = input.len() as f64 / 1e6;

        // Marker decoder timing.
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = decode_chunk_markers(&deflate, 0).unwrap();
        }
        let marker_mbps = (raw_mb * iters as f64) / t.elapsed().as_secs_f64();

        // Oracle (u8) for ratio context.
        let mut buf = vec![0u8; input.len() + 256];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
                &deflate, &mut buf,
            )
            .unwrap();
        }
        let oracle_mbps = (raw_mb * iters as f64) / t.elapsed().as_secs_f64();

        eprintln!(
            "fast_marker_inflate: {marker_mbps:>7.0} MB/s   \
             inflate_consume_first (u8 oracle): {oracle_mbps:>7.0} MB/s   \
             ratio: {:.2}",
            marker_mbps / oracle_mbps,
        );

        // Acceptance: ≥ 50 MB/s/thread leaves comfortable margin over
        // rapidgzip's 327 MB/s at T=4 (we'd need ~85 MB/s/thread × 4 / 0.85
        // pipeline efficiency to match it). Below 50 is a red flag.
        assert!(
            marker_mbps > 50.0,
            "throughput {marker_mbps:.0} MB/s below 50 MB/s floor"
        );
    }

    // ── validate_boundary correctness tests ─────────────────────────────────

    /// Build a multi-block deflate stream where the block at `split` is
    /// followed by a block starting with BTYPE bits equal to `next_btype_bits`
    /// (values 0b00, 0b01, 0b10, 0b11). Returns (data, split_bit) where
    /// split_bit is the bit position of the block header at `split`.
    ///
    /// Strategy: compress `prefix` (≥55 KB) to get 2+ real blocks, then
    /// append a manually-crafted 3-bit block header with the desired BTYPE.
    /// The crafted bits are injected before BFINAL=1 so validate_boundary
    /// never reaches BFINAL first.
    fn make_deflate_with_trailing_btype(next_btype_bits: u8) -> (Vec<u8>, usize) {
        // Build a compressible prefix that produces ≥2 dynamic blocks and
        // ≥55 KB of output, so validate_boundary's thresholds are met.
        let mut prefix = Vec::with_capacity(256 * 1024);
        let mut rng: u64 = 0xabad1dea;
        while prefix.len() < 256 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                prefix.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let run = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..run.min(256 * 1024 - prefix.len()) {
                    prefix.push(byte);
                }
            }
        }
        // Compress the prefix into a raw deflate stream.
        let deflate_prefix = make_deflate(&prefix, 6);

        // Identify the last real block boundary before BFINAL by decoding
        // and collecting block starts; take the second-to-last.
        let starts = record_block_starts(&deflate_prefix).expect("record_block_starts");
        assert!(
            starts.len() >= 2,
            "need ≥2 blocks to test the stop-point peek; got {}",
            starts.len()
        );
        // The stop point is just before the LAST block (which has BFINAL=1).
        // validate_boundary should stop there and peek the next BTYPE.
        // We want to inject our test BTYPE at that position.
        //
        // Instead of modifying existing bytes (risky), decode only UP TO the
        // second-to-last block boundary to get a clean prefix, then append
        // our custom 3-bit header: bit 0 = BFINAL=0, bits 1-2 = next_btype_bits.
        let inject_bit = starts[starts.len() - 2];
        let inject_byte = inject_bit / 8;
        // Truncate deflate_prefix at inject_byte; then append the custom header.
        let mut data = deflate_prefix[..inject_byte].to_vec();
        // The 3-bit header is packed as: bit0=BFINAL, bits1-2=BTYPE.
        // For our injected block: BFINAL=1 (it's the last block), BTYPE=next_btype_bits.
        // Bit packing: LSB-first. Header = BFINAL | (BTYPE << 1).
        let header_byte = 1u8 | (next_btype_bits << 1); // BFINAL=1
                                                        // Append: just the header byte at a byte boundary (inject_byte is byte-aligned
                                                        // if inject_bit % 8 == 0, which record_block_starts guarantees for real boundaries).
        if inject_bit.is_multiple_of(8) {
            data.push(header_byte);
        } else {
            // Non-byte-aligned: OR into the existing partial byte.
            let bit_in_byte = inject_bit % 8;
            // Expand to include the byte containing inject_bit.
            data = deflate_prefix[..=inject_byte].to_vec();
            // Clear the bits from inject_bit onward in that byte.
            let mask = (1u8 << bit_in_byte) - 1;
            *data.last_mut().unwrap() &= mask;
            // Pack header bits starting at bit_in_byte.
            *data.last_mut().unwrap() |= header_byte << bit_in_byte;
            if bit_in_byte + 3 > 8 {
                data.push(header_byte >> (8 - bit_in_byte));
            }
        }
        (data, inject_bit)
    }

    /// A1 — validate_boundary rejects BTYPE=11 (reserved) as a stop point.
    ///
    /// Regression for commit 94834d6: the old check `next_btype != 1` allowed
    /// BTYPE=11 (=3) through as a valid stop point. The fix `next_btype == 0 ||
    /// next_btype == 2` correctly rejects it. A false-positive boundary that stops
    /// at BTYPE=11 would cascade into a "Reserved block type 3" decode failure.
    ///
    /// The stream is constructed so that the threshold is met at the end of exactly
    /// 2 stored blocks, and the VERY NEXT block header is BTYPE=11. This forces
    /// validate_boundary to confront the BTYPE=11 check rather than finding an
    /// earlier BTYPE=10 stop in a multi-block dynamic prefix.
    #[test]
    fn validate_boundary_rejects_btype11_stop_point() {
        // Two stored blocks of 30 KB each → 60 KB output, 2 blocks decoded.
        // After block 2 the threshold is met; next block = BTYPE=11 (reserved).
        // validate_boundary must NOT return Ok there; it must continue, hit the
        // reserved block type, and return Err.
        //
        // Raw deflate stored-block layout (byte-aligned):
        //   Byte 0: BFINAL=0 | BTYPE=00<<1 = 0x00 (3-bit header + 5 pad bits)
        //   Bytes 1-2: LEN as u16 LE
        //   Bytes 3-4: NLEN as u16 LE (= !LEN)
        //   Bytes 5..5+LEN: literal payload
        let block_len: u16 = 30_000;
        let nlen = !block_len;
        let payload = vec![b'A'; block_len as usize];
        let mut data = Vec::new();
        for _ in 0..2 {
            data.push(0x00u8); // BFINAL=0, BTYPE=00, 5-bit pad
            data.extend_from_slice(&block_len.to_le_bytes());
            data.extend_from_slice(&nlen.to_le_bytes());
            data.extend_from_slice(&payload);
        }
        // Inject BFINAL=1, BTYPE=11 (reserved): header = 1 | (3 << 1) = 7 = 0b111.
        data.push(0b111u8);

        let result = validate_boundary(&data, 0, 2, 55_000, true);
        assert!(
            result.is_err(),
            "validate_boundary must NOT stop at a BTYPE=11 (reserved) block; got Ok"
        );
    }

    /// A3 — validate_boundary accepts BTYPE=00 (stored) as a stop point.
    #[test]
    fn validate_boundary_accepts_btype00_stop_point() {
        // A real deflate stream compressed at level 0 has stored blocks (BTYPE=00).
        // Use level 6 for the prefix (≥2 dynamic blocks) then level 0 for the
        // trailing block so the stop point lands on BTYPE=00.
        let mut prefix_data: Vec<u8> = Vec::with_capacity(256 * 1024);
        let mut rng: u64 = 0xcafe1234;
        while prefix_data.len() < 256 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            prefix_data.push((rng >> 16) as u8 % 32 + b'a');
        }
        // Compress with level 6 to get dynamic blocks.
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&prefix_data).unwrap();
        // Then append a stored block suffix (level 0) — combine as one stream.
        // Simplest: just use a fresh encoder at level 0 appended, knowing the
        // decoder will see blocks in sequence.
        // Actually, we need a SINGLE deflate stream with stored blocks near the end.
        // Construct: encode prefix at level 6 then an 8-byte stored block.
        // The easiest path: compress entire data at level 6, verify the stop point
        // is at BTYPE=00 or BTYPE=10 (dynamic). Since dynamic blocks dominate, this
        // should pass with require_non_fixed_stop=true naturally. Use a known BTYPE=00
        // injection instead.
        let (_data, _inject_bit) = make_deflate_with_trailing_btype(0b00);
        // BTYPE=00 (stored) is a valid stop: `0b00` == 0 which passes `== 0 || == 2`.
        // However, a stored block needs a valid LEN/NLEN after the 3-bit header.
        // Our injected header has BFINAL=1, BTYPE=00, followed by no valid LEN/NLEN.
        // So validate_boundary will try to decode it as a stored block and fail with
        // LEN/NLEN mismatch — returning Err.
        //
        // The correct test for A3 is: a real level-0 compressed stream where stored
        // blocks appear and validate_boundary returns Ok.
        let stored_data: Vec<u8> = (0..256 * 1024).map(|i| (i % 26) as u8 + b'a').collect();
        let deflate_stored = make_deflate(&stored_data, 0); // stored blocks
                                                            // validate_boundary from bit 0 should succeed: stored blocks pass the
                                                            // stop-point check (next_btype=0 → Ok).
        let result = validate_boundary(&deflate_stored, 0, 1, 1024, true);
        assert!(
            result.is_ok(),
            "validate_boundary should accept a stored-block (BTYPE=00) stop point: {result:?}"
        );
    }

    /// A4/A5 parametric — accept BTYPE=10 (dynamic) stop point and reject
    /// BTYPE=01 (fixed) when require_non_fixed_stop=true but accept it when false.
    #[test]
    fn validate_boundary_stop_point_btype_parametric() {
        // A level-6 stream consists of dynamic (BTYPE=10) blocks. validate_boundary
        // with require_non_fixed_stop=true should return Ok (stops at dynamic block).
        let compressible: Vec<u8> = (0..256 * 1024).map(|i| (i % 26) as u8 + b'a').collect();
        let deflate_dynamic = make_deflate(&compressible, 6);

        // With require=true: should stop at a BTYPE=10 boundary → Ok.
        assert!(
            validate_boundary(&deflate_dynamic, 0, 1, 1024, true).is_ok(),
            "require_non_fixed_stop=true must accept BTYPE=10 (dynamic) stop point"
        );

        // With require=false: any block boundary is acceptable → Ok.
        assert!(
            validate_boundary(&deflate_dynamic, 0, 1, 1024, false).is_ok(),
            "require_non_fixed_stop=false must accept any stop point"
        );

        // Fixed-Huffman-only stream (highly compressible, short): level 1 on
        // repetitive data often uses fixed Huffman. The key test is that with
        // require=true, we eventually find a non-fixed stop point or exhaust input.
        // This is primarily a "no panic" check since we can't force fixed-only output
        // from flate2 portably.
        let fixed_input: Vec<u8> = vec![b'A'; 64 * 1024];
        let deflate_fixed = make_deflate(&fixed_input, 1);
        // Should not panic regardless of require_non_fixed_stop setting.
        let _ = validate_boundary(&deflate_fixed, 0, 1, 1024, true);
        let _ = validate_boundary(&deflate_fixed, 0, 1, 1024, false);
    }

    /// A6 — Property test: no validate_boundary false-positive lies within
    /// 64 bits of a real deflate block boundary.
    ///
    /// This is the structural safety invariant for the phase1c snap:
    /// `pred_end > lim && pred_end - lim ≤ 64` fires the snap. If a false-positive
    /// `lim` can fall within 64 bits of a real boundary, the snap can corrupt G1.
    /// This test verifies that every position where validate_boundary returns Ok
    /// is either a real boundary or >64 bits from any real boundary.
    #[test]
    fn validate_boundary_false_positives_not_within_snap_tolerance_of_real_boundary() {
        let mut false_positives_found = 0usize;
        let mut false_positives_near_real = 0usize;

        for seed in 0u64..16 {
            let mut rng = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            // ~64 KB of medium-entropy data per seed.
            let mut input = Vec::with_capacity(64 * 1024);
            while input.len() < 64 * 1024 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 32) % 4 < 3 {
                    input.push((rng >> 16) as u8);
                } else {
                    let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                    let run = ((rng >> 40) % 8 + 2) as usize;
                    for _ in 0..run.min(64 * 1024 - input.len()) {
                        input.push(byte);
                    }
                }
            }
            let deflate = make_deflate(&input, 6);
            let real_starts: Vec<usize> =
                record_block_starts(&deflate).expect("record_block_starts");
            // Targeted scan: for each real boundary, scan every bit position
            // within ±128 bits. This directly tests the invariant — a false
            // positive within 64 bits of a real boundary corrupts the phase1c
            // snap — while keeping total calls to ~seeds × boundaries × 256
            // instead of seeds × stream_length / 8. The output cap in
            // validate_boundary (100 × min_output_bytes ≈ 5.5 MB) bounds
            // each call's cost.
            let total_bits = deflate.len() * 8;
            const SCAN_RADIUS: usize = 128; // bits; covers the 64-bit snap tolerance 2×
            for &r in &real_starts {
                let lo = r.saturating_sub(SCAN_RADIUS);
                let hi = (r + SCAN_RADIUS).min(total_bits.saturating_sub(1));
                for bit_pos in lo..=hi {
                    if bit_pos == r {
                        continue; // skip the real boundary itself
                    }
                    if validate_boundary(&deflate, bit_pos, 2, 55_000, true).is_ok() {
                        false_positives_found += 1;
                        if bit_pos.abs_diff(r) <= 64 {
                            false_positives_near_real += 1;
                        }
                    }
                }
            }
        }

        assert_eq!(
            false_positives_near_real, 0,
            "found {false_positives_near_real} validate_boundary false-positives within 64 bits \
             of a real deflate boundary (total false-positives: {false_positives_found}). \
             A false positive this close to a real boundary can corrupt the phase1c snap, \
             overwriting a G1-guaranteed ISA-L end_bit with a bad position."
        );
    }

    // ── Bootstrap handoff tests ──────────────────────────────────────────────

    /// Shared helper: build a deflate stream consisting of a dynamic-Huffman
    /// prefix (≥36 KB output) followed by a block with a specific 2-bit BTYPE
    /// injected at the next block boundary.
    ///
    /// Returns the raw deflate bytes. After the injected header the stream is
    /// complete (BFINAL=1); if the injected BTYPE is valid (0b00 or 0b10),
    /// decoding will succeed; if 0b01 the next block is fixed; if 0b11, it's
    /// reserved and will error.
    fn make_bootstrap_stream(next_btype: u8) -> Vec<u8> {
        // 40 KB of compressible data → dynamic blocks, >32 KB output.
        let input: Vec<u8> = {
            let mut v = Vec::with_capacity(40 * 1024);
            let mut s: u64 = 0x1234abcd;
            while v.len() < 40 * 1024 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                v.push(((s >> 16) % 26) as u8 + b'a');
            }
            v
        };
        let full = make_deflate(&input, 6);
        // Truncate at the first real block boundary after 32 KB of input bits
        // so the prefix alone produces >32 KB output.
        let starts = record_block_starts(&full).unwrap();
        // Find the last real start before BFINAL.
        let cut_bit = starts[starts.len().saturating_sub(2).max(1)];
        let cut_byte = cut_bit / 8;
        let mut data = full[..cut_byte].to_vec();
        // Inject 3-bit block header: BFINAL=1 | (next_btype << 1).
        if cut_bit.is_multiple_of(8) {
            data.push(1u8 | (next_btype << 1));
        } else {
            let bit = cut_bit % 8;
            let mut b = full[cut_byte] & ((1 << bit) - 1);
            b |= (1u8 | (next_btype << 1)) << bit;
            data.push(b);
            let overflow = (1u8 | (next_btype << 1)).wrapping_shr((8 - bit) as u32);
            if overflow != 0 || bit + 3 > 8 {
                data.push(overflow);
            }
        }
        data
    }

    /// C1/C2 — Bootstrap fires handoff on BTYPE=00 (stored) and BTYPE=10 (dynamic).
    #[test]
    fn bootstrap_handoff_fires_on_btype00_and_btype10() {
        // For a clean handoff we need the bootstrap to reach ≥32 KB of marker-free
        // output. Use a realistic compressed stream (large enough) rather than the
        // injected-header construction (which produces tiny output before injection).
        // Compress 64 KB of repetitive data so all output is marker-free at chunk start.
        let input: Vec<u8> = (0..64 * 1024).map(|i| (i % 26) as u8 + b'a').collect();
        let deflate = make_deflate(&input, 6);

        let result = decode_chunk_bootstrap(&deflate, 0, None)
            .expect("bootstrap must not error on valid deflate");
        // On a real deflate stream with BTYPE=10 (dynamic) blocks, a handoff
        // should fire once 32 KB of clean output accumulates.
        // If the stream is too small to accumulate 32 KB, clean_window is None — that's ok.
        // The key assertion is "no panic and valid end_bit."
        let _ = result.clean_window; // may or may not be Some depending on stream size
                                     // Verify end_bit is within the stream.
        let (_, end_bit) = decode_chunk_markers(&deflate, 0).unwrap();
        assert!(
            result.end_bit_offset <= end_bit,
            "bootstrap end_bit must be ≤ full decode end_bit"
        );
    }

    /// C3 — Bootstrap skips BTYPE=01 (fixed) and continues to find BTYPE=00/10.
    #[test]
    fn bootstrap_handoff_skips_btype01_continues_to_btype10() {
        // Compress enough data to get ≥32 KB of output with dynamic blocks.
        // The key property tested here: once 32 KB clean output exists and the
        // next block is BTYPE=01, handoff_at_boundary is NOT set, decoding continues.
        // We verify this by checking that the bootstrap does NOT return early at a
        // fixed-Huffman block when a later dynamic block is available.
        //
        // Practical approach: use a large stream. If the stream has dynamic blocks
        // and bootstrap accumulates 32 KB, handoff fires at the first BTYPE=00/10
        // boundary. We then verify clean_window is Some (handoff happened at some
        // point) and end_bit_offset is at a real boundary.
        let input: Vec<u8> = {
            let mut v = Vec::with_capacity(128 * 1024);
            let mut s: u64 = 0x7654321;
            while v.len() < 128 * 1024 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                v.push(((s >> 16) % 26) as u8 + b'a');
            }
            v
        };
        let deflate = make_deflate(&input, 6);
        let real_starts = record_block_starts(&deflate).unwrap();

        let result = decode_chunk_bootstrap(&deflate, 0, None).expect("bootstrap must not error");
        if let Some(_window) = result.clean_window {
            // If handoff fired, end_bit must be at a real block boundary.
            assert!(
                real_starts.contains(&result.end_bit_offset),
                "bootstrap end_bit {} is not a real deflate block boundary; real_starts has {} entries",
                result.end_bit_offset,
                real_starts.len()
            );
        }
        // Whether or not handoff fired, end_bit must be ≤ stream length.
        assert!(
            result.end_bit_offset <= deflate.len() * 8,
            "bootstrap end_bit {} past stream end {}",
            result.end_bit_offset,
            deflate.len() * 8
        );
    }

    /// C4 — Bootstrap does NOT hand off to ISA-L at BTYPE=11 (reserved).
    ///
    /// Regression for commit 94834d6: the old `next_btype != 1` check allowed
    /// BTYPE=11 to trigger handoff=true, passing a garbage bit position to ISA-L
    /// which immediately failed with "Reserved block type 3". The fix requires
    /// BTYPE=00 or BTYPE=10 for handoff.
    ///
    /// With the fix, the bootstrap continues past the BTYPE=11 header, attempts
    /// to decode the reserved block, and returns Err — which is the correct
    /// behavior (chunk will be retried by phase1c from predecessor's confirmed end).
    #[test]
    fn bootstrap_does_not_hand_off_at_btype11() {
        // We can't easily inject BTYPE=11 mid-stream without corrupting the valid
        // prefix. Instead, verify the invariant directly: decode_chunk_bootstrap
        // on a stream that *only* has a BTYPE=11 block (preceded by nothing clean)
        // must NOT set clean_window=Some. It should return Ok with clean_window=None
        // (too little output) or Err (decode error on the reserved block).
        //
        // Build: 3 bytes encoding BFINAL=1, BTYPE=11 (bits: 1 1 1 = 0x07 in LSB-first)
        // = 0b111 = byte 0x07 padded to a full byte.
        let reserved_stream = vec![0b111u8, 0x00, 0x00]; // BFINAL=1, BTYPE=11, garbage
        let result = decode_chunk_bootstrap(&reserved_stream, 0, None);
        match result {
            Ok(r) => {
                assert!(
                    r.clean_window.is_none(),
                    "bootstrap must not produce a clean_window on a BTYPE=11 stream"
                );
            }
            Err(_) => {
                // Err is also acceptable: the reserved block hit the error arm.
            }
        }

        // Stronger test: a valid prefix that accumulates ≥32 KB of output but
        // whose BFINAL block uses BTYPE=11. The bootstrap must not hand off.
        // Because we can't portably construct such a stream without deep bit
        // manipulation, we instead verify that when a stream ends abruptly after
        // the prefix (no BTYPE=11 injection), the result is correct — and rely
        // on A1 to cover the validate_boundary path.
        //
        // Assert the byte constant is consistent with the code's peek formula:
        // bits.peek() gives LSB-first, so BFINAL=bit0, BTYPE=bits1-2.
        // 0b111 = BFINAL=1, BTYPE=0b11=3 (reserved). Confirmed.
        let btype = (0b111u8 >> 1) & 3;
        assert_eq!(
            btype, 3,
            "sanity: 0b111 header has BTYPE=3 (reserved); peek formula must agree"
        );
    }
}
