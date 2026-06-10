#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup
//! Stateless windowed-bulk DEFLATE decoder using the ISA-L LUT format
//! from `lut_huffman.rs`.
//!
//! ## Why this exists
//!
//! The parallel-SM bulk phase (post-bootstrap, with a known 32 KiB
//! predecessor window) currently routes through `ResumableInflate2`
//! (`src/decompress/inflate/resumable.rs`). Per the 2026-05-27 neurotic
//! perf profile, ResumableInflate2's bulk hot loop accounts for
//!   - 9.39% CPU in `decode_huffman_body_resumable`
//!   - 6.04% CPU in `copy_match_windowed`
//!
//! ResumableInflate2 carries state-machine yield-check overhead from
//! supporting the speculative-decode use case. For the BULK windowed
//! path, no yields are needed — we have a clean output buffer, known
//! input range, no marker emission.
//!
//! ## Design
//!
//! - **Stateless.** No yield-on-output-fill, no resume tokens. Caller
//!   provides the entire (compressed) bit range AND the entire output
//!   buffer (sized to known bounds from the chunk config). The decoder
//!   either completes the block or errors.
//! - **Single-block.** Decodes ONE block per call; caller loops.
//! - **No markers.** Windowed path emits clean u8 directly.
//! - **Zero per-block allocation.** All scratch lives in
//!   [`DecoderScratch`] owned by the caller, reused across blocks.
//! - **Full-output back-ref reach.** Back-references reach into ALL
//!   of `output[..out_pos]` (every byte the caller has decoded into
//!   this output buffer so far, including prior blocks in this chunk).
//!   `predecessor_window` is consulted only for `distance > out_pos`.
//!
//! ## Correctness contract
//!
//! - Caller passes `bits` positioned at the start of a deflate block
//!   header (BFINAL+BTYPE bits next).
//! - Caller passes `output` sized to hold all bytes decoded into this
//!   buffer so far PLUS this block's output.
//! - Caller passes `out_pos` (mutable). The decoder writes starting at
//!   `output[*out_pos]` and advances `*out_pos` by the bytes it wrote.
//! - `predecessor_window` is the bytes immediately preceding
//!   `output[0]`. For a chunk that owns the whole stream from offset 0,
//!   pass `&[]` (or any slice — it won't be read if `out_pos > 0` and
//!   no back-ref exceeds the in-buffer history).
//! - Returns whether this was the final block.

#![cfg(parallel_sm)]

use crate::decompress::inflate::consume_first_decode::Bits;
use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
use crate::decompress::parallel::lut_huffman::{LutDistCode, LutLitLenCode, LIT_LEN};

const MAX_WINDOW_SIZE: usize = 32 * 1024;
const MAX_MATCH_LENGTH: usize = 258;
const END_OF_BLOCK_SYMBOL: u32 = 256;
/// Vendor `DISTANCE_OFFSET` (deflate.hpp:1642): length codes are stored
/// post-expansion as `254 + actual_length` in the ISA-L LUT.
const LENGTH_BASE_OFFSET: u32 = 254;
const MAX_LIT_LEN_SYM: u32 = 512;

const DIST_EXTRA_BIT_COUNT: [u8; 32] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 0, 0,
];

const DIST_START: [u32; 32] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BulkDecodeError {
    InvalidHuffmanCode,
    InvalidLookback,
    OutputOverflow,
    BlockTypeReserved,
    InvalidCodeLengths,
}

#[derive(Debug, Copy, Clone)]
pub struct BulkBlockResult {
    pub bytes_written: usize,
    pub is_final_block: bool,
}

/// Zero-alloc scratch reused across all `decode_block` calls in a
/// stream. Owns the two decoder structs PLUS the code-length scratch
/// arrays that would otherwise be `vec![0u8; N]` per block.
pub struct DecoderScratch {
    pub litlen: LutLitLenCode,
    pub dist: LutDistCode,
    dist_lens: [u8; 32],
    cl_lens: [u8; 19],
    all_lens: [u8; LIT_LEN + 30],
}

impl DecoderScratch {
    pub fn new() -> Self {
        Self {
            litlen: LutLitLenCode::new_empty(),
            dist: LutDistCode::new_empty(),
            dist_lens: [0u8; 32],
            cl_lens: [0u8; 19],
            all_lens: [0u8; LIT_LEN + 30],
        }
    }
}

/// Distance symbol + extra bits → lookback distance (RFC 1951 §3.2.5).
#[inline]
fn distance_from_sym_extra(dist_sym: u32, extra_val: u32) -> Result<usize, BulkDecodeError> {
    if dist_sym >= 30 {
        return Err(BulkDecodeError::InvalidHuffmanCode);
    }
    let distance = (DIST_START[dist_sym as usize] + extra_val) as usize;
    if distance == 0 || distance > MAX_WINDOW_SIZE {
        return Err(BulkDecodeError::InvalidLookback);
    }
    Ok(distance)
}

#[inline]
fn read_distance_extra(bits: &mut Bits<'_>, dist_sym: u32) -> Result<u32, BulkDecodeError> {
    let extra_bits = DIST_EXTRA_BIT_COUNT[dist_sym as usize];
    if extra_bits == 0 {
        return Ok(0);
    }
    let extra = extra_bits as u32;
    if bits.available() < extra {
        bits.refill();
        if bits.available() < extra {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }
    }
    let mask = (1u64 << extra) - 1;
    let v = (bits.peek() & mask) as u32;
    bits.consume(extra);
    Ok(v)
}

/// Decode a distance codeword + RFC extra bits via the ISA-L distance LUT.
#[inline]
fn decode_distance(
    bits: &mut Bits<'_>,
    scratch: &DecoderScratch,
) -> Result<usize, BulkDecodeError> {
    // No leading refill: `dist.decode` refills internally when < 32 bits (it
    // peeks 32), and after consuming the ≤15-bit distance code the ≥17 remaining
    // bits cover the ≤13 extra bits. The old unconditional `bits.refill()` here
    // refilled on EVERY backref even when the buffer was already full — over-eager
    // vs ISA-L. Byte-exact (dist.decode + the extra read are self-sufficient).
    let (dist_sym, dbit) = scratch
        .dist
        .decode(bits)
        .ok_or(BulkDecodeError::InvalidHuffmanCode)?;
    bits.consume(dbit);
    let extra_val = read_distance_extra(bits, dist_sym)?;
    distance_from_sym_extra(dist_sym, extra_val)
}

impl Default for DecoderScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode one DEFLATE block.
///
/// `bits` is positioned at the start of the block header (BFINAL bit).
/// After return, `bits` is positioned immediately after the block body.
///
/// `output[*out_pos..]` is where this block's bytes go. `*out_pos`
/// advances by the number of bytes written.
///
/// Back-references reach into all of `output[..*out_pos]` (i.e., every
/// byte the caller has decoded into this buffer so far). Distances
/// greater than `*out_pos` reach into `predecessor_window`.
pub fn decode_block(
    bits: &mut Bits<'_>,
    output: &mut [u8],
    out_pos: &mut usize,
    predecessor_window: &[u8],
    scratch: &mut DecoderScratch,
) -> Result<BulkBlockResult, BulkDecodeError> {
    bits.refill();
    let header_bits = bits.bitbuf & 0b111;
    bits.consume(3);
    let is_final_block = (header_bits & 0b1) != 0;
    let btype = (header_bits >> 1) & 0b11;

    let start_pos = *out_pos;

    match btype {
        0b00 => {
            return decode_stored_block(bits, output, out_pos, is_final_block);
        }
        0b01 => build_fixed_huffman(scratch)?,
        0b10 => build_dynamic_huffman(bits, scratch)?,
        _ => return Err(BulkDecodeError::BlockTypeReserved),
    }

    // ── Block body: decode symbols via the ISA-L LUT ────────────────────
    // No loop-level refill: `decode()` refills internally when < 32 bits (it
    // peeks 32), and `decode_distance` refills unconditionally before reading
    // the distance + extra bits. The old `if available < 48 { refill }` guard
    // was therefore redundant AND over-eager — on literal-heavy data it refilled
    // after ~1 symbol where ISA-L decodes ~3 per refill (igzip "refill only when
    // the decode can't be guaranteed"). Removing it matches ISA-L's cadence.

    // ── FASTLOOP (libdeflate/ISA-L bounds-check elision) ────────────────────
    // While there are >= FASTLOOP_OUTPUT_MARGIN bytes of output headroom, every
    // literal/match write this iteration (<= 3 packed literals + one match <=258
    // = 261 bytes) is provably in-bounds, so we skip the per-symbol output bounds
    // checks (`*out_pos >= output.len()`, `*out_pos + length > output.len()`).
    // Fulcrum T16: gzippy's `stream_inflate` is 2.35x ISA-L busy; ISA-L's hot
    // loop is bounds-check-free with an output margin. Validity checks (invalid
    // Huffman, EOB, length/distance bounds) are KEPT — only the *output capacity*
    // checks are elided. The existing safe loop below handles the tail (last
    // <MARGIN bytes). Byte-exact: the writes are identical, only the redundant
    // capacity checks are removed inside the proven-safe window.
    const FASTLOOP_OUTPUT_MARGIN: usize = 384; // 3 lits + 258 match + slack; > len+48 for copy_match_fast
    let out_ptr = output.as_mut_ptr();
    while *out_pos + FASTLOOP_OUTPUT_MARGIN <= output.len() {
        let decoded = scratch.litlen.decode(bits);
        if decoded.bit_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }
        bits.consume(decoded.bit_count);

        let mut symbol = decoded.symbol;
        let mut sym_count = decoded.sym_count;
        if sym_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }

        // Packed literal prefix (<= 3, symbol is u32) — margin guarantees room.
        let lit_prefix = (sym_count - 1) as usize;
        if lit_prefix > 0 {
            let mut p = *out_pos;
            let mut s = symbol;
            for _ in 0..lit_prefix {
                // SAFETY: p < *out_pos + 3 < *out_pos + MARGIN <= output.len().
                unsafe {
                    *out_ptr.add(p) = (s & 0xFF) as u8;
                }
                p += 1;
                s >>= 8;
            }
            *out_pos = p;
            symbol = s;
            sym_count = 1;
        }

        while sym_count > 0 {
            let code = (symbol & 0xFFFF) as u16;
            if code as u32 <= 255 {
                // SAFETY: *out_pos < *out_pos_at_iter_top + MARGIN <= output.len().
                unsafe {
                    *out_ptr.add(*out_pos) = (code & 0xFF) as u8;
                }
                *out_pos += 1;
                symbol >>= 8;
                sym_count -= 1;
                continue;
            }
            if code as u32 == END_OF_BLOCK_SYMBOL {
                return Ok(BulkBlockResult {
                    bytes_written: *out_pos - start_pos,
                    is_final_block,
                });
            }
            if code as u32 > MAX_LIT_LEN_SYM {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }
            let length = (symbol - LENGTH_BASE_OFFSET) as usize;
            if length == 0 || length > MAX_MATCH_LENGTH {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }
            let distance = decode_distance(bits, scratch)?;
            if distance > *out_pos + predecessor_window.len() {
                return Err(BulkDecodeError::InvalidLookback);
            }
            // Margin guarantees *out_pos + length (<=258) <= output.len();
            // copy_match's own fast/slow split stays correct.
            copy_match(output, out_pos, distance, length, predecessor_window);
            symbol >>= 8;
            sym_count -= 1;
        }
    }

    // ── SAFE TAIL: bounds-checked loop for the last < MARGIN output bytes ────
    loop {
        let decoded = scratch.litlen.decode(bits);
        if decoded.bit_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }
        bits.consume(decoded.bit_count);

        let mut symbol = decoded.symbol;
        let mut sym_count = decoded.sym_count;
        if sym_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }

        // ISA-L multi-symbol packing guarantees the first `sym_count - 1` packed
        // symbols are LITERALS (only the LAST may be a length/EOB). Vendor writes
        // them as one packed store; mirror that — batch the literal prefix with a
        // SINGLE bounds check instead of the per-byte bounds-checked store the
        // generic loop was doing (igzip_inflate.c decode_next: the packed bytes go
        // out in one move, then `state->write_overflow_lits` advances by count).
        let lit_prefix = (sym_count - 1) as usize;
        if lit_prefix > 0 {
            if *out_pos + lit_prefix > output.len() {
                return Err(BulkDecodeError::OutputOverflow);
            }
            let mut p = *out_pos;
            let mut s = symbol;
            for _ in 0..lit_prefix {
                output[p] = (s & 0xFF) as u8;
                p += 1;
                s >>= 8;
            }
            *out_pos = p;
            symbol = s;
            sym_count = 1;
        }

        while sym_count > 0 {
            let code = (symbol & 0xFFFF) as u16;
            if code as u32 <= 255 {
                if *out_pos >= output.len() {
                    return Err(BulkDecodeError::OutputOverflow);
                }
                output[*out_pos] = (code & 0xFF) as u8;
                *out_pos += 1;
                symbol >>= 8;
                sym_count -= 1;
                continue;
            }

            if code as u32 == END_OF_BLOCK_SYMBOL {
                return Ok(BulkBlockResult {
                    bytes_written: *out_pos - start_pos,
                    is_final_block,
                });
            }
            if code as u32 > MAX_LIT_LEN_SYM {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }

            let length = (symbol - LENGTH_BASE_OFFSET) as usize;
            if length == 0 || length > MAX_MATCH_LENGTH {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }

            // Hide match-source load latency (resumable.rs / libdeflate mirror).
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let prefetch_at = out_pos.saturating_sub(32);
                if prefetch_at < output.len() {
                    // SAFETY: `prefetch_at` is in-bounds; hint only.
                    unsafe {
                        _mm_prefetch(output.as_ptr().add(prefetch_at) as *const i8, _MM_HINT_T0);
                    }
                }
            }

            let distance = decode_distance(bits, scratch)?;
            if *out_pos + length > output.len() {
                return Err(BulkDecodeError::OutputOverflow);
            }
            if distance > *out_pos + predecessor_window.len() {
                return Err(BulkDecodeError::InvalidLookback);
            }

            copy_match(output, out_pos, distance, length, predecessor_window);

            symbol >>= 8;
            sym_count -= 1;
        }
    }
}

fn decode_stored_block(
    bits: &mut Bits<'_>,
    output: &mut [u8],
    out_pos: &mut usize,
    is_final_block: bool,
) -> Result<BulkBlockResult, BulkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::decode_stored_pub;
    let start = *out_pos;
    let new_pos =
        decode_stored_pub(bits, output, *out_pos).map_err(|_| BulkDecodeError::OutputOverflow)?;
    *out_pos = new_pos;
    Ok(BulkBlockResult {
        bytes_written: new_pos - start,
        is_final_block,
    })
}

/// Copy `length` bytes from `*out_pos - distance` to `*out_pos`.
///
/// If the back-reference reaches before the start of `output` (i.e.,
/// `distance > *out_pos`), the prefix comes from `predecessor_window`.
/// Otherwise, source is entirely within `output[..*out_pos]`
/// (RLE-style when `distance < length`).
///
/// Advances `*out_pos` by `length`.
#[inline]
fn copy_match(
    output: &mut [u8],
    out_pos: &mut usize,
    distance: usize,
    length: usize,
    predecessor_window: &[u8],
) {
    let dst = *out_pos;
    if distance <= dst {
        let margin = output.len().saturating_sub(dst);
        if distance >= 8 && length >= 8 && margin >= length + 48 {
            let new_pos = crate::decompress::inflate::consume_first_decode::copy_match_fast(
                output,
                dst,
                distance as u32,
                length as u32,
            );
            *out_pos = new_pos;
            return;
        }
        // Source entirely within output[..dst]. Byte-by-byte handles
        // the RLE case (distance < length).
        let src = dst - distance;
        for i in 0..length {
            output[dst + i] = output[src + i];
        }
    } else {
        // Source extends into the predecessor window.
        let from_window = distance - dst;
        let window_src = predecessor_window.len() - from_window;
        let mut written = 0;
        let window_bytes = from_window.min(length);
        for i in 0..window_bytes {
            output[dst + i] = predecessor_window[window_src + i];
        }
        written += window_bytes;
        // Remaining bytes come from output (RLE-style; could include
        // bytes we just wrote into output above).
        while written < length {
            output[dst + written] = output[dst + written - distance];
            written += 1;
        }
    }
    *out_pos = dst + length;
}

fn build_fixed_huffman(scratch: &mut DecoderScratch) -> Result<(), BulkDecodeError> {
    // RFC 1951 §3.2.6 fixed-Huffman code lengths. Note 280..288 (NOT
    // 280..286): symbols 286 and 287 are RESERVED ("should never
    // actually appear in compressed data, but participate in the code
    // construction"). Their length-8 contributions are required so that
    // count[8] == 152 and canonical codes for length-9 (symbols
    // 144..255) start at the correct value next_code[9] == 0b110010000.
    // Without these two entries count[8] == 150 → next_code[9] shifts
    // down by 4 → every 9-bit literal decodes off by +4 in the byte.
    let mut fixed_lens = [0u8; 288];
    for sym in 0..144 {
        fixed_lens[sym] = 8;
    }
    for sym in 144..256 {
        fixed_lens[sym] = 9;
    }
    for sym in 256..280 {
        fixed_lens[sym] = 7;
    }
    for sym in 280..288 {
        fixed_lens[sym] = 8;
    }
    if !scratch.litlen.rebuild_from(&fixed_lens) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    // Fixed-Huffman dist: 32 codes at length 5 (ISA-L LUT, same as dynamic).
    for sym in 0..32 {
        scratch.dist_lens[sym] = 5;
    }
    if !scratch.dist.rebuild_from(&scratch.dist_lens) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    Ok(())
}

/// DEFLATE Huffman validity test (zlib `inflate_table` `left` accounting).
/// `left` starts at 1 and, per length, doubles then subtracts the codes used.
/// `left < 0` at any point ⇒ over-subscribed; `left > 0` at the end ⇒
/// incomplete. The ISA-L LUT builder assumes a well-formed canonical code
/// (vendor igzip only runs it on validated headers); a malformed length set —
/// which a bad 4 MiB speculative seed produces from random header bits —
/// corrupts the expanded code/bucket tables and indexes the short LUT out of
/// bounds. `require_complete` is set for the litlen code (RFC 1951 requires it
/// complete) and cleared for the distance code (a single distance code is a
/// legal incomplete code). Returns true ⇒ reject, re-sync via
/// `resumable_resync`.
#[inline]
fn huffman_lengths_invalid(lens: &[u8], require_complete: bool) -> bool {
    let mut count = [0u32; 16];
    for &l in lens {
        if l > 15 {
            return true;
        }
        if l != 0 {
            count[l as usize] += 1;
        }
    }
    let mut left: i64 = 1;
    for len in 1..=15 {
        left <<= 1;
        left -= count[len] as i64;
        if left < 0 {
            return true; // over-subscribed
        }
    }
    // `left > 0` is an incomplete code: legal in DEFLATE (a single distance
    // code), so callers requiring completeness (litlen) opt in. The ISA-L
    // builder additionally self-guards against the rare seed that passes this
    // screen yet drives a LUT write out of range (see make_inflate_*).
    require_complete && left != 0
}

fn build_dynamic_huffman(
    bits: &mut Bits<'_>,
    scratch: &mut DecoderScratch,
) -> Result<(), BulkDecodeError> {
    bits.refill();
    let hlit = ((bits.bitbuf & 0x1F) + 257) as usize;
    bits.consume(5);
    let hdist = ((bits.bitbuf & 0x1F) + 1) as usize;
    bits.consume(5);
    let hclen = ((bits.bitbuf & 0x0F) + 4) as usize;
    bits.consume(4);

    if hlit > 286 || hdist > 30 || hclen > 19 {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }

    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    for v in &mut scratch.cl_lens {
        *v = 0;
    }
    for i in 0..hclen {
        bits.refill();
        scratch.cl_lens[CL_ORDER[i]] = (bits.bitbuf & 0x07) as u8;
        bits.consume(3);
    }

    let mut cl_decoder = HuffmanCodingReversedBitsCached::<19>::new();
    let err = cl_decoder.initialize_from_lengths(&scratch.cl_lens, false);
    if err != crate::decompress::parallel::error::Error::None {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }

    let total = hlit + hdist;
    let all_lens = &mut scratch.all_lens[..total];
    for v in all_lens.iter_mut() {
        *v = 0;
    }
    let mut i = 0;
    while i < total {
        let sym = cl_decoder
            .decode(bits)
            .ok_or(BulkDecodeError::InvalidCodeLengths)? as u32;
        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                bits.refill();
                let extra = (bits.bitbuf & 0x3) as usize;
                bits.consume(2);
                let repeat = 3 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                let prev = all_lens[i - 1];
                for _ in 0..repeat {
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill();
                let extra = (bits.bitbuf & 0x7) as usize;
                bits.consume(3);
                let repeat = 3 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                i += repeat;
            }
            18 => {
                bits.refill();
                let extra = (bits.bitbuf & 0x7F) as usize;
                bits.consume(7);
                let repeat = 11 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                i += repeat;
            }
            _ => return Err(BulkDecodeError::InvalidCodeLengths),
        }
    }

    // Reject an over-subscribed code BEFORE it reaches the ISA-L LUT builder.
    // The builder (like vendor igzip, which only runs on validated headers)
    // assumes a well-formed canonical code; an over-subscribed length set —
    // which a bad 4 MiB speculative seed produces from random header bits —
    // overflows `next_code[L]` past 2^L, corrupting the expanded code/bucket
    // tables and indexing the short LUT out of bounds. Standard Kraft check:
    // a valid prefix code never makes `left` go negative. (Incomplete codes,
    // `left > 0` — e.g. a single-symbol distance code — are LEGAL and left to
    // the builder; only over-subscription causes the overflow.) On rejection
    // the chunk re-syncs via `resumable_resync`.
    if huffman_lengths_invalid(&scratch.all_lens[..hlit], true)
        || huffman_lengths_invalid(&scratch.all_lens[hlit..hlit + hdist], false)
    {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }

    let litlen_lens_slice = &scratch.all_lens[..hlit];
    let mut litlen_padded = [0u8; LIT_LEN];
    litlen_padded[..hlit].copy_from_slice(litlen_lens_slice);
    if !scratch.litlen.rebuild_from(&litlen_padded) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    let dist_lens_slice = &scratch.all_lens[hlit..hlit + hdist];
    for v in &mut scratch.dist_lens {
        *v = 0;
    }
    scratch.dist_lens[..hdist].copy_from_slice(dist_lens_slice);
    if !scratch.dist.rebuild_from(&scratch.dist_lens) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(payload: &[u8], level: u32) {
        use std::io::Write;
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        let deflate = enc.finish().unwrap();

        let mut output = vec![0u8; payload.len() + 1024];
        let mut bits = Bits::new(&deflate);
        bits.refill();
        let mut scratch = DecoderScratch::new();
        let predecessor: &[u8] = &[];

        let mut out_pos = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                &mut out_pos,
                predecessor,
                &mut scratch,
            )
            .expect("block decode");
            if result.is_final_block {
                break;
            }
        }
        assert_eq!(&output[..out_pos], payload);
    }

    #[test]
    fn decode_block_round_trips_fixed_huffman() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(5);
        roundtrip(&payload, 1);
    }

    #[test]
    fn decode_block_round_trips_dynamic_huffman() {
        let payload = b"abc123def456ghi789jkl012mno345pqr678stu901vwx234yz".repeat(200);
        roundtrip(&payload, 6);
    }

    #[test]
    fn decode_block_round_trips_stored() {
        let payload = b"stored block test payload".to_vec();
        roundtrip(&payload, 0);
    }

    /// Random data at zlib L1 with a 32 KiB zero predecessor window —
    /// the exact shape of the chunk-0 production call for the
    /// `corpus_large_random` parallel-SM test. If this fails, the
    /// production failure is reproducible in isolation.
    #[test]
    fn decode_block_random_data_l1_with_zero_predecessor() {
        use std::io::Write;
        let mut rng: u64 = 0xcafef00d_deadbeef;
        let mut payload = Vec::with_capacity(4096);
        for _ in 0..4096 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            payload.push((rng >> 24) as u8);
        }
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();
        let (_hdr, hdr_size) =
            crate::decompress::parallel::gzip_format::read_header(&gz).expect("gz header");
        let deflate = &gz[hdr_size..gz.len() - 8];

        let mut output = vec![0u8; payload.len() + 4096];
        let mut bits = Bits::new(deflate);
        bits.refill();
        let mut scratch = DecoderScratch::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut out_pos = 0;
        let result = decode_block(
            &mut bits,
            &mut output,
            &mut out_pos,
            &predecessor[..],
            &mut scratch,
        );
        if let Err(e) = result {
            eprintln!("[repro] decode failed at out_pos={out_pos}: {e:?}");
            eprintln!(
                "[repro] first 32 decoded bytes: {:02x?}",
                &output[..out_pos.min(32)]
            );
            eprintln!("[repro] first 32 expected bytes:  {:02x?}", &payload[..32]);
            // Compare byte-by-byte to find divergence point.
            for i in 0..out_pos.min(payload.len()) {
                if output[i] != payload[i] {
                    eprintln!(
                        "[repro] divergence at byte {i}: got {:02x}, want {:02x}",
                        output[i], payload[i]
                    );
                    break;
                }
            }
            panic!("decode failed");
        }
        eprintln!(
            "[repro] first block decoded {} bytes (final={})",
            out_pos,
            result.unwrap().is_final_block
        );
        assert_eq!(&output[..out_pos], &payload[..out_pos]);
    }

    /// btype01-heavy fixture (mimics tests::routing::make_btype01_heavy_data
    /// compressed at L1). Exercises the full multi-block decode against
    /// flate2 ground truth. Smaller payload (256 KiB instead of 24 MiB)
    /// so it runs fast in the unit-test suite.
    #[test]
    fn decode_block_btype01_heavy_l1_full_stream() {
        use std::io::Write;
        let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
        // 12 MiB — large enough to exercise many block transitions and
        // catch any state-leak between successive decode_block calls.
        let target = 12 * 1024 * 1024;
        let mut payload = Vec::with_capacity(target);
        let mut rng: u64 = 0xb0bd1ec0de;
        while payload.len() < target {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < 70 {
                payload.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[(rng as usize) % phrases.len()];
                let take = phrase.len().min(target - payload.len());
                payload.extend_from_slice(&phrase[..take]);
            }
        }
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();
        let (_hdr, hdr_size) =
            crate::decompress::parallel::gzip_format::read_header(&gz).expect("gz header");
        let deflate = &gz[hdr_size..gz.len() - 8];

        let mut output = vec![0u8; payload.len() + 4096];
        let mut bits = Bits::new(deflate);
        bits.refill();
        let mut scratch = DecoderScratch::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut out_pos = 0;
        let mut block_count = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                &mut out_pos,
                &predecessor[..],
                &mut scratch,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "btype01 L1 block decode failed at block {block_count}, out_pos={out_pos}: {e:?}"
                )
            });
            block_count += 1;
            if result.is_final_block {
                break;
            }
        }
        assert_eq!(out_pos, payload.len(), "output length mismatch");
        for i in (0..out_pos).step_by(4096) {
            let end = (i + 4096).min(out_pos);
            assert_eq!(
                &output[i..end],
                &payload[i..end],
                "divergence at [{i}..{end}]"
            );
        }
    }

    /// Multi-block cross-reference test: a payload large enough to force
    /// multiple Huffman blocks, with strong intra-stream redundancy so
    /// back-references reach across block boundaries.
    ///
    /// The PRIOR API treated each block's prior-blocks-in-this-output
    /// as "predecessor" (wrong) — this test catches the multi-block
    /// cross-reference bug that the silesia gate found at chunk
    /// [155648..159744].
    #[test]
    fn decode_block_multi_block_cross_references() {
        let mut payload = Vec::new();
        for i in 0..50_000 {
            payload.extend_from_slice(format!("entry-{i:08}-with-some-shared-suffix\n").as_bytes());
        }
        // Re-append a prefix-shifted region to force long-distance refs.
        let shift = &payload[..200_000].to_vec();
        payload.extend_from_slice(shift);
        roundtrip(&payload, 6);
    }

    /// REAL-CORPUS correctness gate. Per memory rule
    /// `feedback-real-corpus-test-with-lever`: any new inner-decoder
    /// lever must include a silesia (or equivalent) differential test
    /// IN THE SAME COMMIT as the lever.
    #[test]
    fn decode_block_byte_perfect_silesia_if_available() {
        let candidates = [
            "benchmark_data/silesia-large.gz",
            "benchmark_data/silesia.tar.gz",
            "benchmark_data/silesia-gzip.tar.gz",
        ];
        let data = match candidates.iter().find_map(|p| std::fs::read(p).ok()) {
            Some(d) => d,
            None => {
                eprintln!("[silesia bulk test] no silesia corpus available, skipping");
                return;
            }
        };

        let (_hdr, hdr_size) =
            crate::decompress::parallel::gzip_format::read_header(&data).expect("gzip header");
        let deflate_start = hdr_size;
        let deflate_end = data.len() - 8;
        let deflate = &data[deflate_start..deflate_end];

        use std::io::Read;
        let mut gz = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        gz.read_to_end(&mut expected).expect("flate2 decode");

        let mut output = vec![0u8; expected.len() + 4096];
        let mut bits = Bits::new(deflate);
        bits.refill();
        let mut scratch = DecoderScratch::new();
        let predecessor: &[u8] = &[];

        let mut out_pos = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                &mut out_pos,
                predecessor,
                &mut scratch,
            )
            .unwrap_or_else(|e| panic!("silesia block decode failed at out_pos={out_pos}: {e:?}"));
            if result.is_final_block {
                break;
            }
        }
        output.truncate(out_pos);
        assert_eq!(
            output.len(),
            expected.len(),
            "decoded length mismatch: got {}, expected {}",
            output.len(),
            expected.len()
        );
        const CHUNK: usize = 4096;
        for i in (0..output.len()).step_by(CHUNK) {
            let end = (i + CHUNK).min(output.len());
            assert_eq!(
                &output[i..end],
                &expected[i..end],
                "byte mismatch in chunk [{}..{}]",
                i,
                end
            );
        }
        eprintln!("[silesia bulk test] {} bytes byte-perfect", out_pos);
    }
}
