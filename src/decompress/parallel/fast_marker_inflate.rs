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

// ── Canonical Huffman table ─────────────────────────────────────────────────

/// Maximum bits we ever look up from the bit buffer at once.
const MAX_TABLE_BITS: u32 = 15;

/// Single-level lookup table. For 15-bit codes, table size is 32768 entries —
/// that's 128 KB at 4 bytes each. Built once per block, costs ~25 µs per block.
/// Far simpler than libdeflate's multi-level scheme; can be tightened later.
///
/// Entry layout (little-endian):
/// - bits 0..=15 : decoded symbol
/// - bits 16..=23: code length in bits (1..=15). 0 = invalid.
#[derive(Clone)]
struct HuffTable {
    entries: Vec<u32>,
    table_bits: u32,
}

impl HuffTable {
    /// Build a canonical Huffman table from code lengths.
    ///
    /// `lengths[i]` is the bit length assigned to symbol `i`; 0 means absent.
    fn build(lengths: &[u8]) -> Result<Self> {
        // Find max code length used.
        let max_len = lengths.iter().copied().max().unwrap_or(0) as u32;
        if max_len == 0 {
            // Degenerate: no symbols. Return a dummy table that errors on
            // every decode attempt. Size must match 1 << table_bits so the
            // peek index never goes out of bounds.
            return Ok(Self {
                entries: vec![0u32; 2],
                table_bits: 1,
            });
        }
        if max_len > MAX_TABLE_BITS {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Huffman code length {max_len} exceeds {MAX_TABLE_BITS}"),
            ));
        }
        let table_bits = max_len;
        let table_size = 1usize << table_bits;
        let mut entries = vec![0u32; table_size];

        // RFC 1951 §3.2.2 canonical Huffman: count codes of each length, then
        // assign codes in symbol order.
        let mut bl_count = [0u32; 16];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for len in 1..=max_len {
            code = (code + bl_count[(len - 1) as usize]) << 1;
            next_code[len as usize] = code;
        }

        // For each symbol, compute its (bit-reversed) code and fill every
        // table entry whose low `len` bits equal that code. (The bit buffer
        // delivers bits LSB-first, so codes go in reversed.)
        for (sym, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as u32;
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            let reversed = reverse_bits(code, len);
            let entry = (sym as u32) | (len << 16);
            // Stride = 1 << len. Fill `2^(table_bits - len)` entries.
            let stride = 1usize << len;
            let mut idx = reversed as usize;
            while idx < table_size {
                entries[idx] = entry;
                idx += stride;
            }
        }

        Ok(Self {
            entries,
            table_bits,
        })
    }

    /// Look up a code, consuming its bits from `bits`. Returns the symbol.
    #[inline(always)]
    fn decode(&self, bits: &mut Bits) -> Result<u32> {
        if bits.available() < self.table_bits {
            bits.refill();
        }
        let peek = (bits.peek() & ((1u64 << self.table_bits) - 1)) as usize;
        let entry = self.entries[peek];
        let code_len = (entry >> 16) & 0xFF;
        if code_len == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid Huffman code"));
        }
        bits.consume(code_len);
        Ok(entry & 0xFFFF)
    }
}

#[inline(always)]
fn reverse_bits(mut v: u32, n: u32) -> u32 {
    // Tiny enough that we don't bother with a precomputed table.
    let mut r = 0u32;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
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
    // Set up the bit buffer at the requested offset.
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

    let mut output: Vec<u16> = Vec::with_capacity(data.len() * 4);

    decode_loop(&mut bits, &mut output, byte_offset, bit_in_byte, None)?;

    // Compute end_bit_offset = byte_offset*8 + bit_in_byte + bits_consumed.
    // bits.pos is bytes pulled out of bits.data (which started at byte_offset),
    // bits.available() is leftover bits in bitbuf.
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    // Total bits we've eaten from bits.data: consumed_bytes_from_slice*8 - bits_in_buf
    // (i.e. we pulled this many bytes into the buffer but bits_in_buf are unconsumed).
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    let end_bit_offset = byte_offset * 8 + bit_in_byte as usize + bits_consumed_from_slice;

    Ok((output, end_bit_offset))
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
    base_bit_in_byte: u32,
    mut block_starts: Option<&mut Vec<usize>>,
) -> Result<()> {
    loop {
        if bits.available() < 3 {
            bits.refill();
        }
        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u32;
        // Record the bit position of the start of THIS block (the header
        // bit). This is suitable as a chunk start_bit_offset for a parallel
        // decoder: passing it back into `decode_chunk_markers` will redo the
        // same block (and subsequent ones).
        if let Some(starts) = block_starts.as_mut() {
            let bits_in_buf = bits.available() as usize;
            let consumed = bits.pos.saturating_mul(8).saturating_sub(bits_in_buf);
            let bit_pos = base_byte * 8 + base_bit_in_byte as usize + consumed;
            starts.push(bit_pos);
        }
        bits.consume(3);

        match btype {
            0 => decode_stored(bits, output)?,
            1 => decode_fixed(bits, output)?,
            2 => decode_dynamic(bits, output)?,
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
    decode_loop(&mut bits, &mut output, 0, 0, Some(&mut starts))?;
    Ok(starts)
}

// ── Stored block (BTYPE = 00) ───────────────────────────────────────────────

fn decode_stored(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
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
        output.push((bits.bitbuf & 0xFF) as u16);
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
        output.reserve(remaining);
        for &b in &bits.data[bits.pos..bits.pos + remaining] {
            output.push(b as u16);
        }
        bits.pos += remaining;
    }

    // Reset bit buffer state.
    bits.bitbuf = 0;
    bits.bitsleft = 0;
    Ok(())
}

// ── Fixed Huffman block (BTYPE = 01) ────────────────────────────────────────

fn fixed_litlen_table() -> &'static HuffTable {
    use std::sync::OnceLock;
    static T: OnceLock<HuffTable> = OnceLock::new();
    T.get_or_init(|| {
        // RFC 1951 §3.2.6 fixed Huffman code lengths.
        let mut lens = vec![0u8; 288];
        lens[0..144].fill(8);
        lens[144..256].fill(9);
        lens[256..280].fill(7);
        lens[280..288].fill(8);
        HuffTable::build(&lens).expect("fixed litlen table builds")
    })
}

fn fixed_dist_table() -> &'static HuffTable {
    use std::sync::OnceLock;
    static T: OnceLock<HuffTable> = OnceLock::new();
    T.get_or_init(|| HuffTable::build(&[5u8; 30]).expect("fixed dist table builds"))
}

fn decode_fixed(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
    decode_huffman_block(bits, output, fixed_litlen_table(), fixed_dist_table())
}

// ── Dynamic Huffman block (BTYPE = 10) ──────────────────────────────────────

fn decode_dynamic(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
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
    let cl_table = HuffTable::build(&cl_lens)?;

    // Read HLIT + HDIST code lengths using the code-length code.
    let total = hlit + hdist;
    let mut lens = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = cl_table.decode(bits)?;
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

    let litlen_table = HuffTable::build(&lens[..hlit])?;
    let dist_table = HuffTable::build(&lens[hlit..])?;

    decode_huffman_block(bits, output, &litlen_table, &dist_table)
}

// ── Shared Huffman block body (used by both fixed and dynamic) ──────────────

#[inline]
fn decode_huffman_block(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    litlen: &HuffTable,
    dist: &HuffTable,
) -> Result<()> {
    loop {
        let sym = litlen.decode(bits)?;
        if sym < 256 {
            // Literal.
            output.push(sym as u16);
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

            let dsym = dist.decode(bits)? as usize;
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
            emit_match(output, distance, length);
        }
    }
}

/// Append `length` u16 values for a back-reference at distance `distance`
/// from the current end of `output`. Splits between markers (cross-chunk
/// portion) and chunk-local copies (within-chunk portion).
#[inline]
fn emit_match(output: &mut Vec<u16>, distance: usize, length: usize) {
    let out_pos = output.len();
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
    // for i ∈ marker_count..length. After the markers are pushed, out_pos has
    // advanced by marker_count, so the local copy starts at position
    // `out_pos + marker_count` in the output and reads from
    // `(out_pos + marker_count) - distance`.
    let local_count = length - marker_count;
    if local_count > 0 {
        // Element-by-element copy to correctly handle the RLE case (distance < length)
        // where the destination overlaps the source. Markers carried in the source
        // slice propagate unchanged because we copy u16, not u8.
        let base_dst = out_pos + marker_count;
        for i in 0..local_count {
            let src = base_dst + i - distance;
            let v = output[src];
            output.push(v);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::replace_markers::{replace_markers, u16_to_u8};
    use std::io::Write;

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
        emit_match(&mut output, 4, 3); // distance 4, length 3, out_pos 0
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
        emit_match(&mut output, 5, 8);
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
        emit_match(&mut output, 1, 5);
        assert_eq!(output, vec![b'X' as u16; 6]);
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
}
