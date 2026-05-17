//! Literal port of `rapidgzip::deflate::Block` skeleton + state +
//! `readHeader` + `readDynamicHuffmanCoding`
//! (vendor/.../gzip/deflate.hpp:513-1156).
//!
//! This file ports the Block STATE MACHINE + HEADER PARSER. The
//! inner-loop bodies (`readInternalCompressed`, `readInternalUncompressed`,
//! `read`, `appendToWindow`, `resolveBackreference`) land in follow-up
//! commits — the structural skeleton + header surface is the foundation
//! they hang off of.

#![allow(dead_code)]

use std::io;

use crate::decompress::inflate::consume_first_decode::Bits;

// ── Constants (from rapidgzip definitions.hpp) ──────────────────────────────

pub const MAX_LITERAL_OR_LENGTH_SYMBOLS: usize = 286;
pub const MAX_DISTANCE_SYMBOL_COUNT: usize = 30;
pub const MAX_CODE_LENGTH: u8 = 15;
pub const MAX_PRECODE_LENGTH: u8 = 7;
pub const PRECODE_BITS: u8 = 3;
pub const PRECODE_COUNT_BITS: u8 = 4;
pub const MAX_PRECODE_COUNT: usize = 19;
pub const END_OF_BLOCK_SYMBOL: u16 = 256;
pub const MAX_RUN_LENGTH: usize = 258;
pub const MAX_WINDOW_SIZE: usize = 32768;

/// RFC 1951 precode alphabet order (matches deflate.hpp's PRECODE_ALPHABET).
pub const PRECODE_ALPHABET: [usize; MAX_PRECODE_COUNT] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ── Types ───────────────────────────────────────────────────────────────────

/// Compression type from the 3-bit deflate block header. Mirror of
/// `rapidgzip::deflate::CompressionType` (definitions.hpp).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    Uncompressed = 0,
    FixedHuffman = 1,
    DynamicHuffman = 2,
    Reserved = 3,
}

impl CompressionType {
    pub fn from_btype(btype: u8) -> Self {
        match btype {
            0 => CompressionType::Uncompressed,
            1 => CompressionType::FixedHuffman,
            2 => CompressionType::DynamicHuffman,
            _ => CompressionType::Reserved,
        }
    }
}

/// A back-reference discovered during decoding. Mirror of
/// `Block::Backreference` (deflate.hpp:520-523).
#[derive(Debug, Clone, Copy, Default)]
pub struct Backreference {
    pub distance: u16,
    pub length: u16,
}

/// Errors returned by `Block`'s methods. Subset of rapidgzip's
/// `Error` enum covering only the deflate-Block error paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    EndOfFile,
    UnexpectedLastBlock,
    NonZeroPadding,
    LengthChecksumMismatch,
    ExceededLiteralRange,
    ExceededDistanceRange,
    InvalidCompression,
    InvalidCodeLengths,
    InvalidHuffmanCode,
    ExceededWindowRange,
}

impl From<BlockError> for io::Error {
    fn from(e: BlockError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}"))
    }
}

// ── Block ───────────────────────────────────────────────────────────────────

/// Literal port of `rapidgzip::deflate::Block<ENABLE_STATISTICS>`
/// (deflate.hpp:513-961) — the deflate block state machine.
///
/// This struct holds only the STATE + HEADER-derived metadata. The
/// concrete inner-loop methods (`read`, `read_internal_compressed`,
/// `read_internal_uncompressed`, `append_to_window`, `resolve_back_ref`)
/// land in subsequent commits to keep individual changes reviewable.
#[derive(Debug)]
pub struct Block {
    at_end_of_block: bool,
    at_end_of_file: bool,
    is_last_block: bool,
    compression_type: CompressionType,
    /// Stored-block padding bits (0..7) — zero in well-formed input.
    padding: u8,
    /// Stored-block declared payload length (only set when
    /// `compression_type == Uncompressed`).
    uncompressed_size: usize,
    /// Total decoded bytes across all blocks decoded by this Block
    /// instance (rapidgzip's `m_decodedBytes`, deflate.hpp:921).
    decoded_bytes: usize,
    /// `m_decodedBytes` snapshot at the start of the current block.
    decoded_bytes_at_block_start: usize,
    /// Code lengths for the precode alphabet (P), populated by
    /// `read_dynamic_huffman_coding`.
    pub precode_cl: [u8; MAX_PRECODE_COUNT],
    /// Code lengths for the combined literal/length + distance
    /// alphabets, populated by `read_dynamic_huffman_coding`. First
    /// `literal_code_count` entries are literal/length lengths;
    /// remainder are distance lengths.
    pub literal_cl: Vec<u8>,
    /// Number of literal/length codes from this block's header.
    pub literal_code_count: usize,
    /// Number of distance codes from this block's header.
    pub distance_code_count: usize,
    /// Tracked back-references (debug instrumentation).
    pub backreferences: Vec<Backreference>,
    track_backreferences: bool,
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl Block {
    pub fn new() -> Self {
        Self {
            at_end_of_block: false,
            at_end_of_file: false,
            is_last_block: false,
            compression_type: CompressionType::DynamicHuffman,
            padding: 0,
            uncompressed_size: 0,
            decoded_bytes: 0,
            decoded_bytes_at_block_start: 0,
            precode_cl: [0u8; MAX_PRECODE_COUNT],
            literal_cl: vec![0u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + MAX_DISTANCE_SYMBOL_COUNT],
            literal_code_count: 0,
            distance_code_count: 0,
            backreferences: Vec::new(),
            track_backreferences: false,
        }
    }

    // ── Accessors (deflate.hpp:526-561) ─────────────────────────────────────

    pub fn eob(&self) -> bool {
        self.at_end_of_block
    }
    pub fn eos(&self) -> bool {
        self.at_end_of_block && self.is_last_block
    }
    pub fn eof(&self) -> bool {
        self.at_end_of_file
    }
    pub fn is_last_block(&self) -> bool {
        self.is_last_block
    }
    pub fn compression_type(&self) -> CompressionType {
        self.compression_type
    }
    pub fn padding(&self) -> u8 {
        self.padding
    }
    pub fn uncompressed_size(&self) -> usize {
        if self.compression_type == CompressionType::Uncompressed {
            self.uncompressed_size
        } else {
            0
        }
    }
    pub fn set_track_backreferences(&mut self, enable: bool) {
        self.track_backreferences = enable;
    }
    pub fn track_backreferences(&self) -> bool {
        self.track_backreferences
    }

    /// Reset to a fresh state (rapidgzip's `Block::reset`, deflate.hpp:670+).
    /// Optionally re-arms with an initial window (placeholder until
    /// `set_initial_window` lands in the next slice).
    pub fn reset(&mut self) {
        self.at_end_of_block = false;
        self.at_end_of_file = false;
        self.is_last_block = false;
        self.compression_type = CompressionType::DynamicHuffman;
        self.padding = 0;
        self.uncompressed_size = 0;
        self.decoded_bytes = 0;
        self.decoded_bytes_at_block_start = 0;
        self.precode_cl = [0u8; MAX_PRECODE_COUNT];
        for v in self.literal_cl.iter_mut() {
            *v = 0;
        }
        self.literal_code_count = 0;
        self.distance_code_count = 0;
        self.backreferences.clear();
    }

    // ── Header parser (deflate.hpp:964-1156) ────────────────────────────────

    /// Literal port of `Block::readHeader<treatLastBlockAsError>`
    /// (deflate.hpp:964-1022).
    ///
    /// Reads the 3-bit BFINAL + BTYPE header, then dispatches into
    /// the per-type follow-up parsing (stored-block padding + LEN/NLEN,
    /// or dynamic-Huffman coding).
    ///
    /// On `treat_last_block_as_error == true`, returns
    /// `UnexpectedLastBlock` if BFINAL=1 — used by block-finder
    /// candidate validation (filters out final blocks).
    pub fn read_header(
        &mut self,
        bits: &mut Bits,
        treat_last_block_as_error: bool,
    ) -> Result<(), BlockError> {
        ensure_bits(bits, 3)?;
        let bfinal = (bits.peek() & 1) != 0;
        bits.consume(1);
        self.is_last_block = bfinal;
        if treat_last_block_as_error && bfinal {
            return Err(BlockError::UnexpectedLastBlock);
        }
        let btype = (bits.peek() & 0b11) as u8;
        bits.consume(2);
        self.compression_type = CompressionType::from_btype(btype);

        match self.compression_type {
            CompressionType::Uncompressed => {
                self.read_uncompressed_header(bits)?;
            }
            CompressionType::FixedHuffman => {
                // No further header parsing needed; the fixed coding
                // tables are static (RFC 1951 §3.2.6).
            }
            CompressionType::DynamicHuffman => {
                self.read_dynamic_huffman_coding(bits)?;
            }
            CompressionType::Reserved => {
                return Err(BlockError::InvalidCompression);
            }
        }

        self.at_end_of_block = false;
        self.decoded_bytes_at_block_start = self.decoded_bytes;
        self.backreferences.clear();
        Ok(())
    }

    fn read_uncompressed_header(&mut self, bits: &mut Bits) -> Result<(), BlockError> {
        // Pad to byte boundary; rapidgzip enforces padding == 0
        // (deflate.hpp:991-996).
        let bits_to_drain = bits.available() & 7;
        if bits_to_drain > 0 {
            // Need to consume bits up to the next byte boundary. Since
            // `bits` is bit-oriented, drain the low bits of the buffer.
            ensure_bits(bits, bits_to_drain)?;
            let pad = (bits.peek() & ((1u64 << bits_to_drain) - 1)) as u8;
            bits.consume(bits_to_drain);
            self.padding = pad;
            if pad != 0 {
                return Err(BlockError::NonZeroPadding);
            }
        }
        ensure_bits(bits, 32)?;
        let len = (bits.peek() & 0xFFFF) as u16;
        bits.consume(16);
        let nlen = (bits.peek() & 0xFFFF) as u16;
        bits.consume(16);
        if len != !nlen {
            return Err(BlockError::LengthChecksumMismatch);
        }
        self.uncompressed_size = len as usize;
        Ok(())
    }

    /// Literal port of `Block::readDynamicHuffmanCoding`
    /// (deflate.hpp:1025-1156). Reads HLIT/HDIST/HCLEN, decodes the
    /// precode lengths, builds the precode Huffman code, then decodes
    /// the literal/length + distance code lengths from it.
    ///
    /// On success, populates `self.precode_cl`, `self.literal_cl`,
    /// `self.literal_code_count`, and `self.distance_code_count`.
    pub fn read_dynamic_huffman_coding(&mut self, bits: &mut Bits) -> Result<(), BlockError> {
        ensure_bits(bits, 14)?;
        let literal_code_count = 257 + (bits.peek() & 0x1F) as usize;
        bits.consume(5);
        if literal_code_count > MAX_LITERAL_OR_LENGTH_SYMBOLS {
            return Err(BlockError::ExceededLiteralRange);
        }
        let distance_code_count = 1 + (bits.peek() & 0x1F) as usize;
        bits.consume(5);
        if distance_code_count > MAX_DISTANCE_SYMBOL_COUNT {
            return Err(BlockError::ExceededDistanceRange);
        }
        let code_length_count = 4 + (bits.peek() & 0xF) as usize;
        bits.consume(4);

        self.literal_code_count = literal_code_count;
        self.distance_code_count = distance_code_count;

        // Read the precode lengths in PRECODE_ALPHABET order.
        for v in self.precode_cl.iter_mut() {
            *v = 0;
        }
        for &slot in PRECODE_ALPHABET.iter().take(code_length_count) {
            ensure_bits(bits, PRECODE_BITS as u32)?;
            self.precode_cl[slot] = (bits.peek() & 0x7) as u8;
            bits.consume(PRECODE_BITS as u32);
        }

        // Decode HLIT + HDIST code lengths using the precode.
        let total = literal_code_count + distance_code_count;
        let lit_dist_lengths =
            read_literal_and_distance_code_lengths(bits, &self.precode_cl, total)?;
        self.literal_cl[..total].copy_from_slice(&lit_dist_lengths);

        // End-of-block symbol MUST have a non-zero code length.
        if self.literal_cl[END_OF_BLOCK_SYMBOL as usize] == 0 {
            return Err(BlockError::InvalidCodeLengths);
        }
        Ok(())
    }

    /// Public entry point — literal port of `Block::read`
    /// (deflate.hpp:1192-1300). Decodes up to `n_max_to_decode` bytes
    /// of the CURRENT block's payload, appending u16 values into
    /// `output`. Dynamic/Fixed Huffman bodies are decoded in a
    /// follow-up commit (Step 7c); this slice handles the uncompressed
    /// case end-to-end.
    pub fn read(
        &mut self,
        bits: &mut Bits,
        output: &mut Vec<u16>,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        if self.eob() {
            return Ok(0);
        }
        match self.compression_type {
            CompressionType::Reserved => Err(BlockError::InvalidCompression),
            CompressionType::Uncompressed => {
                self.read_internal_uncompressed(bits, output, n_max_to_decode)
            }
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                self.read_internal_compressed(bits, output, n_max_to_decode)
            }
        }
    }

    /// Literal port of `Block::readInternalUncompressed` semantics
    /// (deflate.hpp:1212-1278): consume `uncompressed_size` bytes from
    /// the bit stream (which are byte-aligned per the deflate spec)
    /// and emit them as literal u16 values into `output`. Caps at
    /// `n_max_to_decode`; sets `at_end_of_block` when the full payload
    /// is consumed.
    pub fn read_internal_uncompressed(
        &mut self,
        bits: &mut Bits,
        output: &mut Vec<u16>,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        let to_read = self.uncompressed_size.min(n_max_to_decode);
        for _ in 0..to_read {
            ensure_bits(bits, 8)?;
            let byte = (bits.peek() & 0xFF) as u16;
            bits.consume(8);
            output.push(byte);
        }
        self.uncompressed_size -= to_read;
        self.decoded_bytes += to_read;
        if self.uncompressed_size == 0 {
            self.at_end_of_block = true;
        }
        Ok(to_read)
    }

    /// Literal port of `Block::readInternalCompressed`
    /// (deflate.hpp:1510-1582). Decodes one Huffman-coded block body
    /// (Fixed or Dynamic) using the already-populated `literal_cl` /
    /// `distance_code_count` from `read_header`. Emits literals as u16
    /// values < 256; emits cross-chunk back-refs via the MapMarkers
    /// encoding from `replace_markers::MARKER_BASE`; in-chunk back-refs
    /// resolve immediately by copying from `output`.
    ///
    /// On hitting END_OF_BLOCK (symbol 256), sets `at_end_of_block`.
    pub fn read_internal_compressed(
        &mut self,
        bits: &mut Bits,
        output: &mut Vec<u16>,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        // Build per-block lit/len + distance decode tables. For Dynamic,
        // they come from self.literal_cl + distance_code_count. For
        // Fixed, RFC 1951 §3.2.6 specifies static lengths.
        let (litlen_lens, dist_lens) = match self.compression_type {
            CompressionType::DynamicHuffman => {
                let lit = self.literal_cl[..self.literal_code_count].to_vec();
                let dist = self.literal_cl
                    [self.literal_code_count..self.literal_code_count + self.distance_code_count]
                    .to_vec();
                (lit, dist)
            }
            CompressionType::FixedHuffman => fixed_huffman_code_lengths(),
            _ => return Err(BlockError::InvalidCompression),
        };

        let litlen_table =
            build_canonical_table::<MAX_LITERAL_OR_LENGTH_SYMBOLS>(&litlen_lens, 15)?;
        let dist_table = build_canonical_table::<MAX_DISTANCE_SYMBOL_COUNT>(&dist_lens, 15)?;

        let start_len = output.len();
        let mut emitted: usize = 0;
        while emitted < n_max_to_decode {
            // Decode one lit/len symbol.
            let (sym, bit_count) = decode_canonical(&litlen_table, 15, bits)?;
            bits.consume(bit_count);

            if sym < 256 {
                output.push(sym);
                emitted += 1;
                continue;
            }
            if sym == END_OF_BLOCK_SYMBOL {
                self.at_end_of_block = true;
                self.decoded_bytes += emitted;
                return Ok(emitted);
            }
            if sym > 285 {
                return Err(BlockError::InvalidHuffmanCode);
            }
            // Length code: read extra bits and resolve the length.
            let lidx = (sym - 257) as usize;
            let length = read_length_extra(bits, lidx)?;
            // Distance: decode symbol + extra bits.
            let (dsym, dbit) = decode_canonical(&dist_table, 15, bits)?;
            bits.consume(dbit);
            if dsym as usize >= DISTANCE_BASE.len() {
                return Err(BlockError::InvalidHuffmanCode);
            }
            let distance = read_distance_extra(bits, dsym as usize)?;
            if distance == 0 || distance > MAX_WINDOW_SIZE {
                return Err(BlockError::ExceededWindowRange);
            }
            // Emit the back-ref. If distance > current in-block position,
            // the missing bytes become MapMarkers indices (= rapidgzip
            // resolveBackreference's behavior when window is empty).
            let out_pos = output.len() - start_len + self.decoded_bytes_at_block_start;
            emit_backref(output, distance, length, out_pos)?;
            emitted += length;
            if self.track_backreferences {
                self.backreferences.push(Backreference {
                    distance: distance as u16,
                    length: length as u16,
                });
            }
        }
        self.decoded_bytes += emitted;
        Ok(emitted)
    }
}

// ── Length / distance extra-bits tables (RFC 1951 §3.2.5) ──────────────────

pub const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

pub const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

pub const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

pub const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn read_length_extra(bits: &mut Bits, lidx: usize) -> Result<usize, BlockError> {
    let extra = LENGTH_EXTRA[lidx] as u32;
    if extra > 0 {
        ensure_bits(bits, extra)?;
    }
    let extra_val = if extra > 0 {
        let v = (bits.peek() & ((1u64 << extra) - 1)) as u16;
        bits.consume(extra);
        v
    } else {
        0
    };
    Ok((LENGTH_BASE[lidx] + extra_val) as usize)
}

fn read_distance_extra(bits: &mut Bits, dsym: usize) -> Result<usize, BlockError> {
    let extra = DISTANCE_EXTRA[dsym] as u32;
    if extra > 0 {
        ensure_bits(bits, extra)?;
    }
    let extra_val = if extra > 0 {
        let v = (bits.peek() & ((1u64 << extra) - 1)) as u32;
        bits.consume(extra);
        v
    } else {
        0
    };
    Ok(DISTANCE_BASE[dsym] as usize + extra_val as usize)
}

/// RFC 1951 §3.2.6 fixed Huffman code lengths.
fn fixed_huffman_code_lengths() -> (Vec<u8>, Vec<u8>) {
    let mut lit = vec![0u8; 288];
    for v in &mut lit[0..144] {
        *v = 8;
    }
    for v in &mut lit[144..256] {
        *v = 9;
    }
    for v in &mut lit[256..280] {
        *v = 7;
    }
    for v in &mut lit[280..288] {
        *v = 8;
    }
    let dist = vec![5u8; 30];
    (lit, dist)
}

/// Build a canonical-Huffman decode table. Returns a table of size
/// `1 << max_bits` where each entry packs `(symbol << 5) | length`,
/// with length=0 meaning no valid code at that bit pattern.
///
/// `MAX_SYMBOLS` is a compile-time bound for the symbol alphabet
/// size. `max_bits` is the longest code length expected (15 for
/// deflate lit/len + distance).
fn build_canonical_table<const MAX_SYMBOLS: usize>(
    code_lengths: &[u8],
    max_bits: u8,
) -> Result<Vec<u32>, BlockError> {
    let mut bl_count = [0u32; 16];
    for &len in code_lengths.iter() {
        if len > max_bits {
            return Err(BlockError::InvalidCodeLengths);
        }
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }
    // Kraft check.
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for b in 1..=max_bits as usize {
        code = (code + bl_count[b - 1]) << 1;
        if code > (1u32 << b) {
            return Err(BlockError::InvalidCodeLengths);
        }
        next_code[b] = code;
    }
    let table_size = 1usize << max_bits;
    let mut table = vec![0u32; table_size];
    for (sym, &len) in code_lengths.iter().enumerate() {
        if len == 0 || sym >= MAX_SYMBOLS {
            continue;
        }
        let canonical = next_code[len as usize];
        next_code[len as usize] += 1;
        let reversed = reverse_bits_u32(canonical, len);
        let entry = ((sym as u32) << 5) | (len as u32);
        let step = 1usize << len;
        let mut idx = reversed as usize;
        while idx < table_size {
            table[idx] = entry;
            idx += step;
        }
    }
    Ok(table)
}

fn reverse_bits_u32(mut v: u32, n: u8) -> u32 {
    let mut r = 0u32;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn decode_canonical(
    table: &[u32],
    max_bits: u8,
    bits: &mut Bits,
) -> Result<(u16, u32), BlockError> {
    // Refill opportunistically. Don't error on insufficient bits —
    // the actual decoded symbol's code length might be short enough
    // that the remaining buffer suffices. We only error if the chosen
    // code length exceeds what's available.
    if bits.available() < max_bits as u32 {
        bits.refill();
    }
    let mask = (1u64 << max_bits) - 1;
    let peek = (bits.peek() & mask) as usize;
    let entry = table[peek];
    let length = entry & 0x1F;
    let symbol = (entry >> 5) as u16;
    if length == 0 {
        return Err(BlockError::InvalidHuffmanCode);
    }
    if bits.available() < length {
        return Err(BlockError::EndOfFile);
    }
    Ok((symbol, length))
}

/// Emit a deflate back-reference of `(distance, length)` at the
/// current end of `output`. Bytes whose back-reference reaches before
/// the chunk start become MapMarkers indices (`MARKER_BASE +
/// (MAX_WINDOW_SIZE - distance + out_pos_in_chunk + i)`); the
/// remainder are chunk-local copies.
///
/// `out_pos` is the decoded-byte position WITHIN THE CHUNK at the
/// start of this back-ref's emission window. For a fresh Block this
/// equals `output.len()`; for multi-block per-chunk use the caller
/// passes the running chunk-relative position.
fn emit_backref(
    output: &mut Vec<u16>,
    distance: usize,
    length: usize,
    out_pos: usize,
) -> Result<(), BlockError> {
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
    output.reserve(length);
    let marker_count = distance.saturating_sub(out_pos).min(length);
    for i in 0..marker_count {
        let idx = MAX_WINDOW_SIZE + out_pos + i - distance;
        output.push(MARKER_BASE + idx as u16);
    }
    let local_count = length - marker_count;
    if local_count == 0 {
        return Ok(());
    }
    let base_dst = output.len();
    if distance >= local_count {
        // Source and destination ranges do not overlap.
        let src_start = base_dst - distance;
        let snapshot: Vec<u16> = output[src_start..src_start + local_count].to_vec();
        output.extend_from_slice(&snapshot);
    } else {
        // Overlap case (e.g. distance=1 RLE): copy one element at a
        // time so prior writes feed subsequent reads.
        for i in 0..local_count {
            let src = base_dst + i - distance;
            let v = output[src];
            output.push(v);
        }
    }
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn ensure_bits(bits: &mut Bits, n: u32) -> Result<(), BlockError> {
    if bits.available() < n {
        bits.refill();
    }
    if bits.available() < n {
        return Err(BlockError::EndOfFile);
    }
    Ok(())
}

/// Decode the literal + distance code lengths from the precode-encoded
/// stream. Mirror of rapidgzip's `readDistanceAndLiteralCodeLengths`
/// (deflate.hpp's literalCL reader, around lines 750-900).
///
/// Uses our existing `block_finder` Huffman decoder primitives via a
/// small ad-hoc canonical decoder — adequate for header parsing
/// (called once per dynamic block); the hot lit/len + distance decode
/// uses `IsalLitLenCode` + `IsalDistCode` from `isal_huffman.rs`.
fn read_literal_and_distance_code_lengths(
    bits: &mut Bits,
    precode_cl: &[u8; MAX_PRECODE_COUNT],
    total: usize,
) -> Result<Vec<u8>, BlockError> {
    // Build a simple canonical-Huffman decoder for the precode.
    // (Precode max length = 7 → a 128-entry decode table.)
    let table = build_precode_table(precode_cl)?;

    let mut out = vec![0u8; total];
    let mut i = 0;
    while i < total {
        ensure_bits(bits, 7)?;
        let peek7 = (bits.peek() & 0x7F) as usize;
        let entry = table[peek7];
        let length = (entry & 0xF) as u32;
        let symbol = entry >> 4;
        if length == 0 {
            return Err(BlockError::InvalidHuffmanCode);
        }
        bits.consume(length);
        match symbol {
            0..=15 => {
                out[i] = symbol as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(BlockError::InvalidCodeLengths);
                }
                ensure_bits(bits, 2)?;
                let repeat = (bits.peek() & 0b11) as usize + 3;
                bits.consume(2);
                let prev = out[i - 1];
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                for _ in 0..repeat {
                    out[i] = prev;
                    i += 1;
                }
            }
            17 => {
                ensure_bits(bits, 3)?;
                let repeat = (bits.peek() & 0b111) as usize + 3;
                bits.consume(3);
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                i += repeat;
            }
            18 => {
                ensure_bits(bits, 7)?;
                let repeat = (bits.peek() & 0x7F) as usize + 11;
                bits.consume(7);
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                i += repeat;
            }
            _ => return Err(BlockError::InvalidHuffmanCode),
        }
    }
    Ok(out)
}

/// Build a 128-entry decode table for the precode. Each entry packs
/// (symbol << 4) | length, with length = 0 indicating no valid code.
fn build_precode_table(precode_cl: &[u8; MAX_PRECODE_COUNT]) -> Result<[u16; 128], BlockError> {
    // Standard canonical-Huffman construction.
    let mut bl_count = [0u16; 16];
    for &len in precode_cl.iter() {
        if len > MAX_PRECODE_LENGTH {
            return Err(BlockError::InvalidCodeLengths);
        }
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }
    // Kraft check.
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for bits in 1..=MAX_PRECODE_LENGTH as usize {
        code = (code + bl_count[bits - 1] as u32) << 1;
        if code > (1u32 << bits) {
            return Err(BlockError::InvalidCodeLengths);
        }
        next_code[bits] = code;
    }
    let mut table = [0u16; 128];
    for (sym, &len) in precode_cl.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let canonical_code = next_code[len as usize];
        next_code[len as usize] += 1;
        // Reverse bits (LSB-first stream).
        let reversed = reverse_bits(canonical_code as u16, len);
        let entry = ((sym as u16) << 4) | (len as u16);
        // Fill all table positions whose low `len` bits match `reversed`.
        let step = 1usize << len;
        let mut idx = reversed as usize;
        while idx < 128 {
            table[idx] = entry;
            idx += step;
        }
    }
    Ok(table)
}

fn reverse_bits(mut v: u16, n: u8) -> u16 {
    let mut r = 0u16;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Test helper: decode the deflate stream in `data` from bit 0,
/// returning every block-start bit position observed. Each position is
/// a valid starting bit for resuming decode at the start of that
/// block's header.
///
/// Used by `backends::isal_decompress` invariant tests to oracle-check
/// ISA-L's `end_bit` values against an independently-derived set of
/// real block boundaries. Implemented by driving `Block::read_header`
/// + `Block::read` block-by-block until BFINAL.
#[cfg(test)]
pub fn record_block_starts(data: &[u8]) -> std::io::Result<Vec<usize>> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    let mut bits = Bits::new(data);
    let mut output: Vec<u16> = Vec::with_capacity(data.len().saturating_mul(4));
    let mut block = Block::new();
    let mut starts = Vec::new();
    loop {
        // Snapshot bit position at the start of this block's header.
        let consumed_bytes = bits.pos;
        let bits_in_buf = bits.available();
        let abs_bit = consumed_bytes
            .saturating_mul(8)
            .saturating_sub(bits_in_buf as usize);
        starts.push(abs_bit);

        block
            .read_header(&mut bits, false)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))?;
        while !block.eob() {
            block
                .read(&mut bits, &mut output, usize::MAX)
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}"))
                })?;
        }
        if block.is_last_block() {
            return Ok(starts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bits(data: &[u8]) -> Bits<'static> {
        // Leak a static copy so the Bits borrow is 'static within the test.
        let boxed: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());
        Bits::new(boxed)
    }

    #[test]
    fn read_header_stored_zero_length() {
        // Build a minimal stored block: BFINAL=0, BTYPE=00, byte-aligned padding=0,
        // LEN=0x0005, NLEN=0xFFFA.
        let mut bytes = vec![0b0000_0000u8]; // BFINAL=0, BTYPE=00, pad=00000
        bytes.extend_from_slice(&[0x05, 0x00, 0xFA, 0xFF]);
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).expect("header");
        assert_eq!(b.compression_type(), CompressionType::Uncompressed);
        assert!(!b.is_last_block());
        assert_eq!(b.uncompressed_size(), 5);
    }

    #[test]
    fn read_header_stored_bad_length_check() {
        let mut bytes = vec![0b0000_0000u8]; // stored
        bytes.extend_from_slice(&[0x05, 0x00, 0x00, 0x00]); // LEN != ~NLEN
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, false),
            Err(BlockError::LengthChecksumMismatch)
        );
    }

    #[test]
    fn read_header_rejects_last_block_when_requested() {
        let bytes = vec![0b0000_0001u8]; // BFINAL=1
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, true),
            Err(BlockError::UnexpectedLastBlock)
        );
    }

    #[test]
    fn read_header_fixed_huffman_no_followup() {
        let bytes = vec![0b0000_0010u8]; // BFINAL=0, BTYPE=01 (fixed)
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).expect("header");
        assert_eq!(b.compression_type(), CompressionType::FixedHuffman);
    }

    #[test]
    fn read_header_reserved_btype_rejected() {
        let bytes = vec![0b0000_0110u8]; // BFINAL=0, BTYPE=11
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, false),
            Err(BlockError::InvalidCompression)
        );
    }

    #[test]
    fn reverse_bits_works() {
        assert_eq!(reverse_bits(0b1011, 4), 0b1101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
        assert_eq!(reverse_bits(0b0001, 4), 0b1000);
    }

    #[test]
    fn read_uncompressed_payload_emits_literal_bytes() {
        // BFINAL=0, BTYPE=00, pad=0, LEN=4, NLEN=0xFFFB, then "test".
        let mut bytes = vec![0b0000_0000u8];
        bytes.extend_from_slice(&[0x04, 0x00, 0xFB, 0xFF]);
        bytes.extend_from_slice(b"test");
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).unwrap();
        let mut output: Vec<u16> = Vec::new();
        let n = b.read(&mut bits, &mut output, 1024).unwrap();
        assert_eq!(n, 4);
        assert_eq!(
            output,
            vec![b't' as u16, b'e' as u16, b's' as u16, b't' as u16]
        );
        assert!(b.eob());
    }

    #[test]
    fn read_round_trips_a_compressed_block() {
        // Both Fixed and Dynamic Huffman bodies should round-trip
        // byte-identical. flate2 picks the encoding based on payload
        // entropy; we test both via large + small payloads.
        for payload in &[
            b"a".repeat(2048),
            b"the quick brown fox jumps over the lazy dog. ".repeat(40),
        ] {
            use flate2::write::DeflateEncoder;
            use flate2::Compression;
            use std::io::Write;
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
            enc.write_all(payload).unwrap();
            let deflate_bytes = enc.finish().unwrap();
            let mut bits = make_bits(&deflate_bytes);
            let mut b = Block::new();
            b.read_header(&mut bits, false).unwrap();
            assert!(
                matches!(
                    b.compression_type(),
                    CompressionType::FixedHuffman | CompressionType::DynamicHuffman
                ),
                "expected compressed block, got {:?}",
                b.compression_type()
            );
            let mut output: Vec<u16> = Vec::new();
            let r = b.read(&mut bits, &mut output, payload.len() * 2);
            assert!(
                b.eob(),
                "decoder should reach end-of-block; read returned {:?}, output.len()={}, payload.len()={}",
                r,
                output.len(),
                payload.len(),
            );
            // For single-block flate2 output every back-ref is in-block,
            // so no markers expected.
            let resolved: Vec<u8> = output
                .iter()
                .map(|&v| {
                    assert!(v < 256, "in-block back-refs only; v={v:#x}");
                    v as u8
                })
                .collect();
            assert_eq!(resolved, *payload);
        }
    }

    #[test]
    fn build_precode_table_rejects_overlong_code() {
        let mut cl = [0u8; MAX_PRECODE_COUNT];
        cl[0] = 8; // > MAX_PRECODE_LENGTH
        assert_eq!(
            build_precode_table(&cl),
            Err(BlockError::InvalidCodeLengths)
        );
    }
}
