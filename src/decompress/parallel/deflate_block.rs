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
    fn build_precode_table_rejects_overlong_code() {
        let mut cl = [0u8; MAX_PRECODE_COUNT];
        cl[0] = 8; // > MAX_PRECODE_LENGTH
        assert_eq!(
            build_precode_table(&cl),
            Err(BlockError::InvalidCodeLengths)
        );
    }
}
