//! Deflate Stream Parser for Speculative Decompression
//!
//! This module provides bit-level parsing of deflate streams to find block
//! boundaries without full decompression. This is the core of the rapidgzip/pugz
//! algorithm for parallel decompression of arbitrary gzip files.
//!
//! # Deflate Block Types
//!
//! - BTYPE=00 (Stored): Raw uncompressed bytes
//! - BTYPE=01 (Fixed Huffman): Uses predefined Huffman tables
//! - BTYPE=10 (Dynamic Huffman): Custom Huffman tables in stream
//! - BTYPE=11: Reserved (invalid)
//!
//! # References
//! - RFC 1951: DEFLATE Compressed Data Format Specification
//! - pugz paper: https://arxiv.org/abs/1905.07224

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::io;

/// Maximum back-reference distance in deflate (32KB)
#[allow(dead_code)]
pub const DEFLATE_WINDOW_SIZE: usize = 32 * 1024;

/// A bit reader for parsing deflate streams
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0-7, bits remaining in current byte
    bits_read: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            bits_read: 0,
        }
    }

    /// Create a reader starting at a specific bit offset
    pub fn new_at_bit(data: &'a [u8], bit_offset: usize) -> Self {
        Self {
            data,
            byte_pos: bit_offset / 8,
            bit_pos: (bit_offset % 8) as u8,
            bits_read: 0,
        }
    }

    /// Current bit position in the stream
    #[inline]
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Bytes remaining in stream
    #[inline]
    pub fn bytes_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            0
        } else {
            self.data.len() - self.byte_pos
        }
    }

    /// Read a single bit (LSB first, as per deflate spec)
    #[inline]
    pub fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of stream",
            ));
        }

        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        self.bits_read += 1;

        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Read multiple bits (up to 32), LSB first
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> io::Result<u32> {
        debug_assert!(count <= 32);
        let mut value = 0u32;

        for i in 0..count {
            let bit = self.read_bit()?;
            value |= (bit as u32) << i;
        }

        Ok(value)
    }

    /// Read bits in reverse order (MSB first, for Huffman codes)
    #[inline]
    pub fn read_bits_reversed(&mut self, count: u8) -> io::Result<u32> {
        debug_assert!(count <= 32);
        let mut value = 0u32;

        for _ in 0..count {
            value = (value << 1) | self.read_bit()? as u32;
        }

        Ok(value)
    }

    /// Align to next byte boundary
    #[inline]
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    /// Read a 16-bit little-endian value (byte-aligned)
    #[inline]
    pub fn read_u16_le(&mut self) -> io::Result<u16> {
        self.align_to_byte();
        if self.byte_pos + 2 > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of stream",
            ));
        }
        let value = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
        self.byte_pos += 2;
        self.bits_read += 16;
        Ok(value)
    }

    /// Skip n bytes (must be byte-aligned)
    #[inline]
    pub fn skip_bytes(&mut self, n: usize) -> io::Result<()> {
        self.align_to_byte();
        if self.byte_pos + n > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of stream",
            ));
        }
        self.byte_pos += n;
        self.bits_read += (n * 8) as u64;
        Ok(())
    }

    /// Get remaining data as slice (byte-aligned)
    pub fn remaining_bytes(&self) -> &'a [u8] {
        let start = if self.bit_pos == 0 {
            self.byte_pos
        } else {
            self.byte_pos + 1
        };
        if start >= self.data.len() {
            &[]
        } else {
            &self.data[start..]
        }
    }
}

/// Information about a deflate block boundary
#[derive(Debug, Clone, Copy)]
pub struct BlockBoundary {
    /// Bit offset where this block starts
    pub bit_offset: usize,
    /// Byte offset (approximate, for the containing byte)
    pub byte_offset: usize,
    /// Block type (0=stored, 1=fixed, 2=dynamic)
    pub block_type: u8,
    /// Whether this is marked as final block
    pub is_final: bool,
    /// Confidence score (higher = more likely valid)
    pub confidence: f32,
}

/// Huffman code table for decoding
#[derive(Clone)]
pub struct HuffmanTable {
    /// Symbols indexed by code
    symbols: Vec<u16>,
    /// Code lengths for each symbol
    code_lengths: Vec<u8>,
    /// Maximum code length
    max_length: u8,
    /// Lookup table for fast decoding (indexed by first N bits)
    lookup: Vec<(u16, u8)>, // (symbol, length)
}

impl HuffmanTable {
    /// Build a Huffman table from code lengths
    pub fn from_lengths(lengths: &[u8]) -> io::Result<Self> {
        if lengths.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty code lengths",
            ));
        }

        let max_length = *lengths.iter().max().unwrap_or(&0);
        if max_length > 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Code length too long",
            ));
        }

        // Count codes of each length
        let mut bl_count = vec![0u32; max_length as usize + 1];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Calculate starting code for each length
        let mut next_code = vec![0u32; max_length as usize + 1];
        let mut code = 0u32;
        for bits in 1..=max_length as usize {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes to symbols
        let mut codes = vec![0u32; lengths.len()];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        // Build lookup table (for codes up to 9 bits)
        let lookup_bits = max_length.min(9);
        let lookup_size = 1 << lookup_bits;
        let mut lookup = vec![(0u16, 0u8); lookup_size];

        for (symbol, (&code, &len)) in codes.iter().zip(lengths.iter()).enumerate() {
            if len > 0 && len <= lookup_bits {
                // Fill all entries that match this code prefix
                let code_reversed = reverse_bits(code, len);
                let fill_count = 1 << (lookup_bits - len);
                for i in 0..fill_count {
                    let idx = (code_reversed | (i << len)) as usize;
                    lookup[idx] = (symbol as u16, len);
                }
            }
        }

        Ok(Self {
            symbols: (0..lengths.len() as u16).collect(),
            code_lengths: lengths.to_vec(),
            max_length,
            lookup,
        })
    }

    /// Decode a symbol from the bit stream
    pub fn decode(&self, reader: &mut BitReader) -> io::Result<u16> {
        // Try fast lookup first
        if reader.bytes_remaining() >= 2 {
            let peek_bits = peek_bits_fast(reader, self.max_length.min(9));
            let (symbol, len) = self.lookup[peek_bits as usize];
            if len > 0 {
                // Consume the bits
                for _ in 0..len {
                    reader.read_bit()?;
                }
                return Ok(symbol);
            }
        }

        // Slow path: decode bit by bit
        let mut code = 0u32;
        for len in 1..=self.max_length {
            code = (code << 1) | reader.read_bit()? as u32;

            // Check if this code matches any symbol
            for (symbol, &sym_len) in self.code_lengths.iter().enumerate() {
                if sym_len == len {
                    let sym_code = self.compute_code(symbol, len);
                    if sym_code == code {
                        return Ok(symbol as u16);
                    }
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ))
    }

    fn compute_code(&self, symbol: usize, _len: u8) -> u32 {
        // Recompute code for symbol (same algorithm as from_lengths)
        let mut bl_count = vec![0u32; self.max_length as usize + 1];
        for &l in &self.code_lengths {
            if l > 0 {
                bl_count[l as usize] += 1;
            }
        }

        let mut next_code = vec![0u32; self.max_length as usize + 1];
        let mut code = 0u32;
        for bits in 1..=self.max_length as usize {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        let mut result = 0u32;
        for (i, &l) in self.code_lengths.iter().enumerate() {
            if l > 0 {
                if i == symbol {
                    result = next_code[l as usize];
                    break;
                }
                next_code[l as usize] += 1;
            }
        }
        result
    }
}

/// Reverse bits in a value
#[inline]
fn reverse_bits(value: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Peek at bits without consuming them
#[inline]
fn peek_bits_fast(reader: &BitReader, count: u8) -> u32 {
    let byte_pos = reader.byte_pos;
    let bit_pos = reader.bit_pos as usize;

    // Read up to 3 bytes to get enough bits
    let mut value = reader.data[byte_pos] as u32 >> bit_pos;
    let mut bits_available = 8 - bit_pos;

    if bits_available < count as usize && byte_pos + 1 < reader.data.len() {
        value |= (reader.data[byte_pos + 1] as u32) << bits_available;
        bits_available += 8;
    }
    if bits_available < count as usize && byte_pos + 2 < reader.data.len() {
        value |= (reader.data[byte_pos + 2] as u32) << bits_available;
    }

    value & ((1 << count) - 1)
}

/// Fixed Huffman table for literal/length codes (BTYPE=01)
pub fn fixed_literal_lengths() -> Vec<u8> {
    let mut lengths = vec![0u8; 288];
    for i in 0..144 {
        lengths[i] = 8;
    }
    for i in 144..256 {
        lengths[i] = 9;
    }
    for i in 256..280 {
        lengths[i] = 7;
    }
    for i in 280..288 {
        lengths[i] = 8;
    }
    lengths
}

/// Fixed Huffman table for distance codes (BTYPE=01)
pub fn fixed_distance_lengths() -> Vec<u8> {
    vec![5u8; 32]
}

/// Try to parse a dynamic Huffman header at the current position
/// Returns the literal/length and distance tables if successful
pub fn try_parse_dynamic_header(
    reader: &mut BitReader,
) -> io::Result<(HuffmanTable, HuffmanTable)> {
    // Read header values
    let hlit = reader.read_bits(5)? as usize + 257; // 257-286
    let hdist = reader.read_bits(5)? as usize + 1; // 1-32
    let hclen = reader.read_bits(4)? as usize + 4; // 4-19

    // Validate ranges
    if hlit > 286 || hdist > 32 || hclen > 19 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid dynamic header values",
        ));
    }

    // Code length alphabet order (RFC 1951)
    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    // Read code length code lengths
    let mut code_length_lengths = [0u8; 19];
    for i in 0..hclen {
        code_length_lengths[CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
    }

    // Build code length Huffman table
    let code_length_table = HuffmanTable::from_lengths(&code_length_lengths)?;

    // Decode literal/length and distance code lengths
    let mut all_lengths = vec![0u8; hlit + hdist];
    let mut i = 0;

    while i < all_lengths.len() {
        let symbol = code_length_table.decode(reader)?;

        match symbol {
            0..=15 => {
                all_lengths[i] = symbol as u8;
                i += 1;
            }
            16 => {
                // Repeat previous length 3-6 times
                if i == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid repeat at start",
                    ));
                }
                let repeat = reader.read_bits(2)? as usize + 3;
                let prev = all_lengths[i - 1];
                for _ in 0..repeat {
                    if i >= all_lengths.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Repeat overflow",
                        ));
                    }
                    all_lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                // Repeat 0 length 3-10 times
                let repeat = reader.read_bits(3)? as usize + 3;
                i += repeat;
                if i > all_lengths.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Repeat overflow",
                    ));
                }
            }
            18 => {
                // Repeat 0 length 11-138 times
                let repeat = reader.read_bits(7)? as usize + 11;
                i += repeat;
                if i > all_lengths.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Repeat overflow",
                    ));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ));
            }
        }
    }

    // Split into literal/length and distance lengths
    let lit_lengths = &all_lengths[..hlit];
    let dist_lengths = &all_lengths[hlit..];

    // Validate: must have at least end-of-block symbol (256)
    if hlit <= 256 || lit_lengths[256] == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing end-of-block code",
        ));
    }

    let lit_table = HuffmanTable::from_lengths(lit_lengths)?;
    let dist_table = HuffmanTable::from_lengths(dist_lengths)?;

    Ok((lit_table, dist_table))
}

/// Scan for potential deflate block boundaries in the data
/// Returns candidate positions sorted by confidence
pub fn find_block_boundaries(data: &[u8], chunk_size: usize) -> Vec<BlockBoundary> {
    let mut boundaries = Vec::new();

    // Always include offset 0
    if let Some(boundary) = try_detect_block_at_offset(data, 0) {
        boundaries.push(boundary);
    }

    // Scan at chunk intervals and nearby positions
    let mut offset = chunk_size;
    while offset < data.len() {
        // Try exact offset and nearby positions
        for delta in [0i64, -1, 1, -2, 2, -4, 4, -8, 8].iter() {
            let test_offset = (offset as i64 + delta) as usize;
            if test_offset < data.len() {
                // Try all 8 bit offsets
                for bit_offset in 0..8 {
                    let bit_pos = test_offset * 8 + bit_offset;
                    if let Some(mut boundary) = try_detect_block_at_bit(data, bit_pos) {
                        // Adjust confidence based on alignment
                        if bit_offset == 0 {
                            boundary.confidence *= 1.2; // Byte-aligned is more likely
                        }
                        boundaries.push(boundary);
                    }
                }
            }
        }

        offset += chunk_size;
    }

    // Sort by bit offset, then by confidence (higher first)
    boundaries.sort_by(|a, b| {
        a.bit_offset
            .cmp(&b.bit_offset)
            .then(b.confidence.partial_cmp(&a.confidence).unwrap())
    });

    // Remove duplicates within 8 bits of each other (keep highest confidence)
    let mut deduped = Vec::new();
    for boundary in boundaries {
        let dominated = deduped.iter().any(|b: &BlockBoundary| {
            (b.bit_offset as i64 - boundary.bit_offset as i64).abs() < 8
                && b.confidence >= boundary.confidence
        });
        if !dominated {
            deduped.push(boundary);
        }
    }

    deduped
}

/// Try to detect a block at a byte offset
fn try_detect_block_at_offset(data: &[u8], offset: usize) -> Option<BlockBoundary> {
    try_detect_block_at_bit(data, offset * 8)
}

/// Try to detect a block at a bit offset
fn try_detect_block_at_bit(data: &[u8], bit_offset: usize) -> Option<BlockBoundary> {
    if bit_offset / 8 >= data.len() {
        return None;
    }

    let mut reader = BitReader::new_at_bit(data, bit_offset);

    // Read BFINAL and BTYPE
    let bfinal = reader.read_bit().ok()?;
    let btype = reader.read_bits(2).ok()? as u8;

    if btype == 3 {
        // Reserved, invalid
        return None;
    }

    let confidence = match btype {
        0 => {
            // Stored block: validate LEN/NLEN
            reader.align_to_byte();
            let len = reader.read_u16_le().ok()?;
            let nlen = reader.read_u16_le().ok()?;
            if len == !nlen {
                1.0 // High confidence
            } else {
                return None;
            }
        }
        1 => {
            // Fixed Huffman: just header, medium confidence
            0.6
        }
        2 => {
            // Dynamic Huffman: try to parse header
            if try_parse_dynamic_header(&mut reader).is_ok() {
                0.9 // High confidence if header parses
            } else {
                return None;
            }
        }
        _ => return None,
    };

    Some(BlockBoundary {
        bit_offset,
        byte_offset: bit_offset / 8,
        block_type: btype,
        is_final: bfinal == 1,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        // Read first byte bit by bit (LSB first)
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
    }

    #[test]
    fn test_bit_reader_multi_bits() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        // Read 3 bits: should be 100 = 4 (LSB first)
        assert_eq!(reader.read_bits(3).unwrap(), 0b100);
        // Read 5 bits: should be 10110 = 22 (LSB first)
        assert_eq!(reader.read_bits(5).unwrap(), 0b10110);
    }

    #[test]
    fn test_fixed_huffman_tables() {
        let lit_lengths = fixed_literal_lengths();
        assert_eq!(lit_lengths.len(), 288);
        assert_eq!(lit_lengths[0], 8);
        assert_eq!(lit_lengths[144], 9);
        assert_eq!(lit_lengths[256], 7);
        assert_eq!(lit_lengths[280], 8);

        let dist_lengths = fixed_distance_lengths();
        assert_eq!(dist_lengths.len(), 32);
        assert!(dist_lengths.iter().all(|&l| l == 5));
    }

    #[test]
    fn test_huffman_table_construction() {
        let lengths = fixed_literal_lengths();
        let table = HuffmanTable::from_lengths(&lengths);
        assert!(table.is_ok());
    }
}
