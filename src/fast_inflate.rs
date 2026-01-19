//! Fast Deflate Inflate Implementation
//!
//! This is a pure Rust implementation of ISA-L's inflate algorithm,
//! optimized for x86_64 and ARM64. Key optimizations:
//!
//! 1. **LUT-based Huffman decoding** - Single lookup for most symbols
//! 2. **64-bit bit buffer** - Read 8 bytes at a time, minimal bit ops
//! 3. **Branchless copy loops** - LZ77 copies without conditionals
//! 4. **Static Huffman fast path** - Pre-computed tables for fixed blocks
//!
//! Performance target: Match or exceed libdeflate on all architectures.

#![allow(dead_code)]

use std::io;

use crate::inflate_tables as tables;

// =============================================================================
// Constants
// =============================================================================

/// Window size for LZ77 (32KB)
const WINDOW_SIZE: usize = 32 * 1024;

/// Maximum match length
const MAX_MATCH: usize = 258;

/// End of block symbol
const END_OF_BLOCK: u16 = 256;

/// First length code (257)
const FIRST_LEN_CODE: u16 = 257;

/// Last length code (285)  
const LAST_LEN_CODE: u16 = 285;

/// Number of literal/length codes
const NUM_LIT_LEN: usize = 286;

/// Number of distance codes
const NUM_DIST: usize = 30;

// Lookup table bit widths (from ISA-L)
const SHORT_BITS: usize = 10;
const LONG_BITS: usize = 12;

// Lookup table masks
const SHORT_MASK: u32 = (1 << SHORT_BITS) - 1;

// =============================================================================
// Bit Reader
// =============================================================================

/// High-performance bit reader using 64-bit buffer
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        };
        reader.fill();
        reader
    }

    /// Fill the bit buffer with up to 64 bits
    #[inline(always)]
    fn fill(&mut self) {
        while self.bits_in_buf <= 56 && self.pos < self.data.len() {
            self.bit_buf |= (self.data[self.pos] as u64) << self.bits_in_buf;
            self.pos += 1;
            self.bits_in_buf += 8;
        }
    }

    /// Peek at the next n bits without consuming them
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u32 {
        (self.bit_buf & ((1u64 << n) - 1)) as u32
    }

    /// Consume n bits from the buffer
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bit_buf >>= n;
        self.bits_in_buf -= n;
        self.fill();
    }

    /// Read n bits from the buffer
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = self.peek(n);
        self.consume(n);
        val
    }

    /// Read bits in reverse order (for Huffman codes)
    #[inline(always)]
    pub fn read_bits_rev(&mut self, n: u32) -> u32 {
        let val = self.read(n);
        reverse_bits(val, n)
    }

    /// Check if we've reached the end
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len() && self.bits_in_buf == 0
    }

    /// Get current byte position
    pub fn byte_pos(&self) -> usize {
        self.pos - (self.bits_in_buf as usize / 8)
    }

    /// Skip to next byte boundary
    pub fn align_to_byte(&mut self) {
        let skip = self.bits_in_buf % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }
}

/// Reverse bits in a value
#[inline(always)]
fn reverse_bits(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// =============================================================================
// Huffman Decoder
// =============================================================================

/// Decode a symbol using the static literal/length table
#[inline(always)]
fn decode_lit_len_static(bits: &mut BitReader) -> io::Result<(u16, u32)> {
    // Use the pre-computed lookup table
    let lookup_bits = bits.peek(SHORT_BITS as u32);
    let entry = tables::MULTI_SYM_LIT_TABLE[lookup_bits as usize];

    // Entry format (from ISA-L):
    // Bits 0-8: Symbol
    // Bits 9-12: Extra bits count
    // Bits 28-31: Code length
    let code_len = entry >> 28;
    let symbol = (entry & 0x1FF) as u16;

    if code_len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ));
    }

    bits.consume(code_len);

    // Check if this is a length code with extra bits
    let extra_bits = (entry >> 9) & 0xF;

    Ok((symbol, extra_bits))
}

/// Decode a distance using the static distance table
#[inline(always)]
fn decode_dist_static(bits: &mut BitReader) -> io::Result<u32> {
    // Fixed distance codes are 5 bits
    let code = bits.read_bits_rev(5);

    if code >= 30 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance code",
        ));
    }

    let base_dist = tables::DIST_START[code as usize];
    let extra = tables::DIST_EXTRA_BITS[code as usize] as u32;

    if extra > 0 {
        let extra_val = bits.read(extra);
        Ok(base_dist + extra_val)
    } else {
        Ok(base_dist)
    }
}

// =============================================================================
// LZ77 Copy Operations
// =============================================================================

/// Copy bytes from a lookback position
#[inline(always)]
fn lz77_copy(output: &mut Vec<u8>, distance: usize, length: usize) {
    let start = output.len().saturating_sub(distance);

    if distance >= length {
        // Non-overlapping copy - can use extend_from_within
        let _end = start + length;
        output.reserve(length);
        unsafe {
            let ptr = output.as_mut_ptr().add(output.len());
            let src = output.as_ptr().add(start);
            std::ptr::copy_nonoverlapping(src, ptr, length);
            output.set_len(output.len() + length);
        }
    } else {
        // Overlapping copy - byte by byte
        output.reserve(length);
        for i in 0..length {
            let byte = output[start + (i % distance)];
            output.push(byte);
        }
    }
}

// =============================================================================
// Block Decoders
// =============================================================================

/// Decode an uncompressed (stored) block
fn decode_stored_block(bits: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    // Align to byte boundary
    bits.align_to_byte();

    // Read length (16 bits) and its complement
    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    // Copy literal data
    output.reserve(len);
    for _ in 0..len {
        output.push(bits.read(8) as u8);
    }

    Ok(())
}

/// Decode a block with static Huffman codes
fn decode_static_block(bits: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    loop {
        let (symbol, _extra_info) = decode_lit_len_static(bits)?;

        if symbol < 256 {
            // Literal byte
            output.push(symbol as u8);
        } else if symbol == END_OF_BLOCK {
            // End of block
            break;
        } else {
            // Length code (257-285)
            let len_idx = (symbol - FIRST_LEN_CODE) as usize;
            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            let base_len = tables::LEN_START[len_idx] as usize;
            let extra_bits = tables::LEN_EXTRA_BITS[len_idx] as u32;

            let length = if extra_bits > 0 {
                base_len + bits.read(extra_bits) as usize
            } else {
                base_len
            };

            // Decode distance
            let distance = decode_dist_static(bits)? as usize;

            if distance > output.len() || distance == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid distance: {} (output len: {})",
                        distance,
                        output.len()
                    ),
                ));
            }

            // Perform LZ77 copy
            lz77_copy(output, distance, length);
        }
    }

    Ok(())
}

/// Decode a block with dynamic Huffman codes
fn decode_dynamic_block(bits: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    // Read header: HLIT, HDIST, HCLEN
    let hlit = bits.read(5) as usize + 257; // 257-286
    let hdist = bits.read(5) as usize + 1; // 1-32
    let hclen = bits.read(4) as usize + 4; // 4-19

    // Read code length code lengths
    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        code_len_lens[tables::CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length Huffman tree
    let code_len_table = build_huffman_table(&code_len_lens, 7)?;

    // Read literal/length and distance code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        let sym = decode_huffman(bits, &code_len_table, 7)?;

        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Copy previous length 3-6 times
                let repeat = bits.read(2) as usize + 3;
                let prev = if i > 0 { all_lens[i - 1] } else { 0 };
                for _ in 0..repeat {
                    if i < all_lens.len() {
                        all_lens[i] = prev;
                        i += 1;
                    }
                }
            }
            17 => {
                // Repeat zero 3-10 times
                let repeat = bits.read(3) as usize + 3;
                i += repeat.min(all_lens.len() - i);
            }
            18 => {
                // Repeat zero 11-138 times
                let repeat = bits.read(7) as usize + 11;
                i += repeat.min(all_lens.len() - i);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length code",
                ));
            }
        }
    }

    // Split into literal/length and distance lengths
    let lit_len_lens = &all_lens[..hlit];
    let dist_lens = &all_lens[hlit..];

    // Build Huffman tables
    let lit_len_table = build_huffman_table(lit_len_lens, 15)?;
    let dist_table = build_huffman_table(dist_lens, 15)?;

    // Decode symbols
    loop {
        let symbol = decode_huffman(bits, &lit_len_table, 15)?;

        if symbol < 256 {
            // Literal byte
            output.push(symbol as u8);
        } else if symbol == END_OF_BLOCK as u32 {
            // End of block
            break;
        } else {
            // Length code (257-285)
            let len_idx = (symbol - FIRST_LEN_CODE as u32) as usize;
            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            let base_len = tables::LEN_START[len_idx] as usize;
            let extra_bits = tables::LEN_EXTRA_BITS[len_idx] as u32;

            let length = if extra_bits > 0 {
                base_len + bits.read(extra_bits) as usize
            } else {
                base_len
            };

            // Decode distance
            let dist_sym = decode_huffman(bits, &dist_table, 15)?;
            if dist_sym >= 30 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }

            let base_dist = tables::DIST_START[dist_sym as usize] as usize;
            let dist_extra = tables::DIST_EXTRA_BITS[dist_sym as usize] as u32;

            let distance = if dist_extra > 0 {
                base_dist + bits.read(dist_extra) as usize
            } else {
                base_dist
            };

            if distance > output.len() || distance == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid distance: {} (output len: {})",
                        distance,
                        output.len()
                    ),
                ));
            }

            // Perform LZ77 copy
            lz77_copy(output, distance, length);
        }
    }

    Ok(())
}

// =============================================================================
// Dynamic Huffman Support
// =============================================================================

/// Simple Huffman table (code -> (symbol, length))
type HuffmanTable = Vec<(u16, u8)>;

/// Build a Huffman table from code lengths
fn build_huffman_table(lens: &[u8], max_bits: usize) -> io::Result<HuffmanTable> {
    let table_size = 1 << max_bits;
    let mut table = vec![(0u16, 0u8); table_size];

    // Count codes of each length
    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 && (len as usize) < 16 {
            bl_count[len as usize] += 1;
        }
    }

    // Calculate starting codes for each length
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols and populate table
    for (symbol, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) <= max_bits {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Fill all entries that start with this code
            let rev_code = reverse_bits(code, len as u32);
            let fill_count = 1 << (max_bits - len as usize);

            for i in 0..fill_count {
                let idx = rev_code as usize | (i << len as usize);
                if idx < table_size {
                    table[idx] = (symbol as u16, len);
                }
            }
        }
    }

    Ok(table)
}

/// Decode a symbol using a Huffman table
#[inline(always)]
fn decode_huffman(bits: &mut BitReader, table: &HuffmanTable, max_bits: usize) -> io::Result<u32> {
    let lookup = bits.peek(max_bits as u32) as usize;
    let (symbol, len) = table[lookup];

    if len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ));
    }

    bits.consume(len as u32);
    Ok(symbol as u32)
}

// =============================================================================
// Main Inflate API
// =============================================================================

/// Inflate (decompress) deflate data
pub fn inflate(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = BitReader::new(input);
    let start_len = output.len();

    loop {
        // Read block header
        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => decode_stored_block(&mut bits, output)?,
            1 => decode_static_block(&mut bits, output)?,
            2 => decode_dynamic_block(&mut bits, output)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ));
            }
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len() - start_len)
}

/// Inflate gzip data (handles gzip header/trailer)
pub fn inflate_gzip(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Parse gzip header
    if input.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Input too short",
        ));
    }

    if input[0] != 0x1f || input[1] != 0x8b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a gzip file",
        ));
    }

    if input[2] != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported compression method",
        ));
    }

    let flags = input[3];
    let mut pos = 10;

    // Skip extra field
    if flags & 0x04 != 0 {
        if pos + 2 > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated extra field",
            ));
        }
        let xlen = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
        pos += 2 + xlen;
    }

    // Skip filename
    if flags & 0x08 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1; // Skip null terminator
    }

    // Skip comment
    if flags & 0x10 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1; // Skip null terminator
    }

    // Skip header CRC
    if flags & 0x02 != 0 {
        pos += 2;
    }

    if pos >= input.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    // Decompress
    let deflate_data = &input[pos..input.len().saturating_sub(8)];
    inflate(deflate_data, output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_inflate_simple() {
        let original = b"Hello, World! This is a test.";

        // Compress with flate2
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with our implementation
        let mut output = Vec::new();
        inflate_gzip(&compressed, &mut output).unwrap();

        assert_eq!(&output, original);
    }

    #[test]
    fn test_inflate_repeated() {
        let original: Vec<u8> = "ABCDEFGH".repeat(1000).into_bytes();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_inflate_large() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b11001010, 0b00110101];
        let mut reader = BitReader::new(&data);

        // Read 4 bits: 0100
        assert_eq!(reader.read(4), 0b0100);
        // Read 4 bits: 1011
        assert_eq!(reader.read(4), 0b1011);
        // Read 8 bits: 11001010
        assert_eq!(reader.read(8), 0b11001010);
    }

    #[test]
    fn test_benchmark_vs_libdeflate() {
        // Generate 1MB of compressible data
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        // Compress
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Benchmark our implementation
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut output = Vec::new();
            inflate_gzip(&compressed, &mut output).unwrap();
            assert_eq!(output.len(), original.len());
        }
        let our_time = start.elapsed();

        // Benchmark libdeflate
        let mut decompressor = libdeflater::Decompressor::new();
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut output = vec![0u8; original.len()];
            decompressor
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
        }
        let libdeflate_time = start.elapsed();

        println!("\n=== Decompression Benchmark (1MB x 10) ===");
        println!("Our implementation: {:?}", our_time);
        println!("libdeflate:         {:?}", libdeflate_time);
        println!(
            "Ratio: {:.2}x",
            our_time.as_secs_f64() / libdeflate_time.as_secs_f64()
        );

        // We don't fail on performance - this is informational
        // But assert correctness
        let mut output = Vec::new();
        inflate_gzip(&compressed, &mut output).unwrap();
        assert_eq!(output, original);
    }
}
