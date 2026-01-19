//! Turbo Inflate - Maximum Performance Decompression
//!
//! This module implements ISA-L's key optimizations in pure Rust:
//! 1. Multi-symbol Huffman decode (2-3 literals per lookup)
//! 2. 64-bit bit buffer with bulk refill
//! 3. Optimized LZ77 copy with pattern expansion
//! 4. Minimal branching in hot path

#![allow(dead_code)]

use std::io;

use crate::inflate_tables::{
    CODE_LENGTH_ORDER, DIST_EXTRA_BITS, DIST_START, LARGE_CODE_LEN_OFFSET, LARGE_FLAG_BIT,
    LARGE_SYM_COUNT_OFFSET, LEN_EXTRA_BITS, LEN_START, MULTI_SYM_LIT_TABLE,
};

// =============================================================================
// Constants
// =============================================================================

const REFILL_THRESHOLD: u32 = 24; // Refill when bits drop below this
const END_OF_BLOCK: u16 = 256;

// =============================================================================
// Turbo Bit Buffer
// =============================================================================

/// Ultra-fast bit buffer optimized for Huffman decoding
pub struct TurboBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
}

impl<'a> TurboBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut tb = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
        };
        tb.refill_full();
        tb
    }

    /// Refill to 56+ bits (ISA-L style)
    #[inline(always)]
    fn refill_full(&mut self) {
        // Read 8 bytes at once if possible
        if self.pos + 8 <= self.data.len() {
            let bytes =
                unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= bytes.to_le() << self.bits;
            let bytes_consumed = (64 - self.bits) / 8;
            self.pos += bytes_consumed as usize;
            self.bits += bytes_consumed * 8;
        } else {
            // Slow path: byte by byte
            while self.bits <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.bits;
                self.pos += 1;
                self.bits += 8;
            }
        }
    }

    /// Quick refill if needed
    #[inline(always)]
    fn refill(&mut self) {
        if self.bits <= REFILL_THRESHOLD {
            self.refill_full();
        }
    }

    /// Peek 12 bits for Huffman lookup
    #[inline(always)]
    fn peek12(&self) -> usize {
        (self.buf & 0xFFF) as usize
    }

    /// Peek n bits
    #[inline(always)]
    fn peek(&self, n: u32) -> u32 {
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Consume n bits
    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits -= n;
    }

    /// Read n bits
    #[inline(always)]
    fn read(&mut self, n: u32) -> u32 {
        let val = self.peek(n);
        self.consume(n);
        val
    }

    /// Align to byte boundary
    #[inline]
    fn align(&mut self) {
        let skip = self.bits % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }
}

// =============================================================================
// Multi-Symbol Decode
// =============================================================================

/// Result of multi-symbol decode
#[derive(Clone, Copy)]
struct DecodeResult {
    /// Packed literals (up to 3)
    lits: u32,
    /// Number of symbols (1-3, or 0 if length code)
    count: u8,
    /// Bits consumed
    bits: u8,
    /// If count=0, this is the length symbol
    symbol: u16,
}

/// Decode using multi-symbol table
#[inline(always)]
fn decode_multi(bits: &mut TurboBits) -> DecodeResult {
    let entry = MULTI_SYM_LIT_TABLE[bits.peek12()];
    let code_len = (entry >> LARGE_CODE_LEN_OFFSET) as u8;

    bits.consume(code_len as u32);

    if entry & LARGE_FLAG_BIT == 0 {
        // Multi-symbol: 2-3 literals packed
        let count = ((entry >> LARGE_SYM_COUNT_OFFSET) & 0x3) as u8 + 1;
        DecodeResult {
            lits: entry & 0xFFFFFF,
            count,
            bits: code_len,
            symbol: 0,
        }
    } else {
        // Single symbol (literal, length code, or end of block)
        let symbol = (entry & 0x3FF) as u16;
        if symbol < 256 {
            DecodeResult {
                lits: symbol as u32,
                count: 1,
                bits: code_len,
                symbol,
            }
        } else {
            // Length code or end of block
            DecodeResult {
                lits: 0,
                count: 0,
                bits: code_len,
                symbol,
            }
        }
    }
}

// =============================================================================
// LZ77 Copy
// =============================================================================

/// Fast LZ77 copy with SIMD optimization
#[inline(always)]
fn lz77_copy(output: &mut Vec<u8>, distance: usize, length: usize) {
    crate::simd_copy::lz77_copy_fast(output, distance, length);
}

// =============================================================================
// Distance Decode
// =============================================================================

#[inline(always)]
fn reverse_bits_5(val: u32) -> u32 {
    ((val & 0x01) << 4)
        | ((val & 0x02) << 2)
        | (val & 0x04)
        | ((val & 0x08) >> 2)
        | ((val & 0x10) >> 4)
}

#[inline(always)]
fn decode_distance_fixed(bits: &mut TurboBits) -> io::Result<usize> {
    bits.refill();
    let code = reverse_bits_5(bits.read(5)) as usize;

    if code >= 30 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance code",
        ));
    }

    let base = DIST_START[code] as usize;
    let extra = DIST_EXTRA_BITS[code] as u32;

    if extra > 0 {
        bits.refill();
        Ok(base + bits.read(extra) as usize)
    } else {
        Ok(base)
    }
}

// =============================================================================
// Block Decoders
// =============================================================================

fn decode_stored_block(bits: &mut TurboBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    output.reserve(len);
    for _ in 0..len {
        bits.refill();
        output.push(bits.read(8) as u8);
    }

    Ok(())
}

/// Decode fixed Huffman block with optimized loop
fn decode_fixed_block_turbo(bits: &mut TurboBits, output: &mut Vec<u8>) -> io::Result<()> {
    // Pre-reserve likely output size
    output.reserve(32 * 1024);

    loop {
        bits.refill();

        // Fast path: decode up to 4 literals without checking for length codes
        // This reduces branches in the common case (literals dominate)
        let entry1 = MULTI_SYM_LIT_TABLE[bits.peek12()];
        let code_len1 = entry1 >> LARGE_CODE_LEN_OFFSET;
        bits.consume(code_len1);

        if entry1 & LARGE_FLAG_BIT == 0 || (entry1 & 0x3FF) < 256 {
            // Literal(s)
            if entry1 & LARGE_FLAG_BIT == 0 {
                let count = ((entry1 >> LARGE_SYM_COUNT_OFFSET) & 0x3) as usize + 1;
                if count == 2 {
                    output.push(entry1 as u8);
                    output.push((entry1 >> 8) as u8);
                } else if count == 3 {
                    output.push(entry1 as u8);
                    output.push((entry1 >> 8) as u8);
                    output.push((entry1 >> 16) as u8);
                } else {
                    output.push(entry1 as u8);
                }
            } else {
                output.push((entry1 & 0xFF) as u8);
            }
            continue;
        }

        // Length code or end of block
        let symbol = (entry1 & 0x3FF) as u16;

        if symbol == END_OF_BLOCK {
            break;
        }

        // Length code (257-285)
        let len_idx = (symbol - 257) as usize;
        if len_idx >= 29 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length code",
            ));
        }

        bits.refill();
        let base_len = LEN_START[len_idx] as usize;
        let extra = LEN_EXTRA_BITS[len_idx] as u32;
        let length = base_len + bits.read(extra) as usize;

        // Decode distance
        let distance = decode_distance_fixed(bits)?;

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

        lz77_copy(output, distance, length);
    }

    Ok(())
}

/// Decode dynamic Huffman block with optimized table lookup
fn decode_dynamic_block(bits: &mut TurboBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.refill();

    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    // Read code length code lengths
    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        bits.refill();
        code_len_lens[CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length Huffman table
    let code_len_table = build_huffman_table(&code_len_lens, 7)?;

    // Read all code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        bits.refill();

        let lookup = bits.peek(7) as usize;
        let (sym, len) = code_len_table[lookup];

        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid code length code",
            ));
        }
        bits.consume(len as u32);

        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                bits.refill();
                let repeat = bits.read(2) as usize + 3;
                let prev = if i > 0 { all_lens[i - 1] } else { 0 };
                for _ in 0..repeat.min(all_lens.len() - i) {
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill();
                let repeat = bits.read(3) as usize + 3;
                i += repeat.min(all_lens.len() - i);
            }
            18 => {
                bits.refill();
                let repeat = bits.read(7) as usize + 11;
                i += repeat.min(all_lens.len() - i);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length code",
                ))
            }
        }
    }

    // Build Huffman tables with original method (handles long codes)
    let lit_len_lens = &all_lens[..hlit];
    let dist_lens = &all_lens[hlit..];

    let lit_len_table = build_huffman_table(lit_len_lens, 15)?;
    let dist_table = build_huffman_table(dist_lens, 15)?;

    // Pre-reserve output
    output.reserve(32 * 1024);

    // Decode symbols with optimized loop
    loop {
        bits.refill();

        let lookup = bits.peek(15) as usize;
        let (symbol, len) = lit_len_table[lookup];

        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code",
            ));
        }
        bits.consume(len as u32);

        if symbol < 256 {
            output.push(symbol as u8);
            continue;
        }

        if symbol == 256 {
            break;
        }

        // Length code (257-285)
        let len_idx = (symbol - 257) as usize;
        if len_idx >= 29 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length code",
            ));
        }

        bits.refill();
        let base_len = LEN_START[len_idx] as usize;
        let extra_bits = LEN_EXTRA_BITS[len_idx] as u32;
        let length = base_len + bits.read(extra_bits) as usize;

        // Decode distance
        let dist_lookup = bits.peek(15) as usize;
        let (dist_sym, dist_len) = dist_table[dist_lookup];

        if dist_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }
        bits.consume(dist_len as u32);

        if dist_sym >= 30 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        bits.refill();
        let base_dist = DIST_START[dist_sym as usize] as usize;
        let dist_extra = DIST_EXTRA_BITS[dist_sym as usize] as u32;
        let distance = base_dist + bits.read(dist_extra) as usize;

        if distance > output.len() || distance == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        lz77_copy(output, distance, length);
    }

    Ok(())
}

/// Build a fast lookup table packed as: [symbol:16][length:16] in u32
fn build_huffman_table_fast(lens: &[u8], max_bits: usize) -> io::Result<Vec<u32>> {
    let table_size = 1 << max_bits;
    let mut table = vec![0u32; table_size];

    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 && (len as usize) < 16 {
            bl_count[len as usize] += 1;
        }
    }

    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for (symbol, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) <= max_bits {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            let rev_code = reverse_bits_n(code, len as u32);
            let fill_count = 1 << (max_bits - len as usize);

            // Pack: symbol in low 16 bits, length in high 16 bits
            let entry = (symbol as u32) | ((len as u32) << 16);

            for i in 0..fill_count {
                let idx = rev_code as usize | (i << len as usize);
                if idx < table_size {
                    table[idx] = entry;
                }
            }
        }
    }

    Ok(table)
}

/// Build a Huffman lookup table (standard single-level)
fn build_huffman_table(lens: &[u8], max_bits: usize) -> io::Result<Vec<(u16, u8)>> {
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

    // Assign codes to symbols
    for (symbol, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) <= max_bits {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            let rev_code = reverse_bits_n(code, len as u32);
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

fn reverse_bits_n(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// =============================================================================
// Main API
// =============================================================================

/// Turbo inflate - fastest pure Rust deflate decompression
pub fn inflate_turbo(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = TurboBits::new(input);
    let start_len = output.len();

    loop {
        bits.refill();

        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => decode_stored_block(&mut bits, output)?,
            1 => decode_fixed_block_turbo(&mut bits, output)?,
            2 => decode_dynamic_block(&mut bits, output)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ))
            }
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len() - start_len)
}

/// Turbo inflate for gzip data
pub fn inflate_gzip_turbo(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
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
            "Unsupported compression",
        ));
    }

    let flags = input[3];
    let mut pos = 10;

    // Skip optional fields
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

    if flags & 0x08 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x10 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x02 != 0 {
        pos += 2;
    }

    if pos >= input.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    let deflate_data = &input[pos..input.len().saturating_sub(8)];
    inflate_turbo(deflate_data, output)
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
    fn test_turbo_inflate_simple() {
        let original = b"Hello, World! This is a test of turbo inflate.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_turbo(&compressed, &mut output).unwrap();

        assert_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_turbo_inflate_repeated() {
        let original: Vec<u8> = "ABCDEFGH".repeat(1000).into_bytes();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_turbo(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_turbo_inflate_large() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_turbo(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_turbo_benchmark() {
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        const WARMUP: usize = 5;
        const ITERS: usize = 50;

        // Warmup
        for _ in 0..WARMUP {
            let mut output = Vec::new();
            inflate_gzip_turbo(&compressed, &mut output).unwrap();
            let mut output2 = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .gzip_decompress(&compressed, &mut output2)
                .unwrap();
        }

        // Benchmark turbo implementation
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = Vec::with_capacity(original.len());
            inflate_gzip_turbo(&compressed, &mut output).unwrap();
            std::hint::black_box(&output);
        }
        let turbo_time = start.elapsed();

        // Benchmark basic implementation
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = Vec::with_capacity(original.len());
            crate::fast_inflate::inflate_gzip(&compressed, &mut output).unwrap();
            std::hint::black_box(&output);
        }
        let basic_time = start.elapsed();

        // Benchmark libdeflate
        let mut decompressor = libdeflater::Decompressor::new();
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len()];
            decompressor
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
            std::hint::black_box(&output);
        }
        let libdeflate_time = start.elapsed();

        let turbo_avg = turbo_time / ITERS as u32;
        let basic_avg = basic_time / ITERS as u32;
        let libdeflate_avg = libdeflate_time / ITERS as u32;

        let turbo_mbps = 1_000_000.0 / turbo_avg.as_secs_f64() / 1_000_000.0;
        let libdeflate_mbps = 1_000_000.0 / libdeflate_avg.as_secs_f64() / 1_000_000.0;

        println!("\n=== TURBO Decompression Benchmark (1MB x {}) ===", ITERS);
        println!(
            "Turbo:      {:>8?}/iter  ({:.0} MB/s)",
            turbo_avg, turbo_mbps
        );
        println!("Basic:      {:>8?}/iter", basic_avg);
        println!(
            "libdeflate: {:>8?}/iter  ({:.0} MB/s)",
            libdeflate_avg, libdeflate_mbps
        );
        println!(
            "Turbo vs libdeflate: {:.2}x",
            turbo_avg.as_secs_f64() / libdeflate_avg.as_secs_f64()
        );
        println!(
            "Turbo vs basic:      {:.2}x",
            turbo_avg.as_secs_f64() / basic_avg.as_secs_f64()
        );
    }
}
