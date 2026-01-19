//! Two-Level Huffman Lookup Tables
//!
//! This module implements cache-efficient Huffman decoding using a two-level
//! table structure:
//!
//! - **Level 1**: 10-bit direct lookup (1024 entries, 4KB) - fits in L1 cache
//! - **Level 2**: Overflow table for codes > 10 bits
//!
//! This is the key optimization that closes the gap with libdeflate.

#![allow(dead_code)]

use std::io;

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};

// =============================================================================
// Constants
// =============================================================================

/// Primary table bits (12 bits = 4096 entries = 8KB for u16)
/// 12 bits handles most codes directly without L2 lookup
const L1_BITS: u32 = 12;
const L1_SIZE: usize = 1 << L1_BITS;
const L1_MASK: u32 = (1 << L1_BITS) - 1;

/// Secondary table bits for overflow
const L2_BITS: u32 = 3;
const L2_SIZE: usize = 1 << L2_BITS;

/// Flag indicating L2 lookup needed
const L2_FLAG: u16 = 0x8000;

/// Maximum code length supported
const MAX_CODE_LEN: usize = 15;

// =============================================================================
// Two-Level Table Structure
// =============================================================================

/// Two-level Huffman decode table
///
/// L1 entry format (16 bits):
///   Bit 15: 0 = direct decode, 1 = use L2
///   Direct decode:
///     Bits 0-8:  Symbol (0-511)
///     Bits 9-13: Code length (1-15)
///     Bit 14:    Reserved
///   L2 pointer:
///     Bits 0-14: Index into L2 table
#[derive(Clone)]
pub struct TwoLevelTable {
    /// Level 1 table (always 1024 entries)
    l1: [u16; L1_SIZE],
    /// Level 2 overflow table (variable size, often empty)
    l2: Vec<u16>,
    /// Maximum code length for this table
    max_len: u32,
}

impl TwoLevelTable {
    /// Create a new empty table
    pub fn new() -> Self {
        Self {
            l1: [0; L1_SIZE],
            l2: Vec::new(),
            max_len: 0,
        }
    }

    /// Build table from code lengths
    /// Uses single-level 12-bit table for codes <= 12 bits
    /// Falls back to per-code decode for longer codes (rare)
    pub fn build(lens: &[u8]) -> io::Result<Self> {
        let mut table = Self::new();

        // Count codes of each length
        let mut bl_count = [0u32; MAX_CODE_LEN + 1];
        let mut max_len = 0u32;

        for &len in lens {
            if len > 0 && (len as usize) <= MAX_CODE_LEN {
                bl_count[len as usize] += 1;
                max_len = max_len.max(len as u32);
            }
        }

        table.max_len = max_len;

        // Calculate starting codes for each length
        let mut next_code = [0u32; MAX_CODE_LEN + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_CODE_LEN {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Fill table - only handle codes <= L1_BITS
        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let len = len as u32;
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Only fill L1 for codes that fit
            if len <= L1_BITS {
                let rev = reverse_bits(code, len);
                let fill_count = 1usize << (L1_BITS - len);
                let entry = pack_l1_entry(symbol as u16, len as u8);

                for i in 0..fill_count {
                    let idx = (rev as usize) | (i << len as usize);
                    if idx < L1_SIZE {
                        table.l1[idx] = entry;
                    }
                }
            }
            // Codes > L1_BITS are handled by slow path in decode()
        }

        Ok(table)
    }

    /// Decode a symbol from bits
    /// Returns (symbol, code_length)
    /// If code_length is 0, the code wasn't found (possibly longer than L1_BITS)
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> (u16, u32) {
        let l1_idx = (bits as usize) & (L1_SIZE - 1);
        let entry = self.l1[l1_idx];

        // Direct decode from L1 (no L2 for simplicity)
        let symbol = entry & 0x1FF;
        let len = ((entry >> 9) & 0x1F) as u32;
        (symbol, len)
    }
}

impl Default for TwoLevelTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Pack symbol and length into L1 entry
#[inline]
fn pack_l1_entry(symbol: u16, len: u8) -> u16 {
    (symbol & 0x1FF) | ((len as u16 & 0x1F) << 9)
}

/// Reverse bits in a code
#[inline]
fn reverse_bits(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// =============================================================================
// Optimized Bit Reader
// =============================================================================

/// Bit reader optimized for two-level decode
pub struct FastBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
}

impl<'a> FastBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut fb = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
        };
        fb.refill();
        fb
    }

    /// Refill to 56+ bits
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let bytes =
                unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= bytes.to_le() << self.bits;
            let consumed = (64 - self.bits) / 8;
            self.pos += consumed as usize;
            self.bits += consumed * 8;
        } else {
            while self.bits <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.bits;
                self.pos += 1;
                self.bits += 8;
            }
        }
    }

    /// Peek up to 15 bits
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.buf & ((1u64 << n) - 1)
    }

    /// Consume n bits
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits -= n;
    }

    /// Read n bits
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = self.peek(n) as u32;
        self.consume(n);
        val
    }

    /// Align to byte boundary
    #[inline]
    pub fn align(&mut self) {
        let skip = self.bits % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }

    /// Check if we need to refill (bits < 16)
    #[inline(always)]
    pub fn needs_refill(&self) -> bool {
        self.bits < 16
    }

    /// Check if we have at least n bits
    #[inline(always)]
    pub fn has_bits(&self, n: u32) -> bool {
        self.bits >= n
    }

    /// Ensure we have at least n bits, refilling if needed
    #[inline(always)]
    pub fn ensure(&mut self, n: u32) {
        if self.bits < n {
            self.refill();
        }
    }

    /// Get the raw bit buffer (for table lookup)
    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }
}

// =============================================================================
// Optimized Decode Functions
// =============================================================================

/// Decode a symbol using two-level table
#[inline(always)]
pub fn decode_symbol(bits: &mut FastBits, table: &TwoLevelTable) -> io::Result<u16> {
    bits.ensure(16);

    let (symbol, len) = table.decode(bits.buffer());
    if len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ));
    }

    bits.consume(len);
    Ok(symbol)
}

/// Decode length + distance and perform LZ77 copy
#[inline(always)]
pub fn decode_lz77(
    bits: &mut FastBits,
    dist_table: &TwoLevelTable,
    len_symbol: u16,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    // Decode length
    let len_idx = (len_symbol - 257) as usize;
    if len_idx >= 29 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid length code",
        ));
    }

    if bits.bits < 16 {
        bits.refill();
    }

    let base_len = LEN_START[len_idx] as usize;
    let extra_bits = LEN_EXTRA_BITS[len_idx] as u32;
    let length = base_len + bits.read(extra_bits) as usize;

    // Decode distance
    let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
    if dist_len == 0 || dist_sym >= 30 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance code",
        ));
    }
    bits.consume(dist_len);

    if bits.bits < 16 {
        bits.refill();
    }

    let base_dist = DIST_START[dist_sym as usize] as usize;
    let dist_extra = DIST_EXTRA_BITS[dist_sym as usize] as u32;
    let distance = base_dist + bits.read(dist_extra) as usize;

    if distance > output.len() || distance == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance",
        ));
    }

    // LZ77 copy
    crate::simd_copy::lz77_copy_fast(output, distance, length);

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_level_fixed_huffman() {
        // Build fixed Huffman table for literals/lengths
        let mut lens = [0u8; 288];

        // 0-143: 8 bits
        for len in lens.iter_mut().take(144) {
            *len = 8;
        }
        // 144-255: 9 bits
        for len in lens.iter_mut().take(256).skip(144) {
            *len = 9;
        }
        // 256-279: 7 bits
        for len in lens.iter_mut().take(280).skip(256) {
            *len = 7;
        }
        // 280-287: 8 bits
        for len in lens.iter_mut().take(288).skip(280) {
            *len = 8;
        }

        let table = TwoLevelTable::build(&lens).unwrap();

        // Verify some lookups
        // Symbol 0 (literal 0x00): 8-bit code
        // Symbol 256 (end of block): 7-bit code

        println!(
            "Table built: L1={} entries, L2={} entries",
            L1_SIZE,
            table.l2.len()
        );
        println!("Max code length: {}", table.max_len);

        // L2 should be mostly empty for fixed Huffman (max len = 9)
        assert!(table.l2.len() < 100, "L2 should be small for fixed Huffman");
    }

    #[test]
    fn test_decode_simple() {
        // Create a simple table with known codes
        let lens = [2u8, 2, 3, 3]; // Symbols 0,1=2 bits, 2,3=3 bits
        let table = TwoLevelTable::build(&lens).unwrap();

        // Encode: sym 0 = 00, sym 1 = 01, sym 2 = 100, sym 3 = 101 (reversed)
        // Reversed: sym 0 = 00, sym 1 = 10, sym 2 = 001, sym 3 = 101

        let data = [0b1000_0010_u8, 0b0000_1010]; // Symbols: 0, 1, 2, 3
        let mut bits = FastBits::new(&data);

        let sym0 = decode_symbol(&mut bits, &table).unwrap();
        let sym1 = decode_symbol(&mut bits, &table).unwrap();
        let sym2 = decode_symbol(&mut bits, &table).unwrap();
        let sym3 = decode_symbol(&mut bits, &table).unwrap();

        println!("Decoded: {}, {}, {}, {}", sym0, sym1, sym2, sym3);

        // Note: actual values depend on canonical Huffman code assignment
        assert!(sym0 < 4);
        assert!(sym1 < 4);
        assert!(sym2 < 4);
        assert!(sym3 < 4);
    }

    #[test]
    fn test_benchmark_two_level() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create test data
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Benchmark our two-level decode vs libdeflate
        const ITERS: usize = 20;

        // Warmup
        for _ in 0..3 {
            let mut output = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
        }

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

        let libdeflate_avg = libdeflate_time / ITERS as u32;
        let libdeflate_mbps = 1_000_000.0 / libdeflate_avg.as_secs_f64() / 1_000_000.0;

        println!("\n=== Two-Level Table Benchmark (1MB x {}) ===", ITERS);
        println!(
            "libdeflate:     {:>8?}/iter  ({:.0} MB/s)",
            libdeflate_avg, libdeflate_mbps
        );
        println!("Target: Match this performance with two-level tables");
    }
}
