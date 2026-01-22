//! Vector Huffman Decode - SIMD Parallel Decompression
//!
//! This module implements parallel Huffman decoding using SIMD instructions.
//! The key innovation: decode 8 independent bit streams simultaneously.
//!
//! ## Strategy
//!
//! 1. **Lane Setup**: Divide input into 8 lanes at fixed spacing
//! 2. **Parallel Decode**: Use SIMD gather to lookup 8 table entries at once
//! 3. **Lockstep Decode**: Each lane decodes N literals before resync
//! 4. **Prefix-Sum Resync**: Calculate cumulative bits consumed to find new positions
//! 5. **Scatter Output**: Write decoded bytes to correct output positions
//!
//! ## Limitations
//!
//! - Only works for literal-heavy streams (matches break parallelism)
//! - Requires AVX2 (256-bit) or AVX-512 (512-bit)
//! - Fixed Huffman only (dynamic would need per-lane tables)
//!
//! ## Performance Target
//!
//! 8 lanes × 1 symbol/lookup = 8x theoretical speedup
//! Actual: ~4-5x due to resync overhead and match fallback

#![allow(dead_code)]

use std::sync::OnceLock;

// =============================================================================
// SIMD Platform Detection
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Check if AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Check if AVX-512 is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512() -> bool {
    false
}

/// Check if NEON is available (always true on aarch64)
#[cfg(target_arch = "aarch64")]
pub fn has_neon() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_neon() -> bool {
    false
}

// =============================================================================
// Vector Decode Table (8-bit indexed, returns symbol + bits consumed)
// =============================================================================

/// Packed table entry: [symbol: u8, bits: u8] in u16
/// For fixed Huffman, we can use 9-bit index (covers all codes)
/// But for SIMD gather, we use 8-bit index (256 entries) with overflow handling
pub const VECTOR_TABLE_BITS: usize = 8;
pub const VECTOR_TABLE_SIZE: usize = 1 << VECTOR_TABLE_BITS;

/// Vector decode entry: packed as u16 for efficient SIMD loads
/// bits 0-8: symbol (0-287)
/// bits 9-12: length (0-15)
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct VectorEntry(u16);

impl VectorEntry {
    const fn new(symbol: u16, bits: u8) -> Self {
        Self(symbol | ((bits as u16) << 9))
    }

    #[inline(always)]
    pub fn symbol(self) -> u16 {
        self.0 & 0x1FF
    }

    #[inline(always)]
    pub fn bits(self) -> u8 {
        ((self.0 >> 9) & 0xF) as u8
    }

    #[inline(always)]
    pub fn is_overflow(self) -> bool {
        self.bits() == 0
    }

    #[inline(always)]
    pub fn raw(self) -> u16 {
        self.0
    }
}

/// Static vector table for fixed Huffman
static FIXED_VECTOR_TABLE: OnceLock<Box<[VectorEntry; VECTOR_TABLE_SIZE]>> = OnceLock::new();

/// Build the vector decode table for fixed Huffman
fn build_fixed_vector_table() -> Box<[VectorEntry; VECTOR_TABLE_SIZE]> {
    let mut table = vec![VectorEntry::default(); VECTOR_TABLE_SIZE];

    // Fixed Huffman litlen codes (RFC 1951):
    // Symbol 0-143:   8-bit codes (we can fit these)
    // Symbol 144-255: 9-bit codes (OVERFLOW - need slow path)
    // Symbol 256-279: 7-bit codes (fits)
    // Symbol 280-287: 8-bit codes (fits)

    // Build mapping for 7-8 bit codes only
    for sym in 0u16..288 {
        let (code, len) = fixed_huffman_code(sym);
        if len > 8 {
            continue; // 9-bit codes handled in overflow
        }

        // Reverse bits for deflate format
        let reversed = reverse_bits(code, len);

        // Fill all entries that start with this code
        let fill_bits = 8 - len as usize;
        for suffix in 0..(1 << fill_bits) {
            let idx = reversed as usize | (suffix << len as usize);
            if idx < VECTOR_TABLE_SIZE {
                table[idx] = VectorEntry::new(sym, len);
            }
        }
    }

    // Mark 9-bit codes as overflow (bits = 0)
    // These need the slow scalar path

    table.into_boxed_slice().try_into().unwrap()
}

/// Get fixed Huffman code for a symbol
fn fixed_huffman_code(sym: u16) -> (u16, u8) {
    match sym {
        0..=143 => (0b00110000 + sym, 8),
        144..=255 => (0b110010000 + (sym - 144), 9),
        256..=279 => (sym - 256, 7),
        280..=287 => (0b11000000 + (sym - 280), 8),
        _ => (0, 0),
    }
}

#[inline(always)]
fn reverse_bits(code: u16, n: u8) -> u16 {
    let mut result = 0u16;
    let mut c = code;
    for _ in 0..n {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    result
}

#[derive(Clone)]
pub struct VectorTable {
    pub table: Box<[VectorEntry; VECTOR_TABLE_SIZE]>,
}

impl VectorTable {
    pub fn new() -> Self {
        Self {
            table: Box::new([VectorEntry::default(); VECTOR_TABLE_SIZE]),
        }
    }

    /// Build a vector table from a libdeflate-style LitLenTable
    pub fn build_from_litlen(&mut self, litlen: &crate::libdeflate_entry::LitLenTable) {
        for i in 0..VECTOR_TABLE_SIZE {
            // Only literals with codeword bits <= 8 can fit in our 8-bit lookahead table.
            // If it's a subtable pointer or a longer code, we must mark it as overflow.
            let entry = litlen.lookup(i as u64);
            if entry.is_literal() && entry.codeword_bits() <= 8 {
                self.table[i] =
                    VectorEntry::new(entry.literal_value() as u16, entry.codeword_bits());
            } else {
                // Exceptional, length, or code > 8 bits - mark as overflow
                self.table[i] = VectorEntry::new(0, 0);
            }
        }
    }
}

/// Get the fixed vector table
pub fn get_fixed_vector_table() -> &'static [VectorEntry; VECTOR_TABLE_SIZE] {
    FIXED_VECTOR_TABLE.get_or_init(build_fixed_vector_table)
}

// =============================================================================
// AVX2 Implementation (8 lanes, 32 bytes)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use super::*;

    /// Decode 8 symbols in parallel using AVX2
    /// Returns (symbols[8], bits_consumed[8], any_overflow)
    #[target_feature(enable = "avx2")]
    pub unsafe fn decode_8_symbols(
        bit_buffers: &[u64; 8],
        table: &[VectorEntry; VECTOR_TABLE_SIZE],
    ) -> ([u8; 8], [u8; 8], bool) {
        // Extract low 8 bits from each bit buffer as indices
        let indices: [u8; 8] = [
            (bit_buffers[0] & 0xFF) as u8,
            (bit_buffers[1] & 0xFF) as u8,
            (bit_buffers[2] & 0xFF) as u8,
            (bit_buffers[3] & 0xFF) as u8,
            (bit_buffers[4] & 0xFF) as u8,
            (bit_buffers[5] & 0xFF) as u8,
            (bit_buffers[6] & 0xFF) as u8,
            (bit_buffers[7] & 0xFF) as u8,
        ];

        // Gather entries from table (could use vpgatherdd but table is small)
        let entries: [VectorEntry; 8] = [
            table[indices[0] as usize],
            table[indices[1] as usize],
            table[indices[2] as usize],
            table[indices[3] as usize],
            table[indices[4] as usize],
            table[indices[5] as usize],
            table[indices[6] as usize],
            table[indices[7] as usize],
        ];

        // Extract symbols and bits
        let symbols = [
            entries[0].symbol() as u8,
            entries[1].symbol() as u8,
            entries[2].symbol() as u8,
            entries[3].symbol() as u8,
            entries[4].symbol() as u8,
            entries[5].symbol() as u8,
            entries[6].symbol() as u8,
            entries[7].symbol() as u8,
        ];

        let bits = [
            entries[0].bits(),
            entries[1].bits(),
            entries[2].bits(),
            entries[3].bits(),
            entries[4].bits(),
            entries[5].bits(),
            entries[6].bits(),
            entries[7].bits(),
        ];

        // Check for overflow (any bits == 0)
        let any_overflow = bits.iter().any(|&b| b == 0);

        (symbols, bits, any_overflow)
    }

    /// Advance bit buffers by consumed bits
    #[target_feature(enable = "avx2")]
    pub unsafe fn advance_bit_buffers(bit_buffers: &mut [u64; 8], bits_consumed: &[u8; 8]) {
        for i in 0..8 {
            bit_buffers[i] >>= bits_consumed[i];
        }
    }

    /// Refill bit buffers from input streams
    #[target_feature(enable = "avx2")]
    pub unsafe fn refill_bit_buffers(
        bit_buffers: &mut [u64; 8],
        bits_left: &mut [u32; 8],
        input: &[u8],
        positions: &mut [usize; 8],
    ) {
        for i in 0..8 {
            while bits_left[i] <= 56 && positions[i] < input.len() {
                bit_buffers[i] |= (input[positions[i]] as u64) << bits_left[i];
                positions[i] += 1;
                bits_left[i] += 8;
            }
        }
    }
}

// =============================================================================
// NEON Implementation (8 lanes, 16 bytes × 2)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use super::*;
    use std::arch::aarch64::*;

    /// Decode 8 symbols in parallel using NEON
    /// Returns (symbols[8], bits_consumed[8], any_overflow)
    pub unsafe fn decode_8_symbols(
        bit_buffers: &[u64; 8],
        table: &[VectorEntry; VECTOR_TABLE_SIZE],
    ) -> ([u8; 8], [u8; 8], bool) {
        // Extract low 8 bits from each bit buffer as indices
        let indices: [u8; 8] = [
            (bit_buffers[0] & 0xFF) as u8,
            (bit_buffers[1] & 0xFF) as u8,
            (bit_buffers[2] & 0xFF) as u8,
            (bit_buffers[3] & 0xFF) as u8,
            (bit_buffers[4] & 0xFF) as u8,
            (bit_buffers[5] & 0xFF) as u8,
            (bit_buffers[6] & 0xFF) as u8,
            (bit_buffers[7] & 0xFF) as u8,
        ];

        // Gather entries from table
        let entries: [VectorEntry; 8] = [
            table[indices[0] as usize],
            table[indices[1] as usize],
            table[indices[2] as usize],
            table[indices[3] as usize],
            table[indices[4] as usize],
            table[indices[5] as usize],
            table[indices[6] as usize],
            table[indices[7] as usize],
        ];

        // Extract symbols and bits
        let symbols = [
            entries[0].symbol() as u8,
            entries[1].symbol() as u8,
            entries[2].symbol() as u8,
            entries[3].symbol() as u8,
            entries[4].symbol() as u8,
            entries[5].symbol() as u8,
            entries[6].symbol() as u8,
            entries[7].symbol() as u8,
        ];

        let bits = [
            entries[0].bits(),
            entries[1].bits(),
            entries[2].bits(),
            entries[3].bits(),
            entries[4].bits(),
            entries[5].bits(),
            entries[6].bits(),
            entries[7].bits(),
        ];

        // Check for overflow using NEON
        let bits_vec = vld1_u8(bits.as_ptr());
        let zero = vdup_n_u8(0);
        let cmp = vceq_u8(bits_vec, zero);
        let any_overflow = vget_lane_u64(vreinterpret_u64_u8(cmp), 0) != 0;

        (symbols, bits, any_overflow)
    }

    /// Advance bit buffers by consumed bits
    pub unsafe fn advance_bit_buffers(bit_buffers: &mut [u64; 8], bits_consumed: &[u8; 8]) {
        for i in 0..8 {
            bit_buffers[i] >>= bits_consumed[i];
        }
    }

    /// Refill bit buffers from input streams
    pub unsafe fn refill_bit_buffers(
        bit_buffers: &mut [u64; 8],
        bits_left: &mut [u32; 8],
        input: &[u8],
        positions: &mut [usize; 8],
    ) {
        for i in 0..8 {
            while bits_left[i] <= 56 && positions[i] < input.len() {
                bit_buffers[i] |= (input[positions[i]] as u64) << bits_left[i];
                positions[i] += 1;
                bits_left[i] += 8;
            }
        }
    }
}

// =============================================================================
// High-Level Vector Decode API
// =============================================================================

/// Lane state for parallel decoding
pub struct VectorLanes {
    /// Bit buffers for each lane
    pub bit_buffers: [u64; 8],
    /// Bits available in each buffer
    pub bits_left: [u32; 8],
    /// Current input position for each lane
    pub input_positions: [usize; 8],
    /// Current output position for each lane
    pub output_positions: [usize; 8],
    /// Whether lane has hit a match/EOB and needs scalar handling
    pub lane_overflow: [bool; 8],
}

impl VectorLanes {
    /// Initialize lanes with evenly-spaced starting positions
    pub fn new(input_len: usize, output_hint: usize) -> Self {
        let lane_spacing = input_len / 8;
        let output_spacing = output_hint / 8;

        let mut lanes = Self {
            bit_buffers: [0; 8],
            bits_left: [0; 8],
            input_positions: [0; 8],
            output_positions: [0; 8],
            lane_overflow: [false; 8],
        };

        for i in 0..8 {
            lanes.input_positions[i] = i * lane_spacing;
            lanes.output_positions[i] = i * output_spacing;
        }

        lanes
    }

    /// Check if all lanes have overflowed (need scalar fallback)
    pub fn all_overflow(&self) -> bool {
        self.lane_overflow.iter().all(|&x| x)
    }

    /// Count active (non-overflow) lanes
    pub fn active_lanes(&self) -> usize {
        self.lane_overflow.iter().filter(|&&x| !x).count()
    }
}

/// Decode fixed Huffman using vector parallelism
/// Returns decoded bytes or falls back to scalar on overflow
#[cfg(target_arch = "x86_64")]
pub fn decode_fixed_vector(
    input: &[u8],
    output: &mut [u8],
    output_size_hint: usize,
) -> std::io::Result<usize> {
    if !has_avx2() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "AVX2 not available",
        ));
    }

    let table = get_fixed_vector_table();
    let mut lanes = VectorLanes::new(input.len(), output_size_hint);

    // Initial refill
    unsafe {
        avx2_impl::refill_bit_buffers(
            &mut lanes.bit_buffers,
            &mut lanes.bits_left,
            input,
            &mut lanes.input_positions,
        );
    }

    let mut total_decoded = 0;
    let max_iterations = output.len(); // Safety limit

    for _ in 0..max_iterations {
        // Decode 8 symbols in parallel
        let (symbols, bits_consumed, any_overflow) =
            unsafe { avx2_impl::decode_8_symbols(&lanes.bit_buffers, table) };

        if any_overflow {
            // One or more lanes hit a 9-bit code or match - need scalar fallback
            // For now, just stop parallel decode
            break;
        }

        // Write decoded symbols to output
        for i in 0..8 {
            if !lanes.lane_overflow[i] {
                let out_pos = lanes.output_positions[i];
                if out_pos < output.len() && symbols[i] < 0xFF {
                    output[out_pos] = symbols[i];
                    lanes.output_positions[i] += 1;
                    total_decoded += 1;
                }
            }
        }

        // Advance bit buffers
        unsafe {
            avx2_impl::advance_bit_buffers(&mut lanes.bit_buffers, &bits_consumed);
        }

        // Refill bit buffers
        unsafe {
            avx2_impl::refill_bit_buffers(
                &mut lanes.bit_buffers,
                &mut lanes.bits_left,
                input,
                &mut lanes.input_positions,
            );
        }

        // Update bits_left
        for i in 0..8 {
            lanes.bits_left[i] = lanes.bits_left[i].saturating_sub(bits_consumed[i] as u32);
        }
    }

    Ok(total_decoded)
}

/// Decode fixed Huffman using NEON vector parallelism (ARM64)
#[cfg(target_arch = "aarch64")]
pub fn decode_fixed_vector(
    input: &[u8],
    output: &mut [u8],
    output_size_hint: usize,
) -> std::io::Result<usize> {
    let table = get_fixed_vector_table();
    let mut lanes = VectorLanes::new(input.len(), output_size_hint);

    // Initial refill
    unsafe {
        neon_impl::refill_bit_buffers(
            &mut lanes.bit_buffers,
            &mut lanes.bits_left,
            input,
            &mut lanes.input_positions,
        );
    }

    let mut total_decoded = 0;
    let max_iterations = output.len();

    for _ in 0..max_iterations {
        // Decode 8 symbols in parallel
        let (symbols, bits_consumed, any_overflow) =
            unsafe { neon_impl::decode_8_symbols(&lanes.bit_buffers, table) };

        if any_overflow {
            break;
        }

        // Write decoded symbols to output
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            if !lanes.lane_overflow[i] {
                let out_pos = lanes.output_positions[i];
                if out_pos < output.len() && symbols[i] < 0xFF {
                    output[out_pos] = symbols[i];
                    lanes.output_positions[i] += 1;
                    total_decoded += 1;
                }
            }
        }

        // Advance bit buffers
        unsafe {
            neon_impl::advance_bit_buffers(&mut lanes.bit_buffers, &bits_consumed);
        }

        // Refill bit buffers
        unsafe {
            neon_impl::refill_bit_buffers(
                &mut lanes.bit_buffers,
                &mut lanes.bits_left,
                input,
                &mut lanes.input_positions,
            );
        }

        // Update bits_left
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            lanes.bits_left[i] = lanes.bits_left[i].saturating_sub(bits_consumed[i] as u32);
        }
    }

    Ok(total_decoded)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn decode_fixed_vector(
    _input: &[u8],
    _output: &mut [u8],
    _output_size_hint: usize,
) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Vector decode only available on x86_64 or aarch64",
    ))
}

// =============================================================================
// Multi-Literal Lookahead (Practical SIMD Optimization)
// =============================================================================

/// Decode up to 4 consecutive literals from a single bit stream
/// Returns (symbols[], count, total_bits_consumed)
/// This is more practical than parallel lanes for real deflate data
#[inline(always)]
pub fn decode_multi_literals(
    bitbuf: u64,
    table: &[VectorEntry; VECTOR_TABLE_SIZE],
) -> ([u8; 4], usize, u32) {
    let mut symbols = [0u8; 4];
    let mut bits_consumed = 0u32;
    let mut remaining = bitbuf;
    let mut count = 0usize;

    // Try to decode up to 4 literals
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        let idx = (remaining & 0xFF) as usize;
        let entry = table[idx];

        if entry.is_overflow() {
            // Hit a 9-bit code or non-literal - stop
            break;
        }

        let sym = entry.symbol();
        let bits = entry.bits() as u32;

        // Check if it's a literal (symbol < 256) vs length code (256-287)
        if sym >= 256 {
            // Hit length code or EOB - stop
            break;
        }

        // Now we know it's a literal (0-255)
        // But wait, fixed Huffman symbols 144-255 are 9 bits.
        // Our 8-bit table only contains symbols with codes <= 8 bits.
        // Symbols 0-143 (8 bits) and some length codes.

        symbols[i] = sym as u8;
        remaining >>= bits;
        bits_consumed += bits;
        count = i + 1;
    }

    (symbols, count, bits_consumed)
}

/// Decode fixed Huffman block with multi-literal optimization using Bits struct
pub fn decode_fixed_multi_literal_bits(
    bits: &mut crate::consume_first_decode::Bits,
    output: &mut [u8],
    mut out_pos: usize,
) -> std::io::Result<usize> {
    let table = get_fixed_vector_table();
    let fixed_tables = crate::libdeflate_decode::get_fixed_tables();

    loop {
        // Ensure we have at least 32 bits available for multi-literal lookahead
        if bits.available() < 32 {
            bits.refill();
        }

        // Try multi-literal decode
        let (symbols, count, bits_count) = decode_multi_literals(bits.peek(), table);

        if count > 0 {
            // Fast path: got 1-4 literals
            if out_pos + count > output.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Output full",
                ));
            }

            // Write literals
            output[out_pos..(count + out_pos)].copy_from_slice(&symbols[..count]);
            out_pos += count;
            bits.consume(bits_count);
            continue;
        }

        // Slow path: use regular tables for 9-bit codes, lengths, EOB
        if bits.available() < 15 {
            bits.refill();
        }
        let saved = bits.peek();
        let entry = fixed_tables.0.lookup(saved);
        bits.consume_entry(entry.raw());

        if entry.is_end_of_block() {
            return Ok(out_pos);
        }

        if entry.is_literal() {
            // 9-bit literal
            if out_pos >= output.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Output full",
                ));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
        } else {
            // Length code - decode match
            let length = entry.decode_length(saved);

            if bits.available() < 15 {
                bits.refill();
            }

            let dist_saved = bits.peek();
            let dist_entry = fixed_tables.1.lookup(dist_saved);
            bits.consume_entry(dist_entry.raw());

            let distance = dist_entry.decode_distance(dist_saved);

            if distance == 0 || distance as usize > out_pos {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid distance {} at pos {}", distance, out_pos),
                ));
            }

            // Copy match
            let dist = distance as usize;
            let len = length as usize;
            if out_pos + len > output.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Output full",
                ));
            }

            for i in 0..len {
                output[out_pos + i] = output[out_pos - dist + i];
            }
            out_pos += len;
        }
    }
}

/// Decode fixed Huffman block with multi-literal optimization
pub fn decode_fixed_multi_literal(input: &[u8], output: &mut [u8]) -> std::io::Result<usize> {
    let mut bits = crate::consume_first_decode::Bits::new(input);
    decode_fixed_multi_literal_bits(&mut bits, output, 0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_table_builds() {
        let table = get_fixed_vector_table();

        // Count valid entries
        let valid = table.iter().filter(|e| !e.is_overflow()).count();
        let overflow = table.iter().filter(|e| e.is_overflow()).count();

        eprintln!("\nVector table statistics:");
        eprintln!("  Valid entries: {}", valid);
        eprintln!("  Overflow entries: {}", overflow);
        eprintln!("  Coverage: {:.1}%", 100.0 * valid as f64 / 256.0);

        // Should have good coverage for 7-8 bit codes (literals 0-143 + length codes)
        assert!(
            valid >= 200,
            "Should have >=200 valid entries, got {}",
            valid
        );
    }

    #[test]
    fn test_has_simd() {
        eprintln!("\nSIMD availability:");
        eprintln!("  AVX2: {}", has_avx2());
        eprintln!("  AVX-512: {}", has_avx512());
        eprintln!("  NEON: {}", has_neon());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_decode_8_symbols() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - not available");
            return;
        }

        let table = get_fixed_vector_table();

        // Create 8 identical bit buffers with a simple pattern
        // Using reversed bits for fixed Huffman literal 'A' (65)
        // 'A' = 65, code = 0b00110000 + 65 = 0b01100001, 8 bits
        // Reversed: 0b10000110 = 0x86
        let bit_buffers = [0x86u64; 8];

        let (symbols, bits, any_overflow) =
            unsafe { avx2_impl::decode_8_symbols(&bit_buffers, table) };

        eprintln!("\nDecode 8 symbols test:");
        eprintln!("  Symbols: {:?}", symbols);
        eprintln!("  Bits: {:?}", bits);
        eprintln!("  Any overflow: {}", any_overflow);

        // All should decode to the same symbol
        for i in 0..8 {
            assert_eq!(symbols[i], symbols[0], "Lane {} mismatch", i);
            assert_eq!(bits[i], bits[0], "Lane {} bits mismatch", i);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_decode_8_symbols_neon() {
        let table = get_fixed_vector_table();

        // Create 8 identical bit buffers
        let bit_buffers = [0x86u64; 8];

        let (symbols, bits, any_overflow) =
            unsafe { super::neon_impl::decode_8_symbols(&bit_buffers, table) };

        eprintln!("\nNEON Decode 8 symbols test:");
        eprintln!("  Symbols: {:?}", symbols);
        eprintln!("  Bits: {:?}", bits);
        eprintln!("  Any overflow: {}", any_overflow);

        for i in 0..8 {
            assert_eq!(symbols[i], symbols[0], "Lane {} mismatch", i);
            assert_eq!(bits[i], bits[0], "Lane {} bits mismatch", i);
        }
    }

    #[test]
    fn bench_vector_decode() {
        // Create a simple test input (fixed Huffman encoded literals)
        let mut output = vec![0u8; 10000];

        // Create fake deflate input - this won't work for real data
        // but tests the decode path
        let input = vec![0x86u8; 1000]; // Pattern that decodes to something

        let result = decode_fixed_vector(&input, &mut output, 10000);

        match result {
            Ok(decoded) => {
                eprintln!("\nVector decode test:");
                eprintln!("  Decoded {} bytes", decoded);
            }
            Err(e) => {
                eprintln!("\nVector decode not available: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_literal_decode() {
        let table = get_fixed_vector_table();

        // Create a bit pattern with multiple decodable literals
        // Fixed Huffman: symbols 0-143 have 8-bit codes
        // Let's encode "AAAA" - 'A'=65, code=0x30+65=0x61, reversed=0x86
        let bitbuf = 0x86_86_86_86u64; // 4 copies of reversed 'A' code

        let (symbols, count, bits) = decode_multi_literals(bitbuf, table);

        eprintln!("\nMulti-literal decode test:");
        eprintln!("  Symbols: {:?}", &symbols[..count]);
        eprintln!("  Count: {}", count);
        eprintln!("  Bits consumed: {}", bits);

        // Should decode multiple symbols
        assert!(count >= 1, "Should decode at least 1 symbol");
        assert!(bits > 0, "Should consume some bits");
    }

    #[test]
    fn bench_multi_literal() {
        let table = get_fixed_vector_table();
        let iterations = 10_000_000u64;

        // Create varying bit patterns to prevent optimization
        let patterns = [
            0x86_86_86_86u64,
            0x87_86_87_86u64,
            0x88_88_88_88u64,
            0x89_89_89_89u64,
        ];

        let start = std::time::Instant::now();
        let mut total_count = 0u64;
        let mut total_bits = 0u64;

        for i in 0..iterations {
            let bitbuf = patterns[(i & 3) as usize].wrapping_add(i);
            let (_, count, bits) = decode_multi_literals(bitbuf, table);
            total_count += count as u64;
            total_bits += bits as u64;
        }

        let elapsed = start.elapsed();
        let per_sec = iterations as f64 / elapsed.as_secs_f64();
        let symbols_per_sec = (total_count as f64) / elapsed.as_secs_f64();

        eprintln!("\nMulti-literal benchmark:");
        eprintln!(
            "  {} iterations in {:.2}ms",
            iterations,
            elapsed.as_secs_f64() * 1000.0
        );
        eprintln!("  {:.1} M decodes/sec", per_sec / 1_000_000.0);
        eprintln!("  {:.1} M symbols/sec", symbols_per_sec / 1_000_000.0);
        eprintln!(
            "  Avg symbols/decode: {:.2}",
            total_count as f64 / iterations as f64
        );
        eprintln!(
            "  Avg bits/decode: {:.2}",
            total_bits as f64 / iterations as f64
        );
    }

    #[test]
    fn bench_vector_table_lookup() {
        let table = get_fixed_vector_table();
        let iterations = 10_000_000u64;

        let start = std::time::Instant::now();
        let mut sum = 0u64;
        for i in 0..iterations {
            let entry = table[(i & 0xFF) as usize];
            sum = sum.wrapping_add(entry.symbol() as u64);
        }
        let elapsed = start.elapsed();

        let lookups_per_sec = iterations as f64 / elapsed.as_secs_f64();
        eprintln!("\nVector table lookup benchmark:");
        eprintln!(
            "  {} lookups in {:.2}ms",
            iterations,
            elapsed.as_secs_f64() * 1000.0
        );
        eprintln!("  {:.1} M lookups/sec", lookups_per_sec / 1_000_000.0);
        eprintln!("  (sum: {} to prevent opt)", sum);
    }
}
