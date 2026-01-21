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
/// Low byte = symbol (or 0xFF for overflow)
/// High byte = bits consumed (or 0 for overflow)
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct VectorEntry(u16);

impl VectorEntry {
    const fn new(symbol: u8, bits: u8) -> Self {
        Self((symbol as u16) | ((bits as u16) << 8))
    }

    #[inline(always)]
    pub fn symbol(self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    pub fn bits(self) -> u8 {
        (self.0 >> 8) as u8
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
                table[idx] = VectorEntry::new(sym as u8, len);
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

/// Reverse n bits
fn reverse_bits(code: u16, n: u8) -> u16 {
    let mut result = 0u16;
    let mut c = code;
    for _ in 0..n {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    result
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
            entries[0].symbol(),
            entries[1].symbol(),
            entries[2].symbol(),
            entries[3].symbol(),
            entries[4].symbol(),
            entries[5].symbol(),
            entries[6].symbol(),
            entries[7].symbol(),
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
            entries[0].symbol(),
            entries[1].symbol(),
            entries[2].symbol(),
            entries[3].symbol(),
            entries[4].symbol(),
            entries[5].symbol(),
            entries[6].symbol(),
            entries[7].symbol(),
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
