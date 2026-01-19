//! Block Finder LUT for rapidgzip-style parallel decompression
//!
//! This module provides a 15-bit lookup table for quickly finding potential
//! deflate block boundaries, plus precode validation to filter false positives.
//!
//! The approach:
//! 1. Scan compressed data bit-by-bit
//! 2. Use LUT to skip invalid patterns (skips 1-15 bits at once)
//! 3. When LUT indicates candidate, validate precode (Huffman tree check)
//! 4. If precode valid, try to decode - if successful, found a block

/// Number of bits in the LUT (15 bits = 32KB table)
pub const LUT_BITS: usize = 15;
pub const LUT_SIZE: usize = 1 << LUT_BITS;

/// Deflate block header structure (first 13 bits):
/// - bit 0: BFINAL (1 = last block)
/// - bits 1-2: BTYPE (0=stored, 1=fixed, 2=dynamic, 3=reserved)
/// - bits 3-7: HLIT-257 (literal/length code count - 257, valid 0-29)
/// - bits 8-12: HDIST-1 (distance code count - 1, valid 0-29)
///
/// Check if a 15-bit pattern could be a valid dynamic Huffman block start
/// We only look for non-final blocks (BFINAL=0) with dynamic Huffman (BTYPE=2)
#[inline]
pub const fn is_deflate_candidate(bits: u32) -> bool {
    let bfinal = bits & 1;
    let btype = (bits >> 1) & 3;
    let hlit = (bits >> 3) & 31;
    let hdist = (bits >> 8) & 31;

    // BFINAL=0 (not final - we want to find interior blocks)
    // BTYPE=2 (dynamic Huffman - most common for compressed data)
    // HLIT <= 29 (valid range)
    // HDIST <= 29 (valid range)
    bfinal == 0 && btype == 2 && hlit <= 29 && hdist <= 29
}

/// For invalid patterns, compute how many bits we can safely skip
/// Returns 1-15 (number of bits to skip)
#[inline]
const fn compute_skip(bits: u32) -> u8 {
    // Try each shift and find the first valid candidate
    let mut shift = 1u8;
    while shift < LUT_BITS as u8 {
        if is_deflate_candidate(bits >> shift) {
            return shift;
        }
        shift += 1;
    }
    LUT_BITS as u8
}

/// Generate the LUT at compile time
/// Entry format:
/// - Negative value (-1 to -15): Valid candidate, value indicates bits consumed for validation
/// - Positive value (1 to 15): Invalid, skip this many bits
/// - Zero: Should not occur
const fn generate_lut() -> [i8; LUT_SIZE] {
    let mut lut = [0i8; LUT_SIZE];
    let mut i = 0usize;
    while i < LUT_SIZE {
        if is_deflate_candidate(i as u32) {
            // Valid candidate - need to check precode next
            // -1 means "this is a candidate at bit 0"
            lut[i] = -1;
        } else {
            // Invalid - compute skip distance
            lut[i] = compute_skip(i as u32) as i8;
        }
        i += 1;
    }
    lut
}

/// The precomputed LUT
pub static DEFLATE_CANDIDATE_LUT: [i8; LUT_SIZE] = generate_lut();

/// Precode alphabet order (RFC 1951)
pub const PRECODE_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Maximum precode count
pub const MAX_PRECODE_COUNT: usize = 19;

/// Bits per precode length
pub const PRECODE_BITS: usize = 3;

/// Bits for HCLEN field
pub const HCLEN_BITS: usize = 4;

/// Read bits from a byte slice (LSB first, like deflate)
#[inline]
pub fn read_bits(data: &[u8], bit_offset: usize, count: usize) -> u64 {
    if count == 0 {
        return 0;
    }

    let byte_offset = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;

    // Read enough bytes to cover the bits we need
    let bytes_needed = (bit_in_byte + count).div_ceil(8);
    if byte_offset + bytes_needed > data.len() {
        return 0;
    }

    let mut result = 0u64;
    let mut bits_read = 0;
    let mut current_byte_offset = byte_offset;
    let mut current_bit = bit_in_byte;

    while bits_read < count {
        if current_byte_offset >= data.len() {
            break;
        }

        let byte = data[current_byte_offset];
        let bits_available = 8 - current_bit;
        let bits_to_read = (count - bits_read).min(bits_available);

        let mask = ((1u64 << bits_to_read) - 1) as u8;
        let value = (byte >> current_bit) & mask;

        result |= (value as u64) << bits_read;

        bits_read += bits_to_read;
        current_byte_offset += 1;
        current_bit = 0;
    }

    result
}

/// Peek 15 bits from data at given bit offset
#[inline]
pub fn peek_bits_15(data: &[u8], bit_offset: usize) -> u32 {
    read_bits(data, bit_offset, 15) as u32
}

/// Validate the precode (Huffman code for code lengths)
/// This checks if the precode forms a valid Huffman tree
///
/// A valid Huffman tree satisfies: sum(2^(max_len - len)) = 2^max_len
/// For deflate, max code length is 7 for precode
pub fn validate_precode(data: &[u8], block_start_bit: usize) -> bool {
    // Bits 0-2: BFINAL + BTYPE (already validated by LUT)
    // Bits 3-7: HLIT (5 bits)
    // Bits 8-12: HDIST (5 bits)
    // Bits 13-16: HCLEN (4 bits) = number of precode lengths - 4

    let hclen_start = block_start_bit + 13;
    let hclen = read_bits(data, hclen_start, HCLEN_BITS) as usize + 4;

    if hclen > MAX_PRECODE_COUNT {
        return false;
    }

    // Read precode lengths (3 bits each)
    let mut precode_lens = [0u8; MAX_PRECODE_COUNT];
    let precode_start = hclen_start + HCLEN_BITS;

    for i in 0..hclen {
        let len = read_bits(data, precode_start + i * PRECODE_BITS, PRECODE_BITS) as u8;
        if len > 7 {
            return false; // Max precode length is 7
        }
        precode_lens[PRECODE_ORDER[i]] = len;
    }

    // Check Huffman tree validity using Kraft inequality
    // For a valid binary Huffman code: sum(2^(-len)) = 1
    // Equivalently: sum(2^(max_len - len)) = 2^max_len
    // For precode, max_len = 7, so we check: sum(2^(7-len)) = 128

    let mut sum = 0u32;
    let mut has_nonzero = false;

    for &len in &precode_lens {
        if len > 0 {
            has_nonzero = true;
            sum += 1 << (7 - len);
        }
    }

    // Empty tree (all zeros) is valid
    // Complete tree: sum == 128
    // Under-full tree is also valid (sum < 128)
    // Over-full tree is invalid (sum > 128)
    !has_nonzero || sum <= 128
}

/// Find the next potential block boundary starting from bit_offset
/// Returns the bit offset of the candidate, or None if not found
pub fn find_next_block_candidate(data: &[u8], start_bit: usize, end_bit: usize) -> Option<usize> {
    let mut bit_pos = start_bit;

    while bit_pos + LUT_BITS <= end_bit && bit_pos / 8 + 2 < data.len() {
        let bits = peek_bits_15(data, bit_pos);
        let skip = DEFLATE_CANDIDATE_LUT[bits as usize];

        if skip < 0 {
            // Candidate found - validate precode
            if validate_precode(data, bit_pos) {
                return Some(bit_pos);
            }
            // Precode invalid, try next bit
            bit_pos += 1;
        } else {
            // Skip forward
            bit_pos += skip as usize;
        }
    }

    None
}

/// Statistics about block finding
#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct BlockFinderStats {
    pub candidates_checked: usize,
    pub precode_validations: usize,
    pub valid_blocks_found: usize,
}

/// Find all potential block boundaries in a range
#[allow(dead_code)]
pub fn find_all_block_candidates(
    data: &[u8],
    start_bit: usize,
    end_bit: usize,
    max_candidates: usize,
) -> (Vec<usize>, BlockFinderStats) {
    let mut candidates = Vec::new();
    let mut stats = BlockFinderStats::default();
    let mut bit_pos = start_bit;

    while bit_pos + LUT_BITS <= end_bit
        && bit_pos / 8 + 2 < data.len()
        && candidates.len() < max_candidates
    {
        let bits = peek_bits_15(data, bit_pos);
        let skip = DEFLATE_CANDIDATE_LUT[bits as usize];
        stats.candidates_checked += 1;

        if skip < 0 {
            stats.precode_validations += 1;
            if validate_precode(data, bit_pos) {
                candidates.push(bit_pos);
                stats.valid_blocks_found += 1;
            }
            bit_pos += 1;
        } else {
            bit_pos += skip as usize;
        }
    }

    (candidates, stats)
}

/// Try to find a valid block start within a search range
/// This is used when we need to find where a chunk should start
pub fn find_block_in_range(
    data: &[u8],
    start_bit: usize,
    search_range_bits: usize,
) -> Option<usize> {
    let end_bit = (start_bit + search_range_bits).min(data.len() * 8);
    find_next_block_candidate(data, start_bit, end_bit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn test_is_deflate_candidate() {
        // BFINAL=0, BTYPE=2 (dynamic), HLIT=0, HDIST=0
        // bits: 0 10 00000 00000 = 0b00000_00000_10_0 = 0x04
        assert!(is_deflate_candidate(0b00000_00000_10_0));

        // BFINAL=1 (final block) - should NOT be candidate
        assert!(!is_deflate_candidate(0b00000_00000_10_1));

        // BTYPE=0 (stored) - should NOT be candidate
        assert!(!is_deflate_candidate(0b00000_00000_00_0));

        // BTYPE=1 (fixed) - should NOT be candidate
        assert!(!is_deflate_candidate(0b00000_00000_01_0));

        // HLIT=30 (invalid, > 29)
        assert!(!is_deflate_candidate(0b00000_11110_10_0));

        // HDIST=30 (invalid, > 29)
        assert!(!is_deflate_candidate(0b11110_00000_10_0));
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings, clippy::unnecessary_cast)]
    fn test_lut_skip_values() {
        // For invalid patterns, should have positive skip
        assert!(DEFLATE_CANDIDATE_LUT[0b00000_00000_00_0 as usize] > 0); // BTYPE=0
        assert!(DEFLATE_CANDIDATE_LUT[0b00000_00000_01_0 as usize] > 0); // BTYPE=1
        assert!(DEFLATE_CANDIDATE_LUT[0b00000_00000_11_0 as usize] > 0); // BTYPE=3

        // For valid candidate, should be -1
        assert_eq!(DEFLATE_CANDIDATE_LUT[0b00000_00000_10_0 as usize], -1);
    }

    #[test]
    fn test_read_bits() {
        let data = [0b10110100, 0b11001010];

        // Read first 4 bits: 0100
        assert_eq!(read_bits(&data, 0, 4), 0b0100);

        // Read bits 4-7: 1011
        assert_eq!(read_bits(&data, 4, 4), 0b1011);

        // Read across byte boundary
        assert_eq!(read_bits(&data, 4, 8), 0b10101011);
    }

    #[test]
    fn test_validate_precode_empty() {
        // Create a minimal valid dynamic block header
        // This is tricky - we need a proper header
        // For now, just test that the function doesn't crash
        let data = [0u8; 100];
        // Will likely return false for all zeros, which is fine
        let _ = validate_precode(&data, 0);
    }
}
