//! Combined Length+Distance Lookup Table
//!
//! Work in progress - not yet integrated into decode loop

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::manual_range_contains)]
//!
//! This is rapidgzip's key optimization: pre-compute entire LZ77 matches
//! (length + distance) in a single lookup table entry.
//!
//! CacheEntry Layout (4 bytes):
//!   - bits_to_skip: u8 - total bits consumed
//!   - symbol_or_length: u8 - literal value OR (length - 3)
//!   - distance: u16 - actual distance OR special marker
//!
//! Special distance values:
//!   - 0: literal byte (symbol_or_length is the value)
//!   - 0xFFFE: length code needs slow path
//!   - 0xFFFF: END_OF_BLOCK

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};
use std::io;

/// LUT bits for the combined table
pub const COMBINED_LUT_BITS: usize = 12;
pub const COMBINED_LUT_SIZE: usize = 1 << COMBINED_LUT_BITS;
pub const COMBINED_LUT_MASK: u64 = (COMBINED_LUT_SIZE - 1) as u64;

/// Special distance values
pub const DIST_LITERAL: u16 = 0;
pub const DIST_SLOW_PATH: u16 = 0xFFFE;
pub const DIST_END_OF_BLOCK: u16 = 0xFFFF;

/// Cache entry for combined decode
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct CacheEntry {
    pub bits_to_skip: u8,
    pub symbol_or_length: u8,
    pub distance: u16,
}

impl CacheEntry {
    #[inline(always)]
    pub fn literal(bits: u8, byte: u8) -> Self {
        Self {
            bits_to_skip: bits,
            symbol_or_length: byte,
            distance: DIST_LITERAL,
        }
    }

    #[inline(always)]
    pub fn end_of_block(bits: u8) -> Self {
        Self {
            bits_to_skip: bits,
            symbol_or_length: 0,
            distance: DIST_END_OF_BLOCK,
        }
    }

    #[inline(always)]
    pub fn slow_path(bits: u8, length_code: u8) -> Self {
        Self {
            bits_to_skip: bits,
            symbol_or_length: length_code,
            distance: DIST_SLOW_PATH,
        }
    }

    #[inline(always)]
    pub fn lz77(bits: u8, length_minus_3: u8, distance: u16) -> Self {
        Self {
            bits_to_skip: bits,
            symbol_or_length: length_minus_3,
            distance,
        }
    }

    #[inline(always)]
    pub fn is_literal(&self) -> bool {
        self.distance == DIST_LITERAL
    }

    #[inline(always)]
    pub fn is_end_of_block(&self) -> bool {
        self.distance == DIST_END_OF_BLOCK
    }

    #[inline(always)]
    pub fn is_slow_path(&self) -> bool {
        self.distance == DIST_SLOW_PATH
    }

    #[inline(always)]
    pub fn length(&self) -> usize {
        self.symbol_or_length as usize + 3
    }
}

/// Combined length+distance decode table
pub struct CombinedLUT {
    pub table: Box<[CacheEntry; COMBINED_LUT_SIZE]>,
}

impl CombinedLUT {
    /// Build a combined LUT from literal/length and distance code lengths
    pub fn build(lit_len_lens: &[u8], dist_lens: &[u8]) -> io::Result<Self> {
        let mut table = Box::new([CacheEntry::default(); COMBINED_LUT_SIZE]);

        // First, build a simple distance decode table for reference
        let dist_table = build_distance_table(dist_lens)?;

        // Build the lit/len table entries
        let (codes, code_lens) = build_huffman_codes(lit_len_lens)?;

        for (symbol, &code_len) in code_lens.iter().enumerate() {
            if code_len == 0 || code_len > COMBINED_LUT_BITS as u8 {
                continue;
            }

            let code = codes[symbol];
            let reversed_code = reverse_bits(code, code_len);

            if symbol < 256 {
                // Literal: just store it
                insert_entry(
                    &mut table,
                    reversed_code,
                    code_len,
                    CacheEntry::literal(code_len, symbol as u8),
                );
            } else if symbol == 256 {
                // End of block
                insert_entry(
                    &mut table,
                    reversed_code,
                    code_len,
                    CacheEntry::end_of_block(code_len),
                );
            } else if symbol <= 285 {
                // Length code - try to pre-compute with distance
                let len_idx = symbol - 257;
                let len_extra = LEN_EXTRA_BITS[len_idx] as u8;

                // Check if we can fit length + length_extra + distance in the table
                let bits_for_length = code_len + len_extra;

                if bits_for_length + 1 > COMBINED_LUT_BITS as u8 {
                    // Can't fit, use slow path
                    insert_entry(
                        &mut table,
                        reversed_code,
                        code_len,
                        CacheEntry::slow_path(code_len, (symbol - 257) as u8),
                    );
                    continue;
                }

                // Enumerate all possible extra length bits
                let num_len_extras = 1u32 << len_extra;
                for len_extra_val in 0..num_len_extras {
                    let length = LEN_START[len_idx] as usize + len_extra_val as usize;

                    // Try to fit distance decodes
                    let combined_code = reversed_code | ((len_extra_val as u16) << code_len);
                    let remaining_bits = COMBINED_LUT_BITS as u8 - bits_for_length;

                    insert_with_distance(
                        &mut table,
                        combined_code,
                        bits_for_length,
                        length,
                        remaining_bits,
                        &dist_table,
                    );
                }
            }
        }

        Ok(Self { table })
    }

    #[inline(always)]
    pub fn decode(&self, bits: u64) -> CacheEntry {
        self.table[(bits & COMBINED_LUT_MASK) as usize]
    }
}

/// Simple distance decode entry
#[derive(Clone, Copy, Default)]
struct DistEntry {
    code_len: u8,
    symbol: u8,
}

fn build_distance_table(dist_lens: &[u8]) -> io::Result<[DistEntry; 32768]> {
    let mut table = [DistEntry::default(); 32768];

    let (codes, code_lens) = build_huffman_codes(dist_lens)?;

    for (symbol, &code_len) in code_lens.iter().enumerate() {
        if code_len == 0 || code_len > 15 {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        // Fill all slots that match this prefix
        let fill_count = 1u32 << (15 - code_len);
        for i in 0..fill_count {
            let idx = (reversed as usize) | ((i as usize) << code_len);
            if idx < 32768 {
                table[idx] = DistEntry {
                    code_len,
                    symbol: symbol as u8,
                };
            }
        }
    }

    Ok(table)
}

fn insert_with_distance(
    table: &mut [CacheEntry; COMBINED_LUT_SIZE],
    base_code: u16,
    bits_used: u8,
    length: usize,
    remaining_bits: u8,
    dist_table: &[DistEntry; 32768],
) {
    // If length - 3 doesn't fit in u8, use slow path
    if length < 3 || length > 258 {
        insert_entry(
            table,
            base_code,
            bits_used,
            CacheEntry::slow_path(bits_used, 0),
        );
        return;
    }
    let length_minus_3 = (length - 3) as u8;

    // Enumerate all possible bit patterns in remaining bits
    let num_patterns = 1u32 << remaining_bits;

    for dist_pattern in 0..num_patterns {
        let full_code = base_code | ((dist_pattern as u16) << bits_used);
        let idx = full_code as usize;
        if idx >= COMBINED_LUT_SIZE {
            continue;
        }

        // Look up distance from this pattern
        let dist_entry = dist_table[(dist_pattern as usize) & 0x7FFF];
        if dist_entry.code_len == 0 {
            // Invalid distance code - mark as slow path
            table[idx] = CacheEntry::slow_path(bits_used, length_minus_3);
            continue;
        }

        let dist_sym = dist_entry.symbol as usize;
        if dist_sym >= 30 {
            table[idx] = CacheEntry::slow_path(bits_used, length_minus_3);
            continue;
        }

        let dist_extra = DIST_EXTRA_BITS[dist_sym] as u8;
        let total_bits = bits_used + dist_entry.code_len + dist_extra;

        if total_bits > COMBINED_LUT_BITS as u8 {
            // Doesn't fit - slow path
            table[idx] = CacheEntry::slow_path(bits_used, length_minus_3);
            continue;
        }

        // Extract distance extra bits
        let extra_shift = dist_entry.code_len;
        let extra_mask = (1u32 << dist_extra) - 1;
        let dist_extra_val = ((dist_pattern >> extra_shift) as u32) & extra_mask;

        let distance = DIST_START[dist_sym] as u32 + dist_extra_val;

        if distance == 0 || distance > 32768 {
            table[idx] = CacheEntry::slow_path(bits_used, length_minus_3);
            continue;
        }

        // Success! Store the full LZ77 entry
        table[idx] = CacheEntry::lz77(total_bits, length_minus_3, distance as u16);
    }
}

fn insert_entry(
    table: &mut [CacheEntry; COMBINED_LUT_SIZE],
    reversed_code: u16,
    code_len: u8,
    entry: CacheEntry,
) {
    let filler_bits = COMBINED_LUT_BITS as u8 - code_len;
    let num_slots = 1u32 << filler_bits;

    for i in 0..num_slots {
        let idx = (reversed_code as usize) | ((i as usize) << code_len);
        if idx < COMBINED_LUT_SIZE {
            table[idx] = entry;
        }
    }
}

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut result = 0u16;
    let mut code = code;
    for _ in 0..len {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

fn build_huffman_codes(lens: &[u8]) -> io::Result<(Vec<u16>, Vec<u8>)> {
    let max_code_len = *lens.iter().max().unwrap_or(&0) as usize;
    if max_code_len > 15 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Code too long"));
    }

    // Count codes of each length
    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // Find the numerical value of the smallest code for each code length
    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..=max_code_len {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    let mut codes = vec![0u16; lens.len()];
    for (n, &len) in lens.iter().enumerate() {
        if len > 0 {
            codes[n] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }

    Ok((codes, lens.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_lut_build() {
        // Build fixed Huffman tables
        let mut lit_len_lens = vec![0u8; 288];
        for i in 0..144 {
            lit_len_lens[i] = 8;
        }
        for i in 144..256 {
            lit_len_lens[i] = 9;
        }
        for i in 256..280 {
            lit_len_lens[i] = 7;
        }
        for i in 280..288 {
            lit_len_lens[i] = 8;
        }

        let dist_lens = vec![5u8; 32];

        let lut = CombinedLUT::build(&lit_len_lens, &dist_lens).unwrap();

        // Check that we have some literal entries
        let mut literal_count = 0;
        let mut lz77_count = 0;
        let mut slow_count = 0;

        for entry in lut.table.iter() {
            if entry.bits_to_skip > 0 {
                if entry.is_literal() {
                    literal_count += 1;
                } else if entry.is_slow_path() {
                    slow_count += 1;
                } else if entry.distance > 0 && entry.distance < 0xFFFE {
                    lz77_count += 1;
                }
            }
        }

        eprintln!(
            "Literals: {}, LZ77: {}, Slow: {}",
            literal_count, lz77_count, slow_count
        );
        assert!(literal_count > 0, "Should have literal entries");
    }
}
