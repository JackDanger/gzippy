//! Hyperoptimized Huffman Tables - libdeflate-style packed entries
//!
//! Key insight from libdeflate:
//! - Pack all decode info into a single u32:
//!   - Low 8 bits: total bits to consume (codeword + extra bits)
//!   - If LITERAL flag set: next 8 bits are the literal value
//!   - If MATCH flag set: next bits are length base
//!   - High bits: flags (literal, end-of-block, subtable pointer)
//!
//! This allows: `bitsleft -= entry` (subtracts whole entry, extracting code length from low byte)

#![allow(dead_code)]

/// Huffman entry flags (libdeflate-style)
const HUFFDEC_LITERAL: u32 = 0x8000_0000; // Entry is a literal (value in bits 8-15)
const HUFFDEC_EOB: u32 = 0x4000_0000; // End of block
const HUFFDEC_SUBTABLE: u32 = 0x2000_0000; // Points to subtable

/// Entry format:
/// - Bits 0-7: Code length (bits to consume from bitstream)
/// - Bits 8-16: Symbol value (0-285 for lit/len, 0-31 for dist, 9 bits needed)
/// - Bits 17-20: Extra bits count (for length/distance)
/// - Bits 21-29: Length/distance base value (9 bits, up to 512)
/// - Bit 30: EOB flag
/// - Bit 31: LITERAL flag (1 = literal, 0 = length/eob)
#[derive(Clone, Copy)]
pub struct HyperEntry(u32);

impl HyperEntry {
    pub const INVALID: Self = HyperEntry(0);

    #[inline(always)]
    pub fn new_literal(literal: u8, code_len: u8) -> Self {
        HyperEntry(HUFFDEC_LITERAL | ((literal as u32) << 8) | (code_len as u32))
    }

    #[inline(always)]
    pub fn new_length(symbol: u16, code_len: u8, _extra_bits: u8, _base: u16) -> Self {
        // Store the full symbol (257-285) using 9 bits
        HyperEntry(((symbol as u32) << 8) | (code_len as u32))
    }

    #[inline(always)]
    pub fn new_eob(code_len: u8) -> Self {
        HyperEntry(HUFFDEC_EOB | (code_len as u32))
    }

    #[inline(always)]
    pub fn new_subtable(offset: u16, code_len: u8) -> Self {
        HyperEntry(HUFFDEC_SUBTABLE | ((offset as u32) << 8) | (code_len as u32))
    }

    #[inline(always)]
    pub fn is_literal(self) -> bool {
        (self.0 & HUFFDEC_LITERAL) != 0
    }

    #[inline(always)]
    pub fn is_eob(self) -> bool {
        (self.0 & HUFFDEC_EOB) != 0
    }

    #[inline(always)]
    pub fn is_subtable(self) -> bool {
        (self.0 & HUFFDEC_SUBTABLE) != 0
    }

    #[inline(always)]
    pub fn code_len(self) -> u32 {
        self.0 & 0xFF
    }

    #[inline(always)]
    pub fn literal(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    #[inline(always)]
    pub fn symbol(self) -> u16 {
        // 9 bits for symbol (supports 0-511)
        ((self.0 >> 8) & 0x1FF) as u16
    }

    #[inline(always)]
    pub fn extra_bits(self) -> u32 {
        (self.0 >> 17) & 0xF
    }

    #[inline(always)]
    pub fn base(self) -> u32 {
        (self.0 >> 21) & 0x1FF
    }

    #[inline(always)]
    pub fn subtable_offset(self) -> usize {
        ((self.0 >> 8) & 0xFFFF) as usize
    }
}

/// Primary table: 12 bits (4096 entries)
/// This covers most codes without needing secondary lookup
pub const PRIMARY_BITS: usize = 12;
pub const PRIMARY_SIZE: usize = 1 << PRIMARY_BITS;

/// HyperTable - optimized Huffman decode table
pub struct HyperTable {
    pub primary: [HyperEntry; PRIMARY_SIZE],
    pub secondary: Vec<HyperEntry>,
}

impl Clone for HyperTable {
    fn clone(&self) -> Self {
        Self {
            primary: self.primary,
            secondary: self.secondary.clone(),
        }
    }
}

impl Default for HyperTable {
    fn default() -> Self {
        Self {
            primary: [HyperEntry::INVALID; PRIMARY_SIZE],
            secondary: Vec::new(),
        }
    }
}

impl HyperTable {
    /// Build table from code lengths
    pub fn build(code_lens: &[u8], is_lit_len: bool) -> std::io::Result<Self> {
        use crate::inflate_tables::{LEN_EXTRA_BITS, LEN_START};

        let mut table = Self::default();
        let mut code = 0u32;
        let mut next_code = [0u32; 16];
        let mut bl_count = [0u32; 16];

        // Count codes of each length
        for &len in code_lens {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute first code for each length (canonical Huffman)
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Build primary table
        for (sym, &len) in code_lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;
            let code = next_code[len];
            next_code[len] += 1;

            // Reverse bits for LSB-first
            let reversed = reverse_bits(code, len);

            if len <= PRIMARY_BITS {
                // Fits in primary table - replicate for all suffixes
                let entry = if is_lit_len {
                    if sym < 256 {
                        HyperEntry::new_literal(sym as u8, len as u8)
                    } else if sym == 256 {
                        HyperEntry::new_eob(len as u8)
                    } else {
                        let len_idx = sym - 257;
                        if len_idx < 29 {
                            HyperEntry::new_length(
                                sym as u16,
                                len as u8,
                                LEN_EXTRA_BITS[len_idx],
                                LEN_START[len_idx],
                            )
                        } else {
                            HyperEntry::INVALID
                        }
                    }
                } else {
                    // Distance table
                    HyperEntry::new_length(sym as u16, len as u8, 0, sym as u16)
                };

                let replicate_count = 1 << (PRIMARY_BITS - len);
                for i in 0..replicate_count {
                    let idx = reversed | (i << len);
                    table.primary[idx as usize] = entry;
                }
            } else {
                // Needs secondary table
                // For now, use two-level approach
                let primary_bits = reversed & ((1 << PRIMARY_BITS) - 1);
                let secondary_bits = (reversed >> PRIMARY_BITS) as usize;
                let secondary_len = len - PRIMARY_BITS;

                // Check if we already have a subtable pointer
                let primary_entry = table.primary[primary_bits as usize];
                let subtable_start = if primary_entry.is_subtable() {
                    primary_entry.subtable_offset()
                } else {
                    // Create new subtable
                    let start = table.secondary.len();
                    let size = 1 << (15 - PRIMARY_BITS); // Max 3 more bits
                    table.secondary.resize(start + size, HyperEntry::INVALID);
                    table.primary[primary_bits as usize] =
                        HyperEntry::new_subtable(start as u16, PRIMARY_BITS as u8);
                    start
                };

                // Fill secondary table
                let entry = if is_lit_len {
                    if sym < 256 {
                        HyperEntry::new_literal(sym as u8, len as u8)
                    } else if sym == 256 {
                        HyperEntry::new_eob(len as u8)
                    } else {
                        let len_idx = sym - 257;
                        if len_idx < 29 {
                            HyperEntry::new_length(
                                sym as u16,
                                len as u8,
                                LEN_EXTRA_BITS[len_idx],
                                LEN_START[len_idx],
                            )
                        } else {
                            HyperEntry::INVALID
                        }
                    }
                } else {
                    HyperEntry::new_length(sym as u16, len as u8, 0, sym as u16)
                };

                let replicate_count = 1 << (15 - PRIMARY_BITS - secondary_len);
                for i in 0..replicate_count {
                    let idx = subtable_start + secondary_bits + (i << secondary_len);
                    if idx < table.secondary.len() {
                        table.secondary[idx] = entry;
                    }
                }
            }
        }

        Ok(table)
    }

    /// Decode a symbol from the bit buffer
    #[inline(always)]
    pub fn decode(&self, buf: u64) -> HyperEntry {
        let idx = (buf as usize) & (PRIMARY_SIZE - 1);
        let entry = self.primary[idx];

        if !entry.is_subtable() {
            return entry;
        }

        // Secondary lookup
        let subtable_start = entry.subtable_offset();
        let secondary_idx = (buf >> PRIMARY_BITS) as usize & 7; // Max 3 bits
        self.secondary
            .get(subtable_start + secondary_idx)
            .copied()
            .unwrap_or(HyperEntry::INVALID)
    }
}

/// Reverse bits for canonical Huffman
#[inline]
fn reverse_bits(code: u32, len: usize) -> u32 {
    let mut result = 0u32;
    let mut code = code;
    for _ in 0..len {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyper_entry() {
        let lit = HyperEntry::new_literal(b'A', 8);
        assert!(lit.is_literal());
        assert!(!lit.is_eob());
        assert_eq!(lit.literal(), b'A');
        assert_eq!(lit.code_len(), 8);

        let eob = HyperEntry::new_eob(7);
        assert!(eob.is_eob());
        assert!(!eob.is_literal());
        assert_eq!(eob.code_len(), 7);

        let len = HyperEntry::new_length(257, 7, 0, 3);
        assert!(!len.is_literal());
        assert!(!len.is_eob());
        assert_eq!(len.code_len(), 7);
        assert_eq!(len.symbol(), 257);
    }
}
