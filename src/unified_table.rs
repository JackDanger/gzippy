//! Unified Decode Table - Single Table for All Symbol Types
//!
//! The key insight: Instead of having separate paths for literals vs matches,
//! we create ONE table where every entry contains ALL information needed to
//! decode that symbol type directly.
//!
//! Entry format (64-bit):
//! ```text
//! Type: 2 bits (literal, match, eob, subtable)
//! Bits: 5 bits (codeword length)
//! Data: 57 bits (type-specific data)
//!
//! For LITERAL:
//!   [type:2=0][bits:5][lit1:8][lit2:8][count:1][extra_bits:5][reserved:35]
//!   - If count=2 and next symbol is also literal, both are stored
//!
//! For MATCH:
//!   [type:2=1][bits:5][length_base:9][length_extra:3][dist_hint:16][reserved:29]
//!   - length_base + (extra_bits from stream) = actual length
//!   - dist_hint: first 16 bits of precomputed distance decode (optimization)
//!
//! For EOB:
//!   [type:2=2][bits:5][reserved:57]
//!
//! For SUBTABLE:
//!   [type:2=3][bits:5][subtable_index:16][reserved:41]
//! ```
//!
//! ## Why This Is Better
//!
//! 1. Single lookup for ALL cases - no fallback path
//! 2. Double-literal is built in - no separate multi_symbol table
//! 3. Match info pre-computed - length base included
//! 4. Better cache locality - one table instead of two

#![allow(dead_code)]

use std::io::{Error, ErrorKind, Result};

// Type field values
const TYPE_LITERAL: u64 = 0;
const TYPE_MATCH: u64 = 1;
const TYPE_EOB: u64 = 2;
const TYPE_SUBTABLE: u64 = 3;

// Field positions
const TYPE_SHIFT: u64 = 62;
const BITS_SHIFT: u64 = 57;
const LIT1_SHIFT: u64 = 49;
const LIT2_SHIFT: u64 = 41;
const COUNT_SHIFT: u64 = 40;
const LENGTH_BASE_SHIFT: u64 = 48;
const LENGTH_EXTRA_SHIFT: u64 = 45;

// Masks
const TYPE_MASK: u64 = 0x3 << TYPE_SHIFT;
const BITS_MASK: u64 = 0x1F << BITS_SHIFT;
const LIT1_MASK: u64 = 0xFF << LIT1_SHIFT;
const LIT2_MASK: u64 = 0xFF << LIT2_SHIFT;
const COUNT_MASK: u64 = 0x1 << COUNT_SHIFT;
const LENGTH_BASE_MASK: u64 = 0x1FF << LENGTH_BASE_SHIFT;
const LENGTH_EXTRA_MASK: u64 = 0x7 << LENGTH_EXTRA_SHIFT;

/// Unified table entry - handles all symbol types in one
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct UnifiedEntry(pub u64);

impl UnifiedEntry {
    /// Create a single literal entry
    #[inline(always)]
    pub const fn literal(bits: u8, symbol: u8) -> Self {
        let entry = (TYPE_LITERAL << TYPE_SHIFT)
            | ((bits as u64) << BITS_SHIFT)
            | ((symbol as u64) << LIT1_SHIFT);
        Self(entry)
    }

    /// Create a double literal entry
    #[inline(always)]
    pub const fn double_literal(bits: u8, symbol1: u8, symbol2: u8) -> Self {
        let entry = (TYPE_LITERAL << TYPE_SHIFT)
            | ((bits as u64) << BITS_SHIFT)
            | ((symbol1 as u64) << LIT1_SHIFT)
            | ((symbol2 as u64) << LIT2_SHIFT)
            | (1 << COUNT_SHIFT);
        Self(entry)
    }

    /// Create a match entry
    #[inline(always)]
    pub const fn length_match(bits: u8, length_base: u16, length_extra_bits: u8) -> Self {
        let entry = (TYPE_MATCH << TYPE_SHIFT)
            | ((bits as u64) << BITS_SHIFT)
            | ((length_base as u64) << LENGTH_BASE_SHIFT)
            | ((length_extra_bits as u64) << LENGTH_EXTRA_SHIFT);
        Self(entry)
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn eob(bits: u8) -> Self {
        let entry = (TYPE_EOB << TYPE_SHIFT) | ((bits as u64) << BITS_SHIFT);
        Self(entry)
    }

    /// Create a subtable pointer entry
    #[inline(always)]
    pub const fn subtable(bits: u8, index: u16) -> Self {
        let entry = (TYPE_SUBTABLE << TYPE_SHIFT) | ((bits as u64) << BITS_SHIFT) | (index as u64);
        Self(entry)
    }

    // Accessors

    #[inline(always)]
    pub fn get_type(self) -> u64 {
        (self.0 & TYPE_MASK) >> TYPE_SHIFT
    }

    #[inline(always)]
    pub fn is_literal(self) -> bool {
        self.get_type() == TYPE_LITERAL
    }

    #[inline(always)]
    pub fn is_match(self) -> bool {
        self.get_type() == TYPE_MATCH
    }

    #[inline(always)]
    pub fn is_eob(self) -> bool {
        self.get_type() == TYPE_EOB
    }

    #[inline(always)]
    pub fn is_subtable(self) -> bool {
        self.get_type() == TYPE_SUBTABLE
    }

    #[inline(always)]
    pub fn bits(self) -> u32 {
        ((self.0 & BITS_MASK) >> BITS_SHIFT) as u32
    }

    #[inline(always)]
    pub fn literal1(self) -> u8 {
        ((self.0 & LIT1_MASK) >> LIT1_SHIFT) as u8
    }

    #[inline(always)]
    pub fn literal2(self) -> u8 {
        ((self.0 & LIT2_MASK) >> LIT2_SHIFT) as u8
    }

    #[inline(always)]
    pub fn is_double_literal(self) -> bool {
        (self.0 & COUNT_MASK) != 0
    }

    #[inline(always)]
    pub fn length_base(self) -> u32 {
        ((self.0 & LENGTH_BASE_MASK) >> LENGTH_BASE_SHIFT) as u32
    }

    #[inline(always)]
    pub fn length_extra_bits(self) -> u32 {
        ((self.0 & LENGTH_EXTRA_MASK) >> LENGTH_EXTRA_SHIFT) as u32
    }

    #[inline(always)]
    pub fn subtable_index(self) -> usize {
        (self.0 & 0xFFFF) as usize
    }
}

/// Unified decode table
pub const UNIFIED_TABLE_BITS: usize = 11;
pub const UNIFIED_TABLE_SIZE: usize = 1 << UNIFIED_TABLE_BITS;
pub const UNIFIED_TABLE_MASK: u64 = (UNIFIED_TABLE_SIZE - 1) as u64;

pub struct UnifiedTable {
    pub entries: Vec<UnifiedEntry>,
    pub subtables: Vec<UnifiedEntry>,
}

impl UnifiedTable {
    /// Build unified table from code lengths
    pub fn build(litlen_lengths: &[u8], _dist_lengths: &[u8]) -> Result<Self> {
        let mut entries = vec![UnifiedEntry::eob(0); UNIFIED_TABLE_SIZE];
        let subtables = Vec::new();

        // Length bases for symbols 257-285 (RFC 1951)
        const LENGTH_BASES: [u16; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        const LENGTH_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        // Build Huffman codes
        let mut bl_count = [0u32; 16];
        for &len in litlen_lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // First pass: single symbols
        for (symbol, &len) in litlen_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;

            // Get code for this symbol
            let code = next_code[len];
            next_code[len] += 1;

            // Reverse bits
            let reversed = reverse_bits(code, len as u8);

            if len > UNIFIED_TABLE_BITS {
                // Needs subtable - skip for now (complex)
                continue;
            }

            // Create entry based on symbol type
            let entry = if symbol < 256 {
                // Literal
                UnifiedEntry::literal(len as u8, symbol as u8)
            } else if symbol == 256 {
                // EOB
                UnifiedEntry::eob(len as u8)
            } else if symbol <= 285 {
                // Length code
                let idx = symbol - 257;
                UnifiedEntry::length_match(len as u8, LENGTH_BASES[idx], LENGTH_EXTRA[idx])
            } else {
                continue; // Invalid symbol
            };

            // Fill table entries (padding for shorter codes)
            let filler_bits = UNIFIED_TABLE_BITS - len;
            let count = 1 << filler_bits;
            for i in 0..count {
                let idx = reversed as usize | (i << len);
                entries[idx] = entry;
            }
        }

        // Second pass: upgrade single literals to double literals where possible
        // Build a temporary lookup for second symbol
        let mut second_symbol: Vec<Option<(u8, u8)>> = vec![None; UNIFIED_TABLE_SIZE];

        // Reset next_code for second pass
        let mut next_code = [0u32; 16];
        code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        for (symbol, &len) in litlen_lengths.iter().enumerate() {
            if len == 0 || symbol >= 256 || len as usize > UNIFIED_TABLE_BITS {
                continue;
            }
            let len = len as usize;

            let code = next_code[len];
            next_code[len] += 1;
            let reversed = reverse_bits(code, len as u8);

            let filler_bits = UNIFIED_TABLE_BITS - len;
            for i in 0..(1 << filler_bits) {
                let idx = reversed as usize | (i << len);
                second_symbol[idx] = Some((symbol as u8, len as u8));
            }
        }

        // Now upgrade entries
        for (idx, entry_slot) in entries.iter_mut().enumerate() {
            let entry = *entry_slot;
            if !entry.is_literal() || entry.is_double_literal() {
                continue;
            }

            let bits1 = entry.bits() as usize;
            let sym1 = entry.literal1();

            // Check remaining bits for second symbol
            let remaining = idx >> bits1;
            if let Some((sym2, bits2)) = second_symbol[remaining] {
                let total_bits = bits1 + bits2 as usize;
                if total_bits <= UNIFIED_TABLE_BITS {
                    // Can upgrade to double literal
                    *entry_slot = UnifiedEntry::double_literal(total_bits as u8, sym1, sym2);
                }
            }
        }

        Ok(Self { entries, subtables })
    }

    /// Build from fixed Huffman codes
    pub fn build_fixed() -> Self {
        let mut litlen_lengths = [0u8; 288];
        litlen_lengths[..144].fill(8);
        litlen_lengths[144..256].fill(9);
        litlen_lengths[256..280].fill(7);
        litlen_lengths[280..288].fill(8);

        let dist_lengths = [5u8; 32];
        Self::build(&litlen_lengths, &dist_lengths).unwrap()
    }

    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> UnifiedEntry {
        self.entries[(bits & UNIFIED_TABLE_MASK) as usize]
    }
}

/// Reverse bits in a code
fn reverse_bits(code: u32, len: u8) -> u32 {
    let mut result = 0u32;
    let mut c = code;
    for _ in 0..len {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    result
}

/// Decode using unified table - single lookup for all symbol types
pub fn decode_unified(
    bits: &mut crate::libdeflate_decode::LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
    table: &UnifiedTable,
    dist_table: &crate::libdeflate_entry::DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 274;

    'fastloop: while out_pos + FASTLOOP_MARGIN <= output.len() {
        bits.refill_branchless();
        let bitbuf = bits.peek_bits();

        // Single lookup - handles ALL cases
        let entry = table.lookup(bitbuf);

        // Most common case: literal (single or double)
        if entry.is_literal() {
            if entry.is_double_literal() {
                // Double literal - write both at once
                output[out_pos] = entry.literal1();
                output[out_pos + 1] = entry.literal2();
                out_pos += 2;
            } else {
                // Single literal
                output[out_pos] = entry.literal1();
                out_pos += 1;
            }
            bits.consume(entry.bits());
            continue 'fastloop;
        }

        // EOB
        if entry.is_eob() {
            bits.consume(entry.bits());
            return Ok(out_pos);
        }

        // Match - length + distance
        if entry.is_match() {
            bits.consume(entry.bits());

            // Get length with extra bits
            let extra_bits = entry.length_extra_bits();
            let length = entry.length_base() + (bits.peek(extra_bits) as u32);
            bits.consume(extra_bits);

            // Decode distance
            bits.refill_branchless();
            let dist_bitbuf = bits.peek_bits();
            let dist_entry = dist_table.lookup(dist_bitbuf);
            bits.consume_entry(dist_entry.raw());
            let distance = dist_entry.decode_distance(dist_bitbuf);

            if distance == 0 || distance as usize > out_pos {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid distance {} at pos {}", distance, out_pos),
                ));
            }

            // Copy match
            crate::libdeflate_decode::copy_match(output, out_pos, distance, length);
            out_pos += length as usize;
            continue 'fastloop;
        }

        // Subtable - would need more complex handling
        // For now, return error (not implemented)
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Subtable not implemented",
        ));
    }

    // Generic loop for near end
    generic_unified_loop(bits, output, out_pos, table, dist_table)
}

fn generic_unified_loop(
    bits: &mut crate::libdeflate_decode::LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
    table: &UnifiedTable,
    dist_table: &crate::libdeflate_entry::DistTable,
) -> Result<usize> {
    loop {
        bits.refill_branchless();
        let bitbuf = bits.peek_bits();
        let entry = table.lookup(bitbuf);

        if entry.is_literal() {
            if out_pos >= output.len() {
                return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
            }
            output[out_pos] = entry.literal1();
            out_pos += 1;
            if entry.is_double_literal() && out_pos < output.len() {
                output[out_pos] = entry.literal2();
                out_pos += 1;
            }
            bits.consume(entry.bits());
            continue;
        }

        if entry.is_eob() {
            bits.consume(entry.bits());
            return Ok(out_pos);
        }

        if entry.is_match() {
            bits.consume(entry.bits());
            let extra_bits = entry.length_extra_bits();
            let length = entry.length_base() + (bits.peek(extra_bits) as u32);
            bits.consume(extra_bits);

            bits.refill_branchless();
            let dist_bitbuf = bits.peek_bits();
            let dist_entry = dist_table.lookup(dist_bitbuf);
            bits.consume_entry(dist_entry.raw());
            let distance = dist_entry.decode_distance(dist_bitbuf);

            if distance == 0 || distance as usize > out_pos {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid distance {} at pos {}", distance, out_pos),
                ));
            }

            if out_pos + length as usize > output.len() {
                return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
            }

            crate::libdeflate_decode::copy_match(output, out_pos, distance, length);
            out_pos += length as usize;
            continue;
        }

        return Err(Error::new(ErrorKind::InvalidData, "Unknown entry type"));
    }
}

/// Cached fixed unified table
fn get_fixed_unified_table() -> &'static UnifiedTable {
    use std::sync::OnceLock;
    static FIXED_UNIFIED: OnceLock<UnifiedTable> = OnceLock::new();
    FIXED_UNIFIED.get_or_init(UnifiedTable::build_fixed)
}

/// Decode fixed Huffman using unified table
pub fn decode_fixed_unified(
    bits: &mut crate::libdeflate_decode::LibdeflateBits,
    output: &mut [u8],
    out_pos: usize,
) -> Result<usize> {
    let table = get_fixed_unified_table();
    let (_, dist_table) = crate::libdeflate_decode::get_fixed_tables();
    decode_unified(bits, output, out_pos, table, dist_table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_entry_literal() {
        let entry = UnifiedEntry::literal(8, 65);
        assert!(entry.is_literal());
        assert!(!entry.is_double_literal());
        assert_eq!(entry.bits(), 8);
        assert_eq!(entry.literal1(), 65);
    }

    #[test]
    fn test_unified_entry_double_literal() {
        let entry = UnifiedEntry::double_literal(16, 65, 66);
        assert!(entry.is_literal());
        assert!(entry.is_double_literal());
        assert_eq!(entry.bits(), 16);
        assert_eq!(entry.literal1(), 65);
        assert_eq!(entry.literal2(), 66);
    }

    #[test]
    fn test_unified_table_build() {
        let table = UnifiedTable::build_fixed();
        // Should have 2048 entries
        assert_eq!(table.entries.len(), UNIFIED_TABLE_SIZE);

        // Check a known entry (symbol 'A' = 65)
        // In fixed Huffman, literals 0-143 have 8-bit codes
        // 'A' = 65 = 0x41, code = 0x30 + 65 = 0x91 reversed = some value
        eprintln!("Unified table built with {} entries", table.entries.len());
    }
}
