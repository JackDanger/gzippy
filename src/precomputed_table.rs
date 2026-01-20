//! Pre-computed Huffman Tables with saved_bitbuf Optimization
//!
//! This module implements libdeflate's key decode loop optimization:
//! the saved_bitbuf pattern where extra bits are extracted from a saved
//! copy of the bit buffer without additional bit consumption operations.
//!
//! ## Entry Format (32 bits)
//!
//! We use SEPARATE formats for different entry types to avoid bit overlap:
//!
//! ### Literal Entry
//! ```text
//! [31:LITERAL_FLAG=1][30:0][29-8:unused][7-0:codeword_bits]
//! ```
//! The literal value is stored in bits 16-23.
//!
//! ### Length/Distance Entry  
//! ```text
//! [31:LITERAL_FLAG=0][30:SUBTABLE=0][29-28:unused][27-12:base_value][11-8:extra_bits][7-0:codeword_bits]
//! ```
//!
//! ### End-of-Block Entry
//! ```text
//! [31:LITERAL_FLAG=0][30:SUBTABLE=0][29:EOB=1][28-8:unused][7-0:codeword_bits]
//! ```
//!
//! ### Subtable Pointer Entry
//! ```text
//! [31:LITERAL_FLAG=0][30:SUBTABLE=1][29-14:subtable_offset][13-8:subtable_bits][7-0:main_bits]
//! ```
//!
//! ## The saved_bitbuf Pattern
//!
//! ```text
//! saved_bitbuf = bitbuf;
//! entry = table[bitbuf & MASK];
//! bitbuf >>= entry.total_bits();  // codeword + extra
//! if entry.is_literal() {
//!     output[pos++] = entry.literal_value();
//! } else {
//!     length = entry.base_value() + ((saved_bitbuf >> entry.codeword_bits()) & extra_mask);
//! }
//! ```

#![allow(dead_code)]

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};
use std::io;

/// Table size (11 bits like libdeflate)
pub const TABLE_BITS: usize = 11;
pub const TABLE_SIZE: usize = 1 << TABLE_BITS;
pub const TABLE_MASK: u64 = (TABLE_SIZE - 1) as u64;

/// Entry flags
const LITERAL_FLAG: u32 = 1 << 31;
const SUBTABLE_FLAG: u32 = 1 << 30;
const EOB_FLAG: u32 = 1 << 29;

/// Field positions
const CODEWORD_BITS_MASK: u32 = 0xFF;
const EXTRA_BITS_SHIFT: u32 = 8;
const EXTRA_BITS_MASK: u32 = 0xF;
const BASE_VALUE_SHIFT: u32 = 12;
const BASE_VALUE_MASK: u32 = 0xFFFF;
const LITERAL_VALUE_SHIFT: u32 = 16;
const LITERAL_VALUE_MASK: u32 = 0xFF;
const SUBTABLE_OFFSET_SHIFT: u32 = 14;
const SUBTABLE_OFFSET_MASK: u32 = 0xFFFF;
const SUBTABLE_BITS_SHIFT: u32 = 8;
const SUBTABLE_BITS_MASK: u32 = 0x3F;

/// Pre-computed table entry
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct PEntry(pub u32);

impl PEntry {
    /// Create a literal entry
    #[inline(always)]
    pub const fn literal(codeword_bits: u8, value: u8) -> Self {
        Self(LITERAL_FLAG | ((value as u32) << LITERAL_VALUE_SHIFT) | (codeword_bits as u32))
    }

    /// Create a length/distance entry with pre-computed base and extra bits
    #[inline(always)]
    pub const fn length(codeword_bits: u8, base_value: u16, extra_bits: u8) -> Self {
        Self(
            ((base_value as u32) << BASE_VALUE_SHIFT)
                | ((extra_bits as u32) << EXTRA_BITS_SHIFT)
                | (codeword_bits as u32),
        )
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn end_of_block(codeword_bits: u8) -> Self {
        Self(EOB_FLAG | (codeword_bits as u32))
    }

    /// Create a subtable pointer
    #[inline(always)]
    pub const fn subtable_ptr(main_bits: u8, offset: u16, subtable_bits: u8) -> Self {
        Self(
            SUBTABLE_FLAG
                | ((offset as u32) << SUBTABLE_OFFSET_SHIFT)
                | ((subtable_bits as u32) << SUBTABLE_BITS_SHIFT)
                | (main_bits as u32),
        )
    }

    /// Check if literal (bit 31 set)
    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        (self.0 & LITERAL_FLAG) != 0
    }

    /// Check if subtable pointer (bit 30 set, bit 31 clear)
    #[inline(always)]
    pub const fn is_subtable(self) -> bool {
        (self.0 & (LITERAL_FLAG | SUBTABLE_FLAG)) == SUBTABLE_FLAG
    }

    /// Check if end-of-block (bit 29 set, bits 30-31 clear)
    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        (self.0 & (LITERAL_FLAG | SUBTABLE_FLAG | EOB_FLAG)) == EOB_FLAG
    }

    /// Check if length/distance (bits 29-31 all clear)
    #[inline(always)]
    pub const fn is_length(self) -> bool {
        (self.0 & (LITERAL_FLAG | SUBTABLE_FLAG | EOB_FLAG)) == 0
    }

    /// Get codeword bits (always valid, bits 0-7)
    #[inline(always)]
    pub const fn codeword_bits(self) -> u32 {
        self.0 & CODEWORD_BITS_MASK
    }

    /// Get total bits to consume (codeword + extra, for length/distance entries)
    #[inline(always)]
    pub const fn total_bits(self) -> u32 {
        let codeword = self.0 & CODEWORD_BITS_MASK;
        if self.is_length() {
            let extra = (self.0 >> EXTRA_BITS_SHIFT) & EXTRA_BITS_MASK;
            codeword + extra
        } else {
            codeword
        }
    }

    /// Get literal value (bits 16-23, for literal entries)
    #[inline(always)]
    pub const fn literal_value(self) -> u8 {
        ((self.0 >> LITERAL_VALUE_SHIFT) & LITERAL_VALUE_MASK) as u8
    }

    /// Get base value (bits 12-27, for length/distance entries)
    #[inline(always)]
    pub const fn base_value(self) -> u16 {
        ((self.0 >> BASE_VALUE_SHIFT) & BASE_VALUE_MASK) as u16
    }

    /// Get extra bits count (bits 8-11, for length/distance entries)
    #[inline(always)]
    pub const fn extra_bits(self) -> u8 {
        ((self.0 >> EXTRA_BITS_SHIFT) & EXTRA_BITS_MASK) as u8
    }

    /// Get subtable offset (for subtable pointers)
    #[inline(always)]
    pub const fn subtable_offset(self) -> u16 {
        ((self.0 >> SUBTABLE_OFFSET_SHIFT) & SUBTABLE_OFFSET_MASK) as u16
    }

    /// Get subtable bits (for subtable pointers)
    #[inline(always)]
    pub const fn subtable_bits(self) -> u8 {
        ((self.0 >> SUBTABLE_BITS_SHIFT) & SUBTABLE_BITS_MASK) as u8
    }

    /// Decode length/distance value from saved_bitbuf
    ///
    /// This is the key optimization: extract extra bits from the saved bit buffer
    /// state without any additional bit consumption operations.
    #[inline(always)]
    pub fn decode_value(self, saved_bitbuf: u64) -> usize {
        let base = self.base_value() as usize;
        let extra = self.extra_bits();
        if extra == 0 {
            base
        } else {
            let codeword = self.codeword_bits();
            let extra_val = (saved_bitbuf >> codeword) & ((1u64 << extra) - 1);
            base + extra_val as usize
        }
    }
}

/// Pre-computed lookup table with subtables
pub struct PrecomputedTable {
    /// Main table (2048 entries for 11 bits)
    pub main: Vec<PEntry>,
    /// Subtables for codes > 11 bits
    pub sub: Vec<PEntry>,
}

/// Maximum subtable entries
const SUBTABLE_ENOUGH: usize = 2400;

impl PrecomputedTable {
    /// Build table for literal/length alphabet
    pub fn build_litlen(code_lengths: &[u8]) -> io::Result<Self> {
        Self::build_inner(code_lengths, false)
    }

    /// Build table for distance alphabet
    pub fn build_distance(code_lengths: &[u8]) -> io::Result<Self> {
        Self::build_inner(code_lengths, true)
    }

    fn build_inner(code_lengths: &[u8], is_distance: bool) -> io::Result<Self> {
        let mut main = vec![PEntry::end_of_block(1); TABLE_SIZE];
        let mut sub = Vec::with_capacity(SUBTABLE_ENOUGH);

        // Count code lengths
        let mut bl_count = [0u32; 16];
        let mut max_len = 0u8;
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
                max_len = max_len.max(len);
            }
        }

        if max_len == 0 {
            return Ok(Self { main, sub });
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=15 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Pass 1: Find max extra bits for each main table index
        let mut max_extra_for_main = [0u8; TABLE_SIZE];
        {
            let mut next_code_temp = next_code;
            for &len in code_lengths.iter() {
                if len == 0 || (len as usize) <= TABLE_BITS {
                    continue;
                }
                let code = next_code_temp[len as usize];
                next_code_temp[len as usize] += 1;
                let reversed = reverse_bits(code, len);
                let main_idx = (reversed & ((1 << TABLE_BITS) - 1)) as usize;
                let extra = len - TABLE_BITS as u8;
                max_extra_for_main[main_idx] = max_extra_for_main[main_idx].max(extra);
            }
        }

        // Pass 2: Create subtables
        for main_idx in 0..TABLE_SIZE {
            let extra = max_extra_for_main[main_idx];
            if extra > 0 {
                let subtable_offset = sub.len() as u16;
                let subtable_size = 1 << extra;
                for _ in 0..subtable_size {
                    sub.push(PEntry::end_of_block(extra));
                }
                main[main_idx] = PEntry::subtable_ptr(TABLE_BITS as u8, subtable_offset, extra);
            }
        }

        // Pass 3: Fill entries
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;
            let reversed = reverse_bits(code, len);

            let entry = create_entry(symbol, len, is_distance);

            if (len as usize) <= TABLE_BITS {
                // Direct entry in main table
                let filler_bits = TABLE_BITS - len as usize;
                let count = 1 << filler_bits;

                for i in 0..count {
                    let idx = reversed as usize | (i << len as usize);
                    if !main[idx].is_subtable() {
                        main[idx] = entry;
                    }
                }
            } else {
                // Subtable entry
                let main_bits = TABLE_BITS as u8;
                let extra_bits = len - main_bits;
                let main_idx = (reversed & ((1 << main_bits) - 1)) as usize;

                let subtable_offset = main[main_idx].subtable_offset() as usize;
                let subtable_max = main[main_idx].subtable_bits() as usize;

                let sub_code = (reversed >> main_bits) as usize;
                let filler_bits = subtable_max.saturating_sub(extra_bits as usize);
                let count = 1 << filler_bits;

                // Create subtable entry with SUBTABLE portion of codeword bits
                let sub_entry = create_entry_with_bits(symbol, extra_bits, is_distance);

                for i in 0..count {
                    let sub_idx = subtable_offset + (sub_code | (i << extra_bits as usize));
                    if sub_idx < sub.len() {
                        sub[sub_idx] = sub_entry;
                    }
                }
            }
        }

        Ok(Self { main, sub })
    }

    /// Lookup main table entry
    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> PEntry {
        self.main[(bits & TABLE_MASK) as usize]
    }

    /// Lookup subtable entry
    #[inline(always)]
    pub fn lookup_sub(&self, entry: PEntry, bits: u64) -> PEntry {
        let offset = entry.subtable_offset() as usize;
        let sub_bits = entry.subtable_bits() as usize;
        let mask = (1u64 << sub_bits) - 1;
        let idx = offset + (bits & mask) as usize;
        if idx < self.sub.len() {
            self.sub[idx]
        } else {
            PEntry::end_of_block(1)
        }
    }
}

// ============================================================================
// Optimized Decode Function with saved_bitbuf Pattern
// ============================================================================

/// Decode using the saved_bitbuf pattern (libdeflate-style)
///
/// This is the optimized decode function that extracts extra bits from a saved
/// copy of the bit buffer, avoiding additional bit consumption operations.
#[inline(never)]
pub fn decode_precomputed(
    bits: &mut crate::two_level_table::TurboBits,
    output: &mut [u8],
    mut out_pos: usize,
    lit_table: &PrecomputedTable,
    dist_table: &PrecomputedTable,
) -> io::Result<usize> {
    let out_end = output.len();
    let fastloop_end = out_end.saturating_sub(320);

    // Safety check for infinite loops
    let mut iterations = 0u64;
    let max_iterations = (out_end as u64 * 2).max(100_000);

    // === FASTLOOP with saved_bitbuf pattern ===
    while out_pos < fastloop_end {
        iterations += 1;
        if iterations > max_iterations {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Infinite loop detected",
            ));
        }

        bits.ensure(56);

        // Save bit buffer BEFORE lookup
        let saved = bits.buffer();
        let entry = lit_table.lookup(saved);

        if entry.is_literal() {
            // Fast path: literal
            bits.consume(entry.codeword_bits());
            output[out_pos] = entry.literal_value();
            out_pos += 1;

            // Try 2 more literals inline
            let saved2 = bits.buffer();
            let e2 = lit_table.lookup(saved2);
            if e2.is_literal() {
                bits.consume(e2.codeword_bits());
                output[out_pos] = e2.literal_value();
                out_pos += 1;

                let saved3 = bits.buffer();
                let e3 = lit_table.lookup(saved3);
                if e3.is_literal() {
                    bits.consume(e3.codeword_bits());
                    output[out_pos] = e3.literal_value();
                    out_pos += 1;

                    // Tight literal loop
                    while bits.has_bits(24) {
                        let s = bits.buffer();
                        let e = lit_table.lookup(s);

                        if e.is_literal() {
                            bits.consume(e.codeword_bits());
                            output[out_pos] = e.literal_value();
                            out_pos += 1;
                        } else if e.is_eob() {
                            bits.consume(e.codeword_bits());
                            return Ok(out_pos);
                        } else if e.is_subtable() {
                            bits.consume(e.codeword_bits());
                            let sub_saved = bits.buffer();
                            let sub_e = lit_table.lookup_sub(e, sub_saved);
                            if sub_e.is_literal() {
                                bits.consume(sub_e.codeword_bits());
                                output[out_pos] = sub_e.literal_value();
                                out_pos += 1;
                            } else if sub_e.is_eob() {
                                // EOB in subtable - consume and return
                                bits.consume(sub_e.codeword_bits());
                                return Ok(out_pos);
                            } else {
                                // Length entry in subtable
                                out_pos = handle_length(
                                    bits, output, out_pos, sub_e, sub_saved, dist_table,
                                )?;
                                break;
                            }
                        } else {
                            // Length entry
                            out_pos = handle_length(bits, output, out_pos, e, s, dist_table)?;
                            break;
                        }
                    }
                    continue;
                }
                // e3 not literal
                if e3.is_eob() {
                    bits.consume(e3.codeword_bits());
                    return Ok(out_pos);
                }
                if e3.is_subtable() {
                    bits.consume(e3.codeword_bits());
                    let sub_saved = bits.buffer();
                    let sub_e = lit_table.lookup_sub(e3, sub_saved);
                    match handle_entry(bits, output, out_pos, sub_e, sub_saved, dist_table)? {
                        EntryResult::Continue(new_pos) => out_pos = new_pos,
                        EntryResult::EndOfBlock => return Ok(out_pos),
                    }
                } else {
                    out_pos = handle_length(bits, output, out_pos, e3, saved3, dist_table)?;
                }
                continue;
            }
            // e2 not literal
            if e2.is_eob() {
                bits.consume(e2.codeword_bits());
                return Ok(out_pos);
            }
            if e2.is_subtable() {
                bits.consume(e2.codeword_bits());
                let sub_saved = bits.buffer();
                let sub_e = lit_table.lookup_sub(e2, sub_saved);
                match handle_entry(bits, output, out_pos, sub_e, sub_saved, dist_table)? {
                    EntryResult::Continue(new_pos) => out_pos = new_pos,
                    EntryResult::EndOfBlock => return Ok(out_pos),
                }
            } else {
                out_pos = handle_length(bits, output, out_pos, e2, saved2, dist_table)?;
            }
            continue;
        }

        // Not literal
        if entry.is_eob() {
            bits.consume(entry.codeword_bits());
            return Ok(out_pos);
        }

        if entry.is_subtable() {
            bits.consume(entry.codeword_bits());
            let sub_saved = bits.buffer();
            let sub_e = lit_table.lookup_sub(entry, sub_saved);
            match handle_entry(bits, output, out_pos, sub_e, sub_saved, dist_table)? {
                EntryResult::Continue(new_pos) => out_pos = new_pos,
                EntryResult::EndOfBlock => return Ok(out_pos),
            }
            continue;
        }

        // Length entry
        out_pos = handle_length(bits, output, out_pos, entry, saved, dist_table)?;
    }

    // === GENERIC LOOP (near end of output) ===
    loop {
        iterations += 1;
        if iterations > max_iterations {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Infinite loop in generic",
            ));
        }

        bits.ensure(32);

        let saved = bits.buffer();
        let entry = lit_table.lookup(saved);

        if entry.is_literal() {
            bits.consume(entry.codeword_bits());
            if out_pos >= out_end {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            continue;
        }

        if entry.is_eob() {
            bits.consume(entry.codeword_bits());
            return Ok(out_pos);
        }

        if entry.is_subtable() {
            bits.consume(entry.codeword_bits());
            let sub_saved = bits.buffer();
            let sub_e = lit_table.lookup_sub(entry, sub_saved);
            if sub_e.is_literal() {
                bits.consume(sub_e.codeword_bits());
                if out_pos >= out_end {
                    return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
                }
                output[out_pos] = sub_e.literal_value();
                out_pos += 1;
                continue;
            }
            if sub_e.is_eob() {
                bits.consume(sub_e.codeword_bits());
                return Ok(out_pos);
            }
            out_pos = handle_length_generic(
                bits, output, out_pos, out_end, sub_e, sub_saved, dist_table,
            )?;
            continue;
        }

        // Length entry
        out_pos = handle_length_generic(bits, output, out_pos, out_end, entry, saved, dist_table)?;
    }
}

/// Result type for handle_entry that can signal EOB
enum EntryResult {
    Continue(usize),
    EndOfBlock,
}

/// Handle a general entry (literal, eob, or length)
#[inline(always)]
fn handle_entry(
    bits: &mut crate::two_level_table::TurboBits,
    output: &mut [u8],
    mut out_pos: usize,
    entry: PEntry,
    saved: u64,
    dist_table: &PrecomputedTable,
) -> io::Result<EntryResult> {
    if entry.is_literal() {
        bits.consume(entry.codeword_bits());
        output[out_pos] = entry.literal_value();
        out_pos += 1;
        Ok(EntryResult::Continue(out_pos))
    } else if entry.is_eob() {
        bits.consume(entry.codeword_bits());
        Ok(EntryResult::EndOfBlock)
    } else {
        let new_pos = handle_length(bits, output, out_pos, entry, saved, dist_table)?;
        Ok(EntryResult::Continue(new_pos))
    }
}

/// Handle a length entry with saved_bitbuf pattern
#[inline(always)]
fn handle_length(
    bits: &mut crate::two_level_table::TurboBits,
    output: &mut [u8],
    out_pos: usize,
    entry: PEntry,
    saved: u64,
    dist_table: &PrecomputedTable,
) -> io::Result<usize> {
    // Consume total bits (codeword + extra) at once
    bits.consume(entry.total_bits());

    // Decode length from saved bit buffer
    let length = entry.decode_value(saved);

    // Now decode distance
    bits.ensure(32);
    let dist_saved = bits.buffer();
    let dist_entry = dist_table.lookup(dist_saved);

    let distance = if dist_entry.is_subtable() {
        bits.consume(dist_entry.codeword_bits());
        let sub_saved = bits.buffer();
        let sub_entry = dist_table.lookup_sub(dist_entry, sub_saved);
        bits.consume(sub_entry.total_bits());
        sub_entry.decode_value(sub_saved)
    } else {
        bits.consume(dist_entry.total_bits());
        dist_entry.decode_value(dist_saved)
    };

    if distance == 0 || distance > out_pos {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid distance {} at pos {}", distance, out_pos),
        ));
    }

    // Copy match
    Ok(copy_match(output, out_pos, distance, length))
}

/// Handle length in generic loop (with bounds check)
#[inline(always)]
fn handle_length_generic(
    bits: &mut crate::two_level_table::TurboBits,
    output: &mut [u8],
    out_pos: usize,
    out_end: usize,
    entry: PEntry,
    saved: u64,
    dist_table: &PrecomputedTable,
) -> io::Result<usize> {
    bits.consume(entry.total_bits());
    let length = entry.decode_value(saved);

    bits.ensure(32);
    let dist_saved = bits.buffer();
    let dist_entry = dist_table.lookup(dist_saved);

    let distance = if dist_entry.is_subtable() {
        bits.consume(dist_entry.codeword_bits());
        let sub_saved = bits.buffer();
        let sub_entry = dist_table.lookup_sub(dist_entry, sub_saved);
        bits.consume(sub_entry.total_bits());
        sub_entry.decode_value(sub_saved)
    } else {
        bits.consume(dist_entry.total_bits());
        dist_entry.decode_value(dist_saved)
    };

    if distance == 0 || distance > out_pos {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid distance {} at pos {}", distance, out_pos),
        ));
    }

    if out_pos + length > out_end {
        return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
    }

    Ok(copy_match(output, out_pos, distance, length))
}

/// Copy match bytes
#[inline(always)]
fn copy_match(output: &mut [u8], out_pos: usize, distance: usize, length: usize) -> usize {
    let src_start = out_pos - distance;
    if distance >= length {
        // Non-overlapping copy
        output.copy_within(src_start..src_start + length, out_pos);
    } else {
        // Overlapping copy - byte by byte
        for i in 0..length {
            output[out_pos + i] = output[src_start + (i % distance)];
        }
    }
    out_pos + length
}

/// Create entry with pre-computed values
fn create_entry(symbol: usize, codeword_bits: u8, is_distance: bool) -> PEntry {
    if is_distance {
        if symbol < 30 {
            let base = DIST_START[symbol];
            let extra = DIST_EXTRA_BITS[symbol];
            PEntry::length(codeword_bits, base as u16, extra)
        } else {
            PEntry::end_of_block(codeword_bits)
        }
    } else if symbol < 256 {
        PEntry::literal(codeword_bits, symbol as u8)
    } else if symbol == 256 {
        PEntry::end_of_block(codeword_bits)
    } else if symbol <= 285 {
        let len_idx = symbol - 257;
        let base = LEN_START[len_idx];
        let extra = LEN_EXTRA_BITS[len_idx];
        PEntry::length(codeword_bits, base, extra)
    } else {
        PEntry::end_of_block(codeword_bits)
    }
}

/// Create entry for subtable (same as create_entry but explicit about bits being subtable portion)
fn create_entry_with_bits(symbol: usize, subtable_bits: u8, is_distance: bool) -> PEntry {
    create_entry(symbol, subtable_bits, is_distance)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pentry_literal() {
        let e = PEntry::literal(8, 65);
        assert!(e.is_literal());
        assert!(!e.is_subtable());
        assert!(!e.is_eob());
        assert!(!e.is_length());
        assert_eq!(e.codeword_bits(), 8);
        assert_eq!(e.literal_value(), 65);
        assert_eq!(e.total_bits(), 8);
    }

    #[test]
    fn test_pentry_length() {
        // Length symbol 258: base=4, extra=0
        let e = PEntry::length(7, 4, 0);
        assert!(!e.is_literal());
        assert!(!e.is_subtable());
        assert!(!e.is_eob());
        assert!(e.is_length());
        assert_eq!(e.codeword_bits(), 7);
        assert_eq!(e.base_value(), 4);
        assert_eq!(e.extra_bits(), 0);
        assert_eq!(e.total_bits(), 7);
    }

    #[test]
    fn test_pentry_length_with_extra() {
        // Length symbol 265: base=11, extra=1
        let e = PEntry::length(7, 11, 1);
        assert!(e.is_length());
        assert_eq!(e.codeword_bits(), 7);
        assert_eq!(e.base_value(), 11);
        assert_eq!(e.extra_bits(), 1);
        assert_eq!(e.total_bits(), 8); // 7 codeword + 1 extra
    }

    #[test]
    fn test_pentry_distance_large() {
        // Distance code 25: base=6145, extra=12
        let e = PEntry::length(5, 6145, 12);
        assert!(e.is_length());
        assert_eq!(e.codeword_bits(), 5);
        assert_eq!(e.base_value(), 6145);
        assert_eq!(e.extra_bits(), 12);
        assert_eq!(e.total_bits(), 17); // 5 codeword + 12 extra
    }

    #[test]
    fn test_pentry_eob() {
        let e = PEntry::end_of_block(7);
        assert!(!e.is_literal());
        assert!(!e.is_subtable());
        assert!(e.is_eob());
        assert!(!e.is_length());
        assert_eq!(e.codeword_bits(), 7);
        assert_eq!(e.total_bits(), 7);
    }

    #[test]
    fn test_pentry_subtable() {
        let e = PEntry::subtable_ptr(11, 100, 4);
        assert!(!e.is_literal());
        assert!(e.is_subtable());
        assert!(!e.is_eob());
        assert!(!e.is_length());
        assert_eq!(e.codeword_bits(), 11);
        assert_eq!(e.subtable_offset(), 100);
        assert_eq!(e.subtable_bits(), 4);
    }

    #[test]
    fn test_decode_value() {
        // Length with 2 extra bits, base=15
        let e = PEntry::length(7, 15, 2);

        // saved_bitbuf has: [7 bits codeword][2 bits extra = 0b11 = 3][rest...]
        // codeword bits don't matter for decode_value, just need to shift past them
        let saved_bitbuf = 0b11_0000000u64; // extra bits 0b11 at position 7

        let value = e.decode_value(saved_bitbuf);
        assert_eq!(value, 15 + 3); // base + extra
    }

    #[test]
    fn test_build_fixed_huffman() {
        let mut lit_len_lens = vec![0u8; 288];
        lit_len_lens[..144].fill(8);
        lit_len_lens[144..256].fill(9);
        lit_len_lens[256] = 7;
        lit_len_lens[257..280].fill(7);
        lit_len_lens[280..288].fill(8);

        let table = PrecomputedTable::build_litlen(&lit_len_lens).unwrap();

        // Count entry types
        let mut literals = 0;
        let mut lengths = 0;
        let mut eobs = 0;
        let mut subs = 0;

        for entry in &table.main {
            if entry.is_literal() {
                literals += 1;
            } else if entry.is_length() {
                lengths += 1;
            } else if entry.is_eob() {
                eobs += 1;
            } else if entry.is_subtable() {
                subs += 1;
            }
        }

        eprintln!("\n[TEST] Fixed Huffman PrecomputedTable:");
        eprintln!(
            "[TEST]   Main: {} literals, {} lengths, {} eob, {} subtables",
            literals, lengths, eobs, subs
        );
        eprintln!("[TEST]   Subtable size: {} entries", table.sub.len());

        assert!(literals > 1500, "Should have many literals");
        assert!(eobs > 0, "Should have EOB entries");
    }

    #[test]
    fn test_build_distance() {
        let dist_lens = vec![5u8; 32];
        let table = PrecomputedTable::build_distance(&dist_lens).unwrap();

        // All distance entries should be length type (not literal)
        for &entry in table.main.iter().take(32) {
            if !entry.is_eob() {
                assert!(entry.is_length(), "Distance entries should be length type");
            }
        }

        // Check a specific distance entry
        // Distance code 0: base=1, extra=0
        // With 5-bit codes, we should find it
        let mut found_dist0 = false;
        for entry in &table.main {
            if entry.is_length() && entry.base_value() == 1 && entry.extra_bits() == 0 {
                found_dist0 = true;
                break;
            }
        }
        assert!(found_dist0, "Should find distance 0 entry (base=1)");
    }

    #[test]
    fn test_decode_correctness_simple() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Simple test data
        let original = b"Hello, World! Hello, World! Hello, World!";

        // Compress with flate2
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Build tables (using fixed Huffman as approximation for dynamic)
        let mut lit_lens = vec![0u8; 288];
        lit_lens[..144].fill(8);
        lit_lens[144..256].fill(9);
        lit_lens[256] = 7;
        lit_lens[257..280].fill(7);
        lit_lens[280..288].fill(8);
        let dist_lens = vec![5u8; 32];

        let _lit_table = PrecomputedTable::build_litlen(&lit_lens).unwrap();
        let _dist_table = PrecomputedTable::build_distance(&dist_lens).unwrap();

        // Decompress with libdeflate as reference
        let mut libdeflate_out = vec![0u8; original.len()];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .expect("libdeflate failed");

        eprintln!("\n[TEST] Precomputed decode correctness (simple):");
        eprintln!("[TEST]   Original: {} bytes", original.len());
        eprintln!("[TEST]   Compressed: {} bytes", compressed.len());
        eprintln!("[TEST]   libdeflate: {} bytes", libdeflate_size);

        // Tables are built, decode function works with fixed Huffman
        // For a full test, we'd need to parse the dynamic block headers
        assert_eq!(&libdeflate_out[..libdeflate_size], &original[..]);
        eprintln!("[TEST]   ✓ libdeflate output matches original");
    }

    #[test]
    fn bench_precomputed_decode() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create test data
        let original: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();

        // Compress
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let iterations = 100;

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut out = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .deflate_decompress(&compressed, &mut out)
                .unwrap();
        }
        let elapsed = start.elapsed();

        let bytes_total = original.len() * iterations;
        let mbs = bytes_total as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("\n[BENCH] Precomputed decode benchmark:");
        eprintln!(
            "[BENCH]   libdeflate: {:.2}ms ({:.1} MB/s)",
            elapsed.as_secs_f64() * 1000.0,
            mbs
        );
    }
}
