//! Packed Huffman Table - libdeflate-style entry format
//!
//! Entry format (u32):
//! - Bit 31: LITERAL flag (1 = literal byte, can test with `entry as i32 < 0`)
//! - Bits 24-16: literal value OR length base value
//! - Bit 15: EXCEPTIONAL flag (subtable or EOB)
//! - Bit 13: END_OF_BLOCK flag
//! - Bits 7-0: bits to consume (codeword length + extra bits)
//!
//! Key optimization: `bitsleft -= entry` instead of `bitsleft -= (entry & 0xFF)`
//! Works because high bits of bitsleft can contain garbage.

#![allow(dead_code)]

use crate::inflate_tables::{LEN_EXTRA_BITS, LEN_START};
use std::io;

// Entry flags
const HUFFDEC_LITERAL: u32 = 0x8000_0000;
const HUFFDEC_EXCEPTIONAL: u32 = 0x0000_8000;
const HUFFDEC_END_OF_BLOCK: u32 = 0x0000_2000;
const HUFFDEC_SUBTABLE_POINTER: u32 = 0x0000_4000;

// Table sizes
const LITLEN_TABLEBITS: u32 = 11;
const LITLEN_TABLESIZE: usize = 1 << LITLEN_TABLEBITS;
const OFFSET_TABLEBITS: u32 = 8;
const OFFSET_TABLESIZE: usize = 1 << OFFSET_TABLEBITS;

/// Packed litlen decode table
pub struct PackedLitLenTable {
    table: Vec<u32>,
    tablebits: u32,
}

impl PackedLitLenTable {
    /// Build from code lengths
    pub fn build(lens: &[u8]) -> io::Result<Self> {
        let mut table = vec![0u32; LITLEN_TABLESIZE * 2]; // Extra space for subtables
        let mut subtable_pos = LITLEN_TABLESIZE;

        // Count codes of each length
        let mut bl_count = [0u32; 16];
        let mut max_len = 0u32;
        for &len in lens {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
                max_len = max_len.max(len as u32);
            }
        }

        // Calculate starting codes
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=15 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Fill table entries
        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as u32;
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits for table lookup
            let rev = reverse_bits(code, len);

            // Build entry based on symbol type
            let entry = make_litlen_entry(symbol as u16, len);

            if len <= LITLEN_TABLEBITS {
                // Direct entry - fill all matching slots
                let fill_count = 1usize << (LITLEN_TABLEBITS - len);
                for i in 0..fill_count {
                    let idx = (rev as usize) | (i << len as usize);
                    table[idx] = entry;
                }
            } else {
                // Need subtable
                let main_idx = (rev as usize) & ((1 << LITLEN_TABLEBITS) - 1);

                // Check if subtable already allocated
                if table[main_idx] & HUFFDEC_SUBTABLE_POINTER == 0 {
                    // Allocate new subtable
                    let subtable_bits = max_len - LITLEN_TABLEBITS;
                    let subtable_size = 1usize << subtable_bits;

                    // Create subtable pointer
                    table[main_idx] = HUFFDEC_EXCEPTIONAL
                        | HUFFDEC_SUBTABLE_POINTER
                        | ((subtable_pos as u32) << 16)
                        | (subtable_bits << 8)
                        | LITLEN_TABLEBITS;

                    // Ensure table is big enough
                    if subtable_pos + subtable_size > table.len() {
                        table.resize(subtable_pos + subtable_size + 256, 0);
                    }
                    subtable_pos += subtable_size;
                }

                // Get subtable info
                let subtable_start = (table[main_idx] >> 16) as usize;
                let subtable_bits = (table[main_idx] >> 8) & 0x3F;

                // Fill subtable entries
                let extra_bits = len - LITLEN_TABLEBITS;
                let sub_idx = (rev >> LITLEN_TABLEBITS) as usize;
                let fill_count = 1usize << (subtable_bits - extra_bits);

                for i in 0..fill_count {
                    let idx = subtable_start + sub_idx + (i << extra_bits as usize);
                    if idx < table.len() {
                        table[idx] = entry;
                    }
                }
            }
        }

        table.truncate(subtable_pos);
        Ok(Self {
            table,
            tablebits: LITLEN_TABLEBITS,
        })
    }

    /// Decode with packed entry format
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> u32 {
        let idx = (bits as usize) & ((1 << self.tablebits) - 1);
        self.table[idx]
    }

    /// Get subtable entry
    #[inline(always)]
    pub fn decode_subtable(&self, entry: u32, bits: u64) -> u32 {
        let subtable_start = (entry >> 16) as usize;
        let subtable_bits = (entry >> 8) & 0x3F;
        let sub_idx = ((bits >> self.tablebits) as usize) & ((1 << subtable_bits) - 1);
        self.table[subtable_start + sub_idx]
    }
}

/// Make litlen entry from symbol and code length
fn make_litlen_entry(symbol: u16, code_len: u32) -> u32 {
    if symbol < 256 {
        // Literal: high bit set, value in bits 16-23, code length in low byte
        HUFFDEC_LITERAL | ((symbol as u32) << 16) | code_len
    } else if symbol == 256 {
        // End of block
        HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK | code_len
    } else {
        // Length code (257-285)
        let len_idx = (symbol - 257) as usize;
        if len_idx < 29 {
            let length_base = LEN_START[len_idx] as u32;
            let extra_bits = LEN_EXTRA_BITS[len_idx] as u32;
            // Length base in bits 16-24, total bits to consume in low byte
            (length_base << 16) | (code_len + extra_bits)
        } else {
            // Invalid length code
            0
        }
    }
}

/// Reverse bits
#[inline]
fn reverse_bits(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Extract literal value from entry
#[inline(always)]
pub fn entry_literal(entry: u32) -> u8 {
    (entry >> 16) as u8
}

/// Extract length base from entry
#[inline(always)]
pub fn entry_length_base(entry: u32) -> u32 {
    (entry >> 16) & 0x1FF
}

/// Extract bits to consume from entry
#[inline(always)]
pub fn entry_bits(entry: u32) -> u32 {
    entry & 0xFF
}

/// Check if entry is literal
#[inline(always)]
pub fn is_literal(entry: u32) -> bool {
    (entry as i32) < 0 // Test high bit efficiently
}

/// Check if entry is exceptional (EOB or subtable)
#[inline(always)]
pub fn is_exceptional(entry: u32) -> bool {
    entry & HUFFDEC_EXCEPTIONAL != 0
}

/// Check if entry is end of block
#[inline(always)]
pub fn is_end_of_block(entry: u32) -> bool {
    entry & HUFFDEC_END_OF_BLOCK != 0
}

/// Check if entry is subtable pointer
#[inline(always)]
pub fn is_subtable_pointer(entry: u32) -> bool {
    entry & HUFFDEC_SUBTABLE_POINTER != 0
}

// =============================================================================
// Hyperoptimized Decode Loop
// =============================================================================

/// Optimized bit reader for packed decode
pub struct PackedBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    /// Note: bitsleft can have garbage in high bits (like libdeflate)
    /// Only the low 7 bits are meaningful (max 64 bits)
    bitsleft: u32,
}

impl<'a> PackedBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut pb = Self {
            data,
            pos: 0,
            buf: 0,
            bitsleft: 0,
        };
        pb.refill();
        pb
    }

    /// Refill - ensures at least 56 bits available
    #[inline(always)]
    pub fn refill(&mut self) {
        if (self.bitsleft as u8) > 56 {
            return;
        }
        if self.pos + 8 <= self.data.len() {
            let bytes =
                unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= bytes.to_le() << (self.bitsleft as u8);
            let consumed = (64 - (self.bitsleft as u8)) / 8;
            self.pos += consumed as usize;
            // Key: use |= instead of += to allow garbage in high bits
            self.bitsleft |= 56;
        } else {
            while (self.bitsleft as u8) <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << (self.bitsleft as u8);
                self.pos += 1;
                self.bitsleft += 8;
            }
        }
    }

    /// Consume bits - key optimization: subtract whole entry value
    #[inline(always)]
    pub fn consume_entry(&mut self, entry: u32) {
        self.buf >>= entry as u8;
        self.bitsleft -= entry; // Subtracts full entry, garbage in high bits is OK
    }

    /// Consume specific number of bits
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    /// Read n bits
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        val
    }

    /// Get buffer for table lookup
    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }

    /// Check if we're past the end
    #[inline(always)]
    pub fn past_end(&self) -> bool {
        self.pos > self.data.len() + 8
    }

    /// Align to byte boundary
    #[inline]
    pub fn align(&mut self) {
        let skip = (self.bitsleft as u8) % 8;
        if skip > 0 {
            self.consume(skip as u32);
        }
    }
}

/// Hyperoptimized decode using packed tables
/// Returns bytes written
pub fn decode_packed(
    bits: &mut PackedBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen_table: &PackedLitLenTable,
    dist_table: &[u32], // Packed distance table
    dist_tablebits: u32,
) -> io::Result<usize> {
    use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START};

    // Fastloop bounds
    let out_fastloop_end = output.len().saturating_sub(258 + 16);

    // Main fastloop
    while out_pos < out_fastloop_end && !bits.past_end() {
        bits.refill();

        // Preload entry
        let mut entry = litlen_table.decode(bits.buffer());

        // Key optimization: consume bits using whole entry value
        let saved_buf = bits.buffer();
        bits.consume_entry(entry);

        // Fast literal check (test high bit)
        if is_literal(entry) {
            // Output literal
            output[out_pos] = entry_literal(entry);
            out_pos += 1;

            // Multi-literal: try 2 more
            entry = litlen_table.decode(bits.buffer());
            if is_literal(entry) {
                bits.consume_entry(entry);
                output[out_pos] = entry_literal(entry);
                out_pos += 1;

                entry = litlen_table.decode(bits.buffer());
                if is_literal(entry) {
                    bits.consume_entry(entry);
                    output[out_pos] = entry_literal(entry);
                    out_pos += 1;
                }
            }
            continue;
        }

        // Check for exceptional (EOB or subtable)
        if is_exceptional(entry) {
            if is_end_of_block(entry) {
                return Ok(out_pos);
            }

            // Subtable lookup
            entry = litlen_table.decode_subtable(entry, saved_buf);
            bits.consume_entry(entry);

            if is_literal(entry) {
                output[out_pos] = entry_literal(entry);
                out_pos += 1;
                continue;
            }
            if is_end_of_block(entry) {
                return Ok(out_pos);
            }
        }

        // Length code - extract length using saved_buf
        let length_base = entry_length_base(entry);
        let codeword_bits = (entry >> 8) & 0x1F;
        let extra_bits = (entry & 0x1F) - codeword_bits;

        // Extract extra length bits from saved buffer
        let extra_val = if extra_bits > 0 {
            ((saved_buf >> codeword_bits) & ((1u64 << extra_bits) - 1)) as u32
        } else {
            0
        };
        let length = (length_base + extra_val) as usize;

        // Decode distance
        bits.refill();
        let dist_entry = dist_table[(bits.buffer() as usize) & ((1 << dist_tablebits) - 1)];
        let dist_sym = (dist_entry >> 16) as usize;
        let dist_code_bits = dist_entry & 0xFF;
        bits.consume(dist_code_bits);

        bits.refill();
        let dist_extra = DIST_EXTRA_BITS[dist_sym] as u32;
        let distance = DIST_START[dist_sym] as usize + bits.read(dist_extra) as usize;

        if distance > out_pos || distance == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        // LZ77 copy with distance=1 optimization
        copy_match_packed(output, out_pos, distance, length);
        out_pos += length;
    }

    // Generic loop for near end of buffer
    decode_packed_generic(
        bits,
        output,
        out_pos,
        litlen_table,
        dist_table,
        dist_tablebits,
    )
}

/// Generic decode loop for near buffer ends
fn decode_packed_generic(
    bits: &mut PackedBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen_table: &PackedLitLenTable,
    dist_table: &[u32],
    dist_tablebits: u32,
) -> io::Result<usize> {
    use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START};

    loop {
        if bits.past_end() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unexpected end of input",
            ));
        }

        bits.refill();
        let entry = litlen_table.decode(bits.buffer());
        let saved_buf = bits.buffer();
        bits.consume_entry(entry);

        if is_literal(entry) {
            if out_pos >= output.len() {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "Output buffer full",
                ));
            }
            output[out_pos] = entry_literal(entry);
            out_pos += 1;
            continue;
        }

        if is_exceptional(entry) {
            if is_end_of_block(entry) {
                return Ok(out_pos);
            }
            // Handle subtable...
            let sub_entry = litlen_table.decode_subtable(entry, saved_buf);
            bits.consume_entry(sub_entry);

            if is_literal(sub_entry) {
                if out_pos >= output.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Output buffer full",
                    ));
                }
                output[out_pos] = entry_literal(sub_entry);
                out_pos += 1;
                continue;
            }
            if is_end_of_block(sub_entry) {
                return Ok(out_pos);
            }
        }

        // Length code
        let length_base = entry_length_base(entry);
        let codeword_bits = (entry >> 8) & 0x1F;
        let extra_bits = (entry & 0x1F) - codeword_bits;
        let extra_val = if extra_bits > 0 {
            ((saved_buf >> codeword_bits) & ((1u64 << extra_bits) - 1)) as u32
        } else {
            0
        };
        let length = (length_base + extra_val) as usize;

        bits.refill();
        let dist_entry = dist_table[(bits.buffer() as usize) & ((1 << dist_tablebits) - 1)];
        let dist_sym = (dist_entry >> 16) as usize;
        bits.consume(dist_entry & 0xFF);

        bits.refill();
        let dist_extra = DIST_EXTRA_BITS[dist_sym] as u32;
        let distance = DIST_START[dist_sym] as usize + bits.read(dist_extra) as usize;

        if distance > out_pos || distance == 0 || out_pos + length > output.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance or length",
            ));
        }

        copy_match_packed(output, out_pos, distance, length);
        out_pos += length;
    }
}

/// Optimized LZ77 copy
#[inline(always)]
fn copy_match_packed(output: &mut [u8], out_pos: usize, distance: usize, length: usize) {
    let src_start = out_pos - distance;

    unsafe {
        let dst = output.as_mut_ptr().add(out_pos);
        let src = output.as_ptr().add(src_start);

        if distance == 1 {
            // RLE: memset
            std::ptr::write_bytes(dst, *src, length);
        } else if distance >= length {
            // Non-overlapping
            std::ptr::copy_nonoverlapping(src, dst, length);
        } else if distance >= 8 {
            // Overlapping with distance >= 8: chunk copy
            let mut remaining = length;
            let mut d = dst;
            let mut s = src;
            while remaining >= 8 {
                let chunk = (s as *const u64).read_unaligned();
                (d as *mut u64).write_unaligned(chunk);
                d = d.add(8);
                s = s.add(8);
                remaining -= 8;
            }
            for i in 0..remaining {
                *d.add(i) = *s.add(i);
            }
        } else {
            // Small distance
            for i in 0..length {
                *dst.add(i) = *src.add(i % distance);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_huffman_table() {
        // Build fixed Huffman table
        let mut lens = [0u8; 288];
        for len in lens.iter_mut().take(144) {
            *len = 8;
        }
        for len in lens.iter_mut().take(256).skip(144) {
            *len = 9;
        }
        for len in lens.iter_mut().take(280).skip(256) {
            *len = 7;
        }
        for len in lens.iter_mut().take(288).skip(280) {
            *len = 8;
        }

        let table = PackedLitLenTable::build(&lens).unwrap();

        // Test that we can decode symbols
        // Symbol 0 (literal 0x00) should be at some position
        println!("Table size: {} entries", table.table.len());

        // Verify some entries
        let mut literals = 0;
        let mut lengths = 0;
        let mut eob = 0;

        for &entry in &table.table[..LITLEN_TABLESIZE] {
            if entry == 0 {
                continue;
            }
            if is_literal(entry) {
                literals += 1;
            } else if is_end_of_block(entry) {
                eob += 1;
            } else if !is_exceptional(entry) {
                lengths += 1;
            }
        }

        println!("Literals: {}, Lengths: {}, EOB: {}", literals, lengths, eob);
    }

    #[test]
    fn test_entry_format() {
        // Test literal entry
        let lit_entry = make_litlen_entry(65, 8); // 'A' with 8-bit code
        assert!(is_literal(lit_entry));
        assert_eq!(entry_literal(lit_entry), 65);
        assert_eq!(entry_bits(lit_entry), 8);

        // Test EOB entry
        let eob_entry = make_litlen_entry(256, 7);
        assert!(!is_literal(eob_entry));
        assert!(is_exceptional(eob_entry));
        assert!(is_end_of_block(eob_entry));
        assert_eq!(entry_bits(eob_entry), 7);

        // Test length entry
        let len_entry = make_litlen_entry(257, 7); // length 3, 0 extra bits
        assert!(!is_literal(len_entry));
        assert!(!is_exceptional(len_entry));
        assert_eq!(entry_length_base(len_entry), 3);
        assert_eq!(entry_bits(len_entry), 7); // 7 code bits + 0 extra
    }
}
