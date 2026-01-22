//! Bytecode Huffman Decoder
//!
//! Generates specialized bytecode for each Huffman table at build time,
//! then interprets it at decode time. This avoids JIT compilation overhead
//! while still having table-specific decode logic.
//!
//! ## Bytecode Format
//!
//! Each entry in the bytecode table is a packed 32-bit value:
//! ```text
//! Literal:  [1:1][symbol:8][bits:5][reserved:18]
//! Length:   [0:1][0:1][base:9][extra:4][bits:5][reserved:12]
//! EOB:      [0:1][1:1][bits:5][reserved:25]
//! Subtable: [0:1][0:1][1:1][offset:13][subbits:4][mainbits:5][reserved:8]
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::jit_decode::TableFingerprint;

/// Bytecode entry flags
const BC_LITERAL: u32 = 0x8000_0000; // Bit 31 = literal
const BC_EOB: u32 = 0x4000_0000; // Bit 30 = end of block
const BC_SUBTABLE: u32 = 0x2000_0000; // Bit 29 = subtable pointer

/// Bytecode table for a specific Huffman code
#[derive(Clone)]
pub struct BytecodeTable {
    /// Main table (2048 entries for 11 bits)
    pub table: Vec<u32>,
    /// Fingerprint for cache lookup
    pub fingerprint: TableFingerprint,
}

impl BytecodeTable {
    /// Build a bytecode table from code lengths
    pub fn build(litlen_lens: &[u8]) -> Option<Self> {
        const TABLE_BITS: usize = 11;
        const TABLE_SIZE: usize = 1 << TABLE_BITS;
        const MAX_SUBTABLE_BITS: usize = 4;

        let fingerprint = TableFingerprint::from_litlen_lengths(litlen_lens);

        // Count codes at each length
        let mut bl_count = [0u32; 16];
        for &len in litlen_lens {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Allocate table with space for subtables
        let max_subtable_entries = (1usize << MAX_SUBTABLE_BITS)
            * litlen_lens
                .iter()
                .filter(|&&l| l > TABLE_BITS as u8)
                .count();
        let mut table = vec![0u32; TABLE_SIZE + max_subtable_entries];
        let mut subtable_next = TABLE_SIZE;

        // Length base values
        const LENGTH_BASES: [u16; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        const LENGTH_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        // Assign codes to symbols
        for (symbol, &len) in litlen_lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;
            let codeword = next_code[len];
            next_code[len] += 1;

            // Reverse bits
            let reversed = reverse_bits(codeword, len as u8);

            if len <= TABLE_BITS {
                // Direct entry in main table
                let entry = if symbol < 256 {
                    // Literal: [1:1][symbol:8][bits:5]
                    BC_LITERAL | ((symbol as u32) << 16) | (len as u32)
                } else if symbol == 256 {
                    // EOB: [0:1][1:1][bits:5]
                    BC_EOB | (len as u32)
                } else if symbol <= 285 {
                    // Length: [base:9][extra:4][bits:5]
                    let idx = symbol - 257;
                    let base = LENGTH_BASES[idx] as u32;
                    let extra = LENGTH_EXTRA[idx] as u32;
                    (base << 16) | (extra << 8) | (len as u32)
                } else {
                    continue;
                };

                // Fill all matching entries
                let stride = 1usize << len;
                let mut idx = reversed as usize;
                while idx < TABLE_SIZE {
                    table[idx] = entry;
                    idx += stride;
                }
            } else {
                // Need subtable
                let main_idx = (reversed & ((1 << TABLE_BITS) - 1)) as usize;
                let extra_bits = len - TABLE_BITS;

                // Check if subtable already exists
                if (table[main_idx] & BC_SUBTABLE) == 0 {
                    // Create new subtable
                    let subtable_start = subtable_next;
                    table[main_idx] = BC_SUBTABLE
                        | ((subtable_start as u32) << 8)
                        | ((MAX_SUBTABLE_BITS as u32) << 4)
                        | (TABLE_BITS as u32);
                    subtable_next += 1 << MAX_SUBTABLE_BITS;
                }

                // Fill subtable entry
                let subtable_start = ((table[main_idx] >> 8) & 0x1FFF) as usize;
                let subtable_idx = (reversed >> TABLE_BITS) as usize;

                let subtable_entry = if symbol < 256 {
                    BC_LITERAL | ((symbol as u32) << 16) | (extra_bits as u32)
                } else if symbol == 256 {
                    BC_EOB | (extra_bits as u32)
                } else if symbol <= 285 {
                    let idx = symbol - 257;
                    let base = LENGTH_BASES[idx] as u32;
                    let extra = LENGTH_EXTRA[idx] as u32;
                    (base << 16) | (extra << 8) | (extra_bits as u32)
                } else {
                    continue;
                };

                let stride = 1usize << extra_bits;
                let mut idx = subtable_idx;
                while idx < (1 << MAX_SUBTABLE_BITS) {
                    table[subtable_start + idx] = subtable_entry;
                    idx += stride;
                }
            }
        }

        table.truncate(subtable_next);
        Some(Self { table, fingerprint })
    }

    /// Decode a single symbol from the bitstream
    #[inline(always)]
    pub fn decode(&self, bitbuf: u64) -> (u16, u8, bool, bool) {
        let entry = self.table[(bitbuf & 0x7FF) as usize];

        // Check for literal (most common)
        if (entry & BC_LITERAL) != 0 {
            let symbol = ((entry >> 16) & 0xFF) as u16;
            let bits = (entry & 0x1F) as u8;
            return (symbol, bits, true, false);
        }

        // Check for EOB
        if (entry & BC_EOB) != 0 {
            let bits = (entry & 0x1F) as u8;
            return (256, bits, false, true);
        }

        // Check for subtable
        if (entry & BC_SUBTABLE) != 0 {
            let subtable_start = ((entry >> 8) & 0x1FFF) as usize;
            let subtable_bits = ((entry >> 4) & 0xF) as usize;
            let main_bits = (entry & 0xF) as usize;

            let sub_idx = ((bitbuf >> main_bits) & ((1 << subtable_bits) - 1)) as usize;
            let sub_entry = self.table[subtable_start + sub_idx];

            if (sub_entry & BC_LITERAL) != 0 {
                let symbol = ((sub_entry >> 16) & 0xFF) as u16;
                let bits = main_bits as u8 + (sub_entry & 0x1F) as u8;
                return (symbol, bits, true, false);
            }

            if (sub_entry & BC_EOB) != 0 {
                let bits = main_bits as u8 + (sub_entry & 0x1F) as u8;
                return (256, bits, false, true);
            }

            // Length from subtable
            let base = ((sub_entry >> 16) & 0x1FF) as u16;
            let bits = main_bits as u8 + (sub_entry & 0x1F) as u8;
            return (base, bits, false, false);
        }

        // Length code
        let base = ((entry >> 16) & 0x1FF) as u16;
        let bits = (entry & 0x1F) as u8;
        (base, bits, false, false)
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

/// Cache of bytecode tables
pub struct BytecodeCache {
    tables: HashMap<TableFingerprint, BytecodeTable>,
    hits: usize,
    misses: usize,
}

impl BytecodeCache {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get_or_build(&mut self, litlen_lens: &[u8]) -> Option<&BytecodeTable> {
        let fingerprint = TableFingerprint::from_litlen_lengths(litlen_lens);

        if self.tables.contains_key(&fingerprint) {
            self.hits += 1;
            return self.tables.get(&fingerprint);
        }

        self.misses += 1;
        let table = BytecodeTable::build(litlen_lens)?;
        self.tables.insert(fingerprint, table);
        self.tables.get(&fingerprint)
    }

    pub fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, rate)
    }
}

impl Default for BytecodeCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_table_build() {
        // Fixed Huffman code lengths
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7; // EOB
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let table = BytecodeTable::build(&litlen_lens);
        assert!(table.is_some());

        let table = table.unwrap();
        eprintln!("[BC] Table size: {} entries", table.table.len());

        // Test decoding some patterns
        // 'e' = 0x65 should be an 8-bit literal
        let (sym, bits, is_lit, is_eob) = table.decode(0x65);
        eprintln!(
            "[BC] Decode 0x65: symbol={}, bits={}, literal={}, eob={}",
            sym, bits, is_lit, is_eob
        );
    }

    #[test]
    fn bench_bytecode_decode() {
        use std::hint::black_box;

        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7;
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let table = BytecodeTable::build(&litlen_lens).unwrap();

        // Benchmark decode speed
        let iterations = 10_000_000;
        let test_patterns: Vec<u64> = (0..1000).map(|i| i * 7919 % 2048).collect();

        let start = std::time::Instant::now();
        let mut total_bits = 0u64;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let (_, bits, _, _) = black_box(&table).decode(black_box(pattern));
                total_bits = total_bits.wrapping_add(bits as u64);
            }
        }
        black_box(total_bits);
        let elapsed = start.elapsed();

        let decodes_per_sec = iterations as f64 / elapsed.as_secs_f64();
        eprintln!("\n[BENCH] Bytecode Decode:");
        eprintln!(
            "[BENCH]   {:.2} M decodes/sec",
            decodes_per_sec / 1_000_000.0
        );
        eprintln!("[BENCH]   Total bits decoded: {}", total_bits);
    }

    #[test]
    fn bench_bytecode_vs_baseline() {
        use crate::libdeflate_entry::LitLenTable;
        use std::hint::black_box;

        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7;
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let bc_table = BytecodeTable::build(&litlen_lens).unwrap();
        let baseline_table = LitLenTable::build(&litlen_lens).unwrap();

        let iterations = 10_000_000;
        let test_patterns: Vec<u64> = (0..1000).map(|i| i * 7919 % 2048).collect();

        // Benchmark bytecode
        let start = std::time::Instant::now();
        let mut bc_bits = 0u64;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let (_, bits, _, _) = black_box(&bc_table).decode(black_box(pattern));
                bc_bits = bc_bits.wrapping_add(bits as u64);
            }
        }
        black_box(bc_bits);
        let bc_elapsed = start.elapsed();
        let bc_rate = iterations as f64 / bc_elapsed.as_secs_f64() / 1_000_000.0;

        // Benchmark baseline
        let start = std::time::Instant::now();
        let mut baseline_bits = 0u64;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let entry = black_box(&baseline_table).lookup(black_box(pattern));
                baseline_bits = baseline_bits.wrapping_add(entry.total_bits() as u64);
            }
        }
        black_box(baseline_bits);
        let baseline_elapsed = start.elapsed();
        let baseline_rate = iterations as f64 / baseline_elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("\n[BENCH] Bytecode vs Baseline:");
        eprintln!("[BENCH]   Bytecode:  {:.2} M decodes/sec", bc_rate);
        eprintln!("[BENCH]   Baseline:  {:.2} M decodes/sec", baseline_rate);
        eprintln!(
            "[BENCH]   Ratio:     {:.1}%",
            bc_rate / baseline_rate * 100.0
        );
    }
}
