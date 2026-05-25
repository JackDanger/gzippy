#![allow(dead_code)]

//! Port of `rapidgzip::HuffmanCodingShortBitsMultiCached<11>`
//! (`vendor/.../huffman/HuffmanCodingShortBitsMultiCached.hpp:24-269`).

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::{BYTE_SIZE, END_OF_BLOCK_SYMBOL, MAX_LITERAL_HUFFMAN_CODE_COUNT};
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};
use super::rfc_tables::{calculate_length, get_length};

/// Vendor's `HuffmanCodingDoubleLiteralCached` uses an adaptive cached
/// bit count `min(max(maxCodeLength, 2*minCodeLength+1), MAX_CODE_LENGTH)`
/// up to 15 bits. gzippy's MultiCached previously was a fixed 11-bit LUT
/// — too small to absorb typical maxCodeLength of 12-15 in silesia
/// literal codings, so the `decode_long` fallback fires often. Bumping
/// to 12 increases LUT size from 16 KB to 32 KB (still L1 on most x86)
/// and reduces fallback rate without doubling memory pressure.
/// Cite: vendor HuffmanCodingDoubleLiteralCached.hpp:89-91.
pub const LUT_BITS_COUNT: u8 = 12;
const CACHE_LEN: usize = 1 << LUT_BITS_COUNT as usize;

/// Packed literal/length symbols use `254 + length` for match entries
/// (vendor `DISTANCE_OFFSET = 254U`, MultiCached.hpp:52).
pub const MULTI_DISTANCE_OFFSET: u32 = 254;

/// Mirror of vendor `Symbols` pair.
pub type DecodedSymbols = (u32, u32);

/// Mirror of `CacheEntry` (MultiCached.hpp:37-47).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CacheEntry {
    need_to_read_distance_bits: bool,
    bits_to_skip: u8,
    symbol_count: u8,
    symbols: u32,
}

/// Mirror of `HuffmanCodingShortBitsMultiCached<11>`.
pub struct HuffmanCodingShortBitsMultiCached {
    base: HuffmanCodingSymbolsPerLength<MAX_LITERAL_HUFFMAN_CODE_COUNT>,
    code_cache: [CacheEntry; CACHE_LEN],
    lut_bits_count: u8,
    bits_to_read_at_once: u8,
    needs_to_be_zeroed: bool,
}

#[derive(Clone, Copy)]
struct HuffmanEntry {
    reversed_code: u16,
    symbol: Symbol,
    length: u8,
}

impl HuffmanCodingShortBitsMultiCached {
    pub const fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: [CacheEntry {
                need_to_read_distance_bits: false,
                bits_to_skip: 0,
                symbol_count: 0,
                symbols: 0,
            }; CACHE_LEN],
            lut_bits_count: LUT_BITS_COUNT,
            bits_to_read_at_once: LUT_BITS_COUNT,
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of `initializeFromLengths` (MultiCached.hpp:55-129).
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }

        self.lut_bits_count = LUT_BITS_COUNT.min(self.base.base.max_code_length);
        self.bits_to_read_at_once = LUT_BITS_COUNT.max(self.base.base.min_code_length);

        if self.needs_to_be_zeroed {
            for entry in &mut self.code_cache {
                entry.bits_to_skip = 0;
            }
        }

        let mut huffman_table = [HuffmanEntry {
            reversed_code: 0,
            symbol: 0,
            length: 0,
        }; MAX_LITERAL_HUFFMAN_CODE_COUNT];
        let mut huffman_table_size = 0usize;

        if code_lengths.len() > huffman_table.len() {
            return Error::InvalidCodeLengths;
        }

        let mut code_values = self.base.base.minimum_code_values_per_level;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length == 0 || length > self.lut_bits_count {
                continue;
            }
            let entry = &mut huffman_table[huffman_table_size];
            huffman_table_size += 1;
            entry.length = length;
            entry.symbol = symbol as Symbol;
            entry.reversed_code = reverse_bits_count_u16(
                code_values[(length - self.base.base.min_code_length) as usize],
                length,
            );
            code_values[(length - self.base.base.min_code_length) as usize] =
                code_values[(length - self.base.base.min_code_length) as usize].wrapping_add(1);
        }

        for huffman_entry in &huffman_table[..huffman_table_size] {
            let cache_entry = CacheEntry {
                bits_to_skip: huffman_entry.length,
                symbols: huffman_entry.symbol as u32,
                symbol_count: 1,
                need_to_read_distance_bits: huffman_entry.symbol > END_OF_BLOCK_SYMBOL,
            };

            if cache_entry.need_to_read_distance_bits {
                self.insert_length_symbol_into_cache(huffman_entry.reversed_code, cache_entry);
            } else {
                self.insert_into_cache(huffman_entry.reversed_code, cache_entry);
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `decode` (MultiCached.hpp:131-156).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> DecodedSymbols {
        match bit_reader.peek(self.lut_bits_count) {
            Ok(peek) => {
                let cache_entry = self.code_cache[peek as usize];
                if cache_entry.bits_to_skip == 0 {
                    return self.decode_long(bit_reader);
                }
                bit_reader.seek_after_peek(cache_entry.bits_to_skip);
                if cache_entry.need_to_read_distance_bits {
                    (
                        self.read_length(cache_entry.symbols as Symbol, bit_reader),
                        cache_entry.symbol_count as u32,
                    )
                } else {
                    (cache_entry.symbols, cache_entry.symbol_count as u32)
                }
            }
            Err(_) => {
                if let Some(sym) = self.base.decode(bit_reader) {
                    (self.read_length(sym, bit_reader), 1)
                } else {
                    (0, 0)
                }
            }
        }
    }

    fn decode_long<R: LsbBitReader>(&self, bit_reader: &mut R) -> DecodedSymbols {
        let base = &self.base.base;
        let mut code: u16 = 0;

        for _ in 0..self.bits_to_read_at_once {
            let bit = match bit_reader.read_one() {
                Ok(b) => b,
                Err(_) => return (0, 0),
            };
            code = (code << 1) | bit as u16;
        }

        for k in (self.bits_to_read_at_once - base.min_code_length)
            ..=(base.max_code_length - base.min_code_length)
        {
            let min_code = base.minimum_code_values_per_level[k as usize];
            if min_code <= code {
                let sub_index = self.base.offsets[k as usize] as usize + (code - min_code) as usize;
                if sub_index < self.base.offsets[(k + 1) as usize] as usize {
                    return (
                        self.read_length(self.base.symbols_per_length[sub_index], bit_reader),
                        1,
                    );
                }
            }
            let bit = match bit_reader.read_one() {
                Ok(b) => b,
                Err(_) => return (0, 0),
            };
            code = (code << 1) | bit as u16;
        }
        (0, 0)
    }

    fn read_length<R: LsbBitReader>(&self, symbol: Symbol, bit_reader: &mut R) -> u32 {
        if symbol <= 256 {
            return symbol as u32;
        }
        match get_length(symbol, bit_reader) {
            Ok(len) => len as u32 + MULTI_DISTANCE_OFFSET,
            Err(_) => 0,
        }
    }

    fn insert_into_cache(&mut self, reversed_code: u16, cache_entry: CacheEntry) {
        let length = cache_entry.bits_to_skip;
        if length > self.lut_bits_count {
            return;
        }
        let filler_bit_count = self.lut_bits_count - length;
        let maximum_padded_code =
            reversed_code as u32 | ((n_lowest_bits_set(filler_bit_count) as u32) << length);
        let increment = 1u32 << length;
        let mut padded_code = reversed_code as u32;
        while padded_code <= maximum_padded_code {
            self.code_cache[padded_code as usize] = cache_entry;
            padded_code += increment;
        }
    }

    fn insert_length_symbol_into_cache(&mut self, reversed_code: u16, input: CacheEntry) {
        if !input.need_to_read_distance_bits {
            self.insert_into_cache(reversed_code, input);
            return;
        }

        let previous_bit_count = (input.symbol_count - 1) as u32 * BYTE_SIZE;
        let symbol = input.symbols >> previous_bit_count;
        let code_length = input.bits_to_skip;
        let previous_symbols = symbol & n_lowest_bits_set(previous_bit_count as u8) as u32;
        let prepend = |length: u32| previous_symbols | (length << previous_bit_count);

        let mut cache_entry = input;
        if symbol <= 264 {
            cache_entry.symbols = prepend(symbol - 257 + 3 + MULTI_DISTANCE_OFFSET);
            cache_entry.need_to_read_distance_bits = false;
            self.insert_into_cache(reversed_code, cache_entry);
        } else if symbol < 285 {
            let length_code = (symbol - 261) as u8;
            let extra_bit_count = length_code / 4;
            if code_length + extra_bit_count <= self.lut_bits_count {
                cache_entry.need_to_read_distance_bits = false;
                cache_entry.bits_to_skip = code_length + extra_bit_count;
                for extra_bits in 0..(1u8 << extra_bit_count) {
                    cache_entry.symbols = prepend(
                        calculate_length(length_code as u16) as u32
                            + u32::from(extra_bits)
                            + MULTI_DISTANCE_OFFSET,
                    );
                    self.insert_into_cache(
                        reversed_code | (u16::from(extra_bits) << code_length),
                        cache_entry,
                    );
                }
            } else {
                cache_entry.symbols = prepend(symbol);
                self.insert_into_cache(reversed_code, cache_entry);
            }
        } else if symbol == 285 {
            cache_entry.need_to_read_distance_bits = false;
            cache_entry.symbols = prepend(258 + MULTI_DISTANCE_OFFSET);
            self.insert_into_cache(reversed_code, cache_entry);
        }
    }
}

impl Default for HuffmanCodingShortBitsMultiCached {
    fn default() -> Self {
        Self::new()
    }
}
