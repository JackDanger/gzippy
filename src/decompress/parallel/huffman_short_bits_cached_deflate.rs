#![allow(dead_code)]

//! Port of `rapidgzip::HuffmanCodingShortBitsCachedDeflate<11>`
//! (`vendor/.../huffman/HuffmanCodingShortBitsCachedDeflate.hpp:22-280`).

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::{END_OF_BLOCK_SYMBOL, MAX_LITERAL_HUFFMAN_CODE_COUNT};
use super::huffman_base::LsbBitReader;
use super::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};
use super::rfc_tables::{
    calculate_length, get_distance_dynamic_canonical, get_length_minus3, DISTANCE_LUT,
};

pub const LUT_BITS_COUNT: u8 = 11;
const CACHE_LEN: usize = 1 << LUT_BITS_COUNT as usize;

pub const DISTANCE_PARTIAL: u16 = 0xFFFE;
pub const DISTANCE_EOB: u16 = 0xFFFF;

/// Mirror of `CacheEntry` (ShortBitsCachedDeflate.hpp:35-40).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheEntry {
    pub bits_to_skip: u8,
    pub symbol_or_length: u8,
    pub distance: u16,
}

/// Mirror of `HuffmanCodingShortBitsCachedDeflate<11>`.
pub struct HuffmanCodingShortBitsCachedDeflate {
    base: HuffmanCodingSymbolsPerLength<MAX_LITERAL_HUFFMAN_CODE_COUNT>,
    code_cache: [CacheEntry; CACHE_LEN],
    lut_bits_count: u8,
    bits_to_read_at_once: u8,
    needs_to_be_zeroed: bool,
}

impl HuffmanCodingShortBitsCachedDeflate {
    pub const fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: [CacheEntry {
                bits_to_skip: 0,
                symbol_or_length: 0,
                distance: 0,
            }; CACHE_LEN],
            lut_bits_count: LUT_BITS_COUNT,
            bits_to_read_at_once: LUT_BITS_COUNT,
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of `initializeFromLengths` (ShortBitsCachedDeflate.hpp:44-118).
    pub fn initialize_from_lengths<const MAX_DIST: usize>(
        &mut self,
        code_lengths: &[u8],
        distance_hc: &HuffmanCodingReversedBitsCached<MAX_DIST>,
    ) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }

        self.lut_bits_count = LUT_BITS_COUNT;
        self.bits_to_read_at_once = LUT_BITS_COUNT.max(self.base.base.min_code_length);

        if self.needs_to_be_zeroed {
            for entry in &mut self.code_cache {
                entry.bits_to_skip = 0;
            }
        }

        let mut code_values = self.base.base.minimum_code_values_per_level;

        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length == 0 || length > self.lut_bits_count {
                continue;
            }

            let k = (length - self.base.base.min_code_length) as usize;
            let reversed_code = reverse_bits_count_u16(code_values[k], length);
            code_values[k] = code_values[k].wrapping_add(1);

            let mut cache_entry = CacheEntry {
                bits_to_skip: length,
                symbol_or_length: 0,
                distance: 0,
            };

            if symbol <= 255 {
                cache_entry.symbol_or_length = symbol as u8;
                cache_entry.distance = 0;
                self.insert_into_cache(reversed_code, cache_entry);
            } else if symbol == END_OF_BLOCK_SYMBOL as usize {
                cache_entry.distance = DISTANCE_EOB;
                self.insert_into_cache(reversed_code, cache_entry);
            } else if symbol <= 264 {
                cache_entry.symbol_or_length = (symbol - 257) as u8;
                self.insert_into_cache_with_distance(
                    reversed_code,
                    cache_entry,
                    distance_hc,
                    (symbol - 257) as u8,
                    cache_entry.bits_to_skip,
                );
            } else if symbol < 285 {
                let length_code = (symbol - 261) as u8;
                let extra_bit_count = length_code / 4;
                #[allow(clippy::int_plus_one)] // vendor ShortBitsCachedDeflate.hpp:92
                if length + extra_bit_count + 1 <= self.lut_bits_count {
                    cache_entry.bits_to_skip += extra_bit_count;
                    for extra_bits in 0..(1u8 << extra_bit_count) {
                        cache_entry.symbol_or_length = (calculate_length(length_code as u16)
                            + u16::from(extra_bits)
                            - 3) as u8;
                        self.insert_into_cache_with_distance(
                            reversed_code | (u16::from(extra_bits) << length),
                            cache_entry,
                            distance_hc,
                            (symbol - 257) as u8,
                            cache_entry.bits_to_skip - extra_bit_count,
                        );
                    }
                } else {
                    cache_entry.symbol_or_length = (symbol - 257) as u8;
                    cache_entry.distance = DISTANCE_PARTIAL;
                    self.insert_into_cache(reversed_code, cache_entry);
                }
            } else if symbol == 285 {
                cache_entry.symbol_or_length = (258 - 3) as u8;
                self.insert_into_cache_with_distance(
                    reversed_code,
                    cache_entry,
                    distance_hc,
                    (symbol - 257) as u8,
                    cache_entry.bits_to_skip,
                );
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `decode` (ShortBitsCachedDeflate.hpp:120-141).
    #[inline]
    pub fn decode<R: LsbBitReader, const MAX_DIST: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingSymbolsPerLength<MAX_DIST>,
    ) -> CacheEntry {
        let peek = match bit_reader.peek(self.lut_bits_count) {
            Ok(v) => v as usize,
            Err(_) => {
                let sym = self.base.decode(bit_reader).unwrap_or(0);
                return self.interpret_symbol(bit_reader, distance_hc, sym);
            }
        };

        let cache_entry = self.code_cache[peek];
        if cache_entry.bits_to_skip == 0 {
            return self.decode_long(bit_reader, distance_hc);
        }

        bit_reader.seek_after_peek(cache_entry.bits_to_skip);
        if cache_entry.distance == DISTANCE_PARTIAL {
            return self.interpret_symbol(
                bit_reader,
                distance_hc,
                cache_entry.symbol_or_length as Symbol + 257,
            );
        }
        cache_entry
    }

    fn decode_long<R: LsbBitReader, const MAX_DIST: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingSymbolsPerLength<MAX_DIST>,
    ) -> CacheEntry {
        let base = &self.base.base;
        let mut code: u16 = 0;

        for _ in 0..base.min_code_length {
            let bit = match bit_reader.read_one() {
                Ok(b) => b,
                Err(_) => return CacheEntry::default(),
            };
            code = (code << 1) | bit as u16;
        }

        let levels = base.max_code_length - base.min_code_length;
        for k in 0..=levels {
            let min_code = base.minimum_code_values_per_level[k as usize];
            if min_code <= code {
                let sub_index = self.base.offsets[k as usize] as usize + (code - min_code) as usize;
                if sub_index < self.base.offsets[(k + 1) as usize] as usize {
                    return self.interpret_symbol(
                        bit_reader,
                        distance_hc,
                        self.base.symbols_per_length[sub_index],
                    );
                }
            }
            let bit = match bit_reader.read_one() {
                Ok(b) => b,
                Err(_) => return CacheEntry::default(),
            };
            code = (code << 1) | bit as u16;
        }
        CacheEntry::default()
    }

    fn interpret_symbol<R: LsbBitReader, const MAX_DIST: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingSymbolsPerLength<MAX_DIST>,
        symbol: Symbol,
    ) -> CacheEntry {
        if symbol <= 255 {
            return CacheEntry {
                bits_to_skip: 0,
                symbol_or_length: symbol as u8,
                distance: 0,
            };
        }
        if symbol == END_OF_BLOCK_SYMBOL {
            return CacheEntry {
                bits_to_skip: 0,
                symbol_or_length: 0,
                distance: DISTANCE_EOB,
            };
        }
        if symbol > 285 {
            return CacheEntry::default();
        }

        let symbol_or_length = match get_length_minus3(symbol, bit_reader) {
            Ok(v) => v,
            Err(_) => return CacheEntry::default(),
        };
        let distance = match get_distance_dynamic_canonical(distance_hc, bit_reader) {
            Ok(d) => d,
            Err(_) => return CacheEntry::default(),
        };
        CacheEntry {
            bits_to_skip: 0,
            symbol_or_length,
            distance,
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
        debug_assert!((maximum_padded_code as usize) < CACHE_LEN);
        let increment = 1u32 << length;
        let mut padded_code = reversed_code as u32;
        while padded_code <= maximum_padded_code {
            self.code_cache[padded_code as usize] = cache_entry;
            padded_code += increment;
        }
    }

    fn insert_into_cache_with_distance<const MAX_DIST: usize>(
        &mut self,
        reversed_code: u16,
        cache_entry: CacheEntry,
        distance_hc: &HuffmanCodingReversedBitsCached<MAX_DIST>,
        length_symbol: u8,
        bits_to_skip_without_distance: u8,
    ) {
        let length = cache_entry.bits_to_skip;
        if length > self.lut_bits_count {
            return;
        }
        let filler_bit_count = self.lut_bits_count - length;
        let dist_cache = distance_hc.code_cache();

        let maximum_padded_code =
            reversed_code as u32 | ((n_lowest_bits_set(filler_bit_count) as u32) << length);
        let increment = 1u32 << length;
        let mut padded_code = reversed_code as u32;
        while padded_code <= maximum_padded_code {
            let free_bits =
                (padded_code >> length) & n_lowest_bits_set(distance_hc.max_code_length()) as u32;
            let (distance_code_length, symbol) = dist_cache[free_bits as usize];
            if distance_code_length == 0 || distance_code_length > filler_bit_count || symbol > 29 {
                self.code_cache[padded_code as usize] = cache_entry;
                self.code_cache[padded_code as usize].bits_to_skip = bits_to_skip_without_distance;
                self.code_cache[padded_code as usize].symbol_or_length = length_symbol;
                self.code_cache[padded_code as usize].distance = DISTANCE_PARTIAL;
            } else if symbol <= 3 {
                self.code_cache[padded_code as usize] = cache_entry;
                self.code_cache[padded_code as usize].bits_to_skip = length + distance_code_length;
                self.code_cache[padded_code as usize].distance = symbol + 1;
            } else {
                let extra_bit_count = ((symbol - 2) / 2) as u8;
                if distance_code_length + extra_bit_count <= filler_bit_count {
                    let extra_bits = (padded_code >> (length + distance_code_length))
                        & n_lowest_bits_set(extra_bit_count) as u32;
                    self.code_cache[padded_code as usize] = cache_entry;
                    self.code_cache[padded_code as usize].bits_to_skip =
                        length + distance_code_length + extra_bit_count;
                    self.code_cache[padded_code as usize].distance =
                        DISTANCE_LUT[symbol as usize] + extra_bits as u16;
                } else {
                    self.code_cache[padded_code as usize] = cache_entry;
                    self.code_cache[padded_code as usize].bits_to_skip =
                        bits_to_skip_without_distance;
                    self.code_cache[padded_code as usize].symbol_or_length = length_symbol;
                    self.code_cache[padded_code as usize].distance = DISTANCE_PARTIAL;
                }
            }
            padded_code += increment;
        }
    }
}

impl Default for HuffmanCodingShortBitsCachedDeflate {
    fn default() -> Self {
        Self::new()
    }
}
