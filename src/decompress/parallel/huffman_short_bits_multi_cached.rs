//! Literal port of `rapidgzip::deflate::HuffmanCodingShortBitsMultiCached`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingShortBitsMultiCached.hpp`).
//!
//! Smaller-LUT variant of the deflate literal/length cached decoder. Limits
//! the LUT to a fixed bit count (the vendor uses this for bzip2 where
//! `MAX_CODE_LENGTH` is 20 and a full LUT would be too large). For deflate
//! the same shape is useful when memory pressure dominates.
//!
//! The `CacheEntry` is a bit-packed `u32` carrying:
//! - 1 bit:  needToReadDistanceBits
//! - 6 bits: bitsToSkip (ceil(log2(2*MAX_CODE_LENGTH(20))) = 6)
//! - 2 bits: symbolCount
//! - 18 bits: packed symbols (one 8-9-bit symbol, or two 8-9-bit symbols)
//!
//! Decoding returns `(packed_symbols, symbol_count)` — the multi-symbol
//! decoder. `DISTANCE_OFFSET = 254` is added to length values so the
//! caller can tell literals (symbol <= 256) from lengths (>= 258 after
//! offset, encoded as `symbol - DISTANCE_OFFSET` for lookup).

#![allow(dead_code)]

use super::bit_manipulation::n_lowest_bits_set;
use super::bit_manipulation::reverse_bits_count_u16;
use super::error::Error;
use super::gzip_definitions::{BYTE_SIZE, END_OF_BLOCK_SYMBOL, MAX_LITERAL_HUFFMAN_CODE_COUNT};
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};
use super::rfc_tables::{calculate_length, length_lookup};

/// `DISTANCE_OFFSET = 254U` per HuffmanCodingShortBitsMultiCached.hpp:52.
pub const DISTANCE_OFFSET: u16 = 254;

/// Bit-packed cache entry — mirror of `CacheEntry`
/// (HuffmanCodingShortBitsMultiCached.hpp:37-48). Stored as `u32` to
/// match the vendor's `static_assert(sizeof(CacheEntry) == 4)`.
///
/// Layout (LSB → MSB):
/// - bit 0:        needToReadDistanceBits
/// - bits 1..6:    bitsToSkip (6 bits)
/// - bits 7..8:    symbolCount (2 bits)
/// - bits 9..26:   symbols (18 bits)
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct CacheEntry(pub u32);

impl CacheEntry {
    pub const fn pack(
        need_distance: bool,
        bits_to_skip: u8,
        symbol_count: u8,
        symbols: u32,
    ) -> Self {
        let mut v = 0u32;
        v |= need_distance as u32;
        v |= ((bits_to_skip as u32) & 0x3F) << 1;
        v |= ((symbol_count as u32) & 0x3) << 7;
        v |= (symbols & ((1 << 18) - 1)) << 9;
        Self(v)
    }

    pub const fn need_distance(&self) -> bool {
        (self.0 & 1) != 0
    }

    pub const fn bits_to_skip(&self) -> u8 {
        ((self.0 >> 1) & 0x3F) as u8
    }

    pub const fn symbol_count(&self) -> u8 {
        ((self.0 >> 7) & 0x3) as u8
    }

    pub const fn symbols(&self) -> u32 {
        (self.0 >> 9) & ((1 << 18) - 1)
    }
}

/// Result of decode: packed symbols + symbol count.
/// Mirror of `Symbols = std::pair<uint32_t, uint32_t>`
/// (HuffmanCodingShortBitsMultiCached.hpp:50).
pub type Symbols = (u32, u32);

/// Mirror of `rapidgzip::deflate::HuffmanCodingShortBitsMultiCached`
/// (HuffmanCodingShortBitsMultiCached.hpp:24-268).
///
/// See `huffman_short_bits_cached_deflate` for the `LUT_SIZE` workaround.
pub struct HuffmanCodingShortBitsMultiCached<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize> {
    pub base: HuffmanCodingSymbolsPerLength<{ MAX_LITERAL_HUFFMAN_CODE_COUNT }>,
    pub code_cache: Box<[CacheEntry; LUT_SIZE]>,
    pub lut_bits_count: u8,
    pub bits_to_read_at_once: u8,
    needs_to_be_zeroed: bool,
}

impl<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize>
    HuffmanCodingShortBitsMultiCached<LUT_BITS_COUNT, LUT_SIZE>
{
    const _LUT_SIZE_OK: () = assert!(
        LUT_SIZE == 1usize << LUT_BITS_COUNT,
        "LUT_SIZE must equal 1 << LUT_BITS_COUNT"
    );

    pub fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: Box::new([CacheEntry::default(); LUT_SIZE]),
            lut_bits_count: LUT_BITS_COUNT,
            bits_to_read_at_once: LUT_BITS_COUNT,
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of `initializeFromLengths`
    /// (HuffmanCodingShortBitsMultiCached.hpp:55-129).
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }
        self.lut_bits_count = core::cmp::min(LUT_BITS_COUNT, self.base.base.max_code_length);
        self.bits_to_read_at_once = core::cmp::max(LUT_BITS_COUNT, self.base.base.min_code_length);

        if self.needs_to_be_zeroed {
            for slot in self.code_cache.iter_mut() {
                // Zero bits_to_skip by zeroing the whole entry.
                *slot = CacheEntry(0);
            }
        }

        // Build flat Huffman table for symbols with length <= lut bits.
        struct HuffmanEntry {
            reversed_code: u16,
            symbol: Symbol,
            length: u8,
        }
        if code_lengths.len() > MAX_LITERAL_HUFFMAN_CODE_COUNT {
            return Error::InvalidCodeLengths;
        }
        let mut huffman_table: Vec<HuffmanEntry> = Vec::with_capacity(code_lengths.len());

        let min_cl = self.base.base.min_code_length;
        let mut code_values = self.base.base.minimum_code_values_per_level;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length == 0 || length > self.lut_bits_count {
                continue;
            }
            let k = (length - min_cl) as usize;
            let code = code_values[k];
            code_values[k] = code.wrapping_add(1);
            let reversed_code = reverse_bits_count_u16(code, length);
            huffman_table.push(HuffmanEntry {
                reversed_code,
                symbol: symbol as Symbol,
                length,
            });
        }

        // Fill cache.
        for he in &huffman_table {
            let need_dist = he.symbol > END_OF_BLOCK_SYMBOL;
            let entry = CacheEntry::pack(need_dist, he.length, 1, he.symbol as u32);
            if need_dist {
                self.insert_length_symbol_into_cache(he.reversed_code, entry);
            } else {
                self.insert_into_cache(he.reversed_code, entry);
            }
        }
        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `insertIntoCache`
    /// (HuffmanCodingShortBitsMultiCached.hpp:199-216).
    fn insert_into_cache(&mut self, reversed_code: u16, entry: CacheEntry) {
        let length = entry.bits_to_skip();
        if length > self.lut_bits_count {
            return;
        }
        let filler_bit_count = self.lut_bits_count - length;
        let maximum_padded_code =
            reversed_code | ((n_lowest_bits_set(filler_bit_count) as u16) << length);
        debug_assert!((maximum_padded_code as usize) < LUT_SIZE);
        let increment: u16 = 1u16 << length;
        let mut padded_code = reversed_code;
        loop {
            self.code_cache[padded_code as usize] = entry;
            if padded_code == maximum_padded_code {
                break;
            }
            padded_code = padded_code.wrapping_add(increment);
        }
    }

    /// Mirror of `insertLengthSymbolIntoCache`
    /// (HuffmanCodingShortBitsMultiCached.hpp:218-260).
    fn insert_length_symbol_into_cache(&mut self, reversed_code: u16, input_entry: CacheEntry) {
        if !input_entry.need_distance() {
            self.insert_into_cache(reversed_code, input_entry);
            return;
        }
        let previous_bit_count = (input_entry.symbol_count() as u32 - 1) * BYTE_SIZE;
        let symbol = (input_entry.symbols()) >> previous_bit_count;
        let code_length = input_entry.bits_to_skip();
        let previous_symbols = symbol & n_lowest_bits_set(previous_bit_count as u8) as u32;
        let prepend_length = |length: u32| previous_symbols | (length << previous_bit_count);

        if symbol <= 264 {
            let val = prepend_length(symbol - 257 + 3 + DISTANCE_OFFSET as u32);
            let entry = CacheEntry::pack(false, code_length, input_entry.symbol_count(), val);
            self.insert_into_cache(reversed_code, entry);
        } else if symbol < 285 {
            let length_code = (symbol - 261) as u8;
            let extra_bit_count = length_code / 4;
            if code_length + extra_bit_count <= self.lut_bits_count {
                for extra_bits in 0..(1u16 << extra_bit_count) {
                    let val = prepend_length(
                        calculate_length(length_code as u16) as u32
                            + extra_bits as u32
                            + DISTANCE_OFFSET as u32,
                    );
                    let entry = CacheEntry::pack(
                        false,
                        code_length + extra_bit_count,
                        input_entry.symbol_count(),
                        val,
                    );
                    let rc = reversed_code | (extra_bits << code_length);
                    self.insert_into_cache(rc, entry);
                }
            } else {
                let val = prepend_length(symbol - 254 + DISTANCE_OFFSET as u32);
                let entry = CacheEntry::pack(
                    input_entry.need_distance(),
                    code_length,
                    input_entry.symbol_count(),
                    val,
                );
                self.insert_into_cache(reversed_code, entry);
            }
        } else if symbol == 285 {
            let val = prepend_length(258 + DISTANCE_OFFSET as u32);
            let entry = CacheEntry::pack(false, code_length, input_entry.symbol_count(), val);
            self.insert_into_cache(reversed_code, entry);
        } else {
            // Vendor throws std::logic_error; surface as a debug panic in tests
            // and otherwise leave the cache unmodified.
            debug_assert!(false, "Symbol count or symbols bit field is inconsistent!");
        }
    }

    /// Mirror of `readLength`
    /// (HuffmanCodingShortBitsMultiCached.hpp:188-197).
    #[inline]
    fn read_length<R: LsbBitReader>(&self, symbol: Symbol, bit_reader: &mut R) -> u32 {
        if symbol <= 256 {
            return symbol as u32;
        }
        // getLength + DISTANCE_OFFSET.
        let look = match length_lookup(symbol) {
            Some(l) => l,
            None => return 0,
        };
        let extra = if look.extra_bits_count > 0 {
            bit_reader.read(look.extra_bits_count).unwrap_or(0) as u16
        } else {
            0
        };
        (look.base_length + extra) as u32 + DISTANCE_OFFSET as u32
    }

    /// Mirror of `decodeLong`
    /// (HuffmanCodingShortBitsMultiCached.hpp:159-185).
    fn decode_long<R: LsbBitReader>(&self, bit_reader: &mut R) -> Symbols {
        let mut code: u16 = 0;
        let min_cl = self.base.base.min_code_length;
        let max_cl = self.base.base.max_code_length;
        let bits_at_once = self.bits_to_read_at_once;
        for _ in 0..bits_at_once {
            let bit = bit_reader.read_one().unwrap_or(0);
            code = (code << 1) | bit as u16;
        }
        // k starts at bits_at_once - min_cl.
        let mut k = bits_at_once - min_cl;
        let k_end = max_cl - min_cl;
        loop {
            let min_code = self.base.base.minimum_code_values_per_level[k as usize];
            if min_code <= code {
                let sub_index = self.base.offsets[k as usize] as usize + (code - min_code) as usize;
                if sub_index < self.base.offsets[(k + 1) as usize] as usize {
                    let s = self.base.symbols_per_length[sub_index];
                    return (self.read_length(s, bit_reader), 1);
                }
            }
            if k == k_end {
                break;
            }
            let bit = bit_reader.read_one().unwrap_or(0);
            code = (code << 1) | bit as u16;
            k += 1;
        }
        (0, 0)
    }

    /// Mirror of `decode<BitReader>`
    /// (HuffmanCodingShortBitsMultiCached.hpp:131-156).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Symbols {
        match bit_reader.peek(self.lut_bits_count) {
            Ok(idx) => {
                let entry = self.code_cache[idx as usize];
                if entry.bits_to_skip() == 0 {
                    return self.decode_long(bit_reader);
                }
                bit_reader.seek_after_peek(entry.bits_to_skip());
                let s = if entry.need_distance() {
                    self.read_length(entry.symbols() as Symbol, bit_reader)
                } else {
                    entry.symbols()
                };
                (s, entry.symbol_count() as u32)
            }
            Err(_) => {
                let result = self.base.decode(bit_reader);
                match result {
                    Some(r) => (self.read_length(r, bit_reader), 1),
                    None => (0, 0),
                }
            }
        }
    }
}

impl<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize> Default
    for HuffmanCodingShortBitsMultiCached<LUT_BITS_COUNT, LUT_SIZE>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::block_finder::BitReader;

    /// Decode literal 0 via the LUT (fixed tree).
    #[test]
    fn fixed_tree_decodes_literal() {
        let mut lens = vec![0u8; 288];
        for v in &mut lens[0..144] {
            *v = 8;
        }
        for v in &mut lens[144..256] {
            *v = 9;
        }
        for v in &mut lens[256..280] {
            *v = 7;
        }
        for v in &mut lens[280..288] {
            *v = 8;
        }
        let mut hc: HuffmanCodingShortBitsMultiCached<9, 512> =
            HuffmanCodingShortBitsMultiCached::new();
        let err = hc.initialize_from_lengths(&lens);
        assert_eq!(err, Error::None);

        let data = [0b0000_1100u8, 0u8];
        let mut br = BitReader::new(&data);
        let (sym, cnt) = hc.decode(&mut br);
        assert_eq!(cnt, 1);
        assert_eq!(sym, 0);
    }
}
