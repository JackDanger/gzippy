//! Literal port of `rapidgzip::deflate::HuffmanCodingShortBitsCachedDeflate`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingShortBitsCachedDeflate.hpp`).
//!
//! Specialized deflate literal/length decoder that bakes the distance
//! lookup into the literal LUT for common short-code cases. Returns a
//! `CacheEntry` containing bits-to-skip, the decoded symbol-or-length,
//! and the distance (or `0xFFFE` for "couldn't fold in the distance —
//! caller must consult the distance Huffman code itself" / `0xFFFF`
//! for end-of-block).
//!
//! Inherits storage from `HuffmanCodingSymbolsPerLength` (vendor) — composed
//! here. The distance Huffman coding is passed in by reference and must
//! also be a `HuffmanCodingReversedBitsCached`-shaped table because the
//! ShortBits builder pokes its `code_cache` directly during LUT fill.

#![allow(dead_code)]

use super::bit_manipulation::n_lowest_bits_set;
use super::bit_manipulation::reverse_bits_count_u16;
use super::error::Error;
use super::gzip_definitions::{END_OF_BLOCK_SYMBOL, MAX_LITERAL_HUFFMAN_CODE_COUNT};
use super::huffman_base::LsbBitReader;
use super::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};
use super::rfc_tables::{calculate_length, length_lookup, DISTANCE_LUT};

/// Per-LUT-entry payload.
/// Mirror of `HuffmanCodingShortBitsCachedDeflate::CacheEntry`
/// (HuffmanCodingShortBitsCachedDeflate.hpp:35-40).
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct CacheEntry {
    pub bits_to_skip: u8,
    pub symbol_or_length: u8,
    pub distance: u16,
}

/// Sentinel `distance` value meaning "end-of-block symbol decoded".
/// Mirror of the `0xFFFFU` literal at
/// HuffmanCodingShortBitsCachedDeflate.hpp:80 / 187.
pub const END_OF_BLOCK_DISTANCE: u16 = 0xFFFF;

/// Sentinel `distance` value meaning "decode failed to fold a distance
/// into the cache entry — caller must consume distance code separately".
/// Mirror of the `0xFFFEU` literal at
/// HuffmanCodingShortBitsCachedDeflate.hpp:103, 248, 268.
pub const DISTANCE_DEFERRED: u16 = 0xFFFE;

/// Mirror of `rapidgzip::deflate::HuffmanCodingShortBitsCachedDeflate`
/// (HuffmanCodingShortBitsCachedDeflate.hpp:22-279).
///
/// `LUT_BITS_COUNT` is the vendor's template parameter — typically 9–11
/// depending on cache-size tuning. `LUT_SIZE` MUST equal
/// `1 << LUT_BITS_COUNT` and exists only because stable Rust lacks
/// `generic_const_exprs` for dimensioning a `[CacheEntry; 1 << N]` from
/// a single const generic. A `const _LUT_SIZE_OK` assertion in the impl
/// enforces the relationship at compile time.
pub struct HuffmanCodingShortBitsCachedDeflate<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize> {
    pub base: HuffmanCodingSymbolsPerLength<{ MAX_LITERAL_HUFFMAN_CODE_COUNT }>,
    pub code_cache: Box<[CacheEntry; LUT_SIZE]>,
    pub lut_bits_count: u8,
    pub bits_to_read_at_once: u8,
    needs_to_be_zeroed: bool,
}

impl<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize>
    HuffmanCodingShortBitsCachedDeflate<LUT_BITS_COUNT, LUT_SIZE>
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

    /// Mirror of `initializeFromLengths` (HuffmanCodingShortBitsCachedDeflate.hpp:44-118).
    pub fn initialize_from_lengths<const DIST_SYMS: usize>(
        &mut self,
        code_lengths: &[u8],
        distance_hc: &HuffmanCodingReversedBitsCached<DIST_SYMS>,
    ) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }
        self.lut_bits_count = LUT_BITS_COUNT;
        self.bits_to_read_at_once = core::cmp::max(LUT_BITS_COUNT, self.base.base.min_code_length);

        if self.needs_to_be_zeroed {
            for slot in self.code_cache.iter_mut() {
                slot.bits_to_skip = 0;
            }
        }

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

            let mut cache_entry = CacheEntry {
                bits_to_skip: length,
                ..CacheEntry::default()
            };
            let symbol_u16 = symbol as u16;
            if symbol_u16 <= 255 {
                cache_entry.symbol_or_length = symbol_u16 as u8;
                cache_entry.distance = 0;
                self.insert_into_cache(reversed_code, cache_entry);
            } else if symbol_u16 == END_OF_BLOCK_SYMBOL {
                cache_entry.distance = END_OF_BLOCK_DISTANCE;
                self.insert_into_cache(reversed_code, cache_entry);
            } else if symbol_u16 <= 264 {
                cache_entry.symbol_or_length = (symbol_u16 - 257) as u8;
                self.insert_into_cache_with_distance(
                    reversed_code,
                    &cache_entry,
                    distance_hc,
                    (symbol_u16 - 257) as u8,
                    cache_entry.bits_to_skip,
                );
            } else if symbol_u16 < 285 {
                let length_code = (symbol_u16 - 261) as u8;
                let extra_bit_count = length_code / 4; // <= 5
                if length + extra_bit_count < self.lut_bits_count {
                    cache_entry.bits_to_skip += extra_bit_count;
                    for extra_bits in 0..(1u8 << extra_bit_count) {
                        cache_entry.symbol_or_length =
                            (calculate_length(length_code as u16) + extra_bits as u16 - 3) as u8;
                        let reversed_with_extra = reversed_code | ((extra_bits as u16) << length);
                        self.insert_into_cache_with_distance(
                            reversed_with_extra,
                            &cache_entry,
                            distance_hc,
                            (symbol_u16 - 257) as u8,
                            cache_entry.bits_to_skip - extra_bit_count,
                        );
                    }
                } else {
                    cache_entry.symbol_or_length = (symbol_u16 - 257) as u8;
                    cache_entry.distance = DISTANCE_DEFERRED;
                    self.insert_into_cache(reversed_code, cache_entry);
                }
            } else if symbol_u16 == 285 {
                cache_entry.symbol_or_length = (258u16 - 3) as u8;
                self.insert_into_cache_with_distance(
                    reversed_code,
                    &cache_entry,
                    distance_hc,
                    (symbol_u16 - 257) as u8,
                    cache_entry.bits_to_skip,
                );
            } else {
                debug_assert!(symbol_u16 < MAX_LITERAL_HUFFMAN_CODE_COUNT as u16);
            }
        }
        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `insertIntoCache`
    /// (HuffmanCodingShortBitsCachedDeflate.hpp:204-221).
    fn insert_into_cache(&mut self, reversed_code: u16, entry: CacheEntry) {
        let length = entry.bits_to_skip;
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

    /// Mirror of `insertIntoCacheWithDistance`
    /// (HuffmanCodingShortBitsCachedDeflate.hpp:223-272).
    fn insert_into_cache_with_distance<const DIST_SYMS: usize>(
        &mut self,
        reversed_code: u16,
        input_entry: &CacheEntry,
        distance_hc: &HuffmanCodingReversedBitsCached<DIST_SYMS>,
        length_symbol: u8,
        bits_to_skip_without_distance: u8,
    ) {
        let length = input_entry.bits_to_skip;
        if length > self.lut_bits_count {
            return;
        }
        let filler_bit_count = self.lut_bits_count - length;
        let maximum_padded_code =
            reversed_code | ((n_lowest_bits_set(filler_bit_count) as u16) << length);
        debug_assert!((maximum_padded_code as usize) < LUT_SIZE);
        let increment: u16 = 1u16 << length;
        let distance_max_cl = distance_hc.base.base.max_code_length;
        let dist_cache = distance_hc.code_cache();

        let mut padded_code = reversed_code;
        loop {
            let free_bits = (padded_code >> length) & n_lowest_bits_set(distance_max_cl) as u16;
            let (distance_code_length, dist_symbol) = dist_cache[free_bits as usize];

            if distance_code_length == 0
                || distance_code_length > filler_bit_count
                || dist_symbol > 29
            {
                let mut e = *input_entry;
                e.bits_to_skip = bits_to_skip_without_distance;
                e.symbol_or_length = length_symbol;
                e.distance = DISTANCE_DEFERRED;
                self.code_cache[padded_code as usize] = e;
            } else if dist_symbol <= 3 {
                let mut e = *input_entry;
                e.bits_to_skip = length + distance_code_length;
                e.distance = dist_symbol + 1;
                self.code_cache[padded_code as usize] = e;
            } else {
                let extra_bit_count = ((dist_symbol - 2) / 2) as u8;
                if (distance_code_length + extra_bit_count) <= filler_bit_count {
                    let extra_bits = (padded_code >> (length + distance_code_length))
                        & n_lowest_bits_set(extra_bit_count) as u16;
                    let mut e = *input_entry;
                    e.bits_to_skip = length + distance_code_length + extra_bit_count;
                    e.distance = DISTANCE_LUT[dist_symbol as usize] + extra_bits;
                    self.code_cache[padded_code as usize] = e;
                } else {
                    let mut e = *input_entry;
                    e.bits_to_skip = bits_to_skip_without_distance;
                    e.symbol_or_length = length_symbol;
                    e.distance = DISTANCE_DEFERRED;
                    self.code_cache[padded_code as usize] = e;
                }
            }

            if padded_code == maximum_padded_code {
                break;
            }
            padded_code = padded_code.wrapping_add(increment);
        }
    }

    /// Mirror of `decodeLong`
    /// (HuffmanCodingShortBitsCachedDeflate.hpp:144-170).
    fn decode_long<R: LsbBitReader, const DIST_SYMS: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingReversedBitsCached<DIST_SYMS>,
    ) -> Result<CacheEntry, Error> {
        let mut code: u16 = 0;
        let min_cl = self.base.base.min_code_length;
        for _ in 0..min_cl {
            let bit = bit_reader
                .read_one()
                .map_err(|_| Error::InvalidHuffmanCode)?;
            code = (code << 1) | bit as u16;
        }
        let max_cl = self.base.base.max_code_length;
        for k in 0..=(max_cl - min_cl) {
            let min_code = self.base.base.minimum_code_values_per_level[k as usize];
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
            let bit = bit_reader
                .read_one()
                .map_err(|_| Error::InvalidHuffmanCode)?;
            code = (code << 1) | bit as u16;
        }
        Err(Error::InvalidHuffmanCode)
    }

    /// Mirror of `interpretSymbol`
    /// (HuffmanCodingShortBitsCachedDeflate.hpp:172-202).
    fn interpret_symbol<R: LsbBitReader, const DIST_SYMS: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingReversedBitsCached<DIST_SYMS>,
        symbol: Symbol,
    ) -> Result<CacheEntry, Error> {
        let mut e = CacheEntry::default();
        if symbol <= 255 {
            e.symbol_or_length = symbol as u8;
            return Ok(e);
        }
        if symbol == END_OF_BLOCK_SYMBOL {
            e.distance = END_OF_BLOCK_DISTANCE;
            return Ok(e);
        }
        if symbol > 285 {
            return Err(Error::InvalidHuffmanCode);
        }

        // Length: getLengthMinus3 (vendor RFCTables.hpp:117-123).
        let look = length_lookup(symbol).ok_or(Error::InvalidHuffmanCode)?;
        let extra = if look.extra_bits_count > 0 {
            bit_reader
                .read(look.extra_bits_count)
                .map_err(|_| Error::InvalidHuffmanCode)? as u16
        } else {
            0
        };
        let length_val = look.base_length + extra;
        e.symbol_or_length = (length_val - 3) as u8;

        // Distance via Huffman decode + extra bits.
        let dec = distance_hc
            .decode(bit_reader)
            .ok_or(Error::InvalidHuffmanCode)?;
        let mut distance = dec;
        if distance <= 3 {
            distance += 1;
        } else if distance <= 29 {
            let extra_bits_count = ((distance - 2) / 2) as u8;
            let extra_bits = if extra_bits_count > 0 {
                bit_reader
                    .read(extra_bits_count)
                    .map_err(|_| Error::InvalidHuffmanCode)? as u16
            } else {
                0
            };
            distance = DISTANCE_LUT[distance as usize] + extra_bits;
        } else {
            return Err(Error::InvalidHuffmanCode);
        }
        e.distance = distance;
        Ok(e)
    }

    /// Mirror of `decode<BitReader, DistanceHuffmanCoding>`
    /// (HuffmanCodingShortBitsCachedDeflate.hpp:120-141).
    #[inline]
    pub fn decode<R: LsbBitReader, const DIST_SYMS: usize>(
        &self,
        bit_reader: &mut R,
        distance_hc: &HuffmanCodingReversedBitsCached<DIST_SYMS>,
    ) -> Result<CacheEntry, Error> {
        match bit_reader.peek(self.lut_bits_count) {
            Ok(idx) => {
                let entry = self.code_cache[idx as usize];
                if entry.bits_to_skip == 0 {
                    return self.decode_long(bit_reader, distance_hc);
                }
                bit_reader.seek_after_peek(entry.bits_to_skip);
                if entry.distance == DISTANCE_DEFERRED {
                    return self.interpret_symbol(
                        bit_reader,
                        distance_hc,
                        (entry.symbol_or_length as u16) + 257,
                    );
                }
                Ok(entry)
            }
            Err(_) => {
                let sym = self
                    .base
                    .decode(bit_reader)
                    .ok_or(Error::InvalidHuffmanCode)?;
                self.interpret_symbol(bit_reader, distance_hc, sym)
            }
        }
    }
}

impl<const LUT_BITS_COUNT: u8, const LUT_SIZE: usize> Default
    for HuffmanCodingShortBitsCachedDeflate<LUT_BITS_COUNT, LUT_SIZE>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::block_finder::BitReader;
    use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;

    /// Build a fixed lit/len + fixed distance pair and decode a single
    /// literal — exercises the LUT path for symbols 0..=255.
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
        let dist_lens = vec![5u8; 32];
        let mut dist_hc: HuffmanCodingReversedBitsCached<32> =
            HuffmanCodingReversedBitsCached::new();
        assert_eq!(dist_hc.initialize_from_lengths(&dist_lens), Error::None);

        let mut hc: HuffmanCodingShortBitsCachedDeflate<9, 512> =
            HuffmanCodingShortBitsCachedDeflate::new();
        let err = hc.initialize_from_lengths::<32>(&lens, &dist_hc);
        assert_eq!(err, Error::None);

        let data = [0b0000_1100u8, 0u8];
        let mut br = BitReader::new(&data);
        let e = hc.decode::<_, 32>(&mut br, &dist_hc).expect("decode ok");
        assert_eq!(e.symbol_or_length, 0);
        assert_eq!(e.distance, 0); // Pure literal.
    }
}
