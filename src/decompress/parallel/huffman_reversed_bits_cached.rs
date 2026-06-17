#![allow(dead_code)] // vendor-faithful rapidgzip port; wired at deflate_block canonical FIXED path

//! Port of `rapidgzip::HuffmanCodingReversedBitsCached`
//! (`vendor/.../huffman/HuffmanCodingReversedBitsCached.hpp:32-136`).
//!
//! LUT-cached canonical decoder: peek `max_code_length` bits, index into
//! `(length, symbol)` cache, `seek_after_peek(length)`.

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};

const CACHE_LEN: usize = 1usize << MAX_CODE_LENGTH;

/// Mirror of `HuffmanCodingReversedBitsCached` (ReversedBitsCached.hpp:32-136).
pub struct HuffmanCodingReversedBitsCached<const MAX_SYMBOL_COUNT: usize> {
    base: HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT>,
    code_cache: [(u8, Symbol); CACHE_LEN],
    needs_to_be_zeroed: bool,
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingReversedBitsCached<MAX_SYMBOL_COUNT> {
    pub const fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: [(0, 0); CACHE_LEN],
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Resident per-thread footprint of the reversed-bits `code_cache`
    /// (`[(u8, Symbol); 1<<MAX_CODE_LENGTH]` = 128 KiB at MAX_CODE_LENGTH=15).
    /// Inline in the struct (the owning `Block` lives in thread-local storage),
    /// but it is a real per-thread resident working-set component, so the
    /// cache-mandate instrument counts it. Counters only.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(&self.code_cache)
    }

    pub fn max_code_length(&self) -> u8 {
        self.base.base.max_code_length
    }

    /// Mirror of `initializeFromLengths` (ReversedBitsCached.hpp:41-101).
    pub fn initialize_from_lengths(
        &mut self,
        code_lengths: &[u8],
        check_optimality: bool,
    ) -> Error {
        let err = self
            .base
            .initialize_from_lengths(code_lengths, check_optimality);
        if err != Error::None {
            return err;
        }

        if self.needs_to_be_zeroed {
            for entry in &mut self.code_cache {
                entry.0 = 0;
            }
        }

        let base = &self.base.base;
        let mut code_values = base.minimum_code_values_per_level;

        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let k = (length - base.min_code_length) as usize;
            let code = code_values[k];
            code_values[k] = code.wrapping_add(1);

            let reversed_code = reverse_bits_count_u16(code, length);
            let filler_bit_count = base.max_code_length - length;
            let maximum_padded_code =
                reversed_code as u32 | ((n_lowest_bits_set(filler_bit_count) as u32) << length);
            debug_assert!((maximum_padded_code as usize) < CACHE_LEN);

            let increment = 1u32 << length;
            let mut padded_code = reversed_code as u32;
            while padded_code <= maximum_padded_code {
                self.code_cache[padded_code as usize] = (length, symbol as Symbol);
                padded_code += increment;
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `decode` (ReversedBitsCached.hpp:103-125).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let max_len = self.base.base.max_code_length;
        let value = match bit_reader.peek(max_len) {
            Ok(v) => v as usize,
            Err(_) => return self.base.decode(bit_reader),
        };
        debug_assert!(value < CACHE_LEN);

        let (length, symbol) = self.code_cache[value];
        if length == 0 {
            return self.base.decode(bit_reader);
        }

        bit_reader.seek_after_peek(length);
        Some(symbol)
    }

    /// Mirror of `codeCache()` (ReversedBitsCached.hpp:127-131).
    pub fn code_cache(&self) -> &[(u8, Symbol); CACHE_LEN] {
        &self.code_cache
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default for HuffmanCodingReversedBitsCached<MAX_SYMBOL_COUNT> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::bit_reader::BitReader;

    #[test]
    fn fixed_tree_cached_matches_symbols_per_length() {
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

        let mut cached: HuffmanCodingReversedBitsCached<512> =
            HuffmanCodingReversedBitsCached::new();
        assert_eq!(cached.initialize_from_lengths(&lens, true), Error::None);

        let mut slow: HuffmanCodingSymbolsPerLength<512> = HuffmanCodingSymbolsPerLength::new();
        assert_eq!(slow.initialize_from_lengths(&lens, true), Error::None);

        let data = [0b0000_1100u8];
        let mut br_cached = BitReader::new(&data);
        let mut br_slow = BitReader::new(&data);
        assert_eq!(cached.decode(&mut br_cached), Some(0));
        assert_eq!(slow.decode(&mut br_slow), Some(0));

        // Extra byte so 9-bit peek succeeds — exercises the LUT fast path.
        let data9 = [0b0000_1100u8, 0];
        let mut br_fast = BitReader::new(&data9);
        assert_eq!(cached.decode(&mut br_fast), Some(0));
    }
}
