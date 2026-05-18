//! Literal port of `rapidgzip::HuffmanCodingReversedBitsCached`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingReversedBitsCached.hpp`).
//!
//! LUT-cached Huffman decoder: peeks `max_code_length` bits and looks them up
//! in a `1 << max_code_length` entry table. Each entry is a `(length, symbol)`
//! pair. On a miss (length == 0) decoding falls back to the bit-by-bit base
//! `decode`. The LUT build duplicates every code into all `2^(max-length)`
//! padded slots so any future peek lands on the right symbol regardless of
//! the high garbage bits.
//!
//! Inherits storage from `HuffmanCodingSymbolsPerLength` (vendor) — composed
//! here in Rust.

#![allow(dead_code)]

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};

/// Storage capacity for the LUT — `1 << MAX_CODE_LENGTH` = 32768 for
/// deflate's 15-bit limit.
pub const CODE_CACHE_SIZE: usize = 1 << MAX_CODE_LENGTH;

/// Mirror of `rapidgzip::HuffmanCodingReversedBitsCached`
/// (HuffmanCodingReversedBitsCached.hpp:32-137).
pub struct HuffmanCodingReversedBitsCached<const MAX_SYMBOL_COUNT: usize> {
    pub base: HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT>,
    /// `m_codeCache` (HuffmanCodingReversedBitsCached.hpp:134).
    /// Stored as `(length, symbol)` pairs — `pair<uint8_t, Symbol>`
    /// in the vendor.
    pub code_cache: Box<[(u8, Symbol); CODE_CACHE_SIZE]>,
    /// `m_needsToBeZeroed` (HuffmanCodingReversedBitsCached.hpp:135).
    /// Skips the wipe on first init.
    needs_to_be_zeroed: bool,
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingReversedBitsCached<MAX_SYMBOL_COUNT> {
    pub fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: Box::new([(0u8, 0 as Symbol); CODE_CACHE_SIZE]),
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of `HuffmanCodingReversedBitsCached::initializeFromLengths`
    /// (HuffmanCodingReversedBitsCached.hpp:42-101).
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }

        let max_cl = self.base.base.max_code_length;
        let min_cl = self.base.base.min_code_length;

        if self.needs_to_be_zeroed {
            for slot in self.code_cache[..(1usize << max_cl)].iter_mut() {
                slot.0 = 0;
            }
        }

        let mut code_values = self.base.base.minimum_code_values_per_level;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let k = (length - min_cl) as usize;
            let code = code_values[k];
            code_values[k] = code.wrapping_add(1);
            let reversed_code = reverse_bits_count_u16(code, length);

            let filler_bit_count = max_cl - length;
            let maximum_padded_code =
                reversed_code | ((n_lowest_bits_set(filler_bit_count) as u16) << length);
            debug_assert!((maximum_padded_code as usize) < CODE_CACHE_SIZE);
            let increment: u16 = 1u16 << length;
            let mut padded_code = reversed_code;
            loop {
                self.code_cache[padded_code as usize] = (length, symbol as Symbol);
                if padded_code == maximum_padded_code {
                    break;
                }
                padded_code = padded_code.wrapping_add(increment);
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `HuffmanCodingReversedBitsCached::decode<BitReader>`
    /// (HuffmanCodingReversedBitsCached.hpp:103-125).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let max_cl = self.base.base.max_code_length;
        match bit_reader.peek(max_cl) {
            Ok(value) => {
                let (length, symbol) = self.code_cache[value as usize];
                if length == 0 {
                    // Non-optimal Huffman tree miss.
                    return None;
                }
                bit_reader.seek_after_peek(length);
                Some(symbol)
            }
            Err(_) => {
                // EOF — fall back to bit-by-bit decode. Vendor catches
                // `EndOfFileReached` and calls the base
                // `decode` (HuffmanCodingReversedBitsCached.hpp:120-124).
                self.base.decode(bit_reader)
            }
        }
    }

    /// Mirror of `HuffmanCodingReversedBitsCached::codeCache`
    /// (HuffmanCodingReversedBitsCached.hpp:127-131). Exposed so the
    /// `ShortBitsCachedDeflate` variant can consult a distance HC's LUT
    /// during table build.
    #[inline]
    pub fn code_cache(&self) -> &[(u8, Symbol); CODE_CACHE_SIZE] {
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
    use crate::decompress::parallel::block_finder::BitReader;

    #[test]
    fn decode_literal_zero_with_fixed_tree() {
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
        let mut hc: HuffmanCodingReversedBitsCached<512> = HuffmanCodingReversedBitsCached::new();
        assert_eq!(hc.initialize_from_lengths(&lens), Error::None);

        // Literal 0 → reversed code 0b00001100 (length 8). LUT lookup
        // by `peek(max=9)` reads 9 bits LSB first → low 8 bits =
        // reversed code; bit 8 is the next bit (don't care).
        let data = [0b0000_1100u8, 0u8];
        let mut br = BitReader::new(&data);
        assert_eq!(hc.decode(&mut br), Some(0));
    }
}
