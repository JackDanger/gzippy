//! Literal port of `rapidgzip::HuffmanCodingReversedBitsCachedSeparateLengths`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingReversedBitsCachedSeparateLengths.hpp`).
//!
//! Same as `HuffmanCodingReversedBitsCached` but stores the code length in a
//! separate, smaller table indexed by symbol. The main LUT then only needs
//! to hold the symbol (plus a 0-marker for "non-optimal tree miss"). Adds
//! one extra small-array lookup per decode in exchange for a slightly
//! smaller hot LUT.

#![allow(dead_code)]

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};

pub const CODE_CACHE_SIZE: usize = 1 << MAX_CODE_LENGTH;

/// Mirror of
/// `rapidgzip::HuffmanCodingReversedBitsCachedSeparateLengths`
/// (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:24-110).
pub struct HuffmanCodingReversedBitsCachedSeparateLengths<const MAX_SYMBOL_COUNT: usize> {
    pub base: HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT>,

    /// Mirror of `m_codeLengths` (separate-length table).
    /// (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:106). Indexed
    /// by `symbol + 1` because `0` is the "miss" sentinel in `code_cache`.
    pub code_lengths_tbl: [u8; MAX_SYMBOL_COUNT],

    /// Mirror of `m_codeCache`
    /// (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:107). Stores
    /// `symbol + 1` so `0` is reserved as the miss sentinel.
    pub code_cache: Box<[Symbol; CODE_CACHE_SIZE]>,

    needs_to_be_zeroed: bool,
}

impl<const MAX_SYMBOL_COUNT: usize>
    HuffmanCodingReversedBitsCachedSeparateLengths<MAX_SYMBOL_COUNT>
{
    pub fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_lengths_tbl: [0u8; MAX_SYMBOL_COUNT],
            code_cache: Box::new([0 as Symbol; CODE_CACHE_SIZE]),
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of
    /// `HuffmanCodingReversedBitsCachedSeparateLengths::initializeFromLengths`
    /// (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:33-78).
    ///
    /// The vendor stores `m_codeLengths[symbol + 1] = length` to keep
    /// index 0 free as the miss sentinel — we replicate that exactly.
    /// `MAX_SYMBOL_COUNT` here is the array size; the vendor's
    /// `code_lengths_tbl` has the same dimension as `m_codeLengths`.
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
        }
        if code_lengths.len() > MAX_SYMBOL_COUNT {
            return Error::ExceededSymbolRange;
        }

        let max_cl = self.base.base.max_code_length;
        let min_cl = self.base.base.min_code_length;

        if self.needs_to_be_zeroed {
            for slot in self.code_cache[..(1usize << max_cl)].iter_mut() {
                *slot = 0;
            }
        }

        let mut code_values = self.base.base.minimum_code_values_per_level;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            // Vendor stores at `symbol + 1` (line 56). The +1 keeps the
            // sentinel free. With MAX_SYMBOL_COUNT == 286, the in-bounds
            // condition is `symbol < MAX_SYMBOL_COUNT - 1`. Vendor
            // assumes the caller passes ≤ MAX_SYMBOL_COUNT - 1 lengths,
            // which we also assume; defensively skip the out-of-bounds
            // case (avoids panics on slightly oversized inputs).
            if symbol + 1 < self.code_lengths_tbl.len() {
                self.code_lengths_tbl[symbol + 1] = length;
            }
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
                // `symbol + 1` to leave 0 as miss sentinel
                // (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:71).
                self.code_cache[padded_code as usize] = (symbol + 1) as Symbol;
                if padded_code == maximum_padded_code {
                    break;
                }
                padded_code = padded_code.wrapping_add(increment);
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of
    /// `HuffmanCodingReversedBitsCachedSeparateLengths::decode<BitReader>`
    /// (HuffmanCodingReversedBitsCachedSeparateLengths.hpp:80-103).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let max_cl = self.base.base.max_code_length;
        match bit_reader.peek(max_cl) {
            Ok(value) => {
                let symbol = self.code_cache[value as usize];
                if symbol == 0 {
                    return None;
                }
                let length = self.code_lengths_tbl[symbol as usize];
                bit_reader.seek_after_peek(length);
                Some(symbol - 1)
            }
            Err(_) => self.base.decode(bit_reader),
        }
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default
    for HuffmanCodingReversedBitsCachedSeparateLengths<MAX_SYMBOL_COUNT>
{
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
        let mut hc: HuffmanCodingReversedBitsCachedSeparateLengths<512> =
            HuffmanCodingReversedBitsCachedSeparateLengths::new();
        assert_eq!(hc.initialize_from_lengths(&lens), Error::None);

        let data = [0b0000_1100u8, 0u8];
        let mut br = BitReader::new(&data);
        assert_eq!(hc.decode(&mut br), Some(0));
    }
}
