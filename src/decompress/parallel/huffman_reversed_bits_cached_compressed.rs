//! Literal port of `rapidgzip::HuffmanCodingReversedBitsCachedCompressed`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingReversedBitsCachedCompressed.hpp`).
//!
//! Same as `HuffmanCodingReversedBitsCached` but packs the `(length, symbol)`
//! pair into a single `Symbol`-width LUT entry: the symbol occupies the low
//! `LENGTH_SHIFT` bits and the length occupies the high bits. Halves the LUT
//! size and (per vendor comment) is ~5 % faster from improved L1 utilization.

#![allow(dead_code)]

use super::bit_manipulation::{n_lowest_bits_set, required_bits, reverse_bits_count_u16};
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::{HuffmanCodingSymbolsPerLength, Symbol};

/// LUT entries are `Symbol` (u16). Length is shifted into the high bits.
pub const CODE_CACHE_SIZE: usize = 1 << MAX_CODE_LENGTH;

/// Mirror of `rapidgzip::HuffmanCodingReversedBitsCachedCompressed`
/// (HuffmanCodingReversedBitsCachedCompressed.hpp:34-146).
pub struct HuffmanCodingReversedBitsCachedCompressed<const MAX_SYMBOL_COUNT: usize> {
    pub base: HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT>,
    /// `m_codeCache` (HuffmanCodingReversedBitsCachedCompressed.hpp:143).
    pub code_cache: Box<[Symbol; CODE_CACHE_SIZE]>,
    needs_to_be_zeroed: bool,
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingReversedBitsCachedCompressed<MAX_SYMBOL_COUNT> {
    /// Mirror of
    /// `HuffmanCodingReversedBitsCachedCompressed::LENGTH_SHIFT`
    /// (HuffmanCodingReversedBitsCachedCompressed.hpp:42).
    pub const LENGTH_SHIFT: u8 = required_bits(MAX_SYMBOL_COUNT as u64);

    const _LENGTH_FITS: () = assert!(
        MAX_SYMBOL_COUNT <= (1usize << Self::LENGTH_SHIFT),
        "Not enough free bits to pack length into Symbol!"
    );

    pub fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: Box::new([0 as Symbol; CODE_CACHE_SIZE]),
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of
    /// `HuffmanCodingReversedBitsCachedCompressed::initializeFromLengths`
    /// (HuffmanCodingReversedBitsCachedCompressed.hpp:46-108).
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths, true);
        if err != Error::None {
            return err;
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

            let value: Symbol = (symbol as Symbol) | ((length as Symbol) << Self::LENGTH_SHIFT);
            debug_assert!(((value >> Self::LENGTH_SHIFT) as u8) == length);
            debug_assert!(
                (value & (n_lowest_bits_set(Self::LENGTH_SHIFT) as Symbol)) == symbol as Symbol
            );

            let mut padded_code = reversed_code;
            loop {
                self.code_cache[padded_code as usize] = value;
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
    /// `HuffmanCodingReversedBitsCachedCompressed::decode<BitReader>`
    /// (HuffmanCodingReversedBitsCachedCompressed.hpp:110-134).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let max_cl = self.base.base.max_code_length;
        match bit_reader.peek(max_cl) {
            Ok(value) => {
                let mut symbol = self.code_cache[value as usize];
                let length = (symbol >> Self::LENGTH_SHIFT) as u8;
                symbol &= n_lowest_bits_set(Self::LENGTH_SHIFT) as Symbol;
                if length == 0 {
                    return None;
                }
                bit_reader.seek_after_peek(length);
                Some(symbol)
            }
            Err(_) => self.base.decode(bit_reader),
        }
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default
    for HuffmanCodingReversedBitsCachedCompressed<MAX_SYMBOL_COUNT>
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
        let mut hc: HuffmanCodingReversedBitsCachedCompressed<512> =
            HuffmanCodingReversedBitsCachedCompressed::new();
        assert_eq!(hc.initialize_from_lengths(&lens), Error::None);

        let data = [0b0000_1100u8, 0u8];
        let mut br = BitReader::new(&data);
        assert_eq!(hc.decode(&mut br), Some(0));
    }
}
