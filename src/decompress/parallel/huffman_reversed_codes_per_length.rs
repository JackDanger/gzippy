//! Literal port of `rapidgzip::HuffmanCodingReversedCodesPerLength`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingReversedCodesPerLength.hpp`).
//!
//! Same alphabet layout as `HuffmanCodingSymbolsPerLength` but stores
//! precomputed reversed codes per length so decoding can `read(min)`
//! all min-length bits at once instead of bit-by-bit. Iterates over the
//! reversed codes and reads one extra bit each time the offsets cross a
//! length boundary.
//!
//! ## Why a base alongside the other variant
//!
//! `HuffmanCodingDoubleLiteralCached` (HuffmanCodingDoubleLiteralCached.hpp)
//! inherits from this rather than `HuffmanCodingSymbolsPerLength` because
//! it needs the per-length reversed codes too. Keeping this as its own
//! storage layer preserves the C++ class hierarchy 1:1.

#![allow(dead_code)]

use super::bit_manipulation::reverse_bits_count_u16;
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::{
    CodeLengthFrequencies, HuffmanCodingBase, LsbBitReader, CODE_LENGTH_STORAGE,
};
use super::huffman_symbols_per_length::Symbol;

/// Mirror of `rapidgzip::HuffmanCodingReversedCodesPerLength`
/// (HuffmanCodingReversedCodesPerLength.hpp:30-143).
pub struct HuffmanCodingReversedCodesPerLength<const MAX_SYMBOL_COUNT: usize> {
    pub base: HuffmanCodingBase,

    /// Mirror of `m_symbolsPerLength`
    /// (HuffmanCodingReversedCodesPerLength.hpp:134).
    pub symbols_per_length: [Symbol; MAX_SYMBOL_COUNT],

    /// Mirror of `m_codesPerLength` — reversed codes
    /// (HuffmanCodingReversedCodesPerLength.hpp:135).
    pub codes_per_length: [u16; MAX_SYMBOL_COUNT],

    /// Mirror of `m_offsets`
    /// (HuffmanCodingReversedCodesPerLength.hpp:140).
    pub offsets: [u16; CODE_LENGTH_STORAGE],
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingReversedCodesPerLength<MAX_SYMBOL_COUNT> {
    const _OFFSET_FITS: () =
        assert!(MAX_SYMBOL_COUNT + MAX_CODE_LENGTH as usize <= u16::MAX as usize);

    pub const fn new() -> Self {
        Self {
            base: HuffmanCodingBase::new(),
            symbols_per_length: [0; MAX_SYMBOL_COUNT],
            codes_per_length: [0; MAX_SYMBOL_COUNT],
            offsets: [0u16; CODE_LENGTH_STORAGE],
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of
    /// `HuffmanCodingReversedCodesPerLength::initializeCodingTable`
    /// (HuffmanCodingReversedCodesPerLength.hpp:42-73).
    fn initialize_coding_table(
        &mut self,
        code_lengths: &[u8],
        frequencies: &CodeLengthFrequencies,
    ) {
        let min_cl = self.base.min_code_length;
        let max_cl = self.base.max_code_length;

        let mut sum: usize = 0;
        for bit_length in min_cl..=max_cl {
            self.offsets[(bit_length - min_cl) as usize] = sum as u16;
            sum += frequencies[bit_length as usize] as usize;
        }
        self.offsets[(max_cl - min_cl + 1) as usize] = sum as u16;

        debug_assert!(
            sum <= MAX_SYMBOL_COUNT,
            "Specified max symbol range exceeded!"
        );

        let mut sizes = self.offsets;
        let mut code_values_per_level = self.base.minimum_code_values_per_level;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length != 0 {
                let k = (length - min_cl) as usize;
                let code = code_values_per_level[k];
                code_values_per_level[k] = code.wrapping_add(1);
                let pos = sizes[k] as usize;
                self.symbols_per_length[pos] = symbol as Symbol;
                self.codes_per_length[pos] = reverse_bits_count_u16(code, length);
                sizes[k] += 1;
            }
        }
    }

    /// Mirror of
    /// `HuffmanCodingReversedCodesPerLength::initializeFromLengths`
    /// (HuffmanCodingReversedCodesPerLength.hpp:76-100).
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self
            .base
            .initialize_min_max_code_lengths(code_lengths, MAX_SYMBOL_COUNT);
        if err != Error::None {
            return err;
        }
        let mut frequencies: CodeLengthFrequencies = [0u32; CODE_LENGTH_STORAGE];
        for &cl in code_lengths {
            frequencies[cl as usize] += 1;
        }
        let err = self
            .base
            .check_code_length_frequencies(&frequencies, code_lengths.len(), true);
        if err != Error::None {
            return err;
        }
        self.base.initialize_minimum_code_values(&mut frequencies);
        self.initialize_coding_table(code_lengths, &frequencies);
        Error::None
    }

    /// Mirror of
    /// `HuffmanCodingReversedCodesPerLength::decode<BitReader>`
    /// (HuffmanCodingReversedCodesPerLength.hpp:102-121).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let min_cl = self.base.min_code_length;
        let max_cl = self.base.max_code_length;

        let mut code = bit_reader.read(min_cl).ok()? as u16;
        let size = self.offsets[(max_cl - min_cl + 1) as usize] as usize;
        let mut relative_code_length: u8 = 0;

        for i in 0..size {
            if self.codes_per_length[i] == code {
                return Some(self.symbols_per_length[i]);
            }
            while self.offsets[(relative_code_length + 1) as usize] as usize == i + 1 {
                let bit = bit_reader.read_one().ok()? as u16;
                code |= bit << (min_cl + relative_code_length);
                relative_code_length += 1;
            }
        }
        None
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default
    for HuffmanCodingReversedCodesPerLength<MAX_SYMBOL_COUNT>
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
        let mut hc: HuffmanCodingReversedCodesPerLength<512> =
            HuffmanCodingReversedCodesPerLength::new();
        assert_eq!(hc.initialize_from_lengths(&lens), Error::None);

        // Literal 0 → code 0b00110000 (length 8). LSB-first stream:
        // 0,0,0,0,1,1,0,0 -> byte 0b00001100.
        let data = [0b0000_1100u8];
        let mut br = BitReader::new(&data);
        assert_eq!(hc.decode(&mut br), Some(0));
    }
}
