//! Literal port of `rapidgzip::HuffmanCodingDoubleLiteralCached`
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingDoubleLiteralCached.hpp`).
//!
//! Multi-symbol decoder: stores up to two consecutive literal symbols per LUT
//! entry by combining two short Huffman codes when they fit within
//! `cached_bit_count` total bits. Each LUT slot is a 2-element `[Symbol; 2]`
//! pair `(symbol_and_length, next_symbol)`. When the first symbol is a deflate
//! special (>= 256), no second symbol is cached and the caller falls back to
//! single-symbol behavior — see vendor comment at
//! HuffmanCodingDoubleLiteralCached.hpp:118.
//!
//! Inherits from `HuffmanCodingReversedCodesPerLength` because it needs the
//! per-length reversed codes during the table build.

#![allow(dead_code)]

use super::bit_manipulation::{n_lowest_bits_set, required_bits};
use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::LsbBitReader;
use super::huffman_reversed_codes_per_length::HuffmanCodingReversedCodesPerLength;
use super::huffman_symbols_per_length::Symbol;

/// LUT size: `2 * (1 << MAX_CODE_LENGTH)`. Two entries per LUT slot.
pub const CODE_CACHE_SIZE: usize = 2 * (1 << MAX_CODE_LENGTH);

/// Mirror of `rapidgzip::HuffmanCodingDoubleLiteralCached`
/// (HuffmanCodingDoubleLiteralCached.hpp:21-323).
pub struct HuffmanCodingDoubleLiteralCached<const MAX_SYMBOL_COUNT: usize> {
    pub base: HuffmanCodingReversedCodesPerLength<MAX_SYMBOL_COUNT>,

    /// `m_cachedBitCount`
    /// (HuffmanCodingDoubleLiteralCached.hpp:314). Chosen at init time as
    /// `clamp(2 * min_code_length + 1, max_code_length, MAX_CODE_LENGTH)`.
    pub cached_bit_count: u32,

    /// `m_nextSymbol` (HuffmanCodingDoubleLiteralCached.hpp:315). Holds
    /// the second symbol of a double-decoded pair across calls.
    /// `NONE_SYMBOL` (u16::MAX) means "nothing pending".
    next_symbol: core::cell::Cell<Symbol>,

    /// `m_doubleCodeCache` (HuffmanCodingDoubleLiteralCached.hpp:321).
    /// `2 * (1 << MAX_CODE_LENGTH)` symbols. Even indices store
    /// `symbol | (length << LENGTH_SHIFT)`, odd indices store the second
    /// cached symbol.
    pub double_code_cache: Box<[Symbol; CODE_CACHE_SIZE]>,
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingDoubleLiteralCached<MAX_SYMBOL_COUNT> {
    /// `LENGTH_SHIFT = 10` per HuffmanCodingDoubleLiteralCached.hpp:31.
    ///
    /// Note this is intentionally not `required_bits(MAX_SYMBOL_COUNT)`:
    /// the vendor hardcodes 10 to leave enough room for the *summed*
    /// code length (up to 2 * MAX_CODE_LENGTH = 30) — the
    /// `_compressed` variant uses required_bits because it only encodes
    /// a single length.
    pub const LENGTH_SHIFT: u32 = 10;

    /// `NONE_SYMBOL = numeric_limits<Symbol>::max()` per
    /// HuffmanCodingDoubleLiteralCached.hpp:33.
    pub const NONE_SYMBOL: Symbol = Symbol::MAX;

    const _SYMBOL_FITS: () = assert!(
        MAX_SYMBOL_COUNT <= Self::NONE_SYMBOL as usize,
        "Not enough unused symbols for special none symbol!"
    );

    /// Used internally for `LENGTH_SHIFT` calculations; ports vendor's
    /// `required_bits(MAX_SYMBOL_COUNT)` reference.
    pub const _MIN_LENGTH_SHIFT: u8 = required_bits(MAX_SYMBOL_COUNT as u64);

    pub fn new() -> Self {
        Self {
            base: HuffmanCodingReversedCodesPerLength::new(),
            cached_bit_count: 0,
            next_symbol: core::cell::Cell::new(Self::NONE_SYMBOL),
            double_code_cache: Box::new([0 as Symbol; CODE_CACHE_SIZE]),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of
    /// `HuffmanCodingDoubleLiteralCached::initializeFromLengths`
    /// (HuffmanCodingDoubleLiteralCached.hpp:40-280).
    ///
    /// Only the `#if 1` branch (the iterative build) is ported; the
    /// alternative recursive `fillCache` lambda at lines 187-258 is
    /// `#if 0` in the vendor and unused.
    pub fn initialize_from_lengths(&mut self, code_lengths: &[u8]) -> Error {
        let err = self.base.initialize_from_lengths(code_lengths);
        if err != Error::None {
            return err;
        }

        // Vendor explicitly forbids the single-symbol case
        // (HuffmanCodingDoubleLiteralCached.hpp:59-61).
        let bb = &self.base.base;
        if bb.min_code_length == 1 && bb.max_code_length == 1 && self.base.offsets[1] == 1 {
            return Error::InvalidCodeLengths;
        }

        self.next_symbol.set(Self::NONE_SYMBOL);

        // Choose cached_bit_count:
        // clamp(2*min + 1, max, MAX_CODE_LENGTH).
        // (HuffmanCodingDoubleLiteralCached.hpp:89-91)
        self.cached_bit_count = core::cmp::min(
            core::cmp::max(bb.max_code_length as u32, 2 * bb.min_code_length as u32 + 1),
            MAX_CODE_LENGTH as u32,
        );

        // Initialize all slots to NONE_SYMBOL.
        for x in self.double_code_cache.iter_mut() {
            *x = Self::NONE_SYMBOL;
        }

        let size =
            self.base.offsets[(bb.max_code_length - bb.min_code_length + 1) as usize] as usize;
        let mut length = bb.min_code_length;
        let mut i: usize = 0;
        while i < size {
            let reversed_code = self.base.codes_per_length[i];
            let symbol = self.base.symbols_per_length[i];

            // Do not greedily decode two symbols if the first is a
            // deflate special (>= 256) — it consumes subsequent bits.
            if (length as u32 + bb.min_code_length as u32 > self.cached_bit_count)
                || (symbol >= 256)
            {
                let filler_bit_count = (self.cached_bit_count - length as u32) as u8;
                let symbol_and_length: Symbol = symbol | ((length as Symbol) << Self::LENGTH_SHIFT);
                let limit = 1u32 << filler_bit_count;
                for filler_bits in 0..limit {
                    let padded_code = ((filler_bits as u16) << length) | reversed_code;
                    self.double_code_cache[(padded_code as usize) * 2] = symbol_and_length;
                    // odd index left as NONE_SYMBOL (vendor leaves it).
                }
            } else {
                let mut length2 = bb.min_code_length;
                let mut i2: usize = 0;
                while i2 < size {
                    let reversed_code2 = self.base.codes_per_length[i2];
                    let symbol2 = self.base.symbols_per_length[i2];
                    let total_length = length as u32 + length2 as u32;

                    if total_length > self.cached_bit_count {
                        debug_assert!(length as u32 <= self.cached_bit_count);
                        let padded_code = ((reversed_code2 << length) | reversed_code)
                            & n_lowest_bits_set(self.cached_bit_count as u8) as u16;
                        self.double_code_cache[(padded_code as usize) * 2] =
                            symbol | ((length as Symbol) << Self::LENGTH_SHIFT);
                    } else {
                        let filler_bit_count = (self.cached_bit_count - total_length) as u8;
                        let merged_code = (reversed_code2 << length) | reversed_code;
                        let symbol_and_length: Symbol =
                            symbol | ((total_length as Symbol) << Self::LENGTH_SHIFT);
                        let limit = 1u32 << filler_bit_count;
                        for filler_bits in 0..limit {
                            let padded_code = ((filler_bits as u16) << total_length) | merged_code;
                            self.double_code_cache[(padded_code as usize) * 2] = symbol_and_length;
                            self.double_code_cache[(padded_code as usize) * 2 + 1] = symbol2;
                        }
                    }

                    i2 += 1;
                    if i2 >= size {
                        break;
                    }
                    while self.base.offsets[(length2 - bb.min_code_length + 1) as usize] as usize
                        == i2
                    {
                        length2 += 1;
                    }
                }
            }

            i += 1;
            if i >= size {
                break;
            }
            while self.base.offsets[(length - bb.min_code_length + 1) as usize] as usize == i {
                length += 1;
            }
        }
        Error::None
    }

    /// Mirror of `HuffmanCodingDoubleLiteralCached::decode<BitReader>`
    /// (HuffmanCodingDoubleLiteralCached.hpp:282-311).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let pending = self.next_symbol.get();
        if pending != Self::NONE_SYMBOL {
            self.next_symbol.set(Self::NONE_SYMBOL);
            return Some(pending);
        }
        match bit_reader.peek(self.cached_bit_count as u8) {
            Ok(value) => {
                let idx = value as usize * 2;
                let mut symbol1 = self.double_code_cache[idx];
                let symbol2 = self.double_code_cache[idx + 1];
                let length = (symbol1 >> Self::LENGTH_SHIFT) as u8;
                if length == 0 {
                    return None;
                }
                self.next_symbol.set(symbol2);
                bit_reader.seek_after_peek(length);
                symbol1 &= n_lowest_bits_set(Self::LENGTH_SHIFT as u8) as Symbol;
                Some(symbol1)
            }
            Err(_) => self.base.decode(bit_reader),
        }
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default for HuffmanCodingDoubleLiteralCached<MAX_SYMBOL_COUNT> {
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
        let mut hc: HuffmanCodingDoubleLiteralCached<512> = HuffmanCodingDoubleLiteralCached::new();
        let err = hc.initialize_from_lengths(&lens);
        assert_eq!(err, Error::None);
        // Some bits with literal 0 followed by something.
        let data = [0b0000_1100u8, 0xFFu8];
        let mut br = BitReader::new(&data);
        let s = hc.decode(&mut br);
        // Literal 0 should decode regardless of the second symbol.
        assert_eq!(s, Some(0));
    }
}
