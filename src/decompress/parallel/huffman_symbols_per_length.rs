//! Literal port of `rapidgzip::HuffmanCodingSymbolsPerLength`
//! (`vendor/rapidgzip/librapidarchive/src/huffman/HuffmanCodingSymbolsPerLength.hpp`).
//!
//! Stores all symbols sorted by code length in a single flat array plus
//! per-length offsets. Decoding reads bit-by-bit and consults the per-level
//! `minimum_code_values_per_level` from `HuffmanCodingBase` to decide whether
//! the current bit pattern matches a code at the current length. Used as a
//! base class for every cached variant that follows.
//!
//! ## Rust mapping
//!
//! `MAX_SYMBOL_COUNT` is a const generic on the Rust struct. Symbol type
//! is `u16` (the only width used by deflate). `MAX_CODE_LENGTH` is fixed
//! at 15 — same rationale as `huffman_base.rs`.

#![allow(dead_code)]

use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;
use super::huffman_base::{
    BitReaderError, CodeLengthFrequencies, HuffmanCodingBase, LsbBitReader, CODE_LENGTH_STORAGE,
};

/// Symbol type used by every deflate Huffman variant. The vendor's
/// `Symbol` template parameter — `uint8_t` for precode, `uint16_t` for
/// lit/len and distance. We unify on `u16`; per-variant `MAX_SYMBOL_COUNT`
/// bounds the actual value range.
pub type Symbol = u16;

/// Mirror of `rapidgzip::HuffmanCodingSymbolsPerLength`
/// (HuffmanCodingSymbolsPerLength.hpp:30-142).
///
/// ## Const generics
///
/// - `MAX_SYMBOL_COUNT` — vendor template parameter. Sizes the flat
///   `symbols_per_length` array.
///
/// `MAX_CODE_LENGTH` is fixed at 15 (see module preamble).
pub struct HuffmanCodingSymbolsPerLength<const MAX_SYMBOL_COUNT: usize> {
    /// Mirror of `HuffmanCodingBase` members — composed instead of
    /// inherited (Rust lacks C++ inheritance).
    pub base: HuffmanCodingBase,

    /// Mirror of `m_symbolsPerLength`
    /// (HuffmanCodingSymbolsPerLength.hpp:137). Symbols sorted first by
    /// code length, then by alphabet order. Aligned to 64 bytes
    /// (vendor `alignas(64)`) — Rust cannot express that on stable
    /// without manual padding, so we live with the natural alignment.
    pub symbols_per_length: [Symbol; MAX_SYMBOL_COUNT],

    /// Mirror of `m_offsets`
    /// (HuffmanCodingSymbolsPerLength.hpp:139). +1 entry to store the
    /// total count at the end as well as 0 in the first slot.
    pub offsets: [u16; CODE_LENGTH_STORAGE],
}

impl<const MAX_SYMBOL_COUNT: usize> HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT> {
    /// Static assert that mirrors the vendor's
    /// `static_assert( MAX_SYMBOL_COUNT + MAX_CODE_LENGTH <= 65535, ... )`
    /// at HuffmanCodingSymbolsPerLength.hpp:140-141.
    const _OFFSET_FITS: () =
        assert!(MAX_SYMBOL_COUNT + MAX_CODE_LENGTH as usize <= u16::MAX as usize);

    pub const fn new() -> Self {
        Self {
            base: HuffmanCodingBase::new(),
            symbols_per_length: [0; MAX_SYMBOL_COUNT],
            offsets: [0u16; CODE_LENGTH_STORAGE],
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    /// Mirror of `HuffmanCodingSymbolsPerLength::initializeSymbolsPerLength`
    /// (HuffmanCodingSymbolsPerLength.hpp:43-68).
    ///
    /// Builds the cumulative `offsets` array and the code-length-sorted
    /// `symbols_per_length` array. Caller must have populated
    /// `self.base.min_code_length` / `max_code_length` first via
    /// `HuffmanCodingBase::initialize_min_max_code_lengths`.
    pub fn initialize_symbols_per_length(
        &mut self,
        code_lengths: &[u8],
        frequencies: &CodeLengthFrequencies,
    ) {
        let min_cl = self.base.min_code_length;
        let max_cl = self.base.max_code_length;

        // Cumulative frequency sums → offsets for each code-length.
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

        // Fill the code-length-sorted alphabet vector.
        let mut sizes = self.offsets;
        for (symbol, &length) in code_lengths.iter().enumerate() {
            if length != 0 {
                let k = (length - min_cl) as usize;
                self.symbols_per_length[sizes[k] as usize] = symbol as Symbol;
                sizes[k] += 1;
            }
        }
    }

    /// Mirror of `HuffmanCodingSymbolsPerLength::initializeFromLengths`
    /// (HuffmanCodingSymbolsPerLength.hpp:71-95).
    pub fn initialize_from_lengths(
        &mut self,
        code_lengths: &[u8],
        check_optimality: bool,
    ) -> Error {
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

        let err = self.base.check_code_length_frequencies(
            &frequencies,
            code_lengths.len(),
            check_optimality,
        );
        if err != Error::None {
            return err;
        }

        // Resets frequencies[0] to 0.
        self.base.initialize_minimum_code_values(&mut frequencies);
        self.initialize_symbols_per_length(code_lengths, &frequencies);

        Error::None
    }

    /// Mirror of `HuffmanCodingSymbolsPerLength::decode<BitReader>`
    /// (HuffmanCodingSymbolsPerLength.hpp:97-124).
    ///
    /// Reads bits one at a time until the accumulated code falls within
    /// the per-level range defined by `minimum_code_values_per_level`.
    /// This is the slow but always-correct fallback used by every cached
    /// variant when their LUT misses (e.g. end-of-stream).
    #[inline]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<Symbol> {
        let base = &self.base;
        let mut code: u16 = 0;

        // Vendor: "Read the first n bytes. Note that we can't call the
        // bitReader with argument > 1 because the bit order would be
        // inversed." (HuffmanCodingSymbolsPerLength.hpp:103-105)
        for _ in 0..base.min_code_length {
            let bit = bit_reader.read_one().ok()?;
            code = (code << 1) | bit as u16;
        }

        let levels = base.max_code_length - base.min_code_length;
        for k in 0..=levels {
            let min_code = base.minimum_code_values_per_level[k as usize];
            if min_code <= code {
                let sub_index = self.offsets[k as usize] as usize + (code - min_code) as usize;
                if sub_index < self.offsets[(k + 1) as usize] as usize {
                    return Some(self.symbols_per_length[sub_index]);
                }
            }
            let bit = bit_reader.read_one().ok()?;
            code = (code << 1) | bit as u16;
        }
        None
    }
}

impl<const MAX_SYMBOL_COUNT: usize> Default for HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT> {
    fn default() -> Self {
        Self::new()
    }
}

// Helper for porting bit-reader EOF propagation when we don't need to
// distinguish missing bits from logic errors.
#[inline]
pub(crate) fn err_to_option<T>(r: Result<T, BitReaderError>) -> Option<T> {
    r.ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::block_finder::BitReader;

    /// Build the RFC 1951 fixed tree and decode every symbol class.
    #[test]
    fn rfc1951_fixed_tree_decodes_literals() {
        // Same layout as in HuffmanCodingBase tests.
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

        let mut hc: HuffmanCodingSymbolsPerLength<512> = HuffmanCodingSymbolsPerLength::new();
        let err = hc.initialize_from_lengths(&lens, true);
        assert_eq!(err, Error::None, "fixed tree must initialize");
        assert!(hc.is_valid());

        // Encode literal 0 using the fixed tree: lit 0 → code 0b00110000
        // (length 8). MSB-first encoded → LSB-first bit stream is
        // 0,0,0,0,1,1,0,0.
        let data = [0b0000_1100u8];
        let mut br = BitReader::new(&data);
        let s = hc.decode(&mut br);
        assert_eq!(s, Some(0));
    }
}
