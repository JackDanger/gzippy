#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::HuffmanCodingBase`
//! (`vendor/rapidgzip/librapidarchive/src/huffman/HuffmanCodingBase.hpp`).
//!
//! The C++ class is a template
//! `<typename HuffmanCode, uint8_t MAX_CODE_LENGTH, typename Symbol,
//!   size_t MAX_SYMBOL_COUNT, bool CHECK_OPTIMALITY = true>` that holds
//! the storage common to every Huffman variant (min/max code length and
//! per-level minimum code values) and validates code-length frequencies.
//!
//! ## Rust mapping
//!
//! Rust stable has no `generic_const_exprs`, so we cannot dimension
//! per-level arrays from a `const MAX_CODE_LENGTH` generic. The vendor's
//! deflate template instantiations all use `MAX_CODE_LENGTH = 15`
//! (bzip2's 20-bit instantiation is unused by gzippy), so the base struct
//! is concrete and sized for the deflate maximum. The vendor's
//! `HuffmanCode` template parameter is fixed to `u16` here for the same
//! reason — that's the only width the deflate variants instantiate.
//!
//! ## BitReader trait
//!
//! rapidgzip's Huffman variants are templated over the `BitReader` type.
//! The shared surface used by the Huffman decoders is small:
//!   - `read(num_bits)` / `read<N>()`
//!   - `peek(num_bits)` / `peek<N>()`
//!   - `seek_after_peek(num_bits)`
//!   - `EndOfFileReached` exception (Rust: `Result<_, BitReaderError::Eof>`)
//!
//! We define a Rust trait `LsbBitReader` (LSB-first, the deflate order)
//! covering these methods so each Huffman variant can be generic over it.
//! See `LsbBitReader` below for the canonical mapping to the rapidgzip
//! `gzip::BitReader` (`vendor/.../filereader/BitReader.hpp:194-289`).

use super::error::Error;
use super::gzip_definitions::MAX_CODE_LENGTH;

/// Storage capacity used by every per-level array.
/// Mirror of the vendor `std::array<HuffmanCode, MAX_CODE_LENGTH + 1>`
/// (HuffmanCodingBase.hpp:212).
pub const CODE_LENGTH_STORAGE: usize = MAX_CODE_LENGTH as usize + 1;

// ============================================================================
// BitReader trait (vendor/.../filereader/BitReader.hpp:41-289)
// ============================================================================

/// Reason a bit-reader call failed. Mirrors rapidgzip's
/// `BitReader::EndOfFileReached` exception
/// (BitReader.hpp:83-85) — the only failure mode the Huffman decoders
/// catch. The vendor uses an exception because in practice this happens
/// at most once per stream; we model it as a `Result` because Rust does
/// not have zero-cost exceptions but `Result` is just as cheap when not
/// constructed (the `Ok` path is monomorphized away).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitReaderError {
    /// `BitReader::EndOfFileReached` (BitReader.hpp:83-85).
    Eof,
}

/// LSB-first bit reader interface used by every deflate Huffman decoder.
///
/// Mirrors the subset of `rapidgzip::BitReader<MSB=false, BitBuffer>`
/// (`vendor/.../filereader/BitReader.hpp:194-289`) that the Huffman
/// variants depend on. The vendor's class is templated over
/// `MOST_SIGNIFICANT_BITS_FIRST`; deflate uses the LSB-first
/// instantiation (`gzip::BitReader`).
///
/// Required methods:
/// - [`Self::read`] — read and advance (BitReader.hpp:194-209).
/// - [`Self::peek`] — read without advancing (BitReader.hpp:222-285).
/// - [`Self::seek_after_peek`] — advance after a peek (BitReader.hpp:285-289).
///
/// Default-provided helpers:
/// - [`Self::read_one`] — one-bit read (BitReader.hpp:298-307).
///
/// Implementations on the existing in-tree bit reader
/// (`bit_reader::BitReader`) are provided in this module.
pub trait LsbBitReader {
    /// Read `num_bits` bits and advance the cursor.
    ///
    /// Vendor: `BitReader::read(bit_count_t)`
    /// (BitReader.hpp:194-209).
    fn read(&mut self, num_bits: u8) -> Result<u64, BitReaderError>;

    /// Peek `num_bits` bits without advancing the cursor.
    ///
    /// Vendor: `BitReader::peek(bit_count_t)`
    /// (BitReader.hpp:222-285).
    fn peek(&mut self, num_bits: u8) -> Result<u64, BitReaderError>;

    /// Advance past `num_bits` bits previously returned by [`Self::peek`].
    ///
    /// Vendor: `BitReader::seekAfterPeek(bit_count_t)`
    /// (BitReader.hpp:285-289). Per the vendor contract, calling this
    /// without a matching peek of at least `num_bits` is undefined.
    fn seek_after_peek(&mut self, num_bits: u8);

    /// Read a single bit. Default impl falls back to `read(1)`.
    ///
    /// Vendor: `BitReader::read<1>()` (BitReader.hpp:298-307).
    #[inline]
    fn read_one(&mut self) -> Result<u8, BitReaderError> {
        Ok(self.read(1)? as u8)
    }
}

// --- Adapter for the in-tree `bit_reader::BitReader` --------------------

impl<'a> LsbBitReader for super::bit_reader::BitReader<'a> {
    #[inline]
    fn read(&mut self, num_bits: u8) -> Result<u64, BitReaderError> {
        if self.is_eof() && num_bits > 0 {
            return Err(BitReaderError::Eof);
        }
        Ok(super::bit_reader::BitReader::read(self, num_bits))
    }

    #[inline]
    fn peek(&mut self, num_bits: u8) -> Result<u64, BitReaderError> {
        if num_bits == 0 {
            return Ok(0);
        }
        if !self.can_read(num_bits) {
            return Err(BitReaderError::Eof);
        }
        Ok(self.peek_refilled(num_bits))
    }

    #[inline]
    fn seek_after_peek(&mut self, num_bits: u8) {
        self.skip(num_bits);
    }
}

// --- Adapter for `consume_first_decode::Bits` -----------------------------
//
// `marker_inflate::Block` drives a `Bits` (the in-tree libdeflate-style bit
// reader) for both header parsing and the canonical-Huffman fallback. To
// let the ported `HuffmanCodingSymbolsPerLength::decode` (and any future
// variant) consume the same bit reader rather than requiring a separate
// `BitReader`, we expose `Bits` through `LsbBitReader`. The conversion is
// 1:1 — peek/consume/refill semantics match the vendor's
// `gzip::BitReader::peek<N>` + `seekAfterPeek` pattern
// (vendor/.../filereader/BitReader.hpp:222-289). EOF is signalled via the
// reader's `available()` count rather than an exception (matches the
// vendor's intent — `EndOfFileReached` thrown when bits are unavailable).

impl<'a> LsbBitReader for super::super::inflate::consume_first_decode::Bits<'a> {
    #[inline]
    fn read(&mut self, num_bits: u8) -> Result<u64, BitReaderError> {
        if num_bits == 0 {
            return Ok(0);
        }
        // Refill if needed (mirrors gzip::BitReader::read's implicit
        // refill at BitReader.hpp:196-198).
        if self.available() < num_bits as u32 {
            self.refill();
            if self.available() < num_bits as u32 {
                return Err(BitReaderError::Eof);
            }
        }
        let mask = super::bit_manipulation::n_lowest_bits_set(num_bits);
        let v = super::super::inflate::consume_first_decode::Bits::peek(self) & mask;
        self.consume(num_bits as u32);
        Ok(v)
    }

    #[inline]
    fn peek(&mut self, num_bits: u8) -> Result<u64, BitReaderError> {
        if num_bits == 0 {
            return Ok(0);
        }
        if self.available() < num_bits as u32 {
            self.refill();
            if self.available() < num_bits as u32 {
                return Err(BitReaderError::Eof);
            }
        }
        let mask = super::bit_manipulation::n_lowest_bits_set(num_bits);
        Ok(super::super::inflate::consume_first_decode::Bits::peek(self) & mask)
    }

    #[inline]
    fn seek_after_peek(&mut self, num_bits: u8) {
        // Vendor: `BitReader::seekAfterPeek(bit_count_t)` advances the
        // peek cursor by num_bits without re-checking buffer length —
        // caller is required to have peeked that many bits already.
        self.consume(num_bits as u32);
    }
}

// ============================================================================
// HuffmanCodingBase (HuffmanCodingBase.hpp:16-213)
// ============================================================================

/// Per-bit-length frequency array.
///
/// Mirror of `HuffmanCodingBase::CodeLengthFrequencies =
/// std::array<Frequency, MAX_CODE_LENGTH + 1>` (HuffmanCodingBase.hpp:37).
/// The vendor uses `Frequency = HuffmanCode` (always u16 for deflate);
/// we widen to `u32` because reading raw code lengths into a `u8`-indexed
/// frequency array uses arithmetic that promotes to `int` in C++ anyway.
pub type CodeLengthFrequencies = [u32; CODE_LENGTH_STORAGE];

/// Shared storage and validation for every Huffman variant.
///
/// Mirror of `rapidgzip::HuffmanCodingBase` (HuffmanCodingBase.hpp:16-213).
///
/// The vendor's `MAX_CODE_LENGTH` template parameter is fixed to 15
/// (deflate) — see this module's preamble for why. `MAX_SYMBOL_COUNT`
/// is taken at the variant level since it varies (lit/len: 512, distance:
/// 30, precode: 19) and Rust stable cannot dimension nested arrays by
/// generic const arithmetic.
///
/// `check_optimality` (vendor template param `CHECK_OPTIMALITY`, default
/// true): when true, [`Self::check_code_length_frequencies`] returns
/// `Error::BloatingHuffmanCoding` for non-fully-utilized trees (excepting
/// the single-symbol case). The deflate precode legitimately contains
/// bloated trees per RFC 1951, so callers that decode precode tables
/// pass `false` here.
pub struct HuffmanCodingBase {
    /// `m_minCodeLength` (HuffmanCodingBase.hpp:208).
    /// Initialized to `u8::MAX` so `is_valid()` returns false until
    /// `initialize_min_max_code_lengths` populates it.
    pub min_code_length: u8,
    /// `m_maxCodeLength` (HuffmanCodingBase.hpp:209).
    /// Initialized to 0 (vendor uses `numeric_limits::min()` which for
    /// an unsigned type is 0).
    pub max_code_length: u8,
    /// `m_minimumCodeValuesPerLevel` (HuffmanCodingBase.hpp:212).
    /// Only indexes `[0, max_code_length - min_code_length]` are valid.
    pub minimum_code_values_per_level: [u16; CODE_LENGTH_STORAGE],
}

impl HuffmanCodingBase {
    /// Construct an uninitialized base. Mirror of the default-constructed
    /// `HuffmanCodingBase` (default member initializers at
    /// HuffmanCodingBase.hpp:208-212).
    pub const fn new() -> Self {
        Self {
            min_code_length: u8::MAX,
            max_code_length: 0,
            minimum_code_values_per_level: [0u16; CODE_LENGTH_STORAGE],
        }
    }

    /// Mirror of `HuffmanCodingBase::isValid()`
    /// (HuffmanCodingBase.hpp:39-43).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min_code_length <= self.max_code_length
    }

    /// Mirror of `HuffmanCodingBase::initializeMinMaxCodeLengths`
    /// (HuffmanCodingBase.hpp:46-69).
    ///
    /// Vendor throws `std::invalid_argument` for over-large alphabets;
    /// we surface the same condition as `Error::ExceededSymbolRange`
    /// instead (Rust idiom — see `error.rs:39-40`). Likewise for
    /// `max_code_length > MAX_CODE_LENGTH` (`Error::ExceededClLimit`).
    pub fn initialize_min_max_code_lengths(
        &mut self,
        code_lengths: &[u8],
        max_symbol_count: usize,
    ) -> Error {
        if code_lengths.is_empty() {
            return Error::EmptyAlphabet;
        }
        if code_lengths.len() > max_symbol_count {
            return Error::ExceededSymbolRange;
        }

        // `m_maxCodeLength = getMax( codeLengths );` (HuffmanCodingBase.hpp:61)
        let mut max_len: u8 = 0;
        // `m_minCodeLength = getMinPositive( codeLengths );` (HuffmanCodingBase.hpp:63)
        let mut min_len: u8 = u8::MAX;
        for &cl in code_lengths {
            if cl > max_len {
                max_len = cl;
            }
            if cl > 0 && cl < min_len {
                min_len = cl;
            }
        }

        if max_len > MAX_CODE_LENGTH {
            return Error::ExceededClLimit;
        }

        self.max_code_length = max_len;
        self.min_code_length = min_len;

        Error::None
    }

    /// Mirror of `HuffmanCodingBase::checkCodeLengthFrequencies`
    /// (HuffmanCodingBase.hpp:71-112). The vendor's `CHECK_OPTIMALITY`
    /// template bool is passed in as a runtime arg — the call sites
    /// monomorphize on it via const propagation when invoked through the
    /// variant `initialize_from_lengths` methods.
    pub fn check_code_length_frequencies(
        &self,
        frequencies: &CodeLengthFrequencies,
        code_lengths_size: usize,
        check_optimality: bool,
    ) -> Error {
        let non_zero_count = code_lengths_size - frequencies[0] as usize;
        let mut unused_symbol_count: u32 = 1u32 << self.min_code_length;
        for bit_length in self.min_code_length..=self.max_code_length {
            let frequency = frequencies[bit_length as usize];
            if frequency > unused_symbol_count {
                return Error::InvalidCodeLengths;
            }
            unused_symbol_count -= frequency;
            // "Because we go down one more level for all unused tree nodes!"
            unused_symbol_count *= 2;
        }

        if check_optimality {
            // HuffmanCodingBase.hpp:104-109
            let expected_unused = 1u32 << self.max_code_length;
            if (non_zero_count == 1 && unused_symbol_count != expected_unused)
                || (non_zero_count > 1 && unused_symbol_count != 0)
            {
                return Error::BloatingHuffmanCoding;
            }
        }
        Error::None
    }

    /// Mirror of `HuffmanCodingBase::initializeMinimumCodeValues`
    /// (HuffmanCodingBase.hpp:117-149).
    ///
    /// Resets `frequencies[0]` to 0 (vendor side-effect documented
    /// at HuffmanCodingBase.hpp:122-123).
    pub fn initialize_minimum_code_values(&mut self, frequencies: &mut CodeLengthFrequencies) {
        frequencies[0] = 0;

        let mut min_code: u32 = 0;
        // "minCodeLength might be zero for empty deflate blocks as can happen
        // when compressing an empty file!" — HuffmanCodingBase.hpp:144
        let start = core::cmp::max(1u8, self.min_code_length);
        for bits in start..=self.max_code_length {
            min_code = (min_code + frequencies[(bits - 1) as usize]) << 1;
            self.minimum_code_values_per_level[(bits - self.min_code_length) as usize] =
                min_code as u16;
        }
    }
}

impl Default for HuffmanCodingBase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// checkHuffmanCodeLengths (HuffmanCodingBase.hpp:216-236)
// ============================================================================

/// Mirror of `rapidgzip::checkHuffmanCodeLengths`
/// (HuffmanCodingBase.hpp:216-236).
///
/// Returns true if `code_lengths` forms a complete (fully-utilized)
/// Huffman tree with maximum length `max_code_length`. The single-symbol
/// case (a code length of 1 representing the lone leaf) is also accepted
/// per the vendor's `greaterThanOne == 0` check. `max_code_length`
/// is a runtime arg (rather than the vendor's template parameter)
/// because Rust stable can't have arithmetic in const-generic positions.
pub fn check_huffman_code_lengths(code_lengths: &[u8], max_code_length: u8) -> bool {
    let mut virtual_leaf_count: u64 = 0;
    for &cl in code_lengths {
        if cl > 0 {
            virtual_leaf_count += 1u64 << (max_code_length - cl);
        }
    }
    if virtual_leaf_count == (1u64 << (max_code_length - 1)) {
        let greater_than_one = code_lengths.iter().filter(|&&cl| cl > 1).count();
        return greater_than_one == 0;
    }
    virtual_leaf_count == (1u64 << max_code_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: empty alphabet yields `EmptyAlphabet`.
    #[test]
    fn empty_alphabet_is_error() {
        let mut base = HuffmanCodingBase::new();
        let err = base.initialize_min_max_code_lengths(&[], 286);
        assert_eq!(err, Error::EmptyAlphabet);
    }

    /// Vendor RFC 1951 fixed tree should validate cleanly.
    #[test]
    fn rfc1951_fixed_tree_is_valid() {
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
        assert!(check_huffman_code_lengths(&lens, 15));
    }

    /// Single-symbol case allowed by `check_huffman_code_lengths`.
    #[test]
    fn single_symbol_accepted() {
        // 1 symbol of code length 1.
        let lens = [1u8];
        assert!(check_huffman_code_lengths(&lens, 15));
    }
}
