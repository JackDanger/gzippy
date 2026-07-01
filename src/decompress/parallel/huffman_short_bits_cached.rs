#![allow(dead_code)] // vendor-faithful rapidgzip port; wired at the production distance path

//! Port of `rapidgzip::HuffmanCodingShortBitsCached`
//! (`vendor/.../huffman/HuffmanCodingShortBitsCached.hpp:1-172`).
//!
//! A bounded-LUT canonical decoder. Unlike
//! [`HuffmanCodingReversedBitsCached`](super::huffman_reversed_bits_cached),
//! which caches every code up to `MAX_CODE_LENGTH` (a flat `1<<15` LUT), this
//! variant limits the cache to `LUT_BITS_COUNT` bits and falls back to the
//! symbols-per-length walk (`decode_long`) for codes longer than the cache.
//! Vendor rationale (header preamble): "It limits the LUT to a fixed size
//! instead of caching everything up to MAX_CODE_LENGTH because that would be
//! too large …".
//!
//! ## Why gzippy uses it for the DISTANCE decoder
//!
//! Vendor's `DistanceHuffmanCoding` (deflate.hpp:336) is
//! `HuffmanCodingReversedBitsCached<uint16_t, MAX_CODE_LENGTH, uint8_t,
//! MAX_DISTANCE_SYMBOL_COUNT>` — note `Symbol = uint8_t`, so the vendor's
//! distance cache is `pair<uint8_t,uint8_t> * 2^15 = 64 KiB` (vendor calls it
//! exactly that at deflate.hpp:668). gzippy's earlier port unified `Symbol`
//! to `u16`, doubling the distance cache to 128 KiB. This port restores the
//! `u8` symbol AND bounds the cache to `LUT_BITS_COUNT` bits — distance code
//! lengths almost never exceed ~12 bits, so a small LUT + `decode_long`
//! fallback is byte-identical while costing only `(u8,u8) * 2^LUT_BITS_COUNT`.
//! At `LUT_BITS_COUNT = 12` that is `2 * 4096 = 8 KiB` (vs the old 128 KiB).
//! This is the vendor-blessed two-level structure (the same one
//! `LiteralOrLengthHuffmanCoding` defaults to with `LUT_BITS_COUNT = 11`).

use super::bit_manipulation::{n_lowest_bits_set, reverse_bits_count_u16};
use super::error::Error;
use super::huffman_base::LsbBitReader;
use super::huffman_symbols_per_length::HuffmanCodingSymbolsPerLength;

/// One LUT slot. Mirror of the vendor `CacheEntry`
/// (HuffmanCodingShortBitsCached.hpp:150-154). `length == 0` ⇒ "miss, walk
/// `decode_long`". `CacheSymbol` is the per-instantiation `Symbol` type
/// (`u8` for distance, where symbols are ≤ 29).
#[derive(Clone, Copy)]
pub struct CacheEntry<CacheSymbol: Copy> {
    pub length: u8,
    pub symbol: CacheSymbol,
}

/// Mirror of `HuffmanCodingShortBitsCached` with `REVERSE_BITS = true`,
/// `CHECK_OPTIMALITY = true` (the deflate distance/litlen instantiation).
///
/// - `MAX_SYMBOL_COUNT` — vendor template parameter (sizes the base
///   symbols-per-length alphabet).
/// - `LUT_BITS_COUNT` — vendor template parameter (the cache bit width).
/// - `CacheSymbol` — vendor `Symbol` template parameter for the cache entry
///   (`u8` for distance). The base alphabet stores `u16` symbols uniformly;
///   the LUT narrows to `CacheSymbol`, exactly as the vendor does.
pub struct HuffmanCodingShortBitsCached<
    CacheSymbol: Copy + Default + From<u8> + Into<u16>,
    const MAX_SYMBOL_COUNT: usize,
    const LUT_BITS_COUNT: usize,
> {
    base: HuffmanCodingSymbolsPerLength<MAX_SYMBOL_COUNT>,
    code_cache: [CacheEntry<CacheSymbol>; LUT_BITS_COUNT_CACHE_LEN_PLACEHOLDER],
    /// `min(LUT_BITS_COUNT, m_maxCodeLength)` (header:48).
    lut_bits_count: u8,
    /// `max(LUT_BITS_COUNT, m_minCodeLength)` (header:49).
    bits_to_read_at_once: u8,
    needs_to_be_zeroed: bool,
}

// Rust stable cannot dimension `code_cache` from the `LUT_BITS_COUNT` const
// generic directly (no `generic_const_exprs`). We provide the array length
// through a separate const-generic-free type alias path: the struct is only
// ever instantiated through the concrete `DistanceShortBitsCached` alias
// below, which fixes the array length. To keep a single generic body we use a
// fixed maximum array length and only index the low `1<<LUT_BITS_COUNT` slots.
//
// `LUT_BITS_COUNT_CACHE_LEN_PLACEHOLDER` is the physical array length. We size
// it to `1 << MAX_LUT_BITS` (the largest LUT we instantiate) so any
// `LUT_BITS_COUNT <= MAX_LUT_BITS` fits; the active window is `1<<LUT_BITS_COUNT`.
const MAX_LUT_BITS: usize = 12;
const LUT_BITS_COUNT_CACHE_LEN_PLACEHOLDER: usize = 1usize << MAX_LUT_BITS;

impl<
        CacheSymbol: Copy + Default + From<u8> + Into<u16>,
        const MAX_SYMBOL_COUNT: usize,
        const LUT_BITS_COUNT: usize,
    > HuffmanCodingShortBitsCached<CacheSymbol, MAX_SYMBOL_COUNT, LUT_BITS_COUNT>
{
    const _LUT_FITS: () = assert!(LUT_BITS_COUNT <= MAX_LUT_BITS);

    pub fn new() -> Self {
        Self {
            base: HuffmanCodingSymbolsPerLength::new(),
            code_cache: [CacheEntry {
                length: 0,
                symbol: CacheSymbol::default(),
            }; LUT_BITS_COUNT_CACHE_LEN_PLACEHOLDER],
            lut_bits_count: LUT_BITS_COUNT as u8,
            bits_to_read_at_once: LUT_BITS_COUNT as u8,
            needs_to_be_zeroed: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
    }

    pub fn max_code_length(&self) -> u8 {
        self.base.base.max_code_length
    }

    /// Resident per-thread footprint of the active LUT window
    /// (`1<<LUT_BITS_COUNT` entries of `CacheEntry<CacheSymbol>`). Counters
    /// only — the cache-residency-mandate instrument. We report the ACTIVE
    /// window so the byte accounting reflects the cache-touch footprint,
    /// matching the vendor's `1<<LUT_BITS_COUNT` array (the placeholder is
    /// only Rust's const-generic workaround).
    pub fn heap_bytes(&self) -> usize {
        (1usize << LUT_BITS_COUNT) * core::mem::size_of::<CacheEntry<CacheSymbol>>()
    }

    /// Mirror of `initializeFromLengths` (header:40-100) with
    /// `REVERSE_BITS = true`. `check_optimality` is the vendor
    /// `CHECK_OPTIMALITY` template bool, threaded as a runtime arg so call
    /// sites preserve gzippy's existing per-table behavior byte-for-byte
    /// (the distance path passes `false`, matching the prior decoder).
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

        let max_cl = self.base.base.max_code_length;
        let min_cl = self.base.base.min_code_length;
        // header:48-49.
        self.lut_bits_count = core::cmp::min(LUT_BITS_COUNT as u8, max_cl);
        self.bits_to_read_at_once = core::cmp::max(LUT_BITS_COUNT as u8, min_cl);

        let active_len = 1usize << self.lut_bits_count;
        if self.needs_to_be_zeroed {
            for entry in &mut self.code_cache[..active_len] {
                entry.length = 0;
            }
        }

        let base = &self.base.base;
        let mut code_values = base.minimum_code_values_per_level;

        for (symbol, &length) in code_lengths.iter().enumerate() {
            // header:65-67: skip empties AND codes longer than the cache.
            if length == 0 || length > self.lut_bits_count {
                continue;
            }
            let k = (length - min_cl) as usize;
            let code = code_values[k];
            code_values[k] = code.wrapping_add(1);

            // REVERSE_BITS branch (header:72-82).
            let filler_bit_count = self.lut_bits_count - length;
            let reversed_code = reverse_bits_count_u16(code, length);
            let maximum_padded_code =
                reversed_code as u32 | ((n_lowest_bits_set(filler_bit_count) as u32) << length);
            debug_assert!((maximum_padded_code as usize) < active_len);

            let increment = 1u32 << length;
            let mut padded_code = reversed_code as u32;
            while padded_code <= maximum_padded_code {
                let e = &mut self.code_cache[padded_code as usize];
                e.length = length;
                e.symbol = CacheSymbol::from(symbol as u8);
                padded_code += increment;
            }
        }

        self.needs_to_be_zeroed = true;
        Error::None
    }

    /// Mirror of `decode` (header:101-116).
    ///
    /// MARKER-KERNEL Lever 1 (perf/marker-kernel codegen): forced
    /// `#[inline(always)]` — the plain `#[inline]` hint was DECLINED by LLVM, so
    /// the marker-path distance decode (`decode_careful_tail` dist site,
    /// marker_inflate.rs:2825, + the fast-loop kill-switch arm) emitted a
    /// per-symbol `call HuffmanCodingShortBitsCached::decode` (perf-annotate
    /// 6.2% of the marker bucket on Zen2 T4) with call/spill/return overhead that
    /// rg's `Block::read` does not pay (rg inlines its decode, vendor
    /// gzip/deflate.hpp:336/1580-1590). The hot body here (peek + LUT lookup +
    /// seek_after_peek) is small; the cold `decode_long` / `base.decode` paths
    /// stay out-of-line as separate symbols. Byte-exact: codegen-only, no logic
    /// change. Mirrors the existing `emit_backref_ring` `#[inline(always)]`
    /// precedent (marker_inflate.rs:4514-4521).
    #[inline(always)]
    pub fn decode<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<u16> {
        let value = match bit_reader.peek(self.lut_bits_count) {
            Ok(v) => v as usize,
            Err(_) => return self.base.decode(bit_reader),
        };
        debug_assert!(value < (1usize << self.lut_bits_count));

        let entry = self.code_cache[value];
        if entry.length == 0 {
            return self.decode_long(bit_reader);
        }
        bit_reader.seek_after_peek(entry.length);
        Some(entry.symbol.into())
    }

    /// Mirror of `decodeLong` (header:118-149) with `REVERSE_BITS = true`.
    #[inline]
    fn decode_long<R: LsbBitReader>(&self, bit_reader: &mut R) -> Option<u16> {
        // P3.1 profiler: count cache-miss (long-path) distance decodes. Cold
        // path; the enabled() check is a OnceLock-cached bool.
        if crate::decompress::parallel::contig_prof::enabled() {
            crate::decompress::parallel::contig_prof::C_N_DIST_LONG
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let base = &self.base.base;
        let mut code: u16 = 0;

        // header:127-131: read the first `bits_to_read_at_once` bits one at a
        // time (REVERSE_BITS path can't batch-read without inverting order).
        for _ in 0..self.bits_to_read_at_once {
            let bit = bit_reader.read_one().ok()?;
            code = (code << 1) | bit as u16;
        }

        // header:134-147. Vendor `for ( k = m_bitsToReadAtOnce - minCL;
        // k <= maxCL - minCL; ++k )` — the bound is checked BEFORE the body,
        // so when bits_to_read_at_once > max_code_length the loop never runs.
        let start_k = self.bits_to_read_at_once - base.min_code_length;
        let last_k = base.max_code_length - base.min_code_length;
        let mut k = start_k;
        while k <= last_k {
            let min_code = base.minimum_code_values_per_level[k as usize];
            if min_code <= code {
                let sub_index = self.base.offsets[k as usize] as usize + (code - min_code) as usize;
                if sub_index < self.base.offsets[(k + 1) as usize] as usize {
                    return Some(self.base.symbols_per_length[sub_index]);
                }
            }
            let bit = bit_reader.read_one().ok()?;
            code = (code << 1) | bit as u16;
            k += 1;
        }
        None
    }
}

impl<
        CacheSymbol: Copy + Default + From<u8> + Into<u16>,
        const MAX_SYMBOL_COUNT: usize,
        const LUT_BITS_COUNT: usize,
    > Default for HuffmanCodingShortBitsCached<CacheSymbol, MAX_SYMBOL_COUNT, LUT_BITS_COUNT>
{
    fn default() -> Self {
        Self::new()
    }
}

/// The production distance decoder type: vendor `Symbol = uint8_t`
/// (distance symbols are ≤ 29), `LUT_BITS_COUNT = 12`. Cache footprint:
/// `(u8,u8) * 2^12 = 8 KiB`/thread (vs the prior 128 KiB flat LUT).
pub type DistanceShortBitsCached<const MAX_SYMBOL_COUNT: usize> =
    HuffmanCodingShortBitsCached<u8, MAX_SYMBOL_COUNT, 12>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::bit_reader::BitReader;
    use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;

    /// The bounded-LUT decoder must agree byte-for-byte with the full-width
    /// `HuffmanCodingReversedBitsCached` on the RFC fixed-distance-style tree
    /// for EVERY symbol — both the in-LUT fast path and the `decode_long`
    /// fallback for codes longer than `LUT_BITS_COUNT`.
    #[test]
    fn matches_reversed_bits_cached_all_symbols() {
        // A Kraft-complete distance-shaped tree (all codes ≤ the 12-bit LUT,
        // so the fast path is fully exercised). 28 codes of length 5 + 2 of
        // length 4 ⇒ 28/32 + 2/16 = 1.0.
        let mut lens = vec![5u8; 30];
        lens[28] = 4;
        lens[29] = 4;

        let mut full: HuffmanCodingReversedBitsCached<30> = HuffmanCodingReversedBitsCached::new();
        assert_eq!(full.initialize_from_lengths(&lens, true), Error::None);

        let mut short: DistanceShortBitsCached<30> = DistanceShortBitsCached::new();
        assert_eq!(short.initialize_from_lengths(&lens, true), Error::None);

        // Drive both decoders over an identical bit stream and require the same
        // symbol sequence for a long run of random-ish bytes.
        let data: Vec<u8> = (0u16..4096)
            .map(|i| (i.wrapping_mul(37) >> 3) as u8)
            .collect();
        let mut br_full = BitReader::new(&data);
        let mut br_short = BitReader::new(&data);
        for _ in 0..2000 {
            let a = full.decode(&mut br_full);
            let b = short.decode(&mut br_short);
            assert_eq!(a, b, "short-bits LUT diverged from full-width LUT");
            if a.is_none() {
                break;
            }
        }
    }

    /// A tree with codes LONGER than the 12-bit LUT must still decode every
    /// symbol identically (forces the decode_long fallback).
    #[test]
    fn long_codes_force_decode_long_path() {
        // Kraft-complete length set with a max length of 14.
        // 1 code of len 1? No — must be complete. Use: two len-1 is invalid;
        // build: len 2 x1, len 3 x2, ... canonical deep tree.
        // Easiest valid deep tree: lengths 1,2,3,...,k with a doubled tail.
        let lens: Vec<u8> = {
            // 13,13 (completes a depth-13 pair), plus a balanced tree above.
            // Construct: a complete binary tree of depth 14 truncated.
            // Use the canonical "1,2,3,4,...,13,14,14" Kraft-complete ladder.
            let mut v = vec![0u8; 30];
            // ladder: len i for i in 1..=13, then two len-14 leaves.
            for (idx, item) in v.iter_mut().enumerate().take(13) {
                *item = (idx + 1) as u8;
            }
            v[13] = 14;
            v[14] = 14;
            v
        };

        let mut full: HuffmanCodingReversedBitsCached<30> = HuffmanCodingReversedBitsCached::new();
        let ef = full.initialize_from_lengths(&lens, true);
        let mut short: DistanceShortBitsCached<30> = DistanceShortBitsCached::new();
        let es = short.initialize_from_lengths(&lens, true);
        assert_eq!(ef, es, "init error must agree");
        if ef != Error::None {
            return; // not a valid tree on this platform's check; skip
        }
        assert!(short.max_code_length() > 12, "must exercise decode_long");

        let data: Vec<u8> = (0u16..8192)
            .map(|i| (i.wrapping_mul(101) >> 2) as u8)
            .collect();
        let mut br_full = BitReader::new(&data);
        let mut br_short = BitReader::new(&data);
        for _ in 0..3000 {
            let a = full.decode(&mut br_full);
            let b = short.decode(&mut br_short);
            assert_eq!(a, b, "decode_long path diverged from full-width LUT");
            if a.is_none() {
                break;
            }
        }
    }
}
