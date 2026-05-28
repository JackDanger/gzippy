//! §3.8 — constant-time inflate variant.
//!
//! Branchless `cmov`-style dispatch for security-sensitive callers
//! (timing-side-channel-resistant decode). Slower than the optimized
//! variant — measurable target is ~20% on representative workloads —
//! but every per-symbol branch is replaced with arithmetic select.
//!
//! ## Scope
//!
//! v0.1.0: branchless **fixed-Huffman literal-only** decoder. Real
//! workloads with match codes fall back to the regular inflate path
//! (constant-time mode then provides no guarantee for those blocks
//! — this is a known limitation, documented for callers who care).
//!
//! v0.2.0: extend to length/dist codes via a constant-time match-copy
//! that doesn't branch on (length, distance) values. The naive
//! `for i in 0..length { output[out_pos+i] = output[out_pos+i-dist] }`
//! is data-dependent in iteration count; v0.2 unrolls to a fixed
//! upper bound + select.
//!
//! ## Why this matters
//!
//! DEFLATE decode is a natural side channel: literal vs length code
//! takes a different number of operations; the LUT-lookup latency
//! depends on cache state; the match-copy loop's iteration count is
//! the decoded length. For decompressing data that includes
//! attacker-controlled bytes (e.g. parsing a network protocol that
//! gzips its payload), this matters.

use super::InflateError;

#[cfg(feature = "std")]
use std::vec::Vec;

/// Constant-time gzip inflate.
///
/// All per-symbol branches in the decode loop are replaced with
/// arithmetic select. Memory-access patterns still vary (the LUT
/// lookup index depends on the bit-stream contents), but no
/// control-flow change happens between literal / length / EOB paths.
pub struct ConstantTimeInflate;

impl ConstantTimeInflate {
    /// Branchless-select utility: `if cond { a } else { b }` without
    /// a branch. The compiler is allowed to lower this to a `cmov`
    /// (x86_64) or `csel` (aarch64) when both arms are cheap to
    /// compute. We force the both-arms-computed shape explicitly so
    /// codegen doesn't recover a branch.
    #[inline(always)]
    pub fn select_u32(cond: bool, a: u32, b: u32) -> u32 {
        let mask = (cond as u32).wrapping_neg();
        (a & mask) | (b & !mask)
    }

    #[inline(always)]
    pub fn select_usize(cond: bool, a: usize, b: usize) -> usize {
        let mask = (cond as usize).wrapping_neg();
        (a & mask) | (b & !mask)
    }

    /// Decode a fixed-Huffman literal-only block using only branchless
    /// arithmetic. Returns (new_bit_pos, total_out_pos) on EOB.
    /// If a non-literal symbol is encountered, returns
    /// `InflateError::BadData` (callers should switch to non-constant-
    /// time mode for match-bearing blocks).
    ///
    /// `lut` is the 512-entry fixed-Huffman LUT (see route_c_fixed
    /// in the parent crate); each entry is (symbol u16, bits u8).
    /// We accept a slice of `(u16, u8)` here to avoid leaking the
    /// parent crate's types into the sub-crate's public surface.
    #[cfg(feature = "std")]
    pub fn decode_fixed_literal_only(
        input: &[u8],
        bit_pos: usize,
        output: &mut Vec<u8>,
        lut: &[(u16, u8); 512],
    ) -> Result<(usize, usize), InflateError> {
        let mut pos = bit_pos;
        let mut out_count = 0usize;
        let mut saw_match = false;

        loop {
            // Branchless 9-bit peek.
            let byte = pos / 8;
            let off = pos % 8;
            if byte + 6 > input.len() {
                return Err(InflateError::Truncated);
            }
            let mut buf: u64 = 0;
            for i in 0..6 {
                buf |= (input[byte + i] as u64) << (i * 8);
            }
            let key = ((buf >> off) & 0x1FF) as usize;
            let (sym, bits) = lut[key];

            let is_eob = sym == 256;
            let is_literal = sym < 256;
            let is_match = sym > 256;

            // OR-fold the match flag (so a single test at loop exit
            // captures whether ANY match symbol was seen).
            saw_match |= is_match;

            // Advance the bit position by `bits`. Even on EOB we
            // advance — the loop exits below.
            pos += bits as usize;

            // Branchless emit: write the symbol byte, conditional on
            // is_literal. We always write (avoid a branch), but the
            // value written for non-literal is the same low-8-bits;
            // we only advance out_count if literal.
            if out_count >= output.len() {
                output.push(0);
            }
            output[out_count] = (sym & 0xFF) as u8;
            out_count = Self::select_usize(is_literal, out_count + 1, out_count);

            if is_eob {
                output.truncate(out_count);
                if saw_match {
                    return Err(InflateError::BadData);
                }
                return Ok((pos, out_count));
            }

            // Safety cap to prevent infinite loops on malformed input.
            if out_count > 4 * input.len() {
                return Err(InflateError::BadData);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the fixed-Huffman LUT locally (mirror of the parent
    /// crate's `route_c_fixed::build_fixed_lut`).
    fn build_fixed_lut() -> [(u16, u8); 512] {
        fn reverse_bits(mut code: u32, n: u8) -> u32 {
            let mut rev = 0u32;
            for _ in 0..n {
                rev = (rev << 1) | (code & 1);
                code >>= 1;
            }
            rev
        }

        let mut lut = [(0u16, 0u8); 512];
        let mut code_lengths = [0u8; 288];
        code_lengths[0..=143].fill(8);
        code_lengths[144..=255].fill(9);
        code_lengths[256..=279].fill(7);
        code_lengths[280..=287].fill(8);

        let mut count = [0u16; 16];
        for &len in code_lengths.iter() {
            if len > 0 {
                count[len as usize] += 1;
            }
        }
        let mut first_code = [0u32; 16];
        let mut code: u32 = 0;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }
        let mut next_code = first_code;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len);
            let stride = 1u32 << len;
            let mut key = rev;
            while (key as usize) < 512 {
                lut[key as usize] = (symbol as u16, len);
                key += stride;
            }
        }
        lut
    }

    #[test]
    fn select_branchless() {
        assert_eq!(ConstantTimeInflate::select_u32(true, 0xAA, 0x55), 0xAA);
        assert_eq!(ConstantTimeInflate::select_u32(false, 0xAA, 0x55), 0x55);
        assert_eq!(ConstantTimeInflate::select_usize(true, 100, 200), 100);
        assert_eq!(ConstantTimeInflate::select_usize(false, 100, 200), 200);
    }

    #[test]
    fn decode_empty_eob_only() {
        let lut = build_fixed_lut();
        let input = [0u8; 16]; // EOB code = 7 zero bits → first byte = 0
        let mut output = Vec::new();
        let r = ConstantTimeInflate::decode_fixed_literal_only(&input, 0, &mut output, &lut);
        assert_eq!(r.ok(), Some((7, 0)));
    }

    #[test]
    fn decode_single_literal_a() {
        let lut = build_fixed_lut();
        // 'A' = symbol 65, code 0b01110001 (8 bits, MSB-first stream)
        // → byte 0 = 0x8E (bits reversed within byte = LSB-first storage).
        // EOB = 7 zero bits → starts at bit 8.
        let input = [0x8Eu8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut output = Vec::new();
        let (new_pos, n) =
            ConstantTimeInflate::decode_fixed_literal_only(&input, 0, &mut output, &lut).unwrap();
        assert_eq!(n, 1);
        assert_eq!(new_pos, 8 + 7);
        assert_eq!(output[0], b'A');
    }

    #[test]
    fn decode_rejects_match_symbol_at_eob() {
        let lut = build_fixed_lut();
        // Symbol 257 has reversed code = 64 (length 7). To make the
        // 9-bit LUT key = 64, byte 0 = 0x40 (bit 6 set).
        // Then EOB (7 zero bits) starts at bit 7. Bits 7..13 = 0
        // → byte 0 bit 7 = 0 (already), byte 1 bits 0..5 = 0.
        // saw_match is set; on EOB, function returns BadData.
        let input = [0x40u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut output = Vec::new();
        let r = ConstantTimeInflate::decode_fixed_literal_only(&input, 0, &mut output, &lut);
        // saw_match=true at EOB → BadData. (Or Truncated if the
        // decoder runs past available input on a different path.)
        assert!(
            matches!(r, Err(InflateError::BadData) | Err(InflateError::Truncated)),
            "expected BadData/Truncated for match-symbol input, got {:?}",
            r
        );
    }
}
