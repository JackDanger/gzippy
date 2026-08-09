//! The far-len-3 cost gate (levels 2-9, greedy + lazy).
//!
//! Vendor diff (the reason this exists): zlib's deflate_fast/deflate_slow —
//! gzip and pigz at every level — accept ANY len-3 match the finder returns,
//! while our greedy/lazy (ported from libdeflate) refuse len-3 at
//! offset > 4096 (greedy) / > 8192 (lazy). On high-entropy content where three
//! literals cost more than a far len-3 match (dd79_bin6: 6.54 bits/byte) the
//! fixed guard donates ~41 KB at L2 / ~25 KB at L3 to gzip (trainer causal
//! probe 2026-08-09, memory leaf project_len3_guard_dd79_mechanism.md).
//!
//! The UNCONDITIONAL drop is measured dead: tie-guard levels 1-9 flipped 53 of
//! 107 T1 libdeflate ties worse (weights.safetensors +203,233 B, tool.bin
//! +135,452 B, sil40 L2 +47,819 B) — len-3 distance policy is
//! content-dependent, so a fixed constant is wrong in BOTH directions.
//!
//! This gate is the cost-aware middle: a len-3 match beyond the old cutoff is
//! accepted only when the block's RUNNING symbol statistics say it is strictly
//! cheaper than the three literals it replaces. It is a parser-internal cost
//! model in the exact mold of `recalculate_min_match_len` (a fixed function of
//! data the encoder already read — NOT a content/corpus detector, see
//! ldx/min_match.rs's clause-3 note), recomputed at the same cadence, and it
//! fails CLOSED: with no evidence (any zero frequency it needs), the mask is 0
//! and behavior is byte-identical to the shipped guard.

use super::super::tables::{
    offset_slot, DEFLATE_FIRST_LEN_SYM, DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS,
    OFFSET_EXTRA_BITS,
};
use super::{bsr32, NUM_LITERALS};

/// Margin (in eighth-bit units) a far len-3 match must clear BELOW the
/// estimated cost of three literals before the gate accepts it. `8` = one
/// full bit: the entropy model prices symbols at fractional ideal cost while
/// the real code pays integer bits and shares one litlen alphabet, so a
/// sub-bit paper win is inside the model's own error.
const FAR_LEN3_MARGIN_EIGHTH_BITS: u32 = 8;

/// Deterministic fixed-point `log2(x)` in eighth-bit units (3 fractional
/// bits), integer-only. No libm: size is arch-invariant and must stay so —
/// a platform-dependent `log2f` ulp would change output bytes per arch.
/// Monotone non-decreasing in `x` (truncation of a monotone function), which
/// [`far_len3_slot_mask`] relies on for its subtractions to stay non-negative.
fn log2_fp3(x: u32) -> u32 {
    debug_assert!(x != 0);
    let int_part = bsr32(x);
    // Q30 mantissa in [1, 2): squaring fits u64 ((2^31)^2 = 2^62).
    let mut m: u64 = ((x as u64) << 30) >> int_part;
    let mut frac = 0u32;
    for _ in 0..3 {
        m = (m * m) >> 30;
        frac <<= 1;
        if m >= (2u64 << 30) {
            frac |= 1;
            m >>= 1;
        }
    }
    (int_part << 3) | frac
}

/// Build the per-offset-slot accept mask for far len-3 matches from the
/// block's running frequencies. Bit `s` set = a len-3 match whose offset
/// falls in slot `s` beats three literals under the running entropy model:
///
///   len3_sym + dist_sym(s) + dist_extra(s) + MARGIN  <=  3 * avg_literal
///
/// where symbol costs are ideal code lengths `log2(total/freq)` from the
/// block so far and `dist_extra` is the exact RFC 1951 extra-bit count.
/// Slots with zero observed frequency stay disallowed (no evidence — the
/// distance alphabet's cost there is unknowable from this block), so the
/// mask can only open where longer far matches already paid for the slot.
pub(super) fn far_len3_slot_mask(
    litlen_freqs: &[u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: &[u32; DEFLATE_NUM_OFFSET_SYMS],
) -> u32 {
    // Length 3 is length-slot 0 => symbol DEFLATE_FIRST_LEN_SYM (257).
    let f_len3 = litlen_freqs[DEFLATE_FIRST_LEN_SYM];
    if f_len3 == 0 {
        return 0;
    }
    let lit_occurrences: u64 = litlen_freqs[..NUM_LITERALS].iter().map(|&f| f as u64).sum();
    if lit_occurrences == 0 {
        return 0;
    }
    let total_off: u32 = offset_freqs.iter().sum();
    if total_off == 0 {
        return 0;
    }
    let total_litlen: u64 = litlen_freqs.iter().map(|&f| f as u64).sum();
    // A block holds well under 2^32 symbols (blocks are length-bounded), so
    // the u32 narrowing is exact.
    debug_assert!(total_litlen <= u32::MAX as u64);
    let log_ll = log2_fp3(total_litlen as u32);
    // Mean literal cost over literal OCCURRENCES under the shared litlen
    // alphabet: sum f_i * (log2 T - log2 f_i) / sum f_i.
    let mut acc: u64 = 0;
    for &f in &litlen_freqs[..NUM_LITERALS] {
        if f > 0 {
            acc += f as u64 * (log_ll - log2_fp3(f)) as u64;
        }
    }
    let three_lit_bits = 3 * (acc / lit_occurrences) as u32;
    let len3_sym_bits = log_ll - log2_fp3(f_len3);
    let log_off = log2_fp3(total_off);
    let mut mask = 0u32;
    for (s, &extra) in OFFSET_EXTRA_BITS.iter().enumerate() {
        let fo = offset_freqs[s];
        if fo == 0 {
            continue;
        }
        let match_bits = len3_sym_bits + (log_off - log2_fp3(fo)) + ((extra as u32) << 3);
        if match_bits + FAR_LEN3_MARGIN_EIGHTH_BITS <= three_lit_bits {
            mask |= 1 << s;
        }
    }
    mask
}

/// Per-position accept test against a [`far_len3_slot_mask`] mask. Called
/// only on the cold arm (len-3 candidate already past the old offset
/// cutoff): one table lookup + shift.
#[inline(always)]
pub(super) fn far_len3_allowed(mask: u32, offset: u32) -> bool {
    mask & (1u32 << offset_slot(offset)) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact on powers of two and monotone non-decreasing everywhere — the
    /// property the mask's subtractions (`log_total - log2(f)`) rely on to
    /// stay non-negative.
    #[test]
    fn log2_fp3_is_exact_on_powers_of_two_and_monotone() {
        for k in 0..32u32 {
            assert_eq!(log2_fp3(1 << k), k << 3, "2^{k}");
        }
        let mut prev = 0;
        for x in 1..100_000u32 {
            let cur = log2_fp3(x);
            assert!(cur >= prev, "log2_fp3 decreased at {x}");
            prev = cur;
        }
        // Spot value: log2(3) = 1.585 -> 12.68 eighth-bits, truncated 12.
        assert_eq!(log2_fp3(3), 12);
    }

    /// The gate fails CLOSED: with no len-3 symbol yet, no literals, or no
    /// offsets, the mask is 0 and behavior is the shipped fixed guard.
    #[test]
    fn no_evidence_means_no_accept() {
        let ll = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let of = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        assert_eq!(far_len3_slot_mask(&ll, &of), 0);

        // Literals + offsets but no len-3 symbol observed: still closed.
        let mut ll2 = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        for f in ll2[..NUM_LITERALS].iter_mut() {
            *f = 100;
        }
        let of2 = [10u32; DEFLATE_NUM_OFFSET_SYMS];
        assert_eq!(far_len3_slot_mask(&ll2, &of2), 0);
    }

    /// The dd79 shape (near-uniform expensive literals, frequent len-3,
    /// used far-distance slots) opens far slots; the text shape (cheap
    /// literals) keeps them closed. This is the content dependence the two
    /// 2026-08-09 findings bracket, as a pinned unit fact.
    #[test]
    fn expensive_literals_open_the_mask_and_cheap_literals_do_not() {
        // ~8-bit literals: 256 literals, uniform.
        let mut hi = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        for f in hi[..NUM_LITERALS].iter_mut() {
            *f = 1000;
        }
        hi[DEFLATE_FIRST_LEN_SYM] = 30_000; // len-3 is common
        let mut offs = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for f in offs.iter_mut() {
            *f = 1000; // every slot has evidence
        }
        let mask_hi = far_len3_slot_mask(&hi, &offs);
        // Slot 0 (distance 1, 0 extra bits) must be open: ~2.6-bit len-3
        // symbol + ~4.9-bit distance symbol vs 3 * ~8.2-bit literals.
        assert_ne!(mask_hi & 1, 0, "high-entropy shape must open near slots");

        // ~4-bit literals: 16 literals, uniform — three literals (~12.6 bits)
        // are cheaper than any far len-3; every far slot must stay closed.
        let mut lo = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        for f in lo[..16].iter_mut() {
            *f = 16_000;
        }
        lo[DEFLATE_FIRST_LEN_SYM] = 30_000;
        let mask_lo = far_len3_slot_mask(&lo, &offs);
        // Far slots (>= slot 24: distances > 4096, 11+ extra bits).
        assert_eq!(
            mask_lo >> 24,
            0,
            "cheap literals must keep far slots closed"
        );
    }
}
