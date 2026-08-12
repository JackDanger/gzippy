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
//! Two simpler policies are measured dead, both on tie-guard levels 1-9
//! (107 T1 libdeflate byte-ties, bar = non-worse on EVERY tie):
//!
//! * UNCONDITIONAL drop: 53 flips (weights.safetensors +203,233 B, tool.bin
//!   +135,452 B, sil40 L2 +47,819 B). len-3 distance policy is
//!   content-dependent; a fixed constant is wrong in both directions.
//! * MEAN-literal-cost gate (accept when cost model says a far len-3 beats
//!   3 * occurrence-weighted average literal cost): 28 flips (tool.bin
//!   +77,501 B, sil40 L2 +28,848 B, minjs +3,804 B). The three bytes a len-3
//!   match replaces belong to a REPEATED trigram — typically common, cheap
//!   bytes — so the block-mean overprices exactly the literals being
//!   replaced and the gate over-accepts on mixed content.
//!
//! This version prices the EXACT three bytes at the candidate position with a
//! per-symbol running cost table. It is a parser-internal cost model in the
//! mold of `recalculate_min_match_len` (a fixed function of data the encoder
//! already read — NOT a content/corpus detector, see ldx/min_match.rs's
//! clause-3 note), recomputed at the same cadence, and it fails CLOSED: with
//! no evidence (any zero frequency it needs), the gate is inert and behavior
//! is byte-identical to the shipped fixed guard.

use super::super::tables::{
    offset_slot, DEFLATE_FIRST_LEN_SYM, DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS,
    OFFSET_EXTRA_BITS,
};
use super::{bsr32, NUM_LITERALS};

/// Margin (in eighth-bit units) a far len-3 match must clear BELOW the
/// estimated cost of the three literals it replaces before the gate accepts
/// it. The entropy model prices symbols at fractional ideal cost while the
/// real code pays integer bits and shares one litlen alphabet, so a paper
/// win below the margin is inside the model's own error. Two bits, paired
/// with greedy's +1 shadow probe (see `greedy.rs`).
pub(super) const GREEDY_MARGIN_EIGHTH_BITS: u32 = 16;

/// Deterministic fixed-point `log2(x)` in eighth-bit units (3 fractional
/// bits), integer-only. No libm: size is arch-invariant and must stay so —
/// a platform-dependent `log2f` ulp would change output bytes per arch.
/// Monotone non-decreasing in `x` (truncation of a monotone function), which
/// [`FarLen3Gate::recalc`] relies on for its subtractions to stay
/// non-negative.
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

/// A closed slot's sentinel cost: large enough that no literal sum reaches
/// it, small enough that `+ MARGIN` cannot overflow.
const CLOSED: u32 = u32::MAX / 2;

/// The per-block running cost tables for the far-len-3 accept decision.
/// Rebuilt from the block's own frequencies at the parser's existing recalc
/// cadence; starts [`FarLen3Gate::INERT`] each block (= the shipped fixed
/// guard).
pub(super) struct FarLen3Gate {
    /// Full cost (eighth-bits) of a len-3 match per offset slot: len-3
    /// litlen symbol + distance symbol + exact RFC 1951 distance extra bits,
    /// plus the accept margin. [`CLOSED`] where the slot has zero observed
    /// frequency (no evidence — the distance alphabet's cost there is
    /// unknowable from this block, so it stays refused).
    match_cost: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// Ideal running cost (eighth-bits) per literal byte value,
    /// `log2(total_litlen / freq)`; unseen bytes are priced as freq-1
    /// (a novel byte is the most expensive literal the block can emit).
    lit_cost: [u32; 256],
    /// False = every slot closed; lets the parser skip the lookups.
    any_open: bool,
}

impl FarLen3Gate {
    /// The do-nothing gate: every slot closed, identical to the shipped
    /// fixed offset guard.
    pub(super) const INERT: Self = Self {
        match_cost: [CLOSED; DEFLATE_NUM_OFFSET_SYMS],
        lit_cost: [0; 256],
        any_open: false,
    };

    /// Rebuild from the block's running frequencies. `margin_eighth_bits` is
    /// the caller's accept margin (see the per-parser constants above).
    pub(super) fn recalc(
        litlen_freqs: &[u32; DEFLATE_NUM_LITLEN_SYMS],
        offset_freqs: &[u32; DEFLATE_NUM_OFFSET_SYMS],
        margin_eighth_bits: u32,
    ) -> Self {
        // Length 3 is length-slot 0 => symbol DEFLATE_FIRST_LEN_SYM (257).
        let f_len3 = litlen_freqs[DEFLATE_FIRST_LEN_SYM];
        let total_off: u32 = offset_freqs.iter().sum();
        // Absolute evidence floor: on near-incompressible content the block
        // has only sparse (often chance) matches, and ideal costs computed
        // from a small sample are noise. Below 1024 observed offsets the
        // gate stays inert.
        if f_len3 == 0 || total_off < 1024 {
            return Self::INERT;
        }
        let total_litlen: u64 = litlen_freqs.iter().map(|&f| f as u64).sum();
        // A block holds well under 2^32 symbols (blocks are length-bounded),
        // so the u32 narrowing is exact.
        debug_assert!(total_litlen <= u32::MAX as u64);
        let log_ll = log2_fp3(total_litlen as u32);
        let mut lit_cost = [0u32; 256];
        for (b, &f) in litlen_freqs[..NUM_LITERALS].iter().enumerate() {
            // Unseen byte: price as if it occurred once (log2(T/1) = log_ll),
            // the dearest a literal can be under this block's code.
            lit_cost[b] = log_ll - log2_fp3(f.max(1));
        }
        let len3_sym_bits = log_ll - log2_fp3(f_len3);
        let log_off = log2_fp3(total_off);
        let mut match_cost = [CLOSED; DEFLATE_NUM_OFFSET_SYMS];
        let mut any_open = false;
        for (s, &extra) in OFFSET_EXTRA_BITS.iter().enumerate() {
            let fo = offset_freqs[s];
            // Evidence floor: a slot must hold at least 1/64 of the block's
            // offsets before its running cost is trusted. A thin slot's ideal
            // cost is dominated by sampling noise (a handful of chance far
            // matches make it look cheap), and mixed content pays for that
            // optimism block-wide.
            if fo == 0 || (fo as u64) * 64 < total_off as u64 {
                continue;
            }
            match_cost[s] = len3_sym_bits
                + (log_off - log2_fp3(fo))
                + ((extra as u32) << 3)
                + margin_eighth_bits;
            any_open = true;
        }
        Self {
            match_cost,
            lit_cost,
            any_open,
        }
    }

    /// Per-position accept test: does a len-3 match at `offset` beat the
    /// THREE ACTUAL BYTES `b0 b1 b2` it would replace? Called only on the
    /// cold arm (len-3 candidate already past the old fixed cutoff).
    #[inline(always)]
    pub(super) fn allows(&self, offset: u32, b0: u8, b1: u8, b2: u8) -> bool {
        if !self.any_open {
            return false;
        }
        let lits =
            self.lit_cost[b0 as usize] + self.lit_cost[b1 as usize] + self.lit_cost[b2 as usize];
        self.match_cost[offset_slot(offset) as usize] <= lits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact on powers of two and monotone non-decreasing everywhere — the
    /// property the cost subtractions (`log_total - log2(f)`) rely on to
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

    /// The gate fails CLOSED: with no len-3 symbol yet or no offsets, the
    /// gate is inert and behavior is the shipped fixed guard.
    #[test]
    fn no_evidence_means_no_accept() {
        let ll = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let of = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        let g = FarLen3Gate::recalc(&ll, &of, GREEDY_MARGIN_EIGHTH_BITS);
        assert!(!g.any_open);
        assert!(!g.allows(32000, 0, 0, 0));

        // Literals + offsets but no len-3 symbol observed: still closed.
        let mut ll2 = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        for f in ll2[..NUM_LITERALS].iter_mut() {
            *f = 100;
        }
        let of2 = [10u32; DEFLATE_NUM_OFFSET_SYMS];
        assert!(!FarLen3Gate::recalc(&ll2, &of2, GREEDY_MARGIN_EIGHTH_BITS).any_open);

        // INERT refuses everything by construction.
        assert!(!FarLen3Gate::INERT.allows(1, 255, 255, 255));
    }

    /// The content-dependence bracket, pinned: a far len-3 replacing three
    /// EXPENSIVE (rare) bytes is accepted; the same match replacing three
    /// CHEAP (dominant) bytes is refused — under one and the same block
    /// statistics. This is the mechanism that separates dd79_bin6 (uniform
    /// expensive literals) from mixed content where the replaced trigram is
    /// made of common bytes.
    #[test]
    fn prices_the_actual_bytes_not_the_block_mean() {
        let mut ll = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        // A dominant cheap byte and 255 rare ones.
        ll[0] = 200_000;
        for f in ll[1..NUM_LITERALS].iter_mut() {
            *f = 100;
        }
        ll[DEFLATE_FIRST_LEN_SYM] = 30_000; // len-3 is common
        let mut of = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for f in of.iter_mut() {
            *f = 1000;
        }
        let g = FarLen3Gate::recalc(&ll, &of, GREEDY_MARGIN_EIGHTH_BITS);
        assert!(g.any_open);
        // Rare bytes (~11 bits each under this code) at a far slot
        // (slot 27: 12 extra bits): match wins.
        assert!(
            g.allows(16000, 7, 8, 9),
            "three rare bytes must lose to a far len-3"
        );
        // The dominant byte (~0.3 bits each): literals win, gate refuses.
        assert!(
            !g.allows(16000, 0, 0, 0),
            "three dominant cheap bytes must beat a far len-3"
        );
    }
}
