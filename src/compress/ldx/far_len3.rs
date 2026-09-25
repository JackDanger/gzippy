//! The far-len-3 cost gate (port of the legacy parser's `far_len3` module,
//! `src/compress/deflate/parse/far_len3.rs`).
//!
//! The legacy L3 (zlib's deflate_slow, the campaign's winning T1 L3) accepts
//! a len-3 match at far offsets only when a per-block cost model says the
//! match beats the three literals it replaces; the libdeflate port's fixed
//! ">8192 offset" guard donates up to ~7% on high-entropy content (deterministic
//! 11-file corpus, 2026-09-01: tabular +18,886 B, text +6,831 B, binary
//! +4,316 B — the L3 size gap that keeps L3 on the legacy routing exception).
//!
//! Ported VERBATIM in behaviour: same evidence floors, same fixed-point log2,
//! same margin, same fail-closed INERT. The slot arithmetic is identical to
//! the legacy because both arms use the RFC 1951 30-slot offset alphabet
//! (`DEFLATE_EXTRA_OFFSET_BITS` here is the same table as the legacy's
//! `OFFSET_EXTRA_BITS`). Only the types move: `&DeflateFreqs` instead of two
//! raw slices.

use super::codes::DeflateFreqs;
use super::tables::{deflate_get_offset_slot, DEFLATE_EXTRA_OFFSET_BITS};
use super::{DEFLATE_FIRST_LEN_SYM, DEFLATE_NUM_LITERALS, DEFLATE_NUM_OFFSET_SYMS};

/// Margin (eighth-bit units) a far len-3 match must clear BELOW the estimated
/// cost of the three literals it replaces before the gate accepts it.
pub(super) const FAR_LEN3_MARGIN_EIGHTH_BITS: u32 = 16;

/// Deterministic fixed-point `log2(x)` in eighth-bit units (3 fractional
/// bits), integer-only. No libm: size is arch-invariant and must stay so.
/// Monotone non-decreasing in `x` (truncation of a monotone function), which
/// [`FarLen3Gate::recalc`] relies on for its subtractions to stay
/// non-negative. (Same function as the legacy's and as `compress_lazy`'s
/// `bsr32`-based one; local copy to keep the module self-contained.)
fn log2_fp3(x: u32) -> u32 {
    debug_assert!(x != 0);
    let int_part = 31 - x.leading_zeros(); // bsr32: floor(log2), same as the legacy's
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
    /// Full cost (eighth-bits) of a len-3 match per offset slot: len-3 litlen
    /// symbol + distance symbol + exact RFC 1951 distance extra bits, plus the
    /// accept margin. [`CLOSED`] where the slot has zero observed frequency.
    match_cost: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// Ideal running cost (eighth-bits) per literal byte value,
    /// `log2(total_litlen / freq)`; unseen bytes are priced as freq-1.
    lit_cost: [u32; DEFLATE_NUM_LITERALS],
    /// Frequency-weighted mean ideal literal cost (eighth-bits) for this block.
    mean_lit_eighth: u32,
    /// False = every slot closed; lets the parser skip the lookups.
    any_open: bool,
}

impl FarLen3Gate {
    /// The do-nothing gate: every slot closed, identical to the shipped fixed
    /// offset guard.
    pub(super) const INERT: Self = Self {
        match_cost: [CLOSED; DEFLATE_NUM_OFFSET_SYMS],
        lit_cost: [0; DEFLATE_NUM_LITERALS],
        mean_lit_eighth: 0,
        any_open: false,
    };

    /// Rebuild from the block's running frequencies. `margin_eighth_bits` is
    /// the caller's accept margin. (The legacy's greedy-only
    /// `accept_slack_eighth` is dropped here: the lazy parser passes 0.)
    pub(super) fn recalc(freqs: &DeflateFreqs, margin_eighth_bits: u32) -> Self {
        let litlen = &freqs.litlen;
        let offset = &freqs.offset;
        // Length 3 is length-slot 0 => symbol DEFLATE_FIRST_LEN_SYM (257).
        let f_len3 = litlen[DEFLATE_FIRST_LEN_SYM as usize];
        let total_off: u32 = offset.iter().sum();
        // Absolute evidence floor: on near-incompressible content the block
        // has only sparse (often chance) matches, and ideal costs computed
        // from a small sample are noise. Below 1024 observed offsets the
        // gate stays inert.
        if f_len3 == 0 || total_off < 1024 {
            return Self::INERT;
        }
        let total_litlen: u64 = litlen.iter().map(|&f| f as u64).sum();
        debug_assert!(total_litlen <= u32::MAX as u64);
        let log_ll = log2_fp3(total_litlen as u32);
        let mut lit_cost = [0u32; DEFLATE_NUM_LITERALS];
        for (b, &f) in litlen[..DEFLATE_NUM_LITERALS].iter().enumerate() {
            lit_cost[b] = log_ll - log2_fp3(f.max(1));
        }
        let mut lit_weighted = 0u64;
        let mut lit_count = 0u64;
        for (b, &f) in litlen[..DEFLATE_NUM_LITERALS].iter().enumerate() {
            if f > 0 {
                lit_weighted += lit_cost[b] as u64 * f as u64;
                lit_count += f as u64;
            }
        }
        let mean_lit_eighth = (lit_weighted / lit_count.max(1)) as u32;
        let len3_sym_bits = log_ll - log2_fp3(f_len3);
        let log_off = log2_fp3(total_off);
        let mut match_cost = [CLOSED; DEFLATE_NUM_OFFSET_SYMS];
        let mut any_open = false;
        for (s, &extra) in DEFLATE_EXTRA_OFFSET_BITS.iter().enumerate() {
            let fo = offset[s];
            // Evidence floor: a slot must hold at least 1/64 of the block's
            // offsets before its running cost is trusted.
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
            mean_lit_eighth,
            any_open,
        }
    }

    #[inline(always)]
    fn trigram_lit_cost(&self, b0: u8, b1: u8, b2: u8) -> u32 {
        self.lit_cost[b0 as usize] + self.lit_cost[b1 as usize] + self.lit_cost[b2 as usize]
    }

    /// Per-position accept test: does a len-3 match at `offset` beat the
    /// THREE ACTUAL BYTES `b0 b1 b2` it would replace?
    ///
    /// The legacy's `accept_slack_eighth` (a borderline-accept slack) is
    /// GREEDY-ONLY: the lazy parser passes 0, which makes the slack arm
    /// `false && ...` — so the legacy lazy's accept is exactly `mc <= lits`
    /// (a CLOSED slot is `u32::MAX/2`, above any literal sum). Ported 1:1.
    #[inline(always)]
    pub(super) fn allows(&self, offset: u32, b0: u8, b1: u8, b2: u8) -> bool {
        if !self.any_open {
            return false;
        }
        let lits = self.trigram_lit_cost(b0, b1, b2);
        self.match_cost[deflate_get_offset_slot(offset) as usize] <= lits
    }

    #[inline(always)]
    pub(super) fn inert(&self) -> bool {
        !self.any_open
    }
}
