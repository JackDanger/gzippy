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

// ============================================================================
// THE #119 GRADED SHEET (probe machinery: ladder-tune builds only — the
// shipped binary must byte-compile identical to main, so none of this
// exists there).
//
// Upstream: google/zopfli PR #119 (fhanau's ECT) retuned `GetLengthScore`
// (src/zopfli/lz77.c) from a blanket distance penalty to a length-graded one:
//
//     static int GetLengthScore(int length, int distance) {
//       return (length == 3 && distance > 1024) || (length == 4 &&
//       distance > 2048) || (length == 5 && distance > 4096)
//         ? length - 1 : length;
//     }
//
// ≈ a curve where longer matches are allowed farther: len-3 keeps its
// penalty past 1024, len-4's starts at 2048, len-5's at 4096, len-6+ never.
// Adaptation: keep the per-block fixed-point sheet that won #364's far gate
// and grade the margin by length — near len-3 (the implicit-accept shadow
// the blanket min-match floor opened) is sheet-priced with the shipped
// margin above its 1024 grade and WITHOUT the margin below it, and the
// sheet also catches near len-4 / len-5 candidates past their grades (they
// are accepted with no guard at all otherwise).
// ============================================================================

/// Len-3 pays the margin past this distance (its #119 grade).
#[cfg(feature = "ladder-tune")]
const LEN3_GRADE_DIST: u32 = 1024;
/// Len-4's grade: sheet-checked with the margin only beyond this.
#[cfg(feature = "ladder-tune")]
const LEN4_GRADE_DIST: u32 = 2048;
/// Len-5's grade.
#[cfg(feature = "ladder-tune")]
const LEN5_GRADE_DIST: u32 = 4096;

/// The per-block cost sheet for the graded accept surface: margin-free base
/// costs (length symbol + distance symbol + exact extra bits) per offset slot
/// for lengths 3/4/5, plus the literal pricing; the margin is the test's own
/// grade. Probe-only; never compiled into a shipped binary.
#[cfg(feature = "ladder-tune")]
pub(super) struct LenGradeSheet {
    /// Ideal running cost (eighth-bits) per literal byte, recomputed per
    /// block on the same evidence floors as [`FarLen3Gate::recalc`].
    lit_cost: [u32; DEFLATE_NUM_LITERALS],
    /// `base[k][slot]`: margin-free match cost for length `3 + k` (CLOSED
    /// where the length or the slot carries no evidence).
    base: [[u32; DEFLATE_NUM_OFFSET_SYMS]; 3],
    any_open: [bool; 3],
}

#[cfg(feature = "ladder-tune")]
impl LenGradeSheet {
    /// The do-nothing sheet: every graded length closed.
    pub(super) const INERT: Self = Self {
        lit_cost: [0; DEFLATE_NUM_LITERALS],
        base: [[CLOSED; DEFLATE_NUM_OFFSET_SYMS]; 3],
        any_open: [false; 3],
    };

    /// Rebuild from the block's running frequencies at the parser's existing
    /// recalc cadence — the same global evidence floor as the shipped gate,
    /// plus a per-length floor (zero emissions of that length keep its table
    /// closed), plus the same per-slot 1/64-of-total floor.
    #[rustfmt::skip]
    pub(super) fn recalc(freqs: &DeflateFreqs) -> Self {
        let litlen = &freqs.litlen;
        let offset = &freqs.offset;
        let total_off: u32 = offset.iter().sum();
        if total_off < 1024 {
            return Self::INERT;
        }
        let len_freqs: [u32; 3] = [
            litlen[DEFLATE_FIRST_LEN_SYM as usize],
            litlen[DEFLATE_FIRST_LEN_SYM as usize + 1],
            litlen[DEFLATE_FIRST_LEN_SYM as usize + 2],
        ];
        let total_litlen: u64 = litlen.iter().map(|&f| f as u64).sum();
        debug_assert!(total_litlen <= u32::MAX as u64);
        let log_ll = log2_fp3(total_litlen as u32);
        let mut lit_cost = [0u32; DEFLATE_NUM_LITERALS];
        for (b, &f) in litlen[..DEFLATE_NUM_LITERALS].iter().enumerate() {
            lit_cost[b] = log_ll - log2_fp3(f.max(1));
        }
        let log_off = log2_fp3(total_off);
        let mut base = [[CLOSED; DEFLATE_NUM_OFFSET_SYMS]; 3];
        let mut any_open = [false; 3];
        for (s, &extra) in DEFLATE_EXTRA_OFFSET_BITS.iter().enumerate() {
            let fo = offset[s];
            if fo == 0 || (fo as u64) * 64 < total_off as u64 {
                continue;
            }
            let off_bits = (log_off - log2_fp3(fo)) + ((extra as u32) << 3);
            for (k, &f) in len_freqs.iter().enumerate() {
                if f != 0 {
                    base[k][s] = (log_ll - log2_fp3(f)) + off_bits;
                    any_open[k] = true;
                }
            }
        }
        Self {
            lit_cost,
            base,
            any_open,
        }
    }

    /// Length 3 is length-slot 0. The graded verdict: margin-free within
    /// #119's 1024, the shipped margin beyond it.
    #[inline(always)]
    pub(super) fn allows_len3(&self, offset: u32, b0: u8, b1: u8, b2: u8) -> bool {
        let s = deflate_get_offset_slot(offset) as usize;
        let margin = if offset > LEN3_GRADE_DIST {
            FAR_LEN3_MARGIN_EIGHTH_BITS
        } else {
            0
        };
        self.base[0][s] + margin
            <= self.lit_cost[b0 as usize] + self.lit_cost[b1 as usize] + self.lit_cost[b2 as usize]
    }

    /// Only called past the length's grade, where the margin always applies.
    #[inline(always)]
    pub(super) fn allows_len4(&self, offset: u32, b0: u8, b1: u8, b2: u8, b3: u8) -> bool {
        debug_assert!(offset > LEN4_GRADE_DIST);
        let s = deflate_get_offset_slot(offset) as usize;
        let cost = self.base[1][s] + FAR_LEN3_MARGIN_EIGHTH_BITS;
        cost <= self.lit_cost[b0 as usize]
            + self.lit_cost[b1 as usize]
            + self.lit_cost[b2 as usize]
            + self.lit_cost[b3 as usize]
    }

    #[inline(always)]
    pub(super) fn allows_len5(&self, offset: u32, b0: u8, b1: u8, b2: u8, b3: u8, b4: u8) -> bool {
        debug_assert!(offset > LEN5_GRADE_DIST);
        let s = deflate_get_offset_slot(offset) as usize;
        let cost = self.base[2][s] + FAR_LEN3_MARGIN_EIGHTH_BITS;
        cost <= self.lit_cost[b0 as usize]
            + self.lit_cost[b1 as usize]
            + self.lit_cost[b2 as usize]
            + self.lit_cost[b3 as usize]
            + self.lit_cost[b4 as usize]
    }

    /// Fail-open per length: a length with no observed emissions keeps its
    /// catches inactive so the surface degrades to exactly the shipped
    /// parser. (Len-2 never exists — DEFLATE_MIN_MATCH_LEN is 3 — so the
    /// grades cover every length that does.)
    #[inline(always)]
    pub(super) fn len3_active(&self) -> bool {
        self.any_open[0]
    }

    #[inline(always)]
    pub(super) fn len4_active(&self) -> bool {
        self.any_open[1]
    }

    #[inline(always)]
    pub(super) fn len5_active(&self) -> bool {
        self.any_open[2]
    }
}
