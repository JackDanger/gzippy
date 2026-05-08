//! Optimal LZ77 by dynamic programming + iterative re-statisticking.
//! Port of vendor/zopfli/src/zopfli/squeeze.c.
//!
//! Built across plan Steps 10 (`SymbolStats`, cost models, `RanState`),
//! 11 (`get_best_lengths`, `trace_backwards`, `follow_path`), and
//! 12 (`lz77_optimal`, `lz77_optimal_fixed`, end-to-end FFI oracle).

#![allow(dead_code)]

use super::lz77::LZ77Store;
use super::symbols::{
    dist_extra_bits, dist_symbol, length_extra_bits, length_symbol, ZOPFLI_NUM_D, ZOPFLI_NUM_LL,
};
use super::tree::calculate_entropy;

// ── SymbolStats ──────────────────────────────────────────────────────────────

/// Per-symbol frequencies + entropy bit-lengths used as the squeeze cost model.
/// Mirrors C `SymbolStats` field-for-field.
#[derive(Clone)]
pub struct SymbolStats {
    pub litlens: [usize; ZOPFLI_NUM_LL],
    pub dists: [usize; ZOPFLI_NUM_D],
    pub ll_symbols: [f64; ZOPFLI_NUM_LL],
    pub d_symbols: [f64; ZOPFLI_NUM_D],
}

impl Default for SymbolStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolStats {
    pub fn new() -> Self {
        Self {
            litlens: [0; ZOPFLI_NUM_LL],
            dists: [0; ZOPFLI_NUM_D],
            ll_symbols: [0.0; ZOPFLI_NUM_LL],
            d_symbols: [0.0; ZOPFLI_NUM_D],
        }
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.litlens = src.litlens;
        self.dists = src.dists;
        self.ll_symbols = src.ll_symbols;
        self.d_symbols = src.d_symbols;
    }

    pub fn clear_freqs(&mut self) {
        self.litlens = [0; ZOPFLI_NUM_LL];
        self.dists = [0; ZOPFLI_NUM_D];
    }

    /// In-place: `self = self*w1 + other*w2`. The C `AddWeighedStatFreqs` has
    /// a third `result` argument, but the only call site (`squeeze.c:509`)
    /// passes `result == stats1`. Also forces `litlens[256] = 1` (end symbol)
    /// to mirror the C tail.
    pub fn add_weighted(&mut self, w1: f64, other: &Self, w2: f64) {
        for i in 0..ZOPFLI_NUM_LL {
            self.litlens[i] = (self.litlens[i] as f64 * w1 + other.litlens[i] as f64 * w2) as usize;
        }
        for i in 0..ZOPFLI_NUM_D {
            self.dists[i] = (self.dists[i] as f64 * w1 + other.dists[i] as f64 * w2) as usize;
        }
        self.litlens[256] = 1;
    }

    /// Recomputes entropy bit-lengths from the current frequencies.
    pub fn calculate_statistics(&mut self) {
        calculate_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_entropy(&self.dists, &mut self.d_symbols);
    }

    /// Histograms a freshly-built `LZ77Store` into a new `SymbolStats`,
    /// forcing the end symbol and computing entropy bit-lengths.
    /// Equivalent to C `GetStatistics`.
    pub fn from_store(store: &LZ77Store<'_>) -> Self {
        let mut s = Self::new();
        for i in 0..store.size() {
            if store.dists[i] == 0 {
                s.litlens[store.litlens[i] as usize] += 1;
            } else {
                let ls = length_symbol(store.litlens[i] as i32) as usize;
                let ds = dist_symbol(store.dists[i] as i32) as usize;
                s.litlens[ls] += 1;
                s.dists[ds] += 1;
            }
        }
        s.litlens[256] = 1;
        s.calculate_statistics();
        s
    }
}

// ── RanState (Marsaglia MWC) ─────────────────────────────────────────────────

/// 32-bit Multiply-With-Carry PRNG with seeds `(m_w=1, m_z=2)`. Must produce
/// the exact sequence the C `RanState` does — full squeeze byte-equality
/// depends on it.
pub struct RanState {
    pub m_w: u32,
    pub m_z: u32,
}

impl Default for RanState {
    fn default() -> Self {
        Self::new()
    }
}

impl RanState {
    pub fn new() -> Self {
        Self { m_w: 1, m_z: 2 }
    }

    #[inline]
    pub fn next(&mut self) -> u32 {
        self.m_z = 36969u32
            .wrapping_mul(self.m_z & 0xffff)
            .wrapping_add(self.m_z >> 16);
        self.m_w = 18000u32
            .wrapping_mul(self.m_w & 0xffff)
            .wrapping_add(self.m_w >> 16);
        (self.m_z << 16).wrapping_add(self.m_w)
    }

    pub fn randomize_freqs(&mut self, freqs: &mut [usize]) {
        let n = freqs.len();
        for i in 0..n {
            if (self.next() >> 4).is_multiple_of(3) {
                freqs[i] = freqs[(self.next() as usize) % n];
            }
        }
    }

    /// Randomises both histograms then forces the end symbol back to 1.
    pub fn randomize_stat_freqs(&mut self, stats: &mut SymbolStats) {
        self.randomize_freqs(&mut stats.litlens);
        self.randomize_freqs(&mut stats.dists);
        stats.litlens[256] = 1;
    }
}

// ── Cost models ──────────────────────────────────────────────────────────────

/// Cost of one (litlen, dist) pair under the fixed Huffman tree (btype=01).
pub fn cost_fixed(litlen: u32, dist: u32) -> f64 {
    if dist == 0 {
        if litlen <= 143 {
            8.0
        } else {
            9.0
        }
    } else {
        let dbits = dist_extra_bits(dist as i32);
        let lbits = length_extra_bits(litlen as i32);
        let lsym = length_symbol(litlen as i32);
        let mut cost: i32 = 0;
        cost += if lsym <= 279 { 7 } else { 8 };
        cost += 5; // every dist symbol has length 5 in the fixed tree
        (cost + dbits + lbits) as f64
    }
}

/// Cost of one (litlen, dist) pair under the dynamic statistical model.
pub fn cost_stat(litlen: u32, dist: u32, stats: &SymbolStats) -> f64 {
    if dist == 0 {
        stats.ll_symbols[litlen as usize]
    } else {
        let lsym = length_symbol(litlen as i32) as usize;
        let lbits = length_extra_bits(litlen as i32) as f64;
        let dsym = dist_symbol(dist as i32) as usize;
        let dbits = dist_extra_bits(dist as i32) as f64;
        lbits + dbits + stats.ll_symbols[lsym] + stats.d_symbols[dsym]
    }
}

/// Trait the DP loop dispatches over. Per the plan's FAQ, generic
/// monomorphization gives the same machine code as C's indirect call —
/// usually better. `FixedCost` and `StatCost<'a>` are the only impls.
pub trait CostModel {
    fn cost(&self, litlen: u32, dist: u32) -> f64;
}

pub struct FixedCost;
impl CostModel for FixedCost {
    #[inline]
    fn cost(&self, litlen: u32, dist: u32) -> f64 {
        cost_fixed(litlen, dist)
    }
}

pub struct StatCost<'a>(pub &'a SymbolStats);
impl CostModel for StatCost<'_> {
    #[inline]
    fn cost(&self, litlen: u32, dist: u32) -> f64 {
        cost_stat(litlen, dist, self.0)
    }
}

/// Distances that introduce a new dist symbol per RFC 1951 §3.2.5. Probing
/// only these covers all 30 dist symbols without iterating all 32K distances.
const DSYMBOLS: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// The minimum cost any (length, distance) pair can produce under `model`.
/// Mirrors C `GetCostModelMinCost` exactly: scan lengths with dist=1, scan
/// dist symbols with length=3, then return `model(bestlength, bestdist)`.
pub fn cost_model_min_cost<C: CostModel>(model: &C) -> f64 {
    use super::symbols::ZOPFLI_LARGE_FLOAT;

    let mut bestlength: u32 = 0;
    let mut mincost = ZOPFLI_LARGE_FLOAT;
    for i in 3..259u32 {
        let c = model.cost(i, 1);
        if c < mincost {
            bestlength = i;
            mincost = c;
        }
    }

    let mut bestdist: u32 = 0;
    let mut mincost = ZOPFLI_LARGE_FLOAT;
    for &d in &DSYMBOLS {
        let c = model.cost(3, d);
        if c < mincost {
            bestdist = d;
            mincost = c;
        }
    }

    model.cost(bestlength, bestdist)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent 64-bit reference for the Marsaglia MWC. Computes each
    /// step in full-width integer math then narrows; if our wrapping-`u32`
    /// implementation disagrees, the test pinpoints which step diverged.
    /// This is genuinely independent (no `wrapping_*` ops, no shared code)
    /// even though it computes the same formula.
    fn mwc_reference(steps: usize) -> Vec<u32> {
        let mut m_w: u64 = 1;
        let mut m_z: u64 = 2;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            m_z = 36969 * (m_z & 0xffff) + (m_z >> 16);
            m_w = 18000 * (m_w & 0xffff) + (m_w >> 16);
            // 32-bit truncated combine.
            let v = (((m_z & 0xffff_ffff) << 16) + (m_w & 0xffff_ffff)) & 0xffff_ffff;
            out.push(v as u32);
        }
        out
    }

    #[test]
    fn ran_state_matches_64bit_reference() {
        let expected = mwc_reference(64);
        let mut r = RanState::new();
        for (i, &exp) in expected.iter().enumerate() {
            let got = r.next();
            assert_eq!(
                got, exp,
                "step {} mismatch: got 0x{:08x}, expected 0x{:08x}",
                i, got, exp
            );
        }
    }

    #[test]
    fn ran_state_progress() {
        // Smoke-test that successive `next` calls produce distinct values
        // (a stuck PRNG would silently break squeeze without this).
        let mut r = RanState::new();
        let a = r.next();
        let b = r.next();
        let c = r.next();
        assert!(a != b && b != c && a != c);
    }

    #[test]
    fn cost_fixed_literal_lo_lengths() {
        assert_eq!(cost_fixed(0, 0), 8.0);
        assert_eq!(cost_fixed(143, 0), 8.0);
        assert_eq!(cost_fixed(144, 0), 9.0);
        assert_eq!(cost_fixed(255, 0), 9.0);
    }

    #[test]
    fn cost_fixed_match() {
        // length=3, dist=1 → lsym=257 (≤279, +7), +5 dist + 0 dbits + 0 lbits
        assert_eq!(cost_fixed(3, 1), 12.0);
        // length=258, dist=32768 → lsym=285 (>279, +8), +5 dist + 13 dbits + 0 lbits
        assert_eq!(cost_fixed(258, 32_768), 8.0 + 5.0 + 13.0);
    }

    #[test]
    fn cost_model_min_cost_fixed_is_finite() {
        // sanity bound only — the full numeric oracle arrives in Step 12.
        let m = cost_model_min_cost(&FixedCost);
        assert!(m.is_finite());
        assert!(m > 0.0);
    }

    #[test]
    fn add_weighted_in_place_and_end_symbol() {
        let mut a = SymbolStats::new();
        let mut b = SymbolStats::new();
        a.litlens[10] = 100;
        a.dists[3] = 8;
        b.litlens[10] = 50;
        b.dists[3] = 4;
        a.add_weighted(0.5, &b, 0.5);
        // 100*0.5 + 50*0.5 = 75
        assert_eq!(a.litlens[10], 75);
        // 8*0.5 + 4*0.5 = 6
        assert_eq!(a.dists[3], 6);
        // End symbol must always be 1.
        assert_eq!(a.litlens[256], 1);
    }
}
