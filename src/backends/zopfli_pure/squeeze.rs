//! Optimal LZ77 by dynamic programming + iterative re-statisticking.
//! Port of Google Zopfli squeeze.c.
//!
//! Built across plan Steps 10 (`SymbolStats`, cost models, `RanState`),
//! 11 (`get_best_lengths`, `trace_backwards`, `follow_path`), and
//! 12 (`lz77_optimal`, `lz77_optimal_fixed`, end-to-end FFI oracle).

use super::deflate_size::calculate_block_size;
use super::hash::ZopfliHash;
use super::lz77::{find_longest_match, lz77_greedy, verify_len_dist, BlockState, LZ77Store};
use super::symbols::{
    dist_extra_bits, dist_symbol, length_extra_bits, length_symbol, ZOPFLI_LARGE_FLOAT,
    ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH, ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_MASK,
    ZOPFLI_WINDOW_SIZE,
};
use super::tree::calculate_entropy;

// ── SymbolStats ──────────────────────────────────────────────────────────────

/// Length of `len_cost`: one entry per match length 3..=258.
const LEN_COST_LEN: usize = ZOPFLI_MAX_MATCH - ZOPFLI_MIN_MATCH + 1;

/// Per-symbol frequencies + entropy bit-lengths used as the squeeze cost model.
/// Mirrors C `SymbolStats` field-for-field, plus a precomputed `len_cost`
/// table hoisted out of the squeeze DP inner loop (plan.md Phase 12.1).
#[derive(Clone)]
pub struct SymbolStats {
    pub litlens: [usize; ZOPFLI_NUM_LL],
    pub dists: [usize; ZOPFLI_NUM_D],
    pub ll_symbols: [f64; ZOPFLI_NUM_LL],
    pub d_symbols: [f64; ZOPFLI_NUM_D],
    /// `len_cost[k - ZOPFLI_MIN_MATCH] = length_extra_bits(k) +
    /// ll_symbols[length_symbol(k)]` for `k` in 3..=258. **Lockstep
    /// invariant:** any code that mutates `ll_symbols` must also call
    /// `rebuild_len_cost`. The audit (plan.md §12.1) is binding: the
    /// only writers are `calculate_statistics` and `copy_from`, and
    /// both are updated below. `add_weighted` writes only `litlens` /
    /// `dists` and is always followed by `calculate_statistics` at
    /// the call site (squeeze.rs:533).
    pub len_cost: [f64; LEN_COST_LEN],
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
            len_cost: [0.0; LEN_COST_LEN],
        }
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.litlens = src.litlens;
        self.dists = src.dists;
        self.ll_symbols = src.ll_symbols;
        self.d_symbols = src.d_symbols;
        // Lockstep with ll_symbols.
        self.len_cost = src.len_cost;
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

    /// Recomputes entropy bit-lengths from the current frequencies and
    /// rebuilds `len_cost` from the fresh `ll_symbols`.
    pub fn calculate_statistics(&mut self) {
        calculate_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_entropy(&self.dists, &mut self.d_symbols);
        self.rebuild_len_cost();
    }

    /// Refreshes `len_cost` from the current `ll_symbols`. Lockstep
    /// helper for the invariant on the field.
    fn rebuild_len_cost(&mut self) {
        for k in ZOPFLI_MIN_MATCH..=ZOPFLI_MAX_MATCH {
            let lsym = length_symbol(k as i32) as usize;
            let lbits = length_extra_bits(k as i32) as f64;
            self.len_cost[k - ZOPFLI_MIN_MATCH] = lbits + self.ll_symbols[lsym];
        }
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
/// Match path uses `stats.len_cost` (precomputed `lbits + ll_symbols[lsym]`,
/// plan.md Phase 12.1) so the inner DP loop drops two table lookups per
/// iteration. Sum order changes vs C — the corpus oracle (Phase 11.2) is
/// the contract that this reordering doesn't shift any decision by 1 ULP.
pub fn cost_stat(litlen: u32, dist: u32, stats: &SymbolStats) -> f64 {
    if dist == 0 {
        stats.ll_symbols[litlen as usize]
    } else {
        let dsym = dist_symbol(dist as i32) as usize;
        let dbits = dist_extra_bits(dist as i32) as f64;
        dbits + stats.d_symbols[dsym] + stats.len_cost[(litlen - ZOPFLI_MIN_MATCH as u32) as usize]
    }
}

/// Trait the DP loop dispatches over. Per the plan's FAQ, generic
/// monomorphization gives the same machine code as C's indirect call —
/// usually better. `FixedCost` and `StatCost<'a>` are the only impls.
///
/// `literal_cost` is a Phase 12.2 specialization: the literal-candidate
/// site in the DP loop hits this on every byte and only ever passes
/// `dist=0`, so a dedicated method lets the optimizer skip the
/// `if dist == 0` branch entirely. Default impl forwards to `cost`
/// for any future model that doesn't bother specializing.
pub trait CostModel {
    fn cost(&self, litlen: u32, dist: u32) -> f64;

    #[inline]
    fn literal_cost(&self, byte: u8) -> f64 {
        self.cost(byte as u32, 0)
    }
}

pub struct FixedCost;
impl CostModel for FixedCost {
    #[inline]
    fn cost(&self, litlen: u32, dist: u32) -> f64 {
        cost_fixed(litlen, dist)
    }

    #[inline]
    fn literal_cost(&self, byte: u8) -> f64 {
        if byte <= 143 {
            8.0
        } else {
            9.0
        }
    }
}

pub struct StatCost<'a>(pub &'a SymbolStats);
impl CostModel for StatCost<'_> {
    #[inline]
    fn cost(&self, litlen: u32, dist: u32) -> f64 {
        cost_stat(litlen, dist, self.0)
    }

    #[inline]
    fn literal_cost(&self, byte: u8) -> f64 {
        self.0.ll_symbols[byte as usize]
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

// ── Step 11: get_best_lengths / trace_backwards / follow_path / optimal_run ──

/// Forward DP pass. For every byte `j` in `[0, blocksize]`, `length_array[j]`
/// is filled with the optimal length to reach byte `j` from a previous byte;
/// `costs[j]` is the cost of that path under `model`. Returns the cost of the
/// full path (`costs[blocksize]`).
///
/// **Floating-point widths are load-bearing.** `costs[]` is `f32`; cost-model
/// returns are `f64`; comparisons promote the f32 to f64 with explicit `as
/// f64` and stores narrow with explicit `as f32`. Pinned by the plan FAQ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_best_lengths<C: CostModel>(
    s: &mut BlockState<'_>,
    in_: &[u8],
    instart: usize,
    inend: usize,
    model: &C,
    length_array: &mut [u16],
    h: &mut ZopfliHash,
    costs: &mut [f32],
) -> f64 {
    if instart == inend {
        return 0.0;
    }

    let blocksize = inend - instart;
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    let mincost: f64 = cost_model_min_cost(model);

    h.reset(ZOPFLI_WINDOW_SIZE);
    h.warmup(in_, windowstart, inend);
    for i in windowstart..instart {
        h.update(in_, i, inend);
    }

    for c in &mut costs[1..=blocksize] {
        *c = ZOPFLI_LARGE_FLOAT as f32;
    }
    costs[0] = 0.0;
    length_array[0] = 0;

    let mut sublen = [0u16; 259];
    let mut i = instart;
    while i < inend {
        let mut j = i - instart;
        h.update(in_, i, inend);

        // ZOPFLI_SHORTCUT_LONG_REPETITIONS: skip ZOPFLI_MAX_MATCH bytes
        // ahead when we're inside a long uniform run, both before and after
        // the current position.
        if h.same[i & ZOPFLI_WINDOW_MASK] as usize > ZOPFLI_MAX_MATCH * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && h.same[(i - ZOPFLI_MAX_MATCH) & ZOPFLI_WINDOW_MASK] as usize > ZOPFLI_MAX_MATCH
        {
            let symbolcost: f64 = model.cost(ZOPFLI_MAX_MATCH as u32, 1);
            for _ in 0..ZOPFLI_MAX_MATCH {
                costs[j + ZOPFLI_MAX_MATCH] = (costs[j] as f64 + symbolcost) as f32;
                length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH as u16;
                i += 1;
                j += 1;
                h.update(in_, i, inend);
            }
        }

        let mut leng: u16 = 0;
        let mut dist: u16 = 0;
        find_longest_match(
            s,
            h,
            in_,
            i,
            inend,
            ZOPFLI_MAX_MATCH,
            Some(&mut sublen),
            &mut dist,
            &mut leng,
        );

        // Literal candidate.
        if i < inend {
            let new_cost: f64 = model.literal_cost(in_[i]) + costs[j] as f64;
            debug_assert!(new_cost >= 0.0);
            if new_cost < costs[j + 1] as f64 {
                costs[j + 1] = new_cost as f32;
                length_array[j + 1] = 1;
            }
        }

        // Length candidates.
        let kend = (leng as usize).min(inend - i);
        let mincostaddcostj: f64 = mincost + costs[j] as f64;
        for k in ZOPFLI_MIN_MATCH..=kend {
            if (costs[j + k] as f64) <= mincostaddcostj {
                continue;
            }
            let new_cost: f64 = model.cost(k as u32, sublen[k] as u32) + costs[j] as f64;
            debug_assert!(new_cost >= 0.0);
            if new_cost < costs[j + k] as f64 {
                debug_assert!(k <= ZOPFLI_MAX_MATCH);
                costs[j + k] = new_cost as f32;
                length_array[j + k] = k as u16;
            }
        }

        i += 1;
    }

    debug_assert!(costs[blocksize] >= 0.0);
    costs[blocksize] as f64
}

/// Walks `length_array` from the end back to 0, recording the picked lengths,
/// then mirrors them in place. Output buffer is `path`; the C version
/// allocates and returns size, we just push.
pub(crate) fn trace_backwards(size: usize, length_array: &[u16], path: &mut Vec<u16>) {
    if size == 0 {
        return;
    }
    let mut index = size;
    loop {
        let l = length_array[index];
        path.push(l);
        debug_assert!((l as usize) <= index);
        debug_assert!((l as usize) <= ZOPFLI_MAX_MATCH);
        debug_assert!(l != 0);
        index -= l as usize;
        if index == 0 {
            break;
        }
    }
    path.reverse();
}

/// Replays `path` against the input, populating `store` with literals and
/// length/dist pairs; recovers each distance via a fresh `find_longest_match`
/// call (cheap because the LMC is already populated from the forward pass).
pub(crate) fn follow_path(
    s: &mut BlockState<'_>,
    in_: &[u8],
    instart: usize,
    inend: usize,
    path: &[u16],
    store: &mut LZ77Store<'_>,
    h: &mut ZopfliHash,
) {
    if instart == inend {
        return;
    }

    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    h.reset(ZOPFLI_WINDOW_SIZE);
    h.warmup(in_, windowstart, inend);
    for i in windowstart..instart {
        h.update(in_, i, inend);
    }

    let mut pos = instart;
    for &raw_length in path {
        debug_assert!(pos < inend);

        h.update(in_, pos, inend);

        let length: u16 = if (raw_length as usize) >= ZOPFLI_MIN_MATCH {
            // Recover the distance with the same window search; the assert
            // matches the C code's sanity check that the longest match
            // returned now is the same length we picked in the forward pass
            // (or both fall under the 2-byte threshold).
            let mut dist: u16 = 0;
            let mut dummy_length: u16 = 0;
            find_longest_match(
                s,
                h,
                in_,
                pos,
                inend,
                raw_length as usize,
                None,
                &mut dist,
                &mut dummy_length,
            );
            debug_assert!(!(dummy_length != raw_length && raw_length > 2 && dummy_length > 2));
            verify_len_dist(in_, pos, dist, raw_length);
            store.store_lit_len_dist(raw_length, dist, pos);
            raw_length
        } else {
            store.store_lit_len_dist(in_[pos] as u16, 0, pos);
            1
        };

        debug_assert!(pos + length as usize <= inend);
        for j in 1..length as usize {
            h.update(in_, pos + j, inend);
        }
        pos += length as usize;
    }
}

/// One squeeze iteration: forward DP → trace → follow. Returns the DP cost.
/// The `path`, `length_array`, and `costs` scratch buffers are reused
/// across iterations by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lz77_optimal_run<C: CostModel>(
    s: &mut BlockState<'_>,
    in_: &[u8],
    instart: usize,
    inend: usize,
    path: &mut Vec<u16>,
    length_array: &mut [u16],
    model: &C,
    store: &mut LZ77Store<'_>,
    h: &mut ZopfliHash,
    costs: &mut [f32],
) -> f64 {
    let cost = get_best_lengths(s, in_, instart, inend, model, length_array, h, costs);
    path.clear();
    trace_backwards(inend - instart, length_array, path);
    follow_path(s, in_, instart, inend, path, store, h);
    debug_assert!(cost < ZOPFLI_LARGE_FLOAT);
    cost
}

// ── Step 12: lz77_optimal / lz77_optimal_fixed ───────────────────────────────

/// Multi-iteration optimal squeeze. Each iteration uses the previous run's
/// histogram as the cost model; the cheapest run wins. Mirrors C
/// `ZopfliLZ77Optimal` literally — including the magic numbers `i > 5`,
/// the 1.0/0.5 weighted blend after the randomization kicks in, and the
/// `lastrandomstep == -1` activation gate.
pub fn lz77_optimal<'a>(
    s: &mut BlockState<'_>,
    in_: &'a [u8],
    instart: usize,
    inend: usize,
    numiterations: i32,
    store: &mut LZ77Store<'a>,
) {
    let blocksize = inend - instart;
    let mut length_array = vec![0u16; blocksize + 1];
    let mut costs = vec![0f32; blocksize + 1];
    let mut path: Vec<u16> = Vec::new();

    let mut currentstore = LZ77Store::new(in_);
    let mut h = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);

    let mut beststats = SymbolStats::new();
    let mut laststats = SymbolStats::new();

    let mut bestcost = ZOPFLI_LARGE_FLOAT;
    let mut lastcost: f64 = 0.0;
    let mut ran_state = RanState::new();
    let mut lastrandomstep: i32 = -1;

    // Initial run: greedy parse → seed statistics.
    lz77_greedy(s, in_, instart, inend, &mut currentstore, &mut h);
    let mut stats = SymbolStats::from_store(&currentstore);

    for i in 0..numiterations {
        currentstore.reset();
        let model = StatCost(&stats);
        lz77_optimal_run(
            s,
            in_,
            instart,
            inend,
            &mut path,
            &mut length_array,
            &model,
            &mut currentstore,
            &mut h,
            &mut costs,
        );
        let cost = calculate_block_size(&currentstore, 0, currentstore.size(), 2);
        if s.options.verbose_more != 0 || (s.options.verbose != 0 && cost < bestcost) {
            eprintln!("Iteration {}: {} bit", i, cost as i64);
        }
        if cost < bestcost {
            // Copy the run's output into the caller's store.
            store.reset();
            store.append_from(&currentstore);
            beststats.copy_from(&stats);
            bestcost = cost;
        }
        laststats.copy_from(&stats);
        stats.clear_freqs();
        stats = SymbolStats::from_store(&currentstore);
        if lastrandomstep != -1 {
            // Once randomness has kicked in, blend last + current to slow
            // convergence but improve final ratio. C uses an in-place
            // (`&stats, &laststats, &stats`) call; our `add_weighted` is
            // already in-place per the FAQ.
            stats.add_weighted(1.0, &laststats, 0.5);
            stats.calculate_statistics();
        }
        if i > 5 && cost == lastcost {
            stats.copy_from(&beststats);
            ran_state.randomize_stat_freqs(&mut stats);
            stats.calculate_statistics();
            lastrandomstep = i;
        }
        lastcost = cost;
    }
}

/// Single-shot squeeze under the fixed Huffman tree (btype=01). Mirrors
/// `ZopfliLZ77OptimalFixed`. No iteration; the fixed tree is known so
/// `FixedCost` is the optimal cost model.
pub fn lz77_optimal_fixed<'a>(
    s: &mut BlockState<'_>,
    in_: &'a [u8],
    instart: usize,
    inend: usize,
    store: &mut LZ77Store<'a>,
) {
    let blocksize = inend - instart;
    let mut length_array = vec![0u16; blocksize + 1];
    let mut costs = vec![0f32; blocksize + 1];
    let mut path: Vec<u16> = Vec::new();
    let mut h = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);

    s.blockstart = instart;
    s.blockend = inend;

    lz77_optimal_run(
        s,
        in_,
        instart,
        inend,
        &mut path,
        &mut length_array,
        &FixedCost,
        store,
        &mut h,
        &mut costs,
    );
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
