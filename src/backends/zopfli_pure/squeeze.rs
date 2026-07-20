//! Optimal LZ77 by dynamic programming + iterative re-statisticking.
//! Port of Google Zopfli squeeze.c.
//!
//! Built across plan Steps 10 (`SymbolStats`, cost models, `RanState`),
//! 11 (`get_best_lengths`, `trace_backwards`, `follow_path`), and
//! 12 (`lz77_optimal`, `lz77_optimal_fixed`, end-to-end FFI oracle).

use super::deflate_size::calculate_block_size;
use super::hash::ZopfliHash;
use super::lz77::{lz77_greedy, verify_len_dist, BlockState, LZ77Store};
use super::lzfind::{MatchFinder, MAX_MATCH_U16};
use super::symbols::{
    dist_extra_bits, dist_symbol, length_extra_bits, length_symbol, ZOPFLI_LARGE_FLOAT,
    ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH, ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_SIZE,
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

    /// Length-only component of a match cost (the length symbol code length
    /// plus its extra bits), without the distance component. Split out so the
    /// ECT-shaped DP can precompute `litlentable[3..=258]` and `disttable[..]`
    /// separately and sum them in the inner loop (see `get_best_lengths`).
    fn len_cost(&self, len: u32) -> f64;

    /// Distance-only component of a match cost (the distance symbol code length
    /// plus its extra bits).
    fn dist_cost(&self, dist: u32) -> f64;
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

    #[inline]
    fn len_cost(&self, len: u32) -> f64 {
        let lsym = length_symbol(len as i32);
        let lbits = length_extra_bits(len as i32);
        ((if lsym <= 279 { 7 } else { 8 }) + lbits) as f64
    }

    #[inline]
    fn dist_cost(&self, dist: u32) -> f64 {
        // Every distance symbol has code length 5 in the fixed tree.
        (5 + dist_extra_bits(dist as i32)) as f64
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

    #[inline]
    fn len_cost(&self, len: u32) -> f64 {
        self.0.len_cost[(len as usize) - ZOPFLI_MIN_MATCH]
    }

    #[inline]
    fn dist_cost(&self, dist: u32) -> f64 {
        let dsym = dist_symbol(dist as i32) as usize;
        dist_extra_bits(dist as i32) as f64 + self.0.d_symbols[dsym]
    }
}

// ── LzFind-driven optimal DP (ECT `GetBestLengths` shape) ────────────────────

/// Bit width by which a distance is shifted when packed alongside a length in a
/// `length_array` entry (`entry = len | (dist << 9)`; `len` occupies the low 9
/// bits, max 258 < 512).
const DIST_SHIFT: u32 = 9;
/// Mask isolating the length component of a packed `length_array` entry.
const LEN_MASK: u32 = (1 << DIST_SHIFT) - 1;

/// ECT's `LZCache`: on the first optimal-parse iteration the matchfinder's
/// emitted pairs are recorded here; every later iteration (which uses a new
/// cost model but sees the *same* matches) replays them instead of re-running
/// the binary-tree search. This is what makes 15–60 iterations affordable.
///
/// Layout: for every position that runs the matchfinder, one length-prefixed
/// record `[count, len, dist, len, dist, …]` where `count` is the number of
/// following `u16` values (`= 2 * numPairs`, `<= 512`). Positions skipped by
/// the `ML_RLE` fast path emit no record — the replay walks the identical fast
/// path and skips them too, so the read cursor stays aligned.
pub struct LzCache {
    data: Vec<u16>,
    read_pos: usize,
}

impl LzCache {
    fn new(blocksize: usize) -> Self {
        Self {
            data: Vec::with_capacity(blocksize + 513),
            read_pos: 0,
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.read_pos = 0;
    }

    fn reset_read(&mut self) {
        self.read_pos = 0;
    }

    /// Appends one record for the pairs in `pairs` (`pairs.len() == 2*numPairs`).
    fn store(&mut self, pairs: &[u16]) {
        self.data.push(pairs.len() as u16);
        self.data.extend_from_slice(pairs);
    }

    /// Copies the next record's pairs into `out`, returning the count of `u16`.
    fn read_into(&mut self, out: &mut [u16]) -> usize {
        let count = self.data[self.read_pos] as usize;
        self.read_pos += 1;
        out[..count].copy_from_slice(&self.data[self.read_pos..self.read_pos + count]);
        self.read_pos += count;
        count
    }
}

/// Source of match pairs for the DP: run the matchfinder and discard, run it
/// and record into the cache, or replay a previously recorded cache.
pub(crate) enum Mode<'c> {
    NoCache,
    Store(&'c mut LzCache),
    Replay(&'c mut LzCache),
}

/// Length of the run of `in_[i-1]` starting at `i` (distance-1 match), returned
/// as an absolute end index in `in_`. Used by the `ML_RLE` fast path. Mirrors
/// ECT's `GetMatch(&in[i], &in[i-1], …)`.
fn run_end_dist1(in_: &[u8], i: usize, inend: usize) -> usize {
    let mut s = i;
    let mut m = i - 1;
    let safe = inend.saturating_sub(8);
    while s < safe {
        let a = u64::from_le_bytes(in_[s..s + 8].try_into().unwrap());
        let b = u64::from_le_bytes(in_[m..m + 8].try_into().unwrap());
        let x = a ^ b;
        if x != 0 {
            return s + (x.trailing_zeros() >> 3) as usize;
        }
        s += 8;
        m += 8;
    }
    while s < inend && in_[s] == in_[m] {
        s += 1;
        m += 1;
    }
    s
}

/// Forward DP pass, ECT `GetBestLengths` shape. For every byte `j` in
/// `[0, blocksize]`, `length_array[j]` receives the optimal packed
/// `len | (dist << 9)` to reach byte `j` from a previous byte (`dist == 0` for
/// a literal, encoded as the bare value `1`); `costs[j]` is that path's cost.
/// Returns `costs[blocksize]`.
///
/// The DP is driven by the LzFind binary-tree matchfinder's match frontier
/// (or its cached replay), relaxing every `(len, dist)` pair with ECT's
/// `+6.0` promise-pruning heuristic and its `ML_RLE` long-run fast path.
pub(crate) fn get_best_lengths<C: CostModel>(
    in_: &[u8],
    instart: usize,
    inend: usize,
    model: &C,
    length_array: &mut [u32],
    costs: &mut [f32],
    mut mode: Mode<'_>,
) -> f64 {
    if instart == inend {
        return 0.0;
    }

    let blocksize = inend - instart;
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    // Precompute the cost tables hoisted out of the inner loop. Kept in `f64`
    // (with the DP relaxation summed in `f64`, `costs[]` stored as `f32`) so the
    // parse tie-breaking matches stock zopfli's precision exactly — the BT
    // frontier then yields the same parse as the hash chain where the chain
    // wasn't capped, and a better one where it was.
    let mut litlentable = [0f64; 259];
    for (k, slot) in litlentable
        .iter_mut()
        .enumerate()
        .take(ZOPFLI_MAX_MATCH + 1)
        .skip(ZOPFLI_MIN_MATCH)
    {
        *slot = model.len_cost(k as u32);
    }
    let mut disttable = vec![0f64; ZOPFLI_WINDOW_SIZE];
    for (d, slot) in disttable.iter_mut().enumerate().skip(1) {
        *slot = model.dist_cost(d as u32);
    }
    let mut literals = [0f64; 256];
    for (b, slot) in literals.iter_mut().enumerate() {
        *slot = model.literal_cost(b as u8);
    }
    // Cost of one distance-1 max-match, used by the ML_RLE fast path.
    let symbolcost = litlentable[ZOPFLI_MAX_MATCH] + disttable[1];

    let large = ZOPFLI_LARGE_FLOAT as f32;
    for c in &mut costs[1..=blocksize] {
        *c = large;
    }
    costs[0] = 0.0;

    // Lower bound on the cost of any single length/distance symbol, used for
    // the lossless DP prune below (stock-zopfli `GetCostModelMinCost`): the
    // match cost is separable into `len_cost + dist_cost`, so the global
    // minimum is the min of each table.
    let min_len = litlentable[ZOPFLI_MIN_MATCH..=ZOPFLI_MAX_MATCH]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let min_dist = disttable[1..].iter().copied().fold(f64::INFINITY, f64::min);
    let mincost = min_len + min_dist;

    // Matchfinder is only needed when we're actually searching (NoCache/Store).
    let mut mf = match mode {
        Mode::Replay(ref mut c) => {
            c.reset_read();
            None
        }
        Mode::Store(ref mut c) => {
            c.clear();
            Some(MatchFinder::new(in_, windowstart, instart, inend))
        }
        Mode::NoCache => Some(MatchFinder::new(in_, windowstart, instart, inend)),
    };

    let mut scratch = [0u16; MAX_MATCH_U16];
    let mut match_type_rle = false;

    let mut i = instart;
    while i < inend {
        let mut j = i - instart;

        // ML_RLE: inside a long distance-1 run, force a max-match at every
        // position and skip the matchfinder past the whole run. Exact shortcut
        // (a run's optimal parse is always max-matches at distance 1).
        if match_type_rle {
            let match_end = run_end_dist1(in_, i, inend);
            if match_end >= i + ZOPFLI_MAX_MATCH {
                let matchn = match_end - i - ZOPFLI_MAX_MATCH + 1;
                for _ in 0..matchn {
                    costs[j + ZOPFLI_MAX_MATCH] = (costs[j] as f64 + symbolcost) as f32;
                    length_array[j + ZOPFLI_MAX_MATCH] =
                        ZOPFLI_MAX_MATCH as u32 | (1u32 << DIST_SHIFT);
                    j += 1;
                }
                if let Some(mf) = mf.as_mut() {
                    mf.skip2(matchn);
                }
                i += matchn;
            }
            match_type_rle = false;
        }

        // Fetch this position's match frontier.
        let count = match &mut mode {
            Mode::NoCache => mf.as_mut().unwrap().get_matches(&mut scratch),
            Mode::Store(cache) => {
                let c = mf.as_mut().unwrap().get_matches(&mut scratch);
                cache.store(&scratch[..c]);
                c
            }
            Mode::Replay(cache) => cache.read_into(&mut scratch),
        };

        if count != 0 {
            let costj = costs[j] as f64;
            // A distance-1 max-length match means we're entering a run; arm the
            // ML_RLE fast path for the next position (in a run the nearest match
            // is dist 1 and already maxes out, so it is `scratch[0..2]`).
            if scratch[0] as usize == ZOPFLI_MAX_MATCH && scratch[1] == 1 {
                match_type_rle = true;
            }

            // Full relaxation over the binary-tree frontier. The strictly
            // increasing `(len, dist)` pairs encode `sublen[k]` exactly (for
            // `k` in `(prev_len, len]`, the closest distance is `dist`), so this
            // is stock-zopfli's optimal-parse relaxation — never worse than the
            // hash chain — but driven by the uncapped BT finder. The `mincost`
            // guard is a lossless skip (a slot already at the theoretical floor
            // can't be improved). Sum order matches stock zopfli exactly:
            // `(dist_cost + len_cost) + costs[j]`.
            let mincostaddcostj = mincost + costj;
            let mut curr = ZOPFLI_MIN_MATCH;
            let mut p = 0;
            while p < count {
                let len = scratch[p] as usize;
                let dist = scratch[p + 1] as usize;
                p += 2;
                let dcost = disttable[dist];
                while curr <= len {
                    if (costs[j + curr] as f64) > mincostaddcostj {
                        let x = dcost + litlentable[curr] + costj;
                        if x < costs[j + curr] as f64 {
                            costs[j + curr] = x as f32;
                            length_array[j + curr] = curr as u32 | ((dist as u32) << DIST_SHIFT);
                        }
                    }
                    curr += 1;
                }
            }
        }

        // Literal candidate. Sum order matches stock zopfli: `literal + costs[j]`.
        let new_cost = literals[in_[i] as usize] + costs[j] as f64;
        if new_cost < costs[j + 1] as f64 {
            costs[j + 1] = new_cost as f32;
            length_array[j + 1] = 1;
        }

        i += 1;
    }

    costs[blocksize] as f64
}

/// Walks `length_array` from the end back to 0, recording the picked packed
/// entries, then mirrors them into forward order in `path`.
pub(crate) fn trace_backwards(size: usize, length_array: &[u32], path: &mut Vec<u32>) {
    if size == 0 {
        return;
    }
    let mut index = size;
    loop {
        let e = length_array[index];
        path.push(e);
        let l = (e & LEN_MASK) as usize;
        debug_assert!(l <= index);
        debug_assert!(l <= ZOPFLI_MAX_MATCH);
        debug_assert!(l != 0);
        index -= l;
        if index == 0 {
            break;
        }
    }
    path.reverse();
}

/// Replays the packed `path` against the input, populating `store`. The
/// distance is read straight from each packed entry — no re-search — with
/// `verify_len_dist` kept as a debug safety check.
pub(crate) fn follow_path(in_: &[u8], instart: usize, path: &[u32], store: &mut LZ77Store<'_>) {
    let mut pos = instart;
    for &e in path {
        let length = (e & LEN_MASK) as u16;
        if length as usize >= ZOPFLI_MIN_MATCH {
            let dist = (e >> DIST_SHIFT) as u16;
            verify_len_dist(in_, pos, dist, length);
            store.store_lit_len_dist(length, dist, pos);
            pos += length as usize;
        } else {
            store.store_lit_len_dist(in_[pos] as u16, 0, pos);
            pos += 1;
        }
    }
}

/// One squeeze iteration: forward DP → trace → follow. Returns the DP cost.
/// The `path`, `length_array`, and `costs` scratch buffers are reused across
/// iterations by the caller; `mode` selects fresh-search / record / replay.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lz77_optimal_run<C: CostModel>(
    in_: &[u8],
    instart: usize,
    inend: usize,
    path: &mut Vec<u32>,
    length_array: &mut [u32],
    model: &C,
    store: &mut LZ77Store<'_>,
    costs: &mut [f32],
    mode: Mode<'_>,
) -> f64 {
    let cost = get_best_lengths(in_, instart, inend, model, length_array, costs, mode);
    path.clear();
    trace_backwards(inend - instart, length_array, path);
    follow_path(in_, instart, path, store);
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
    let mut length_array = vec![0u32; blocksize + 1];
    let mut costs = vec![0f32; blocksize + 1];
    let mut path: Vec<u32> = Vec::new();
    // First iteration records the matchfinder's pairs; later iterations replay
    // them (same matches, new cost model).
    let mut cache = LzCache::new(blocksize);

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
        let mode = if i == 0 {
            Mode::Store(&mut cache)
        } else {
            Mode::Replay(&mut cache)
        };
        lz77_optimal_run(
            in_,
            instart,
            inend,
            &mut path,
            &mut length_array,
            &model,
            &mut currentstore,
            &mut costs,
            mode,
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
    in_: &'a [u8],
    instart: usize,
    inend: usize,
    store: &mut LZ77Store<'a>,
) {
    let blocksize = inend - instart;
    let mut length_array = vec![0u32; blocksize + 1];
    let mut costs = vec![0f32; blocksize + 1];
    let mut path: Vec<u32> = Vec::new();

    lz77_optimal_run(
        in_,
        instart,
        inend,
        &mut path,
        &mut length_array,
        &FixedCost,
        store,
        &mut costs,
        Mode::NoCache,
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
    fn len_dist_cost_split_sums_to_cost() {
        // len_cost + dist_cost must equal the combined match cost for both
        // models (the DP relies on this decomposition).
        let f = FixedCost;
        for &(len, dist) in &[(3u32, 1u32), (10, 5), (258, 32_768), (128, 1234)] {
            assert!((f.len_cost(len) + f.dist_cost(dist) - f.cost(len, dist)).abs() < 1e-9);
        }
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
