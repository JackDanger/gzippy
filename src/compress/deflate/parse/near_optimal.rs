//! Near-optimal parser (levels 10-12).
//!
//! Faithful transliteration of libdeflate's near-optimal compressor
//! (`vendor/libdeflate/lib/deflate_compress.c`):
//!   * `deflate_compress_near_optimal` (`:3592-3849`) — the driver: cache all
//!     matches for a block with the bt matchfinder, gather split/length stats,
//!     skip the interior of long matches, trigger block ends (max length / cache
//!     overflow / split heuristic), and rewind-on-split.
//!   * `deflate_find_min_cost_path` (`:3327-3399`) — the backward min-cost-path
//!     DP over the cached matches, using the smallest offset available for each
//!     length (the vendor heuristic that relies on the match list being sorted
//!     by strictly-increasing length and non-decreasing offset).
//!   * `deflate_optimize_and_flush_block` (`:3416-3530`) — iterative refinement:
//!     seed costs, repeatedly find the path and re-cost from the resulting
//!     Huffman codes until the true (whole-bit) cost stops improving, and choose
//!     the cheapest of the optimized dynamic path, an only-literals block, and
//!     (for small blocks) a static-Huffman-optimized path.
//!
//! Bridge to the shared substrate: rather than re-porting `deflate_flush_block`,
//! the chosen min-cost path is walked into the substrate's [`super::Sink`] and
//! handed to [`super::emit_block`], which independently emits the cheapest of a
//! stored / static / dynamic coding of that exact token stream — the same
//! final decision `deflate_flush_block` makes. The near-optimal work (which
//! *path* to code) is the DP's; the coding choice is emit_block's.

use super::super::bitstream::BitWriter;
use super::super::block_split::{BlockSplitStats, NUM_OBSERVATION_TYPES};
use super::super::costs::{
    set_initial_costs, DeflateCosts, OffsetSlotFull, OptimumNode, BIT_COST, OPTIMUM_LEN_MASK,
    OPTIMUM_OFFSET_SHIFT,
};
use super::super::huffman::{build_dynamic_header, make_huffman_code, HuffmanCode};
use super::super::level::LevelParams;
use super::super::matchfinder::bt::{
    BtMatchfinder, LzMatch, BT_MATCHFINDER_REQUIRED_NBYTES, WINDOW_SIZE,
};
use super::super::tables::{
    length_slot, DEFLATE_END_OF_BLOCK, DEFLATE_FIRST_LEN_SYM, DEFLATE_MAX_MATCH_LEN,
    DEFLATE_MIN_MATCH_LEN, DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS, LENGTH_EXTRA_BITS,
    MAX_LITLEN_CODEWORD_LEN, MAX_OFFSET_CODEWORD_LEN, OFFSET_EXTRA_BITS,
};
use super::{
    adjust_max_and_nice_len, calculate_min_match_len, choose_max_block_end, emit_block, Sink,
    StaticCodes,
};

const MIN_MATCH_LEN: u32 = DEFLATE_MIN_MATCH_LEN;
const MAX_MATCH_LEN: u32 = DEFLATE_MAX_MATCH_LEN;
const NUM_LEN_SLOTS: usize = 29;

/// `MAX_MATCHES_PER_POS` (`:177-178`).
const MAX_MATCHES_PER_POS: usize = (MAX_MATCH_LEN - MIN_MATCH_LEN + 1) as usize;
/// `MATCH_CACHE_LENGTH = SOFT_MAX_BLOCK_LENGTH * 5` (`:158`).
const MATCH_CACHE_LENGTH: usize = 300_000 * 5;
/// Total cached-match slots incl. worst-case overflow slop (`:571-573`).
const MATCH_CACHE_TOTAL: usize =
    MATCH_CACHE_LENGTH + MAX_MATCHES_PER_POS + (MAX_MATCH_LEN as usize - 1);
/// `MAX_BLOCK_LENGTH` (`:188-190`).
const MAX_BLOCK_LENGTH: usize = {
    let a = 300_000 + 5_000 - 1;
    let b = 300_000 + 1 + MAX_MATCH_LEN as usize;
    if a > b {
        a
    } else {
        b
    }
};
/// `optimum_nodes` length (`:583-584`).
const OPTIMUM_NODES_LEN: usize = MAX_BLOCK_LENGTH + 1;

/// The min-cost-path result for one pass: the chosen path's symbol frequencies
/// and the Huffman codes built from them.
struct PathCodes {
    litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
    litcode: HuffmanCode,
    offcode: HuffmanCode,
}

/// Everything the DP / flush phase needs (the bt matchfinder lives separately in
/// [`run`] so it can write into `match_cache` without aliasing).
struct Optimizer {
    match_cache: Vec<LzMatch>,
    optimum_nodes: Vec<OptimumNode>,
    costs: DeflateCosts,
    costs_saved: DeflateCosts,
    offset_slot_full: OffsetSlotFull,
    /// Merged approximate greedy match-length histogram (`match_len_freqs`).
    match_len_freqs: Vec<u32>,
}

impl Optimizer {
    fn new() -> Self {
        Optimizer {
            match_cache: vec![LzMatch::default(); MATCH_CACHE_TOTAL],
            optimum_nodes: vec![OptimumNode::default(); OPTIMUM_NODES_LEN],
            costs: DeflateCosts::default(),
            costs_saved: DeflateCosts::default(),
            offset_slot_full: OffsetSlotFull::new(),
            match_len_freqs: vec![0u32; MAX_MATCH_LEN as usize + 1],
        }
    }

    /// `deflate_find_min_cost_path` (`:3327-3399`): backward DP filling
    /// `optimum_nodes[0..=block_length]`, then tally the resulting path into
    /// symbol frequencies and build the Huffman codes.
    ///
    /// `cache_end` is the index one past the block's last position header in
    /// `match_cache` (the C `cache_ptr`). The nodes at `block_length+1 ..` must
    /// already be pinned to `0x8000_0000` by the caller.
    ///
    /// ## Soundness invariant (unchecked-index hot loop)
    ///
    /// The body below drops Rust's bounds checks to match libdeflate's C
    /// codegen (measured: ~21-23% of this function's excess instruction count
    /// vs `deflate_find_min_cost_path` at L10-12 is inlined
    /// `core::slice::index` panic-path overhead). Every elided check is
    /// discharged by construction, same as the `hc.rs` matchfinder hot loop:
    ///  * **`cptr` walks `match_cache` backward.** It starts at `cache_end
    ///    <= match_cache.len()` and only decreases (by 1 per node, and by
    ///    `num_matches` when rewound to `first`), always landing on a header
    ///    or match slot the FORWARD pass already wrote contiguously below
    ///    `cache_end` — so `cptr`/`mi` stay in `0..match_cache.len()`.
    ///  * **`node` walks `optimum_nodes` backward**, from `block_length` down
    ///    to `0`. `node+1` and `node+len` (`len <= m.length <= MAX_MATCH_LEN`)
    ///    stay `<= hi < optimum_nodes.len()` because the caller
    ///    (`optimize_and_flush`) pins `optimum_nodes[block_length..=hi]`
    ///    (`hi = min(block_length-1+MAX_MATCH_LEN, optimum_nodes.len()-1)`)
    ///    before calling — the exact invariant the checked code already
    ///    relied on to read sentinel costs past the block end.
    ///  * **`literal`** is a cache-header byte (`buf[pos] as u16` widened to
    ///    `u32`), so always `< 256 == costs.literal.len()`.
    ///  * **`len`** ranges `MIN_MATCH_LEN..=m.length` with `m.length <=
    ///    MAX_MATCH_LEN`, so always `< costs.length.len() ==
    ///    MAX_MATCH_LEN+1`.
    ///  * **`offset_slot`** comes from `OffsetSlotFull::slot_unchecked`,
    ///    itself bound to `offset in 1..=MAX_MATCH_OFFSET`, and the map only
    ///    ever emits the 30 real slots, so `< 32 == costs.offset_slot.len()`.
    fn find_min_cost_path(&mut self, block_length: usize, cache_end: usize) -> PathCodes {
        let mut node = block_length;
        self.optimum_nodes[node].cost_to_end = 0;
        let mut cptr = cache_end;

        loop {
            node -= 1;
            cptr -= 1;
            // SAFETY: see the soundness invariant above (`cptr` bound).
            debug_assert!(cptr < self.match_cache.len());
            let header = unsafe { *self.match_cache.get_unchecked(cptr) };
            let num_matches = header.length as usize;
            let literal = header.offset as u32;

            // A literal is always available.
            // SAFETY: `literal < 256`; `node + 1 <= hi < optimum_nodes.len()`
            // (invariant above).
            debug_assert!((literal as usize) < self.costs.literal.len());
            debug_assert!(node + 1 < self.optimum_nodes.len());
            let mut best_cost = unsafe {
                self.costs
                    .literal
                    .get_unchecked(literal as usize)
                    .wrapping_add(self.optimum_nodes.get_unchecked(node + 1).cost_to_end)
            };
            let mut best_item = (literal << OPTIMUM_OFFSET_SHIFT) | 1;

            if num_matches != 0 {
                let first = cptr - num_matches;
                let mut mi = first;
                let mut len = MIN_MATCH_LEN as usize;
                loop {
                    // SAFETY: `mi < cptr <= match_cache.len()` (invariant above).
                    debug_assert!(mi < self.match_cache.len());
                    let m = unsafe { *self.match_cache.get_unchecked(mi) };
                    let offset = m.offset as u32;
                    // SAFETY: `offset` is a valid DEFLATE match offset
                    // (`1..=MAX_MATCH_OFFSET`) — see `slot_unchecked`'s own
                    // debug_assert.
                    let offset_slot = unsafe { self.offset_slot_full.slot_unchecked(offset) };
                    debug_assert!(offset_slot < self.costs.offset_slot.len());
                    let offset_cost = unsafe { *self.costs.offset_slot.get_unchecked(offset_slot) };
                    loop {
                        // SAFETY: `len <= m.length <= MAX_MATCH_LEN` and
                        // `node + len <= hi < optimum_nodes.len()` (invariant
                        // above).
                        debug_assert!(len < self.costs.length.len());
                        debug_assert!(node + len < self.optimum_nodes.len());
                        let cost_to_end = offset_cost
                            .wrapping_add(unsafe { *self.costs.length.get_unchecked(len) })
                            .wrapping_add(unsafe {
                                self.optimum_nodes.get_unchecked(node + len).cost_to_end
                            });
                        if cost_to_end < best_cost {
                            best_cost = cost_to_end;
                            best_item = (len as u32) | (offset << OPTIMUM_OFFSET_SHIFT);
                        }
                        len += 1;
                        if len > m.length as usize {
                            break;
                        }
                    }
                    mi += 1;
                    if mi == cptr {
                        break;
                    }
                }
                cptr = first;
            }

            // SAFETY: `node < optimum_nodes.len()` (invariant above).
            debug_assert!(node < self.optimum_nodes.len());
            unsafe {
                let n = self.optimum_nodes.get_unchecked_mut(node);
                n.item = best_item;
                n.cost_to_end = best_cost;
            }
            if node == 0 {
                break;
            }
        }

        self.tally_and_build_codes(block_length)
    }

    /// `deflate_tally_item_list` (`:2843-2868`) + `deflate_make_huffman_codes`.
    ///
    /// ## Soundness invariant (unchecked-index path walk)
    ///
    /// Same DP-output invariants `find_min_cost_path` already established, walked
    /// forward instead of backward:
    ///  * **`node`** only ever holds a value `< block_length` at the point it
    ///    indexes `optimum_nodes` — the loop reads `optimum_nodes[node]`, then
    ///    advances `node += length` and breaks IMMEDIATELY if that lands exactly
    ///    on `block_length` (a chosen path always partitions `0..block_length`
    ///    exactly, per the DP), so `node < block_length <= MAX_BLOCK_LENGTH <
    ///    optimum_nodes.len()` on every read.
    ///  * **`hi` as a literal byte** (`length == 1`) is always `< 256 ==
    ///    litlen_freqs.len()` by the DP's own encoding (`best_item = (literal <<
    ///    SHIFT) | 1` with `literal < 256`, `find_min_cost_path`).
    ///  * **`hi` as a match offset** (`length != 1`) is a valid DEFLATE offset,
    ///    so `offset_slot_full.slot_unchecked(hi)` is sound (same contract as
    ///    its other call site).
    ///  * **`length`** is `MIN_MATCH_LEN..=MAX_MATCH_LEN`, so `length_slot`
    ///    returns `< NUM_LEN_SLOTS` and `DEFLATE_FIRST_LEN_SYM + ls <= 257 + 28
    ///    == 285 < 288 == litlen_freqs.len()`.
    fn tally_and_build_codes(&self, block_length: usize) -> PathCodes {
        let mut litlen_freqs = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let mut offset_freqs = [0u32; DEFLATE_NUM_OFFSET_SYMS];

        let mut node = 0usize;
        loop {
            debug_assert!(node < self.optimum_nodes.len());
            let item = unsafe { self.optimum_nodes.get_unchecked(node).item };
            let length = (item & OPTIMUM_LEN_MASK) as usize;
            let hi = item >> OPTIMUM_OFFSET_SHIFT;
            if length == 1 {
                debug_assert!((hi as usize) < litlen_freqs.len());
                unsafe {
                    *litlen_freqs.get_unchecked_mut(hi as usize) += 1;
                }
            } else {
                let ls = length_slot(length as u32) as usize;
                debug_assert!(DEFLATE_FIRST_LEN_SYM + ls < litlen_freqs.len());
                unsafe {
                    *litlen_freqs.get_unchecked_mut(DEFLATE_FIRST_LEN_SYM + ls) += 1;
                }
                // SAFETY: `hi` is a valid DEFLATE match offset here (same
                // contract as `slot_unchecked`'s other call site).
                let slot = unsafe { self.offset_slot_full.slot_unchecked(hi) };
                debug_assert!(slot < offset_freqs.len());
                unsafe {
                    *offset_freqs.get_unchecked_mut(slot) += 1;
                }
            }
            node += length;
            if node == block_length {
                break;
            }
        }
        litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

        build_codes(litlen_freqs, offset_freqs)
    }

    /// `deflate_compute_true_cost` (`:2889-2921`): exact whole-bit cost of the
    /// block if the tallied path were coded with the built Huffman codes.
    fn compute_true_cost(&self, codes: &PathCodes) -> u32 {
        let header = build_dynamic_header(&codes.litcode.lens, &codes.offcode.lens);
        let mut cost: u64 = header.header_bits();

        for sym in 0..DEFLATE_FIRST_LEN_SYM {
            cost += codes.litlen_freqs[sym] as u64 * codes.litcode.lens[sym] as u64;
        }
        for slot in 0..NUM_LEN_SLOTS {
            let sym = DEFLATE_FIRST_LEN_SYM + slot;
            cost += codes.litlen_freqs[sym] as u64
                * (codes.litcode.lens[sym] as u64 + LENGTH_EXTRA_BITS[slot] as u64);
        }
        for slot in 0..OFFSET_EXTRA_BITS.len() {
            cost += codes.offset_freqs[slot] as u64
                * (codes.offcode.lens[slot] as u64 + OFFSET_EXTRA_BITS[slot] as u64);
        }
        cost as u32
    }

    /// `deflate_optimize_and_flush_block` (`:3416-3530`). Chooses the token path
    /// and emits the block via [`emit_block`]. Returns whether the "only
    /// literals" strategy was used (feeds the next block's min-len heuristic).
    #[allow(clippy::too_many_arguments)]
    fn optimize_and_flush(
        &mut self,
        buf: &[u8],
        block_begin: usize,
        block_length: usize,
        cache_end: usize,
        is_first: bool,
        is_final: bool,
        params: &LevelParams,
        statics: &StaticCodes,
        split_stats: &BlockSplitStats,
        prev_observations: &[u32; NUM_OBSERVATION_TYPES],
        prev_num_observations: u32,
        bw: &mut BitWriter,
    ) -> bool {
        let block = &buf[block_begin..block_begin + block_length];
        let np = &params.near_optimal;

        // (a) only-literals candidate.
        let only_lits_codes = build_all_literals_codes(block);
        let only_lits_cost = self.compute_true_cost(&only_lits_codes);

        // Pin the nodes past the block end so a match cannot extend past it.
        let hi = (block_length - 1 + MAX_MATCH_LEN as usize).min(self.optimum_nodes.len() - 1);
        for i in block_length..=hi {
            self.optimum_nodes[i].cost_to_end = 0x8000_0000;
        }

        // (b) static-Huffman-optimized candidate (small blocks only).
        let mut static_cost = u32::MAX;
        if block_length <= np.max_len_to_optimize_static_block as usize {
            self.costs_saved = self.costs.clone();
            self.costs
                .set_from_codes(&statics.litcode.lens, &statics.offcode.lens);
            self.find_min_cost_path(block_length, cache_end);
            static_cost = (self.optimum_nodes[0].cost_to_end / BIT_COST).wrapping_add(7);
            self.costs = self.costs_saved.clone();
        }

        // (c) iterative dynamic optimization.
        set_initial_costs(
            &mut self.costs,
            block,
            &self.match_len_freqs,
            params.max_search_depth,
            is_first,
            split_stats.observations(),
            split_stats.num_observations(),
            prev_observations,
            prev_num_observations,
        );

        let mut best_true_cost = u32::MAX;
        // Assigned on the first (always-executed) loop iteration; read after.
        let mut true_cost;
        let mut num_passes_remaining = np.max_optim_passes;
        loop {
            let codes = self.find_min_cost_path(block_length, cache_end);
            true_cost = self.compute_true_cost(&codes);
            if true_cost.wrapping_add(np.min_improvement_to_continue) > best_true_cost {
                break;
            }
            best_true_cost = true_cost;
            self.costs_saved = self.costs.clone();
            self.costs
                .set_from_codes(&codes.litcode.lens, &codes.offcode.lens);
            num_passes_remaining -= 1;
            if num_passes_remaining == 0 {
                break;
            }
        }

        // (d) selection.
        let mut used_only_literals = false;
        if only_lits_cost.min(static_cost) < best_true_cost {
            if only_lits_cost < static_cost {
                // Only literals is cheapest.
                used_only_literals = true;
                self.costs
                    .set_from_codes(&only_lits_codes.litcode.lens, &only_lits_codes.offcode.lens);
            } else {
                // Static block is cheapest: regenerate its path.
                self.costs
                    .set_from_codes(&statics.litcode.lens, &statics.offcode.lens);
                self.find_min_cost_path(block_length, cache_end);
            }
        } else if true_cost >= best_true_cost.wrapping_add(np.min_bits_to_use_nonfinal_path) {
            // The final pass regressed; recover the best non-final path.
            self.costs = self.costs_saved.clone();
            let codes = self.find_min_cost_path(block_length, cache_end);
            self.costs
                .set_from_codes(&codes.litcode.lens, &codes.offcode.lens);
        }
        // else: optimum_nodes already holds the final (good) path.

        // Build the token stream for the chosen path and emit.
        let mut sink = Sink::new();
        if used_only_literals {
            for &b in block {
                sink.push_literal(b);
            }
        } else {
            let mut node = 0usize;
            while node < block_length {
                let item = self.optimum_nodes[node].item;
                let length = (item & OPTIMUM_LEN_MASK) as usize;
                let hi = item >> OPTIMUM_OFFSET_SHIFT;
                if length == 1 {
                    sink.push_literal(hi as u8);
                    node += 1;
                } else {
                    sink.push_match(length as u32, hi);
                    node += length;
                }
            }
        }
        emit_block(bw, buf, block_begin, &sink, statics, is_final);

        used_only_literals
    }
}

/// `deflate_make_huffman_codes`: build the litlen + offset codes from freqs.
fn build_codes(
    litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
) -> PathCodes {
    let litcode = make_huffman_code(
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &litlen_freqs,
    );
    let offcode = make_huffman_code(
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        &offset_freqs,
    );
    PathCodes {
        litlen_freqs,
        offset_freqs,
        litcode,
        offcode,
    }
}

/// `deflate_choose_all_literals`: frequencies + codes for an all-literal block.
fn build_all_literals_codes(block: &[u8]) -> PathCodes {
    let mut litlen_freqs = [0u32; DEFLATE_NUM_LITLEN_SYMS];
    for &b in block {
        litlen_freqs[b as usize] += 1;
    }
    litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;
    build_codes(litlen_freqs, [0u32; DEFLATE_NUM_OFFSET_SYMS])
}

/// Near-optimal driver. Compresses `buf[data_start..in_end]` into DEFLATE blocks
/// appended to `bw` (a preset dictionary in `buf[..data_start]` is seeded into
/// the matchfinder but not coded). Port of `deflate_compress_near_optimal`.
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
) {
    let mut opt = Box::new(Optimizer::new());
    let mut bt_mf = BtMatchfinder::new();

    let depth = params.max_search_depth;
    let mut max_len = MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);
    let mut next_hashes = [0u32; 2];

    // Matchfinder window bookkeeping (indices into `buf`).
    let mut in_cur_base = 0usize;
    let mut in_next = 0usize;
    let mut in_next_slide = in_end.min(WINDOW_SIZE);

    // Seed a preset dictionary (untested in this increment; dict is always empty
    // here). Insert positions [0, data_start) into the bt tree without coding.
    while in_next < data_start {
        let remaining = in_end - in_next;
        if in_next == in_next_slide {
            bt_mf.slide_window();
            in_cur_base = in_next;
            in_next_slide = in_next + remaining.min(WINDOW_SIZE);
        }
        let mut ml = MAX_MATCH_LEN;
        let mut nl = nice_len;
        adjust_max_and_nice_len(&mut ml, &mut nl, remaining);
        if ml >= BT_MATCHFINDER_REQUIRED_NBYTES {
            bt_mf.skip_byte(
                buf,
                in_cur_base,
                (in_next - in_cur_base) as isize,
                nl,
                depth,
                &mut next_hashes,
            );
        }
        in_next += 1;
    }

    let mut in_block_begin = in_next;
    let mut split_stats = BlockSplitStats::new();
    let mut new_match_len_freqs = vec![0u32; MAX_MATCH_LEN as usize + 1];
    let mut prev_observations = [0u32; NUM_OBSERVATION_TYPES];
    let mut prev_num_observations = 0u32;
    let mut prev_block_used_only_literals = false;

    // deflate_near_optimal_init_stats: split_stats + match_len_freqs already zero.
    let mut cache_ptr = 0usize;

    loop {
        // Starting a new DEFLATE block.
        let in_max_block_end = choose_max_block_end(in_block_begin, in_end);
        let mut prev_end_block_check: Option<usize> = None;
        let mut change_detected = false;
        let mut next_observation = in_next;

        let min_len = if prev_block_used_only_literals {
            MAX_MATCH_LEN + 1
        } else {
            calculate_min_match_len(&buf[in_block_begin..in_max_block_end], depth)
        };

        // ## Soundness invariant (unchecked `match_cache` writes — forward fill)
        //
        // `opt.match_cache` is sized `MATCH_CACHE_TOTAL == MATCH_CACHE_LENGTH +
        // MAX_MATCHES_PER_POS + (MAX_MATCH_LEN - 1)` (`Optimizer::new`) — the
        // exact libdeflate slop (`:571-573`) that guarantees room for one more
        // FULL position's worth of writes (a header + up to `MAX_MATCHES_PER_POS`
        // matches, or up to `MAX_MATCH_LEN - 1` skip-header writes) after the
        // loop's own `cache_ptr >= MATCH_CACHE_LENGTH` overflow check fires. This
        // is the SAME invariant `find_min_cost_path`'s Tranche-1 unchecked reads
        // already trust ("cptr/mi walk match_cache within the forward pass's
        // contiguous write region") — this increment makes the write side that
        // produces that region unchecked too.
        loop {
            let remaining = in_end - in_next;

            // Slide the window forward if needed.
            if in_next == in_next_slide {
                bt_mf.slide_window();
                in_cur_base = in_next;
                in_next_slide = in_next + remaining.min(WINDOW_SIZE);
            }

            // Find and cache matches at the current position.
            let matches_start = cache_ptr;
            let mut best_len = 0u32;
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, remaining);
            if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES {
                let n = bt_mf.get_matches(
                    buf,
                    in_cur_base,
                    (in_next - in_cur_base) as isize,
                    max_len,
                    nice_len,
                    depth,
                    &mut next_hashes,
                    &mut opt.match_cache[matches_start..],
                );
                cache_ptr = matches_start + n;
                if n > 0 {
                    // SAFETY: see the soundness invariant above; `cache_ptr - 1
                    // == matches_start + n - 1` is a slot `get_matches` just
                    // wrote (`n <= MAX_MATCHES_PER_POS` slots from
                    // `matches_start`, within the cache's slop capacity).
                    debug_assert!(cache_ptr - 1 < opt.match_cache.len());
                    best_len =
                        unsafe { opt.match_cache.get_unchecked(cache_ptr - 1).length as u32 };
                }
            }

            // Observe a match or literal for the split / cost statistics.
            if in_next >= next_observation {
                if best_len >= min_len {
                    split_stats.observe_match(best_len);
                    next_observation = in_next + best_len as usize;
                    new_match_len_freqs[best_len as usize] += 1;
                } else {
                    split_stats.observe_literal(buf[in_next]);
                    next_observation = in_next + 1;
                }
            }

            // Write this position's cache header (num matches, literal byte).
            // SAFETY: see the soundness invariant above (`cache_ptr` bound).
            debug_assert!(cache_ptr < opt.match_cache.len());
            unsafe {
                let hdr = opt.match_cache.get_unchecked_mut(cache_ptr);
                hdr.length = (cache_ptr - matches_start) as u16;
                hdr.offset = buf[in_next] as u16;
            }
            in_next += 1;
            cache_ptr += 1;

            // Skip the interior of a very long match (don't cache its bytes).
            if best_len >= MIN_MATCH_LEN && best_len >= nice_len {
                let mut skip = best_len - 1;
                loop {
                    let remaining = in_end - in_next;
                    if in_next == in_next_slide {
                        bt_mf.slide_window();
                        in_cur_base = in_next;
                        in_next_slide = in_next + remaining.min(WINDOW_SIZE);
                    }
                    adjust_max_and_nice_len(&mut max_len, &mut nice_len, remaining);
                    if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES {
                        bt_mf.skip_byte(
                            buf,
                            in_cur_base,
                            (in_next - in_cur_base) as isize,
                            nice_len,
                            depth,
                            &mut next_hashes,
                        );
                    }
                    // SAFETY: see the soundness invariant above (`cache_ptr` bound).
                    debug_assert!(cache_ptr < opt.match_cache.len());
                    unsafe {
                        let hdr = opt.match_cache.get_unchecked_mut(cache_ptr);
                        hdr.length = 0;
                        hdr.offset = buf[in_next] as u16;
                    }
                    in_next += 1;
                    cache_ptr += 1;
                    skip -= 1;
                    if skip == 0 {
                        break;
                    }
                }
            }

            // Maximum block length or end of input reached?
            if in_next >= in_max_block_end {
                break;
            }
            // Match cache overflowed?
            if cache_ptr >= MATCH_CACHE_LENGTH {
                break;
            }
            // Not ready to check for a block end (again)?
            if !split_stats.ready_to_check_block(in_next - in_block_begin, in_end - in_next) {
                continue;
            }
            // Would ending the block be worthwhile?
            if split_stats.do_end_block_check((in_next - in_block_begin) as u32) {
                change_detected = true;
                break;
            }
            // Not worthwhile: merge the recent stats and remember this point.
            merge_stats(
                &mut split_stats,
                &mut opt.match_len_freqs,
                &mut new_match_len_freqs,
            );
            prev_end_block_check = Some(in_next);
        }

        // Choose the block end + the item sequence, then flush.
        let rewind_end = if change_detected {
            prev_end_block_check
        } else {
            None
        };
        if let Some(in_block_end) = rewind_end {
            let block_length = in_block_end - in_block_begin;
            let is_first = in_block_begin == data_start;
            let mut num_bytes_to_rewind = in_next - in_block_end;

            // Rewind the match cache to the chosen block end.
            // SAFETY: `cache_ptr` only ever walks backward here, starting
            // `< orig_cache_ptr <= opt.match_cache.len()` and stepping onto
            // header slots the forward fill above already wrote (a header's
            // `length` is exactly its own match count, so subtracting it lands
            // on the PRECEDING header) — the same backward-walk invariant
            // `find_min_cost_path` relies on for the same array.
            let orig_cache_ptr = cache_ptr;
            while num_bytes_to_rewind != 0 {
                cache_ptr -= 1;
                debug_assert!(cache_ptr < opt.match_cache.len());
                cache_ptr -= unsafe { opt.match_cache.get_unchecked(cache_ptr).length as usize };
                num_bytes_to_rewind -= 1;
            }
            let cache_len_rewound = orig_cache_ptr - cache_ptr;
            let block_cache_end = cache_ptr;

            prev_block_used_only_literals = opt.optimize_and_flush(
                buf,
                in_block_begin,
                block_length,
                block_cache_end,
                is_first,
                false,
                params,
                statics,
                &split_stats,
                &prev_observations,
                prev_num_observations,
                bw,
            );

            // Move the rewound tail back to the start of the cache.
            opt.match_cache
                .copy_within(cache_ptr..cache_ptr + cache_len_rewound, 0);
            cache_ptr = cache_len_rewound;

            save_stats(
                &split_stats,
                &mut prev_observations,
                &mut prev_num_observations,
            );
            // Clear the flushed block's stats, keep the next block's beginning.
            split_stats.clear_old_observations();
            for f in opt.match_len_freqs.iter_mut() {
                *f = 0;
            }
            in_block_begin = in_block_end;
        } else {
            let block_length = in_next - in_block_begin;
            let is_first = in_block_begin == data_start;
            // BFINAL only on the last internal block AND only if this is the
            // last chunk of the stream; a non-final chunk closes with the
            // caller-appended sync-flush marker instead.
            let is_final = is_last && in_next == in_end;

            merge_stats(
                &mut split_stats,
                &mut opt.match_len_freqs,
                &mut new_match_len_freqs,
            );
            prev_block_used_only_literals = opt.optimize_and_flush(
                buf,
                in_block_begin,
                block_length,
                cache_ptr,
                is_first,
                is_final,
                params,
                statics,
                &split_stats,
                &prev_observations,
                prev_num_observations,
                bw,
            );

            cache_ptr = 0;
            save_stats(
                &split_stats,
                &mut prev_observations,
                &mut prev_num_observations,
            );
            // init_stats: reset split stats + match_len_freqs for the next block.
            split_stats.reset();
            for f in opt.match_len_freqs.iter_mut() {
                *f = 0;
            }
            for f in new_match_len_freqs.iter_mut() {
                *f = 0;
            }
            in_block_begin = in_next;
        }

        if in_next == in_end {
            break;
        }
    }
}

/// `deflate_near_optimal_merge_stats`: fold the recent split observations and the
/// new match-length frequencies into the running totals.
fn merge_stats(
    split_stats: &mut BlockSplitStats,
    match_len_freqs: &mut [u32],
    new_match_len_freqs: &mut [u32],
) {
    split_stats.merge_new_observations();
    for (dst, src) in match_len_freqs
        .iter_mut()
        .zip(new_match_len_freqs.iter_mut())
    {
        *dst += *src;
        *src = 0;
    }
}

/// `deflate_near_optimal_save_stats`.
fn save_stats(
    split_stats: &BlockSplitStats,
    prev_observations: &mut [u32; NUM_OBSERVATION_TYPES],
    prev_num_observations: &mut u32,
) {
    *prev_observations = *split_stats.observations();
    *prev_num_observations = split_stats.num_observations();
}

#[cfg(test)]
mod tests {
    // near_optimal -> parse(super) -> deflate(super::super); compress_gzip /
    // compress_oneshot live in deflate::mod.
    use super::super::super::{compress_gzip, compress_oneshot};
    use std::io::Read;

    fn decode(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 decode");
        out
    }

    #[test]
    fn near_optimal_roundtrips_levels_10_11_12() {
        let mut data = Vec::new();
        let phrase = b"near-optimal deflate must roundtrip byte for byte across blocks. ";
        for i in 0..12000 {
            data.extend_from_slice(phrase);
            if i % 11 == 0 {
                data.extend_from_slice(format!("<{i}>").as_bytes());
            }
        }
        for level in [10u32, 11, 12] {
            let gz = compress_gzip(&data, level);
            assert_eq!(decode(&gz), data, "L{level} roundtrip");
        }
    }

    #[test]
    fn near_optimal_beats_lazy2_on_text() {
        // The DP should not be WORSE than the L9 lazy2 parse on compressible text.
        let mut data = Vec::new();
        let phrase = b"the near optimal parser weighs fractional bit costs. ";
        for _ in 0..8000 {
            data.extend_from_slice(phrase);
        }
        let l9 = compress_oneshot(&data, 9).len();
        let l12 = compress_oneshot(&data, 12).len();
        assert!(
            l12 <= l9,
            "L12 near-optimal ({l12}) worse than L9 lazy2 ({l9})"
        );
    }
}
