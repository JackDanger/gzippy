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
use super::super::encode_types::HeaderBudget;
use super::super::huffman::{
    build_dynamic_header, make_huffman_code, CodeScratch, HeaderScratch, HuffmanCode,
};
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
    /// Per-block sequence sink, reused across blocks via `begin()` (which
    /// `.clear()`s `seqs` without dropping its capacity) instead of being
    /// reconstructed fresh per block. A brand-new `Sink::new()` per block
    /// forces `seqs` to regrow from empty every time — DHAT on the L10
    /// bin6/parquet corpora showed this as the #1-by-count allocation site
    /// (`push_seq`, thousands of allocs, one growth spurt per block).
    sink: Sink,
    /// `build_dynamic_header`'s scratch buffers (see `HeaderScratch`'s doc
    /// comment), reused the same way `sink` is: `compute_true_cost` calls
    /// `build_dynamic_header` 2-4x per block (once per refinement pass) plus
    /// once more in `optimize_and_flush`'s final emit, so a fresh `Vec` per
    /// call was the same waste class as the pre-fix `sink`.
    header_scratch: HeaderScratch,
    code_scratch: CodeScratch,
}

impl Optimizer {
    fn new(budget: HeaderBudget) -> Self {
        Optimizer {
            match_cache: vec![LzMatch::default(); MATCH_CACHE_TOTAL],
            optimum_nodes: vec![OptimumNode::default(); OPTIMUM_NODES_LEN],
            costs: DeflateCosts::default(),
            costs_saved: DeflateCosts::default(),
            offset_slot_full: OffsetSlotFull::new(),
            match_len_freqs: vec![0u32; MAX_MATCH_LEN as usize + 1],
            sink: Sink::new(),
            header_scratch: HeaderScratch::new(),
            code_scratch: CodeScratch {
                budget,
                ..Default::default()
            },
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
    fn compute_true_cost(&mut self, codes: &PathCodes) -> u32 {
        let header = build_dynamic_header(
            &codes.litcode.lens,
            &codes.offcode.lens,
            &mut self.header_scratch,
        );
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

        // Build the token stream for the chosen path and emit. Reuse the
        // persistent `self.sink` (cleared, capacity kept) instead of a fresh
        // `Sink::new()` per block — see the field doc comment.
        self.sink.begin();
        if used_only_literals {
            for &b in block {
                self.sink.push_literal(b);
            }
        } else {
            let mut node = 0usize;
            while node < block_length {
                let item = self.optimum_nodes[node].item;
                let length = (item & OPTIMUM_LEN_MASK) as usize;
                let hi = item >> OPTIMUM_OFFSET_SHIFT;
                if length == 1 {
                    self.sink.push_literal(hi as u8);
                    node += 1;
                } else {
                    self.sink.push_match(length as u32, hi);
                    node += length;
                }
            }
        }
        emit_block(
            bw,
            buf,
            block_begin,
            &self.sink,
            statics,
            is_final,
            &mut self.header_scratch,
            &mut self.code_scratch,
            false,
            None,
        );

        // Lever-#3 freshness-chain probe (2026-09-25): the flip rate of the
        // only-literals selection that feeds the NEXT block's
        // `min_match_len`. feature-off builds compile both to zero bytes.
        crate::anatomy_count!(near_opt_flush_blocks);
        if used_only_literals {
            crate::anatomy_count!(near_opt_only_literals_blocks);
        }

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

// ── `near-opt-parallel-flush` test/observability surface ────────────────────
// Thin re-exports of the `parallel_flush` module's probe accessors at THIS
// module's top level so `parse::near_opt_flush_probe` can name them (the
// module itself stays private to the near-optimal parser).

// The whole block is allowed dead: its only non-test caller is nothing (the
// integration tests drive it; see the probe re-export in parse/mod.rs).

#[cfg(feature = "near-opt-parallel-flush")]
#[allow(dead_code)]
pub(super) fn parallel_flush_writes() -> u64 {
    parallel_flush::parallel_flush_writes()
}

#[cfg(feature = "near-opt-parallel-flush")]
#[allow(dead_code)]
pub(super) fn set_force_serial(v: bool) {
    parallel_flush::set_force_serial(v);
}

#[cfg(feature = "near-opt-parallel-flush")]
#[allow(dead_code)]
pub(super) fn stale_flag_fired() -> bool {
    parallel_flush::stale_flag_fired()
}

#[cfg(feature = "near-opt-parallel-flush")]
#[allow(dead_code)]
pub(super) fn set_stale_flag_for_tests(v: bool) {
    parallel_flush::set_stale_flag_for_tests(v);
}

// ===========================================================================
// LEVER #3 — block-parallel `optimize_and_flush` (feature
// `near-opt-parallel-flush`, DEFAULT OFF). Docs/board/sprint-2026-09-25.md,
// "Lever ledger" row 3.
// ===========================================================================

/// The flush-dispatch instrument for [`run`] (feature-gated; compiles to
/// nothing without `near-opt-parallel-flush`).
///
/// ## What leaves the chunk's serial critical path
///
/// Today's per-internal-block shape is [bt FILL: per-position cache + split
/// stats] → [`optimize_and_flush`] → [next block's fill], all on one thread.
/// Lever-0's conserved split prices the flush share at 52.9% of the 3 MB chunk
/// (13 blocks: fill 104.1 ms / flush 117.7 ms). With this feature on, the fill
/// thread keeps producing and hands each completed block's flush sub-phase
/// (the DP refinement passes + per-candidate Huffman construction + block
/// emission) to a pool of `min(2, num_cpus - 1)` worker threads; the chunk's
/// own thread only fills, snapshots, and — at chunk end — repacks the finished
/// fragments into `bw` IN BLOCK ORDER (see `Dispatcher::finish_and_write`).
///
/// ## The carriers contract (verified against this file)
///
/// `optimize_and_flush`'s inputs are all FILL-produced: the block's byte
/// range; its cached bt matches — `opt.match_cache[0..cache_end]`, which for
/// EVERY block starts at index 0 (a non-rewind flush leaves `cache_ptr == 0`,
/// and a rewind flush copies the rewound tail back to index 0 before the next
/// fill); the merged `match_len_freqs`; the final `split_stats`; the previous
/// block's `prev_observations`/`prev_num_observations` (a pure copy of block
/// N-1's end-of-fill stats, taken after the previous flush but not BY it);
/// `params`/`statics` (shared immutable inputs); `is_first`/`is_final`.
///
/// **One input is NOT fill-produced, and the mission's carrier list omits
/// it**: `self.costs` at flush entry. For every non-first block,
/// `set_initial_costs` runs `adjust_costs`, which BLENDS the entering cost
/// model toward the block's default costs (`costs.rs:320-348` —
/// `adjust_impl` reads the current `self.literal[i]` / `self.length[len]` /
/// `self.offset_slot[slot]` values), and that entering value is the previous
/// flush's EXIT state (`optimize_and_flush`'s (d) selection leaves
/// `self.costs` set from the chosen path's code lengths, `:409-428`). The
/// b5281a96 spike cleared the freshness chain's OTHER carrier
/// (`prev_block_used_only_literals`, 0 flips / 47 flushes); this one is
/// structural — it carries data on every similar-block pair, i.e. exactly on
/// the campaign corpora. Two flushes of the same chunk therefore cannot run
/// CONCURRENTLY and stay byte-exact.
///
/// The design is consequently the pipelined shape the lever itself names:
/// flush N runs concurrently with fill N+1 (and with other chunks' fills in
/// the T>1 dispatcher), with the flushes THEMSELVES chained in block order.
/// Consecutive flushes on this chain cost the chunk
/// ≈ one last-flush tail, matching the ledger's projected ~0.60-0.75x chunk
/// wall; the flush share does NOT divide by the worker count, and this
/// module documents why instead of silently producing shifted bytes.
///
/// ## Writeback (mission contract #3)
///
/// Each flush runs into its own `BitWriter::from_vec(Vec::new())` and reports
/// `finish_unaligned()`'s `(bytes, pad_bits)` — the per-block pending state.
/// `finish_and_write` takes the results IN BLOCK ORDER and appends them to the
/// real `bw` through `BitWriter::append_fragment`, which shifts the stream's
/// accumulated partial byte across the fragment (and, for the only fragment
/// family that needs it, a block whose emission chose STORED — which
/// byte-aligns relative to its own start — replays the serial writer's own
/// 3-bit header + `align_to_byte` + verbatim tail). The concatenation is
/// bit-identical to one writer having produced the whole chunk, pinned whole-
/// chunk by `tests/l9_t4_chunk_cost_probe.rs::parallel_flush_*` on the probe
/// corpus and the four campaign corpora.
///
/// ## The stale-flag guard (mission contract #4)
///
/// `finish_and_write` tallies the chunk's `used_only_literals` returns —
/// exactly the condition behind the `near_opt_flush_blocks` /
/// `near_opt_only_literals_blocks` counters (which keep firing, from the
/// workers, as process-wide atomics). A nonzero tally logs to stderr ONCE per
/// process and latches a static veto that routes every subsequent chunk
/// (any near-optimal `run` in this process) to the plain serial loop. The
/// zero-flip finding means the latch never fires on the benchmarks; the
/// within-chunk dispatch order cannot retro-fit the boolean into block N+1's
/// `min_match_len` anyway — that one-block drift is what the guard
/// quarantines rather than what it can repair, which is exactly why the
/// count must stand at chunk end.
///
/// ## Deviation from the mission's thread-shape letter
///
/// The mission says "std::thread scope". The pool uses plain
/// `std::thread::spawn` + `JoinHandle`s joined at write-back, with fully
/// owned jobs (block bytes + cache-region copies, `LevelParams` copied into
/// the job — it is `Copy`). Sharing `&buf`/`&LevelParams` through a
/// `thread::scope` closure would have required moving the entire ~300-line
/// fill loop inside the scope body (reindenting every line of the hot loop
/// the feature must not touch); owned jobs keep the fill loop untouched
/// (mission constraint #6) at the cost of one bounded snapshot copy per
/// flushed block (~1.3 MB typical, worst case ~6 MB at the match-cache
/// overflow cap). Teardown is deterministic all the same: dispatch never
/// leaks a worker; `finish_and_write` joins every handle before returning.
///
/// ## The anatomy instruments under this feature
///
/// `anatomy_wall_time!(near_opt_flush_ns, ...)`: the dispatched flushes never
/// enter it (their work is off the fill thread by construction — an
/// instrument run must not report the same wall twice, serial or parallel).
/// The serial timer sites are untouched. The flip counters DO fire from the
/// pool workers. This module's own `WRITES` counter is the test/observability
/// surface that distinguishes "the pool actually engaged" from a feature-off
/// build — a byte-identity test whose parallel arm silently ran serial would
/// otherwise pass vacuously.
#[cfg(feature = "near-opt-parallel-flush")]
mod parallel_flush {
    use super::*;

    use std::collections::{BTreeMap, VecDeque};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Condvar, Mutex};

    /// Process-wide stale-flag latch (mission contract #4): a chunk whose
    /// parallelized flushes used the only-literals strategy vetoes the pool
    /// for the rest of the process. The b5281a96 finding (0 flips across 47
    /// flushes) means this never fires on the campaign corpora.
    static STALE_FLAG: AtomicBool = AtomicBool::new(false);
    /// One-shot stderr note for the first latch (mission contract #4: log
    /// once per process, not once per chunk).
    static STALE_FLAG_LOGGED: AtomicBool = AtomicBool::new(false);
    /// Test/observability toggle: force the pool off regardless of CPU count
    /// or the stale flag, without a second build. Same rationale as the
    /// repo's other measurement-feature surfaces (`probe::enable`/`take`,
    /// `encode_census::reset`): the byte-identity probe needs BOTH arms of
    /// the same binary, and no env var may pick an encode path in a
    /// production build (CLAUDE.md non-negotiable #3) — this static exists
    /// only under the default-off feature.
    static FORCE_SERIAL: AtomicBool = AtomicBool::new(false);
    /// Number of pools that completed a write-back (test/observability; see
    /// the module doc — an identity test must prove its parallel arm ran).
    static WRITES: AtomicU64 = AtomicU64::new(0);

    /// `min(2, cpus - 1)` extra flush workers (mission contract #2). A
    /// single-CPU host gets 0 and never spawns the pool (pure serial bytes
    /// fall out of the `None` arm).
    fn worker_count() -> usize {
        num_cpus::get().saturating_sub(1).min(2)
    }

    /// Whether this `run` should engage the pool (feature on, no stale flag,
    /// no test override, at least one worker).
    pub(super) fn engaged() -> bool {
        !FORCE_SERIAL.load(Relaxed) && !STALE_FLAG.load(Relaxed) && worker_count() > 0
    }

    // ── test/observability surface (`parse` re-exports this block) ──
    // (allowed dead: driven from the integration tests through the
    // `near_optimal::` re-exports, never from the binary)

    /// Probe accessors, re-exported by `parse::near_opt_flush_probe` for the
    /// integration tests. Private to this module otherwise.
    #[allow(dead_code)]
    pub(super) fn parallel_flush_writes() -> u64 {
        WRITES.load(Relaxed)
    }

    #[allow(dead_code)]
    pub(super) fn note_parallel_flush_write() {
        WRITES.fetch_add(1, Relaxed);
    }

    #[allow(dead_code)]
    pub(super) fn set_force_serial(v: bool) {
        FORCE_SERIAL.store(v, Relaxed);
    }

    #[allow(dead_code)]
    pub(super) fn stale_flag_fired() -> bool {
        STALE_FLAG.load(Relaxed)
    }

    #[allow(dead_code)]
    pub(super) fn set_stale_flag_for_tests(v: bool) {
        // Never log from a test-driven write: the one-shot stderr note is for
        // real chunks only (the test asserts the LATCH, not the print).
        STALE_FLAG_LOGGED.store(true, Relaxed);
        STALE_FLAG.store(v, Relaxed);
    }

    // ── job + result plumbing ──

    /// A finished block fragment, ready for in-order write-back.
    struct Fragment {
        /// `finish_unaligned`'s bytes: complete bytes, then the final partial
        /// byte zero-padded under `pad_bits`.
        bytes: Vec<u8>,
        /// High bits of the fragment's LAST byte that are padding (0..=7).
        pad_bits: u8,
        /// The block's emission chose STORED (at least one BTYPE=00
        /// sub-block): the fragment byte-aligns relative to its own start
        /// and must be written back through the alignment replay, never
        /// plain bit-shifting. Relayed onto the chunk thread for
        /// `STORED_BLOCK_EMITTED` (the T>1 splicer's tripwire lives in a
        /// thread-local set by the emitting thread).
        stored: bool,
        /// The flush's only-literals decision — the per-chunk stale-flag
        /// tally (the runtime guard's condition).
        used_only_literals: bool,
    }

    /// One flushable block: every input `optimize_and_flush` needs,
    /// snapshotted at dispatch (see the module doc's carriers audit).
    struct Job {
        index: usize,
        /// `buf[block_begin .. block_begin + block_length]` copied — the two
        /// consumers that read raw bytes (`build_all_literals_codes` and the
        /// emit's literal runs) must see the fill instant's bytes.
        block: Vec<u8>,
        /// `opt.match_cache[0..cache_end]`: the block's bt match headers +
        /// matches, ALWAYS starting at index 0 (see the module doc).
        cache_region: Vec<LzMatch>,
        /// The merged approximate match-length histogram at block end.
        match_len_freqs: Vec<u32>,
        split_stats: BlockSplitStats,
        prev_observations: [u32; NUM_OBSERVATION_TYPES],
        prev_num_observations: u32,
        is_first: bool,
        is_final: bool,
        params: LevelParams,
    }

    /// Shared worker state: the FIFO job queue (bounded at the worker count —
    /// the block-order chain, below, makes a deeper backlog pure memory), the
    /// per-index result slots, and this chunk's flush→flush exit-cost mailbox
    /// (keyed by flushed-index; the chain is per-chunk because chunks of the
    /// T>1 dispatcher run concurrently and each starts a fresh cost chain).
    struct Core {
        queue: Mutex<Queue>,
        queue_cv: Condvar,
        results: Mutex<Vec<Option<Fragment>>>,
        results_cv: Condvar,
        /// `exited[k]` = the cost model block `k`'s flush LEFT. Job `k+1`'s
        /// worker parks on this key (and consumes it) before optimizing.
        exited: Mutex<BTreeMap<usize, DeflateCosts>>,
        exited_cv: Condvar,
    }

    struct Queue {
        jobs: VecDeque<Job>,
        open: bool,
    }

    /// The flush pool. Created per `run`; workers spawn lazily on the first
    /// dispatch and join in `finish_and_write` (see the module doc's
    /// thread-shape note).
    pub(super) struct Dispatcher {
        workers: usize,
        statics: &'static StaticCodes,
        budget: HeaderBudget,
        core: Arc<Core>,
        handles: Vec<std::thread::JoinHandle<()>>,
        spawned: bool,
        /// Ordinal of the next dispatched block (the result-slot index).
        next_index: usize,
    }

    impl Dispatcher {
        /// A dispatcher, or `None` when the pool must not run: `min(2,
        /// cpus-1) == 0` (a single-CPU host has no spare flush worker), the
        /// test/observability override is on, or the stale-flag guard has
        /// latched a previous chunk's only-literals flip. For every `None`
        /// the serial arms of the flush sites below run byte-identical to
        /// today's code — the structural no-op the contract requires.
        pub(super) fn maybe_new(
            statics: &'static StaticCodes,
            budget: HeaderBudget,
        ) -> Option<Self> {
            if !engaged() {
                return None;
            }
            Some(Dispatcher {
                workers: worker_count(),
                statics,
                budget,
                core: Arc::new(Core {
                    queue: Mutex::new(Queue {
                        jobs: VecDeque::new(),
                        open: true,
                    }),
                    queue_cv: Condvar::new(),
                    results: Mutex::new(Vec::new()),
                    results_cv: Condvar::new(),
                    exited: Mutex::new(BTreeMap::new()),
                    exited_cv: Condvar::new(),
                }),
                handles: Vec::new(),
                spawned: false,
                next_index: 0,
            })
        }

        /// Hand one completed block's flush to the pool: the dispatcher
        /// snapshots every FILL-produced input at this instant (the fill will
        /// overwrite the match-cache region and zero the freqs immediately
        /// after) and pushes the job. Blocks are pushed in fill order and the
        /// workers process them as a chained pipeline (each flush parks on
        /// its predecessor's exit costs), so the queue bound of `workers` is
        /// enough: the third-and-later pending flushes would sit on their
        /// chain slot anyway.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn snapshot_and_dispatch(
            &mut self,
            block: Vec<u8>,
            cache_region: Vec<LzMatch>,
            match_len_freqs: Vec<u32>,
            split_stats: BlockSplitStats,
            prev_observations: [u32; NUM_OBSERVATION_TYPES],
            prev_num_observations: u32,
            is_first: bool,
            is_final: bool,
            params: LevelParams,
        ) {
            let index = self.next_index;
            self.next_index += 1;
            let job = Job {
                index,
                block,
                cache_region,
                match_len_freqs,
                split_stats,
                prev_observations,
                prev_num_observations,
                is_first,
                is_final,
                params,
            };
            if !self.spawned {
                self.spawned = true;
                for _ in 0..self.workers {
                    let core = Arc::clone(&self.core);
                    let (statics, budget) = (self.statics, self.budget);
                    self.handles.push(std::thread::spawn(move || {
                        flush_worker(&core, statics, budget)
                    }));
                }
            }
            {
                let mut results = self.core.results.lock().unwrap();
                debug_assert!(
                    results.len() == index,
                    "near-opt parallel flush: results registered out of order"
                );
                results.push(None);
            }
            {
                let mut queue = self.core.queue.lock().unwrap();
                // Backpressure: hold the fill thread while the queue is at
                // capacity. The chain means each queued flush's predecessor
                // is already in a worker's hands, so this parks only during
                // real contention.
                while queue.jobs.len() >= self.workers {
                    queue = self.core.queue_cv.wait(queue).unwrap();
                }
                queue.jobs.push_back(job);
            }
            self.core.queue_cv.notify_one();
        }

        /// Close the queue, wait for every dispatched flush, join the
        /// workers, then write the finished fragments into `bw` in block
        /// order. Returns the chunk's only-literals count (the guard's
        /// condition).
        pub(super) fn finish_and_write(mut self, bw: &mut BitWriter) -> usize {
            {
                let mut queue = self.core.queue.lock().unwrap();
                queue.open = false;
            }
            self.core.queue_cv.notify_all();
            {
                let mut results = self.core.results.lock().unwrap();
                while results.iter().any(std::option::Option::is_none) {
                    results = self.core.results_cv.wait(results).unwrap();
                }
            }
            for handle in self.handles.drain(..) {
                let _ = handle.join();
            }
            let mut flips = 0usize;
            {
                let mut results = self.core.results.lock().unwrap();
                for slot in results.iter_mut() {
                    let frag = slot.take().expect("all slots filled before write-back");
                    // RELAY the stored tripwire onto the chunk thread: the
                    // emission ran here, but `STORED_BLOCK_EMITTED` (and the
                    // T>1 splicer that reads it) belongs to the thread that
                    // ran `run`.
                    if frag.stored {
                        super::super::super::note_stored_block_emitted();
                    }
                    if frag.used_only_literals {
                        flips += 1;
                    }
                    bw.append_fragment(&frag.bytes, frag.pad_bits, frag.stored);
                }
            }
            note_parallel_flush_write();
            if flips > 0 {
                // Mission contract #4: the runtime guard. The zero-flip
                // finding makes this unreachable on the campaign corpora;
                // when it does fire the process latches to serial.
                STALE_FLAG.store(true, Relaxed);
                if !STALE_FLAG_LOGGED.swap(true, Relaxed) {
                    eprintln!(
                        "gzippy: near-optimal parallel flush guard tripped \
                         ({flips} only-literals flushes in one chunk) -- \
                         falling back to the serial optimize_and_flush for the \
                         rest of this process"
                    );
                }
            }
            flips
        }
    }

    /// One flush worker: pull jobs FIFO, wait on the chain, run the EXACT
    /// serial `optimize_and_flush` into a private writer, record the
    /// fragment in this job's slot.
    fn flush_worker(core: &Core, statics: &'static StaticCodes, budget: HeaderBudget) {
        // One pooled Optimizer per worker: the serial Optimizer's exact shape
        // (match cache, DP node table, cost model, sink, header/code
        // scratch), reused across blocks of every chunk this worker serves.
        // Nothing in it survives a flush that the next flush reads EXCEPT
        // `costs` — which the chain re-seeds per job (see `run_job`).
        let mut opt = Optimizer::new(budget);
        loop {
            let job = {
                let mut queue = core.queue.lock().unwrap();
                loop {
                    if let Some(job) = queue.jobs.pop_front() {
                        core.queue_cv.notify_all();
                        break job;
                    }
                    if !queue.open {
                        return;
                    }
                    queue = core.queue_cv.wait(queue).unwrap();
                }
            };
            let index = job.index;
            let frag = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_job(core, &mut opt, job, statics)
            }))
            .unwrap_or_else(|_| {
                // A panicking flush must not hang its successors: the
                // increment below posts a POISONED entry into the
                // flush→flush chain (zero costs — the next flush then
                // computes *something*, and the empty fragment above makes
                // the whole-chunk stream a LOUD mismatch the byte-identity
                // tests reject), where `run_job`'s own post would have
                // been skipped.
                core.exited
                    .lock()
                    .unwrap()
                    .insert(index, DeflateCosts::default());
                core.exited_cv.notify_all();
                Fragment {
                    // A panicking flush leaves its slot EMPTY-sized: the
                    // write-back then produces a stream that the
                    // byte-identity tests reject loudly instead of the
                    // process hanging on a poisoned pipeline. (In release
                    // builds `panic = "abort"` takes the process first.)
                    bytes: Vec::new(),
                    pad_bits: 0,
                    stored: false,
                    used_only_literals: false,
                }
            });
            {
                let mut results = core.results.lock().unwrap();
                results[index] = Some(frag);
            }
            core.results_cv.notify_all();
        }
    }

    /// The flush itself: identical inputs, identical call, only the writer
    /// is this block's private `BitWriter`.
    fn run_job(core: &Core, opt: &mut Optimizer, job: Job, statics: &StaticCodes) -> Fragment {
        // The costs carrier the carrier list omitted (see the module doc):
        // block N's entering cost model is block N-1's exit state; block 0
        // enters with the fresh Optimizer's zero costs, i.e. exactly what the
        // serial run's fresh `Optimizer` holds. For every index > 0 the
        // worker has already parked on (and consumed) the predecessor's
        // exit entry before this call.
        if job.index > 0 {
            park_on_exit_costs(core, opt, job.index);
        }
        opt.match_cache[..job.cache_region.len()].copy_from_slice(&job.cache_region);
        opt.match_len_freqs.copy_from_slice(&job.match_len_freqs);
        // Per-flush stored tripwire on THIS thread: reset-read around the
        // emit, relayed at write-back.
        super::super::super::clear_stored_block_emitted();
        let mut w = BitWriter::from_vec(Vec::new());
        let used_only_literals = opt.optimize_and_flush(
            &job.block,
            0,
            job.block.len(),
            job.cache_region.len(),
            job.is_first,
            job.is_final,
            &job.params,
            statics,
            &job.split_stats,
            &job.prev_observations,
            job.prev_num_observations,
            &mut w,
        );
        // Chain: post this flush's EXIT state where job index+1's worker
        // will find it. Serial `run` just leaves `opt.costs` in place for the
        // next block; the pipelined worker does the same thing through the
        // mailbox (the pooled Optimizer's costs are re-seeded at every job
        // entry, so nothing else in it matters across jobs).
        record_exit_costs(core, job.index, opt.costs.clone());
        let (bytes, pad_bits) = w.finish_unaligned();
        Fragment {
            bytes,
            pad_bits,
            stored: super::super::super::stored_block_emitted_on_this_thread(),
            used_only_literals,
        }
    }

    /// Block `index > 0` parks here until block `index - 1`'s worker posts
    /// the exited cost model (index key `k` = the mode LEAVING flush `k`):
    /// the flush→flush chain the carriers audit found. FIFO queue discipline
    /// guarantees the predecessor was popped before this job.
    fn park_on_exit_costs(core: &Core, opt: &mut Optimizer, index: usize) {
        let mut exited = core.exited.lock().unwrap();
        while !exited.contains_key(&(index - 1)) {
            exited = core.exited_cv.wait(exited).unwrap();
        }
        if let Some(costs) = exited.remove(&(index - 1)) {
            opt.costs = costs;
        }
    }

    /// Post the flushed block's exited cost model where block `index+1`'s
    /// worker will consume it (and wake it).
    fn record_exit_costs(core: &Core, index: usize, costs: DeflateCosts) {
        core.exited.lock().unwrap().insert(index, costs);
        core.exited_cv.notify_all();
    }
}

/// Near-optimal driver. Compresses `buf[data_start..in_end]` into DEFLATE blocks
/// appended to `bw` (a preset dictionary in `buf[..data_start]` is seeded into
/// the matchfinder but not coded). Port of `deflate_compress_near_optimal`.
///
/// `statics` is `&'static` because the pool workers outlive this call's frame
/// (`near-opt-parallel-flush` flush workers; every other use is reference
/// sharing, unchanged). The only caller builds it from the process-wide
/// `OnceLock`, so this widening is free.
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &'static StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
    budget: HeaderBudget,
) {
    let mut opt = Box::new(Optimizer::new(budget));
    let mut bt_mf = BtMatchfinder::new();

    // LEVER #3 (feature `near-opt-parallel-flush`): the flush dispatcher.
    // `None` without the feature, on a single-CPU host, or after the stale
    // flag latched — every `None` arm is exactly today's serial code (see the
    // parallel_flush module doc for the carriers audit and the guard).
    #[cfg(feature = "near-opt-parallel-flush")]
    let mut flush_pipe = parallel_flush::Dispatcher::maybe_new(statics, budget);

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
        //
        // LIVE SUBJECTION: one `anatomy_wall_time!` per OUTER-loop iteration
        // wraps the whole inner fill loop (per-internal-block granularity —
        // the contract; per-position timing is contract-forbidden). The
        // inner loop's `continue`/`break` both target this loop, which lives
        // INSIDE the timer's body block, so the accounting stays correct.
        crate::anatomy_wall_time!(near_opt_fill_ns, near_opt_fill_calls, {
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
        });

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

            #[cfg(feature = "near-opt-parallel-flush")]
            let prev_used_only_literals = match flush_pipe.as_mut() {
                // LEVER #3 dispatch: hand the flush sub-phase to a pool
                // worker with fill-instant snapshots of every carrier; the
                // only chunk-visible boolean result (the only-literals
                // backedge) is ASSUMED false here — the b5281a96 zero-flip
                // finding makes the assumption true on the campaign corpora,
                // and finish_and_write's chunk-end guard is what keeps it
                // honest (see the parallel_flush module doc).
                Some(pipe) => {
                    pipe.snapshot_and_dispatch(
                        buf[in_block_begin..in_block_end].to_vec(),
                        opt.match_cache[..block_cache_end].to_vec(),
                        opt.match_len_freqs.clone(),
                        split_stats.clone(),
                        prev_observations,
                        prev_num_observations,
                        is_first,
                        false,
                        *params,
                    );
                    false
                }
                None => crate::anatomy_wall_time!(near_opt_flush_ns, near_opt_flush_calls, {
                    opt.optimize_and_flush(
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
                    )
                }),
            };
            #[cfg(not(feature = "near-opt-parallel-flush"))]
            let prev_used_only_literals =
                crate::anatomy_wall_time!(near_opt_flush_ns, near_opt_flush_calls, {
                    opt.optimize_and_flush(
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
                    )
                });
            prev_block_used_only_literals = prev_used_only_literals;

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

            #[cfg(feature = "near-opt-parallel-flush")]
            let prev_used_only_literals = match flush_pipe.as_mut() {
                // LEVER #3 dispatch (final-branch): same contract as the
                // rewind branch above. `is_final` is known at dispatch (it
                // only depends on the fill's own position).
                Some(pipe) => {
                    pipe.snapshot_and_dispatch(
                        buf[in_block_begin..in_next].to_vec(),
                        opt.match_cache[..cache_ptr].to_vec(),
                        opt.match_len_freqs.clone(),
                        split_stats.clone(),
                        prev_observations,
                        prev_num_observations,
                        is_first,
                        is_final,
                        *params,
                    );
                    false
                }
                None => crate::anatomy_wall_time!(near_opt_flush_ns, near_opt_flush_calls, {
                    opt.optimize_and_flush(
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
                    )
                }),
            };
            #[cfg(not(feature = "near-opt-parallel-flush"))]
            let prev_used_only_literals =
                crate::anatomy_wall_time!(near_opt_flush_ns, near_opt_flush_calls, {
                    opt.optimize_and_flush(
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
                    )
                });
            prev_block_used_only_literals = prev_used_only_literals;

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

    // LEVER #3 writeback: consume the finished flush buffers IN BLOCK ORDER,
    // joining the pool. The dispatch order plus this in-order repack (with
    // `BitWriter::append_fragment`'s bit-exact pending-byte shifting) is what
    // makes the parallel chunk byte-identical to the serial chunk; the
    // returned flip tally is the runtime guard it applies for later chunks.
    #[cfg(feature = "near-opt-parallel-flush")]
    if let Some(pipe) = flush_pipe.take() {
        pipe.finish_and_write(bw);
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
    // near_optimal -> parse(super) -> deflate(super::super); encode_gzip_bytes_to_vec /
    // encode_deflate_bytes_to_vec live in deflate::mod.
    use super::super::super::{encode_deflate_bytes_to_vec, encode_gzip_bytes_to_vec};
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
            let gz = encode_gzip_bytes_to_vec(&data, level);
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
        let l9 = encode_deflate_bytes_to_vec(&data, 9).len();
        let l12 = encode_deflate_bytes_to_vec(&data, 12).len();
        assert!(
            l12 <= l9,
            "L12 near-optimal ({l12}) worse than L9 lazy2 ({l9})"
        );
    }
}
