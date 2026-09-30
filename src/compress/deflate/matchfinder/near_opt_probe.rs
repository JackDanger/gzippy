//! d3 PROBE — cheap ordered candidate fill for the near-optimal
//! parser (feature `near-opt-d3-probe`, DEFAULT OFF; see
//! `docs/board/sprint-2026-09-25.md`, "cheap ordered fill under the same DP").
//! The near-opt fill's wall is
//! where the bt descent lives: `bt_probe_attempts` 7.3-10M and
//! `bt_child_table_writes` 9.6-12.3M per T4-MiB — the descent loop's
//! tree-MAINTENANCE writes (one per visited node, plus the pending re-roots)
//! are paid at every fill/skip byte even when they produce no candidate.
//!
//! ## The probe mechanism (candidate (a): chain-walk fill via hc machinery)
//!
//! Same hash state, CANDIDATE source swapped:
//!   * the hash3 2-way LRU singleton table is bt's own (same reads/writes,
//!     same len-3 recording rules) — carried over unchanged;
//!   * the hash-4 BINARY TREE (root table + `child_tab` descent) becomes an
//!     hc-style linked chain: hash4 head + `next_tab` ring. Inserting a
//!     position costs 2 writes + 1 read instead of the descent's per-step
//!     child writes, and `skip_byte` degenerates to that O(1) insert plus the
//!     rolling-hash update — on repeat-heavy content the interior walk bt pays
//!     a full depth-capped descent for costs the chain a link write;
//!   * the candidate gather walks the chain head first (most recent) up to
//!     the gather walk caps at `CHAIN_WALK_BUDGET` links (≤ `max_depth`; the
//!     mission's own 100-150 "actual read" band), re-extending each same-hash4 candidate
//!     word-at-a-time with [`lz_extend`], recording every length-improving
//!     candidate — the same record-on-improvement policy bt's descent uses.
//!
//! ## The layout contract, guaranteed by construction
//!
//! `find_min_cost_path`'s vendor heuristic trusts the fill layout to be
//! sorted by strictly-increasing length with NON-DECREASING offsets
//! (`near_optimal.rs:9-12`; bt.rs's own unit tests pin both on bt's real
//! output). A chain walk records improvements in walk order, which is recency
//! order — offsets come out arbitrary. The canonical shape is imposed in
//! post: per match LENGTH the entry carries the MINIMUM offset among
//! candidates with at least that extension (a backward running-min over the
//! length-ascending record; a candidate of extension L is also a real match
//! of every length below L, so the rewrite is always a valid match and never
//! a length change). Per length the running min is monotone non-increasing
//! as the eligible set shrinks with rising length — reversed into the
//! emission order that IS "strictly-increasing lengths, non-decreasing
//! offsets", with the LAST entry still the longest match (best_len, read by
//! the fill for the split stats and the skip-interior decision). This is the
//! per-length-minimal-offset list the vendor DP effectively consumes out of
//! a bt layout, produced directly from the walk.
//!
//! ## Chosen from the mission's three candidates, rejections recorded
//!
//!   * (a) chain-walk-limited fill via hc machinery — CHOSEN, mechanically
//!     simplest: the hc insert/walk/slide machinery already exists and is
//!     battle-tested (`hc.rs`), the chain layout kills the child-table write
//!     class entirely, and the layout guarantee needs only the backward
//!     running-min pass (no sort needed: improvements arrive length-ascending).
//!   * (b) two-phase L1-cache-light probe (single-bucket hash then bounded
//!     horizontal walk) — REJECTED: the same chain-walk skeleton with an
//!     extra table to maintain; no simpler layout guarantee, strictly more
//!     state.
//!   * (c) plain reduced-depth gather (bt at depth 100) — REJECTED for THIS
//!     lever: the exact-bt-layout guarantee is free (it IS bt), but it is
//!     lever #1's d1 depth knob re-measured (ledger row 1, FLAT locally), not
//!     a d3 gatherer; the descent and its child-table writes stay.
//!
//! ## Soundness invariant (unchecked indexing — the same class as bt.rs)
//!
//!   * **hash3/hash4 head indices.** `next_hashes` entries are `lz_hash`
//!     outputs at orders 16/16 (`< 2^16`), or `0` on the very first call —
//!     so both `< HASH3_LEN` (2-way pair, even-indexed) and `< HASH4_LEN`.
//!   * **`next_tab` chain index.** `(node as u16 & WINDOW_MASK) < WINDOW_SIZE`
//!     for ANY node value (pure masking), exactly `hc.rs`'s chain walks.
//!   * **buffer reads.** The caller contract (`BT_MATCHFINDER_REQUIRED_NBYTES`
//!     gate in `near_optimal::run` + `adjust_max_and_nice_len` clamping)
//!     guarantees `in_next + 1 + 4 <= buf.len()` for the rolling-hash load
//!     and `matchptr + len < in_next + max_len <= buf.len()` for every
//!     candidate read — the exact bounds bt.rs's `advance` documents. The
//!     cutoff / `MATCHFINDER_INITVAL` gate keeps every walked `node` pointing
//!     at a STRICTLY EARLIER position than `in_next`.

use super::bt::BtMatchfinder;
use super::common::{
    load_u24, load_u32, lz_extend, lz_hash, matchfinder_init, matchfinder_rebase, LzMatch,
    MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE,
};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

pub const HASH3_ORDER: u32 = 16;
pub const HASH4_ORDER: u32 = 16;

/// 32 KiB DEFLATE window (same positional frame as bt).
pub const WINDOW_SIZE: usize = MATCHFINDER_WINDOW_SIZE as usize;
const WINDOW_MASK_RING: u16 = (WINDOW_SIZE - 1) as u16;

/// Two-way LRU len-3 table (bt's dimensions), then hash4 heads, then links.
const HASH3_LEN: usize = (1 << HASH3_ORDER) * 2;
const HASH4_LEN: usize = 1 << HASH4_ORDER;
const NEXT_LEN: usize = WINDOW_SIZE;
const HASH4_OFF: usize = HASH3_LEN;
const NEXT_OFF: usize = HASH3_LEN + HASH4_LEN;
const TOTAL_LEN: usize = NEXT_OFF + NEXT_LEN;

/// Cap on recorded length-improving candidates per position (a local array;
/// realistic counts are single digits — repeat runs hit `nice_len` on the
/// first link). Past the cap the walk keeps tracking `best_len` (the skip
/// semantics stay intact) but stops recording. The fill loop's `out` region
/// always holds `>= nice_len - 2 >= 254` slots, so the cap never binds.
const MAX_CANDIDATES: usize = 64;

/// The probe's own cheapness shape (candidate (a) as the mission states it):
/// the chain walk's link budget is the DP's "actual read" depth band, NOT the
/// production bt descent's depth knob. Ledger row 1 (d1) measured the DP's
/// quality flat from depth 100 on (400 == 100 on this corpus), so the budget
/// offered here is the mission's own 100-150 band — 150 bridges the two. The
/// bt arm keeps the production depth untouched; this caps ONLY the chain's
/// walk (`walk` below: `depth_remaining = max_depth.min(CHAIN_WALK_BUDGET)`).
pub const CHAIN_WALK_BUDGET: u32 = 150;

/// The chain-walk candidate gatherer.
///
/// `[hash3 2-way | hash4 heads | next links]` in one contiguous
/// `Box<[i16]>` under the same sentinel machinery as [`super::bt`].
pub struct ChainGather {
    tab: Box<[i16]>,
}

impl ChainGather {
    pub fn new() -> Self {
        let mut mf = ChainGather {
            tab: vec![MATCHFINDER_INITVAL; TOTAL_LEN].into_boxed_slice(),
        };
        mf.init();
        mf
    }

    /// Clear only the hash heads; the ring entries the chain walk reads are
    /// written (linked) before they are ever visited for that position —
    /// the same precondition bt states for `child_tab` — so their initial
    /// value is immaterial.
    #[inline]
    pub fn init(&mut self) {
        matchfinder_init(&mut self.tab[..HASH3_LEN + HASH4_LEN]);
    }

    /// Rebase the whole slab by one window (same as `bt.slide_window`).
    #[inline]
    pub fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.tab);
    }

    /// bt-`get_matches`-shaped: gather the candidates at `cur_pos` into `out`
    /// in the trusted layout, advancing the chain state. Returns the count.
    ///
    /// `max_len >= BT_MATCHFINDER_REQUIRED_NBYTES`, `nice_len <= max_len`,
    /// `max_depth >= 1`, and `out` holds `>= nice_len - 2` slots — the fill
    /// loop's call sites guarantee all three. `probe_budget`
    /// (`near-opt-bt-probebudget`) is accepted to keep `D3Fill`'s two arms
    /// signature-identical but NOT spent: the chain walker's link budget is
    /// `CHAIN_WALK_BUDGET` (this probe's own d3 cap), not the bt monster's
    /// per-descent knob.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn get_matches(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        max_len: u32,
        nice_len: u32,
        max_depth: u32,
        #[cfg(feature = "near-opt-bt-probebudget")] _probe_budget: u32,
        next_hashes: &mut [u32; 2],
        out: &mut [LzMatch],
    ) -> usize {
        self.advance::<true>(
            buf,
            in_base,
            cur_pos,
            max_len,
            nice_len,
            max_depth,
            next_hashes,
            out,
        )
    }

    /// bt-`skip_byte`-shaped: advance one byte without gathering. The chain
    /// insert is O(1) (head + link), so there is no descent and no depth
    /// budget to spend — the lever-#2 point on the skip-interior path.
    /// `probe_budget` (`near-opt-bt-probebudget`) ignored, same as above.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn skip_byte(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        _nice_len: u32,
        _max_depth: u32,
        #[cfg(feature = "near-opt-bt-probebudget")] _probe_budget: u32,
        next_hashes: &mut [u32; 2],
    ) {
        self.advance::<false>(buf, in_base, cur_pos, 0, 0, 0, next_hashes, &mut []);
    }

    /// The hash3 LRU insert + hash4 chain insert + rolling-hash advance,
    /// shared by both call shapes; `REC` records candidates.
    #[allow(clippy::too_many_arguments)]
    fn advance<const REC: bool>(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        max_len: u32,
        nice_len: u32,
        max_depth: u32,
        next_hashes: &mut [u32; 2],
        out: &mut [LzMatch],
    ) -> usize {
        let in_next = (in_base as isize + cur_pos) as usize;
        let cutoff: i32 = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE;

        // Rolling-hash state — byte-for-byte bt's (`load_u32` at `in_next + 1`,
        // the same `lz_hash` orders bt's tables were seeded with).
        debug_assert!(in_base as isize + cur_pos >= 0);
        debug_assert!(in_next + 1 + 4 <= buf.len());
        let next_hashseq = unsafe { load_u32(buf.as_ptr(), in_next + 1) };
        let hash3 = next_hashes[0] as usize;
        let hash4 = next_hashes[1] as usize;
        next_hashes[0] = lz_hash(next_hashseq & 0x00FF_FFFF, HASH3_ORDER);
        next_hashes[1] = lz_hash(next_hashseq, HASH4_ORDER);

        // ---- hash3: 2-way LRU, at most one length-3 candidate (bt's rules) ----
        let h3_base = hash3 * 2;
        debug_assert!(h3_base + 1 < HASH3_LEN);
        let cur_node3 = unsafe { *self.tab.get_unchecked(h3_base) as i32 };
        let cur_node3_2 = unsafe { *self.tab.get_unchecked(h3_base + 1) as i32 };
        unsafe {
            *self.tab.get_unchecked_mut(h3_base) = cur_pos as i16;
            *self.tab.get_unchecked_mut(h3_base + 1) = cur_node3 as i16;
        }

        let mut cands = [(0u16, 0u16); MAX_CANDIDATES];
        let mut n_cand = 0usize;
        // bt keeps `best_len = 3` unconditionally: recorded candidates must
        // EXCEED it, which every chain candidate does (a same-hash4 hit is at
        // least 4 bytes once the 4-byte word compare passes).
        let mut best_len: u32 = 3;
        // The CURRENT position's own 4-byte word (the candidate compare key —
        // not `next_hashseq`, which is the +1-offset seed for the NEXT call).
        // SAFETY: `in_next + 4 <= buf.len()` (the caller contract, see below).
        let seq4 = unsafe { load_u32(buf.as_ptr(), in_next) };

        if REC && cur_node3 > cutoff {
            debug_assert!(in_next + 4 <= buf.len());
            let seq3 = unsafe { load_u24(buf.as_ptr(), in_next) };
            let mp = (in_base as isize + cur_node3 as isize) as usize;
            debug_assert!(mp < in_next && mp + 4 <= buf.len());
            let way0 = unsafe { seq3 == load_u24(buf.as_ptr(), mp) };
            let way1 = !way0 && cur_node3_2 > cutoff && {
                let mp2 = (in_base as isize + cur_node3_2 as isize) as usize;
                debug_assert!(mp2 < in_next && mp2 + 4 <= buf.len());
                unsafe { seq3 == load_u24(buf.as_ptr(), mp2) }
            };
            if way0 || way1 {
                let off = if way0 {
                    (in_next - mp) as u16
                } else {
                    let mp2 = (in_base as isize + cur_node3_2 as isize) as usize;
                    (in_next - mp2) as u16
                };
                cands[n_cand] = (3, off);
                n_cand += 1;
            }
        }

        // ---- hash4: chain-head insert, then the bounded walk ----
        debug_assert!(HASH4_OFF + hash4 < TOTAL_LEN);
        let first_link = unsafe { *self.tab.get_unchecked(HASH4_OFF + hash4) as i32 };
        debug_assert!(((cur_pos as u16 & WINDOW_MASK_RING) as usize) < NEXT_LEN);
        unsafe {
            *self.tab.get_unchecked_mut(HASH4_OFF + hash4) = cur_pos as i16;
            *self
                .tab
                .get_unchecked_mut(NEXT_OFF + (cur_pos as u16 & WINDOW_MASK_RING) as usize) =
                first_link as i16;
        }

        // SAFETY (candidate loads below): the caller contract is
        // `max_len >= 5` and (from `adjust_max_and_nice_len` in the parse
        // driver) `in_next + max_len <= buf.len()`, so the 4-byte candidate
        // load at `matchptr` is in bounds (`matchptr < in_next` via the
        // cutoff gate); `lz_extend`'s own contract carries
        // `in_next + max_len <= buf.len()` and `matchptr + max_len <=
        // buf.len()`. Every walked `cur_node` points at a STRICTLY EARLIER
        // tree position (cutoff gate) at `<= max_len` extension.
        let mut cur_node = first_link;
        // The probe's cheapness shape: the chain's link budget is the DP's
        // "actual read" depth band (see CHAIN_WALK_BUDGET's doc), NOT the
        // production bt descent's depth knob — the bt arm's depth budget is
        // untouched.
        let mut depth_remaining = max_depth.min(CHAIN_WALK_BUDGET);
        while REC && cur_node > cutoff && depth_remaining > 0 {
            let matchptr = (in_base as isize + cur_node as isize) as usize;
            debug_assert!(matchptr < in_next && matchptr + 4 <= buf.len());
            if unsafe { load_u32(buf.as_ptr(), matchptr) } == seq4 {
                // First-4 word equal: a same-hash4 hit confirmed as a real
                // >=4-byte match; re-extend word-at-a-time from 4.
                let len = lz_extend(buf, in_next, matchptr, 4, max_len);
                if len > best_len {
                    best_len = len;
                    if n_cand < MAX_CANDIDATES {
                        cands[n_cand] = (len as u16, (in_next - matchptr) as u16);
                        n_cand += 1;
                    }
                }
                if best_len >= nice_len {
                    break;
                }
            }
            depth_remaining -= 1;
            cur_node = unsafe {
                *self
                    .tab
                    .get_unchecked(NEXT_OFF + (cur_node as u16 & WINDOW_MASK_RING) as usize)
            } as i32;
        }

        if REC {
            // Canonical layout: backward running-min over the length-ascending
            // record (the module doc's construction argument), then emit.
            let mut best_off = u16::MAX;
            let mut i = n_cand;
            while i > 0 {
                i -= 1;
                if cands[i].1 < best_off {
                    best_off = cands[i].1;
                }
                cands[i].1 = best_off;
            }
            debug_assert!(n_cand <= out.len());
            for (k, c) in cands[..n_cand].iter().enumerate() {
                out[k] = LzMatch {
                    length: c.0,
                    offset: c.1,
                };
            }
            return n_cand;
        }
        0
    }
}

impl Default for ChainGather {
    fn default() -> Self {
        Self::new()
    }
}

// ── the probe selector (`near_optimal::run`'s fill matchfinder) ─────────────

/// Production bt fill vs the d3 chain gather, behind ONE type so
/// `near_optimal::run`'s three call sites stay untouched.
///
/// With `near-opt-d3-probe` on, new picks [`D3Fill::Chain`] unless the
/// measurement surface latched [`set_force_bt`] — the probe's two arms in one
/// binary. Without the feature this type does not exist and the parser uses
/// `BtMatchfinder` directly (the `matchfinder/mod.rs` re-export aliases it
/// back under the same name).
pub enum D3Fill {
    Bt(BtMatchfinder),
    Chain(ChainGather),
}

/// Off switch for the probe within one process (measurement surface, the
/// `near_opt_flush_probe::set_force_serial_flush` precedent): force-bt makes
/// subsequent near-optimal runs bit-for-bit TODAY's production shape, so the
/// probe's side-by-side arms live in one binary. No production call site
/// reads this; no env var gates behaviour.
static FORCE_BT: AtomicBool = AtomicBool::new(false);

/// Restore the production bt fill for subsequent near-optimal runs (the
/// probe's serial arm). See [`FORCE_BT`].
#[allow(dead_code)] // driven by tests/l9_t4_chunk_cost_probe.rs; unused in the binary
pub fn set_force_bt(v: bool) {
    FORCE_BT.store(v, Relaxed);
}

/// Whether the force-bt override is latched (the probe's arm witness).
#[allow(dead_code)] // driven by tests/l9_t4_chunk_cost_probe.rs; unused in the binary
pub fn bt_forced() -> bool {
    FORCE_BT.load(Relaxed)
}

impl D3Fill {
    /// The near-optimal fill matchfinder: the d3 chain gather, unless the
    /// measurement surface forced the production bt descent.
    pub fn new() -> Self {
        if FORCE_BT.load(Relaxed) {
            D3Fill::Bt(BtMatchfinder::new())
        } else {
            D3Fill::Chain(ChainGather::new())
        }
    }

    /// bt-signature delegation (the fill loop's three call sites unchanged).
    ///
    /// `probe_budget` under `near-opt-bt-probebudget` forwards to the bt arm
    /// (the lever's own descent cap) and is IGNORED by the chain arm (the
    /// chain gather carries its own `CHAIN_WALK_BUDGET`; a per-descent cap
    /// there is d1/d3's already-priced territory, not this lever's).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn get_matches(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        max_len: u32,
        nice_len: u32,
        max_depth: u32,
        #[cfg(feature = "near-opt-bt-probebudget")] probe_budget: u32,
        next_hashes: &mut [u32; 2],
        out: &mut [LzMatch],
    ) -> usize {
        match self {
            D3Fill::Bt(mf) => mf.get_matches(
                buf,
                in_base,
                cur_pos,
                max_len,
                nice_len,
                max_depth,
                #[cfg(feature = "near-opt-bt-probebudget")]
                probe_budget,
                next_hashes,
                out,
            ),
            D3Fill::Chain(mf) => mf.get_matches(
                buf,
                in_base,
                cur_pos,
                max_len,
                nice_len,
                max_depth,
                #[cfg(feature = "near-opt-bt-probebudget")]
                probe_budget,
                next_hashes,
                out,
            ),
        }
    }

    /// bt-`skip_byte`-shaped delegation.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn skip_byte(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        nice_len: u32,
        max_depth: u32,
        #[cfg(feature = "near-opt-bt-probebudget")] probe_budget: u32,
        next_hashes: &mut [u32; 2],
    ) {
        match self {
            D3Fill::Bt(mf) => mf.skip_byte(
                buf,
                in_base,
                cur_pos,
                nice_len,
                max_depth,
                #[cfg(feature = "near-opt-bt-probebudget")]
                probe_budget,
                next_hashes,
            ),
            D3Fill::Chain(mf) => mf.skip_byte(
                buf,
                in_base,
                cur_pos,
                nice_len,
                max_depth,
                #[cfg(feature = "near-opt-bt-probebudget")]
                probe_budget,
                next_hashes,
            ),
        }
    }

    /// Window rebase delegation.
    #[inline]
    pub fn slide_window(&mut self) {
        match self {
            D3Fill::Bt(mf) => mf.slide_window(),
            D3Fill::Chain(mf) => mf.slide_window(),
        }
    }
}

impl Default for D3Fill {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::bt::BT_MATCHFINDER_REQUIRED_NBYTES;
    use super::*;

    /// A padded working buffer, mirroring what the parser hands the fill.
    fn padded(data: &[u8]) -> Vec<u8> {
        let mut b = data.to_vec();
        b.extend_from_slice(&[0u8; 16]);
        b
    }

    /// Drive the gatherer across `data` from position 0, returning the match
    /// lists recorded at each position (the bt.rs test-harness shape).
    fn run_all(data: &[u8], nice_len: u32, max_depth: u32) -> Vec<Vec<LzMatch>> {
        let buf = padded(data);
        let in_end = data.len();
        let mut mf = ChainGather::new();
        let mut next_hashes = [0u32; 2];
        let mut out = vec![LzMatch::default(); 260];
        let mut result = Vec::with_capacity(in_end);
        let in_base = 0usize;
        for pos in 0..in_end {
            let remaining = in_end - pos;
            let max_len = 258u32.min(remaining as u32);
            let nl = nice_len.min(max_len);
            if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES {
                let n = mf.get_matches(
                    &buf,
                    in_base,
                    pos as isize,
                    max_len,
                    nl,
                    max_depth,
                    #[cfg(feature = "near-opt-bt-probebudget")]
                    max_depth, // the chain arm ignores it; kept signature-identical
                    &mut next_hashes,
                    &mut out,
                );
                result.push(out[..n].to_vec());
            } else {
                result.push(Vec::new());
            }
        }
        result
    }

    #[test]
    fn matches_are_sorted_increasing_length() {
        let block: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
        let mut data = Vec::new();
        for _ in 0..40 {
            data.extend_from_slice(&block);
        }
        let per_pos = run_all(&data, 258, 100);
        for (pos, matches) in per_pos.iter().enumerate() {
            let mut prev_len = 0u16;
            let mut prev_off = 0u16;
            for m in matches {
                assert!(
                    m.length > prev_len,
                    "pos {pos}: lengths not strictly increasing ({} then {})",
                    prev_len,
                    m.length
                );
                if prev_off != 0 {
                    assert!(
                        m.offset >= prev_off,
                        "pos {pos}: offsets not non-decreasing ({} then {})",
                        prev_off,
                        m.offset
                    );
                }
                prev_len = m.length;
                prev_off = m.offset;
            }
        }
    }

    #[test]
    fn offsets_are_valid_back_references() {
        let block: Vec<u8> = (0..100u32).map(|i| (i.wrapping_mul(97)) as u8).collect();
        let repeated: Vec<u8> = block.iter().cloned().cycle().take(5000).collect();
        let buf = padded(&repeated);
        let per_pos = run_all(&repeated, 258, 100);
        for (pos, matches) in per_pos.iter().enumerate() {
            for m in matches {
                let off = m.offset as usize;
                let len = m.length as usize;
                assert!(
                    (1..=WINDOW_SIZE).contains(&off),
                    "pos {pos}: bad offset {off}"
                );
                assert!(off <= pos, "pos {pos}: offset {off} points before input");
                for i in 0..len {
                    assert_eq!(
                        buf[pos + i],
                        buf[pos - off + i],
                        "pos {pos}: match len {len} off {off} mismatched at {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn last_entry_is_the_longest_match() {
        let mut data = Vec::new();
        let seed = b"the quick brown fox jumps over the lazy dog. ".to_vec();
        for i in 0..300 {
            data.extend_from_slice(&seed);
            if i % 5 == 0 {
                data.push((i & 0xff) as u8);
            }
        }
        let per_pos = run_all(&data, 258, 100);
        for (pos, matches) in per_pos.iter().enumerate() {
            if !matches.is_empty() {
                let max_len = matches.iter().map(|m| m.length).max().unwrap();
                assert_eq!(
                    matches.last().unwrap().length,
                    max_len,
                    "pos {pos}: the LAST entry must be the longest (best_len convention)"
                );
                assert_eq!(
                    matches.len(),
                    matches
                        .iter()
                        .map(|m| m.length)
                        .collect::<std::collections::HashSet<_>>()
                        .len(),
                    "pos {pos}: lengths must be DISTINCT (strictly increasing)"
                );
            }
        }
    }

    #[test]
    fn no_matches_on_unique_data() {
        let data: Vec<u8> = (0..4000u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        let per_pos = run_all(&data, 258, 100);
        for (pos, matches) in per_pos.iter().enumerate() {
            for m in matches {
                assert!(
                    m.length as usize <= data.len() - pos,
                    "pos {pos}: match longer than remaining"
                );
            }
        }
    }

    #[test]
    fn d3_selector_defaults_to_chain_and_force_bt_latches() {
        // Default: chain (the probe is active). Force-bt restores bt.
        assert!(!super::bt_forced());
        assert!(matches!(D3Fill::new(), D3Fill::Chain(_)));
        super::set_force_bt(true);
        assert!(super::bt_forced());
        assert!(matches!(D3Fill::new(), D3Fill::Bt(_)));
        super::set_force_bt(false);
        assert!(matches!(D3Fill::new(), D3Fill::Chain(_)));
    }
}
