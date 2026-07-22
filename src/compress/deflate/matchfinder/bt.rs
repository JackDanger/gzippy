//! Binary-tree Lempel-Ziv matchfinder (near-optimal parsing).
//!
//! Faithful transliteration of libdeflate `vendor/libdeflate/lib/bt_matchfinder.h`.
//! Each hash-4 bucket roots a binary search tree of prior sequences keyed by the
//! bytes at each position; a single traversal at every byte simultaneously
//! (a) collects the matches found and (b) re-roots the tree at the current
//! position using the `best_lt_len` / `best_gt_len` common-prefix lengths
//! (`bt_matchfinder_advance_one_byte`, :140-262). A separate 2-way hash-3 table
//! catches length-3 matches (`:180-202`).
//!
//! The three tables live in ONE contiguous `Box<[i16]>` so the shared
//! `matchfinder_init` / `matchfinder_rebase` sentinel machinery from
//! [`super::common`] operates on it exactly as the C `(mf_pos_t *)mf` cast does:
//! `init` clears only the hash prefix (`BT_MATCHFINDER_TOTAL_HASH_SIZE`,
//! `:104-111`), `slide_window` rebases the whole struct (`:113-119`). Positions
//! are `mf_pos_t` (`i16`, 32 KiB window) relative to a caller-tracked `in_base`;
//! the API carries them as `isize` and truncates to `i16` only at table indices,
//! matching the C `s32`/`mf_pos_t` split.

use super::common::{
    load_u24, load_u32, lz_extend, lz_hash, matchfinder_init, matchfinder_rebase,
    MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE,
};
// `LzMatch` is the shared matchfinder vocabulary type (Stage D, matchfinder/mod.rs) —
// moved to `common` since near_optimal.rs already imports it from this module path;
// re-exported here so that import keeps working unchanged.
pub use super::common::LzMatch;

pub const BT_MATCHFINDER_HASH3_ORDER: u32 = 16;
pub const BT_MATCHFINDER_HASH3_WAYS: usize = 2;
pub const BT_MATCHFINDER_HASH4_ORDER: u32 = 16;

/// Minimum `max_len` for [`BtMatchfinder::get_matches`] / [`BtMatchfinder::skip_byte`]:
/// there must be enough bytes remaining to load a 32-bit word from the *next*
/// position (`BT_MATCHFINDER_REQUIRED_NBYTES`, `:136`).
pub const BT_MATCHFINDER_REQUIRED_NBYTES: u32 = 5;

/// 32 KiB DEFLATE window.
pub const WINDOW_SIZE: usize = MATCHFINDER_WINDOW_SIZE as usize;
const WINDOW_MASK: u32 = (WINDOW_SIZE - 1) as u32;

const HASH3_LEN: usize = (1 << BT_MATCHFINDER_HASH3_ORDER) * BT_MATCHFINDER_HASH3_WAYS;
const HASH4_LEN: usize = 1 << BT_MATCHFINDER_HASH4_ORDER;
const CHILD_LEN: usize = 2 * WINDOW_SIZE;

const HASH4_OFF: usize = HASH3_LEN;
const CHILD_OFF: usize = HASH3_LEN + HASH4_LEN;
/// Length of the initialized hash prefix (`BT_MATCHFINDER_TOTAL_HASH_SIZE / sizeof`).
const TOTAL_HASH_LEN: usize = HASH3_LEN + HASH4_LEN;
const TOTAL_LEN: usize = HASH3_LEN + HASH4_LEN + CHILD_LEN;

/// The binary-tree matchfinder state (~512 KiB).
pub struct BtMatchfinder {
    /// `[hash3_tab | hash4_tab | child_tab]`, contiguous (see module docs).
    tab: Box<[i16]>,
}

impl BtMatchfinder {
    /// `bt_matchfinder_init`: allocate and clear the hash prefix to the sentinel.
    pub fn new() -> Self {
        // Allocate the whole array to the sentinel. child_tab is always written
        // (a node's children are set when it is inserted) before it is read, so
        // its initial value is immaterial; clearing it keeps things deterministic.
        let mut mf = BtMatchfinder {
            tab: vec![MATCHFINDER_INITVAL; TOTAL_LEN].into_boxed_slice(),
        };
        mf.init();
        mf
    }

    /// `bt_matchfinder_init`: clear only the hash tables (not child_tab).
    #[inline]
    pub fn init(&mut self) {
        matchfinder_init(&mut self.tab[..TOTAL_HASH_LEN]);
    }

    /// `bt_matchfinder_slide_window`: rebase every stored position by one window.
    #[inline]
    pub fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.tab);
    }

    #[inline]
    fn left_child_idx(&self, node: i32) -> usize {
        CHILD_OFF + 2 * ((node as u32 & WINDOW_MASK) as usize)
    }

    #[inline]
    fn right_child_idx(&self, node: i32) -> usize {
        CHILD_OFF + 2 * ((node as u32 & WINDOW_MASK) as usize) + 1
    }

    /// Retrieve the matches at `cur_pos` (relative to `in_base`), recording them
    /// into `out` sorted by strictly-increasing length. Returns the count.
    ///
    /// `bt_matchfinder_get_matches` (`:296-315`). `max_len >=
    /// BT_MATCHFINDER_REQUIRED_NBYTES`, `nice_len <= max_len`, `max_depth >= 1`.
    /// `out` must hold at least `nice_len - 2` slots.
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

    /// Advance the matchfinder one byte without recording matches
    /// (`bt_matchfinder_skip_byte`, `:323-340`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn skip_byte(
        &mut self,
        buf: &[u8],
        in_base: usize,
        cur_pos: isize,
        nice_len: u32,
        max_depth: u32,
        next_hashes: &mut [u32; 2],
    ) {
        // C passes max_len = nice_len for the skip path.
        self.advance::<false>(
            buf,
            in_base,
            cur_pos,
            nice_len,
            nice_len,
            max_depth,
            next_hashes,
            &mut [],
        );
    }

    /// Port of `bt_matchfinder_advance_one_byte` (`:140-262`), monomorphized on
    /// `REC` (record_matches).
    ///
    /// ## Soundness invariant (unchecked `tab` indexing — hash/child tables)
    ///
    /// `self.tab` (`TOTAL_LEN` slots) is indexed only three ways below, each
    /// bounded by construction (no external invariant needed, same as the
    /// `hc.rs` Increment 5 hash-table indices):
    ///  * **`h3_base` / `h3_base + 1`.** `hash3 = next_hashes[0] < 2^16`
    ///    (produced by `lz_hash(_, BT_MATCHFINDER_HASH3_ORDER=16)`, or `0` on
    ///    the very first call), so `h3_base = hash3 * 2 < 2 * 2^16 ==
    ///    HASH3_LEN`, and `h3_base` is even, so `h3_base + 1 <= HASH3_LEN - 1`.
    ///  * **`HASH4_OFF + hash4`.** `hash4 = next_hashes[1] < 2^16 == HASH4_LEN`
    ///    (same `lz_hash` construction), so `HASH4_OFF + hash4 < HASH4_OFF +
    ///    HASH4_LEN == TOTAL_HASH_LEN <= TOTAL_LEN`.
    ///  * **`left_child_idx` / `right_child_idx`.** Both are `CHILD_OFF + 2 *
    ///    ((node as u32) & WINDOW_MASK) [+ 1]`; `& WINDOW_MASK` masks into
    ///    `0..WINDOW_SIZE` unconditionally, so the result is always `<
    ///    CHILD_OFF + CHILD_LEN == TOTAL_LEN`, for ANY `node` value (the
    ///    functions are pure modular arithmetic, no caller invariant needed).
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
        let mut depth_remaining = max_depth;
        let cutoff: i32 = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE;
        let mut n_out = 0usize;
        let mut best_len: u32 = 3;

        // SAFETY (word loads): the caller contract is `max_len >=
        // BT_MATCHFINDER_REQUIRED_NBYTES (5)` and (from `adjust_max_and_nice_len`
        // in the parse driver) `in_next + max_len <= in_end <= buf.len() -
        // BUF_PAD`, so `in_next + 5 <= buf.len()` always — the same bound
        // `hc.rs`'s Increment 5 relies on for its next-hash read.
        debug_assert!(in_next + 1 + 4 <= buf.len());
        let next_hashseq = unsafe { load_u32(buf.as_ptr(), in_next + 1) };
        let hash3 = next_hashes[0] as usize;
        let hash4 = next_hashes[1] as usize;
        next_hashes[0] = lz_hash(next_hashseq & 0x00FF_FFFF, BT_MATCHFINDER_HASH3_ORDER);
        next_hashes[1] = lz_hash(next_hashseq, BT_MATCHFINDER_HASH4_ORDER);

        // ---- hash3: 2-way LRU, records at most one length-3 match ----
        let h3_base = hash3 * BT_MATCHFINDER_HASH3_WAYS;
        // SAFETY: see the soundness invariant above (`h3_base`/`h3_base+1` bound).
        debug_assert!(h3_base + 1 < self.tab.len());
        let cur_node3 = unsafe { *self.tab.get_unchecked(h3_base) as i32 };
        unsafe {
            *self.tab.get_unchecked_mut(h3_base) = cur_pos as i16;
        }
        let cur_node3_2 = unsafe { *self.tab.get_unchecked(h3_base + 1) as i32 };
        unsafe {
            *self.tab.get_unchecked_mut(h3_base + 1) = cur_node3 as i16;
        }

        if REC && cur_node3 > cutoff {
            // SAFETY: `in_next + 4 <= buf.len()` (word-load bound above, since
            // `4 < 5`). `mp`/`mp2` are earlier tree-node positions (`< in_next`,
            // the same "stored position is always earlier" matchfinder
            // invariant `hc.rs` relies on for its own candidate reads), so
            // `mp + 4 < in_next + 4 <= buf.len()` (same for `mp2`).
            debug_assert!(in_next + 4 <= buf.len());
            let seq3 = unsafe { load_u24(buf.as_ptr(), in_next) };
            let mp = (in_base as isize + cur_node3 as isize) as usize;
            debug_assert!(mp < in_next && mp + 4 <= buf.len());
            if seq3 == unsafe { load_u24(buf.as_ptr(), mp) } {
                out[n_out] = LzMatch {
                    length: 3,
                    offset: (in_next - mp) as u16,
                };
                n_out += 1;
            } else if cur_node3_2 > cutoff {
                let mp2 = (in_base as isize + cur_node3_2 as isize) as usize;
                debug_assert!(mp2 < in_next && mp2 + 4 <= buf.len());
                if seq3 == unsafe { load_u24(buf.as_ptr(), mp2) } {
                    out[n_out] = LzMatch {
                        length: 3,
                        offset: (in_next - mp2) as u16,
                    };
                    n_out += 1;
                }
            }
        }

        // ---- hash4: root of the binary tree for length-4+ matches ----
        // SAFETY: see the soundness invariant above (`HASH4_OFF + hash4` bound).
        debug_assert!(HASH4_OFF + hash4 < self.tab.len());
        let mut cur_node = unsafe { *self.tab.get_unchecked(HASH4_OFF + hash4) as i32 };
        unsafe {
            *self.tab.get_unchecked_mut(HASH4_OFF + hash4) = cur_pos as i16;
        }

        let mut pending_lt = self.left_child_idx(cur_pos as i32);
        let mut pending_gt = self.right_child_idx(cur_pos as i32);

        if cur_node <= cutoff {
            // SAFETY: see the soundness invariant above (child-index bound —
            // pure modular arithmetic, holds for any `node`).
            debug_assert!(pending_lt < self.tab.len() && pending_gt < self.tab.len());
            unsafe {
                *self.tab.get_unchecked_mut(pending_lt) = MATCHFINDER_INITVAL;
                *self.tab.get_unchecked_mut(pending_gt) = MATCHFINDER_INITVAL;
            }
            return n_out;
        }

        let mut best_lt_len: u32 = 0;
        let mut best_gt_len: u32 = 0;
        let mut len: u32 = 0;

        // SAFETY (tree-walk single-byte compares below): `len` only ever
        // decreases or holds (it is reassigned to `min(len, best_lt_len)` /
        // `min(len, best_gt_len)`, and both of those were themselves earlier
        // values of `len`), and it starts at `0` then is only ever set from
        // `lz_extend`'s return value, which is bounded by `max_len` (its own
        // documented contract). So `len <= max_len` holds on every iteration,
        // hence `in_next + len <= in_next + max_len <= buf.len()` (the
        // word-load bound above). `matchptr = in_base + cur_node` is always a
        // STRICTLY EARLIER tree position than `in_next = in_base + cur_pos`
        // (the standard matchfinder invariant: a stored node was inserted at
        // an earlier position than the one now querying it, or it fails the
        // `cutoff`/`MATCHFINDER_INITVAL` gate before reaching this loop — the
        // same invariant `hc.rs` documents for its own candidate reads), so
        // `matchptr + len < in_next + len <= buf.len()`.
        loop {
            let matchptr = (in_base as isize + cur_node as isize) as usize;
            debug_assert!(matchptr < in_next && matchptr + (len as usize) < buf.len());
            debug_assert!(in_next + (len as usize) <= buf.len());

            if unsafe {
                *buf.get_unchecked(matchptr + len as usize)
                    == *buf.get_unchecked(in_next + len as usize)
            } {
                len = lz_extend(buf, in_next, matchptr, len + 1, max_len);
                if !REC || len > best_len {
                    if REC {
                        best_len = len;
                        out[n_out] = LzMatch {
                            length: len as u16,
                            offset: (in_next - matchptr) as u16,
                        };
                        n_out += 1;
                    }
                    if len >= nice_len {
                        // Re-root: the current node's subtrees become pending.
                        // SAFETY: see the soundness invariant above (child-index
                        // bound, pure modular arithmetic).
                        let lc_idx = self.left_child_idx(cur_node);
                        let rc_idx = self.right_child_idx(cur_node);
                        debug_assert!(
                            lc_idx < self.tab.len()
                                && rc_idx < self.tab.len()
                                && pending_lt < self.tab.len()
                                && pending_gt < self.tab.len()
                        );
                        unsafe {
                            let lc = *self.tab.get_unchecked(lc_idx);
                            let rc = *self.tab.get_unchecked(rc_idx);
                            *self.tab.get_unchecked_mut(pending_lt) = lc;
                            *self.tab.get_unchecked_mut(pending_gt) = rc;
                        }
                        return n_out;
                    }
                }
            }

            // SAFETY: same bound as the equality compare above (`len` hasn't
            // changed since, or was just set from `lz_extend`, still `<= max_len`).
            debug_assert!(
                matchptr + (len as usize) < buf.len() && in_next + (len as usize) <= buf.len()
            );
            if unsafe {
                *buf.get_unchecked(matchptr + len as usize)
                    < *buf.get_unchecked(in_next + len as usize)
            } {
                // SAFETY: see the soundness invariant above (child-index bound,
                // pure modular arithmetic — holds for any `cur_node`/`pending_lt`).
                debug_assert!(pending_lt < self.tab.len());
                unsafe {
                    *self.tab.get_unchecked_mut(pending_lt) = cur_node as i16;
                }
                pending_lt = self.right_child_idx(cur_node);
                debug_assert!(pending_lt < self.tab.len());
                cur_node = unsafe { *self.tab.get_unchecked(pending_lt) as i32 };
                best_lt_len = len;
                if best_gt_len < len {
                    len = best_gt_len;
                }
            } else {
                debug_assert!(pending_gt < self.tab.len());
                unsafe {
                    *self.tab.get_unchecked_mut(pending_gt) = cur_node as i16;
                }
                pending_gt = self.left_child_idx(cur_node);
                debug_assert!(pending_gt < self.tab.len());
                cur_node = unsafe { *self.tab.get_unchecked(pending_gt) as i32 };
                best_gt_len = len;
                if best_lt_len < len {
                    len = best_lt_len;
                }
            }

            depth_remaining -= 1;
            if cur_node <= cutoff || depth_remaining == 0 {
                debug_assert!(pending_lt < self.tab.len() && pending_gt < self.tab.len());
                unsafe {
                    *self.tab.get_unchecked_mut(pending_lt) = MATCHFINDER_INITVAL;
                    *self.tab.get_unchecked_mut(pending_gt) = MATCHFINDER_INITVAL;
                }
                return n_out;
            }
        }
    }
}

impl Default for BtMatchfinder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A padded working buffer, mirroring what the parser hands the matchfinder.
    fn padded(data: &[u8]) -> Vec<u8> {
        let mut b = data.to_vec();
        b.extend_from_slice(&[0u8; 16]);
        b
    }

    /// Brute-force: for every candidate offset, the longest match length at
    /// `pos` (min length 3), returning the (length, offset) with the SMALLEST
    /// offset achieving the maximum length — the oracle for "a real match
    /// exists".
    fn brute_longest(data: &[u8], pos: usize, max_len: usize) -> Option<(usize, usize)> {
        let mut best_len = 0usize;
        let mut best_off = 0usize;
        let lo = pos.saturating_sub(WINDOW_SIZE);
        for start in lo..pos {
            let mut l = 0usize;
            while l < max_len && pos + l < data.len() && data[start + l] == data[pos + l] {
                l += 1;
            }
            if l >= 3 && l > best_len {
                best_len = l;
                best_off = pos - start;
            }
        }
        if best_len >= 3 {
            Some((best_len, best_off))
        } else {
            None
        }
    }

    /// Drive the matchfinder across `data` from position 0, returning the match
    /// lists recorded at each position.
    fn run_all(data: &[u8], nice_len: u32, max_depth: u32) -> Vec<Vec<LzMatch>> {
        let buf = padded(data);
        let in_end = data.len();
        let mut mf = BtMatchfinder::new();
        let mut next_hashes = [0u32; 2];
        let mut out = vec![LzMatch::default(); 260];
        let mut result = Vec::with_capacity(in_end);
        let in_base = 0usize;
        for pos in 0..in_end {
            let remaining = in_end - pos;
            let mut max_len = 258u32.min(remaining as u32);
            let nl = nice_len.min(max_len);
            if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES {
                let n = mf.get_matches(
                    &buf,
                    in_base,
                    pos as isize,
                    max_len,
                    nl,
                    max_depth,
                    &mut next_hashes,
                    &mut out,
                );
                result.push(out[..n].to_vec());
            } else {
                // Not enough bytes to search; still advance the tree so hashes
                // stay consistent is impossible here (needs 5 bytes), so record
                // an empty list.
                let _ = &mut max_len;
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
        let per_pos = run_all(&data, 258, 50);
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
                // offsets are non-strictly increasing with length
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
                // The referenced bytes must actually equal the source bytes.
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
    fn cross_check_against_brute_force_longest() {
        // Two correctness invariants against an exhaustive O(n^2) oracle:
        //
        //  (1) UPPER BOUND: bt can never report a match longer than the
        //      brute-force longest (brute searches every offset in the window),
        //      so any bt overshoot is a real bug.
        //  (2) QUALITY: with full depth/nice_len bt must find the brute-force
        //      longest at the VAST majority of positions. It legitimately misses
        //      a few due to libdeflate's documented behaviors — position 0 is
        //      misfiled into the {0,0} bootstrap bucket, hash-4 collisions, and
        //      positions covered by a preceding long match are never searched.
        let mut data = Vec::new();
        let seed: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".to_vec();
        for i in 0..300 {
            data.extend_from_slice(&seed);
            if i % 5 == 0 {
                data.push((i & 0xff) as u8);
            }
        }
        let per_pos = run_all(&data, 258, 1000);
        let mut brute_hits = 0usize;
        let mut bt_matched_brute = 0usize;
        for (pos, matches) in per_pos.iter().enumerate() {
            let remaining = data.len() - pos;
            let max_len = 258usize.min(remaining);
            let brute = brute_longest(&data, pos, max_len);
            let bt_best = matches.iter().map(|m| m.length as usize).max().unwrap_or(0);
            if let Some((blen, _)) = brute {
                // (1) upper bound
                assert!(
                    bt_best <= blen,
                    "pos {pos}: bt reported {bt_best} > brute longest {blen}"
                );
                brute_hits += 1;
                if bt_best == blen {
                    bt_matched_brute += 1;
                }
            } else {
                assert_eq!(
                    bt_best, 0,
                    "pos {pos}: bt found a match where brute found none"
                );
            }
        }
        // (2) quality: >90% of brute matches are matched exactly by bt.
        assert!(
            bt_matched_brute * 100 >= brute_hits * 90,
            "bt matched brute at only {bt_matched_brute}/{brute_hits} positions (<90%)"
        );
    }

    #[test]
    fn no_matches_on_unique_data() {
        let data: Vec<u8> = (0..4000u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        let per_pos = run_all(&data, 258, 100);
        // Some spurious short matches may exist by chance, but verify each is a
        // real back-reference (covered by offsets_are_valid too); mainly ensure
        // no panics and lengths never exceed remaining.
        for (pos, matches) in per_pos.iter().enumerate() {
            for m in matches {
                assert!(m.length as usize <= data.len() - pos);
            }
        }
    }
}
