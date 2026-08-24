//! C: `vendor/libdeflate/lib/hc_matchfinder.h` — Lempel-Ziv matchfinding with a hash
//! table + linked lists (the "hash chains" matchfinder). Drives levels 2-9.
//!
//! From the C's header comment: the code uses one loop for finding the first match and
//! one loop for finding a longer match — each tuned for its task, and in combination
//! faster than a single generalized loop. The inner loop only compares the last and
//! first bytes of a potential match; a full extension is attempted only when those
//! match.
//!
//! # `init` clears the HASH TABLES ONLY; `slide_window` rebases EVERYTHING
//!
//! `hc_matchfinder_init` passes `HC_MATCHFINDER_TOTAL_HASH_SIZE` — hash3 + hash4 —
//! while `hc_matchfinder_slide_window` passes `sizeof(*mf)`, which additionally covers
//! `next_tab`. That asymmetry is deliberate and correct: `next_tab[pos]` is only ever
//! read after `hash4_tab` has led there, and `hash4_tab` starts all-out-of-window, so
//! stale link entries are unreachable until they are overwritten. Initialising
//! `next_tab` too would be 32768 extra stores per stream for nothing. Reproduced
//! exactly — an "obvious" symmetry fix here is pure cost.
//!
//! # Our shipping `deflate/matchfinder/hc.rs` IS a faithful port of this
//!
//! Unlike `ht.rs`, a function-by-function audit found `hc.rs` semantically EXACT, with
//! zero output-affecting divergence — consistent with L2 and L4-L9 being byte ties
//! today. This module exists so the whole `ldx` path is one lineage and the
//! differential can be run per level, not because `hc.rs` was suspected.

use super::matchfinder_common::{
    lz_extend, lz_hash, matchfinder_init, matchfinder_rebase, prefetchw, MfPos,
    MATCHFINDER_WINDOW_SIZE,
};

/// C: `#define HC_MATCHFINDER_HASH3_ORDER 15` (:110)
pub(crate) const HC_MATCHFINDER_HASH3_ORDER: u32 = 15;
/// C: `#define HC_MATCHFINDER_HASH4_ORDER 16` (:111)
pub(crate) const HC_MATCHFINDER_HASH4_ORDER: u32 = 16;

const HASH3_LEN: usize = 1 << HC_MATCHFINDER_HASH3_ORDER;
const HASH4_LEN: usize = 1 << HC_MATCHFINDER_HASH4_ORDER;
const NEXT_LEN: usize = MATCHFINDER_WINDOW_SIZE as usize;
const HASH_TABLE_LEN: usize = HASH3_LEN + HASH4_LEN;
const HC_TABLE_LEN: usize = HASH_TABLE_LEN + NEXT_LEN;

/// C: `struct MATCHFINDER_ALIGNED hc_matchfinder`.
///
/// The three tables must remain one object.  Besides allowing the one-pass C-shaped
/// init/rebase operations below, this makes their addresses fixed offsets from one
/// aligned base in the matchfinder's hot probe loop.  Three independent boxed slices
/// are semantically equivalent but force three unrelated base loads there.
#[repr(C, align(32))]
struct HcTables {
    hash3_tab: [MfPos; HASH3_LEN],
    hash4_tab: [MfPos; HASH4_LEN],
    next_tab: [MfPos; NEXT_LEN],
}

const _: () = assert!(core::mem::align_of::<HcTables>() >= 32);
const _: () =
    assert!(core::mem::size_of::<HcTables>() == HC_TABLE_LEN * core::mem::size_of::<MfPos>());
const _: () = assert!(core::mem::size_of::<HcTables>().is_multiple_of(1024));

impl HcTables {
    #[inline(always)]
    fn hash_tables_mut(&mut self) -> &mut [MfPos] {
        // `repr(C)` keeps hash3 and hash4 adjacent, exactly as C's
        // HC_MATCHFINDER_TOTAL_HASH_SIZE range requires.
        unsafe {
            core::slice::from_raw_parts_mut((self as *mut Self).cast::<MfPos>(), HASH_TABLE_LEN)
        }
    }

    #[inline(always)]
    fn all_mut(&mut self) -> &mut [MfPos] {
        unsafe {
            core::slice::from_raw_parts_mut((self as *mut Self).cast::<MfPos>(), HC_TABLE_LEN)
        }
    }
}

/// C: `struct hc_matchfinder` (:119)
pub(crate) struct HcMatchfinder {
    tables: Box<HcTables>,
}

impl HcMatchfinder {
    pub(crate) fn new() -> Self {
        let mut mf = Self {
            tables: Box::new(HcTables {
                hash3_tab: [0; HASH3_LEN],
                hash4_tab: [0; HASH4_LEN],
                next_tab: [0; NEXT_LEN],
            }),
        };
        mf.init();
        mf
    }

    /// C: `hc_matchfinder_init(struct hc_matchfinder *mf)` (:135)
    ///
    /// Prepare the matchfinder for a new input buffer. **Hash tables only** — see the
    /// module docs for why `next_tab` is deliberately left alone.
    pub(crate) fn init(&mut self) {
        matchfinder_init(self.tables.hash_tables_mut());
    }

    /// C: `hc_matchfinder_slide_window(struct hc_matchfinder *mf)` (:144)
    ///
    /// Rebases ALL THREE arrays, including `next_tab` — the links are positions too.
    #[inline(always)]
    pub(crate) fn slide_window(&mut self) {
        matchfinder_rebase(self.tables.all_mut());
    }
}

/// C: `static forceinline` (`hc_matchfinder.h` / `ht_matchfinder.h`). Ours carried
/// NO inline attribute, so this was a real ABI call — and it takes 10 arguments,
/// past AArch64's 8 argument registers, so every call spilled to the stack.
/// Measured: the deficit vs the C is call-shape-dependent — at L9 (depth 600,
/// few long calls) we BEAT it 0.88x, at L2 (depth 6, many short calls) we lose
/// 1.34x. Matching the vendor's `forceinline`.
/// C: `hc_matchfinder_longest_match(...)` (:181)
///
/// Find the longest match longer than `best_len` bytes. Returns the length of the
/// match found, or `best_len` if no longer match was found.
///
/// # `best_len` is an INPUT, and the caller uses it as a min-match filter
///
/// `deflate_compress_greedy` passes `min_len - 1`, so the matchfinder only reports a
/// match that beats the minimum. The three-way branch on entry (`best_len < 3`,
/// `best_len < 4`, else) is how that filter turns into skipped work rather than a
/// post-hoc rejection.
///
/// # The `& (MATCHFINDER_WINDOW_SIZE - 1)` on every link read
///
/// `cur_node4` is a signed position that may be negative; the mask makes it a valid
/// index into `next_tab` by wrapping modulo the window. It is NOT a bounds check —
/// the `cur_node4 <= cutoff` test before each dereference is what guarantees the entry
/// is live. Dropping the mask indexes out of bounds; dropping the cutoff test reads a
/// stale link.

/// Per-call chain-walk accumulator, mirroring
/// `compress::deflate::matchfinder::hc::HcLocalCounters` event for event so the
/// anatomy pins hold when a level is routed to the port. Batched in locals and
/// flushed once per call: an atomic at every visited node costs 10-14% wall at
/// L6-L9, which is why the legacy instrument batches too.
#[cfg(feature = "anatomy-counters")]
#[derive(Default)]
struct LdxHcCounters {
    attempts: u64,
    too_short: u64,
    accepted: u64,
    chain_reads: u64,
}

#[cfg(feature = "anatomy-counters")]
impl LdxHcCounters {
    #[inline(always)]
    fn flush(self) {
        crate::anatomy_count!(hc_probe_attempts, self.attempts);
        // A probe that neither advanced `best_len` nor was rejected as too short
        // MISSED the 4-byte test. Deriving keeps the hot path to one increment per
        // event, exactly as the legacy accumulator does.
        crate::anatomy_count!(
            hc_probe_outcome_miss,
            self.attempts
                .saturating_sub(self.accepted)
                .saturating_sub(self.too_short)
        );
        crate::anatomy_count!(hc_probe_outcome_too_short, self.too_short);
        crate::anatomy_count!(hc_probe_outcome_accepted, self.accepted);
        crate::anatomy_count!(hc_chain_table_reads, self.chain_reads);
    }
}

/// No-op stand-in so the walk reads identically with the feature off.
#[cfg(not(feature = "anatomy-counters"))]
struct LdxHcCounters;
#[cfg(not(feature = "anatomy-counters"))]
impl LdxHcCounters {
    #[inline(always)]
    fn default() -> Self {
        LdxHcCounters
    }
    #[inline(always)]
    fn flush(self) {}
}

#[cfg(not(feature = "anatomy-counters"))]
macro_rules! bump {
    ($l:ident . $f:ident) => {{
        let _ = &$l;
    }};
}
#[cfg(feature = "anatomy-counters")]
macro_rules! bump {
    ($l:ident . $f:ident) => {
        $l.$f += 1
    };
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn hc_matchfinder_longest_match(
    mf: &mut HcMatchfinder,
    buf: &[u8],
    in_base: &mut usize,
    in_next: usize,
    mut best_len: u32,
    max_len: u32,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
    offset_ret: &mut u32,
) -> u32 {
    #[allow(unused_mut)]
    let mut local = LdxHcCounters::default();
    let mut depth_remaining = max_search_depth;
    let mut best_matchptr: usize = in_next;
    let mut cur_pos = (in_next - *in_base) as i32;

    if cur_pos == MATCHFINDER_WINDOW_SIZE {
        mf.slide_window();
        *in_base += MATCHFINDER_WINDOW_SIZE as usize;
        cur_pos = 0;
    }

    let in_base_v = *in_base;
    let cutoff: MfPos = (cur_pos - MATCHFINDER_WINDOW_SIZE) as MfPos;

    // Can we read 4 bytes from 'in_next + 1'?
    if max_len < 5 {
        *offset_ret = (in_next - best_matchptr) as u32;
        {
            local.flush();
            return best_len;
        }
    }

    // Get the precomputed hash codes.
    let hash3 = next_hashes[0] as usize;
    let hash4 = next_hashes[1] as usize;

    // From the hash buckets, get the first node of each linked list.
    debug_assert!(hash3 < mf.tables.hash3_tab.len() && hash4 < mf.tables.hash4_tab.len());
    let (cur_node3, mut cur_node4) = unsafe {
        (
            *mf.tables.hash3_tab.get_unchecked(hash3),
            *mf.tables.hash4_tab.get_unchecked(hash4),
        )
    };

    // Update for length 3 matches. This replaces the singleton node in the 'hash3'
    // bucket with the node for the current sequence.
    debug_assert!(hash3 < mf.tables.hash3_tab.len());
    crate::anatomy_count!(hc_head_table_reads, 2u64);
    crate::anatomy_count!(hc_head_table_writes, 2u64);
    unsafe { *mf.tables.hash3_tab.get_unchecked_mut(hash3) = cur_pos as MfPos };

    // Update for length 4 matches. This prepends the node for the current sequence to
    // the linked list in the 'hash4' bucket.
    debug_assert!(
        hash4 < mf.tables.hash4_tab.len() && (cur_pos as usize) < mf.tables.next_tab.len()
    );
    unsafe {
        *mf.tables.hash4_tab.get_unchecked_mut(hash4) = cur_pos as MfPos;
        *mf.tables.next_tab.get_unchecked_mut(cur_pos as usize) = cur_node4;
    }

    // Compute the next hash codes.
    let next_hashseq = load_u32(buf, in_next + 1);
    crate::anatomy_count!(hc_hash_computations);
    next_hashes[0] = lz_hash(next_hashseq & 0xFF_FFFF, HC_MATCHFINDER_HASH3_ORDER);
    next_hashes[1] = lz_hash(next_hashseq, HC_MATCHFINDER_HASH4_ORDER);
    prefetchw(unsafe { mf.tables.hash3_tab.as_ptr().add(next_hashes[0] as usize) });
    prefetchw(unsafe { mf.tables.hash4_tab.as_ptr().add(next_hashes[1] as usize) });

    let mut matchptr: usize;

    if best_len < 4 {
        // No match of length >= 4 found yet.

        // Check for a length 3 match if needed.
        if cur_node3 <= cutoff {
            *offset_ret = (in_next - best_matchptr) as u32;
            {
                local.flush();
                return best_len;
            }
        }

        let seq4 = load_u32(buf, in_next);

        if best_len < 3 {
            matchptr = node_ptr(in_base_v, cur_node3);
            if load_u24(buf, matchptr) == loaded_u32_to_u24(seq4) {
                best_len = 3;
                best_matchptr = matchptr;
            }
        }

        // Check for a length 4 match.
        if cur_node4 <= cutoff {
            *offset_ret = (in_next - best_matchptr) as u32;
            {
                local.flush();
                return best_len;
            }
        }

        loop {
            // No length 4 match found yet. Check the first 4 bytes.
            matchptr = node_ptr(in_base_v, cur_node4);

            bump!(local.attempts);
            if load_u32(buf, matchptr) == seq4 {
                break;
            }

            // The first 4 bytes did not match. Keep trying.
            // CHAIN WALK: masked by `MATCHFINDER_WINDOW_SIZE - 1` on a table whose len IS
            // MATCHFINDER_WINDOW_SIZE (power of two) — the bounds check is provably dead.
            // Runs `max_search_depth` times PER POSITION (600 at L9).
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            bump!(local.chain_reads);
            if cutoff_or_exhausted(cur_node4, cutoff, &mut depth_remaining) {
                *offset_ret = (in_next - best_matchptr) as u32;
                {
                    local.flush();
                    return best_len;
                }
            }
        }

        // Found a match of length >= 4. Extend it to its full length.
        best_matchptr = matchptr;
        bump!(local.accepted);
        best_len = lz_extend(buf, in_next, best_matchptr, 4, max_len);
        if best_len >= nice_len {
            *offset_ret = (in_next - best_matchptr) as u32;
            {
                local.flush();
                return best_len;
            }
        }
        // CHAIN WALK: masked by `MATCHFINDER_WINDOW_SIZE - 1` on a table whose len IS
        // MATCHFINDER_WINDOW_SIZE (power of two) — the bounds check is provably dead.
        // Runs `max_search_depth` times PER POSITION (600 at L9).
        let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
        debug_assert!(ni < mf.tables.next_tab.len());
        cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
        bump!(local.chain_reads);
        if cutoff_or_exhausted(cur_node4, cutoff, &mut depth_remaining) {
            *offset_ret = (in_next - best_matchptr) as u32;
            {
                local.flush();
                return best_len;
            }
        }
    } else if cur_node4 <= cutoff || best_len >= nice_len {
        *offset_ret = (in_next - best_matchptr) as u32;
        {
            local.flush();
            return best_len;
        }
    }

    // Check for matches of length >= 5.
    loop {
        loop {
            matchptr = node_ptr(in_base_v, cur_node4);

            // Already found a length 4 match. Try for a longer match; start by
            // checking either the last 4 bytes and the first 4 bytes, or the last
            // byte. (The last byte, the one which would extend the match length by 1,
            // is the most important.)
            bump!(local.attempts);
            if load_u32(buf, matchptr + best_len as usize - 3)
                == load_u32(buf, in_next + best_len as usize - 3)
                && load_u32(buf, matchptr) == load_u32(buf, in_next)
            {
                break;
            }

            // Continue to the next node in the list.
            // CHAIN WALK: masked by `MATCHFINDER_WINDOW_SIZE - 1` on a table whose len IS
            // MATCHFINDER_WINDOW_SIZE (power of two) — the bounds check is provably dead.
            // Runs `max_search_depth` times PER POSITION (600 at L9).
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            bump!(local.chain_reads);
            if cutoff_or_exhausted(cur_node4, cutoff, &mut depth_remaining) {
                *offset_ret = (in_next - best_matchptr) as u32;
                {
                    local.flush();
                    return best_len;
                }
            }
        }

        // UNALIGNED_ACCESS_IS_FAST: the 4-byte prefix was just re-verified above, so
        // the extension may start at 4 rather than 0.
        let len = lz_extend(buf, in_next, matchptr, 4, max_len);
        if len <= best_len {
            // Passed the 4-byte test but did not beat the incumbent — the legacy
            // accumulator calls this outcome `too_short`.
            bump!(local.too_short);
        }
        if len > best_len {
            bump!(local.accepted);
            // This is the new longest match.
            best_len = len;
            best_matchptr = matchptr;
            if best_len >= nice_len {
                *offset_ret = (in_next - best_matchptr) as u32;
                {
                    local.flush();
                    return best_len;
                }
            }
        }

        // Continue to the next node in the list.
        // CHAIN WALK: masked by `MATCHFINDER_WINDOW_SIZE - 1` on a table whose len IS
        // MATCHFINDER_WINDOW_SIZE (power of two) — the bounds check is provably dead.
        // Runs `max_search_depth` times PER POSITION (600 at L9).
        let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
        debug_assert!(ni < mf.tables.next_tab.len());
        cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
        bump!(local.chain_reads);
        if cutoff_or_exhausted(cur_node4, cutoff, &mut depth_remaining) {
            *offset_ret = (in_next - best_matchptr) as u32;
            {
                local.flush();
                return best_len;
            }
        }
    }
}

/// C: `static forceinline` (`hc_matchfinder.h` / `ht_matchfinder.h`). Ours carried
/// NO inline attribute, so this was a real ABI call — and it takes 10 arguments,
/// past AArch64's 8 argument registers, so every call spilled to the stack.
/// Measured: the deficit vs the C is call-shape-dependent — at L9 (depth 600,
/// few long calls) we BEAT it 0.88x, at L2 (depth 6, many short calls) we lose
/// 1.34x. Matching the vendor's `forceinline`.
#[inline(always)]
/// C: `hc_matchfinder_skip_bytes(...)` (:360)
///
/// Advance the matchfinder, but don't search for matches. `count` must be > 0.
///
/// Note this slides the window INSIDE the loop, unlike the ht matchfinder which slides
/// once up front — hc can be asked to skip a whole match length and cross the boundary
/// mid-run.
pub(crate) fn hc_matchfinder_skip_bytes(
    mf: &mut HcMatchfinder,
    buf: &[u8],
    in_base: &mut usize,
    in_next: usize,
    in_end: usize,
    count: u32,
    next_hashes: &mut [u32; 2],
) {
    let mut remaining = count;

    if count as usize + 5 > in_end - in_next {
        return;
    }
    crate::anatomy_count!(hc_positions_skipped, count as u64);
    if remaining == 0 {
        // The C's `do { } while (--remaining)` documents `count > 0`; guard rather
        // than wrap.
        return;
    }

    let mut cur_pos = (in_next - *in_base) as i32;
    let mut hash3 = next_hashes[0] as usize;
    let mut hash4 = next_hashes[1] as usize;
    let mut p = in_next;

    loop {
        if cur_pos == MATCHFINDER_WINDOW_SIZE {
            mf.slide_window();
            *in_base += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos = 0;
        }
        // PROVEN-BOUNDS REGION. `lz_hash(seq, n)` is `(seq * K) >> (32 - n)`, so
        // its result is `< 2^n` BY CONSTRUCTION: hash3 < 1<<HASH3_ORDER ==
        // hash3_tab.len(), hash4 < 1<<HASH4_ORDER == hash4_tab.len(). `cur_pos`
        // is reset to 0 at MATCHFINDER_WINDOW_SIZE by the slide check directly
        // above, and next_tab.len() == MATCHFINDER_WINDOW_SIZE.
        //
        // This is the hottest loop at the SHALLOW levels: at L2
        // `max_search_depth = 6`, so almost no chain walking happens and the
        // cost is this insert. Measured on ARM 2026-08-22: L2's deficit vs the C
        // DIVERGES with input size (1.07x at 16 KB -> 1.37x at 12 MB) while L9's
        // CONVERGES and wins (1.07x -> 0.87x) — the deep path is already ahead,
        // the shallow insert is the deficit.
        debug_assert!(hash3 < mf.tables.hash3_tab.len() && hash4 < mf.tables.hash4_tab.len());
        debug_assert!((cur_pos as usize) < mf.tables.next_tab.len());
        unsafe {
            *mf.tables.hash3_tab.get_unchecked_mut(hash3) = cur_pos as MfPos;
            *mf.tables.next_tab.get_unchecked_mut(cur_pos as usize) =
                *mf.tables.hash4_tab.get_unchecked(hash4);
            *mf.tables.hash4_tab.get_unchecked_mut(hash4) = cur_pos as MfPos;
        }

        p += 1;
        let next_hashseq = load_u32(buf, p);
        hash3 = lz_hash(next_hashseq & 0xFF_FFFF, HC_MATCHFINDER_HASH3_ORDER) as usize;
        hash4 = lz_hash(next_hashseq, HC_MATCHFINDER_HASH4_ORDER) as usize;
        cur_pos += 1;

        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    prefetchw(unsafe { mf.tables.hash3_tab.as_ptr().add(hash3) });
    prefetchw(unsafe { mf.tables.hash4_tab.as_ptr().add(hash4) });
    next_hashes[0] = hash3 as u32;
    next_hashes[1] = hash4 as u32;
}

/// C: `if (cur_node4 <= cutoff || !--depth_remaining)`
///
/// **The `||` SHORT-CIRCUITS, so the decrement does not happen when the cutoff test
/// fires.** Decrementing unconditionally and then testing gives the same control flow
/// here — both paths return — but it is a different program, and in Rust it also opens
/// an underflow when `depth_remaining` is already 0. Reproducing the short-circuit
/// keeps the shape and closes that class.
#[inline(always)]
fn cutoff_or_exhausted(cur_node: MfPos, cutoff: MfPos, depth_remaining: &mut u32) -> bool {
    if cur_node <= cutoff {
        return true;
    }
    *depth_remaining -= 1;
    *depth_remaining == 0
}

/// C: `matchptr = &in_base[cur_node]` — SIGNED, see `ht_matchfinder::node_ptr`.
#[inline(always)]
fn node_ptr(in_base: usize, cur_node: MfPos) -> usize {
    (in_base as isize + cur_node as isize) as usize
}

/// C: `loaded_u32_to_u24(u32 v)` (`matchfinder_common.h:21`), little-endian arm.
#[inline(always)]
fn loaded_u32_to_u24(v: u32) -> u32 {
    v & 0xFF_FFFF
}

/// C: `load_u24_unaligned(const u8 *p)` (`matchfinder_common.h:35`).
/// **At least 4 bytes (not 3) must be available at `p`** — the C says so explicitly.
#[inline(always)]
fn load_u24(buf: &[u8], i: usize) -> u32 {
    loaded_u32_to_u24(load_u32(buf, i))
}

#[inline(always)]
fn load_u32(buf: &[u8], i: usize) -> u32 {
    // The C reads 4 bytes unchecked; its callers guarantee the room via
    // HT_MATCHFINDER_REQUIRED_NBYTES / the compressor's BUF_PAD. Our checked
    // form compiled to a never-taken cmp+jcc->panic cluster in the hottest
    // loop, re-reading a stack-spilled `buf.len()` every iteration
    // (attributed 2026-08-11: 57 such clusters = 59% of the port's Ir excess
    // over the C, and this class is ~12M of 16.5M).
    //
    // SAFETY: every caller is inside a region that has already proven at least
    // 4 readable bytes at `i` — the tail-shortfall paths return before
    // reaching here, and the compressor allocates BUF_PAD past the input. The
    // `debug_assert` below makes that contract fail loudly in every debug and
    // test build rather than silently, per the standing rule that an elided
    // bound carries a debug assertion.
    debug_assert!(
        i + 4 <= buf.len(),
        "load_u32 out of range: i={i} len={}",
        buf.len()
    );
    let p = unsafe { buf.as_ptr().add(i) as *const u32 };
    u32::from_le(unsafe { p.read_unaligned() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every match reported must be REAL: the bytes at the reported offset must equal
    /// the bytes at the current position for the reported length, and the offset must
    /// be inside the 32 KiB window. A phantom match is the only failure mode that
    /// produces a corrupt stream.
    #[test]
    fn every_reported_match_is_real_and_in_window() {
        let mut state: u32 = 0x5EED_1234;
        let mut buf: Vec<u8> = Vec::with_capacity(250_000);
        while buf.len() < 250_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            if r < 110 && buf.len() > 2000 {
                let off = 3 + (r as usize * 13) % 1800;
                let start = buf.len() - off;
                let n = 3 + (r as usize % 50);
                for k in 0..n {
                    let b = buf[start + k];
                    buf.push(b);
                }
            } else {
                buf.push((r % 70) as u8);
            }
        }
        buf.extend_from_slice(&[0u8; 16]);

        let mut mf = HcMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hashes = [0u32; 2];
        let in_end = buf.len() - 16;

        let mut pos = 0usize;
        let mut matches = 0usize;
        let mut slides = 0usize;
        let mut last_base = 0usize;

        while pos + 16 < in_end {
            let remaining = (in_end - pos) as u32;
            if remaining < 5 {
                break;
            }
            let max_len = core::cmp::min(258, remaining);
            let mut offset = 0u32;
            let len = hc_matchfinder_longest_match(
                &mut mf,
                &buf,
                &mut in_base,
                pos,
                2, // best_len = min_len - 1 for min_len 3
                max_len,
                core::cmp::min(32, max_len),
                16,
                &mut next_hashes,
                &mut offset,
            );
            if in_base != last_base {
                slides += 1;
                last_base = in_base;
            }

            if len > 2 {
                matches += 1;
                assert!(
                    offset >= 1 && (offset as usize) <= pos,
                    "pos={pos} offset={offset}"
                );
                assert!(
                    (offset as i32) <= MATCHFINDER_WINDOW_SIZE,
                    "pos={pos}: offset {offset} exceeds the window"
                );
                let src = pos - offset as usize;
                assert_eq!(
                    &buf[src..src + len as usize],
                    &buf[pos..pos + len as usize],
                    "pos={pos} offset={offset} len={len}: phantom match after {slides} slides"
                );
                hc_matchfinder_skip_bytes(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    pos + 1,
                    in_end,
                    len - 1,
                    &mut next_hashes,
                );
                pos += len as usize;
            } else {
                pos += 1;
            }
        }

        assert!(
            matches > 1000,
            "only {matches} matches on deliberately repetitive data — the finder is not \
             working, so the validity checks proved nothing"
        );
        assert!(slides >= 4, "expected several window slides, saw {slides}");
    }

    /// `best_len` is a floor: the finder must never return LESS than it was given, and
    /// must never report a match at all when nothing beats the floor.
    #[test]
    fn best_len_is_a_floor() {
        let mut buf: Vec<u8> = b"abcdefgh".repeat(4000);
        buf.extend_from_slice(&[0u8; 16]);
        let in_end = buf.len() - 16;

        for floor in [0u32, 3, 4, 8, 200] {
            // A FRESH matchfinder per pass. Rewinding `in_next` to 0 while the table
            // still holds positions from the previous pass is something the C never
            // does, and it makes the finder report a "match" AHEAD of the current
            // position. The first draft of this test shared one matchfinder and
            // underflowed `in_next - best_matchptr`.
            let mut mf = HcMatchfinder::new();
            let mut in_base = 0usize;
            let mut next_hashes = [0u32; 2];
            let mut pos = 0usize;
            while pos + 300 < in_end {
                let mut offset = 0u32;
                let len = hc_matchfinder_longest_match(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    pos,
                    floor,
                    258,
                    32,
                    16,
                    &mut next_hashes,
                    &mut offset,
                );
                assert!(len >= floor, "pos={pos} floor={floor}: returned {len}");
                if len > floor {
                    let src = pos - offset as usize;
                    assert_eq!(&buf[src..src + len as usize], &buf[pos..pos + len as usize]);
                }
                pos += 1;
            }
        }
    }

    /// `max_search_depth` must actually bound the work AND be monotone in the answer:
    /// a deeper search can never find a SHORTER match than a shallower one at the same
    /// position with the same table state.
    #[test]
    fn deeper_search_never_finds_a_shorter_match() {
        let unit = b"alpha beta gamma delta epsilon zeta eta theta ";
        let mut buf = Vec::new();
        while buf.len() < 60_000 {
            buf.extend_from_slice(unit);
        }
        buf.extend_from_slice(&[0u8; 16]);
        let in_end = buf.len() - 16;

        let run = |depth: u32| -> Vec<u32> {
            let mut mf = HcMatchfinder::new();
            let mut in_base = 0usize;
            let mut next_hashes = [0u32; 2];
            let mut out = Vec::new();
            let mut pos = 0usize;
            while pos + 300 < in_end {
                let mut offset = 0u32;
                let len = hc_matchfinder_longest_match(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    pos,
                    2,
                    258,
                    258,
                    depth,
                    &mut next_hashes,
                    &mut offset,
                );
                out.push(len);
                pos += 1;
            }
            out
        };

        let shallow = run(2);
        let deep = run(64);
        assert_eq!(shallow.len(), deep.len());
        for (i, (&s, &d)) in shallow.iter().zip(deep.iter()).enumerate() {
            assert!(d >= s, "pos {i}: depth 64 found {d} but depth 2 found {s}");
        }
        assert!(
            deep.iter().sum::<u32>() > shallow.iter().sum::<u32>(),
            "a 32x deeper search found no more total match length — depth is not wired up"
        );
    }

    /// `skip_bytes` must decline when fewer than `count + 5` bytes remain, leaving both
    /// hashes untouched — otherwise it hashes past the end of the input.
    #[test]
    fn skip_bytes_declines_near_the_end_of_input() {
        let buf = vec![3u8; 64];
        let mut mf = HcMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hashes = [0xAAAA_1111u32, 0xBBBB_2222];
        let before = (mf.tables.hash3_tab[0], mf.tables.hash4_tab[0]);

        hc_matchfinder_skip_bytes(&mut mf, &buf, &mut in_base, 54, 64, 8, &mut next_hashes);
        assert_eq!(next_hashes, [0xAAAA_1111u32, 0xBBBB_2222]);
        assert_eq!((mf.tables.hash3_tab[0], mf.tables.hash4_tab[0]), before);
    }

    /// `init` clears the hash tables and deliberately does NOT touch `next_tab`. Pinned
    /// so a later "symmetry fix" — 32768 pointless stores per stream — fails a test.
    #[test]
    fn init_clears_the_hash_tables_but_not_next_tab() {
        let mut mf = HcMatchfinder::new();
        mf.tables.next_tab[0] = 1234;
        mf.tables.hash3_tab[7] = 99;
        mf.tables.hash4_tab[9] = 77;

        mf.init();

        assert_eq!(
            mf.tables.hash3_tab[7],
            super::super::matchfinder_common::MATCHFINDER_INITVAL
        );
        assert_eq!(
            mf.tables.hash4_tab[9],
            super::super::matchfinder_common::MATCHFINDER_INITVAL
        );
        assert_eq!(
            mf.tables.next_tab[0], 1234,
            "C: hc_matchfinder_init passes TOTAL_HASH_SIZE, not sizeof(*mf)"
        );
    }
}
