//! C: `vendor/libdeflate/lib/hc_matchfinder.h` — Lempel-Ziv matchfinding with a hash
//! table + linked lists (the "hash chains" matchfinder). Drives levels 2-9.
//!
//! From the C's header comment: the code uses one loop for finding the first match and
//! one loop for finding a longer match — each tuned for its task, and in combination
//! faster than a single generalized loop. The inner loop only compares the last and
//! first bytes of a potential match; a full extension is attempted only when those
//! match.
//!
//! # The trigram chain (`next3_tab` + `hash3_chain_depth`) — OUR deviation, vendor-named
//!
//! libdeflate's `hash3_tab` is a SINGLETON: one position per 15-bit trigram key, no
//! chain (`hc_matchfinder.h:112-122`; "Update for length 3 matches" overwrites the
//! head and no successor slot ever exists). gzip and zlib hash THREE bytes into
//! their one big table and chain every insertion (`vendor/gzip/deflate.c:282`
//! `UPDATE_HASH` + `prev[]`), so their L2/L3 walk many length-3 candidates per
//! position. That is the vendor diff behind the board's `dd79_bin6:L2/L3` cells
//! (gzip finds ~51% more of the SHORTEST matches there — aa memo §2.2), and it has
//! an in-tree measured receipt: the legacy engine's own `next3_tab` +
//! `hash3_chain_depth` sweep (`matchfinder/hc.rs`) measured dd79_bin6 L2 −4,469 B at
//! depth 4 and L3 −11,304 B at depth 8 before the port replaced the legacy
//! encoder (`docs/production-paths.md`). This port reproduces the legacy shape:
//! `next3_tab[pos]` holds the PREVIOUS node in the same 15-bit trigram-key chain,
//! `longest_match` walks it from the head while `best_len < 3`, and `skip_bytes`
//! leaves `next3_tab` untouched — the legacy skip loop's exact shape, so the port's
//! hottest loop (the shallow-level insert) gains no store. Stale links are safe the
//! same way `next_tab`'s are (candidates are cutoff-checked before dereference and
//! byte-verified before producing an offset); chains are only armed where the
//! MIN_LEN-3 alphabet stands (`choose_min_match_len` stays adaptive), which is the
//! l2min3 containment: a narrow-alphabet block never lowers `min_len` to 3, so the
//! walk is unreachable on exactly the inputs that killed the forced-min shape.
//! Cross-STREAM staleness is why `next3_tab` joins `init()` (below).
//!
//! # `init` clears the HASH TABLES plus `next3_tab`; `slide_window` rebases EVERYTHING
//!
//! `hc_matchfinder_init` passes `HC_MATCHFINDER_TOTAL_HASH_SIZE` — hash3 + hash4 —
//! while `hc_matchfinder_slide_window` passes `sizeof(*mf)`, which additionally covers
//! `next_tab`. That asymmetry is deliberate and correct: `next_tab[pos]` is only ever
//! read after `hash4_tab` has led there, and `hash4_tab` starts all-out-of-window, so
//! stale link entries are unreachable until they are overwritten. Initialising
//! `next_tab` too would be 32768 extra stores per stream for nothing. Reproduced
//! exactly — an "obvious" symmetry fix here is pure cost.
//!
//! `next3_tab` does NOT share that argument once the chain is armed: the head the
//! walk starts from is fresh, but a head inserted by `skip_bytes` does NOT carry a
//! fresh link, so the walk dereferences `next3_tab[head]` whose slot may hold the
//! value written by a PREVIOUS stream at that index — and a previous stream's
//! positive position can sit greater than this stream's `cur_pos`, i.e. AHEAD of the
//! cursor (a follow-on member of a concatenated lookahead, or smaller buffers where
//! `in_base + value` reads past the input slice). We therefore fill `next3_tab` at
//! `init` — one 64 KiB sentinel pass per stream, the legacy engine's own answer
//! (`hc.rs::reset` fills all four tables) — while `next_tab` keeps the vendor's
//! hash-only init (its links are only reachable through fresh hash4 heads; the
//! singleton trigram never walks a second link at all, which is why libdeflate
//! never needed this).
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
/// hash3 + hash4 + `next3_tab` (the trigram-chain links — ours, see the module docs).
const HASH_AND_CHAIN_TABLE_LEN: usize = HASH_TABLE_LEN + NEXT_LEN;
const HC_TABLE_LEN: usize = HASH_AND_CHAIN_TABLE_LEN + NEXT_LEN;

/// C: `struct MATCHFINDER_ALIGNED hc_matchfinder`.
///
/// The tables must remain one object.  Besides allowing the one-pass C-shaped
/// init/rebase operations below, this makes their addresses fixed offsets from one
/// aligned base in the matchfinder's hot probe loop.  Independent boxed slices
/// are semantically equivalent but force unrelated base loads there.
#[repr(C, align(32))]
struct HcTables {
    hash3_tab: [MfPos; HASH3_LEN],
    hash4_tab: [MfPos; HASH4_LEN],
    /// ours (legacy `matchfinder/hc.rs`'s `next3_tab`): the trigram-chain successor
    /// slot. Filled at `init` (see module docs); rebased with everything.
    next3_tab: [MfPos; NEXT_LEN],
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
    fn hash_and_chain_tables_mut(&mut self) -> &mut [MfPos] {
        // `repr(C)` puts `next3_tab` right after hash4, so hash3 + hash4 + next3 is
        // one contiguous sentinel pass.
        unsafe {
            core::slice::from_raw_parts_mut(
                (self as *mut Self).cast::<MfPos>(),
                HASH_AND_CHAIN_TABLE_LEN,
            )
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
                next3_tab: [0; NEXT_LEN],
                next_tab: [0; NEXT_LEN],
            }),
        };
        mf.init();
        mf
    }

    /// C: `hc_matchfinder_init(struct hc_matchfinder *mf)` (:135)
    ///
    /// Prepare the matchfinder for a new input buffer. **Hash tables plus
    /// `next3_tab`** — see the module docs for why `next_tab` is deliberately left
    /// alone and why `next3_tab` must NOT be.
    pub(crate) fn init(&mut self) {
        matchfinder_init(self.tables.hash_and_chain_tables_mut());
    }

    /// C: `hc_matchfinder_slide_window(struct hc_matchfinder *mf)` (:144)
    ///
    /// Rebases EVERY array, including both link tables — the links are positions too.
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
    good_match: u32,
    // ours (legacy `hash3_chain_depth`): trigram-chain walk depth. 0 keeps the
    // vendor's singleton behaviour bit-for-bit (no `next3_tab` insert, head probe
    // only); N > 0 walks up to N candidates per probe (`matchfinder/hc.rs:374-407`).
    hash3_chain_depth: u32,
    next_hashes: &mut [u32; 2],
    offset_ret: &mut u32,
) -> u32 {
    #[allow(unused_mut)]
    let mut local = LdxHcCounters::default();
    // zlib/gzip/pigz `good_length` rule (deflate_slow: "Do not waste too much time
    // if we have already found a good match"): when the current best is already
    // >= good_match, quarter the chain walk. libdeflate has no such knob; the
    // legacy zlib arm did (and won the L6 cells with it). 0 disables.
    let mut depth_remaining = if good_match != 0 && best_len >= good_match {
        (max_search_depth >> 2).max(1)
    } else {
        max_search_depth
    };
    // C deliberately carries the current position as `u32`.  The table stores a
    // signed 16-bit relative position, but the live position itself is never
    // negative.  Keeping it unsigned avoids a sign-extension every time it is
    // used to address `next_tab` while retaining the C wrap-before-narrowing
    // cutoff calculation below.
    let mut cur_pos = (in_next - *in_base) as u32;

    if cur_pos == MATCHFINDER_WINDOW_SIZE as u32 {
        mf.slide_window();
        *in_base += MATCHFINDER_WINDOW_SIZE as usize;
        cur_pos = 0;
    }

    let cutoff: MfPos = cur_pos.wrapping_sub(MATCHFINDER_WINDOW_SIZE as u32) as MfPos;
    // Keep C's cursor representation through the probe.  Reconstructing these
    // pointers from a slice base and integer index at every chain node left a
    // repeated base reload in the inner loop on x86.
    let in_base_ptr = unsafe { buf.as_ptr().add(*in_base) };
    let in_next_ptr = unsafe { buf.as_ptr().add(in_next) };
    let mut best_matchptr = in_next_ptr;

    // Can we read 4 bytes from 'in_next + 1'?
    if max_len < 5 {
        *offset_ret = offset_between(in_next_ptr, best_matchptr);
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

    // ours: prepend the node for the current sequence to the LENGTH-3 chain, exactly
    // as the hash4 line above prepends to the length-4 chain (legacy
    // `matchfinder/hc.rs:334-336`). A position lives in both chains, so it needs its
    // own successor slot. Gated on `hash3_chain_depth > 0` so the shipped
    // singleton shape writes nothing extra.
    debug_assert!((cur_pos as usize) < mf.tables.next3_tab.len());
    if hash3_chain_depth > 0 {
        unsafe { *mf.tables.next3_tab.get_unchecked_mut(cur_pos as usize) = cur_node3 };
    }

    // Compute the next hash codes.
    let next_hashseq = unsafe { load_u32_ptr(in_next_ptr.add(1)) };
    crate::anatomy_count!(hc_hash_computations);
    next_hashes[0] = lz_hash(next_hashseq & 0xFF_FFFF, HC_MATCHFINDER_HASH3_ORDER);
    next_hashes[1] = lz_hash(next_hashseq, HC_MATCHFINDER_HASH4_ORDER);
    prefetchw(unsafe { mf.tables.hash3_tab.as_ptr().add(next_hashes[0] as usize) });
    prefetchw(unsafe { mf.tables.hash4_tab.as_ptr().add(next_hashes[1] as usize) });

    let mut matchptr: *const u8;

    if best_len < 4 {
        // No match of length >= 4 found yet.

        // Check for a length 3 match if needed.
        if cur_node3 <= cutoff {
            *offset_ret = offset_between(in_next_ptr, best_matchptr);
            {
                local.flush();
                return best_len;
            }
        }

        let seq4 = unsafe { load_u32_ptr(in_next_ptr) };

        if best_len < 3 {
            // WALK the length-3 chain instead of probing its head only (legacy
            // `matchfinder/hc.rs:374-407`). depth 0 reproduces the singleton probe
            // exactly — the same single u24 compare at the head, no second
            // dereference.
            let mut node3 = cur_node3;
            let mut left = hash3_chain_depth.max(1);
            loop {
                // SAFETY: `node3 > cutoff` (checked before entry for the head,
                // after every advance below), so the candidate points at processed
                // input; `load_u24_ptr` reads 4 bytes at a position < `in_next`,
                // which stays in bounds under the parser's BUF_PAD contract.
                matchptr = unsafe { in_base_ptr.offset(node3 as isize) };
                if unsafe { load_u24_ptr(matchptr) } == loaded_u32_to_u24(seq4) {
                    best_len = 3;
                    best_matchptr = matchptr;
                    break;
                }
                left -= 1;
                if left == 0 || hash3_chain_depth == 0 {
                    break;
                }
                // SAFETY: masked chain index `< next3_tab.len()`, same masking
                // discipline as the `next_tab` reads below. The candidate at the
                // loaded node is cutoff-checked immediately after (and byte-verified
                // before it can produce an offset), so a stale link terminates or
                // resolves to an in-window position without ever yielding a forged
                // offset.
                node3 = unsafe {
                    *mf.tables
                        .next3_tab
                        .get_unchecked((node3 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize)
                };
                if node3 <= cutoff {
                    break;
                }
            }
        }

        // Check for a length 4 match.
        if cur_node4 <= cutoff {
            *offset_ret = offset_between(in_next_ptr, best_matchptr);
            {
                local.flush();
                return best_len;
            }
        }

        // C-shaped chain walk (libdeflate `hc_matchfinder_longest_match`): the
        // cutoff/depth test happens on the node we have just loaded, BEFORE the next
        // compare, so one check per step and the depth budget is a single `sub`.
        // `matchptr` keeps the base folded into the live pointer so the base is not
        // reloaded from the stack on every step.
        matchptr = unsafe { in_base_ptr.offset(cur_node4 as isize) };
        #[cfg(not(feature = "ladder-tune"))]
        {
            loop {
                bump!(local.attempts);
                if unsafe { load_u32_ptr(matchptr) } == seq4 {
                    break;
                }

                // The first 4 bytes did not match. Advance to the next node in the
                // list, spending one step of the budget on the node we just loaded.
                // The candidate pointer moves by the node delta, so the base is
                // never reloaded per step.
                bump!(local.chain_reads);
                let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
                debug_assert!(ni < mf.tables.next_tab.len());
                let prev = cur_node4;
                cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
                if cur_node4 <= cutoff {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                depth_remaining -= 1;
                if depth_remaining == 0 {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                matchptr = unsafe { matchptr.offset(cur_node4 as isize - prev as isize) };
            }

            // Found a match of length >= 4. Extend it to its full length.
            best_matchptr = matchptr;
            bump!(local.accepted);
            best_len =
                unsafe { lz_extend(buf, in_next, index_from_buf(buf, best_matchptr), 4, max_len) };
            if best_len >= nice_len {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            // One more chain step to enter the >=5 search, spending the budget in C's
            // order.
            bump!(local.chain_reads);
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            if cur_node4 <= cutoff {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            depth_remaining -= 1;
            if depth_remaining == 0 {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
        }

        // SOFTWARE-PIPELINED CHAIN WALK — the `deflate/matchfinder/hc.rs` walk's
        // link-load shape ported here (hc.rs:414-504). The NEXT link is loaded
        // BEFORE the current candidate is compared, so the link load overlaps the
        // probe latency instead of serialising behind each miss; on a hit the
        // already-loaded link feeds the >=5 search without a second walk step. Each
        // preload also hints the next candidate's data line one node ahead (the
        // `prefetchw` at `matchfinder_common.rs:109`): a hint only, issued only for
        // a LIVE node (the same cutoff test the next step would apply), so no
        // address outside the walker's reachable window is ever formed.
        //
        // Visit order, cutoff/depth checks and every compare are IDENTICAL to the
        // classic walk above — same nodes, same order, same returns — so the
        // reported match and the stream are unchanged by construction. Counter
        // totals stay event-for-event too (each preload site bumps `chain_reads`
        // exactly where the classic walk's advance loads): the one exception is a
        // `nice_len` exit, where the pipeline has already paid one extra chain
        // load the classic walk defers — the same accounting the legacy walk has.
        #[cfg(feature = "ladder-tune")]
        {
            bump!(local.chain_reads);
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            let mut next_node = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            if next_node > cutoff {
                crate::compress::ldx::matchfinder_common::prefetch_read(unsafe {
                    in_base_ptr.offset(next_node as isize)
                });
            }
            loop {
                bump!(local.attempts);
                if unsafe { load_u32_ptr(matchptr) } == seq4 {
                    break;
                }

                // Miss: consume the preloaded link, budget it, then register the
                // following one. Same checks, same order, as the classic walk.
                cur_node4 = next_node;
                if cur_node4 <= cutoff {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                depth_remaining -= 1;
                if depth_remaining == 0 {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                matchptr = unsafe { in_base_ptr.offset(cur_node4 as isize) };
                bump!(local.chain_reads);
                let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
                debug_assert!(ni < mf.tables.next_tab.len());
                next_node = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
                if next_node > cutoff {
                    crate::compress::ldx::matchfinder_common::prefetch_read(unsafe {
                        in_base_ptr.offset(next_node as isize)
                    });
                }
            }

            // Found a match of length >= 4. Extend it to its full length.
            best_matchptr = matchptr;
            bump!(local.accepted);
            best_len =
                unsafe { lz_extend(buf, in_next, index_from_buf(buf, best_matchptr), 4, max_len) };
            if best_len >= nice_len {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            // Advance to the next node — already loaded (and prefetched) by the
            // pipeline, replacing the classic one-more-chain-step load.
            cur_node4 = next_node;
            if cur_node4 <= cutoff {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            depth_remaining -= 1;
            if depth_remaining == 0 {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
        }
    } else if cur_node4 <= cutoff || best_len >= nice_len {
        *offset_ret = offset_between(in_next_ptr, best_matchptr);
        {
            local.flush();
            return best_len;
        }
    }

    // Check for matches of length >= 5.
    #[cfg(not(feature = "ladder-tune"))]
    {
        loop {
            // C-shaped inner chain walk: same budget order as the length-4 loop.
            matchptr = unsafe { in_base_ptr.offset(cur_node4 as isize) };
            loop {
                // Already found a length 4 match. Try for a longer match; start by
                // checking either the last 4 bytes and the first 4 bytes, or the last
                // byte. (The last byte, the one which would extend the match length
                // by 1, is the most important.)
                bump!(local.attempts);
                if unsafe { load_u32_ptr(matchptr.add(best_len as usize - 3)) }
                    == unsafe { load_u32_ptr(in_next_ptr.add(best_len as usize - 3)) }
                    && unsafe { load_u32_ptr(matchptr) } == unsafe { load_u32_ptr(in_next_ptr) }
                {
                    break;
                }

                // Continue to the next node in the list, spending one step of the
                // budget on the node we just loaded. The candidate pointer moves by
                // the node delta, so the base is never reloaded per step.
                bump!(local.chain_reads);
                let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
                debug_assert!(ni < mf.tables.next_tab.len());
                let prev = cur_node4;
                cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
                if cur_node4 <= cutoff {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                depth_remaining -= 1;
                if depth_remaining == 0 {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                matchptr = unsafe { matchptr.offset(cur_node4 as isize - prev as isize) };
            }

            // UNALIGNED_ACCESS_IS_FAST: the 4-byte prefix was just re-verified above,
            // so the extension may start at 4 rather than 0.
            let len = unsafe { lz_extend(buf, in_next, index_from_buf(buf, matchptr), 4, max_len) };
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
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
            }

            // Continue to the next node in the list, spending the budget in C's
            // order.
            bump!(local.chain_reads);
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            cur_node4 = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            if cur_node4 <= cutoff {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            depth_remaining -= 1;
            if depth_remaining == 0 {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
        }
    }

    // Length >= 5 walk, software-pipelined identically to the length-4 walk above:
    // `cur_node4 > cutoff` and `depth_remaining > 0` hold on entry (both entry paths
    // guarantee it), so the next chain link is precomputed BEFORE the first probe
    // and each miss reuses it — the compare of the current candidate overlaps the
    // link load and the next candidate's data prefetch. The classic walk's visit
    // order, checks and returns are reproduced exactly, so bytes cannot change.
    #[cfg(feature = "ladder-tune")]
    {
        bump!(local.chain_reads);
        let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
        debug_assert!(ni < mf.tables.next_tab.len());
        let mut next_node = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
        if next_node > cutoff {
            crate::compress::ldx::matchfinder_common::prefetch_read(unsafe {
                in_base_ptr.offset(next_node as isize)
            });
        }
        loop {
            loop {
                matchptr = unsafe { in_base_ptr.offset(cur_node4 as isize) };
                // Already found a length 4 match. Try for a longer match; start by
                // checking either the last 4 bytes and the first 4 bytes, or the
                // last byte. (The last byte, the one which would extend the match
                // length by 1, is the most important.)
                bump!(local.attempts);
                if unsafe { load_u32_ptr(matchptr.add(best_len as usize - 3)) }
                    == unsafe { load_u32_ptr(in_next_ptr.add(best_len as usize - 3)) }
                    && unsafe { load_u32_ptr(matchptr) } == unsafe { load_u32_ptr(in_next_ptr) }
                {
                    break;
                }

                // Miss: consume the preloaded link, budget it, then register the
                // following one. Same checks, same order, as the classic walk.
                cur_node4 = next_node;
                if cur_node4 <= cutoff {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                depth_remaining -= 1;
                if depth_remaining == 0 {
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
                bump!(local.chain_reads);
                let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
                debug_assert!(ni < mf.tables.next_tab.len());
                next_node = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
                if next_node > cutoff {
                    crate::compress::ldx::matchfinder_common::prefetch_read(unsafe {
                        in_base_ptr.offset(next_node as isize)
                    });
                }
            }

            // UNALIGNED_ACCESS_IS_FAST: the 4-byte prefix was just re-verified above,
            // so the extension may start at 4 rather than 0.
            let len = unsafe { lz_extend(buf, in_next, index_from_buf(buf, matchptr), 4, max_len) };
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
                    *offset_ret = offset_between(in_next_ptr, best_matchptr);
                    {
                        local.flush();
                        return best_len;
                    }
                }
            }

            // Continue to the next node — already loaded (and prefetched) by the
            // pipeline, replacing the classic advance load.
            cur_node4 = next_node;
            if cur_node4 <= cutoff {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            depth_remaining -= 1;
            if depth_remaining == 0 {
                *offset_ret = offset_between(in_next_ptr, best_matchptr);
                {
                    local.flush();
                    return best_len;
                }
            }
            bump!(local.chain_reads);
            let ni = (cur_node4 as i32 & (MATCHFINDER_WINDOW_SIZE - 1)) as usize;
            debug_assert!(ni < mf.tables.next_tab.len());
            next_node = unsafe { *mf.tables.next_tab.get_unchecked(ni) };
            if next_node > cutoff {
                crate::compress::ldx::matchfinder_common::prefetch_read(unsafe {
                    in_base_ptr.offset(next_node as isize)
                });
            }
        }
    }
}

/// C: `static forceinline` (`hc_matchfinder.h` / `ht_matchfinder.h`), matched by
/// our `#[inline(always)]` — the C's own call-shape argument applies to this
/// 10-argument call.
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

    // See `hc_matchfinder_longest_match`: this is a nonnegative live position,
    // so use C's unsigned representation for direct table addressing.
    let mut cur_pos = (in_next - *in_base) as u32;
    let mut hash3 = next_hashes[0] as usize;
    let mut hash4 = next_hashes[1] as usize;
    let mut p = in_next;

    loop {
        if cur_pos == MATCHFINDER_WINDOW_SIZE as u32 {
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
        // cost is this insert.
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

    next_hashes[0] = hash3 as u32;
    next_hashes[1] = hash4 as u32;
}

/// C: `in_next - best_matchptr`.
#[inline(always)]
fn offset_between(in_next: *const u8, matchptr: *const u8) -> u32 {
    // Both cursors are derived from the same input slice and the chain only
    // reaches earlier positions, so the difference is nonnegative and fits the
    // DEFLATE window.
    unsafe { in_next.offset_from(matchptr) as u32 }
}

/// Return a cursor's byte index in `buf` for the index-based extension helper.
///
/// SAFETY: `p` must be derived from `buf` and point within it.  The cutoff
/// checks before every chain-node dereference establish that here.
#[inline(always)]
unsafe fn index_from_buf(buf: &[u8], p: *const u8) -> usize {
    unsafe { p.offset_from(buf.as_ptr()) as usize }
}

/// C: `loaded_u32_to_u24(u32 v)` (`matchfinder_common.h:21`), little-endian arm.
#[inline(always)]
fn loaded_u32_to_u24(v: u32) -> u32 {
    v & 0xFF_FFFF
}

/// C: `load_u32_unaligned(const u8 *p)` (`matchfinder_common.h:31`).
///
/// SAFETY: `p..p+4` must be within the input allocation.  The longest-match
/// caller has the same tail proof as C (`max_len >= 5`); chain pointers have
/// already passed the cutoff check.
#[inline(always)]
unsafe fn load_u32_ptr(p: *const u8) -> u32 {
    u32::from_le(unsafe { (p as *const u32).read_unaligned() })
}

/// C: `load_u24_unaligned(const u8 *p)` (`matchfinder_common.h:35`).
/// **At least 4 bytes (not 3) must be available at `p`** — the C says so explicitly.
#[inline(always)]
unsafe fn load_u24_ptr(p: *const u8) -> u32 {
    loaded_u32_to_u24(unsafe { load_u32_ptr(p) })
}

#[inline(always)]
fn load_u32(buf: &[u8], i: usize) -> u32 {
    // The C reads 4 bytes unchecked; its callers guarantee the room via
    // HT_MATCHFINDER_REQUIRED_NBYTES / the compressor's BUF_PAD. Our checked
    // form compiled to a never-taken cmp+jcc->panic cluster in the hottest
    // loop, re-reading a stack-spilled `buf.len()` every iteration.
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
        every_reported_match_at_chain_depth(0);
    }

    /// Same oracle with the trigram chain armed: the walk takes the stale-link
    /// paths (skip-inserted heads carry no fresh `next3_tab` store, by the legacy
    /// shape) across window slides at every repetitive position, so any masked
    /// chain-index or cutoff miss shows up as a phantom match here.
    #[test]
    fn every_reported_match_is_real_at_chain_depth_8() {
        every_reported_match_at_chain_depth(8);
    }

    fn every_reported_match_at_chain_depth(chain3: u32) {
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
                0, // good_match: disabled in the oracle test
                chain3,
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
            // position.
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
                    0, // good_match: disabled in the invariant test
                    0, // chain3: vendor singleton in the invariant test
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
                    0, // good_match: disabled in the monotonicity test
                    0, // chain3: exercised separately below
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

    /// `init` clears the hash tables plus `next3_tab` and deliberately does NOT touch
    /// `next_tab`. Pinned so a later "symmetry fix" — 32768 pointless stores per
    /// stream — fails a test, AND so a later "we can skip the next3 fill too" change
    /// (which would let a previous stream's chain positions leak past the cutoff as
    /// phantom cursor-ahead candidates) fails a test as well.
    #[test]
    fn init_clears_the_hash_tables_and_next3_but_not_next_tab() {
        let mut mf = HcMatchfinder::new();
        mf.tables.next_tab[0] = 1234;
        mf.tables.next3_tab[0] = 4321;
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
            mf.tables.next3_tab[0],
            super::super::matchfinder_common::MATCHFINDER_INITVAL,
            "a stale cross-stream trigram-can link must not survive init"
        );
        assert_eq!(
            mf.tables.next_tab[0], 1234,
            "C: hc_matchfinder_init passes TOTAL_HASH_SIZE, not sizeof(*mf)"
        );
    }

    /// The trigram-chain mechanism, pinned: when the trigram head holds a
    /// key-COLLIDING trigram (same 15-bit bucket, different bytes), the singleton
    /// probe finds nothing, but an armed chain walks back to the genuinely
    /// matching earlier occurrence — the exact discovery gap gzip's `prev[]`
    /// chain serves and libdeflate's singleton cannot. All insertion happens
    /// through `longest_match` (the only path that writes `next3_tab` links),
    /// stepped one position at a time exactly as the greedy parser advances;
    /// the homogeneous filler keeps every other position out of the colliding
    /// hash3 bucket, so the walk is deterministic.
    #[test]
    fn trigram_chain_finds_a_match_behind_a_colliding_head() {
        use super::super::matchfinder_common::lz_hash;
        let h3_of = |s: [u8; 3]| {
            let seq = s[0] as u32 | (s[1] as u32) << 8 | (s[2] as u32) << 16;
            lz_hash(seq, HC_MATCHFINDER_HASH3_ORDER)
        };

        // A,B,C is the real repeated trigram; [A,B2,C2] shares its hash3 bucket
        // (found by a fixed scan — determines deterministically on every host)
        // and X keeps the homogeneous filler out of the colliding bucket
        // entirely.
        let (a, b, c, x) = (0x41u8, 0x42u8, 0xEEu8, 0x55u8);
        let trigram = [a, b, c];
        let collide = (0u8..=255)
            .find_map(|b2| {
                (0u8..=255).find_map(|c2| {
                    if (b2, c2) != (b, c)
                        && b2 != x
                        && c2 != x
                        && h3_of([a, b2, c2]) == h3_of(trigram)
                    {
                        Some([a, b2, c2])
                    } else {
                        None
                    }
                })
            })
            .expect("32768 buckets over 65536 trigrams: a collision exists by pigeonhole");
        assert_eq!(h3_of(collide), h3_of(trigram), "sanity: collision picked");
        assert_ne!(
            h3_of([x, x, x]),
            h3_of(trigram),
            "the homogeneous filler must stay out of the colliding bucket"
        );
        let junk = [x; 32];

        // trigram @ q | junk | collide | junk | probe=A,B,C (bytes end; the
        // rest is the parser's trailing pad).
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&trigram);
        let q = 0usize;
        buf.extend_from_slice(&junk);
        let _k = buf.len();
        buf.extend_from_slice(&collide);
        buf.extend_from_slice(&junk);
        let probe = buf.len();
        buf.extend_from_slice(&trigram);
        buf.extend_from_slice(&[0u8; 16]);

        // Drive the matchfinder one longest_match per position, advancing one
        // byte, `next_hashes` built for each position from its own first four
        // bytes.
        let run = |chain3: u32| -> (u32, u32) {
            let mut mf = HcMatchfinder::new();
            let mut in_base = 0usize;
            let mut scratch = 0u32;
            for p in 0..probe {
                let s3 = buf[p] as u32 | (buf[p + 1] as u32) << 8 | (buf[p + 2] as u32) << 16;
                let s4 = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
                let mut nh = [
                    lz_hash(s3, HC_MATCHFINDER_HASH3_ORDER),
                    lz_hash(s4, HC_MATCHFINDER_HASH4_ORDER),
                ];
                hc_matchfinder_longest_match(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    p,
                    2, // min_len 3 — the only state where the chain engages
                    258,
                    258,
                    2,
                    0, // good_match disabled
                    chain3,
                    &mut nh,
                    &mut scratch,
                );
            }
            // The probe itself: its trigram's bucket head is the colliding
            // trigram, inserted LAST among same-bucket positions.
            let s3 =
                buf[probe] as u32 | (buf[probe + 1] as u32) << 8 | (buf[probe + 2] as u32) << 16;
            let s4 =
                u32::from_le_bytes([buf[probe], buf[probe + 1], buf[probe + 2], buf[probe + 3]]);
            let mut nh = [
                lz_hash(s3, HC_MATCHFINDER_HASH3_ORDER),
                lz_hash(s4, HC_MATCHFINDER_HASH4_ORDER),
            ];
            let mut offset = 0u32;
            let len = hc_matchfinder_longest_match(
                &mut mf,
                &buf,
                &mut in_base,
                probe,
                2,
                258,
                258,
                2,
                0,
                chain3,
                &mut nh,
                &mut offset,
            );
            (len, offset)
        };

        // depth 0 = the vendor's singleton: the head (colliding) fails the u24
        // compare and no len-3 is reported.
        let (len0, _) = run(0);
        assert!(len0 < 3, "singleton must not find the len-3 (got {len0})");
        // depth >= 2 walks past the colliding head to `q` and reports the real
        // len-3 at its distance.
        for depth in [2u32, 8] {
            let (len, offset) = run(depth);
            assert_eq!(len, 3, "chain depth {depth} must recover the len-3");
            let src = probe - offset as usize;
            assert_eq!(src, q, "the match must be the earlier occurrence");
            assert_eq!(&buf[src..src + 3], &trigram, "chain match must be real");
        }
    }
}
