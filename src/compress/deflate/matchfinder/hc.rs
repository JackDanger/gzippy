//! Hash-chains Lempel-Ziv matchfinder.
//!
//! Port of libdeflate `vendor/libdeflate/lib/hc_matchfinder.h`: a hash table of
//! linked lists (chains) for length-4+ matches plus a separate chain-less hash
//! table for length-3 matches. Positions are stored as `mf_pos_t` (`i16` for the
//! 32 KiB DEFLATE window), relative to a sliding `in_base`, with the saturating
//! `matchfinder_init`/`matchfinder_rebase` sentinel machinery from Increment 1's
//! [`super::common`].
//!
//! Faithful transliterations: `hc_matchfinder_longest_match` (the two-loop
//! find-first-then-find-longer structure with the last-4+first-4 prefilter,
//! ~:182-338) and `hc_matchfinder_skip_bytes` (~:360-399). The signed 16-bit
//! position arithmetic (cutoff comparisons, the `& (WINDOW_SIZE - 1)` chain
//! index that survives a rebase because the rebase only flips the sign bit) is
//! reproduced exactly; correctness is pinned by the roundtrip + proptest nets in
//! `src/tests/deflate_encoder_matches.rs`.

//! ## Soundness invariant (Increment 5: unsafe raw-pointer / unchecked codegen)
//!
//! The hot loops below drop Rust's bounds checks and checked arithmetic to match
//! libdeflate's C codegen. Every elided check is discharged by construction:
//!
//! * **Hash-table indices.** `hash3 = lz_hash(seq, 15) >> (32-15)` so
//!   `hash3 < 2^15 == HASH3_SIZE`; `hash4 = lz_hash(seq, 16)` so
//!   `hash4 < 2^16 == HASH4_SIZE`. Both come straight out of `lz_hash`/`next_hashes`
//!   (never arithmetic), so `hash3_tab`/`hash4_tab` `get_unchecked[_mut]` are in
//!   bounds.
//! * **Chain / next indices.** `next_tab` is indexed by `cur_pos ∈ 0..WINDOW_SIZE`
//!   (reset to 0 whenever it reaches `WINDOW_SIZE`) and by
//!   `(cur_node as u16 & WINDOW_MASK) < WINDOW_SIZE` — both `< 32768 == next_tab.len()`.
//! * **Buffer reads.** The parser pads its working buffer by `BUF_PAD (=16)` bytes
//!   past `in_end`, and `adjust_max_and_nice_len` clamps `max_len` so
//!   `in_next + max_len <= in_end`. The `max_len < 5` early-return makes the
//!   `in_next + 1` (4-byte) hash read safe; every candidate read is at
//!   `matchptr < in_next` with offset `<= max_len`, and the `+ (best_len-3)`
//!   prefilter reads land at `<= in_next + best_len + 1 <= in_end + 1 < buf.len()`.
//!   Each `load_u32`/`load_u24` site carries a `debug_assert!` proving `off+4 <= len`.

use super::common::{
    load_u24, load_u32, lz_extend, lz_hash, matchfinder_rebase, prefetch_write, MATCHFINDER_INITVAL,
};

/// Per-`longest_match`-call local accumulator for the chain-walk counters
/// (`anatomy-counters` feature only). The walk visits O(`max_search_depth`)
/// chain nodes per call, so an atomic `fetch_add` at every visited node
/// (measured: ~10-14% wall overhead at L6-L9, where chains run deep) costs
/// far more than accumulating in plain locals and flushing with ONE
/// `fetch_add` per counter at the single return point below — same exact
/// final counts (this is a pure batching of the same events), far fewer
/// atomic ops.
#[cfg(feature = "anatomy-counters")]
#[derive(Default)]
struct HcLocalCounters {
    attempts: u64,
    miss: u64,
    too_short: u64,
    accepted: u64,
    chain_reads: u64,
}

#[cfg(feature = "anatomy-counters")]
impl HcLocalCounters {
    #[inline(always)]
    fn flush(self) {
        crate::anatomy_count!(hc_probe_attempts, self.attempts);
        crate::anatomy_count!(hc_probe_outcome_miss, self.miss);
        crate::anatomy_count!(hc_probe_outcome_too_short, self.too_short);
        crate::anatomy_count!(hc_probe_outcome_accepted, self.accepted);
        crate::anatomy_count!(hc_chain_table_reads, self.chain_reads);
    }
}

pub const HC_HASH3_ORDER: u32 = 15;
pub const HC_HASH4_ORDER: u32 = 16;

const HASH3_SIZE: usize = 1 << HC_HASH3_ORDER;
const HASH4_SIZE: usize = 1 << HC_HASH4_ORDER;

/// 32 KiB DEFLATE window.
pub const WINDOW_SIZE: usize = 1 << 15;
const WINDOW_MASK: u16 = (WINDOW_SIZE - 1) as u16;

/// The hash-chains matchfinder state.
///
/// `next_tab` is stored as an INLINE fixed-size array rather than a `Vec<i16>`
/// (libdeflate's representation): a chain-walk read is then `self + const_offset
/// + i*2` (one x86 addressing mode, disp32 immediate) instead of a `RawVec`
/// pointer deref, and no register is tied up holding the table base. The struct
/// is >64 KiB, so it is always heap-boxed (`new` returns `Box<Self>`) and never
/// passed/constructed by value.
pub struct HcMatchfinder {
    /// Singleton nodes for length-3 matches (`hash3_tab`).
    hash3_tab: [i16; HASH3_SIZE],
    /// First node of each length-4+ chain (`hash4_tab`).
    hash4_tab: [i16; HASH4_SIZE],
    /// `next_tab[pos]` = the node following `pos` in its chain.
    next_tab: [i16; WINDOW_SIZE],
}

thread_local! {
    /// One recycled `HcMatchfinder` per thread (see [`HcMatchfinder::acquire`]).
    /// `thread_local!` rather than a shared pool: each `infra::scheduler`
    /// worker thread gets its own slot, so there is no cross-thread mutable
    /// state and no synchronization — a fresh page-table entry per OS thread,
    /// not a data structure that needs locking.
    static HC_POOL: std::cell::RefCell<Option<Box<HcMatchfinder>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII handle returned by [`HcMatchfinder::acquire`]. `Deref`/`DerefMut` to
/// `HcMatchfinder` so every existing `&mut mf` call site (which relies on the
/// same deref-coercion `Box<HcMatchfinder>` already provided) needs no
/// change. On drop, the box is returned to this thread's [`HC_POOL`] instead
/// of being freed, so the NEXT `acquire()` on this same thread reuses the
/// allocation rather than requesting a new one from the allocator.
pub struct PooledHc(Option<Box<HcMatchfinder>>);

impl std::ops::Deref for PooledHc {
    type Target = HcMatchfinder;
    #[inline]
    fn deref(&self) -> &HcMatchfinder {
        // SAFETY/invariant: `self.0` is `Some` for the entire lifetime of a
        // `PooledHc` outside of `Drop::drop` (which is the only place it is
        // taken and never observed again afterward).
        self.0.as_deref().expect("PooledHc used after drop")
    }
}

impl std::ops::DerefMut for PooledHc {
    #[inline]
    fn deref_mut(&mut self) -> &mut HcMatchfinder {
        self.0.as_deref_mut().expect("PooledHc used after drop")
    }
}

impl Drop for PooledHc {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            HC_POOL.with(|cell| {
                *cell.borrow_mut() = Some(b);
            });
        }
    }
}

impl HcMatchfinder {
    /// `hc_matchfinder_init`: allocate and initialize every table to the sentinel.
    ///
    /// Returns a heap `Box` (the struct is >64 KiB with the inline `next_tab`).
    /// Built through `Box::new_uninit` so no 64 KiB+ temporary ever lands on the
    /// stack; every field is written before `assume_init`.
    pub fn new() -> Box<Self> {
        let mut boxed = Box::<Self>::new_uninit();
        // SAFETY: `new_uninit` gives an aligned, fully-owned allocation for one
        // `HcMatchfinder`. We initialize EVERY `i16` of all three inline tables
        // before `assume_init`, writing `MATCHFINDER_INITVAL` (the exact value
        // `matchfinder_init` writes — a `-WINDOW_SIZE`/`0x8000` sentinel, NOT
        // zero). `addr_of_mut!` avoids forming a reference to uninit memory;
        // each `.add(i)` for `i < LEN` stays inside its `[i16; LEN]` field.
        unsafe {
            let p = boxed.as_mut_ptr();
            let h3 = core::ptr::addr_of_mut!((*p).hash3_tab) as *mut i16;
            for i in 0..HASH3_SIZE {
                h3.add(i).write(MATCHFINDER_INITVAL);
            }
            let h4 = core::ptr::addr_of_mut!((*p).hash4_tab) as *mut i16;
            for i in 0..HASH4_SIZE {
                h4.add(i).write(MATCHFINDER_INITVAL);
            }
            let nt = core::ptr::addr_of_mut!((*p).next_tab) as *mut i16;
            for i in 0..WINDOW_SIZE {
                nt.add(i).write(MATCHFINDER_INITVAL);
            }
            boxed.assume_init()
        }
    }

    /// Re-arm every table to the sentinel value in place — the same
    /// postcondition as [`Self::new`], but reusing the existing 256 KiB
    /// allocation instead of requesting a fresh one. Used by [`Self::acquire`]
    /// to recycle a thread-local instance across chunks.
    fn reset(&mut self) {
        self.hash3_tab.fill(MATCHFINDER_INITVAL);
        self.hash4_tab.fill(MATCHFINDER_INITVAL);
        self.next_tab.fill(MATCHFINDER_INITVAL);
    }

    /// Pooled equivalent of [`Self::new`]: hands back a `HcMatchfinder` reset
    /// to its initial sentinel state, reused from a thread-local free list
    /// when one is available (this thread already allocated one for an
    /// earlier chunk) instead of paying the 256 KiB allocation again.
    ///
    /// This targets the T>1 parallel path, where `run()` (greedy/lazy/gated)
    /// is invoked once per CHUNK, but chunk count can exceed thread count —
    /// each `infra::scheduler` worker thread claims and processes MANY chunks
    /// sequentially from the atomic work queue (`compress_parallel`'s
    /// `worker_loop_timed`), so without pooling every one of those chunks pays
    /// a fresh 256 KiB allocation + sentinel fill (DHAT: "1x / 9-28x = chunk
    /// count" cadence). The pool is `thread_local!`, so each worker thread
    /// gets its own instance — no state is EVER shared across threads, only
    /// reused across the sequential chunks one thread claims, so this does
    /// not break the chunk-independence the T>1 path relies on (each chunk
    /// still starts from a freshly-reset matchfinder, byte-identical to a
    /// brand-new `Self::new()`).
    pub fn acquire() -> PooledHc {
        let existing = HC_POOL.with(|cell| cell.borrow_mut().take());
        let mut mf = existing.unwrap_or_else(Self::new);
        mf.reset();
        PooledHc(Some(mf))
    }

    /// `hc_matchfinder_slide_window`: rebase every stored position by one window.
    #[inline]
    fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.hash3_tab[..]);
        matchfinder_rebase(&mut self.hash4_tab[..]);
        matchfinder_rebase(&mut self.next_tab[..]);
    }

    /// Find the longest match longer than `best_len_in` at `in_next`.
    ///
    /// Faithful port of `hc_matchfinder_longest_match`. Returns
    /// `(best_len, offset)`; when no match longer than `best_len_in` is found the
    /// returned length is `best_len_in` and the offset is meaningless (0). The
    /// caller must ensure `buf` is padded so 4-byte loads up to
    /// `in_next + best_len + 1` stay in bounds.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn longest_match(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        best_len_in: u32,
        max_len: u32,
        nice_len: u32,
        max_search_depth: u32,
        next_hashes: &mut [u32; 2],
    ) -> (u32, u32) {
        // `bucket-oracle-null-mf` (Cargo.toml doc comment): the CEILING
        // ORACLE for Task B/matchfinder-share bounding. Returns "no match"
        // before touching hash3_tab/hash4_tab/next_tab or walking any chain
        // -- the caller's `length >= min_len` accept check always fails
        // (`best_len_in` is exactly `min_len - 1`), so every position takes
        // the literal path and the lazy parser's accept/defer DECISION code
        // (never reached without a candidate match) is bounded jointly with
        // probe, not separately -- see the feature's doc comment for why
        // that joint bound is the honest claim here. `next_hashes` is left
        // un-updated: fine for an oracle (never referenced again on this
        // path) but means this build must never be used for anything but
        // wall-time bounding.
        #[cfg(feature = "bucket-oracle-null-mf")]
        {
            let _ = (
                buf,
                in_base,
                in_next,
                max_len,
                nice_len,
                max_search_depth,
                next_hashes,
            );
            return (best_len_in, 1);
        }
        #[cfg(not(feature = "bucket-oracle-null-mf"))]
        {
            debug_assert!(max_search_depth >= 1, "max_search_depth must be >= 1");
            let mut best_len = best_len_in;
            let mut depth_remaining = max_search_depth;
            let mut best_matchptr = in_next; // absolute offset into `buf`

            let mut cur_pos = in_next - *in_base;
            if cur_pos == WINDOW_SIZE {
                self.slide_window();
                *in_base += WINDOW_SIZE;
                cur_pos = 0;
            }
            let in_base_v = *in_base;
            let cutoff: i32 = cur_pos as i32 - WINDOW_SIZE as i32;

            // Can we read 4 bytes from `in_next + 1`?
            if max_len < 5 {
                return (best_len, (in_next - best_matchptr) as u32);
            }

            // Raw buffer pointer + length for the unchecked loads. `blen` is only used
            // by the debug_assert bounds checks, so it is dead in release builds.
            let base = buf.as_ptr();
            let blen = buf.len();

            #[cfg(feature = "anatomy-counters")]
            let mut local = HcLocalCounters::default();

            let hash3 = next_hashes[0] as usize;
            let hash4 = next_hashes[1] as usize;

            // SAFETY: `hash3 < HASH3_SIZE` and `hash4 < HASH4_SIZE` by the module
            // soundness invariant (they are `lz_hash` outputs of order 15/16), and
            // `cur_pos ∈ 0..WINDOW_SIZE == next_tab.len()`.
            crate::anatomy_count!(hc_head_table_reads, 2u64);
            crate::anatomy_count!(hc_head_table_writes, 2u64);
            let (cur_node3, mut cur_node4) = unsafe {
                debug_assert!(hash3 < HASH3_SIZE && hash4 < HASH4_SIZE && cur_pos < WINDOW_SIZE);
                let cur_node3 = *self.hash3_tab.get_unchecked(hash3);
                let cur_node4 = *self.hash4_tab.get_unchecked(hash4);
                // Insert the current sequence: replace hash3 singleton, prepend to hash4.
                *self.hash3_tab.get_unchecked_mut(hash3) = cur_pos as i16;
                *self.hash4_tab.get_unchecked_mut(hash4) = cur_pos as i16;
                *self.next_tab.get_unchecked_mut(cur_pos) = cur_node4;
                (cur_node3, cur_node4)
            };

            // SAFETY: `max_len >= 5` (checked above) and `in_next + max_len <= in_end`
            // (parser clamp), so `in_next + 1 + 4 <= in_end + 4 <= buf.len()` (BUF_PAD).
            let next_hashseq = unsafe {
                debug_assert!(in_next + 1 + 4 <= blen);
                load_u32(base, in_next + 1)
            };
            next_hashes[0] = lz_hash(next_hashseq & 0xFF_FFFF, HC_HASH3_ORDER);
            next_hashes[1] = lz_hash(next_hashseq, HC_HASH4_ORDER);
            crate::anatomy_count!(hc_hash_computations);
            // Vendor `prefetchw` (hc_matchfinder.h:238-239): warm the hash buckets
            // for `in_next + 1` in an exclusive state — they are stored to on the
            // next call. Pure hint; cannot change which match is found.
            // SAFETY: `next_hashes[0] < HASH3_SIZE` and `next_hashes[1] < HASH4_SIZE`
            // (lz_hash order-15/16 outputs), so both `.add` land in-allocation.
            unsafe {
                prefetch_write(self.hash3_tab.as_ptr().add(next_hashes[0] as usize) as *const u8);
                prefetch_write(self.hash4_tab.as_ptr().add(next_hashes[1] as usize) as *const u8);
            }

            // SAFETY: same as above — `in_next + 4 <= in_end <= buf.len()`.
            let seq4 = unsafe {
                debug_assert!(in_next + 4 <= blen);
                load_u32(base, in_next)
            };

            // `matchptr` carries the candidate that entered the length>=5 loop.
            let mut matchptr;

            'search: {
                if best_len < 4 {
                    // Length-3 match check.
                    if (cur_node3 as i32) <= cutoff {
                        break 'search;
                    }
                    if best_len < 3 {
                        let mp = (in_base_v as isize + cur_node3 as isize) as usize;
                        // SAFETY: `cutoff < cur_node3` so `mp < in_next`, and it points
                        // into processed input; the u24 load reads 4 bytes at `mp` and
                        // `mp + 4 <= in_next + 4 <= buf.len()`.
                        let cand = unsafe {
                            debug_assert!(mp < in_next && mp + 4 <= blen);
                            load_u24(base, mp)
                        };
                        if cand == seq4 & 0xFF_FFFF {
                            best_len = 3;
                            best_matchptr = mp;
                        }
                    }

                    // Length-4 match check.
                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    // FALSIFY/FALSIFIED 2026-07-28 — DO NOT de-pipeline this. The hoist
                    // looks vestigial: the prefetch it was written to feed was
                    // measured as a loss and deleted, so "it now keeps a value
                    // live for nothing" is the obvious reading. It is WRONG. The
                    // hoist independently hides the DEPENDENT-LOAD LATENCY of the
                    // pointer chase — `next_tab[cur_node4 & MASK]` feeds the next
                    // iteration's address, so reading it at the loop bottom
                    // serialises load->use on the chain's critical path.
                    // De-pipelining both walks, frozen Zen2, n=31, output
                    // byte-identical, vs its exact parent:
                    //     L1 1.0016 noisy   L2 1.0131   L6 1.0624   L9 1.0992
                    // all RESOLVED SLOWER, and worse the deeper the chain — the
                    // signature of exposed load latency. It went the wrong way
                    // even though it REMOVED instructions (L2 Ir -1.77%, L6
                    // -2.91%) and cut L2 reads (Dr -3.87%). Fewer instructions,
                    // far more wall: instruction and read counts LOCATE, they do
                    // not predict.
                    //
                    // Software-pipelined chain walk (Increment 5b). Hoist the NEXT
                    // chain node one step ahead so this iteration can prefetch the
                    // FOLLOWING candidate's match data — the second, reducible
                    // dependent load, which sits off the chain's critical path —
                    // while the current node's compare runs. `next_tab` is never
                    // mutated during a walk, so reading a node early yields the
                    // identical value the un-pipelined form read at the loop bottom;
                    // the sequence of `cur_node4` visited, every cutoff/depth check,
                    // and the resulting match are byte-identical (pinned by
                    // `matches_equal_scalar_*`). The prefetch is a pure hint.
                    // SAFETY: `(cur_node4 as u16 & WINDOW_MASK) < WINDOW_SIZE == next_tab.len()`.
                    let mut next_node = unsafe {
                        *self
                            .next_tab
                            .get_unchecked((cur_node4 as u16 & WINDOW_MASK) as usize)
                    };
                    #[cfg(feature = "anatomy-counters")]
                    {
                        local.chain_reads += 1;
                    }
                    // CHAIN BASE, hoisted out of the walk to free a register.
                    //
                    // The walk used to compute `matchptr = in_base_v + cur_node4` and then
                    // `load_u32(base, matchptr)` — TWO adds, with `base` and `in_base_v`
                    // both live across every iteration. Folding them into one pointer
                    // leaves ONE add and ONE live value.
                    //
                    // A register-pressure fix, NOT a load hoist — the distinction matters
                    // because load hoisting is falsified twice in this file. cg_annotate on
                    // 6,000,000 B of dickens at L6 T1 attributed 30,311,345 DATA READS to
                    // `matchptr = (in_base_v as isize + ...)`, a line that is an ADD OF TWO
                    // LOCALS and should carry none. `in_base_v` was ALREADY a local, so this
                    // was never aliasing and no by-value signature change could reach it.
                    //
                    // SAFETY: `wrapping_add` never dereferences; every load below is at
                    // `cur_node4 > cutoff`, i.e. exactly the addresses the old `matchptr`
                    // form produced, whose bounds the SAFETY note there covers.
                    let chain_base = base.wrapping_add(in_base_v);
                    loop {
                        // FALSIFY/FALSIFIED 2026-07-28 — DO NOT RE-ADD WITHOUT MEASURING.
                        // A `prefetch_read` of the next chain node used to sit
                        // here, one iteration ahead. It was a net LOSS: it cost
                        // 103M L1 loads on dickens L6 (412.6M -> 309.4M when
                        // removed) to buy misses we were not taking. Against
                        // libdeflate on identical output we were MISSING LESS
                        // (7.0% vs their 16.4%) while executing 70% MORE loads
                        // — the signature of over-prefetching. Removing it:
                        // Intel -5.1% wall / -5.2% cycles, M1 geomean 0.9610
                        // (-10% at L6/L9), AMD Zen2 frozen geomean 0.9993.
                        // Gate was pre-registered before measuring: no cell
                        // above 1.02 and geomean <= 1.0, across 4 entropy
                        // classes x L1/6/9 x 3 microarchitectures. Output
                        // byte-identical throughout. The hardware prefetcher
                        // already covers this access pattern on every core we
                        // ship to.
                        // SAFETY: `cutoff < cur_node4` so `matchptr < in_next`, thus
                        // `matchptr + 4 <= in_next + 4 <= buf.len()`.
                        // SAFETY: as the chain_base note above.
                        let cand = unsafe {
                            #[cfg(debug_assertions)]
                            {
                                let mp = (in_base_v as isize + cur_node4 as isize) as usize;
                                debug_assert!(mp < in_next && mp + 4 <= blen);
                            }
                            u32::from_le(core::ptr::read_unaligned(
                                chain_base.wrapping_offset(cur_node4 as isize) as *const u32,
                            ))
                        };
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.attempts += 1;
                        }
                        if cand == seq4 {
                            break;
                        }
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.miss += 1;
                        }
                        cur_node4 = next_node;
                        if (cur_node4 as i32) <= cutoff {
                            break 'search;
                        }
                        depth_remaining -= 1;
                        if depth_remaining == 0 {
                            break 'search;
                        }
                        // SAFETY: masked chain index `< next_tab.len()`.
                        next_node = unsafe {
                            *self
                                .next_tab
                                .get_unchecked((cur_node4 as u16 & WINDOW_MASK) as usize)
                        };
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.chain_reads += 1;
                        }
                    }

                    // Found a length-4 match; extend it fully. `matchptr` is materialised
                    // HERE, once, instead of on every walk iteration.
                    matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                    best_matchptr = matchptr;
                    best_len = lz_extend(buf, in_next, matchptr, 4, max_len);
                    #[cfg(feature = "anatomy-counters")]
                    {
                        local.accepted += 1;
                    }
                    if best_len >= nice_len {
                        break 'search;
                    }
                    // Advance to the next node — already loaded by the pipeline
                    // (`next_node == next_tab[cur_node4 & MASK]` holds at the break).
                    cur_node4 = next_node;
                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    depth_remaining -= 1;
                    if depth_remaining == 0 {
                        break 'search;
                    }
                } else {
                    if (cur_node4 as i32) <= cutoff || best_len >= nice_len {
                        break 'search;
                    }
                }

                // Length >= 5 loop, software-pipelined identically to the length-4
                // walk above. `cur_node4 > cutoff` and `depth_remaining > 0` hold
                // here (both entry paths guarantee it); precompute the next chain
                // node so the compare can overlap the following candidate's prefetch.
                // SAFETY: masked chain index `< next_tab.len()`.
                let mut next_node = unsafe {
                    *self
                        .next_tab
                        .get_unchecked((cur_node4 as u16 & WINDOW_MASK) as usize)
                };
                #[cfg(feature = "anatomy-counters")]
                {
                    local.chain_reads += 1;
                }
                // FALSIFY 2026-07-30 (FALSIFIED) — do NOT apply the `chain_base` hoist to
                // THIS walk. It WINS on the length-4 walk above and LOSES here, and the
                // difference is instructive.
                //
                // Same transformation, one loop lower: hoist `base + in_base_v` out, index
                // the candidate by `cur_node4`, materialise `matchptr` only on a hit.
                // Output byte-identical. Measured, 6,000,000 B of dickens at L6 T1:
                //     hc.rs Dr    108,282,726 -> 86,198,253    (-20.4%)
                //     program Dr  161,009,178 -> 142,887,715   (-11.3%)
                //     program Dw   52,744,347 -> 58,688,632    (+11.3%)   <- the tell
                // Frozen paired wall (solvency, n=15, /dev/null both arms) came back
                // SLOWER: photo.jpg 1.0567, armexe.elf 1.0119, tool.bin 1.0095, dickens
                // 1.0085, geomean ~1.012. photo.jpg is the file the length-4 hoist made
                // 5.6% FASTER, so this is not noise and not the same effect twice.
                //
                // MECHANISM, legible in the counters: the reads did not vanish, they became
                // STORES. Deferring `matchptr` keeps `cur_node4` live across a longer region
                // in a loop that ALREADY needs `best_len`, `off`, `n_hi` and `n_lo`, and the
                // allocator paid for it in spill writes. The length-4 walk has a much
                // smaller live set, which is exactly why the identical change wins there.
                // The transformation is not what matters; the live-set budget of the
                // specific loop is.
                //
                // ⚠ PROVENANCE CORRECTION 2026-07-31, applies to BOTH verdicts above and
                // to the length-4 hoist that shipped: `photo.jpg` is a GATE member
                // (`corpus_split.json`), not TUNE. It was quoted as the headline for the
                // shipped win ("0.9443", "our worst wall cell 1.2274 -> 1.159") and again
                // here as the tell. No parameter was fitted to it, so no promotion is
                // void — but the win is WEAKER than it was reported. Recomputed on TUNE
                // members only: armexe.elf 0.9849, symbols.dwarf 0.9969, aozora.txt
                // 0.9983, dickens 1.0023, data.csv 1.0081 => geomean 0.9981, i.e. 0.19%,
                // INSIDE this rig's ~1.5% A/A noise floor, with only armexe.elf arguably
                // outside it. The 0.9889 figure was carried by the gate file.
                // Select and headline on TUNE; let GATE only ever confirm.
                //
                // GENERAL: a -20% READ count in a matchfinder does not imply a faster wall.
                // Two independent instances now — this, and the u32-packed bucket in
                // `matchfinder::ht` (-26.7% writes, zero wall movement). Any counter-only
                // argument here needs a frozen paired wall run before it is believed.
                //
                // NEXT LEVER, FROM A VENDOR DIFF (2026-07-31) — NOT YET BUILT.
                // The prefilter below is TERM-FOR-TERM libdeflate's (`hc_matchfinder.h`,
                // "Check for matches of length >= 5"): same four u32 loads, same two
                // comparisons, same order. So the 1.88x read excess (ours 108,282,726 Dr
                // vs their 57,593,044, identical D1 misses) is NOT in the comparison
                // shape. It is in the ADDRESSING MODE one level down:
                //     libdeflate:  matchptr = &in_base[cur_node4];   // a real pointer
                //                  load_u32_unaligned(matchptr + best_len - 3)
                //     ours:        matchptr: usize                  // an INDEX
                //                  load_u32(base, matchptr + off)  => base.add(idx + off)
                // Ours keeps `base` live across the ENTIRE walk and pays a two-register
                // add per load where libdeflate pays one register plus a displacement.
                // That is the same defect the shipped `chain_base` hoist removed from the
                // length-4 walk, which is why that hoist won.
                //
                // HOW THIS DIFFERS FROM THE FALSIFICATION RECORDED BELOW, which is the
                // only reason it may be attempted at all: the falsified attempt DEFERRED
                // materialising `matchptr` to after the walk, which extended `cur_node4`'s
                // live range and turned reads into spill STORES (+11.3% Dw). libdeflate
                // materialises the pointer IMMEDIATELY, every iteration, and never extends
                // anything's live range. Change the TYPE (usize -> *const u8), not the
                // TIMING. Any attempt must show Dr down AND Dw not up, then a frozen
                // paired wall run on TUNE members before it is believed.
                //
                // ADDRESSING-MODE FIX, built from that diff. Both values below are
                // loop-invariant across the entire walk and are computed ONCE:
                //   `mp_base` makes the match side a real pointer, so a candidate is
                //   `mp_base + cur_node4` (one add) instead of `base.add(idx + off)`
                //   (two), matching libdeflate's `matchptr = &in_base[cur_node4]`.
                //   `in_ptr` does the same for the `in_next` side, which never moves
                //   inside this walk.
                // Together they take `base` OUT of the inner loop entirely — this is a
                // live-set reduction, not a load-count trick. `matchptr` is still
                // materialised IMMEDIATELY every iteration (below); the timing is
                // deliberately unchanged, because DEFERRING it is what was falsified.
                // SAFETY: `wrapping_add`/`wrapping_offset` never dereference; every load
                // is at `cur_node4 > cutoff`, i.e. exactly the addresses the old
                // `load_u32(base, ..)` form produced, whose bounds the SAFETY note on the
                // prefilter covers.
                let mp_base = base.wrapping_add(in_base_v);
                let in_ptr = base.wrapping_add(in_next);
                loop {
                    loop {
                        matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                        // FALSIFY/FALSIFIED 2026-07-28 — DO NOT RE-ADD WITHOUT MEASURING.
                        // A `prefetch_read` of the next chain node used to sit
                        // here, one iteration ahead. It was a net LOSS: it cost
                        // 103M L1 loads on dickens L6 (412.6M -> 309.4M when
                        // removed) to buy misses we were not taking. Against
                        // libdeflate on identical output we were MISSING LESS
                        // (7.0% vs their 16.4%) while executing 70% MORE loads
                        // — the signature of over-prefetching. Removing it:
                        // Intel -5.1% wall / -5.2% cycles, M1 geomean 0.9610
                        // (-10% at L6/L9), AMD Zen2 frozen geomean 0.9993.
                        // Gate was pre-registered before measuring: no cell
                        // above 1.02 and geomean <= 1.0, across 4 entropy
                        // classes x L1/6/9 x 3 microarchitectures. Output
                        // byte-identical throughout. The hardware prefetcher
                        // already covers this access pattern on every core we
                        // ship to.
                        //
                        // FALSIFY/FALSIFIED 2026-07-28 (second time, different change) —
                        // DO NOT hand-hoist the two operands below that describe
                        // the CURRENT position. It looks like free money: this
                        // prefilter reads four values per candidate, and two of
                        // them (`in_next + off`, `in_next`) are loop-invariant
                        // across the rejection walk — `in_next` does not move and
                        // `best_len` only changes on acceptance, which exits the
                        // loop. `load_u32(base, in_next)` is even just `seq4`,
                        // already in a variable.
                        //
                        // LLVM ALREADY HOISTS THEM. Doing it by hand made things
                        // WORSE, measured on the shipped build shape at L2 on 8 MB
                        // of silesia:
                        //     Ir  555,124,179 -> 552,722,907  (-0.43%)
                        //     Dr  103,975,406 -> 104,103,912  (+0.12%)  <-- UP
                        // Data reads went UP after removing two source-level loads
                        // per candidate, which only happens if the loads were never
                        // being issued and the extra live values cost spill
                        // reloads. L6 regressed too: 916,724,493 -> 921,418,528
                        // Ir (+0.51%).
                        //
                        // The lesson is general and cost this project two reverts
                        // in one session: SOURCE-LEVEL LOAD COUNT IS NOT MACHINE-
                        // LEVEL LOAD COUNT. The 1.30x load ratio against libdeflate
                        // is real and measured (perf stat, worst cell), but it
                        // cannot be attacked by reading this function and deleting
                        // loads that look redundant — the compiler has already
                        // taken those. Any future attempt here must show a Dr
                        // DECREASE, not a source-level one.
                        //
                        // Prefilter: compare the last 4 and the first 4 bytes before
                        // attempting a full extension.
                        let off = best_len as usize - 3;
                        // SAFETY: `matchptr < in_next` (cutoff guard). `off = best_len-3`
                        // with `best_len <= max_len`, so `in_next + off + 4 <=
                        // in_next + max_len + 1 <= in_end + 1 < buf.len()` (BUF_PAD>=16),
                        // and `matchptr + off + 4 < in_next + off + 4` likewise in bounds.
                        // These are exactly the four addresses the `load_u32(base, idx)`
                        // form produced; only the ADDRESSING changed (see `mp_base`).
                        let (m_hi, n_hi, m_lo, n_lo) = unsafe {
                            debug_assert!(matchptr < in_next);
                            debug_assert!(matchptr + off + 4 <= blen && in_next + off + 4 <= blen);
                            let mp = mp_base.wrapping_offset(cur_node4 as isize);
                            (
                                u32::from_le(core::ptr::read_unaligned(mp.add(off) as *const u32)),
                                u32::from_le(core::ptr::read_unaligned(
                                    in_ptr.add(off) as *const u32
                                )),
                                u32::from_le(core::ptr::read_unaligned(mp as *const u32)),
                                u32::from_le(core::ptr::read_unaligned(in_ptr as *const u32)),
                            )
                        };
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.attempts += 1;
                        }
                        if m_hi == n_hi && m_lo == n_lo {
                            break;
                        }
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.miss += 1;
                        }
                        cur_node4 = next_node;
                        if (cur_node4 as i32) <= cutoff {
                            break 'search;
                        }
                        depth_remaining -= 1;
                        if depth_remaining == 0 {
                            break 'search;
                        }
                        // SAFETY: masked chain index `< next_tab.len()`.
                        next_node = unsafe {
                            *self
                                .next_tab
                                .get_unchecked((cur_node4 as u16 & WINDOW_MASK) as usize)
                        };
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.chain_reads += 1;
                        }
                    }

                    let len = lz_extend(buf, in_next, matchptr, 4, max_len);
                    if len > best_len {
                        best_len = len;
                        best_matchptr = matchptr;
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.accepted += 1;
                        }
                        if best_len >= nice_len {
                            break 'search;
                        }
                    } else {
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.too_short += 1;
                        }
                    }
                    // Advance to the next node — already loaded by the pipeline.
                    cur_node4 = next_node;
                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    depth_remaining -= 1;
                    if depth_remaining == 0 {
                        break 'search;
                    }
                    // SAFETY: masked chain index `< next_tab.len()`.
                    next_node = unsafe {
                        *self
                            .next_tab
                            .get_unchecked((cur_node4 as u16 & WINDOW_MASK) as usize)
                    };
                    #[cfg(feature = "anatomy-counters")]
                    {
                        local.chain_reads += 1;
                    }
                }
            }

            #[cfg(feature = "anatomy-counters")]
            local.flush();
            (best_len, (in_next - best_matchptr) as u32)
        } // #[cfg(not(feature = "bucket-oracle-null-mf"))]
    }

    /// `hc_matchfinder_skip_bytes`: insert `count` positions without searching.
    ///
    /// Advances the matchfinder over `[in_next, in_next + count)`, updating
    /// `next_hashes` to the hashes for `in_next + count`. No-op if there is not
    /// enough lookahead (`count + 5 > in_end - in_next`), matching the vendor.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn skip_bytes(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        in_end: usize,
        count: usize,
        next_hashes: &mut [u32; 2],
    ) {
        if count + 5 > in_end - in_next {
            return;
        }
        crate::anatomy_count!(hc_positions_skipped, count);
        let base = buf.as_ptr();
        let blen = buf.len();
        let mut in_next = in_next;
        let mut cur_pos = in_next - *in_base;
        let mut hash3 = next_hashes[0] as usize;
        let mut hash4 = next_hashes[1] as usize;
        let mut remaining = count;
        // FALSIFY — hoisting this wrap test out of the loop is a TRAP. It looks
        // free: the test costs 2 Ir on every position but can only fire once per
        // WINDOW_SIZE (32768) of them, which is 12.5M Ir, 2.26% of the whole
        // program, at L2 on 8 MB of silesia. Two ways of removing it were built
        // and measured on the shipped build shape (lto=fat, cgu=1); BOTH lost.
        //
        //                        L2 Ir                 L6 Ir
        //   this loop        555,127,066            916,724,493
        //   chunked runs     541,600,498 (-2.44%)   920,416,350 (+0.40%)
        //   one test/call    552,888,310 (-0.40%)   925,234,382 (+0.93%)
        //
        // "chunked runs" = `while remaining > 0 { run = min(remaining,
        // WINDOW_SIZE - cur_pos); ... }`. On frozen Zen2 it measured wall 0.9765
        // at L2 (RESOLVED) but 1.0103 at L6 and 1.0070 at L9 — both RESOLVED
        // SLOWER. "one test/call" hoists the test to a single per-call check
        // with a macro'd body; it is worse than chunking at BOTH levels, because
        // duplicating the body costs more than the branch it removes.
        //
        // Do not retry either without an explanation for the DEEP-level loss.
        // Mine was "lazy skips in short runs so the outer loop cannot amortise"
        // — and that is WRONG: lazy runs a larger nice_len (65 vs 10), so its
        // skips are LONGER, and chunking should have helped L6 more, not less.
        // The regression is real (Ir is deterministic) but unexplained; suspect
        // inlining/layout, since this body inlines into both greedy's and lazy's
        // run_block. Measure Ir at L2 AND a deep level before believing any fix:
        // measuring only L2 and generalising is exactly what produced the
        // regression above.
        loop {
            if cur_pos == WINDOW_SIZE {
                self.slide_window();
                *in_base += WINDOW_SIZE;
                cur_pos = 0;
            }
            // SAFETY: `hash3 < HASH3_SIZE`, `hash4 < HASH4_SIZE` (lz_hash outputs),
            // and `cur_pos ∈ 0..WINDOW_SIZE == next_tab.len()` (reset above).
            unsafe {
                debug_assert!(hash3 < HASH3_SIZE && hash4 < HASH4_SIZE && cur_pos < WINDOW_SIZE);
                *self.hash3_tab.get_unchecked_mut(hash3) = cur_pos as i16;
                *self.next_tab.get_unchecked_mut(cur_pos) = *self.hash4_tab.get_unchecked(hash4);
                *self.hash4_tab.get_unchecked_mut(hash4) = cur_pos as i16;
            }

            in_next += 1;
            // SAFETY: the `count + 5 > in_end - in_next` guard proves
            // `in_next + count + 5 <= in_end`; here `in_next <= start + count`, so
            // `in_next + 4 <= in_end <= buf.len()`.
            let next_hashseq = unsafe {
                debug_assert!(in_next + 4 <= blen);
                load_u32(base, in_next)
            };
            hash3 = lz_hash(next_hashseq & 0xFF_FFFF, HC_HASH3_ORDER) as usize;
            hash4 = lz_hash(next_hashseq, HC_HASH4_ORDER) as usize;
            cur_pos += 1;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        // Vendor `prefetchw` (hc_matchfinder.h:395-396): warm the buckets for the
        // final position in an exclusive state. Pure hint; no effect on state.
        // SAFETY: `hash3 < HASH3_SIZE` and `hash4 < HASH4_SIZE` (lz_hash outputs).
        unsafe {
            prefetch_write(self.hash3_tab.as_ptr().add(hash3) as *const u8);
            prefetch_write(self.hash4_tab.as_ptr().add(hash4) as *const u8);
        }
        next_hashes[0] = hash3 as u32;
        next_hashes[1] = hash4 as u32;
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

    // ====================================================================
    // Increment-5 matches-equal-scalar reference (the load-bearing net).
    //
    // `RefHc` is a verbatim copy of the pre-Increment-5 CHECKED matchfinder
    // (bounds-checked slice loads, indexed table access). Increment 5 rewrote
    // `HcMatchfinder` with unsafe raw-pointer / unchecked codegen whose ONLY
    // permitted effect is codegen — it must find byte-identical matches and
    // leave byte-identical table state. `matches_equal_scalar_*` drives BOTH
    // over identical inputs and asserts identical `(best_len, offset)` at every
    // position AND identical `hash3_tab`/`hash4_tab`/`next_tab`/`in_base`/
    // `next_hashes` after every op. If they ever diverge, the rewrite is wrong.
    // ====================================================================

    /// Checked unaligned little-endian 4-byte load (pre-Inc5 form).
    fn ref_load_u32(buf: &[u8], pos: usize) -> u32 {
        u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
    }

    /// Verbatim pre-Increment-5 (checked) reference matchfinder.
    struct RefHc {
        hash3_tab: Vec<i16>,
        hash4_tab: Vec<i16>,
        next_tab: Vec<i16>,
    }

    impl RefHc {
        fn new() -> Self {
            RefHc {
                hash3_tab: vec![MATCHFINDER_INITVAL; HASH3_SIZE],
                hash4_tab: vec![MATCHFINDER_INITVAL; HASH4_SIZE],
                next_tab: vec![MATCHFINDER_INITVAL; WINDOW_SIZE],
            }
        }

        fn slide_window(&mut self) {
            matchfinder_rebase(&mut self.hash3_tab);
            matchfinder_rebase(&mut self.hash4_tab);
            matchfinder_rebase(&mut self.next_tab);
        }

        #[allow(clippy::too_many_arguments)]
        fn longest_match(
            &mut self,
            buf: &[u8],
            in_base: &mut usize,
            in_next: usize,
            best_len_in: u32,
            max_len: u32,
            nice_len: u32,
            max_search_depth: u32,
            next_hashes: &mut [u32; 2],
        ) -> (u32, u32) {
            let mut best_len = best_len_in;
            let mut depth_remaining = max_search_depth;
            let mut best_matchptr = in_next;

            let mut cur_pos = in_next - *in_base;
            if cur_pos == WINDOW_SIZE {
                self.slide_window();
                *in_base += WINDOW_SIZE;
                cur_pos = 0;
            }
            let in_base_v = *in_base;
            let cutoff: i32 = cur_pos as i32 - WINDOW_SIZE as i32;

            if max_len < 5 {
                return (best_len, (in_next - best_matchptr) as u32);
            }

            let hash3 = next_hashes[0] as usize;
            let hash4 = next_hashes[1] as usize;

            let cur_node3 = self.hash3_tab[hash3];
            let mut cur_node4 = self.hash4_tab[hash4];

            self.hash3_tab[hash3] = cur_pos as i16;
            self.hash4_tab[hash4] = cur_pos as i16;
            self.next_tab[cur_pos] = cur_node4;

            let next_hashseq = ref_load_u32(buf, in_next + 1);
            next_hashes[0] = lz_hash(next_hashseq & 0xFF_FFFF, HC_HASH3_ORDER);
            next_hashes[1] = lz_hash(next_hashseq, HC_HASH4_ORDER);

            let seq4 = ref_load_u32(buf, in_next);
            let mut matchptr;

            'search: {
                if best_len < 4 {
                    if (cur_node3 as i32) <= cutoff {
                        break 'search;
                    }
                    if best_len < 3 {
                        let mp = (in_base_v as isize + cur_node3 as isize) as usize;
                        if ref_load_u32(buf, mp) & 0xFF_FFFF == seq4 & 0xFF_FFFF {
                            best_len = 3;
                            best_matchptr = mp;
                        }
                    }

                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    loop {
                        matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                        if ref_load_u32(buf, matchptr) == seq4 {
                            break;
                        }
                        cur_node4 = self.next_tab[(cur_node4 as u16 & WINDOW_MASK) as usize];
                        if (cur_node4 as i32) <= cutoff {
                            break 'search;
                        }
                        depth_remaining -= 1;
                        if depth_remaining == 0 {
                            break 'search;
                        }
                    }

                    best_matchptr = matchptr;
                    best_len = lz_extend(buf, in_next, matchptr, 4, max_len);
                    if best_len >= nice_len {
                        break 'search;
                    }
                    cur_node4 = self.next_tab[(cur_node4 as u16 & WINDOW_MASK) as usize];
                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    depth_remaining -= 1;
                    if depth_remaining == 0 {
                        break 'search;
                    }
                } else {
                    if (cur_node4 as i32) <= cutoff || best_len >= nice_len {
                        break 'search;
                    }
                }

                loop {
                    loop {
                        matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                        let off = best_len as usize - 3;
                        if ref_load_u32(buf, matchptr + off) == ref_load_u32(buf, in_next + off)
                            && ref_load_u32(buf, matchptr) == ref_load_u32(buf, in_next)
                        {
                            break;
                        }
                        cur_node4 = self.next_tab[(cur_node4 as u16 & WINDOW_MASK) as usize];
                        if (cur_node4 as i32) <= cutoff {
                            break 'search;
                        }
                        depth_remaining -= 1;
                        if depth_remaining == 0 {
                            break 'search;
                        }
                    }

                    let len = lz_extend(buf, in_next, matchptr, 4, max_len);
                    if len > best_len {
                        best_len = len;
                        best_matchptr = matchptr;
                        if best_len >= nice_len {
                            break 'search;
                        }
                    }
                    cur_node4 = self.next_tab[(cur_node4 as u16 & WINDOW_MASK) as usize];
                    if (cur_node4 as i32) <= cutoff {
                        break 'search;
                    }
                    depth_remaining -= 1;
                    if depth_remaining == 0 {
                        break 'search;
                    }
                }
            }

            (best_len, (in_next - best_matchptr) as u32)
        }

        #[allow(clippy::too_many_arguments)]
        fn skip_bytes(
            &mut self,
            buf: &[u8],
            in_base: &mut usize,
            in_next: usize,
            in_end: usize,
            count: usize,
            next_hashes: &mut [u32; 2],
        ) {
            if count + 5 > in_end - in_next {
                return;
            }
            let mut in_next = in_next;
            let mut cur_pos = in_next - *in_base;
            let mut hash3 = next_hashes[0] as usize;
            let mut hash4 = next_hashes[1] as usize;
            let mut remaining = count;
            loop {
                if cur_pos == WINDOW_SIZE {
                    self.slide_window();
                    *in_base += WINDOW_SIZE;
                    cur_pos = 0;
                }
                self.hash3_tab[hash3] = cur_pos as i16;
                self.next_tab[cur_pos] = self.hash4_tab[hash4];
                self.hash4_tab[hash4] = cur_pos as i16;

                in_next += 1;
                let next_hashseq = ref_load_u32(buf, in_next);
                hash3 = lz_hash(next_hashseq & 0xFF_FFFF, HC_HASH3_ORDER) as usize;
                hash4 = lz_hash(next_hashseq, HC_HASH4_ORDER) as usize;
                cur_pos += 1;
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
            next_hashes[0] = hash3 as u32;
            next_hashes[1] = hash4 as u32;
        }
    }

    /// Assert the new (unsafe) matchfinder and the checked reference stay in
    /// perfect lockstep — identical returns AND identical table state — while
    /// driving both with the greedy parser's exact call pattern (longest_match
    /// at every position; `skip_bytes(len-1)` after each accepted match). This
    /// mirrors production usage, so it exercises window slides, near-EOF
    /// `max_len < 5`, and deep chains exactly as the encoder does.
    fn assert_lockstep(data: &[u8]) {
        let buf = padded(data);
        let in_end = data.len();
        let max_search_depth = 32u32;
        let nice = 258u32;

        let mut mf_new = HcMatchfinder::new();
        let mut mf_ref = RefHc::new();
        let (mut base_new, mut base_ref) = (0usize, 0usize);
        let (mut nh_new, mut nh_ref) = ([0u32; 2], [0u32; 2]);

        // Exercise the preset-dictionary skip path first (a short seed skip),
        // then the greedy walk over the rest.
        let seed = (in_end / 4).min(37);
        if seed > 0 {
            mf_new.skip_bytes(&buf, &mut base_new, 0, in_end, seed, &mut nh_new);
            mf_ref.skip_bytes(&buf, &mut base_ref, 0, in_end, seed, &mut nh_ref);
            assert_state_eq(
                &mf_new,
                &mf_ref,
                base_new,
                base_ref,
                &nh_new,
                &nh_ref,
                "after seed",
            );
        }

        let mut in_next = seed;
        while in_next < in_end {
            let max_len = (in_end - in_next).min(258) as u32;
            let nice_len = nice.min(max_len);
            let (l_new, o_new) = mf_new.longest_match(
                &buf,
                &mut base_new,
                in_next,
                2,
                max_len,
                nice_len,
                max_search_depth,
                &mut nh_new,
            );
            let (l_ref, o_ref) = mf_ref.longest_match(
                &buf,
                &mut base_ref,
                in_next,
                2,
                max_len,
                nice_len,
                max_search_depth,
                &mut nh_ref,
            );
            // Offset is only meaningful when a real match was found (len > 2);
            // compare the meaningful pair. (Both impls also return the same raw
            // best_matchptr, so compare raw too when a match exists.)
            assert_eq!(l_new, l_ref, "best_len diverged at pos {in_next}");
            if l_new > 2 {
                assert_eq!(
                    o_new, o_ref,
                    "offset diverged at pos {in_next} (len {l_new})"
                );
            }
            assert_state_eq(
                &mf_new,
                &mf_ref,
                base_new,
                base_ref,
                &nh_new,
                &nh_ref,
                "after longest_match",
            );

            // Greedy acceptance: len>=4 (or len==3 with small offset) => take it
            // and skip the interior; else emit a literal. The exact predicate
            // doesn't matter for equivalence — both sides follow the SAME branch
            // because their returns are identical — it only shapes coverage.
            if l_new >= 4 || (l_new == 3 && o_new <= 4096) {
                let len = l_new as usize;
                mf_new.skip_bytes(
                    &buf,
                    &mut base_new,
                    in_next + 1,
                    in_end,
                    len - 1,
                    &mut nh_new,
                );
                mf_ref.skip_bytes(
                    &buf,
                    &mut base_ref,
                    in_next + 1,
                    in_end,
                    len - 1,
                    &mut nh_ref,
                );
                assert_state_eq(
                    &mf_new,
                    &mf_ref,
                    base_new,
                    base_ref,
                    &nh_new,
                    &nh_ref,
                    "after skip_bytes",
                );
                in_next += len;
            } else {
                in_next += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_state_eq(
        mf_new: &HcMatchfinder,
        mf_ref: &RefHc,
        base_new: usize,
        base_ref: usize,
        nh_new: &[u32; 2],
        nh_ref: &[u32; 2],
        ctx: &str,
    ) {
        assert_eq!(base_new, base_ref, "in_base diverged {ctx}");
        assert_eq!(nh_new, nh_ref, "next_hashes diverged {ctx}");
        assert!(
            mf_new.hash3_tab[..] == mf_ref.hash3_tab[..],
            "hash3_tab diverged {ctx}"
        );
        assert!(
            mf_new.hash4_tab[..] == mf_ref.hash4_tab[..],
            "hash4_tab diverged {ctx}"
        );
        assert!(
            mf_new.next_tab[..] == mf_ref.next_tab[..],
            "next_tab diverged {ctx}"
        );
    }

    #[test]
    fn matches_equal_scalar_corner_cases() {
        // All-same-byte: deepest possible chains (every position hashes alike).
        assert_lockstep(&vec![0x5Au8; 70_000]);
        // Incompressible / high-entropy: forces chain walks that miss.
        let incompressible: Vec<u8> = (0..40_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        assert_lockstep(&incompressible);
        // Window-slide straddle: a repeat longer than the 32 KiB window so
        // matches reference positions across a slide + rebase.
        let mut straddle = Vec::new();
        let unit: Vec<u8> = (0..251u32).map(|i| i as u8).collect();
        while straddle.len() < 80_000 {
            straddle.extend_from_slice(&unit);
        }
        assert_lockstep(&straddle);
        // Near-EOF short inputs exercise the `max_len < 5` early-return.
        for n in 0..12usize {
            assert_lockstep(&vec![0xABu8; n]);
        }
        // Mixed runs + literals.
        let mut mixed = Vec::new();
        for i in 0..5000u32 {
            if i % 7 == 0 {
                mixed.extend_from_slice(&[i as u8; 20]);
            } else {
                mixed.push((i.wrapping_mul(48271) >> 16) as u8);
            }
        }
        assert_lockstep(&mixed);
    }

    #[test]
    fn matches_equal_scalar_silesia() {
        // ~400 KiB slice slides the window a dozen+ times with real matches.
        let path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar");
        let Ok(mut f) = std::fs::File::open(&path) else {
            eprintln!("note: {} missing; skipped silesia lockstep", path.display());
            return;
        };
        use std::io::{Read, Seek};
        let mut data = vec![0u8; 400 * 1024];
        f.seek(std::io::SeekFrom::Start(1 << 16)).unwrap();
        if f.read_exact(&mut data).is_err() {
            eprintln!("note: silesia.tar too small; skipped");
            return;
        }
        assert_lockstep(&data);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

        /// The rewrite must find byte-identical matches on ANY input — random
        /// bytes, runs, repeats, and near-boundary lengths.
        #[test]
        fn matches_equal_scalar_proptest(data in gen_input()) {
            assert_lockstep(&data);
        }
    }

    /// Adversarial byte-vector generator: interleaves random bytes, byte-runs,
    /// and repeated blocks (redundancy => populated chains + real matches),
    /// across lengths that straddle the `max_len < 5` gate and small windows.
    fn gen_input() -> impl proptest::strategy::Strategy<Value = Vec<u8>> {
        use proptest::prelude::*;
        let chunk = prop_oneof![
            // A run of one byte (deep chains / long matches).
            (any::<u8>(), 1usize..40).prop_map(|(b, n)| vec![b; n]),
            // Random bytes (chain misses).
            proptest::collection::vec(any::<u8>(), 1..24),
            // A short repeated motif (medium-offset matches).
            (proptest::collection::vec(any::<u8>(), 2..6), 1usize..8).prop_map(|(seed, reps)| seed
                .iter()
                .cloned()
                .cycle()
                .take(seed.len() * reps)
                .collect()),
        ];
        proptest::collection::vec(chunk, 0..64)
            .prop_map(|chunks| chunks.into_iter().flatten().collect::<Vec<u8>>())
    }

    #[test]
    fn finds_simple_repeat() {
        // "abcdefabcdef..." repeated. Position 0 lands in the {0,0} bootstrap
        // bucket (vendor behavior), so we search at position 12, which should
        // match the correctly-hashed position 6 at offset 6.
        let data: Vec<u8> = b"abcdef".repeat(8); // 48 bytes
        let buf = padded(&data);
        let in_end = data.len();
        let mut mf = HcMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hashes = [0u32; 2];

        // Seed positions 0..12 so the chains are populated (1..11 hashed
        // correctly; next_hashes ends pointing at position 12).
        mf.skip_bytes(&buf, &mut in_base, 0, in_end, 12, &mut next_hashes);

        let (len, off) = mf.longest_match(
            &buf,
            &mut in_base,
            12,
            2, // best_len_in = min_len - 1
            (in_end - 12) as u32,
            258,
            32,
            &mut next_hashes,
        );
        assert!(len >= 4, "expected a match, got len {len}");
        assert_eq!(off, 6, "expected offset 6, got {off}");
        // The matched bytes must actually equal the source.
        for i in 0..len as usize {
            assert_eq!(buf[12 + i], buf[12 - off as usize + i]);
        }
    }

    #[test]
    fn no_match_on_unique_data() {
        let data: Vec<u8> = (0..200u32).map(|i| (i * 37) as u8).collect();
        let buf = padded(&data);
        let in_end = data.len();
        let mut mf = HcMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hashes = [0u32; 2];
        // Walk a few positions; distinct byte stream => no length-4 match found.
        let mut pos = 0usize;
        let mut found_any = false;
        while pos + 8 < in_end {
            let (len, _off) = mf.longest_match(
                &buf,
                &mut in_base,
                pos,
                2,
                (in_end - pos) as u32,
                258,
                32,
                &mut next_hashes,
            );
            if len >= 3 {
                found_any = true;
            }
            pos += 1;
        }
        assert!(!found_any, "unique data should yield no matches");
    }
}
