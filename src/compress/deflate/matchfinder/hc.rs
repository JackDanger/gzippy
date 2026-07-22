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
    load_u24, load_u32, lz_extend, lz_hash, matchfinder_rebase, prefetch_read, prefetch_write,
    MATCHFINDER_INITVAL,
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
                loop {
                    matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                    // Prefetch the next node's match data one iteration ahead.
                    // `wrapping_offset` keeps pointer formation defined even when
                    // `next_node <= cutoff` (a chain end maps off-object); the
                    // prefetch itself never faults.
                    prefetch_read(base.wrapping_offset(in_base_v as isize + next_node as isize));
                    // SAFETY: `cutoff < cur_node4` so `matchptr < in_next`, thus
                    // `matchptr + 4 <= in_next + 4 <= buf.len()`.
                    let cand = unsafe {
                        debug_assert!(matchptr < in_next && matchptr + 4 <= blen);
                        load_u32(base, matchptr)
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

                // Found a length-4 match; extend it fully.
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
            loop {
                loop {
                    matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                    // Prefetch the next node's match data one iteration ahead
                    // (see the length-4 walk for the correctness argument).
                    prefetch_read(base.wrapping_offset(in_base_v as isize + next_node as isize));
                    // Prefilter: compare the last 4 and the first 4 bytes before
                    // attempting a full extension.
                    let off = best_len as usize - 3;
                    // SAFETY: `matchptr < in_next` (cutoff guard). `off = best_len-3`
                    // with `best_len <= max_len`, so `in_next + off + 4 <=
                    // in_next + max_len + 1 <= in_end + 1 < buf.len()` (BUF_PAD>=16),
                    // and `matchptr + off + 4 < in_next + off + 4` likewise in bounds.
                    let (m_hi, n_hi, m_lo, n_lo) = unsafe {
                        debug_assert!(matchptr < in_next);
                        debug_assert!(matchptr + off + 4 <= blen && in_next + off + 4 <= blen);
                        (
                            load_u32(base, matchptr + off),
                            load_u32(base, in_next + off),
                            load_u32(base, matchptr),
                            load_u32(base, in_next),
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
