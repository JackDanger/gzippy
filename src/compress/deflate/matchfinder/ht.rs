//! Hash-table matchfinder with 2-entry inline buckets (`ht_matchfinder`).
//!
//! Faithful port of `vendor/libdeflate/lib/ht_matchfinder.h` (Eric Biggers,
//! 2022) — `ht_matchfinder_init` / `_slide_window` / `_longest_match` /
//! `_skip_bytes`, at `HT_MATCHFINDER_BUCKET_SIZE == 2`, which is the
//! hand-unrolled arm libdeflate actually ships at level 1.
//!
//! # Why this exists, and why it is not another knob
//!
//! `parse::fast`'s fused finder is CHAINLESS SINGLE-PROBE: one `u32` head table
//! of 64 K entries (256 KiB) plus a separate 3-byte-keyed `head3` side table
//! (128 KiB), one candidate examined per position. libdeflate's level 1 instead
//! keeps the chain INLINE in the bucket: `[[i16; 2]; 1 << 15]` — **128 KiB
//! total, two candidates per position, and no length-3 table at all.**
//!
//! The measured consequence of having one candidate instead of two, from
//! `fulcrum why libdeflate:data.csv:L1:T1:size` (2026-07-30, the automated
//! vendor diff, structure layer):
//!
//! | | ours | libdeflate |
//! |---|---|---|
//! | matches | 1,846,129 | **1,930,665** (+4.58%) |
//! | literals | **741,183** | 256,099 (**we emit 189% more**) |
//! | input covered by matches | 97.20% | **99.03%** |
//! | header bits | 166,614 | 166,000 |
//!
//! Headers are within 0.4% of each other, so this is NOT block sizing and NOT
//! table quality — the diff's own verdict is "POSITION COUNTS DIFFER: different
//! parse decisions — the gap is ALGORITHMIC". We find slightly LONGER matches on
//! average (13.95 B vs 13.59 B) while covering LESS input, which is the exact
//! signature of a single-probe finder taking the first candidate its hash offers
//! and missing the better one a second way would have held.
//!
//! # REOPEN of the bucket2 falsification, and why this is a different mechanism
//!
//! `c0f69036` recorded a FROZEN ship gate on `17283ee6` ("insert-depth=8 +
//! bucket2(gate=64)"): **SIZE PASSED cleanly** — "a genuine, confirmed, LARGE
//! ratio-only win" — and WALL FAILED DECISIVELY, a 12-29% self-tax at ~26
//! standard deviations. It was reverted (`158af467`). That record names its own
//! reopen condition verbatim:
//!
//! > "Not re-attempted without either (1) **a materially cheaper way to get the
//! > same length-3-8 reach / second-candidate signal**, or (2) a re-scoped ship
//! > rule that accepts size-only status."
//!
//! This is condition (1), and the mechanism is REPLACE rather than ADD — which
//! is precisely what the falsified attempt did not do:
//!
//! | | falsified `17283ee6` | this port |
//! |---|---|---|
//! | second candidate | ADDED, gated at 64 | inline in the bucket, ungated |
//! | `head` 64 K × u32 | kept (256 KiB) | replaced |
//! | `head3` side table | kept (128 KiB) | **deleted — no length-3 table** |
//! | insert depth | raised to 8 | unchanged (1 insert/position) |
//! | working set | ~384 KiB **and growing** | **128 KiB, 3x smaller** |
//!
//! The falsified version paid for a second probe ON TOP OF an already-large
//! two-table working set and deepened inserts as well; its self-tax is exactly
//! what that predicts. Three times less memory touched per position, with one
//! probe site instead of two different-key probes, is a different mechanism and
//! not "another threshold sweep on this same shape". It is also NOT a content
//! detector: the second bucket entry is read unconditionally at every position,
//! so there is no data-dependent branch to gate and nothing to tune — which is
//! the point, and why it can retire `L1_HASH3_GATE_LIT_THRESHOLD_PCT` (a
//! constant fitted two points off one file's cliff) rather than join it.
//!
//! **The wall leg is still the binding risk and is NOT claimed here.** Size is
//! deterministic and free, so per `CLAUDE.md` (cheapest falsifier first) it is
//! measured first; a size win obliges a frozen paired wall run on solvency
//! before anything ships, and the prior falsification is the reason to expect
//! that leg to be hard rather than to assume the smaller table wins it.

use super::common::{
    load_u32, lz_extend, lz_hash, matchfinder_rebase, prefetch_write, MATCHFINDER_INITVAL,
    MATCHFINDER_WINDOW_SIZE,
};

/// `HT_MATCHFINDER_HASH_ORDER`.
pub const HT_HASH_ORDER: u32 = 15;
/// `HT_MATCHFINDER_BUCKET_SIZE`. The port below is the hand-unrolled `== 2` arm;
/// changing this constant alone does NOT change the algorithm.
pub const HT_BUCKET_SIZE: usize = 2;
/// Number of buckets.
pub const HT_TAB_LEN: usize = 1 << HT_HASH_ORDER;
/// `HT_MATCHFINDER_MIN_MATCH_LEN`. Asserted throughout the port, exactly as
/// libdeflate's `STATIC_ASSERT` does: the 4-byte `seq` compare depends on it.
pub const HT_MIN_MATCH_LEN: u32 = 4;
/// `HT_MATCHFINDER_REQUIRED_NBYTES` — minimum `max_len` for [`HtMatchfinder::longest_match`].
pub const HT_REQUIRED_NBYTES: u32 = 5;

/// `struct ht_matchfinder`. 128 KiB of inline `[i16; 2]` buckets.
///
/// Inline fixed-size array rather than a `Vec`, for the same reason
/// [`super::hc::HcMatchfinder`] does it: a bucket read becomes `self +
/// const_offset + i*4` in one addressing mode with no table-base register tied
/// up. The struct is >64 KiB, so it is always heap-boxed and never constructed
/// or passed by value.
pub struct HtMatchfinder {
    hash_tab: [[i16; HT_BUCKET_SIZE]; HT_TAB_LEN],
}

thread_local! {
    /// One recycled `HtMatchfinder` per thread — see [`HtMatchfinder::acquire`].
    /// Thread-local rather than shared: no cross-thread mutable state, no
    /// synchronization, and T>1 chunk independence is preserved because every
    /// chunk still starts from a fully re-armed table.
    static HT_POOL: std::cell::RefCell<Option<Box<HtMatchfinder>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII handle from [`HtMatchfinder::acquire`]; returns its box to [`HT_POOL`]
/// on drop instead of freeing it.
pub struct PooledHt(Option<Box<HtMatchfinder>>);

impl std::ops::Deref for PooledHt {
    type Target = HtMatchfinder;
    #[inline]
    fn deref(&self) -> &HtMatchfinder {
        // SAFETY/invariant: `Some` for a `PooledHt`'s whole lifetime outside
        // `Drop::drop`, which is the only place it is taken.
        self.0.as_deref().expect("PooledHt used after drop")
    }
}

impl std::ops::DerefMut for PooledHt {
    #[inline]
    fn deref_mut(&mut self) -> &mut HtMatchfinder {
        self.0.as_deref_mut().expect("PooledHt used after drop")
    }
}

impl Drop for PooledHt {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            HT_POOL.with(|cell| {
                *cell.borrow_mut() = Some(b);
            });
        }
    }
}

impl HtMatchfinder {
    /// `ht_matchfinder_init`: allocate and set every entry to the sentinel.
    ///
    /// Built through `Box::new_uninit` so no 128 KiB temporary lands on the
    /// stack; every entry is written before `assume_init`.
    pub fn new() -> Box<Self> {
        let mut boxed = Box::<Self>::new_uninit();
        // SAFETY: `new_uninit` gives one aligned, fully-owned `HtMatchfinder`.
        // Every `i16` of the only field is written with `MATCHFINDER_INITVAL`
        // (the exact value `matchfinder_init` writes — a `-WINDOW_SIZE`
        // sentinel, NOT zero) before `assume_init`. `addr_of_mut!` avoids
        // forming a reference to uninit memory, and each `.add(i)` for
        // `i < HT_TAB_LEN * HT_BUCKET_SIZE` stays inside the field.
        unsafe {
            let p = boxed.as_mut_ptr();
            let tab = core::ptr::addr_of_mut!((*p).hash_tab) as *mut i16;
            for i in 0..(HT_TAB_LEN * HT_BUCKET_SIZE) {
                tab.add(i).write(MATCHFINDER_INITVAL);
            }
            boxed.assume_init()
        }
    }

    /// Re-arm every entry in place, reusing the existing 128 KiB allocation.
    /// Same postcondition as [`Self::new`].
    fn reset(&mut self) {
        // One flat fill: `[[i16; 2]; N]` is contiguous, so this is a single
        // memset-shaped loop rather than N two-element fills.
        let flat: &mut [i16] = self.as_flat_mut();
        flat.fill(MATCHFINDER_INITVAL);
    }

    /// The bucket array viewed as one contiguous `[i16]`, for `fill`/rebase.
    #[inline]
    fn as_flat_mut(&mut self) -> &mut [i16] {
        // SAFETY: `[[i16; HT_BUCKET_SIZE]; HT_TAB_LEN]` is a contiguous array
        // of `i16` with no padding (arrays are laid out contiguously and `i16`
        // has no alignment slack inside `[i16; 2]`), so reinterpreting it as
        // `[i16; HT_TAB_LEN * HT_BUCKET_SIZE]` is layout-identical. The length
        // is exactly the element count, and the borrow is unique.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.hash_tab.as_mut_ptr() as *mut i16,
                HT_TAB_LEN * HT_BUCKET_SIZE,
            )
        }
    }

    /// Pooled equivalent of [`Self::new`] — a table reset to its sentinel
    /// state, reusing this thread's allocation when it already has one.
    ///
    /// `run()` is invoked once per CHUNK on the T>1 path and chunk count
    /// exceeds thread count, so without pooling every chunk would pay a fresh
    /// 128 KiB allocation plus sentinel fill. That per-`run()` cadence is a
    /// documented waste class in this codebase (DHAT: "1x / 9-28x = chunk
    /// count"); this avoids re-introducing it.
    pub fn acquire() -> PooledHt {
        let existing = HT_POOL.with(|cell| cell.borrow_mut().take());
        let mut mf = existing.unwrap_or_else(Self::new);
        mf.reset();
        PooledHt(Some(mf))
    }

    /// `ht_matchfinder_slide_window`: rebase every stored position by one window.
    #[inline]
    fn slide_window(&mut self) {
        matchfinder_rebase(self.as_flat_mut());
    }

    /// `ht_matchfinder_longest_match`. Returns `(best_len, offset)`; `best_len ==
    /// 0` means no match and the offset is meaningless.
    ///
    /// Faithful to the `BUCKET_SIZE == 2` hand-unrolled arm, including its two
    /// deliberate quirks, both of which are load-bearing for byte-for-byte
    /// agreement with libdeflate's parse decisions:
    ///   * entry 0 is copied to entry 1 **even when `nice_len` is reached on the
    ///     first candidate** (libdeflate: "this can be slightly faster");
    ///   * the second candidate is pre-screened by comparing the 4 bytes ending
    ///     at `best_len` BEFORE calling `lz_extend`, so a candidate that cannot
    ///     beat the incumbent is rejected without extending.
    ///
    /// Contract, inherited from the C: `max_len >= HT_REQUIRED_NBYTES` (5), and
    /// `buf` is padded so the 4-byte loads at `in_next + 1` and at
    /// `in_next + best_len - 3` stay in bounds.
    #[inline(always)]
    pub fn longest_match(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        max_len: u32,
        nice_len: u32,
        next_hash: &mut u32,
    ) -> (u32, u32) {
        debug_assert!(max_len >= HT_REQUIRED_NBYTES);
        debug_assert_eq!(HT_MIN_MATCH_LEN, 4);

        let mut best_len: u32 = 0;
        let mut best_match_pos = in_next;
        let mut cur_pos = (in_next - *in_base) as i32;

        if cur_pos == MATCHFINDER_WINDOW_SIZE {
            self.slide_window();
            *in_base += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos = 0;
        }
        let in_base_now = *in_base;
        let cutoff = cur_pos - MATCHFINDER_WINDOW_SIZE;

        // SAFETY for every `load_u32` below. `load_u32(base, off)` reads 4 bytes
        // at `off`; all offsets used here are `in_next`, `in_next + 1`,
        // `in_next + best_len - 3`, `match_pos`, and `match_pos + best_len - 3`.
        //   * `in_next + 1 + 4 <= buf.len()` because the caller guarantees
        //     `max_len >= HT_REQUIRED_NBYTES == 5` bytes remain at `in_next`,
        //     and `buf` additionally carries `BUF_PAD` (16) trailing pad bytes.
        //   * `best_len <= max_len`, so `in_next + best_len` is in bounds and
        //     `in_next + best_len - 3 + 4 = in_next + best_len + 1` is too. The
        //     `- 3` cannot underflow: this expression is only evaluated after
        //     `best_len` was set by `lz_extend` starting at `HT_MIN_MATCH_LEN`
        //     (4), so `best_len >= 4`.
        //   * `match_pos = in_base_now + cur_node` with `cur_node > cutoff`, and
        //     it is a position we previously inserted, hence `< in_next`; the
        //     same pad argument therefore covers `match_pos + best_len + 1`.
        let base = buf.as_ptr();
        let seq = unsafe { load_u32(base, in_next) };
        // The next position's hash is computed HERE, one position ahead, so the
        // caller's loop never recomputes it — this is what makes
        // HT_REQUIRED_NBYTES 5 rather than 4.
        let hash = *next_hash as usize;
        *next_hash = lz_hash(unsafe { load_u32(base, in_next + 1) }, HT_HASH_ORDER);
        debug_assert!((*next_hash as usize) < HT_TAB_LEN);
        // Prefetch the bucket the NEXT position will touch, matching
        // libdeflate's `prefetchw(&mf->hash_tab[*next_hash])`.
        // SAFETY: `lz_hash(_, HT_HASH_ORDER)` returns < 1 << 15 == HT_TAB_LEN,
        // so this stays inside `hash_tab`; and a prefetch is a pure hint that
        // cannot fault regardless.
        unsafe {
            prefetch_write(self.hash_tab.as_ptr().add(*next_hash as usize) as *const u8);
        }

        // --- entry 0: read, then insert this position ---
        let mut cur_node = self.hash_tab[hash][0] as i32;
        self.hash_tab[hash][0] = cur_pos as i16;
        if cur_node <= cutoff {
            return (0, 0);
        }
        let mut match_pos = in_base_now + cur_node as usize;

        // --- entry 1: shift entry 0 down into it, unconditionally ---
        // libdeflate copies entry 0 into entry 1 even when `nice_len` is reached
        // on the first candidate; keeping that makes the parse decisions match.
        let to_insert = cur_node;
        cur_node = self.hash_tab[hash][1] as i32;
        self.hash_tab[hash][1] = to_insert as i16;

        unsafe {
            if load_u32(base, match_pos) == seq {
                best_len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                best_match_pos = match_pos;
                if cur_node <= cutoff || best_len >= nice_len {
                    return (best_len, (in_next - best_match_pos) as u32);
                }
                match_pos = in_base_now + cur_node as usize;
                // Pre-screen: the same 4-byte head AND the 4 bytes ending at the
                // incumbent's last matched byte. A candidate failing either
                // cannot beat `best_len`, so it is rejected without extending.
                if load_u32(base, match_pos) == seq
                    && load_u32(base, match_pos + best_len as usize - 3)
                        == load_u32(base, in_next + best_len as usize - 3)
                {
                    let len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                    if len > best_len {
                        best_len = len;
                        best_match_pos = match_pos;
                    }
                }
            } else {
                if cur_node <= cutoff {
                    return (0, 0);
                }
                match_pos = in_base_now + cur_node as usize;
                if load_u32(base, match_pos) == seq {
                    best_len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                    best_match_pos = match_pos;
                }
            }
        }

        if best_len == 0 {
            return (0, 0);
        }
        (best_len, (in_next - best_match_pos) as u32)
    }

    /// `ht_matchfinder_skip_bytes`: insert `count` positions without searching.
    ///
    /// Faithful port, including the early return: if fewer than `count +
    /// HT_REQUIRED_NBYTES` bytes remain, libdeflate inserts NOTHING at all
    /// rather than inserting a truncated run. That asymmetry is part of the
    /// algorithm — it is what keeps the `next_hash` pipeline valid — not an
    /// oversight to tidy up.
    // `cur_pos + count - 1 >= WINDOW_SIZE` is libdeflate's literal form. clippy
    // prefers the arithmetically identical `cur_pos + count > WINDOW_SIZE`; the
    // vendor's is kept so this port stays diffable against
    // `ht_matchfinder_skip_bytes` line for line. Faithfulness to the source we are
    // converging on is worth more here than the lint, and the two forms are equal
    // for all i32 values reachable here (`count >= 1`, so no underflow).
    #[allow(clippy::int_plus_one)]
    #[inline(always)]
    pub fn skip_bytes(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        in_end: usize,
        count: u32,
        next_hash: &mut u32,
    ) {
        if count as usize + HT_REQUIRED_NBYTES as usize > in_end - in_next {
            return;
        }
        let mut cur_pos = (in_next - *in_base) as i32;
        if cur_pos + count as i32 - 1 >= MATCHFINDER_WINDOW_SIZE {
            self.slide_window();
            *in_base += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos -= MATCHFINDER_WINDOW_SIZE;
        }

        // SAFETY: the early return above guarantees `count + HT_REQUIRED_NBYTES
        // <= in_end - in_next`, so every `pos` reached below satisfies
        // `pos + 4 <= in_next + count + 5 <= in_end <= buf.len()`.
        let base = buf.as_ptr();
        let mut hash = *next_hash as usize;
        let mut pos = in_next;
        let mut remaining = count;
        loop {
            debug_assert!(hash < HT_TAB_LEN);
            // Shift the bucket down by one and insert at the head.
            let mut i = HT_BUCKET_SIZE - 1;
            while i > 0 {
                self.hash_tab[hash][i] = self.hash_tab[hash][i - 1];
                i -= 1;
            }
            self.hash_tab[hash][0] = cur_pos as i16;

            pos += 1;
            hash = lz_hash(unsafe { load_u32(base, pos) }, HT_HASH_ORDER) as usize;
            cur_pos += 1;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        // SAFETY: as above — `hash < HT_TAB_LEN`, and prefetch cannot fault.
        unsafe {
            prefetch_write(self.hash_tab.as_ptr().add(hash) as *const u8);
        }
        *next_hash = hash as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table must start at the sentinel, not at zero: position 0 is a VALID
    /// position, so a zero-initialised table would report a match against
    /// offset 0 on the very first probe.
    #[test]
    fn new_and_reset_arm_the_sentinel() {
        let mut mf = HtMatchfinder::new();
        assert!(mf
            .hash_tab
            .iter()
            .all(|b| b.iter().all(|&e| e == MATCHFINDER_INITVAL)));
        mf.hash_tab[123][0] = 7;
        mf.hash_tab[123][1] = 9;
        mf.reset();
        assert!(mf
            .hash_tab
            .iter()
            .all(|b| b.iter().all(|&e| e == MATCHFINDER_INITVAL)));
    }

    /// `acquire` must hand back a re-armed table even when it reused a box that
    /// a previous run had dirtied — otherwise T>1 chunk independence breaks.
    #[test]
    fn acquire_rearms_a_recycled_table() {
        {
            let mut mf = HtMatchfinder::acquire();
            mf.hash_tab[42][0] = 1234;
        } // returned to the pool DIRTY
        let mf = HtMatchfinder::acquire();
        assert_eq!(mf.hash_tab[42][0], MATCHFINDER_INITVAL);
    }

    /// A second candidate that the first probe would have missed must be found.
    /// This is the whole point of the 2-way bucket, so it is asserted directly
    /// rather than inferred from a corpus size.
    #[test]
    fn finds_the_better_second_candidate() {
        // Two occurrences of "ABCD" that collide in the same bucket, where the
        // OLDER one extends further. A single-probe finder keeps only the most
        // recent and would return the shorter match.
        let mut data = Vec::new();
        data.extend_from_slice(b"ABCDXXXXXXXXXXXX"); // 0: ABCD + long tail
        data.extend_from_slice(b"ABCDY"); //            16: ABCD + short tail
        data.extend_from_slice(b"ABCDXXXXXXXXXXXX"); // 21: the search position
        data.resize(data.len() + 64, 0); // pad for the word-at-a-time loads

        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        // SAFETY: `data` was padded by 64 bytes above, so a 4-byte load at 0 is
        // trivially in bounds.
        let mut next_hash = lz_hash(unsafe { load_u32(data.as_ptr(), 0) }, HT_HASH_ORDER);

        // Insert positions 0..21 the way the parse loop would.
        for pos in 0..21 {
            let _ = mf.longest_match(&data, &mut in_base, pos, 32, 258, &mut next_hash);
        }
        let (len, off) = mf.longest_match(&data, &mut in_base, 21, 32, 258, &mut next_hash);

        // The best available match at 21 is against position 0 (16 bytes), not
        // the more recent position 16 (4 bytes).
        assert!(
            len >= 8,
            "expected the longer candidate, got len={len} off={off}"
        );
        assert_eq!(off, 21, "expected the offset to name position 0");
    }
}
