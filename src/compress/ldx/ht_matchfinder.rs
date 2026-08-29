//! C: `vendor/libdeflate/lib/ht_matchfinder.h` — a Hash Table (ht) matchfinder.
//!
//! This is a variant of the Hash Chains (hc) matchfinder that is optimized for very
//! fast compression. The ht_matchfinder stores the hash chains inline in the hash
//! table, whereas the hc_matchfinder stores them in a separate array. Storing the hash
//! chains inline is the faster method when `max_search_depth` (the maximum chain
//! length) is very small. It is not appropriate when `max_search_depth` is larger, as
//! then it uses too much memory.
//!
//! **Due to its focus on speed, the ht_matchfinder doesn't support length 3 matches.**
//! It also doesn't allow `max_search_depth` to vary at runtime; it is fixed at build
//! time as `HT_MATCHFINDER_BUCKET_SIZE`.
//!
//! # This is NOT the same thing as our shipping `deflate/matchfinder/ht.rs`
//!
//! That file adds a length-3 table which the C explicitly refuses (see the paragraph
//! above, `ht_matchfinder.h:38-40`) and imports `HT_MAX_LEN3_OFFSET = 4096` from a
//! DIFFERENT C function. It is a derivative, not a port, and it must not be diffed
//! against this file as though it were one.
//!
//! The binding FALSIFY record at `src/compress/deflate/parse/mod.rs:540` covers both
//! prior attempts to route L1 through a ht-style finder: attempt 1 DIED ON SIZE
//! (clause 3, 7 pass->fail flips) because `fast`'s `head3` length-3 table wins on
//! BINARIES and `ht_matchfinder` has no length-3 support; attempt 2 (2-way buckets AND
//! a length-3 table) passed size and died on the T1 WALL at 1.2662x. **Read it before
//! proposing to route anything from here.** This module is the FAITHFUL C, built so
//! that a third attempt can at least be measured against the real thing rather than
//! against a derivative.

use super::matchfinder_common::{
    lz_extend, lz_hash, matchfinder_init, matchfinder_rebase, prefetchw, MfPos,
    MATCHFINDER_WINDOW_SIZE,
};

/// C: `#define HT_MATCHFINDER_HASH_ORDER 15` (:49)
pub(crate) const HT_MATCHFINDER_HASH_ORDER: u32 = 15;
/// C: `#define HT_MATCHFINDER_BUCKET_SIZE 2` (:50)
pub(crate) const HT_MATCHFINDER_BUCKET_SIZE: usize = 2;
/// C: `#define HT_MATCHFINDER_MIN_MATCH_LEN 4` (:52)
///
/// **4, not DEFLATE's 3.** See the module docs — this is the design constraint that
/// killed the first attempt to route L1 here.
pub(crate) const HT_MATCHFINDER_MIN_MATCH_LEN: u32 = 4;
/// C: `#define HT_MATCHFINDER_REQUIRED_NBYTES 5` (:54)
///
/// Minimum value of `max_len` for `ht_matchfinder_longest_match`: it hashes
/// `in_next + 1` as a 4-byte sequence, so it reads 5 bytes from `in_next`.
pub(crate) const HT_MATCHFINDER_REQUIRED_NBYTES: u32 = 5;

const HASH_TAB_ENTRIES: usize = (1usize << HT_MATCHFINDER_HASH_ORDER) * HT_MATCHFINDER_BUCKET_SIZE;

/// C: `struct ht_matchfinder` (:57)
///
/// `hash_tab[1 << HASH_ORDER][BUCKET_SIZE]` flattened, because `matchfinder_init` and
/// `matchfinder_rebase` both take it as one flat `mf_pos_t *` — the C's own casts say
/// the flat view is the primary one.
pub(crate) struct HtMatchfinder {
    pub(crate) hash_tab: Box<[MfPos; HASH_TAB_ENTRIES]>,
}

impl HtMatchfinder {
    /// C: `ht_matchfinder_init(struct ht_matchfinder *mf)` (:65)
    pub(crate) fn new() -> Self {
        let mut mf = Self {
            hash_tab: vec![0 as MfPos; HASH_TAB_ENTRIES]
                .into_boxed_slice()
                .try_into()
                .expect("exact size"),
        };
        mf.init();
        mf
    }

    /// C: `ht_matchfinder_init` (:65)
    pub(crate) fn init(&mut self) {
        matchfinder_init(&mut self.hash_tab[..]);
    }

    /// C: `ht_matchfinder_slide_window(struct ht_matchfinder *mf)` (:73)
    #[inline(always)]
    pub(crate) fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.hash_tab[..]);
    }

    #[inline(always)]
    fn slot(&self, hash: u32, i: usize) -> usize {
        (hash as usize) * HT_MATCHFINDER_BUCKET_SIZE + i
    }
}

/// C: `static forceinline` (`hc_matchfinder.h` / `ht_matchfinder.h`). Ours carried
/// NO inline attribute, so this was a real ABI call — and it takes 10 arguments,
/// past AArch64's 8 argument registers, so every call spilled to the stack.
/// Measured: the deficit vs the C is call-shape-dependent — at L9 (depth 600,
/// few long calls) we BEAT it 0.88x, at L2 (depth 6, many short calls) we lose
/// 1.34x. Matching the vendor's `forceinline`.
#[inline(always)]
/// C: `ht_matchfinder_longest_match(...)` (:80)
///
/// Returns the best match length (0 if none) and writes the offset to `offset_ret`.
/// `max_len` must be >= `HT_MATCHFINDER_REQUIRED_NBYTES`.
///
/// # The bucket-2 path, and its one deliberate asymmetry
///
/// The C's comment on this branch: *"Hand-unrolled version for BUCKET_SIZE == 2. The
/// logic here also differs slightly in that it copies the first entry to the second
/// even if nice_len is reached on the first, as this can be slightly faster."*
///
/// So the insert into slot 1 happens BEFORE the `nice_len` early-out, unconditionally.
/// Moving it after — which reads as the obvious cleanup, since the value is unused on
/// that path — changes what the table holds at every position where a nice-length
/// match was found, and therefore changes every subsequent match. It is a
/// table-state divergence, not a dead store.
///
/// # The second-candidate guard
///
/// Before extending the second candidate the C checks BOTH a 4-byte sequence equality
/// AND `load_u32(matchptr + best_len - 3) == load_u32(in_next + best_len - 3)` — a
/// cheap test that the candidate can beat the incumbent at its far end before paying
/// for `lz_extend`. Dropping it finds the same matches more slowly; keeping it but
/// getting the `- 3` wrong finds DIFFERENT matches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ht_matchfinder_longest_match(
    mf: &mut HtMatchfinder,
    buf: &[u8],
    in_base: &mut usize,
    in_next: usize,
    max_len: u32,
    nice_len: u32,
    next_hash: &mut u32,
    offset_ret: &mut u32,
) -> u32 {
    let mut best_len: u32 = 0;
    let mut best_matchptr: usize = in_next;
    let mut in_base_local = *in_base;
    let in_base_ptr = in_base as *mut usize;
    let mut cur_pos = (in_next - in_base_local) as i32;

    // This is assumed throughout this function.
    const _: () = assert!(HT_MATCHFINDER_MIN_MATCH_LEN == 4);

    if cur_pos == MATCHFINDER_WINDOW_SIZE {
        mf.slide_window();
        in_base_local += MATCHFINDER_WINDOW_SIZE as usize;
        cur_pos = 0;
    }
    // Raw pointer for the hot loop (helps register allocation vs usize on stack).
    let in_next_ptr = unsafe { buf.as_ptr().add(in_next) };
    let cutoff: MfPos = (cur_pos - MATCHFINDER_WINDOW_SIZE) as MfPos;

    let hash = *next_hash;
    const _: () = assert!(HT_MATCHFINDER_REQUIRED_NBYTES == 5);
    *next_hash = lz_hash(load_u32(buf, in_next + 1), HT_MATCHFINDER_HASH_ORDER);
    let seq = load_u32(buf, in_next);
    prefetchw(unsafe {
        mf.hash_tab
            .as_ptr()
            .add(*next_hash as usize * HT_MATCHFINDER_BUCKET_SIZE)
    });

    // --- C: the BUCKET_SIZE == 2 hand-unrolled version ---
    let s0 = mf.slot(hash, 0);
    let s1 = mf.slot(hash, 1);

    let mut cur_node = mf.hash_tab[s0];
    mf.hash_tab[s0] = cur_pos as MfPos;
    if cur_node <= cutoff {
        *offset_ret = (in_next - best_matchptr) as u32;
        *in_base = in_base_local;
        return best_len;
    }
    let mut matchptr = unsafe { in_next_ptr.sub((cur_pos - cur_node as i32) as usize) };

    // C: `to_insert = cur_node; cur_node = mf->hash_tab[hash][1];
    //     mf->hash_tab[hash][1] = to_insert;`
    //
    // Clippy sees a manual swap and suggests `core::mem::swap`. It is one, but the C
    // spells it as three statements around a named `to_insert`, and the copy to slot 1
    // happening HERE — before the `nice_len` early-out below — is the branch's one
    // documented asymmetry. Keeping the three statements keeps that visible.
    #[allow(clippy::manual_swap, reason = "C: three statements around `to_insert`")]
    let to_insert = cur_node;
    #[allow(clippy::manual_swap, reason = "C: three statements around `to_insert`")]
    {
        cur_node = mf.hash_tab[s1];
        mf.hash_tab[s1] = to_insert;
    }

    if unsafe { load_u32_ptr(matchptr) } == seq {
        best_len = unsafe {
            lz_extend(
                buf,
                in_next,
                matchptr as usize - buf.as_ptr() as usize,
                4,
                max_len,
            )
        };
        best_matchptr = matchptr as usize - buf.as_ptr() as usize;
        if cur_node <= cutoff || best_len >= nice_len {
            *offset_ret = (in_next - best_matchptr) as u32;
            *in_base = in_base_local;
            return best_len;
        }
        matchptr = unsafe { in_next_ptr.sub((cur_pos - cur_node as i32) as usize) };
        if unsafe { load_u32_ptr(matchptr) } == seq
            && unsafe { load_u32_ptr(matchptr.add(best_len as usize - 3)) }
                == load_u32(buf, in_next + best_len as usize - 3)
        {
            let len = unsafe {
                lz_extend(
                    buf,
                    in_next,
                    matchptr as usize - buf.as_ptr() as usize,
                    4,
                    max_len,
                )
            };
            if len > best_len {
                best_len = len;
                best_matchptr = matchptr as usize - buf.as_ptr() as usize;
            }
        }
    } else {
        if cur_node <= cutoff {
            *offset_ret = (in_next - best_matchptr) as u32;
            *in_base = in_base_local;
            return best_len;
        }
        matchptr = unsafe { in_next_ptr.sub((cur_pos - cur_node as i32) as usize) };
        if unsafe { load_u32_ptr(matchptr) } == seq {
            best_len = unsafe {
                lz_extend(
                    buf,
                    in_next,
                    matchptr as usize - buf.as_ptr() as usize,
                    4,
                    max_len,
                )
            };
            best_matchptr = matchptr as usize - buf.as_ptr() as usize;
        }
    }

    *offset_ret = (in_next - best_matchptr) as u32;
    unsafe { *in_base_ptr = in_base_local };
    best_len
}

/// C: `static forceinline` (`hc_matchfinder.h` / `ht_matchfinder.h`). Ours carried
/// NO inline attribute, so this was a real ABI call — and it takes 10 arguments,
/// past AArch64's 8 argument registers, so every call spilled to the stack.
/// Measured: the deficit vs the C is call-shape-dependent — at L9 (depth 600,
/// few long calls) we BEAT it 0.88x, at L2 (depth 6, many short calls) we lose
/// 1.34x. Matching the vendor's `forceinline`.
#[inline(always)]
/// C: `ht_matchfinder_skip_bytes(...)` (:196)
///
/// Insert `count` consecutive positions into the table without searching. Used after a
/// match is taken, so the interior of the match is still indexed.
///
/// # The early return is not an optimisation
///
/// `if (count + HT_MATCHFINDER_REQUIRED_NBYTES > in_end - in_next) return;` — with
/// fewer than 5 bytes left, hashing would read past the input. Skipping the whole
/// insert leaves those tail positions unindexed, which is why the last few bytes of a
/// stream are always literals.
pub(crate) fn ht_matchfinder_skip_bytes(
    mf: &mut HtMatchfinder,
    buf: &[u8],
    in_base: &mut usize,
    in_next: usize,
    in_end: usize,
    count: u32,
    next_hash: &mut u32,
) {
    let mut cur_pos = (in_next - *in_base) as i32;
    let mut remaining = count;

    if count as usize + HT_MATCHFINDER_REQUIRED_NBYTES as usize > in_end - in_next {
        return;
    }
    if remaining == 0 {
        // The C's `do { ... } while (--remaining)` runs at least once and would wrap
        // on count == 0. Its callers never pass 0 (a match is at least 4 long, so
        // `length - 1` is at least 3), so this is a guard, not a behaviour change.
        return;
    }

    // C: `if (cur_pos + count - 1 >= MATCHFINDER_WINDOW_SIZE)`. Clippy would have this
    // as `> MATCHFINDER_WINDOW_SIZE - 1`; same predicate, different source line.
    #[allow(
        clippy::int_plus_one,
        reason = "C: the `- 1 >=` form is the source line"
    )]
    if cur_pos + count as i32 - 1 >= MATCHFINDER_WINDOW_SIZE {
        mf.slide_window();
        *in_base += MATCHFINDER_WINDOW_SIZE as usize;
        cur_pos -= MATCHFINDER_WINDOW_SIZE;
    }

    let mut hash = *next_hash;
    let mut p = in_next;
    loop {
        let base = (hash as usize) * HT_MATCHFINDER_BUCKET_SIZE;
        for i in (1..HT_MATCHFINDER_BUCKET_SIZE).rev() {
            mf.hash_tab[base + i] = mf.hash_tab[base + i - 1];
        }
        mf.hash_tab[base] = cur_pos as MfPos;

        p += 1;
        hash = lz_hash(load_u32(buf, p), HT_MATCHFINDER_HASH_ORDER);
        cur_pos += 1;

        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    prefetchw(unsafe {
        mf.hash_tab
            .as_ptr()
            .add(hash as usize * HT_MATCHFINDER_BUCKET_SIZE)
    });
    *next_hash = hash;
}

/// C: `matchptr = &in_base[cur_node];`
///
/// **`cur_node` is SIGNED and is routinely negative, so this is signed pointer
/// arithmetic and not an unsigned add.** After a window slide, `in_base` has advanced
/// by `MATCHFINDER_WINDOW_SIZE` and every surviving table entry has had the same
/// amount subtracted, so entries in the older half of the window are negative — and
/// `cur_node > cutoff` accepts them, because `cutoff` is `cur_pos - WINDOW_SIZE` and
/// is itself negative.
///
/// Writing this as `in_base + cur_node as usize` compiles, works for the whole first
/// 32 KiB, and then panics (debug) or reads gigabytes out of bounds (release) at the
/// first slide. It is exactly the bug `matches_stay_valid_across_a_window_slide`
/// exists to catch, and it did.
#[inline(always)]
fn node_ptr(in_base: usize, cur_node: MfPos) -> usize {
    (in_base as isize + cur_node as isize) as usize
}

/// C: `load_u32_unaligned` / `get_unaligned_le32`.
///
/// The C reads 4 bytes with no bounds check because its callers guarantee the room
/// (`HT_MATCHFINDER_REQUIRED_NBYTES`, and the `BUF_PAD` the compressor allocates). We
/// read from a slice, so out-of-range would panic; the callers here maintain the same
/// guarantee, and the tail-shortfall paths return before reaching this.
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
/// Load a 32-bit LE value from a raw pointer.
/// SAFETY: the caller guarantees 4 readable bytes at `ptr`.
#[inline(always)]
unsafe fn load_u32_ptr(ptr: *const u8) -> u32 {
    u32::from_le(unsafe { (ptr as *const u32).read_unaligned() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the matchfinder over a buffer with a known repeat and check it finds it.
    /// The reference is not another implementation — it is the definition: the longest
    /// prefix match at any earlier position within the window.
    fn brute_force_longest(buf: &[u8], pos: usize, max_len: u32, window: usize) -> (u32, u32) {
        let lo = pos.saturating_sub(window);
        let mut best = (0u32, 0u32);
        for cand in lo..pos {
            let mut l = 0u32;
            while l < max_len
                && (pos + l as usize) < buf.len()
                && buf[cand + l as usize] == buf[pos + l as usize]
            {
                l += 1;
            }
            if l >= HT_MATCHFINDER_MIN_MATCH_LEN && l > best.0 {
                best = (l, (pos - cand) as u32);
            }
        }
        best
    }

    /// Every match the finder returns must be a REAL match: the bytes at the reported
    /// offset must actually equal the bytes at the current position for the reported
    /// length, and the offset must be inside the window. A finder that returns a
    /// phantom match produces a corrupt stream, which is the only failure mode that
    /// matters.
    #[test]
    fn every_reported_match_is_real_and_in_window() {
        let mut state: u32 = 0xC0FF_EE00;
        // Compressible-ish data with plenty of repeats.
        let mut buf: Vec<u8> = Vec::with_capacity(200_000);
        while buf.len() < 200_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            if r < 100 && buf.len() > 1000 {
                let off = 1 + (r as usize * 7) % 900;
                let start = buf.len() - off;
                let n = 4 + (r as usize % 40);
                for k in 0..n {
                    let b = buf[start + k];
                    buf.push(b);
                }
            } else {
                buf.push((r % 64) as u8);
            }
        }
        // Room for the 5-byte lookahead at the very end.
        buf.extend_from_slice(&[0u8; 16]);

        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hash = 0u32;
        let in_end = buf.len() - 16;

        let mut pos = 0usize;
        let mut matches = 0usize;
        while pos + 16 < in_end {
            let remaining = (in_end - pos) as u32;
            if remaining < HT_MATCHFINDER_REQUIRED_NBYTES {
                break;
            }
            let max_len = core::cmp::min(258, remaining);
            let mut offset = 0u32;
            let len = ht_matchfinder_longest_match(
                &mut mf,
                &buf,
                &mut in_base,
                pos,
                max_len,
                core::cmp::min(24, max_len),
                &mut next_hash,
                &mut offset,
            );

            if len != 0 {
                matches += 1;
                assert!(
                    len >= HT_MATCHFINDER_MIN_MATCH_LEN,
                    "pos={pos}: length {len} is below the finder's minimum"
                );
                assert!(offset >= 1, "pos={pos}: offset 0 is not a match");
                assert!(
                    (offset as usize) <= pos,
                    "pos={pos}: offset {offset} points before the buffer"
                );
                assert!(
                    (offset as i32) <= MATCHFINDER_WINDOW_SIZE,
                    "pos={pos}: offset {offset} exceeds the 32768 window"
                );
                let src = pos - offset as usize;
                for k in 0..len as usize {
                    assert_eq!(
                        buf[src + k],
                        buf[pos + k],
                        "pos={pos} offset={offset} len={len}: byte {k} differs"
                    );
                }
                ht_matchfinder_skip_bytes(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    pos + 1,
                    in_end,
                    len - 1,
                    &mut next_hash,
                );
                pos += len as usize;
            } else {
                pos += 1;
            }
        }

        assert!(
            matches > 500,
            "only {matches} matches found on deliberately repetitive data — \
             the finder is not working, so the validity checks proved nothing"
        );
    }

    /// The window slide must fire and must keep producing valid matches across it.
    /// This is where a wrong `matchfinder_rebase` shows up as a phantom match.
    #[test]
    fn matches_stay_valid_across_a_window_slide() {
        // Well over 32768 bytes, so at least one slide happens.
        let unit = b"the quick brown fox jumps over the lazy dog 0123456789 ";
        let mut buf = Vec::new();
        while buf.len() < 120_000 {
            buf.extend_from_slice(unit);
        }
        buf.extend_from_slice(&[0u8; 16]);

        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hash = 0u32;
        let in_end = buf.len() - 16;

        let mut pos = 0usize;
        let mut slides_observed = 0;
        let mut last_base = 0usize;
        while pos + 16 < in_end {
            let remaining = (in_end - pos) as u32;
            if remaining < HT_MATCHFINDER_REQUIRED_NBYTES {
                break;
            }
            let max_len = core::cmp::min(258, remaining);
            let mut offset = 0u32;
            let len = ht_matchfinder_longest_match(
                &mut mf,
                &buf,
                &mut in_base,
                pos,
                max_len,
                core::cmp::min(32, max_len),
                &mut next_hash,
                &mut offset,
            );
            if in_base != last_base {
                slides_observed += 1;
                last_base = in_base;
            }
            if len != 0 {
                let src = pos - offset as usize;
                assert_eq!(
                    &buf[src..src + len as usize],
                    &buf[pos..pos + len as usize],
                    "pos={pos} offset={offset}: phantom match after {slides_observed} slides"
                );
                ht_matchfinder_skip_bytes(
                    &mut mf,
                    &buf,
                    &mut in_base,
                    pos + 1,
                    in_end,
                    len - 1,
                    &mut next_hash,
                );
                pos += len as usize;
            } else {
                pos += 1;
            }
        }
        assert!(
            slides_observed >= 2,
            "expected at least two window slides, saw {slides_observed}"
        );
    }

    /// A freshly initialised table must report no matches: every slot holds
    /// `MATCHFINDER_INITVAL`, which is <= every possible cutoff.
    #[test]
    fn a_fresh_table_finds_nothing() {
        let buf = b"abcdefghijklmnopqrstuvwxyz0123456789".to_vec();
        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hash = 0u32;
        let mut offset = 0u32;
        let len = ht_matchfinder_longest_match(
            &mut mf,
            &buf,
            &mut in_base,
            0,
            20,
            20,
            &mut next_hash,
            &mut offset,
        );
        assert_eq!(len, 0);
    }

    /// `skip_bytes` must decline when fewer than `count + 5` bytes remain, leaving the
    /// hash unchanged — otherwise it would hash past the end of the input.
    #[test]
    fn skip_bytes_declines_near_the_end_of_input() {
        let buf = vec![7u8; 64];
        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        let mut next_hash = 0xABCD_1234u32;
        let before = mf.hash_tab[0];

        // 10 bytes left, asking to skip 8: 8 + 5 > 10, so it must decline.
        ht_matchfinder_skip_bytes(&mut mf, &buf, &mut in_base, 54, 64, 8, &mut next_hash);
        assert_eq!(next_hash, 0xABCD_1234, "the hash must not advance");
        assert_eq!(mf.hash_tab[0], before, "the table must not be touched");
    }
}
