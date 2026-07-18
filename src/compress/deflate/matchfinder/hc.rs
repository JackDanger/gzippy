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

use super::common::{
    lz_extend, lz_hash, matchfinder_init, matchfinder_rebase, MATCHFINDER_INITVAL,
};

pub const HC_HASH3_ORDER: u32 = 15;
pub const HC_HASH4_ORDER: u32 = 16;

const HASH3_SIZE: usize = 1 << HC_HASH3_ORDER;
const HASH4_SIZE: usize = 1 << HC_HASH4_ORDER;

/// 32 KiB DEFLATE window.
pub const WINDOW_SIZE: usize = 1 << 15;
const WINDOW_MASK: u16 = (WINDOW_SIZE - 1) as u16;

/// Unaligned little-endian 4-byte load. Caller guarantees `pos + 4 <= buf.len()`
/// (the parser pads its working buffer so every hot-loop load stays in bounds).
#[inline]
fn load_u32(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

/// The hash-chains matchfinder state.
pub struct HcMatchfinder {
    /// Singleton nodes for length-3 matches (`hash3_tab`).
    hash3_tab: Vec<i16>,
    /// First node of each length-4+ chain (`hash4_tab`).
    hash4_tab: Vec<i16>,
    /// `next_tab[pos]` = the node following `pos` in its chain.
    next_tab: Vec<i16>,
}

impl HcMatchfinder {
    /// `hc_matchfinder_init`: allocate and initialize every table to the sentinel.
    pub fn new() -> Self {
        let mut mf = HcMatchfinder {
            hash3_tab: vec![MATCHFINDER_INITVAL; HASH3_SIZE],
            hash4_tab: vec![MATCHFINDER_INITVAL; HASH4_SIZE],
            next_tab: vec![MATCHFINDER_INITVAL; WINDOW_SIZE],
        };
        mf.reset();
        mf
    }

    /// Re-initialize all tables for a new input buffer.
    pub fn reset(&mut self) {
        matchfinder_init(&mut self.hash3_tab);
        matchfinder_init(&mut self.hash4_tab);
        matchfinder_init(&mut self.next_tab);
    }

    /// `hc_matchfinder_slide_window`: rebase every stored position by one window.
    #[inline]
    fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.hash3_tab);
        matchfinder_rebase(&mut self.hash4_tab);
        matchfinder_rebase(&mut self.next_tab);
    }

    /// Find the longest match longer than `best_len_in` at `in_next`.
    ///
    /// Faithful port of `hc_matchfinder_longest_match`. Returns
    /// `(best_len, offset)`; when no match longer than `best_len_in` is found the
    /// returned length is `best_len_in` and the offset is meaningless (0). The
    /// caller must ensure `buf` is padded so 4-byte loads up to
    /// `in_next + best_len + 1` stay in bounds.
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

        let hash3 = next_hashes[0] as usize;
        let hash4 = next_hashes[1] as usize;

        let cur_node3 = self.hash3_tab[hash3];
        let mut cur_node4 = self.hash4_tab[hash4];

        // Insert the current sequence: replace hash3 singleton, prepend to hash4.
        self.hash3_tab[hash3] = cur_pos as i16;
        self.hash4_tab[hash4] = cur_pos as i16;
        self.next_tab[cur_pos] = cur_node4;

        // Precompute the next position's hashes.
        let next_hashseq = load_u32(buf, in_next + 1);
        next_hashes[0] = lz_hash(next_hashseq & 0xFF_FFFF, HC_HASH3_ORDER);
        next_hashes[1] = lz_hash(next_hashseq, HC_HASH4_ORDER);

        let seq4 = load_u32(buf, in_next);

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
                    if load_u32(buf, mp) & 0xFF_FFFF == seq4 & 0xFF_FFFF {
                        best_len = 3;
                        best_matchptr = mp;
                    }
                }

                // Length-4 match check.
                if (cur_node4 as i32) <= cutoff {
                    break 'search;
                }
                loop {
                    matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                    if load_u32(buf, matchptr) == seq4 {
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

                // Found a length-4 match; extend it fully.
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

            // Length >= 5 loop.
            loop {
                loop {
                    matchptr = (in_base_v as isize + cur_node4 as isize) as usize;
                    // Prefilter: compare the last 4 and the first 4 bytes before
                    // attempting a full extension.
                    let off = best_len as usize - 3;
                    if load_u32(buf, matchptr + off) == load_u32(buf, in_next + off)
                        && load_u32(buf, matchptr) == load_u32(buf, in_next)
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

    /// `hc_matchfinder_skip_bytes`: insert `count` positions without searching.
    ///
    /// Advances the matchfinder over `[in_next, in_next + count)`, updating
    /// `next_hashes` to the hashes for `in_next + count`. No-op if there is not
    /// enough lookahead (`count + 5 > in_end - in_next`), matching the vendor.
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
            let next_hashseq = load_u32(buf, in_next);
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

impl Default for HcMatchfinder {
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
