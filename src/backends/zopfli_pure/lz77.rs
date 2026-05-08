//! LZ77 store + block state + longest-match finder + greedy parse.
//! Port of vendor/zopfli/src/zopfli/lz77.c
//!
//! Built up across plan Steps 6 (LZ77Store + histogram), 7 (BlockState +
//! find_longest_match), and 8 (lz77_greedy + end-to-end oracle).

#![allow(dead_code)]

use super::cache::LongestMatchCache;
use super::hash::ZopfliHash;
use super::symbols::{
    dist_symbol, length_symbol, ZOPFLI_MAX_CHAIN_HITS, ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH,
    ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_MASK, ZOPFLI_WINDOW_SIZE,
};
use super::ZopfliOptions;

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Stores the LZ77 lit/len + dist stream produced by greedy or squeeze.
pub struct LZ77Store<'a> {
    pub litlens: Vec<u16>,
    pub dists: Vec<u16>,
    pub data: &'a [u8],
    pub pos: Vec<usize>,
    pub ll_symbol: Vec<u16>,
    pub d_symbol: Vec<u16>,
    /// Cumulative running histogram, one entry per LZ77 symbol; wraps every
    /// `ZOPFLI_NUM_LL` entries.
    pub ll_counts: Vec<usize>,
    /// Cumulative running histogram, one entry per LZ77 symbol; wraps every
    /// `ZOPFLI_NUM_D` entries.
    pub d_counts: Vec<usize>,
}

impl<'a> LZ77Store<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            litlens: Vec::new(),
            dists: Vec::new(),
            data,
            pos: Vec::new(),
            ll_symbol: Vec::new(),
            d_symbol: Vec::new(),
            ll_counts: Vec::new(),
            d_counts: Vec::new(),
        }
    }

    /// Number of LZ77 symbols stored.
    #[inline]
    pub fn size(&self) -> usize {
        self.litlens.len()
    }

    /// Reset to an empty store, retaining capacity.
    pub fn reset(&mut self) {
        self.litlens.clear();
        self.dists.clear();
        self.pos.clear();
        self.ll_symbol.clear();
        self.d_symbol.clear();
        self.ll_counts.clear();
        self.d_counts.clear();
    }

    /// Appends a literal (`dist == 0`) or length-distance pair.
    pub fn store_lit_len_dist(&mut self, length: u16, dist: u16, pos: usize) {
        debug_assert!(length < 259);

        let origsize = self.size();
        let llstart = ZOPFLI_NUM_LL * (origsize / ZOPFLI_NUM_LL);
        let dstart = ZOPFLI_NUM_D * (origsize / ZOPFLI_NUM_D);

        // Each time the index wraps, append a fresh chunk seeded with the
        // previous chunk's tail (or zeros for the first chunk).
        if origsize.is_multiple_of(ZOPFLI_NUM_LL) {
            for i in 0..ZOPFLI_NUM_LL {
                let v = if origsize == 0 {
                    0
                } else {
                    self.ll_counts[origsize - ZOPFLI_NUM_LL + i]
                };
                self.ll_counts.push(v);
            }
        }
        if origsize.is_multiple_of(ZOPFLI_NUM_D) {
            for i in 0..ZOPFLI_NUM_D {
                let v = if origsize == 0 {
                    0
                } else {
                    self.d_counts[origsize - ZOPFLI_NUM_D + i]
                };
                self.d_counts.push(v);
            }
        }

        self.litlens.push(length);
        self.dists.push(dist);
        self.pos.push(pos);

        if dist == 0 {
            self.ll_symbol.push(length);
            self.d_symbol.push(0);
            self.ll_counts[llstart + length as usize] += 1;
        } else {
            let ls = length_symbol(length as i32) as u16;
            let ds = dist_symbol(dist as i32) as u16;
            self.ll_symbol.push(ls);
            self.d_symbol.push(ds);
            self.ll_counts[llstart + ls as usize] += 1;
            self.d_counts[dstart + ds as usize] += 1;
        }
    }

    /// Appends every entry of `src` (replays `store_lit_len_dist` so the
    /// running histograms stay coherent).
    pub fn append_from(&mut self, src: &Self) {
        for i in 0..src.size() {
            self.store_lit_len_dist(src.litlens[i], src.dists[i], src.pos[i]);
        }
    }

    /// Bytes of original input covered by the LZ77 symbols in `[lstart, lend)`.
    pub fn byte_range(&self, lstart: usize, lend: usize) -> usize {
        if lstart == lend {
            return 0;
        }
        let l = lend - 1;
        let span = if self.dists[l] == 0 {
            1
        } else {
            self.litlens[l] as usize
        };
        self.pos[l] + span - self.pos[lstart]
    }

    fn get_histogram_at(
        &self,
        lpos: usize,
        ll_counts: &mut [usize; ZOPFLI_NUM_LL],
        d_counts: &mut [usize; ZOPFLI_NUM_D],
    ) {
        let llpos = ZOPFLI_NUM_LL * (lpos / ZOPFLI_NUM_LL);
        let dpos = ZOPFLI_NUM_D * (lpos / ZOPFLI_NUM_D);
        ll_counts.copy_from_slice(&self.ll_counts[llpos..llpos + ZOPFLI_NUM_LL]);
        let mut i = lpos + 1;
        while i < llpos + ZOPFLI_NUM_LL && i < self.size() {
            ll_counts[self.ll_symbol[i] as usize] -= 1;
            i += 1;
        }
        d_counts.copy_from_slice(&self.d_counts[dpos..dpos + ZOPFLI_NUM_D]);
        let mut i = lpos + 1;
        while i < dpos + ZOPFLI_NUM_D && i < self.size() {
            if self.dists[i] != 0 {
                d_counts[self.d_symbol[i] as usize] -= 1;
            }
            i += 1;
        }
    }

    /// Histogram of lit/len + dist symbols in `[lstart, lend)`. Excludes the
    /// implicit end-of-block symbol 256.
    pub fn get_histogram(
        &self,
        lstart: usize,
        lend: usize,
        ll_counts: &mut [usize; ZOPFLI_NUM_LL],
        d_counts: &mut [usize; ZOPFLI_NUM_D],
    ) {
        if lstart + ZOPFLI_NUM_LL * 3 > lend {
            for c in ll_counts.iter_mut() {
                *c = 0;
            }
            for c in d_counts.iter_mut() {
                *c = 0;
            }
            for i in lstart..lend {
                ll_counts[self.ll_symbol[i] as usize] += 1;
                if self.dists[i] != 0 {
                    d_counts[self.d_symbol[i] as usize] += 1;
                }
            }
        } else {
            self.get_histogram_at(lend - 1, ll_counts, d_counts);
            if lstart > 0 {
                let mut ll2 = [0usize; ZOPFLI_NUM_LL];
                let mut d2 = [0usize; ZOPFLI_NUM_D];
                self.get_histogram_at(lstart - 1, &mut ll2, &mut d2);
                for i in 0..ZOPFLI_NUM_LL {
                    ll_counts[i] -= ll2[i];
                }
                for i in 0..ZOPFLI_NUM_D {
                    d_counts[i] -= d2[i];
                }
            }
        }
    }
}

// ── Step 7: BlockState + find_longest_match ──────────────────────────────────

/// State for compressing a single block (currently just owns the LMC).
pub struct BlockState<'opt> {
    pub options: &'opt ZopfliOptions,
    pub lmc: Option<LongestMatchCache>,
    pub blockstart: usize,
    pub blockend: usize,
}

impl<'opt> BlockState<'opt> {
    pub fn new(options: &'opt ZopfliOptions, start: usize, end: usize, add_lmc: bool) -> Self {
        let lmc = if add_lmc {
            Some(LongestMatchCache::new(end - start))
        } else {
            None
        };
        Self {
            options,
            lmc,
            blockstart: start,
            blockend: end,
        }
    }
}

/// Debug-asserts that a length/distance pair really matches the data at `pos`.
pub fn verify_len_dist(data: &[u8], pos: usize, dist: u16, length: u16) {
    debug_assert!(pos + length as usize <= data.len());
    for i in 0..length as usize {
        debug_assert_eq!(
            data[pos - dist as usize + i],
            data[pos + i],
            "len/dist mismatch at pos={} dist={} length={} i={}",
            pos,
            dist,
            length,
            i
        );
    }
}

/// Equivalent of `GetMatch`: walks `scan` and `match_` while equal, up to
/// `end`. `safe_end` is `end - 8` so we can compare 8 bytes at once.
#[inline]
fn get_match(array: &[u8], scan_start: usize, match_start: usize, end: usize) -> usize {
    let mut scan = scan_start;
    let mut m = match_start;
    let safe_end = end.saturating_sub(8);
    while scan < safe_end {
        let s = u64::from_le_bytes(array[scan..scan + 8].try_into().unwrap());
        let mb = u64::from_le_bytes(array[m..m + 8].try_into().unwrap());
        if s != mb {
            break;
        }
        scan += 8;
        m += 8;
    }
    while scan < end && array[scan] == array[m] {
        scan += 1;
        m += 1;
    }
    scan
}

/// Mirrors `TryGetFromLongestMatchCache`. Returns `Some((length, distance))`
/// if the cache had a hit, possibly populating `sublen`. Updates `*limit` if
/// only partial cache info is usable.
fn try_get_from_longest_match_cache(
    s: &BlockState<'_>,
    pos: usize,
    limit: &mut usize,
    sublen: Option<&mut [u16; 259]>,
) -> Option<(u16, u16)> {
    let lmc = s.lmc.as_ref()?;
    let lmcpos = pos - s.blockstart;

    let cache_available = lmc.length[lmcpos] == 0 || lmc.dist[lmcpos] != 0;
    if !cache_available {
        return None;
    }

    let limit_ok_for_cache = *limit == ZOPFLI_MAX_MATCH
        || (lmc.length[lmcpos] as usize) <= *limit
        || (sublen.is_some()
            && lmc.max_cached_sublen(lmcpos, lmc.length[lmcpos] as u32) as usize >= *limit);
    if !limit_ok_for_cache {
        return None;
    }

    let want_sublen = sublen.is_some();
    let length_full = lmc.length[lmcpos];
    let max_cached = lmc.max_cached_sublen(lmcpos, length_full as u32);
    if !want_sublen || (length_full as u32) <= max_cached {
        let mut length = length_full;
        if (length as usize) > *limit {
            length = *limit as u16;
        }
        let distance = if let Some(sublen) = sublen {
            lmc.cache_to_sublen(lmcpos, length as u32, sublen);
            let d = sublen[length as usize];
            if *limit == ZOPFLI_MAX_MATCH && length >= ZOPFLI_MIN_MATCH as u16 {
                debug_assert_eq!(sublen[length as usize], lmc.dist[lmcpos]);
            }
            d
        } else {
            lmc.dist[lmcpos]
        };
        return Some((length, distance));
    }

    // Partial cache hit: only the chain length limit is useful.
    *limit = length_full as usize;
    None
}

/// Mirrors `StoreInLongestMatchCache`. Only stores when `limit == MAX_MATCH`
/// and a sublen array was provided and the slot is currently empty.
fn store_in_longest_match_cache(
    s: &mut BlockState<'_>,
    pos: usize,
    limit: usize,
    sublen: Option<&[u16; 259]>,
    distance: u16,
    length: u16,
) {
    let Some(lmc) = s.lmc.as_mut() else {
        return;
    };
    let lmcpos = pos - s.blockstart;
    let cache_available = lmc.length[lmcpos] == 0 || lmc.dist[lmcpos] != 0;
    if limit != ZOPFLI_MAX_MATCH || sublen.is_none() || cache_available {
        return;
    }
    debug_assert!(lmc.length[lmcpos] == 1 && lmc.dist[lmcpos] == 0);
    let (cache_dist, cache_length) = if (length as usize) < ZOPFLI_MIN_MATCH {
        (0u16, 0u16)
    } else {
        (distance, length)
    };
    lmc.dist[lmcpos] = cache_dist;
    lmc.length[lmcpos] = cache_length;
    debug_assert!(!(lmc.length[lmcpos] == 1 && lmc.dist[lmcpos] == 0));
    if let Some(sublen) = sublen {
        lmc.sublen_to_cache(sublen, lmcpos, length as u32);
    }
}

/// Port of `ZopfliFindLongestMatch`. Writes the best `(distance, length)`
/// for `pos` and, if `sublen` is provided, the closest distance for every
/// length up to `bestlength`.
#[allow(clippy::too_many_arguments)]
pub fn find_longest_match(
    s: &mut BlockState<'_>,
    h: &ZopfliHash,
    array: &[u8],
    pos: usize,
    size: usize,
    mut limit: usize,
    mut sublen: Option<&mut [u16; 259]>,
    distance: &mut u16,
    length: &mut u16,
) {
    if let Some(hit) = try_get_from_longest_match_cache(s, pos, &mut limit, sublen.as_deref_mut()) {
        debug_assert!(pos + hit.0 as usize <= size);
        *length = hit.0;
        *distance = hit.1;
        return;
    }

    debug_assert!(limit <= ZOPFLI_MAX_MATCH);
    debug_assert!(limit >= ZOPFLI_MIN_MATCH);
    debug_assert!(pos < size);

    if size - pos < ZOPFLI_MIN_MATCH {
        *length = 0;
        *distance = 0;
        return;
    }

    if pos + limit > size {
        limit = size - pos;
    }

    let arrayend = pos + limit;

    let hpos = (pos & ZOPFLI_WINDOW_MASK) as u16;
    let mut bestdist: u16 = 0;
    let mut bestlength: u16 = 1;

    // Hash chain we're currently walking — start with the primary hash, swap
    // to the secondary one once it's strictly more efficient (see C lines
    // 509-518).
    let mut use_hash2 = false;
    let mut hval = h.val;

    let pp = h.head[hval as usize] as u16;
    debug_assert!(pp == hpos);
    let mut p = h.prev[pp as usize];
    let mut pp_local = pp;

    let mut dist: u32 = if p < pp_local {
        (pp_local - p) as u32
    } else {
        (ZOPFLI_WINDOW_SIZE as u32) - p as u32 + pp_local as u32
    };

    let mut chain_counter: i32 = ZOPFLI_MAX_CHAIN_HITS;

    while dist < ZOPFLI_WINDOW_SIZE as u32 {
        let mut currentlength: u16 = 0;

        debug_assert!((p as usize) < ZOPFLI_WINDOW_SIZE);
        debug_assert_eq!(p, h.prev_for(use_hash2)[pp_local as usize]);
        debug_assert_eq!(h.hashval_for(use_hash2)[p as usize], hval);

        if dist > 0 {
            debug_assert!(pos < size);
            debug_assert!((dist as usize) <= pos);
            let mut scan = pos;
            let mut match_ = pos - dist as usize;

            // Test the byte at position bestlength first; this is a small
            // speedup on chains with many short matches.
            if pos + bestlength as usize >= size
                || array[scan + bestlength as usize] == array[match_ + bestlength as usize]
            {
                // Skip ahead by `same` if we're inside a long run.
                let same0 = h.same[pos & ZOPFLI_WINDOW_MASK];
                if same0 > 2 && array[scan] == array[match_] {
                    let same1 = h.same[(pos - dist as usize) & ZOPFLI_WINDOW_MASK];
                    let mut same = same0.min(same1) as usize;
                    if same > limit {
                        same = limit;
                    }
                    scan += same;
                    match_ += same;
                }
                let scan_end = get_match(array, scan, match_, arrayend);
                currentlength = (scan_end - pos) as u16;
            }

            if currentlength > bestlength {
                if let Some(ref mut sl) = sublen {
                    for j in (bestlength as usize + 1)..=currentlength as usize {
                        sl[j] = dist as u16;
                    }
                }
                bestdist = dist as u16;
                bestlength = currentlength;
                if currentlength as usize >= limit {
                    break;
                }
            }
        }

        // Switch to the secondary hash once it's at least as efficient.
        if !use_hash2 && bestlength >= h.same[hpos as usize] && h.val2 == h.hashval2[p as usize] {
            use_hash2 = true;
            hval = h.val2;
        }

        pp_local = p;
        p = h.prev_for(use_hash2)[p as usize];
        if p == pp_local {
            break;
        }

        dist += if p < pp_local {
            (pp_local - p) as u32
        } else {
            (ZOPFLI_WINDOW_SIZE as u32) - p as u32 + pp_local as u32
        };

        chain_counter -= 1;
        if chain_counter <= 0 {
            break;
        }
    }

    store_in_longest_match_cache(s, pos, limit, sublen.as_deref(), bestdist, bestlength);

    debug_assert!(bestlength as usize <= limit);

    *distance = bestdist;
    *length = bestlength;
    debug_assert!(pos + *length as usize <= size);
}

// ── Step 8: greedy LZ77 + first end-to-end oracle ────────────────────────────

#[inline]
fn get_length_score(length: i32, distance: i32) -> i32 {
    // Long distances cost a lot of extra bits; nudge the score down for them.
    if distance > 1024 {
        length - 1
    } else {
        length
    }
}

/// Greedy LZ77 with lazy matching. Mirrors `ZopfliLZ77Greedy`.
pub fn lz77_greedy(
    s: &mut BlockState<'_>,
    in_: &[u8],
    instart: usize,
    inend: usize,
    store: &mut LZ77Store<'_>,
    h: &mut ZopfliHash,
) {
    if instart == inend {
        return;
    }

    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    h.reset(ZOPFLI_WINDOW_SIZE);
    h.warmup(in_, windowstart, inend);
    for i in windowstart..instart {
        h.update(in_, i, inend);
    }

    let mut prev_length: u32 = 0;
    let mut prev_match: u32 = 0;
    let mut match_available = false;

    let mut dummysublen = [0u16; 259];

    let mut i = instart;
    while i < inend {
        h.update(in_, i, inend);

        let mut leng: u16 = 0;
        let mut dist: u16 = 0;
        find_longest_match(
            s,
            h,
            in_,
            i,
            inend,
            ZOPFLI_MAX_MATCH,
            Some(&mut dummysublen),
            &mut dist,
            &mut leng,
        );
        let lengthscore = get_length_score(leng as i32, dist as i32);

        let prevlengthscore = get_length_score(prev_length as i32, prev_match as i32);
        if match_available {
            match_available = false;
            if lengthscore > prevlengthscore + 1 {
                store.store_lit_len_dist(in_[i - 1] as u16, 0, i - 1);
                if lengthscore >= ZOPFLI_MIN_MATCH as i32 && (leng as usize) < ZOPFLI_MAX_MATCH {
                    match_available = true;
                    prev_length = leng as u32;
                    prev_match = dist as u32;
                    i += 1;
                    continue;
                }
            } else {
                let leng = prev_length as u16;
                let dist = prev_match as u16;
                verify_len_dist(in_, i - 1, dist, leng);
                store.store_lit_len_dist(leng, dist, i - 1);
                for _ in 2..leng as usize {
                    debug_assert!(i < inend);
                    i += 1;
                    h.update(in_, i, inend);
                }
                i += 1;
                continue;
            }
        } else if lengthscore >= ZOPFLI_MIN_MATCH as i32 && (leng as usize) < ZOPFLI_MAX_MATCH {
            match_available = true;
            prev_length = leng as u32;
            prev_match = dist as u32;
            i += 1;
            continue;
        }

        // Add to output (no lazy hold).
        let stored_leng = if lengthscore >= ZOPFLI_MIN_MATCH as i32 {
            verify_len_dist(in_, i, dist, leng);
            store.store_lit_len_dist(leng, dist, i);
            leng
        } else {
            store.store_lit_len_dist(in_[i] as u16, 0, i);
            1
        };
        for _ in 1..stored_leng as usize {
            debug_assert!(i < inend);
            i += 1;
            h.update(in_, i, inend);
        }
        i += 1;
    }
}

impl<'a> Clone for LZ77Store<'a> {
    fn clone(&self) -> Self {
        // Mirror C ZopfliCopyLZ77Store: histograms are sized to a multiple
        // of NUM_LL / NUM_D (one chunk per chunkful of stored symbols).
        let llsize = ZOPFLI_NUM_LL * ceil_div(self.size(), ZOPFLI_NUM_LL);
        let dsize = ZOPFLI_NUM_D * ceil_div(self.size(), ZOPFLI_NUM_D);
        Self {
            litlens: self.litlens.clone(),
            dists: self.dists.clone(),
            data: self.data,
            pos: self.pos.clone(),
            ll_symbol: self.ll_symbol.clone(),
            d_symbol: self.d_symbol.clone(),
            ll_counts: self.ll_counts[..llsize].to_vec(),
            d_counts: self.d_counts[..dsize].to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_histogram(
        store: &LZ77Store<'_>,
        lstart: usize,
        lend: usize,
    ) -> ([usize; ZOPFLI_NUM_LL], [usize; ZOPFLI_NUM_D]) {
        let mut ll = [0usize; ZOPFLI_NUM_LL];
        let mut d = [0usize; ZOPFLI_NUM_D];
        for i in lstart..lend {
            ll[store.ll_symbol[i] as usize] += 1;
            if store.dists[i] != 0 {
                d[store.d_symbol[i] as usize] += 1;
            }
        }
        (ll, d)
    }

    #[test]
    fn store_basic() {
        let data = b"hello world";
        let mut s = LZ77Store::new(data);
        s.store_lit_len_dist(b'h' as u16, 0, 0);
        s.store_lit_len_dist(b'i' as u16, 0, 1);
        assert_eq!(s.size(), 2);
        assert_eq!(s.dists, vec![0, 0]);
        assert_eq!(s.litlens, vec![b'h' as u16, b'i' as u16]);
        assert_eq!(s.byte_range(0, 2), 2);
    }

    #[test]
    fn append_from_replays_histogram() {
        let data = b"abcdef";
        let mut a = LZ77Store::new(data);
        let mut b = LZ77Store::new(data);
        for (i, &c) in data.iter().enumerate() {
            a.store_lit_len_dist(c as u16, 0, i);
            b.store_lit_len_dist(c as u16, 0, i);
        }
        let mut a2 = LZ77Store::new(data);
        a2.append_from(&a);
        assert_eq!(a.litlens, a2.litlens);
        assert_eq!(a.dists, a2.dists);
        assert_eq!(a.ll_counts, a2.ll_counts);
    }

    fn brute_longest_match(data: &[u8], pos: usize, limit: usize) -> (u16, u16) {
        // Mirrors find_longest_match's selection rule: walk distances from
        // closest (1) outward; the largest length wins; on ties keep the
        // earliest (smallest) distance.
        let max_len = limit.min(data.len() - pos);
        if max_len < ZOPFLI_MIN_MATCH {
            return (0, 0);
        }
        let mut best_len: u16 = 1;
        let mut best_dist: u16 = 0;
        let max_dist = ZOPFLI_WINDOW_SIZE.min(pos);
        for dist in 1..=max_dist {
            let mut len = 0;
            while len < max_len && data[pos + len] == data[pos - dist + len] {
                len += 1;
            }
            if len as u16 > best_len {
                best_len = len as u16;
                best_dist = dist as u16;
                if best_len as usize >= max_len {
                    break;
                }
            }
        }
        if (best_len as usize) < ZOPFLI_MIN_MATCH {
            (0, 0)
        } else {
            (best_len, best_dist)
        }
    }

    #[test]
    fn find_longest_match_brute_force_small() {
        // Small inputs that don't have huge same-byte runs (so the chain-skip
        // and MAX_CHAIN_HITS cap stay inactive) and that fit O(n^2 * MAX_MATCH)
        // brute force comfortably.
        let inputs: Vec<Vec<u8>> = vec![
            b"hello world hello world hello world hello world".to_vec(),
            b"abcabcabcabcabcabcabcabcabcabcabcabc".to_vec(),
            b"aaaabbbbccccaaaabbbbccccaaaabbbbcccc".to_vec(),
            (0..256u32).map(|i| (i * 17 % 251) as u8).collect(),
        ];
        let opts = ZopfliOptions::default();
        for data in &inputs {
            let mut s = BlockState::new(&opts, 0, data.len(), false);
            let mut h = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);
            h.warmup(data, 0, data.len());
            for pos in 0..data.len() {
                h.update(data, pos, data.len());
                let mut sublen = [0u16; 259];
                let mut dist = 0u16;
                let mut length = 0u16;
                find_longest_match(
                    &mut s,
                    &h,
                    data,
                    pos,
                    data.len(),
                    ZOPFLI_MAX_MATCH,
                    Some(&mut sublen),
                    &mut dist,
                    &mut length,
                );
                let (b_len, b_dist) = brute_longest_match(data, pos, ZOPFLI_MAX_MATCH);

                if b_len < ZOPFLI_MIN_MATCH as u16 {
                    // No match reachable; zopfli may still leave bestlength=1
                    // with bestdist=0 (no match found in the chain).
                    assert!(length <= 1, "pos={} length={}", pos, length);
                } else {
                    assert_eq!(
                        length, b_len,
                        "length mismatch at pos={} got={} brute={}",
                        pos, length, b_len
                    );
                    assert_eq!(
                        dist, b_dist,
                        "dist mismatch at pos={} got=({},{}) brute=({},{})",
                        pos, length, dist, b_len, b_dist
                    );
                }
            }
        }
    }

    #[test]
    fn get_histogram_matches_brute() {
        let data = b"ignored placeholder";
        let mut s = LZ77Store::new(data);

        // Replay ~10k stores: mix of literals and matches with lengths in the
        // full 3..=258 range and varied distances. This forces wrap-arounds in
        // both ll_counts (every 288) and d_counts (every 32).
        let mut rng: u32 = 0xDEADBEEF;
        let n = 10_000usize;
        for i in 0..n {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let r = rng >> 16;
            if r.is_multiple_of(4) {
                let len = 3 + (r as u16 % 256);
                let dist = 1 + (r as u16 % 32_000);
                s.store_lit_len_dist(len, dist, i);
            } else {
                s.store_lit_len_dist((r & 0xff) as u16, 0, i);
            }
        }

        // Compare 50 random ranges, including ranges that hit the small-block
        // fast path (lstart + 3*NUM_LL > lend) and the cumulative path.
        let mut rng2: u32 = 0x12345;
        for _ in 0..50 {
            rng2 = rng2.wrapping_mul(1103515245).wrapping_add(12345);
            let a = (rng2 as usize) % n;
            rng2 = rng2.wrapping_mul(1103515245).wrapping_add(12345);
            let b = (rng2 as usize) % n;
            let (lstart, lend) = if a < b { (a, b + 1) } else { (b, a + 1) };

            let mut ll = [0usize; ZOPFLI_NUM_LL];
            let mut d = [0usize; ZOPFLI_NUM_D];
            s.get_histogram(lstart, lend, &mut ll, &mut d);
            let (ll_b, d_b) = brute_histogram(&s, lstart, lend);
            assert_eq!(ll, ll_b, "ll mismatch lstart={} lend={}", lstart, lend);
            assert_eq!(d, d_b, "d mismatch lstart={} lend={}", lstart, lend);
        }
    }
}
