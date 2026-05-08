//! LZ77 store + block state + longest-match finder + greedy parse.
//! Port of vendor/zopfli/src/zopfli/lz77.c
//!
//! Built up across plan Steps 6 (LZ77Store + histogram), 7 (BlockState +
//! find_longest_match), and 8 (lz77_greedy + end-to-end oracle).

#![allow(dead_code)]

use super::symbols::{dist_symbol, length_symbol, ZOPFLI_NUM_D, ZOPFLI_NUM_LL};

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
