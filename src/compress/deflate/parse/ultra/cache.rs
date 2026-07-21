//! Longest-match cache used by `find_longest_match`.
//! Port of Google Zopfli cache.c

use super::symbols::ZOPFLI_CACHE_LENGTH;

pub struct LongestMatchCache {
    pub length: Box<[u16]>, // len blocksize, init 1
    pub dist: Box<[u16]>,   // len blocksize, init 0
    pub sublen: Box<[u8]>,  // len ZOPFLI_CACHE_LENGTH * 3 * blocksize, init 0
}

impl LongestMatchCache {
    pub fn new(blocksize: usize) -> Self {
        Self {
            length: vec![1u16; blocksize].into_boxed_slice(),
            dist: vec![0u16; blocksize].into_boxed_slice(),
            sublen: vec![0u8; ZOPFLI_CACHE_LENGTH * 3 * blocksize].into_boxed_slice(),
        }
    }

    /// Stores the relevant length-distance pairs from `sublen[0..=length]`
    /// into the cache slot for `pos`.
    pub fn sublen_to_cache(&mut self, sublen: &[u16; 259], pos: usize, length: u32) {
        if length < 3 {
            return;
        }
        let base = ZOPFLI_CACHE_LENGTH * pos * 3;
        let cache = &mut self.sublen[base..base + ZOPFLI_CACHE_LENGTH * 3];
        let mut j: usize = 0;
        let mut bestlength: u32 = 0;
        for i in 3..=length as usize {
            if i == length as usize || sublen[i] != sublen[i + 1] {
                cache[j * 3] = (i - 3) as u8;
                cache[j * 3 + 1] = (sublen[i] & 0xff) as u8;
                cache[j * 3 + 2] = ((sublen[i] >> 8) & 0xff) as u8;
                bestlength = i as u32;
                j += 1;
                if j >= ZOPFLI_CACHE_LENGTH {
                    break;
                }
            }
        }
        if j < ZOPFLI_CACHE_LENGTH {
            debug_assert_eq!(bestlength, length);
            cache[(ZOPFLI_CACHE_LENGTH - 1) * 3] = (bestlength - 3) as u8;
        } else {
            debug_assert!(bestlength <= length);
        }
        debug_assert_eq!(bestlength, self.max_cached_sublen(pos, length));
    }

    /// Reconstructs `sublen[prev_length..=max_cached_sublen]` from the cache.
    pub fn cache_to_sublen(&self, pos: usize, length: u32, sublen: &mut [u16; 259]) {
        if length < 3 {
            return;
        }
        let maxlength = self.max_cached_sublen(pos, length);
        let mut prevlength: u32 = 0;
        let base = ZOPFLI_CACHE_LENGTH * pos * 3;
        let cache = &self.sublen[base..base + ZOPFLI_CACHE_LENGTH * 3];
        for j in 0..ZOPFLI_CACHE_LENGTH {
            let entry_len = cache[j * 3] as u32 + 3;
            let dist = cache[j * 3 + 1] as u16 | ((cache[j * 3 + 2] as u16) << 8);
            for i in prevlength..=entry_len {
                sublen[i as usize] = dist;
            }
            if entry_len == maxlength {
                break;
            }
            prevlength = entry_len + 1;
        }
    }

    /// Returns the maximum length cached for `pos`, or 0 if nothing is cached.
    pub fn max_cached_sublen(&self, pos: usize, _length: u32) -> u32 {
        let base = ZOPFLI_CACHE_LENGTH * pos * 3;
        let cache = &self.sublen[base..base + ZOPFLI_CACHE_LENGTH * 3];
        if cache[1] == 0 && cache[2] == 0 {
            return 0;
        }
        cache[(ZOPFLI_CACHE_LENGTH - 1) * 3] as u32 + 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_sublen(length: u32) -> [u16; 259] {
        // Construct a plausible monotone sublen: at each length the closest
        // matching distance only increases as the length grows.
        let mut s = [0u16; 259];
        let mut dist: u16 = 0;
        for (i, slot) in s.iter_mut().enumerate().take(length as usize + 1).skip(3) {
            if i % 4 == 3 {
                dist = dist.saturating_add(7);
            }
            *slot = dist.max(1);
        }
        s
    }

    #[test]
    fn roundtrip_small() {
        let mut lmc = LongestMatchCache::new(4);
        let length: u32 = 12;
        let sublen = build_sublen(length);
        lmc.sublen_to_cache(&sublen, 1, length);

        let max = lmc.max_cached_sublen(1, length);
        assert_eq!(max, length);

        let mut got = [0u16; 259];
        lmc.cache_to_sublen(1, length, &mut got);
        for i in 3..=max as usize {
            assert_eq!(got[i], sublen[i], "mismatch at i={}", i);
        }
    }

    #[test]
    fn roundtrip_truncated_when_many_distinct_distances() {
        let mut lmc = LongestMatchCache::new(2);
        let mut sublen = [0u16; 259];
        // 9 distinct distances → cache holds only 8.
        for (i, slot) in sublen.iter_mut().enumerate().take(21).skip(3) {
            *slot = (i as u16) * 11;
        }
        lmc.sublen_to_cache(&sublen, 0, 20);
        let max = lmc.max_cached_sublen(0, 20);
        assert!(max <= 20);
        assert!(max >= 3);
        let mut got = [0u16; 259];
        lmc.cache_to_sublen(0, 20, &mut got);
        for i in 3..=max as usize {
            assert_eq!(got[i], sublen[i], "mismatch at i={}", i);
        }
    }

    #[test]
    fn empty_cache_reports_zero() {
        let lmc = LongestMatchCache::new(2);
        assert_eq!(lmc.max_cached_sublen(0, 100), 0);
        assert_eq!(lmc.max_cached_sublen(1, 100), 0);
    }
}
