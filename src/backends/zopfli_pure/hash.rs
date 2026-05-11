//! Rolling hash + same-byte run + dual hash for LZ77 longest-match.
//! Port of Google Zopfli hash.c

use super::symbols::{ZOPFLI_MIN_MATCH, ZOPFLI_WINDOW_MASK};

pub const HASH_MASK: i32 = 32_767;
pub const HASH_SHIFT: i32 = 5;

pub struct ZopfliHash {
    pub head: Box<[i32]>,    // len 65_536
    pub prev: Box<[u16]>,    // len window_size
    pub hashval: Box<[i32]>, // len window_size
    pub val: i32,

    pub head2: Box<[i32]>,    // len 65_536
    pub prev2: Box<[u16]>,    // len window_size
    pub hashval2: Box<[i32]>, // len window_size
    pub val2: i32,

    pub same: Box<[u16]>, // len window_size
}

impl ZopfliHash {
    pub fn new(window_size: usize) -> Self {
        let mut h = Self {
            head: vec![-1i32; 65_536].into_boxed_slice(),
            prev: vec![0u16; window_size].into_boxed_slice(),
            hashval: vec![-1i32; window_size].into_boxed_slice(),
            val: 0,
            head2: vec![-1i32; 65_536].into_boxed_slice(),
            prev2: vec![0u16; window_size].into_boxed_slice(),
            hashval2: vec![-1i32; window_size].into_boxed_slice(),
            val2: 0,
            same: vec![0u16; window_size].into_boxed_slice(),
        };
        h.reset(window_size);
        h
    }

    pub fn reset(&mut self, window_size: usize) {
        self.val = 0;
        for x in self.head.iter_mut() {
            *x = -1;
        }
        for i in 0..window_size {
            self.prev[i] = i as u16;
            self.hashval[i] = -1;
            self.same[i] = 0;
        }
        self.val2 = 0;
        for x in self.head2.iter_mut() {
            *x = -1;
        }
        for i in 0..window_size {
            self.prev2[i] = i as u16;
            self.hashval2[i] = -1;
        }
    }

    #[inline]
    fn update_hash_value(&mut self, c: u8) {
        self.val = ((self.val << HASH_SHIFT) ^ (c as i32)) & HASH_MASK;
    }

    pub fn warmup(&mut self, array: &[u8], pos: usize, end: usize) {
        self.update_hash_value(array[pos]);
        if pos + 1 < end {
            self.update_hash_value(array[pos + 1]);
        }
    }

    /// Selects either `prev` (primary hash chain) or `prev2` (secondary).
    #[inline]
    pub fn prev_for(&self, use_hash2: bool) -> &[u16] {
        if use_hash2 {
            &self.prev2
        } else {
            &self.prev
        }
    }

    /// Selects either `hashval` (primary hash) or `hashval2` (secondary).
    #[inline]
    pub fn hashval_for(&self, use_hash2: bool) -> &[i32] {
        if use_hash2 {
            &self.hashval2
        } else {
            &self.hashval
        }
    }

    pub fn update(&mut self, array: &[u8], pos: usize, end: usize) {
        let hpos = pos & ZOPFLI_WINDOW_MASK;

        let c = if pos + ZOPFLI_MIN_MATCH <= end {
            array[pos + ZOPFLI_MIN_MATCH - 1]
        } else {
            0
        };
        self.update_hash_value(c);
        self.hashval[hpos] = self.val;

        let head_v = self.head[self.val as usize];
        if head_v != -1 && self.hashval[head_v as usize] == self.val {
            self.prev[hpos] = head_v as u16;
        } else {
            self.prev[hpos] = hpos as u16;
        }
        self.head[self.val as usize] = hpos as i32;

        // Update "same".
        let mut amount: usize = 0;
        let prev_same = self.same[(pos.wrapping_sub(1)) & ZOPFLI_WINDOW_MASK];
        if prev_same > 1 {
            amount = (prev_same - 1) as usize;
        }
        while pos + amount + 1 < end
            && array[pos] == array[pos + amount + 1]
            && amount < u16::MAX as usize
        {
            amount += 1;
        }
        self.same[hpos] = amount as u16;

        // Update second hash.
        self.val2 = ((self.same[hpos] as i32 - ZOPFLI_MIN_MATCH as i32) & 255) ^ self.val;
        self.hashval2[hpos] = self.val2;

        let head2_v = self.head2[self.val2 as usize];
        if head2_v != -1 && self.hashval2[head2_v as usize] == self.val2 {
            self.prev2[hpos] = head2_v as u16;
        } else {
            self.prev2[hpos] = hpos as u16;
        }
        self.head2[self.val2 as usize] = hpos as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::zopfli_pure::symbols::ZOPFLI_WINDOW_SIZE;

    #[test]
    fn hash_initial_state() {
        let h = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);
        assert_eq!(h.val, 0);
        assert_eq!(h.val2, 0);
        assert!(h.head.iter().all(|&x| x == -1));
        assert!(h.hashval.iter().all(|&x| x == -1));
        for (i, &p) in h.prev.iter().enumerate() {
            assert_eq!(p as usize, i);
        }
        assert!(h.same.iter().all(|&x| x == 0));
    }

    #[test]
    fn hash_update_sequence_self_consistent() {
        // Build a deterministic 8 KB input with both random data and a long
        // run so the `same` machinery exercises every branch.
        let mut data = Vec::with_capacity(8192);
        let mut s: u32 = 0xCAFEBABE;
        for _ in 0..4096 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((s >> 16) as u8);
        }
        data.extend(std::iter::repeat_n(b'A', 4096));

        let mut h = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);
        let end = data.len();
        h.warmup(&data, 0, end);
        for pos in 0..end {
            h.update(&data, pos, end);
            let hpos = pos & ZOPFLI_WINDOW_MASK;

            // hashval[hpos] == val
            assert_eq!(h.hashval[hpos], h.val, "pos={}", pos);

            // head[val] points to a slot whose hashval is val
            let head = h.head[h.val as usize];
            assert!(head >= 0);
            assert_eq!(h.hashval[head as usize], h.val);

            // prev[hpos] either equals hpos (uninit) or points to an earlier
            // occurrence with the same hash value
            let prev = h.prev[hpos] as usize;
            assert!(prev == hpos || h.hashval[prev] == h.val);

            // `same[hpos]` matches a brute-force scan of consecutive equal
            // bytes starting at pos.
            let mut count: usize = 0;
            while pos + count + 1 < end
                && data[pos] == data[pos + count + 1]
                && count < u16::MAX as usize
            {
                count += 1;
            }
            assert_eq!(h.same[hpos] as usize, count, "same mismatch at pos={}", pos);
        }
    }
}
