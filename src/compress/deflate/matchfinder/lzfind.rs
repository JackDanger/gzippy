//! Binary-tree matchfinder — faithful Rust port of ECT's LzFind `Bt3Zip`
//! (7-zip `LzFind.c` as trimmed by Felix Hanau in Efficient Compression Tool).
//!
//! This is the matchfinder that drives ECT's optimal-parse DP and is the lever
//! behind ECT's best-in-class deflate ratio. It maintains a per-hash-bucket
//! binary search tree over the sliding window and, for each position, emits the
//! Pareto frontier of `(length, distance)` matches: for every achievable match
//! length `>= 3`, the nearest distance that achieves it, with lengths strictly
//! increasing.
//!
//! Port notes / deliberate deviations from ECT (all preserve byte-valid output
//! and the emitted match frontier; see the property test):
//!
//!  * **Safe Rust.** All buffer accesses are bounds-checked `Vec` indices, not
//!    raw pointers. The matchfinder works on a private padded copy of the block
//!    (`in_[windowstart..inend]` plus `PAD` trailing bytes) so the 8-byte
//!    `get_match` over-reads and the `cur[len_limit]` boundary reads that ECT
//!    satisfies with caller-allocated padding are always in bounds here.
//!  * **Portable hash.** We always use the non-SSE xor/crc hash. Because the
//!    hash is a pure function of the 3 bytes at a position, any two positions
//!    with an equal 3-byte prefix land in the same bucket regardless of the
//!    hash function, so the *emitted match set is identical* to the SSE-CRC
//!    build — only the collision rate (a speed detail) differs.
//!  * **No `mfinexport` cross-block carry.** ECT can hand a warmed matchfinder
//!    to the adjacent block via thread-local state; we always create fresh and
//!    warm the pre-window with `skip`, matching ECT's common `else` branch.

use crate::compress::deflate::parse::ultra::symbols::{
    ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH, ZOPFLI_WINDOW_SIZE,
};

const WINDOW: usize = ZOPFLI_WINDOW_SIZE; // 32768
const WINDOW_MASK: u32 = (ZOPFLI_WINDOW_SIZE as u32) - 1; // 32767
const HASH_LOG: u32 = 15;
const HASH_SIZE: usize = 1 << HASH_LOG; // 32768
const HASH_MASK: u32 = (HASH_SIZE as u32) - 1;

/// Trailing bytes appended to the private buffer so the 8-byte word compares in
/// [`get_match`] and the `cur[len_limit]` boundary reads never run off the end.
const PAD: usize = 16;

/// Maximum number of `u16` values one [`MatchFinder::get_matches`] can emit:
/// one `(len, dist)` pair per length in `3..=258` → `256 * 2 = 512`.
pub const MAX_MATCH_U16: usize = (ZOPFLI_MAX_MATCH - ZOPFLI_MIN_MATCH + 1) * 2;

/// zlib CRC32 constants, exactly as in ECT `LzFind.h` (used by the portable
/// hash of the middle byte).
#[rustfmt::skip]
static CRC: [u32; 256] = [
    0, 1996959894, 3993919788, 2567524794, 124634137, 1886057615, 3915621685, 2657392035,
    249268274, 2044508324, 3772115230, 2547177864, 162941995, 2125561021, 3887607047, 2428444049,
    498536548, 1789927666, 4089016648, 2227061214, 450548861, 1843258603, 4107580753, 2211677639,
    325883990, 1684777152, 4251122042, 2321926636, 335633487, 1661365465, 4195302755, 2366115317,
    997073096, 1281953886, 3579855332, 2724688242, 1006888145, 1258607687, 3524101629, 2768942443,
    901097722, 1119000684, 3686517206, 2898065728, 853044451, 1172266101, 3705015759, 2882616665,
    651767980, 1373503546, 3369554304, 3218104598, 565507253, 1454621731, 3485111705, 3099436303,
    671266974, 1594198024, 3322730930, 2970347812, 795835527, 1483230225, 3244367275, 3060149565,
    1994146192, 31158534, 2563907772, 4023717930, 1907459465, 112637215, 2680153253, 3904427059,
    2013776290, 251722036, 2517215374, 3775830040, 2137656763, 141376813, 2439277719, 3865271297,
    1802195444, 476864866, 2238001368, 4066508878, 1812370925, 453092731, 2181625025, 4111451223,
    1706088902, 314042704, 2344532202, 4240017532, 1658658271, 366619977, 2362670323, 4224994405,
    1303535960, 984961486, 2747007092, 3569037538, 1256170817, 1037604311, 2765210733, 3554079995,
    1131014506, 879679996, 2909243462, 3663771856, 1141124467, 855842277, 2852801631, 3708648649,
    1342533948, 654459306, 3188396048, 3373015174, 1466479909, 544179635, 3110523913, 3462522015,
    1591671054, 702138776, 2966460450, 3352799412, 1504918807, 783551873, 3082640443, 3233442989,
    3988292384, 2596254646, 62317068, 1957810842, 3939845945, 2647816111, 81470997, 1943803523,
    3814918930, 2489596804, 225274430, 2053790376, 3826175755, 2466906013, 167816743, 2097651377,
    4027552580, 2265490386, 503444072, 1762050814, 4150417245, 2154129355, 426522225, 1852507879,
    4275313526, 2312317920, 282753626, 1742555852, 4189708143, 2394877945, 397917763, 1622183637,
    3604390888, 2714866558, 953729732, 1340076626, 3518719985, 2797360999, 1068828381, 1219638859,
    3624741850, 2936675148, 906185462, 1090812512, 3747672003, 2825379669, 829329135, 1181335161,
    3412177804, 3160834842, 628085408, 1382605366, 3423369109, 3138078467, 570562233, 1426400815,
    3317316542, 2998733608, 733239954, 1555261956, 3268935591, 3050360625, 752459403, 1541320221,
    2607071920, 3965973030, 1969922972, 40735498, 2617837225, 3943577151, 1913087877, 83908371,
    2512341634, 3803740692, 2075208622, 213261112, 2463272603, 3855990285, 2094854071, 198958881,
    2262029012, 4057260610, 1759359992, 534414190, 2176718541, 4139329115, 1873836001, 414664567,
    2282248934, 4279200368, 1711684554, 285281116, 2405801727, 4167216745, 1634467795, 376229701,
    2685067896, 3608007406, 1308918612, 956543938, 2808555105, 3495958263, 1231636301, 1047427035,
    2932959818, 3654703836, 1088359270, 936918000, 2847714899, 3736837829, 1202900863, 817233897,
    3183342108, 3401237130, 1404277552, 615818150, 3134207493, 3453421203, 1423857449, 601450431,
    3009837614, 3294710456, 1567103746, 711928724, 3020668471, 3272380065, 1510334235, 755167117,
];

/// Binary-tree matchfinder over one block plus its 32 KiB pre-window.
pub struct MatchFinder {
    /// Private padded copy: `in_[windowstart..inend]` followed by `PAD` bytes.
    buf: Vec<u8>,
    /// Logical end of real data within `buf` (= `inend - windowstart`). Matches
    /// never extend past here.
    len_data: usize,
    /// 7-zip logical position; starts at `WINDOW` so that `curMatch == 0` (an
    /// empty hash slot) always yields `delta = pos - 0 >= WINDOW` and is
    /// rejected as out-of-window.
    pos: u32,
    /// Index of the current position within `buf`.
    cur: usize,
    /// Cyclic buffer position in `[0, WINDOW)`; indexes tree nodes in `son`.
    cyclic_pos: u32,
    /// Hash bucket → most recent position (`pos`-valued; `0` = empty).
    hash: Vec<u32>,
    /// Binary-tree links: two children per cyclic position.
    son: Vec<u32>,
}

impl MatchFinder {
    /// Creates a matchfinder for block `[instart, inend)` with pre-window
    /// `[windowstart, instart)` already warmed via `skip`. `windowstart` must be
    /// `<= instart <= inend <= in_.len()`.
    pub fn new(in_: &[u8], windowstart: usize, instart: usize, inend: usize) -> Self {
        debug_assert!(windowstart <= instart && instart <= inend && inend <= in_.len());
        let len_data = inend - windowstart;
        // Copy a few real bytes past `inend` when the input has them so the
        // boundary comparisons match ECT for non-final blocks; otherwise the
        // tail stays zero-padded (deterministic).
        let real_end = (inend + PAD).min(in_.len());
        let copy_len = real_end - windowstart;
        let mut buf = vec![0u8; len_data + PAD];
        buf[..copy_len].copy_from_slice(&in_[windowstart..real_end]);

        let mut mf = Self {
            buf,
            len_data,
            pos: WINDOW as u32,
            cur: 0,
            cyclic_pos: 0,
            hash: vec![0u32; HASH_SIZE],
            son: vec![0u32; 2 * WINDOW],
        };
        mf.skip(instart - windowstart);
        mf
    }

    /// Portable 3-byte hash: `(cur[2] | (cur[0] << 8)) ^ crc[cur[1]]`, masked.
    #[inline]
    fn hash_at(&self, idx: usize) -> u32 {
        let c0 = self.buf[idx] as u32;
        let c1 = self.buf[idx + 1] as usize;
        let c2 = self.buf[idx + 2] as u32;
        ((c2 | (c0 << 8)) ^ CRC[c1]) & HASH_MASK
    }

    #[inline]
    fn move_pos(&mut self) {
        self.cyclic_pos = (self.cyclic_pos + 1) & WINDOW_MASK;
        self.cur += 1;
        self.pos += 1;
    }

    /// Number of real bytes remaining from the current position.
    #[inline]
    fn remaining(&self) -> usize {
        self.len_data - self.cur
    }

    /// Common-prefix length of `buf[a..]` and `buf[b..]`, capped so `a` never
    /// passes `end`. Mirrors ECT `GetMatch` (cloudflare 8-byte compare).
    #[inline]
    fn get_match(buf: &[u8], mut a: usize, mut b: usize, end: usize) -> usize {
        let safe_end = end.saturating_sub(8);
        while a < safe_end {
            let sv = u64::from_le_bytes(buf[a..a + 8].try_into().unwrap());
            let mv = u64::from_le_bytes(buf[b..b + 8].try_into().unwrap());
            let x = sv ^ mv;
            if x != 0 {
                return a + (x.trailing_zeros() >> 3) as usize;
            }
            a += 8;
            b += 8;
        }
        while a < end && buf[a] == buf[b] {
            a += 1;
            b += 1;
        }
        a
    }

    /// Emits the match frontier for the current position into `distances`
    /// (interleaved `len, dist, len, dist, …`, strictly increasing `len`) and
    /// advances one position. Returns the number of `u16` values written
    /// (`2 * numPairs`). `distances` must have length `>= MAX_MATCH_U16`.
    /// Port of `Bt3Zip_MatchFinder_GetMatches`.
    pub fn get_matches(&mut self, distances: &mut [u16]) -> usize {
        let lenl = self.remaining();
        if lenl < ZOPFLI_MIN_MATCH {
            self.move_pos();
            return 0;
        }
        let len_limit = lenl.min(ZOPFLI_MAX_MATCH);
        let h = self.hash_at(self.cur) as usize;
        let cur_match = self.hash[h];
        self.hash[h] = self.pos;
        let n = self.bt_get_matches(len_limit, cur_match, distances);
        self.move_pos();
        n
    }

    /// Tree descent for [`get_matches`]. Port of static `GetMatches`.
    fn bt_get_matches(
        &mut self,
        len_limit: usize,
        mut cur_match: u32,
        distances: &mut [u16],
    ) -> usize {
        let cyclic = self.cyclic_pos;
        let cur = self.cur;
        let pos = self.pos;
        let mut ptr0 = ((cyclic << 1) + 1) as usize;
        let mut ptr1 = (cyclic << 1) as usize;
        let mut len0: usize = 0;
        let mut len1: usize = 0;
        let mut max_len: usize = 2;
        let mut count: usize = 0;

        loop {
            let delta = pos - cur_match;
            if (delta as usize) >= WINDOW {
                self.son[ptr0] = 0;
                self.son[ptr1] = 0;
                return count;
            }
            let pair = (((cyclic + WINDOW as u32 - delta) & WINDOW_MASK) << 1) as usize;
            let pb = cur - delta as usize;
            let mut len = len0.min(len1);
            if self.buf[pb + len] == self.buf[cur + len] {
                len += 1;
                if self.buf[pb + len] == self.buf[cur + len] {
                    len = Self::get_match(&self.buf, cur + len, pb + len, cur + len_limit) - cur;
                }
                if max_len < len {
                    distances[count] = len as u16;
                    distances[count + 1] = delta as u16;
                    count += 2;
                    max_len = len;
                    if len == len_limit {
                        self.son[ptr1] = self.son[pair];
                        self.son[ptr0] = self.son[pair + 1];
                        return count;
                    }
                }
            }
            if self.buf[pb + len] < self.buf[cur + len] {
                self.son[ptr1] = cur_match;
                ptr1 = pair + 1;
                cur_match = self.son[ptr1];
                len1 = len;
            } else {
                self.son[ptr0] = cur_match;
                ptr0 = pair;
                cur_match = self.son[ptr0];
                len0 = len;
            }
        }
    }

    /// Advances `num` positions, inserting each into the tree without emitting
    /// matches. Port of `Bt3Zip_MatchFinder_Skip`.
    pub fn skip(&mut self, num: usize) {
        for _ in 0..num {
            let lenl = self.remaining();
            let len_limit = lenl.min(ZOPFLI_MAX_MATCH);
            let h = self.hash_at(self.cur) as usize;
            let cur_match = self.hash[h];
            self.hash[h] = self.pos;
            self.skip_matches(len_limit, cur_match);
            self.move_pos();
        }
    }

    /// Tree descent for [`skip`]. Port of static `SkipMatches`.
    fn skip_matches(&mut self, len_limit: usize, mut cur_match: u32) {
        let cyclic = self.cyclic_pos;
        let cur = self.cur;
        let pos = self.pos;
        let mut ptr0 = ((cyclic << 1) + 1) as usize;
        let mut ptr1 = (cyclic << 1) as usize;
        let mut len0: usize = 0;
        let mut len1: usize = 0;

        loop {
            let delta = pos - cur_match;
            if (delta as usize) >= WINDOW {
                self.son[ptr0] = 0;
                self.son[ptr1] = 0;
                return;
            }
            let pair = (((cyclic + WINDOW as u32 - delta) & WINDOW_MASK) << 1) as usize;
            let pb = cur - delta as usize;
            let mut len = len0.min(len1);
            if self.buf[pb + len] == self.buf[cur + len] {
                len = Self::get_match(&self.buf, cur + len, pb + len, cur + len_limit) - cur;
                if len == len_limit {
                    self.son[ptr1] = self.son[pair];
                    self.son[ptr0] = self.son[pair + 1];
                    return;
                }
            }
            if self.buf[pb + len] < self.buf[cur + len] {
                self.son[ptr1] = cur_match;
                ptr1 = pair + 1;
                cur_match = self.son[ptr1];
                len1 = len;
            } else {
                self.son[ptr0] = cur_match;
                ptr0 = pair;
                cur_match = self.son[ptr0];
                len0 = len;
            }
        }
    }

    /// Fast `O(1)`-per-position advance for a run of identical bytes at
    /// distance 1 (all positions share one hash). Port of
    /// `Bt3Zip_MatchFinder_Skip2` + `SkipMatches2`. Only valid inside such a
    /// run; used by the squeeze `ML_RLE` fast path. Leaves an approximate
    /// (but always match-verified) tree, exactly as ECT does.
    pub fn skip2(&mut self, num: usize) {
        if num == 0 {
            return;
        }
        let h = self.hash_at(self.cur) as usize;
        for _ in 0..num {
            self.hash[h] = self.pos;
            let cyclic = self.cyclic_pos;
            let ptr1 = (cyclic << 1) as usize;
            let ptr0 = ((cyclic << 1) + 1) as usize;
            let pair = (((cyclic + WINDOW as u32 - 1) & WINDOW_MASK) << 1) as usize;
            let p0 = self.son[pair];
            let p1 = self.son[pair + 1];
            self.son[ptr1] = p0;
            self.son[ptr0] = p1;
            self.move_pos();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force Pareto frontier at `pos`: scan distances nearest-first,
    /// emit `(len, dist)` whenever `len` strictly beats the running best and is
    /// a valid `>= 3` match, capped at `ZOPFLI_MAX_MATCH`. This is exactly the
    /// set the BT matchfinder is specified to return.
    fn brute_pairs(data: &[u8], pos: usize) -> Vec<(u16, u16)> {
        let mut out = Vec::new();
        let remaining = data.len() - pos;
        let cap = remaining.min(ZOPFLI_MAX_MATCH);
        if cap < ZOPFLI_MIN_MATCH {
            return out;
        }
        let max_d = pos.min(WINDOW - 1);
        let mut best = 2usize;
        for d in 1..=max_d {
            let mut l = 0usize;
            while l < cap && data[pos + l] == data[pos - d + l] {
                l += 1;
            }
            if l > best && l >= ZOPFLI_MIN_MATCH {
                out.push((l as u16, d as u16));
                best = l;
                if l == cap {
                    break;
                }
            }
        }
        out
    }

    fn mf_pairs_all(data: &[u8]) -> Vec<Vec<(u16, u16)>> {
        let mut mf = MatchFinder::new(data, 0, 0, data.len());
        let mut scratch = vec![0u16; MAX_MATCH_U16];
        let mut all = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            let n = mf.get_matches(&mut scratch);
            let mut pairs = Vec::with_capacity(n / 2);
            let mut k = 0;
            while k < n {
                pairs.push((scratch[k], scratch[k + 1]));
                k += 2;
            }
            all.push(pairs);
        }
        all
    }

    fn assert_frontier(data: &[u8], label: &str) {
        let got = mf_pairs_all(data);
        for pos in 0..data.len() {
            let expected = brute_pairs(data, pos);
            assert_eq!(
                got[pos], expected,
                "{label}: frontier mismatch at pos={pos}\n got={:?}\n exp={:?}",
                got[pos], expected
            );
        }
    }

    #[test]
    fn frontier_random() {
        // LCG pseudo-random bytes.
        let mut s: u32 = 0x1234_5678;
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (s >> 16) as u8
            })
            .collect();
        assert_frontier(&data, "random");
    }

    #[test]
    fn frontier_dna_4symbol() {
        // Low-entropy 4-symbol (DNA-like) input: many length-3+ matches.
        let alphabet = *b"ACGT";
        let mut s: u32 = 0xdead_beef;
        let data: Vec<u8> = (0..8192)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                alphabet[((s >> 16) & 3) as usize]
            })
            .collect();
        assert_frontier(&data, "dna");
    }

    #[test]
    fn frontier_runs() {
        // Alternating long runs of the same byte plus short transitions — the
        // pathological input the tree could mishandle.
        let mut data = Vec::new();
        for (i, &b) in b"abcde".iter().enumerate() {
            data.extend(std::iter::repeat_n(b, 200 + i * 37));
        }
        data.extend_from_slice(b"abcdeabcdeabcde");
        assert_frontier(&data, "runs");
    }

    #[test]
    fn frontier_text() {
        let data = b"the quick brown fox jumps over the lazy dog. \
                     the quick brown fox jumps over the lazy dog. \
                     pack my box with five dozen liquor jugs. \
                     pack my box with five dozen liquor jugs."
            .to_vec();
        assert_frontier(&data, "text");
    }

    #[test]
    fn frontier_prewindow_warms_dictionary() {
        // With a pre-window, matches into the dictionary must still be found.
        let data = b"COMMON_PREFIX_STRING and then COMMON_PREFIX_STRING again".to_vec();
        let instart = 20;
        let mut mf = MatchFinder::new(&data, 0, instart, data.len());
        let mut scratch = vec![0u16; MAX_MATCH_U16];
        for pos in instart..data.len() {
            let n = mf.get_matches(&mut scratch);
            let mut got = Vec::new();
            let mut k = 0;
            while k < n {
                got.push((scratch[k], scratch[k + 1]));
                k += 2;
            }
            assert_eq!(
                got,
                brute_pairs(&data, pos),
                "prewindow mismatch at pos={pos}"
            );
        }
    }
}
