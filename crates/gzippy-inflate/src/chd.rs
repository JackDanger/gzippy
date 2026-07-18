//! §3.5 — per-block perfect-hash decode tables (CHD).
//!
//! ## Design (v0.2 — real CHD displacement table)
//!
//! CHD = Compress-Hash-Displace (Belazzougui et al., 2009). Two-level
//! perfect hash:
//!   - h1(key) → bucket index in `[0, m)`.
//!   - For each bucket (largest first), choose a displacement `d` such
//!     that `(h2(key) ^ d) mod n` is unique across all assigned keys.
//!   - Lookup: `value[(h2(key) ^ displacements[h1(key)]) mod n]`.
//!
//! ## Adapting CHD to DEFLATE
//!
//! DEFLATE codes are variable-length (1..15 bits) and stored MSB-first
//! into LSB-first byte stream → caller peeks a 15-bit window and we
//! key on `(reversed_codeword, length)` per actual code.
//!
//! For each (code, len) pair, the LUT slot we'd traditionally
//! populate for EVERY 15-bit key whose low-`len` bits == rev_code
//! becomes a single CHD entry that records `(symbol, length)`. At
//! lookup time, the caller peeks 15 bits and we strip the high bits
//! per-length until we find a match; the per-length table is small
//! enough that this is cache-friendly.
//!
//! ## v0.2 implementation
//!
//! We use a simplified two-table scheme that achieves the same
//! "no flat 32 KiB table" goal:
//!   - `displacements: [u16; m]` where m ≈ N/4 (typically 64-128
//!     for the 286-symbol litlen alphabet).
//!   - `values: Vec<(u16, u8)>` sized to next-prime above N.
//!   - On lookup, h1 picks a bucket, h2 + displacement picks the
//!     value index.
//!
//! For DEFLATE specifically we ALSO need to validate the lookup
//! result (since CHD only guarantees uniqueness for INSERTED keys;
//! any 15-bit pattern that doesn't correspond to a real codeword
//! must return None). We store the original 15-bit key alongside
//! the value and compare at lookup.
//!
//! ## Tradeoffs vs v0.1 flat table
//!
//! v0.1: 64 KiB flat lookup, O(1) lookup with 1 load. Cache-unfriendly
//! for the LARGE litlen alphabet (286 entries × 256 bits = ~9 KB of
//! "live" working set per block, vs the 64 KiB table's 16 cache lines
//! per lookup).
//!
//! v0.2: ~2-3 KiB combined (displacements + values + keys). One extra
//! load on lookup but stays in L1. Per-block build cost ~10 µs
//! (vs v0.1's instant flat-array fill).

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Per-block perfect-hash decode table (v0.2 CHD).
pub struct ChdTable {
    /// CHD displacement table. `displacements[h1(key) % m] = d`.
    displacements: Vec<u16>,
    /// Value table. `values[(h2(key) ^ d) % n] = (key, symbol, length)`.
    values: Vec<ChdEntry>,
    /// Modulus for the displacement table (m).
    m: usize,
    /// Modulus for the value table (n).
    n: usize,
}

#[derive(Clone, Copy, Debug)]
struct ChdEntry {
    /// Original 15-bit canonical key (with low `length` bits being the
    /// reversed codeword). `0xFFFF` sentinel = empty slot.
    key: u16,
    /// Symbol decoded at this position. Unused when slot is empty.
    symbol: u16,
    /// Code length in bits. `0` when slot is empty.
    length: u8,
}

impl ChdEntry {
    const EMPTY: ChdEntry = ChdEntry {
        key: 0xFFFF,
        symbol: 0,
        length: 0,
    };
}

/// Public CHD lookup result.
pub type ChdLookup = Option<(u16, u8)>;

impl ChdTable {
    /// Build from per-symbol code lengths (DEFLATE canonical Huffman).
    /// Returns the displacement-table CHD layout (v0.2).
    pub fn build(code_lengths: &[u8]) -> Self {
        // First materialize the set of (key, symbol, length) tuples
        // where key = reverse_bits(canonical_code, length).
        let mut count = [0u16; 16];
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                count[len as usize] += 1;
            }
        }
        let mut first_code = [0u32; 16];
        let mut code: u32 = 0;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }
        let mut next_code = first_code;

        // Collect codes (we only insert one entry per code, NOT
        // per-15-bit-key — the lookup masks per length).
        let mut codes: Vec<(u16, u16, u8)> = Vec::new(); // (key, symbol, length)
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len) as u16;
            codes.push((rev, symbol as u16, len));
        }

        if codes.is_empty() {
            // Empty table — every lookup returns None.
            return Self {
                displacements: vec![0],
                values: vec![ChdEntry::EMPTY],
                m: 1,
                n: 1,
            };
        }

        let n_keys = codes.len();
        // Value table size: next prime above n_keys * 5/4 for low
        // load factor (helps the greedy displacement assignment
        // converge fast). For DEFLATE's typical 286 codes this is
        // ~360, well within L1.
        let n = next_prime((n_keys * 5 / 4).max(16));
        // Bucket count: ~n_keys / 4 buckets (CHD literature suggests
        // λ ≈ 4 entries/bucket maximizes compactness vs build cost).
        let m = ((n_keys / 4).max(4)).next_power_of_two();

        // Group codes into buckets by h1(key) % m.
        let mut buckets: Vec<Vec<(u16, u16, u8)>> = vec![Vec::new(); m];
        for &(key, sym, len) in &codes {
            let b = (hash1(key) as usize) % m;
            buckets[b].push((key, sym, len));
        }

        // Sort bucket indices by descending size for greedy assignment.
        let mut bucket_order: Vec<usize> = (0..m).collect();
        bucket_order.sort_by_key(|&i| std::cmp::Reverse(buckets[i].len()));

        let mut displacements = vec![0u16; m];
        let mut values = vec![ChdEntry::EMPTY; n];

        // Greedy CHD: for each bucket (largest first), try
        // displacements 0..MAX_DISP; the first d that places every
        // bucket member into an empty value slot wins.
        const MAX_DISP: u32 = 65536;
        for &b in &bucket_order {
            let bucket = &buckets[b];
            if bucket.is_empty() {
                continue;
            }
            let mut found = false;
            for d in 0..MAX_DISP {
                let mut slots: Vec<usize> = Vec::with_capacity(bucket.len());
                let mut ok = true;
                for &(key, _sym, _len) in bucket {
                    let slot = ((hash2(key) ^ d) as usize) % n;
                    if values[slot].length != 0 || slots.contains(&slot) {
                        ok = false;
                        break;
                    }
                    slots.push(slot);
                }
                if ok {
                    displacements[b] = d as u16;
                    for (i, &(key, sym, len)) in bucket.iter().enumerate() {
                        values[slots[i]] = ChdEntry {
                            key,
                            symbol: sym,
                            length: len,
                        };
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                // CHD construction failed — fall back to a sparse layout
                // by growing n. For DEFLATE-sized alphabets this should
                // be rare with the 5/4 load factor; we panic in v0.2
                // and document the rebuild path for v0.3.
                panic!(
                    "CHD construction failed for bucket {} of size {} (n={}, m={})",
                    b,
                    bucket.len(),
                    n,
                    m
                );
            }
        }

        Self {
            displacements,
            values,
            m,
            n,
        }
    }

    /// Lookup a 15-bit key. Returns `Some((symbol, code_length))` if
    /// the low-`code_length` bits of `key` match an inserted codeword;
    /// returns `None` otherwise.
    ///
    /// Implementation: peek the 15-bit key. For DEFLATE, the actual
    /// codeword length is unknown at lookup time, so we try keys
    /// masked to each plausible length (1..15) and return the first
    /// CHD match whose stored key matches the masked input. This is
    /// a small fixed cost (~15 iterations max, typically 8-9 since
    /// most DEFLATE codes are 7-9 bits).
    ///
    /// A future v0.3 will eliminate the per-length scan by storing
    /// a length-discriminating hint in the displacement table.
    #[inline]
    pub fn lookup_15bit(&self, key: u16) -> ChdLookup {
        // Try each plausible code length, shortest first (so common
        // short codes terminate the loop fast).
        for len in 1u8..=15 {
            let masked = key & ((1u16 << len) - 1);
            let b = (hash1(masked) as usize) % self.m;
            let d = self.displacements[b];
            let slot = ((hash2(masked) ^ d as u32) as usize) % self.n;
            let entry = self.values[slot];
            if entry.length == len && entry.key == masked {
                return Some((entry.symbol, entry.length));
            }
        }
        None
    }

    /// Total table size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.displacements.len() * core::mem::size_of::<u16>()
            + self.values.len() * core::mem::size_of::<ChdEntry>()
    }
}

/// Murmur-style fast hash for 16-bit keys. h1 is the bucket hash;
/// h2 is the value-table hash. They MUST be uncorrelated for CHD to
/// converge fast on typical inputs.
#[inline(always)]
fn hash1(key: u16) -> u32 {
    let mut x = key as u32;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

#[inline(always)]
fn hash2(key: u16) -> u32 {
    let mut x = key as u32 ^ 0xdead_beef;
    x = x.wrapping_mul(0xff51_afd7);
    x ^= x >> 13;
    x = x.wrapping_mul(0x4c19_5f5e);
    x ^= x >> 16;
    x
}

fn next_prime(n: usize) -> usize {
    fn is_prime(n: usize) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }
        let mut i = 3;
        while i * i <= n {
            if n.is_multiple_of(i) {
                return false;
            }
            i += 2;
        }
        true
    }
    let mut k = n;
    while !is_prime(k) {
        k += 1;
    }
    k
}

fn reverse_bits(mut code: u32, n: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..n {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chd_v2_decodes_fixed_huffman_codes() {
        let mut code_lengths = [0u8; 288];
        code_lengths[0..=143].fill(8);
        code_lengths[144..=255].fill(9);
        code_lengths[256..=279].fill(7);
        code_lengths[280..=287].fill(8);

        let chd = ChdTable::build(&code_lengths);

        // Walk every (symbol, code) pair and verify the lookup
        // returns the matching symbol+length.
        let mut count = [0u16; 16];
        for &len in code_lengths.iter() {
            if len > 0 {
                count[len as usize] += 1;
            }
        }
        let mut first_code = [0u32; 16];
        let mut code: u32 = 0;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }
        let mut next_code = first_code;
        let mut checked = 0;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len) as u16;
            // The 15-bit key has the rev'd codeword in the low `len` bits;
            // upper bits can be anything (we use 0 here).
            let key = rev;
            let r = chd.lookup_15bit(key).unwrap_or_else(|| {
                panic!("CHD lookup failed for symbol {symbol}, len {len}, key 0x{key:04x}")
            });
            assert_eq!(r.0, symbol as u16, "wrong symbol at key 0x{key:04x}");
            assert_eq!(r.1, len, "wrong length at key 0x{key:04x}");
            checked += 1;
        }
        assert_eq!(checked, 288, "should check all 288 fixed-Huffman codes");
    }

    #[test]
    fn chd_v2_size_under_4kib() {
        let mut code_lengths = [0u8; 288];
        code_lengths[0..=143].fill(8);
        code_lengths[144..=255].fill(9);
        code_lengths[256..=279].fill(7);
        code_lengths[280..=287].fill(8);
        let chd = ChdTable::build(&code_lengths);
        let size = chd.size_bytes();
        eprintln!("CHD v2 table size for fixed-Huffman: {size} bytes");
        // Plan §3.5 target: ~2.3 KiB. We're more lax in v0.2 since
        // we store the key + symbol + length per entry (6 bytes) vs
        // the theoretical 3 bytes — this is the correctness-first
        // version.
        assert!(
            size <= 4 * 1024,
            "CHD table {size} bytes exceeds 4 KiB target"
        );
    }

    #[test]
    fn chd_v2_lookup_invalid_returns_none() {
        // Build a sparse table: only symbol 0 with code length 8.
        let mut code_lengths = [0u8; 288];
        code_lengths[0] = 8;
        let chd = ChdTable::build(&code_lengths);
        // Symbol 0 has code 0 (8 bits), reversed = 0. Key 0 with low 8
        // bits = 0 should hit.
        assert_eq!(chd.lookup_15bit(0).unwrap(), (0, 8));
        // Any other low-8-bits pattern should NOT hit.
        for key in 1u16..=255 {
            let r = chd.lookup_15bit(key);
            assert!(
                r.is_none() || r.unwrap().0 != 0 || r.unwrap().1 != 8,
                "unexpected hit at key 0x{key:04x}: {:?}",
                r
            );
        }
    }

    #[test]
    fn chd_v2_smaller_than_flat_table() {
        // Compare v2 (this impl) vs the flat 32 KiB lookup it replaces.
        let mut code_lengths = [0u8; 288];
        code_lengths[0..=143].fill(8);
        code_lengths[144..=255].fill(9);
        code_lengths[256..=279].fill(7);
        code_lengths[280..=287].fill(8);
        let chd = ChdTable::build(&code_lengths);
        let v0_1_size = (1 << 15) * core::mem::size_of::<(u16, u8)>();
        assert!(
            chd.size_bytes() < v0_1_size / 8,
            "v0.2 ({} bytes) should be at least 8x smaller than v0.1 ({} bytes)",
            chd.size_bytes(),
            v0_1_size
        );
    }
}
