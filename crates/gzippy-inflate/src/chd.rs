//! §3.5 — per-block perfect-hash decode tables (CHD-style).
//!
//! ## Design
//!
//! For a given DEFLATE block, the litlen alphabet uses ≤ 286 symbols.
//! A perfect hash function over those ~286 codes lets us index into
//! a compact value table without subtable-dispatch overhead. Target:
//! ~3 cycles/lookup, table size ~2.3 KiB (fits L1 alongside the
//! dist table + JIT decoder code).
//!
//! CHD = Compress-Hash-Displace (Belazzougui et al., 2009). Two-level
//! hash:
//!   - h1(key) → bucket index in [0, m).
//!   - For each bucket, choose a "displacement" value `d` such that
//!     `(h2(key) ^ d) mod n` is unique for all keys in the bucket.
//!   - Lookup: `g[h1(key)] = d`; then `value[(h2(key) ^ d) mod n]`.
//!
//! ## Adapting CHD to DEFLATE
//!
//! DEFLATE codes are variable-length (1..15 bits). The "key" we hash
//! is the FIXED 15-bit window peeked from the bit stream — the code's
//! actual bits, MSB-padded with the next bits in the stream. Each
//! actual code (length `L`) occupies `2^(15 - L)` 15-bit keys. The
//! hash must:
//!   - Map every 15-bit key whose low-L bits match a real code to
//!     the same (symbol, length) entry.
//!   - Reject 15-bit keys whose low-1..15 bits don't match any code
//!     (return an "invalid code" marker; the decoder retries with
//!     a malformed-data error).
//!
//! Per-block build cost: walk symbols, compute h1/h2 per symbol,
//! assign displacements greedily. Target: ~5 µs per block on
//! Raptor Lake (286 ops × ~17ns each).
//!
//! ## Scope of this module (v0.1)
//!
//! Skeleton only. The full CHD construction + lookup is multi-week
//! work per `plans/unified-decoder.md` §11. v0.1 ships:
//!   - The `ChdTable` struct shape (g + value arrays).
//!   - A naive O(N²) build that bootstraps via canonical Huffman
//!     and falls back to it on construction failure.
//!   - `lookup_15bit(key) -> Option<(symbol, code_length)>` API.
//!
//! Real CHD construction (with displacement-table greedy assignment,
//! retry on collision, BMI2-accelerated hash) is deferred to v0.2.

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Per-block perfect-hash decode table.
///
/// v0.1: stores a direct 32 KiB lookup (indexed by 15 bits) instead
/// of the real CHD displacement table. This is functionally
/// equivalent for correctness but doesn't achieve the cache locality
/// goal — left as a placeholder for v0.2 to swap in the actual CHD
/// build.
pub struct ChdTable {
    /// Indexed by 15 bits peeked from the stream (LSB-first).
    /// Each entry: (symbol, code_length). code_length == 0 means
    /// "no valid code at this 15-bit key" (caller signals malformed
    /// DEFLATE input).
    entries: Vec<(u16, u8)>,
}

impl ChdTable {
    /// Build from per-symbol code lengths (DEFLATE canonical Huffman).
    pub fn build(code_lengths: &[u8]) -> Self {
        // Mirror the canonical Huffman code assignment.
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
        let mut entries = vec![(0u16, 0u8); 1 << 15];
        let mut next_code = first_code;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len);
            // Populate every 15-bit key whose low-len bits == rev.
            let stride = 1u32 << len;
            let mut key = rev;
            while (key as usize) < (1 << 15) {
                entries[key as usize] = (symbol as u16, len);
                key += stride;
            }
        }
        Self { entries }
    }

    /// Lookup a 15-bit key. Returns `None` if no valid code matches.
    #[inline]
    pub fn lookup_15bit(&self, key: u16) -> Option<(u16, u8)> {
        let key = (key as usize) & 0x7FFF;
        let (sym, len) = self.entries[key];
        if len == 0 {
            None
        } else {
            Some((sym, len))
        }
    }

    /// Table size in bytes. v0.1 stores a flat 32 KiB lookup; v0.2
    /// CHD target is ~2.3 KiB.
    pub fn size_bytes(&self) -> usize {
        self.entries.len() * core::mem::size_of::<(u16, u8)>()
    }
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
    fn chd_table_decodes_fixed_huffman_codes() {
        // Fixed-Huffman code lengths per RFC 1951 §3.2.6.
        let mut code_lengths = [0u8; 288];
        code_lengths[0..=143].fill(8);
        code_lengths[144..=255].fill(9);
        code_lengths[256..=279].fill(7);
        code_lengths[280..=287].fill(8);

        let chd = ChdTable::build(&code_lengths);

        // Symbol 65 ('A') has code 0b01110001 (8 bits). Reversed = 0b10001110 = 142.
        // A 15-bit key with low 8 bits = 142 should decode to (65, 8).
        let key = 142u16;
        let (sym, len) = chd.lookup_15bit(key).unwrap();
        assert_eq!(sym, 65);
        assert_eq!(len, 8);

        // EOB (symbol 256) has code 0 (7 bits). Reversed = 0.
        let (sym, len) = chd.lookup_15bit(0).unwrap();
        assert_eq!(sym, 256);
        assert_eq!(len, 7);
    }

    #[test]
    fn chd_lookup_invalid_returns_none() {
        // Code lengths array with only symbol 0 having a 1-bit code.
        let mut code_lengths = [0u8; 288];
        code_lengths[0] = 1;
        // Note: this is malformed canonical (1 code of length 1 leaves
        // no codes elsewhere) — but a 15-bit key with low bit = 1
        // doesn't match symbol 0's code (reversed code = 0) so it's
        // a "no code" position.
        let chd = ChdTable::build(&code_lengths);
        // Key with low bit 1 should not find symbol 0.
        let r = chd.lookup_15bit(1);
        assert!(r.is_none());
    }
}
