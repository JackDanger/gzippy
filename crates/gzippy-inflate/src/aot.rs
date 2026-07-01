//! AOT-fingerprint runtime lookup. Pairs with `build.rs` which emits
//! `aot_fingerprints.rs` into `$OUT_DIR`.
//!
//! Runtime checks the parsed block's code-length fingerprint against
//! the AOT table; on hit, skips per-block CHD construction (plan §3.1
//! AOT half).

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Pull in the build.rs output.
include!(concat!(env!("OUT_DIR"), "/aot_fingerprints.rs"));

/// Stable 64-bit hash of (litlen | dist) code lengths. MUST match
/// `build.rs::fingerprint_hash` exactly so AOT entries and runtime
/// fingerprints share keys.
pub fn fingerprint_hash(litlen: &[u8], dist: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in litlen.iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    for &b in dist.iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Returns the human-readable name of the AOT'd fingerprint matching
/// `(litlen, dist)`, or `None` if no AOT entry matches.
pub fn match_aot_fingerprint(litlen: &[u8], dist: &[u8]) -> Option<&'static str> {
    let h = fingerprint_hash(litlen, dist);
    for &(fp, name) in AOT_FINGERPRINTS.iter() {
        if fp == h {
            return Some(name);
        }
    }
    None
}

/// AOT fingerprints baked in by build.rs.
pub fn all_aot_fingerprints() -> Vec<(Fingerprint, &'static str)> {
    AOT_FINGERPRINTS.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// build.rs must emit AT LEAST the RFC 1951 fixed-Huffman fingerprint.
    #[test]
    fn aot_table_has_fixed_huffman() {
        assert!(AOT_FINGERPRINT_COUNT >= 1);
        let mut litlen = [0u8; 288];
        for entry in litlen.iter_mut().take(144) {
            *entry = 8;
        }
        for entry in litlen.iter_mut().take(256).skip(144) {
            *entry = 9;
        }
        for entry in litlen.iter_mut().take(280).skip(256) {
            *entry = 7;
        }
        for entry in litlen.iter_mut().take(288).skip(280) {
            *entry = 8;
        }
        let dist = [5u8; 30];
        let name = match_aot_fingerprint(&litlen, &dist);
        assert_eq!(name, Some("fixed-huffman-rfc1951"));
    }

    /// Non-fixed-Huffman tables miss the AOT cache.
    #[test]
    fn random_fingerprint_misses() {
        let mut litlen = [0u8; 288];
        // Arbitrary alphabet — not the RFC 1951 fixed Huffman.
        for (i, entry) in litlen.iter_mut().enumerate() {
            *entry = (i % 10) as u8;
        }
        let dist = [4u8; 30];
        let name = match_aot_fingerprint(&litlen, &dist);
        assert_eq!(name, None);
    }

    /// `fingerprint_hash` is deterministic + stable.
    #[test]
    fn fingerprint_hash_is_deterministic() {
        let litlen = [5u8; 288];
        let dist = [5u8; 30];
        let h1 = fingerprint_hash(&litlen, &dist);
        let h2 = fingerprint_hash(&litlen, &dist);
        assert_eq!(h1, h2);
        // Different input → different hash (with overwhelming probability).
        let mut litlen2 = litlen;
        litlen2[0] = 6;
        let h3 = fingerprint_hash(&litlen2, &dist);
        assert_ne!(h1, h3);
    }
}
