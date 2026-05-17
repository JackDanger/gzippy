//! Port of `rapidgzip::WindowMap` (WindowMap.hpp). Stores propagated
//! 32-KiB windows keyed by the compressed-bit-offset they're meant to
//! seed. Append-only; insertions must be in strictly increasing key
//! order (matches rapidgzip's BlockMap invariant).
//!
//! Used by the chunk fetcher (next module): as chunks complete in
//! order, the main thread extracts the last 32 KiB of each chunk's
//! output and inserts it under the chunk's `encoded_offset_bits +
//! encoded_size_bits` key. Workers reading from `WindowMap::get(offset)`
//! for the next chunk wait until the value appears.
//
// Allowed dead_code: step 6 of rapidgzip-port-design.md migration;
// consumed by chunk_fetcher.rs in step 7.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub struct WindowMap {
    entries: BTreeMap<usize, Arc<[u8; 32768]>>,
}

impl WindowMap {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn get(&self, encoded_offset_bits: usize) -> Option<Arc<[u8; 32768]>> {
        self.entries.get(&encoded_offset_bits).cloned()
    }

    /// Insert a window keyed at `encoded_offset_bits`. Must be strictly
    /// greater than any previously inserted key. Mirror of rapidgzip's
    /// `BlockMap::push` invariant (BlockMap.hpp:99-106).
    pub fn insert(&mut self, encoded_offset_bits: usize, window: Arc<[u8; 32768]>) {
        debug_assert!(
            self.entries
                .keys()
                .next_back()
                .is_none_or(|last| *last < encoded_offset_bits),
            "WindowMap insertion must be strictly increasing (last={:?}, new={})",
            self.entries.keys().next_back(),
            encoded_offset_bits
        );
        self.entries.insert(encoded_offset_bits, window);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn window_of(value: u8) -> Arc<[u8; 32768]> {
        Arc::new([value; 32768])
    }

    #[test]
    fn new_is_empty() {
        let m = WindowMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert!(m.get(0).is_none());
    }

    #[test]
    fn insert_and_get_round_trips() {
        let mut m = WindowMap::new();
        m.insert(0, window_of(0xAA));
        m.insert(1024, window_of(0xBB));
        m.insert(2048, window_of(0xCC));
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(0).unwrap()[0], 0xAA);
        assert_eq!(m.get(1024).unwrap()[0], 0xBB);
        assert_eq!(m.get(2048).unwrap()[0], 0xCC);
    }

    #[test]
    fn get_returns_arc_clone() {
        let mut m = WindowMap::new();
        let w = window_of(0x42);
        m.insert(100, w.clone());
        let retrieved = m.get(100).unwrap();
        assert!(Arc::ptr_eq(&w, &retrieved));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "strictly increasing")]
    fn insert_out_of_order_panics_in_debug() {
        let mut m = WindowMap::new();
        m.insert(2048, window_of(0));
        m.insert(1024, window_of(0)); // smaller — invariant violation
    }
}
