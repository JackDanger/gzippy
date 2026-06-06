#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Shared, thread-safe port of `rapidgzip::WindowMap` (WindowMap.hpp).
//! Stores propagated 32-KiB windows keyed by the compressed-bit offset
//! they seed; workers and the consumer share one handle.
//!
//! A plain `BTreeMap<usize, Window>` behind a `Mutex` — no `Condvar`,
//! matching vendor's `std::map` + `std::mutex`. Workers never block on
//! the map: `BlockFetcher::get` waits on the per-block future, which
//! guarantees a predecessor's tail-window is emplaced before the
//! consumer pulls the next chunk.
//!
//! `Window = Arc<CompressedVector>` (vendor's `SharedWindow =
//! shared_ptr<const Window>`); `get` returns the shared pointer with
//! zero allocation. Callers materialize bytes via `cv.raw_bytes()`
//! (None-compression: borrowed slice) or `cv.decompress()` (Zlib: one
//! allocation).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::decompress::parallel::compressed_vector::{CompressedVector, CompressionType};

/// A 32 KiB window held in compressed form. Callers receive an
/// `Arc<CompressedVector>` from `get` and materialize the raw bytes
/// only when needed. Mirror of vendor's `SharedWindow =
/// shared_ptr<const CompressedVector>` (WindowMap.hpp:24).
pub type Window = Arc<CompressedVector>;

/// The compression strategy used for in-map window storage. Matches
/// rapidgzip's default `WindowMap` strategy (Zlib).
pub const DEFAULT_WINDOW_COMPRESSION: CompressionType = CompressionType::Zlib;

struct Inner {
    entries: BTreeMap<usize, Window>,
    /// Speculative side-slot (Design B). Windows here are the
    /// provably-clean tails of RANGE-speculative chunks (`max > encoded`)
    /// that the WORKER publishes right after decode — BEFORE the consumer
    /// confirms the chunk's start. They are keyed at the chunk's would-be
    /// `chunk_end_bit` on accept (= `encoded_offset_bits + encoded_size_bits`,
    /// which `set_encoded_offset` preserves through the consumer's accept
    /// rewrite — see `insert_speculative`'s callers for the proof).
    ///
    /// CORRECTNESS: this map is INTENTIONALLY invisible to `get`,
    /// `get_predecessor`, `has_predecessor`, and `contains` — a successor
    /// must NEVER resolve against an unconfirmed window. The consumer
    /// moves an entry into `entries` only at ACCEPT (`promote_speculative`)
    /// and drops it at REJECT (`evict_speculative`), both synchronously on
    /// the single consumer thread before it advances to any successor.
    speculative: BTreeMap<usize, Window>,
}

/// Thread-safe handle to the underlying map. Cheaply clonable; all
/// clones share state. Mirror of rapidgzip's `std::shared_ptr<WindowMap>`.
///
/// Locking matches vendor (WindowMap.hpp:172 `mutable std::mutex
/// m_mutex`): one `Mutex` around the `BTreeMap`, no `Condvar`.
#[derive(Clone)]
pub struct WindowMap {
    state: Arc<Mutex<Inner>>,
    compression: CompressionType,
}

impl WindowMap {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(Inner {
                entries: BTreeMap::new(),
                speculative: BTreeMap::new(),
            })),
            compression: DEFAULT_WINDOW_COMPRESSION,
        }
    }

    /// Construct a WindowMap whose stored windows use the given
    /// compression type. Mostly used in tests / for benchmarks; prod
    /// uses `new()` with `DEFAULT_WINDOW_COMPRESSION`.
    pub fn with_compression(compression: CompressionType) -> Self {
        Self {
            state: Arc::new(Mutex::new(Inner {
                entries: BTreeMap::new(),
                speculative: BTreeMap::new(),
            })),
            compression,
        }
    }

    /// Compression strategy in effect for this map.
    pub fn compression(&self) -> CompressionType {
        self.compression
    }

    /// Non-blocking lookup. Returns the shared `Arc<CompressedVector>`
    /// keyed at `encoded_offset_bits` if present, else None. **Zero
    /// allocation** on hit — only an `Arc` ref-count bump. Mirror of
    /// vendor's `WindowMap::get` (WindowMap.hpp:79-90).
    pub fn get(&self, encoded_offset_bits: usize) -> Option<Window> {
        let r = self
            .state
            .lock()
            .unwrap()
            .entries
            .get(&encoded_offset_bits)
            .cloned();
        r
    }

    /// Insert a pre-built `CompressedVector` keyed at
    /// `encoded_offset_bits`. **Overwriting semantics** — if a window
    /// already exists for the key, the new value replaces it. Mirror
    /// of vendor's `WindowMap::emplaceShared` (WindowMap.hpp:65-76),
    /// which uses `std::map::insert_or_assign`: "Simply overwrite
    /// windows if they do exist already... overwriting non-compressed
    /// windows with asynchronically compressed and made-sparse
    /// windows."
    ///
    /// Overwrites are intentional when the consumer publishes a
    /// subchunk window after the critical-path tail (vendor
    /// `appendSubchunksToIndexes`, GzipChunkFetcher.hpp:429-458).
    pub fn insert(&self, encoded_offset_bits: usize, window: Window) {
        self.state
            .lock()
            .unwrap()
            .entries
            .insert(encoded_offset_bits, window);
    }

    /// Convenience: build a `CompressedVector` from raw bytes (with
    /// this map's compression strategy) and insert. Overwrite
    /// semantics — see `insert`. Mirror of vendor's
    /// `WindowMap::emplace(offset, WindowView, CompressionType)`
    /// (WindowMap.hpp:39-46), which is one line:
    /// `emplaceShared(offset, make_shared<Window>(window, type))`.
    pub fn insert_bytes(&self, encoded_offset_bits: usize, bytes: &[u8]) {
        self.insert_bytes_with_compression(encoded_offset_bits, bytes, self.compression);
    }

    /// Zero-copy `CompressionType::None` insert: takes OWNERSHIP of a
    /// freshly-built tail `Vec<u8>` and wraps it directly in the map's
    /// `Arc<CompressedVector>`, with NO `to_vec()`. Behaviourally
    /// identical to `insert_bytes_with_compression(offset, &bytes,
    /// CompressionType::None)` but saves the second 32 KiB heap copy on
    /// the consumer's serial publish path (the gate for successor
    /// dispatch). See `CompressedVector::from_owned_none`.
    pub fn insert_owned_none(&self, encoded_offset_bits: usize, bytes: Vec<u8>) {
        // memlife: a 32 KiB tail-window buffer built by get_last_window_vec /
        // last_32kib_window_vec then stored here. rapidgzip stores a COMPRESSED
        // window in WindowMap; gzippy's owned-none stores it raw. Record the
        // alloc + the write that built it. (Gated: memlife is parallel_sm-only;
        // window_map itself is compiled in non-parallel_sm builds too.)
        #[cfg(parallel_sm)]
        {
            use crate::decompress::parallel::memlife::{self, AllocPath, Component};
            memlife::alloc(Component::Window, bytes.len(), AllocPath::Glibc);
            memlife::written(Component::Window, bytes.len());
        }
        let cv = Arc::new(CompressedVector::from_owned_none(bytes));
        self.insert(encoded_offset_bits, cv);
    }

    /// Mirror of vendor `WindowMap::emplace(offset, window, compressionType)`.
    pub fn insert_bytes_with_compression(
        &self,
        encoded_offset_bits: usize,
        bytes: &[u8],
        compression: CompressionType,
    ) {
        let cv = Arc::new(CompressedVector::from_bytes(bytes, compression));
        self.insert(encoded_offset_bits, cv);
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }

    /// Diagnostic: snapshot all published window keys (encoded bit offsets).
    pub fn keys_snapshot(&self) -> Vec<usize> {
        self.state.lock().unwrap().entries.keys().copied().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Presence-only lookup. Returns true without cloning the Arc —
    /// for callers that only need to test whether a predecessor's
    /// tail has been published. Vendor's `WindowMap::get` already
    /// returns a `shared_ptr<const Window>` (zero alloc), so a
    /// presence test via `get(..).is_some()` already costs only one
    /// Arc clone; this API saves even that.
    pub fn contains(&self, encoded_offset_bits: usize) -> bool {
        self.state
            .lock()
            .unwrap()
            .entries
            .contains_key(&encoded_offset_bits)
    }

    /// Largest published window key at or before `encoded_offset_bits`.
    /// Used when a chunk's partition seed overshoots the predecessor's
    /// published tail key (spacing guess > last confirmed end).
    pub fn get_predecessor(&self, encoded_offset_bits: usize) -> Option<(usize, Window)> {
        let g = self.state.lock().unwrap();
        g.entries
            .range(..=encoded_offset_bits)
            .next_back()
            .map(|(k, v)| (*k, v.clone()))
    }

    /// Worker-only exact lookup: confirmed `entries` then `speculative`.
    /// Consumer `get` stays confirmed-only so successors never resolve
    /// against un-promoted tails.
    pub fn get_at_worker(&self, encoded_offset_bits: usize) -> Option<Window> {
        let g = self.state.lock().unwrap();
        g.entries
            .get(&encoded_offset_bits)
            .or_else(|| g.speculative.get(&encoded_offset_bits))
            .cloned()
    }

    /// Worker-only predecessor from **confirmed** `entries` only.
    /// Used for Design H′ (decode at `pred_key` before partition seed): speculative
    /// side-slot keys can be chain-invalid on low-entropy fixtures.
    pub fn get_confirmed_predecessor_for_worker(
        &self,
        encoded_offset_bits: usize,
    ) -> Option<(usize, Window)> {
        let g = self.state.lock().unwrap();
        g.entries
            .range(..=encoded_offset_bits)
            .next_back()
            .map(|(k, v)| (*k, v.clone()))
    }

    /// Worker-only `get_predecessor` merging confirmed + speculative maps.
    pub fn get_predecessor_for_worker(
        &self,
        encoded_offset_bits: usize,
    ) -> Option<(usize, Window)> {
        let g = self.state.lock().unwrap();
        let pick = |map: &BTreeMap<usize, Window>| {
            map.range(..=encoded_offset_bits)
                .next_back()
                .map(|(k, v)| (*k, v.clone()))
        };
        match (pick(&g.entries), pick(&g.speculative)) {
            (Some((k1, w1)), Some((k2, w2))) => {
                if k1 >= k2 {
                    Some((k1, w1))
                } else {
                    Some((k2, w2))
                }
            }
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    pub fn has_predecessor(&self, encoded_offset_bits: usize) -> bool {
        self.get_predecessor(encoded_offset_bits).is_some()
    }

    /// Worker-only handoff lookup for KEY-MISMATCH speculative prefetch.
    /// Returns the largest published window key in `(low_exclusive, high_inclusive]`,
    /// merging confirmed `entries` and worker early-published `speculative` tails.
    /// Consumer `get_predecessor` / `contains` intentionally ignore speculative;
    /// without this merge, handoff decode misses ~97% of predecessor windows that
    /// exist only in the side-slot until consumer promote.
    pub fn get_handoff_in_partition(
        &self,
        low_exclusive: usize,
        high_inclusive: usize,
    ) -> Option<(usize, Window)> {
        use std::ops::Bound;
        if low_exclusive >= high_inclusive {
            return None;
        }
        let g = self.state.lock().unwrap();
        let pick = |map: &BTreeMap<usize, Window>| {
            map.range((
                Bound::Excluded(low_exclusive),
                Bound::Included(high_inclusive),
            ))
            .next_back()
            .map(|(k, v)| (*k, v.clone()))
        };
        match (pick(&g.entries), pick(&g.speculative)) {
            (Some((k1, w1)), Some((k2, w2))) => {
                if k1 >= k2 {
                    Some((k1, w1))
                } else {
                    Some((k2, w2))
                }
            }
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    // ── Speculative side-slot (Design B) ──────────────────────────────
    //
    // The worker publishes a RANGE-speculative chunk's provably-clean tail
    // window into a SEPARATE map keyed at the chunk's would-be accept-time
    // `chunk_end_bit`. Successor resolution (`get`/`get_predecessor`/
    // `has_predecessor`/`contains`) never reads this map, so an unconfirmed
    // window can never seed a successor. The consumer moves it into the
    // real map at ACCEPT (`promote_speculative`) or drops it at REJECT
    // (`evict_speculative`). Both run on the single consumer thread,
    // synchronously, before it advances to any successor → atomic w.r.t.
    // accept/reject from every reader's point of view.

    /// Worker-side: stash a speculative tail window keyed at `key`
    /// (the chunk's accept-time `chunk_end_bit`). Overwrite semantics
    /// match `insert` — a later worker re-decode at the same key wins.
    /// Returns whether an entry was newly inserted (false = overwrote).
    pub fn insert_speculative(&self, key: usize, window: Window) {
        self.state.lock().unwrap().speculative.insert(key, window);
    }

    /// Zero-copy `CompressionType::None` speculative insert — twin of
    /// `insert_owned_none` for the side-slot. Takes ownership of a freshly
    /// built 32 KiB tail `Vec<u8>` (no `to_vec`).
    pub fn insert_speculative_owned_none(&self, key: usize, bytes: Vec<u8>) {
        let cv = Arc::new(CompressedVector::from_owned_none(bytes));
        self.insert_speculative(key, cv);
    }

    /// Consumer-side ACCEPT: move the speculative window at `key` into the
    /// real map (making it visible to successors). No-op if absent (the
    /// worker may have skipped early-publish — e.g. unclean tail — in
    /// which case the consumer's serial publish handles the key). Returns
    /// whether a speculative entry was promoted.
    ///
    /// Promotion is an IDENTITY-KEY move: on accept the consumer's
    /// `chunk_end_bit` equals the worker's speculative key by construction
    /// (set_encoded_offset preserves `encoded_offset_bits + encoded_size_bits`),
    /// and the bytes are the chunk's predecessor-independent clean tail, so
    /// the promoted window is byte-identical to the window the consumer
    /// would otherwise publish serially.
    pub fn promote_speculative(&self, key: usize) -> bool {
        let mut g = self.state.lock().unwrap();
        if let Some(w) = g.speculative.remove(&key) {
            g.entries.insert(key, w);
            true
        } else {
            false
        }
    }

    /// Consumer-side REJECT: drop the speculative window at `key` so a
    /// stale unconfirmed tail can never be promoted later. No-op if absent.
    /// Returns whether an entry was evicted.
    pub fn evict_speculative(&self, key: usize) -> bool {
        self.state
            .lock()
            .unwrap()
            .speculative
            .remove(&key)
            .is_some()
    }

    /// Diagnostic: number of windows currently parked in the speculative
    /// side-slot (un-promoted, un-evicted).
    pub fn speculative_len(&self) -> usize {
        self.state.lock().unwrap().speculative.len()
    }
}

impl Default for WindowMap {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn window_of(value: u8) -> Window {
        Arc::new(CompressedVector::from_bytes(
            &[value; 32768],
            CompressionType::None,
        ))
    }

    /// Materialize bytes from a window for test assertions. Production
    /// callers do this via `cv.raw_bytes()` (None) or `cv.decompress()`
    /// (Zlib). Tests use None compression so `raw_bytes()` is exact.
    fn bytes_of(w: &Window) -> Vec<u8> {
        if w.compression_type() == CompressionType::None {
            w.raw_bytes().to_vec()
        } else {
            w.decompress()
        }
    }

    #[test]
    fn new_is_empty() {
        let m = WindowMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert!(m.get(0).is_none());
    }

    #[test]
    fn default_compression_is_zlib() {
        let m = WindowMap::new();
        assert_eq!(m.compression(), CompressionType::Zlib);
    }

    #[test]
    fn insert_and_get_round_trips() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert(0, window_of(0xAA));
        m.insert(1024, window_of(0xBB));
        m.insert(2048, window_of(0xCC));
        assert_eq!(m.len(), 3);
        let w0 = m.get(0).unwrap();
        assert_eq!(w0.raw_bytes()[0], 0xAA);
        assert_eq!(w0.raw_bytes()[32767], 0xAA);
        assert_eq!(m.get(1024).unwrap().raw_bytes()[0], 0xBB);
        assert_eq!(m.get(2048).unwrap().raw_bytes()[0], 0xCC);
    }

    #[test]
    fn handle_is_shared_across_clones() {
        let m1 = WindowMap::with_compression(CompressionType::None);
        let m2 = m1.clone();
        m1.insert(100, window_of(0x42));
        assert_eq!(m2.get(100).unwrap().raw_bytes()[0], 0x42);
    }

    #[test]
    fn get_returns_none_when_absent() {
        let m = WindowMap::new();
        assert!(m.get(99).is_none());
    }

    #[test]
    fn cross_thread_insert_is_visible() {
        // Vendor's WindowMap has no Condvar — callers are responsible
        // for ordering insert-then-get via higher-level synchronization
        // (per BlockFetcher::get's per-block future). This test mirrors
        // that: spawn an insert, join, then assert the value is
        // visible on the main thread.
        let m = WindowMap::with_compression(CompressionType::None);
        let m2 = m.clone();
        let handle = std::thread::spawn(move || {
            m2.insert(7, window_of(0xEE));
        });
        handle.join().unwrap();
        let result = m.get(7);
        assert!(result.is_some());
        assert_eq!(result.unwrap().raw_bytes()[0], 0xEE);
    }

    #[test]
    fn contains_is_zero_alloc_presence_check() {
        let m = WindowMap::with_compression(CompressionType::None);
        assert!(!m.contains(42));
        m.insert(42, window_of(0x37));
        assert!(m.contains(42));
        assert!(!m.contains(43));
    }

    #[test]
    fn duplicate_insert_overwrites_existing() {
        // Vendor WindowMap.hpp:75 uses `std::map::insert_or_assign`
        // explicitly so that phase-2 sparsified/compressed windows
        // replace the worker's earlier placeholder.
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert(50, window_of(0x01));
        m.insert(50, window_of(0x02));
        assert_eq!(m.get(50).unwrap().raw_bytes()[0], 0x02);
    }

    #[test]
    fn speculative_slot_is_invisible_to_successor_resolution() {
        // The whole correctness premise: an un-promoted speculative window
        // must NEVER be reachable by get / get_predecessor / has_predecessor
        // / contains — otherwise a successor resolves against an
        // unconfirmed (possibly-stale) window and corrupts output.
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert_speculative(1000, window_of(0xAB));
        assert!(m.get(1000).is_none());
        assert!(!m.contains(1000));
        assert!(!m.has_predecessor(1000));
        assert!(m.get_predecessor(1000).is_none());
        assert!(
            m.is_empty(),
            "speculative entry must not count as a real entry"
        );
        assert_eq!(m.speculative_len(), 1);
    }

    #[test]
    fn get_at_worker_reads_speculative_without_exposing_to_consumer_get() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert_speculative(2000, window_of(0x55));
        assert!(m.get(2000).is_none());
        assert_eq!(m.get_at_worker(2000).unwrap().raw_bytes()[0], 0x55);
    }

    #[test]
    fn get_predecessor_for_worker_merges_speculative() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert(100, window_of(0x11));
        m.insert_speculative(500, window_of(0x22));
        let (k, w) = m.get_predecessor_for_worker(1000).unwrap();
        assert_eq!(k, 500);
        assert_eq!(w.raw_bytes()[0], 0x22);
    }

    #[test]
    fn get_confirmed_predecessor_for_worker_ignores_speculative() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert(100, window_of(0x11));
        m.insert_speculative(500, window_of(0x22));
        let (k, w) = m.get_confirmed_predecessor_for_worker(1000).unwrap();
        assert_eq!(k, 100);
        assert_eq!(w.raw_bytes()[0], 0x11);
    }

    #[test]
    fn handoff_in_partition_merges_confirmed_and_speculative() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert(100, window_of(0x11));
        m.insert_speculative(5000, window_of(0x22));
        m.insert(9000, window_of(0x33));
        // Partition (1000, 10000]: largest key is 9000 confirmed.
        let (k, w) = m.get_handoff_in_partition(1000, 10000).unwrap();
        assert_eq!(k, 9000);
        assert_eq!(w.raw_bytes()[0], 0x33);
        // Partition (1000, 6000]: speculative 5000 wins over nothing else in range.
        let (k2, w2) = m.get_handoff_in_partition(1000, 6000).unwrap();
        assert_eq!(k2, 5000);
        assert_eq!(w2.raw_bytes()[0], 0x22);
        // Keys at or below low are excluded; empty interval returns None.
        assert!(m.get_handoff_in_partition(9000, 9000).is_none());
        assert!(m.get_handoff_in_partition(9500, 10000).is_none());
    }

    #[test]
    fn promote_makes_speculative_window_visible_with_same_key_and_bytes() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert_speculative(2048, window_of(0xCD));
        assert!(m.get(2048).is_none());
        assert!(m.promote_speculative(2048));
        // Now visible to successor resolution, byte-identical.
        let w = m.get(2048).expect("promoted into real map");
        assert_eq!(w.raw_bytes()[0], 0xCD);
        assert_eq!(w.raw_bytes()[32767], 0xCD);
        assert!(m.contains(2048));
        assert!(m.has_predecessor(2048));
        // Speculative slot is now empty (moved, not copied).
        assert_eq!(m.speculative_len(), 0);
        // Promoting an absent key is a harmless no-op.
        assert!(!m.promote_speculative(9999));
    }

    #[test]
    fn evict_drops_speculative_window_so_it_can_never_promote() {
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert_speculative(4096, window_of(0xEF));
        assert!(m.evict_speculative(4096));
        assert_eq!(m.speculative_len(), 0);
        // Still invisible, and a later promote is a no-op (no stale window).
        assert!(m.get(4096).is_none());
        assert!(!m.promote_speculative(4096));
        assert!(m.get(4096).is_none());
        // Evicting an absent key is a harmless no-op.
        assert!(!m.evict_speculative(123));
    }

    #[test]
    fn speculative_overwrite_keeps_latest_then_promotes_latest() {
        // A worker re-decode at the same key overwrites; promotion must
        // surface the latest bytes.
        let m = WindowMap::with_compression(CompressionType::None);
        m.insert_speculative(64, window_of(0x11));
        m.insert_speculative(64, window_of(0x22));
        assert_eq!(m.speculative_len(), 1);
        assert!(m.promote_speculative(64));
        assert_eq!(m.get(64).unwrap().raw_bytes()[0], 0x22);
    }

    #[test]
    fn speculative_owned_none_round_trips() {
        let m = WindowMap::with_compression(CompressionType::None);
        let bytes = vec![0x5A_u8; 32768];
        m.insert_speculative_owned_none(800, bytes);
        assert!(m.get(800).is_none());
        assert!(m.promote_speculative(800));
        let w = m.get(800).unwrap();
        assert_eq!(w.compression_type(), CompressionType::None);
        assert_eq!(w.raw_bytes()[0], 0x5A);
    }

    #[test]
    fn insert_bytes_round_trips_zlib() {
        // A 32 KiB window of identical bytes compresses to a few hundred
        // bytes via zlib. This test confirms insert_bytes constructs a
        // CompressedVector with the map's compression and round-trips.
        let m = WindowMap::with_compression(CompressionType::Zlib);
        let mut w = [0u8; 32768];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        m.insert_bytes(0, &w);
        let got = m.get(0).expect("present");
        assert_eq!(got.compression_type(), CompressionType::Zlib);
        assert!(got.compressed_size() < w.len());
        let materialized = bytes_of(&got);
        assert_eq!(&materialized[..], &w[..]);
    }
}
