//! Shared, thread-safe port of `rapidgzip::WindowMap` (WindowMap.hpp).
//! Stores propagated 32-KiB windows keyed by the compressed-bit offset
//! they're meant to seed. Workers AND the consumer share one handle.
//!
//! Audit step 5 (2026-05-17) — removed the `Condvar` that workers
//! formerly waited on via `get_or_wait`. Vendor's `WindowMap`
//! (vendor/rapidgzip/.../WindowMap.hpp:19-186) is a plain
//! `std::map<size_t, SharedWindow>` guarded by `mutable std::mutex` —
//! NO `condition_variable`, NO `wait`. Workers don't block on the
//! WindowMap because the `BlockFetcher::get` dispatch model
//! (BlockFetcher.hpp:245-329) waits on the per-block future, which
//! guarantees the predecessor's decode (and its tail-window emplace)
//! has finished before the consumer pulls the next chunk.
//!
//! **Window type (2026-05-18)** — `Window = Arc<CompressedVector>`,
//! matching vendor's `SharedWindow = shared_ptr<const Window>` exactly
//! (WindowMap.hpp:22-24). `get` returns the shared pointer with **zero
//! allocation** — vendor's WindowMap.hpp:79-90 pattern. Callers
//! materialize bytes on demand via `cv.raw_bytes()` (None-compression
//! path: zero-alloc slice borrow) or `cv.decompress()` (Zlib path: one
//! allocation for the decompressed buffer). Mirror of vendor's
//! consumer pattern at GzipChunkFetcher.hpp:341 (`sharedLastWindow->
//! decompress()`).

#![allow(dead_code)]

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
        self.state
            .lock()
            .unwrap()
            .entries
            .get(&encoded_offset_bits)
            .cloned()
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
    /// In gzippy this collision is real: the worker publishes the
    /// chunk's clean tail at `end_bit` (chunk_fetcher.rs:773) and the
    /// phase-2 post-process worker later publishes a per-subchunk
    /// window at the same key (chunk_fetcher.rs:836). Vendor's
    /// pattern is that the phase-2 (post-processed) version wins
    /// because it reflects the post-processed / sparsified state.
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
        let cv = Arc::new(CompressedVector::from_bytes(bytes, self.compression));
        self.insert(encoded_offset_bits, cv);
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
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
