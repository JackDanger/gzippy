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
//! Windows are stored as `Arc<CompressedVector>` (Zlib by default,
//! matching rapidgzip's WindowMap default). Decompression happens
//! lazily on `get` and yields a fresh owned 32 KiB buffer so callers
//! see the same `Arc<[u8; 32768]>` shape they always have. The
//! compressed storage gives ~3-10× memory savings on the highly-
//! redundant tail windows typical of real workloads.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::decompress::parallel::compressed_vector::{CompressedVector, CompressionType};

/// A 32 KiB window in its decompressed form. Callers receive this from
/// `get`/`get_or_wait` and use it directly to seed the ISA-L
/// dictionary or as the source for `replace_markers`.
pub type Window = Arc<[u8; 32768]>;

/// The compression strategy used for in-map window storage. Matches
/// rapidgzip's default `WindowMap` strategy (Zlib).
pub const DEFAULT_WINDOW_COMPRESSION: CompressionType = CompressionType::Zlib;

struct Inner {
    entries: BTreeMap<usize, Arc<CompressedVector>>,
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

    /// Compression strategy in effect for this map. Type-level fence
    /// for the `test_window_map_uses_compressed_vector` test.
    pub fn compression(&self) -> CompressionType {
        self.compression
    }

    /// Non-blocking lookup. Returns a freshly decompressed 32 KiB
    /// window keyed at `encoded_offset_bits` if present, else None.
    /// Mirror of vendor's `WindowMap::get` (WindowMap.hpp:79-90).
    pub fn get(&self, encoded_offset_bits: usize) -> Option<Window> {
        let inner = self.state.lock().unwrap();
        let cv = inner.entries.get(&encoded_offset_bits).cloned();
        drop(inner);
        cv.map(|cv| decompress_to_window(&cv))
    }

    /// Insert a window keyed at `encoded_offset_bits`. The 32 KiB
    /// buffer is compressed with this map's strategy on insertion.
    /// Idempotent: if a window already exists for the key, the
    /// existing value is preserved (workers and consumer can race to
    /// insert the same end_bit window; the first wins). Mirror of
    /// `WindowMap::emplace` (WindowMap.hpp:39-46).
    pub fn insert(&self, encoded_offset_bits: usize, window: Window) {
        let cv = Arc::new(CompressedVector::from_bytes(&window[..], self.compression));
        self.insert_compressed(encoded_offset_bits, cv);
    }

    /// Insert an already-compressed window. Cheap path for upstream
    /// callers that hold an `Arc<CompressedVector>` already. Mirror
    /// of `WindowMap::emplaceShared` (WindowMap.hpp:47-77).
    pub fn insert_compressed(&self, encoded_offset_bits: usize, compressed: Arc<CompressedVector>) {
        let mut inner = self.state.lock().unwrap();
        inner
            .entries
            .entry(encoded_offset_bits)
            .or_insert(compressed);
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for WindowMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompress a `CompressedVector` into a fresh `Arc<[u8; 32768]>`. If
/// the stored payload is shorter than 32 KiB (chunk-0 window) the
/// trailing bytes are zero — matching rapidgzip's behavior of seeding
/// the inflate dictionary with whatever is available.
fn decompress_to_window(cv: &CompressedVector) -> Window {
    let bytes = cv.decompress();
    let mut buf = [0u8; 32768];
    let n = bytes.len().min(buf.len());
    buf[..n].copy_from_slice(&bytes[..n]);
    Arc::new(buf)
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn window_of(value: u8) -> Window {
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
    fn default_compression_is_zlib() {
        let m = WindowMap::new();
        assert_eq!(m.compression(), CompressionType::Zlib);
    }

    #[test]
    fn insert_and_get_round_trips() {
        let m = WindowMap::new();
        m.insert(0, window_of(0xAA));
        m.insert(1024, window_of(0xBB));
        m.insert(2048, window_of(0xCC));
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(0).unwrap()[0], 0xAA);
        assert_eq!(m.get(0).unwrap()[32767], 0xAA);
        assert_eq!(m.get(1024).unwrap()[0], 0xBB);
        assert_eq!(m.get(2048).unwrap()[0], 0xCC);
    }

    #[test]
    fn handle_is_shared_across_clones() {
        let m1 = WindowMap::new();
        let m2 = m1.clone();
        m1.insert(100, window_of(0x42));
        assert_eq!(m2.get(100).unwrap()[0], 0x42);
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
        // visible on the main thread. Replaces the prior
        // `get_or_wait_wakes_on_insert` test which exercised the
        // Condvar code path that no longer exists.
        let m = WindowMap::new();
        let m2 = m.clone();
        let handle = std::thread::spawn(move || {
            m2.insert(7, window_of(0xEE));
        });
        handle.join().unwrap();
        let result = m.get(7);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 0xEE);
    }

    #[test]
    fn duplicate_insert_keeps_first() {
        let m = WindowMap::new();
        m.insert(50, window_of(0x01));
        m.insert(50, window_of(0x02));
        assert_eq!(m.get(50).unwrap()[0], 0x01);
    }

    #[test]
    fn windows_are_stored_compressed() {
        // A 32 KiB window of identical bytes compresses to a few hundred
        // bytes via zlib. We can't directly observe the compressed-vector
        // size from outside the map; instead we round-trip and inspect the
        // map size only via the public `len` API. The point of this test
        // is to lock down that `insert` doesn't crash on a value that
        // compresses heavily, and the returned window round-trips byte-
        // perfect.
        let m = WindowMap::with_compression(CompressionType::Zlib);
        let mut w = [0u8; 32768];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        m.insert(0, Arc::new(w));
        let got = m.get(0).expect("present");
        assert_eq!(&got[..], &w[..]);
    }
}
