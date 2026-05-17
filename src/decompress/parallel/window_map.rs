//! Shared, thread-safe port of `rapidgzip::WindowMap` (WindowMap.hpp).
//! Stores propagated 32-KiB windows keyed by the compressed-bit offset
//! they're meant to seed. Workers AND the consumer share one handle.
//!
//! Workers may call `get_or_wait(start_bit, timeout)` to check whether
//! the predecessor chunk's tail has been published (enabling the fast
//! path in [`crate::decompress::parallel::gzip_chunk::decode_chunk_with_window`]).
//! On insert, all waiters are notified.
//!
//! Windows are stored as `Arc<CompressedVector>` (Zlib by default,
//! matching rapidgzip's WindowMap default). Decompression happens
//! lazily on `get`/`get_or_wait` and yields a fresh owned 32 KiB
//! buffer so callers see the same `Arc<[u8; 32768]>` shape they always
//! have. The compressed storage gives ~3-10× memory savings on the
//! highly-redundant tail windows typical of real workloads.
//!
//! Mirror of rapidgzip's `mutable std::mutex` + `std::condition_variable`
//! pattern. BTreeMap is used so future range queries are cheap.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

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
#[derive(Clone)]
pub struct WindowMap {
    state: Arc<(Mutex<Inner>, Condvar)>,
    compression: CompressionType,
}

impl WindowMap {
    pub fn new() -> Self {
        Self {
            state: Arc::new((
                Mutex::new(Inner {
                    entries: BTreeMap::new(),
                }),
                Condvar::new(),
            )),
            compression: DEFAULT_WINDOW_COMPRESSION,
        }
    }

    /// Construct a WindowMap whose stored windows use the given
    /// compression type. Mostly used in tests / for benchmarks; prod
    /// uses `new()` with `DEFAULT_WINDOW_COMPRESSION`.
    pub fn with_compression(compression: CompressionType) -> Self {
        Self {
            state: Arc::new((
                Mutex::new(Inner {
                    entries: BTreeMap::new(),
                }),
                Condvar::new(),
            )),
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
    pub fn get(&self, encoded_offset_bits: usize) -> Option<Window> {
        let (lock, _cvar) = &*self.state;
        let cv = {
            let inner = lock.lock().unwrap();
            inner.entries.get(&encoded_offset_bits).cloned()
        };
        cv.map(|cv| decompress_to_window(&cv))
    }

    /// Block until the window keyed at `encoded_offset_bits` is inserted
    /// or `timeout` elapses. Returns the decompressed 32 KiB window on
    /// success, None on timeout. Mirror of rapidgzip's
    /// `WindowMap::get` with a wait.
    pub fn get_or_wait(&self, encoded_offset_bits: usize, timeout: Duration) -> Option<Window> {
        let (lock, cvar) = &*self.state;
        let deadline = std::time::Instant::now() + timeout;
        let mut inner = lock.lock().unwrap();
        loop {
            if let Some(cv) = inner.entries.get(&encoded_offset_bits).cloned() {
                drop(inner);
                return Some(decompress_to_window(&cv));
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let remaining = deadline - now;
            let (i, _wait) = cvar.wait_timeout(inner, remaining).unwrap();
            inner = i;
        }
    }

    /// Insert a window keyed at `encoded_offset_bits`. The 32 KiB
    /// buffer is compressed with this map's strategy on insertion.
    /// Wakes all waiters. Idempotent: if a window already exists for
    /// the key, the existing value is preserved (workers and consumer
    /// can race to insert the same end_bit window; the first wins).
    pub fn insert(&self, encoded_offset_bits: usize, window: Window) {
        let cv = Arc::new(CompressedVector::from_bytes(&window[..], self.compression));
        self.insert_compressed(encoded_offset_bits, cv);
    }

    /// Insert an already-compressed window. Cheap path for upstream
    /// callers that hold an `Arc<CompressedVector>` already.
    pub fn insert_compressed(&self, encoded_offset_bits: usize, compressed: Arc<CompressedVector>) {
        let (lock, cvar) = &*self.state;
        let mut inner = lock.lock().unwrap();
        inner
            .entries
            .entry(encoded_offset_bits)
            .or_insert(compressed);
        cvar.notify_all();
    }

    pub fn len(&self) -> usize {
        let (lock, _cvar) = &*self.state;
        let inner = lock.lock().unwrap();
        inner.entries.len()
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
    fn get_or_wait_times_out_when_absent() {
        let m = WindowMap::new();
        let t0 = std::time::Instant::now();
        let result = m.get_or_wait(99, Duration::from_millis(50));
        assert!(result.is_none());
        assert!(t0.elapsed() >= Duration::from_millis(40));
    }

    #[test]
    fn get_or_wait_wakes_on_insert() {
        let m = WindowMap::new();
        let m2 = m.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            m2.insert(7, window_of(0xEE));
        });
        let result = m.get_or_wait(7, Duration::from_millis(500));
        handle.join().unwrap();
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
