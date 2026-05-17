//! Shared, thread-safe port of `rapidgzip::WindowMap` (WindowMap.hpp).
//! Stores propagated 32-KiB windows keyed by the compressed-bit offset
//! they're meant to seed. Workers AND the consumer share one handle.
//!
//! Workers may call `get_or_wait(start_bit, timeout)` to check whether
//! the predecessor chunk's tail has been published (enabling the fast
//! path in [`crate::decompress::parallel::gzip_chunk::decode_chunk_with_window`]).
//! On insert, all waiters are notified.
//!
//! Mirror of rapidgzip's `mutable std::mutex` + `std::condition_variable`
//! pattern. Used the BTreeMap structure so `get_lower_bound` / range
//! queries are available for future enhancements (currently only `get`
//! is used).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub type Window = Arc<[u8; 32768]>;

struct Inner {
    entries: BTreeMap<usize, Window>,
}

/// Thread-safe handle to the underlying map. Cheaply clonable; all
/// clones share state. Mirror of rapidgzip's `std::shared_ptr<WindowMap>`.
#[derive(Clone)]
pub struct WindowMap {
    state: Arc<(Mutex<Inner>, Condvar)>,
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
        }
    }

    /// Non-blocking lookup. Returns the window keyed at `encoded_offset_bits`
    /// if present, else None.
    pub fn get(&self, encoded_offset_bits: usize) -> Option<Window> {
        let (lock, _cvar) = &*self.state;
        let inner = lock.lock().unwrap();
        inner.entries.get(&encoded_offset_bits).cloned()
    }

    /// Block until the window keyed at `encoded_offset_bits` is inserted
    /// or `timeout` elapses. Returns the window on success, None on
    /// timeout. Mirror of rapidgzip's `WindowMap::get` with a wait.
    pub fn get_or_wait(&self, encoded_offset_bits: usize, timeout: Duration) -> Option<Window> {
        let (lock, cvar) = &*self.state;
        let deadline = std::time::Instant::now() + timeout;
        let mut inner = lock.lock().unwrap();
        loop {
            if let Some(w) = inner.entries.get(&encoded_offset_bits) {
                return Some(w.clone());
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

    /// Insert a window keyed at `encoded_offset_bits`. Wakes all waiters.
    /// Idempotent: if a window already exists for the key, the existing
    /// value is preserved (workers and consumer can race to insert the
    /// same end_bit window; the first wins; debug_assert in release
    /// builds is too aggressive for races).
    pub fn insert(&self, encoded_offset_bits: usize, window: Window) {
        let (lock, cvar) = &*self.state;
        let mut inner = lock.lock().unwrap();
        inner.entries.entry(encoded_offset_bits).or_insert(window);
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
    fn insert_and_get_round_trips() {
        let m = WindowMap::new();
        m.insert(0, window_of(0xAA));
        m.insert(1024, window_of(0xBB));
        m.insert(2048, window_of(0xCC));
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(0).unwrap()[0], 0xAA);
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
}
