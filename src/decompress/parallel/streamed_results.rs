#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
//! Literal port of `rapidgzip::StreamedResults`
//! (vendor/rapidgzip/librapidarchive/src/core/StreamedResults.hpp:27-158).
//!
//! Append-only result queue with `push` / `finalise` on the producer side
//! and blocking `get(position, timeout)` on consumers. Used by the async
//! raw block-finder coordinator (`raw_block_finder.rs`), mirroring vendor
//! `core/BlockFinder<RawFinder>`'s `m_blockOffsets` member
//! (BlockFinder.hpp:202).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Mirror of `BlockFinderInterface::GetReturnCode`
/// (vendor/.../core/BlockFinderInterface.hpp:13-17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamedGetReturnCode {
    Success,
    Failure,
    Timeout,
}

/// Mirror of `StreamedResults::Values` (StreamedResults.hpp:35).
#[allow(dead_code)] // vendor parity; no production caller yet
pub type StreamedValues<V> = VecDeque<V>;

struct Inner<V> {
    results: VecDeque<V>,
    finalized: bool,
}

/// Mirror of `rapidgzip::StreamedResults<Value>` (StreamedResults.hpp:27).
pub struct StreamedResults<V> {
    inner: Mutex<Inner<V>>,
    changed: Condvar,
}

impl<V: Clone> StreamedResults<V> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                results: VecDeque::new(),
                finalized: false,
            }),
            changed: Condvar::new(),
        }
    }

    /// Mirror of `size()` (StreamedResults.hpp:64-68).
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().results.len()
    }

    /// Mirror of empty check (StreamedResults.hpp — vendor `empty()`).
    #[allow(dead_code)] // vendor parity
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mirror of `finalized()` (StreamedResults.hpp:128-132).
    pub fn finalized(&self) -> bool {
        self.inner.lock().unwrap().finalized
    }

    /// Mirror of `push` (StreamedResults.hpp:98-109).
    pub fn push(&self, value: V) {
        let mut inner = self.inner.lock().unwrap();
        if inner.finalized {
            panic!("You may not push to finalized StreamedResults!");
        }
        inner.results.push_back(value);
        self.changed.notify_all();
    }

    /// Mirror of `finalize` (StreamedResults.hpp:111-126).
    pub fn finalize(&self) {
        self.finalize_to(None);
    }

    pub fn finalize_to(&self, results_count: Option<usize>) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(count) = results_count {
            if count > inner.results.len() {
                panic!("You may not finalize to a size larger than the current results buffer!");
            }
            inner.results.truncate(count);
        }
        inner.finalized = true;
        self.changed.notify_all();
    }

    /// Mirror of `setResults` (StreamedResults.hpp:141-149).
    #[allow(dead_code)] // vendor parity; no production caller yet
    pub fn set_results(&self, results: VecDeque<V>) {
        let mut inner = self.inner.lock().unwrap();
        inner.results = results;
        inner.finalized = true;
        self.changed.notify_all();
    }

    /// Mirror of `get(position, timeoutInSeconds)` (StreamedResults.hpp:75-96).
    ///
    /// Vendor timeout semantics:
    /// - `timeout == Duration::ZERO`: do not wait.
    /// - `timeout` very large / "forever": wait until available or finalized.
    pub fn get(&self, position: usize, timeout: Duration) -> (Option<V>, StreamedGetReturnCode) {
        let mut inner = self.inner.lock().unwrap();

        if !timeout.is_zero() {
            let predicate = |state: &Inner<V>| state.finalized || position < state.results.len();
            if timeout == Duration::MAX {
                while !predicate(&inner) {
                    inner = self.changed.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
            } else {
                let (guard, timed_out) = self
                    .changed
                    .wait_timeout(inner, timeout)
                    .unwrap_or_else(|e| e.into_inner());
                inner = guard;
                if timed_out.timed_out() && !predicate(&inner) {
                    return (None, StreamedGetReturnCode::Timeout);
                }
            }
        }

        if position < inner.results.len() {
            let value = inner.results[position].clone();
            (Some(value), StreamedGetReturnCode::Success)
        } else if inner.finalized {
            (None, StreamedGetReturnCode::Failure)
        } else {
            (None, StreamedGetReturnCode::Timeout)
        }
    }

    /// Mirror of `results()` / `ResultsView` (StreamedResults.hpp:39-61, 134-139).
    #[allow(dead_code)] // vendor parity; no production caller yet
    pub fn results_view(&self) -> ResultsView<'_, V> {
        let guard = self.inner.lock().unwrap();
        ResultsView { _guard: guard }
    }
}

impl<V: Clone> Default for StreamedResults<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII lock holder mirroring `StreamedResults::ResultsView`.
#[allow(dead_code)] // vendor parity; no production caller yet
pub struct ResultsView<'a, V> {
    _guard: std::sync::MutexGuard<'a, Inner<V>>,
}

impl<V: Clone> ResultsView<'_, V> {
    #[allow(dead_code)] // vendor parity; no production caller yet
    pub fn results(&self) -> &VecDeque<V> {
        &self._guard.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn push_get_success() {
        let sr = StreamedResults::<usize>::new();
        sr.push(100);
        sr.push(200);
        let (v, code) = sr.get(0, Duration::ZERO);
        assert_eq!(code, StreamedGetReturnCode::Success);
        assert_eq!(v, Some(100));
        let (v, code) = sr.get(1, Duration::ZERO);
        assert_eq!(code, StreamedGetReturnCode::Success);
        assert_eq!(v, Some(200));
    }

    #[test]
    fn get_before_push_times_out_with_zero_timeout() {
        let sr = StreamedResults::<usize>::new();
        let (v, code) = sr.get(0, Duration::ZERO);
        assert_eq!(v, None);
        assert_eq!(code, StreamedGetReturnCode::Timeout);
    }

    #[test]
    fn finalize_marks_past_end_as_failure() {
        let sr = StreamedResults::<usize>::new();
        sr.push(42);
        sr.finalize();
        let (v, code) = sr.get(1, Duration::ZERO);
        assert_eq!(v, None);
        assert_eq!(code, StreamedGetReturnCode::Failure);
    }

    #[test]
    fn blocking_get_waits_for_push() {
        use std::sync::Arc;
        let sr = Arc::new(StreamedResults::<usize>::new());
        let producer = Arc::clone(&sr);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            producer.push(999);
        });
        let (v, code) = sr.get(0, Duration::from_secs(1));
        handle.join().unwrap();
        assert_eq!(code, StreamedGetReturnCode::Success);
        assert_eq!(v, Some(999));
    }
}
