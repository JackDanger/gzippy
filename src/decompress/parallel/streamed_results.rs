//! Literal port of `rapidgzip::StreamedResults`
//! (vendor/rapidgzip/librapidarchive/src/core/StreamedResults.hpp:27-157).
//!
//! From the vendor header doc (StreamedResults.hpp:22-26):
//! > Stores results in the order they are pushed and also stores a flag
//! > signaling that nothing will be pushed anymore.
//! > The blockfinder will push block offsets and other actors, e.g., the
//! > prefetcher, may wait for and read the offsets.
//! > Results will never be deleted, so you can assume the size to only grow.
//!
//! This is the bridge between the producer (block finder running on a
//! background worker) and the consumer (prefetcher / chunk fetcher polling
//! for the next ready offset). Used directly by `ParallelGzipReader` and by
//! `BlockFinder` (the disk-streaming variant) — both ports come later.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `template<typename Value> class StreamedResults`
//!   (StreamedResults.hpp:27-28) -> generic over `Value: Clone`.
//! - `using Values = std::deque<Value>` (StreamedResults.hpp:35) ->
//!   [`std::collections::VecDeque<Value>`]. The vendor picked deque to
//!   avoid the O(N) reallocate cost on growth; VecDeque has the same
//!   amortized-O(1) push_back property.
//! - `std::mutex` + `std::condition_variable` ->
//!   [`std::sync::Mutex`] + [`std::sync::Condvar`]. Vendor wraps every
//!   read/write in `std::scoped_lock` (StreamedResults.hpp:67, 79, 101,
//!   114, 144); we do the same via [`Mutex::lock`].
//! - `std::atomic<bool> m_finalized` (StreamedResults.hpp:156) ->
//!   [`std::sync::atomic::AtomicBool`]. The vendor reads `m_finalized`
//!   without the mutex in [`finalized()`](StreamedResults.hpp:128-132) so
//!   it must be atomic; we mirror that.
//! - `GetReturnCode { SUCCESS, TIMEOUT, FAILURE }`
//!   (BlockFinderInterface.hpp:13-18) -> [`GetReturnCode`] enum here.
//!   Surface the type alongside `StreamedResults` because every caller
//!   pulls it from the same include path; when the full
//!   `BlockFinderInterface` lands it can re-export this.
//! - `ResultsView` (StreamedResults.hpp:39-61) -> [`ResultsView<'a, V>`],
//!   a guard that holds the mutex for the lifetime of the borrow,
//!   exposing the underlying `&VecDeque<Value>`.
//!
//! Notes on `setResults`
//! --------------------
//! `setResults(Values)` (StreamedResults.hpp:142-149) is a bulk-overwrite
//! that drops the queue and replaces it in one shot, then marks finalized.
//! Used by the index-replay path (`ParallelGzipReader::setIndex`) to
//! short-circuit block discovery. We expose it under the same name.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Outcome of [`StreamedResults::get`]. Mirror of
/// `BlockFinderInterface::GetReturnCode` (BlockFinderInterface.hpp:13-18).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GetReturnCode {
    /// Position was readable; result is `Some`. Mirror of `SUCCESS`.
    Success,
    /// Wait timed out before the position became readable AND before the
    /// stream was finalized. Mirror of `TIMEOUT`.
    Timeout,
    /// Position is past the finalized end of the stream. Mirror of
    /// `FAILURE`.
    Failure,
}

/// Sentinel value for "wait forever". Mirror of
/// `std::numeric_limits<double>::infinity()` (StreamedResults.hpp:77,
/// BlockFinderInterface.hpp:37). Pass [`Timeout::Infinite`] to block until
/// the position resolves.
#[derive(Debug, Clone, Copy)]
pub enum Timeout {
    /// `timeoutInSeconds = 0` (StreamedResults.hpp:81): do not wait.
    Zero,
    /// `timeoutInSeconds > 0 && finite` (StreamedResults.hpp:84-86).
    Finite(Duration),
    /// `timeoutInSeconds = infinity` (StreamedResults.hpp:87-89).
    Infinite,
}

/// Producer-finalized stream of results. Mirror of
/// `StreamedResults<Value>` (StreamedResults.hpp:27-157).
pub struct StreamedResults<Value: Clone> {
    /// `Values m_results` (StreamedResults.hpp:155). Protected by `mutex`.
    results: Mutex<VecDeque<Value>>,
    /// `std::condition_variable m_changed` (StreamedResults.hpp:153).
    changed: Condvar,
    /// `std::atomic<bool> m_finalized` (StreamedResults.hpp:156). Read
    /// lock-free in [`finalized()`](StreamedResults::finalized); the
    /// mutex-guarded copy lives implicitly in the lock-ordered code path
    /// of `push` / `finalize` / `set_results`.
    finalized: AtomicBool,
}

impl<Value: Clone> Default for StreamedResults<Value> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Value: Clone> StreamedResults<Value> {
    pub fn new() -> Self {
        Self {
            results: Mutex::new(VecDeque::new()),
            changed: Condvar::new(),
            finalized: AtomicBool::new(false),
        }
    }

    /// `size_t size() const` (StreamedResults.hpp:64-69).
    pub fn size(&self) -> usize {
        self.results
            .lock()
            .expect("StreamedResults mutex poisoned")
            .len()
    }

    /// `bool finalized() const` (StreamedResults.hpp:128-132). Lock-free.
    pub fn is_finalized(&self) -> bool {
        self.finalized.load(Ordering::Acquire)
    }

    /// `void push(Value)` (StreamedResults.hpp:98-109).
    ///
    /// # Panics
    ///
    /// Panics if the stream is already finalized — mirror of the
    /// `std::invalid_argument` throw at StreamedResults.hpp:103-105.
    pub fn push(&self, value: Value) {
        let mut guard = self.results.lock().expect("StreamedResults mutex poisoned");
        if self.finalized.load(Ordering::Acquire) {
            panic!(
                "StreamedResults::push called after finalize \
                 (vendor throws std::invalid_argument; StreamedResults.hpp:103-105)"
            );
        }
        guard.push_back(value);
        // Mirror of `m_changed.notify_all()` (StreamedResults.hpp:108).
        // We notify_all rather than notify_one because multiple waiters
        // may be parked on different positions, and any of them might
        // become ready on this push (though in practice the next-pushed
        // position serves only one waiter).
        self.changed.notify_all();
    }

    /// `void finalize(std::optional<size_t> resultsCount = {})`
    /// (StreamedResults.hpp:111-126).
    ///
    /// If `truncate_to` is `Some(n)` and `n <= current_size`, the queue is
    /// truncated to `n` elements before being marked finalized — used by
    /// the index-replay path to clip an over-counted scan. Mirror of the
    /// `resize(*resultsCount)` call at StreamedResults.hpp:121.
    ///
    /// # Panics
    ///
    /// Panics if `truncate_to.unwrap() > current_size` — mirror of the
    /// `std::invalid_argument` throw at StreamedResults.hpp:117-119.
    pub fn finalize(&self, truncate_to: Option<usize>) {
        let mut guard = self.results.lock().expect("StreamedResults mutex poisoned");
        if let Some(n) = truncate_to {
            if n > guard.len() {
                panic!(
                    "StreamedResults::finalize truncate_to={n} exceeds current size {}; \
                     vendor throws std::invalid_argument (StreamedResults.hpp:117-119)",
                    guard.len()
                );
            }
            guard.truncate(n);
        }
        self.finalized.store(true, Ordering::Release);
        // `m_changed.notify_all()` (StreamedResults.hpp:125).
        self.changed.notify_all();
    }

    /// `void setResults(Values)` (StreamedResults.hpp:142-149). Bulk
    /// overwrite + finalize in one operation. Used by `setIndex` on the
    /// reader, which knows the full offset set ahead of time.
    pub fn set_results(&self, results: VecDeque<Value>) {
        let mut guard = self.results.lock().expect("StreamedResults mutex poisoned");
        *guard = results;
        self.finalized.store(true, Ordering::Release);
        self.changed.notify_all();
    }

    /// `std::pair<std::optional<size_t>, GetReturnCode> get(size_t position, double timeoutInSeconds)`
    /// (StreamedResults.hpp:75-96).
    ///
    /// Returns the result at `position` (waiting up to `timeout`), or:
    /// - `(None, Timeout)` if `timeout` elapsed before resolution AND the
    ///   stream is not finalized.
    /// - `(None, Failure)` if the stream is finalized and `position` is
    ///   past the end.
    pub fn get(&self, position: usize, timeout: Timeout) -> (Option<Value>, GetReturnCode) {
        let guard = self.results.lock().expect("StreamedResults mutex poisoned");

        // `if ( timeoutInSeconds > 0 )` (StreamedResults.hpp:81). Zero-
        // timeout skips the wait entirely.
        let guard: MutexGuard<'_, VecDeque<Value>> = match timeout {
            Timeout::Zero => guard,
            Timeout::Finite(dur) => {
                self.changed
                    .wait_timeout_while(guard, dur, |q| {
                        !self.finalized.load(Ordering::Acquire) && position >= q.len()
                    })
                    .expect("StreamedResults mutex poisoned during condvar wait_timeout")
                    .0
            }
            Timeout::Infinite => self
                .changed
                .wait_while(guard, |q| {
                    !self.finalized.load(Ordering::Acquire) && position >= q.len()
                })
                .expect("StreamedResults mutex poisoned during condvar wait"),
        };

        // `if ( position < m_results.size() ) return { m_results[position], SUCCESS };`
        // (StreamedResults.hpp:92-94).
        if let Some(v) = guard.get(position).cloned() {
            return (Some(v), GetReturnCode::Success);
        }
        // `return { std::nullopt, m_finalized ? FAILURE : TIMEOUT };`
        // (StreamedResults.hpp:95).
        let code = if self.finalized.load(Ordering::Acquire) {
            GetReturnCode::Failure
        } else {
            GetReturnCode::Timeout
        };
        (None, code)
    }

    /// `ResultsView results() const` (StreamedResults.hpp:135-139).
    /// Hands back a guard that locks the mutex for the lifetime of the
    /// borrow. Vendor: "a view to the results, which also locks access to
    /// it using RAII".
    pub fn results(&self) -> ResultsView<'_, Value> {
        let guard = self.results.lock().expect("StreamedResults mutex poisoned");
        ResultsView { guard }
    }
}

/// RAII view that holds the StreamedResults mutex for the lifetime of the
/// borrow. Mirror of `class ResultsView` (StreamedResults.hpp:39-61).
pub struct ResultsView<'a, Value: Clone> {
    guard: MutexGuard<'a, VecDeque<Value>>,
}

impl<'a, Value: Clone> ResultsView<'a, Value> {
    /// `const Values& results() const` (StreamedResults.hpp:52-56).
    pub fn results(&self) -> &VecDeque<Value> {
        &self.guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn empty_stream_has_zero_size_not_finalized() {
        let r: StreamedResults<usize> = StreamedResults::new();
        assert_eq!(r.size(), 0);
        assert!(!r.is_finalized());
    }

    #[test]
    fn push_grows_size_and_get_returns_success() {
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(10);
        r.push(20);
        r.push(30);
        assert_eq!(r.size(), 3);

        let (v, code) = r.get(0, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(10));

        let (v, code) = r.get(2, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(30));
    }

    #[test]
    fn get_past_end_unfinalized_returns_timeout_with_zero_wait() {
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(1);
        let (v, code) = r.get(5, Timeout::Zero);
        assert_eq!(v, None);
        assert_eq!(code, GetReturnCode::Timeout);
    }

    #[test]
    fn finalize_then_get_past_end_returns_failure() {
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(1);
        r.push(2);
        r.finalize(None);
        assert!(r.is_finalized());

        let (v, code) = r.get(5, Timeout::Zero);
        assert_eq!(v, None);
        assert_eq!(code, GetReturnCode::Failure);
    }

    #[test]
    fn finalize_with_truncate_clips_queue() {
        // Mirror of StreamedResults.hpp:120-122: resize(*resultsCount).
        let r: StreamedResults<u32> = StreamedResults::new();
        for i in 0..10 {
            r.push(i);
        }
        r.finalize(Some(4));
        assert_eq!(r.size(), 4);
        let (v, code) = r.get(3, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(3));
        let (v, code) = r.get(4, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Failure);
        assert_eq!(v, None);
    }

    #[test]
    #[should_panic(expected = "exceeds current size")]
    fn finalize_truncate_too_large_panics() {
        // Mirror of std::invalid_argument throw at
        // StreamedResults.hpp:117-119.
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(1);
        r.finalize(Some(5));
    }

    #[test]
    #[should_panic(expected = "called after finalize")]
    fn push_after_finalize_panics() {
        // Mirror of std::invalid_argument throw at
        // StreamedResults.hpp:103-105.
        let r: StreamedResults<u32> = StreamedResults::new();
        r.finalize(None);
        r.push(7);
    }

    #[test]
    fn waiter_unblocks_on_push() {
        // Producer pushes after a short delay; waiter on the future
        // position unblocks via the condvar.
        let r: Arc<StreamedResults<u32>> = Arc::new(StreamedResults::new());
        let r_prod = Arc::clone(&r);
        let producer = thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            r_prod.push(42);
        });
        let (v, code) = r.get(0, Timeout::Infinite);
        producer.join().unwrap();
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(42));
    }

    #[test]
    fn waiter_unblocks_on_finalize_with_failure_code() {
        let r: Arc<StreamedResults<u32>> = Arc::new(StreamedResults::new());
        let r_prod = Arc::clone(&r);
        let producer = thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            r_prod.finalize(None);
        });
        let (v, code) = r.get(0, Timeout::Infinite);
        producer.join().unwrap();
        assert_eq!(v, None);
        assert_eq!(code, GetReturnCode::Failure);
    }

    #[test]
    fn finite_timeout_expires_with_timeout_code() {
        let r: StreamedResults<u32> = StreamedResults::new();
        let (v, code) = r.get(0, Timeout::Finite(Duration::from_millis(15)));
        assert_eq!(v, None);
        assert_eq!(code, GetReturnCode::Timeout);
    }

    #[test]
    fn set_results_bulk_overwrite_finalizes() {
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(99);
        let bulk: VecDeque<u32> = (10..15).collect();
        r.set_results(bulk);
        assert!(r.is_finalized());
        assert_eq!(r.size(), 5);
        let (v, code) = r.get(0, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(10));
        let (v, code) = r.get(4, Timeout::Zero);
        assert_eq!(code, GetReturnCode::Success);
        assert_eq!(v, Some(14));
    }

    #[test]
    fn results_view_locks_for_lifetime_of_borrow() {
        let r: StreamedResults<u32> = StreamedResults::new();
        r.push(1);
        r.push(2);
        {
            let view = r.results();
            assert_eq!(view.results().len(), 2);
            assert_eq!(view.results().front(), Some(&1));
        }
        // After the view drops, the mutex is released and subsequent
        // operations succeed without deadlock.
        r.push(3);
        assert_eq!(r.size(), 3);
    }
}
