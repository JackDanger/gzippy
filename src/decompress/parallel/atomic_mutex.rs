//! Literal port of `rapidgzip::AtomicMutex`
//! (vendor/rapidgzip/librapidarchive/src/core/AtomicMutex.hpp:9-32).
//!
//! Spinlock-style mutex backed by a single atomic boolean. The vendor uses
//! this in hot paths where the contention is short enough that the cost of a
//! `std::condition_variable` notify + futex syscall dominates the actual
//! critical-section work. Examples in rapidgzip include the per-block stats
//! aggregators in `BlockFetcher::processReadyBlocks` and the small accounting
//! locks inside `ChunkData`.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `std::atomic<bool>`            -> [`std::sync::atomic::AtomicBool`].
//! - `std::memory_order_relaxed`    -> [`Ordering::Relaxed`].
//! - `std::memory_order_acquire`    -> [`Ordering::Acquire`].
//! - `std::memory_order_release`    -> [`Ordering::Release`].
//! - `std::this_thread::sleep_for(10ns)` (AtomicMutex.hpp:20)
//!   -> [`std::thread::sleep`] with [`std::time::Duration::from_nanos(10)`].
//!   On both Linux and macOS the actual minimum sleep granularity is
//!   much larger than 10ns; the call mostly devolves into a CPU yield
//!   plus a syscall. We match the vendor semantics exactly.
//!
//! Why not `parking_lot::Mutex`?
//! ---------------------------
//! `parking_lot::Mutex` is *adaptive*: it spins briefly then parks via futex.
//! The vendor's design is a pure spinlock with a hard 10ns nap between
//! attempts — no parking, no fairness, no poisoning. The two are not
//! interchangeable behaviorally. Since the policy is "PORT, don't innovate"
//! we mirror the AtomicBool implementation directly.
//!
//! Why not `std::sync::Mutex`?
//! -------------------------
//! `std::sync::Mutex` parks the thread on futex on Linux and `os_unfair_lock`
//! on macOS, with poisoning semantics that interfere with cross-language
//! intent. Same answer: faithful port over idiomatic substitute.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Spinlock-style mutex.
///
/// Mirror of `class AtomicMutex` (AtomicMutex.hpp:9-32). Does NOT implement
/// poisoning, RAII guards, or any of `std::sync::Mutex<T>`'s richer API; it
/// is the bare-minimum primitive needed to drop in where the vendor uses
/// `AtomicMutex` directly with `std::scoped_lock<AtomicMutex>`.
pub struct AtomicMutex {
    /// `std::atomic<bool> m_flag{ false }` (AtomicMutex.hpp:31).
    flag: AtomicBool,
}

impl AtomicMutex {
    /// `m_flag{ false }` initializer (AtomicMutex.hpp:31).
    pub const fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
        }
    }

    /// `void lock()` (AtomicMutex.hpp:12-22).
    ///
    /// ```c++
    /// while ( m_flag.load( std::memory_order_relaxed ) ||
    ///         m_flag.exchange( true, std::memory_order_acquire ) ) {
    ///     std::this_thread::sleep_for( 10ns );
    /// }
    /// ```
    ///
    /// The double-check pattern (relaxed load first, then exchange) avoids
    /// the cache-line lock that `exchange` takes when the flag is already
    /// observed true. Exchange acquires on success so that any writes the
    /// previous holder made under the same lock are visible to us.
    pub fn lock(&self) {
        while self.flag.load(Ordering::Relaxed) || self.flag.swap(true, Ordering::Acquire) {
            // `using namespace std::chrono_literals; std::this_thread::sleep_for( 10ns );`
            // (AtomicMutex.hpp:19-20). The 10ns is a documented "yield-ish"
            // hint — Linux nanosleep rounds up to the scheduler tick, but the
            // syscall still releases the timeslice. We preserve the exact
            // value to keep the behavior identical.
            std::thread::sleep(Duration::from_nanos(10));
        }
    }

    /// `void unlock()` (AtomicMutex.hpp:24-28).
    ///
    /// ```c++
    /// m_flag.store( false, std::memory_order_release );
    /// ```
    pub fn unlock(&self) {
        self.flag.store(false, Ordering::Release);
    }

    /// Convenience RAII wrapper. Mirror of `std::scoped_lock<AtomicMutex>`
    /// — there's no vendor `lock_guard` member, but every caller uses
    /// `std::scoped_lock` to acquire and release; the Rust idiom is a
    /// guard returned from a method.
    pub fn scoped(&self) -> AtomicMutexGuard<'_> {
        self.lock();
        AtomicMutexGuard { mutex: self }
    }
}

impl Default for AtomicMutex {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that unlocks on drop. The vendor side uses
/// `std::scoped_lock<AtomicMutex>` for the same purpose.
pub struct AtomicMutexGuard<'a> {
    mutex: &'a AtomicMutex,
}

impl<'a> Drop for AtomicMutexGuard<'a> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

// AtomicMutex is Send + Sync by virtue of containing only AtomicBool, which
// is itself Send + Sync. The vendor type has no equivalent annotation
// because C++ leaves that to the user; ours follows from Rust's auto-trait
// inference.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AOrdering};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn lock_unlock_serially() {
        let m = AtomicMutex::new();
        m.lock();
        m.unlock();
        m.lock();
        m.unlock();
    }

    #[test]
    fn guard_unlocks_on_drop() {
        let m = AtomicMutex::new();
        {
            let _g = m.scoped();
        }
        // If the guard failed to unlock, this second acquire would spin
        // forever; the test would hang.
        m.lock();
        m.unlock();
    }

    #[test]
    fn mutual_exclusion_under_contention() {
        // Mirror of how rapidgzip uses AtomicMutex: many short critical
        // sections, no long blocking sleeps. The counter increment is
        // intentionally non-atomic (we mutate the AtomicU32 via a load +
        // store pair under the lock) so we can detect a missed exclusion.
        let m = Arc::new(AtomicMutex::new());
        let counter = Arc::new(AtomicU32::new(0));
        let n_threads = 8;
        let iters = 200;
        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let m = Arc::clone(&m);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..iters {
                    m.lock();
                    let v = c.load(AOrdering::Relaxed);
                    c.store(v + 1, AOrdering::Relaxed);
                    m.unlock();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(AOrdering::Acquire), n_threads * iters);
    }

    #[test]
    fn scoped_guard_under_contention() {
        let m = Arc::new(AtomicMutex::new());
        let counter = Arc::new(AtomicU32::new(0));
        let n = 4;
        let iters = 100;
        let mut handles = Vec::new();
        for _ in 0..n {
            let m = Arc::clone(&m);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..iters {
                    let _g = m.scoped();
                    let v = c.load(AOrdering::Relaxed);
                    c.store(v + 1, AOrdering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(AOrdering::Acquire), n * iters);
    }
}
