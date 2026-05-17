//! Literal port of `rapidgzip::JoiningThread`
//! (vendor/rapidgzip/librapidarchive/src/core/JoiningThread.hpp:13-60).
//!
//! From the vendor comment (JoiningThread.hpp:9-12):
//! > Similar to the planned C++20 std::jthread, this class joins in the
//! > destructor. Additionally, it ensures that all threads created with
//! > this interface correctly initialize rpmalloc!
//!
//! In rapidgzip this exists because `std::thread`'s destructor calls
//! `std::terminate` if the thread is still joinable; `JoiningThread` wraps
//! that and joins on destruction instead. Rust's `std::thread::JoinHandle`
//! does *not* automatically join on drop either — dropping the handle simply
//! detaches the thread. So a faithful port still has value: it gives us a
//! type whose `Drop` calls `join()`, matching the vendor RAII contract used
//! across `ThreadPool::m_threads` (ThreadPool.hpp:247) and other call sites.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `std::thread`            -> [`std::thread::JoinHandle<()>`].
//! - move ctor                -> Rust moves are the default.
//! - copy ctor `= delete`     -> `JoinHandle` is already non-`Copy`/non-`Clone`.
//! - destructor `~JoiningThread`
//!   (JoiningThread.hpp:32-37) -> [`Drop`] impl joins if joinable.
//! - `joinable()`             -> tracked locally via `Option<JoinHandle>`.
//! - `get_id()`               -> [`std::thread::JoinHandle::thread().id()`].
//! - `join()`                 -> moves the handle out and calls `.join()`.
//!
//! Note on `rpmalloc`: rapidgzip's comment mentions rpmalloc thread-init.
//! gzippy does not use rpmalloc (see Cargo.toml; the system allocator is in
//! play), so this port has nothing analogous to bootstrap on entry.
//! `ThreadPool` (thread_pool.rs) likewise has no per-worker allocator init.

#![allow(dead_code)]

use std::thread::{self, JoinHandle, ThreadId};

/// RAII wrapper around `std::thread::JoinHandle` that joins on drop.
///
/// Mirror of `class JoiningThread` (JoiningThread.hpp:13-60).
pub struct JoiningThread {
    /// `std::thread m_thread` (JoiningThread.hpp:58). Wrapped in `Option`
    /// so `join()` and `Drop` can take ownership of the handle without
    /// requiring `&mut` reentrancy tricks. `None` = already joined.
    handle: Option<JoinHandle<()>>,
}

impl JoiningThread {
    /// `template<class Function, class ... Args>
    ///  explicit JoiningThread(Function&& function, Args&&... args)
    ///      : m_thread(std::forward<Function>(function), std::forward<Args>(args)...)`
    /// (JoiningThread.hpp:16-20).
    ///
    /// The variadic constructor in C++ forwards `(function, args...)` to
    /// the `std::thread` constructor. Rust closures already capture their
    /// arguments by move/borrow, so the simpler signature here mirrors
    /// what gets used in practice: `JoiningThread::new(|| { ... })`.
    pub fn new<F>(function: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            handle: Some(thread::spawn(function)),
        }
    }

    /// `std::thread::id get_id() const noexcept` (JoiningThread.hpp:39-43).
    ///
    /// Returns `None` if the thread has already been joined (the C++
    /// version returns a default-constructed `std::thread::id` in that
    /// situation; Rust's `ThreadId` has no default, so we surface `None`
    /// instead).
    pub fn get_id(&self) -> Option<ThreadId> {
        self.handle.as_ref().map(|h| h.thread().id())
    }

    /// `bool joinable() const` (JoiningThread.hpp:45-49).
    ///
    /// True iff the wrapped thread has not yet been joined.
    pub fn joinable(&self) -> bool {
        self.handle.is_some()
    }

    /// `void join()` (JoiningThread.hpp:51-55).
    ///
    /// Blocks until the thread completes. Calling `join()` on an
    /// already-joined `JoiningThread` panics, mirroring the C++ behavior
    /// where calling `std::thread::join()` on a joined thread throws
    /// `std::system_error` with `errc::invalid_argument`.
    ///
    /// The thread closure's panic (if any) is propagated by re-panicking
    /// here. The vendor C++ analogue is `std::thread::join()` returning
    /// normally for any function that exits via exception — the exception
    /// would have already terminated the program because std::thread
    /// requires the function to return normally — so neither runtime
    /// strictly matches, but propagating the panic is the lossless choice.
    pub fn join(&mut self) -> thread::Result<()> {
        let handle = self
            .handle
            .take()
            .expect("JoiningThread::join called on non-joinable thread");
        handle.join()
    }
}

impl Drop for JoiningThread {
    /// `~JoiningThread()` (JoiningThread.hpp:32-37).
    ///
    /// ```c++
    /// if ( m_thread.joinable() ) {
    ///     m_thread.join();
    /// }
    /// ```
    ///
    /// We swallow any worker panic here — surfacing it would require
    /// either aborting (the C++ destructor exits normally if join
    /// succeeds, period) or storing the panic for later retrieval (no
    /// vendor analogue). Callers that need the result should invoke
    /// `join()` explicitly before the drop point.
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// `JoiningThread(JoiningThread&&) = default;` (JoiningThread.hpp:22) —
// Rust moves are the default for any non-`Copy` type, so no impl needed.
//
// `JoiningThread(const JoiningThread&) = delete;` (JoiningThread.hpp:24) and
// `JoiningThread& operator=(...) = delete;` (JoiningThread.hpp:26-30) — Rust
// does not auto-derive `Clone` or `Copy`, and `Option<JoinHandle>` is
// neither, so both are statically prevented.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn join_returns_after_thread_completes() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let mut t = JoiningThread::new(move || {
            c.fetch_add(1, Ordering::AcqRel);
        });
        assert!(t.joinable());
        t.join().unwrap();
        assert!(!t.joinable());
        assert_eq!(counter.load(Ordering::Acquire), 1);
    }

    #[test]
    fn drop_joins_thread() {
        // Mirror of the destructor contract (JoiningThread.hpp:32-37):
        // dropping a still-joinable JoiningThread must block until the
        // thread completes, NOT detach or terminate.
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        {
            let _t = JoiningThread::new(move || {
                // Sleep briefly to make detection of "drop did not wait"
                // observable: if Drop detaches instead of joining, the
                // assertion below would race.
                std::thread::sleep(Duration::from_millis(20));
                c.fetch_add(1, Ordering::AcqRel);
            });
            // _t goes out of scope here -> Drop -> join()
        }
        assert_eq!(counter.load(Ordering::Acquire), 1);
    }

    #[test]
    fn get_id_reports_thread_id_until_joined() {
        let mut t = JoiningThread::new(|| {});
        assert!(t.get_id().is_some());
        t.join().unwrap();
        assert!(t.get_id().is_none());
    }

    #[test]
    fn joinable_flips_after_explicit_join() {
        let mut t = JoiningThread::new(|| {});
        assert!(t.joinable());
        t.join().unwrap();
        assert!(!t.joinable());
    }

    #[test]
    fn worker_panic_propagates_through_explicit_join() {
        // std::thread::join in Rust returns Err(payload) when the closure
        // panicked. We surface that to the caller so a deliberate test can
        // observe it; Drop swallows it instead (no vendor analogue for
        // storing the panic).
        let mut t = JoiningThread::new(|| panic!("worker boom"));
        let result = t.join();
        assert!(result.is_err());
    }
}
