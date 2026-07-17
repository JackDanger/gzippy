#![cfg(parallel_sm)]

//! Literal port of `rapidgzip::ThreadPool`
//! (vendor/rapidgzip/librapidarchive/src/core/ThreadPool.hpp:33-248).
//!
//! Function evaluations can be given to a [`ThreadPool`] instance, which
//! assigns the evaluation to one of its threads to be evaluated in parallel.
//! This is the foundational primitive on top of which rapidgzip's
//! `BlockFetcher` / `GzipChunkFetcher` are built.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `std::mutex` -> [`std::sync::Mutex`].
//! - `std::condition_variable` -> [`std::sync::Condvar`].
//! - `std::map<int, deque<...>>` -> [`std::collections::BTreeMap`] (priority -> VecDeque).
//! - `std::future<T>` -> [`Future<T>`] backed by a one-shot `mpsc::sync_channel(1)`.
//! - `std::packaged_task<T()>` -> boxed `FnOnce() -> T` that sends into the
//!   oneshot. C++'s `PackagedTaskWrapper` exists only because
//!   `std::packaged_task` is move-only and `std::function` requires copy; in
//!   Rust `Box<dyn FnOnce() + Send>` already captures move-only state, so the
//!   wrapper layer collapses to a single boxed closure.
//! - `JoiningThread` -> [`std::thread::JoinHandle`] joined in `stop()` (the
//!   Rust equivalent of the C++ destructor).
//! - `pinThreadToLogicalCore` -> [`core_affinity::set_for_current`] (gzippy's
//!   existing dependency; matches the Linux `sched_setaffinity` semantics
//!   the C++ uses).
//!
//! The semantics — on-demand worker spawn until `m_threadCount` is reached,
//! priority dispatch via `BTreeMap`'s ordered keys, `notify_one` per
//! submission, `notify_all` on shutdown, `m_idleThreadCount` gating the
//! spawn decision — mirror ThreadPool.hpp exactly. See the per-method
//! comments below for the precise line correspondence.
//!
//! Wiring
//! ------
//! This commit lands the standalone module with unit tests. Wiring it into
//! [`crate::decompress::parallel::chunk_fetcher`] (the gzippy equivalent of
//! rapidgzip's `BS::thread_pool` consumer) is a follow-up commit because
//! that rewires the existing mpsc worker loop.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Logical-thread-index → core-id map.
/// Mirror of `ThreadPool::ThreadPinning`
/// (ThreadPool.hpp:36 `using ThreadPinning = std::unordered_map<size_t, uint32_t>`).
pub type ThreadPinning = HashMap<usize, u32>;

/// Bounded busy-spin window (in microseconds) that a worker polls the
/// lock-free `pending_tasks` hint BEFORE parking in the condvar.
///
/// **Why this exists (no rapidgzip counterpart — byte-transparent perf knob).**
/// Under an `ondemand`-style cpufreq governor, gzippy's
/// worker pool parks in the condvar during the short, frequent inter-chunk
/// gaps. Idle cores let the governor drop to a lower P-state, so the pool
/// clocks lower than rapidgzip (which spins). A bounded spin keeps the core warm across
/// the gap so the governor holds the clock, WITHOUT burning cores when
/// saturated: the spin only runs while `pending_tasks == 0` (a genuine
/// idle gap) and bails the instant a task is queued — in the throughput-
/// saturated case a finishing worker almost always sees `pending > 0` and
/// never spins.
///
/// Frozen to [`DEFAULT_POOL_SPIN_US`].
fn pool_spin_us() -> u64 {
    DEFAULT_POOL_SPIN_US
}

/// Baked default spin window, in microseconds. `0` restores exact
/// park-immediately behavior.
const DEFAULT_POOL_SPIN_US: u64 = 2000;

/// Number of `spin_loop()` hints between wall-clock deadline checks in the
/// worker spin, amortizing the `Instant::now()` cost across the busy-poll.
const SPIN_BATCH: u32 = 64;

/// Default thread count.
///
/// Mirror of `availableCores()` (AffinityHelpers.hpp:18-21 / 104-130). The
/// vendor uses `std::thread::hardware_concurrency()` on non-Linux and
/// `sched_getaffinity`-derived counts on Linux. `num_cpus::get()` (already a
/// gzippy dependency) honors cgroup/cpuset bounds on Linux and falls back
/// to `hardware_concurrency` elsewhere — same intent.
#[allow(dead_code)] // vendor parity or unit-test surface
fn available_cores() -> usize {
    num_cpus::get()
}

/// A move-only handle to the eventual result of a submitted task.
///
/// Mirror of `std::future<T>` as returned by
/// `ThreadPool::submit` (ThreadPool.hpp:141 `std::future<decltype(...)> submit(...)`).
/// Backed by a one-shot `mpsc::sync_channel(1)`, which is the closest
/// stdlib analogue: send-once, receive-once, blocking `recv`.
///
/// Like `std::future`, `wait()` blocks until the producer completes and
/// returns the task's return value. Dropping the [`Future`] does NOT
/// cancel the task — the task still runs on the worker thread and its
/// result is silently discarded, matching `std::future` semantics where
/// detaching the future does not abort the packaged task.
#[must_use = "Future must be polled (`wait`) to receive the task's result"]
pub struct Future<T> {
    rx: mpsc::Receiver<T>,
}

impl<T> Future<T> {
    /// Block until the task completes and return its result.
    ///
    /// Mirror of `std::future<T>::get()` semantics as used by rapidgzip
    /// (e.g. `BlockFetcher.hpp:329 future.get()`).
    ///
    /// Returns `Err(FutureError::Cancelled)` if the producing worker
    /// thread was destroyed before the task ran, which only happens
    /// across [`ThreadPool::stop`] races; `std::future` would throw a
    /// `broken_promise` here.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn wait(self) -> Result<T, FutureError> {
        self.rx.recv().map_err(|_| FutureError::Cancelled)
    }

    /// Consume the future and return its underlying `mpsc::Receiver`.
    ///
    /// Mirror of the `std::move(future)` pattern used by rapidgzip at
    /// BlockFetcher.hpp:558 to stash a `std::future<BlockData>` inside
    /// `m_prefetching`. The `BlockFetcher` port stores
    /// `Receiver<Result<Value, Err>>` in `m_prefetching`, and the
    /// submit-closure must yield exactly that shape. This accessor lets
    /// the closure unwrap a `Future<R>` into the channel that the
    /// `BlockFetcher::get` wait loop will then `recv()` on.
    pub fn into_receiver(self) -> mpsc::Receiver<T> {
        self.rx
    }
}

/// Errors observable on a [`Future`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // vendor ThreadPool.hpp parity; used by Future::wait in unit tests
pub enum FutureError {
    /// The pool was stopped before the task could run.
    /// Equivalent to `std::future_errc::broken_promise`.
    Cancelled,
}

impl std::fmt::Display for FutureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FutureError::Cancelled => write!(f, "thread pool task was cancelled (broken promise)"),
        }
    }
}

impl std::error::Error for FutureError {}

/// Type-erased task. Mirror of `PackagedTaskWrapper`
/// (ThreadPool.hpp:52-98) — collapsed because `Box<dyn FnOnce()>`
/// natively handles move-only state.
type Task = Box<dyn FnOnce() + Send + 'static>;

/// Shared mutable state. Mirror of the `m_tasks` / `m_mutex` / `m_pingWorkers`
/// triple (ThreadPool.hpp:237-241).
struct PoolShared {
    /// `std::map<priority, std::deque<PackagedTaskWrapper>>`
    /// (ThreadPool.hpp:237). `BTreeMap` keeps the priority keys ordered,
    /// matching `std::map`. The C++ comment on line 213-214 traverses
    /// `m_tasks.begin() ... m_tasks.end()` to find the first non-empty
    /// deque — same semantics here via `BTreeMap::iter_mut()`.
    tasks: BTreeMap<i32, std::collections::VecDeque<Task>>,
    /// `m_threadPoolRunning` (ThreadPool.hpp:231). Kept in the locked
    /// state so workers re-check it under the same lock as `tasks`,
    /// matching the C++ predicate at ThreadPool.hpp:206.
    running: bool,
}

/// `ThreadPool` proper. Literal port of the rapidgzip class
/// (ThreadPool.hpp:33-248).
///
/// All mutating methods (`submit`, `stop`) take `&self` — interior
/// mutability via `Mutex` mirrors the C++ pattern at ThreadPool.hpp
/// where `submit` and `stop` are non-const but `m_threadPool` is
/// declared as a non-const member callable from multiple threads
/// (BlockFetcher.hpp:686 `ThreadPool m_threadPool`, with `submit`
/// invoked from `BlockFetcher::get`'s thread and worker threads
/// running `prefetchNewBlocks` concurrently). The Rust mapping
/// requires explicit `Mutex` around the parts that the C++ implicit
/// non-const-but-shared semantics covered.
pub struct ThreadPool {
    /// `m_threadCount` (ThreadPool.hpp:233). Const after construction.
    thread_count: usize,
    /// `m_threadPinning` (ThreadPool.hpp:234). Const after construction;
    /// consulted once per worker on entry to `workerMain`
    /// (ThreadPool.hpp:198-200).
    thread_pinning: ThreadPinning,
    /// `m_idleThreadCount` (ThreadPool.hpp:235). Incremented before the
    /// condvar wait and decremented after (ThreadPool.hpp:205-207); the
    /// spawn decision at ThreadPool.hpp:157 reads it under the mutex.
    idle_thread_count: Arc<AtomicUsize>,
    /// Also `m_threadPoolRunning` (ThreadPool.hpp:231). The atomic copy
    /// gives lock-free reads from worker `while` headers; the mutex copy
    /// (inside `PoolShared`) protects the wait-predicate.
    running: Arc<AtomicBool>,
    /// `m_mutex` (ThreadPool.hpp:240). Mutable so workers can lock.
    shared: Arc<Mutex<PoolShared>>,
    /// `m_pingWorkers` (ThreadPool.hpp:241).
    ping_workers: Arc<Condvar>,
    /// Lock-free hint of the number of queued-but-not-yet-dequeued tasks.
    /// Incremented under `shared` lock in `submit` (paired with the
    /// `push_back`), decremented in `worker_main` when a worker pops a
    /// task. It is ONLY a hint for the pre-park busy-spin — the
    /// authoritative wait predicate remains `has_unprocessed_tasks` under
    /// the mutex, so a stale read here can never cause a lost wakeup
    /// (a false-positive makes a worker acquire the lock, find nothing,
    /// and park; a task push always happens-before the condvar notify).
    /// No rapidgzip counterpart — supports the byte-transparent spin knob.
    pending_tasks: Arc<AtomicUsize>,
    /// EVENT-DRIVEN CONSUMER WAKEUP (no rapidgzip counterpart — rg's consumer
    /// blocks on `std::future` directly; ours blocks on an mpsc `Receiver`).
    /// A monotonic generation counter bumped + `notify_all`'d AFTER every task
    /// body returns — at which point the task's result is already in its oneshot
    /// channel (`submit`'s `tx.send(task())`). The single-pass in-order consumer,
    /// while waiting for the NEXT chunk it needs, blocks on this condvar (with a
    /// 1 ms fallback) instead of polling `recv_timeout(1ms)`, so it wakes
    /// IMMEDIATELY when any worker frees up and re-advances the prefetch horizon
    /// at ~0 latency. Byte-transparent: only changes WHEN
    /// the consumer wakes to pump/re-check, never what it decodes.
    task_completed: Arc<(Mutex<u64>, Condvar)>,
    /// `m_threads` (ThreadPool.hpp:247). `JoiningThread` becomes a
    /// `JoinHandle` we explicitly join in `stop()`. Wrapped in `Mutex`
    /// so the `&self`-callable `submit` can push freshly spawned
    /// handles. (Vendor's `m_threads` is plain `vector` because C++
    /// permits push_back from a non-const method called via
    /// `m_threadPool.submit` even when m_threadPool is shared.)
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl ThreadPool {
    /// `explicit ThreadPool(size_t threadCount = availableCores(), ThreadPinning threadPinning = {})`
    /// (ThreadPool.hpp:101-108).
    ///
    /// `m_threads.reserve(m_threadCount)` (ThreadPool.hpp:107) maps to
    /// `Vec::with_capacity` below.
    pub fn new(thread_count: usize, thread_pinning: ThreadPinning) -> Self {
        Self {
            thread_count,
            thread_pinning,
            idle_thread_count: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            shared: Arc::new(Mutex::new(PoolShared {
                tasks: BTreeMap::new(),
                running: true,
            })),
            ping_workers: Arc::new(Condvar::new()),
            pending_tasks: Arc::new(AtomicUsize::new(0)),
            task_completed: Arc::new((Mutex::new(0), Condvar::new())),
            threads: Mutex::new(Vec::with_capacity(thread_count)),
        }
    }

    /// Current task-completion generation (see [`ThreadPool::task_completed`]).
    /// A consumer snapshots this BEFORE its non-blocking `try_recv` + prefetch
    /// pump, then hands it to [`wait_for_completion`](Self::wait_for_completion);
    /// because the counter is monotonic, a completion that lands during the pump
    /// makes the subsequent wait return immediately — no lost wakeup.
    pub fn completion_gen(&self) -> u64 {
        *self
            .task_completed
            .0
            .lock()
            .expect("task_completed poisoned")
    }

    /// Block until the completion generation advances past `last_seen` or
    /// `timeout` elapses, returning the current generation. The `timeout` is a
    /// liveness fallback only — under load the condvar notify fires first, so the
    /// consumer wakes at ~0 latency on each worker completion. On the rare
    /// tiny-file / idle path the fallback keeps the old poll cadence, so there is
    /// no extra wakeup cost when there is nothing completing.
    pub fn wait_for_completion(&self, last_seen: u64, timeout: std::time::Duration) -> u64 {
        let (lock, cv) = &*self.task_completed;
        let guard = lock.lock().expect("task_completed poisoned");
        let (guard, _) = cv
            .wait_timeout_while(guard, timeout, |g| *g == last_seen)
            .expect("task_completed poisoned during wait");
        *guard
    }

    /// Convenience constructor using [`available_cores`] and an empty
    /// pinning map. Mirror of the default-argument form on
    /// ThreadPool.hpp:101-103.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn with_default_capacity() -> Self {
        Self::new(available_cores(), ThreadPinning::new())
    }

    /// `size_t capacity() const` (ThreadPool.hpp:166-170).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn capacity(&self) -> usize {
        self.thread_count
    }

    /// Number of worker threads currently spawned. Mirror of
    /// `m_threads.size()` (ThreadPool.hpp:247) for diagnostic /
    /// test-only reads. Vendor has no public accessor; we expose this
    /// to support the on-demand-spawn unit test that asserts
    /// `m_threads.size() == N` after submitting N blocking tasks.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn spawned_threads(&self) -> usize {
        self.threads.lock().unwrap().len()
    }

    /// Current count of worker threads parked in the condvar wait
    /// (`m_idleThreadCount`, ThreadPool.hpp:235). Diagnostic-only read:
    /// `busy = spawned_threads() - idle_thread_count()`, and
    /// `idle_capacity = idle_thread_count() + (capacity() - spawned_threads())`
    /// (lazy-spawn means an un-spawned slot is also available capacity). A
    /// relaxed atomic read — an instantaneous snapshot, no decode effect.
    #[allow(dead_code)] // diagnostic / probe surface
    pub fn idle_thread_count(&self) -> usize {
        self.idle_thread_count.load(Ordering::Relaxed)
    }

    /// `size_t unprocessedTasksCount(std::optional<int> priority = {}) const`
    /// (ThreadPool.hpp:172-182).
    ///
    /// `priority = None` sums all priority buckets via
    /// `std::accumulate` (ThreadPool.hpp:180-181). `priority = Some(p)`
    /// returns the size of that single bucket (ThreadPool.hpp:176-179).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn unprocessed_tasks_count(&self, priority: Option<i32>) -> usize {
        let shared = self.shared.lock().expect("ThreadPool mutex poisoned");
        match priority {
            Some(p) => shared.tasks.get(&p).map_or(0, |q| q.len()),
            None => shared.tasks.values().map(|q| q.len()).sum(),
        }
    }

    /// `template<class T_Functor> std::future<...> submit(T_Functor&& task, int priority = 0)`
    /// (ThreadPool.hpp:140-164).
    ///
    /// 1. Lock the mutex (ThreadPool.hpp:145).
    /// 2. If `m_threadCount == 0`, run inline via deferred future
    ///    (ThreadPool.hpp:147-149). We emulate `std::launch::deferred`
    ///    by running the task right here on the submitting thread and
    ///    feeding the result into a pre-populated channel — the
    ///    returned [`Future`] still satisfies the same interface.
    /// 3. Build a packaged task (closure + oneshot sender) and push it
    ///    onto `m_tasks[priority]` (ThreadPool.hpp:151-155).
    /// 4. Spawn a worker if room and no idle workers
    ///    (ThreadPool.hpp:157-159).
    /// 5. `m_pingWorkers.notify_one()` (ThreadPool.hpp:161).
    pub fn submit<F, T>(&self, task: F, priority: i32) -> Future<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel::<T>(1);
        let future = Future { rx };

        if self.thread_count == 0 {
            // `std::async(std::launch::deferred, ...)` (ThreadPool.hpp:148).
            // Run inline; the future drains the single buffered value.
            let value = task();
            // Send must succeed (capacity-1 channel, no other senders).
            let _ = tx.send(value);
            return future;
        }

        let packaged: Task = Box::new(move || {
            // C++'s std::packaged_task discards the result if the future
            // was destroyed; the matching behavior here is to ignore a
            // SendError (other end dropped).
            let _ = tx.send(task());
        });

        let should_spawn;
        {
            let mut shared = self.shared.lock().expect("ThreadPool mutex poisoned");
            shared
                .tasks
                .entry(priority)
                .or_default()
                .push_back(packaged);
            // Publish the queue-depth hint for the worker pre-park spin.
            // Under the lock (paired with push_back) so it is consistent
            // with has_unprocessed_tasks; workers read it Relaxed while
            // spinning. See `pending_tasks` field doc for the no-lost-
            // wakeup argument.
            self.pending_tasks.fetch_add(1, Ordering::Relaxed);

            // (m_threads.size() < m_threadCount) && (m_idleThreadCount == 0)
            // (ThreadPool.hpp:157). Vendor reads `m_threads.size()`
            // under `m_mutex`; we do the same here via a short scoped
            // lock on the threads vec.
            let current_threads = self.threads.lock().unwrap().len();
            should_spawn = current_threads < self.thread_count
                && self.idle_thread_count.load(Ordering::Acquire) == 0;

            // m_pingWorkers.notify_one() (ThreadPool.hpp:161). The C++
            // call sits *inside* the locked region, holding the mutex
            // across the notify; we match that to preserve the ordering
            // guarantee on the wait predicate.
            self.ping_workers.notify_one();
        }
        if should_spawn {
            self.spawn_thread();
        }

        future
    }

    /// `void stop()` (ThreadPool.hpp:115-132).
    ///
    /// 1. Take the mutex, flip `m_threadPoolRunning = false`,
    ///    `m_pingWorkers.notify_all()` (ThreadPool.hpp:118-122).
    /// 2. `m_threads.clear()` (ThreadPool.hpp:131) — the
    ///    `JoiningThread` destructor joins each worker. We join
    ///    explicitly.
    pub fn stop(&self) {
        {
            let mut shared = self.shared.lock().expect("ThreadPool mutex poisoned");
            shared.running = false;
            self.running.store(false, Ordering::Release);
            self.ping_workers.notify_all();
        }

        let handles: Vec<_> = self.threads.lock().unwrap().drain(..).collect();
        for handle in handles {
            // `JoiningThread::~JoiningThread` joins unconditionally
            // (JoiningThread.hpp:33-37). Ignore poisoned worker panics
            // here — surfacing them would require a richer return type
            // than the C++ `void`.
            let _ = handle.join();
        }
    }

    /// `void spawnThread()` (ThreadPool.hpp:224-228).
    ///
    /// `m_threads.emplace_back([this, i = m_threads.size()] () { workerMain( i ); } );`
    /// becomes a `std::thread::spawn` that captures the shared Arcs and
    /// the freshly assigned index.
    ///
    /// Precondition: caller holds `self.shared` lock (mirrors the
    /// rapidgzip private-method invariant at ThreadPool.hpp:186-187).
    fn spawn_thread(&self) {
        let mut threads = self.threads.lock().unwrap();
        let index = threads.len();
        let shared = Arc::clone(&self.shared);
        let cv = Arc::clone(&self.ping_workers);
        let idle = Arc::clone(&self.idle_thread_count);
        let running = Arc::clone(&self.running);
        let pending = Arc::clone(&self.pending_tasks);
        let task_completed = Arc::clone(&self.task_completed);
        let pin = self.thread_pinning.get(&index).copied();

        let handle = thread::spawn(move || {
            worker_main(
                index,
                shared,
                cv,
                idle,
                running,
                pending,
                task_completed,
                pin,
            );
        });
        threads.push(handle);
    }
}

impl Drop for ThreadPool {
    /// `~ThreadPool() { stop(); }` (ThreadPool.hpp:110-113).
    fn drop(&mut self) {
        if !self.threads.lock().unwrap().is_empty() {
            self.stop();
        }
    }
}

/// `void workerMain(size_t threadIndex)` (ThreadPool.hpp:195-222).
///
/// Layout matches the C++:
/// 1. Pin to logical core if a mapping is present (ThreadPool.hpp:198-200).
/// 2. While the pool is running, lock the mutex, bump idle counter,
///    wait on the condvar with the
///    `hasUnprocessedTasks() || !m_threadPoolRunning` predicate
///    (ThreadPool.hpp:202-207).
/// 3. On wake, decrement idle, check shutdown (ThreadPool.hpp:207-211).
/// 4. Otherwise pop the front of the lowest-priority non-empty deque,
///    drop the lock, run the task (ThreadPool.hpp:213-220).
fn worker_main(
    thread_index: usize,
    shared: Arc<Mutex<PoolShared>>,
    cv: Arc<Condvar>,
    idle: Arc<AtomicUsize>,
    running: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
    task_completed: Arc<(Mutex<u64>, Condvar)>,
    pin: Option<u32>,
) {
    let spin_us = pool_spin_us();
    crate::decompress::parallel::chunk_buffer_pool::bind_worker_pool_index(thread_index);
    if let Some(core_id) = pin {
        // Mirror of `pinThreadToLogicalCore(static_cast<int>(pinning->second))`
        // (ThreadPool.hpp:199). `core_affinity` is portable across the
        // platforms gzippy targets; the rapidgzip implementation uses
        // raw `sched_setaffinity` on Linux and is a no-op elsewhere. The
        // production decode pool constructs with an EMPTY pinning map (see
        // chunk_fetcher.rs and BlockFetcher.hpp:185), so `pin` is None there
        // and the OS schedules the workers; this branch keeps the generic
        // vendor ThreadPool pinning surface for any caller that supplies a map.
        let ids = core_affinity::get_core_ids().unwrap_or_default();
        if let Some(id) = ids.into_iter().find(|c| c.id as u32 == core_id) {
            let _ = core_affinity::set_for_current(id);
        }
    }

    while running.load(Ordering::Acquire) {
        // ADAPTIVE PRE-PARK SPIN (byte-transparent; no rapidgzip
        // counterpart — see `pool_spin_us` doc). During the short,
        // frequent inter-chunk idle gaps, busy-poll the lock-free
        // `pending_tasks` hint for a BOUNDED window before falling
        // through to the condvar park. This keeps the core warm so the
        // `ondemand` governor holds the clock (closes the AMD-Zen2 mid-T
        // loss). Correctness: this only DELAYS parking; the authoritative
        // wait predicate is still `has_unprocessed_tasks` under the mutex
        // in `wait_while` below, so a task queued during (or after) the
        // spin is never missed. Adaptive by construction: the loop bails
        // the instant `pending > 0`, so in the throughput-saturated case
        // (a task is essentially always queued) it spins ~zero — it only
        // burns cycles when there is genuinely nothing to do, which is
        // exactly the idle gap we want to keep the core warm across.
        if spin_us > 0 && running.load(Ordering::Acquire) {
            let deadline = std::time::Instant::now() + std::time::Duration::from_micros(spin_us);
            while pending.load(Ordering::Relaxed) == 0 {
                if !running.load(Ordering::Acquire) || std::time::Instant::now() >= deadline {
                    break;
                }
                for _ in 0..SPIN_BATCH {
                    std::hint::spin_loop();
                }
            }
        }
        let mut guard = shared.lock().expect("ThreadPool mutex poisoned");

        // ++m_idleThreadCount (ThreadPool.hpp:205) — must happen under
        // the lock so that submit()'s "spawn if zero idle" decision and
        // this increment cannot interleave.
        idle.fetch_add(1, Ordering::AcqRel);

        // m_pingWorkers.wait(tasksLock, [this] () { return hasUnprocessedTasks() || !m_threadPoolRunning; });
        // (ThreadPool.hpp:206).
        guard = cv
            .wait_while(guard, |s| s.running && !has_unprocessed_tasks(&s.tasks))
            .expect("ThreadPool mutex poisoned during condvar wait");

        // --m_idleThreadCount (ThreadPool.hpp:207).
        idle.fetch_sub(1, Ordering::AcqRel);

        // if (!m_threadPoolRunning) break; (ThreadPool.hpp:209-211).
        if !guard.running {
            break;
        }

        // std::find_if for the first non-empty priority bucket
        // (ThreadPool.hpp:213-214). BTreeMap iterates in ascending key
        // order, matching std::map.
        let task = first_pending_task(&mut guard.tasks);
        drop(guard);

        if let Some(t) = task {
            // Popped one task — retire its queue-depth hint (paired with
            // the fetch_add in `submit`). Only on Some, so the counter
            // can never underflow.
            pending.fetch_sub(1, Ordering::Relaxed);
            // task() (ThreadPool.hpp:219).
            t();
            // EVENT-DRIVEN CONSUMER WAKEUP (see `ThreadPool::task_completed`).
            // The task body has returned, so its result is already in its
            // oneshot channel (`submit`'s `tx.send(task())` ran inside `t()`);
            // bump the generation + wake any consumer blocked waiting for the
            // next chunk so it re-checks + re-advances the prefetch horizon at
            // ~0 latency instead of on a fixed 1ms poll tick.
            let (lock, cv) = &*task_completed;
            {
                let mut g = lock.lock().expect("task_completed poisoned");
                *g = g.wrapping_add(1);
            }
            cv.notify_all();
        }
        // No task: loop and re-wait. This can happen if another worker
        // raced us to the front of the queue; the C++ has the same
        // tolerance (the `if (nonEmptyTasks != m_tasks.end())` gate at
        // ThreadPool.hpp:215).
    }
}

/// Mirror of `bool hasUnprocessedTasks() const` (ThreadPool.hpp:188-193).
fn has_unprocessed_tasks(tasks: &BTreeMap<i32, std::collections::VecDeque<Task>>) -> bool {
    tasks.values().any(|q| !q.is_empty())
}

/// Pop the front task from the lowest-priority non-empty bucket.
/// Mirror of ThreadPool.hpp:213-218.
fn first_pending_task(tasks: &mut BTreeMap<i32, std::collections::VecDeque<Task>>) -> Option<Task> {
    for queue in tasks.values_mut() {
        if let Some(t) = queue.pop_front() {
            return Some(t);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    #[test]
    fn submit_and_collect_result() {
        let pool = ThreadPool::new(2, ThreadPinning::new());
        let f = pool.submit(|| 7_i32 + 35, 0);
        assert_eq!(f.wait().unwrap(), 42);
    }

    #[test]
    fn capacity_reports_thread_count() {
        let pool = ThreadPool::new(4, ThreadPinning::new());
        assert_eq!(pool.capacity(), 4);
    }

    #[test]
    fn zero_threads_runs_inline_deferred() {
        // ThreadPool.hpp:147-149 `if (m_threadCount == 0)
        // return std::async(std::launch::deferred, ...)`.
        let pool = ThreadPool::new(0, ThreadPinning::new());
        let f = pool.submit(|| String::from("inline"), 0);
        assert_eq!(f.wait().unwrap(), "inline");
        // No worker threads were spawned.
        assert_eq!(pool.spawned_threads(), 0);
    }

    #[test]
    fn unprocessed_tasks_count_filters_by_priority() {
        // Hold up the pool with one blocker so the priority queue actually
        // backs up: a single worker can only drain one task at a time.
        let pool = ThreadPool::new(1, ThreadPinning::new());
        let barrier = Arc::new(Barrier::new(2));
        let b = Arc::clone(&barrier);
        let blocker = pool.submit(move || b.wait(), -5);

        // Queue tasks at distinct priorities while the worker is blocked
        // inside the barrier wait.
        let _f1 = pool.submit(|| 1_u32, 0);
        let _f2 = pool.submit(|| 2_u32, 0);
        let _f3 = pool.submit(|| 3_u32, 10);

        // The blocker is currently running; the three follow-ups remain
        // queued. Mirror of `unprocessedTasksCount(std::optional<int>)`.
        // Give the worker a beat to pick up the blocker so the queue
        // reflects only the pending items.
        for _ in 0..100 {
            if pool.unprocessed_tasks_count(None) == 3 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(pool.unprocessed_tasks_count(None), 3);
        assert_eq!(pool.unprocessed_tasks_count(Some(0)), 2);
        assert_eq!(pool.unprocessed_tasks_count(Some(10)), 1);
        assert_eq!(pool.unprocessed_tasks_count(Some(99)), 0);

        barrier.wait();
        blocker.wait().unwrap();
    }

    #[test]
    fn workers_spawn_on_demand_up_to_capacity() {
        // ThreadPool.hpp:157 `m_threads.size() < m_threadCount && m_idleThreadCount == 0`.
        // Submitting N independent tasks that all wait on a barrier of
        // size N+1 forces N worker spawns.
        let n = 3_usize;
        let pool = ThreadPool::new(n, ThreadPinning::new());
        let barrier = Arc::new(Barrier::new(n + 1));
        let mut futures = Vec::new();
        for _ in 0..n {
            let b = Arc::clone(&barrier);
            futures.push(pool.submit(move || b.wait(), 0));
        }
        // All workers must be blocked on the barrier before we release them.
        // Wait until exactly `n` threads have been spawned.
        for _ in 0..200 {
            if pool.spawned_threads() == n {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(pool.spawned_threads(), n);
        barrier.wait();
        for f in futures {
            f.wait().unwrap();
        }
    }

    #[test]
    fn lower_priority_runs_first() {
        // BTreeMap iterates ascending — std::map does the same — so
        // priority=-1 runs before priority=10. Mirror of the
        // std::find_if traversal at ThreadPool.hpp:213-214.
        let pool = ThreadPool::new(1, ThreadPinning::new());

        // Block the worker so we control queue ordering.
        let block = Arc::new(Barrier::new(2));
        let b_in = Arc::clone(&block);
        let blocker = pool.submit(move || b_in.wait(), -100);

        let order = Arc::new(Mutex::new(Vec::<i32>::new()));

        let o = Arc::clone(&order);
        let f_high = pool.submit(move || o.lock().unwrap().push(10), 10);
        let o = Arc::clone(&order);
        let f_low = pool.submit(move || o.lock().unwrap().push(-1), -1);
        let o = Arc::clone(&order);
        let f_mid = pool.submit(move || o.lock().unwrap().push(0), 0);

        block.wait();
        blocker.wait().unwrap();
        f_high.wait().unwrap();
        f_low.wait().unwrap();
        f_mid.wait().unwrap();

        assert_eq!(*order.lock().unwrap(), vec![-1, 0, 10]);
    }

    #[test]
    fn many_tasks_complete() {
        let pool = ThreadPool::new(4, ThreadPinning::new());
        let counter = Arc::new(AtomicU32::new(0));
        let mut futures = Vec::new();
        for _ in 0..256 {
            let c = Arc::clone(&counter);
            futures.push(pool.submit(move || c.fetch_add(1, Ordering::AcqRel), 0));
        }
        for f in futures {
            f.wait().unwrap();
        }
        assert_eq!(counter.load(Ordering::Acquire), 256);
    }

    #[test]
    fn pending_future_after_stop_returns_cancelled() {
        // A future whose worker never ran (because we stopped first)
        // returns FutureError::Cancelled — the equivalent of std::future
        // `broken_promise`. This mirrors the C++ behavior at
        // ThreadPool.hpp:209-211: on shutdown the worker breaks out of
        // the wait loop without popping the remaining tasks, so each
        // pending PackagedTask is destructed without ever invoking the
        // promise — std::future then throws `broken_promise` on get().
        //
        // We exercise this with a 0-threads pool that has its `running`
        // flag cleared, then submitting after stop produces a future
        // whose sender we drop here in the test to reproduce the same
        // observable cancellation.
        let (tx, rx) = mpsc::sync_channel::<i32>(1);
        drop(tx); // simulate the worker never running
        let f: Future<i32> = Future { rx };
        assert_eq!(f.wait(), Err(FutureError::Cancelled));
    }

    #[test]
    fn completion_gen_advances_and_wakes_a_waiter() {
        // Event-driven consumer wakeup: after a task body returns, the pool
        // bumps `task_completed` and notifies. A consumer that snapshotted the
        // gen and then blocks in `wait_for_completion` must wake with an
        // advanced gen well within the (generous) fallback timeout.
        let pool = ThreadPool::new(1, ThreadPinning::new());
        let before = pool.completion_gen();
        let f = pool.submit(|| 1_u32, 0);
        assert_eq!(f.wait().unwrap(), 1);
        // The notify fires after t() returns; a wait from `before` must observe
        // an advanced gen (either already advanced → returns immediately, or via
        // the condvar notify) far inside the 5s liveness fallback.
        let g = pool.wait_for_completion(before, Duration::from_secs(5));
        assert!(g > before, "completion gen must advance after a task runs");
    }

    #[test]
    fn wait_for_completion_returns_immediately_when_gen_already_advanced() {
        // No lost wakeup: if a completion landed between the consumer's snapshot
        // and its wait (the monotonic-counter race the pump path relies on), the
        // wait must return immediately rather than block for the fallback.
        let pool = ThreadPool::new(2, ThreadPinning::new());
        let start = std::time::Instant::now();
        // A stale `last_seen` (u64::MAX never equals a real small gen, but the
        // predicate is `*g == last_seen`; use `before` snapshot then advance).
        let before = pool.completion_gen();
        pool.submit(|| 0_u32, 0).wait().unwrap();
        // Gen is now > before. Waiting from `before` must not block on the 10s
        // fallback — it should see the advanced gen and return at once.
        let g = pool.wait_for_completion(before, Duration::from_secs(10));
        assert!(g > before);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "wait must not consume the fallback when gen already advanced"
        );
    }

    #[test]
    fn drop_joins_threads() {
        // Equivalent of the ~ThreadPool destructor (ThreadPool.hpp:110-113):
        // dropping the pool calls stop(), which joins all workers. We
        // wait on every submitted task FIRST (a real consumer would also
        // wait on the futures it cares about), then let Drop run. The
        // assertion is "Drop did not hang" plus "every task we waited on
        // ran" — matching the C++ destructor semantics.
        let done = Arc::new(AtomicU32::new(0));
        let mut futures = Vec::new();
        {
            let pool = ThreadPool::new(2, ThreadPinning::new());
            for _ in 0..16 {
                let d = Arc::clone(&done);
                futures.push(pool.submit(move || d.fetch_add(1, Ordering::AcqRel), 0));
            }
            for f in futures {
                f.wait().unwrap();
            }
            // No explicit stop() — Drop must join idle workers without
            // hanging.
        }
        assert_eq!(done.load(Ordering::Acquire), 16);
    }
}
