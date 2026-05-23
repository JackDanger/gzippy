//! Cross-thread Vec recycler for `ChunkData`'s `data` (`Vec<u8>`) and
//! `data_with_markers` (`Vec<u16>`) buffers.
//!
//! ## Why this exists
//!
//! Vendor's per-Vec allocator wiring at
//! `vendor/.../core/FasterVector.hpp:120-128`:
//!
//! ```cpp
//! #if defined( LIBRAPIDARCHIVE_WITH_RPMALLOC )
//! template<typename T>
//! using FasterVector = std::vector<T, RpmallocAllocator<T>>;
//! #else
//! template<typename T>
//! using FasterVector = std::vector<T>;
//! #endif
//! ```
//!
//! `RpmallocAllocator` plugs vendor's `FasterVector<uint8_t>` (the
//! analog of gzippy's `ChunkData::data`) into rpmalloc's per-thread
//! arena. Releasing a vector returns its pages to rpmalloc's free
//! list; the next allocation on the same thread reuses those pages
//! without faulting them. Across a multi-chunk decode, this is what
//! keeps wall-time cost roughly chunk-sized rather than
//! chunk-count-times-chunk-sized.
//!
//! Stable Rust has no allocator parameter on `Vec`. Wiring an
//! arena-allocator crate (mimalloc / rpmalloc-rs) as the **global**
//! allocator would change behavior for every allocation in gzippy.
//! Instead, this module recycles the specific Vecs that
//! `ChunkData::new_with_buffers` accepts — a narrower mirror of
//! vendor's per-Vec arena. Captures the same first-touch-amortization
//! benefit without changing global allocation behavior.
//!
//! ## Per-worker LIFO pools (May 2026)
//!
//! Buffers are keyed by `ThreadPool` worker index (`bind_worker_pool_index`
//! in `worker_main`, after core pinning). Each worker's `take_u8` /
//! `take_u16` pops from that worker's LIFO; `ChunkData::Drop` returns
//! to the owning worker's pool via `pool_worker_index` recorded at
//! allocation time. This preserves Vec **capacity** across chunks on
//! the same worker — avoiding re-faulting pages on re-take — but does
//! **not** replace a true per-thread rpmalloc arena (freed pages may
//! still be munmapped when the pool is full or on miss). With the
//! `arena-allocator` feature, `take_*` / `return_*` use
//! `Vec<T, RpmallocAlloc>` so freed pages stay in rpmalloc's per-thread
//! free list (vendor `FasterVector.hpp:120-128`).
//!
//! ## Lifecycle
//!
//! 1. Worker dispatches a decode. `take_u8 / take_u16` pop a recycled
//!    Vec from the worker's pool (or `Vec::with_capacity(cap)` on miss).
//! 2. `ChunkData::new_with_buffers(offset, config, u16, u8)` wraps the
//!    Vecs. The chunk's data fills these buffers during decode.
//! 3. After the consumer is done with the chunk (post `drain_one_pending`
//!    wrote `chunk.data` to the writer and `apply_window` resolved
//!    `data_with_markers` into `data`), the chunk is dropped. The
//!    `Drop` impl on `ChunkData` calls `return_u8_to_worker /
//!    return_u16_to_worker`, pushing the buffers back into the owning
//!    worker's pool.
//!
//! ## Bounds
//!
//! Pool caps at `MAX_POOLED` Vecs per worker per type to prevent
//! unbounded growth. Sized to comfortably hold `parallelization * 2`
//! chunks in flight per worker.
//!
//! ## Page-fault gap vs vendor (open)
//!
//! Neurotic x86_64 silesia profile: gzippy spends ~40% of CPU in
//! `asm_exc_page_fault` + `clear_page_erms`; rapidgzip spends ~17%.
//! The gap is rpmalloc's per-thread arena keeping pages warm across
//! malloc/free cycles within a process. `std::alloc::System` munmaps
//! large-Vec deallocations and remaps fresh on next allocation,
//! re-faulting every page. Per-worker LIFO pools address part of this
//! (capacity reuse on the same worker) but have **not** been measured
//! to close the 40%→17% gap — treat as an open empirical question.
//!
//! A previous experiment pre-warmed the pool by touching pages on the
//! consumer thread before workers spawn. Measured -50% throughput on
//! the bench (each fresh CLI process paid ~170 ms of pre-touch work
//! on a ~750 ms decode). Reverted; **do not re-add a prewarm call
//! without a daemon-mode caller AND a 20-trial bench-on-branch gate.**
//!
//! Closing the remaining gap still requires one of:
//!   - `allocator-api2` polyfill + `Vec<T, RpmallocAlloc>` for chunk
//!     buffers only (per-Vec, matches vendor's `FasterVector<u8,
//!     RpmallocAllocator>`).
//!   - `#[global_allocator] = rpmalloc::RpMalloc` (global; the
//!     mimalloc/jemalloc tries regressed, so this is unproven).
//!   - Daemon-mode CLI wrapper (sidesteps the fresh-process problem).

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

use crate::decompress::parallel::rpmalloc_alloc::types::{self, U16, U8};

/// Cap on pool size per worker per Vec type. Sized to a handful of
/// in-flight chunks per worker (not the old shared cap of 64).
const MAX_POOLED: usize = 8;
const MAX_WORKERS: usize = 64;

thread_local! {
    static WORKER_POOL_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

fn u8_pools() -> &'static [Mutex<Vec<U8>>] {
    static POOLS: OnceLock<Vec<Mutex<Vec<U8>>>> = OnceLock::new();
    POOLS.get_or_init(|| (0..MAX_WORKERS).map(|_| Mutex::new(Vec::new())).collect())
}

fn u16_pools() -> &'static [Mutex<Vec<U16>>] {
    static POOLS: OnceLock<Vec<Mutex<Vec<U16>>>> = OnceLock::new();
    POOLS.get_or_init(|| (0..MAX_WORKERS).map(|_| Mutex::new(Vec::new())).collect())
}

/// Called once per `ThreadPool` worker thread on entry to `worker_main`,
/// after core pinning and before any decode task runs.
pub fn bind_worker_pool_index(index: usize) {
    debug_assert!(
        index < MAX_WORKERS,
        "worker index {index} exceeds MAX_WORKERS {MAX_WORKERS}"
    );
    WORKER_POOL_INDEX.with(|c| c.set(Some(index.min(MAX_WORKERS - 1))));
}

pub fn current_worker_pool_index() -> Option<usize> {
    WORKER_POOL_INDEX.with(|c| c.get())
}

fn pool_index_for_take() -> usize {
    current_worker_pool_index().unwrap_or(0)
}

/// Take a `Vec<u8>` from the current worker's pool (or worker 0 if unbound).
/// Decode tasks run on pool workers and record the correct index; trial
/// decodes on the consumer thread bucket returns to worker 0's pool.
pub fn take_u8(min_capacity: usize) -> U8 {
    let idx = pool_index_for_take();
    if let Ok(mut pool) = u8_pools()[idx].lock() {
        if let Some(mut v) = pool.pop() {
            TAKE_U8_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v.clear();
            if v.capacity() < min_capacity {
                v.reserve(min_capacity - v.capacity());
            }
            return v;
        }
    }
    TAKE_U8_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    types::u8_with_capacity(min_capacity)
}

/// Take a `Vec<u16>` from the current worker's pool.
pub fn take_u16(min_capacity: usize) -> U16 {
    let idx = pool_index_for_take();
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        if let Some(mut v) = pool.pop() {
            TAKE_U16_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v.clear();
            if v.capacity() < min_capacity {
                v.reserve(min_capacity - v.capacity());
            }
            return v;
        }
    }
    TAKE_U16_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    types::u16_with_capacity(min_capacity)
}

/// Return a `Vec<u8>` to the pool for `owner_worker` (recorded at take time).
pub fn return_u8_to_worker(owner_worker: usize, mut v: U8) {
    if v.capacity() == 0 {
        return;
    }
    RETURN_U8_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let idx = owner_worker.min(MAX_WORKERS - 1);
    if let Ok(mut pool) = u8_pools()[idx].lock() {
        if pool.len() < MAX_POOLED {
            v.clear();
            pool.push(v);
        }
    }
}

/// Return a `Vec<u16>` to the pool for `owner_worker`.
pub fn return_u16_to_worker(owner_worker: usize, mut v: U16) {
    if v.capacity() == 0 {
        return;
    }
    RETURN_U16_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let idx = owner_worker.min(MAX_WORKERS - 1);
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        if pool.len() < MAX_POOLED {
            v.clear();
            pool.push(v);
        }
    }
}

pub static TAKE_U8_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U8_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RETURN_U8_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U16_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U16_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RETURN_U16_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Unit tests intentionally omitted: the pool is a process-global LIFO
// that other tests (via `ChunkData::new`) concurrently take/return
// from, so any "round-trip" test that pops what it just pushed is
// inherently flaky. Integration coverage comes from every test that
// exercises `ChunkData::new` — 800+ tests run through this path.
