#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]

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
            TAKE_U8_HITS_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v.clear();
            if v.capacity() < min_capacity {
                v.reserve(min_capacity - v.capacity());
            }
            return v;
        }
    }
    TAKE_U8_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TAKE_U8_MISSES_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    types::u8_with_capacity(min_capacity)
}

/// Take a `Vec<u16>` from the current worker's pool.
pub fn take_u16(min_capacity: usize) -> U16 {
    let idx = pool_index_for_take();
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        if let Some(mut v) = pool.pop() {
            TAKE_U16_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            TAKE_U16_HITS_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v.clear();
            if v.capacity() < min_capacity {
                v.reserve(min_capacity - v.capacity());
            }
            return v;
        }
    }
    TAKE_U16_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TAKE_U16_MISSES_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    types::u16_with_capacity(min_capacity)
}

/// Return a `Vec<u8>` to the pool for `owner_worker` (recorded at take time).
pub fn return_u8_to_worker(owner_worker: usize, mut v: U8) {
    if v.capacity() == 0 {
        return;
    }
    RETURN_U8_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let idx = owner_worker.min(MAX_WORKERS - 1);
    RETURN_U8_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    RETURN_U16_BY_WORKER[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        if pool.len() < MAX_POOLED {
            v.clear();
            pool.push(v);
        }
    }
}

// =============================================================================
// Z-allocator lever: worker-side pre-warm
// =============================================================================
//
// Per `project_allocator_lever_confirmed` (May 27 2026): chunk_buffer_pool's
// u16 pool has a ~92% miss rate on speculative chunks, contributing to the
// +26pp memcpy+clear_page gap vs vendor (~130 ms wall potential — ~75% of
// remaining gap to ISA-L).
//
// Mechanism: most window-known chunks return their u16 Vec with capacity=0
// (lazy-alloc path was never grown). `return_u16_to_worker` filters cap-0
// Vecs out, so the pool never accumulates warm u16 buffers. The next
// speculative chunk takes from an empty pool and pays a fresh fault cost.
//
// Fix: at WORKER init (after `bind_worker_pool_index`, on the worker's own
// thread), pre-allocate N Vecs per pool with the expected chunk capacity.
// 16 workers fault their own 2× 16 MB buffers in parallel — total wall is
// ~32 MB worth of faults distributed across cores, vs ~512 MB serial on the
// consumer thread (the prior failed approach in `feedback_z2_falsified`).
//
// ## Falsification gate (`project_allocator_lever_confirmed` §"Falsification gate")
//
// Default OFF per the memory rule "do not re-add a prewarm call without a
// daemon-mode caller AND a 20-trial bench-on-branch gate". Enable on neurotic
// via `GZIPPY_PREWARM_POOL=1` env-var (no rebuild), then:
//
//   - Wall drops ≥ 50 ms p50: pre-warm is the fix. Flip default to ON.
//   - Wall neutral: u16 pool ISN'T dominant source. Falsifies hypothesis #1.
//     Move to `absorb_isal_tail` (#2) or `submit_post_process_to_pool` (#3).
//   - Wall regresses: different pre-warm strategy needed. Revert.
//
// The new approach (worker-side, post-bind) is mechanistically distinct from
// the prior failed consumer-side approach — different RSS dynamics, fault
// parallelization. Pre-blessed by `project_allocator_lever_confirmed`.

/// Number of u8 Vecs to pre-allocate per worker pool. Sized to a handful of
/// in-flight chunks (2 — the front of decode + the back of consumer-drain).
/// Capped well below `MAX_POOLED`=8 so normal decode-time hits still cycle.
const PREWARM_U8_COUNT: usize = 2;

/// Number of u16 Vecs to pre-allocate per worker pool. Speculative chunks
/// emit u16 markers; ~half of chunks are speculative on a cold corpus.
const PREWARM_U16_COUNT: usize = 2;

/// u8 cap for the worker-init prewarm. Sized at the common
/// `split_chunk_size` (4 MiB) rather than `max_decoded_chunk_size` (80 MiB)
/// — most chunks are at the common size, so prewarm at 4 MiB covers the
/// hot path; the rare 20× growth chunks pay the realloc tax (low aggregate
/// impact). Total reserved address space at startup:
/// `PREWARM_U8_COUNT * PREWARM_U8_CAP * 16 workers = 128 MiB`. Pages are
/// NOT committed by prewarm (no pre-fault), so RSS stays unchanged until
/// the decode actually writes to them.
const PREWARM_U8_CAP: usize = 4 * 1024 * 1024;

/// u16 cap (in elements). 4 MiB elements = 8 MiB bytes. Same sizing
/// rationale as `PREWARM_U8_CAP`.
const PREWARM_U16_CAP: usize = 4 * 1024 * 1024;

/// Per-worker pre-warm runs counter. Read by the routing-trap test to confirm
/// the lever actually fired when `GZIPPY_PREWARM_POOL=1`.
pub static WORKER_PREWARM_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn prewarm_kill_switch_active() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("GZIPPY_PREWARM_POOL").is_ok())
}

/// Pre-allocate `PREWARM_U8_COUNT` u8 Vecs at `PREWARM_U8_CAP` and
/// `PREWARM_U16_COUNT` u16 Vecs at `PREWARM_U16_CAP` into the current
/// worker's pool. Must be called from the worker thread *after*
/// [`bind_worker_pool_index`]. No-op when the `GZIPPY_PREWARM_POOL`
/// env-var is unset (default), or when called off a bound worker thread.
///
/// The allocations are pushed straight into the pool's per-worker LIFO.
/// Vecs have address-space capacity but no committed pages — page faults
/// still happen during decode-time writes, on the worker thread, exactly
/// like today. What pre-warm eliminates is the **Vec-doubling-realloc
/// thrash** when `append_markered` grows a take_u16(0) Vec from cap=0
/// (the lazy-alloc path used by speculative chunks): without prewarm,
/// growing from 0 → 4 MiB triggers ~11 intermediate reallocations, each
/// one mmap'ing a fresh region; with prewarm, the take_u16(0) pops a
/// 4 MiB Vec already, no realloc needed for chunks within that bound.
pub fn prewarm_current_worker() {
    if !prewarm_kill_switch_active() {
        return;
    }
    let idx = match current_worker_pool_index() {
        Some(i) => i,
        None => return, // not on a bound worker thread; skip
    };
    WORKER_PREWARM_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let u8_cap = PREWARM_U8_CAP;
    let u16_cap = PREWARM_U16_CAP;

    // Pre-allocate-only — no page-fault touch. Per `feedback_z2_falsified`,
    // pre-faulting at allocation time serializes ~20480 faults that would
    // otherwise parallelize across workers as bytes are actually written.
    // What pre-warm DOES buy us: subsequent `take_u16(0)` finds a Vec with
    // capacity = u16_cap, so `append_markered` (the speculative-chunk
    // marker push path) does NOT trigger Vec-doubling-realloc. The
    // realloc-doubling path would mmap 11 intermediate sizes (16 KB → 32 KB
    // → ... → 16 MB), each one fresh-faulted; with pre-warm those 10 extra
    // mmaps + their fault cycles are eliminated. Page faults for the actual
    // data writes still happen, but only once per page and on the worker
    // thread that uses them — parallelized.
    if let Ok(mut pool) = u8_pools()[idx].lock() {
        for _ in 0..PREWARM_U8_COUNT {
            if pool.len() >= MAX_POOLED {
                break;
            }
            pool.push(types::u8_with_capacity(u8_cap));
        }
    }
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        for _ in 0..PREWARM_U16_COUNT {
            if pool.len() >= MAX_POOLED {
                break;
            }
            pool.push(types::u16_with_capacity(u16_cap));
        }
    }
}

pub static TAKE_U8_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U8_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RETURN_U8_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U16_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TAKE_U16_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RETURN_U16_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-worker hit/miss/return counters. Surface the distribution
/// across worker pool indexes — the global counters above can't
/// distinguish "all 16 workers see 5 misses + 1 hit" (cold-start /
/// decode-to-Drop lag) from "worker 0 sees 80 misses, others see 0"
/// (consumer-thread-on-wrong-bucket). Different fixes for each.
pub static TAKE_U8_HITS_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];
pub static TAKE_U8_MISSES_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];
pub static TAKE_U16_HITS_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];
pub static TAKE_U16_MISSES_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];
pub static RETURN_U8_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];
pub static RETURN_U16_BY_WORKER: [std::sync::atomic::AtomicU64; MAX_WORKERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; MAX_WORKERS];

/// Test-only entry-point that runs the prewarm logic unconditionally,
/// bypassing the `GZIPPY_PREWARM_POOL` env-var check. Used by unit tests
/// to verify the prewarm body without relying on the OnceLock-cached
/// env-var (which can't be toggled mid-process).
#[cfg(test)]
pub fn prewarm_current_worker_force() {
    let idx = match current_worker_pool_index() {
        Some(i) => i,
        None => return,
    };
    WORKER_PREWARM_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let u8_cap = PREWARM_U8_CAP;
    let u16_cap = PREWARM_U16_CAP;
    if let Ok(mut pool) = u8_pools()[idx].lock() {
        for _ in 0..PREWARM_U8_COUNT {
            if pool.len() >= MAX_POOLED {
                break;
            }
            pool.push(types::u8_with_capacity(u8_cap));
        }
    }
    if let Ok(mut pool) = u16_pools()[idx].lock() {
        for _ in 0..PREWARM_U16_COUNT {
            if pool.len() >= MAX_POOLED {
                break;
            }
            pool.push(types::u16_with_capacity(u16_cap));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pre-warm pushes `PREWARM_*_COUNT` Vecs of the right cap into the
    /// current worker's pool. Uses a high worker index (60) to avoid
    /// collision with real production worker indices (typically 0..16).
    /// Cleans up the pool at end so concurrent tests aren't affected.
    #[test]
    fn prewarm_populates_worker_pool_with_capacity() {
        const TEST_WORKER: usize = 60;
        bind_worker_pool_index(TEST_WORKER);

        // Drain any prior state from this slot (other tests using index 60).
        u8_pools()[TEST_WORKER].lock().unwrap().clear();
        u16_pools()[TEST_WORKER].lock().unwrap().clear();

        prewarm_current_worker_force();

        let u8_pool = u8_pools()[TEST_WORKER].lock().unwrap();
        let u16_pool = u16_pools()[TEST_WORKER].lock().unwrap();
        assert_eq!(
            u8_pool.len(),
            PREWARM_U8_COUNT,
            "prewarm should push {} u8 Vecs",
            PREWARM_U8_COUNT
        );
        assert_eq!(
            u16_pool.len(),
            PREWARM_U16_COUNT,
            "prewarm should push {} u16 Vecs",
            PREWARM_U16_COUNT
        );
        for v in u8_pool.iter() {
            assert!(
                v.capacity() >= PREWARM_U8_CAP,
                "u8 prewarm cap {} < expected {}",
                v.capacity(),
                PREWARM_U8_CAP
            );
        }
        for v in u16_pool.iter() {
            assert!(
                v.capacity() >= PREWARM_U16_CAP,
                "u16 prewarm cap {} < expected {}",
                v.capacity(),
                PREWARM_U16_CAP
            );
        }
        drop(u8_pool);
        drop(u16_pool);

        // Clean up.
        u8_pools()[TEST_WORKER].lock().unwrap().clear();
        u16_pools()[TEST_WORKER].lock().unwrap().clear();
    }
}
