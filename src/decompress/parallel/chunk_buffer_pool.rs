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
//! ## Lifecycle
//!
//! 1. Worker dispatches a decode. `take_u8 / take_u16` pop a recycled
//!    Vec from the pool (or `Vec::with_capacity(cap)` on miss).
//! 2. `ChunkData::new_with_buffers(offset, config, u16, u8)` wraps the
//!    Vecs. The chunk's data fills these buffers during decode.
//! 3. After the consumer is done with the chunk (post `drain_one_pending`
//!    wrote `chunk.data` to the writer and `apply_window` resolved
//!    `data_with_markers` into `data`), the chunk is dropped. The
//!    `Drop` impl on `ChunkData` calls `return_u8 / return_u16`,
//!    pushing the buffers back into the pool. Drop happens on
//!    whichever thread holds the last `Arc<ChunkData>`; the pool is
//!    cross-thread shared so this is safe.
//!
//! ## Bounds
//!
//! Pool caps at `MAX_POOLED` Vecs per type to prevent unbounded
//! growth (e.g., if the consumer is slow and chunks pile up briefly).
//! Vecs exceeding the cap are dropped normally. Sized to comfortably
//! hold `parallelization * 2` chunks in flight (pool-size 16 → 32 ×
//! ~80 MiB = 2.5 GiB worst case, well under host memory).
//!
//! ## Page-fault gap vs vendor (open)
//!
//! Neurotic x86_64 silesia profile: gzippy spends ~40% of CPU in
//! `asm_exc_page_fault` + `clear_page_erms`; rapidgzip spends ~17%.
//! The gap is rpmalloc's per-thread arena keeping pages warm across
//! malloc/free cycles within a process. `std::alloc::System` munmaps
//! large-Vec deallocations and remaps fresh on next allocation,
//! re-faulting every page.
//!
//! A previous experiment pre-warmed the pool by touching pages on the
//! consumer thread before workers spawn. Measured -50% throughput on
//! the bench (each fresh CLI process paid ~170 ms of pre-touch work
//! on a ~750 ms decode). Reverted; **do not re-add a prewarm call
//! without a daemon-mode caller AND a 20-trial bench-on-branch gate.**
//!
//! Closing this band requires one of:
//!   - `allocator-api2` polyfill + `Vec<T, RpmallocAlloc>` for chunk
//!     buffers only (per-Vec, matches vendor's `FasterVector<u8,
//!     RpmallocAllocator>`).
//!   - `#[global_allocator] = rpmalloc::RpMalloc` (global; the
//!     mimalloc/jemalloc tries regressed, so this is unproven).
//!   - Daemon-mode CLI wrapper (sidesteps the fresh-process problem).

#![allow(dead_code)]

use std::sync::Mutex;

/// Cap on pool size per Vec type. Sized to absorb a brief in-flight
/// burst without unbounded growth.
const MAX_POOLED: usize = 64;

static U8_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static U16_POOL: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());

/// Take a `Vec<u8>` from the pool (cleared). Falls back to a fresh
/// allocation with the given capacity hint if the pool is empty.
pub fn take_u8(min_capacity: usize) -> Vec<u8> {
    if let Ok(mut pool) = U8_POOL.lock() {
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
    Vec::with_capacity(min_capacity)
}

/// Take a `Vec<u16>` from the pool (cleared). See `take_u8`.
pub fn take_u16(min_capacity: usize) -> Vec<u16> {
    if let Ok(mut pool) = U16_POOL.lock() {
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
    Vec::with_capacity(min_capacity)
}

/// Return a `Vec<u8>` to the pool. Vec is cleared (capacity retained)
/// and pushed back if room remains; otherwise it drops normally.
pub fn return_u8(mut v: Vec<u8>) {
    if v.capacity() == 0 {
        return;
    }
    RETURN_U8_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut pool) = U8_POOL.lock() {
        if pool.len() < MAX_POOLED {
            v.clear();
            pool.push(v);
        }
    }
}

/// Return a `Vec<u16>` to the pool. See `return_u8`.
pub fn return_u16(mut v: Vec<u16>) {
    if v.capacity() == 0 {
        return;
    }
    RETURN_U16_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut pool) = U16_POOL.lock() {
        if pool.len() < MAX_POOLED {
            v.clear();
            pool.push(v);
        }
    }
}

// A previous revision had `prewarm()` + `prefault_pages()` here.
// They were measured to regress SM throughput by 50% on neurotic
// (fresh-CLI-process bench paid ~170 ms of consumer-thread page-touch
// without amortization). Deleted to avoid dead-code-with-future-promises.
// Notes on the page-fault gap live in the module-level docs above;
// closing it requires a real per-Vec allocator (allocator-api2 +
// rpmalloc-rs) or daemon-mode CLI wiring, not a pre-touch loop.

/// Test-only counters that prove the recycle path is being exercised.
/// Catches the silent-rot case where someone reverts the worker call
/// sites to `ChunkData::new` (fresh-allocate path) without flipping
/// the corresponding test — exactly the failure mode that masked
/// today's MAX-ENCODED-OFFSET regression.
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
