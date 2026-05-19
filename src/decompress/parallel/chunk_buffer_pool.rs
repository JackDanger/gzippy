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
//! ## Page pre-fault
//!
//! `Vec::with_capacity(N)` reserves address space but does NOT commit
//! pages — the kernel lazily zero-fills on first write. Each fresh
//! worker thread therefore page-faults its way through a fresh chunk
//! buffer on its first decode (4 KB at a time, via `asm_exc_page_fault`
//! → `clear_page_erms`). On real x86_64 silesia benchmarks, this
//! accounts for **~40% of total CPU time** vs vendor's ~17% (rpmalloc
//! pre-faults pages on `mmap` via `MAP_POPULATE` and recycles them
//! warm through its arena).
//!
//! `prewarm(num_buffers, capacity)` pre-allocates `num_buffers` Vecs
//! AND touches every page via byte writes, then pushes them into the
//! pool with warm pages. Called by the parallel-SM driver before
//! workers spawn: subsequent `take_u8` calls all hit the pool with
//! warm pages, and the per-worker first-touch page-fault storm
//! collapses to zero.

#![allow(dead_code)]

use std::sync::Mutex;

/// Cap on pool size per Vec type. Sized to absorb a brief in-flight
/// burst without unbounded growth.
const MAX_POOLED: usize = 64;

/// Linux x86_64 base page. Conservative default that works on every
/// host we care about. Hugepages, when active, just mean some of these
/// page-strided touches will hit the same page — harmless.
const PAGE_SIZE: usize = 4096;

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

/// Pre-fault every page of a buffer by writing one byte per `PAGE_SIZE`
/// bytes. After this returns, every page is committed in the worker's
/// resident-set; first writes during decode hit warm pages.
///
/// Mirror of `MAP_POPULATE` semantics for rpmalloc-allocated regions.
/// `unsafe` because we write into uninitialized memory via raw pointer;
/// the writes are bytes of value 0 which is also what the kernel would
/// have lazily set them to, so this is value-preserving.
#[inline]
unsafe fn prefault_pages(ptr: *mut u8, capacity: usize) {
    if capacity == 0 {
        return;
    }
    let mut off = 0usize;
    while off < capacity {
        // Write a zero byte at this offset. Forces a page fault if the
        // page is not yet committed; subsequent fault becomes a no-op.
        // SAFETY: caller guarantees `ptr..ptr+capacity` is owned
        // allocation valid for writes (Vec::with_capacity guarantees
        // this for the reserved-but-uninitialized range).
        std::ptr::write_volatile(ptr.add(off), 0u8);
        off += PAGE_SIZE;
    }
}

/// Pre-allocate `count` `Vec<u8>` and `Vec<u16>` buffers of the given
/// capacity, pre-fault every page, and push them into the pools.
///
/// Called by `single_member::decompress_parallel` BEFORE workers spawn.
/// The consumer thread eats the page-fault cost serially — total work
/// is the same, but it stays off the workers' critical path, and the
/// kernel `mm_lock` contention from N workers faulting in parallel is
/// avoided.
///
/// Subsequent `take_u8` / `take_u16` calls all hit the pool (subject
/// to MAX_POOLED) with warm pages. The pool stays populated until
/// process exit because Drop returns buffers to the pool, and the
/// pool's static lifetime outlives all decode threads.
///
/// `count` is typically `num_threads + 2` (1 buffer per worker plus a
/// small overflow for the consumer's pending queue). `byte_capacity`
/// is the chunk_size_bytes from the driver; `u16_capacity` is the same
/// in u16 units (the marker pipeline produces one u16 per decoded
/// byte, so byte_capacity == u16_capacity for the slow path).
pub fn prewarm(count: usize, byte_capacity: usize) {
    if count == 0 || byte_capacity == 0 {
        return;
    }
    // u8 pool — pre-fault the full byte capacity.
    if let Ok(mut pool) = U8_POOL.lock() {
        while pool.len() < MAX_POOLED && pool.len() < count {
            let mut v: Vec<u8> = Vec::with_capacity(byte_capacity);
            // SAFETY: v has `byte_capacity` bytes reserved (uninit).
            unsafe { prefault_pages(v.as_mut_ptr(), byte_capacity) };
            pool.push(v);
            PREWARM_U8_PUSHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    // u16 pool — pre-fault the BYTE extent (capacity * 2 bytes).
    if let Ok(mut pool) = U16_POOL.lock() {
        while pool.len() < MAX_POOLED && pool.len() < count {
            let mut v: Vec<u16> = Vec::with_capacity(byte_capacity);
            // SAFETY: v has `byte_capacity` u16s reserved = capacity*2
            // bytes. Cast to u8 ptr and pre-fault the byte extent.
            unsafe { prefault_pages(v.as_mut_ptr() as *mut u8, byte_capacity.saturating_mul(2)) };
            pool.push(v);
            PREWARM_U16_PUSHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

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
pub static PREWARM_U8_PUSHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREWARM_U16_PUSHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Unit tests intentionally omitted: the pool is a process-global LIFO
// that other tests (via `ChunkData::new`) concurrently take/return
// from, so any "round-trip" test that pops what it just pushed is
// inherently flaky. Integration coverage comes from every test that
// exercises `ChunkData::new` — 800+ tests run through this path.
