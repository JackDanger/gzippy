#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

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

/// Lever L1 (2026-05-28): hint the kernel to use 2 MiB transparent huge
/// pages for fresh chunk-buffer allocations. Reduces page-fault count
/// ~512× on first touch (12 MiB chunk: 3072 4 KiB faults → 6 2 MiB
/// faults). Default OFF until A/B on neurotic confirms a win.
///
/// MADV_HUGEPAGE is a hint; the kernel grants it opportunistically via
/// khugepaged. On a one-shot CLI process khugepaged may not get
/// scheduling time, so the hint may be ignored. MADV_COLLAPSE (kernel
/// 6.1+) is synchronous but adds latency at the madvise call site.
///
/// Per perf attribution 2026-05-28: clear_page_erms is 13.26% of CPU
/// (kernel page-zeroing on first touch); the per-fault overhead is the
/// bigger cost than the zeroing itself. Fewer faults → less overhead.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn use_hugepage_hint() -> bool {
    use std::sync::OnceLock as Once;
    static USE: Once<bool> = Once::new();
    *USE.get_or_init(|| std::env::var_os("GZIPPY_HUGEPAGE").is_some())
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn use_hugepage_hint() -> bool {
    false
}

/// Apply `MADV_HUGEPAGE` hint to a freshly-allocated chunk buffer.
/// Only effective on Linux; called on miss path of `take_u8` so the
/// hint is set once per backing allocation, not per pool reuse.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn hugepage_hint(ptr: *mut u8, len: usize) {
    if !use_hugepage_hint() || len < 2 * 1024 * 1024 {
        return;
    }
    // Round pointer up to 4 KiB page boundary and length down to whole
    // pages. madvise requires page-aligned args; the kernel rejects
    // unaligned calls with EINVAL (silently dropped since we ignore
    // errors here).
    const PAGE_SIZE: usize = 4096;
    let addr = ptr as usize;
    let aligned_addr = (addr + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let end = addr + len;
    let aligned_end = end & !(PAGE_SIZE - 1);
    if aligned_end > aligned_addr {
        let aligned_len = aligned_end - aligned_addr;
        unsafe {
            libc::madvise(
                aligned_addr as *mut libc::c_void,
                aligned_len,
                libc::MADV_HUGEPAGE,
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[inline(always)]
#[allow(dead_code)]
fn hugepage_hint(_ptr: *mut u8, _len: usize) {}

/// Cap on pool size per worker per Vec type. Sized to a handful of
/// in-flight chunks per worker (not the old shared cap of 64).
// Sized for T16 in-flight depth: eager Ready-drain returns buffers sooner,
// but several Async heads can still hold buffers while workers decode ahead.
const MAX_POOLED: usize = 12;
const MAX_WORKERS: usize = 64;

thread_local! {
    static WORKER_POOL_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

// (GZIPPY_SHARED_POOL experiment removed 2026-05-28: a shared cross-worker
// pool showed NO measurable wall effect on a quiet-enough box — page-faults
// −3.7% but cycles within noise. The per-worker pool collapses at T=16
// because chunks sit in the consumer reorder buffer until Drop (buffer
// RETURN LATENCY), not because of pool topology. See
// project_real_gap_pinned_2026_05_28 / project_parallel_test_hang memory; the
// real fix is earlier buffer return, queued for the parallel-pipeline work.)

fn u8_pools() -> &'static [Mutex<Vec<U8>>] {
    static POOLS: OnceLock<Vec<Mutex<Vec<U8>>>> = OnceLock::new();
    POOLS.get_or_init(|| (0..MAX_WORKERS).map(|_| Mutex::new(Vec::new())).collect())
}

#[allow(dead_code)]
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

/// MEASUREMENT-ONLY rule-#3 oracle (`GZIPPY_PREFAULT_ARENA=<MiB>`, default OFF):
/// pre-touch a per-worker arena of rpmalloc-backed pages at worker startup so the
/// chunk decode's first-touch page faults are PAID ONCE here, OFF the wall-critical
/// decode path, instead of cold on the hot path. Byte-transparent (allocates +
/// memsets + frees scratch; rpmalloc retains the freed pages warm in this thread's
/// cache). Used to falsify the page-warmth thesis: if faults drop toward rg's but
/// the MATCHED gz_null wall does not move, the excess faults were slack (advisor
/// arithmetic: fault-work ~33-55ms >> matched wall gap 17ms at depth-14).
pub fn prefault_worker_arena() {
    static MIB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let mib = *MIB.get_or_init(|| {
        std::env::var("GZIPPY_PREFAULT_ARENA")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    });
    if mib == 0 {
        return;
    }
    // Touch `mib` MiB of u16 + `mib` MiB of u8 scratch in 128 KiB spans (the
    // marker/data segment size) so every page is faulted+zeroed once, then drop
    // them so rpmalloc's per-thread cache keeps the warm pages for the next
    // real take_marker_segment / take_u8 on THIS worker.
    let spans = (mib * 1024) / 128; // 128 KiB spans
    let mut keep_u16: Vec<U16> = Vec::with_capacity(spans);
    let mut keep_u8: Vec<U8> = Vec::with_capacity(spans);
    for _ in 0..spans {
        let mut v16 = types::u16_with_capacity(MARKER_SEGMENT_ELEMENTS);
        // Touch every page of the allocation (write forces the fault now).
        v16.resize(MARKER_SEGMENT_ELEMENTS, 0);
        for i in (0..v16.len()).step_by(2048) {
            v16[i] = 1;
        }
        keep_u16.push(v16);
        let mut v8 = types::u8_with_capacity(128 * 1024);
        v8.resize(128 * 1024, 0);
        for i in (0..v8.len()).step_by(4096) {
            v8[i] = 1;
        }
        keep_u8.push(v8);
    }
    // Drop all at once → rpmalloc returns these warm pages to THIS thread's cache.
    drop(keep_u16);
    drop(keep_u8);
}

fn pool_index_for_take() -> usize {
    current_worker_pool_index().unwrap_or(0)
}

thread_local! {
    /// T1 cache-residency scope (set only by `drive_thin_t1_oracle`, on the
    /// single serial decode thread, via [`T1ResidentScope`]). When active it
    /// turns ON the manual buffer pool AND pins every reserve to the fixed
    /// resident cap — exactly the `GZIPPY_RESIDENT_OUTPUT_POOL` oracle that the
    /// gated discrimination measured (minor-faults drop toward igzip, monorepo
    /// wall 1.39→1.29 both arches; see former plans/T1-CACHE-RESIDENCY-RESULTS.md), but
    /// SCOPED to the T1 thread so the T>1 parallel workers (which never set it)
    /// keep their faithful per-chunk ratio-informed reserve + pooling behavior.
    static T1_RESIDENT_SCOPE: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that activates the T1 cache-residency scope for the current thread
/// and restores the prior value on drop (panic-safe).
pub struct T1ResidentScope {
    prev: bool,
}

impl T1ResidentScope {
    pub fn enter() -> Self {
        let prev = T1_RESIDENT_SCOPE.with(|c| c.replace(true));
        Self { prev }
    }
}

impl Drop for T1ResidentScope {
    fn drop(&mut self) {
        T1_RESIDENT_SCOPE.with(|c| c.set(self.prev));
    }
}

#[inline]
fn t1_resident_scope_active() -> bool {
    T1_RESIDENT_SCOPE.with(|c| c.get())
}

fn manual_buffer_pool_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *EN.get_or_init(|| {
        std::env::var_os("GZIPPY_MANUAL_BUFFER_POOL").is_some() || env_resident_output_pool()
    });
    env || t1_resident_scope_active()
}

/// RESIDENT-OUTPUT-POOL ORACLE flag (`GZIPPY_RESIDENT_OUTPUT_POOL=1`, default OFF,
/// byte-transparent, MEASUREMENT-ONLY). Turns on the manual LIFO pool AND pins every
/// chunk's upfront reserve to a single fixed size (see `compute_initial_reserve`), so
/// recycled output buffers share one capacity, are never realloc'd on reuse, and keep
/// their pages RESIDENT across chunks. Determination tool for the BEAT-IGZIP-T1
/// question: can a resident, reused output buffer reach igzip's T1 fault profile while
/// holding T>1 rapidgzip parity?
fn env_resident_output_pool() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("GZIPPY_RESIDENT_OUTPUT_POOL").is_some())
}

pub fn resident_output_pool_enabled() -> bool {
    env_resident_output_pool() || t1_resident_scope_active()
}

/// Take a `Vec<u8>` from the current worker's pool (or worker 0 if unbound).
/// Decode tasks run on pool workers and record the correct index; trial
/// decodes on the consumer thread bucket returns to worker 0's pool.
pub fn take_u8(min_capacity: usize) -> U8 {
    // Vendor FasterVector.hpp: rpmalloc per-thread cache only — no manual LIFO pool.
    // `GZIPPY_MANUAL_BUFFER_POOL=1` restores the legacy mutex pool for A/B.
    if manual_buffer_pool_enabled() {
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
    }
    types::u8_with_capacity(min_capacity)
}

/// Take a `Vec<u16>` from the current worker's pool.
#[allow(dead_code)]
pub fn take_u16(min_capacity: usize) -> U16 {
    if manual_buffer_pool_enabled() {
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
    }
    types::u16_with_capacity(min_capacity)
}

/// Return a `Vec<u8>` to the pool for `owner_worker` (recorded at take time).
pub fn return_u8_to_worker(owner_worker: usize, mut v: U8) {
    if v.capacity() == 0 {
        return;
    }
    if !manual_buffer_pool_enabled() {
        return; // drop — rpmalloc thread cache retains pages (vendor model)
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
#[allow(dead_code)]
pub fn return_u16_to_worker(owner_worker: usize, mut v: U16) {
    if v.capacity() == 0 {
        return;
    }
    if !manual_buffer_pool_enabled() {
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

// ── Per-worker pool of 128 KiB U16 marker segments (SegmentedU16) ─────────
const MARKER_SEGMENT_ELEMENTS: usize = 64 * 1024;
const MAX_POOLED_SEGMENTS: usize = 64;

fn marker_segment_pools() -> &'static [Mutex<Vec<U16>>] {
    static POOLS: OnceLock<Vec<Mutex<Vec<U16>>>> = OnceLock::new();
    POOLS.get_or_init(|| (0..MAX_WORKERS).map(|_| Mutex::new(Vec::new())).collect())
}

/// Take one 128 KiB rpmalloc-backed marker segment (`len == 0`).
pub fn take_marker_segment() -> U16 {
    if manual_buffer_pool_enabled() {
        let idx = pool_index_for_take();
        if let Ok(mut pool) = marker_segment_pools()[idx].lock() {
            if let Some(mut v) = pool.pop() {
                MARKER_SEG_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                v.clear();
                if v.capacity() < MARKER_SEGMENT_ELEMENTS {
                    v.reserve(MARKER_SEGMENT_ELEMENTS - v.capacity());
                }
                return v;
            }
        }
        MARKER_SEG_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    types::u16_with_capacity(MARKER_SEGMENT_ELEMENTS)
}

/// Return marker segments to the owner worker's pool.
pub fn return_marker_segments_to_worker(owner_worker: usize, segments: Vec<U16>) {
    if segments.is_empty() || !manual_buffer_pool_enabled() {
        return;
    }
    let idx = owner_worker.min(MAX_WORKERS - 1);
    if let Ok(mut pool) = marker_segment_pools()[idx].lock() {
        for mut v in segments {
            if v.capacity() == 0 {
                continue;
            }
            if pool.len() >= MAX_POOLED_SEGMENTS {
                break;
            }
            v.clear();
            pool.push(v);
        }
    }
}

pub static MARKER_SEG_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MARKER_SEG_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

// Unit tests intentionally omitted: the pool is a process-global LIFO
// that other tests (via `ChunkData::new`) concurrently take/return
// from, so any "round-trip" test that pops what it just pushed is
// inherently flaky. Integration coverage comes from every test that
// exercises `ChunkData::new` — 800+ tests run through this path.
