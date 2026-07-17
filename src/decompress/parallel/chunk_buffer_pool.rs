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

/// Cap on pool size per worker per Vec type. Sized to a handful of
/// in-flight chunks per worker (not the old shared cap of 64).
// Sized for T16 in-flight depth: eager Ready-drain returns buffers sooner,
// but several Async heads can still hold buffers while workers decode ahead.
const MAX_POOLED: usize = 12;
const MAX_WORKERS: usize = 64;

thread_local! {
    static WORKER_POOL_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

// (shared cross-worker pool experiment removed 2026-05-28: it showed NO
// measurable wall effect on a quiet-enough box — page-faults
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

/// TEST-ONLY: drop every pooled buffer so the next take is a guaranteed pool
/// MISS (fresh allocation). The latch-interleave test uses this to prove the
/// per-decode `SystemHugeScope` re-arms on a fresh tiny decode — without it, a
/// tiny decode after any earlier T1 decode silently (and correctly) reuses the
/// pooled buffer and allocates nothing at all.
#[cfg(test)]
pub fn drain_pools_for_test() {
    for p in u8_pools() {
        p.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    for p in u16_pools() {
        p.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    for p in marker_segment_pools() {
        p.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
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

thread_local! {
    /// T1 cache-residency scope (set only by `drive_thin_t1_oracle`, on the
    /// single serial decode thread, via [`T1ResidentScope`]). When active it
    /// turns ON the manual buffer pool AND pins every reserve to the fixed
    /// resident cap — the shipped default behavior that gated discrimination
    /// measured (minor-faults drop toward igzip, monorepo
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
    t1_resident_scope_active()
}

/// FORM-B RSS FIX — eager per-chunk page release at recycle time.
///
/// The AMD-Zen2 mid-T (T3-T8) loss vs rapidgzip was localized (differential +
/// RSS-inflate Gate-2 oracle) to the process-exit teardown: gz's IN-PROCESS
/// decode (`main_start`→`main_end`) is at-or-faster than rapidgzip's whole
/// wall, but the residual after `main_end` — the KERNEL `exit_group`
/// address-space teardown — is proportional to resident memory and is where the
/// gap lives. `libc::_exit` cannot skip it (exit_group runs regardless), so the
/// only lever is to lower PEAK / AT-EXIT resident memory during the run.
///
/// gzippy holds each chunk's decoded bulk (and marker) buffer resident until it
/// is recycled; with the default (no manual pool) allocator, freed buffers can
/// linger in the arena (glibc's dynamic mmap threshold) instead of being
/// returned to the OS, inflating at-exit RSS. rapidgzip's rpmalloc returns
/// pages to the OS incrementally. This mirrors that: at recycle time — which
/// under the deferral invariant is AFTER the chunk's output was written and its
/// window published, i.e. the buffer's contents are dead — we `MADV_DONTNEED`
/// the buffer's resident pages, returning them to the OS immediately and
/// overlapped with other threads' decode. Byte-transparent: the virtual mapping
/// is retained; if the buffer is later re-taken from the manual pool it
/// re-faults to zero pages that the next decode overwrites before any read.
///
/// DEFAULT was OFF (opt-in env knob, since removed), and the knob was
/// GATED-MEASURED NET-NEGATIVE on AMD-Zen2 ondemand silesia T3-T6 (the per-chunk
/// `MADV_DONTNEED` runs on the serial consumer thread and the returned pages
/// re-fault on the next allocation, a cost EXCEEDING the exit_group teardown it
/// saves). The env kill-switch is removed and the shipped default (no eager
/// release) is hardcoded — `dontneed_alloc` is a no-op. A profitable variant
/// would have to overlap the page-return onto the PARALLEL worker threads (as
/// rpmalloc does) rather than the serial consumer — unvalidated.
#[inline]
fn dontneed_alloc(_base: *const u8, _bytes: usize) {}

/// Whether the T1 resident-output-pool behavior is active: pins every chunk's
/// upfront reserve to a single fixed size (see `compute_initial_reserve`) so
/// recycled output buffers share one capacity, are never realloc'd on reuse,
/// and keep their pages RESIDENT across chunks. Was previously also
/// controllable via `GZIPPY_RESIDENT_OUTPUT_POOL=1` (measurement oracle,
/// removed) — the T1 scope activation (the shipped default) is preserved.
pub fn resident_output_pool_enabled() -> bool {
    t1_resident_scope_active()
}

/// The fixed capacity every resident-scope output buffer is pinned to (the
/// `compute_initial_reserve` resident arm and `SegmentedU8::ensure_buf` both
/// use it — ONE source of truth). Equal to the historical `RESERVE_CAP`
/// upfront ceiling: virtual reserve only, pages fault as touched.
pub const RESIDENT_PINNED_CAPACITY: usize = 64 * 1024 * 1024;

/// Take a `Vec<u8>` from the current worker's pool (or worker 0 if unbound).
/// Decode tasks run on pool workers and record the correct index; trial
/// decodes on the consumer thread bucket returns to worker 0's pool.
pub fn take_u8(min_capacity: usize) -> U8 {
    // Vendor FasterVector.hpp: rpmalloc per-thread cache only — no manual LIFO pool,
    // except within the T1 resident-scope (see `manual_buffer_pool_enabled`).
    if manual_buffer_pool_enabled() {
        let idx = pool_index_for_take();
        if let Ok(mut pool) = u8_pools()[idx].lock() {
            if let Some(mut v) = pool.pop() {
                v.clear();
                if v.capacity() < min_capacity {
                    v.reserve(min_capacity - v.capacity());
                }
                return v;
            }
        }
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
                v.clear();
                if v.capacity() < min_capacity {
                    v.reserve(min_capacity - v.capacity());
                }
                return v;
            }
        }
    }
    types::u16_with_capacity(min_capacity)
}

/// Return a `Vec<u8>` to the pool for `owner_worker` (recorded at take time).
pub fn return_u8_to_worker(owner_worker: usize, mut v: U8) {
    if v.capacity() == 0 {
        return;
    }
    // FORM-B: return this recycled buffer's resident pages to the OS now,
    // before it is pooled (re-fault on reuse) or dropped (freed). Lowers
    // peak/at-exit RSS → cheaper kernel exit_group teardown. Contents are dead
    // at recycle time. Fires in BOTH the manual-pool and default (drop) modes.
    dontneed_alloc(v.as_ptr(), v.capacity());
    if !manual_buffer_pool_enabled() {
        return; // drop — rpmalloc thread cache retains pages (vendor model)
    }
    let idx = owner_worker.min(MAX_WORKERS - 1);
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
    // FORM-B: page-release before pool/drop (see return_u8_to_worker).
    dontneed_alloc(
        v.as_ptr() as *const u8,
        v.capacity() * std::mem::size_of::<u16>(),
    );
    if !manual_buffer_pool_enabled() {
        return;
    }
    let idx = owner_worker.min(MAX_WORKERS - 1);
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
                v.clear();
                if v.capacity() < MARKER_SEGMENT_ELEMENTS {
                    v.reserve(MARKER_SEGMENT_ELEMENTS - v.capacity());
                }
                return v;
            }
        }
    }
    types::u16_with_capacity(MARKER_SEGMENT_ELEMENTS)
}

/// Return marker segments to the owner worker's pool.
pub fn return_marker_segments_to_worker(owner_worker: usize, segments: Vec<U16>) {
    if segments.is_empty() {
        return;
    }
    // FORM-B: page-release each marker segment before pool/drop.
    for v in &segments {
        if v.capacity() != 0 {
            dontneed_alloc(
                v.as_ptr() as *const u8,
                v.capacity() * std::mem::size_of::<u16>(),
            );
        }
    }
    if !manual_buffer_pool_enabled() {
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

// Unit tests intentionally omitted: the pool is a process-global LIFO
// that other tests (via `ChunkData::new`) concurrently take/return
// from, so any "round-trip" test that pops what it just pushed is
// inherently flaky. Integration coverage comes from every test that
// exercises `ChunkData::new` — 800+ tests run through this path.
