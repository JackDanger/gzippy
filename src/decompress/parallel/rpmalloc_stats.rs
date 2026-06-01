#![allow(dead_code)] // optional instrumentation; on-demand via GZIPPY_TIMELINE

//! rpmalloc_stats.rs — PER-REGION, PER-THREAD ALLOCATION snapshots for
//! Fulcrum's `alloc` / `decompose` allocation tier.
//!
//! This is the producer side of the "allocation fully describable in one run"
//! instrument. It snapshots the rpmalloc allocator's OWN counters (not a
//! `#[global_allocator]` shim, not LD_PRELOAD — both are BLIND to gzippy's
//! typed hot path, which calls `rpmalloc_sys::rpmalloc`/`rpaligned_alloc`
//! directly via [`super::rpmalloc_alloc::RpmallocAlloc`]). At region
//! boundaries we emit the DELTA of:
//!
//!   - **Per-thread span-cache REUSE** (the centerpiece, the rapidgzip-delta
//!     signal) from `rpmalloc_thread_statistics`: spans served `from_cache` /
//!     `from_reserved` / `from_global` (WARM, no fault) vs `map_calls` (COLD OS
//!     mmap → first-touch fault). The reuse rate is
//!     `warm / (warm + map_calls)`.
//!   - **Global huge-alloc CHURN** from `rpmalloc_global_statistics`:
//!     `mapped_total` / `unmapped_total` deltas. gzippy's ~12 MiB `ChunkData`
//!     and the 503 MiB output buffer are LARGER than rpmalloc's 2 MiB
//!     `LARGE_SIZE_LIMIT`, so they take the **huge** path, which munmaps on
//!     free and re-faults on every reuse — the suspected mechanism behind the
//!     ~2.5× minor-fault gap vs rapidgzip (which uses 128 KiB sub-buffers that
//!     stay span-cache-resident). The span-cache stats do NOT see the huge
//!     buffers; the global churn does.
//!   - **THP-backed fraction** of the process anon memory from
//!     `/proc/self/smaps_rollup` `AnonHugePages` — a first-touch-cost
//!     multiplier (a 2 MiB THP fault is one fault for 512 base pages; a
//!     sub-2-MiB buffer misses THP and pays 512× the minor faults).
//!
//! `ru_minflt` (the honest kernel fault number) is emitted separately by
//! [`super::residual`]; this module is the allocator-internal *cause* that
//! Fulcrum joins against that *effect* on the same `(tid, region)` cell.
//!
//! ## Byte-identical to production
//!
//! Every public fn early-returns when `trace_v2::is_enabled()` is false
//! (`GZIPPY_TIMELINE` unset — the production default): no FFI call, no
//! `/proc` read, no emit. The snapshot struct is plain integers on the stack.
//! No allocation, no behavioral branch in the decode itself.
//!
//! ## Statistics-feature gating
//!
//! rpmalloc only populates these counters when built with `ENABLE_STATISTICS`
//! (the `rpmalloc-stats` cargo feature → `rpmalloc-sys/statistics`). Without
//! it the FFI structs read back as zero. We still emit the (zero) instants and
//! a `stats_enabled` flag so Fulcrum can flag a run captured without the
//! feature rather than silently report "no allocation activity".

use super::trace_v2;

/// Whether rpmalloc was built with `ENABLE_STATISTICS`. Fulcrum reads this to
/// distinguish "no alloc activity" from "stats not compiled in".
#[cfg(all(feature = "arena-allocator", feature = "rpmalloc-stats"))]
pub const STATS_ENABLED: bool = true;
#[cfg(not(all(feature = "arena-allocator", feature = "rpmalloc-stats")))]
pub const STATS_ENABLED: bool = false;

/// A per-thread allocation snapshot. Cheap: a handful of integers, no alloc.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocSnapshot {
    /// Spans served WARM (no OS map): from thread cache + reserved + global.
    pub spans_warm: u64,
    /// Raw OS mmap calls (COLD; each backs fresh, first-touch-faulting pages).
    pub map_calls: u64,
    /// Bytes moved thread→global cache (eviction pressure).
    pub thread_to_global: u64,
    /// Bytes moved global→thread cache (cross-thread reuse).
    pub global_to_thread: u64,
    /// Peak outstanding spans across all size classes (working-set proxy).
    pub span_peak: u64,
    /// GLOBAL (process-wide, not per-thread): total bytes ever mapped from OS.
    pub mapped_total: u64,
    /// GLOBAL: total bytes ever unmapped to OS (huge-buffer munmap churn).
    pub unmapped_total: u64,
    /// GLOBAL: current huge-allocation bytes (the >2 MiB monolithic buffers).
    pub huge_alloc: u64,
    /// THP-backed anon KiB (smaps_rollup AnonHugePages). First-touch multiplier.
    pub anon_thp_kib: u64,
}

impl AllocSnapshot {
    /// Snapshot the current thread's allocator counters. Zeroed when tracing is
    /// disabled (so no FFI/`proc` cost in production) or when the platform/
    /// feature is unavailable.
    #[inline]
    pub fn capture() -> AllocSnapshot {
        if !trace_v2::is_enabled() {
            return AllocSnapshot::default();
        }
        capture_impl()
    }

    /// Emit the delta `self → end` for `region` as a `alloc.region` instant.
    /// No-op when tracing is disabled. GLOBAL fields (`mapped_total`,
    /// `unmapped_total`, `huge_alloc`, `anon_thp_kib`) are emitted as DELTAS
    /// too — interpret with care: they are process-wide, so on a per-thread
    /// region they over-count if other threads mapped concurrently. Fulcrum's
    /// alloc view treats per-thread span fields as exact and the global fields
    /// as run-level context (it never sums them across threads).
    pub fn emit_region_delta(&self, region: &str) {
        if !trace_v2::is_enabled() {
            return;
        }
        let end = capture_impl();
        let warm = end.spans_warm.saturating_sub(self.spans_warm);
        let maps = end.map_calls.saturating_sub(self.map_calls);
        let body = format!(
            r#""region":"{region}","spans_warm":{},"map_calls":{},"thread_to_global":{},"global_to_thread":{},"span_peak":{},"mapped_total_d":{},"unmapped_total_d":{},"huge_alloc":{},"anon_thp_kib":{},"stats_enabled":{}"#,
            warm,
            maps,
            end.thread_to_global.saturating_sub(self.thread_to_global),
            end.global_to_thread.saturating_sub(self.global_to_thread),
            end.span_peak,
            end.mapped_total.saturating_sub(self.mapped_total),
            end.unmapped_total.saturating_sub(self.unmapped_total),
            end.huge_alloc,
            end.anon_thp_kib,
            STATS_ENABLED,
        );
        trace_v2::emit_instant("alloc.region", &body, "t");
    }
}

/// RAII guard: capture on construction, emit the region delta on Drop. Declare
/// it AFTER the region's `trace_v2::SpanGuard` (and after the
/// [`super::residual::ResidualGuard`]) so it drops FIRST — the `alloc.region`
/// instant then lands while the region span is still open, so Fulcrum's
/// per-thread containment join attributes it to that region.
pub struct AllocGuard {
    start: AllocSnapshot,
    region: &'static str,
    armed: bool,
}

impl AllocGuard {
    #[inline]
    pub fn begin(region: &'static str) -> Self {
        let armed = trace_v2::is_enabled();
        AllocGuard {
            start: if armed {
                capture_impl()
            } else {
                AllocSnapshot::default()
            },
            region,
            armed,
        }
    }
}

impl Drop for AllocGuard {
    fn drop(&mut self) {
        if self.armed {
            self.start.emit_region_delta(self.region);
        }
    }
}

#[cfg(all(feature = "arena-allocator", feature = "rpmalloc-stats"))]
fn capture_impl() -> AllocSnapshot {
    // SAFETY: both rpmalloc_*_statistics fill a caller-provided zeroed POD
    // struct; the FFI is always bound. Values are nonzero only under
    // ENABLE_STATISTICS (the `rpmalloc-stats` feature, asserted by the cfg).
    let (ts, gs) = unsafe {
        let mut ts: rpmalloc_sys::rpmalloc_thread_statistics_t = std::mem::zeroed();
        rpmalloc_sys::rpmalloc_thread_statistics(&mut ts);
        let mut gs: rpmalloc_sys::rpmalloc_global_statistics_t = std::mem::zeroed();
        rpmalloc_sys::rpmalloc_global_statistics(&mut gs);
        (ts, gs)
    };
    let mut spans_warm: u64 = 0;
    let mut map_calls: u64 = 0;
    let mut span_peak: u64 = 0;
    for s in ts.span_use.iter() {
        spans_warm += (s.from_cache + s.from_reserved + s.from_global) as u64;
        map_calls += s.map_calls as u64;
        span_peak += s.peak as u64;
    }
    AllocSnapshot {
        spans_warm,
        map_calls,
        thread_to_global: ts.thread_to_global as u64,
        global_to_thread: ts.global_to_thread as u64,
        span_peak,
        mapped_total: gs.mapped_total as u64,
        unmapped_total: gs.unmapped_total as u64,
        huge_alloc: gs.huge_alloc as u64,
        anon_thp_kib: read_anon_thp_kib(),
    }
}

#[cfg(not(all(feature = "arena-allocator", feature = "rpmalloc-stats")))]
fn capture_impl() -> AllocSnapshot {
    // Stats feature off: still surface the THP fraction (free, no rpmalloc dep)
    // so a run captured without `rpmalloc-stats` is not entirely blind.
    AllocSnapshot {
        anon_thp_kib: read_anon_thp_kib(),
        ..AllocSnapshot::default()
    }
}

/// Read `AnonHugePages` (KiB) from `/proc/self/smaps_rollup`. THP-backed anon
/// memory; the delta over a region is the THP first-touch multiplier signal.
/// Returns 0 if unreadable. Linux-only.
#[cfg(target_os = "linux")]
fn read_anon_thp_kib() -> u64 {
    match std::fs::read_to_string("/proc/self/smaps_rollup") {
        Ok(s) => {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("AnonHugePages:") {
                    return rest
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(|t| t.parse::<u64>().ok())
                        .unwrap_or(0);
                }
            }
            0
        }
        Err(_) => 0,
    }
}

#[cfg(not(target_os = "linux"))]
fn read_anon_thp_kib() -> u64 {
    0
}
