#![cfg(parallel_sm)]
#![allow(dead_code)] // optional instrumentation; on-demand via GZIPPY_MEMLIFE

//! memlife.rs — per-buffer memory-LIFECYCLE byte attribution for the parallel
//! single-member decode path, cross-tool-comparable with rapidgzip.
//!
//! ## What this is
//!
//! `fulcrum memlife` needs a per-COMPONENT (named buffer + code file:line)
//! breakdown of memory traffic — bytes ALLOCATED / WRITTEN / READ / COPIED and
//! first-touch faults — so the gzippy-vs-rapidgzip excess can be localized to a
//! specific buffer rather than a global "gzippy has more memory pressure".
//!
//! Unlike `trace_v2` (a TIME-window Chrome-trace, attributed by a per-thread
//! timestamp join whose purity drops below 1.0 at T8 because workers' spans
//! overlap), this module records EXACT byte totals as process-global atomics
//! keyed by a fixed component enum. Byte counts are frequency-independent and
//! commutative, so there is no attribution ambiguity and the per-component sum
//! is exactly closeable against the allocator total / getrusage — the closure
//! check that the validation mandate requires.
//!
//! ## Activation / cost when OFF
//!
//! Inert unless `GZIPPY_MEMLIFE` is set (to a path or `1`). Every record call
//! is gated by a single relaxed-atomic load of [`ENABLED`]; with the env unset
//! the gate is false and the hot path does one predict-taken branch and
//! returns. Output is byte-identical with the knob off (the recorder never
//! touches decoded bytes) — verified by the sha check in the bench.
//!
//! ## The components (named buffer ↔ code site ↔ rapidgzip counterpart)
//!
//! | Component             | gzippy buffer / site                              | rapidgzip counterpart |
//! |-----------------------|---------------------------------------------------|-----------------------|
//! | `DataWithMarkers`     | `ChunkData::data_with_markers` (Vec<u16>, 2× width)| `DecodedData::dataWithMarkers` (MarkerVector) |
//! | `Data`                | `ChunkData::data` (U8, clean bulk)                | `DecodedData::data` (VectorView) |
//! | `Narrowed`            | `ChunkData::narrowed_len` prefix of marker segs   | IN-PLACE u8 view (no separate `narrowed` vec) |
//! | `Window`              | 32 KiB tail windows + WindowMap storage           | `WindowMap` (compressed) |
//! | `OutputWrite`         | bytes streamed to the writer                      | `toIoVec` writev gather |
//!
//! For each component we record, where the code touches it:
//!  - `alloc(bytes, path)`   — a heap allocation backing the buffer + its path
//!    (rpmalloc-span / rpmalloc-huge / glibc / pool-hit). Emitted from
//!    `rpmalloc_alloc::RpmallocAlloc` (size-classed) and the std-Vec pool.
//!  - `written(bytes)`       — bytes the decoder/narrower STORED into the buffer.
//!  - `read(bytes)`          — bytes a later pass LOADED from the buffer
//!    (apply_window reads data_with_markers; the writer reads narrowed/data).
//!  - `copied(bytes)`        — bytes moved buffer→buffer by a memcpy/memmove
//!    (append_markered, clean_unmarked_data shift, narrow loop).
//!
//! `Narrowed` tracks in-place resolve: `narrow_markers_in_place` reinterprets the
//! marker segment prefix as u8 (rapidgzip `MapMarkers` shape); no third buffer.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// The fixed set of memory-lifecycle components. Order is the table order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    DataWithMarkers = 0,
    Data = 1,
    Narrowed = 2,
    Window = 3,
    OutputWrite = 4,
}

pub const COMPONENT_COUNT: usize = 5;

impl Component {
    pub const fn name(self) -> &'static str {
        match self {
            Component::DataWithMarkers => "data_with_markers",
            Component::Data => "data",
            Component::Narrowed => "narrowed",
            Component::Window => "window",
            Component::OutputWrite => "output_write",
        }
    }
    /// The code site (file:line-ish) + rapidgzip counterpart, for the report.
    pub const fn site(self) -> &'static str {
        match self {
            Component::DataWithMarkers => {
                "chunk_data.rs:150 (Vec<u16>); rg DecodedData::dataWithMarkers"
            }
            Component::Data => "chunk_data.rs:154 (U8); rg DecodedData::data",
            Component::Narrowed => {
                "chunk_data.rs:166 + chunk_fetcher.rs:2375/2387; rg IN-PLACE applyWindow (no buffer)"
            }
            Component::Window => "window_map.rs / get_last_window; rg WindowMap",
            Component::OutputWrite => "consumer write path; rg toIoVec writev",
        }
    }
}

/// The allocator path an allocation took — so cross-tool alloc traffic is
/// comparable by class, not just total bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocPath {
    /// rpmalloc small/medium span-cache (< huge threshold). Warm-reused.
    RpmallocSpan = 0,
    /// rpmalloc huge alloc (>= ~3.94 MiB): mmap-on-alloc, munmap-on-free →
    /// re-faults on reuse (the first-touch-fault driver).
    RpmallocHuge = 1,
    /// std/glibc allocator (the std-Vec pool path for data_with_markers).
    Glibc = 2,
    /// Served from a warm pool (no new mapping, no first-touch fault).
    PoolHit = 3,
}

pub const ALLOC_PATH_COUNT: usize = 4;

impl AllocPath {
    pub const fn name(self) -> &'static str {
        match self {
            AllocPath::RpmallocSpan => "rpmalloc-span",
            AllocPath::RpmallocHuge => "rpmalloc-huge",
            AllocPath::Glibc => "glibc",
            AllocPath::PoolHit => "pool-hit",
        }
    }
    /// rpmalloc huge-alloc threshold (the munmap-on-free boundary). Matches
    /// rpmalloc's span/huge boundary used in `rpmalloc_alloc.rs`.
    pub const HUGE_THRESHOLD: usize = 3 * 1024 * 1024;
}

// ── The counter storage ───────────────────────────────────────────────────
//
// One contiguous block of atomics. `[component][event]` for the byte+count
// traffic; a separate `[component][alloc_path]` matrix for the alloc-path
// split. All u64 atomics; relaxed ordering (we want totals at the end, not a
// happens-before edge per event).

/// Per-component byte/count events.
const EV_ALLOC_BYTES: usize = 0;
const EV_ALLOC_COUNT: usize = 1;
const EV_WRITTEN: usize = 2;
const EV_READ: usize = 3;
const EV_COPIED: usize = 4;
const EV_FREE_BYTES: usize = 5;
const EV_COUNT: usize = 6;

struct Counters {
    /// [component][event]
    ev: [[AtomicU64; EV_COUNT]; COMPONENT_COUNT],
    /// [component][alloc_path] → bytes allocated via that path
    alloc_path_bytes: [[AtomicU64; ALLOC_PATH_COUNT]; COMPONENT_COUNT],
    alloc_path_count: [[AtomicU64; ALLOC_PATH_COUNT]; COMPONENT_COUNT],
    /// Independently-measured allocator total (every alloc through
    /// `RpmallocAlloc`, irrespective of component tagging) — the CLOSURE
    /// anchor. If Σ component alloc bytes ≪ this, components are missing.
    allocator_total_bytes: AtomicU64,
    allocator_total_count: AtomicU64,
    /// Decoded output bytes (the per-MB normalizer). Set once at the end.
    decoded_bytes: AtomicU64,
    /// Worker count for the run (T) — recorded so the report knows T1 vs T8.
    workers: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        // const-init helpers (atomics aren't Copy).
        #[allow(clippy::declare_interior_mutable_const)]
        const Z: AtomicU64 = AtomicU64::new(0);
        #[allow(clippy::declare_interior_mutable_const)]
        const EVROW: [AtomicU64; EV_COUNT] = [Z, Z, Z, Z, Z, Z];
        #[allow(clippy::declare_interior_mutable_const)]
        const APROW: [AtomicU64; ALLOC_PATH_COUNT] = [Z, Z, Z, Z];
        Counters {
            ev: [EVROW, EVROW, EVROW, EVROW, EVROW],
            alloc_path_bytes: [APROW, APROW, APROW, APROW, APROW],
            alloc_path_count: [APROW, APROW, APROW, APROW, APROW],
            allocator_total_bytes: Z,
            allocator_total_count: Z,
            decoded_bytes: Z,
            workers: Z,
        }
    }
}

static COUNTERS: Counters = Counters::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: OnceLock<bool> = OnceLock::new();

/// One-time env read. `GZIPPY_MEMLIFE` set (to a path or anything) turns the
/// recorder ON. Cheap after the first call (relaxed bool load).
#[inline]
pub fn is_enabled() -> bool {
    if ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    // First-call slow path: read env once, latch.
    *INIT.get_or_init(|| {
        let on = std::env::var_os("GZIPPY_MEMLIFE").is_some();
        ENABLED.store(on, Ordering::Relaxed);
        on
    })
}

#[inline]
fn ev(c: Component, e: usize) -> &'static AtomicU64 {
    &COUNTERS.ev[c as usize][e]
}

// ── Record API (all gated; no-op when OFF) ──────────────────────────────────

/// Record bytes WRITTEN (stored) into a component's buffer.
#[inline]
pub fn written(c: Component, bytes: usize) {
    if !is_enabled() {
        return;
    }
    ev(c, EV_WRITTEN).fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Record bytes READ (loaded) from a component's buffer by a later pass.
#[inline]
pub fn read(c: Component, bytes: usize) {
    if !is_enabled() {
        return;
    }
    ev(c, EV_READ).fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Record bytes COPIED buffer→buffer (a memcpy/memmove). This double-counts as
/// neither pure read nor pure write because a copy is BOTH a load and a store
/// of the same bytes; we keep it in its own column so the report can show "this
/// many bytes crossed the bus twice for a copy gzippy does and rapidgzip
/// doesn't".
#[inline]
pub fn copied(c: Component, bytes: usize) {
    if !is_enabled() {
        return;
    }
    ev(c, EV_COPIED).fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Record an allocation backing a component, classified by [`AllocPath`].
/// Pass the TRUE backing-byte size (capacity reserved), not the logical len.
#[inline]
pub fn alloc(c: Component, bytes: usize, path: AllocPath) {
    if !is_enabled() {
        return;
    }
    ev(c, EV_ALLOC_BYTES).fetch_add(bytes as u64, Ordering::Relaxed);
    ev(c, EV_ALLOC_COUNT).fetch_add(1, Ordering::Relaxed);
    COUNTERS.alloc_path_bytes[c as usize][path as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS.alloc_path_count[c as usize][path as usize].fetch_add(1, Ordering::Relaxed);
}

/// Record a free of a component-backing allocation (for peak/lifetime).
#[inline]
pub fn freed(c: Component, bytes: usize) {
    if !is_enabled() {
        return;
    }
    ev(c, EV_FREE_BYTES).fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Independent allocator-total tap: EVERY allocation through `RpmallocAlloc`,
/// component-agnostic. This is the closure anchor (Σ tagged-component allocs
/// must ≈ this). Called from the allocator's `allocate`.
#[inline]
pub fn allocator_total(bytes: usize) {
    if !is_enabled() {
        return;
    }
    COUNTERS
        .allocator_total_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS
        .allocator_total_count
        .fetch_add(1, Ordering::Relaxed);
}

/// Set the decoded-output byte total (the per-MB normalizer) + worker count.
pub fn set_run_totals(decoded_bytes: usize, workers: usize) {
    if !is_enabled() {
        return;
    }
    COUNTERS
        .decoded_bytes
        .store(decoded_bytes as u64, Ordering::Relaxed);
    COUNTERS.workers.store(workers as u64, Ordering::Relaxed);
}

// ── Dump ─────────────────────────────────────────────────────────────────

fn load(a: &AtomicU64) -> u64 {
    a.load(Ordering::Relaxed)
}

/// Serialize the collected counters to the JSON the `fulcrum memlife` view
/// reads. Written to the path in `GZIPPY_MEMLIFE` (or `/tmp/gzippy-memlife.json`
/// if it's set to `1`). Called once at the end of the parallel run, alongside
/// `trace_v2::flush_all`.
///
/// Also appends getrusage(RUSAGE_SELF) minflt/majflt + maxrss as the
/// independent fault/RSS anchors for the closure check.
pub fn dump() {
    if !is_enabled() {
        return;
    }
    let path = match std::env::var("GZIPPY_MEMLIFE") {
        Ok(p) if p != "1" && !p.is_empty() => p,
        _ => "/tmp/gzippy-memlife.json".to_string(),
    };

    let (minflt, majflt, maxrss_kb) = rusage_self();

    let mut s = String::with_capacity(4096);
    s.push_str("{\n");
    s.push_str("  \"tool\": \"gzippy\",\n");
    s.push_str(&format!(
        "  \"decoded_bytes\": {},\n",
        load(&COUNTERS.decoded_bytes)
    ));
    s.push_str(&format!("  \"workers\": {},\n", load(&COUNTERS.workers)));
    s.push_str(&format!(
        "  \"allocator_total_bytes\": {},\n",
        load(&COUNTERS.allocator_total_bytes)
    ));
    s.push_str(&format!(
        "  \"allocator_total_count\": {},\n",
        load(&COUNTERS.allocator_total_count)
    ));
    s.push_str(&format!("  \"rusage_minflt\": {minflt},\n"));
    s.push_str(&format!("  \"rusage_majflt\": {majflt},\n"));
    s.push_str(&format!("  \"rusage_maxrss_kb\": {maxrss_kb},\n"));
    s.push_str("  \"components\": [\n");

    let comps = [
        Component::DataWithMarkers,
        Component::Data,
        Component::Narrowed,
        Component::Window,
        Component::OutputWrite,
    ];
    let paths = [
        AllocPath::RpmallocSpan,
        AllocPath::RpmallocHuge,
        AllocPath::Glibc,
        AllocPath::PoolHit,
    ];
    for (ci, c) in comps.iter().enumerate() {
        let i = *c as usize;
        s.push_str("    {\n");
        s.push_str(&format!("      \"component\": \"{}\",\n", c.name()));
        s.push_str(&format!("      \"site\": \"{}\",\n", c.site()));
        s.push_str(&format!(
            "      \"alloc_bytes\": {},\n",
            load(&COUNTERS.ev[i][EV_ALLOC_BYTES])
        ));
        s.push_str(&format!(
            "      \"alloc_count\": {},\n",
            load(&COUNTERS.ev[i][EV_ALLOC_COUNT])
        ));
        s.push_str(&format!(
            "      \"written_bytes\": {},\n",
            load(&COUNTERS.ev[i][EV_WRITTEN])
        ));
        s.push_str(&format!(
            "      \"read_bytes\": {},\n",
            load(&COUNTERS.ev[i][EV_READ])
        ));
        s.push_str(&format!(
            "      \"copied_bytes\": {},\n",
            load(&COUNTERS.ev[i][EV_COPIED])
        ));
        s.push_str(&format!(
            "      \"freed_bytes\": {},\n",
            load(&COUNTERS.ev[i][EV_FREE_BYTES])
        ));
        s.push_str("      \"alloc_paths\": {");
        let mut first = true;
        for p in &paths {
            let b = load(&COUNTERS.alloc_path_bytes[i][*p as usize]);
            let n = load(&COUNTERS.alloc_path_count[i][*p as usize]);
            if b == 0 && n == 0 {
                continue;
            }
            if !first {
                s.push_str(", ");
            }
            first = false;
            s.push_str(&format!("\"{}\": [{}, {}]", p.name(), b, n));
        }
        s.push('}');
        s.push('\n');
        if ci + 1 < comps.len() {
            s.push_str("    },\n");
        } else {
            s.push_str("    }\n");
        }
    }
    s.push_str("  ]\n}\n");

    if let Err(e) = std::fs::write(&path, &s) {
        eprintln!("[memlife] failed to write {path}: {e}");
    } else {
        eprintln!("[memlife] wrote {path} ({} bytes)", s.len());
    }
}

/// getrusage(RUSAGE_SELF) → (minflt, majflt, maxrss_kb). Linux maxrss is KiB.
#[cfg(unix)]
fn rusage_self() -> (u64, u64, u64) {
    // SAFETY: zeroed rusage filled by the kernel; POD struct.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            (
                ru.ru_minflt as u64,
                ru.ru_majflt as u64,
                ru.ru_maxrss as u64,
            )
        } else {
            (0, 0, 0)
        }
    }
}

#[cfg(not(unix))]
fn rusage_self() -> (u64, u64, u64) {
    (0, 0, 0)
}

/// Classify an allocation size into its rpmalloc path. (`pool-hit` is recorded
/// separately by the pool, not here — this is only for the allocator tap.)
#[inline]
pub fn classify_rpmalloc(bytes: usize) -> AllocPath {
    if bytes >= AllocPath::HUGE_THRESHOLD {
        AllocPath::RpmallocHuge
    } else {
        AllocPath::RpmallocSpan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: these tests force-enable the recorder by setting the env BEFORE the
    // OnceLock latches. Because ENABLED is process-global and the latch is
    // one-shot, we run the positive-control logic in a single test to avoid
    // cross-test latch races.
    #[test]
    fn positive_control_records_and_closes() {
        std::env::set_var("GZIPPY_MEMLIFE", "1");
        // Re-arm: the OnceLock may already be latched from another test in the
        // same process; force ENABLED true directly for a deterministic check.
        ENABLED.store(true, Ordering::Relaxed);

        // Inject a known allocation + write + copy into Narrowed.
        let before_alloc = load(&COUNTERS.ev[Component::Narrowed as usize][EV_ALLOC_BYTES]);
        let before_written = load(&COUNTERS.ev[Component::Narrowed as usize][EV_WRITTEN]);
        let before_copied = load(&COUNTERS.ev[Component::Narrowed as usize][EV_COPIED]);

        alloc(Component::Narrowed, 4096, AllocPath::RpmallocSpan);
        written(Component::Narrowed, 4096);
        copied(Component::Narrowed, 4096);

        assert_eq!(
            load(&COUNTERS.ev[Component::Narrowed as usize][EV_ALLOC_BYTES]) - before_alloc,
            4096
        );
        assert_eq!(
            load(&COUNTERS.ev[Component::Narrowed as usize][EV_WRITTEN]) - before_written,
            4096
        );
        assert_eq!(
            load(&COUNTERS.ev[Component::Narrowed as usize][EV_COPIED]) - before_copied,
            4096
        );
        // alloc-path split caught it.
        assert!(
            load(
                &COUNTERS.alloc_path_bytes[Component::Narrowed as usize]
                    [AllocPath::RpmallocSpan as usize]
            ) >= 4096
        );
    }

    #[test]
    fn classify_threshold() {
        assert_eq!(classify_rpmalloc(1024), AllocPath::RpmallocSpan);
        assert_eq!(classify_rpmalloc(4 * 1024 * 1024), AllocPath::RpmallocHuge);
    }
}
