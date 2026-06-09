//! Debug-only memory accounting for the gzippy-native cache mandate.
//!
//! The mandate (`plans/gzippy-native-design-mandate.md`) requires a tiny
//! per-thread decode working set + shared read-only tables, with footprint
//! roughly flat as thread-count `T` rises. Process RSS is too coarse to see the
//! per-thread 128 KiB staging scratch (≈2 MiB at T16 against a ~300-400 MiB
//! backdrop), so this module is the PRIMARY instrument: direct in-process byte
//! accounting with a zero noise floor.
//!
//! Behind env `GZIPPY_MEM_STATS=1` (read ONCE into a `OnceLock<bool>`) it records
//! per-thread staging-pool high-water marks, pool alloc-vs-reuse counts, the
//! distinct worker-thread count, and the shared fixed-Huffman table footprint,
//! and prints a compact table to stderr at process exit.
//!
//! COUNTERS ONLY — no decode behavior changes. When the flag is unset every hook
//! is an inlined `if !enabled() { return; }`, so decoded bytes and timing are
//! identical with the flag on or off (proved by the DUAL-SHA gate). Counters are
//! relaxed atomics / thread-locals; the only lock (the per-thread registry) is
//! taken on the cold new-peak path.
//!
//! Positive control: env `GZIPPY_MEM_BALLAST_MIB=N` makes each worker thread
//! allocate + TOUCH (page-resident) `N` MiB, held for the thread's life, so the
//! RSS-vs-T regression can recover a known per-thread slope. The ballast is on
//! its own switch (independent of `GZIPPY_MEM_STATS`) so the RSS guard can drive
//! it without the accounting overhead. The ballast `Vec` is never read by the
//! decode, so it is byte-transparent.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use super::staged_bits::INPUT_STAGING_BYTES;

/// Per-thread native engine (`marker_inflate::Block`) resident working-set byte
/// breakdown. This is the REAL native per-thread state after the flip-in-place
/// fold (`gzip_chunk.rs` `BOOTSTRAP_BLOCK`); the staging-box hooks below are
/// DEAD on native (they instrumented Engine C, which the fold removed). Built
/// by `marker_inflate::Block::heap_bytes()`. Counters only.
#[derive(Clone, Copy, Default)]
pub struct BlockHeapBytes {
    pub total: usize,
    pub ring: usize,
    pub litlen_lut: usize,
    pub dist_cache: usize,
    pub misc_vecs: usize,
}

/// Whether in-process accounting is active. Read once.
#[inline]
pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("GZIPPY_MEM_STATS").is_some())
}

/// Per-thread positive-control ballast size in MiB. Read once. `0` = disabled.
#[inline]
fn ballast_mib() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("GZIPPY_MEM_BALLAST_MIB")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Retained-pool cap for the staging free-list (`staged_bits::return_staging_box`).
/// Default 4 == today's production behavior. `GZIPPY_STAGING_POOL_CAP=0` disables
/// pooling (every take is a fresh `Box::new`) — the D3 pooling-delta mechanism.
#[inline]
pub fn staging_pool_cap() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("GZIPPY_STAGING_POOL_CAP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4)
    })
}

static STAGING_ALLOC: AtomicU64 = AtomicU64::new(0);
static STAGING_REUSE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Default)]
struct ThreadStat {
    peak_live_boxes: usize,
}

fn registry() -> &'static Mutex<HashMap<ThreadId, ThreadStat>> {
    static R: OnceLock<Mutex<HashMap<ThreadId, ThreadStat>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Native-engine per-thread working-set registry: one entry per worker thread
/// that ever primed a `BOOTSTRAP_BLOCK`, holding that thread's resident
/// `Block` byte breakdown. This is the PRIMARY cache-mandate instrument after
/// the fold — it reports `threads > 0` on a native silesia decode (the staging
/// registry above does NOT, because Engine C is gone from native).
fn block_registry() -> &'static Mutex<HashMap<ThreadId, BlockHeapBytes>> {
    static R: OnceLock<Mutex<HashMap<ThreadId, BlockHeapBytes>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

thread_local! {
    /// Set once per thread after this thread has registered its Block size, so
    /// the cold registry lock is taken at most once per worker thread.
    static BLOCK_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

/// Record this worker thread's native `Block` resident working set. Called from
/// the native bootstrap (`gzip_chunk.rs` `marker_decode_step_vendor_block` /
/// `_marker_ring`) once the thread-local engine is primed. No-op unless
/// `GZIPPY_MEM_STATS` is set, and the registry lock is taken at most ONCE per
/// thread (subsequent calls early-return on the thread-local flag). Counters
/// only — does not touch decode state.
#[inline]
#[allow(dead_code)] // sole caller is under cfg(parallel_sm) (gzip_chunk.rs)
pub fn on_block_active(bytes: BlockHeapBytes) {
    // Ballast (positive control) is on its OWN switch, independent of
    // GZIPPY_MEM_STATS, so the RSS-vs-T guard can drive it without accounting
    // overhead. This is the NATIVE materialization point (the staging-box
    // `on_take` path is dead on native), so it must run before the enabled()
    // gate, once per worker thread. Byte-transparent: the ballast Vec is never
    // read by the decode.
    ensure_ballast();

    if !enabled() {
        return;
    }
    if BLOCK_REGISTERED.with(|c| c.get()) {
        return;
    }
    let id = std::thread::current().id();
    if let Ok(mut reg) = block_registry().lock() {
        reg.insert(id, bytes);
    }
    BLOCK_REGISTERED.with(|c| c.set(true));
}

thread_local! {
    static LIVE_BOXES: Cell<usize> = const { Cell::new(0) };
    static PEAK_BOXES: Cell<usize> = const { Cell::new(0) };
    static BALLAST: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

/// Called from `take_staging_box`. `reused` = popped from the per-thread pool
/// (vs freshly `Box::new`'d). No-op for accounting unless `GZIPPY_MEM_STATS` is
/// set; ballast (if configured) is materialized regardless.
#[inline]
pub fn on_take(reused: bool) {
    // Ballast is on its own switch so the RSS guard can drive it without the
    // accounting overhead. Both are debug-only and byte-transparent.
    ensure_ballast();

    if !enabled() {
        return;
    }
    if reused {
        STAGING_REUSE.fetch_add(1, Ordering::Relaxed);
    } else {
        STAGING_ALLOC.fetch_add(1, Ordering::Relaxed);
    }
    let live = LIVE_BOXES.with(|c| {
        let n = c.get() + 1;
        c.set(n);
        n
    });
    let new_peak = PEAK_BOXES.with(|c| {
        if live > c.get() {
            c.set(live);
            true
        } else {
            false
        }
    });
    if new_peak {
        // Cold path: only on a per-thread high-water increase.
        let id = std::thread::current().id();
        if let Ok(mut reg) = registry().lock() {
            reg.entry(id).or_default().peak_live_boxes = live;
        }
    }
}

/// Called from `return_staging_box`. No-op unless enabled.
#[inline]
pub fn on_return() {
    if !enabled() {
        return;
    }
    LIVE_BOXES.with(|c| c.set(c.get().saturating_sub(1)));
}

/// Allocate + touch the per-thread ballast on first use (positive control).
#[inline]
fn ensure_ballast() {
    let mib = ballast_mib();
    if mib == 0 {
        return;
    }
    BALLAST.with(|b| {
        let mut slot = b.borrow_mut();
        if slot.is_none() {
            let bytes = mib * 1024 * 1024;
            let mut v = vec![0u8; bytes];
            // Touch every 4 KiB page so the pages are RESIDENT (counts in RSS).
            let mut i = 0;
            while i < bytes {
                v[i] = 1;
                i += 4096;
            }
            *slot = Some(v);
        }
    });
}

/// Print the accounting table to stderr. Call once at process exit (before
/// `process::exit`, which skips destructors). No-op unless enabled.
pub fn report() {
    if !enabled() {
        return;
    }
    let alloc = STAGING_ALLOC.load(Ordering::Relaxed);
    let reuse = STAGING_REUSE.load(Ordering::Relaxed);
    let total = alloc + reuse;

    // ── PRIMARY: native per-thread engine (Block) working set ──────────────
    let (nthreads, per_thread, agg) = match block_registry().lock() {
        Ok(reg) => {
            let nthreads = reg.len();
            // Per-thread working set is uniform (every Block is sized
            // identically); report the max as the canonical per-thread figure.
            let mut per_thread = BlockHeapBytes::default();
            let mut agg = BlockHeapBytes::default();
            for b in reg.values() {
                if b.total > per_thread.total {
                    per_thread = *b;
                }
                agg.total += b.total;
                agg.ring += b.ring;
                agg.litlen_lut += b.litlen_lut;
                agg.dist_cache += b.dist_cache;
                agg.misc_vecs += b.misc_vecs;
            }
            (nthreads, per_thread, agg)
        }
        Err(_) => (0, BlockHeapBytes::default(), BlockHeapBytes::default()),
    };

    // Shared read-only fixed-Huffman tables — ONE copy via OnceLock.
    let (litlen, dist) = super::libdeflate_decode::get_fixed_tables();
    let shared_litlen = litlen.heap_bytes();
    let shared_dist = dist.heap_bytes();
    let shared_total = shared_litlen + shared_dist;

    let kib = |b: usize| b as f64 / 1024.0;
    eprintln!("=== GZIPPY_MEM_STATS (native per-thread working set) ===");
    eprintln!("worker threads observed (native Block): {nthreads}");
    if nthreads > 0 {
        eprintln!("PER-THREAD working set (persistent thread-local Block engine):");
        eprintln!(
            "  output_ring (u16 128KiB):           {:>9} bytes ({:>6.1} KiB)",
            per_thread.ring,
            kib(per_thread.ring)
        );
        eprintln!(
            "  lut_litlen (ISA-L lit/len LUT):     {:>9} bytes ({:>6.1} KiB)",
            per_thread.litlen_lut,
            kib(per_thread.litlen_lut)
        );
        eprintln!(
            "  dist_hc code_cache:                 {:>9} bytes ({:>6.1} KiB)",
            per_thread.dist_cache,
            kib(per_thread.dist_cache)
        );
        eprintln!(
            "  misc Vecs (literal_cl/backrefs):    {:>9} bytes ({:>6.1} KiB)",
            per_thread.misc_vecs,
            kib(per_thread.misc_vecs)
        );
        eprintln!(
            "  PER-THREAD TOTAL:                   {:>9} bytes ({:>6.1} KiB)",
            per_thread.total,
            kib(per_thread.total)
        );
        eprintln!(
            "AGGREGATE over {nthreads} threads:               {:>9} bytes ({:>6.1} KiB / {:.2} MiB)",
            agg.total,
            kib(agg.total),
            agg.total as f64 / (1024.0 * 1024.0)
        );
    } else {
        eprintln!("  (no native Block primed — wrong path? assert GZIPPY_DEBUG path=ParallelSM)");
    }
    eprintln!("SHARED read-only (ONE copy, all threads — fixed-Huffman tables):");
    eprintln!("  LitLenTable: {shared_litlen} bytes  DistTable: {shared_dist} bytes  total: {shared_total} bytes");

    // ── Secondary: staging-box pool (DEAD on native; live only on the ──────
    //    gzippy-isal two-phase / Engine-C path). Kept for that comparison.
    if total > 0 {
        eprintln!("--- staging-box pool (Engine-C / isal path only) ---");
        eprintln!(
            "  staging box size: {INPUT_STAGING_BYTES} bytes; pool cap: {}; alloc: {alloc}; reuse: {reuse} ({:.1}%)",
            staging_pool_cap(),
            100.0 * reuse as f64 / total as f64
        );
    }
    if ballast_mib() > 0 {
        eprintln!(
            "ballast (positive control):           {} MiB/thread touched-resident",
            ballast_mib()
        );
    }
    eprintln!("========================================================");
}
