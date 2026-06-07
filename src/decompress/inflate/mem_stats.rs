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

    let (threads, max_peak_boxes, sum_peak_boxes) = match registry().lock() {
        Ok(reg) => {
            let threads = reg.len();
            let max_peak = reg.values().map(|s| s.peak_live_boxes).max().unwrap_or(0);
            let sum_peak: usize = reg.values().map(|s| s.peak_live_boxes).sum();
            (threads, max_peak, sum_peak)
        }
        Err(_) => (0, 0, 0),
    };

    // Shared read-only fixed-Huffman tables — ONE copy via OnceLock.
    let (litlen, dist) = super::libdeflate_decode::get_fixed_tables();
    let litlen_bytes = litlen.heap_bytes();
    let dist_bytes = dist.heap_bytes();

    eprintln!("=== GZIPPY_MEM_STATS (in-process byte accounting) ===");
    eprintln!("worker threads observed:              {threads}");
    eprintln!(
        "staging box size:                     {INPUT_STAGING_BYTES} bytes ({} KiB)",
        INPUT_STAGING_BYTES / 1024
    );
    eprintln!(
        "staging pool cap (per thread):        {}",
        staging_pool_cap()
    );
    eprintln!("staging boxes allocated (Box::new):   {alloc}");
    eprintln!("staging boxes reused (from pool):     {reuse}");
    if total > 0 {
        eprintln!(
            "pool reuse rate:                      {:.1}% ({reuse}/{total})",
            100.0 * reuse as f64 / total as f64
        );
    }
    eprintln!(
        "per-thread peak live staging boxes (max): {max_peak_boxes} => {} bytes",
        max_peak_boxes * INPUT_STAGING_BYTES
    );
    eprintln!(
        "sum of per-thread peak live boxes:        {sum_peak_boxes} => {} bytes",
        sum_peak_boxes * INPUT_STAGING_BYTES
    );
    eprintln!("shared fixed Huffman tables (ONE copy via OnceLock):");
    eprintln!("  LitLenTable: {litlen_bytes} bytes");
    eprintln!("  DistTable:   {dist_bytes} bytes");
    eprintln!("  total:       {} bytes", litlen_bytes + dist_bytes);
    if ballast_mib() > 0 {
        eprintln!(
            "ballast (positive control):           {} MiB/thread touched-resident",
            ballast_mib()
        );
    }
    eprintln!("=====================================================");
}
