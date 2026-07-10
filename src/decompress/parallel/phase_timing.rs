//! Diagnostic phase-breakdown emitter — feature `phase-timing` (OFF by
//! default; ZERO effect on the production build/bytes).
//!
//! Every public item here compiles to a true no-op under
//! `#[cfg(not(feature = "phase-timing"))]`: no atomics, no `Instant::now()`,
//! no env reads, no allocation, no output. The feature exists to replace a
//! series of hand-rolled, throwaway one-off timing hacks with ONE
//! deterministic, self-validating instrument: it turns the previously
//! hand-measured "consumer's `future_recv` wait dominates the wall" claim
//! into tool output that `fulcrum phasebreak` can conservation-check and
//! report on (Gate-0). NO rapidgzip counterpart — this is gzippy-only
//! measurement scaffolding, mirrored in wiring style off `coz_probe.rs`
//! (same "feature-gated shim, no-op without it" shape) but a flat JSON-line
//! emitter instead of a Coz probe.
//!
//! ## Call sites (all in `chunk_fetcher.rs`, all `#[cfg(feature =
//! "phase-timing")]`-gated at the call site)
//!
//! 1. `drive_impl`: brackets the whole per-decode span (wall + this-thread
//!    CPU time via `CLOCK_THREAD_CPUTIME_ID`) and the finalize tail, then
//!    calls [`emit`] once per decode.
//! 2. `consumer_loop`'s `block_fetcher.get_with_prefetch(...)` call (the
//!    on-demand/cache-miss branch) → RAII guard accumulating into
//!    [`DECODE_WAIT_NS`]. **Deviation from the original brief** (noted per
//!    this project's honesty rule, cf. fulcrum's `abmeasure.rs` module doc):
//!    dogfooding against real corpora showed Gate-0 conservation REFUSING
//!    every run — `consumer_cpu_us` was ~2% of `consumer_wall_us` but the
//!    four originally-named phases summed to only ~30%. Root cause:
//!    `block_fetcher.try_take_prefetched_pumping(...)` (the prefetch-HIT-
//!    but-still-in-flight sibling branch, a few lines above the miss-path
//!    call the brief named) was uninstrumented — its OWN doc comment names
//!    it "THE wait that gates the in-order wall (~97% of it)". It is now
//!    ALSO wrapped and folded into the same `DECODE_WAIT_NS` accumulator
//!    (both branches are "consumer waiting for the next chunk to become
//!    ready" — hit vs miss on the prefetch cache). With this, Gate-0
//!    conservation holds on silesia at T2/T4 and on a stored-heavy corpus.
//! 3. `consumer_loop`'s `block_finder.get(...)` call → RAII guard
//!    accumulating into [`BLOCKFIND_NS`].
//! 4. `consumer_loop`'s per-iteration `harvest_ready_postprocess` (first
//!    occurrence, top of loop) → [`add_iter`] increments [`ITERS`].
//! 5. `recv_post_process_blocking` — RAII guard over the WHOLE function body
//!    accumulating into [`FUTURE_RECV_NS`]. This is the consumer's real
//!    serial marker-resolution wait — the phase this instrument exists to
//!    surface deterministically.
//! 6. `drain_one_pending` — RAII guard over the whole function body
//!    accumulating into [`DRAIN_NS`].
//!
//! The atomics are reset once per `drive_impl` call (fresh snapshot per
//! decode) so a multi-decode test harness process does not accumulate
//! across calls.
//!
//! Without the `phase-timing` feature, every call site in `chunk_fetcher.rs`
//! that would reference this module is itself `#[cfg]`-gated away, so the
//! whole module is legitimately unreferenced dead code from the default
//! build's point of view — same shape as `coz_probe.rs`'s `noop` module.
#![cfg_attr(not(feature = "phase-timing"), allow(dead_code))]

#[cfg(feature = "phase-timing")]
use std::sync::atomic::{AtomicU64, Ordering};

/// Consumer time spent blocked inside `BlockFetcher::get_with_prefetch` on
/// the on-demand (cache-miss) decode path.
#[cfg(feature = "phase-timing")]
pub static DECODE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// Consumer time spent blocked inside `recv_post_process_blocking` — the
/// serial marker-resolution wait (rg `waitForReplacedMarkers`'s blocking
/// tail). This is the phase this instrument exists to surface.
#[cfg(feature = "phase-timing")]
pub static FUTURE_RECV_NS: AtomicU64 = AtomicU64::new(0);
/// Time spent inside `drain_one_pending` (head-of-FIFO write + CRC + the
/// `recv_post_process_blocking` wait it may call into — that inner span is
/// double-counted into both `DRAIN_NS` and `FUTURE_RECV_NS` by design; the
/// conservation check in `fulcrum phasebreak` accounts for this as overlap,
/// not double-counted disjoint time).
#[cfg(feature = "phase-timing")]
pub static DRAIN_NS: AtomicU64 = AtomicU64::new(0);
/// Time spent inside `block_finder.get(...)` on the consumer's loop.
#[cfg(feature = "phase-timing")]
pub static BLOCKFIND_NS: AtomicU64 = AtomicU64::new(0);
/// Number of `consumer_loop` iterations (top-of-loop harvest calls).
#[cfg(feature = "phase-timing")]
pub static ITERS: AtomicU64 = AtomicU64::new(0);

// ── STEP-1 wasted-work counters (2026-07-09, storedheavy sub-cause brief) ──
// Instrument EVERY call into `run_decode_task` (chunk_fetcher.rs) — the sole
// choke point through which all worker decode invocations pass (both the
// window-PRESENT confirmed/exact path via `decode_chunk_with_until_exact`
// and the window-ABSENT speculative path via `speculative_decode_find_
// boundary`). These answer: do workers decode MORE total bytes / MORE total
// invocations than the chunk count and output size would require (wasted
// speculative / re-decode work), and what fraction of invocations run
// window-absent (the known "most chunks become all-marker" failure mode)?

/// Total number of times `run_decode_task` ran (one per submitted decode
/// job — speculative prefetch AND on-demand alike; includes any job whose
/// result is later discarded by the consumer, e.g. speculative results that
/// mismatch the confirmed block chain).
#[cfg(feature = "phase-timing")]
pub static WORKER_DECODE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Sum of `ChunkData::decoded_size()` over every SUCCESSFUL worker decode
/// invocation. Compare against the true output size: if this sum exceeds
/// the output size, some decoded bytes were produced more than once
/// (re-decode) or a decode's output was never used (discarded).
#[cfg(feature = "phase-timing")]
pub static WORKER_DECODED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Invocations that took the window-PRESENT branch (chunk 0's zero-window
/// sentinel, or a real predecessor window already published —
/// `decode_chunk_with_until_exact`).
#[cfg(feature = "phase-timing")]
pub static WORKER_DECODE_WINDOW_PRESENT: AtomicU64 = AtomicU64::new(0);
/// Invocations that took the window-ABSENT / speculative branch
/// (`speculative_decode_find_boundary` — no predecessor window published
/// yet, so the worker bootstraps with u16 markers and searches for a block
/// boundary).
#[cfg(feature = "phase-timing")]
pub static WORKER_DECODE_WINDOW_ABSENT: AtomicU64 = AtomicU64::new(0);

/// Reset all atomics to zero. Called once at the start of every `drive_impl`
/// invocation so per-decode snapshots do not accumulate across calls in a
/// long-lived process (e.g. a test harness that decodes repeatedly).
#[cfg(feature = "phase-timing")]
#[inline]
pub fn reset_atomics() {
    DECODE_WAIT_NS.store(0, Ordering::Relaxed);
    FUTURE_RECV_NS.store(0, Ordering::Relaxed);
    DRAIN_NS.store(0, Ordering::Relaxed);
    BLOCKFIND_NS.store(0, Ordering::Relaxed);
    ITERS.store(0, Ordering::Relaxed);
    WORKER_DECODE_INVOCATIONS.store(0, Ordering::Relaxed);
    WORKER_DECODED_BYTES.store(0, Ordering::Relaxed);
    WORKER_DECODE_WINDOW_PRESENT.store(0, Ordering::Relaxed);
    WORKER_DECODE_WINDOW_ABSENT.store(0, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn reset_atomics() {}

#[cfg(feature = "phase-timing")]
#[inline]
pub fn add_decode_wait(ns: u64) {
    DECODE_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn add_decode_wait(_ns: u64) {}

#[cfg(feature = "phase-timing")]
#[inline]
pub fn add_future_recv(ns: u64) {
    FUTURE_RECV_NS.fetch_add(ns, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn add_future_recv(_ns: u64) {}

#[cfg(feature = "phase-timing")]
#[inline]
pub fn add_drain(ns: u64) {
    DRAIN_NS.fetch_add(ns, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn add_drain(_ns: u64) {}

#[cfg(feature = "phase-timing")]
#[inline]
pub fn add_blockfind(ns: u64) {
    BLOCKFIND_NS.fetch_add(ns, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn add_blockfind(_ns: u64) {}

#[cfg(feature = "phase-timing")]
#[inline]
pub fn add_iter() {
    ITERS.fetch_add(1, Ordering::Relaxed);
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn add_iter() {}

/// Record one worker decode invocation: `window_present` selects which
/// branch of `run_decode_task` ran; `decoded_bytes` is `Some(n)` on a
/// successful decode (`ChunkData::decoded_size()`), `None` on error (no
/// bytes to count).
#[cfg(feature = "phase-timing")]
#[inline]
pub fn record_worker_decode(window_present: bool, decoded_bytes: Option<u64>) {
    WORKER_DECODE_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    if window_present {
        WORKER_DECODE_WINDOW_PRESENT.fetch_add(1, Ordering::Relaxed);
    } else {
        WORKER_DECODE_WINDOW_ABSENT.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(n) = decoded_bytes {
        WORKER_DECODED_BYTES.fetch_add(n, Ordering::Relaxed);
    }
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn record_worker_decode(_window_present: bool, _decoded_bytes: Option<u64>) {}

/// This-thread CPU time in nanoseconds (`CLOCK_THREAD_CPUTIME_ID`). Used to
/// compute `consumer_cpu_ns` as a delta across the `drive_impl` span — the
/// consumer thread runs `consumer_loop` inline (see `chunk_fetcher.rs`
/// doc), so this is specifically the CONSUMER's own CPU time, not a
/// process-wide figure.
#[cfg(feature = "phase-timing")]
#[inline]
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, live `timespec` on the stack; the call
    // requests the calling thread's own CPU-time clock, which is always
    // available on Linux/macOS. A failure (impossible for this clock id on
    // supported platforms) leaves `ts` zeroed, which degrades to a 0 delta
    // rather than UB.
    unsafe {
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}
#[cfg(not(feature = "phase-timing"))]
#[inline(always)]
pub fn thread_cpu_ns() -> u64 {
    0
}

/// RAII guard: on drop, adds the elapsed wall time since construction to
/// the accumulator selected by `add`. A true no-op (zero-sized, no
/// `Instant::now()` call) without the `phase-timing` feature.
#[cfg(feature = "phase-timing")]
#[must_use]
pub struct PhaseGuard {
    start: std::time::Instant,
    add: fn(u64),
}
#[cfg(feature = "phase-timing")]
impl PhaseGuard {
    #[inline]
    pub fn new(add: fn(u64)) -> Self {
        Self {
            start: std::time::Instant::now(),
            add,
        }
    }
}
#[cfg(feature = "phase-timing")]
impl Drop for PhaseGuard {
    #[inline]
    fn drop(&mut self) {
        (self.add)(self.start.elapsed().as_nanos() as u64);
    }
}

#[cfg(not(feature = "phase-timing"))]
#[must_use]
pub struct PhaseGuard;
#[cfg(not(feature = "phase-timing"))]
impl PhaseGuard {
    #[inline(always)]
    pub fn new(_add: fn(u64)) -> Self {
        Self
    }
}

/// One decode's phase-breakdown snapshot. Nanosecond fields as measured;
/// [`emit`] converts to microseconds on the wire (matches the JSON schema
/// in the module doc).
#[cfg(feature = "phase-timing")]
pub struct Phases {
    pub total_ns: u64,
    pub consumer_wall_ns: u64,
    pub consumer_cpu_ns: u64,
    pub finalize_ns: u64,
    pub decode_wait_ns: u64,
    pub future_recv_ns: u64,
    pub drain_ns: u64,
    pub blockfind_ns: u64,
    pub iters: u64,
    pub threads: usize,
    pub worker_decode_invocations: u64,
    pub worker_decoded_bytes: u64,
    pub worker_decode_window_present: u64,
    pub worker_decode_window_absent: u64,
}

#[cfg(feature = "phase-timing")]
impl Phases {
    /// Build a `Phases` snapshot from the caller-measured span timings plus
    /// a snapshot read of the module's accumulators.
    pub fn snapshot(
        consumer_wall_ns: u64,
        consumer_cpu_ns: u64,
        finalize_ns: u64,
        total_ns: u64,
        threads: usize,
    ) -> Self {
        Phases {
            total_ns,
            consumer_wall_ns,
            consumer_cpu_ns,
            finalize_ns,
            decode_wait_ns: DECODE_WAIT_NS.load(Ordering::Relaxed),
            future_recv_ns: FUTURE_RECV_NS.load(Ordering::Relaxed),
            drain_ns: DRAIN_NS.load(Ordering::Relaxed),
            blockfind_ns: BLOCKFIND_NS.load(Ordering::Relaxed),
            iters: ITERS.load(Ordering::Relaxed),
            threads,
            worker_decode_invocations: WORKER_DECODE_INVOCATIONS.load(Ordering::Relaxed),
            worker_decoded_bytes: WORKER_DECODED_BYTES.load(Ordering::Relaxed),
            worker_decode_window_present: WORKER_DECODE_WINDOW_PRESENT.load(Ordering::Relaxed),
            worker_decode_window_absent: WORKER_DECODE_WINDOW_ABSENT.load(Ordering::Relaxed),
        }
    }
}

/// Write ONE structured JSON line describing `p` to the path in env
/// `GZIPPY_PHASE_OUT` (append; created if absent), else to stderr prefixed
/// `[phase-timing] `. Hand-formatted flat object (no nesting, no serde_json
/// dependency needed — gzippy does not otherwise depend on serde_json).
///
/// Schema (protocol 1; all `_us` fields are integer microseconds). Fields
/// added 2026-07-09 (storedheavy sub-cause brief, STEP 1: re-decode/wasted-
/// work check) are additive-only, so protocol stays 1 — any consumer keyed
/// on the original field set still parses:
/// `{"kind":"phasebreak","protocol":1,"wall_us":..,"consumer_wall_us":..,
/// "consumer_cpu_us":..,"decode_wait_us":..,"future_recv_us":..,
/// "drain_us":..,"blockfind_us":..,"finalize_us":..,"iters":..,"threads":..,
/// "worker_decode_invocations":..,"worker_decoded_bytes":..,
/// "worker_decode_window_present":..,"worker_decode_window_absent":..}`
#[cfg(feature = "phase-timing")]
pub fn emit(p: &Phases) {
    let line = format!(
        "{{\"kind\":\"phasebreak\",\"protocol\":1,\"wall_us\":{},\"consumer_wall_us\":{},\"consumer_cpu_us\":{},\"decode_wait_us\":{},\"future_recv_us\":{},\"drain_us\":{},\"blockfind_us\":{},\"finalize_us\":{},\"iters\":{},\"threads\":{},\"worker_decode_invocations\":{},\"worker_decoded_bytes\":{},\"worker_decode_window_present\":{},\"worker_decode_window_absent\":{}}}",
        p.total_ns / 1000,
        p.consumer_wall_ns / 1000,
        p.consumer_cpu_ns / 1000,
        p.decode_wait_ns / 1000,
        p.future_recv_ns / 1000,
        p.drain_ns / 1000,
        p.blockfind_ns / 1000,
        p.finalize_ns / 1000,
        p.iters,
        p.threads,
        p.worker_decode_invocations,
        p.worker_decoded_bytes,
        p.worker_decode_window_present,
        p.worker_decode_window_absent,
    );
    match std::env::var_os("GZIPPY_PHASE_OUT") {
        Some(path) => {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(f, "{line}");
            }
        }
        None => {
            eprintln!("[phase-timing] {line}");
        }
    }
}
