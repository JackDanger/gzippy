#![cfg(parallel_sm)]

//! PERFECT-OVERLAP removal oracle (`GZIPPY_PERFECT_OVERLAP=1`).
//!
//! THE REGISTERED DECIDER for the scheduling/serial ceiling
//! (git history (campaign plan, removed) §Oracle). Measurement-only,
//! byte-transparent on the output.
//!
//! ## What it removes (and what it KEEPS)
//! The production wall has (at least) three terms: (1) the per-thread
//! window-absent MARKER decode compute, (2) the scheduling/overlap loss
//! (pool-fill gap before steady state + the in-order consumer's
//! head-of-line wait when the head chunk is not yet DISPATCHED — the named
//! project_confirmed_offset_prefetch_gap dispatch-TIMING term), and (3) the
//! irreducible SERIAL marker-resolution window chain (chunk i's
//! predecessor window needs chunk i-1's resolved tail — rapidgzip's named
//! "critical path that cannot be parallelized", GzipChunkFetcher.hpp:559).
//!
//! This oracle removes term (2) ONLY, via a CORRECTED OVERLAP schedule
//! (2026-06-07; the PRIOR warm-all-then-drain version was an ANTI-overlap,
//! advisor-REFUTED — git history (campaign plan, removed)). The
//! dispatch phase submits EVERY chunk's decode as an IN-FLIGHT prefetch
//! up-front (non-blocking `submit_prefetch`) and returns IMMEDIATELY, so
//! the unchanged in-order `consumer_loop` runs CONCURRENTLY with the
//! still-running decodes: it drains chunk i (resolve markers off chunk
//! i-1's window + write) WHILE chunks i+1.. are still decoding on the pool.
//! That is decode↔drain OVERLAP — the schedule production and rapidgzip
//! actually use. The real MARKER engine (term 1, KEPT at its true rate —
//! chunks are NOT window-seeded) and the serial resolve chain + drain +
//! write (term 3) run faithfully. The timed wall (drive_t0) is the
//! overlapped wall — it bounds how far the dispatch-depth fix can collapse
//! production's 0.177s toward the ~0.117-0.13s decode↔drain floor.
//!
//! This is DISTINCT from `GZIPPY_SEED_WINDOWS` (seedfull): seedfull seeds
//! predecessor windows BEFORE decode, which flips every chunk to the CLEAN
//! engine (removing term (1) too) — so seedfull cannot isolate scheduling
//! from the engine. This oracle keeps the marker engine and isolates term
//! (2).
//!
//! ## Self-test (Rule 4 — validate before trusting)
//! 1. Output sha MUST be byte-identical to a normal decode (the dispatch
//!    phase only pre-issues the SAME decodes the prefetcher would have run).
//! 2. `warm_chunks` (decodes dispatched in flight) and `warm_hits`
//!    (consumer `get`s that hit a dispatched in-flight/cached chunk) are
//!    reported. warm_hits/total must be ≈1.0 — if it is not, the oracle did
//!    NOT remove the head-of-line dispatch wait and the number is void.

use std::sync::atomic::{AtomicU64, Ordering};

static WARM_CHUNKS: AtomicU64 = AtomicU64::new(0);
static WARM_HITS: AtomicU64 = AtomicU64::new(0);
static WARM_MISSES: AtomicU64 = AtomicU64::new(0);

/// `GZIPPY_PERFECT_OVERLAP=1` — run the corrected overlap-dispatch oracle.
pub fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var_os("GZIPPY_PERFECT_OVERLAP").is_some())
}

pub fn record_warm_chunk() {
    WARM_CHUNKS.fetch_add(1, Ordering::Relaxed);
}

/// A consumer `get` that hit a warm-cached chunk (head-of-line wait removed).
pub fn record_warm_hit() {
    WARM_HITS.fetch_add(1, Ordering::Relaxed);
}

/// A consumer `get` that MISSED the warm cache (a residual head-of-line
/// decode wait the oracle failed to remove — pushes the self-test below 1.0).
pub fn record_warm_miss() {
    WARM_MISSES.fetch_add(1, Ordering::Relaxed);
}

/// Report warm stats to stderr (the harness reads these for the Rule-4
/// self-test). Wall split (warm vs drain) is printed by the caller.
pub fn report_stats() {
    if !enabled() {
        return;
    }
    let chunks = WARM_CHUNKS.load(Ordering::Relaxed);
    let hits = WARM_HITS.load(Ordering::Relaxed);
    let misses = WARM_MISSES.load(Ordering::Relaxed);
    let total = hits + misses;
    let hit_frac = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };
    eprintln!(
        "  PERFECT_OVERLAP: warm_chunks={chunks} consumer_get hits={hits} misses={misses} \
         warm_hit_frac={hit_frac:.3} (self-test: must be ~1.0 or the number is void)"
    );
}
