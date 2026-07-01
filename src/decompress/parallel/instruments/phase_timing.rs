//! Phase-timing instrument (env-gated `GZIPPY_PHASE_TIMING=1`, byte-transparent).
//!
//! NO rapidgzip counterpart — this is OUR measurement tool. It splits the
//! end-to-end single-member decode wall into the SERIAL PHASES of the
//! `decompress_parallel` → `read_parallel_sm_inner` → `chunk_fetcher::drive_impl`
//! → `consumer_loop` call chain, to LOCATE the AMD-T2 serial-wrapper excess
//! (`project_amd_t2t4_locate_2026_06_22`: ~13ms gz-vs-rg serial OUTSIDE the
//! instrumented drive; clue = consumer blocks 26.5ms for the FIRST chunk).
//!
//! Phases (mark order, monotonic):
//!   decode_entry      decompress_parallel t0 (single_member.rs)
//!   envelope_parsed   gzip header+footer parsed, config built (sm_driver)
//!   scaffold_built    block_finder/window_map/block_fetcher/thread_pool built
//!                     == drive_t0 in drive_impl (block-finder bootstrap + pool spawn)
//!   first_output      first chunk dequeued for write in drain_one_pending
//!                     (first-chunk-to-first-output latency)
//!   consumer_done     consumer_loop returned (steady-state parallel decode + drain)
//!   finalize_done     thread_pool.stop + output flush + block_map/finder finalize
//!   crc_verified      trailer CRC32 + ISIZE verified (sm_driver)
//!
//! Self-validation (Gate-0):
//!   * conservation: the per-phase deltas telescope to (last − first) EXACTLY,
//!     proving the marks are monotonic and none was dropped/reordered.
//!   * non-inert: the report prints the FIRED mark count; a healthy run fires
//!     all 7. A missing mark (e.g. first_output never fired) is loud.
//!   * the instrument is OFF (zero work, no allocation past the OnceLock read)
//!     unless `GZIPPY_PHASE_TIMING` is set, so production bytes are unchanged.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("GZIPPY_PHASE_TIMING").is_some())
}

// (name, Instant) marks in fire order. One decode per CLI process for the bench;
// a Mutex<Vec> keeps it correct even if a test drives multiple decodes (reset()
// clears between runs).
static MARKS: Mutex<Vec<(&'static str, Instant)>> = Mutex::new(Vec::new());
// Names that have already fired this run, so `mark_once` is idempotent (the
// first_output mark sits on the per-chunk drain path and must record only the
// FIRST chunk).
static FIRED_ONCE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// Clear all marks. Called at `decode_entry` so a fresh decode starts clean.
pub fn reset() {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = MARKS.lock() {
        m.clear();
    }
    if let Ok(mut f) = FIRED_ONCE.lock() {
        f.clear();
    }
}

/// Record a phase boundary. No-op unless the gate is set.
#[inline]
pub fn mark(name: &'static str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    if let Ok(mut m) = MARKS.lock() {
        m.push((name, now));
    }
}

/// Record a phase boundary only the FIRST time it is reached this run (for marks
/// on a hot per-chunk path). No-op unless the gate is set.
#[inline]
pub fn mark_once(name: &'static str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    if let Ok(mut f) = FIRED_ONCE.lock() {
        if f.contains(&name) {
            return;
        }
        f.push(name);
    }
    if let Ok(mut m) = MARKS.lock() {
        m.push((name, now));
    }
}

/// Print the phase breakdown + Gate-0 self-validation to stderr. No-op unless the
/// gate is set. Called once at the end of `decompress_parallel`.
pub fn report() {
    if !enabled() {
        return;
    }
    let m = match MARKS.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if m.len() < 2 {
        eprintln!(
            "[phase-timing] INERT: only {} mark(s) fired — instrument did not span the decode",
            m.len()
        );
        return;
    }
    let first = m[0].1;
    let last = m[m.len() - 1].1;
    let total_ms = last.duration_since(first).as_secs_f64() * 1e3;

    eprintln!(
        "[phase-timing] fired_marks={} wall(first->last)={:.3}ms",
        m.len(),
        total_ms
    );
    let mut sum = 0.0f64;
    let mut monotonic = true;
    for w in m.windows(2) {
        let dt = w[1].1.duration_since(w[0].1).as_secs_f64() * 1e3;
        if w[1].1 < w[0].1 {
            monotonic = false;
        }
        sum += dt;
        let pct = if total_ms > 0.0 {
            dt / total_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:>16} -> {:<16} {:>9.3}ms  {:>5.1}%",
            w[0].0, w[1].0, dt, pct
        );
    }
    let gap = (sum - total_ms).abs();
    eprintln!(
        "  [conservation] sum_phases={:.3}ms total={:.3}ms gap={:.6}ms monotonic={} -> {}",
        sum,
        total_ms,
        gap,
        monotonic,
        if gap < 1e-3 && monotonic {
            "PASS"
        } else {
            "FAIL"
        }
    );
    // Non-inert proof: list which expected marks are MISSING.
    const EXPECTED: [&str; 9] = [
        "main_start",
        "decode_entry",
        "envelope_parsed",
        "scaffold_built",
        "first_output",
        "consumer_done",
        "finalize_done",
        "crc_verified",
        "main_end",
    ];
    let present: Vec<&str> = m.iter().map(|(n, _)| *n).collect();
    let missing: Vec<&str> = EXPECTED
        .iter()
        .copied()
        .filter(|e| !present.contains(e))
        .collect();
    if missing.is_empty() {
        eprintln!("  [non-inert] all 9 expected marks fired");
    } else {
        eprintln!("  [non-inert] MISSING marks: {missing:?}");
    }
}
