//! Byte-transparent, env-gated slow-injection knob for the CLEAN-mode inner
//! decode loop — a causal-perturbation pre-gate instrument.
//!
//! Purpose (Measurement PROCESS rule #1 in CLAUDE.md): to test whether the
//! clean-mode inner Huffman loop GATES the decode wall, we change ONLY that
//! loop's per-iteration time by a known factor and watch the interleaved wall
//! response. A monotonic, proportional response ⇒ the loop is on the critical
//! path; a flat response ⇒ it is slack. This module is the slow-injection knob
//! (the substitute for coz, which needs perf-event sampling that is blocked in
//! the LXC / unavailable here).
//!
//! ## Env contract
//!
//! * `GZIPPY_SLOW_MODE` — a numeric percent as a string. `"25"` ⇒ +25%, `"50"`
//!   ⇒ +50%, `"100"` ⇒ +100% of the clean loop's own per-iteration compute.
//!   Unset, `"0"`, negative, or unparseable ⇒ OFF (no injection at all).
//! * `GZIPPY_SLOW_KIND` — `"spin"` (default) is a busy ALU loop; `"sleep"` is
//!   the frequency-neutral CONTROL (see the kind note below).
//!
//! Both are read ONCE into module-level `OnceLock`s (the `mem_stats::enabled()`
//! pattern). Callers snapshot the resolved per-iteration spin count into a LOCAL
//! once, BEFORE the hot loop, so the per-iteration cost when OFF is a single
//! branch on a local `== 0` that the optimizer hoists / predicts away. Combined
//! with the `CONTAINS_MARKERS` const generic (callers pass `0` in the markered
//! arm), the injection is compiled AWAY entirely on the marker path and is a
//! single hoistable branch on the clean path when OFF.
//!
//! ## Byte transparency
//!
//! The injected work only reads/writes a black-boxed scratch accumulator; it
//! never touches decode state, the bit reader, or the output ring. So with the
//! knob OFF *or* ON the decoded bytes are identical — proved by the DUAL-SHA
//! gate (OFF sha == ON sha == the canonical silesia reference).
//!
//! ## Calibration (`BASE_SPIN`)
//!
//! The injection magnitude is PROPORTIONAL to the loop's work: each clean-loop
//! iteration is exactly one Huffman decode event (one LUT/codeword decode that
//! emits 1–3 literals or one back-reference), which is the unit of decode work
//! we are perturbing. We add `(BASE_SPIN as f64 * F) as u64` iterations of a
//! black-boxed no-op per decode event, where `F = percent / 100`. `BASE_SPIN`
//! is picked so that `F = 1.0` (`GZIPPY_SLOW_MODE=100`) roughly DOUBLES the
//! single-thread clean-loop decode wall on arm64 native. Perfect calibration is
//! not required — what matters is MONOTONICITY (larger `F` ⇒ more wall) and the
//! documented linear relationship. See `plans/slow-knob-impl.md` for the
//! measured OFF-vs-ON ratio that fixed `BASE_SPIN`.

use core::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// SITE-VALIDITY hit counter (instrument-validation only, TASK 1). Counts the
/// number of clean-mode decode events that pass through [`inject`], regardless
/// of whether the knob is ON or OFF. Used ONCE to PROVE the injection site is
/// the real native clean loop: build gzippy-native, decode silesia at T8, and
/// assert this counter fires ∝ clean decoded bytes (NOT ~0). Reported by
/// [`report_hits`] when `GZIPPY_SLOW_HITS=1`. The increment is a single relaxed
/// atomic add gated behind the same hoistable OFF branch as the spin work when
/// counting is disabled, so it is perf-transparent unless `GZIPPY_SLOW_HITS=1`.
static HIT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[inline]
fn count_hits() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| matches!(std::env::var("GZIPPY_SLOW_HITS").ok().as_deref(), Some("1")))
}

/// Print the clean-loop decode-event hit count when `GZIPPY_SLOW_HITS=1`. Call
/// from `main` after the decode completes. No-op otherwise.
pub fn report_hits() {
    if count_hits() {
        eprintln!(
            "[slow_knob] clean-loop inject hits = {}",
            HIT_COUNTER.load(Ordering::Relaxed)
        );
    }
}

/// Per-decode-event spin iteration count at `F = 1.0`. Calibrated so
/// `GZIPPY_SLOW_MODE=100` roughly doubles the single-thread clean-loop wall on
/// arm64 native (see module + `plans/slow-knob-impl.md`).
const BASE_SPIN: u64 = 22;

/// Slow factor as a fraction (`percent / 100`). `0.0` ⇒ OFF. Read once.
#[inline]
fn slow_factor() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GZIPPY_SLOW_MODE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|pct| pct / 100.0)
            .unwrap_or(0.0)
    })
}

/// `false` ⇒ busy ALU spin (default). `true` ⇒ frequency-neutral yield control.
/// Read once.
#[inline]
fn sleep_kind() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_SLOW_KIND").ok().as_deref(),
            Some("sleep")
        )
    })
}

/// Resolved per-decode-event spin iteration count: `(BASE_SPIN * F)`, or `0`
/// when the knob is OFF. Call this ONCE before the hot loop and snapshot into a
/// local; pass that local to [`inject`] inside the loop so the OFF fast path is
/// a single branch on a hoistable local.
#[inline]
pub fn spin_iters() -> u64 {
    let f = slow_factor();
    if f <= 0.0 {
        return 0;
    }
    (BASE_SPIN as f64 * f) as u64
}

/// `true` when the frequency-neutral CONTROL kind (`GZIPPY_SLOW_KIND=sleep`) is
/// selected. Snapshot alongside [`spin_iters`] before the loop.
///
/// ## Why a REAL batched sleep (not a pause hint)
///
/// The control's whole job (Measurement PROCESS rule #2) is to add a wall delay
/// COMPARABLE to the busy-spin kind but via a mechanism that YIELDS the core, so
/// it does not depress all-core turbo: if the T8 wall rise SURVIVES the swap from
/// spin to sleep, the criticality is real, not a turbo artifact. For that to be
/// a valid control the sleep must add a comparable amount of wall — an
/// architectural pause hint (`core::hint::spin_loop`) does NOT: measured, even a
/// 5× factor of pause hints adds ~7% where the ALU spin adds ~44%, so a "flat
/// under pause-hint" result would be uninterpretable (could mean "not on the
/// path" OR "the hint is nearly free"). So this control issues a genuine
/// `std::thread::sleep` (nanosleep — actually deschedules the thread).
///
/// At per-decode-event granularity a per-event sleep is impossible (millions of
/// events; nanosleep granularity is ~µs), so the owed delay is ACCUMULATED in a
/// thread-local nanosecond debt (proportional to `spin` per event, calibrated to
/// the ALU spin's measured per-iter cost) and discharged with a single real
/// sleep whenever the debt crosses [`SLEEP_BATCH_NS`]. Net injected wall ≈ the
/// spin kind at the same factor, but the thread is descheduled during it.
#[inline]
pub fn yield_kind() -> bool {
    sleep_kind()
}

/// Calibrated injected wall per spin-iter for the ALU busy-spin, in nanoseconds.
/// Measured on the perf-target build (Rosetta x86_64 native, T1 silesia): the
/// ALU spin at F=2.0 (spin=44 iters/event) added ~0.571 s over ~40.13 M decode
/// events ≈ 14.2 ns/event ≈ 0.32 ns per spin-iter. The sleep control accumulates
/// `spin * NS_PER_SPIN_ITER` ns of debt per event so its injected wall matches
/// the spin kind at the same factor. Exact match is not required (the control is
/// about turbo-neutrality, not magnitude parity) — comparable is sufficient.
const NS_PER_SPIN_ITER: f64 = 0.32;

/// Discharge accumulated sleep debt in batches of this many ns (coarse enough to
/// be above nanosleep granularity, fine enough to spread evenly across decode).
const SLEEP_BATCH_NS: u64 = 50_000;

/// Inject one decode-event's worth of extra work. `spin` is the snapshotted
/// [`spin_iters`] value; `yield_hint` is the snapshotted [`yield_kind`] value.
///
/// `spin == 0` (OFF) returns immediately — the single hoistable branch. The
/// busy work only mutates a black-boxed local accumulator, so it is
/// byte-transparent; `black_box` prevents the optimizer from deleting it.
#[inline(always)]
pub fn inject(spin: u64, yield_hint: bool) {
    // SITE-VALIDITY counter (TASK 1): only active when GZIPPY_SLOW_HITS=1, so
    // the production OFF path stays perf-transparent. Placed BEFORE the
    // `spin == 0` early-return so it counts every clean decode event even when
    // no slow work is injected — that is exactly the site-fires-∝-bytes proof.
    if count_hits() {
        HIT_COUNTER.fetch_add(1, Ordering::Relaxed);
    }
    if spin == 0 {
        return;
    }
    if yield_hint {
        // Frequency-neutral control: accumulate this event's owed delay as a
        // thread-local ns debt and discharge it with a REAL sleep (which
        // yields the core) whenever the debt crosses SLEEP_BATCH_NS. A real
        // sleep is what makes this turbo-neutral; batching makes it feasible at
        // per-event granularity. See `yield_kind` for the calibration rationale.
        thread_local! {
            static SLEEP_DEBT_NS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
        }
        SLEEP_DEBT_NS.with(|debt| {
            let owed = (spin as f64 * NS_PER_SPIN_ITER) as u64;
            let total = debt.get() + owed;
            if total >= SLEEP_BATCH_NS {
                std::thread::sleep(std::time::Duration::from_nanos(total));
                debt.set(0);
            } else {
                debt.set(total);
            }
        });
    } else {
        // Default busy ALU spin proportional to decode work.
        let mut acc: u64 = black_box(0);
        for i in 0..spin {
            acc = acc.wrapping_add(black_box(i).wrapping_mul(0x9E37_79B9));
        }
        black_box(acc);
    }
}
