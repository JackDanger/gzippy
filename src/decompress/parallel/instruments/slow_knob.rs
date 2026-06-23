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
//! documented linear relationship. See `git history (campaign plan, removed)` for the
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

/// MARKER-SITE-VALIDITY hit counter (instrument-validation only). Counts the
/// number of u16 MARKER-mode (`CONTAINS_MARKERS == true`) careful-loop decode
/// events that pass through [`marker_inject`]. DISTINCT from [`HIT_COUNTER`]
/// (which counts the clean `<false>` careful-loop events) — the two never share
/// a site, so a non-zero value here PROVES the marker knob fired on the
/// marker-mode decode path SPECIFICALLY (not the clean contig loop). Reported by
/// [`report_hits`] when `GZIPPY_SLOW_MARKER_HITS=1`.
static MARKER_HIT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[inline]
fn count_hits() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| matches!(std::env::var("GZIPPY_SLOW_HITS").ok().as_deref(), Some("1")))
}

#[inline]
fn count_marker_hits() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_SLOW_MARKER_HITS").ok().as_deref(),
            Some("1")
        )
    })
}

/// Print the clean-loop decode-event hit count when `GZIPPY_SLOW_HITS=1`. Call
/// from `main` after the decode completes. No-op otherwise.
pub fn report_hits() {
    report_ring_inject();
    if count_hits() {
        eprintln!(
            "[slow_knob] clean-loop inject hits = {}",
            HIT_COUNTER.load(Ordering::Relaxed)
        );
    }
    if count_marker_hits() {
        eprintln!(
            "[slow_knob] marker-loop inject hits = {}",
            MARKER_HIT_COUNTER.load(Ordering::Relaxed)
        );
    }
    if count_hits() {
        eprintln!(
            "[slow_knob] marker-CEILING oracle hits = {}",
            MARKER_CEILING_HITS.load(Ordering::Relaxed)
        );
    }
    // U16-preserving ceiling arms: always report when armed (Gate-0 non-inert
    // proof: HITS must match the u8 ceiling HITS and RESOLVE_BYTES > 0).
    if marker_ceiling_u16() || count_hits() {
        eprintln!(
            "[slow_knob] marker-CEILING-U16 (consumer-serial) hits = {} resolve_bytes = {} (seeded-decode hits = {})",
            MARKER_CEILING_U16_HITS.load(Ordering::Relaxed),
            MARKER_CEILING_U16_RESOLVE_BYTES.load(Ordering::Relaxed),
            MARKER_CEILING_HITS.load(Ordering::Relaxed),
        );
    }
    if marker_ceiling_u16w() || count_hits() {
        eprintln!(
            "[slow_knob] marker-CEILING-U16W (worker-parallel) hits = {} resolve_bytes = {} (seeded-decode hits = {})",
            MARKER_CEILING_U16W_HITS.load(Ordering::Relaxed),
            MARKER_CEILING_U16W_RESOLVE_BYTES.load(Ordering::Relaxed),
            MARKER_CEILING_HITS.load(Ordering::Relaxed),
        );
    }
}

/// Per-decode-event spin iteration count at `F = 1.0`. Calibrated so
/// `GZIPPY_SLOW_MODE=100` roughly doubles the single-thread clean-loop wall on
/// arm64 native (see module + `git history (campaign plan, removed)`).
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

/// MARKER-mode slow factor (`GZIPPY_SLOW_MARKER_MODE`, percent/100). `0.0` ⇒ OFF.
/// Read once. This is the u16 `<true>`-path twin of [`slow_factor`]: it perturbs
/// ONLY the per-decode-event time of the u16 marker (`CONTAINS_MARKERS`) inner
/// loop, so the u16-path-gates-the-wall question can be answered by the same
/// causal-perturbation method that established the clean-path (`<false>`) ceiling.
/// Byte-transparent (the injected work touches only a black-boxed accumulator).
#[inline]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
fn slow_marker_factor() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GZIPPY_SLOW_MARKER_MODE")
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

/// MARKER-mode resolved per-decode-event spin count (`GZIPPY_SLOW_MARKER_MODE`),
/// or `0` when OFF. The u16 `<true>`-path twin of [`spin_iters`]. Snapshot once
/// before the marker careful loop; `GZIPPY_SLOW_KIND=sleep` selects the same
/// frequency-neutral control via [`yield_kind`].
#[inline]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
pub fn marker_spin_iters() -> u64 {
    let f = slow_marker_factor();
    if f <= 0.0 {
        return 0;
    }
    (BASE_SPIN as f64 * f) as u64
}

// ── DECODE-COMPUTE vs STORE-BANDWIDTH LOCALIZATION KNOBS ────────────────────
//
// The whole-loop-body knob (`GZIPPY_SLOW_MODE`) proved the contig clean loop is
// on the T8 critical path, but a per-loop-body inject CANNOT separate the
// Huffman table-lookup + bit-extraction COMPUTE from the literal-store /
// back-ref-copy STORE bandwidth (advisor verdict on contig-clean-perturbation).
// These two SEPARATE knobs each perturb exactly one sub-resource so a causal
// perturbation can tell WHICH one binds:
//
//   * `GZIPPY_SLOW_DECODE` — injects ONLY at Huffman decode events (after each
//     `lut_litlen.decode()` and each `dist_hc.decode()`), perturbing the pure
//     table-lookup + bit-extraction compute. If THIS moves the wall and STORE
//     does NOT, decode-compute is the binder ⇒ BMI2 PEXT/BZHI / packed-u32 LUT.
//   * `GZIPPY_SLOW_STORE` — injects ONLY at literal-store / back-ref-copy events,
//     perturbing store/copy bandwidth. If THIS moves the wall and DECODE does
//     not, the binder is store bandwidth (and the already-TIE'd packed
//     multi-literal store is the relevant lever, exhausted).
//
// Unlike `GZIPPY_SLOW_MODE`, these knobs do NOT force the careful loop — the
// production VAR_V fast loop stays gated on `slow_spin == 0` only — so the
// perturbation lands on the PRODUCTION fast path (which handles ~69% of clean
// decode events) as well as the careful tail. `GZIPPY_SLOW_KIND=sleep` selects
// the same frequency-neutral control. Byte-transparent (DUAL-SHA gate): the
// injected work touches only a black-boxed accumulator / a ns-debt cell.

#[inline]
fn slow_decode_factor() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GZIPPY_SLOW_DECODE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|pct| pct / 100.0)
            .unwrap_or(0.0)
    })
}

#[inline]
fn slow_store_factor() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GZIPPY_SLOW_STORE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|pct| pct / 100.0)
            .unwrap_or(0.0)
    })
}

/// Resolved per-decode-event spin count for the DECODE-COMPUTE knob
/// (`GZIPPY_SLOW_DECODE`), or `0` when OFF. Snapshot once before the loop.
#[inline]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
pub fn decode_spin_iters() -> u64 {
    let f = slow_decode_factor();
    if f <= 0.0 {
        return 0;
    }
    (BASE_SPIN as f64 * f) as u64
}

/// Resolved per-store-event spin count for the STORE-BANDWIDTH knob
/// (`GZIPPY_SLOW_STORE`), or `0` when OFF. Snapshot once before the loop.
#[inline]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
pub fn store_spin_iters() -> u64 {
    let f = slow_store_factor();
    if f <= 0.0 {
        return 0;
    }
    (BASE_SPIN as f64 * f) as u64
}

/// `true` when EITHER localization knob is active AND the sleep control kind is
/// selected. Snapshot alongside the per-knob spin counts before the loop. Gated
/// on a non-zero localization spin so a `GZIPPY_SLOW_KIND=sleep` set for the
/// OLD whole-body knob does not leak into a localization measurement.
#[inline]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
pub fn localize_yield_kind(spin: u64) -> bool {
    spin != 0 && sleep_kind()
}

/// Inject one event's worth of extra work for a LOCALIZATION knob (decode-only
/// or store-only). Same mechanism as [`inject`] but WITHOUT the
/// `GZIPPY_SLOW_HITS` site-validity counter (that counter belongs to the
/// whole-body clean knob). `spin == 0` (OFF) returns immediately —
/// byte-transparent, a single hoistable branch on the production path.
#[inline(always)]
#[allow(dead_code)] // instrument: only reached from the feature-gated decode loops
pub fn inject_localize(spin: u64, yield_hint: bool) {
    if spin == 0 {
        return;
    }
    if yield_hint {
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
        let mut acc: u64 = black_box(0);
        for i in 0..spin {
            acc = acc.wrapping_add(black_box(i).wrapping_mul(0x9E37_79B9));
        }
        black_box(acc);
    }
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

// ── MFAST-PROBE KNOBS (probe/mfast-phase0) ──────────────────────────────────
//
// Two new knobs for the mfast-phase0 causal probe. They follow the same
// OnceLock-cached pattern as the existing localization knobs (`dec_spin` /
// `st_spin`) — a single predictable branch per site when OFF, zero runtime cost
// when OFF and the optimizer hoists the `spin == 0` check.
//
//   GZIPPY_MFAST_DISABLE=1   — skip `'mfast` entry (discriminator arm). The
//                               careful loop handles 100% of marker decode at
//                               baseline per-event cost. Wall-flat ⇒ mfast is
//                               slack (competing binder = head-of-line stalls).
//   GZIPPY_SLOW_MFAST_MODE=N — per-event inject INSIDE `'mfast` WITHOUT gating
//                               its entry (unlike GZIPPY_SLOW_MARKER_MODE, which
//                               sets slow_spin != 0 and therefore DISABLES the
//                               fast loop entry). N% ⇒ BASE_SPIN*N/100 iters.
//                               GZIPPY_SLOW_KIND=sleep selects the
//                               frequency-neutral control via localize_yield_kind.
//
// Both are byte-transparent (inject never touches decode state; disable uses the
// same careful-loop path that already exists). DUAL-SHA gate required before
// any arm.

/// `GZIPPY_MFAST_DISABLE=1` — skip the `'mfast` marker fast-loop entry entirely;
/// the careful loop handles 100% of marker decode at baseline per-event cost.
/// OnceLock-cached, `false` by default. Byte-transparent.
#[inline]
#[allow(dead_code)]
pub fn mfast_disabled() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_MFAST_DISABLE").ok().as_deref(),
            Some("1")
        )
    })
}

/// `GZIPPY_SLOW_MFAST_MODE` — per-event spin count for the localized inject
/// inside `'mfast` (NOT tied to the entry gate). Follows the same BASE_SPIN
/// convention as [`spin_iters`]: N% ⇒ `(BASE_SPIN * N / 100)` iters.
/// Returns `0` when the knob is OFF or unparseable. Snapshot once before the
/// `'mfast` block; pass to [`inject_localize`] after the loop guard.
#[inline]
#[allow(dead_code)]
pub fn mfast_spin_iters() -> u64 {
    static F: OnceLock<u64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GZIPPY_SLOW_MFAST_MODE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|pct| (BASE_SPIN as f64 * pct / 100.0) as u64)
            .unwrap_or(0)
    })
}

/// Inject one decode-event's worth of extra work. `spin` is the snapshotted
/// [`spin_iters`] value; `yield_hint` is the snapshotted [`yield_kind`] value.
///
/// `spin == 0` (OFF) returns immediately — the single hoistable branch. The
/// busy work only mutates a black-boxed local accumulator, so it is
/// byte-transparent; `black_box` prevents the optimizer from deleting it.
/// MARKER-mode twin of [`inject`]. Identical injection mechanism (byte-transparent
/// spin / frequency-neutral sleep), but its site-validity counter is the
/// MARKER-specific [`MARKER_HIT_COUNTER`] (gated on `GZIPPY_SLOW_MARKER_HITS=1`),
/// NOT the clean-path [`HIT_COUNTER`]. Call this ONLY from the `CONTAINS_MARKERS`
/// (u16 marker) careful-loop inject site so a non-zero counter PROVES the marker
/// knob fired on the marker decode path specifically. `spin == 0` (OFF) is a
/// single hoistable branch — perf-transparent on the production marker path.
#[inline(always)]
#[allow(dead_code)] // instrument: only reached from the feature-gated marker decode loop
pub fn marker_inject(spin: u64, yield_hint: bool) {
    if count_marker_hits() {
        MARKER_HIT_COUNTER.fetch_add(1, Ordering::Relaxed);
    }
    inject_localize(spin, yield_hint);
}

// ── STEP-1 CEILING ORACLE (GZIPPY_MARKER_CEILING) ───────────────────────────
//
// Causal REMOVAL oracle for the Zen2 marker-asm decision (CLAUDE.md Gate-2: "to
// BOUND a speed-up you must REMOVE the region and measure"). When ON, the
// window-absent (speculative) decode in `decode_chunk_window_absent` seeds a
// ZEROED 32 KiB window, routing the whole bootstrap chunk through the clean asm
// `run_contig` path (~4.7 cyc/B) instead of the 11.7 cyc/B u16-marker loop. The
// huffman bitstream decode is identical (same symbols, same lengths, same EOB),
// so the chunk size + block boundaries + control flow are UNCHANGED — only the
// back-reference byte VALUES are wrong (they resolve into the zero window). Output
// is therefore wrong-on-purpose (sha mismatch EXPECTED — this is a perturbation,
// not a product); the full pipeline still runs end to end and the final CRC fails
// only AFTER all bytes are written, so a `perf stat duration_time` wall captures
// the full decode+write. This yields the ABSOLUTE (generous) ceiling: all decode
// at clean asm speed, marker machinery + the separate marker-resolution pass both
// removed. If the AMD/Zen2 wall STILL loses to rapidgzip under this oracle, no
// decode-throughput asm can pay (STOP); if it closes to <=~1.01 the realistic
// marker asm (which lands between baseline and this ceiling) is worth building.

static MARKER_CEILING_HITS: AtomicU64 = AtomicU64::new(0);

/// `GZIPPY_MARKER_CEILING=1` — STEP-1 ceiling oracle (see module note above).
/// OnceLock-cached, `false` by default. NOT byte-transparent (output is
/// wrong-on-purpose) — this is a perturbation oracle, never the product path.
#[inline]
#[allow(dead_code)]
pub fn marker_ceiling() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_MARKER_CEILING").ok().as_deref(),
            Some("1")
        )
    })
}

/// Record that the ceiling oracle fired on one window-absent chunk (Gate-0
/// non-inert proof). Reported by [`report_hits`] when `GZIPPY_SLOW_HITS=1`.
#[inline]
#[allow(dead_code)]
pub fn note_marker_ceiling_hit() {
    MARKER_CEILING_HITS.fetch_add(1, Ordering::Relaxed);
}

// ── STEP-1 U16-PRESERVING CEILING ORACLE (corrected; cursor-agent design-reviewed) ──
//
// The plain u8 `GZIPPY_MARKER_CEILING` above is OVER-GENEROUS: it seeds a zeroed
// window so the speculative chunk decodes through the clean asm `run_contig` u8 path
// (~4.7 cyc/B) AND, because `data_with_markers` ends up EMPTY, the consumer's
// `resolve_and_narrow_markers_in_place` becomes a NO-OP — so it deletes BOTH the u16
// marker buffer (2× write traffic + footprint) AND the apply-window resolve/gather
// pass. A real marker-decode asm can only speed the per-symbol DECODE; it CANNOT
// remove the u16 buffer or the resolve gather. So the realistic prize is bracketed:
//
//   baseline(~11.7 cyc/B) > U16-ceiling(clean decode + u16 write + resolve) > u8-ceiling
//
// The U16-preserving ceiling keeps the u16 write traffic + the real fused
// `resolve_and_narrow_in_place` LUT/gather pass, crediting ONLY clean-asm decode
// speed. Two location arms bracket where the resolve runs in the real pipeline (pool
// post-process, consumer-blocked):
//   * `GZIPPY_MARKER_CEILING_U16W` — phantom resolve in the DECODE WORKER (parallel;
//     OPTIMISTIC on resolve location — under-counts the serial gate).
//   * `GZIPPY_MARKER_CEILING_U16`  — phantom resolve INLINE on the CONSUMER thread,
//     serialized per chunk (PESSIMISTIC location — over-serializes vs the real
//     pool-parallel-but-consumer-blocked resolve).
// The truth lies between the two arms. Both close ⇒ robust CONFIRM; neither ⇒ robust
// REFUTE. Output bytes are wrong-on-purpose (perturbation, never the product).

static MARKER_CEILING_U16_HITS: AtomicU64 = AtomicU64::new(0);
static MARKER_CEILING_U16_RESOLVE_BYTES: AtomicU64 = AtomicU64::new(0);
static MARKER_CEILING_U16W_HITS: AtomicU64 = AtomicU64::new(0);
static MARKER_CEILING_U16W_RESOLVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// `GZIPPY_MARKER_CEILING_U16=1` — consumer-inline (serial) U16-preserving ceiling.
#[inline]
#[allow(dead_code)]
pub fn marker_ceiling_u16() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_MARKER_CEILING_U16").ok().as_deref(),
            Some("1")
        )
    })
}

/// `GZIPPY_MARKER_CEILING_U16W=1` — worker-side (parallel) U16-preserving ceiling.
#[inline]
#[allow(dead_code)]
pub fn marker_ceiling_u16w() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_MARKER_CEILING_U16W").ok().as_deref(),
            Some("1")
        )
    })
}

/// True if any U16-preserving ceiling arm is armed (so the caller takes the
/// seeded-clean-decode path used by all ceiling arms).
#[inline]
#[allow(dead_code)]
pub fn marker_ceiling_any_u16() -> bool {
    marker_ceiling_u16() || marker_ceiling_u16w()
}

/// Inject EXACTLY the u16 marker write traffic + the real fused resolve/gather
/// pass for one ceiling chunk of `n` decoded bytes, then discard the scratch. This
/// is the cost a marker-decode asm CANNOT remove (it lands on top of the clean-asm
/// decode speed credited by the seeded-window decode). `worker_arm` selects the
/// counter so the worker (parallel) vs consumer (serial) location arms stay
/// distinct. Builds a `SegmentedU16` with ~31% scattered markers (matching the
/// measured replaced-marker fraction) and the rest literals, then runs the SAME
/// `resolve_and_narrow_in_place` kernel the production consumer runs.
///
/// Gated to `parallel_sm`: it references the marker buffer + resolve kernel which
/// only exist in the parallel single-member engine (its only callers are
/// parallel_sm-gated). The knob readers above stay ungated (plain env reads).
#[cfg(parallel_sm)]
#[allow(dead_code)]
pub fn phantom_marker_resolve_traffic(n: usize, worker_arm: bool) {
    if n == 0 {
        return;
    }
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
    use crate::decompress::parallel::segmented_markers::SegmentedU16;

    let mut scratch = SegmentedU16::default();
    const FILL_CHUNK: usize = 8192;
    let mut buf = [0u16; FILL_CHUNK];
    let mut remaining = n;
    let mut base = 0usize;
    while remaining > 0 {
        let m = remaining.min(FILL_CHUNK);
        for (j, slot) in buf.iter_mut().enumerate().take(m) {
            let i = base + j;
            // Deterministic scatter (~31% markers) into the 32 KiB window, the
            // rest literals (< 256). Avoids artificial sequential gather locality.
            let h = i.wrapping_mul(2_654_435_761);
            *slot = if (h >> 7) % 100 < 31 {
                MARKER_BASE + ((h % 32768) as u16)
            } else {
                (i & 0xFF) as u16
            };
        }
        scratch.push_slice(&buf[..m]);
        base += m;
        remaining -= m;
    }
    // Real production resolve kernel (64 KiB u8 LUT, window gather, u16→u8 narrow).
    let zero_window = [0u8; 32768];
    scratch.resolve_and_narrow_in_place(&zero_window);
    std::hint::black_box(&scratch);
    if worker_arm {
        MARKER_CEILING_U16W_HITS.fetch_add(1, Ordering::Relaxed);
        MARKER_CEILING_U16W_RESOLVE_BYTES.fetch_add(n as u64, Ordering::Relaxed);
    } else {
        MARKER_CEILING_U16_HITS.fetch_add(1, Ordering::Relaxed);
        MARKER_CEILING_U16_RESOLVE_BYTES.fetch_add(n as u64, Ordering::Relaxed);
    }
}

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

// ── RING / FOLD-DRAIN PER-CHUNK INJECTOR (ring_other region perturbation) ─────
//
// Causal-perturbation (CLAUDE.md Gate-2) knob for the per-worker `ring_other`
// region = R_WORKER − R_TABLE − R_DECODE (per-chunk scaffold: ChunkData setup +
// the per-block loop machinery in `finish_decode_chunk_contig_native` + finalize
// + the ~1% ContigFoldSink drain copy). The fulcrum F-9c5ca01d020d sub-decomp
// flagged this as the LARGEST raw gz>rg gap but did NOT prove it MOVES the wall.
//
// Contract:
//   GZIPPY_RING_INJECT_NS=<ns>  — inject ~<ns> nanoseconds of work PER CHUNK in
//     the ring_other region (the call site is in the chunk_fetcher worker
//     wrapper, inside the R_WORKER rdtsc window but OUTSIDE the Block-method
//     R_TABLE/R_DECODE spans, so the injected cycles land in `ring_other`).
//     Unset / 0 / unparseable ⇒ OFF (a single hoistable branch per chunk —
//     chunk granularity, NOT per-decode-event, so OFF cost is negligible).
//   GZIPPY_SLOW_KIND=sleep      — frequency-neutral CONTROL (real nanosleep that
//     yields the core) instead of the busy rdtsc-bounded spin. Reuses the same
//     env knob the inner-loop perturbation uses.
//
// Byte transparency: the busy arm only mutates a black-boxed accumulator; the
// sleep arm only deschedules. Decoded bytes are IDENTICAL ON vs OFF (DUAL-SHA
// gate). Non-inert proof: RING_INJECT_HITS (== chunk decode count) and
// RING_INJECT_NS_TOTAL (> 0) reported by `report_hits` when the knob is set.

/// Non-inert proof: number of chunks the ring injector fired on.
pub static RING_INJECT_HITS: AtomicU64 = AtomicU64::new(0);
/// Non-inert proof: total nanoseconds the ring injector TARGETED (sum of per-
/// chunk `<ns>`). Actual injected wall ≈ this on the frozen box.
pub static RING_INJECT_NS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Per-chunk target injection in nanoseconds (`GZIPPY_RING_INJECT_NS`). `0` ⇒
/// OFF. Read once.
#[inline]
pub fn ring_inject_ns() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GZIPPY_RING_INJECT_NS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}

/// Calibrated TSC ticks per nanosecond, measured ONCE at first use against a
/// 10 ms `Instant` window. On the frozen Zen2 box (invariant TSC, gov=
/// performance, boost=0) TSC ≈ core cycles, so an rdtsc-bounded spin yields a
/// precise wall-ns injection — turbo-neutral by construction (freq is pinned),
/// which is what makes the busy arm and the sleep control comparable.
#[inline]
#[allow(dead_code)] // instrument: only reached from the parallel_sm worker call site
fn tsc_per_ns() -> f64 {
    static C: OnceLock<f64> = OnceLock::new();
    *C.get_or_init(|| {
        let rd = || crate::decompress::parallel::instruments::contig_prof::rdtsc(true);
        let t0 = std::time::Instant::now();
        let c0 = rd();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let c1 = rd();
        let ns = t0.elapsed().as_nanos() as f64;
        let dc = c1.wrapping_sub(c0) as f64;
        if ns > 0.0 && dc > 0.0 {
            dc / ns
        } else {
            2.8 // EPYC 7282 base ~2.8 GHz fallback
        }
    })
}

/// Inject one chunk's worth of ring-region work. Snapshot `ring_inject_ns()` /
/// `yield_kind()` is unnecessary here — chunk granularity makes the OnceLock
/// read negligible. OFF (ns==0) returns on a single predicted branch.
#[inline]
#[allow(dead_code)] // instrument: only reached from the parallel_sm worker call site
pub fn ring_inject() {
    let ns = ring_inject_ns();
    if ns == 0 {
        return;
    }
    RING_INJECT_HITS.fetch_add(1, Ordering::Relaxed);
    RING_INJECT_NS_TOTAL.fetch_add(ns, Ordering::Relaxed);
    if yield_kind() {
        // Frequency-neutral control: a real nanosleep (yields the core).
        std::thread::sleep(std::time::Duration::from_nanos(ns));
    } else {
        // Busy rdtsc-bounded spin: precise wall-ns on the frozen box.
        let target = (ns as f64 * tsc_per_ns()) as u64;
        let rd = || crate::decompress::parallel::instruments::contig_prof::rdtsc(true);
        let t0 = rd();
        let mut acc: u64 = black_box(0);
        while rd().wrapping_sub(t0) < target {
            acc = acc.wrapping_add(black_box(acc).wrapping_mul(0x9E37_79B9).wrapping_add(1));
        }
        black_box(acc);
    }
}

/// Report ring-injector non-inert counters when the knob is armed. Called from
/// `report_hits`.
pub fn report_ring_inject() {
    if ring_inject_ns() > 0 {
        eprintln!(
            "[slow_knob] RING inject: hits(chunks)={} target_ns_total={} per_chunk_ns={} kind={}",
            RING_INJECT_HITS.load(Ordering::Relaxed),
            RING_INJECT_NS_TOTAL.load(Ordering::Relaxed),
            ring_inject_ns(),
            if yield_kind() { "sleep" } else { "spin" },
        );
    }
}
