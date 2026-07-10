//! Gate-2 causal-perturbation instrument — feature `perturb` (OFF by
//! default; ZERO effect on the production build/bytes/timing).
//!
//! Built for the storedheavy sub-cause brief (2026-07-09) STEP 2: an
//! advisor (cursor-agent Composer) noted that "decode_wait dominates the
//! wall" only forces the lever to be WORKER per-thread decode throughput —
//! it does NOT distinguish (H) Huffman/bit-reader COMPUTE from (M) the u16
//! ring-store memory-BANDWIDTH from (S) scheduling/granularity. This module
//! is the two-arm discriminator: it injects a KNOWN, calibrated delay at one
//! of two exact sites in `marker_inflate.rs`'s `decode_marker_fast_loop`
//! (selected + sized via env, read once and cached):
//!
//!   - Arm H (`GZIPPY_PERTURB_ARM=H`): injected AFTER the litlen
//!     `lut.decode`/`consume` completes, BEFORE the u16 ring write — pure
//!     added compute between decode and store.
//!   - Arm M (`GZIPPY_PERTURB_ARM=M`): injected immediately AFTER the
//!     widened u16 ring store (the `write_unaligned` in the fast loop) —
//!     pure added cost attached to the store itself.
//!
//! `GZIPPY_PERTURB_NS` sets the target per-injection-site delay in
//! nanoseconds (e.g. run once at a small magnitude, once at ~10x). The
//! delay is realized as a calibrated count of `std::hint::spin_loop()`
//! iterations (PAUSE-instruction spin on x86, YIELD on aarch64) — NOT a
//! naive busy loop and NOT an OS-yielding sleep: at this per-symbol
//! (tens-of-nanoseconds-per-iteration) granularity a real syscall-based
//! sleep's own overhead would dwarf and distort the injected magnitude, and
//! a plain tight loop can be optimized away or cause disproportionate
//! frequency/thermal effects. `spin_loop()` keeps the core busy (same
//! power/frequency profile as the real work it stands in for) while still
//! being a documented, portable "this is a spin-wait" hint — the standard
//! primitive for exactly this kind of fine-grained calibrated injection.
//!
//! Calibration: measured ONCE per process (first call), by timing a fixed
//! count of `spin_loop()` iterations with `Instant`, giving a robust
//! ns/iteration constant used to convert `GZIPPY_PERTURB_NS` to an
//! iteration count. No rg counterpart — measurement-only.
//!
//! Every symbol here compiles to a true no-op under `#[cfg(not(feature =
//! "perturb"))]` — same shape as `phase_timing.rs` / `coz_probe.rs`.
#![cfg_attr(not(feature = "perturb"), allow(dead_code))]

#[cfg(feature = "perturb")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "perturb")]
use std::sync::OnceLock;

/// Gate-0 self-validation counter (CLAUDE.md instrument rule (c): "any
/// oracle/perturbation run produces output that PROVABLY DIFFERS from the
/// baseline (counter fired, hits>0) — else it is inert and is silently
/// measuring the normal path"). Incremented once per actually-executed
/// injection (i.e. only when the call site's arm matches the env-selected
/// arm AND `iters>0`) — NOT once per `maybe_inject` call (which would count
/// every loop iteration of both arms regardless of firing). Read via
/// [`hits`] and printed by the caller so a run can be checked for hits==0
/// (inert — do not trust the wall number) before any Δwall/Δinject claim.
#[cfg(feature = "perturb")]
static HITS: AtomicU64 = AtomicU64::new(0);

/// No arm selected — `maybe_inject` is a no-op regardless of call site.
#[cfg(feature = "perturb")]
pub const ARM_NONE: u8 = 0;
/// Compute arm: injected after litlen decode/consume, before the ring write.
#[cfg(feature = "perturb")]
pub const ARM_H: u8 = 1;
/// Bandwidth arm: injected immediately after the u16 ring store.
#[cfg(feature = "perturb")]
pub const ARM_M: u8 = 2;

#[cfg(feature = "perturb")]
static CONFIG: OnceLock<(u8, u64)> = OnceLock::new();

/// Read `GZIPPY_PERTURB_ARM` / `GZIPPY_PERTURB_NS` once, cache `(arm,
/// spin_iters)`. A malformed/absent env value degrades to `(ARM_NONE, 0)`
/// (the injection is silently absent, matching this project's "no-op on
/// miss" convention for diagnostic env knobs) — the measurement script is
/// responsible for confirming the arm actually fired (Gate-0 self-test: see
/// `perturb_smoke_test` below, which asserts a nonzero measured wall delta
/// at a large injected magnitude).
#[cfg(feature = "perturb")]
fn config() -> (u8, u64) {
    *CONFIG.get_or_init(|| {
        let arm = match std::env::var("GZIPPY_PERTURB_ARM").ok().as_deref() {
            Some("H") => ARM_H,
            Some("M") => ARM_M,
            _ => ARM_NONE,
        };
        let ns: u64 = std::env::var("GZIPPY_PERTURB_NS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let iters = if arm == ARM_NONE || ns == 0 {
            0
        } else {
            ns_to_spin_iters(ns)
        };
        (arm, iters)
    })
}

/// Calibrate ns-per-`spin_loop()`-iteration once, convert `ns` to an
/// iteration count (minimum 1 whenever `ns > 0`, so a small requested delay
/// is never silently rounded to zero injections).
#[cfg(feature = "perturb")]
fn ns_to_spin_iters(ns: u64) -> u64 {
    static NS_PER_ITER: OnceLock<f64> = OnceLock::new();
    let per_iter = *NS_PER_ITER.get_or_init(|| {
        const CAL_ITERS: u64 = 4_000_000;
        let t0 = std::time::Instant::now();
        for _ in 0..CAL_ITERS {
            std::hint::spin_loop();
        }
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        (elapsed_ns / CAL_ITERS as f64).max(0.001)
    });
    (((ns as f64) / per_iter).round() as u64).max(1)
}

/// Call at a named injection site. Spins the calibrated iteration count IFF
/// `arm` matches the env-selected arm; a true no-op (single relaxed
/// `OnceLock` read, no branch taken) otherwise.
#[cfg(feature = "perturb")]
#[inline(always)]
pub fn maybe_inject(arm: u8) {
    let (selected_arm, iters) = config();
    if selected_arm == arm && iters > 0 {
        HITS.fetch_add(1, Ordering::Relaxed);
        for _ in 0..iters {
            std::hint::spin_loop();
        }
    }
}
#[cfg(not(feature = "perturb"))]
#[inline(always)]
pub fn maybe_inject(_arm: u8) {}

/// Total number of injections that actually fired (see [`HITS`] doc). `0`
/// whenever no arm is selected (`GZIPPY_PERTURB_ARM` unset/invalid) or the
/// selected arm's call site was never reached.
#[cfg(feature = "perturb")]
pub fn hits() -> u64 {
    HITS.load(Ordering::Relaxed)
}
#[cfg(not(feature = "perturb"))]
pub fn hits() -> u64 {
    0
}

/// The env-selected arm (`ARM_NONE`/`ARM_H`/`ARM_M`) and the calibrated
/// spin-iteration count per injection, for self-test reporting.
#[cfg(feature = "perturb")]
pub fn selected_config() -> (u8, u64) {
    config()
}
#[cfg(not(feature = "perturb"))]
pub fn selected_config() -> (u8, u64) {
    (ARM_NONE, 0)
}
#[cfg(not(feature = "perturb"))]
pub const ARM_NONE: u8 = 0;

#[cfg(all(test, feature = "perturb"))]
mod tests {
    use super::*;

    /// Gate-0 self-test: a large calibrated injection count actually costs
    /// wall time proportional to the requested `ns` (within a generous
    /// factor — this is a coarse smoke test, not the precision instrument;
    /// the real measurement is the interleaved `fulcrum score` run). Proves
    /// the calibration path is not inert (e.g. always returning 0 iters).
    #[test]
    fn calibrated_spin_costs_measurable_wall_time() {
        let iters = ns_to_spin_iters(2_000_000); // ~2ms requested
        assert!(iters > 0, "calibration produced zero iterations for 2ms");
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::spin_loop();
        }
        let elapsed = t0.elapsed();
        // Generous bounds: real hardware will land close to 2ms; require
        // only that it's within [0.2ms, 40ms] to avoid flaking under CI
        // contention while still proving the injection is not inert (an
        // inert path would cost ~0ms).
        assert!(
            elapsed.as_micros() > 200,
            "injected spin cost only {:?}, suspiciously close to inert",
            elapsed
        );
        assert!(
            elapsed.as_millis() < 40,
            "injected spin cost {:?}, wildly over calibration target",
            elapsed
        );
    }
}
