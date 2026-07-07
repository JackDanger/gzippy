//! Shared `rdtsc` timing helper.
//!
//! This module formerly hosted the `GZIPPY_CONTIG_PROF` env-gated rdtsc
//! cycle-class profiler for the clean-decode hot loops (removed as
//! measurement-stats instrumentation). The `rdtsc` primitive is retained here
//! because the `slow_knob` causal-perturbation injector still uses it to
//! calibrate its turbo-neutral busy spin against a wall-clock window.
#![allow(dead_code)]

/// Read the invariant TSC on x86_64; return 0 on other arches. `on == false`
/// short-circuits to 0 so callers can gate the read without a branch of their
/// own.
#[inline(always)]
pub fn rdtsc(on: bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if on {
            unsafe { core::arch::x86_64::_rdtsc() }
        } else {
            0
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = on;
        0
    }
}
