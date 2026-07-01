//! Thin shim over `fulcrum::probe` for Coz causal profiling.
//!
//! Under `--features coz`, `scope`/`progress` forward to `fulcrum::probe`,
//! which (when the binary is built with coz and run under `coz run`) emits
//! Coz latency counters per region and a throughput counter at the in-order
//! consumer's per-chunk emit. That lets `fulcrum rank` fuse measured
//! wall-elasticity (∂wall/∂speed) with the critical-path share.
//!
//! Without the feature there is no `fulcrum` dependency at all and every call
//! compiles to a no-op (the `Scope` is a zero-size guard), so the production
//! decode path is untouched.

#[cfg(feature = "coz")]
pub use fulcrum::probe::{progress, scope, Scope};

// Without the `coz` feature these are no-ops. Their only callers are the
// parallel-SM decode probes (`#[cfg(parallel_sm)]`), so under the default
// feature set (no parallel_sm) they have no callers — allow that rather than
// fail `-D warnings`.
#[cfg(not(feature = "coz"))]
#[allow(dead_code)]
mod noop {
    /// Zero-size RAII guard; drops with no effect.
    #[must_use]
    pub struct Scope;
    #[inline(always)]
    pub fn scope(_name: &'static str) -> Scope {
        Scope
    }
    #[inline(always)]
    pub fn progress(_name: &'static str) {}
}

#[cfg(not(feature = "coz"))]
#[allow(unused_imports)]
pub use noop::{progress, scope, Scope};
