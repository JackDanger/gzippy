#![allow(dead_code)]
// probe surface is live on x86_64+pure-rust-inflate
// (the parallel SM decode path); dead on arm64/mac
// where that path is cfg'd out — same as trace_v2.rs.

//! FULCRUM causal-profiler probe points.
//!
//! FULCRUM (`tools/fulcrum/`) is gzippy's causal-mechanistic pipeline
//! profiler. Standard profilers report CPU-time SUMS, which LIE for a
//! parallel pipeline gated by a critical path: a 212 ms `absorb_isal_tail`
//! memcpy was eliminated and the wall moved 0.0 % because the copy was
//! overlapped / off the in-order consumer's critical path. FULCRUM uses
//! Coz-style *virtual speedup* to measure each region's wall-elasticity
//! (∂program-speedup / ∂region-speedup) — the only metric that would have
//! told us absorb is a non-lever and the bootstrap is.
//!
//! This module is the gzippy-side hook surface. It expands to NOTHING
//! unless the `fulcrum` feature is on, so production builds pay zero. When
//! on, it inserts:
//!   - ONE throughput progress point [`chunk_emitted`] at the in-order
//!     consumer emit — Coz's "unit of work complete" marker. Wall
//!     throughput == visit rate to this point.
//!   - Named latency SCOPES around the four candidate levers four agents
//!     have circled but not discriminated: [`Region::Bootstrap`] (the slow
//!     window-absent pure-Rust bootstrap), [`Region::BulkInflate`] (the
//!     clean ISA-L-parity bulk decode), [`Region::Absorb`] (the
//!     stitch/append copies — the known non-lever), [`Region::Scan`]
//!     (speculation / block-boundary scan). Coz reports, per scope, the
//!     effect of virtually speeding that scope on the chunk_emitted rate.
//!
//! Why named scopes and not just line-level Coz: the production binary is
//! built with `lto = "fat"`, which inlines bootstrap / bulk / absorb into
//! overlapping address ranges — line-level attribution would smear them.
//! Coz's `begin!`/`end!` counters are keyed by NAME and survive inlining,
//! so the four regions stay separable. See `tools/fulcrum/README` (design
//! note) and `[profile.fulcrum]` in Cargo.toml (keeps line tables).
//!
//! Coz linkage: the `coz` crate is dlsym-based — the macros look up
//! `_coz_get_counter` / `_coz_add_delays` at run time. Outside `coz run`
//! the lookup fails gracefully and every probe is a cheap no-op, so a
//! `--features fulcrum` binary still runs normally (e.g. under `perf` or
//! `interleaved_ab` for the empirical cross-check) — only `coz run`
//! activates the experiments.

/// The four candidate lever regions FULCRUM discriminates. The string
/// names are the Coz counter identifiers; keep them stable — the
/// `tools/fulcrum` analyzer matches `profile.coz` rows by these names.
#[derive(Clone, Copy, Debug)]
pub enum Region {
    /// `bootstrap_with_deflate_block` — the slow window-absent pure-Rust
    /// bootstrap decode that gates the ~7 heavy "overshoot" chunks.
    Bootstrap,
    /// `decode_block` bulk phase — the clean ISA-L-parity inflate.
    BulkInflate,
    /// `absorb_isal_tail` + append stitch copies — the KNOWN non-lever
    /// (212 ms eliminated, 0.0 % wall). FULCRUM must reproduce ≈0
    /// elasticity here, else FULCRUM is wrong.
    Absorb,
    /// Speculative block-boundary scan / candidate evaluation.
    Scan,
}

impl Region {
    /// Stable Coz counter name. MUST match the analyzer's expectations.
    #[inline]
    pub const fn coz_name(self) -> &'static str {
        match self {
            Region::Bootstrap => "fulcrum.bootstrap",
            Region::BulkInflate => "fulcrum.bulk_inflate",
            Region::Absorb => "fulcrum.absorb",
            Region::Scan => "fulcrum.scan",
        }
    }
}

/// RAII latency-scope guard. On construction emits a Coz `begin` for the
/// region; on Drop emits the matching `end` — even on early return / `?`,
/// which is exactly why this is RAII and not a manual begin/end pair (the
/// decode paths are riddled with `?`). No-op (zero-sized, no Drop work)
/// unless the `fulcrum` feature is active.
#[must_use = "the scope ends when this guard drops; bind it to a named local"]
pub struct RegionScope {
    #[cfg(feature = "fulcrum")]
    name: &'static str,
}

impl RegionScope {
    /// Open a latency scope for `region`. Bind the returned guard to a
    /// local (`let _f = ...;`) so it lives for the region's duration.
    #[inline(always)]
    pub fn enter(region: Region) -> Self {
        #[cfg(feature = "fulcrum")]
        {
            let name = region.coz_name();
            coz::Counter::begin(name).increment();
            return RegionScope { name };
        }
        #[cfg(not(feature = "fulcrum"))]
        {
            let _ = region;
            RegionScope {}
        }
    }
}

#[cfg(feature = "fulcrum")]
impl Drop for RegionScope {
    #[inline(always)]
    fn drop(&mut self) {
        coz::Counter::end(self.name).increment();
    }
}

/// Mark the completion of one unit of pipeline output — the in-order
/// consumer emitting the next chunk's bytes. Coz measures the visit rate
/// to this point as the program's throughput; virtual-speedup experiments
/// report their effect as a change in THIS rate. Place it at the END of a
/// successful in-order emit, once per emitted chunk. No-op without
/// `fulcrum`.
#[inline(always)]
pub fn chunk_emitted() {
    #[cfg(feature = "fulcrum")]
    {
        coz::progress!("chunk_emitted");
    }
}

/// Convenience: open a [`RegionScope`]. Mirrors the ergonomics of the
/// `trace_v2_span!` macro already used at these call sites.
///
/// ```ignore
/// let _f = fulcrum_scope!(Region::Bootstrap);
/// ```
#[macro_export]
macro_rules! fulcrum_scope {
    ($region:expr) => {
        $crate::fulcrum_probe::RegionScope::enter($region)
    };
}
