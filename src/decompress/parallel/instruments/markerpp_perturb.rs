//! Marker post-process REMOVAL-ORACLE knobs (Gate-2 sub-component bounding).
//!
//! NO rapidgzip counterpart. Env-gated, byte-transparent in production (default
//! OFF). When ON they SKIP a sub-component of `resolve_chunk_markers_on_chunk`
//! and therefore produce GARBAGE output by design — these are BOUNDING oracles
//! only (the cyc/B drop bounds that sub-component's cost), never a shippable
//! path. Each carries a FIRED counter (>0 proves non-inert).
//!
//! The two leaves match the R_MARKERPP rdtsc sub-partition in `region_prof`:
//!   GZIPPY_PERTURB_SKIP_RESOLVE=1     → skip `resolve_and_narrow_markers_in_place_crc`
//!                                       (the per-output-byte resolve+narrow+CRC).
//!   GZIPPY_PERTURB_SKIP_APPLYWINDOW=1 → skip the real per-subchunk window
//!                                       copy+mask+compress in `populate_subchunk_windows`;
//!                                       a cheap shared zero-window is substituted so the
//!                                       window-publish chain does not starve (the
//!                                       process still completes; output garbage).
//!
//! The two skips are INDEPENDENT, so the matrix {baseline, skip-resolve,
//! skip-applywindow, skip-both} attributes each leaf's clean increment.

#![cfg(parallel_sm)]
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub static SKIP_RESOLVE_FIRED: AtomicU64 = AtomicU64::new(0);
pub static SKIP_SUBWIN_FIRED: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn skip_resolve_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GZIPPY_PERTURB_SKIP_RESOLVE").is_some())
}

#[inline(always)]
pub fn skip_subwin_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GZIPPY_PERTURB_SKIP_APPLYWINDOW").is_some())
}

#[inline(always)]
pub fn note_skip_resolve() {
    SKIP_RESOLVE_FIRED.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub fn note_skip_subwin() {
    SKIP_SUBWIN_FIRED.fetch_add(1, Ordering::Relaxed);
}

/// Loud Gate-0 self-report. Only prints when at least one knob is set, so the
/// production path stays silent.
pub fn report_if_enabled() {
    let sr = skip_resolve_enabled();
    let sw = skip_subwin_enabled();
    if !sr && !sw {
        return;
    }
    let rf = SKIP_RESOLVE_FIRED.load(Ordering::Relaxed);
    let wf = SKIP_SUBWIN_FIRED.load(Ordering::Relaxed);
    eprintln!(
        "[markerpp-perturb] BOUNDING-ORACLE (garbage output by design): skip_resolve={sr} fired={rf} | skip_applywindow={sw} fired={wf}"
    );
    let ok = (!sr || rf > 0) && (!sw || wf > 0);
    eprintln!(
        "[markerpp-perturb] SELF-TEST (enabled knob must fire >0): {}",
        if ok { "PASS" } else { "FAIL (INERT — number is invalid)" }
    );
}
