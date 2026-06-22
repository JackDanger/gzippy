//! AMD-residual re-attribution region profiler (`GZIPPY_REGION_PROF=1`,
//! byte-transparent). NO rapidgzip counterpart in shape, but DESIGNED to be
//! SYMMETRIC with rg counters (rg_region_prof_patch.py) so a gz-vs-rg ABSOLUTE
//! cycle-delta attribution can be computed per region (the cursor-agent-reviewed
//! AMD-RESIDUAL-ATTRIB design).
//!
//! Coarse, EXCLUSIVE, leaf top-level rdtsc(TSC) spans summed over ALL worker +
//! consumer threads (one rdtsc pair per chunk/per-write — negligible perturbation
//! vs MB-sized chunks; NO per-iteration perturbation). The regions partition the
//! decode-relevant cycle budget:
//!   R_WORKER  : the whole per-chunk decode CALL (huffman decode + commit + clean-
//!               CRC + ring) — measured at the 4 worker call sites of decode_chunk*.
//!               (decode itself is ACQUITTED faster-than-rg; this region's D should
//!               be <=0; it exists to PROVE the excess is NOT here.)
//!   R_MARKERPP: marker post-process (resolve_and_narrow + narrowed-CRC + subchunk
//!               window populate) — rg counterpart = ChunkData::applyWindow.
//!   R_OUTPUT  : consumer output path (iovec assembly + CRC-combine + writev) — rg
//!               counterpart = the write path.
//! R_OTHER (pipeline/blockfinder/windowmap/pool/alloc) = perf_total - sum(above),
//! computed externally; valid only when the box is FROZEN so TSC ~= core cycles
//! (gov=performance + boost=0; the driver does that, NO llama pause).
//!
//! Verdict criterion (per the falsifier): ABSOLUTE D_r = gz_cyc(region) -
//! rg_cyc(region). cyc/B + bytes are diagnostic only.
#![allow(dead_code)]

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// ── Mutual-exclusion (overlap) invariant (cursor-agent review fix #1) ──────────
// The three regions must be EXCLUSIVE non-nested leaves. If any region span is
// entered while another is already active on this thread, conservation could pass
// SPURIOUSLY (one region over-captures while another under-captures by the same
// amount, so R_OTHER stays small). We detect that at runtime: a per-thread depth
// counter increments on region enter; entering at depth>0 is an overlap VIOLATION.
// A healthy partition run reports OVERLAP_VIOLATIONS == 0.
thread_local! {
    static REGION_DEPTH: Cell<u32> = const { Cell::new(0) };
}
pub static OVERLAP_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn region_enter() {
    REGION_DEPTH.with(|d| {
        let v = d.get();
        if v > 0 {
            OVERLAP_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
        }
        d.set(v + 1);
    });
}

#[inline(always)]
pub fn region_exit() {
    REGION_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
}

pub static R_WORKER_CYC: AtomicU64 = AtomicU64::new(0);
pub static R_WORKER_N: AtomicU64 = AtomicU64::new(0);
pub static R_WORKER_BYTES: AtomicU64 = AtomicU64::new(0);

pub static R_MARKERPP_CYC: AtomicU64 = AtomicU64::new(0);
pub static R_MARKERPP_N: AtomicU64 = AtomicU64::new(0);
pub static R_MARKERPP_BYTES: AtomicU64 = AtomicU64::new(0); // marker bytes resolved

pub static R_OUTPUT_CYC: AtomicU64 = AtomicU64::new(0);
pub static R_OUTPUT_N: AtomicU64 = AtomicU64::new(0);
pub static R_OUTPUT_BYTES: AtomicU64 = AtomicU64::new(0); // payload bytes written

#[inline(always)]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_REGION_PROF").is_ok_and(|v| v == "1"))
}

#[inline(always)]
pub fn rdtsc(on: bool) -> u64 {
    crate::decompress::parallel::instruments::contig_prof::rdtsc(on)
}

/// RAII span: measures wall-rdtsc of the enclosing scope on THIS thread and adds
/// it to (cyc, n). Drop-based so every early-return path of the wrapped region is
/// covered. Exclusive: place at a top-level region boundary, never nested inside
/// another region's span.
pub struct Span {
    on: bool,
    t0: u64,
    cyc: &'static AtomicU64,
    n: &'static AtomicU64,
}

impl Span {
    #[inline(always)]
    pub fn new(cyc: &'static AtomicU64, n: &'static AtomicU64) -> Self {
        let on = enabled();
        if on {
            region_enter();
        }
        Self {
            on,
            t0: rdtsc(on),
            cyc,
            n,
        }
    }
}

impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let dt = rdtsc(true).wrapping_sub(self.t0);
        region_exit();
        self.cyc.fetch_add(dt, Ordering::Relaxed);
        self.n.fetch_add(1, Ordering::Relaxed);
    }
}

/// Measure a closure's rdtsc into (cyc,n), also crediting `bytes`. Exact (no scope
/// leakage). Used at the worker decode call sites.
#[inline(always)]
pub fn measure_worker<T>(bytes: u64, f: impl FnOnce() -> T) -> T {
    let on = enabled();
    if !on {
        return f();
    }
    region_enter();
    let t0 = rdtsc(true);
    let r = f();
    let dt = rdtsc(true).wrapping_sub(t0);
    region_exit();
    R_WORKER_CYC.fetch_add(dt, Ordering::Relaxed);
    R_WORKER_N.fetch_add(1, Ordering::Relaxed);
    R_WORKER_BYTES.fetch_add(bytes, Ordering::Relaxed);
    r
}

pub fn dump_if_enabled() {
    if !enabled() {
        return;
    }
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let cpb = |c: u64, b: u64| if b > 0 { c as f64 / b as f64 } else { 0.0 };
    let (wc, wn, wb) = (
        load(&R_WORKER_CYC),
        load(&R_WORKER_N),
        load(&R_WORKER_BYTES),
    );
    let (mc, mn, mb) = (
        load(&R_MARKERPP_CYC),
        load(&R_MARKERPP_N),
        load(&R_MARKERPP_BYTES),
    );
    let (oc, on_, ob) = (
        load(&R_OUTPUT_CYC),
        load(&R_OUTPUT_N),
        load(&R_OUTPUT_BYTES),
    );
    eprintln!("[region-prof] AMD-residual region partition (GZIPPY_REGION_PROF=1; TSC cycles):");
    eprintln!(
        "  R_WORKER   cyc={:>16} calls={:>9} bytes={:>12} cyc/B={:.3}  (decode+commit+clean-crc+ring; ACQUITTED-faster, expect D<=0)",
        wc, wn, wb, cpb(wc, wb)
    );
    eprintln!(
        "  R_MARKERPP cyc={:>16} calls={:>9} mkbytes={:>11} cyc/mkB={:.3}  (resolve+narrow+narrowed-crc+subwin == rg applyWindow)",
        mc, mn, mb, cpb(mc, mb)
    );
    eprintln!(
        "  R_OUTPUT   cyc={:>16} calls={:>9} bytes={:>12} cyc/B={:.3}  (iovec+crc-combine+writev == rg write)",
        oc, on_, ob, cpb(oc, ob)
    );
    eprintln!(
        "  [region-prof] sum(R_WORKER+R_MARKERPP+R_OUTPUT)={} ; R_OTHER = perf_total - this (frozen box, externally)",
        wc + mc + oc
    );
    eprintln!(
        "  [region-prof] non-inert: WORKER={} MARKERPP={} OUTPUT={} (all must be >0)",
        wn, mn, on_
    );
    let overlap = OVERLAP_VIOLATIONS.load(Ordering::Relaxed);
    let non_inert = wn > 0 && mn > 0 && on_ > 0;
    eprintln!(
        "  [region-prof] OVERLAP_VIOLATIONS={overlap} (must be 0 — proves regions are exclusive non-nested leaves)"
    );
    eprintln!(
        "  [region-prof] SELF-TEST: overlap==0:{} non-inert:{} -> {}  env_fingerprint(GZIPPY_REGION_PROF=1)",
        overlap == 0,
        non_inert,
        if overlap == 0 && non_inert { "PASS" } else { "FAIL" }
    );
}
