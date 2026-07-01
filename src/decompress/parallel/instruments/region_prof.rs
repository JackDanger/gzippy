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

// ── R_WORKER SUB-PARTITION (R_WORKER = R_TABLE + R_DECODE + ring_other) ─────────
// Sub-decomposition of the per-chunk decode call to localize the silesia-specific
// instruction surplus (finding F-f81bd0c136c6). Two exclusive sequential leaves +
// an arithmetic residual, MATCHED to rg's readHeader-vs-inner-read-loop split:
//   R_TABLE  : ALL Huffman table materialization — `Block::read_header` (precode +
//              code-lengths + eager litlen LUT) PLUS the lazily-built dist tables
//              (`ensure_dist_hc`/`ensure_dist_table`/`ensure_flat_litlen`), which gz
//              defers into the body. rg builds every table eagerly in readHeader, so
//              to keep the leaf MATCHED gz must claw those nested builds back out of
//              the decode loop — done via TL_TABLE_NESTED (cursor-agent Q2 fix).
//   R_DECODE : the decode body (`Block::read` ring path + `decode_clean_into_contig`
//              contig path), incl. interleaved clean-CRC (cursor-agent Q1: fold), but
//              EXCLUSIVE of the nested table builds (subtracted at span exit).
//   ring_other = R_WORKER - R_TABLE - R_DECODE (chunk setup, fold-drain, finalize).
// Clean-vs-marker is intentionally NOT split: gz flips a window-absent chunk to clean
// mid-chunk while rg keeps it fully markered, so a clean/marker split is NOT matched
// gz-vs-rg and would mislabel (cursor-agent Q3/Q6). marker decode is folded into
// R_DECODE for conservation (it is already ACQUITTED faster-than-rg).
// Own depth counter (SUB_OVERLAP) — the leaves nest INSIDE the outer R_WORKER span,
// so they cannot share R_WORKER's depth counter.
pub static R_TABLE_CYC: AtomicU64 = AtomicU64::new(0);
pub static R_TABLE_N: AtomicU64 = AtomicU64::new(0); // read_header calls (cyc/header diagnostic)
pub static R_TABLE_NESTED_CYC: AtomicU64 = AtomicU64::new(0); // dist builds clawed from body
pub static R_DECODE_CYC: AtomicU64 = AtomicU64::new(0);
pub static R_DECODE_N: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SUB_DEPTH: Cell<u32> = const { Cell::new(0) };
    // Cumulative cycles spent in nested table builds on THIS thread; a decode span
    // snapshots it at entry and credits (exit-entry) back out of R_DECODE.
    static TL_TABLE_NESTED: Cell<u64> = const { Cell::new(0) };
}
pub static SUB_OVERLAP_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
fn sub_enter() {
    SUB_DEPTH.with(|d| {
        let v = d.get();
        if v > 0 {
            SUB_OVERLAP_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
        }
        d.set(v + 1);
    });
}
#[inline(always)]
fn sub_exit() {
    SUB_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
}

/// RAII span over `Block::read_header` → R_TABLE (exclusive leaf).
pub struct TableHeaderSpan {
    on: bool,
    t0: u64,
}
impl TableHeaderSpan {
    #[inline(always)]
    pub fn new() -> Self {
        let on = enabled();
        if on {
            sub_enter();
        }
        Self { on, t0: rdtsc(on) }
    }
}
impl Default for TableHeaderSpan {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for TableHeaderSpan {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let dt = rdtsc(true).wrapping_sub(self.t0);
        sub_exit();
        R_TABLE_CYC.fetch_add(dt, Ordering::Relaxed);
        R_TABLE_N.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII span over a decode-body call (`Block::read` / `decode_clean_into_contig`).
/// Credits to R_DECODE the body cycles MINUS the nested table-build cycles that
/// fired during this span (kept exclusive of R_TABLE).
pub struct DecodeSpan {
    on: bool,
    t0: u64,
    nested0: u64,
}
impl DecodeSpan {
    #[inline(always)]
    pub fn new() -> Self {
        let on = enabled();
        if on {
            sub_enter();
        }
        let nested0 = if on {
            TL_TABLE_NESTED.with(|c| c.get())
        } else {
            0
        };
        Self {
            on,
            t0: rdtsc(on),
            nested0,
        }
    }
}
impl Default for DecodeSpan {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for DecodeSpan {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let dt = rdtsc(true).wrapping_sub(self.t0);
        sub_exit();
        let nested = TL_TABLE_NESTED.with(|c| c.get()).wrapping_sub(self.nested0);
        // body cycles exclusive of nested table builds
        R_DECODE_CYC.fetch_add(dt.wrapping_sub(nested), Ordering::Relaxed);
        R_DECODE_N.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII timer for a nested table-build function (`ensure_dist_hc`/
/// `ensure_dist_table`/`ensure_flat_litlen`): on drop add dt to R_TABLE and to the
/// per-thread nested accumulator so the enclosing DecodeSpan subtracts it (keeps
/// R_TABLE/R_DECODE exclusive). Does NOT touch the sub-depth counter — it is an
/// accounting transfer, not an exclusive leaf.
pub struct NestedTableSpan {
    on: bool,
    t0: u64,
}
impl NestedTableSpan {
    #[inline(always)]
    pub fn new() -> Self {
        let on = enabled();
        Self { on, t0: rdtsc(on) }
    }
}
impl Default for NestedTableSpan {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for NestedTableSpan {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let dt = rdtsc(true).wrapping_sub(self.t0);
        R_TABLE_CYC.fetch_add(dt, Ordering::Relaxed);
        R_TABLE_NESTED_CYC.fetch_add(dt, Ordering::Relaxed);
        TL_TABLE_NESTED.with(|c| c.set(c.get().wrapping_add(dt)));
    }
}

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
    // ── R_WORKER sub-partition dump ────────────────────────────────────────
    let (tc, tn, tnest) = (
        load(&R_TABLE_CYC),
        load(&R_TABLE_N),
        load(&R_TABLE_NESTED_CYC),
    );
    let (dc, dn) = (load(&R_DECODE_CYC), load(&R_DECODE_N));
    let ring_other = wc.saturating_sub(tc).saturating_sub(dc);
    let cpcall = |c: u64, n: u64| if n > 0 { c as f64 / n as f64 } else { 0.0 };
    eprintln!(
        "  [subregion-prof] R_WORKER sub-partition (R_WORKER = R_TABLE + R_DECODE + ring_other):"
    );
    eprintln!(
        "    R_TABLE  cyc={:>16} hdr_calls={:>9} nested_dist_cyc={:>14} cyc/B={:.3} cyc/hdr={:.1}  (read_header + ensure_dist_*)",
        tc, tn, tnest, cpb(tc, wb), cpcall(tc, tn)
    );
    eprintln!(
        "    R_DECODE cyc={:>16} calls={:>9} cyc/B={:.3}  (Block::read + decode_clean_into_contig, excl. nested table, incl. clean-CRC)",
        dc, dn, cpb(dc, wb)
    );
    eprintln!(
        "    ring_other cyc={:>14} (= R_WORKER - R_TABLE - R_DECODE; chunk setup/fold-drain/finalize) cyc/B={:.3}",
        ring_other, cpb(ring_other, wb)
    );
    let sub_overlap = SUB_OVERLAP_VIOLATIONS.load(Ordering::Relaxed);
    let sub_sum = tc + dc + ring_other;
    let sub_non_inert = tn > 0 && dn > 0;
    eprintln!(
        "    [subregion-prof] SUB_OVERLAP_VIOLATIONS={sub_overlap} (must be 0) conservation R_TABLE+R_DECODE+ring_other={sub_sum} vs R_WORKER={wc}"
    );
    eprintln!(
        "    [subregion-prof] SELF-TEST: sub_overlap==0:{} non-inert(table&decode):{} ring_other>=0:{} -> {}",
        sub_overlap == 0,
        sub_non_inert,
        wc >= tc + dc,
        if sub_overlap == 0 && sub_non_inert && wc >= tc + dc { "PASS" } else { "FAIL" }
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
