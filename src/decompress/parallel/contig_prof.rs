//! P3.1 env-gated cycle profiler for the clean-decode hot loops.
//!
//! `GZIPPY_CONTIG_PROF=1` activates rdtsc iteration-class accounting in BOTH
//! clean engines so one binary can profile either arm via the existing
//! kill-switches:
//!   * `Block::decode_clean_into_contig` (the ONE-engine seeded/fold path,
//!     gzippy-native production) — `C_*` counters.
//!   * `decode_huffman_body_resumable` (the wrapper arm, `GZIPPY_SEEDED_BLOCK=0`)
//!     — `W_*` counters.
//!
//! Method: ONE `rdtsc` per loop iteration, chained — the delta between
//! consecutive iteration tops is attributed to the class of the iteration that
//! just completed (literal-single / literal-pack / back-ref). This keeps the
//! perturbation ~25 cycles/iteration (TSC is invariant, so shares are valid on
//! the frozen box); absolute walls under the knob are NOT comparable to
//! production. OFF (default) = one predicted branch per site, byte-identical
//! output either way (the knob never touches decode state).
//!
//! perf(1) is blocked in the LXC; this is the substitute the P3 arsenal aims
//! with. Dumped from the `GZIPPY_VERBOSE` end-of-decode stats block.
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// ── contig (Block::decode_clean_into_contig) ───────────────────────────────
pub static C_CYC_LIT1: AtomicU64 = AtomicU64::new(0); // single-literal iters
pub static C_CYC_LITPACK: AtomicU64 = AtomicU64::new(0); // multi-literal-pack iters
pub static C_CYC_BACKREF: AtomicU64 = AtomicU64::new(0); // iters ending in a back-ref
pub static C_CYC_CAREFUL: AtomicU64 = AtomicU64::new(0); // whole careful tail loop
pub static C_CYC_CALL: AtomicU64 = AtomicU64::new(0); // whole fn body
pub static C_N_LIT1: AtomicU64 = AtomicU64::new(0);
pub static C_N_LITPACK: AtomicU64 = AtomicU64::new(0);
pub static C_N_BACKREF: AtomicU64 = AtomicU64::new(0);
pub static C_N_CALLS: AtomicU64 = AtomicU64::new(0);
pub static C_N_DIST_LONG: AtomicU64 = AtomicU64::new(0); // dist_hc long-path decodes
pub static C_BYTES_BACKREF: AtomicU64 = AtomicU64::new(0);
pub static C_BYTES_LITPACK: AtomicU64 = AtomicU64::new(0);
pub static C_BYTES_CAREFUL: AtomicU64 = AtomicU64::new(0);

// ── wrapper (decode_huffman_body_resumable) ────────────────────────────────
pub static W_CYC_LIT: AtomicU64 = AtomicU64::new(0); // literal-path iterations
pub static W_CYC_MATCH: AtomicU64 = AtomicU64::new(0); // match-path iterations
pub static W_CYC_CALL: AtomicU64 = AtomicU64::new(0);
pub static W_N_LIT: AtomicU64 = AtomicU64::new(0);
pub static W_N_MATCH: AtomicU64 = AtomicU64::new(0);
pub static W_N_CALLS: AtomicU64 = AtomicU64::new(0);

pub const CLASS_NONE: usize = 0;
pub const CLASS_LIT1: usize = 1;
pub const CLASS_LITPACK: usize = 2;
pub const CLASS_BACKREF: usize = 3;

#[inline(always)]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_CONTIG_PROF").is_ok_and(|v| v == "1"))
}

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

/// Per-call local accumulator flushed to the atomics on Drop (covers every
/// return path of the instrumented fn without touching its control flow).
pub struct ContigFlush {
    pub on: bool,
    pub t_entry: u64,
    pub cyc: [u64; 4], // indexed by CLASS_*; [0] unused
    pub n: [u64; 4],
    pub cyc_careful: u64,
    pub n_lit1: u64,
    pub bytes_backref: u64,
    pub bytes_litpack: u64,
    pub bytes_careful: u64,
}

impl ContigFlush {
    #[inline(always)]
    pub fn new(on: bool) -> Self {
        Self {
            on,
            t_entry: rdtsc(on),
            cyc: [0; 4],
            n: [0; 4],
            cyc_careful: 0,
            n_lit1: 0,
            bytes_backref: 0,
            bytes_litpack: 0,
            bytes_careful: 0,
        }
    }
}

impl Drop for ContigFlush {
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let total = rdtsc(true).wrapping_sub(self.t_entry);
        C_CYC_CALL.fetch_add(total, Ordering::Relaxed);
        C_N_CALLS.fetch_add(1, Ordering::Relaxed);
        C_CYC_LIT1.fetch_add(self.cyc[CLASS_LIT1], Ordering::Relaxed);
        C_CYC_LITPACK.fetch_add(self.cyc[CLASS_LITPACK], Ordering::Relaxed);
        C_CYC_BACKREF.fetch_add(self.cyc[CLASS_BACKREF], Ordering::Relaxed);
        C_N_LIT1.fetch_add(self.n[CLASS_LIT1], Ordering::Relaxed);
        C_N_LITPACK.fetch_add(self.n[CLASS_LITPACK], Ordering::Relaxed);
        C_N_BACKREF.fetch_add(self.n[CLASS_BACKREF], Ordering::Relaxed);
        C_CYC_CAREFUL.fetch_add(self.cyc_careful, Ordering::Relaxed);
        C_BYTES_BACKREF.fetch_add(self.bytes_backref, Ordering::Relaxed);
        C_BYTES_LITPACK.fetch_add(self.bytes_litpack, Ordering::Relaxed);
        C_BYTES_CAREFUL.fetch_add(self.bytes_careful, Ordering::Relaxed);
    }
}

/// Wrapper-loop sibling of [`ContigFlush`].
pub struct WrapperFlush {
    pub on: bool,
    pub t_entry: u64,
    pub cyc_lit: u64,
    pub cyc_match: u64,
    pub n_lit: u64,
    pub n_match: u64,
}

impl WrapperFlush {
    #[inline(always)]
    pub fn new(on: bool) -> Self {
        Self {
            on,
            t_entry: rdtsc(on),
            cyc_lit: 0,
            cyc_match: 0,
            n_lit: 0,
            n_match: 0,
        }
    }
}

impl Drop for WrapperFlush {
    fn drop(&mut self) {
        if !self.on {
            return;
        }
        let total = rdtsc(true).wrapping_sub(self.t_entry);
        W_CYC_CALL.fetch_add(total, Ordering::Relaxed);
        W_N_CALLS.fetch_add(1, Ordering::Relaxed);
        W_CYC_LIT.fetch_add(self.cyc_lit, Ordering::Relaxed);
        W_CYC_MATCH.fetch_add(self.cyc_match, Ordering::Relaxed);
        W_N_LIT.fetch_add(self.n_lit, Ordering::Relaxed);
        W_N_MATCH.fetch_add(self.n_match, Ordering::Relaxed);
    }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

fn per(cyc: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        cyc as f64 / n as f64
    }
}

pub fn dump_if_enabled() {
    if !enabled() {
        return;
    }
    let (cl1, clp, cbr, cca, ctot) = (
        C_CYC_LIT1.load(Ordering::Relaxed),
        C_CYC_LITPACK.load(Ordering::Relaxed),
        C_CYC_BACKREF.load(Ordering::Relaxed),
        C_CYC_CAREFUL.load(Ordering::Relaxed),
        C_CYC_CALL.load(Ordering::Relaxed),
    );
    let (nl1, nlp, nbr, ncall) = (
        C_N_LIT1.load(Ordering::Relaxed),
        C_N_LITPACK.load(Ordering::Relaxed),
        C_N_BACKREF.load(Ordering::Relaxed),
        C_N_CALLS.load(Ordering::Relaxed),
    );
    let classed = cl1 + clp + cbr;
    eprintln!("[contig-prof] CONTIG (Block::decode_clean_into_contig):");
    eprintln!(
        "  calls={} total_cyc={} classed_cyc={} ({:.1}% of total; rest=careful+entry/exit+unchained tail)",
        ncall,
        ctot,
        classed,
        pct(classed, ctot)
    );
    eprintln!(
        "  lit1   : iters={:>12} cyc={:>14} {:>5.1}% of classed, {:>6.1} cyc/iter",
        nl1,
        cl1,
        pct(cl1, classed),
        per(cl1, nl1)
    );
    eprintln!(
        "  litpack: iters={:>12} cyc={:>14} {:>5.1}% of classed, {:>6.1} cyc/iter, lits={}",
        nlp,
        clp,
        pct(clp, classed),
        per(clp, nlp),
        C_BYTES_LITPACK.load(Ordering::Relaxed)
    );
    eprintln!(
        "  backref: iters={:>12} cyc={:>14} {:>5.1}% of classed, {:>6.1} cyc/iter, bytes={} dist_long={}",
        nbr,
        cbr,
        pct(cbr, classed),
        per(cbr, nbr),
        C_BYTES_BACKREF.load(Ordering::Relaxed),
        C_N_DIST_LONG.load(Ordering::Relaxed)
    );
    eprintln!(
        "  careful: cyc={} ({:.1}% of total) outer_iters={}",
        cca,
        pct(cca, ctot),
        C_BYTES_CAREFUL.load(Ordering::Relaxed)
    );
    let (wl, wm, wtot) = (
        W_CYC_LIT.load(Ordering::Relaxed),
        W_CYC_MATCH.load(Ordering::Relaxed),
        W_CYC_CALL.load(Ordering::Relaxed),
    );
    let (wnl, wnm, wnc) = (
        W_N_LIT.load(Ordering::Relaxed),
        W_N_MATCH.load(Ordering::Relaxed),
        W_N_CALLS.load(Ordering::Relaxed),
    );
    let wclassed = wl + wm;
    eprintln!("[contig-prof] WRAPPER (decode_huffman_body_resumable):");
    eprintln!(
        "  calls={} total_cyc={} classed_cyc={} ({:.1}%)",
        wnc,
        wtot,
        wclassed,
        pct(wclassed, wtot)
    );
    eprintln!(
        "  lit    : iters={:>12} cyc={:>14} {:>5.1}% of classed, {:>6.1} cyc/iter",
        wnl,
        wl,
        pct(wl, wclassed),
        per(wl, wnl)
    );
    eprintln!(
        "  match  : iters={:>12} cyc={:>14} {:>5.1}% of classed, {:>6.1} cyc/iter",
        wnm,
        wm,
        pct(wm, wclassed),
        per(wm, wnm)
    );
}
