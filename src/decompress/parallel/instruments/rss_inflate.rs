//! STARTUP RSS-INFLATION ORACLE (`GZIPPY_RSS_INFLATE_MIB=<N>`, byte-transparent).
//!
//! Leg-2 ceiling oracle for the AMD-T2 "process teardown ∝ peak RSS" finding
//! ([[project_amd_t2_phase_and_t4_floor_2026_06_22]]): the gz-vs-rapidgzip T2
//! wall excess was located (Gate-2, Zen2) as kernel address-space TEARDOWN of
//! gz's elevated peak RSS at `process::exit`, MONOTONIC+PROPORTIONAL at
//! +0.054 ms/MiB. To BOUND the teardown prize on a second arch WITHOUT touching
//! the decode path (and WITHOUT reusing the FALSIFIED `GZIPPY_FREE_MARKERS`,
//! which traded RSS for throughput), this knob INFLATES peak RSS by a known N
//! MiB of resident-but-unused memory and holds it to `_exit`. Sweeping N over
//! {0, 20, 40, 60} and measuring full process wall on a fixed corpus/T gives the
//! teardown slope directly: the decode is bit-identical across the sweep, so the
//! wall delta is purely the extra teardown (plus any first-touch fault cost,
//! which is paid up-front BEFORE the timed decode region — see below). Invert the
//! slope at ΔRSS ≈ (gz_peak − rg_peak) to bound the eviction-port prize.
//!
//! ## Mechanics / why it is sound
//! * Allocates `N MiB` as a `Vec<u8>` and TOUCHES one byte per 4 KiB page so the
//!   pages are RESIDENT (RSS, not just virtual) — an untouched allocation does
//!   not raise RSS and would not exercise teardown. A running XOR of the touched
//!   bytes is printed in the banner so the optimizer cannot elide the writes
//!   (non-inert proof) and so residency is observable.
//! * The allocation + first-touch happens at the TOP of `main` (call site in
//!   `main.rs`, right after `main_start`), i.e. BEFORE `run()` and therefore
//!   before the timed decode/parallel region. The fault-in cost is in the PRE
//!   span, not in decode; only the resident footprint persists into teardown.
//! * The buffer is `mem::forget`-leaked so it stays mapped until the kernel tears
//!   the address space down at `_exit` (gz already calls `process::exit`, which
//!   skips destructors — same teardown shape the finding measured).
//! * Byte-transparent: decode output is unchanged; `N` unset ⇒ a single
//!   `OnceLock`-resolved `None` branch, zero allocation.
//!
//! OFF by default. This is a measurement-only instrument with NO rapidgzip
//! counterpart; it never runs on the production path.

#![allow(dead_code)]

use std::sync::OnceLock;

/// Parsed `GZIPPY_RSS_INFLATE_MIB` (MiB to pin resident), resolved once.
fn inflate_mib() -> Option<usize> {
    static M: OnceLock<Option<usize>> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("GZIPPY_RSS_INFLATE_MIB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// Inflate peak RSS by `N` MiB of resident memory and hold it to `_exit`.
/// No-op unless `GZIPPY_RSS_INFLATE_MIB=<N>` (N>0). Call once at the top of
/// `main`, before the timed region. Prints a non-inert banner (size + touched
/// XOR) so the run is provably non-inert and the residency is observable.
pub fn engage() {
    let Some(mib) = inflate_mib() else { return };
    let bytes = mib.saturating_mul(1024 * 1024);
    // Allocate and first-touch one byte per 4 KiB page to force residency.
    let mut buf: Vec<u8> = vec![0u8; bytes];
    const PAGE: usize = 4096;
    let mut xor: u8 = 0;
    let mut i = 0usize;
    while i < bytes {
        // Write a non-trivial value derived from the index so the store cannot
        // be folded to a constant memset the allocator might lazy-zero.
        let v = (i as u8) ^ 0xA5;
        buf[i] = v;
        xor ^= v;
        i += PAGE;
    }
    eprintln!(
        "\n████ RSS-INFLATE ACTIVE — pinned {mib} MiB resident ({bytes} B), \
         touched_xor=0x{xor:02x} ████\n\
         ████ decode path UNCHANGED (byte-transparent); held to _exit so the   ████\n\
         ████ kernel address-space teardown pays for it (Leg-2 ceiling oracle). ████\n"
    );
    // Leak so it stays mapped until the kernel tears the address space down at
    // `_exit` (process::exit skips destructors anyway; this is explicit).
    std::mem::forget(buf);
}
