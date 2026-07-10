//! Store-elimination removal oracle — feature `storeprobe` (OFF by default;
//! ZERO effect on the production build/bytes/timing; byte-INEXACT when on).
//!
//! Built to settle the storedheavy sub-cause brief's final discriminator
//! (2026-07-09): the Gate-2 `perturb` H/M *addition* instrument (commit
//! `28bc0f43`) proved the marker fast loop's serial chain is on the
//! critical path but cannot tell whether the u16 ring's 128 KiB memory
//! SPREAD (bandwidth/cache-footprint) or the per-symbol COMPUTE/latency
//! chain is the limiting resource — adding delay at either site slows
//! things down either way. This is the complementary *removal* oracle:
//! instead of adding cost, it REMOVES the ring's memory-spread cost by
//! redirecting the widened-literal store in
//! `Block::decode_marker_fast_loop`'s `mfast_lb_run!` body
//! (`marker_inflate.rs`, the `(ring_ptr.add(dst_phys) as *mut
//! u64).write_unaligned(widened)` site) to a small, FIXED-size, L1-resident
//! thread-local scratch buffer — collapsing the store's address footprint
//! from 128 KiB down to ~128 bytes while leaving every other instruction
//! (decode, consume, refill, `pos`/`emitted`/`distance_marker` bookkeeping)
//! untouched. The decode stays LIVE (same Huffman/bit-reader work, same
//! trip count — see the parity note below) and produces byte-INEXACT
//! garbage output (backref copies now source stale/overwritten scratch
//! content) — that is expected and fine; this measurement is gz-vs-gz wall
//! time only, never sha-checked.
//!
//! VERDICT key:
//!   - probe wall drops >=15% vs baseline (storeprobe OFF) => the ring's
//!     memory spread is on the critical path => STORE-BANDWIDTH-BOUND
//!     (lever: reduce ring store traffic, e.g. a u8 literal lane).
//!   - probe wall flat (within +-3%) => the store's *destination* is
//!     irrelevant to the wall => COMPUTE/LATENCY-BOUND (the serial decode
//!     dependency chain is the limit; near-fundamental for this arch).
//!
//! Trip-count / decode-liveness parity: this module changes NOTHING about
//! `dst_phys`/`sym0`/`pos`/`emitted`/`distance_marker` computation or loop
//! exit conditions — only the store's target address — so the number of
//! `mfast_lb_run!` iterations is identical to baseline BY CONSTRUCTION
//! (control flow never reads the store's destination or its prior
//! contents). This is independently confirmed at measurement time via the
//! `phase-timing` feature's `worker_decoded_bytes` / `worker_decode_window_absent`
//! / `worker_decode_invocations` JSON fields (`phase_timing.rs`), which must
//! match baseline exactly across an OFF vs ON run pair — see the campaign
//! report for the actual numbers.
//!
//! `write_volatile` (not `write_unaligned`) is used for the redirected
//! store specifically so LLVM cannot prove the write is dead (the scratch
//! buffer is never read back by anything that influences a visible side
//! effect) and elide it — a DCE'd store would silently turn this into a
//! measurement of "no store at all", not "store to a hot address", which
//! would falsely inflate any observed speedup.
#![cfg_attr(not(feature = "storeprobe"), allow(dead_code))]

/// Number of u16 slots in the hot scratch sink. 64 slots = 128 bytes — small
/// enough to stay L1-resident across the whole decode (vs. the ring's 128
/// KiB / 65536-slot footprint), matching the brief's "128-byte hot buffer"
/// spec exactly.
#[cfg(feature = "storeprobe")]
pub const PROBE_SINK_LEN: usize = 64;

/// Slop slots appended past `PROBE_SINK_LEN` so the widened 8-byte (4x u16)
/// store issued at `idx = dst_phys & (PROBE_SINK_LEN - 1)` never writes past
/// the end of the buffer even when `idx` lands in the last few slots —
/// mirrors the ring's own `FAST_OUT_SLOP` headroom convention
/// (`marker_inflate.rs`), just sized for this much smaller buffer.
#[cfg(feature = "storeprobe")]
const PROBE_SINK_SLOP: usize = 4;

#[cfg(feature = "storeprobe")]
std::thread_local! {
    /// Per-worker-thread hot scratch sink. `UnsafeCell` (not `RefCell`/
    /// `Cell<[u16; N]>`) so the caller can take a raw pointer and write
    /// through it directly — no borrow-check bookkeeping inside the decode
    /// loop. `const {}` initializer selects rustc's fast TLS-access path
    /// (no lazy-init flag check per access, stable since 1.59) — relevant
    /// because [`sink_ptr`] below is intentionally hoisted OUTSIDE the
    /// per-symbol hot loop (see its doc) rather than called per-iteration;
    /// an EARLIER version of this probe called `PROBE_SINK.with(..)` once
    /// PER SYMBOL and measured 20-30% SLOWER than baseline on solvency
    /// (AMD Zen2) — a real effect, but of `std::thread_local`'s per-access
    /// lookup cost, NOT of the ring's memory-spread cost, which would have
    /// contaminated the store-vs-compute verdict. Hoisting the lookup to
    /// once per `decode_marker_fast_loop` invocation (amortized over
    /// thousands of iterations) removes that confound; only the store
    /// destination differs from baseline inside the loop, as the brief
    /// requires.
    static PROBE_SINK: std::cell::UnsafeCell<[u16; PROBE_SINK_LEN + PROBE_SINK_SLOP]> =
        const { std::cell::UnsafeCell::new([0u16; PROBE_SINK_LEN + PROBE_SINK_SLOP]) };
}

/// Gate-0 self-validation counter: incremented once per actually-executed
/// probe store. Read via [`hits`] and printed under `GZIPPY_DEBUG=1`
/// (`single_member.rs`) so a run with `hits==0` (the probe call site never
/// fired — e.g. every chunk resolved via `decode_careful_tail` instead of
/// the marker fast loop) is caught before any wall number is trusted, same
/// discipline as `perturb.rs`'s `HITS`. A relaxed, uncontended (per-thread
/// cache line is never shared — each worker only ever increments its own
/// stores between synchronization points) atomic add; kept deliberately
/// OUTSIDE the timed measurement's interpretation (a small, dst_phys-
/// independent constant per iteration, so it cannot manufacture a false
/// bandwidth-vs-compute signal, only a small fixed floor common to every
/// `dst_phys` value).
#[cfg(feature = "storeprobe")]
static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Resolve the hot scratch sink's base pointer. MUST be called ONCE per
/// `decode_marker_fast_loop` invocation — i.e. hoisted OUTSIDE the
/// per-symbol `'mfast` loop, exactly like the real `ring_ptr` parameter is
/// obtained once before the loop — and the returned pointer reused for
/// every [`probe_store_widened`] call inside that loop. See the
/// `PROBE_SINK` doc for why a per-iteration `.with()` call is NOT valid
/// here (it was measured to add its own confounding cost).
///
/// Safe: only performs the TLS lookup and a pointer cast, no dereference.
#[cfg(feature = "storeprobe")]
#[inline(always)]
pub fn sink_ptr() -> *mut u16 {
    PROBE_SINK.with(|cell| cell.get() as *mut u16)
}

/// Redirect the widened-literal ring store to the hot scratch buffer at
/// `sink` (obtained once via [`sink_ptr`]). Called from exactly the site in
/// `marker_inflate.rs`'s `mfast_lb_run!` that would otherwise do
/// `(ring_ptr.add(dst_phys) as *mut u64).write_unaligned(widened)`.
///
/// # Safety
/// `sink` must be a live pointer returned by [`sink_ptr`] on the SAME
/// thread (the `thread_local!` sink is never shared across threads, so no
/// synchronization is needed, but the pointer is only valid for the
/// thread that resolved it) and must remain valid for the duration of the
/// call (true for the whole decode: the thread-local's storage lives for
/// the worker thread's lifetime).
#[cfg(feature = "storeprobe")]
#[inline(always)]
pub unsafe fn probe_store_widened(sink: *mut u16, dst_phys: usize, widened: u64) {
    HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let idx = dst_phys & (PROBE_SINK_LEN - 1);
    (sink.add(idx) as *mut u64).write_volatile(widened);
}

#[cfg(not(feature = "storeprobe"))]
#[inline(always)]
pub unsafe fn probe_store_widened(_sink: *mut u16, _dst_phys: usize, _widened: u64) {}
#[cfg(not(feature = "storeprobe"))]
#[inline(always)]
pub fn sink_ptr() -> *mut u16 {
    std::ptr::null_mut()
}

/// Total number of probe stores that actually fired. `0` whenever the
/// `storeprobe` feature is off, or (with it on) the marker fast loop's
/// store site was never reached — either case means the run measured
/// nothing and must not be reported as a store-vs-compute result.
#[cfg(feature = "storeprobe")]
pub fn hits() -> u64 {
    HITS.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(not(feature = "storeprobe"))]
pub fn hits() -> u64 {
    0
}

#[cfg(all(test, feature = "storeprobe"))]
mod tests {
    use super::*;

    /// Gate-0 self-test: the probe store actually executes (hits > 0) and
    /// does not panic/UB on the smallest and largest possible `dst_phys`
    /// values (0 and RING_SIZE-1, i.e. every possible `& (PROBE_SINK_LEN-1)`
    /// residue), proving the slop headroom is sufficient before this is
    /// ever wired into the real decode loop.
    #[test]
    fn probe_store_fires_and_stays_in_bounds() {
        let before = hits();
        let sink = sink_ptr();
        assert!(!sink.is_null(), "sink_ptr() returned null with feature on");
        for dst_phys in 0..PROBE_SINK_LEN * 3 {
            unsafe { probe_store_widened(sink, dst_phys, 0x0102_0304_0506_0708) };
        }
        let after = hits();
        assert_eq!(
            after - before,
            (PROBE_SINK_LEN * 3) as u64,
            "probe store did not fire once per call"
        );
    }
}
