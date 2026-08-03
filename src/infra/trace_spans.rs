//! Chrome-trace span emission for the T>1 ENCODER pipeline — the instrument
//! `fulcrum trace` (critpath/occupancy/consumer/vs) consumes.
//!
//! Feature-gated behind `trace` (default OFF): production builds contain none
//! of this — no env read, no mutex, nothing (non-negotiable #3). A `--features
//! trace` build writes a Chrome trace to the path named by `FULCRUM_TRACE`
//! when that env is set, and behaves identically otherwise.
//!
//! WHY THIS EXISTS (the blocked question, per the instrument rule): PR #251's
//! single confirmed wall flip — libdeflate:symbols.dwarf:L1:T4 at 1.081,
//! n=45 — shows the two arms TIE on every hardware counter when serialized
//! (instr/B 1.0065, cyc/B 1.0064, all Zen2 stalls TIE) yet differ 10.8% on
//! the unpinned parallel wall. Equal total work, different wall = a
//! SCHEDULING difference, and per-thread span timelines are the instrument
//! that names those. The decode campaign won its parallel war with exactly
//! this tooling; the encoder path never had an emitter until now.
//!
//! Span vocabulary (the writer thread is tid 0; workers are tid 1..=N):
//!   chunk_compress  {block_idx, len}   worker: one per claimed block
//!   write_wait      {block_idx}        writer: spin until the slot is ready
//!   write           {block_idx, len}   writer: the ordered write itself
//! `fulcrum trace` reads these with its generic profile; occupancy and
//! critpath are vocabulary-agnostic.

#[cfg(feature = "trace")]
mod imp {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    struct Span {
        name: &'static str,
        tid: u32,
        ts_us: u64,
        dur_us: u64,
        block_idx: usize,
        len: usize,
    }

    fn epoch() -> Instant {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    fn spans() -> &'static Mutex<Vec<Span>> {
        static SPANS: OnceLock<Mutex<Vec<Span>>> = OnceLock::new();
        SPANS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Monotonic microseconds since first use — the trace time base.
    pub fn now_us() -> u64 {
        epoch().elapsed().as_micros() as u64
    }

    pub fn record(name: &'static str, tid: u32, start_us: u64, block_idx: usize, len: usize) {
        let dur = now_us().saturating_sub(start_us);
        spans().lock().unwrap().push(Span {
            name,
            tid,
            ts_us: start_us,
            dur_us: dur.max(1),
            block_idx,
            len,
        });
    }

    /// Write everything recorded so far to `$FULCRUM_TRACE` (whole-file
    /// rewrite, so the final flush always wins). Called at the end of each
    /// `compress_parallel`; cheap no-op when the env is unset.
    pub fn flush() {
        let Ok(path) = std::env::var("FULCRUM_TRACE") else {
            return;
        };
        let guard = spans().lock().unwrap();
        let mut out = String::with_capacity(guard.len() * 96 + 2);
        out.push('[');
        for (i, s) in guard.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            // B/E pairs, not complete-X events: fulcrum's trace parser
            // reconciles Begin/End pairs (its RECONCILE gate says so), and an
            // X-only timeline reads as 0ms everywhere — measured on this
            // emitter's first hookup.
            out.push_str(&format!(
                "{{\"name\":\"{n}\",\"ph\":\"B\",\"pid\":1,\"tid\":{t},\"ts\":{b},\"args\":{{\"block_idx\":{i},\"len\":{l}}}}},{{\"name\":\"{n}\",\"ph\":\"E\",\"pid\":1,\"tid\":{t},\"ts\":{e}}}",
                n = s.name,
                t = s.tid,
                b = s.ts_us,
                e = s.ts_us + s.dur_us,
                i = s.block_idx,
                l = s.len
            ));
        }
        out.push(']');
        let _ = std::fs::write(&path, out);
    }
}

#[cfg(feature = "trace")]
pub use imp::{flush, now_us, record};

#[cfg(not(feature = "trace"))]
#[inline(always)]
pub fn now_us() -> u64 {
    0
}
#[cfg(not(feature = "trace"))]
#[inline(always)]
pub fn record(_: &'static str, _: u32, _: u64, _: usize, _: usize) {}
#[cfg(not(feature = "trace"))]
#[inline(always)]
pub fn flush() {}
