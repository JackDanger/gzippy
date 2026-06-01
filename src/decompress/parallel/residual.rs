#![allow(dead_code)] // optional instrumentation; on-demand via GZIPPY_TIMELINE

//! residual.rs — PERF-FREE per-region OS-counter snapshots for the
//! `fulcrum decompose` residual tier.
//!
//! At region boundaries we snapshot, for the CURRENT THREAD:
//!   - getrusage(RUSAGE_THREAD): ru_minflt / ru_majflt (page faults),
//!     ru_nvcsw / ru_nivcsw (voluntary / INvoluntary context switches).
//!   - /proc/self/task/<tid>/schedstat: cumulative runnable-waiting ns.
//!
//! The DELTA across a region is emitted as a Chrome-trace INSTANT event named
//! `rusage.region` whose args carry the counter names Fulcrum's `decompose`
//! view reads (`ru_minflt`, `ru_majflt`, `nvcsw`, `nivcsw`,
//! `sched_runnable_ns`). The per-thread join in Fulcrum attributes each
//! instant to the `(tid, region)` cell whose span contains its timestamp — so
//! we just emit at the END of a region with the delta since its START.
//!
//! ## Why this is byte-identical to production
//!
//! Every public fn here early-returns when `trace_v2::is_enabled()` is false
//! (i.e. `GZIPPY_TIMELINE` unset), which is the production default. No counter
//! is read, no syscall is made, no event is emitted. The struct is a couple of
//! `i64`s on the stack. There is no allocation and no behavioral branch in the
//! decode itself — only an extra getrusage at a span boundary when tracing.
//!
//! RUSAGE_THREAD + schedstat are Linux-only; on other platforms the snapshot
//! is a zero-valued no-op (the bench host `neurotic` is Linux, so the residual
//! tier is fully populated there; macOS dev builds simply emit nothing).

use super::trace_v2;

/// A per-thread OS-counter snapshot. Cheap: a few integers, no allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidualSnapshot {
    pub minflt: i64,
    pub majflt: i64,
    pub nvcsw: i64,
    pub nivcsw: i64,
    /// Cumulative runnable-but-not-running time, nanoseconds (schedstat field 2).
    pub runnable_ns: i64,
}

impl ResidualSnapshot {
    /// Snapshot the current thread's counters. Returns a zeroed snapshot when
    /// tracing is disabled (so the syscall never happens in production) or on
    /// a non-Linux platform.
    #[inline]
    pub fn capture() -> ResidualSnapshot {
        if !trace_v2::is_enabled() {
            return ResidualSnapshot::default();
        }
        capture_impl()
    }

    /// Emit the delta `self → end` for `region` as an instant trace event.
    /// No-op when tracing is disabled.
    pub fn emit_region_delta(&self, region: &str) {
        if !trace_v2::is_enabled() {
            return;
        }
        let end = capture_impl();
        let body = format!(
            r#""region":"{region}","ru_minflt":{},"ru_majflt":{},"nvcsw":{},"nivcsw":{},"sched_runnable_ns":{}"#,
            end.minflt - self.minflt,
            end.majflt - self.majflt,
            end.nvcsw - self.nvcsw,
            end.nivcsw - self.nivcsw,
            end.runnable_ns - self.runnable_ns,
        );
        trace_v2::emit_instant("rusage.region", &body, "t");
    }
}

/// RAII guard: capture on construction, emit the region delta on Drop. Declare
/// it AFTER the region's `trace_v2::SpanGuard` so it drops FIRST — i.e. the
/// `rusage.region` instant lands while the region span is still open, so
/// Fulcrum's per-thread containment join attributes it to that region.
pub struct ResidualGuard {
    start: ResidualSnapshot,
    region: &'static str,
    armed: bool,
}

impl ResidualGuard {
    #[inline]
    pub fn begin(region: &'static str) -> Self {
        let armed = trace_v2::is_enabled();
        ResidualGuard {
            start: if armed {
                capture_impl()
            } else {
                ResidualSnapshot::default()
            },
            region,
            armed,
        }
    }
}

impl Drop for ResidualGuard {
    fn drop(&mut self) {
        if self.armed {
            self.start.emit_region_delta(self.region);
        }
    }
}

#[cfg(target_os = "linux")]
fn capture_impl() -> ResidualSnapshot {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // RUSAGE_THREAD == 1 on Linux; libc exposes it as RUSAGE_THREAD.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) };
    let (minflt, majflt, nvcsw, nivcsw) = if rc == 0 {
        (
            ru.ru_minflt as i64,
            ru.ru_majflt as i64,
            ru.ru_nvcsw as i64,
            ru.ru_nivcsw as i64,
        )
    } else {
        (0, 0, 0, 0)
    };
    ResidualSnapshot {
        minflt,
        majflt,
        nvcsw,
        nivcsw,
        runnable_ns: read_schedstat_runnable_ns(),
    }
}

#[cfg(not(target_os = "linux"))]
fn capture_impl() -> ResidualSnapshot {
    ResidualSnapshot::default()
}

/// Read field 2 of /proc/self/task/<tid>/schedstat (runnable-but-not-running
/// time, ns). Returns 0 if unreadable. Linux-only.
#[cfg(target_os = "linux")]
fn read_schedstat_runnable_ns() -> i64 {
    // /proc/self/task/<gettid>/schedstat: "<run_ns> <runnable_ns> <timeslices>"
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let path = format!("/proc/self/task/{tid}/schedstat");
    match std::fs::read_to_string(&path) {
        Ok(s) => s
            .split_whitespace()
            .nth(1)
            .and_then(|t| t.parse::<i64>().ok())
            .unwrap_or(0),
        Err(_) => 0,
    }
}
