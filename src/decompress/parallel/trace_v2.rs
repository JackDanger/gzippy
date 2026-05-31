#![allow(dead_code)] // optional instrumentation; on-demand via GZIPPY_TIMELINE

//! Chrome trace JSON timeline instrumentation for cross-tool comparison
//! with rapidgzip.
//!
//! Activated by env var `GZIPPY_TIMELINE=/tmp/out.json`. On every
//! emit point the writer formats a Chrome trace event and appends it to
//! a per-thread `Vec<u8>` buffer. The buffer flushes under a global
//! mutex every ~4 KiB (and on thread exit / explicit flush_all).
//!
//! Output: Chrome trace "JSON array format" — opens directly in
//! `chrome://tracing`, `perfetto.dev`, or `speedscope`. The Python
//! analyzer at `scripts/timeline_analyze.py` consumes the same format
//! emitted by the rapidgzip-side instrumentation, so we get
//! apples-to-apples critical-path comparison.
//!
//! pid convention (so the cross-tool diff lays them out side-by-side):
//!   - gzippy:    pid = 1
//!   - rapidgzip: pid = 2 (when the C++-side equivalent is wired up)
//!
//! Span names use SEMANTIC tags (not Rust function names) so they map
//! 1:1 with rapidgzip's equivalent sites:
//!   worker.decode_chunk, worker.bootstrap, worker.inflate_bulk,
//!   consumer.iter, consumer.drain, post_process.apply_window,
//!   block_finder.scan, wait.future_recv, wait.mutex.*,
//!   alloc.<site>.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const GZIPPY_PID: u32 = 1;
const FLUSH_THRESHOLD: usize = 4096;

struct TraceState {
    file: Mutex<File>,
    anchor: Instant,
}

static STATE: OnceLock<Option<TraceState>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

fn state() -> Option<&'static TraceState> {
    STATE
        .get_or_init(|| {
            let path = std::env::var("GZIPPY_TIMELINE").ok()?;
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .ok()?;
            // Chrome trace JSON array format: open bracket. No close
            // needed — the tool tolerates partial arrays.
            file.write_all(b"[\n").ok()?;
            ENABLED.store(true, Ordering::Relaxed);
            Some(TraceState {
                file: Mutex::new(file),
                anchor: Instant::now(),
            })
        })
        .as_ref()
}

#[inline]
pub fn is_enabled() -> bool {
    // Under Coz profiling, force-OFF so trace_v2's per-call OnceLock+atomic
    // (executed by every SpanGuard begin/drop, ~2 per deflate block in the
    // bootstrap loop) doesn't sit ON the measured path and inflate the
    // attribution of whatever region it's nested in. The advisor caught this:
    // trace_v2 itself showed up as a Coz "lever" with the timeline OFF.
    #[cfg(feature = "coz")]
    {
        false
    }
    #[cfg(not(feature = "coz"))]
    {
        // Cheap: avoid OnceLock::get_or_init in hot path after first call.
        ENABLED.load(Ordering::Relaxed) || state().is_some()
    }
}

thread_local! {
    static BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(FLUSH_THRESHOLD * 2));
    // Cached thread id, monotonic per-thread, assigned on first emit.
    static TID: RefCell<u32> = const { RefCell::new(0) };
}

static NEXT_TID: AtomicU32 = AtomicU32::new(1);

fn current_tid() -> u32 {
    TID.with(|c| {
        let mut tid = c.borrow_mut();
        if *tid == 0 {
            *tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
        }
        *tid
    })
}

fn now_us() -> f64 {
    // Microseconds as f64 carrying sub-µs (nanosecond) precision. Chrome-trace
    // `ts` is conventionally microseconds and accepts floats, so this stays
    // perfetto/speedscope-compatible AND gives nanosecond resolution — exact
    // ns numbers for aggregate span sums and for short (sub-µs) spans, which
    // integer-µs truncation was silently dropping. Unit is still µs (hence the
    // name); it's just fractional now.
    state()
        .map(|s| s.anchor.elapsed().as_nanos() as f64 / 1000.0)
        .unwrap_or(0.0)
}

fn flush_locked(buf: &mut Vec<u8>) {
    if buf.is_empty() {
        return;
    }
    if let Some(s) = state() {
        if let Ok(mut f) = s.file.lock() {
            let _ = f.write_all(buf);
        }
    }
    buf.clear();
}

fn append_event(line: &str) {
    BUF.with(|c| {
        let mut buf = c.borrow_mut();
        buf.extend_from_slice(line.as_bytes());
        if buf.len() >= FLUSH_THRESHOLD {
            flush_locked(&mut buf);
        }
    });
}

/// Explicit drain — call from parallel-SM entry on completion so
/// pooled-thread tail bytes don't sit until process exit.
pub fn flush_all() {
    BUF.with(|c| {
        let mut buf = c.borrow_mut();
        flush_locked(&mut buf);
    });
}

/// Begin phase. `args_body` should be the inner JSON of an args object
/// WITHOUT surrounding braces, e.g. `r#""chunk_id":7,"start_bit":12345"#`.
/// Pass `""` for no args.
pub fn emit_begin(name: &str, args_body: &str) {
    if !is_enabled() {
        return;
    }
    let ts = now_us();
    let tid = current_tid();
    let line = if args_body.is_empty() {
        format!(
            r#"{{"name":"{name}","ph":"B","ts":{ts:.3},"pid":{GZIPPY_PID},"tid":{tid}}},
"#
        )
    } else {
        format!(
            r#"{{"name":"{name}","ph":"B","ts":{ts:.3},"pid":{GZIPPY_PID},"tid":{tid},"args":{{{args_body}}}}},
"#
        )
    };
    append_event(&line);
}

pub fn emit_end(name: &str) {
    if !is_enabled() {
        return;
    }
    let ts = now_us();
    let tid = current_tid();
    let line = format!(
        r#"{{"name":"{name}","ph":"E","ts":{ts:.3},"pid":{GZIPPY_PID},"tid":{tid}}},
"#
    );
    append_event(&line);
}

/// Instant event (single point, e.g. allocation site).
/// Scope `s` is one of `"g"` (global), `"p"` (process), `"t"` (thread).
pub fn emit_instant(name: &str, args_body: &str, scope: &str) {
    if !is_enabled() {
        return;
    }
    let ts = now_us();
    let tid = current_tid();
    let line = if args_body.is_empty() {
        format!(
            r#"{{"name":"{name}","ph":"i","ts":{ts:.3},"pid":{GZIPPY_PID},"tid":{tid},"s":"{scope}"}},
"#
        )
    } else {
        format!(
            r#"{{"name":"{name}","ph":"i","ts":{ts:.3},"pid":{GZIPPY_PID},"tid":{tid},"s":"{scope}","args":{{{args_body}}}}},
"#
        )
    };
    append_event(&line);
}

/// L3 allocation event. Site is `"file:line"` (use `file!()` /
/// `line!()` from caller). bytes is the allocation size.
pub fn emit_alloc(bytes: usize, site: &str) {
    if !is_enabled() {
        return;
    }
    let body = format!(r#""bytes":{bytes},"site":"{site}""#);
    emit_instant("alloc", &body, "t");
}

/// RAII span: emit_begin on construction, emit_end on Drop.
/// Use the `span!` macro for ergonomic call sites.
pub struct SpanGuard {
    name: &'static str,
}

impl SpanGuard {
    pub fn begin(name: &'static str) -> Self {
        if is_enabled() {
            emit_begin(name, "");
        }
        Self { name }
    }

    pub fn begin_with(name: &'static str, args_body: &str) -> Self {
        if is_enabled() {
            emit_begin(name, args_body);
        }
        Self { name }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if is_enabled() {
            emit_end(self.name);
        }
    }
}

/// L4 lock span: wait_for_mutex (B/E) around the lock acquisition,
/// then lock_held (B/E) around the holding duration. Returns the
/// guard so the caller binds it in scope.
///
/// Usage:
///   let g = lock_span!("blockfetcher.cache", mutex);
///   // ... use the locked value via *g ...
///
/// On Drop of the returned wrapper, the lock_held end event fires.
pub struct LockSpan<'a, T> {
    guard: std::sync::MutexGuard<'a, T>,
    name: &'static str,
}

impl<T> std::ops::Deref for LockSpan<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> std::ops::DerefMut for LockSpan<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for LockSpan<'_, T> {
    fn drop(&mut self) {
        if is_enabled() {
            emit_end("lock.held");
        }
    }
}

impl<'a, T> LockSpan<'a, T> {
    pub fn acquire(mutex: &'a Mutex<T>, name: &'static str) -> Self {
        let wait_args = format!(r#""lock":"{name}""#);
        emit_begin("lock.wait", &wait_args);
        let guard = mutex.lock().unwrap_or_else(|p| p.into_inner());
        emit_end("lock.wait");
        emit_begin("lock.held", &wait_args);
        LockSpan { guard, name }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Macros
// ─────────────────────────────────────────────────────────────────────────

/// Span an entire block. `$args` is a pre-formatted JSON inner body
/// (e.g., `r#""chunk_id":7"#`) or `""` for no args.
#[macro_export]
macro_rules! trace_v2_span {
    ($name:literal, $args:expr, $body:block) => {{
        let _guard = $crate::decompress::parallel::trace_v2::SpanGuard::begin_with($name, $args);
        $body
    }};
    ($name:literal, $body:block) => {{
        let _guard = $crate::decompress::parallel::trace_v2::SpanGuard::begin($name);
        $body
    }};
}

/// Wait span. Conceptually identical to span! but exists as a separate
/// macro so the analyzer can recognize the "this thread is blocked on
/// another" category cheaply via the name prefix.
#[macro_export]
macro_rules! trace_v2_wait {
    ($name:literal, $args:expr, $body:block) => {{
        let _guard = $crate::decompress::parallel::trace_v2::SpanGuard::begin_with($name, $args);
        $body
    }};
    ($name:literal, $body:block) => {{
        let _guard = $crate::decompress::parallel::trace_v2::SpanGuard::begin($name);
        $body
    }};
}
