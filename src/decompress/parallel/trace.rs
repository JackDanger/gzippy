//! Verbose structured tracing for the parallel single-member decoder.
//!
//! When the environment variable `GZIPPY_LOG_FILE` is set, every
//! decision point in the fetcher / chunk decoder writes a JSON-lines
//! event to that path. The file is intentionally too verbose to load
//! into a human's working memory in one sitting — instead, query it
//! with the helper tools in `scripts/parallel_sm_log_*`.
//!
//! Events are append-only and lock-protected so multiple worker
//! threads can write concurrently without interleaving lines.
//!
//! Each event line:
//! ```json
//! {"t_ns": 1234567890, "thread": "worker-3", "ev": "chunk_decode_done",
//!  "partition_idx": 5, "start_bit": 167948953, "end_bit": 168266736,
//!  "decoded": 39718, "markers": 39718, "clean": 0,
//!  "preemptive": false, "duration_us": 1843, "source": "speculative"}
//! ```
//!
//! Field conventions:
//! - `t_ns`: monotonic nanoseconds since process start (Instant::now).
//! - `thread`: short label (`consumer`, `worker-N`, `boundary-N`).
//! - `ev`: event kind (snake_case verb).
//! - Bit positions are absolute in the deflate stream.

#![allow(dead_code)]

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct TraceState {
    file: Mutex<std::fs::File>,
    epoch: Instant,
}

static STATE: OnceLock<Option<TraceState>> = OnceLock::new();

fn state() -> Option<&'static TraceState> {
    STATE
        .get_or_init(|| {
            let path = std::env::var("GZIPPY_LOG_FILE").ok()?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()?;
            Some(TraceState {
                file: Mutex::new(file),
                epoch: Instant::now(),
            })
        })
        .as_ref()
}

#[inline]
pub fn is_enabled() -> bool {
    state().is_some()
}

/// Emit one JSON-lines event. `body` should be a string of comma-
/// separated `"key":value` pairs (no surrounding braces); `ev` is the
/// event kind and `thread` is a short label. The wrapper adds the
/// timestamp + thread + event-kind preamble and the surrounding braces.
///
/// We keep the API string-based (rather than using serde_json) to
/// avoid adding a dependency just for tracing and to keep emit cost
/// low — the body is built with `format!` at the call site.
pub fn emit(thread: &str, ev: &str, body: &str) {
    let Some(state) = state() else {
        return;
    };
    let t_ns = state.epoch.elapsed().as_nanos();
    let line = if body.is_empty() {
        format!(
            r#"{{"t_ns":{t_ns},"thread":"{thread}","ev":"{ev}"}}{}"#,
            "\n"
        )
    } else {
        format!(
            r#"{{"t_ns":{t_ns},"thread":"{thread}","ev":"{ev}",{body}}}{}"#,
            "\n"
        )
    };
    let mut f = state.file.lock().unwrap();
    let _ = f.write_all(line.as_bytes());
}

/// Helper for the common "thread = worker-N" pattern.
pub fn worker_label(idx: usize) -> String {
    format!("worker-{idx}")
}

/// Read process RSS in KiB from /proc/self/status (Linux only).
/// Returns 0 on non-Linux or read failure. Used to embed memory
/// numbers in trace events so post-hoc analysis can correlate
/// memory pressure with specific dispatcher actions.
pub fn rss_kib() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    return rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    0
}

/// Minimal JSON string escape — only what we emit (escape `"`, `\`, and
/// control bytes by collapsing them). Sufficient for `format!("{e:?}")`
/// output of Rust error types which can include embedded quotes from
/// the Debug derivation.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// Helper for the common "thread = boundary-N" pattern.
pub fn boundary_label(idx: usize) -> String {
    format!("boundary-{idx}")
}

/// Monotonic seconds since the trace epoch (process start when trace
/// was first initialized; falls back to a per-call `Instant::now()` if
/// tracing is disabled — value is still useful for relative
/// difference within the same call site).
///
/// Used by `ChunkFetcherStatistics::note_decode_block_start` /
/// `note_decode_block_end` to compute pool efficiency.
pub fn now_secs() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_secs_f64()
}
