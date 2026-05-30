#![allow(dead_code)]
// command-builders + struct fields are the embeddable
// API surface (used by `plan`-generated scripts, programmatic callers, and
// kept for completeness); not all are exercised by the CLI default path.
//! Chrome-trace JSON ingestion + B/E span pairing.
//!
//! Consumes the timeline emitted by gzippy's `trace_v2.rs` (activated by
//! `GZIPPY_TIMELINE=/path.json`). The format is the Chrome-trace "JSON
//! array format": a stream of `{"name","ph","ts","pid","tid","args"}`
//! objects, `ph` ∈ {B(egin), E(nd), i(nstant)}. trace_v2 writes an open
//! `[` and tolerates a partial array (trailing comma, no close), so the
//! loader repairs that before parsing — identical handling to the proven
//! `scripts/timeline_analyze.py`.
//!
//! Pairing reconstructs, per (pid,tid), the begin/end nesting into spans
//! with a duration, a parent name (the enclosing open B), and the args
//! object (carries `chunk_id`, `start_bit`, etc. — the keys the
//! critical-path layer correlates on).

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// One raw Chrome-trace event.
#[derive(Debug, Deserialize)]
pub struct Event {
    pub name: String,
    pub ph: String,
    #[serde(default)]
    pub ts: f64,
    #[serde(default)]
    pub pid: u64,
    #[serde(default)]
    pub tid: u64,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// A paired span: a begin matched to its end on the same thread.
#[derive(Debug, Clone)]
pub struct Span {
    pub name: String,
    pub parent: String,
    pub pid: u64,
    pub tid: u64,
    pub ts_start: f64,
    pub ts_end: f64,
    /// Duration in microseconds (trace_v2 emits fractional-µs = ns precision).
    pub dur: f64,
    pub args: serde_json::Value,
}

impl Span {
    /// Read an integer arg (e.g. `chunk_id`, `partition_idx`). Accepts the
    /// value being a JSON number or a numeric string.
    pub fn arg_u64(&self, key: &str) -> Option<u64> {
        match self.args.get(key) {
            Some(serde_json::Value::Number(n)) => n.as_u64(),
            Some(serde_json::Value::String(s)) => s.parse().ok(),
            _ => None,
        }
    }

    /// True for the "this thread is blocked on another" span categories —
    /// the wait edges the critical-path layer attributes idle time across.
    pub fn is_wait(&self) -> bool {
        self.name.starts_with("wait.")
            || self.name == "lock.wait"
            || self.name == "pool.pick.wait"
            || self.name.ends_with(".wait")
            || self.name == "consumer.wait_replaced_markers"
            || self.name == "ttp.rx_recv_block"
            || self.name == "ttp.get_if_available"
    }
}

/// Load + repair + parse a trace_v2 Chrome-trace file.
pub fn load_events(path: &Path) -> std::io::Result<Vec<Event>> {
    let mut s = std::fs::read_to_string(path)?;
    let trimmed = s.trim_end();
    s = trimmed.to_string();
    if s.starts_with('[') && !s.ends_with(']') {
        // strip trailing comma/newline, close the array
        while s.ends_with(',') || s.ends_with('\n') {
            s.pop();
        }
        s.push('\n');
        s.push(']');
    } else if s.ends_with(',') {
        while s.ends_with(',') || s.ends_with('\n') {
            s.pop();
        }
        if !s.ends_with(']') {
            s.push(']');
        }
    }
    if !s.starts_with('[') {
        s.insert(0, '[');
    }
    let events: Vec<Event> = serde_json::from_str(&s).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("trace parse {}: {e}", path.display()),
        )
    })?;
    Ok(events)
}

/// Pair B/E events into spans with parent nesting. Mismatched ends are
/// dropped (best-effort, like the Python reference).
pub fn pair_spans(events: &[Event]) -> Vec<Span> {
    let mut stacks: HashMap<(u64, u64), Vec<&Event>> = HashMap::new();
    let mut spans = Vec::new();
    for e in events {
        match e.ph.as_str() {
            "B" => stacks.entry((e.pid, e.tid)).or_default().push(e),
            "E" => {
                let key = (e.pid, e.tid);
                if let Some(stack) = stacks.get_mut(&key) {
                    if let Some(b) = stack.pop() {
                        let parent = stack
                            .last()
                            .map(|p| p.name.clone())
                            .unwrap_or_else(|| "<root>".to_string());
                        spans.push(Span {
                            name: b.name.clone(),
                            parent,
                            pid: b.pid,
                            tid: b.tid,
                            ts_start: b.ts,
                            ts_end: e.ts,
                            dur: e.ts - b.ts,
                            args: b.args.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    spans
}

/// Instant events (`ph == "i"`), e.g. `alloc` sites.
pub fn instant_events(events: &[Event]) -> Vec<&Event> {
    events.iter().filter(|e| e.ph == "i").collect()
}

/// Overall wall of the trace (max end − min start across all spans), µs.
pub fn wall_us(spans: &[Span]) -> f64 {
    if spans.is_empty() {
        return 0.0;
    }
    let min = spans
        .iter()
        .map(|s| s.ts_start)
        .fold(f64::INFINITY, f64::min);
    let max = spans
        .iter()
        .map(|s| s.ts_end)
        .fold(f64::NEG_INFINITY, f64::max);
    max - min
}

/// Format µs the way the Python analyzer does (ns/µs/ms/s).
pub fn fmt_us(us: f64) -> String {
    if us >= 1_000_000.0 {
        format!("{:.3}s", us / 1_000_000.0)
    } else if us >= 1000.0 {
        format!("{:.2}ms", us / 1000.0)
    } else if us >= 1.0 {
        format!("{:.2}us", us)
    } else {
        format!("{:.0}ns", us * 1000.0)
    }
}
