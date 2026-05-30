#![allow(dead_code)] // CritEntry/output-struct fields are the embeddable API.
//! Critical-path layer (wPerf-style), specialized for gzippy's in-order
//! streaming pipeline.
//!
//! The naive "critical path = longest dependent chain through a DAG"
//! requires explicit producer→consumer edges, which the trace does not
//! carry directly. But this pipeline has a structural shortcut: the
//! IN-ORDER CONSUMER gates the wall. Output bytes can only leave in chunk
//! order, so the program's wall ≈ the consumer thread's own timeline:
//!
//!     wall  ≈  Σ(consumer self-work spans)  +  Σ(consumer wait spans)
//!
//! Therefore the critical path IS the consumer thread, and the levers are
//! whatever (a) inflates the consumer's own work and (b) fills the
//! consumer's WAITS. A consumer wait is time the consumer sat blocked
//! because the next in-order chunk wasn't ready — so we ATTRIBUTE each
//! consumer wait to the worker span that was producing that chunk during
//! the wait window. This is the wPerf "blame the blocker" move, and it is
//! what surfaces the ~7 heavy "overshoot" bootstrap chunks: they appear as
//! long consumer waits attributed to specific `worker.bootstrap` spans.
//!
//! This avoids the CPU-time-SUM lie by construction: a worker span that is
//! never on the consumer's wait-attribution path contributes ZERO to the
//! critical path, no matter how much CPU it burned. (That is exactly why
//! `absorb_isal_tail` — fully overlapped on a worker — must show ~0 here,
//! the same verdict Coz must independently reach.)

use crate::trace::{pair_spans, wall_us, Event, Span};
use std::collections::HashMap;

/// Identify the consumer thread: the (pid,tid) that owns the in-order
/// drain spans. We pick the thread with the most `consumer.*` span time.
pub fn consumer_tid(spans: &[Span]) -> Option<(u64, u64)> {
    let mut score: HashMap<(u64, u64), f64> = HashMap::new();
    for s in spans {
        if s.name.starts_with("consumer.") {
            *score.entry((s.pid, s.tid)).or_default() += s.dur;
        }
    }
    score
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(k, _)| k)
}

/// A region of attributed critical-path time.
#[derive(Debug, Clone)]
pub struct CritEntry {
    /// What the time is attributed to: a span name (consumer self-work) or
    /// `"blocked-on:<worker-span>"` for attributed waits.
    pub label: String,
    pub on_path_us: f64,
    pub fraction: f64,
    /// How many distinct spans contributed (for the overshoot count).
    pub count: usize,
    /// Max single contribution (µs) — flags the bimodal heavy chunks.
    pub max_us: f64,
}

/// Result of the critical-path analysis.
pub struct CritPath {
    pub wall_us: f64,
    pub consumer: (u64, u64),
    pub consumer_busy_us: f64,
    pub consumer_wait_us: f64,
    pub entries: Vec<CritEntry>,
    /// The heavy "overshoot" chunks: consumer waits whose attributed
    /// blocker is a `worker.bootstrap`/`worker.decode_chunk` span longer
    /// than `heavy_threshold_us`.
    pub heavy_chunks: Vec<HeavyChunk>,
}

#[derive(Debug, Clone)]
pub struct HeavyChunk {
    pub blocker_span: String,
    pub chunk_id: Option<u64>,
    pub wait_us: f64,
    pub blocker_dur_us: f64,
}

/// Spans on `tid` that are NOT the consumer, overlapping `[a,b)`, ranked
/// by overlap. Used to attribute a consumer wait to its blocker.
fn overlapping_workers<'a>(
    spans: &'a [Span],
    consumer: (u64, u64),
    a: f64,
    b: f64,
) -> Vec<(&'a Span, f64)> {
    let mut out = Vec::new();
    for s in spans {
        if (s.pid, s.tid) == consumer {
            continue;
        }
        // Only "real work" spans are candidate blockers — not the
        // blocker's OWN waits/locks (those would double-count idle).
        if s.is_wait() || s.name.starts_with("lock.") || s.name.starts_with("pool.pick") {
            continue;
        }
        let ov = (s.ts_end.min(b) - s.ts_start.max(a)).max(0.0);
        if ov > 0.0 {
            out.push((s, ov));
        }
    }
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    out
}

/// The work-span granularity we attribute blame to. We prefer the most
/// SPECIFIC worker span overlapping a wait (e.g. `worker.bootstrap` over
/// its enclosing `worker.decode_chunk`), so blame lands on the real lever.
const PREFERRED_BLOCKERS: &[&str] = &[
    "worker.bootstrap",
    "worker.pure_bulk_inflate",
    "worker.isal_stream_inflate",
    "worker.block_body",
    "worker.scan_run",
    "worker.scan_candidate",
    "post_process.apply_window",
];

fn pick_blocker<'a>(cands: &'a [(&'a Span, f64)]) -> Option<&'a (&'a Span, f64)> {
    // First try a preferred (specific) blocker with meaningful overlap.
    for pref in PREFERRED_BLOCKERS {
        if let Some(c) = cands.iter().find(|(s, _)| &s.name == pref) {
            return Some(c);
        }
    }
    // Fallback: the largest-overlap non-consumer work span.
    cands.first()
}

/// Run the consumer-anchored critical-path analysis.
///
/// `heavy_threshold_us`: a blocker span longer than this, attributed to a
/// consumer wait, is flagged as an "overshoot" heavy chunk.
pub fn analyze(events: &[Event], heavy_threshold_us: f64) -> CritPath {
    let spans = pair_spans(events);
    let wall = wall_us(&spans);
    let consumer = consumer_tid(&spans).unwrap_or((1, 1));

    // Consumer self-work vs consumer waits. We sum LEAF time so nested
    // consumer spans don't double count: attribute each consumer span's
    // SELF time (its dur minus time covered by its direct children on the
    // same thread). Simpler + robust: bucket top-level consumer spans
    // (parent == "<root>" or parent not a consumer span) by name, and
    // handle waits separately.
    let mut busy = 0.0_f64;
    let mut wait = 0.0_f64;
    let mut self_by_name: HashMap<String, (f64, usize, f64)> = HashMap::new();
    let mut blocked_by: HashMap<String, (f64, usize, f64)> = HashMap::new();
    let mut heavy: Vec<HeavyChunk> = Vec::new();

    for s in &spans {
        if (s.pid, s.tid) != consumer {
            continue;
        }
        if s.is_wait() {
            wait += s.dur;
            // Attribute this wait to the worker span producing the awaited
            // chunk during the wait window.
            let cands = overlapping_workers(&spans, consumer, s.ts_start, s.ts_end);
            let label = match pick_blocker(&cands) {
                Some((blocker, _ov)) => {
                    if blocker.dur >= heavy_threshold_us
                        && (blocker.name.contains("bootstrap")
                            || blocker.name.contains("decode")
                            || blocker.name.contains("inflate"))
                    {
                        heavy.push(HeavyChunk {
                            blocker_span: blocker.name.clone(),
                            chunk_id: blocker
                                .arg_u64("chunk_id")
                                .or_else(|| s.arg_u64("chunk_id")),
                            wait_us: s.dur,
                            blocker_dur_us: blocker.dur,
                        });
                    }
                    format!("blocked-on:{}", blocker.name)
                }
                None => "blocked-on:<unknown>".to_string(),
            };
            let e = blocked_by.entry(label).or_insert((0.0, 0, 0.0));
            e.0 += s.dur;
            e.1 += 1;
            e.2 = e.2.max(s.dur);
        } else if !s.name.starts_with("lock.held") {
            // Consumer self-work. Bucket by name; only count spans whose
            // parent is NOT a consumer work span (top-level), to avoid
            // double-counting nested consumer spans. `consumer.iter` is the
            // umbrella, so we exclude it and credit its children.
            if s.name == "consumer.iter" {
                continue;
            }
            busy += s.dur;
            let e = self_by_name.entry(s.name.clone()).or_insert((0.0, 0, 0.0));
            e.0 += s.dur;
            e.1 += 1;
            e.2 = e.2.max(s.dur);
        }
    }

    let mut entries: Vec<CritEntry> = Vec::new();
    for (name, (sum, count, mx)) in self_by_name.into_iter() {
        entries.push(CritEntry {
            label: name,
            on_path_us: sum,
            fraction: if wall > 0.0 { sum / wall } else { 0.0 },
            count,
            max_us: mx,
        });
    }
    for (label, (sum, count, mx)) in blocked_by.into_iter() {
        entries.push(CritEntry {
            label,
            on_path_us: sum,
            fraction: if wall > 0.0 { sum / wall } else { 0.0 },
            count,
            max_us: mx,
        });
    }
    entries.sort_by(|a, b| b.on_path_us.partial_cmp(&a.on_path_us).unwrap());
    heavy.sort_by(|a, b| b.wait_us.partial_cmp(&a.wait_us).unwrap());

    CritPath {
        wall_us: wall,
        consumer,
        consumer_busy_us: busy,
        consumer_wait_us: wait,
        entries,
        heavy_chunks: heavy,
    }
}
