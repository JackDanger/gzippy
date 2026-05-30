#![allow(dead_code)]
// command-builders + struct fields are the embeddable
// API surface (used by `plan`-generated scripts, programmatic callers, and
// kept for completeness); not all are exercised by the CLI default path.
//! Mechanistic layer (Linux perf): attach a WHY to each region.
//!
//! Each lever in the ranked list carries a mechanism so the next move is
//! obvious: a DRAM-bound region wants prefetch / footprint shrink (u16→u8);
//! a branch-miss region wants a flatter decode; a false-sharing region
//! wants padding / per-thread state.
//!
//! Per Advisor-2's MVP scope, attribution is FUNCTION-LEVEL (via perf's own
//! symbol resolution), not a per-span (tid,timestamp) join — that join is
//! fragile (synthetic trace tids ≠ OS tids; clock-base offset) and is a
//! documented STRETCH. Function-level is robust and already answers "is the
//! bootstrap DRAM-bound or branch-bound", because the four FULCRUM regions
//! map to distinct hot functions (bootstrap_with_deflate_block,
//! decode_block, absorb_isal_tail, the scan/block-finder fns).
//!
//! Three captures:
//!   1. TMA top-down via `perf stat` topdown events → the dominant bound
//!      (Frontend/Backend/Retiring/BadSpec) for the whole run.
//!   2. PEBS memory profile via `perf record -e mem_load... ` (or
//!      `perf mem record`) → per-function DRAM/L3 share.
//!   3. `perf c2c record` → HITM lines (false sharing) per function.
//!
//! All run inside the freeze window. We don't parse perf's binary
//! perf.data; we run the matching `perf report`/`perf stat -x,` text/CSV
//! and parse that — robust across perf versions.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// The four TMA top-level categories (a run is bound by the largest).
#[derive(Debug, Clone, Default)]
pub struct TopDown {
    pub retiring: f64,
    pub bad_speculation: f64,
    pub frontend_bound: f64,
    pub backend_bound: f64,
}

impl TopDown {
    pub fn dominant(&self) -> (&'static str, f64) {
        let v = [
            ("retiring", self.retiring),
            ("bad-speculation", self.bad_speculation),
            ("frontend-bound", self.frontend_bound),
            ("backend-bound", self.backend_bound),
        ];
        v.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("unknown", 0.0))
    }
}

/// Per-function mechanism evidence collected from the perf captures.
#[derive(Debug, Clone, Default)]
pub struct FuncMech {
    /// % of cpu cycles samples in this function (perf report overhead).
    pub cycles_pct: f64,
    /// % of memory-load samples that missed to DRAM (PEBS).
    pub dram_pct: f64,
    /// HITM (modified-line cross-core) sample count (perf c2c) — false sharing.
    pub hitm: f64,
}

/// Full mechanistic snapshot.
#[derive(Default)]
pub struct Mech {
    pub topdown: TopDown,
    /// function name → evidence.
    pub by_func: BTreeMap<String, FuncMech>,
}

impl Mech {
    /// Summarize the mechanism for the function(s) backing a region. The
    /// region→function map is the same hot-function knowledge the coz layer
    /// uses; pass the substrings that identify the region's functions.
    pub fn region_mechanism(&self, func_substrings: &[&str]) -> String {
        let mut cycles = 0.0;
        let mut dram = 0.0;
        let mut hitm = 0.0;
        let mut matched = Vec::new();
        for (name, m) in &self.by_func {
            if func_substrings.iter().any(|s| name.contains(s)) {
                cycles += m.cycles_pct;
                dram += m.dram_pct;
                hitm += m.hitm;
                matched.push(name.as_str());
            }
        }
        let (td, td_pct) = self.topdown.dominant();
        let mut tags = Vec::new();
        if dram > 20.0 {
            tags.push(format!("DRAM-bound({dram:.0}% loads miss to DRAM)"));
        }
        if hitm > 0.0 {
            tags.push(format!("false-sharing({hitm:.0} HITM)"));
        }
        if tags.is_empty() {
            tags.push(format!("run-level {td}({td_pct:.0}%)"));
        }
        format!(
            "{} | cycles≈{cycles:.0}% | funcs[{}]",
            tags.join(", "),
            matched.join(",")
        )
    }
}

/// Build a `perf stat` command capturing TMA top-down counters around the
/// looped harness. CSV output (`-x,`) to stderr is parsed by `parse_topdown`.
pub fn perf_topdown_command(
    harness_bin: &Path,
    input: &Path,
    iters: usize,
    threads: usize,
    cpus: Option<&str>,
    csv_out: &Path,
) -> Command {
    let mut cmd = Command::new("perf");
    cmd.arg("stat")
        .arg("-x")
        .arg(",")
        .arg("-o")
        .arg(csv_out)
        // topdown metric group; on Intel this expands to the L1 TMA
        // breakdown. `--topdown` is the portable spelling; fall back to
        // explicit events if the kernel lacks it (caller can swap).
        .arg("--topdown");
    if let Some(c) = cpus {
        cmd.arg("--cpu").arg(c);
    }
    cmd.arg("--")
        .arg(harness_bin)
        .arg(input)
        .arg("--iters")
        .arg(iters.to_string())
        .arg("--threads")
        .arg(threads.to_string());
    cmd
}

/// Parse `perf stat --topdown` output for the four TMA percentages.
///
/// Modern `perf stat --topdown` (the box's perf 6.x) emits a space-aligned
/// HEADER row naming the metrics in column order, then a DATA row of the
/// percentages — NOT a per-line keyword+value CSV:
///
///   ` %  tma_backend_bound  %  tma_retiring  %  tma_bad_speculation  %  tma_frontend_bound`
///   `                  40.1               22.5                  21.3                   16.1`
///
/// So we (1) find the header row (the one carrying the `tma_*` / `topdown-*`
/// metric names), record the metric ORDER, then (2) map the next row's
/// floats positionally. We also keep a fallback for the old same-line CSV
/// form (`,,12.3,,topdown-retiring`).
pub fn parse_topdown(text: &str) -> TopDown {
    let mut td = TopDown::default();
    let classify = |name: &str, pct: f64, td: &mut TopDown| {
        let l = name.to_lowercase();
        if l.contains("retiring") {
            td.retiring = pct;
        } else if l.contains("bad") && (l.contains("spec") || l.contains("speculation")) {
            td.bad_speculation = pct;
        } else if l.contains("frontend") || l.contains("fe_bound") || l.contains("fe-bound") {
            td.frontend_bound = pct;
        } else if l.contains("backend") || l.contains("be_bound") || l.contains("be-bound") {
            td.backend_bound = pct;
        }
    };

    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        // Header row: contains the TMA metric tokens.
        let names: Vec<&str> = line
            .split_whitespace()
            .filter(|t| {
                let t = t.to_lowercase();
                t.contains("tma_") || t.contains("topdown")
            })
            .collect();
        if names.len() >= 2 && (lower.contains("retiring") || lower.contains("bound")) {
            // The DATA row is the next non-empty line of floats.
            if let Some(data) = lines.get(i + 1..).and_then(|rest| {
                rest.iter()
                    .find(|l| l.split_whitespace().any(|t| t.parse::<f64>().is_ok()))
            }) {
                let nums: Vec<f64> = data
                    .split_whitespace()
                    .filter_map(|t| t.trim_end_matches('%').parse::<f64>().ok())
                    .collect();
                for (name, pct) in names.iter().zip(nums.iter()) {
                    classify(name, *pct, &mut td);
                }
                return td;
            }
        }
        // Fallback: same-line keyword + percentage (old CSV / -x, form).
        if let Some(pct) = line
            .split(|c| c == ',' || c == ' ' || c == '\t')
            .find_map(|c| c.trim().trim_end_matches('%').parse::<f64>().ok())
        {
            if lower.contains("retiring")
                || lower.contains("bound")
                || (lower.contains("bad") && lower.contains("spec"))
            {
                classify(&lower, pct, &mut td);
            }
        }
    }
    td
}

/// Parse `perf report --stdio -n` overhead table → function → cycles_pct.
/// Lines look like: `    12.34%   1234   gzippy  [.] bootstrap_with_deflate_block`.
pub fn parse_perf_report(report: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for line in report.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(pct_tok) = line.split_whitespace().next() else {
            continue;
        };
        let Some(pct) = pct_tok.trim_end_matches('%').parse::<f64>().ok() else {
            continue;
        };
        // The symbol is after the `[.]` / `[k]` DSO marker.
        if let Some(idx) = line.find("[.]").or_else(|| line.find("[k]")) {
            let sym = line[idx + 3..].trim();
            // Strip trailing offset / template noise; keep the leading ident.
            let name = sym
                .split(|c: char| c == '+' || c == ' ')
                .next()
                .unwrap_or(sym);
            *out.entry(name.to_string()).or_insert(0.0) += pct;
        }
    }
    out
}
