//! Startup-cost PIN suite: exact setup-work snapshots for a 1-byte compress.
//!
//! Compressing ONE byte is almost pure setup: table builds, arena/buffer
//! allocation, thread-pool spin-up at T>1 — the per-invocation fixed cost that
//! every wall verdict pays before the first real input byte. The campaign's
//! T>1 wall verdicts carry a ~6.2 ms invocation floor; a regression in that
//! floor (a new table build, arena growth, an extra allocation on the init
//! path) erodes EVERY wall cell at once and is invisible in per-byte counters
//! on real corpora. This suite makes it visible on a laptop: it pins the exact
//! `anatomy-counters` totals (including `alloc_events`/`alloc_bytes`) for a
//! 1-byte in-process compress at levels {1, 6, 9} x threads {1, 4} against
//! `tests/fingerprints/startup_cost.tsv`, exactly, zero tolerance.
//!
//! Same philosophy and mechanics as `tests/anatomy_pins.rs`, one input class
//! down: anatomy_pins pins the work volume of REAL inputs (1 MiB fixtures,
//! T1 only); this file pins the FIXED COST an invocation pays before any
//! input, and is the only pin suite that covers the T>1 entry point at all.
//!
//! To regenerate after an INTENTIONAL change (the TSV diff is the change's
//! init-cost mechanism, reviewable line by line — commit it in the same PR):
//!
//! ```text
//! UPDATE_STARTUP_COST=1 cargo test --release --test startup_cost --features anatomy-counters
//! ```
//!
//! (An env var in TEST tooling is the standard snapshot-test pattern; the
//! no-env-knobs rule binds the SHIPPED binary, and this entire file plus the
//! counters it reads do not exist in a default build.)
//!
//! ## Why in-process, and why ONE `#[test]`
//!
//! Identical to `anatomy_pins.rs`: `AnatomyCounters` is one process-wide
//! static, so this file is its own integration binary containing exactly one
//! `#[test]` — nothing else can touch the static between reset and snapshot.
//! T1 cells call the production T1 entry point (`encode_gzip_bytes_to_vec`);
//! T4 cells call `gzippy::compress_with_threads` (the library route into
//! `PipelinedGzEncoder`, the sole production T>1 path), so pool spin-up and
//! the T>1 chunking/framing setup are inside the measured window.
//!
//! ## Determinism, including at T4
//!
//! Every cell is measured TWICE (reset, compress, snapshot; again) and the two
//! snapshots must agree on every non-excluded counter before the pin
//! comparison runs. T>1 work DISTRIBUTION is schedule-dependent in general —
//! which is why `anatomy_pins.rs` refuses to pin T>1 on real inputs — but a
//! 1-byte input is a single chunk on a single worker, and the double-run guard
//! verified (at pin generation, on aarch64-apple-darwin AND enforced on every
//! run since) that every exposed counter, `alloc_events`/`alloc_bytes`
//! included, is run-to-run exact at T4 for this input. `RACY_COUNTERS` below
//! is therefore EMPTY today; if the guard ever fires, add the counter there
//! with the measured reason rather than loosening the exact compare.

#![cfg(feature = "anatomy-counters")]

use gzippy::compress::deflate::anatomy_counters;
use gzippy::compress::deflate::encode_gzip_bytes_to_vec;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The pinned (level, threads) grid. Levels 1/6/9 span the strategy ladder
/// (fast / lazy / near-optimal — each free to lazy-init its own statics);
/// T1 is the single-member path, T4 the pipelined parallel path, so the two
/// rows measure DIFFERENT code's setup work at the same level.
const CELLS: &[(u32, usize)] = &[(1, 1), (1, 4), (6, 1), (6, 4), (9, 1), (9, 4)];

/// The one-byte input. Content is irrelevant (any single byte is a literal at
/// every level); what matters is that per-byte work is ~zero, so every count
/// below is setup cost.
const INPUT: &[u8] = b"x";

/// Counters excluded from BOTH the determinism guard and the pins because a
/// double run measured them racy at T>1. EMPTY as of pin generation: all 51
/// exposed counters were run-to-run exact at T1 and T4 for the 1-byte input
/// (single chunk, single worker — no schedule-dependent distribution). Add a
/// counter here ONLY with the measured double-run disagreement in a comment.
const RACY_COUNTERS: &[&str] = &[];

const REGEN_CMD: &str =
    "UPDATE_STARTUP_COST=1 cargo test --release --test startup_cost --features anatomy-counters";

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/startup_cost.tsv")
}

/// Parse the flat `{"key":123,...}` JSON `AnatomyCounters::to_json` emits
/// (same shape, same parser rationale as `tests/anatomy_pins.rs`).
fn parse_flat_json(s: &str) -> BTreeMap<String, u64> {
    let body = s
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or_else(|| panic!("not a flat JSON object: {s}"));
    let mut map = BTreeMap::new();
    for pair in body.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed key:value pair {pair:?}"));
        map.insert(
            k.trim().trim_matches('"').to_string(),
            v.trim()
                .parse()
                .unwrap_or_else(|_| panic!("non-integer value for {k}: {v:?}")),
        );
    }
    map
}

/// One measured cell: reset the global counters, compress the 1-byte input
/// in-process at (level, threads), snapshot every counter, THEN verify the
/// stream roundtrips (decode after the snapshot so a decoder can never
/// perturb what we pinned). Racy-excluded counters are dropped from the
/// snapshot so they can neither trip the determinism guard nor be pinned.
fn measure(level: u32, threads: usize) -> BTreeMap<String, u64> {
    anatomy_counters::reset();
    let compressed = if threads == 1 {
        encode_gzip_bytes_to_vec(INPUT, level)
    } else {
        gzippy::compress_with_threads(INPUT, level as u8, threads)
            .expect("T>1 compress of 1 byte must succeed")
    };
    let mut snapshot = parse_flat_json(&anatomy_counters::COUNTERS.to_json());
    for racy in RACY_COUNTERS {
        snapshot.remove(*racy);
    }

    // Correctness rides along: a pinned count for an invalid stream would
    // ratchet garbage. Independent decoder (flate2), byte-exact.
    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .unwrap_or_else(|e| panic!("invalid gzip stream at L{level}/T{threads}: {e}"));
    }
    assert_eq!(decoded, INPUT, "roundtrip failed at L{level}/T{threads}");
    snapshot
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn commas_i(n: i128) -> String {
    if n < 0 {
        format!("-{}", commas(n.unsigned_abs() as u64))
    } else {
        format!("+{}", commas(n as u64))
    }
}

/// Render every changed counter for one (level, threads) cell as a table
/// sorted by |delta| descending — the biggest mechanism first. Identical
/// rendering to `tests/anatomy_pins.rs`.
fn render_cell_diff(
    level: u32,
    threads: usize,
    pinned: &BTreeMap<String, u64>,
    now: &BTreeMap<String, u64>,
) -> Option<String> {
    let mut rows: Vec<(&str, Option<u64>, Option<u64>)> = Vec::new();
    for (k, &old) in pinned {
        let new = now.get(k).copied();
        if new != Some(old) {
            rows.push((k, Some(old), new));
        }
    }
    for (k, &new) in now {
        if !pinned.contains_key(k) {
            rows.push((k, None, Some(new)));
        }
    }
    if rows.is_empty() {
        return None;
    }
    rows.sort_by_key(|(_, old, new)| {
        std::cmp::Reverse((new.unwrap_or(0) as i128 - old.unwrap_or(0) as i128).unsigned_abs())
    });

    let mut out = format!(
        "STARTUP COST MOVED 1byte:L{level}:T{threads} — {} counter(s) changed\n    {:<30} {:>16} {:>16} {:>14} {:>9}\n",
        rows.len(),
        "counter",
        "pinned",
        "now",
        "delta",
        "%"
    );
    for (name, old, new) in rows {
        match (old, new) {
            (Some(o), Some(n)) => {
                let delta = n as i128 - o as i128;
                let pct = if o == 0 {
                    "new".to_string()
                } else {
                    format!("{:+.1}%", delta as f64 * 100.0 / o as f64)
                };
                let _ = writeln!(
                    out,
                    "    {name:<30} {:>16} {:>16} {:>14} {pct:>9}",
                    commas(o),
                    commas(n),
                    commas_i(delta),
                );
            }
            (Some(o), None) => {
                let _ = writeln!(
                    out,
                    "    {name:<30} {:>16} {:>16} {:>14} {:>9}",
                    commas(o),
                    "ABSENT",
                    "-",
                    "gone"
                );
            }
            (None, Some(n)) => {
                let _ = writeln!(
                    out,
                    "    {name:<30} {:>16} {:>16} {:>14} {:>9}",
                    "UNPINNED",
                    commas(n),
                    "-",
                    "new"
                );
            }
            (None, None) => unreachable!(),
        }
    }
    Some(out)
}

fn tsv_header() -> String {
    "# startup_cost.tsv — exact setup-work pins for a 1-byte in-process compress.\n\
     #\n\
     # Cargo feature: anatomy-counters. These counters exist ONLY in that\n\
     # feature's build; the default (shipped) build compiles every call site to\n\
     # zero bytes and never reads this file.\n\
     # A 1-byte input makes per-byte work ~zero, so every value below is\n\
     # per-invocation FIXED cost: table builds, buffer allocation, T>1 pool\n\
     # spin-up and seam framing. This is the laptop-visible proxy for the\n\
     # ~6.2 ms invocation floor that T>1 wall verdicts carry — an init-cost\n\
     # regression moves these counts BEFORE it erodes every wall cell at once.\n\
     # Grid: 1-byte input x levels {1, 6, 9} x threads {1, 4}. T1 = production\n\
     # single-member entry (encode_gzip_bytes_to_vec); T4 = library route into\n\
     # PipelinedGzEncoder, the sole production T>1 path. T4 determinism is\n\
     # enforced by a double-run guard on every run (1 byte = single chunk on a\n\
     # single worker; no counter was racy at pin generation).\n\
     #\n\
     # Counts catch CHANGES; they do not price them. A pin diff names the\n\
     # mechanism; the wall verdict still needs paired runs on the frozen box.\n\
     #\n\
     # Regenerate (the diff is the change's mechanism — commit it in the same PR):\n\
     #   UPDATE_STARTUP_COST=1 cargo test --release --test startup_cost --features anatomy-counters\n\
     level\tthreads\tcounter\tvalue\n"
        .to_string()
}

/// The whole suite as ONE test: serial by construction, so nothing else in
/// this process touches the global counter static between reset and snapshot.
#[test]
fn startup_cost_pins_match() {
    // ── Warm-up: retire once-per-PROCESS work before any measurement. ─────
    // Same trick, same receipt as `anatomy_pins.rs`: `StaticCodes`
    // (parse/mod.rs) is a `OnceLock` built on the FIRST compression in the
    // process (+2 `huffman_make_code_calls` on that run only), and each
    // (level, threads) route is free to lazy-init statics of its own. One
    // throwaway run per cell puts every measured cell in identical
    // steady state — this suite pins the PER-INVOCATION setup cost, not the
    // once-per-process cost.
    for &(level, threads) in CELLS {
        let _ = measure(level, threads);
    }

    // ── Measure every cell, twice, and demand exact determinism. ──────────
    // At T4 this is the racy-counter check the module doc describes: any
    // schedule-dependent counter fails HERE, loudly, as its own defect —
    // never as a confusing pin mismatch.
    let mut current: BTreeMap<(u32, usize), BTreeMap<String, u64>> = BTreeMap::new();
    for &(level, threads) in CELLS {
        let a = measure(level, threads);
        let b = measure(level, threads);
        if a != b {
            let diff = render_cell_diff(level, threads, &a, &b)
                .unwrap_or_else(|| "(no per-counter diff?)".into());
            panic!(
                "\nNONDETERMINISTIC COUNTER(S) at 1byte:L{level}:T{threads} — two identical \
                 in-process runs disagreed.\nRun A = 'pinned' column, run B = 'now':\n{diff}\n\
                 A pinned counter must be exact. If this is T>1 schedule noise, add the\n\
                 counter to RACY_COUNTERS with the measured reason; otherwise find the\n\
                 source (uninitialized read? time/address-dependent branch?).\n"
            );
        }
        current.insert((level, threads), a);
    }

    // ── Regeneration mode: rewrite the TSV instead of asserting. ──────────
    if std::env::var("UPDATE_STARTUP_COST").as_deref() == Ok("1") {
        let mut tsv = tsv_header();
        for &(level, threads) in CELLS {
            for (counter, value) in &current[&(level, threads)] {
                let _ = writeln!(tsv, "{level}\t{threads}\t{counter}\t{value}");
            }
        }
        std::fs::write(pins_path(), tsv).expect("write startup_cost.tsv");
        eprintln!(
            "startup_cost: regenerated {} ({} cells x {} counters)",
            pins_path().display(),
            current.len(),
            current.values().next().map_or(0, |c| c.len())
        );
        return;
    }

    // ── Compare against the pinned TSV, exactly. ──────────────────────────
    let tsv = std::fs::read_to_string(pins_path()).unwrap_or_else(|e| {
        panic!(
            "{} missing ({e}) — generate it once:\n    {REGEN_CMD}",
            pins_path().display()
        )
    });
    let mut pinned: BTreeMap<(u32, usize), BTreeMap<String, u64>> = BTreeMap::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("level\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "malformed pin row: {line:?}");
        pinned
            .entry((cols[0].parse().unwrap(), cols[1].parse().unwrap()))
            .or_default()
            .insert(cols[2].to_string(), cols[3].parse().unwrap());
    }

    let empty = BTreeMap::new();
    let mut failures = Vec::new();
    for (&(level, threads), now) in &current {
        let pins = pinned.get(&(level, threads)).unwrap_or(&empty);
        if pins.is_empty() {
            failures.push(format!(
                "UNPINNED CELL 1byte:L{level}:T{threads} — no rows in the TSV; regenerate pins"
            ));
            continue;
        }
        if let Some(diff) = render_cell_diff(level, threads, pins, now) {
            failures.push(diff);
        }
    }
    for &(level, threads) in pinned.keys() {
        if !current.contains_key(&(level, threads)) {
            failures.push(format!(
                "STALE PIN CELL 1byte:L{level}:T{threads} — pinned but no longer measured \
                 (cell grid changed); regenerate pins"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\nPer-invocation SETUP work changed. If INTENTIONAL (a lever), regenerate the \
         pin in this PR —\nits diff is your mechanism, stated for review:\n    {REGEN_CMD}\n\
         If you did not mean to change init cost, the counters above name what moved.\n\
         Init cost is the ~6.2 ms invocation floor every T>1 wall verdict pays — a\n\
         regression here erodes every wall cell at once.\n",
        failures.join("\n")
    );
}
