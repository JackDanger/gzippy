//! Anatomy-counter PIN suite: exact T1 work-volume snapshots as a ratchet.
//!
//! The `anatomy-counters` feature exposes exact semantic-work-unit counters
//! (matchfinder probe attempts, table reads/writes, inserts, bytes allocated,
//! positions processed — see `src/compress/deflate/anatomy_counters.rs`). At
//! T1 these are DETERMINISTIC: same input, same level, same counts, on every
//! platform, with no valgrind — an instruction-count-ratchet equivalent that
//! runs anywhere. This suite pins EVERY exposed counter for every synthetic
//! fixture (`src/fixtures.rs`, hash-frozen) x levels {1, 2, 6, 9} at T1
//! against `tests/fingerprints/anatomy_pins.tsv`, exactly, zero tolerance.
//!
//! What a failure buys you: the message is a per-cell table of every changed
//! counter — old, new, absolute delta, % — sorted by |delta|, so the reader
//! can name the MECHANISM ("+5.5M fast_head_table_writes at L1 on binary =
//! len3 maintenance") without rerunning anything. Same philosophy as
//! `tests/fingerprint_suite.rs`, one layer deeper: fingerprints pin the
//! OUTPUT's shape; these pin the WORK VOLUME that produced it.
//!
//! To regenerate after an INTENTIONAL change (the TSV diff is the lever's
//! mechanism, reviewable line by line — commit it in the same PR):
//!
//! ```text
//! UPDATE_ANATOMY_PINS=1 cargo test --release --test anatomy_pins --features anatomy-counters
//! ```
//!
//! (An env var in TEST tooling is the standard snapshot-test pattern; the
//! no-env-knobs rule binds the SHIPPED binary, and this entire file plus the
//! counters it reads do not exist in a default build.)
//!
//! ## Why in-process, and why ONE `#[test]`
//!
//! `AnatomyCounters` is one process-wide static. `tests/anatomy_counters.rs`
//! spawns subprocesses because the CRATE unit-test binary runs many tests
//! concurrently; this file is its OWN integration binary containing exactly
//! one `#[test]`, so nothing else in the process can touch the static while
//! it runs. In-process measurement lets us `reset()` between cells and call
//! the production T1 entry point (`encode_gzip_bytes_to_vec` — the same
//! function the anatomy-wall root span instruments) directly.
//!
//! ## Determinism
//!
//! Every cell is measured TWICE (reset, compress, snapshot; again) and the
//! two snapshots must be identical before the pin comparison runs — so a
//! counter going nondeterministic fails loudly as ITS OWN defect, never as a
//! confusing pin mismatch. As of pin generation, ALL exposed counters are
//! deterministic across identical T1 runs; NONE is excluded. One counter
//! needed a warm-up rather than an exclusion: `huffman_make_code_calls`
//! includes `StaticCodes::build`'s 2 calls, and `StaticCodes` is built
//! exactly once per PROCESS (`OnceLock`, parse/mod.rs) — the double-run
//! guard caught the first process run reading 2 higher than every later run
//! (50 vs 48 on the first cell). The fix is `warm_up()` below: one throwaway
//! compression per pinned level before any measurement, so every measured
//! cell sees identical steady-state (per-invocation) work and the counter
//! stays pinned instead of dropped. If this guard ever fires again, exclude
//! that counter here WITH the measured reason, and regenerate.
//!
//! ## What pins are NOT
//!
//! Counters are exact work volume, not a wall-clock price: a measured -27%
//! write-count moved wall 0%, while a -21% Ir moved wall 5.4%. Pins catch
//! CHANGES; pricing a change needs per-arch calibration (paired wall runs on
//! the frozen box). A pin diff names the mechanism — it never grades it.

#![cfg(feature = "anatomy-counters")]

use gzippy::compress::deflate::anatomy_counters;
use gzippy::compress::deflate::encode_gzip_bytes_to_vec;
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The pinned level grid. T1 only (see module doc); T>1 work distribution is
/// schedule-dependent and is NOT pinned here.
const LEVELS: &[u32] = &[1, 2, 6, 9];

const REGEN_CMD: &str =
    "UPDATE_ANATOMY_PINS=1 cargo test --release --test anatomy_pins --features anatomy-counters";

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/anatomy_pins.tsv")
}

/// Parse the flat `{"key":123,...}` JSON `AnatomyCounters::to_json` emits
/// (same shape, same parser rationale as `tests/anatomy_counters.rs`).
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

/// One measured cell: reset the global counters, compress in-process at T1,
/// snapshot every counter, THEN verify the stream roundtrips (decode after
/// the snapshot so a decoder can never perturb what we pinned).
fn measure(data: &[u8], level: u32) -> BTreeMap<String, u64> {
    anatomy_counters::reset();
    let compressed = encode_gzip_bytes_to_vec(data, level);
    let snapshot = parse_flat_json(&anatomy_counters::COUNTERS.to_json());

    // Correctness rides along: a pinned count for an invalid stream would
    // ratchet garbage. Independent decoder (flate2), byte-exact.
    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("encode_gzip_bytes_to_vec must produce a valid gzip stream");
    }
    assert_eq!(decoded, data, "roundtrip failed at L{level}");
    snapshot
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
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

/// Render every changed counter for one fixture x level cell as a table
/// sorted by |delta| descending — the biggest mechanism first.
fn render_cell_diff(
    fixture: &str,
    level: u32,
    pinned: &BTreeMap<String, u64>,
    now: &BTreeMap<String, u64>,
) -> Option<String> {
    // (name, old, new) for every counter present on either side.
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
        "ANATOMY PINS MOVED {fixture}:L{level}:T1 — {} counter(s) changed\n    {:<30} {:>16} {:>16} {:>14} {:>9}\n",
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
    "# anatomy_pins.tsv — exact per-counter work-volume pins for the T1 encoder.\n\
     #\n\
     # Cargo feature: anatomy-counters. These counters exist ONLY in that\n\
     # feature's build; the default (shipped) build compiles every call site to\n\
     # zero bytes and never reads this file.\n\
     # Scope: T1 ONLY (in-process encode_gzip_bytes_to_vec, the production T1\n\
     # entry point). T>1 work distribution is schedule-dependent and unpinned.\n\
     # Grid: src/fixtures.rs synthetic fixtures x levels {1, 2, 6, 9}.\n\
     #\n\
     # Counters are a calibrated wall proxy ONLY after per-arch calibration:\n\
     # a measured -27% write-count moved wall 0%, while a -21% Ir moved wall\n\
     # 5.4%. Pins catch CHANGES — they do not price them. A pin diff names the\n\
     # mechanism; the wall verdict still needs paired runs on the frozen box.\n\
     #\n\
     # Regenerate (the diff is the change's mechanism — commit it in the same PR):\n\
     #   UPDATE_ANATOMY_PINS=1 cargo test --release --test anatomy_pins --features anatomy-counters\n\
     fixture\tlevel\tcounter\tvalue\n"
        .to_string()
}

/// The whole suite as ONE test: serial by construction, so nothing else in
/// this process touches the global counter static between reset and snapshot.
#[test]
fn anatomy_pins_match() {
    // ── Warm-up: retire once-per-PROCESS work before any measurement. ─────
    // `StaticCodes` (parse/mod.rs) is built via `OnceLock` on the first
    // compression this process runs, adding exactly 2 `make_huffman_code`
    // calls to that one run — measured, not theorized: the double-run guard
    // below caught it (50 vs 48). One throwaway compression per pinned level
    // (levels route to different strategies, each free to lazy-init its own
    // statics) puts every measured cell in identical steady state.
    for &level in LEVELS {
        let _ = encode_gzip_bytes_to_vec(b"warm-up: retire OnceLock init work", level);
    }

    // ── Measure every cell, twice, and demand exact determinism. ──────────
    let mut current: BTreeMap<(String, u32), BTreeMap<String, u64>> = BTreeMap::new();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        for &level in LEVELS {
            let a = measure(&data, level);
            let b = measure(&data, level);
            if a != b {
                let diff = render_cell_diff(name, level, &a, &b)
                    .unwrap_or_else(|| "(no per-counter diff?)".into());
                panic!(
                    "\nNONDETERMINISTIC COUNTER(S) at {name}:L{level}:T1 — two identical \
                     in-process runs disagreed.\nRun A = 'pinned' column, run B = 'now':\n{diff}\n\
                     A T1 counter must be exact. Find the source (uninitialized read? time/\n\
                     address-dependent branch?) or exclude the counter here with the reason.\n"
                );
            }
            current.insert((name.to_string(), level), a);
        }
    }

    // ── Regeneration mode: rewrite the TSV instead of asserting. ──────────
    if std::env::var("UPDATE_ANATOMY_PINS").as_deref() == Ok("1") {
        let mut tsv = tsv_header();
        for &name in fixtures::NAMES {
            for &level in LEVELS {
                for (counter, value) in &current[&(name.to_string(), level)] {
                    let _ = writeln!(tsv, "{name}\t{level}\t{counter}\t{value}");
                }
            }
        }
        std::fs::write(pins_path(), tsv).expect("write anatomy_pins.tsv");
        eprintln!(
            "anatomy_pins: regenerated {} ({} cells x {} counters)",
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
    let mut pinned: BTreeMap<(String, u32), BTreeMap<String, u64>> = BTreeMap::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "malformed pin row: {line:?}");
        pinned
            .entry((cols[0].to_string(), cols[1].parse().unwrap()))
            .or_default()
            .insert(cols[2].to_string(), cols[3].parse().unwrap());
    }

    let empty = BTreeMap::new();
    let mut failures = Vec::new();
    for (cell, now) in &current {
        let pins = pinned.get(cell).unwrap_or(&empty);
        if pins.is_empty() {
            failures.push(format!(
                "UNPINNED CELL {}:L{}:T1 — no rows in the TSV; regenerate pins",
                cell.0, cell.1
            ));
            continue;
        }
        if let Some(diff) = render_cell_diff(&cell.0, cell.1, pins, now) {
            failures.push(diff);
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            failures.push(format!(
                "STALE PIN CELL {}:L{}:T1 — pinned but no longer measured (fixture or \
                 level grid changed); regenerate pins",
                cell.0, cell.1
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\nT1 work volume changed. If INTENTIONAL (a lever), regenerate the pin in \
         this PR —\nits diff is your mechanism, stated for review:\n    {REGEN_CMD}\n\
         If you did not mean to change encoder work, the counters above name what moved.\n\
         Remember: a count delta is a MECHANISM, not a price — wall verdicts still need\n\
         paired runs on the frozen box (a -27% write count once moved wall 0%).\n",
        failures.join("\n")
    );
}
