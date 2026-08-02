//! The mechanism-fingerprint suite: the campaign goal as executable tests.
//!
//! Three layers, each layer's failure output the next investigation's input:
//!
//! 1. `won_cells_stay_won` — the LEDGER. Every cell (fixture, level, threads,
//!    rival) we have ever beaten on size must still be beaten. Monotone in
//!    the goal, silent about the means: no implementation detail is pinned,
//!    so no tunnel to dig. When it fails, the message is the per-axis
//!    fingerprint diff against that rival — the mechanism, not a boolean.
//!
//! 2. `fingerprints_match_pins` — the SNAPSHOT. Our own per-cell fingerprints
//!    must equal tests/fingerprints/ours.tsv. An INTENTIONAL change (a lever)
//!    regenerates the file in the same PR — the file's diff IS the lever's
//!    mechanism, reviewable line by line. An UNINTENTIONAL change fails here
//!    with the axis that moved.
//!
//! 3. Correctness rides along: every fingerprinted stream's decoded size is
//!    verified against the input, and the walker itself errors on any
//!    malformed stream.
//!
//! Everything runs against the REAL binary (CARGO_BIN_EXE_gzippy), on frozen
//! synthetic fixtures (src/fixtures.rs — hash-pinned), in seconds, anywhere.
//! The promotion rule on the real corpus remains the final adjudicator; this
//! suite is the per-commit loop.

use gzippy::decompress::block_walker::{fingerprint_gzip, StreamFingerprint};
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::process::Command;

const LEVELS: &[u32] = &[1, 2, 6, 9];
const THREADS: &[u32] = &[1, 4];

type Cell = (String, u32, u32, String);

fn parse_pins(body: &str, ncols: usize) -> Vec<Vec<String>> {
    body.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("fixture\t") && !l.trim().is_empty())
        .map(|l| l.split('\t').map(|c| c.to_string()).collect::<Vec<_>>())
        .filter(|c| c.len() == ncols)
        .collect()
}

fn ours() -> BTreeMap<Cell, StreamFingerprint> {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let mut out = BTreeMap::new();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.path().join(name);
        std::fs::write(&path, &data).unwrap();
        for &level in LEVELS {
            for &threads in THREADS {
                let o = Command::new(bin)
                    .args([
                        &format!("-{level}"),
                        "-p",
                        &threads.to_string(),
                        "-c",
                        path.to_str().unwrap(),
                    ])
                    .output()
                    .unwrap();
                assert!(
                    o.status.success(),
                    "gzippy failed on {name} L{level} T{threads}"
                );
                let fp = fingerprint_gzip(&o.stdout).unwrap_or_else(|e| {
                    panic!("unparseable stream {name} L{level} T{threads}: {e}")
                });
                assert_eq!(
                    fp.decoded_bytes,
                    data.len() as u64,
                    "stream for {name} L{level} T{threads} decodes to the wrong size"
                );
                out.insert((name.to_string(), level, threads, "gzippy".into()), fp);
            }
        }
    }
    out
}

fn render_diff(ours: &StreamFingerprint, other: &StreamFingerprint) -> String {
    ours.diff(other)
        .iter()
        .map(|(axis, o, r)| {
            let rel = if *r == 0 {
                "new".to_string()
            } else {
                format!("{:+.1}%", (*o as f64 - *r as f64) * 100.0 / *r as f64)
            };
            format!("    {axis:<14} ours {o:>12}  pinned {r:>12}  ({rel})")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Layer 1: the goal itself. Cells, once won, stay won.
#[test]
fn won_cells_stay_won() {
    let ledger = std::fs::read_to_string("tests/fingerprints/ledger.tsv")
        .expect("tests/fingerprints/ledger.tsv missing — cargo run --release --example fingerprint_tool -- ledger");
    let rivals_tsv = std::fs::read_to_string("tests/fingerprints/rivals.tsv").unwrap_or_default();
    let rival_fps: BTreeMap<Cell, StreamFingerprint> =
        parse_pins(&rivals_tsv, 4 + StreamFingerprint::TSV_FIELDS.len())
            .into_iter()
            .filter_map(|c| {
                let vals: Vec<u64> = c[4..].iter().filter_map(|v| v.parse().ok()).collect();
                Some((
                    (
                        c[0].clone(),
                        c[1].parse().ok()?,
                        c[2].parse().ok()?,
                        c[3].clone(),
                    ),
                    StreamFingerprint::from_tsv_values(&vals)?,
                ))
            })
            .collect();
    let ours = ours();
    let mut regressions = Vec::new();
    for row in parse_pins(&ledger, 5) {
        let (fixture, level, threads, rival) = (
            row[0].clone(),
            row[1].parse::<u32>().unwrap(),
            row[2].parse::<u32>().unwrap(),
            row[3].clone(),
        );
        let rival_bytes: u64 = row[4].parse().unwrap();
        let Some(ofp) = ours.get(&(fixture.clone(), level, threads, "gzippy".into())) else {
            continue; // ledger may cover cells outside this grid (e.g. igzip rows pinned on a box)
        };
        if ofp.file_bytes > rival_bytes {
            let mech = rival_fps
                .get(&(fixture.clone(), level, threads, rival.clone()))
                .map(|rfp| render_diff(ofp, rfp))
                .unwrap_or_default();
            regressions.push(format!(
                "LEDGER REGRESSION {fixture}:L{level}:T{threads} vs {rival}: ours {} > rival {rival_bytes} (+{} B)\n{mech}",
                ofp.file_bytes,
                ofp.file_bytes - rival_bytes,
            ));
        }
    }
    assert!(
        regressions.is_empty(),
        "\n{}\n\nA won cell regressed. Fix the change or revert it — the ledger is append-only\nand is never edited to fit a result.\n",
        regressions.join("\n\n")
    );
}

/// Layer 2: the mechanism snapshot. Any output change shows up as a per-axis
/// diff; intentional changes regenerate the pin in the same PR.
#[test]
fn fingerprints_match_pins() {
    let pinned_tsv = std::fs::read_to_string("tests/fingerprints/ours.tsv")
        .expect("tests/fingerprints/ours.tsv missing — cargo run --release --example fingerprint_tool -- pin-ours");
    let pinned: BTreeMap<Cell, StreamFingerprint> =
        parse_pins(&pinned_tsv, 4 + StreamFingerprint::TSV_FIELDS.len())
            .into_iter()
            .filter_map(|c| {
                let vals: Vec<u64> = c[4..].iter().filter_map(|v| v.parse().ok()).collect();
                Some((
                    (
                        c[0].clone(),
                        c[1].parse().ok()?,
                        c[2].parse().ok()?,
                        c[3].clone(),
                    ),
                    StreamFingerprint::from_tsv_values(&vals)?,
                ))
            })
            .collect();
    let ours = ours();
    let mut diffs = Vec::new();
    for (cell, ofp) in &ours {
        match pinned.get(cell) {
            None => diffs.push(format!("UNPINNED CELL {cell:?} — regenerate pins")),
            Some(pfp) if pfp != ofp => diffs.push(format!(
                "FINGERPRINT MOVED {}:L{}:T{}\n{}",
                cell.0,
                cell.1,
                cell.2,
                render_diff(ofp, pfp)
            )),
            _ => {}
        }
    }
    assert!(
        diffs.is_empty(),
        "\n{}\n\nOutput changed. If INTENTIONAL (a lever), regenerate the pin in this PR —\nits diff is your mechanism, stated for review:\n    cargo run --release --example fingerprint_tool -- pin-ours\nIf you did not mean to change output, the axes above name what moved.\n",
        diffs.join("\n\n")
    );
}
