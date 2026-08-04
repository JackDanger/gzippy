//! ldx divergence CENSUS pins: the exact per-class anatomy of where our T1
//! encoder's DEFLATE token stream departs from the in-tree exact libdeflate
//! port (`src/compress/ldx/`), pinned against
//! `tests/fingerprints/ldx_census.tsv`, exactly, zero tolerance.
//!
//! The oracle (`src/compress/ldx_oracle.rs`, PR #264) classifies every
//! divergent position: we-literal-they-match, we-match-they-literal,
//! both-match-different-len, both-match-different-dist, misaligned starts
//! (the positional cascade), and block-boundary framing diffs. L1 is the only
//! level that diverges today, and its counts ARE the anatomy of the
//! campaign's live L1 size gap. Nothing pinned them before this file: an L1
//! lever could shift parse structure invisibly.
//!
//! Grid: `src/fixtures.rs` synthetic fixtures (hash-frozen; never corpus
//! files — open-source repo) x levels {1, 2, 6, 9} at T1. Levels 2/6/9 are
//! byte-tie levels: their pinned rows are all-zero, so ANY divergence there
//! fails as its own loud event.
//!
//! ## These are SNAPSHOT pins, not aspirations
//!
//! We need NOT converge to libdeflate's parse — the goal is size <= theirs,
//! not identity (byte-identity is a cage, not an asset). A changed census is
//! not automatically bad. The pin exists so that parse-structure movement is
//! DELIBERATE and visible in the diff of the pin file, reviewable class by
//! class next to the size delta it bought.
//!
//! To regenerate after an INTENTIONAL change (commit the TSV diff in the
//! same PR — it is the change's parse-structure mechanism, stated for
//! review):
//!
//! ```text
//! UPDATE_LDX_CENSUS=1 cargo test --release --test ldx_census
//! ```
//!
//! (An env var in TEST tooling is the standard snapshot pattern; the
//! no-env-knobs rule binds the SHIPPED binary, which never reads this.)

use gzippy::compress::ldx_oracle::divergence_at_level;
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The pinned grid. L1 is the divergent level; 2/6/9 are the byte-tie levels
/// whose zero-divergence status this suite also pins (a divergence appearing
/// there is parse movement exactly as much as an L1 count changing).
const LEVELS: &[u32] = &[1, 2, 6, 9];

const REGEN_CMD: &str = "UPDATE_LDX_CENSUS=1 cargo test --release --test ldx_census";

/// The census of one (fixture, level) cell. Field order == TSV column order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Census {
    /// Uncompressed position of the first divergent decision; None if the
    /// two token streams are decision-identical.
    first_pos: Option<u64>,
    total_divergent: u64,
    we_literal_they_match: u64,
    we_match_they_literal: u64,
    both_match_different_len: u64,
    both_match_different_dist: u64,
    misaligned_starts: u64,
    block_boundary: u64,
    /// Raw DEFLATE stream sizes (no gzip framing), both sides.
    ours_bytes: u64,
    ldx_bytes: u64,
}

/// The counted columns, named — one source of truth for TSV I/O and the
/// failure table. `first_pos` is handled separately (it is positional, not a
/// count, and may be absent).
const COUNT_COLS: &[&str] = &[
    "total_divergent",
    "we_literal_they_match",
    "we_match_they_literal",
    "both_match_different_len",
    "both_match_different_dist",
    "misaligned_starts",
    "block_boundary",
    "ours_bytes",
    "ldx_bytes",
];

impl Census {
    fn counts(&self) -> [u64; 9] {
        [
            self.total_divergent,
            self.we_literal_they_match,
            self.we_match_they_literal,
            self.both_match_different_len,
            self.both_match_different_dist,
            self.misaligned_starts,
            self.block_boundary,
            self.ours_bytes,
            self.ldx_bytes,
        ]
    }

    fn first_pos_str(&self) -> String {
        self.first_pos.map_or_else(|| "-".into(), |p| p.to_string())
    }
}

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/ldx_census.tsv")
}

/// Run the oracle for one cell. The oracle itself verifies both streams
/// decode to the same length; its internal invariants (total == class sum)
/// are re-checked here so a pin can never encode an inconsistent report.
fn measure(input: &[u8], name: &str, level: u32) -> Census {
    let r = divergence_at_level(input, level)
        .unwrap_or_else(|| panic!("{name} L{level}: ldx does not implement this level"))
        .unwrap_or_else(|e| panic!("{name} L{level}: oracle failed: {e}"));
    let c = Census {
        first_pos: r.first.map(|f| f.pos),
        total_divergent: r.total_divergent(),
        we_literal_they_match: r.we_literal_they_match,
        we_match_they_literal: r.we_match_they_literal,
        both_match_different_len: r.both_match_different_len,
        both_match_different_dist: r.both_match_different_dist,
        misaligned_starts: r.misaligned_starts,
        block_boundary: r.block_boundary,
        ours_bytes: r.ours.compressed_len as u64,
        ldx_bytes: r.ldx.compressed_len as u64,
    };
    // Report self-consistency: the total is definitionally the class sum,
    // and a divergence count without a first position (or vice versa) would
    // be an oracle bug, not a parse movement.
    assert_eq!(
        c.total_divergent,
        c.counts()[1..7].iter().sum::<u64>(),
        "{name} L{level}: oracle total != class sum"
    );
    assert_eq!(
        c.first_pos.is_some(),
        c.total_divergent > 0,
        "{name} L{level}: first-divergence presence disagrees with the counts"
    );
    c
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
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

/// Render one cell's pinned-vs-now diff: first-divergence position, then a
/// per-class table old -> new with deltas (the size rows put the size delta
/// on both sides right next to the class movement that produced it).
fn render_cell_diff(fixture: &str, level: u32, pinned: &Census, now: &Census) -> String {
    let mut out = format!(
        "LDX CENSUS MOVED {fixture}:L{level}:T1\n    {:<26} {:>14} {:>14} {:>12}\n",
        "field", "pinned", "now", "delta"
    );
    if pinned.first_pos != now.first_pos {
        let _ = writeln!(
            out,
            "    {:<26} {:>14} {:>14} {:>12}",
            "first_divergence_pos",
            pinned.first_pos_str(),
            now.first_pos_str(),
            "moved"
        );
    }
    for (name, (o, n)) in COUNT_COLS
        .iter()
        .zip(pinned.counts().into_iter().zip(now.counts()))
    {
        if o != n {
            let _ = writeln!(
                out,
                "    {name:<26} {:>14} {:>14} {:>12}",
                commas(o),
                commas(n),
                commas_i(n as i128 - o as i128)
            );
        }
    }
    out
}

fn tsv_header() -> String {
    format!(
        "# ldx_census.tsv — per-class divergence census of the shipped T1 encoder vs the\n\
         # exact libdeflate port (src/compress/ldx/), via the oracle in\n\
         # src/compress/ldx_oracle.rs.\n\
         #\n\
         # Grid: src/fixtures.rs synthetic fixtures x levels {{1, 2, 6, 9}} at T1.\n\
         # Levels 2/6/9 are byte-tie levels: their rows pin divergence == 0.\n\
         # ours_bytes / ldx_bytes are raw DEFLATE stream sizes (no gzip framing).\n\
         # first_pos is the uncompressed position of the first divergent decision\n\
         # ('-' when the token streams are decision-identical).\n\
         #\n\
         # These are SNAPSHOT pins, not aspirations. We need NOT converge to\n\
         # libdeflate's parse — the goal is size <= theirs, not identity — and a\n\
         # changed census is not automatically bad. The pin exists so that\n\
         # parse-structure movement is DELIBERATE and visible in the diff of this\n\
         # file, reviewed next to the size delta it bought.\n\
         #\n\
         # Regenerate after an intentional change (commit the diff in the same PR):\n\
         #   {REGEN_CMD}\n\
         fixture\tlevel\tfirst_pos\t{}\n",
        COUNT_COLS.join("\t")
    )
}

/// One test, serial by construction: measure the whole grid, then either
/// regenerate the TSV (UPDATE_LDX_CENSUS=1) or compare against it exactly.
#[test]
fn ldx_census_matches() {
    let mut current: BTreeMap<(String, u32), Census> = BTreeMap::new();
    for &name in fixtures::NAMES {
        let input = fixtures::generate(name);
        for &level in LEVELS {
            current.insert((name.to_string(), level), measure(&input, name, level));
        }
    }

    // ── Regeneration mode: rewrite the TSV instead of asserting. ──────────
    if std::env::var("UPDATE_LDX_CENSUS").as_deref() == Ok("1") {
        let mut tsv = tsv_header();
        for &name in fixtures::NAMES {
            for &level in LEVELS {
                let c = &current[&(name.to_string(), level)];
                let _ = writeln!(
                    tsv,
                    "{name}\t{level}\t{}\t{}",
                    c.first_pos_str(),
                    c.counts().map(|v| v.to_string()).join("\t")
                );
            }
        }
        std::fs::write(pins_path(), tsv).expect("write ldx_census.tsv");
        eprintln!(
            "ldx_census: regenerated {} ({} cells)",
            pins_path().display(),
            current.len()
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
    let mut pinned: BTreeMap<(String, u32), Census> = BTreeMap::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            3 + COUNT_COLS.len(),
            "malformed pin row: {line:?}"
        );
        let n = |i: usize| -> u64 {
            cols[i]
                .parse()
                .unwrap_or_else(|_| panic!("non-integer column {i} in pin row: {line:?}"))
        };
        pinned.insert(
            (cols[0].to_string(), cols[1].parse().unwrap()),
            Census {
                first_pos: (cols[2] != "-").then(|| n(2)),
                total_divergent: n(3),
                we_literal_they_match: n(4),
                we_match_they_literal: n(5),
                both_match_different_len: n(6),
                both_match_different_dist: n(7),
                misaligned_starts: n(8),
                block_boundary: n(9),
                ours_bytes: n(10),
                ldx_bytes: n(11),
            },
        );
    }

    let mut failures = Vec::new();
    for (cell, now) in &current {
        match pinned.get(cell) {
            None => failures.push(format!(
                "UNPINNED CELL {}:L{}:T1 — no row in the TSV; regenerate pins",
                cell.0, cell.1
            )),
            Some(pin) if pin != now => {
                failures.push(render_cell_diff(&cell.0, cell.1, pin, now));
            }
            Some(_) => {}
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
        "\n{}\n\nparse structure moved — if intended, regenerate with UPDATE_LDX_CENSUS=1 \
         and justify the movement in the commit message.\n\
         (These are snapshot pins, not aspirations: converging to libdeflate's parse is\n\
         not the goal — size <= theirs is. A moved census is not automatically bad; it\n\
         must merely be deliberate, with the TSV diff committed in the same PR.)\n    {REGEN_CMD}\n",
        failures.join("\n")
    );
}
