//! T4 pins: the whole-stream suite (fingerprint_suite.rs) pins each cell's
//! fingerprint; this suite gives the PARALLEL path its own pin surface and
//! promotes the campaign's largest failing class — the T>1 SEAM TAX
//! (size_t4 - size_t1) — to a first-class pinned quantity, so a seam
//! regression on the laptop fails a test by name instead of waiting for a
//! board run on the boxes.
//!
//! Grid: every synthetic fixture (src/fixtures.rs) x levels {1,2,6,9} at
//! -p 4 vs -p 1, through the REAL binary (CARGO_BIN_EXE_gzippy) via
//! FILE-based invocation (never stdin — the stdin trap silently routes T1 at
//! every -p; the tell is T1==T4 byte-identity where seams should differ).
//!
//! The pin surface only exists if T>1 output is deterministic, so the grid
//! builder runs every T4 cell TWICE and refuses to compare anything if the
//! two runs differ — nondeterminism is a finding, not a flake.
//!
//! tax_bytes MAY BE NEGATIVE: the body term is negative on some inputs; only
//! per-chunk framing is always positive. A SHRINKING tax is an improvement to
//! bank (update the pin deliberately), not a failure to silence.
//!
//! Intentional change (a lever): regenerate BOTH files in the same PR —
//!     UPDATE_T4_PINS=1 cargo test --release --test t4_pins
//!     (or: cargo run --release --example fingerprint_tool -- pin-t4)
//! and remember ours.tsv also carries T4 rows (pin-ours).

use gzippy::decompress::block_walker::{fingerprint_gzip, StreamFingerprint};
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::OnceLock;

const LEVELS: &[u32] = &[1, 2, 6, 9];
const T4_PIN_PATH: &str = "tests/fingerprints/ours_t4.tsv";
const SEAM_PIN_PATH: &str = "tests/fingerprints/seam_tax.tsv";

/// (T1 fingerprint, T4 fingerprint) per (fixture, level).
type Grid = BTreeMap<(String, u32), (StreamFingerprint, StreamFingerprint)>;

fn update_mode() -> bool {
    std::env::var("UPDATE_T4_PINS").as_deref() == Ok("1")
}

/// Compress a fixture FILE (never stdin) through the real binary.
fn compress(bin: &str, path: &std::path::Path, level: u32, threads: u32) -> Vec<u8> {
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
        "gzippy failed on {} L{level} T{threads}",
        path.display()
    );
    o.stdout
}

/// Build the grid once per test binary: T1 once, T4 twice (the determinism
/// gate), fingerprints checked against the input size (correctness rides
/// along, as in every suite here).
fn grid() -> &'static Grid {
    static GRID: OnceLock<Grid> = OnceLock::new();
    GRID.get_or_init(|| {
        let bin = env!("CARGO_BIN_EXE_gzippy");
        let dir = tempfile::tempdir().unwrap();
        let mut out = Grid::new();
        let mut t4_differs = 0usize;
        for &name in fixtures::NAMES {
            let data = fixtures::generate(name);
            let path = dir.path().join(name);
            std::fs::write(&path, &data).unwrap();
            for &level in LEVELS {
                let t1 = compress(bin, &path, level, 1);
                let t4a = compress(bin, &path, level, 4);
                let t4b = compress(bin, &path, level, 4);
                if t4a != t4b {
                    let first = t4a
                        .iter()
                        .zip(t4b.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or_else(|| t4a.len().min(t4b.len()));
                    panic!(
                        "T4 OUTPUT IS NONDETERMINISTIC on {name}:L{level}: two identical \
                         -p 4 runs on the same file produced different bytes \
                         ({} vs {} bytes, first divergence at offset {first}). \
                         A nondeterministic surface cannot be pinned — find the race \
                         before touching any pin file.",
                        t4a.len(),
                        t4b.len()
                    );
                }
                if t4a != t1 {
                    t4_differs += 1;
                }
                let fp1 = fingerprint_gzip(&t1)
                    .unwrap_or_else(|e| panic!("unparseable stream {name} L{level} T1: {e}"));
                let fp4 = fingerprint_gzip(&t4a)
                    .unwrap_or_else(|e| panic!("unparseable stream {name} L{level} T4: {e}"));
                for (t, fp) in [(1, &fp1), (4, &fp4)] {
                    assert_eq!(
                        fp.decoded_bytes,
                        data.len() as u64,
                        "stream for {name} L{level} T{t} decodes to the wrong size — corrupt?"
                    );
                }
                out.insert((name.to_string(), level), (fp1, fp4));
            }
        }
        assert!(
            t4_differs > 0,
            "T4 output was byte-identical to T1 on EVERY cell — either the parallel \
             path did not engage (the stdin trap: this suite must invoke on a FILE) \
             or seams have been eliminated entirely (the bit-splice goal). Know which \
             before trusting these pins."
        );
        out
    })
}

// ---------------------------------------------------------------------------
// Writers. MUST stay byte-identical to `t4_pins_body` / `SeamRow` /
// `seam_row` / `seam_tax_body` in examples/fingerprint_tool.rs — the test's
// UPDATE_T4_PINS=1 mode and the tool's pin-t4 write the same files.
// ---------------------------------------------------------------------------

fn t4_pins_body(rows: &BTreeMap<(String, u32), StreamFingerprint>) -> String {
    let mut s = String::from(
        "# OUR T4 (-p 4) whole-stream fingerprints on the frozen fixtures, levels\n\
         # {1,2,6,9}. T>1 output legally differs from T1 (byte-identity is a cage,\n\
         # not a goal) but it IS deterministic — verified on every regeneration —\n\
         # so these rows pin the PARALLEL path the way ours.tsv pins each cell.\n\
         # Regenerate ONLY when a lever intentionally changes output, in the same\n\
         # PR as ours.tsv (which also carries T4 rows):\n\
         #   UPDATE_T4_PINS=1 cargo test --release --test t4_pins\n\
         #   (or: cargo run --release --example fingerprint_tool -- pin-t4)\n",
    );
    s.push_str(&format!(
        "fixture\tlevel\tthreads\timpl\t{}\n",
        StreamFingerprint::TSV_FIELDS.join("\t")
    ));
    for ((f, l), fp) in rows {
        let vals: Vec<String> = fp.tsv_values().iter().map(|v| v.to_string()).collect();
        s.push_str(&format!("{f}\t{l}\t4\tgzippy\t{}\n", vals.join("\t")));
    }
    s
}

/// One seam-tax row: the derived T>1 size cost of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeamRow {
    size_t1: u64,
    size_t4: u64,
    /// size_t4 - size_t1. MAY BE NEGATIVE: the body term is negative on some
    /// inputs; only the per-chunk framing term is always positive.
    tax_bytes: i64,
    members_t1: u32,
    members_t4: u32,
    blocks_t1: u32,
    blocks_t4: u32,
}

impl SeamRow {
    const FIELDS: &'static [&'static str] = &[
        "size_t1",
        "size_t4",
        "tax_bytes",
        "members_t1",
        "members_t4",
        "blocks_t1",
        "blocks_t4",
    ];

    fn values(&self) -> [i64; 7] {
        [
            self.size_t1 as i64,
            self.size_t4 as i64,
            self.tax_bytes,
            self.members_t1 as i64,
            self.members_t4 as i64,
            self.blocks_t1 as i64,
            self.blocks_t4 as i64,
        ]
    }
}

fn seam_row(t1: &StreamFingerprint, t4: &StreamFingerprint) -> SeamRow {
    let blocks = |fp: &StreamFingerprint| fp.blocks_stored + fp.blocks_fixed + fp.blocks_dynamic;
    SeamRow {
        size_t1: t1.file_bytes,
        size_t4: t4.file_bytes,
        tax_bytes: t4.file_bytes as i64 - t1.file_bytes as i64,
        members_t1: t1.members,
        members_t4: t4.members,
        blocks_t1: blocks(t1),
        blocks_t4: blocks(t4),
    }
}

fn seam_tax_body(rows: &BTreeMap<(String, u32), SeamRow>) -> String {
    let mut s = String::from(
        "# THE SEAM TAX, pinned: size_t4 - size_t1 per fixture x level {1,2,6,9} —\n\
         # the campaign's T>1 seam class as an exact laptop number. tax_bytes MAY BE\n\
         # NEGATIVE (the body term is negative on some inputs; only per-chunk framing\n\
         # is always positive). A SHRINKING tax is an improvement to bank, not a\n\
         # failure to silence. Regenerate ONLY when a lever intentionally changes\n\
         # output:\n\
         #   UPDATE_T4_PINS=1 cargo test --release --test t4_pins\n\
         #   (or: cargo run --release --example fingerprint_tool -- pin-t4)\n\
         fixture\tlevel\tsize_t1\tsize_t4\ttax_bytes\tmembers_t1\tmembers_t4\tblocks_t1\tblocks_t4\n",
    );
    for ((f, l), r) in rows {
        s.push_str(&format!(
            "{f}\t{l}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.size_t1, r.size_t4, r.tax_bytes, r.members_t1, r.members_t4, r.blocks_t1, r.blocks_t4
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// Parsers.
// ---------------------------------------------------------------------------

fn data_lines(body: &str) -> impl Iterator<Item = Vec<&str>> {
    body.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("fixture\t") && !l.trim().is_empty())
        .map(|l| l.split('\t').collect())
}

fn parse_t4_pins(body: &str) -> BTreeMap<(String, u32), StreamFingerprint> {
    let mut out = BTreeMap::new();
    for cols in data_lines(body) {
        if cols.len() != 4 + StreamFingerprint::TSV_FIELDS.len() || cols[2] != "4" {
            continue;
        }
        let vals: Vec<u64> = cols[4..].iter().filter_map(|c| c.parse().ok()).collect();
        let (Ok(level), Some(fp)) = (cols[1].parse(), StreamFingerprint::from_tsv_values(&vals))
        else {
            continue;
        };
        out.insert((cols[0].to_string(), level), fp);
    }
    out
}

fn parse_seam_pins(body: &str) -> BTreeMap<(String, u32), SeamRow> {
    let mut out = BTreeMap::new();
    for cols in data_lines(body) {
        if cols.len() != 9 {
            continue;
        }
        let nums: Vec<i64> = cols[2..].iter().filter_map(|c| c.parse().ok()).collect();
        let (Ok(level), [t1, t4, tax, m1, m4, b1, b4]) = (cols[1].parse(), nums.as_slice()) else {
            continue;
        };
        out.insert(
            (cols[0].to_string(), level),
            SeamRow {
                size_t1: *t1 as u64,
                size_t4: *t4 as u64,
                tax_bytes: *tax,
                members_t1: *m1 as u32,
                members_t4: *m4 as u32,
                blocks_t1: *b1 as u32,
                blocks_t4: *b4 as u32,
            },
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Renderers.
// ---------------------------------------------------------------------------

fn render_fp_diff(current: &StreamFingerprint, pinned: &StreamFingerprint) -> String {
    current
        .diff(pinned)
        .iter()
        .map(|(axis, cur, pin)| {
            let rel = if *pin == 0 {
                "new".to_string()
            } else {
                format!("{:+.1}%", (*cur as f64 - *pin as f64) * 100.0 / *pin as f64)
            };
            format!("    {axis:<14} pinned {pin:>12}  current {cur:>12}  ({rel})")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_seam_diff(fixture: &str, level: u32, pinned: &SeamRow, current: &SeamRow) -> String {
    let headline = match current.tax_bytes.cmp(&pinned.tax_bytes) {
        std::cmp::Ordering::Greater => format!(
            "seam tax GREW by {} bytes: pinned {:+} B -> current {:+} B",
            current.tax_bytes - pinned.tax_bytes,
            pinned.tax_bytes,
            current.tax_bytes
        ),
        std::cmp::Ordering::Less => format!(
            "seam tax SHRANK by {} bytes — an improvement; update the pin deliberately: \
             pinned {:+} B -> current {:+} B",
            pinned.tax_bytes - current.tax_bytes,
            pinned.tax_bytes,
            current.tax_bytes
        ),
        std::cmp::Ordering::Equal => format!(
            "seam tax unchanged at {:+} B, but T1 and T4 moved together",
            current.tax_bytes
        ),
    };
    let mut msg = format!("SEAM ROW MOVED {fixture}:L{level}: {headline}\n");
    for ((axis, pin), cur) in SeamRow::FIELDS
        .iter()
        .zip(pinned.values())
        .zip(current.values())
    {
        if pin != cur {
            msg.push_str(&format!(
                "    {axis:<12} pinned {pin:>12}  current {cur:>12}\n"
            ));
        }
    }
    msg
}

// ---------------------------------------------------------------------------
// The tests.
// ---------------------------------------------------------------------------

/// The T4 snapshot: our parallel-path fingerprints must equal the pin file.
/// Any T4 output change fails here naming the fixture:level and each moved
/// axis, pinned -> current.
#[test]
fn t4_fingerprints_match_pins() {
    let current: BTreeMap<(String, u32), StreamFingerprint> = grid()
        .iter()
        .map(|(cell, (_t1, t4))| (cell.clone(), t4.clone()))
        .collect();
    if update_mode() {
        std::fs::write(T4_PIN_PATH, t4_pins_body(&current)).unwrap();
        eprintln!("wrote {T4_PIN_PATH} ({} cells)", current.len());
        return;
    }
    let body = std::fs::read_to_string(T4_PIN_PATH).unwrap_or_else(|_| {
        panic!("{T4_PIN_PATH} missing — regenerate: UPDATE_T4_PINS=1 cargo test --release --test t4_pins")
    });
    let pinned = parse_t4_pins(&body);
    let mut diffs = Vec::new();
    for (cell, cur) in &current {
        match pinned.get(cell) {
            None => diffs.push(format!(
                "UNPINNED CELL {}:L{}:T4 — regenerate the T4 pins",
                cell.0, cell.1
            )),
            Some(pin) if pin != cur => diffs.push(format!(
                "T4 FINGERPRINT MOVED {}:L{}:T4\n{}",
                cell.0,
                cell.1,
                render_fp_diff(cur, pin)
            )),
            _ => {}
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            diffs.push(format!(
                "PINNED CELL {}:L{}:T4 not produced by the current grid",
                cell.0, cell.1
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "\n{}\n\nT4 output changed. If INTENTIONAL (a lever), regenerate the pins in this\nPR — the diff is your mechanism, stated for review:\n    UPDATE_T4_PINS=1 cargo test --release --test t4_pins\n(and ours.tsv carries T4 rows too: pin-ours). If you did not mean to change\nT>1 output, the axes above name what moved.\n",
        diffs.join("\n\n")
    );
}

/// The seam tax itself: size_t4 - size_t1 per cell, exact. A regression says
/// GREW; an improvement says SHRANK and asks for a deliberate pin update —
/// the seam class's board movement, visible on the laptop per commit.
#[test]
fn seam_tax_matches_pins() {
    let current: BTreeMap<(String, u32), SeamRow> = grid()
        .iter()
        .map(|(cell, (t1, t4))| (cell.clone(), seam_row(t1, t4)))
        .collect();
    if update_mode() {
        std::fs::write(SEAM_PIN_PATH, seam_tax_body(&current)).unwrap();
        let total: i64 = current.values().map(|r| r.tax_bytes).sum();
        eprintln!(
            "wrote {SEAM_PIN_PATH} ({} cells, total tax {total:+} B)",
            current.len()
        );
        return;
    }
    let body = std::fs::read_to_string(SEAM_PIN_PATH).unwrap_or_else(|_| {
        panic!("{SEAM_PIN_PATH} missing — regenerate: UPDATE_T4_PINS=1 cargo test --release --test t4_pins")
    });
    let pinned = parse_seam_pins(&body);
    let mut diffs = Vec::new();
    for (cell, cur) in &current {
        match pinned.get(cell) {
            None => diffs.push(format!(
                "UNPINNED CELL {}:L{} — regenerate the seam-tax pins",
                cell.0, cell.1
            )),
            Some(pin) if pin != cur => {
                diffs.push(render_seam_diff(&cell.0, cell.1, pin, cur));
            }
            _ => {}
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            diffs.push(format!(
                "PINNED CELL {}:L{} not produced by the current grid",
                cell.0, cell.1
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "\n{}\n\nThe seam tax moved. GREW = a T>1 size regression: fix or revert. SHRANK =\nan improvement: bank it by regenerating the pin in the same PR:\n    UPDATE_T4_PINS=1 cargo test --release --test t4_pins\n",
        diffs.join("\n")
    );
}

/// Both pin files on disk are exactly what the writers would emit today — no
/// hand edits, no drift between this test's writers and the tool's pin-t4.
#[test]
fn t4_pin_files_are_canonical() {
    if update_mode() {
        // The other two tests are rewriting the files right now (tests run in
        // parallel); checking canonical form against a moving target is a race.
        return;
    }
    let t4 = std::fs::read_to_string(T4_PIN_PATH).unwrap_or_else(|_| {
        panic!("{T4_PIN_PATH} missing — regenerate: UPDATE_T4_PINS=1 cargo test --release --test t4_pins")
    });
    let reparsed = parse_t4_pins(&t4);
    assert!(!reparsed.is_empty(), "{T4_PIN_PATH} parsed to zero rows");
    assert_eq!(
        t4_pins_body(&reparsed),
        t4,
        "{T4_PIN_PATH} is not in canonical writer format (hand-edited?) — regenerate it"
    );

    let seam = std::fs::read_to_string(SEAM_PIN_PATH).unwrap_or_else(|_| {
        panic!("{SEAM_PIN_PATH} missing — regenerate: UPDATE_T4_PINS=1 cargo test --release --test t4_pins")
    });
    let reparsed = parse_seam_pins(&seam);
    assert!(!reparsed.is_empty(), "{SEAM_PIN_PATH} parsed to zero rows");
    assert_eq!(
        seam_tax_body(&reparsed),
        seam,
        "{SEAM_PIN_PATH} is not in canonical writer format (hand-edited?) — regenerate it"
    );

    // The derived column is derived: tax_bytes == size_t4 - size_t1 on every
    // row, and the tax the pins carry is consistent with the T4 pin file.
    let t4_rows = parse_t4_pins(&t4);
    for (cell, r) in &reparsed {
        assert_eq!(
            r.tax_bytes,
            r.size_t4 as i64 - r.size_t1 as i64,
            "{SEAM_PIN_PATH} {}:L{}: tax_bytes is not size_t4 - size_t1",
            cell.0,
            cell.1
        );
        if let Some(fp) = t4_rows.get(cell) {
            assert_eq!(
                r.size_t4, fp.file_bytes,
                "{SEAM_PIN_PATH} {}:L{}: size_t4 disagrees with {T4_PIN_PATH}",
                cell.0, cell.1
            );
        }
    }
}
