//! Small-file size pins: the sub-1-MiB grid the board never grades.
//!
//! The campaign board grades 1 MiB+ inputs, but 4-100 KB files are the
//! high-frequency real-world case (source files, JSON payloads, log chunks).
//! This suite pins OUR output size EXACTLY per (fixture, size, level) at
//! -p 1, on 4Ki/16Ki/64Ki/256Ki prefixes of the frozen text and binary
//! fixtures (src/fixtures.rs: a shorter `generate_sized` output is a
//! byte-exact prefix of a longer one, so these vary only length, never
//! content class).
//!
//! The pin file (tests/fingerprints/smallfile.tsv) also records gzip's and
//! pigz's sizes for the same inputs, taken from the LOCAL CLIs at regen time
//! with their versions noted in the header. Those columns are INFORMATIONAL:
//! the test asserts only OUR pins, but the report prints ours-vs-gzip deltas
//! so the small-file competitive picture is visible on every run. A losing
//! delta is a finding to flag in a PR, not a test failure.
//!
//! Wall time at small sizes is startup-dominated and needs box grading — out
//! of scope here; this suite pins SIZE only.
//!
//! Intentional output change (a lever): regenerate in the same PR —
//!     UPDATE_SMALLFILE_PINS=1 cargo test --release --test smallfile_pins

use gzippy::fixtures;
use std::collections::BTreeMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

const LEVELS: &[u32] = &[1, 6, 9];
const PIN_PATH: &str = "tests/fingerprints/smallfile.tsv";

/// (fixture, size, level) -> our exact output size at -p 1.
type Grid = BTreeMap<(String, u64, u32), u64>;

/// One pinned row: our size (asserted) plus the rivals' (informational).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    ours_bytes: u64,
    gzip_bytes: Option<u64>,
    pigz_bytes: Option<u64>,
}

fn update_mode() -> bool {
    std::env::var("UPDATE_SMALLFILE_PINS").as_deref() == Ok("1")
}

/// Compress a file through a CLI, returning stdout (the gzip stream).
fn cli_compress(bin: &str, extra: &[&str], level: u32, path: &std::path::Path) -> Option<Vec<u8>> {
    let mut args: Vec<String> = vec![format!("-{level}")];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.push("-c".into());
    args.push(path.to_str().unwrap().into());
    let o = Command::new(bin)
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !o.status.success() {
        return None;
    }
    Some(o.stdout)
}

/// Build our grid once per test binary: compress each cell at -p 1 through
/// the real binary and roundtrip through flate2 (an independent decoder)
/// against the exact input bytes — a corrupt-but-small output cannot pin.
fn grid() -> &'static Grid {
    static GRID: OnceLock<Grid> = OnceLock::new();
    GRID.get_or_init(|| {
        let bin = env!("CARGO_BIN_EXE_gzippy");
        let dir = tempfile::tempdir().unwrap();
        let mut out = Grid::new();
        for &name in fixtures::SMALL_NAMES {
            for &size in fixtures::SMALL_SIZES {
                let data = fixtures::generate_sized(name, size);
                let path = dir.path().join(format!("{name}.{size}"));
                std::fs::write(&path, &data).unwrap();
                for &level in LEVELS {
                    let gz = cli_compress(bin, &["-p", "1"], level, &path)
                        .unwrap_or_else(|| panic!("gzippy failed on {name}:{size}:L{level}"));
                    let mut decoded = Vec::new();
                    flate2::read::MultiGzDecoder::new(&gz[..])
                        .read_to_end(&mut decoded)
                        .unwrap_or_else(|e| {
                            panic!("flate2 cannot decode {name}:{size}:L{level}: {e}")
                        });
                    assert_eq!(
                        decoded, data,
                        "{name}:{size}:L{level} roundtripped to the wrong bytes — corrupt stream"
                    );
                    out.insert((name.to_string(), size as u64, level), gz.len() as u64);
                }
            }
        }
        out
    })
}

/// First line of `bin --version` (pigz prints to stderr on some builds), or
/// None if the tool is absent.
fn tool_version(bin: &str) -> Option<String> {
    let o = Command::new(bin)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let text = if o.stdout.is_empty() {
        o.stderr
    } else {
        o.stdout
    };
    Some(
        String::from_utf8_lossy(&text)
            .lines()
            .next()
            .unwrap_or("unknown")
            .trim()
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Writer / parser. The `# tools:` line is part of the canonical format so a
// regeneration on a box with different rival versions shows up in the diff.
// ---------------------------------------------------------------------------

fn opt(v: Option<u64>) -> String {
    v.map_or_else(|| "-".to_string(), |n| n.to_string())
}

fn body(rows: &BTreeMap<(String, u64, u32), Row>, tools: &str) -> String {
    let mut s = String::from(
        "# Small-file size pins: OUR exact gzip output size per (fixture, size,\n\
         # level) at -p 1 — the sub-1-MiB grid the board never grades, but the\n\
         # high-frequency real-world case. Inputs are 4Ki/16Ki/64Ki/256Ki prefixes\n\
         # of the frozen text/binary fixtures (src/fixtures.rs).\n\
         # gzip_bytes / pigz_bytes are INFORMATIONAL ONLY: recorded from the local\n\
         # CLIs (versions on the tools line below) at regen time, '-' when absent.\n\
         # The test asserts ONLY ours_bytes; it prints ours-vs-gzip deltas so the\n\
         # competitive picture stays visible. A losing delta is a finding to flag\n\
         # in a PR, not a failure.\n\
         # Wall at small sizes is startup-dominated and needs box grading — out of\n\
         # scope here; this file pins SIZE only.\n\
         # Regenerate deliberately, in the same PR as the lever that moved it:\n\
         #   UPDATE_SMALLFILE_PINS=1 cargo test --release --test smallfile_pins\n",
    );
    s.push_str(&format!("# tools: {tools}\n"));
    s.push_str("fixture\tsize\tlevel\tours_bytes\tgzip_bytes\tpigz_bytes\n");
    for ((f, size, l), r) in rows {
        s.push_str(&format!(
            "{f}\t{size}\t{l}\t{}\t{}\t{}\n",
            r.ours_bytes,
            opt(r.gzip_bytes),
            opt(r.pigz_bytes)
        ));
    }
    s
}

fn parse(text: &str) -> (BTreeMap<(String, u64, u32), Row>, String) {
    let mut rows = BTreeMap::new();
    let mut tools = String::new();
    for line in text.lines() {
        if let Some(t) = line.strip_prefix("# tools: ") {
            tools = t.to_string();
            continue;
        }
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 {
            continue;
        }
        let (Ok(size), Ok(level), Ok(ours)) =
            (cols[1].parse(), cols[2].parse(), cols[3].parse::<u64>())
        else {
            continue;
        };
        rows.insert(
            (cols[0].to_string(), size, level),
            Row {
                ours_bytes: ours,
                gzip_bytes: cols[4].parse().ok(),
                pigz_bytes: cols[5].parse().ok(),
            },
        );
    }
    (rows, tools)
}

// ---------------------------------------------------------------------------
// The tests.
// ---------------------------------------------------------------------------

/// The pin: our size per cell must equal the pin file exactly. In update mode
/// it regenerates the file, re-measuring gzip/pigz from the local CLIs.
/// Either way it prints the ours-vs-gzip delta table.
#[test]
fn smallfile_sizes_match_pins() {
    let current = grid();
    if update_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mut rows = BTreeMap::new();
        for ((name, size, level), &ours) in current {
            let path = dir.path().join(format!("{name}.{size}"));
            if !path.exists() {
                std::fs::write(&path, fixtures::generate_sized(name, *size as usize)).unwrap();
            }
            rows.insert(
                (name.clone(), *size, *level),
                Row {
                    ours_bytes: ours,
                    gzip_bytes: cli_compress("gzip", &[], *level, &path).map(|v| v.len() as u64),
                    pigz_bytes: cli_compress("pigz", &["-p", "1"], *level, &path)
                        .map(|v| v.len() as u64),
                },
            );
        }
        let tools = format!(
            "gzippy={} gzip={} pigz={}",
            env!("CARGO_PKG_VERSION"),
            tool_version("gzip").as_deref().unwrap_or("-"),
            tool_version("pigz").as_deref().unwrap_or("-"),
        );
        std::fs::write(PIN_PATH, body(&rows, &tools)).unwrap();
        eprintln!("wrote {PIN_PATH} ({} cells)", rows.len());
        report(&rows);
        return;
    }
    let text = std::fs::read_to_string(PIN_PATH).unwrap_or_else(|_| {
        panic!(
            "{PIN_PATH} missing — regenerate: \
             UPDATE_SMALLFILE_PINS=1 cargo test --release --test smallfile_pins"
        )
    });
    let (pinned, _tools) = parse(&text);
    let mut diffs = Vec::new();
    for ((name, size, level), &ours) in current {
        match pinned.get(&(name.clone(), *size, *level)) {
            None => diffs.push(format!(
                "UNPINNED CELL {name}:{size}:L{level} — regenerate the small-file pins"
            )),
            Some(pin) if pin.ours_bytes != ours => diffs.push(format!(
                "SMALL-FILE SIZE MOVED {name}:{size}:L{level}: pinned {} B -> current {} B ({:+} B)",
                pin.ours_bytes,
                ours,
                ours as i64 - pin.ours_bytes as i64
            )),
            _ => {}
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            diffs.push(format!(
                "PINNED CELL {}:{}:L{} not produced by the current grid",
                cell.0, cell.1, cell.2
            ));
        }
    }
    report(&pinned);
    assert!(
        diffs.is_empty(),
        "\n{}\n\nSmall-file output size changed. If INTENTIONAL (a lever), regenerate the\npins in this PR:\n    UPDATE_SMALLFILE_PINS=1 cargo test --release --test smallfile_pins\nOtherwise the rows above name exactly which (fixture, size, level) moved.\n",
        diffs.join("\n")
    );
}

/// The informational report: ours vs gzip (and pigz where recorded), per
/// cell. Printed on every run so the small-file competitive picture never
/// goes dark. A LOSING row is a finding, not a failure.
fn report(rows: &BTreeMap<(String, u64, u32), Row>) {
    eprintln!("\nsmall-file sizes, ours vs gzip (informational; sizes in bytes):");
    eprintln!(
        "{:<8} {:>8} {:>5} {:>10} {:>10} {:>8} {:>8} {:>10}  verdict",
        "fixture", "size", "level", "ours", "gzip", "delta", "delta%", "pigz"
    );
    for ((name, size, level), r) in rows {
        let (gzip, delta, pct, verdict) = match r.gzip_bytes {
            Some(g) => {
                let d = r.ours_bytes as i64 - g as i64;
                (
                    g.to_string(),
                    format!("{d:+}"),
                    format!("{:+.2}%", d as f64 * 100.0 / g as f64),
                    if d > 0 { "LOSING" } else { "smaller-or-tie" },
                )
            }
            None => ("-".into(), "-".into(), "-".into(), "no-gzip-recorded"),
        };
        eprintln!(
            "{name:<8} {size:>8} {level:>5} {:>10} {gzip:>10} {delta:>8} {pct:>8} {:>10}  {verdict}",
            r.ours_bytes,
            opt(r.pigz_bytes),
        );
    }
}

/// The pin file on disk is exactly what the writer would emit from its own
/// parsed content — no hand edits, no format drift.
#[test]
fn smallfile_pin_file_is_canonical() {
    if update_mode() {
        // The other test is rewriting the file right now; comparing canonical
        // form against a moving target is a race.
        return;
    }
    let text = std::fs::read_to_string(PIN_PATH).unwrap_or_else(|_| {
        panic!(
            "{PIN_PATH} missing — regenerate: \
             UPDATE_SMALLFILE_PINS=1 cargo test --release --test smallfile_pins"
        )
    });
    let (rows, tools) = parse(&text);
    assert!(!rows.is_empty(), "{PIN_PATH} parsed to zero rows");
    assert_eq!(
        body(&rows, &tools),
        text,
        "{PIN_PATH} is not in canonical writer format (hand-edited?) — regenerate it"
    );
}
