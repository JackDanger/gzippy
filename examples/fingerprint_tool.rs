//! Fingerprint pinning + gap reporting for the mechanism-fingerprint suite.
//!
//!   cargo run --release --example fingerprint_tool -- pin-ours
//!       Recompute OUR fingerprints on every fixture cell and rewrite
//!       tests/fingerprints/ours.tsv. Run this when a lever INTENTIONALLY
//!       changes output; the file's diff in the PR is the lever's mechanism,
//!       reviewable line by line.
//!
//!   cargo run --release --example fingerprint_tool -- pin-rivals
//!       Run every rival CLI present on this box over the fixtures and
//!       rewrite tests/fingerprints/rivals.tsv (bytes + full fingerprint per
//!       cell, provenance header). Run on a box with all four rivals.
//!
//!   cargo run --release --example fingerprint_tool -- ledger
//!       Rewrite tests/fingerprints/ledger.tsv: every (fixture, level,
//!       threads, rival) cell where OUR bytes <= the rival's. The suite
//!       asserts this set never shrinks — cells, once won, stay won.
//!
//!   cargo run --release --example fingerprint_tool -- report
//!       Per-axis fingerprint diff for every cell we LOSE, worst first, plus
//!       a class aggregation across cells ("literals moved on 5 cells").
//!
//!   cargo run --release --example fingerprint_tool -- blocks <file.gz>
//!       Print the per-block fingerprint table for one gzip file: index,
//!       member, btype, final, header/data bits, token counts, span.
//!
//!   cargo run --release --example fingerprint_tool -- pin-blocks
//!       Recompute OUR per-block rows on every fixture x level {1,2,6,9} at
//!       T1 and rewrite tests/fingerprints/ours_blocks.tsv. Same discipline
//!       as pin-ours: regenerate ONLY when a lever intentionally changes
//!       output. (Equivalent: UPDATE_BLOCK_PINS=1 cargo test --release
//!       --test block_pins.)
//!
//!   cargo run --release --example fingerprint_tool -- pin-t4
//!       Recompute OUR T4 (-p 4) fingerprints on every fixture x level
//!       {1,2,6,9} and rewrite tests/fingerprints/ours_t4.tsv AND
//!       tests/fingerprints/seam_tax.tsv (the derived size_t4 - size_t1 per
//!       cell — the campaign's T>1 seam class as an exact laptop number).
//!       Same discipline as pin-ours; a lever that touches T>1 output
//!       regenerates ours.tsv, ours_t4.tsv and seam_tax.tsv in the same PR.
//!       (Equivalent: UPDATE_T4_PINS=1 cargo test --release --test t4_pins.)
//!
//! Ours is invoked through the REAL binary (GZIPPY_BIN, default
//! target/release/gzippy) — the shipped quantity, not a library shortcut.

use gzippy::decompress::block_walker::{
    fingerprint_gzip, fingerprint_gzip_blocks, BlockFingerprint, StreamFingerprint,
};
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::process::{Command, Stdio};

const LEVELS: &[u32] = &[1, 2, 6, 9];
const THREADS: &[u32] = &[1, 4];
const PIN_DIR: &str = "tests/fingerprints";

fn gzippy_bin() -> String {
    std::env::var("GZIPPY_BIN").unwrap_or_else(|_| "target/release/gzippy".into())
}

/// Compress a FIXTURE FILE through an external CLI, returning the gzip bytes.
///
/// A real file, not stdin: gzippy's T>1 mmap path (and pigz's parallelism)
/// route on a seekable input, and piping stdin silently measured the T1 path
/// at every thread count — caught on this tool's first run when T1 and T4
/// bytes came back identical.
fn run_compressor(argv: &[String], input: &std::path::Path) -> Option<Vec<u8>> {
    let mut full = argv.to_vec();
    full.push(input.to_string_lossy().into_owned());
    let out = Command::new(&full[0])
        .args(&full[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.len() < 18 {
        return None;
    }
    Some(out.stdout)
}

/// Write every fixture to a stable temp path once; return (name -> path).
fn staged_fixtures() -> Vec<(&'static str, std::path::PathBuf, Vec<u8>)> {
    let dir = std::env::temp_dir().join(format!("gzippy-fingerprint-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    fixtures::NAMES
        .iter()
        .map(|&name| {
            let data = fixtures::generate(name);
            let p = dir.join(name);
            std::fs::write(&p, &data).unwrap();
            (name, p, data)
        })
        .collect()
}

/// (impl-name, argv-template) for every rival. `{L}` = level, `{T}` = threads.
/// A rival missing from PATH is skipped with a warning — the pin file records
/// which rivals it covers.
fn rival_commands() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("gzip", vec!["gzip", "-{L}", "-c"]),
        ("pigz", vec!["pigz", "-{L}", "-p", "{T}", "-c"]),
        ("libdeflate", vec!["libdeflate-gzip", "-{L}", "-c"]),
        ("igzip", vec!["igzip", "-{L}", "-T", "{T}", "-c"]),
    ]
}

fn instantiate(tpl: &[&str], level: u32, threads: u32) -> Vec<String> {
    tpl.iter()
        .map(|a| {
            a.replace("{L}", &level.to_string())
                .replace("{T}", &threads.to_string())
        })
        .collect()
}

/// igzip only has levels 0-3; every other rival covers 1-9.
fn rival_supports(name: &str, level: u32) -> bool {
    name != "igzip" || level <= 3
}

/// Thread-invariant rivals (single-threaded CLIs) are measured once at T1 and
/// the cell is recorded per thread-count anyway: the BOARD's contract is
/// per-(file, level, T) against the rival's best, which for them is the same
/// bytes at every T.
fn tsv_header() -> String {
    format!(
        "fixture\tlevel\tthreads\timpl\t{}\n",
        StreamFingerprint::TSV_FIELDS.join("\t")
    )
}

fn tsv_row(fixture: &str, level: u32, threads: u32, imp: &str, fp: &StreamFingerprint) -> String {
    let vals: Vec<String> = fp.tsv_values().iter().map(|v| v.to_string()).collect();
    format!(
        "{fixture}\t{level}\t{threads}\t{imp}\t{}\n",
        vals.join("\t")
    )
}

type Cell = (String, u32, u32, String); // fixture, level, threads, impl

fn parse_pins(path: &str) -> BTreeMap<Cell, StreamFingerprint> {
    let mut out = BTreeMap::new();
    let Ok(body) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in body.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 4 + StreamFingerprint::TSV_FIELDS.len() {
            continue;
        }
        let vals: Vec<u64> = cols[4..].iter().filter_map(|c| c.parse().ok()).collect();
        if let Some(fp) = StreamFingerprint::from_tsv_values(&vals) {
            out.insert(
                (
                    cols[0].to_string(),
                    cols[1].parse().unwrap_or(0),
                    cols[2].parse().unwrap_or(0),
                    cols[3].to_string(),
                ),
                fp,
            );
        }
    }
    out
}

fn ours_fingerprints() -> BTreeMap<Cell, StreamFingerprint> {
    let bin = gzippy_bin();
    let mut out = BTreeMap::new();
    for (name, path, data) in staged_fixtures() {
        for &level in LEVELS {
            for &threads in THREADS {
                let argv: Vec<String> = vec![
                    bin.clone(),
                    format!("-{level}"),
                    "-p".into(),
                    threads.to_string(),
                    "-c".into(),
                ];
                let gz = run_compressor(&argv, &path)
                    .unwrap_or_else(|| panic!("{bin} failed on {name} L{level} T{threads}"));
                let fp = fingerprint_gzip(&gz)
                    .unwrap_or_else(|e| panic!("fingerprint {name} L{level} T{threads}: {e}"));
                assert_eq!(
                    fp.decoded_bytes,
                    data.len() as u64,
                    "decoded size mismatch on {name} L{level} T{threads} — corrupt stream?"
                );
                out.insert((name.into(), level, threads, "gzippy".into()), fp);
            }
        }
    }
    out
}

fn write_pins(path: &str, provenance: &str, rows: &BTreeMap<Cell, StreamFingerprint>) {
    std::fs::create_dir_all(PIN_DIR).unwrap();
    let mut s = String::new();
    s.push_str(provenance);
    s.push_str(&tsv_header());
    for ((f, l, t, i), fp) in rows {
        s.push_str(&tsv_row(f, *l, *t, i, fp));
    }
    std::fs::write(path, s).unwrap();
    println!("wrote {path} ({} cells)", rows.len());
}

/// Per-block rows for every fixture x level at T1, through the real binary.
/// Keyed (fixture, level); T is pinned to 1 — the per-block table is a T1
/// mechanism instrument (T>1 may legally emit different bytes per run config,
/// so its block grid is not a stable pin surface).
fn ours_block_rows() -> BTreeMap<(String, u32), Vec<BlockFingerprint>> {
    let bin = gzippy_bin();
    let mut out = BTreeMap::new();
    for (name, path, data) in staged_fixtures() {
        for &level in LEVELS {
            let argv: Vec<String> = vec![
                bin.clone(),
                format!("-{level}"),
                "-p".into(),
                "1".into(),
                "-c".into(),
            ];
            let gz = run_compressor(&argv, &path)
                .unwrap_or_else(|| panic!("{bin} failed on {name} L{level} T1"));
            let (fp, rows) = fingerprint_gzip_blocks(&gz)
                .unwrap_or_else(|e| panic!("fingerprint {name} L{level} T1: {e}"));
            assert_eq!(
                fp.decoded_bytes,
                data.len() as u64,
                "decoded size mismatch on {name} L{level} T1 — corrupt stream?"
            );
            out.insert((name.to_string(), level), rows);
        }
    }
    out
}

/// The exact bytes of tests/fingerprints/ours_blocks.tsv. MUST stay
/// byte-identical to `pins_body` in tests/block_pins.rs — the test's
/// UPDATE_BLOCK_PINS=1 mode and this tool's pin-blocks write the same file.
fn block_pins_body(rows: &BTreeMap<(String, u32), Vec<BlockFingerprint>>) -> String {
    let mut s = String::from(
        "# OUR per-block fingerprints on the frozen fixtures, levels {1,2,6,9}, T1.\n\
         # One row per DEFLATE block; any size change names the exact block that\n\
         # moved. Regenerate ONLY when a lever intentionally changes output:\n\
         #   UPDATE_BLOCK_PINS=1 cargo test --release --test block_pins\n\
         #   (or: cargo run --release --example fingerprint_tool -- pin-blocks)\n",
    );
    s.push_str(&format!(
        "fixture\tlevel\t{}\n",
        BlockFingerprint::TSV_FIELDS.join("\t")
    ));
    for ((f, l), blocks) in rows {
        for b in blocks {
            let vals: Vec<String> = b.tsv_values().iter().map(|v| v.to_string()).collect();
            s.push_str(&format!("{f}\t{l}\t{}\n", vals.join("\t")));
        }
    }
    s
}

/// The exact bytes of tests/fingerprints/ours_t4.tsv. MUST stay
/// byte-identical to `t4_pins_body` in tests/t4_pins.rs — the test's
/// UPDATE_T4_PINS=1 mode and this tool's pin-t4 write the same file.
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
    s.push_str(&tsv_header());
    for ((f, l), fp) in rows {
        s.push_str(&tsv_row(f, *l, 4, "gzippy", fp));
    }
    s
}

/// One seam-tax row: the derived T>1 size cost of a cell. MUST stay identical
/// to `SeamRow` in tests/t4_pins.rs.
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

/// The exact bytes of tests/fingerprints/seam_tax.tsv. MUST stay
/// byte-identical to `seam_tax_body` in tests/t4_pins.rs.
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

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "pin-ours" => {
            let ours = ours_fingerprints();
            for &fixture in fixtures::NAMES {
                for &level in LEVELS {
                    let t1 = ours.get(&(fixture.to_string(), level, 1, "gzippy".into()));
                    let t4 = ours.get(&(fixture.to_string(), level, 4, "gzippy".into()));
                    if let (Some(a), Some(b)) = (t1, t4) {
                        if a == b && level > 0 {
                            eprintln!(
                                "note: {fixture} L{level} T1==T4 byte-identical — either the T>1 \
                                 route did not engage (the stdin trap: verify with a file input) \
                                 or seams have been eliminated (the bit-splice goal). Know which."
                            );
                        }
                    }
                }
            }
            write_pins(
                &format!("{PIN_DIR}/ours.tsv"),
                "# OUR fingerprints on the frozen fixtures. Regenerate ONLY when a lever\n\
                 # intentionally changes output: cargo run --release --example fingerprint_tool -- pin-ours\n\
                 # The diff of this file in a PR is the lever's mechanism.\n",
                &ours,
            );
        }
        "pin-rivals" => {
            let mut rows = BTreeMap::new();
            let mut covered = Vec::new();
            for (name, tpl) in rival_commands() {
                let mut any = false;
                for (fixture, path, data) in staged_fixtures() {
                    for &level in LEVELS {
                        if !rival_supports(name, level) {
                            continue;
                        }
                        for &threads in THREADS {
                            let argv = instantiate(&tpl, level, threads);
                            let Some(gz) = run_compressor(&argv, &path) else {
                                continue;
                            };
                            let Ok(fp) = fingerprint_gzip(&gz) else {
                                eprintln!("warn: cannot fingerprint {name} on {fixture} L{level} T{threads} (unusual stream shape) — pinning bytes only");
                                let mut fp = StreamFingerprint {
                                    file_bytes: gz.len() as u64,
                                    ..Default::default()
                                };
                                fp.decoded_bytes = data.len() as u64;
                                rows.insert(
                                    (fixture.to_string(), level, threads, name.to_string()),
                                    fp,
                                );
                                any = true;
                                continue;
                            };
                            rows.insert(
                                (fixture.to_string(), level, threads, name.to_string()),
                                fp,
                            );
                            any = true;
                        }
                    }
                }
                if any {
                    covered.push(name);
                } else {
                    eprintln!("warn: rival '{name}' not on this box — not pinned");
                }
            }
            let versions: Vec<String> = covered
                .iter()
                .map(|name| {
                    let (bin, vflag) = match *name {
                        // libdeflate-gzip has no --version; -V is its version flag.
                        "libdeflate" => ("libdeflate-gzip", "-V"),
                        b => (b, "--version"),
                    };
                    let v = Command::new(bin)
                        .arg(vflag)
                        .output()
                        .ok()
                        .map(|o| {
                            String::from_utf8_lossy(if o.stdout.is_empty() {
                                &o.stderr
                            } else {
                                &o.stdout
                            })
                            .lines()
                            .next()
                            .unwrap_or("?")
                            .trim()
                            .to_string()
                        })
                        .unwrap_or_else(|| "?".into());
                    format!("#   {name}: {v}\n")
                })
                .collect();
            let prov = format!(
                "# Rival fingerprints on the frozen fixtures. Regenerate on a box with all\n\
                 # rivals: cargo run --release --example fingerprint_tool -- pin-rivals\n\
                 # A pin is only comparable at these rival versions:\n{}# covered: {}\n",
                versions.join(""),
                covered.join(",")
            );
            write_pins(&format!("{PIN_DIR}/rivals.tsv"), &prov, &rows);
        }
        "ledger" => {
            let ours = ours_fingerprints();
            let rivals = parse_pins(&format!("{PIN_DIR}/rivals.tsv"));
            assert!(!rivals.is_empty(), "no rivals.tsv — run pin-rivals first");
            let mut won = 0;
            let mut lost = 0;
            let mut s = String::from(
                "# WON CELLS — append-only. Every row asserts: our bytes <= this rival's\n\
                 # bytes at this cell, forever. A lever that closes a cell appends it here\n\
                 # (cargo run --release --example fingerprint_tool -- ledger); nothing may\n\
                 # remove a row except a git revert of the lever that added it.\n\
                 fixture\tlevel\tthreads\trival\trival_bytes\n",
            );
            for ((f, l, t, rival), rfp) in &rivals {
                let Some(ofp) = ours.get(&(f.clone(), *l, *t, "gzippy".into())) else {
                    continue;
                };
                if ofp.file_bytes <= rfp.file_bytes {
                    s.push_str(&format!("{f}\t{l}\t{t}\t{rival}\t{}\n", rfp.file_bytes));
                    won += 1;
                } else {
                    lost += 1;
                }
            }
            std::fs::write(format!("{PIN_DIR}/ledger.tsv"), s).unwrap();
            println!("ledger: {won} won, {lost} open");
        }
        "report" => {
            let ours = ours_fingerprints();
            let rivals = parse_pins(&format!("{PIN_DIR}/rivals.tsv"));
            assert!(!rivals.is_empty(), "no rivals.tsv — run pin-rivals first");
            let mut axis_classes: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
            let mut losses: Vec<(u64, String)> = Vec::new();
            for ((f, l, t, rival), rfp) in &rivals {
                let Some(ofp) = ours.get(&(f.clone(), *l, *t, "gzippy".into())) else {
                    continue;
                };
                if ofp.file_bytes <= rfp.file_bytes {
                    continue;
                }
                let gap = ofp.file_bytes - rfp.file_bytes;
                let mut msg = format!(
                    "LOSE {f}:L{l}:T{t} vs {rival}  +{gap} B ({:+.3}%)\n",
                    gap as f64 * 100.0 / rfp.file_bytes as f64
                );
                for (axis, o, r) in ofp.diff(rfp).into_iter().take(6) {
                    if axis == "file_bytes" {
                        continue;
                    }
                    let rel = if r == 0 {
                        f64::INFINITY
                    } else {
                        (o as f64 - r as f64) * 100.0 / r as f64
                    };
                    msg.push_str(&format!(
                        "    {axis:<14} ours {o:>12}  rival {r:>12}  ({rel:+.1}%)\n"
                    ));
                    axis_classes
                        .entry(axis)
                        .or_default()
                        .push(format!("{f}:L{l}:T{t}:{rival}"));
                }
                losses.push((gap, msg));
            }
            losses.sort_by_key(|l| std::cmp::Reverse(l.0));
            for (_, m) in &losses {
                println!("{m}");
            }
            println!("== axis classes (cells sharing a moved mechanism) ==");
            let mut classes: Vec<_> = axis_classes.into_iter().collect();
            classes.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
            for (axis, cells) in classes {
                println!("  {axis:<14} {} cells: {}", cells.len(), cells.join(" "));
            }
            if losses.is_empty() {
                println!("no losing cells on the fixture grid.");
            }
        }
        "blocks" => {
            let Some(path) = std::env::args().nth(2) else {
                eprintln!("usage: fingerprint_tool blocks <file.gz>");
                std::process::exit(2);
            };
            let gz = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            let (fp, rows) = fingerprint_gzip_blocks(&gz)
                .unwrap_or_else(|e| panic!("cannot fingerprint {path}: {e}"));
            println!("{}", BlockFingerprint::TSV_FIELDS.join("\t"));
            for b in &rows {
                let vals: Vec<String> = b.tsv_values().iter().map(|v| v.to_string()).collect();
                println!("{}", vals.join("\t"));
            }
            eprintln!(
                "{}: {} blocks in {} member(s), {} -> {} bytes",
                path,
                rows.len(),
                fp.members,
                fp.decoded_bytes,
                fp.file_bytes
            );
        }
        "pin-blocks" => {
            let rows = ours_block_rows();
            std::fs::create_dir_all(PIN_DIR).unwrap();
            let path = format!("{PIN_DIR}/ours_blocks.tsv");
            let body = block_pins_body(&rows);
            let n: usize = rows.values().map(|v| v.len()).sum();
            std::fs::write(&path, body).unwrap();
            println!("wrote {path} ({} cells, {n} block rows)", rows.len());
        }
        "pin-t4" => {
            let ours = ours_fingerprints();
            let mut t4_rows = BTreeMap::new();
            let mut seam_rows = BTreeMap::new();
            for &fixture in fixtures::NAMES {
                for &level in LEVELS {
                    let t1 = &ours[&(fixture.to_string(), level, 1, "gzippy".into())];
                    let t4 = &ours[&(fixture.to_string(), level, 4, "gzippy".into())];
                    if t1 == t4 {
                        eprintln!(
                            "note: {fixture} L{level} T1==T4 byte-identical — either the T>1 \
                             route did not engage (the stdin trap: verify with a file input) \
                             or seams have been eliminated (the bit-splice goal). Know which."
                        );
                    }
                    t4_rows.insert((fixture.to_string(), level), t4.clone());
                    seam_rows.insert((fixture.to_string(), level), seam_row(t1, t4));
                }
            }
            std::fs::create_dir_all(PIN_DIR).unwrap();
            let t4_path = format!("{PIN_DIR}/ours_t4.tsv");
            std::fs::write(&t4_path, t4_pins_body(&t4_rows)).unwrap();
            println!("wrote {t4_path} ({} cells)", t4_rows.len());
            let seam_path = format!("{PIN_DIR}/seam_tax.tsv");
            std::fs::write(&seam_path, seam_tax_body(&seam_rows)).unwrap();
            let total: i64 = seam_rows.values().map(|r| r.tax_bytes).sum();
            println!(
                "wrote {seam_path} ({} cells, total tax {total:+} B)",
                seam_rows.len()
            );
        }
        "fp" => {
            // Fingerprint arbitrary .gz files (paths as trailing args) and, with
            // exactly two paths, print the per-axis diff. This is the Step-0 /
            // before-after instrument for stream-shape work (e.g. seam levers):
            // it answers "is the T4-vs-T1 delta exactly framing?" in one command.
            let paths: Vec<String> = std::env::args().skip(2).collect();
            assert!(
                !paths.is_empty(),
                "usage: fingerprint_tool fp <a.gz> [b.gz]"
            );
            let fps: Vec<StreamFingerprint> = paths
                .iter()
                .map(|p| {
                    let gz = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
                    fingerprint_gzip(&gz).unwrap_or_else(|e| panic!("fingerprint {p}: {e}"))
                })
                .collect();
            for (p, fp) in paths.iter().zip(&fps) {
                println!("{p}");
                for (f, v) in StreamFingerprint::TSV_FIELDS.iter().zip(fp.tsv_values()) {
                    println!("  {f:<14} {v}");
                }
            }
            if let [a, b] = &fps[..] {
                println!("== diff ({} vs {}) ==", paths[0], paths[1]);
                let d = a.diff(b);
                if d.is_empty() {
                    println!("  byte-shape identical on every axis");
                }
                for (axis, x, y) in d {
                    println!("  {axis:<14} {x:>12} vs {y:>12}");
                }
            }
        }
        other => {
            eprintln!(
                "usage: fingerprint_tool pin-ours | pin-rivals | ledger | report | blocks <file.gz> | pin-blocks | pin-t4 | fp (got '{other}')"
            );
            std::process::exit(2);
        }
    }
}
