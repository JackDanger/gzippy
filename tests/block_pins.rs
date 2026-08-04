//! Per-BLOCK fingerprint pins: the whole-stream suite (fingerprint_suite.rs)
//! localizes a size change to a FILE; this suite localizes it to the exact
//! DEFLATE block that moved.
//!
//! Grid: every synthetic fixture (src/fixtures.rs) x levels {1,2,6,9} at T1,
//! through the REAL binary (CARGO_BIN_EXE_gzippy) via file-based invocation
//! (never stdin — the stdin trap silently measures the wrong path). T is
//! pinned to 1 because the per-block grid is a T1 mechanism surface; T>1 may
//! legally emit different bytes and is covered by the whole-stream suite.
//!
//! On mismatch the failure names: the first differing block index, WHICH AXIS
//! moved (header_bits = Huffman table description; data_bits/token counts =
//! entropy coding or the parse; span_bytes = a block-BOUNDARY shift, which
//! shows up as span-length changes cascading from one index onward), the old
//! vs new values, and the block totals before/after.
//!
//! Intentional change (a lever): regenerate the pin in the same PR —
//!     UPDATE_BLOCK_PINS=1 cargo test --release --test block_pins
//!     (or: cargo run --release --example fingerprint_tool -- pin-blocks)

use gzippy::decompress::block_walker::{fingerprint_gzip_blocks, BlockFingerprint};
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::process::Command;

const LEVELS: &[u32] = &[1, 2, 6, 9];
const PIN_PATH: &str = "tests/fingerprints/ours_blocks.tsv";

type Grid = BTreeMap<(String, u32), Vec<BlockFingerprint>>;

/// Per-block rows for the whole grid, from the shipped binary.
fn current_rows() -> Grid {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let mut out = Grid::new();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.path().join(name);
        std::fs::write(&path, &data).unwrap();
        for &level in LEVELS {
            let o = Command::new(bin)
                .args([
                    &format!("-{level}"),
                    "-p",
                    "1",
                    "-c",
                    path.to_str().unwrap(),
                ])
                .output()
                .unwrap();
            assert!(o.status.success(), "gzippy failed on {name} L{level} T1");
            let (fp, rows) = fingerprint_gzip_blocks(&o.stdout)
                .unwrap_or_else(|e| panic!("unparseable stream {name} L{level} T1: {e}"));
            assert_eq!(
                fp.decoded_bytes,
                data.len() as u64,
                "stream for {name} L{level} T1 decodes to the wrong size — corrupt stream?"
            );
            out.insert((name.to_string(), level), rows);
        }
    }
    out
}

/// The exact bytes of the pin file. MUST stay byte-identical to
/// `block_pins_body` in examples/fingerprint_tool.rs — this test's
/// UPDATE_BLOCK_PINS=1 mode and the tool's pin-blocks write the same file.
fn pins_body(rows: &Grid) -> String {
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

fn parse_pins(body: &str) -> Grid {
    let mut out = Grid::new();
    for line in body.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 2 + BlockFingerprint::TSV_FIELDS.len() {
            continue;
        }
        let vals: Vec<u64> = cols[2..].iter().filter_map(|c| c.parse().ok()).collect();
        let (Ok(level), Some(row)) = (cols[1].parse(), BlockFingerprint::from_tsv_values(&vals))
        else {
            continue;
        };
        out.entry((cols[0].to_string(), level))
            .or_default()
            .push(row);
    }
    out
}

/// Name the axis class of the first divergence, matching the mechanism
/// vocabulary of the whole-stream suite: header vs data entropy, parse
/// (token counts), or a block-boundary shift (span drift that cascades).
fn render_group_diff(
    fixture: &str,
    level: u32,
    pinned: &[BlockFingerprint],
    current: &[BlockFingerprint],
) -> String {
    let shared = pinned.len().min(current.len());
    let first = (0..shared).find(|&i| pinned[i] != current[i]);
    let mut msg = format!(
        "BLOCK ROWS MOVED {fixture}:L{level}:T1  total blocks: pinned {} -> current {}\n",
        pinned.len(),
        current.len()
    );
    let Some(first) = first else {
        // No divergence in the shared prefix: the stream gained/lost blocks
        // at the tail (still a boundary-grid change).
        msg.push_str(&format!(
            "  first {} blocks identical; the stream {} {} trailing block(s)\n  axis class: BLOCK GRID (count changed with no interior divergence)",
            shared,
            if current.len() > pinned.len() { "gained" } else { "lost" },
            pinned.len().abs_diff(current.len()),
        ));
        return msg;
    };
    let (p, c) = (&pinned[first], &current[first]);
    msg.push_str(&format!(
        "  first differing block: index {first} (member {}, btype pinned {} / current {}, final pinned {} / current {})\n",
        c.member_index, p.btype, c.btype, p.is_final, c.is_final
    ));
    let axes = c.diff(p); // (axis, current, pinned)
    for (axis, cur, pin) in &axes {
        msg.push_str(&format!(
            "    {axis:<12} pinned {pin:>12}  current {cur:>12}\n"
        ));
    }
    let moved: Vec<&str> = axes.iter().map(|(a, ..)| *a).collect();
    if moved.contains(&"span_bytes") {
        let cascade = (first + 1..shared)
            .filter(|&i| pinned[i].span_bytes != current[i].span_bytes)
            .count();
        msg.push_str(&format!(
            "  axis class: BOUNDARY/SPAN DRIFT — the uncompressed span moved at index {first} and\n  span_bytes differs on {cascade} of the {} following comparable block(s) (a boundary\n  shift cascades; the FIRST index is where the encoder's decision changed)",
            shared - first - 1
        ));
    } else if moved.iter().any(|a| *a == "literals" || *a == "matches") {
        msg.push_str(
            "  axis class: PARSE — token counts moved inside a fixed block span (the\n  match/literal decisions changed, not the block grid)",
        );
    } else if moved == ["header_bits"] {
        msg.push_str(
            "  axis class: ENTROPY HEADER — same tokens and span, different Huffman\n  table description",
        );
    } else if moved
        .iter()
        .all(|a| *a == "header_bits" || *a == "data_bits")
    {
        msg.push_str(
            "  axis class: ENTROPY CODING — same tokens and span, different code lengths\n  (header_bits/data_bits only)",
        );
    } else {
        msg.push_str(&format!("  axis class: MIXED ({})", moved.join(", ")));
    }
    msg
}

/// The per-block snapshot. Any T1 output change fails here naming the exact
/// block and axis that moved; intentional changes regenerate the pin in the
/// same PR (UPDATE_BLOCK_PINS=1).
#[test]
fn block_rows_match_pins() {
    let current = current_rows();
    if std::env::var("UPDATE_BLOCK_PINS").as_deref() == Ok("1") {
        let n: usize = current.values().map(|v| v.len()).sum();
        std::fs::write(PIN_PATH, pins_body(&current)).unwrap();
        eprintln!("wrote {PIN_PATH} ({} cells, {n} block rows)", current.len());
        return;
    }
    let body = std::fs::read_to_string(PIN_PATH).unwrap_or_else(|_| {
        panic!(
            "{PIN_PATH} missing — regenerate: UPDATE_BLOCK_PINS=1 cargo test --release --test block_pins"
        )
    });
    let pinned = parse_pins(&body);
    let mut diffs = Vec::new();
    for (cell, cur) in &current {
        match pinned.get(cell) {
            None => diffs.push(format!(
                "UNPINNED CELL {}:L{}:T1 — regenerate the block pins",
                cell.0, cell.1
            )),
            Some(pin) if pin != cur => diffs.push(render_group_diff(&cell.0, cell.1, pin, cur)),
            _ => {}
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            diffs.push(format!(
                "PINNED CELL {}:L{}:T1 not produced by the current grid",
                cell.0, cell.1
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "\n{}\n\nPer-block output changed. If INTENTIONAL (a lever), regenerate the pin in\nthis PR — its diff names the exact blocks your mechanism moved:\n    UPDATE_BLOCK_PINS=1 cargo test --release --test block_pins\nIf you did not mean to change output, the block index and axis above name\nwhat moved, before any profiler runs.\n",
        diffs.join("\n\n")
    );
}

/// The pin file on disk is exactly what the writer would emit today — no
/// hand edits, no drift between the test's writer and the tool's writer.
#[test]
fn block_pin_file_is_canonical() {
    let body = std::fs::read_to_string(PIN_PATH).unwrap_or_else(|_| {
        panic!(
            "{PIN_PATH} missing — regenerate: UPDATE_BLOCK_PINS=1 cargo test --release --test block_pins"
        )
    });
    let reparsed = parse_pins(&body);
    assert!(!reparsed.is_empty(), "{PIN_PATH} parsed to zero rows");
    assert_eq!(
        pins_body(&reparsed),
        body,
        "{PIN_PATH} is not in canonical writer format (hand-edited?) — regenerate it"
    );
}
