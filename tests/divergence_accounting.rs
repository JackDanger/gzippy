//! Exact bit-accounting pins for the L1 divergence between the shipped T1
//! encoder and the exact libdeflate port (`ldx`) — the deep companion to the
//! ldx census pins (`tests/ldx_census.rs` / `tests/fingerprints/ldx_census.tsv`):
//! the census counts divergent POSITIONS; this file pins WHERE THE BYTES GO.
//!
//! Two tests:
//!
//! 1. `bit_accounting_residual_is_zero` — the self-checking-instrument
//!    invariant. For every synthetic fixture at L1 the attributed bits sum
//!    EXACTLY to 8 x (size_ours - size_ldx), per side and in total. If the
//!    analyzer's model of the streams (token alignment, per-block Huffman
//!    costing, header/EOB/padding framing) drifts from reality, this fails.
//!    A green residual proves the attribution is exact, not estimated.
//!
//! 2. `text_l1_attribution_matches_pins` — pins the per-class attribution
//!    for text/tabular/binary @L1 against
//!    `tests/fingerprints/divergence_accounting.tsv`, exactly. These are
//!    SNAPSHOT pins, not aspirations (byte-identity to libdeflate is a cage,
//!    not an asset): a moved attribution is not automatically bad, it must
//!    merely be DELIBERATE — regenerate and commit the TSV diff in the same
//!    PR, reviewed next to the size delta it bought:
//!
//!    ```text
//!    UPDATE_DIVERGENCE_ACCOUNTING=1 cargo test --release --test divergence_accounting
//!    ```
//!
//!    (An env var in TEST tooling is the standard snapshot pattern; the
//!    no-env-knobs rule binds the SHIPPED binary, which never reads this.)

use gzippy::compress::ldx_oracle::{account, BitAccounting};
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::PathBuf;

const REGEN_CMD: &str =
    "UPDATE_DIVERGENCE_ACCOUNTING=1 cargo test --release --test divergence_accounting";

/// The pinned grid: the fixtures where L1 diverges. `noise` is all
/// stored/identical framing and is covered by the residual test.
const PINNED_FIXTURES: &[&str] = &["text", "tabular", "binary"];
const LEVEL: u32 = 1;

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/divergence_accounting.tsv")
}

/// Compress one fixture with both encoders, verify BOTH streams roundtrip
/// byte-exactly through an independent decoder (bit accounting of an invalid
/// stream would be exact nonsense), then run the accounting.
fn measure(name: &str) -> BitAccounting {
    let input = fixtures::generate(name);
    let ldx = gzippy::compress::ldx::compress_for_diff(LEVEL, &input)
        .unwrap_or_else(|| panic!("{name}: ldx does not implement L{LEVEL}"));
    let ours = gzippy::compress::deflate::encode_deflate_bytes_to_vec(&input, LEVEL);
    for (side, stream) in [("ours", &ours), ("ldx", &ldx)] {
        let mut back = Vec::with_capacity(input.len());
        flate2::read::DeflateDecoder::new(&stream[..])
            .read_to_end(&mut back)
            .unwrap_or_else(|e| panic!("{name} L{LEVEL}: {side} does not decode: {e}"));
        assert_eq!(back, input, "{name} L{LEVEL}: {side} roundtrip failed");
    }
    account(&input, &ours, &ldx).unwrap_or_else(|e| panic!("{name} L{LEVEL}: account: {e}"))
}

/// The self-checking-instrument invariant: the residual is EXACTLY 0 on
/// every synthetic fixture, and each side's accounted bits equal 8 x its
/// compressed size (the per-side form catches two compensating errors that
/// a zero total residual could hide).
#[test]
fn bit_accounting_residual_is_zero() {
    for &name in fixtures::NAMES {
        let acc = measure(name);
        assert_eq!(
            acc.side_accounted_bits(),
            acc.side_total_bits(),
            "{name} L{LEVEL}: per-side accounted bits != 8 x compressed size \
             (header/EOB/padding/ident/region partition is not exhaustive)"
        );
        assert_eq!(
            acc.residual_bits(),
            0,
            "{name} L{LEVEL}: attributed bits do not sum to 8 x (size_ours - size_ldx) \
             — the analyzer's model of the streams has drifted from reality"
        );
    }
}

/// One pinned row: identity, sizes, and the six signed per-class bit deltas
/// (exact integers; bytes are deltas/8 and are rendered, not stored).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    ours_bytes: u64,
    ldx_bytes: u64,
    gap_bits: i64,
    attribution: [i64; 6],
}

impl Row {
    fn from_accounting(acc: &BitAccounting) -> Row {
        Row {
            ours_bytes: acc.ours_bytes,
            ldx_bytes: acc.ldx_bytes,
            gap_bits: acc.gap_bits(),
            attribution: acc.attribution_bits(),
        }
    }
}

fn tsv_header() -> String {
    format!(
        "# divergence_accounting.tsv — EXACT per-class bit attribution of the L1 size\n\
         # gap between the shipped T1 encoder and the exact libdeflate port (ldx),\n\
         # via src/compress/ldx_oracle.rs::account. The deep companion to\n\
         # ldx_census.tsv: the census counts divergent positions, this pins where\n\
         # the bytes go.\n\
         #\n\
         # Grid: {{text, tabular, binary}} x L1 at T1. All *_bits columns are SIGNED\n\
         # bit deltas (ours - ldx); they sum EXACTLY to gap_bits = 8 x (ours_bytes -\n\
         # ldx_bytes) — the residual-zero invariant in tests/divergence_accounting.rs\n\
         # proves the attribution is exact, not estimated. Classes:\n\
         #   we_lit_they_match / diff_len / diff_dist / we_match_they_lit — bits\n\
         #     spent inside divergent token regions, keyed by the region's first\n\
         #     divergent decision;\n\
         #   table_drift — bits both sides spent on IDENTICAL tokens (pure Huffman\n\
         #     table drift);\n\
         #   headers_eob — block headers + EOB symbols + final-byte padding.\n\
         #\n\
         # SNAPSHOT pins, not aspirations: a moved attribution is not automatically\n\
         # bad (byte-identity to libdeflate is a cage, not an asset); it must merely\n\
         # be deliberate. Regenerate and commit the diff in the same PR:\n\
         #   {REGEN_CMD}\n\
         fixture\tlevel\tours_bytes\tldx_bytes\tgap_bits\t{}\n",
        BitAccounting::ATTRIBUTION_CLASSES.join("\t")
    )
}

/// Render one cell's pinned-vs-now diff as a per-class old -> new table,
/// bytes beside bits (the absolute next to the relative).
fn render_cell_diff(fixture: &str, pinned: &Row, now: &Row) -> String {
    let mut out = format!(
        "DIVERGENCE ATTRIBUTION MOVED {fixture}:L{LEVEL}:T1\n    {:<22} {:>14} {:>14} {:>12}\n",
        "field", "pinned", "now", "delta"
    );
    let mut line = |name: &str, o: i64, n: i64| {
        if o != n {
            let _ = writeln!(
                out,
                "    {name:<22} {o:>14} {n:>14} {:>+12}  ({:+.1} B -> {:+.1} B)",
                n - o,
                o as f64 / 8.0,
                n as f64 / 8.0
            );
        }
    };
    line(
        "ours_bytes*8",
        pinned.ours_bytes as i64 * 8,
        now.ours_bytes as i64 * 8,
    );
    line(
        "ldx_bytes*8",
        pinned.ldx_bytes as i64 * 8,
        now.ldx_bytes as i64 * 8,
    );
    line("gap_bits", pinned.gap_bits, now.gap_bits);
    for (k, name) in BitAccounting::ATTRIBUTION_CLASSES.iter().enumerate() {
        line(name, pinned.attribution[k], now.attribution[k]);
    }
    out
}

/// Pin the per-class byte attribution for text/tabular/binary @L1, exactly.
#[test]
fn text_l1_attribution_matches_pins() {
    let mut current: BTreeMap<String, Row> = BTreeMap::new();
    for &name in PINNED_FIXTURES {
        let acc = measure(name);
        // A pin row may never encode an unsound accounting.
        assert_eq!(acc.residual_bits(), 0, "{name} L{LEVEL}: nonzero residual");
        current.insert(name.to_string(), Row::from_accounting(&acc));
    }

    // ── Regeneration mode: rewrite the TSV instead of asserting. ──────────
    if std::env::var("UPDATE_DIVERGENCE_ACCOUNTING").as_deref() == Ok("1") {
        let mut tsv = tsv_header();
        for &name in PINNED_FIXTURES {
            let r = &current[name];
            let _ = writeln!(
                tsv,
                "{name}\t{LEVEL}\t{}\t{}\t{}\t{}",
                r.ours_bytes,
                r.ldx_bytes,
                r.gap_bits,
                r.attribution.map(|v| v.to_string()).join("\t")
            );
        }
        std::fs::write(pins_path(), tsv).expect("write divergence_accounting.tsv");
        eprintln!(
            "divergence_accounting: regenerated {} ({} rows)",
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
    let mut pinned: BTreeMap<String, Row> = BTreeMap::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 11, "malformed pin row: {line:?}");
        assert_eq!(
            cols[1],
            LEVEL.to_string(),
            "unexpected level in row: {line:?}"
        );
        let n = |i: usize| -> i64 {
            cols[i]
                .parse()
                .unwrap_or_else(|_| panic!("non-integer column {i} in pin row: {line:?}"))
        };
        pinned.insert(
            cols[0].to_string(),
            Row {
                ours_bytes: n(2) as u64,
                ldx_bytes: n(3) as u64,
                gap_bits: n(4),
                attribution: [n(5), n(6), n(7), n(8), n(9), n(10)],
            },
        );
    }

    let mut failures = Vec::new();
    for (name, now) in &current {
        match pinned.get(name) {
            None => failures.push(format!(
                "UNPINNED CELL {name}:L{LEVEL}:T1 — no row in the TSV; regenerate pins"
            )),
            Some(pin) if pin != now => failures.push(render_cell_diff(name, pin, now)),
            Some(_) => {}
        }
    }
    for name in pinned.keys() {
        if !current.contains_key(name) {
            failures.push(format!(
                "STALE PIN CELL {name}:L{LEVEL}:T1 — pinned but no longer measured; \
                 regenerate pins"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\nthe bit attribution moved — if intended, regenerate with \
         UPDATE_DIVERGENCE_ACCOUNTING=1 and justify the movement in the commit message.\n\
         (Snapshot pins, not aspirations: converging to libdeflate's parse is not the\n\
         goal — size <= theirs is. A moved attribution is not automatically bad; it\n\
         must merely be deliberate, with the TSV diff committed in the same PR.)\n    {REGEN_CMD}\n",
        failures.join("\n")
    );
}
