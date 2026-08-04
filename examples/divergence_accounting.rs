//! divergence_accounting — EXACT per-region Huffman bit accounting of the
//! size gap between the shipped T1 encoder and the exact libdeflate port
//! (`ldx`), on top of the divergence oracle (`src/compress/ldx_oracle.rs`).
//!
//! The ldx census (`tests/ldx_census.rs`) counts divergent POSITIONS; this
//! tool weighs them. It compresses the input with BOTH encoders in-process at
//! T1, aligns the two token streams, and prints:
//!
//!  1. divergence-class histograms with length buckets (3, 4-7, 8-15, 16+)
//!     and distance buckets (<=64, <=4096, >4096) for each side;
//!  2. exact bit accounting: both streams partitioned into maximal
//!     identical/divergent regions, every token costed from its OWN block's
//!     real Huffman tables, the total size delta attributed across
//!     we-lit-they-match / diff-len / diff-dist / we-match-they-lit /
//!     identical-tokens-table-drift / headers+EOB(+padding).
//!
//! The invariant (pinned by `tests/divergence_accounting.rs`): the attributed
//! bits sum EXACTLY to 8 x (size_ours - size_ldx). The residual is printed;
//! nonzero means the analyzer's model of the streams has drifted from
//! reality.
//!
//! Usage:
//!   cargo run --release --example divergence_accounting -- \
//!       [--level N] [fixture-or-file ...]
//!
//! Arguments are `src/fixtures.rs` names (text/tabular/binary/noise) or file
//! paths; default is every synthetic fixture. Level defaults to 1 (the only
//! level that diverges today).

use std::io::Read;

use gzippy::compress::ldx_oracle::{account, BitAccounting, DivergenceClass, Token};
use gzippy::fixtures;

/// Histogram over the buckets named in the header comment.
#[derive(Default, Clone)]
struct Hist {
    len3: u64,
    len4_7: u64,
    len8_15: u64,
    len16p: u64,
    d64: u64,
    d4096: u64,
    dbig: u64,
    n: u64,
    len_sum: u64,
    dist_sum: u64,
}

impl Hist {
    fn add(&mut self, t: &Token) {
        debug_assert!(!t.is_literal());
        self.n += 1;
        self.len_sum += t.len as u64;
        self.dist_sum += t.dist as u64;
        match t.len {
            3 => self.len3 += 1,
            4..=7 => self.len4_7 += 1,
            8..=15 => self.len8_15 += 1,
            _ => self.len16p += 1,
        }
        match t.dist {
            0..=64 => self.d64 += 1,
            65..=4096 => self.d4096 += 1,
            _ => self.dbig += 1,
        }
    }

    fn print(&self, label: &str) {
        if self.n == 0 {
            println!("    {label}: (none)");
            return;
        }
        println!(
            "    {label}: n={} | len 3:{} 4-7:{} 8-15:{} 16+:{} (mean {:.2}) | dist <=64:{} 65-4096:{} >4096:{} (mean {:.0})",
            self.n, self.len3, self.len4_7, self.len8_15, self.len16p,
            self.len_sum as f64 / self.n as f64,
            self.d64, self.d4096, self.dbig,
            self.dist_sum as f64 / self.n as f64,
        );
    }
}

fn main() {
    let mut level: u32 = 1;
    let mut names: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" | "-l" => {
                level = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| die("--level needs an integer"));
            }
            "--help" | "-h" => {
                println!(
                    "usage: divergence_accounting [--level N] [fixture-or-file ...]\n\
                     fixtures: {:?}; default all fixtures at L1",
                    fixtures::NAMES
                );
                return;
            }
            _ => names.push(a),
        }
    }
    if names.is_empty() {
        names = fixtures::NAMES.iter().map(|s| s.to_string()).collect();
    }
    for name in &names {
        analyze(name, level);
        println!();
    }
}

fn die(msg: &str) -> ! {
    eprintln!("divergence_accounting: {msg}");
    std::process::exit(1)
}

fn load(name: &str) -> Vec<u8> {
    if fixtures::NAMES.contains(&name) {
        fixtures::generate(name)
    } else {
        std::fs::read(name).unwrap_or_else(|e| die(&format!("cannot read '{name}': {e}")))
    }
}

fn analyze(name: &str, level: u32) {
    let input = load(name);
    let ldx = gzippy::compress::ldx::compress_for_diff(level, &input)
        .unwrap_or_else(|| die(&format!("ldx does not implement level {level}")));
    let ours = gzippy::compress::deflate::encode_deflate_bytes_to_vec(&input, level);

    // Validity first: bit accounting of an invalid stream is exact nonsense.
    for (side, stream) in [("ours", &ours), ("ldx", &ldx)] {
        let mut back = Vec::with_capacity(input.len());
        flate2::read::DeflateDecoder::new(&stream[..])
            .read_to_end(&mut back)
            .unwrap_or_else(|e| die(&format!("{side} does not decode: {e}")));
        assert_eq!(back, input, "{side} roundtrip failed on {name} L{level}");
    }

    let acc = account(&input, &ours, &ldx).unwrap_or_else(|e| die(&format!("account: {e}")));

    println!(
        "=== {name} L{level} T1 | input {} B | ours {} B | ldx {} B | gap {:+} B ===",
        input.len(),
        acc.ours_bytes,
        acc.ldx_bytes,
        acc.ours_bytes as i64 - acc.ldx_bytes as i64,
    );

    // ── (1) divergence-class histograms, per side ─────────────────────────
    // For the one-sided classes only the match side has (len,dist) to
    // histogram; for diff-len/diff-dist both sides do.
    let mut ours_h = vec![Hist::default(); 4];
    let mut ldx_h = vec![Hist::default(); 4];
    for (c, a, b) in &acc.aligned_pairs {
        if !a.is_literal() {
            ours_h[*c as usize].add(a);
        }
        if !b.is_literal() {
            ldx_h[*c as usize].add(b);
        }
    }
    println!(
        "  aligned divergent decisions ({}):",
        acc.aligned_pairs.len()
    );
    for c in DivergenceClass::ALL {
        let (o, l) = (&ours_h[c as usize], &ldx_h[c as usize]);
        if o.n == 0 && l.n == 0 {
            continue;
        }
        println!("    {} ({}):", c.name(), o.n.max(l.n));
        match c {
            DivergenceClass::WeLitTheyMatch => l.print("ldx match"),
            DivergenceClass::WeMatchTheyLit => o.print("our match"),
            _ => {
                o.print("ours");
                l.print("ldx ");
            }
        }
    }

    // ── (2) exact bit accounting ──────────────────────────────────────────
    println!("  exact bit accounting (gap = ours - ldx):");
    println!(
        "    {:<22} {:>8} {:>12} {:>12} {:>10} {:>10}",
        "class", "regions", "bits_ours", "bits_ldx", "delta_bit", "delta_B"
    );
    let attr = acc.attribution_bits();
    for c in DivergenceClass::ALL {
        let r = &acc.regions[c as usize];
        row(
            c.name(),
            Some(r.regions),
            r.bits_ours,
            r.bits_ldx,
            r.delta_bits(),
        );
    }
    row(
        "table_drift",
        Some(acc.ident_tokens),
        acc.ident_bits[0],
        acc.ident_bits[1],
        attr[4],
    );
    let frame = |s: usize| acc.header_bits[s] + acc.eob_bits[s] + acc.padding_bits[s];
    row("headers_eob", None, frame(0), frame(1), attr[5]);
    println!(
        "      (headers {:+}, EOB {:+}, final padding {:+} bits)",
        acc.header_bits[0] as i64 - acc.header_bits[1] as i64,
        acc.eob_bits[0] as i64 - acc.eob_bits[1] as i64,
        acc.padding_bits[0] as i64 - acc.padding_bits[1] as i64,
    );
    println!(
        "    {:<22} {:>8} {:>12} {:>12} {:>10} {:>10.1}",
        "TOTAL (8 x sizes)",
        "",
        acc.side_total_bits()[0],
        acc.side_total_bits()[1],
        format!("{:+}", acc.gap_bits()),
        acc.gap_bits() as f64 / 8.0,
    );
    println!(
        "    residual (gap - attributed): {} bits{}",
        acc.residual_bits(),
        if acc.residual_bits() == 0 {
            " — attribution is EXACT"
        } else {
            " — MODEL DRIFT, see tests/divergence_accounting.rs"
        }
    );
    debug_assert_eq!(acc.side_accounted_bits(), acc.side_total_bits());
    let _ = BitAccounting::ATTRIBUTION_CLASSES;
}

fn row(name: &str, count: Option<u64>, bits_o: u64, bits_l: u64, delta: i64) {
    println!(
        "    {:<22} {:>8} {:>12} {:>12} {:>10} {:>10.1}",
        name,
        count.map(|c| c.to_string()).unwrap_or_default(),
        bits_o,
        bits_l,
        format!("{delta:+}"),
        delta as f64 / 8.0,
    );
}
