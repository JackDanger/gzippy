//! PROPOSER RECALL — how many profitable DEFLATE block boundaries does the
//! 10-bucket SAD splitter never even offer?
//!
//! The blocked question is PR #342's surviving point 3 (`9c4cccb6`): "zopfli has
//! NO proposer — it searches for the minimum. A confirm-only design is ceilinged
//! by the proposer's recall." That ceiling had never been measured, so every
//! confirm-after-propose splitter design — including the one #342 parked — was
//! being costed against an unknown bound.
//!
//!     cargo run --release --features split-recall --example proposer_recall -- \
//!         --levels 3,9 [--fine 0] CORPUSDIR_OR_FILES...
//!
//! Prints one row per (file, level): the confusion matrix over candidate
//! boundaries and the bits sitting in the missed ones. See
//! `src/compress/deflate/parse/split_recall.rs` for the profitability criterion
//! and its five stated limits.

#[cfg(not(feature = "split-recall"))]
fn main() {
    eprintln!("proposer_recall needs `--features split-recall`");
    std::process::exit(2);
}

#[cfg(feature = "split-recall")]
fn main() {
    use gzippy::compress::deflate::parse::split_recall::{analyze_file, Recall};

    let mut levels: Vec<u32> = vec![3, 9];
    let mut fine: usize = 0;
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--levels" => {
                let v = args.next().expect("--levels needs a value");
                levels = v.split(',').map(|s| s.parse().expect("level")).collect();
            }
            "--fine" => {
                fine = args
                    .next()
                    .expect("--fine needs a value")
                    .parse()
                    .expect("stride");
            }
            _ => paths.push(std::path::PathBuf::from(a)),
        }
    }
    if paths.is_empty() {
        eprintln!("usage: proposer_recall [--levels 3,9] [--fine N] FILE|DIR ...");
        std::process::exit(2);
    }

    // Expand directories, sorted so the table is reproducible.
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut kids: Vec<_> = std::fs::read_dir(&p)
                .expect("read_dir")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect();
            kids.sort();
            files.extend(kids);
        } else {
            files.push(p);
        }
    }

    println!("{}", Recall::header());
    let mut all: Vec<Recall> = Vec::new();
    for f in &files {
        let data = match std::fs::read(f) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}: {e}", f.display());
                continue;
            }
        };
        if data.is_empty() {
            continue;
        }
        let name = f
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        for &l in &levels {
            let r = analyze_file(&name, &data, l, fine);
            println!("{}", r.row());
            all.push(r);
        }
    }

    // Totals per level, then overall.
    println!();
    for &l in &levels {
        let rows: Vec<&Recall> = all.iter().filter(|r| r.level == l).collect();
        if rows.is_empty() {
            continue;
        }
        let mut t = Recall {
            file: format!("TOTAL L{l}"),
            level: l,
            ..Default::default()
        };
        for r in rows {
            t.input_bytes += r.input_bytes;
            t.output_bytes += r.output_bytes;
            t.blocks += r.blocks;
            t.tp += r.tp;
            t.fp += r.fp;
            t.forced += r.forced;
            t.fneg += r.fneg;
            t.tn += r.tn;
            t.tp_bits += r.tp_bits;
            t.fp_bits += r.fp_bits;
            t.fn_bits_sum += r.fn_bits_sum;
            t.fn_bits_best += r.fn_bits_best;
            t.fn_bits_greedy += r.fn_bits_greedy;
            t.fn_bits_max = t.fn_bits_max.max(r.fn_bits_max);
            t.fine_candidates += r.fine_candidates;
            t.fine_profitable += r.fine_profitable;
            t.fine_bits_best += r.fine_bits_best;
            t.predicted_bits += r.predicted_bits;
        }
        println!("{}", t.row());
        println!(
            "  L{l}: FP cost {} bits; largest single missed cut {} bits ({} B); \
best-single-per-block {} B; realisable(greedy re-split) {} B = {:.4}% of {} output bytes; \
independent-sum upper bound {} B ({:.2}% miss)",
            t.fp_bits,
            t.fn_bits_max,
            t.fn_bits_max / 8,
            t.fn_bits_best / 8,
            t.fn_bits_greedy / 8,
            t.missed_greedy_pct_of_output(),
            t.output_bytes,
            t.fn_bits_sum / 8,
            100.0 * t.missed_saving_fraction_indep(),
        );
        // Model self-check: this cost model's price for the blocks AS SHIPPED,
        // against the bytes the encoder actually wrote.
        println!(
            "  L{l}: cost-model self-check: predicted {} B vs actual {} B ({:+.4}%)",
            t.predicted_bits / 8,
            t.output_bytes,
            100.0 * ((t.predicted_bits as f64 / 8.0) - t.output_bytes as f64)
                / t.output_bytes as f64
        );
        if fine > 0 {
            println!(
                "  L{l}: fine grid (every {fine} seqs, never-consulted positions): \
{} candidates, {} profitable, best-per-block {} bits ({} B)",
                t.fine_candidates,
                t.fine_profitable,
                t.fine_bits_best,
                t.fine_bits_best / 8
            );
        }
    }
}
