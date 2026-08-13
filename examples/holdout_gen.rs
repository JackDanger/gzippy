//! holdout_gen — materialize the NEVER-TUNE holdout corpus from seeds.
//!
//! Usage:  holdout_gen <outdir>
//!
//! Writes every member of `gzippy::holdout::NAMES` into <outdir> and prints
//! one line per member: `name<TAB>bytes<TAB>sha256`. Every sha256 is verified
//! against `gzippy::holdout::PINS`; any mismatch is reported and the process
//! exits 3, so a drifted generator can never silently grade as "the holdout".
//! (Repinning is a conscious act: update PINS in src/holdout.rs, and know that
//! it voids every previously recorded holdout win-rate.)
//!
//! The holdout is never stored in-repo as data — this binary IS the corpus.
//! See docs/generalization-instruments.md for the protocol.

use gzippy::holdout;
use std::io::Write;

fn main() {
    let outdir = match std::env::args().nth(1) {
        Some(d) => std::path::PathBuf::from(d),
        None => {
            eprintln!("usage: holdout_gen <outdir>");
            std::process::exit(2);
        }
    };
    std::fs::create_dir_all(&outdir).expect("create outdir");

    let mut drifted = 0u32;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for &name in holdout::NAMES {
        let data = holdout::generate(name);
        let sha = holdout::sha256_hex(&data);
        std::fs::write(outdir.join(name), &data).expect("write member");
        writeln!(out, "{name}\t{}\t{sha}", data.len()).unwrap();
        let pinned = holdout::PINS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or("");
        if sha != pinned {
            eprintln!("holdout_gen: PIN MISMATCH for {name}: generated {sha}, pinned {pinned}");
            drifted += 1;
        }
    }
    if drifted > 0 {
        eprintln!(
            "holdout_gen: {drifted} member(s) drifted from src/holdout.rs PINS — refusing. \
             A generator change must consciously repin src/holdout.rs, which \
             voids every previously recorded holdout win-rate."
        );
        std::process::exit(3);
    }
}
