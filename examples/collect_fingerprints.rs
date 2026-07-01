//! Walk gzip files via `gzippy::decompress::block_walker` and emit a
//! corpus fingerprint summary (top-N most-frequent dynamic-Huffman
//! fingerprints + block-type histogram).
//!
//! Output schema (stdout JSON):
//! ```json
//! {
//!   "files_scanned": N,
//!   "blocks_total": N,
//!   "blocks_by_btype": {"stored": N, "fixed": N, "dynamic": N},
//!   "decoded_bytes_total": N,
//!   "unique_fingerprints": N,
//!   "top_n": [
//!     {"fingerprint": "0x...", "frequency": N, "decoded_bytes": N,
//!      "litlen_nonzero": N, "dist_nonzero": N},
//!     ...
//!   ]
//! }
//! ```
//!
//! Feeds `crates/gzippy-inflate/build.rs` (v2): the JSON is checked
//! into `profiles/aot-decoder-fingerprints.json`; the build script
//! reads it and bakes top-256 CHD tables into the AOT path.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::ExitCode;

use gzippy::decompress::block_walker::walk_block_boundaries;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: collect_fingerprints <file.gz> [file.gz...]");
        return ExitCode::from(2);
    }

    // (frequency, total decoded bytes, litlen_nonzero, dist_nonzero)
    let mut counts: HashMap<u64, (u64, u64, u32, u32)> = HashMap::new();
    let mut files_scanned = 0u32;
    let mut blocks_total = 0u64;
    let mut stored = 0u64;
    let mut fixed = 0u64;
    let mut dynamic = 0u64;
    let mut decoded_bytes_total = 0u64;

    for path in &args {
        let gz = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error reading {path}: {e}");
                continue;
            }
        };
        let blocks = match walk_block_boundaries(&gz) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error walking {path}: {e}");
                continue;
            }
        };
        files_scanned += 1;
        for b in &blocks {
            blocks_total += 1;
            decoded_bytes_total += b.decoded_bytes;
            match b.btype {
                0 => stored += 1,
                1 => fixed += 1,
                2 => {
                    dynamic += 1;
                    if let (Some(fp), Some(ll), Some(d)) =
                        (b.fingerprint, b.litlen_nonzero, b.dist_nonzero)
                    {
                        let e = counts.entry(fp).or_insert((0, 0, ll, d));
                        e.0 += 1;
                        e.1 += b.decoded_bytes;
                    }
                }
                _ => {}
            }
        }
        eprintln!(
            "scanned {path}: {} blocks ({} dynamic), {} decoded bytes",
            blocks.len(),
            blocks.iter().filter(|b| b.btype == 2).count(),
            blocks.iter().map(|b| b.decoded_bytes).sum::<u64>(),
        );
    }

    // Sort by frequency desc.
    let mut sorted: Vec<(u64, u64, u64, u32, u32)> = counts
        .iter()
        .map(|(&k, &(f, b, l, d))| (k, f, b, l, d))
        .collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
    let top_n: Vec<_> = sorted.iter().take(256).collect();

    println!("{{");
    println!("  \"files_scanned\": {},", files_scanned);
    println!("  \"blocks_total\": {},", blocks_total);
    println!("  \"blocks_by_btype\": {{");
    println!("    \"stored\": {},", stored);
    println!("    \"fixed\": {},", fixed);
    println!("    \"dynamic\": {}", dynamic);
    println!("  }},");
    println!("  \"decoded_bytes_total\": {},", decoded_bytes_total);
    println!("  \"unique_fingerprints\": {},", counts.len());
    println!("  \"top_n\": [");
    for (i, &(fp, freq, bytes, ll, d)) in top_n.iter().copied().enumerate() {
        let comma = if i + 1 < top_n.len() { "," } else { "" };
        println!("    {{");
        println!("      \"fingerprint\": \"{:#018x}\",", fp);
        println!("      \"frequency\": {},", freq);
        println!("      \"decoded_bytes\": {},", bytes);
        println!("      \"litlen_nonzero\": {},", ll);
        println!("      \"dist_nonzero\": {}", d);
        println!("    }}{comma}");
    }
    println!("  ]");
    println!("}}");

    ExitCode::SUCCESS
}
