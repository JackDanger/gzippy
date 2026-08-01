//! BTYPE census for ANY gzip stream, ours or a rival's.
//!
//! The per-block cost class (photo.jpg L2, weights.safetensors L9) was blocked on one unknown:
//! what BLOCK TYPES gzip emits where we emit dynamic. Every count-based reading of that class
//! has been falsified, and the remaining question is per-block COST, which starts with BTYPE.
//!
//! `decompress::block_walker::walk_block_boundaries` already answers it for any stream — it is
//! part of the finished decoder. This example just exposes it on the CLI.
//!
//!   cargo run --release --example blockcensus -- FILE.gz [FILE2.gz ...]
use gzippy::decompress::block_walker::walk_block_boundaries;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: blockcensus FILE.gz [FILE.gz ...]");
        std::process::exit(2);
    }
    println!(
        "{:<34} {:>7} {:>7} {:>7} {:>7} {:>12} {:>9}",
        "file", "blocks", "stored", "fixed", "dynamic", "bytes", "bits/blk"
    );
    for path in &args {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        match walk_block_boundaries(&data) {
            Ok(blocks) => {
                let (mut st, mut fx, mut dy) = (0usize, 0usize, 0usize);
                for b in &blocks {
                    match b.btype {
                        0 => st += 1,
                        1 => fx += 1,
                        _ => dy += 1,
                    }
                }
                let n = blocks.len().max(1);
                let bits = blocks.last().map(|b| b.end_bit).unwrap_or(0);
                println!(
                    "{:<34} {:>7} {:>7} {:>7} {:>7} {:>12} {:>9}",
                    std::path::Path::new(path)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| path.clone()),
                    blocks.len(),
                    st,
                    fx,
                    dy,
                    data.len(),
                    bits as usize / n
                );
            }
            Err(e) => eprintln!("{path}: walk failed: {e}"),
        }
    }
}
