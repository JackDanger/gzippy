//! Block-span distribution for any gzip stream. Answers whether a splitter fires on a fixed
//! cadence (low spread) or adaptively on content (high spread) — the PLACEMENT question that
//! survived after block COUNT was falsified at matched counts.
use gzippy::decompress::block_walker::walk_block_boundaries;
fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let blocks = walk_block_boundaries(&data).expect("walk");
        let mut spans: Vec<u64> = blocks.iter().map(|b| b.end_bit - b.start_bit).collect();
        spans.sort_unstable();
        if spans.is_empty() {
            continue;
        }
        let n = spans.len();
        let sum: u64 = spans.iter().sum();
        let mean = sum as f64 / n as f64;
        let med = spans[n / 2] as f64;
        let var = spans
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        println!(
            "{:<24} n={:<6} min={:<9} p50={:<10} max={:<10} mean={:<11.0} cv={:.3}",
            std::path::Path::new(&path)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            n,
            spans[0],
            med,
            spans[n - 1],
            mean,
            var.sqrt() / mean
        );
    }
}
