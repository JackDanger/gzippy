//! Uncompressed-offset block BOUNDARIES of a gzip stream, one per line.
//!
//! `blockcensus` counts blocks and `blockspans` measures their spread; neither
//! says WHERE the cuts are, which is the question when two splitters are being
//! compared. It exists for one named check: `lever/postparse-split` drives
//! `FindMinimum` with a cheap entropy estimator and takes the accept/reject
//! decision with the exact cost, and the thing that has to be measured is
//! whether the CHOSEN POSITIONS move — a size delta alone cannot distinguish
//! "the estimator ranks probes correctly" from "the positions moved and this
//! corpus happened to like it".
//!
//!   cargo run --release --example blockcuts -- FILE.gz
use gzippy::decompress::block_walker::walk_block_boundaries;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let blocks = walk_block_boundaries(&data).expect("walk");
        let mut off: u64 = 0;
        for b in &blocks {
            off += b.decoded_bytes;
            // The final block's end is the file end, not a chosen cut.
            if !b.is_final {
                println!("{off}");
            }
        }
    }
}
