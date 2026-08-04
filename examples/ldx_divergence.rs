//! ldx_divergence — per-decision oracle: shipped encoder vs the exact
//! libdeflate port (`src/compress/ldx/`).
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example ldx_divergence -- <file> <level>
//! cargo run --release --example ldx_divergence -- fixture:<name> <level>
//! ```
//!
//! `fixture:<name>` uses a synthetic fixture from `src/fixtures.rs`
//! (text, tabular, binary, noise) instead of a file on disk.
//!
//! Compresses `<file>` with BOTH the shipped T1 encoder path
//! (`compress::deflate::encode_deflate_bytes_to_vec`) and the ldx port
//! (`compress::ldx::compress_for_diff`) in-process, decodes both raw DEFLATE
//! outputs back into token streams with the in-tree block walker
//! (`decompress::block_walker::walk_deflate_tokens`), and reports:
//!
//! * the FIRST divergent decision — absolute uncompressed position, block
//!   index in each stream, our token vs theirs, e.g.
//!   `pos 48213: ours=match(3,892) ldx=literal 'e'`;
//! * aggregate stats — total divergent positions, counts by class
//!   (we-literal-they-match, we-match-they-literal, both-match-different-len,
//!   both-match-different-dist, plus the misaligned cascade starts), and
//!   per-stream block counts/sizes.
//!
//! Why it exists: the whole-file differential
//! (`scripts/campaign/ldx-differential.sh`) can only say "file X is +N bytes
//! at level L", which starts a box round-trip to investigate. This tool
//! converts that into "the first divergent decision is at position P" in
//! milliseconds on a laptop. Both encoders run at T1; ldx implements levels
//! 0-9 (the exotic 10-12 are not ported).
//!
//! Tooling only: nothing here routes into a shipping path, and both outputs
//! are validated to decode back to the input before any diff is reported.

use std::io::Read;

fn main() {
    let mut args = std::env::args().skip(1);
    let (path, level) = match (args.next(), args.next().and_then(|l| l.parse::<u32>().ok())) {
        (Some(p), Some(l)) => (p, l),
        _ => {
            eprintln!("usage: ldx_divergence <file> <level 0-9>");
            std::process::exit(2);
        }
    };
    let input = if let Some(name) = path.strip_prefix("fixture:") {
        gzippy::fixtures::generate(name)
    } else {
        let mut input = Vec::new();
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("open {path}: {e}"))
            .read_to_end(&mut input)
            .unwrap();
        input
    };

    let ldx = gzippy::compress::ldx::compress_for_diff(level, &input)
        .unwrap_or_else(|| panic!("level {level} is not ported in ldx (only 0-9 are)"));
    let ours = gzippy::compress::deflate::encode_deflate_bytes_to_vec(&input, level);

    // Validity first: both streams must decode byte-exactly to the input.
    // flate2 (an independent decoder) rather than our decompress_raw_bytes
    // helper: the helper rejects a valid stored-block grid our L1 emits on
    // incompressible data (e.g. the `noise` fixture), which flate2 and the
    // bit-walker both accept.
    for (name, stream) in [("ours", &ours), ("ldx", &ldx)] {
        let mut back = Vec::with_capacity(input.len());
        flate2::read::DeflateDecoder::new(&stream[..])
            .read_to_end(&mut back)
            .unwrap_or_else(|e| panic!("{name} stream does not decode: {e}"));
        assert_eq!(back, input, "{name} stream does not roundtrip {path}");
    }

    println!(
        "{path} level {level}: input {} B, ours {} B, ldx {} B ({:+} B)",
        input.len(),
        ours.len(),
        ldx.len(),
        ours.len() as i64 - ldx.len() as i64
    );

    let report = gzippy::compress::ldx_oracle::compare(&ours, &ldx).expect("token walk failed");

    let fmt_blocks = |s: &gzippy::compress::ldx_oracle::StreamStats| {
        let lens = &s.block_uncompressed_lens;
        let (min, max) = (
            lens.iter().min().copied().unwrap_or(0),
            lens.iter().max().copied().unwrap_or(0),
        );
        format!(
            "{} blocks (uncompressed min {min} / max {max} B), {} literals, {} matches, {} B compressed",
            s.block_count(),
            s.literals,
            s.matches,
            s.compressed_len
        )
    };
    println!("  ours: {}", fmt_blocks(&report.ours));
    println!("  ldx:  {}", fmt_blocks(&report.ldx));

    match report.first {
        None => {
            println!(
                "  zero divergence: every decision identical ({} tokens)",
                report.ldx.literals + report.ldx.matches
            );
        }
        Some(f) => {
            use gzippy::compress::ldx_oracle::DivergenceKind;
            let (ours_desc, ldx_desc) = match f.kind {
                DivergenceKind::Token { ours, ldx } => {
                    (ours.describe(&input), ldx.describe(&input))
                }
                DivergenceKind::BlockBoundary { ours, ldx } => {
                    let d = |bt: Option<u8>| match bt {
                        Some(0) => "block-start(stored)".to_string(),
                        Some(1) => "block-start(fixed)".to_string(),
                        Some(2) => "block-start(dynamic)".to_string(),
                        Some(b) => format!("block-start(btype {b})"),
                        None => "mid-block".to_string(),
                    };
                    (d(ours), d(ldx))
                }
            };
            println!(
                "  FIRST divergence at pos {} (block {} ours / {} ldx): ours={} ldx={}",
                f.pos,
                report.ours.block_index_at(f.pos),
                report.ldx.block_index_at(f.pos),
                ours_desc,
                ldx_desc
            );
            println!(
                "  divergent positions: {} total | we-literal-they-match {} | we-match-they-literal {} | both-match-different-len {} | both-match-different-dist {} | block-boundary {} | misaligned cascade starts {}",
                report.total_divergent(),
                report.we_literal_they_match,
                report.we_match_they_literal,
                report.both_match_different_len,
                report.both_match_different_dist,
                report.block_boundary,
                report.misaligned_starts
            );
        }
    }
}
