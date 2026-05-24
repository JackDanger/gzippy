//! Phase B step-1 diagnostic — isolate marker-bookkeeping cost from
//! u16-ring/decoder costs in `bootstrap_with_deflate_block`'s hot loop.
//!
//! `plans/pure-rust-perf.md` step-zero finding: bootstrap takes
//! p50=14.2ms per chunk while the inflate-only bench shows 334 MB/s
//! pure-Rust → expected ~5-8ms. Per-block bootstrap is ~14× the
//! bench rate.
//!
//! Both code paths use multi-symbol Huffman caching (bench uses
//! libdeflate-style scalar; bootstrap uses
//! `HuffmanCodingShortBitsMultiCached` TRIPLE_SYM). So the gap is NOT
//! scalar-vs-SIMD as `plans/rust-rapidgzip.md` work item #1(b) framed
//! it.
//!
//! This bench monomorphizes the `read_internal_compressed_specialized
//! <CONTAINS_MARKERS>` template on identical input:
//!   - markers ON  via `Block::reset(Some(&mut out), None)` (empty window)
//!   - markers OFF via `Block::reset(Some(&mut out), Some(&[0u8; 32768]))`
//!     which calls `set_initial_window_impl` and flips
//!     `contains_marker_bytes = false` (deflate_block.rs:537).
//!
//! Large gap (true ≫ false): marker bookkeeping dominates; fix is the
//! `distance_marker += count` per-literal counter in the hot loop.
//! Small gap (both slow): cost is in u16-ring writes + modulo or
//! decoder cache footprint; next bench isolates ring vs decoder.
//!
//! Run on neurotic:
//! ```text
//! cargo bench --features pure-rust-inflate --bench bootstrap_marker_overhead -- --nocapture
//! ```

#[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
mod bench {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::deflate_block::Block;
    use std::time::Instant;

    fn load_silesia_deflate() -> Option<Vec<u8>> {
        let path = std::path::Path::new("benchmark_data/silesia-gzip.tar.gz");
        let data = std::fs::read(path).ok()?;
        let (_hdr, header) = gzippy::decompress::parallel::gzip_format::read_header(&data).ok()?;
        Some(data[header..data.len().saturating_sub(8)].to_vec())
    }

    fn synthetic_deflate(uncompressed_len: usize) -> Vec<u8> {
        use std::io::Write;
        let mut raw = vec![0u8; uncompressed_len];
        // High-entropy LCG so DYNAMIC blocks dominate (not STORED).
        for (i, b) in raw.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761) as u8;
        }
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&raw).unwrap();
        enc.finish().unwrap()
    }

    /// Drive blocks block-by-block via the public Block API.
    /// `with_markers = true` mirrors bootstrap's marker-emitting mode.
    /// `with_markers = false` seeds a 32 KiB zero window so back-refs
    /// resolve to literal source bytes (no marker bookkeeping).
    fn decode_full_stream(deflate: &[u8], with_markers: bool) -> (usize, usize) {
        let mut block = Block::new();
        let mut output: Vec<u16> = Vec::with_capacity(deflate.len() * 4);
        if with_markers {
            block.reset(Some(&mut output), None);
        } else {
            // Need a fresh Vec for set_initial_window to accept it.
            let zero_window = vec![0u8; 32768];
            block.reset(Some(&mut output), Some(&zero_window));
        }
        let mut bits = Bits::new(deflate);
        let mut blocks_decoded = 0usize;
        let mut bytes_emitted = 0usize;
        // Decode blocks until EOF/BFINAL or error.
        loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            blocks_decoded += 1;
            while !block.eob() {
                match block.read(&mut bits, &mut output, usize::MAX) {
                    Ok(n) => {
                        bytes_emitted += n;
                    }
                    Err(_) => return (blocks_decoded, bytes_emitted),
                }
            }
            if block.is_last_block() {
                break;
            }
        }
        (blocks_decoded, bytes_emitted)
    }

    fn bench_best_of(runs: usize, deflate: &[u8], with_markers: bool) -> (f64, usize) {
        let mut best_ns_per_byte = f64::INFINITY;
        let mut bytes_seen = 0usize;
        for _ in 0..runs {
            let t = Instant::now();
            let (_blocks, bytes) = decode_full_stream(deflate, with_markers);
            let dur_ns = t.elapsed().as_nanos() as f64;
            bytes_seen = bytes;
            if bytes > 0 {
                let ns = dur_ns / bytes as f64;
                if ns < best_ns_per_byte {
                    best_ns_per_byte = ns;
                }
            }
        }
        (best_ns_per_byte, bytes_seen)
    }

    pub fn run() {
        let deflate = if let Some(gz) = load_silesia_deflate() {
            eprintln!("loaded silesia: {} bytes deflate", gz.len());
            gz
        } else {
            eprintln!("benchmark_data/silesia-gzip.tar.gz missing — using 4 MiB synthetic deflate");
            synthetic_deflate(4 * 1024 * 1024)
        };

        // Warm one run before timing — JIT-y effects from rpmalloc + Huffman
        // table builds.
        let _ = decode_full_stream(&deflate, true);
        let _ = decode_full_stream(&deflate, false);

        let runs = 3;
        let (with_ns, with_bytes) = bench_best_of(runs, &deflate, true);
        let (no_ns, no_bytes) = bench_best_of(runs, &deflate, false);

        let with_mbps = if with_ns.is_finite() {
            1000.0 / with_ns
        } else {
            0.0
        };
        let no_mbps = if no_ns.is_finite() {
            1000.0 / no_ns
        } else {
            0.0
        };
        let gap_ratio = if no_ns > 0.0 { with_ns / no_ns } else { 0.0 };

        println!(
            "\nbootstrap marker overhead (best of {runs} runs, deflate={} bytes):",
            deflate.len()
        );
        println!(
            "  markers ON  : {with_ns:.2} ns/literal  ({with_mbps:.0} MB/s)  bytes={with_bytes}"
        );
        println!("  markers OFF : {no_ns:.2} ns/literal  ({no_mbps:.0} MB/s)  bytes={no_bytes}");
        println!("  ratio ON/OFF: {gap_ratio:.2}x");
        println!();
        println!("Interpretation:");
        println!("  ratio ≥ 1.5x  → marker bookkeeping dominates the hot loop");
        println!("                  (fix: per-literal `distance_marker += count`).");
        println!("  ratio ≈ 1.0x  → cost is u16-ring/modulo or decoder cache");
        println!("                  (next bench: flat u8 output vs u16 ring).");
    }
}

#[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
fn main() {
    bench::run();
}

#[cfg(not(all(target_arch = "x86_64", feature = "pure-rust-inflate")))]
fn main() {
    eprintln!("bootstrap_marker_overhead requires --features pure-rust-inflate on x86_64");
}
