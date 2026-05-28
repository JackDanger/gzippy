//! Encapsulated inner-DEFLATE-decode microbench (2026-05-28).
//!
//! Isolates the inner inflate loop from the parallel pipeline / markers /
//! CRC / page-fault churn so it can be measured with EXACT, noise-free
//! single-threaded perf counters (instructions/byte, cycles/byte) on any
//! core with no container freeze required.
//!
//! Decodes one raw-DEFLATE stream (silesia, gzip header/trailer stripped)
//! repeatedly through ONE decoder chosen by argv, so `perf stat` attributes
//! cleanly:
//!
//!   perf stat -e instructions,cpu_core/cycles/ -- inner_bench pure 5
//!   perf stat -e instructions,cpu_core/cycles/ -- inner_bench libdeflate 5
//!
//! Then instructions/byte = instructions / (iters * output_len).
use std::time::Instant;

fn load_raw_deflate(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Strip the gzip header (parse) + 8-byte trailer to get raw DEFLATE.
    let (_h, header) =
        gzippy::decompress::parallel::gzip_format::read_header(&data).expect("parse gzip header");
    data[header..data.len() - 8].to_vec()
}

#[cfg(feature = "pure-rust-inflate")]
fn decode_pure(deflate: &[u8], out: &mut [u8]) -> usize {
    use gzippy::decompress::inflate::resumable::ResumableInflate2;
    let mut d = ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).expect("init");
    d.set_window(&[]).expect("window");
    let mut total = 0usize;
    loop {
        let r = d.read_stream(&mut out[total..]).expect("read");
        total += r.bytes_written;
        if r.finished || r.bytes_written == 0 {
            break;
        }
    }
    total
}

fn decode_libdeflate(deflate: &[u8], out: &mut [u8]) -> usize {
    let mut d = libdeflater::Decompressor::new();
    d.deflate_decompress(deflate, out).expect("libdeflate")
}

// In-repo libdeflate-STYLE pure-Rust decoder (NON-resumable). The A/B that
// isolates whether ResumableInflate2's 2x-vs-libdeflate instruction gap is
// the RESUMABLE CONTRACT (yield checks / pending_match / snapshots) or the
// core algorithm: if this is ~libdeflate-fast, it's the contract.
#[cfg(feature = "pure-rust-inflate")]
fn decode_consume_first(deflate: &[u8], out: &mut [u8]) -> usize {
    gzippy::decompress::inflate::consume_first_decode::inflate_consume_first(deflate, out)
        .expect("consume_first")
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "pure".into());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "benchmark_data/silesia-gzip.tar.gz".into());

    let deflate = load_raw_deflate(&path);
    // Learn the output size once (over-allocate generously, then size exact).
    let mut probe = vec![0u8; deflate.len() * 20];
    let out_len = decode_libdeflate(&deflate, &mut probe);
    drop(probe);
    let mut out = vec![0u8; out_len];

    let t0 = Instant::now();
    let mut total = 0usize;
    for _ in 0..iters {
        total += match which.as_str() {
            "libdeflate" => decode_libdeflate(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "pure" => decode_pure(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "consume_first" => decode_consume_first(&deflate, &mut out),
            other => panic!("unknown decoder {other:?} (or pure-rust-inflate feature off)"),
        };
    }
    let secs = t0.elapsed().as_secs_f64();
    let mbps = total as f64 / secs / 1e6;
    eprintln!(
        "decoder={which} iters={iters} out_len={out_len} total_bytes={total} \
         wall={secs:.3}s {mbps:.0} MB/s  (perf: instructions/{total} = ins/byte)",
    );
    // Print total decoded bytes on stdout for scripting (ins/byte denominator).
    println!("{total}");
}
