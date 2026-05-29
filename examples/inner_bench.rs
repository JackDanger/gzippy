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

// Window-absent BOOTSTRAP decoder (`deflate_block::Block`) — the marker-path
// inner loop that decodes ~31% of silesia (the speculative window-absent
// chunks) at ~175 MB/s in production. Decoding from offset 0 with no window
// exercises the SAME per-symbol decode (IsalLitLenCodePure litlen +
// get_distance_dynamic/apply_distance_extra + u16-ring writes) that runs on
// the marker path, isolating its instructions/byte from the parallel pipeline.
// This is the measurement gate for the bootstrap-unification lever.
#[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
fn decode_bootstrap(deflate: &[u8], out16: &mut Vec<u16>) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::deflate_block::Block;
    let mut block = Block::new();
    block.reset(None, None);
    let mut bits = Bits::new(deflate);
    out16.clear();
    loop {
        if block.read_header(&mut bits, false).is_err() {
            break;
        }
        while !block.eob() {
            if block.read(&mut bits, out16, usize::MAX).is_err() {
                return out16.len();
            }
        }
        if block.is_last_block() {
            break;
        }
    }
    out16.len()
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

    // Bootstrap (window-absent marker-path) mode uses a u16 output buffer.
    #[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
    if which == "bootstrap" {
        let mut out16: Vec<u16> = Vec::with_capacity(deflate.len() * 4);
        let t0 = Instant::now();
        let mut total = 0usize;
        let mut out_len = 0usize;
        for _ in 0..iters {
            out_len = decode_bootstrap(&deflate, &mut out16);
            total += out_len;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "decoder=bootstrap iters={iters} out_len={out_len} total_bytes={total} \
             wall={secs:.3}s {:.0} MB/s  (perf: instructions/{total} = ins/byte)",
            total as f64 / secs / 1e6,
        );
        println!("{total}");
        return;
    }

    // Over-allocate a generous output buffer ONCE (deflate ratio is well under
    // 8x). NO libdeflate sizing-probe: a probe decode contaminated every run
    // (~16% of a `consume_first` perf-stat was actually the one libdeflate
    // probe decode), poisoning the cross-decoder instruction ratio. The buffer
    // is faulted once and reused across iters — identical cost for every
    // decoder, so it cancels in the ratio.
    let mut out = vec![0u8; deflate.len() * 8];

    let t0 = Instant::now();
    let mut total = 0usize;
    let mut out_len = 0usize;
    for _ in 0..iters {
        out_len = match which.as_str() {
            "libdeflate" => decode_libdeflate(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "pure" => decode_pure(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "consume_first" => decode_consume_first(&deflate, &mut out),
            other => panic!("unknown decoder {other:?} (or pure-rust-inflate feature off)"),
        };
        total += out_len;
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
