//! Compare patched ISA-L vs pure-Rust resumable inflate throughput.
//!
//! Run (silesia corpus):
//! ```text
//! cargo bench --release --features isal-compression -- \
//!   --bench inflate_isal_vs_pure_rust -- --nocapture
//! ```

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    not(feature = "pure-rust-inflate")
))]
mod bench {
    use gzippy::decompress::inflate::resumable::ResumableInflate2;
    use gzippy::decompress::parallel::inflate_wrapper::StreamingInflateWrapper;
    use std::time::Instant;

    fn bench_throughput<F: FnMut() -> usize>(mut f: F) -> f64 {
        let mut best = 0.0f64;
        for _ in 0..3 {
            let start = Instant::now();
            let nbytes = f();
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            best = best.max(nbytes as f64 / secs / 1e6);
        }
        best
    }

    fn load_silesia_deflate() -> Option<Vec<u8>> {
        let path = std::path::Path::new("benchmark_data/silesia-gzip.tar.gz");
        let data = std::fs::read(path).ok()?;
        let (_hdr, header) = gzippy::decompress::parallel::gzip_format::read_header(&data).ok()?;
        Some(data[header..data.len().saturating_sub(8)].to_vec())
    }

    fn synthetic_deflate(uncompressed_len: usize) -> Vec<u8> {
        use std::io::Write;
        let mut raw = vec![0u8; uncompressed_len];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761) as u8;
        }
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(&raw).unwrap();
        enc.finish().unwrap()
    }

    pub fn run() {
        let deflate = if let Some(gz) = load_silesia_deflate() {
            gz
        } else {
            eprintln!("benchmark_data/silesia-gzip.tar.gz missing — using 4 MiB synthetic deflate");
            synthetic_deflate(4 * 1024 * 1024)
        };

        let isal_mbps = bench_throughput(|| {
            let mut w = StreamingInflateWrapper::new(&deflate, 0).expect("isal init");
            w.set_window(&[]).expect("window");
            let mut out = vec![0u8; deflate.len() * 16];
            let mut total = 0usize;
            loop {
                let r = w.read_stream(&mut out[total..]).expect("isal read");
                total += r.bytes_written;
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
            total
        });

        let rust_mbps = bench_throughput(|| {
            let mut d = ResumableInflate2::with_until_bits(&deflate, 0, deflate.len() * 8)
                .expect("rust init");
            d.set_window(&[]).expect("window");
            let mut out = vec![0u8; deflate.len() * 16];
            let mut total = 0usize;
            loop {
                let r = d.read_stream(&mut out[total..]).expect("rust read");
                total += r.bytes_written;
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
            total
        });

        let ratio = isal_mbps / rust_mbps.max(1e-9);
        println!("B4 inflate bench (wrapper API, full stream):");
        println!("  ISA-L:      {isal_mbps:.0} MB/s");
        println!("  pure-Rust:  {rust_mbps:.0} MB/s");
        println!("  ISA-L/Rust: {ratio:.2}x (Tier-1 gate: <= 1.5x)");
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    not(feature = "pure-rust-inflate")
))]
fn main() {
    bench::run();
}

#[cfg(not(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    not(feature = "pure-rust-inflate")
)))]
fn main() {
    eprintln!(
        "inflate_isal_vs_pure_rust: requires x86_64 + isal-compression without pure-rust-inflate"
    );
}
