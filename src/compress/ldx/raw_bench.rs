//! In-process raw-DEFLATE measurement harness for LDX versus libdeflate.
//!
//! This is deliberately an ignored unit test rather than a production command.
//! It keeps input, compressor state, and one common output buffer alive across iterations,
//! so `perf` sees compressor work rather than CLI startup, file I/O, gzip framing,
//! or allocator setup.  Run the compiled test binary directly for profiling, e.g.:
//!
//! ```text
//! GZIPPY_LDX_BENCH_INPUT=/path/to/input \
//! GZIPPY_LDX_BENCH_LEVEL=3 GZIPPY_LDX_BENCH_SIDE=rust \
//! cargo test --lib raw_deflate_in_process --release -- --ignored --nocapture
//! ```
//!
//! `SIDE` is `rust`, `c`, or `alternate` (the default).  Use `rust` or `c` for
//! `perf stat` and `perf record`; alternating intentionally perturbs the other
//! compressor's cache footprint and is only a directional wall-time check. `ITERS`
//! and `WARMUP` default to 100 and 3. The libdeflater crate uses libdeflate 1.25,
//! matching this repository's vendored API version. This harness measures aggregate
//! raw-compression work; phase attribution still requires sampled symbols or explicit
//! counters.

use super::compress::LdxCompressor;
use flate2::read::DeflateDecoder;
use libdeflater::{CompressionLvl, Compressor as LibdeflateCompressor};
use std::hint::black_box;
use std::io::Read;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Rust,
    C,
    Alternate,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be an unsigned integer"))
        })
        .unwrap_or(default)
}

fn side_from_env() -> Side {
    match std::env::var("GZIPPY_LDX_BENCH_SIDE")
        .unwrap_or_else(|_| "alternate".to_owned())
        .as_str()
    {
        "rust" => Side::Rust,
        "c" => Side::C,
        "alternate" => Side::Alternate,
        other => panic!("GZIPPY_LDX_BENCH_SIDE must be rust, c, or alternate; got {other:?}"),
    }
}

fn decode_exact(compressed: &[u8], input: &[u8]) {
    let mut decoded = Vec::with_capacity(input.len());
    DeflateDecoder::new(compressed)
        .read_to_end(&mut decoded)
        .expect("raw DEFLATE output must decode");
    assert_eq!(decoded, input, "raw DEFLATE output must round-trip");
}

fn run_rust(c: &mut LdxCompressor, input: &[u8], out: &mut [u8]) -> usize {
    let input = black_box(input);
    let out = black_box(out);
    let n = c.compress(input, input.len(), out);
    assert_ne!(n, 0, "LDX output buffer must be large enough");
    black_box(&out[..n]);
    black_box(n)
}

fn run_c(c: &mut LibdeflateCompressor, input: &[u8], out: &mut [u8]) -> usize {
    let input = black_box(input);
    let out = black_box(out);
    let n = c
        .deflate_compress(input, out)
        .expect("libdeflate output buffer must be large enough");
    black_box(&out[..n]);
    black_box(n)
}

/// Reusable raw-DEFLATE comparator for hardware-counter collection.
///
/// This test is ignored because it deliberately runs a real corpus repeatedly.
/// Its input path is required: keeping the corpus out of the source tree makes the
/// harness usable on trainer and AMD without silently changing its workload.
#[test]
#[ignore = "measurement harness; set GZIPPY_LDX_BENCH_INPUT and run explicitly"]
fn raw_deflate_in_process() {
    let input_path = std::env::var("GZIPPY_LDX_BENCH_INPUT")
        .expect("set GZIPPY_LDX_BENCH_INPUT to the corpus file to profile");
    let input = std::fs::read(&input_path).expect("read GZIPPY_LDX_BENCH_INPUT");
    assert!(
        !input.is_empty(),
        "the raw-DEFLATE harness needs a nonempty input"
    );

    let level = env_usize("GZIPPY_LDX_BENCH_LEVEL", 3) as u32;
    assert!(
        (1..=9).contains(&level),
        "LDX harness supports levels 1 through 9"
    );
    let iters = env_usize("GZIPPY_LDX_BENCH_ITERS", 100);
    assert!(iters != 0, "GZIPPY_LDX_BENCH_ITERS must be nonzero");
    let warmup = env_usize("GZIPPY_LDX_BENCH_WARMUP", 3);
    let side = side_from_env();

    let mut rust = LdxCompressor::new(level).expect("validated LDX level");
    let mut vendor = LibdeflateCompressor::new(
        CompressionLvl::new(level as i32).expect("validated libdeflate level"),
    );
    let ours_bound = input.len() + input.len().div_ceil(65_535) * 5 + 64;
    let bound = ours_bound.max(vendor.deflate_compress_bound(input.len()));
    // A common allocation removes output alignment/cache-color as a possible
    // explanation for a difference in the bitstream store path. The compressors
    // run serially and never read the prior output, so sharing is sound.
    let mut out = vec![0u8; bound];

    // Validate the actual buffers once.  Timed iterations use the same fully
    // initialized objects, but no decoder or allocation work.
    let rust_size = run_rust(&mut rust, &input, &mut out);
    decode_exact(&out[..rust_size], &input);
    let vendor_size = run_c(&mut vendor, &input, &mut out);
    decode_exact(&out[..vendor_size], &input);

    for i in 0..warmup {
        match side {
            Side::Rust => {
                black_box(run_rust(&mut rust, &input, &mut out));
            }
            Side::C => {
                black_box(run_c(&mut vendor, &input, &mut out));
            }
            Side::Alternate => {
                if i & 1 == 0 {
                    black_box(run_rust(&mut rust, &input, &mut out));
                    black_box(run_c(&mut vendor, &input, &mut out));
                } else {
                    black_box(run_c(&mut vendor, &input, &mut out));
                    black_box(run_rust(&mut rust, &input, &mut out));
                }
            }
        }
    }

    let mut rust_elapsed = Duration::ZERO;
    let mut vendor_elapsed = Duration::ZERO;
    for i in 0..iters {
        match side {
            Side::Rust => {
                let start = Instant::now();
                black_box(run_rust(&mut rust, &input, &mut out));
                rust_elapsed += start.elapsed();
            }
            Side::C => {
                let start = Instant::now();
                black_box(run_c(&mut vendor, &input, &mut out));
                vendor_elapsed += start.elapsed();
            }
            Side::Alternate => {
                if i & 1 == 0 {
                    let start = Instant::now();
                    black_box(run_rust(&mut rust, &input, &mut out));
                    rust_elapsed += start.elapsed();
                    let start = Instant::now();
                    black_box(run_c(&mut vendor, &input, &mut out));
                    vendor_elapsed += start.elapsed();
                } else {
                    let start = Instant::now();
                    black_box(run_c(&mut vendor, &input, &mut out));
                    vendor_elapsed += start.elapsed();
                    let start = Instant::now();
                    black_box(run_rust(&mut rust, &input, &mut out));
                    rust_elapsed += start.elapsed();
                }
            }
        }
    }

    let report = |name: &str, elapsed: Duration, calls: usize| {
        if calls != 0 {
            let ms = elapsed.as_secs_f64() * 1_000.0 / calls as f64;
            let mib_s =
                input.len() as f64 * calls as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
            eprintln!("{name}: {ms:.3} ms/call, {mib_s:.1} MiB/s");
        }
    };
    match side {
        Side::Rust => report("rust", rust_elapsed, iters),
        Side::C => report("libdeflate", vendor_elapsed, iters),
        Side::Alternate => {
            report("rust", rust_elapsed, iters);
            report("libdeflate", vendor_elapsed, iters);
            eprintln!(
                "ratio rust/c={:.4} (raw sizes rust={rust_size}, c={vendor_size})",
                rust_elapsed.as_secs_f64() / vendor_elapsed.as_secs_f64(),
            );
        }
    }
}
