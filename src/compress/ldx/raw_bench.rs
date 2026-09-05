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
//! and `WARMUP` default to 100 and 3. Set `COLD=1` to include compressor
//! construction in each call (useful for small files and concurrent job setup).
//! `VALIDATE=0` skips the one-time dual
//! compression/decode check for callgrind attribution; leave it enabled for all
//! ordinary measurements. The libdeflater crate uses libdeflate 1.25, matching this
//! repository's vendored API version. This harness measures aggregate raw-compression
//! work; phase attribution still requires sampled symbols or explicit counters.

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

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).as_deref() {
        Ok("1" | "true") => true,
        Ok("0" | "false") => false,
        Ok(other) => panic!("{name} must be 1, 0, true, or false; got {other:?}"),
        Err(_) => default,
    }
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

/// Run a measurement call with either a reused compressor, or a fresh one when
/// cold-start allocation and initialization are intentionally in scope.
fn run_rust_measurement(
    c: Option<&mut LdxCompressor>,
    level: u32,
    input: &[u8],
    out: &mut [u8],
) -> usize {
    match c {
        Some(c) => run_rust(c, input, out),
        None => {
            let mut c = LdxCompressor::new(level).expect("validated LDX level");
            run_rust(&mut c, input, out)
        }
    }
}

fn run_c_measurement(
    c: Option<&mut LibdeflateCompressor>,
    level: u32,
    input: &[u8],
    out: &mut [u8],
) -> usize {
    match c {
        Some(c) => run_c(c, input, out),
        None => {
            let mut c = LibdeflateCompressor::new(
                CompressionLvl::new(level as i32).expect("validated libdeflate level"),
            );
            run_c(&mut c, input, out)
        }
    }
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
    assert!(level <= 9, "LDX harness supports levels 0 through 9");
    let iters = env_usize("GZIPPY_LDX_BENCH_ITERS", 100);
    assert!(iters != 0, "GZIPPY_LDX_BENCH_ITERS must be nonzero");
    let warmup = env_usize("GZIPPY_LDX_BENCH_WARMUP", 3);
    let validate = env_bool("GZIPPY_LDX_BENCH_VALIDATE", true);
    let cold = env_bool("GZIPPY_LDX_BENCH_COLD", false);
    let side = side_from_env();

    let mut rust = (!cold).then(|| LdxCompressor::new(level).expect("validated LDX level"));
    let mut vendor = (!cold).then(|| {
        LibdeflateCompressor::new(
            CompressionLvl::new(level as i32).expect("validated libdeflate level"),
        )
    });
    let ours_bound = input.len() + input.len().div_ceil(65_535) * 5 + 64;
    let bound = ours_bound.max(
        vendor
            .as_mut()
            .map_or(ours_bound, |c| c.deflate_compress_bound(input.len())),
    );
    // A common allocation removes output alignment/cache-color as a possible
    // explanation for a difference in the bitstream store path. The compressors
    // run serially and never read the prior output, so sharing is sound.
    let mut out = vec![0u8; bound];

    // Validate the actual buffers once. Timed iterations use the same fully
    // initialized objects, but no decoder or allocation work. Callgrind needs
    // the ability to omit this setup entirely so it can count one selected
    // compressor call without a decoder or the opposite compressor mixed in.
    let validated_sizes = if validate {
        let rust_size = run_rust_measurement(rust.as_mut(), level, &input, &mut out);
        decode_exact(&out[..rust_size], &input);
        let vendor_size = run_c_measurement(vendor.as_mut(), level, &input, &mut out);
        decode_exact(&out[..vendor_size], &input);
        Some((rust_size, vendor_size))
    } else {
        None
    };

    for i in 0..warmup {
        match side {
            Side::Rust => {
                black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
            }
            Side::C => {
                black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
            }
            Side::Alternate => {
                if i & 1 == 0 {
                    black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
                    black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
                } else {
                    black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
                    black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
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
                black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
                rust_elapsed += start.elapsed();
            }
            Side::C => {
                let start = Instant::now();
                black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
                vendor_elapsed += start.elapsed();
            }
            Side::Alternate => {
                if i & 1 == 0 {
                    let start = Instant::now();
                    black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
                    rust_elapsed += start.elapsed();
                    let start = Instant::now();
                    black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
                    vendor_elapsed += start.elapsed();
                } else {
                    let start = Instant::now();
                    black_box(run_c_measurement(vendor.as_mut(), level, &input, &mut out));
                    vendor_elapsed += start.elapsed();
                    let start = Instant::now();
                    black_box(run_rust_measurement(rust.as_mut(), level, &input, &mut out));
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
            let sizes = validated_sizes
                .map(|(rust, c)| format!("raw sizes rust={rust}, c={c}"))
                .unwrap_or_else(|| "round-trip validation disabled".to_owned());
            eprintln!(
                "ratio rust/c={:.4} ({sizes})",
                rust_elapsed.as_secs_f64() / vendor_elapsed.as_secs_f64(),
            );
        }
    }
}
