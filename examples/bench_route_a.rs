//! Bake-off bench for Route A (streaming-libdeflate-rs pure-Rust port)
//! vs the two C oracles (libdeflate-sys + zlib-ng-sys) on silesia.
//!
//! Build:
//!   cargo build --release --features streaming-libdeflate-rs --example bench_route_a
//!
//! Run:
//!   ./target/release/examples/bench_route_a benchmark_data/silesia-large.gz
//!
//! Prints per-trial throughput and final medians. Compare the
//! "Route A vs libdeflate-C" gap to plans/unified-decoder.md §4.5
//! decision matrix:
//!   - within 5pp → Route A ships; project ends
//!   - 5-20pp → Route A + Phase 2 (bit-reader rewrite)
//!   - ≥20pp → confirms language-level codegen gap; Phase 3 (dynasm) path

use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_libdeflate_c(gz: &[u8], expected_size: usize, trials: usize) -> Vec<f64> {
    use std::ptr::NonNull;
    let mut tps = Vec::with_capacity(trials);
    for _ in 0..trials {
        let decompressor = unsafe { libdeflate_sys::libdeflate_alloc_decompressor() };
        let decompressor = NonNull::new(decompressor).unwrap();
        let mut out = vec![0u8; expected_size];
        let mut out_n = 0usize;
        let mut in_n = 0usize;

        let t0 = Instant::now();
        let r = unsafe {
            libdeflate_sys::libdeflate_gzip_decompress_ex(
                decompressor.as_ptr(),
                gz.as_ptr() as *const _,
                gz.len(),
                out.as_mut_ptr() as *mut _,
                out.len(),
                &mut in_n,
                &mut out_n,
            )
        };
        let dt = t0.elapsed().as_secs_f64();
        assert_eq!(r, 0);
        unsafe { libdeflate_sys::libdeflate_free_decompressor(decompressor.as_ptr()) };

        let mbps = (out_n as f64) / 1_000_000.0 / dt;
        tps.push(mbps);
    }
    tps
}

fn bench_zlibng_c(gz: &[u8], expected_size: usize, trials: usize) -> Vec<f64> {
    use libz_ng_sys as zng;
    use std::mem;
    use std::ptr;

    let mut tps = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut strm: zng::z_stream = unsafe {
            let mut m = mem::MaybeUninit::<zng::z_stream>::uninit();
            ptr::write_bytes(m.as_mut_ptr(), 0, 1);
            m.assume_init()
        };
        unsafe {
            zng::inflateInit2_(
                &mut strm,
                31,
                ptr::null(),
                mem::size_of::<zng::z_stream>() as i32,
            );
        }
        strm.next_in = gz.as_ptr() as *mut _;
        strm.avail_in = gz.len() as u32;
        let mut out = vec![0u8; expected_size];
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len() as u32;

        let t0 = Instant::now();
        loop {
            let r = unsafe { zng::inflate(&mut strm, zng::Z_NO_FLUSH) };
            if r == zng::Z_STREAM_END {
                break;
            }
            if r != zng::Z_OK && r != zng::Z_BUF_ERROR {
                break;
            }
            if strm.avail_in == 0 && strm.avail_out == out.len() as u32 {
                break;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let written = out.len() - strm.avail_out as usize;
        unsafe { zng::inflateEnd(&mut strm) };

        let mbps = (written as f64) / 1_000_000.0 / dt;
        tps.push(mbps);
    }
    tps
}

// `backends` is private inside gzippy. For the example bench we inline a
// copy of the Route A wrapper here (functionally identical to
// `src/backends/libdeflate_rs.rs::decompress_gzip`). Keeps the lib
// surface clean and the example self-contained.
#[cfg(feature = "streaming-libdeflate-rs")]
fn route_a_decompress(gz: &[u8]) -> Vec<u8> {
    use streaming_libdeflate_rs::{
        decompress_gzip::libdeflate_gzip_decompress,
        libdeflate_alloc_decode_tables,
        streams::{
            deflate_chunked_buffer_input::DeflateChunkedBufferInput,
            deflate_chunked_buffer_output::DeflateChunkedBufferOutput,
        },
        DeflateInput,
    };
    let mut cursor = 0usize;
    let mut input_stream = DeflateChunkedBufferInput::new(
        |buf: &mut [u8]| {
            let remaining = gz.len().saturating_sub(cursor);
            let take = remaining.min(buf.len());
            buf[..take].copy_from_slice(&gz[cursor..cursor + take]);
            cursor += take;
            take
        },
        64 * 1024,
    );
    let collected: std::sync::Arc<std::sync::Mutex<Vec<u8>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let cc = std::sync::Arc::clone(&collected);
    let mut output_stream = DeflateChunkedBufferOutput::new(
        move |chunk: &[u8]| -> Result<(), ()> {
            cc.lock().unwrap().extend_from_slice(chunk);
            Ok(())
        },
        64 * 1024,
    );
    let mut decoder = libdeflate_alloc_decode_tables();
    while {
        input_stream.ensure_overread_length();
        input_stream.has_valid_bytes_slow()
    } {
        libdeflate_gzip_decompress(&mut decoder, &mut input_stream, &mut output_stream).unwrap();
    }
    drop(output_stream);
    std::sync::Arc::try_unwrap(collected)
        .unwrap()
        .into_inner()
        .unwrap()
}

#[cfg(feature = "streaming-libdeflate-rs")]
fn bench_route_a(gz: &[u8], trials: usize) -> Vec<f64> {
    let mut tps = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let out = route_a_decompress(gz);
        let dt = t0.elapsed().as_secs_f64();
        let mbps = (out.len() as f64) / 1_000_000.0 / dt;
        tps.push(mbps);
    }
    tps
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gz_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "benchmark_data/silesia-large.gz".to_string());
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let gz = std::fs::read(&gz_path).expect("read gz file");
    let expected_size = {
        let tail = &gz[gz.len() - 4..];
        u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize
    };

    eprintln!(
        "=== Bake-off bench: silesia-large.gz ({} bytes input → {} bytes output, {} trials) ===",
        gz.len(),
        expected_size,
        trials
    );

    eprintln!("\n--- libdeflate (C, oracle 1) ---");
    let lib_tps = bench_libdeflate_c(&gz, expected_size, trials);
    for (i, t) in lib_tps.iter().enumerate() {
        eprintln!("  trial {}: {:.0} MB/s", i + 1, t);
    }
    let lib_med = median(lib_tps);
    eprintln!("  median: {:.0} MB/s", lib_med);

    eprintln!("\n--- zlib-ng (C, oracle 2) ---");
    let zng_tps = bench_zlibng_c(&gz, expected_size, trials);
    for (i, t) in zng_tps.iter().enumerate() {
        eprintln!("  trial {}: {:.0} MB/s", i + 1, t);
    }
    let zng_med = median(zng_tps);
    eprintln!("  median: {:.0} MB/s", zng_med);

    #[cfg(feature = "streaming-libdeflate-rs")]
    {
        eprintln!("\n--- Route A: streaming-libdeflate-rs (pure-Rust port) ---");
        let rt_a = bench_route_a(&gz, trials);
        for (i, t) in rt_a.iter().enumerate() {
            eprintln!("  trial {}: {:.0} MB/s", i + 1, t);
        }
        let rt_a_med = median(rt_a);
        eprintln!("  median: {:.0} MB/s", rt_a_med);

        let gap_vs_lib = (rt_a_med - lib_med) / lib_med * 100.0;
        let gap_vs_zng = (rt_a_med - zng_med) / zng_med * 100.0;
        eprintln!(
            "\n=== VERDICT ===\nRoute A vs libdeflate-C: {:+.1}% ({})\nRoute A vs zlib-ng-C:   {:+.1}% ({})",
            gap_vs_lib,
            if gap_vs_lib.abs() <= 5.0 {
                "WITHIN 5pp — §4.5 outcome 1: VENDOR + SHIP, project ends"
            } else if gap_vs_lib >= -20.0 {
                "5-20pp behind — §4.5 outcome 2: vendor + Phase 2"
            } else {
                "≥20pp behind — §4.5 outcome 3: confirms codegen gap"
            },
            gap_vs_zng,
            if gap_vs_zng.abs() <= 5.0 {
                "WITHIN 5pp"
            } else {
                "outside 5pp"
            },
        );
    }
}
