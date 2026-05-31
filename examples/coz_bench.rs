//! Coz steady-state harness for the parallel single-member decode.
//!
//! Coz needs many epochs and many progress-point visits to build stable
//! virtual-speedup curves; a single ~0.3s decode yields too few. This loops
//! the production `decompress_parallel` over one input N times so the
//! `chunk_emitted` progress marker (and the `marker_bootstrap` / `clean_isal`
//! latency scopes wired into the decode) are visited thousands of times.
//!
//! Build + run under Coz (Linux):
//!   cargo build --release --example coz_bench --features pure-rust-inflate,coz
//!   coz run --- ./target/release/examples/coz_bench <file.gz> <threads> <iters>
//! then:
//!   fulcrum rank /tmp/anything profile.coz --config fulcrum-region-config.json
//!
//! A coz-enabled binary runs normally outside `coz run` (the runtime is
//! dlsym-resolved), so this doubles as a plain throughput loop.

use std::io;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "benchmark_data/silesia-large.gz".to_string());
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("GZIPPY_COZ_LOOP")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(60);

    let data = std::fs::read(&path).expect("read input .gz");
    eprintln!(
        "coz_bench: {} ({} bytes), threads={threads}, iters={iters}",
        path,
        data.len()
    );

    let mut total: u64 = 0;
    for i in 0..iters {
        // Discard output (io::sink) so we measure decode, not the sink tax.
        let mut sink = io::sink();
        match gzippy::decompress::parallel::single_member::decompress_parallel(
            &data, &mut sink, threads,
        ) {
            Ok(n) => total = total.wrapping_add(n),
            Err(e) => {
                eprintln!("iter {i}: decode error: {e:?}");
                std::process::exit(1);
            }
        }
    }
    // Touch `total` so the loop can't be optimized away.
    eprintln!("coz_bench: done, summed decoded bytes = {total}");
}
