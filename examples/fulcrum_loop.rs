//! FULCRUM in-process decode-loop harness — Coz statistical-power driver.
//!
//! gzippy decodes silesia-large in ~0.4 s. That is far TOO SHORT for Coz:
//! a causal-profiling experiment needs many epochs (each ~tens of ms of
//! virtual-speedup at one selected line) AND many visits to the throughput
//! progress point. A single 0.4 s process yields ~1 epoch — useless, and
//! process churn (mmap, pool spawn, page faults) swamps the signal.
//!
//! This harness loops `decompress_to_writer_with_threads` `--iters` times
//! IN ONE PROCESS. The pure-Rust parallel decode path's `chunk_emitted()`
//! progress point therefore fires `iters * n_chunks` times under a single
//! `coz run`, with the pool warm and startup amortized — exactly the
//! steady-state Coz wants. Coz further appends across processes into one
//! `profile.coz`, so the `fulcrum coz` driver runs THIS binary in an outer
//! loop too; both layers compound the sample count.
//!
//! Output bytes go to a byte-counting sink (`CountSink`) — we exercise the
//! full decode + every in-order emit (so the progress point and all four
//! region scopes run), but never touch real I/O, keeping the measured wall
//! on the decoder, not the writer. The decoded length is checked stable
//! across iterations as a cheap correctness guard.
//!
//! Usage (on the box, under coz):
//!   coz run --- ./target/fulcrum/examples/fulcrum_loop \
//!       benchmark_data/silesia-large.gz --iters 20 --threads 8
//!
//! Build:
//!   cargo build --profile fulcrum --no-default-features \
//!     --features pure-rust-inflate,fulcrum --example fulcrum_loop

use std::io::{self, Write};

/// A `Write` sink that counts bytes and discards them. Zero allocation,
/// so the decode wall isn't polluted by writer-side work.
struct CountSink(u64);

impl Write for CountSink {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0 += buf.len() as u64;
        Ok(())
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    if let Some(p) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(p + 1) {
            if let Ok(parsed) = v.parse::<T>() {
                return parsed;
            }
        }
    }
    default
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: fulcrum_loop <input.gz> [--iters N] [--threads T] [--warmup W]");
            std::process::exit(2);
        });
    let iters: usize = parse_arg(&args, "--iters", 20usize);
    let threads: usize = parse_arg(
        &args,
        "--threads",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8),
    );
    let warmup: usize = parse_arg(&args, "--warmup", 1usize);

    let data = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("fulcrum_loop: cannot read {path}: {e}");
        std::process::exit(2);
    });

    eprintln!(
        "fulcrum_loop: input={path} bytes={} iters={iters} threads={threads} warmup={warmup}",
        data.len()
    );

    // Warm-up iterations: fault in the pool / caches so they don't taint
    // the first measured epoch. Not counted toward correctness.
    for _ in 0..warmup {
        let mut sink = CountSink(0);
        let _ = gzippy::decompress_to_writer_with_threads(&data, &mut sink, threads);
    }

    let mut first_len: Option<u64> = None;
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        let mut sink = CountSink(0);
        match gzippy::decompress_to_writer_with_threads(&data, &mut sink, threads) {
            Ok(n) => {
                // Cheap correctness guard: every iteration must produce the
                // same decoded length. A drift would mean the in-process
                // loop corrupted shared state — fail loud rather than feed
                // Coz garbage.
                debug_assert_eq!(n, sink.0, "returned len != bytes written");
                match first_len {
                    None => first_len = Some(sink.0),
                    Some(l) if l != sink.0 => {
                        eprintln!("fulcrum_loop: FATAL iter {i} decoded {} != {l}", sink.0);
                        std::process::exit(1);
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("fulcrum_loop: decode error on iter {i}: {e}");
                std::process::exit(1);
            }
        }
    }
    let dt = t0.elapsed();
    let total_bytes = first_len.unwrap_or(0) * iters as u64;
    let mbps = total_bytes as f64 / dt.as_secs_f64() / 1e6;
    eprintln!(
        "fulcrum_loop: {iters} iters in {:.3}s  decoded_len={}  aggregate {:.0} MB/s",
        dt.as_secs_f64(),
        first_len.unwrap_or(0),
        mbps
    );
    // Flush any pooled-thread trace tails (no-op unless GZIPPY_TIMELINE set).
    let _ = io::stdout().flush();
}
