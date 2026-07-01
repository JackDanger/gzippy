//! THROWAWAY T1 sizing harness — NOT production.
//!
//! Decodes a single-member gzip file with gzippy's pure-Rust inflate
//! (`inflate_consume_first`) over the WHOLE deflate stream: NO chunk pipeline,
//! NO marker ring, NO speculation. This is the "native serial single-shot"
//! arm used to size the recoverable ParallelSM-at-T1 serialization tax (Q1).
//!
//! Usage:
//!   serial_sizing <file.gz> [reps]
//!     -> times `reps` (default 9) decodes, prints "ms: <t1> <t2> ..." to stderr
//!        and "min_ms=<x> median_ms=<y>" summary. Writes nothing to stdout.
//!   VERIFY=1 serial_sizing <file.gz>
//!     -> decodes once and writes the decoded bytes to stdout (pipe | sha256sum).
//!
//! Build: cargo build --release --no-default-features --features pure-rust-inflate
//!        --example serial_sizing

use std::io::Write;
use std::time::Instant;

use gzippy::decompress::inflate::consume_first_decode::inflate_consume_first;

fn parse_gzip_header_size(data: &[u8]) -> Option<usize> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }
    let flags = data[3];
    let mut pos = 10;
    if flags & 0x04 != 0 {
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x02 != 0 {
        pos += 2;
    }
    Some(pos)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: serial_sizing <file.gz> [reps]");
        std::process::exit(2);
    }
    let path = &args[1];
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);

    let data = std::fs::read(path).expect("read input file");
    let hdr = parse_gzip_header_size(&data).expect("parse gzip header");
    let isize_field = {
        let b = &data[data.len() - 4..];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
    };
    // Uninitialized buffer (NOT vec![0u8;n]) so we don't pay a zeroing pass the
    // production decode path never pays — that double-write skews the process wall.
    // inflate writes sequentially and returns n; we only read out[..n].
    let mut out: Vec<u8> = Vec::with_capacity(isize_field);
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(isize_field);
    }
    let deflate = &data[hdr..];

    let verify = std::env::var("VERIFY").map(|v| v == "1").unwrap_or(false);

    if verify {
        let n = inflate_consume_first(deflate, &mut out).expect("inflate");
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        lock.write_all(&out[..n]).expect("write stdout");
        return;
    }

    let mut times = Vec::with_capacity(reps);
    let mut last_n = 0usize;
    for _ in 0..reps {
        let t0 = Instant::now();
        let n = inflate_consume_first(deflate, &mut out).expect("inflate");
        let dt = t0.elapsed();
        last_n = n;
        times.push(dt.as_secs_f64() * 1000.0);
    }
    assert_eq!(last_n, isize_field, "decoded len != ISIZE");

    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let median = sorted[sorted.len() / 2];
    eprint!("ms:");
    for t in &times {
        eprint!(" {:.2}", t);
    }
    eprintln!();
    eprintln!(
        "decoded={} bytes min_ms={:.2} median_ms={:.2}",
        last_n, min, median
    );
}
