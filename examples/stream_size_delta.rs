//! What does streaming COST in output size, on the real corpus?
//!
//! The synthetic identity test found the streaming encoder emits +1 byte on a
//! chunk-boundary case at L1. One byte is not automatically negligible in this
//! campaign: there is an open class of sub-1% per-label size cells, and a cell
//! sitting at an exact tie flips on a single byte. So measure the real
//! distribution rather than reasoning about whether one byte matters.
//!
//! Usage: cargo run --release --example stream_size_delta -- <file>...

use gzippy::compress::deflate::{
    encode_gzip_reader_to_writer_chunked, encode_gzip_slack_padded_to_vec, INPLACE_TAIL_PAD,
    STREAM_CHUNK,
};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Leading `--chunk=N` (multiples of 65535) and `--levels=a,b,c` are
    // measurement seams. Chunk defaults to the shipped STREAM_CHUNK so the
    // default run measures what production actually does.
    let mut chunk = STREAM_CHUNK;
    let mut levels: Vec<u32> = (0..=9).collect();
    while let Some(a) = args.first().cloned() {
        if let Some(v) = a.strip_prefix("--chunk=") {
            chunk = v.parse().expect("--chunk=<bytes>");
            args.remove(0);
        } else if let Some(v) = a.strip_prefix("--levels=") {
            levels = v
                .split(',')
                .map(|x| x.parse().expect("--levels=0,1,2"))
                .collect();
            args.remove(0);
        } else {
            break;
        }
    }
    eprintln!(
        "chunk = {chunk} bytes ({:.2} MiB)",
        chunk as f64 / 1048576.0
    );
    let files: Vec<String> = args;
    if files.is_empty() {
        eprintln!("usage: stream_size_delta <file>...");
        std::process::exit(2);
    }

    println!(
        "{:<20} {:>3} {:>12} {:>12} {:>9} {:>10}",
        "file", "L", "whole", "streamed", "delta", "pct"
    );

    let mut worst_pct = 0.0f64;
    let mut worst_desc = String::new();
    let mut total_delta: i64 = 0;
    let mut identical = 0usize;
    let mut cells = 0usize;

    for f in &files {
        let data = match std::fs::read(f) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {f}: {e}");
                continue;
            }
        };
        let name = std::path::Path::new(f)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| f.clone());

        let mut padded = Vec::with_capacity(data.len() + INPLACE_TAIL_PAD);
        padded.extend_from_slice(&data);
        padded.resize(data.len() + INPLACE_TAIL_PAD, 0);

        for &level in &levels {
            let w = encode_gzip_slack_padded_to_vec(&padded, data.len(), level).len();
            let mut out = Vec::new();
            let mut src = data.as_slice();
            encode_gzip_reader_to_writer_chunked(&mut src, &mut out, level, chunk).expect("stream");
            let s = out.len();

            let delta = s as i64 - w as i64;
            let pct = 100.0 * delta as f64 / w as f64;
            cells += 1;
            total_delta += delta;
            if delta == 0 {
                identical += 1;
            }
            if pct > worst_pct {
                worst_pct = pct;
                worst_desc = format!("{name} L{level} {delta:+} bytes");
            }
            if delta != 0 {
                println!("{name:<20} {level:>3} {w:>12} {s:>12} {delta:>+9} {pct:>9.5}%");
            }
        }
    }

    println!();
    println!(
        "cells: {cells}   byte-identical: {identical}   differing: {}",
        cells - identical
    );
    println!("total delta across all cells: {total_delta:+} bytes");
    if worst_desc.is_empty() {
        println!("worst regression: NONE — streaming never emitted a larger stream");
    } else {
        println!("worst regression: {worst_desc} ({worst_pct:.5}%)");
    }
}
