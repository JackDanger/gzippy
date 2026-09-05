//! Print the parallel chunk grid the encoder will actually use.
//!
//! Any arithmetic in docs/plan-2026-09-one-encoder.md (fragments per file,
//! expected per-chunk seam cost, replay volume) is checkable in one command
//! instead of inferred. The encoder's own pure function is the single source
//! of truth — `perf_shape`'s pin suite uses the same one — so this is the
//! encoder answering, not a re-implementation.
//!
//!   cargo run --release --example chunkgrid -- FILE FILE2 ...
//!   cargo run --release --example chunkgrid -- 268435456 L=1,6 T=1,4
//!
//! Your -p flag changes the answer at the thread axis; pass T= for the exact
//! thread counts you plan to run.
use gzippy::compress::pipelined::pipelined_block_size;

fn main() {
    let mut levels: Vec<u32> = (1..=9).collect();
    let mut threads: Vec<usize> = vec![1, 2, 4, 8, 16];
    let mut sizes: Vec<usize> = Vec::new();

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("L=") {
            levels = v.split(',').filter_map(|x| x.parse().ok()).collect();
        } else if let Some(v) = arg.strip_prefix("T=") {
            threads = v.split(',').filter_map(|x| x.parse().ok()).collect();
        } else if let Ok(n) = arg.parse::<usize>() {
            sizes.push(n);
        } else {
            match std::fs::metadata(&arg) {
                Ok(m) => sizes.push(m.len() as usize),
                Err(e) => {
                    eprintln!("chunkgrid: cannot stat {}: {}", arg, e);
                    std::process::exit(2);
                }
            }
        }
    }
    if sizes.is_empty() {
        eprintln!("usage: chunkgrid [--] [FILES | byte-sizes ...] [L=1,6] [T=1,4]");
        eprintln!("prints pipelined_block_size(input_len, threads, level) and the chunk count");
        std::process::exit(2);
    }

    println!(
        "{:>12} {:>4} {:>4} {:>12} {:>8}",
        "input_bytes", "L", "T", "chunk_bytes", "chunks"
    );
    for &size in &sizes {
        for &level in &levels {
            for &t in &threads {
                let chunk = pipelined_block_size(size, t, level);
                let chunks = size.div_ceil(chunk);
                println!(
                    "{:>12} {:>4} {:>4} {:>12} {:>8}",
                    size, level, t, chunk, chunks
                );
            }
        }
    }
}
