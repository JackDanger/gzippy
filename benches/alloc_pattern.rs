//! Framework Step 4 — `alloc_pattern` microbench
//!
//! Purpose: reproduce gzippy's exact memory shape (16 workers × 12 MiB
//! target × BTYPE-mixed writes + consumer read-side sweep) so we can
//! measure the impact of allocator + chunk-shape changes in 30s
//! instead of a 20-trial neurotic A/B.
//!
//! Mirrors the empirically-attributed write path: the marker bootstrap
//! + chunk-data extend_from_slice pattern (see
//!   `docs/perf/2026-05-28-framework-step2-pebs-attribution.md`).
//!
//! Test matrix (advisor-locked, all six combinations run by default):
//!
//! Allocator  × Chunk shape        × Read sweep
//! ----------------------------------------------------------
//! glibc      × single 12 MiB Vec  × ON   (gzippy baseline today)
//! glibc      × many 128 KiB Boxes × ON   (chunk-shape only)
//! rpmalloc   × single 12 MiB Vec  × ON   (allocator only)
//! rpmalloc   × many 128 KiB Boxes × ON   (combined — rapidgzip target)
//! glibc      × single 12 MiB Vec  × OFF  (write-only baseline)
//! rpmalloc   × many 128 KiB Boxes × OFF  (write-only target)
//!
//! Plus a separate page-fault reporter (`pf` mode) that runs each
//! variant once and emits raw counts via /proc/self/stat.
//!
//! Run:
//! ```text
//! cargo bench --bench alloc_pattern -- --nocapture
//! cargo bench --bench alloc_pattern --features global-rpmalloc -- --nocapture
//! GZIPPY_ALLOC_PF=1 cargo bench --bench alloc_pattern -- --nocapture
//! ```
//!
//! Pass gate (advisor-locked): a variant must show ≥10% page-fault
//! reduction AND ≥5% cycle reduction vs gzippy-baseline (glibc +
//! single-Vec) before promoting to the production A/B in Step 5.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

// Shape matches gzippy parallel-SM on silesia-gzip9.gz at T=16
const N_WORKERS: usize = 16;
const TARGET_BYTES_PER_WORKER: usize = 12 * 1024 * 1024; // 12 MiB
const CHUNK_SIZE_SMALL: usize = 128 * 1024; // 128 KiB — matches rapidgzip's ALLOCATION_CHUNK_SIZE
const BTYPE_MIXED_BLOCK_SIZE: usize = 64 * 1024; // typical deflate block size

#[derive(Clone, Copy, Debug)]
enum ChunkShape {
    SingleVec,
    ManyBoxes,
}

#[derive(Clone, Copy, Debug)]
enum ReadSweep {
    Off,
    On,
}

/// Worker thread: write TARGET_BYTES_PER_WORKER bytes into a buffer
/// in the gzippy shape (interspersed btype01 + dynamic write patterns)
/// then optionally sweep-read the buffer to simulate the consumer's
/// write_all / CRC pass.
fn worker(shape: ChunkShape, read_sweep: ReadSweep, ticker: &AtomicU64) {
    match shape {
        ChunkShape::SingleVec => {
            let mut buf: Vec<u8> = Vec::with_capacity(TARGET_BYTES_PER_WORKER);
            // BTYPE-mixed writes: extend_from_slice in ~64 KiB blocks
            // with occasional 512-byte short-tail writes (BTYPE=01)
            let block = vec![0xABu8; BTYPE_MIXED_BLOCK_SIZE];
            let short_tail = vec![0xCDu8; 512];
            let mut written = 0;
            while written + BTYPE_MIXED_BLOCK_SIZE + 512 <= TARGET_BYTES_PER_WORKER {
                buf.extend_from_slice(&block);
                buf.extend_from_slice(&short_tail);
                written += BTYPE_MIXED_BLOCK_SIZE + 512;
            }
            // Fill remainder
            if written < TARGET_BYTES_PER_WORKER {
                buf.resize(TARGET_BYTES_PER_WORKER, 0xEFu8);
            }
            // Consumer read sweep
            if matches!(read_sweep, ReadSweep::On) {
                let mut sum = 0u64;
                for chunk in buf.chunks(4096) {
                    sum = sum.wrapping_add(chunk.iter().map(|&b| b as u64).sum::<u64>());
                }
                ticker.fetch_add(sum, Ordering::Relaxed);
            } else {
                ticker.fetch_add(buf[0] as u64, Ordering::Relaxed);
            }
            drop(buf);
        }
        ChunkShape::ManyBoxes => {
            // Many fixed-size 128 KiB boxes — rapidgzip's pattern
            let mut chunks: Vec<Box<[u8; CHUNK_SIZE_SMALL]>> = Vec::new();
            let mut written = 0;
            while written + CHUNK_SIZE_SMALL <= TARGET_BYTES_PER_WORKER {
                let mut chunk: Box<[u8; CHUNK_SIZE_SMALL]> = Box::new([0u8; CHUNK_SIZE_SMALL]);
                // Mixed BTYPE write pattern within the chunk
                let mut pos = 0;
                while pos + 512 < CHUNK_SIZE_SMALL {
                    let n = (BTYPE_MIXED_BLOCK_SIZE).min(CHUNK_SIZE_SMALL - pos);
                    chunk[pos..pos + n].fill(0xABu8);
                    pos += n;
                    if pos + 512 < CHUNK_SIZE_SMALL {
                        chunk[pos..pos + 512].fill(0xCDu8);
                        pos += 512;
                    }
                }
                chunks.push(chunk);
                written += CHUNK_SIZE_SMALL;
            }
            // Consumer read sweep across chunks
            if matches!(read_sweep, ReadSweep::On) {
                let mut sum = 0u64;
                for chunk in chunks.iter() {
                    sum = sum.wrapping_add(chunk.iter().map(|&b| b as u64).sum::<u64>());
                }
                ticker.fetch_add(sum, Ordering::Relaxed);
            } else {
                ticker.fetch_add(chunks[0][0] as u64, Ordering::Relaxed);
            }
            drop(chunks);
        }
    }
}

fn run_variant(label: &str, shape: ChunkShape, read_sweep: ReadSweep, iters: usize) -> (f64, u64) {
    let ticker = Arc::new(AtomicU64::new(0));
    let mut total_ns = 0u128;
    for _iter in 0..iters {
        let barrier = Arc::new(Barrier::new(N_WORKERS));
        let mut handles = Vec::with_capacity(N_WORKERS);
        for _ in 0..N_WORKERS {
            let barrier = Arc::clone(&barrier);
            let ticker = Arc::clone(&ticker);
            handles.push(thread::spawn(move || {
                barrier.wait();
                worker(shape, read_sweep, &ticker);
            }));
        }
        let start = Instant::now();
        for h in handles {
            h.join().unwrap();
        }
        total_ns += start.elapsed().as_nanos();
    }
    let avg_ms = (total_ns as f64) / (iters as f64) / 1_000_000.0;
    println!(
        "{:<48} avg_ms={:.2} ticker={}",
        label,
        avg_ms,
        ticker.load(Ordering::Relaxed)
    );
    (avg_ms, ticker.load(Ordering::Relaxed))
}

#[cfg(target_os = "linux")]
fn read_page_faults() -> (u64, u64) {
    // Returns (minor, major) page faults from /proc/self/stat
    // Fields 10 (minflt) and 12 (majflt) — 1-indexed per proc(5)
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Skip first field (comm in parens) which can contain spaces
    let (_pid_and_comm, rest) = match stat.find(')') {
        Some(i) => stat.split_at(i + 1),
        None => return (0, 0),
    };
    let parts: Vec<&str> = rest.split_whitespace().collect();
    // After ')' the next fields start at index 0 = state, 1 = ppid, ...
    // Per proc(5): minflt is field 10, majflt is field 12 (1-indexed
    // from start of line, so after the ')' it's field 10-2 = 8 and
    // 12-2 = 10 (0-indexed in `parts`).
    let minflt = parts.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let majflt = parts.get(10).and_then(|s| s.parse().ok()).unwrap_or(0);
    (minflt, majflt)
}

#[cfg(not(target_os = "linux"))]
fn read_page_faults() -> (u64, u64) {
    (0, 0)
}

fn main() {
    let pf_mode = std::env::var("GZIPPY_ALLOC_PF").is_ok();
    if pf_mode {
        println!("=== page-fault mode (1 iter per variant) ===");
        let (m0, j0) = read_page_faults();
        run_variant(
            "glibc+singleVec+readON",
            ChunkShape::SingleVec,
            ReadSweep::On,
            1,
        );
        let (m1, j1) = read_page_faults();
        println!("  page-faults Δ minor={} major={}", m1 - m0, j1 - j0);

        let (m2, j2) = read_page_faults();
        run_variant(
            "glibc+manyBoxes+readON",
            ChunkShape::ManyBoxes,
            ReadSweep::On,
            1,
        );
        let (m3, j3) = read_page_faults();
        println!("  page-faults Δ minor={} major={}", m3 - m2, j3 - j2);

        let (m4, j4) = read_page_faults();
        run_variant(
            "glibc+singleVec+readOFF",
            ChunkShape::SingleVec,
            ReadSweep::Off,
            1,
        );
        let (m5, j5) = read_page_faults();
        println!("  page-faults Δ minor={} major={}", m5 - m4, j5 - j4);

        let (m6, j6) = read_page_faults();
        run_variant(
            "glibc+manyBoxes+readOFF",
            ChunkShape::ManyBoxes,
            ReadSweep::Off,
            1,
        );
        let (m7, j7) = read_page_faults();
        println!("  page-faults Δ minor={} major={}", m7 - m6, j7 - j6);
        return;
    }

    let iters = 10;
    println!("=== alloc_pattern (n={} per variant) ===", iters);
    println!(
        "N_WORKERS={}, TARGET_BYTES_PER_WORKER={}, CHUNK_SIZE_SMALL={}",
        N_WORKERS, TARGET_BYTES_PER_WORKER, CHUNK_SIZE_SMALL
    );
    println!();

    let active_alloc = if cfg!(feature = "global-rpmalloc") {
        "rpmalloc"
    } else {
        "glibc"
    };
    println!("Active global allocator: {}", active_alloc);
    println!();

    // Warm up once
    run_variant(
        "WARMUP                                       ",
        ChunkShape::SingleVec,
        ReadSweep::On,
        1,
    );
    println!();

    let (baseline_ms, _) = run_variant(
        &format!("{}+singleVec+readON              (BASELINE)", active_alloc),
        ChunkShape::SingleVec,
        ReadSweep::On,
        iters,
    );
    let (many_read_ms, _) = run_variant(
        &format!("{}+manyBoxes+readON                         ", active_alloc),
        ChunkShape::ManyBoxes,
        ReadSweep::On,
        iters,
    );
    let (single_write_ms, _) = run_variant(
        &format!(
            "{}+singleVec+readOFF              (write-only)",
            active_alloc
        ),
        ChunkShape::SingleVec,
        ReadSweep::Off,
        iters,
    );
    let (many_write_ms, _) = run_variant(
        &format!(
            "{}+manyBoxes+readOFF              (write-only)",
            active_alloc
        ),
        ChunkShape::ManyBoxes,
        ReadSweep::Off,
        iters,
    );

    println!();
    println!("=== Deltas vs BASELINE ===");
    let delta = |label: &str, ms: f64| {
        let pct = (ms - baseline_ms) / baseline_ms * 100.0;
        println!("  {:<40} {:+.1}% (avg_ms={:.2})", label, pct, ms);
    };
    delta("manyBoxes+readON", many_read_ms);
    delta("singleVec+readOFF (write-only)", single_write_ms);
    delta("manyBoxes+readOFF (write-only)", many_write_ms);

    println!();
    println!("Pass gate: variant must show ≥10% wall reduction over baseline to promote.");
    println!("(page-fault gate measured separately via GZIPPY_ALLOC_PF=1)");
}

#[cfg(feature = "global-rpmalloc")]
#[global_allocator]
static ALLOC: rpmalloc::RpMalloc = rpmalloc::RpMalloc;
