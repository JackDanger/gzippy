use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;

#[allow(dead_code)]
pub const DATASET_SIZE: usize = 211 * 1024 * 1024; // ~211 MB matching silesia

/// Write `data` to `path` atomically: write to a per-thread temp file, then
/// `rename` it into place. cargo runs `#[test]`s concurrently and several call
/// `prepare_datasets()`, so a plain `fs::write` lets one thread observe a
/// 0-byte / half-written fixture mid-write — the source of the transient
/// `gz[3]` index-out-of-bounds panics in the bench/corpus tests. `rename` is
/// atomic on POSIX, so a concurrent reader sees either the old file or the
/// complete new one, never a partial.
#[allow(dead_code)]
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!(
        "tmp.{}.{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Generate a synthetic software archive (source code patterns)
#[allow(dead_code)]
pub fn generate_software_archive(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let mut file = File::create(path)?;
    let patterns = [
        "    pub fn new() -> Self {\n        Self {\n            data: Vec::with_capacity(1024),\n            count: 0,\n        }\n    }\n",
        "    #[inline(always)]\n    pub fn get_count(&self) -> usize {\n        self.count\n    }\n",
        "// TODO: Implement SIMD optimization for this loop\nfor i in 0..data.len() {\n    sum += data[i] as u64;\n}\n",
        "/*\n * Copyright (c) 2026 The gzippy Authors. All rights reserved.\n * Use of this source code is governed by a BSD-style license that can be\n * found in the LICENSE file.\n */\n",
        "    match result {\n        Ok(val) => println!(\"Success: {}\", val),\n        Err(e) => eprintln!(\"Error: {}\", e),\n    }\n",
        "fn main() {\n    let args: Vec<String> = env::args().collect();\n    if args.len() > 1 {\n        process_file(&args[1]);\n    }\n}\n",
        "pub trait Decoder {\n    fn decode(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize>;\n    fn reset(&mut self);\n}\n",
    ];

    let mut written = 0;
    let mut i = 0;
    while written < DATASET_SIZE {
        let pattern = patterns[i % patterns.len()];
        // Add a bit of variation to prevent perfect RLE
        let variation = format!("// Line variant {}\n{}", written % 1000, pattern);
        file.write_all(variation.as_bytes())?;
        written += variation.len();
        i += 1;
    }
    Ok(())
}

/// Generate a synthetic log dataset (highly repetitive)
#[allow(dead_code)]
pub fn generate_repetitive_logs(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let mut file = File::create(path)?;
    let log_patterns = [
        "2026-01-20 14:30:05 INFO [com.gzippy.core] Processed block {} in {}ms\n",
        "2026-01-20 14:30:05 DEBUG [com.gzippy.sched] Lane {} claimed chunk at offset 0x{:x}\n",
        "2026-01-20 14:30:06 INFO [com.gzippy.core] Processed block {} in {}ms\n",
        "2026-01-20 14:30:06 ERROR [com.gzippy.io] Write failed: Connection reset by peer (attempt {})\n",
        "2026-01-20 14:30:07 WARN [com.gzippy.mem] Memory usage reaching {}% of quota\n",
    ];

    let mut written = 0;
    let mut i = 0;
    while written < DATASET_SIZE {
        let pattern = log_patterns[i % log_patterns.len()];
        let entry = match i % 5 {
            0 => pattern
                .replace("{}", &(i % 10000).to_string())
                .replace("{}", &(i % 10).to_string()),
            1 => pattern
                .replace("{}", &(i % 8).to_string())
                .replace("{:x}", &(i * 4096).to_string()),
            2 => pattern
                .replace("{}", &((i + 1) % 10000).to_string())
                .replace("{}", &(i % 12).to_string()),
            3 => pattern.replace("{}", &(i % 3).to_string()),
            4 => pattern.replace("{}", &(70 + (i % 25)).to_string()),
            _ => pattern.to_string(),
        };
        file.write_all(entry.as_bytes())?;
        written += entry.len();
        i += 1;
    }
    Ok(())
}

/// Get all benchmark datasets (name, raw_path, gzip_path)
/// Ensures they are generated and compressed if needed.
#[allow(dead_code)]
pub fn prepare_datasets() -> Result<Vec<(&'static str, String, String)>> {
    let dir = Path::new("benchmark_data");
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }

    let datasets = vec![
        (
            "silesia",
            "benchmark_data/silesia.tar",
            "benchmark_data/silesia-gzip.tar.gz",
        ),
        (
            "software",
            "benchmark_data/software.archive",
            "benchmark_data/software.archive.gz",
        ),
        (
            "logs",
            "benchmark_data/logs.txt",
            "benchmark_data/logs.txt.gz",
        ),
    ];

    let mut results = Vec::new();

    for (name, raw, gz) in datasets {
        let raw_path = Path::new(raw);
        let gz_path = Path::new(gz);

        if name == "software" {
            generate_software_archive(raw_path)?;
        } else if name == "logs" {
            generate_repetitive_logs(raw_path)?;
        }

        // Ensure compressed version exists. Skip if the raw source is empty
        // (e.g. silesia.tar still mid-extraction in a sibling test) — writing a
        // gz for empty input would yield a tiny member the corpus assertions
        // then reject; better to leave the fixture absent and let consumers
        // skip cleanly. The gz is written atomically (see `atomic_write`).
        if raw_path.exists() && !gz_path.exists() && std::fs::metadata(raw_path)?.len() > 0 {
            eprintln!("[BENCH] Encoding {} into distinct format {}...", name, gz);
            let raw_data = std::fs::read(raw_path)?;
            if raw_data.is_empty() {
                continue;
            }

            if name == "software" {
                // SOFTWARE: Single-member, libdeflate L12 (Deep match search)
                let mut compressed = vec![0u8; raw_data.len() + 1024];
                let mut compressor =
                    libdeflater::Compressor::new(libdeflater::CompressionLvl::new(12).unwrap());
                let size = compressor
                    .gzip_compress(&raw_data, &mut compressed)
                    .unwrap();
                compressed.truncate(size);
                atomic_write(gz_path, &compressed)?;
            } else if name == "logs" {
                // LOGS: Single-member, libdeflate L1 (Fastest, often produces fixed Huffman blocks)
                let mut compressed = vec![0u8; raw_data.len() + 1024];
                let mut compressor =
                    libdeflater::Compressor::new(libdeflater::CompressionLvl::new(1).unwrap());
                let size = compressor
                    .gzip_compress(&raw_data, &mut compressed)
                    .unwrap();
                compressed.truncate(size);
                atomic_write(gz_path, &compressed)?;
            } else {
                // SILESIA: Single-member (flate2 best)
                let mut compressed = Vec::new();
                use flate2::write::GzEncoder;
                use flate2::Compression;
                let mut enc = GzEncoder::new(&mut compressed, Compression::best());
                enc.write_all(&raw_data)?;
                enc.finish()?;
                atomic_write(gz_path, &compressed)?;
            }
        }

        if gz_path.exists() {
            results.push((name, raw.to_string(), gz.to_string()));
        }
    }

    Ok(results)
}

/// Load the real silesia gzip benchmark file if present on disk.
/// Returns `None` when `benchmark_data/silesia-gzip.tar.gz` is missing
/// (typical in CI; the bench host has the corpus).
#[cfg(all(test, parallel_sm))]
pub fn load_silesia_gzip() -> Option<Vec<u8>> {
    let path = Path::new("benchmark_data/silesia-gzip.tar.gz");
    if !path.exists() {
        return None;
    }
    // Treat an empty/short file as absent: a sibling test can transiently
    // race `prepare_datasets()` on this fixture and observe a 0-byte or
    // partially-written file. A valid gzip member is at least 18 bytes
    // (10-byte header + 8-byte trailer); anything shorter is not usable, so
    // skip instead of handing a truncated buffer to the corpus assertions.
    match std::fs::read(path) {
        Ok(d) if d.len() >= 18 => Some(d),
        _ => None,
    }
}
