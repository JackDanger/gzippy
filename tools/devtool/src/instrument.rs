//! Instrumentation: decompress a file with detailed timing breakdown.
//!
//! Measures time spent in each phase of decompression to identify bottlenecks.

use std::process::{Command, Stdio};
use std::time::Instant;

pub fn run(file: &str, threads: Option<&str>) -> Result<(), String> {
    let threads = threads.unwrap_or("1");
    let path = std::path::Path::new(file);
    if !path.exists() {
        return Err(format!("File not found: {}", file));
    }

    let file_size = std::fs::metadata(file)
        .map_err(|e| format!("Cannot stat {}: {}", file, e))?
        .len();

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  INSTRUMENTED DECOMPRESS: {} ({:.1} MB, {} threads)",
        path.file_name().unwrap_or_default().to_string_lossy(),
        file_size as f64 / 1_000_000.0,
        threads
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Phase 1: Read file into memory
    let t0 = Instant::now();
    let data = std::fs::read(file).map_err(|e| format!("Cannot read {}: {}", file, e))?;
    let read_time = t0.elapsed();
    println!(
        "\n  Read file:    {:>8.2} ms  ({:.0} MB/s)",
        read_time.as_secs_f64() * 1000.0,
        file_size as f64 / read_time.as_secs_f64() / 1_000_000.0
    );

    // Phase 2: Analyze structure
    let structure = analyze_structure(&data);
    println!("  Structure:    {}", structure);

    // Phase 3: Decompress with GZIPPY_DEBUG for internal timing
    println!("\n  Decompression timing (5 runs, best-of-5):");

    let mut best_time = f64::MAX;
    let mut debug_output = String::new();

    for i in 0..5 {
        let t = Instant::now();
        let mut child = Command::new("./target/release/gzippy")
            .args(["-d", &format!("-p{}", threads)])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .env("GZIPPY_DEBUG", "1")
            .spawn()
            .map_err(|e| format!("Cannot run gzippy: {}", e))?;

        if let Some(ref mut stdin) = child.stdin {
            use std::io::Write;
            stdin
                .write_all(&data)
                .map_err(|e| format!("Write to stdin failed: {}", e))?;
        }
        // Close stdin by dropping it
        drop(child.stdin.take());

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Wait failed: {}", e))?;
        let elapsed = t.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if elapsed_ms < best_time {
            best_time = elapsed_ms;
            debug_output = String::from_utf8_lossy(&output.stderr).to_string();
        }

        let throughput = file_size as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        println!(
            "    Run {}: {:>8.2} ms  ({:.0} MB/s compressed throughput)",
            i + 1,
            elapsed_ms,
            throughput
        );
    }

    // Parse debug output for internal timings
    if !debug_output.is_empty() {
        println!("\n  Debug output (best run):");
        for line in debug_output.lines() {
            println!("    {}", line);
        }
    }

    // Phase 4: Compare against competitors
    println!("\n  Competitor comparison (best-of-5):");

    let pigz_threads = format!("-p{}", threads);
    let competitors: Vec<(&str, Vec<&str>)> = vec![
        ("pigz", vec!["pigz", "-d", &pigz_threads]),
        ("gzip", vec!["gzip", "-d"]),
    ];

    for (name, cmd_args) in &competitors {
        let cmd_name = cmd_args[0];
        if Command::new(cmd_name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            println!("    {:<12} (not found)", name);
            continue;
        }

        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let mut child = Command::new(cmd_name)
                .args(&cmd_args[1..])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("Cannot run {}: {}", cmd_name, e))?;

            if let Some(ref mut stdin) = child.stdin {
                use std::io::Write;
                let _ = stdin.write_all(&data);
            }
            drop(child.stdin.take());
            let _ = child.wait();
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            if elapsed < best {
                best = elapsed;
            }
        }

        let throughput = file_size as f64 / (best / 1000.0) / 1_000_000.0;
        let vs_gzippy = (best_time / best - 1.0) * 100.0;
        let vs_label = if vs_gzippy < 0.0 {
            format!("gzippy is {:.1}% faster", -vs_gzippy)
        } else {
            format!("gzippy is {:.1}% slower", vs_gzippy)
        };
        println!(
            "    {:<12} {:>8.2} ms  ({:.0} MB/s)  — {}",
            name, best, throughput, vs_label
        );
    }

    // Summary
    let output_estimate = estimate_output_size(&data);
    let decompressed_throughput = output_estimate as f64 / (best_time / 1000.0) / 1_000_000.0;
    println!(
        "\n  Summary: {:.2} ms best, ~{:.0} MB/s decompressed throughput",
        best_time, decompressed_throughput
    );

    Ok(())
}

fn analyze_structure(data: &[u8]) -> String {
    if data.len() < 2 {
        return "too small".to_string();
    }
    if data[0] != 0x1f || data[1] != 0x8b {
        return "not gzip".to_string();
    }

    // Check for BGZF
    if data.len() >= 18 {
        let has_bgzf_extra = data[3] & 0x04 != 0; // FEXTRA flag
        if has_bgzf_extra && data.len() >= 16 {
            // Look for BC subfield
            let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
            if xlen >= 4 && 12 + xlen <= data.len() {
                let extra = &data[12..12 + xlen];
                if extra.len() >= 4 && extra[0] == b'B' && extra[1] == b'C' {
                    return "BGZF (parallel-friendly)".to_string();
                }
            }
        }
    }

    // Count members
    let mut count = 0;
    let mut pos = 0;
    let magic = [0x1f, 0x8b];
    while pos + 1 < data.len() {
        if data[pos] == magic[0] && data[pos + 1] == magic[1] {
            count += 1;
            pos += 10; // skip past header minimum
        } else {
            pos += 1;
        }
    }

    if count > 1 {
        format!("multi-member ({} members)", count)
    } else {
        "single-member".to_string()
    }
}

fn estimate_output_size(data: &[u8]) -> u64 {
    if data.len() < 4 {
        return 0;
    }
    let isize_bytes = &data[data.len() - 4..];
    u32::from_le_bytes([
        isize_bytes[0],
        isize_bytes[1],
        isize_bytes[2],
        isize_bytes[3],
    ]) as u64
}
