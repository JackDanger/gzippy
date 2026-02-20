//! Local benchmarking: run decompression benchmarks with proper methodology.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const MIN_TRIALS: u32 = 5;
const MAX_TRIALS: u32 = 30;
const TARGET_CV: f64 = 0.05; // 5% coefficient of variation

pub fn run(dataset: Option<&str>) -> Result<(), String> {
    let gzippy = find_gzippy()?;
    let data_dir = find_benchmark_data()?;

    let datasets = discover_datasets(&data_dir, dataset)?;
    if datasets.is_empty() {
        return Err("No benchmark data found. Run: ./scripts/prepare_benchmark_data.sh".to_string());
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LOCAL DECOMPRESSION BENCHMARK                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Binary: {gzippy}");

    for (name, path) in &datasets {
        let file_size = std::fs::metadata(path)
            .map_err(|e| format!("Cannot stat {}: {e}", path.display()))?
            .len();

        println!("\n  {name} ({:.1} MB compressed)", file_size as f64 / 1_048_576.0);
        println!("  {}", "─".repeat(50));

        // Benchmark gzippy
        let gzippy_result = benchmark_tool(&gzippy, &["-d", "-c"], path, file_size)?;
        println!(
            "    gzippy:     {:>8.1} MB/s  (stddev {:.1}%, {} trials)",
            gzippy_result.0,
            gzippy_result.1 * 100.0,
            gzippy_result.2,
        );

        // Try other tools for comparison
        for (tool_name, binary, args) in [
            ("pigz -d", "unpigz", vec!["-c"]),
            ("igzip -d", "igzip", vec!["-d", "-c"]),
            ("rapidgzip", "rapidgzip", vec!["-d", "-c"]),
            ("gzip -d", "gzip", vec!["-d", "-c"]),
        ] {
            if let Ok(path_str) = which(binary) {
                match benchmark_tool(&path_str, &args.iter().map(|s| *s).collect::<Vec<_>>(), path, file_size) {
                    Ok((speed, cv, trials)) => {
                        let vs_gzippy = (gzippy_result.0 / speed - 1.0) * 100.0;
                        let icon = if vs_gzippy > 5.0 {
                            "WIN"
                        } else if vs_gzippy > -5.0 {
                            "TIE"
                        } else {
                            "GAP"
                        };
                        println!(
                            "    {:<11} {:>8.1} MB/s  (stddev {:.1}%, {} trials)  [{icon} {:>+.1}%]",
                            tool_name, speed, cv * 100.0, trials, vs_gzippy,
                        );
                    }
                    Err(_) => {
                        println!("    {:<11} (failed)", tool_name);
                    }
                }
            }
        }
    }

    Ok(())
}

fn benchmark_tool(
    binary: &str,
    args: &[&str],
    input_file: &PathBuf,
    _compressed_size: u64,
) -> Result<(f64, f64, u32), String> {
    let mut speeds = Vec::new();

    for trial in 0..MAX_TRIALS {
        let start = Instant::now();
        let output = Command::new(binary)
            .args(args)
            .arg(input_file)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("Failed to run {binary}: {e}"))?;

        if !output.status.success() {
            return Err(format!("{binary} failed with status {}", output.status));
        }

        let elapsed = start.elapsed();
        let decompressed_size = output.stdout.len() as f64;
        let throughput = decompressed_size / elapsed.as_secs_f64() / 1_048_576.0;
        speeds.push(throughput);

        if trial >= MIN_TRIALS - 1 {
            let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
            let variance = speeds.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / speeds.len() as f64;
            let stddev = variance.sqrt();
            let cv = stddev / mean;
            if cv < TARGET_CV {
                return Ok((mean, cv, speeds.len() as u32));
            }
        }
    }

    let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
    let variance = speeds.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / speeds.len() as f64;
    let stddev = variance.sqrt();
    let cv = stddev / mean;
    Ok((mean, cv, speeds.len() as u32))
}

fn find_gzippy() -> Result<String, String> {
    let candidates = [
        "target/release/gzippy",
        "../target/release/gzippy",
        "../../target/release/gzippy",
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return Ok(c.to_string());
        }
    }
    Err("gzippy binary not found. Run: cargo build --release".to_string())
}

fn find_benchmark_data() -> Result<PathBuf, String> {
    let candidates = [
        "benchmark_data",
        "../benchmark_data",
        "../../benchmark_data",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.is_dir() {
            return Ok(p);
        }
    }
    Err("benchmark_data/ not found".to_string())
}

fn discover_datasets(dir: &PathBuf, filter: Option<&str>) -> Result<Vec<(String, PathBuf)>, String> {
    let mut datasets = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("Cannot read {}: {e}", dir.display()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "gz") {
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
            if let Some(f) = filter {
                if !name.contains(f) {
                    continue;
                }
            }
            datasets.push((name.to_string(), path));
        }
    }

    datasets.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(datasets)
}

fn which(binary: &str) -> Result<String, String> {
    let output = Command::new("which")
        .arg(binary)
        .output()
        .map_err(|e| format!("which failed: {e}"))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(format!("{binary} not found in PATH"))
    }
}
