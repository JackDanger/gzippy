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
                match benchmark_tool(&path_str, &args.to_vec(), path, file_size) {
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
        if path.extension().is_some_and(|e| e == "gz") {
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

// ─── A/B benchmark comparison ───────────────────────────────────────────────

pub fn run_ab(ref_a: &str, ref_b: &str, dataset: Option<&str>, threads: Option<&str>) -> Result<(), String> {
    let data_dir = find_benchmark_data()?;
    let datasets = discover_datasets(&data_dir, dataset)?;
    if datasets.is_empty() {
        return Err("No benchmark data found. Run: ./scripts/prepare_benchmark_data.sh".to_string());
    }

    // Save current state
    let current_branch = git_output(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let is_dirty = !git_output(&["status", "--porcelain"])?.is_empty();

    if is_dirty {
        println!("  Stashing uncommitted changes...");
        git_run(&["stash", "push", "-m", "gzippy-dev bench ab"])?;
    }

    let thread_count = threads.unwrap_or("1");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  A/B BENCHMARK COMPARISON                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("  Ref A: {ref_a}");
    println!("  Ref B: {ref_b}");
    if thread_count != "1" {
        println!("  Threads: {thread_count}");
    }

    let result = run_ab_inner(ref_a, ref_b, &datasets, thread_count);

    // Restore original state regardless of outcome
    let checkout_result = git_run(&["checkout", current_branch.trim()]);
    if is_dirty {
        let _ = git_run(&["stash", "pop"]);
    }
    checkout_result?;

    result
}

fn run_ab_inner(
    ref_a: &str,
    ref_b: &str,
    datasets: &[(String, PathBuf)],
    threads: &str,
) -> Result<(), String> {
    let tmp_dir = std::env::temp_dir().join("gzippy-ab");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create {}: {e}", tmp_dir.display()))?;

    let binary_a = tmp_dir.join("gzippy-a");
    let binary_b = tmp_dir.join("gzippy-b");

    // Build ref A
    println!("\n  Building ref A ({ref_a})...");
    git_run(&["checkout", ref_a])?;
    cargo_build_release()?;
    std::fs::copy("target/release/gzippy", &binary_a)
        .map_err(|e| format!("Cannot copy binary: {e}"))?;
    let commit_a = git_output(&["rev-parse", "--short", "HEAD"])?;
    println!("    Built {} ({})", ref_a, commit_a.trim());

    // Build ref B
    println!("  Building ref B ({ref_b})...");
    git_run(&["checkout", ref_b])?;
    cargo_build_release()?;
    std::fs::copy("target/release/gzippy", &binary_b)
        .map_err(|e| format!("Cannot copy binary: {e}"))?;
    let commit_b = git_output(&["rev-parse", "--short", "HEAD"])?;
    println!("    Built {} ({})", ref_b, commit_b.trim());

    let thread_flag = format!("-p{threads}");

    // Benchmark both
    println!("\n  {:<36} {:>10} {:>10} {:>9}",
             "Dataset", ref_a, ref_b, "Change");
    println!("  {}", "─".repeat(68));

    let mut total_a = 0.0f64;
    let mut total_b = 0.0f64;
    let mut count = 0u32;

    for (name, path) in datasets {
        // Read input once and reuse for all trials
        let input_data = std::fs::read(path)
            .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;

        let bin_a = binary_a.to_str().unwrap();
        let bin_b = binary_b.to_str().unwrap();

        // Interleave trials: A B A B A B ... to reduce temporal bias
        let mut speeds_a = Vec::new();
        let mut speeds_b = Vec::new();

        for _ in 0..MAX_TRIALS {
            if let Ok(s) = measure_with_threads(bin_a, &input_data, &thread_flag) {
                speeds_a.push(s);
            }
            if let Ok(s) = measure_with_threads(bin_b, &input_data, &thread_flag) {
                speeds_b.push(s);
            }

            if speeds_a.len() >= MIN_TRIALS as usize && speeds_b.len() >= MIN_TRIALS as usize {
                let cv_a = coefficient_of_variation(&speeds_a);
                let cv_b = coefficient_of_variation(&speeds_b);
                if cv_a < TARGET_CV && cv_b < TARGET_CV {
                    break;
                }
            }
        }

        if speeds_a.is_empty() || speeds_b.is_empty() {
            println!("  {:<36} (benchmark failed)", name);
            continue;
        }

        let mean_a = speeds_a.iter().sum::<f64>() / speeds_a.len() as f64;
        let mean_b = speeds_b.iter().sum::<f64>() / speeds_b.len() as f64;
        let change = (mean_b / mean_a - 1.0) * 100.0;

        let flag = if change > 5.0 {
            " FASTER"
        } else if change < -5.0 {
            " SLOWER"
        } else {
            ""
        };

        println!("  {:<36} {:>8.1}  {:>8.1}  {:>+7.1}%{}",
                 name, mean_a, mean_b, change, flag);

        total_a += mean_a;
        total_b += mean_b;
        count += 1;
    }

    if count > 0 {
        let overall = (total_b / total_a - 1.0) * 100.0;
        println!("  {}", "─".repeat(68));
        println!("  {:<36} {:>8.1}  {:>8.1}  {:>+7.1}%",
                 "TOTAL (sum of means)", total_a, total_b, overall);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(())
}

fn measure_with_threads(binary: &str, input_data: &[u8], thread_flag: &str) -> Result<f64, String> {
    let start = Instant::now();
    let mut child = Command::new(binary)
        .args(["-d", thread_flag])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to run {binary}: {e}"))?;

    if let Some(ref mut stdin) = child.stdin {
        use std::io::Write;
        stdin.write_all(input_data)
            .map_err(|e| format!("Write to stdin failed: {e}"))?;
    }
    drop(child.stdin.take());

    let output = child.wait_with_output()
        .map_err(|e| format!("Wait failed: {e}"))?;

    if !output.status.success() {
        return Err(format!("{binary} failed"));
    }

    let elapsed = start.elapsed();
    let decompressed_size = output.stdout.len() as f64;
    Ok(decompressed_size / elapsed.as_secs_f64() / 1_048_576.0)
}

pub(crate) fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::MAX;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return f64::MAX;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean
}

fn git_run(args: &[&str]) -> Result<(), String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git {} failed: {e}", args[0]))?;

    if !output.status.success() {
        return Err(format!("git {} failed: {}", args[0],
                          String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git {} failed: {e}", args[0]))?;

    if !output.status.success() {
        return Err(format!("git {} failed: {}", args[0],
                          String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn cargo_build_release() -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["build", "--release"])
        .output()
        .map_err(|e| format!("cargo build failed: {e}"))?;

    if !output.status.success() {
        return Err(format!("cargo build --release failed:\n{}",
                          String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_empty() {
        assert_eq!(coefficient_of_variation(&[]), f64::MAX);
    }

    #[test]
    fn test_cv_single_value() {
        let cv = coefficient_of_variation(&[100.0]);
        assert_eq!(cv, 0.0);
    }

    #[test]
    fn test_cv_identical_values() {
        let cv = coefficient_of_variation(&[100.0, 100.0, 100.0]);
        assert!(cv < 0.001, "CV should be ~0 for identical values, got {cv}");
    }

    #[test]
    fn test_cv_known_values() {
        // mean = 100, stddev = 10, cv = 0.1
        let cv = coefficient_of_variation(&[90.0, 100.0, 110.0]);
        assert!((cv - 0.0816).abs() < 0.01, "CV should be ~0.08, got {cv}");
    }

    #[test]
    fn test_cv_zeros() {
        assert_eq!(coefficient_of_variation(&[0.0, 0.0]), f64::MAX);
    }

    #[test]
    fn test_cv_high_variance() {
        let cv = coefficient_of_variation(&[1.0, 100.0]);
        assert!(cv > 0.5, "CV should be high for very different values, got {cv}");
    }

    #[test]
    fn test_cv_low_variance() {
        let cv = coefficient_of_variation(&[100.0, 101.0, 99.0, 100.5, 99.5]);
        assert!(cv < 0.01, "CV should be low for tight values, got {cv}");
    }
}
