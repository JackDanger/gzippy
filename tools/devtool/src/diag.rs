//! Structured diagnostics for fleet benchmarks.
//!
//! `gzippy-dev diag [--direction compress|decompress] [--dataset NAME]`
//!
//! Outputs JSON to stdout with platform info, ISA-L build config,
//! compress timing breakdowns, decompress path traces, etc.
//! Designed to run identically on local dev machines and fleet instances.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::bench::{find_repo_root};

pub struct DiagArgs {
    pub direction: Option<String>,
    pub dataset: Option<String>,
}

pub fn run(args: &DiagArgs) -> Result<(), String> {
    let repo_root = find_repo_root()?;
    let data_dir = find_data_dir(&repo_root);
    let bin_dir = find_bin_dir(&repo_root);
    let platform = detect_platform();

    let mut diag = serde_json::Map::new();
    diag.insert("platform".into(), serde_json::Value::String(platform.clone()));

    // Platform info
    collect_platform_info(&mut diag);

    // Tool versions
    collect_tool_versions(&mut diag, &repo_root, &bin_dir);

    // ISA-L build config (x86 only)
    if platform == "x86_64" {
        collect_isal_build_config(&mut diag, &repo_root);
        collect_avx_counts(&mut diag, &repo_root, &bin_dir);
    }

    // Compress diagnostics
    let do_compress = args.direction.as_deref() != Some("decompress");
    if do_compress && platform == "x86_64" {
        collect_compress_diagnostics(&mut diag, &data_dir, &repo_root, &bin_dir, args.dataset.as_deref());
    }

    // Decompress path traces
    let do_decompress = args.direction.as_deref() != Some("compress");
    if do_decompress {
        collect_decompress_paths(&mut diag, &data_dir, &repo_root, args.dataset.as_deref());
    }

    let output = serde_json::Value::Object(diag);
    println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    Ok(())
}

fn detect_platform() -> String {
    match std::env::consts::ARCH {
        "x86_64" | "x86" => "x86_64".into(),
        "aarch64" => "arm64".into(),
        other => other.into(),
    }
}

fn find_data_dir(repo_root: &Path) -> PathBuf {
    for candidate in [PathBuf::from("/dev/shm"), repo_root.join("benchmark_data")] {
        if candidate.is_dir() {
            let has_data = candidate.join("silesia.tar").exists()
                || candidate.join("software.archive").exists()
                || candidate.join("logs.txt").exists();
            if has_data {
                return candidate;
            }
        }
    }
    repo_root.join("benchmark_data")
}

fn find_bin_dir(repo_root: &Path) -> Option<PathBuf> {
    let p = repo_root.join("bin");
    if p.is_dir() { Some(p) } else { None }
}

fn cmd_stdout(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn shell_stdout(cmd: &str) -> Option<String> {
    Command::new("sh")
        .args(["-c", cmd])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn collect_platform_info(diag: &mut serde_json::Map<String, serde_json::Value>) {
    let arch = cmd_stdout("uname", &["-m"]).unwrap_or_else(|| "unknown".into());
    diag.insert("arch".into(), serde_json::Value::String(arch));

    // CPU features (Linux only)
    if let Some(features) = shell_stdout("grep -oE 'avx[^ ]*|neon|asimd' /proc/cpuinfo 2>/dev/null | sort -u | tr '\\n' ' '") {
        diag.insert("cpu_features".into(), serde_json::Value::String(features.trim().into()));
    } else if let Some(features) = shell_stdout("sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | tr ' ' '\\n' | sort -u | tr '\\n' ' '") {
        diag.insert("cpu_features".into(), serde_json::Value::String(features.trim().to_lowercase()));
    }

    if let Some(cpus) = shell_stdout("nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null") {
        diag.insert("cpu_count".into(), serde_json::Value::String(cpus));
    }
}

fn collect_tool_versions(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    repo_root: &Path,
    bin_dir: &Option<PathBuf>,
) {
    let mut versions = serde_json::Map::new();

    // gzippy commit
    if let Some(commit) = cmd_stdout("git", &["-C", &repo_root.to_string_lossy(), "rev-parse", "--short", "HEAD"]) {
        versions.insert("gzippy_commit".into(), serde_json::Value::String(commit));
    }

    // igzip version
    if let Some(bd) = bin_dir {
        let igzip = bd.join("igzip");
        if igzip.exists() {
            if let Some(v) = shell_stdout(&format!("{} --version 2>&1 | head -1", igzip.display())) {
                versions.insert("igzip".into(), serde_json::Value::String(v));
            }
        }
        let rg = bd.join("rapidgzip");
        if rg.exists() {
            if let Some(v) = shell_stdout(&format!("{} --version 2>&1 | head -1", rg.display())) {
                versions.insert("rapidgzip".into(), serde_json::Value::String(v));
            }
        }
    }

    // ISA-L version from isal-sys build
    if let Some(v) = shell_stdout(&format!(
        "grep \"^PACKAGE_VERSION=\" {}/target/release/build/isal-sys-*/out/isa-l/config.log 2>/dev/null | head -1 | sed \"s/PACKAGE_VERSION='\\(.*\\)'/\\1/\"",
        repo_root.display()
    )) {
        versions.insert("isal_sys_version".into(), serde_json::Value::String(v));
    }

    diag.insert("versions".into(), serde_json::Value::Object(versions));
}

fn collect_isal_build_config(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    repo_root: &Path,
) {
    let mut config = serde_json::Map::new();

    let config_log_pattern = format!(
        "{}/target/release/build/isal-sys-*/out/isa-l/config.log",
        repo_root.display()
    );

    // HAVE_NASM
    if let Some(nasm) = shell_stdout(&format!(
        "grep \"ac_cv_prog_HAVE_NASM=\" {} 2>/dev/null | head -1",
        config_log_pattern
    )) {
        let has_nasm = nasm.contains("yes");
        config.insert("have_nasm".into(), serde_json::Value::Bool(has_nasm));
        config.insert("have_nasm_raw".into(), serde_json::Value::String(nasm));
    } else {
        config.insert("have_nasm".into(), serde_json::Value::Bool(false));
        config.insert("config_log_found".into(), serde_json::Value::Bool(false));
    }

    // CFLAGS
    if let Some(cflags) = shell_stdout(&format!(
        "grep \"^CFLAGS=\" {} 2>/dev/null | head -1",
        config_log_pattern
    )) {
        config.insert("cflags".into(), serde_json::Value::String(cflags));
    }

    diag.insert("isal_build".into(), serde_json::Value::Object(config));
}

fn collect_avx_counts(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    repo_root: &Path,
    bin_dir: &Option<PathBuf>,
) {
    let gzippy_bin = repo_root.join("target/release/gzippy");
    if gzippy_bin.exists() {
        if let Some(count) = shell_stdout(&format!(
            "objdump -d {} 2>/dev/null | grep -cE 'v(p|add|sub|mul|perm|blend|gather|shuf)' || echo 0",
            gzippy_bin.display()
        )) {
            if let Ok(n) = count.parse::<u64>() {
                diag.insert("avx_insns_gzippy".into(), serde_json::json!(n));
            }
        }
    }

    if let Some(bd) = bin_dir {
        let igzip_bin = bd.join("igzip");
        if igzip_bin.exists() {
            if let Some(count) = shell_stdout(&format!(
                "objdump -d {} 2>/dev/null | grep -cE 'v(p|add|sub|mul|perm|blend|gather|shuf)' || echo 0",
                igzip_bin.display()
            )) {
                if let Ok(n) = count.parse::<u64>() {
                    diag.insert("avx_insns_igzip".into(), serde_json::json!(n));
                }
            }
        }
    }
}

fn collect_compress_diagnostics(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    data_dir: &Path,
    repo_root: &Path,
    bin_dir: &Option<PathBuf>,
    dataset_filter: Option<&str>,
) {
    let gzippy_bin = repo_root.join("target/release/gzippy");
    if !gzippy_bin.exists() {
        diag.insert("compress_error".into(), serde_json::json!("gzippy binary not found"));
        return;
    }

    let datasets: Vec<(&str, &str)> = vec![
        ("software", "software.archive"),
        ("logs", "logs.txt"),
        ("silesia", "silesia.tar"),
    ];

    let mut compress_timings = serde_json::Map::new();

    for (name, file) in &datasets {
        if let Some(filter) = dataset_filter {
            if *name != filter { continue; }
        }

        let raw_path = data_dir.join(file);
        if !raw_path.exists() { continue; }

        let input_size = std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0);
        if input_size == 0 { continue; }

        // Run gzippy with GZIPPY_DEBUG=1 to get timing breakdown
        let output = Command::new(&gzippy_bin)
            .args(["-1", "-c", "-p1"])
            .stdin(std::fs::File::open(&raw_path).unwrap())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .env("GZIPPY_DEBUG", "1")
            .output();

        let mut timing = serde_json::Map::new();
        timing.insert("input_bytes".into(), serde_json::json!(input_size));
        timing.insert("input_mb".into(), serde_json::json!((input_size as f64 / 1e6 * 10.0).round() / 10.0));

        if let Ok(out) = output {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            timing.insert("debug_output".into(), serde_json::Value::String(stderr.clone()));

            // Parse "[isal] L1 compress_gzip_to_writer: alloc=X.Xms compress=X.Xms (XXXX MB/s) write=X.Xms"
            for line in stderr.lines() {
                if line.contains("compress_gzip_to_writer") {
                    if let Some(alloc) = extract_ms(line, "alloc=") {
                        timing.insert("alloc_ms".into(), serde_json::json!(alloc));
                    }
                    if let Some(compress) = extract_ms(line, "compress=") {
                        timing.insert("compress_ms".into(), serde_json::json!(compress));
                    }
                    if let Some(write) = extract_ms(line, "write=") {
                        timing.insert("write_ms".into(), serde_json::json!(write));
                    }
                    if let Some(mbps) = extract_paren_mbps(line) {
                        timing.insert("compress_mbps".into(), serde_json::json!(mbps));
                    }
                }
            }

            // Also measure total wall time ourselves
            let t0 = std::time::Instant::now();
            let _ = Command::new(&gzippy_bin)
                .args(["-1", "-c", "-p1"])
                .stdin(std::fs::File::open(&raw_path).unwrap())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let elapsed = t0.elapsed();
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let total_mbps = input_size as f64 / elapsed.as_secs_f64() / 1e6;
            timing.insert("total_ms".into(), serde_json::json!((total_ms * 10.0).round() / 10.0));
            timing.insert("total_mbps".into(), serde_json::json!((total_mbps * 10.0).round() / 10.0));
        } else {
            timing.insert("error".into(), serde_json::json!("failed to run gzippy"));
        }

        // Compare with igzip on same data
        if let Some(bd) = bin_dir {
            let igzip = bd.join("igzip");
            if igzip.exists() {
                let t0 = std::time::Instant::now();
                let _ = Command::new(&igzip)
                    .args(["-1", "-c"])
                    .stdin(std::fs::File::open(&raw_path).unwrap())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                let elapsed = t0.elapsed();
                let igzip_mbps = input_size as f64 / elapsed.as_secs_f64() / 1e6;
                timing.insert("igzip_mbps".into(), serde_json::json!((igzip_mbps * 10.0).round() / 10.0));
                timing.insert("igzip_ms".into(), serde_json::json!((elapsed.as_secs_f64() * 1000.0 * 10.0).round() / 10.0));
            }
        }

        compress_timings.insert((*name).into(), serde_json::Value::Object(timing));
    }

    diag.insert("compress_timing".into(), serde_json::Value::Object(compress_timings));

    // Dispatch check: 1KB block at L0
    collect_dispatch_check(diag, &gzippy_bin);
}

fn collect_dispatch_check(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    gzippy_bin: &Path,
) {
    // We can't call isal::compress_into directly from the devtool (different binary).
    // Instead, run the test that does the dispatch check.
    let output = Command::new("cargo")
        .args(["test", "--release", "--features", "isal-compression",
               "bench_isal_compress_throughput", "--", "--nocapture"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    let mut check = serde_json::Map::new();

    if let Ok(out) = output {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        // Parse "[diag] ISA-L 1KB L0 dispatch check: XXXX MB/s (XXX ns/call) — ASSEMBLY (good)"
        for line in combined.lines() {
            if line.contains("dispatch check") {
                check.insert("raw".into(), serde_json::Value::String(line.trim().into()));
                if let Some(mbps) = extract_first_number(line) {
                    check.insert("1kb_mbps".into(), serde_json::json!(mbps));
                    let verdict = if mbps > 1500.0 { "ASSEMBLY" } else { "C_FALLBACK" };
                    check.insert("verdict".into(), serde_json::Value::String(verdict.into()));
                }
            }
            // Also capture per-level throughput
            if line.contains("[bench] ISA-L compress") {
                if let Some(mbps) = extract_first_number(line) {
                    let key = line.trim().split(':').next().unwrap_or("").trim();
                    check.insert(
                        key.replace("[bench] ", "").replace(' ', "_"),
                        serde_json::json!(mbps),
                    );
                }
            }
        }

        if !out.status.success() {
            check.insert("test_passed".into(), serde_json::Value::Bool(false));
            if !gzippy_bin.exists() {
                check.insert("note".into(), serde_json::json!("isal-compression feature may not be enabled"));
            }
        } else {
            check.insert("test_passed".into(), serde_json::Value::Bool(true));
        }
    } else {
        check.insert("error".into(), serde_json::json!("failed to run cargo test"));
    }

    diag.insert("dispatch_check".into(), serde_json::Value::Object(check));
}

fn collect_decompress_paths(
    diag: &mut serde_json::Map<String, serde_json::Value>,
    data_dir: &Path,
    repo_root: &Path,
    dataset_filter: Option<&str>,
) {
    let gzippy_bin = repo_root.join("target/release/gzippy");
    if !gzippy_bin.exists() {
        diag.insert("decompress_error".into(), serde_json::json!("gzippy binary not found"));
        return;
    }

    let archives = [
        ("silesia", "gzip", "silesia-gzip.gz"),
        ("silesia", "bgzf", "silesia-bgzf.gz"),
        ("silesia", "pigz", "silesia-pigz.gz"),
        ("software", "gzip", "software-gzip.gz"),
        ("software", "bgzf", "software-bgzf.gz"),
        ("software", "pigz", "software-pigz.gz"),
        ("logs", "gzip", "logs-gzip.gz"),
        ("logs", "bgzf", "logs-bgzf.gz"),
        ("logs", "pigz", "logs-pigz.gz"),
    ];

    let mut paths = serde_json::Map::new();
    let mut isal_fallback_detected = false;
    let mut isal_fallback_files: Vec<String> = Vec::new();

    for (dataset, archive_type, filename) in &archives {
        if let Some(filter) = dataset_filter {
            if *dataset != filter { continue; }
        }

        let gz_path = data_dir.join(filename);
        if !gz_path.exists() { continue; }

        let output = Command::new(&gzippy_bin)
            .args(["-d", "-p4"])
            .stdin(std::fs::File::open(&gz_path).unwrap())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .env("GZIPPY_DEBUG", "1")
            .output();

        let key = format!("{dataset}-{archive_type}");
        if let Ok(out) = output {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let path_info = parse_decompress_path(&stderr);
            if stderr.contains("ISA-L decompress failed") {
                isal_fallback_detected = true;
                isal_fallback_files.push(key.clone());
            }
            paths.insert(key, serde_json::Value::String(path_info));
        } else {
            paths.insert(key, serde_json::json!("error: failed to run"));
        }
    }

    diag.insert("decompress_paths".into(), serde_json::Value::Object(paths));
    diag.insert("isal_decompress_fallback".into(), serde_json::json!(isal_fallback_detected));
    if !isal_fallback_files.is_empty() {
        diag.insert("isal_fallback_files".into(), serde_json::json!(isal_fallback_files));
    }
}

fn parse_decompress_path(debug_output: &str) -> String {
    // Collect all debug lines to build a complete picture
    let mut io_method = String::new();
    let mut route = String::new();

    for line in debug_output.lines() {
        let trimmed = line.trim();

        // I/O method
        if trimmed.contains("stdin mmap") {
            io_method = "mmap".into();
        }

        // Route (later lines override earlier ones — most specific wins)
        if trimmed.contains("BGZF parallel") || trimmed.contains("BGZF path") {
            route = format!("BGZF parallel | {trimmed}");
        } else if trimmed.contains("BGZF decompress") {
            route = format!("BGZF | {trimmed}");
        } else if trimmed.contains("Multi-member parallel") || trimmed.contains("Multi-member path") {
            route = format!("multi-member parallel | {trimmed}");
        } else if trimmed.contains("Single-member path") || trimmed.contains("Single-member parallel") {
            route = format!("single-member | {trimmed}");
        } else if trimmed.contains("decompress_file stdout") || trimmed.contains("decompress_stdin:") {
            // Parse the format detection line to build route info
            let bgzf = trimmed.contains("bgzf=true");
            let multi = trimmed.contains("multi=true");
            let parallel = trimmed.contains("parallel=true");
            let path_desc = if bgzf {
                if parallel { "BGZF parallel" } else { "BGZF sequential" }
            } else if multi {
                if parallel { "multi-member parallel" } else { "multi-member sequential" }
            } else {
                "single-member sequential"
            };
            route = format!("{path_desc} | {trimmed}");
        }
    }

    if route.is_empty() && io_method.is_empty() {
        if debug_output.trim().is_empty() {
            return "no debug output (GZIPPY_DEBUG not working?)".into();
        }
        return format!("unknown path | {}", debug_output.lines().next().unwrap_or(""));
    }

    if route.is_empty() {
        format!("{io_method} | sequential (no route info)")
    } else if io_method.is_empty() {
        route
    } else {
        format!("{io_method} -> {route}")
    }
}

// ─── Parsing helpers ──────────────────────────────────────────────────────────

fn extract_ms(line: &str, prefix: &str) -> Option<f64> {
    let start = line.find(prefix)? + prefix.len();
    let rest = &line[start..];
    let end = rest.find("ms")?;
    rest[..end].parse().ok()
}

fn extract_paren_mbps(line: &str) -> Option<f64> {
    let start = line.find('(')? + 1;
    let rest = &line[start..];
    let end = rest.find(" MB/s")?;
    rest[..end].parse().ok()
}

fn extract_first_number(line: &str) -> Option<f64> {
    // Find first floating-point number after ": "
    let colon_pos = line.find(": ")?;
    let rest = &line[colon_pos + 2..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end].parse().ok()
}
