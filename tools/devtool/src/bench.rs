//! THE benchmark. One implementation, used everywhere: local, cloud, CI.
//!
//! Discovers datasets, archive variants, tools, and thread configs automatically.
//! Outputs human-readable results by default, JSON with --json.
//!
//! Usage:
//!   gzippy-dev bench                      # full benchmark, human output
//!   gzippy-dev bench --json               # full benchmark, JSON to stdout
//!   gzippy-dev bench --dataset silesia    # single dataset
//!   gzippy-dev bench --min-trials 50 --max-trials 200 --target-cv 0.005

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const DEFAULT_MIN_TRIALS: u32 = 10;
const DEFAULT_MAX_TRIALS: u32 = 40;
const DEFAULT_TARGET_CV: f64 = 0.03;

pub struct BenchArgs {
    pub dataset: Option<String>,
    pub archive: Option<String>,
    pub threads: Option<usize>,
    pub json: bool,
    pub min_trials: u32,
    pub max_trials: u32,
    pub target_cv: f64,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            dataset: None,
            archive: None,
            threads: None,
            json: false,
            min_trials: DEFAULT_MIN_TRIALS,
            max_trials: DEFAULT_MAX_TRIALS,
            target_cv: DEFAULT_TARGET_CV,
        }
    }
}

#[derive(Clone)]
struct ToolResult {
    dataset: String,
    archive: String,
    threads: String,
    tool: String,
    speed_mbps: f64,
    cv: f64,
    trials: u32,
    status: String,
    error: Option<String>,
}

// ─── Main entry point ─────────────────────────────────────────────────────────

pub fn run(args: &BenchArgs) -> Result<(), String> {
    let platform = detect_platform();
    let max_threads = num_cpus();
    let repo_root = find_repo_root()?;
    let data_dir = find_data_dir(&repo_root)?;
    let bin_dir = find_bin_dir(&repo_root);

    let tools = discover_tools(&bin_dir);
    if tools.is_empty() {
        return Err("No benchmark tools found".into());
    }

    let datasets = discover_datasets(&data_dir, args.dataset.as_deref())?;
    if datasets.is_empty() {
        return Err(format!(
            "No benchmark datasets found in {}. Run: ./scripts/prepare_benchmark_data.sh",
            data_dir.display()
        ));
    }

    let thread_configs: Vec<(usize, &str)> = match args.threads {
        Some(1) => vec![(1, "T1")],
        Some(n) => vec![(n, "Tmax")],
        None if max_threads > 1 => vec![(1, "T1"), (max_threads, "Tmax")],
        None => vec![(1, "T1")],
    };

    if !args.json {
        eprintln!("Platform:    {platform}");
        eprintln!("CPUs:        {max_threads}");
        eprintln!("Data dir:    {}", data_dir.display());
        eprintln!("Trials:      {}-{}, CV<{:.1}%", args.min_trials, args.max_trials, args.target_cv * 100.0);
        eprintln!("Tools:       {}", tools.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));
        eprintln!("Datasets:    {}", datasets.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(", "));
        eprintln!();
    }

    let mut results: Vec<ToolResult> = Vec::new();
    let tmp_dir = std::env::temp_dir().join("gzippy-bench");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let output_file = tmp_dir.join("bench-output.bin");

    for ds in &datasets {
        for (archive_name, compressed_path) in &ds.archives {
            if let Some(ref af) = args.archive {
                if archive_name != af { continue; }
            }
            for &(threads, threads_label) in &thread_configs {
                if !args.json {
                    eprint!("  {}-{archive_name} {threads_label}  ", ds.name);
                    let _ = std::io::stderr().flush();
                }

                for (tool_name, tool_path) in &tools {
                    // Skip multi-threaded for single-threaded tools
                    if threads > 1 && is_single_threaded(tool_name) {
                        continue;
                    }

                    let result = benchmark_one(
                        tool_name, tool_path, compressed_path,
                        &ds.raw_path, ds.original_size,
                        threads, &output_file,
                        args.min_trials, args.max_trials, args.target_cv,
                    );

                    let tr = match result {
                        Ok((speed, cv, trials)) => {
                            if !args.json {
                                eprint!("{tool_name}:{speed:.0}  ");
                                let _ = std::io::stderr().flush();
                            }
                            ToolResult {
                                dataset: ds.name.clone(),
                                archive: archive_name.clone(),
                                threads: threads_label.to_string(),
                                tool: tool_name.clone(),
                                speed_mbps: speed, cv, trials,
                                status: "pass".into(),
                                error: None,
                            }
                        }
                        Err(e) => {
                            if !args.json {
                                eprint!("{tool_name}:ERR  ");
                                let _ = std::io::stderr().flush();
                            }
                            ToolResult {
                                dataset: ds.name.clone(),
                                archive: archive_name.clone(),
                                threads: threads_label.to_string(),
                                tool: tool_name.clone(),
                                speed_mbps: 0.0, cv: 0.0, trials: 0,
                                status: "fail".into(),
                                error: Some(e),
                            }
                        }
                    };
                    results.push(tr);
                }
                if !args.json {
                    eprintln!();
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    if args.json {
        output_json(&platform, &results);
    } else {
        output_human(&results);
    }

    Ok(())
}

// ─── Benchmark engine ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn benchmark_one(
    tool: &str, binary: &str,
    compressed: &Path, original: &Path, original_size: u64,
    threads: usize, output_file: &Path,
    min_trials: u32, max_trials: u32, target_cv: f64,
) -> Result<(f64, f64, u32), String> {
    let cmd_args = tool_decompress_args(tool, threads);

    // Warmup + correctness check: decompress to file and verify
    run_decompress(binary, &cmd_args, compressed, output_file)?;
    let out_size = std::fs::metadata(output_file)
        .map_err(|e| format!("output file: {e}"))?
        .len();
    // Use actual decompressed size if original file unavailable
    let ref_size = if original.exists() { original_size } else { out_size };

    if original.exists() && out_size != original_size {
        return Err(format!(
            "size mismatch: decompressed {out_size} != original {original_size}"
        ));
    }

    // Adaptive timing trials
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..max_trials {
        let elapsed = time_decompress(binary, &cmd_args, compressed, output_file)?;
        times.push(elapsed);

        if times.len() >= min_trials as usize {
            let (_, _, _, cv) = trimmed_stats(&times);
            if cv < target_cv {
                break;
            }
        }
    }

    let (_, mean, _, cv) = trimmed_stats(&times);
    let speed = ref_size as f64 / mean / 1_000_000.0;
    Ok((speed, cv, times.len() as u32))
}

fn run_decompress(binary: &str, args: &[String], input: &Path, output: &Path) -> Result<(), String> {
    let fin = std::fs::File::open(input)
        .map_err(|e| format!("open {}: {e}", input.display()))?;
    let fout = std::fs::File::create(output)
        .map_err(|e| format!("create {}: {e}", output.display()))?;

    let status = Command::new(binary)
        .args(args)
        .stdin(fin)
        .stdout(fout)
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("{binary}: {e}"))?;

    if !status.success() {
        return Err(format!("{binary} exit {status}"));
    }
    Ok(())
}

fn time_decompress(binary: &str, args: &[String], input: &Path, output: &Path) -> Result<f64, String> {
    let fin = std::fs::File::open(input)
        .map_err(|e| format!("open {}: {e}", input.display()))?;
    let fout = std::fs::File::create(output)
        .map_err(|e| format!("create {}: {e}", output.display()))?;

    let start = Instant::now();
    let status = Command::new(binary)
        .args(args)
        .stdin(fin)
        .stdout(fout)
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("{binary}: {e}"))?;

    if !status.success() {
        return Err(format!("{binary} exit {status}"));
    }
    Ok(start.elapsed().as_secs_f64())
}

fn tool_decompress_args(tool: &str, threads: usize) -> Vec<String> {
    match tool {
        "gzippy" => vec!["-d".into(), format!("-p{threads}")],
        "unpigz" | "pigz" => vec![format!("-p{threads}")],
        "rapidgzip" => vec!["-d".into(), "-P".into(), threads.to_string()],
        "igzip" => vec!["-d".into()],
        "gzip" => vec!["-d".into()],
        _ => vec!["-d".into()],
    }
}

fn is_single_threaded(tool: &str) -> bool {
    matches!(tool, "gzip" | "igzip")
}

fn trimmed_stats(times: &[f64]) -> (Vec<f64>, f64, f64, f64) {
    let mut sorted: Vec<f64> = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let trimmed = if sorted.len() > 4 {
        sorted[1..sorted.len() - 1].to_vec()
    } else {
        sorted.clone()
    };

    let mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    let stdev = if trimmed.len() > 1 {
        (trimmed.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / trimmed.len() as f64).sqrt()
    } else {
        0.0
    };
    let cv = if mean > 0.0 { stdev / mean } else { 1.0 };
    (trimmed, mean, stdev, cv)
}

// ─── Discovery ────────────────────────────────────────────────────────────────

struct Dataset {
    name: String,
    raw_path: PathBuf,
    original_size: u64,
    archives: Vec<(String, PathBuf)>,
}

fn discover_datasets(data_dir: &Path, filter: Option<&str>) -> Result<Vec<Dataset>, String> {
    let known = [
        ("silesia", "silesia.tar"),
        ("software", "software.archive"),
        ("logs", "logs.txt"),
    ];
    let archive_types = ["gzip", "bgzf", "pigz"];

    let mut datasets = Vec::new();
    for (name, raw_name) in &known {
        if let Some(f) = filter {
            if !name.contains(f) { continue; }
        }

        // Look for raw file in data_dir or /dev/shm
        let raw_path = [
            data_dir.join(raw_name),
            PathBuf::from(format!("/dev/shm/{raw_name}")),
        ].into_iter().find(|p| p.exists());

        let raw_path = match raw_path {
            Some(p) => p,
            None => continue,
        };

        let original_size = std::fs::metadata(&raw_path)
            .map_err(|e| format!("{}: {e}", raw_path.display()))?
            .len();

        let mut archives = Vec::new();
        for atype in &archive_types {
            let compressed_name = format!("{name}-{atype}.gz");
            let compressed_path = [
                PathBuf::from(format!("/dev/shm/{compressed_name}")),
                data_dir.join(&compressed_name),
            ].into_iter().find(|p| p.exists());

            if let Some(p) = compressed_path {
                archives.push((atype.to_string(), p));
            }
        }

        if archives.is_empty() {
            // Fall back: look for any .gz file with this dataset name
            if let Ok(entries) = std::fs::read_dir(data_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let fname = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
                    if fname.starts_with(name) && fname.ends_with(".gz") {
                        let atype = fname.trim_start_matches(name)
                            .trim_start_matches('-')
                            .trim_end_matches(".gz")
                            .trim_end_matches(".tar")
                            .trim_end_matches(".txt")
                            .trim_end_matches(".archive");
                        let label = if atype.is_empty() { "gzip" } else { atype };
                        archives.push((label.to_string(), p));
                    }
                }
            }
        }

        if !archives.is_empty() {
            datasets.push(Dataset { name: name.to_string(), raw_path, original_size, archives });
        }
    }

    Ok(datasets)
}

fn discover_tools(bin_dir: &Option<PathBuf>) -> Vec<(String, String)> {
    let mut tools = Vec::new();

    let candidates = [
        ("gzippy", &["target/release/gzippy", "bin/gzippy"][..]),
        ("unpigz", &["pigz/unpigz", "bin/unpigz"]),
        ("igzip", &["isa-l/build/igzip", "bin/igzip"]),
        ("rapidgzip", &["rapidgzip/librapidarchive/build/src/tools/rapidgzip", "bin/rapidgzip"]),
        ("gzip", &["/usr/bin/gzip"]),
    ];

    for (name, paths) in &candidates {
        // Check bin_dir first
        if let Some(bd) = bin_dir {
            let p = bd.join(name);
            if p.exists() && is_executable(&p) {
                tools.push((name.to_string(), p.to_string_lossy().to_string()));
                continue;
            }
        }

        // Check relative paths (from repo root)
        let mut found = false;
        for rpath in *paths {
            let p = PathBuf::from(rpath);
            if p.exists() && is_executable(&p) {
                tools.push((name.to_string(), rpath.to_string()));
                found = true;
                break;
            }
        }
        if found { continue; }

        // Check PATH
        if let Ok(output) = Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    tools.push((name.to_string(), path));
                }
            }
        }
    }

    tools
}

fn find_repo_root() -> Result<PathBuf, String> {
    // Walk up from CWD looking for Cargo.toml with gzippy
    let mut dir = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("src").join("main.rs").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Err("Cannot find gzippy repo root (no Cargo.toml found)".into());
        }
    }
}

fn find_data_dir(repo_root: &Path) -> Result<PathBuf, String> {
    let candidates = [
        repo_root.join("benchmark_data"),
        PathBuf::from("/dev/shm"),
    ];
    for c in &candidates {
        if c.is_dir() {
            // Verify it has at least one relevant file
            if c.join("silesia.tar").exists()
                || c.join("software.archive").exists()
                || c.join("logs.txt").exists()
            {
                return Ok(c.clone());
            }
        }
    }
    // Default to benchmark_data even if it doesn't exist yet
    Ok(repo_root.join("benchmark_data"))
}

fn find_bin_dir(repo_root: &Path) -> Option<PathBuf> {
    let p = repo_root.join("bin");
    if p.is_dir() { Some(p) } else { None }
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        path.exists()
    }
}

fn detect_platform() -> String {
    let arch = std::env::consts::ARCH;
    match arch {
        "x86_64" | "x86" => "x86_64".into(),
        "aarch64" => "arm64".into(),
        other => other.into(),
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ─── Output ───────────────────────────────────────────────────────────────────

fn output_json(platform: &str, results: &[ToolResult]) {
    let items: Vec<serde_json::Value> = results.iter().map(|r| {
        let mut obj = serde_json::json!({
            "dataset": r.dataset,
            "archive": r.archive,
            "threads": r.threads,
            "tool": r.tool,
            "speed_mbps": r.speed_mbps,
            "cv": r.cv,
            "trials": r.trials,
            "status": r.status,
        });
        if let Some(e) = &r.error {
            obj["error"] = serde_json::json!(e);
        }
        obj
    }).collect();

    let output = serde_json::json!({
        "platform": platform,
        "results": items,
    });

    println!("{}", serde_json::to_string(&output).unwrap_or_default());
}

fn output_human(results: &[ToolResult]) {
    eprintln!("\n══════════════════════════════════════════════════════════════════════════");

    let mut scenarios: Vec<(String, String, String)> = results.iter()
        .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
        .collect();
    scenarios.sort();
    scenarios.dedup();

    eprintln!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} Verdict",
        "Scenario", "gzippy", "unpigz", "igzip", "rapidgzip", "gzip", "CV%");
    eprintln!("  {}", "─".repeat(100));

    let mut wins = 0u32;
    let mut losses = 0u32;

    for (ds, arch, thr) in &scenarios {
        let scenario: Vec<&ToolResult> = results.iter()
            .filter(|r| r.dataset == *ds && r.archive == *arch && r.threads == *thr && r.status == "pass")
            .collect();

        let get = |tool: &str| scenario.iter().find(|r| r.tool == tool);
        let fmt = |tool: &str| get(tool).map(|r| format!("{:.1}", r.speed_mbps)).unwrap_or_else(|| "—".into());

        let gzippy = get("gzippy");
        let gzippy_cv = gzippy.map(|r| r.cv).unwrap_or(0.0);
        let best_comp = ["unpigz", "igzip", "rapidgzip", "gzip"].iter()
            .filter_map(|t| get(t))
            .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());

        let verdict = if let (Some(g), Some(b)) = (gzippy, best_comp) {
            let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
            if gap >= 0.0 {
                wins += 1;
                format!("WIN +{:.1}% vs {}", gap, b.tool)
            } else {
                losses += 1;
                format!("LOSS {:.1}% vs {}", gap, b.tool)
            }
        } else {
            "—".into()
        };

        let name = format!("{ds}-{arch} {thr}");
        eprintln!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5.1} {}",
            name, fmt("gzippy"), fmt("unpigz"), fmt("igzip"), fmt("rapidgzip"), fmt("gzip"),
            gzippy_cv * 100.0, verdict);
    }

    let total = wins + losses;
    eprintln!("\n  WINS: {wins}/{total}    LOSSES: {losses}/{total}");
    if losses > 0 {
        eprintln!("\n  Losses:");
        for (ds, arch, thr) in &scenarios {
            let scenario: Vec<&ToolResult> = results.iter()
                .filter(|r| r.dataset == *ds && r.archive == *arch && r.threads == *thr && r.status == "pass")
                .collect();
            let gzippy = scenario.iter().find(|r| r.tool == "gzippy");
            let best = scenario.iter().filter(|r| r.tool != "gzippy")
                .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());
            if let (Some(g), Some(b)) = (gzippy, best) {
                let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
                if gap < 0.0 {
                    eprintln!("    {ds}-{arch} {thr}: gzippy {:.1} vs {} {:.1} ({:+.1}%)",
                        g.speed_mbps, b.tool, b.speed_mbps, gap);
                }
            }
        }
    }
    eprintln!();
}

// ─── A/B comparison (kept as-is) ──────────────────────────────────────────────

pub fn run_ab(ref_a: &str, ref_b: &str, dataset: Option<&str>, threads: Option<&str>) -> Result<(), String> {
    let repo_root = find_repo_root()?;
    let data_dir = find_data_dir(&repo_root)?;
    let datasets_raw = discover_datasets(&data_dir, dataset)?;
    if datasets_raw.is_empty() {
        return Err("No benchmark data found. Run: ./scripts/prepare_benchmark_data.sh".into());
    }

    // Collect all compressed file paths for benchmarking
    let mut bench_files: Vec<(String, PathBuf)> = Vec::new();
    for ds in &datasets_raw {
        for (aname, apath) in &ds.archives {
            bench_files.push((format!("{}-{}", ds.name, aname), apath.clone()));
        }
    }

    let current_branch = git_output(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let is_dirty = !git_output(&["status", "--porcelain"])?.is_empty();
    if is_dirty {
        eprintln!("  Stashing uncommitted changes...");
        git_run(&["stash", "push", "-m", "gzippy-dev bench ab"])?;
    }

    let thread_count: usize = threads.and_then(|t| t.parse().ok()).unwrap_or(1);
    eprintln!("A/B: {ref_a} vs {ref_b}, threads={thread_count}");

    let result = run_ab_inner(ref_a, ref_b, &bench_files, thread_count);

    let _ = git_run(&["checkout", current_branch.trim()]);
    if is_dirty { let _ = git_run(&["stash", "pop"]); }

    result
}

fn run_ab_inner(
    ref_a: &str, ref_b: &str,
    files: &[(String, PathBuf)],
    threads: usize,
) -> Result<(), String> {
    let tmp_dir = std::env::temp_dir().join("gzippy-ab");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let binary_a = tmp_dir.join("gzippy-a");
    let binary_b = tmp_dir.join("gzippy-b");
    let out_file = tmp_dir.join("ab-output.bin");

    eprintln!("  Building {ref_a}...");
    git_run(&["checkout", ref_a])?;
    cargo_build_release()?;
    std::fs::copy("target/release/gzippy", &binary_a).map_err(|e| format!("copy: {e}"))?;

    eprintln!("  Building {ref_b}...");
    git_run(&["checkout", ref_b])?;
    cargo_build_release()?;
    std::fs::copy("target/release/gzippy", &binary_b).map_err(|e| format!("copy: {e}"))?;

    let args_a = tool_decompress_args("gzippy", threads);
    let args_b = tool_decompress_args("gzippy", threads);

    eprintln!("\n  {:<30} {:>10} {:>10} {:>9}", "Dataset", ref_a, ref_b, "Change");
    eprintln!("  {}", "─".repeat(65));

    for (name, path) in files {
        let mut speeds_a = Vec::new();
        let mut speeds_b = Vec::new();
        let ba = binary_a.to_str().unwrap();
        let bb = binary_b.to_str().unwrap();

        for _ in 0..DEFAULT_MAX_TRIALS {
            if let Ok(t) = time_decompress(ba, &args_a, path, &out_file) {
                let size = std::fs::metadata(&out_file).map(|m| m.len()).unwrap_or(0) as f64;
                speeds_a.push(size / t / 1_000_000.0);
            }
            if let Ok(t) = time_decompress(bb, &args_b, path, &out_file) {
                let size = std::fs::metadata(&out_file).map(|m| m.len()).unwrap_or(0) as f64;
                speeds_b.push(size / t / 1_000_000.0);
            }
            if speeds_a.len() >= DEFAULT_MIN_TRIALS as usize && speeds_b.len() >= DEFAULT_MIN_TRIALS as usize {
                let (_, _, _, cv_a) = trimmed_stats(&speeds_a);
                let (_, _, _, cv_b) = trimmed_stats(&speeds_b);
                if cv_a < DEFAULT_TARGET_CV && cv_b < DEFAULT_TARGET_CV { break; }
            }
        }

        if speeds_a.is_empty() || speeds_b.is_empty() {
            eprintln!("  {:<30} (failed)", name);
            continue;
        }

        let (_, mean_a, _, _) = trimmed_stats(&speeds_a);
        let (_, mean_b, _, _) = trimmed_stats(&speeds_b);
        let change = (mean_b / mean_a - 1.0) * 100.0;
        let flag = if change > 3.0 { " FASTER" } else if change < -3.0 { " SLOWER" } else { "" };
        eprintln!("  {:<30} {:>8.1}  {:>8.1}  {:>+7.1}%{}", name, mean_a, mean_b, change, flag);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn git_run(args: &[&str]) -> Result<(), String> {
    let output = Command::new("git").args(args).output().map_err(|e| format!("git: {e}"))?;
    if !output.status.success() {
        return Err(format!("git {}: {}", args[0], String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let output = Command::new("git").args(args).output().map_err(|e| format!("git: {e}"))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn cargo_build_release() -> Result<(), String> {
    let output = Command::new("cargo").args(["build", "--release"]).output()
        .map_err(|e| format!("cargo build: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo build failed:\n{}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_stats_drops_outliers() {
        let (trimmed, _, _, _) = trimmed_stats(&[1.0, 2.0, 2.0, 2.0, 2.0, 100.0]);
        assert!(!trimmed.contains(&1.0));
        assert!(!trimmed.contains(&100.0));
    }
}
