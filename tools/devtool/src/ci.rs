//! CI monitoring: fetch, watch, parse, and analyze GitHub Actions results.

use serde::Deserialize;
use std::io::{self, Write};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_secs(30);
const MAX_WAIT: Duration = Duration::from_secs(45 * 60); // 45 minutes

// ─── Data types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct Run {
    #[serde(rename = "databaseId")]
    id: u64,
    #[serde(rename = "displayTitle")]
    title: String,
    status: String,
    conclusion: Option<String>,
    #[serde(rename = "headBranch")]
    branch: String,
    #[serde(rename = "workflowName")]
    workflow: String,
    #[serde(rename = "createdAt")]
    #[allow(dead_code)]
    created: String,
    #[serde(rename = "updatedAt")]
    #[allow(dead_code)]
    updated: String,
    url: String,
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    #[allow(dead_code)]
    pub job: String,        // "Decompress silesia-gzip Tmax (x86_64)"
    pub tool: String,       // "gzippy", "rapidgzip", etc.
    pub speed_mbps: f64,
    pub trials: u32,
    pub ratio: Option<f64>, // compression ratio (compress only)
    // Parsed from job name:
    pub dataset: String,    // "silesia", "software", "logs"
    pub archive: String,    // "gzip", "bgzf", "pigz"
    pub threads: String,    // "T1", "Tmax"
    pub platform: String,   // "x86_64", "arm64"
    pub mode: String,       // "Decompress", "Compress"
    pub level: String,      // "L1", "L6", "L9" (compress only)
}

// ─── Public commands ─────────────────────────────────────────────────────────

pub fn status(branch: Option<&str>) -> Result<(), String> {
    let runs = fetch_runs(branch)?;
    if runs.is_empty() {
        println!("No CI runs found.");
        return Ok(());
    }

    println!("{:<12} {:<10} {:<10} {:<28} {}", "ID", "Status", "Branch", "Workflow", "Title");
    println!("{}", "─".repeat(90));
    for run in runs.iter().take(10) {
        let conclusion = run.conclusion.as_deref().unwrap_or("");
        let status = if run.status == "completed" {
            conclusion
        } else {
            &run.status
        };
        println!(
            "{:<12} {:<10} {:<10} {:<28} {}",
            run.id,
            status,
            truncate(&run.branch, 10),
            truncate(&run.workflow, 28),
            truncate(&run.title, 50),
        );
    }
    Ok(())
}

pub fn watch(run_id: Option<&str>, branch: Option<&str>) -> Result<(), String> {
    let target = resolve_benchmark_run(run_id, branch)?;
    println!("Watching run {} ({})...", target.id, target.title);
    println!("URL: {}", target.url);

    let start = Instant::now();
    loop {
        let current = fetch_run_by_id(target.id)?;
        if current.status == "completed" {
            println!("\nRun completed: {}", current.conclusion.as_deref().unwrap_or("unknown"));
            return results_for_run(target.id);
        }

        if start.elapsed() > MAX_WAIT {
            return Err(format!("Timed out after {}s waiting for run {}", MAX_WAIT.as_secs(), target.id));
        }

        let elapsed = start.elapsed().as_secs();
        print!("\r  Waiting... ({elapsed}s elapsed, polling every {}s)  ", POLL_INTERVAL.as_secs());
        io::stdout().flush().ok();
        thread::sleep(POLL_INTERVAL);
    }
}

pub fn results(run_id: Option<&str>, branch: Option<&str>) -> Result<(), String> {
    let target = resolve_benchmark_run(run_id, branch)?;
    if target.status != "completed" {
        println!("Run {} is still {}, watching...", target.id, target.status);
        return watch(Some(&target.id.to_string()), None);
    }
    results_for_run(target.id)
}

pub fn gaps(run_id: Option<&str>, branch: Option<&str>) -> Result<(), String> {
    let target = resolve_benchmark_run(run_id, branch)?;
    if target.status != "completed" {
        println!("Run {} is still {}, watching...", target.id, target.status);
        watch(Some(&target.id.to_string()), None)?;
    }
    let benchmarks = parse_run_logs(target.id)?;
    print_gap_analysis(&benchmarks);
    Ok(())
}

// ─── Internal: run resolution ────────────────────────────────────────────────

fn resolve_benchmark_run(run_id: Option<&str>, branch: Option<&str>) -> Result<Run, String> {
    if let Some(id) = run_id {
        let id: u64 = id.parse().map_err(|_| format!("Invalid run ID: {id}"))?;
        return fetch_run_by_id(id);
    }

    let runs = fetch_runs(branch)?;
    runs.into_iter()
        .find(|r| r.workflow == "Benchmarks")
        .ok_or_else(|| "No Benchmarks workflow run found".to_string())
}

fn fetch_runs(branch: Option<&str>) -> Result<Vec<Run>, String> {
    let mut cmd = Command::new("gh");
    cmd.args(["run", "list", "--json",
        "databaseId,displayTitle,status,conclusion,headBranch,workflowName,createdAt,updatedAt,url",
        "--limit", "20"]);
    if let Some(b) = branch {
        cmd.args(["--branch", b]);
    }

    let output = cmd.output().map_err(|e| format!("Failed to run gh: {e}"))?;
    if !output.status.success() {
        return Err(format!("gh failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse gh output: {e}"))
}

fn fetch_run_by_id(id: u64) -> Result<Run, String> {
    let output = Command::new("gh")
        .args(["run", "view", &id.to_string(), "--json",
            "databaseId,displayTitle,status,conclusion,headBranch,workflowName,createdAt,updatedAt,url"])
        .output()
        .map_err(|e| format!("Failed to run gh: {e}"))?;

    if !output.status.success() {
        return Err(format!("gh failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse run: {e}"))
}

// ─── Internal: log parsing ───────────────────────────────────────────────────

fn parse_run_logs(run_id: u64) -> Result<Vec<BenchResult>, String> {
    let output = Command::new("gh")
        .args(["run", "view", &run_id.to_string(), "--log"])
        .output()
        .map_err(|e| format!("Failed to fetch logs: {e}"))?;

    if !output.status.success() {
        return Err(format!("gh log fetch failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let log = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::new();
    let mut current_job = String::new();

    for line in log.lines() {
        // Lines look like:
        // "Decompress silesia-gzip T1 (x86_64)\tUNKNOWN STEP\t...\t  gzippy: 257.8 MB/s, 23 trials"
        if let Some(tab1) = line.find('\t') {
            let job_name = &line[..tab1];
            if job_name != current_job {
                current_job = job_name.to_string();
            }

            // Look for benchmark result lines: "  tool: NNN.N MB/s, N trials"
            if let Some(result_part) = extract_after_timestamp(line) {
                if let Some(br) = parse_result_line(&current_job, result_part.trim()) {
                    results.push(br);
                }
            }
        }
    }

    Ok(results)
}

fn extract_after_timestamp(line: &str) -> Option<&str> {
    // Format: "JobName\tSTEP\t2026-02-19T18:51:43.3866118Z   content here"
    // Find the 2nd tab (start of timestamp), then skip past the "Z " marker
    let mut tabs = 0;
    for (i, c) in line.char_indices() {
        if c == '\t' {
            tabs += 1;
            if tabs == 2 {
                let ts_and_content = &line[i + 1..];
                // Timestamp ends with "Z " or "Z\t" — find the Z followed by whitespace
                if let Some(z_pos) = ts_and_content.find("Z ") {
                    return Some(&ts_and_content[z_pos + 2..]);
                }
                return None;
            }
        }
    }
    None
}

fn parse_result_line(job_name: &str, line: &str) -> Option<BenchResult> {
    // "  gzippy: 257.8 MB/s, 23 trials"
    // "  gzippy: 328.3 MB/s, ratio 0.369, 30 trials"
    let trimmed = line.trim();
    let colon = trimmed.find(':')?;
    let tool = trimmed[..colon].trim().to_string();

    // Must be a known tool name
    if !matches!(tool.as_str(),
        "gzippy" | "pigz" | "unpigz" | "igzip" | "rapidgzip" | "gzip" | "zopfli"
    ) {
        return None;
    }

    let after_colon = trimmed[colon + 1..].trim();

    // Parse "NNN.N MB/s"
    let mbps_idx = after_colon.find("MB/s")?;
    let speed_str = after_colon[..mbps_idx].trim();
    let speed_mbps: f64 = speed_str.parse().ok()?;

    // Parse trials
    let trials: u32 = after_colon
        .split(',')
        .filter_map(|part| {
            let p = part.trim();
            if p.ends_with("trials") {
                p.strip_suffix("trials")?.trim().parse().ok()
            } else {
                None
            }
        })
        .next()
        .unwrap_or(0);

    // Parse ratio if present
    let ratio = after_colon
        .split(',')
        .filter_map(|part| {
            let p = part.trim();
            if p.starts_with("ratio ") {
                p.strip_prefix("ratio ")?.trim().parse().ok()
            } else {
                None
            }
        })
        .next();

    // Parse job name components
    let (mode, dataset, archive, threads, platform, level) = parse_job_name(job_name);

    Some(BenchResult {
        job: job_name.to_string(),
        tool,
        speed_mbps,
        trials,
        ratio,
        dataset,
        archive,
        threads,
        platform,
        mode,
        level,
    })
}

fn parse_job_name(name: &str) -> (String, String, String, String, String, String) {
    let mut mode = String::new();
    let mut dataset = String::new();
    let mut archive = String::new();
    let mut threads = String::new();
    let mut platform = String::new();
    let mut level = String::new();

    // "Decompress silesia-gzip T1 (x86_64)"
    // "Compress silesia L9 Tmax (x86_64)"
    let name = name.trim();

    if name.starts_with("Decompress ") {
        mode = "decompress".to_string();
        let rest = &name[11..];
        // "silesia-gzip T1 (x86_64)"
        if let Some(paren) = rest.rfind('(') {
            platform = rest[paren + 1..].trim_end_matches(')').trim().to_string();
            let mid = rest[..paren].trim();
            let parts: Vec<&str> = mid.split_whitespace().collect();
            if parts.len() >= 2 {
                let archive_full = parts[0]; // "silesia-gzip"
                threads = parts[1].to_string();
                if let Some(dash) = archive_full.find('-') {
                    dataset = archive_full[..dash].to_string();
                    archive = archive_full[dash + 1..].to_string();
                } else {
                    dataset = archive_full.to_string();
                }
            }
        }
    } else if name.starts_with("Compress ") {
        mode = "compress".to_string();
        let rest = &name[9..];
        // "silesia L9 Tmax (x86_64)"
        if let Some(paren) = rest.rfind('(') {
            platform = rest[paren + 1..].trim_end_matches(')').trim().to_string();
            let mid = rest[..paren].trim();
            let parts: Vec<&str> = mid.split_whitespace().collect();
            if parts.len() >= 3 {
                dataset = parts[0].to_string();
                level = parts[1].to_string();
                threads = parts[2].to_string();
            }
        }
    }

    (mode, dataset, archive, threads, platform, level)
}

// ─── Internal: display ───────────────────────────────────────────────────────

fn results_for_run(run_id: u64) -> Result<(), String> {
    let benchmarks = parse_run_logs(run_id)?;
    if benchmarks.is_empty() {
        println!("No benchmark results found in run {run_id}.");
        return Ok(());
    }

    print_results_table(&benchmarks);
    println!();
    print_gap_analysis(&benchmarks);
    Ok(())
}

fn print_results_table(benchmarks: &[BenchResult]) {
    // Group by (mode, dataset, archive, level, threads, platform)
    let mut groups: std::collections::BTreeMap<String, Vec<&BenchResult>> =
        std::collections::BTreeMap::new();

    for b in benchmarks {
        let key = if b.mode == "compress" {
            format!("{} {} {} {} {}", b.mode, b.dataset, b.level, b.threads, b.platform)
        } else {
            format!("{} {}-{} {} {}", b.mode, b.dataset, b.archive, b.threads, b.platform)
        };
        groups.entry(key).or_default().push(b);
    }

    for (group, tools) in &groups {
        println!("\n  {group}");
        println!("  {}", "─".repeat(60));
        for t in tools {
            let ratio_str = t.ratio.map_or(String::new(), |r| format!(" ratio={r:.3}"));
            println!(
                "    {:<12} {:>8.1} MB/s  ({} trials){ratio_str}",
                t.tool, t.speed_mbps, t.trials
            );
        }
    }
}

fn print_gap_analysis(benchmarks: &[BenchResult]) {
    // Find all cases where gzippy is slower than a competitor
    #[derive(Debug)]
    struct Gap {
        scenario: String,
        competitor: String,
        gzippy_speed: f64,
        competitor_speed: f64,
        gap_pct: f64,
    }

    let mut gaps = Vec::new();
    let mut wins = Vec::new();

    // Group by scenario
    let mut by_scenario: std::collections::BTreeMap<String, Vec<&BenchResult>> =
        std::collections::BTreeMap::new();

    for b in benchmarks {
        let key = if b.mode == "compress" {
            format!("{} {} {} {} {}", b.mode, b.dataset, b.level, b.threads, b.platform)
        } else {
            format!("{} {}-{} {} {}", b.mode, b.dataset, b.archive, b.threads, b.platform)
        };
        by_scenario.entry(key).or_default().push(b);
    }

    for (scenario, tools) in &by_scenario {
        let gzippy = tools.iter().find(|t| t.tool == "gzippy");
        let gzippy = match gzippy {
            Some(g) => g,
            None => continue,
        };

        for t in tools {
            if t.tool == "gzippy" || t.tool == "gzip" || t.tool == "zopfli" {
                continue;
            }
            let competitor = if t.tool == "unpigz" { "pigz" } else { &t.tool };
            let gap_pct = (gzippy.speed_mbps / t.speed_mbps - 1.0) * 100.0;

            if gap_pct < 0.0 {
                gaps.push(Gap {
                    scenario: scenario.clone(),
                    competitor: competitor.to_string(),
                    gzippy_speed: gzippy.speed_mbps,
                    competitor_speed: t.speed_mbps,
                    gap_pct,
                });
            } else {
                wins.push(Gap {
                    scenario: scenario.clone(),
                    competitor: competitor.to_string(),
                    gzippy_speed: gzippy.speed_mbps,
                    competitor_speed: t.speed_mbps,
                    gap_pct,
                });
            }
        }
    }

    gaps.sort_by(|a, b| a.gap_pct.partial_cmp(&b.gap_pct).unwrap());
    wins.sort_by(|a, b| b.gap_pct.partial_cmp(&a.gap_pct).unwrap());

    let total = gaps.len() + wins.len();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  PERFORMANCE GAP ANALYSIS — {} wins, {} gaps out of {} comparisons",
             wins.len(), gaps.len(), total);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    if !gaps.is_empty() {
        println!("\n  GAPS (gzippy is SLOWER):");
        println!("  {:<55} {:<12} {:>8} {:>8} {:>7}",
                 "Scenario", "Competitor", "gzippy", "them", "gap");
        println!("  {}", "─".repeat(92));
        for g in &gaps {
            println!(
                "  {:<55} {:<12} {:>7.1} {:>7.1} {:>+6.1}%",
                g.scenario, g.competitor, g.gzippy_speed, g.competitor_speed, g.gap_pct,
            );
        }
    }

    if !wins.is_empty() {
        println!("\n  WINS (gzippy is FASTER):");
        println!("  {:<55} {:<12} {:>8} {:>8} {:>7}",
                 "Scenario", "Competitor", "gzippy", "them", "lead");
        println!("  {}", "─".repeat(92));
        for w in wins.iter().take(15) {
            println!(
                "  {:<55} {:<12} {:>7.1} {:>7.1} {:>+6.1}%",
                w.scenario, w.competitor, w.gzippy_speed, w.competitor_speed, w.gap_pct,
            );
        }
        if wins.len() > 15 {
            println!("  ... and {} more wins", wins.len() - 15);
        }
    }

    // Priority ranking
    if !gaps.is_empty() {
        println!("\n  PRIORITY ACTIONS:");
        println!("  {}", "─".repeat(72));
        for (i, g) in gaps.iter().take(5).enumerate() {
            let root_cause = diagnose_gap(&g.scenario, &g.competitor);
            println!("  {}. [{:>+5.1}%] {} vs {}", i + 1, g.gap_pct, g.scenario, g.competitor);
            println!("     Root cause: {}", root_cause);
        }
    }
}

fn diagnose_gap(scenario: &str, competitor: &str) -> &'static str {
    if scenario.contains("Tmax") && competitor == "rapidgzip" && !scenario.contains("bgzf") {
        "No parallel single-member decompression. rapidgzip speculatively parallelizes."
    } else if scenario.contains("bgzf") && scenario.contains("T1") {
        "BGZF T1 overhead from parallel detection. Consider fast-path for T1."
    } else if scenario.contains("Tmax") && scenario.contains("bgzf") {
        "BGZF Tmax contention or decompressor allocation overhead."
    } else if scenario.contains("Tmax") && scenario.contains("pigz") {
        "Multi-member parallel not matching rapidgzip's pipelining."
    } else if scenario.contains("compress") && competitor == "igzip" {
        "ISA-L uses hand-tuned SIMD assembly for compression."
    } else if scenario.contains("T1") && competitor == "igzip" {
        "ISA-L inflate uses hand-tuned x86 assembly."
    } else {
        "Investigate with local profiling."
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
