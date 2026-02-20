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

    println!("{:<12} {:<10} {:<10} {:<28} Title", "ID", "Status", "Branch", "Workflow");
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

pub fn triage(run_id: Option<&str>, branch: Option<&str>) -> Result<(), String> {
    let target = resolve_benchmark_run(run_id, branch)?;
    if target.status != "completed" {
        println!("Run {} is still {}, watching...", target.id, target.status);
        watch(Some(&target.id.to_string()), None)?;
    }
    let benchmarks = parse_run_logs(target.id)?;
    print_triage(&benchmarks);
    Ok(())
}

pub fn compare(run_a: &str, run_b: &str) -> Result<(), String> {
    let id_a: u64 = run_a
        .parse()
        .map_err(|_| format!("Invalid run ID: {run_a}"))?;
    let id_b: u64 = run_b
        .parse()
        .map_err(|_| format!("Invalid run ID: {run_b}"))?;

    let run_a_info = fetch_run_by_id(id_a)?;
    let run_b_info = fetch_run_by_id(id_b)?;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  CI COMPARISON                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "  A: {} ({}, {})",
        id_a,
        run_a_info.branch,
        run_a_info.title.chars().take(40).collect::<String>()
    );
    println!(
        "  B: {} ({}, {})",
        id_b,
        run_b_info.branch,
        run_b_info.title.chars().take(40).collect::<String>()
    );

    let bench_a = parse_run_logs(id_a)?;
    let bench_b = parse_run_logs(id_b)?;

    // Build lookup: scenario → gzippy speed
    let mut map_a: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut map_b: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();

    for b in &bench_a {
        if b.tool == "gzippy" {
            let key = scenario_key(b);
            map_a.insert(key, b.speed_mbps);
        }
    }
    for b in &bench_b {
        if b.tool == "gzippy" {
            let key = scenario_key(b);
            map_b.insert(key, b.speed_mbps);
        }
    }

    // Find all common scenarios and compare
    let mut diffs: Vec<(String, f64, f64, f64)> = Vec::new();
    for (key, &speed_a) in &map_a {
        if let Some(&speed_b) = map_b.get(key) {
            let change_pct = (speed_b / speed_a - 1.0) * 100.0;
            diffs.push((key.clone(), speed_a, speed_b, change_pct));
        }
    }

    diffs.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    println!(
        "\n  {:<55} {:>8} {:>8} {:>8}",
        "Scenario", "Run A", "Run B", "Change"
    );
    println!("  {}", "─".repeat(85));

    let significant: Vec<_> = diffs
        .iter()
        .filter(|(_, _, _, pct)| pct.abs() > 3.0)
        .collect();

    if significant.is_empty() {
        println!("  No significant changes (>3%) detected.");
        println!("  All {} scenarios within noise margin.", diffs.len());
    } else {
        println!("  Significant changes (>3%):");
        for (key, a, b, pct) in &significant {
            let flag = if pct.abs() > 10.0 { " <<<" } else { "" };
            println!(
                "  {:<55} {:>7.1} {:>7.1} {:>+7.1}%{}",
                key, a, b, pct, flag
            );
        }

        let noise: Vec<_> = diffs
            .iter()
            .filter(|(_, _, _, pct)| pct.abs() <= 3.0)
            .collect();
        if !noise.is_empty() {
            println!(
                "\n  {} scenarios within noise margin (<=3%)",
                noise.len()
            );
        }
    }

    Ok(())
}

pub(crate) fn scenario_key(b: &BenchResult) -> String {
    if b.mode == "compress" {
        format!(
            "{} {} {} {} {}",
            b.mode, b.dataset, b.level, b.threads, b.platform
        )
    } else {
        format!(
            "{} {}-{} {} {}",
            b.mode, b.dataset, b.archive, b.threads, b.platform
        )
    }
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

pub(crate) fn extract_after_timestamp(line: &str) -> Option<&str> {
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

pub(crate) fn parse_result_line(job_name: &str, line: &str) -> Option<BenchResult> {
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

pub(crate) fn parse_job_name(name: &str) -> (String, String, String, String, String, String) {
    let mut mode = String::new();
    let mut dataset = String::new();
    let mut archive = String::new();
    let mut threads = String::new();
    let mut platform = String::new();
    let mut level = String::new();

    // "Decompress silesia-gzip T1 (x86_64)"
    // "Compress silesia L9 Tmax (x86_64)"
    let name = name.trim();

    if let Some(rest) = name.strip_prefix("Decompress ") {
        mode = "decompress".to_string();
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
    } else if let Some(rest) = name.strip_prefix("Compress ") {
        mode = "compress".to_string();
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

// ─── Triage: categorized gap analysis ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum GapCategory {
    Architecture,
    Simd,
    Actionable,
    Noise,
}

impl GapCategory {
    fn label(&self) -> &'static str {
        match self {
            GapCategory::Architecture => "ARCHITECTURE",
            GapCategory::Simd => "SIMD",
            GapCategory::Actionable => "ACTIONABLE",
            GapCategory::Noise => "NOISE",
        }
    }
    fn description(&self) -> &'static str {
        match self {
            GapCategory::Architecture =>
                "Needs parallel single-member decompression (rapidgzip-style pipeline)",
            GapCategory::Simd =>
                "Competitor uses hand-tuned SIMD/AVX assembly we can't match in pure Rust",
            GapCategory::Actionable =>
                "Can potentially be closed with code-level optimizations",
            GapCategory::Noise =>
                "Within measurement variance (<2%), not worth investigating",
        }
    }
    fn effort(&self) -> &'static str {
        match self {
            GapCategory::Architecture => "months — dedicated block-finder + decoder thread pool",
            GapCategory::Simd => "won't fix — requires hand-tuned asm to match ISA-L",
            GapCategory::Actionable => "days — profiling + targeted code changes",
            GapCategory::Noise => "none — measurement artifact",
        }
    }
}

struct TriageGap {
    scenario: String,
    competitor: String,
    gzippy_speed: f64,
    competitor_speed: f64,
    gap_pct: f64,
    category: GapCategory,
    action: String,
}

pub(crate) fn categorize_gap(scenario: &str, competitor: &str, gap_pct: f64) -> (GapCategory, String) {
    let is_compress = scenario.starts_with("compress ");
    let is_decompress = scenario.starts_with("decompress ");
    let is_bgzf = scenario.contains("bgzf");
    let is_tmax = scenario.contains("Tmax");
    let is_t1 = scenario.contains("T1");
    let is_single_member = !is_bgzf
        && (scenario.contains("-gzip") || scenario.contains("-pigz"));

    if gap_pct.abs() < 2.0 {
        return (GapCategory::Noise, "Within noise margin".to_string());
    }

    // Tmax decompression on single-member archives = needs parallel single-member
    if is_decompress && is_tmax && is_single_member {
        return (GapCategory::Architecture,
            format!("Single-member parallel decompression needed. {} has a pipeline \
                     architecture (block-finder + decoder threads) that we lack. \
                     Gap: {:.1}%", competitor, gap_pct.abs())
        );
    }

    // Compression vs igzip — ISA-L has hand-tuned SIMD assembly
    if is_compress && competitor == "igzip" {
        return (GapCategory::Simd,
            format!("ISA-L uses hand-tuned AVX2/AVX-512 assembly for L1 compression. \
                     Gap is {:.1}% — only closable with SIMD intrinsics.", gap_pct.abs()));
    }

    // T1 decompression vs igzip — their inflate is optimized C
    if is_decompress && is_t1 && competitor == "igzip" {
        return (GapCategory::Actionable,
            "igzip inflate uses optimized C with SIMD. Profile with `gzippy-dev instrument` \
             to find bottleneck — likely buffer allocation or IO path overhead."
            .to_string());
    }

    // BGZF T1 gaps — PR #52 should address
    if is_decompress && is_bgzf && is_t1 {
        return (GapCategory::Actionable,
            "BGZF T1 path overhead. PR #52 routes BGZF through optimized path at T1. \
             If gap persists after PR #52, profile block parsing and buffer allocation."
            .to_string());
    }

    // BGZF Tmax gaps — thread pool overhead
    if is_decompress && is_bgzf && is_tmax {
        return (GapCategory::Actionable,
            "BGZF Tmax gap. Check thread pool overhead, lock contention, or \
             decompressor allocation cost. Profile with `gzippy-dev instrument --threads 4`."
            .to_string());
    }

    // Compression vs pigz — should be winning
    if is_compress && competitor == "pigz" {
        return (GapCategory::Actionable,
            "Should be winning vs pigz. Check if ratio-probe or block sizing \
             is suboptimal for this dataset/level combination."
            .to_string());
    }

    // Generic small gaps
    if gap_pct.abs() < 5.0 {
        return (GapCategory::Actionable,
            format!("Small gap ({:.1}%). Profile with `gzippy-dev instrument` \
                     and check for unnecessary allocations or IO overhead.", gap_pct.abs()));
    }

    (GapCategory::Actionable,
     format!("Significant gap ({:.1}%). Needs profiling to identify root cause. \
              Run: gzippy-dev instrument <file> --threads <N>", gap_pct.abs()))
}

fn print_triage(benchmarks: &[BenchResult]) {
    let mut triage_gaps: Vec<TriageGap> = Vec::new();
    let mut total_wins = 0u32;
    let mut total_comparisons = 0u32;

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
        let gzippy = match tools.iter().find(|t| t.tool == "gzippy") {
            Some(g) => g,
            None => continue,
        };

        for t in tools {
            if t.tool == "gzippy" || t.tool == "gzip" || t.tool == "zopfli" {
                continue;
            }
            total_comparisons += 1;
            let competitor = if t.tool == "unpigz" { "pigz" } else { &t.tool };
            let gap_pct = (gzippy.speed_mbps / t.speed_mbps - 1.0) * 100.0;

            if gap_pct >= 0.0 {
                total_wins += 1;
                continue;
            }

            let (category, action) = categorize_gap(scenario, competitor, gap_pct);
            triage_gaps.push(TriageGap {
                scenario: scenario.clone(),
                competitor: competitor.to_string(),
                gzippy_speed: gzippy.speed_mbps,
                competitor_speed: t.speed_mbps,
                gap_pct,
                category,
                action,
            });
        }
    }

    let total_gaps = triage_gaps.len() as u32;
    let win_rate = total_wins as f64 / total_comparisons as f64 * 100.0;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  GAP TRIAGE — {} wins / {} gaps / {} total ({:.0}% win rate)",
             total_wins, total_gaps, total_comparisons, win_rate);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Group by category
    let mut by_cat: std::collections::BTreeMap<GapCategory, Vec<&TriageGap>> =
        std::collections::BTreeMap::new();
    for g in &triage_gaps {
        by_cat.entry(g.category).or_default().push(g);
    }

    // Summary table
    println!("\n  CATEGORY SUMMARY");
    println!("  {:<16} {:>5} {:>10} {:>10}  Effort", "Category", "Gaps", "Worst", "Avg");
    println!("  {}", "─".repeat(78));

    for cat in &[GapCategory::Architecture, GapCategory::Simd, GapCategory::Actionable, GapCategory::Noise] {
        if let Some(gaps) = by_cat.get(cat) {
            let worst = gaps.iter().map(|g| g.gap_pct).fold(f64::MAX, f64::min);
            let avg = gaps.iter().map(|g| g.gap_pct).sum::<f64>() / gaps.len() as f64;
            println!("  {:<16} {:>5} {:>+9.1}% {:>+9.1}%  {}",
                     cat.label(), gaps.len(), worst, avg, cat.effort());
        }
    }

    // Win rate projection
    println!("\n  WIN RATE PROJECTIONS");
    println!("  {}", "─".repeat(78));
    println!("  Current:                                                  {:>5.1}% ({}/{})",
             win_rate, total_wins, total_comparisons);

    let noise_count = by_cat.get(&GapCategory::Noise).map_or(0, |v| v.len()) as u32;
    let actionable_count = by_cat.get(&GapCategory::Actionable).map_or(0, |v| v.len()) as u32;
    let arch_count = by_cat.get(&GapCategory::Architecture).map_or(0, |v| v.len()) as u32;
    let simd_count = by_cat.get(&GapCategory::Simd).map_or(0, |v| v.len()) as u32;

    if noise_count > 0 {
        let proj = (total_wins + noise_count) as f64 / total_comparisons as f64 * 100.0;
        println!("  If NOISE gaps are wins (measurement variance):           {:>5.1}% ({}/{})",
                 proj, total_wins + noise_count, total_comparisons);
    }
    if actionable_count > 0 {
        let proj = (total_wins + noise_count + actionable_count) as f64 / total_comparisons as f64 * 100.0;
        println!("  If ACTIONABLE gaps are closed:                           {:>5.1}% ({}/{})",
                 proj, total_wins + noise_count + actionable_count, total_comparisons);
    }
    if arch_count > 0 {
        let proj = (total_wins + noise_count + actionable_count + arch_count) as f64 / total_comparisons as f64 * 100.0;
        println!("  If ARCHITECTURE gaps are closed (parallel single-member): {:>5.1}% ({}/{})",
                 proj, total_wins + noise_count + actionable_count + arch_count, total_comparisons);
    }
    if simd_count > 0 {
        println!("  If ALL gaps closed (including SIMD — unrealistic):       100.0% ({}/{})",
                 total_comparisons, total_comparisons);
    }

    // Detailed gaps by category
    for cat in &[GapCategory::Architecture, GapCategory::Simd, GapCategory::Actionable, GapCategory::Noise] {
        if let Some(gaps) = by_cat.get(cat) {
            println!("\n  ── {} ({}) ──", cat.label(), cat.description());
            let mut sorted: Vec<&&TriageGap> = gaps.iter().collect();
            sorted.sort_by(|a, b| a.gap_pct.partial_cmp(&b.gap_pct).unwrap());

            for g in &sorted {
                println!("    [{:>+5.1}%] {} vs {}  ({:.1} vs {:.1} MB/s)",
                         g.gap_pct, g.scenario, g.competitor,
                         g.gzippy_speed, g.competitor_speed);
            }

            if *cat != GapCategory::Noise {
                // Show action for first gap as representative
                if let Some(first) = sorted.first() {
                    println!("    Action: {}", first.action);
                }
            }
        }
    }

    // Bottom-line recommendation
    println!("\n  ── RECOMMENDED NEXT STEPS ──");
    println!("  {}", "─".repeat(72));

    if actionable_count > 0 {
        println!("  1. Close ACTIONABLE gaps ({} scenarios, {:.0}% win rate uplift):",
                 actionable_count,
                 actionable_count as f64 / total_comparisons as f64 * 100.0);

        if let Some(gaps) = by_cat.get(&GapCategory::Actionable) {
            let mut sorted: Vec<&&TriageGap> = gaps.iter().collect();
            sorted.sort_by(|a, b| a.gap_pct.partial_cmp(&b.gap_pct).unwrap());
            for g in sorted.iter().take(3) {
                println!("     - [{:>+5.1}%] {}: {}",
                         g.gap_pct, g.scenario,
                         g.action.chars().take(70).collect::<String>());
            }
        }
    }

    if arch_count > 0 {
        println!("  2. Parallel single-member decompression ({} scenarios, largest gaps):", arch_count);
        println!("     Implement rapidgzip-style pipeline with dedicated block-finder threads.");
        println!("     This is the only path to beating rapidgzip on Tmax single-member files.");
    }

    println!("  3. Run `gzippy-dev bench ab HEAD~1 HEAD` to verify changes help locally.");
    println!("  4. Run `gzippy-dev ci gaps` after CI completes to measure impact.");
}

// ─── History: track trends across CI runs ───────────────────────────────────

pub fn history(branch: Option<&str>, limit: usize) -> Result<(), String> {
    let runs = fetch_runs(branch)?;
    let benchmark_runs: Vec<_> = runs.into_iter()
        .filter(|r| r.workflow == "Benchmarks" && r.status == "completed")
        .take(limit)
        .collect();

    if benchmark_runs.is_empty() {
        println!("No completed benchmark runs found.");
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  PERFORMANCE HISTORY — last {} runs", benchmark_runs.len());
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let mut all_run_data: Vec<(u64, String, u32, u32, u32)> = Vec::new();

    for run in &benchmark_runs {
        let benchmarks = match parse_run_logs(run.id) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let (wins, gaps, total) = count_wins_gaps(&benchmarks);
        all_run_data.push((run.id, run.branch.clone(), wins, gaps, total));
    }

    println!("\n  {:<14} {:<20} {:>6} {:>6} {:>6} {:>8}",
             "Run ID", "Branch", "Wins", "Gaps", "Total", "Win %");
    println!("  {}", "─".repeat(68));

    for (id, branch, wins, gaps, total) in &all_run_data {
        let pct = if *total > 0 { *wins as f64 / *total as f64 * 100.0 } else { 0.0 };
        let trend = if pct >= 90.0 { " ★" }
            else if pct >= 80.0 { " ↑" }
            else { "" };
        println!("  {:<14} {:<20} {:>6} {:>6} {:>6} {:>7.1}%{}",
                 id, truncate(branch, 20), wins, gaps, total, pct, trend);
    }

    if all_run_data.len() >= 2 {
        let latest = &all_run_data[0];
        let prev = &all_run_data[1];
        let latest_pct = latest.2 as f64 / latest.4 as f64 * 100.0;
        let prev_pct = prev.2 as f64 / prev.4 as f64 * 100.0;
        let delta = latest_pct - prev_pct;
        println!("\n  Trend: {:>+.1}% vs previous run ({} → {})",
                 delta, prev.0, latest.0);
    }

    Ok(())
}

fn count_wins_gaps(benchmarks: &[BenchResult]) -> (u32, u32, u32) {
    let mut wins = 0u32;
    let mut gaps = 0u32;

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

    for tools in by_scenario.values() {
        let gzippy = match tools.iter().find(|t| t.tool == "gzippy") {
            Some(g) => g,
            None => continue,
        };
        for t in tools {
            if t.tool == "gzippy" || t.tool == "gzip" || t.tool == "zopfli" {
                continue;
            }
            let gap_pct = (gzippy.speed_mbps / t.speed_mbps - 1.0) * 100.0;
            if gap_pct >= 0.0 { wins += 1; } else { gaps += 1; }
        }
    }

    (wins, gaps, wins + gaps)
}

// ─── vs-main: auto-compare current branch against main ──────────────────────

pub fn vs_main(branch: Option<&str>) -> Result<(), String> {
    let current = current_git_branch()?;
    let branch_name = branch.unwrap_or(current.trim());

    if branch_name == "main" || branch_name == "master" {
        return Err("Already on main — nothing to compare against.".to_string());
    }

    println!("  Finding latest Benchmarks runs...");
    let branch_run = find_latest_benchmark_run(Some(branch_name))?;
    let main_run = find_latest_benchmark_run(Some("main"))?;

    if branch_run.status != "completed" {
        println!("  Branch run {} is {}, watching...", branch_run.id, branch_run.status);
        watch(Some(&branch_run.id.to_string()), None)?;
    }

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  {} vs main", branch_name);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("  Branch: {} (run {})", branch_name, branch_run.id);
    println!("  Main:   main (run {})", main_run.id);
    println!("  URL:    {}", branch_run.url);

    let bench_branch = parse_run_logs(branch_run.id)?;
    let bench_main = parse_run_logs(main_run.id)?;

    // Build lookup maps
    let mut map_main: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut map_branch: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();

    for b in &bench_main {
        if b.tool == "gzippy" {
            map_main.insert(scenario_key(b), b.speed_mbps);
        }
    }
    for b in &bench_branch {
        if b.tool == "gzippy" {
            map_branch.insert(scenario_key(b), b.speed_mbps);
        }
    }

    let mut diffs: Vec<(String, f64, f64, f64)> = Vec::new();
    for (key, &speed_main) in &map_main {
        if let Some(&speed_branch) = map_branch.get(key) {
            let change_pct = (speed_branch / speed_main - 1.0) * 100.0;
            diffs.push((key.clone(), speed_main, speed_branch, change_pct));
        }
    }
    diffs.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    let regressions: Vec<_> = diffs.iter().filter(|(_, _, _, p)| *p < -3.0).collect();
    let improvements: Vec<_> = diffs.iter().filter(|(_, _, _, p)| *p > 3.0).collect();
    let neutral = diffs.len() - regressions.len() - improvements.len();

    if !regressions.is_empty() {
        println!("\n  REGRESSIONS (>3% slower than main):");
        println!("  {:<55} {:>8} {:>8} {:>8}", "Scenario", "main", "branch", "change");
        println!("  {}", "─".repeat(85));
        for (key, main_s, branch_s, pct) in &regressions {
            let flag = if *pct < -10.0 { " <<<" } else { "" };
            println!("  {:<55} {:>7.1} {:>7.1} {:>+7.1}%{}",
                     key, main_s, branch_s, pct, flag);
        }
    }

    if !improvements.is_empty() {
        println!("\n  IMPROVEMENTS (>3% faster than main):");
        println!("  {:<55} {:>8} {:>8} {:>8}", "Scenario", "main", "branch", "change");
        println!("  {}", "─".repeat(85));
        for (key, main_s, branch_s, pct) in improvements.iter().rev() {
            let flag = if *pct > 10.0 { " <<<" } else { "" };
            println!("  {:<55} {:>7.1} {:>7.1} {:>+7.1}%{}",
                     key, main_s, branch_s, pct, flag);
        }
    }

    println!("\n  Summary: {} regressions, {} improvements, {} neutral",
             regressions.len(), improvements.len(), neutral);

    // Also show triage for the branch
    println!();
    print_triage(&bench_branch);

    Ok(())
}

pub fn push_and_watch() -> Result<(), String> {
    let branch = current_git_branch()?;
    let branch = branch.trim();

    if branch == "main" || branch == "master" {
        return Err("Won't push directly to main. Create a branch first.".to_string());
    }

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  PUSH & WATCH — {}", branch);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Push
    println!("\n  Pushing {} to origin...", branch);
    let output = Command::new("git")
        .args(["push", "-u", "origin", "HEAD"])
        .output()
        .map_err(|e| format!("git push failed: {e}"))?;

    if !output.status.success() {
        return Err(format!("git push failed: {}",
                          String::from_utf8_lossy(&output.stderr)));
    }
    println!("  Pushed.");

    // Wait for the CI run to appear
    println!("  Waiting for Benchmarks run to start...");
    let mut attempts = 0;
    let run = loop {
        thread::sleep(Duration::from_secs(10));
        attempts += 1;
        let runs = fetch_runs(Some(branch))?;
        if let Some(r) = runs.into_iter().find(|r| r.workflow == "Benchmarks") {
            if r.status != "completed" || attempts <= 2 {
                break r;
            }
        }
        if attempts > 18 {
            return Err("No Benchmarks run appeared after 3 minutes.".to_string());
        }
        print!(".");
        io::stdout().flush().ok();
    };

    println!("\n  Run {} started: {}", run.id, run.url);

    // Watch it
    watch(Some(&run.id.to_string()), None)?;

    // Auto-triage + vs-main
    println!();
    vs_main(Some(branch))
}

fn find_latest_benchmark_run(branch: Option<&str>) -> Result<Run, String> {
    let runs = fetch_runs(branch)?;
    runs.into_iter()
        .find(|r| r.workflow == "Benchmarks")
        .ok_or_else(|| format!("No Benchmarks run found for branch {:?}", branch))
}

fn current_git_branch() -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map_err(|e| format!("git failed: {e}"))?;
    if !output.status.success() {
        return Err("Not in a git repository".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_job_name ──

    #[test]
    fn test_parse_decompress_job() {
        let (mode, dataset, archive, threads, platform, level) =
            parse_job_name("Decompress silesia-gzip T1 (x86_64)");
        assert_eq!(mode, "decompress");
        assert_eq!(dataset, "silesia");
        assert_eq!(archive, "gzip");
        assert_eq!(threads, "T1");
        assert_eq!(platform, "x86_64");
        assert_eq!(level, "");
    }

    #[test]
    fn test_parse_decompress_bgzf() {
        let (mode, dataset, archive, threads, platform, _) =
            parse_job_name("Decompress logs-bgzf Tmax (arm64)");
        assert_eq!(mode, "decompress");
        assert_eq!(dataset, "logs");
        assert_eq!(archive, "bgzf");
        assert_eq!(threads, "Tmax");
        assert_eq!(platform, "arm64");
    }

    #[test]
    fn test_parse_compress_job() {
        let (mode, dataset, archive, threads, platform, level) =
            parse_job_name("Compress silesia L9 Tmax (x86_64)");
        assert_eq!(mode, "compress");
        assert_eq!(dataset, "silesia");
        assert_eq!(archive, "");
        assert_eq!(threads, "Tmax");
        assert_eq!(platform, "x86_64");
        assert_eq!(level, "L9");
    }

    #[test]
    fn test_parse_compress_l1() {
        let (mode, dataset, _, threads, platform, level) =
            parse_job_name("Compress software L1 T1 (arm64)");
        assert_eq!(mode, "compress");
        assert_eq!(dataset, "software");
        assert_eq!(threads, "T1");
        assert_eq!(platform, "arm64");
        assert_eq!(level, "L1");
    }

    #[test]
    fn test_parse_pigz_job() {
        let (mode, dataset, archive, threads, platform, _) =
            parse_job_name("Decompress silesia-pigz Tmax (x86_64)");
        assert_eq!(mode, "decompress");
        assert_eq!(dataset, "silesia");
        assert_eq!(archive, "pigz");
        assert_eq!(threads, "Tmax");
        assert_eq!(platform, "x86_64");
    }

    // ── parse_result_line ──

    #[test]
    fn test_parse_simple_result() {
        let br = parse_result_line(
            "Decompress silesia-gzip T1 (x86_64)",
            "  gzippy: 257.8 MB/s, 23 trials",
        ).unwrap();
        assert_eq!(br.tool, "gzippy");
        assert!((br.speed_mbps - 257.8).abs() < 0.01);
        assert_eq!(br.trials, 23);
        assert!(br.ratio.is_none());
        assert_eq!(br.mode, "decompress");
        assert_eq!(br.dataset, "silesia");
        assert_eq!(br.archive, "gzip");
    }

    #[test]
    fn test_parse_result_with_ratio() {
        let br = parse_result_line(
            "Compress software L1 T1 (x86_64)",
            "  gzippy: 328.3 MB/s, ratio 0.369, 30 trials",
        ).unwrap();
        assert_eq!(br.tool, "gzippy");
        assert!((br.speed_mbps - 328.3).abs() < 0.01);
        assert_eq!(br.trials, 30);
        assert!((br.ratio.unwrap() - 0.369).abs() < 0.001);
        assert_eq!(br.mode, "compress");
        assert_eq!(br.level, "L1");
    }

    #[test]
    fn test_parse_competitor_result() {
        let br = parse_result_line(
            "Decompress logs-bgzf T1 (arm64)",
            "  rapidgzip: 148.1 MB/s, 15 trials",
        ).unwrap();
        assert_eq!(br.tool, "rapidgzip");
        assert!((br.speed_mbps - 148.1).abs() < 0.01);
        assert_eq!(br.trials, 15);
    }

    #[test]
    fn test_parse_unpigz_result() {
        let br = parse_result_line(
            "Decompress silesia-gzip T1 (x86_64)",
            "  unpigz: 200.5 MB/s, 10 trials",
        ).unwrap();
        assert_eq!(br.tool, "unpigz");
    }

    #[test]
    fn test_parse_unknown_tool_returns_none() {
        let br = parse_result_line(
            "Decompress silesia-gzip T1 (x86_64)",
            "  Step 3: Run benchmark",
        );
        assert!(br.is_none());
    }

    #[test]
    fn test_parse_no_mbps_returns_none() {
        let br = parse_result_line(
            "Decompress silesia-gzip T1 (x86_64)",
            "  gzippy: running...",
        );
        assert!(br.is_none());
    }

    // ── extract_after_timestamp ──

    #[test]
    fn test_extract_timestamp() {
        let line = "Decompress silesia-gzip T1 (x86_64)\tRun benchmarks\t2026-02-19T18:51:43.3866118Z   gzippy: 257.8 MB/s, 23 trials";
        let result = extract_after_timestamp(line).unwrap();
        assert_eq!(result, "  gzippy: 257.8 MB/s, 23 trials");
    }

    #[test]
    fn test_extract_no_tabs_returns_none() {
        assert!(extract_after_timestamp("no tabs here").is_none());
    }

    #[test]
    fn test_extract_one_tab_returns_none() {
        assert!(extract_after_timestamp("one\ttab").is_none());
    }

    // ── scenario_key ──

    #[test]
    fn test_scenario_key_decompress() {
        let br = BenchResult {
            job: String::new(), tool: "gzippy".into(), speed_mbps: 100.0,
            trials: 10, ratio: None, dataset: "silesia".into(),
            archive: "gzip".into(), threads: "T1".into(),
            platform: "x86_64".into(), mode: "decompress".into(), level: String::new(),
        };
        assert_eq!(scenario_key(&br), "decompress silesia-gzip T1 x86_64");
    }

    #[test]
    fn test_scenario_key_compress() {
        let br = BenchResult {
            job: String::new(), tool: "gzippy".into(), speed_mbps: 100.0,
            trials: 10, ratio: Some(0.35), dataset: "logs".into(),
            archive: String::new(), threads: "Tmax".into(),
            platform: "arm64".into(), mode: "compress".into(), level: "L6".into(),
        };
        assert_eq!(scenario_key(&br), "compress logs L6 Tmax arm64");
    }

    // ── categorize_gap ──

    #[test]
    fn test_categorize_noise() {
        let (cat, _) = categorize_gap("decompress silesia-gzip T1 x86_64", "igzip", -1.5);
        assert_eq!(cat, GapCategory::Noise);
    }

    #[test]
    fn test_categorize_architecture_gzip_tmax() {
        let (cat, _) = categorize_gap("decompress silesia-gzip Tmax x86_64", "rapidgzip", -20.0);
        assert_eq!(cat, GapCategory::Architecture);
    }

    #[test]
    fn test_categorize_architecture_pigz_tmax() {
        let (cat, _) = categorize_gap("decompress silesia-pigz Tmax arm64", "rapidgzip", -11.5);
        assert_eq!(cat, GapCategory::Architecture);
    }

    #[test]
    fn test_categorize_simd_compress_igzip() {
        let (cat, _) = categorize_gap("compress software L1 T1 x86_64", "igzip", -12.0);
        assert_eq!(cat, GapCategory::Simd);
    }

    #[test]
    fn test_categorize_actionable_bgzf_t1() {
        let (cat, _) = categorize_gap("decompress logs-bgzf T1 arm64", "pigz", -14.0);
        assert_eq!(cat, GapCategory::Actionable);
    }

    #[test]
    fn test_categorize_actionable_bgzf_tmax() {
        let (cat, _) = categorize_gap("decompress silesia-bgzf Tmax x86_64", "rapidgzip", -3.4);
        assert_eq!(cat, GapCategory::Actionable);
    }

    #[test]
    fn test_categorize_actionable_decompress_igzip_t1() {
        let (cat, _) = categorize_gap("decompress silesia-bgzf T1 x86_64", "igzip", -2.2);
        assert_eq!(cat, GapCategory::Actionable);
    }

    #[test]
    fn test_categorize_decompress_not_simd() {
        // "decompress" contains "compress" — make sure it doesn't match SIMD rule
        let (cat, _) = categorize_gap("decompress silesia-bgzf T1 x86_64", "igzip", -2.5);
        assert_ne!(cat, GapCategory::Simd);
    }

    #[test]
    fn test_categorize_actionable_compress_pigz() {
        let (cat, action) = categorize_gap("compress software L9 Tmax x86_64", "pigz", -3.6);
        assert_eq!(cat, GapCategory::Actionable);
        assert!(action.contains("pigz"), "action should mention pigz: {}", action);
    }

    #[test]
    fn test_categorize_bgzf_not_architecture() {
        // BGZF Tmax should be ACTIONABLE, not ARCHITECTURE
        let (cat, _) = categorize_gap("decompress logs-bgzf Tmax x86_64", "rapidgzip", -5.0);
        assert_ne!(cat, GapCategory::Architecture);
        assert_eq!(cat, GapCategory::Actionable);
    }

    // ── GapCategory methods ──

    #[test]
    fn test_gap_category_labels() {
        assert_eq!(GapCategory::Architecture.label(), "ARCHITECTURE");
        assert_eq!(GapCategory::Simd.label(), "SIMD");
        assert_eq!(GapCategory::Actionable.label(), "ACTIONABLE");
        assert_eq!(GapCategory::Noise.label(), "NOISE");
    }

    #[test]
    fn test_gap_category_ordering() {
        assert!(GapCategory::Architecture < GapCategory::Simd);
        assert!(GapCategory::Simd < GapCategory::Actionable);
        assert!(GapCategory::Actionable < GapCategory::Noise);
    }

    // ── truncate ──

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hell…");
    }

    // ── count_wins_gaps ──

    #[test]
    fn test_count_wins_gaps_basic() {
        let benchmarks = vec![
            BenchResult {
                job: "Decompress silesia-gzip T1 (x86_64)".into(),
                tool: "gzippy".into(), speed_mbps: 260.0, trials: 10,
                ratio: None, dataset: "silesia".into(), archive: "gzip".into(),
                threads: "T1".into(), platform: "x86_64".into(),
                mode: "decompress".into(), level: String::new(),
            },
            BenchResult {
                job: "Decompress silesia-gzip T1 (x86_64)".into(),
                tool: "rapidgzip".into(), speed_mbps: 250.0, trials: 10,
                ratio: None, dataset: "silesia".into(), archive: "gzip".into(),
                threads: "T1".into(), platform: "x86_64".into(),
                mode: "decompress".into(), level: String::new(),
            },
            BenchResult {
                job: "Decompress silesia-gzip T1 (x86_64)".into(),
                tool: "igzip".into(), speed_mbps: 270.0, trials: 10,
                ratio: None, dataset: "silesia".into(), archive: "gzip".into(),
                threads: "T1".into(), platform: "x86_64".into(),
                mode: "decompress".into(), level: String::new(),
            },
        ];
        let (wins, gaps, total) = count_wins_gaps(&benchmarks);
        assert_eq!(wins, 1);  // beats rapidgzip
        assert_eq!(gaps, 1);  // loses to igzip
        assert_eq!(total, 2);
    }

    #[test]
    fn test_count_wins_gaps_skips_gzip_zopfli() {
        let benchmarks = vec![
            BenchResult {
                job: "test".into(), tool: "gzippy".into(), speed_mbps: 260.0,
                trials: 10, ratio: None, dataset: "s".into(), archive: "g".into(),
                threads: "T1".into(), platform: "x".into(),
                mode: "decompress".into(), level: String::new(),
            },
            BenchResult {
                job: "test".into(), tool: "gzip".into(), speed_mbps: 500.0,
                trials: 10, ratio: None, dataset: "s".into(), archive: "g".into(),
                threads: "T1".into(), platform: "x".into(),
                mode: "decompress".into(), level: String::new(),
            },
            BenchResult {
                job: "test".into(), tool: "zopfli".into(), speed_mbps: 500.0,
                trials: 10, ratio: None, dataset: "s".into(), archive: "g".into(),
                threads: "T1".into(), platform: "x".into(),
                mode: "decompress".into(), level: String::new(),
            },
        ];
        let (wins, gaps, total) = count_wins_gaps(&benchmarks);
        assert_eq!(total, 0);  // gzip and zopfli are excluded
        assert_eq!(wins, 0);
        assert_eq!(gaps, 0);
    }
}
