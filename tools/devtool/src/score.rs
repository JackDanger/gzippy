//! Scorecard: read cloud-results.json and display current win/loss status.

use crate::bench::find_repo_root;
use serde::Deserialize;

#[derive(Deserialize)]
struct CloudResults {
    timestamp: String,
    wins: usize,
    losses: usize,
    total_scenarios: usize,
    scorecard: Vec<ScoreEntry>,
}

#[derive(Deserialize)]
struct ScoreEntry {
    platform: String,
    scenario: String,
    gzippy_mbps: f64,
    best_competitor: String,
    competitor_mbps: f64,
    gap_pct: f64,
    verdict: String,
}

pub fn run() -> Result<(), String> {
    let data = load_results()?;

    println!(
        "\n╔══════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  SCORECARD: {}W / {}L  ({}%)    {}  ║",
        data.wins,
        data.losses,
        data.wins * 100 / data.total_scenarios,
        &data.timestamp[..19],
    );
    println!(
        "╚══════════════════════════════════════════════════════════════╝\n"
    );

    let mut wins: Vec<&ScoreEntry> = Vec::new();
    let mut loss_list: Vec<&ScoreEntry> = Vec::new();

    for s in &data.scorecard {
        if s.verdict == "WIN" {
            wins.push(s);
        } else {
            loss_list.push(s);
        }
    }

    loss_list.sort_by(|a, b| a.gap_pct.partial_cmp(&b.gap_pct).unwrap());

    if !loss_list.is_empty() {
        println!("  LOSSES ({}):", loss_list.len());
        for s in &loss_list {
            let platform_short = platform_tag(&s.platform);
            println!(
                "    {:<30} {:>7.1} vs {:<10} {:>7.1}  {:>+6.1}%  [{}]",
                s.scenario, s.gzippy_mbps, s.best_competitor, s.competitor_mbps, s.gap_pct, platform_short,
            );
        }
    }

    println!(
        "\n  WINS: {} scenarios where gzippy is fastest.\n",
        wins.len()
    );

    Ok(())
}

pub fn losses() -> Result<(), String> {
    let data = load_results()?;

    let mut loss_list: Vec<&ScoreEntry> = data
        .scorecard
        .iter()
        .filter(|s| s.verdict != "WIN")
        .collect();
    loss_list.sort_by(|a, b| a.gap_pct.partial_cmp(&b.gap_pct).unwrap());

    if loss_list.is_empty() {
        println!("No losses! gzippy wins every scenario.");
        return Ok(());
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  {} LOSSES — grouped by root cause                          ║", loss_list.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Group by root cause
    let mut igzip_l1: Vec<&ScoreEntry> = Vec::new();
    let mut tmax_parallel: Vec<&ScoreEntry> = Vec::new();
    let mut near_parity: Vec<&ScoreEntry> = Vec::new();
    let mut arm64_compress: Vec<&ScoreEntry> = Vec::new();
    let mut other: Vec<&ScoreEntry> = Vec::new();

    for s in &loss_list {
        let is_decompress = s.platform.contains("decompress");
        if s.best_competitor == "igzip" && s.scenario.contains("L1") {
            igzip_l1.push(s);
        } else if s.scenario.contains("Tmax") && is_decompress
            && (s.best_competitor == "rapidgzip" || s.best_competitor == "unpigz")
        {
            tmax_parallel.push(s);
        } else if s.scenario.contains("Tmax") && !is_decompress {
            arm64_compress.push(s);
        } else if s.gap_pct > -3.0 {
            near_parity.push(s);
        } else {
            other.push(s);
        }
    }

    if !near_parity.is_empty() {
        println!("  ── NEAR PARITY (within 3%, likely noise) ──");
        println!("  Action: May flip to wins with measurement variance. Low priority.\n");
        for s in &near_parity {
            print_loss(s);
        }
        println!();
    }

    if !arm64_compress.is_empty() {
        println!("  ── ARM64 COMPRESS SCALING ──");
        println!("  Action: Investigate mmap vs read_to_end on Graviton, madvise hints.\n");
        for s in &arm64_compress {
            print_loss(s);
        }
        println!();
    }

    if !igzip_l1.is_empty() {
        println!("  ── IGZ L1 (AVX-512 assembly advantage) ──");
        println!("  Action: Check ISA-L AVX-512 build flags. May be unfixable.\n");
        for s in &igzip_l1 {
            print_loss(s);
        }
        println!();
    }

    if !tmax_parallel.is_empty() {
        println!("  ── TMAX SINGLE-MEMBER PARALLEL DECOMPRESS ──");
        println!("  Action: Implement rapidgzip-style block-finder + marker pipeline.\n");
        for s in &tmax_parallel {
            print_loss(s);
        }
        println!();
    }

    if !other.is_empty() {
        println!("  ── OTHER ──\n");
        for s in &other {
            print_loss(s);
        }
        println!();
    }

    let fixable = near_parity.len() + arm64_compress.len();
    let hard = igzip_l1.len() + tmax_parallel.len();
    println!("  Summary: {} likely fixable, {} require architecture changes, {} other", fixable, hard, other.len());

    Ok(())
}

fn print_loss(s: &ScoreEntry) {
    let tag = platform_tag(&s.platform);
    println!(
        "    {:<30} {:>7.1} vs {:<10} {:>7.1}  {:>+6.1}%  [{}]",
        s.scenario, s.gzippy_mbps, s.best_competitor, s.competitor_mbps, s.gap_pct, tag,
    );
}

fn platform_tag(platform: &str) -> &str {
    let is_decompress = platform.contains("decompress");
    if platform.contains("x86_64") {
        if is_decompress { "x86 decomp" } else { "x86 comp" }
    } else if platform.contains("arm64") {
        if is_decompress { "arm64 decomp" } else { "arm64 comp" }
    } else {
        platform
    }
}

fn load_results() -> Result<CloudResults, String> {
    let repo_root = find_repo_root()?;
    let path = repo_root.join("cloud-results.json");
    if !path.exists() {
        return Err(format!(
            "No cloud-results.json found at {}. Run: gzippy-dev cloud bench",
            path.display()
        ));
    }
    let contents = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))
}
