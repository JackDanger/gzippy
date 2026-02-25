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
    #[serde(default)]
    diagnostics: Option<serde_json::Value>,
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

pub fn losses(explain: bool) -> Result<(), String> {
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

    let diags = data.diagnostics.as_ref();
    let has_diags = diags.map(|d| d.as_object().map(|o| !o.is_empty()).unwrap_or(false)).unwrap_or(false);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  {} LOSSES — grouped by root cause                          ║", loss_list.len());
    if explain && !has_diags {
        println!("║  (--explain: no diagnostics in cloud-results.json)         ║");
    }
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
            if explain { print_explanation(s, diags); }
        }
        println!();
    }

    if !arm64_compress.is_empty() {
        println!("  ── ARM64 COMPRESS SCALING ──");
        println!("  Action: Investigate mmap vs read_to_end on Graviton, madvise hints.\n");
        for s in &arm64_compress {
            print_loss(s);
            if explain { print_explanation(s, diags); }
        }
        println!();
    }

    if !igzip_l1.is_empty() {
        println!("  ── IGZ L1 (AVX-512 assembly advantage) ──");
        println!("  Action: Check ISA-L AVX-512 build flags. May be unfixable.\n");
        for s in &igzip_l1 {
            print_loss(s);
            if explain { print_explanation(s, diags); }
        }
        println!();
    }

    if !tmax_parallel.is_empty() {
        println!("  ── TMAX SINGLE-MEMBER PARALLEL DECOMPRESS ──");
        println!("  Action: Implement rapidgzip-style block-finder + marker pipeline.\n");
        for s in &tmax_parallel {
            print_loss(s);
            if explain { print_explanation(s, diags); }
        }
        println!();
    }

    if !other.is_empty() {
        println!("  ── OTHER ──\n");
        for s in &other {
            print_loss(s);
            if explain { print_explanation(s, diags); }
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

fn print_explanation(s: &ScoreEntry, diags: Option<&serde_json::Value>) {
    let diags = match diags.and_then(|d| d.as_object()) {
        Some(d) => d,
        None => {
            println!("      (no diagnostics available)");
            return;
        }
    };

    // Find the matching diagnostic entry by platform label
    // Platform format: "arm64/silesia/decompress" or "x86_64/software/compress"
    let diag = find_matching_diag(s, diags);
    let diag = match diag {
        Some(d) => d,
        None => {
            println!("      (no matching diagnostic for {})", s.platform);
            return;
        }
    };

    let is_compress = s.platform.contains("compress");
    let is_decompress = s.platform.contains("decompress");

    if is_compress {
        // Show compress timing breakdown
        if let Some(timings) = diag.get("compress_timing").and_then(|t| t.as_object()) {
            // Find dataset from scenario (e.g., "software-L1 T1" -> "software")
            let dataset = s.scenario.split('-').next().unwrap_or("");
            if let Some(timing) = timings.get(dataset).and_then(|t| t.as_object()) {
                let mut parts = Vec::new();
                if let Some(alloc) = timing.get("alloc_ms").and_then(|v| v.as_f64()) {
                    parts.push(format!("alloc={alloc:.1}ms"));
                }
                if let Some(compress) = timing.get("compress_ms").and_then(|v| v.as_f64()) {
                    parts.push(format!("compress={compress:.1}ms"));
                }
                if let Some(write) = timing.get("write_ms").and_then(|v| v.as_f64()) {
                    parts.push(format!("write={write:.1}ms"));
                }
                if let Some(mbps) = timing.get("compress_mbps").and_then(|v| v.as_f64()) {
                    parts.push(format!("raw={mbps:.0} MB/s"));
                }
                if let Some(total) = timing.get("total_mbps").and_then(|v| v.as_f64()) {
                    parts.push(format!("e2e={total:.0} MB/s"));
                }
                if let Some(igzip) = timing.get("igzip_mbps").and_then(|v| v.as_f64()) {
                    parts.push(format!("igzip={igzip:.0} MB/s"));
                }
                if !parts.is_empty() {
                    println!("      Timing: {}", parts.join(", "));
                }
                if let Some(debug) = timing.get("debug_output").and_then(|v| v.as_str()) {
                    for line in debug.lines().take(3) {
                        if !line.trim().is_empty() {
                            println!("      Debug: {}", line.trim());
                        }
                    }
                }
            }
        }
        // Show dispatch check
        if let Some(dispatch) = diag.get("dispatch_check").and_then(|d| d.as_object()) {
            if let Some(verdict) = dispatch.get("verdict").and_then(|v| v.as_str()) {
                let mbps = dispatch.get("1kb_mbps").and_then(|v| v.as_f64()).unwrap_or(0.0);
                println!("      Dispatch: {verdict} ({mbps:.0} MB/s on 1KB)");
            }
        }
        // Show AVX counts
        if let (Some(gzippy_avx), Some(igzip_avx)) = (
            diag.get("avx_insns_gzippy").and_then(|v| v.as_u64()),
            diag.get("avx_insns_igzip").and_then(|v| v.as_u64()),
        ) {
            println!("      AVX insns: gzippy={gzippy_avx} igzip={igzip_avx}");
        }
    }

    if is_decompress {
        // Show decompress path for this scenario
        if let Some(paths) = diag.get("decompress_paths").and_then(|p| p.as_object()) {
            // scenario like "silesia-gzip Tmax" -> look up "silesia-gzip"
            let archive_key = s.scenario.split_whitespace().next().unwrap_or("");
            if let Some(path) = paths.get(archive_key).and_then(|v| v.as_str()) {
                println!("      Path: {path}");
            }
        }
    }

    // Show ISA-L build info
    if let Some(isal) = diag.get("isal_build").and_then(|b| b.as_object()) {
        if let Some(nasm) = isal.get("have_nasm").and_then(|v| v.as_bool()) {
            print!("      ISA-L: NASM={nasm}");
            if let Some(versions) = diag.get("versions").and_then(|v| v.as_object()) {
                if let Some(ver) = versions.get("isal_sys_version").and_then(|v| v.as_str()) {
                    print!(" version={ver}");
                }
            }
            println!();
        }
    }
}

fn find_matching_diag<'a>(
    s: &ScoreEntry,
    diags: &'a serde_json::Map<String, serde_json::Value>,
) -> Option<&'a serde_json::Value> {
    // Direct match on platform
    if let Some(d) = diags.get(&s.platform) {
        return Some(d);
    }

    // Try to match by arch + dataset + direction from platform string
    // Platform: "arm64/silesia/decompress" or "x86_64/software/compress"
    let parts: Vec<&str> = s.platform.split('/').collect();
    if parts.len() == 3 {
        let key = format!("{}/{}/{}", parts[0], parts[1], parts[2]);
        if let Some(d) = diags.get(&key) {
            return Some(d);
        }
    }

    // Fuzzy match: find any diagnostic for same arch+direction
    let is_compress = s.platform.contains("compress");
    let is_x86 = s.platform.contains("x86_64");
    let is_arm = s.platform.contains("arm64");

    for (key, val) in diags {
        let key_compress = key.contains("compress") && !key.contains("decompress");
        let key_decompress = key.contains("decompress");
        let key_x86 = key.contains("x86_64");
        let key_arm = key.contains("arm64");

        if is_compress && key_compress && ((is_x86 && key_x86) || (is_arm && key_arm)) {
            return Some(val);
        }
        if !is_compress && key_decompress && ((is_x86 && key_x86) || (is_arm && key_arm)) {
            return Some(val);
        }
    }

    None
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
