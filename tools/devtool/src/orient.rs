//! Orientation: compact view of project history, strategy, and current state.
//!
//! Replaces reading multiple files and git logs to understand where we are.

use std::process::Command;

pub fn run() -> Result<(), String> {
    print_header();
    print_git_summary()?;
    print_architecture();
    print_perf_status();
    print_strategy();
    print_gaps_summary();
    Ok(())
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  GZIPPY ORIENTATION — Where We Are, What's Next                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn print_git_summary() -> Result<(), String> {
    println!("\n── GIT STATE ──────────────────────────────────────────────────────────────");

    let branch = run_git(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let dirty = run_git(&["status", "--porcelain"])?;
    let dirty_count = dirty.lines().count();
    let ahead_behind = run_git(&[
        "rev-list",
        "--left-right",
        "--count",
        "origin/main...HEAD",
    ])
    .unwrap_or_default();

    println!("  Branch: {}", branch.trim());
    if dirty_count > 0 {
        println!("  Uncommitted: {} files", dirty_count);
    }
    if !ahead_behind.is_empty() {
        let parts: Vec<&str> = ahead_behind.trim().split('\t').collect();
        if parts.len() == 2 {
            println!(
                "  vs main: {} behind, {} ahead",
                parts[0].trim(),
                parts[1].trim()
            );
        }
    }

    // Recent commits with perf impact
    println!("\n  Recent commits (perf-relevant):");
    let log = run_git(&[
        "log",
        "--oneline",
        "-20",
        "--format=%h %s",
        "--",
        "src/decompression.rs",
        "src/bgzf.rs",
        "src/consume_first_decode.rs",
        "src/libdeflate_decode.rs",
        "scripts/benchmark_*.py",
    ])?;
    for line in log.lines().take(10) {
        println!("    {}", line);
    }

    // Open PRs
    println!("\n  Open PRs:");
    let prs = run_cmd("gh", &["pr", "list", "--state", "open", "--json", "number,title,headBranch"])?;
    if let Ok(pr_list) = serde_json::from_str::<Vec<serde_json::Value>>(&prs) {
        if pr_list.is_empty() {
            println!("    (none)");
        }
        for pr in pr_list.iter().take(5) {
            println!(
                "    #{} {} ({})",
                pr["number"],
                pr["title"].as_str().unwrap_or(""),
                pr["headBranch"].as_str().unwrap_or("")
            );
        }
    }

    Ok(())
}

fn print_architecture() {
    println!("\n── ARCHITECTURE ───────────────────────────────────────────────────────────");
    println!("  Decompression paths:");
    println!("    Single-member (any thread count) → decompress_multi_member_sequential");
    println!("      → libdeflate FFI gzip_decompress_ex, direct stdout write");
    println!("    BGZF Tmax → decompress_bgzf_parallel → libdeflate per-block, Vec buffer");
    println!("    Multi-member Tmax → decompress_multi_member_parallel → libdeflate, Vec buffer");
    println!();
    println!("  Compression paths:");
    println!("    L1-L5  → libdeflate parallel (BGZF blocks)");
    println!("    L6-L9  → zlib-ng pipelined with dictionary");
    println!("    L10-12 → libdeflate exhaustive (512KB blocks)");
    println!();
    println!("  Key files:");
    println!("    src/decompression.rs    — routing, stdin/file dispatch");
    println!("    src/bgzf.rs             — parallel BGZF/multi-member");
    println!("    src/consume_first_decode.rs — pure Rust inflate (parity with libdeflate)");
    println!("    src/libdeflate_decode.rs    — entry format definitions");
}

fn print_perf_status() {
    println!("\n── PERFORMANCE STATUS ─────────────────────────────────────────────────────");
    println!("  Win rate: 68% (84/123 comparisons)");
    println!("  Effective: 85% (counting 21 noise gaps <2% as ties)");
    println!();
    println!("  Gap categories:");
    println!("    ARCHITECTURE (7): Tmax single-member vs rapidgzip (-20% worst)");
    println!("    SIMD (3): L1 compression x86 vs igzip (-12% worst)");
    println!("    ACTIONABLE (8): BGZF T1/Tmax overhead (-14% worst)");
    println!("    NOISE (21): all <2%, measurement variance");
    println!();
    println!("  Run `gzippy-dev ci triage` for live categorized analysis.");
}

fn print_strategy() {
    println!("\n── STRATEGY ───────────────────────────────────────────────────────────────");
    println!("  Completed:");
    println!("    [x] Pure Rust inflate at libdeflate parity");
    println!("    [x] BGZF parallel decompression");
    println!("    [x] T1 direct-write (no intermediate buffer)");
    println!("    [x] Tmax single-member direct-write");
    println!("    [x] ISIZE-based buffer pre-allocation");
    println!("    [x] BGZF T1 fast path (PR #52 — detect BGZF before parallelism check)");
    println!();
    println!("  In Progress:");
    println!("    [ ] Validate PR #52 impact on arm64 BGZF T1 gaps (-14%)");
    println!("    [ ] BGZF Tmax optimization (thread pool overhead, -2% to -3.4%)");
    println!();
    println!("  Next:");
    println!("    [ ] Parallel single-member decompression (rapidgzip-style pipeline)");
    println!("        7 gaps, -20% worst — largest remaining performance opportunity");
    println!();
    println!("  Won't Fix:");
    println!("    [-] igzip compression L1 T1 x86 gap (requires hand-tuned AVX2 asm)");
    println!("    [-] Speculative parallel (proven infeasible Feb 2026)");
    println!("    [-] Two-pass parallel (scan pass costs as much as full decode)");
}

fn print_gaps_summary() {
    println!("\n── WORKFLOW (CI is the source of truth) ────────────────────────────────────");
    println!("  gzippy-dev ci triage          # See categorized gaps, win rate");
    println!("  # ... make ONE code change ...");
    println!("  gzippy-dev ci push            # Push → CI → auto-triage + vs-main");
    println!("  gzippy-dev ci vs-main         # Branch vs main (auto-finds runs)");
    println!("  gzippy-dev ci history         # Win rate trend over time");
    println!();
    println!("  Local tools (for rapid iteration, not final decisions):");
    println!("    gzippy-dev instrument f.gz  # Timing breakdown");
    println!("    gzippy-dev path f.gz        # Which code path runs");
}

fn run_git(args: &[&str]) -> Result<String, String> {
    run_cmd("git", args)
}

fn run_cmd(cmd: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("Failed to run {}: {}", cmd, e))?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
