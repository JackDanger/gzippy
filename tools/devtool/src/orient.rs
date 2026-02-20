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
        &format!("origin/main...HEAD"),
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
    println!("  Single-thread inflate (ARM M3):");
    println!("    Silesia: ~1400 MB/s (99% of libdeflate)  ← AT PARITY");
    println!("    Software: ~21500 MB/s (106% of libdeflate)  ← EXCEEDS");
    println!("    Logs: ~9100 MB/s (114% of libdeflate)  ← EXCEEDS");
    println!();
    println!("  Parallel (BGZF, 8 threads): 3770 MB/s (2.7x libdeflate single-thread)");
    println!();
    println!("  CI benchmarks (stdin-piped, 4 threads):");
    println!("    Compression: BEAT pigz everywhere, close to igzip on L1 x86");
    println!("    Decompression: competitive except Tmax single-member vs rapidgzip");
}

fn print_strategy() {
    println!("\n── STRATEGY ───────────────────────────────────────────────────────────────");
    println!("  Completed:");
    println!("    [x] Pure Rust inflate at libdeflate parity");
    println!("    [x] BGZF parallel decompression (3770 MB/s)");
    println!("    [x] T1 direct-write (no intermediate buffer)");
    println!("    [x] Tmax single-member direct-write (PR #50)");
    println!("    [x] ISIZE-based buffer pre-allocation");
    println!();
    println!("  In Progress:");
    println!("    [ ] CI sampling improvements (MIN_TRIALS=10, trimmed stats, CV=3%)");
    println!();
    println!("  Next:");
    println!("    [ ] Tmax BGZF/multi-member direct-write via Mutex<stdout>");
    println!("    [ ] Speculative parallel for single-member (closes rapidgzip gap)");
    println!();
    println!("  Won't Fix:");
    println!("    [-] igzip compression L1 T1 x86 gap (requires hand-tuned AVX2 asm)");
    println!("    [-] Two-pass parallel (scan pass costs as much as full decode)");
}

fn print_gaps_summary() {
    println!("\n── GAP CATEGORIES ─────────────────────────────────────────────────────────");
    println!("  A. Tmax vs rapidgzip (single-member): 15 gaps, 5 are >10%");
    println!("     Root cause: no parallel single-member decompress");
    println!("     Fix: speculative parallel decode (Phase 3)");
    println!();
    println!("  B. T1 residual: 25 gaps, most <2% (noise)");
    println!("     Only 3 meaningful: software-bgzf T1 arm64, compress L1 x86 vs igzip");
    println!("     Fix for bgzf: already addressed in PR #50");
    println!("     Fix for igzip: not feasible without SIMD assembly");
    println!();
    println!("  C. Tmax vs pigz on arm64: 7 gaps, 2 are >8%");
    println!("     Root cause: Tmax double-buffering (fixed in PR #50)");
    println!("     Remaining: sequential inflate speed diff on Graviton");
    println!();
    println!("  Run `gzippy-dev ci gaps` for live numbers.");
    println!("  Run `gzippy-dev ci gaps --run ID1 --compare ID2` to compare runs.");
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
