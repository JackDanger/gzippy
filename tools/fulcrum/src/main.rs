//! FULCRUM — causal-mechanistic pipeline profiler for gzippy's parallel
//! pure-Rust gzip decoder. Finds the leverage point: the code region whose
//! speedup moves the wall the most (wall-elasticity), with on/off-critical-
//! path classification, a per-region mechanism (DRAM-bound / branch-miss /
//! false-sharing), and a confidence interval.
//!
//! Four fused layers over ONE span+dependency graph the program already
//! emits (trace_v2 / GZIPPY_TIMELINE) plus Coz + perf:
//!   1. Causal (Coz virtual speedup)  — the primary ∂wall/∂speed metric.
//!   2. Critical-path (wPerf-style)    — consumer-anchored wait attribution.
//!   3. Mechanistic (Linux perf)       — TMA / PEBS / c2c → the WHY.
//!   4. (stretch) structural what-if   — documented, not in MVP.
//!
//! Subcommands (run `fulcrum help`):
//!   critpath <trace.json>            critical-path from a trace_v2 timeline
//!   coz-parse <profile.coz>          parse a coz profile → per-region curves
//!   mech-report <perf_report.txt>    parse a perf report → per-func cycles
//!   rank <trace.json> [profile.coz] [perf_report.txt]
//!                                    fuse → ranked lever list
//!   validate <trace.json> <profile.coz>
//!                                    check vs known ground truth (the gate)
//!   plan                             print the box command plan (coz/perf/AB)

mod coz;
mod critpath;
mod mech;
mod rank;
mod trace;
mod validate;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "FULCRUM — gzippy causal-mechanistic pipeline profiler\n\
\n\
USAGE:\n\
  fulcrum critpath <trace_v2.json> [--heavy-ms 30]\n\
  fulcrum coz-parse <profile.coz> [--progress chunk_emitted]\n\
  fulcrum mech-report <perf_report.txt>\n\
  fulcrum rank <trace_v2.json> [profile.coz] [perf_report.txt]\n\
  fulcrum validate <trace_v2.json> <profile.coz>\n\
  fulcrum plan [--repo /root/gzippy] [--cpus 0,2,4,6,8,10,12,14] [--threads 8]\n\
\n\
The trace_v2.json is produced by running the gzippy decode with\n\
GZIPPY_TIMELINE=/path.json. profile.coz is produced by `fulcrum plan`'s\n\
coz step (or a manual coz run of examples/fulcrum_loop). All wall-relevant\n\
runs happen on the frozen box; this binary is the analyzer + orchestrator.\n"
    );
    ExitCode::from(2)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn positional(args: &[String]) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            i += 2; // skip flag + value
        } else {
            out.push(a.as_str());
            i += 1;
        }
    }
    out
}

fn cmd_critpath(args: &[String]) -> ExitCode {
    let pos = positional(args);
    let Some(trace_path) = pos.first() else {
        return usage();
    };
    let heavy_ms: f64 = flag(args, "--heavy-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let events = match trace::load_events(Path::new(trace_path)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("fulcrum: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cp = critpath::analyze(&events, heavy_ms * 1000.0);
    print_critpath(&cp);
    ExitCode::SUCCESS
}

fn print_critpath(cp: &critpath::CritPath) {
    println!("\n========  CRITICAL PATH (consumer-anchored)  ========");
    println!("wall            : {}", trace::fmt_us(cp.wall_us));
    println!(
        "consumer tid    : pid {}/tid {}",
        cp.consumer.0, cp.consumer.1
    );
    println!(
        "consumer busy   : {} ({:.1}% of wall)",
        trace::fmt_us(cp.consumer_busy_us),
        100.0 * cp.consumer_busy_us / cp.wall_us.max(1.0)
    );
    println!(
        "consumer wait   : {} ({:.1}% of wall)  ← gated by producers",
        trace::fmt_us(cp.consumer_wait_us),
        100.0 * cp.consumer_wait_us / cp.wall_us.max(1.0)
    );
    println!("\nOn-critical-path attribution (top 14):");
    println!(
        "  {:<46} {:>10} {:>8} {:>10}",
        "label", "on-path", "share", "max"
    );
    for e in cp.entries.iter().take(14) {
        println!(
            "  {:<46} {:>10} {:>7.1}% {:>10}",
            e.label,
            trace::fmt_us(e.on_path_us),
            e.fraction * 100.0,
            trace::fmt_us(e.max_us),
        );
    }
    if !cp.heavy_chunks.is_empty() {
        println!(
            "\nHEAVY OVERSHOOT CHUNKS ({} — the long-pole blockers gating the wall):",
            cp.heavy_chunks.len()
        );
        println!(
            "  {:<28} {:>9} {:>12} {:>10}",
            "blocker span", "chunk_id", "blocker dur", "wait"
        );
        for h in cp.heavy_chunks.iter().take(12) {
            println!(
                "  {:<28} {:>9} {:>12} {:>10}",
                h.blocker_span,
                h.chunk_id
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".into()),
                trace::fmt_us(h.blocker_dur_us),
                trace::fmt_us(h.wait_us),
            );
        }
    }
}

fn cmd_coz_parse(args: &[String]) -> ExitCode {
    let pos = positional(args);
    let Some(prof) = pos.first() else {
        return usage();
    };
    let progress = flag(args, "--progress").unwrap_or("chunk_emitted");
    let maps = coz::default_region_maps();
    match coz::parse_profile(Path::new(prof), progress, &maps) {
        Ok(p) => {
            print_coz(&p);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fulcrum: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_coz(p: &coz::CozProfile) {
    println!("\n========  COZ CAUSAL PROFILE  ========");
    println!("progress point  : {}", p.progress_point);
    println!("experiments     : {}", p.n_experiments);
    println!("\nPer-REGION wall-elasticity (∂program-speedup / ∂region-speedup):");
    println!(
        "  {:<14} {:>10} {:>16} {:>10} {:>9}",
        "region", "median", "95% CI (proxy)", "PEAK-line", "peak-n"
    );
    for (region, rc) in &p.region_curves {
        let (e, lo, hi) = rc.elasticity_ci();
        let (peak, peak_n) = rc.peak_line_elasticity();
        println!(
            "  {:<14} {:>+10.3} {:>16} {:>+10.3} {:>9.0}",
            region,
            e,
            format!("[{:+.3},{:+.3}]", lo, hi),
            peak,
            peak_n,
        );
    }
    println!(
        "  (median can be masked by a high-sample ~0 line; PEAK-line = the\n   \
         single highest-confidence line you'd actually optimize)"
    );
    if !p.region_latency.is_empty() {
        println!("\nRegion latency points (begin!/end! scopes):");
        println!(
            "  {:<20} {:>10} {:>12} {:>14}",
            "region", "arrivals", "departures", "Σdiff(ns)"
        );
        for (name, (a, d, diff)) in &p.region_latency {
            println!("  {:<20} {:>10.0} {:>12.0} {:>14.0}", name, a, d, diff);
        }
    }
    println!(
        "\nTop per-LINE curves (confidence-ranked |slope|·√samples; \
         samples≥{:.0} trusted):",
        coz::MIN_LINE_SAMPLES
    );
    println!(
        "  {:<46} {:>9} {:>9} {}",
        "selected (file:line)", "slope", "samples", "region"
    );
    let maps = coz::default_region_maps();
    for c in p
        .line_curves
        .iter()
        .filter(|c| c.total_samples >= 5.0)
        .take(14)
    {
        let region = coz::region_of(&c.selected, &maps).unwrap_or_else(|| "-".into());
        let mark = if c.total_samples >= coz::MIN_LINE_SAMPLES {
            " "
        } else {
            "~" // low-confidence
        };
        println!(
            "  {}{:<45} {:>+9.3} {:>9.0} {}",
            mark,
            c.selected,
            c.slope(),
            c.total_samples,
            region
        );
    }
}

fn cmd_mech_report(args: &[String]) -> ExitCode {
    let pos = positional(args);
    let Some(rep) = pos.first() else {
        return usage();
    };
    let text = match std::fs::read_to_string(rep) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("fulcrum: {e}");
            return ExitCode::FAILURE;
        }
    };
    let by_func = mech::parse_perf_report(&text);
    println!("\n========  PERF REPORT (function cycles%)  ========");
    let mut rows: Vec<_> = by_func.iter().collect();
    rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (name, pct) in rows.iter().take(25) {
        println!("  {:>6.2}%  {}", pct, name);
    }
    ExitCode::SUCCESS
}

fn load_mech_from_report(path: Option<&str>) -> Option<mech::Mech> {
    let path = path?;
    let text = std::fs::read_to_string(path).ok()?;
    let by_func_pct = mech::parse_perf_report(&text);
    let mut m = mech::Mech::default();
    for (name, pct) in by_func_pct {
        m.by_func.entry(name).or_default().cycles_pct = pct;
    }
    Some(m)
}

fn cmd_rank(args: &[String]) -> ExitCode {
    let pos = positional(args);
    let Some(trace_path) = pos.first() else {
        return usage();
    };
    let coz_path = pos.get(1).copied();
    let perf_path = pos.get(2).copied();

    let events = match trace::load_events(Path::new(trace_path)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("fulcrum: trace: {e}");
            return ExitCode::FAILURE;
        }
    };
    let heavy_ms: f64 = flag(args, "--heavy-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let cp = critpath::analyze(&events, heavy_ms * 1000.0);

    let maps = coz::default_region_maps();
    let coz_prof =
        coz_path.and_then(|p| coz::parse_profile(Path::new(p), "chunk_emitted", &maps).ok());
    let mech = load_mech_from_report(perf_path);

    print_critpath(&cp);
    if let Some(c) = &coz_prof {
        print_coz(c);
    } else {
        println!("\n(no profile.coz supplied — ranking by critical-path on-path share only)");
    }

    let levers = rank::rank(coz_prof.as_ref(), &cp, mech.as_ref());
    print!("{}", rank::render(&levers));

    ExitCode::SUCCESS
}

fn cmd_validate(args: &[String]) -> ExitCode {
    let pos = positional(args);
    let (Some(trace_path), Some(coz_path)) = (pos.first(), pos.get(1)) else {
        eprintln!("validate needs <trace_v2.json> <profile.coz>");
        return usage();
    };
    let events = match trace::load_events(Path::new(trace_path)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("fulcrum: {e}");
            return ExitCode::FAILURE;
        }
    };
    let heavy_ms: f64 = flag(args, "--heavy-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let cp = critpath::analyze(&events, heavy_ms * 1000.0);
    let maps = coz::default_region_maps();
    let coz_prof = coz::parse_profile(Path::new(coz_path), "chunk_emitted", &maps).ok();

    let v = validate::check_against_ground_truth(coz_prof.as_ref(), &cp);
    println!("\n========  VALIDATION vs KNOWN GROUND TRUTH  ========");
    for c in &v.checks {
        println!(
            "  [{}] {}\n        expect : {}\n        measured: {}",
            if c.passed { "PASS" } else { "FAIL" },
            c.name,
            c.expectation,
            c.measured
        );
    }
    println!(
        "\n  VERDICT: {}",
        if v.all_passed() {
            "FULCRUM reproduces the empirical oracle — TRUSTWORTHY."
        } else {
            "FULCRUM diverges from ground truth — investigate before trusting."
        }
    );
    if v.all_passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_plan(args: &[String]) -> ExitCode {
    let repo = flag(args, "--repo").unwrap_or("/root/gzippy");
    let cpus = flag(args, "--cpus").unwrap_or("0,2,4,6,8,10,12,14");
    let threads: usize = flag(args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let input = flag(args, "--input").unwrap_or("benchmark_data/silesia-large.gz");
    let iters: usize = flag(args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let runs: usize = flag(args, "--runs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let raw: u64 = flag(args, "--raw")
        .and_then(|s| s.parse().ok())
        .unwrap_or(503_627_776);

    let harness = format!("{repo}/target/fulcrum/examples/fulcrum_loop");
    let _ = (PathBuf::from(&harness), coz::default_region_maps());

    println!("# ============================================================");
    println!("# FULCRUM box plan — run each phase inside the freeze window.");
    println!("# repo={repo} cpus={cpus} threads={threads} input={input}");
    println!("# ============================================================\n");

    println!("## 0. Sync this branch + build the fulcrum-profile binary + harness");
    println!("cd {repo} && git fetch origin feat/fulcrum && \\");
    println!("  git checkout -f -B feat/fulcrum origin/feat/fulcrum && \\");
    println!("  git reset --hard origin/feat/fulcrum && \\");
    println!("  git submodule update --init --recursive");
    println!("cargo build --profile fulcrum --no-default-features \\");
    println!("  --features pure-rust-inflate,fulcrum --example fulcrum_loop");
    println!("cargo build --release --manifest-path tools/fulcrum/Cargo.toml\n");

    println!("## 1. Critical-path trace (one frozen run with GZIPPY_TIMELINE)");
    println!("ssh neurotic 'bash {repo}/scripts/freeze_wrapper.sh \\");
    println!("  pct exec 199 -- env GZIPPY_TIMELINE=/tmp/fulcrum_tl.json \\");
    println!("    taskset -c {cpus} {harness} {input} --iters 1 --threads {threads}'");
    println!("# pull /tmp/fulcrum_tl.json, then:");
    println!("fulcrum critpath /tmp/fulcrum_tl.json --heavy-ms 30\n");

    println!("## 2. Coz causal profile");
    println!(
        "# PRIMARY: ONE long in-process run — `--iters {}` makes the harness",
        iters * 10
    );
    println!(
        "# decode ~{}x (~{}s of steady-state), so coz runs MANY virtual-speedup",
        iters * 10,
        (iters * 10) * 4 / 10
    );
    println!("# epochs in a SINGLE process and the chunk_emitted point is hit");
    println!(
        "# {}x*n_chunks times — enough power without cross-process append.",
        iters * 10
    );
    println!("ssh neurotic 'bash {repo}/scripts/freeze_wrapper.sh \\");
    println!("  pct exec 199 -- bash -c \"cd {repo}; rm -f /tmp/profile.coz; \\");
    println!("    taskset -c {cpus} coz run --output /tmp/profile.coz \\");
    println!("      --source-scope %/decompress/parallel/% --binary-scope MAIN \\");
    println!(
        "      --- {harness} {input} --iters {} --threads {threads}\"'",
        iters * 10
    );
    println!("# BOOSTER (optional): {runs} end-to-end runs APPEND more experiments:");
    println!("#   for i in $(seq {runs}); do coz run --output /tmp/profile.coz --end-to-end \\");
    println!("#     --source-scope %/decompress/parallel/% --binary-scope MAIN \\");
    println!("#     --- {harness} {input} --iters {iters} --threads {threads}; done");
    println!("# pull /tmp/profile.coz, then:");
    println!("fulcrum coz-parse /tmp/profile.coz --progress chunk_emitted\n");

    println!("## 3. Mechanism (perf TMA + report) — frozen");
    println!("ssh neurotic 'bash {repo}/scripts/freeze_wrapper.sh \\");
    println!("  pct exec 199 -- bash -c \"cd {repo}; \\");
    println!("    perf stat --topdown -- taskset -c {cpus} {harness} {input} --iters {iters} --threads {threads} 2>/tmp/fulcrum_topdown.txt; \\");
    println!("    perf record -g -o /tmp/fulcrum.data -- taskset -c {cpus} {harness} {input} --iters {iters} --threads {threads}; \\");
    println!("    perf report -i /tmp/fulcrum.data --stdio -n > /tmp/fulcrum_report.txt\"'");
    println!("fulcrum mech-report /tmp/fulcrum_report.txt\n");

    println!("## 4. Empirical oracle cross-check (THE wall metric) — frozen");
    println!("ssh neurotic 'bash {repo}/scripts/freeze_wrapper.sh \\");
    println!("  pct exec 199 -- env RAW={raw} N=15 CPUS={cpus} REF=base \\");
    println!("    bash {repo}/scripts/interleaved_ab.sh \\");
    println!("      \"base={harness} {input} --iters {iters} --threads {threads}\"'\n");

    println!("## 5. Validate + fuse → ranked lever list");
    println!("fulcrum validate /tmp/fulcrum_tl.json /tmp/profile.coz");
    println!("fulcrum rank /tmp/fulcrum_tl.json /tmp/profile.coz /tmp/fulcrum_report.txt");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(sub) = args.first().cloned() else {
        return usage();
    };
    let rest = &args[1..];
    match sub.as_str() {
        "critpath" => cmd_critpath(rest),
        "coz-parse" => cmd_coz_parse(rest),
        "mech-report" => cmd_mech_report(rest),
        "rank" => cmd_rank(rest),
        "validate" => cmd_validate(rest),
        "plan" => cmd_plan(rest),
        "help" | "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("fulcrum: unknown subcommand '{other}'");
            usage()
        }
    }
}
