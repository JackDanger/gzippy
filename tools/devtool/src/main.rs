//! gzippy-dev: Development tool for the gzippy project.
//!
//! Automates CI monitoring, performance analysis, benchmarking,
//! instrumentation, and project orientation.

mod bench;
mod ci;
mod cloud;
mod instrument;
mod orient;
mod path_trace;
mod score;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "ci" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev ci <status|watch|results|gaps|compare>");
                std::process::exit(1);
            }
            let run_id = find_flag(&args, "--run");
            let branch = find_flag(&args, "--branch");
            match args[2].as_str() {
                "status" => ci::status(branch.as_deref()),
                "watch" => ci::watch(run_id.as_deref(), branch.as_deref()),
                "results" => ci::results(run_id.as_deref(), branch.as_deref()),
                "gaps" => ci::gaps(run_id.as_deref(), branch.as_deref()),
                "triage" => ci::triage(run_id.as_deref(), branch.as_deref()),
                "history" => {
                    let limit = find_flag(&args, "--limit")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(5);
                    ci::history(branch.as_deref(), limit)
                }
                "compare" => {
                    if args.len() < 5 {
                        eprintln!("Usage: gzippy-dev ci compare <run_id_a> <run_id_b>");
                        std::process::exit(1);
                    }
                    ci::compare(&args[3], &args[4])
                }
                "vs-main" => ci::vs_main(branch.as_deref()),
                "push" => ci::push_and_watch(),
                _ => {
                    eprintln!("Unknown ci subcommand: {}", args[2]);
                    std::process::exit(1);
                }
            }
        }
        "bench" => {
            if args.len() >= 3 && args[2] == "ab" {
                if args.len() < 5 {
                    eprintln!("Usage: gzippy-dev bench ab <ref-a> <ref-b> [--dataset NAME] [--threads N]");
                    std::process::exit(1);
                }
                let dataset = find_flag(&args, "--dataset");
                let threads = find_flag(&args, "--threads");
                bench::run_ab(&args[3], &args[4], dataset.as_deref(), threads.as_deref())
            } else {
                let direction = match find_flag(&args, "--direction").as_deref() {
                    Some("compress") => bench::BenchDirection::Compress,
                    Some("both") => bench::BenchDirection::Both,
                    _ => bench::BenchDirection::Decompress,
                };
                let bench_args = bench::BenchArgs {
                    dataset: find_flag(&args, "--dataset"),
                    archive: find_flag(&args, "--archive"),
                    threads: find_flag(&args, "--threads").and_then(|s| s.parse().ok()),
                    json: args.iter().any(|a| a == "--json"),
                    min_trials: find_flag(&args, "--min-trials")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(10),
                    max_trials: find_flag(&args, "--max-trials")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(40),
                    target_cv: find_flag(&args, "--target-cv")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.03),
                    direction,
                };
                bench::run(&bench_args)
            }
        }
        "path" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev path <file.gz>");
                std::process::exit(1);
            }
            path_trace::trace(&args[2])
        }
        "instrument" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev instrument <file.gz> [--threads N]");
                std::process::exit(1);
            }
            let threads = find_flag(&args, "--threads");
            instrument::run(&args[2], threads.as_deref())
        }
        "cloud" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev cloud <bench|cleanup>");
                std::process::exit(1);
            }
            match args[2].as_str() {
                "bench" => cloud::bench(),
                "cleanup" => cloud::cleanup_all(),
                _ => {
                    eprintln!("Unknown cloud subcommand: {}", args[2]);
                    std::process::exit(1);
                }
            }
        }
        "orient" => orient::run(),
        "score" => score::run(),
        "losses" => score::losses(),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn find_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_usage() {
    eprintln!(
        r#"gzippy-dev — Development tool for the gzippy project

CI WORKFLOW (use these — CI is the source of truth):
  ci triage [--run ID]         Categorized gap analysis with root causes and actions
  ci vs-main                   Compare current branch vs main (auto-finds runs)
  ci push                      Push, wait for CI, auto-triage + compare vs main
  ci history [--limit N]       Win rate trend across recent CI runs

CI DETAILS:
  ci status                    Show status of recent CI runs
  ci watch [--run ID]          Block until a CI run completes
  ci results [--run ID]        Parse and display benchmark results
  ci gaps [--run ID]           Show performance gaps vs all competitors
  ci compare <ID_A> <ID_B>     Compare gzippy results between two CI runs

SCORECARD:
  score                        Show current win/loss scorecard from cloud-results.json
  losses                       Show losses grouped by root cause with actions

CLOUD (dedicated hardware, low jitter):
  cloud bench                  Launch EC2 fleet, run full benchmarks, tear down
  cloud cleanup                Delete any leaked cloud resources from prior runs

BENCHMARK (one source of truth for all perf numbers):
  bench [FLAGS]                Run decompression benchmark (human output)
  bench --json                 Same, but JSON to stdout (used by cloud fleet)
  bench ab <ref-a> <ref-b>     A/B comparison of two git refs
  path <file.gz>               Trace which decompression path a file takes
  instrument <file.gz>         Decompress with detailed timing breakdown
  orient                       Show project state, strategy, and what's next

BENCH FLAGS:
  --dataset NAME               silesia, software, logs (default: all found)
  --archive TYPE               gzip, bgzf, pigz (default: all found)
  --threads N                  1 for T1, >1 for Tmax (default: both)
  --direction MODE             decompress, compress, both (default: decompress)
  --json                       Machine-readable JSON output on stdout
  --min-trials N               Minimum trial runs (default: 10)
  --max-trials N               Maximum trial runs (default: 40)
  --target-cv F                Target coefficient of variation (default: 0.03)

OPTIONS:
  --run ID                     Specific GitHub Actions run ID
  --branch NAME                Filter CI runs to a specific branch
  --threads N                  Thread count for bench ab / instrument (default: 1)
  --limit N                    Number of CI runs for history (default: 5)

TYPICAL WORKFLOW:
  gzippy-dev ci triage                     # 1. See where we stand
  # ... make code changes ...
  gzippy-dev ci push                       # 2. Push → CI → auto-triage + vs-main
  gzippy-dev ci vs-main                    # 3. Check anytime: branch vs main
"#
    );
}
