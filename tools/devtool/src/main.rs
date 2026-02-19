//! gzippy-dev: Development tool for the gzippy project.
//!
//! Automates tedious CI monitoring, performance analysis, benchmarking,
//! and code-path tracing tasks.
//!
//! Usage:
//!   gzippy-dev ci status                  # Latest CI run status
//!   gzippy-dev ci watch [--run ID]        # Block until CI completes
//!   gzippy-dev ci results [--run ID]      # Formatted benchmark results
//!   gzippy-dev ci gaps [--run ID]         # Performance gaps vs competitors
//!   gzippy-dev bench [--dataset silesia]  # Local decompression benchmark
//!   gzippy-dev path <file.gz>             # Trace decompression path

mod ci;
mod bench;
mod path_trace;

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
                eprintln!("Usage: gzippy-dev ci <status|watch|results|gaps>");
                std::process::exit(1);
            }
            let run_id = find_flag(&args, "--run");
            let branch = find_flag(&args, "--branch");
            match args[2].as_str() {
                "status" => ci::status(branch.as_deref()),
                "watch" => ci::watch(run_id.as_deref(), branch.as_deref()),
                "results" => ci::results(run_id.as_deref(), branch.as_deref()),
                "gaps" => ci::gaps(run_id.as_deref(), branch.as_deref()),
                _ => {
                    eprintln!("Unknown ci subcommand: {}", args[2]);
                    std::process::exit(1);
                }
            }
        }
        "bench" => {
            let dataset = find_flag(&args, "--dataset");
            bench::run(dataset.as_deref())
        }
        "path" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev path <file.gz>");
                std::process::exit(1);
            }
            path_trace::trace(&args[2])
        }
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

COMMANDS:
  ci status                 Show status of recent CI runs
  ci watch [--run ID]       Block until a CI run completes, then show results
  ci results [--run ID]     Parse and display benchmark results
  ci gaps [--run ID]        Show performance gaps vs all competitors
  bench [--dataset NAME]    Run local decompression benchmark
  path <file.gz>            Trace which decompression path a file takes

OPTIONS:
  --run ID                  Specific GitHub Actions run ID
  --branch NAME             Filter CI runs to a specific branch
  --dataset NAME            silesia, software, logs (default: all available)

EXAMPLES:
  gzippy-dev ci watch                      # Watch latest CI run
  gzippy-dev ci gaps --branch main         # Gaps on main branch
  gzippy-dev bench --dataset silesia       # Local benchmark on silesia
  gzippy-dev path benchmark_data/file.gz   # Which path does this file take?
"#
    );
}
