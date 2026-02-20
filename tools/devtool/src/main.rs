//! gzippy-dev: Development tool for the gzippy project.
//!
//! Automates CI monitoring, performance analysis, benchmarking,
//! instrumentation, and project orientation.

mod bench;
mod ci;
mod instrument;
mod orient;
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
                "compare" => {
                    if args.len() < 5 {
                        eprintln!("Usage: gzippy-dev ci compare <run_id_a> <run_id_b>");
                        std::process::exit(1);
                    }
                    ci::compare(&args[3], &args[4])
                }
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
        "instrument" => {
            if args.len() < 3 {
                eprintln!("Usage: gzippy-dev instrument <file.gz> [--threads N]");
                std::process::exit(1);
            }
            let threads = find_flag(&args, "--threads");
            instrument::run(&args[2], threads.as_deref())
        }
        "orient" => orient::run(),
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
  ci status                    Show status of recent CI runs
  ci watch [--run ID]          Block until a CI run completes
  ci results [--run ID]        Parse and display benchmark results
  ci gaps [--run ID]           Show performance gaps vs all competitors
  ci compare <ID_A> <ID_B>     Compare gzippy results between two CI runs
  bench [--dataset NAME]       Run local decompression benchmark
  path <file.gz>               Trace which decompression path a file takes
  instrument <file.gz>         Decompress with detailed timing breakdown
  orient                       Show project state, strategy, and what's next

OPTIONS:
  --run ID                     Specific GitHub Actions run ID
  --branch NAME                Filter CI runs to a specific branch
  --dataset NAME               silesia, software, logs (default: all available)
  --threads N                  Thread count for instrument (default: 1)

EXAMPLES:
  gzippy-dev orient                              # Where are we? What's next?
  gzippy-dev ci gaps                             # Live gap analysis
  gzippy-dev ci compare 22205101167 22206170307  # Did PR help or hurt?
  gzippy-dev instrument data.gz --threads 4      # Timing breakdown
  gzippy-dev bench --dataset silesia             # Local benchmark
"#
    );
}
