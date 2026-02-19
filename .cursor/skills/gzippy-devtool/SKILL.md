---
name: gzippy-devtool
description: Use gzippy-dev CLI tool for CI monitoring, performance analysis, benchmarking, and path tracing. Use when checking CI results, analyzing performance gaps, running benchmarks, diagnosing decompression paths, or after pushing a PR to watch for results.
---

# gzippy-dev Tool

The `gzippy-dev` binary lives at `tools/devtool/` and is built with `cargo build --release` from that directory. A convenience symlink at the repo root (`./gzippy-dev`) points to the built binary.

## Build

```bash
cd tools/devtool && cargo build --release
```

## Commands

### CI Status — See Recent Runs

```bash
./gzippy-dev ci status
./gzippy-dev ci status --branch main
```

Shows the last 10 CI runs with IDs, status, branch, and workflow name.

### CI Watch — Block Until Complete

```bash
./gzippy-dev ci watch                     # Latest Benchmarks run
./gzippy-dev ci watch --run 22194843448   # Specific run
./gzippy-dev ci watch --branch my-branch  # Latest on branch
```

Polls every 30 seconds (up to 45 minutes). When complete, prints full results and gap analysis.

**Use this after pushing a PR** to get results without manual polling.

### CI Results — Parse Completed Run

```bash
./gzippy-dev ci results                     # Latest Benchmarks run
./gzippy-dev ci results --run 22194843448   # Specific run
```

Outputs a grouped table of every benchmark result (tool, speed, trials, ratio) organized by scenario.

### CI Gaps — Performance Gap Analysis

```bash
./gzippy-dev ci gaps                     # Latest Benchmarks run
./gzippy-dev ci gaps --branch main       # Latest on main
```

**This is the most important command.** It compares gzippy against every competitor and produces:

1. **GAPS** — Sorted list of scenarios where gzippy is slower, with percentage
2. **WINS** — Scenarios where gzippy is faster
3. **PRIORITY ACTIONS** — Top 5 gaps with diagnosed root causes

### Local Benchmark — Run Decompression Tests

```bash
./gzippy-dev bench                         # All available datasets
./gzippy-dev bench --dataset silesia       # Specific dataset
```

Runs adaptive benchmarking (5-30 trials, target 5% CV) against all installed competitor tools. Reports throughput, stddev, and relative performance.

Requires `benchmark_data/` directory with `.gz` files and gzippy built (`cargo build --release`).

### Path Trace — Diagnose Decompression Route

```bash
./gzippy-dev path benchmark_data/silesia-large.gz
```

Shows exactly which decompression code path a file will take:
- BGZF parallel vs multi-member parallel vs single-member sequential
- Member count and sizes
- Expected thread usage and throughput

## Workflow: After Pushing Changes

1. `./gzippy-dev ci watch` — block until CI finishes
2. Read the gap analysis output
3. If gaps remain, investigate the top priority action
4. Make targeted changes, rebuild, `./gzippy-dev bench` to verify locally
5. Push and repeat

## Workflow: Before Starting Optimization Work

1. `./gzippy-dev ci gaps --branch main` — understand current baseline
2. `./gzippy-dev path <target-file>` — confirm which code path to optimize
3. Make changes
4. `./gzippy-dev bench --dataset <dataset>` — verify locally
5. Push PR and `./gzippy-dev ci watch`
