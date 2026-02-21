---
name: gzippy-ci-analysis
description: Analyze gzippy CI benchmark results from GitHub Actions. Use when checking CI status, reading benchmark results, triaging performance gaps, comparing runs, or investigating regressions.
---

# gzippy CI Analysis

## Fastest Path to Current Status

```bash
gzippy-dev ci triage
```

This is the single most useful command. It fetches the latest completed
Benchmarks run and prints a categorized scorecard.

## Command Reference

### `ci status`
Shows recent CI runs across all workflows.
```
ID           Status     Branch     Workflow                     Title
──────────────────────────────────────────────────────────────────────────
22248973441  success    perf/bgzf  Benchmarks                   perf: route BGZF...
```

### `ci triage [--run ID] [--branch NAME]`
The primary analysis tool. Groups every benchmark scenario into categories:

**Output structure:**
```
═══ TRIAGE: Run 22248973441 ═══

WINS (18):
  ✓ silesia-bgzf T1 x86_64:  gzippy 257.8 vs igzip 245.9  (+4.8%)
  ...

PARITY (14):
  ≈ software-gzip T1 x86_64:  gzippy 402.4 vs igzip 410.9  (-2.1%)
  ...

LOSSES (4):
  ✗ silesia-gzip Tmax x86_64:  gzippy 264.0 vs rapidgzip 314.2  (-16.0%)
    Root cause: No parallel single-member decode
    Action: Requires rapidgzip-style pipeline architecture
  ...
```

### `ci results [--run ID]`
Full benchmark table — every tool, every scenario, sorted by speed.

### `ci gaps [--run ID]`
Shows only scenarios where gzippy is slower than any competitor.

### `ci vs-main [--branch NAME]`
Compares the current branch against main. Shows improvements and regressions.
```
═══ vs main: perf/bgzf-t1-fast-path ═══
  ▲ silesia-bgzf T1 x86_64:  257.8 → 257.8  (+6.5% vs main's 242.0)
  ▼ software-gzip Tmax arm64: 154.7 → 151.2  (-2.3% regression)
```

### `ci push`
All-in-one: pushes current branch, waits for CI to complete (~20-30 min),
then runs triage + vs-main automatically.

### `ci history [--limit N]`
Win rate trend across recent CI runs.

### `ci compare <run_id_a> <run_id_b>`
Direct comparison of gzippy speeds between any two CI runs.

## How CI Benchmarks Work

### Workflow: `benchmarks.yml`
```
Stage 1: Build tools (x86_64 + arm64, parallel)
Stage 2: Prepare benchmark data (one runner, uploads artifact)
Stage 3: Compression benchmarks (36 parallel jobs)
Stage 4: Decompression benchmarks (36 parallel jobs)
Stage 5: Guards + summary (aggregates, posts to PR)
```

Each benchmark job uses Python scripts (`scripts/benchmark_*.py`) that
run adaptive trials (10-40 runs, target CV <3%).

### Data Flow
```
benchmark job → results/*.json artifact
    ↓
guards job → downloads all artifacts → aggregates → summary.md → PR comment
    ↓
gzippy-dev ci → `gh api` to fetch job logs → parses "tool: N MB/s" lines
```

### CI Runner Characteristics
- **x86_64**: `ubuntu-latest` (shared runner, 2-4 cores, variable performance)
- **arm64**: `ubuntu-24.04-arm` (shared Graviton, 2-4 cores, higher variance)
- **Small files (~22MB)** show CV 10-30% — too noisy for definitive verdicts
- **Silesia (211MB)** gives clean signal (CV 1-3%)

## Interpreting Results

### What "PARITY" means
Within measurement noise on shared CI runners. The cloud fleet with dedicated
hardware is needed to break ties. Don't waste time optimizing parity scenarios.

### Known structural losses
These can't be fixed without major architecture changes:
- **gzip/pigz Tmax**: rapidgzip parallelizes single-member files; we don't
- **L1 T1 x86 vs igzip**: igzip uses AVX-512 optimized L1 compressor

### Red flags in CI
- CV > 10% on silesia: runner was contended, results unreliable
- Sudden speed changes on software/logs: noise, not real
- gzippy speed varies > 5% between runs: check for code changes in hot paths
