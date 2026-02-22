---
name: gzippy-performance-workflow
description: Run gzippy performance benchmarks, interpret results, and iterate on optimizations. Use when benchmarking, optimizing performance, checking scores, or comparing speeds against competitors.
---

# gzippy Performance Workflow

## The Loop

```
1. gzippy-dev score                  # Current win/loss from cloud fleet
2. gzippy-dev losses                 # Losses grouped by root cause + actions
3. # ... make ONE focused code change ...
4. cargo test --release              # Correctness
5. gzippy-dev ci push                # Push → CI → auto-triage
6. source .env && gzippy-dev cloud bench  # Authoritative numbers on dedicated HW
```

## Quick Reference

### Scorecard (first thing to check)
```bash
gzippy-dev score                  # Win/loss scorecard from cloud-results.json
gzippy-dev losses                 # Root-cause-grouped losses with actions
```

### CI (GitHub Actions — regressions)
```bash
gzippy-dev ci triage              # Categorized gaps with root causes
gzippy-dev ci push                # Push, wait for CI, auto-triage
gzippy-dev ci vs-main             # Compare current branch vs main
gzippy-dev ci history             # Win rate trend across runs
```

### Cloud Fleet (dedicated EC2 — absolute speed)
```bash
source .env && gzippy-dev cloud bench   # 12 instances, ~20 min, ~$0.50
gzippy-dev cloud cleanup               # Delete leaked resources
```

### Local Benchmarks (fast iteration, not authoritative)
```bash
gzippy-dev bench                       # Decompress all datasets
gzippy-dev bench --direction compress  # Compress only
gzippy-dev bench ab <ref-a> <ref-b>    # A/B compare two git refs
```

### Diagnostics
```bash
gzippy-dev path <file.gz>              # Which decompression path?
gzippy-dev instrument <file.gz>        # Timing breakdown
gzippy-dev orient                      # Project strategy overview
```

## Rules

1. **ONE change at a time** — never batch multiple optimizations
2. **Benchmark BEFORE and AFTER** — record the baseline first
3. **Revert regressions immediately** — don't try to fix forward
4. **CI for regressions, cloud fleet for absolute speed**
5. **Small files (software, logs) are noisy on CI** — use cloud fleet
