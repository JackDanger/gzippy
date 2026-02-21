---
name: gzippy-performance-workflow
description: Run gzippy performance benchmarks, interpret results, and iterate on optimizations. Use when benchmarking, optimizing performance, running CI, checking results, or comparing speeds against competitors like pigz, igzip, and rapidgzip.
---

# gzippy Performance Workflow

## The Loop

Every performance change follows this cycle:

```
1. gzippy-dev ci triage          # Where do we stand?
2. # ... make ONE focused code change ...
3. cargo test --release           # Correctness
4. gzippy-dev ci push             # Push → CI → auto-triage
5. gzippy-dev ci vs-main          # Compare against main
```

For high-precision numbers on dedicated hardware, use the cloud fleet:

```
source .env && gzippy-dev cloud bench   # 12 EC2 instances, ~20 min
```

## Quick Reference: All Commands

### CI (GitHub Actions — source of truth for regressions)
```bash
gzippy-dev ci status             # Recent CI run statuses
gzippy-dev ci triage             # Categorized gaps with root causes
gzippy-dev ci results            # Full benchmark table
gzippy-dev ci gaps               # Performance gaps vs competitors
gzippy-dev ci vs-main            # Compare current branch vs main
gzippy-dev ci push               # Push, wait for CI, auto-triage
gzippy-dev ci history            # Win rate trend across runs
gzippy-dev ci compare <A> <B>    # Compare two CI run IDs
```

### Cloud Fleet (dedicated EC2 hardware — source of truth for absolute speed)
```bash
source .env                      # Load AWS credentials
gzippy-dev cloud bench           # Launch 12-instance fleet
gzippy-dev cloud cleanup         # Delete leaked resources
```

If `aws-vault` hangs, ensure `AWS_ACCESS_KEY_ID` is in `.env` — the tool
detects env vars and bypasses `aws-vault` automatically.

### Local Benchmarks (fast iteration, not authoritative)
```bash
gzippy-dev bench                           # Decompression, all datasets
gzippy-dev bench --direction compress      # Compression only
gzippy-dev bench --direction both          # Everything
gzippy-dev bench --dataset silesia         # Single dataset
gzippy-dev bench --json                    # Machine-readable
gzippy-dev bench ab <ref-a> <ref-b>        # A/B compare two git refs
```

### Makefile (build + quick local tests)
```bash
make build          # Build gzippy
make deps           # Build all competitor tools
make bench          # Quick benchmark (gzippy only, L6)
make bench-all      # Full multi-tool comparison
make lint           # cargo fmt + clippy
```

### Diagnostics
```bash
gzippy-dev path <file.gz>                  # Which decompression path?
gzippy-dev instrument <file.gz>            # Timing breakdown
gzippy-dev orient                          # Project strategy overview
```

## Reading Results

### CI Triage Output

`gzippy-dev ci triage` categorizes every scenario into:

- **WIN**: gzippy is fastest
- **PARITY**: within measurement noise
- **LOSS**: a competitor is faster — includes root cause and suggested action

Focus on LOSS scenarios. Each one tells you exactly what to fix.

### Cloud Fleet Output

The fleet prints results as they stream in:

```
silesia-bgzf T1 gzippy: 575.2 MB/s (CV 0.3%)
silesia-bgzf T1 igzip: 521.0 MB/s (CV 0.3%)
```

Low CV (<1.5%) means clean signal. Results are authoritative.
At the end, a scorecard shows WIN/LOSS/PARITY for every scenario.

### Key Metrics

| Metric | Meaning |
|--------|---------|
| MB/s | Throughput (uncompressed bytes / wall time) |
| CV | Coefficient of variation — lower = less noise |
| ratio | Compressed size / original size (compression only) |

## Benchmark Matrix

### Decompression: 36 scenarios
- **Datasets**: silesia (211MB), software (~22MB), logs (~22MB)
- **Archives**: gzip (single-member), bgzf (gzippy format), pigz
- **Threads**: T1, Tmax (4 on CI/cloud)
- **Platforms**: x86_64, arm64
- **Competitors**: unpigz, igzip, rapidgzip, gzip

### Compression: 36 scenarios
- **Datasets**: silesia, software, logs
- **Levels**: L1, L6, L9
- **Threads**: T1, Tmax
- **Platforms**: x86_64, arm64
- **Competitors**: pigz, igzip (L1 only), gzip

## Rules

1. **ONE change at a time** — never batch multiple optimizations
2. **Benchmark BEFORE and AFTER** — record the baseline first
3. **Revert regressions immediately** — don't try to fix forward
4. **CI is for regressions, cloud fleet is for absolute speed**
5. **Small files (software, logs) are noisy on CI** — use cloud fleet for signal
6. **Never use `-C target-cpu=native` in CI** — causes SIGILL on heterogeneous runners
