---
name: gzippy-ci-analysis
description: Analyze gzippy CI benchmark results from GitHub Actions. Use when checking CI status, reading benchmark results, triaging performance gaps, or comparing runs.
---

# gzippy CI Analysis

## Fastest Path to Current Status

```bash
gzippy-dev score     # Cloud fleet results (authoritative)
gzippy-dev losses    # Losses grouped by root cause
gzippy-dev ci triage # CI results (noisier, but faster)
```

## Command Reference

### `ci triage [--run ID] [--branch NAME]`
Groups every CI benchmark scenario into WIN / PARITY / LOSS with root causes.

### `ci push`
All-in-one: pushes current branch, waits for CI (~20-30 min), auto-triages.

### `ci vs-main [--branch NAME]`
Compares current branch against main. Shows improvements and regressions.

### `ci history [--limit N]`
Win rate trend across recent CI runs.

### `ci compare <run_id_a> <run_id_b>`
Direct comparison of gzippy speeds between two CI runs.

### `ci status`
Recent CI run statuses.

### `ci results [--run ID]`
Full benchmark table — every tool, every scenario.

### `ci gaps [--run ID]`
Only scenarios where gzippy is slower than any competitor.

## Interpreting Results

- **WIN**: gzippy is fastest
- **PARITY**: within measurement noise (don't optimize these)
- **LOSS**: a competitor is faster

### CI Runner Characteristics
- **x86_64**: ubuntu-latest (shared, 2-4 cores, variable)
- **arm64**: ubuntu-24.04-arm (shared Graviton, higher variance)
- Small files (~22MB) show CV 10-30% — too noisy for verdicts
- Silesia (211MB) gives clean signal (CV 1-3%)

### Red Flags
- CV > 10% on silesia: runner contended, unreliable
- Speed change >5% without touching that path: noise
