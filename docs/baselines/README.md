# Trace parity baselines

Frozen `profile_diff.py` output JSON files. Filename: `trace-parity-YYYY-MM-DD.json`.

Capture:

```bash
scripts/trace_parity_check.sh --profile --format single-member \
  --source benchmark_data/silesia-large.bin --threads 16
```

Do not overwrite a baseline when experimenting; write to `target/tooling/` instead.
