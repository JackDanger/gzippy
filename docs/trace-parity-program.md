# Trace parity program (all archive formats)

**Goal:** gzippy production decode paths match rapidgzip’s *work distribution* (flamegraph bands), not a target MB/s.

**North-star:** `scripts/profile_diff.py` **L1 band distance** → 0, per format.

## Formats in scope

| Slug | Fixture factory | gzippy route | rapidgzip comparable |
|------|-----------------|--------------|----------------------|
| `single-member` | `gzip -9` preferred; else flate2 | `IsalParallelSM` / `IsalSingle` / libdeflate | yes |
| `multi-member` | N× `gzip -c` concat | `MultiMemberPar` | yes |
| `bgzf` | `bgzip -c` | `GzippyParallel` | yes |
| `gz-subfield` | `gzippy -c -i` | `GzippyParallel` | no (gzippy-only) |

## Frozen measurement (do not change mid-program)

- Source: `benchmark_data/silesia-large.bin` or 24 MiB synthetic (tests).
- Threads: **16** for profile; correctness sweeps **T ∈ {1, 2, 4, 8, 16}**.
- Tooling: `scripts/trace_parity_check.sh` → `cargo test trace_parity` + optional `profile_capture.sh`.

## Phase gates

| Phase | Gate |
|-------|------|
| 0 | Harness + baseline JSON committed |
| 1 | `cargo test trace_parity` green (all formats, all T) |
| 2 | SM: `isal_inflate` Δpp &lt; 10 vs vendor |
| 3 | All comparable formats: L1 &lt; 0.25 |
| 4 | All comparable formats: L1 &lt; 0.10 |

## Per-commit workflow

```bash
cargo test --release trace_parity
# optional on Linux/neurotic:
scripts/trace_parity_check.sh --profile --threads 16 --source benchmark_data/silesia-large.bin
```

Revert if L1 increases or `trace_parity` fails.
