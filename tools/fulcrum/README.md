# FULCRUM

Causal-mechanistic pipeline profiler for gzippy's parallel pure-Rust gzip
decoder. Finds the **leverage point**: the code region whose speedup moves
the wall the most (wall-elasticity = ∂wall/∂speed), with on/off-critical-path
classification, a per-region mechanism (DRAM-bound / branch-miss /
false-sharing), and a confidence interval.

## Why

Standard profilers report CPU-time SUMS, which **lie** for a pipeline gated by
a critical path: eliminating a 212 ms `absorb_isal_tail` memcpy moved the wall
**0.0%** (the copy was overlapped / off the in-order consumer's critical
path). FULCRUM measures wall-elasticity directly, so a non-lever like that
shows ≈0 and a real lever shows positive — automatically.

## Four layers (over the trace gzippy already emits + Coz + perf)

1. **Causal (Coz virtual speedup)** — primary ∂wall/∂speed per region.
   Hooked via `coz::progress!("chunk_emitted")` at the in-order consumer
   emit + named `begin!/end!` scopes around the four candidate regions
   (`src/fulcrum_probe.rs`, behind the `fulcrum` feature).
2. **Critical-path (wPerf-style)** — consumer-anchored wait attribution
   from the `trace_v2` timeline: the in-order consumer gates the wall, so
   each consumer wait is blamed on the worker span producing the awaited
   chunk. Surfaces the heavy "overshoot" bootstrap chunks.
3. **Mechanistic (Linux perf)** — TMA top-down + PEBS + `perf c2c`,
   attributed per hot function → the WHY behind each lever.
4. **Structural what-if simulator** — STRETCH/v2, documented not built.

## Build

```bash
# analyzer (standalone, builds anywhere)
cargo build --release --manifest-path tools/fulcrum/Cargo.toml

# gzippy profiling binary + harness (x86_64 + pure-rust decode path)
cargo build --profile fulcrum --no-default-features \
  --features pure-rust-inflate,fulcrum --example fulcrum_loop
```

`--profile fulcrum` keeps production-faithful `lto=fat`/`opt3` but retains
line tables (`debug=1`, `strip=false`) so Coz/perf can attribute. The
`fulcrum` feature is zero-cost when off (Coz macros vanish).

## Run on the box

`fulcrum plan` prints the exact frozen-box command sequence (sync, build,
trace, coz, perf, the `interleaved_ab` empirical cross-check, validate,
rank):

```bash
fulcrum plan --repo /root/gzippy --cpus 0,2,4,6,8,10,12,14 --threads 8
```

Then the analyzer subcommands on the pulled artifacts:

```bash
fulcrum critpath  /tmp/fulcrum_tl.json --heavy-ms 30   # critical path + heavy chunks
fulcrum coz-parse /tmp/profile.coz                     # per-region elasticity
fulcrum mech-report /tmp/fulcrum_report.txt            # per-function cycles
fulcrum validate  /tmp/fulcrum_tl.json /tmp/profile.coz  # the trust gate
fulcrum rank      /tmp/fulcrum_tl.json /tmp/profile.coz /tmp/fulcrum_report.txt
```

## Validation (the trust gate)

`fulcrum validate` checks the measured output against the empirical
frozen-A/B oracle:

- `absorb` region elasticity ≈ 0 (the known −0.0% non-lever),
- a decode region (`bulk_inflate`/`bootstrap`) elasticity > 0 (the inline
  match-copy banked +5.2% T16),
- `bulk_inflate` out-levers `absorb`,
- the critical path surfaces ≥1 heavy overshoot bootstrap chunk.

If these don't reproduce, FULCRUM is wrong — fix it before trusting a
ranking.

Coz: <https://github.com/plasma-umass/coz> (`coz = 0.1.3`, dlsym-linked;
the runtime is supplied by `coz run`).
