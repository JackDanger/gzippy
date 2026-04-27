# Contributing to gzippy

Welcome. This guide gets you productive in the codebase. The authoritative
spec for routing, lessons, and PR workflow lives in [`CLAUDE.md`](CLAUDE.md);
this file is the on-ramp.

## What gzippy is

A drop-in replacement for `gzip` and `pigz` that aims to be the fastest gzip
implementation on any hardware. Same RFC 1952 output — any decompressor on
Earth reads gzippy's files.

## Quick start

```bash
git clone --recursive https://github.com/JackDanger/gzippy
cd gzippy
cargo build --release
cargo test --release
./target/release/gzippy --help
```

## Architecture

```
gzippy file.txt
     │
     ▼
┌─────────────┐
│   main.rs   │   parse args, route to compress / decompress
└──────┬──────┘
       │
       ├─────────────────────────┐
       ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│ compress::mod    │     │ decompress::mod  │
│ (routing)        │     │ (classify+route) │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ├──── T>1 L6-L9 ──┐      ├──── parallel SM ──┐
         │                 │      │  (ISA-L+T>1+>10MB)│
         ▼                 ▼      ▼                   ▼
┌────────────────┐ ┌──────────────┐ ┌─────────────────────┐
│ compress::      │ │ compress::   │ │ decompress::        │
│ pipelined       │ │ parallel     │ │ parallel::          │
│ (single-member  │ │ (GZ subfield │ │ single_member       │
│  zlib-ng)       │ │  multi-block │ │ (rapidgzip-style    │
└────────────────┘ └──────────────┘ │  + ISA-L inflatePrime)│
                                    └─────────────────────┘
```

Full routing table (every path the CLI can take, with thread-count and
size gates) is in [`CLAUDE.md` § Production Routing](CLAUDE.md#production-routing-apr-2026).

## File guide

| Path | Role |
|------|------|
| `src/main.rs` | CLI entry point, signal handling, file iteration |
| `src/cli.rs` | gzip-compatible argument parsing |
| `src/compress/mod.rs` | Compression routing (level + thread count → encoder) |
| `src/compress/parallel.rs` | T>1 L0–L5: independent blocks, "GZ" FEXTRA subfield |
| `src/compress/pipelined.rs` | T>1 L6–L9: dictionary-shared single-member output |
| `src/compress/io.rs` | Compression I/O: stdin/file in, stdout/file out |
| `src/decompress/mod.rs` | Decompression entry, format detection, routing |
| `src/decompress/bgzf.rs` | gzippy-parallel + multi-member parallel core engine |
| `src/decompress/parallel/single_member.rs` | Parallel single-member (v0.3.0) |
| `src/decompress/inflate/` | Pure-Rust inflate primitives + Huffman tables |
| `src/backends/isal_compress.rs` | ISA-L compression (x86_64, T1 L0–L3) |
| `src/backends/isal_decompress.rs` | ISA-L decompression (x86_64) |
| `src/backends/libdeflate.rs` | libdeflate FFI |
| `src/backends/inflate_bit.rs` | Universal inflate-from-bit (ISA-L on x86_64, libz-ng elsewhere) |
| `src/infra/scheduler.rs` | Compression block scheduler (pigz N+1 model) |
| `src/infra/thread_pool.rs` | Worker thread pool |
| `src/tests/` | Cross-cutting integration tests (routing, fixtures, oracles) |

## Iteration loop

```bash
# 1. Make a focused change.
# 2. Check which path the CLI takes.
GZIPPY_DEBUG=1 ./target/release/gzippy -d -c testfile.gz > /dev/null

# 3. Local sanity (≈30s) — catches catastrophic regressions.
make

# 4. Correctness.
cargo test --release

# 5. Authoritative numbers — only after step 3 passes.
source .env && make ship
```

`make route-check` generates 1 MiB and 10 MiB test files and shows routing
+ timing vs pigz across T1/T4 × 1 MiB/10 MiB. Run it before any
decompression-routing change.

## PR workflow

`main` is protected — every change goes through a PR. The pre-push hook
(installed by `cargo build`) enforces this locally.

```bash
git checkout -b fix/my-change       # or feat/, refactor/, chore/, perf/
# … make changes, commit …
git push origin fix/my-change
gh pr create --fill
# CI runs → maintainer merges when green
```

Releases are tag-driven: `git tag v0.X.Y && git push origin v0.X.Y` triggers
the Release workflow, which opens a formula-update PR.

## Pre-commit checks

The pre-commit hook runs:

1. `cargo fmt --check`
2. `cargo clippy --release --all-targets -- -D warnings`

If lint fails:

```bash
cargo fmt
cargo clippy --fix --allow-staged --release --all-targets
```

## PR checklist

- [ ] `cargo test --release` passes
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --release --all-targets -- -D warnings` clean
- [ ] `make` (or `make quick`) passes
- [ ] `make ship` (homelab bench) shows no regression

## House rules

Before changing decompression routing, read [`CLAUDE.md`'s Hard-Won
Lessons](CLAUDE.md#hard-won-lessons). It records what's been tried and why
some "obvious" optimizations make things slower (mmap on single-thread,
two-pass scan-then-decode, speculative parallel on arm64, etc.). Don't
relitigate without reading it first.

## Questions?

Open an issue.
