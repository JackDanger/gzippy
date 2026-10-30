# Contributing to gzippy

Welcome. This guide gets you productive in the codebase. The project charter (project rules, work discipline) lives in
[`CLAUDE.md`](CLAUDE.md); this file is the on-ramp.

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
       ├───────────────────────────┐
       ▼                           ▼
┌──────────────────┐     ┌──────────────────────┐
│ compress::mod    │     │ decompress::mod      │
│ (routing)        │     │ (classify → Decode-  │
│ level+threads →  │     │  Path: format +      │
│ encoder          │     │  thread count)       │
└────────┬─────────┘     └──────────┬───────────┘
         │                         │
         ├─ T1, L0–L12 ──┐         ├─ "GZ" FEXTRA subfield → bgzf/
         │               │         │     parallel member decode
         │               │         ├─ multi-member → MultiMemberPar /
         │               │         │     MultiMemberGrid / MultiMemberSeq
         │               │         └─ single-member → parallel::
         │               │             single_member (ParallelSM — the
         ▼               ▼             SOLE single-member path, every
┌───────────────────┐ ┌───────────────────┐ size × thread count × arch;
│ deflate (pure T1: │ │ pipelined (pure   │ StoredParallel when stored-
│ whole-buffer      │ │ T>1: PipelinedGz- │ blocks dominate)
│ encode: the ldx   │ │ Encoder::compress_│
│ libdeflate port   │ │ buffer_pure —     │
│ L0–9 except       │ │ standard single-  │
│ near-optimal      │ │ gzip at every     │
│ parse L10–12)     │ │ level)            │
└───────────────────┘ └───────────────────┘
```

Both compress arms above run the pure-Rust DEFLATE engine in EVERY build
flavor (Increment 7): T1 routes through
`deflate::encode_gzip_reader_to_writer_sized` (with a byte-identical mmap
fast path for FILE inputs over 128 KiB), T>1 through
`PipelinedGzEncoder::compress_buffer_pure`. One encode per input — no trial
encodes. Which engine serves a level is `level_uses_ldx`
(`src/compress/deflate/mod.rs`); `tests/one_encode_only.rs` counts encoder
entries so a second-encode regression fails instead of asserting.

**One C footnote.** The CI x86_64 flavor builds with
`--features gzippy-native,isal-compression` (`.github/workflows/ci.yml`).
`isal-compression` brings in the vendored ISA-L C compressor as a
DIFFERENTIAL-ORACLE LIBRARY ONLY: `src/backends/isal_compress.rs` compiles
under that feature (also under `test` / `ffi-oracle`) but is
`#![allow(dead_code)]` with no production call site — the routing graph
never invokes it, in any flavor. The same feature enables rpmalloc chunk
arenas for the ParallelSM decoder (`arena-allocator`). ci.yml's comment
about ISA-L *compression* "(T1 L0–L3)" describes the pre-Increment-7
routing and is historical; the routing functions — not the build flavor —
decide where your bytes run.

The full decode table lives in the `DecodePath` enum and `classify_gzip`
(`src/decompress/mod.rs`), and the compress routes are documented at
`src/compress/mod.rs`'s top (`route` module).

## File guide

| Path | Role |
|------|------|
| `src/main.rs` | CLI entry point, signal handling, file iteration |
| `src/cli.rs` | gzip-compatible argument parsing |
| `src/compress/mod.rs` | Compression routing (level + thread count → encoder) |
| `src/compress/io.rs` | Compression I/O: stdin/file in, stdout/file out |
| `src/compress/deflate/` | The whole-buffer DEFLATE engine: level params, parsers, matchfinders; `level_uses_ldx` and the one-encode census live here |
| `src/compress/ldx/` | The libdeflate port — production encoder for L0–9 except L1/L6, and the per-decision differential oracle |
| `src/compress/pipelined.rs` | T>1: the pure parallel single-member pipeline (`compress_buffer_pure`), every level |
| `src/compress/parallel.rs` | Shared format helpers (`GzipHeaderInfo`, the "GZ" subfield ID, rsyncable chunking) + C-FFI differential-oracle encoders (test / `ffi-oracle` only) |
| `src/decompress/mod.rs` | Decompression entry, format detection, routing (the `DecodePath` table) |
| `src/decompress/bgzf/` | "GZ"-subfield and BGZF parallel member decode + member-boundary scanning |
| `src/decompress/parallel/single_member.rs` | The ParallelSM driver — sole single-member decode path, pure Rust |
| `src/decompress/inflate/` | Pure-Rust inflate primitives + Huffman tables |
| `src/backends/isal_compress.rs` | ISA-L compression — differential oracle only (test / `ffi-oracle` / `isal-compression` builds; off the routing graph) |
| `src/backends/isal_decompress.rs` | ISA-L decode FFI differential oracle (+ the resumable from-bit oracle export in `lib.rs`) |
| `src/backends/libdeflate.rs` | libdeflate decode FFI differential oracle (tests only) |
| `src/backends/inflate_bit.rs` | Resumable inflate-from-bit decode FFI oracle (tests only) |
| `src/infra/scheduler.rs` | Compression block scheduler (pigz N+1 model) — drives the T>1 pipelined path |
| `src/infra/thread_pool.rs` | Persistent worker pool for parallel decompression |
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
make ship
```

`make ship` is homelab-bench-limited: after the local checks it PUSHES the
current branch and runs the benchmark on the remote homelab box (SSH
`neurotic ...`, ~20–25 min for step 6 — see the `ship` target in the
Makefile). Without access to that box, run `make ship-local` — steps 1–5
(fmt, tests, clippy, corpus oracle, smoke) and skip the bench.

`make route-check` generates 1 MiB and 10 MiB test files and shows routing
+ timing vs pigz across T1/T4 × 1 MiB/10 MiB. Run it before any
decompression-routing change.

## PR workflow

`main` is protected — every change goes through a PR. The pre-push hook
(installed by `cargo build` via `build.rs`) enforces this locally: it
refuses direct pushes to `main` and runs
`scripts/verify_submodule_remotes.sh`. CI reports a single required status
check named `build` — the aggregate job in `.github/workflows/ci.yml` that
fans the Build & Test / Integration / Anatomy / Lint jobs into one name.

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

The pre-commit hook (also installed by `build.rs`) runs only for the file
kinds staged:

1. staged `.rs` files → `cargo fmt --check`, then
   `cargo clippy --all-targets -- -D warnings` (default features)
2. staged `VERSION` / `Cargo.toml` → VERSION-vs-Cargo.toml consistency
3. staged `.github/**.yml` → YAML syntax check

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
- [ ] if you have homelab access: `make ship` shows no regression
      (otherwise `make ship-local` covers the local part)

## House rules

Before changing decompression routing, read [`CLAUDE.md`](CLAUDE.md) — in
particular its *Non-negotiables* and *Hard stops* sections. They record
what the project counts as closed and why some "obvious"
optimizations lose (mmap on single-thread, two-pass scan-then-decode,
speculative parallel on arm64, etc.). Don't relitigate without reading
them first.

## Questions?

Open an issue.
