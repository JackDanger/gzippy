# CLAUDE.md — gzippy Development Guide

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before `make ship`, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — `make ship` (homelab bench on `neurotic`) is authoritative; local `make` is for iteration.
4. **REVERT REGRESSIONS** — if `make` or `make ship` shows a loss, revert immediately.
5. **NEVER COMPROMISE PERFORMANCE** — clippy, style, readability: none justify slower code.

## Production Routing (Apr 2026)

### Decompression

```
Input → decompress::mod: decompress_gzip_libdeflate
  ├─ gzippy-parallel? ("GZ" subfield in FEXTRA)
  │     → bgzf::decompress_bgzf_parallel (libdeflate FFI, T1 or Tmax internally)
  ├─ Multi-member? (trailing gzip headers detected)
  │     T1  → decompress_multi_member_sequential (libdeflate, member-by-member)
  │     Tmax → bgzf::decompress_multi_member_parallel (libdeflate FFI)
  └─ Single-member?
        ISA-L + T>1 + compressed > 10 MiB
            → parallel::single_member::decompress_parallel
              (v0.6 marker pipeline; ~1.1N total compute work, scales ~T
               from T=2 upward. Phase 1 parallel workers run
               fast_marker_inflate producing Vec<u16> with markers for
               cross-chunk back-refs. Phase 2 sequential resolves markers
               via SIMD replace_markers using each predecessor's last
               32 KB as window, converts u16→u8. Phase 3 verifies CRC and
               size against gzip trailer BEFORE writing — never partial
               output on Err. Counter `MARKER_PIPELINE_RUNS` proves
               production routing called us; see deletion-trap killer
               test in src/tests/routing.rs)
        x86_64 (ISA-L available)        → isal_decompress::decompress_gzip_stream
        any arch, data > 1 GiB (no ISA-L) → decompress_single_member_streaming (zlib-ng)
        default                          → decompress_single_member_libdeflate
```

### Compression

```
T1 L0–L3, ISA-L available → backends::isal_compress::compress_gzip_{to_writer,stream_direct}
T1 L1–L5                  → libdeflate one-shot (ratio probe) or flate2 streaming (zlib-ng)
T1 L6–L9                  → flate2 streaming (zlib-ng)
T>1 L6–L9                 → compress::pipelined::PipelinedGzEncoder → single-member output
T>1 L0–L5                 → compress::parallel::ParallelGzEncoder  → "GZ" subfield multi-block
```

**"GZ" subfield**: gzippy's own parallel format (not standard BGZF). Files produced by
`ParallelGzEncoder` carry a "GZ" FEXTRA subfield with per-block size info; decompression
routes them to `bgzf::decompress_bgzf_parallel`. `PipelinedGzEncoder` output is plain
single-member — decompresses on the single-member path.

## Optimization Branches

There is no `src/experiments/`. Every module on `main` is reachable from a
production code path or is a test fixture / supportive script.

To prototype a new path: add the module under the relevant subsystem
(`src/decompress/`, `src/compress/`, etc.), wire a feature-gated or size-gated
call site in the routing table above, and add a strict correctness test (no
silent fallback). When `make ship` confirms the win, lift the gate. When
abandoned, delete the module — `main` does not host dead code.

Two regression tests lock the parallel single-member wiring:

1. `tests::routing::tests::test_single_member_routing_multithread` — runs
   `decompress_single_member(T=4)` on a 24 MiB input and asserts byte-perfect
   output. Covers "parallel path takes the input" and "falls back correctly
   when speculation fails on adversarial chunks."

2. `tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
   — runs the same fixture at T=1 (sequential) and T=4 (parallel) and asserts
   `parallel_elapsed < 1.3 × sequential_elapsed`. This is the local-CI guard
   that would have caught the v0.3.0 regression (parallel was 1.75× slower
   than sequential).

## Active port: rapidgzip → gzippy parallel single-member

**Before changing anything in `src/decompress/parallel/`, read
`docs/rapidgzip-port-reference.md`.** That file is the living ground
truth: rapidgzip architecture with C++ line citations, gzippy current
state, gap matrix (G1..G13) with status, trace event catalog, and a
pre-commit judgment-call checklist. If a change appears to "work" but
contradicts the reference, the change is suspect.

Capture a trace with `GZIPPY_LOG_FILE=/tmp/sm.log` and analyze via
`scripts/parallel_sm_log_summary.py` before claiming a perf change worked.

## Hard-Won Lessons

**What works**: mmap stdin for multi-threaded (zero-copy, +44%), BufWriter for
stdout, direct FFI, BGZF parallel, 1MB streaming buffer, lock-free parallel.

**What doesn't**: mmap for single-threaded (4x slower from page faults!),
larger blocks for L1 (no help), **speculative parallel decode on arm64**
(16x slower on low-redundancy data — block boundaries are rare, most chunks
become all-marker forcing huge sequential re-decodes),
two-pass scan-then-decode, large pre-allocations.

**arm64 single-member**: currently falls through to libdeflate one-shot (fast enough).
Streaming path only for files > 1 GiB. ISA-L is unavailable on arm64; the parallel
single-member path is gated on `isal_decompress::is_available()` so arm64 never
takes it.

**v0.5.1 — parallel single-member, speculative-window design** (May 2026): replaced
the v0.3.0 "phase-2 redecodes everything sequentially" bug with a true two-pass
parallel design. Phase 1 decodes each chunk with an empty dict in parallel
(ISA-L, ~1500 MB/s/thread). Phase 2 re-decodes chunks 1..T-1 in parallel using
each predecessor's phase-1 last 32 KB as a *speculative* dict — almost always
correct on real data (error propagation requires ≥ chunk_size/32 KB consecutive
near-max-distance back-references). Phase 3 combines per-chunk CRC32s (computed
in phase 2 workers via `crc32fast::Hasher::combine`), verifies against the gzip
trailer, then writes — so a failed speculation never produces partial output
to the writer. Speedup ≈ T/2; ties sequential at T=2 (CI), scales to 4× at T=8.
Wired into `decompress::decompress_single_member` behind
`isal_decompress::is_available() && num_threads > 1 && data.len() > 10 MiB`.
The old "32 KB prefix correction" plan (`docs/parallel-single-member-redesign.md`)
was wrong: cross-chunk back-references resolve to zeros in phase 1, then propagate
forward via chunk-local back-references arbitrarily far — the prefix correction
can't unwind that. Test
`tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
catches regressions of the v0.3.0 class before push.

**v0.3.0 — parallel single-member, BUGGY (superseded by v0.5.1)**: ISA-L
`inflatePrime` re-decodes "confirmed chunks" at non-byte-aligned bit offsets.
But phase 1 stored only `(start_bit, end_bit)` — the decoded bytes were
discarded, and phase 2 ended up re-decoding the entire stream sequentially.
Net result: 1.75× *slower* than sequential. The CI guards at the time were
absolute (vs rapidgzip/pigz) without a "parallel ≥ sequential" floor, so the
regression slipped through. Both the algorithm and the missing guard are fixed
in v0.5.1.

## Branch and PR Workflow

**main is protected** — no direct pushes. Every change goes through a PR.
The pre-push hook (auto-installed by `cargo build`) enforces this locally.

```bash
# Start work
git checkout -b fix/my-change     # or feat/, refactor/, chore/, etc.

# Iterate (same as before)
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null
make
cargo test --release

# Ship
git push origin fix/my-change
gh pr create --fill               # title + body from commit messages
# CI runs → merge when green
```

Emergency bypass (never for performance-affecting changes):
```bash
git push --no-verify origin fix/my-change   # skip local hook only
```

Releasing:
```bash
git tag v0.X.Y && git push origin v0.X.Y   # triggers Release workflow
# Release workflow creates a formula-update PR and auto-merges it
```

## Iteration Loop

```bash
# 1. Make one focused change
# 2. Check routing
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null   # Shows which path is taken

# 3. Local sanity (30s) — catches catastrophic regressions
make

# 4. Correctness
cargo test --release

# 5. Authoritative numbers — only after make passes
make ship   # SSH to neurotic homelab, runs gzippy-dev bench
```

`make route-check` — generates 1MB+10MB test files and shows routing + timing
vs pigz for all four combos (T1/T4 × 1MB/10MB). Use this before ANY decompression change.

## Key Files

| File | Role |
|------|------|
| `src/decompress/mod.rs` | Decompression entry, format detect, routing |
| `src/decompress/bgzf.rs` | gzippy-parallel + multi-member parallel (core engine) |
| `src/decompress/scan_inflate.rs` | Streaming scan-and-inflate path |
| `src/decompress/parallel/single_member.rs` | v0.3.0 parallel SM — ISA-L `inflatePrime` |
| `src/decompress/parallel/{block_finder,marker_decode,ultra_fast_inflate}.rs` | Speculation supporting primitives |
| `src/decompress/inflate/consume_first_decode.rs` | Pure-Rust inflate (production helpers used by `bgzf`, `scan_inflate`) |
| `src/decompress/inflate/{consume_first_table,jit_decode,libdeflate_decode,libdeflate_entry,specialized_decode,vector_huffman,double_literal,bmi2}.rs` | Huffman/inflate building blocks |
| `src/decompress/{combined_lut,inflate_tables,packed_lut,simd_copy,simd_huffman,two_level_table}.rs` | SIMD + LUT primitives shared with bgzf |
| `src/backends/isal_decompress.rs` | ISA-L streaming inflate (x86_64 production path) |
| `src/backends/inflate_bit.rs` | Universal inflate-from-bit (ISA-L on x86_64, libz-ng elsewhere) |
| `src/backends/libdeflate.rs` | libdeflate FFI — `gzip_decompress_ex` |
| `src/compress/mod.rs` | Compression entry, routing |
| `src/compress/parallel.rs` | ParallelGzEncoder — T>1 L0–L5, "GZ" multi-block output |
| `src/compress/pipelined.rs` | PipelinedGzEncoder — T>1 L6–L9, single-member output |
| `src/backends/isal_compress.rs` | ISA-L compression (x86_64 T1 L0–L3) |
