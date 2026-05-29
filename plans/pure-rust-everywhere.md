# NORTH STAR: pure-Rust DEFLATE everywhere — delete libdeflate / ISA-L / zlib-ng

**User directive (2026-05-29, definitive):** "Take that large, hard-to-build
Rust implementation of the DEFLATE decoder that we were comparing to
rapidgzip, and polish it and generalize it and optimize it until we do NOT
use libdeflate, ISA-L, or zlib-ng anywhere in our production path."

This is the project. Everything else (the lever grind, the segmented-buffer
idea, the C-vs-C cell comparisons) is subordinate to this. The deliverable is
a pure-Rust decoder that is byte-exact on every gzip-family input and FASTER
than libdeflate / ISA-L / zlib-ng / rapidgzip on every archive type and
thread count — shipped as the only decode path, with the C FFI deleted.

## The seed (measured 2026-05-29, frozen/pinned/best-of-7, silesia-large)

gzippy's own pure-Rust `ResumableInflate2` (the `--features pure-rust-inflate`
build), byte-exact (md5 c070ed84):

| T  | gzippy-PURE (own Rust) | gzippy-ISAL (C) | rapidgzip | libdeflate (ST) |
|----|------------------------|------------------|-----------|------------------|
| 1  | 898  | 1083 | 884  | ~1083 |
| 4  | 1150 | 1378 | 1726 | — |
| 8  | 1623 | 1835 | 2235 | — |
| 16 | 1211 | 1477 | 2340 | — |

Targets to beat (all): ISA-L 1835, rapidgzip 2235 (T8). Pure is currently
~12% behind ISA-L on the engine and ~27% behind rapidgzip overall.

## Where C is used today (the demolition list)

Audit (verified file:line, see [[project audit in session]]):
- **C ISA-L FFI** — x86_64 single-member parallel CLEAN chunks
  (`IsalInflateWrapper` under `isal-compression`, `inflate_wrapper.rs:160`);
  x86_64 single-member T1/<10MiB (`isal_decompress::decompress_gzip_stream`).
- **C libdeflate FFI** — multi-member (T1 + Tmax), BGZF / "GZ"-parallel
  (`bgzf.rs:473`), AND all of arm64/macOS single-member (default features).
- **C zlib-ng** — single-member > 1 GiB streaming.
- **gzippy pure-Rust ALREADY** — the window-absent bootstrap
  (`deflate_block::Block`) in BOTH builds; `ResumableInflate2` clean decode
  under `pure-rust-inflate` only.

Cargo deps to eventually remove: `isal-rs`, `libdeflate-sys`, `zlib-ng`/flate2
backend. Keep flate2/zlib-ng only as a TEST oracle, never in the prod path.

## Roadmap (each step: byte-exact differential vs flate2+libdeflate oracle; commit)

### Phase A — make the pure-Rust decoder the fastest DEFLATE decoder

A1. **Bootstrap overshoot (THE biggest single lever).** The window-absent
    decode (`deflate_block::Block`) runs on ~31% / 156 MB of silesia because
    silesia's 2–14 MB deflate blocks delay the handoff to the fast clean
    path. It owns the rapidgzip gap even on the C-ISA-L build (which is why
    gzippy-ISAL 1835 < rapidgzip 2235 despite IPC parity 1.44 vs 1.42). FIX:
    the window-absent decoder must be as fast as the clean one. Note mid-
    block handoff to ISA-L is impossible (DEFLATE blocks enter at headers),
    so the answer is NOT "hand off earlier" — it is **unify the window-absent
    path onto the optimized `ResumableInflate2` inner loop** (emit markers
    only for refs into the unknown predecessor window; everything else uses
    the FASTLOOP). rapidgzip's own block decoder is fast; match it.
A2. **Inner-loop engine: close 12% to ISA-L, then beat it.** bad-spec
    (inner-loop branch mispredict) + the 1.25× algorithmic gap vs libdeflate
    (per [[project_inner_loop_resumable_tax]]). Open territory per CLAUDE.md.
    Bench via `examples/inner_bench.rs` (instructions/byte) + frozen wall.

### Phase B — generalize pure-Rust to every production path

B1. single-member T1 / small / >1 GiB → pure-Rust (replace ISA-L stream +
    zlib-ng streaming). A non-parallel pure-Rust streaming inflate.
B2. multi-member (T1 + parallel) → pure-Rust (replace libdeflate).
B3. BGZF / "GZ"-parallel → pure-Rust (replace `bgzf.rs` libdeflate FFI).
    Independent-block decode is the EASY case (empty window, no markers) —
    the pure clean decoder should already win here; BGZF clean parallel hit
    6100 MB/s with libdeflate, so the bar is high but the structure is simple.
B4. arm64 / macOS → pure-Rust (replace libdeflate one-shot). NEON SIMD for
    the match-copy + table lookups (per CLAUDE.md arch-specific is in scope).

### Phase C — flip and demolish

C1. Make `pure-rust-inflate` the DEFAULT feature; route every format to it.
C2. Delete `isal-rs`, `libdeflate-sys`, zlib-ng-as-decoder from Cargo + code.
    Keep them ONLY behind a `dev`/`oracle` feature for differential tests.
C3. Update CLAUDE.md routing tables + the release workflow feature flags.

## Guardrails

- Correctness is the prime directive. Every step: silesia md5 c070ed84 +
  multi-member differential + corpus differential vs flate2 AND libdeflate.
- Measure on the box (x86_64 i7-13700T, LXC 199) frozen (LXC 111/105) +
  P-core pinned, best-of-N, same-session A/B. T8 (8 P-cores) is the reliable
  scaling point; T16 is HT-contended on this box (8 P-cores) — not a clean
  test. Build the pure binary: `--no-default-features --features
  pure-rust-inflate`.
- No hard reset of the inner-loop work (f01eb74/d4f2497) — it is the seed.
- HONEST: this is a multi-week-to-multi-month effort. Commit per step.

## First action

**Phase A1 — the bootstrap overshoot**, because it owns the largest share of
the gap, is already pure-Rust, and benefits the engine on every input with
large deflate blocks. Start by unifying the window-absent decode onto the
optimized inner loop (or porting the FASTLOOP into `deflate_block::Block`),
measured against the `bootstrap.outcome` slow-path-share + T8 wall.
