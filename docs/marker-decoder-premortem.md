# Premortem — v0.6 Marker-Based Parallel Single-Member Decompressor

> A premortem is written **before** the work, listing every plausible way it
> can fail and the mitigation that ships *together with* the code at risk.
> If a mitigation requires future discipline, it doesn't count — only
> structural defenses (tests, asserts, gates, atomic counters) do.
>
> This document is the source-of-truth referenced by
> `src/decompress/parallel/RATIONALE.md`, `docs/marker-decoder-plan.md`, and
> PRs #90 / #93 / #94. Earlier rollouts cited "PR #90 review comments" as
> the premortem location — that was never actually written there. This file
> is the durable artifact.

---

## The thing we are trying to do

Replace the v0.5.1 speculative-window two-pass parallel single-member
decompressor (correct, but ~2N total compute work, lost to rapidgzip at 4
cores) with a **rapidgzip-style marker pipeline**: each chunk decodes its
deflate range in parallel to a `Vec<u16>` where cross-chunk back-references
appear as marker sentinels (`MARKER_BASE + offset`); a final sequential
pass resolves the markers against the predecessor chunk's last 32 KiB using
SIMD (`replace_markers_avx2` / `replace_markers_neon`). Total compute work
is **~1.1N** (one decode + cheap marker resolution), so speedup scales
linearly with `T` from T=2 upward. Target: ≥ 0.99× rapidgzip on the CI 4-
core x86_64 runner, ≥ pigz on every hardware we test.

## How prior attempts died

Every attempt to ship a marker decoder in gzippy died for one of the
reasons below. The patterns repeat. The mitigations below are designed to
make each pattern *impossible*, not "remembered."

### F1. The "22 MB/s pure-Rust" failure  (`b3bf6df`)

The first marker decoder decoded bit-by-bit, cloned its input on
construction, and had no canonical-Huffman lookup table. It hit 22 MB/s on
arm64 — **slower than sequential ISA-L (~1500 MB/s/thread)** — so a T=4
parallel run was 17× slower than just calling ISA-L sequentially. The
pipeline was correct but pointless, the next cleanup deleted it.

### F2. The "wrong API" failure  (`4bbf04f`, commit message in `RATIONALE.md`)

`marker_decode::MarkerDecoder` required *byte-aligned* input but production
chunk starts are *bit-aligned* (offsets 0..7 within a byte). Every
production call returned `None`. Output tests still passed because routing
silently fell back to sequential ISA-L. The marker pipeline was dead code
in production for months before the commit that deleted it noticed.

### F3. The "9300-line dead-experiment cleanup"  (`3eba641`)

`hyper_parallel.rs`, `simd_parallel_decode.rs`, `parallel_decompress.rs`,
`parallel_inflate.rs`, `ultra_decompress.rs` — five modules carrying the
good SIMD work (the `replace_markers_avx2` we ultimately had to *resurrect*
from git history). All five were deleted together because none were wired
into the CLI and nothing called them. They looked dead because they *were*
dead — but they contained the seeds of what we now need. The cleanup was
correct given what was visible; the failure was upstream: the work had
never been wired in.

### F4. The "v0.3.0 phase-1-discards-bytes" failure

`inflatePrime` re-decode design: phase 1 stored only `(start_bit,
end_bit)` per chunk and **discarded the decoded bytes**. Phase 2 then re-
decoded the entire stream sequentially. Net result: 1.75× *slower* than
sequential. The CI guards at the time were *absolute* (vs rapidgzip /
pigz) without a "parallel ≥ sequential" floor, so the regression slipped
through review and into a release.

### F5. The "v0.5.1 ratio shortfall"

Correct two-pass speculative-window design fixed F4, but did **2N total
compute work** — each chunk decoded twice (once with empty dict, once with
the predecessor's last 32 KiB). On a 4-physical-core CI runner: 288 MB/s
vs. rapidgzip's 327 MB/s = 0.88×. Below the 0.99 floor the project wants.
Algorithmically optimal for *that* design; the design itself was wrong for
the perf target.

### F6. The "false-positive boundary" failure  (current red CI on PR #90)

`try_decode_at` validates a candidate boundary by asking ISA-L to decode
32 KiB from it. ISA-L is permissive — it accepts some positions that
happen to decode plausible-looking bytes for 32 KiB before diverging from
the real stream. The stricter `fast_marker_inflate` later rejects these
boundaries mid-block. `decompress_parallel` returns `Err`. Routing
silently falls back to sequential ISA-L. **Output tests pass.** Only the
deletion-trap killer counter test catches that the marker pipeline didn't
actually run.

---

## Failure-mode catalogue with structural mitigations

Each row has a unique tag. Mitigations live in code that ships in the same
PR as the risk; this document is the index, not the implementation.

### A. The marker decoder is too slow to win

| Tag | Failure | Structural mitigation |
|---|---|---|
| A1 | `Vec<u16>` output bandwidth bottleneck (2× store width) makes per-thread throughput sub-ISA-L | **Synthetic bench checked in: `examples/u16_output_cost.rs`.** Measured ratio 1.03 on arm64. Decision is data-justified, not vibes. Result recorded in `docs/marker-decoder-plan.md`. |
| A2 | Pure-Rust Huffman decode at < ISA-L speed loses on T=2 | **`#[ignore]` throughput test `throughput_vs_oracle`** in `fast_marker_inflate.rs` compares against `inflate_consume_first` (production u8 decoder). Acceptance: ≥ ISA-L/2 per thread so T=2 ties sequential, T=3 wins. Failed measurement on x86_64 is what triggers PR #95 (SIMD inner loop) — see F2 / A6 below. |
| A3 | Phase-1 chunk-Vec growth via `push` causes capacity-doubling reallocations on hot path | `decode_chunk_markers_bounded` pre-allocates `Vec<u16>` to `4 × deflate_bytes` (matches deflate's worst-case expansion). One reallocation worst case, zero in steady state. |
| A4 | Boundary search hits adversarial inputs and probes O(N) candidates per chunk | `try_decode_at` is *only* called by `search_boundary_forward` over a bounded window per chunk (`chunk_size / 2`); search is O(window) not O(N). Window bound is asserted in unit tests. |
| A5 | The sequential `replace_markers` phase becomes Amdahl-bottleneck at high T | `replace_markers_avx2` runs at memory bandwidth (~12 GB/s). For inputs we care about it's a few ms total. **Bench `examples/replace_markers_throughput.rs`** (PR #95) measures it; if it ever becomes Amdahl-relevant we pipeline phase 1+2 per-chunk. Not needed today. |
| A6 | x86_64 per-thread throughput shortfall — measured on CI, not arm64 | Decompose hot path: bit refill, Huffman peek, match emit. The inner loop is auditable; PR #95 adds AVX2/BMI2 fast paths *only if* CI shows < 0.99× rapidgzip. Deferred-by-default to avoid premature complexity. |

### B. The marker decoder produces wrong bytes

| Tag | Failure | Structural mitigation |
|---|---|---|
| B1 | Marker emitted for back-ref that actually lands inside the chunk (false marker → wrong byte after resolve) | `emit_match` checks `dist > already_emitted_in_chunk` — only then does it emit `MARKER_BASE + (dist - already_emitted_in_chunk)`. Unit-tested with fixture distances spanning the boundary. |
| B2 | Chunk-local copy reads from chunk output that *contains* unresolved markers (and the copy doesn't recursively mark) | When emit_match copies inside the chunk, the bytes copied may be markers themselves; `replace_markers` resolves them in-place when phase 2 runs (markers are `u16` values, the copy preserves them). Fuzz test in `fast_marker_inflate.rs` covers this path with 200 random trials. |
| B3 | Cross-chunk back-ref distance > 32 KiB (impossible per RFC 1951, but who knows) silently produces garbage | `emit_match` asserts `1 ≤ dist ≤ 32_768` per RFC 1951; a violating stream produces an explicit error, never wrong output. |
| B4 | `replace_markers` u16→u8 conversion silently truncates leftover markers from phase-1-failed chunks | `u16_to_u8` fails fast with an explicit error if any `u16 ≥ MARKER_BASE` survives the resolve pass. Tested with synthetic input. |
| B5 | CRC32 verification skipped when expected CRC == 0 (Copilot caught this in PR #90 review) | `verify_and_write` always checks CRC and ISIZE. No `if expected_crc != 0` guard. Test: a fixture whose CRC32 truncates to zero is in `correctness.rs`. |
| B6 | Partial output written before CRC verification, so failed runs leave corrupt data in the writer | **Phase 3 buffers all output in `Vec<u8>` and verifies CRC32 + ISIZE *before* `writer.write_all`.** A failed verification returns `Err(...)` with the writer untouched. The `test_corruption_detected` test mutates the trailer and asserts no bytes were written. |
| B7 | False-positive boundary accepted by ISA-L's `try_decode_at`, rejected later by marker decoder (F6 above) | **Two-stage validation in `try_decode_at`**: ISA-L fast filter then a bounded strict re-validation by the marker decoder over the same first few KiB. If they disagree, the position is rejected and search advances. Cost: a small constant per *accepted* candidate, not per probe. |

### C. The marker pipeline silently doesn't run in production

This is the failure that killed F1, F2, F3 and almost killed F6 — every
prior attempt to ship a marker decoder. The mitigation isn't "remember to
test routing"; it's a structural counter.

| Tag | Failure | Structural mitigation |
|---|---|---|
| C1 | Routing gate tightens by accident; CLI falls through to sequential ISA-L; output is byte-perfect → all output tests still pass | **`MARKER_PIPELINE_RUNS: AtomicU64`** incremented at the *successful* tail of `decompress_parallel`. **`tests::routing::test_marker_pipeline_actually_runs_on_x86_64_isal`** snapshots the counter, invokes `decompress_single_member(T=4)` on the production routing test fixture, and asserts the counter moved. On `cfg(all(target_arch = "x86_64", feature = "isal-compression"))` only. |
| C2 | The killer test gets deleted as "redundant with output tests" | The counter and the test are documented in `RATIONALE.md` with explicit "do not delete" lines and a paragraph explaining the failure mode the test catches. The PR that introduces them ties them to the marker pipeline in the routing block of `CLAUDE.md`. |
| C3 | The marker decoder lands behind a feature flag for "safety", flag never gets enabled, code rots | **No feature flag.** The marker pipeline is the *only* parallel single-member path for x86_64 ISA-L + `T > 1` + `data > 10 MiB`. If it doesn't work the test goes red, not the path goes dormant. |
| C4 | "Cleanup" PR deletes a module in `src/decompress/parallel/` because it "looked dead" (the F3 pattern) | **`src/decompress/parallel/RATIONALE.md`** is the first thing a cleanup author sees in this directory. It lists every protected symbol and points back here. The CI test from C1 makes deletion of `MARKER_PIPELINE_RUNS` immediately visible. |
| C5 | Multi-PR rollout slips, last PR never lands, the new code lives uncalled (F3 setup) | **`docs/marker-decoder-plan.md`** is the milestone document. Each PR has a standalone success criterion. PR #94 is **the** wiring PR — until it lands, PR #93's code is unwired but its tests still run, so it never rots silently. |

### D. The work gets abandoned mid-rollout

| Tag | Failure | Structural mitigation |
|---|---|---|
| D1 | Long rollout loses momentum; some PRs land, others don't, we end up with half a marker pipeline + the v0.5.1 code still wired | Rollout split into the smallest possible PRs (#93 correctness only, #94 wiring + deletion-trap killer, #95 SIMD only if needed). Each is independently revertable. Each PR description names what *doesn't* ship in that PR. |
| D2 | Plan lives only in a Slack message or chat history; future agents can't see why a half-finished structure exists | **`docs/marker-decoder-plan.md`** checked into the repo with a decision log. Every "we tried X and it didn't work" gets a dated entry. |
| D3 | Throughput target on CI gets quietly lowered when it's hard to hit (F4 pattern that originally let v0.3.0 ship) | CI threshold is hardcoded in `scripts/check_guards.py` and `scripts/benchmark_single_member.py` at 0.99 / 1.0. Lowering them in a "make CI green" PR is visible in the diff and contradicts this document. |

### E. The win on CI doesn't translate to homelab

| Tag | Failure | Structural mitigation |
|---|---|---|
| E1 | 4-core CI numbers hide a serialization bottleneck that only appears at T=8/16/32 | `make ship` runs on `neurotic` (16 physical cores) and is authoritative. PR #95 only lands *after* `make ship` shows where to spend SIMD effort. |
| E2 | Optimization for x86_64 inner-loop regresses arm64 (zlib-ng fallback path) | The marker pipeline is gated on `isal_decompress::is_available()` which is x86_64-only. arm64 never takes the marker path. Regression test `test_arm64_falls_through_to_libdeflate` asserts this. |
| E3 | Apple Silicon Mac numbers diverge from x86_64 CI in either direction and we can't reproduce | Both architectures are in CI matrix (`Build macos arm64` / `Build linux x86_64` / `Build linux arm64`); divergent failure surfaces on push. |

---

## The decision log

Major design choices, with the reasoning that was current at decision time:

- **2026-05-12** — chose marker pipeline over v0.5.1 speculative-window:
  v0.5.1 ratio = 0.88× rapidgzip; target 0.99× requires ~1.1N work instead
  of 2N; only marker pipeline meets the work bound.
- **2026-05-12** — chose pure-Rust port of `inflate_consume_first` over an
  ISA-L-callback wrapper: ISA-L can't emit markers without C-level
  modification; we'd diverge from upstream. Pure Rust loses some perf per
  thread but recovers it via parallelism since total work is ~1.1N.
- **2026-05-12** — `u16` output justified empirically not theoretically:
  `examples/u16_output_cost.rs` measures the cost; ratio 1.03 on arm64
  retired the bandwidth concern.
- **2026-05-12** — SIMD inner loop deferred to PR #95: don't ship
  complexity before `make ship` says we need it.
- **2026-05-12** — deletion-trap killer counter is *not* behind a `cfg(test)`
  guard: incrementing one atomic on the successful tail of every parallel
  decode is sub-nanosecond and ships in production. Anything else lets the
  counter rot. (`Relaxed` ordering; the test reads it with `Relaxed`.)

---

## What "done" looks like

1. CI green on PR #90: `Single-Member Decomp Tmax (x86_64)` reports
   `gzippy ≥ 0.99× rapidgzip` and `gzippy ≥ 1.0× unpigz` on Silesia.
2. `tests::routing::test_marker_pipeline_actually_runs_on_x86_64_isal`
   passes on every x86_64 build.
3. `tests::routing::test_single_member_parallel_not_slower_than_sequential`
   passes on every build (sequential and parallel measured against each
   other on the same fixture; parallel must be < 1.3× sequential
   elapsed).
4. `make ship` on `neurotic` shows the marker pipeline scales linearly
   from T=2 up to physical-core count.
5. arm64 still routes to libdeflate one-shot for single-member — no
   regression.

When all five hold, the marker pipeline is the production path and the
deletion-trap defenses keep it there. If any of them slips, the failure
mode tag above tells you which mitigation is missing.
