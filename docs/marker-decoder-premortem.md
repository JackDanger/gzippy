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

### F7. The "BTYPE=01-heavy region returns None" failure  (acknowledged shipping limitation, PR #90)

Some compressed regions of real-world input (observed on Silesia, plus a
synthetic adversarial fixture in `routing.rs::make_btype01_heavy_data`)
contain mostly fixed-Huffman blocks (BTYPE=01). RFC 1951 §3.2.6 makes
the fixed-Huffman code maximally compact — every 9-bit code is used and
the block has *no header redundancy* to validate. BlockFinder excludes
BTYPE=01 candidates by design, and tier-2's byte-aligned brute force
misses non-byte-aligned positions.

Three attempts to add cheap BTYPE=01 boundary detection failed:

1. **Emit BTYPE=01 candidates from BlockFinder** (PR #90 commit 86d35bc).
   Candidate count exploded (~25% of bit positions pass the 3-bit
   header check); `test_find_blocks_parallel_matches_sequential` slowed
   from 0.5 s to 200 s due to O(N²) `.contains()` over millions of
   candidates. Reverted.
2. **Tier-3 bit-by-bit sweep with `validate_boundary(min_blocks=2)`**
   (PR #90 commit b898a11). Local repro produced 62 MB output for 25 MB
   expected — false-positive boundaries caused chunks to *double-cover*
   the same byte range. Reverted.
3. **Tier-3 with strict `validate_boundary(min_blocks=4, min_output=32KiB)`**
   (PR #90 commit 86d35bc). Same double-coverage on the BTYPE=01-heavy
   fixture. Reverted.

Opus advisor review (PR #90 thread) named the structural issue: cheap
per-position validation cannot reject all BTYPE=01 false positives
because fixed Huffman has no header redundancy. Tightening
`validate_boundary` thresholds is a probabilistic game that can't be
won — random fixed-Huffman data routinely decodes 4+ blocks of
valid-looking symbols by chance.

Rapidgzip's source acknowledges the same limitation
(`vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`):
*"when the deflate block finder failed to find any valid block inside the
partition, e.g., because it only contains fixed Huffman blocks."*
Their fix is **chain-decoding from a confirmed offset**, which serializes
that one chunk but preserves correctness.

**Mitigations shipped** (PR #90 originally, refined in PR #96; full
table in section G below):

- **G1** (refined PR #96): `decode_chunk_markers_bounded` exits cleanly
  at the first real block boundary at or past `end_bit_limit`. PR #90's
  exact-match contract was too strict; the speculative-then-correct
  pattern handles misaligned limits via phase 1c.
- **G2** (refined PR #96): `decompress_parallel` no longer rejects on
  mismatch — instead it *corrects* `start_bits[N+1]` to
  `chunks[N].end_bit_offset` (which IS a real boundary by G1
  invariant). Re-decodes chunk N+1 from the corrected start. The
  induction (chunk 0 starts at a real boundary; decode lands at a real
  boundary; correction propagates) makes the pipeline self-healing.
- **G3** (PR #90): Routing-layer typed fallback. `ParallelError::TooSmall`
  falls through silently (intended). Every other error increments
  `MARKER_PIPELINE_BOUNDARY_MISSED` and prints `[gzippy] parallel
  single-member fell back to sequential: {e}` to stderr unconditionally.
  In debug builds, panics.
- **G4** (PR #90): Bench script reads the routing-trace stderr signal
  and hard-fails CI with a specific message — "marker pipeline did not
  run end-to-end" — instead of the ambiguous "0.62× rapidgzip".
- **G5** (PR #96): Cross-chunk consistency correction sweep in
  `phase1c_resolve_consistency`. Single forward pass through pairs
  (N, N+1); when chunks[N].end_bit != start_bits[N+1], correct and
  re-decode chunk N+1. Each chunk re-decodes at most once. Wall-time
  deadline (2 s) bounds adversarial cases.

The cross-chunk consistency correction is **the structural fix** for
F7. PR #90's top-K + strictness-ramp design (proposed in an earlier
revision of this document) was probabilistic and didn't converge on
BTYPE=01-heavy inputs — Opus advisor review identified that as the
wrong layer to defend. The correction sweep is induction-based: every
decode that starts at a real boundary lands at a real boundary, so
chunk N's end_bit is always trustworthy. PR #96 is the rewrite.

The test `test_marker_pipeline_runs_on_btype01_heavy_input` is now
un-ignored (was `#[ignore]`'d in PR #90 pending this PR).

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

### G. Boundary-detection consistency (added 2026-05-13 after Opus advisor review)

The defenses for F7: per-position validators can't catch all
BTYPE=01 false positives, so we add structural defenses one layer
up — at the contract between phase 1a (boundary picks) and phase 1b
(chunk decode) — so wrong picks fail loudly instead of producing
wrong output.

| Tag | Failure | Structural mitigation |
|---|---|---|
| G1 (refined in PR #96) | Original concern: chunk N silently overshoots misaligned `end_bit_limit` into N+1's territory. PR #90 used an exact-match `==` contract that returned Err on `>`. **PR #96 reverts to `bit_pos >= limit ⇒ Ok(actual_end)`** — the actual decoded end may exceed the speculative limit and is the new ground truth (real boundary by induction). G5 (below) uses it to *correct* misaligned starts instead of rejecting. |
| G2 (refined in PR #96) | Phase 1a returns speculative boundary picks. BTYPE=01 false positives slip past `validate_boundary`. | PR #90's design: `decompress_parallel` verifies `chunks[N].end_bit == start_bits[N+1]` and returns Err on mismatch. PR #96's design: still verifies, but on mismatch **corrects** `start_bits[N+1] = chunks[N].end_bit` and re-decodes chunk N+1 (see G5). G2 is now a passive check inside G5's loop, not a separate rejection path. |
| G3 | Routing's `Err(_) ⇒ fall back silently` hides "marker pipeline tried and failed" inside the same path as "input too small for parallel." Production looks identical to libdeflate; CI sees a generic perf shortfall. | Typed routing fallback: `ParallelError::TooSmall` is the only error that falls through silently. Every other variant increments `MARKER_PIPELINE_BOUNDARY_MISSED` and prints `[gzippy] parallel single-member fell back to sequential: {e}` to stderr unconditionally. `debug_assert!` panics in debug builds. (PR #90 commit 751b450.) |
| G4 | Bench reports "gzippy 0.62× rapidgzip" without distinguishing "ran slow" from "never ran." | Bench script (`scripts/benchmark_single_member.py`) captures gzippy stderr with `GZIPPY_DEBUG=1` and parses for the G3 routing-trace message. Fails CI with a specific actionable reason: "marker pipeline did not run end-to-end on this fixture (silent fallback to sequential libdeflate). Throughput numbers reflect libdeflate, not the parallel path." (PR #90 commit c0f4f6d, extended in 751b450.) |
| G5 (PR #96) | Cross-chunk consistency correction. `phase1c_resolve_consistency` walks pairs (N, N+1) once forward. When `chunks[N].end_bit != start_bits[N+1]`, the latter was a false positive (BTYPE=01 most often). Correct `start_bits[N+1] = chunks[N].end_bit` (which is a real boundary by G1 invariant) and re-decode chunk N+1. Propagate forward. Each chunk re-decodes at most once. Bounded by 2 s wall-time deadline. `MARKER_PIPELINE_RETRY_ITERATIONS` counts corrections — `>0` on BTYPE=01-heavy inputs, 0 on healthy data. |

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
- **2026-05-13** — accepted BTYPE=01-heavy regions fall back to
  sequential libdeflate (F7). Three attempts to add a cheap BTYPE=01
  validator all failed for the same structural reason: fixed Huffman
  has no header redundancy. Opus advisor review (PR #90 thread)
  recommended moving the defense from per-position validation to
  cross-chunk consistency. Shipped G1–G4 in PR #90; the cross-chunk
  retry loop (Opus approach #2) is the next PR. Rapidgzip's source
  acknowledges the same limitation
  (`vendor/rapidgzip/.../GzipChunkFetcher.hpp`) and chain-decodes in
  that case — validation that we're not missing an obvious cheap
  approach.
- **2026-05-13** — implemented G5 (cross-chunk consistency correction).
  First attempt (top-K candidates per chunk with strictness-ramped
  `validate_boundary`) was overcomplicated and didn't converge on
  BTYPE=01 fixtures; reverted. Second design (forward correction
  sweep, induction from chunk 0's real start) is **strictly simpler**
  and structurally sound: chunk N's decoded end_bit is always a real
  boundary, so correcting chunk N+1's start to chunks[N].end_bit makes
  it a real boundary by construction. Each chunk re-decodes at most
  once. Reverted G1 from PR #90's exact-match `==` to `>=` (the
  original speculative contract) since G5 now does the correction work
  G1's strict contract was trying to surface. PR #96 lands the change.

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
