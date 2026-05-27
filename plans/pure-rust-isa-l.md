# Pure-Rust ISA-L for the parallel single-member path

**Worktree.** `/home/user/www/gzippy-reimplement-isal` on branch
`reimplement-isa-l`. All work lands here; the main checkout is the
reference for measurement diffs.

**Goal.** Delete every C-ABI inflate call from the
`parallel::single_member::decompress_parallel` hot path. After this plan
lands, the parallel-SM production path links to no ISA-L code; the
libdeflate audit confirms it isn't reachable from the parallel-SM
pipeline either. Correctness and performance are owned end-to-end by
pure-Rust modules under `src/decompress/`.

**Non-goal.** Removing ISA-L or libdeflate from the *non-parallel-SM*
backends (multi-member parallel via libdeflate FFI, single-member
sequential via ISA-L stream or libdeflate one-shot, ISA-L compressor for
L0–L3). Those are out of scope. This plan touches only what the
parallel-SM chunk pipeline actually calls.

**Anchor commit.** `2464f28` (handoff_reason + bytes_decoded
instrumentation). Citations and line numbers below are pinned to this
tree on `reimplement-isa-l`.

---

## 1. What's actually in the parallel-SM path today

Tracing the call graph from
`crate::decompress::parallel::single_member::decompress_parallel`:

| Concern | Module(s) | Backend today |
|---|---|---|
| Phase-1 marker decode (speculative bootstrap) | `parallel/deflate_block.rs`, `parallel/isal_huffman.rs`, `parallel/gzip_chunk.rs::bootstrap_with_deflate_block` | **Pure Rust** (already) |
| Phase-2 bulk inflate (clean window) | `parallel/inflate_wrapper.rs::IsalInflateWrapper` | Two cfg-gated impls: **C FFI** under `isal-compression` (`inflate_wrapper.rs:160-646`); **pure Rust** (`ResumableInflate2`) under `pure-rust-inflate` (`inflate_wrapper.rs:650-796`) |
| Block-boundary scan (slow path) | `parallel/block_finder.rs`, `parallel/raw_block_finder.rs` | **Pure Rust** (already) |
| Stopping-point semantics (`END_OF_BLOCK*`, `END_OF_STREAM*`) | `parallel/inflate_wrapper.rs` + patched ISA-L | C-side under `isal-compression`; pure Rust on `inflate/resumable.rs` |
| Routing eligibility for `IsalParallelSM` | `decompress/mod.rs:99-103` via `sm_cfg::PARALLEL_SM` | **Already correct** — gate is `cfg!(all(target_arch="x86_64", any(feature="isal-compression", feature="pure-rust-inflate")))` (`sm_cfg.rs:4-7`) |
| Output writing | `parallel/chunk_fetcher.rs` consumer drain → `BufWriter` | Pure Rust |
| CRC32 / ISIZE verify | `parallel/sm_driver.rs`, `parallel/crc32.rs` | Pure Rust |

### Key facts the draft of this plan got wrong (corrected by advisor)

- **`decompress/mod.rs:105` is NOT the routing gate for `IsalParallelSM`.**
  Line 105 (`isal_decompress::is_available()`) selects the *sequential*
  fallback `IsalSingle`. The parallel-SM gate is the
  `sm_cfg::PARALLEL_SM` check at `mod.rs:99-103`, which already covers
  `pure-rust-inflate` via the `any(feature = …)` branch in
  `sm_cfg.rs:4-7`. **Nothing in `mod.rs` needs to change to take Step 0;
  a single `cargo build --no-default-features --features pure-rust-inflate`
  flips the production routing.**
- **`benches/inflate_isal_vs_pure_rust.rs` is gated `not(feature =
  "pure-rust-inflate")` (`inflate_isal_vs_pure_rust.rs:9-13`).** Building
  with both features yields an empty bench. The bench compares the two
  backends *side by side under the C cfg gate* — `ResumableInflate2` is
  always reachable as a library type. Don't add `pure-rust-inflate` to
  this bench's `required-features` thinking it makes both backends live.
- **libdeflate is not called inside the parallel-SM hot loop today.**
  `git grep libdeflate src/decompress/parallel/` returns six hits, all
  in comments or doc-strings (`huffman_base.rs:138`, `deflate_block.rs:1004`,
  `inflate_wrapper.rs:951,1027`, `block_finder.rs:365`, `single_member.rs:21`).
  The "audit libdeflate" sub-step (Step 2.6) locks this in with a
  regression test; no live calls to remove.
- **`build.rs` does not reference ISA-L** — the static library is built
  by the `isal-sys` crate via the `[patch.crates-io]` entry in
  `Cargo.toml:159`. Deleting `vendor/isa-l/` requires removing the
  patch and the `isal-rs` dep, not editing `build.rs`.

### Bootstrap is already pure Rust — what does "replace bootstrap" mean?

`bootstrap_with_deflate_block` reads deflate blocks via
`deflate_block::Block::read_internal_compressed_specialized`, which uses
`IsalLitLenCode` / `IsalDistCode` from `parallel/isal_huffman.rs`. These
are *vendor-shape* Huffman LUTs ported into Rust, not FFI. So "replace
bootstrap" in the user brief does not mean rewriting a C dependency; it
means "finish the work so bootstrap and Phase-2 bulk live behind one
unified pure-Rust inflate driver, so the two-phase handoff plumbing
that exists *because* we had to cross a C-ABI boundary mid-chunk can
be revisited." That is Step 3 below, and it is **deferred and
conditional**, not part of the FFI deletion.

---

## 2. State of the existing pure-Rust replacement (`ResumableInflate2`)

`src/decompress/inflate/resumable.rs` already implements:

- The full `IsalInflateWrapper` surface used by `gzip_chunk.rs`:
  `new`, `with_until_bits`, `set_window`, `set_stopping_points`,
  `read_stream`, `tell_compressed`, `is_final_block`, `btype`,
  `reset_for_next_stream`, `remaining_input`, `advance_input`,
  `read_footer_at_current`, `session_pending`. Forwarded from
  `inflate_wrapper.rs:650-796` (the `pure-rust-inflate` branch).
- A 32 KiB `SlidingWindow` seeded by `set_window`.
- The four `StoppingPoint` flags with vendor semantics
  (`stopping_point.rs:13-18`).
- A `PendingMatch` resume primitive for mid-block yields when `output`
  fills.
- Unit tests (`inflate_wrapper.rs::tests`) that match flate2's output
  byte-for-byte across stopping-point combinations and at 128 KiB
  buffer granularity.

### Three semantic divergences from patched ISA-L (load-bearing — verify in Step 0)

1. **`session_pending()` always returns `false`** (`resumable.rs:376-378`).
   Three call sites in `gzip_chunk.rs:239, 273, 419` use it as part of
   the outer-loop continuation condition. The old ISA-L wrapper returned
   `true` whenever post-stop bytes were still buffered. Failure mode:
   chunks that previously needed one extra iteration to drain return
   short → byte-count mismatch downstream. Symptom: silent truncation
   that surfaces as a CRC32 mismatch on the final chunk's tail. No test
   forces this condition today.

2. **The no-progress termination guard at `resumable.rs:512-522`** (added
   per a prior Opus advisor review) silently breaks the inner loop when
   `read_stream` makes zero progress, no stop fires, and `bit_position`
   does not advance. Failure mode: on a chunk where `encoded_until_bits`
   lands mid-dynamic-block, the guard fires and the caller sees
   `finished=false, bytes_written < expected` with no error. This
   would surface as a CRC32 mismatch, not as `Err`.

3. **`read_footer_at_current` returns `InflateError::Internal(-1)`
   on insufficient bytes** (`inflate_wrapper.rs:749`). The old ISA-L
   wrapper returned a different errno. Probably non-load-bearing
   because `sm_driver.rs` parses the trailer from the outer envelope,
   but worth confirming: grep all callers and confirm no caller
   pattern-matches the specific error.

### Coverage status

Existing unit tests use flate2-encoded fixtures, not silesia /
silesia-large / software-gzip / logs-gzip. The
`inflate_isal_vs_pure_rust` bench can read silesia
(`benches/inflate_isal_vs_pure_rust.rs:30-35`) but only when built
with `isal-compression` (without `pure-rust-inflate`). The
`pure-rust-inflate` parallel-SM path has had no documented end-to-end
silesia run.

---

## 3. Plan

### Step 0 — bring up the pure-Rust path in production-shape

**Why first.** Every remaining step depends on this baseline. The single
biggest historical failure mode (`plans/rust-rapidgzip.md §3`) is acting
on a profile snapshot that doesn't reflect the current tree. Without a
working `pure-rust-inflate` parallel-SM bench number we are operating on
hope.

**Tasks.**

1. From the worktree:
   ```
   cargo build --release --no-default-features --features pure-rust-inflate
   ```
   No `isal-compression`. This produces the post-FFI-removal binary
   shape, ahead of the actual removal. If this build fails, the
   `pure-rust-inflate` cfg has rotted and Step 0a follows.

2. **Differential decode harness.** Write a single ignored integration
   test that:
   - Decodes silesia, silesia-large, silesia-gzip9, software-gzip, and
     logs-gzip fixtures via the pure-Rust parallel-SM path.
   - Compares each output bytewise against the reference output (the
     fixture's known sha256, OR a libdeflate one-shot decode of the
     same input — pick one and lock it).
   - Verifies the final CRC32 + ISIZE against the gzip trailer.
   Place at `src/tests/pure_rust_inflate_corpus.rs`, gate with
   `#[ignore]` so it runs only via `cargo test --release --features
   pure-rust-inflate --ignored corpus`. Runs in CI on the bench host.

   This harness is the only artifact that catches the
   `session_pending() → false` regression (§2 divergence 1), the
   no-progress-guard truncation (§2 divergence 2), and any other
   semantic-divergence-induced silent corruption.

3. Run `cargo test --release --no-default-features --features
   pure-rust-inflate` AND the new ignored corpus test locally
   (everything that does not need neurotic) plus
   `tests::routing::tests::test_single_member_routing_multithread`
   (which is in-process; no neurotic needed). Acceptance: green.

4. **Bench on neurotic.** Build the `pure-rust-inflate`-only binary and
   run `make bench-sm` from the worktree. Note `make bench-sm` pushes
   the current branch to origin before invoking the homelab
   (`Makefile:275`), so the branch must be pushable. Capture:
   - gzippy MB/s under `pure-rust-inflate`-only
   - rapidgzip MB/s (same neurotic window)
   - ratio

5. **Wrapper-only A/B.** Run `cargo bench --release --features
   isal-compression --bench inflate_isal_vs_pure_rust` from a separate
   build (the bench cfg-gate excludes `pure-rust-inflate`). This isolates
   the wrapper MB/s delta from the end-to-end delta. Tier-1 gate target
   ≤ 1.5× (per the bench's own comment block); record the actual value.

**Acceptance gate.** Numbers from Steps 0.4 and 0.5 pasted into a new
§7 "Numbers as of Step 0" subsection. No further work proceeds without
this baseline.

**Step 0a — fix any failing tests.** If Step 0.2 / 0.3 fail on real
corpus, that becomes the first actionable subtask. Likely suspects in
order of probability:

- `session_pending() → false` interaction with `gzip_chunk.rs:239` outer
  loop on chunks where ISA-L would have buffered post-stop bytes.
- No-progress guard masking a legitimate need-more-input case on a
  chunk that crosses `encoded_until_bits` mid-block.
- `pending_stream_header_stop` handling across `reset_for_next_stream`
  boundaries (parallel-SM doesn't actually call `reset_for_next_stream`
  in its hot loop — single member only — but verify).

Test each hypothesis by reducing to a minimal failing fixture *before*
guessing.

**Estimated cost.** Half a day if `ResumableInflate2` is correct on
silesia. One to three days if any §2 divergence has bite.

### Step 1 — close any throughput gap that blocks production use

**Gate.** Step 0 reveals one of three states:

- **Green** (gzippy/rapidgzip ratio under `pure-rust-inflate` ≥ 0.55×
  AND within 10 % of today's `isal-compression` ratio of 0.62×): skip
  to Step 2.
- **Yellow** (ratio drops 10–25 %): diagnose with `perf record`.
  **Note:** the `worker.isal_stream_inflate` SpanGuard at
  `gzip_chunk.rs:193-199` fires in *both* backends (it wraps
  `decode_chunk_isal_impl`, not the wrapper internals), so the
  per-call timeline still reads correctly. But: any *inner* ISA-L span
  inside the C wrapper is gone — perf is the only inner-loop diagnostic.
- **Red** (> 25 % regression): stop. Run one Opus pre-flight
  cross-check on the root cause before spending more than four hours
  on a fix. Reference `plans/rust-rapidgzip.md §3` for advisor-use
  patterns.

**Constraint.** No "innovation" optimizations. Per
`feedback_no_innovation`, every change must have a vendor file:line
counterpart in `vendor/rapidgzip/librapidarchive/`, `vendor/isa-l/`, or
libdeflate (allowed as a third reference for techniques rapidgzip
absorbed). If a candidate optimization has no vendor counterpart, do
not implement it on this plan.

Likely fix candidates (each requires vendor citation BEFORE
implementation, falsifiable bench AFTER):

- FASTLOOP multi-literal lookahead inlining (`resumable.rs` —
  verify `MULTI_LITERAL_HITS` counter is non-zero on silesia).
- `copy_match_windowed` cross-window splice cost (when the match's
  source crosses the `output[0]` boundary into the sliding window).

**Falsification.** Each landed fix: 20-trial A/B/A on neurotic; ≥ 30 ms
wall improvement on silesia-large T=16 ships; < 10 ms reverts;
in-between gets one more measurement.

### Step 2 — cut the ISA-L FFI link, in one PR

Structural and (mostly) irreversible. Sequenced last because once the
feature flags collapse there is no quick rollback.

1. **Audit `arena-allocator` consumers first** —
   `git grep 'feature = "arena-allocator"' src/`. Today the activation
   chain runs from `isal-compression → arena-allocator` AND from
   `pure-rust-inflate → arena-allocator` (`Cargo.toml:50-51`). Removing
   `isal-compression` won't break the chain. But: any code that today
   relies on *both* features being on simultaneously will break. Confirm
   none does.

2. **`Cargo.toml` deletions.**
   - Remove `isal-rs` optional dep.
   - Remove `[features].isal` and `[features].isal-compression`.
   - Remove `[patch.crates-io].isal-sys = { path = "vendor/isal-rs/isal-sys" }`.
   - `arena-allocator` stays — it's now activated solely by
     `pure-rust-inflate`.

3. **`src/decompress/parallel/inflate_wrapper.rs`.**
   - Delete the C-FFI gated branch (`#[cfg(all(feature =
     "isal-compression", not(feature = "pure-rust-inflate"), target_arch
     = "x86_64"))]`) — the `IsalInflateWrapper` impl at
     `inflate_wrapper.rs:160-646` and the non-x86_64 stub
     `inflate_wrapper.rs:799-822` if the pure-Rust backend supports
     non-x86_64 (verify: `ResumableInflate2`'s hot path uses no x86
     intrinsics, but the SpanGuard call sites in `gzip_chunk.rs` still
     `#[cfg(target_arch = "x86_64")]`-gate via `sm_cfg::PARALLEL_SM`).
   - Promote the `pure-rust-inflate + x86_64` branch to the only
     `IsalInflateWrapper`. Keep its `#[cfg(target_arch = "x86_64")]`
     gate — explicit comment that arm64 enablement is a separate plan.
   - Do NOT mechanically delete by line number — let the compiler
     guide the deletion. The "lines 160-646" range cited in earlier
     drafts is approximate; the actual deletion has shared types and
     stubs interleaved.

4. **Cfg sweep across `parallel/*.rs`.** Replace
   `#[cfg(any(feature = "isal-compression", feature =
   "pure-rust-inflate"))]` with a simple `#[cfg(feature =
   "pure-rust-inflate")]` (or remove the cfg if the path no longer
   needs feature-gating). `sm_cfg.rs:6` becomes
   `feature = "pure-rust-inflate"` only.

5. **Delete from worktree.**
   - `src/backends/isal_decompress.rs` — only if `cargo build
     --workspace --no-default-features --features pure-rust-inflate`
     succeeds without it. Verify by deleting and rebuilding.
   - `src/backends/isal.rs` (probe wrapper) — same check.
   - `src/backends/isal_compress.rs` — IN SCOPE because the C library
     is being removed, but the ISA-L *compression* path used by L0–L3
     becomes unavailable. **Trade-off question for the user:** keep
     ISA-L compression (means we still vendor / link to the C library
     and the FFI link removal is incomplete) OR drop ISA-L compression
     (means L0–L3 falls back to libdeflate one-shot / flate2; small
     compression-perf regression). Default in this plan: **keep ISA-L
     compression for now** (out-of-scope per §non-goals) — but that
     means `vendor/isa-l/` and `vendor/isal-rs/` STAY. We can only
     delete the parallel-SM-decode FFI, not the C library wholesale.
     **Revise the user brief or expand scope before deleting `vendor/`.**
   - If the user opts to drop ISA-L compression: delete `vendor/isa-l/`,
     `vendor/isal-rs/`, `packaging/isal-patches/`, and
     `src/backends/isal_compress.rs` in this same commit.

6. **`decompress/mod.rs` routing.** No change needed for parallel-SM
   eligibility (`sm_cfg::PARALLEL_SM` already correct).
   `DecodePath::IsalSingle` still depends on `is_available()` at
   line 105 — that's the *non*-parallel-SM fallback and stays as long
   as ISA-L compression stays. Do NOT rename `IsalParallelSM →
   ParallelSM`; cosmetic, violates "speed not fidelity."

7. **libdeflate audit + regression lock.** Add a compile-time test
   in `src/tests/routing.rs` (or a new `src/tests/no_libdeflate_in_sm.rs`)
   that fails the build if any module under `src/decompress/parallel/`
   imports anything from `libdeflater` / `libdeflate_sys`. Implementation:
   a `build.rs` sub-pass OR a `#[test]` that greps the binary's symbol
   table for `libdeflate_` symbols reachable from the
   `parallel::single_member::*` namespace — pick whichever is simpler.
   This converts a stated user intent into a falsifiable artifact.

8. Single commit: `refactor: delete ISA-L FFI from parallel-SM
   (pure-Rust inflate only)`.

**Acceptance gates (all four must be green to land):**

1. `cargo test --release --workspace --no-default-features --features
   pure-rust-inflate` green.
2. New corpus test from Step 0.2 green (corpus correctness on real
   fixtures, not just unit-test fixtures).
3. `make bench-sm` on neurotic shows post-removal wall **within ±5 %
   of the Step-0.4 baseline** (a regression here means the FFI deletion
   accidentally took a fast path; bisect). A pass here is weak
   evidence — Step 0 already ran the pure-Rust path; the deletion
   should be a no-op for wall.
4. **Behavioral gate:** run `nm target/release/gzippy | grep -i isal`
   in the post-deletion build and confirm zero ISA-L symbols
   remain in the parallel-SM-reachable code (allow them only if ISA-L
   *compression* stayed). Without this gate, gates 1-3 trivially pass
   even if the FFI is still linked.

**Estimated cost.** One to two days. Most of the time is in the cfg
sweep, the corpus-test build-out, and the user-decision on ISA-L
compression scope.

### Step 3 (deferred / conditional) — collapse the two-phase decode

Now that bootstrap and Phase-2 both live in pure Rust, the two-phase
handoff at `gzip_chunk.rs::decode_chunk_marker_bootstrap_then_isal`
exists only because crossing a C-ABI boundary mid-chunk was expensive.
With both phases in Rust, the option to unify becomes available — but
the current shape works and matches vendor.

**Defer this.** Revisit only if post-Step-2 profile shows the
bootstrap/Phase-2 boundary is no longer load-bearing (handoff_reason
histogram shows handoffs are rare under the new code). Until then, two
phases stay.

---

## 4. What we will NOT do

- **Build a new inflate from scratch.** `ResumableInflate2` exists with
  passing unit tests; the work is to make it production-grade, not to
  rewrite it.
- **Port ISA-L's hand-tuned x86 asm.** Out of scope per `CLAUDE.md` and
  `project_bootstrap_perf_diag.md`.
- **Touch the libdeflate-using code paths.** Multi-member parallel,
  sequential SM, compression L1–L5 one-shot probe all stay on
  libdeflate.
- **Daemon-mode amortization.** Orthogonal; out of scope.
- **Rename `IsalParallelSM` → `ParallelSM`.** Cosmetic; violates "speed
  not fidelity."
- **Patch `decompress/mod.rs:105`.** Wrong line — that gates the
  sequential ISA-L fallback, not the parallel path.
- **Add `pure-rust-inflate` to the `inflate_isal_vs_pure_rust` bench's
  `required-features`.** The bench cfg-excludes it deliberately.
- **Delete `vendor/isa-l/` without first deciding whether ISA-L
  compression stays.** That decision changes scope and must be made
  explicitly.
- **Innovation optimizations in Step 1** without a vendor counterpart
  citation. `feedback_no_innovation`.
- **Lower the handoff threshold below 32 KiB** — covered by
  `pure-rust-phase1-speedup.md §3.B`, not this plan.

---

## 5. Risk register

1. **`session_pending() → false` silently truncates chunks.** Three
   `gzip_chunk.rs` outer-loop conditions changed semantics. Symptom:
   final-chunk CRC32 mismatch. Artifact: the Step 0.2 differential
   corpus test catches this on real silesia. **Mitigation: gate Step 1
   on Step 0.2 green.**

2. **`resumable.rs:512-522` no-progress guard masks truncation.** On a
   chunk where `encoded_until_bits` lands mid-block the guard returns
   without error. Symptom: short byte count downstream; surfaces as
   CRC32 mismatch. Artifact: same corpus test, with the
   silesia-large-T=16 fixture specifically exercising
   `encoded_until_bits`-mid-block boundaries. **Mitigation: add a
   targeted unit test for this exact condition before promoting to
   prod.**

3. **`read_footer_at_current` ErrorKind mismatch.** Probably
   non-load-bearing (the trailer is parsed by `sm_driver.rs` from the
   outer envelope). Symptom: any caller pattern-matching the specific
   errno gets a different error path. **Mitigation: grep the callers,
   one paragraph in Step 0.**

4. **Heavy-tail chunk throughput regresses.** 6/42 silesia-large chunks
   have 200+ ms Phase-1 (`pure-rust-phase1-speedup.md`). If
   `ResumableInflate2`'s clean-decode is even 10 % slower per-byte
   than ISA-L, Phase-2 on those same chunks regresses by the same
   factor and wall regresses with them. **Mitigation: Step 1's
   throughput gate — FFI removal in Step 2 only after end-to-end ratio
   is within 10 % of the `isal-compression` build.**

5. **CI matrix shrinks invisibly.**
   `.github/workflows/ci.yml`, `benchmarks.yml`, and `release.yml`
   reference `isal-compression` and/or `pure-rust-inflate`. Removing
   `isal-compression` shrinks the matrix; some latent coverage
   disappears. **Mitigation: audit the three workflow files in Step
   2, compensate with one additional matrix entry if removing the
   feature shrinks coverage of a target OS / arch / rust toolchain.**

6. **Scope ambiguity on ISA-L compression.** The user brief says
   "replace the bootstrap + libdeflate + ISA-L FFI bindings for
   specifically the parallel single-member decoding production path."
   That language scopes to decode, not compression. But
   `src/backends/isal_compress.rs` shares the `vendor/isa-l/` C
   library with the decode FFI. **We cannot delete `vendor/isa-l/`
   without dropping ISA-L compression too. Step 2.5 explicitly flags
   this as a user decision before any `vendor/` deletion.**

7. **arena-allocator consumers may rely on both features being on
   simultaneously.** Today's chain is
   `{isal-compression, pure-rust-inflate} → arena-allocator`. After
   Step 2 only `pure-rust-inflate` activates it; if any caller silently
   depends on the dual activation we'll see a build failure (good —
   loud and immediate). **Mitigation: Step 2.1 grep, done before the
   deletion.**

---

## 6. Done-when criteria

User's controlling goal (set 2026-05-27): *fully implement a performant
ISA-L in pure Rust that matches ISA-L but without FFI bindings. We're
not done until an Opus advisor considers our work thus far and agrees
we've taken the right path and tried everything.*

Two tiers. **Tier 1 (FFI removal)** is the structural prerequisite;
**Tier 2 (perf parity + advisor sign-off)** is what closes the project.

### Tier 1 — FFI removal (structural)

1. `Cargo.toml` contains no `isal-rs` dependency and no
   `isal-compression` / `isal` feature. `[patch.crates-io].isal-sys`
   removed.
2. `git grep -E '\bisal' src/decompress/parallel/` returns only doc
   comments / archaeology references; no live calls into ISA-L's C
   library.
3. `git grep libdeflate src/decompress/parallel/` returns only doc
   comments. Step 2.7's regression lock keeps it that way.
4. `nm target/release/gzippy | grep -i 'isal_inflate'` returns no
   symbols (unless ISA-L compression intentionally stayed — see
   Risk #6 / Step 2.5).
5. `cargo build --workspace --no-default-features --features
   pure-rust-inflate` succeeds.
6. Corpus differential test (Step 0.2 →
   `src/tests/pure_rust_inflate_corpus.rs`) green on neurotic
   for all five fixtures.
7. One `make ship` green on the merge commit.

### Tier 2 — performance parity + advisor sign-off (controlling)

8. `make bench-sm` on neurotic reports gzippy `pure-rust-inflate`-only
   throughput **within 10 % of the `isal-compression` build's
   throughput** on the same neurotic snapshot. Numerically: if today's
   `isal-compression` build is at ratio `R_isal` vs rapidgzip, the
   pure-Rust build must reach `R_pure ≥ 0.9 × R_isal`. This is the
   "performant ISA-L" criterion — bytes-per-second within 10 % of the
   FFI baseline on the production corpus.
9. `inflate_isal_vs_pure_rust` Tier-1 gate (`bench` says ≤ 1.5× —
   tighten to **≤ 1.15× when both backends run on the same neurotic
   snapshot**). Wrapper-level isolation of the per-byte gap.
10. Heavy-tail chunks: the 6/42 silesia-large pacemakers
    (`pure-rust-phase1-speedup.md`) must NOT regress on `pure-rust-inflate`
    relative to the `isal-compression` build. Tracked via the existing
    `worker.bootstrap` + `worker.isal_stream_inflate` p95/max
    timeline spans.
11. **Final gate — Opus advisor sign-off.** Spawn a fresh Opus advisor
    with the post-Tier-1+2 worktree pinned. Prompt it to argue against
    the claim "we've reached perf parity and tried every reasonable
    lever inside the no-innovation constraint." Sign-off requires the
    advisor to either (a) concede the claim, or (b) point to a
    specific remaining lever we then either land or formally falsify
    before re-running the gate. Mirror the worktree-cd guardrail used
    in this plan's own critique pass.

---

## 7. Numbers as of Step 0

Empty; fill on first bench. Expected schema:

```
date:                       YYYY-MM-DD HH:MM (neurotic snapshot)
gzippy pure-rust-only MB/s: <number> (5-run mean, drop warmup)
rapidgzip MB/s:             <number>
ratio:                      <number>x
wrapper-A/B (ISA-L/Rust):   <number>x   (Tier-1 gate ≤ 1.5x)
corpus differential:        <pass|fail>  (silesia | silesia-large |
                            silesia-gzip9 | software-gzip | logs-gzip)
```

---

## 8. Open questions to resolve during execution

- **Scope question (blocks Step 2.5).** Does ISA-L compression stay or
  go? If go, `vendor/isa-l/` + `vendor/isal-rs/` + `packaging/isal-patches/`
  + `src/backends/isal_compress.rs` all delete; if stay, the C library
  remains vendored but no FFI in parallel-SM. **Default: stay; revise
  if user disagrees.**
- **Step 0 result.** What is the neurotic `pure-rust-inflate`-only
  parallel-SM ratio today? Drives Step 1 branch (Green/Yellow/Red).
- **Step 1 (if Yellow/Red).** Is the gap in dynamic-Huffman table
  build, FASTLOOP multi-literal inlining, or cross-window match copy?
  `perf record` answers; do not guess.
- **Step 3 trigger.** Does the post-Step-2 handoff_reason histogram
  show enough handoffs to justify keeping the two-phase split?
