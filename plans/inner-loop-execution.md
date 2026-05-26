# Inner-loop optimization: execution plan (v2, post-Opus-critique)

## Source decisions

This plan consolidates:

- `plans/inner-loop-architecture.md` — architecture comparison (7 candidates × 6 axes).
- Opus advisor architecture pick: **B'** (single impl, free function, minimal const generics, same crate, no runtime dispatch) + PGO + `#[inline(never)]` boundaries + chunk-boundary `#[target_feature]` dispatch.
- Opus reframe: the 1.54× per-byte gap is likely ~1.15-1.25× in inner-loop isolation; the rest is pipeline/marker-bootstrap overhead. Inner-loop isolation measurement is the load-bearing test BEFORE committing to inner-loop refactor work.
- Opus critique of v1 execution plan, applied throughout v2:
  - Phase 1 made a **STOP-AND-REPLAN gate**, not a "harness works" gate.
  - **PGO on current code** sequenced BEFORE extraction (new Phase 1.7).
  - **`perf record` on current production** added (new Phase 1.3).
  - σ/µ gate relaxed; `iai-callgrind` adopted as the determinism path.
  - Tier 2 `perf_event_open` instruction-count tests replaced with `iai-callgrind`.
  - L1d / branch-miss counters moved to manual `make perf-counters` (not unit tests).
  - Compile time / symbol count / alloc count categories added.
  - Asm-diff mechanism made concrete (script + committed baseline).
  - Target-cpu policy set (`x86-64-v3` for parallel SM).
  - `bench_baselines.json` at repo root, reviewer-checked.
  - Scalar reference variant kept permanently in `benches/inflate_variants/` as differential anchor.

## Prime directive (unchanged)

**Measurement before architecture.** 4 of 4 recent inner-loop optimizations regressed despite being structurally sound. The bottleneck is signal, not surface area. Build the diagnostic toolchain that can detect a 2% inner-loop win **with provable determinism** before changing any code in the loop itself.

## Phase 0 — Pre-flight policy decisions

These get committed to writing BEFORE Phase 1 starts. No litigating mid-execution.

### 0.1 Target-CPU policy

**Decision**: parallel SM path is built with `RUSTFLAGS="-Ctarget-cpu=x86-64-v3"` (Haswell+). Documented in `.cargo/config.toml` and CLAUDE.md.

**Reason**: PGO numbers are not reproducible across target-cpu values; vendor's `Block::read` runs with `-march=native` or equivalent on a Haswell+ machine. To compare apples to apples we must commit to a target-cpu floor.

### 0.2 Pivot criteria (pre-registered)

**Pre-registered before Phase 1 produces any numbers** so the team isn't emotionally invested when the data lands.

#### Pivot table

| Inner-loop-isolated gap (vendor vs gzippy per-block) | Action |
|---|---|
| ≥ 1.40× | Proceed with full Phase 2-5 plan; inner loop is the lever. |
| 1.20× – 1.40× | Proceed with Phase 1.7 PGO + Phase 4 DoubleLiteralCached only. Skip Phase 2 extraction unless PGO + variant port falls short. |
| 1.05× – 1.20× | **Kill Phases 2-5. Pivot to pipeline/marker-bootstrap work.** Inner loop is not the dominant cost. |
| ≤ 1.05× | Inner loop is at parity. The "1.54× gap" was entirely upstream. Reroute investigation to chunk pipeline, marker resolution, allocator behavior, or worker scheduling. |

#### Tie-breaking rule (mandatory, no debate)

Every Phase 1 gap measurement reports **`[p5, median, p95]`** over the corpus, not a single number. Bucket membership is determined by:

- **Upper boundary** (which bucket to move INTO from above): p95.
- **Lower boundary** (which bucket to move INTO from below): p5.
- **If p5 and p95 disagree on bucket**: re-run with the Calgary corpus (Phase 1.1). If Calgary's [p5, p95] also straddles the boundary, **default to the LOWER-investment bucket** (e.g., 1.22× median with [p5=1.18, p95=1.28] → default to 1.05–1.20× pivot-away, not 1.20–1.40× partial-engagement).

**Rationale**: lower-investment default avoids the sunk-cost trap; if the gap turns out to be larger after pivoting, a subsequent measurement can move us back up the table.

**Signed by**: <owner name>. Committed to this file. Bench numbers from Phase 1 trigger the matching action without further debate.

### 0.3 Backreferences tracking

**Decision**: `track_backreferences` is **moved out of the parallel-SM hot loop entirely**, before any refactor. Future seekable-index path will reintroduce it via a separate inner-loop variant if needed. This is a small PR that lands before Phase 2 starts.

### 0.4 Vendor-citation policy for novel SIMD and structural deviations

**Decision**: if a SIMD variant or structural element lands without a direct vendor counterpart, it requires a written design doc in `plans/inner-loop-design-decisions.md` justifying the deviation. The doc must enumerate (a) what vendor does, (b) why our approach is different, (c) what microarchitectural evidence supports the deviation. PR-time review enforces.

#### Pre-declared deviations requiring design-doc entries

These are KNOWN vendor deviations the plan introduces; design-doc entries land alongside their implementation:

- **Runtime CPU detection at chunk boundary via `#[target_feature]`**: vendor uses compile-time `#if defined(__AVX2__)` selection (builds N binaries). gzippy ships one binary per target tuple per Rust ecosystem convention; runtime detection at chunk boundary is the closest equivalent without N-binary shipping. Design doc entry required when Phase 3.2 lands.
- **iai-callgrind for instruction-count regression**: vendor doesn't have an analog. Design doc justifies adoption (cachegrind determinism vs `perf stat` variance).

### 0.7 `HuffmanCodingDoubleLiteralCached` integration shape (pre-declared)

**Decision** (made in Phase 0, not litigated in Phase 4): if Phase 4 ports `HuffmanCodingDoubleLiteralCached`, it does so as a **direct replacement** for `HuffmanCodingShortBitsMultiCached`. The inner-loop body is updated to match the new decode signature (single-symbol-plus-stash instead of packed multi-symbol). No `LitlenHuffman` trait abstraction. Vendor doesn't have one; we don't introduce one.

The `m_nextSymbol` "stash next" state moves into `WriterState` as an `Option<u16>` field (zero-overhead when unused per Rust niche optimization).

If a future case requires multiple Huffman variants coexisting, that requires a new Phase 0 design decision.

### 0.5 Bench baseline maintenance

**Decision**: `bench_baselines.json` lives at repo root. Schema:

```json
{
  "<phase_id>": {
    "rustc_version": "1.84.0",
    "target_cpu": "x86-64-v3",
    "pgo_state": "off|profiled|optimized",
    "pgo_profile_git_sha": "abc123",
    "machine_class": "neurotic-i7-13700T",
    "metrics": {
      "silesia_t9_best_ms": 420,
      "silesia_t9_p50_ms": 446,
      "block_corpus_mb_s_median": 224,
      "block_corpus_mb_s_p5": 198,
      "block_corpus_mb_s_p95": 251,
      "vendor_block_corpus_mb_s_median": 290,
      "vendor_block_corpus_mb_s_p5": 270,
      "vendor_block_corpus_mb_s_p95": 312,
      "per_block_cycles_iai": [12345, 12567, ...]
    },
    "git_sha": "abc123",
    "date": "2026-05-25"
  }
}
```

#### PGO staleness enforcement (mandatory)

`pgo_profile_git_sha` records the source SHA at PGO profile capture time. A `make pgo-staleness-check` rule (also run in CI) computes the current `git rev-parse HEAD:src/decompress/parallel/` (parallel-SM source tree hash) and fails if it doesn't match `pgo_profile_git_sha` from the most recent `pgo_state: "optimized"` baseline entry. Bench results recorded against a stale profile are explicitly invalid.

This converts PGO discipline from policy to enforcement.

#### Baseline-update protocol (mandatory)

A PR may update multiple baselines in one commit. The PR description MUST include a table:

```
| Baseline file | Old value | New value | Justification link |
|---|---|---|---|
| bench_baselines.json:phase_2 | 224 MB/s | 261 MB/s | plans/bench-logs/2026-06-01-phase2-result.md |
| tests/baselines/iai_inflate.json | 12500 | 11240 | (same) |
```

`justification_link` points to either a rebench log committed in `plans/bench-logs/<date>-<topic>.md` or a flamegraph in `plans/perf-snapshots/`. The default reviewer is the parallel-SM module owner. Single-line PR comments are NOT sufficient justification for baseline updates.

### 0.6 iai-callgrind SIMD-fidelity spike (mandatory, blocking Phase 1)

**Question**: does `iai-callgrind`'s cachegrind backend count SIMD instructions deterministically and at a granularity that distinguishes SIMD from scalar variants?

**Why this gates everything**: if cachegrind models SIMD as scalar (counts `vpackuswb` as one instruction the same as `mov` would be), then iai-callgrind's instruction count for a SIMD variant decreases relative to scalar (fewer ops total to do the same work) — which trips the ±2% baseline gate in Tier 1 6.1.2. The "perf gate fires on SIMD variant" failure mode collapses the unit-test discipline. If cachegrind doesn't count SIMD at all (treats it as opaque), the gate is meaningless.

**Spike**:

1. Write a trivial benchmark: a function that computes population count via `count_ones()` and one that uses `_popcnt64` intrinsic. Same input, same output.
2. Run both under iai-callgrind. Compare instruction counts.
3. Inspect via `cg_annotate` what cachegrind reports for the SIMD instruction.
4. Test the same with a wider example: scalar narrow loop vs `_mm256_packus_epi16` (the actual SIMD path we'd use in Phase 5).

**Possible outcomes**:

- **Cachegrind counts SIMD instructions correctly AND distinguishes throughput**: iai-callgrind is the right Tier 1 gate. Plan unchanged.
- **Cachegrind counts SIMD as scalar (same instruction-count delta)**: Tier 1 6.1.2 must use BOTH (a) `iai-callgrind` instruction count for correctness/regression detection on scalar AND (b) a CI-only wall-clock gate using `perf stat -e cycles` on a fixed-machine-class runner for SIMD variants. Plan adds new gate.
- **Cachegrind doesn't model SIMD usefully at all**: Tier 1 6.1.2 covers scalar variants only. SIMD variants are gated by Tier 3 manual `make perf-counters` on neurotic only. Plan demotes SIMD perf gating.

**Output**: a document in `plans/iai-callgrind-simd-spike.md` recording the decision and any test programs. Until this doc exists, Phase 1 does not start.

**Owner**: <name>. Time: 1 day.

## Phase 1 — Diagnostic infrastructure

NO production code changes in this phase. Only new diagnostic tools + tests.

### 1.1 Block-corpus extractor

`tools/extract_blocks/` — separate Rust binary.

Runs gzippy in instrumented mode (new feature flag `extract-corpus`) on a silesia decode. For each `CompressionType::DynamicHuffman` block, dumps:

- Compressed bitstream (raw bytes, bit-aligned start offset).
- Initial 32 KiB window (zeroed if first block of stream).
- Litlen + distance Huffman table code lengths.
- Expected decoded output (reference bytes, for correctness).
- Metadata: block index, decoded size, compressed-bit length, marker count, max-code-length per table (lets bench grouping show LUT-pressure correlation).

Output: `corpus/silesia_blocks/<idx>.bin` (one file per block). Target 30-50 blocks covering diverse Huffman shapes (sparse trees, deep codes, dense literals, length-heavy).

**Corpus is committed to the repo**, not regenerated. Determinism is non-negotiable.

A sibling `tools/extract_blocks_calgary/` extracts from the Calgary corpus too — second-corpus discipline for catching silesia-specific overfitting.

### 1.2 Microbenchmark harness

`benches/inflate_block.rs` (Criterion).

- Loads corpus once, allocates output buffers once.
- Iterates `decode_dynamic_huffman_block` (or current macro-host method via shim) over each block.
- Reports total MB/s + per-block MB/s.
- Records cycles/byte via `rdtsc` (Linux x86_64).

**Variance discipline**:

- Bench machine: pinned CPU, frequency-locked, turbo off, SMT siblings offline. Setup documented in `CLAUDE.md` as `make bench-prep`.
- ≥ 200 samples per bench.
- σ/µ gate: **≤ 1.5%** (relaxed from Opus critique; 0.5% was fantasy outside iai).
- Per-block table sorted by σ/µ; outlier blocks investigated separately.

### 1.3 `perf record` profile of current production

NEW (per Opus critique).

`tools/perf_profile_silesia.sh`:
- Builds gzippy at current HEAD.
- Runs `perf record -g -F 99 -- taskset -c 1 ./target/release/gzippy -d -c -p 1 /root/benchmark_data/silesia-gzip.tar.gz > /dev/null` on neurotic.
- Generates flamegraph via `flamegraph.pl` or similar.
- Extracts top-20 functions by self-cycles.
- Inner-loop cycle attribution: count samples in `litlen_hc.decode` (Huffman decode), the ring-write path, the back-ref-copy path, the bits refill path, the loop overhead.

**Output**: a one-page summary in `plans/perf-snapshot-pre-refactor.md` showing where the cycles ACTUALLY go.

This is the cheapest experiment that could redirect the entire effort. If Huffman decode dominates (>40% of inner-loop cycles), the answer is "port DoubleLiteralCached"; if ring-write dominates, "B' refactor + SIMD batch store"; if bit reader dominates, "specialize the bit reader templates per vendor."

### 1.4 Vendor C++ bench harness

`tools/vendor_inflate_bench/main.cpp` — single .cpp file.

- Includes vendor's `rapidgzip/gzip/deflate.hpp`.
- Reads the SAME corpus files (binary-compatible format).
- Calls vendor's `Block::read` directly.
- Prints JSON-line output: `{"block_idx": 0, "mb_per_sec": 350.2, "cycles_per_byte": 12.5}\n`.

`Makefile`:
- `make vendor-bench` — builds vendor library + harness, runs against corpus, prints output.
- `make compare-inflate` — runs gzippy bench + vendor bench, prints comparison table.

### 1.5 `iai-callgrind`-driven cycle counts (new in v2)

`benches/inflate_block_callgrind.rs` using `iai-callgrind` crate.

For each corpus block:
- Decodes via `decode_dynamic_huffman_block` (or current macro shim).
- Records: instructions executed, L1 hits/misses, LL hits/misses, RAM accesses, branches, branch mispredictions.

**Why iai-callgrind, not perf_event_open**:
- Cachegrind simulates the cache hierarchy → **deterministic** counts regardless of wall-clock noise.
- Works on macOS dev machines + Linux CI; no `perf_event_open` privileges required.
- Single source of truth for instruction-level perf; no per-machine-class baseline files.

Numbers committed to `bench_baselines.json` as `per_block_cycles_iai`.

### 1.6 Microarchitectural diagnostic harness (manual)

`benches/inflate_perf_counters.rs` — custom harness using `perf_event_open`.

Same corpus, measures: cycles, instructions, IPC, branch mispredicts, L1d misses, LLC misses, TLB misses.

NOT run as part of `cargo test`. Triggered manually via `make perf-counters` on the homelab bench host only. Used for attributing wall-time regressions to specific microarchitectural events when iai-callgrind numbers and wall-time diverge.

### 1.7 PGO on current code (NEW per Opus critique — was Phase 3 in v1)

**Run PGO against the CURRENT macro, before any extraction work.**

`make pgo-current`:
1. `cargo pgo build` — instrumented build with current code.
2. Run instrumented binary on silesia: `taskset -c 1,3,4,5,6,7,10,13,15 ./target/release/gzippy -d -c -p 9 silesia-gzip.tar.gz > /dev/null` to collect profile.
3. `cargo pgo optimize` — final optimized build.
4. Run microbench + silesia bench against the optimized build.

Numbers recorded in `bench_baselines.json` as `pgo_state: "optimized"`.

**Why PGO before extraction**:
- Near-zero correctness risk on current code.
- If PGO closes 30% of the gap on current macro, Phase 2's ceiling is lower; we know how much of the remaining work is codegen vs algorithm.
- If PGO closes <5%, the gap isn't switch-density. Reroute.

### 1.8 Phase 1 gate (STOP-AND-REPLAN, per Opus critique)

Phase 1 is **NOT** complete when the harnesses run. It's complete when the harness outputs trigger a documented decision.

**Required outputs before Phase 2 starts**:

1. Block-corpus exists, committed, reproducible.
2. gzippy microbench reports per-block MB/s with σ/µ ≤ 1.5%.
3. Vendor microbench reports per-block MB/s on same corpus.
4. **Inner-loop-isolated gap** computed: `vendor_per_block_mb_s / gzippy_per_block_mb_s` median across corpus.
5. iai-callgrind instruction counts captured for gzippy.
6. `perf record` snapshot generated; inner-loop cycle attribution table published.
7. PGO-on-current numbers recorded.
8. **Pivot decision applied per Phase 0.2 table**, signed off, recorded in this plan.

**No Phase 2 work begins until step 8 completes.**

## Phase 2 — Function extraction (B' refactor)

**Only executed if Phase 1.8 says inner-loop gap is ≥ 1.20×.**

### 2.1 `WriterState` struct

`src/decompress/parallel/inflate_inner.rs` (new module).

```rust
pub struct WriterState {
    pub ring_pos: u64,
    pub decoded_bytes: usize,
    pub distance_to_last_marker_byte: u64,
    pub at_end_of_block: bool,
}
```

(No `track_backreferences` — already removed per Phase 0.3.)

### 2.2 `decode_dynamic_huffman_block` free function

```rust
// Cite: vendor/rapidgzip/.../gzip/deflate.hpp:1589-1666
#[inline(never)]
pub fn decode_dynamic_huffman_block<W: WindowBuffer, const CONTAINS_MARKERS: bool>(
    state: &mut WriterState,
    ring: &mut [W::Element; RING_SIZE],
    bits: &mut Bits,
    litlen_hc: &HuffmanCodingShortBitsMultiCached,
    dist_hc: &HuffmanCodingShortBitsMultiCached,
    n_max_to_decode: usize,
) -> Result<usize, BlockError>
```

`WindowBuffer` trait: minimal, abstracts `u8` (post-window-known fast path) vs `u16` (marker bootstrap). Mirrors vendor's `Window` template axis.

Both `::<true>` and `::<false>` instantiations ship. The u8/post-known-window path is **production-reachable** via a future caller in `chunk_fetcher.rs` (when predecessor window is published before decode starts); we add the call site as part of this phase to avoid "no dead code on main" violation.

#### `#[inline]` policy (defined trigger)

- **During Phase 2**: function ships `#[inline(never)]`. Required for asm-diff to be meaningful (stable symbol).
- **End of Phase 3.1** (after PGO + LTO are wired): `#[inline(never)]` is removed in the same PR that lands PGO. Asm-diff baselines are regenerated as part of that PR (Phase 2.4 mechanism). PGO + LTO decides inlining; we don't second-guess.
- **From Phase 3.1 onward**: function ships with no inline directive. If a candidate variant in Phase 5 needs `#[inline(always)]` for a specific reason, that's stated in its design doc per Phase 0.4.

### 2.3 Shadow-mode verification

`#[cfg(feature = "shadow-decode")]` wrapper. Runs both old macro and new function; diffs ring bytes + ALL `WriterState` fields + bit-reader cursor position.

**Two-week / 5-investigation-hour cap.** Beyond that, shadow wrapper itself is suspect; delete it.

### 2.4 Asm-diff verification (NEW concrete mechanism per Opus critique)

`tools/asm_diff/`:
- Runs `objdump -d` (or `cargo asm`) on `decode_dynamic_huffman_block::<true>` and `::<false>`.
- Counts: `push %r*`, `mov %r*, -0x*(%rsp)` (spills), `call <...>` (non-eliminated calls).
- Compares counts to baseline in `tools/asm_diff/baselines/<rustc_version>/<target_cpu>.json`.
- Fail if spill count increases or new calls appear.

Auto-failing CI check. Baselines committed; regenerated by `make update-asm-baselines` (reviewer-checked diff).

### 2.5 Phase 2 gates

- All 33 routing tests pass with `shadow-decode` feature ON.
- Microbench MB/s within ±5% of Phase 1 baseline.
- iai-callgrind instruction count within ±2% of Phase 1 baseline.
- Silesia bench within ±3% of pre-refactor.
- Asm-diff shows no structural regression (zero new spills, zero new calls).
- Both `::<true>` and `::<false>` instantiations exist as distinct symbols in the binary (`nm` check).

## Phase 3 — Post-extraction optimization sweep

### 3.1 PGO on extracted function

`make pgo-rebuild` integrates `cargo-pgo` into the release flow. Each inner-loop edit triggers `make pgo-rebuild` before re-benching (PGO profile is stale after any inner-loop change; documented in CLAUDE.md).

### 3.2 `#[target_feature]` chunk-boundary dispatch

Per Opus: dispatch SIMD-vs-scalar variants at the chunk boundary, not the symbol boundary. The current `decode_chunk_isal` / `decode_chunk_marker_bootstrap_then_isal` callers in `chunk_fetcher.rs` get `#[target_feature(enable = "avx2,bmi2")]` wrappers that dispatch once per chunk. Inner loop is then free to use intrinsics without per-symbol dispatch overhead.

### 3.3 Phase 3 expected outcome

Opus prior: PGO closes 30-50% of the inner-loop gap (whatever that gap is per Phase 1 measurement).

Falsifiable: if PGO + chunk-boundary `#[target_feature]` together close <10%, the gap isn't codegen. Reroute per Phase 0.2 pivot criteria.

## Phase 4 — Huffman variant port

**Only executed if Phase 1.7's PGO + Phase 3's PGO-on-extracted close <80% of the inner-loop gap.**

Port `HuffmanCodingDoubleLiteralCached` from vendor as a **replacement** for `HuffmanCodingShortBitsMultiCached`. The new Huffman type becomes the `CompressionType::DynamicHuffman` table; the inner loop body updates to use its single-symbol-plus-stash decode signature.

If it wins on microbench AND silesia AND iai-callgrind: land. Old impl deleted.
If it loses on any gate: revert. New impl lives in `benches/inflate_variants/` for comparison.

## Phase 5 — Variant prototyping space

`benches/inflate_variants/`:

- `scalar.rs` — **permanent** scalar reference variant (per Opus critique). Survives even after a faster variant ships. Sole purpose: differential anchor for ALL future variants. CLAUDE.md "no dead code" applies to production only; bench code is exempt.
- `simd_avx2.rs`, `bmi2.rs`, `double_literal_v2.rs`, etc. — experimental variants.

Each variant:
- Has its own Criterion bench (`cargo bench --bench inflate_block -- variant_name`).
- Has its own iai-callgrind harness.
- Must pass `tests/inflate_variants_diff.rs` differential test against scalar reference.
- Must cite vendor `file:line` if vendor counterpart exists; OR add a design-doc entry per Phase 0.4.

When a variant wins production: it REPLACES the production impl. Loser stays in `benches/inflate_variants/` (per Opus, not deleted) as historical comparison anchor.

#### Variant graveyard cap (mandatory)

To prevent compounding maintenance burden:

- The scalar reference variant is **permanent** (sole differential anchor).
- All other variants in `benches/inflate_variants/` that have not been re-run in 6 months OR have broken under a rustc bump are deleted in a **quarterly cleanup PR**.
- The cleanup PR description lists deleted variants with brief explanation. Reviewer signs off.
- Variants deleted this way can be re-added later by re-porting from git history if needed.

CLAUDE.md "no dead code" exemption for `benches/` is asserted explicitly in CLAUDE.md (the exemption clause itself is a separate edit landing alongside this plan).

## Phase 6 — Unit-test perf gates

Per user's specific ask, plus Opus's reframe (use `iai-callgrind`, not `perf_event_open`).

### 6.1 Tier 1 — runs by default in `cargo test`

These run on every developer machine, every CI run, every commit. Fast (<200ms total), deterministic, machine-portable.

#### 6.1.1 Differential output

`tests/inflate_inner_differential.rs` — for each variant in `benches/inflate_variants/`, decode the canned corpus (5-10 blocks committed in `tests/fixtures/inflate_blocks/`), assert byte-equality with scalar reference.

Cost: ~50ms total. Mandatory.

#### 6.1.2 iai-callgrind instruction-count regression

`tests/inflate_inner_iai.rs` — uses `iai-callgrind` to count instructions for the canned-corpus decode. Asserts within ±2% of committed baseline.

Cost: ~500ms (one-time callgrind setup) per test invocation. **Deterministic**: same toolchain + same binary + same corpus → same instruction count (within callgrind's noise floor, which is ~0.01%).

**Per-variant baselines** (Phase 0.6 spike outcome (b) confirmed): SIMD variants reduce instruction count by 5-10× vs scalar (cachegrind sees the wider work as fewer iterations but doesn't model widened throughput). A single shared baseline would trip whenever a SIMD variant lands. Each variant ships with its own baseline:

- Scalar reference: `tests/baselines/iai_inflate_scalar.json`
- AVX2 variant: `tests/baselines/iai_inflate_simd_avx2.json`
- BMI2 variant: `tests/baselines/iai_inflate_simd_bmi2.json`
- etc.

The ±2% gate applies to the variant's OWN baseline, not the scalar baseline.

**rustc-bump exception path applies**: each baseline file embeds `rustc_version`. If `rustc --version` at test time differs, the test prints "rustc bump detected; regenerate baselines via `make update-baselines`" and is `#[ignore]`d for that run. Applies to scalar AND SIMD variant baselines.

**CI-only wall-clock supplement for SIMD variants**: `make perf-counters` on a fixed-machine-class CI runner (or neurotic) measures actual cycles via `perf stat -e cycles` on the corpus. Each SIMD variant has a pre-declared wall-clock floor in its design doc (Phase 0.4):

- **Narrow variant (AVX2 `_mm256_packus_epi16`)**: floor ≥ **2.0× scalar reference**. Citation: production code at `src/decompress/parallel/chunk_fetcher.rs:1665-1731` documents the AVX2 license-downclock concern; the floor must be high enough to make the SIMD path worth the downclock tax. Spike data shows cachegrind reports 7.2× fewer instructions and 2.9× fewer estimated cycles; real wall-clock typically 5-8× faster.
- **Future SIMD variants**: floor declared in `plans/inner-loop-design-decisions.md` BEFORE the bench runs, not after.

#### 6.1.3 Asm-pattern assertions

`tests/inflate_inner_asm.rs`:

- `decode_dynamic_huffman_block::<true>` and `::<false>` symbols exist (via `objdump -t` parse).
- Hot loop in `::<false>` contains no reference to `MARKER_BASE`-related code (Opus's flagged monomorphization-trap check).
- Hot loop in either instantiation contains no `call <core::panicking::*>`, `call <core::option::unwrap_failed>`, or other Rust-runtime handlers (proves no hidden bounds check / unwrap in hot path).
- Specific SIMD instruction families appear when a variant is active (e.g., `vpackuswb` if AVX2 narrow loop is shipped).

Cost: ~100ms (objdump parse once per test invocation).

Brittle across rustc versions; pinned via `rust-toolchain.toml`. Updated by `make update-asm-baselines`.

#### 6.1.4 Symbol-size regression

`tests/inflate_inner_size.rs` — assert `decode_dynamic_huffman_block::<true>` and `::<false>` symbol sizes (from `nm --size-sort`) are within ±20% of committed baselines.

Catches accidental inline-everything bloat.

Cost: ~10ms.

**rustc-bump exception path**: each baseline file embeds `rustc_version`. If `rustc --version` at test time differs from the baseline's recorded version, the test prints "rustc bump detected; regenerate baselines via `make update-baselines`" and is `#[ignore]`d for that run. The rustc-bump PR mandatorily runs `make update-baselines`; reviewer diff-checks the new symbol sizes. This is the only way the gate doesn't break every 6 weeks.

#### 6.1.5 Monomorphized symbol count

`tests/inflate_inner_mono_count.rs` — `nm | grep | wc -l` on all `decode_dynamic_huffman_block*` symbols. Assert ≤ committed limit (e.g., 4: u8/u16 × CONTAINS_MARKERS true/false).

Catches "I added a const generic that monomorphizes 32 times" bug.

Cost: ~10ms.

Same `rustc-bump exception path` as 6.1.4.

#### 6.1.6 Allocation-count assertion

`tests/inflate_inner_alloc.rs` — uses the `cap` allocator crate (`https://crates.io/crates/cap`) wrapped around `std::alloc::System`. The test thread's allocator is scoped via `cap::Cap::new(System, usize::MAX)`; after a baseline allocation count is captured pre-decode, the canned-corpus decode runs; post-decode count is compared. Asserts zero net allocations within `decode_dynamic_huffman_block`.

Catches the class of regression where someone introduces a Vec allocation in the inner loop. Cheap, deterministic, exactly the class of bug traces miss.

Why `cap` specifically: stable Rust, GlobalAlloc-based, deterministic across threads when scoped, and integrates with existing test allocators without nightly features. `dhat` is too heavy for unit-test-suite use (nightly-only, requires session setup). Custom `GlobalAlloc` wrappers are implementable three different ways and at least one will be wrong; standardize on `cap`.

Cost: ~50ms.

#### 6.1.7 Compile-time regression (CI-only)

`tests/inflate_inner_compile_time.rs` — parses `cargo build --timings` output for the `inflate_inner` module. Asserts compile time within 2× of baseline.

Catches generic blowup. Cheap (parses existing build output).

**CI-only via `#[cfg_attr(not(env_ci_runner), ignore)]`** — absolute compile times vary 3-5× between developer laptops and CI runners. Running this on developer machines produces false positives constantly; running only on CI (where the machine class is fixed) is the determinism path. Baseline is recorded per CI runner profile, refreshed via `make update-baselines` in CI.

Cost: ~0ms in CI (re-uses build output). N/A on dev machines.

### 6.2 Tier 2 — opt-in via `#[ignore]` + `--include-ignored`

Run in CI but not on developer machines by default. Slower, requires more environment control.

#### 6.2.1 Per-block iai-callgrind detailed counters

Same as Tier 1 6.1.2 but with per-block breakdown of L1/LL/RAM accesses, not just total instructions. Surfaces cache-footprint issues per input class.

Cost: ~2-5s. Run by `make test-perf`.

### 6.3 Tier 3 — manual, homelab-only

NOT in `cargo test` at all. Triggered manually by `make perf-counters` on neurotic.

- `perf_event_open`-based wall-clock + cycle measurement.
- Branch-mispredict counts.
- L1d / LLC miss counts.
- Vendor-side comparison (vendor C++ harness on same corpus).

These are research tools, not regression gates.

### 6.4 Baseline maintenance discipline

- `tests/baselines/` contains: `iai_inflate.json`, `asm_patterns.json`, `symbol_sizes.json`, `mono_counts.json`.
- All committed. Reviewer-checked diff on every change.
- Updated via `make update-baselines` (regenerates all). Manual review of the diff is mandatory; CI fails if baselines updated without justification in PR description.
- Tier 1 baselines targeted at `x86_64-unknown-linux-gnu` with `target-cpu=x86-64-v3` per Phase 0.1. Other platforms exempt (test passes if platform doesn't match).

## Cross-cutting policy (carry forward from CLAUDE.md)

- **Vendor parity**: every new pub fn carries `/// Cite: vendor/.../file.hpp:L-L`.
- **No dead production code**: but `benches/inflate_variants/` and `tests/` and `tools/` are exempt.
- **arm64 gated**: parallel SM is x86_64-only. `inflate_inner` module carries the same `cfg(all(target_arch = "x86_64", any(feature = "isal-compression", feature = "pure-rust-inflate")))` as the existing parallel-SM code.
- **Corpus committed**: `corpus/silesia_blocks/*.bin` and `corpus/calgary_blocks/*.bin` in repo.
- **Baselines committed**: `bench_baselines.json` + `tests/baselines/*.json`.

## What this plan inherits from prior work

- Hoist `pos % RING_SIZE` (commit 9c1e2e2).
- Lever G — Arc deep-clone elim (commit 4890e81).
- Lever H — pump prefetch during rx.recv (commit 5a9e51c).
- Kraft Huffman check + subchunk gate (deep-dive agent commits 43518fe, b529b3c).

Cumulative starting point: 666 ms → 420 ms best silesia wall (-37%). Vendor target: 147 ms (3.2× behind from here).

## What this plan does NOT do

- Workspace crate split (same crate per Opus).
- Runtime function-pointer dispatch (`#[target_feature]` at chunk boundary instead, per Opus).
- Multiple production impls simultaneously (single shipping impl; variants in `benches/`).
- C++ FFI link (standalone vendor harness instead).
- DSL / codegen (off-axis per Opus).
- arm64 inner-loop work (path is x86_64-only).
- `perf_event_open` in unit tests (replaced by `iai-callgrind`).

## Falsifiable success criteria

| Phase | Gate | Action if fail |
|---|---|---|
| 0 | All decisions signed & committed | Don't start Phase 1. |
| 1.1-1.6 | Harnesses report numbers; σ/µ ≤ 1.5% on microbench; iai-callgrind deterministic | Debug harness; don't proceed. |
| 1.7 (PGO-on-current) | PGO closes ≥ 5% of inner-loop gap OR doesn't (either is signal) | Record number; informs 1.8 decision. |
| **1.8 (pivot gate)** | Apply pivot table from Phase 0.2 to measured inner-loop gap | Document decision in this file; execute matching branch. |
| 2 (extraction) | Shadow green, microbench ±5%, silesia ±3%, asm-diff clean, both monos exist | Revert; pivot to in-place tuning or pipeline work. |
| 3 (PGO + target_feature) | ≥10% additional gap closed on microbench AND silesia | Reroute per 0.2 pivot. |
| 4 (DoubleLiteralCached) | ≥10% additional gap closed | Revert; investigate bit reader or pipeline. |
| 5 (each variant) | ≥10% microbench AND silesia AND all Tier 1 unit tests pass | Per-variant revert. |
| 6 (unit-test perf gates) | All Tier 1 tests green; baselines current | Block PR; require baseline update with justification. |

## Open questions

(All blocking questions resolved by Phase 0 decisions / spikes. Items below are non-blocking and addressed during execution.)

1. ~~iai-callgrind SIMD-fidelity~~ — **resolved by Phase 0.6 spike** (mandatory before Phase 1). Outcome determines whether Tier 1 6.1.2 is the sole perf gate or needs a CI-only wall-clock companion.

2. ~~`HuffmanCodingDoubleLiteralCached` integration shape~~ — **pre-declared in Phase 0.7**. Direct replacement, `m_nextSymbol` moves into `WriterState` as `Option<u16>`. No trait abstraction.

3. ~~Corpus update policy~~ — **decided**: never. The corpus is a fixed measurement substrate. Frozen at extraction time.

4. (Non-blocking) Phase 5's variant prototyping: should each variant land as its own crate-feature flag for build-time selection, or always live in `benches/`? Default: `benches/` only until at least 2 variants ship to production successively, then re-evaluate.

5. (Non-blocking) Calgary corpus: how many blocks? Default: 20-30 (smaller than silesia's 30-50; sole purpose is the boundary-tie-breaking in Phase 0.2). Settled during Phase 1.1.

6. (Non-blocking) `bench_baselines.json` schema versioning: when fields are added, do old baselines need migration? Default: append-only schema; readers tolerate missing fields. Settled during Phase 0.5 implementation.
