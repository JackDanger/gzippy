# Tight-loop optimization harness — plan (v2, post-Opus-critique)

## Goal

Encapsulate gzippy's marker-bootstrap inner loop so we can iterate on its
performance with **vendor-level intensity** (codegen inspection, SIMD, BMI2,
hand-tuning, vendor Huffman variants) **without risking semantic regressions**
in the parallel-SM pipeline, with a **dedicated benchmark harness** that
compares gzippy vs rapidgzip on the same pre-extracted compressed corpus.

Target: close the 1.5× per-byte gap between gzippy's `run_multi_cached_loop`
(224 MB/s) and vendor's `deflate::Block::read` (345 MB/s).

## Critical design decisions (post-critique)

1. **No `OutputSink` trait.** Vendor (`deflate.hpp:1589-1666`) parameterizes
   only on `Window::value_type` (u8 vs u16, i.e. `containsMarkerBytes`).
   A trait per-symbol call boundary would shove a function-shape into the
   middle of a loop whose entire point is to amortize multi-symbol cache
   hits via `symbol >>= 8` peeling. Use a free function with const generic
   `CONTAINS_MARKERS: bool` instead. The state that the macro currently
   mutates on `self` moves into a `WriterState` struct passed `&mut`.

2. **Bench BEFORE refactor.** Build the microbench against the current
   macro through a shim FIRST. Establish "224 MB/s ± noise" baseline.
   Then refactor. Any drop in microbench MB/s post-refactor is a regression
   the silesia ±3% wall gate is too loose to catch.

3. **No vendor FFI.** Standalone C++ harness in `vendor-bench/main.cpp`
   that includes vendor headers and prints MB/s. Driven by `make
   vendor-bench`. No symbol resolution, no namespace mangling, no recurring
   submodule breakage.

4. **Shadow-mode wrapper.** Feature-gated `cfg(feature = "shadow-decode")`
   wrapper runs both old macro and new function on every chunk, diffs ALL
   state + output bytes, panics on divergence. Run in CI for one week
   post-merge, then delete. This catches subtle macro→function divergences
   the byte-output check alone would miss (e.g., `distance_to_last_marker_byte`
   off-by-one that only matters cross-chunk on specific inputs).

5. **Drop `LinearKnownWindowSink`.** No production caller — gzippy's worker
   doesn't know its upstream window until `apply_window`. Adding it would
   violate CLAUDE.md "no dead code on main." The 1.5× per-byte gap is in
   the marker case anyway; non-marker is a side-quest.

6. **Profile FIRST, refactor LATER.** Before any structural change, run
   `perf record` on a T=1 silesia decode through this path. Confirm that
   write/copy is ≥40% of inner-loop cycles. If Huffman decode (LUT probe +
   canonical fallback) dominates, pivot directly to porting
   `HuffmanCodingDoubleLiteralCached` and this harness is premature.

## Current state (what we have to extract from)

The hot loop is `macro_rules! run_multi_cached_loop!` at
`src/decompress/parallel/deflate_block.rs:1282-1380`, inside method
`Block::read_internal_compressed_canonical_specialized`. It mutates:

- `self.output_ring: [u16; RING_SIZE]`
- `self.ring_pos`
- `self.decoded_bytes`
- `self.distance_to_last_marker_byte`
- `self.at_end_of_block`
- `self.track_backreferences`, `self.backreferences: Vec<_>` (if tracking on)

And uses `CONTAINS_MARKERS` as a const generic on the surrounding method.

The vendor reference is `deflate.hpp:1589-1666` (the inner loop) +
`deflate.hpp:1673-1700+` (the `WITH_DEFLATE_SPECIFIC_HUFFMAN_DECODER`
specialized path that the next optimization target,
`HuffmanCodingDoubleLiteralCached`, hooks into — it's NOT a sink swap, it's
a different `CompressionType::DynamicHuffman` match arm).

## Why the current shape is bad for optimization

1. **Macro-embedded** — duplicated per call site; cannot `cargo asm` a
   single named symbol; cannot swap implementations.
2. **Mutates `self`** — `&mut self` aliasing prevents LLVM from keeping
   `ring_pos` in a register across early-return paths; every `commit!`
   spills. Vendor's `Window` is passed by reference too but everything
   inlines in a single TU.
3. **Output couples with decode** — interleaved literal-write and back-ref
   resolution. Cannot isolate decode-rate for microbenchmark timing.
4. **No reference impl to diff against** — a new optimized impl has no
   property test target; the only correctness signal is a full silesia
   round-trip.
5. **No microbenchmark** — perf changes go through silesia (15-trial,
   ±5% on quiet system, ±25% under load).

## The plan, in dependency order

### Step 0 — Profile to confirm the lever (REQUIRED BEFORE STARTING)

Run `perf record -g -F 9999 -- taskset -c 1 ./target/release/gzippy -d -c -p 1
/root/benchmark_data/silesia-gzip.tar.gz > /dev/null` on neurotic. Generate
flamegraph. **Gate**: write+copy+ring-bookkeeping is ≥ 40% of inner-loop
samples. If Huffman decode (LUT probe + `decode_long` fallback) dominates,
**scrap this plan** and port `HuffmanCodingDoubleLiteralCached` directly.

Half a day; could redirect the entire refactor.

### Step 1 — Build the microbenchmark FIRST (against the current macro)

`benches/decode_block.rs` using Criterion. Bench the current code by
wrapping the macro in a small `pub fn decode_block_current(...)` shim that
builds a throwaway `Block` instance from a prepared input, runs through
the existing code path, returns decoded byte count.

Corpus: generated by `make bench-corpus`, NOT committed to repo (gitignore
`benches/fixtures/*.bin`). The script dumps raw deflate blocks from a real
silesia decode using existing decode infrastructure, plus their pre-parsed
litlen + distance Huffman tables.

**Two corpora**:
- `silesia_blocks_inner_loop.bin`: pre-parsed Huffman trees + raw deflate
  payload. Bench measures pure inner-loop time.
- `silesia_blocks_with_header.bin`: raw header bits + payload. Bench
  measures realistic per-block cost including Huffman-tree construction.

The inner-loop corpus is the primary target for porting
`HuffmanCodingDoubleLiteralCached`; the with-header corpus catches
regression in table-build cost.

**Note**: corpus is read into memory ONCE before measurement; `black_box`
on per-block result; iCache pollution across blocks is realistic
(matches production). Document that vendor-side comparison MUST use the
same corpus order or comparison is invalid.

Establish baseline: **"current gzippy at X MB/s ± Y on the inner-loop
corpus."** Without a fixed baseline, post-refactor microbench can't
detect a 5-10% regression.

Half to one day.

### Step 2 — Standalone vendor C++ bench harness

`vendor-bench/main.cpp` — one file, includes vendor's
`rapidgzip/gzip/deflate.hpp`, reads the same corpus binary, calls
`Block::readInternalCompressedMultiCached<VectorView<uint16_t>>` (or the
appropriate vendor entry point) on each block, times with
`std::chrono::steady_clock`, prints `MB/s\n` to stdout. Driven by
`make vendor-bench`.

No FFI, no Rust-side build dependency on vendor, no symbol resolution
across compile boundaries. The comparison is a shell script that runs both
benches on the same corpus and prints both numbers.

Half a day.

### Step 3 — Define `WriterState` struct (NOT a trait)

```rust
// src/decompress/parallel/decode_state.rs
pub struct WriterState {
    pub ring_pos: u64,
    pub decoded_bytes: usize,
    pub distance_to_last_marker_byte: u64,
    pub at_end_of_block: bool,
    /// Pass `Some(&mut vec)` when tracking back-references for the seekable
    /// path; `None` otherwise. Mirrors the current `track_backreferences`
    /// bool but moves it from `self` to an explicit parameter.
    /// Cite: rapidgzip ChunkData::backreferences (ChunkData.hpp:?).
    pub backrefs: Option<&'static mut Vec<Backreference>>, // TODO lifetime
}
```

(Exact lifetimes will be tightened during impl — placeholder `'static`
above is illustrative.)

No trait. Just a struct + free generic function below. Cite:
deflate.hpp:1589-1666.

### Step 4 — Extract `decode_dynamic_huffman_block` as a free function

```rust
// src/decompress/parallel/decode_dynamic.rs
//
// Cite: vendor/rapidgzip/.../gzip/deflate.hpp:1589-1666 (Block::read inner loop)
pub fn decode_dynamic_huffman_block<const CONTAINS_MARKERS: bool>(
    state: &mut WriterState,
    ring: &mut [u16; RING_SIZE],
    bits: &mut Bits,
    litlen_hc: &HuffmanCodingShortBitsMultiCached,
    dist_hc: &HuffmanCodingShortBitsMultiCached,  // or vendor's exact type
    n_max_to_decode: usize,
) -> Result<usize /*emitted*/, BlockError>
```

The inner `symbol_count` peeling stays in-function (`while symbol_count > 0
{ ... symbol >>= 8 }`). The vendor's structure is preserved 1:1; only the
state-passing changes from `&mut self` to explicit `&mut WriterState +
&mut [u16; RING_SIZE]`.

`commit!()` becomes a `return Ok(emitted)` or `return Err(...)` pattern;
`WriterState` is mutated through the lifetime of the call.

The const-generic instantiation produces two named symbols (one per value
of `CONTAINS_MARKERS`), each `cargo asm`-able directly. That's the asm
visibility unlock — no trait needed.

The macro becomes a 5-line wrapper:
```rust
let mut state = WriterState { ring_pos: self.ring_pos, ... };
let emitted = decode_dynamic_huffman_block::<CONTAINS_MARKERS>(
    &mut state, &mut self.output_ring, bits, litlen_hc, dist_hc, n_max_to_decode)?;
self.ring_pos = state.ring_pos; self.decoded_bytes = state.decoded_bytes; ...
```

One to two days (realistic; the optimistic "1 day" in v1 ignored
`backreferences` tracking + `at_end_of_block` semantics + the
`commit!()` early-return pattern interacting with multiple state writes).

### Step 5 — Shadow-mode wrapper (feature-gated)

```rust
#[cfg(feature = "shadow-decode")]
fn run_multi_cached_loop_shadow(...) {
    // Snapshot self state, run new function, diff.
    let pre = (self.ring_pos, self.decoded_bytes, self.distance_to_last_marker_byte,
               self.at_end_of_block, self.output_ring.clone());
    let result_old = run_old_macro(...);  // original macro path
    let post_old = (self.ring_pos, ...);
    // restore pre, run new
    self.ring_pos = pre.0; ...
    let result_new = decode_dynamic_huffman_block(...);
    let post_new = (self.ring_pos, ...);
    assert_eq!(post_old, post_new, "shadow divergence at block <ctx>");
    assert_eq!(result_old, result_new);
}
```

Run in CI for one week post-merge with `cargo test --features shadow-decode`
across all 33 routing tests. Then delete. Catches subtle divergences the
byte-output check alone misses.

### Step 6 — Routing test gate + microbench gate

**33 routing tests** must pass with `shadow-decode` feature ON.
**Microbench**: post-refactor MB/s must equal pre-refactor MB/s within
±5%. (Stricter than silesia ±3% wall gate, because microbench is less
noisy and the goal is "zero regression in the inner loop.")

Silesia bench: must NOT regress > 3% on best-of-15.

### Step 7 — Differential property test suite

`tests/decode_block_fuzz.rs`:

- (a) **libdeflate-generated random blocks** — happy path coverage:
  compress random bytes with libdeflate, decode with each impl, assert
  byte-equality.
- (b) **~30 hand-built adversarial fixtures** — known boundary classes:
  max-length codes (15-bit Huffman), sparse tables, single-symbol
  alphabets (must be rejected), length=258, distance=32768, EOB-immediate,
  length=3 with distance=1 RLE, truncated/malformed input. These are what
  catch SIMD-port bugs.
- (c) **Real silesia corpus replay** — take blocks from the bench corpus,
  assert byte-equality across all impls.

Reference impl: the extracted scalar `decode_dynamic_huffman_block`. Once,
on a fixed seed corpus committed as a small test fixture, the scalar's
output is cross-checked against libdeflate. After that the scalar is the
reference for all other impls.

### Step 8 — Asm inspection target

After step 4, `cargo asm gzippy::decompress::parallel::decode_dynamic::decode_dynamic_huffman_block::<true>`
produces the marker-mode inner-loop assembly directly. Compare line-by-line
with `objdump -d` on vendor's `Block::read` after the standalone C++ bench
build. Document the diff in `plans/inner-loop-asm-diff.md` for ongoing
reference.

### Step 9 — Iterate on optimizations

Each candidate impl (DoubleLiteralCached port, SIMD batch decode, BMI2
pext, larger-LUT variants) lives as either:

- A new free function in `decode_dynamic.rs` (for full inner-loop rewrites)
- A new Huffman-coding type in `src/decompress/parallel/huffman_*` + a new
  `CompressionType::DynamicHuffmanXxx` match arm (for table-shape changes
  like `DoubleLiteralCached`).

Each must:
- Pass all property tests (a/b/c above).
- Have a vendor file:line citation in its doc comment (CLAUDE.md "port
  don't innovate").
- Improve microbench MB/s by ≥ 10% AND silesia wall by ≥ 3% to land.

**Production picks the fastest impl that passes runtime CPU-feature
checks** (BMI2 / AVX2 / etc.). Default impl remains the scalar reference
for arm64 / fallback.

### Step 10 — CI integration

- `cargo test --features shadow-decode -- routing` runs in CI for one
  week after the refactor lands. Then drop the feature.
- `cargo bench --bench decode_block` runs in CI as **a correctness check
  only** (does it compile? does the differential test pass?). MB/s is not
  gated on shared CI runners due to 10-20% noise.
- `make bench-loop` runs the bench locally on neurotic and stores a
  per-host baseline in `benches/baseline.<hostname>.json`. Manual gate
  before merging perf changes.

## Constraints (carry forward from CLAUDE.md)

- **Vendor parity required**: every new pub fn carries a `/// Cite:
  vendor/.../file.hpp:L-L` doc comment. Add a lint or PR-checklist item.
- **No dead code on main**: `LinearKnownWindowSink` and any other
  abstraction without a production caller is OUT.
- **arm64 gated**: parallel SM is x86_64-only. New `decode_dynamic` module
  carries the same `cfg(all(target_arch = "x86_64", ...))` as the existing
  parallel-SM code.
- **Corpus binaries not committed**: `benches/fixtures/*.bin` in
  `.gitignore`. Generated by `make bench-corpus` from a real silesia
  decode.

## What we DON'T change

- No semantic change to the parallel-SM pipeline.
- `apply_window` post-processing unchanged.
- Routing tests run unchanged.
- The `commit!()` early-return semantics preserved (function returns
  `Result<usize, BlockError>`; caller writes WriterState fields back).
- Marker emission semantics identical (function with `CONTAINS_MARKERS=true`
  mirrors current macro behavior bit-for-bit).
- arm64 build unaffected (gated behind same cfg as existing path).

## Estimated effort

| Step | Days | Risk |
|---|---:|---|
| 0 — profile + gate decision | 0.5 | low |
| 1 — microbench against current code | 1 | low-medium |
| 2 — vendor C++ bench | 0.5 | medium (vendor build) |
| 3 — WriterState struct | 0.25 | low |
| 4 — extract free function | 1-2 | medium (state semantics) |
| 5 — shadow wrapper | 0.5 | low |
| 6 — gating | 0.25 | low |
| 7 — differential test suite | 1 | medium (adversarial fixtures) |
| 8 — asm doc | 0.25 | low |
| **Total** before first optimization | **5-6 days** | |
| 9 — each candidate impl | 1-3 | per-impl |

## Falsifiable success criteria

- **Step 0 gate**: perf record shows write/copy ≥ 40% of inner-loop cycles.
  Else: pivot to direct `HuffmanCodingDoubleLiteralCached` port.
- **Step 1 baseline**: microbench reports 224 MB/s ± 5 MB/s (matches the
  decode-rate finding from prior trace work).
- **Step 2 vendor baseline**: vendor microbench reports 345 MB/s ± 20
  MB/s.
- **Step 6 post-refactor**: microbench MB/s within ±5% of step 1 baseline;
  silesia wall within ±3% of pre-refactor.
- **Step 7 differential**: 100% pass on the 30 adversarial fixtures, plus
  libdeflate cross-check on 1000+ random blocks.
- **Step 9 first optimization**: ≥ 10% microbench MB/s AND ≥ 3% silesia
  wall, both directions, AND all property tests pass.

## Resolved open questions (from v1)

- ~~Q1: `OutputSink` boundary?~~ **Resolved**: drop the trait; use a free
  function + `WriterState` + const generic.
- ~~Q2: `CONTAINS_MARKERS` const generic?~~ **Resolved**: keep as
  const generic on the free function; that's the only polymorphism vendor
  uses.
- ~~Q3: Corpus size?~~ **Resolved**: two corpora; size set by "10s of MB,
  large enough to dominate measurement overhead, small enough to fit in
  L3 between iterations." Generated, not committed.
- ~~Q4: Vendor FFI?~~ **Resolved**: standalone C++ bench, no FFI.
- ~~Q5: Less invasive starting point?~~ **Resolved**: yes — extract
  function only, drop trait, drop sink concept. That's THIS plan.

## Open questions remaining

1. What's the right `Backreference` tracking shape — passing
   `Option<&mut Vec<Backreference>>` adds a per-iteration branch. Vendor
   doesn't track backreferences inline (seekable-index path is a separate
   pass). Should we feature-gate `track_backreferences` out of the hot
   loop entirely?
2. Where does `DoubleLiteralCached`'s `m_nextSymbol` "stash next" trick
   fit? It's stateful between calls — `WriterState` would need to carry
   it, OR the function returns up to 2 symbols per call and the caller
   peels.
3. Should we ship `bench-corpus` generation as a one-shot test fixture
   (commit ~5 KB of canned blocks) for CI determinism, AND a larger
   on-demand generator (10s of MB, local only) for serious benchmarking?
