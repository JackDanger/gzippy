# Inner-loop architecture comparison

## Premise

Implementation cost is not a factor. Choose the architecture that maximizes
the long-term ceiling for optimizing gzippy's marker-bootstrap inner loop
toward vendor parity (currently 224 MB/s gzippy vs 345 MB/s rapidgzip,
1.54× per-byte gap on silesia). The architecture must support intense,
iterative optimization (SIMD, BMI2, multiple Huffman variants, hand-tuned
asm if needed) with full correctness safety and high-fidelity measurement.

## What we know about the loop

- Located at `src/decompress/parallel/deflate_block.rs:1282-1380` inside
  `macro_rules! run_multi_cached_loop!`.
- Hot operation: per-symbol decode (Huffman LUT probe) → write to ring
  buffer (u16, marker-emitting) → optional back-reference copy.
- Symbol decode returns up to 3 packed symbols per cache hit; loop peels
  with `symbol >>= 8`.
- Mutates `self.{ring_pos, decoded_bytes, distance_to_last_marker_byte,
  at_end_of_block, output_ring, track_backreferences, backreferences}`.
- Const generic `CONTAINS_MARKERS` parameterizes marker emission.
- Vendor counterpart: `vendor/rapidgzip/.../gzip/deflate.hpp:1589-1666`
  (`Block::read` inner loop), with `vendor/.../gzip/deflate.hpp:1673+`
  (the `WITH_DEFLATE_SPECIFIC_HUFFMAN_DECODER` specialized path that
  uses `HuffmanCodingDoubleLiteralCached`).

## Comparison axes

Each architecture is weighted across six axes, +/0/− qualitative:

1. **Headroom** — how far can the architecture take us toward vendor parity? Does it impose ceilings? Does it enable hand-asm / SIMD / BMI2 cleanly?
2. **Vendor parity fit** — does porting vendor patterns (DoubleLiteralCached, Window-templated decode, etc.) feel natural or contorted?
3. **Correctness safety** — how easy to introduce subtle bugs? How tractable is catching them (property tests, shadow mode, etc.)?
4. **Iteration speed (post-setup)** — once the architecture exists, how quickly can a new optimization candidate be tried, measured, and landed?
5. **Code quality / maintainability** — abstractions, boundaries, clarity of dataflow, long-term legibility.
6. **Optionality** — can we add new directions (e.g., new Huffman variant, runtime CPU dispatch, alternate bit-readers, separate seekable path) without throwing away work?

## Candidate architectures

### A. Status quo (in-place macro edits)

- Inner loop stays as macro at `deflate_block.rs:1282-1380`.
- Optimizations edit the macro directly.
- Measurement: silesia round-trip + 33 routing tests.
- Single impl at a time (variants live on branches, not in main).

| Axis | Score | Note |
|---|---|---|
| Headroom | − | Single-impl ceiling. SIMD attempted inside the macro must coexist with scalar fallback as separate code paths interleaved at runtime; no clean swap. |
| Vendor parity fit | 0 | Mirrors what the macro already does; new Huffman variants become new match arms. |
| Correctness safety | − | Macro edits route only through silesia. Subtle state bugs not caught until full pipeline diverges. |
| Iteration speed | − | Edit → 30-minute silesia bench (noisy) → revert if regress. Slow loop. |
| Code quality | − | Macro is opaque; everything inlined; no separation of decode/output. |
| Optionality | − | Cannot have two impls coexisting in production; runtime CPU dispatch impossible without restructure. |

**Verdict**: works, but is the floor. Acceptable for one-off patches; not for the planned intense optimization sweep.

### B. Free function + `WriterState` struct + const generic

- Extract loop as `decode_dynamic_huffman_block<const CONTAINS_MARKERS: bool>(state, ring, bits, litlen_hc, dist_hc, n_max) -> Result<usize, BlockError>`.
- `WriterState` struct holds the formerly-self state.
- Macro becomes a 5-line wrapper.
- Same crate, no module split.

| Axis | Score | Note |
|---|---|---|
| Headroom | + | Function-level boundary lets LLVM optimize without `&mut self` aliasing constraints. Asm inspection on the symbol works. |
| Vendor parity fit | + | Mirrors vendor's `Block::read` signature shape (state + buffer + reader); const generic mirrors `Window::value_type` polymorphism. |
| Correctness safety | 0 | State surface explicit (good); but no architectural support for multiple impls running side-by-side, so shadow-mode is bespoke. |
| Iteration speed | 0 | Microbench possible; differential property test possible. But each new impl is a fork of the function — only one active per build. |
| Code quality | + | Clear function boundary; explicit state; readable. |
| Optionality | 0 | Adding multiple impls means multiple functions and a dispatch site; doable but not pre-designed for it. |

**Verdict**: clean, faithful, but optimizes for ONE impl evolution at a time. Doesn't pre-build the structure for parallel-impl optimization.

### C. Separate `gzippy-inflate` workspace crate

- Inner loop moves to its own workspace member: `crates/gzippy-inflate/`.
- Public API: `pub fn decode_dynamic_huffman_block(...)` (or set of variants).
- Own `Cargo.toml`, `lib.rs`, `tests/`, `benches/`, `fixtures/`.
- gzippy main crate adds it as a path dependency.
- Multiple impls coexist in sibling modules: `scalar.rs`, `double_literal.rs`, `simd_avx2.rs`, `bmi2.rs`, etc.
- Crate-level differential test suite + Criterion bench.
- Optionally publishable as a standalone crate.

| Axis | Score | Note |
|---|---|---|
| Headroom | ++ | Maximum isolation lets each impl be tuned without touching the integration pipeline; can use crate-level `#![feature(...)]` for nightly Rust experiments, isolated `RUSTFLAGS`, etc. |
| Vendor parity fit | + | Vendor's huffman/ and gzip/ headers map naturally to module boundaries inside the crate. Easy to add new vendor-ported Huffman types as new modules. |
| Correctness safety | ++ | Crate-private invariants enforced by API surface. Property tests and differential tests at crate level catch bugs before they reach gzippy main. |
| Iteration speed | ++ | `cargo bench -p gzippy-inflate` doesn't rebuild gzippy main. `cargo test -p gzippy-inflate` is seconds. |
| Code quality | ++ | Strongest possible boundary in Rust. Each module is self-contained. Public surface is small. |
| Optionality | ++ | Multiple impls live as sibling modules with no entanglement. Runtime CPU dispatch is natural. The crate can also be used by other consumers (e.g., a future async-API gzippy). |

**Verdict**: highest ceiling. Maximum isolation = maximum tuning freedom. Cross-crate inlining is mitigated by `#[inline]`+LTO; with `crate-type = ["rlib"]` + workspace LTO, no measurable difference from same-crate.

### D. Vendor C++ FFI (use vendor's inner loop directly)

- Link rapidgzip vendor C++ statically; expose `Block::read` via `cxx` or `bindgen`.
- gzippy's pure-Rust pipeline calls vendor's C++ inner loop for the hot path.

| Axis | Score | Note |
|---|---|---|
| Headroom | + | By construction equals vendor's performance. |
| Vendor parity fit | + | Maximum: literally vendor code. |
| Correctness safety | + | Vendor is well-tested upstream. |
| Iteration speed | − | Optimization requires editing vendor C++ or carrying patches; cross-language dev loop. |
| Code quality | − | Hybrid Rust+C++ codebase. Build system complexity. Binary distribution complications. |
| Optionality | − | Cannot pursue pure-Rust SIMD experiments; cannot specialize for Rust patterns. |

**Verdict**: defeats the project's identity (pure-Rust gzip). Carries CMake/cross-platform pain. Off-axis for the team's goals even ignoring cost.

### E. Multiple parallel impls behind runtime CPU dispatch (in-crate, no workspace split)

- All impls live as siblings in `src/decompress/parallel/inflate_impls/{scalar.rs, double_literal.rs, simd_avx2.rs, bmi2.rs}`.
- Each impl exports a `decode_dynamic_huffman_block` function with identical signature.
- A dispatcher selects at startup: `static IMPL: OnceLock<DecodeFn> = ...; fn select() { if avx2_available() { Some(simd_avx2::decode) } else ... }`.
- Production runs the fastest available; tests run ALL impls and diff.

| Axis | Score | Note |
|---|---|---|
| Headroom | ++ | Multiple impls coexist as production code. New SIMD/BMI2 ports land alongside scalar instead of replacing. |
| Vendor parity fit | + | Vendor uses a similar pattern (`gzip/isal/IsalInflateWrapper.hpp` layer-above; the inflate layer below has compile-time variants). gzippy can do runtime where vendor does compile-time, slightly different but equivalent. |
| Correctness safety | + | Differential testing across impls is structurally required (every impl must match the scalar reference). |
| Iteration speed | + | Add new impl in its own file; integration is one line of dispatch. |
| Code quality | + | Clean module separation. |
| Optionality | + | Runtime dispatch is the standard pattern; supports CPU diversity (Intel/AMD/older arch). |

**Verdict**: similar to C in capability, but without the workspace boundary. Slightly lower ceiling on iteration speed (rebuilds gzippy main on every change) but no workspace setup. Strong middle ground.

### F. Generic in input/output types (vendor-faithful template-style)

- Inner loop is `fn decode_dynamic_huffman_block<W, R, const CONTAINS_MARKERS: bool>(...)` where:
  - `W: WindowBuffer` — abstracts ring buffer write (could be `[u8; N]` or `[u16; N]`).
  - `R: LsbBitReader` — abstracts bit reader.
  - `CONTAINS_MARKERS: bool` — const generic.
- Each instantiation produces a separate monomorphized symbol.
- Plus separate `gzippy-inflate` crate (C).

| Axis | Score | Note |
|---|---|---|
| Headroom | ++ | Vendor instantiates `Block::read` on `VectorView<uint8_t>` AND `VectorView<uint16_t>` AND multiple bit readers. Maximum generic-template parity. |
| Vendor parity fit | ++ | Direct mirror of vendor's templated approach. |
| Correctness safety | + | Each instantiation independently tested. |
| Iteration speed | + | Same as C/E with extra type-axis. |
| Code quality | + | More complex generics; slightly harder to read but maps to vendor 1:1. |
| Optionality | ++ | Adding a new bit-reader variant or buffer type is a new instantiation, not a rewrite. |

**Verdict**: maximum vendor-template parity. The right choice if vendor's templating approach is itself the source of vendor's performance (it lets LLVM specialize each combination perfectly).

### G. Auto-generated from DSL

- Describe deflate state machine + Huffman decoders in a custom DSL.
- Code generator emits Rust scalar, Rust SIMD, hand-asm versions from single source.

| Axis | Score | Note |
|---|---|---|
| Headroom | ++ | Theoretical maximum; could generate code beyond what humans write by hand. |
| Vendor parity fit | − | Vendor doesn't do this. Off-axis. |
| Correctness safety | 0 | DSL itself needs verification; correctness moves up a level. |
| Iteration speed | 0 | DSL changes regenerate all backends; fast for cross-cutting changes, slow for impl-specific tweaks. |
| Code quality | 0 | The DSL is now the codebase; the generated code is opaque. |
| Optionality | + | New backends are codegen targets. |

**Verdict**: research-grade ambition. Doesn't match the team's actually-known levers (port DoubleLiteralCached, add SIMD); the DSL would be solving a meta-problem we don't have.

## Composite recommendation (my pre-Opus reading)

C (separate crate) + F (generic input/output types) + E (runtime CPU dispatch
within the crate) is the maximum-headroom architecture:

- Workspace crate provides the boundary.
- Generic input/output types match vendor's templated approach.
- Multiple impls coexist as sibling modules inside the crate.
- Runtime dispatch picks the best at startup.
- Each impl has its own benches and property tests within the crate.

This combo gives:
- Maximum isolation from the integration pipeline (less regression risk).
- Maximum optionality for future impls (new SIMD, BMI2, scalar variants all land as new modules).
- Maximum vendor parity (templates → generics directly).
- Maximum correctness coverage (differential tests across all impls structurally enforced).
- Maximum iteration speed (crate-local builds + benches).

The composite has no obvious ceiling. The remaining performance question
becomes "what specific Huffman / SIMD / asm techniques close the 1.5× per-byte
gap?" — which is what the architecture enables us to explore freely.

## Cross-cutting concerns these architectures must support

Regardless of which architecture is chosen, the architecture must support:

- **Property tests** with libdeflate-compressed random inputs (happy path)
  plus hand-built adversarial fixtures (boundary classes).
- **Differential tests** that diff output across all live impls.
- **Microbenchmarks** on a pre-extracted silesia-blocks corpus (two
  variants: with and without Huffman-tree-build cost).
- **Vendor side-by-side comparison** via a standalone C++ harness (no
  FFI link) that reads the same corpus and prints MB/s.
- **`cargo asm` inspection** of each impl's hot path; comparison to
  vendor's `objdump`.
- **Shadow-mode** (feature-gated) wrapper that runs new impls against
  scalar reference on production inputs for one release cycle.
- **CPU feature detection** at startup; dispatcher picks best available.
- **Vendor file:line citation** in each impl's doc comment (CLAUDE.md
  port-don't-innovate); novel impls justify their novelty in a design
  doc.

## The question for the advisor

Given infinite implementation time, which of A-G (or another not listed)
best positions gzippy to close the 1.5× per-byte inner-loop gap with
rapidgzip? Specifically:

1. Is the composite (C + E + F) right, or does it overweight optionality
   at the cost of focus?
2. Are there architectural moves that DON'T fit on these axes — e.g.,
   moving to `no_std` for the inflate crate, requiring `nightly` for SIMD
   intrinsics, embedding hand-written assembly?
3. Is there a meta-level architecture decision we're missing — e.g., the
   problem isn't "the inner loop" but "the pipeline shape" (out-of-order
   completion, larger chunks, bigger prefetch ring) and architecting the
   inner loop differently won't close the gap regardless?
4. Given the team's track record (4 of last 4 attempted optimizations were
   net-zero or regressions despite being structurally sound), is the right
   architecture itself the lever, or is the lever to first nail down a
   measurement methodology that makes wins detectable?
5. What's the SHAPE of the iteration loop that produces vendor-parity
   performance? Single impl with deep tuning, or many impls with breadth?
   The architecture choice depends on which answer is right.

Pick the architecture, justify with concrete optimization paths it enables,
and identify the highest-impact opening move within that architecture.
