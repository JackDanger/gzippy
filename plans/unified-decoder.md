# Unified deflate decoder

**Scope.** The absolute-best pure-Rust deflate decoder for gzippy's
parallel-SM path. Effort is not a constraint. This plan describes
the target end-state; phasing is a separate concern (§6) that does
not shape the design.

**Anchor commit.** `85b5ad3`. Ground truth: every file:line citation
in this doc resolves against this commit.

**Audience.** Future implementers. Read end-to-end; do not skim
sections looking for "what to do first" — the design is a whole.

---

## 1. What the ideal decoder IS (one sentence)

A single Rust type — `Inflate<MODE: DecodeMode, ARCH: ArchProfile>` —
whose hot loop is fully monomorphised, runs to completion without
yielding when output is bounded (parallel-SM), preserves resumability
when output is streaming (BGZF / multi-member), shares one set of
libdeflate-shape Huffman tables across speculative-marker mode and
clean-window mode, and is provably byte-equivalent to libdeflate on
all RFC-1951-valid input.

The current two-decoder split (`Block` + `ResumableInflate2`) is a
historical accident of porting order — `Block` was first as a literal
rapidgzip port, `ResumableInflate2` was bolted on later as an ISA-L
FFI replacement. The unified design does not preserve that split;
it replaces both.

---

## 2. The architectural pillars

### 2.1. Three orthogonal const generics, one inner loop

```rust
trait DecodeMode {
    type Elem: Copy + Default;       // u8 (Clean) or u16 (Markers)
    const EMITS_MARKERS: bool;
    const ELEM_BYTES: usize;         // 1 or 2
}

struct Clean;   impl DecodeMode for Clean   { type Elem = u8;  const EMITS_MARKERS = false; const ELEM_BYTES = 1; }
struct Markers; impl DecodeMode for Markers { type Elem = u16; const EMITS_MARKERS = true;  const ELEM_BYTES = 2; }

trait ArchProfile {
    const HAS_BMI2: bool;
    const HAS_AVX2: bool;
    const HAS_NEON: bool;
    const HAS_PCLMUL: bool;
    /// Bit-buffer width in bits (64 or 128).
    const BITBUF_BITS: u32;
}

struct Generic;     impl ArchProfile for Generic     { /* u64 bitbuf, scalar */ ... }
struct X86_64Bmi2;  impl ArchProfile for X86_64Bmi2  { const HAS_BMI2 = true; const BITBUF_BITS = 128; ... }
struct AArch64Neon; impl ArchProfile for AArch64Neon { const HAS_NEON = true; const BITBUF_BITS = 128; ... }

trait OutputModel {
    /// Resumable: yield on output-full. Owned: run to completion.
    const RESUMABLE: bool;
    /// Yield checks per iteration (true) or only at end (false).
    const PER_ITER_YIELD_CHECKS: bool;
}

struct OwnedOutput;     impl OutputModel for OwnedOutput     { const RESUMABLE = false; const PER_ITER_YIELD_CHECKS = false; }
struct ResumableOutput; impl OutputModel for ResumableOutput { const RESUMABLE = true;  const PER_ITER_YIELD_CHECKS = true; }

pub struct Inflate<M: DecodeMode, A: ArchProfile, O: OutputModel> { ... }
```

LLVM monomorphises into the cross product. The TRUE hot path for
parallel-SM speculative decode is
`Inflate<Markers, X86_64Bmi2, OwnedOutput>`. The clean continuation is
`Inflate<Clean, X86_64Bmi2, OwnedOutput>`. BGZF / multi-member use
`Inflate<Clean, *, ResumableOutput>`. Six total specializations across
the cross product; LLVM dead-code-eliminates all the per-iter checks
that don't apply to the chosen monomorphisation.

This is the rapidgzip pattern (`deflate.hpp::Block<containsMarkerBytes>`)
generalized to Rust + extended to arch-profile + resumability axes.

### 2.2. The owned-output mode eliminates the yield tax

The resumable contract exists because today's callers (parallel-SM
workers) pass fixed-size scratch buffers. The decoder yields on
fill; the caller drains and re-calls. This pays a per-iteration
yield check tax — typically 1-3% of cycles, more in chunked-call
patterns.

In `OwnedOutput` mode, the decoder owns its output buffer (sized
to the chunk's max-expansion-ratio bound, ~`5 × deflate_len` or
~80 MiB whichever smaller, allocated via `RpmallocAlloc` so the
arena amortizes the allocation cost). The FASTLOOP runs without
yield checks. The SAFE LOOP only handles the BFINAL tail.

Parallel-SM workers transition to `OwnedOutput` natively. BGZF /
multi-member keep `ResumableOutput` because their callers want
streaming. The Huffman inner loop is shared across both via the
const generic; the `PER_ITER_YIELD_CHECKS = false` monomorphisation
literally has zero yield checks emitted in the inner loop.

Net: parallel-SM workers run a strictly tighter inner loop than
today's `ResumableInflate2`, which has yield checks even after the
T4 FASTLOOP yield-elide because the SAFE LOOP fallback compiles them
in.

### 2.3. 128-bit bit-buffer via two u64s + SIMD refill

Today's `Bits` uses a 64-bit `bitbuf` refilled in 8-byte chunks via
unaligned `u64` load. Min refill threshold ~48 bits → refill every
~3 symbols.

The ideal uses two `u64`s (`bitbuf_lo`, `bitbuf_hi`) forming a
128-bit shift register. Refill loads 16 bytes via `_mm_loadu_si128`
(x86) / `vld1q_u8` (arm64), splits into high/low halves, shifts in.
Refill threshold ~96 bits → refill every ~6 symbols.

Halves the refill frequency. Same per-refill cost (one 16-byte load
is single-cycle on M-series / Zen 3 vs two 8-byte loads). The
arithmetic for shift/consume is a two-step (low → high) but folds
into the same number of µops on modern OoO cores.

This is NOT a vendor pattern — libdeflate uses 64-bit. We diverge
here because the 128-bit buffer halves a documented hot cost. CLAUDE.md
2026-05-27 (final) authorizes inner-loop innovation.

### 2.4. Dynamic main-table size per block

Today `LitLenTable::TABLE_BITS = 12` is a const. Lever 2 (TABLE_BITS=13)
was falsified on arm64.

The ideal: choose `TABLE_BITS` per block based on the block's max
code length, determined at table-build time:
- `max_code_len ≤ 10`: `TABLE_BITS = 10`, no subtables (table = 4 KiB)
- `max_code_len ≤ 12`: `TABLE_BITS = 12`, rare subtables (table = 16 KiB)
- `max_code_len ≤ 15`: `TABLE_BITS = 12` with subtables (current)

The vast majority of real blocks have `max_code_len ≤ 12` (English
text, code, JSON, etc.). The smaller table fits hot in L1 and reduces
table-build cost (4 KiB vs 16 KiB to zero-initialize per block).

Block-level dispatch via a function pointer set after table-build:
```rust
enum DecodeStrategy {
    Tiny(Inflate<..., TABLE_BITS=10>),
    Standard(Inflate<..., TABLE_BITS=12>),
}
```
Selected once per block, runs in tight loop until block end.

### 2.5. 8-literal multi-batch with adaptive depth

Today T3 goes to 4 literals max with the carry-forward refill fix.

The ideal: up to 8 literals per iteration with packed u64 write.
Adaptive depth: track recent literal-run length per block; if the
last 4 iterations were all 8-literal batches, the predictor is hot —
try 8 again. If the last was a length code at depth 2, back off to
2 next iteration.

```rust
struct LiteralBatchPredictor {
    last_depth: u8,           // 1..=8
    consecutive_full_batches: u8,
}
```

This is genuinely novel relative to libdeflate (which always tries
the same depth). On highly-text inputs the predictor sustains 8;
on mixed inputs it backs off and avoids wasted lookahead.

In `<Markers>` mode, the packed write is a `_mm_storeu_si128` of 8
u16s (16 bytes) on x86, or `vst1q_u16` on arm64.

### 2.6. Branchless subtable detection via table layout

Today `is_subtable_ptr()` checks bit 14 of every entry. Mispredicts
on transitions between literal-heavy and length-heavy regions.

The ideal: lay out the table so subtable entries are in a contiguous
high-index region. The main table's logical mapping becomes:
- Indices `[0, N_codeword_entries)`: literal/length/EOB entries
- Indices `[N_codeword_entries, table_size)`: subtable pointers

A single bound check (`if idx >= subtable_threshold { subtable_path }`)
replaces the bit-test. Predicts perfectly for inputs where the input
distribution doesn't cross the threshold often.

Plus: subtable entries can be laid out contiguously with their
target sub-arrays adjacent in memory — prefetcher pulls in the
sub-array on the same cache line as the subtable pointer. Zero L1
miss on subtable resolve.

### 2.7. Runtime BMI2 dispatch via function-pointer

Today `bmi2::decode_extra_bits` is compile-time-gated on
`target_feature = "bmi2"`. Portable binaries don't get BZHI.

The ideal: decoder construction runs `is_x86_feature_detected!("bmi2")`
once and stores a function-pointer table on the `Inflate` instance.
The hot loop reads `self.extract_bits_fn(saved, codeword, extra)` —
LLVM hoists the pointer load to a register at function entry; the
inner-loop cost is one indirect call (~3 cycles) vs the BZHI win
(~2 cycles vs shift+mask).

For portable builds this is net-neutral. For BMI2-enabled builds
the function-pointer is `decode_extra_bits_bmi2` and inlines to a
direct call → BZHI emit.

Critically: the dispatch is per-decoder-construction (cold), not
per-symbol (hot). Eliminates the today's compile-time gate without
the per-symbol indirect-call tax.

### 2.8. Marker-tail invariant as a phantom-type proof

The bootstrap → Phase 2 flip is gated by "trailing 32 KiB is
marker-free." Today this is an `assert!()` at `gzip_chunk.rs:1290`.

The ideal: encode the invariant in the type system.

```rust
pub struct MarkerBuffer<const CLEAN_TAIL_BYTES: usize> {
    storage: Box<[u16]>,
    written: usize,
}

impl MarkerBuffer<0> {
    pub fn new(cap: usize) -> Self { ... }
}

impl<const N: usize> MarkerBuffer<N> {
    /// Returns Self<{ N + bytes }> if the appended bytes are clean.
    /// Returns Self<0> if any markers landed in the trailing window.
    pub fn append_clean_run<const BYTES: usize>(self, bytes: &[u8; BYTES])
        -> MarkerBuffer<{ N + BYTES }> { ... }

    pub fn append_with_potential_markers(self, ...) -> MarkerBuffer<0> { ... }
}

impl MarkerBuffer<{ MAX_WINDOW_SIZE }> {
    /// Type-only transition. Compiles only when CLEAN_TAIL_BYTES = 32768.
    pub fn flip_to_clean(self) -> (CleanBuffer, Window32KiB) { ... }
}
```

The flip from `<Markers>` to `<Clean>` becomes a type-system proof.
You CAN'T call `flip_to_clean` unless the trailing 32 KiB is
provably marker-free.

Caveat: `const_generic_exprs` is partly unstable. The ideal accepts
nightly Rust if needed; the cost is worth the proof. If stuck on
stable, the type-state pattern degrades to a runtime check with the
same shape (`fn try_flip_to_clean(self) -> Result<(CleanBuffer, Window)>`)
but without the compile-time guarantee.

### 2.9. Match copy unified across modes

Today: `copy_match_fast` for clean (u8 + SIMD), `emit_backref_ring`
for markers (u16 + ring arithmetic).

The ideal: one generic `copy_match<M: DecodeMode>` that uses
const-generic dispatch:
- `M::Elem = u8` → SIMD u8 copy (current `copy_match_fast` shape)
- `M::Elem = u16` → SIMD u16 copy + marker-zone arithmetic (current
  `emit_backref_ring` shape, generalized to use SIMD widening)

The shared logic (dist/length validation, RLE fast path for dist=1,
back-ref bounds check) lives in one place. Mode-specific bits
(marker bookkeeping, distance-to-last-marker counter update) are
const-generic-gated.

### 2.10. Cache-aware Huffman table layout

Today the litlen + dist tables are heap-allocated `Vec<u32>`, no
alignment guarantee.

The ideal: tables are `#[repr(align(64))]`, padded to cache-line
boundaries, allocated from a per-block table arena. After a block
completes, the arena resets — old tables are explicitly invalidated
via `_mm_clflushopt` (x86) / `dc cvac` (arm64) to avoid LRU pressure
on the new tables' first-touch.

Tables for hot blocks (BTYPE=01 fixed Huffman, top-N most-common
dynamic patterns) are pre-built at startup, cached, and re-used —
this is rapidgzip's pattern (`HuffmanCodingISAL` lookup cache)
applied to Rust.

### 2.11. Cross-chunk window propagation via Arc

Today windows are passed via `WindowMap` (mutex-guarded HashMap).
Each successor chunk `clone()`s the 32 KiB Vec<u8>.

The ideal: windows are `Arc<[u8; 32768]>`, published lock-free into
a `Vec<Mutex<Option<Arc<...>>>>` indexed by chunk index. Successor
chunks `Arc::clone()` — zero copy, zero allocation. The
`Mutex<Option<Arc>>` is poll-via-condvar so successors block exactly
as long as needed and wake instantly when the predecessor publishes.

### 2.12. Provable correctness — four layers

1. **Property-based tests** (proptest crate, deterministic shrinking).
   Random payloads 1B–4MiB × 9 compression levels × ~20 output buffer
   sizes × both modes × both arch profiles. Shrinks to minimal repro
   on failure.

2. **Differential fuzzing** (cargo-fuzz, libfuzzer-sys) with libdeflate
   as the oracle. Mutator produces both valid and malformed deflate
   streams. Both decoders must agree on success/output-bytes; on
   malformed input they may differ in error type but not in
   data-corruption behavior.

3. **Real-corpus integration test**: silesia (211 MB) + linux-source
   (~1 GB) + every other corpus we can scrape. Full file decode at
   gzip -1, -6, -9 via both modes. Byte-identical to libdeflate
   one-shot.

4. **Model checking** via `kani` on the `Bits` state machine. Prove
   the invariant `bitsleft ≤ BITBUF_BITS` for all sequences of
   `refill()`, `consume(n)` with `n ∈ [0, 48]`. ~50-line harness;
   kills the entire class of bit-buffer-management bugs.

These four layers compose: proptest catches API-shape bugs, fuzz
catches malformed-input bugs, real-corpus catches workload-specific
bugs (T3 was caught by real-corpus, missed by proptest), kani
catches state-machine invariants.

---

## 3. What we delete

Once the ideal decoder lands AND has neurotic confirmation of
parity, the following are deleted in a single commit:

- `src/decompress/parallel/deflate_block.rs` (`Block`, ~2,200 lines)
- `src/decompress/parallel/huffman_short_bits_multi_cached.rs`
- `src/decompress/parallel/huffman_reversed_bits_cached.rs`
- `src/decompress/parallel/huffman_short_bits_cached_deflate.rs`
- `src/decompress/parallel/huffman_symbols_per_length.rs`
- `src/decompress/parallel/huffman_base.rs`
- `src/decompress/parallel/rfc_tables.rs`
- `src/decompress/parallel/isal_huffman.rs`
- `src/decompress/inflate/resumable.rs` (folded into `inflate.rs`)
- `src/decompress/inflate/libdeflate_decode.rs`
- `src/decompress/inflate/libdeflate_entry.rs` (folded — tables move to
  `inflate.rs`)
- `src/decompress/inflate/consume_first_decode.rs` (BGZF / sequential
  decompress consumer; folded into `inflate.rs` as the
  `<Clean, *, ResumableOutput>` monomorphisation)
- `src/decompress/inflate/specialized_decode.rs`,
  `consume_first_table.rs`, `jit_decode.rs`, `two_level_table.rs`,
  `vector_huffman.rs`, `double_literal.rs`, `bmi2.rs` (folded;
  techniques absorbed into the unified hot loop)
- The `IsalInflateWrapper` `pure-rust-inflate` cfg branch in
  `inflate_wrapper.rs` (callers go direct to `Inflate`)

Same commit, also delete the ISA-L FFI:
- `vendor/isa-l/` (submodule)
- `vendor/isal-rs/` (submodule)
- `packaging/isal-patches/`
- `src/backends/isal_decompress.rs`
- `src/backends/isal.rs`
- `Cargo.toml`'s `isal-rs` dep, `isal-compression` feature,
  `[patch.crates-io].isal-sys`
- The `isal-compression` cfg branches in `inflate_wrapper.rs`

Estimated net: ~10,000 lines deleted, ~3,500 lines added. The new
`inflate.rs` is ~3,500 lines of one decoder with all axes
monomorphised, all tests, all docs.

Reduction in test-surface: today we maintain test coverage for two
decoders. After: one decoder, four monomorphisations. Each
optimization lands once.

---

## 4. The C/C++ library boundary, re-examined

The prior plan accepted libdeflate's role as "the one-shot fast
decoder we don't replace." That deferred to effort.

The ideal asks: should `Inflate<Clean, *, OwnedOutput>` ALSO replace
libdeflate in sequential single-member and BGZF? Three considerations:

1. **libdeflate is hand-tuned C with 8+ years of investment.** Beating
   it in pure Rust is hard but not impossible — we already match
   it within 1.6× on arm64 (chunked), within 1.3× on x86_64 mono
   (extrapolated from neurotic data). With the ideal architecture's
   wins (128-bit bitbuf, dynamic table size, 8-literal batches,
   marker-tail type proof) we project closing to within 5%.

2. **libdeflate's GZIP path includes header parse + CRC32 + ISIZE
   verification.** The ideal `Inflate` exposes only DEFLATE; a thin
   GzipReader wrapper (Rust, ~200 lines) provides the gzip-format
   adapter using `crc32fast`'s pclmulqdq path.

3. **libdeflate has multi-version SIMD dispatch.** The ideal's
   `ArchProfile` const generic + runtime dispatch covers the same
   ground.

**Decision in the ideal: yes, replace libdeflate too.** Once
`Inflate` is within 5% of libdeflate on a representative benchmark,
the BGZF / multi-member / sequential SM paths route through it. The
`libdeflater` dep is deleted. The result: gzippy is end-to-end pure
Rust for decode, with no C dependencies for the inflate kernel.

For compression we keep ISA-L (L0-L3) and zlib-ng (L6-L9). Different
problem, out of scope.

---

## 5. Done-when

The design is done when:

1. **Functional correctness:** all four correctness layers (§2.12)
   pass on all corpora.
2. **Parallel-SM perf:** `Inflate<Markers, X86_64Bmi2, OwnedOutput>`
   + `Inflate<Clean, X86_64Bmi2, OwnedOutput>` together via the
   parallel-SM pipeline reaches **≥ 0.95× of rapidgzip** (not just
   ISA-L) on neurotic `make bench-sm`. We pick the higher bar
   because that's the target.
3. **Sequential perf:** `Inflate<Clean, *, ResumableOutput>` for
   BGZF / multi-member / sequential SM reaches **≥ 0.95× of
   libdeflate one-shot** on representative corpora.
4. **Code reduction:** ~10,000 lines deleted across the modules in
   §3; net is ~3,500-line `inflate.rs`.
5. **Zero C dependencies for inflate:** `libdeflater`, `isal-rs`
   removed from `Cargo.toml`. (Compression-only `flate2`/`isal-rs`
   stays in scope.)
6. **Opus advisor sign-off** on the neurotic bench AND a 14-day
   soak (catches load-dependent regressions).

No "ship if ≥0.93×, revert if <0.93×" hedge. The target is 0.95× of
rapidgzip; if we miss, we iterate until we hit, not revert.

---

## 6. Phasing (separate from design)

The above describes the end state. Implementation can phase how the
implementer prefers — this section is non-normative; the design does
not depend on it.

Suggested order:
1. Build `Inflate<Clean, *, ResumableOutput>` (drop-in replacement
   for `ResumableInflate2`). Validates the const-generic shape on
   the known-correct path.
2. Add `<Markers>` monomorphisation. Bootstrap-equivalent. Validates
   the marker-tail type proof + ring path.
3. Add `<*, *, OwnedOutput>` monomorphisations. Eliminates yield
   tax for parallel-SM.
4. Add `ArchProfile` axis. Per-arch builds.
5. Migrate parallel-SM callers.
6. Migrate BGZF / multi-member / sequential SM callers.
7. Delete all the modules in §3.
8. Delete libdeflate dep.

Some phases parallelize (1+2+3 can be one work-stream while
correctness tests are built; 5+6 can be parallel after 4). The
phasing affects schedule, not design.

---

## 7. Risks

1. **Const-generic explosion bloats binary.** Cross-product is 2×3×2 = 12
   monomorphisations of a ~3,500-line function. At ~30 KB per
   monomorphisation that's ~360 KB of code in the binary. Mitigation:
   `#[inline(never)]` on cold monomorphisations, `#[inline]` only on
   the truly-hot pair.

2. **128-bit bit-buffer's two-step shift hurts non-SIMD targets.** On
   targets without 16-byte SIMD load, the refill costs two scalar
   loads + ALU shift compose. Falls back to 64-bit buffer on Generic
   arch profile.

3. **Type-state marker-tail proof requires nightly Rust.** Mitigation:
   accept nightly for the type-proof variant; provide a stable-Rust
   `try_flip_to_clean()` fallback with the same runtime check (just
   without the compile-time guarantee).

4. **8-literal batch may pessimize on non-literal-dominated workloads.**
   The adaptive predictor handles this; needs differential perf vs
   fixed-depth-4 across corpora to confirm.

5. **Cache flush instructions add cycles** on transitions; only worth
   it if table-build is happening frequently (small blocks). Gate by
   block size: skip flush if next block is >32 KiB output.

6. **Replacing libdeflate is a separate strategic decision.** If we
   can't reach 0.95× of libdeflate on sequential SM, we keep
   libdeflate and only replace ISA-L FFI. Design remains valid; just
   stops shy of step 8.

---

## 8. Provability requirements (the non-negotiable)

The user's directive: *"provably correct."* The ideal design
treats this as a hard constraint:

- Every `unsafe` block has an explicit SAFETY comment with the
  invariant it relies on, paired with a `debug_assert!` checking
  the invariant in debug builds.
- The bit-buffer state machine has a kani harness proving the
  `bitsleft ≤ BITBUF_BITS` invariant for all input sequences.
- The marker-tail invariant is type-encoded (phantom-type const
  generic) so flip-to-clean is a compile-time-checked transition.
- Property-based tests (proptest) exhaustively explore
  `(payload, buffer_size, schedule, level)` configuration space.
- Differential fuzzing (cargo-fuzz vs libdeflate) runs in CI for
  N hours per commit, with corpus accumulation across builds.
- Real-corpus tests cover silesia, linux-source, web-archive
  samples, JSON streams, code repos — workloads that produce
  different deflate patterns than synthetic fixtures.

A bug class that escapes ALL of these is one we accept as residual
risk. The T3-on-silesia bug we caught was caught by layer 3; we
add layers 1, 2, 4 to catch the next one in the cheapest layer
available.
