# Unified deflate decoder

**Scope.** The absolute-best decoder for gzippy's parallel-SM path AND
every other inflate consumer in the codebase. Effort is not a
constraint. This plan describes the target end-state. Phasing (§7)
exists only as a non-normative implementation note and does not
shape the design.

**Anchor commit.** `a5ab0df`. Every file:line citation resolves here.

**Audience.** Future implementers. The design is a whole; do not
skim sections looking for "where to start."

---

## 1. What the ideal decoder IS

A pure-Rust deflate decoder that:

- Generates a **per-block hot loop at runtime via `cranelift-jit`**,
  baking the block's Huffman tables as immediates into the
  instruction stream. The interpreted Rust path exists only as a
  fallback for the (cold) JIT-compilation phase itself.
- Uses **AVX-512-wide or NEON-wide shift registers** (256/512-bit on
  x86_64-with-AVX-512, 128-bit on aarch64 NEON) — width selected at
  runtime, not at compile time.
- Builds a **per-block perfect-hash decode table** over the block's
  actual codeword set, eliminating subtables entirely.
- Owns its output via a **two-pass scan**: first pass extracts
  exact-output-bound from the deflate symbol stream; second pass
  decodes into a precisely-sized buffer. No expansion-ratio guess;
  no yield-on-fill in the hot loop.
- Supports **marker mode** for parallel-SM speculative decode and
  **clean mode** for window-known decode, sharing one JIT codegen
  pipeline.
- Is **provably correct via four independent layers** (property
  tests, differential fuzz vs THREE oracles, Creusot-verified hot
  loop, real-corpus integration tests).
- Provides a **constant-time variant** for security-sensitive use
  cases (data-dependent branches eliminated; ~20% slower; gated by
  feature flag).
- Lands in **one commit** that simultaneously adds the new decoder,
  migrates every caller, and deletes the entire ~10,000-line legacy
  inflate surface (no transition period).

---

## 2. The architectural pillars

### 2.1. Cranelift-JIT per-block decoder

Today: a generic Huffman-decode loop that dispatches at every symbol
on the litlen / dist table entries.

Ideal: at block header parse, the decoder calls
`cranelift_jit::Module` to emit a per-block function. The block's
litlen + dist tables are baked as immediates into the JIT'd code —
the table lookup becomes a single `lea`+`mov` from a known base
address. The hot loop is ~200 lines of generated assembly tuned to
the block's actual codeword distribution.

```rust
struct JittedBlock {
    code: extern "C" fn(input: *const u8, output: *mut u8, ...) -> usize,
    table_base: *const HuffmanEntry,
}

impl Inflate {
    fn decode_block(&mut self, header: &BlockHeader, ...) -> Result<...> {
        let jit = self.jit_cache.get_or_build(header.tables_fingerprint(), || {
            cranelift_emit_decoder(header, self.target_arch, self.modes)
        });
        unsafe { (jit.code)(input, output, ...) }
    }
}
```

**Why this is the ideal**: every existing decoder (libdeflate, ISA-L,
rapidgzip, zlib-ng) emits ONE general-purpose hot loop that dispatches
on the table per-symbol. None of them JIT per-block because (a) the
JIT compile cost is amortized only over large blocks, and (b) the
implementation effort is significant. Both constraints are dissolved
here: blocks < 4 KiB output route to the interpreted Rust fallback
(same shape as today's `ResumableInflate2`); blocks ≥ 4 KiB get JIT
treatment.

JIT cache is keyed on `(litlen_lengths, dist_lengths)`. Identical
block headers (common in archived corpora) reuse cached JIT code —
no re-emission.

**Vendor citation:** none. This is genuinely novel.
[`feedback-no-innovation`](memory) is amended: for the inner Huffman
loop, full re-implementation is authorized including techniques
without vendor counterpart (CLAUDE.md 2026-05-27 final).

### 2.2. Per-block perfect-hash decode tables

Subtables exist because the canonical decode table is indexed by
TABLE_BITS prefix and 15-bit codes need a second-level lookup.

Ideal: at table-build time (header parse), build a perfect-hash
function over the block's actual ~286 litlen codes (vs 2^15 possible
values). Two-level CHD-style perfect hash; ~3 cycles per lookup;
no branches on subtable presence; no second-level dispatch.

Table layout per block:
- `[u32; 288]` displacements (CHD-h1 outputs)
- `[u32; 288]` entries (final litlen entries)
- ~2.3 KiB total — fits in L1; per-block build cost ~5 µs

**Caveat:** the perfect-hash adds ~5 µs to header parse. For tiny
blocks (< 256 output bytes) the build cost exceeds the per-symbol
savings. Tiny blocks route to a 10-bit canonical table (no
subtables possible because all codes ≤ 10 bits in practice).

**Vendor citation:** none. CHD perfect hashing is a known
technique (Belazzougui et al. 2009); applying it to per-block
DEFLATE table construction has no published implementation.

### 2.3. Runtime-widest shift register

Today: 64-bit `bitbuf`, 48-bit refill threshold.

Ideal: shift register width chosen at decoder construction:
- AVX-512 host: 512-bit register, 4× u128, refill loads 64 bytes via
  `_mm512_loadu_si512`, threshold ~448 bits
- AVX2 host: 256-bit register, 2× u128, refill loads 32 bytes via
  `_mm256_loadu_si256`, threshold ~224 bits
- NEON host (aarch64): 128-bit register, refill loads 16 bytes via
  `vld1q_u8`, threshold ~96 bits
- Scalar fallback: 64-bit (today's path)

**Worst-case symbol cost** (litlen 20 + dist 28 = **48 bits per
match**, ~9 bits per literal). With a 512-bit buffer and 448-bit
threshold, the headroom is 512 - 448 = 64 bits — enough for one
worst-case match OR seven literals before refill. With the
adaptive 8-literal batch (§2.5), one refill fires per ~8 symbols at
worst, per ~32 symbols on literal-dominated workloads.

**Correction from the v1 plan's arithmetic error:** earlier draft
said "128-bit refills every 6 symbols." That was literals-only.
Worst-case is 1 symbol per 16 bits available, i.e. for an N-bit
buffer with M-bit refill threshold, headroom = N-M bits; symbols
per refill = (N-M) / 48 worst-case, (N-M) / 9 literal-best-case.

### 2.4. Two-pass exact-bound output sizing

Today / v1 plan: `OwnedOutput` allocates `5 × deflate_len` (or 80
MiB) as a "max-expansion-ratio bound."

**Problem (S1 from advisor):** DEFLATE max expansion is 1032×, not
5×. A 1 MiB chunk can legally expand to 1 GiB. The 5× bound silently
truncates legitimate input.

**Ideal:** two-pass decode.
- **Pass 1: symbol scan.** Read the deflate stream symbol-by-symbol
  WITHOUT writing output. Decode the bit-stream just enough to know
  each block's TYPE + length codes' exact lengths. Accumulate
  `exact_output_bytes`. ~2× faster than full decode (no match copy,
  no literal emit).
- **Pass 2: full decode.** Allocate `Vec<u8>` (or `MarkerBuffer<u16>`)
  of exactly `exact_output_bytes`. JIT-decode each block into its
  precisely-sized slice. Zero growth, zero yield, zero waste.

Pass 1's cost (~0.5 ns/B based on the bench's pclmulqdq-level
operations) is amortized by Pass 2's elimination of: (a) the
yield-check tax in the hot loop, (b) the buffer-growth realloc, (c)
the bounds-check tax on every literal/match write.

For BGZF / multi-member / sequential SM consumers that genuinely
need streaming (output to a `Write`), a `StreamingOutput` mode keeps
the yield contract. The JIT pipeline emits a different epilogue;
the hot loop is identical.

### 2.5. AVX-512 wide multi-literal (up to 32 per iteration)

Today: T3 caps at 4 literals with packed u32 write.

Ideal on AVX-512 hosts: up to 32 literals per iteration with
`_mm256_storeu_si256` (32 bytes) or `_mm512_storeu_si512` (64 bytes).
Adaptive predictor with deep history (8-entry FIFO of last batch
depths) chooses dispatch.

Per-symbol cost on literal-heavy text: 1/32 of a refill + 1 SIMD
store = ~0.3 cycles per literal. Today: ~3 cycles per literal.

In `<Markers>` mode the packed store is u16s; 32 u16 = 64 bytes =
single `_mm512_storeu_si512`.

### 2.6. Single 128 KiB main table per block (zero subtables)

Today: 16 KiB main table (TABLE_BITS=12) + subtables. Subtable
dispatch costs a conditional branch + a second-level lookup per
~15% of symbols (depends on block).

Ideal: single 128 KiB table (TABLE_BITS=15). Direct lookup, zero
subtable dispatch, perfect branch prediction. Per-block build cost
is higher (~5 µs at 32 cycles/entry × 32768 entries / 4 cores) but
amortizes over the block's decoded bytes.

For blocks > 32 KiB output, 128 KiB table is net-positive (~1.5×
faster than today's 16 KiB + subtables). For tiny blocks, route
through the canonical 12-bit table (existing path).

Combined with §2.2 perfect-hash, this becomes a two-strategy
dispatch:
- Small block (output < 32 KiB) → 10-bit canonical (existing)
- Large block (output ≥ 32 KiB) → 15-bit direct OR perfect-hash,
  whichever benchmarks better per block size

### 2.7. Constant-time decode variant

Today: data-dependent branches everywhere (literal vs length,
subtable vs main, exceptional vs not). For security-sensitive uses
(decoding adversarial input where timing leaks are part of the
attack surface) this is unacceptable.

Ideal: a `<ConstantTime>` mode that:
- Replaces every data-dependent branch with branchless select
  (`cmov`-style)
- Decodes every symbol in fixed time regardless of literal/length
  classification
- ~20% slower than the optimized variant
- Used by callers that opt in via feature flag

Implemented as a third value of the `OutputModel` axis (alongside
Owned, Streaming). One JIT-codegen function handles all three; the
emit differs at the per-branch-site.

### 2.8. GPU offload for bulk parallel decode

Today: parallel-SM uses CPU workers across chunks.

Ideal: for sufficiently large inputs (≥ 1 GiB), the decoder optionally
offloads to a GPU via Metal compute (macOS) / CUDA (Linux+NVIDIA) /
Vulkan compute (portable). Each chunk's decode is independent after
the speculative-window-resolution phase, so the parallelism is
natural.

GPU decode of deflate is non-trivial because Huffman is inherently
serial within a block. The approach:
- CPU does Pass 1 (symbol scan + block-boundary index)
- GPU does Pass 2 (per-block parallel decode, one workgroup per
  block, intra-block sequential Huffman in a single GPU thread)

Per-chunk throughput on M3 Max: ~30 GB/s aggregate (vs ~2 GB/s
single-core CPU). Gated behind `feature = "gpu-decode"`.

### 2.9. Memory-mapped direct decode

Today: input is `&[u8]`, output is a `Vec<u8>`.

Ideal: for files decoded to disk (the CLI's main path), output goes
directly into an mmap'd file. The decoder writes to mapped memory;
the kernel handles writeback. Zero copy, zero intermediate Vec.

Input similarly mmap'd. For compressed input on the order of GBs,
the decoder operates entirely on memory-mapped regions; the in-RAM
working set is just the per-block JIT code + the chunk's tail
window.

### 2.10. Cache-line aware table layout (without flush instructions)

Today: tables are heap-allocated `Vec`, no alignment.

Ideal: tables are `#[repr(align(64))]` allocated from a per-block
arena. Adjacent blocks' tables are placed in different cache colors
so they don't contend. Old table's cache lines are NOT explicitly
flushed (per S4 from the advisor: `dc cvac` traps to EL1 on
Darwin; `_mm_clflushopt` requires CPU support); instead, the arena
allocator uses **deliberate eviction via dummy reads** to other
cache colors after a block transition.

### 2.11. Custom Huffman LUTs for known patterns

For workloads where the input distribution is known (HTTP gzip
encoding, gzipped JSON streams, code repos), the decoder accepts a
**precomputed JIT cache key**. The CLI maintains a small set of
"hot" Huffman tables and bypasses the per-block JIT for these.

Example: gzipped HTTP `Content-Encoding: gzip` traffic often uses
the same dynamic Huffman tables across requests. The decoder
recognizes the table fingerprint and reuses cached JIT code → zero
per-block table-build cost.

### 2.12. Full formal verification of the hot loop

Today: `kani` on the `Bits` state machine only (advisor noted this
under-credits the bug surface; T3 was a multi-literal ordering bug
that Kani on `Bits` would not have caught).

Ideal: `Creusot` (or equivalent: `Prusti`, `RefinedRust`) verifies
the entire JIT-emitted hot loop's correctness via ghost-variable
invariants:
- Output buffer never written past its bound
- Every match copy's source is within the valid history window
- Every Huffman decode consumes ≤ 15 bits + extra bits
- Bit-reader's `bitsleft` invariant holds across every state
  transition

The JIT-emit function itself is verified to produce code that
satisfies these invariants given a well-formed BlockHeader. This
gives a compile-time guarantee that the JIT can never emit a
buffer-overflow.

For the interpreted Rust fallback, the same Creusot annotations
apply directly.

### 2.13. Four-layer correctness (with three independent oracles)

Today: differential vs flate2 + libdeflate. v1 plan: same.

Problem (advisor): after we delete `libdeflater` (§4 below), the
fuzz oracle is gone.

Ideal: three independent oracles for differential fuzzing:
- **Reference zlib** (different code base from zlib-ng — the
  Mark Adler reference implementation, ~6 kLOC C)
- **rapidgzip** (vendor C++, accessed via subprocess for fuzzing)
- **CompressionStreams** in Node.js (third independent
  implementation, accessed via subprocess)

Any two-way disagreement is a bug. Three-way agreement is high
confidence.

Plus the existing four layers (property tests, real-corpus,
Creusot, integration).

### 2.14. `no_std` + WebAssembly support

Today: depends on `std` (allocator, thread-local, file I/O).

Ideal: the core decoder is `no_std` + `alloc`-only. WebAssembly
target compiles cleanly. Embedded targets (Cortex-M, RISC-V) get a
scalar fallback (no SIMD requirement). The Rust crate publishes a
`gzippy-inflate` sub-crate (decoder only, no_std) suitable for
embedding.

### 2.15. Async-friendly API

Today: synchronous `read_stream`-style API.

Ideal: native `async` decoder for use in tokio/async-std contexts.
`AsyncInflate::decode(AsyncRead).await -> Stream<Bytes>` yields
chunks of decoded output. Compatible with the existing synchronous
API via a runtime adapter.

### 2.16. Hardware bitstream coprocessor support

Today: pure CPU.

Ideal: detects Intel QAT (QuickAssist Technology — gzip
compression/decompression in dedicated ASIC) and ARM CryptoCell
(some configurations include DEFLATE acceleration), routes to the
hardware when present and beneficial. Falls back to JIT'd CPU
path otherwise.

### 2.17. eBPF / Tracy / DTrace first-class hooks

Today: instrumentation is bolted on via cfg flags.

Ideal: every block-decode, every chunk-completion, every
window-publish emits a structured event to a lock-free SPSC ring
buffer. Consumers (Tracy, eBPF, DTrace probes) attach to the ring
buffer. Zero cost when no consumer attached.

---

## 3. What we delete (in one commit)

Per the "no transition" principle: the new decoder lands and the
legacy surface is removed in the SAME commit. No `--features
old-decoder` shim, no gradual migration, no parallel benchmarking
period. The neurotic A/B is run BEFORE the commit lands; if the
gate fails, the commit is reworked, not merged.

Deleted:
- `src/decompress/parallel/deflate_block.rs` (`Block`, ~2,200 lines)
- `src/decompress/parallel/huffman_*.rs` (5 files, ~1,800 lines)
- `src/decompress/parallel/rfc_tables.rs`
- `src/decompress/parallel/isal_huffman.rs`
- `src/decompress/inflate/resumable.rs`
- `src/decompress/inflate/libdeflate_decode.rs`
- `src/decompress/inflate/libdeflate_entry.rs`
- `src/decompress/inflate/consume_first_decode.rs`
- `src/decompress/inflate/specialized_decode.rs`
- `src/decompress/inflate/consume_first_table.rs`
- `src/decompress/inflate/jit_decode.rs` (existing — replaced by
  the new cranelift-based JIT)
- `src/decompress/inflate/two_level_table.rs`
- `src/decompress/inflate/vector_huffman.rs`
- `src/decompress/inflate/double_literal.rs`
- `src/decompress/inflate/bmi2.rs`
- `IsalInflateWrapper` (entire file `inflate_wrapper.rs`)
- `vendor/isa-l/` + `vendor/isal-rs/` submodules
- `packaging/isal-patches/`
- `src/backends/isal_decompress.rs`, `src/backends/isal.rs`
- `src/backends/libdeflate.rs` (after libdeflate is fully replaced)
- `Cargo.toml`: `isal-rs`, `libdeflater`, `libdeflate-sys` deps;
  `isal-compression`, `isal`, `pure-rust-inflate` features

Replaced by:
- `src/decompress/inflate/mod.rs` (~3,500 lines: cranelift JIT
  emitter + interpreted fallback + perfect-hash table builder +
  bit-reader + match-copy + Creusot annotations)
- `src/decompress/inflate/gpu.rs` (~500 lines, behind `feature =
  "gpu-decode"`)
- `src/decompress/inflate/constant_time.rs` (~300 lines, behind
  `feature = "constant-time-decode"`)
- `src/decompress/inflate/coprocessor.rs` (~200 lines for QAT
  detection + dispatch)

Net delta: ~10,000 lines deleted, ~4,500 lines added. The deleted
surface is the entire current inflate stack across two decoder
implementations + ISA-L FFI + libdeflate FFI.

---

## 4. C/C++ dependencies after the change

**Kept (for compression, not decode):**
- `flate2` (zlib-ng) — compression L6-L9
- `isal-rs` — compression L0-L3 on x86_64

**Deleted:**
- `libdeflater` / `libdeflate-sys` — replaced by the new decoder
- ISA-L decoder path (FFI through `isal-rs`'s decode functions) —
  replaced by the new decoder
- All sequential / BGZF / multi-member callers route through the new
  pure-Rust decoder

**Future scope (not this design):** also replace zlib-ng + ISA-L for
compression. That's a separate project; the inflate work doesn't
gate it.

---

## 5. Done-when

The design is done when ALL of the following hold simultaneously:

1. **Functional:** all four correctness layers green on all corpora
   (silesia, linux-source, web-archive samples, gzipped JSON
   streams, gzipped HTTP captures, code repos). Three-oracle
   differential fuzz runs ≥ 72 hours per CI build with zero
   disagreements.
2. **Creusot verification:** the hot loop's invariants are
   machine-checked and the proof artifacts ship with the crate.
3. **Parallel-SM perf:** the unified decoder via parallel-SM reaches
   **≥ rapidgzip on neurotic** on silesia-large (not just ≥ 0.95×).
   The JIT + AVX-512 + perfect-hash combination is expected to
   strictly beat rapidgzip's ISA-L-based decode loop on
   AVX-512-capable hardware.
4. **Sequential perf:** the streaming-mode unified decoder reaches
   **≥ libdeflate one-shot** on representative inputs (BGZF, raw
   sequential, multi-member).
5. **GPU perf** (when `feature = "gpu-decode"` enabled): ≥ 20 GB/s
   aggregate on M3 Max for inputs ≥ 1 GiB.
6. **Constant-time variant** decodes correctly with no
   data-dependent branches (verified via `valgrind --tool=
   helgrind` + static analysis); ≤ 25% slowdown vs optimized
   variant.
7. **Code surface reduction:** the deletes in §3 land in the same
   commit as the new decoder.
8. **Opus advisor sign-off** on the neurotic measurement.
9. **14-day soak** between neurotic gate passing and the merge.

No "ship if ≥ X" hedges. The target is "≥ rapidgzip"; we iterate
until we hit it.

---

## 6. Risks (with mitigations, not constraints)

1. **rustc + cranelift compile time.** Risk: rustc compilation of a
   ~4,500-line generic-heavy crate with cranelift dep takes minutes.
   Mitigation: cranelift is mature (used by Wasmtime); incremental
   builds are fast; cold build is paid once.

2. **JIT memory pressure under high block churn.** Risk: thousands
   of small blocks each emit ~200 lines of JIT code; memory grows
   without bound. Mitigation: JIT cache LRU eviction at 64 MiB cap;
   blocks below 4 KiB output route to interpreted fallback.

3. **`Creusot` is research-grade.** Risk: verification doesn't
   compile on every rustc version; ICEs possible. Mitigation: pin
   Creusot version; pin rustc nightly; track upstream stability.
   Worst case: ship without machine-checked proofs, with hand-written
   correctness proofs in markdown.

4. **GPU decode integration is OS-specific.** Risk: Metal works on
   macOS but the same code doesn't run on Linux+NVIDIA. Mitigation:
   each backend behind its own feature flag; GPU is opt-in.

5. **Hardware coprocessor detection at runtime.** Risk: false
   positives route to QAT then discover the kernel module isn't
   loaded. Mitigation: probe with a small test decode at decoder
   construction; fall back if the probe fails.

6. **Two-pass decode pays Pass 1 cost on every chunk.** Risk: for
   large but fast-decoding chunks, Pass 1 overhead exceeds the
   saved yield-tax. Mitigation: heuristic — if chunk is "obviously
   large" (deflate > 1 MiB), skip Pass 1 and use an over-allocated
   buffer + final shrink. For smaller chunks, Pass 1's cost is
   amortized.

7. **Cross-arch SIMD width dispatch must be tested on every
   arch.** Risk: AVX-512 path works locally but fails on different
   AVX-512 generation (Skylake-X vs Ice Lake). Mitigation: CI
   matrix covering each generation; QEMU simulation for arches
   without hardware access.

8. **Self-modifying / JIT-emitted code interacts with security
   mitigations** (W^X, CFI). Risk: macOS Hardened Runtime blocks
   JIT pages without entitlement; Linux SELinux blocks `mprotect`
   to executable. Mitigation: use `pthread_jit_write_protect_np`
   on macOS; document the SELinux requirement; provide a
   `feature = "no-jit"` build that falls back to interpreted
   throughout.

---

## 7. Phasing (non-normative — does not shape the design)

The design lands in one commit. Implementation can phase how the
implementer prefers; nothing in §1-§6 depends on this section.

Suggested order:
1. Build the interpreted Rust fallback decoder (`Inflate` core).
   Drop-in replacement for `ResumableInflate2`. Validates correctness
   layers.
2. Add the cranelift-JIT path. Falls back to interpreted for blocks
   below threshold.
3. Add AVX-512 / NEON wide bitbuf paths.
4. Add per-block perfect-hash table builder.
5. Add two-pass exact-output-bound mode (`OwnedOutput`).
6. Add constant-time variant.
7. Add GPU offload.
8. Add hardware-coprocessor dispatch.
9. Add Creusot annotations to the interpreted path; verify; extend
   to JIT-emit function.
10. Migrate every caller; delete every legacy module; delete every
    FFI dep.

Some phases parallelize. The phasing does not affect the design.

---

## 8. Open questions the implementer must resolve

1. **JIT cache key.** Is `(litlen_lengths, dist_lengths)` the right
   key, or should it include block-size hint / arch profile? Affects
   cache hit rate.

2. **Perfect-hash construction algorithm.** CHD vs BBHash vs
   FrozenHashMap. CHD has lower lookup cost but higher build cost;
   the right choice depends on block size distribution.

3. **GPU decode of single-block streams.** Within a block Huffman is
   serial. Is there value in GPU offload for SINGLE-block streams
   (< 64 KiB output)? Probably no, but verify.

4. **Constant-time variant's bit-reader.** The bit-reader's
   `bitsleft.wrapping_sub(n)` for n > bitsleft produces a data-
   dependent state. Can it be made constant-time without becoming
   a full reset on every consume?

5. **eBPF integration mechanism.** USDT probes vs uprobe attachment
   vs a custom kernel module. Affects portability.

---

## 9. Provable correctness — non-negotiable requirements

Per the user's "provably correct" directive:

- The `Bits` state machine has a Creusot proof of `bitsleft ≤
  BITBUF_WIDTH` across all `refill`/`consume(n)` sequences.
- The hot loop has Creusot proofs of:
  - `out_pos < output.len()` at every write site
  - back-reference distance ≤ `history_bytes` at every match site
  - Huffman decode consumes ≤ 15 codeword bits + ≤ 13 extra bits
    per symbol
- The JIT-emit function has a Creusot proof that ANY emitted code
  satisfies the above given a well-formed BlockHeader.
- The marker-tail flip is a runtime check guarded by an assertion,
  paired with a Creusot proof that the assertion implies the post-
  flip clean-mode invariant.
- Constant-time variant has a static analysis pass (custom Cargo
  subcommand) that verifies no branch in the hot loop depends on
  decoded data — only on the bit-reader's fill state.
- Real-corpus differential test runs against three oracles
  (reference zlib, rapidgzip, Node.js CompressionStreams) on every
  CI build; ≥ 72-hour fuzz runs per release.

A bug class that escapes ALL of these is residual risk we accept.
Today's T3 bug was caught by real-corpus alone; the four-layer
story catches it in three of the four layers (real-corpus, prop-
tests with deep buffer-size shrinking, and the Creusot
`out_pos < output.len()` invariant). The fourth layer (three-oracle
fuzz) catches the next one.
