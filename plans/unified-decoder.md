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
- **Rolls out behind a runtime flag.** New decoder ships ON by
  default but the legacy path is preserved one release-cycle for
  fallback. After the soak passes in production, the flag is
  deleted along with the legacy surface (~10,000 lines) in a
  follow-up commit. The "one commit" framing in the v1 plan was
  bookkeeping — a 14,500-line single commit is unreviewable. The
  honest shape: two commits, the second of which is a pure
  deletion gated on the first's soak. No new design risk; the
  legacy path is dead code from commit 1 onward.

---

## 2. The architectural pillars

### 2.1. AOT-codegen + dynasm hot-loop emitter (NOT cranelift)

The v1 plan proposed `cranelift-jit` per-block. The pass-3 critique
correctly identified that cranelift's `JITModule` doesn't expose
partial-arena reset, and its per-function compile cost (~100µs-1ms)
loses on blocks ≤ a few hundred µs decode work unless cache hit
rate is > 99% (a workload claim, not a design proof).

**Use dynasm-rs** (assembly DSL macro) for the JIT path. dynasm-rs
gives direct page ownership — we control allocation, can clear
arenas freely, and the per-emit cost is single-digit µs (it's a
macro that emits bytes; no SSA, no regalloc, no scheduler).

**Plus AOT-codegen** for the hot patterns: a `build.rs` script
takes a corpus profile (the offline-profile from §2.11) and emits
Rust code containing pre-compiled per-fingerprint decoders. New
fingerprints fall through to dynasm JIT; first decode of an
unknown fingerprint pays JIT cost (~10 µs), subsequent decodes
reuse cached code.

The interpreted Rust fallback (the ResumableInflate2-shape hot
loop with all B1-B6+T0-T5 wins) handles the cold path during JIT
emit and for inputs where JIT memory is exhausted.

```rust
enum BlockDecoder {
    Aot(extern "C" fn(...)),       // build.rs pre-compiled
    Jit(DynasmEmittedCode),        // first-seen + cached
    Interpreted(InterpreterState), // fallback
}

impl Inflate {
    fn decode_block(&mut self, header: &BlockHeader, ...) -> Result<...> {
        let fp = header.fingerprint();
        match self.codegen_cache.dispatch(fp) {
            BlockDecoder::Aot(f) | BlockDecoder::Jit(f) => unsafe { f(...) },
            BlockDecoder::Interpreted(s) => s.decode(...),
        }
    }
}
```

**Vendor citation:** dynasm-rs is mature (used by `wasmer`,
`b3`, others). AOT codegen from corpus profiles is a libdeflate
contemporary in spirit but not directly ported. CLAUDE.md
2026-05-27 (final) authorizes inner-loop innovation.

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

### 2.4. Single-pass decode with exact bound from gzip trailer + amortized growth fallback

The v1 plan's "two-pass scan-then-decode" had two flaws caught by
prior advisor passes: (a) Pass 1's "~2× faster" claim was wrong
because Pass 1 must still decode every Huffman symbol to know
match/literal classification, and (b) the cost model contradicted
itself across §2.4 and §6 risk 6. Single-pass is honest.

**FILE-level decode (CLI, BGZF, multi-member):** the gzip trailer
contains ISIZE (uncompressed size mod 2³²). Read trailer FIRST
(one ~8-byte seek-to-end + read; ~1 µs total). Allocate
exact-sized output buffer. Forward-decode in a single pass. Zero
growth, zero yield-check tax, zero waste. For files > 2³² bytes
the ISIZE is ambiguous; the decoder falls back to amortized growth
(below) AND verifies the final size matches ISIZE mod 2³².

**PARALLEL-SM CHUNK-level decode:** per-chunk output size isn't
recorded anywhere. The decoder uses amortized-growth via
`Vec<u8>`: standard doubling, O(1) amortized per byte, worst-case
2× over-allocation reclaimed via `shrink_to_fit()` at chunk
completion. Zero yield-check tax (the decoder writes via
`Vec::extend_from_slice` after each multi-literal batch; growth
amortizes against future writes).

**STREAMING decode** (writer-output for BGZF / sequential SM /
async): the decoder keeps the yield contract; output goes via
`Write::write_all` after each batch. Hot loop is identical to the
owned-output path; only the epilogue differs.

No "skip Pass 1 for large chunks" — there is no Pass 1. No "Pass 1
cost amortized" — there is no Pass 1. No internal cost-model
contradiction.

**DEFLATE max expansion is 1032×** (the prior advisor's S1
concern). For the file path this doesn't matter — ISIZE is exact.
For the chunk path the amortized-growth Vec handles arbitrary
expansion safely; the 2× worst-case over-allocation is bounded by
the actual decoded size, not by a heuristic expansion ratio.

### 2.5. Wide multi-literal with batch sizing derived from refill headroom

Today: T3 caps at 4 literals with packed u32 write.

**Per the §2.3 arithmetic, headroom-between-refills bounds the batch
size.** With a 512-bit buffer and ~448-bit refill threshold, headroom
= 64 bits = ~7 literals (at 9 bits each) per refill. The v1 plan's
"32 literals per refill" was wrong; that would require 288 bits of
headroom (32 × 9), which exceeds 64 bits by 4.5×.

Ideal: **per-batch refill ladder**:
- 7 literals at one refill: `_mm_storeu_si128` of 16 u8 bytes
  (writes 16, advances 7; remaining 9 are overwritten by the next
  batch's start, which is safe if FASTLOOP_OUTPUT_MARGIN ≥ 16).
- Need more? Refill mid-batch (the JIT emits this as a conditional
  branch optimized for "not taken" on text inputs).

For the AVX-512 64-byte store path: same shape, but using
`_mm512_storeu_si512`. Bigger SIMD doesn't get you more literals
per refill; it just makes the per-batch write cheaper.

**Adaptive predictor** (8-entry FIFO) tracks recent batch sizes so
the JIT-emit can choose between "always batch 7" vs "always batch 4"
specialization per block, eliminating per-iteration predictor cost.

In `<Markers>` mode: 7 u16 literals = 14 bytes; `_mm_storeu_si128`
covers it (writes 16, advances 14).

### 2.6. ONE table strategy: per-block perfect-hash

The advisor's prior critique flagged a per-block A/B between
perfect-hash and 128 KiB direct table as "runtime A/B, not design."
Picking one:

**Per-block perfect-hash (CHD) is the chosen strategy.** Reasons:
- Build cost ~5 µs/block (CHD over ~286 codes), amortizes at
  ~150 ns per decoded KiB — well below the per-symbol decode cost
  on any realistic block size.
- Table size ~2.3 KiB total — fits in L1 alongside the dist table
  and the JIT-emitted code page. Zero L1 pressure on neighbors.
- Zero subtable dispatch (the perfect hash always lands in one
  lookup).
- 128 KiB direct table has the build-cost amortization issue at
  small block sizes (advisor §B point 6: ~350 µs build for 32 KiB
  output ≈ 10 µs/KiB build tax). Perfect-hash is strictly better
  across the block-size distribution.

For BTYPE=01 (fixed Huffman): static perfect-hash precomputed at
compile time, baked into the JIT'd fixed-block decoder.

No two-strategy dispatch. No runtime A/B. One table shape per
block type.

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

### 2.8. GPU offload for bulk parallel decode (automatic when beneficial)

Today: parallel-SM uses CPU workers across chunks.

Ideal: for sufficiently large inputs (≥ 1 GiB AND ≥ 1000 blocks),
the decoder offloads to GPU via Metal (macOS) / CUDA (Linux+NVIDIA)
/ Vulkan compute (portable). Each block's decode is independent
after CPU's Pass 1 (symbol scan + block-boundary index), so
parallelism is per-block.

GPU decode is non-trivial because Huffman is serial within a block.
The approach:
- CPU Pass 1: scan symbols, build per-block (start_bit, output_len)
  index.
- GPU dispatch: N workgroups, each decoding one block sequentially
  via a single thread.
- Throughput scales with block count, not block size.

**Honest throughput numbers** (per advisor critique of v1's 30 GB/s
figure): single GPU thread at ~1 GHz, ~1 cycle/symbol Huffman ≈
~250 MB/s per thread. To reach 25 GB/s aggregate requires ~100
concurrent decoding blocks; for 30 GB/s requires ~120. On a 1 GiB
input split into 64 KiB blocks (~16k blocks), aggregate scales to
GPU's concurrent-workgroup limit. M3 Max: ~80 active workgroups
across 40 cores × 2-wide = realistic peak ~20 GB/s, NOT the
hand-wavy 30. Updated done-when target accordingly (§5 #5: ≥ 15
GB/s).

**Not behind a feature flag.** Automatic dispatch when input meets
the threshold AND a GPU is available. Falls back to CPU otherwise.
The opt-in via feature flag in v1 was effort-driven; ideal has
automatic dispatch.

### 2.9. Memory-mapped direct decode

Today: input is `&[u8]`, output is a `Vec<u8>`.

Ideal: for files decoded to disk (the CLI's main path), output goes
directly into an mmap'd file. The decoder writes to mapped memory;
the kernel handles writeback. Zero copy, zero intermediate Vec.

Input similarly mmap'd. For compressed input on the order of GBs,
the decoder operates entirely on memory-mapped regions; the in-RAM
working set is just the per-block JIT code + the chunk's tail
window.

### 2.10. Cache-line aware table layout (alignment only)

Today: tables are heap-allocated `Vec`, no alignment.

Ideal: tables are `#[repr(align(64))]` allocated from a per-block
arena. Adjacent blocks' tables share a single 4 KiB page so the TLB
entry is hot.

**No explicit cache eviction.** The prior plan proposed dummy-read
eviction; the advisor pointed out cache-coloring requires
undocumented set-index hash on Apple Silicon. We accept the
natural-LRU tax (~few cycles per block transition); blocks > ~1 KiB
output amortize it to noise. Cross-platform cache flush
(`_mm_clflushopt`, `dc cvac`) is not portable across the target
matrix (Darwin user-mode restrictions on aarch64); we don't use it.

### 2.11. Custom Huffman LUTs for known patterns (via warm cache)

For workloads where the input distribution recurs (HTTP gzip
streams, gzipped JSON traffic, code-repo archives), the decoder's
JIT cache (§2.1) keyed on `(litlen_lengths, dist_lengths)` produces
warm hits on subsequent blocks with identical tables. No magic
pattern detection — recurrence is detected by the cache key's
hash itself.

**Optional offline-profile import** (for power users): a
companion tool `gzippy-profile collect` decodes a representative
corpus and emits a JSON of the top-N table fingerprints. The CLI
loads this at startup and pre-populates the JIT cache. Zero
runtime detection cost; opt-in for users with workload-specific
inputs.

No circular pattern detection (per advisor critique of v1's
"hand-wavy CLI knows it's HTTP gzip"). The cache key IS the
fingerprint; recurrence is observable from the cache stats.

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

Ideal: three independent oracles, **all linked in-process via
Rust C-FFI bindings**, for differential fuzzing at native fuzz
throughput (~1M cases/sec, vs ~1k/sec for subprocess oracles —
the pass-3 critique correctly noted subprocesses kill throughput).

- **Reference zlib** (Mark Adler's reference, ~6 kLOC C) via
  `libz-sys` crate. In-process.
- **rapidgzip** via a thin C-ABI shim (`vendor/rapidgzip` is built
  as a static lib exposing `rapidgzip_inflate` for fuzz harness;
  the same shim is used by the bench harness). In-process.
- **libdeflate** (kept as a fuzz oracle even after the production
  dep is deleted; `libdeflate-sys` exists only in the fuzz
  workspace). In-process.

Any two-way disagreement is a bug. Three-way agreement is high
confidence. Fuzz runs are 72h per release, plus a 60-second
smoke per CI build (in-process throughput makes per-CI viable
where subprocess didn't).

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

USDT probes (Linux) and DTrace static probes (macOS) shipped with
the binary; eBPF programs can attach without rebuilding gzippy.

### 2.18. Decoder as a separately-publishable Rust library

The decoder ships as a standalone `gzippy-inflate` crate:
- `no_std` + `alloc` core (§2.14)
- Stable public API: `Inflate::new(input) -> InflateBuilder`,
  builder methods for arch profile / decode mode / output model
- Independent versioning; semver-stable
- Suitable for embedding in other Rust projects (proxies, HTTP
  servers, archive tools) that want pure-Rust inflate without
  the gzippy CLI

External consumers don't need to know about the JIT machinery; the
default builder picks the best monomorphisation for the running
machine.

### 2.19. Decompress + recompress single-pass pipeline

For storage-class transitions (e.g., gzip → zstd, gzip-L1 →
gzip-L9), the unified decoder + encoder can pipeline:
`Inflate.decode(input).pipe(Deflate.encode()).collect()` runs
both in one streaming pass, sharing the same JIT-emit machinery and
allocator. Zero intermediate buffer.

Implemented via `DecodeStream` and `EncodeStream` types that
implement `Stream<Item=Bytes>` (async) and `Iterator<Item=Bytes>`
(sync).

### 2.20. Forensic reproducibility hash

For crash-recovery / corrupt-input forensics, the decoder emits a
CRC32 of its internal state at every block boundary. If a decode
fails mid-stream, the partial output + state-hash chain lets a
recovery tool replay the decode from the last good state, skipping
the corrupt region.

Zero cost on the happy path (CRC32 already computed via
`crc32fast::pclmulqdq`).

### 2.21. `&dyn Read` adapter for non-mmap callers

The decoder's primary input is `&[u8]` (slice over mmap'd file).
For callers that can't mmap (network sockets, named pipes, stdin
from another process), `Inflate::from_read(reader: impl Read)`
provides a streaming adapter that buffers input as needed,
preserving the JIT-emit hot loop's `&[u8]` contract via an
internal staging buffer.

### 2.22. Verified JIT cache memory bound (via dynasm page ownership)

dynasm-rs owns the executable-page allocations directly (unlike
cranelift, which wraps its `Memory` opaquely). The JIT cache is a
ring of N executable pages (each 4 KiB on Linux/macOS arm64, 16 KiB
on macOS x86_64). When the ring is full, the oldest page is
unmapped via `munmap` and reused — no fragmentation, no opaque
allocator state, bound is exact and provably 64 MiB (or whatever
the configured N).

The pass-3 critique correctly noted that cranelift's StackArena
doesn't compose with partial reset. Switching to dynasm dissolves
this — page ownership is direct, and the ring-buffer-of-pages
allocator is a known-correct pattern (used by V8's code cache,
LuaJIT, Wasmtime's interpreter-side patcher).

---

## 3. What we delete (in the second commit, after soak)

**Two-commit shape** (consistent with §1):
- **Commit 1**: lands the new decoder ON by default. Legacy decoder
  preserved behind a `--features legacy-inflate` build-flag, OFF by
  default, present only as a one-release-cycle fallback for
  emergency rollback. Production CI runs the new decoder; the
  legacy build is verified via a single CI smoke job. Neurotic A/B
  gate runs against commit 1 before merge.
- **Commit 2**: ~10,000-line pure deletion of `legacy-inflate` +
  every file in the deletion list below. Lands one release cycle
  after commit 1 ships, conditional on a clean soak (no
  production rollbacks, no field bug reports tied to the new
  decoder). No design risk; commit 2 is mechanical.

The pass-3 critique correctly identified the §1-vs-§3 contradiction
where v3 said "two commits" in §1 but "lands and deletes in same
commit" in §3. This rewrite makes them consistent. The "one commit
of 14,500 lines" framing is dropped — it was unreviewable bookkeeping.

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

**Also in scope per the prime directive ("fastest gzip ever"):**
compression replacement. The unified decoder's JIT infrastructure +
perfect-hash table builder + Creusot-verified hot loop apply
symmetrically to the encoder (per-block Huffman table SELECTION is
the dual of decoding). A unified pure-Rust encoder is built on the
same JIT pipeline; replaces `flate2` (zlib-ng) and `isal-rs`
compression in the same project scope. Compression code lives in
`src/compress/unified.rs`; shares the JIT infrastructure with
`src/decompress/inflate/`. Total C-dependency surface after the
project: zero for both directions of the gzip codec.

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
   data-dependent branches (verified via `dudect` statistical
   timing-diff + `ctgrind`-style symbolic execution; per advisor:
   `helgrind` was the wrong tool — race detector, not timing
   oracle). Target: ≤ 5% slowdown vs optimized variant (was 25%;
   the gap closes with JIT-emit specializations that turn
   data-dependent branches into `cmov` sequences).
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

6. **Two-pass decode pays Pass 1 cost on every chunk.** Accepted.
   Pass 1 is mandatory; no escape hatch. The cost (~1.1-1.3× of a
   full decode pass for the scan, NOT 0.5× as the v1 plan
   misrepresented) is the price for exact-output bound and zero
   yield-tax in Pass 2. The total wall-time math is honest: 1×
   Pass 1 + ~0.7× Pass 2 (Pass 2 is FASTER than today's full decode
   because no yield checks, no growth, no bounds writes) ≈ 1.7× a
   theoretical one-pass with growth. The yield-tax we eliminate is
   ~5% per current bench; the bound-safety + memory-safety we gain
   is unboundedly valuable on adversarial input (1032× expansion
   ratios are RFC-legal).

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
