# Unified deflate decoder — stretch goals

**⚠️ EXPLICITLY IGNORABLE FOR CURRENT PLANNING (as of 2026-05-27).**

`plans/unified-decoder.md` was rewritten under the "infinite labor /
infinite time" lens; items previously deferred here (constant-time,
GPU, perfect-hash, async API, sub-crate publish, mmap, three-oracle
fuzz, etc.) are now in §3 of the main plan as parallel workstreams.

What remains in THIS document is what stays out of the main plan
even under infinite labor: things that don't touch the inflate hot
path AND don't pay off for the perf goal AND aren't required for
correctness (Creusot proofs, hardware coprocessor dispatch, eBPF
probes, forensic recovery API, coroutine suspend/resume).

Do not consult this document while planning current work. Its
content is preserved for archival reference only.

---

(Below: original prior content, preserved verbatim.)

This document captures architectural goals that were intentionally
removed from `plans/unified-decoder.md` to keep that roadmap focused
on closing the measured 22pp perf gap to ISA-L FFI. These items
are valid end-state targets per CLAUDE.md prime directive ("fastest
gzip ever") but **not on the critical path** for the 1-2pp goal.

Each item here was in the prior v6 of unified-decoder.md and was
moved out per advisor review 2026-05-27. Promote any item back to
the main roadmap only when:

1. The 22pp gap is closed per `unified-decoder.md` §4.
2. The item has a measured perf or correctness justification.
3. The item has a per-phase kill criterion analogous to the main
   roadmap's §3.0.

---

## Stretch architecture (prior unified-decoder.md §2.7-§2.22)

### Constant-time decode variant

For security-sensitive uses (decoding adversarial input where timing
leaks are part of the attack surface). Replaces every data-dependent
branch with branchless select (`cmov`-style). ~20% slower than the
optimized variant. Used by callers that opt in via feature flag.

### GPU offload

For sufficiently large inputs (compressed > 1 GiB AND compressed/
avg-block-size estimated ≥ 1000 blocks via header-density heuristic
— no Pass 1 scan), the decoder offloads to GPU via Metal (macOS) /
CUDA (Linux+NVIDIA) / Vulkan compute (portable). Each block's decode
is independent.

Honest throughput numbers: M3 Max realistic peak ~20 GB/s (per
prior advisor's correction of v1's hand-wavy 30 GB/s). Done-when
target ≥ 15 GB/s.

### Memory-mapped direct decode

For files decoded to disk (the CLI's main path), output goes directly
into an mmap'd file. The decoder writes to mapped memory; the kernel
handles writeback. Zero copy, zero intermediate Vec.

Input similarly mmap'd. For compressed input on the order of GBs, the
decoder operates entirely on memory-mapped regions; the in-RAM working
set is just the per-block JIT code + the chunk's tail window.

### Cache-line aware table layout (alignment only)

Tables `#[repr(align(64))]` allocated from a per-block arena.
Adjacent blocks' tables share a single 4 KiB page so the TLB entry
is hot. No explicit cache eviction (cache-coloring requires
undocumented set-index hash on Apple Silicon).

### Custom Huffman LUTs for known patterns via warm cache

For workloads where the input distribution recurs (HTTP gzip streams,
gzipped JSON traffic, code-repo archives), the decoder's JIT cache
keyed on `(litlen_lengths, dist_lengths)` produces warm hits on
subsequent blocks with identical tables.

Optional offline-profile import (for power users): a companion tool
decodes a representative corpus and emits a JSON of the top-N table
fingerprints. The CLI loads this at startup and pre-populates the
JIT cache. (This is the same mechanism as the main roadmap's Phase 4
AOT — promote to main when Phase 4 lands.)

### Full Creusot verification of the hot loop

Today: `kani` on the `Bits` state machine only. Stretch ideal:
`Creusot` (or equivalent: `Prusti`, `RefinedRust`) verifies the
entire JIT-emitted hot loop's correctness via ghost-variable
invariants:
- Output buffer never written past its bound
- Every match copy's source is within the valid history window
- Every Huffman decode consumes ≤ 15 bits + extra bits
- Bit-reader's `bitsleft` invariant holds across every state
  transition

The JIT-emit function itself is verified to produce code that
satisfies these invariants given a well-formed BlockHeader.

**Why this is stretch, not roadmap:** Creusot is research-grade.
No production gzip implementation has Creusot proofs. Pinning the
project's done-when on it means either we don't ship or we delete
the requirement in month 3 looking sheepish.

### Four-layer correctness with three independent oracles

Today: differential vs flate2 + libdeflate.

Stretch ideal: three independent oracles, all linked in-process via
Rust C-FFI bindings, for differential fuzzing at native fuzz
throughput (~1M cases/sec):

- **Reference zlib** (Mark Adler's reference, ~6 kLOC C) via
  `libz-sys` crate. In-process.
- **rapidgzip** via a C-ABI shim. Static lib via `cc::Build` in
  `build.rs`. Rapidgzip stays in `vendor/` as a permanent test dep.
- **libdeflate** (kept as a fuzz oracle even after the production
  dep is deleted; `libdeflate-sys` exists only in the fuzz workspace).

Any two-way disagreement is a bug. Three-way agreement is high
confidence.

### `no_std` + WebAssembly support

The core decoder is `no_std` + `alloc`-only. WebAssembly target
compiles cleanly. Embedded targets (Cortex-M, RISC-V) get a scalar
fallback. The Rust crate publishes a `gzippy-inflate` sub-crate
(decoder only, no_std) suitable for embedding.

### Async-friendly API

Native `async` decoder for use in tokio/async-std contexts.
`AsyncInflate::decode(AsyncRead).await -> Stream<Bytes>` yields
chunks of decoded output.

### Hardware bitstream coprocessor support

Detect Intel QAT (QuickAssist Technology — gzip compression/
decompression in dedicated ASIC) and ARM CryptoCell (some
configurations include DEFLATE acceleration). Route to the hardware
when present and beneficial. Falls back to JIT'd CPU path otherwise.

### eBPF / Tracy / DTrace first-class hooks

Every block-decode, every chunk-completion, every window-publish
emits a structured event to a lock-free SPSC ring buffer. Consumers
(Tracy, eBPF, DTrace probes) attach to the ring buffer. Zero cost
when no consumer attached.

USDT probes (Linux) and DTrace static probes (macOS) shipped with
the binary; eBPF programs can attach without rebuilding gzippy.

### Decoder as a separately-publishable Rust library

The decoder ships as a standalone `gzippy-inflate` crate with stable
public API:

```rust
pub struct Inflate;

impl Inflate {
    pub fn decode_gzip(input: &[u8]) -> Result<Vec<u8>, Error>;
    pub fn decode_deflate_into(input: &[u8], output: &mut [u8])
        -> Result<usize, Error>;
    pub fn decode_stream(input: impl std::io::Read) -> impl std::io::Read;
    pub fn decode_async(input: impl AsyncRead) -> impl Stream<Item = Result<Vec<u8>>>;
    pub fn builder() -> InflateBuilder;
}
```

`no_std` + `alloc` core for `decode_deflate_into` (caller-supplied
buffer, no allocator needed). Independent versioning; semver-stable.

### Decompress + recompress single-pass pipeline

For storage-class transitions (gzip → zstd, gzip-L1 → gzip-L9), the
unified decoder + encoder can pipeline:
`Inflate.decode(input).pipe(Deflate.encode()).collect()` runs both
in one streaming pass, sharing the same JIT-emit machinery and
allocator. Zero intermediate buffer.

### Forensic reproducibility hash

For crash-recovery / corrupt-input forensics, the decoder emits a
CRC32 of its internal state at every block boundary. If a decode
fails mid-stream, the partial output + state-hash chain lets a
recovery tool replay the decode from the last good state, skipping
the corrupt region. Zero cost on the happy path (CRC32 already
computed via `crc32fast::pclmulqdq`).

### Partial-recovery decode API

```rust
impl Inflate {
    pub fn try_recover_from(
        input: &[u8],
        last_good_state: ForensicState,
    ) -> Result<(Vec<u8>, RecoveryReport), Error>;
}
```

Given a corrupt stream and the last good state-hash, scan forward
for a block boundary, re-seed the window from
`last_good_state.window`, and resume decode. Returns recovered
output + a report of what was skipped.

### Coroutine-style decoder for cross-thread suspend

For parallel-SM patterns where a chunk's decode might be paused
mid-block (worker steal-back, priority shift), the decoder exposes
a `suspend()` checkpoint that serializes the entire decoder state
to a `DecodeContinuation` value. A different worker on a different
thread can `resume(DecodeContinuation)` and continue.

### Huffman primitives shared with zstd / brotli backends

The dynasm JIT + perfect-hash table builder + Bits state machine
are LZ77-codec-agnostic. The `gzippy-huffman-primitives` sub-crate
factors these out so future zstd / brotli decoders in pure Rust can
link them.

### Verified JIT cache memory bound (via dynasm page ownership)

The JIT cache is a ring of N executable pages (each 4 KiB on
Linux/macOS arm64, 16 KiB on macOS x86_64). When the ring is full,
the oldest page is unmapped via `munmap` and reused — no
fragmentation, no opaque allocator state, bound is exact and
provably 64 MiB (or whatever the configured N).

---

## Stretch perf goals (post-roadmap)

After `unified-decoder.md` §4's near-term done-when:

### Beat ISA-L FFI on representative workloads (CLAUDE.md prime directive)

Closing the gap to within 1-2pp is the near-term goal. Going *past*
ISA-L requires techniques ISA-L doesn't have:

- AOT specialization (Phase 4 of main roadmap) gives per-fingerprint
  branch-free decode that ISA-L's static C cannot match.
- Per-CPU dispatch (BMI2 PEXT on Haswell+, AVX-512 PEXT2 on Sapphire
  Rapids+) can outperform ISA-L's single-binary multibinary dispatch.
- Hand-tuned aarch64 NEON path beats ISA-L (which is x86_64-only).

None of these are required to hit the 1-2pp goal.

### Wide multi-literal with batch sizing derived from refill headroom

With a 512-bit shift register (AVX-512 hosts only — not neurotic
Raptor Lake), headroom = 64 bits ≈ 7 literals per refill. Per-batch
refill ladder writes 16 u8 bytes via `_mm_storeu_si128` (advances 7;
remaining 9 overwritten by next batch's start, safe if
FASTLOOP_OUTPUT_MARGIN ≥ 16).

In `<Markers>` mode: 7 u16 literals = 14 bytes; `_mm_storeu_si128`
covers it (writes 16, advances 14).

### Single-pass decode with exact bound from gzip trailer (sequential CLI only)

For sequential single-member CLI decode, read ISIZE from the gzip
trailer FIRST (one ~8-byte seek-to-end + read; ~1 µs total).
Allocate exact-sized output buffer. Forward-decode in a single pass.
Zero growth, zero yield-check tax.

**Does NOT apply to the parallel-SM path** (ISIZE is total
uncompressed size, not per-chunk). The parallel-SM path stays on
amortized-growth `Vec<u8>` per chunk; this is documented as a
sequential-only stretch optimization.

### Per-block perfect-hash decode tables

CHD-style perfect hash over the block's actual ~286 litlen codes.
Two-level hash; ~3 cycles per lookup; no subtable dispatch. Build
cost ~5 µs/block, table size ~2.3 KiB (fits L1).

Tiny blocks (< 256 output bytes) route to 10-bit canonical table.

Promote to main roadmap if Phase 3 perf attribution shows ≥3pp
in subtable dispatch.

### Runtime-widest shift register

Width chosen at decoder construction via runtime CPU detection:
- AVX-512: 512-bit register, refill 64 bytes
- AVX2 (production target on neurotic): 256-bit register, refill 32 bytes
- NEON (aarch64): 128-bit register, refill 16 bytes
- Scalar: 64-bit (today's path)

The main roadmap's Phase 3 dynasm work emits x86_64 AVX2-targeted
code; AVX-512 and NEON variants are stretch.

---

## C/C++ dependencies after the project

**Kept (for compression, not decode):**
- `flate2` (zlib-ng) — compression L6-L9
- `isal-rs` — compression L0-L3 on x86_64

**Deleted:**
- `libdeflater` / `libdeflate-sys` — replaced by the new decoder
- ISA-L decoder FFI through `isal-rs`'s decode functions — replaced
- All sequential / BGZF / multi-member callers route through new pure-Rust

**Out of scope for `unified-decoder.md`:** compression rewrite. The
prime directive ("fastest gzip ever") implies both directions
eventually become pure Rust, but the compression rewrite is its own
multi-month project tracked separately if/when scoped.
