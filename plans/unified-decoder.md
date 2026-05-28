# Unified deflate decoder — design + roadmap

This document is the plan for closing the measured 22% perf gap
between pure-rust gzippy inflate and ISA-L FFI to within "a percentage
point or two" (user's goal). It's a roadmap with decision gates, not
an aspirational end-state design.

The end-state vision (constant-time variants, GPU offload, coprocessor
dispatch, Creusot proofs, eBPF probes, etc.) is preserved in
`plans/unified-decoder-stretch.md`. This document is the critical
path.

---

## 1. Measured baseline (2026-05-27)

Paired alternating bench, 12 trials, neurotic i7-13700T,
silesia-large.gz at T=16:

| Build | env | Median MB/s | Δ vs ISA-L |
|---|---|---|---|
| `--features isal-compression` | — | ~1050 | (baseline) |
| `--features pure-rust-inflate` (libdeflate-inner) | default | ~820 | -22% |
| `--features pure-rust-inflate` | `GZIPPY_RESUMABLE_ISAL_INNER=1` | ~770 | -27% |

**Compile flags did not move the gap.** Measured: `LTO=fat`,
`codegen-units=1`, `RUSTFLAGS=-C target-cpu=native` — no change vs
default release. PGO has NOT been tried; it's a Week-1 experiment
(§3.1).

**Eight session-length optimization attempts** all either produced
no improvement or regressed on the experimental ISA-L LUT inner
path. The libdeflate-inner production path was not moved by any of
them. Falsification record at end of §3.

**What this means.** The 22pp gap was not closed by single-lever
inner-loop tuning. Closing it likely requires either (a) codegen-
level change (PGO, per-block specialization), or (b) wholesale
algorithmic substitution (vendor a known-good Rust port like
libdeflate-rs). We do not yet know which; §3.1 finds out cheaply
in week 1.

### What this session DID land (real wins regardless of next step)

- **RFC 1951 §3.2.6 reserved-symbol fix** for litlen 286/287 and
  dist 30/31 in fixed-Huffman canonical code construction. Without
  this, every 9-bit literal decoded with +4 byte offset on any
  fixed-Huffman block.
- **Two env-flag-gated experimental paths** (`GZIPPY_ISAL_PURE_BULK`,
  `GZIPPY_RESUMABLE_ISAL_INNER`), both silesia byte-perfect,
  default OFF, production unchanged. Detail in commit history.

---

## 2. Goal

**Near-term:** pure-rust inflate within 1-2pp of ISA-L FFI on
neurotic silesia-large at T=16, paired-bench median ≥ 12 trials.

**Escape hatch:** if after exhausting §3, the measured gap is ≤5pp
across silesia AND linux-source AND a web-archive sample, we declare
victory and ship. Pushing from 5pp to 1pp may require multi-quarter
work that's not commensurate with the marginal user-visible win.

**Long-term (CLAUDE.md prime directive):** beat ISA-L/libdeflate.
Not a Phase-4-or-earlier goal; documented as the post-roadmap
horizon if measurements warrant.

---

## 3. The roadmap

### 3.0 Per-phase methodology (binding)

1. **Cite evidence.** Reference the §3.1.3 perf attribution that
   this phase addresses. No attribution → don't do the phase.
2. **Projection ≠ commitment.** Each phase has a per-phase
   projection range and a kill criterion. No cumulative projections
   across phases — re-baseline after each phase lands.
3. **Kill criterion.** Defined per phase; if measured delta <50% of
   projection AND correctness gates green, stop and re-diagnose. Do
   not paper over with the next phase.
4. **Correctness gate FIRST.** Real-corpus differential test in the
   SAME commit as the implementation. Silesia byte-perfect is the
   floor.
5. **Paired-bench on neurotic.** Single-run results on shared
   hardware are misleading. Use `make ship` or the paired-alternating
   loop from commit `dae8bd2`.
6. **Opus advisor sign-off** before claiming the phase done.
   `feedback_advisor_consult` enforces this.

### 3.1 Week 1: decision gates

Four cheap experiments. Ordered by scope-killing power: experiments
that could collapse later phases run first. **Do not skip; do not
parallelize without sequencing**, because the result of each gates
the framing of the next.

#### 3.1.1 Day 1: count distinct Huffman fingerprints in real corpora

A `(litlen_lengths, dist_lengths)` fingerprint is the key Phase 4
(AOT specialization) would compile for. If silesia has 1,000+
distinct fingerprints, specializing 256 of them is hit-rate
gambling; AOT is dead before it starts.

Implementation: half-day script. Parse silesia, linux-source, a
web-archive sample (HTTP gzip captures). Count distinct fingerprints
per corpus. Compute frequency distribution.

**Gate:** if top-256 fingerprints cover ≥80% of blocks in all three
corpora, AOT is viable. If <40%, AOT is dead (Phase 4 deleted).
40-80% → maybe; revisit after Phase 3.

#### 3.1.2 Day 2-3: libdeflate-rs spike

[libdeflate-rs](https://crates.io/crates/libdeflate-rs) is a Rust
port of libdeflate's C inflate. Bench it on neurotic silesia AND on
synthetic random/btype01-heavy corpora.

**This is the single most important Week-1 experiment.** If
libdeflate-rs is already close to ISA-L, the whole JIT plan dies
and the project becomes "vendor + tune" — 1 week of work, not 6-9.

**Decision matrix:**
- Within 5pp of ISA-L → **vendor it, tune the binding, ship**.
  Phases 2-4 are deleted; project ends at Week 2.
- 5-20pp behind ISA-L → vendor it (it's a better starting point
  than current libdeflate-inner) + run Phase 2.
- ≥20pp behind ISA-L → confirms the gap is language-level codegen
  for this code shape; JIT/AOT case (Phases 3-4) is the right move.

#### 3.1.3 Day 3-5: perf attribution

`perf record + perf annotate` on both inflate hot loops on neurotic.

```bash
perf record -F 999 -e cycles --call-graph dwarf -- \
    ./gzippy-purerust -d -c -p 16 silesia-large.gz > /dev/null
perf annotate decode_huffman_body_resumable

# Same for ISA-L FFI:
perf record ... -- ./gzippy-isal -d ...
perf annotate decode_huffman_code_block_stateless_base
```

Attribute the 22pp gap to specific causes:
- N pp from bit-reader codegen (per-symbol bounds check)
- N pp from Huffman LUT lookup latency / cache misses
- N pp from per-iter overhead (yield checks, refill checks)
- N pp from instruction count / register pressure

**Gate:** every later phase MUST cite this attribution. If Phase
2's bit-reader rewrite is justified by "ISA-L uses signed-i32
rollback," that's recall, not evidence; the attribution either
shows ≥5pp lives in bit-reader stalls or it doesn't.

#### 3.1.4 Day 5: PGO spike

Budget 1 day, not 4 hours. PGO with `--no-default-features
--features pure-rust-inflate` requires that feature combination
to build end-to-end on Linux with matching `llvm-profdata`. Profile
collection from a 16-thread parallel binary collides; collect with
T=1 first, then bench with T=16.

```bash
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --no-default-features --features pure-rust-inflate
./target/release/gzippy -d -c -p 1 benchmark_data/silesia-large.gz > /dev/null
llvm-profdata merge -o /tmp/pgo.profdata /tmp/pgo-data
RUSTFLAGS="-Cprofile-use=/tmp/pgo.profdata" \
    cargo build --release --no-default-features --features pure-rust-inflate
```

Sequence PGO AFTER perf attribution so we know where the profile-
guided wins are landing in the binary.

**Gate:** if PGO closes 10+pp, re-baseline and reassess remaining
gap. Smaller wins still kept; flow through to next phase with the
new baseline.

#### 3.1.5 Decision matrix after Week 1

Read top-down; first matching row wins. Prefer the path with fewer
new deps when results tie.

| PGO | libdeflate-rs | fingerprint count | Next |
|---|---|---|---|
| ≥10pp AND total within 5pp | any | any | **Ship PGO-only; project ends.** No new deps. |
| <ship | within 5pp | any | Vendor libdeflate-rs; ship; project ends |
| <ship | 5-20pp behind | any | Vendor libdeflate-rs + Phase 2 |
| <ship | ≥20pp behind | ≥80% top-256 | Phase 2 + Phase 3; revisit Phase 4 after Phase 3 |
| <ship | ≥20pp behind | <40% top-256 | Phase 2 + Phase 3; Phase 4 dead |
| <ship | ≥20pp behind | 40-80% top-256 | Phase 2 + Phase 3; defer Phase 4 decision to post-Phase-3 attribution |

**Re-measurement gate between Phase 2 and Phase 3:** after Phase 2
lands, re-run the paired-bench. If remaining gap to ISA-L is <5pp,
ship per §2 escape hatch — do NOT enter Phase 3 just because the
matrix listed it.

### 3.2 Phase 2: bit-reader rewrite (1-2 weeks)

**Trigger:** §3.1.3 attribution shows ≥5pp lives in bit-reader
codegen. If attribution shows <5pp here, **skip to Phase 3 directly**.

**Premise (to be verified by §3.1.3, not asserted):** the per-symbol
bounds check in our unsigned-u32 `bitsleft` is more expensive than
ISA-L's signed-i32 optimistic-consume + per-block rollback check.

**Implementation:**
- `consume_first_decode::Bits::bitsleft` from `u32` to `i32`.
- `consume(n)` is signed subtraction; positive = OK, negative =
  underflow.
- Bounds check moves from per-`consume` to per-block.
- Callers updated: `Bits`, `IsalLitLenCodePure::decode`,
  `HuffmanCodingReversedBitsCached::decode`,
  `decode_huffman_body_resumable` (libdeflate path),
  `decode_huffman_body_resumable_isal`, `isal_lut_bulk::decode_block`.

**Correctness gates:** all 783 lib tests + silesia byte-perfect +
`resumable_matches_oracles_across_buffer_sizes` + new property test
for `consume`/`refill` interaction at underflow boundaries.

**Projection:** 3-10pp (range from advisor recall; pinned only by
§3.1.3).

**Kill criterion:** measured <2pp after correct impl → abandon; the
bit-reader pattern wasn't the bottleneck the attribution suggested.

### 3.3 Phase 3: per-block hand-written x86_64 assembly via dynasm-rs (3-4 weeks)

**Trigger:** Week 1 decision matrix points here AND Phase 2 didn't
close to within 5pp.

**Be honest about what this is.** dynasm-rs is not a JIT compiler in
the cranelift sense. It's an assembler macro you author by hand for
each instruction sequence. "Per-block JIT" here means: at the point
we've parsed a dynamic-Huffman block header, we emit a sequence of
x86_64 instructions tailored to that block's specific code lengths,
into a fresh executable page, then call into it. The labor is
hand-written x86_64 assembly with parametric branches keyed on the
block's parsed structure.

**This is the labor budget that ISA-L pays.** ISA-L itself is C
+ AT&T-syntax asm; its inflate hot loop is C, but the surrounding
match-copy, CRC, and decompress wrappers are asm. Closing the gap
in pure Rust requires similar labor.

**Implementation:**
- `dynasm` + `dynasm-runtime` crate deps. **x86_64 only for v1.**
  aarch64 + scalar fallback land as separate later phases — do not
  attempt parallel arch development.
- Per-block emit: parse code lengths, emit a specialized decode loop
  that hardcodes the (litlen, dist) table for this fingerprint.
  Branch-free fixed-Huffman variant.
- JIT cache: ring of executable pages, `munmap` LRU eviction at
  configurable cap (default 64 MiB).
- `pthread_jit_write_protect_np` for macOS Hardened Runtime.
- `feature = "no-jit"` falls back to the current interpreted loop.

**Correctness gates:** same as Phase 2, plus three-oracle
differential fuzz (flate2 + `libz-sys` reference zlib +
`libdeflate-sys` in fuzz workspace) ≥72h per release.

**Projection:** unknown. This is the only known path to close a
language-level codegen gap; justification is elimination, not
estimate.

**Kill criterion:** measured ≤2pp gain after working JIT → the
specialization isn't load-bearing for this corpus. Investigate
before Phase 4.

### 3.4 Phase 4: AOT specialization (2-3 weeks)

**Trigger:** Phase 3 lands AND §3.1.1 fingerprint count showed ≥80%
top-256 coverage AND a fresh perf attribution (post-Phase-3) shows
JIT-emit cost is ≥3% of total decode time. Run that attribution
explicitly as Phase 4 entry; don't trust the §3.1.3 baseline.

**Implementation:**
- `profiles/aot-decoder-fingerprints.json` checked into repo,
  generated by `gzippy profile collect <corpus>`.
- `build.rs` reads JSON, emits `target/.../aot_decoders.rs` with
  `extern "C"` per-fingerprint decoder functions.
- First-seen fingerprints at runtime → dynasm JIT.
- Truly cold fingerprints → interpreted Rust loop.

**Projection:** 2-5pp on corpora matching the AOT profile; 0 on
mismatched.

**Kill criterion:** AOT fingerprint hit rate <40% in production
→ AOT pays a build-time tax (~30s cold rustc) without enough hits
to justify. Don't ship.

### 3.5 Explicitly NOT in this roadmap

- **SIMD match-copy (AVX2 16-byte chunks)**: silesia avg match ~10
  bytes makes AVX2 mostly wasted; RLE overlap (`distance < length`)
  is a known correctness trap. Stretch goal after Phase 3 lands
  AND if perf attribution shows ≥5pp in copy_match.
- **aarch64 dynasm port**: x86_64 first. aarch64 is a separate
  multi-week phase, not parallel work.
- **Compression replacement**: a different decoder is in scope for
  this document. Compression is OUT of scope; rewriting it is its
  own multi-month project.
- All `unified-decoder-stretch.md` items (constant-time variant, GPU
  offload, hardware coprocessors, eBPF probes, Creusot proofs,
  forensic recovery, no_std/wasm, async API, &dyn Read adapter).
  These are valid end-state goals and don't block the 22pp gap.

### 3.6 Falsification record from 2026-05-27 (do not re-attempt without addressing the specific failure)

The "hypothesized cause" column is interpretation, not measurement.
Future attempts should validate the hypothesis (via perf counters
or codegen inspection) before retrying.

| Attempt | Measured | Hypothesized cause |
|---|---|---|
| B2 PRELOAD naive (lookup at iter end) | -10% | Wasted lookups on yield/EOB; preload at wrong pipeline point |
| 2-unroll FASTLOOP body | -10% | Icache pressure from doubled (100-line) macro body |
| FASTLOOP-scoped PRELOAD with cached entry | no change | OOO engine on Raptor Lake may already overlap lookup with prior iter's writes; not validated via stall counters |
| Single-literal fast path inside FASTLOOP | no change | Compiler likely already branch-predicts the common case; not validated |
| Writeback-skip around copy_match_windowed | -9pp | Borrow-checker pessimism plausibly caused spills; not validated via codegen diff |
| LTO=fat + codegen-units=1 + native | no change | Compile flags alone insufficient |
| `target-cpu=native` alone | no change | Same |

---

## 4. Near-term done-when

ALL of:

1. Pure-rust inflate within 1-2pp of ISA-L FFI on neurotic silesia-
   large at T=16, paired-bench median over ≥12 trials.
   **Escape hatch:** ≤5pp counts if §3 was fully exhausted.
2. All 783+ lib tests pass.
3. Three-oracle differential fuzz (flate2 + libz-sys + libdeflate-sys)
   runs ≥72h with zero disagreements on the next release.
4. Opus advisor sign-off on the neurotic measurement.

**Not in near-term done-when** (preserved as stretch goals in
`unified-decoder-stretch.md`): Creusot proofs, GPU offload,
constant-time variant, hardware coprocessor support, eBPF probes,
JIT entitlements packaging.

---

## 5. The architecture (only the parts on the critical path)

This section specifies the architectural decisions for Phases 2-4.
Stretch architecture (perfect-hash tables, AVX-512 wide bitbuf,
mmap direct decode, etc.) lives in `unified-decoder-stretch.md`.

### 5.1 The dynasm-rs hand-asm + AOT model (Phase 3-4)

dynasm-rs is an assembler macro; we author the per-fingerprint decode
loops in x86_64 assembly via the macro. Build-time AOT emits the
top-N fingerprints from `profiles/aot-decoder-fingerprints.json`;
runtime dynasm handles first-seen fingerprints; interpreted Rust
handles cold fallback.

```rust
enum BlockDecoder {
    Aot(extern "C" fn(...)),       // build.rs pre-compiled
    Jit(DynasmEmittedCode),        // first-seen + cached
    Interpreted(InterpreterState), // fallback
}
```

JIT cache: ring of executable pages, LRU eviction at 64 MiB cap.
`pthread_jit_write_protect_np` for macOS.

### 5.2 Bit-reader (Phase 2)

Signed `i32` `bitsleft`. Per-block bounds check, not per-symbol.
Implementation details in §3.2.

### 5.3 Output handling (current production)

Today the parallel-SM path uses amortized-growth `Vec<u8>` with
per-chunk capacity hints. **Note:** the "read ISIZE from gzip
trailer and pre-allocate exact-sized output buffer" pattern from
prior plan revisions does NOT help the parallel-SM path (ISIZE is
total uncompressed size, not per-chunk). It would help only a
sequential single-member CLI path, which is orthogonal to the 22pp
gap this roadmap addresses. Out of scope here; tracked in
`unified-decoder-stretch.md`.

### 5.4 Hardware target

Production target hardware: **x86_64 AVX2 + BMI2** (the i7-13700T
neurotic uses for benches; Raptor Lake P-cores lack AVX-512 by
Intel's segmentation). AVX-512 is documented in stretch only.

---

## 6. Risks

1. **PGO regresses some workloads even if it helps silesia.** PGO
   profile must cover all corpus types before ship. >5pp regression
   on any tracked corpus → ship without PGO; fall back to Phase 2-4.

2. **libdeflate-rs is unmaintained / has correctness bugs.** Spike
   includes a silesia byte-perfect check + 1M-iteration fuzz vs
   flate2. If correctness is shaky, vendor + fix or abandon
   libdeflate-rs option.

3. **dynasm-rs introduces a substantial dependency.** Proc-macro +
   runtime page management. Worst case: vendor it.

4. **AOT compile-time tax.** rustc compiles 200KB of generated
   decoders. Budget ≤30s cold add per CI build; if exceeded, cut
   AOT count from 256 to 128.

5. **JIT memory pressure under high block churn.** LRU page ring
   eviction at 64 MiB. Truly cold fingerprints → interpreted.

6. **`feature = "no-jit"` builds get NO Phase 3-4 benefit.** They
   stay at Phase 2 perf. Document this explicitly; it's the price
   of supporting environments that prohibit dynamic codegen.

7. **The 22pp gap might be unreachable by ANY combination of
   Phases 2-4.** If after Phase 4 we're still at ≥5pp, ship per the
   escape hatch in §4 and re-scope. Don't extend into Phase 5+ that
   doesn't exist in this doc.

---

## 7. Open questions (load-bearing — owned by Week-1 tasks)

These must be answered BEFORE Phase 3 starts. Each has a specific
Week-1 deliverable that answers it; they're not "future implementer
decides."

1. **JIT cache key.** `(litlen_lengths, dist_lengths)` alone, or
   does cache hit rate improve with block-size hint included?
   **Answered by:** §3.1.1 fingerprint-count script — extend it
   to compare hit rates with vs without block-size as part of
   the key.

2. **Perfect-hash construction algorithm** (only if perfect-hash
   is promoted from stretch to main per §3.4). CHD vs BBHash vs
   FrozenHashMap. CHD: ~3 cycles per lookup, ~5 µs/block build.
   **Answered by:** half-day microbench on three candidate impls
   against silesia block-size distribution from §3.1.1.

---

## 8. Phase 5: cleanup commit (after one release-cycle soak)

The two-commit rollout shape: commit 1 lands the new decoder ON by
default with `GZIPPY_LEGACY_INFLATE=1` env-var rollback escape;
commit 2 deletes the legacy paths.

**Soak gate:** 14 days production traffic with zero rollbacks
attributed to the new decoder AND no field bug reports tied to it.

**Files deleted (~10,000+ lines):**
```
src/decompress/parallel/deflate_block.rs            ~2,200
src/decompress/parallel/huffman_*.rs (5 files)      ~1,800
src/decompress/parallel/rfc_tables.rs
src/decompress/parallel/isal_huffman.rs
src/decompress/inflate/resumable.rs                 ~2,346
src/decompress/inflate/libdeflate_decode.rs         ~1,226
src/decompress/inflate/libdeflate_entry.rs          ~1,019
src/decompress/inflate/consume_first_decode.rs      ~3,614
src/decompress/inflate/specialized_decode.rs          ~602
src/decompress/inflate/consume_first_table.rs         ~898
src/decompress/inflate/jit_decode.rs                  ~527
src/decompress/inflate/two_level_table.rs
src/decompress/inflate/vector_huffman.rs              ~966
src/decompress/inflate/double_literal.rs              ~289
src/decompress/inflate/bmi2.rs                        ~168
src/decompress/parallel/inflate_wrapper.rs (IsalInflateWrapper)
vendor/isa-l/ + vendor/isal-rs/ (submodules)
packaging/isal-patches/
src/backends/isal_decompress.rs, src/backends/isal.rs
src/backends/libdeflate.rs
Cargo.toml: isal-rs, libdeflater, libdeflate-sys deps;
            isal-compression, isal, pure-rust-inflate features
```

**Conditional:** if the Week-1 matrix routed to "vendor libdeflate-rs
+ ship," the deletion list is much smaller (libdeflater +
libdeflate-sys + isal-rs decode functions only; the libdeflate-inner
and ISA-L LUT inner paths stay). Rewrite this section against the
actual matrix outcome before commit 2.

---

## 9. Anchor commit

This document's commit (`HEAD`). File:line citations in §3.6
falsification table and §1 "what landed" resolve here.

The stretch document (`plans/unified-decoder-stretch.md`) carries
the preserved end-state architecture (constant-time, GPU, Creusot,
coprocessor, eBPF, perfect-hash, etc.) and the long-horizon goals.
