# Pure-Rust Inflate Perf Roadmap

**Goal.** Close the measured 22% gap between pure-rust `gzippy`'s inflate
and ISA-L FFI on neurotic silesia-large.gz to within 1-2 percentage
points. This is the user's stated "within 1-2%" target.

**Anchor measurement** (paired alternating bench, 12 trials,
`silesia-large.gz` at T=16 on i7-13700T):

```
ISA-L FFI (C, GCC -O3 + asm shims):     ~1050 MB/s
pure-rust libdeflate-inner (prod):      ~820 MB/s    -22% from ISA-L
pure-rust ISA-L LUT inner (experiment): ~770 MB/s    -27% from ISA-L
```

LTO=fat + codegen-units=1 + RUSTFLAGS=`-C target-cpu=native` did not
move the gap. Eight inner-loop optimization attempts on the experimental
ISA-L LUT path (B1 bits-locals, T0 raw-pointer literal writes, T4
FASTLOOP yield-elide, B2 PRELOAD naive, 2-unroll FASTLOOP, FASTLOOP-
scoped PRELOAD, single-literal fast path, LTO+native+codegen-units=1)
all either produced no improvement or regressed. The gap is structural.

**Framing.** This roadmap is a series of deltas against
[`plans/unified-decoder.md`](unified-decoder.md), not a greenfield
plan. The unified-decoder document already specifies dynasm-rs JIT
(§2.1), AOT codegen from corpus profile (§2.1), per-block perfect-hash
tables (§2.2), and the rollout architecture (§3, §5). What's missing
from that plan is **how to sequence the work given today's measured
22pp gap** and **which cheap experiments to run first** that could
collapse the rest of the work.

This document specifies that sequencing.

---

## Week 1: decision gates (do NOT skip)

Two cheap experiments and one diagnostic step. Each can collapse the
rest of the plan if it succeeds, OR strengthen the case for the more
expensive phases if it doesn't. **Run all three before committing to
Phase 2+.**

### 1.1 PGO spike (4 hours)

Build the current pure-rust binary with profile-guided optimization:

```bash
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release \
    --no-default-features --features pure-rust-inflate
./target/release/gzippy -d -c -p 16 benchmark_data/silesia-large.gz > /dev/null
/usr/lib/llvm-*/bin/llvm-profdata merge -o /tmp/pgo.profdata /tmp/pgo-data
RUSTFLAGS="-Cprofile-use=/tmp/pgo.profdata" cargo build --release \
    --no-default-features --features pure-rust-inflate
```

Bench against ISA-L FFI on neurotic. **Gate:** if PGO closes 10+ pp,
the rest of the plan shrinks dramatically — re-evaluate phases 2-4.

PGO has never been tried on this codebase. The hot loop has many
branch sites (literal/length/EOB dispatch, sym_count cases, match-copy
overlap). PGO is exactly the optimization for this code shape.

### 1.2 libdeflate-rs spike (1 day)

[`libdeflate-rs`](https://crates.io/crates/libdeflate-rs) is a Rust
port of libdeflate's C inflate. Bench it on neurotic silesia.

**Gate:**
- If libdeflate-rs is within 5pp of ISA-L: the structural gap is
  pure-rust-specific, not algorithm-specific. The plan becomes
  "vendor libdeflate-rs and tune," not "build a JIT." Massive scope
  reduction.
- If libdeflate-rs is also ≥20pp behind ISA-L: the gap is genuinely
  in language-level codegen for the inflate shape, and the JIT case
  (Phase 3) is the right move.
- If libdeflate-rs is between (5-20pp behind): vendor it AND do
  Phase 2 (bit-reader rewrite); skip Phase 3+.

### 1.3 perf attribution (3 days)

Run `perf annotate` on both inflate hot loops on neurotic during a
silesia decode:

```bash
perf record -e cycles --call-graph dwarf -- \
    ./gzippy-purerust -d -c -p 16 silesia-large.gz > /dev/null
perf annotate decode_huffman_body_resumable

# Same for ISA-L FFI:
perf record ... -- ./gzippy-isal -d ...
perf annotate decode_huffman_code_block_stateless_base
```

Goal: attribute the 22pp gap to specific causes:
- N pp from bit-reader codegen (signed-i32 rollback vs unsigned-u32
  per-consume bounds check)
- N pp from Huffman LUT lookup latency / cache misses
- N pp from per-iter overhead (yield checks, refill checks)
- N pp from instruction count / register pressure

**Gate:** every later phase MUST cite the attribution it's addressing.
"X pp lives in bit-reader stalls" → Phase 2. "Y pp lives in dispatch
overhead per block" → Phase 3 (JIT specialization). No attribution
support → don't do the phase.

This is the discipline that was missing from this session's
optimization attempts. Memory `feedback_instrument_before_optimize`
documents that advisor priors have been 3-5× off on this codebase;
every phase needs measurement support.

### Decision matrix after Week 1

| PGO result    | libdeflate-rs result | Next phase                                    |
|---------------|----------------------|-----------------------------------------------|
| ≥10pp gain    | any                  | Re-baseline; reassess remaining gap           |
| <10pp gain    | within 5pp of ISA-L  | Vendor libdeflate-rs; Phase 2 only            |
| <10pp gain    | 5-20pp behind        | Vendor libdeflate-rs + Phase 2                |
| <10pp gain    | ≥20pp behind         | Phase 2 + Phase 3 + Phase 4                   |

---

## Phase 2: bit-reader rewrite (1-2 weeks)

**Trigger:** perf attribution shows ≥5pp lives in bit-reader codegen.

**Premise:** ISA-L uses signed `i32 read_in_length` where negative
means underflow. The decode optimistically consumes and checks the
sign ONCE PER BLOCK, not per symbol. My Rust uses unsigned `u32
bitsleft` with wrap detection — requires per-consume bounds check
because unsigned wraparound is silent corruption.

**Implementation:**

- Change `Bits::bitsleft` from `u32` to `i32`.
- Rewrite `consume(n)` to do signed subtraction; positive values mean
  "n bits still available," negative means underflow.
- Move the bounds check from per-`consume` to per-block (once at
  block entry + once at block exit).
- All callers updated: `consume_first_decode::Bits`,
  `IsalLitLenCodePure::decode`, `HuffmanCodingReversedBitsCached::decode`,
  `decode_huffman_body_resumable`, `decode_huffman_body_resumable_isal`,
  `isal_lut_bulk::decode_block`.

**Correctness gates:**
- All 783 lib tests pass.
- Silesia byte-perfect (`corpus_silesia_if_available`).
- `resumable_matches_oracles_across_buffer_sizes` (cross-checks vs
  flate2 and libdeflate at varying buffer sizes).
- `cross_chunk_resume_silesia_gzip9_chunk0_handoff` if the
  pre-existing failure is unrelated; otherwise document.

**Perf gate:** paired-bench shows the projected pp delta on neurotic.
If <50% of the perf-attribution-estimated delta, stop and re-diagnose.

**Kill criterion:** if measured delta is <2pp after correct
implementation, abandon — the unsigned-vs-signed difference wasn't
the bottleneck the attribution suggested.

---

## Phase 3: per-block dynasm-rs JIT (3-4 weeks)

**Trigger:** Week 1 decision matrix points here AND Phase 2 didn't
close to within 5pp.

**Direct port of `plans/unified-decoder.md` §2.1.** No greenfield
design — that document specifies the dynasm + AOT pipeline in detail.
This phase is the implementation:

- Add `dynasm` + `dynasm-runtime` crate dependencies.
- Per-block JIT-emit a specialized hot loop from the parsed dynamic-
  Huffman code lengths. Branch-free fixed-Huffman variant.
- JIT cache as a ring of executable pages (4 KiB on Linux/macOS arm64,
  16 KiB on macOS x86_64), `munmap`-based LRU eviction at 64 MiB cap.
- `pthread_jit_write_protect_np` on macOS for Hardened Runtime.
- `feature = "no-jit"` build flag falls back to the current
  interpreted loop for SELinux-restricted environments.

**Correctness gates** (same as Phase 2):
- All 783 lib tests.
- Silesia byte-perfect.
- Three-oracle differential (flate2 + libz-sys reference zlib +
  libdeflate-sys), 72-hour fuzz per release.

**Perf gate:** paired-bench shows ≥8pp gain over Phase 2 baseline.

**Kill criterion:** if measured <4pp gain after working JIT, the
specialization isn't the bottleneck. Investigate before Phase 4.

**Risks:**
- dynasm-rs is mature but adds proc-macro + runtime page management
  maintenance burden. `plans/unified-decoder.md` §6 risk 1 documents
  the bounded rustc compile-time impact.
- JIT memory pressure under high block churn. §6 risk 2 documents
  the LRU page ring mitigation.

---

## Phase 4: AOT specialization (2-3 weeks)

**Trigger:** Phase 3 lands; per-block JIT emit cost (~10 µs) shows
up in the perf profile as a meaningful fraction of total decode time.

**Direct port of `plans/unified-decoder.md` §2.1 AOT half.**

- `profiles/aot-decoder-fingerprints.json` checked into repo, generated
  by `gzippy profile collect <corpus>`.
- `build.rs` reads the JSON and emits `target/.../aot_decoders.rs`
  containing `extern "C"` per-fingerprint decoder functions (~256
  fingerprints × ~200 lines each ≤ 200 KiB; rustc compiles in < 30 s
  cold per §2.1's budget).
- First-seen fingerprints at runtime fall through to dynasm JIT.

**Perf gate:** ≥3pp gain over Phase 3 baseline on corpus matching the
AOT profile.

**Kill criterion:** AOT fingerprint hit rate <40% on the production
corpus → AOT doesn't pay off; the dynasm JIT cache already covers the
hot fingerprints. Don't ship AOT in this case.

---

## What's NOT in this roadmap (and why)

### SIMD match-copy (deferred per advisor 2026-05-27)

Silesia average match length is ~10 bytes. AVX2 16-byte chunks are
mostly wasted on this corpus. RLE overlap correctness (`distance <
length`) has bitten every implementation attempt. Realistic gain is
0-2pp with real regression risk. **Demote to stretch goal after Phase
3 lands; revisit only if perf attribution shows ≥5pp in copy_match.**

### Inline x86_64 assembly for the hot loop

Tracked separately as a portability concern for `feature = "no-jit"`
builds. Not a perf phase — JIT codegen specializes per-block, which
inline asm can't match because the asm is static.

### Reverting to libdeflate-style inner

If Week 1 libdeflate-rs spike shows that vendoring it closes most of
the gap, the experimental ISA-L LUT inner (`GZIPPY_RESUMABLE_ISAL_INNER`)
should be DELETED, not kept as a tier. The user's directive forbids
correctness-relevant tier boundaries.

---

## Per-phase methodology (binding for every phase)

Each phase MUST follow this sequence:

1. **Cite the attribution.** Reference the Week 1.3 perf-annotate
   result that this phase addresses.
2. **State the projected pp gain.** No cumulative projections across
   phases — re-baseline after each phase lands.
3. **Define the kill criterion.** If measured delta <50% of projection
   AND correctness gates green, stop and re-diagnose. Don't paper
   over with another phase.
4. **Land a correctness gate FIRST.** Real-corpus differential test
   in the same commit as the implementation (memory rule
   `feedback_real_corpus_test_with_lever`).
5. **Paired-bench on neurotic.** Use `make ship` infrastructure or
   the paired-alternating script from this session's transcript.
   Single-run bench results are misleading on shared hardware.
6. **Opus advisor sign-off** before claiming the phase done.
   `feedback_advisor_consult` enforces this.

## Falsified during 2026-05-27 session — do NOT re-attempt without addressing the specific failure mode

| Attempt | Result | Why it failed |
|---------|--------|---------------|
| B2 PRELOAD naive (lookup at iter end) | -10% | Wasted lookups on yield/EOB paths; preload was at wrong point in pipeline |
| 2-unroll FASTLOOP body | -10% | Icache pressure from 100-line macro body |
| FASTLOOP-scoped PRELOAD with cached entry | no change | OOO engine on Raptor Lake already overlaps lookup with prior iter's writes |
| Single-literal fast path inside FASTLOOP | no change | Compiler already branch-predicts the common case |
| Writeback-skip around copy_match_windowed | -9pp | Borrow-checker pessimism caused spills; libdeflate-inner already does this correctly |

If a future session wants to re-attempt any of these, it must FIRST
explain why the specific failure mode is now addressed.

## Won't-be-done scope

These are listed in `plans/unified-decoder.md` but are NOT in this
performance roadmap because they don't address the measured 22pp gap:

- Constant-time decode variant (§2.7) — security feature, ~20%
  SLOWER, not a perf phase.
- GPU offload (§2.8) — only triggers for inputs ≥ 1 GiB; silesia
  is 503 MB.
- Hardware coprocessor support (§2.16) — QAT not present on neurotic.
- eBPF/Tracy/DTrace hooks (§2.17) — observability, not perf.

These remain valid end-state goals; just not on the critical path
for closing the 22pp gap.
