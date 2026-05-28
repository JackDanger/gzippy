# Unified deflate decoder — design + bake-off plan

**Framing for this revision.** Infinite development capacity, infinite
development time. The cost lens is removed; the **information lens is
not**. Cheap experiments still belong in the plan — not because they
save labor but because they produce information that changes what we
build. Measurement is a hard constraint, not a budget item.

This is the plan for closing the measured perf gap to ISA-L FFI to
within 1-2pp (CLAUDE.md prime directive: ultimately *beat* ISA-L and
libdeflate). With infinite labor we build competing routes in parallel
and pick the winner by measurement, rather than sequencing phases by
cost.

**Measured gap (corrected 2026-05-28)**: ~10pp (down from the 22pp
figure in earlier drafts of this plan — see §1.1 methodology gate
for the contamination explanation). The route sizing (§4) and the
risk #10 reference (§9) were written against 22pp; they have been
annotated, not rewritten — most of the architecture stays.

The "stretch" companion (Creusot proofs, eBPF telemetry, GPU
backends, etc.) was folded into §3.8-§3.13 here when it was
load-bearing, then deleted (`plans/unified-decoder-stretch.md`).
Recover from git history if needed: `git log --diff-filter=D --
plans/unified-decoder-stretch.md`.

---

## 1. Measured baseline

### 1.1 Methodology gate (CRITICAL — added 2026-05-28)

Before any bench claim, verify clean release profile:

```
grep -E "^strip" Cargo.toml | head -3
# Expect first line: strip = true
```

The 2026-05-27 baseline below was measured under a contaminated
Cargo.toml (debug-info enabled in release profile during a perf
symbolization experiment). That contamination accounts for an
~12pp overstatement of the gap. The 2026-05-28 row reflects clean
builds.

### 1.2 Baseline measurements

Paired alternating bench, neurotic i7-13700T, silesia-gzip9.gz at T=16:

| Date | Trials | Build | env | Median MB/s | Δ vs ISA-L |
|------|--------|-------|-----|-------------|------------|
| 2026-05-27 (contaminated) | 12 | `--features isal-compression` | — | ~1050 | (baseline) |
| 2026-05-27 (contaminated) | 12 | `--features pure-rust-inflate` (libdeflate-inner) | default | ~820 | -22% |
| 2026-05-27 (contaminated) | 12 | `--features pure-rust-inflate` | `GZIPPY_RESUMABLE_ISAL_INNER=1` | ~770 | -27% |
| **2026-05-28 (clean)** | **10** | `--features isal-compression` | — | **1212** | (baseline) |
| **2026-05-28 (clean)** | **10** | `--features pure-rust-inflate` (libdeflate-inner) | default | **1092** | **-10%** |
| **2026-05-28 (clean, post-T3)** | **20** | `--features pure-rust-inflate` (libdeflate-inner) | default | **904** | (different system load — noisy) |

**NOTE (insufficiently verified)**: the **2026-05-28 10pp row is
below §7's ≥12-trial floor**. Treat as directional; re-run at 20+
trials before treating "10%" as canonical for future planning.
The 20-trial T3 row's lower absolute throughput vs the 10-trial
row reflects CPU thermal/clock throttling between runs, not a
T3 regression — interleaved A/B within the 20-trial set showed
T3 at +1.9% vs the pre-T3 baseline.

### 1.3 Compile flags status

Compile flags (`LTO=fat`, `codegen-units=1`,
`RUSTFLAGS=-C target-cpu=native`) did not move the gap. Eleven
session-length lever attempts (7 from May 27 + 6 from May 28)
all either no-op'd or regressed — preserved in §6 falsification
record.

### What this session DID land (real wins regardless of next step)

- **RFC 1951 §3.2.6 reserved-symbol fix** for litlen 286/287 and
  dist 30/31 in fixed-Huffman canonical construction. Every 9-bit
  literal previously decoded with +4 byte offset on fixed-Huffman
  blocks.
- **`decode_block` API multi-block back-ref reach** fix.
- **Multi-pack LUT trailing element handling** (the trailing element
  of a 2-/3-pack can be a length code, not only literals).
- **Two env-flag-gated experimental paths** (`GZIPPY_ISAL_PURE_BULK`,
  `GZIPPY_RESUMABLE_ISAL_INNER`), both silesia byte-perfect, default
  OFF, production unchanged.

---

## 2. Goal

**Match ISA-L FFI within 1pp** on neurotic silesia-large at T=16,
paired-bench median ≥ 12 trials. No escape hatch — push to 1pp.

**Stretch (CLAUDE.md "fastest gzip ever"): beat ISA-L** on
representative workloads. Independent of the 1pp goal — not a
fallback. If the bake-off (§4) demonstrates that no combination of
Routes A-D reaches 1pp on this CPU, that's a failure mode, not a
graceful escalation; the team takes that result back to design,
not to a renamed goalpost.

Correctness floor: all 783+ lib tests pass; silesia byte-perfect;
three-oracle differential fuzz (flate2 + libz-sys reference zlib +
libdeflate-sys) ≥72h per release with zero disagreements.

---

## 3. End-state architecture

Single description of what the system looks like after all parallel
workstreams have landed. No phasing language here — see §5 for
the dependency graph that constrains build order.

### 3.1 Inner Huffman decode

**Per-block specialized code, three-tier dispatch:**

```rust
enum BlockDecoder {
    Aot(extern "C" fn(...)),       // build.rs pre-compiled (top-N fingerprints)
    Jit(DynasmEmittedCode),        // first-seen → dynasm emit (~10 µs)
    Interpreted(InterpreterState), // cold fallback / no-jit builds
}
```

- **dynasm-rs hand-written x86_64 assembly** (not "JIT compilation" in
  the cranelift sense — it's an assembler macro you author by hand
  per fingerprint). Per-block emit specializes a hot loop for the
  parsed (litlen, dist) tables. Branch-free fixed-Huffman variant.
- **AOT codegen for top-N fingerprints**. `profiles/aot-decoder-
  fingerprints.json` checked into repo, generated by
  `gzippy profile collect <corpus>`. `build.rs` emits
  `target/.../aot_decoders.rs` (~256 fingerprints × ~200 lines ≤
  200 KiB; rustc compiles in <30s cold).
- **aarch64 dynasm backend** built in parallel with x86_64, sharing
  the AOT fingerprint store and the runtime cache infrastructure.
- **Interpreted fallback** for `feature = "no-jit"` and for
  fingerprints not yet specialized.
- **JIT cache** as a ring of executable pages (4 KiB Linux/macOS
  aarch64, 16 KiB macOS x86_64), `munmap` LRU eviction at
  configurable cap (default 64 MiB). `pthread_jit_write_protect_np`
  for macOS Hardened Runtime.

### 3.2 Bit-reader

**Signed-i32 rollback pattern mirroring ISA-L**:

- `Bits::bitsleft` is `i32`. `consume(n)` is signed subtraction;
  negative means underflow.
- Bounds check moves from per-`consume` to per-block (block entry +
  block exit). Decode optimistically; roll back on negative `bitsleft`.

### 3.3 Output handling

- **Sequential CLI (single-member file):** read ISIZE from gzip
  trailer FIRST (~1 µs); allocate exact-sized output buffer;
  forward-decode single-pass. Zero growth, zero yield tax.
- **Parallel-SM chunk path:** amortized-growth `Vec<u8>` per chunk.
  ISIZE is total uncompressed size, not per-chunk — doesn't help
  here; documented and accepted.
- **Streaming (writer-output):** keeps the yield contract;
  `Write::write_all` after each batch.

### 3.4 Shift register width

Width chosen at decoder construction via runtime CPU detection:

| Target CPU | Width | Refill | Threshold |
|---|---|---|---|
| AVX-512 | 512 bits | `_mm512_loadu_si512` (64B) | 448 bits |
| AVX2+BMI2 (production target — neurotic Raptor Lake) | 256 bits | `_mm256_loadu_si256` (32B) | 224 bits |
| aarch64 NEON | 128 bits | `vld1q_u8` (16B) | 96 bits |
| Scalar fallback | 64 bits | u64 load | 48 bits |

Production target neurotic i7-13700T = AVX2+BMI2 (Raptor Lake P-cores
lack AVX-512 by Intel segmentation). AVX-512 path is built for hosts
that have it; not required for the 1pp goal.

### 3.5 Per-block perfect-hash decode tables

CHD-style perfect hash over the block's actual ~286 litlen codes.
~3 cycles per lookup, no subtable dispatch. Per-block build cost
~5 µs; table size ~2.3 KiB (fits L1 alongside dist table and
JIT-emitted code page).

Tiny blocks (< 256 output bytes) route to a 10-bit canonical table
where the perfect-hash build cost exceeds the per-symbol savings.

For BTYPE=01 (fixed Huffman): static perfect-hash precomputed at
compile time, baked into the JIT'd fixed-block decoder.

### 3.6 Match-copy

SIMD overlap-safe copy:
- 16-byte AVX2 chunks for `distance ≥ 16`.
- Scalar byte-by-byte for `distance < 16` (RLE-style; AVX2 can't
  safely overlap-write).
- Inlined into the JIT-emitted hot loop (not a function call).

### 3.7 Marker mode + clean mode share one codegen pipeline

`<Markers>` mode (parallel-SM speculative decode without window) and
`<Clean>` mode (window-known) compile from the same dynasm template;
they differ at the per-literal write site (u16 marker store vs u8
clean store). One pipeline, two monomorphizations.

### 3.8 Constant-time variant

Branchless `cmov`-based dispatch for security-sensitive callers.
~20% slower than the optimized variant. Opt-in via builder API.
Compiles from the same dynasm template with branchless emit
specialization.

### 3.9 GPU offload (large inputs only)

For inputs ≥ 1 GiB compressed AND ≥ 1000 estimated blocks, offload
to GPU via Metal (macOS) / CUDA (Linux+NVIDIA) / Vulkan compute
(portable). CPU does header-only scan to find block boundaries
(~10 GB/s bandwidth-bound), then dispatches per-block GPU decode.

Realistic peak ~20 GB/s on M3 Max (NOT the v1 plan's hand-wavy
30 GB/s). Done-when target ≥ 15 GB/s.

### 3.10 Three-oracle differential correctness

All three linked in-process via Rust C-FFI for native fuzz throughput
(~1M cases/sec):
- **Reference zlib** via `libz-sys`.
- **rapidgzip** via a C-ABI shim (`vendor/rapidgzip-cshim/wrapper.cpp`).
- **libdeflate** via `libdeflate-sys` (kept as a fuzz oracle even
  after production dep is deleted; fuzz workspace only).

Any two-way disagreement is a bug. Three-way agreement is high
confidence. ≥72h fuzz per release.

### 3.11 Decoder ships as a separately-publishable Rust library

`gzippy-inflate` sub-crate with stable public API:

```rust
pub struct Inflate;
impl Inflate {
    pub fn decode_gzip(input: &[u8]) -> Result<Vec<u8>>;
    pub fn decode_deflate_into(input: &[u8], output: &mut [u8]) -> Result<usize>;
    pub fn decode_stream(input: impl Read) -> impl Read;
    pub fn decode_async(input: impl AsyncRead) -> impl Stream<Item = Result<Vec<u8>>>;
    pub fn builder() -> InflateBuilder;
}
```

`no_std` + `alloc` core for `decode_deflate_into`. Independent
versioning; semver-stable. Compiles to WebAssembly cleanly.

### 3.12 Async API

`AsyncInflate::decode(AsyncRead).await -> Stream<Bytes>`. Compatible
with the sync API via a runtime adapter.

### 3.13 Memory-mapped direct decode

For CLI file decode, output goes directly into an mmap'd file; the
kernel handles writeback. Zero copy. Input similarly mmap'd. For
GB-scale inputs the in-RAM working set is just per-block JIT code +
chunk tail window.

---

## 4. The bake-off (the one place infinite labor genuinely changes the plan)

Three or four routes could plausibly close the 22pp gap. Under
finite labor we'd run cheap experiments to pick one. Under infinite
labor we **build all of them in parallel and pick the winner by
measurement**.

### 4.1 Route A: vendor libdeflate-rs

[libdeflate-rs](https://crates.io/crates/libdeflate-rs) is a Rust
port of libdeflate's C inflate. Vendor it under `vendor/libdeflate-rs/`,
patch any correctness issues found via three-oracle fuzz (§3.10),
wire it as the production inflate.

**Effort:** 1 person-week (vendor + patch + binding).

**Risk:** crate maintenance status; correctness bugs caught only by
the three-oracle fuzz; potential licensing review.

### 4.2 Route B: PGO on the current libdeflate-inner

Profile-guided optimization on the existing pure-rust libdeflate-
inner path. Collect profile with T=1 (parallel collection collides),
bench rebuild with T=16.

```bash
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --no-default-features --features pure-rust-inflate
./target/release/gzippy -d -c -p 1 silesia-large.gz > /dev/null
llvm-profdata merge -o /tmp/pgo.profdata /tmp/pgo-data
RUSTFLAGS="-Cprofile-use=/tmp/pgo.profdata" \
    cargo build --release --no-default-features --features pure-rust-inflate
```

**Effort:** 1 person-week (build infra + profile-corpus selection +
per-corpus regression sweep).

**Risk:** PGO can regress some workloads even if it helps silesia.
Mitigation: PGO profile MUST cover silesia + linux-source + web-
archive + random + btype01-heavy. Ship without PGO if any tracked
corpus regresses >5pp.

### 4.3 Route C: dynasm-rs hand-asm per-block (§3.1)

Build the full architecture in §3.1: dynasm x86_64 backend + AOT
codegen + interpreted fallback + JIT cache.

**Effort:** 3 person-months for x86_64; aarch64 backend +1 month
in parallel.

**Risk:** dynasm-rs maintenance status; macOS Hardened Runtime
entitlement; per-fingerprint asm authoring quality.

**2026-05-28 caveat (weakening this route)**: Route C v3.7 and
v3.9 spikes (commits `e642d7f`, `50d2930`) demonstrated that
**pure scalar dynasm asm is at parity with rustc-generated code**
on the literal-decode loop (369-405 MB/s asm vs 364-400 MB/s
rustc on 3350 silesia blocks). rustc with `-C target-cpu=native`
already emits equivalent scalar codegen. Route C only earns its
~3 person-month cost if the dynasm-emitted code is structurally
DIFFERENT from rustc output — i.e., uses BMI2 PEXT for bit
extraction OR AVX2 vpshufb for multi-byte literal output OR
speculative-parallel LUT lookups. The pure scalar asm pattern is
falsified as the differentiator.

**Implication for v1 scope**: §5's "Route C v1 ships with fixed
Huffman" should be amended to "Route C v1 ships with AVX2
vector-decode for the inner-loop hot path." Scalar Route C v1
doesn't earn its keep at the corrected ~10pp gap.

### 4.4 Route D: bit-reader rewrite (§3.2) on the current libdeflate-inner

Land the signed-i32 rollback pattern on the existing path (no JIT).
Targets the bit-reader codegen specifically.

**Effort:** 2 person-weeks.

**Risk:** if perf attribution shows <5pp lives in bit-reader codegen,
this route doesn't matter standalone — but it's a building block
for Routes B and C anyway, so the work isn't wasted.

### 4.5 How the bake-off resolves

All four routes built in parallel. Routes A/B/D land in weeks; Route
C lands in months. Interim production ships whichever of A/B/D wins
first; Route C replaces it later if it wins a rebake against the
interim winner.

After each lands, paired-bench on neurotic silesia + linux-source +
web-archive + random + btype01-heavy. **Three possible outcomes:**

1. **One route hits ≤1pp solo.** That route ships; others archived
   in `vendor/` for future reference.
2. **Routes compose** (e.g., Route D bit-reader rewrite + Route B
   PGO + Route C dynasm). Compose the winners; ship the combined
   build.
3. **No route hits 1pp alone or composed.** This is a failure mode
   — design decision goes back to the team to determine whether
   the language ceiling, the JIT codegen pattern, or the algorithm
   itself needs to change. No graceful goalpost rename.

The bake-off is **the** strategic decision the infinite-labor lens
changes. Everything else in the plan is end-state architecture or
correctness scaffolding.

---

## 5. Dependency graph

Even under infinite labor, some things genuinely depend on others.
This section captures the unavoidable build order.

```
                   ┌─────────────────────────────────┐
                   │ Three-oracle fuzz infra (§3.10) │
                   │ — REQUIRED for all routes —     │
                   └────────────────┬────────────────┘
                                    │
       ┌──────────────────┬─────────┴──────────┬─────────────────┐
       │                  │                    │                 │
       ▼                  ▼                    ▼                 ▼
 Route A:           Route B: PGO         Route D:           Route C:
 vendor             on current path      bit-reader         dynasm
 libdeflate-rs     (§4.2)               rewrite (§3.2)     hand-asm
 (§4.1)                                                    backend (§3.1)
                                              │              │
                                              │              │ depends on
                                              │              ▼
                                              │      JIT emit format
                                              │      definition
                                              │              │
                                              │              ▼
                                              │      AOT codegen
                                              │      (§3.1) — emits
                                              │      bit-reader pattern
                                              │      from §3.2
                                              └──────►   targets
                                                         this
```

Build-order constraints:
- **Three-oracle fuzz infrastructure (§3.10) is a prerequisite for
  ALL routes** — without it we can't validate correctness of any
  alternative inflate.
- **Bit-reader rewrite (§3.2) is a building block** for the AOT
  codegen (the emitted decoders bake in the bit-reader pattern).
- **JIT emit format must be defined** before AOT codegen targets it.
- **Route C's dynasm-emitted asm bakes in BOTH the shift-register
  width (§3.4) and the table-lookup shape (§3.5 perfect-hash vs
  canonical).** Route C v1 ships with fixed choices (256-bit AVX2
  register; canonical lookup). Route C v2 absorbs §3.4/§3.5 once
  measured; do not treat them as freely composable with v1.
- All other architectural pieces (§3.3, §3.6-§3.13) are independent
  workstreams that can land in any order, subject only to the
  correctness gate.

Routes A, B, D have no inter-dependencies; build all three in
parallel as soon as oracle infra exists.

---

## 6. Falsification record (preserved — institutional memory)

Future attempts must address the specific hypothesized failure mode
or validate that mode was wrong via perf counters before retrying.
Even under infinite labor, re-walking these dead ends is wasted
information.

### 6.1 From 2026-05-27

| Attempt | Measured | Hypothesized cause |
|---|---|---|
| B2 PRELOAD naive (lookup at iter end) | -10% | Wasted lookups on yield/EOB paths; preload at wrong pipeline point |
| 2-unroll FASTLOOP body | -10% | Icache pressure from doubled (100-line) macro body |
| FASTLOOP-scoped PRELOAD with cached entry | no change | OOO engine may already overlap lookup with prior iter's writes; not validated via stall counters |
| Single-literal fast path inside FASTLOOP | no change | Compiler likely already branch-predicts the common case; not validated |
| Writeback-skip around copy_match_windowed | -9pp | Borrow-checker pessimism plausibly caused spills; not validated via codegen diff |
| LTO=fat + codegen-units=1 + native | no change | Compile flags alone insufficient |
| `target-cpu=native` alone | no change | Same |

### 6.2 From 2026-05-28

| Attempt | Measured | Hypothesized cause | Future-retry guard |
|---|---|---|---|
| Route C v3.7 dynasm asm literal decode + byte-by-byte refill | at parity (369 vs 364 MB/s) | rustc with `-C target-cpu=native` already emits equivalent scalar codegen for the tight literal loop | Don't retry pure-scalar asm — needs SIMD (vpshufb / BMI2 PEXT) to be different from rustc output |
| Route C v3.9 dynasm asm + libdeflate-style chunked 8-byte refill | at parity (405 vs 400 MB/s) | Same as v3.7 — confirmation that refill strategy isn't the lever | Same guard |
| S1 u32 packed multi-literal store | +0.4% (within noise, 10 trials) | LLVM + x86 store-buffer coalescing already merges 4 adjacent byte stores as one cache-line write | Don't reorganize byte stores; the OoO core's store buffer already optimizes them |
| S2 bulk window-copy in `copy_match_windowed` slow path | at parity (657.8 vs 655.7 MB/s, 10-trial interleaved) | Slow path fires on ~3% of chunk bytes only — too rare to move overall throughput | Attack the FAST path (copy_match_fast) only, not the slow path |
| L1 madvise(MADV_HUGEPAGE) on fresh chunk allocations | -38% (487 vs 789 MB/s) | khugepaged contention when 16 workers all hint at once + madvise call latency exceeds savings | Don't use madvise hints under high worker count; explore MAP_HUGETLB direct mmap or daemon-mode prewarm instead |
| `GZIPPY_RESUMABLE_ISAL_INNER=1` as production default | -6% (630 vs 670 MB/s, 10-trial interleaved) | The ISA-L LUT inner loop's multi-pack symbol-emit path adds overhead per literal that libdeflate-LUT's single-symbol-emit doesn't have on production silesia | Keep libdeflate-LUT as default; ISA-L LUT helps marker phase only |
| C-fastloop iter-count batch (replace `while ... { decode_one_symbol!() }` with `for _ in 0..safe_iters`) | INFINITE LOOP (hung cargo test 4+ hours per stuck process) | When `safe_iters = 0` near chunk tail, for-loop ran zero iters and re-entered fastloop with same bounds | Any iter-count rewrite needs `safe_iters >= 1` invariant OR explicit fall-through to safe loop |

### 6.3 Confirmed positive results (2026-05-28)

| Attempt | Measured | Mechanism |
|---|---|---|
| **T3-simplify** (4-literal cap → vendor's 2-extra-literal) | +1.9% (904 vs 887 MB/s, 20-trial interleaved) | Removes a measured-bad path: vendor libdeflate `decompress_template.h:370` explicitly comments 3+ extras "actually decreases performance slightly" — gzippy's 4-cap was worse than vendor's 2-cap. Reduces i-cache pressure + branch-pred state. **Needs 30-trial confirm** before treating as canonical (currently below §7's ≥12 floor, but well above paired-bench noise floor on these 20 trials). |

### 6.4 Perf attribution (2026-05-28 symbolized)

Fresh `perf record` on neurotic (force-frame-pointers + debuginfo=1)
shows 19.10% memmove is **entirely in marker-bootstrap path**:
- `emit_backref_ring` (marker u16 ring backref copy): 2.84%
- `Vec::extend_from_slice` from `Block::drain_to_output`: 1.61%
- `copy_within` in `ChunkData::clean_unmarked_data`: 1.82%

Bulk inflate (`decode_huffman_body_resumable`,
`copy_match_windowed`) contributes essentially zero memmove. The
marker bootstrap runs in BOTH `pure-rust-inflate` and
`isal-compression` builds (same code; only bulk phase differs),
so memmove % is NOT what differentiates pure-rust from ISA-L
throughput. The gap really IS in the bulk inflate inner loop.

This redirects optimization effort: don't attack the per-byte slow
path of copy_match_windowed (S2 falsification confirms); do
consider the marker-phase u8 ring + parallel bitmap as a separate
~5% lever in the marker-decode workload.

See `docs/perf/2026-05-28-memmove-symbolized.md`.

---

## 7. Done-when

ALL of:

1. **Perf:** pure-rust inflate within 1pp of ISA-L FFI on neurotic
   silesia-large at T=16, paired-bench median over ≥ 12 trials.
   Confirmed on linux-source AND a web-archive sample AND random
   AND btype01-heavy corpora — no per-corpus regression worse than
   ISA-L's relative position on that same corpus.
2. **Correctness:** all 783+ lib tests; silesia byte-perfect through
   parallel-SM and sequential CLI; three-oracle differential fuzz
   ≥ 72h on every release with zero disagreements.
3. **Stretch perf (if 1pp requires it):** the AOT + dynasm
   combination strictly *beats* ISA-L on AVX2-or-later hardware per
   CLAUDE.md prime directive. Not required for "done"; required if
   the language ceiling means matching ISA-L requires going past it.
4. **Surface area cleanup:** one release-cycle soak (14 days
   production traffic, zero rollbacks attributed to the new decoder,
   no field bug reports), then delete the legacy inflate paths per
   §8.
5. **Opus advisor sign-off**. Per user process rule (2026-05-27),
   advisor consultation is required for **every judgment call AND
   every claimed task completion**, not only on the final neurotic
   1pp measurement. Enforced by `.claude/settings.json` hooks.
   Scope examples:
   - Before picking the next lever from §6.2/§6.3.
   - Before claiming a falsification is sufficient to update §6.
   - Before promoting a feature-gated experiment to production default.
   - Before declaring the 1pp threshold is met (the final gate).
   - Before any commit that touches the inflate hot path.

---

## 8. Cleanup commit (post-soak)

Once the bake-off winner is the production default and has soaked
one release cycle (14 days production traffic, zero rollbacks
attributed to the new decoder, no field bug reports):

Delete the legacy inflate paths (`src/decompress/inflate/resumable.rs`,
`libdeflate_decode.rs`, `consume_first_decode.rs`, the
`huffman_*.rs` family, `IsalInflateWrapper`, vendored ISA-L and
related Cargo features). **Exact file list depends on which route
won the bake-off** — rewrite this section against the actual
outcome. Net delta is ~10,000+ lines deleted, ~4,500 lines added
in the all-routes-compose case; substantially smaller if Route A
(vendor libdeflate-rs) wins solo.

---

## 9. Risks (real even under infinite labor)

1. **Building the wrong thing.** "Infinite labor → build it all"
   loses the falsifying information from cheap spikes. Mitigation:
   the bake-off explicitly runs Route A (libdeflate-rs) in parallel
   with C (dynasm) — if A wins, B and C die without us having to
   pre-judge.
2. **Integration nightmare.** ~13 simultaneous workstreams (§3.1-
   §3.13) all touch the inflate hot path. Mitigation: §5 dependency
   graph; oracle infra first; serialize §3.1 → §3.2 → AOT chain.
3. **dynasm-rs maintenance.** Vendor it under `vendor/dynasm-rs/`
   to control upgrades.
4. **AOT compile-time tax.** Bounded ≤30s cold add per §3.1 budget.
   If exceeded, cut AOT count from 256 to 128.
5. **JIT memory pressure.** Ring LRU eviction at 64 MiB cap.
6. **`feature = "no-jit"` builds get only Routes A/B/D benefits.**
   They never see Route C. Document explicitly.
7. **Three-oracle fuzz finds correctness divergences in
   libdeflate-rs.** Vendor + patch upstream OR abandon Route A.
8. **PGO regresses non-silesia workloads.** Profile coverage must
   include all tracked corpora; ship without PGO if any regresses
   >5pp.
9. **GPU dispatch OS-specific.** Each backend behind its own feature
   flag; GPU is opt-in.
10. **The 22pp gap may be unreachable by ALL four routes composed.**
    Activates §2 "beat ISA-L" path: implement BMI2 PEXT in
    hand-tuned asm; per-CPU dispatch; arch-specific binaries.
    Multi-quarter even under infinite labor because the work is
    sequential learning, not parallel coding.
    **2026-05-28 UPDATE**: gap is ~10pp after clean-build re-measure
    (not 22pp). This risk now reads "the 10pp gap may be unreachable"
    — still real, but the routes have less work to do.
11. **Iter-count rewrites of the inflate fastloop need a non-zero
    invariant.** The 2026-05-28 C-fastloop attempt replaced
    `while in_pos < in_fl_end && out_pos < out_fl_end { ... }` with
    `for _ in 0..safe_iters { ... }`. When `safe_iters = 0` near
    the chunk tail (e.g., `in_fl_end - in_pos < 8`), the for loop
    ran zero iterations and `continue`d to outer loop, which
    re-entered the fastloop with the same bounds → infinite loop.
    Hung `cargo test resumable` 4+ hours (235 minutes CPU) before
    kill. **Mitigation**: any iter-count rewrite of the fastloop
    MUST ensure `safe_iters >= 1` OR fall through to a bounded
    safe-loop iteration. Pre-flight gate: cargo test resumable
    must finish in <30s (vs hanging) before benching.
12. **Scalar codegen is already optimal in rustc + LLVM**
    (2026-05-28 — confirmed by 3 falsifications: Route C v3.7/v3.9
    dynasm asm at parity, S1 packed store at parity). New attempts
    on the inflate inner loop must demonstrate a structural
    difference (SIMD, BMI2 PEXT, vector-shaped data movement) or
    pre-justify why they wouldn't be LLVM-defeated.
13. **Bench contamination from debug-info in release profile.**
    Adding `debug = "line-tables-only"` and `strip = false` to
    Cargo.toml's `[profile.release]` (e.g., for perf symbolization)
    materially slows the binary. This session showed -12pp
    throughput on the contaminated build. **Mitigation**: every
    bench session starts with `grep -E "^strip" Cargo.toml | head -3`
    showing `strip = true`.

---

## 10. Open questions (load-bearing — owned by parallel workstreams)

1. **Top-256 fingerprint coverage of real corpora.**
   **Answered by:** half-day script parsing silesia + linux-source +
   web-archive samples, counting distinct `(litlen_lengths,
   dist_lengths)` tuples and their frequency. Determines whether
   AOT specialization (§3.1) yields hit rate worth the build-time
   tax.
   **2026-05-28 status: ANSWERED — falsified.** Corpus walker on
   silesia found 3348 dynamic blocks with 3348 unique fingerprints
   (zero repeats). Top-N AOT specialization yields ~0% hit rate on
   silesia. Memory: [[project_phase1_codegen_audit]]. Combined
   with the gap correction (10pp not 22pp), AOT fingerprints are
   downgraded as a lever. Re-prioritize toward AVX2 vector decode
   for Route C v1.
2. **CHD vs BBHash vs FrozenHashMap for perfect-hash construction
   (§3.5).** **Answered by:** half-day microbench on silesia block-
   size distribution.
3. **dynasm-rs aarch64 maturity.** **Answered by:** spike — emit a
   single fixed-Huffman decoder, verify byte-perfect on Apple
   Silicon.

---

## 11. Effort estimate (for sizing the engagement, not gating it)

- Route A (vendor libdeflate-rs): ~1 person-week.
- Route B (PGO): ~1 person-week.
- Route D (bit-reader rewrite): ~2 person-weeks.
- Route C (dynasm x86_64): ~3 person-months. aarch64 backend +1
  month in parallel.
- §3.5 perfect-hash: ~2 person-weeks.
- §3.8 constant-time: ~3 person-weeks.
- §3.9 GPU offload: ~2 person-months per backend; can ship
  Metal-only first.
- §3.10 three-oracle fuzz infra: ~1 person-month (rapidgzip C-ABI
  shim is the long pole).
- §3.11 sub-crate publish + no_std + WebAssembly: ~1 person-month.
- §3.13 mmap direct decode: ~2 person-weeks.

Total: ~6 person-months full architecture, perfectly parallelized;
~12 person-months realistic parallelism with integration friction.
Under literal infinite labor, ~3 months calendar. Under realistic
"large team" (~5-8 engineers), ~6 months calendar.

---

## 12. Out-of-scope (was the stretch document; now deleted)

The "stretch" plan (`plans/unified-decoder-stretch.md`) was deleted;
recover from git history if needed. Most of its content was pulled
into §3 here (constant-time, GPU, perfect-hash, sub-crate publish,
async API, mmap, three-oracle fuzz). What remains explicitly
out-of-scope: Creusot formal verification (research-grade); hardware
coprocessor dispatch (QAT not present on target hardware); eBPF
probes (observability, not perf); forensic recovery API (post-decode
feature, not inflate hot path); coroutine suspend/resume (parallel-
SM internal, not user-visible).
