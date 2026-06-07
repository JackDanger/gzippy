# TIER-1 DESIGN + TIER-2 PROOF PLAN (leader deliverable → supervisor + advisor)

Charter: plans/tier-approach-mandate.md. Bar: **1.0× TIE** of gzippy-native vs
rapidgzip on T8 silesia parallel single-member decode. Pure-Rust + **inline ASM
allowed**. Method is TIERED: DESIGN (here) → PROVE feasibility with tooling BEFORE
building (TIER-2 plan here) → ALIGN gzippy-native (TIER-3, only after supervisor +
independent-advisor corroborate this doc).

This document is the checkpoint. **No TIER-3 implementation has begun.**

---

## 0. The diagnosis this design is built on (TIER-1 research, all source-cited)

Two read-only Opus subagents + leader first-hand vendor reading converged on a
single root cause that **corrects** two hypotheses the prior arc carried.

### 0.1 What rapidgzip actually does (the existence proof)
rapidgzip's measured ~0.46s T8 wall is **~99% ISA-L (igzip) C/SIMD decode**, not
its pure marker engine. Vendor dispatch in
`vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp`:
- **Known-window chunk → 100% ISA-L** (`:440-444`, `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`). This is `d_c`.
- **Window-absent chunk → pure `deflate::Block` for ONLY the ≤32 KiB markered
  prefix, then ISA-L for the multi-MiB remainder** (`:520-526`, switch when
  `cleanDataCount >= MAX_WINDOW_SIZE`). This is `d_w`.
- The in-place u16→u8 "flip" exists in `deflate::Block` (`deflate.hpp:1282-1289`)
  but is **vestigial in production** — ISA-L takes over right after it would fire.
- `LIBRAPIDARCHIVE_WITH_ISAL` defaults **ON** (`librapidarchive/CMakeLists.txt:14`,
  "more than twice as fast … than zlib"); the harness builds the trace binary with
  no override, so the measured rapidgzip **is** the ISA-L build. `WITH_RPMALLOC`
  also ON (faster allocator — a secondary factor).

So the **uniform ~1.85× on both `d_c` and `d_w`** is ONE shared mechanism: the same
igzip engine decodes ~all bytes in both bands. (parity-final.md ledger A already
noted vendor `worker.isal_stream_inflate` busy = 1289.9ms — non-zero — confirming
the handoff.) The ISA-L share is **≥94%, prefix÷chunk-bounded** (32 KiB prefix over
a multi-MiB chunk ≈ 99%; even a pessimistic 512 KiB chunk ≈ 94%) — state it as a
bound, not a load-bearing "99%" constant. There are **three** ISA-L handoffs, all
`#ifdef LIBRAPIDARCHIVE_WITH_ISAL`: known-window (`:440-444`), new-stream
(`:501-505`), post-prefix (`:520-526`) — advisor independently re-verified all three.
[Cosmetic: the u8 reinterpret is `deflate.hpp:806` not `:893`; vendor paths are the
`librapidarchive/` tree.]

### 0.2 Two prior hypotheses — one REFUTED, one RE-FRAMED (advisor disproof)
- **Ring-STORAGE width ("narrower u8 buffer") is REFUTED.** rapidgzip's
  `deflate::Block` uses the **identical** 128 KiB u16 ring
  (`deflate.hpp:805 std::array<uint16_t, 2*MAX_WINDOW_SIZE>`, `alignas(64)`,
  reinterpret-cast to u8 in place at `:806`). gzippy's `output_ring:
  Box<[u16; RING_SIZE]>` (`marker_inflate.rs:290`) is a faithful port. Both pure
  decoders store the ring as u16. So storage width is NOT a divergence.
- **BUT clean-bulk WRITE + RESOLVE traffic width is a LIVE asymmetry, NOT refuted
  (advisor correction — important).** Two distinct facts were collapsed in an
  earlier draft: (a) ring-storage width [refuted, above]; (b) the *traffic* on the
  clean bulk. rapidgzip's PRODUCTION clean path is ISA-L, which writes **u8 directly**
  (`isal.hpp:257 next_out=output`) and **never touches the u16 ring** for the bulk —
  and confines markers (hence the u16 resolve pass) to the ≤32 KiB prefix. gzippy's
  clean-mode loop writes **u16 per literal** into the ring on ~100% of bytes
  (`marker_inflate.rs:1526 ring_ptr:*mut u16 .write(sym)`) AND re-streams the full
  ~20 MiB u16 chunk buffer in the resolve pass (§0.4). So gzippy pays ~2× write
  traffic + a full-chunk u16 re-read on ~100% of bytes that rapidgzip pays on ~0%.
  This is a **bandwidth** difference in the exact T8 regime §0.3 shows is
  bandwidth/cache-sensitive — a plausibly-PRIMARY, LOW-RISK lever, distinct from the
  high-risk SIMD-compute work. (Note: rapidgzip's *pure* decoder also writes the u16
  ring, so gap-A [gzippy-pure vs rapidgzip-pure] is compute; gap-B [pure-u16 vs
  ISA-L-u8] BUNDLES asm-compute AND the u8-traffic/no-resolve advantage — gap-B
  cannot be closed by SIMD compute alone, part is architectural traffic.)
- **"BMI2 / per-symbol ALU is the lever."** Already rejected with mechanism (arc:
  `314f94b`, byte-exact A/B TIE; BMI2 BZHI already compiled into the perf build).
  Now *explained*: the gap is not one extract op — it's that gzippy runs a **scalar
  Rust marker loop for ~100% of bytes** where rapidgzip runs **igzip AVX2 assembly**
  (`vendor/isa-l/igzip/igzip_decode_block_stateless_04.asm`, selected via igzip
  multibinary dispatch) for ~99%.

### 0.3 The ceiling evidence (vendor's own benchmark, `deflate.hpp:72-93,137-145`)
Vendor benchmarked decoder variants on the same CPU. **Single-thread silesia**
(run deliberately to "reduce contributions of memory bandwidth"):

| decoder | MB/s (single-thread silesia) | note |
|---|---|---|
| HuffmanCodingISAL (igzip asm) | **720** | the engine in the measured rapidgzip |
| ShortBitsMultiCached-11 | **337** | rapidgzip's *best pure* (non-asm) decoder |
| ShortBitsCached-11 | 328 | simpler pure LUT |
| DoubleLiteralCached | **252** | the class gzippy ported (IsalLitLenCodePure lineage) |

**Multi-thread silesia (20×):** ISA-L 5024 vs ShortBitsCached-11 3927 → only
**1.28×** (the gap shrinks at high T). **WHY it shrinks matters (advisor correction):**
it is NOT purely DRAM bandwidth. Vendor's own text (`deflate.hpp:170-172`) attributes
the multi-thread degradation of `DoubleLiteralCached` to **its LUT being too large
for cache when two HW threads share a core — i.e. CACHE CONTENTION**, a *different*
mechanism with a *different* remedy (shrink the per-thread working set → the dist-cache
shrink) than DRAM bandwidth (move fewer bytes → the u8-clean-write fix). So the high-T
convergence is consistent with **cache/traffic** binding — which the traffic/residency
levers (u8 writes, drop clean-resolve, dist-cache shrink) address directly. This is why
those levers must NOT be pre-ranked below the SIMD-compute work (§1.2) — the binding
term is a measurement, not an assumption.

**CPU-transplant caveat:** these are a "Frankensystem" with ±10–20% variance noted
(`deflate.hpp:96-107`). The absolute MB/s (337/720) are NOT legitimate guest targets —
TIER-2 gates on **guest-measured RATIOS** (the in-bench ISA-L oracle), never these
absolutes.

Read this carefully — it sets the feasibility envelope:
1. ISA-L is **2.1×** the best *pure* decoder single-thread → a pure **scalar** loop
   physically cannot reach igzip throughput; the asm matters.
2. At high T the *observable* gap is only ~1.28× (bandwidth masks compute) — but
   it is **not** zero, so the workload is NOT fully bandwidth-bound; engine speed
   still reaches the wall.
3. gzippy measures **1.85×** at T8 — **worse** than rapidgzip's own 1.28× pure/ISA-L.
   gzippy's pure engine underperforms even rapidgzip's *pure* decoder. So there are
   TWO stacked gaps: (gap-A) gzippy-pure vs rapidgzip-pure ≈ 1.85/1.28 ≈ **1.45×**
   of headroom in pure-Rust alone, and (gap-B) the residual rapidgzip-pure vs ISA-L
   ≈ **1.28×** that needs SIMD/asm-class decode to close.

### 0.4 gzippy native per-thread footprint (subagent A, source-cited)
~**279 KiB** decode hot state per worker, dominated by **two** 128 KiB structures:
`output_ring` (128 KiB) + an **inline (unboxed)** distance `code_cache` (128 KiB,
`HuffmanCodingReversedBitsCached<…30>`, faithful to vendor `deflate.hpp:336`) +
16 KiB lit/len short LUT. Overflows a 256 KiB L2, fits a ≥512 KiB L2. Per-chunk
~20 MiB u16 buffer + the resolve pass are **DRAM-bandwidth**, not residency.

**Three prior-note corrections (must reconcile in plans):**
1. `FIXED_TABLES`/`LitLenTable`/`DistTable` (libdeflate path) are **NOT live** on
   native — the prior "shared tables already done" checked a dead path. Live tables
   are per-thread, per-block-**rebuilt**; **no shared read-only table** on the hot
   loop. (So the cache-mandate "share tables" item is *unmet*, not done.)
2. dist cache is a 128 KiB **inline** array → `thread_local Block` ≈ 279 KiB.
3. resolve streams the ~20 MiB chunk buffer, not the 128 KiB ring.

### 0.5 Structure-mandate finding (a "BMI2-was-already-on"-class trap)
`guest_fulcrum_capture.sh:69-71` claims `GZIPPY_BUILD_FEATURES=isal-compression`
gives "the SAME ISA-L clean decode rapidgzip uses (apples-to-apples engine A/B)".
`build.rs:90-96` proves this **FALSE** post-fold: `isal-compression` no longer
enables ISA-L *decode*; it would rebuild the pure decoder. The only real-ISA-L
gzippy decode build is `gzippy-isal` (Design-A tail, deferred). Fix the comment in
the structure pass; record that **no in-one-binary engine-swap A/B exists yet**.

---

## 1. TIER-1 DESIGN — one coherent architecture aimed at 1.0×

### 1.1 The governing tension (decide this FIRST — supervisor call)
The **faithful rapidgzip structure is NOT one engine** — it decodes the markered
prefix with `deflate::Block` and hands the clean tail to **igzip (C FFI)**. The
governing memory (`project_faithful_unified_decoder_over_perf`) mandates **ONE
MarkerRing engine, flip-in-place, NO 2nd engine, NO C-FFI on the decode graph** —
and the campaign just did that fold (`8cfad3a`). These cannot both hold at a 1.0×
TIE: to match igzip's bulk throughput you need *either* igzip itself (FFI — banned
on native) *or* a pure-Rust+ASM decoder that equals it.

**Design resolution (proposed):** keep the **one-engine, no-FFI** invariant for
gzippy-native (honor the governing memory + the FFI-free decode mandate), and close
the gap by making that **single engine's clean-path inner loop igzip-class** via
pure-Rust + **inline ASM** (explicitly charter-permitted). I.e. we do NOT
re-introduce a 2nd engine or FFI; we make the one engine fast. The
"faithful-to-rapidgzip-structure" goal is then satisfied at the *pipeline*
granularity (chunk lifecycle / block finder / window map / marker resolve — all
already faithful per parity-final.md), and the inner clean decode is the
explicitly-authorized open-territory reimplementation ("build the fastest possible
raw Huffman decoder"). **This needs supervisor ratification** because it formally
accepts that gzippy-native's inner loop *diverges* from rapidgzip's
(igzip-asm-vs-our-asm) while the architecture stays faithful — which is consistent
with CLAUDE.md's scoping ("port faithfully scoped to architecture; inner Huffman
loop is full-reimplementation territory").

### 1.2 The architecture (single cache-resident pure-Rust+ASM engine)
Components below. **NOTE (advisor correction): the ordering is the candidate set, NOT
a pre-ranking.** The TIER-2 PRE-GATE (§2.0) ranks them by measured T8-wall response
BEFORE any expensive build — the design must NOT pre-judge SIMD-compute as the lever
over the traffic/residency fixes (that is the tier-discipline trap the BMI2 arc fell
into). There are TWO classes of lever, and §0.2/§0.3 say the cheap class may dominate
at T8: **(class-T) traffic/residency** — u8-direct clean writes, eliminate the clean-
byte u16 resolve pass, shrink the 128 KiB dist cache (all low-risk, hours); and
**(class-C) SIMD compute** — igzip-class vectorized clean inner loop (high-risk,
weeks). The PRE-GATE decides which class to invest in.

**(A) Two-mode single engine — markered prefix vs clean bulk.** Keep the existing
`marker_inflate::Block` and its in-place flip (faithful, already folded). Split the
inner decode into two `const`-specialized loops on the SAME cursor/buffer:
  - **Markered mode** (`CONTAINS_MARKERS=true`): today's loop, scalar, runs only
    while `distance_to_last_marker_byte < MAX_WINDOW_SIZE` (the ≤32 KiB prefix —
    structurally the same window rapidgzip gives the marker engine). Low volume;
    leave scalar. This is the only place markers exist.
  - **Clean mode** (`CONTAINS_MARKERS=false`): the post-flip bulk — **this is where
    ~94–99% of bytes are and where the 1.85× lives.** Two stacked gaps live here
    (§0.3): gap-A (pure-Rust decode-class headroom, gzippy underperforms even
    rapidgzip's pure decoder) and gap-B (the pure→ISA-L residual, which BUNDLES asm
    compute AND the u8-traffic/no-resolve advantage). The clean-mode rewrite therefore
    has two independent fronts — **class-T traffic** (u8-direct writes; drop clean
    resolve) and **class-C compute** (vectorized inner loop) — sequenced by the §2.0
    PRE-GATE, not assumed.

**(B-class-T) Traffic/residency fixes (LOW-RISK; gate FIRST via §2.0).**
  - **u8-direct clean writes.** In clean mode write decoded literals as **u8 into the
    flipped (u8-interpreted) ring** instead of u16 (`marker_inflate.rs:1526`) — halves
    clean-path write traffic on ~100% of bytes (the gap-B traffic component, §0.2).
  - **Eliminate the clean-byte resolve pass.** Clean bytes have no markers; the
    ~20 MiB u16 → u8 resolve+narrow stream (`segmented_markers.rs:481`) should run
    over the ≤32 KiB markered prefix ONLY, not the whole chunk. Removes a full-chunk
    u16 re-read that rapidgzip never does.
  - **Distance-cache shrink** (§0.4 128 KiB inline `code_cache`): a two-level dist
    table (small hot L1 LUT + cold escape) instead of the flat 128 KiB array — cuts
    the second 128 KiB per-thread structure AND the scattered per-back-ref access
    (the cache-contention mechanism vendor names at `deflate.hpp:170-172`).

**(B-class-C) SIMD compute (HIGH-RISK; gate only if §2.0 says compute binds).**
  1. **Vector back-ref copy** (igzip overlapping SIMD/`rep_movs` copy): copy runs of
     mean ~6–7 bytes today via scalar u64.
  2. **Packed multi-literal WRITE** (historically-regressing `ca52389` — non-binding,
     re-measure): collapse 2–3 decoded literals into one wide store.
  3. **Wider refill + branch-lean availability** in the clean loop only
     (FASTLOOP_OUTPUT_MARGIN path; markered mode keeps per-iter refill, backref-safe).
  Inline ASM permitted for refill + packed store + copy where Rust codegen lags.
  This is the path that risks landing at the ~337 pure ceiling, not 720 — fenced by
  §2 PROOF-1.

**(C) Cache-residency (the mandate, but right-sized by §0.4).** The mandate's
"share read-only tables across threads" is currently **unmet** (no shared table on
the live path). BUT fixed-Huffman tables ARE shareable (one OnceLock, faithful to
vendor `deflate.hpp:920 static constexpr` fixed coding). Dynamic tables are
inherently per-block (rapidgzip also rebuilds them). So the cache work is: (i)
share the FIXED table (small, rare in silesia — low leverage, do for correctness of
the mandate not for wall); (ii) shrink the 128 KiB dist cache (class-T, real leverage
per vendor's cache-contention note); (iii) keep `output_ring` 128 KiB (faithful, same
as vendor — do NOT shrink, it would diverge and vendor proves storage-width is not the
cause). Net target per-thread hot state: ~279 KiB → ~160 KiB (drop the 128 KiB dist
cache to a ~16–32 KiB two-level table), which then fits a 256 KiB L2. **Whether this
moves the wall is a §2.0/MPKI question — it is NOT assumed. (Footprint note: the
fixed-Huffman block path instantiates a SECOND 128 KiB dist cache locally
(`marker_inflate.rs:1587`), so a fixed-block transient spikes to ~400 KiB; minor for
mostly-dynamic silesia but the dist-cache shrink helps this too.)**

**(D) Allocator.** rapidgzip uses rpmalloc. The per-chunk ~20 MiB u16 buffer +
resolve are DRAM-bandwidth; allocator churn is a candidate secondary factor. Low
priority; measure only if TIER-2 shows alloc on the critical path.

### 1.3 What this design explicitly does NOT do
- Does NOT re-introduce C-FFI / ISA-L on the native decode graph (governing memory).
- Does NOT add a 2nd engine for the clean tail (the fold stands; the one engine gets
  a fast clean mode instead).
- Does NOT shrink the 128 KiB u16 ring (faithful to vendor; vendor refutes width as
  the cause).
- Does NOT pursue BMI2-as-a-lever (rejected, already on).
- Does NOT hill-climb: the SIMD clean loop is ONE coherent inner-loop rewrite gated
  by an isolation benchmark, not a sequence of speculative micro-levers on the wall.

---

## 2. TIER-2 PROOF PLAN — prove feasibility BEFORE building (the gate to TIER-3)

The charter requires PROVING the design can reach ~0.46–0.54s with **real measured
constants + a positive control**, not asserting it. There is a CHEAP DECISIVE
PRE-GATE (§2.0) that runs FIRST and ranks the lever classes; then two complementary
proofs (PROOF-1 isolation bench, PROOF-2 model). The PRE-GATE decides whether the
expensive class-C SIMD proof is even worth running.

### PROOF 0 (PRE-GATE) — cheapest decisive experiments, run BEFORE PROOF-1/2
CLAUDE.md's own method: test "is compute the T8 lever?" with a **causal perturbation
before a work-stretch**, not a priority list. Two cheap byte-exact experiments on the
locked harness (interleaved, sha-gated, frequency-neutral control), each ~hours:

- **PG-A: clean-loop slow-injection.** Slow ONLY the clean-mode inner decode by a
  known factor (the `GZIPPY_SLOW_BOOTSTRAP`/ballast spin template, scoped to the
  `CONTAINS_MARKERS=false` loop). Measure the interleaved T8 wall response + a SLEEP
  (frequency-neutral) control. **Monotonic+proportional ⇒ clean-loop COMPUTE is on
  the T8 critical path ⇒ class-C SIMD is worth PROOF-1. FLAT ⇒ bandwidth/cache/publish
  already binds ⇒ the AVX2 project is MOOT (falsified in ~an hour, not after a
  multi-week build).** This is the gate the BMI2 arc should have had.
- **PG-B: u8-clean-write + drop-clean-resolve A/B.** A small byte-exact change: write
  clean-mode literals as u8 and skip the resolve pass over clean bytes (§1.2 B-class-T).
  Measure the interleaved T8 wall delta. This DIRECTLY isolates the class-T traffic
  component of gap-B (§0.2) — a measured wall move here proves traffic is a real lever
  independent of any SIMD work, and it is low-risk enough to potentially KEEP regardless.

PRE-GATE verdict routes the rest: PG-A monotonic → run PROOF-1 (class-C). PG-B moves
the wall → class-T is a confirmed lever, fast-track it. PG-A flat AND PG-B flat →
the lever is elsewhere (publish chain / pipeline) — re-open the diagnosis before any
engine build. **Δ < inter-run spread ⇒ TIE on that experiment, per the rules.**

### PROOF 1 — Standalone toy decoder benchmarked in isolation (the existence proof)
(Run only if PROOF-0 PG-A says clean-loop compute binds the T8 wall.)
**Question it answers:** can a pure-Rust + inline-ASM clean DEFLATE inner loop reach
**igzip-class throughput on the GUEST CPU** — measured as a guest RATIO vs the
in-bench ISA-L oracle, NOT vs the vendor's Frankensystem absolutes (§0.3 caveat: the
337/720 MB/s numbers are a different CPU and must not be used as guest targets)?

**Build:** a standalone bench (a `#[bench]`-style binary or a criterion harness,
NOT the full pipeline) that decodes a *known-window clean* DEFLATE stream (the d_c
case — the dominant one) three ways on the SAME guest x86_64 CPU under the build
lock:
  - (i) gzippy's CURRENT clean inner loop (scalar u16, baseline),
  - (ii) the proposed SIMD/wide clean loop (vector copy + packed literal + wide
    refill; inline ASM where needed),
  - (iii) **ISA-L `isal_inflate` itself** as the upper-bound oracle (call the C lib
    directly in the bench — this is a *measurement* oracle, not production; FFI stays
    off the native decode graph).
Input: a clean (window-pre-seeded) silesia chunk extracted once. Output verified
byte-exact each variant (a fast decoder with wrong bytes is void).

**Positive control (charter-mandated):** the bench must reproduce vendor's OWN
qualitative ratio on the GUEST — variant (iii) ISA-L should read ~2× variant (i)'s
scalar class single-thread (the RATIO, not the absolute MB/s, must reproduce). If it
does NOT, the bench is broken — fix before trusting any (ii) number. (Instrument
self-test: a known ratio must reproduce.)

**PASS criterion (guest-ratio terms):** variant (ii)/variant (iii) throughput ratio
on the guest is high enough that, fed through the PROOF-2 T8 model, the projected T8
wall ties rapidgzip (≤ rapidgzip_wall × (1 + spread)). If (ii) plateaus near (i)'s
scalar class (i.e. SIMD didn't materialize a real Rust win) and the model says that
leaves >spread on the T8 wall, **the design is NOT proven** — report the achievable
floor + residual to the supervisor; do NOT start TIER-3 on an unproven design.

### PROOF 2 — Executable bandwidth-vs-compute model with REAL measured constants
**Question it answers:** at T8 on the guest, how much of the single-thread engine
speedup from PROOF-1(ii) actually reaches the WALL before DRAM bandwidth (or the
publish chain) re-binds? (The vendor's 1.28× multi-thread vs 2.1× single-thread gap
proves bandwidth masks compute at high T — we must quantify it for the guest.)

**Measured constants (captured WITH positive controls on the guest, locked host):**
  - Per-T (1,2,4,8,16) **decode symbol-throughput** of the current engine (from the
    locked Fulcrum `d_c`/`d_w` already captured: 85.5/122.5 ms gzippy, 44.1/66.7
    rapidgzip — re-confirm at this HEAD).
  - Per-T **L2/L3 MPKI + memory-stall cycles** via `perf stat` on the guest (the
    charter's missing measurement — the BMI2-TIE proved "not ALU" but NOT "is
    cache"; this is where we finally measure it). Positive control: the existing
    `GZIPPY_MEM_BALLAST_MIB` knob (inflates per-thread resident bytes by a known N)
    must move MPKI/RSS monotonically — validate the instrument first.
  - **Aggregate DRAM bandwidth ceiling** of the guest (STREAM-style probe, or derive
    from the T-scaling knee where added threads stop adding throughput).
  - The per-chunk ~20 MiB u16 traffic + resolve-pass traffic (subagent A) as the
    bandwidth demand per decoded byte.

**The model:** `wall(T) = max( compute_term(T, engine_throughput),
bandwidth_term(T, bytes_per_decoded_byte, dram_ceiling), publish_chain_term(T) )`
parameterized by the constants above.

**Anti-overfit (advisor correction — a 3-term max() validated by reproducing the two
walls it was tuned on is near-circular; reproduction ≠ predictive power):**
  - **Hold-out cross-validation.** We already capture per-T {1,2,4,8,16}. FIT the
    constants on T1/T2/T4 and PREDICT T8/T16 as HELD-OUT points. Trust the faster-
    engine projection only if the held-out walls land within spread.
  - **Measure the binding term DIRECTLY, don't infer it from the fit.** Use the
    guest `perf stat` MPKI / mem-stall + the DRAM-ceiling probe (with the
    `GZIPPY_MEM_BALLAST` positive control) to ASSERT whether T8 is compute- or
    bandwidth/cache-bound. That direct measurement is the verdict; the model is only
    a projector of "how much of PROOF-1(ii)'s speedup reaches the wall."

**PASS criterion:** the model — held-out-validated AND consistent with the direct
perf-stat binding-term measurement — projects gzippy-native T8 wall ≤ rapidgzip T8
wall + spread when fed PROOF-1(ii)'s (and/or PROOF-0 PG-B's class-T) measured
improvement. If the wall re-binds on bandwidth/publish BEFORE the TIE (even an
infinitely-fast engine can't TIE), that is a **falsification of the 1.0× bar via THIS
design** — report it; the design must then add the traffic/bandwidth surface (class-T:
u8 writes, drop clean-resolve, dist-cache shrink) or the bar is revisited.

### Disciplines for both proofs (non-negotiable)
- ALL builds via `scripts/cargo-lock.sh` (global mkdir-mutex); ONE build at a time.
- Numbers ONLY from the locked Fulcrum harness / the guest under host-lock; never a
  hand-rolled wall script (CLAUDE.md rule 8). The isolation bench is an *instrument*
  — validate it (positive control) before trusting it.
- Byte-exact every variant; sha-gate.
- Pre-register the falsifier for each proof (above) BEFORE running it.
- Independent Opus advisor corroborates this design + both proof designs BEFORE
  TIER-3, and corroborates the PROOF results before TIER-3 commits.

---

## 3. Structure-mandate work (do alongside, all byte-exact)
Per charter §STRUCTURE. Sequence after this checkpoint is ratified:
- Move the two paths into two subdirectories: `gzippy-isal` tree (faithful rapidgzip
  + FFI clean-tail handoff — reference baseline) and `gzippy-native` tree (the one
  pure-Rust engine). (Today they're feature-gated in shared files; the dir split is
  the orientation aid the mandate wants.)
- Remove dead code: `unified.rs` `HAS_BMI2=false` dead placeholder; the
  specialized_decode + SPEC_CACHE production-dead cluster (the deferred coupled
  edit); fix the STALE `isal-compression` engine-A/B comment in
  `guest_fulcrum_capture.sh:69-71` (§0.5).
- Names describe behavior (continue the naming-truth discipline). A technique that
  is ON is visibly ON; no dead consts reading as live gates.
- Keep all 800+ lib tests green; dual-sha 028bd002…cb410f each step.

---

## 4. Independent advisor disproof — verdict (done this session)
An independent Opus advisor re-read every load-bearing citation against source
(disproof-driven). VERDICT: diagnosis sound enough to design on, with corrections
(all incorporated above):
- AGREED & re-verified first-hand: CLAIM 1 (ISA-L dominance, all 3 handoffs), the
  128 KiB u16 ring identity, the 128 KiB INLINE dist cache + no-shared-table, the
  BMI2 rejection, and that the harness binary is the ISA-L build.
- CORRECTED (now in §0.2/§0.3/§1.2/§2): (1) separate ring-STORAGE width [refuted]
  from clean-bulk WRITE+RESOLVE traffic width [LIVE lever, ~100% of bytes]; (2) the
  high-T convergence is partly CACHE CONTENTION (vendor `deflate.hpp:170-172`), not
  pure bandwidth — so class-T traffic/residency levers must NOT be pre-ranked below
  class-C SIMD; (3) PROOF-2's validate-by-reproduction is overfit-circular → added
  hold-out cross-validation + direct perf-stat binding-term measurement; (4) strip
  absolute MB/s targets → guest-measured ratios; (5) added PROOF-0 PRE-GATE (cheap
  causal perturbation + u8-write A/B) AHEAD of the expensive proofs.
- Advisor's bottom line: PROOF-1 sound; PROOF-2 fixed; the PRE-GATE is the missing
  cheapest decisive gate and should run first.

## 5. Checkpoint ask to supervisor + advisor (DO NOT start TIER-3 before this)
1. **Ratify the governing-tension resolution (§1.1):** one no-FFI engine whose clean
   inner loop is reimplemented to igzip-class via pure-Rust+inline-ASM — accepting
   inner-loop divergence from igzip-asm while architecture stays faithful. (The only
   charter-legal way to honor BOTH the 1.0× bar and the no-FFI / one-engine memory;
   advisor concurs it is the only legal path AND the highest-risk one — hence the
   proof gates.)
2. **Corroborate the diagnosis (§0)** — already independently disproof-checked this
   session; supervisor's advisor can re-confirm the cited vendor lines.
3. **Approve the revised TIER-2 plan (§2):** PRE-GATE (PG-A slow-injection + PG-B
   u8-write A/B) FIRST, then PROOF-1 (isolation bench, guest ratios) only if compute
   binds, PROOF-2 (held-out-validated model + direct perf-stat). Or amend.
4. If the PRE-GATE/PROOF-2 shows the 1.0× bar is unreachable by THIS design
   (bandwidth/publish/cache-contention re-binds before the TIE), decide: extend the
   design (class-T traffic/residency surface) or revise the bar. **The measurement,
   not optimism, decides.**
