# TIER-1 DESIGN v2 — the revised architecture for the 1.0× TIE (STEP-C deliverable)

Charter: plans/step-c-revised-design.md. This REVISES plans/tier1-design-and-tier2-proof.md
with the TIER-2-PROVEN, advisor-vetted ceilings. It is the gate to TIER-3.

Bar: **1.0× TIE** of gzippy-native vs rapidgzip, T8 silesia-large parallel single-member
decode, locked guest, interleaved best-of-N≥7, sha-verified. Pure-Rust + inline-ASM allowed.

## 0. What TIER-2 PROVED (the inputs this design is built on — all advisor-vetted)

| quantity (locked guest, T8, silesia-large) | value | source |
|---|---|---|
| gzippy baseline T8 wall | 1.124s (≈2.08× rg) | A.2 oracle, orchestrator-status:852 |
| rapidgzip T8 wall | 0.5396s | A.2 oracle |
| **placement-perfect floor** (ramp-consistent) | **0.56–0.66s** (+7–26%) | Oracle-P, step-a-advisor-verdict |
| **engine clean-only ceiling** (publish-chain intact, byte-exact) | **0.6134s** (+13.7%, 3.47σ) | A.2 oracle, step-a2-advisor-verdict |
| gzippy clean per-chunk rate | **92.7 ms** | A.2 seeded trace |
| rapidgzip clean per-chunk rate | **39 ms** | step-a advisor (1297/33) |
| per-chunk clean-rate gap | **2.38×** | A.2 |

THREE proven structural facts that FORCE this design:
1. **Placement is NECESSARY-BUT-INSUFFICIENT.** Even perfect boundary placement lands
   gzippy at 0.56–0.66s — still 7–26% over rapidgzip. (The earlier "placement sufficient"
   verdict was a floor-vs-wall mismatch; struck by the step-a advisor.)
2. **Engine residual SURVIVES perfect placement.** Forcing every chunk clean (free windows,
   ZERO marker-resolve pass — a gzippy *coordination advantage*) still loses 13.7% at 3.47σ.
   So the engine must reach ~igzip-class clean rate. This is a CONSERVATIVE lower bound.
3. **PLACEMENT and ENGINE are STRUCTURALLY COUPLED, and the order is FORCED.** Clean decode
   is only possible at a CONFIRMED deflate boundary with its predecessor window — i.e.
   confirmed-boundary dispatch IS the placement fix. The A.2 oracle could only isolate the
   engine by FIRST pre-seeding confirmed boundaries + windows. So: **placement first (it is
   the prerequisite that lets the engine run clean), then engine.** They are CO-PRIMARY
   (both ~13% gaps, both required for the tie), not primary/secondary.

NOTE on what this design DROPS from v1: v1's lever-class PRE-GATE (PG-A/PG-B) and the
PROOF-0/1/2 sequence were the TIER-2 PROVE machinery — that phase is now COMPLETE (the
pre-gate ran: compute is on the critical path but ~11–29% at the current placement-bottle-
necked operating point; the removal oracles ran: both ceilings landed). The remaining
1b (u8-write A/B) is folded into the ENGINE section §2 as a named sub-technique to measure
during TIER-3, not a gate. The governing-tension resolution (§1.1 of v1: one no-FFI engine,
inner loop reimplemented igzip-class via pure-Rust+inline-ASM) STANDS and is the spine of §2.

---

## 1. PLACEMENT (first; FAITHFUL PORT — architecture, NO innovation)

The placement fix recovers parallel efficiency (T8 runs ~42%) AND is the prerequisite that
lets workers decode at confirmed boundaries = clean engine decode. Per the faithful-port
mandate ([[feedback_bias_guardrails]], CLAUDE.md rule 6) this section TRANSLITERATES
rapidgzip; it introduces NO novel scheduling. Vendor source mapped read-only this session
(plans/ ← /tmp/scheduler-vendor-map.md, all lines verified against the files).

### 1.1 Port status of the 5 scheduler mechanisms (two-column, source-cited)

Vendor `V/rg/` = `vendor/rapidgzip/librapidarchive/src/rapidgzip/`,
`V/core/` = `.../src/core/`. gzippy `G/` = `src/decompress/parallel/`.

| # | mechanism | vendor file:line | gzippy file:line | STATUS |
|---|---|---|---|---|
| 1 | Prefetch strategy (which indexes, in-flight/cache tracking) | `V/core/Prefetcher.hpp:88-336` (FetchNextAdaptive, FetchMultiStream); `V/core/BlockFetcher.hpp:458-571` prefetchNewBlocks; `:131,405-450` in-flight map | `G/prefetcher.rs:83-400`; `G/block_fetcher.rs:676-960`; `:66,536-538,996-1026` | **FAITHFUL** (only model swap `std::future`→`mpsc::Receiver`) |
| 2 | Confirmed-boundary dispatch (next prefetch targets the CONFIRMED offset) | `V/rg/GzipBlockFinder.hpp:105-158` insert/get; `V/rg/GzipChunkFetcher.hpp:357,371-375` appendSubchunksToIndexes → `m_blockFinder->insert(off+size)`; `:591-654` getBlock | `G/gzip_block_finder.rs:147-205`; `G/chunk_fetcher.rs:2718-2761`; consumer `:1264-1396` | **PORTED but DEFEATED by missing #3** |
| 3 | **Interior/subchunk reuse** (reuse a guess-decoded chunk at a CONFIRMED interior boundary) | `V/rg/GzipChunkFetcher.hpp:206-225` get→findDataOffset→getIndexedChunk; **`:254-309` getIndexedChunk** (read `m_unsplitBlocks`, fetch parent from cache, return `(decodedOffsetInBytes, parentChunk)`); `:377-396` write `m_unsplitBlocks` | `G/block_map.rs:135` find_data_offset (EXISTS, never called); `G/chunk_fetcher.rs:2752` unsplit_blocks (WRITTEN, never read); `:108-130,546` UnsplitBlocks | **MISSING — the prime defect** |
| 4 | Window propagation (worker N tail-window → N+1 predecessor; published BEFORE hand-off) | `V/rg/GzipChunkFetcher.hpp:553-575` queueChunkForPostProcessing (main-thread emplace last window); `:429-458` per-subchunk; `V/rg/WindowMap.hpp:39-90` | `G/chunk_fetcher.rs:1479-1602` publish_end_window_before_post_process; `:2766-2784` publish_subchunk_windows; `G/window_map.rs` | **FAITHFUL** |
| 5 | Consumer overlap (pull in order; JOIN an in-flight decode, not cold-start) | `V/rg/ParallelGzipReader.hpp:554-646` read loop; `V/core/BlockFetcher.hpp:263-329` get (join in-flight + 1ms prefetch pump) | `G/chunk_fetcher.rs:1009-1730` consumer_loop; `G/block_fetcher.rs:304-419` get_with_prefetch; `:223-269` try_take_prefetched_pumping | **FAITHFUL machinery; breaks only in the #2/#3 case** |

### 1.2 The single missing piece — port `getIndexedChunk` (interior reuse)

The defect ([[project_confirmed_offset_prefetch_gap]], ~318ms consumer lag / ~40% of T8
wall) is NOT a prefetch-scheduling bug (3 prior consumer-confirmation prefetch attempts all
regressed and were reverted at 43f1685 — confirmation comes only ~1 chunk ahead, too short
a lead). The root cause, now precisely located: a chunk decoded from a partition-GUESS
offset overshoots the next partition boundary; when the consumer later wants a boundary
INTERIOR to that guess chunk's `[encoded, maxEncoded]` range, gzippy CANNOT reuse it —
`matches_encoded_offset` accepts the range, but there is no path to emit `[confirmed, end]`
from the parent chunk's interior. So the consumer rejects (`PREFETCH_REJECT_BY_GUARD`,
`G/chunk_fetcher.rs:1307`) and synchronously COLD re-decodes (`get_with_prefetch` →
`wait.block_fetcher_get`, `:1366-1394`). That blocking cold decode IS the lag.

rapidgzip never pays this: `getIndexedChunk` (`V/rg/GzipChunkFetcher.hpp:254-309`) looks up
`m_unsplitBlocks[blockOffset]` → fetches the PARENT chunk from `BaseType::cache()` → verifies
`blockOffset ∈ [parent.encoded, parent.encoded+parent.encodedSize)` → returns
`(decodedOffsetInBytes, parentChunk)`. The consumer entry `get(decodedOffset)`
(`:206-225`) calls `m_blockMap->findDataOffset(offset)` FIRST, reusing the already-decoded
parent at its interior offset — no cold re-decode, overlap preserved.

**THE PORT (the ONE structural change):** add an interior-reuse front-door to gzippy's
`consumer_loop` that, BEFORE the inline `processNextChunk`/`getBlock` body, fetches an
already-decoded PARENT chunk and emits its `[confirmed, end]` interior — transliterating
`V/rg/GzipChunkFetcher.hpp:254-309` (`getIndexedChunk`). The data needed for the interior
emit EXISTS: `Subchunk` records `encoded_offset_bits` / `decoded_offset` / `decoded_size`
(`G/chunk_data.rs:34-46`), so a parent that overshot CAN emit `[confirmed, end]`. (Memory-
note correction this session: gzippy's `matches_encoded_offset` IS a true range check
`encoded ≤ off ≤ maxEncoded`, `G/chunk_data.rs:461-469`, vendor `ChunkData.hpp:396-403` —
the prior memory claim of `== decode_start` equality was STALE.)

**THREE port pre-conditions the advisor surfaced — resolve BEFORE building (not "just wire
two dead structures"):**
1. **COORDINATE SYSTEM.** Vendor's `findDataOffset`/`getIndexedChunk` entry is keyed by
   DECODED-BYTE offset (the `get(decodedOffset)` consumer). gzippy's `consumer_loop` is
   ENCODED-BIT-keyed (`furthest_decoded_bit`, `G/chunk_fetcher.rs:1029`; `next_block_offset`).
   `block_map.find_data_offset` (`G/block_map.rs:135`) and `get_encoded_offset` (`:152`,
   exact-match only) do NOT serve an encoded-RANGE-CONTAINING lookup. So the faithful port
   is NOT "call find_data_offset" — it is build the encoded-bit interior-CONTAINING reuse
   keyed the way gzippy's consumer already requests (its own offset space), mirroring
   `getIndexedChunk`'s LOGIC (parent lookup → range verify → return at interior offset)
   without changing the consumer's request model (that would be a forbidden consumer
   redesign). This keeps it a faithful logic-port, not an architecture change.
2. **MAP POPULATION.** `unsplit_blocks` is written ONLY when `subchunks.len() > 1`
   (`G/chunk_fetcher.rs:2736-2758`). A single-subchunk overshoot parent (the common case)
   writes NO entry — the port must also index single-subchunk parents (or reuse via the
   parent's own range directly).
3. **PARENT-CACHED PRECONDITION (the deepest — answer FIRST).** getIndexedChunk fetches the
   parent from `BlockFetcher::cache()` (cap = max(16, pool)=16 @ T8, `:527-528`). The cited
   memory MEASURED the parent EVICTED before the lagging consumer reaches it (the ~318ms lag
   → eviction → cold re-decode), and left a PRE-REGISTERED, UNANSWERED discriminator: "is
   the containing partition chunk in-flight/cached at the gzippy stall? YES ⇒ interior reuse
   IS the fix; NO ⇒ deeper consumer-throughput gap (gzippy discards what rapidgzip keeps)."
   If NO, there is a chicken-and-egg: reuse needs the parent cached, but the parent is
   evicted BECAUSE the consumer lags. **TIER-3 placement step 0 = answer this discriminator
   (byte-exact probe) before writing the port.** If NO, the placement fix is upstream of
   interior reuse (cache residency / consumer pace) and the design's §1 recipe is amended.

### 1.3 Placement gate (TIER-3)
- Byte-exact (sha 028bd002…cb410f) every step; dual-sha native + isal.
- The STALL probe ([[project_confirmed_offset_prefetch_gap]]: count should drop 4→1).
- Locked-Fulcrum A/B: `consumer.wait.block_fetcher_get` should fall toward rapidgzip's
  overlap profile (rapidgzip 11/12 waits IN-FLIGHT vs gzippy 4/4 COLD). Target wall after
  placement-only ≈ the Oracle-P floor 0.56–0.66s.
- FALSIFIER (pre-register before TIER-3 placement work): if the interior-reuse port does
  NOT drop the stall count AND the wall does NOT move toward 0.66s, the defect is deeper
  than getIndexedChunk (re-open the diagnosis; do not patch-and-pray). A correct byte-exact
  port that ties is KEPT even if the wall only partially moves (CLAUDE.md rule 7a).

---

## 2. ENGINE (co-primary; inner-loop OPEN TERRITORY — pure-Rust + inline-ASM authorized)

Goal: close the **2.38× per-chunk clean gap** (92.7ms → ~39ms) in the CLEAN-mode inner
loop. This is the explicitly-authorized "build the fastest possible raw Huffman decoder"
territory (CLAUDE.md "Permission to fully reimplement the inner inflate"; inner loop is NOT
under the faithful-port rule — only architecture is). The clean mode is where ~94–99% of
bytes are (`CONTAINS_MARKERS=false` arm of `read_internal_compressed_specialized`,
`G/marker_inflate.rs`); the markered ≤32 KiB prefix stays scalar (low volume, faithful).

### 2.1 Why pure-Rust+ASM and not FFI (governing-tension resolution — STANDS from v1 §1.1)
The faithful rapidgzip structure hands its clean tail to igzip C/SIMD (FFI). The governing
memory [[project_faithful_unified_decoder_over_perf]] mandates ONE MarkerRing engine,
flip-in-place, NO 2nd engine, NO C-FFI on the decode graph — and the fold (8cfad3a) did
exactly that. To TIE without FFI, the ONE engine's clean inner loop must be reimplemented to
igzip-class via pure-Rust + inline-ASM. This ACCEPTS inner-loop divergence (our-ASM vs
igzip-asm) while the ARCHITECTURE stays faithful (chunk lifecycle / block finder / window
map / marker resolve all faithful per parity-final.md + §1). Consistent with CLAUDE.md's
scoping. **Requires supervisor ratification** (it formally accepts the inner-loop divergence
as the only charter-legal path to BOTH the 1.0× bar and the no-FFI/one-engine memory).

### 2.2 The concrete igzip-class techniques (each isolation-benchmarked before integration)

Storage width is FAITHFUL and STAYS: `output_ring: Box<[u16; RING_SIZE]>`
(`G/marker_inflate.rs:290`) is a byte-for-byte port of vendor `m_window16`
(`deflate.hpp:805`). The gap is COMPUTE + WRITE-TRAFFIC in the clean inner loop, NOT
storage width (vendor refutes width; do not shrink the ring). Techniques, in the order
they will be PROVEN (PROOF-1-style isolation bench, see §2.3) before any integration:

**(E1) u8-direct clean WRITE (the 1b sub-lever — LOW-RISK, first).**
Today the clean path writes u16 per literal into `output_ring`, then narrows u16→u8 at
drain (`push_clean_u8`, `G/marker_inflate.rs:42,743-750`). The 5256075 u8-direct work was
at the DRAIN/TAIL level; the post-flip *ring-store* path was never made u8-direct (per
[[project_faithful_unified_decoder_over_perf]] "u8-direct post-flip path was NEVER ported"
into the ring backing). E1 = port vendor's in-place flip so clean-mode literals store u8
DIRECTLY into the u8-reinterpreted ring (vendor `setInitialWindow` value-downcast +
`reinterpret_cast` u8 view, `deflate.hpp:1245,1742-1785,806`) — halves clean-path write
traffic on ~100% of bytes. Faithful to vendor's own one-engine memory model AND a compute
win. Top trap (from the faithful memory): post-flip max-distance-32768 back-ref across the
repositioned seam — needs the existing adversarial seam unit test (A‖A + dist-32768).

**(E2) Wide SIMD back-ref COPY.**
Back-refs copy mean ~6–7 bytes today via scalar `copy_nonoverlapping`/per-byte
(`emit_backref_ring`, `G/marker_inflate.rs:1429,1594`). igzip uses overlapping SIMD /
`rep movs`-class copy. Port a vectorized overlapping copy for the clean (`CONTAINS_MARKERS
=false`) ring path. HAZARD (documented 05a3835): a rounded word-copy corrupts repetitive
data via circular-ring overshoot — the SIMD copy must respect the ring wrap + overlap
exactly (per-byte tail for the overshoot region). Inline ASM where Rust autovectorization
lags.

**(E3) Packed multi-literal WRITE.**
The live native loop ALREADY has a 2-/3-literal speculative chain
(`run_multi_cached_loop`, `G/marker_inflate.rs:1782-1841`, per the bmi2 falsifier finding).
E3 = collapse the decoded literals into ONE wide store in clean mode (the historically-
regressing ca52389 path — NON-binding per charter, re-measure against the post-PRELOAD/post-
fold loop). Pairs with E1 (u8 packing makes a wider store cheaper).

**(E4) Wide refill + branch-lean availability.**
The clean loop already has a branchless `refill_fast` (REFILL_THRESHOLD=48,
`G/marker_inflate.rs:1715-1731`). E4 = widen the refill and elide the resumable yield-check
tax when output has FASTLOOP_OUTPUT_MARGIN headroom, clean-mode only (markered mode keeps
per-iter refill for back-ref safety). This is the FASTLOOP-margin technique CLAUDE.md
authorizes.

**NOTE on already-rejected per-symbol-ALU levers (do NOT re-spend here):** BMI2 BZHI is
already compiled into the perf build (target-cpu=native) and a forced-off A/B was a TIE
(rejected with mechanism, orchestrator-status:507-535). PEXT has no scattered-field use to
exploit. So E1–E4 are TRAFFIC + COPY + STORE-WIDTH + REFILL — the memory-bandwidth/store
surface the A.2 oracle implicates — NOT another per-symbol extract op.

### 2.3 Engine PROOF gate (the existence proof — MANDATORY before TIER-3 integration)
Per CLAUDE.md rule 3 ("slow-down slope ≠ speed-up ceiling") and the step-a2 advisor's RULE-3
flag: the pre-gate licenses only "compute is not moot"; the SPEED-UP ceiling must be bounded
by a REMOVAL/isolation oracle before any multi-week SIMD build. The mechanism:

- **In-bench ISA-L positive control (charter-mandated).** A standalone bench (criterion or
  `#[bench]` binary, NOT the full pipeline) decodes a known-window CLEAN silesia chunk three
  ways on the SAME guest x86_64 CPU under the build lock: (i) gzippy's current clean loop
  (baseline), (ii) the E1–E4 loop, (iii) ISA-L `isal_inflate` itself as the upper-bound
  oracle (C lib called directly IN THE BENCH — a measurement oracle, FFI stays OFF the
  production decode graph). The infra already exists (the STEP-1b differential proved a
  byte-identical ISA-L tail is callable, 8d026a8). Byte-exact every variant; sha-gate.
- **Self-test:** variant (iii) ISA-L must read ~2× variant (i)'s scalar class single-thread
  (the GUEST RATIO, not the Frankensystem absolutes 337/720 — those are illegitimate guest
  targets per the step-a2 caveat). If the known ratio does not reproduce, the bench is
  broken — fix before trusting any (ii) number.
- **PASS criterion (guest-ratio):** (ii)/(iii) high enough that, fed the §3 model, the
  projected T8 wall ties rapidgzip. If (ii) plateaus near (i) (SIMD didn't materialize a
  real Rust win) and the residual is > spread, the engine front is NOT proven — report the
  achievable floor; do NOT start integration on an unproven inner loop.

---

## 3. REACHABILITY ARITHMETIC (coupled, NOT naive-additive)

The two levers are COUPLED, so you cannot multiply two independent speedups. The honest
model is: **placement converts the operating point from marker-rate to clean-rate; the
engine then sets the clean-rate.** Work it through:

- **Today:** T8 wall 1.124s. Decode is mostly the MARKER path (168 ms/chunk incl scan) at
  ~42% parallel efficiency.
- **After PLACEMENT alone** (interior reuse closes the cold-re-decode lag; every chunk
  decodes at a confirmed boundary with its predecessor window → CLEAN rate, no speculation
  premium): the operating point becomes gzippy's *current clean rate* 92.7 ms/chunk. That is
  exactly what the A.2 clean-only oracle measured: **0.6134s** (= 39 chunks × 92.7ms / 8 ×
  1.36 ramp). Oracle-P's 0.56–0.66s is the same floor from the makespan side. So
  placement-perfect ⇒ ~**0.61s** (+13.7%). **Necessary but NOT a tie.**
- **After PLACEMENT + ENGINE** (clean rate driven from 92.7 → ~39 ms/chunk, igzip-class):
  the SAME makespan arithmetic with the faster per-chunk rate ⇒ 39 chunks × ~39ms / 8 ×
  1.36 ≈ **0.26s of decode-bound wall** — i.e. decode is no longer the binding term; the
  wall re-binds on rapidgzip's OWN floor (pipeline fill / in-order consumer / DRAM), which
  is ~0.54s. So **placement + engine-at-igzip-class ⇒ projected ~0.54s = the TIE.**
  - This is why it is COUPLED not additive: placement alone gets you to the clean-rate
    operating point (0.61s); the engine then shrinks the clean rate until decode stops
    binding and the wall sits at the shared pipeline floor (~0.54s). Neither alone reaches
    it — placement leaves +13.7% engine residual; engine-without-placement can't run clean
    (structural coupling, A.2 finding).

**Residual risk (stated honestly):**
1. **The engine front is HIGH-RISK.** Matching igzip AVX2 in pure-Rust is the hard part.
   Vendor's own bench: ISA-L is 2.1× the best PURE decoder single-thread — a pure SCALAR
   loop physically cannot reach igzip; the win must come from real SIMD/ASM (E2 copy + E3
   packed store + E4 refill) materializing in Rust codegen. If E1–E4 plateau near the ~337
   pure ceiling rather than ~720 igzip, the clean rate lands at ~50–60 ms not ~39 ms, the
   decode-bound wall lands at ~0.34–0.40s, and the TOTAL wall at ~0.54–0.60s — a NARROW
   miss, possibly a TIE within spread, possibly +5–10%. The §2.3 isolation bench is what
   resolves this BEFORE committing the multi-week build.
2. **The ramp (1.36) may not hold post-placement.** It is descriptive (actual/makespan on
   the current pipeline). If interior reuse changes the consumer overlap profile, the ramp
   could improve (helping) or the in-order consumer could become the new binding term
   (~0.54s rg floor is itself ramp-laden). The §1.3 Fulcrum A/B re-measures it.
3. **DRAM bandwidth at T8** could re-bind before the engine reaches 39ms (vendor's
   multi-thread gap shrinks to 1.28× as bandwidth masks compute). E1 (u8 writes, half the
   clean traffic) directly attacks this; if the wall re-binds on bandwidth before the tie,
   that is a finding for the supervisor, not a silent miss.

**THE NON-DECODE FLOOR CAVEAT (advisor, load-bearing — read §3 as ≥0.54s, not =0.54s).**
The 0.61→0.54 step assumes gzippy's NON-decode pipeline floor EQUALS rapidgzip's 0.54s.
That equality is UNPROVEN and there is measured evidence against it: gzippy carries
~225ms of in-order consumer-serial bookkeeping (`window_publish_marker/get_last_window_vec
+ dispatch_post_process + queue_prefetched_postproc`, [[project_confirmed_offset_prefetch_gap]]
ROOT-CAUSE) that rapidgzip runs OFF the in-order path; and A.2's clean trace showed
`wait.block_fetcher_get = 0.497s` on the single consumer thread (with FREE pre-seeded
windows — a real run is ≥ that). Neither term shrinks when the engine speeds up — they are
floor terms surviving BOTH levers. So when the engine hits 39ms/chunk (decode-optimal
≈ 0.19s), the wall re-binds on GZIPPY's non-decode floor, which the evidence puts MATERIALLY
ABOVE 0.54s. **Therefore: placement+engine-maxed ⇒ ≥0.54s, plausibly 0.54–0.62s; the TIE
is reached only at the optimistic edge unless the consumer-serial work is moved off the
in-order path (a SEPARATE faithful structural item — vendor's consumer does window-publish/
apply-window off-critical-path).**

**OWED THIRD MEASUREMENT (advisor-mandated, before/with TIER-3 — sets the true floor).**
Decompose the placement-perfect 0.61s consumer block into DECODE-WAIT vs SERIAL-BOOKKEEPING
(the memory's own owed discriminator). If the non-decode SERIAL floor is > 0.54s, the engine
front is chasing a tie the consumer structurally forbids — that is a SUPERVISOR-LEVEL finding
(revisit the bar, OR add a 4th lever: move the serial consumer work off the in-order path,
faithfully mirroring rapidgzip). This measurement gates whether the engine build is worth it.

**Verdict on reachability:** the math says the tie is REACHABLE **conditional on TWO floors
holding** — (i) the engine reaches ~igzip-class clean rate (≤~39–45 ms/chunk; HIGH-RISK,
gated by §2.3's bench), AND (ii) gzippy's non-decode consumer-serial floor is ≤0.54s (the
owed third measurement; if not, a 4th off-critical-path lever is required). Placement's
LOGIC is faithfully portable, conditional on the parent-cached discriminator (§1.2) being
YES. **If the bench shows the clean loop plateaus OR the non-decode floor is >0.54s, the
1.0× bar is NOT reachable by this design as-stated and the supervisor must revisit the bar,
add the off-critical-path consumer lever, or accept FFI.** The measurement, not optimism,
decides — and we now have the three instruments (interior-reuse stall probe, engine
isolation bench, consumer-block decompose) to make that call before the multi-week build.

---

## 4. STRUCTURE mandate (sequenced into TIER-3, all byte-exact)

Per charter §STRUCTURE. After placement + engine land, or alongside as byte-neutral cleanup:
- **Subdir split:** move the two paths into `gzippy-isal` (faithful rapidgzip + FFI clean-
  tail, reference baseline) and `gzippy-native` (the one pure-Rust+ASM engine) subtrees.
  Today they are feature-gated in shared files; the dir split is the orientation aid the
  mandate wants (the native=marker_inflate vs isal=resumable confusion mis-sited the
  slow_knob — exactly the trap this prevents).
- **Dead-code removal (advisor-gate each):** the specialized_decode + SPEC_CACHE cluster
  (production-dead, coupled to test stats — the deferred coupled edit); `unified.rs`
  `HAS_BMI2=false` dead placeholder; fix the STALE `isal-compression` engine-A/B comment in
  `guest_fulcrum_capture.sh:69-71` (build.rs:90-96 proves it false post-fold).
- **Names describe behavior** (continue the naming-truth discipline; a technique that is ON
  is visibly ON; no dead consts reading as live gates).
- Keep all 850+ lib tests green; dual-sha 028bd002…cb410f each step.

---

## 5. SEQUENCE (TIER-3, after supervisor ratification — DO NOT START YET)
0. **TWO PRE-REGISTERED DISCRIMINATORS (cheap, byte-exact, run FIRST):**
   (a) the **parent-cached-at-stall** probe (§1.2 precond 3) — is the containing chunk
   cached/in-flight when the consumer stalls? Decides whether interior reuse is the fix or
   the gap is upstream (cache residency / consumer pace).
   (b) the **consumer-block decompose** (§3) — split the placement-perfect 0.61s consumer
   block into decode-wait vs serial-bookkeeping; sets gzippy's true non-decode floor. If
   >0.54s, surface to supervisor BEFORE the engine build (add the off-critical-path lever).
1. **PLACEMENT** (interior-reuse port, §1.2, conditional on 0(a)=YES) — byte-exact,
   stall-probe (count 4→1) + Fulcrum A/B gate. Re-measure: expect ~0.61s. Prerequisite for
   clean engine decode.
2. **ENGINE PROOF** (§2.3 isolation bench + ISA-L positive control) — bound the speed-up
   ceiling BEFORE the build. Go/no-go for §2.2 integration.
3. **ENGINE INTEGRATION** (E1→E2→E3→E4, each isolation-proven then wall-measured) — only if
   step 2 passes. E1 (u8-write) first (low-risk, also closes the 1b sub-lever).
4. **STRUCTURE** (§4) — byte-neutral, alongside.
5. PHASE-3 3-way locked Fulcrum (rapidgzip vs gzippy-isal vs gzippy-native) confirms the tie.

## 6. CHECKPOINT (STOP — TIER-3 gate)
Independent disproof advisor (read-only, synchronous) attacks: (a) the §3 reachability
arithmetic (is the coupled 0.61→0.54 step real, or does a term re-bind?); (b) the §1
faithful-port-vs-innovation boundary (is getIndexedChunk a faithful port or a redesign?);
(c) whether E1–E4 can plausibly hit igzip-class in pure-Rust. Verdict →
plans/step-c-design-advisor-verdict.md. Then STOP for supervisor ratification. NO TIER-3
implementation until authorized.
