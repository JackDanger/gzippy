# Disproof-advisor VERDICT — CORRECTED overlap oracle (GZIPPY_PERFECT_OVERLAP)

Independent, read-only. Source-verified first-hand against HEAD 7aae6c4a +
overlay. Convention: wall ratio = rg_wall / gzippy_wall; tie zone gzippy ≤ 0.137s
(1.05× rg 0.130).

## Bottom line up front

**C1 REFUTED (overclaim). C2 UPHELD-WITH-CAVEATS. C3 split: (a) faithfulness
UPHELD, (b) "no-headroom" interpretation REFUTED.** F1 **CANNOT be decided** by
this oracle, and the weight of evidence is that **F1 LIKELY HOLDS** (the T8 tie is
reachable in principle). The oracle is not a valid removal oracle — it runs
*slower* than the very baseline it is supposed to lower-bound, it conflates two
changes (dispatch-depth removal **plus** a 4096-entry retention blowup), and it
never touches the in-order drain/resolve term that its own WHY-table fingers as the
binder.

---

## Source verification (what the code actually does)

- **Dispatch is a flood, not the production schedule.** `perfect_overlap_warm`
  (chunk_fetcher.rs:2141-2198) loops over EVERY block index and calls
  `block_fetcher.submit_prefetch(part_key, rx)` directly (block_fetcher.rs:552-554
  = bare `prefetching.insert`). It **bypasses** `prefetch_new_blocks` and therefore
  the `threadPoolSaturated()` gate `prefetching_len()+1 >= parallelization`
  (block_fetcher.rs:737-755). Production caps in-flight prefetches at ≈ 8 (T8); the
  oracle parks **all ~17** in flight from t0.
- **Retention is unbounded.** chunk_fetcher.rs:541-544 bumps BOTH
  `cache_capacity` and `prefetch_capacity` to `max(4096)` "so the cache must hold
  ALL chunks without eviction." Production sizing is `max(16, pool)=16` cache + `2*pool=16`
  prefetch (chunk_fetcher.rs:533-534). So the oracle holds the full member's worth
  of decoded **markered (u16, 2×-width)** chunks live simultaneously; production
  holds ≈ 8.
- **Marker engine is genuinely kept (faithful).** Warm decodes go through
  `submit_decode_to_pool` → `run_decode_task` with `is_speculative_prefetch=true`.
  Window lookup is `window_map.get(params.start_bit)` (chunk_fetcher.rs:2425-2429);
  the map is NOT seeded (drive_impl builds a fresh `CompressionType::None` map,
  :482-486) and partition-GUESS start bits are not real boundaries, so the lookup
  misses → `decode_mode_clean=false` → window-absent marker bootstrap
  (`decode_chunk_unified_marker`, gzip_chunk.rs:824-826). Only start_bit==0 takes
  the clean path, exactly as in production. **No seedfull-style clean flip.**
- **Keying is faithful.** submit side keys on
  `block_finder.partition_offset_containing_offset(start)` (chunk_fetcher.rs:2151);
  the consumer queries the same key via `partition_offset_for(&next_block_offset)` →
  `try_take_prefetched_pumping` → `take_prefetch` = `prefetching.remove(key)`
  (block_fetcher.rs:536-538, 1336-1355). Acceptance still gated on
  `matches_encoded_offset` (chunk_fetcher.rs:1360), identical to production.
- **The binder is real and on the consumer's critical path.** The in-order drain
  waits on the decode future (`get_with_prefetch` futureWait, chunk_fetcher.rs:1517;
  `ttp.rx_recv_block`, block_fetcher.rs:247-268) and on the in-order pending-write
  queue; `apply_window`/resolve runs on the pool at priority −1 but the consumer
  must still await it in output order.

So the byte-exactness, marker-engine-kept, and same-keys parts of C3 are **true**.
The disagreement is entirely about what the wall number *means*.

---

## C1 (DECIDER) — REFUTED as stated

Claim: the corrected oracle REFUTES F1 ("the T8 tie is reachable via decode↔drain
overlap / earlier dispatch"); flooding all chunks at t0 does not collapse the wall.

Three independent reasons it does not refute F1:

1. **An oracle that runs slower than its baseline is not a lower bound.** A removal
   oracle's job (CLAUDE.md PROCESS rule 3) is to REMOVE a term and measure the
   *floor* — the result must be ≤ baseline or the removal mechanism injected a new
   cost. perfovl interleaved best 0.186-0.189s is **above** production 0.176-0.177s.
   You cannot bound "how far earlier-dispatch can collapse the wall" with a
   configuration that *adds* wall. The number tells us nothing about the floor.

2. **The slowdown is consistent with a self-inflicted artifact, and the
   decode-phase numbers positively contradict the "no headroom" reading.** The
   oracle changes TWO things: (a) dispatch-all-at-t0 (intended) and (b) cache caps
   → 4096 with no eviction (unintended). At T8 that holds ≈17 markered u16 chunks
   live vs production's ≈8 — for silesia (~200 MB out / 17 ≈ 12 MB/chunk × 2 for
   u16 ≈ 24 MB/chunk) that is ~400 MB resident vs ~190 MB, a ~200 MB working-set
   inflation on a memory-bandwidth-bound decode. A 5-7% wall regression is exactly
   what LLC/bandwidth pressure of that magnitude produces. Crucially, the WHY-table
   shows the decode phase **improved** (Pool Fill 74.8→79.9%, Real Decode
   0.140→0.125, −11%). "Earlier dispatch recovered measurable decode-phase headroom"
   is the OPPOSITE of "production already saturates the pool / no headroom." The
   wall got worse while the thing the oracle targeted got better — that is the
   signature of a confound, not a clean negative.

3. **The vendor existence proof stands against the claim (CLAUDE.md PROCESS rule
   7).** rapidgzip carries the SAME u16 marker machinery and pays a measured
   ~0.113s apply-window/resolve pass (CLAUDE.md), yet ties at 0.130s — i.e. rg
   *hides* most of decode+resolve under overlap. If the same decode+resolve budget
   fits in 0.130s on the vendor, there is no structural reason it cannot on gzippy.
   To reject F1 you must supply a mechanism AND explain how rg does it differently;
   the brief supplies neither, and rg does the *same* thing. F1 is not refuted —
   it is relocated to a term this oracle never touched.

What the oracle DOES legitimately show: the **dispatch-DEPTH / prefetch-TIMING
sub-lever alone** (project_confirmed_offset_prefetch_gap) is small at T8 — earlier
dispatch buys only ~+5pp fill / −11% decode-phase and does not move the binder.
That is a real, useful negative on ONE sub-lever. It is not a refutation of F1.

## C2 (the binder) — UPHELD-WITH-CAVEATS

Claim: the binder is the in-order consumer wait (std::future::get ~0.10s, unmoved),
gated by each chunk's marker-engine decode time; drain is co-equal with decode and
does NOT hide; the 0.117s warm-alone lower bound was real but unachievable.

- **Upheld:** std::future::get / the in-order consumer wait IS the wall-gating
  serial wait, and it barely moved (0.1038→0.1024). That the consumer spends ~0.10s
  *waiting on decodes* means the consumer out-runs pool supply — decode throughput,
  not consumer-side write, is the proximate binder. Earlier dispatch can't help
  once the pool is near-saturated and the chunk count (17) is not a clean multiple
  of T (8) — the tail wave granularity, not dispatch starvation, owns most of the
  remaining 20% idle.
- **Caveat that inverts the conclusion:** C2 concedes drain/resolve is **co-equal
  with decode and does not hide**. That is precisely the UNREMOVED, UNTESTED lever —
  not a dead end. rg hides its 0.113s apply-window under decode; gzippy apparently
  does not. The right reading of "the binder is decode+unhidden-resolve" is "the
  open lever is resolve-ahead / off-consumer drain," not "the wall is stuck."
- **Reconciliation with seedfull (angle 2):** seedfull (clean engine) ties; here
  the marker engine is kept and overlap doesn't move the wall. Together these say
  the ENGINE (marker decode + its resolve tax) is co-primary with scheduling, NOT
  that scheduling is exhausted. Consistent with
  [[project_pregate_placement_is_dominant_lever]] (clean-rate 2.3× gap survives
  perfect placement). This does not reverse the engine's importance — it confirms
  it — but it argues AGAINST C1, because the lever it points to (hide resolve / use
  the clean engine) is real and vendor-demonstrated.

## C3 (faithfulness) — SPLIT

- **(a) "faithful overlap; marker engine kept; same keys; byte-exact" — UPHELD.**
  Verified above: window-absent marker decode for all non-zero chunks, vendor
  `m_prefetching.emplace` key parity, `matches_encoded_offset` acceptance retained,
  no window seeding. The brief's flip_to_clean=12 identical in head/perfovl is
  consistent with the code.
- **(b) "slowness is NOT an over-dispatch artifact but a real no-headroom signal" —
  REFUTED.** The 4096 retention cap + saturation-gate bypass are concrete,
  source-confirmed artifacts (chunk_fetcher.rs:541-544 vs block_fetcher.rs:737),
  and the +5pp fill / −11% Real Decode directly contradict "already saturates."
  The honest statement is: *production already extracts most of the dispatch-depth
  headroom; the residual wall lives in granularity + unhidden resolve, and the
  oracle's retention blowup masks even the small dispatch-depth win.*

---

## Single most load-bearing correction

**This is not a valid removal oracle, so its wall cannot decide F1.** It bundles the
intended term-2 removal (dispatch-all-at-t0) with an unintended 4096-entry,
no-eviction retention blowup that holds the whole member's u16 markered working set
resident (~2× the bytes, ~2× the live chunk count vs production), and it leaves the
in-order drain/resolve term — which C2 itself names as co-equal-and-unhidden — fully
intact. The result runs *slower* than the baseline it is meant to lower-bound while
the decode phase it targets gets *faster*: the classic confound signature. Combined
with the vendor existence proof (rapidgzip pays the same marker decode + ~0.113s
apply-window and ties at 0.130s), the tie remains reachable in principle, so this
oracle refutes only the narrow dispatch-depth sub-lever, not F1.

## Can F1 be DECIDED now? — NO. Direction: F1 LIKELY HOLDS.

F1 ("T8 tie reachable via decode↔drain overlap") is **not decidable** from this
oracle. It tested one sub-lever (dispatch depth), is confounded by retention blowup,
and never tested the sub-lever its own analysis fingers (resolve-ahead / off-consumer
drain hiding under decode). The vendor — same machinery, ties — is the standing
existence proof that the tie is reachable.

The clean experiment that COULD decide F1 (and the bar for any future REJECT under
rule 7):

1. **Bounded retention.** Keep vendor cache sizing (`max(16,pool)` + `2*pool`); do
   NOT bump to 4096. If the oracle must hold more, hold it as resolved u8, not u16.
2. **Metered dispatch.** Drive via `prefetch_new_blocks` honoring the saturation
   gate (or a small fixed depth ≈ 2T), so the schedule is production-shaped, not a
   flood. Isolate dispatch-depth as ONE factor at a time.
3. **Test the actual open lever: drain-hiding.** Push resolve+write off the in-order
   consumer's critical path (resolve-ahead on the pool for successors with a
   published predecessor window) and measure whether the consumer wait collapses
   toward the decode÷T floor.
4. **Finer chunking** so chunk_count ≫ T and the pool stays saturated through the
   tail (removes the 17-not-divisible-by-8 granularity loss).

Measure THAT wall, interleaved + sha-verified, vs rg. If a bounded-retention,
metered, drain-hiding, finely-chunked schedule still cannot reach 0.137s, THEN F1 is
refuted with a mechanism. Until then it stands, and the corrected oracle's number is
void as a decider.

---

# FOLLOW-UP VERDICT (resolve-ahead saturated)

Read-only, source-verified against HEAD 7aae6c4a + overlay. The follow-up data
(brief lines 108-136) knocks out the two legs my prior rescue of F1 stood on. I
change my position on the overlap/drain flavor of F1. Adversarial review below; the
direction now points at the engine, with ONE scheduling lever still genuinely owed.

## (1) Does resolve-ahead ok=14/14 refute the "untested drain-hiding lever"? — YES (the overlap rescue is dead), but the 14/14 number is the weakest part of the case.

**The ratio is a tautology, not a saturation measurement.** I source-verified the
counter wiring:
- `chunk_may_resolve_markers_early` (chunk_fetcher.rs:2671-2677) gates eligibility on
  `window_map.get(chunk_consumer_handoff_bit(chunk)).is_some()`.
- In `queue_prefetched_marker_postprocess`, an ineligible chunk is `continue`'d at
  :2747 **before** `RESOLVE_AHEAD_ATTEMPTS` is incremented at :2771.
- `confirmed_predecessor_window` (:2650-2655) then performs the **same**
  `window_map.get(handoff_bit)` lookup that the pre-screen already proved `is_some()`,
  so `RESOLVE_AHEAD_OK` (:2777) equals `ATTEMPTS` by construction (modulo a negligible
  TOCTOU window). The in-code comment at :2765-2770 says so outright: "OK tracks
  ATTEMPTS." So `ok=14/14=100%` means "of the chunks that already passed a
  window-presence screen, all had a window present" — it does **not** mean "100% of
  the resolvable drain was hidden." Honest coverage is the eager-submit count:
  **14/17 ≈ 82%**, not 100%. Do not cite 14/14 as a saturation proof; it cannot
  over-saturate.

**But the rescue still falls, for two source-backed reasons that do NOT depend on the
ratio:**
1. **The lever is no longer untested.** The same comment (:2765-2770) records that
   `RESOLVE_AHEAD_*`/`HANDOFF_WINDOW_PUBLISHED` were previously *dead counters* (the
   "Worker resolve-ahead" verbose line at :1039-1043 always read 0). They are now
   **live on the production resolve-ahead path** (`queue_prefetched_marker_postprocess`
   is called from the consumer at :1499, off the blocking `get_with_prefetch` at
   :1516). My prior verdict named "resolve-ahead / off-consumer drain" as the open,
   UNREMOVED lever (clean-experiment #3). It is now exercised at ~82% coverage and the
   wall did not move. That is exactly the test I asked for, run.
2. **The retention confound — the OTHER leg of my prior refutation — is removed and
   the oracle is STILL slow.** My prior C1-refutation leaned heavily on the
   4096-entry no-eviction blowup (chunk_fetcher.rs:541-544) inflating the u16 working
   set. The follow-up removed that bump (vendor sizing kept) and perfovl is *still*
   0.684-0.693× rg (0.187-0.192s), slower than production. So the slowness is **not**
   the retention artifact I blamed; it is a real null on the dispatch-flood.

**Consumer-wait location confirmed.** The blocking wait (`wait.block_fetcher_get`,
:1509-1516) is on `get_with_prefetch`, whose miss-closure submits the **decode** task
(:1519). Resolve-ahead is submitted earlier at :1499 and runs on the pool. So
"std::future::get is a wait on the DECODE future, not on resolve" is source-correct:
with resolve hidden, the residual consumer wait is decode time.

**Verdict on (1): the OVERLAP / drain-hiding flavor of F1 is REFUTED.** Both legs of
my prior rescue (retention confound; untested resolve-ahead) are gone. I withdraw
"F1 likely holds *via overlap*." What survives of F1 is a different, untested lever —
see (2).

## (2) Is the residual ~0.73× gap correctly relocated to the per-thread marker-engine DECODE RATE? — DIRECTION UPHELD; MAGNITUDE UNVERIFIED; ONE scheduling lever NOT yet excluded.

**Direction holds on gzippy-internal numbers alone** (no rg trace needed):
decodeBlock SUM 0.827s ÷ 8 = 0.103s is the decode floor, already ~79% of rg's 0.130s
WALL. A maximal-overlap oracle (all chunks dispatched at t0) could not push the wall
below production, so scheduling has little headroom left above that floor. To land
*reliably* at/under 0.130 you must lower the floor → faster per-thread engine. That
is the correct relocation.

**Two adversarial caveats:**

- **MAGNITUDE is uncited.** "rg decodeBlock 0.50s / 1.6×" appears nowhere I can
  source-verify (read-only; CLAUDE.md only records rg's *0.113s apply-window* and
  *31.25% replaced markers*, not a decode-sum). The 1.6× is plausible and matches the
  arithmetic below, but treat it as a hypothesis pending a real traced-rg decode-sum,
  not a measured fact.

- **A scheduling lever IS still live: tail-wave quantization (finer chunking).** This
  is the one item from my prior clean-experiment list (#4) the follow-up did NOT test,
  and the arithmetic says it is not small. 17 chunks on 8 threads is **3 waves**
  (8+8+1), so the realistic decode wall is `ceil(17/8) × avg_chunk ≈ 3 × (0.827/17) ≈
  3 × 0.0487 ≈ 0.146s` — NOT the 0.103s "÷8" floor. The last wave runs 1 chunk with 7
  threads idle. Finer chunks (chunk_count ≫ T, even last wave) would collapse 0.146 →
  ~0.103, a **~0.043s** scheduling gain — enough on its own to move gzippy from
  ~0.176 toward ~0.133, i.e. essentially the tie, with the SAME engine. The
  dispatch-flood oracle does NOT exclude this: flooding all chunks at t0 cannot fix an
  *uneven wave count*; it is a different lever. So the engine is **co-primary, not
  proven sole** — "relocated to the engine" is right that overlap is dead, but it
  over-reaches if read as "no scheduling lever remains."

So: residual gap = decode floor (engine) **+** tail-wave granularity (finer chunking),
with overlap/drain excluded. Engine is the larger and the vendor-demonstrated term;
finer chunking is the cheap untested one.

## (3) Is the faithful next step matching rg's u16 marker decode rate (readInternalCompressedMultiCached), NOT more overlap? — UPHELD as the highest-value faithful direction, with two corrections.

Per CLAUDE.md rule 7, rg is the existence proof and now the mechanism is identified:
rg ties at 0.130 carrying the *same* marker machinery, so if its decode-sum is ~1.6×
smaller the tie comes from a faster engine, not more overlap (rg's overlap headroom is
not larger than gzippy's — gzippy's floor is already near rg's wall). "Not more
overlap" is correct. Two corrections to keep the port faithful:

- **Correction A — "not overlap" ≠ "no scheduling." Run the finer-chunking test
  FIRST; it is cheap and arithmetic-significant (~0.04s, see (2)).** It is also a
  faithful lever (chunk-size is a vendor parameter), and it might close most of the
  gap without touching the inner loop. Excluding it is the bar rule 7 sets before
  declaring the engine the sole binder.

- **Correction B — target the RIGHT loop. The 1.6× may be the u16 width itself, not
  an inner-loop micro-gap.** The GOVERNING memory
  ([[project_faithful_unified_decoder_over_perf]]) is explicit: rg decodes the clean
  BULK **u8-direct** and uses the u16 marker width only where markers are actually
  unresolved; gzippy's all-u16 ring is the *shortcut/deviation*. `decodeBlock` here is
  the window-absent **marker** path (verified in my prior verdict: non-zero chunks
  miss the window map → `decode_chunk_unified_marker`), which is 2×-width by
  construction. So "match rg's u16 marker decode rate via
  readInternalCompressedMultiCached" is half right: readInternalCompressedMultiCached
  IS rg's marker loop and worth porting, but a large share of rg's decode-rate
  advantage likely comes from rg NOT running u16 over the clean bulk at all. Before
  attributing the 1.6× to the marker inner loop, source-diff gzippy's
  `decode_chunk_unified_marker` against vendor and confirm whether the gap is (i) the
  marker inner loop, (ii) u16-vs-u8 width on the clean bulk, or (iii) upstream (table
  build / bounds checks). The faithful target per the governing memory is the
  **u8-direct clean path**, not a faster u16 ring.

## Bottom line (follow-up)

I REVERSE my prior "F1 likely holds via overlap." With resolve-ahead live (~82%
coverage, and a tautological-but-directionally-clear 14/14) and the retention confound
removed leaving the flood still slow, **F1-via-overlap/drain is REFUTED**. The binder
relocates to the per-thread decode floor (engine) — UPHELD in direction, with the
1.6× magnitude UNVERIFIED and tail-wave finer-chunking (~0.04s) the one scheduling
lever still owed. The faithful next step is the engine, specifically the **u8-direct
clean path** the governing memory mandates (rg's real decode-rate source), with
readInternalCompressedMultiCached as the marker-path port — but run the cheap
finer-chunking test first so the engine is not credited with a granularity loss.
