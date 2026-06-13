# T4-trough vs T8-recovery — novel hypotheses + investigation design

Advisor strategy note. Read-only analysis. The verdict is always a causal
perturbation (CLAUDE.md Measurement PROCESS); everything below is a
hypothesis generator + the discriminator that would confirm/refute it.

## The puzzle (restated)

gzippy-isal vs rapidgzip 0.16.0, frozen guest, silesia, ~14 chunks:

| point | gz wall | rg wall | ratio (rg/gz) | note |
|-------|---------|---------|---------------|------|
| T1 single-shot (BYPASS) | — | — | **1.200 WIN** | hands straight to ISA-L, NO machinery |
| T4 | ~0.549s | ~0.498s | **0.906 TROUGH** | machinery in use |
| T8 | ~0.361s | ~0.374s | **1.038 recover** | machinery in use |

Scaling 4→8: **gz 1.52×, rg 1.33×** (ideal 2.0×). gz scales BETTER but
starts BEHIND. Engine kernel is byte-identical ISA-L (disasm-proven). The
T1-WIN is the BYPASS path, not T1-through-the-machinery.

## Load-bearing source facts established for THIS analysis

1. **silesia.gz = 68,229,982 B = 65.07 MiB compressed** (`/tmp/silesia.gz`).
2. gzippy chunk-size adjustment (`single_member.rs:75-88`, faithful port of
   `ParallelGzipReader.hpp:294-306`) fires only when
   `default_chunk(4MiB)·2·T > file_size`. For silesia that is `8·T > 65.07`
   → **T ≥ 9**. Therefore:
   - **T1–T8: chunk size is CONSTANT 4 MiB → chunk count CONSTANT (~14–17,
     pinned to real deflate boundaries near the 4 MiB spacing).**
   - T9+: chunk shrinks to ~2.5 MiB → count jumps to ~26.
3. Tail-wave quantization with a FIXED 14 chunks is **symmetric** across
   T4/T8: makespan/ideal = ceil(14/4)/(14/4) = 4/3.5 = 1.143 at T4, and
   ceil(14/8)/(14/8) = 2/1.75 = 1.143 at T8. **Identical.** ⇒ pure
   tail-wave imbalance does NOT explain T4<T8 here (REFUTED by arithmetic
   for the fixed-14 regime; see H4).
4. In-flight depth is faithfully ported: cache `max(16,P)`, prefetch cache
   `2P`, ≤ `P-1` prefetched + 1 on-demand in flight, single shared pool of
   `P` for decode AND post-process (`chunk_fetcher.rs:572-579, 624-631`;
   vendor `BlockFetcher.hpp:181-185, 453, 564`).
5. Production consumer (`consumer_loop`, `chunk_fetcher.rs:1148+`) is a
   single in-order `get`/prefetch loop, NO wave barrier (the wave barrier
   only ever existed in the oracle, since removed). ⇒ "production
   wave-barrier" is REFUTED as written; the in-order serial work lives in
   marker-resolve + window-publish on the consumer thread.

These facts kill the most obvious story (granularity switches at the
T4/T8 boundary — it does NOT, the switch is at T9) and force a model where
the SHAPE comes from a **fixed serial floor crossing a 1/T parallel term**.

---

## Ranked novel hypotheses (most-likely-to-explain-the-trough first)

### H1 (LEAD) — Amdahl crossover: gz has a SLOWER parallel decode phase but a LOWER serial floor than rg

Model each tool's wall as `wall(T) = S + W/T_eff` where `S` = the
T-independent serial floor (the in-order marker-resolve + window-publish
dependency chain, length = chunk count, run on the consumer thread) and
`W` = parallelizable decode work (the per-chunk window-absent / marker
decode, which uses the NON-ISA-L inner loop in BOTH tools).

Claim that fits EVERY number:
- gz **W is larger** (pure-Rust marker/inner loop slower per symbol than
  rg's C++ inner loop — the "2.3× clean-rate / inner-loop-ASM" gap in
  MEMORY). ⇒ at low T (decode-bound) gz loses → **T4 trough**.
- gz **S is smaller** (recent publish-chain-overlap + post-process-alloc
  cuts: commits 85ad00a, 0a40d5e, 99ff098, 0a3e9a3). ⇒ at high T
  (floor-bound) gz wins → **T8 1.038**.
- Two lines `S_gz + W_gz/T` and `S_rg + W_rg/T` with `W_gz>W_rg`,
  `S_gz<S_rg` cross exactly ONCE — between T4 and T8. This single model
  reproduces the trough AND the 1.52× vs 1.33× scaling (rg hits its higher
  floor sooner ⇒ worse 4→8 scaling; gz still partly decode-bound at T8 ⇒
  better scaling but only just overtakes).
- It also explains why the marker penalty looks "uniform per-chunk" yet is
  NOT uniform in WALL IMPACT: its impact is gated by whether the wall is
  decode-bound (low T) or floor-bound (high T).

**Discriminator (definitive):** run the full curve (below), then
least-squares fit `wall = S + W/T` for each tool over T1(forced-SM)…T9.
Confirm iff `W_gz > W_rg` AND `S_gz < S_rg` AND the fitted lines cross in
(4,8). Cross-check S directly: gz's serial floor = consumer-thread busy
time in marker-resolve+publish with all workers saturated (instrument
already present: `EARLY_WINDOW_PUBLISHED`, `HANDOFF_WINDOW_PUBLISHED`,
publish-chain counters). rg's floor = `--verbose` "decodeBlock" vs
"applyWindow"/post-process split and the CPU-utilization line ("X CPUs
utilized" < T ⇒ floor-bound). **Vendor source to check:**
`GzipChunkFetcher.hpp:447` (parallelization!=1 branch),
`waitForReplacedMarkers` (:497-511, the serial harvest the consumer does),
`BlockFetcher.hpp:111-112` (prefetch/on-demand stats),
`core/BlockFetcher.hpp:181-185` (one shared pool — the S/W coupling point).

### H2 — Shared-pool contention: post-process (apply_window/marker-replace, priority −1) steals a LARGER fraction of decode parallelism at T4

One pool of `P` serves both decode (prio 0) and post-process (prio −1)
(`chunk_fetcher.rs:624-631`, vendor `BlockFetcher.hpp:185`). A post-process
job occupies 1/P of the pool: **25% at T4, 12.5% at T8.** If gzippy's
apply_window/marker-replace post-process is heavier (or allocates more)
than rg's, it steals relatively 2× more decode throughput at T4 — directly
deepening the trough. This is distinct from H1: even with equal W, unequal
post-process WEIGHT × the 1/P occupancy fraction bends the curve.

**Discriminator:** instrument the pool's wall-time split decode-vs-postprocess
at each T (gz: tag `submit_decode_to_pool` vs the priority-−1 apply_window
submits; trace_v2 already classifies). Predict post-process share of pool
time falls ~linearly with 1/T and is HIGHER in gz than rg's
`applyWindow`/`postProcessChunk` share. Perturbation: artificially inflate
post-process time by a known factor (`GZIPPY_SLOW_*` knob, already wired)
and confirm the wall response is steeper at T4 than T8 (the 1/P leverage).
**Vendor source:** `GzipChunkFetcher.hpp` post-process submit + priority,
`ChunkData.hpp:447,881` (`split`), `BlockFetcher.hpp:185`.

### H3 — Prefetch-depth starvation: effective in-flight depth at T4 is below rg's, so decode workers idle on the window/consumer

At T4 vendor keeps ≤ `P-1`=3 prefetched + 1 on-demand = 4 in flight;
prefetch cache 2P=8. The consumer blocks on chunk i's future and pumps
`prefetch_new_blocks` every 1 ms (`chunk_fetcher.rs:1406-1417`). If gz's
real steady-state depth is < rg's (because the consumer is busy doing
serial marker-resolve and under-pumps the prefetcher, or the
`threadPoolSaturated` guard `m_prefetching.size()+1 >= P` trips earlier),
workers go idle — a low-T-specific starvation (at T8 there's enough
backlog to keep all busy). Note this is a SISTER of H1 (same consumer
serial work) but the symptom is worker IDLE, not consumer BUSY.

**Discriminator:** per-thread busy/idle fraction at each T (gz:
`stall_residency` / `statistics` modules; rg: `--verbose` CPUs-utilized).
Predict gz worker-idle fraction at T4 markedly > T8 AND > rg's at T4.
Perturbation: raise the prefetch pump frequency / depth and re-measure —
if the T4 wall drops and T8 doesn't, starvation confirmed. **Vendor:**
`BlockFetcher.hpp:453-474, 564-568` (≤P-1 prefetch, saturation guard),
`Prefetcher.hpp` FetchNextAdaptive (gz `prefetcher.rs:80-...`).

### H4 — Tail-wave / chunk-0-clean heterogeneity (LIKELY REFUTED, keep as control)

Fixed-14 quantization is symmetric (fact 3) so uniform tail-wave is out.
The only residue: chunk 0 is the lone guaranteed-CLEAN ISA-L decode (fast)
while 1..13 are slow marker decodes — a heterogeneous-cost schedule whose
worst-case packing differs at T4 vs T8. Arithmetic says the effect is
second-order vs H1.

**Discriminator:** per-chunk decode-time histogram (gz trace_v2 per-chunk
busy; already captured). If chunk-0 vs rest is ~uniform after the clean
head, H4 is dead. Keep only as the negative control for H1's "W is the
marker decode" premise.

### H5 — T9 granularity kink (NOT a T4 explanation; a curve-shape prediction to VALIDATE the model)

Because the adjustment fires at T≥9 (fact 2), chunk count jumps ~14→~26 at
T9. H1/H2 predict a visible SCALING KINK at T9 (more, smaller chunks =
more serial-chain links but finer load balance). If the curve shows a
discontinuity precisely at T9 for gz (and rg, same formula), it validates
that the chunk-count model — not noise — governs the shape, and confirms
the T1–T8 plateau in chunk count that makes H1's clean S+W/T fit valid.

---

## The structural gz-vs-rg difference that PRODUCES a T4 trough + T8 recovery

Not the engine kernel (identical) and not granularity (constant T1–T8).
It is the **(W large, S small) vs (W small, S large) inversion**: gzippy
traded parallel decode speed (slower pure-Rust marker/inner loop) for a
cheaper serial publish chain (the recent overlap/alloc commits). rapidgzip
is the opposite. Two such cost curves cross once, in the T4–T8 window —
trough then recovery is the generic signature of that crossover, and it is
self-consistent with the 1.52×-vs-1.33× scaling asymmetry.

If confirmed, the lever is unambiguous and matches MEMORY's standing
verdict: **close W (inner-loop / marker-decode symbol rate) — that is the
low-T headline.** S is already at-or-below rg.

---

## Full thread-count curve — measurement spec (primary discriminator)

Goal: wall, ratio_vs_rg, and scaling-vs-T1 for BOTH tools at
**T = 1,2,3,4,5,6,7,8,9**, silesia, to fit `S + W/T` and locate the
crossover + the T9 kink.

Rules (CLAUDE.md): frozen guest; build `--no-default-features --features
pure-rust-inflate`; assert `GZIPPY_DEBUG=1 → path=ParallelSM`; interleaved
best-of-N ≥ 7 (`scripts/measure.sh`, never a hand script — rule 8);
sha-verified output every cell.

CRITICAL controls:
- **Two T1 rows.** (a) T1 BYPASS (production single-shot, the 1.200 win)
  and (b) **T1 THROUGH the machinery** (`GZIPPY_FORCE_PARALLEL_SM=1`, the
  Amdahl intercept point). The fit uses (b); (a) only calibrates the
  bypass headroom. This is the single most important currently-missing
  cell — it tells us W vs S at zero parallelism.
- Pin/disable turbo or run the frequency-neutral check; the 4→8 turbo
  ramp can fake an Amdahl bend. Re-confirm the trough survives a SLEEP
  control on any injected slow-knob (rule 2).
- Record per-T: chunk_count (gz: `ADJUSTED_CHUNK_SIZE_APPLIED` + finder
  count; rg: `--verbose` "Total Fetched"), CPUs-utilized (rg `--verbose`),
  worker busy/idle (gz `statistics`/`stall_residency`), and the
  decode-vs-postprocess pool split (gz trace_v2). These feed H1/H2/H3
  discriminators in the SAME runs.

Extract & plot: ratio_vs_rg(T) — locate the trough; speedup_vs_T1(T) for
both — locate where slopes diverge; fitted (S,W) per tool — confirm
W_gz>W_rg ∧ S_gz<S_rg ∧ crossover∈(4,8); chunk_count(T) — confirm flat
1..8, jump at 9.

Per-hypothesis confirm/refute table:
- H1: lines cross in (4,8), W_gz>W_rg, S_gz<S_rg.
- H2: gz post-process pool-share > rg's AND falls ~1/T; slow-knob wall
  response steeper at T4 than T8.
- H3: gz worker-idle(T4) ≫ idle(T8) and ≫ rg(T4); prefetch-depth bump
  drops T4 wall only.
- H4: per-chunk decode times ~uniform after chunk 0 ⇒ dead.
- H5: discontinuity in both tools' curves exactly at T9.
