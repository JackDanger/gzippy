# Cache-residency mandate — MEASUREMENT VERDICT (2026-06-08)

Closes the DECISIVE goal gap: the cache-residency clause of
`plans/gzippy-native-design-mandate.md` was asserted but NEVER measured (the
byte-accounting instrument was hooked to the fold-deleted Engine-C staging box
and reported `threads=0` on native; `rss_vs_t.sh` did not exist; MPKI was never
run for native). This turn re-hooked the instrument to the REAL native
per-thread working set, built the measurement spine, and measured on the frozen
quiet guest (neurotic→REDACTED_IP, 16-core Raptor Lake i7-13700T, governor=
performance, no_turbo=1, runnable_avg=1.75, host-frozen). gzippy bin_sha
8a3524088d47fcd9, feature=gzippy-native, every decode sha-verified ==
028bd002…cb410f, path=ParallelSM.

## Instrument: re-hooked + VALIDATED (threads>0)
`mem_stats.rs` now hooks the native `BOOTSTRAP_BLOCK` (`marker_inflate::Block`)
via `on_block_active(block.heap_bytes())` in `marker_decode_step_vendor_block`
(gzip_chunk.rs) — the real native engine after the flip-in-place fold. Reports
`worker threads observed = 16` at T16 (was 0). Byte-transparent: OFF==identity
(0 stderr lines, identical sha with the flag on/off); 881 lib tests pass.

TWO native engines, by chunk type (a measured nuance, not a bug): the
instrument reports `threads=0` at T1 and `threads=8/16` at T8/16. This is
CORRECT — at T1 the single sequential chunk is WINDOW-SEEDED and decodes through
`StreamingInflateWrapper`/`ResumableInflate2` (a 32 KiB `SlidingWindow` +
per-call libdeflate `LitLenTable`/`DistTable` ≈ 20 KB → a ~52 KiB/thread
working set, L2-friendly), NOT the 278 KiB Block. The 278 KiB Block is the
WINDOW-ABSENT bootstrap/marker engine, which dominates at high T (most chunks
are speculatively bootstrapped from no window). So the mandate's "tiny per-thread
working set" is ALREADY MET on the window-seeded engine (~52 KiB) and is the
open question only on the bootstrap Block (278 KiB). The instrument is scoped to
the Block (the larger, mandate-relevant one); a follow-up could add a
`ResumableInflate2::heap_bytes()` hook to report both per chunk-type.

POSITIVE CONTROL (validates the RSS-vs-T mechanism, mandate rule "measure it,
don't assert it"): `GZIPPY_MEM_BALLAST_MIB=N` (now materialized on the NATIVE
hook, not the dead staging hook) recovers the per-thread slope at T16:
- 0→8 MiB: 16.16 threads-recovered/MiB
- 8→16 MiB: 15.80
- 16→32 MiB: 16.01
i.e. the incremental RSS slope == the 16 worker threads, linear. The instrument
resolves a KNOWN per-thread allocation. perf-stat control: ballast moved
cache-misses 1.23× and LLC-misses 1.09× monotonically up (perf instrument live).

## (a) PEAK RSS vs T (min-of-5, KiB) — gzippy-native vs rapidgzip
| T  | gzippy KiB | rapidgzip KiB | ratio g/rg |
|----|-----------:|--------------:|-----------:|
| 1  |    122,800 |        67,164 |       1.83 |
| 8  |    322,160 |       211,444 |       1.52 |
| 16 |    388,072 |       312,136 |       1.24 |
gzippy RSS growth T1→T16: **+216%** (122.8→388.1 MB). NOT flat.

NOTE — the prior `wall-progress.md:1200` reading (gzippy 1044 vs rapidgzip
347 MB @T8, ~3×) is STALE: that was the pre-fold `narrowed`+`MAX_POOLED=8`+
contiguous-data model. Post-fold the gap is 1.52× @T8 / 1.24× @T16 — the
footprint has already substantially converged. The absolute RSS is dominated by
the ~211 MB decoded-silesia output + 68 MB input mmap (shared by both tools),
NOT the per-thread decode scratch.

## (b) Per-thread working-set byte accounting (native, T16, sha=OK)
PER-THREAD persistent thread-local `Block` engine:
| component                        | bytes   | KiB   |
|----------------------------------|--------:|------:|
| output_ring (u16, 2×32KiB)       | 131,072 | 128.0 |
| dist_hc code_cache [(u8,u16);2^15]| 131,072 | 128.0 |
| lut_litlen (ISA-L lit/len LUT)   |  23,032 |  22.5 |
| misc Vecs (literal_cl/backrefs)  |     316 |   0.3 |
| **PER-THREAD TOTAL**             | **285,492** | **278.8** |
AGGREGATE over 16 threads: 4.36 MiB.
SHARED read-only (ONE copy, all threads, fixed-Huffman tables): 18,496 bytes.

## (c) L2/L3 MPKI + mem-stall (perf stat, T16, cpu_core counts)
| run            | instr (B) | LLC-miss MPKI | L1d-miss MPKI | cache-miss MPKI | L1d-miss% |
|----------------|----------:|--------------:|--------------:|----------------:|----------:|
| gzippy-native  |     10.11 |         0.205 |         4.547 |           1.942 |     1.97% |
| rapidgzip      |      6.85 |         0.379 |         5.766 |           2.520 |     2.73% |
gzippy executes **1.48× more instructions** for the same output, but its MEMORY
behavior is BETTER per-instruction: LLC-miss MPKI 0.205 vs 0.379, L1d-miss MPKI
4.55 vs 5.77. gzippy is NOT memory-stall-bound relative to rapidgzip.

## VERDICT — mandate clause-by-clause
The mandate has THREE testable claims. Scored against the numbers:

1. **"hot-in-cache much of the time / low MPKI"** — **MET.** LLC-miss MPKI 0.205
   (0.21 misses per 1000 instructions) is tiny and BELOW rapidgzip's 0.379. The
   hot decode streams; the 278 KiB/thread working set does NOT manifest as a
   cache-residency penalty at the miss-rate level (the ring + dist cache are
   touched with high temporal locality). gzippy is not memory-bound here.

2. **"shared read-only tables across threads"** — **PARTIALLY MET.** The
   fixed-Huffman tables ARE shared (one 18 KB OnceLock copy). BUT the per-block
   dynamic-Huffman decode tables are PER-THREAD, and the 128 KiB
   `dist_hc.code_cache` is a per-thread duplicate of a structure that is purely
   a function of the block's distance code lengths — not shared.

3. **"tiny per-thread working set sized to stay in L1/L2 / RSS roughly flat as T
   rises"** — **NOT MET.** Per-thread working set is **278.8 KiB**, dominated by
   TWO 128 KiB structures (output_ring + dist_hc code_cache). That is ~6× the
   48 KiB L1d and overflows a per-thread L2 budget; on this 16-core part the L2
   is shared per P-core cluster, so 8–16 simultaneous 278 KiB working sets do
   NOT co-reside in L2. RSS grows +216% T1→T16, not flat.

**OVERALL: the cache-residency mandate is NOT (fully) MET on the
working-set/RSS-flatness clauses, but the *consequence* the mandate cares about
(cache-residency → low MPKI) is currently fine** — gzippy's MPKI is already
better than rapidgzip's. The 278 KiB/thread is large on paper but is NOT today
causing a measurable cache-miss penalty, because the access pattern is
streaming/local. The genuine, measured gaps are: (i) the per-thread
`dist_hc.code_cache` is a 128 KiB un-shared duplicate that could be shrunk or
shared; (ii) RSS scales steeply with T (mostly the output buffer + per-thread
ring, not the tables).

CROSS-CHECK with the banked x86 STOP (wall-progress.md 2026-06-02, 5
refutations): the footprint-ceiling oracle already proved a 30% RSS cut → ~3%
IPC (overlapped slack); the faithful DecodedData port proved a 29% RSS cut →
wall TIE. So footprint/working-set is a SLACK CORRELATE of the wall on x86, not
a wall lever. The mandate is a DESIGN goal co-equal with speed, independent of
whether it moves the wall — but the measured MPKI says the design goal's
*purpose* (cache-residency) is substantially satisfied already.

## SCOPED architecture work for NEXT turn (do NOT start this turn)
Ordered by mandate-impact × measured-tractability. Each must be byte-exact +
re-measured on rss_vs_t.sh (RSS-flat + MPKI must not regress) + the wall on
parity.sh (must not regress — footprint is wall-slack, so the bar is "free").

1. **Shrink/share `dist_hc.code_cache` (128 KiB/thread, the #1 duplicate).**
   It is `[(u8, u16); 1<<15]` = a full 32 K-entry reverse-lookup. rapidgzip's
   distance decode (`HuffmanCodingReversedBitsCached`, deflate.hpp:336) is the
   SAME structure — so this is NOT a divergence to delete; the lever is the
   CACHE-LEN sizing (MAX_CODE_LENGTH=15 → 32 K entries) vs the ~30 distance
   symbols actually present. Options: (a) a two-level/smaller cache (distance
   codes rarely exceed ~10–12 bits, so a 1<<12 cache + fallback halves it to
   64 KiB or quarters it); (b) pack the entry to 3 bytes / use a single u32.
   Measure: per-thread total 278.8→~150 KiB, MPKI must hold ≤0.205.

2. **Right-size / pool the output buffer that drives the +216% RSS-vs-T.** The
   RSS growth is dominated by per-chunk decoded output + the per-thread 128 KiB
   ring, not the tables. The faithful SegmentedU8 DecodedData port (commit
   2b8bfae, already built+correct, −29% RSS @T8, wall-TIE) is the existing
   blueprint — re-land it on the post-fold tree and re-measure RSS-flatness.

3. **(Lower priority) the 128 KiB output_ring** is a faithful vendor port
   (`m_window16`, deflate.hpp:805) — do NOT shrink below 2×32 KiB (back-refs
   need 32 KiB reach without wrap). This one is a faithful-structure floor.

NON-GOAL (per the 5-refutation STOP): do NOT pursue footprint as a WALL lever.
The architecture work above is justified by the MANDATE (design goal: small
shared hot working set), measured on RSS+MPKI, with the wall as a no-regress
guard only.

## Independent disproof attempt (no Task/advisor tool this session — owner self-disproof)
Per the standing ADVISOR-GAP policy (orchestrator-status.md:1778), the disproof
discipline was run directly:
- NEGATIVE control: binary vs ITSELF — two identical native runs report
  byte-identical per-thread accounting (deterministic, no jitter in the byte
  numbers). PASS.
- THREAD-TRACKING: `threads observed` tracks `-p` exactly (0 @T1 / 4 @T4 / 8 @T8;
  T1=0 EXPLAINED above — window-seeded engine, not a miss). PASS.
- HAND-CHECK: `heap_bytes()` matches an independent `size_of` computation from the
  struct definitions EXACTLY (output_ring 131072, dist_cache 131072, lut_litlen
  23032). PASS.
- OFF==IDENTITY: flag-off emits zero stderr and the identical decoded sha; 881
  lib tests pass with the hooks compiled in. PASS.
- POSITIVE control (the one that matters): ballast slope linear == thread count
  (above). PASS.
The verdict's load-bearing claim — "278 KiB/thread, but MPKI already low/better
than rapidgzip" — was attacked for the obvious confound (is MPKI low only because
the OUTPUT dominates and dilutes?): the L1d-miss MPKI (4.55) is high enough that
the decode IS visible in the counter, yet still below rapidgzip's 5.77, so the
low-miss reading is not a dilution artifact. The honest residual: a single
silesia workload at one host; MPKI per chunk-type was not separated.

## Tooling delivered this turn
- `src/decompress/inflate/mem_stats.rs` — re-hooked to native `Block`
  (`BlockHeapBytes` + `on_block_active`); ballast control on the native hook.
- `marker_inflate::Block::heap_bytes()`, `LutLitLenCode::heap_bytes()`,
  `HuffmanCodingReversedBitsCached::heap_bytes()` (counters only).
- `scripts/bench/rss_vs_t.sh` + `scripts/bench/_rss_vs_t_guest.sh` — the
  cache-residency measurement spine (host-freeze bracket, sha-verify every run,
  path=ParallelSM assert, RSS-vs-T + per-thread accounting + ballast control +
  perf MPKI, validated-instrument-first).
