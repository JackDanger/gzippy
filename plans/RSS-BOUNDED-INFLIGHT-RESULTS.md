# RSS bounded-in-flight chunk lever — RESULTS (2026-06-22)

Branch `kernel-converge-A`, worktree `/home/user/www/gzippy-amd-t2t4`.
Commits: 6aecf3fb (preserve CLEAN-WALL), pre-reg falsifier, 49565e71 (instrument).
LEADER cycle. Gating: feedback_rss_vs_wall_scoreboards (RSS!=WALL; no wall claim).

## VERDICT: ISOLATION **REFUTED** → STOP before coding (per pre-registered falsifier).
The mission's "free/recycle post-write retained chunks" / "bounded in-flight =
fewer simultaneously-live chunks" lever does NOT close the monorepo-T2 RSS gap.
No fix coded.

## INSTRUMENT (Gate-0 PASS, byte-transparent)
`GZIPPY_LIFECYCLE_SPLIT=1` (chunk_data.rs): decomposes simultaneously-live
ChunkData at the peak-LIVE construct into AHEAD(prefetch+main cache) /
PENDING(consumer) / RETAINED(recycle_deferral, post-write) / DECODING(derived).
Per-holder gauges set at mutation sites (block_fetcher prefetch/main/prefetching;
consumer pending; recycle_deferral). OFF==identity (cached OnceLock). Non-inert
(fired=5) + conservation (accounted≤live, decoding≥1) PASS both arches.
sha256==zcat (0dd50d07b014) instrument-on AND -off → byte-transparent.

## STAGE 1 — ISOLATION (both arches identical = Gate-3 replicated)
monorepo-T2, pool=2, chunks=5, max_live=5 (whole small file live at peak):
  AHEAD(prefetch=0+main=0)=0% | PENDING=20% (1) | RETAINED=40% (2) | DECODING=40% (2)
  in_flight_decode_jobs=1
- **AHEAD=0%** — NOTHING is decoded-ahead sitting in the prefetch cache. The
  mission's KEY INSIGHT ("rg's lower RSS = fewer simultaneously-live via bounded
  prefetch / eviction") is **REFUTED**: gz does not accumulate prefetched chunks.
- By COUNT, RETAINED (recycle_deferral, depth=2) = 2 of 5 (40%) — these ARE
  written-and-resolved chunks held post-write. That count alone would read as
  "retained-dominant." But the COUNT is not the verdict — the causal perturbation
  below is (Gate-2; count×avg-bytes is the kind of synthesized absolute the
  anti-bias protocol forbids).

## STAGE 1b — GATE-2 CAUSAL PERTURBATION (the verdict)
recycle-deferral depth = the actual "free immediately after write" fix
(GZIPPY_RECYCLE_DEFER_DEPTH; depth=0 = recycle buffers the instant the chunk is
written). Peak RSS (load-immune /usr/bin/time, x86 solvency, 4K pages):
  depth=0 → 86.8 MiB | depth=1 → 96.0 | depth=2(default) → 96.8 | depth=4 → 97.0
  (macOS aarch64, 16K pages: depth=0 ≈ 94 vs depth=2 102.5 — same ~8-10 MiB)
- depth=2 96.8 MiB reproduces the gated 96.5. Freeing RETAINED immediately
  (depth=0) drops peak RSS only **~10 MiB = ~26% of the 37.8 MiB gap to rg's 59**,
  **< the pre-registered 50% (~19 MiB) CONFIRMED threshold** → REFUTED.
- Why the 2 retained chunks (~31 MiB by buffer-count) free only ~10 MiB: the peak
  watermark is set by the genuinely in-flight decode set (2 DECODING + pipeline),
  and freed buffers' pages are RETAINED by rpmalloc's thread cache (same wall the
  MADV_DONTNEED marker-free hit; MADV_FREE/drop don't drop max-RSS). sha byte-exact
  at all depths on THIS cell (note: depth<2 corrupts silesia T4/T8 per prior record
  — so depth=0 is not shippable anyway).

## STAGE 1c — what DOES govern peak RSS (chunk-KIB sweep, x86, Gate-2)
monorepo-T2 peak RSS vs chunk size, with max_live and lifecycle breakdown CONSTANT:
  K=256 → 39.8 MiB | K=512 → 53.0 | K=1024 → 76.0 | K=4096(default) → 96.8
  (max_live=5 and AHEAD=0/PENDING=20/RETAINED=40/DECODING=40 at EVERY K)
⇒ Peak RSS ∝ per-chunk decode-buffer BYTES (chunk size), INDEPENDENT of live
count or post-write retention. The resident bytes at peak are the genuinely
IN-FLIGHT decode buffers. (K=256 even undercuts rg's 59 — chunk SIZE is a real
RSS knob, but it is a config with throughput implications and is a per-chunk-SIZE
lever, NOT this mission's "bounded in-flight count / free-after-write" lever, and
is already characterized in prior cycles. Not shipped/claimed here.)

## AGAINST THE PRE-REGISTERED FALSIFIER
- CONFIRMED-RETAINED (RETAINED freeable bytes > 50% of gap): **NO** (causal = ~26%).
- CONFIRMED-AHEAD (prefetch-depth > 50% of gap): **NO** (AHEAD = 0%).
- REFUTED (dominant live bytes genuinely in-flight / needed): **YES** →
  report + STOP, no fix code. Confirmed on macOS aarch64 AND linux x86 (Gate-3).

## BOXES
solvency x86 (root@REDACTED_IP): worktree /root/gz-rss-inflight built + REMOVED at
exit; llama NEVER touched (RSS load-immune, no freeze); left clean. macOS local
build kept. No bench-lock touched. No wall claim made.
