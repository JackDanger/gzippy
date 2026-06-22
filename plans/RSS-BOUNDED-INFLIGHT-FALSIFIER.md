# RSS bounded-in-flight chunk lever — PRE-REGISTERED falsifier (2026-06-22)

Branch `kernel-converge-A`, worktree `/Users/jackdanger/www/gzippy-amd-t2t4`
(base 6cd6b4b1; +6aecf3fb CLEAN-WALL preserve). LEADER: RSS bounded-in-flight.

## Governing frame
RSS != WALL (feedback_rss_vs_wall_scoreboards). This lever is gated on the
LOAD-IMMUNE peak-RSS isolation oracle ONLY. NO wall-win claim permitted from it.
It must NOT regress throughput. Faithful-rg port is the method (cite vendor
file:line). No code from an unconfirmed model: ISOLATE first.

## Gated starting point (prior cycles, Gate-2 causal, load-immune)
- gz monorepo-T2 peak RSS = 96.5 MiB vs rapidgzip ~59 MiB. Gap ≈ 37.5 MiB.
- Peak RSS ∝ in-flight decoded chunk bytes (chunk-KiB sweep, Gate-2).
- monorepo-T2: chunks=5, max_live_chunks=5 (whole small file live at peak).
- MADV_DONTNEED per-chunk marker-free: byte-exact, RSS 96.9→75.4, but
  +2.85% throughput regression (TLB shootdowns) → NOT shipped.

## STAGE 1 — ISOLATION (gating prerequisite; pre-registered thresholds)
Question: WHY are all 5 chunks live at peak? Classify each live ChunkData at
the peak-LIVE moment into lifecycle state:
  - DECODING        — worker actively building (genuinely in-flight, needed)
  - AHEAD           — decoded, sitting in prefetch_cache/main cache awaiting
                      the consumer (prefetch depth — decoded-ahead-for-overlap)
  - PENDING         — consumer holds it (post-process / awaiting in-order write)
  - RETAINED        — written to output, still held in recycle_deferral before
                      its buffers are recycled/freed (FREEABLE after write)

Instrument: GZIPPY_LIFECYCLE_SPLIT=1 — per-holder current-size gauges
(prefetch_cache, main cache, prefetching map, pending, recycle_deferral),
snapshotted at the construct that sets a new MAX_LIVE_CHUNKS (peak is always at
a construct). Byte-transparent; OFF == identity. Gate-0: non-inert
(snapshot fired, gauges move) + conservation (DECODING+AHEAD+PENDING+RETAINED
== peak LIVE within ±1 for the in-hand chunk).

ISOLATION VERDICTS (pre-registered):
- CONFIRMED-RETAINED iff RETAINED chunk bytes > ~50% of the ~37.5 MiB gap
  (~19 MiB). → fix = free/recycle right after write (the mission's framing).
- CONFIRMED-AHEAD iff AHEAD (prefetch-depth) chunk bytes > ~50% of the gap.
  → fix = bound prefetch_capacity / in-flight set (rg-faithful BlockFetcher
  cache/prefetch sizing). STILL the bounded-in-flight lever, different knob.
- REFUTED iff the dominant live bytes are DECODING (genuinely in-flight,
  needed for parallel throughput) → report + STOP, no code.

(The mission's literal CONFIRMED criterion is RETAINED>50%. Per the anti-bias
rule I report whichever state actually dominates and route the fix to it; if
DECODING dominates I STOP.)

## STAGE 2 — FIX (only if Stage 1 CONFIRMED, routed to the dominant state)
Faithful-rg bounded/evicted in-flight chunks, churn-free (NO per-chunk madvise/
realloc on the hot path; prefer drop/recycle/bounded-cap). Byte-exact;
preserve streaming-output + CRC/ISIZE-after-final-chunk contract; free ONLY
after bytes written AND no later chunk needs this chunk's window.

FIX CONFIRMED iff ALL hold:
- BYTE-EXACT: sha256==zcat AND flate2+libdeflate differential on silesia+
  monorepo across multiple chunk sizes, IN THE SAME COMMIT.
- RSS (load-immune, /usr/bin/time, spread ≤0.02 MiB): monorepo-T2 peak RSS
  drops materially toward rg ~59 MiB. Replicated macOS aarch64 AND linux x86.
- THROUGHPUT NON-REGRESSION (mandatory): interleaved best-of-N≥9, /dev/null
  both arms, on T2/T4/T8; T1/clean path untouched. Also silesia (33% markered)
  + a large file (where in-flight is genuinely needed) show NO regression.
- Gate-4: GZIPPY_DEBUG=1 → path=ParallelSM; binary built the change.

FIX FALSIFIED iff: no material RSS drop, OR any byte mismatch, OR any
throughput regression beyond spread on any cell → revert.

NO WALL-WIN CLAIM is permitted regardless of outcome.
