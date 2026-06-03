# Fixed-architecture design — the consumer/window-resolution chain (DESIGN ONLY, for adversarial disproof)

> **STATUS: HISTORICAL DESIGN (2026-06-03).** Design D (resolve-ahead) measured **TIE/inert**
> on silesia — see [`sm-parity-gap-matrix.md`](sm-parity-gap-matrix.md) row C. Consumer null-oracle
> (~100 ms T8) is a separate bounded lever (row L). Do not treat “consumer-only” as the sole gap.

---

Status: DESIGN, not built, not measured. Provisional on the causal C2 test (agent a001c98,
in flight) confirming the gap is the consumer chain. To be disproven by an advisor before any build.

## The measured target (what a fix must move)
- SHIPPED ISA-L gzippy loses to rapidgzip 1.02×/1.19×/1.45× at T4/T8/T16 (frozen, RUN_TRUSTWORTHY).
- gzippy executes ~0.95× the instructions/MB (FEWER) yet loses ⇒ NOT inner decode, NOT work — STALLS.
- dtlb_store_walk 1.9× (isal) / 3.26× (pure-rust); scales with T; consumer is 94–96% WAIT.
- Gap is architectural, present in BOTH decoders, T-dependent (worsens with thread count).

## What gzippy does today (the serial chain)
`consumer_loop` (chunk_fetcher.rs:909), single in-order thread, per chunk i:
1. `wait.block_fetcher_get` — block until chunk i's DECODE RESULT is ready (the dominant wait).
2. resolve markers: `apply_window` on chunk i against predecessor window (small, L_resolve 8–35µs).
3. publish chunk i's tail window: `window_map.insert_owned_none(end_bit, tail)` — UNBLOCKS i+1's resolve.
4. write chunk i's bytes to the output, fold CRC.
Steps 2–4 are SERIAL on the consumer thread; step 1 is a wait on the worker pool.

## Vendor's structure (the un-refuted lever)
`queuePrefetchedChunkPostProcessing` (GzipChunkFetcher.hpp:521): when a chunk is prefetched AND its
predecessor window is already in the WindowMap, vendor submits that chunk's `applyWindow` to the
THREAD POOL — resolved + window-published AHEAD of the consumer reaching it, OFF the serial thread.
The consumer then reads an already-resolved result and only writes output.

## Why gzippy's PRIOR attempt failed (must not repeat)
`eager_postprocess_prefetched` (chunk_fetcher.rs:1294, default OFF) was REFUTED: it did the
clone + apply_window SUBMISSION on the CONSUMER thread at the future_recv/block_fetcher_get stall
= +195ms net loss (dumped its cost on the thread it meant to relieve), and an earlier variant
hooked the has_predecessor stall and ran 0×. Lesson: the work must run on a POOL WORKER, triggered
by window-availability, NOT submitted by the consumer at its own stall.

## THE CRITICAL TENSION this design must survive (the disproof target)
The consumer's dominant wait is `block_fetcher_get` = waiting on chunk i's DECODE, not its RESOLVE.
Resolve (apply_window) is tiny (8–35µs). So moving RESOLVE off the serial path
(queuePrefetchedChunkPostProcessing) may NOT move the wall — because the consumer is blocked on
DECODE, which is already on the worker pool. IF the wait is decode-bound, this design is INERT.
⇒ The design is only valid if the gap is the SERIAL DEPENDENCY CHAIN (resolve i needs published
window i-1 needs resolved i-1 ...), i.e. the per-chunk resolve+publish LATENCY accumulates into a
critical path even though each link is small — OR if the decode-wait is itself a SYMPTOM of the
publish chain (workers stall because they can't get a predecessor window to decode clean).

## Design D — faithful worker-side resolve-ahead (the proposed fix)
1. When chunk i's DECODE completes on a pool worker AND chunk i-1's window is already published,
   the WORKER itself (not the consumer) immediately runs chunk i's apply_window + computes its
   tail window, and publishes window i into the WindowMap — all off the consumer thread.
2. This cascades: publishing window i lets worker i+1 (if decoded) resolve, etc. — the resolve
   chain runs AHEAD of the consumer across the pool, not serially on it.
3. The consumer's per-chunk work shrinks to: read already-resolved bytes + write output + fold CRC.
   It no longer does apply_window or window-publish on its critical path.
4. Correctness: identical to today (apply_window is deterministic; publishing earlier only changes
   WHEN, not WHAT; output bytes + CRC + window contents unchanged). sha-verify + the seam tests.
5. Distinct from the failed eager-postproc: trigger = "worker finished decode + predecessor window
   available" (worker-side, event-driven), NOT "consumer hit a stall + submit work" (consumer-side).

## What this design EXPECTS to move (falsifiable predictions)
- If C2 holds (gap is the serial resolve/publish chain): consumer WAIT drops, wall drops toward
  rapidgzip, store-walks may drop (fewer serial passes over marker buffers on the consumer).
- If the wait is DECODE-bound (not resolve): wall FLAT ⇒ design INERT ⇒ the lever is elsewhere
  (decode scheduling / why workers can't decode clean / the prefetch-vs-frontier interaction).

## Open questions for the advisor to attack
- Is the consumer wait decode-bound or resolve-chain-bound? (the running causal C2 test answers this;
  the design is premature if C2 isn't confirmed.)
- Does worker-side resolve-ahead actually shorten the CRITICAL PATH, or just relocate slack work
  (apply_window was already overlapped/tiny)? Is this the copies-wall-neutral trap again?
- Does the store-walk gap (1.9–3.26×, the actual measured difference) even live in the resolve
  chain, or in the decode writes? If the latter, this design doesn't touch it.
- T-dependence: the gap WORSENS with T. Does resolve-ahead help MORE or LESS at high T?
- Is there a simpler/more-faithful structural difference vs vendor we're missing (prefetch depth,
  window-chain keying, the splitIndex/FetchingStrategy) that's the real architectural gap?
