# Dead End: Fill lever (consumer-wait 167 ms vs rapidgzip 18 ms)

## Hypothesis

A cross-tool trace located gzippy's consumer-wait at 167 ms vs rapidgzip's 18 ms
(9× gap, spread mid-run at chunks 11/41/54). This looked like a new "fill lever"
reachable via two fixes: (a) relaxing the exact-equality prefetch guard to range-accept
to recover thrown-away range-valid prefetches, or (b) frontier-priority dispatch
to reduce the number of stall events.

## How Measured

**Counter read** (`PREFETCH_REJECT_BY_GUARD`): **guard-rejects ≈ 1** per full run.
The guard (`chunk_fetcher.rs:1152`, `max_acceptable_start_bit == next_block_offset`)
barely fires — there are essentially no thrown-away range-valid prefetches to recover.
Range-accept would be a corruption risk (gzippy's `[encoded, max)` is a PREFIX-GAP
of real predecessor data, NOT vendor's stored-block zero-padding ALIAS range; accepting
`S < max` drops `[S, max)`) for a lever that would help ~0 chunks.

**Frontier-priority oracle** (`feat/frontier-placement-oracle`, T4/T8/T16): TIE
(sd<1.5%, oracle/base wall ≈ 1.00× all T). The priority-dispatch fix is ALREADY
SHIPPED: `chunk_fetcher.rs:1861-1867` gives on-demand frontier decode priority –1 vs
priority-0 speculatives, popped from a BTreeMap priority queue (`thread_pool.rs:499`).
Re-prefetch-during-wait (Lever H, `chunk_fetcher.rs:1167`) also already shipped.
`frontier_already_inflight 34/36 = 94%` — gzippy already decodes the frontier ahead.
`fulcrum schedule` S1: RATE-100% (decode was started before each consumer stall).

## Verdict: DEAD — the 167 ms is the KNOWN RATE FLOOR

The 167 ms vs 18 ms is NOT a new lever. It is **count × rate**:
- gzippy: ~4 stalls × ~42 ms/stall = 167 ms (including ~19 ms startup floor)
- rapidgzip: ~1 stall × ~18 ms/stall = 18 ms

Per-stall, each stall is the frontier-chunk decode latency (pure-Rust single-symbol
~150 MB/s vs rapidgzip multi-symbol ~340 MB/s ≈ 2.27×). `saturated=372/452` (82%,
7+ in-flight) confirms RATE (not placement/dispatch): all 8 workers are mid-execution
when the consumer stalls, so priority –1 cannot make the frontier start sooner.
`cache-miss=33` = "frontier not done yet" (RATE).

Both proposed fix-halves were dead before measuring:
- Range-accept: guard-rejects≈0 (nothing to recover) AND corruption (prefix-gap
  data, not alias range).
- Frontier-priority: already shipped and oracle-TIE'd.

## Residual

The honest remaining directions (not clean levers):
- (A) Decode RATE: port ISA-L-class multi-symbol decode into pure-Rust; ~15% bounded
  per `project_t8_saturated_pool_diag`. Does NOT reach 1.47× parity alone.
- (B) Diffuse architectural stall gap: gzippy 1.18–1.54× slower with FEWER instructions
  — no single located lever. This is fundamental in-order-consumer coordination.

## Code Locations

- `src/decompress/parallel/chunk_fetcher.rs:1152` — exact-equality guard (correct, do not relax)
- `src/decompress/parallel/chunk_fetcher.rs:1861-1867` — priority –1 on-demand dispatch (shipped)
- `src/decompress/parallel/chunk_fetcher.rs:1167` — re-prefetch-during-wait / Lever H (shipped)
- `src/decompress/parallel/thread_pool.rs:499` — BTreeMap priority queue

## Related Entries

- `project_fill_lever_dead_2026_06_01` memory — advisor-confirmed STOP with full accounting
- `docs/dead-ends/placement-resolve-ahead.md` — placement oracle TIE
- `docs/dead-ends/eager-resolve-phantom.md` — eager post-process (0 ready tasks)
