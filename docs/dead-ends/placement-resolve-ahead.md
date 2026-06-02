# Dead End: Frontier-placement / prefetch-resolve-ahead

## Hypothesis

The wall gap is caused by gzippy decoding the in-order frontier chunk ON the
consumer's critical path — i.e., the consumer blocks waiting for a chunk to finish
decoding because that chunk was not pre-decoded ahead of the consumer's arrival.
Porting rapidgzip's `BlockFetcher` prefetch-ahead + `queuePrefetchedChunkPostProcessing`
(GzipChunkFetcher.hpp:520-583, BlockFetcher.hpp:289-315) would move decode off the
frontier and let the consumer pay only `applyWindow`, closing the gap.

Motivation: vendor source shows rapidgzip's in-order frontier carries cheap
`applyWindow` while expensive `decodeChunk` runs in prefetch threads ahead of the
frontier.

## How Measured

**Frontier-placement oracle** (`feat/frontier-placement-oracle`, commit `9ae9048`):
forced the in-order frontier chunk fully decoded AHEAD of the consumer arriving at
it (placement changed, per-byte rate unchanged). Byte-correct (sha == gzip both
ON/OFF). neurotic frozen-clock, N=9 interleaved, RUN_TRUSTWORTHY=true, sd<1.5%:

| T | oracle/base wall | ratio gz/rg |
|---|---|---|
| 4 | 0.992 TIE | 0.861/0.854 |
| 8 noSMT (headline) | 1.012 TIE | 0.739/0.747 |
| 16 SMT | 0.997 TIE | 0.608/0.606 |

All removal deltas within sd. Oracle instrumentation confirmed non-trivial:
`frontier_already_inflight 34/36 = 94%` — gzippy ALREADY prefetches the frontier
ahead off the consumer thread, like vendor `BlockFetcher`. The consumer's 92.9%
wall-wait IS the frontier chunk's own decode time; placement cannot be shortened
further.

**Fulcrum `schedule` S1**: independently classified stalls as RATE-100% (the decode
was already started on a worker before the consumer's stall began; the consumer
waits on completion, not dispatch). This corroborates the oracle.

## Verdict: CAUSALLY REFUTED

Placement (Lever B) is causally dead. The prior vendor-source-based "placement is the
lever" reasoning was source-reading inference; the causal oracle overturns it. This is
the trap the user flagged: code-reading can locate a structural difference without it
being a wall lever. The measurement is always the verdict.

The gap is per-byte DECODE WORK on the in-order frontier chunk, not scheduling.

## Code Location

`src/decompress/parallel/chunk_fetcher.rs` — priority queue (BTreeMap, thread 
`pool.rs:499`), on-demand frontier dispatch at priority -1 (lines 1861-1867).
Oracle code lives on branch `feat/frontier-placement-oracle`.

## Related Entries

- `docs/dead-ends/fill-lever.md` — the consumer-wait 167 ms stall (the fill lever),
  which is the RATE floor not a placement lever
- `project_fill_lever_dead_2026_06_01` memory — advisor-confirmed STOP
- `project_t8_gap_fully_mapped_2026_06_02` memory — PLACEMENT is item #1 in DEAD list
