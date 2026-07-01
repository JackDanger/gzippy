# Dead End: Eager post-process / resolve-ahead during consumer stall

## Hypothesis

The consumer stalls waiting for the in-order frontier chunk. During this stall, the
consumer could eagerly submit `apply_window` / post-process work for already-decoded
successor chunks whose predecessor windows are available, converting stall time into
useful work and shortening subsequent blocking latencies. Port of rapidgzip's
`queuePrefetchedChunkPostProcessing` (GzipChunkFetcher.hpp:521-581).

Several variants were attempted:

1. **Consumer-side eager post-process pump** (`feat/consumer-postprocess-pump`):
   during `wait_replaced_markers`, submit post-process for ready prefetched successors.

2. **Worker-side eager apply_window**: resolve a speculatively-decoded chunk's markers
   in parallel the moment its predecessor window arrives, off the consumer's serial path.

## How Measured

**Variant 1 (consumer-side pump)**:
- Enqueued **0 tasks** across the full run. The probe fired 0× because the consumer's
  cache was empty during stalls — nothing was ready to eagerly submit at that hook site.
- Wall: TIE.

**Variant 2 (worker-side eager apply_window probe)**:
- `EAGER_PROBE_SUBMITTED = 0` — zero ready successors found during stalls. The work
  is dependency-blocked on the same frontier the consumer waits on; moving submission
  to a worker cannot manufacture work that doesn't exist.
- Confirmed dead by advisor without building: "0 ready successors" is structural
  (the awaited work is dependency-blocked), not a hook-site error.

**Variant (eager pair, gz_eager, T8)**:
- Built and measured the consumer-side eager pair (Design B early-window + eager
  post-process at `future_recv` wait).
- `consumer.eager_postproc = 195 ms` of consumer SELF-WORK added to the critical
  path (per-successor `(*arc).clone()` + submission runs ON the consumer thread that
  it was supposed to relieve).
- NET LOSS (+195 ms consumer self-work on critical path).
- Fires correctly (23/23 reused, spec_promoted=29, byte-identical) but targets the
  small post-process term and dumps its submission overhead on the bottleneck thread.

## Verdict: REFUTED — dependency-blocked, inert

The eager-resolve direction has two converging refutations:
1. No ready work exists during the consumer's stall (0 ready successors, 0 tasks
   enqueued). The fundamental blocker is that successor windows are not available
   because they are downstream of the same frontier chunk the consumer waits on.
2. The consumer-side implementation adds more work to the critical path than it
   saves (–195 ms net loss).

The advisor triangulation (bound-depth starves + eager inert + wall = frontier
decode latency) points to per-byte DECODE SPEED as the residual lever, not scheduling.

## Code Location

`src/decompress/parallel/chunk_fetcher.rs` — `GZIPPY_EAGER_POSTPROC` gate (default
OFF, kept as diagnostic dead code). Do not re-enable in production without a fresh
oracle showing ready-work count > 0 during stalls.

## Related Entries

- `docs/dead-ends/placement-resolve-ahead.md` — placement oracle (frontier already
  94% prefetched; consumer waits on decode completion)
- `docs/dead-ends/fill-lever.md` — the fill lever (consumer-wait 167 ms); this eager
  direction was one of its proposed fix-halves
- `project_t8_gap_fully_mapped_2026_06_02` memory — DEAD list includes eager-resolve
  (both tools already submit apply_window to pool; B-fork slack 0/16)
