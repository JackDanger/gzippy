# Dead End: Copy-elimination (absorb_isal_tail and consumer copies)

## Hypothesis

Eliminating the 212 ms CPU copy in `absorb_isal_tail` (copying the ISA-L-decoded
clean suffix into the chunk buffer) and the consumer-side window-publish copy would
close a proportional fraction of the wall gap. These copies appeared expensive in
`GZIPPY_TIMELINE` leaf budgets.

## How Measured

**absorb_isal_tail copy** (commit `328696e`): replaced the 212 ms
`copy_nonoverlapping(tail.data→chunk.data)` with an O(1) `mem::swap` (byte-exact
swap instead of copy). Measured with frozen interleaved A/B, N=15, P-core pinned.

Result: **1.000× = –0.0% wall** (base 1171.0 / absorb 1170.7 MB/s median), sha256
IDENTICAL.

**Consumer window-publish copy** (consumer-serial window-publish copy-elim,
`from_owned_none` — kills the 2nd 32 KiB to_vec): layered into the `feat/fulcrum-causal-sweep`
stack. consumer.iter –18 ms, try_take_prefetched –14 ms. Wall: within noise of the
cumulative layered A/B.

## Verdict: REFUTED — copies are overlapped, wall-neutral

The pipeline hides per-chunk copies behind other workers' decode. A 212 ms CPU
operation can be 0.0% of wall when it runs entirely in the shadow of worker decode
work across 8 threads.

**DECISIVE LESSON:** CPU-sum ≠ wall. `GZIPPY_TIMELINE` leaf budgets show the
*total CPU* across all workers — a region can be large in that metric and
wall-neutral when it is overlapped. Copies are the load-bearing example of this.

The absorb-move (`mem::swap`) is kept as a strictly-fewer-ops simplification; it is
NOT a speed win. Do not re-classify it as a performance win in future notes.

## Code Location

`src/decompress/parallel/gzip_chunk.rs` — `absorb_isal_tail` function; the swap
is production-shipped.

`src/decompress/parallel/chunk_fetcher.rs` — consumer-side window-publish copy-elim;
shipped as `from_owned_none` path.

## Do Not Re-attempt

Any copy-elimination-as-wall-lever framing. Copies are overlapped slack by direct
measurement. If a new copy appears on the critical path, establish that first via
causal perturbation (not CPU-sum attribution) before treating it as a lever.

## Related Entries

- `project_copies_wall_neutral_2026_05_29` memory — the decisive A/B with full
  accounting
- `docs/dead-ends/footprint-bandwidth.md` — broader footprint/copy framing
