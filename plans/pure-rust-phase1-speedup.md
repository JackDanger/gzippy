# Make Rust phase-1 deflate faster

**Branch:** `perf/rapidgzip-accel`
**Current head:** `16755a5`
**Goal:** close ≥ 60 ms of the remaining 172 ms wall gap to rapidgzip on
silesia-large 16T by reducing the cost of gzippy's pure-Rust deflate
phase-1 (`bootstrap_with_deflate_block` / `Block::read_internal_compressed_specialized`).

## State at the start of this plan

Per-chunk decode anatomy (silesia-large 16T, 42 chunks, cum 5591 ms CPU):

```
worker.bootstrap            1906 ms  34.1%   p50=1.7 ms   p95=172   max=237 ms
worker.isal_stream_inflate  2141 ms  38.3%   p50=62      p95=104   max=213 ms
worker.append_markered       739 ms  13.2%   p50=9       p95=94    max=99 ms
worker.absorb_isal_tail      763 ms  13.6%   p50=22      p95=50    max=95 ms
unattributed                  42 ms   0.7%
```

vs vendor `worker.decode_chunk` p50 = 61.6 ms (the WHOLE chunk).

**Headline observation:** 6 of 42 chunks (14 %) have `bootstrap` running
for 200–237 ms with **zero** ISA-L follow-up (`worker.isal_stream_inflate`
absent on those chunks). These chunks are the wall-time pacemakers —
heavy-tail chunks don't parallelize cleanly across 16 workers because
the join condition is the latest finishing chunk.

## The decisive question

The 3× per-heavy-chunk gap could come from either:

- **(A)** Phase-1 cycles/byte is genuinely 3× slower than vendor's
  pure-C++ phase-1 (codegen / FFI / wrapper gap), OR
- **(B)** Phase-1 *ran the whole chunk* because the 32 KiB clean-window
  handoff to ISA-L never triggered. Vendor would have hit the handoff
  and finished the rest in fast ISA-L.

These two worlds require completely different fixes:
- (A) → asm diff, microbenchmarks, codegen tweaks.
- (B) → tune the handoff trigger or speculation strategy. **No
  bytes-per-second changes at all.**

Advisor prior: 70 % (B), 20 % per-block setup tax, 10 % (A).
Memory `project_bootstrap_perf_diag.md` also explicitly recommends
NOT porting ISA-L techniques to Rust — gzippy already has all of them.

We must resolve (A) vs (B) before any further work. **One span-arg
instrumentation commit + one bench answers it.**

## Step 1 — instrument bytes_decoded on `worker.bootstrap`

**Change:** in
`src/decompress/parallel/gzip_chunk.rs::bootstrap_with_deflate_block`,
add three args to the existing SpanGuard:

- `bytes_decoded`: `output.len() * 2` (u16s × 2 bytes/elem) at the
  moment the bootstrap returns Ok. From the inner function's
  `output.len()` just before `Ok(DeflateBootstrap { markers: ..., })`.
- `clean_window`: already in `bootstrap.outcome`; keep redundant here
  for joins.
- `bfinal_hit`: same.
- `handoff_reason`: one of `clean_window_armed` | `bfinal_hit` |
  `stop_hint_reached` | `non_handoff_terminal_block` | `error_in_body`
  | `error_in_header`. Distinguishes WHY phase-1 stopped.

Existing `worker.bootstrap.outcome` instant event already captures
`result`, `markers_len`, `end_bit`, `clean_window`, `bfinal`. Extend
it to ALSO carry `handoff_reason` and `bytes_decoded`. One file, one
commit.

Acceptance: `GZIPPY_TIMELINE=… ./target/release/gzippy …` produces a
trace whose `worker.bootstrap.outcome` instants carry both fields.

## Step 2 — compute MB/s + handoff-reason histogram

Add a script (or one-shot Python) that joins
`worker.bootstrap` durations with the matching `worker.bootstrap.outcome`
args. Output:

- MB/s histogram across all 73 bootstrap calls (p50 / p95 / max)
- For the 6 heavy-tail chunks specifically: MB/s + handoff_reason
- Counts of each handoff_reason

If gzippy's phase-1 throughput p50 is **≥ 100 MB/s**: phase-1 isn't
slow. Branch to **Step 3-B (handoff)**.
If **≤ 50 MB/s**: phase-1 IS slow. Branch to **Step 3-A (codegen)**.
If 50–100 MB/s: both branches in parallel, ordered by lower cost.

This is the falsification gate. Without it, every "fix" is a guess.

## Step 3 — branch on the data

### Branch B (handoff threshold) — likely path

Read vendor's handoff condition at
`vendor/rapidgzip/librapidarchive/src/rapidgzip/chunkdecoding/GzipChunk.hpp:520-525`:

```cpp
if ( cleanDataCount >= deflate::MAX_WINDOW_SIZE ) {
    return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(
        bitReader, untilOffset, result.getLastWindow( {} ), ...
    );
}
```

That's exactly the same `cleanDataCount >= MAX_WINDOW_SIZE = 32 KiB`
threshold gzippy uses at `gzip_chunk.rs:854` (`trailing_clean >=
MAX_WINDOW_SIZE`). So the THRESHOLD matches. If branch B is
correct, the gap lies elsewhere in handoff:

- **Threshold sub-condition.** Vendor handoff also requires
  `clean_handoff_armed` (vendor uses a different state-machine variable
  whose semantics may differ from gzippy's; line-cite the difference).
- **MAX_WINDOW_SIZE itself.** Both use 32 KiB. Try lower (16 KiB or
  8 KiB) — risk: ISA-L's dictionary is undersized, decode fails. Verify
  ISA-L's API requirements.
- **`bfinal_hit` short-circuit.** gzippy returns immediately on BFINAL
  even if no clean window. Vendor does the same. Equal.
- **Speculation candidate quality.** If gzippy speculation lands the
  decode start at a position with more back-refs to marker territory
  than vendor's, gzippy never accumulates a clean run. Comparison
  requires diffing the speculation start-bit positions between
  gzippy and vendor traces (already have `start_bit` arg on
  `worker.bootstrap`).

Falsifiable fix: lower the handoff threshold to 16 KiB IFF ISA-L's
inflate-with-dict API accepts < 32 KiB dictionaries cleanly. Bench:
run 20-trial A/B/A on neurotic, expect ≥ 30 ms wall improvement on
silesia-large 16T. Abandon if < 10 ms.

### Branch A (cycles/byte codegen) — fallback path

Only if Step 2 shows gzippy's phase-1 MB/s is genuinely lower than
vendor's at the same byte count.

1. `perf annotate` on a fresh silesia-large 16T run with debuginfo.
   Focus on `read_internal_compressed_specialized` hot loop opcodes.
2. Compile vendor with `-g -fno-omit-frame-pointer`, `perf annotate`
   `readInternalCompressedMultiCached`.
3. Diff hot-loop assembly. Look for: extra branches, scalar vs
   vector ops, register pressure, cache-line patterns.
4. Pick one structural divergence. Fix. Bench.

**Skip the synthetic microbench.** Per the advisor: "synthetic
all-markers input is a *different workload* and will mislead."
`perf annotate` on the real bench is the right diagnostic.

## Step 4 — falsification + ship

Whichever branch was taken: 20-trial A/B/A bench on neurotic
(silesia-large 16T) with the proper baseline ahead of the fix and
the drift control after. Acceptance:

- **Primary**: p50 wall improvement ≥ 30 ms with non-overlapping IQRs
- **Secondary**: cross-corpus correctness (silesia-large, silesia-gzip,
  silesia-gzip9, software-gzip, logs-gzip — all sha256 match T=1)
- **Tertiary**: counter check — `worker.bootstrap` p95/max for the
  heavy-tail chunks drops by ≥ 50 ms each, or MB/s improves accordingly

If primary fails (improvement < 10 ms), revert and reassess.

## Explicit non-goals

- **NOT porting ISA-L techniques to Rust.** Memory
  `project_bootstrap_perf_diag.md` is explicit: the inflate algorithm
  is already faithful; the cycles aren't in the algorithm. Re-attempting
  this would violate `feedback-instrument-before-optimize`.
- **NOT touching `append_markered` or `absorb_isal_tail` yet.** Those
  spans account for 27 % cum CPU (~94 ms wall budget combined) but
  the heavy-tail chunks already include their share of these spans —
  fixing handoff would also reduce them as a side-effect (fewer
  marker-only chunks = less append_markered work).
- **NOT changing speculation candidate offsets.** Sub-partition emits
  were closed in Fix #3 (commit 62818b2). Don't re-litigate.
- **NOT adding cross-worker buffer steal back.** Reverted commit
  79c1cc0 — neutral-to-negative on wall, see
  `feedback-instrument-before-optimize`.

## Why this plan order matters

- The advisor's prior (70 % handoff, 10 % codegen) is calibrated on
  data, not vibes. If the inner-loop hypothesis were right, my earlier
  audit of gzippy's primitives (TRIPLE_SYM, refill, const-generic
  markers, etc.) would have flagged a gap. It didn't — gzippy has
  every ISA-L technique vendor uses.
- The phase-1-ran-the-whole-chunk pattern is *observable* in the
  existing trace data (6 chunks with bootstrap > 200 ms and zero
  isal_stream_inflate). That's not speculation; it's a recorded fact.
- One commit's worth of instrumentation kills 80 % of the plan
  branches. Don't write a 3-day asm-diff plan before paying that
  cheap toll.

## Estimated cost

- Step 1 instrumentation: 30 minutes (edit + commit + push)
- Step 2 analysis: 20 minutes (Python join)
- Step 3-B (handoff threshold tweak): 1 hour for a safe change + bench
- Step 3-A fallback (asm diff): 3–4 hours
- Step 4 bench: 15 minutes per A/B/A pass (20 trials × 2 versions × 1.4s/trial)

Total expected: **2 hours if Branch B works; 5 hours if Branch A.**

## Open uncertainties to revisit during execution

1. Does ISA-L's `isal_inflate_set_dict` require the full 32 KiB
   dictionary, or accept smaller? Check `vendor/isa-l/include/igzip_lib.h`
   and confirm before lowering the handoff threshold.
2. Is the 6-chunk heavy tail content-correlated (same offset region
   in silesia)? If yes, the input itself has a region that's
   adversarial for gzippy's speculation. Cross-corpus bench will
   surface this.
3. Vendor's `cleanDataCount` accounting may differ from gzippy's
   `trailing_clean` — verify by reading vendor's
   `appendDeflateBlock` and comparing the increment/reset rules.
