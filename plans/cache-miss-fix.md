# Cache-miss fix plan

## The target

`scripts/timeline_analyze.py` on a silesia T=9 P-cores run shows:

```
wait.block_fetcher_get : 210.4 ms (43% of 486 ms wall) across 4 cache misses
  chunk_id=0    offset=0          started @ 1.2 ms   wait=39.2 ms (cold start)
  chunk_id=201  offset=188 MB     started @ 141 ms   wait=74.5 ms (mid-stream)
  chunk_id=1855 offset=440 MB     started @ 353 ms   wait=52.8 ms (mid-stream)
  chunk_id=2545 offset=503 MB     started @ 420 ms   wait=43.9 ms (mid-stream)
```

Each miss = consumer paying one full chunk decode time synchronously.

Reference: rapidgzip on same hardware: 0 large cache misses, total
`wait.block_fetcher_get` = 84 ms across 26 short waits (~3 ms each).

## Hypothesis (refined from the chase log)

The misses fall into two categories with distinct fixes:

| Miss | Cause | Candidate fix |
| --- | --- | --- |
| chunk 0 cold start (~39 ms) | No prefetch has fired yet when consumer asks for chunk 0 | Issue prefetch for chunks 0..N at `drive()` entry, BEFORE consumer's first `get_with_prefetch` |
| chunks 201/1855/2545 (~170 ms) | Consumer's `next_unprocessed_block_index` advances past the prefetcher's `lastFetchedIndex` window | Investigate vendor's `FetchingStrategy::FetchMultiStream` behavior; check whether gzippy's `record_fetch` is updating the strategy correctly |

These are HYPOTHESES, not facts. The plan below verifies before
fixing.

## Anti-mistake rules (binding for this plan)

Per `plans/rust-rapidgzip.md`'s anti-mistake catalogue:

1. **Measure first**: each fix has a falsifiable prediction. If
   wall doesn't drop within the predicted band, revert.
2. **Implement-and-measure beats theorize**: every step ends with
   a real run on neurotic, T=9 P-cores, silesia, and a numerical
   delta vs baseline 486 ms.
3. **Cross-tool diff is the source of truth**: after each fix,
   re-run `timeline_analyze.py /tmp/gz.tl.json /tmp/rg.tl.json` to
   confirm the gzippy-specific event has moved without disturbing
   anything else.
4. **Revert eagerly**: if any step regresses anything (wall, pool
   eff, cache hit, byte correctness, isal-compression path), revert
   that step in the same session.

## Step-by-step plan

### Step 0 — reproducibility check (15 minutes, NOT optional)

Concern: the 4-miss finding is from a single trace (n=1). The advisor
flagged this as the largest unaddressed risk in the diagnosis.

```bash
ssh -J neurotic root@10.30.0.199 \
  'cd ~/gzippy && cargo build --release --features pure-rust-inflate'

# Run 10 traces, extract miss chunk_ids from each
for i in $(seq 1 10); do
  ssh -J neurotic root@10.30.0.199 \
    "rm -f /tmp/gz.$i.json &&
     taskset -c 1,3,4,5,6,7,10,13,15 \
       env GZIPPY_TIMELINE=/tmp/gz.$i.json \
       /root/gzippy/target/release/gzippy -d -c -p 9 \
       /root/benchmark_data/silesia-gzip.tar.gz > /dev/null &&
     python3 /tmp/show_misses.py /tmp/gz.$i.json"
done
```

**Pass criterion**: the same chunk_ids appear in ≥7 of 10 runs (allow
some jitter; expect chunk 0 in 10/10, chunks 201/1855/2545 in most).

**Fail criterion**: miss chunk_ids vary widely run-to-run → the
finding is noise, abort the plan and go back to instrumentation.

### Step 1 — cold start fix (1 hour, ~40 ms expected)

Smallest, cleanest target. Chunk 0 ALWAYS misses because no prefetch
has fired yet when `consumer_loop` makes its first `get_with_prefetch`
call. The vendor avoids this because their consumer issues a
prefetch-warming pass at construction.

Look at `src/decompress/parallel/chunk_fetcher.rs::consumer_loop`
(line 668+). The first iteration calls `block_finder.get(0)` and then
`block_fetcher.get_with_prefetch(...)` — on the very first call,
`should_drive_prefetch` is true (last_fetched_before is None), so
prefetch IS dispatched. The issue: prefetch is dispatched IN PARALLEL
with the on-demand fetch for chunk 0. Chunk 0 still costs 39 ms wall
because IT'S the on-demand work.

**Real fix**: chunk 0's wait is *unavoidable* — it IS the decode
time on the critical path. We can't make decode 0 finish faster.
BUT we can confirm by:

(a) Hand the consumer chunks 1..N as a hand-rolled prefetch at
    `drive()` entry, BEFORE entering `consumer_loop`. Measure
    whether chunk 0's wait drops below 39 ms (it shouldn't — it's
    the decode itself).

(b) IF chunk 0 wait stays at 39 ms after (a), CONFIRMED chunk 0 is
    the decode-time floor. SKIP. Move to Step 2.

**Pass criterion**: Step 1 either eliminates chunk 0's wait OR
confirms (with measurement) that it's irreducible. Either result is
a useful narrowing.

**Predicted savings**: 0-39 ms wall (skewed low — likely irreducible).

### Step 2 — mid-stream prefetch behavior (2-3 hours)

The 3 mid-stream misses (chunks 201, 1855, 2545) are the real lever.
Consumer outran the prefetch window OR the prefetcher fell behind.

#### Step 2a — confirm WHICH: outran-window OR prefetcher-fell-behind

Add a span around each prefetch-strategy decision in
`block_fetcher.rs::prefetch_new_blocks`:

```rust
trace_v2::SpanGuard::begin_with("prefetch.dispatch",
    &format!(r#""last_fetched_idx":{},"submitted":{},"queue_depth":{}"#,
             ...));
```

Then re-run and join with `wait.block_fetcher_get` events. For each
cache miss, see what the prefetch state was at the miss moment:

- If `last_fetched_idx < miss_chunk_id`: the prefetcher hadn't reached
  this chunk yet (window-too-narrow OR prefetcher-slow).
- If `last_fetched_idx >= miss_chunk_id` but the chunk isn't in
  prefetch_cache: it was dispatched but evicted (cache capacity issue).

**Pass criterion**: each miss is classified into one of two buckets.

#### Step 2b — fix based on Step 2a classification

Case A (prefetcher behind): the consumer iterates faster than the
prefetcher dispatches. Increase prefetch fan-out OR change scheduling.
WARNING: we already tried `prefetch_capacity 2x → 4x` (regressed). The
fix has to be more nuanced — perhaps speeding up the prefetcher
dispatch rate, not the cache size.

Case B (cache evicted): the prefetcher hit the chunk early but it
got evicted from `prefetch_cache` before consumer asked. Bigger
prefetch_cache might help (different from `prefetch_capacity` which
also affects in-flight count).

Case C (other): TBD; data will say.

**Pass criterion**: wall drops 50-150 ms (the 3 misses total 170 ms;
fix should capture most but allow for some not closing).

**Failure mode**: if wall regresses or doesn't drop, revert and
either re-classify (Step 2a was wrong) or accept that mid-stream
misses are structurally unavoidable and move to per-chunk decode
speed (separate work item).

### Step 3 — measure pool efficiency change

After Steps 1+2, re-run the trace and check:

- `pool.pick` sum should drop proportionally (workers idle less when
  consumer doesn't block).
- per-thread busy time σ should shrink.

If pool.pick savings are 5× the wall savings (the σ × consumer-wait
ratio), the fix is properly amortizing. If only 1×, the fix only
helped one thread; load is still imbalanced.

### Step 4 — re-baseline against vendor

Run the cross-tool diff:

```bash
ssh -J neurotic root@10.30.0.199 \
  'python3 /tmp/timeline_analyze.py /tmp/gz.tl.json /tmp/rg.tl.json | head -60'
```

Expected after Steps 1+2 succeed:
- gzippy wall: 486 → 290-350 ms
- gap to rapidgzip: 3.4× → 2.0-2.4×
- `wait.block_fetcher_get` should drop from 210 ms to ~50 ms (chunk 0 only)
- `pool.pick` should drop substantially (5-10× the wall savings)
- pool eff should rise from 45% toward 60%

**Done criterion for this fix**: wall < 350 ms on silesia T=9
P-cores AND no regression on `cargo test --release -- routing`.

**Stretch criterion**: wall < 250 ms (would close most of the
remaining gap to vendor).

## What this plan EXPLICITLY does not attempt

- Per-chunk decode rate improvements (1.9× gap) — separate work item;
  table-bits tunings already exhausted, would need SIMD inner-inflate
  changes (advisor-flagged as dead-end for vendor parity).
- Load-imbalance fixes that aren't cache-miss-derived — confirmed by
  Phase 3 dispatch data that imbalance is mostly a CONSEQUENCE of
  cache misses.
- Architectural changes to the consumer loop or BlockFetcher — vendor
  parity has been verified at `chunk_fetcher.rs:1059,1062` for
  forward-threading and `:663-666` for the two-key dance.
- New rapidgzip-side instrumentation — Phase 3 ships the matching
  span set; no further vendor patches needed for this fix's
  verification.

## Failure escape hatches

If Steps 1+2 don't move the needle:

1. **Maybe the misses are structurally irreducible** — consumer
   genuinely cannot know about chunks until BlockFinder confirms
   them. Document, declare cache-miss work-item complete, move on.
2. **Maybe load imbalance has a separate cause** — gzippy's worker
   busy σ remains high even after cache misses close. Per-worker
   busy timeline analysis would be the next investigation.
3. **Maybe per-chunk decode rate is the real lever after all** — the
   advisor's ceiling test estimated even closing all cache misses
   leaves wall at ~290 ms vs vendor's 162 ms. Time to attack the
   per-chunk rate (`worker.bootstrap`, `worker.block_body` sub-spans).

In any of these cases, the cross-tool diff harness in Phase 3 stays
useful — every subsequent investigation reuses it.

## Confidence ratings

| Step | Confidence in plan being right | Confidence in stated savings |
| --- | --- | --- |
| 0 (repro check) | HIGH (cheap, falsifiable) | N/A |
| 1 (cold start) | MEDIUM (chunk 0 likely irreducible) | LOW (0-40 ms predicted, likely 0) |
| 2a (classify misses) | HIGH (instrumentation work is clean) | N/A |
| 2b (fix based on class) | MEDIUM (depends on class) | MEDIUM (predicted 50-150 ms wall, real number depends on cause) |
| 3 (pool eff measurement) | HIGH (existing instrumentation reads) | N/A |
| 4 (cross-tool diff) | HIGH (harness exists) | N/A |

Don't oversell. The plan's downside case is ~50 ms wall savings; its
upside case is ~150 ms. Either is useful; neither closes the gap
alone. Per-chunk decode rate work follows regardless.
