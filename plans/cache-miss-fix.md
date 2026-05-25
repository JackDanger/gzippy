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

## Revised plan (post-adversarial review 2026-05-25)

The first draft of this plan had three real defects (see comments at
end of file). Revised:

### Step 0 — cheapest possible falsification (30 minutes)

Per the adversarial-advisor recommendation: **skip the instrumentation
work entirely if a brute-force test refutes the theory.**

```rust
// In chunk_fetcher.rs, BEHIND a GZIPPY_BURST_PREFETCH env var:
//   - Bump prefetch_cache.capacity() to 64 (vs current pool_size * 2 = 18).
//   - At drive() entry, BEFORE entering consumer_loop, dispatch
//     prefetches for chunk indices 0..16 in a tight loop.
// One-line capacity change + ~10 lines of burst dispatch.
```

Then run ONCE (best of 3 trials) on neurotic, silesia, T=9 P-cores.

**Falsification criterion**: if `wait.block_fetcher_get` does NOT drop
below 150 ms (from the 210 ms baseline), the cache-miss theory has a
ceiling well below this plan's claimed savings. ABORT and reallocate
to per-chunk decode work (separate work item).

**Confirmation criterion**: if `wait.block_fetcher_get` drops to
< 100 ms AND wall drops by ≥ 80 ms (486 → 406 ms or better), the
theory holds at scale — proceed to Step 1 to design a non-regressing
fix (the brute force one may have its own overhead).

**If wall drops < 50 ms despite cache-misses dropping**: the misses
weren't on the critical path (they overlapped with worker decode
time, so eliminating them shifts the bottleneck elsewhere). This is
the "hidden assumption #7" the advisor flagged. Document and stop.

### Step 0a — reproducibility + critical-path check (in parallel with Step 0, 1 hour)

The 4-miss finding is from n=1. The adversarial advisor flagged TWO
risks:

(i) **chunk_id stability**: the SAME chunk_ids appear across runs?
(ii) **wait magnitude stability**: do the waits have the same size?

Run 20 traces (10 with `echo 3 > /proc/sys/vm/drop_caches` between
runs, 10 with warm page cache):

```bash
for cache_state in cold warm; do
  if [ $cache_state = cold ]; then
    ssh -J neurotic root@REDACTED_IP 'echo 3 > /proc/sys/vm/drop_caches'
  fi
  for i in $(seq 1 10); do
    ssh -J neurotic root@REDACTED_IP \
      "rm -f /tmp/gz.$cache_state.$i.json &&
       taskset -c 1,3,4,5,6,7,10,13,15 \
         env GZIPPY_TIMELINE=/tmp/gz.$cache_state.$i.json \
         /root/gzippy/target/release/gzippy -d -c -p 9 \
         /root/benchmark_data/silesia-gzip.tar.gz > /dev/null"
    ssh -J neurotic root@REDACTED_IP \
      "python3 /tmp/show_misses.py /tmp/gz.$cache_state.$i.json"
  done
done
```

**Pass criterion**: chunk_ids stable in ≥ 15 of 20 runs. Wait
magnitudes within 2× across runs for the same chunk_id.

**Fail criterion**: chunk_ids jitter OR wait magnitudes swing > 3×
between runs → the diagnosis itself is noisy and Step 0's brute-force
test is the only reliable signal.

**ALSO** (separate concern): for each miss in each trace, check
whether `block_finder.last_found_idx` was ≥ `miss_chunk_id` at the
miss moment. Requires adding a `BLOCK_FINDER_LAST_FOUND` atomic
counter, snapshotted into `wait.block_fetcher_get` span args. If
the finder was BEHIND the consumer at miss moment, the prefetcher
COULDN'T have helped because there was nothing to prefetch — the
fix needs to attack the finder, not the prefetcher.

### Step 1 — DELETED

Original plan had a "cold start fix" step with predicted-zero
savings. Adversarial advisor: "dead weight, delete it." Agreed.

Chunk 0's wait = decode time, structurally on the critical path,
cannot be shrunk without making chunk 0 itself smaller (compromises
elsewhere) or pre-decoding before consumer asks (requires changing
the API contract).

### Step 2 — mid-stream prefetch (REVISED, 3-4 hours)

#### Step 2a — full per-miss state instrumentation

Add a `prefetch.snapshot` instant event INSIDE
`block_fetcher.rs::get_with_prefetch`, emitted at the moment a miss
is detected (i.e., when `take_prefetch` returns None and on-demand
fetch is about to fire). Capture:

- `chunk_id` (the requested index)
- `prefetch_cache_size` (current cache occupancy)
- `prefetch_inflight_ids` (list of in-flight prefetch indices)
- `prefetch_completed_ids` (list of completed-but-not-taken prefetches)
- `last_fetched_idx` (the prefetcher's last dispatch index)
- `block_finder_last_found_idx` (the finder's confirmed-index high
  watermark)
- `time_since_drive_start_ms`

This is the **per-miss state snapshot** the adversarial advisor said
was missing. With this we can classify each miss into one of FOUR
buckets (vs the original plan's two):

1. **Finder-behind**: `block_finder_last_found_idx < miss_chunk_id`.
   Prefetcher couldn't have helped. Fix is in the finder, not the
   prefetcher.
2. **Prefetcher-behind**: finder ahead, but
   `last_fetched_idx < miss_chunk_id`. Prefetcher fell behind
   consumer rate.
3. **In-flight**: chunk was prefetched but still decoding when
   consumer asked. Wait time = decode time. Mitigation: dispatch
   prefetches earlier.
4. **Evicted**: chunk was completed but evicted from prefetch_cache.
   Bigger cache (or LRU eviction tuning) would help.

**Pass criterion**: each of the 4 misses gets a clean classification.

#### Step 2b — fix based on Step 2a, with explicit mechanism

Adversarial advisor: "your Case A fix risks repeating the killed
2x→4x prefetch_capacity regression unless you have a distinguishing
mechanism." Acknowledge. Specific mechanisms per class:

| Bucket | Mechanism | Distinguishes from killed 2x→4x how? |
| --- | --- | --- |
| Finder-behind | Force BlockFinder to confirm offsets aggressively at drive() entry | NOT a prefetch-side change |
| Prefetcher-behind | Dispatch N prefetches per consumer iteration instead of waiting for `should_drive_prefetch` re-trigger | Doesn't change cache CAPACITY, changes dispatch RATE |
| In-flight | Issue prefetches EARLIER (at drive() entry, burst N) so they finish before consumer asks | Doesn't change capacity, changes TIMING |
| Evicted | Sharded prefetch_cache to remove lock contention, then bigger cache | Killed 2x→4x might have regressed from lock contention, not capacity overhead — separate fix |

For each bucket present in the data, implement the matching fix
behind a feature flag (e.g., `GZIPPY_FIX_PREFETCH_DISPATCH`),
measure, keep or revert.

**Pass criterion (per bucket fix)**: wall drops by the predicted
band (Step 4 spelled out below); no regression on routing tests.

**Predicted savings (REVISED, honest band)**:

Per adversarial advisor: "your <350 ms criterion contradicts your
own ceiling math." Corrected:

- Realistic band: wall drops 50-100 ms (486 → 386-436 ms)
- Stretch (all buckets fix cleanly): 100-170 ms (486 → 316-386 ms)
- Best case lower than vendor 162 ms: not achievable from this plan
  alone; per-chunk decode rate is the missing 1.9× factor.

### Step 3 — pool eff + load balance re-measurement

Re-run after Step 2b succeeds. Predicted side-effects:

- `pool.pick` sum should drop 5-10× the wall savings (workers idle
  less when consumer doesn't block).
- per-thread busy time σ should shrink from 130 ms toward 50-80 ms
  (not vendor's 7 ms — that requires fixing per-chunk decode rate
  too).

### Step 4 — cross-tool diff verification

```bash
ssh -J neurotic root@REDACTED_IP \
  'python3 /tmp/timeline_analyze.py /tmp/gz.tl.json /tmp/rg.tl.json | head -60'
```

**Done criterion for this fix**:

- `wait.block_fetcher_get` ≤ 80 ms (from 210 ms baseline)
- Wall in the predicted band (386-436 ms realistic, 316-386 ms stretch)
- pool eff > 50% (from 45% baseline)
- `cargo test --release -- routing` green
- isal-compression path green (`cargo test --features isal-compression -- routing`)

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
ssh -J neurotic root@REDACTED_IP \
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

## Confidence ratings (REVISED)

| Step | Confidence in plan being right | Confidence in stated savings |
| --- | --- | --- |
| 0 (brute-force falsification) | HIGH | N/A — falsifies or confirms in 30 min |
| 0a (repro + critical-path check) | HIGH | N/A — finder-behind discriminates whole-plan validity |
| 1 (deleted) | n/a | n/a |
| 2a (per-miss 4-bucket classification) | HIGH (instrumentation is clean) | N/A |
| 2b (per-bucket fix with explicit mechanism) | MEDIUM (depends on bucket distribution) | MEDIUM (50-100 ms realistic, 100-170 ms stretch) |
| 3 (pool eff measurement) | HIGH (existing reads) | N/A |
| 4 (cross-tool diff) | HIGH (harness exists) | N/A |

Honest framing: the plan's realistic outcome is 50-100 ms wall
savings (486 → 386-436 ms). The stretch is 100-170 ms (486 → 316-386 ms).
**Neither closes the gap alone** — per-chunk decode rate is the
separate 1.9× lever and must be attacked next regardless.

## EXECUTION RESULT (2026-05-25)

**Step 0 falsification failed. Plan aborted.**

Mechanism tested (per advisor): `GZIPPY_BURST_PREFETCH=1` env raised
`BlockFetcher::new`'s `parallelization` argument from `pool_size`
(9) to `pool_size * 2` (18). This decouples the saturation gate at
`block_fetcher.rs:584` (`prefetching_len() + 1 >= parallelization`)
from worker count, allowing up to 17 in-flight prefetches at T=9.

Result on silesia T=9 P-cores (3-trial median, commit `821d917`):

| Metric | Baseline | BURST | Δ |
|---|---|---|---|
| Wall | 551 ms | 649 ms | **+18%** (REGRESSED) |
| `wait.block_fetcher_get` total | 242 ms | **319 ms** | **+31%** (WORSE) |
| Cache miss count | 4 | **5** | +1 |
| Miss chunk_ids | 0, 201, 1855, 2545 | 0, 1543, 1790, 2342, 2622 | shifted |

Both dual-gate criteria failed. Per the plan's binding anti-mistake
rule "implement + measure beats theorize", abort and reallocate to
per-chunk decode rate work.

Observation worth keeping (NOT to chase under this plan): the miss
chunk_ids SHIFTED rather than disappearing. More in-flight prefetches
→ more pool contention → consumer waits in NEW places. This is
consistent with "prefetch saturation isn't the right lever" rather
than "prefetch saturation needs further tuning."

The env-var knob is left in place behind `GZIPPY_BURST_PREFETCH` for
future reproducibility (defaults to baseline when unset).

Next work: per-chunk decode rate (`plans/pure-rust-perf.md` Verified
Finding #4 — gzippy 47 ms/chunk vs rapidgzip 25 ms/chunk).

## Adversarial-advisor feedback log

For traceability if compacted:

1. **Step 0 was n=1**, advisor flagged single-trace risk. Revised
   Step 0a to 20 traces (10 cold + 10 warm cache) AND check wait
   magnitude stability, not just chunk_id stability.

2. **Step 1 was dead weight** — chunk 0 wait = decode time, no fix
   possible. Advisor: "delete or downgrade to 5-min sanity check."
   Deleted.

3. **Step 2a needed more per-miss state**, not just `last_fetched_idx`.
   Revised to 7 fields per miss (inflight, completed, finder
   high-water, etc.) supporting 4-bucket classification (not 2).

4. **Step 2b Case A risked repackaging killed 2x→4x experiment.**
   Revised with explicit mechanism table distinguishing dispatch
   RATE from cache CAPACITY (the failed knob).

5. **Pass criterion <350 ms contradicted ceiling math.** Advisor:
   "fraudulent." Corrected to honest 386-436 ms realistic band,
   316-386 ms stretch.

6. **Cheapest falsification < 30 min wasn't in original plan.**
   Added as new Step 0 (brute-force prefetch_cache=64 + burst
   dispatch). If theory holds at scale, proceed; if not, abort
   without doing instrumentation work.

7. **Hidden assumption: misses on critical path.** Advisor: "what
   if misses overlap with worker decode time, so eliminating them
   shifts the bottleneck?" Added explicit critical-path check via
   `block_finder.last_found_idx` snapshot. If finder is behind
   consumer at miss moment, prefetcher couldn't have helped — fix
   is in the finder, not the prefetcher.

8. **Prefetch_cache lock contention as alternative.** Advisor noted
   the killed 2x→4x might have regressed from lock contention, not
   capacity overhead. Sharded map is a separate fix listed in Step
   2b's Evicted bucket.
