# Parallel profiling plan — deterministic bottleneck illumination

> **Problem**: gzippy at T=1 *beats* rapidgzip, but at Tmax it loses
> by ~2×. The bottleneck is in parallel scheduling, contention, or
> consumer-side serialization — invisible to the existing tools.
>
> **Constraint**: must be deterministic, not statistical. perf record
> samples at 999 Hz and misses sub-ms events. We need per-event
> ground truth.

## The current tools and what they DON'T show

| Tool | What it shows | What it misses |
|---|---|---|
| `bench_stats.py` | End-to-end throughput + CI | Where the time went |
| `trace_diff.py` | Per-phase summary | Per-thread timeline, idle gaps |
| `profile_compare.sh` (perf record) | On-CPU hot functions | Off-CPU time (locks, channel waits) |
| `static_map.py` | Symbol coverage | Runtime behavior |

The Tmax-loss bug almost certainly lives in OFF-CPU time and per-thread
IDLE gaps that perf sampling can't see directly. Three deterministic
tools fill the gap.

## Tool 1 — Per-thread timeline (deterministic Gantt chart)

**`scripts/timeline.py`** + enhanced `trace.rs` events.

### What we add to `src/decompress/parallel/trace.rs`

Already emitting (post commit `ea15a5e`):
- `phase_a_done`, `phase_bc_done`, `drive_total_done`
- `chunk_decode_done` (worker)
- `decode_err`, `worker_exhausted`
- `apply_window_done`, `chunk_consume_done` (consumer)

**Add** (this plan):
- `worker_pull_start{worker_id}` / `worker_pull_done{worker_id, idx}` — bracket the lock-free pool's `pop()` call.
- `worker_idle{worker_id, duration_us}` — when worker found job queue empty and exited the loop. Tells us how much workers idled at end-of-stream.
- `consumer_wait_start{partition_idx}` / `consumer_wait_done{partition_idx, wait_us}` — bracket `rx.recv()` for each chunk's per-channel pull. The `wait_us` is the time consumer was BLOCKED waiting for worker N. Critical for "is consumer the bottleneck?".
- `buffer_pool_pop{worker_id, was_empty}` — every `pool.pop()` call. `was_empty=true` means worker had to alloc fresh.
- `buffer_pool_push{partition_idx}` — every recycle event.

All gated behind `GZIPPY_LOG_FILE`. Trace overhead is ~5μs per emit (mutex+vec push) — negligible when on, zero when off.

### What `scripts/timeline.py` outputs

1. **ASCII Gantt chart**: one row per thread (worker-0..worker-N + consumer). X-axis = wall time in ms. Characters: `=` decoding, `_` idle, `W` waiting on lock/channel, `A` apply_window.

   ```
   worker-0  |======================================_________________________|
   worker-1  |________======================WWWWWWWWWWWWWWW___________________|
   worker-2  |________________======WWW___________________________________|
   consumer  |______AAAAWWWWWWWWAAAAWWWWWWWWAAAA===========================|
   ```

2. **Critical-path analysis**: identify the chunk whose completion
   forced the wall to be wall_max. If consumer was idle waiting, that
   chunk's worker was the bottleneck. If consumer was busy when chunk
   arrived, the consumer was the bottleneck.

3. **Per-event summary**:
   - Total worker decode time (CPU-summed)
   - Total worker idle time (CPU-summed)
   - Total consumer apply_window time
   - Total consumer wait-for-recv time
   - Worker utilization (decode / (decode + idle))
   - Consumer utilization (work / (work + wait))

### What it reveals deterministically

- If `worker idle` summed across all workers ≈ wall × pool_size × 0.5 → workers are idle 50% of the time → consumer serializes.
- If `consumer wait` >> 0 → consumer was blocked on recv → workers were too slow OR scheduled badly.
- If `buffer_pool_pop was_empty=true` count > pool_size → recycle not keeping up → memory churn.

No sampling. No 999 Hz misses. Every microsecond accounted for.

## Tool 2 — Scaling sweep (T=1, 2, 4, 8, 16)

**`scripts/scaling_sweep.sh`** — wraps `bench_stats.py` + `timeline.py`.

For each T in {1, 2, 4, 8, 16}:
1. Run bench (N=10 trials) → median throughput.
2. Run gzippy with `GZIPPY_LOG_FILE` → timeline analysis.
3. Compute `actual_speedup(T) = throughput(T) / throughput(1)`.
4. Compute `ideal_speedup(T) = T`.
5. `efficiency(T) = actual / ideal`.

### Output table

```
T  | gzippy MB/s | rapidgzip MB/s | gzippy speedup | rapidgzip speedup | gzippy/rapidgzip
---|-------------|----------------|----------------|-------------------|------------------
1  |  600        |  500           | 1.00×          | 1.00×             | 1.20× (gzippy wins!)
2  |  900        |  800           | 1.50×          | 1.60×             | 1.13×
4  | 1300        | 1500           | 2.17×          | 3.00×             | 0.87×
8  | 1700        | 2400           | 2.83×          | 4.80×             | 0.71×
16 |  850        | 1700           | 1.42×          | 3.40×             | 0.50×  <- scaling cliff
```

The "cliff" — the T where efficiency suddenly drops — names the
contention point. If efficiency drops between T=8 and T=16, the
16-thread workload is hitting some shared resource (allocator,
lock, channel).

### What it reveals deterministically

- If gzippy scales perfectly to T=4 then plateaus → 4-thread sweet
  spot, hint at NUMA / cache.
- If gzippy scales worse than rapidgzip across the board →
  per-thread overhead too high.
- Combined with timeline.py per-T: shows whether contention rises with T.

## Tool 3 — Off-CPU flamegraph (where threads are SLEEPING)

**`scripts/off_cpu_profile.sh`** — Linux perf sched events.

Standard perf record samples while threads RUN. Off-CPU profiling
samples while threads are BLOCKED (sleeping on locks, channel recvs,
page faults stalled, etc.). The two together account for 100% of
each thread's wall time.

### Commands (on neurotic)

```bash
sudo perf record -e sched:sched_switch,sched:sched_stat_sleep \
                 --call-graph=dwarf -o /tmp/g.offcpu.data -- \
    ./target/release/gzippy -d -P 16 < benchmark_data/silesia-large.gz > /dev/null
sudo perf script -i /tmp/g.offcpu.data | \
    inferno-collapse-perf --offcpu | \
    inferno-flamegraph --title "gzippy off-CPU" > docs/runs/offcpu.svg
```

### What it reveals deterministically

- Wide flame at `std::sync::mpsc::Receiver::recv` → consumer blocked on workers.
- Wide flame at `parking_lot::Mutex::lock_slow` → lock contention.
- Wide flame at `__do_page_fault` → page-zero stalls.
- Wide flame at `futex_wait_queue` → kernel-level condvar contention.

A single off-CPU flamegraph from a Tmax run side-by-side with on-CPU
will name the dominant block point.

## Implementation order

1. ✅ This plan doc (DONE — you're reading it).
2. Enhance `trace.rs` with the 5 new events. ~30 lines.
3. `scripts/timeline.py` — ingest + emit ASCII Gantt + summary. ~150 lines.
4. `scripts/scaling_sweep.sh` — bash loop over T values invoking
   bench_stats + timeline. ~50 lines.
5. `scripts/off_cpu_profile.sh` — perf record + inferno wrapper. ~30 lines.

## Hand-off after implementation

Each commit lands one tool. After all four:

```bash
make rapidgzip-scaling-sweep
# Runs all tools at all T values. Emits a single docs/runs/scaling-TIMESTAMP/
# directory with: bench.md per T, timeline.txt per T, offcpu.svg per T,
# scaling.md summary, and next_action.prompt.md naming the cliff.
```

The next operator should be able to read `scaling.md` + one timeline.txt
and immediately know whether the Tmax bottleneck is:
- Consumer-side serialization (consumer wait >> 0)
- Worker contention (lock-slow flame in off-CPU)
- Memory allocator (page-fault flame in off-CPU)
- Channel backpressure (worker idle waiting on full sync_channel)

No guessing. Profile says "fix X" and the next change targets X
exactly.

## What this is NOT

- Not a fix. The plan illuminates; the operator (or next sub-agent)
  fixes.
- Not statistical. The trace events are ground-truth; the bench numbers
  are bootstrap-CI'd as before.
- Not free. Trace overhead under `GZIPPY_LOG_FILE` is ~5μs/event,
  ~5ms over a full Silesia decode — negligible vs the 400ms wall.

## Why the existing tools couldn't have answered this

The Tmax bug ("T=1 wins, T=16 loses") is a SCALING bug. Single-run
tools (bench_stats, trace_diff) at one T value can't see it. Sweep
across T is required.

The dominant cost at Tmax is OFF-CPU (threads waiting), not ON-CPU
(threads computing). perf record at 999 Hz samples on-CPU only —
shows where the running thread is, not where the blocked threads
went. Off-CPU flamegraph is the only tool that captures the wait.
