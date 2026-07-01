# Dead End: Cache-coherence / lock-contention

## Hypothesis

The 1.47× T8-noSMT wall gap is caused by cache-coherence traffic (false sharing,
HITM — "hit in modified" cross-core invalidations) or lock contention on gzippy's
shared structures (window_map Mutex, thread_pool Mutex, mpsc handoff, global stat
counters).

Motivation: IPC collapses 2.15→1.06 as thread count grows (T1→T8), suggesting
contention. The gap onsets with T (1.10 at T1, 1.47 at T8), which is a contention
signature.

## How Measured

`perf c2c` (cache-to-cache transfer profiling), run on BOTH gzippy and rapidgzip at
T8 and T16 on the frozen-clock neurotic harness. (2026-06-02, branch
`feat/dataplane-2touch`, Fulcrum 3-bucket+TMA+static instrument.)

Results:
- HITM fraction: **0.05–0.15% of loads** in gzippy — tiny.
- **gzippy ≈ rapidgzip at T16** (35.7 vs 37.0 HITM count per run) — TIE where the
  IPC gap is LARGEST.
- Every contended cacheline was **kernel-side**: `lock_vma_under_rcu`, page-allocator,
  memcg, scheduler. Zero gzippy application structures or atomics appeared.

The shared gzippy structures (window_map, thread_pool, mpsc handoff) are
byte-for-byte vendor ports with the same locking patterns. The ~40 global AtomicU64
stat counters use Relaxed ordering and are per-chunk (too cheap to surface in c2c).

## Verdict: REFUTED (5th footprint/contention refutation)

Cache-coherence and lock-contention are **not gzippy-specific**. rapidgzip has
essentially identical HITM at T16 where the IPC gap is largest. The kernel-side
contention (page allocator, scheduler) affects both tools identically.

The IPC collapse signature is real but is **fundamental in-order-consumer
coordination** (more workers → more in-flight speculative chunks → more out-of-order
completion the consumer serializes through), not recoverable via coherence tuning.

## Do Not Re-attempt

- False-sharing padding on shared structures: rapidgzip has the same structures with
  the same patterns; any gzippy-specific padding would diverge from the vendor port
  without a measured gzippy-specific hotspot.
- Lock-free replacements for window_map / thread_pool: already uses BTreeMap under
  Mutex (same as vendor). Per `c2c`, these structures do not appear in the contended
  cacheline list.
- AtomicU64 stat counter contention: too rare to surface in profiling.

## Related Entries

- `docs/dead-ends/footprint-bandwidth.md` — the broader memory/footprint theory
- `project_t8_gap_fully_mapped_2026_06_02` memory — full 3-bucket+TMA mapping
  including coherence refutation
