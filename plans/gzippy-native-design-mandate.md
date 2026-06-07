# gzippy-native DESIGN MANDATE (supervisor addendum, user /goal 2026-06-06)

AUGMENTS the leader charter for PHASE 2 `gzippy-native`. This is a FIRST-CLASS
design goal, co-equal with speed — not an afterthought.

## The cache-residency mandate
`gzippy-native` must be an awesome parallel decoder whose **aggregate memory
footprint across ALL threads and ALL decode-path components is so small it is
hot-in-cache much of the time** — at 8, 16, or 32+ threads. The decode path
should use a **shared, small, hot working set**, not N independent large
per-thread buffers.

### Implications (design rules)
- **Share read-only structures across threads**: one copy of Huffman LUTs /
  decode tables (the ISA-L hot-technique tables), referenced by all workers — not
  rebuilt or duplicated per thread.
- **Tiny per-thread working set**: per-thread mutable state (ring buffer, marker
  buffer, bit reader) sized to stay in L1/L2; avoid the large per-thread
  128 KiB+ scratch that blows the cache. Prefer compact representations.
- **Pooled / reused buffers**: no per-chunk large allocations; reuse small
  thread-local scratch; total RSS should grow ~sublinearly with thread count.
- **One engine, one cursor** (the governing-memory faithful pattern) so there is
  no second-engine duplicate state for the clean tail.
- ISA-L's hot techniques (BMI2 PEXT/BZHI, multi-symbol LUT, lean refill) are
  STOLEN into this single shared engine — chosen/laid-out for cache-residency.

### Measurement (add to the 3-way Fulcrum for gzippy-native)
Beyond wall: capture **total RSS**, **per-thread working-set size**, and
**cache behavior (L2/L3 MPKI, mem-stall)** vs thread count (T1/T8/T16). The win
is "aggregate working set fits in cache at high T" — measure it, don't assert it.
Target: per-thread working set + shared tables ≪ L2/L3; RSS roughly flat as T rises.

## Expansive support
The leader has explicit authorization to spawn AS MANY Opus subagents as needed —
design, implementation, review/feedback, cache/perf profiling, measurement —
generously. Do not under-resource this. (Still strictly serialize heavy BUILDS:
one cargo/rustc/bench at a time, prefer neurotic — the Mac OOM'd once.)

## Status
Supervisor (parent) will verify this mandate is incorporated into the Phase-2
gzippy-native design before implementation, and that the 3-way Fulcrum reports
the memory/cache metrics above for gzippy-native.
