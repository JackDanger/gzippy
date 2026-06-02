# Dead End: Over-marking / excess window-absent speculation fraction

## Hypothesis

gzippy decodes more chunks window-absent than rapidgzip (runtime fraction 90–95%
observed vs rapidgzip's ~31% static symbol-level replacement rate), suggesting
gzippy "over-marks" — emits u16 markers for chunks that could have been decoded
clean. Reducing this excess fraction would proportionally cut the bootstrap decode
cost and close the wall gap.

Two related phantom claims:

1. **62% vs 31% non-marker scare** (measurement artifact): a
   counter called `non_marker_count` in `ChunkStatistics` appeared to show gzippy
   had a very different marker fraction from rapidgzip.

2. **Runtime 95% vs static 31%** (real but not a lever): the causal trace showed
   95% of chunks decode window-absent at runtime vs the 31% static symbol-level
   replacement fraction in rapidgzip's verbose output.

## How Measured

**Static fraction verification**: rapidgzip verbose output reports "Replaced marker
symbol buffers 31.25%"; gzippy decodes 31.97% window-absent by the SAME metric.
gzippy's clean-window arming condition is a byte-for-byte port of
`deflate.hpp:1282-1284` ↔ `deflate_block.rs:781-783`, including the 64 KiB / no-marker-ever
clauses (vendor comments the 64 KiB is deliberate). The static fraction MATCHES vendor.

**Counter audit**: `non_marker_count` is misnamed — it counts the whole
window-absent VOLUME, not non-markers. gzippy's marker symbols are actually
**FEWER** than rapidgzip: 25.5% vs 31.46% by the static symbol-level metric.

**Runtime fraction explanation**: 95% of chunks decode window-absent at runtime
because workers race the in-order consumer and always win — workers prefetch 2–13
chunks ahead of where the consumer has resolved and published windows. The static
31% is a symbol-level metric across the compressed stream; the 95% is a chunk-level
metric at decode time. These are different quantities measuring different things.
The runtime 95% is a CONSEQUENCE of the speculation depth, not a cause.

**Reducing the fraction**: bound-prefetch-depth experiments showed that capping
speculation to reduce the window-absent fraction STARVES the consumer
(D6 named worse-measurement: consumer `wait.block_fetcher_get` EXPLODES 4→53 ms
as k drops from unbounded to 1). The deep speculation is load-bearing.

## Verdict: NOT A LEVER — fraction matches vendor at static level; runtime excess is structural

- Static symbol-level fraction: MATCH with vendor (31.97% vs 31.25%)
- Runtime chunk-level fraction: 95% vs vendor's "88% clean" — different metric
  (chunk-level vs symbol-level); the runtime excess is caused by speculation depth,
  not over-marking
- Reducing the fraction INCREASES starvation (D6)
- gzippy marks FEWER symbols than rapidgzip at the symbol level (25.5% vs 31.46%)

Reducing the window-absent fraction would be UNFAITHFUL to the vendor port and is dead.

## Code Locations

- `src/decompress/parallel/deflate_block.rs:781-783` — clean-window arming (port of
  `vendor/rapidgzip/librapidarchive/src/rapidgzip/deflate.hpp:1282-1284`)
- `src/decompress/parallel/chunk_data.rs` — `ChunkStatistics::non_marker_count`
  (misnamed: counts window-absent volume, not non-marker count)

## Related Entries

- `docs/dead-ends/placement-resolve-ahead.md` — the 95% runtime fraction is the
  speculation-depth symptom the placement oracle measured (94% already in-flight)
- `project_t8_gap_fully_mapped_2026_06_02` memory — DEAD list: "over-marking (gzippy
  marks FEWER than rapidgzip 25.5% vs 31.46%)"
