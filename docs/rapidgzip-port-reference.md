# rapidgzip → gzippy parallel single-member port: one-page reference

Last verified against HEAD `1e21b1e`. If the code below disagrees with
what `grep` finds, the code wins — update this file.

## What lives where

```
src/decompress/parallel/
├── single_member.rs    # public entry: decompress_parallel + skip_gzip_header
├── chunk_fetcher.rs    # port of GzipChunkFetcher + BlockFetcher (worker pool, prefetch)
├── chunk_data.rs       # port of ChunkData + Subchunk
├── gzip_chunk.rs       # port of GzipChunk: decode_chunk_with_window (fast)
│                       #     + finish_decode_chunk_with_inexact_offset (slow)
├── inflate_wrapper.rs  # port of IsalInflateWrapper
├── window_map.rs       # port of WindowMap (mutex + condvar)
├── block_finder.rs     # port of blockfinder/DynamicHuffman.hpp
├── apply_window.rs     # port of ChunkData::applyWindow
├── replace_markers.rs  # SIMD marker resolution (AVX2 / NEON)
├── fast_marker_inflate.rs  # gzippy slow-path equivalent of rapidgzip's
│                            # deflate::Block (pure-Rust marker decoder)
├── trace.rs            # gzippy-specific: GZIPPY_LOG_FILE structured trace
└── mod.rs
```

Rapidgzip source under `vendor/rapidgzip/librapidarchive/src/rapidgzip/`.
(NOT `vendor/rapidgzip/src/rapidgzip/` — that path doesn't exist.)

## Live trace events

Set `GZIPPY_LOG_FILE=/path/to/log.jsonl` and grep / `scripts/parallel_sm_log_summary.py`. Events as of HEAD `ed0fee7`:

| event | thread | when |
|-------|--------|------|
| `authoritative_prefetch` | consumer | submits chunk N+1's authoritative dispatch |
| `fast_path_start` | worker-N | worker takes the fast path (window known) |
| `slow_path_start` | worker-N | worker takes the slow path (no window) |
| `decode_ok` | worker-N | worker completes successfully |
| `decode_err` | worker-N | worker fails (passes Err back to consumer) |
| `authoritative_prefetch_wait` | consumer | consumer blocking-recv on in-flight prefetch |
| `apply_window_done` | consumer | apply_window finished for a chunk with markers |
| `consume_done` | consumer | one chunk fully processed; expected_start updated |

## Architecture (current)

Pure depth-2 authoritative chain (advisor's recommendation, commit `a8bc533`):

1. Consumer submits `pending_auth[0]` at `start=0`.
2. While processing chunk N (apply_window/write/window-insert), submits `pending_auth[N+1]` at `start = chunk N's actual_end`.
3. Worker for N+1 fast-paths because predecessor's window is in `WindowMap`.
4. When iteration advances to N+1, consumer waits on the in-flight prefetch.

No speculative pre-scan, no triple-fallback consumer logic. Worker's
`decode_or_iterate` (chunk_fetcher.rs) ports rapidgzip's `tryToDecode`
at `GzipChunk.hpp:712-841`: tries direct at `start_bit` first, then
iterates `BlockFinder::find_blocks` candidates within 512 KiB.

## Measured baseline

Silesia gzip -9 (162 MB → 503 MB, T=16, neurotic homelab x86_64):

| HEAD | gzippy MB/s | vs rapidgzip | notes |
|------|-------------|--------------|-------|
| `ed0fee7` | 574 | 0.30× | clean depth-2 chain, all non-rapidgzip code deleted |
| `1e21b1e` | 602 | 0.33× | BlockFinder dynamic-only (matches rapidgzip), +5% |
| `7994675` | 645 | 0.35× | partition-seed-keyed speculation (rapidgzip BlockFetcher) |
| `9672baf` | 628 | 0.34× | + ported `seekToNonFinalUncompressedDeflateBlock` |
| `88b84a4` | 383 | 0.19× | + literal `nextDeflateCandidate` LUT (regressed, fixed below) |
| `7458880` | 344 | 0.25× | + 15-bit LUT |
| `84883ae` | 675 | 0.45× | + BitReader refills before peek(57); hit rate 62→82% |

rapidgzip ground truth (`--verbose`):
- Pool Efficiency 77%.
- Prefetch Cache: **32 hits / 1 miss = 97.4% hit rate**.
- decodeBlock 1.62s aggregate CPU; wall 0.15s @ 16T.
- 38 prefetched chunks, 1 fetched on-demand.

The 0.30× gap is in the BlockFinder candidate quality. Rapidgzip's
first-candidate-past-partition-seed is on the natural decode path 97%
of the time; ours is 8/38 = 21%. The depth-2 chain saturates because
all chunks beyond chunk-0 take the fast path with a known window —
but a fast path that can't run more than 2 in parallel ≈ 2-core
throughput on a 16-core machine.

## Closing the rest of the gap

The serial-chain ceiling is broken if BlockFinder candidates land on
the natural decode path. Then partition-seed-keyed prefetch (the
real rapidgzip pattern) can use the pool's full width.

Strategy: literal port of `blockfinder/DynamicHuffman.hpp`
(`seekToNonFinalDynamicDeflateBlock`) to match rapidgzip exactly,
including the 15-bit LUT and the in-worker tryToDecode candidate
iteration. After that, switch to partition-seed-keyed prefetch
matching `BlockFetcher::get` / `GzipChunkFetcher::processNextChunk`.
