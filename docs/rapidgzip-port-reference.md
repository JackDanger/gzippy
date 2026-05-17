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
| `75f5f52` | 716 | 0.40× | + skip-direct, always iterate via find_blocks; hit 90% |
| `185d572` | 721 | 0.39× | (post HuffmanCodingISAL FFI infra) revert wiring; perf restored |
| `a22e305` | 700 | 0.38× | (cleanDataCount handoff tried + reverted — needs decoder amortization) |
| `7919d8a` | 708 | 0.38× | + remove BTYPE filter at handoff (matches rapidgzip literally) |
| `432e883` | 708 | 0.38× | (cap bootstrap output tried + reverted — wasted speculative work) |
| `e32cf94` | 560 | 0.28× | full deflate::Block hot loop ported (IsalLitLenCode + IsalDistCode); per-block ISA-L table rebuild dominates L1 cache (4% cache hit rate on Silesia per-chunk reuse) — literal port now lives in production despite measured perf cost |

## Why the 2.5× gap persists

This branch has ALL the literal-port pieces of rapidgzip's discrete
algorithms (block finder, candidate iteration, consumer pattern, LUT
generator, BitReader). What's missing is the **integrated marker
decoder + ISA-L handoff loop**.

The hard chunks (9/38 on Silesia gzip -9) decode 12 MB output via
pure-Rust marker decoder because trailing-clean never accumulates
(marker propagation through back-refs). Rapidgzip handles these by
handing off at `cleanDataCount >= MAX_WINDOW_SIZE` (total clean,
not trailing), then having `getLastWindow({})` throw if the window
has markers, then having `tryToDecode` catch and try the next
BlockFinder candidate.

For our architecture, the recovery-on-throw path triggers a
speculation-miss cascade: the consumer falls back to authoritative
re-dispatch (serial chain), which is slower than running the full
bootstrap in parallel. The fix needs ALL THREE TOGETHER:

1. `IsalLitLenCode` decoder (ported; needs per-chunk amortization).
2. `cleanDataCount` handoff (ported; reverted because of cascade).
3. Recovery path that DOESN'T cascade speculation misses
   (e.g., return marker-only chunk on marker-tail rather than error,
   accepting some bootstrap work that doesn't hand off but doesn't
   force serial auth chain).

Atomically landing these three is the remaining work. Each was
attempted in isolation this session and reverted (commits ff56db2,
d08239c, df48369) because each in isolation regresses perf.

## Opus advisor diagnostic (commit b5ecdf5 attempt)

Opus advisor reviewed why IsalLitLenCode integration consistently
regresses (USER time drops, WALL time rises, pool util drops 22% → 8%).
**Root cause: L1 cache thrash from rebuilding the 19 KB
`inflate_huff_code_large` table per dynamic block**.

- Silesia gzip -9 has many small blocks (~few-KB output each).
- Per-block table build (`set_and_expand_lit_len_huffcode` +
  `make_inflate_huff_code_lit_len`) writes ~19 KB.
- 16 threads × thread-local 19 KB tables × per-block rebuild = constant
  L1 eviction storm.
- USER time drops because cache-miss stalls don't count as CPU work;
  WALL rises because threads stall on memory.

Cache-by-code-lengths fix (commit b5ecdf5): measured **4% hit rate**.
Silesia gzip -9 blocks have unique code lengths — depth-1 cache is
useless. Deeper LRU may help marginally but won't close the gap.

## Real fix (per advisor)

**Port rapidgzip's full `deflate::Block`** (~2000 lines, file
`vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp`).
Their inner loop:
- Pre-loads bit-buffer refills
- Decodes 2 symbols at a time when both have short codes
  (HuffmanCodingDoubleLiteralCached)
- Uses ISA-L's distance table too (HuffmanCodingDistanceISAL)
- Reuses huffman objects across consecutive same-code-length blocks
  via internal hash

The per-block-rebuild model (what we have) is the wrong SHAPE for
the data. Closing the 2.5× gap requires the per-block-decoder model
match rapidgzip's structure end-to-end, not piecemeal optimization
of the current shape.

Alternative if `deflate::Block` port is too costly: investigate
whether ISA-L exposes a "decode one block stopping at EOB" API that
manages its own tables internally. If so, that's the same code path
ISA-L's `isal_inflate` uses for normal decode (just terminated at EOB
boundaries). This sidesteps the per-block table rebuild entirely.

## Infrastructure landed for next iteration (commits 6b9d754, 1d19f21)

- `crates/isal-sys-patched/src/lib.rs` `isal_internals` module exposes
  `set_and_expand_lit_len_huffcode` + `make_inflate_huff_code_lit_len`
  + `huff_code` struct via direct FFI (matching the patch rapidgzip
  applies to vendored ISA-L).
- `src/decompress/parallel/isal_huffman.rs` `IsalLitLenCode` — Rust
  port of `HuffmanCodingISAL.hpp` (table build via FFI, decode via
  short_code_lookup/long_code_lookup tables). Thread-local instance
  for per-thread allocation amortization.

### Why integrating it didn't yield perf yet

Wiring `IsalLitLenCode` into `decode_dynamic` regressed perf (0.40× →
0.28×) because `make_inflate_huff_code_lit_len` is called per dynamic
block (~7000 blocks across all chunks on Silesia). Rapidgzip avoids
this cost by bootstrap-handing-off after just 1–2 blocks
(`cleanDataCount >= MAX_WINDOW_SIZE`), then running ISA-L bulk with
its own internal tables (no per-block Rust builds).

Porting that handoff (`a22e305`) alone regresses too, because chunks
that previously bootstrap-completed-as-marker-only now error →
speculation miss → consumer falls into serial authoritative chain.

The fix needs ALL THREE TOGETHER:
1. `IsalLitLenCode` in the bootstrap (fast per-symbol decode).
2. `cleanDataCount` handoff (stop bootstrap after ~1–2 blocks).
3. Worker recovery path when handoff window has markers (per
   rapidgzip's `tryToDecode` catch + iterate to next candidate AND
   fall back to bootstrap-to-completion on exhaustion).

This is task #23's next concrete piece.

## Remaining bottleneck: bootstrap throughput

Per-chunk breakdown on Silesia gzip -9 (T=16, 38 slow-path chunks):
- median slow-path decode: 108 ms
- median markers (pure-Rust marker-decode): 1.1 MB output
- median clean (ISA-L bulk): 8.6 MB output
- **9 chunks (24%) do EVERYTHING via bootstrap** (clean=0): never accumulate
  a marker-free trailing 32 KiB, so handoff never triggers.
- Pool utilization: 51% (= 4.5 sec CPU / 16 / 0.55 sec wall).

Effective per-thread bootstrap throughput: ~14 MB/s output. Rapidgzip's
deflate::Block is ~113 MB/s (= 1800 MB/s ÷ 16 if all chunks bootstrapped).
The 8× gap is the next phase of the port: speed up `fast_marker_inflate`
or replace with a deflate::Block-equivalent decoder.

Rapidgzip's handoff condition is `cleanDataCount >= MAX_WINDOW_SIZE`
(total clean ≥ 32 KiB), NOT "trailing 32 KiB is clean" like ours. On
markers in last 32 KiB at handoff, getLastWindow throws via MapMarkers
(empty previousWindow); tryToDecode catches and tries next candidate.
Porting this exactly would let us hand off earlier in marginal cases.

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
