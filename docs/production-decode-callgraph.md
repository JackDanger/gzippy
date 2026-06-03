# Production decode call-graph (pure-Rust, x86_64)

**Last updated: 2026-06-03.** Authoritative routing: [`production-paths.md`](production-paths.md).

## Entry (what the CLI runs)

```
decompress_gzip_libdeflate → classify_gzip → DecodePath::IsalParallelSM
  → single_member::decompress_parallel → sm_driver::read_parallel_sm
  → chunk_fetcher::drive
```

Build: `--no-default-features --features pure-rust-inflate` → `pure_inflate_decode` →
**no C ISA-L in the decode graph**. `IsalInflateWrapper` is a name; it wraps
`ResumableInflate2` (`inflate_wrapper.rs`).

## Per-chunk decode (same *shape* as rapidgzip)

**One function:** `gzip_chunk::decode_chunk_with_rapidgzip_impl` (vendor
`decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset`).

It is **not** the refuted “two-pass scan + re-decode” architecture. It **is** a
**two-stage loop inside each chunk**:

| Stage | When | Code | Output |
|-------|------|------|--------|
| **1. Marker bootstrap** | No exact predecessor window at worker start | `marker_decode_step` → `deflate_block::Block` | u16 into `chunk.data_with_markers` until 32 KiB clean at a block boundary |
| **2. Clean streaming inflate** | After handoff (or immediately if 32 KiB window known) | `finish_decode_chunk_inexact_offset` → `ResumableInflate2::read_stream` | u8 into `chunk.data` (segmented 128 KiB tails) |

rapidgzip uses the **same outer shape** (marker bytes until clean window, then fast
stream decode on the same chunk). The gap is **implementation speed** (Rust marker
ring + resumable inflate vs vendor’s fast paths), plus gzippy-specific **post-process**
(`apply_window` / `narrow_markers_in_place`) and **publish-chain** latency — not a
missing second stage.

Legacy `bootstrap_with_deflate_block` remains in the tree for tests/trials; **production
workers call `decode_chunk` / `decode_chunk_window_absent` → `decode_chunk_with_rapidgzip_impl`.**

## Worker dispatch (`chunk_fetcher::run_decode_task`)

```
pool.run_task → worker.decode_chunk
  ├─ chunk 0 or window_map.get(start_bit) exact  → decode_chunk (often skips stage 1)
  ├─ Design H handoff window (spec prefetch)     → decode_chunk @ handoff key (usually 0 hits)
  └─ else                                        → speculative_decode_find_boundary
       → decode_chunk_window_absent (stage 1 → 2)
```

## After decode (still on critical path for wall)

```
post_process.task: resolve_chunk_markers (apply_window) + narrow_markers_in_place + CRC
consumer: wait for chunk → write output + publish 32 KiB tail window (serial chain)
```

`append_markered` / `absorb_isal_tail` were **removed from the hot unified path**; markers
are written via `MarkerSink` during stage 1. Narrowing still runs in post-process when markers
are present.

## NOT production (do not optimize as “shipping”)

See [`production-paths.md`](production-paths.md) §5 — oracles, bypass, `GZIPPY_CLEAN_WINDOW_ORACLE`
(validate instrument before trust), experimental modules not on the trace tree.

## Regenerate the live tree

```bash
GZIPPY_TIMELINE=/tmp/t.json.gz GZIPPY_FORCE_PARALLEL_SM=1 \
  gzippy -d -c -p8 /path/to/silesia-large.gz > /dev/null
# then timeline_analyze.py --tree on the guest/host artifact
```
