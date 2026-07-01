# Production decode call-graph (pure-Rust, x86_64)

**Last updated: 2026-06-05.** Authoritative routing: [`production-paths.md`](production-paths.md).

## Entry (what the CLI runs)

```
decompress_gzip_libdeflate → classify_gzip → DecodePath::IsalParallelSM
  → single_member::decompress_parallel → sm_driver::read_parallel_sm
  → chunk_fetcher::drive
```

Build: `--no-default-features --features pure-rust-inflate` → `pure_inflate_decode` →
**no C ISA-L in the decode graph**. `IsalInflateWrapper` is a name; it wraps
`ResumableInflate2` (`inflate_wrapper.rs`) and is now reached **only** on the
bad-speculative-seed fallback (`resumable_resync`), never on the clean tail.

## Per-chunk decode (same *shape* as rapidgzip)

**One function:** `chunk_decode::decode_chunk_with_rapidgzip_impl` →
`decode_chunk_unified_marker` (vendor single `deflate::Block<containsMarkerBytes>`,
`decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset`).

It is **not** the refuted “two-pass scan + re-decode” architecture, and (since
2026-06-05) it is **not** two engines either: it is **one `MarkerRing`, one bit
cursor**, with an in-place marker→clean flip — the two phases below are the SAME
ring continuing, not a handoff to a second decoder.

| Phase | When | Code | Output |
|-------|------|------|--------|
| **1. Marker arm** | window absent (`contains_marker_bytes()`) | `marker_decode_step` → `isal_lut_bulk::MarkerRing` | u16 into `chunk.data_with_markers` |
| **2. Clean arm** | after the flip (`!contains_marker_bytes()`, vendor `!m_containsMarkerBytes`) | **same** `MarkerRing` on the **same cursor** → `CleanTailSink` narrows u16→u8 | u8 into `chunk.data` (CRC + subchunk split via `append_clean_narrowed` / `note_block_boundary`) |

The flip fires once (`MarkerDecodeCtx::flipped`): the ring already holds the 32 KiB
window, so the clean tail resolves back-refs **with no window copy and no second
decode engine**. `resumable_resync` (the old clean-tail path: a separate
`ResumableInflate2` wrapper seeded from a copied window) is **off the flip path** —
it survives only as the internal re-sync when the speculative seed is not a real
block boundary (`decode_chunk_with_rapidgzip_impl`’s `Err` arm). Remaining gap vs
rapidgzip is **implementation speed** + gzippy-specific **post-process**
(`apply_window` / `narrow_markers_in_place`) and **publish-chain** latency.

The retired `marker_inflate::Block` bootstrap engine is now `#[cfg(test)]`-gated
(test oracle only); **production workers call `decode_chunk` /
`decode_chunk_window_absent` → `decode_chunk_with_rapidgzip_impl`.**

## Worker dispatch (`chunk_fetcher::run_decode_task`)

```
pool.run_task → worker.decode_chunk
  ├─ chunk 0 or window_map.get(start_bit) exact  → decode_chunk (often skips stage 1)
  ├─ Design H handoff window (spec prefetch)     → decode_chunk @ handoff key (usually 0 hits)
  └─ else                                        → speculative_decode_find_boundary
       → decode_chunk_window_absent (phase 1 marker → phase 2 clean, same ring)
```

## After decode (still on critical path for wall)

```
post_process.task: resolve_chunk_markers (apply_window) + narrow_markers_in_place + CRC
consumer: wait for chunk → write output + publish 32 KiB tail window (serial chain)
```

`append_markered` / `absorb_isal_tail` were **removed from the hot unified path**; markers
are written via `MarkerSink` during phase 1, and the phase-2 clean tail narrows inline via
`append_clean_narrowed`. Narrowing of the marker prefix still runs in post-process
(`clean_unmarked_data` + `apply_window`) when markers are present.

## NOT production (do not optimize as “shipping”)

See [`production-paths.md`](production-paths.md) §5 — oracles, bypass, `GZIPPY_CLEAN_WINDOW_ORACLE`
(validate instrument before trust), experimental modules not on the trace tree.

## Regenerate the live tree

```bash
GZIPPY_TIMELINE=/tmp/t.json.gz GZIPPY_FORCE_PARALLEL_SM=1 \
  gzippy -d -c -p8 /path/to/silesia-large.gz > /dev/null
# then timeline_analyze.py --tree on the guest/host artifact
```
