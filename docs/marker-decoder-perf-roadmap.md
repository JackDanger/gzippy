# Marker Decoder Performance Roadmap

> **Historical (May 2026):** Targets `fast_marker_inflate`, which is no longer in the
> tree. Production SM prefetch/on-demand decode uses `decode_chunk_isal_inexact`.

Produced by Opus analysis (2026-05-15), then checked against the worktree and
`vendor/rapidgzip` on disk.

## Reality Check

gzippy's `fast_marker_inflate` is not the older "decode the whole chunk as
markers" path anymore. `decode_chunk_bootstrap` stops at the first block boundary
whose trailing 32 KiB is marker-free, then `single_member` hands the rest of the
chunk to ISA-L/zlib-ng via `inflatePrime`. That makes pure-Rust marker decode a
bootstrap cost, not the whole per-chunk cost.

rapidgzip has the same shape, but with two important differences:

- `GzipChunk.hpp` uses `cleanDataCount >= MAX_WINDOW_SIZE` to hand off to ISA-L,
  and `deflate::Block` maintains `m_distanceToLastMarkerByte` while decoding.
- `DecodedData` stores marked and clean data separately. Once enough clean data
  exists, it appends `uint8_t` buffers directly instead of keeping every byte in
  a `uint16_t` vector.

The old priority order overstated clean-window scan cost. The scan still exists
in gzippy, but only during bootstrap and only until handoff. The hotter remaining
gap is table construction and decode-table shape inside that bootstrap.

## A — Replace `fast_marker_inflate::HuffTable`

**Problem**: Every dynamic block builds three local single-level tables. The
lit/len and distance tables can grow to 32,768 `u32` entries each, so a bootstrap
that crosses several dynamic blocks burns time on allocation, zero-fill, and a
large lookup footprint.

**Reality**: gzippy already has `ConsumeFirstTable` plus `CachedTablePair`, but it
is not wired into `fast_marker_inflate`. rapidgzip's default non-ISA-L path uses
`HuffmanCodingShortBitsCached`, not the larger multi-cached decoder; the
multi-cached variant exists but is compile-gated. With ISA-L enabled, rapidgzip
uses `HuffmanCodingISAL` for lit/len decoding.

**Fix**: First add a marker-specific table wrapper that reuses the compact
consume-first table layout without touching production u8 decode. Keep the local
`HuffTable` API (`decode(&mut Bits) -> symbol`) until benchmarks prove the table
swap is a win. Then consider the existing cache only if repeated dynamic headers
show up in profiles; a global `Mutex<HashMap<...>>` can easily cost more than it
saves in parallel workers.

## B — Track Clean Tail Incrementally

**Problem**: `decode_loop` checks
`output[output.len() - WINDOW_SIZE..].iter().all(|&v| v < MARKER_BASE)` at every
bootstrap block boundary. `decode_chunk_bootstrap` then walks the same 32 KiB
again to assert and cast the clean window to `Vec<u8>`.

**Reality**: This is not `N_blocks_per_chunk * 32 KiB`; it is only
`N_blocks_until_handoff * 32 KiB`. Still, rapidgzip avoids this by carrying
clean-byte state (`m_distanceToLastMarkerByte` / `cleanDataCount`).

**Fix**: Add a `clean_tail: usize` to the bootstrap decode state:

- Literal push: increment.
- Marker emission: reset to zero.
- Non-overlapping local copy: increment if the copied source range is entirely
  within the trailing clean tail, otherwise scan just the copied range.
- Overlapping copy: preserve correctness first; update incrementally only for
  the trivial clean source case, otherwise scan the emitted match.

Keep the release assertion for the ISA-L dict handoff, but feed the `Vec<u8>`
conversion from the same known-clean suffix so there is one checked walk, not two.

## C — Split Bootstrap Storage Like rapidgzip

**Problem**: gzippy's bootstrap output is a single `Vec<u16>` until handoff. Once
the suffix is known clean, the handoff window is copied to `Vec<u8>`, and the
remaining marker prefix still goes through later marker replacement and `u16_to_u8`.

**Reality**: rapidgzip keeps `dataWithMarkers` and `data` as separate buffers.
That lets clean bytes stay byte-wide and keeps marker replacement scoped to the
prefix that can actually contain markers.

**Fix**: Do not refactor the whole pipeline yet. Add an internal
`BootstrapResult { marked_prefix, clean_window }` shape only after A and B are
measured. This is a larger memory-layout change and should be driven by profiles
showing `u16_to_u8`, marker replacement, or allocation pressure still visible.

## D — Encapsulate Bit Positions

`bits.pos * 8 - bits.available()` still appears in multiple places and the file
documents prior double-count bugs around that formula. Add one helper on `Bits`
or a small local helper in `fast_marker_inflate` before doing more retry or
handoff work. rapidgzip's equivalent is `bitReader.tell()`.

## E — Small Local Cleanups

These are worthwhile only after the table and clean-tail work is measured:

- Prebuild fixed tables in the same compact format if A lands.
- Add a distance-1 RLE fast path in `emit_match`; rapidgzip's unmarked path uses
  `memset`, but the marker path still has to preserve marker propagation.
- Reuse storage for `lens: Vec<u8>` in `decode_dynamic` if allocation shows up in
  profiling.
- Reuse a 32 KiB dict scratch if clean-window allocation remains visible.

## Recommended Sequence

1. **A first**: table shape is the largest mismatch with rapidgzip's actual hot
   decoder and is easy to benchmark in isolation.
2. **B second**: small, correctness-sensitive, and aligned with rapidgzip's clean
   tail tracking, but no longer the biggest expected win.
3. **D with B or immediately after**: it reduces risk before more retry/handoff
   changes.
4. **C only if profiles still show marker-prefix conversion cost**.
5. **E as measured follow-ups**, not as one bundle.

Acceptance: keep the existing routing and correctness tests passing, then compare
`HANDOFF_FIRED`, `SLOW_PATH_USED`, per-chunk bootstrap bytes, and end-to-end
single-member Tmax throughput before and after each step. Do not rely on a marker
microbenchmark alone; the production win is handoff speed plus retry behavior.
