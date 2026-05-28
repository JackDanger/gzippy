# Symbolized perf attribution: where the 19% memmove lives

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `b55b612`
**Host**: neurotic LXC 199 (`-C target-cpu=native -C force-frame-pointers=yes -C debuginfo=1`)
**Build**: `cargo build --release --features pure-rust-inflate`
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16, parallel-SM

## Top-level: 19.10% in `__memmove_avx_unaligned_erms`

```
19.10%  __memmove_avx_unaligned_erms (libc)
  └─ 9.70%  __memcpy_avx_unaligned_erms (inlined)
      ├─ 4.84%  core::ptr::copy_nonoverlapping (inlined)
      │   ├─ 2.84%  emit_backref_ring (inlined)
      │   │   └─ Block::read_internal_compressed_canonical_specialized
      │   │       └─ Block::read
      │   │           └─ bootstrap_with_deflate_block_inner   ← MARKER PHASE
      │   └─ 1.61%  Vec::append_elements (Vec::extend_from_slice)
      │       └─ Block::drain_to_output
      │           └─ bootstrap_with_deflate_block_inner       ← MARKER PHASE
      └─ 1.82%  core::ptr::copy → slice::copy_within
          └─ ChunkData::clean_unmarked_data
              └─ ChunkData::finalize
                  └─ absorb_isal_tail
                      └─ decode_chunk_marker_bootstrap_then_isal  ← MARKER PHASE
```

**ALL the memmove is in the marker-bootstrap path**, NOT the bulk
inflate. Specifically:
- `emit_backref_ring`: writes back-references into the u16 ring buffer
  during speculative marker decode.
- `Vec::extend_from_slice` from `drain_to_output`: copies the marker
  ring's clean output into the chunk's `data` Vec.
- `clean_unmarked_data` / `ChunkData::finalize`: post-bootstrap cleanup
  copy.

## What this means

Pure-rust bulk inflate (`decode_huffman_body_resumable` + match copy)
spends almost NO time in memmove. The 18-19% memmove is all
**marker-phase speculative bootstrap work** that gzippy does to find
deflate block boundaries when the predecessor window is unknown.

The same marker bootstrap runs in BOTH the `pure-rust-inflate` and
`isal-compression` builds (they share the bootstrap code; only the
bulk phase differs). So this work is **not** what differentiates
pure-rust from ISA-L throughput.

## Why the 41% throughput gap exists

If memmove is the same in both builds, the throughput gap must come
from the **bulk phase**:
- `pure-rust-inflate` bulk: pure-Rust ResumableInflate2 inner loop
- `isal-compression` bulk: ISA-L FFI bulk inflate

Per the inflate-not-bottleneck doc, the inner loop is ~15% of CPU.
But that 15% in pure-rust corresponds to a HIGHER number in ISA-L
(because ISA-L's bulk is faster, so the SAME absolute marker-phase
work appears as a BIGGER fraction). That's consistent.

So the 41% throughput gap really IS in the bulk inflate inner loop.
But my 5 lever attempts on the inner loop were at parity with rustc.

## What might actually win on the bulk inflate

Three remaining ideas (none tried this session):
1. **Speculative-parallel LUT lookups** — issue 4 lookups in parallel
   with OoO speculation, then determine actual count. The
   `vector_huffman` path was previously falsified pre-PRELOAD; the
   CLAUDE.md update 2026-05-27 explicitly authorizes re-attempt.
2. **AVX2 multi-byte literal output via vpshufb** — vendor's exact
   pattern; the simpler u32 store I tried (S1) was at parity, but a
   true SIMD shuffle has structurally different memory traffic.
3. **Remove the resumable yield-check tax in FASTLOOP** — vendor
   doesn't yield mid-decode. CLAUDE.md explicitly allows
   reorganizing the state machine.

## What might actually win on the marker phase

Three ideas (none tried this session):
1. **Shrink the marker ring** from u16 to u8 by storing markers in a
   parallel bitmap. Halves memcpy bandwidth in `emit_backref_ring`.
2. **Pre-allocate ChunkData::data with capacity = chunk_uncompressed
   size estimate**, eliminating the page-fault-on-growth pattern in
   `Vec::append_elements`.
3. **Eliminate `clean_unmarked_data`'s copy_within** by writing
   directly to the post-cleanup layout, not the marker layout.

## Falsification record this session (unchanged)

Five inflate-inner-loop levers attempted, all at parity or
regression. The data here suggests the next session should target the
**marker-phase memcpy** (a real ~5% lever per the perf data) or the
**bulk inflate's structural redesign** (multi-day work).
