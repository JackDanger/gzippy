# Marker Decoder Performance Roadmap

Produced by Opus analysis (2026-05-15). Recommendations in estimated impact order.

## A — O(1) clean-window scan (biggest win, smallest patch)

**Problem**: At every block boundary, bootstrap walks 32,768 u16s with
`.all(|v| v < MARKER_BASE)`. On L6 deflate (~50 blocks per chunk before
handoff) that's 50 × 32 KB = ~6 MB of u16 memory traffic per worker.
A second redundant 32 KB scan happens in `decode_chunk_bootstrap` when
converting the clean tail to `Vec<u8>` for ISA-L dict.

**Fix**: Track `clean_count: usize` incrementally:
- In `emit_match` marker-emission branch: `clean_count = 0` when `marker_count > 0`.
- Literal pushes: `clean_count += 1`.
- Chunk-local memcpy (non-overlapping): if source range lies entirely within trailing
  `clean_count` bytes → `clean_count += local_count`. Otherwise scan only the copied
  range (≤ 258 bytes) and update.
- Block boundary check: `if clean_count >= WINDOW_SIZE && output.len() >= WINDOW_SIZE`.

Also remove the second scan in `decode_chunk_bootstrap` (the assert-and-cast walk).
Once `clean_count` is the invariant source, the assert is dead code.

rapidgzip equivalent: `m_distanceToLastMarkerByte` counter in `deflate::Block`.

**Impact**: Removes O(N_blocks × 32 KB) u16 traffic. ~4 ms saved per worker on a
24 MB L6 Silesia chunk at T=4.

## B — Reuse `ConsumeFirstTable` instead of fresh 128 KB single-level tables

**Problem**: Every dynamic block calls `HuffTable::build` three times. Each build
allocates `vec![0u32; 32768]` (128 KB zero-fill, ~10 µs), then writes entries.
~1.25 ms per chunk for table construction alone (50 blocks × 25 µs). The 128 KB
working set pollutes L2/L3 cache.

**Fix**: `src/decompress/inflate/consume_first_table.rs::ConsumeFirstTable` is the
production multi-level table (~4 KB primary, subtable arena). Refactor
`fast_marker_inflate::HuffTable` to use `ConsumeFirstTable` (or delete it and call
`ConsumeFirstTable` directly). The marker decoder needs identical primitive —
"decode one symbol, consume code-length bits."

Bonus: `CachedTablePair` at line 526 of `consume_first_table.rs` caches litlen +
distance pairs to avoid rebuilding on identical dynamic block headers — plug in.

**Impact**: ~9× smaller primary table stays L1-resident. Table build ~2-3 µs/block
(vs ~25 µs). Decode throughput +20-40% from cache residency alone.

## C — Multi-symbol Huffman decode (rapidgzip's multi-cached trick)

**Problem**: `decode_huffman_block` does one `litlen.decode` per output byte.
For ASCII text (7-8 bit codes) that's one full peek+lookup+consume per byte.

**Fix**: Widen primary table entries to carry a "stride" field (how many literals this
entry produces, packed into the data field). Hot loop becomes
`for (; count > 0; count--, sym >>= 8)` — 2-4 bytes per Huffman lookup on
ASCII-heavy data.

Alternative (simpler): template-parameterise `decode_block_consume_first` from
`consume_first_decode.rs` on u8 vs u16+markers. One Huffman inner loop, generic emit.

**Impact**: 1.5-2× on ASCII/log/FASTQ. Smaller gains on binary data (bootstraps fast).

## D — Pre-build fixed-Huffman tables with ConsumeFirstTable format

Already cached via `OnceLock<HuffTable>`. Switching to `ConsumeFirstTable` shrinks
statics from 128 KB to ~4 KB, improves icache density. Free if B is done first.

## E — Encapsulate the bit-position formula (correctness, not perf)

`bits.pos*8 - bits.available()` appears at four sites. One was a double-count bug
(PR #90); another in `decode_stored` (rewind `pos` by buffered bytes before zeroing).
Add `bits.consumed_bits(base_byte) -> usize` method to eliminate the class.

rapidgzip equivalent: `bitReader->tell()`.

## F — Skip second u16→u8 copy for clean window

When `clean_window = Some(...)`, code allocates 32 KB `Vec<u8>` and walks trailing
32 KB of `output` byte-by-byte. Use a reused thread-local 32 KB scratch or expose
a packed-u8 view of the u16 slice. No allocation, same ISA-L dict input.

## G — RLE fast path for distance==1

byte-by-byte loop in `emit_match`'s overlap branch: 258 sequential read-write cycles
for a max-length RLE match. Add:
```rust
if distance == 1 && out_pos > 0 && local_count > 0 {
    let fill = output[out_pos + marker_count - 1];
    output.resize(out_pos + marker_count + local_count, fill);
}
```

## H — Thread-local arena for `lens: Vec<u8>` in `decode_dynamic`

Allocates `vec![0u8; total]` once per dynamic block (~316 bytes, 50×4 workers = 200
small allocs). Replace with 320-byte `ArrayVec` or reused `Vec` cleared between blocks.

---

## Recommended sequence

1. **A** first — ≤50 lines, highest impact, verifiable against existing fuzz/oracle tests.
2. **B** second — aligns marker decoder with production u8 decoder's table format.
3. **C** after A+B confirm wins on `make ship`.
4. **E, G, H** as cleanup bundles.
5. **F** if profiling shows u16→u8 walk still visible after A/B/C.

Acceptance: > 50 MB/s/thread; A+B+C together should double that against L6 text,
putting per-thread parity with sequential ISA-L within reach.
