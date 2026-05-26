# Parallel-SM architectural floor

Per Opus's pre-registered abandon criterion, declaring the architectural floor for
gzippy's pure-Rust parallel-single-member decode after exhaustive optimization.

## Final state

| | best | p50 | mean |
|---|---:|---:|---:|
| gzippy (--features isal-compression) | 356 ms | 382 ms | 392 ms |
| rapidgzip (vendor, ISA-L on) | 124 ms | 148 ms | 145 ms |
| **Ratio** | **2.87×** | **2.58×** | **2.70×** |

Improvement this session: gzippy 440 → 356 ms best wall (**-19%**).
Cumulative across recent sessions: 666 → 356 ms (**-47%** since pre-Lever-G/H baseline).

## Confirmed wins (landed)

- **append_markered O(n) classification scan removal** (`2edf63b`): -5.6% best, -9.2% p50.
- **Switch to `--features isal-compression`** (build flag): bootstrap uses ISA-L LUT (~-15% wall vs pure-rust-inflate).
- Prior session: Lever G (Arc deep-clone elim) + Lever H (rx.recv pump) + distance-HC swap + Kraft check + subchunk gate.

## Falsified levers (this session)

| Lever | Predicted | Actual | Verdict |
|---|---|---|---|
| Bootstrap exit via `Block::contains_marker_bytes()` flip | -15-25% wall | 0% (trace -11% worker time but wall unchanged) | Reverted — slow chunks (outliers) never trigger the flip; per-chunk avg drops but max doesn't |
| Disable speculation (`GZIPPY_NO_PREFETCH=1`) | possible 15-25% if right | **+42% regression** | Speculation is net positive even with bootstrap cost |

## Top-outlier analysis

Per-chunk worker decode times from trace:

```
Total decode_chunk events: 40 (across 9 workers)
Distribution: min=4.0 p25=23.8 p50=46.4 p75=65.0 p95=76.4 max=86.6 ms
Top 5 cumulative: 377 ms (22% of all worker time in 5 chunks)
Top 10: all 65-87 ms, 9/10 speculative=True
```

The wall is bounded by max worker event time on any thread. Vendor's max is likely ~30 ms; gzippy's is 87 ms.

## Why the floor is here

Three structural costs compound:

1. **Rust-vs-C++ codegen on identical algorithm**: ~1.25× per-byte gap (measured apples-to-apples on the same `HuffmanCodingShortBitsMultiCached<11>` algorithm — see `2026-05-25-phase-1.4-apples-to-apples.md`). Closing this requires hand-asm or inline-asm in the inner loop, which gzippy explicitly avoids (pure-Rust goal).

2. **Speculative-chunk bootstrap overhead**: workers decode ahead of consumer demand to fill the parallel pipeline. Speculative chunks have no predecessor window → must bootstrap with markers throughout (no 32 KB no-marker run materializes → `Block::contains_marker_bytes()` never flips on these chunks → bootstrap runs the entire chunk).
   - Vendor pays this same cost but the per-byte rate is faster (point #1), so vendor's max worker event is smaller.
   - Eliminating speculation regresses 42% (workers can't fill the pipeline) — net loss.

3. **Strict-FIFO consumer write order**: consumer can't drain chunks out of order. Wall = max worker event on the critical-path chunk. Vendor uses strict-FIFO too. Out-of-order writes would deviate from vendor architecture significantly.

## Remaining theoretical levers (NOT pursuing)

- **Hand-asm inner loop**: gzippy explicitly avoids inline-asm; mismatched with the pure-Rust identity.
- **Inline-asm bit reader / Huffman decode**: same.
- **Out-of-order chunk completion + reorder buffer**: deviates from vendor architecture; not justified by data (Opus's prior analysis: vendor doesn't do this either; vendor wins via faster per-chunk).
- **Native ISA-L port (replace the C library with pure Rust)**: gzippy's ISA-L wrapper is NOT the bottleneck (per-symbol decode is at parity with vendor). Native port would be massive engineering effort for unclear gain.

## Hot symbols at the floor (perf record evidence)

From the most recent perf record on the optimized build:

```
48.72% Block::read_internal_compressed_canonical (dispatcher; inner loop body + emit_backref_ring inlined)
19.78% HuffmanCodingShortBitsMultiCached::decode (vendor-parity algorithm)
10.79% __memmove_avx_unaligned_erms (libc; same on both sides)
13.84% emit_backref_ring (back-ref copy + marker scan; vendor has same)
```

All hot symbols are vendor-parity. No actionable profile-driven lever remains.

## Conclusion

**Practical floor for parallel-SM pure-Rust gzip on the current architecture is ~2.6-2.9× behind vendor.**

Closing further requires:
- (a) leaving pure-Rust (hand-asm / inline-asm), OR
- (b) deviating from vendor's architecture (out-of-order writes, eliminating speculation despite measured regression)

Neither is in-scope. Team should pivot energy to other subsystems if more wall improvement is needed elsewhere.
