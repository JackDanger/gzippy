# S1 — u32 packed multi-literal store: FALSIFIED at parity

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `dd86cc3`
**Host**: neurotic LXC 199 (`-C target-cpu=native`)
**Build**: `cargo build --release --features pure-rust-inflate`
**Fixture**: `benchmark_data/silesia-gzip9.gz` (212 MB output)
**Routing**: parallel-SM, T=16, libdeflate-LUT inner loop

## Result

10-trial A/B, internal `parallel_sm:v0.6` timer:

| Decoder      | T1   | T2   | T3   | T4   | T5   | T6   | T7   | T8   | T9   | T10  | Median |
|--------------|------|------|------|------|------|------|------|------|------|------|--------|
| Default      | 629  | 635  | 657  | 650  | 597  | 637  | 704  | 625  | 634  | 607  | 634.5  |
| `GZIPPY_PACKED_LIT_STORE=1` | 668 | 705 | 706 | 636 | 638 | 641 | 611 | 571 | 618 | 623 | 637    |

Units: MB/s.

**Delta: +2.5 MB/s = +0.4%. Within noise.**

## Why falsified

The hypothesis: 4 separate 1-byte literal stores → 4 store μops → 4
cycles. 1 unaligned u32 store → 1 store μop → 1 cycle. Savings = 3
cycles per 4-literal batch.

Reality: LLVM + x86 store-buffer coalescing already merges contiguous
adjacent stores. The 4 1-byte stores execute as effectively a single
cache line write. The "savings" the u32 store would deliver are
already realized by the existing code.

Additional cost paid by the packed path:
- Extra `out_pos + 4 <= output_len` bounds check
- More complex `carry: Option<LitLenEntry>` tracking via Option (LLVM
  represents as 2-word struct with tag — adds branch)
- Loss of locality: the existing code interleaves lookup + store +
  lookup, hiding lookup latency. The packed path serializes 4
  lookups then issues 1 store — store-buffer can't help.

Net: parity. The hypothesis was wrong about what's costly.

## What would actually help

Vendor `libdeflate` packs 8 literals into a u64 store. But that's NOT
the lever either — the same store-buffer coalescing applies. The
vendor's real win is **speculative parallel lookups**: issue 4 LUT
lookups concurrently (each at offset `i*12` bits into bitbuf), let
the OoO core execute them in parallel, then determine the actual
literal count post-hoc. That hides lookup latency, not store latency.

That's the next-tier lever. Multiple lookups in parallel requires:
- Pre-shifting bitbuf by speculative consume amounts (depends on each
  lookup's actual code_bits, which is the very thing we're looking up)
- Speculative restart if any lookup hits a non-literal

This is what `vector_huffman::decode_multi_literals` attempted at
commit `ca52389` — and falsified at -15%. The CLAUDE.md update
2026-05-27 explicitly says to re-attempt with the post-PRELOAD shape;
that re-attempt has not yet landed.

## Next move

Either:
A) **Re-attempt the speculative-parallel-lookup** in the new
   post-PRELOAD shape (CLAUDE.md authorized; previously falsified).
B) **SSE2 overlap-safe copy_match_windowed slow path** — the
   per-byte slow path (window touch) is ~4.14% absolute CPU per
   attribution. SSE2 vectorized copy with `lz_overlapping_load`
   pattern could halve that.
C) **Status quo + measurement plan rebuild** — the attribution doc
   (884 MB/s pure-rust) doesn't match today's measurement (~634
   MB/s on the same fixture). Something has changed; we need fresh
   profile data before picking the next lever blindly.

Recommend C first, then B (lower implementation risk than A's
speculative-parallel).
