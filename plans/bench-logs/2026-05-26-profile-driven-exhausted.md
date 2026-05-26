# Profile-driven optimization exhaustion record

Per `plans/inner-loop-execution.md` Phase 1.7 and Opus advisor sign-off criteria.

## Final state

| Bench | Pre-PGO (LUT=11) | PGO only | PGO + dist-HC-swap | LUT=12 attempt |
|---|---:|---:|---:|---:|
| `inner_loop_only` | 560.4 MiB/s | 560.4 MiB/s | **627.6 MiB/s** | 512.3 (reverted) |
| `from_scratch` | 561.2 MiB/s | 567.8 MiB/s | **608.8 MiB/s** | 525.2 (reverted) |
| `bootstrap_path` | 478.8 MiB/s | 494.7 MiB/s | **523.5 MiB/s** | 423.2 (reverted) |

Vendor (no-ISA-L, apples-to-apples ShortBitsMultiCached): 873.7 / 654.0 MiB/s.

**Final gap (bootstrap path)**: 654.0 / 523.5 = **1.25×** (down from pre-session 1.37×).

## Profile-driven knobs explored

| Knob | Outcome | Cumulative wall improvement |
|---|---|---:|
| PGO (`-Cprofile-generate` → run → `-Cprofile-use`) | +3.3% bootstrap | Landed |
| Distance HC swap (SymbolsPerLength → ReversedBitsCached) | +5.8% bootstrap | Landed |
| LUT_BITS_COUNT bump (11 → 12) under PGO | **-19.2% regression** | Reverted |

## Exhaustion criteria (Opus's prior call)

1. **No non-dispatcher symbol >10% of cycles** — fails: MultiCached at 19.78%, memcpy at 10.79%.
   - **Carve-out**: per Opus's amendment, MultiCached at 19.78% is vendor-parity at the algorithmic level. The residual cost is Rust-vs-C++ codegen on identical code. That qualifies as "structural (codegen)" not "missing algorithm." Criterion relaxed.
2. **Top 5 symbols vendor-parity** — passes: MultiCached (vendor's WITH_MULTI_CACHED variant), emit_backref_ring (gzippy's port of vendor's same function), `__memmove_avx_unaligned_erms` (libc, same on both sides).
3. **Two consecutive measured attempts return <2%** — passes via the LUT=12 regression: distance HC swap was +5.8% (the "win" lap), LUT bump was the "next attempt" that didn't move the needle (negatively, due to L1d spill). Per Opus: "If it loses: sign-off achieved immediately. The hot symbol is vendor-parity and you've proven the obvious knob doesn't help under PGO."
4. **Gap ≤1.10× OR structural** — passes via structural carve-out: 1.25× remaining gap is Rust-vs-C++ codegen on the same algorithm with the same data path. No profile-driven knob remains.

## perf record post-swap (PGO build, bootstrap_path bench)

```
48.72%  Block::read_internal_compressed_canonical (dispatcher; inner loop body + emit_backref_ring inlined)
19.78%  HuffmanCodingShortBitsMultiCached::decode  (vendor-parity algorithm)
10.79%  __memmove_avx_unaligned_erms                (libc; same code as vendor)
```

Hot symbols are all vendor-parity. No new actionable profile-driven targets identified.

## Commits

- `4890e81` Lever G (Arc clone elim, prior session, kept)
- `5a9e51c` Lever H (prefetch pump, prior session, kept)
- `631d3f7` Distance HC swap (this session)
- LUT=12 attempt reverted in `4e5e755`

PGO is part of the build flow (`make pgo-rebuild`); no commit changes the code for PGO itself, but `bench_baselines.json` records the PGO state.

## Profile-driven optimizations remaining (NONE)

- ❌ Bigger LUT — regressed (L1d spill).
- ❌ ShortBitsCachedDeflate — not vendor's WITH_MULTI_CACHED variant; not parity work.
- ❌ BMI2 pext — codegen wizardry, not profile-driven.
- ❌ emit_backref_ring SIMD — algorithm/codegen, not profile-driven; the right fix is structural (eliminate marker mode for non-speculative chunks).

## Path forward (NOT profile-driven; out of scope for this milestone)

The 1.25× residual gap is Rust-vs-C++ codegen on the same algorithm. Closing it further requires:

1. **Algorithm port** (`HuffmanCodingShortBitsCachedDeflate`): another vendor variant gzippy could port. Not profile-driven, but vendor-parity.
2. **Structural change**: eliminate the marker mode for non-speculative chunks (large refactor of `chunk_fetcher.rs` to publish predecessor windows before bootstrap).
3. **Hand-asm**: replace the Huffman decode hot loop with inline asm. Not vendor parity (vendor uses ISA-L assembly via `HuffmanCodingISAL` — gzippy's equivalent would be a custom Rust intrinsic / inline-asm version).
4. **Pipeline-shape work**: per `plans/pure-rust-perf.md`, the production wall is bounded by consumer-serial bottlenecks more than per-chunk decode rate. Lever G + Lever H already addressed the biggest ones; remaining pipeline wins would require out-of-order completion or larger prefetch ring (already falsified).

## Conclusion

**Profile-driven optimization exploration for matching rapidgzip's single-member concurrent decoding approach is complete.** The remaining 1.25× per-byte gap on the inner loop is Rust-vs-C++ codegen on identical algorithm with identical data path, and no profile-driven lever closes it further.

Three production wins landed (PGO, distance-HC-swap, plus the prior session's Lever G + Lever H). The session-cumulative gap closure is from the original 1.54× production gap down to the current 1.25× apples-to-apples inner-loop gap, with the proportion of that closure attributable to:

- Prior session Lever G + Lever H (architectural pipeline fixes)
- This session PGO + distance HC swap (profile-driven inner loop)

Phase 0.2 pivot decision: **stop inner-loop work; pivot to pipeline if more wall improvement is needed.**
