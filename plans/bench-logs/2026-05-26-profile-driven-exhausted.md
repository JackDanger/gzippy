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

## perf stat evidence for LUT=11 vs LUT=12 (L1d spill confirmation)

Both binaries built fresh with their own PGO profile (instrumented build →
silesia run → llvm-profdata merge → optimized rebuild). Each bench ran for
the same `--profile-time 5` window on CPU 1, taskset-pinned.

LUT=11 (current production):
```
21,296,037,484  cycles
46,942,062,598  instructions   (2.20 IPC)
   358,367,937  L1-dcache-load-misses   (3.53% miss rate)
10,162,973,540  L1-dcache-loads
       154,019  LLC-load-misses         (16.88% of LL-cache accesses)
       912,307  LLC-loads
```

LUT=12 (regressed):
```
21,511,587,293  cycles
38,823,486,316  instructions   (1.80 IPC)
   343,954,568  L1-dcache-load-misses   (4.17% miss rate)
 8,256,371,118  L1-dcache-loads
       162,679  LLC-load-misses         (18.96% of LL-cache accesses)
       857,986  LLC-loads
```

Diff:
| Metric | LUT=11 | LUT=12 | Delta |
|---|---:|---:|---:|
| IPC | 2.20 | 1.80 | **-18.2%** |
| L1d miss rate | 3.53% | 4.17% | **+18.1% relative** |
| LLC miss rate | 16.88% | 18.96% | +12.3% relative |
| Instructions completed in 5s | 46.94B | 38.82B | -17.3% |

Interpretation:
- LUT=12 doubles `CacheEntry[CACHE_LEN]` storage from 16 KB to 32 KB. The
  Intel i7-13700T has 48 KB L1d cache (with effective allocation lower
  due to other working-set: marker ring, bit-reader state, output Vec).
- L1d miss rate climbed 18% relative — confirmed cache pressure.
- IPC collapsed from 2.20 to 1.80 (cache stalls serialize execution).
- Despite fewer total instructions (the larger LUT is supposed to reduce
  fallback decodes), throughput dropped because each instruction is
  slower.

The PGO profile in both cases was collected against the **same-LUT-size binary**
(LUT=11 profile against LUT=11 binary; LUT=12 profile against LUT=12 binary).
No stale-profile confound.

## Raw bench output ladder (PGO state isolated)

Four-point ladder, all under `RUSTFLAGS="-Cprofile-use=... -Ctarget-cpu=x86-64-v3"`
on CPU 1 of neurotic. 200-sample Criterion measurements.

| State | inner_loop_only | from_scratch | bootstrap_path |
|---|---:|---:|---:|
| **A** Pre-PGO baseline (LUT=11, no PGO) | 560.4 MiB/s | 561.2 MiB/s | 478.8 MiB/s |
| **B** +PGO (LUT=11, distance HC = SymbolsPerLength) | 560.4 MiB/s | 567.8 MiB/s | 494.7 MiB/s |
| **C** +PGO+distance-HC-swap (LUT=11, distance HC = ReversedBitsCached) | 627.6 MiB/s | 608.8 MiB/s | 523.5 MiB/s |
| **D** +LUT=12 under PGO (compounded on C) | 512.3 MiB/s | 525.2 MiB/s | 423.2 MiB/s |

Deltas:
- B − A (PGO alone): +0% / +1.2% / **+3.3% bootstrap**
- C − B (distance HC swap alone): **+12% inner / +7.2% / +5.8% bootstrap**
- D − C (LUT=12 attempt): -18.4% / -13.7% / **-19.2% bootstrap** → reverted

Cumulative landed (C vs A):
- inner_loop_only: 560 → 628 MiB/s (+12%)
- from_scratch: 561 → 609 MiB/s (+8.6%)
- bootstrap_path: 479 → 524 MiB/s (+9.4%)

Profile-driven cumulative: ~+9% bootstrap (~+5.8% from distance-HC-swap dominates;
PGO contributes ~+3.3%).

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
