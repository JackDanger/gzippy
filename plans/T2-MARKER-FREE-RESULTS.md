# T2 MARKER-FREE — RESULTS

Branch: kernel-converge-A (worktree gzippy-amd-t2t4). Base HEAD 8d0d24bd →
instrument 05de3e81. Primary metric: PEAK RSS (load-immune). Box: solvency
AMD EPYC 7282 Zen2 root@10.0.2.240 (x86_64; matches the x86 gated baseline arch).
Build: `--no-default-features --features pure-rust-inflate` (`gzippy-native`),
RUSTFLAGS=`-C target-cpu=native`, in /dev/shm. Gate-4: GZIPPY_DEBUG=1 →
`path=ParallelSM threads=2`; sha256(gzippy -p2) == sha256(zcat) for monorepo.

## ISOLATION MEASUREMENT — **CONFIRMED**
Instrument GZIPPY_RSS_SPLIT=1 (byte-transparent; Gate-0 self-tests PASS:
non_inert=true, conservation sum==total_decoded=true). monorepo @ T2:

```
[rss_split] chunks=5 max_live_chunks=5 narrowed(u16 2x src)=31.00MiB \
            clean(u8 1x)=17.55MiB sum=48.56MiB total_decoded=48.56MiB
[rss_split] peak_resident_u16_markers≈2x_narrowed=62.01MiB \
            freeable_upper_half≈narrowed=31.00MiB | GATE0 non_inert=true conservation=true
```

Baseline peak RSS @ T2 (deterministic, 3 runs): 99072 / 99328 / 99072 KiB
= **96.75 MiB** (reproduces the inherited gated 96.5 MiB; spread 0.25 MiB).

### Verdict vs the pre-registered falsifier
- ALL 5 chunks are simultaneously live at peak (max_live_chunks == chunks == 5)
  — confirms the whole small file is resident at once (cache holds all chunks).
- 31.00 MiB of decoded payload flowed through the u16 `data_with_markers` 2×
  buffer; that buffer is held at 2× = ~62 MiB resident, whose **upper (wasted)
  half = 31.00 MiB**.
- gz-vs-rg gap = 96.75 − 59 = ~37.75 MiB. The u16 2× wasted half (31.00 MiB) =
  **82.1% of the gap (> 50%)** ⇒ **ISOLATION CONFIRMED**: the unfreed 2× u16
  marker buffer is the dominant source of the gz-vs-rg RSS gap.
- Reconciliation of peak: 2×31.00 (u16) + 17.55 (u8 clean) = 79.55 MiB decoded
  payload + ~17.2 MiB overhead (9.8 MiB input mmap + pools/LUTs) = 96.75 MiB. ✓
  Freeing the 31 MiB upper half would bring gz → ~65.8 MiB, near rg's 59 MiB.

### CAVEAT on realizability (HYPOTHESIS — to be measured by the fix)
Vendor `DecodedData::applyWindow` (DecodedData.hpp:374-379) deliberately does
NOT shrink the 2× buffer ("leaves half of the chunk space unused… shrink_to_fit
would be expensive"; suggests joining neighbor chunks as a @todo). So rg holds
the same 2× per chunk IN-FLIGHT; rg's lower RSS comes from holding FEWER chunks
simultaneously live (bounded prefetch/recycle) — whereas gz caches all 5 at once
(max_live==chunks). Therefore the realizable RSS win depends on whether the
upper half can be released BEFORE peak; peak may be decode-bound (all chunks hit
2× during decode before resolve). The fix below MEASURES this directly.

### Cross-arch isolation replication (Gate-3) — instrument validated both arches
macOS aarch64, silesia T2 (same SegmentedU16 data model, byte-exact sha==gzcat):
```
[rss_split] chunks=17 max_live_chunks=6 narrowed(u16 2x src)=66.86MiB \
            clean(u8 1x)=135.29MiB sum=202.15MiB total_decoded=202.15MiB | GATE0 PASS
```
Conservation + non-inert PASS on both arches. **CORPUS DEPENDENCE:** silesia is
only 33% markered (66.86 / 202 MiB) with 6/17 chunks live at peak, vs monorepo
64% markered (31.0 / 48.6) with ALL 5 live. So the u16 2× share of RSS is
corpus-dependent — dominant on monorepo, secondary on silesia. (CLAUDE.md
corpus-breadth caveat: do not over-generalize from one corpus.)

## FIX — MADV_DONTNEED post-narrow upper-half free (GZIPPY_FREE_MARKERS) → **FALSIFIED on throughput**

Mechanism: after narrow + CRC + subchunk-window extraction read the low (narrowed)
half, `release_narrowed_upper_pages()` MADV_DONTNEED's the page-aligned dead upper
half of each marker segment (faithful to rg applyWindow @todo DecodedData.hpp:374-379;
churn-free — no realloc/copy, the mission's preferred shrink/reuse). Gated behind
GZIPPY_FREE_MARKERS (OFF by default — production unchanged). Non-inert: freed=29.79
MiB, chunks_fired=4.

RSS (load-immune, solvency AMD x86_64, monorepo T2, 3 runs each):
| arm | peak RSS |
|-----|----------|
| OFF (baseline) | 99072/99328/99328 KiB = **96.9 MiB** |
| ON  (DONTNEED) | 76976/77232/77744 KiB = **75.4 MiB** |

→ **-21.5 MiB** toward rg's 59 (closes ~57% of the gap). Byte-exact: sha256==zcat.

THROUGHPUT (interleaved best-of-9, /dev/null sink both arms, frozen
gov=performance/boost=0, box restored after):
| workload | ON/OFF | verdict |
|----------|--------|---------|
| monorepo T2 | 1.0112 | +1.1% regress |
| silesia  T2 | 1.0285 | **+2.85% REGRESSION** (Δ13ms > intra-arm spread ~6ms) |
| silesia  T4 | 1.0009 | tie |
| silesia  T8 | 0.9969 | tie |

MADV_DONTNEED forces per-call TLB shootdowns on the post-process worker; at T2
(fewest workers) this is on the critical path. **MADV_FREE variant tested:** no
throughput cost BUT does NOT drop peak max-RSS (lazy reclaim isn't reflected in
getrusage max RSS — monorepo-T2 stayed 99 MiB) → fails the RSS goal. So DONTNEED
is required for the metric, and its T2 throughput cost is intrinsic.

### VERDICT vs pre-registered falsifier
- byte-exact ✓ ; RSS drops materially toward rg ✓ ; **NO throughput regression ✗
  (silesia-T2 +2.85%)** ⇒ **FIX FALSIFIED on the throughput gate.** Kept behind
  GZIPPY_FREE_MARKERS as a measurement tool (OFF by default); NOT shipped.

## NEXT FAITHFUL LEVER (identified by this measurement)
rg keeps the 2× buffer in-flight too (source @todo) — its lower RSS comes from
holding FEWER simultaneously-live chunks, not from freeing the 2×. gz holds the
WHOLE small file live (monorepo max_live==chunks==5). The rg-faithful convergence
is to BOUND the in-flight/cached chunk set (cache_capacity / prefetch depth —
RSS-CONVERGE candidate #2/#3), which drops peak RSS WITHOUT per-chunk madvise
throughput cost. That is the recommended next mission.

