# T2 MARKER-FREE — RESULTS

Branch: kernel-converge-A (worktree gzippy-amd-t2t4). Base HEAD 8d0d24bd →
instrument 05de3e81. Primary metric: PEAK RSS (load-immune). Box: solvency
AMD EPYC 7282 Zen2 root@REDACTED_IP (x86_64; matches the x86 gated baseline arch).
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
