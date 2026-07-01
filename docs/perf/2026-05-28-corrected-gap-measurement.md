# CORRECTED ground-truth gap measurement

**Date**: 2026-05-28 (end of session)
**Branch**: `reimplement-isa-l` @ `750a6d3`
**Host**: neurotic LXC 199 (`-C target-cpu=native`, stripped release builds)
**Fixture**: `benchmark_data/silesia-gzip9.gz`, T=16, parallel-SM
**Methodology**: 10-trial interleaved A/B with clean release builds
(no debug info, no frame pointers, default release profile).

## Result

| Trial | Pure-rust | ISA-L FFI |
|-------|-----------|-----------|
| 1     | 1088      | 1208      |
| 2     | 1114      | 1216      |
| 3     | 1122      | 1196      |
| 4     | 1009      | 1227      |
| 5     | 1125      | 1210      |
| 6     | 1049      | 1160      |
| 7     | 1096      | 1248      |
| 8     | 1156      | 1221      |
| 9     | 826       | 1120      |
| 10    | 1037      | 1215      |
| **Median** | **1092** | **1212.5** |
| **Mean**   | **1062** | **1202**   |

Units: MB/s.

**Gap (median): 1212.5 - 1092 = 120 MB/s = 10% slower.**
**Gap (mean):   1202   - 1062 = 140 MB/s = 12% slower.**

Goal: within 1pp ≈ within 12 MB/s of 1212.5 → need pure-rust ≥ 1200 MB/s.

## What happened earlier this session

Earlier measurements showed pure-rust at **670 MB/s** — a 41% gap. That
measurement was **CONTAMINATED**. The session's debug-symbol experiment
(adding `debug = "line-tables-only"` and `strip = false` to release
profile in Cargo.toml to enable symbolized perf) was not cleanly
reverted before the subsequent A/B benches. The 670 MB/s figure was
measured against a release-with-debug-info binary that's materially
slower than true release.

When I restored Cargo.toml to its original state (`strip = true` for
release, default debug=0) and re-built, the ACTUAL gap is 10-12%, not
41%.

## Impact on this session's narrative

The 5 falsifications of inflate-inner-loop levers (Route C, S1, S2, L1,
ISA-L LUT inner) were all measured against the SAME contaminated
baseline OR the same-build comparator. Their "at parity" verdict is
still valid within their measurement set, but the absolute numbers are
wrong.

In particular: the session-level conclusion that "the 41% gap is in
the allocator, not inflate" is **wrong**. The real 10-12% gap may very
well be in the inflate inner loop after all, and the 5 falsifications
mean we haven't yet found the right inflate lever.

## What this means for the next session

- **The goal is much closer than the session thought** — 10-12% gap,
  not 41%.
- **Re-measure everything** with clean release builds before
  concluding anything.
- The falsified levers (Route C asm at parity, S1 u32 store at parity,
  S2 bulk window copy at parity) are still falsified — but for a tighter
  measurement space. They each save 0-1% absolute; insufficient to
  close 10%.
- The remaining structural levers (speculative-parallel LUT lookups,
  AVX2 vpshufb, FASTLOOP yield-check elision, u8 marker ring) are now
  more credible candidates — any one of them landing 10%+ would close
  the goal.
- The "memmove is in marker bootstrap" finding stands — the
  symbolization was correct, but the priority shifts. With only 10-12%
  to close, even a 5% marker-phase win is meaningful.
