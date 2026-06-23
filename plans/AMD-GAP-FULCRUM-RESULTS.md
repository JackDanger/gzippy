# AMD/Zen2 gz-vs-rapidgzip GAP — FULCRUM DECOMPOSITION (exact numbers)

**Date:** 2026-06-22 · **Box:** solvency EPYC 7282 Zen2 (`ssh root@REDACTED_IP`),
FROZEN gov=performance boost=0, llama SIGSTOP'd (restored). · **gz src:** kernel-converge-A
HEAD `39acc213` · **gz bin sha** `1b7473f8ff8d` (`parallel-sm+pure`, gzippy-native, FFI off)
· **rg bin sha** `b407b3d56849` (region-patched). · **Finding:** `F-f81bd0c136c6`
(`.fulcrum/findings.jsonl`). · **Data:** `plans/amd-gap-data/`.

The deliverable = **fulcrum's** output, not prose. All analysis routed through the
existing fulcrum analyzer + the existing region-prof capture (no parallel analyzer built).

## THE ANSWER — where the AMD gap goes (fulcrum excess, gated)

```
fulcrum excess  loss=silesia control=nasa arch=amd-zen2 ε=0.050   [NOT-YET-LAW]
   region        verdict        loss g/r  ctrl g/r   recoverable(cyc/byte)
   R_WORKER      EXCESS           1.119    0.923       1.1564
   R_MARKERPP    INCONCLUSIVE     0.699    0.690       0.0000
   R_OUTPUT      INCONCLUSIVE     0.272    0.259       0.0000
   RECOVERABLE BUDGET = 1.1564 cyc/byte (1 EXCESS region)
```

**R_WORKER (per-chunk decode call: huffman decode + commit + clean-CRC + ring) is the
WHOLE gap.** gz pays **+1.156 cyc/byte EXCESS** decoding silesia: gz 10.888 vs rg 9.731
cyc/byte (loss ratio 1.119), yet on the nasa CONTROL gz is FASTER (7.247 vs 7.848, ratio
0.923). The excess is silesia-specific ⇒ recoverable, not intrinsic.

- **R_MARKERPP (marker resolve / applyWindow):** gz CHEAPER on BOTH corpora (ratio ~0.69)
  → INCONCLUSIVE (not a gz-recoverable gap). **Reconfirms the marker-kernel acquittal.**
- **R_OUTPUT (iovec+crc-combine+writev vs writeAll):** gz CHEAPER on both (~0.27). The
  prior "gz OUTPUT 68× heavier" was a **file-sink artifact** (gz wrote 211 MB to /tmp; rg
  to /dev/null). With matched /dev/null sinks gz R_OUTPUT = 0.8M vs rg 2.9M cyc. KILLED.

**Conservation (silesia, T4):** total gz−rg gap = +0.177e9 cyc = R_WORKER +0.245e9 cyc
OFFSET by R_MARKERPP −0.060e9 cyc (gz's marker advantage), R_OUTPUT ≈ 0, R_OTHER ≈ 0.

## WHAT HW RESOURCE R_WORKER BURNS — instruction-bound, NOT memory-bound

perf mem record -k CLOCK_MONOTONIC unsupported on this kernel → used the full-count
2-event instruction totals (from the excess capture, non-multiplexed) + a -dd profile
(62% multiplexed, weaker). R_WORKER = 94% of gz cycles, so whole-decode ≈ R_WORKER.

| metric (silesia T4)      | gz       | rg       | gz/rg  |
|--------------------------|----------|----------|--------|
| instructions (full-count)| 5.034e9  | 4.770e9  | **1.055** |
| instructions (nasa ctrl) | 3.70e9   | 4.285e9  | **0.865** |
| branches (-dd)           | 958.8M   | 777.0M   | 1.23   |
| branch-miss RATE         | 3.66%    | 3.90%    | lower  |
| cache-misses             | 5.49M    | 5.52M    | ~1.00  |
| L1-dcache-miss RATE      | 3.42%    | 3.56%    | ~1.00  |

The instruction ratio **FLIPS** (1.055 silesia / 0.865 nasa): the R_WORKER cycle excess is
an **INSTRUCTION SURPLUS** (gz emits ~256M more instr to decode silesia), branch-heavy,
with cache/L1 behaviour at PARITY. ⇒ the lever is **fewer instructions in the clean
decode / table-build path on silesia-like (many-table / high-entropy) data**, NOT a
memory/cache fix. (HW tier = whole-program-attribution / WEAK except the instr ratio
which is full-count.)

## SCALING — gap is per-thread WORK, not parallel-deficit (fulcrum REFUSED; un-blessed)

`fulcrum scaling` REFUSED the verdict (closure/conservation failed: Σbuckets−wall
0.5–4.5 ms; its anti-bias gate firing — NOT banked). Visible un-blessed signal: gz
self-speedup ≥ rg at every T (T2 1.05× vs 1.05×, T4 1.84× vs 1.83×, T8 2.86× vs 2.43×) ⇒
gz "gives up" NEGATIVE ms of scaling. Corroborates "gap is per-thread R_WORKER work,"
but stays HYPOTHESIS (fulcrum refused). rg walls silesia: T1 437.2 / T2 416.7 / T4 238.6
/ T8 179.9 ms.

## GATE STAMPS (self-validating instrument)

- **Gate-0:** OVERLAP_VIOLATIONS=0 (gz & rg); region non-inert (all calls>0); sha==zcat
  all reps; matched per-region byte denominators (MARKERPP gz==rg byte-identical
  73124965/184311599; WORKER & OUTPUT = total-out both arms); cursor-agent-reviewed design
  (caught the R_WORKER denominator + flagged the symbol-universe check, both resolved).
  **SINK LAW fix:** first run wrote gz→file (A/A 27% off + phantom R_OUTPUT); corrected to
  /dev/null both arms → A/A 0.04%.
- **Gate-1:** N=9 interleaved; R_WORKER silesia cyc/B spread 4.1%; loss-vs-control flip
  (1.119 vs 0.923 = 21% relative) ≫ spread. Significant.
- **Gate-3:** Intel replication OWED — NOT-YET-LAW (Zen2 single-arch; fulcrum stamped it).
- **Gate-4:** path=ParallelSM verified; HEAD 39acc213; flavor parallel-sm+pure.

## NEXT STEPS (made clear by the exact numbers)

1. **The only recoverable region is R_WORKER, and it is instruction-bound on silesia.**
   Decompose R_WORKER below the call: split clean-decode vs huffman-table-build vs
   clean-CRC vs ring-commit instructions, on silesia vs nasa, to find WHICH sub-step
   carries the +256M-instruction silesia surplus. (Extend region-prof with sub-spans;
   feed fulcrum excess again.)
2. Replicate on Intel (Gate-3) to make it LAW.
3. Do NOT pursue R_MARKERPP / R_OUTPUT / a marker-asm (acquitted / cheaper / artifact),
   and do NOT pursue a parallel-pipeline fix (scaling shows gz ≥ rg per-T).
