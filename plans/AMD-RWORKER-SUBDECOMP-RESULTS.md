# R_WORKER SUB-DECOMPOSITION — fulcrum excess (exact numbers)

**Date:** 2026-06-22 · **Box:** solvency EPYC 7282 Zen2 (`ssh root@REDACTED_IP`), FROZEN
gov=performance boost=0, **llama REAPPEARED → SIGSTOP'd during capture, SIGCONT'd +
verified Running after** (902427 llama-server + 902407 bench_he.py). · **gz src:**
kernel-converge-A `39acc213` + sub-region instrument (region_prof.rs / marker_inflate.rs),
bin parallel-sm+pure, FFI off. · **rg:** region-patched + sub-region patch
(rg_subregion_prof_patch.py). · **Finding:** `F-9c5ca01d020d` (refines `F-f81bd0c136c6`).
· **Data:** `plans/amd-gap-data/amd_subdecomp_*`.

## THE ANSWER — where the R_WORKER excess sub-localizes (fulcrum excess, gated)

```
fulcrum excess  loss=silesia control=nasa arch=amd-zen2 ε=0.050   [NOT-YET-LAW]
   region          verdict        loss g/r  ctrl g/r   recoverable(cyc/byte)
   RING_OTHER      INTRINSIC        6.641    2.152       0.0000
   DECODE_TOTAL    INCONCLUSIVE     1.041    0.902       0.0000
   RECOVERABLE BUDGET = 0.0000 cyc/byte (0 EXCESS region @ ε=0.05)

ε-sensitivity (ε=0.020):
   DECODE_TOTAL    EXCESS           1.041    0.902       0.3916
   RING_OTHER      INTRINSIC        6.641    2.152       0.0000
```

### What this says (fulcrum is the oracle, not this prose)
- **DECODE_TOTAL** (ALL decode incl clean-CRC; table_build folded in — gz pure-Rust
  `read_header`+`Block::read`/`decode_clean_into_contig` vs rg deflate-bootstrap +
  **ISA-L `readStream`** clean tail): **near-parity** — gz/rg loss 1.041, control 0.902
  (gz FASTER on nasa). The silesia-specific recoverable component is **+0.3916 cyc/byte**,
  but it sits **below the standard ε=0.05 EXCESS threshold** (1.041 < 1.05) → fulcrum
  INCONCLUSIVE at ε=0.05, EXCESS only at ε≤0.04. ⇒ gz's pure-Rust decode kernel is
  ROUGHLY AT PARITY with rg's ISA-L decode; the silesia gap is small and corpus-dependent.
- **RING_OTHER** (gz fold-drain copy + chunk setup/finalize vs rg append/move): the
  LARGEST gz>rg gap (gz 0.786 vs rg 0.118 cyc/B on silesia) **but INTRINSIC** — gz is
  2.15× heavier on the nasa CONTROL too. ⇒ a gz ARCHITECTURAL overhead present on ALL
  corpora (the fold-drain copy rg avoids), **NOT a silesia-specific recoverable excess**.
- **table_build ACQUITTED**: gz-internal it is only **3.2% of R_WORKER** on silesia
  (0.349 cyc/B) / 0.9% on nasa — too small to carry the surplus, and per-header cost does
  not grow on silesia. Not the lever.

### Refinement of the prior finding (F-f81bd0c136c6)
The prior R_WORKER excess (loss 1.110 / control 0.925, reproduced here) **decomposes** to:
a **sub-ε decode-kernel gap** (gz pure-Rust vs rg ISA-L, +0.39 cyc/B silesia, gz faster on
nasa) **+ an INTRINSIC gz ring/fold-drain overhead** (gz heavier on all corpora). It does
**NOT** cleanly localize to a single large recoverable decode-kernel EXCESS. The decode
kernel is near-parity; the biggest raw gz>rg gap (ring) is intrinsic, not recoverable.

### gz-internal sub-split (DIAGNOSTIC; rg fuses tables into ISA-L so not a matched region)
```
gz silesia(loss)  R_TABLE=0.349  R_DECODE_body=9.711  ring=0.786  R_WORKER=10.857 cyc/B  (2843 hdr; table=3.2%)
gz nasa(ctrl)     R_TABLE=0.067  R_DECODE_body=6.979  ring=0.295  R_WORKER= 7.358 cyc/B  ( 373 hdr; table=0.9%)
```

## WHY rg's R_DECODE needed the ISA-L wrap (a near-mislabel caught)
rg is built **WITH_ISAL**: after 32 KiB of clean output it flips the clean tail to
`finishDecodeChunkWithInexactOffset<IsalInflateWrapper>` (and window-resolved chunks use
`decodeChunkWithInexactOffset`), both decoding via `inflateWrapper.readStream` — BYPASSING
the deflate::Block inner loop. The first sub-region patch (block-loop only) put rg's ENTIRE
clean decode (797M cyc, 43% of WORKER) into `ring_other`, which would have FALSELY shown
gz R_DECODE ≫ rg R_DECODE. Wrapping the two ISA-L decode functions dropped rg ring_other
to 1.4% and made DECODE_TOTAL matched. (cursor-agent Q6 risk #5 — residual absorbing real
decode on one arm — realized and fixed.)

## GATE STAMPS
- **Gate-0:** SUB_OVERLAP_VIOLATIONS=0 (gz & rg, all 18 reps); non-inert (table & decode
  >0 both); conservation EXACT (R_TABLE+R_DECODE+ring_other == R_WORKER, gz & rg);
  sha==zcat all reps both tools; /dev/null both arms (SINK LAW); A/A 0.71%. cursor-agent
  design review incorporated (fold clean_crc into decode; ALL table materialization
  clawed into table_build via TL nested-subtraction; clean/marker NOT split — gz flips
  mid-chunk while rg stays markered, so a split is not matched).
- **Gate-1:** N=9 interleaved (gz,rg,gzAA); spread ~4%; the loss-vs-control FLIP of
  DECODE_TOTAL (1.041 vs 0.902 = 14% rel) exceeds spread; RING control 2.15 ≫ 1+ε.
- **Gate-4:** path=ParallelSM; HEAD 39acc213; parallel-sm+pure.
- **Gate-3 OWED:** Intel replication → NOT-YET-LAW (Zen2 single-arch; fulcrum-stamped).
- **instr metric OWED:** per-region instructions need rdpmc in BOTH codebases (high-risk
  per-thread perf_event_open). Per cursor-agent Q7, frozen-box (boost=0) cyc/byte with
  IPC-parity (established by F-f81bd0c136c6: branch-miss + cache at parity) is the accepted
  instruction-surplus proxy. The whole-program +256M-instr surplus (1.055/0.865) is NOT
  yet attributed per-sub-region.

## NEXT (made clear by the numbers)
1. The decode kernel is NEAR-PARITY (gz pure-Rust ≈ rg ISA-L) — the silesia gap is small
   (+0.39 cyc/B, sub-ε) and gz is FASTER on nasa. A decode-kernel rewrite has a SMALL,
   corpus-dependent, sub-ε ceiling here — poor ROI as a silesia lever.
2. The LARGEST raw gz>rg gap is **RING_OTHER (gz fold-drain copy)** — but INTRINSIC
   (all corpora). If pursued, it is a uniform gz-vs-rg architectural deficit (eliminate
   the drain copy → rg-style direct append), NOT a silesia-specific recoverable win.
3. Intel replicate (Gate-3). 4. Per-region rdpmc for the instr metric (Gate-7 owed) if a
   sub-region needs instruction-level confirmation.
```
