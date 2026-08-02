# L4 → `Lazy(12,30)`, gated to T>1: **13 cells closed, 0 opened** (size leg)

**Measured 2026-08-01.** Size only — arch-invariant, load-immune. **Not a wall claim; the
wall leg is unmeasured and is the only remaining gate.**

## The result

| coordinate | opened | closed |
|---|---|---|
| **T4** | **0** | **13** |
| T1 | **2** | 2 |

The two T1 opens are `ecoli.fastq` and `weights.safetensors` vs libdeflate — the same two
files where `Lazy` is genuinely worse than `Greedy`, and both are T1 byte-tie cells that
`Lazy` breaks. **So the lever must be GATED TO T>1.** With that gate, T1 is untouched by
construction (0 opened, 0 closed) and T4 is a clean 13-cell win.

## The 13, and why they are the seam class

```
access.log   vs libdeflate  2,857,647 -> 2,837,428   (rival 2,857,413; was +234)
winexe.exe   vs libdeflate  1,551,015 -> 1,521,358   (rival 1,551,013; was +2 !!)
photo.jpg    vs libdeflate  6,473,541 -> 6,472,160   (rival 6,473,476; was +65)
movie.mp4    vs libdeflate 12,891,866 ->12,890,711   (rival 12,891,391; was +475)
dd79_bin6    vs libdeflate  4,501,135 -> 4,461,540   (rival 4,500,874; was +261)
monorepo.tar vs libdeflate 10,338,189 ->10,195,614   (rival 10,337,816; was +373)
data.parquet vs libdeflate 14,162,357 ->14,070,896   (rival 14,160,264; was +2,093)
tool.bin     vs libdeflate 21,400,715 ->21,043,134   (rival 21,400,590; was +125)
minjs.min.js vs libdeflate  1,129,759 -> 1,105,973   (rival 1,129,252; was +507)
data.csv     vs libdeflate  3,646,184 -> 3,445,400   (rival 3,645,905; was +279)
data.json    vs libdeflate  1,736,308 -> 1,644,394   (rival 1,736,197; was +111)
data.sqlite  vs gzip       14,804,146 ->12,372,131   (rival 14,784,601)
data.sqlite  vs pigz       14,804,146 ->12,372,131   (rival 14,779,904)
```

**`winexe.exe` currently loses by 2 BYTES. `data.json` by 111. `photo.jpg` by 65.** These
are the zero-headroom byte-tie cells: at T1 we tie libdeflate exactly, and the T>1 seam
(~100-500 B) pushes us over. Ten of the thirteen have a pre-change excess under 600 B.

This is **exactly** the mechanism CLAUDE.md prescribes for the seam class — *"closed by
monotone T1 size wins that buy headroom to spend"* — demonstrated on 13 cells. It is the
same mechanism the L3 arm already relies on (see
`parity-at-T1-is-a-liability-at-T4.md`), applied at the level that currently has none.

## Why this is a different proposition from the parked L4 lever

`level.rs:267` parks `Lazy(10,30)` and closes the family on clause 5 at T4. Three things
have changed:

1. **The candidate is wrong in the record.** `Lazy(10,30)` still sags on 5 of 22 files;
   `Lazy(12,30)` is 0/22 and is 268,884 B smaller. See
   `the-parked-L4-candidate-does-not-fix-the-ladder.md`.
2. **The lever was never scored as a CELL lever.** The park record justifies it on total
   size ("772,154 B") and on the ladder. **Nobody counted cells.** It closes 13.
3. **It should be T>1-GATED**, which the park record does not consider. Gating removes
   the 2 T1 opens entirely and puts the wall question at T4, where the recorded slack is
   249-330% instead of T1's 0-8%.

## What is still required, and it is the only thing

**A `fulcrum ab paired` wall run at T4 on solvency, with aa_bias reported.** Clause 5 is
an erosion budget and this lever spends wall by construction (lazy does roughly twice the
matchfinder work of greedy; depth 12 vs 16 partly offsets it). **The size leg above does
not and cannot answer that.**

It also needs `params_parallel()` — per-thread-count level params — which exists **only on
#227's branch**. That is a second, independent reason #227 is the gating item.

## Coordinates and limits

T1 and T4, level 4, all 22 corpus files, gzip / pigz / libdeflate (**igzip skipped: its
real ladder is L0-L3, so it has no L4 cell**). `-p4` with pigz matched at `-p4`.
Candidate produced with the `ladder-tune` Cargo feature (a build feature, not a shipped
env knob), A/A-verified 22/22 identical against the default build at default params.
Size only. **Nothing here is routed and `level.rs` is not modified.**
