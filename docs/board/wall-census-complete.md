# The first complete L1-9 wall census — 51 LOSS cells, all at T1, and 122 VOID

Completed 2026-08-01. `scripts/campaign/board-wall.sh tune`, subject frozen
read-only by the W4 guard (sha `54079f43`, commit `ce5506a8`, `dirty=0`),
11 TUNE files x L1-9 x T1,T4 x 4 rivals, n=9, `/dev/null` both arms.
Artifact: `~/www/gzippy-bench/campaign/wall-tune-ce5506a8/`.

**Before this, the wall board had only ever been run at L6 — one level of nine.**

## ⚠ READ THIS FIRST: this is the LOCAL M1, which is the OUTLIER ARCHITECTURE

Measured the same day: at L6/T1 vs libdeflate, on byte-identical output,
**AMD Zen2 1.0810 LOSS, Intel 1.0395 LOSS, Apple M1 0.9059 WIN**. Both x86
boxes lose where this box wins. See `docs/board/fixed-vs-pernode.md` and
[[project_wall_is_arch_dependent]].

So this census is **one leg of a clause-7 trio, not a verdict.** The authority
box is solvency. Everything below describes a machine the goal is not scored on,
and the level distribution in particular is known to differ on x86.

## The totals

    declared 792   measured_ok 538   ABSENT 132   VOID 122   slower 51

| wall_class | cells |
|---|---|
| WIN | 468 |
| **LOSS** | **51** |
| TIE | 19 |
| VOID | 37 |

## The class: T1 only, and libdeflate is 92% of it

| rival x threads | non-WIN of graded |
|---|---|
| **libdeflate T01** | **59 of 94** |
| **igzip T01** | **9 of 31** |
| gzip T01 | 2 of 95 |
| gzip T04 | 0 of 71 |
| libdeflate T04 | **0 of 76** |
| pigz T01 | 0 of 93 |
| pigz T04 | 0 of 78 |

By the tool's own slower-counts: `libdeflate_slower=47 igzip_slower=4
gzip_slower=0 pigz_slower=0`.

**Zero losses at T4 against every rival.** The wall class on this box is
entirely T1 — which is the coordinate where the single-threaded rivals are at
their strongest and our slack is thinnest.

⚠ This CORRECTS an earlier claim of mine that every wall loss was
`libdeflate@T1`. igzip has 9 non-WIN cells (4 LOSS). That claim came from three
text files before the binaries had landed.

## By level and by file — the worst cells are SHALLOW and INCOMPRESSIBLE

    L01:10  L02:9  L03:12  L04:5  L05:8  L06:3  L07:2  L08:1  L09:1

    movie.mp4 11 | data.parquet 7 | armexe.elf 6 | tool.bin 4 | symbols.dwarf 4
    aozora.txt 4 | minjs.min.js 4 | data.json 4 | dickens 3 | data.csv 2 | engine.wasm 2

**31 of 51 losses are at L1-L3.** The worst individual cells are
`igzip:movie.mp4:L03:T01 = 1.6044`, `libdeflate:movie.mp4:L01:T01 = 1.4086`,
`libdeflate:data.parquet:L01:T01 = 1.3328` — near-incompressible data at shallow
levels, where there is little match work to amortise our per-position cost.

That is consistent with the L2/T1 component map: the parse loop carries 81.8% of
our instruction excess and the matchfinder gives 12.1% back, so a file with few
matches pays our overhead without collecting the matchfinder's advantage.

## 122 VOID is 15% of the board, and it is NOT evenly spread

    VOID by threads: T4 105, T1 17      <- 6.2x heavier at T4
    VOID by rival:   igzip 35, gzip 32, libdeflate 28, pigz 27

**The "zero losses at T4" result must be read against 105 VOID T4 cells.** The
honest statement is *zero losses among the T4 cells that RESOLVED*. A VOID is an
unmeasured cell, not a passing one, and the two kinds differ:
`pin_ok=false` (the rival never averages two cores — a real property, correctly
refused) versus `pin_ok=true` (the paired test could not separate the arms, which
happens near ratio 1.0 — exactly where a marginal loss would live).

**Never quote the T4 pass rate without the VOID count beside it.**

## What to do with this

1. It does NOT replace `.claude/board-state.json` (that is the SIZE axis, 22
   files, a different denominator).
2. The same census must run on **solvency** before any wall conclusion is drawn.
   This one exists to prove the harness works end-to-end at all levels and to
   give the aarch64 leg of clause 7.
3. Re-run the 122 VOID cells at higher `--n`. If a paired test still cannot
   resolve at n=21, record the cell as a genuine TIE rather than dropping it
   from the denominator.
