<!-- 2026-09-03: SUPERSEDED by docs/plan-2026-09-one-encoder.md. Kept as a
     measured receipt; its commit and its encoder topology predate the ldx
     pivot (#357-#360, 2026-08-23). Do not plan from these numbers without
     re-censusing the current stack (board-size.sh / parity-census.sh). -->

# The board, re-measured on main: 200/1320 CONFIRMED — and one cell is 63% of it

Full 22-file size census, solvency (AMD Zen2), main `120bfa9c`, vanilla binary
`2848c640`, `dirty=0`, L1-9, T1+T4, four rivals, `CAMPAIGN_PROMOTE=1` stamped
into the artifact. Instrument verified at `origin/main` (`8364a05`).
Artifact: `/root/www/gzippy-bench/campaign/size-all-120bfa9c/census.json`.

    declared 1584   measured_ok 1320   absent 264   VOID 0   bigger 200

**Identical total to the board at `12fcd0ed`. Twenty-four commits moved the
count by ZERO.** The standing 200/1320 was real and current the whole time.

⚠ I spent part of this session warning that the figure was stale and might have
moved in either direction. It had not. The warning was still correct to make —
`12e93fff` removed an ungated `PROBE_CPT` env read from the shipped T>1 chunk
grid, which provably changes T4 output — but the outcome was no drift, and I had
cast doubt on a good number. **"Possibly stale" is a reason to re-measure, not a
reason to disbelieve.**

## By rival x threads — the concentration is extreme

| rival x threads | failing |
|---|---|
| **libdeflate T04** | **125 — 63% of the whole board** |
| libdeflate T01 | 17 |
| gzip T01 / T04 | 16 / 16 |
| pigz T01 / T04 | 10 / 10 |
| igzip T01 / T04 | 3 / 3 |

Totals by rival: libdeflate 142, gzip 32, pigz 20, igzip 6.

## By level

    L1=35  L2=23  L3=12  L4=17  L5=25  L6=25  L7=26  L8=17  L9=20

**L1 is the largest single level**, reproducing `055bd4b5` (#234)'s count of 35
on a DIFFERENT commit and a re-run census. That is an independent confirmation,
not a citation.

## The two largest classes are 160 of 200, and both have work standing ready

1. **`libdeflate @ T4` — 125 cells.** PR #227 is a re-gated SHIP: 48 cells
   closed, 0 opened, no pass->fail flips across 660 decidable cells, graded at
   all nine levels. **Every closure is at T4.** Mergeable and clean.
   On the TUNE corpus the same run measured main at 85 failing -> 37 after,
   i.e. it removes 56% of them there.
2. **L1 — 35 cells.** Gap is ALGORITHMIC, uniquely among levels (`fulcrum why`:
   POSITION COUNTS DIFFER, 2.89x more literals than libdeflate). Mechanism named
   (single-probe vs 2-way bucket). Port proven EXACT (`ht_fast` minus hash3
   matches libdeflate's size on 8/8 files). Four-rival ratchet built, 0.19 s per
   iteration. **Blocked on #227**, because the routing must be gated T>1 and
   `params_parallel()` exists only on that branch.

## What is NOT claimed

- This is the SIZE axis. The wall axis is counted separately and its only
  complete census is on the OUTLIER arch (local M1) — see
  `docs/board/wall-census-complete.md` and [[project_wall_is_arch_dependent]].
- 264 ABSENT cells are igzip at levels it does not implement. Correctly declared.
- `VOID 0` — every declared cell that could be measured, was. Unlike the wall
  census, which VOIDed 122 of 792.
