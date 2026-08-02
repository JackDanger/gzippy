# What is left after #227: two classes, opposite mechanisms, and L1 is 43% of it

Source: the re-gate artifact itself (`/root/www/gzippy-bench/campaign/lever-regate-227/try.json`,
792 cells, 660 graded, n=15, **solvency = AMD Zen2 = the authority box**, TUNE
corpus, L1-9, T1+T4, 4 rivals). Not a log scrape — the adjudicator's own per-cell
records, with `base_failing` / `after_failing` / `after_ratio` per cell.

## The denominator, which we had lost

    FAILING on main (120bfa9c):  85
    FAILING after #227:          37
    net closed:                  48

**#227 removes 56% of main's failing SIZE cells on this corpus.** That is a
stronger statement than "48 cells" alone, because it is a fraction of a
*measured* denominator rather than a count against the stale 200/1320 board
(which is 24 commits old and covers a different, 22-file corpus).

Main's 85, by rival x threads: **libdeflate T04 61 (72%)**, libdeflate T01 8,
gzip 4+4, pigz 4+4.

## The residual 37 splits in TWO, and the halves need OPPOSITE mechanisms

| class | cells | worst | what it needs |
|---|---|---|---|
| **near-tie seam** | **15 of 37 are within 0.1% of a tie** | `libdeflate:movie.mp4:L07:T04 = 1.000005` (**5 ppm**) | a handful of BYTES per block — header/boundary encoding |
| **L1 coding deficit** | **16 of 37 (43%)** | `libdeflate:data.csv:L01:T01 = 1.0457` | a better L1 MATCHFINDER |

Median `after_ratio` = 1.002406, i.e. a median gap to a tie of **0.2406%**.

Residual by rival x threads: libdeflate T04 21, libdeflate T01 8, pigz T01 4.
Residual by level: **L01 16**, L04 1, L05 3, L06 5, L07 3, L08 5, L09 4.
Residual by corpus: aozora 7, minjs 7, movie.mp4 6, data.csv 4, data.json 4,
data.parquet 3, dickens 3, engine.wasm 3.

### Why the split matters more than the total

The seam cells are at **1.000005 - 1.000139** — five parts per million on
movie.mp4. `project_t4_seam_is_a_step_function` already establishes that this
class cannot be closed by "making the seam smaller" in general, and these ratios
show why: they are already essentially tied. They need a *bytes-per-block*
mechanism (block boundaries, header/Huffman encoding), not parse strength.

The L1 cells are 100-1000x further away (up to 4.6%) and are a different problem
entirely — a coding deficit, not a seam. **They are also the ones with a proven
mechanism already in the tree.**

## L1 is 43% of the residual, and this is the THIRD independent sighting

1. `055bd4b5` (#234): splitting the 22-file size census by level x rival found
   **L1 = 35 failing cells, 29 vs libdeflate**, the largest level.
2. Today's M1 wall census: the single worst wall cell is
   `libdeflate:data.parquet:L01:T01 = 1.3328`, and the only igzip losses are at
   L1 and L3.
3. **This artifact: L1 is 16 of 37 residual size cells (43%), and holds the four
   worst ratios.**

Different corpora, different axes, different boxes, different commits. The lever
is specified in `docs/board/l1-next-lever.md` — attempt 2's `ht_fast` synthesis
(2-way buckets AND the length-3 table, already written and `fulcrum verify`-clean
at 220 cells) gated T>1-only — and it is **blocked on #227 landing**, because the
gating needs `params_parallel()`, which exists only on that branch.

## Scope

- TUNE corpus (11 files), not the 22-file set behind the 200/1320 board. The two
  denominators are NOT interchangeable; do not substitute one for the other.
- SIZE axis only. `--size-only` was used deliberately: size is an exact integer
  byte count, arch-invariant and load-immune, so it runs on a busy box.
- 132 of 792 cells are ABSENT (igzip has no L4-L9). Correctly declared, not a gap.

## ✅ CLAUDE.md's "the needed margin is ~0.01%" CHECKS OUT — do not "correct" it

`CLAUDE.md:54` states *"The needed margin is ~0.01%"*. I nearly filed that as
superseded by today's measurement, and it would have been a conflation of two
DIFFERENT quantities. Recording the check so the next reader does not repeat it:

- **"Needed margin"** = how much smaller a lever must make our output to close a
  **SEAM** cell — a cell that already essentially ties libdeflate. For the 15
  residual cells within 0.1% of a tie, the gaps run **5 ppm (0.0005%) to 0.1%**,
  which brackets 0.01% exactly. The banked
  `project_t4_seam_is_a_step_function` independently says median 255 B ≈ 0.006%.
  **Consistent. The claim stands.**
- **My "median gap to a tie = 0.2406%"** is computed across ALL 37 residual
  cells, which MIXES the seam class with the L1 coding-deficit class (up to
  4.6%). It is a different population and answers a different question.

The two numbers are not in tension; they describe different subsets. A
"correction" replacing 0.01% with 0.24% would have made the doc wrong and would
have mis-sized every future seam lever by ~40x — the same order as the 40x
budget error that closed the parse-config space against the wrong coordinate.

**Rule this illustrates:** before retracting a banked constant, check that your
number measures the SAME POPULATION. Today's session has three receipts for the
opposite failure (a T16 profile labelled T1, a 3-file class generalised to 11, a
module total divided by an excess as if the vendor paid nothing) — all of them
right arithmetic on the wrong set.
