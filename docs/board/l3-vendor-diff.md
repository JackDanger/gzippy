# The wall board has THREE regimes, and only one of them is a config question

Status: PRE-REGISTERED PREDICTION. Written 2026-08-01 from a 169-row partial read
of the first-ever L1-9 wall board, BEFORE the full board (~792 rows) finished.
The point of writing it now is that it can be wrong when the data lands.

## The vendor diff (hard stop #1), read from the vendor source

`vendor/libdeflate/lib/deflate_compress.c:3920-3960`, `deflate_alloc_compressor`,
against `src/compress/deflate/level.rs:params_inner`:

| level | libdeflate | gzippy | same? |
|---|---|---|---|
| 1 | `fastest`, nice=32 | `Fast`, chainless | different parser (igzip-class) |
| 2 | `greedy`(6,10) | `Greedy`(6,10) | **identical** |
| 3 | `greedy`(12,14) | **`Lazy`(12,14)** | **ONE BIT: the parser** |
| 4 | `greedy`(16,30) | `Greedy`(16,30) | **identical** |
| 5 | `lazy`(16,30) | `Lazy`(16,30) | **identical** |
| 6 | `lazy`(35,65) | `Lazy`(35,65) | **identical** |
| 7 | `lazy`(100,130) | `Lazy`(100,130) | **identical** |

L3 is our only divergence in the greedy/lazy half of the ladder, and it is a
single variable: same `max_search_depth`, same `nice_match_length`, different
parser. That is the cleanest diff this project has ever had to work with.

## What the board says, at the same coordinates

libdeflate @ T1 wall_ratio (>1.000 = we LOSE), 3 files fully graded:

| file | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---|---|---|---|---|---|---|---|---|
| dickens | 1.0453 | 1.0231 | **1.2393** | 0.9821 | 1.0175 | -- | 0.9096 | 0.8770 | 0.8776 |
| aozora.txt | 1.0585 | 1.0404 | **1.1948** | 0.9711 | 1.0148 | 0.9059 | 0.8887 | 0.8384 | 0.8447 |
| data.json | 0.9222 | 1.0455 | **1.1422** | 1.0298 | 1.0466 | 0.9550 | 0.9483 | 0.8788 | 0.8764 |

## THE DECOMPOSITION — this is the finding

The naive read is "L3 is slow because we run Lazy where they run Greedy." That
explains the L3 spike and NOTHING ELSE, and it is contradicted by its own
neighbours: at L2, L4 and L5 our configuration is IDENTICAL to libdeflate's —
same parser, same two knobs — and we still lose by 1.5-4.7%. Identical
algorithm, identical parameters, slower. No config change can explain that.

And it inverts. At L6-L9, configuration STILL identical, we WIN by 4.5-16%.

So there are three regimes, not one deficit:

  R1  L1        different parser entirely (chainless igzip-class). Mixed:
                loses on 2 text files, wins on data.json. Own regime, not
                analysed here.
  R2  L2,L4,L5  IDENTICAL config, we lose 1.5-4.7%.   depth = 6, 16, 16
      L6-L9     IDENTICAL config, we WIN 4.5-16%.     depth = 35, 65, 100, 150
  R3  L3        the ONE config divergence, we lose 14-24%.

R2 is the whole story and it is not about parser choice at all. Sort R2 by
`max_search_depth` and the sign of the deficit is monotone in depth: we lose at
6/16/16 and win at 35+. The crossover sits between depth 16 (L5, we lose) and
depth 35 (L6, we win).

## THE MECHANISM, and it makes a checkable prediction

A per-position cost decomposes as `fixed + depth * per_node`. If our `fixed` is
LARGER than libdeflate's and our `per_node` is SMALLER, then the ratio
`ours/theirs` falls monotonically as depth rises and crosses 1.0 exactly once.
That is the observed shape, and it is the only two-parameter shape that
produces a sign inversion with identical parameters.

It also matches CLAUDE.md's own standing sentence: *"Our level->config map is
currently a copy of libdeflate's, which is why we run their algorithm slower
than they do."* — true at shallow depth, FALSE at deep depth, and the board is
the first artifact able to say which.

PREDICTIONS, registered before the full board lands (~792 rows, 11 files):

  P1  Across all 11 files, libdeflate@T1 wall_ratio at L2,L4,L5,L6,L7,L8,L9 is
      monotone DECREASING in `max_search_depth`, with exactly one crossing of
      1.0, between depth 16 and depth 35.
  P2  L3 sits ABOVE that trend line on every file — the Lazy penalty is a
      constant added to a depth curve, not a separate phenomenon.
  P3  L4 (depth 16) and L5 (depth 16) have the SAME depth and different
      parsers, so their ratios differ by the Greedy-vs-Lazy penalty alone.
      Partial data: dickens 0.9821 vs 1.0175, aozora 0.9711 vs 1.0148,
      data.json 1.0298 vs 1.0466 — a Lazy penalty of +1.7%, +4.5%, +1.7%.
      Predict the L3 excess over its depth-12 trend point is the same order,
      i.e. the 14-24% L3 spike is NOT explained by Lazy alone and something
      else is wrong at L3 specifically.

P3 is the one I expect to fail, and it is the most useful either way. If the
Lazy penalty really is ~2-5% at depth 16, then Lazy cannot account for a 14-24%
spike at depth 12, and the L3 cell has a second cause that no ladder retuning
will reach.

## WHY THIS MATTERS FOR THE SIZE BOARD TOO

`level.rs:126-128` records, from the T1 size census: we are an EXACT BYTE TIE
with libdeflate on 154 of 198 cells, and *"L3 is the ONLY deep level where we
diverge and it is the only one where we hold a margin: smaller on 20/22 files,
median 44 KB."*

Same coordinate, opposite sign. L3 is simultaneously the only level with size
headroom and the worst level on the wall. `project_t4_seam_is_a_step_function`
says 109 failing T4 size cells have ZERO headroom and need a median 255 B; L3
holds ~44 KB, 170x more than required, and is the only level that holds any.

So "revert L3 to the vendor's Greedy" would close the worst wall cells and
destroy the only size headroom on the board. It is a trade, not a fix, and
clause 3 (no pass->fail flips) is absolute. The question worth measuring is
whether a config exists that is BOTH cheaper than `Lazy(12,14)` and still
smaller than `Greedy(12,14)` — which is verbatim what `ladder-tune`
(`level.rs:118-139`) was built to find and has never been pointed at L3.

Adjacent precedent, already on disk (`level.rs:267-281`): at L4, `Lazy` with
`max_search_depth: 10` wins SIZE on 11 of 11 tune files vs libdeflate-4 with
ZERO cells opened, failing only clause 5 on wall — and `symbols.dwarf` is
strictly Pareto-dominant there (6,553 B smaller AND 4.8% faster at T4). Lazy at
REDUCED depth is a measured-productive direction one level away from L3.

## WHAT IS NOT CLAIMED

- Nothing here is a wall claim about the shipped binary beyond the 169 graded
  rows quoted. 3 of 11 files. The board is re-running.
- Our side of the vendor table is a SOURCE READ of `level.rs`, not an
  execution. `fulcrum explain` can assert it against `LEVEL_DECLARED`, but that
  needs an `anatomy-counters` build and the box is measuring.
- The `fixed + depth*per_node` decomposition is a MODEL that fits 9 points on 3
  files. It is not a counter. P1 tests its shape on 11 files; naming which
  component carries `fixed` needs a profile.
- Scoped to T1 and to libdeflate (hard stop #3). At T4 we lose ZERO wall cells
  to anyone, and against gzip/pigz/igzip we lose zero at either thread count.

## THE ONE THING TO DO NEXT

Nothing until the board finishes. Then check P1/P2/P3 against 11 files, and if
P3 fails as expected, profile L3 specifically rather than retuning the ladder.
