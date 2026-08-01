# The deficit is 8.5 ms of FIXED per-pass cost, and our chain walk is 17% CHEAPER

Measured 2026-08-01, local M1 Pro, `fulcrum ab paired --mode compress`, n=9,
`/dev/null` both arms, `sign=9/9` and `aa_bias <= 0.0084` on every leg.
Our binary: vanilla sha `54079f4306cc6723`. Their binary: `vendor/libdeflate`
built from `22f6e023` plus the measurement patch below, verified to produce
byte-identical output to the `/opt/homebrew/bin/libdeflate-gzip` that every
board number in this campaign was measured against (both 4,582,861 B at `-5`
on dickens).

## The question this answers

At L5 and at L9, `fulcrum why` reports POSITION COUNTS MATCH at Δ0.00% — same
tokens, same matches, same literals, same total bits, byte-identical output.
Same algorithm, same parameters. And yet we are 1.5% SLOWER at L5 and 12.3%
FASTER at L9. Identical work, sign inversion. That is a fact about our
implementation and nothing else, and no ladder retuning can touch it.

## Why every earlier fit of this was confounded

libdeflate exposes no way to move `max_search_depth` independently of
`nice_match_length`: EVERY adjacent pair of its presets moves both at once
(L2->L4 is 6,10 -> 16,30; L5->L7 is 16,30 -> 100,130). A fit across its levels
therefore cannot separate the two. I did exactly that fit, and its out-of-sample
prediction for dickens L6 missed by 6 percentage points (predicted ratio 0.983,
observed 0.920) — because L6 moves nice 30->65, and their arm is more sensitive
to nice than ours (+16.0% vs +8.6%).

The fix was to give the vendor the knob our own `ladder-tune` feature already
gives us: `docs/instruments/libdeflate-probe-knobs.patch` adds
`LIBDEFLATE_PROBE_DEPTH` / `LIBDEFLATE_PROBE_NICE` to
`deflate_alloc_compressor_ex`. Measurement instrument only, in a vendor tree,
never shipped and never quoted as a product.

Discriminator run before trusting it — change the input, see whether the number
moves: `-5` output went 4,682,381 / 4,582,861 / 4,512,264 / 4,502,728 B at
PROBE_DEPTH 6 / 16 / 100 / 600, and unset reproduced the stock 4,582,861 B.

## The two clean curves

Both sides swept over DEPTH ALONE at fixed `nice=30`, lazy strategy, dickens
(12.17 MB), level 5. The output-size column is the control: it is IDENTICAL on
both sides at every depth, so the two programs are provably doing the same work.

| depth | ours ms | libdeflate ms | ours/theirs | output bytes (both sides) |
|---|---|---|---|---|
| 6 | 106.8 | 101.6 | 1.0516 | 4,682,381 |
| 12 | 125.0 | 122.6 | 1.0200 | 4,606,080 |
| 16 | 133.0 | 133.3 | **0.9981** | 4,582,861 |
| 35 | 162.5 | 177.2 | 0.9173 | 4,539,639 |
| 100 | 206.6 | 227.2 | 0.9095 | 4,512,264 |
| 300 | 240.5 | 268.1 | 0.8971 | 4,503,543 |
| 600 | 246.1 | 276.2 | 0.8913 | 4,502,728 |

## The fit, and the number

Over the shallow regime d=6..16, `T = F + depth * P`:

|  | fixed `F` | per-node `P` |
|---|---|---|
| ours | 91.1 ms | 2.621 ms/depth-unit |
| libdeflate | 82.5 ms | 3.171 ms/depth-unit |

    dF = +8.5 ms   we pay 10.3% MORE fixed cost per pass
    dP = -0.550    our chain walk is 17.3% CHEAPER per node

**8.5 ms over 12.17 MB = 0.70 ns/byte, about 2.2 cycles/byte, that libdeflate
does not pay.** It is per-position work that is not chain walking: hashing,
hash-table insertion, chain-head update, the outer loop, literal emission.

## The second consequence, checked (CLAUDE.md: one measurement, one claim)

The model was fitted on d=6 and d=16 only. It then predicts the crossover — the
depth at which `d * dP` finally pays off `dF` — at

    d = dF / dP = 8.5 / 0.550 = 15.5

The crossover was NOT used in the fit. Observed: ratio 1.0200 at d=12 and
0.9981 at d=16, i.e. the curve crosses 1.0 between 12 and 16. The prediction
lands inside that interval.

## Why this is the whole failing class

From the wall board (169 graded rows, 3 files): every losing wall cell is
`libdeflate @ T1`, and every one is at L1-L5. Their depths:

    L2 d=6    L3 d=12    L4 d=16    L5 d=16

**Every failing cell lives at depth <= 16 — inside the linear regime, on the
wrong side of a crossover at 15.5.** L6-L9 (d=35,100,300,600) are all wins, all
on the right side. That is not a coincidence to be explained; it is the same
single number seen twice.

## What is NOT claimed

- **Scope: local M1 Pro, dickens only, lazy only, nice=30 only, T1 only.**
  Not re-measured on solvency (busy) or the trainer box. Hard stop #3: nothing
  here generalises to another file or another arch until measured there.
- The linear model **breaks above d=16** — chains saturate, and at d=600 it
  would predict 1,669 ms against 246 observed. It is a shallow-regime model and
  is used only in the shallow regime, which is where the failing cells are.
- `F` is a FITTED INTERCEPT, not a counter. It says how much per-pass cost is
  depth-independent; it does NOT say which component carries it. Naming that
  needs callgrind or hw counters, and `fulcrum why` reported both layers SKIPPED
  on macOS — layer [2] needs valgrind (trainer box), layer [3] needs Linux
  counters. **That is the next measurement, and it should not be guessed at
  from source.**
- L3's ratio is 1.2393 against their GREEDY, not against their lazy; this
  sweep compares lazy to lazy. The L3 cell mixes the fixed-cost gap with a
  greedy-vs-lazy difference and a nice=14-vs-30 difference, and has not been
  decomposed. See the retraction in `docs/board/l3-vendor-diff.md`.

## What this kills

Any lever aimed at making our chain walk faster. It is already 17.3% cheaper
per node than the vendor's, measured at matched depth on byte-identical output.
Per-node work is not where the failing cells are.

---

## ⚠ CORRECTION (same day) — "our chain walk is 17.3% CHEAPER per node" is a
## MILLISECOND statement and is FALSE in instructions

Callgrind on trainer (Intel), dickens, L9 `lazy2(600,MAX)`, byte-identical
output:

    ours   hc.rs:lazy::run_resumable                1,866,549,864 Ir
    theirs hc_matchfinder.h:deflate_compress_lazy2  1,480,691,091 Ir   ours +26.1%

We execute 26% MORE instructions in the matchfinder and win that wall cell by
12.3%. The `P` term above is real as a TIME-per-depth-unit; it is not an
instruction count, and it must never be quoted as one. We issue more and stall
less — consistent with `project_encoder_deficit_is_loads_not_stalls`.

This is the campaign's standing rule ("instruction counts LOCATE, they never
predict the wall") applied to this file's own conclusion.

The `F` term survives and is CORROBORATED: see
`docs/board/l2-instruction-attribution.md`, where `block_split.rs` (33.0M Ir)
and libc `memcpy` (18.3M Ir) are FLAT in absolute instructions from L2 (depth 6)
to L9 (depth 600) while the total triples — a depth-independent per-pass cost
found by a second instrument, on a second box, on a second architecture.
