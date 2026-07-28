# What each level SHOULD do — and how we check that it does

This is a falsifiable hypothesis, written before the measurement, about the behaviour a
DEFLATE encoder must exhibit at each level. `fulcrum explain` exists to confirm or refute it
against whatever binary is currently compiled.

The point is not to produce numbers to optimise. It is that a mismatch between a level's
DECLARED parameters and its OBSERVED behaviour is a defect — a knob that is set but not
doing anything, or doing something other than its name.

## The declared table (`src/compress/deflate/level.rs`)

| L | strategy | max_search_depth | nice_match_length |
|---|---|---|---|
| 0 | Fast0 | 0 | 32 |
| 1 | Fast | 1 | 32 |
| 2 | Greedy | 6 | 10 |
| 3 | LazyGated | 12 | 14 |
| 4 | Greedy | 16 | 30 |
| 5 | Lazy | 16 | 30 |
| 6 | Lazy | 35 | 65 |
| 7 | Lazy | 100 | 130 |
| 8, 9 | Lazy2 | (see table) | |
| 10 | NearOptimal | 35 | 75 |
| 11 | NearOptimal | 100 | 150 |
| 12 | NearOptimal | (see table) | |

## Predictions — each is checkable, each can fail

**P1 — Search depth must show up as chain-walk work.** `max_search_depth` bounds how many
hash-chain candidates a position may examine. So mean observed chain-walk length must rise
monotonically with it, and must never exceed it. L2 (6) → L4 (16) → L6 (35) → L7 (100)
should be visibly stepped. *Failure mode this catches:* a depth knob that is set but where
the loop exits early for an unrelated reason, so raising the level buys nothing.

**P2 — Nice length must show up as early termination.** `nice_match_length` is the length at
which the search stops looking for better. So the fraction of accepted matches whose length
is exactly >= nice_len should be substantial at low nice_len (L2 = 10) and shrink as nice_len
rises (L7 = 130). *Failure mode:* nice_len ignored, so high levels pay full search cost for
no ratio.

**P3 — Instructions per input byte must rise monotonically with level.** More search is more
work. Any inversion (level N+1 cheaper than level N) is either a bug or a mis-set knob. Note
L4 is Greedy while L5 is Lazy at the SAME depth/nice (16/30) — so L5 must cost more than L4
and compress better, and that difference isolates the cost of laziness exactly.

**P4 — Compressed size must fall monotonically with level.** Non-negotiable: a user typing a
higher number must not get a bigger file. Cheap to check, and it is the contract.

**P5 — Literal/match mix must shift with level.** Deeper search finds more and longer
matches, so match count per input byte should rise and mean match length should rise, while
literal fraction falls.

**P6 — Block count and block sizes should be governed by the splitter, not by input
arrival.** Blocks are capped at `SOFT_MAX_BLOCK_LENGTH` (300000) and ended early by
`BlockSplitStats`. Observed block sizes should cluster below the cap with a spread that
reflects content, not a spike at a buffer boundary. *Failure mode this catches:* exactly the
chunk-seam bug shipped on 2026-07-27, where blocks were forced to end at a 4 MiB refill
boundary.

**P7 — Level 0 does no matching at all.** Zero chain walks, zero hash inserts, and a byte
count out equal to input plus stored framing. Any matchfinder work at L0 is pure waste.

**P8 — The parallel path must not change the answer.** T>1 output must be byte-identical to
T1, or never larger. Same for a re-run at a different thread count.

## How this is checked

`fulcrum verify` — correctness, and P4/P8: compress at every level and thread count,
decompress with OUR OWN decoder at every thread count, sha256 against the original, plus one
independent decoder as a cross-check. Deterministic; no rig, no timing.

`fulcrum explain` — P1/P2/P3/P5/P6/P7: run the compiled binary under callgrind and DHAT,
and print, per level, the declared knobs beside the observed behaviour, with instructions
attributed by source file and function. The output is arranged so a declared-vs-observed
mismatch is impossible to read past.

Neither tool emits a score. They describe construction. The board — per-label size and wall
against the four rivals — remains the only thing that says whether we are winning.
