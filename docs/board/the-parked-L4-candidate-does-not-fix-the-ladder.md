# The parked L4 candidate does not fix the ladder — `Lazy(12,30)` does

**Measured 2026-08-01.** Deterministic size, T1, all 22 corpus files, via the
`ladder-tune` Cargo feature (a build feature, **not** a shipped env knob). Not a wall
claim.

`level.rs:267-277` parks L4 as `Lazy(10,30)`, justified as *"wins SIZE on 11/11 TUNE
files (772,154 B), 0 opened"* — i.e. selected on **total size**. Scored on the criterion
the ladder actually needs, it does not do the job.

A valid L4 must be **<= L3 on every file** (no sag) and **>= L5** (must not overtake the
next level).

| L4 candidate | total size | sags vs L3 | overtakes L5 |
|---|---|---|---|
| `Greedy(16,30)` — **ships today** | 213,942,000 | **17/22** | 1/22 |
| `Lazy(10,30)` — **the PARKED candidate** | 210,447,579 | **5/22** | 1/22 |
| **`Lazy(12,30)`** | **210,178,695** | **0/22** | **0/22** |
| `Lazy(14,30)` | 209,984,635 | 0/22 | 1/22 |
| `Lazy(16,30)` — identical to L5 | 209,808,452 | 0/22 | 0/22 |

The five files where the parked `Lazy(10,30)` still sags:

```
aozora.txt    +15,237      dickens     +13,865      dd79_text6  +8,511
engine.wasm      +363      movie.mp4       +74
```

## Why the sweep missed it

The record says *"depths 6/8/10/12 all measured, relation monotone, no interior point
exists — do not re-sweep."* Depth 12 WAS measured — but the candidates were ranked by
**total size**, and 10 was chosen to stay cheap on wall. **Nobody scored them for
monotonicity**, because monotonicity is not a board cell and nothing in the campaign
grades it.

`Lazy(12,30)` is also **268,884 B smaller in total** than the parked candidate, so it
wins on the record's own objective too. Depth 12 matches L3's own depth, making it the
minimum depth that does not downgrade search strength on the way up the ladder.

## What this does NOT do

**It does not unblock the lever.** `Lazy(12,30)` costs MORE wall than `Lazy(10,30)`, and
depth 10 already failed clause 5 on 9/11 TUNE. If anything this makes the wall problem
harder.

What it changes is the TARGET: **if a solvency wall run is ever spent on this family, it
must measure `Lazy(12,30)`, not `Lazy(10,30)`.** Re-measuring the parked point would burn
a run on a candidate that does not solve the problem it was parked for.

`level.rs` is not modified by this document; the accompanying record correction adds
these numbers to the PARK note itself.
