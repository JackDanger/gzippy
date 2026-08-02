# Peak memory is O(input), not O(1) — measured

**Measured 2026-08-01**, local M1, tree `976d558e`, `target/release/gzippy` sha256
`54079f43…`, `/usr/bin/time -l` peak RSS. Memory is a deterministic-enough,
load-immune quantity; this needs no frozen box. **This is not a wall claim.**

The user's stated destination: *"we will absolutely be doing this at some point in a
massively parallel way that also minimizes memory usage and still beats vendors on
speed and compression."* This is the measurement of where we stand on that axis.

## The two facts, measured directly — no model needed

**1. At T1, L1 is O(n) while every other level is O(1).**

| level | peak RSS on a 90,868,376-byte input | × input |
|---|---|---|
| **L1** | **177,192,960** | **1.95** |
| L2 | 13,713,408 | 0.15 |
| L6 | 13,680,640 | 0.15 |
| L9 | 13,697,024 | 0.15 |

Confirmed by the second consequence, checked rather than assumed: across inputs from
868 KB to 90 MB (a 105× range), **L1's RSS grows 35× while L6's grows 2.9×**.

| file | input | L1 RSS | L6 RSS | L1 ÷ input |
|---|---|---|---|---|
| engine.wasm | 868,202 | 5,062,656 | 4,702,208 | 5.83 |
| dickens | 12,174,519 | 20,168,704 | 9,207,808 | 1.66 |
| data.json | 14,215,394 | 18,939,904 | 8,798,208 | 1.33 |
| sil40 | 40,000,000 | 59,572,224 | 12,468,224 | 1.49 |
| monorepo.tar | 50,915,328 | 65,748,992 | 9,486,336 | 1.29 |
| weights.safetensors | 90,868,376 | 177,192,960 | 13,697,024 | 1.95 |

Mechanism, already recorded: `Strategy::Fast` is absent from
`parse::level_has_resumable_parser` (`src/compress/deflate/parse/mod.rs:770`), so L1
alone takes the whole-buffer fallback — `read_to_end` the entire input plus a reserve
of `input/2`. **Verified still present on `976d558e`.**

**2. At T>1 the level stops mattering: EVERY level becomes O(n).**

| level | -p1 | -p4 | -p16 |
|---|---|---|---|
| L1 | 1.95× | 2.44× | 2.58× |
| **L6** | **0.15×** | **2.44×** | **2.57×** |

This CORRECTS the existing record, which says "L1 alone takes the whole-buffer
fallback". That is true **at T1 only**. At T>1 the parallel path materialises the whole
input for every level, and L6 is exactly as bad as L1. On a 90 MB file at T16 we hold
234 MB — 2.57× the input.

## The linear model, and its honest range

Fitting `RSS_T4 = a·input + b·output + c` exactly on three points gives

```
RSS_T4  ≈  1.498 · input  +  0.860 · output  +  14.2 MB
```

The `1.498` is a striking corroboration of the mechanism — `read_to_end` (1.0×) plus
the `input/2` reserve (0.5×) is exactly 1.5×.

**But a 3-point exact solve fits by construction, so I tested it on five held-out
files:**

| file | input | predicted | measured | error |
|---|---|---|---|---|
| data.sqlite | 48,308,224 | 97,144,286 | 97,501,184 | **+0.4%** |
| tool.bin | 62,480,352 | 125,705,332 | 128,073,728 | **+1.8%** |
| data.parquet | 20,859,349 | 57,527,009 | 54,788,096 | −5.0% |
| ecoli.fastq | 26,214,271 | 57,307,675 | 52,019,200 | −10.2% |
| dickens | 12,174,519 | 36,341,403 | 30,670,848 | **−18.5%** |

**So the model is good to ~2% above roughly 48 MB and over-predicts by 5–19% below
that.** It is NOT a general law and must not be quoted as one — there is a
nonlinearity at small inputs it does not capture. The two facts in the section above
are direct measurements and do not depend on it.

## What this does and does not license

* It does **not** close a cell. Memory is not on the board; the board is size and wall.
* It does **not** claim a wall effect. Fewer page faults is a plausible wall mechanism
  and the existing record prices it at ~32% of the L1 gap, but that is a solvency
  measurement and was not attempted here.
* It **does** name a structural defect on an axis the user has explicitly stated as a
  destination, with the coordinate (T>1, every level), the magnitude (~1.5× input), and
  the mechanism (`read_to_end` + `input/2` reserve).
* The T1 L1 fix is **not a one-line gate change**: `parse_resumable` panics for
  strategies other than Greedy/Lazy/Lazy2, so routing `Strategy::Fast` through it
  requires `parse/fast.rs` to gain a resumable entry point first.

## Reproduce

```sh
/usr/bin/time -l ./target/release/gzippy -1 -c -p1 corpus/weights.safetensors >/dev/null
/usr/bin/time -l ./target/release/gzippy -6 -c -p4 corpus/weights.safetensors >/dev/null
```
