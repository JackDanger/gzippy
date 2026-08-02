# Our L3 improvement CAUSES the L4 sag — and reverting it gives 22/22 byte-parity

**Measured 2026-08-01.** Deterministic size, T1, all 22 corpus files. main `120bfa9c`.
Not a wall claim.

## The mechanism, from the level map

```
L3: Strategy::Lazy,   depth 12, nice 14     <- OURS. libdeflate's L3 is GREEDY(12,14)
L4: Strategy::Greedy, depth 16, nice 30     <- inherited unchanged: a PARSER DOWNGRADE
L5: Strategy::Lazy,   depth 16, nice 30     <- identical knobs to L4, stronger parser
```

**L4 is the only `Greedy` above L2.** Going L3 -> L4 downgrades the parser from lazy to
greedy, and lazy beats greedy on size. L4 and L5 have *identical* search knobs; only the
parser differs, so L4 is strictly weaker than L5 at the same nominal search cost.

libdeflate's own ladder is Greedy -> Greedy across L3/L4 and stays monotone:

| | files sagging in L2..L5 |
|---|---|
| **libdeflate** | **3 of 22** (dd79_bin6 +117 B, movie.mp4 +25 B, weights.safetensors) |
| **ours** | **17 of 22 at L4 alone** — worst `data.sqlite` **+2,125,650 (16.8%)** |

**Our L4 output is byte-identical to libdeflate's L4.** Only L3 moved. So the sag is not
inherited — we created it.

## The arithmetic

| | |
|---|---|
| our L3 (Lazy) beats libdeflate L3 (Greedy) by | **3,556,831 B** |
| the L4 sag that win creates costs | **2,836,732 B** |
| **net** | **720,099 B** |
| residual sag if L3 were Greedy(12,14) | **25 B, 1 file** (movie.mp4 — libdeflate sags there too) |

**The L3 win is not worth 3.56 MB. It is worth 720 KB**, because 80% of it is handed
back one level up.

## Option (a) MEASURED, not asserted: revert L3 to Greedy(12,14)

Via the `ladder-tune` Cargo feature (`GZIPPY_LADDER=greedy:12:14`), which is a build
feature and **not** a shipped env knob:

```
byte-ties with libdeflate -3 : 22 / 22
L3 -> L4 monotone            : 21 / 22
```

**Every corpus file byte-identical to libdeflate's L3.** That is phase 1's stated goal at
the one deep level where we currently diverge, and it removes the L4 sag as a side effect
because L4 already ties libdeflate.

### Method notes — two things nearly went wrong

1. **`/tmp` is unwritable in this environment.** The first A/A printed "differ=22" — that
   was the binary never executing, not a result.
2. **`cargo build --release --features ladder-tune` OVERWRITES `target/release/gzippy`.**
   Both binaries were checked with `--version` before anything was believed:
   `INSTRUMENTED(ladder-tune)` vs clean `parallel-sm+pure`.
3. **A/A then passed 22/22** with the override set to L3's real params, proving the
   instrumented build is size-neutral and the override is a true no-op at defaults.

## The trade

| | (a) revert L3 to Greedy | (b) upgrade L4 to Lazy |
|---|---|---|
| ladder | fixed, 21/22 monotone | fixed |
| libdeflate parity at L3 | **22/22 byte-tie** | still diverges |
| net size | **−720,099 B** | 0 |
| blocker | none; size-only, deterministic | **clause 5 on 9/11 TUNE — PARKED** (`level.rs:267-277`) |

Neither closes a board cell: L3 currently *wins* those cells and a tie also passes. This
is a ladder-quality decision, not a board decision — which is exactly why per-label
grading never surfaced it.

**`level.rs` is NOT modified by this document.** Option (a) deliberately gives up a
measured size win; CLAUDE.md says park monotone work rather than delete it, and the choice
is the user's.

Scripts: `scripts/campaign/curve.sh`.
