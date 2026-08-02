# Byte-parity with libdeflate at T1 is a LIABILITY at T4

**Measured 2026-08-01.** Size only — arch-invariant and load-immune, so the coordinate
that matters here is THREAD COUNT, not the box. Not a wall claim.

## The experiment

Revert L3 from our `Lazy(12,14)` to libdeflate's own `Greedy(12,14)`. This is the
straightforward reading of phase 1 ("tie libdeflate on every cell"), and the T1 numbers
are excellent:

| L3 @ T1 | |
|---|---|
| byte-ties with libdeflate -3 | **22 / 22** |
| cells OPENED | **0** |
| cells CLOSED | **3** (ecoli.fastq + weights.safetensors vs libdeflate, weights vs igzip) |
| L4 ladder sag | **removed on 17 of 22 files** |

Every one of those numbers is real. **At T4 the same revert OPENS 12 CELLS:**

```
access.log  data.json  data.parquet  data.sqlite  dd79_bin6  dd79_text6
dickens     markup.xml monorepo.tar  movie.mp4    photo.jpg  sil40      — all vs libdeflate
```

Clause 3 is absolute. NO-SHIP.

## The mechanism — and it generalises

```
                    T1                       T4
reverted L3    delta   0 B  (exact tie)   +224 B   -> FAIL
current  L3    delta −26,874 B (margin)   −26,633 B -> PASS
                                          seam tax ~224-241 B either way
```

**A byte-tie at T1 has ZERO headroom, and the T>1 seam costs a couple hundred bytes.**
The tie is flipped by the seam; the margin absorbs it. Same seam, opposite outcome.

## What this inverts

Our Lazy L3 is **not** a phase-2 indulgence sitting in front of phase-1 parity. It is a
*monotone T1 size win buying headroom that is spent at T4* — precisely the mechanism
CLAUDE.md prescribes for the seam class: *"closed by monotone T1 size wins that buy
headroom to spend."*

It is **load-bearing for 12 T4 cells**, and the L4 ladder sag documented in
`our-L3-win-causes-the-L4-sag.md` is its **price**, not an independent defect. Anyone
"fixing" the sag by reverting L3 trades 12 T4 cells for a smoother curve.

## The corollary, which is the part worth carrying

**"Tie libdeflate on every cell" is, at T4, in tension with the board.** Parity means
zero margin, and the seam eats the margin. This is the same structure already recorded as
*"154/198 T1 cells are EXACT byte ties, so 109 T4 cells have ZERO headroom"* — this
document supplies the causal experiment behind that number, at a level where we currently
have margin and can watch it being spent.

A reopen needs a **new mechanism that removes the T>1 seam tax itself** — not a different
L3 config, and not a re-run at T1.

## Coordinates and limits

T1 and T4, level 3, all 22 corpus files, gzip/pigz/libdeflate/igzip (igzip built from
`vendor/isa-l`). Size only. The T4 arm used `-p4`; pigz was matched at `-p4`. The
candidate was produced with the `ladder-tune` Cargo feature (a build feature, **not** a
shipped env knob), A/A-verified 22/22 identical against the default build at default
params.
