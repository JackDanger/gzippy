# The L1 gap is ALGORITHMIC: we emit 17% more literals and our matches are shorter

**`fulcrum why libdeflate:access.log:L1:T1:size`**, fulcrum `8364a0596f04` (built from
`origin/main`, not from a branch — the first attempt was refused by the tool because the
deployed binary was on `fix/counters-compress-mode`, and that refusal was correct).

**DENOMINATOR: 1 of 4 layers ran.** Layer 2 (callgrind per-line Ir) skipped — no
valgrind on this host. Layer 3 (hw counters) skipped — not Linux. Layer 4 (declared
params) skipped — the binary lacks `--features anatomy-counters`. **Nothing here is a
wall or instruction claim.**

## What the structure layer says

```
[1 STRUCTURE] POSITION COUNTS DIFFER
  matches Δ6.56%   matched-positions Δ0.55%   literals Δ16.98%
  -> different parse decisions — the gap is ALGORITHMIC

  ours : 2,143,724 tokens (1,180,433 matches, 963,291 literals), 29,204,461 bits
  rival: 1,931,229 tokens (1,107,753 matches,  823,476 literals), 26,481,182 bits
  header: ours 204,082 bits   rival 205,958 bits
```

## Reading it

**The header is not the story.** Ours is 204,082 bits, theirs 205,958 — we are
*smaller* there. The entire 2.7 Mbit gap is data bits (1.106 vs 1.002 per input byte).

**We emit MORE of everything**: 72,680 more matches AND 139,815 more literals, 212,495
more tokens in total (+11%). Covering the same input with more tokens means the tokens
are shorter. Average match length, derived from the per-byte rows:

```
ours  = (1 - 0.036747) / (0.081777 - 0.036747) = 21.4 bytes
rival = (1 - 0.031413) / (0.073671 - 0.031413) = 22.9 bytes
```

**Our matches are ~6.7% shorter, and 17% more of the input falls through as literals.**
That is the signature of a weaker matchfinder — one that fails to find the longer
candidate and gives up — not of a costing, framing or header problem.

## The consequence that matters for the lever

`access.log` is TEXT, and **libdeflate wins it decisively while having NO length-3
match support at all** (`ht_matchfinder.h:38-40` refuses it explicitly). So on text our
`head3` length-3 table is not what is holding us up, and removing it would not be the
loss — **what libdeflate has and we do not is the 2-way bucket**: `ht_matchfinder`
checks two candidates per hash and takes the better one, where igzip's chainless
`fast.rs` checks one and takes it.

Combined with the binary-file result recorded in
`after-227-the-whole-real-deficit-is-L1.md` — where our `head3` is the only thing
keeping `armexe.elf`, `data.sqlite` and `dd79_bin6` passing against gzip and pigz — the
target is now specified from BOTH sides:

> **2-way bucket (libdeflate's, for text) + length-3 table (ours, for binaries).**

That is exactly attempt 2 in the binding record at `src/compress/deflate/parse/mod.rs:540`,
which **passed its size leg and died on the T1 wall at 1.2662x**. This adds the
mechanism to that record: it is not a coincidence that the combination is what works —
each half is load-bearing on a different half of the corpus, and the vendor diff shows
why.

## What is still unknown

The wall. Layers 2 and 3 both skipped here, and the 1.2662x that killed attempt 2 is a
solvency measurement. **No amount of local size work will answer it.**
