# The 2-way-bucket falsification is scoped to T1, on an M1, by hyperfine

**Not a reopen. A scoping correction**, of exactly the kind CLAUDE.md hard stop #3 says
nothing enforces: *"NOTHING enforces a hard stop against the TEXT OF A RECORD, so a
falsification may itself violate #3 and then survive indefinitely, closing a class for
every session that greps it."*

## What the record says, and what its own commit message says

`src/compress/deflate/parse/fast.rs:1290-1296` closes the 2-way bucket by citing
`e0e4c44d`. That commit's message gives the coordinate verbatim:

```
Wall (M1, -p1, 60MB corpus, hyperfine mean of 25 runs post-warmup)
  ... cost L1 +33.6% wall on text60 and +24.2% on bin60
```

So the measurement is **T1**, on an **Apple M1**, by **hyperfine**. Three separate
scoping problems, none of which the `fast.rs` note carries:

1. **T1 only.** The wall budget at T1 is 0-8%. At T4 — where the rival is
   single-threaded — the recorded slack is **249-330%**. CLAUDE.md's own receipt is this
   exact error: *"the entire parse-config space was closed as 'unaffordable' against T1
   wall slack of 0-8%. The failing cells are T4… a 40x budget error that made every
   configuration look impossible."* **+33.6% against a 0-8% budget is impossible;
   +33.6% against a 249% budget is not.**
2. **M1 only.** The M1 is the recorded OUTLIER arch for wall: same commit, byte-identical
   output, L6/T1 vs libdeflate reads **AMD 1.081 / Intel 1.040 = LOSS, Apple M1 0.906 =
   WIN**. A whole session's wall conclusions from the M1 turned out to be an
   Apple-silicon artefact.
3. **hyperfine, not `fulcrum ab paired`.** Hard stop #6 forbids hand-rolled wall
   measurement, and the same-day receipt is a hand-rolled harness scoring the SAME BINARY
   at 6.10x against itself. There is no aa_bias for these numbers.

## What is INTRINSIC in that record, and stays

> *"the extra random-access candidate lookup itself (not the match-extend compute) is the
> tax, and no gating threshold tried reduced it below ~15-20% **because bin-class data's
> first candidate is weak often enough that the gate rarely skips the second lookup**"*

The mechanism is sound and permanent: a second probe costs on every lookup, hit AND miss,
and pays only on the hits. **That part is not coordinate-dependent and this document does
not dispute it.**

## The observation the record itself supports

The gate failed **because of bin-class data**. But of the 13 L1 cells that actually fail
at T4 (measured this session, `after-227-the-whole-real-deficit-is-L1.md`), **ten are
text-class**: access.log, aozora.txt, data.csv, data.json, dd79_text6, dickens,
ecoli.fastq, markup.xml, minjs.min.js, monorepo.tar. Three are binary-ish
(data.parquet, weights.safetensors, engine.wasm).

And the three binaries we must not break — `armexe.elf`, `data.sqlite`, `dd79_bin6` —
**are not in the failing set at all.** They pass today.

So the class of files where the gate was measured to fail is largely disjoint from the
class of files that need closing.

## What this does NOT license

* **It is not permission to rebuild it.** `parse/ht_fast.rs` already exists, compiled and
  unrouted. Two attempts are recorded. This document adds a coordinate, not a mechanism.
* **The size leg is NOT free either.** Routing L1 to exact-libdeflate is 14 closed / 6
  opened at BOTH T1 and T4 (measured). A 2-way bucket that KEEPS `head3` is a different
  shape and its size leg is unmeasured — but do not assume it inherits the ht result.
* **No wall claim is made here.** This box is an M1 and cannot settle it. Any reopen
  needs `fulcrum ab paired` at **T4 on solvency**, with aa_bias reported.

## The concrete, bounded ask

Add the coordinate to the record itself, at `fast.rs:1290`: *"measured at T1, on an M1,
by hyperfine"*. That is a one-line record-hygiene fix and it is what stops the next
session reading a T1/M1 number as a universal ceiling. **It should land as its own change
in the shipping repo — this port branch is the wrong home for it.**
