# What the vendors actually do — parameters and structure, side by side

This exists because three consecutive optimisation attempts failed in one session,
and all three failed the same way: they were found by opening OUR profile, picking
the biggest line, and shaving it. Every change that has ever WORKED here was found
by diffing against a vendor — their `seq++` against our `Vec::push`, their miss
rate against ours, what their build actually does against what we assumed.

So this file is not a scoreboard and contains no targets. It records the STRUCTURE
of each implementation so that a difference can be spotted, explained, and stolen.
Numbers here are for locating structure, never for optimising directly.

## 1. Level parameter tables

**gzip and zlib-ng** (`gzip/deflate.c:242`, `zlib-ng/deflate.c:105`) — one table,
`{good_length, max_lazy, nice_length, max_chain}` plus a strategy function:

| L | good | lazy | nice | chain | strategy |
|---|---|---|---|---|---|
| 1 | 4 | 4 | 8 | 4 | `deflate_fast` (zlib-ng: `deflate_quick`) |
| 2 | 4 | 5 | 16 | 8 | `deflate_fast` |
| 3 | 4 | 6 | 32 | 32 | `deflate_fast` |
| 4 | 4 | 12 | 32 | 24 | **zlib-ng: `deflate_medium`** / gzip: `deflate_slow` |
| 5 | 8 | 16 | 32 | 32 | `deflate_slow` |
| 6 | 8 | 16 | 128 | 128 | `deflate_slow` |
| 7 | 8 | 32 | 128 | 256 | `deflate_slow` |
| 8 | 32 | 128 | 258 | 1024 | `deflate_slow` |
| 9 | 32 | 258 | 258 | 4096 | `deflate_slow` |

**libdeflate** (`deflate_compress.c:3921`) — and OURS, which is a literal copy:

| L | strategy | max_search_depth | nice_match_length |
|---|---|---|---|
| 1 | fastest | (unused) | 32 |
| 2 | greedy | 6 | 10 |
| 3 | greedy | 12 | 14 |
| 4 | **greedy** | 16 | 30 |
| 5 | lazy | 16 | 30 |
| 6 | lazy | 35 | 65 |
| 7 | lazy | 100 | 130 |
| 8 | lazy2 | 300 | 258 |
| 9 | lazy2 | 600 | 258 |

⚠ **THIS TABLE USED TO STOP AT L6.** zlib-ng's table above runs to L9; ours stopped three
rows short, so the LARGEST divergence in the whole comparison had no row to sit beside and
was structurally invisible. Put the rows in before drawing a conclusion — a diff that omits
the rows is not a diff. Side by side, the chain column reads:

| L | zlib-ng chain | ours (max_search_depth) | we search |
|---|---|---|---|
| 5 | 32 | 16 | 2.0x shallower |
| 6 | 128 | 35 | **3.7x shallower** |
| 7 | 256 | 100 | 2.6x shallower |
| 8 | 1024 | 300 | 3.4x shallower |
| 9 | 4096 | 600 | **6.8x shallower** |

gzip -9 IS chain 4096. We were faster than gzip at L9 because we were doing a seventh of
the search, and bigger for exactly the same reason. Measured 2026-07-31: matching zlib's
chain depths at L5-L9 closes **84** failing size cells (and opens 13 — see below).

**Structural difference #1 — where lazy starts. CORRECTED 2026-07-30.** gzip switches to
`deflate_slow` (lazy) at L4. **zlib-ng does NOT** — its default L4 is `deflate_medium`
with knobs {4,12,32,24} (`vendor/zlib-ng/deflate.c:123-126`); the `deflate_slow` L4 row
is inside `#ifdef NO_MEDIUM_STRATEGY`, and `WITH_NEW_STRATEGIES` is ON by default
(`vendor/zlib-ng/CMakeLists.txt:102-103`). `deflate_medium` exists precisely to make
L3-6 monotonic at a fraction of lazy's cost, so it is a candidate for the P4 fix that
may fit inside promotion-rule clause 5 with no law change. The original claim here said
both vendors go lazy at L4 and was used to justify a lazy-at-L4 experiment that measured
17.7% wall. libdeflate stays greedy through L4 and switches at L5. We copied libdeflate.
That is the direct cause of our P4 violation (typing `-4` yields a BIGGER file than
`-3`): our L3 is `LazyGated` while our L4 is plain Greedy, so the ladder goes
lazy → greedy → lazy. gzip's ladder never does this. The fix is not a new
algorithm; it is a table entry, and the table is explicitly ours to change.

## 2. Block sizing — how much input per DEFLATE block

| implementation | bound | value |
|---|---|---|
| libdeflate (and ours) | input bytes / sequences | 300,000 / 50,000 |
| libdeflate FAST path | input bytes / sequences | 65,535 / 8,192 |
| zlib-ng / gzip | SYMBOLS (`lit_bufsize`) | 16,384 (`1 << (memLevel+6)`) |

**Structural difference #2.** libdeflate emits blocks roughly 3x larger by symbol
count than zlib does. Bigger blocks amortise the Huffman header over more symbols
(better ratio) but hold a bigger working set. We inherited libdeflate's number
without ever testing it as a parameter.

## 3. Hash geometry — the working set of the match search

| implementation | tables | entries | approx bytes |
|---|---|---|---|
| gzip | 1 | 2^15 | 64 KB |
| zlib-ng | 1 | 2^16 | 128 KB |
| libdeflate hc | 2 (hash3 + hash4) + next | 2^15 + 2^16 + 2^15 | ~256 KB |
| **igzip L0/L1** | **1** | **2^13 (8 K)** | **~16-32 KB** |
| igzip L2/L3 | 1 | 2^15 | ~64 KB |
| **ours, L0/L1 fast** | 2 (head + hash3) | **2^16 + 2^15** | ~192 KB |
| ours, L2+ hc | 2 + next | 2^15 + 2^16 + 2^15 | ~256 KB |

**Structural difference #3, and the one worth acting on.** igzip keeps its
level-0/1 table at **8 K entries** — small enough to stay resident in L1d (32 KB on
Zen2). We deliberately widened that same table to 64 K, 8x, and added a second
32 K hash3 table beside it.

The justification in `parse/fast.rs` is: *"at near-zero speed cost (same one load,
one compare per position)"*. That is an INSTRUCTION-COUNT argument, and eight lines
below it the same file records the measurement that refutes it: the dependent
`head[h]` load is *"69% of the L1 fast path's D1 read misses, and perf-confirmed as
the IPC collapse vs igzip on binary data (IPC 1.32 vs 2.46)"*. `PF_DIST`
prefetching was then added to paper over the miss the widening created.

Two adjacent comments, one counting loads and one counting misses, never
reconciled. Same loads, 8x the working set, is not the same speed.

## 4. Operation-level cost, us vs libdeflate (L2, silesia 8 MB, shipped build shape)

Identical position counts — 1,726,082 in both — so this is the same algorithm doing
the same work, and any difference is implementation.

| | libdeflate Ir | libdeflate Dr | gzippy Ir | gzippy Dr |
|---|---|---|---|---|
| matchfinder core | 259.3M | 34.2M | 249.4M | 51.2M |
| matchfinder common | 55.3M | 2.7M | 55.6M | 7.3M |
| parse | 79.5M | 14.7M | ~66.5M | ~9.7M |
| emit / flush block | 63.8M | 12.9M | ~57.4M | ~8.6M |
| **total** | **496.0M** | **83.8M** | **555.1M** | **104.0M** |

Our ALGORITHMIC loads match theirs exactly, line for line:
`next_tab[cur_pos] = hash4_tab[hash4]` is 6,273,912 reads in both; the hash-head
reads are 1,726,082 in both. The excess is loop state we read from MEMORY that they
hold in REGISTERS — `cutoff`, `nice_len` and `next_hashes[1]` cost us 1.7-2.5M
reads each and cost them zero — plus 13.7M reads (27% of ours) in unattributed
spill/reload code.

**Structural difference #4: register pressure, not algorithm.** Our
`longest_match` keeps more state live than theirs. This retro-explains two failed
levers: hand-hoisting the prefilter's invariant operands ADDED live values and
drove Dr *up*, and the deep levels (more live state) regressed hardest both times.

## 5. Where each vendor is worth stealing from

- **libdeflate** — register discipline in `longest_match`; the emit path is the one
  place we already beat them (0.96x), so the debt is entirely parse+matchfinder.
- **igzip** — the small-table fast path. Its L0/L1 working set is an order of
  magnitude below ours, which is the standing explanation for the 5-6x L0 gap.
- **zlib-ng** — `deflate_quick` (a distinct L1 strategy, not just tuned knobs), and
  a much smaller block/symbol budget than libdeflate's.
- **zopfli / ECT** — levels 10-12 only; out of scope until steps 1-2 close.

## How to use this file

Before proposing a change, name the vendor difference it is stealing and the
measurement that shows the difference is real. A change with no vendor counterpart
is allowed, but then say so explicitly — it means there is no existing proof that
the idea pays, and the bar is a measurement rather than a precedent.

## FALSIFIED — retuning the L2 level map (2026-07-28)

Our L2 is a copy of libdeflate's Greedy/6/10, and is byte-SIZE-IDENTICAL to
their -2 on ALL 20 files of the canonical corpus (`/root/archive/corpora`,
ratio 1.00000 everywhere). Since the per-label rule only requires size <=
theirs, the obvious move is to find a cheaper config that still clears the
bound. Lazy finds better matches per position, so it should reach the bound at
lower depth.

It does — and it is much MORE expensive, not less:

    L2 = Greedy/6/10 (shipped)   555,130,339 Ir
    L2 = Lazy/4/10               684,902,744 Ir   (+23.4%)

Lazy runs TWO searches per position; the lower depth does not come close to
paying for that. The route cannot help the wall.

It also fails the size leg independently. Lazy/4/10 is smaller on 16 of 20
files (data.sqlite -9.0%, data.json -2.9%, data.csv -1.8%) but LARGER on four:
access.log +0.49%, ecoli.fastq +0.61%, nasa-http +0.28%, aozora.txt +0.22% —
token-structured, highly repetitive text, where deferring a match costs.
Per-label means every file, so 16 wins do not buy the four losses.

FALSIFY: do not re-open "retune the L2 knobs for wall" without first measuring
the instruction cost of the candidate STRATEGY.

⚠ **CORRECTION 2026-07-31 — the next sentence used to read "Depth is not the cost; the
strategy is." That generalisation is FALSE and it was the single most expensive sentence in
this file.** It was measured at **L2 ONLY**, which `CLAUDE.md` hard stop #3 forbids
generalising ("never generalise a measurement across levels ... measured at a SHALLOW and a
DEEP level before it is believed"). At L5-L9 depth IS the cost: raising `max_search_depth`
to zlib-ng's chain values, changing NO strategy, closed 84 failing size cells. The claim
holds where it was measured — at L2, against L2's Greedy/6/10 — and nowhere else.

**A record can violate a hard stop and survive indefinitely, because hard stops are applied
to CHANGES and never to RECORDS.** This sentence closed the depth class for every session
that grepped it: `.git/logs/HEAD:235-237` shows `probe/l5-depth` created and abandoned 102
seconds later with no commit. Scope every falsification to the levels it was measured at,
in the sentence itself, or it will close a class it never tested.

Measured Ir before a size sweep would have killed this in one run instead of several.

A METHOD DEFECT this exposed, worth more than the result: the first pass of
this sweep was graded on four ad-hoc local files, and the two candidates it
liked (Lazy/3/10, Lazy/4/10) both failed once graded on the canonical corpus.
`reference_compression_corpus` says to use squishy and to avoid ad-hoc corpora
precisely for this reason. Size legs are cheap and deterministic — grade them
on all 20 files, always.

## THE ONLY CORRECTNESS REQUIREMENT: valid gzip content

User, 2026-07-28, after I twice tried to keep a weaker version of this rule:

**The output must be valid gzip content. That is the whole requirement.**

Concretely: the bytes we emit decode back to the exact original input, through
any conformant decoder (our own, gzip, pigz, libdeflate), and are format-legal
per RFC 1951/1952. Verified by roundtrip + sha256 at every thread count.

Nothing else is required. Specifically NOT required:

* Byte-identity with libdeflate, or any vendor. Matching a vendor's bytes only
  certifies that we reproduce their algorithm — which is exactly why we run
  their algorithm slower than they do. Receipt: our L2 and L4-L9 output is
  byte-SIZE-identical to libdeflate on all 20 canonical corpus files (ratio
  1.00000 everywhere), so those cells have ZERO size slack and must be won on
  wall alone against an implementation of the identical algorithm. We built that
  position ourselves by copying their level table.
* Byte-identity with our own T1 output. T>1 may emit different bytes.
* Byte-identity between streaming and whole-buffer encodes.
* Run-to-run determinism, even at a fixed thread count.

The goal is per-label: at the level the user typed, output at least as small as
the rival's AND less wall time. Any encoding that is valid gzip and achieves
that is permitted — a different parse, block-splitting policy, symbol budget,
level->config map, or an encoding no vendor would ever emit.

**Delete on sight**: any test or gate asserting our compressed bytes equal some
other bytes; any design decision justified by "this keeps us byte-identical to
X"; any framing of a divergence as a regression. A cell that was "tied" because
we emitted a vendor's exact bytes was never a win — it was a guarantee that we
could not beat them.

Audited 2026-07-28: no test asserts byte-equality against a vendor. Every
`libdeflater` use in `src/tests/` is either the roundtrip DECODER oracle or a
size BAR to beat; both are legitimate and both stay. The rationale in
`parse/mod.rs::level_has_resumable_parser` was reframed. Module docs calling a
file a "faithful transliteration" of a libdeflate header record where the code
CAME FROM — they impose no obligation to stay identical, and any of them may be
rewritten the moment a measurement says a different structure is better.

Practical consequence for #50 (103 of 165 failing size cells are T>1): fix it by
making seams SMALLER, not by making T4 reproduce T1. pigz's 10-bit
empty-static-block pad is exactly that.
