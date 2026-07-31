# The encoder we should be building, and every gap between it and what we ship

Written 2026-07-30, after a session that read all five vendors and measured our own
implementation against them cell by cell. This is the **target shape** and the **gap
list**, not a task queue: it exists because the campaign has been optimising locally with
no written target, and memory records that as the real defect ("enormous effort on local
optimization with NO written target shape, so the structure drifted").

Governing direction, from the user, 2026-07-30:

> Move in the direction of what's structurally better, ignoring whether it immediately
> moves any number. We're designing an implementation that will be optimal and it may be
> slow and bad until the very last edit.

So this document ranks by STRUCTURAL DISTANCE, not by cells closed. A gap that closes no
cell today still ranks if the target shape requires it.

---

## 0. What is already right — do not "improve" these

Two things are finished, and both were established by measurement, not assertion.

* **The emit path.** At L2 our emit executes **0.90x libdeflate's instructions** (57.4M vs
  63.8M) for identical output. Huffman build, precode RLE, the codeword LUT, the
  cheapest-of-{dynamic,static,stored} compare — all of it is already better than the
  vendor we are chasing. The entire remaining debt is parse + matchfinder.
* **The microarchitectural profile.** On the worst cell our IPC, frontend stalls, L1D miss
  rate and branch behaviour all BEAT libdeflate. **61% of our instruction excess is LOAD
  instructions.** This kills the whole "we need better SIMD / dispatch / cache behaviour"
  family outright: we do not have a stall problem, we have a *loads issued* problem.

Everything below is therefore about **how many loads and stores each coded position
costs**, and about **how many times per file we pay a fixed cost we did not need to pay**.

---

## 1. The target shape

### 1.1 One matchfinder family, parameterised — not four types

Every vendor's matchfinder is the same three steps: hash the position, retrieve N
candidates, extend the best. They differ on exactly three axes:

| axis | libdeflate L1 | libdeflate L2-9 | igzip L1 | zlib-ng L1 | ours today |
|---|---|---|---|---|---|
| candidates/position | 2 (inline bucket) | chain, depth 12-300 | 1 | 1 | **1 at L1**, chain at L2-9 |
| length-3 support | no | yes (`hash3_tab`) | no | no | yes at L1 (`head3`) and L2-9 |
| position width | **i16** | **i16** | i16 | u16 | **u32 at L1**, i16 at L2-9 |

Ours is the only implementation that mixes width. That is not a stylistic difference, it
is the largest single structural defect in the encoder — see gap **G1**.

**Target:** one finder type with `BUCKET: usize` and `LEN3: bool` as const generics over a
uniform `i16`/rebase position model. L1 becomes `<2, true>`, L2-9 `<chain, true>`, L0
`<1, false>`. `bt`/`lzfind` stay separate — they return different shapes (sorted lists,
packed Pareto pairs) for real reasons.

### 1.2 No data-dependent branches in the parse. At all.

Three of four vendors have **no** content adaptivity in the block-end decision, and none
has a content detector choosing a parser. We have both. `CLAUDE.md` non-negotiable #3
orders these deleted; they are still shipping. See **G2**.

### 1.3 Block ending should be COST-based, not drift-based

We ship libdeflate's drift detector bit-for-bit, and it is the *only* vendor with one:

| impl | block-end signal |
|---|---|
| **ours + libdeflate** | drift detector (SAD over 10 buckets) AND byte budget AND seq cap |
| zlib-ng | fixed 16,383-SYMBOL quantum, then per-block cheapest-of-three |
| igzip | fixed 65,536-TOKEN quantum, per-block hufftables + stored compare |
| gzip | lazy from L4, plus a periodic **estimated-cost early flush** every 4096 symbols |

**Measured evidence that ours under-splits** (`fulcrum why gzip:minjs.min.js:L7:T1:size`):
gzip spends **more** header bits than us (48,511 vs 40,595) and gets **fewer** data bits.
It splits into more, better-fitted blocks and wins on net.

But the budget dial is dead in both directions — **falsified this session**: raising it
(300K -> 450/600/900K) regresses incompressible data; lowering it (-> 150K/65,535)
regresses 4 of 5 files at L7, dickens by 8,915 B. So the missing mechanism is not a
different threshold, it is a **different signal**: nobody prices the block. See **G3**.

### 1.4 T>1 chunks must stop restarting the coder

Every seam today costs a sync-flush AND a block-grid restart AND a fresh Huffman
histogram. The first is 5 bytes; the other two are the expensive ones. See **G5**.

---

## 1.5 THE WALL BOARD, measured at last (2026-07-30) — the only wall front is libdeflate at T1

`fulcrum board wall`, frozen solvency (AMD EPYC 7282, boost=0, governor=performance,
tenants SIGSTOPped and restored), paired interleaved, per-pair median, n=9, /dev/null
sink, pin-gated. Scope declared: all 22 squishy members including `sil40`, **L6** (the
level a drop-in user gets by default), T1 and T4, all four rivals. 160 declared cells.
Artifact `/root/wallboard-L6/census.json`.

    measured_ok 75    ABSENT 40    VOID 45    slower 19

| rival | slower / measured |
|---|---|
| gzip | **0 / 20** |
| pigz | **0 / 36** |
| igzip | **0 / 0** (ABSENT — its CLI has no L6) |
| **libdeflate** | **19 / 19 — every single one, and every one at T1** |

**We beat gzip and pigz on every cell measured, usually by a lot** — gzip:ecoli.fastq
0.1762, pigz:ecoli.fastq 0.2454, gzip:aozora.txt 0.2506, gzip:engine.wasm 0.2864. Three to
five times faster.

**And we lose to libdeflate on all 19 of its measurable cells, all at T1**, from 1.0433
(access.log) to 1.2274 (photo.jpg), median ~1.10.

So the wall front and the size front are THE SAME FRONT: libdeflate holds 142 of the 198
failing size cells and 19 of 19 losing wall cells. **The campaign's entire remaining
problem is libdeflate, and at T1.** gzip, pigz and igzip are not wall fronts at L6 at all.

### The 45 VOIDs are an instrument limit, not a result — and it is a real gap

41 of them read `pin-gate FAIL: ours cpu%=~400 (ok=true) rival cpu%=100.0 (ok=false)`.
That is the gate working exactly as documented — it VOIDs a cell whose arm did not reach
the declared concurrency — but the arm that missed is the RIVAL, and it missed because
**gzip and libdeflate are single-threaded and have no `-p` flag to give them.** Every
T>1 cell against a single-threaded rival therefore VOIDs, permanently.

That is a genuine hole in the goal's coverage: "along every thread count" cannot be
scored against gzip or libdeflate as the gate stands. The user-facing question is
wall-clock at each tool's own default — a user types `gzip -6` and gets one thread,
types `gzippy -6` and gets many — so the comparison is meaningful even though the
concurrencies differ. Resolving it needs a declared-asymmetry mode in `wallcensus`
(rival is structurally single-threaded, record it and score anyway) rather than a VOID.
Until then, the T>1 wall board covers pigz only.

The 4 remaining VOIDs are `aa_bias` (0.0030-0.0089) — the harness's own A/A certificate
refusing cells where the two arms disagreed with themselves. That is the gate working
and needs no fix.

---

## 2. The gaps, ranked by structural distance

### G0 — THE L6 T1 WALL GAP IS REGISTER SPILL IN THE CHAIN WALK. **Located 2026-07-30. Start here.**

This is the highest-value locate the campaign has produced, and it is the ONLY remaining
wall front: libdeflate at T1 (19 of 19 losing wall cells, 1.0433-1.2274 at L6).

**Whole-program, cachegrind, 6,000,000 B at L6 T1, Zen2, ours vs libdeflate:**

| | dickens | photo.jpg |
|---|---|---|
| I refs | 1.24x | 1.38x |
| D reads | **1.49x** | **1.36x** |
| **D1 misses** | **1.003x** | **1.003x** |

**Cache behaviour is IDENTICAL.** The gap is pure issued work, and the wall tracks it
(photo.jpg has both the worst instruction ratio and the worst wall ratio).

**Per-function:** `matchfinder/hc.rs`, inlined into `lazy::run_resumable`, is **61.3% of
instructions and 68.2% of reads** — 112,415,312 reads, which alone is about libdeflate's
ENTIRE program (111,064,646).

**Per-line, and this is the finding.** Our excess over libdeflate is ~54M reads. These
lines account for ~40M of it, and every one of them is arithmetic on values that should
be in registers:

| Dr | line | what it should cost |
|---|---|---|
| **30,311,345** | `matchptr = (in_base_v as isize + cur_node4 as isize) as usize;` | **~0 — an ADD of two locals** |
| 33,425,344 | `<unknown (line 0)>` — the chain-walk inner loop | the real pointer chase |
| 16,737,027 | `if m_hi == n_hi && m_lo == n_lo` | legitimate: loads the candidate's bytes |
| 7,381,028 | `(best_len, (in_next - best_matchptr) as u32)` — the RETURN | **~0** |
| 2,727,454 | `if best_len >= nice_len` | **~0 — `nice_len` is a parameter** |
| 2,079,550 | `let mut cur_pos = in_next - *in_base;` | 1 load, not 2M |

**It is REGISTER PRESSURE, not aliasing — and that distinction is load-bearing.**
`in_base_v` is ALREADY a local (`let in_base_v = *in_base;` at the top of the function)
and is still reloaded 30 million times. So the fix is NOT to change `&mut usize`
parameters to by-value; the aliasing is already gone. LLVM is spilling because too many
values are live across the chain-walk loop. The plan's older T1-wall hypothesis
("LLVM cannot prove non-aliasing … so `cutoff`/`nice_len` live in memory") named the right
SYMPTOM and the wrong CAUSE.

**What that implies for the fix:** reduce what is live across the walk, not what is
borrowed. Candidates, unmeasured: sink the `hash3` path out of the length-4 loop so its
values are not live across it; drop `blen` (used only by `debug_assert!`, should be dead in
release but costs a register if it is not); split the function so the chain walk is its own
`#[inline(never)]` body with a minimal live set. Each is testable by re-running this exact
cg_annotate and watching the 30M line.

**Why this is NOT another instance of the "LLVM already did it" trap** (four instances
recorded in `matchfinder::ht`): those were cases where the compiler had ALREADY performed
the transform and hand-doing it added work. This is the opposite — a transform the compiler
demonstrably has NOT performed, with the cost visible per line. The falsifier is the same
command that found it, so a failed attempt costs one cachegrind run.

---

### G1 — L1: we are ALREADY the instruction champion and still lose on ratio. **Reframed 2026-07-30.**

**Measure this before doing anything else with L1.** Cachegrind, 6,000,000 B of data.csv
at L1, AMD Zen2, same input, three binaries:

| | I refs | vs libdeflate |
|---|---|---|
| **ours, shipped `parse::fast`** | **92,057,657** | **0.63x** |
| libdeflate L1 (`ht_matchfinder`) | 147,160,910 | 1.00x |
| our `ht` port (2-way + hash3) | 201,359,346 | 1.37x |

Two facts fall out, and both contradict how this gap was framed before:

1. **We are not bloated at L1 — we are LEANER than the vendor and worse anyway.** Our
   single-probe fast path executes **37% fewer instructions than libdeflate** and still
   emits 189% more literals. libdeflate BUYS its ratio with instructions; it is not
   getting the ratio for free and we are not paying for ours. Any framing of L1 as "we
   waste work libdeflate doesn't" is wrong and should not be repeated.
2. **The port has a ~54M implementation gap against the algorithm it copies** — 1.37x
   libdeflate for the same structure. That is almost exactly the per-position cost of the
   `hash3` table libdeflate's ht does not have (one extra hash, one load, one store, over
   ~6M positions). So the port is not "the ht structure measured"; it is the ht structure
   plus a third table, and the third table is the difference.

**Wall, frozen solvency, paired interleaved n=15, /dev/null both arms, L1 T1** — the ht
port against shipped `parse::fast`: data.csv **1.3709**, tool.bin 1.1805, aozora 1.1481,
dickens 1.1329, armexe.elf 1.0802. Slower everywhere, and NOT because of stores: reducing
the skip path from 3 stores per position to 2 (u32-packed bucket, output bit-identical)
cut D writes 26,927,232 -> 19,730,873 (**-26.7%**) and moved the wall not at all. Writes
were never the binding cost; instructions were.

**So the trade at L1 is explicit and it is a real trade, not an inefficiency to remove:**
the second candidate and the length-3 table each buy ratio and each cost instructions.
libdeflate pays for one of them (the bucket) and refuses the other (length-3, "due to its
focus on speed"). We currently refuse both at L1 and pay for it in ratio.

The unresolved question — and the ONLY one worth spending on here — is whether a second
candidate can be had for materially less than the ~55M instructions our bucket costs.
igzip's two-positions-per-iteration (P12) is the one shape that attacks exactly that: it
derives the second probe from work already done for the first, instead of doing a second
probe's worth of work. Everything else in this class is now closed:

* insert-density reduction: LIMIT_HASH_UPDATE, head-only bucket, and dropping the
  length-3 insert inside matches — **all three cost ratio.** Density IS the ratio.
* store-count reduction: u32-packed bucket — **-26.7% writes, zero wall movement.**
* bounds-check elision — byte-identical, zero movement, LLVM already did it.

---

### G1 (original framing, kept for the structure table)

`parse::fast`'s finder is fused into the parse loop as a **u32 head table of 64 K entries
(256 KiB) plus a 3-byte-keyed `head3` side table (128 KiB) = 384 KiB, single-probe**.
libdeflate's L1 is `[[i16; 2]; 1<<15]` — **128 KiB, two candidates**. Ours is 3x the
memory for half the candidates.

Measured consequence (`fulcrum why libdeflate:data.csv:L1:T1:size`, 26.5 MB input):

| | ours | libdeflate |
|---|---|---|
| matches | 1,846,129 | **1,930,665** (+4.58%) |
| literals | **741,183** | 256,099 (**+189.41%**) |
| input covered by matches | 97.20% | **99.03%** |
| header bits | 166,614 | 166,000 (within 0.37%) |

Headers agree, so this is purely the parse. **34 of the 90 fail-at-both-thread-counts
cells are L1**, and the worst cell on the whole promotion board is `access.log` L1 at
**+10.30%**.

*Status:* ported (`matchfinder::ht`, with a `hash3_tab` added — the synthesis no vendor
ships, since libdeflate has buckets without len3 at L1 and len3 without buckets at L2-9).
Size verdict: **4 cells closed, 0 flips, 0 erosion, fail-gap -67.97%**, every closed cell
landing at ratio exactly 1.0000. Wall verdict: **failed**, writes 3.68x.
*Blocking sub-gap:* **G1a**.

### G1a — inserts cost 2-3 stores where libdeflate pays 1

A 2-entry bucket costs two stores per insert (shift + head) and the length-3 table a
third. Inserting at every interior position of a ~14-byte average match is ~42 stores per
match against `parse::fast`'s ~3. Cachegrind, 6 MB of data.csv at L1 (Zen2):

    I refs   92,057,699 -> 190,151,913  (2.07x)
    D reads  17,484,786 ->  33,552,458  (1.92x)   <- expected: two candidates
    D WRITES  7,320,227 ->  26,927,232  (3.68x)   <- the regression

**Insert density and write traffic are the same dial** — limiting inserts (igzip's
`LIMIT_HASH_UPDATE`) gives the ratio straight back, past main on two files. Falsified.

*The target:* make each insert cheaper, not rarer. Unmeasured, in order:
(a) inside a match write only bucket slot 0 and skip the shift — **falsified, costs ratio**;
(b) drop the length-3 insert inside matches only (len3 pays off at match STARTS);
(c) **igzip's two-positions-per-iteration (P12)** — hash pos and pos+1 from ONE
computation, probe both, speculatively load both literal codes. This is the only named
mechanism that amortises work per position instead of trading density, and it is the
single most promising unbuilt thing in the encoder.

### G2 — content detectors and env knobs still ship, against a non-negotiable

`CLAUDE.md` #3 orders these deleted and they are still in the binary's source path:

* `L1_HASH3_GATE_LIT_THRESHOLD_PCT = 48` — a per-block content detector, with the
  threshold **fitted two points off a 2-point-wide cliff on the single file `dd79_bin6`**.
  This is the purest instance of the failure mode the user named ("focusing on a cell, or
  on one number").
* `LAZY_PEEK_GATED` + its own 48% threshold — a second, dead content gate kept "for
  provenance".
* the `l1-tune` module — 22 env knobs and a `RwLock<L1Tune>` read per `run()`, behind a
  build flavour its own note records at **1.1702x slower than what ships**. Its driver
  `fulcrum l1search` was already deleted as constitutionally banned. Knob search has
  produced **zero** of this project's counted wins.

*Correction to a standing claim:* the tune module is `#[cfg(feature = "l1-tune")]`,
default-off, and compiled OUT — verified `strings target/release/gzippy | grep -c L1TUNE`
= 0. So it does not corrupt measurements, and B1 was never blocked on deleting it. It
should still go, on the grounds above.

**These die for free the moment G1 lands**, because the ht finder has no gate to fire —
both tables are read and written at every position with no data-dependent branch. That is
the strongest structural argument for G1 and it is independent of any cell.

### G3 — nothing in the encoder ever PRICES a block

We decide block ends by drift and by budget. We never ask "would this block be cheaper
split in two?" zlib-ng does a per-block cheapest-of-three; igzip does per-block hufftables
plus a stored compare; gzip does a periodic estimated-cost early flush.

We do have `stored_block_bits` / `dynamic_bits` / `static_bits` at emit time — the pricing
machinery exists, it is simply never consulted by the *splitter*. A cost-based split check
at the existing 512-observation cadence would reuse it directly.

*Why this and not the budget:* the budget dial is falsified in both directions (§1.3), and
the drift detector is a copy of the only vendor that has one, so there is nothing left to
steal there. Cost is the signal nobody in this codebase has tried.

*Caution recorded:* the emit path already prices blocks 2-4x per block; adding a third
pricing call per 512 observations is a wall risk, and this cell's deficit is loads.

### G4 — the level table is libdeflate's, including its defects

L3 is `Lazy/12/14`, L4 is `Greedy/16/30`. Lazy beats greedy, so **-4 produces a bigger
file than -3 on 9 of 11 files** — a P4 contract violation a user can trip in one command.
We inherited it wholesale; `CLAUDE.md` says the map "is free to change".

Lazy at L4 fixes it and beats all four rivals by 1.2-3.1% on size, at **17.7% wall**.
zlib-ng's answer is `deflate_medium`, which exists precisely to make L3-6 monotonic at a
fraction of lazy's cost — **unmeasured here, and the only remaining route on this gap that
needs new code rather than a ruling.**

### G5 — T>1 seams restart the coder, and that is 60% of the failing board

**133 of 223 failing cells pass at T1 and fail only at T4.** Median gap 567 B; our own
T4-T1 deltas are 1,000-3,300 B. Each seam costs a 5-byte sync flush (pigz gets this to
<=4 with conditional 10-bit static padding — falsified on wall) plus, far more expensively,
a block-grid restart and a fresh Huffman histogram.

*Landed:* the chunk grid now follows available parallelism (`input/(threads*4)`, 512 KiB
floor, 8 MiB RSS cap) with seams rounded to the block budget. Collapses the seam cost —
tool.bin T4-T1 **2,828 -> 80 B**, weights.safetensors **2,962 -> -1 B**. Board 223 -> 203.

*The remaining structural fix, and the FALSIFIED note names it:* chunks must be **cheaper
to code independently** — a shared or pre-trained Huffman table across chunks, so a seam
does not forfeit the table. igzip already ships the machinery pattern (`isal_update_
histogram` + `isal_create_hufftables`, and a compiled-in trained table for L0). **Nothing
in our T>1 path shares coding state across chunks; only the dictionary is shared.**

### G6 — no large-match fast path, no literal-run skip

igzip: a match >= 264 enters an emit loop writing repeated 258-length codes with **no
re-searching**, re-hashing only the trailing positions; and after 2<<10 bytes without a
match it emits **8 literals per iteration with no hash lookups at all**, with the skip
growing 32 -> 128.

We have neither at L1+. A 300 KiB zero block costs us ~1,160 searches where igzip does ~73
compares. This is pure loads-per-position — the exact quantity §0 identifies as our
deficit — and it is why igzip dominates incompressible data.

### G7 — allocation and layout

libdeflate allocates **exactly 3 times**, all state in one per-level-sized struct.
zlib-ng carves window/prev/head/pending from ONE zalloc with 64-byte-aligned sub-buffers
and a cacheline-grouped state struct. We are at 68-83 after the ratchet, with **no
deliberate cacheline grouping of hot loop state** — which is the layout half of the
register/spill finding in `docs/encoder-campaign-plan.md`'s T1 wall item.

---

## 3. What this says to do next, in order

1. **G1a(c) — igzip's two-positions-per-iteration.** Unblocks G1, which deletes G2 for
   free. The only mechanism that adds candidates without adding per-position traffic.
2. **G5 shared coding state across chunks.** The named fix for 60% of the board; the grid
   change was the cheap half.
3. **G3 cost-based split check.** The one block-end signal nobody here has tried, and the
   dial-based alternatives are exhausted.
4. **G6 large-match + literal-run skip.** Directly targets loads-per-position on the data
   classes where we are worst.
5. **G4 `deflate_medium`.** Fixes a user-visible contract violation.

Not on this list, deliberately: any further sweep of `SOFT_MAX_BLOCK_LENGTH` (closed both
directions), any further chunk-grid constant (five shapes, all flip), bounds-check
elision (LLVM already does it), `LIMIT_HASH_UPDATE` on the ht finder (costs ratio), and
the signal-gated block-end bias (third instance of a banned detector). All seven are
falsified at their code sites.
