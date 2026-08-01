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

    measured_ok 111   ABSENT 40    VOID 9    slower 19

| rival | T1 | T4 |
|---|---|---|
| gzip | 0 slower / 18 | **0 slower / 19** — worst ratio 0.1941, i.e. 5x faster |
| pigz | 0 slower / 16 | **0 slower / 20** |
| igzip | ABSENT (its CLI has no L6) | ABSENT |
| **libdeflate** | **19 slower / 19** | **0 slower / 19** — 0.3052 to 0.5693, i.e. 2-3x faster |

**We are faster on 92 of 111 measured cells.** The 19 losses are ALL libdeflate and ALL
T1. At T4 we beat libdeflate on every single file, by 2-3x.

**We beat gzip and pigz on every cell measured, usually by a lot** — gzip:ecoli.fastq
0.1762, pigz:ecoli.fastq 0.2454, gzip:aozora.txt 0.2506, gzip:engine.wasm 0.2864. Three to
five times faster.

**And we lose to libdeflate on all 19 of its measurable cells, all at T1**, from 1.0433
(access.log) to 1.2274 (photo.jpg), median ~1.10.

So the wall front and the size front are THE SAME FRONT: libdeflate holds 142 of the 198
failing size cells and 19 of 19 losing wall cells. **The campaign's entire remaining
problem is libdeflate, and at T1.** gzip, pigz and igzip are not wall fronts at L6 at all.

### Re-measured 2026-07-31 on main @fd8fde5e — TUNE set, L6 T1, frozen, guarded

First wall board run through `scripts/campaign/board-wall.sh` (the guarded path), so the
binary is attested vanilla and the box was frozen. Artifacts: `docs/boards/wall-L6T1-fd8fde5e.tsv`.

    declared=44  measured_ok=29  absent=11  void=4  slower=8
    vs gzip        0 of 10 failing        vs pigz   0 of 11 failing
    vs igzip       0 of 0  (all 11 cells structurally ABSENT — igzip's CLI has no level 6)
    vs libdeflate  8 of 8 failing:
      armexe.elf 1.1387 · engine.wasm 1.1240 · movie.mp4 1.1222 · minjs.min.js 1.1199
      data.parquet 1.0738 · data.csv 1.0679 · tool.bin 1.0616 · dickens 1.0582

**Every measured libdeflate cell at L6 T1 fails; every gzip and pigz cell passes.** That is
the same shape as the older board and it is not improving: the worst cell is now
`armexe.elf` at +13.9%.

The 4 VOIDs are A/A harness bias 0.41–0.87% (aozora.txt, data.json, symbols.dwarf vs
libdeflate; data.parquet vs gzip) — the census refusing to report a ratio when its own
control drifted. Legitimate VOIDs, re-runnable, NOT results.

⚠ The previous line's "1.2274 (photo.jpg)" is a GATE member and should never have been the
headline. On TUNE members only, the chain-base win recomputes to geomean 0.9981 (0.19%) —
inside the ~1.5% noise floor. See the provenance correction in
`src/compress/deflate/matchfinder/hc.rs`. **Select and headline on TUNE; GATE only confirms.**

### The VOIDs were an instrument limit — FIXED, and it unlocked 38 cells

First run: 45 VOID of 160, **41 of them** `pin-gate FAIL: ours cpu%=~400 (ok=true) rival
cpu%=100.0 (ok=false)`. The gate was working as documented, but the arm that missed the
declared concurrency was the RIVAL, and it missed because **gzip and libdeflate are
single-threaded and have no `-p` flag to give them.** Every T>1 cell against a
single-threaded rival VOIDed permanently, so "never lose at any thread count" could not be
scored against them at all.

Fixed in fulcrum (`rival_is_thread_pinnable` + a `rival_single_threaded` flag declared on
the cell): a rival with no `{threads}` token in its template is a DECLARED ASYMMETRY, not a
mis-pinned arm. Our own arm is still gated normally, and a rival that DOES carry
`{threads}` and misses the window still VOIDs. Re-run: **VOID 45 -> 9, measured 75 -> 111,
38 declared-asymmetric cells scored for the first time — and every one of them is a win.**

The 9 remaining VOIDs are `aa_bias` — the A/A certificate refusing cells whose arms
disagreed with themselves. That is the gate working and needs no fix.

### Historical note on the original VOID diagnosis

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

**VENDOR-ANCHORED, and this is the number that matters.** Both matchfinders cachegrinded
on the same 6,000,000 B of dickens at L6, libdeflate built `-O2 -g` so its profile is not
one opaque symbol (hard stop #1):

| matchfinder, inlined into its lazy parser | Ir | **Dr** | D1 read misses |
|---|---|---|---|
| libdeflate `hc_matchfinder.h` | 358,511,770 | **57,593,044** | 26,260,946 |
| ours `hc.rs` | 414,673,369 | **108,282,726** | 26,285,498 |
| ratio | 1.157x | **1.880x** | **1.001x** |

**We issue 1.88x libdeflate's DATA READS inside the matchfinder, for byte-identical output
and an identical number of cache misses.** That is ~50.7M excess reads, and the whole
program's excess is 49.9M (161.0M vs 111.1M). **The entire L6 T1 gap is this one number.**

The misses matching to 0.1% is what makes it diagnostic: identical misses means identical
memory FOOTPRINT and identical chain-node visits. Same algorithm, same nodes, same cache
behaviour — twice the loads issued to walk them. Loads that hit L1 and do not exist in the
vendor's version are, by elimination, **spill/reload traffic**.

**LINE-FOR-LINE against the vendor, which narrows it further.** Same runs, both
matchfinders annotated:

| line | libdeflate Dr | ours Dr | ratio |
|---|---|---|---|
| pre-screen compare (`load_u32(matchptr + best_len - 3)` vs our `m_hi/m_lo`) | 16,161,531 | 16,737,027 | **1.04x — at parity** |
| `next_tab[cur_pos] = hash4_tab[hash4]` | 3,920,446 | 3,920,446 | **identical** |
| `if max_len < 5` | 2,079,550 | 2,079,550 | **identical** |
| chain chase / walk body | 15,586,216 | ~31,588,577 | **~2.0x** |
| the return (`*offset_ret = in_next - best_matchptr` vs our tuple) | 3,679,991 | 9,460,576 | **2.6x** |

**Everything that touches memory for a REASON is already at parity.** The pre-screen
compare, the table write and the tail check match to within 4%. The excess lives in
exactly two places, and neither is algorithmic:

* **the walk body, ~2x** — and note that de-pipelining it is FALSIFIED (1.0131 / 1.0624 /
  1.0992 slower at L2/L6/L9), so the pipelining is earning its keep; the cost is the extra
  live value it needs, not the pipelining itself.
* **the return, 2.6x** — 5.8M reads to hand back two `u32`s that should be in registers.
  `best_len` and `best_matchptr` are being spilled and reloaded at the return point.

One vendor difference is visible and is NOT available to us: libdeflate issues
`prefetchw(&mf->hash3_tab[next_hashes[0]])` for the NEXT position's head slot. Ours had a
prefetch of the next CHAIN NODE and it is falsified (it cost 103M L1 loads on this very
input). Those are different prefetches and the vendor's is untried here — but the falsified
one is close enough that any retry needs its own REOPEN, not this note as cover.

So the remaining target is exact and vendor-anchored: **get `hc.rs`'s Dr from 108M to
~58M.** Output is byte-identical at L2 and L4-L9, so no size cell can move while doing it —
a pure wall lever with a free correctness oracle.


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

## G8 — T>1 COPIES THE WHOLE INPUT FOR PADDING IT ALREADY HAS (verified 2026-07-31, NOT built)

**Verified in code, not inferred.** `infra/scheduler.rs:271` and `:275` hand each worker

```rust
let block = &input[start..end];
let dict  = Some(&input[dict_start..start]);   // dict_end == start
```

— two slices of ONE contiguous buffer that are **exactly adjacent**. `deflate/mod.rs:163-169`
then allocates `dict.len() + data.len() + BUF_PAD` and copies BOTH in:

```rust
let mut buf = Vec::with_capacity(cap);
buf.extend_from_slice(dict);
buf.extend_from_slice(data);
buf.resize(in_end + parse::BUF_PAD, 0);
```

The copy exists **only** to obtain `BUF_PAD` readable trailing bytes. `&input[dict_start..end]`
is already the same bytes in the same order, and for every chunk except the last,
`end < input.len()`, so those trailing bytes already exist in the mapping.

T=1 already solved exactly this: `encode_deflate_slack_padded_to_sink` (`deflate/mod.rs:265`)
takes a buffer that carries its own trailing pad and copies nothing. **T>1 never got the
equivalent.** This is the largest remaining structural difference between the two drivers,
and it is per-chunk.

Cost: at T=1 the staging buffers are the two biggest allocations after the `Sink`
(`mod.rs:395` 4,564,792 B and `mod.rs:404` 2,098,144 B on a 6 MB input); at T>1 the same
copy runs once per chunk, over the entire input plus 32 KiB of dictionary per chunk.

**Why it is not built here.** It is an unsafe-bounds change on the T>1 hot path. `BUF_PAD`
exists so `load_u32` near `in_end` cannot read out of bounds; with a shared mapping those
over-reads land in the NEXT chunk's bytes, which is valid readable memory, and the parser
already clamps `max_len` so no match can be emitted past `in_end`. That argument is sound
but it must be discharged for EVERY parser, not the one that is convenient —
`feedback_unsafe_bound_must_cover_all_callers` records a SAFETY comment here that held for
2 of 4 parsers while `silesia -10` was at 86% of the allocation. The final chunk genuinely
has no slack and needs real padding.

**Gate to pre-register before attempting:** allocated bytes DOWN at T4 and T16, output
byte-identical at T1/T4/T8 across L1/L6/L9, roundtrip through our decoder AND gzip at every
thread count, and the bound discharged in writing for all four parsers.

## G9 — L1 IS THE CAMPAIGN (full-board evidence, 2026-07-31)

The FALSIFY record in `parse/mod.rs` that closed the L1 class was measured on the TUNE
set only — 96 failing cells. The full board (`all`: 22 files, L1-9, T1+T4, four rivals,
1,584 declared / 1,320 measured, **0 VOID**, artifact `/root/sizeboard-all-12fcd0ed/`)
changes what we know about WHERE the campaign is stuck. Ranked by BYTES, not percent:

    FAILING: 200 of 1,320
      libdeflate  142 cells   4,326,245 bytes lost
      gzip         32 cells     528,386
      pigz         20 cells     370,271
      igzip         6 cells     354,208

    BY LEVEL:  L1 35 · L2 23 · L3 12 · L4 17 · L5 25 · L6 25 · L7 26 · L8 17 · L9 20

    WORST CELLS — every one is L1 vs libdeflate:
      access.log   L1  +10.32%  (+341,529 B)      monorepo.tar L1  +5.65% (+639,359 B)
      data.csv     L1   +4.57%                    aozora.txt   L1  +4.05%
      ecoli.fastq  L1   +2.83%                    markup.xml   L1  +2.52%
      minjs.min.js L1   +2.26%                    dickens      L1  +2.12%

**L1 has the most failing cells AND all of the large ones.** L2-L9 losses are almost all
under 1% and many are under 0.01% (`tool.bin` L4 is 125 B on 21 MB). The board is not 200
independent problems: it is one frontier at L1 plus a long sub-1% tail.

### The L1 frontier is real, and both obvious routes are measured and closed

See the two FALSIFY records at `parse/mod.rs`'s `Strategy::Fast` arm:
* **Replacement** (route to `ht_fast`): 9 cells closed, every one landing at ratio
  EXACTLY 1.0000 — the transliteration is faithful — but **7 opened**, all binaries,
  one mechanism: our `head3` length-3 table beats libdeflate there and `ht_matchfinder`
  deliberately has no length-3 support.
* **Synthesis** (2-way bucket AND hash3): size passed cleanly, **wall killed it** — 19
  cells eroded, our own L1 15-50% slower. Mechanism: the size win comes from more work
  per position, and halving bytes-resident does not pay for extra dependent LOADS ISSUED.

That is 2 for 2 against size-only arguments in this class, and it agrees with
`project_encoder_deficit_is_loads_not_stalls` and with this session's independent
`hc.rs` result (a -20.4% read count that lost the wall).

### The sanctioned reopen, now vendor-located

The record requires "a mechanism that adds candidates WITHOUT adding dependent loads per
position", naming igzip's P12/P13. **P12 is located** (hard stop #1 discharged):
`vendor/isa-l/igzip/igzip_icf_body_h1_gr_bt.asm` computes `hash` AND `hash2` in the same
loop iteration (`:263-266`, `:298-311`) with `lea tmp3, [f_i + 1]` (`:358`) — it probes
position i and i+1 from one iteration, interleaving two dependent load chains so they
overlap instead of serialising. The C reference `igzip_base.c:55-100` does NOT do this;
it is one position per iteration, so reading only the C would have missed it.

Note what P12 is and is not: it is a LOADS/LATENCY mechanism, not a ratio mechanism. It
does not close a size cell by itself. Its role is to buy back the per-position budget the
synthesis overspent, making the synthesis's ratio win affordable on the wall. So the
sequence is P12 first, measured on wall alone with size held byte-identical, and only
then re-attempt the synthesis on top of it.

**NOT BUILT.** Stated plainly so the next session does not mistake a located mechanism
for a validated one.

## G10 — THE L5-L7 DEPTH LEVER IS BLOCKED BY THE COST MODEL, NOT BY SEARCH (measured 2026-07-31)

Matching zlib-ng's chain depths at **L5/L6/L7 only** (32/128/256; L8/L9 deliberately untouched)
closes **65** failing size cells and flips **7**. Board artifact `/root/size-L567/`, 396 cells,
0 VOID. Scoping to L5-L7 also removes the wall catastrophe that made the L5-L9 version NO-SHIP
(`data.csv` L9 went 0.98s -> 2.39s at depth 4096; at L9 we already emit libdeflate's EXACT
output while being faster than it, so there was never anything to win there).

The 7 flips have one signature: **libdeflate, T1, near-incompressible data, and our baseline is
BYTE-IDENTICAL to libdeflate.**

    dd79_bin6           L5  4,461,731 -> 4,461,736      L6  4,461,731 -> 4,461,732
    movie.mp4           L5 12,890,419 -> 12,890,473     L6 12,890,404 -> 12,890,464
    photo.jpg           L7  6,472,036 -> 6,472,041
    weights.safetensors L5 83,113,545 -> 83,113,685     L7 83,113,840 -> 83,113,842

### The mechanism, measured — it is HUFFMAN, not the matchfinder

`anatomy-counters`, movie.mp4 at L6, depth 35 vs depth 128:

    literals_emitted          12,762,619  ->  12,762,598   (-21)
    matches_emitted               47,029  ->      46,969   (-60)
    match_length_bytes_total     179,638  ->     179,659   (+21)

Deeper search emits **81 FEWER symbols** and covers **more input with matches** — strictly better
by every match-finding metric — and the output is **60 bytes BIGGER**.

So the loss is not in finding matches; it is in CODING them. On near-incompressible input the
handful of matches present is what skews the litlen/dist distributions. Replacing 60 short
matches with slightly longer ones flattens those distributions, and the extra bits-per-symbol
exceed the symbols saved. We accept a longer match because it is longer, never asking whether
it is CHEAPER IN BITS.

### What follows

* **No amount of parameter tuning fixes this.** Depth, `nice_match_length` and zlib's
  `good_match` brake were each measured; none restores the flips (`good_match` makes
  `data.csv` L9 worse, and changes output even at stock depth, so it is not free).
* **The fix is cost-based match acceptance** — compare the bit cost of the match against the
  literals it replaces, using the running symbol frequencies we already maintain in `Sink`.
  That is what `near_optimal` does at L10-12 and what zlib-ng's `deflate_medium` does with its
  deferred-match arithmetic. It is vendor-precedented twice and needs no content detection.
* **Do not re-run the depth sweep before the cost model exists.** The size verdict is banked:
  65 closed / 7 flipped at L5-L7, wall-neutral. The lever is worth 65 cells the day acceptance
  becomes cost-aware, and clause 3 blocks it until then.

## G11 — THE MISSING MIDDLE IS `deflate_medium`'s `fizzle_matches` (located 2026-07-31, NOT built)

Three independent instruments now agree the L5-L7 blocker is the PARSE, and this names the
vendor mechanism that sits exactly in the wall gap.

### The size headroom is PROVEN, with shipping code

Routing L6 to `Strategy::NearOptimal` — no new code, the L10-12 parser — gives **0 of 132
failing cells** at L6 (all four rivals, T1+T4, 0 VOID, artifact `/root/size-L6-nearopt/`),
against 25 failing on main. Margins are large, not marginal:

    aozora.txt  3,781,301 vs best rival 4,049,212   -6.6%
    dickens     4,322,130 vs           4,539,505    -4.8%
    dd79_bin6   4,285,099 vs           4,461,731    -4.0%

Even the CHEAPEST near-optimal config (`max_search_depth: 6`, `max_optim_passes: 1`) still
beats libdeflate on dickens by 3.0%. The headroom is not config-sensitive.

### The wall cost, corrected — it is worse on text than on binaries

    dickens L6      main 0.15s   libdeflate 0.17s
      near-optimal  depth 35 / 2 passes  0.93s   6.2x
                    depth 12 / 1 pass    0.76s   5.1x
                    depth  6 / 1 pass    0.68s   4.5x

⚠ An earlier note in this session said "~2.5x". That came from movie.mp4, photo.jpg and
dd79_bin6 — all near-incompressible, so few matches and a cheap DP. On text it is 4.5-6.2x.
Generalising from one file CLASS is the same defect as generalising across levels
(hard stop #3). The 2.5x figure is retracted.

Located cost, from `anatomy-counters` on dd79_bin6: near-optimal uses the binary-tree finder,
which issues **2.1x the head-table reads of the hash-chain finder plus 16.4M child-table
writes `hc` never makes** (bt 18,874,356 reads / 16,412,505 child writes vs hc 8,796,490 / 0).

### THE MECHANISM, and it is not dynamic programming

`fulcrum anatomy ratio map` on movie.mp4 shows the optimal-parse frontier beating both us and
libdeflate by 10,172 bits, with gzippy vs libdeflate at Δ=0. Its winning regions look like:

    ours      (4,18850)@2482408   (3,5796)@2482412        53 bits
    frontier  lit x2@2482408      (5,5796)@2482410        41 bits

The frontier moved the SECOND match two positions LEFT (length 3 -> 5) and dissolved the
first into literals. That is exactly `fizzle_matches` in zlib-ng's
`vendor/zlib-ng/deflate_medium.c:128-175`: it holds `current` and `next`, slides the boundary
between them left while the bytes still match, and commits only if `current` collapses to <= 1
(so it becomes literals) and `next` did not degenerate to length 2.

**We do not have it.** `grep -rn "medium" src/compress/deflate/level.rs` returns nothing; our
strategy ladder is Fast0/Fast/Greedy/Lazy/Lazy2/NearOptimal, with nothing between Lazy2 and
NearOptimal. zlib-ng puts `deflate_medium` at L4-L6 by default.

Why this is the right shape and a threshold was not: a blanket too-far rule for length-4 was
built and FALSIFIED (dickens +14,791 B) precisely because the frontier's choice was
CONTEXTUAL — it depended on a better match existing two positions on. `fizzle_matches` is
bounded (one match pair, no table, no DP) and looks at exactly that context.

### Pre-registered gate for the attempt

Size: must not regress any currently-passing L5-L7 cell (clause 3 is absolute), measured on
the full 22-file board at T1 and T4. Wall: must stay under the L6 erosion budget against
gzip/pigz AND not flip any libdeflate cell — the depth lever died on exactly that leg, so
wall is not optional here and a size-only argument is insufficient (3 for 3 in this class).

## G30 — the depth/size curve SATURATES, and the saturation point is absolute, not relative

The shipped T>1 change (#236) scales `max_search_depth` x4. That factor was a first guess.
Measuring the curve rather than sweeping for a better number (sil40, T4, exact bytes):

    level      x2           x4           x8           x16          libdeflate
    L2    15,964,129   15,901,805   15,869,951   15,849,165   16,059,080
    L6    15,516,055   15,494,305   15,482,185   15,477,116   15,555,063
    L9    15,451,122   15,450,182   15,450,724   15,450,696   15,452,666
                        ^ best       ^ WORSE

L9 SATURATES AT x4 AND REGRESSES BEYOND IT. L2 and L6 are still improving at x16 (L2 gains
another 52,640 B from x4 to x16; L6 another 17,189 B).

The mechanism is that saturation is roughly constant in ABSOLUTE depth, not in the multiplier.
L9 already searches 600 nodes; x4 puts it at 2,400, past the point where another chain node
finds a better match often enough to pay for itself — and past it, the extra work can pick a
different (slightly worse) match at equal length. L2 searches 6; even x16 leaves it at 96,
still shallow.

So a uniform multiplier is the wrong SHAPE: it over-spends at deep levels and under-spends at
shallow ones. The structurally right form is a multiplier with an ABSOLUTE CAP — around x8
capped near 2,400 would take L2 to 48 and L6 to 280 (both at their x8 sizes) while leaving L9
at its x4 optimum, i.e. strictly better than the shipped x4 on all three.

NOT CHANGED HERE. #236's x4 is under graded adjudication on the frozen box with a SHIP verdict
already banked on the size axis (39 cells closed, 0 opened). Hard stop #5 says land gated work
before starting new work; changing the constant now would void that run. The cap is the
follow-up, and it must be re-gated on its own.

## G31 — the gzip/pigz class is a DEPTH deficit against zlib, and zlib affords its depth with `good_match`

The frozen-box census (G28) splits the 68 failures by rival. The gzip and pigz cells have a
completely different signature from the libdeflate ones, and they come in T1/T4 PAIRS with
near-identical deltas — so they are parse-quality deficits, not seam:

    rival  file                 L   T1 delta   T4 delta   ratio
    pigz   monorepo.tar         6   +53,495    +53,896    1.0054
    pigz   dd79_bin6            2   +36,101    +36,570    1.0082
    gzip   aozora.txt           6   +23,082    +23,338    1.0057
    pigz   aozora.txt           6   +18,327    +18,583    1.0046
    gzip   weights.safetensors  9   +17,815    +18,139    1.0002
    gzip   minjs.min.js         6    +3,719     +3,574    1.0034
    gzip   photo.jpg            2    +2,905     +2,963    1.0005

0.3-0.8% deficits, against libdeflate's 0.001-0.05%. Different magnitude, different cause.

VENDOR DIFF, read from `vendor/zlib-ng/deflate.c`'s `configuration_table` (gzip and pigz are
both zlib-family):

    level   zlib-ng: good lazy nice chain        ours (libdeflate port): depth nice
    L6            8   16  128   128                              35   65
    L7            8   32  128   256                             100  130
    L8           32  128  258  1024                             300  258

AT L6 THEY SEARCH 128 CHAIN NODES TO OUR 35 — 3.66x deeper, with nice 128 vs our 65. That is
the mechanism behind the whole 0.5%-class, and it is a direct consequence of inheriting
libdeflate's table (G15) rather than choosing our own: libdeflate trades depth for speed at L6,
and against the zlib family that trade loses on size.

CONFIRMATION FROM THE SHIPPED FIX. #236 scales T>1 depth x4, putting L6 at 140 — essentially
zlib's 128. The graded lever closed exactly the T4 halves of this class:
`gzip:aozora.txt:L6:T4`, `gzip:minjs.min.js:L6:T4`, `gzip:monorepo.tar:L6:T4` and the three
matching `pigz:` cells. The prediction and the result agree, which is what makes this a
mechanism rather than a correlation.

### The T1 halves need a different lever, and zlib names it

The T1 twins of those cells are untouched, because at T1 the binding wall constraint is
libdeflate (4.4% slack at L6 on sil40), and going from depth 35 to 128 costs far more than that.

But zlib runs chain=128 at L6 and is still not slow, because it also sets `good_match = 8`: the
chain walk STOPS EARLY once a match of at least `good_match` length is found. Deep chains are
therefore cheap on average and expensive only where they pay. We have no such early exit — our
depth is a hard loop bound.

So the T1 class is not "we cannot afford depth". It is "we buy depth at full price and they buy
it at a discount". The lever is the early exit, not the bound.

NOT ATTEMPTED HERE. Note that an uncommitted `good_match` implementation was found in this
checkout belonging to no branch (see the contamination note in G29) — another writer is
already on this. Coordinate before duplicating it.

## G31a — the `good_match` work (ANOTHER WRITER'S, not mine) closes the T1 gzip/pigz class

G31 predicted that the T1 halves of the gzip/pigz class need zlib's `good_match` early exit
rather than more depth. There is an uncommitted `good_match` implementation in this checkout
belonging to no branch (found as contamination, G29). Its level table is `4,4,4,8,8,8,32,32` —
zlib-ng's `good` column exactly, so it is a direct port of the mechanism G31 named.

TESTED WITHOUT TOUCHING THE OTHER WRITER'S TREE: applied to a detached worktree off `main`
(`git worktree add --detach`), vendor symlinked, built vanilla. Sizes are deterministic, T1,
local M1. This measures THEIR work; it is recorded here only so the result is not lost.

    file           L  rival   rival bytes   main T1      +good_match   verdict
    aozora.txt     6  gzip      4,049,212   4,072,294    4,013,389     CLOSES (-35,823)
    aozora.txt     6  pigz      4,053,967   4,072,294    4,013,389     CLOSES
    monorepo.tar   6  pigz      9,898,201   9,951,696    9,870,485     CLOSES (-27,716)
    minjs.min.js   6  gzip      1,088,768   1,092,487    1,087,222     CLOSES
    dd79_bin6      2  pigz      4,464,656   4,500,757    4,500,757     unchanged

FOUR OF FIVE CLOSE, and they are the four the mechanism predicts: all L6, where zlib walks 128
chain nodes to our 35 and pays for it with the early exit. `dd79_bin6` at L2 is untouched —
a separate cell needing a separate mechanism, and it should not be attributed to this lever.

This is the second prediction from the G31 vendor diff to be confirmed by execution (the first
was #236's depth x4 closing exactly the T4 halves of the same class). The two levers are
complementary, not competing: depth-at-T>1 closes the T4 halves, `good_match` closes the T1
halves, and together they cover the class.

COORDINATION, not a claim: this work is not mine, is not on a branch, and is not gated. It
needs its owner to land it through `fulcrum try` like anything else. Recorded so that (a) the
measurement exists, and (b) nobody re-implements it.

## G32 — the worst cell on the board (dd79_bin6 L2, ratio 1.0082): we are WEAKER at L2 than zlib

`pigz:dd79_bin6:L2` is the largest ratio deficit on the frozen board (+36,101 B, 1.0082).
Profiled across levels (6,291,456 B input, T1, exact bytes, local M1):

    L    pigz         ours         delta        libdeflate    ours-vs-ld
    1    4,491,598    4,480,440    -11,158      4,667,981     -187,541
    2    4,464,656    4,500,757    +36,101      4,500,757            0   <- THE CELL
    3    4,441,970    4,461,737    +19,767      4,500,874      -39,137
    4    4,527,176    4,500,874    -26,302      4,500,874            0
    6    4,493,254    4,461,731    -31,523      4,461,731            0
    9    4,493,254    4,452,384    -40,870      4,452,384            0

THREE FACTS, none of which is "we need a better parser":

1. WE ARE BYTE-IDENTICAL TO LIBDEFLATE AT L2, L4, L6 AND L9 on this file (delta exactly 0).
   The cage (G15) again: at those labels we ARE libdeflate, so we inherit their L2 weakness.

2. PIGZ IS NON-MONOTONIC HERE: its L4 (4,527,176) is WORSE than its L2 (4,464,656), and its
   L6 and L9 are identical (4,493,254). zlib's own ladder is not ordered on this input.

3. OUR L3 ALREADY BEATS PIGZ'S L2 — 4,461,737 vs 4,464,656. The capability exists in our own
   ladder; it is on the wrong label.

VENDOR DIFF (stock zlib, which pigz uses — note this differs from zlib-ng's table read in
G31): zlib L2 is `good=4 lazy=5 nice=16 chain=8`; ours is `Greedy depth=6 nice=10`. At L2 we
search SHALLOWER (6 vs 8) with a lower nice-length (10 vs 16). We are weaker at L2, and the
G31 story (we are shallower than zlib) holds at L2 as well as L6 — the numbers are just
smaller.

### Why this cell is the hardest one on the board

L2 at T1 is the one place both constraints bind at once. On SIZE we need a STRONGER L2 (zlib's
8/16 rather than our 6/10). On WALL we are already LOSING at L2 T1 — 1.044 against libdeflate
on sil40 — so we cannot simply spend more search. Stronger and faster, simultaneously.

That is exactly the shape `good_match` fixes, and it is why the tested `good_match` patch
(G31a) did NOT move this cell: it adds the early exit at L2 but leaves depth at 6 and nice at
10. The proposal that follows from both diffs is the PAIR — early exit AND zlib's L2 knobs
(depth 8, nice 16) — where the exit pays for the extra depth. Neither half alone does it,
which is the composition rule from `CLAUDE.md`'s rule 5 stated in advance rather than after.

NOT ATTEMPTED. It touches the T1 table, so it must clear T1 wall against libdeflate, which is
the tightest budget on the board.

## G32a — FALSIFIED, and I predicted it in advance: `good_match` + zlib's L2 knobs does NOT close the worst cell

G32 proposed the PAIR — the `good_match` early exit plus zlib's L2 knobs (depth 8, nice 16) —
on the argument that neither half alone moves `pigz:dd79_bin6:L2` and the exit would pay for
the extra depth. Built and measured (detached worktree off `main`, the foreign `good_match`
patch plus the knob change, vanilla build, T1, exact bytes):

    file          pigz         libdeflate   main         composed     verdict
    dd79_bin6     4,464,656    4,500,757    4,500,757    4,500,804    WORSE than main (+47)
    data.csv      4,425,984    3,923,216    3,923,216    3,941,022    OPENS a cell (+17,806)
    dickens       5,167,504    4,772,260    4,772,260    4,737,378    beats both
    winexe.exe    1,637,306    1,569,179    1,569,179    1,561,372    beats both
    photo.jpg     6,480,485    6,473,516    6,473,516    6,473,492    beats both
    sil40        16,869,300   16,059,080   16,059,080   15,972,405    beats both

THE TARGET CELL DID NOT MOVE — it got 47 B worse — and `data.csv` REGRESSED by 17,806 B,
turning a passing cell into a failing one. The proposal is dead as stated.

WHY, and it is a correction to G32's reading of the vendor. zlib's `good_match` does not
merely exist alongside the chain bound; it SHORTENS THE SEARCH when a good-enough match is
found (zlib reduces `chain_length` once `len >= good_match`). At L2 with `good_match = 4`, a
4-byte match is found almost immediately on binary data, so the exit fires at once and the
EFFECTIVE depth drops BELOW the nominal 8 — which is why raising the bound to 8 bought
nothing on `dd79_bin6` and why a file that depends on longer searches (`data.csv`) lost.

So `good_match` is not a free way to buy depth. It is a trade: cheaper on inputs where short
matches are good enough, WORSE on inputs where they are not. That is a data-dependent effect
in a static parameter — the same shape as the Greedy/Lazy failure in #236, where lazy helped
dense-match data and hurt sparse-match data.

WHAT SURVIVES. G31a's measurement stands on its own terms: the foreign `good_match` patch
UNCHANGED (depth 6, nice 10) closes four L6 T1 cells. That is its owner's result and this
falsification does not touch it. What is falsified is MY proposal to combine it with raised L2
knobs.

REOPEN would require a mechanism that explains why `data.csv` regressed and `dd79_bin6` did
not move — not another (depth, nice, good_match) triple. Do not sweep the triple; the failure
is not a bad point in that space, it is that the space itself trades one input class against
another.
## G14 — 77% OF THE FAILING BOARD IS THE T>1 PATH, NOT THE PARSE (2026-07-31)

The single most useful cut of the banked board, and it was available all along. Failing size
cells split by THREAD COUNT (`/root/sizeboard-all-12fcd0ed/census.json`, 1,320 measured):

    band     T4    T1
    L8-L9    35     2
    L5-L7    61    15
    L2-L4    41    11
    L1       17    18
    TOTAL   154    46

**154 of 200 failing cells are T4.** At T1 our output is BYTE-IDENTICAL to libdeflate on
every file tested at L6 and L9 — symbols.dwarf 366,624; data.csv 3,300,291; dickens
4,480,689; sil40 15,452,666 — i.e. we TIE exactly and pass. The loss appears only when the
input is cut into chunks:

    file            L   T1           T4           seam cost
    symbols.dwarf   9     366,624      366,918      +294
    data.csv        9   3,300,291    3,301,579    +1,288
    sil40           9  15,452,666   15,453,781    +1,115
    dickens         9   4,480,689    4,480,950      +261

And the wall board already showed we beat libdeflate **2-3x at T4** — so there is large wall
headroom exactly where the size cells fail. That is the reverse of the T1 situation, where
wall is the binding constraint.

### The waste is named, and it is not "too many chunks"

`CLAUDE.md` STEP 2 sanctions "making seams SMALLER — pad choice, chunk grid, block
splitting", and G5 already states the mechanism: **every seam costs a sync-flush AND a
block-grid restart AND a fresh Huffman histogram. The first is 5 bytes; the other two are the
expensive ones.**

Halving the chunk count (`CHUNKS_PER_THREAD` 2 -> 1) is only a PARAMETER against that waste.
Measured at L8-L9 over 264 common cells: **8 closed, 3 opened** (net -5, still a clause-3
failure). It recovers 65-80% of the per-file seam cost (data.csv L9 +1,288 -> +533; sil40
+1,115 -> +230; dickens +261 -> +25) by having fewer restarts — not by making a restart
cheaper. It also spends the load-balancing margin that `CHUNKS_PER_THREAD` exists for.

**The structural fix is to make the seam FREE, not rarer**: carry the coding state across the
chunk boundary so a seam costs the 5-byte sync-flush and nothing else. That is G5, it is
sanctioned by STEP 2, and it addresses 154 cells rather than the 46 the entire rest of this
session was aimed at.

### Method note

This session spent ~11 levers on T1 parse mechanisms — matchfinder register pressure, hash
geometry, cost model, block cadence — and closed zero cells. Ten were falsified. The board
had the answer in a one-line group-by the whole time. **Before choosing a mechanism, cut the
failing set by every axis you have (level, rival, thread count, file class) and work the
largest block.** Thread count was the axis nobody had cut by.

### The seam cost is MISALIGNMENT, not per-seam overhead — vendor-anchored

pigz states its own seam cost in `vendor/pigz/pigz.c:233-240`: each chunk ends with a
Z_SYNC_FLUSH empty stored block, "a very small four to five byte overhead (average **3.75
bytes**) to the output for each input chunk", with the previous 32K supplied as a preset
dictionary (`:248-252`). Its default block is **128 KiB**.

Ours, measured at L9 (exact bytes, `-p1` baseline, local M1 vanilla build):

    file        T2 /seam   T4 /seam   T8 /seam      pigz
    sil40          76.7      159.3      107.7       3.75
    dickens         8.3       37.3       35.8       3.75
    data.csv      177.7      184.0       30.1       3.75

**The cost is NON-MONOTONIC in seam count**: data.csv totals +1,288 B at T4 (7 seams) but
only +452 B at T8 (15 seams). Twice the seams, a third of the cost. A fixed per-seam
overhead cannot do that.

So the waste is NOT "we restart the coder N times". It is WHERE THE CUT LANDS. pigz pays
3.75 bytes because its seam coincides with a 128 KiB block boundary it was going to emit
anyway — the seam is free by construction, and only the flush is charged. Our chunk boundary
falls at an arbitrary offset relative to the block grid, so a chunk can end in a runt block
whose header is not amortised, and the penalty is whatever that misalignment happens to
break.

`pipelined_block_size` already floors the chunk span to a multiple of `SOFT_MAX_BLOCK_LENGTH`
— but the drift detector ends blocks EARLY and data-dependently, so actual block boundaries
are not at multiples of the budget and the alignment is nominal only.

THE STRUCTURAL FIX FOLLOWS, and it is pigz's: make the chunk boundary COINCIDE with a block
boundary the parser was going to emit anyway, so the seam costs only the flush. That is
different from CHUNKS_PER_THREAD (fewer seams, each still misaligned) and different from G5
(carry coding state across the seam). It is the cheapest of the three and it is the one the
vendor actually ships. NOT YET BUILT.

### THE SEAM COST IS FULLY ACCOUNTED: 3 EXTRA BLOCK HEADERS, NOT 8 CODER RESTARTS

`anatomy-counters`, sil40 at L9, same binary, T1 vs T4:

    T1   blocks_dynamic 913   blocks_stored 0   make_code_calls 2744
    T4   blocks_dynamic 916   blocks_stored 8   make_code_calls 2753

    +3 dynamic block headers  (~350 B each)  ~ 1,050 B
    +8 sync-flush stored blocks (~5 B each)  =    40 B
                                             ~ 1,090 B
    measured T4-T1 size delta                  +1,115 B

The accounting closes. And it corrects the intuition: the cost is NOT eight coder restarts.
Seven of the eight chunk boundaries cost essentially nothing — `pipelined_block_size` already
floors the chunk span to a multiple of `SOFT_MAX_BLOCK_LENGTH`, and it mostly works. What
costs is that **three chunks round UP to one extra block**, and an extra DYNAMIC header is
~350 bytes.

That also explains the non-monotonicity (data.csv +1,288 B at T4 with 7 seams but only +452 B
at T8 with 15): the penalty tracks how many chunks happen to round up, not how many seams
exist. More seams can round up less often.

CONSEQUENCES for the three candidate fixes:
* `CHUNKS_PER_THREAD` 2->1 — helps only by making fewer opportunities to round up. Measured
  8 closed / 3 opened at L8-L9. A parameter, not a fix.
* G5 "carry coding state across the seam" — would remove the 40 B of sync-flushes and the
  restart, but the restarts are NOT where the 1,115 B is.
* **ALIGNMENT — the actual target.** Make each chunk's span an exact whole number of blocks
  so no chunk rounds up. Upper bound on the win is ~1,050 of the ~1,115 B on this cell, and
  it costs nothing at the wall.

The obstacle is named and real: the drift detector ends blocks EARLY and data-dependently, so
the true block grid is not at multiples of the budget and cannot be predicted from the chunk
span alone. Any alignment scheme has to reckon with a block boundary the parser chooses, not
one the scheduler assigns. NOT YET BUILT.

### FALSIFIED — the extra blocks are NOT runts, and a tail guard makes it worse

If the +3 dynamic headers at T4 were runt blocks created by the drift detector splitting too
close to a chunk end, refusing to split there would recover them. Ablated by raising the
`bytes_remaining` threshold in `ready_to_check_block` (libdeflate's own value is
MIN_BLOCK_LENGTH = 5,000), L9, `-p4`, exact bytes:

    file        T1           guard 5,000   32,768       131,072      262,144
    sil40       15,452,666   15,453,781    15,454,738   15,461,035   15,466,720
    data.csv     3,300,291    3,301,579     3,301,587    3,302,154    3,302,080
    dickens      4,480,689    4,480,950     4,480,950    4,480,947    4,481,224

MONOTONICALLY WORSE. Those late splits EARN their headers — the data changed and a fresh
table pays. The extra blocks are legitimate, not waste.

So the +3 is genuinely the boundary MISMATCH: with 8 chunks there are 7 forced block
boundaries, 4-5 of which happen to coincide with where the parser would have ended a block
anyway, and 3 of which do not. Where the parser wants a boundary is DATA-DEPENDENT and not
knowable when the scheduler assigns the grid.

THAT is why the only remaining alignment fix is the one the review proposed: **let the PARSER
choose the boundary and the scheduler follow it**, rather than the scheduler assigning a grid
the parser must honour. Chunk boundaries become scheduling artifacts, not coding boundaries.
Every cheaper approach is now measured and dead:
    CHUNKS_PER_THREAD 2->1   8 closed / 3 opened at L8-L9 — a parameter, still misaligned
    tail guard               monotonically worse (above)
    G5 carry coding state    removes 40 B of flushes; the 1,115 B is not the restarts

## G15 — THE CAGE IS MEASURED: 77.8% of our T1 cells vs libdeflate are EXACT BYTE TIES

Source: `/root/sizeboard-all-12fcd0ed/census.json` (solvency, frozen, 22 corpus files x L1-9 x
{gzip,pigz,libdeflate,igzip} x T1/T4, 1,320 OK cells, 0 VOID).

    T1 size vs each rival           n    EXACT TIE      we smaller   we bigger
    gzip                          198     0 ( 0.0%)        182           16
    pigz                          198     0 ( 0.0%)        188           10
    igzip                          66     0 ( 0.0%)         63            3
    libdeflate                    198   154 (77.8%)         27           17

    libdeflate, per level (22 files each):
      L1  tie  0   smaller  7   bigger 15     <- genuine deficit
      L2  tie 22   smaller  0   bigger  0
      L3  tie  0   smaller 20   bigger  2     <- the ONE level we diverged
      L4  tie 22 | L5 tie 22 | L6 tie 22 | L7 tie 22 | L8 tie 22 | L9 tie 22

`level.rs::params_inner` transliterates libdeflate's preset table, so at L2 and L4-L9 we run
their strategy at their `max_search_depth` and their `nice_match_length` and emit their exact
bytes on every file. L3 is the only deep level we changed (Lazy where they run Greedy at the
same knobs) and it is the only one holding a margin: smaller on 20/22, median 44 KB.

### Why this is the whole T4 story

A tie is not a win — it is zero headroom. Of the 200 failing cells:

    109  T4 fails / T1 passes vs libdeflate   deficit == T4-T1 growth EXACTLY, on all 109,
                                              with T1 headroom = 0 bytes on all 109
     29  L1 vs libdeflate                     genuine T1 deficit (15 T1 + 14 T4)
     58  gzip/pigz/igzip                      genuine T1 deficit (T1 count == T4 count)
      4  L3 vs libdeflate

The 109 do not need a big size win. Their deficits are min 2 / median 255 / max 2,093 bytes —
on sil40 that is 0.007%. ANY margin above ~0.01% closes all 109 at once.

### The frontier is measured, and the parse-strategy route does not reach it

`ladder-tune` (this PR) makes the level->params map overridable so the frontier can be
observed rather than inherited. Validated three ways before use: L4 forced to `lazy:12:14`
is byte-identical to real L3; forcing a level to its own values is a no-op; unset is inert.

LAW FOUND (holds at every level tested): at a FIXED depth, the stronger parse strategy is
strictly smaller. libdeflate spends depth where it should have spent a defer.

    L2  their Greedy(6,10)     lazy:4:10     smaller on 4/4 at 2/3 their depth
    L4  their Greedy(16,30)    lazy:8:30     smaller on 4/4 at HALF their depth
    L5  their Lazy(16,30)      lazy2:16:30   smaller on 4/4 at equal depth
    L6  their Lazy(35,65)      lazy2:35:65   smaller on 4/4 at equal depth

BUT THE WALL DOES NOT PAY FOR IT. Wall slack vs libdeflate, T1, sil40, VANILLA build
(hyperfine, 5 runs, both arms to /dev/null):

    L2 ratio 1.044 WE LOSE | L4 0.998 tie | L5 1.038 WE LOSE | L6 0.956 +4.4% | L7 0.922 +7.8%

    L4  lazy:8:30    +7.7% wall  (half depth still costs MORE than greedy at full depth --
                                  lazy's overhead is per-POSITION, not per-probe)
    L6  lazy2:35:65  +9.29% wall vs 4.4% slack   EXCEEDS
    L6  lazy2:24:65  free, but winexe.exe -0.183%  FAILS SIZE
    L7  lazy2:100:130 +11.02% vs 7.8% slack      EXCEEDS
    L7  lazy2:70:130  +1.96%, but winexe.exe -0.103%  FAILS SIZE

The binding file is a BINARY (winexe.exe), and the depth response is monotone with no
crossing inside the budget:

    L6 winexe vs libdeflate-6 by lazy2 depth:
      24 -0.183%   28 -0.141%   30 -0.078%   32 -0.022%   35 +0.002%
      cost at depth 24: free           cost at depth 35: +9.29%      slack: 4.4%

Trading `nice_match_length` down to buy the time back is worse still: at L6, nice 50 and 40
put data.json at -0.188% and -2.505%.

FALSIFIED: the parse-strategy upgrade cannot fund the margin at L2/L4/L5/L6/L7. Every
configuration that is smaller on all files costs more wall than we have.

### What this implies — the margin must be ENCODER-SIDE

The margin needed is ~0.01%, it is needed at SEVEN levels at once, and it must cost
essentially zero wall. The parse is the wrong place to buy it: every parse improvement scales
its cost with search effort. An ENCODING improvement (block-boundary choice, length-limited
Huffman construction, code-length RLE) is paid once per block, not once per position, and
would move all seven tied levels simultaneously -- INCLUDING L8/L9, where no stronger parse
strategy exists below NearOptimal and the parse route is therefore not merely too expensive
but unavailable.

CAVEAT ON THESE MARGINS: measured on the local M1 with the local libdeflate build, where our
BASE is within +-0.05% of libdeflate rather than exactly equal. The frontier SHAPE (strategy
dominance at fixed depth, depth monotonicity, the sign of the wall costs) is what this
section claims. Any absolute margin must be re-measured on solvency against the frozen rival
before it gates anything.

## G16 — the margin is NOT in Huffman construction: libdeflate's heuristic is within 0.001% of exact

G15 concluded the ~0.01% of margin the 109 zero-headroom cells need must be ENCODER-side,
because a parse improvement scales its cost per position while Huffman construction is paid
once per block. That was the right cost shape and the wrong place. Both forms were built.

UNCONDITIONAL SWAP (`fast.rs` heuristic -> `optimal.rs` exact Katajainen package-merge):
a wash, and it OPENS cells.

    dickens L2 +10 B | data.csv L2 -326 B | winexe.exe L6 +385 B | data.json L6 -737 B

winexe.exe L6 goes from 178 B smaller than libdeflate to 207 B larger. Package-merge
minimises CODED-DATA bits, but a dynamic block also transmits its length vector in an
RLE-coded header and the two builders produce different vectors: EXACT ON DATA IS NOT
EXACT ON TOTAL.

COSTED DUAL CANDIDATE (build both codes, cost header+data for each, emit the cheaper;
strict `<` so ties keep today's bytes). The size invariant held exactly as designed —
**49 of 49 cells smaller, 0 worse**, T1, every one roundtripping through our decoder,
gzip and libdeflate:

    dickens -16..-33 | data.csv -7..-16 | winexe -5..-18
    aozora  -21..-66 | markup   -2..-7  | sil40  -122..-166

But the margin is ~0.001% and the wall cost is 10-14% (two extra Huffman builds, one
extra header build, one extra cost evaluation per block). Against libdeflate, sil40, T1:

    L2 1.044 -> 1.193 | L5 1.038 -> 1.150 | L6 0.956 -> 1.043 | L7 0.922 -> 0.986

L6 FLIPS from WE WIN to WE LOSE. Clause 3 forbids a pass->fail flip, so this is reverted.

THE FAMILY THIS CLOSES. The interesting number is not the wall cost, it is the 0.001%:
libdeflate's heuristic length limiter is ALREADY within one part in 100,000 of the exact
minimum-redundancy code. No amount of speeding up package-merge changes that ceiling. The
whole "our Huffman leaves size on the table" family is dead — the margin is not in code
CONSTRUCTION. What remains encoder-side is block BOUNDARY choice (where the tail-guard
probe showed our splits are already earning their headers, but says nothing about whether
a DIFFERENT boundary set is better) and the parse itself.

## G17 — every lever measured today SLIDES ALONG libdeflate's frontier; none MOVES it

Collecting the day's measurements in one table. All T1, sil40 unless noted, vanilla builds,
size deterministic, wall by hyperfine n=5 to /dev/null:

    lever                                   size vs libdeflate     wall vs libdeflate
    ------------------------------------    -------------------    ------------------
    ship their config (L2, L4-L9)           EXACT TIE (0 bytes)    1.04 / 1.00 / 1.04 (L2/L4/L5)
    L3 Lazy(12,14) vs their Greedy(12,14)   +1.331% SMALLER        1.215  WE LOSE
    L4 lazy:8:30 (HALF their depth)         smaller on 4/4         +7.7%  vs ~0% slack
    L6 lazy2:35:65 (equal depth)            smaller on 4/4         +9.29% vs 4.4% slack
    L6 lazy2:24:65 (fits the wall)          winexe.exe -0.183%     free
    L7 lazy2:100:130 (equal depth)          smaller on 4/4         +11.02% vs 7.8% slack
    exact package-merge Huffman             49/49 smaller, 0 worse +10-14%, flips L6

EVERY ONE IS A TRADE. Not one of them is smaller AND cheaper. We are sitting ON libdeflate's
size/wall frontier: where we copy their config we reproduce their size exactly and pay a few
percent more wall (we run their algorithm slower than they do); where we diverge we buy size
with wall, at L3 at a rate of 21.5% wall for 1.33% size.

THE PER-LABEL BAR CANNOT BE MET BY SLIDING. At level N we must be <= their level N on BOTH
axes. Any configuration change moves us along the frontier, improving one axis by spending
the other. That is why 11 of the last 13 levers failed on the axis they did not target, and
it is a structural property of the search, not bad luck in picking knobs.

WHAT MOVES THE FRONTIER is executing the SAME parse for fewer instructions — which is
exactly what the banked profile already says the deficit is:
`project_encoder_deficit_is_loads_not_stalls` — on the worst cell our IPC, stalls, cache and
branch behaviour ALL BEAT libdeflate, and 61% of the gap is extra LOAD INSTRUCTIONS issued.

That reorders the campaign. The lever is NOT "find a better level->config map" — today
measured that space and every point in it is a trade. The lever is the per-position load
count in the matchfinder. Close that and the wall slack appears at every level at once; a
parse upgrade then becomes affordable, and we already know what it is worth (lazy2 at equal
depth: +0.4% to +1.4% size, i.e. 40-140x the ~0.01% the 109 zero-headroom cells need).

Order of work implied: (1) cut loads per position in `hc`, (2) re-measure slack, (3) spend
the recovered slack on the parse upgrade the sweep already priced.

## G18 — our search is a PERFECT clone of libdeflate's, probe-for-probe, at every level

The frontier result (G17) said the only way off libdeflate's size/wall curve is running the
SAME parse for fewer instructions, and the banked profile
(`project_encoder_deficit_is_loads_not_stalls`) says 61% of the L2 wall gap is extra LOAD
instructions: 991,837,377 vs 761,047,000 L1-dcache-loads, a ratio of 1.3033, while our IPC,
stalls, L1D miss rate and branch behaviour all BEAT theirs.

That leaves exactly two possibilities: we visit MORE chain nodes, or we pay MORE PER NODE.
Ablated by instrumenting BOTH implementations to count the identical quantity — every
chain-node evaluation in the hash-chain matchfinder.

Method: `hc_probe_attempts` (already in `anatomy_counters`) vs a `g_hc_probes++` added to
libdeflate's `hc_matchfinder_longest_match` at both sites that materialise `matchptr` from
`cur_node4` (`lib/hc_matchfinder.h`, the `best_len < 4` loop and the `>= 5` loop). The
instrumented libdeflate emits byte-identical output at every level tested, so the counter is
behaviour-neutral. Deterministic counts, local M1, `dickens`, T1 — no wall run needed.

    level   gzippy hc_probe_attempts   libdeflate g_hc_probes   ratio
    L2              8,163,515                 8,163,515        1.0000
    L4             14,940,322                14,940,322        1.0000
    L6             34,629,888                34,629,888        1.0000
    L9            100,451,406               100,451,406        1.0000

EXACT EQUALITY, to the last digit, at a SHALLOW and a DEEP level (hard stop #3). Our
matchfinder visits precisely the same chain nodes in precisely the same order as theirs.

### The family this closes

Every hypothesis of the form "our SEARCH does more work" is dead: hash geometry, hash3
singleton-vs-chained, chain quality, cutoff handling, depth accounting, nice-length
early-exit, insert policy. None of them can be the deficit, because none of them changed the
node count and the node count is identical.

Combined with the banked profile, the deficit is now pinned to a single quantity: LOADS
ISSUED PER NODE (and per position), with the node count held fixed. That is a code-generation
question, not an algorithm question — the candidates are Rust slice fat-pointers and their
length reloads versus C raw pointers, redundant re-materialisation of `in_base`/`cutoff`
across the loop, and the `i16` table loads' sign-extension shape.

NEXT MEASUREMENT (not yet run): cachegrind/callgrind on trainer (the only box with valgrind)
attributing Dr BY LINE inside `hc.rs::longest_match`, against the same region of libdeflate
built with `-g`. Probe count is now excluded as a variable, so any line-level Dr difference
is the deficit itself. Do NOT hand-hoist "obviously redundant" loads first — hard stop #4
records that doing so drove data reads UP, because LLVM had already hoisted them and the
hoist only added register pressure.

## G19 — the ENTIRE data-read deficit lives inside `hc`, which runs FEWER instructions than libdeflate

G18 proved we visit exactly the same chain nodes as libdeflate at every level. This attributes
the remaining gap by line, with probe count excluded as a variable.

PROVENANCE: trainer (lxc199, Intel i7-13700T, x86_64 — the only box with valgrind), valgrind
3.24 cachegrind `--cache-sim=yes`, L2, T1, identical input `/root/cg_text8` (8,000,000 B, head
of `/root/frontier-corpora/text`). Ours built `CARGO_PROFILE_RELEASE_DEBUG=2
CARGO_PROFILE_RELEASE_STRIP=false` (release profile otherwise untouched — lto=fat,
codegen-units=1, opt-level=3; debug info does not change codegen). libdeflate built
RelWithDebInfo `-O2 -g`. Counts are deterministic. Artifacts `/root/cg.ours2`, `/root/cg.ld`.

WHOLE PROGRAM:

                    gzippy         libdeflate      ratio
    Ir         562,981,991        500,108,050      1.126
    Dr         116,446,149         97,101,434      1.199
    Dw          62,999,584         42,787,926      1.472

Dw 1.472 is the WORST ratio and was never named before — the banked profile
(`project_encoder_deficit_is_loads_not_stalls`) measured loads only.

THE MATCHFINDER REGION, head to head:

                             Ir             Dr             Dw
    ours    hc.rs      251,331,366     56,872,118     35,498,696
    theirs  hc_mf.h    254,868,099     30,664,002     25,363,595
    ratio                     0.986          1.855          1.400

WE EXECUTE FEWER INSTRUCTIONS AND NEARLY DOUBLE THE READS, for the same probes. The +26.2M
read excess inside `hc` is LARGER than the whole program's +19.3M, so we are net BETTER than
libdeflate everywhere else (parse + emit combined). The deficit is not spread; it is one
function.

Same algorithm, same node count, fewer instructions, 1.855x the reads is the signature of
values being RE-MATERIALISED FROM MEMORY instead of held in registers — register pressure /
spilling in the hot loop, not algorithm and not instruction selection. It is consistent with
every microarchitectural fact already banked (our IPC, stalls, L1D miss rate and branch
behaviour all BEAT theirs: we issue more memory ops and absorb them well).

### A false lever caught by the vendor diff, recorded so it is not re-found

The top single Dr line inside `hc.rs` is 6,274,515 Dr (5.4% of all program reads):

    *self.next_tab.get_unchecked_mut(cur_pos) = *self.hash4_tab.get_unchecked(hash4);

in the bulk-insert path. It LOOKS like a redundant reload that a `cur_node4` local would kill.
It is not: libdeflate's `hc_matchfinder_skip_bytes` does character-for-character the same
thing —

    mf->hash3_tab[hash3] = cur_pos;
    mf->next_tab[cur_pos] = mf->hash4_tab[hash4];
    mf->hash4_tab[hash4] = cur_pos;

— so that read is matched by theirs and is not the difference. Diffing the vendor BEFORE
proposing is what caught it.

NEXT: the target is Dr inside `hc.rs::longest_match` with node count held fixed. Per hard
stop #4, do NOT hand-hoist loads — that drove data reads UP last time because LLVM had
already hoisted them and the hoist only added register pressure. The measurement to run is
the per-line Dr diff of our loop against theirs, and the candidate class is what keeps values
live across the chain walk (`&mut` reference parameters that must be re-read after any
aliasing store, slice fat-pointer lengths, and the number of simultaneously-live locals).

## G20 — the reads have NO SOURCE LINE: 24.1M Dr of compiler-generated traffic, against libdeflate's ZERO

Continuing G19's attribution down to the line, same artifacts (`/root/cg.ours2`, `/root/cg.ld`,
trainer, L2, T1, identical 8,000,000 B input).

FIRST, a port confirmation. libdeflate's bulk-insert line and ours cost the SAME to the byte:

    ours    *self.next_tab.get_unchecked_mut(cur_pos) = *self.hash4_tab...   6,274,515 Dr
    theirs  mf->next_tab[cur_pos] = mf->hash4_tab[hash4];                    6,274,515 Dr

Identical. Our hot lines that DO have source attribution match theirs. The gap is elsewhere.

    profile        unknown-line (line 0) Dr entries
    libdeflate     NONE. Every Dr in their profile attributes to a source line.
    gzippy         24,099,848 (20.7% of ALL program reads) in hc.rs alone,
                   plus ~3.3M more across common.rs / greedy.rs / parse/mod.rs

Our whole-program Dr excess over libdeflate is 19,344,715, and the excess inside `hc` is
26,208,116. The unattributed 24.1M accounts for essentially all of it. Code carrying no source
line is compiler-generated — register spill/reload and moves.

MACHINE-LEVEL CORROBORATION (objdump, same binaries):

                              ours (greedy::run_resumable)   theirs (deflate_compress_greedy)
    static instructions                987                              675
    insns with memory operand          345                              268
      of which stack-relative          263  (76%)                       128  (48%)
    stack frame allocated            0x7b8 = 1,976 B                  0xb8 = 184 B   (10.7x)

HONEST LIMIT OF THIS EVIDENCE. The frame is allocated once per CALL, not per position, so
frame size alone does not prove the inner loop spills — it proves the function carries far
more live state than libdeflate's equivalent. The load-bearing facts are the 24.1M
unattributed Dr (dynamic, and matched by ZERO on their side) and the 2.05x stack-operand
ratio (static). The CONFIRMING measurement, not yet run, is cachegrind `--dump-instr=yes` to
get per-INSTRUCTION Dr inside the chain walk, which shows directly whether the executing
spill traffic is in the loop body or in per-block prologue.

WHY THIS IS NOT hard-stop-#4 TERRITORY. Hard stop #4 forbids hand-hoisting "obviously
redundant" loop-invariant loads, because LLVM had already hoisted them and the hoist only
added register pressure. This is the OPPOSITE direction: the finding is that we HAVE too much
register pressure, and the lever is to REDUCE what is live across the chain walk, not to hoist
more into it. A candidate worth measuring first: `emit_block` copies `sink.litlen_freqs`
(`[u32; 288]` = 1,152 B) by value, and if that inlines into the same frame it is most of the
1,976 B on its own.

STILL FORBIDDEN: proposing any of this as a win without measuring Ir/Dr at a SHALLOW and a
DEEP level (hard stop #3) and confirming on the wall (instruction counts LOCATE, they never
predict the wall — receipts: a change that cut Ir 1.77% and Dr 3.87% was 9.9% SLOWER at L9).

## G21 — CORRECTION to G19/G20: removing a QUARTER of the reads made the wall WORSE

G19/G20 located the deficit inside `hc` and named spill traffic as the mechanism. That
localisation stands. The implied conclusion — that reads are the wall-blocking quantity —
does NOT. It was tested and it failed.

TEST: `#[inline(never)]` on `hc::longest_match`, which gives the chain walk its own small
frame instead of sharing `greedy::run_resumable`'s 1,976-byte one. Output byte-IDENTICAL at
L2/L6/L9 (sha256), so this is a clean A/B on the same bytes.

    level          Ir                      Dr                         Dw
    L2 (8 MB)  562,981,991 -> 577,918,803  116,446,149 -> 118,093,512  +12.6%
    L9 (2 MB)  449,743,993 -> 450,076,251  106,693,133 ->  81,059,242  +27.9%
                        (+0.07%)                    (-24.0%)

    WALL (hyperfine n=7, 58 MB text, T1, both arms to /dev/null, trainer):
    L2 1.0078s -> 1.0639s (1.0557 SLOWER)
    L6 2.0238s -> 2.0910s (1.0332 SLOWER)
    L9 5.5427s -> 5.6406s (1.0177 SLOWER)

AT L9 WE DELETED 25,633,891 DATA READS, HELD Ir FLAT, AND LOST 1.77% OF WALL.

Why: Dw rose 27.9% (arguments spilled to make the call) and every position gained call/return
dependent latency. The spill READS are largely absorbed by the machine — which is exactly what
the banked profile already implied, since our IPC, stalls, L1D miss rate and branch behaviour
all BEAT libdeflate's. A machine that absorbs memory traffic well does not pay full price for
extra loads, and does pay for added latency and stores.

WHAT SURVIVES: `hc` is where the excess lives (G19 — same probes, fewer instructions, 1.855x
reads, and an excess larger than the whole program's). What is NOT established, and was
briefly asserted here, is that lowering that read count lowers the wall.

WHAT IS ALSO NOW ON RECORD: measuring only L2 would have rejected this for the wrong reason
(it looks bad on Ir AND Dr there), and measuring only L9 on counters alone would have promoted
a 1.77% wall REGRESSION as a 24% win. Hard stop #3 (shallow AND deep) and the
counters-never-predict-the-wall rule each caught a different half of the same mistake.

REOPEN requires a mechanism that lowers reads WITHOUT adding a call per position and WITHOUT
raising Dw — fewer simultaneously-live values in the caller is the candidate class, not
relocating the callee — plus a wall measurement at a shallow AND a deep level.

## G22 — the matchfinder is at PARITY; the instruction excess is the PARSE/EMIT path, and block splitting costs 14x

Full Ir decomposition by ROLE from the same two profiles (trainer, L2, T1, identical
8,000,000 B input; `/root/cg.ours2`, `/root/cg.ld`). Files summed to 99.7%/99.9% of each
program's total, so nothing material is hidden below the cut.

    role                          gzippy         libdeflate      ratio
    matchfinder search+extend   349,349,702    346,344,561       1.009   <- PARITY
    crc32                         2,625,024      2,250,004       1.17    (0.4% of program)
    parse / emit / block split  207,177,313    151,190,672       1.370   <- THE EXCESS

      gzippy side: parse/mod.rs 82,036,989 + greedy.rs 22,841,792 + bitstream.rs 22,775,726
        + block_split.rs 21,816,711 + uint_macros.rs 20,566,683 + tables.rs 19,082,899
        + const_ptr.rs 10,945,358 + slice/iter/macros.rs 3,593,348 + non_null 1,358,588
        + range 1,357,998 + huffman/fast.rs 794,248
      libdeflate side: deflate_compress.c 149,682,890 + common_defs.h(flush) 1,507,782

+55,986,641 instructions, which is 89% of the whole-program excess of +62,873,941.

THIS INVERTS THE CAMPAIGN'S TARGET. The banked note
(`project_encoder_deficit_is_loads_not_stalls`) said "hc.rs is 249.4M Ir, 44.9% of the program
at L2 ... the entire excess is parse+matchfinder". Measured directly against the vendor, the
MATCHFINDER IS AT PARITY (1.009) — it already runs slightly FEWER instructions than libdeflate's
(G19) on identical probes (G18). Every remaining instruction of the gap is in the code AROUND
the search.

### The single largest component: block splitting, 14x

    ours    block_split.rs                                  21,816,711 Ir
    theirs  do_end_block_check 721,264 + calculate_min_match_len 790,842 = 1,512,106 Ir
                                                                          ratio 14.4x

+20,304,605 instructions — a THIRD of the entire whole-program excess — in the component that
decides where blocks end. libdeflate checks for a block boundary cheaply and rarely; we run
per-position observation bookkeeping.

Two supporting oddities on our side with no vendor counterpart at all:
`uint_macros.rs` 20,566,683 Ir and `tables.rs` 19,082,899 Ir (39.6M combined, 7.0% of the
program) — integer-helper and lookup-table code that libdeflate does not spend separately
because the equivalent work is folded into `deflate_compress.c`.

### Why this is the RIGHT target and the matchfinder was the wrong one

G21 proved reads are not the wall-blocking quantity (deleting 25.6M of them at L9 LOST 1.77%
of wall). Instructions remain the live candidate — Ir 1.126 whole-program against a wall ratio
of roughly the same order. And the instruction excess is NOT in the matchfinder, which is where
every previous session looked.

CAVEAT, stated because G21 exists: this locates instructions, and instruction counts LOCATE,
they never predict the wall. Nothing here is a promised win. The next step is to diff
`block_split.rs` against `deflate_should_end_block`/`do_end_block_check` as ALGORITHMS — how
often each runs and what it computes per call — and then measure Ir AND wall at a shallow and
a deep level before believing anything.

## G22a — RETRACTION of the "block splitting is 14x" line in G22

WRONG, AND MINE. G22 compared our `block_split.rs` (21,816,711 Ir) against libdeflate's
`do_end_block_check` (721,264) + `calculate_min_match_len` (790,842) and called it 14.4x.
That is not like-for-like.

libdeflate's PER-POSITION split cost is `observe_literal`/`observe_match`
(`deflate_compress.c:2110,2121`), both `forceinline` and called from `deflate_compress_greedy`
(`:3695-3700`). Their cost therefore lands inside `deflate_compress.c:deflate_compress_greedy`
(84,045,480 Ir), NOT inside `do_end_block_check`, which is only the rare amortised check. Our
21,816,711 in `block_split.rs` is the per-position observe PLUS the check, attributed to
`greedy::run_resumable`. I compared our observe+check against their check alone.

There is no measured 14x. The component may still be worse than theirs; this profile cannot
say, because their observe cost is not separable from their parse loop.

### What survives, restated honestly

Role-level totals are unaffected (files sum to 99.7%/99.9% of program Ir):

    role                          gzippy         libdeflate      ratio
    matchfinder search+extend   349,349,702    346,344,561       1.009   PARITY
    parse / emit / block split  207,177,313    151,190,672       1.370   THE EXCESS

Splitting that second row by what the code DOES, using the one boundary the profiles agree
on (per-position loop body vs per-block flush):

    emit / flush                                 gzippy        libdeflate     ratio
      ours: parse/mod.rs::emit_sequences 34,667,354 + bitstream.rs 22,775,726
            + huffman/fast.rs 794,248            58,237,328
      theirs: deflate_flush_block 62,772,475 + common_defs.h(flush) 1,507,782
                                                             64,280,257      0.906  WE WIN

    per-position parse loop body (all remaining non-matchfinder)
      ours: parse/mod.rs(run_resumable) 47,256,132 + greedy.rs 22,841,792
            + block_split.rs 21,816,711 + uint_macros.rs 20,566,683
            + tables.rs 19,082,899 + const_ptr.rs 10,945,358 + slice/iter/macros.rs 3,593,348
            + non_null 1,358,588 + range 1,357,998           148,819,509
      theirs: deflate_compress.c::deflate_compress_greedy      84,045,480    1.771

CAVEAT ON THAT 1.771: the ours-side grouping puts `const_ptr.rs`, `non_null.rs` and
`slice/iter/macros.rs` (15,897,294 combined) in the parse row, but cachegrind attributes them
only to `run_resumable`, which also contains the matchfinder. Some of that belongs in the
matchfinder row. The ratio is therefore an UPPER bound; excluding all three gives
132,922,215 / 84,045,480 = 1.582. The conclusion is the same at either end: the per-position
parse loop body costs 1.6-1.8x libdeflate's, our emit path is CHEAPER than theirs (0.906), and
the matchfinder is at parity.

So the target is the non-matchfinder per-position work — sequence storage, the observe, the
slot-table lookups — not the search, not the emit, and not (on this evidence) block splitting
in particular.

## G23 — DEFECT: `-b/--blocksize` is advertised, parsed, validated, and then silently ignored

Found while testing whether the T>1 chunk grid could be widened to shrink the seam.

    ours:  gzippy -6 -p4 -b {1024, 65536, 1048576, 8388608} -c dickens
           -> ALL FOUR produce the IDENTICAL output sha256 (618d2590842e772e...)
    pigz:  pigz  -6 -p4 -b 32   -c dickens -> 896cf24edf00dac8...  4,563,394 B
           pigz  -6 -p4 -b 1024 -c dickens -> b8e991e0ae48e6a5...  4,550,145 B

pigz's `-b` changes the block grid and therefore the output (13,249 B apart here, larger
blocks being smaller). Ours changes nothing at all, across a 8192x range of values.

The flag is REAL up to the last step: `src/main.rs:645` advertises it in `--help`,
`src/cli.rs:9,66` defines it with a 128 KiB default, `src/cli.rs:235,316,451` parse all three
spellings, and `src/cli.rs:508` rejects values below 1024. Then the compress path never reads
it — `src/compress/pipelined.rs:200` computes the grid as
`pipelined_block_size(input_len, num_threads, _level)`, a function whose parameters do not
include the user's value, and `:495,:587,:672` are its only callers.

This violates CLAUDE.md non-negotiable #4 (least surprise, cite the contract): pigz's CLI is
the contract for `-b`, we advertise the same flag, and we honour it nowhere. A user tuning
`-b` gets silence rather than an effect or an error.

It is also a CAMPAIGN lever we currently cannot reach. The 109 zero-headroom cells are T4-only
and caused by forced chunk boundaries; the grid is exactly the thing `-b` should control, and
pigz's own numbers show the grid moves output size by far more than the ~255 B median deficit.
Wiring it does NOT by itself close a cell (the board runs default flags), but it makes the
grid measurable instead of hard-coded.

NOT YET FIXED. Wiring it changes T>1 output whenever `-b` is passed, which is the correct
behaviour but must be gated: roundtrip through our decoder + gzip + libdeflate at several
`-b` values and thread counts, and a check that the DEFAULT path stays byte-identical.

## G24 — the seam CANNOT be closed by grid tuning: the residual is always > 0

Measurable for the first time because #223 made `-b` actually reach the chunk grid. sil40
(40,000,000 B), L9, T4, local M1, vanilla build; size deterministic, wall by hyperfine n=5
warmup 1, both arms to /dev/null.

    grid                       output bytes   vs T1     wall       vs T4-default
    T1 (reference)             15,452,666       ---     1.1357s        3.557x
    T4 default (8 chunks)      15,453,781    +1,115     0.3193s        1.000x
    T4 -b 10M  (4 chunks)      15,452,728       +62     0.3495s        1.095x
    T4 -b 16M  (3 chunks)      15,452,697       +31     0.5963s        1.868x
    T4 -b 64M  (1 chunk)       15,452,672        +6     1.1660s        3.652x

ONE CHUNK PER THREAD REMOVES 94.4% OF THE SEAM FOR 9.5% WALL (+1,115 -> +62 B, 0.3193 ->
0.3495 s). That is a far better exchange rate than anything the parse-config sweep offered.

AND IT STILL DOES NOT CLOSE THE CELL. The residual never reaches zero: +62 at 4 chunks, +31
at 3, +6 at 1 — and "1 chunk" is T1 with extra steps (wall 1.1660s vs T1's 1.1357s, i.e. we
pay 2.7% to pretend to be parallel). Every chunk boundary costs bytes. Because we TIE
libdeflate byte-for-byte at T1 (G15), a cell needs T4 <= T1, so even +6 B FAILS.

    seam residual by boundary count (sil40 L9):
      7 boundaries -> +1,115 B     3 -> +62 B     2 -> +31 B     0 -> +6 B

The same shape holds on the other files measured: dickens L9 default +261 -> `-b 4M` -41
(SMALLER than T1 — boundary placement can occasionally help); data.csv L9 default +1,288 ->
`-b 64M` +9; sil40 L6 default -30 -> essentially flat.

### What this settles

The 109 zero-headroom cells have exactly two possible fixes, and grid tuning is NOT one of
them:

  1. SIZE MARGIN AT T1, so a small positive seam still lands under the rival. MEASURED AND
     BLOCKED — G17: every configuration that is smaller on all files exceeds the wall budget,
     and G16: the encoder side yields only 0.001% at 10-14% wall.
  2. BOUNDARIES THAT DO NOT RESTART CODING — workers emit parse artifacts, the consumer owns
     the final bitstream, so a chunk boundary is a scheduling artifact rather than a block
     boundary. NOT ATTEMPTED. This is the only remaining route.

Grid tuning improves the exchange rate but cannot reach zero, so it can never close a cell on
its own. It is worth revisiting ONLY in combination with (1) — if T1 ever carries even ~0.01%
of margin, a 4-chunk grid's +62 B would sit comfortably under it.

DO NOT change the DEFAULT grid on the strength of this. `pipelined_block_size`'s own FALSIFY
record requires a deliberate variation gated on BOTH axes, and CHUNKS_PER_THREAD 2->1 was
already tried and scored 8 cells closed / 3 opened. The 9.5% wall here is a single file at a
single level on a non-frozen box.

### G24a — the NAIVE form of route (2) is disqualified by a measured Amdahl bound

Route (2) as previously sketched (adversarial review, and repeated in the G14 seam notes) was
"workers emit parse artifacts; the CONSUMER owns final bitstream emission". Price it with the
profile rather than adopting it: `emit_sequences` is 63,180,388 Ir of 562,981,991 = 11.2% of
the program (trainer, L2, 8 MB; and libdeflate's counterpart `deflate_flush_block` is
64,280,257, so this is not a gzippy inefficiency — it is what emission costs).

Making that 11.2% serial:

    T4  -> 2.99x of 4x ideal      T8 -> 4.48x of 8x      T16 -> 5.96x of 16x

At T16 the wall would lose roughly a factor of 2.7. That trades 109 SIZE cells for a wall
regression across every T8/T16 cell on the board — a strictly worse deal, and clause 3 forbids
the pass->fail flips it would cause.

WHAT SURVIVES IS THE NARROW FORM. The cross-chunk coordination is only needed where a block
SPANS a chunk boundary — measured at 3 extra headers over 916 blocks at T4 (G14). So only
~(T-1) blocks need the consumer to own their table and boundary; the other ~900 stay
worker-local and fully parallel. Serial fraction becomes ~0.3% rather than 11.2%, and Amdahl
at T16 is then ~15.5x instead of 5.96x.

That is the design the evidence supports: NOT "the consumer emits everything", but "the
consumer owns only the seam blocks". Still unattempted, and still the only route to the 109
cells — but it is now a bounded change to the boundary handling rather than a rewrite of the
emission path.

## G25 — "recover the grid cost by scheduling" is NOT available: 8 chunks is already the optimum

#226's next-step list said the one-chunk-per-thread grid's ~11.4% wall could be recovered by
scheduling rather than by widening the grid back. Measured (sil40 40,000,000 B, L9, T4,
hyperfine n=5 warmup 1, /dev/null, local M1):

    grid                    wall       user       CPU% (user/wall)
    8 chunks (default)     0.3258 s   1.5386 s      472%
    4 chunks (1/thread)    0.3639 s   1.5763 s      433%
    16 chunks (-b 2.5M)    0.3585 s   1.5708 s      438%

USER TIME IS FLAT (+2.4% from 8 to 4 chunks) while WALL rises 11.7%, so the regression is
parallel efficiency, not extra work — that much of the #226 framing was right.

BUT 8 CHUNKS IS ALREADY THE OPTIMUM, IN BOTH DIRECTIONS. 16 chunks is also slower (0.3585)
than 8 (0.3258): past a point, per-chunk matchfinder warm-up costs more than the balance it
buys. And below it, four uneven work units cannot be balanced across four threads by ANY
scheduler — there is nothing to steal. `CHUNKS_PER_THREAD = 2` is not an untuned default; it
is the measured peak.

So the seam/balance conflict is structural, not a scheduling bug:
  - few chunks  -> few seams, poor balance   (4 chunks: -66 B but +11.7% wall)
  - many chunks -> good balance, many seams  (8 chunks: +1,115 B, best wall)
and no chunk COUNT escapes it, because the chunk is simultaneously the scheduling unit and
the coding unit.

### Everything now converges on the same fix

Three independent lines have arrived at the same place:
  G24  grid tuning cannot reach a zero seam residual
  G25  scheduling cannot recover the cost of a low chunk count (this section)
  G24a the naive "consumer emits everything" is disqualified (11.2% serial -> T16 caps 5.96x)

What survives all three is the NARROW form: keep the 8-chunk grid for scheduling, and have the
CONSUMER own only the ~(T-1) blocks that span a chunk boundary — 3 of 916 blocks at T4, a
~0.3% serial fraction. That decouples the scheduling unit from the coding unit, which is the
single assumption every dead end above shares.

Combined with the #226 composition (dual-candidate Huffman margin), the arithmetic then reads:
seam ~0 instead of +62 B, and 122-166 B of margin still in hand — with the 8-chunk grid's
wall, not the 4-chunk grid's.

NOT ATTEMPTED. It is the only route left to the 109 zero-headroom cells.

## G26 — RETRACTION of G-note #226's headline: "closes 4 of 6" was chunk-placement luck

#226 reported that dual-candidate Huffman + one-chunk-per-thread closes 4 of 6 test cells.
Those numbers came from hand-picked `-b` values, and THE SHIPPED GRID CANNOT PRODUCE THEM:
`pipelined_block_size` rounds (dickens is 12,174,519 B, so one-chunk-per-thread is 3,000,000
after rounding, not the 3,043,630 I passed to `-b`), and different rounding lands on different
block boundaries.

Baked in as a shippable default (`CHUNKS_PER_THREAD = 1`, no env, no `-b`):

    file      L   libdeflate(=T1)   T4 shipped   delta   verdict
    dickens   6      4,539,505       4,539,692    +187   fails
    dickens   9      4,480,689       4,480,684      -5   CLOSES
    data.csv  6      3,372,612       3,372,359    -253   CLOSES
    data.csv  9      3,300,291       3,300,815    +524   fails
    sil40     6     15,555,063      15,555,612    +549   fails
    sil40     9     15,452,666      15,452,762     +96   fails

TWO of six, not four — and a DIFFERENT two. The outcome is governed by where chunk boundaries
fall, not by a mechanism: the margin (122-166 B) is the same order as the seam's variance
across placements (hundreds of bytes). G24 already showed this (dickens L9: `-b 4M` -41 B but
`-b 3.04M` +325 B) and I read it as noise rather than as the governing term.

WHAT SURVIVES IS CHEAPER. At the DEFAULT grid (`CHUNKS_PER_THREAD = 2`, the measured wall
optimum per G25) the Huffman margin ALONE closes the same count:

    data.csv L6   3,372,268 vs  3,372,612  = -344  CLOSES
    sil40    L6  15,554,873 vs 15,555,063  = -190  CLOSES

2 of 6 for +2.3% wall, against 2 of 6 for +13.7% with the grid change. The grid half costs 6x
the wall and buys nothing net. CHUNKS_PER_THREAD stays at 2.

The structural point that resurrected the Huffman half still holds — G16 deleted a strict size
win (49/49 smaller, 0 worse) over a wall cost that is 6x smaller at T4 than the T1 number it
was judged on. What does NOT hold is the inference that COMPOSITION WITH THE GRID was the
mechanism. It was placement.

This strengthens G25's conclusion rather than weakening it: a margin this small cannot ride on
a seam whose size swings by hundreds of bytes with placement. The seam has to be ~0, not merely
smaller — the narrow "consumer owns the seam blocks" form.

## G27 — the next size lever is BLOCK COUNT: headers are 2.07% of output, 2,600x the Huffman margin

Measured with `anatomy-counters` (sil40 40,000,000 B, L9, T1, local M1), plus the per-header
cost already established by the G14 seam accounting (+3 dynamic blocks cost ~1,050 B, i.e.
~350 B per header):

    blocks_emitted_dynamic       913
    blocks_emitted_fixed           1
    blocks_emitted_stored          0
    huffman_exact_code_chosen    145      (15.9% of dynamic blocks)

    header mass  ~= 913 x 350 B = 319,550 B = 2.07% of the 15,452,666 B output
    Huffman dual-candidate margin = 122 B = 0.0008% of output
    ratio                                     ~2,600x

The exact-code candidate WINS OFTEN (145 of 913 blocks) but each win is worth ~0.84 B. That is
why it closes only the cells whose seam happens to be small: it is a real mechanism with a tiny
amplitude.

BLOCK COUNT IS THE LEVER WITH ACTUAL AMPLITUDE. Every dynamic block pays ~350 B to transmit
its own literal/length and offset code lengths, RLE-coded through a precode. 913 of them is
2.07% of the file. Cutting the block count 20% without worsening the data coding would be
~0.4% — four hundred times what the Huffman candidate delivered, and comfortably more than any
seam on the board.

The catch is the same one the splitter already trades against: fewer blocks means each table
fits its span worse, so data bits rise as header bits fall. Our splitter is a faithful port of
libdeflate's (`block_split.rs` vs `deflate_should_end_block`/`do_end_block_check`), which is
exactly why we tie their bytes — so we are currently making THEIR trade, not a better one.
zopfli and ECT both beat that trade with real block-splitting search, which is where their 1-3%
comes from.

NOT ATTEMPTED. The candidate pattern that worked for the Huffman code applies directly and is
non-worse by construction: cost the accumulated block as ONE block against the same tokens as
TWO blocks split at a chosen point, and emit whichever is cheaper. Cost is per block, not per
position — the affordable shape (G16/G17). Unlike the Huffman candidate, the amplitude is large
enough to matter on its own.

CAVEAT: 350 B/header is inferred from the G14 seam accounting, not from a direct sum of
`header_bits()`. Before this gates anything, add that sum as a counter and measure it — a
header-mass claim built on an inferred constant is exactly the kind of number this campaign has
been burned by.

## G27a — CORRECTION: header mass is 0.634%, not 2.07%. The seam is BAD PLACEMENT, not headers

G27 argued block count was the next lever using ~350 B/header INFERRED from the seam
accounting, and flagged that the inference had to be replaced by a direct sum before it gated
anything. Done — `dynamic_header_bits_total` sums `header_bits()` over every dynamic block
actually emitted (sil40 40,000,000 B and dickens, L9, T1, anatomy-counters build):

    file      blocks   header bits   bytes    per header   share of output
    sil40       913       783,318    97,914     107.2 B       0.634%
    dickens      49        31,745     3,968      81.0 B       0.089%

THE INFERRED 350 B/HEADER WAS 3.3x TOO HIGH, and header mass is 0.634% of sil40, not 2.07%.
G27's headline number is withdrawn.

WHY THE INFERENCE FAILED, and it is the useful part. The seam accounting measured that 3 EXTRA
BLOCKS cost ~1,050 B (~350 B each). Only ~107 B of that is the header. The remaining ~240 B per
extra block is the cost of splitting AT A FORCED CHUNK BOUNDARY instead of where the parse
wanted to split — a worse-fitted pair of tables on both sides.

So the T>1 seam is NOT primarily a header-count problem. It is a BOUNDARY-PLACEMENT problem,
and ~70% of its cost is the misplacement rather than the extra header. That is consistent with
the tail-guard falsification (G-note: suppressing late splits made every file WORSE, because
well-placed splits earn their headers) and it explains why grid tuning moved the seam so much
(G24: +1,115 B at 7 boundaries down to +6 B at 0) — fewer forced boundaries, not fewer headers.

### What this does to the two candidate levers

  BLOCK COUNT (fewer, larger blocks): headers are 0.634% of output, so a 20% cut is at most
  0.13% BEFORE the data-bit penalty of worse-fitted tables. Still ~160x the Huffman margin
  (0.0008%) and still far above the ~0.01% the zero-headroom cells need — but it is a
  0.13% ceiling, not the 0.4% G27 claimed.

  BOUNDARY PLACEMENT: ~70% of the seam is misplacement. This is the same conclusion G25 reached
  from the scheduling side and G24 from the grid side, now from the byte accounting: what the
  T>1 path needs is for chunk boundaries to stop FORCING block boundaries — the narrow
  "consumer owns the seam blocks" form — not fewer or cheaper headers.

Three independent accountings now name the same fix. That is the one to build.

METHOD NOTE. This is the fifth claim of mine this session that measurement retracted, and the
second where an INFERRED constant stood in for a measured one. The rule that caught it is the
one already in CLAUDE.md — a gate may only cite a dataset that exists — and the fix each time
was to add the counter rather than to argue about the constant.

## G28 — the decomposition CONFIRMED on the frozen box (first authoritative board data this session)

Every board number above this line was measured on the local M1. The base arm of the solvency
wall run re-measures `origin/main` on the frozen authority, and the structure holds:

    440 decidable cells (22 files x L2/L6/L9 x T1/T4 x 4 rivals; 88 ABSENT = igzip gaps)
     68 failing on size (15.5%; the full L1-9 board is 200/1320 = 15.2% — consistent)

    failing by rival and thread count:
      libdeflate  T4   48      <- 70.6% of ALL failures
      gzip        T1    6      gzip  T4   6
      pigz        T1    4      pigz  T4   4
      libdeflate  T1    0      <- we tie or beat libdeflate at EVERY T1 cell here

`libdeflate T1 = 0` and `libdeflate T4 = 48` is the zero-headroom structure stated directly:
at T1 we match or beat them everywhere in this level set, and every libdeflate failure is a
T>1 cell where the seam pushes us over a tie. That is G15's finding, re-derived on the
authority box from an independent run rather than inherited from the frozen census.

It also sets the honest scale for what has been achieved: the T>1-vs-libdeflate class is
70.6% of the board, the Huffman margin closes the handful of its cells whose seam is small
enough (5 on the LOCAL corpus; the solvency verdict is still pending and already disagrees on
at least one — `movie.mp4 L9 T4` shows us 19 B BIGGER there), and the remaining ~43 need the
seam itself to go to zero.

## G29 — THE FRONTIER RESULT WAS MEASURED AT THE WRONG COORDINATE. T4 has 249-330% wall slack

G17 concluded that we sit ON libdeflate's size/wall frontier and that no parse configuration
can win both axes. That conclusion measured WALL SLACK AT T1, where it is 0-8%. It is wrong
for the cells that actually fail.

libdeflate-gzip is SINGLE-THREADED. Our failures are overwhelmingly T4 (G28: 48 of 68, and
libdeflate T1 = 0). At T4 the comparison is our 4 threads against their 1:

    sil40, vanilla build, hyperfine n=5 warmup 1, /dev/null sink, local M1
      L6   gzippy T4 0.1185 s   libdeflate 0.4135 s   we are 3.49x FASTER   slack 249%
      L9   gzippy T4 0.3197 s   libdeflate 1.3742 s   we are 4.30x FASTER   slack 330%

Every parse configuration G17 priced and rejected was rejected against a 0-8% budget. The
budget where the cells fail is 249-330%.

### What that buys, measured

`lazy2:35:65` at L6 (the config G17 rejected at +9.29% against 4.4% of T1 slack), run at T4 on
a clean tree (HEAD 05dda708, `ladder-tune` build, all four files):

    file          libdeflate-6    ours T4 default      ours T4 lazy2:35:65
    sil40         15,555,063      15,554,873  (-190)   15,535,712  (-19,351)
    dickens        4,539,505       4,539,848  (+343)    4,520,157  (-19,348)   <- default FAILS
    data.csv       3,372,612       3,372,268  (-344)    3,348,563  (-24,049)
    winexe.exe     1,510,118       1,509,934  (-184)    1,510,071  (-47)

    wall, sil40 L6:  libdeflate 0.4153 s | ours T4 default 0.1244 s (3.34x) |
                     ours T4 lazy2 0.1380 s (3.01x, i.e. +10.9% of OUR time)

19,000-24,000 B of margin — about 100x the seam it needs to absorb — while remaining 3x faster
than the rival. This is the libdeflate-T4 class, 48 of the 68 failures on the frozen box.

### The change this implies is SANCTIONED, not a knob

The level->config map must become THREAD-COUNT-AWARE: keep today's config at T1 (little slack),
spend the parallel slack at T>1. That is not an env var and not content detection — CLAUDE.md
non-negotiable #3 explicitly permits "parameter tuning (write-buffer size, shared memory per
thread count)", and STEP 2 states outright that "T>1 may emit different bytes than T1".

### THE METHOD LESSON, which is the important part

This was found by asking: when something is "measured dead with receipts", how do we know it
was not a good idea whose co-parameter was simply unoptimised?

The answer is a SIGNATURE. A verdict is coordinate-dependent, not intrinsic, when:
  * the benefit is MONOTONE (non-worse by construction on some axis), and
  * the cost is NOT intrinsic — it depends on a coordinate (thread count, level, what else is
    enabled) that was held fixed and unexamined.

Two receipts from this session alone:
  * G16 killed the dual-candidate Huffman as "0.001% for 10-14% wall". Both halves were
    coordinate artefacts: the cost is +2.3% at T4 (the work parallelises), and the 0.001% is
    not a size win but SEAM ABSORPTION. It now has a SHIP verdict.
  * G17 killed the whole parse-config space against a T1 budget. The cells fail at T4, where
    the budget is 40x larger. This section.

So a FALSIFY note must record the COORDINATE it was measured at and separate an INTRINSIC
CEILING from a COORDINATE-DEPENDENT VERDICT. "libdeflate's heuristic is within 0.001% of the
mathematical optimum" is intrinsic and permanent. "Therefore it is dead" was neither. And
structurally-right-but-currently-losing work should be PARKED WITH ITS COORDINATE, never
deleted — deleting it is what made G16 need rediscovering.

## G38 — after the 42-cell composition: 26 residual cells, and 6 of them belong to work already verified

Residual computed from the AFTER arm of the combined lever
(`lever-combined2/try.json`, 528 cells / 440 decidable) — measured, not projected.

    base failing 68  ->  after failing 26   (42 closed)

    libdeflate T4  12 | gzip T1 6 | pigz T1 4 | gzip T4 3 | pigz T4 1

    worst first
      gzip  T4  dd79_bin6           L2  1.00935      libdeflate T4 data.csv    L9  1.00034
      gzip  T1  dd79_bin6           L2  1.00927      libdeflate T4 data.json   L9  1.00022
      pigz  T4  dd79_bin6           L2  1.00818      gzip       T4 weights     L9  1.00022
      pigz  T1  dd79_bin6           L2  1.00809      gzip       T1 weights     L9  1.00021
      gzip  T1  monorepo.tar        L6  1.00592      libdeflate T4 engine.wasm L9  1.00014
      gzip  T1  aozora.txt          L6  1.00568      libdeflate T4 dd79_bin6   L2  1.00008
      pigz  T1  monorepo.tar        L6  1.00540      libdeflate T4 dd79_bin6   L9  1.00006
      pigz  T1  aozora.txt          L6  1.00450      libdeflate T4 minjs       L9  1.00003
      gzip  T1  minjs.min.js        L6  1.00341      libdeflate T4 photo.jpg   L9  1.00001
      pigz  T1  minjs.min.js        L6  1.00311      libdeflate T4 movie.mp4   L6  1.00001
      gzip  T4  photo.jpg           L2  1.00046      libdeflate T4 photo.jpg   L6  1.00001
      gzip  T1  photo.jpg           L2  1.00045      libdeflate T4 weights     L2  1.00001
                                                     libdeflate T4 photo.jpg   L2  1.00001
                                                     libdeflate T4 weights     L9  1.00000

### Three groups, and only one of them is open work for me

**(a) SIX L6 T1 CELLS — already solved by someone else.** monorepo.tar, aozora.txt and
minjs.min.js against BOTH gzip and pigz. G31a measured the foreign `good_match` patch closing
exactly these (aozora -35,823 B, monorepo -27,716 B, minjs to 1,087,222 vs gzip's 1,088,768).
They are T1 cells and the composed change is T>1-only, so the two are disjoint and should ADD:
26 - 6 = 20 residual. THIS IS NOT MINE TO LAND. Its owner should gate it; I have only measured
it. Duplicating it would be the worse outcome.

**(b) TWELVE libdeflate T4 CELLS AT RATIO <= 1.00034, most at 1.00001.** One part in 100,000 —
on a 6.4 MB file that is ~64 B. These are the pure-seam cells (G34): T1 already ties libdeflate
byte-for-byte, so the entire deficit is the T>1 seam and the margin needed is tiny. Every
mechanism that could supply it cheaply has now been measured: parse depth (saturates, G30),
block count (dead at matched counts, G37e), exact Huffman (already in the composed change).
What is left for them is the seam itself.

**(c) EIGHT CELLS ON dd79_bin6 / photo.jpg / weights.safetensors.** dd79_bin6 L2 is the worst
on the board (1.00935) and G36 attributed it to SHORT-MATCH DISCOVERY — gzip finds 319,260 more
matches, concentrated in the shortest bucket, because our hash3 is a one-deep singleton.
photo.jpg and weights are the per-block-COST class whose count-based readings are all falsified
(G37e); their remaining unknown is gzip's BTYPE mix, which no tool currently reports.

So of 26: 6 belong to work already verified and owned elsewhere, 12 need the seam
re-architecture, and 8 need either hash3 chaining (attributed, 4 cells) or a per-block cost
measurement that does not yet exist (4 cells).

## G40 — the T>1 seam DECOMPOSED per cell: three terms, and no single one dominates

The 12 residual libdeflate T4 cells fail by ~40-1,400 B. Measured what that is actually made
of, T1 vs T4, with the block-type counters (anatomy-counters build, exact bytes):

    file             L   T1 dyn/stored   T4 dyn/stored   seam    stored x5 B   unexplained
    photo.jpg        6      25 / 0          25 / 10       +70         50           ~20
    engine.wasm      9      17 / 0          17 /  2       +61         10           ~51
    minjs.min.js     9      66 / 0          69 / 11       +39         55           -16
    movie.mp4        6      54 / 0          55 /  9      +195         45          ~150

THREE TERMS, NONE DOMINANT:
  1. SYNC-FLUSH overhead — the stored blocks, 5 B each, one per chunk seam.
  2. EXTRA DYNAMIC HEADERS — 0 on photo.jpg and engine.wasm, +3 on minjs, +1 on movie.mp4.
  3. A RESIDUAL that neither explains — +150 B on movie.mp4 with only 9 flushes and 1 extra
     header; NEGATIVE on minjs (its 11 flushes and 3 headers "should" cost 55+ B, yet the
     total seam is only 39 B, so something at T4 is SMALLER).

Term 3 is boundary MISPLACEMENT: a chunk edge forces a block to end where the parse did not
want one, so the tables either side fit worse — and occasionally BETTER, which is why minjs
comes out under its own overhead.

### An overclaim caught by checking a second file

photo.jpg alone reads as "the seam is ENTIRELY sync flushes": same dynamic count, 10 stored
blocks, +70 B against 50 B of flush overhead. That reading would have justified building
bit-level concatenation to remove the flushes, on the strength of one file. engine.wasm
refutes it immediately — same dynamic count, only 2 flushes (10 B), and still +61 B.

So REMOVING THE SYNC FLUSHES ALONE WOULD NOT CLOSE MOST OF THESE CELLS. It would close
photo.jpg L6 (+70 -> ~+20, and our T1 is 6,472,061 against libdeflate's 6,472,062) and leave
engine.wasm, movie.mp4 and the rest failing. That is worth knowing BEFORE building it: the
bit-shift concatenation is real work and it buys one cell of this class, not twelve.

### What this does to the re-architecture estimate

G24a scoped "consumer owns the seam blocks" as the fix, on the assumption the seam is headers.
It is not, on this class: headers are 0 extra on half the cells measured. The consumer-owned
seam block would fix term 2 and term 3 (it removes the forced boundary entirely) but NOT term
1 unless the concatenation also goes bit-level. Both halves are needed for this class, and the
cheaper half alone was about to be built on a one-file reading.
