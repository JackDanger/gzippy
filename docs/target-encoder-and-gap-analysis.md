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
