# The encoder campaign: state, method, next actions

Read this before touching the encoder. Written to be picked up cold.
`CLAUDE.md` has the rules; this has the board, what already failed, and what to do.

---

## STATUS: DEFERRED (2026-07-30) — blocked on decision D3, not on engineering

**The board is 223 failing cells of 1,320** (squishy GATE+TUNE, 4 rivals, L1-9 x T1/T4,
0 VOID, subject `d2e47469`). The campaign is **explicitly deferred**, not abandoned and
not quietly stalled: every remaining route terminates in a decision the engineer is not
permitted to take, and each is characterised well enough to resume from cold.

| class | cells | terminates in |
|---|---|---|
| T4 chunk overhead | 133 (60%) | **D3** — clause 3, 9 flips all under 0.02%, FIVE grid shapes tried |
| L1 ratio | 34 | insert-density vs write-traffic dial; 3 escapes tried, all trade back |
| L4 monotonicity | ~22 | **D1** — clause 5, lazy-at-L4 costs 17.7% wall; `deflate_medium` unmeasured |

**Nothing here is waiting on someone to think of an idea.** The levers are built,
measured, and recorded; PR #189 holds a working change that closes 29 cells and is
blocked by one clause. Resume by answering D3 (below) — or, if D3 is answered "clause 3
stands", by taking the `deflate_medium` path in D1, which is the only remaining route
that needs new code rather than a ruling.

**What NOT to do on resume:** re-sweep `SOFT_MAX_BLOCK_LENGTH` (closed both directions),
re-try chunk-grid constants (five shapes, all flip), re-try bounds-check elision or
`LIMIT_HASH_UPDATE` on the ht finder (both falsified in code), or re-attempt the
signal-gated block-end bias (third instance of a banned content detector). Seven levers
are falsified at their sites; `fulcrum candidates` surfaces them.

---

## 1. Where we stand

**165 failing per-label SIZE cells.** Roundtrip-verified census, canonical corpus (17 files staged of
the 20 canonical members) x L1-9 x 4 rivals x T1/T4; 1020 cells measured, 0 VOID.

> ⚠ **THIS HEADLINE IS UNATTESTED AND ITS DENOMINATOR DOES NOT RECONCILE** (audited
> 2026-07-30). Four different corpus cardinalities are in simultaneous use across the
> repos — 21, "20 canonical / 17 staged" (here and `parse/mod.rs`), 11 (the only banked
> size census), and 22 declared. The declared set is the only one with a contract:
> `corpus_split.json` names **11 GATE + 11 TUNE = 22** members, of which **21 are staged
> locally and `sil40` is absent** — and `sil40` is a `goal::MIN_CORPORA` member, i.e. part
> of the mandated minimum promotion surface, with **no sha, size, or generator recorded
> anywhere in either repo**. Worse, the only banked size census
> (`~/www/gzippy-bench/sizecensus_threads/`) has `"attested": false`,
> `"gzippy_commit": null`, covers **11 files and THREE rivals with igzip simply absent** —
> so §1's igzip row cites no dataset that exists on this box, and `CLAUDE.md` hard stop #7
> ("a measurement from an unidentified binary is not a measurement") is violated by this
> plan's own foundation. Re-derive with `scripts/campaign/board-size.sh` (§8), which
> refuses undeclared members, refuses a missing rival, and refuses an unidentified
> instrument. Do not quote the 165 until it has been reproduced.

| rival | T1 fails | T4 fails |
|---|---|---|
| libdeflate | 11 | 113 |
| gzip | 11 | 12 |
| pigz | 8 | 8 |
| igzip | 1 | 1 |

**The wall axis is UNDER-measured, not unmeasured.** (Corrected 2026-07-30; the previous
text said "has no census yet… Build it before running a wall lever", which would send the
next session to build a tool that exists and re-derive data that exists.) The tool is
`fulcrum board wall`, whose own module doc calls itself "the missing half of the goal
scoreboard", and one census is already banked:
`~/www/gzippy-bench/wallcensus_t1_mac/census.json` — **189 cells, 145 OK, 44 VOID.** Its
scope is the limit, and it is declared in the artifact: T1 only, aarch64, 7 files, THREE
rivals (no igzip), on an unfrozen ~27%-busy Mac, labelled PROVISIONAL and never banked as
authority. So what is missing is a FROZEN, four-rival, multi-thread wall census on
solvency — a run, not a build.

**Two facts that shape everything:**

1. **Our T1 output is byte-size-identical to libdeflate at L2 and L4-L9** on all 20
   canonical files. We ship their algorithm, so we have **zero size slack** at those
   labels and every small T>1 overhead turns a tie into a failed cell. We chose this by
   copying their level table; it is not a law.
2. **At L2 we execute 1.12x libdeflate's instructions (555.1M vs 496.0M) at better
   IPC**, with fewer frontend stalls and a lower L1D miss rate. 61% of the excess is
   LOAD instructions. **Our emit path is already 0.90x theirs (57.4M vs 63.8M) — the
   whole debt is parse+matchfinder.**

## 2. THE BOARD, MEASURED (2026-07-30) — and it splits by THREAD COUNT, not by front

First attested promotion board: `scripts/campaign/board-size.sh all`, subject `d2e47469`,
squishy GATE+TUNE (22 members), 4 rivals, L1-9 x T1/T4, **1,320 cells measured, 0 VOID**.
Artifact `~/www/gzippy-bench/campaign/size-all-d2e47469/`.

**223 failing.** And GATE is WORSE than TUNE, which is exactly what the TUNE/GATE split
exists to expose — parameters were historically fitted on TUNE, so the surface promotions
are actually judged on had never been looked at:

| | GATE (promotion surface) | TUNE |
|---|---|---|
| **total** | **127 / 660** | 96 / 660 |
| libdeflate | 87 / 198 | 79 / 198 |
| gzip | 24 / 198 | 9 / 198 |
| pigz | 12 / 198 | 8 / 198 |
| igzip | **4 / 66** | 0 / 66 |

**THE DECOMPOSITION THAT MATTERS: 133 of the 223 — 60% — PASS AT T1 AND FAIL ONLY AT T4.**
Median gap 567 B; 97 of 133 under 1,000 B; 132 of 133 under 5,000 B. Against our own
T4-T1 deltas of 1,000-3,300 B on the big files, so **the gap IS the chunk-seam cost**, not
a ratio deficit. The remaining 90 fail at both thread counts and are real ratio deficits,
34 of them at L1.

So the board is two problems, and the bigger one is T>1 framing:

| class | cells | what it is |
|---|---|---|
| **T4-only** | **133** | chunk overhead. ~~STEP 2 sanctions closing it by making seams smaller~~ — **RETRACTED 2026-08-01: seam-shrinking cannot close this class at all.** All 109 such libdeflate cells tie byte-for-byte at T1 (headroom min=median=max=0), so there is ZERO partial credit; 90% off the seam tax closes 0 of 109. This is exactly why "FIVE grid shapes were tried and all flip" below — every one shrank the seam and none reached zero. The route is a **monotone T1 size win that buys headroom** — but NOT from Huffman construction, which is closed at ~0.001% against the ~0.01% needed (`src/compress/deflate/huffman/fast.rs:432`). Block boundaries or the parse. |
| fail at both | 90 | real ratio; 34 at L1 (Front B), the rest spread L2-L9 |

**The T4-only class is closable and is blocked on a RULE, not on engineering.** A
thread-aware chunk grid takes the board to **203 (29 closed, 9 opened)** and collapses the
seam cost (tool.bin T4-T1 2,828 -> 80 B; weights.safetensors 2,962 -> -1 B). It fails
promotion clause 3 on 9 flips, every one a T4 cell with a margin under 0.02% (winexe.exe
L9 +59 B on 1.5 MB). **FIVE grid shapes were tried and all flip** — including a plain
1 MiB fixed grid with no thread term at all, which flips 8. The flips are caused by
changing the chunk size AT ALL; those cells are won by the specific 512 KiB grid, not by
anything robust. See `pipelined.rs` for the table and PR #189 for the probe.

Whether clause 3's absolute no-flip bar should apply to sub-0.02% SIZE ties is a decision
for the user. Until it is taken, the lever against 60% of the board is foreclosed — and
that should be deliberate rather than a side effect.

## 3. The pre-census framing (kept for its per-chunk arithmetic)

The failing T4 cells, gap-to-rival divided by (chunks x per-chunk overhead), where
chunks are the 512 KiB pipelined grid (`pipelined.rs:65`). NOTE: §1's rival table sums to
134 and this decomposition covers 133 — recount before trusting either number; the
missing cell is unidentified.

| gap / chunk-overhead | cells | reading |
|---|---|---|
| <= 1 | 38 (29%) | chunk overhead alone explains it |
| <= 2 | 30 (23%) | reachable with slack + grid together |
| <= 5 | 32 (24%) | |
| > 5 | 33 (25%) | real ratio deficit — almost all L1 |

**Front A — chunk overhead (~100 cells).** Measured per-chunk constant: **+18.7 B at
L2, +32.1 B at L6** on 512 KiB chunks (silesia, 405 chunks) = 0.0036-0.0061% of input.
Seams are only ~5.4 B; the dominant term is extra dynamic-header mass from restarting
the block grid inside every chunk, and it grows with level. (Provenance: these two
constants and the cell split below came from runs whose artifacts are not in-repo. They
are consistent internally and with the census, but re-derive them before betting a large
change on them — `CLAUDE.md` says a gate may only cite a dataset that exists.)

The **31 T1 fails are not decomposed here.** Assign each to a front (most are Front B's
L1 class plus the D1/L4 monotonicity family) so every one of the 165 cells belongs to
exactly one of Front A, Front B, or D1. A cell in no bucket is a plan gap.

**Front B — the L1 ratio class (33 cells).** Gaps of 60 K-636 K bytes: access.log at
911x the chunk overhead, monorepo 515x, data.csv 421x, aozora 380x. Our `Fast` parser
against libdeflate's `deflate_compress_fastest`. No grid, seam or coding-locus change
touches it. Board task #25.

## 3. The method that wins here, counted

Decode's four winning months, plus this campaign:

| method | wins | falsifications |
|---|---|---|
| vendor structural diff / convergence / port | ~9 | — |
| causal perturbation (removal oracle, ablation, blocked-on decomposition) | ~8 | — |
| **shaving our own profile's top line** | **0** | **>= 17** |

Rules that follow, all present in `CLAUDE.md` as hard stops:

- **A lever from our own profile is not a candidate.** Name the vendor difference it
  steals, or state "no counterpart, bar is the measurement".
- **Converge on the vendor's structure BEFORE deleting anything.** Decode halved the
  igzip gap by faithful transliteration; only then did single-op deletions become
  visible and worth 5-10% each. Deleting first is what produced this campaign's
  falsifications.
- **The causal-perturbation family has never been built here.** Every decode win that
  was not a vendor port came from stubbing a region and watching the number respond.
  Build it when a named cell needs it, not speculatively.
- **Instruction counts LOCATE, they never predict the wall.** Receipt: a change that
  cut Ir 1.77% and Dr 3.87% was 9.9% SLOWER at L9.
- **Name the cell AND the axis before starting.** A byte-identical change can never
  close a size cell. Two real wins landed here and moved the board by zero for exactly
  that reason.
- **Land gated work first.** `gh pr list` before starting anything.

## 4. Already falsified — needs a NEW mechanism, not a retry

Each has a `FALSIFY` note at the tempting code site; the `commit-msg` hook requires a
`REOPEN:` line naming a new mechanism to touch those sites.

> ⚠ **THAT SENTENCE WAS FALSE FOR THE TWO ROWS THE NEXT ACTION EDITS** (found and fixed
> 2026-07-30). `block_split.rs` carried **zero** FALSIFY notes on `main` and
> `SOFT_MAX_BLOCK_LENGTH` carried none — the block-budget note existed only on the
> unmerged branch `perf/blockend-recalibrate`, so the record of a falsification was itself
> never landed. Since the `commit-msg` hook keys on **proximity to a FALSIFY note in
> code**, and `fulcrum candidates` also surfaces falsifications by scanning **code**, both
> anti-repetition mechanisms were structurally blind to most of this table: it lives in
> markdown, and markdown does not fail closed. Both notes are now in the source.
> **The general rule this implies: a row here without a note at the site is not enforced.
> Adding the row is not the record — the in-code note is.**
>
> ⚠ **AND THE EVIDENCE FOR THE TWO BLOCK-BUDGET ROWS IS ON UNDECLARED FILES.**
> `logs.txt`, `text-1MB.txt` and `shortmatch-4M` are none of them members of
> `corpus_split.json`: the first two are synthetics generated by
> `scripts/prepare_benchmark_data.sh` and `scripts/generate_test_data.py`, the third is a
> gitignored local generation not present on this box at all. So those rows are not merely
> narrow, they are **unrepeatable** — nobody can re-run them. Project memory already holds
> a case where a synthetic said "+1 byte" where the real corpus said "+2.02%". Treat the
> mechanism as falsified and the magnitudes as unverified.

| attempt | result |
|---|---|
| Window-wrap hoist in `skip_bytes` | wins L2, regresses L6/L9; mechanism unexplained |
| Per-call variant of it | worse than the loop at both levels |
| Hand-hoisting the prefilter's invariant loads | **Dr went UP** — LLVM already hoisted them; the hoist only added register pressure |
| De-pipelining the chain walk | 1.0131 / 1.0624 / 1.0992 slower at L2/L6/L9 — the hoist hides dependent-load latency |
| Level-scaled chunk grid | 0.01% size for 2-4% wall |
| pigz 10-bit static seam pad | 0.0007% size for 0.7-0.9% wall |
| L2 level-map retune (Lazy/4/10) | +23.4% instructions; lazy runs two searches per position |
| L5 depth 16 -> 32 (gzip's chain) | closes ~10% of the gap on the failing file, regresses another |
| Block budget 300K -> 900K | no knee — table below |
| Coding-locus rework | falsified on paper before any code — below |

**Block budget** (L6, gap vs libdeflate, T1 / T4):

| budget | logs.txt | text-1MB | shortmatch-4M |
|---|---|---|---|
| 300K (shipped) | +0 / +5698 | +0 / +451 | +0 / +85 |
| 450K | -4025 / +4861 | -39 / +451 | **+468** / +85 |
| 600K | -4796 / **-745 PASS** | -39 / +451 | **+660** / +85 |
| 900K | -5464 / **-745 PASS** | -39 / +451 | **+660** / +85 |

The incompressible T1 regression appears above 300K and saturates at +660; the
compressible gain saturates at 600K. No budget keeps one without the other, and
per-label means every file. NOT a mis-scaled threshold — the split check is
budget-independent (`block_split.rs:18` fires every 512 observations, `:136` requires
`MIN_BLOCK_LENGTH` 5000 B). It is an adaptivity trade: one table over more symbols
loses to local statistical drift wherever there is no structure to exploit.

**Coding-locus rework, and why it died on paper.** "Workers hand the writer tokens; one
writer emits with the T1 grid, closing ~103 cells by construction" is false. Histograms
are built in the PARSE (at `Seq` push time in the `Sink`; `parse/mod.rs:121-131` is the
`Seq` doc comment, the frequency bump is in the push path) and the block grid is a per-symbol
observation stream with per-block state (`block_split.rs:14-61`), so a writer
re-blocking across seams must re-run those observations serially over every literal
byte — nowhere near the claimed 4-6% of work. If workers keep their own grid, the grid
still restarts per segment, which IS the dominant term. And exactness is required: T1
ties libdeflate at L2/L4-L9, so +1 byte fails, while seam-parsed tokens cannot
reproduce T1's exactly (pending lazy deferral, a match spanning the boundary, grid
phase) with sign-indefinite divergence. It returns only as a fallback, gated by a
seam-token diff: parse dict-seeded segments, concatenate, diff against T1's token
stream. Deterministic, no pipeline code.

## 5. Next actions, ranked

> ### ⚠ THE FIRST ATTESTED SIZE BOARD SAYS START WITH B1, NOT A1 (2026-07-30)
>
> `scripts/campaign/board-size.sh tune` — 4 rivals, commit-pinned, roundtrip-verified,
> **0 VOID**, artifact at `~/www/gzippy-bench/campaign/size-tune-bfd44096/`. Subject
> `bfd44096`, gzippy sha256 `8c7c05be0d1b`. Scope, stated: TUNE set (11 files) x L1-9 x
> T1,T4, SIZE only, aarch64. **96 failing cells of 660 measured.**
>
> | rival | failing / measured | worst | where the mass is |
> |---|---|---|---|
> | **libdeflate** | **79 / 198** | **+4.61%** | **the seven worst cells are ALL L1** |
> | gzip | 9 / 198 | +0.58% | aozora + minjs at L5-L7, T1 AND T4 alike |
> | pigz | 8 / 198 | +0.46% | the same cells as gzip |
> | **igzip** | **0 / 66** | — | **not a size front at all** |
>
> **Three corrections to the ranking below, each from this artifact:**
>
> 1. **B1 outranks A1 on size mass, by an order of magnitude.** The L1 class is
>    data.csv +4.61%, aozora +4.05%, minjs +2.26%, dickens +2.15%, data.json +1.77%,
>    engine.wasm +1.25%. Every non-L1 libdeflate cell is **<= 0.19%**. A1 targets Front A
>    (per-chunk overhead), which this census shows is the SMALLEST of the three clusters —
>    and A1's preferred mechanism is now blocked pending the user (see A1). B1 needs no
>    permission, is a pure vendor diff, and is where the bytes are.
> 2. **igzip is closed on size on this set** — 0 of 66. §1's igzip row (1 T1 / 1 T4) does
>    not reproduce, consistent with no artifact on this box being able to produce it.
> 3. **A cluster the plan never named: L5-L7 on aozora and minjs, against gzip AND pigz
>    simultaneously, at T1 and T4 in equal measure.** Equal at both thread counts means it
>    is NOT a seam or chunk-grid artifact — it is a real parse/ratio deficit in the
>    greedy/lazy band, and it is the only place where gzip beats us at all. It belongs on
>    the board as its own front.
>
> Not yet measured, and NOT to be inferred from the above: the GATE half (`make
> board-size-promote`, blocked only by the absent `sil40`), T8/T16, and the entire wall
> axis. This artifact is size on TUNE. Say so when quoting it.

**The ranking below is the pre-census one; item order is superseded by the box above.**
A1 was ranked first because it had a located mechanism, a vendor precedent, a named axis
and a one-build falsifier — all still true, but its mechanism now needs user sign-off and
the measured size mass is elsewhere. A2 depends on what A1 finds about safe block lengths.
B1 is a separate front and can run in parallel by another hand. The T1 wall item needs a
frozen four-rival wall census (§1) run first — the tool exists.

**A1 — recalibrate the block-END rule so the budget can rise. This is the `REOPEN:` of
the falsified "block budget 300K -> 900K" row, and it ships as a PAIR.**

The sweep's fact pair: budget 600K wins -4,796 B at T1 and flips T4 to PASS (-745) on
logs.txt, and costs +660 B at T1 on shortmatch-4M, whose T4 gap is flat at +85 at every
budget because the 512 KiB chunk grid (`pipelined.rs:65`) already gives near-random data
the smaller tables it wants. So the lever is a recalibrated end-of-block rule that cuts
structureless data short below the budget, PLUS `SOFT_MAX_BLOCK_LENGTH`
(`parse/mod.rs:64`) raised. A splitter change alone at the shipped 300K budget is
near-byte-identical, and per §3 a byte-identical change can never close a size cell.

**The vendor diff is DONE. This table is the result — do not redo it.**

| impl | end-of-block signal | parameters | where |
|---|---|---|---|
| ours L2-9 | drift detector AND byte budget AND seq cap | every 512 obs; cutoff 200/512; small-block penalty (<10000 B, <8192 obs); length bias `/4096`; 300,000 B; 50,000 seqs; MIN_BLOCK_LENGTH 5000 | `block_split.rs:14-147`, `parse/mod.rs:64`, `parse/mod.rs:606-618` |
| libdeflate L2-9 | **identical — ours is a verified term-for-term port** | same five numbers | `deflate_compress.c:443,66,81,93`, `:2142-2197`, `:2591-2595` |
| ours L1 | fixed byte quantum, no detector | 65,536 B | `parse/fast.rs:1549`; never calls `should_end_block` (`parse/mod.rs:254-256`) |
| libdeflate L1 | fixed byte quantum + seq cap | 65,535 B / 8,192 seqs | `deflate_compress.c:102,108`, `:2469-2470` |
| zlib-ng L2-9 | fixed SYMBOL quantum, no detector; then per-block cheapest-of-three | 16,383 symbols (`lit_bufsize-1`, memLevel 8) | `zlib-ng/deflate.c:289,311,357-360`, `deflate_p.h:84,114`, `trees.c:668,676,686` |
| igzip L1-3 | fixed TOKEN quantum, no detector; per-block hufftables + stored compare | 65,536 tokens x 4 B (CLI default) | `igzip_lib.h:296,310-315`, `igzip.c:307-324`, `:406`, `:365-416` |

**What the diff teaches, and it is a calibration fact, not a bug.** libdeflate is the
ONLY vendor with a drift detector and we ship it bit-for-bit — so there was never
anything to steal from them here. The other three rivals get small tables on
structureless data for free, from a fixed small quantum. Our detector's only
structureless-data response is the length-bias term `(block_length/4096)*num_obs`
(`block_split.rs:122` = `deflate_compress.c:2192`); at check time the cutoff is
`200*num_obs`, so that term alone fires only at `block_length >= 200*4096 = 819,200 B`
stationary — **calibrated dead below any budget we would ship.** That is the dial, and it
is why the +660 regression saturates between the 600K and 900K rows.

**Candidate mechanisms — REORDERED 2026-07-30, and the old (1) now carries a warning.**

**(1) PREFERRED: delete the detector; adopt a fixed symbol/token quantum.** zlib-ng ends
blocks on a fixed 16,383-SYMBOL quantum, igzip on a fixed 65,536-TOKEN quantum, and gzip
has no drift detector either — **3 of 4 rivals carry no detector at all.** libdeflate is
the only one that does, and `block_split.rs` is a verified term-for-term port of theirs,
so there was never anything to steal from them here. The other three get small,
well-fitted tables on structureless data *for free*, from a quantum that needs no signal.
This route REMOVES a data-dependent branch instead of adding one, is vendor-precedented
three times over, and is the shape the user named as what worked: "one plain good
implementation that didn't do anything wasteful… easy to optimize incrementally."
Optionally add zlib-ng's per-block cheapest-of-three (stored/fixed/dynamic) cost check,
which is a deterministic emit-time comparison, not a predictor.

**(2) ⚠ DO NOT BUILD WITHOUT ASKING THE USER: recalibrate the bias, gated on an
emitted-symbol signal.** This was listed first and was tried on
`perf/blockend-recalibrate`; it is **the third instance of a mechanism `CLAUDE.md`
non-negotiable #3 orders deleted**, and the argument offered for it — "reads symbols
already emitted, never input ahead, so not a content detector" — is verbatim the argument
that shipped the two prior instances:
- `parse/fast.rs`'s L1 hash3 gate SHIPS this exact shape today, justified as "free off the
  already-populated `Sink::litlen_freqs` histogram — no extra scan", with
  `L1_HASH3_GATE_LIT_THRESHOLD_PCT = 48` **fitted two points off a 2-point-wide cliff on
  the single file `dd79_bin6`.** `CLAUDE.md` names "the L1 hash3 content gate" among the
  things to delete.
- `parse/gated.rs` was the second, and was **deleted by user order** on 2026-07-27.

Non-negotiable #3 requires bringing any data-dependent decision to the user *first*, and
only as parameter tuning. A signal-gated split threshold is not parameter tuning. So route
(2) is blocked pending the user, and route (1) is not — which is also the cheaper of the
two to falsify.

**The failure mode to design against, for either route:** shortmatch must split early;
text must not. A bias that fires early on STRUCTURED stationary data (text at 600K) gives
back exactly the amortisation the budget raise just bought. Note that a fixed quantum
sidesteps this by not predicting anything — it splits on a count, so structured data pays
only the amortisation it would have paid at that count anyway.

State up front, before building: 600K roughly doubles `SEQ_STORE_CAPACITY`
(`parse/mod.rs:108-115`) — give the RSS delta per D2. And `continue_block` also ends
blocks at `SEQ_STORE_LENGTH` 50,000 seqs, which may bind before 600K on matchy data —
establish which bound produced the logs.txt gain before assuming the byte budget did.

Axis **size**, targets Front A via T1 slack. L1 is untouched by construction (the fast
parser never calls `should_end_block`).

**Falsifier: the promotion rule (`docs/promotion-rule.md`), both axes, cheapest first.**
Size leg first — free and deterministic: one build, full canonical corpus x L1-9 x
T1/T4, per-label against all four rivals. Clause 3 has real teeth here: this breaks
byte-identity with libdeflate at every currently-tied cell BY DESIGN, so every tie is one
bad file from a pass->fail flip. Wall leg only if size survives — this can cost wall
twice (the check itself, and more blocks means more header builds and `min_match_len`
recalcs): one paired interleaved run on the frozen box at L2/L6/L9, T1, n per clause 8.
Counterfactual written first: **if shortmatch-4M T1 at the raised budget exceeds its
300K-budget bytes with the new rule active, the recalibration failed and the whole pair
reverts — the budget raise does not ship alone.**

**A2 — thread-aware chunk grid.** Chunks of `input/(k*T)` cut chunk count 25-100x,
shrinking Front A's residual to a few hundred bytes per file. Now legal: the
T-invariance "HARD INVARIANT" at `pipelined.rs:77-90` was retracted 2026-07-28 (the
only bar is valid gzip). The prior level-scaled-grid revert divided the constant WITHOUT
slack and paid 2-4% wall from imbalance; a T-aware grid with a split tail is a different
shape and is unmeasured. Axis **size**, with a wall gate.

**B1 — Front B: `Fast` vs `deflate_compress_fastest`.** The 33 remaining cells, up to
+10.28%. Pure vendor diff, candidates already cited in the technique index: igzip's 8K
L1 head table (M1 — we widened it to 64K plus a 32K side table, and our own `fast.rs`
records that the dependent `head[h]` load is "69% of the L1 fast path's D1 read misses"
and "the IPC collapse vs igzip, 1.32 vs 2.46"); igzip's large-match emit loop (M15); its
literal-run skip (P13). Axis **size** for the cells, wall as a watch.

> ### B1 ATTEMPT 1 (2026-07-30): libdeflate's `ht_matchfinder` ported. NO-SHIP, and it
> located the real lever precisely.
>
> The vendor diff came from `fulcrum why libdeflate:data.csv:L1:T1:size` — the automated
> diff, structure layer, on 26,500,000 B: we emitted **741,183 literals to libdeflate's
> 256,099 (+189.41%)** and found **84,536 fewer matches (-4.58%)**, while header bits agreed
> within 0.37% (166,614 vs 166,000). So the L1 gap is the PARSE, not block sizing and not
> table quality. That also killed my first hypothesis — the missing L1 seq cap
> (`FAST_SEQ_STORE_LENGTH` 8192) — before it cost anything: 614 header bits cannot account
> for 1,434,580 total.
>
> Ported `vendor/libdeflate/lib/ht_matchfinder.h` at `BUCKET_SIZE 2` (2 candidates per
> position, one 128 KiB table, no length-3 table) plus `deflate_compress_fastest`, and
> routed L1 through it. `fulcrum verify`: **220 cells, 0 roundtrip failures.** Then
> `scripts/campaign/board-size.sh tune`, 4 rivals, 0 VOID
> (`~/www/gzippy-bench/campaign/size-tune-{bfd44096,htport}/`):
>
> | promotion-rule clause | result |
> |---|---|
> | 4 — fail-gap | **0.396822 -> 0.103242, -73.98%** — a large, real ratio win |
> | failing cells | 96 -> 94 (**9 closed, 7 opened**) |
> | **3 — no pass->fail flips** | **VIOLATED: 7 flips.** Clause 3 is absolute |
> | **5 — erosion budget** | **VIOLATED: armexe.elf +0.0345 against a 0.0050 budget (6.9x)** |
>
> **The 9 closed cells are all libdeflate L1 and every one lands at ratio EXACTLY 1.0000** —
> data.csv 1.0456, aozora 1.0405, minjs 1.0226, dickens 1.0211, data.json 1.0177,
> engine.wasm 1.0125, all -> 1.0000. The transliteration is faithful.
>
> **The 7 opened cells are ONE mechanism: on BINARIES we were already WINNING, and
> converging on libdeflate gave the win up.** armexe.elf T1 599,781 -> 621,027 B (it was
> **0.9658** vs libdeflate — a 3.4% win, and it also beat gzip and pigz); symbols.dwarf
> 394,736 -> 396,048; tool.bin 22,565,629 -> 22,673,676. Those wins are `fast`'s `head3`
> LENGTH-3 table, which `ht_matchfinder` deliberately lacks — libdeflate's own header says
> "Due to its focus on speed, the ht_matchfinder doesn't support length 3 matches."
>
> **SO THE LEVER IS A SYNTHESIS NO VENDOR SHIPS.** Length-3 matches earn real bytes on
> binaries; 2-way bucketing earns far more on text and structured data. libdeflate L1 has
> buckets and no hash3; our `fast` has hash3 and a single probe. REOPEN requires **both** —
> ht's 128 KiB 2-way bucket PLUS a small length-3 table, replacing the 256 KiB `head`, which
> still comes in under today's ~384 KiB. Caution, and it is the binding one: `17283ee6`
> (`c0f69036`) is the nearest prior attempt at a combination and it died on WALL with a
> 12-29% self-tax, so a combination must show its working-set arithmetic and take a frozen
> paired wall run. A size-only argument is not sufficient for that shape.
>
> `matchfinder/ht.rs` and `parse/ht_fast.rs` are KEPT compiled and unit-tested but unrouted;
> only the routing was reverted, verified by execution (L1 bytes back to main's exactly:
> armexe.elf 599,781, data.csv 4,111,742). They are the measured half of the synthesis.
>
> ### B1 ATTEMPT 2 (2026-07-30): the synthesis. SIZE LEG PASSES — 4 cells closed, 0 flips.
> **WALL LEG OWED; it is the gate, and the prior falsification in this class died on wall.**
>
> Added a length-3 singleton table to the 2-way bucket, in the shape libdeflate's OWN
> `hc_matchfinder` uses at L2-9. Working set is **192 KiB against the shipped 384 KiB —
> exactly half** (2-way `[[i16;2]; 32K]` 128 KiB + `[i16; 32K]` 64 KiB, versus `head`
> 64K x u32 256 KiB + `head3` 128 KiB); the `i16` position encoding pays for it. Both hash
> keys come from one 4-byte load, so the second table costs no extra input read. No
> data-dependent branch anywhere in it, so nothing to gate and no threshold to fit.
>
> `fulcrum verify`: **220 cells, 0 roundtrip failures.** Size census
> (`~/www/gzippy-bench/campaign/size-tune-393625ac/`, subject `393625ac`, 4 rivals, 0 VOID):
>
> | clause | result |
> |---|---|
> | 3 — no pass->fail flips | **0 flips** ✓ |
> | 4 — progress | fail-gap **0.396822 -> 0.127085, −67.97%** ✓ (needs >= 1%) |
> | 5 — erosion budget | **worst delta +0.0000** on any passing cell ✓ |
> | cells | 96 -> 92, **4 CLOSED** ✓ |
>
> **Cells closed, by name:** libdeflate L1 data.parquet T1 (1.0026 -> 0.9995) and T4
> (1.0027 -> 0.9994); libdeflate L1 movie.mp4 T1 and T4 (1.0002 -> 1.0000).
>
> **Every L1 cell improved, and the existing wins deepened** — tool.bin 0.9952 -> **0.9787**
> (22,565,629 -> 22,190,348 B, now 2.1% smaller than libdeflate), symbols.dwarf 0.9967 ->
> 0.9909, armexe.elf 0.9658 -> 0.9640. The remaining L1 shortfalls collapsed from up to
> +4.6% to at most +1.25%: data.csv 1.0456 -> **1.0006**, minjs 1.0226 -> 1.0019,
> engine.wasm 1.0125 -> 1.0022, dickens 1.0211 -> 1.0047, aozora 1.0405 -> 1.0056,
> data.json 1.0177 -> 1.0125.
>
> ### ⚠ WALL VERDICT (2026-07-30): NO-SHIP. Size passed, the WALL killed it — clause 5 + 6.
>
> `fulcrum try`, frozen solvency (AMD EPYC 7282, boost=0, governor=performance, tenants
> SIGSTOPped and restored), paired interleaved, n=9, /dev/null both arms, L1+L6, T1, full
> TUNE set, 176 cells / 145 decidable. Artifact `/root/wall-l1-synth/try.json`.
>
> | clause | result |
> |---|---|
> | 1 verify | OK — zero roundtrip failures |
> | 3 flips | OK — none across 145 decidable cells |
> | 4 progress | OK — closed `libdeflate:data.parquet:L1:T1:size`, `libdeflate:movie.mp4:L1:T1:size` |
> | **5 erosion** | **FAIL — 19 WALL cells past the 0.0050 budget** |
> | **6 net** | **FAIL — improvement 0.1593 < 2x harm 2.0388** |
>
> Self-tax at L1 (ratio vs rival, LOWER IS FASTER — we stay faster than gzip and pigz, but
> our own L1 got 15-50% slower): `gzip:data.json` 0.4549 -> 0.6861, `pigz:data.json`
> 0.6029 -> 0.9044, `gzip:data.csv` 0.4068 -> 0.5589, `gzip:tool.bin` 0.4607 -> 0.5455,
> and 15 more.
>
> **WHY THE REOPEN ARGUMENT WAS WRONG, and this is the transferable part.** The REOPEN
> rested on the working set being HALVED (192 KiB vs 384 KiB) where `17283ee6` had grown
> it. The arithmetic was correct; the inference was not. The size win comes from doing
> MORE WORK PER POSITION — two bucket candidates plus a third table read/write — and
> halving the bytes RESIDENT does not pay for the extra dependent loads ISSUED. Working-set
> size bounds cache pressure; it says nothing about load count on the critical path. This
> campaign already knew that about this exact cell: §1 records that **61% of our excess over
> libdeflate at L2 is LOAD INSTRUCTIONS**, with our IPC, stalls, cache and branch behaviour
> all already better than theirs. I predicted from bytes-resident; the wall answered on
> loads-issued.
>
> **REOPEN now needs a mechanism that adds candidates WITHOUT adding dependent loads per
> position** — igzip's two-positions-per-iteration pipeline (M/P12: probes pos and pos+1
> off ONE hash computation) and its literal-run skip (P13). A third table probed at every
> position is not it. **A size-only argument is not sufficient for this class: 2 for 2.**
>
> `matchfinder/ht.rs` (with `hash3_tab`) and `parse/ht_fast.rs` stay compiled and
> unit-tested but unrouted; routing reverted, verified by execution (L1 bytes back to
> main's exactly). They are the working half of any P12-shaped retry.

> **What the size leg alone had said, before the wall ran.** The promotion rule needs
> BOTH axes and clause 7 needs both arches. `17283ee6` (`c0f69036`) passed size and died on
> WALL at a 12-29% self-tax, 26 standard deviations — a frozen paired run on solvency
> (`root@10.0.2.240`) at L1, T1 and T>1, is the gate. The halved working set is a REASON to
> expect better than that attempt, not evidence. Do not merge on the size leg alone.
>
> P4 is unchanged by this: all 12 monotonicity violations are the pre-existing L4>L3 family
> (D1) plus two deep-level ties, identical to main's run, none at L1.

**T1 wall — converge, then delete.** Flatten the parse/matchfinder interface so one loop
owns its scalars, rather than hoisting inside the current shape (three failures). Our
hot loop threads state by reference (`in_base: &mut usize`, `next_hashes: &mut [u32; 2]`)
into an `#[inline(always)]` `longest_match`; LLVM cannot prove non-aliasing against
`buf`/`sink`, so `cutoff`/`nice_len`/`next_hashes[1]` live in memory where libdeflate
keeps them in registers. Distinguish it from the failed hoists IN ADVANCE: name the
spilled scalars, show them spilling in the current asm, predict their disappearance, and
add a wall criterion on the frozen box — Dr closing with wall flat is banking a number,
not a cell. Free oracle: output is byte-identical.

## 6. Open decisions for the user — not engineering

**D3 — does promotion-rule clause 3 apply to sub-noise SIZE ties? THIS IS THE ONE THAT
BLOCKS 60% OF THE BOARD.**

A thread-aware chunk grid (PR #189, branch `perf/t-aware-chunk-grid`) takes the board
from 223 to 203: **29 cells closed, 9 opened**, fail-gap 1.0184 -> 0.9995. It collapses
the per-chunk seam cost — our own T4-T1 at L6 goes tool.bin 2,828 -> 80 B,
weights.safetensors 2,962 -> -1 B, data.sqlite 2,258 -> 559 B. Correctness verified
through our own decoder and gzip at T4.

Clause 3 says "No pass -> fail flips. Not one." There are 9. Every one is a T4 cell whose
margin was already under 0.02%: winexe.exe L9 **+59 B on 1.5 MB**, data.csv L8 +102 B,
aozora.txt L2 +20 B, ecoli.fastq L6 +321 B.

**Five grid shapes were tried and all flip** — including a plain 1 MiB fixed grid with no
thread term at all, which flips 8. So the flips are not caused by thread-awareness or by
seam alignment; they are caused by changing the chunk size AT ALL. Those cells are won by
the SPECIFIC 512 KiB grid, not by anything robust. There is no shape with zero flips, and
looking for one is a closed search.

The question is therefore not "can this be engineered around" — it cannot — but "is a
59-byte regression on 1.5 MB a pass->fail flip for the purposes of a size gate". Both
answers are legitimate and both unblock work:

* **"Clause 3 stands."** Record it, close the chunk-grid class permanently, and redirect
  to the 90 fail-at-both cells (34 at L1, ~22 at L4/D1). PR #189 gets closed with the
  falsification kept.
* **"Sub-noise size ties are not flips."** Then per `docs/promotion-rule.md` the rule
  change lands SEPARATELY and FIRST, with its own symmetry/timing/backtest justification,
  and only then is #189 re-evaluated against it.

It is deliberately NOT decided here. The rule forbids a rule change authored by the
session whose result it would rescue, and this is that session.


**D1 — P4 monotonicity at L4 (board #49).** `-4` yields a bigger file than `-3`
(+143,807 B on silesia 40 MB). Lazy at L4 fixes it and beats all four rivals on size by
1.2-3.1%, at **17.7% wall**. Greedy cannot fix it at any depth — parse strategy
dominates depth. Promotion rule clause 5 caps erosion on a passing cell at 0.5%, so it
is blocked without a pre-registered carve-out landed separately and first. libdeflate
ships the same violation.
*Possible third path, unmeasured:* `vendor-structure-comparison.md` §1 claims gzip and
zlib-ng both go lazy at L4. **Wrong for zlib-ng as shipped** — its default L4 is
`deflate_medium` (the `deflate_slow` row is inside `#ifdef NO_MEDIUM_STRATEGY`).
`deflate_medium` exists to make L3-6 monotonic at a fraction of lazy's cost and may fit
inside clause 5 with no law change. Fix that doc claim and measure this first.

**D2 — RSS as a graded axis.** T>1 holds ~2.6-3x input plus ~5 MB/thread against pigz's
flat 2 MB. Named user-visible, never given a number. Any Front A design that queues more
state needs the bound stated first.

## 7. Known-stale claims — fix, do not trust

- `vendor-structure-comparison.md` §1 — the zlib-ng L4 claim above.
- `encoder-architecture.md:17-23` still calls byte-identity with libdeflate "a
  contract". Retracted 2026-07-28; the only bar is valid gzip.
- `encoder-architecture.md` "structural suspects, in order" (indirection, prefetch,
  bounds checks) predates the operation-level verdict and is effectively falsified for
  the L2 band. The debt is loads and register pressure at the parse/matchfinder
  interface.
- `pipelined.rs:77-90` presents T-invariant bytes as a "HARD INVARIANT". A product
  choice at most, and it is what forbids A2.
- `vendor-structure-comparison.md` §2 omits igzip: it bounds blocks by ICF TOKENS,
  65,536 x 4 B at CLI default (`igzip_lib.h:296,310-315`), with per-block hufftables
  (`igzip.c:406`) and a stored-block cost compare (`igzip.c:365-416`).
- ~~`parse/fast.rs:675` still reads `GZIPPY_L1TUNE_BLOCK_LENGTH` … B1's measurements are
  meaningless while a knob can vary the shape.~~ **HALF WRONG, corrected 2026-07-30.** The
  source is real but the whole `tune` module is `#[cfg(feature = "l1-tune")]` and the
  feature is default-off (`Cargo.toml`), so it is **compiled out of the shipped binary** —
  verified: `strings target/release/gzippy | grep -c L1TUNE` = **0**. No knob can vary the
  shape of what ships, so B1's measurements are NOT compromised and B1 is not blocked on a
  deletion. The deletion is still worth doing, for two better reasons: the `tune` module is
  22 env knobs plus a `RwLock<L1Tune>` read per `run()` behind a build flavour its own note
  records at **1.1702x slower than what ships** — and `CLAUDE.md`'s measurement rule
  forbids quoting a tuned build against a rival, which is what the entire L1 tuning history
  did. Its driver `fulcrum l1search` was already deleted as constitutionally banned. Knob
  search has produced **zero** of this project's counted wins.
- The `gated.rs` doc comment orphaned by the 2026-07-27 user-ordered deletion survived in
  `parse/mod.rs` until 2026-07-30, dangling on `mod greedy;` and still advertising
  `GZIPPY_L3TUNE_GATE_*` env vars and "a `--tune`-style channel + `fulcrum l3search`" as "a
  real, named, un-taken next step". Deleted. A stale comment proposing forbidden work is
  not inert documentation — it is an instruction to the next session.
- Board #50 overstates its case: per-chunk overhead owns about half the T4 board, not
  103 cells. §2 has the split.

## 8. Tooling

Three git hooks, sourced from `scripts/` and installed by `build.rs` into the effective
hooks dir: `pre-commit` (fmt/clippy/version), `pre-push` (refuses direct pushes to
main), `commit-msg` (FALSIFY/`REOPEN:` discipline; an absolute figure and provenance
beside any quoted ratio). **Do not set `core.hooksPath`** — it overrides `.git/hooks`
and silently disabled the first two for part of 2026-07-29.

`commit-msg` is crude: it matches proximity to a FALSIFY note, not semantics. 3 correct
refusals, 1 incorrect (it blocked a legitimate revert until `RESTORE:` was added as an
escape). Expect misfires and fix it rather than routinely using `--no-verify`.

**How to measure anything — USE THE COMMITTED ENTRY POINT, and never hand-roll.**

```
scripts/campaign/board-size.sh [tune|gate|all]      # the cheapest falsifier; run this first
CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all   # the promotion board
```

`scripts/campaign/lib.sh` is now the ONE definition of the measurement surface, and each of
its guards cites the incident it prevents. It refuses to run when the corpus set contains an
undeclared member, when any of the four rivals is missing without a written reason, when the
fulcrum binary is dirty or unidentified, and when GATE files are reached for outside a
declared promotion. **Every one of those four refusals corresponds to something that
actually happened and got quoted as a result** — see the file's header comment for the
receipts.

Why this exists: before it, every census was an uncommitted shell script in
`~/www/gzippy-bench`, which **is not a git repository** (eight such scripts, zero version
control), each encoding its own corpus and its own rival set. That is the mechanism behind
the unreconciled denominators in §1 and §2, the missing igzip row, and the two undeclared-
corpus falsifications in §4. `corpus_split.json` — the TUNE/GATE contract that decides
whether a promotion is even valid — lived in the *fulcrum* repo but not beside the code it
grades; it is now committed here.

The deeper form of this fix, not yet done and worth doing: make the board a GENERATED
artifact and delete §1/§2/§4's hand-typed tables in favour of a link to it. Add
`sha256` + `bytes` + synthetic-provenance per member to `corpus_split.json` and refuse on a
sha mismatch, which is what would make "a gate may only cite a dataset that exists"
mechanically true rather than aspirational. Acceptance test for that change: **no number in
this document that a fulcrum command can compute.** Today nearly every number in §1-§4 is a
hand-typed fork of tool output, which is precisely why 20 levers cost 20 bespoke
measurement designs.

Fulcrum already closes the loop, and this plan did not reference it once before 2026-07-30:
`board` (where do we stand) -> `why <cell>` (the automated vendor diff) -> `candidates
<cell>` (vendor-precedented techniques, with FALSIFY records surfaced) -> `try <ref>`
(the promotion rule applied clause by clause, both arms built from git refs, no-ops and
single-level verdicts REFUSED) -> `board`. **§5's hand-written falsifier paragraph is a
re-description of `fulcrum try`.** Prefer the command.

Size is deterministic and arch-invariant, so it needs one box; any file that gets bigger
kills the change. Wall needs the frozen box, paired, with an A/A certificate. Never quote a
ratio without the absolute figure beside it.

Wall verdicts come from the frozen box (solvency, AMD Zen2). Size is bit-identical
across aarch64/Zen2/Intel — verified — so size needs one box only. Neither remote box
currently passes the full Gate-0 suite (`profile rss` fails on both, `lib levelsweep` on
solvency) and `make deploy` correctly refuses to certify them; fix those two gates
before trusting a fresh instrument there.

## L1 IS THE LARGEST FAILING CLASS, AND IT IS NOT THE SEAM (measured 2026-08-01)

The board had been decomposed by rival x thread count, which surfaced the T4 seam
(109 zero-headroom cells). Decomposing the SAME artifact by **level x rival** —
`/root/sizeboard-all-12fcd0ed/census.json`, 1584 cells, 200 failing, commit
`12fcd0ed` — shows a second class that is larger per level and far more tractable:

```
        gzip     pigz  libdeflate  igzip   tot
  L1       2        0          29      4    35     <- largest level on the board
  L2       4        2          17      0    23
  L3       4        2           4      2    12
  L4       2        2          13      0    17
  L5       4        4          17      0    25
  L6       6        6          13      0    25
  L7       6        4          16      0    26
  L8       2        0          15      0    17
  L9       2        0          18      0    20
```

**L1 has none of the seam class's pathologies.** Its cells fail at T1 and T4 with
near-identical ratios (access.log 1.1028 / 1.1032; monorepo.tar 1.0565 / 1.0565), so
this is a PURE CODING DEFICIT, not seam growth. And where the seam cells tie
libdeflate byte-for-byte with 0 bytes of headroom, these are 0.02%-10.3% BIGGER:

```
  libdeflate T1/T4  access.log    1.1028 / 1.1032   +340,410 / +341,529 B
  libdeflate T1/T4  monorepo.tar  1.0565 / 1.0565   +639,158 / +639,359 B
  libdeflate T1/T4  data.csv      1.0456 / 1.0457   +179,323 / +179,667 B
  libdeflate T1/T4  aozora.txt    1.0405 / 1.0405   +184,853 / +185,069 B
  libdeflate T1/T4  ecoli.fastq   1.0282 / 1.0283   +135,198 / +135,773 B
  ... 24 more, down to movie.mp4 1.0002
```

Two consequences. Clause 3 cannot be tripped by improving an already-FAILING cell,
so the zero-tolerance constraint that governs the tie cage does not apply here. And
the margins are three orders of magnitude larger than the ~0.01% the seam needs — so
unlike the seam, a partial improvement CLOSES CELLS.

### The mechanism is already identified, and it is already in the tree

We run **igzip's** L1 algorithm (`Strategy::Fast`: chainless, single probe, plus a
length-3 `head3` table) while being graded against **libdeflate's** L1
(`deflate_compress_fastest` + `ht_matchfinder`: 2-entry buckets, no length-3).
`parse/mod.rs`'s two FALSIFY records already proved this is THE mechanism by
execution: routing L1 to `ht_fast` lands nine libdeflate L1 cells at ratio EXACTLY
1.0000 (data.csv 1.0456->1.0000, aozora 1.0405->1.0000, minjs 1.0226->1.0000,
dickens 1.0211->1.0000, data.json 1.0177->1.0000, engine.wasm 1.0125->1.0000).

Both prior attempts died, and NEITHER died on size:

- **attempt 1**, route as a REPLACEMENT: 9 closed / 7 OPENED. Clause 3 violated. The
  7 are one mechanism — on BINARIES our `head3` length-3 table beats libdeflate and
  the port gives that win up (armexe.elf was a 3.4% WIN at 0.9658).
- **attempt 2**, the SYNTHESIS (2-way bucket AND length-3, which `matchfinder::ht`
  still IS today — only the routing was reverted): **clause 3 OK across 145
  decidable cells, clause 4 closed 2, the SIZE LEG PASSED CLEANLY.** It died on
  clause 5/6: 19 wall cells eroded, our own L1 15-50% slower.

### What is untried: the COORDINATE

Attempt 2's wall verdict was taken at **`L1+L6, T1`** (artifact
`/root/wall-l1-synth/try.json`). Every clause-5 erosion it reports is a T1 ratio
against a SINGLE-THREADED rival — gzip:data.json 0.4549 -> 0.6861, pigz:data.csv
0.5444 -> 0.7418. At T4 those same cells run our 4 threads against their 1, where
measured slack is 249-330% rather than T1's 0-8%; the same 15-50% self-tax erodes
roughly a quarter as much in ratio terms. That is the 40x coordinate error this
project has already made once, and the board says half the L1 cells are at T4.

There is shipped precedent for the fix shape: `try_exact_huffman` and the
`max_search_depth` x4 scaling are BOTH gated T>1-only for exactly this reason, in
`level.rs`'s own words — "THE REASON IS THE WALL BUDGET ... It is therefore T>1
ONLY". A T>1-only L1 routing is the same move on a change whose size leg has already
passed.

**This is NOT the size-only argument the record forbids** ("a size-only argument is
not sufficient for this class: that is now 2 for 2"). The claim is about the WALL, at
a thread count that verdict never measured.

### Pre-registered rule, declared once, before any measurement

Route `Strategy::Fast` to the `ht_fast` synthesis from `params_parallel` ONLY (T>1),
leaving T1 byte-unchanged. Judged by `fulcrum try --threads 1,4`, full TUNE set,
L1 AND a deep level (hard stop #3), frozen box, vanilla build:

- SHIP iff clauses 1-6 all pass, INCLUDING clause 3 at T1 (which must be a no-op —
  verify the T1 arms are byte-identical) and clause 5 at T4.
- NO-SHIP if clause 5 fails at T4. That would mean the parallel budget does not
  absorb the self-tax either, and this class is then closed on the wall at BOTH
  coordinates — record it and stop, do not re-sample.

Cheapest falsifier first: run the SIZE leg at T1,T4 before any wall run. If the T4
size win is not materially larger than attempt 2's 2 cells, the lever is not worth
the wall risk and is dropped without a frozen-box run.

### ⚠ DO NOT TREAT THE MATCHFINDER AS THE SOLE EXPLANATION

The section above reaches for `ht_matchfinder` because a FALSIFY record handed that
mechanism over ready-made. But that record proves the routing CLOSES CELLS; it does
not prove the matchfinder CAUSES the whole L1 deficit, and two things say it does not:

**The nine cells attempt 1 closed do not include the two worst.** The record names
data.csv, aozora.txt, minjs.min.js, dickens, data.json and engine.wasm. The largest
L1 deficits on the board — access.log at +340,410 B (1.1028) and monorepo.tar at
+639,158 B (1.0565) — are NOT among them. Whatever costs us 340 KB on access.log
survived the matchfinder swap.

**The spread is too wide for one mechanism.** L1 deficits run from +10.3%
(access.log) to +0.02% (movie.mp4). A single cause producing both, on the same
algorithm, is not credible; there are at least two.

Three explanations that have NOT been excluded, each with its falsifier:

1. **BLOCK GEOMETRY, not match finding.** Our L1 block budget
   (`fast::FAST_BLOCK_LENGTH`, `LIMIT_HASH_UPDATE_INSERTS_L1`) vs libdeflate's
   `FAST_SOFT_MAX_BLOCK_LENGTH` 65,535 / `FAST_SEQ_STORE_LENGTH` 8,192. Different
   block sizes change header amortisation directly. `parse/fast.rs:1549` records
   "do NOT fix this to 65,535 to match libdeflate" — so the constant has been
   TOUCHED, which is not the same as EXONERATED as a cause.
   FALSIFIER: `examples/blockcensus` on our L1 output vs libdeflate's — block count
   and bits/block.

2. **THE PER-BLOCK BTYPE DECISION, not the parse.** Our L1 costs
   cheapest-of-{dynamic,static,stored} per block. If we choose dynamic where they
   choose static we pay a header they never emit, which in aggregate bytes is
   indistinguishable from a worse parse.
   FALSIFIER: the same `blockcensus` run — the BTYPE mix. One command tests 1 and 2.

3. **INSERTION/SKIP POLICY, not probe count.** libdeflate's fastest calls
   `ht_matchfinder_skip_bytes` after every match; we use
   `LIMIT_HASH_UPDATE_INSERTS_L1`. That changes what HISTORY later positions can
   see, so "2-way buckets win" may really be "their insert policy retains better
   history" — with the bucket taking credit for the skip.
   FALSIFIER: match/literal counts and mean match length per byte.

**THE DISCRIMINATING COMMAND IS `fulcrum why`, AND IT WAS NOT RUN.** Hard stop #6
lists it first precisely for this: it reports match/literal/header/data per byte and
states which of its four layers it skipped, naming the mechanism in ONE run.

    fulcrum why libdeflate:access.log:L1:T1 --ours <bin> \
        --rival-cmd 'libdeflate-gzip -{level} -c {input}' --corpus access.log

Run that, and `blockcensus` on both outputs, BEFORE the routing lever. If the bytes
are in headers rather than data, explanations 1-2 are the class and the matchfinder
is the wrong lever entirely.

### ⚠⚠ CORRECTION TO BOTH SECTIONS ABOVE — access.log AND monorepo.tar ARE **GATE** FILES

`corpus_split.json` splits the corpus and states the contract: *"GATE files are run
ONLY at promotion time, by the census/goal tools, and NEVER inspected while choosing
a parameter. A promotion is judged on GATE. If a change was fitted on GATE, the
promotion is void regardless of the numbers."*

```
GATE: access.log data.sqlite dd79_bin6 dd79_text6 ecoli.fastq markup.xml
      monorepo.tar photo.jpg sil40 weights.safetensors winexe.exe
TUNE: aozora.txt armexe.elf data.csv data.json data.parquet dickens
      engine.wasm minjs.min.js movie.mp4 symbols.dwarf tool.bin
```

**access.log and monorepo.tar — the two cells the section above builds its whole
argument on — are GATE.** Two consequences, and the second one retracts a claim:

**(a) The diagnostic as written would VOID the promotion.** The section above
prescribes `fulcrum why libdeflate:access.log:L1:T1` to decide which lever to build.
That is inspecting a GATE file while choosing a parameter. It is exactly the failure
`_rule` in `corpus_split.json` was written to prevent ("parameters were once fitted
to the one file blocking a gate ... BOTH later blowups landed off the tuning set").
**Run the diagnostic on TUNE members only** — data.csv (1.0456), aozora.txt (1.0405)
and dickens (1.0211) are the largest L1 deficits that are legal to inspect.

**(b) "The two worst cells survived the matchfinder swap" is RETRACTED.** It is
unsupported. Attempt 1's record states its own coordinate — `board-size.sh tune`,
**TUNE x L1-9 x T1,T4** — and access.log and monorepo.tar are GATE, so they were
NEVER IN THAT MEASUREMENT. All six files it names as closing (data.csv, aozora.txt,
minjs.min.js, dickens, data.json, engine.wasm) are TUNE members. The swap did not
fail on the two largest deficits; it was never tested against them.

So the matchfinder explanation is STRONGER than the previous section allowed: on the
TUNE set it closed essentially every libdeflate L1 cell available to it. The three
alternative explanations (block geometry, the per-block BTYPE decision, insert/skip
policy) remain UNEXCLUDED and still need `fulcrum why` — but they no longer have the
"two worst survived" evidence behind them, because that evidence does not exist.

**THE GENERAL LESSON, which is the reusable part:** a cell's TUNE/GATE membership is
part of its coordinate, and a measurement's corpus SUBSET is part of its result. Two
errors here came from reading "9 cells closed" as a statement about the board when it
was a statement about TUNE. Before citing any cell as evidence, check which set it is
in; before citing any prior measurement, read the corpus it ran on.

### MEASURED: the L1 deficit is MATCH COVERAGE. Headers are NOT the cause.

`fulcrum why libdeflate:<file>:L1:T1:size`, three TUNE members, trainer (Intel LXC),
gzippy `8d948cef` sha `f7a53025`, fulcrum `8364a059`, corpus sha-verified against
solvency. Solvency was untouched — it was holding a paired wall gate.

```
dickens     ours 3,133,556 tok (1,923,158 M, 1,210,398 L)  40,619,097 b (117,739 hdr)
            rival 2,827,981 tok (2,025,030 M,   802,951 L)  39,781,354 b (154,454 hdr)
data.csv    ours 2,587,312 tok (1,846,129 M,   741,183 L)  32,893,788 b (166,614 hdr)
            rival 2,186,764 tok (1,930,665 M,   256,099 L)  31,459,208 b (166,000 hdr)
aozora.txt  ours 2,997,325 tok (1,683,389 M, 1,313,936 L)  38,009,450 b (111,502 hdr)
            rival 2,678,110 tok (1,735,132 M,   942,978 L)  36,530,632 b (123,678 hdr)

  file        literals Δ    matched-positions Δ    header (ours vs rival)
  dickens       +50.74%           -3.58%           117,739  <  154,454
  data.csv     +189.41%           -1.85%           166,614  ~= 166,000
  aozora.txt    +39.34%           -3.35%           111,502  <  123,678
```

**Explanations 1 and 2 are REFUTED.** Our header mass is SMALLER than libdeflate's on
two of three files and equal on the third — we are already AHEAD on headers by 36,715
bits on dickens and 12,176 on aozora. Block geometry and the per-block
dynamic/static/stored decision cannot be the cause of a deficit we are winning.

**The whole deficit is DATA bits, and the mechanism is literal emission.** dickens
total delta 837,743 bits = 104,718 B, which reproduces the census excess for that
cell EXACTLY (+104,718 B) — an independent confirmation that this diff explains the
whole cell and not a fraction of it. We emit 39-189% more literals because libdeflate
matches at 1.85-3.58% more POSITIONS. Every position we fail to match costs a literal.

**And we carry an EXTRA table while matching less.** `parse::fast` has a length-3
`head3` table that libdeflate's `ht_matchfinder` deliberately does not ("Due to its
focus on speed, the ht_matchfinder doesn't support length 3 matches"), and we still
find FEWER matches. The single-probe limitation dominates the length-3 advantage.
That is a size argument for the bucket independent of the earlier routing attempts.

**Live explanation, narrowed to one class:** whatever raises match COVERAGE at L1 —
the 2-way bucket (more history per slot ⇒ more candidates) and/or the insert/skip
policy (`ht_matchfinder_skip_bytes` vs our `LIMIT_HASH_UPDATE_INSERTS_L1`, which
changes what history later positions can SEE). These are not yet separated from each
other; both are coverage mechanisms and the diff above cannot distinguish them.

DENOMINATOR, as the tool reports it: **2 of 4 layers ran.** [3 COUNTERS] skipped (the
gzip oracle exited 1 on this box) and [4 PARAMS] skipped (the vanilla binary emits no
`LEVEL_DECLARED`; that needs `--features anatomy-counters`, which must NOT be the
binary any wall claim is quoted from). No claim here rests on those two layers — this
is a SIZE/structure finding only, and no wall claim is made.

Scope: L1 only, three TUNE files. Hard stop #3 forbids generalising across levels —
do not read this as a statement about L2-L9, which have their own (tie-cage) shape.

### ABLATION: routing L1 to `ht_fast` OVERSHOOTS — and the residual is LENGTH-3 MATCHES

Branch `measure/l1-htfast-ablation` (MEASUREMENT ONLY, never merged), binary sha
`19da2d1a` vs main's `f7a53025`, trainer, dickens L1 T1, `fulcrum why`:

```
                  literals   matched-pos/B   total_bits/B   match_len_L00/B
  main (fast)    1,210,398      0.900579       3.336403         --
  ablation (ht)    430,527      0.964637       3.282904      0.018815
  libdeflate       802,951      0.934047       3.267591      0.000000
```

**Pre-registered prediction 1 held DIRECTIONALLY BUT OVERSHOT.** Literals did not
approach libdeflate's 802,951 — they blew past it to 430,527, and we now match at MORE
positions than libdeflate (0.9646 vs 0.9340).

**Pre-registered prediction 2 FAILED, and that is the finding.** Total bits fell
81,416 B of the 104,718 B gap. We remain **+23,302 B BIGGER while emitting MORE
matches and FEWER literals.** More coverage is not the same as smaller output.

**The residual is named by the tool: `match_len_L00`, ours 0.018815/B vs rival
0.000000.** `len_code_index` (fulcrum `src/ratio/mod.rs:242`, base[0] = 3) makes that
bucket EXACTLY length-3 matches — 229,064 of them on dickens, against libdeflate's
zero. `ht_matchfinder` has no length-3 support at all; our port added a `hash3_tab`
beside it. Those length-3 matches REPLACE literals AT A NET LOSS on text.

Magnitude check (counted, then compared): 229,064 length-3 matches losing ~0.8 bits
each against the ~3 literals they displace is ~22,900 B, against an observed residual
of 23,302 B. The length-3 matches plausibly ARE the whole remaining gap.

**THIS EXPLAINS BOTH PRIOR ATTEMPTS WITH ONE MECHANISM**, which neither record could
do on its own:
  - attempt 1 = the FAITHFUL port (buckets, NO hash3). Text lands at ratio EXACTLY
    1.0000 — because that IS libdeflate's algorithm. Binaries lose, because length-3
    matches genuinely pay there.
  - attempt 2 = port + hash3. Binaries kept, but text no longer reaches 1.0000 —
    because the length-3 matches it keeps cost bytes on text.

**We already apply the too-far rule.** `HT_MAX_LEN3_OFFSET = 4096` in
`matchfinder/ht.rs:149`, guarding `length > DEFLATE_MIN_MATCH_LEN || offset <= 4096`
— the same constant as gzip's `TOO_FAR` (`vendor/gzip/deflate.c:130`, "Matches of
length 3 are discarded if their distance exceeds TOO_FAR"). So all 229,064 are ALREADY
within 4096 and still net-negative on text. The threshold was inherited, never tuned
for L1, and no FALSIFY record covers it.

⚠ **BEFORE ANY LEVER HERE: an outstanding USER-ORDERED DELETION is implicated.**
`CLAUDE.md` non-negotiable #3 orders the L1 hash3 content gate deleted; it is still
present as `L1_HASH3_GATE_LIT_THRESHOLD_PCT = 48` (`parse/fast.rs:945`). The
measurement above explains WHY that gate was ever written: hash3 at L1 pays on
binaries and costs on text, and a literal-fraction detector is exactly how that split
gets papered over. The working rules say no new lever starts while a user-ordered
deletion sits undone — and a content detector is precisely the wrong answer to a split
this measurement now describes without one.

Scope: L1, dickens, T1. NOT generalised — the binaries claim (armexe.elf,
symbols.dwarf, tool.bin) is inherited from attempt 1's record and has NOT been
re-measured here.

### L1_HASH3_MAX_DIST 4096: real, monotone on text, and ~1.4% of the gap — PARKED

`L1_HASH3_MAX_DIST` (`parse/fast.rs`) ships as `WINDOW` (32,768), i.e. the length-3
profitability gate NEVER rejects on distance — while its own doc comment states the
cost model that justifies rejecting ("a length-3 match at a far distance often costs
more bits than 3 literals"). gzip's `TOO_FAR` is 4096 (`vendor/gzip/deflate.c:130`)
and our OWN sibling path uses 4096 (`matchfinder/ht.rs:149`, `HT_MAX_LEN3_OFFSET`).

Measured, local arm64, vanilla release, L1, TUNE members only:

```
  file          WINDOW      4096        delta
  dickens      5,080,065  5,078,617    -1,448
  aozora.txt   4,751,582  4,750,647      -935
  data.csv     4,112,512  4,111,870      -642
```

Smaller on all three. **But it is ~1.4% of the deficit** (-1,448 B against dickens'
+104,718 B), so it does not close cells. PARKED, not deleted, per the rule that
monotone work is parked rather than discarded — it composes.

REOPEN basis, recorded for the next reader: the binding FALSIFY on this constant
(`parse/fast.rs`, 2026-07-25) measured `32768 -> 0` (FULL hash3 shutoff) and its
blocker is that 0 destroys `dd79_bin6`'s pigz-1 size win (0.997516 -> 1.040685). That
establishes 0 is fatal; it does NOT establish 4096 is, and 4096 is the intermediate
both gzip and our own `ht` path use. Note `dd79_bin6` is a GATE member — that check
belongs at promotion time, never while choosing the value.

### ⚠ SIZE IS NOT STRICTLY ARCH-INVARIANT (~0.03%), and a stale binary nearly said 1.66%

Same commit, vanilla release both arms, L1:

```
  file          arm64 (M1)   x86 (trainer)   delta
  dickens        5,080,065     5,081,832     1,767 B  (0.035%)
  aozora.txt     4,751,582     4,751,843       261 B
  data.csv       4,112,512     4,114,243     1,731 B
```

Small, but NOT zero — which corrects an "arch-invariant" claim made earlier in this
campaign, and it matters for the TIE CAGE: cells that tie libdeflate BYTE-FOR-BYTE on
x86 need not tie on arm64, and `CLAUDE.md` STEP 1 requires both arches. The board is
measured on x86 only.

**HOW THIS WAS ALMOST REPORTED AS A 1.66% ARCH DIVERGENCE.** The first comparison ran
trainer's `target/release/gzippy` after a `git checkout origin/main` WITHOUT a
rebuild, so the binary was still the `measure/l1-htfast-ablation` build (sha
`19da2d1a`, confirmed identical to the saved `/root/gzippy-htfast`). It reported
dickens x86 at 4,997,100 — and 5,080,065 - 4,997,100 = 82,965 B, which reproduces the
ablation's own measured 81,416 B gap closure almost exactly. The tell was that the
result was implausibly good, and the disconfirmation was structural: the ONLY
arch-divergent code in the L1 path is `prefetch_write` (`matchfinder/common.rs:144`),
a pure hint that cannot change output bytes. Verify the BINARY, not the checkout.

### FALSIFIED: no single length-3 distance threshold closes L1. The text LOSS and the binary WIN are the SAME bytes.

Hypothesis (mine, from the ablation above): the length-3 matches that cost bytes on
text are the FAR ones, so one global `HT_MAX_LEN3_OFFSET` should drop those while
keeping the near ones that pay on binaries. **Measured and false.**

Sweep on `measure/l1-htfast-ablation` (L1 routed to `ht_fast`), local arm64, vanilla
release, L1, TUNE members only. Ratios vs `libdeflate-gzip -1` (<= 1.0 PASSES; the
libdeflate references were verified equal to the board census on all five files):

```
  thresh    dickens   aozora.txt   data.csv   armexe.elf   symbols.dwarf
   4096     1.00484    1.00568     1.00055     0.96437       0.98883
   1024     1.00292    1.00426     0.99875     0.96962       0.98814
    256     1.00181    1.00271     0.99847     0.97775       0.98861
     64     1.00100    1.00142     0.99884     0.98690       0.99041
      0     1.00016    1.00014     1.00002     0.99952       0.99988
```

**Text improves monotonically as the gate tightens and NEVER crosses 1.0** — 1.00484
-> 1.00016 on dickens, still failing at full length-3 shutoff. **And the binary win
collapses in lockstep**: armexe.elf 0.96437 -> 0.99952, a 3.6% win reduced to 0.05%.
At threshold 0 all five files converge to ~1.0000: we simply become libdeflate, which
is attempt 1's result reached from the other direction.

**THE STRUCTURAL FACT: the text loss and the binary win are the SAME BYTES.** Both are
length-3 matches at distance <= 4096. There is no distance at which one is present and
the other is not, so no single global constant separates them. This is not a tuning
failure; it is the shape of the problem.

That is exactly why a CONTENT GATE was written here
(`L1_HASH3_GATE_LIT_THRESHOLD_PCT = 48`, `parse/fast.rs:945`) — it is the only device
that separates these two populations, and `CLAUDE.md` non-negotiable #3 forbids it AND
orders it deleted. So this class is genuinely hard rather than merely unattempted: the
one mechanism that resolves the tradeoff is banned, on purpose.

What the sweep DOES establish, and is worth keeping:
  * `HT_MAX_LEN3_OFFSET = 256` passes 3 of these 5 cells against 4096's 2 (data.csv
    1.00055 -> 0.99847 crosses), so the inherited 4096 is not the best constant for
    this path even though no constant closes the class.
  * The length>=4 candidates that `head3` surfaces contribute almost nothing: at
    threshold 0, where only length-EXACTLY-3 is rejected, every file sits within
    0.02% of libdeflate.

NOT a wall claim, and NOT a promotion proposal: the `ht_fast` routing this sweep runs
on is already NO-SHIP twice (clause 3 on binaries; clause 5/6 on the wall at T1).
Scope L1, five TUNE files, one arch.

### CORRECTION + RESULT: on the FULL TUNE set, `ht_fast` @ 256 doubles the passing L1 cells (4 -> 8), zero regressions

The falsification above was measured on FIVE files and was too strong. Re-run on all
ELEVEN TUNE members, L1, ratio vs `libdeflate-gzip -1`, PASS decided by exact integer
compare (ours <= rival), `*` = PASS:

```
shipped-fast   pass= 4/11  aozora 1.04036  armexe .96952* data.csv 1.04563  data.json 1.01408
                           data.parquet 1.00266  dickens 1.02130  engine.wasm 1.00934
                           minjs 1.01698  movie.mp4 .99986* symbols .99497* tool.bin .99223*
ht_fast@4096   pass= 5/11  aozora 1.00568  armexe .96437* data.csv 1.00055  data.json 1.01302
                           data.parquet .99890* dickens 1.00484  engine.wasm 1.00304
                           minjs 1.00228  movie.mp4 .99999* symbols .98883* tool.bin .97879*
ht_fast@256    pass= 8/11  aozora 1.00271  armexe .97775* data.csv .99847* data.json 1.00358
                           data.parquet .99939* dickens 1.00181  engine.wasm .99951*
                           minjs .99671* movie.mp4 .99994* symbols .98861* tool.bin .97906*
ht_fast@0      pass= 3/11  aozora 1.00014  armexe .99952* data.csv 1.00002  data.json 1.00051
                           data.parquet .99962* dickens 1.00016  engine.wasm 1.00061
                           minjs 1.00027  movie.mp4 1.00003  symbols .99988* tool.bin 1.00011
```

**`ht_fast` @ 256 doubles the passing cells, 4 -> 8, and regresses NOTHING.** All four
cells the shipped path passes (armexe.elf, movie.mp4, symbols.dwarf, tool.bin) are
retained; four more cross: data.csv 1.04563 -> 0.99847, data.parquet 1.00266 ->
0.99939, engine.wasm 1.00934 -> 0.99951, minjs.min.js 1.01698 -> 0.99671. That is
consistent with attempt 2's record, which also found clause 3 OK — but with the tuned
threshold it closes FOUR cells on this set instead of two.

**What was wrong with the five-file falsification, precisely:** the claim "no single
global threshold closes L1" is still true in the strict sense — aozora (1.00271),
dickens (1.00181) and data.json (1.00358) fail at every threshold tried. But I let
that become "the class will not move", and the five-file sample simply did not contain
the four files that flip. A negative result on a SUBSET is not a negative result on
the class. The tradeoff between text loss and binary win is real; it just is not
total.

Note the curve is not monotone and 0 is NOT the optimum: at 0 we converge to
libdeflate everywhere (3/11, worse than shipped's 4/11 because we give up the binary
wins), and at 4096 the near-length-3 matches are too permissive on text. 256 sits at a
genuine interior optimum, and engine.wasm/minjs.min.js pass ONLY there.

STATUS AND WHAT IS STILL MISSING — this is the SIZE case only:
  * The `ht_fast` routing remains NO-SHIP on the WALL (attempt 2, clause 5/6, measured
    at `L1+L6, T1`). Nothing here changes that; the untried coordinate is T>1, whose
    plumbing (`params_parallel`) is in the unmerged PR #227.
  * 256 was FITTED ON TUNE, which is what TUNE is for. The promotion must be judged on
    GATE (`corpus_split.json`), which has NOT been inspected and must not be until
    promotion time.
  * Local arm64, deterministic byte counts, one arch, L1 and T1 only. Size is NOT
    strictly arch-invariant (~0.03%, measured above), and several of these cells pass
    by less than that — `engine.wasm` 0.99951 and `movie.mp4` 0.99994 are inside the
    arch delta. Those specific cells must be re-measured on the frozen box before any
    claim that they close.

### x86 CONFIRMATION AT T1 **AND** T4: 10 board cells close on SIZE, 0 open

Re-run on trainer (Intel x86, the same arch family the board is measured on), byte
counts, both thread counts. **The `shipped` column reproduces the board census
EXACTLY** — aozora 4,751,200; data.csv 4,111,742; dickens 5,077,406; engine.wasm
426,271; minjs 1,211,746; movie.mp4 12,903,670; data.parquet 14,424,479 — so these are
the actual board cells, not a lookalike.

```
=== L1 T1 ===                                  === L1 T4 ===
file           libdeflate    shipped     ht256      shipped     ht256
aozora.txt      4,566,347  4,751,200  4,578,206   4,751,416  4,578,509   fail->fail
armexe.elf        621,027    599,781    607,417     600,310    607,211   PASS->PASS
data.csv        3,932,419  4,111,742  3,926,359   4,112,086  3,926,566   CLOSES
data.parquet   14,386,710 14,424,479 14,383,856  14,426,706 14,383,721   CLOSES
dickens         4,972,688  5,077,406  4,981,078   5,078,221  4,981,395   fail->fail
engine.wasm       421,013    426,271    420,714     425,688    420,809   CLOSES
minjs.min.js    1,184,930  1,211,746  1,180,722   1,211,684  1,181,042   CLOSES
movie.mp4      12,901,167 12,903,670 12,899,895  12,903,821 12,900,349   CLOSES
symbols.dwarf     396,048    394,736    391,698     394,830    391,539   PASS->PASS
                            2/9    ->    7/9        2/9    ->    7/9
```

**Five files close at T1 and the same five at T4 = 10 of the 200 failing board cells,
with ZERO opened.** It holds at T4, which is the coordinate where half the L1 board
fails and where the two prior attempts never measured.

The arm64 caveat from the previous section is RESOLVED in the right direction and was
warranted: on arm64 the shipped path passed `movie.mp4` (0.99986) while on x86 it
FAILS (12,903,670 > 12,901,167). That is exactly the ~0.03% arch sensitivity flagged
earlier, landing on exactly the cell flagged. x86 is what the board uses, so x86 is the
number that counts, and it is the STRONGER of the two (2/9 -> 7/9, versus 4/11 -> 8/11
on arm).

EROSION TO WATCH AT PROMOTION (clause 5, not a flip): `armexe.elf` stays PASS but grows
599,781 -> 607,417 B (+7,636 B, +1.27%) — the binary length-3 win being partly given
up, exactly as the threshold sweep predicts. `symbols.dwarf` improves, 394,736 ->
391,698 B.

WHAT THIS IS NOT:
  * **NOT a wall result.** The `ht_fast` routing is NO-SHIP on the wall (attempt 2,
    clause 5/6, `L1+L6, T1`), and nothing here re-opens that. The size leg being
    strong at T4 is the ARGUMENT for measuring the wall at T4 — it is not a
    substitute. Trainer is not the frozen box and no timing was taken.
  * **NOT the full board.** 9 of 22 corpus files. The others are GATE members (must
    not be inspected while choosing 256) or were not staged (tool.bin, data.json;
    data.json failed at 1.00358 on arm64 and is the likeliest non-closer).
  * **NOT promotable as-is.** 256 was fitted on TUNE; promotion is judged on GATE via
    `fulcrum try --threads 1,4`, on the frozen box, from a vanilla build.

### CORRECTNESS: ht_fast@256 output IS valid gzip. The `verdict FAIL` is P4, and P4 fails on main too.

Non-negotiable #1 before any size claim counts. `fulcrum verify` (compress, decompress
with OUR OWN decoder at every thread count, sha256 vs original, plus `gzip -dc` as the
independent cross-check), 5 corpus files x L0-9 x compress-threads 1,2,4 = 150 cells:

```
  main    binary sha f7a53025:  cells 150 | failed 0 | verdict FAIL
                                P4 MONOTONIC SIZE VIOLATED
  ht256   binary sha fc12a8d8:  cells 150 | failed 0 | verdict FAIL
                                P4 MONOTONIC SIZE VIOLATED
```

**Zero roundtrip failures on both arms** — the encoder is correct and the output is
valid gzip. The `FAIL` verdict is `fulcrum verify`'s OTHER gating assertion, P4
(monotonic size: a higher level must not give a bigger file), and **main fails it
identically**. So P4 is PRE-EXISTING and this change neither causes nor worsens it.
Reporting the verdict alone without the paired main arm would have looked like the
change broke correctness; it does not.

(Decode ran at `--decode-threads 16` on the user's suggestion — the decoder is finished
and parallel, so the oracle costs almost nothing at max parallelism.)

### The P4 violation is the L3/L4 STRATEGY ABUTMENT, and `LazyGated` no longer exists

`params_inner` currently ladders:

```
  L2  Greedy  depth  6  nice 10
  L3  Lazy    depth 12  nice 14
  L4  Greedy  depth 16  nice 30     <-- lazy -> GREEDY -> lazy
  L5  Lazy    depth 16  nice 30
```

Lazy beats greedy at comparable knobs, and L4's deeper search (16 vs 12) does not
always compensate — so `-4` can be BIGGER than `-3`. L4 is greedy only because our
table transliterates libdeflate's, which stays greedy through L4; CLAUDE.md says that
table is explicitly ours to change.

⚠ DOC BUG, worth fixing separately: `level.rs:44-51` still carries a doc comment
describing `Strategy::LazyGated` ("per-block GREEDY-vs-LAZY dispatch under a two-sided
content detector") as though it ships. It does NOT — that variant and `parse/gated.rs`
were deleted by user order under non-negotiable #3. The `Strategy` enum today is
`Fast0, Fast, Greedy, Lazy, Lazy2, NearOptimal`. Anyone reading that comment would
believe L3 is detector-gated; the `3 =>` arm is plain `Strategy::Lazy`.

The candidate the vendor diff already names (`docs/vendor-structure-comparison.md` §1):
zlib-ng's `deflate_medium`, a THIRD strategy that is neither greedy nor lazy and
"exists precisely to make L3-6 monotonic at a fraction of lazy's cost". Note a prior
naive lazy-at-L4 experiment measured 17.7% wall, so the cheap-monotonic property is the
whole point.

### FULL TUNE SET on x86, T1 AND T4: 10 board cells close, 0 open — and the WALL kills it anyway

All 11 TUNE members, trainer (Intel x86), vanilla builds, exact byte counts:

```
=== L1 T1 ===                                       === L1 T4 ===
                libdeflate    shipped      ht256      shipped      ht256
aozora.txt         4566347    4751200    4578206     4751416    4578509   fail
armexe.elf          621027     599781     607417      600310     607211   PASS
data.csv           3932419    4111742    3926359     4112086    3926566   CLOSES
data.json          1840461    1873127    1846101     1873815    1846239   fail
data.parquet      14386710   14424479   14383856    14426706   14383721   CLOSES
dickens            4972688    5077406    4981078     5078221    4981395   fail
engine.wasm         421013     426271     420714      425688     420809   CLOSES
minjs.min.js       1184930    1211746    1180722     1211684    1181042   CLOSES
movie.mp4         12901167   12903670   12899895    12903821   12900349   CLOSES
symbols.dwarf       396048     394736     391698      394830     391539   PASS
tool.bin          22673676   22565629   22196465    22565472   22195660   PASS
                 shipped 3/11 -> ht256 8/11        shipped 3/11 -> ht256 8/11
```

**10 of the 200 failing board cells close, ZERO open.** `tool.bin` also improves
369,164 B while already passing. The three that resist (aozora, data.json, dickens) are
exactly the text files the length-3 mechanism predicts will resist.

### THE PRE-REGISTERED NO-SHIP FIRES: clause 5 fails at T4 too

The pre-registered rule above said: "NO-SHIP if clause 5 fails at T4 ... this class is
then closed on the wall at BOTH coordinates — record it and stop, do not re-sample."

`fulcrum ab paired --mode compress`, n=15, /dev/null both arms, A/A certificate clean,
minjs.min.js L1, trainer (NOT the frozen box — this is a SCREEN, `freeze_checked=false`):

```
  ht256 vs shipped  T1: ratio 1.2097  a=69.321ms b=55.919ms  sign 15/15  (+13.40 ms)
  ht256 vs shipped  T4: ratio 1.1098  a=25.832ms b=23.373ms  sign 14/15  (+ 2.46 ms)

  vs gzip at T4:  shipped ratio 0.1949  |  ht256 ratio 0.2090
                  EROSION 0.0141  against the clause-5 budget 0.0050  =  2.8x OVER
```

T4 helps enormously — the absolute penalty is 5.4x smaller than T1's, and the erosion
falls from attempt 2's T1 figure of 0.2312 (46x over budget) to 0.0141 (2.8x over), a
16x improvement. **It still misses.** The 21% T1 self-tax independently reproduces
attempt 2's recorded 15-50%.

**VERDICT: the `ht_fast` routing is NO-SHIP at BOTH coordinates.** The coordinate
argument — the one genuinely untried axis — is now spent. Recorded and stopped, not
re-sampled on other files until one passes.

SEPARATE THE CEILING FROM THE VERDICT: the 10-cell SIZE result is real, roundtrip-clean,
and does not expire. What is closed is winning it *via this routing at this wall cost*.

THE NAMED REOPEN CANDIDATE, for a session that has not spent its two strikes on
load-shaving: `matchfinder/ht.rs`'s `longest_match` reads AND writes `hash3_tab`
UNCONDITIONALLY before the 4-byte search, yet `cur_node3` is only USED when that search
misses. Reordering (search first; on hit, blind-store with no load; on miss,
load-then-store) preserves table quality exactly and removes one dependent load per HIT
position — which is literally attempt 2's stated reopen condition ("a mechanism that
adds candidates WITHOUT adding dependent loads per position"). NOT BUILT HERE: this is
load-shaving, and `project_encoder_deficit_is_loads_not_stalls` is 2-for-2 against that
class (deleting 25.6M loads at L9 made the wall WORSE), so CLAUDE.md's two-strikes rule
closes it for this session. The store touches the same cache line either way; only the
latency ordering changes.

### PROVENANCE FAILURE CAUGHT BY THE CENSUS MATCH

An uncommitted `L1_HASH3_MAX_DIST = 4096` patch was left in the local working tree and
rode along across several `git checkout`s (uncommitted changes survive branch switches).
It contaminated ONE reported figure: the local arm64 "shipped-fast 4/11" baseline was
built WITH it (dickens 5,078,617, not clean main's 5,080,065).

**The x86 results are provably clean, and the check that proves it is the census match:**
trainer's tree was clean and its `shipped` column reproduces
`/root/sizeboard-all-12fcd0ed/census.json` byte-for-byte (dickens 5,077,406), which a
patched build cannot do. That is why "does the baseline reproduce the board?" is worth
running on every measurement — it caught a contamination that discipline did not.
The patch is now parked on `measure/l1-hash3-maxdist`; the tree rebuilds to 5,080,065.

## THE BIGGEST LEVER ON THE BOARD: zlib chain depths at T>1 — 70 cells, 1 flip

`docs/vendor-structure-comparison.md` records "matching zlib's chain depths at L5-L9
closes 84 failing size cells (and opens 13)". Diffing the artifact behind that claim
(`/root/size-zlibdepths/census.json`, ours_sha `276a941c`) against the baseline board
(`/root/sizeboard-all-12fcd0ed/census.json`) over the 1,320 common cells shows what
those 13 openings actually ARE:

```
CLOSED 84:  70 at T4,  14 at T1
OPENED 13:  12 at T1,   1 at T4
```

**All 13 openings are libdeflate, and 12 of the 12 T1 openings are EXACT BYTE TIES in
the base board** — the zero-headroom cage, perturbed by as little as ONE byte:

```
  dd79_bin6      L6 T1   4,461,731 = rival        ->  +1 B over
  weights        L7/L8/L9 T1                      ->  +2 B over each
  dd79_bin6      L5 T1                            ->  +4 B
  photo.jpg      L7 T1                            ->  +5 B
  movie.mp4      L5 / L6 T1                       -> +54 / +60 B
  symbols.dwarf  L8 T1                            -> +125 B
  weights        L5 T1                            -> +138 B
  data.csv       L9 T1                            -> +226 B
  engine.wasm    L8 T4  (the ONLY non-tie: was a 158 B WIN) -> +57 B
```

### The cage is sidestepped by construction, not by tuning

Apply the depth change in `params_parallel` (T>1) ONLY:

```
  closes 70, opens 1   ->  net +69, clause 3 sees ONE flip
  closed@T4 by rival:  libdeflate 56, gzip 7, pigz 7
  opened@T4:           libdeflate engine.wasm L8, +57 B
```

T1 output is untouched, so all 12 tie-cage openings vanish **by construction rather
than by fitting**. And T4 output depends only on `params_parallel`, so the census's T4
column carries over exactly — this is a re-reading of an existing measurement, not an
extrapolation.

**70 of the 200 failing cells is 35% of the board**, the largest single lever measured
this campaign, and it needs one flip resolved rather than thirteen.

### Relationship to PR #227 — same family, and #227 is the conservative member

#227 ships `p.max_search_depth = p.max_search_depth.saturating_mul(4)` in
`params_parallel` and closes 42 cells. zlib's actual per-level depths are not a uniform
multiple:

```
  level        L5    L6    L7     L8     L9
  ours         16    35   100    300    600
  zlib         32   128   256   1024   4096
  ratio       2.0x  3.66x 2.56x  3.41x  6.83x
  #227 (x4)    64   140   400   1200   2400
```

⚠ **CORRECTED — the "~28 cells beyond #227" claim first written here is WRONG, and
backwards.** Read the table again: zlib is SHALLOWER than x4 at L5, L6, L7 and L8, and
deeper only at L9. And #227's gate runs `--levels 2,6,9` while the zlib census covers
L1-L9, so 70-vs-42 was never apples-to-apples. Restricted to the comparable coordinate:

```
  ALL levels, ALL threads                  closed 84  opened 13  net +71
  ALL levels, T4 only                      closed 70  opened  1  net +69
  L2/L6/L9 only                            closed 31  opened  5  net +26
  L2/L6/L9, T4 only  <-- #227's coordinate closed 25  opened  0  net +25
```

**zlib's depths close 25 where #227 closes 42.** #227's configuration is BETTER at the
levels it gates — exactly what the shallower-at-L5-L8 table predicts. There is no
"+28 cells" to collect by switching to zlib's numbers.

**THE REAL FINDING IS WHERE THE OTHER CLOSURES LIVE.** The 70 T4 closures split by
level as `L5:18  L6:16  L7:17  L8:10  L9:9`. **45 of the 70 are at L5, L7 and L8 —
levels the standard gate never measures.** `--levels 2,6,9` samples 3 of the 9 levels
that carry failures, while `params_parallel` applies at EVERY level. So **#227's "42
cells closed" is a FLOOR on its board effect, not a measurement of it**, and the same
undercount applies to every lever ever graded at L2/L6/L9.

ACTION when #227's gate returns: re-measure the FULL board across all levels rather
than quoting 42.

### WHAT IS NOT ESTABLISHED — the wall, and it is the binding constraint

This is a SIZE re-reading of an existing artifact. **No wall number is quoted and none
exists for the T>1-only configuration.** Deeper chains cost time in direct proportion:
L9 at 4096 walks 6.83x our current 600. The T4 budget is 249-330% slack, which is large
but not unlimited, and #227's own note reports that even x4 at L9 (2,400 nodes) stays
ahead of both rivals — so the marginal question is 2,400 -> 4,096, not 600 -> 4,096.

Provenance caveat: `/root/size-zlibdepths/meta.json` carries `"attested": false`, so the
84/13 artifact is un-attested. The DIFF above is sound as a description of that
artifact; promoting anything from it requires a fresh gated run.

ORDER OF WORK: this composes with, and partly supersedes, #227. Land #227 on its
running wall gate FIRST (land-gated-work-first), then tune `params_parallel` depths
toward zlib's per-level values and gate that, with the single engine.wasm L8 T4 flip as
the known blocker to resolve.

## THE WALL BOARD REDUCES TO ONE CELL CLASS: libdeflate at T1

`/root/wallboard-L6/census.json` (L6, 20 corpus files, 4 rivals, T1+T4, commit
`e6e6ad30`, gzippy sha `eb9a0a50`, 111 measured cells, 19 failing). Split by rival and
thread count:

```
  gzip        T1   18 measured,  0 failing   worst ratio 0.5305
  gzip        T4   19 measured,  0 failing   worst ratio 0.1941
  pigz        T1   16 measured,  0 failing   worst ratio 0.6045
  pigz        T4   20 measured,  0 failing   worst ratio 0.7948
  libdeflate  T1   19 measured, 19 FAILING   worst ratio 1.2092
  libdeflate  T4   19 measured,  0 failing   worst ratio 0.5693
```

**Every wall failure on the board is libdeflate at T1, and it is ALL of them — 19 of
19.** We are 8-21% slower there (photo.jpg 1.2092, movie.mp4 1.1883, symbols.dwarf
1.1700, armexe.elf 1.1613, weights 1.1523, engine.wasm 1.1512). **At T4 the same 19
cells all PASS** (0.3940-0.5693). Against gzip and pigz we never lose on wall at any
thread count.

### Composed with the size board, this is the campaign's hard core

```
  vs libdeflate at T1:  TIE on size (154/198 cells byte-identical)  +  LOSE on wall 19/19
  vs libdeflate at T4:  lose on size only by the SEAM               +  WIN on wall 19/19
  vs gzip / pigz:       win on wall everywhere; size gaps are 0.02-1.1%
```

**The T1-vs-libdeflate cell has zero headroom in BOTH directions simultaneously** —
byte-identical on size, 8-21% behind on wall. That is why no lever moves it: there is
nothing to trade.

### This sharpens the clause-5 finding rather than replacing it

Clause 5 does not block size levers in the abstract. **It blocks them to protect our
2-5x margin over gzip and pigz** — the cells at ratio 0.19-0.79, all passing
comfortably, which is where every measured erosion landed (gzip:minjs:T4,
gzip:dickens:L4:T1, gzip:data.json:T1). Meanwhile the cell class we ACTUALLY lose on
wall, libdeflate T1, is unprotected by clause 5 because it is already failing.

So the rule is spending its entire protective budget on margins we do not need, against
rivals we dominate, while the real competitive deficit sits outside its scope. Stated as
an observation only — `CLAUDE.md` forbids rewriting a promotion rule to fit a result,
and nothing here does.

### What this implies for lever selection, concretely

  * A size lever that costs wall is affordable ONLY where the wall cell is already
    failing (libdeflate T1) or where the margin is large enough that 0.005 is not the
    binding term (`old_ratio > 0.98`). Neither is the common case.
  * **The libdeflate-T1 WALL deficit is a first-class target in its own right** and is
    NOT blocked by clause 5 — those 19 cells are already failing, so improving them
    cannot flip anything. This is the one axis where work is unconstrained.
  * T>1 is not where the wall problem is. At T4 we already win every libdeflate cell.

PROVENANCE CAVEAT: this artifact is L6-only and predates current main (`e6e6ad30`,
2026-07-31 01:14). The SHAPE (all wall failures are libdeflate T1; T4 rescues them all)
is what is being claimed, not the exact ratios. Re-measure before quoting a number.

## THE WALL DEFICIT AT L6 IS 2.16x THE INSTRUCTIONS FOR BYTE-IDENTICAL OUTPUT

`fulcrum why libdeflate:movie.mp4:L6:T1:wall`, trainer, vanilla `cargo build --release`,
main. movie.mp4 is a TUNE member (GATE files were not inspected).

```
[1 STRUCTURE] POSITION COUNTS MATCH (matches, matched-positions, literals all Δ0.00%)
  ours : 12,809,648 tokens (47,029 matches, 12,762,619 literals), 103,123,085 bits
  rival: 12,809,648 tokens (47,029 matches, 12,762,619 literals), 103,123,085 bits
    -> identical parse decisions AND byte-identical output

[2 LINES]
  ours  total Ir: 20,125,529,337
  rival total Ir:  9,315,411,125          -> WE EXECUTE 2.16x THE INSTRUCTIONS
```

The tool's own verdict: **"same algorithm; the excess is IMPLEMENTATION."** 10.8 BILLION
excess instructions on a single 12.9 MB file, for output that is byte-for-byte the same.

### This is an order of magnitude worse than the figure the campaign has been using

`docs/vendor-structure-comparison.md` §4 records the operation-level gap as **496.0M vs
555.1M Ir = 11.9%**, measured at **L2 on silesia 8 MB**. At **L6 on movie.mp4 it is
116%**. Hard stop #3 — "never generalise a measurement across levels" — exists for
precisely this, and §4's number has been the campaign's working figure for the
implementation gap. It does not describe the coordinate that fails.

### What the composition says the target is

movie.mp4 at L6 is **12,762,619 literals against 47,029 matches** — 271 literals per
match. The hot path on this file is therefore almost entirely the LITERAL path: the
matchfinder searching and FAILING, plus literal emission. That is a much narrower target
than "the matchfinder", and it is consistent with the wall board's shape (the failing
libdeflate-T1 cells are led by photo.jpg, movie.mp4, symbols.dwarf, armexe.elf,
weights, engine.wasm — the low-match-density files).

### Ir LOCATES, it does not predict the wall — and here the ratio proves it

Ir is 2.16x while the measured wall ratio for this cell is 1.1883 (+18.8%). So we retire
~1.8x more instructions per unit time than libdeflate: our IPC is far higher and the
deficit is instruction COUNT, not stalls. That agrees with
`project_encoder_deficit_is_loads_not_stalls`, which found our IPC, stalls, cache and
branch behaviour all BEAT libdeflate — and it is why "reduce instructions on the
literal path" is the shape of the lever rather than any microarchitectural fix.

### DENOMINATOR AND CAVEATS, as the tool states them

  * **2 of 4 layers ran.** [3 COUNTERS] skipped (the gzip oracle exited 1 on this file)
    and [4 PARAMS] skipped (the vanilla binary emits no `LEVEL_DECLARED`; that needs
    `--features anatomy-counters`, which must never be the binary a wall claim is quoted
    from). No claim here rests on those layers.
  * **The per-line attribution is NOT usable**: the rival shows `???:0` at 75.37%, i.e.
    libdeflate is built without `-g` and is one opaque symbol. Hard stop #1 warns about
    exactly this. The TOTAL Ir is still valid — callgrind counts instructions regardless
    of symbolisation — so the 2.16x stands while "which line" does not.
  * Coordinate: L6, T1, movie.mp4, main. NOT generalised to other levels or files; the
    next step is to repeat it on symbols.dwarf and armexe.elf (both TUNE, both failing
    wall cells) before treating "the literal path" as the class.

### PREDICTION TESTED AND WEAKENED: the excess is GLOBAL, not literal-path

The section above hypothesised that the 2.16x instruction excess lives on the LITERAL
path, because movie.mp4 at L6 runs 271 literals per match. Per "ONE MEASUREMENT
SUPPORTS ONE CLAIM — name the mechanism, then predict a SECOND consequence and check
it", the prediction was: a high-match-density file should come in much closer to 1.0.

Same command, same coordinate, dickens (TUNE, match-dominated):

```
                 lit/match    ours Ir           rival Ir          ratio
  movie.mp4        271.4      20,125,529,337    9,315,411,125     2.16
  dickens            0.57     16,710,714,581    9,291,141,632     1.80
```

**A 475x change in composition moves the ratio only from 2.16 to 1.80.** The prediction
is directionally right and quantitatively wrong: if the excess were literal-path work,
a match-dominated file should have shed most of it. It shed about a fifth.

Normalised per input byte (movie.mp4 12,942,257 B; dickens 12,174,519 B):

```
              ours Ir/byte   rival Ir/byte   excess/byte
  movie.mp4       1,555            720           +835
  dickens         1,373            763           +610
```

Two things fall out, and the second is the more interesting:

1. **Our excess is ~610-835 instructions PER INPUT BYTE on both files** — of the same
   order regardless of whether the input is 99.6% literals or 64% matches. So the
   dominant term is a GLOBAL per-position overhead, not literal-specific work. The
   "literal path" framing in the previous section is DOWNGRADED to a secondary effect
   worth at most the 2.16-vs-1.80 difference.
2. **libdeflate's instruction count is nearly content-independent at L6** — 9,315M vs
   9,291M Ir (0.3% apart) across two files with completely different match structure,
   while ours moves 20,126M -> 16,711M (17% apart). Their cost is flat in content;
   ours is not. That asymmetry is itself a structural clue and was not visible from
   either file alone.

What this does NOT change: position counts still match exactly on both files, output is
still byte-identical on both, and the excess is still IMPLEMENTATION rather than
algorithm. What it changes is the target — "make the literal path cheaper" is not the
lever; the lever is whatever costs us ~600+ instructions per position that costs
libdeflate ~0.

Coordinate: L6, T1, main, TUNE members only, trainer. Ir LOCATES and never predicts the
wall (movie.mp4 is 2.16x Ir at 1.1883x wall). Per-line attribution remains unusable —
libdeflate is built without `-g`.

## STRUCTURE, NOT RATIOS: the level knobs are INERT on exactly the cells we lose

`fulcrum anatomy explain` — the tool that "puts each level's DECLARED knobs beside the
OBSERVED behaviour and refuses loudly when they disagree". Two TUNE members, L0-9, T1,
origin/main built `--features anatomy-counters` (an instrumented build — it emits no
score and NO WALL NUMBER IS QUOTED FROM IT).

Mean chain-walk candidates per search:

```
              L2    L3    L4    L5    L6    L7    L8    L9     declared depth 6 -> 600
  dickens    3.10  4.52  5.91  5.46  8.22 12.63 15.84 17.58    knob ACTIVE
  movie.mp4  0.32  0.32  0.32  0.32  0.33  0.33  0.34  0.34    knob INERT
```

**On movie.mp4 every level from L2 to L9 does the SAME work.** The tool's verdict:

    max_search_depth=600 is INERT: only 0.1% of it is ever used (0.34 candidates/search)
    nice_match_length=258 is INERT: mean accepted match is 3.83 bytes, 1.5% of it
    search effort is not monotonic in level (P3): L3 walks 0.32 but L4 walks 0.32

The mechanism is plain once seen: on near-random input the hash chains are EMPTY (no
collisions to walk), so the search terminates immediately no matter how deep it is
allowed to go.

### Why this matters more than any ratio measured today

**Every failing wall cell is a literal-dense file** — photo.jpg, movie.mp4,
symbols.dwarf, armexe.elf, weights.safetensors, engine.wasm, winexe.exe (see the wall
board section above, 19 of 19 libdeflate T1). On exactly those cells the level ladder is
a NO-OP and **100% of our cost is the FIXED PER-POSITION PATH** — hash computation,
table insertion, bookkeeping done at every position whether or not a search happens.

That closes the loop on the two Ir results above, which I had been refining as numbers:
  * our excess is ~610-835 instructions PER INPUT BYTE, roughly independent of content —
    because it is per-position fixed cost, not search;
  * libdeflate's instruction count is content-INDEPENDENT (9,315M vs 9,291M, 0.3% apart)
    — because their fixed per-position cost is small and flat, while ours is large.
**The target is the fixed per-position path, NOT the search and NOT the literal path.**
Every depth-based lever measured today was operating on a knob that does nothing here.

### The ladder is also mostly inert at DEEP levels on match-dense input

dickens: L8 uses 5.3% of its declared 300 (15.84 walked); L9 uses 2.9% of 600 (17.58).
`nice_match_length` is INERT everywhere measured — mean accepted match is 3.8-7.3 bytes
against declared 65/130/258, so early termination essentially never fires.

Per the tool's own NEXT line: **"a declared knob that does not move observed behaviour
is a defect, not a tuning opportunity."** Our L7/L8/L9 declare 100/300/600 and walk
12.6/15.8/17.6. Whatever separates those levels, it is not what the table says.

### METHOD NOTE — recorded because it cost four turns

I measured this gap four times, at increasing precision (2.16x, then 1.80x, then
610-835 Ir/byte), without once asking what libdeflate's loop DOES differently. The user
asked "are you fixated on numbers again instead of looking at structure?" and "did you
forget about fulcrum?" — both landed. I had also just hand-rolled `valgrind
--tool=callgrind` + `callgrind_annotate` for attribution, which is hard stop #6 verbatim,
while `fulcrum anatomy explain` was sitting in the guide index answering exactly this.
ONE command produced more than four turns of ratio-refinement. Numbers teach you about
the construction; they are not the finding.

Coordinate: L0-9, T1, movie.mp4 + dickens (TUNE), origin/main, trainer.
NOT generalised to the other failing wall files — confirm on symbols.dwarf and
armexe.elf before treating "literal-dense => inert ladder" as the class.

## ⚠ TOOL DEFECT: `fulcrum anatomy --exec` returns a STALE CACHED total_ir

Discovered 2026-08-01 on trainer, fulcrum 0.3.0 (8364a059). The `--exec` layer's
`total_ir` and its per-bucket `ir=` figures are **identical to the digit across
different input files and do not vary with `--level`**:

```
  engine.wasm  raw=   868,202    total_ir=1703992455
  dickens      raw=12,174,519    total_ir=1703992455
  movie.mp4    raw=12,942,257    total_ir=1703992455
  --level 1  -> no exec output at all
  --level 6  -> total_ir=1703992455   match_finder ir=785217966
  --level 9  -> no exec output at all
```

Three files spanning 15x in size return the same instruction count. That cannot be a
measurement. The value is the one produced by the FIRST `--exec` invocation in the
session (movie.mp4, L6, with the encoder UNPINNED so it ran at the box's default thread
count), replayed thereafter.

`fulcrum selftest "anatomy"` PASSES (RATIO_SELFTEST=PASS checks=5) — the Gate-0 covers
the ratio pipeline, not the exec layer's cache key. So a green selftest does NOT cover
this.

**Consequences for anything quoted from that layer:**
  * Its bucket shares compare OURS AT THE DEFAULT THREAD COUNT against LIBDEFLATE AT T1
    — a mismatched-thread comparison — and are then replayed for unrelated inputs.
  * The `match_finder 46.08% vs 47.20% / block_split 11.32% vs 11.27%` reading is VOID.
    It was quoted here as "the part that survives instrument disagreement"; it does not
    survive, because it comes from the same cached run.
  * The layer already self-labels "UNCALIBRATED ... Measurement-Gate-5 WEAK/HYPOTHESIS
    tier, never a Gate-2 finding". That warning should be read as literal.

**What this resolves:** the 12x disagreement between `fulcrum why`'s callgrind totals
(movie.mp4 20,125,529,337; dickens 16,710,714,581 — they VARY with the input, as a real
measurement must) and `anatomy --exec`'s cachegrind total (constant). The callgrind
numbers behave like measurements; the cachegrind ones do not. Reconciliation resolves in
favour of `fulcrum why`.

**Method receipt, and it is the uncomfortable one:** I first diagnosed the discrepancy as
a thread mismatch, then RETRACTED that diagnosis because "the exec numbers were identical
across both runs, so threads weren't the cause". The identity WAS the evidence — of
caching, not of thread-independence. A number that refuses to move when an input changes
is not a stable measurement, it is not a measurement. The discriminating test cost one
command: run it on a file 15x smaller and see whether the count moves.

## ⛔ RETRACTION: the "2.16x instructions" finding was a BROKEN INSTRUMENT. True ratio is 1.43x.

Both fulcrum Ir layers are wrong, in different ways, and everything I derived from them
above is void. Hand-measured ground truth — cachegrind, `--cache-sim=no
--branch-sim=no`, binaries invoked DIRECTLY (no shell wrapper: valgrind does not follow
`exec` by default and will silently profile only `/bin/sh`), ours = the symbolised
release build whose output is byte-identical to the shipped one:

```
                     ours Ir        Ir/B    libdeflate Ir     Ir/B   TRUE ratio
  movie.mp4     1,650,693,672        128    1,153,377,248       89       1.43
  dickens       1,385,371,628        114    1,155,052,955       95       1.20
  engine.wasm     100,723,312        116            —            —        —

  fulcrum why (callgrind) CLAIMED:  movie.mp4 2.16   dickens 1.80
  fulcrum anatomy --exec CLAIMED:   1,703,992,455 for EVERY file, every level
```

**`fulcrum why`'s callgrind layer inflates ASYMMETRICALLY** — our arm x12.06, the rival's
x8.04 on dickens — which manufactures a 1.80 where the truth is 1.20. A symmetric
inflation would have preserved the ratio; this does not. So the ratio could not be
rescued from it either.

**`fulcrum anatomy --exec` returns a constant** (see the section above): 1,703,992,455
for engine.wasm (true 100,723,312 — 17x off), dickens and movie.mp4 alike.

### What is retracted, by name

  * "we execute 2.16x libdeflate's instructions for byte-identical output" — FALSE, it
    is 1.43x on movie.mp4 and 1.20x on dickens.
  * "the campaign's 11.9% figure is an L2 number and the real gap at L6 is 116%" —
    FALSE. `vendor-structure-comparison.md` §4's 11.9% (L2/silesia) is much closer to
    the truth than anything I claimed; the L6 gap is 20-43%, not 116%.
  * "our excess is ~610-835 instructions per input byte" — FALSE, it is 19-38 Ir/byte.
  * "reconciliation resolves in favour of `fulcrum why`" — FALSE. Both layers are wrong.

### What SURVIVES, and is now hand-confirmed rather than tool-reported

  * **libdeflate's instruction count really is content-independent**: 1,153,377,248
    (movie.mp4) vs 1,155,052,955 (dickens) — **0.15% apart** across inputs with utterly
    different match structure — while ours moves 19% (1,650M vs 1,385M). Their inner
    loop has a flat per-position cost; ours does not. This was the interesting half and
    it holds.
  * **The excess IS content-dependent, and the "literal path" hypothesis REVIVES.**
    Excess per input byte: **38 Ir/B on literal-dense movie.mp4 vs 19 Ir/B on
    match-dense dickens — a clean 2x.** With the broken numbers this looked like 1.37x
    and I "corrected" the hypothesis to a global overhead. With true numbers the
    decomposition is roughly: ~19 Ir/byte GLOBAL excess, plus ~19 Ir/byte MORE on
    literal-dense input.
  * The `anatomy explain` finding is untouched — it reads deterministic counters out of
    our own binary, not either broken Ir layer: on movie.mp4 the chain walk is 0.32-0.34
    at every declared depth from 6 to 600, i.e. the level knobs are INERT on exactly the
    cells we lose.

### Method receipt — the failure chain, because it repeated three times

1. Quoted `fulcrum why`'s Ir totals without sanity-checking Ir-per-byte. 1,555 Ir/byte
   for a mostly-literal file should have been implausible on its face; the true 128 is.
2. When a second instrument disagreed 12x, I "reconciled" by REASONING about which was
   more plausible instead of measuring a third time.
3. Diagnosed the disagreement as a thread mismatch, then retracted that diagnosis on the
   grounds that the numbers did not move between runs — when NOT MOVING WAS THE
   EVIDENCE (of caching).

The discriminating test in every case was the same and cost one command: **change the
input and see whether the number moves.** A constant across a 15x size range is not a
measurement; an Ir/byte that differs 12x from a sane estimate is not a measurement.
COUNT IT, NEVER INFER IT applies to the instrument as much as to the constant.

Hand-rolling was justified here ONLY because the tool was demonstrably wrong — hard
stop #6's "if the tool is missing on a box, FIX THE BOX" extends to a tool that lies.
The fix belongs in fulcrum; these numbers are the reference to fix it against.

## WITH THE INSTRUMENT FIXED: the attribution is real, and block_split is 28% of our excess

Re-run of `fulcrum why libdeflate:movie.mp4:L6:T1:wall` with the repaired callgrind
parser (see the fulcrum commits `c07f24e`, `f585650`) and BOTH arms symbolised — ours a
release build with debug info whose output is byte-identical to shipped, libdeflate
built `-g` per hard stop #1.

**The fixed parser reproduces hand-measured ground truth on both arms**, which is what
makes the attribution trustworthy this time:

```
              fulcrum why (fixed)   hand cachegrind     agreement
  ours          1,650,068,646        1,650,693,672        0.04%
  libdeflate    1,146,907,002        1,153,377,248        0.6%
  TRUE RATIO           1.439
```

Top SELF costs (previously unusable — the old parser reported call-chain inclusive
frames like `main.rs:381` and `libc-start.c:363`):

```
  OURS                                          RIVAL
  12.99%  hc.rs:0                214,327,137    6.68%  :2112                 76,575,714
   5.46%  parse/mod.rs:832        90,039,243    4.51%  matchfinder_common.h   51,686,687
   5.40%  block_split.rs:212      89,102,895    4.47%  :2800                 51,269,578
   3.11%  hc.rs:304               51,323,614    3.36%  hc_matchfinder.h:201   38,570,028
   3.10%  parse/mod.rs:847        51,220,492    3.36%  hc_matchfinder.h:223   38,570,019
   3.09%  block_split.rs:133      51,050,476    3.35%  :2644                 38,416,851
   2.46%  hc.rs:388               40,544,723    2.83%  hc_matchfinder.h:252   32,418,684
```

### The named target, and a FALSIFY record predicted it exactly

`block_split.rs` accounts for **140,153,371 Ir — 8.5% of our total and 28% of our
503M excess over libdeflate**:

  * `block_split.rs:212` = `ready_to_check_block`, **89,102,895 Ir**. Three comparisons
    (`num_new_observations >= 512 && bytes_in_block >= MIN_BLOCK_LENGTH &&
    bytes_remaining >= MIN_BLOCK_LENGTH`) evaluated at EVERY position — ~7 Ir across
    ~12.7M positions — to fire roughly once per 512 observations.
  * `block_split.rs:133` = `observe_literal`, **51,050,476 Ir**, called on every literal
    (12,762,619 of them on this file).

That file's own FALSIFY record, written before any of this was measured, says:

    The detector's real cost is hot-loop work (`observe_literal` on EVERY literal,
    `observe_match` on every match, at L2-L9). That cost is REAL and unmeasured here.
    The live lever is therefore to make an observation CHEAPER, not to delete it —
    deleting it forfeits 1.4-2.3% on the two hardest size files to save it.

**It is now measured.** The prescription stands unchanged and is now quantified: the
prize for making the observation cheaper is up to 140M Ir on this cell, and route (b)
— deleting the detector — remains closed (+2.250% armexe.elf L2, +1.678% sil40 L2).

### Caveats that bound this

  * Ir LOCATES, never predicts the wall. The measured wall ratio for this cell is
    1.1883 while Ir is 1.439 — we retire more instructions per unit time than
    libdeflate, so a 140M Ir saving is an upper bound on the opportunity, not a wall
    prediction. Any change must be confirmed paired, on the frozen box.
  * The RIVAL side cannot be compared line-for-line here: libdeflate's hot lines are
    bare `:2112`/`:2800`/`:2644` — inside `deflate_compress.c`, where its own block
    splitter lives, but not separably attributed. So "we spend 8.5% on block_split" is
    ours-only; it is NOT "libdeflate spends less on the same thing".
  * `hc.rs:0` at 12.99% is a whole-file bucket (line 0 = unattributed within the file),
    not a single site.
  * Coordinate: L6, T1, movie.mp4 (TUNE), origin/main, trainer.

### ⚠ CORRECTION: block_split is NOT where the excess is — libdeflate spends the same there

The section above reported block_split at "28% of our excess", caveated as OURS-ONLY
because libdeflate's hot lines were bare `:2112`/`:2800`. Those map, and the caveat
resolves AGAINST the finding:

```
  deflate_compress.c:2112  =  stats->new_observations[((lit >> 5) & 0x6) | (lit & 1)]++;
                              i.e. observe_literal's body
  deflate_compress.c:2800  =  the seq/should_end_block loop condition (greedy parse)
  deflate_compress.c:2644/2670 = inside deflate_compress_lazy_generic (the parse loop)
```

Comparable totals:

```
                        OURS                              LIBDEFLATE
  observe_literal   block_split.rs:133   51,050,476   :2112   76,575,714  <- THEIRS MORE
  split check       block_split.rs:212   89,102,895   :2800   51,269,578
                                        -----------          -----------
                                         140,153,371          127,845,292
```

**Difference 12,308,079 Ir = 2.4% of our 503M excess, not 28%.** libdeflate's
`observe_literal` is in fact MORE expensive than ours (76.6M vs 51.1M). Block splitting
is a real per-position cost in BOTH encoders and is not the deficit. Its FALSIFY record
is still right that the cost is real; it is NOT where we lose to libdeflate.

### Where the excess actually sits, from the same (top-8) attribution

```
  matchfinder   ours hc.rs 214.3M + 51.3M + 40.5M      = 306.1M
                theirs matchfinder_common.h 51.7M
                     + hc_matchfinder.h 38.6+38.6+32.4 = 161.3M     -> +145M
  parse         ours parse/mod.rs 90.0M + 51.2M + 51.2M = 192.4M
                theirs :2644 38.4M + :2670 38.3M        =  76.7M     -> +116M
```

Those two account for ~261M of the 503M excess on the visible lines alone. **The
matchfinder and the parse loop are the class; block_split is not.** This is only the
top-8 lines per arm, so it is a locate, not a budget — but it points somewhere very
different from the previous section.

**Method note:** the previous section's number was not wrong so much as UNCOMPARABLE,
and it said so. Writing the caveat is what made the correction possible one command
later; a confident "28% of our excess" without it would have sent the next lever at
`observe_literal` — a function where we are already CHEAPER than the vendor.

### LINE-FOR-LINE: the same statements cost us 1.25-1.33x. §4 was right; my "116%" was the broken parser.

The hot matchfinder lines map onto each other exactly:

```
  ours   hc.rs:304   let mut cur_pos = in_next - *in_base;         51,323,614
  theirs hc_mf.h:201 u32 cur_pos = in_next - *in_base_p;           38,570,028   1.33x

  ours   hc.rs:388   load_u24(base, mp) ... == seq4 & 0xFF_FFFF    40,544,723
  theirs hc_mf.h:252 load_u24_unaligned(m) == loaded_u32_to_u24()  32,418,684   1.25x
```

Identical statements, identical operations, byte-identical output, position counts
matching to the digit — and 25-33% more instructions. **We are not doing different work;
we are doing the same work in more instructions.**

That is precisely `docs/vendor-structure-comparison.md` §4's conclusion, reached at L2 on
silesia: *"Structural difference #4: register pressure, not algorithm. Our
`longest_match` keeps more state live than theirs."* **§4 was right, and the earlier
claim in this document that its 11.9% "does not describe the coordinate that fails" was
an artifact of the broken callgrind parser.** The true L6 ratio (1.44 on movie.mp4, 1.20
on dickens) is the same order as §4's L2 figure — the gap does not explode at depth.

Composed with `anatomy explain`: on movie.mp4 the chain walk is 0.32-0.34 candidates at
every declared depth from 6 to 600, so the matchfinder barely SEARCHES on this file.
Costing 306M against their 161M while doing near-zero search means the excess is in the
PER-POSITION PROLOGUE — hash, table read/write, bookkeeping — which reads
structurally identical to libdeflate's (both: 3 loads, 3 stores, 2 hashes from one
4-byte load, 2 prefetches). Same shape, more instructions: codegen.

**Consequence for lever selection.** The deficit class is CODEGEN (register pressure /
spills on an already-correct algorithm), not algorithm, not search depth, not block
splitting. That class has a long falsification record here — hand-hoisting
loop-invariant loads drove Dr UP because LLVM had already hoisted them (hard stop #4),
and `hc.rs` carries eight FALSIFY notes of that family. The remaining
vendor-precedented shape is G7 from `fulcrum candidates`: igzip's hand-written kernels
(body/finish/icf_body/encode/map are asm on x86 AND aarch64). `candidates` states the
ordering plainly: *"Our register-pressure findings (structure-comparison §4) are exactly
the problem asm solves; Rust-side alternatives (fewer live locals, monomorphized loops)
come first."*

Coordinate: L6, T1, movie.mp4 (TUNE), origin/main, trainer, both arms symbolised,
fulcrum with the repaired parser. Top-8 lines per arm — a locate, not a budget.

### The 8 hc.rs falsifications cover the WALK; the measured cost is the PROLOGUE

`make falsified Q=hc` lists eight binding records in `matchfinder/hc.rs`, at lines 227,
398, 455, 557, 597, 664, 681 and 831. **Every one is inside the chain-walk loop** —
`#[inline(never)]`, de-pipelining a hoist, the `chain_base` hoist, an addressing-mode
fix, hand-hoisting current-position operands, hoisting the wrap test out of the loop.

The hot lines measured at the failing coordinate are **`hc.rs:304` (51.3M) and
`hc.rs:388` (40.5M)** — the per-position PROLOGUE, before any chain walking, and both
BELOW the first record at :398. On these files the walk barely executes at all (0.32
candidates per position at declared depth 600). So the heavily-falsified region and the
expensive region are DIFFERENT CODE, and no record covers the prologue.

⚠ **But line attribution here is weak evidence, and it must not be treated as a pin.**
`[profile.release]` is `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`. Under whole-
program inlining, callgrind attributes an instruction to wherever the optimiser recorded
it, which need not be the statement that "owns" the work. `hc.rs:304` is
`let mut cur_pos = in_next - *in_base;` — one subtraction, credited 4 Ir per position.
That is not a subtraction's cost; it is a bucket for surrounding inlined work.

So the honest statement is: **the excess is in the per-position prologue REGION, which
carries no falsification** — not "line 304 is slow". Pinning it further needs
instruction-level attribution (`--dump-instr=yes`) or a targeted ablation, and the
FALSIFY family above says the thing NOT to do is guess at a statement and hand-hoist it:
that is 3-for-3 against, and hard stop #4 exists because hoisting "obviously redundant"
loads drove Dr UP when LLVM had already hoisted them.

`fulcrum candidates` orders this class explicitly: *"Our register-pressure findings
(structure-comparison §4) are exactly the problem asm solves; Rust-side alternatives
(fewer live locals, monomorphized loops) come first."* Neither Rust-side alternative
appears in the eight records — both are about the WALK — so both remain open, on the
prologue, with the attribution caveat above governing how they are aimed.

## COMPOSED: L4 Lazy + a DEPTH REDUCTION — 3x less wall than Lazy(12), and Pareto-dominant on data.csv

"COMPOSE BEFORE CONCLUDING: two changes that each miss the bar can clear it together."
L4 Lazy(12,30) wins on size and dies on clause 5 at 17.8x over. The reason is
mechanical: **lazy runs ~2 searches per position against greedy's 1**, so Lazy(D) costs
~2D probes where the shipped Greedy(16) costs 16. Lazy(12) is ~24 probes — a 50% budget
increase bought with no compensation. Compose it with a depth cut to hold the budget.

Size, T1, 11 TUNE members, x86 (trainer), vanilla builds:

```
  config                      beats libdeflate-4   L3>=L4 monotone   ~probes
  Greedy(16,30)  [SHIPPED]         0/11 (ties)          1/11            16
  Lazy( 6,30)                      9/11                 2/11           ~12
  Lazy( 8,30)                     10/11                 5/11           ~16   <- cost-neutral
  Lazy(10,30)                     11/11                 7/11           ~20
  Lazy(12,30)                     11/11                11/11           ~24
```

Wall, `fulcrum ab paired --mode compress`, n=15, /dev/null both arms, T1, L4, trainer
(a SCREEN — not the frozen box):

```
  Lazy(8,30) vs shipped   dickens   wall 1.0578 (+5.8%)   size 0.994396 (-0.56%)
                          data.csv  wall 0.9909 (-0.9%)   size 0.955756 (-4.4%)
  (Lazy(12,30) for contrast:  dickens wall 1.1844, +18.4%)
```

**On data.csv, Lazy(8,30) is FASTER AND SMALLER than the shipped config** — strictly
Pareto-dominant, the first such result this session. On dickens it costs 5.8%, a 3x
reduction from Lazy(12)'s 18.4%.

### Still not clean, and the honest arithmetic

Clause 5 on dickens: our L4 T1 ratio vs gzip is 0.4619, so a 5.78% self-tax gives
0.4619 x 1.0578 = 0.4886, **erosion 0.0267 against the 0.005 budget = 5.3x over** —
down from 17.8x but still failing. data.csv, being faster, has NEGATIVE erosion and no
clause-5 exposure at all. So the verdict is FILE-DEPENDENT, which means this needs the
full board rather than two files before any promotion claim.

### What the frontier says

There is a real Pareto frontier here, and the shipped point is not on it:

  * Greedy(16,30) — the SHIPPED config — is dominated: 0/11 per-label (all exact ties
    with libdeflate) at 16 probes, while Lazy(8,30) gets 10/11 at the same ~16 probes.
  * Depth buys per-label (9 -> 10 -> 11 of 11) and P4 monotonicity (2 -> 5 -> 7 -> 11)
    at ~2 probes per unit of depth.
  * The cheapest point that wins per-label on every file is Lazy(10,30) at ~20 probes
    (+25%), not Lazy(12,30) at ~24 (+50%). Lazy(12) buys only P4, which is a separate
    (and pre-existing-broken) property.

NEXT, and NOT done here: size + wall on the full TUNE set at Lazy(8,30) and Lazy(10,30),
then `fulcrum try --threads 1,4` on the frozen box. Two files is a direction, not a
verdict — this document has already been burned once this session by generalising from
a subset.

### ⛔ FALSIFIED: Lazy(8,30) is NOT cost-neutral. 10 of 11 files exceed the clause-5 threshold.

The section above argued Lazy(8,30) is cost-neutral because "lazy runs ~2 searches per
position, so Lazy(8) ~= 16 probes = the shipped Greedy(16,30)". **That is an INFERENCE,
and it is wrong.** Measured across the full TUNE set (`fulcrum ab paired --mode
compress`, n=9, /dev/null both arms, T1, L4, trainer; ratios are Lazy(8,30) vs shipped):

```
  file            wall     size      within the 1.011 clause-5 threshold?
  data.csv       1.0001  0.955756    YES  (wall-neutral AND 161,310 B smaller)
  aozora.txt     1.0158  1.005166    no   -- and its SIZE is WORSE than shipped
  data.json      1.0256  0.954288    no
  symbols.dwarf  1.0416  0.983963    no
  dickens        1.0527  0.994396    no
  tool.bin       1.0680  0.988007    no
  minjs.min.js   1.0691  0.985189    no
  movie.mp4      1.0751  0.999937    no
  data.parquet   1.0954  0.994415    no
  engine.wasm    1.0967  0.982711    no
  armexe.elf     1.0994  0.987064    no
```

**Only data.csv clears it.** The threshold comes from the clause-5 arithmetic: erosion =
`ratio_vs_gzip x (self_tax - 1)`, so at a T1 ratio of ~0.46 the 0.005 budget permits a
self-tax of only 1.011. Ten files land at 1.016-1.099.

**The probe model was too crude.** Lazy's cost is not merely "one extra search of depth
D": there is deferral bookkeeping, a second candidate to hold live, and an extra branch
per position, none of which the 2D estimate captured. COUNT IT, NEVER INFER IT applies
to cost models exactly as much as to constants — this is the fourth inferred figure to
fail this session.

### Correction to the frontier claim

The previous section said "the shipped Greedy(16,30) is DOMINATED". **It is not.** It is
dominated on SIZE (0/11 per-label against Lazy(8,30)'s 10/11) but it is the CHEAPEST
point on wall — every lazy variant measured costs more on 10 of 11 files. Greedy(16,30)
sits on the Pareto frontier; it is simply at the wall-favouring end of it.

### What survives

  * **data.csv remains strictly Pareto-dominant**: 3,645,905 B (an exact tie with
    libdeflate-4) -> 3,484,595 B, i.e. **161,310 B smaller at a wall ratio of 1.0001**.
    One file is not a lever, but it does prove the frontier has slack somewhere.
  * The SIZE frontier stands and was separately measured: 9/11, 10/11, 11/11, 11/11 at
    depths 6/8/10/12 against the shipped 0/11.
  * The P4 result stands: violations 13 -> 4 at Lazy(12,30).

**The L4 class is now closed on the wall the same way L1 and the seam were**: the size
is available, the wall budget is not. That is three independent classes killed by
clause 5 today, which is itself the finding —
see `project_clause5_is_the_binding_constraint`.

### ⚠ COORDINATE CORRECTION: that falsification was measured at T1; the L4 cells fail at T4

The section above closed the L4 class on a T1-only sweep. **The L4 board fails at T4** —
13 of its 17 failing cells are `libdeflate`@T4. Measuring at the coordinate that fails
changes the answer, exactly as hard stop #3 and the 40x receipt warn.

The clause-5 threshold is not a constant: erosion = `ratio_vs_gzip x (self_tax - 1)`, so
the permitted self-tax is `1 + 0.005/ratio_vs_gzip`. At T1 (ratio ~0.46) that is
**1.011**; at T4 (ratio ~0.19, four threads against a single-threaded gzip) it is
**1.026** — 2.4x looser.

Lazy(8,30) vs shipped Greedy(16,30), same paired method, n=9, both coordinates:

```
  file            T1       T4      within the T4 threshold (1.026)?
  data.csv       1.0001   0.9843   YES  -- FASTER than shipped at T4
  armexe.elf     1.0994   1.0218   YES
  data.json      1.0256   1.0216   YES
  engine.wasm    1.0967   1.0245   YES
  dickens        1.0527   1.0348   no
  tool.bin       1.0680   1.0408   no
  minjs.min.js   1.0691   1.0420   no
  aozora.txt     1.0158   1.0489   no  (and its SIZE is worse than shipped)
  movie.mp4      1.0751   1.0553   no
  data.parquet   1.0954   1.0843   no
  symbols.dwarf  1.0416   1.1591   no  -- WORSE at T4 than at T1
```

**4 of 11 clear at T4 against 1 of 11 at T1.** The coordinate alone is worth a 4x
difference in how many cells survive.

**So "the L4 class is closed on the wall" is RETRACTED as stated.** The accurate verdict
is: at the coordinate where the cells actually fail, Lazy(8,30) is affordable on 4 of 11
TUNE files and not on 7. That is a coordinate-dependent verdict, not an intrinsic
ceiling — the distinction the rules demand and that I collapsed one commit ago.

Two things worth noting in the data itself:
  * The self-tax is NOT uniformly smaller at T4. `symbols.dwarf` gets WORSE (1.0416 ->
    1.1591) and `aozora.txt` too (1.0158 -> 1.0489). Any model that assumes "T>1 always
    dilutes the tax" is wrong; it must be measured per cell.
  * `data.csv` is faster than shipped at BOTH coordinates (1.0001 T1, 0.9843 T4) while
    161,310 B smaller. It is not a rounding artifact.

STILL NOT A LEVER: 7 of 11 exceed the T4 threshold, and the promotion run must be
`fulcrum try --threads 1,4` on the frozen box over the GATE set, not a TUNE screen on
trainer. What this establishes is that the class deserves that run rather than a
dismissal.

### THE L4 FRONTIER, measured at T4 on both axes: ~4 of 11 files clear BOTH

Lazy(6,30) vs shipped Greedy(16,30), T4, paired n=9, trainer. Threshold 1.026:

```
  file            wall     size(vs shipped)   wall OK?   size beats libdeflate-4?
  data.csv       0.9487      0.968084          YES        YES   <- 5.1% faster, 3.2% smaller
  dickens        0.9507      1.002194          YES        no    (size WORSE than shipped)
  aozora.txt     0.9796      1.015732          YES        no    (size WORSE)
  symbols.dwarf  0.9898      0.987677          YES        YES   <- faster AND smaller
  data.json      0.9908      0.959831          YES        YES   <- faster AND 4.0% smaller
  engine.wasm    1.0101      0.986057          YES        YES
  tool.bin       1.0354      0.992227          no         YES
  movie.mp4      1.0541      0.999920          no         YES
  data.parquet   1.0725      0.994857          no         YES
  minjs.min.js   1.0931      0.989985          no         YES
  armexe.elf     1.0976      0.988777          no         YES
```

**6 of 11 clear the wall threshold; 3 are strictly Pareto-dominant** (faster AND smaller
than a config that exactly TIES libdeflate). Intersecting both axes: **4 files clear
everything — data.csv, symbols.dwarf, data.json, engine.wasm.**

Depth 8 reaches the same intersection of 4 by a different route (4 wall-OK at T4, all of
them size-OK). So the frontier converges on ~4 of 11 TUNE files either way, i.e. up to
8 board cells (4 files x T1,T4) — real, but modest, and NOT the whole L4 class.

```
  config            size (beats ld-4)   wall-OK at T4   BOTH
  Greedy(16,30)         0/11 (ties)         n/a          0
  Lazy( 6,30)           9/11                6/11         4
  Lazy( 8,30)          10/11                4/11         4
  Lazy(10,30)          11/11                 ?           ?
  Lazy(12,30)          11/11                 ?           ?   (T1 tax 1.1844: far over)
```

**The honest verdict on L4: neither an intrinsic ceiling nor a clean win.** The size is
fully available (11/11 at depth 12) but the wall budget only affords it on ~4 of 11
files, and WHICH files differ by depth. A shipping config would have to hold at every
cell, and none measured here does.

What is NOT yet measured and would complete the picture: wall at depths 10 and 12 at T4,
and the same sweep on GATE members via `fulcrum try --threads 1,4` on the frozen box.
Everything above is a TUNE screen on a non-frozen box (`freeze_checked=false`).

### THE SHARP STATEMENT: clause 3 forces depth >= 10; clause 5 forbids depth >= 10

The frontier table above under-read its own data. `size_ratio` there is Lazy(D) vs the
SHIPPED config — and the shipped config **exactly ties libdeflate-4 on all 11 files**.
So any `size_ratio > 1` is not merely "smaller win", it is a **pass -> fail flip**, and
clause 3 is ABSOLUTE:

```
  depth  6:  aozora 1.015732, dickens 1.002194   ->  OPENS 2 cells  ->  clause 3 FAIL
  depth  8:  aozora 1.005166                     ->  OPENS 1 cell   ->  clause 3 FAIL
  depth 10:  11/11 beat libdeflate-4             ->  opens 0        ->  clause 3 OK
  depth 12:  11/11 beat libdeflate-4             ->  opens 0        ->  clause 3 OK
```

**So the two depths that were affordable on the wall (6 and 8) are disqualified by
clause 3, and the two that satisfy clause 3 (10 and 12) are the expensive ones** —
depth 12's T1 self-tax is 1.1844, far past the 1.011 threshold; depth 10 sits between
and is unmeasured on the wall.

**L4 is squeezed from both sides**: clause 3 forces depth >= 10, clause 5 forbids the
wall cost that depth >= 10 carries. That is a much stronger statement than "4 of 11
clear both axes", and it supersedes that framing — the per-file intersection is
irrelevant once a single opened cell fails the change outright.

**The one measurement that could still open the class**: depth 10 at T4. It is the only
point that satisfies clause 3 while being cheaper than depth 12, and its wall is the
single unmeasured cell in the table. If depth 10's T4 self-tax lands under 1.026 on
every file it is a candidate; if not, L4 is closed on the promotion rule as written, at
BOTH coordinates, and that closure is then an intrinsic property of the rule rather than
a coordinate artifact.

Everything here remains a TUNE screen on a non-frozen box; GATE members are untouched
and promotion needs `fulcrum try --threads 1,4`.

## ⛔ L4 CLOSED ON THE RULE (pre-registered): depth 10 satisfies clause 3 and fails clause 5 on 9 of 11

The previous section named ONE measurement that could still open the class — depth 10 at
T4 — and declared in advance: *"Under a 1.026 self-tax on every file and it is a
candidate; otherwise L4 is closed on the promotion rule as written at BOTH coordinates,
and that closure is a property of the RULE rather than a coordinate artifact."*

Measured, T4, 11 TUNE members, paired n=9, trainer:

```
  file            libdeflate-4      ours(d10)     saved B   T4 wall   <= 1.026?
  symbols.dwarf       378,809       372,256        6,553    0.9519    YES (faster)
  engine.wasm         409,051       401,197        7,854    1.0116    YES
  data.csv          3,645,905     3,461,124      184,781    1.0460    no
  data.json         1,736,197     1,648,936       87,261    1.0474    no
  movie.mp4        12,891,391    12,890,786          605    1.0546    no
  armexe.elf          586,728       578,183        8,545    1.0732    no
  tool.bin         21,400,590    21,084,500      316,090    1.0825    no
  minjs.min.js      1,129,252     1,108,654       20,598    1.0935    no
  aozora.txt        4,197,277     4,189,888        7,389    1.1029    no
  data.parquet     14,160,264    14,077,068       83,196    1.1210    no
  dickens           4,672,714     4,623,432       49,282    1.1213    no
```

**SIZE: 11 of 11 beat libdeflate-4, ZERO cells opened — clause 3 fully satisfied**, with
772,154 B of total wins where the shipped config ties or barely passes.
**WALL: only 2 of 11 clear the threshold; nine exceed it.** So clause 5 blocks it.

**L4 IS THEREFORE CLOSED ON THE PROMOTION RULE AS WRITTEN, AT BOTH COORDINATES.** The
condition was declared before the measurement and it fired. Recorded and stopped, not
re-sampled at other depths: 6 and 8 fail clause 3, 10 and 12 fail clause 5, and the
monotone cost/size relationship between them leaves no interior point that could satisfy
both.

### Separate the ceiling from the verdict, as the rules require

  * **INTRINSIC and permanent:** the L4 size win EXISTS and is large — 772,154 B across
    11 TUNE files, 11/11 beating libdeflate-4, and `symbols.dwarf` is strictly
    Pareto-dominant (6,553 B smaller AND 4.8% faster at T4). This does not expire and
    should not be re-derived.
  * **COORDINATE-DEPENDENT and contingent:** "unshippable" is a verdict about
    `docs/promotion-rule.md` clause 5 as configured, NOT about the encoder. Every one of
    these cells stays comfortably faster than gzip and pigz; what fails is an erosion
    budget that permits ~1-2.6% of self-slowdown on cells we already win by 2-5x.

This is the THIRD independent class this session with a verified size win and no
affordable wall: L1 `ht_fast@256` (10 board cells), the T4 seam (109 cells), and now L4.
See `project_clause5_is_the_binding_constraint` — the pattern is now 4-for-4 and is the
session's most reusable finding.

## ⚠ THE FALSIFICATION CORPUS IS MOSTLY SINGLE-MACHINE — 5 of 42 records carry a multi-arch verdict

Counted 2026-08-01 over every `FALSIF(Y|IED|IES)` block in `src/compress/` (a 20-line
window after each marker — approximate, but the shape is not in doubt):

```
  blocks naming >= 2 arches (multi-arch verdict):    5
  blocks naming exactly ONE arch:                    9
  blocks naming NO arch at all:                     28
```

**37 of 42 do not record a multi-arch coordinate, and 28 record no machine at all** —
against a charter whose STEP 1 says "on every corpus file, **on both arches**", and a
standing rule that says RECORD THE COORDINATE.

### This is not hypothetical: a record in the tree shows the verdict flipping by machine

`matchfinder/hc.rs:455-470`, on removing a `prefetch_read`:

```
  Intel            -5.1% wall / -5.2% cycles
  M1               geomean 0.9610   (-10% at L6/L9)
  AMD Zen2 frozen  geomean 0.9993   (i.e. NEUTRAL)
```

**A change worth 10% on M1 measured as nothing on the box that issues verdicts.** That
record is one of the 5 that did the multi-arch work, and it is precisely why it caught
this. Had it been run on solvency alone it would read "no effect, not a lever" — and the
10% would have been left on the floor on the arch most users are on.

### What this does and does not put in doubt

  * **DOES put in doubt: every WALL verdict decided on solvency alone.** Wall is a
    property of a microarchitecture. 9 records name one arch and 28 name none; those are
    verdicts about a machine, recorded as verdicts about a mechanism.
  * **DOES NOT put in doubt: the instruction-count findings.** Ir is machine-independent
    by construction. The measured 1.20-1.44x instruction excess over libdeflate at
    byte-identical output (hand-verified with cachegrind: movie.mp4 1,650,693,672 vs
    1,153,377,248; dickens 1,385,371,628 vs 1,155,052,955) holds on any box that runs
    the same binaries. That is why it is the safest thing on this page to build on.
  * **SIZE has a small arch term too, which was a surprise.** Same commit, L1: dickens
    5,080,065 B on arm64 vs 5,081,832 B on x86 (~0.03%). Small, but it means cells that
    tie libdeflate BYTE-FOR-BYTE on x86 need not tie on arm64 — and the tie cells are
    the zero-tolerance ones.

### The rule's Scope clause does not mention microarchitecture

`docs/promotion-rule.md:62-66` names levels, rivals, corpus and both axes: "A change
evaluated on a NARROWER SLICE HAS NOT BEEN EVALUATED." It does not name the machine.
Recorded here as an OBSERVATION about what the rule covers — not as a proposal to change
it, which `CLAUDE.md` forbids.

### Cheapest de-risking, in cost order — none of these is a rule change

  1. **Make the arch part of every recorded verdict's coordinate.** Free. A record that
     says "solvency only" is honest; one that says nothing invites the reader to assume
     generality. 28 records currently invite that.
  2. **Re-check the 9 single-arch WALL verdicts on a second machine before treating any
     of them as closing a class.** The prefetch record shows the expected yield is not
     zero.
  3. Per-arch level tables would collide with the drop-in goal and with non-negotiable #3
     — noted and NOT proposed.
