# The encoder campaign: state, method, next actions

Read this before touching the encoder. Written to be picked up cold.
`CLAUDE.md` has the rules; this has the board, what already failed, and what to do.

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

## 2. The board splits into two fronts

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
