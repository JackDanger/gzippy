# The encoder campaign: state, method, next actions

Read this before touching the encoder. Written to be picked up cold.
`CLAUDE.md` has the rules; this has the board, what already failed, and what to do.

---

## 1. Where we stand

**165 failing per-label SIZE cells.** Roundtrip-verified census, canonical 17-file
corpus x L1-9 x 4 rivals x T1/T4; 1020 cells measured, 0 VOID.

| rival | T1 fails | T4 fails |
|---|---|---|
| libdeflate | 11 | 113 |
| gzip | 11 | 12 |
| pigz | 8 | 8 |
| igzip | 1 | 1 |

**The wall axis has no census yet.** That is a gap: `CLAUDE.md` grades size AND wall,
and several items below are wall questions with no board to score them against. Build
it before running a wall lever.

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

All 133 failing T4 cells, gap-to-rival divided by (chunks x per-chunk overhead):

| gap / chunk-overhead | cells | reading |
|---|---|---|
| <= 1 | 38 (29%) | chunk overhead alone explains it |
| <= 2 | 30 (23%) | reachable with slack + grid together |
| <= 5 | 32 (24%) | |
| > 5 | 33 (25%) | real ratio deficit — almost all L1 |

**Front A — chunk overhead (~100 cells).** Measured per-chunk constant: **+18.7 B at
L2, +32.1 B at L6** on 512 KiB chunks (silesia, 405 chunks) = 0.0036-0.0061% of input.
Seams are only ~5.4 B; the dominant term is extra dynamic-header mass from restarting
the block grid inside every chunk, and it grows with level.

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
are built in the PARSE (`parse/mod.rs:121-131`) and the block grid is a per-symbol
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

**Start with A1.** It is the only item that already has a located mechanism, a vendor
precedent, a named axis, and a falsifier that costs one build. A2 depends on what A1
finds about safe block lengths. B1 is a separate front and can run in parallel by
another hand. The T1 wall item needs the wall census (§1) built first.

**A1 — diff the block-END heuristic against three vendors.** The budget sweep falsified
the budget but located this: `shortmatch`'s T4 gap is **flat at +85 across every
budget** while its T1 degrades to +660 — chunking HELPS near-random data, because more,
smaller tables track drifting statistics better. So `should_end_block` should already be
cutting blocks short where there is no exploitable structure, and it is not. Compare
ours against libdeflate's `should_end_block`, zlib-ng's and igzip's. Data-responsive
without being a content detector: it reads symbols already emitted, never input ahead.
Axis **size**, targets Front A. Falsifier: deterministic size leg on all 20 canonical
files — any file that gets bigger kills it.

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

**How to measure anything:** size is deterministic — build, run, compare bytes, and
grade per-label against all four rivals on the canonical corpus, any file that gets
bigger kills the change. Wall needs the frozen box, paired, with an A/A certificate.
Never quote a ratio without the absolute figure beside it.

Wall verdicts come from the frozen box (solvency, AMD Zen2). Size is bit-identical
across aarch64/Zen2/Intel — verified — so size needs one box only. Neither remote box
currently passes the full Gate-0 suite (`profile rss` fails on both, `lib levelsweep` on
solvency) and `make deploy` correctly refuses to certify them; fix those two gates
before trusting a fresh instrument there.
