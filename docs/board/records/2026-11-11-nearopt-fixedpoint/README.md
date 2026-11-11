# 2026-11-11 — near-opt fixed-point skip across passes — instrumented and priced (lever/nearopt-fixedpoint-skip)

Question (agent-40's pass-loop study, top ranked byte-exact lever): pass N+1 of
`optimize_and_flush`'s refinement loop can consume a cost sheet bitwise-equal to
the sheet pass N consumed, in which case it is a pure recomputation producing
identical outputs and tripping the `min_improvement_to_continue` gate
identically — a skippable no-op. The instrument (Step 1) decides whether the
fixed point fires where it matters; the break (Step 2) rides the same PR
compile-time gate.

**VERDICT: SHIP (lever-grade)** — byte-identical on every stream, wall wins
exactly at the cells the counters predicted (L11/L12 `-p3+`), ties everywhere
the mechanism cannot fire. See "Verdict" below for the honest budget line and
the L9 falsification.

## The mechanism, walked by hand

`optimize_and_flush` (c) section (`src/compress/deflate/parse/near_optimal.rs`,
the (c) refinement loop; the loop the 2026-11-08 L9-mingate record already
cited as `:393-394` gate + `:395` knob):

```
set_initial_costs(...)                      // sheet S_1 (built per block)
loop {                                      // max_optim_passes iterations
    pass_entries += 1
    pass k: DP on sheet S_k  -> path -> huffman codes -> true_cost_k
    gate:  true_cost_k + min_improvement_to_continue > best ? -> break
    costs_saved = S_k            (the sheet this pass CONSUMED)
    path_items_best = items
    set_from_codes(codes)        -> S_{k+1}   (the NEXT pass's sheet)
    if S_{k+1} == S_k  -> [counters] break      <-- THE LEVER
    if num_passes_remaining == 0 -> break
}
```

Gate ordering facts the walk fixes:

* The improvement-gate check happens at the TOP of the next pass, AFTER its DP;
  with `min_improvement_to_continue >= 1` (every shipped route: L10 32, L11 16,
  L9/L11-reused 16, L12 1) an identical pass always breaks. So when
  `S_{k+1} == S_k`, pass k+1 — if the budget allows it — always runs the DP
  once, produces the identical path/codes/cost, and stops without touching
  anything (no `costs_saved`/item re-save). **The break-before-next-DP is a
  no-op dedupe**: every observable state after the loop (`true_cost`,
  `best_true_cost`, `costs_saved`, `path_items_best`, `optimum_nodes`, and the
  `costs` the next block's `set_initial_costs` blends from) is identical.
  Same lemma shape as PR #457's item-snapshot restore (pure re-derivation of
  unchanged state).
* Both passes of a `passes=2` route always run (the mingate record's point) —
  but the fixed-point equality is a DIFFERENT gate from the improvement knob:
  it fires at a pass END, before budget exhaustion, so a `passes=2` route CAN
  skip its pass 2 when `S_2 == S_1`. Where that can actually happen is
  constrained by the cost-shapes: `S_1` is `set_default`/`adjust_impl`-shaped
  (fractions like `(default + 3c)/4`, values that need not be multiples of
  `BIT_COST`), while every `S_{k>1}` is `set_from_codes`-shaped (`bits *
  BIT_COST`). Fixpoints therefore become LIKELY only between two consecutive
  *derived* sheets (k >= 1), which is why the pass budget matters: L10 (2
  passes) has one derived pair (`S_2` vs `S_3` — measurable only at
  the exhaustion boundary, which cannot save a pass); L11 (4) and L12 (10)
  have 3 and 9 derived-vs-derived junctions per block.
* `-p2` is inert BY ROUTE since the wave-9 flip (merged 2026-11-11):
  `level.rs` `default_params_parallel_route` sends L9-12 at `threads <= 2` to
  the chunked Lazy2-600 arm and the near-opt parser never runs — every `-p2`
  cell below is an exact tie through route selection, not measurement.

## The instrument (`anatomy-counters`, default off)

Added to `src/compress/deflate/anatomy_counters.rs` (counted per event,
process-level relaxed atomics, flushed at process end under
`ANATOMY_COUNTERS=`):

| counter | semantic event |
|---|---|
| `nearopt_pass_entries` | every entry of the (c) loop = every DP pass started |
| `set_from_codes_calls` | production `DeflateCosts::set_from_codes` calls (counted in the method; all sites are near_optimal.rs) |
| `cost_sheet_equal_flags` | pass end where the NEW sheet is bitwise-equal over the DP-read ranges (`DeflateCosts::dp_input_sheet_eq`, 542 read entries — it excludes the three never-read `length[0..3]`/`offset_slot[30..32]` entries) to the sheet the pass consumed (`costs_saved`) |
| `nearopt_finite_skip_would_save` | passes the break saves at equality events that HAVE a next pass (1/event for `min_improvement >= 1`; `num_passes_remaining` for the MI==0 theoretical case) |
| `cost_sheet_equal_at_last_pass` | equality events at `num_passes_remaining == 0` — a bitwise-stable exit that saves nothing |

The equality predicate lives on `DeflateCosts` (`costs.rs`) because that is
where the arrays are; it compares exactly the ranges the DP reads
(`literal[0..256]`, `length[3..=258]`, `offset_slot[0..30]`).

## The break (Step 2, under `nearopt-fixedpoint-skip`, default off)

After a pass end's `set_from_codes`, break the (c) loop when the new sheet
equals the consumed sheet. Byte-exactness receipts: `data/byte_shas.csv` —
__ streams (51 locally staged members x L9-12 x -p2/-p4), arms = the SAME
source built twice (`--features nearopt-fixedpoint-skip` ON/OFF = the
compile-time-off gate), sha-256 of EVERY stream pairwise + roundtrip of every
stream through an independent decoder (Apple zcat, verified non-gzippy by
`--version`). Result: `diffs=0 invalid=0`.

Arms (each its own `--target-dir`, release profile, sha-stamped; see
`data/arms.txt` for the full provenance):

| arm | build | sha256 |
|---|---|---|
| BASE | default features (skip off) | `e0827354…` |
| SKIP | `--features nearopt-fixedpoint-skip` | `43a59101…` |
| CNTR | `--features anatomy-counters` | `202a7da7…` |
| SKIPC | `--features anatomy-counters,nearopt-fixedpoint-skip` | `ae69d937…` |

## Results — counters (CNTR arm, skip OFF; full per-member rows in `data/counters_cntr.csv`)

51 locally staged members ran at L9/L10/L11/L12 `-p4` (the DP-running route) +
the three fixtures' `-p2` routes (inert): aggregate over all members:

| level (-p4) | blocks | pass entries | mean passes/block | equality events | **mid-loop (skippable)** | boundary (no next pass) | skips as % of pass entries |
|---|---|---|---|---|---|---|---|
| L9 (passes 2)  | 9,207 | 18,414 | **exactly 2.00** | 31 | **0** | 31 | 0.0% |
| L10 (passes 2) | 9,216 | 18,432 | **exactly 2.00** | 20 | **0** | 20 | 0.0% |
| L11 (passes 4) | 9,241 | 30,952 | 3.35 | 372 | 293 | 79 | **0.95%** |
| L12 (passes 10)| 9,269 | 42,219 | 4.55 | 3,123 | 3,111 | 12 | **7.37%** |
| all `-p2` rows | — | **0** | — | **0** | 0 | 0 | route-inert (wave-9 flip: near-opt never runs) |

Key members (skippable passes as % of that member's pass entries):

| level | member | blocks | passes | equal | mid-loop skips | boundary | skip % of passes |
|---|---|---|---|---|---|---|---|
| L11 | in16m.bin | 366 | 1,206 | 12 | 12 | 0 | 1.00% |
| L11 | silesia-agg.bin | 178 | 591 | 7 | 7 | 0 | 1.18% |
| L11 | logs16m.bin | 67 | 191 | 6 | 5 | 1 | 2.62% |
| L11 | logs64m.bin | 270 | 736 | 25 | 24 | 1 | 3.26% |
| L11 | weights.safetensors | 1,023 | 4,036 | 7 | 4 | 3 | 0.10% |
| L11 | monorepo.tar | 1,047 | 3,554 | 78 | 71 | 7 | 2.00% |
| L11 | data.sqlite | 2,416 | 7,090 | 45 | 26 | 19 | 0.37% |
| L11 | sil40 | 1,169 | 4,109 | 37 | 28 | 9 | 0.68% |
| L11 | tool.bin | 1,753 | 6,140 | 90 | 64 | 26 | 1.04% |
| L12 | in16m.bin | 360 | 1,623 | 110 | 110 | 0 | 6.78% |
| L12 | silesia-agg.bin | 177 | 805 | 50 | 50 | 0 | 6.21% |
| L12 | logs16m.bin | 62 | 243 | 44 | 44 | 0 | **18.11%** |
| L12 | logs64m.bin | 245 | 950 | 172 | 172 | 0 | **18.11%** |
| L12 | weights.safetensors | 1,025 | 6,049 | 298 | 286 | 12 | 4.73% |
| L12 | monorepo.tar | 1,092 | 5,030 | 500 | 500 | 0 | 9.94% |
| L12 | data.sqlite | 2,420 | 9,140 | 465 | 465 | 0 | 5.09% |
| L12 | sil40 | 1,160 | 5,534 | 385 | 385 | 0 | 6.96% |
| L12 | tool.bin | 1,778 | 8,439 | 724 | 724 | 0 | 8.58% |
| L12 | access.log | 107 | 481 | 43 | 43 | 0 | 8.94% |
| L12 | ecoli.fastq | 92 | 394 | 29 | 29 | 0 | 7.36% |
| L12 | minjs.min.js | 117 | 532 | 66 | 66 | 0 | 12.41% |

**The L9 gate-ordering walk (task's falsifier):** L9 sees 31 equality events —
NOT zero, but every one is at the exhaustion boundary. The hand-walk: at a
`passes=2` route the only derived-vs-derived sheet junction is pass 2's end
(`S_3 == S_2`, both `set_from_codes`-shaped), which the loop reaches with
`num_passes_remaining == 0`; the loop exits at the budget check there with no
further DP, so the event saves nothing (`nearopt_finite_skip_would_save = 0`
at L9/L10 everywhere, by count). An L9 skip would require `S_2 == S_1` — the
first pass reproducing the INITIAL blended sheet — which is shape-excluded
except for 16-multiple coincidences across all 542 read entries (never
observed). Passing armed with 4 (L11) / 10 (L12) passes gives the skip
derived-vs-derived junctions WITH budget left, which is exactly where the
per-member skip rates of 0.1%-18% (table below) appear. So: **no gate-ordering
anomaly — the L9 counters are explained by construction**, and the
"per-pass cost-sheet reuse at L9" hope in `level.rs:618` + the
2026-11-08 mingate record is DEAD at L9: there is no pass 2 rework to
reuse — pass 2 always consumes a DIFFERENT sheet (all 18,414 blocks ran
exactly 2 passes; the improvement gate never fired after pass 1) and never
re-derives anything the engine didn't already remarshal.

## Results — actually skipped work (SKIPC arm − CNTR arm, same flushes)

| level | pass entries (skip off) | pass entries (skip on) | delta | mid-loop events (expect == delta) |
|---|---|---|---|---|
| L9 -p4 | 18,414 | 18,414 | 0 | 0 ✓ |
| L10 -p4 | 18,432 | 18,432 | 0 | 0 ✓ |
| L11 -p4 | 30,952 | 30,659 | **293** | 293 ✓ |
| L12 -p4 | 42,219 | 39,108 | **3,111** | 3,111 ✓ |

The SKIPC `pass_entries` reduction EXACTLY equals the CNTR mid-loop equality
count at every level — the counters quantify the same mechanism from both
sides, and the skip removes exactly the flagged passes and nothing else.
Instrument-arm byte identity (counters ON vs OFF under the skip): in16m
L9 `cntr == skipc` stream sha `327b0835…`; the 408-stream battery already
pins the feature-off/feature-on pair at every member.

## Results — interleaved wall medians

Pair: `.rel-base2` (skip off) vs `.rel-skip2` (skip on) — both rebuilt from the
final source; output streams re-verified identical (`bcabfc5f…` in16m L12 -p4,
matching the battery). Interleaved paired runs, alternating arm order per rep,
medians of raw samples (`data/wall.txt` has every sample; rows show median),
both arms to `/dev/null`. Load 5.2-6.6 on 14 cores. Wall is hypothesis-tier on
an unfrozen box (charter) — the frozen box adjudicates; the mechanism the
skip-rate column predicts (DP passes skipped, %) is what the deltas below
must, and do, track.

| cell | base ms | skip ms | B/A | skip rate of passes (counters) |
|---|---|---|---|---|
| in16m L9 -p4 | 355.3 | 354.4 | 0.9975 (tie) | 0% — mechanism cannot fire |
| in16m L10 -p4 | 336.0 | 335.8 | 0.9992 (tie) | 0% |
| in16m L11 -p4 | 463.2 | 463.3 | 1.0001 (tie) | 0.95% |
| in16m L12 -p4 | 574.1 | 548.2 | **0.9549** | 6.78% |
| in16m L9/L10/L11/L12 -p2 | 111.0-111.4 | 111.1-111.7 | 0.9993-1.0036 (tie) | route-inert |
| silesia-agg L9 -p4 | 131.6 | 132.0 | 1.0030 (tie) | 0% |
| silesia-agg L11 -p4 | 173.3 | 171.1 | 0.9870 | 1.18% |
| silesia-agg L12 -p4 | 236.8 | 232.7 | 0.9826 | 6.21% |
| logs16m L11 -p4 | 211.8 | 207.9 | 0.9817 | 2.62% |
| logs16m L12 -p4 | 557.6 | 480.6 | **0.8621** | 18.11% |
| logs16m L12 -p2 | 51.2 | 51.3 | 1.0018 (tie) | route-inert |
| logs64m L12 -p4 | 1,812.7 | 1,568.2 | **0.8651** | 18.11% |
| weights L11 -p4 | 2,467.5 | 2,470.3 | 1.0011 (tie) | 0.10% |
| weights L12 -p4 | 3,045.3 | 2,968.0 | 0.9746 | 4.73% |
| weights L12 -p2 | 1,640.2 | 1,646.2 | 1.0036 (tie) | route-inert |
| monorepo.tar L12 -p4 | 1,424.3 | 1,325.4 | **0.9306** | 9.94% |

Mechanism discipline in one line: every cell the counters mark at 0%
pass-skip is a tie (L9/L10 -p4 and every -p2 row — the
route the wave-9 flip sends to Lazy2); every wall gain sits exactly at the
cells with real skip rates, and gain size tracks the skip rate (18.11% of
passes -> 13.5-13.8% wall; 9.94% -> 6.9%; 6.2-6.8% -> 4.5-5.4% endpoints and
1.7% silesia; 0.95-3.26% -> within-noise 0-2%). Wall NEVER loses on any
measured cell (worst +0.36% = a -p2 inert route tie band row).

## Verdict

**SHIP (lever-grade, wall-only, zero size cost).** The honest budget line:

* Size budget: **0 bytes, by proof** — the lever is a no-op dedupe; the
  408-stream battery pins every member-level-thread stream byte-identical
  (diffs=0, invalid=0), so there is NO size clause to adjudicate and no
  1%-spend to price.
* Wall budget: the win concentrates exactly where the instrument says it
  must: **L12 -p4** (7.4% of DP passes skipped; 4.5-13.8% wall at -p4 on the
  measured corpus classes) and secondarily **L11 -p4** (0.2-1.9% wall where
  the skip rate is 1-3%). It is construction-monotone on wall (strictly less
  work, identical output): the mechanism cannot make anything slower beyond
  the one equality compare per pass end (a 542-entry compare measured inside
  the tie band on every -p2/-p4 row).
* **The L9 story the counters tell is the negative half of the record and it
  is real:** at L9 (and L10, symmetric) the shipped `passes=2` route runs both
  passes on every block and only ever reaches a fixed point AT the
  exhaustion boundary, where nothing follows to skip: `nearopt_finite_skip_
  would_save = 0` on all 9,207+9,216 blocks. The "per-pass cost-sheet reuse"
  named in `level.rs:618-619` and the 2026-11-08 mingate record as the
  remaining L9-T>1 wall reducer is therefore FALSIFIED AT L9 — there is no
  duplicated pass-2 work at L9 to dedupe; the sheet pass 2 consumes is always
  a different sheet from pass 1's, and the cited lever class has nothing to
  hand it. The value lives at L11/L12 passes, as the brief predicted.
* Cell scoring: SHIP candidates are the L11/L12 `-p3+` cells (not cells the
  comparator cares about at other thread counts); `-p2` rows are route-ties;
  wall rows are hypothesis-tier pending the frozen box. Nothing else on the
  ladder (L0-L8, T1) executes this code path.

Promotion step still owed (per CLAUDE.md laptop-loop bounds): the frozen-box
`make lever REF=lever/nearopt-fixedpoint-skip ARGS="--threads 4,2"` board run
to convert the hypothesis-tier wall medians; open the PR (blocked here by the
no-push brief). Candidate default flip: on promotion, move
`nearopt-fixedpoint-skip` into the default feature list exactly the way
`near-opt-bt-probebudget` was promoted (mechanism shipped default, counters
stay behind `anatomy-counters`).

## Branch state

`lever/nearopt-fixedpoint-skip` = one commit on top of `origin/main` @
`13c32862` (the origin head the brief pinned; commit hash rotates with
amends — read `git log -1` for the tip). While the session ran,
origin/main gained +13 records/docs/probe-receipt commits (`0ea00acb …
0f49e5b1` — wave-B censuses, L5-depth-32 probe receipt, doc syncs; none
touch `near_optimal.rs`/`costs.rs`/`anatomy_counters.rs`/`Cargo.toml`;
`level.rs` gains only unrelated comment blocks), so THIS branch reads
"behind 13" against the updated remote-tracking ref; rebase before
opening the PR. Worktree `.review-tmp/wA-fixpoint`; NOT pushed (brief).

## Reproduce

Everything ran on the laptop box (14 cores, load 6.9-8.4 background counted
for the wall rows via interleaved medians; wall numbers are hypothesis-tier —
the frozen box adjudicates).

```bash
# worktree + branch (off origin/main @ 13c32862)
git fetch origin && git worktree add .review-tmp/wA-fixpoint \
    -b lever/nearopt-fixedpoint-skip origin/main
scripts/dev/worktree-link.sh .review-tmp/wA-fixpoint
cd .review-tmp/wA-fixpoint
# four arms, separate --target-dir each (the shared-target trap)
cargo build --release --target-dir .rel-base
cargo build --release --features nearopt-fixedpoint-skip --target-dir .rel-skip
cargo build --release --features anatomy-counters --target-dir .rel-count
cargo build --release --features anatomy-counters,nearopt-fixedpoint-skip --target-dir .rel-skipc
```
