# Branch garden — dispositions (2026-09-03)

145 remote branches; 145 minus a handful is dead weight. Every `git fetch` pays
for all of them, and a checkout 26 commits behind once made an entire
falsified-record search look empty. Rule applied: **nothing is deleted while it
carries unrecoverable work**; a tip that is reachable from `origin/main` loses
nothing by deleting the ref (the commits stay reachable from main). For
everything else the tip sha is recorded HERE first, so any content can be
re-fetched or re-read later even after ref deletion.

## Batch 1 — DELETE NOW (tips reachable from `origin/main`, zero content)

    origin/bucket-split-oracle          origin/dropin-divergence-fix
    origin/chore/anatomy-wall-arm       origin/chore/anatomy-wall-l2l9-coverage
    origin/feat/alloc-fix-compress      origin/feat/deflate-crown
    origin/feat/l0-stored               origin/feat/pure-rust-encoder
    origin/fix/dropin-ordering          origin/gate4-compress-routing
    origin/inc6-levers                  origin/inc6-parallel-tgt1
    origin/inc7-ffi-removal             origin/probe/block-budget
    origin/sf1-fastlevel                origin/sf2-parser-taxstrip
    origin/sf4-elim-copies              origin/sf6-inline-tables
    origin/t1seam-locate

## Batch 2 — RESOLVED (2026-09-28, the open-PR census after #403/#404/#405 landed)

| branch / PR | closure disposition |
|---|---|
| `perf/t1-output-cap` (the stack) | landed through #366/#367 (merged 09-05 / 09-24) + the ladder renames #400; content resident on trunk |
| `lever/ldx-good-match` | **MERGED via PR #363** (321e944f) — the good_match pair + the L7 routing retirement; the L6 half was re-measured on its own box leg and REVERTED (the routed port's L6-T1 wall was +1.8% vs the legacy arm; see the mod.rs record) |
| `lever/ldx-len3` | **MERGED via PR #364** (84f22994) — the len-3 machinery + the L3 retirement; its landing carried the L6-revert re-pins |
| `lever/one-encode-per-level` (PR #356) | closed with disposition (superseded by the stack, per the older census note) |
| `lever/postparse-split` (PR #346) | closed NO-SHIP (measured-and-stopped) |
| `lever/ldx-forceinline` (PR #368) | CLOSED 2026-09-28, landed-or-superseded (1.0.0-target PR census): the vendor forceinline landed re-measured (the matchfinders carry the `#[inline(always)]` markers + the 1.34x note at `hc_matchfinder.rs:114`); the ARM profiling tools landed (`scripts/campaign/profattr.py`, `profile-ldx.sh`); the bounds-elision pair superseded by the post-pivot matchfinder rework (pointer-resident cursors, C-shaped walk, hw/a1 pipelining) — hand-rolled numbers never had promotion authority per charter, so no residual-board card |
| the #369 L2 carry | CLOSED superseded — the machinery it carried landed via #364; the L2 min-match-3 lever card lives on the residual board |

## Batch 3 — ARCHIVE-LATER (falsified / superseded / probes; no open PR, no live ref)

Not deleted in this batch because each holds at least one measured artifact or
one branch-sha the docs may cite. Delete after the plan doc's Phase 1/2 PRs
land, when their content value has expired. Tip shas banked here:

    perf/depth-cap f8e0f6e0 2026-07-31         probe/l1-bucket-decomp 2a5f... 2026-07-31
    perf/thread-aware-config 2026-07-31        measure/l1-htfast-ablation 2026-07-31
    lever/l4-lazy-t4 7588ba6f 2026-08-01       measure/l1-hash3-maxdist 2026-07-31
    lever/rle-shape-t4 2026-08-01              probe/ht-implementation-gap 2026-08-01
    port/libdeflate-exact 4eeb2e9d 2026-08-01  probe/l1-length-keyed 2026-08-13
    merge/227-onto-main 669e9a0c 2026-08-01    probe/l1-lenkey-inert 2026-08-13
    measure/zlib-depths-parallel 2026-07-31    probe/l1-stride-inserts 2026-08-12
    perf/combined* (4 branches) 2026-07-31     lever/l1-batched-inserts 2026-08-12
    lever/l3-gzip-deflate-fast-pickmin 2026-08-20
    (the rest of the July sweep + L1/L4 family: same rule; if a tip is not
     found in this file, its sha is in `git reflog` of whoever cut it — the
     repo's contract is commits, not refs, and every verdict landed in src/)

## Batch 4 — RELOCATE (not campaign branches)

* `release/*` (11 formula tags, Apr–May 2026) — packaging history, not WIP.
  Convert to git tags or leave; they cost little.
* `gh-pages` — site publishing; leave.
* `rescue/solvency/*` (35 branches, June–July 2026 decode-era, 1000–1600
  commits each) — biggest fetch cost. The decode campaign is DONE and banked
  (PR #116, CLAUDE.md header); record each tip sha in one commit here, then
  delete them all. Decision left to the owner's ack because these were cut by
  an unavailable box's reflog.

## The batch-1 deletion command (run from a checkout after this doc merges)

    git push origin --delete \
      $(for b in origin/dropin-divergence-fix origin/feat/l0-stored \
           origin/chore/anatomy-wall-arm origin/feat/pure-rust-encoder \
           origin/fix/dropin-ordering origin/gate4-compress-routing \
           origin/inc6-levers origin/inc6-parallel-tgt1 \
           origin/inc7-ffi-removal origin/probe/block-budget \
           origin/sf1-fastlevel origin/sf2-parser-taxstrip \
           origin/sf4-elim-copies origin/sf6-inline-tables \
           origin/t1seam-locate origin/bucket-split-oracle \
           origin/feat/alloc-fix-compress origin/feat/deflate-crown \
           origin/chore/anatomy-wall-l2l9-coverage; do echo "$b"; done)
