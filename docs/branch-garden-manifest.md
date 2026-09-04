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

## Batch 2 — KEEP (live work: open PR / unmerged stack)

| branch | tip sha | why it stays |
|---|---|---|
| `perf/t1-output-cap` | `ee0c1d2c` | the unmerged one-encoder stack; PR to be opened FIRST (its content is the landing order below) |
| `lever/ldx-good-match` | `df31c2a5` | PR #363; rebase onto the stack's next landed point before CI rerun |
| `lever/ldx-len3` | `f8b9c7e5` | PR #364; contains #363 |
| `lever/one-encode-per-level` | `4d368406` | PR #356, superseded by the stack's `d9418505`; closes with disposition pointing at the stack |
| `lever/postparse-split` | `4893dbff` | PR #346, verdict NO-SHIP recorded in-body; closes as measured-and-stopped |

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
