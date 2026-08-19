# gzippy encoder campaign — handoff

**Date:** 2026-08-17 (updated 2026-08-18)  
**Done when:** **0 / 1320** failing GATE promotion-board cells (size AND wall, per-label, T1 and T>1).  
**Not done when:** a session summary, a partial lever, or a hand measurement without `make lever`.

---

## 🛑 CRITICAL, MEASURED 2026-08-18 — the whole ratchet mechanism has a severe, likely-disqualifying wall cost, on BOTH routes

Discovered while responding to a fourth review's demand for real wall/RSS measurement of the
streaming fix (see "PROMOTION PAUSED" section below for that review's four findings — all
confirmed correct, but this is bigger than any of them). **This is not scoped to streaming.**
The mmap/file route from `b4b821c9` — reviewed clean by Fable, cursor-agent, and a prior
Codex pass — has the SAME problem, because it's the SAME mechanism (cumulative multi-arm
pick-min via `deflate_one_shot_t1_ratcheted`).

**Two independent measurements, both executed, neither noise:**

1. **Streaming route, directly measured (hyperfine, 15+ paired runs):** 3 MiB incompressible
   pipe input, `-5`. Branch: 295ms. Old code (main): 40ms. libdeflate: 34ms. gzip: 66ms. pigz:
   61ms. **The branch is 7.3x slower than our own prior code and 8.7x slower than libdeflate**
   — on an ordinary, common-shape request. 11x more instructions retired, 3x peak RSS.

2. **Mmap/file route, from the in-flight wall census** (`CAMPAIGN_OUT=/root/lever-b4b821c9-wall-3`,
   `--threads 1 --levels 1,2,3,4,5,6`, launched to validate the ORIGINAL ratchet from `b4b821c9`,
   not the streaming fix): **319 confirmed wall losses vs 75 confirmed wins** (81% loss rate)
   among `RESOLVED` (non-noise, non-VOID) cells so far, still running. Confirmed via the
   per-cell JSON artifacts (`a` = our gzippy, `b` = rival; `RESOLVED-a-slower` = we lost,
   re-verified against a `wall_class: WIN` example to be certain of the field semantics before
   concluding anything). Worst confirmed ratios: up to 29x slower than a rival. Multiple cells
   show the box's own 3x-n re-confirmation mechanism OVERRIDING an initial win into a confirmed
   loss (e.g. `gzip:weights.safetensors:L3:T1`: first pass 0.42x [win] → confirmed 2.43x
   [loss]) — this is exactly what CLAUDE.md's flip-confirmation exists to catch, and it is
   catching a real regression, not noise.

**Root cause (reasoned, not yet independently re-verified against a profiler — name this
before trusting it further):** `deflate_one_shot_t1_ratcheted` computes the CUMULATIVE union
of every level 1..N's own multi-arm pick-min to guarantee monotonicity. At L2 this is
~4-5 arms (L1's 2 + L2's 2-3); by L5 it's ~10-12 arms. Each arm is a FULL independent parse of
the input. On files small enough that a single rival pass is already fast (sub-100ms), running
10+ of our own passes to save a few hundred bytes is a wall-losing trade — the SIZE win the
ratchet buys is real and was correctly SHIP-verdicted, but nothing in this campaign's earlier
review chain (mine, Fable's, cursor-agent's, an earlier Codex pass) priced this cost before
now, because no one had run a real wall census against this exact ref until this session.

**What this means for the branch:** size-only promotion (the `--size-only` SHIP verdict from
earlier this session) is NOT sufficient on its own — CLAUDE.md's own bar is size AND wall, and
this looks headed for a wall NO-SHIP. Do not merge on the size verdict alone. Do not spend
further effort polishing streaming test coverage (the fourth review's findings #1/#3/#4) until
this is resolved — if the ratchet mechanism gets redesigned to cut arm count, those tests may
need to change again anyway.

**Options, not yet decided:**
1. **Cap the arm count / narrow the ratchet's cumulative scope** — e.g., each level compares
   against only level N-1's WINNING arm (not the full re-derivation this session's recursive
   and iterative implementations both do), which is a fundamentally different, cheaper
   construction than "compute everyone's own arms and fold." Needs its own design pass.
2. **Gate the ratchet on file size** — only pay the multi-arm cost above some threshold where
   the absolute wall cost stops mattering relative to rivals (small files are exactly where
   the RATIO looks worst even though the ABSOLUTE time is small). Same class of routing CLAUDE.md
   already permits (input-size, not content) but changes what "closes a size cell" means for
   small files.
3. **Revert to before `c8bbde67`** and re-approach the original L3 lever with wall priced in
   from the start, per the CLAUDE.md rule this session already learned the hard way once
   (run `cargo test` first) and is now learning again at a larger scale (measure wall before
   declaring a multi-arm mechanism safe, not after it's spread across two routes).

**Next action: get the actual `fulcrum try` verdict (SHIP/NO-SHIP/UNDECIDED with clause
numbers) once the census finishes** — the tallies above are this session's own log-reading,
useful for orientation but not the adjudicator. Do not decide among the three options above
without it.

---

## ⛔ PROMOTION PAUSED 2026-08-18 — Codex pre-merge review found a real, confirmed gap

The `lever/l3-gzip-deflate-fast-pickmin` branch (now at `8b0a21b0`, worktree
`~/www/gzippy-lever-l3-pickmin` — the primary `~/www/gzippy` worktree is back on
`main`, see "LOCAL LAYOUT" note in memory `feedback_repo_discipline.md`) is a three-agent-reviewed
(Fable, cursor-agent x2, Codex) ratchet fix for a ladder-monotonicity regression the L3
pick-min lever introduced. Codex's review found what all three prior reviews missed:

**`deflate_one_shot_t1_ratcheted` only reaches the T1 whole-buffer/mmap route.** The
STREAMING route — what the CLI actually uses for stdin/pipe input
(`gzippy -N -c - < file`) and what every library caller of `compress_bytes`/
`compress_with_pipeline` gets regardless of input source — dispatches to entirely
separate code (`encode_gzip_reader_to_writer_sized`) never wired into the ratchet, and
is NOT ladder-monotone in production. **Confirmed by direct CLI execution** (not just
trusted): `binary` fixture, T1, `-p1`, real binary — L1=662,577 → L2=666,107 (up) →
L3=661,353 (down) → L4=663,583 (up again) → L5=657,593 (down).

**This is NOT a regression this branch introduced** — verified via a throwaway probe
that `origin/main` (before c8bbde67/b4b821c9 ever existed) shows the IDENTICAL sag set
on the streaming route today. It was simply never covered by `ladder_is_monotone_t1`
(which only ever tested `encode_gzip_bytes_to_vec`), so nothing caught it until now.

**User decision (2026-08-18): the ladder invariant is REQUIRED on the streaming route
too — not a vendor-fidelity nicety.** CLAUDE.md non-negotiable #5 applies. Fixing this
is now part of finishing this branch, not a follow-up.

**Done so far (commits `22f8042a`, `8b0a21b0`):**
1. Fixed a second, independent Codex finding: `deflate_one_shot_t1_ratcheted` was
   recursive, holding up to 5 near-input-sized buffers alive simultaneously at peak
   (an 83 MiB input could peak ~400+ MiB of ratchet-owned `Vec<u8>` alone). Rewritten
   as an ascending iterative fold — same computation, byte-identical output, only 2
   buffers alive at any moment.
2. Added two RED tests pinning the streaming gap as the fix's acceptance criteria
   (deliberately not silenced with an allowlist): `size_invariants.rs::streaming_t1_is_ladder_monotone`
   (in-process, `compress_bytes`, all fixtures) and
   `t1_mmap_route.rs::t1_pipe_stdin_is_ladder_monotone_l1_to_l5` (real CLI binary,
   piped stdin, L1-5). Both correctly exclude ONLY the two pre-existing `("text",7)`/
   `("text",8)` sags already shared with (and accepted for) the mmap route — confirmed
   present on `origin/main` too, unrelated to any pick-min/ratchet lever. Full measured
   streaming sag set: `("text",7)` +567 B, `("text",8)` +413 B (both excluded, shared),
   `("tabular",3)` +15,770 B, `("binary",1)` +3,530 B, `("binary",3)` +2,230 B,
   `("noise",1)` +5 B (these four are the real gap, currently failing).

**NOT yet done — this is the next work:** design and implement a fix that makes the
streaming route ladder-monotone WITHOUT losing its point (bounded memory for large
piped input — true single-pass streaming architecturally cannot re-read input to try
multiple candidate encodings and pick the smallest, so "just add pick-min to
streaming" does not work as a design; ANY hard-guarantee mechanism trades some of
that memory-boundedness away, at least past some size threshold). This is a real
architecture question, not a bug-fix — being routed through Fable and cursor-agent
for a design pass before implementation, per this session's established practice for
judgment calls of this weight.

**The remote wall census** (`--threads 1 --levels 1,2,3,4,5,6`, PID 975533 on solvency,
`CAMPAIGN_OUT=/root/lever-b4b821c9-wall-3`) was left running through this — it
measures the mmap-route ratchet's wall cost, which the iterative-vs-recursive rewrite
does not change (same computation). Its result remains informative for that question
but is NOT the final gating measurement — a fresh lever run against the truly final
commit (once streaming is fixed) is still required before promotion.

---

---

## Goal

Drop-in replacement for gzip, pigz, libdeflate-gzip, and igzip:

- Same CLI and observable behaviour
- Valid gzip (roundtrip through gzip, pigz, libdeflate — sha256, not `wc -c`)
- **Per-label** size AND wall wins at every level 0–9 on every corpus file
- T1 and T>1 both in scope

Decompression is done and frozen. This campaign is **encoder only**.

**Build order (do not regress earlier steps):**

1. **T=1** beats every rival on size and wall
2. **T>1** parallel path (valid gzip only; byte-identity to T1 or vendors is never a gate)
3. Exotic levels −10/−11/−12

---

## Where we are

| Milestone | Status |
|-----------|--------|
| Main | `e888ac9f` — merged PR #332 (L2 mmap pick-min) |
| GATE board | **14 / 1320** size failures (`gzippy-bench/campaign/size-all-e888ac9f/`) |
| In-flight branch | `lever/l3-gzip-deflate-fast-pickmin` @ `b6fa4f08` (code @ `c8bbde67`) — **unmerged, unpushed** |
| Size lever on L3 | **NOT RUN** |
| Wall lever on #332 | **NOT RUN** (prior attempt killed mid clause-8 confirm) |

---

## GATE board @ `e888ac9f` — 14 failing cells (by name)

Artifact: `~/www/gzippy-bench/campaign/size-all-e888ac9f/summary.txt`  
Corpus: 22 GATE files · levels 1–9 · threads 1 and 4 · 4 rivals.

### gzip (11)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `gzip:dd79_bin6:L3:T1:size` | 4,452,246 | 4,436,422 | +0.36% |
| `gzip:dd79_bin6:L3:T4:size` | 4,452,034 | 4,436,422 | +0.35% |
| `gzip:photo.jpg:L3:T1:size` | 6,472,401 | 6,469,678 | +0.04% |
| `gzip:photo.jpg:L3:T4:size` | 6,472,369 | 6,469,678 | +0.04% |
| `gzip:photo.jpg:L2:T4:size` | 6,473,446 | 6,470,611 | +0.04% |
| `gzip:photo.jpg:L1:T4:size` | 6,474,687 | 6,471,864 | +0.04% |
| `gzip:dd79_bin6:L2:T4:size` | 4,460,801 | 4,459,419 | +0.03% |
| `gzip:weights.safetensors:L8:T1:size` | 83,117,467 | 83,099,633 | +0.02% |
| `gzip:weights.safetensors:L9:T1:size` | 83,117,467 | 83,099,652 | +0.02% |
| `gzip:weights.safetensors:L7:T1:size` | 83,113,840 | 83,112,303 | +0.00% |
| `gzip:weights.safetensors:L7:T4:size` | 83,112,978 | 83,112,303 | +0.00% |

### libdeflate (1)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `libdeflate:weights.safetensors:L4:T4:size` | 83,112,979 | 83,082,171 | +0.04% |

### pigz (2)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `pigz:dd79_bin6:L3:T1:size` | 4,452,246 | 4,441,970 | +0.23% |
| `pigz:dd79_bin6:L3:T4:size` | 4,452,034 | 4,441,914 | +0.23% |

**Class notes (do not re-derive):**

- **T4 failures** on `photo.jpg` / `dd79_bin6` are T>1 seam class — mmap pick-min does not touch them.
- **`weights.safetensors`** libdeflate L4–L9 thin-margin ties were mostly closed by #332; L4 T4 and gzip L7–L9 remain.
- **`dd79_bin6` L3** is the worst gzip margin and blocks pigz too.

---

## Landed — PR #332 @ `e888ac9f`

**Mechanism:** L2 T1 mmap pick-min (3 arms):

1. Shipped `params(2)` greedy baseline  
2. L1 gzip-primary hash arm  
3. Gzip `deflate_fast` arm (`params_l2_gzip_deflate_fast`: chain 8, nice 16, min-match 3, `hash3_chain_depth=8` gated to this arm only)

Pick-min threading: **2 parallel + 1 sequential** (3 parallel arms VOID'd wall pin-gate at cpu%=253).

**Cells closed (size lever SHIP, `CAMPAIGN_PROMOTE=1`):**

| Cell | Evidence |
|------|----------|
| `gzip:dd79_bin6:L2:T1:size` | 4,448,467 vs gzip 4,459,419 (−10,952 B) |
| `gzip:photo.jpg:L2:T1:size` | 6,462,189 vs gzip 6,470,611 (−8,422 B) |

Promotion record: `~/www/gzippy-bench/campaign/promotion-records/lever-332-size-SHIP-5ef737a9.json`  
Gates at merge: tie-guard **0/33** flips · clause 3 OK · `fingerprint_suite` + `block_pins` pass.

**Board impact:** 27 → **14** failures (`size-all-18238262` → `size-all-e888ac9f`).

---

## In-flight — L3 mmap pick-min @ `c8bbde67`

**Branch:** `lever/l3-gzip-deflate-fast-pickmin`  
**Commits:** `c8bbde67` (code) · `b6fa4f08` (this PLAN only)

### Mechanism (vendor-diff first)

`fulcrum why gzip:photo.jpg:L3:T1:size` and `fulcrum why gzip:dd79_bin6:L3:T1:size`:

- **gzip L3:** `deflate_fast` — chain 32, nice 32 (`vendor/gzip/deflate.c` `configuration_table[3]`)
- **libdeflate L3:** greedy, depth 12, nice 14 (we ship **Lazy** at same depth/nice)
- Our L3 gap on photo is algorithmic (−63% matches vs gzip), not Huffman

**Implementation (mirror L2):**

- `level_uses_t1_mmap_pick_min` → `1..=4`
- `deflate_one_shot_t1_l3_pick_min`: Lazy baseline ∥ libdeflate greedy ∥ gzip deflate_fast
- `params_l3_libdeflate_greedy()` — `Strategy::Greedy`, depth 12 / nice 14
- `params_l3_gzip_deflate_fast()` — Greedy, chain 32, nice 32, min-match 3, `hash3_chain_depth` in fast arm only
- Wired through all mmap pick-min match sites + `encode_gzip_unpadded_l3_pickmin`

**Files:**

- `src/compress/deflate/mod.rs`
- `src/compress/deflate/level.rs`

### Hand-measured on branch (NOT a verdict — run lever)

**Board invocation** (fulcrum / `board-size.sh`):

```bash
gzippy -{level} -p {threads} -c $CORPUS/file
```

This is **PureT1Mmap** (`encode_gzip_unpadded_slice_to_writer`).  
Stdin redirect (`-c - < file`) is **PureT1 streaming** — wrong path for these cells.

| Cell | main `e888ac9f` | branch `c8bbde67` | Rival | Expected |
|------|-----------------|-------------------|-------|----------|
| `gzip:photo.jpg:L3:T1:size` | 6,472,401 (+2,723) | **6,462,140 (−7,538)** | 6,469,678 | **CLOSES** |
| `gzip:dd79_bin6:L3:T1:size` | 4,452,246 (+15,824) | 4,448,460 (+12,038) | 4,436,422 | −3,786 B; **still FAIL** |
| `pigz:dd79_bin6:L3:T1:size` | +10,276 | +6,490 | 4,441,970 | **still FAIL** |
| libdeflate ties (photo L3, weights L3) | — | unchanged | — | OK |

**Expected post-merge board:** **13 / 1320** if only `gzip:photo.jpg:L3:T1:size` closes.  
T4 cells unchanged (mmap pick-min is T1-only).

### Gates on `c8bbde67`

| Gate | Result | Notes |
|------|--------|-------|
| `scripts/campaign/tie-guard.sh HEAD` | **0/33 flips** PASS | vs `origin/main`; re-run before merge |
| `cargo test --release --test fingerprint_suite --test block_pins` | **PASS** | no pin regen needed |
| `CAMPAIGN_PROMOTE=1 lever.sh c8bbde67 --size-only` | **NOT RUN** | **first command for next agent** |

---

## Next agent — do this in order

### 1. Size lever (required before PR)

```bash
cd ~/www/gzippy
git checkout lever/l3-gzip-deflate-fast-pickmin
cargo build --release
scripts/campaign/tie-guard.sh HEAD
cargo test --release --test fingerprint_suite --test block_pins
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-c8bbde67-size \
  scripts/campaign/lever.sh c8bbde67 --size-only
```

- If **SHIP** → open PR, merge, then re-board:

```bash
CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all
# artifact: ~/www/gzippy-bench/campaign/size-all-<sha>/
```

- Report cells closed **by name**. Zero is information.
- Fix any clause-3 regression before merge; never hand-roll measurements.

### 2. If L3 SHIPs — next size front: `dd79_bin6` L3

```bash
fulcrum why gzip:dd79_bin6:L3:T1:size --repo .
```

Pick-min exhausted the obvious vendor arms (gzip deflate_fast + libdeflate greedy). Remaining gap is parse/block-boundary class. Do **not** ship global `hash3_chain_depth` or strategy changes without tie-guard + L6/L9 spot check (receipt: global hash3 flipped libdeflate L6/L9 ties).

**Judgment call routed through Fable (2026-08-17), cross-check via cursor-agent in flight —
this is a RECOMMENDATION for the next lever session, not yet built or measured:**

`fulcrum why` on this cell: matches Δ1.81%, literals Δ4.59% — gzip covers 61,994 more
positions using only 26,350 more matches (≈2.35 B/match ⇒ predominantly short/len-3-class
matches we're missing). Algorithmic/parse gap, not Huffman.

Fable rejected the two obvious zlib-ng candidates ([P14] early-exit-on-non-improving —
wrong direction, can only produce equal-or-worse matches; [P10] deflate_medium — unmotivated
midpoint of two arms that both already lose here) and instead named two composable defects
inside the **existing** L3 gzip-deflate_fast arm:

1. **Depth mismatch.** gzip L3 walks its 3-byte-keyed chain to depth 32
   (`vendor/gzip/deflate.c` `configuration_table[3]`). Our `params_l3_gzip_deflate_fast()`
   (`level.rs:370-381`) sets `max_search_depth=32` for the **hash4** chain only — the hash3
   side-chain is pinned at `hash3_chain_depth = HC_HASH3_CHAIN_DEPTH = 8` (`hc.rs:91`), 4x
   shallower than gzip at L3. (At L2, gzip's own chain depth is 8, which is offered as why L2
   closed but L3 didn't — the existing depth-8 hash3 happened to be faithful there.)
2. **Stale chain maintenance.** `HcMatchfinder::skip_bytes` (`hc.rs:643-702`), called on every
   match interior, overwrites `hash3_tab[hash3]` heads but never writes `next3_tab` — interior
   inserts leave dead/stale hash3 chain links, unlike gzip's uniformly-linked chain. Same class
   of defect as the already-proven-causal L1 bucket-maintenance mechanism
   ([[project_l1_bucket_maintenance_mechanism]]).

Falsify-record check: `src/compress/ldx/ht_matchfinder.rs:21` cites a FALSIFY record at
`src/compress/deflate/parse/mod.rs:540` that **no longer exists at that line** (current line
540 is an unrelated struct field) — treated as non-binding/dangling per CLAUDE.md hard stop 2.
Doc comment needs updating (hygiene only, not a gate).

Proposed lever (NOT implemented): (A) one line — `hash3_chain_depth = 32` in
`params_l3_gzip_deflate_fast()`, arm-gated so L2's arm is untouched. (B) thread
`maintain_hash3_chain: bool` into `skip_bytes` to also write `next3_tab` when true, called
from `greedy.rs` gated on `params.hash3_chain_depth > 0` — **this changes L2's gzip-arm bytes
too** since `skip_bytes` is shared, so grading must cover L2+L3 together. (C) if A+B leave a
residual: separate arm-gated block-flush-cadence lever (gzip flushes far more often, trading
header bits for better local fit — rival spends 49,427 header bits vs our 16,760 on this
file) — measured and gated independently, composed only if needed.

**Do not start this lever until:** (1) the c8bbde67 size-only `make lever` verdict lands and,
if SHIP, is merged (land the win first); (2) cursor-agent's independent cross-check of this
recommendation completes — see next PLAN.md update.

**cursor-agent cross-check (2026-08-17), CONCUR WITH CHANGES:** independently opened
`ht_matchfinder.rs:21` (confirmed dangling — the cited line is a struct field, not a FALSIFY
record) and `hc.rs`'s `longest_match`/`skip_bytes` (confirmed defect 2 is real: `skip_bytes`
updates `hash3_tab` heads but never writes `next3_tab`, unlike `longest_match`). **Sequencing
correction to Fable's plan: lead with defect (B) — the `skip_bytes`/`next3_tab` chain repair —
not defect (A) — `hash3_chain_depth = 32`.** `hc.rs:76-90` already carries an in-file note that
depth gains on `dd79_bin6` L3 saturate at depth 4 (d=8 is a wash or worse) — that measurement
likely predates the chain-repair fix, so depth 32 alone is unlikely to close the gap and should
be re-swept AFTER (B), not shipped as the headline change. Cheap first experiment once B lands:
`hash3_chain_depth = max_search_depth` in the L3 gzip arm, then sweep. Also flagged: `lazy.rs`
calls `skip_bytes` too — any future arm enabling `hash3_chain_depth` there needs the same gate.
D3 (periodic cost-based flush, "front C" in Fable's plan) confirmed as the right independent
residual lever if B(+A) underdeliver — different mechanism, compose don't conflate.

**Revised proposed order for the next `dd79_bin6` L3 session: (B) chain-repair → re-sweep
depth on the L3 gzip arm → (A) if still short → (C) block-cadence, each gated and measured
separately, `tie-guard.sh` before every T1-output edit.**

---

## ⛔ BLOCKING — new regression found on `c8bbde67` (2026-08-17), confirmed by execution

`cargo test --release --test size_invariants ladder_is_monotone_t1`:

- **FAILS on this branch** (HEAD `2b5e860a`): `"text" level 4 produced a LARGER output than
  level 3 (333536 > 324803 bytes, +8733 B, +2.7%)`.
- **PASSES on `origin/main` (`e888ac9f`)** — verified directly in a side-by-side worktree build
  (`~/www/gzippy/lever-c8bbde67-size-full/arm-e888ac9fd7fe`), not inferred.

**This blocks merge regardless of the GATE promotion-board verdict** (a board SHIP with a
failing size-invariant test is still NO-MERGE — CLAUDE.md non-negotiable 5).

**Root mechanism (cursor-agent diagnosis, code-verified):** the branch widened L3's pick-min to
three arms including `params_l3_gzip_deflate_fast()` (Greedy, chain **32**/nice **32**, forced
len-3 acceptance) but left L4's pick-min untouched (two arms, Greedy/Lazy at libdeflate-table
**16**/**30**, no len-3 forcing). On the `text` synthetic fixture (1 MiB prose, aozora/dickens
class — not a toy edge case), L3's new deeper/more-aggressive arm now out-parses L4's capped
arms, so L3 leapfrogs L4. Arm diversity is asymmetric, not a depth/strategy abutment already
covered by `KNOWN_SAGS`.

**Fix recommendation: widen L4, do NOT add `("text", 3)` to `KNOWN_SAGS`.** The sag is not
coordinate-locked to a synthetic-only artifact (text models real prose shape) and `KNOWN_SAGS`
documents accepted defects, not ones with an available fix. Mirror the L2→L3 precedent:

1. **`level.rs`** — add `params_l4_gzip_deflate_fast()`: same envelope as
   `params_l3_gzip_deflate_fast()` (chain 32/nice 32/forced len-3) but based on `params(4)` so
   L4-specific fields stay intact. NOT a gzip-level-4 transliteration (gzip `-4` is actually
   lazy/16/16) — purely to give L4 a third arm that can reproduce L3's winning candidate class.
2. **`mod.rs`** — widen `deflate_one_shot_t1_l4_pick_min` to three arms (two parallel + one
   sequential, mirroring L3's shape), picking min against the new gzip-class arm.
3. Verify: `ladder_is_monotone_t1` passes (all fixtures, both directions); per-arm ablation on
   `text` L3/L4 confirms which arm wins; `tie-guard.sh` shows L1/L2 untouched (only L4 pick-min
   changes); re-run the branch's promotion measurement after the fix.

**Scope:** L1/L2/L3/T>1/streaming untouched by this fix — only the L4 T1 mmap pick-min path
widens. If 32/32 doesn't fully close the gap, escalate depth/nice above L3's values (L4 must at
minimum reproduce L3's winning candidate).

**Next agent: implement this fix BEFORE re-running any lever on `c8bbde67`.**

**UPDATE (2026-08-17, same session): the L4 fix above was implemented and built — it
correctly closed L3→L4 (L4 now ties L3 at 324,803 B) — but `ladder_is_monotone_t1` is
STILL RED: the violation moved one level up.**

```
thread 'ladder_is_monotone_t1' panicked: on fixture 'text', level 5 produced a LARGER
output than level 4 (333536 > 324803 bytes, +8733 bytes)
```

L5 uses a DIFFERENT code path (`deflate_one_shot_t1_zlib_pick_min`, `level_uses_t1_zlib_pick_min`
= `5..=7`, 2 arms: baseline Lazy vs zlib-shaped Lazy at deeper chain) with no gzip-`deflate_fast`
Greedy arm — same mechanism, next level up. **This is whack-a-mole, not a fix**, and both Fable
and cursor-agent (independent architecture reviews, full reports below) converged on the same
verdict before I could keep hand-mirroring arms: **stop mirroring arms level-by-level; the
structural fix is a ratchet — level N's output is capped at level (N-1)'s bytes when N's own
pick-min regresses.**

**Both reviewers' key findings (independently reached, both from static code reading):**
- Root cause: pick-min arm SETS are heterogeneous per level (different vendor-shaped candidates,
  not a monotonically-deepening single parser), so `params_inner` depth monotonicity does NOT
  imply pick-min output monotonicity. `level.rs`'s OWN existing receipt (engine.wasm L8: deeper
  search made output LARGER — "a longer match at position i displaces a better match at i+k")
  proves parameter reasoning alone can never certify monotonicity; only a byte comparison can.
- A ratchet only ever REPLACES output with something STRICTLY SMALLER, so **tie-guard risk is
  ~zero when the ratchet itself fires** (it's a no-op on any cell that's already monotone,
  which includes all current libdeflate ties). Real tie-guard exposure is on the underlying
  arm-mirroring edits (e.g. adding a new gzip-fast arm to a level), same as any T1 change —
  run tie-guard regardless.
- **Full L1–L9 ratchet is too expensive to ship unmeasured**: naive recursive re-derivation
  costs ~13-20x a single encode at L9 (sum of every inherited level's arm count). A **scoped
  ratchet over L1–L5** (or L1-L7, Fable's slightly larger scope) is cheap (single wrapper,
  wall cost dominated by the deepest arm already run in parallel) but the wall bill is
  UNMEASURED and must be gated by `fulcrum try` before landing, not assumed.
- Disagreement only on immediate tactic: cursor-agent says mirror one more gzip-fast arm onto
  L5 as a quick stopgap AND land the scoped ratchet as the structural follow-up; Fable says skip
  the mirror and go straight to a "cumulative arm registry" (mathematically the same thing as a
  ratchet, expressed as a static per-level superset of arms instead of a runtime byte-compare
  recursion) for L1–L7 directly. **Recommendation: take Fable's registry framing (it's
  statically testable — "each level's arm set is a superset of the level below's" — which is
  exactly non-negotiable #5's "test the INVARIANT, not the value") but cursor-agent's SCOPE
  (L1–L5 first, wall-measure before extending to L6/L7).**

**Implementation sketch for next session (do this instead of any more manual arm-mirroring):**

1. `level.rs`: add `LevelParams: PartialEq` (or an equivalent identity key) and
   `pub fn t1_pick_min_arms(level: u32) -> Vec<LevelParams>` built as
   `t1_pick_min_arms(level - 1)` extended with that level's OWN new arm(s), deduplicated. Delete
   the now-redundant `params_l4_gzip_deflate_fast()` (L4 inherits L3's arm for free once the
   registry exists) — do NOT add a `params_l5_gzip_deflate_fast()` by hand; let the registry
   inherit it.
2. `mod.rs`: replace the four separate `deflate_one_shot_t1_l{1..4}_pick_min` functions (and
   fold in `deflate_one_shot_t1_zlib_pick_min` for L5) with one `deflate_one_shot_t1_pick_min(data,
   level)` that runs the registry's arms (generalize `pick_min_two_vecs` to N arms — keep the
   `anatomy-counters` winner-attribution branch), folding with strict `<` in native-arms-first
   order (inherited arms must not swap in different bytes on an exact tie with the incumbent —
   incumbent wins ties). Collapse the current 4 near-duplicate level-dispatch `match` blocks
   (`encode_deflate_segment_to_sink`, `encode_deflate_slack_padded_to_sink`,
   `encode_gzip_bytes_to_vec`, `encode_gzip_unpadded_slice_to_writer`) onto one predicate —
   note `encode_gzip_bytes_to_vec` currently does NOT even route L5-L7 through the zlib pick-min
   at all (falls through to single-arm `params(level)`), a **pre-existing entry-point
   inconsistency** worth fixing in the same pass, flagged but not yet acted on.
3. New unit test: `t1_arm_sets_are_cumulative` — asserts inclusion, not values, per
   non-negotiable #5.
4. Delist any `KNOWN_SAGS` pairs the registry heals in the SAME commit (test forces this).
5. Adjudicate in cheapest-falsifier order: `ladder_is_monotone_t1` + `make holdout`/`make
   surface` (deterministic, free) → `scripts/campaign/tie-guard.sh <ref>` (T1 bytes can change,
   downward only, but arm-mirroring edits still need the cage) → `fulcrum try <ref> --threads
   1,4 --levels 1-9` (or at minimum `3,5,6,9` per cursor-agent) for the actual wall number. **If
   L5-L7 wall fails the budget, fall back to scoping the registry to L1–L5 only** and park L6/L7
   inclusion with the measured wall number as the receipt — a decision the adjudicator makes,
   not one pre-announced here.

**Scope limits to state in the eventual commit:** T1 whole-buffer/mmap route only. The stdin
streaming path (`encode_gzip_single_pass`, single-arm `params(level)`) and the T>1 path
(`params_parallel`) are untouched and have their own (untested, for streaming) ladders — do not
assume this fix reaches them.

**Status: branch NOT mergeable.** `ladder_is_monotone_t1` is still red on the working tree
(uncommitted L4 partial fix + unmodified L3 code). Do not add `("text", 4)` (or `("text", 5)`
once the violation moves again) to `KNOWN_SAGS` to silence this — CLAUDE.md's "re-writing the
rule to fit the result is not allowed," and the allowlist is explicitly for defects with no
available fix, not ones a lever just created.

---

## GATE promotion verdict on `c8bbde67`, full 1-9 level size census (2026-08-17)

The FIRST lever run (levels 2,6,9 only) returned a void NO-SHIP — confirmed by inspecting its
`try.json`: `"levels": [2, 6, 9]`, zero L3 cells, so it could not see this L3-scoped change at
all. Re-ran per the tool's own prescribed remedy:

```bash
CAMPAIGN_PROMOTE=1 CAMPAIGN_LEVELS=1-9 CAMPAIGN_OUT=lever-c8bbde67-size-full \
  scripts/campaign/lever.sh c8bbde67 --size-only
```

```
TRY — promotion evaluation over 792 cells (660 decidable on both arms; 132 not)
  clause 1 OK: verify — zero roundtrip failures
  clause 2 OK: arms differ
  clause 3 OK: no pass->fail flips across 660 decidable cells
  clause 4 OK: closed failing cell(s): gzip:photo.jpg:L3:T1:size
  clause 5 OK: every passing cell inside its erosion budget
  clause 6 OK: improvement 0.0033 vs residual harm 0.0000
  clause 7 OK: all required arch(s) covered: aarch64
VERDICT: SHIP
```

Matches the hand-measured expectation exactly: **closes `gzip:photo.jpg:L3:T1:size`** (the only
cell the L3 pick-min lever was built for), zero regressions anywhere else on the corpus.
Artifact: `~/www/gzippy/lever-c8bbde67-size-full/try.json`.

**This SHIP verdict is scoped to the GATE promotion-board rubric only — it does NOT know about
`ladder_is_monotone_t1`, a separate pinned invariant test outside the board's corpus.** A GATE
SHIP with a failing size-invariant test is still NO-MERGE (CLAUDE.md non-negotiable 5). The
commit `c8bbde67` ITSELF carries the ladder regression (confirmed: fails on `2b5e860a` which
is `c8bbde67`'s code + doc commits, passes on `origin/main` `e888ac9f`) — it was not introduced
by my later uncommitted L4 edit, the L4 edit only made the SYMPTOM move from L4 to L5.

**Net next-session task: land the L1–L5 arm registry (see implementation sketch above), confirm
`ladder_is_monotone_t1` passes clean, re-run `tie-guard.sh` + this same full-level `--size-only`
lever (now against the registry-patched ref) to confirm `gzip:photo.jpg:L3:T1:size` still closes
with zero new regressions, THEN merge. Do not merge `c8bbde67` as committed today.**

### 3. T4 cluster (after T1 L3 front moves)

Open T4 cells (not fixed by mmap pick-min):

- `gzip:photo.jpg:L1:T4:size`
- `gzip:photo.jpg:L2:T4:size`
- `gzip:photo.jpg:L3:T4:size`
- `gzip:dd79_bin6:L2:T4:size`
- `gzip:dd79_bin6:L3:T4:size`
- `pigz:dd79_bin6:L3:T4:size`

T>1 seam-shrinking is a **closed class** (CLAUDE.md census). Need T1 headroom or T>1-specific parse lever.

### 4. `weights.safetensors`

- `libdeflate:weights.safetensors:L4:T4:size` — thin margin, T4 only
- `gzip:weights.safetensors:L7:T1:size`, `L7:T4`, `L8:T1`, `L9:T1` — separate class

### 5. Wall leg for #332 (optional record; size already merged)

```bash
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-e888ac9f-wall \
  scripts/campaign/lever.sh e888ac9f --threads 1
```

Prior run killed ~3h in (after-grid-wall 264/264 done, clause-8 confirm pending). Pick-min 2+1 threading fix is in `e888ac9f`.

---

## Campaign rules (binding)

| Rule | Command / action |
|------|------------------|
| Vendor-diff before proposing a lever | `fulcrum why <cell> --repo .` |
| Tie cage before any T1-output change | `scripts/campaign/tie-guard.sh <ref>` (~2 min) |
| Never hand-roll size/wall | `make lever REF=<ref> ARGS="--size-only"` or `CAMPAIGN_PROMOTE=1 scripts/campaign/lever.sh` |
| GATE board (not tune board) | `CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all` |
| Fingerprint pins | `cargo test --release --test fingerprint_suite --test block_pins` |
| Land cleared PR before starting new work | — |
| Report progress | cells closed **by name**; zero is information |

**Hard stops:** do not generalise L2-only measurements to L5–L9; do not use stdin path when board uses `-c file`; 3-thread pick-min VOIDs wall pin-gate.

---

## Toolbox

| Question | Command |
|----------|---------|
| Why does this cell fail? | `fulcrum why <cell> --repo .` |
| Is this change good? | `CAMPAIGN_PROMOTE=1 scripts/campaign/lever.sh <ref> --size-only` |
| Where do we stand? | `CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all` |
| Tie pre-check | `scripts/campaign/tie-guard.sh <ref>` |
| Falsified / parked records | `make falsified` |

Corpus root: `~/www/gzippy-bench/corpus` (`CAMPAIGN_CORPUS_ROOT`).

---

## Disk / environment (M1 Mac)

**Freed this session (~2 GB):** `gzippy-bench/breadth_tmp/`, `gzippy/benchmark_data/`, `/tmp/gzippy-*` worktrees, stale git worktrees, large agent-tools logs.

**Kept:** `target/release/gzippy` (~1 GB), corpus, `size-all-e888ac9f/`, promotion records.

**Safe to delete if space needed:**

- `cargo clean` in gzippy (~1 GB; rebuild required)
- `~/www/gzippy-bench/{weights_diag,decision_diag,residuals_diag}/` (~390 MB diag scratch)

**Do not delete:** `~/www/gzippy-bench/corpus/`, `~/www/gzippy-bench/campaign/size-all-e888ac9f/`, promotion-record JSONs.

---

## Key code map

| Location | What |
|----------|------|
| `src/compress/deflate/mod.rs` | `deflate_one_shot_t1_l3_pick_min`, `encode_gzip_unpadded_l3_pickmin`, `level_uses_t1_mmap_pick_min` (`1..=4`) |
| `src/compress/deflate/level.rs` | `params_l3_libdeflate_greedy()`, `params_l3_gzip_deflate_fast()`, `params_l2_gzip_deflate_fast()`, `hash3_chain_depth` |
| `src/compress/deflate/matchfinder/hc.rs` | hash3 chain walk (gated via `LevelParams.hash3_chain_depth`) |
| `src/compress/io.rs` | T1 file > 128 KiB → mmap → `encode_gzip_unpadded_slice_to_writer` (**board path**) |
| `scripts/campaign/lever.sh` | Promotion adjudicator |
| `scripts/campaign/board-size.sh` | Size axis; `CAMPAIGN_PROMOTE=1` for GATE set |

---

## Cells closed this campaign (cumulative, by name)

| Cell | When | Evidence |
|------|------|----------|
| `gzip:dd79_bin6:L2:T1:size` | PR #332 | lever-332-size-SHIP-5ef737a9.json |
| `gzip:photo.jpg:L2:T1:size` | PR #332 | lever-332-size-SHIP-5ef737a9.json |
| `gzip:photo.jpg:L3:T1:size` | **pending** L3 lever @ `c8bbde67` | hand −7,538 B; **needs `make lever`** |

**Remaining after L3 (estimate): 13 / 1320** — see board table above minus `gzip:photo.jpg:L3:T1:size`.
