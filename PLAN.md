# gzippy encoder campaign — handoff

**Date:** 2026-08-17  
**Done when:** **0 / 1320** failing GATE promotion-board cells (size AND wall, per-label, T1 and T>1).  
**Not done when:** a session summary, a partial lever, or a hand measurement without `make lever`.

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
