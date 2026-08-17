# Encoder campaign — handoff (2026-08-17, pre-reboot)

## Goal

Drop-in gzip/pigz/libdeflate/igzip encoder: valid gzip, per-label **size AND wall** wins at
levels 0–9 on every corpus file, T1 and T>1. **Done = 0 failing GATE board cells**
(1320 measured at promotion time).

## Board baseline — main @ `e888ac9f` (merged #332)

**`gzippy-bench/campaign/size-all-e888ac9f/summary.txt`:** **14 / 1320** failures.

| Rival | Failures | Worst |
|-------|----------|-------|
| gzip | 11 | `dd79_bin6` L3 T1 +0.36% |
| libdeflate | 1 | `weights.safetensors` L4 T4 +0.04% |
| pigz | 2 | `dd79_bin6` L3 T1/T4 +0.23% |

#332 closed **`gzip:dd79_bin6:L2:T1:size`** and **`gzip:photo.jpg:L2:T1:size`** (artifact:
`gzippy-bench/campaign/promotion-records/lever-332-size-SHIP-5ef737a9.json`).

## In-flight — branch `lever/l3-gzip-deflate-fast-pickmin` @ `c8bbde67`

**NOT merged.** Unpushed. Implements L3 mmap pick-min (mirror L2):

- `level_uses_t1_mmap_pick_min` → levels `1..=4`
- Arms: shipped `params(3)` Lazy, `params_l3_libdeflate_greedy()`, `params_l3_gzip_deflate_fast()` (chain 32, nice 32, hash3 chain in fast arm only)
- Files: `src/compress/deflate/mod.rs`, `src/compress/deflate/level.rs`

### Hand-measured (campaign cmd: `gzippy -{L} -p 1 -c $CORPUS/file`)

**Important:** `-c file` hits **PureT1Mmap** pick-min. Stdin redirect (`-c - < file`) hits **PureT1** streaming and is **not** the board path.

| Cell | main `e888ac9f` | branch `c8bbde67` | Rival | Verdict |
|------|-----------------|-------------------|-------|---------|
| **`gzip:photo.jpg:L3:T1:size`** | 6,472,401 (+2,723) | **6,462,140 (−7,538)** | gzip 6,469,678 | **CLOSES** |
| **`gzip:dd79_bin6:L3:T1:size`** | 4,452,246 (+15,824) | 4,448,460 (+12,038) | gzip 4,436,422 | Improved −3,786 B; **still FAIL** |
| `pigz:dd79_bin6:L3:T1:size` | +10,276 | +6,490 | pigz 4,441,970 | Improved; **still FAIL** |
| libdeflate ties (photo L3, weights L3) | unchanged | unchanged | — | OK |

### Gates run on `c8bbde67`

| Gate | Result |
|------|--------|
| `tie-guard.sh HEAD` (vs `origin/main`) | **0/33 flips** PASS |
| `fingerprint_suite` + `block_pins` | **PASS** (no pin regen needed) |
| `CAMPAIGN_PROMOTE=1 lever.sh` | **NOT RUN** — do first after reboot |

### Expected lever outcome

Likely **partial SHIP**: closes `gzip:photo.jpg:L3:T1:size`; dd79 L3 gzip/pigz T1 still open.
Does **not** touch T4 cells (mmap pick-min is T1-only).

## Remaining GATE failures after L3 merge (estimate **13 / 1320**)

**gzip (10):** `dd79_bin6` L3 T1/T4; `photo.jpg` L1/L2/L3 T4; `weights.safetensors` L7–L9 T1/T4.

**libdeflate (1):** `weights.safetensors` L4 T4.

**pigz (2):** `dd79_bin6` L3 T1/T4.

(T4 L2 `photo.jpg` / `dd79_bin6` remain open — T>1 seam class, not fixed by mmap pick-min.)

## Next agent — priority order

### 1. Size lever for L3 (first command after reboot)

```bash
cd ~/www/gzippy
git checkout lever/l3-gzip-deflate-fast-pickmin   # @ c8bbde67
cargo build --release
scripts/campaign/tie-guard.sh HEAD
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-c8bbde67-size \
  scripts/campaign/lever.sh c8bbde67 --size-only
```

If SHIP → PR, merge, re-board:

```bash
CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all
# expect ~13/1320 if only photo L3 T1 closes
```

### 2. dd79_bin6 L3 — still algorithmic gap

```bash
fulcrum why gzip:dd79_bin6:L3:T1:size --repo .
```

Pick-min arms exhausted the vendor-diff levers (gzip deflate_fast + libdeflate greedy). Next
mechanism is parse/block-boundary, not another knob copy from L2. Do **not** ship global
hash3 or strategy changes without tie-guard + L6/L9 spot check.

### 3. Wall leg for #332 (optional record)

```bash
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-e888ac9f-wall \
  scripts/campaign/lever.sh e888ac9f --threads 1
```

Prior run killed mid clause-8 confirm (~3h). Pick-min 2+1 threading fix is in `e888ac9f`.

### 4. T4 failures

`photo.jpg` L1/L2/L3 T4, `dd79_bin6` L2/L3 T4 — T>1 path; seam class closed per CLAUDE.md.
Need monotone T1 headroom or T>1-specific parse lever, not mmap pick-min.

### 5. `weights.safetensors`

- libdeflate L4 T4 (+0.04%) — thin margin, T4 only
- gzip L7–L9 T1/T4 — separate class

## Disk cleanup done (this session, pre-reboot)

| Removed | ~size |
|---------|-------|
| `gzippy-bench/breadth_tmp/` | 774 MB |
| `gzippy/benchmark_data/` | 735 MB |
| `/tmp/gzippy-*` worktrees | ~420 MB |
| Stale git worktrees (`/private/tmp/gz-*`, `gzippy-*`) | ~560 MB |
| Large agent-tools log blobs | ~130 MB |

**Kept:** `target/release/gzippy` (~889 MB), `gzippy-bench/corpus/`, `gzippy-bench/campaign/size-all-e888ac9f/`, promotion record JSON.

**Safe if more space needed:** `cargo clean` in gzippy (~889 MB), `gzippy-bench/{weights_diag,decision_diag,residuals_diag}/` (~390 MB diag scratch).

## Key files

- `src/compress/deflate/mod.rs` — `deflate_one_shot_t1_l3_pick_min`, `encode_gzip_unpadded_l3_pickmin`, routing `1..=4`
- `src/compress/deflate/level.rs` — `params_l3_libdeflate_greedy()`, `params_l3_gzip_deflate_fast()`
- `src/compress/io.rs` — T1 mmap threshold: file > 128 KiB → `encode_gzip_unpadded_slice_to_writer` (board path)
