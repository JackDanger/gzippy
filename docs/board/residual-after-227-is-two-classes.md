# The residual after #227 is TWO disjoint classes, and one of them is not ours

**Measured 2026-08-01**, deterministic size, T4, levels 2/6/9 × 22 corpus files × 3
rivals = 198 cells. main `120bfa9c` (clean worktree build, sha256 `bbddb09b…`) vs #227
`669e9a0c` (sha256 `9e98a177…`). Not a wall claim. **igzip is not installed on this box
and is not covered.**

With #227 applied: **42 closed, 0 opened, 16 residual, 140 passing both.**
**14 of the 16 are within 0.1% of a tie.**

## Class A — 12 cells vs libdeflate: the seam, and it is TINY

| | |
|---|---|
| cells | 12 |
| **total excess across all 12** | **3,674 B** |
| max single cell | 1,124 B |
| ratios | 1.0000 – 1.0003 |

The worst offender is 1,124 bytes on a 3.3 MB output. Six cells are under 100 bytes.

This is the known zero-headroom class, and CLAUDE.md already says how it closes: *"the
T>1 size leg is NOT closed by making seams smaller… a 2-byte seam fails the cell exactly
as hard as a 2,093-byte one… closed by monotone T1 size wins that buy headroom to
spend."* This measurement is consistent with that and adds the magnitude: the entire
class is **3.7 KB of total headroom debt**.

## Class B — 4 cells vs gzip/pigz: 100% INHERITED FROM LIBDEFLATE

| lvl | file | rival | ours | rival | libdeflate | verdict |
|---|---|---|---|---|---|---|
| 2 | dd79_bin6 | gzip | 4,501,132 | 4,459,419 | 4,500,757 | libdeflate also loses, by 41,338 (0.927%) |
| 2 | dd79_bin6 | pigz | 4,501,132 | 4,464,596 | 4,500,757 | libdeflate also loses, by 36,161 (0.810%) |
| 2 | photo.jpg | gzip | 6,473,565 | 6,470,611 | 6,473,516 | libdeflate also loses, by 2,905 (0.045%) |
| 9 | weights.safetensors | gzip | 83,117,764 | 83,099,652 | 83,117,467 | libdeflate also loses, by 17,815 (0.021%) |

**Inherited: 4. Our own defect: 0.**

On every one of these, libdeflate loses the same cell, and we are only 49–375 bytes
worse than libdeflate. **These cannot be closed by better parity.** They are the
concrete, post-#227 instance of
`libdeflate-parity-is-necessary-but-not-sufficient.md`: closing them requires BEATING
libdeflate, which is phase 2.

## What this says about where the work is

Once #227 lands, at the graded coordinates the board is no longer a "we have a
compression deficit" problem. It is:

* **3.7 KB of total headroom debt** spread over 12 tied-to-four-decimal-places cells,
  closable only by a monotone T1 size win, and
* **4 cells that libdeflate itself cannot pass.**

Neither is addressed by more porting. The port's job — parity — is done at L0-L9.

## Coverage limits

T4 only; **levels 2/6/9 only**, so per CLAUDE.md this is a FLOOR and says nothing about
L1/L3/L4/L5/L7/L8; gzip, pigz and libdeflate only (**no igzip**); size only, no wall;
and this is the residual after a PR that **has not merged**.

Script: `scripts/campaign/reattest227.sh`.
