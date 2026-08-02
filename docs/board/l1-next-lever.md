# The next lever: L1 synthesis gated T>1 — prerequisites, arithmetic, and the trap

Written 2026-08-01 as a HANDOFF, not a result. Nothing here has been measured
today; every number is quoted from a named record and attributed.

## Why L1 and not the wall class I spent today on

`055bd4b5` (#234, ON MAIN) decomposed the size census by **level × rival**
instead of rival × thread. That surfaces the biggest class on the board:

- **L1 = 35 failing size cells, 29 vs libdeflate.** More than any other level.
- **A pure coding deficit, not the seam**: T1 and T4 ratios are near-identical
  (access.log 1.1028 / 1.1032; monorepo.tar 1.0565 / 1.0565).
- Cells are **0.02–10.3% BIGGER**, so unlike the byte-tied seam class there is
  headroom and **a partial improvement closes cells**.

The board's goal metric counts SIZE cells. Today's work was on the WALL axis.

## The mechanism is already in the tree — and it is the SYNTHESIS, not the port

`src/compress/deflate/matchfinder/ht.rs:158`:

```rust
pub struct HtMatchfinder {
    hash_tab:  [[i16; HT_BUCKET_SIZE]; HT_TAB_LEN],  // 2-way buckets (libdeflate)
    hash3_tab: [i16; HT_HASH3_SIZE],                 // length-3 (libdeflate's ht LACKS this)
}
```

That is **attempt 2**, already written, already verified (`fulcrum verify`: 220
cells, 0 roundtrip failures through our decoder plus gzip/pigz/libdeflate at
every thread count). Only the ROUTING was reverted.

## ⚠ READ `parse/mod.rs:540` — it is BINDING and it corrects #234

#234 says "NEITHER attempt died on size". Imprecise. The FALSIFY record is the
authority:

- **Attempt 1 (replacement routing) DIED ON SIZE**: clause 3 VIOLATED with
  **7 pass→fail flips** (absolute), clause 5 violated (armexe.elf +0.0345 on a
  0.0050 budget, 6.9x). 96 → 94 cells: 9 closed, 7 opened.
- The 9 closed are all libdeflate L1 landing at **ratio EXACTLY 1.0000**
  (data.csv 1.0456→1.0000, aozora 1.0405→1.0000, dickens 1.0211→1.0000 …) — the
  transliteration is faithful.
- The 7 opened are ONE mechanism: `fast`'s `head3` LENGTH-3 table earns real
  bytes **on binaries**, where we were WINNING (armexe.elf 0.9658 = a 3.4% win),
  and `ht_matchfinder` deliberately has no length-3. The port hands that back.
- **Attempt 2 (the synthesis) PASSED its size leg** and died on the **WALL** at
  T1 — the same way `17283ee6`/`c0f69036` died in this class.

## The REOPEN contract, and the working-set arithmetic it demands

The record: *"REOPEN requires BOTH — ht's 128 KiB 2-way bucket PLUS a small
length-3 table, replacing the 256 KiB `head` … a combination must show its
working-set arithmetic and take a frozen paired wall run — a size-only argument
is not enough for that shape."*

Arithmetic, from the constants (`ht.rs:126-142`, `fast.rs:1064`):

| table | shape | bytes | KiB |
|---|---|---|---|
| `hash_tab` | `[[i16; 2]; 1<<15]` | 131,072 | **128** |
| `hash3_tab` | `[i16; 1<<15]` | 65,536 | **64** |
| **HtMatchfinder total** | | **196,608** | **192** |

`fast` uses `HASH_BITS = 16` → `HASH_SIZE = 65,536` entries. The record's
"~384 KiB today" and "256 KiB head" are its figures, not re-derived here.
**192 KiB vs ~384 KiB is half the working set** — which satisfies the contract's
arithmetic requirement.

## ⛔ THE TRAP: halving the working set is the argument that ALREADY FAILED

`feedback_bytes_resident_vs_loads_issued`: **halving the working set passed size
and LOST the wall on 19 cells.** And this record says it in its own words: *"The
halved working set was a real reason to expect otherwise and it was not enough."*

So the 192-vs-384 KiB number **satisfies a contract requirement and is NOT the
argument**. Anyone who leads with it is repeating a falsified lever.

## What IS untried: the COORDINATE

Attempt 2's wall verdict was taken at **T1**, against single-threaded rivals,
where slack is 0–8%. Half the L1 board is at **T4**, where slack is 249–330%.
Gating the routing T>1-only is the same move `level.rs` already ships for
`try_exact_huffman` and the depth×4 scaling, for the same stated reason.

**BLOCKER:** that gating needs `params_parallel()`, which exists ONLY on #227's
branch — `git grep 'fn params_parallel' origin/main -- src/` returns NOTHING.
**#227 must land first.** It is not merely its own 42 cells; it is the
infrastructure this class needs.

## The order, cheapest falsifier first

1. Land #227 (re-gate it — its SHIP verdict predates 24 commits on main,
   including `12e93fff`, which removed a `PROBE_CPT` env read from the shipped
   T>1 chunk grid this lever's sibling depends on).
2. Route L1 → `ht_fast` under `params_parallel` ONLY. T1 untouched by
   construction, so the 7 binary regressions cannot recur at T1.
3. **Size at T1 and T4 first** — deterministic, load-immune, runs on a busy box
   (`fulcrum try … --size-only --levels 1-9 --threads 1,4`). If the T4 size win
   is not materially larger than attempt 2's, DROP IT (#234's own pre-registered
   stop rule).
4. Only then the frozen paired wall run at T4, with `aa_bias` read.
5. Pre-register the NO-SHIP branch before measuring: if clause 5 fails at T4,
   the class is closed on the wall at BOTH coordinates — record and stop, do not
   re-sample. (#234 declared exactly this; honour it.)
