# L1 is ALGORITHMIC, not implementation — we emit 2.89x more literals than libdeflate

`fulcrum why libdeflate:data.csv:L01:T01:size` on trainer, 2026-08-01. The worst
residual L1 cell (1.045601 after #227). **Layer [1] is deterministic** — position
counts are byte-exact properties of the output — so box noise is irrelevant here.

    [1 STRUCTURE] POSITION COUNTS DIFFER (matches Δ4.38%, matched-positions Δ1.85%,
                  literals Δ189.41%): different parse decisions — the gap is ALGORITHMIC.

| | ours | libdeflate |
|---|---|---|
| tokens | 2,587,312 | 2,186,764 |
| matches | 1,846,129 | **1,930,665** (+4.6%) |
| **literals** | **741,183** | **256,099** |
| total bits | 32,893,788 | 31,459,208 |
| header bits | 166,614 | 166,000 (near-identical) |

**We emit 2.89x more literals and find 4.4% fewer matches.**

## Why this is the most useful thing measured today

Every other coordinate examined this session — L2/T1, L5/T1, L9/T1 — reported
**POSITION COUNTS MATCH at Δ0.00%**: same tokens, same matches, same literals,
byte-identical output. Those gaps are pure implementation, which is why the L2
component map is about where instructions go.

**L1 is the opposite.** The parse itself is worse. That is a different KIND of
problem and it admits a different kind of fix — and unlike an implementation gap,
a better parse pays in SIZE, which is the axis the board is scored on.

Note the header bits are essentially equal (166,614 vs 166,000), so this is not a
header/seam effect. It is the match search.

## The mechanism is known and the fix is already written

Our L1 is an igzip-class **chainless single-probe** matchfinder (`Strategy::Fast`,
one probe per position). libdeflate's L1 is `ht_matchfinder` — **2-entry buckets**,
so it gets a second candidate per hash. A single probe that misses falls back to a
literal, which is exactly the 2.89x literal excess measured above.

`matchfinder::ht::HtMatchfinder` (ht.rs:158) already implements the synthesis:

```rust
hash_tab:  [[i16; HT_BUCKET_SIZE]; HT_TAB_LEN],  // libdeflate's 2-way buckets
hash3_tab: [i16; HT_HASH3_SIZE],                 // the length-3 table THEY lack
```

`fulcrum verify`-clean at 220 cells, 0 roundtrip failures. Only the ROUTING was
reverted (`parse/mod.rs:540`).

## This is the FOURTH independent sighting of L1

1. `055bd4b5` (#234) — L1 = 35 failing size cells on the 22-file census, the
   largest level, 29 vs libdeflate.
2. Today's M1 wall census — the worst single wall cell is
   `libdeflate:data.parquet:L01:T01 = 1.3328`; the only igzip losses are L1/L3.
3. The #227 re-gate residual — L1 is 16 of 37 (43%) and holds the four worst
   ratios.
4. **This**: the L1 gap is ALGORITHMIC where every other level's is
   implementation, with a named mechanism (single probe vs 2-way bucket) and a
   measured magnitude (2.89x literals).

## What is NOT claimed

- One file (data.csv), one level, one thread count. The Δ189.41% literal excess
  is data.csv's; other files will differ in magnitude.
- **That routing L1 to `ht_fast` closes these cells.** The binding FALSIFY at
  `parse/mod.rs:540` records that the straight replacement OPENED 7 binary cells
  (clause 3, absolute) because `fast`'s `head3` length-3 table wins on binaries —
  armexe.elf was a 3.4% WIN we handed back. The synthesis (buckets AND hash3) is
  what must be routed, gated T>1, and it must still show working-set arithmetic
  and take a frozen paired wall run. See `docs/board/l1-next-lever.md`.
- Layer [2] in this run is UNTRUSTWORTHY: trainer's fulcrum is `8364a05`
  (origin/main), which predates the callgrind parser fixes — it reported 77.77%
  of Ir on an unattributed `:0` line. Only layer [1] is quoted here.
