# 2026-11-11 — L8 passes=2 pricing (`lever/l8-price-and-ldx-walk-falsifier` @ fc64c606) — SUPERSEDED by PR #446; no box slot owed

Branch `lever/l8-price-and-ldx-walk-falsifier` @ fc64c606 (2026-11-07, **comment-only** —
a `level.rs` receipt text rewrite, 16+/11-, no behaviour change; verified **never
merged** into main, `git merge-base` clean on 2026-11-11).

**The pricing (fc64c606's receipt block, measured 2026-11-06/07 on this Mac,
ladder-tune binary, `GZIPPY_LADDER` override, medians of 5 (T>1) / 9 (T1), vs
the shipped passes=4):**

- T>1 (-p4): silesia-mix x0.783 wall / +0.068% size; bin-heavy x0.776 / +0.085%
  (passes=1 comparison row: x0.643-0.655 / +0.48-0.51%).
- T1 (-p1): silesia-mix x1.058, text-1MB x1.043 — WORSE, both corpora agree.
- The branch shipped nothing and deferred the T>1 leg to the box as its own
  words put it: a NEW no-level routing dimension ("threads beyond 1 → passes=2
  … adjudication belongs to the board") — an open **board question**.

**SUPERSEDED 2026-11-08 by merged PR #446 (`lever/l8-t4-lazy2-reroute`): the
shape the pricing was taken on no longer exists at the L8-T>1 cell.** The
merge dropped the chunked near-opt engine (the L11-knob alias) from L8-T>1
entirely — `params_parallel(8)` → `params_inner(8)`, the plain chunked
Lazy2-300 arm — and it runs:

- x0.12-0.14 of the near-opt route it replaced (`.review-tmp/w6-census/walls.json`,
  walls re-driven 2026-11-10 on clean 90f75548: L8-p2 939.7→109.6 ms x0.117,
  L8-p4 552→77 ms x0.139, L8-p16 x0.131).
- x0.369 of C libdeflate -8 at -p4 (73.4 vs 198.7 ms, interleaved medians of 5,
  in16m; 1a9edd00) — under the price path by ≈5.5x, holding denominators fixed
  (x0.783-of-route vs x0.139-of-route; cross-checked x-C: 0.783 × 2.61 = x2.04
  of C vs the shipped x0.369).
- T1-L8 stays where the pricing assumed parity: the deflate payload is
  byte-identical to C libdeflate -8 after the 10-byte header (cb040be8; the
  only delta is the XFL container byte). The ticket-size cost of the reroute
  (+2.43% at -p4 vs the near-opt route, 1a9edd00) is #446's adjudicated cell,
  not this lever's ledger line.

**VERDICT: no box slot owed.** The x0.783 pricing survives only as a historical
receipt for the near-opt shape — passes=2-vs-passes=4 *inside* an engine the
L8-T>1 cell stopped running on 2026-11-08. A wall question at L8-T>1 now means
pricing the Lazy2-300 shape against its own mechanism (engine work), not
resuming the pass ladder; re-opening the ladder surface at a level would be a
new lever with its own census.

Related receipts on the same branch tip: the ldx pipelined chain-walk
falsifier, separately parked — under the wave-1 noise-floor law the measured
"0.996-1.027×, sign flips per corpus" is a local tie (`tie = NO-SHIP`),
documented in the closed PR #444's receipt (`.review-tmp/p444.md`); it holds
there for the authority box to resume if sub-5% wall deltas ever become
decidable.
