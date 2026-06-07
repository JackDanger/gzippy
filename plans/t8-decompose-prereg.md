# PRE-REGISTERED FALSIFIER — decompose the T8 seeding bundle (2026-06-07)

## Goal
The Phase-0 oracle proved: pure-Rust ENGINE already TIES rapidgzip at T8 once
`GZIPPY_SEED_WINDOWS` is on (0.97-1.00×), but unseeded production LOSES (0.65-0.69×).
The advisor flagged that the `GZIPPY_SEED_WINDOWS` knob BUNDLES three removals:
  (a) u16 marker-COMPUTE  — skipped because each chunk gets its correct predecessor window.
  (b) block-finder REAL-boundary pre-seed vs prod partition-GUESS — ALIGNMENT.
  (c) the 13 header-SPECULATION-FAILURE re-decodes — eliminated when (a)+(b) hold.

We must separate which sub-lever dominates the ~1.5× unseeded→seeded T8 wall gap
BEFORE choosing the Phase-1 fix.

## The decomposition (two new measurement-only knobs, OFF==identity, byte-exact)
The seed file already holds two independently-consumable parts:
  - `windows` HashMap (start_bit→32KiB) consumed at chunk_fetcher.rs:2269 `seed_window_for`.
  - `seedable_chunk_starts()` (boundary keys) consumed at chunk_fetcher.rs:498-505
    `block_finder.insert(off)`.
Knobs (only meaningful WITH `GZIPPY_SEED_WINDOWS` set):
  - `GZIPPY_SEED_NO_BOUNDARIES=1` ⇒ SKIP the block_finder pre-seed loop. Keeps windows.
    = **seed-ONLY-windows** (windows given; boundaries stay prod partition-GUESS).
  - `GZIPPY_SEED_NO_WINDOWS=1` ⇒ make `seed_window_for` always return None. Keeps the
    block_finder pre-seed. = **seed-ONLY-boundaries** (real boundaries; chunks still
    speculate/marker-decode because no window is handed to them).

Byte-exactness: both knobs only ever REMOVE a seeded shortcut and fall back to the
correct production path (speculation is byte-correct; partition-guess boundaries are
byte-correct). Output sha MUST stay 028bd002…cb410f every run. A non-matching sha VOIDS.

## Four production-pipeline T8 wall cells (locked guest, interleaved measure.sh, sha-verified)
  1. prod      — no seeding (baseline LOSS, ~0.69×)
  2. seed-full — GZIPPY_SEED_WINDOWS only (TIE, ~1.00×)  [reproduce the bundle]
  3. seed-only-windows   — GZIPPY_SEED_WINDOWS + GZIPPY_SEED_NO_BOUNDARIES
  4. seed-only-boundaries — GZIPPY_SEED_WINDOWS + GZIPPY_SEED_NO_WINDOWS
All vs rapidgzip (rg) as 1.000 anchor, same run, interleaved.

## PRE-REGISTERED MAPPING observation → sub-lever (decide BEFORE seeing numbers)
Let W_prod, W_full, W_onlyW, W_onlyB be the T8 walls (lower=faster). The bundle gap
to close is G = W_prod − W_full. Define the recovered fraction of each isolated knob:
  f_windows  = (W_prod − W_onlyW) / G    (how much of the gap windows-alone recover)
  f_boundary = (W_prod − W_onlyB) / G    (how much boundaries-alone recover)

Decision rule (point estimates; "dominates" = recovers ≥ ~60% of G AND its Δ exceeds
inter-run spread; "TIE" if Δ < spread):
  - If f_windows ≳ 0.6 and f_boundary ≲ 0.3  ⇒ **marker-COMPUTE (a) dominates.**
    The window-absent u16 decode itself is the lever. (asm/inner-kernel work may apply
    HERE on the marker path — ISA-L can't emit u16 so the Phase-0 oracle never tested it.)
  - If f_boundary ≳ 0.6 and f_windows ≲ 0.3  ⇒ **boundary-ALIGNMENT (b) dominates.**
    The lever is the block-finder / prefetch-horizon (project_confirmed_offset_prefetch_gap),
    faithfully ported from how rapidgzip aligns dispatch.
  - If BOTH f_windows and f_boundary are large (≈ each recovers most of G, i.e. they
    OVERLAP / are non-additive ⇒ f_windows + f_boundary ≫ 1) ⇒ the two are COUPLED:
    real boundaries enable correct windows and vice-versa; sub-lever (c) speculation-
    FAILURE is the shared mechanism (both knobs cut the 13 re-decodes). Then the lever
    is **better speculation/confirmation** (reduce re-decodes), and we corroborate with
    the spec-failure counter (`13` in prod) measured per cell.
  - If NEITHER recovers ≥ 0.3 of G alone but seed-full ties ⇒ the gain is SUPER-ADDITIVE
    (needs both windows AND boundaries together) ⇒ same as the COUPLED verdict (c).

## Corroborating counters per cell (read from --verbose / debug stderr)
  - header-speculation failures (prod=13, seed-full=0): per cell.
  - Pool Fill Factor (prod 77%, seed-full 90%): per cell.
  - finished_no_flip / window_seeded mode split: per cell.
  - seed_hits/seed_misses (self-test the knob actually changed the path).

## Controls / disproof
  - OFF==identity: with neither NO_* knob, seed-full must reproduce ~1.00× (else the
    knob wiring regressed the bundle).
  - sha-verify EVERY cell (028bd002…cb410f). Any mismatch ⇒ VOID, do not interpret.
  - Inter-run spread: ≥2 independent interleaved runs; report spread; Δ<spread ⇒ TIE.
  - Self-test the knobs: seed-only-windows must show seed_hits>0 (windows still used);
    seed-only-boundaries must show seed_hits==0 / seed_misses>0 (windows suppressed) yet
    the block_finder still pre-seeded (real boundaries) — verify via the mode split
    (chunks land clean? no — window absent ⇒ speculative, but at REAL boundaries).
