# T(N)==T1 byte-parity unification — the design decision (2026-09-25)

## The problem
`-p N` streams differ from T1 bytes because the chunked pipeline (a) chunks with a
thread-dependent grid in the un-clamped band, (b) runs `level::params_parallel`
(depth ×4, Generous headers, try_exact_huffman, good_match=0, and at L8/L9 the
whole near-opt parser) instead of T1's params, (c) restarts matchfinder state at
seams. All decode-identical; but equal-bytes across -p was the campaign's charter
hypothesis that this instrument finally measured on the real corpus.

## The decision options (workshop defaults bolded)

**Option A — T1-adopt-the-stronger-parse.** Promote near-opt at L9-T1, Generous
headers at every T, drop the params_parallel params divergence. Cost: T1 wall
triples at L9 (measured 2.6x on the chunk probe, banked) — the wall budget isn't
tendered. Not viable until the T1 wall deficit closes.

**Option B — chunked engine per level (machine-identical, wins Sizes)**: put the
SERIAL route's params into the parallel pipeline (params, Lean budget, no
try_exact_huffman, level_uses_ldx extended to support (dict, data)). Cost: T4 L8-9
size reverts (+23,413 B/8 MiB, inside the ≤1% cap with paired-wall), wall wins
everywhere measured. Requires extending `compress::ldx::compress_into` with a dict
window — the deferred L1-in-ldx item (sprint plan §7) — as the mechanism.

**Option C — the stitched-stream, parallel subconscious**: keep parallel params
for wall; make only the GLUE identical (framing + emit boundaries pinned to the
grid seams, synced flushes exactly at seams, and dict seams identical to Serial's
windows) — which needs grid determinism (Option B's step 2, WITHOUT the params
unification). Cost: streams stay distinct unless the per-chunk stream boundaries
coincide with the final block boundaries in both routes. Measured: T4 already
BLOCK-ALIGNED at silesia-levels (bin/tab/txt body rows stop in one block) — this
makes A and B colocate more often than not. Complexity: the grid must be pinned
(canonical T4) so the layout is stable.

## Positions from the receipts
- The per-cell census at silesia.tar is +0.0061% (passes2). The wall win is the
  thing that matters; bytes-per-cell for these fixtures is firm.
- The `pigz L9/T4` and `libdeflate L2/T5` cells are the two named LOSSES. Option B
  reopens both walls toward parity-or-win and keeps size inside the cap. It is the
  only option where the compressor keeps winning at T4 while the encoder becomes
  ONE code path per level.

## The unification plan (Option B-first)
1. Extend `compress::ldx` to accept a (dict, data) window (the L1-in-ldx sprint
   item, agent-17's P2 sibling). Unit probe: roundtrip parity through the three
   oracles at seam offsets k*grid ± {0..64}.
2. Extend `pipelined_block_size` to drop the parallel-multiplier arm (pin to the
   canonical serial grid); the grid pin test (thread_byte_parity grid-layout pin)
   turns green here.
3. Route `deflate_into`'s parallel=true branch at L8/L9 through the chunked-SERIAL
   params (params(9) at depth 600, Lean header) and REMOVE the near-opt at T>1.
   The params_parallel function retires.
4. The remaining footprint: try_exact_huffman (T1 keeps off; T>1 must switch off
   in the chunk pipeline for parity), the Generous-vs-Lean header budget (Lean
   for parity), the seam-state restart (agent-18 §3.4 — accept as the cost of
   chunking; it is what the byte-identity tolerance covers), and the x4 depth
   leak (retires with params_parallel).
5. Byte-parity gates turn green: `tests/thread_byte_parity.rs` at all levels, the
   grid pin, the seam proptest. `won_cells_stay_won` re-verifies the ledger rows
   still won (expect +2,258 B stays: L9/T4 size vs gzip still won).
6. Then: the Ir budgets at the changed cells re-derive (regen UPDATE_IR_VS_LDX),
   the fingerprint pins regenerate per the retune discipline, and the frozen-box
   wall lap decides both the parity unification AND the L9 strategy revert in one
   run + one adjudication.

## The costs we accept (ledger rows come itemized)
- L9-T4 size: +23,413 B/8 MiB (the banked pre-restructure row) — the pinned near-opt
  upgrade's size win is what we are trading for the parity + the 0.58x wall.
- The x4 depth multiplier's byte savings at WHAT contributed — witness: the depth
  split measured a no-op on dense corpora; the multiplier only mattered at depth
  400 near-opt — it retires with the strategy.
- The L1/L3/L6/L7 legacy-route exclusions stay UNLESS the evidence demands; the
  parity fix does not need to retire them — those routes are T1-only and the
  parallel pipeline stopped using them after this retune.

## ⚠ CORRECTION (2026-09-25, direct M1 measurement on the real corpus)

The listed L9-T4 cost row is WRONG by ~10x. Direct silesia.tar CLI, trunk
546dc07d, released build:
`L1: T4 +0.0021% | L6: T4 −0.34% | L9: T4 −2.91%` (T1 66,715,402 B →
T4 64,771,367 B at L9). The near-opt@L11-knobs parse buys −2.9% corpus size at
T4 — consistent with probe 1b's own chunk numbers (+3.5% if reverted) — so:
- Retiring the near-opt at L9/T>1 (plan step 3) costs ~**+3% size on the whole
  T4/L9 cell**, NOT +0.29%. It breaches the owner's ≤1% cap and is NOT tendered.
- The 0.29% figure double-divides (23,413 B/8 MiB *is* 0.279% — treating the
  same number again as if it were per-1MB). The mixed-corpus corpus dilutes the
  text-class loss; the text-class chunk cost remains +22,926 B/1.8 MB.
- CONSEQUENCE: step 3 must be dropped or re-scoped. The viable parity shape is
  cross-T parity (T2==T4==T8==T16 bit-identical at a pinned grid, keeping each
  level's T>1 params — the banked passes2 retune stays) with T1 remaining a
  distinct stream (whole-buffer port), which is exactly the alternative the
  FINAL-ADJUDICATION names ("retire via the parity-census instrument (or by
  making the parallel path bit-match T1)"). Bit-matching T1 from the parallel
  side would additionally need the seam-match hold-back that no option here
  prices; NOT in this design.

## Sequencing
This unification lands AFTER the frozen-box lap adjudicates the passes2 retune —
the same box session covers both (the retune lap runs first; the unification's
wall*size lap runs after, on the same box).

REVISED per the correction above: the unification ships the *cross-T parity*
shape only (pinned grid, shared params per level — steps 1, 2, 4, 5 with step 3
dropped). If the retune ABORTs at the pigz L9/T4 trigger, the located fail is
not resolved by reverting the strategy (that now costs +3% size, past the cap);
the abort adjudication must instead pick between keeping the size (wall loss
stays) and a NEW wall lever priced under the cap — that pick is the box
session's job, not this doc's.
