# T(N) byte-parity unification — design decision v3 (2026-09-25)

Ship shape: **cross-T (N>1) bit-parity + T1 as a distinct, census-tied stream**,
landed as three separable PRs. v3 supersedes v1/v2 after adversarial review
(agent-22, findings block-by-block) plus a direct silesia.tar measurement that
killed the v1 cost ledger.

## What changed v1 → v3, and why

### Withdrawn: exact T1 == T(N) digest parity (v1 plan step 5 / gates)
The gate file asserted `digest(T1) == digest(T4)`. That is unreachable by
chunked encoding: whole-stream T1 matches straddle seams (SOFT_MAX blocks may
run ≤258 B past a boundary; greedy caps `max_len` at the chunk end), so even
identical params/engine/dict/grid give distinct streams — exactly what
`records/2026-09-22-ec2-c7a4xl/FINAL-ADJUDICATION.md:48` measured on the real
corpus. The only route to green is re-routing T1 through the chunked grid
(= make T1 the serial executor of the pipeline); that is NOT tendered. Pricing:
T1 wall risk (per-chunk `LdxCompressor::new` ~1.9 MB scratch + glue copies —
the 2026-08-30 solvency receipt already convicted chunk-count-driven wall at
big files), T1 byte changes on all levels, staleness of the whole pin stack
(`one_encode_only` count contract, perf_shape, ir_vs_ldx, startup_cost, tie
cage). Re-opens it only if the owner raises the size cap above the wall win it
would need; the cap is currently ≤1% — Tooltip: the charter decides:
cross-`-p` equal bytes is what the instrument was built to retire.

### Killed: retiring the near-opt strategy at L9/T>1 (v1 plan step 3)
Direct silesia.tar on trunk 546dc07d (M1, released CLI):
`L1 T4 +0.0021% | L6 T4 −0.34% | L9 T4 −2.91%` vs T1.
Reverting L9/T>1 to Lazy2 = **+2.9% size on the T4/L9 cell** — 10× the v1
"+23,413 B/8 MiB ≈ +0.29%" row (that row double-divides; 23,413/8 MiB *is*
0.279%, and the text-class chunk is +22,926 B/1.8 MB = +1.27%). The owner cap
is ≤1%: the revert is not tendered by ANY bundle. The pigz L9/T4 wall stays
the named open cell, attacked only where the allowance covers it: the trunk's
banked passes2 retune (−16% chunk wall at +0.6% size) is exactly that lever,
and the frozen-box lap adjudicates it (below).

### Accepted: the reviewer's split-bundle (blast-radius control)
One mega-PR = the clause structure's refuse case. Three PRs, each with its own
narrow abort and pin envelope.

## Per-level engine ownership (the table v1 lacked)

| levels | T1 engine | T>1 chunk engine | v3 parity PR |
|---|---|---|---|
| L1, L3, L6, L7 | legacy (`level_uses_ldx` exceptions) | legacy (`params_parallel` level class) | PR-2 |
| L0, L2, L4, L5, L8, L9 | ldx port whole-buffer | pipelined legacy today | PR-3 (ldx dict-chunk) |

PR-2 touches ONLY the first row. Its params-source rule: at each of those
levels the T>1 chunk params stay **the level's current T>1 params** (NOT the
T1 set — at L1 that means `apply_l1_fast_parallel_knobs`; do NOT import the
T1-only `apply_l1_match_reach_t1_knobs`, whose T4 pairing is the adjudicated
NO-SHIP `pigz:ecoli.fastq:L1:T4:wall` cell). Parity = engine, grid, header
budget, and flush seams identical across T with per-level params frozen; T1
stays whole-buffer.

## PR-2 — cross-T parity at the legacy levels (the first landing)
1. `pipelined_block_size` loses the `by_parallelism` arm: the canonical grid is
   the serial bound per level — the L6+ 2 MB clamp
   (`MAX_T_AWARE_BLOCK_SIZE_L6_UP`), the L1–L5 8 MB bound — with chunk count
   computed from input length only (`ceil(input/grid)`), never from `threads`.
   This keeps the shipped big-file walls (the 2 MB cap is the measured state)
   and removes the thread-coupling in one stroke.
2. Header budget: one source for the legacy chunk path (the T>1 Generous
   budget retires in favour of whichever the T>1 rows at these levels are
   already pinned to; no size-spend is authorized here — the census decides,
   and any cell beyond the tie tolerance blocks the PR).
3. `compress_exact_to_writer` (T1) is untouched. T1 streams keep their
   whole-buffer whole-stream shape.
4. Gates rewritten (same commit): `tests/thread_byte_parity.rs` asserts
   `digest(T2) == digest(T4) == digest(T8) == digest(T16)` across a seam-heavy
   payload + roundtrip at L1/L3/L6/L7; T1-vs-T(N) moves to the census
   instrument (size ratios on the real-corpus fixtures within tie tolerance,
   roundtrip-identity always). The T1 leg of the digests is deleted with the
   note above as the reason it can never hold.
5. Pin regen, in the SAME commits as each behavior flip: perf_shape rows for
   the touched levels (grid change alters per-fixture chunk counts → anatomy
   rows move), `seam_tax`, `startup_cost` fingerprints; the routing pin
   `t1_vs_parallel_l897_routing_asymmetry_is_deterministic` is inspected
   against the new layout (L8/L9 route-reading must still be deterministic —
   it pins a level.rs decision, not a byte); `one_encode_only` count contract
   (per-input encodes are unchanged — chunking count is not the pin's arity).
   `ir_vs_ldx` stays untouched (T1 arms unchanged).
6. Local gates: size census across the representative corpora at the touched
   levels (per-cell ≤ tie tolerance), wall microprobe (T4 on dense + binary)
   before ANY push.

## PR-3 — ldx dict-chunk engine at the port levels (deferred, gated)
Needed for cross-T parity at L0/L2/L4/L5/L8/L9 (T1 is whole-buffer-port there
— the shipped engine; T>1 must then run ldx per chunk or the streams' engines
differ at every port level). The ldx chunk protocol needs, per the review:
- `is_last` / starting-offset (`data_start`) parameters — BFINAL must be
  suppressible mid-stream, and only the final chunk takes it.
- an alignment report for stored blocks (`ChunkMeta{pad_bits,
  needs_alignment}`-equivalent) so the splice protocol can lock seams, and
  passthrough keyed on the DATA extent (not the glued length) so tiny chunks
  take stored blocks instead of tripping the MIN_BLOCK invariant.
- one glue copy per chunk (dict + data joined) — the "no scratch, no cop"y
  `compress_into` contract is amended for this API variant only;
  hc-matchfinder's sliding rebase makes a disjoint two-slice window unsound.
Gates: three-oracle seam probe (roundtrip parity through the oracles at
`k*grid ± {0..64}`, stored-fragment coverage, passthrough edges), then the
per-cell census at every touched level before its routing flips, then the
Ir/budget rows and `ir_vs_ldx` regen on the box.
This PR does NOT block the lap; its first gate is the probe, landing only if
the census stays inside the tie tolerance.

## PR-1 — the wall lap (already staged)
The frozen-box session adjudicates trunk's passes2 retune vs aa682fcc per
`docs/board/retune-wall-lap-runbook.md` (pigz:silesia.tar:L9:T4:wall trigger).
Revised fallback struct: the v1 "revert to Lazy2 pair-for-pair" leg is dead by
the cap; if ABORT fires, the adjudicated fallback is keep-size + accept the
wall cell + open a NEW wall lever under the cap (the lever hunt is the box
session's brief).

## Stale-comment sweeps owed (review minors)
- `src/compress/ldx/mod.rs:225-228`: "test/differential entry point, not a
  shipping one … nothing routes here" — wrong since the production routing
  landed; amend to "the port's output entry point; chunked regime also uses
  the dict-chunk API below when PR-3 lands".
- `docs/board/sprint-2026-09-25.md` §P2 reference to "sprint plan §7" — the
  deferred L1-in-ldx item is tracked here (PR-3), not in a nonexistent §7.
- `docs/board/sprint-2026-09-25.md` line 11's `level.rs:539` citation →
  `level.rs:560-570` on this branch.

## Sequencing
Box lap first (PR-1's adjudication) — it is already staged and SSO-gated.
PR-2 lands from microbench + census receipts AFTER the lap's trunk-shifted
pins re-bind. PR-3 lands per its own gates, independently.
