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

## PR-2 — cross-T parity at all pipelined levels (the first landing)

REVISED v3.1 after the second adversarial review (agent-23) + the P4/P4b
probes (sprint doc): the grid function is GLOBAL — the change is not scoped to
the legacy-engine rows; the census envelope covers every pipelined level.

1. `pipelined_block_size` anchors `target_chunks = GRID_REF_THREADS × cpt`
   (the T=4 layout bit-for-bit: `num_threads = 4` reproduces the former arm's
   exact value), dropping the LIVE thread count. The surviving structure is
   UNCHANGED and load-bearing: the 512 KiB `MAX_PARALLEL_BLOCK_SIZE` floor,
   `clamp(MIN, max_block)` (2 MiB L6+ / 8 MiB L1–5), `min(input_len)` and the
   `SOFT_MAX_BLOCK_LENGTH` alignment (the 2 MiB clamp floor-aligns to
   **1,800,000**, the number the tests already pin). Any implementer taking
   "2 MiB" literally without the alignment gets 2,097,152 and changes the
   silesia cells — do not.
2. Header budget: the chunk path keeps `HeaderBudget::Generous` at every T>1
   (keyed on `parallel`, never on T); T1 keeps `Lean`. NOTHING changes bytes
   here — the freeze is a no-op by construction.
3. `compress_exact_to_writer` (T1) is untouched. T1 streams keep their
   whole-buffer whole-stream shape and stay a distinct stream class (digest
   distinctness is structural; the gate asserts the 0.5% size tie instead).
   `one_encode_only`'s count contract verified intact (chunk workers do not
   increment the whole-buffer counters).
4. Scope claim: cross-`-p` digest parity is asserted for inputs above the
   L2–L5 routing-escape threshold (102,400 B — `optimal_thread_count` halves
   tiny `-p2` requests into the T1 whole-buffer encoder; a pre-existing,
   separately-receipted T-ROUTING rule). The gates derive payloads from
   `pipelined_block_size`, never hardcode grids.
5. Wall envelope: T2/T4 identical by construction; T8/T16 measured
   wall-neutral-or-better on the 32 MB probe (pinned 10–50 ms vs today
   20–60 ms medians) — the runbook lap re-verifies on c7a with the
   will-not-regress rule at every (level, T) cell it sweeps.
   available_parallelism() anchoring was CONSIDERED and REJECTED (per-host
   pins; T2 big-file fixed-cost overhead — the 0.3 s/chunk ledger class).
6. Pin regen, in the same commits as the behavior flip:
   `tests/fingerprints/ours.tsv` + `ours_t4.tsv` (T4/T1 grids unchanged ⇒
   expect byte-identical pins; regen to PROVE), `perf_shape` rows (grid
   changes move per-fixture chunk counts in the un-clamped band),
   `seam_tax`, `startup_cost` (1-byte grid = 128 KiB min either way; empty
   diff expected), ir_vs_ldx untouched (T1 arms), routing pin
   `t1_vs_parallel_l897` re-binds with PR-1's retune only.

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
