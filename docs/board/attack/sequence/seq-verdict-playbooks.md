# Attempt-4 verdict playbooks + the concurrent-brainpower partnership (2026-09-05)

Written while the Box runs attempt 4 (n=30, floors staged, amended rule). Four
outcomes, four pre-verified runbooks. Each ends in a merge state, never in a
guess. The knowledge lanes landing concurrently (docs/board/attack/*) make the
post-verdict turns pre-verified too: read the memo, implement, measure.

## The partnership pattern (why this works)

Reasoning burst + tool construction:
1. **FAN OUT** the inference (multi-agent, one deliverable file per agent, each
   with a named cell/merge-step attached) — cheap parallelism, no box contention.
2. **COMMIT once** and cross-link in the live PR thread (receipts, not floating
   text).
3. **Verify at the box** the winners of each memo (the campaign's rule: never
   generalize, always instrument) — each memo carries its `fulcrum` grid so the
   box time is one command, not one exploration.
4. **Track the DAG**: cell → lever → branch → try → merge; the attempt-4
   playbooks attach to that chain at the arrow the verdict selects.

## Playbook A — SHIP (≤90 minutes)
1. `gh pr merge 370` (rules) — already merged? then skip.
2. `gh pr merge 367 --merge` (the structure slice; the branch tip is
   `4cd5deca` — CI green since the rules sweep; the fulcrum #35 binary is
   already on the box).
3. Fulcrum #35: merge (its CI is the box rebuild's receipt; the box binary is
   already rebuilt — the 86/86 selftest ran from `f3406ff`).
4. Board re-run: `CAMPAIGN_PROMOTE=1 /root/scripts/campaign/board-size.sh all`
   on the merged main (`/root/gzippy` fetch origin main first).
5. First real `parity-census.sh` run (hardened) — 23-file byte-matrix.
6. ir_budget re-pin from the paired-Ir banked numbers (the rows are
   37–53% BELOW main — the raise direction).
7. #363 rebase (`git rebase --onto main 37cb96c7..df31c2a5` shape — branch is
   1 commit) + #364 rebase (2 commits) + scoped `fulcrum try` legs.

## Playbook B — UNDECIDED on NEW VOIDs (never-a-guess list)
1. Collect the VOID set; classify: if the same cells VOID twice (2 of 4 runs
   now have), the cell has a systematic slot-order bias — write it into
   `docs/board/attack/atlas/` as a measurement-class finding, then:
2. `fulcrum wallcensus --n 45` re-measure JUST the VOIDed coordinates with the
   layout floors staged (wallcensus has --pin-reps/--warmup knobs the try
   lacks) — banked runs merge via `wallcensus report`.
3. IF the box stays noisy: the resident vLLM co-tenant is root's own process;
   the ask-back is one sentence to the owner ("the frozen box needs a quiet
   window — kill/pause the vLM serving process for ~4 h, or give wall
   adjudication a dedicated vLLM-free window").
4. Escalate the harness: the A/A certificate could alternate slots (a→b
   ordering) — that is a 20-line paired.rs change killing a systematic
   slot-order bias; PR it into fulcrum (separate, dated).

## Playbook C — NO-SHIP with new rows
Re-diagnose with `fulcrum why` on the new conviction; the falsifier lanes are
already written (the L2 falsification arc, PR #369 lattice) — the Phase 1 or
Phase 3 design docs become the active build targets instead of the lever.

## Playbook D — any clause FAIL (1-roundtrip, flips)
Immediate revert per the standing rule; the branch stays parked; the board
re-run records the reversion receipt; the campaign continues on the next lever.

## The standing "brainpower up front" pattern (this burst's notes)
- **Parallel planning agents** (this run: 8) each produce ONE authoritative
  artifact, committed in one sweep — cheap parallelism, no box contention.
- **Red-team first**: every rule/instrument change gets a hostile reviewer
  before a human sees it.
- **Outcome playbooks**: the verdict arrives into a prepared session.
- **Falsifiers pre-written**: each memo names its own cheapest
  instrument + death modes, so a failed lever costs hours not days.
- **One box at a time**: the box stays the serial bottleneck of
  measurements; inference alone does not acquire measurements — it
  eliminates the pointless ones beforehand.
