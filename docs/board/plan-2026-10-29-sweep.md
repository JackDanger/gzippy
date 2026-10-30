# Plan — 2026-10-29: ship the backlog

Source of every item: the findings harvested and fixed across the 2026-10-26/29
review campaign (five parallel code hunts, 12 adversarial review rounds).
Branches that exist but are not merged are listed with their tips.

## Phase A — PR the review-clean branches (no disk dependency)

| # | Branch | Content | Gate |
|---|--------|---------|------|
| A1 | `interop/no-gz-fextra-write` @ `cc96dcd9` | legacy "GZ" FEXTRA subfield retired from every writer; fail-closed `tests/gzip_interop.rs`; manpage rewritten as legacy note | suites green (1222/1223 + interop 2/2); external decode roundtrips run (gzip, pigz, python zlib) |
| A2 | `docs/readme` @ `634dda0d` | README rewrite; `encoder-campaign-plan.md` §8 hooks paragraph | copy verified by round-5/6 reviews |
| A3 | `hygiene/hook-gates` @ `c2194036` | hook-gate blind spots: typechange staging, PARK deletion hole, pathspec shadowing (HIGH), rename false positive + receipts | scratch-repo receipts in-commit |

Actions: push branches, open PRs, let CI run. A1 gets one targeted
round-7 review pass before merge (it has not had its own dedicated
adversarial pass yet — A2/A3 did).

## Phase B — clear the measurement lane

The gate (`/tmp/w1-gate.sh`) fires the W1 `stab×3` chain when the box has
≥9 GB free and the foreign CPU is quiet.

- B1: user-side — drain `/tmp/r3-*` + `/tmp/fixw-probe` (the other
  campaign's ~16 GB), or signal that campaign to release. Nothing here
  touches those directories.
- B2: upgrade the gate into a QUEUE: after W1's chain exits, relaunch
  against `wt-w2` (`2b6d6cf9`) then `wt-w6` (`04c2feec`) automatically —
  three chains, ~4–9 h each, serialized.
- B3: on each SHIP-MERGED verdict → push the wave branch, open the PR
  citing the graded artifact (the stabilizer json + the try run dir), and
  record it on PR #434. On NO-SHIP → honest report per charter, branch
  stays unmerged, pivot to the next lever.

## Phase C — wave-B code batch (small, verified, each needs its own chain)

Order by cost-to-verify; every output-affecting lever merges only after
its own stab×3 (implement + test now, grade when the box allows).

| # | Lever | Source of truth | Chain needed |
|---|-------|-----------------|--------------|
| C1 | wire `output_writer::submit_chunk` into `drain_one_pending` (decode overlap writer) | `output_writer.rs:130` zero call sites; oracle receipt 0.79×→0.98× rapidgzip T8 | wall-leg T4/T8 + stab (byte-transparent by construction) |
| C2 | stdin pipe honors `-p N` (pure pipeline on pipe input) + `<file` stdin mmap arm (`io.rs:659-671`, `:573-596`) | pigz parity gap; stdin-based pins re-baseline | yes (pipe T>1 bytes change class) |
| C3 | L1 tiny-input passthrough wiring (≤51 B bypasses the igzip engine) | `vendor-technique-index.md:518-521`; `startup_cost.tsv` regen | size may grow ≤3 B per tiny file — size pins re-stamp in the same PR |
| C4 | delete the dead content-type sampler open+read (`optimization.rs:58-60`, `io.rs:316-321`) | pure per-file syscalls; `-vv` text changes only | hygiene-scale; still get a try for the binary delta |
| C5 | tail-pad MAP_FIXED overlay kills the 1.0×-input copy fallback (`io.rs:123-126`, `775-785`) | anatomy falsifier named in the hunt | full chain |
| C6 | C1a cost-aware near len-3 branch (L5 `access.log` compat) | `sprint-2026-09-25.md:662-667` | tie-guard then chain |
| C7 | C1b dynamic-block under-splitting (data.sqlite L4 header tax) | `attack/cadence:163-170` ceiling 0.133% | chain |

Order C1–C5 before C6/C7: wall first, then size spend.

## Phase D — anatomy-gated wall campaigns (two-strikes rule)

No build until `fulcrum profile` names the blocked load inside the target:

- D1 zero-run fast path (incompressible spans; the wall-census-complete
  class), falsified shape ≠ this shape; K6 precedent.
- D2 L1 wall re-census (31/51 historic losses, no post-pivot receipt);
  then head-bits 16→14/13 + P12 probe inside the two L1 gates.
- D3 libdeflate's `prefetchw next_hash3` probe (our falsified sibling was
  the chain-node prefetch — different site).
- D4 decode consumer µs batch: marker handoff chain, 100 µs poll,
  member-walk rebuild — lands AFTER C1 (same code path).

## Phase E — hygiene batch-2 (fold into A-branch PRs or a follow-up)

- `docs/production-paths.md` §2: rewrite the stale ISA-L direct-streaming chart.
- `HARDENING-BACKLOG.md`: amend the "0-ignored" invariant for `ratio/squeeze.rs:860`.
- CI: add `cargo clippy -p gzippy-inflate` and a `--features isal-compression`
  release arm to the clippy job (hook covers default features only).
- Delete the orphaned `pr86_body.md` at the project root after copying its
  remaining receipts into PR #434's body.
- W2 note stays: `flush_worker`'s per-chunk `Optimizer::new`
  (`near_optimal.rs:1168`) when `near-opt-parallel-flush` reconnects.

## Phase F — open PR reconciliation

- #433 (improvements batch): rebase onto current main, re-run suites +
  gates; resolve the flag-honesty interplay (`-z`/`-R`/`-K` refuse
  loudly until this lands; after merge the manpage/README blockers lift).
- #434: stays the receipts/campaign record; closed once superseded by the
  wave-A PRs carrying the same receipts.

## Cost accounting

Seven full chains at ~4–9 h each (B + C) ≈ 24–54 box-hours, serialized;
runs overnight while the other campaign sleeps. Phases A/E are a few
hours of small work now; D needs its own anatomy sessions per item and
is deliberately last.

## Standing rules (unchanged)

One lever, one branch, one stab×3 verdict before merge; tie-guard before
any T1-byte change; cross-layout confirm before any wall conviction; never
rewrite a failed rule. The plan file lives in tree so the next agent picks
it up; this file is on `docs/readme` and rides PR A2.

---

## Status ledger (2026-10-30, live)

- All Phase A branches PUSHED + PRs open: #433 wave1/improvements, #434 lever/perf-wave,
  #435 docs/readme (this file; review-clean CI-green), #436 interop/no-gz-fextra-write,
  #437 hygiene/hook-gates.
- PR #438 fix/ci-2026-10-toolchain (opened 2026-10-30): unblocks the two main-carried CI
  gates the 2026-10 runner toolchain bump broke for every branch — clippy scrutinee deny
  (block_fetcher rs:239 bound-form) + symbol-canary BufWriter demangle re-pin
  (std::io -> alloc::io path; apple emit_sequences row removed pending own-box re-pin).
  Both Anatomy legs CI-verified PASS; Build legs pending; other PRs rebase onto it and
  re-run CI (interop already carries the same fix at 2d20a384; Lint green).
- Phase C branches landed: wave-b/decode-output-writer, wave-b/stdin-parallel, wave-b/tiny-input-passthrough,
  wave-b/sampler-removal, wave-b/tail-pad-fixed.
- W1 stab×3 (chunk-buf pool, 52fa20aa) grading — run 1: NO-SHIP with 3 confirmed REAL cells
  (armexe L9-T1 −0.29%; minjs L6-T4 −0.69%; minjs L9-T4 +2.4%; each libdeflate-axes at thin margins;
  3 others acquitted LAYOUT-ARTIFACT). Run 2: NO-SHIP. Run 3 in census (tool.bin corpora); merged
  verdict pending.
- W2/W6 chains queued (/tmp/wave-gate.sh). Wave-B branches enter the queue after wave-A verdicts.
- Fulcrum upstream carries the staleness fixes through 483e413.
- Trainer-box tooling note: libz-ng-sys (zlib-ng) needs cmake — absent system-wide here;
  project-local venv at .tooling/venv-cmake carries it (`PATH=.tooling/venv-cmake/bin:$PATH`).
  release-syms twin takes ~40 min LTO; cached after.
