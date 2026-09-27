# 363/403 CLOSE-OUT RUNBOOK (2026-09-27, agent-authored)

State at writing: PR #402 MERGED (census receipts); PR #403 (probebudget-48
promotion, pin receipt heads) and PR #363 (good_match machinery + the routing
flip 1234f71e, reviewed BLOCKER-no with byte-neutrality-at-T4 proven by
construction) both OPEN on CI with auto-merge armed. The bt-probebudget lever
is CENSUS-PRICED; flush lever #3 is census-REFUSED at production shape
(record: docs/board/sprint-2026-09-25.md, "bt-probebudget CENSUS RESULT").

## Remaining sequence, with gate conditions

1. **PR #403 merge** — auto on green. After it: sync trunk. The shipped
   binary carries budget-48 (default feature set). Post-merge ledger row 7
   flips to PROMOTED (one close-out record commit does all ledger rows at
   once — step 5).
2. **PR #363 merge** — auto on green; its wall try rides step 3. If CI is
   green but the wall try is still pending, do NOT merge-without-verdict:
   the wall verdict is the PR's own pending item 2.
3. **try363 on the FLIPPED tree** (needs one SSO click on
   `launchdarkly-development`): `/tmp/try363.sh` already targets
   `origin/lever/ldx-good-match` = the flipped head (1234f71e). It runs
   `fulcrum try levels=4,7` (the guard's shallow+deep pairing; the 6,7-only
   set is refused) on c7a.4xlarge with the census-verified bootstrap
   (ami-0fef201115eefe936, subnet 0676cc1b07166ec50, profile
   gzippy-adjudication-ssm-core, /dev/xvda 150G gp3, S3 sink
   s3://gzippy-adjudication-20260923/try363/, self-shutdown + fuse).
   Same visit: fetch the FULL machinery-try cell grid from
   probebudget-census/ip-10-50-6-72/try-363-l47/try.json (the 3
   won-with-margin cells' details).
   Adjudication: the wall verdict is the PR's clause-4 read (gate: ≥1%
   no-regression anywhere decisive + the L6/L7 wall improvement the
   branch's in-process data projects). If the flipped tree re-measures a
   FAIL at L6/L7 vs the legacy-arm byte-streams, the flip reverts and the
   exceptions stay (a failed rule is never rewritten).
4. **#364 lands after #363** — its own 2 commits (len-3 machinery +
   structure receipt), rebased onto whatever trunk then is; its pins
   (one_encode_only's L3 row + the pitch tests) re-verified locally.
5. **Close-out record commit** (records/* branch): ledger row 7 →
   PROMOTED; the L6/L7 exception retirement record; the final failing-cell
   census against the 30 named at plan-2026-09-one-encoder.md §2b (the
   likely movement: good_match makes binary:L6/text:L6 nine cells stay-won
   by construction since bytes are identical; probebudget moves the wall
   cell). Then branch/PR hygiene (delete merged branches, prune, close
   the stale manifest rows in docs/branch-garden-manifest.md including the
   by-then-merged good_match/len3 rows).

## Hard stops

- No wall verdict → no ships on wall-tier claims (the promotion-rule
  clause 8 needs frozen-box paired runs).
- The VOID cell (gzip:silesia.tar:L7:T1, aa_bias 0.0017) re-measures in
  the try363 leg's own grid (levels 4,7 include it).
- SSO lives 1 h: launch the box FIRST, then fetch S3 artifacts while the
  shell still resolves; every leg script shuts its own instance down
  without it.
