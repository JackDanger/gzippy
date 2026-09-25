# EC2 c7a.4xlarge adjudication — attempt 1 (KILLED) and attempt 2 (running)

## Attempt 1 — timeline (all times UTC, 2026-09-22 → 09-23)

| time | event |
| --- | --- |
| 22:02:19 | `c7a.4xlarge` (dedicated, `ami-0fef201115eefe936`, subnet `subnet-0676cc1b07166ec50`) boots with the v3 user-data |
| 22:0x–22:4x | partial bootstrap (v3's own stages; `--branch` flags meant **single-branch clones** — a latent defect) |
| 22:44 | operator session starts: fulcrum gate physics decoded, rivals built (pigz 2.8, libdeflate-gzip), corpus re-tar verified (12 members, 211,957,760 B) |
| 22:50, 22:53 | two `run`/`calibrate` attempts REFUSED by fulcrum's selfver gate: binary at `owner-2026-09-05/size-spend-ceiling` (5d45e52) ≠ origin/main, and the single-branch clone could not show `origin/main` at all. Gate behavior correct. |
| 22:51 | fulcrum PR #35 fast-forwarded to include the build fix (`73f3367`) and merged → fulcrum main `d738eae` (size-spend ≤ 1% ceiling + standalone build fix) |
| 22:54 | three-part fix applied: refspec widened, `main` checked out, clean rebuild @ `d738eae` — selftest 86/0, gate PASSED (stamp clean, binary == origin/main) |
| 22:54–23:02 | layout calibrate: variant1 `0.9993` (RESOLVED-b-slower), variant2 `0.9999` (NOISY, ci [0.999, 1.000]); floors banked (`layout_floors.tsv`, 684 B) |
| 23:05:31 | wave2 `try s2-fixes/review-1 --base origin/main` started (levels 2,6,9 × threads 1,4 × n=45 vs gzip/pigz/libdeflate; floors loaded) |
| 23:0x | gzippy repo ALSO fixed: refspec widened; `origin/main` = `8748c20f` (PR #370 arbitration merge), ref arm `4cd5deca` |
| ~03:2x | full **sizecensus**: all 12 rival×cell pairs OK. Wins at L9: vs gzip T01 `0.9862` T04 `0.9561`; vs pigz T01 `0.9860` T04 `0.9558`; vs libdeflate T01 `1.0000` (dead tie) T04 `0.9699` |
| ~04:1x–04:2x | full **wallcensus**: all 18 rival×cells resolved; `pin_ok=true` everywhere; 2 cells carry accepted A/A bias notes (gzip L2/T4 VOID-aa_bias=0.0009 ratio 0.2020; gzip L6/T4 VOID-aa_bias=0.0008 ratio 0.1317 — both with alternate resolved ratios) |
| 04:22 | shutdown reservation set for 12:02Z; wallcensus L9 and confirm/aggregation continuing |
| 05:27 | **SSO session expired** (1-h credential from user account) — operator blind |
| 05:55–06:18 | SSO re-login loop; user authorizes at 06:18:12Z |
| 06:02:19 | **instance terminated by the v3 user-data's 8 h in-host fuse** — `( sleep 28800 && /sbin/shutdown -h now ) &` fired at boot+8h, inside the SSO-blind window. Extend attempt at 06:2x hit `InvalidInstanceId` (box gone). Terminate-on-shutdown deleted the 120 GB EBS root with ALL attempt-verdict artifacts. |
| 06:44 | post-mortem: no CloudTrail terminate events visible in the local account trail (org audit not queryable here). Salvage: every census line the wave printed is preserved verbatim in the poller tapes below (full grid captured). |

Salvage quality: raw per-cell census lines for the ENTIRE wave (size + wall, all 18 wallcensus cells) are in the three `.log` files here — copied verbatim from the poller task transcripts. What died with the volume: the formal `try` verdict artifacts, the rescore, and `/root/wave-out/attempt-verdict.txt`.

## Attempt 2 — running (launched 2026-09-23 ~06:45Z)

- New instance `i-043ccd7298e2df783`, SAME launch parameters (type/subnet/SG/AMI/dedicated/gp3-120/delete-on-terminate), **no keypair** (deleted earlier; SSM-only access), instance profile re-used.
- user-data v4 (`/tmp/user-data-v4.sh`, local): full-refs clones (`--no-single-branch` + full refspec fixes), fulcrum at `main` `d738eae`, rivals built from upstream, fresh tar, calibrate → try n=45 → rescore → `/root/wave-out/attempt-verdict.txt` + `RUN_DONE`/`BOOTSTRAP_DONE` markers. Everything logged to `/var/log/gzippy-bootstrap.log`.
- Defused fuse: 12 h from bootstrap end (≈ 18:45Z) instead of the 8 h kill that murdered attempt 1; earliest of this, an external `shutdown` reservation, and my manual `terminate-instances` wins.
- Poller (checkpointing): every 150 s a status fetch whose output is appended to `attempt-2-tape.log` locally BEFORE the next fetch — a box death can no longer burn a measurement.
- Expected: calibrate ≈ 15 min (arm builds ~2×50 s + 2×144 s floor cells), try ≈ 6–7 h, verdict ≈ 13:30–14:30Z.

## Why attempt 1 could not be "resumed"

`try` is a paired-arms whole-run measurement; no resume primitive exists (the confirm queue runs live). A partial census is a screening aid only — the verdict must come from one complete run under one binary. Hence the rerun.

## Reproduction recipe (for next time without re-derivation)

1. Launch `c7a.4xlarge` dedicated in `launchdarkly-development` us-east-1, `subnet-0676cc1b07166ec50`, SG `sg-04bdb4952d5a5aacf`, profile `gzippy-adjudication-ssm-core`, gp3 120, terminate-on-shutdown. NO SSH key needed.
2. user-data: run the v4 script shape above (defused fuse; full refs).
3. Poll via SSM `AWS-RunShellScript` JSON-file pattern (`/tmp/ssm-*.json` on the operator side); keep every fetched tail on the operator's disk.
4. From the box: fetch the small artifacts off (floors .tsv/.json, cell JSONs, verdict txt) — the arm trees are reproducible builds, not evidence.
5. Bank artifacts + tape here; commit `records/2026-09-22-ec2-c7a4xl`; push.
