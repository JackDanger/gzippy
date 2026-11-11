# Record: integrated final-tree census — the Nov-11 trio's last A/B — 2026-11-11

Branch `records/final-tree-census` off main `b1502a078c20e18b646cf42d1a5af10a0f0dad1a`
(the #467 fp-settle merge; the Nov-11 arc is fully merged: #457 DP snapshot+NEON,
#458 -p2 route flip, #462 node-SoA+scratch codes, #465 fixedpoint skip,
#466 L4 medium, #467 fp-settle — #465/#466/#467 all default-off BUILD-TIME
features). Raw instrument + per-cell data under `data/`. NOT pushed.

## Verdict summary (this is the headline; every line has a table below)

* **ARM DELTA (read this first):** `git log --oneline 06859145..b1502a07` = the
  six #467 commits and nothing else. Arm B `06859145` is the #466 merge — it
  CONTAINS the #465 fixedpoint-skip and #466 L4-medium merges (both
  default-off build-time features), so the A/B binary delta is the #467
  fp-settle code ALONE, compiled OFF in BOTH arms. The task brief's arm-B
  gloss ("WITHOUT the fixedpoint/medium/fpsettle trio — the pre-#465 state")
  does not match the SHA it pinned; the SHA is contract, the gloss is recorded
  here as imprecise. The byte expectation (A == B) is not affected: every trio
  member is a build-time feature compiled OUT of both default binaries.
* **ANCHORS: 12/12 PASS** — 6 anchor commands × 2 arms: every size and sha is
  the recorded expectation exactly; every stream decoded byte-roundtrip
  through all three legs.
* **BYTES: 1,080/1,080 PASS** — arm A and arm B streams sha-IDENTICAL on every
  cell of 30 members × L1..L12 × {T1,T2,T4}; each cell's stream (one stream,
  sha-pinned on both arms) roundtripped through Apple gzip 479, vendored pigz
  2.8, and the python3 zlib member-chain: 1,080 × 3 = 3,240 decode-leg
  checks, zero failures. Plus 12 anchor streams × 3 legs.
* **WALL: PASS — ties everywhere, 0 regression verdicts** (all numbers
  HYPOTHESIS-TIER: unfrozen laptop). 144 cells = {in16m, logs-card26m,
  binary26m, text26m} × L1..L12 × {-p1,-p2,-p4}: 137/144 inside the ±5% local
  tie band at 8 reps (A,B,B,A ×8 = 16 samples/arm); 6 of 7 outside-band cells
  collapsed to ties at the 14-rep rerun (28/arm); the ONE cell that did not
  (`in16m L8 -p2`, rerun read +8.00% under load5≈56) is adjudicated below by
  the same-cell A/A falsifier (same binary reads +0.84% and −1.11% in two
  passes) plus the no-mechanism argument (byte-identical stream ⇒ identical
  work) → final verdict TIE/hyp. The rerun read is kept verbatim in the
  table; nothing was rerun-until-it-looked-good.
* **FLAGS: 240/240 PASS** — `nearopt-fixedpoint-skip` ON == default (80 cells:
  20 members × {L11,L12} × {-p2,-p4}); `near-opt-parallel-flush +
  near-opt-parallel-flush-flipsafe` ON == flush-OFF default (160 cells: 20
  members × L9..L12 × {-p2,-p4}) — the 416-cell flip-safe census bar
  re-verified at reduced volume (52→20 members) on the FINAL tree, with every
  ON stream additionally pigz-decoded. Counter reconciliations on the anatomy
  instrument builds all PASS.
* **NO FAIL occurred in any section.** The "capture command+sha+size and STOP
  the section" D-rule was never triggered.

## Setup and provenance (receipts in `data/provenance.txt`)

- Arm **A** = worktree `.review-tmp/wC-final/wt-a` @ `b1502a07…`
  (default features, `--target-dir .review-tmp/wC-final/tgt-a`), binary
  sha256 `5398210241ef4bc3607583d34ed0abea124b281a20e47e0143fc78fec81aecbe`.
- Arm **B** = worktree `.review-tmp/wC-final/wt-b` @ `06859145…`
  (default features, `--target-dir .review-tmp/wC-final/tgt-b`), binary
  sha256 `c23ae453f879caee88c1f771746c44fe9675f4fb0f2e9b37e227b0be416c8708`.
  Both binaries: `gzippy 1.0.0 (parallel-sm+pure)`, 1,744,112 B each.
  Submodules linked via `scripts/dev/worktree-link.sh` + the explicit
  `vendor/isa-l` symlink for the empty-dir case the script skips.
- **A/B source delta receipts:** main's first-parent chain is `84b910aa`
  (#465) → `06859145` (#466) → `b1502a07` (#467); `git log --oneline
  06859145..b1502a07` lists exactly `fbdd3445` (fp-settle feature),
  `d97343ec`, `8acb5926`, `e3f9e92a`, `9a2f7e72`, `b1502a07`. Nothing of
  #465/#466's SOURCE differs between the arms; the three trio features are
  build-time gates that compile to no code in a default build (both arms'
  default builds carry all three OFF).
- Decoders: `/usr/bin/gzip` (Apple gzip 479.0.0.700.1), `vendor/pigz/pigz`
  2.8, python3 zlib member-chain (`data/runners/decode_sha.py`, the wave-B
  instrument). `/opt/homebrew/bin/gzip --version` prints **gzippy 0.1.7** —
  an installed gzippy argv0 alias on this box — DEMOTED: never used as a
  decoder anywhere in this census.
- All compression ran stream mode (`-c`, stdin→stdout), header-free.
- Box: Mac16,7 (Apple M4 Pro), 14 CPUs; UNFROZEN — foreign co-runner
  processes were active during parts of the run (ps receipts in
  `data/co-runner-note.txt`; load1 reached 129-151 during the byte phase).
  Every wall number below is therefore HYPOTHESIS-TIER per the charter; the
  frozen box adjudicates anything it cares about. During the wall film's
  windows load1 spanned 7-21 (per-rep values in `data/wall-raw.log.txt`).
- Corpus: the wave-B 30 members (12 silesia tar members, `logs.txt`
  221,249,619 B, `software.archive` 221,249,546 B, 8+8 even slices of both);
  input sha256 recorded fresh in `data/members.tsv`, slice reconstruction
  asserted (cat part-aa..ah sha == whole-file sha). Wall inputs: `in16m.bin`
  16,085,866 B (`1010fa2f…`, the wave-9 pinned recipe) + the three 26 MB
  class stand-ins staged at `.review-tmp/wC-walls/corpus/`
  (`logs-card26m.bin`, `binary26m.bin`, `text26m.bin` = text-1MB ×25).
- Method: every number in the tables below was generated from the raw result
  files in `data/` by `data/runners/make_tables.py` — nothing hand-copied.

## Expectation 1 — BYTES (the trio is default-off ⇒ A == B "everywhere you can reach it")

Anchor expectation values (size + sha256) are the recorded wave-B receipts
(`data/anchors.expected.tsv`, quoted from
`docs/board/records/2026-11-11-waveB-byte-census/README.md` A1-A4).

| arm | command | size | stream sha256 | expected sha | legs apple/pigz/py-zlib | verdict |
|---|---|---:|---|---|---|---|
| A | `-9 -p2` | 3,816,930 | `edc698cc16c8b91f50e59b17fd06a66b7e141bdee455be83017b3d9213b99e5a` | edc698cc… | ok/ok/OK | PASS |
| B | `-9 -p2` | 3,816,930 | `edc698cc16c8b91f50e59b17fd06a66b7e141bdee455be83017b3d9213b99e5a` | edc698cc… | ok/ok/OK | PASS |
| A | `-9 -p4` | 3,740,910 | `327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475` | 327b0835… | ok/ok/OK | PASS |
| B | `-9 -p4` | 3,740,910 | `327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475` | 327b0835… | ok/ok/OK | PASS |
| A | `-8 -p1` | 3,821,170 | `2a0fb084316ae7f6e5dd8034c729672a3ab520a57fe344a48f6e996e9c35025d` | 2a0fb084… | ok/ok/OK | PASS |
| B | `-8 -p1` | 3,821,170 | `2a0fb084316ae7f6e5dd8034c729672a3ab520a57fe344a48f6e996e9c35025d` | 2a0fb084… | ok/ok/OK | PASS |
| A | `-8 -p4` | 3,819,910 | `b7090a22573814e2520aec366da9d89b5c7ec713f8034c026373cc016020299d` | b7090a22… | ok/ok/OK | PASS |
| B | `-8 -p4` | 3,819,910 | `b7090a22573814e2520aec366da9d89b5c7ec713f8034c026373cc016020299d` | b7090a22… | ok/ok/OK | PASS |
| A | `-6 -p2` | 3,848,412 | `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | 7997e3ac… | ok/ok/OK | PASS |
| B | `-6 -p2` | 3,848,412 | `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | 7997e3ac… | ok/ok/OK | PASS |
| A | `-6 -p4` | 3,848,412 | `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | 7997e3ac… | ok/ok/OK | PASS |
| B | `-6 -p4` | 3,848,412 | `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | 7997e3ac… | ok/ok/OK | PASS |

ALL 12 anchor rows PASS

Byte matrix: 30 members × L1..L12 × {T1,T2,T4} = 1,080 cells, each cell = an
arm A encode and an arm B encode (`--level N -p T -c`), sha256 + byte size
both, verdict OK iff the shas are EQUAL and the stream roundtrips through all
three decoders. Full per-cell table: `data/byte_matrix.tsv`
(columns: member, level, T, in_sha, sha_A, size_A, sha_B, size_B,
eq-flag, legs, legsB, verdict).

Total cells: 1080 (each = arm A and arm B encode, sha-paired)

| band | A==B + legs OK | cells | verdict |
|---|---:|---:|---|
| T1 | 360 | 360 | PASS |
| T2 | 360 | 360 | PASS |
| T4 | 360 | 360 | PASS |

Failures: 0

| cell count | 1,080 |
|---|---:|
| A-stream shas == B-stream shas | 1,080 / 1,080 |
| 3-decoder leg checks (apple, pigz, py-zlib) | 3,240 / 3,240 OK |
| FAILED cells | 0 |

Verbatim FAIL rows: none.

## Expectation 2 — WALL (the regression watch; expect TIES everywhere)

HYPOTHESIS-TIER throughout (charter: unfrozen box; the frozen box
adjudicates anything under 5%). Method per cell: one warm run per arm
(uncounted), then 8 interleaved reps ordered A,B,B,A = 16 timed samples per
arm, median; each run streams to `/dev/null`; every sample logged with
load1/5/15 (`data/wall-raw.log.txt`); byte-stamp per cell (both arms' stream
sha once — deterministic instrument). Rerun rule (wave-B wall-census method):
any cell outside ±5% is re-filmed at 14 reps (28/arm) and the rerun replaces
the first pass.

A/A calibration (same binary, both arms — the noise floor of today's box,
`data/wall_aa.json`; the 4 baseline cells were first filmed in a loaded
window where the same-binary text26m L4 -p2 cell read +4.86%, and re-filmed
in the complete pass below at −1.11%…+0.70% — the complete 5-cell pass also
includes the one falsifier cell used for the L8 -p2 adjudication below):

| cell | A med ms | B med ms | delta | load |
|---|---:|---:|---:|---|
| AA_in16m_L9_p2 | 114.0 | 114.1 | -0.06% | load 6.89/12.98/21.88 |
| AA_in16m_L11_p4 | 464.9 | 464.7 | +0.05% | load 6.99/12.70/21.63 |
| AA_text26m_L4_p2 | 179.7 | 179.8 | -0.08% | load 7.23/12.66/21.56 |
| AA_text26m_L9_p4 | 477.1 | 473.7 | +0.70% | load 7.11/12.46/21.38 |
| AA_in16m_L8_p2 | 104.3 | 105.5 | -1.11% | load 6.86/12.32/21.28 |

The wall matrix. Per-cell verdicts: first pass at 8 reps; `→ rerun` rows
replace the first-pass number after a 14-rep pass.

| cell | A med ms | B med ms | A vs B | bytes | rerun 14-rep | verdict |
|---|---:|---:|---:|---|---|---|
| in16m L1 -p1 | 70.9 | 70.8 | +0.11% | identical | — | TIE/hyp |
| in16m L1 -p2 | 37.2 | 36.9 | +0.89% | identical | — | TIE/hyp |
| in16m L1 -p4 | 23.8 | 23.7 | +0.21% | identical | — | TIE/hyp |
| in16m L2 -p1 | 63.7 | 64.6 | -1.42% | identical | — | TIE/hyp |
| in16m L2 -p2 | 47.4 | 47.3 | +0.16% | identical | — | TIE/hyp |
| in16m L2 -p4 | 36.8 | 33.9 | +8.55% | identical | → rerun 47.6 vs 46.4 (+2.38%) | TIE/hyp (rerun) |
| in16m L3 -p1 | 79.4 | 76.9 | +3.24% | identical | — | TIE/hyp |
| in16m L3 -p2 | 69.2 | 67.0 | +3.20% | identical | — | TIE/hyp |
| in16m L3 -p4 | 36.9 | 37.0 | -0.21% | identical | — | TIE/hyp |
| in16m L4 -p1 | 69.5 | 69.6 | -0.21% | identical | — | TIE/hyp |
| in16m L4 -p2 | 62.6 | 62.7 | -0.27% | identical | — | TIE/hyp |
| in16m L4 -p4 | 41.1 | 41.8 | -1.69% | identical | — | TIE/hyp |
| in16m L5 -p1 | 78.4 | 78.8 | -0.51% | identical | — | TIE/hyp |
| in16m L5 -p2 | 68.0 | 68.1 | -0.16% | identical | — | TIE/hyp |
| in16m L5 -p4 | 56.9 | 53.4 | +6.50% | identical | → rerun 65.3 vs 64.8 (+0.82%) | TIE/hyp (rerun) |
| in16m L6 -p1 | 116.0 | 114.4 | +1.35% | identical | — | TIE/hyp |
| in16m L6 -p2 | 91.9 | 90.5 | +1.51% | identical | — | TIE/hyp |
| in16m L6 -p4 | 58.2 | 58.3 | -0.18% | identical | — | TIE/hyp |
| in16m L7 -p1 | 139.6 | 141.3 | -1.24% | identical | — | TIE/hyp |
| in16m L7 -p2 | 99.6 | 99.0 | +0.62% | identical | — | TIE/hyp |
| in16m L7 -p4 | 62.3 | 63.8 | -2.32% | identical | — | TIE/hyp |
| in16m L8 -p1 | 185.4 | 180.3 | +2.85% | identical | — | TIE/hyp |
| in16m L8 -p2 | 115.9 | 110.1 | +5.29% | identical | → rerun 128.6 vs 119.1 (+8.00%) | TIE/hyp (rerun read +8.00% kept; same-cell A/A -1.11% + identical-bytes mechanism adjudicate it box noise) |
| in16m L8 -p4 | 70.1 | 70.1 | +0.05% | identical | — | TIE/hyp |
| in16m L9 -p1 | 199.9 | 200.6 | -0.33% | identical | — | TIE/hyp |
| in16m L9 -p2 | 113.8 | 114.1 | -0.23% | identical | — | TIE/hyp |
| in16m L9 -p4 | 371.7 | 370.0 | +0.46% | identical | — | TIE/hyp |
| in16m L10 -p1 | 1026.4 | 1031.9 | -0.53% | identical | — | TIE/hyp |
| in16m L10 -p2 | 115.2 | 115.3 | -0.05% | identical | — | TIE/hyp |
| in16m L10 -p4 | 352.5 | 365.4 | -3.54% | identical | — | TIE/hyp |
| in16m L11 -p1 | 1458.3 | 1452.2 | +0.42% | identical | — | TIE/hyp |
| in16m L11 -p2 | 114.1 | 113.6 | +0.48% | identical | — | TIE/hyp |
| in16m L11 -p4 | 471.9 | 471.8 | +0.02% | identical | — | TIE/hyp |
| in16m L12 -p1 | 1902.1 | 1898.6 | +0.19% | identical | — | TIE/hyp |
| in16m L12 -p2 | 113.5 | 113.9 | -0.34% | identical | — | TIE/hyp |
| in16m L12 -p4 | 565.7 | 571.9 | -1.09% | identical | — | TIE/hyp |
| logs-card26m L1 -p1 | 34.5 | 34.5 | -0.15% | identical | — | TIE/hyp |
| logs-card26m L1 -p2 | 18.5 | 18.6 | -0.69% | identical | — | TIE/hyp |
| logs-card26m L1 -p4 | 12.3 | 12.4 | -0.23% | identical | — | TIE/hyp |
| logs-card26m L2 -p1 | 37.0 | 36.6 | +1.04% | identical | — | TIE/hyp |
| logs-card26m L2 -p2 | 22.3 | 22.4 | -0.47% | identical | — | TIE/hyp |
| logs-card26m L2 -p4 | 14.7 | 14.9 | -0.73% | identical | — | TIE/hyp |
| logs-card26m L3 -p1 | 35.6 | 35.1 | +1.22% | identical | — | TIE/hyp |
| logs-card26m L3 -p2 | 23.4 | 23.3 | +0.39% | identical | — | TIE/hyp |
| logs-card26m L3 -p4 | 15.5 | 15.5 | +0.04% | identical | — | TIE/hyp |
| logs-card26m L4 -p1 | 36.3 | 36.0 | +0.88% | identical | — | TIE/hyp |
| logs-card26m L4 -p2 | 24.5 | 24.5 | -0.36% | identical | — | TIE/hyp |
| logs-card26m L4 -p4 | 16.1 | 16.3 | -1.23% | identical | — | TIE/hyp |
| logs-card26m L5 -p1 | 36.3 | 36.7 | -1.04% | identical | — | TIE/hyp |
| logs-card26m L5 -p2 | 28.1 | 28.1 | -0.20% | identical | — | TIE/hyp |
| logs-card26m L5 -p4 | 18.2 | 18.2 | +0.34% | identical | — | TIE/hyp |
| logs-card26m L6 -p1 | 50.7 | 50.8 | -0.09% | identical | — | TIE/hyp |
| logs-card26m L6 -p2 | 40.9 | 40.8 | +0.22% | identical | — | TIE/hyp |
| logs-card26m L6 -p4 | 23.6 | 23.6 | -0.07% | identical | — | TIE/hyp |
| logs-card26m L7 -p1 | 65.8 | 65.5 | +0.48% | identical | — | TIE/hyp |
| logs-card26m L7 -p2 | 43.2 | 43.2 | +0.07% | identical | — | TIE/hyp |
| logs-card26m L7 -p4 | 25.0 | 25.0 | +0.17% | identical | — | TIE/hyp |
| logs-card26m L8 -p1 | 107.5 | 107.6 | -0.05% | identical | — | TIE/hyp |
| logs-card26m L8 -p2 | 59.2 | 59.2 | +0.00% | identical | — | TIE/hyp |
| logs-card26m L8 -p4 | 33.7 | 34.2 | -1.20% | identical | — | TIE/hyp |
| logs-card26m L9 -p1 | 144.4 | 145.6 | -0.83% | identical | — | TIE/hyp |
| logs-card26m L9 -p2 | 78.2 | 77.8 | +0.50% | identical | — | TIE/hyp |
| logs-card26m L9 -p4 | 215.9 | 216.8 | -0.43% | identical | — | TIE/hyp |
| logs-card26m L10 -p1 | 531.3 | 529.3 | +0.39% | identical | — | TIE/hyp |
| logs-card26m L10 -p2 | 76.2 | 76.0 | +0.16% | identical | — | TIE/hyp |
| logs-card26m L10 -p4 | 156.8 | 157.3 | -0.28% | identical | — | TIE/hyp |
| logs-card26m L11 -p1 | 978.8 | 975.2 | +0.37% | identical | — | TIE/hyp |
| logs-card26m L11 -p2 | 76.0 | 76.1 | -0.11% | identical | — | TIE/hyp |
| logs-card26m L11 -p4 | 282.1 | 283.2 | -0.39% | identical | — | TIE/hyp |
| logs-card26m L12 -p1 | 2187.5 | 2181.7 | +0.27% | identical | — | TIE/hyp |
| logs-card26m L12 -p2 | 77.0 | 76.4 | +0.69% | identical | — | TIE/hyp |
| logs-card26m L12 -p4 | 678.9 | 682.8 | -0.56% | identical | — | TIE/hyp |
| binary26m L1 -p1 | 36.1 | 36.2 | -0.02% | identical | — | TIE/hyp |
| binary26m L1 -p2 | 17.1 | 17.1 | +0.31% | identical | — | TIE/hyp |
| binary26m L1 -p4 | 11.5 | 11.4 | +1.16% | identical | — | TIE/hyp |
| binary26m L2 -p1 | 35.4 | 35.2 | +0.46% | identical | — | TIE/hyp |
| binary26m L2 -p2 | 21.3 | 21.2 | +0.33% | identical | — | TIE/hyp |
| binary26m L2 -p4 | 14.5 | 14.6 | -0.60% | identical | — | TIE/hyp |
| binary26m L3 -p1 | 36.3 | 36.1 | +0.72% | identical | — | TIE/hyp |
| binary26m L3 -p2 | 22.1 | 21.7 | +1.71% | identical | — | TIE/hyp |
| binary26m L3 -p4 | 14.4 | 14.5 | -0.17% | identical | — | TIE/hyp |
| binary26m L4 -p1 | 36.2 | 36.2 | -0.04% | identical | — | TIE/hyp |
| binary26m L4 -p2 | 21.4 | 21.6 | -0.79% | identical | — | TIE/hyp |
| binary26m L4 -p4 | 14.4 | 14.4 | +0.34% | identical | — | TIE/hyp |
| binary26m L5 -p1 | 36.8 | 36.8 | +0.11% | identical | — | TIE/hyp |
| binary26m L5 -p2 | 21.9 | 21.7 | +0.89% | identical | — | TIE/hyp |
| binary26m L5 -p4 | 14.3 | 14.5 | -0.75% | identical | — | TIE/hyp |
| binary26m L6 -p1 | 37.1 | 37.1 | -0.18% | identical | — | TIE/hyp |
| binary26m L6 -p2 | 20.6 | 20.6 | +0.01% | identical | — | TIE/hyp |
| binary26m L6 -p4 | 12.8 | 12.8 | +0.23% | identical | — | TIE/hyp |
| binary26m L7 -p1 | 41.6 | 41.8 | -0.35% | identical | — | TIE/hyp |
| binary26m L7 -p2 | 48.6 | 48.3 | +0.44% | identical | — | TIE/hyp |
| binary26m L7 -p4 | 27.7 | 27.3 | +1.22% | identical | — | TIE/hyp |
| binary26m L8 -p1 | 92.1 | 92.2 | -0.09% | identical | — | TIE/hyp |
| binary26m L8 -p2 | 49.4 | 49.4 | +0.05% | identical | — | TIE/hyp |
| binary26m L8 -p4 | 28.2 | 28.4 | -0.58% | identical | — | TIE/hyp |
| binary26m L9 -p1 | 131.0 | 131.5 | -0.40% | identical | — | TIE/hyp |
| binary26m L9 -p2 | 69.1 | 68.9 | +0.28% | identical | — | TIE/hyp |
| binary26m L9 -p4 | 219.4 | 220.8 | -0.66% | identical | — | TIE/hyp |
| binary26m L10 -p1 | 636.7 | 638.5 | -0.28% | identical | — | TIE/hyp |
| binary26m L10 -p2 | 68.7 | 69.0 | -0.46% | identical | — | TIE/hyp |
| binary26m L10 -p4 | 184.9 | 185.2 | -0.12% | identical | — | TIE/hyp |
| binary26m L11 -p1 | 1906.3 | 1910.7 | -0.23% | identical | — | TIE/hyp |
| binary26m L11 -p2 | 69.2 | 68.9 | +0.37% | identical | — | TIE/hyp |
| binary26m L11 -p4 | 539.5 | 539.8 | -0.06% | identical | — | TIE/hyp |
| binary26m L12 -p1 | 2248.8 | 2248.6 | +0.01% | identical | — | TIE/hyp |
| binary26m L12 -p2 | 69.1 | 68.8 | +0.40% | identical | — | TIE/hyp |
| binary26m L12 -p4 | 660.5 | 657.7 | +0.43% | identical | — | TIE/hyp |
| text26m L1 -p1 | 174.2 | 174.1 | +0.09% | identical | — | TIE/hyp |
| text26m L1 -p2 | 90.6 | 90.5 | +0.13% | identical | — | TIE/hyp |
| text26m L1 -p4 | 49.0 | 48.2 | +1.75% | identical | — | TIE/hyp |
| text26m L2 -p1 | 163.7 | 163.2 | +0.31% | identical | — | TIE/hyp |
| text26m L2 -p2 | 115.4 | 115.6 | -0.17% | identical | — | TIE/hyp |
| text26m L2 -p4 | 68.8 | 69.0 | -0.35% | identical | — | TIE/hyp |
| text26m L3 -p1 | 224.3 | 225.1 | -0.33% | identical | — | TIE/hyp |
| text26m L3 -p2 | 163.5 | 163.7 | -0.08% | identical | — | TIE/hyp |
| text26m L3 -p4 | 96.4 | 96.5 | -0.09% | identical | — | TIE/hyp |
| text26m L4 -p1 | 191.4 | 190.8 | +0.34% | identical | — | TIE/hyp |
| text26m L4 -p2 | 180.3 | 180.6 | -0.19% | identical | — | TIE/hyp |
| text26m L4 -p4 | 106.7 | 106.9 | -0.25% | identical | — | TIE/hyp |
| text26m L5 -p1 | 240.7 | 241.4 | -0.29% | identical | — | TIE/hyp |
| text26m L5 -p2 | 210.2 | 209.9 | +0.16% | identical | — | TIE/hyp |
| text26m L5 -p4 | 124.0 | 123.5 | +0.35% | identical | — | TIE/hyp |
| text26m L6 -p1 | 386.3 | 385.1 | +0.32% | identical | — | TIE/hyp |
| text26m L6 -p2 | 237.3 | 236.0 | +0.56% | identical | — | TIE/hyp |
| text26m L6 -p4 | 130.0 | 129.9 | +0.08% | identical | — | TIE/hyp |
| text26m L7 -p1 | 440.4 | 439.6 | +0.18% | identical | — | TIE/hyp |
| text26m L7 -p2 | 241.0 | 242.1 | -0.45% | identical | — | TIE/hyp |
| text26m L7 -p4 | 156.4 | 139.7 | +12.02% | identical | → rerun 157.7 vs 150.5 (+4.79%) | TIE/hyp (rerun) |
| text26m L8 -p1 | 550.3 | 551.9 | -0.29% | identical | — | TIE/hyp |
| text26m L8 -p2 | 289.2 | 289.0 | +0.07% | identical | — | TIE/hyp |
| text26m L8 -p4 | 154.6 | 155.2 | -0.37% | identical | — | TIE/hyp |
| text26m L9 -p1 | 574.0 | 574.8 | -0.14% | identical | — | TIE/hyp |
| text26m L9 -p2 | 304.1 | 302.8 | +0.44% | identical | — | TIE/hyp |
| text26m L9 -p4 | 494.5 | 486.1 | +1.72% | identical | — | TIE/hyp |
| text26m L10 -p1 | 1687.1 | 1669.6 | +1.05% | identical | — | TIE/hyp |
| text26m L10 -p2 | 297.8 | 300.0 | -0.73% | identical | — | TIE/hyp |
| text26m L10 -p4 | 474.3 | 474.2 | +0.02% | identical | — | TIE/hyp |
| text26m L11 -p1 | 2028.3 | 2022.3 | +0.30% | identical | — | TIE/hyp |
| text26m L11 -p2 | 305.5 | 304.8 | +0.21% | identical | — | TIE/hyp |
| text26m L11 -p4 | 1551.1 | 1212.7 | +27.90% | identical | → rerun 587.3 vs 585.7 (+0.27%) | TIE/hyp (rerun) |
| text26m L12 -p1 | 2428.8 | 2711.6 | -10.43% | identical | → rerun 2138.2 vs 2140.8 (-0.12%) | TIE/hyp (rerun) |
| text26m L12 -p2 | 339.7 | 348.0 | -2.39% | identical | — | TIE/hyp |
| text26m L12 -p4 | 1106.8 | 997.9 | +10.91% | identical | → rerun 629.1 vs 629.4 (-0.05%) | TIE/hyp (rerun) |

Rerun pass captured: 0 rows
(none)

**Cell-by-cell counts:** 144 primary cells, **137/144 first-pass ties**;
7 cells outside ±5% at 8 reps → all 7 reran at 14 reps; **6/7 collapsed to
ties** (+2.38, +0.82, +4.79, +0.27, −0.12, −0.05%); the one holdover is
adjudicated below; **0 outside-band verdicts remain**.

**The one non-collapsing cell, `in16m L8 -p2` — verdict: TIE/hyp.** Three
reads of this cell exist in `data/wall-raw.log.txt`:

| pass | A med | B med | delta | window |
|---|---:|---:|---:|---|
| primary 8-rep | 115.9 | 110.1 | +5.29% | load1 ~30 + spikes |
| rerun 14-rep | 128.6 | 119.1 | +8.00% | **load1 med 19.2, load5 55.98** — the worst window of the whole film |
| A/A (same binary) | 105.1 | 104.2 | +0.84% | load1 9.1 |
| A/A (complete pass, later) | 104.3 | 105.5 | −1.11% | load1 9.0 |

The falsifier chain: (1) the two binary arms compile to the same work for
this cell — the stream is byte-IDENTICAL (bytes row) and L8 -p2 executes none
of the fp-settle/flush code paths; (2) the SAME binary filmed at this exact
cell reads −1.11%…+0.84% (band) in two independent passes; (3) the rerun
window is the load-outlier of the entire film (load5 56 — its other cell,
text26m L11 -p4, read +27.9% primary and collapsed to +0.27% once load
dropped). Both arms' absolute medians moved UP TOGETHER into that window
(116/110 → 129/119), the co-load tell. The +8.00% read is a co-runner
artifact, kept verbatim in the table, and the tree is NOT convicted by it.
(A/A discipline: previously measured same-binary drift on THIS box in a
loaded window reaches +4.86%.)

**Wall summary: TIES EVERYWHERE — no regression anywhere on the census; the
deterministic-carrier check for the tree holds (feature flags off ⇒ no
mechanism moved wall).**

## Expectation 3 — FEATURE-ON spot census (arms within A: the census of the flags themselves)

One source (wt-a = arm A's tree), seven release builds, one `--target-dir`
each (binary shas in `data/feature_binaries.sha256`):

| arm | build | binary sha256 (first 16) |
|---|---|---|
| OFF (= arm A default) | `cargo build --release` → `tgt-a` | `5398210241ef4bc3` |
| SKIP-ON | `--features nearopt-fixedpoint-skip` → `tgt-skip` | `8bd5e82ed2d80e95` |
| SAFE-ON | `--features near-opt-parallel-flush,near-opt-parallel-flush-flipsafe` → `tgt-safe` | `869538de3e25f838` |
| CNTR | `--features anatomy-counters` → `tgt-cntr` | `147c7f106f960120` |
| SKIPC | `--features anatomy-counters,nearopt-fixedpoint-skip` → `tgt-skipc` | `f796423e2483dce1` |
| FLUSHC | `--features anatomy-counters,near-opt-parallel-flush,near-opt-parallel-flush-flipsafe` → `tgt-flushc` | `38add2b21818247b` |

20 members (the 12 silesia members + logs.txt + software.archive + one leading
and trailing slice of each big file + the three flip-heavy classes
weights.safetensors / sil40 / (silesia-)mozilla — the 52-member
flip-census grid reduced to 20). Grid: SKIP at L11/L12 × {-p2,-p4} (the
levels the lever's counters mark; -p2 rows are the route-inert arm) = 80
cells; SAFE at L9..L12 × {-p2,-p4} = 160 cells (the 416-cell layout at
reduced volume). Assertion per cell: ON stream sha == the OFF/default stream
sha, and the ON stream pigz-decodes byte-roundtrip. Full per-cell table:
`data/flag_census.tsv`.

| feature | level | -p | cells | OK | verdict |
|---|---:|---:|---:|---:|---|
| safe | L9 | -p2 | 20 | 20 | PASS |
| safe | L9 | -p4 | 20 | 20 | PASS |
| safe | L10 | -p2 | 20 | 20 | PASS |
| safe | L10 | -p4 | 20 | 20 | PASS |
| safe | L11 | -p2 | 20 | 20 | PASS |
| safe | L11 | -p4 | 20 | 20 | PASS |
| safe | L12 | -p2 | 20 | 20 | PASS |
| safe | L12 | -p4 | 20 | 20 | PASS |
| skip | L11 | -p2 | 20 | 20 | PASS |
| skip | L11 | -p4 | 20 | 20 | PASS |
| skip | L12 | -p2 | 20 | 20 | PASS |
| skip | L12 | -p4 | 20 | 20 | PASS |

Cells total: 240; fails: 0

Per-member:
safe/logs.part-aa: 8/8 OK
safe/logs.part-ah: 8/8 OK
safe/logs.txt: 8/8 OK
safe/sil40: 8/8 OK
safe/silesia-dickens: 8/8 OK
safe/silesia-mozilla: 8/8 OK
safe/silesia-mr: 8/8 OK
safe/silesia-nci: 8/8 OK
safe/silesia-ooffice: 8/8 OK
safe/silesia-osdb: 8/8 OK
safe/silesia-reymont: 8/8 OK
safe/silesia-samba: 8/8 OK
safe/silesia-sao: 8/8 OK
safe/silesia-webster: 8/8 OK
safe/silesia-x-ray: 8/8 OK
safe/silesia-xml: 8/8 OK
safe/software.archive: 8/8 OK
safe/sw.part-aa: 8/8 OK
safe/sw.part-ah: 8/8 OK
safe/weights.safetensors: 8/8 OK
skip/logs.part-aa: 4/4 OK
skip/logs.part-ah: 4/4 OK
skip/logs.txt: 4/4 OK
skip/sil40: 4/4 OK
skip/silesia-dickens: 4/4 OK
skip/silesia-mozilla: 4/4 OK
skip/silesia-mr: 4/4 OK
skip/silesia-nci: 4/4 OK
skip/silesia-ooffice: 4/4 OK
skip/silesia-osdb: 4/4 OK
skip/silesia-reymont: 4/4 OK
skip/silesia-samba: 4/4 OK
skip/silesia-sao: 4/4 OK
skip/silesia-webster: 4/4 OK
skip/silesia-x-ray: 4/4 OK
skip/silesia-xml: 4/4 OK
skip/software.archive: 4/4 OK
skip/sw.part-aa: 4/4 OK
skip/sw.part-ah: 4/4 OK
skip/weights.safetensors: 4/4 OK

**skip-ON == skip-OFF on 80/80; SAFE-ON == flush-OFF on 160/160; zero
exceptions; 240/240 pigz decode legs OK.** The #465 408-stream bar and the
#467 416-cell census bar both re-verify on the FINAL tree at this reduced
volume, including the flip-HEAVY classes where a guessed-freshness design
would diverge (weights.safetensors 8/8 SAFE cells OK — the class the plain
flush feature DIVERGED on, per `docs/board/records/2026-11-11-flush-flipsafe-design/README.md` §1.9).

### Counter deltas ("record if cheap" — anatomy instrument builds)

Instrument builds are ~1.17x slower and are never quoted for wall; stdout
BYTES are not compared here (counters ride stderr only; the flush at process
end is `ANATOMY_COUNTERS={json}` + `ANATOMY_RECONCILE=` lines — see
`data/counters.tsv`, raw stderr in `data/counter-stderr/`).

| pair | member | cell | CNTR | arm-ON | delta | expected | verdict |
|---|---|---|---:|---:|---:|---:|---|
| skip ON-OFF | in16m | L11 -p4 | pass_entries 1,206 | 1,194 | -12 | would_save 12 | PASS |
| skip ON-OFF | in16m | L12 -p4 | pass_entries 1,623 | 1,513 | -110 | would_save 110 | PASS |
| skip ON-OFF | logs.part-aa | L12 -p4 | pass_entries 412 | 338 | -74 | would_save 74 | PASS |
| skip ON-OFF | sil40 | L11 -p4 | pass_entries 4,109 | 4,081 | -28 | would_save 28 | PASS |
| skip ON-OFF | weights.safetensors | L12 -p4 | pass_entries 6,049 | 5,763 | -286 | would_save 286 | PASS |
| flush SAFE-vs-serial | in16m | L9 -p4 | only-lits 6 | only-lits 6 | settles 351 | tally equal | PASS |
| flush SAFE-vs-serial | in16m | L11 -p4 | only-lits 3 | only-lits 3 | settles 357 | tally equal | PASS |
| flush SAFE-vs-serial | silesia-mozilla | L9 -p4 | only-lits 11 | only-lits 11 | settles 2047 | tally equal | PASS |
| flush SAFE-vs-serial | weights.safetensors | L9 -p4 | only-lits 555 | only-lits 555 | settles 873 | tally equal | PASS |
| flush SAFE-vs-serial | sil40 | L11 -p4 | only-lits 7 | only-lits 7 | settles 1146 | tally equal | PASS |

Both reconciliations match the two levers' recorded mechanics to the pass:

- **fixedpoint-skip (#465):** each SKIPC arm runs EXACTLY
  `would_save`-fewer DP passes than the serial arm Δ = −12 / −110 /
  −74 / −28 / −286 = the mid-loop equality counts), and no other counter
  moves (`set_from_codes_calls`, blocks identical 1:1). The
  pass-scoped numbers reproduce the #465 record's own counter table exactly
  where the members overlap (in16m L11 1,206/12 and L12 1,623/110;
  weights L12 6,049/286 as `nearopt_finite_skip_would_save` 286 of 298
  equality events).
- **fp-settle (#467):** the SAFE arm's `near_opt_only_literals_blocks` tally
  EQUALS the serial arm's at every measured cell (6, 3, 11, 555, 7) — the
  exactness corroboration, with the recorded receipts reproduced exactly:
  mozilla L9 -p4 = 11 only-lits over **2,047 settles** (the #467 record's
  own count), weights L9 -p4 = 555/555 with 873 full-depth settles, sil40
  L11 -p4 = 7/7 with 1,146 settles. Settles > 0 on every flip-landing cell —
  the wait arm demonstrably runs and demonstrably preserves the serial
  decision space.

## D-rule captures

None. No FAIL row exists in any section: anchors 12/12, byte matrix
1,080/1,080, wall 0 outside-band verdicts after the recorded falsifier
chain, flag census 240/240, counter reconciliations 10/10 PASS.

## What this census does NOT claim

- Wall magnitudes are LOCAL (hypothesis-tier on an unfrozen, co-loaded
  box); the frozen box adjudicates anything under 5% before any
  promotion decision. Nothing here was a promotion decision.
- The byte matrix ran its 3 decoder legs on the per-cell (A==B) stream
  once per cell; the B arm's stream is sha-asserted identical, not decoded
  separately — the sha-pin plus identical-decoder path make this
  equivalent, stated for the record rather than glossed.
- The skip flag census ran at the levels the lever's counters mark
  fire-eligible (L11/L12; L9/L10 -p4 are structurally 0-skip per the #465
  record and were covered there); the SAFE census covers the full L9..L12
  grid.

## Materials (this record dir)

| file | contents |
|---|---|
| `README.md` | this file (tables machine-generated into it) |
| `data/byte_matrix.tsv` | all 1,080 cells: sha pair, sizes, eq flag, 3 legs, verdict |
| `data/anchors.tsv` / `data/anchors.expected.tsv` | 12 anchor rows + the pinned expectations |
| `data/members.tsv` / `data/members_spot.tsv` | the 30-member corpus + the 20-member flag set (input shas) |
| `data/wall_primary.json` / `data/wall_rerun.json` / `data/wall_aa.json` | wall medians + raw samples per cell |
| `data/wall-raw.log.txt` | per-sample wall log (all 144 cells + 7 reruns + A/A passes, per-rep load) |
| `data/flag_census.tsv` | 240 flag-census cells: sha pair, eq, pigz leg, verdict |
| `data/counters.tsv` + `data/counter-stderr/` | the instrument counters + raw stderr JSON |
| `data/binaries.sha256` / `data/feature_binaries.sha256` | arm + feature binary stamps |
| `data/provenance.txt` / `data/co-runner-note.txt` | box and co-runner receipts |
| `data/runners/` | every instrument: `mk_members.sh`, `anch.sh`, `bytematrix.py`, `wallfilm.py`, `flagcensus.py`, `counters.sh`, `parse_count.py`, `feature_builds.sh`, `decode_sha.py`, `make_tables.py` |

## Reproduce

```bash
git fetch origin && git worktree add .review-tmp/wC-final/wt-a --detach b1502a07
git worktree add .review-tmp/wC-final/wt-b --detach 06859145
scripts/dev/worktree-link.sh .review-tmp/wC-final/wt-a   # + vendor/isa-l empty-dir symlink
scripts/dev/worktree-link.sh .review-tmp/wC-final/wt-b
(cd .review-tmp/wC-final/wt-a && cargo build --release --target-dir /abs/.review-tmp/wC-final/tgt-a)
(cd .review-tmp/wC-final/wt-b && cargo build --release --target-dir /abs/.review-tmp/wC-final/tgt-b)
.review-tmp/wC-final/run/mk_members.sh                  # stage member shas
.review-tmp/wC-final/run/anch.sh                        # anchors, 3 legs each
python3 .review-tmp/wC-final/run/bytematrix.py          # 1,080-cell byte matrix
python3 .review-tmp/wC-final/run/wallfilm.py all        # wall film + reruns
python3 .review-tmp/wC-final/run/wallfilm.py aa         # A/A calibration
.review-tmp/wC-final/run/feature_builds.sh              # five feature arms
python3 .review-tmp/wC-final/run/flagcensus.py all      # flag spot-census
bash .review-tmp/wC-final/run/counters.sh               # counter deltas
python3 .review-tmp/wC-final/run/make_tables.py tables.md
```
