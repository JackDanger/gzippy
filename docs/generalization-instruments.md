# Generalization instruments — "have we overfit to the 22-file board?"

Every size number this campaign has ever quoted comes from one 22-file corpus.
`corpus_split.json` splits it TUNE/GATE, which stops a parameter being fitted to
the one file blocking a gate — but both halves were drawn from the same
population, so neither can answer the question a user actually asks: *will this
be smaller on MY archive, which is not in your corpus?*

Two instruments answer different halves of that question.

| question | command | artifact |
|---|---|---|
| does the win-rate survive on archive types we never tuned on? | `make holdout` | `report.txt`, `holdout.tsv`, `board.tsv` |
| where in content space does the verdict FLIP? | `make surface` | surface TSV + `CLIFF` lines |

Both are MEASUREMENTS, not ratchets. Tests pin the GENERATORS (sha256, so two
runs are comparable) and never pin a ratio — a surface that moves is data, not a
regression.

---

## 1. The holdout corpus (`scripts/campaign/holdout.sh`)

A third population of 12 members, each 4 MiB, whose archive TYPES are absent
from the tuning corpus:

| member | class | why it is not on the board |
|---|---|---|
| `src.tar` | USTAR tarball of C-like sources | board's `monorepo.tar` is GATE, and this is source text, not mixed binaries |
| `events.jsonl` | JSONL logs | different schema and statistics than `data.json` |
| `proto.tlv` | protobuf-style varint TLV | no length-delimited binary record format on the board |
| `wide.csv` | 24-column float-heavy CSV | `data.csv` is 6 columns of small integers |
| `vm.img` | page-granular zero/code/config/pointer/noise mix | no mixed-page image on the board |
| `cjk.txt` | UTF-8 Chinese prose, Zipf-skewed | `aozora.txt` is Japanese with kana; different symbol statistics |
| `feed.xml` | attribute-heavy nested feed | `markup.xml` is GATE and differently shaped |
| `dna.fasta` | 4-letter alphabet + N runs | `ecoli.fastq` is GATE and carries quality lines |
| `mail.mime` | MIME text + base64 bodies | no base64 armour anywhere on the board |
| `apache.log` | combined access log | `access.log` is GATE |
| `heap.bin` | 8-byte-aligned pointer dump | pointer-shaped binary is absent |
| `repo.md` | Markdown with code fences and tables | mixed prose/code at paragraph scale is absent |

**Generated from seeds, never stored as data.** `examples/holdout_gen.rs`
materializes them by seeded integer arithmetic — identical bytes on every
platform and every run — and verifies each against `gzippy::holdout::PINS`,
refusing (exit 3) on any drift. A drifted generator therefore cannot silently
grade as "the holdout", and repinning consciously voids every earlier win-rate.

**Grading.** Every cell is `(member, level 1-9, threads 1/4, rival)`, a win is
`ours <= rival` at the same level (per-label; a tie is a win), and our output is
roundtripped through `gzip -dc` with sha256 at every cell — a mismatch ABORTS
the run rather than scoring a corrupt-but-smaller output. The comparison leg
grades the TUNE set with the same code in the same invocation, so a
holdout-vs-board gap cannot be a methodology difference.

**The alarm.** Holdout win-rate materially below board win-rate (the report
draws the line at 6 points, roughly the board's own residual failure rate) means
the board is measuring fit rather than compression. The report also splits by
rival, thread count, level, and member, because a headline delta with no class
behind it is not actionable.

**THE ONE RULE: nothing is ever tuned against these files.** Reading a holdout
number to choose a knob turns the only unbiased estimate we have into another
tuning set. Use it to detect fit, never to fix it — fix it on TUNE, then re-read
the holdout.

---

## 2. The response surface (`examples/surface_probe.rs`)

`src/fixtures.rs` grew four named fixtures imitating corpus classes. The sampler
(`surface_generate` / `surface_points`) is the SPACE those points live in, with
five axes:

* `entropy_bits` — order-0 entropy of fresh literals, 2/4/6/8 bits (hit by
  mixing a hot subset with the full alphabet; the probe reports the MEASURED
  entropy beside the target)
* `period` — every back-reference copies from exactly this distance: 16 / 128 /
  1024 / 8192
* `long_matches` — match-length profile, short (3-8) vs long (32-258)
* `alphabet` — fresh-literal alphabet size: 16 / 64 / 256
* `records` — a fixed 16-byte record skeleton every ~256 bytes, or none

60 declared points, compressed at L1/L6/L9 at T1, each roundtripped through
`gzip -dc` (mismatch aborts), each scored against libdeflate and gzip.

**Cliffs.** Two points are adjacent when they differ along exactly ONE axis by
one grid step. A cliff is an adjacent pair where ratio-vs-rival crosses 1.0
(the verdict flips) or moves more than 2 points. Each cliff prints its rival,
level, axis and both coordinates — a named generalization boundary, i.e. the
failure mode a new archive type hits when its content crosses that line.

Cliffs are diagnostics, not gates. The useful reading is the AXIS: a cliff on
`period` says our verdict depends on match distance; a cliff on `entropy` says
it depends on literal statistics; a cliff on `records` says a structural grid
flips it. That names the mechanism to diff against the vendor.

---

## Running them

```sh
make holdout                                  # holdout + board comparison, L1-9, T1/T4
CAMPAIGN_LEVELS=1,6,9 make holdout            # cheaper sweep
scripts/campaign/holdout.sh --holdout-only    # no local corpus needed
make surface OUT=/tmp/surface.tsv             # TSV + CLIFF lines on stdout
```

`make holdout` needs the tuning corpus at `CAMPAIGN_CORPUS_ROOT` for the
comparison leg only; the holdout leg runs anywhere, because the corpus is a
binary rather than a download. Missing rivals (igzip is not packaged on every
box) are RECORDED in the report header rather than silently dropped.

Pinned by tests, not by ratios:

* `holdout::tests::holdout_members_are_frozen` — sha256 of all 12 members
* `holdout::tests::sha256_known_vectors` — so a bug in the local sha256 cannot
  re-pin everything
* `fixtures::tests::surface_generators_are_frozen` — one manifest sha over all
  60 surface points
* `fixtures::tests::surface_points_are_prefix_stable` — the probe's 1 MiB and
  the test's 64 KiB are the same stream
* `fixtures::tests::surface_entropy_axis_is_monotone` — the entropy axis
  actually orders the content it claims to
