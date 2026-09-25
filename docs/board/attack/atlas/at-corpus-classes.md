# The SHAPE ATLAS — which parse-shape family wins byte size, per member class

Written 2026-09-05 as a land-before-measurement artifact for the attack board
(`docs/board/attack/` knowledge lanes, attempt-4 partnership pattern:
`../sequence/seq-verdict-playbooks.md`).
READ+COMPUTE-ONLY session: python3 stdlib feature panel + the campaign's banked
receipts; no box time, no cargo builds, no lever runs. This file exists so a
future lever can predict its accept-policy split **before** it runs anything
destructive — the exact thing that killed the L2 min-3 attempt (PR #369).

Provenance: computed at brain worktree `5900d17a` on the local Mac. Every
number is either measured by the panel script below (declared protocol) or a
banked campaign receipt quoted verbatim with its citation. Nothing here was
re-measured on the box.

## 1. Campaign reality this atlas must fit

* The campaign grades compression **per (file, level, rival, threads)** cell;
  there is no win until the *file* wins. Members file into shape classes
  (`corpus_split.json:10-22` declares the 12 content classes; `:24-41` the
  TUNE/GATE member lists).
* The **L2 min-3 lever is falsified** (PR #369, both lanes measured to the
  stop rule). The owner's arbitration (PR #370, 2026-09-05): *"I delete the
  pick-min and all that terrible two-encode machinery… size may spend <= 1%
  per cell but we will not lose on wall clock under any conditions."*
* The deleted hybrid at L2 was **`params_l2_gzip_deflate_fast()`**
  (`level.rs:346-358`): gzip `deflate_fast` shape — chain depth 8, nice 16,
  `forced_min_match_len = 3`, `greedy_accept_far_len3 = true` (no gate,
  no shadow), hosted as the second arm of the two-arm legacy pick-min
  (`deflate/mod.rs:564-583`; the pick-min is the machinery the owner deleted).
  Per PR #369's body it places **+1.06% matched positions / -7.67% literals**
  vs the port's min-4 greedy — "~2M 3-byte matches the port can currently
  never find."
* The killed-lever ledger, quoted from the falsification record
  (`.worktrees/rules/docs/plan-2026-09-one-encoder.md:290-297`): *the deleted
  hybrid's 0.27% silesia margin was a PER-FILE pick-min choice between two
  full encodes; the min-3 arm is byte-identical to mine (whole-corpus A/B sha
  `3182904e`) and floods synthetic tabular with +7.9% len-3 noise that no
  accept policy (unconditional, slack-gated, absent) can remove —
  `min_len = 3` itself is the spender. One encode must hold both classes;
  min-4 holds the synthetic tie and wins the wall class.*
  The handoff addendum records the same triangle
  (`.worktrees/rules/docs/handoff-2026-09-04.md:253-257`).
* The measured cells this atlas must place:
  - **+7.9%** — synthetic tabular, L2, the min-3 lane's noise
    (`.worktrees/rules/docs/handoff-2026-09-04.md:256`); the gated shape's
    own adjudication measured tabular:L2 at **302,492 vs libdeflate
    281,326 = +7.5%** with 18,327 *near* len-3 accepts through the main arm
    offsetting everything the far gate did (PR #369 comment, "Falsification
    verdict").
  - **+0.19/+0.16** — `minjs.min.js L5` vs gzip/pigz
    (`docs/plan-2026-09-one-encoder.md:70`, the §2b board census, brain copy).
  - **+0.13/+0.17** — `data.sqlite L4` vs gzip/pigz (same line).
  - The **0.27%** archive-mix margin the pick-min used to bank
    (`silesia.tar L2/T1`, ratio 0.9973→1.0000 vs libdeflate — PR #369 body).
* Session-given receipt from the parent plan (atlas commission): the archive
  mix class measured at **2.682 vs 2.645 bits/byte** with vs without the
  trigram-chain (min-3) parse on the megabyte tar — a ~1.4% class win for
  chains that the byte-identity A/B shows the single min-3 shape reproduces
  on the tar while it cannot hold on synthetic tabular. Repo-stored
  band-adjacent numbers: the gate-variant run emitted **70,463,509 B**
  (2.660 bits/byte) vs the unconditional lane **70,877,952 B** (2.676
  bits/byte) over the corpus's 203 MB tar (= `benchmark_data/silesia.tar`
  211,968,000 B; bits/byte values derived here by size×8/input) — the gate
  recovered **-0.58% (-414,443 B)** vs unconditioned far accepts while
  keeping the min-3 chain (PR #369 comment).

The atlas's job: predict, for ANY member class, which of the four parse
shape families wins byte size vs libdeflate's bytes — without running a
lever — and price the accept-policy split (near ≤4096 / far-lane) from the
byte statistics of the class itself.

## 2. The four shape families (the columns of the matrix)

| # | family | in-tree anatomy | far-len-3 policy |
|---|---|---|---|
| **T3** | **min-match-3 chain + far accept** — the deleted hybrid arm | `params_l2_gzip_deflate_fast()` `level.rs:346-358`; hash3 chain depth 8 `hc.rs:91,109-112` | accepts far len-3 unconditionally (`greedy_accept_far_len3=true`) |
| **M4** | **min-4 greedy (the libdeflate port shape)** — shipped | `ldx/compress_greedy.rs:55,118` and legacy `parse/greedy.rs:244-245`: accept iff `length >= min_len && (length > 3 \|\| offset <= 4096 …)` | fixed guard: len-3 refused beyond 4096 (greedy) / 8192 (lazy) (`parse/far_len3.rs:5-6`) |
| **M4G** | **min-4 + FarLen3Gate (lazy gated far-len-3)** — the working single-encode compromise | `parse/greedy.rs:209-227` (`FarLen3Gate::recalc` at 10 KB→doubling cadence; INERT at block start = byte-identical to the shipped guard, `far_len3.rs:28-30`); margins `far_len3.rs:44-49` | len-3 beyond the guard only when the gate prices the EXACT three bytes cleanly below a 2-bit margin (+1 eighth-bit slack when all three are expensive) |
| **NO** | **near-optimal parse** (L10-12 ultra engine) | `level.rs:58` `Strategy::NearOptimal`; `cli.rs:222-226`, `lib.rs:22-23` | cost-model priced per candidate; no fixed min-match flood by construction |

Fixed-guard receipts (why a *policy* alone can never fix the flood,
`parse/far_len3.rs:11-22`): the two simpler policies are measured dead on the
tie-guard levels — **UNCONDITIONAL drop: 53 flips** (weights.safetensors
+203,233 B, tool.bin +135,452 B, sil40 L2 +47,819 B); **mean-literal-cost
gate: 28 flips** (tool.bin +77,501 B, sil40 L2 +28,848 B, minjs +3,804 B).
"len-3 distance policy is content-dependent; a fixed constant is wrong in
both directions." The shipped alternative prices the exact bytes per-candidate
(`far_len3.rs:24-30`) — a parser-internal cost model, not a content detector.

Working-set note: the port's `ht` matchfinder has no length-3 table at all
(the libdeflate two-way bucket shape, `ht.rs:5,237,254-260` — the 64 KiB
`hash3_tab` addition is attempt-2's still-open L1 lane, `ht.rs:260`;
`l1-next-lever.md:57-69`), so an M4/M4G lever pays a *hash3 insertion*
working-set cost per level it touches; the deleted arm's depth-8 hash3 chain
is `hc.rs:91`.

## 3. The measured feature panel (this session; frozen-FNV-verified fixtures)

Method (declared protocol):
1. The four fixtures are **byte-exact python replicas of `src/fixtures.rs`**
   (`fixtures.rs:33-43` XorShift64 `13>>/7>/17<<`; seeds `fixtures.rs:72,103,127,142`).
   Verification: the `fixtures_are_frozen` FNV-1a pins `fixtures.rs:402-407`
   — all four match (`atlas/at-corpus-features.py` exit 0, "frozen FNV OK").
2. Silesia members unpacked from `benchmark_data/silesia.tar.xz`
   (`tar -xJf … -C /tmp/atlas-silesia` — 12 members, 211,938,953 B total).
   Nothing in the campaign tree was touched; extraction lives in /tmp.
3. Every sample longer than 4 MiB is reduced to its **first 4 MiB prefix**
   (fixtures are their full 1 MiB). Note: `silesia.tar`'s prefix is therefore
   mostly `dickens` (first tar member) — its row is a proxy for the tar's
   head, not a member census; the class-grade surrogate for "the megabyte
   tar" is the member mix, per row below.
4. Trigram "match" = position whose 3-byte key occurred before, distance =
   to the **nearest previous occurrence** (matchfinder-equivalent), window
   32,768; distance buckets cut exactly at the two fixed guards (4096 greedy,
   8192 lazy). Script: **`docs/board/attack/atlas/at-corpus-features.py`**
   (python3 stdlib only; inline as Appendix A).

### Panel 1 — full feature table (script output, verbatim)

```
### Panel 1 - full feature table

| member | source | trigram positions | dist <=64 | 65-512 | 513-2048 | 2049-4096 (%) | 4097-8192 | 8193-16384 | 16385-32768 (%) | >32768 (%) | first (%) | H3 bt/tri | H3/3 | trigram kinds % | H0 bb | byte vals | top-8 mass % | acorr4 % | acorr8 % | acorr16 % | acorr32 % | stride max % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text (fixture)` | fixture replica 1 MiB, frozen FNV OK | 1,048,576 | 23.89 | 51.40 | 18.14 | 3.29 | 1.95 | 0.93 | 0.26 | 0.071 | 0.1 | 8.323 | 2.77 | 0.07 | 4.08 | 41 | 68.2 | 10.16 | 8.46 | 8.63 | 8.80 | 10.16 |
| `tabular (fixture)` | fixture replica 1 MiB, frozen FNV OK | 1,048,576 | 48.05 | 22.96 | 11.57 | 6.21 | 5.48 | 3.76 | 1.60 | 0.224 | 0.2 | 8.415 | 2.81 | 0.16 | 4.30 | 31 | 59.0 | 3.22 | 3.72 | 7.78 | 4.24 | 7.78 |
| `binary (fixture)` | fixture replica 1 MiB, frozen FNV OK | 1,048,576 | 31.60 | 8.69 | 0.29 | 0.35 | 0.59 | 0.86 | 0.86 | 1.808 | 55.0 | 13.093 | 4.36 | 54.95 | 6.08 | 256 | 44.9 | 26.56 | 23.29 | 15.20 | 4.55 | 26.56 |
| `noise (fixture)` | fixture replica 1 MiB, frozen FNV OK | 1,048,576 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.05 | 0.09 | 2.886 | 96.9 | 19.938 | 6.65 | 96.92 | 8.00 | 256 | 3.2 | 0.39 | 0.38 | 0.38 | 0.39 | 0.39 |
| `dickens` | 4,194,304 of 10,192,446 B | 4,194,304 | 11.12 | 32.74 | 25.62 | 10.22 | 7.55 | 5.03 | 3.15 | 4.142 | 0.4 | 10.780 | 3.59 | 0.43 | 4.53 | 98 | 59.1 | 7.07 | 6.59 | 6.67 | 6.68 | 7.07 |
| `mozilla` | 4,194,304 of 51,220,480 B | 4,194,304 | 7.64 | 8.84 | 2.69 | 0.88 | 0.73 | 0.57 | 0.79 | 10.884 | 67.0 | 19.352 | 6.45 | 66.99 | 7.70 | 256 | 14.8 | 6.97 | 6.67 | 6.04 | 5.06 | 6.97 |
| `mr` | 4,194,304 of 9,970,564 B | 4,194,304 | 50.43 | 12.97 | 14.37 | 4.67 | 3.71 | 2.88 | 2.23 | 6.832 | 1.9 | 8.383 | 2.79 | 1.90 | 3.58 | 256 | 75.6 | 63.51 | 61.81 | 59.64 | 57.58 | 63.51 |
| `nci` | 4,194,304 of 33,553,445 B | 4,194,304 | 66.14 | 19.44 | 3.72 | 4.25 | 2.67 | 1.23 | 0.79 | 1.588 | 0.2 | 5.717 | 1.91 | 0.16 | 2.43 | 58 | 90.7 | 24.56 | 23.48 | 33.96 | 33.59 | 33.96 |
| `ooffice` | 4,194,304 of 6,152,192 B | 4,194,304 | 17.17 | 17.93 | 10.77 | 5.12 | 4.93 | 4.71 | 4.68 | 22.284 | 12.4 | 14.900 | 4.97 | 12.40 | 6.57 | 256 | 34.9 | 5.46 | 5.02 | 5.06 | 4.85 | 5.46 |
| `osdb` | 4,194,304 of 10,085,684 B | 4,194,304 | 4.45 | 10.97 | 13.42 | 10.98 | 13.71 | 13.39 | 8.51 | 13.518 | 11.0 | 14.047 | 4.68 | 11.04 | 6.59 | 256 | 27.2 | 5.80 | 4.20 | 2.74 | 0.79 | 5.80 |
| `reymont` | 4,194,304 of 6,627,202 B | 4,194,304 | 31.04 | 35.08 | 15.09 | 4.91 | 4.06 | 4.69 | 2.17 | 2.737 | 0.2 | 9.342 | 3.11 | 0.22 | 4.76 | 81 | 58.5 | 4.15 | 6.25 | 6.46 | 6.48 | 6.48 |
| `samba` | 4,194,304 of 21,606,400 B | 4,194,304 | 21.54 | 33.73 | 16.70 | 6.21 | 5.31 | 4.00 | 3.04 | 5.118 | 4.4 | 12.012 | 4.00 | 4.35 | 5.63 | 256 | 40.5 | 8.84 | 8.53 | 8.25 | 7.77 | 8.84 |
| `sao` | 4,194,304 of 7,251,944 B | 4,194,304 | 5.03 | 11.69 | 11.60 | 5.11 | 4.17 | 3.31 | 2.70 | 15.124 | 41.3 | 17.071 | 5.69 | 41.27 | 7.49 | 256 | 19.5 | 1.24 | 1.34 | 0.34 | 1.13 | 1.34 |
| `webster` | 4,194,304 of 41,458,703 B | 4,194,304 | 14.85 | 38.83 | 17.19 | 7.75 | 6.62 | 5.10 | 3.58 | 5.514 | 0.6 | 10.725 | 3.57 | 0.57 | 4.98 | 97 | 48.6 | 5.82 | 4.44 | 4.85 | 4.48 | 5.82 |
| `x-ray` | 4,194,304 of 8,474,240 B | 4,194,304 | 9.74 | 8.61 | 6.02 | 13.66 | 7.47 | 7.02 | 6.40 | 31.696 | 9.4 | 15.288 | 5.10 | 9.39 | 6.44 | 256 | 46.9 | 43.50 | 41.05 | 38.07 | 33.27 | 43.50 |
| `xml` | 4,194,304 of 5,345,280 B | 4,194,304 | 43.43 | 31.55 | 14.75 | 3.48 | 2.15 | 1.47 | 0.98 | 1.522 | 0.7 | 10.296 | 3.43 | 0.68 | 5.42 | 90 | 40.1 | 2.29 | 5.74 | 6.25 | 4.11 | 6.25 |
| `logs.txt` | 4,194,304 of 221,249,597 B | 4,194,304 | 10.60 | 77.00 | 4.49 | 3.33 | 2.96 | 0.03 | 0.07 | 1.482 | 0.0 | 8.122 | 2.71 | 0.04 | 5.22 | 59 | 43.9 | 3.94 | 5.56 | 2.82 | 1.89 | 5.56 |
| `software.archive` | 4,194,304 of 221,249,576 B | 4,194,304 | 13.83 | 30.91 | 52.18 | 0.99 | 0.01 | 1.00 | 0.03 | 1.009 | 0.0 | 8.508 | 2.84 | 0.04 | 5.04 | 68 | 50.4 | 4.78 | 3.29 | 6.03 | 5.29 | 6.03 |
| `silesia.tar` | 4,194,304 of 211,968,000 B | 4,194,304 | 11.14 | 32.73 | 25.61 | 10.22 | 7.55 | 5.02 | 3.14 | 4.141 | 0.4 | 10.781 | 3.59 | 0.43 | 4.53 | 99 | 59.1 | 7.10 | 6.62 | 6.70 | 6.70 | 7.10 |
```

Column glossary: trigram positions = all length-3 windows of the sample.
The seven distance buckets split the *repeat* positions. `>32768` mass is
outside the DEFLATE window (uncopyable); `first` = positions whose trigram
was never seen before in the sample. `H3` = entropy of the trigram
distribution (bits per trigram occurrence), `H3/3` its per-byte value —
**the cheap-literal axis**: this is a SOURCE-statistic, distinct from the
campaign's compressed-output bits/byte (2.682/2.645, §1); do not conflate.
`trigram kinds %` = distinct trigrams / trigram positions (low = near-total
recycling = chains dense). `byte vals`/`top-8` = alphabet width and
concentration. `acorrN` = byte-equality autocorrelation at stride 4/8/16/32
(the columnar stride signature).

### Panel 2 + 3 — the policy windows and class aggregates (script output, verbatim)

```
| member | shape class | near<=4096 % | 4097-8192 % | 8193-32768 % | matched<=32768 % | >window % | acorr_max % |
|---|---|---:|---:|---:|---:|---:|---:|
| `text (fixture)` | prose-text | 96.72 | 1.95 | 1.20 | 99.86 | 0.071 | 10.2 |
| `tabular (fixture)` | precision-columns | 88.79 | 5.48 | 5.36 | 99.62 | 0.224 | 7.8 |
| `binary (fixture)` | native-binary | 40.93 | 0.59 | 1.72 | 43.24 | 1.808 | 26.6 |
| `noise (fixture)` | incompressible | 0.02 | 0.03 | 0.15 | 0.19 | 2.886 | 0.4 |
| `dickens` | prose-text | 79.70 | 7.55 | 8.17 | 95.43 | 4.142 | 7.1 |
| `mozilla` | native-binary | 20.04 | 0.73 | 1.36 | 22.13 | 10.884 | 7.0 |
| `mr` | scientific-stride | 82.44 | 3.71 | 5.12 | 91.27 | 6.832 | 63.5 |
| `nci` | precision-columns | 93.55 | 2.67 | 2.02 | 98.25 | 1.588 | 34.0 |
| `ooffice` | already-compressed | 51.00 | 4.93 | 9.39 | 65.32 | 22.284 | 5.5 |
| `osdb` | precision-columns | 39.82 | 13.71 | 21.90 | 75.44 | 13.518 | 5.8 |
| `reymont` | prose-text | 86.12 | 4.06 | 6.86 | 97.04 | 2.737 | 6.5 |
| `samba` | archive-mix | 78.18 | 5.31 | 7.04 | 90.53 | 5.118 | 8.8 |
| `sao` | scientific-stride | 33.42 | 4.17 | 6.01 | 43.61 | 15.124 | 1.3 |
| `webster` | prose-text | 78.62 | 6.62 | 8.68 | 93.92 | 5.514 | 5.8 |
| `x-ray` | scientific-stride | 38.03 | 7.47 | 13.42 | 58.92 | 31.696 | 43.5 |
| `xml` | cadence-markup | 93.20 | 2.15 | 2.45 | 97.80 | 1.522 | 6.3 |
| `logs.txt` | local-log | 95.42 | 2.96 | 0.10 | 98.48 | 1.482 | 5.6 |
| `software.archive` | archive-mix | 97.91 | 0.01 | 1.03 | 98.95 | 1.009 | 6.0 |
| `silesia.tar` | archive-mix | 79.71 | 7.55 | 8.17 | 95.43 | 4.141 | 7.1 |

| shape class | members | near mean % | far mean % | matched mean % | H3/3 mean | H0 mean | top-8 mean % | acorr max mean % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| already-compressed | `ooffice` | 51.00 | 14.32 | 65.32 | 4.97 | 6.57 | 34.9 | 5.5 |
| archive-mix | `samba`, `software.archive`, `silesia.tar` | 85.27 | 9.70 | 94.97 | 3.48 | 5.07 | 50.0 | 7.3 |
| cadence-markup | `xml` | 93.20 | 4.60 | 97.80 | 3.43 | 5.42 | 40.1 | 6.3 |
| incompressible | `noise` | 0.02 | 0.17 | 0.19 | 6.65 | 8.00 | 3.2 | 0.4 |
| local-log | `logs.txt` | 95.42 | 3.06 | 98.48 | 2.71 | 5.22 | 43.9 | 5.6 |
| native-binary | `binary`, `mozilla` | 30.48 | 2.20 | 32.68 | 5.41 | 6.89 | 29.9 | 16.8 |
| precision-columns | `tabular`, `nci`, `osdb` | 74.05 | 17.05 | 91.10 | 3.13 | 4.44 | 59.0 | 15.8 |
| prose-text | `text`, `dickens`, `reymont`, `webster` | 85.29 | 11.27 | 96.56 | 3.26 | 4.59 | 58.6 | 7.4 |
| scientific-stride | `mr`, `sao`, `x-ray` | 51.30 | 13.30 | 64.60 | 4.53 | 5.84 | 47.4 | 36.1 |
```

(Panel 2's class labels are the atlas's working shape classes — see §4; the
aggregate table recomputes means from Panel 2 rows. Panel 1/2/3 are the
script's own verbatim stdout.)

## 4. The taxonomy — member classes the campaign actually grades

The fixture-class anchors from `src/fixtures.rs:60-146` (each fixture imitates
one class the real corpus taught): `text` → prose/aozora/dickens, `tabular` →
"data.csv class (where the length-3 rule says DISABLE)" (`fixtures.rs:98-101`),
`binary` → "armexe/tool.bin class (where the length-3 rule says ENABLE and
earns real bytes)" (`fixtures.rs:123-125`), `noise` → movie.mp4 stored class.
The corpus classes they proxy: `corpus_split.json:10-22`.

Mapping the Campaign board names (TUNE+GATE sets, `corpus_split.json:26-55`)
onto the shape classes measured above:

| atlas shape class | measured local members (this panel) | campaign members it covers |
|---|---|---|
| **precision-columns** | `tabular` fixture (H3/3 2.81, 31 byte vals, near 88.79%, 48.05% within ≤64 B), `nci` (H0 2.43!), `osdb` | `data.csv`, `data.sqlite` (structured-data), `data.parquet` (columnar-db) — the numeric-cadence cells |
| **archive-mix** | `samba` (near 78.18 + far 12.35, H3/3 4.00, 256 byte vals), `software.archive` (mid-band 52.18% at 513-2048), `silesia.tar` | `monorepo.tar`, `sil40`, the silesia tar captures — member-boundary header/index trigram chains at wide alphabet |
| **prose-text** | `text`, `dickens`, `webster`, `reymont` (75-86% near + 11-15% mid-far) | `aozora.txt`, `dd79_text6` |
| **local-log** | `logs.txt` (77% mass in ONE band 65-512, H3/3 2.71) | `access.log` |
| **cadence-markup** | `xml` (93.2% near, 90 byte vals, H3/3 3.43) | `markup.xml`, `data.json` |
| **minified-code** | (no local sample; far_len3.rs receipts carry the class) | `minjs.min.js`, `sil40` |
| **native-binary** | `binary` (43.24% matched, H0 6.08, acorr4 26.56 = 16 B record skeleton), `mozilla` (22.13% matched, H0 7.70) | `armexe.elf`, `tool.bin`, `dd79_bin6`, `engine.wasm`, `winexe.exe`, `symbols.dwarf` |
| **scientific-stride** | `mr` (acorr 63.51), `x-ray` (43.5 + 31.7% >window), `sao` (43.61 matched) | `ecoli.fastq`-adjacent numeric payloads |
| **already-compressed** | `ooffice` (65.32 matched, 22.3% >window) | `photo.jpg`, `movie.mp4` |
| **incompressible** | `noise` (matched 0.19%) | `weights.safetensors` |

`dd79_text6`/`dd79_bin6`/`sil40` are gate shorthand in the receipts
(`corpus_split.json:28-29`); `dd79_bin6`'s literal cost is named in-tree at
`parse/far_len3.rs:7` (6.54 bits/byte).

## 5. Conclusions per class — why the ledger split the way it did

**archive-mix: trigram-RICH and the trigram chains PAY.** The class's
matched-in-window mass is 90-99% (samba 90.53%, software.archive 98.95%)
with an alphabet-wide, mid-distance distance profile (software.archive puts
52.18% of trigram mass at 513-2048; samba keeps 12.35% beyond 4096; beyond
32 KiB only 5.1%) and EXPENSIVE literals (H3/3 3.48-4.00; members' H0 up to
5.63; dd79_bin6 6.54 bits/byte receipt at `parse/far_len3.rs:7`). A len-3
match there replaces three costly literals with a len-symbol + a
distance-symbol whose alphabet the block has already paid for — that is the
mechanism the vendor diff names in zlib's favor. Measured receipts that the
chains pay here, and ONLY here:
* pick-min archived the deflate-fast (T3) bytes exactly on this class and the
  libdeflate-tie (M4) bytes on cadence data (PR #369 prompt record — "both
  proven by the `-2` A/B"); the archived margin over libdeflate is the
  **+0.27%** `silesia.tar L2/T1` cell (PR #369 body).
* On synthetic tabular the SAME shape floods **+7.9%** len-3 noise that no
  accept policy removes (`handoff-2026-09-04.md:256`); the whole-corpus A/B
  sha `3182904e` says the single min-3 shape is byte-identical to the deleted
  hybrid everywhere the pick-min's archive-mix picks had archived its bytes
  (`.worktrees/rules/docs/plan-2026-09-one-encoder.md:291-293`) — the tension
  is the two classes, not a bug inside either one.
* The session-given 2.682 vs 2.645 bits/byte quantifies the class's trigram
  appetite on the megabyte tar (~1.38% smaller output with chains); the
  band-adjacent banked pair 70,463,509 vs 70,877,952 (2.660 vs 2.676
  bits/byte derived from the PR #369 gate/unconditional sizes over
  211,968,000 B) brackets the same effect for the *far lane alone*: the gate
  recovered -0.58% (-414,443 B) of the unconditional lane's overspend while
  KEEPING the chain. So the archive-mix row's money lives in BOTH halves:
  near chains (min-3) + a priced far gate.

**precision-columns: trigram-rich but the chains DROWN.** The class is
trigram-saturated near-locally with a tiny literal alphabet: tabular puts
48.05% of trigram matches within ≤64 B, 88.79% within 4096; nci 93.55% within
4096 at H0 2.43 (top-8 mass 90.7%); no stride (tabular max acorr 7.78).
The bytes the chain replaces are already-cheap numerals/punctuation fitted
into the same litlen alphabet — the token overhead (len sym + dist sym +
extra bits) exceeds three cheap literals' cost, AND the eager min-3 accepts
break the parse before it discovers the repeated record skeletons that a
min-4 pass would take as long matches. Measured anatomy (PR #369 comment,
falsification verdict): the gate-variant build on `tabular:L2` stayed at
**302,492 vs libdeflate 281,326 (+7.5%**); a reverted-probe with the gate
inert showed the residual is **18,327 NEAR len-3 matches (offset ≤ 4096)**
through the forced min-3 main arm — "they alone outweigh the +20 KB
residual, and the gate's 715 far accepts are net +723 B. No far-len-3 accept
policy (unconditional, slack-gated, or absent) closes this: the forced
**min-match-3 half** of the hybrid shape is what spends the tabular wins."
Every number in this paragraph is a banked receipt plus Panel-2 measured
mass — this is the class the atlas exists to warn about.

**Where the named cells sit in that taxonomy:**
* **+7.9%** = precision-columns, specifically the synthetic tabular fixture
  cell at L2 (1 MiB `tabular`, the `data.csv` class — `corpus_split.json:63`
  covers data.csv under log-text's tune slot, but the fixture is the class's
  cadence anchor). The three receipts stack: +7.5% (gated shape, PR #369),
  +7.9% ("no accept policy removes"). Panel-2 predicts it from two numbers:
  near ≤4096 ≥ 85% + trigram kinds < 0.2%.
* **+0.19/+0.16** = `minjs.min.js L5` vs gzip/pigz (board survivors,
  `plan-2026-09-one-encoder.md:70`) — minified-code: trigram-rich (near-heavy)
  with expensive mixed literals. The class responds to per-symbol pricing
  (the mean-literal gate OVER-accepted minjs by +3,804 B, `far_len3.rs:19-20`)
  — the residue is a parse-cadence cell, not a min-3 hole.
* **+0.13/+0.17** = `data.sqlite L4` — precision-columns but WITHOUT any
  min-3 lane in play: the shipped M4/M4G shape carries this cell, and the
  documented symptom is the ladder-sag class (legacy greedy emitted
  14.8 MB at data.sqlite where the L1 fast path emitted 12.9 MB —
  `deflate/mod.rs:575-578`), not a len-3 flood.
* The **board-worst +1.07%** is `access.log L5 vs gzip` (plan §2b:68-70) —
  local-log: 95.42% of its trigram mass sits in the 65-512 B band; that is a
  laziness/cadence gap at L5, not a len-3 distance policy gap.

**Why "min-4 holds the tie" holds synthetically**: `binary` at L2 sits
**4 B under libdeflate-gzip -2 (666,108 vs 666,112)** inside the tie cage
(`tests/size_invariants.rs:76`), and the tabular ties are the pick-min's
libdeflate-archived bytes (PR #369 prompt record) — the M4 shape is the only
one whose synthetic ledger is neutral by construction, which is why it is
the one-encoder answer.

## 6. THE CONSULT TABLE — class → parse shape, with predicted accept-policy split

What a future lever reads here INSTEAD of running a destructive grid. The
accept-policy split = how much of the class's trigram mass sits in each
policy window (Panel 2's columns) TIMES how the class prices literals.
Predictions marked [M] are banked measurements; [P] are the atlas's
pre-registered predictions (bands, falsifiers in §7).

| shape class | T3 min-3 chain+far | M4 min-4+4096 guard (shipped) | M4G min-4+FarLen3Gate | NO near-optimal | accept-policy split |
|---|---|---|---|---|---|
| **archive-mix** | **WINS +0.27% [M]** vs libdeflate (silesia.tar L2/T1; pick-min archive of T3 bytes); class win band ~1.0-1.4% [P, from 2.682→2.645] | loses the archive-mix margin (the deleted hybrid's exact conviction, superseded to ≤1% ceiling spend) [M] | keeps most of T3's win; measured -0.58% vs T3-unconditional on the tar [M]; predicted -0.2..-0.6% vs T3, +0.1..+0.4% vs libdeflate [P] | ≥ M4G by construction of its cost model [P] (never differentially promoted — wall-priced at L10-12) | near-lane: accept (chains pay); far-lane: accept ONLY where the gate prices the exact 3 bytes under a 2-bit margin; expected far accept share small (class far mass 9.7%) |
| **precision-columns** | **LOSES +7.5..+7.9%** [M] — the killer cell; the spender is `min_len=3` itself, no policy fixes | tie (libdeflate cell, 302,492-class avoided) [M] | tie — gate inert (near flood is not the far lane; gate's far accepts net +723 B there [M]) | ≥ M4G [P] (prices away the flood) | near-lane: DO NOT ENABLE min-3 (18,327 near accepts measured); far-lane: irrelevant (flood is near-side at ≤64 B, 48% of trigram mass) |
| **prose-text** | neutral ±0.2% [P] (mid-distance chain mass 11-15% at 5.0-5.6-bit literals — mild win, mild cost; no prose member appears in the recorded policy-flip lists [M]) | tie-to-tie [M: TUNE zero bigger] | ~tie; gate opens sparingly [P] | ≥ [P] | far-lane accept share ~ proportional to the 9-15% mid-far mass |
| **local-log** | mild win vs M4 on the 65-512 band (single dominant near band) [P ≤ +0.3%]; the +1.07% cell itself is [M], its mechanism predicted to be the L5 lazy band, not len-3 [P] | carries the +1.07% gap [M] | same as M4 + gate [P] | leverage via state carry, not len-3 | far-lane cheap (0.10% of trigram mass beyond 8 KiB, measured) — gate inert |
| **cadence-markup** | mild lose (ties) [P: like prose, no measured flip in far_len3 policy trials on xml] | tie [M] | tie [P] | ≥ [P] | gate opens on the 4.6% far mass only if tag-expensive; slack lens applies |
| **minified-code** | risk class: policies measured DEAD on it (sil40 +47,819 unconditional; minjs +3,804 mean-gate [M]) | carries +0.19/+0.16 residue [M] | the per-symbol gate is the right family here (minjs responds to exact-byte pricing) [M-flip evidence] | ≥ M4G | near+far mass is expensive-literal cadence; expected gate-accept share moderate; the residue belongs to parse cadence, NOT to a min-3 lane |
| **native-binary / debug-symbols** | unguarded far accepts flood it (+135 KB tool.bin [M]); but conditional len-3 at L1 **earns real bytes** (armexe 0.9658 = 3.4% win receipt [M], `l1-next-lever.md:51-52`) | the L2 donor cell (4 B tie cage) [M] | **the responsive shape [P]**: fixed guard donates ~41 KB at L2 / ~25 KB at L3 on dd79_bin6 [M: `far_len3.rs:8-9`]; gate prices exact bytes with expensive-slack | ≥ [P] | the only class where the gate's TRIGRAM_EXPENSIVE_ACCEPT_SLACK matters: H0 6.08-7.70, matched only 22-43% |
| **scientific-stride** | misshapen: strides want long matches; len-3 chains shatter RLE-like cadence [P] | carries [M: sao/x-ray stored-ish rows] | gate helps x-ray far spans (x-ray 20.89% of trigram mass beyond 4096 = 7.47+13.42; mr 8.83%) [P] | ≥ [P] | long-match lane; len-3 policy secondary; beyond-window mass needs stored coalescing |
| **already-compressed** | framing only; all shapes tie [P: matched 65%] | stores [M: noise L2 grid tie] | tie | tie | framing/stored; len-3 irrelevant |
| **incompressible** | flood danger documented (+203 KB weights.safetensors [M]) | stored grid [M] | stored grid + inert gate [M-far_len3 FAILS CLOSED] | tie | nothing to accept; gate must stay inert (its 2.886% beyond-window noise mass) |

Decision rule for a NEW member (consult before touching a len-3 lane):
1. Read the member's Panel-2 row (or run the panel on a new sample).
2. **Flood gate [R1]** — if `near<=4096% ≥ 85` AND `≤64 B` band ≥ 20% AND
   trigram kinds < 0.2% AND alphabet ≤ ~64 distinct bytes → the class is a
   chain-flood class; **any min_len=3 lane is predicted harmful** (+5..9%
   band) regardless of far governance. Members passing: tabular, nci.
   (xml escapes on alphabet width; logs.txt on band shape.)
3. **Chain-love gate [R2]** — if matched-mass ≥ 85% AND mid-far (4097-32768)
   ≥ ~9% AND H3/3 ≥ 3.4 → trigram chains pay; T3's bytes exist to be
   archived/banked; the single-encode route is M4G (+0.1..+0.4% [P]).
4. **Stride gate [R3]** — acorr max ≥ 30% → prefer the long-match lane and
   cost-priced/far-accept slack in MID-INT classes; len-3 chains neutral at most.
5. **Framing gate [R4]** — matched ≤ 45% → all len-3 policy is noise; the
   lever family is stored-grid/coalescing, not parse shape.
6. If R1 and R2 both pass for different members of the SAME mixed file
   (the tar case) → the class is a settled archive-mix: min-3 + priced far
   gate, and expected gain ≤ the ≤1% ceiling (banked +0.27%: lawful spend).

## 7. Pre-registered predictions (cheapest falsifiers first — 2-file or one-cell instruments, never a 22-file grid)

1. **[P1]** Any resurrected `min_len=3` lane at L2 **loses tabular cells by
   +5..+9%** regardless of its far-len-3 accept policy. Instrument: one
   `fulcrum why` on `tabular:L2` (bounded at the ±167 B target), NOT a grid.
2. **[P2]** Enabling/tuning the FarLen3Gate at L2 **changes no byte on the
   minjs/data.sqlite/weights.safetensors cells**: the gate starts INERT each
   block and only opens on exactly-priced wins, failing closed
   (`parse/far_len3.rs:28-30`), and the recorded dead-policy flip members put
   the class's far-lane volume in the thin band (sil40 +28,848 B, minjs
   +3,804 B under the dead mean-literal gate, `parse/far_len3.rs:17-20` [M]).
   The +0.19/+0.13 residues belong to parse cadence, not len-3. Instrument:
   per-file A/B at one level per file.
3. **[P3]** On the archive-mix (tar captures) the M4G gate keeps most of its
   price: predicted +0.1..+0.4% vs libdeflate on silesia-tar cells at L2
   (banked anchor: hedge +0.27% [M] with gate measured -0.58% below the
   unconditional lane [M]). Instrument: `fulcrum why silesia.tar:L2`.
4. **[P4]** `dd79_bin6`-class binaries: the fixed 4096 guard donates ~41 KB
   at L2 / ~25 KB at L3 [M, `parse/far_len3.rs:8-9`]; an M4G with the
   expensive-accept slack is predicted to close some of the +0.93% dd79_bin6
   L2/L3 vs gzip/pigz cells [P] without touching the L2 tie cage
   (4 B margin [M]).
5. **[P5]** Any min-3 lane opened for archive-mix MUST be gated T-level or
   per-window: the atlas's taxonomy predicts the flood re-appears in the
   synthetic corpus (tabular, nci) at ~7-9% the moment min-3 leaves the
   fixture. This is the two-divergent-classes falsifier that killed the L2
   min-3 lever; consult the §6 precision-columns row BEFORE designing it.

## 8. Honesty column

* No byte-level lever was run: the atlas predicts from feature panels + banked
  receipts, and its [P] bands are hypotheses with cheap falsifiers, not
  verdicts. Panels were computed single-threaded on prefixes (protocol §3);
  per-cell campaign-grade ratios could be added by a `libdeflate`-exact
  instrument that this laptop session did not run (the allowed ONE local
  cargo binary went unused — no applicable prebuilt binary was verified here).
* `silesia.tar`'s Panel-1 row is a prefix proxy (first member dominates); the
  class-level rows are what should be consulted, not the tar row alone.
* The 2.682/2.645 pair is a campaign session receipt (atlas mandate), not a
  repo-stored ledger; the repo-stored band-adjacent pair (70,463,509 vs
  70,877,952) brackets it for the far-gate half of the effect.
* Class labels are this atlas's own working taxonomy; `corpus_split.json`'s
  12 declared corpus classes map onto them in §4 (many-to-one on purpose:
  structured-data + columnar-db → precision-columns).
* The one-encoder law and the ≤1% size ceiling (PR #370) outrank every row
  here: a predicted archive-mix win is not a ship argument by itself —
  wall and clause-5/6 arithmetic still adjudicate.

## Appendix A — the panel script (verbatim, `docs/board/attack/atlas/at-corpus-features.py`)

```python
#!/usr/bin/env python3
"""Byte-level feature panel for the gzippy SHAPE ATLAS (at-corpus-classes.md).

Computes, per corpus member / synthetic fixture:
  - repeat-distance histogram of 3-byte trigram "matches": for every position
    whose trigram occurred before, the distance to the NEAREST previous
    occurrence, bucketed around the libdeflate far-len-3 guards
    (<=64, 65..512, 513..2048, 2049..4096, 4097..8192, 8193..16384,
    16385..32768, >32768) and the first-occurrence ("never") share,
  - matched-position fraction (distance <= 32768, i.e. inside the gzip window),
  - near (<=4096, the greedy fixed guard) and far (>4096) mass split,
  - trigram-distribution entropy H3 (bits/trigram) and distinct-trigram density,
  - byte-value entropy H0, distinct byte count, top-8 byte-value mass share,
  - columnar stride periodicity: byte-equality autocorrelation at 4/8/16/32 B.

Fixtures are generated replicating src/fixtures.rs (XorShift64>>13/7/17,
same seeds, same append order) and verified against the frozen FNV-1a pins of
the fixtures_are_frozen test. Data files are the extracted Silesia members
(/tmp/atlas-silesia/silesia) and benchmark_data; every sample longer than the
cap is reduced to its first 4 MiB prefix (declared protocol).

python3 stdlib only. Emits a markdown table on stdout.
"""
import array
import math
import os
import sys

MASK64 = (1 << 64) - 1
LEN = 1 << 20
CAP = 4 << 20  # uniform prefix cap for corpus members

API = " "  # markdown table separator


class XorShift:
    """u64 back-shift xorshift, identical to src/fixtures.rs:33-43."""

    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        self.s = x
        return x


def fnv1a(data):
    """FNV-1a 64-bit, the hash the fixtures_are_frozen test pins."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


# ---------------------------------------------------------------------------
# Fixture generators — byte-exact replicas of src/fixtures.rs:58-151
# ---------------------------------------------------------------------------

def gen_text(limit):
    words = ("the of and to in was it his that he her with for had is you not be "
             "she on at by which have from this him they were all are but said one "
             "when there them would been will who more no if out so what up their "
             "then time into little about could than like other some only over "
             "such down your").split()
    assert len(words) == 64
    rng = XorShift(0x7465787400000001)
    out = bytearray()
    words_in_sentence = 0
    while len(out) < limit:
        w = words[rng.next() % 64]
        if words_in_sentence == 0:
            c = w.encode()
            # .to_ascii_uppercase() on the first byte (fixtures.rs:78)
            first = c[0]
            c = bytes([first if not (97 <= first <= 122) else first - 32]) + c[1:]
            out += c
        else:
            out += w.encode()
        words_in_sentence += 1
        r = rng.next() % 100
        if r < 8 and words_in_sentence > 3:
            out += b". "
            words_in_sentence = 0
            if rng.next() % 4 == 0:
                out.append(10)
        elif r < 12:
            out += b", "
        else:
            out.append(32)
    return bytes(out[:limit])


def gen_tabular(limit):
    status = ("active", "inactive", "pending", "active", "active")
    rng = XorShift(0x7461627500000002)
    out = bytearray(b"id,timestamp,region,status,value,flag\n")
    v = 100_000
    while len(out) < limit:
        v += 1
        ts = 1_700_000_000 + rng.next() % 86_400
        region = rng.next() % 4
        st = status[rng.next() % 5]
        value = rng.next() % 100_000
        flag = rng.next() % 2
        out += ("{},{},region-{:02},{},{}.{:02},{}\n".format(
            v, ts, region, st, value // 100, value % 100, flag)).encode()
    return bytes(out[:limit])


def gen_binary(limit):
    rng = XorShift(0x62696E6100000003)
    out = bytearray()
    while len(out) < limit:
        out += bytes((0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01, 0x00))
        out += (rng.next() & 0xFFFFFFFF).to_bytes(4, "little")  # rng.next() as u32
        out += (((len(out) & 0xFFFFFFFF) ^ 0xDEADBEEF)).to_bytes(4, "little")
        for _ in range(rng.next() % 6 + 2):
            out += rng.next().to_bytes(8, "little")
        out += b"\x00" * (rng.next() % 48)
    return bytes(out[:limit])


def gen_noise(limit):
    rng = XorShift(0x6E6F697300000004)
    out = bytearray()
    while len(out) < limit:
        out += rng.next().to_bytes(8, "little")
    return bytes(out[:limit])


GENERATORS = {
    "text": gen_text,
    "tabular": gen_tabular,
    "binary": gen_binary,
    "noise": gen_noise,
}
FROZEN_FNV = {  # src/fixtures.rs:402-407
    "text": 0xD2AD5CB3D9F2AC83,
    "tabular": 0x8F132A1F79EC4511,
    "binary": 0xFE903199456D928D,
    "noise": 0xCDFD5FB185201167,
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

DIST_BUCKETS = (64, 512, 2048, 4096, 8192, 16384, 32768)


def features(d):
    n = len(d)
    # ---- trigram repeat-distance pass: direct-addressed last-occurrence table
    # 2^24 buckets == every possible 3-byte key exactly once -> no collisions.
    tbl = array.array("i", bytes(4 * (1 << 24)))
    counts = [0] * (len(DIST_BUCKETS) + 2)  # 7 buckets + beyond + first
    last = 0
    d1 = d[1:]
    d2 = d[2:]
    i = 0
    for x, y, z in zip(d, d1, d2):
        k = (x << 16) | (y << 8) | z
        p = tbl[k]
        if p:
            dist = i - (p - 1)
            if dist <= DIST_BUCKETS[0]:
                counts[0] += 1
            elif dist <= DIST_BUCKETS[1]:
                counts[1] += 1
            elif dist <= DIST_BUCKETS[2]:
                counts[2] += 1
            elif dist <= DIST_BUCKETS[3]:
                counts[3] += 1
            elif dist <= DIST_BUCKETS[4]:
                counts[4] += 1
            elif dist <= DIST_BUCKETS[5]:
                counts[5] += 1
            elif dist <= DIST_BUCKETS[6]:
                counts[6] += 1
            else:
                counts[7] += 1
        else:
            counts[8] += 1
        tbl[k] = i + 1
        i += 1
    del tbl
    ntri = n - 2
    pct = [100.0 * c / ntri for c in counts]
    near3 = sum(counts[:4])      # <= 4096 (= greedy's fixed guard)
    far3 = counts[4] + counts[5] + counts[6]  # 4097..32768
    beyond = counts[7]
    matched = near3 + far3       # <= 32768, inside the DEFLATE window

    # ---- trigram distribution entropy (second pass over a count table)
    cnt = array.array("i", bytes(4 * (1 << 24)))
    for x, y, z in zip(d, d1, d2):
        cnt[(x << 16) | (y << 8) | z] += 1
    distinct = (1 << 24) - cnt.count(0)
    h3 = 0.0
    for c in cnt:
        if c:
            p = c / ntri
            h3 -= p * math.log2(p)
    del cnt

    # ---- byte-value stats (C-speed counting over full payload)
    counts8 = [0] * 256
    for b in d:
        counts8[b] += 1
    h0 = 0.0
    for c in counts8:
        if c:
            p = c / n
            h0 -= p * math.log2(p)
    nb = sum(1 for c in counts8 if c)
    top8 = 100.0 * sum(sorted(counts8, reverse=True)[:8]) / n

    # ---- stride autocorrelation, C-speed via bigint XOR + zero-byte count
    stride = {}
    for t in (4, 8, 16, 32):
        if n <= t:
            stride[t] = 0.0
            continue
        x = int.from_bytes(d[:-t], "big") ^ int.from_bytes(d[t:], "big")
        z = x.to_bytes(n - t, "big").count(0)
        stride[t] = 100.0 * z / (n - t)

    return {
        "bytes": n,
        "near3": pct[0:4],
        "far3": pct[4:7],
        "beyond": pct[7],
        "first": pct[8],
        "near3_tot": 100.0 * near3 / ntri,
        "far3_tot": 100.0 * far3 / ntri,
        "far_a": 100.0 * counts[4] / ntri,        # 4097-8192: lazy fixed guard
        "mid_tot": 100.0 * (counts[5] + counts[6]) / ntri,  # 8193-32768
        "matched": 100.0 * matched / ntri,
        "h3": h3,
        "h3_per_b": h3 / 3.0,
        "distinct_tri": distinct,
        "tri_density": distinct / ntri,
        "h0": h0,
        "nbytes": nb,
        "top8": top8,
        "stride": stride,
        "smax": max(stride.values()),
    }


# ---------------------------------------------------------------------------
# Shape-class assignment (the atlas's working taxonomy; see at-corpus-classes.md)
# ---------------------------------------------------------------------------

SHAPE_CLASSES = {
    "text (fixture)": "prose-text",
    "tabular (fixture)": "precision-columns",
    "binary (fixture)": "native-binary",
    "noise (fixture)": "incompressible",
    "dickens": "prose-text",
    "webster": "prose-text",
    "reymont": "prose-text",
    "mozilla": "native-binary",
    "samba": "archive-mix",
    "silesia.tar": "archive-mix",
    "software.archive": "archive-mix",
    "xml": "cadence-markup",
    "logs.txt": "local-log",
    "nci": "precision-columns",
    "osdb": "precision-columns",
    "ooffice": "already-compressed",
    "sao": "scientific-stride",
    "mr": "scientific-stride",
    "x-ray": "scientific-stride",
}


def row(name, src, f):
    vals = [
        "`{}`".format(name), src, "{:,}".format(f["bytes"]),
        *["{:.2f}".format(v) for v in f["near3"]],
        *["{:.2f}".format(v) for v in f["far3"]],
        "{:.3f}".format(f["beyond"]), "{:.1f}".format(f["first"]),
        "{:.3f}".format(f["h3"]), "{:.2f}".format(f["h3_per_b"]),
        "{:.2f}".format(f["tri_density"] * 100),
        "{:.2f}".format(f["h0"]), "{}".format(f["nbytes"]),
        "{:.1f}".format(f["top8"]),
        *["{:.2f}".format(f["stride"][t]) for t in (4, 8, 16, 32)],
        "{:.2f}".format(max(f["stride"].values())),
    ]
    assert len(vals) == 23, "cell count {} != 23".format(len(vals))
    return "| " + " | ".join(vals) + " |\n"


def main():
    results = []
    head = (
        "| member | source | trigram positions "
        "| dist <=64 | 65-512 | 513-2048 | 2049-4096 (%) "
        "| 4097-8192 | 8193-16384 | 16385-32768 (%) "
        "| >32768 (%) | first (%) | H3 bt/tri | H3/3 | trigram kinds % "
        "| H0 bb | byte vals | top-8 mass % "
        "| acorr4 % | acorr8 % | acorr16 % | acorr32 % | stride max % |"
    )
    print("### Panel 1 - full feature table\n")
    print(head)
    print("|---|" + "---:|" * 22)

    for name, gen in GENERATORS.items():
        d = gen(LEN)
        got = fnv1a(d)
        want = FROZEN_FNV[name]
        if got != want:
            sys.exit("fixture replica mismatch: " + name)
        src = "fixture replica 1 MiB, frozen FNV OK"
        results.append((name + " (fixture)", features(d), src))

    corpus = []
    silesia_dir = "/tmp/atlas-silesia/silesia"
    for f in sorted(os.listdir(silesia_dir)):
        corpus.append(("/tmp/atlas-silesia/silesia/" + f, f))
    for f in ("logs.txt", "software.archive", "silesia.tar"):
        corpus.append(("/Users/jackdanger/www/gzippy/benchmark_data/" + f, f))

    for path, name in corpus:
        with open(path, "rb") as fh:
            d = fh.read(CAP)
        src = "{:,} of {:,} B".format(len(d), os.path.getsize(path))
        results.append((name, features(d), src))

    for name, f, src in results:
        print(row(name, src, f), end="")

    # Panel 2: derived splits over the windows the len-3 policies speak in:
    # near <=4096 (greedy fixed guard, parse/greedy.rs:245), 4097-8192 (lazy
    # fixed guard), 8193-32768 (clean far), all <=32768 (matched), >window.
    print("\n| member | shape class | near<=4096 % | 4097-8192 % | 8193-32768 % "
          "| matched<=32768 % | >window % | acorr_max % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, f, _ in results:
        print("| `{}` | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.1f} |".format(
            name, SHAPE_CLASSES[name], f["near3_tot"], f["far_a"], f["mid_tot"],
            f["matched"], f["beyond"], f["smax"]))

    # Panel 3: per-shape-class means over the members of each class.
    print("\n| shape class | members | near mean % | far mean % | matched mean % "
          "| H3/3 mean | H0 mean | top-8 mean % | acorr max mean % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    groups = {}
    for name, f, _ in results:
        groups.setdefault(SHAPE_CLASSES[name], []).append((name, f))
    for cls in sorted(groups):
        ms = groups[cls]
        def m(key):
            return sum(f[key] for _, f in ms) / len(ms)
        names = ", ".join("`{}`".format(n.split(" (")[0]) for n, _ in ms)
        print("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.1f} | {:.1f} |".format(
            cls, names, m("near3_tot"), m("far3_tot"), m("matched"), m("h3_per_b"),
            m("h0"), m("top8"), m("smax")))


if __name__ == "__main__":
    main()

```

## Appendix B — exact commands

```
mkdir -p /tmp/atlas-silesia
tar -xJf /Users/jackdanger/www/gzippy/benchmark_data/silesia.tar.xz -C /tmp/atlas-silesia
cd /Users/jackdanger/www/gzippy/.worktrees/brain/docs/board/attack/atlas
python3 at-corpus-features.py > /tmp/atlas-feature-panel.md
```
Fixture fidelity check: the script exits non-zero on any FNV-1a pin mismatch
against `fixtures.rs:402-407`; this run printed "frozen FNV OK" ×4.
