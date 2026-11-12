# 2026-11-12 — FIRST full Linux (aarch64) per-label size census — termag `jdanger-rust-builds`

The campaign's own guarded instrument (`scripts/campaign/board-size.sh tune` → fulcrum `sizecensus`, G2 all-rivals declared, W2 vanilla build, W3 identified binary: `fulcrum=483e413b2e3f`, `gzippy=e2c73d44…dirty=0`) run on the user's termag Linux arm64 box — the first time the per-label board ran off-Mac. Rivals: system `gzip 1.12`, `pigz 2.8`, `libdeflate-gzip 1.19`, vendored `igzip 2.31.1` (built in-PR from `vendor/isa-l`). Corpus: the pinned TUNE set (11 members; shas stamped in census.json), staged from the canonical `squishy.jackdanger.com` registry + sha-locked `silesia.tar` (pin `028bd002…`) sil40 derivation. gzippy at `main e2c73d44` (everything of the Nov-11 arc).

## Headline (1,122 measured cells; per-rival verdict table)

| rival | measured | wins (<−0.1%) | WORSE (>+0.1%) | worst |
|---|---:|---:|---:|---|
| gzip 1.12 | 297 | 286 | **1** | +0.21% (`minjs.min.js` L5-T1 — the standing known cell) |
| pigz 2.8 | 330 | 296 | **26** | +17.86% |
| libdeflate 1.19 | 396 | 188 | **32** | +17.58% |
| igzip 2.31.1 | 99 | 92 | **0** | — (rival only accepts its own levels; 297 absent by `goal::rival_accepts_level`, excluded per design) |

`ours == libdeflate` byte-exactness reproduced ON THE BOX at every T1 port cell (e.g. aozora L10-T1 `3,781,386 == 3,781,386`) — the "port == libdeflate byte-exact" claim holds arch-invariant on arm64-Linux.

## TWO failing families (59/1,122 = 5.3%)

**Family 1 — L10-12 × T2 (the wave-9 `-p2` route-flip band): ~46 cells vs libdeflate AND pigz at +0.2..+17.86%** (data.csv +17.6/17.9%, aozora +5.5-6%, dickens +3.6-8%, minjs/armexe/tool/engine/symbols +2.3-4.5%). The wave-9 flip (#458) was re-priced ONLY against `min(gzip-9, pigz-9)` — versus **libdeflate's own levels 10-12 the flip's chunked Lazy2-600 stream loses its size bar on many members** (and vs `pigz -11` too on data.csv). gzip's column is CLEAN at 10-12 (only a pre-existing pigz hairline): the flip kept the gzip/pigz-9 bar, broke vs the strongest rival. Exhaustive lists: `summary.txt` + `census.tsv` (every failing row verbatim). This is the next wave's top code target: the flip's rationale (wall x8 vs near-opt at T2) must be re-derived against a T2 route whose SIZE holds vs libdeflate, or the band renegotiated per charter with these exact numbers.

**Family 2 — L11-vs-pigz hairlines: 13 cells at +0.18..+0.95%** (dickens/aozora/data.json/parquet/minjs/armexe/tool/engine/symbols — mostly T1/T4 with some T2). Matches the standing L11-T4 knife-edge class (`plan-2026-10-29-sweep` W2 rejects) — never re-measured since 2026-09; still beyond the ±0.1% band.

Everything else: 1,063 cells inside-or-below the tie band, including every T1 port cell and the entire win column at L1-9.

## Notes

- The `THREAD SHORTCUT VOIDED` prints during the run are the instrument's own mechanism: post-#458, L9-12 T2 ≠ T4 by design (distinct stream class, renegotiated #458/#470); each such cell was measured individually rather than trusting the T2==T4 shortcut — correctness preserved, wall cost of the census increased as expected.
- Raw receipts committed beside this README: `census.tsv` (1,123 lines), `census.json`, `summary.txt` (fulcrum's own failing-cell report), `meta.json` (binary shas + templates).
- The wall-axis of this census (`board-wall.sh tune`) is running under an 18h `fulcrum freeze` — filed separately when it banks.
