# Probe: flush-scheduling (`near-opt-parallel-flush`) + RSS (-p grid) — 2026-11-10

Branch `probe/flush-and-rss` off origin/main @ `56e9cc0b` (no source changes; probe is
a build-and-measure run). Worktree `.review-tmp/w8-flush/probe-wt`.

Binaries (release, this worktree):
- `gzippy-base` sha256 `571af294159c9d4921f325ba0031333366afa899` (default features)
- `gzippy-feat` sha256 `ea0e4d3a47f16204a7bcd3b076ceabed2905b7b3` (`--features near-opt-parallel-flush`)

Corpus `in16m.bin` = 16,085,866 B, sha256 `1010fa2f787d0a92…`
(`cat corpus/silesia_blocks/*.bin ×2 + test_data/text-1MB.txt ×2`). 14-core macOS host,
all runs quiet, interleaved arm order, `-k -f` on file-mode runs, mtime zeroed by gzippy.

## 1. `near-opt-parallel-flush` pricing at the L9/L11-T>1 coordinate (-p4)

Walls (medians of 5, strict alternation; per-arm spreads ≤1.8%):

| cell | base med | feat med | feat delta | feat latches (guard tripped) |
|---|---|---|---|---|
| L9 -p4 | 0.375 s (0.371–0.376) | 0.336 s (0.333–0.339) | −10.4% wall | 5/5 (+ warmup) |
| L11 -p4 | 0.513 s (0.506–0.514) | 0.445 s (0.439–0.453) | −13.3% wall | 5/5 (+ warmup) |

Byte-identity (stdout mode, `-c`, headers verified byte-equal between arms):

| cell | base | feat | identical? |
|---|---|---|---|
| L9 -p4 | 3,740,910 B, sha `327b083588efca21…` (= the #448 shipped anchor) | 3,740,925 B (+15), sha `ff98ecd595a0522a…` | NO |
| L11 -p4 | 3,731,478 B, sha `1d4aed15d917fd4c…` | 3,731,429 B (−49), sha `86440d79a137b183…` | NO |

The on-by-default anchor holds for the default binary: L9 -p4 = exactly 3,740,910 B
(file-mode adds the 10-byte `in16m.bin\0` FNAME → 3,740,920 B, sha `90221302…`).

Why identity fails on this corpus: every feature-on run tripped the process-wide
stale-flag guard (`src/compress/deflate/parse/near_optimal.rs:1025`):
`near-optimal parallel flush guard tripped (1 only-literals flushes in one chunk) —
falling back to the serial optimize_and_flush for the rest of this process`. The
parallel dispatch assumes `prev_block_used_only_literals == false`
(near_optimal.rs:1546 snapshot arm passes literal `false`); this corpus has a flip in
one chunk, so that chunk's bytes diverge from the serial shape (+15 B at L9 per the
documented quarantine semantics; later chunks re-serialize and the stream heals at
member boundaries). Roundtrips: python gzip + vendored gzip `-t` + gzippy `-t` pass on
all four outputs (L9 base/feat, L11 base/feat).

## Verdict (flush lever): FALSIFIED as a byte-identity-gated free lever — CLARIFIED as priced

The ci.yml M1 gate ("chunks_are_byte_identical_to_the_serial_stream") holds on its
pinned fixture but NOT on the campaign corpus at either priced coordinate: bytes change
(+15 B @L9, −49 B @L11), sizes do not match the shipped anchor stream, and the guard
latches on every process. Under the charter (valid gzip + per-label size vs
anchor-shape stream), that forfeits the identity gate regardless of the wall win.
The wall win is real and large (−10% to −13%, >> the 5% tie band and the ±1-2%
within-arm spread). Recommendation: keep the feature DEFAULT OFF; a promotion needs
a flip-safe design (serialize the fill→flush carriers only when a flip can occur) or
an explicit anchor-stream change adjudication — not a default-feature flip.

## 2. RSS for the #450 thread-aware grid (base binary, L2, stamped corpus)

Peak RSS via `/usr/bin/time -l` (maxRSS = ru_maxrss bytes; footprint =
`peak memory footprint`), medians with n=9 (IQR in parens):

| -p | maxRSS median (IQR) | footprint median | wall med |
|---|---|---|---|
| 4 | 36.0 MiB (34.3–37.4) | 19.6 MiB | ~0.04 s |
| 16 | 39.0 MiB (37.7–40.1) | 22.6 MiB | ~0.03 s |
| 28 | 38.6 MiB (36.7–39.0) | 22.3 MiB | ~0.03 s |

- The grid's more-smaller-chunks shape does NOT blow up RSS: `-p16`/`-p28` sit at
  ≈ +2.6–3.0 MiB maxRSS over `-p4` (~+8%) and plateau (p28 ≤ p16). For a 16 MB
  input the user-visible ceiling is ~40 MiB maxRSS (~2.5× input), ~23 MiB footprint.
- Individual maxRSS readings vary up to ±4 MiB between runs at fixed `-p` (spread
  spread above), so treat sub-3-MiB RSS deltas as noise; the `-p4 → -p16` step (+3.0 MiB
  median) is real but small; no -p-proportional growth.
- L2 output bytes depend on `-p` (file mode): -p4 = 4,235,409 B (`0dd7bf76…`);
  -p16 = -p28 = 4,236,782 B (`3973a005…`, byte-identical between them) — the
  thread-aware grid changes chunking (+1,373 B framing) and saturates by -p16.

## Receipt numbers for #450

Peak RSS on the stamped corpus at L2: `-p4` 36.0 MiB, `-p16` 39.0 MiB, `-p28`
38.6 MiB (medians of 9; footprint 19.6/22.6/22.3 MiB). The D2 ceiling claim holds:
more, smaller chunks in flight do not open RSS linearly with `-p`; the grid packs
in-flight work into a ~+3 MiB envelope at 4-7× the worker count.

## Unverified / caveats

- MaxRSS on macOS counts peak over the child lifetime including shared pages;
  run-to-run spread (±4 MiB) overlaps the p4↔p28 delta — deltas <3 MiB stay
  noise-tier (hypothesis-tier walls apply to the wall column).
- The CI identity test itself was not re-run (budget); the corpus-level identity
  measurement above is the adjudicating evidence either way.
- L11 = `-11` (near-opt more passes) as shipped; L12/-best, -p>4 walls and the
  L2/L6 coordinates are out of scope here.
- RSS/wall numbers are laptop-tier (charter: sub-5% wall deltas are ties; the
  flush lever's wall win is quoted as its ACTUAL delta, not a tier claim).
