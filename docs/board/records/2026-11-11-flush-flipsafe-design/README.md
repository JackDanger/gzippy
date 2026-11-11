# Design record: the fp-settle flip-safe shape for `near-opt-parallel-flush` — 2026-11-11

Branch `lever/flush-piplining-flipsafe` off `origin/main` @ `30252fba`
(implementation commit `142c4c77`). Worktree `.review-tmp/wA-flipfix`,
isolated `--target-dir`s per arm. No pushes.

Named lever: **`near-opt-parallel-flush-flipsafe`** (new Cargo feature, DEFAULT
OFF, no change to the existing `near-opt-parallel-flush` arm — that feature's
guard, dispatch, and bytes are exactly as shipped on 30252fba).

## 1. The mechanism (the freshness chain, 10 lines)

1. The near-opt parser splits a chunk into internal blocks: FILL (per-position
   match caching + split statistics) → `optimize_and_flush` (the DP +
   Huffman refinement + emission) → next block's FILL.
2. `optimize_and_flush` returns `used_only_literals` — true iff the block's
   (a) all-literals candidate was strictly the cheapest of the candidates.
3. The serial shape feeds that boolean BACK into the block loop: the next
   block's `min_len` (`prev_block_used_only_literals` at
   `near_optimal.rs`, `let min_len = if … MAX_MATCH_LEN + 1 else
   calculate_min_match_len(...)`).
4. `min_len` gates the split-heuristic's OBSERVATION stream: with `min_len =
   MAX_MATCH_LEN + 1` every position's `best_len` (< 259) counts as
   `observe_literal`, so block k+1's `split_stats`, its block-BOUNDARY
   decisions, and (via `save_stats`) block k+2's `prev_observations` differ.
5. Under `near-opt-parallel-flush`, block k's flush runs on a pool worker
   while block k+1's fill runs on the chunk thread; the fill needs the
   verdict BEFORE block k+1's `min_len` is computed — microseconds after
   dispatching block k — so the dispatch arm passes a literal `false` and
   breaks the freshness edge (the one carrier the carriers audit omitted).
6. On a block that goes only-literals, the parallel fill derives a DIFFERENT
   observation stream than the serial fill would: one chunk diverges from the
   serial stream (+15 B in16m L9 -p4, −49 B L11 -p4 — receipt
   `docs/board/records/2026-11-10-flush-rss-probe/README.md`, reproduced on
   30252fba this session: OFF/`327b083588…` vs ON/`ff98ecd595…` and
   `1d4aed15…` vs `86440d79…`).
7. The shipped stale-flag guard latches AFTER such a chunk (every later chunk
   serial), but the flipped chunk's bytes stay diverged — that quarantine is
   the 2026-11-10 record's PROMOTION BLOCKER (#3-ranked precondition:
   serialize the edge, or stall `min_len` on a possible flip).
8. Measured flip shapes on the staged corpus (envelope instrument, 52
   members × L9-12 `-p4`, this branch's probe/): 1,885 flip blocks spread over
   `matched(cache slots − block len) ∈ [1 .. 125_031]` and
   `block len ∈ [5_000 .. 300_126]`:
   - SH1 (in16m/silesia/reymont): skimpy blocks, 1-108 match entries,
     only-literals wins by 19-36 bits;
   - SH2 (weights.safetensors, mozilla, sil40): the dynamic path is WORSE
     than all-literals by KILOBITS on far-offset/short matches over skewed
     alphabets — weights alone: 547 flipping blocks in one process,
     matched entries up to 125,031.
9. Byte-consequence is measured, not inferred: plain-feature ON ≠ OFF on
   weights/mozilla/sil40 (three-way sha check in this record), ON == OFF on
   xml/photo/dickens/ecoli (there the poison never lands).
10. Therefore BYTE-CONSISTENCY cannot come from a fill-side shape predicate
    (no completeness is available) — the design must take the REAL verdict.

## 2. The design (fp-settle)

**Invariant preserved:** flush block k+1's entering state is a pure function
of fill-produced data and the flush chain's own carriers. The one
non-fill-produced input it used to guess (`verdict(k)` → `min_len(k+1)`) is
now taken from the flush's OWN exact semantics — never guessed.

**The wait is truncated where PROVEN safe** by the (d) decision's own shape:

    verdict = true  ⟹  only_lits_cost < best_true_cost        (nested `if
                        only_lits_cost < static_cost` inside
                        `min(only, static) < best_true_cost`)
    best_true_cost  =  min over refinement passes ≤ FIRST pass's
                        true cost (F1)                        (the loop sets
                        `best_true_cost` on pass 1 and only lowers it)

Necessarily: `verdict = true ⟹ only_lits_cost < F1`. Contrapositive:
**`only_lits_cost ≥ F1 ⟹ used_only_literals == false, guaranteed`.**

Structure:

* the flush worker publishes `(F1, only_lits)` right after the first
  refinement pass (new cfg'd `first_pass_observer` tail-parameter on
  `optimize_and_flush`; the serial arms pass `None` — zero change);
* the fill thread's `settle_poison` parks on the per-chunk settle mailbox at
  the top of the block loop (before `min_len`), and:
    1. nothing dispatched yet → `false` (the serial arm's only value there);
    2. `(F1, only_lits)` present and `only_lits ≥ F1` → `false` PROVEN, the
       flush's remaining passes keep running off-thread (the common case on
       match-dense blocks — measured `only_lits/F1` ratios of 1.2-8);
    3. else → wait for the REAL verdict (the rare negative-margin block).
* the dispatched fragment itself is unchanged (bit-exact repack as the
  feature already shipped); the stale-flag LATCH is disabled under flipsafe
  because the verdict is exact — a flip is no longer a divergence event and
  the latch would serialize every later chunk. (First measured with the
  latch in: settles = 24 of 1,023 flushes on weights — everything after
  chunk 1 went serial.) The plain-feature build keeps the shipped guard,
  including the one-shot stderr note.

**Why not the first draft's envelope predicate:** the first implementation
gated the wait on `(block_len ≤ 65_536, matched ≤ 112)`. The SH2 measurement
(then-row: every weights.safetensors flip outside the envelope, mozilla 7/11
outside) killed it — an incomplete catch builds a THIRD stream: measured
`SAFE ≠ OFF ≠ ON` on mozilla/sil40 L9 (and weights SAFE == ON because the
envelope fired zero times there). The envelope consts were deleted; the
fp-soundness is per-block, per-corpus, by construction.

**Why the wait cannot deadlock** (argued in-code): the settle parks only on
the settle mailbox; its writers (flush workers) park only on the flush→flush
exit-cost mailbox (fed by other workers) and never on the fill thread; the
queue is closed only after every dispatched slot is filled.

## 3. The census matrix (the deliverable bar)

Cells ran across the branch at TWO revisions during the run (the restructurings
in between were comment/doc text only; every sha is a compile of committed
logic):

* `OFF` = `cargo build --release --target-dir tgt-flush-off`
  (default features — the parallel-flush-OFF baseline; binary sha256
  `a9c22a7dfc2bf8e155ec199c09d8097df560c37a7406f1e700d63b4429ee2aff`;
  byte-identical to shipped main: in16m L9 -p4 = 3,740,910 B /
  `327b083588efca21…` = the #448 anchor)
* `SAFE` = `cargo build --release --target-dir tgt-flipsafe-census
  --features near-opt-parallel-flush,near-opt-parallel-flush-flipsafe,anatomy-counters`
  (binary sha256
  `02052a77d07ac1bfefdfe3fd7454f61902d6d18bcd836a6d405d2724609e8e47`;
  anatomy-counters rides ONLY as the settle-witness stderr line — the stdout
  bytes compared are the flipsafe arm alone)

Cell: same input, same level, same `-p`; compress `-c` with each binary;
sha256 both; `OK` iff identical. 52 locally staged members × L9..L12 ×
{p2, p4} = 416 cells × 832 encodes. Runner + per-cell TSVs:
`probe/census_flipfix.sh` → `probe/census/`.

Result (verbatim from `probe/census/census.tsv`; all 416 per-cell rows
committed under `probe/census/*_L*.tsv`, columns
tag/level/-p/size_OFF/sha_OFF/size_SAFE/sha_SAFE/verdict/settles-only-literals):

```
cells=416   OK=416   DIFF=0
settle boundaries resolved: 59,769 ; only-literals blocks in the SAFE
(serial-decision) space: 1,898 ; 52/52 members at 8/8 cells OK
```

- the SAFE arm's flip tally EQUALS the serial arm's on the cells checked
  both ways: in16m L11 -p4 OFF = 3 only-lits over 366 flushes, SAFE = 3/366;
  mozilla L9 -p4 = 11/2,047 settles — the exactness corroboration (the plain
  ON Arm's diverged chunk measures a different sequence, 11 → this is
  evidence the SAFE stream truly IS the serial stream, not a stream that
  merely happens to tie on total bytes).
- p2 rows (the L9-12 route at ≤2 threads is the chunked Lazy2-600 #458 arm —
  no near-opt flush): both arms reproduce the shipped p2 anchors (in16m L12
  -p2 = 3,816,930 / `edc698cc16c8…` = the wave-B A1 anchor), settles=0.
- heavy-cell spot checks (sizes match the wave-B census record); e.g.
  logs.txt L12 -p4 = 8,450,419 B `d7716a42156c…`, software.archive L12
  -p4 = 4,299,945 `160b7a5ca2f5…`, weights.safetensors L12 -p4 =
  210,238,025 `7cf4f14d414d…`.
- members: 12 silesia tar members, logs.txt + software.archive, the 8+8 even
  slices of both, the 16 split-corpus members, the 5 wall-census stand-ins
  (1byte.bin, binary26m.bin, in16m.bin, logs-card26m.bin, text26m.bin),
  alice.txt — the "34-member-set stand-in = 51 locally staged members" bar
  is met with one MORE member staged (52 unique files).

## 4. Wall (interleaved medians, hypothesis-tier — unfrozen laptop box)

`probe/wall_film.sh` (strict alternation, one warmup per arm uncounted,
`date +%s%N` timing; data films in `probe/film_*.tsv`):

**Film 1 (in16m.bin, 16,085,866 B, box load ~7/14; medians of 10/arm):**

| cell | OFF med | SAFE med | SAFE/OFF |
|---|---:|---:|---:|
| L9 -p4 | 357.0 ms | 326.5 ms | 0.9146 (**−8.54%**) |
| L10 -p4 | 338.0 ms | 313.0 ms | 0.9260 (**−7.40%**) |
| L11 -p4 | 460.5 ms | 397.0 ms | 0.8621 (**−13.79%**) |
| L12 -p4 | 562.5 ms | 479.5 ms | 0.8524 (**−14.76%**) |
| sil40 L9 -p4 | 582.0 ms | 525.5 ms | 0.9029 (**−9.71%**) |
| weights.safetensors L11 -p4 | 2330.0 ms | 2320.5 ms | 0.9959 (**−0.41%**) |

**Film 2 — the prize coordinates re-filmed at medians of 14 (L9 under a
spiked box load ~23; L11 load ~18):**

| cell | OFF med (spread) | SAFE med (spread) | SAFE/OFF |
|---|---|---|---:|
| L9 -p4 | 362.0 (360–364) | 334.0 (331–337) | 0.9227 (**−7.73%**) |
| L11 -p4 | 468.0 (466–490) | 405.5 (402–446) | 0.8665 (**−13.35%**) |

Both films agree on the shape: the fp-settle recovers most of the
2026-11-10 probe's plain-flush win (−10.4%/−13.3%) — film 1 at L11 even
exceeds it (−13.79%) within the noise band. The flip-HEAVY class
(weights.safetensors: 555 flips/1,023 flushes at L9) sits at a tie —
its settles resolve at full-flush depth, i.e. the design's floor is the
serial wall, and the local film prices the residual tax at ≈ 0 (a
tie-band number, not a regression: the per-block waiting replaces the
flush's serial share one-for-one).

Mechanism note (for the frozen-box step): the wait is `max(0, F1
− elapsed-since-dispatch)` at each block boundary; L12's 10-pass flushes
make F1 ≈ 10-15% of the flush (the win −14.76%); L9's 4-pass L11-knobs
flushes make F1 ≈ 25-40% (the win −8..−9%); the 2-pass L10 flushes make
F1 ≈ 50% (the win −7.4%). The measured ordering matches the predicted
recovery ordering monotonically.

## 5. tests

* `tests/l9_t4_chunk_cost_probe.rs::flipsafe_pool_stream_is_the_serial_stream`
  (new, both-features build): whole-stream `compress_with_threads(L9, T4)`
  pool arm vs the forced-serial arm on the flip record's in16m recipe —
  byte-equal, settles witness > 0, guard NOT latched, roundtrip decode ==
  input; plus a deterministic fp-negative synthetic (64 KiB incompressible
  noise tail forces at least one full-depth settle).
* the existing `parallel_flush_probes` identity suite passes under the
  both-features build (the plain-feature contract stays green).
* `cargo build --release` (no features) byte-identical to the shipped
  default build; the flipsafe-only build (no flush feature) compiles the
  whole feature set to nothing.

## 6. Verdict: **SHIP (as an adjudication candidate lever)**

* **The bar is met exactly**: 416/416 census cells OK — the SAFE arm's
  stream is byte-identical to the parallel-flush-OFF baseline on 52
  members × L9-12 × -p2/-p4 (832 encodes, zero sha deltas), including every
  flip-shaped class the corpus stages. The 2026-11-10 promotion blocker
  ("either serialize the freshness-chain edge, or stall the next block's
  min_len on a possible flip") is satisfied by the second route's
  generalization: the min_len always settles on the REAL verdict, waiting
  only where the wait is proven safe to truncate.
* **The wall prize survives, minus a bounded haircut**: at the 2026-11-10
  probe's own coordinates the optional feature measured L9 -p4 −10.4% /
  L11 −p4 −13.3%; the flip-safe build films at −7.7..−8.5% (L9) and
  −13.4..−13.8% (L11), plus L10 −7.4%, L12 −p4 −14.8%, sil40 −9.7%, and a
  tie on the flip-heavy weights class. Well beyond the 5% local-tie band
  except the weights tie (≈0.4%, explicitly a floor, not a regression).
* **Scope of this record**: laptop-tier (charter: hypothesis-tier walls,
  frozen box still adjudicates the promotion; the size bar holds with zero
  size deltas by construction — the streams ARE the serial streams, so
  every rival comparison the shipped build wins stays won).
* Risks carried: none NEW for byte correctness (the fp-soundness is
  per-block/per-corpus by construction, and the census + unit tests + a
  3-run determinism film on the flip-heaviest member are the falsifiers);
  the wall claim needs the frozen box (`make lever` / board-wall) before it
  could promote the default-off feature. The envelope-predicate history is
  retained in `probe/envelope/` precisely to *document why* the fp-settle
  replaced it (an incomplete shape predicate manufactured a third stream).

## 7. What could still kill this (and the pre-planned falsifier)

1. A corpus where the ONLY-LITERALS verdict is systematically on the
   negative-margin side at near-100% rate AND the fp-depiction underestimates
   F1 → every settle goes full-depth, the feature degenerates to the serial
   wall (the weights measurement — tie, not a regress — says even the
   degenerate case does not LOSE).
2. A missed carrier beyond the two the carriers audit covers —
   the census (832 encodes over mixed chunk grids) is its falsifier, zero
   exceptions measured.
3. The frozen box moving the wall bands — fp-soundness is not wall-sensitive.

## Receipt index

* `probe/census_flipfix.sh` / `probe/census/*.tsv` — the matrix
* `probe/wall_film.sh` / `probe/film_*.tsv` — the wall film
* `probe/envelope_run.sh` / `probe/envelope/*.stat` — the flip-shape
  instrument (FLUSHSTAT throwaway, reverted in the committed tree)
* `probe/repro_flip.sh` — the 2026-11-10 receipt reproduction (+15/−49)
