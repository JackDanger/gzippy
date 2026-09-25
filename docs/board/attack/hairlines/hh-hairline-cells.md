# HH — the hairline cells: minjs L5 · data.sqlite L4 · weights L7-L9 · movie.mp4 L6

Attack memo, 2026-09-05, written read-only off `brain/2026-09-05-attack` @ `5900d17a`
(no cargo runs, no box access, no commits — the reader must re-derive the
STEP-0 pins before touching code). Sibling memos: `attack/l4l5/` (the L4/L5
cadence band and the `good_match` lowering), `attack/cadence/`, `attack/phase1/`
(state carry). This one owns the four cells that are sub-tenth-percent from the
tie band, the cells the tie cage refuses to help: a tied rival row has ZERO
tolerance both ways (`CLAUDE.md` working rules: "a tie PASSES but has ZERO
tolerance — one byte either way flips it, and clause 3 refuses that absolutely"),
so a hairline cell — failing by +0.0008% to +0.19% — is the only place a knob
can still be spent without either riding a tie or tripping clause 3
(promotion-rule.md §3: one pass→fail flip blocks the ship, absolute).

The class, from the frozen board (plan §2b, measured at stack tip `ee0c1d2c`,
1,320 cells, 30 failing):

| cell | margin | corpus | class |
|---|---|---|---|
| `minjs.min.js:L5` | +0.19 / +0.16 | TUNE (minified-code) | lazy-band parse price |
| `data.sqlite:L4` | +0.13 / +0.17 | GATE (columnar-db) | L4 sag |
| `weights.safetensors:L7-L9` | +0.00–0.02 | GATE (float tensors, ~83 MB) | near-incompressible trap |
| `movie.mp4:L6` | +0.0008 | TUNE (mp4 container) | seam-scale residue |

The brief's phrase "minjs L5 +0.19% vs libdeflate" conflicts with the code
record, and the split matters because it picks between two different
mechanisms:

* After the stack lands, shipped `-5` is `ldx` `deflate_compress_lazy` at
  (16,30) (`src/compress/ldx/compress.rs:128`, `compress_lazy.rs:11-12`) — a
  byte-for-byte transliteration of libdeflate `-5` (`libdeflate/lib/
  deflate_compress.c:3946-3948`; the vendored C lives at the repo top level
  in this checkout). A T1 row against libdeflate -5 MUST then be
  an exact tie (0.00), as it was pre-pivot on 22/22 files (2026-08 census,
  `/root/sizeboard-all-12fcd0ed`). A +0.19% loss can only be vs gzip/pigz
  (zlib's L5 config {good 8, lazy 16, nice 32, chain 32},
  `vendor/zlib-ng/deflate.c:120`), which is precisely what the pre-pivot
  census recorded — "gzip … aozora + minjs at L5-L7" (old plan trailing
  receipts) — and what plan §5 says re-opens post-pick-min ("the aozora/minjs
  L5-band cells were won by the winning arm, not by a synthesized arm").
* The alternative reading is that the +0.19 row is T4 vs libdeflate (a seam
  residue, like movie.mp4 L6's +0.0008%). That changes the mechanism (below)
  to a state-carry row and dissolves the knob question.

STEP 0 (one turn, no code): `fulcrum board size` to re-rank, `fulcrum why
<cell>` on each of the four (it prints position counts and names which of
its four layers it skipped), plus a per-cell pull from the frozen census
(`/root/www/gzippy-bench/campaign/size-all-ee0c1d2c/census.json`: failing
at T1, at T4, or both) on all four cells. Do not write a line of lever code
before that output exists.

---

## 1. What the crown engine actually contains, and which knob is even a
candidate for these rows

`src/compress/deflate/parse/ultra/` = the pure-Rust zopfli port, extended
ECT-grade (module doc, `ultra/mod.rs:1-6`):

* **Parse = iterated min-cost DP ("squeeze")**: greedy pass → `numiterations`
  ≤15 shortest-path runs, each priced by the previous run's symbol statistics,
  with statistics perturbed by a weighed-merge/regression-restart when stalled
  (`ultra/squeeze.rs`, port of `zopfli/src/zopfli/squeeze.c:446-519`;
  `RandomizeStatFreqs` multi-seed at :512-517). Per-position costs carry a
  full length/distance sub-choice `sublen[k]` — every match length k can take
  its own best distance (`squeeze.c:217-309`), which is the real
  full-Pareto property the L1-9 parsers' longest-match-only parsers lack.
* **Split = uncapped exact-cost recursion**: greedy `block_split` +
  `RecursiveSplitter` — 9-way narrowing plus the final brute-force window,
  memoized `(a,b)` costs, `auto_type` = best-of-{stored,fixed,dynamic} per
  candidate (`ultra/blocksplit.rs:272-424`; C at
  `zopfli/src/zopfli/blocksplitter.c:215-273`), then per-final-block
  re-squeeze with block-local prices and the ECT `twice`-mode feedback loop
  (`ultra/deflate.rs:487-577`).
* **All capped by an oracle**: the whole engine is pinned byte-for-byte to
  the vendored C zopfli (`ultra/oracle_tests.rs:141-150`).

What the L1-9 ladder already measured against this engine, on record:

1. **The full crown was measured and rejected for L1-L9**: 21-38x L9 wall,
   F1 output LARGER than the near-optimal chunked output (`level.rs:454-457`).
2. **The splitter transplanted alone onto the shipped parse measured
   -364/-268/+16 B** — "the crown's win is the PARSE, not the boundaries"
   (`level.rs:459-461`). Every boundary-side knob (uncap, recursive
   auto-type cost, re-split/re-squeeze) is therefore dead for the L1-9
   rows unless the parse also moves. Do not spend a session on splitter
   knobs for hairline cells; that measuring is done.
3. **The parse side has a measured surrogate**: the near-optimal chunk route
   (`params_parallel(11)`) captures 92-99% of crown-F15's win at ~1/30 cost,
   measured on 8 MiB fixtures (`level.rs:454-458`, `:467-490`).

So the only crown-class knob that can move an L5-L9 cell is **the parse
itself** — either a min-cost DP at that level (near-optimal-shaped) or the
knob levers the L1-9 parsers own (`max_search_depth`, `nice_match_length`,
lazy lookahead). The measured history of the near-optimal parser at mid
levels is coordinate-dependent and must be re-derived post-pivot: the
2026-08-11 down-ladder probe concluded "L6/L7 do not pay" BECAUSE no failing
L6/L7 cell existed then (`level.rs:482-487`) — hard stop #3 explicitly
forbids carrying that verdict to the post-pivot board where
`weights.safetensors:L7` IS a failing cell. The lever-level records to
consult before any of it: `grep -rnE 'FALSIF|REOPEN|PARK' src/ --include='*.rs'`
(charter hard stop #2) — the parked L4 Lazy record and the depth
non-monotonicity record are quoted below and are binding in *their measured
coordinates only*.

## 2. minjs.min.js:L5 — the minified-JS shape, and what (16,30) vs (12,30)
actually trades

minified JS is token-dense short text: 4-30-character identifier and operator
runs separated by single punctuation bytes, alphabet ~87-97 distinct bytes
(ASCII letters+digits+punct), match lengths rarely above ~30 (≈ `nice`), match
distances inside one 32 KiB window for long stretches.

The L5 engine post-pivot = libdeflate's lazy (16,30), with these two
deliberately-shallower-than-zlib characteristics:

* **primary chain walk 16** (`max_search_depth`) — zlib's L5 walks 32;
* **one-ahead lookahead at depth>>1 = 8** (`compress_lazy.rs:63-64, 233`) —
  zlib's `deflate_slow` runs its lookahead at FULL depth (32). This is the
  biggest single parse characteristic price against gzip/pigz on
  identifier-dense text: every deferred candidate is judged by a search that
  saw only ~1/4 of the chain material zlib saw.
* nice 30 vs 32 — second-order.

**The (12,30) computation the brief asks for.** Read as "trade depth down to
12 at L5", every effect is one of:

1. min_len interactions — fire only when it matters: `choose_min_match_len`
   clamps min_len to ≤7 whenever `max_search_depth < 16` (band 10≤d<16,
   `ldx/min_match.rs:51-58`). For minified JS the first-4-KiB alphabet is
   ≥80 distinct bytes with p≈1, so the unclamped min_len is already 3 and the
   clamp is a NO-OP on this file. (It is NOT a no-op on narrow-alphabet
   text — that interaction is the recorded L4 confound, `level.rs:533-535`.)
2. primary walk 12 vs 16, lookahead 6 vs 8 — every candidate chain that runs
   past node 12 now truncates four nodes earlier, on exactly the token-dense
   file where chains exceed 12 routinely. Expected direction: shorter or
   equal matches ⇒ output LARGER or equal; never smaller by mechanism.
   Nothing in the vendor's table goes shallower than this rung's nice either.
3. wall −25% of the chain sub-cost — the only thing (12,30) buys.

Verdict: **(12,30) cannot close +0.19%** — it optimizes the wrong axis; it is
a wall lever, and the "smaller = dangerous next to the tie cage at L3-L9
rows" warning in the brief is exactly right: ANY byte change at L5 unties the
22-file libdeflate-5 byte-tie band, and a shallower walk is not signed
smaller. Record (12,30) closed-by-analysis unless the STEP 0 pin reveals the
real failing rival parses differently than assumed.

**The knob that CAN move minjs L5.** The class's pre-pivot cells were held
by pick-min over multiple arms, and plan §5's own census question names the
zlib-style arm as the reproduction candidate: `apply_zlib_t1_search_knobs`
gives L5 `good_match=8, depth=32` (`level.rs:684-700`, zlib-ng heritage
quartering; the code is in THIS tree and live behind
`level_uses_t1_zlib_pick_min` = `5..=7`, `deflate/mod.rs:386-388`,
`deflate_one_shot_t1_zlib_pick_min` :463-494). Post-pick-min that arm
vanishes with the pick-min, and plan §5 already prescribes the revival path:
"If even the single-arm cannot hold them, L5 stays an exception at T1 like
L1 — same shape, no new rules" — but before that, at one encode, the honest
order is:

1. **Free probe, zero code**: build with the default-off measurement feature
   (`--features ladder-tune`) and `GZIPPY_LADDER=lazy:32:30` — one env layer
   that exists only in `+INSTRUMENTED` builds (`level.rs:644-677`), sweep
   the 11 TUNE files at L5 (minjs is one of them, freely usable) at (32,30)
   and (35,65). This prices depth 16→32/35 WITHOUT good_match and without a
   code change.
2. **The code-level card**: port `good_match` quartering into the ldx
   matchfinder (`hc_matchfinder.rs` has no such parameter today — the legacy
   engine's `matchfinder/hc.rs:271,296` carries it; #363's PR is exactly this
   port for L6/L7 at (128,65,8)/(256,130,32)) and route L5 onto it with the
   zlib-L5 rung values (good 8, depth 32) — the "#363 treatment" for the
   minjs band, named verbatim in plan §5.
3. *(12,30) is not a step on this road; it is a step off it.*

Cost model: chain-walk work scales ~linearly in depth over the walking
positions: 16→32 doubles it; the L2 x4-depth record measured 2.73x vs
libdeflate / 1.39x vs pigz wall at T4 on a match-dense file (`level.rs:521-523`)
— i.e. L5 depth 32 lands well inside T4's post-stack slack, but the T1 wall
row is the one to watch (`fulcrum try <ref> --threads 1,4` before any census
claim; charter hard stop #10: only `fulcrum try` promotes).

Falsifiers, cheapest first: `GZIPPY_LADDER` sweep (size only, TUNE) →
`scripts/campaign/tie-guard.sh <ref>` (the T1 tie subset, 22-file enum) →
`scripts/campaign/board-size.sh tune` → `fulcrum try <ref> --levels 5
--threads 1,4` → ledger append via
`cargo run --release --example fingerprint_tool -- ledger`.

Death modes (each one the L2-min-3 analog where the class matches):

* **D1 — tie-cage flip, not a margin miss.** Deeper depth is NOT monotone in
  output size: a longer match at i displaces a better one at i+k — measured
  at engine.wasm L8 where x2, x3, x4 ALL flipped the cell the same direction,
  so no multiplier rescues a flip (`level.rs:499-511`), and the same lever
  family (zlib good_match) previously closed 31 and FLIPPED 17 (data.csv L2
  1.0000→1.0431, `CLAUDE.md` tie-cage paragraph). minjs-only fitting hides
  this: the file is TUNE; the flips live in GATE-class text. Mitigation is
  structural: full-corpus census, never a sample; flip = NO-SHIP, no
  re-measure.
* **D2 — synthetic-first fitting.** The L2 min-3 lane died
  synthetic-tabular-first; the analog here is fitting (32,30) or
  (8,32)-vs-(16,30) on minjs alone and landing (8,32) as "L5's new knobs".
  The `_contract` in `corpus_split.json` (F4 rule in its header) exists
  because a parameter was once fitted to the one file blocking a gate.
  Sweep on TUNE; promotion judged on GATE only (plan Phase-3 rules).
* **D3 — per-label drift.** "At level N we beat their level N" — the lever
  must publish minjs L5 wins while NOT regressing L5 elsewhere; adopting the
  L6 rung at L5 is legal effort-wise (effort rises with level —
  `compress_lazy.rs:486-511`'s invariant test stays green), but if the
  census shows L5 closing minjs while opening any other L5 cell, the lever
  is clause-3 dead regardless of net mass. Do NOT split-arm L5
  (`level_uses_ldx`'s history: "the port ships as a RANGE, not a
  checkerboard" — `deflate/mod.rs:826-833`) without the same range rule.

## 3. data.sqlite:L4 and weights.safetensors:L7-L9 — the columnar /
tensor rows

Both rows are the same content family in the corpus taxonomy:
`data.sqlite` = columnar-db (SQLite page image: long regular runs, row-wise
4-byte/8-byte periodicity), `weights.safetensors` = float32 tensor stream —
near-incompressible mantissa noise on top of a rigid 4-byte cadence. The
vendor approaches to sparse/periodic columnar data, in attack order:

* **zopfli's iterative splitter** (the vendor's only content-adaptive
  boundary finder): greedy bisection on auto-type cost, uncapped, >10-symbol
  floor (`squeeze`-independent, `blocksplitter.c:215-273`) — but it prices
  boundaries on a FIXED parse; it cannot recover the 4-byte cadence as
  matches, only find blocks where the same parse is cheaper split. Measured
  transplant onto the shipped parse: -364/-268/+16 B — dead for hairline
  margins (§1).
* **libdeflate's lazy2** (the L8/L9 rung, two positions of lookahead at
  depth>>2) — deeper work, not a periodicity detector. At these rows its
  family is measured the WRONG WAY: the Lazy→Lazy2 / strategy-step family
  made `weights.safetensors` STRICTLY WORSE ("on
  near-incompressible float tensors Lazy defers matches and emits more
  literals", +31,651 B reproduced at L4 compose, `level.rs:561-584`) — the
  clause-3 flip that killed the L4-family extension to L5-L7 (⛔ "L4 IS THE
  MAXIMUM CLEAN SUBSET", `level.rs:569-584`).
* **The cost-model parse (near-optimal)** is the one measured mechanism that
  helps tensors: on the weights-like float32 fixture the chunked near-opt
  pass went 7,774,144 → 7,766,521 vs libdeflate's 7,774,214 — ~7.7 KB DOWN
  (0.098% of the fixture), while text/binary/tabular rows did not regress
  (`level.rs:448-452`, the "near-incompressible Lazy trap … does NOT recur
  under the cost-model parse" record). The margin order (0.1%) is ABOVE this
  row's hairline (+0.02% max), so the mechanism has clinical headroom.

**The victory condition, split by coordinate (this is why STEP 0 exists):**

* **If the weights/L7-9 and movie/L6 rows are T4-only (pass at T1, fail only
  under the chunk grid):** the victory condition is NOT a knob — it is Phase
  1's exact state carry (plan §3): seed the chunk compressor with matchfinder
  table + block-split detector state, T>1 converges to the T1 bytes, every
  T4-only cell closes BY CONSTRUCTION on the very same build. Clause-3
  cannot trip (output can only shrink toward T1 = the shipped T1 bytes),
  which is the whole point of the pivot's design. The correct move for these
  cells is to WAIT for Phase 1's parity-census verdict, not to spend
  hairline knobs on them. Candidate knob donors are unrecoverable at a
  0.0008% margin next to a construction that eliminates the entire class.
* **If any row is fail-both (fails at T1 too), the movable mechanism is one
  knob: route that level's T1 parse to the near-optimal (cost-model) parser
  at the recorded L11 knobs** — machinery already in-tree on the T>1 side
  (`params_parallel` routes `level 8|9 → params_parallel(11)`,
  `level.rs:488-490`; the L7 route was probed pre-pivot, measured "no
  failing fixture cell" then, 2.5-4.9x wall — and therefore not shipped).
  The single-knob change = `params(7) → near_optimal at L11's knobs
  (100,150; 4 passes)` for the failing L7 row (and analogously L8/L9 to L12's
  knobs if THEIR rows fail at T1). It never lifts a cadence-band row IF the
  route is taken only on the named cell's level AND the census shows the
  other L7 rows still pass — which is a clause-3 question, answerable only
  by the census, never by argument.
* The T>1-only form of that route (route inside `params_parallel`, T1 bytes
  untouched) is the cheap fallback for a fail-at-T4-only row that is a REAL
  parse-price failure on the chunked path — but a T4-only loss against
  libdeflate on a level whose T1 output ties libdeflate byte-for-byte is a
  SEAM cost (chunk-detector state restart + re-emitted Huffman headers at
  chunk boundaries), and seam cost buys nothing from a bigger parse; its
  designed fix is Phase 1. So: near-opt T>1 route only if the row is
  fail-both, or if the seam residue is measured large enough that an
  in-chunk parse win can absorb it — counted, never inferred
  (`examples/chunkgrid.rs`).

Cost model (near-opt at L7): the probe's own numbers — 2.5-4.9x vs the
shipped lazy parse at that level (`level.rs:484-485`), worth it against the
post-stack wall slack only where the cells actually fail; the 2026-08-11
sample measured "no failing fixture cell at L6/L7" and the pivot changed the
GATE half's population. Price it once: `fulcrum try <ref> --levels 7
--threads 1,4` on the frozen box; a wall regression on any T4 cell it
previously passed is the stop rule (plan §8.5-style: recorded
closed-at-both-coordinates, no re-sampling).

Death modes:

* **D1 — the weights trade runs the WRONG WAY on the very file being saved.**
  The lever family's flagship failure: every recorded attempt to strengthen
  the lazy-family parse on near-incompressible data emitted MORE literals
  (+31,516/+32,975 B, `level.rs:561-567`, :494-497); lazy2-style lookahead
  knobs lift the row they were aimed at. If the pin shows a fail-BOTH
  weights row, the near-opt route is the ONLY knob ever measured moving the
  tensor class in the small direction — it is also the one whose own record
  says "L6/L7 do not pay" pre-pivot. That tension is why the card must be
  scoped to a single level and judged on the census, not rolled out.
* **D2 — cadence-band lift at T1.** Routing L7 at T1 to near-opt breaks the
  L7 libdeflate byte-tie band for every other file (154-cell T1-tie census
  of 2026-08 includes L7 22/22); one regressor = clause-3 NO-SHIP. The
  checkerboard prohibition (`deflate/mod.rs:826-833`) means the fix cannot
  be "route only weights" — the route must be per-LEVEL and win the census,
  or it does not exist.
* **D3 — synthetic-fixture instancing.** The 7.77 MB numbers are the
  weights-LIKE fixture, not the 83 MB GATE file; hard stop #3's analog of
  the L2 min-3 lane death. The mechanism must be re-instantiated on the
  real file before any card is written, and the cost must be counted
  (`examples/chunkgrid.rs` exists to keep chunk arithmetic counted; plan §7).
* **And the standing one: never generalize across levels.** The per-level
  scopes in `level.rs` (L4-only compose; L8 excluded from depth-x4; L6/L7
  excluded from port routing) are each single-measurement verdicts. None of
  them licenses a rule for a neighbor row, and none forbids a re-measure at
  a changed coordinate (the pivot changed the coordinate).

## 4. The decision table — one move per cell, or record closed

| cell | probable coordinate | mechanism that can move it | status |
|---|---|---|---|
| `minjs.min.js:L5` | T1 fit (gzip/pigz parse price) | GZIPPY_LADDER probe at (32,30)/(35,65), then port `good_match` rung to ldx L5 (#363 template) at (8,32) | **PANS OUT** — the card is plan §5's own prescription, zero novel mechanism |
| `minjs.min.js:L5` (12,30) probe | T1 | depth 12 trade | **RECORDED CLOSED** — no size mechanism, min_len clamp and chain truncation both point the wrong way for this file |
| `data.sqlite:L4` | STEP 0 decides (T1-only vs T4-only; pre-pivot the T>1 Lazy@64 record closed it and the post-pivot one-encode arm deleted it) | re-instantiate the L4 strategy step (Lazy, depth ≥16) at port codegen; the L4-ladder card = plan §5's zlib-ng `deflate_medium` rung | **PANS OUT as a re-measure** — the parked record (`level.rs:873-884`: 11/11 TUNE smaller, 0 cells opened, clause-5 wall death on PRE-PORT codegen) says its wall verdict expired with the regime; it is the FIRST lever the record itself nominates for revival |
| `weights.safetensors:L7-L9` | T4-only seam residue (expected), fail-both unlikely | none at T1 (L7's libdeflate tie band; lazy/deferral measured WORSE here); T>1: Phase-1 state carry; fallback single knob = L7→near-opt-L11 route | **NO KNOB unless STEP 0 shows fail-both** — then and only then the scoped L7 route card |
| `movie.mp4:L6` (+0.0008%) | T4-only seam on a T1 tie (L6 is excluded from the port and byte-tied at T1 pre-pivot) | none — seam class; Phase 1 | **RECORD: construction-closed, no spend** |

Per-label ledger rule: a lever that closes any of these appends the row to
`tests/fingerprints/ledger.tsv` (append-only WON facts — header comment:
"nothing may remove a row except a git revert of the lever that added it");
none of the four files currently has an entry there, so every future closed
row is NEW ledger mass, which is the only progress metric the charter
counts. A card that clears its falsifiers is a ledger card; a lever that
fails any clause is a record-closed line in this memo's successor, and the
session ends — "never generalize across levels" then forbids re-deriving
the verdict at a neighbor level.

## 5. Recommended move (one card, in order, stop on first clause-3 trip)

1. **STEP 0 (frozen-box, no code): pin the coordinate.** `fulcrum why` on
   the four cells + the census fail-at-T1/T4/both split. The entire memo's
   lever assignment hangs on this; it is ~5 minutes of instrument time.
2. **Card M1 (conditionally first): L5 depth probe → good_match port.**
   Probe with `GZIPPY_LADDER=lazy:32:30` and `lazy:35:65` on TUNE;
   if either shrinks minjs AND does not regress the other 10 TUNE files on
   the sweep, route the winning rung through the L5 level row
   (`compress.rs:128` → matching (35,65) literally reproduces libdeflate's
   own L6 rung at L5 — a one-line map change in the post-stack world) and
   THEN price `good_match=8` separately, per the #363 template. Ledger card
   condition: `minjs.min.js:L5` rows close vs gzip AND pigz with zero L5
   flips → append.
3. **Card M2: L4 re-measure of the parked Lazy config** (depth held ≥16 to
   avoid the recorded min_len confound; per-label falsifier
   `ladder_is_monotone_t1` + `won_cells_stay_won` + census; wall gate
   `fulcrum try --levels 4 --threads 1,4`). Ledger card condition: the
   data.sqlite L4 rows close while the thinnest currently-passing
   weights-class row (igzip per the L2-family record) does not open — its
   +31.5 KB own-size regression is the known price of the step, so read
   which weights L4 rival row is thinnest from STEP 0's census first.
4. **Cards M3: none now.** If STEP 0 shows fail-both rows at weights L7-9 or
   movie L6, open exactly ONE scoped near-opt route card at the failing
   level with the frozen board's cell as the named target; otherwise these
   cells belong to Phase 1's construction and the hairline memo records
   them CLOSED-PENDING-PHASE-1 rather than levered.
5. **If every card above is dead in its falsifier's first leg (tie-guard
   flip, census flip, or wall leg outside clause 5 at the scoped
   coordinate) — record each verdict closed at its own coordinate and stop;
   do not re-derive at neighbor levels.** The four cells would then be
   replayed after Phase 1 lands (the state-carry construction closes their
   T4 halves for free), and the remaining T1 hairlines stay open for
   Phase 3's knob class with NEW census coordinates.

## Honesty column (what this memo did NOT verify)

* Every cell margin and rival attribution quoted here is from the plan §2b
  frozen-board text + in-tree records, NOT re-measured (read-only planning;
  the box was not touched). STEP 0 is the verification, and its outcome can
  reassign which card M1/M2 applies to which row — the mechanism map above
  is written so the reassignment is local.
* The "(12,30)" trade is closed by analysis of the vendor table and the
  recorded min_len clamp semantics, not by a measured sweep; if the sweep
  contradicts the analysis (some displacement break goes RIGHT through
  rows), the analysis loses to the sweep — cheap to check with the
  existing instrument.
* Crown-side numbers (21-38x, −364/−268/+16, 92-99%) are the in-tree
  records' own measurements; per charter rule they bind no future verdict
  outside their recorded coordinates and are cited only to keep a session
  from re-deriving them.
