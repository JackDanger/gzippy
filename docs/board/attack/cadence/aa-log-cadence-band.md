# aa — the L2–L5 GZIP-CADENCE BAND: dd79_bin6, access.log, photo.jpg

2026-09-05 · planning artifact (READ+WRITE only — no cargo, no box). One
deliverable of the pre-load fan-out (see `docs/board/attack/sequence/
seq-verdict-playbooks.md` for the DAG this attaches to: cell → lever → branch →
try → merge). Read paired with `docs/plan-2026-09-one-encoder.md` §2b (the
frozen board) and §5 (Phase 3).

Every number below carries its artifact. Byte pins marked **[CENSUS]** come from
the frozen-board artifact on the authority box
(`/root/www/gzippy-bench/campaign/size-all-ee0c1d2c/census.json`, commit
`ee0c1d2c`, 22 files, 4 rivals, L1-9, T1+T4 — handoff §3.2); no re-derivation of
those pins is attempted here — they are the referral contract. The in-tree
receipts cited alongside them come from EARLIER artifacts with different corpus
staging; where magnitudes differ, the ratio (not the byte count) is the
transferable quantity, and the §5 grid re-counts everything.

---

## 0. The memo in four lines

* The band's residual (`dd79_bin6 L2/L3` +0.93% cap, `access.log L5` +1.07%
  worst, `access.log L3`-vs-libdeflate +0.42%, `photo.jpg L1-L3` +0.04%) splits
  into two cell families: `dd79_bin6` (matches the rival finds in the
  SHORT-LENGTH bucket at FAR-offset trigram positions while our shipped parsers
  search a **one-deep trigram table** — the "MIN-MATCH-4 boundary") and
  `access.log` (+ depth/cadence on the lazy lane), with `photo.jpg` in the
  G37 block-cadence class — the same cells' header-mass receipts name both.
* The trigram-chain mechanism that pays for it is **already built, already
  measured, and switched off at every shipped level**: `next3_tab` +
  `hash3_chain_depth` sweep in the legacy `matchfinder/hc.rs` (d=4 saturates;
  measured −4,469 B at dd79_bin6 L2, −11,304 B at L3 on the old staging). The
  port (`ldx/hc_matchfinder.rs`) reproduces libdeflate's pure singleton and has
  no chain at all.
* The L1 "MATCH-REACH" receipt correctly says its two halves transfer to L2
  *only* as a mechanism re-derivation, not a knob move — the `l2min3` branch
  (PR #369 shape) proved the transfer dies synthetic-tabular-first when it
  lands as a forced `min_len = 3`; keeping the vendor's own
  `choose_min_match_len` adaptive gate is the shape that contains it.
* ONE scoped `fulcrum try` (+ `tie-guard.sh` + two `fulcrum why` runs) decides
  the whole band: §5 gives the exact grid and the pre-registered stop rules.

---

## 1. The cells and their pins

Band = the board's concentration note (`plan §2b`): *"L2-L5 gzip-cadence band
plus dd79_bin6/access.log"*.

| cell (rival) | pinned deficit **[CENSUS]** | archive corroborations (older staging, exact bytes) | what it grades |
|---|---|---|---|
| `gzip:dd79_bin6:L2:T1` | **+9,472 B** (+0.93%) | pre-pivot staging: +36,101 B vs pigz-L2 (1.0082), worst-vs-gzip ratio 1.00935/1.00927 (G31/G32, target-encoder §829-857, §2039-2040); ours == libdeflate-2 byte-exact | parse quality on binary cadence |
| `pigz:dd79_bin6:L2:T1` | ≤ same cap | 4,464,656 rival vs 4,500,757 ours (G31 table :753) | same |
| `dd79_bin6:L3` (gzip/pigz) | **+2,230 B** | legacy L3 residual vs pigz-3/gzip-3 measured **+0.445–0.767%** as a speed-only miss at `88cf1b09` adjudication, then as +19,767 B vs pigz-L3 = +0.445% (G32 :837, level.rs:833-844) | lazy-arm residual after far-len3+224-modulator |
| `gzip:access.log:L5:T1` | **+1.07%** ([CENSUS] worst on the board) | no counted row in-tree | spin-off of depth + block cadence on sparse text |
| `pigz:access.log:L5` | +0.75% | — | same |
| `libdeflate:access.log:L3:T1` | **+0.42%** | #364's far-len3 machinery lives on the legacy arm (plan §2b, handoff:98) | len-3 machinery parity vs libdeflate |
| `gzip:photo.jpg:L1-L3` | **+0.04%** | photo.jpg cell family counted in G37 :2166-2170 (gzip 3.5x our matches, **6.8x our header bits**) | hairline; block-cadence class |
| `data.sqlite:L4` / `minjs.min.js:L5` | +0.13/+0.17% / +0.19/+0.16% | — | band-edge cells the same grid grades for free |

Implied sizes from the pins, where a ratio and byte pin coexist (arithmetic
shown, not asserted): +9,472 B at +0.93% ⇒ rival ≈ 9,472/0.0093 = **1,018,495 B**
compressed; IF the member keeps the old staging's compression ratio
(4,464,656/6,291,456 ≈ 0.709) the frozen-corpus dd79_bin6 input is ≈ 1.44 MB.
The L3 byte pin then reads 2,230/9,472 = 23.5% of the L2 deficit. The
pre-pivot staging was 6,291,456 B in with compressed ≈ 4.46 MB (G32 :832) — a
~4.4x staging delta. **Every absolute prediction below scales by the
deficit's own size; the grid re-counts.**

---

## 2. Why these two shapes sit at the MIN-MATCH-4 / LAZY-vs-MATCH-REACH boundary

### 2.0 What each band level actually runs today (the routing facts)

| level | producer (post-stack, pre-#363/#364) | parser | len-3 posture | search knobs |
|---|---|---|---|---|
| L1 | legacy `parse/fast.rs` (igzip-derived) | chainless single-probe + `head3` | hash3 probe gated (lit-fraction cliff at 48%, `fast.rs:884`), chain depth 8 (`HC_HASH3_CHAIN_DEPTH`), reach knobs T1 | 1 probe/pos |
| L2 | `ldx` port `compress_greedy` | greedy + `hc_matchfinder` | **singleton** `hash3_tab`, no chain, **no far gate** | depth 6, nice 10 |
| L3 | legacy `parse/lazy.rs` | lazy + `far_len3_gate` + sparse modulator **224x** | far len-3 priced per-block (evidence ≤ literals), #364's lane | 12/14 lazy |
| L4 | `ldx` port `compress_greedy` | greedy + hc | singleton, no gate | 16/30 |
| L5 | `ldx` port `compress_lazy` | lazy + hc | singleton + the **offset>8192 len-3 refusal** | 16/30 lazy, half-depth lookahead |

Sources: port table `src/compress/ldx/compress.rs:114-199` (mirrors
`vendor/libdeflate/lib/deflate_compress.c:3920-3948`); legacy table
`src/compress/deflate/level.rs:770-933`; exception accounting
`src/compress/deflate/mod.rs:776-871` (`level_uses_ldx = matches!(0|8|9)` at
this tip; on the #367-stack branch `!matches!(1|6|7) && level <= 9` —
`.worktrees/l2min3` at `src/compress/deflate/mod.rs:626`).

### 2.2 `dd79_bin6` — binary cadence: the SHORT-MATCH-DISCOVERY boundary

* `dd79_bin6` = repeating program bytes; the dominant repeat matches are
  3–4 bytes (G36's bucket receipt: `match_len_L00` ours 0.118908 vs gzip
  0.179643 per input byte — **gzip finds ~51% more of the SHORTEST matches**),
  and gzip covers 0.780420 of input positions with matches vs our 0.651770,
  319,260 more matches, 58.59% fewer literals (G36 counts, one `fulcrum why`
  run, exact position counts, not inferred).
* The mechanism is **discovery, not depth**. Three depth/good_match levers
  already measured no movement on this cell: depth ×8 +377 B (G30/G33), zlib
  L2 knobs (depth 8, nice 16) with or without `good_match` +47 B WORSE (G32a,
  falsified in advance: zlib's `good_match` *shortens* the chain once a ≥4
  match appears, so on binary-cadence data the exit fires immediately and the
  effective depth DROPS below nominal — G32a :892-902), lazy:48:16 still
  +2,316 B. What gzip has that we do not is a **3-byte hash with a real
  chain** (`vendor/gzip/deflate.c:282` `UPDATE_HASH`, 15-bit, `prev[]`
  chain) — while ours is a one-deep singleton: a len-3 is findable at position
  *p* only if its previous occurrence is *the last one written for that
  trigram key* (`matchfinder/hc.rs:76-91`, `ldx/hc_matchfinder.rs:220-225`).
* This is why "MIN-MATCH-4" is the right name for the symptom and the wrong
  name for the cause: we *accept* len-3 matches, `DEFLATE_MIN_MATCH_LEN` is 3,
  and libdeflate's per-block adaptive `min_len` lowers to 3 whenever the
  alphabet is wide (≥80 used literals — program bytes qualify;
  `vendor/libdeflate/.../deflate_compress.c:2299-2327`, port
  `ldx/min_match.rs:30-61`). The practical floor is 4 because only the most
  recent trigram position is reachable. Gzip's L2/L3 (`deflate_fast`, config
  `good=4 lazy=5 nice=16 chain=8` and `4/6/32/32`,
  `vendor/gzip/deflate.c:242-254`) walks trigram chains with **no distance
  gate at all** at those levels — the `TOO_FAR 4096` len-3 discard applies
  only in `deflate_slow` L4+ (`deflate.c:129-132, 686, 712-717`).
* The in-tree test of that mechanism is exact and banked: chaining the member
  (`next3_tab`, legacy `matchfinder/hc.rs:115-125` walk at `:395-425`) moved
  `dd79_bin6` **L2 4,500,757 → 4,496,288 at d=4** and **L3 4,461,737 →
  4,450,433 at d=8** (`hc.rs:84-91`; data.sqlite L2 −382 / L3 −767; dickens
  byte-unchanged — the walk is a no-op on text where the 4-byte chain already
  reaches `best_len >= 3`). 64% of our dd79 L2 matches came out of the
  one-deep table (744,797 of 1,168,118, `hc.rs:120-122`) — the chain targets
  exactly the bucket the cell loses in.
* Data bits explain the whole cell at the old staging: ours 5.716180 data
  bits/B vs gzip 5.662213 ⇒ 6,291,456 × 0.053967/8 = **42,440 B data excess**,
  part-refunded by our smaller header mass (42,812 vs 51,558 bits = +1,093 B
  for gzip) — which is why G36 sentences the residual "SHORT-MATCH DISCOVERY"
  and separately notes gzip *buys* more, better-fitted blocks (the cadence
  half, §2.3/P3).

### 2.3 `access.log` — sparse texty log: the LAZY-lane boundaries are depth + block cadence

* access.log L5 is **already our lazy parser** (port `compress_lazy`,
  depth 16, nice 30; libdeflate routes L5 lazy at
  `deflate_compress.c:3945-3948`). gzip's L5 is also lazy (`deflate_slow`,
  `good=8 lazy=16 nice=32 chain=32`). The per-candidate deltas, code-read:
  our chain cap is **16 vs gzip's 32** (`deflate.c:250`), our second-position
  lookahead searches at `depth >> 1`
  (`deflate_compress.c:2694-2721` — "it's more worthwhile to use a greater
  search depth on the initial match"), and gzip shortens its own chain 4x
  (`>>= 2`) only past `good_match` (8 at L5, `deflate.c:405-407`) — so at
  sparse-texty
  distances the first-candidate walks diverge 2x on the axis that actually
  runs. That is a vendor diff, nameable in one line, but it is UNMEASURED on
  access.log — G10's depth lever died against the **old codegen's** cost model
  (blocked, not refuted: `target-encoder §610-650`, and the wall regime
  changed 1.6–4.1x → 1.03–1.11x since).
* Len-3 is NOT the access.log L5 donor from the vendor table: gzip's
  `deflate_slow` discards far len-3 at `TOO_FAR 4096`, our lazy keeps them to
  8192 (`deflate_compress.c:2666-2673`) — ours is the LOOSER gate there. Do
  not start from a len-3 story for this cell; run the `fulcrum why` first
  (§5 leg 2). What the header-mass receipts do name for access.log's class:
  on near-incompressible/sparse texty data, "gzip emits far more, smaller
  blocks with tighter-fitted Huffman tables" and **wins it back on data bits**
  (G37: photo.jpg gzip 103,925 header bits vs our 15,374 AND 3.5x matches;
  weights.safetensors AR 8.7x header mass, data −0.017 bits/B, net win) —
  the under-splitting mechanism (G37a: "our block-end check has no SPARSITY
  term, only a drift detector").
* The measured block-placement ceiling caps what cadence can buy:
  `examples/split_headroom.rs` / `proposer_recall.rs` measured
  **0.126–0.133%** (plan §7 tool table). Arithmetic: access.log L5 needs
  +1.07%; cadence/topology carries ≤ 0.133% of input — **cadence alone cannot
  close access.log L5 by a factor ≥ 8**; it cleanly covers photo.jpg (+0.04%,
  the G37 class), and it is a co-lever, not the band's main instrument.

### 2.4 The L1 MATCH-REACH receipt, and why it does not transfer as a one-line change

* The receipt (KNOWN_SAGS, `tests/size_invariants.rs:68-87`, re-pinned at
  26fc4505): the L1 MATCH-REACH lever = **dense match-interior indexing + the
  2-way bucket shift that vendor `ht_matchfinder_skip_bytes` does** — L1 fell
  666,379 → 662,577 (−0.57%) on the `binary` fixture while L2 stayed at its
  libdeflate tie; the sag is L1 getting better, and "L2 is a different parser
  entirely (Greedy + the hc matchfinder), so the reach mechanism proved here
  does not transfer as a one-line change."
* Mechanism read, from the lever's own code: the two halves are (a) the
  interior insert loop that hashes EVERY skipped byte up to `insert_end`
  (`fast.rs:2085-2124`, shipping `usize::MAX` = dense on T1-REACH, capped 8
  for the hash3 table at `:899-908`), and (b) the whole-bucket shift-insert
  `slot1 <- slot0; slot0 <- pos` on one cache line (`fast.rs:2096-2119`,
  `apply_l1_match_reach_t1_knobs`).
* On the L2-lane parser both halves are **already effectively present** — and
  that is exactly why the L1 lever does not transfer: hc's bulk `skip_bytes`
  inserts hash3+hash4 at every interior position it advances
  (`vendor/libdeflate/lib/hc_matchfinder.h:361-398`; port mirror
  `ldx/hc_matchfinder.rs`), so interior indexing is already dense; and
  chain-walkers need no 2-way bucket (that is an igzip-class single-probe
  lever, `fast.rs:1015-1028`). What L2 actually lacks is (c) — **chain depth
  on the trigram table** — which is a third mechanism the L1 receipt never
  carried, and which legacy `hc.rs` already built and measured (§2.2). The
  "reach does not transfer" sentence is true *because* the transferring
  capability is next3-chains, not bucket shape.
* The 2-way bucket's own half-confirmations: L1's INTERLEAVED bucket lever
  paid +34%/+16% T1 wall when shaped as two arrays (the very falsification
  its comment banks, `fast.rs:1015-1028`) — a working-set warning that P1's
  ported `next3_tab` (a plain `i16 × WINDOW` mirror of `next_tab`,
  `hc.rs:124`) inline-array repr inherits deliberately.

### 2.5 The l2min3 lane: what the transfer actually died of (synthetic-tabular-first)

PR #369's branch (`.worktrees/l2min3`, tips `d697be44` → `df311593` →
`23c114dd`) is the closest prior attempt and its three commits carry the
complete errs-and-verdicts record:

| commit | shape | measured stop |
|---|---|---|
| `d697be44` | port of the L3 len-3 machinery INTO ldx (far_len3.rs + lazy gate + 224x sparse modulator); L3 byte-identical 11/11 files vs legacy; L4 sag receipts re-pinned | passed (the L3 byte-identity leg; only tabular/binary L3→L4 SAGs re-opened by pick-min deletion remain listed) |
| `df311593` | L2 min-3 HYBRID: unconditional far-len-3 lane; trigram chain per-decision faithful (silesia L2 sha identical to the deleted hybrid arm) | `won_cells_stay_won` RED: **tabular:L2 = 303,559 vs libdeflate 281,326 (+7.9%), 35,039 len-3 matches (pin: 0)**; silesia.tar L2 −0.58% (70,877,952 → the gated 70,463,509) |
| `23c114dd` | L2 far-len-3 accept armed with the legacy cost gate (`FarLen3Gate::allows`, `TRIGRAM_EXPENSIVE_ACCEPT_SLACK_EIGHTH`, recalc cadence) | **tabular:L2 = 302,492 > pin 281,516 — stop rule hit.** Diagnosis: the gate accepted only 715 far len-3 (+723 B); the bleed is 18,327 NEAR (offset ≤ 4096) len-3s that the main accept arm takes because **the hybrid shape FORCES `min_len = 3`**. "No far-len-3 accept policy can close a gap made of near len-3s." |

The contained read-out for this memo: the lane's emitter of death was
`forced_min_match_len: 3` overriding the vendor's own per-block adaptive
`choose_min_match_len` (tabular's ~6-literal alphabet maps to `min_len` 9,
depth-clamped to 5 at L2 — `min_match.rs:167-176` pins both bands). Under
`choose_min_match_len`, tabular NEVER accepts a len-3 at L2 and the pinned
281,326 tie survives **by construction**; dd79/access.log run full alphabets
(`min_len = 3`) and are exactly the inputs the chain serves. The l2min3
sweep's own fixture rows agree: `binary:L2 665,499 (pinned 666,112, −0.1%)`,
`text:L2 342,903 (pinned 348,859, −1.7%)` with the same lane shapes
(commit message, 23c114dd). That is the shape correction this memo carries
forward, and it is why the KNOWN_SAGS sentence is right and still points here.

---

## 3. Vendor reads, line by line, for these cell shapes

### 3.1 gzip — the cadence donor (`vendor/gzip/`)

* `trees.c:991-1006` — **`ct_tally`'s estimated-cost early flush**, the thing
  the band name comes from: every 0x1000 (=4096) symbols, `level > 2` only,
  compute `out_length = last_lit*8 + Σ dyn_dtree[dcode].Freq*(5+extra_dbits)`
  and flush the block if `last_dist < last_lit/2 && out_length < in_length/2`.
  A cheap, literal-heavy-block trigger: on line-shaped logs and cadence
  binaries it ends blocks where the Huffman tables are still well-fitted to a
  REGIME, not drift-detected late. Exceptional gating word: it is a fixed
  function of the running histograms — no content detection, and the same
  shape libdeflate's block-split detector already legalizes
  (`block_split.rs` is the exact port; technique-index :275).
* `deflate.c:242-258` — the per-level configuration table: L2 `{4,5,16,8}`
  fast, L3 `{4,6,32,32}` fast, L4+ slow; `good_match` chain-halving at
  `:405-407`; len-3 `TOO_FAR 4096` discard at `:712-717` (slow only).
* What it gives the band: full trigram chains at ALL levels (our engine stops
  at depth 0!), a fast-lane without the far-len3 refusal at L2/L3, and a
  block cadence that pays ~1,093 B of extra header for a measured
  ~42.4 KB data-bit surplus on the worst cell (G36's own bits/B counts:
  5.716180 − 5.662213 = 0.053967 bits/B × 6,291,456 = 42,440 B). Its
  per-block rewrite trick (whole-file stored member, `trees.c:899-912`) is
  out of scope here.

### 3.2 igzip —`(D6)` level_buf grading, block size as a memory knob

* `vendor/isa-l/include/igzip_lib.h:296-329` (via technique-index D6
  :289-292): token space graded MIN(1·K)/SMALL(16K)/MEDIUM(32K)/LARGE(64K)/
  EXTRA_LARGE(128K), default LARGE; a block opens per `level_buf` fill and
  `flush_icf_block` (`igzip.c:200-228`) rebuilds exact per-block tables off
  `isal_mod_hist` (E8's semi-dynamic two-pass). Mechanism: **block sizing is
  a MEMORY parameter, not a heuristic** — the buffer the caller provides IS
  the block size.
* What it gives the band: it is prior art that "block size is a legal tuning
  parameter" under the rule-3 carve-out (technique-index :292 says this in so
  many words), and O-6's open question (libdeflate 300K/50K vs zlib-ng 16K)
  is the same knob. As a LADDER change it is the biggest-single-diff item here
  and does not rank for a one-turn falsifier; it is the parked shape if the
  cheap levers miss (P4).

### 3.3 libdeflate — the shipped parser, diffed for these shapes

* `ht_matchfinder.h:104` (`HT_MATCHFINDER_MIN_MATCH_LEN == 4`), `:131-160` —
  the 2-wide bucket, shift-insert at probe; `:197-232` —
  `ht_matchfinder_skip_bytes` shifting the whole bucket every interior step.
  This is the L1 arm's ancestry; L2+ runs hc, where none of it applies.
* `hc_matchfinder.h:71` ("typical greedy or lazy-style compressors, where
  length 3 matches ..."), `:112-122` — `hash3_tab[1<<15]`, ONE node per
  trigram key, `:218-239` insert-overwrite-then-walk-4; `:361-398` —
  `hc_matchfinder_skip_bytes` inserts EVERY position's hash3+hash4 (dense
  interior, the reach half already present at L2 in the port).
* `deflate_compress.c:2666-2673` — the lazy near-len-3 floor: reject a found
  len-3 entirely past offset 8192. Our L5 keeps this; gzip's slow path
  discards len-3 past 4096 (`TOO_FAR`); our L2/L4 (greedy) keep NOTHING — a
  found far len-3 is
  accepted unpriced (that is why the legacy far-len3 gate exists at shipped
  L2/L3 today: `level.rs:787, 865`).
* `deflate_compress.c:2296-2377` + port `min_match.rs:30-120` — adaptive
  per-block `min_len` (4 KiB first read; `recalculate_min_match_len` per
  ~10 KB recalc cadence — same cadence the legacy gate recalc shares).
  THE guard that contains P1 (§2.4).
* What does NOT exist in libdeflate and P1 adds: `next3_tab` chains. The
  vendor shape is what `[CENSUS]` pins: our L2 bytes == libdeflate-2
  byte-exact (G32 fact #1) — so the census deficit vs gzip is a LIBDEFLATE
  deficit the vendor would have to re-derive gzip-style per level to close.
  The pitch: our deviating from libdeflate at L2-L5 by ONE mechanism is
  exactly the "adopting-parse difference adopted one level at a time" the
  #363/#364 template already runs.

### 3.4 zlib-ng `deflate_medium` — the middle rung, already named

* `vendor/zlib-ng/deflate_medium.c:123-176` (`fizzle_matches`: slide the NEXT
  match left one byte at a time, stealing from `current`, under
  `current.match_length > 1`, `next.match_length >= 256`
  break, `c.match_length <= 1 && n.match_length != 2` commit) with
  `early_exit = s->level < 5` (`:184-185`).
* This is G11's "missing middle" and plan §5's named L4 candidate. For the
  band it is the pre-built rung for `data.sqlite:L4` (+0.13/+0.17%) and the
  KNOWN_SAGS L4 inversion (`(tabular,3) +9,472 B`, `(binary,3) +2,538 B`,
  re-pinned at d9418505 — note the SAME tabular +9,472 number appears there;
  unrelated cell, do not conflate). P2 consumes this precedent at L4/L5.

---

## 4. Mechanisms ranked: expected per-cell gain vs falsifier cost

| # | mechanism | vendor precedent | touches | expected band effect | falsifier cost |
|---|---|---|---|---|---|
| P1 | **Trigram chain** (`next3_tab`) at L2/L3/L4/L5 in the port, `hash3_chain_depth = 8` (sweep-saturated value), keep `choose_min_match_len` ADAPTIVE — do NOT force `min_len` (§2.4, death mode 1) | gzip / zlib single 3-byte hash + `prev[]` chain (G36 [H3]; zlib-ng M11 `hash3_tab` at L9); legacy `hc.rs` carries the code+receipt | L2/L3/L4/L5 parse stream (GATE+TUNE) | measured −10.8% of the gzip L2 deficit and −57.2% of the L3 deficit on the same file/shape; data.sqlite/dickens receipts say text is cost-0 | port of shipped legacy code into ldx + fixture ladder + tie-guard + §6 grid |
| P2 | **L4/L5 effort rung**: depth → 32 with gzip's `good_match` early exit (`>>2`-on-`prev_length ≥ good`) funding it; or plan §5/#363 port the `(good,depth,nice)` rung | gzip `deflate.c:405-407, 250`; zlib-ng medium `early_exit` suffix; the PARKED L4 lazy-d10 rung (11/11 TUNE win, 0 opens, clause-5 wall fail — encoder-campaign-plan §4, "expires if the wall budget changes") | L4/L5 knobs only | access.log L5 is the target; L4 maintenance cell rides along | knob + one-line exit in the port lazy loop; same grid |
| P3 | **Block cadence**: add gzip's 4096-symbol estimated-cost flush as a SPLIT SITE feeding the exact-cost three-way per-block chooser (level > 2) | `trees.c:991-1006`; G37/G37a under-splitting; blockspans cv receipts | block boundaries (all levels > 2) — broadest blast radius | closes ≤ 0.133% ⇒ photo.jpg (+0.04%) class only; co-lever for dd79 header trade (~≤1,093 B); **cannot close access.log L5** (0.133% < 1.07%) | same grid; needs the exact-cost per-block plumbing we already have (`parse/mod.rs:673-768` E6) |
| P4 | igzip ICF/semi-dynamic re-grade (D6+E8 block-size-as-memory + pass-2-able token format) | igzip level_buf grades | whole emit path | range unknown; ranges over ALL band cells if landing | large; park (P11's falsified half binds the emit side) |
| base | **PR #364 rebase first** (L3 port byte-identity + far-len3 + 224 modulator in `ldx`), then P1 rides | #364 (open), already 11/11 byte-identical vs legacy L3 on the l2min3 fork | L3 route prefix | makes P1-at-L3 a delta, not a shape change | rebase + scoped try levels 3,6,7 (playbook A step 7) |

Cost order is the charter's cheapest-falsifier-first: P1 is mostly-transcribed
legacy code (lowest truth-risk), P2 is a knob + one guard line, P3 is a
bounded emit-loop site, P4 is parked. The §6 grid grades P1+P2 (+sentinels);
P3 enters only after its row in §6 (its ceiling arithmetic says it cannot
carry the band alone).

### The three death modes per mechanism (pre-registered; a death lands, the lever parks)

**P1 trigram chain**

1. *Synthetic-tabular-first (measured once already):* forced `min_len=3`
   bleeds tabular +7.9% / +21,233 B (df311593) — death if any commit re-introduces
   the forced-min shape; the l2min3 23c114dd shape (adaptive min kept, gate on
   far lane only) is the only shippable posture. Pin: `tabular:L2:T1` stays
   281,326 = libdeflate tie; `won_cells_stay_won` ledger on tabular:L2:T1 green.
2. *Tie cage flips both ways at text-like labels*: text:L2 (min_len 3 blocks)
   will move bytes. A tie cell that gets BIGGER than libdeflate fails
   `won_cells_stay_won`/`tie-guard.sh` immediately (precedents: hash3-chaining
   6 closed/12 flipped; zlib `good_match` 31 closed/17 flipped with
   data.csv L2 1.0000 → 1.0431, CLAUDE.md tie paragraph). Death if any
   sentinel flip reads net-worse after P2/P3 are priced — record closed-at-
   both-coordinates, do not re-sample.
3. *Wall: dependent trigram-chain loads at every hit-able position* — the
   exact load-ordering pre-registration the parked L1-hash3 record demands
   (`l1-next-lever.md`: search first, blind store on hit, load-then-store on
   miss) plus the depth cap ≤ 8 (sweep: d=4 saturates, d=8 already
   "identical or a byte worse"); clause-5 budget on the frozen box decides;
   the L6/L7 `good_match` wall regressions are the family tree this dies in
   if ignored.

**P2 L4/L5 depth rung**

1. *Good-exit short-circuit repricing*: at L2 `good_match=4`-shaped exits were
   measured WORSE (+47 B, G32a) because the exit fires immediately on short
   matches — at L4/L5 (`good=8/16`) the same cliff could reprice sparse-text
   cells; death if access.log:5 loses data-cvs-relative margin vs the depth
   gain.
2. *The L4 lane's own history: pure depth ≠ the cell* — PARKED lazy depth-10
   L4 runs died clause-5 9/11 files at T4 under the OLD codegen
   (level.rs:873-888); if the new-codegen wall leg re-fails, the class is
   recorded closed on the wall at both coordinates (plan stop rule), no
   re-sampling.
3. *Ladder incoherence*: raising L4/L5 effort must not produce L5 < L4
   inversions (KNOWN_SAGS tabular:3/binary:3 pair) — `ladder_is_monotone_t1`
   is the 2-min kill, and it fails closed because the sag list must SHRINK
   for a listed pair to unlist.

**P3 block cadence**

1. *Ceiling exhaustion*: ≤ 0.133% measured placement headroom < the +1.07%
   access.log cell ⇒ the co-lever framing is a no-op row measured; only
   photo-class cells decide this lever. Death is cheap and fine.
2. *Tie-cage blast radius: every T1 byte at L>2 changes* — the 66/66 tie set
   (CLAUDE.md working rule) forbids ANY byte flip on tied cells; the only
   spaces where cadence can change bytes without flipping a tie are cells
   where libdeflate-cadence is ALREADY losing (precisely the [CENSUS] band) —
   that is a narrow, checkable claim, killable by one sentinel row.
3. *Splitter interaction*: the port's block-split detector and the flush site
   must compose without double-flush artifacts (the #266-era StoredCoalescer
   story); a single mis-ordered flush on stored-dominated access.log runs
   re-increases the framing overhead — `noise_expansion_bounded_*` slack
   pins catch it in-process in seconds.

**P4 ICF re-grade** — parked; death modes only if built: (1) emit-path
redesigns are falsified territory (`parse/mod.rs:968-1465`, five counts);
(2) token-layout change required before the falsified gather-emit ever
measured clean (E8/F4 note); (3) T>1 bit-splicer contract (Phase 1) makes
emission-layout churn the wrong phase order — land Phase 1 first.

---

## 5. The batch falsifier — one grid decides the band

The instrument sequence, cheapest first, one ref (`<after-ref>` = the
#363→#364 rebased stack; per handoff §4.5 and playbook A step 7 the rebase
land order is fixed — the band grid reuses its levels-3,6,7 byte-identity
receipt for L3):

1. **In-process, seconds, fail-closed:**

        cargo test --release --test size_invariants

   KNOWN_SAGS must not gain a row; `("tabular",1)`'s pre-existing unlisted
   sag (23c114dd note) is the L2 lane's known synthetic shape — the memo's
   nav: if P1's port form worsens it FURTHER, that is death mode 1, stop.
   `noise_expansion_bounded_*` bounds & the fixture ladder both green = continue.

2. **Tie subset, ~2 min (charter working rule, before any T1-byte edit is
   even reasoned about):**

        scripts/campaign/tie-guard.sh <ref>

   Pass bar: NON-WORSE ON EVERY TIE. (Being SMALLER than a vendor on a tied
   cell opens it as a win; being larger kills the lever — two net-positive
   levers already died exactly here.)

3. **Attribution legs (one command each, no box contention — these name the
   mechanism per cell before the grid judges amounts):**

        fulcrum why 'gzip:dd79_bin6:L2:T1:size' --ours $BIN \
          --rival-cmd 'gzip -{level} -c {input}' --corpus dd79_bin6
        fulcrum why 'gzip:access.log:L5:T1:size' --ours $BIN \
          --rival-cmd 'gzip -{level} -c {input}' --corpus access.log

   Expected: dd79 = the G36 panel again (short-bucket position-count delta);
   access.log L5 = whichever of {matched-positions, literals, header-mass}
   leads. If access.log's panel is NOT favors-depth/cadence-shaped, P2/P3 die
   for THIS cell and the memo's row is voided by that print, not by opinion.

4. **THE deciding grid — one scoped `fulcrum try`, size leg first
   (`--size-only` is a busy-box variant), with deterministic seeded
   sentinels watching the clause-3 envelope:** the scope grammar and sentinel
   semantics are `fulcrum/src/promote.rs:1336-1487` — keys `levels`,
   `threads`, `corpus`; out-of-scope cells are measured normally and graded
   ONLY for clause-3 flips; the plan is recorded in try.json; a declared
   scope covering the whole grid is REFUSED.

        fulcrum try <after-ref> --size-only --threads 1,4 \
          --scope "levels=2-5;threads=1,4;corpus=dd79_bin6,access.log,photo.jpg,data.sqlite,minjs.min.js"

   This is the single instrument that decides the band: 4 rivals × 5 corpora
   × levels 2-5 × T{1,4} × size axis in-scope, judged in full, with the
   seeded stratified sentinel draw stating exactly what it did not look at
   (`ScopePlan`, `promote.rs:1406-1487`). The five corpora carry every
   failing band cell **[CENSUS]** plus the two edge cells; weights/movie
   hairlines sit behind sentinels.

5. **Wall leg (promotion instrument, only on a size SHIP):**

        fulcrum try <after-ref> --threads 1,4 \
          --scope "levels=2-5;threads=1,4;corpus=dd79_bin6,access.log,photo.jpg,data.sqlite,minjs.min.js"

   Same grid, both axes; frozen box; P1's clause-5 budget is exactly this
   leg's row for any clause-5 wall regressions the trigram walk adds.

Pre-registered stop rules (write into the try's notes before running):
P1-only closes `dd79_bin6:L3` only if the recovered fraction ≥ +2,230 B — the
arithmetic predicts it does NOT (§6); the honest composite is P1+P2 at
L4/L5-knob levels and P1+P3's tiny photo row. A scoped NO-SHIP parks the
branch per playbook C with the new `why` rows as its successor memo's first
inputs.

---

## 6. Cell → mechanism → predicted delta (arithmetic shown)

Symbols: Deficit₀ = [CENSUS] pin; archive Fraction = measured recovery ÷
archive deficit (exact byte receipts, transferable as an input-scale-proportional
fraction); Scale = implied staging ratio ≈ 0.228 (= 9,472/41,527 — the two
stagings' deficit ratio; treats content as the same corpus member at
different sizes, which the sha-identical staging checks on the box confirm
or refute in one command).

| cell (deficit₀) | mechanism | predicted delta | arithmetic | verdict band |
|---|---|---|---|---|
| `dd79_bin6 L2` (+9,472 B) | P1 chain d=8 | **−1,019 B** open after P1 | archive: −4,469 of +41,527 ⇒ 0.1076; 0.1076 × 9,472 = 1,019 | cell OPEN alone; closes only composed |
| `dd79_bin6 L2` | P1 + P2 knobs at L2 | −1,019 − ~0..−600 B | G32a: depth{6→8} alone = +47 (WORSE); with chain preseeding it is unmeasured — call it ≤ −600 | open |
| `dd79_bin6 L3` (+2,230 B) | P1 chain d=8 | **−1,276 B**, open by ~954 | archive: −11,304 of +19,767 ⇒ 0.572 × 2,230 = 1,276; far len-3 gate already on this lane, 224-modulator already on | cell OPEN unless cadence co-lever lands |
| `dd79_bin6 L3` | P1 + P3 (photo row's split cadence) | −1,276 − ≤~600 B | G36: gzip pays +8,746 header bits ≈ 1,093 B MORE than ours at this file; half of that is the plausible co-recovery | borderline close; the grid decides |
| `access.log L5` (+1.07% / +0.75%) | P2 depth-to-32 + good-exit | 0.3–0.8 × Deficit | no in-tree measured row (G10's depth lever died on OLD codegen's wall, coordinate-expired); the `fulcrum why` leg 3 decides BEFORE the grid — mechanism-only claim |
| `access.log L5` | P3 cadence | ≤ 0.133% of size | split_headroom/proposer_recall ceiling 0.126–0.133% < 1.07% ⇒ cannot carry | co-lever only |
| `access.log L3` vs libdeflate (+0.42%) | P1 at L3 (post-#364 prefix) | −(0.4..0.8) × Deficit₀ | data.sqlite L3 measured −767 B / dickens 0.0 on the chain; far len-3s are exactly what the offset density of logs rewards — expect material, unprecise | open, row graded in grid |
| `photo.jpg L1-L3` (+0.04%) | P3 cadence | FULL deficit if placement recovers ~0.04–0.13% | 0.04% < 0.133% ceiling; G37's 6.8x header-mass receipt is the mechanism | closest race on the board |
| `photo.jpg L2/L3` | P1 chain | small ± | photo shares dd79's discovery shape at lower density; sweep had no row — sentinel | open |
| `data.sqlite L4` (+0.13/0.17%) | P2 rung / #363-#364 spill | unpriced | rides the same grid row | open |
| `minjs.min.js L5` (+0.19/0.16%) | P2/P1 | unpriced | same | open |

Sum sanity: P1's measured archive effects ≈ −4,469−11,304 = −15,773 B on the
old staging; scaled ≈ −3,600 B across the two dd79 cells (the pinned pair is
+11,702 B) ⇒ ~31% of the band's dd79 mass by the cheapest instrument. The
remainder sits in cadence (capped ~0.133%) and depth (unpriced until leg 3).
Do not round this up into a closed-cell claim; the falsifier is the claim.

---

## 7. Containment, sag interplay, and the do-not-sweep fences

* **The lattice order is fixed by already-banked gates**: land the stack
  slices (#366/#367) → #363 → #364 (scoped try levels 3,6,7 + 23-file
  byte-identity checklist in its PR body) → then the band ref. #364's
  byte-identical L3 port is P1-at-L3's PARENT: net new behavior comes from
  the chain alone on top of byte-identity, so any cell flip is attributable
  to exactly the one added mechanism at a time (charter #1).
* **KNOWN_SAGS**: P1 does not touch the L4 rung; the (`tabular`,3)
  +9,472 B and (`binary`,3) +2,538 B listings stay until plan §5's L4 lever
  (zlib-ng medium / #363-style rung) lands; the list only shrinks. P1's
  inbox: (`binary`,1) L1→L2 must not heal LATER than the fix that closes L2
  properly (the test fails a healed listed pair — it must be UNLISTED by
  whoever closes L2, size_invariants:81-86).* **The frozen-knob fences** (do not sweep in a band lever's build turn):
  `L1_HASH3_GATE_LIT_THRESHOLD_PCT = 48` is a 2-point-wide cliff fitted on
  `dd79_bin6` L1 (`fast.rs:831-884`); `L1_HASH3_INTERIOR_INSERTS = 8`
  frozen by one-lever-per-lever charter (`fast.rs:899-908`); the l2min3
  branch's `(false, true, 8)` L2 config is the shape to REBASE, and its
  `forced_min_match_len` is the line NOT to carry (§2.4, death mode 1).
  Hygiene queue separately: `L1_HASH3_GATE_LIT_THRESHOLD_PCT` + the
  `l1-tune` module deletions are owner-ordered (handoff §4.8) — do them in
  the hygiene PR, not inside a band lever.
* **No content detectors**: the P3 flush condition is a fixed function of
  running token histograms (level > 2, 4096 symbols, out_length < in_length/2
  && last_dist < last_lit/2) — the same artifact class libdeflate's
  `choose_min_match_len` already is; the `min_match.rs:15-20` clause-3 note
  is the precedent sentence for this class.

## 8. What this memo could not count (box legs, named)

1. The [CENSUS] staging of `dd79_bin6` (implied ~1.44 MB input, ~1.02 MB
   compressed): settle sha + byte count with one command before the band
   grid (`fulcrum board --size` re-run recovers it canonically).
2. access.log L5's structure panel: the `fulcrum why` leg-3 run above.
3. P2's wall price under the NEW codegen (the parked L4 clause-5 verdict
   expired only in regime; it has no standing number).
4. The l2min3 fork's `ldx/far_len3.rs`, `compress_greedy.rs` gate wiring, and
   the `TRIGRAM_EXPENSIVE_ACCEPT_SLACK_EIGHTH` shape — written, clippy-clean,
   1,237 lib tests green on that branch (23c114dd) — are the P1 host; the
   rebase diff vs the post-#364 port must be diffed line-by-line in the band
   build turn (it is the fork's one unmerged gold).

— end of memo. Sources: all `path:line` refs are in-tree at the brain worktree
tip `5900d17a` unless a branch name is given; the named staging receipts
(artifacts quoted above) live on the solvency box and are re-counted, not
re-derived, by any coder acting on this memo.
