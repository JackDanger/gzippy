# vv — zopfli / ECT / gzip mechanism cards for the named-30 board cells

Written 2026-09-05, pre-registered. This is the vendor-falsification catalog for
the 30 measured board survivors (`docs/plan-2026-09-one-encoder.md` §2b) and the
crown engine's opponent set (CLAUDE.md STEP 3). Each card is a mechanism read
line by line from the vendor source, priced on the campaign's cost lattice, tied
to a named cell class, and shipped with its own cheapest falsifier and three
death modes — so a coder can act on any card in one turn and a failed lever
costs hours, not days (`docs/board/attack/sequence/seq-verdict-playbooks.md`).

Reading basis (honesty column): the `.worktrees/brain/vendor/*` submodule paths
are UNCHECKED-OUT stubs in this worktree. All line numbers below were read from
the physical checkouts at the repo root, which hold exactly the committed
submodule SHAs named in `docs/vendor-technique-index.md:22` (zopfli
`zopfli-1.0.3-9-gccf9f05`):

    vendor path used in citations      physical read location
    vendor/zopfli/src/zopfli/*         /Users/jackdanger/www/gzippy/zopfli/src/zopfli/
    vendor/gzip/{trees.c,deflate.c}    /Users/jackdanger/www/gzippy/gzip/
    vendor/rapidgzip/**                /Users/jackdanger/www/gzippy/rapidgzip/

Crown-engine facts assumed (all receipted elsewhere): the crown engine —
Step-3 `src/compress/deflate/parse/ultra/`, the zopfli-class near-optimal parse
— already beats zopfli on exact bytes 4/4 (CLAUDE.md:131-133) and holds the
Squishy crown 3.205795 @ -F 80 > ect-10009 (compressor-architecture.md:25). The
crown engine is therefore NOT the target of these cards; the target is the
numeric ladder (L1-L9), whose surviving size cells are the cadence band and the
hairlines. "ECT itself is not vendored" (vendor-technique-index.md:167) — the
ECT-shaped code that exists is ours.

## Cell classes being served (the named-30 survivors, verbatim from plan §2b)

* **Cadence band L2-L5 (vs gzip/pigz — zlib-family cadence)**: `access.log L5
  vs gzip +1.07%` / pigz +0.75%; `dd79_bin6 L2/L3 vs gzip/pigz` up to +0.93%;
  `photo.jpg L1-L3 vs gzip +0.04%` (legacy L1-arm class, not cadence). Sibling
  defect on the same files: `access.log L3 vs libdeflate +0.42%` (#364's
  far-len3 legacy arm, NOT in scope of these cards — #364 owns it). Priority:
  access.log, dd79_bin6, data.sqlite are GATE corpus rows
  (`corpus_split.json`); these two cells are the worst margins on the board.
* **Hairlines (≤0.2%)**: `minjs.min.js L5 +0.19/+0.16`; `data.sqlite L4
  +0.13/+0.17` (GATE); `weights.safetensors L7-L9 +0.00-0.02%`;
  `movie.mp4 L6 vs libdeflate +0.0008%` (minjs/movie are TUNE rows).

The governing citation for the whole card set — CLAUDE.md:126-128: the
Huffman-construction class is CLOSED at ~0.001% released margin, and "the
remaining candidates are **block BOUNDARIES or the parse** — which is where
`examples/blockspans` points too: gzip splits on a fixed ~34,000-symbol cadence
(cv=0.023) while our spans run 8x longer at cv=0.373."

## What "cost on the ldx lattice" means

The campaign's decomposed cost layers, each with a banked measured anchor. A
card's cost entry states WHICH layer it bills to and at what amplitude; levers
that bill to per-block layers are affordable, per-position/per-pass layers are
not (G16/G17 rule: "cost is per block, not per position").

| layer | measured anchor |
|---|---|
| per-pass FIXED (F) | +8.5 ms/pass our side vs libdeflate, ~2.2 cyc/byte (`docs/board/fixed-vs-pernode.md`) |
| per-node chain (P) | ours already 17.3% CHEAPER per node at matched depth (fixed-vs-pernode.md; it is a ms fit, not an Ir count — l2-instruction-attribution correction) |
| per-position observe | `block_split.rs` 33.0M Ir at T1 = 3.7%, FLAT in depth L2→L9 (`docs/board/l2-instruction-attribution.md:40,61`) |
| per-block header | 107.2 B/header measured (sil40: 913 blocks, 97,914 B = 0.634% of output), NOT the inferred 350 B (`target-encoder-and-gap-analysis.md` G27a:1792-1798) |
| per-FORCED-boundary misplacement | ~240 B per forced split (G27a:1802-1810) — 70% of seam cost is placement, not headers |
| per-split-search eval | O(span) per candidate split (each eval builds a full block cost) — zopfli pays this only outside position loops |
| per-pass DP sweep | zopfli squeeze = full `GetBestLengths` relaxation per iteration (squeeze.c:446-526) — the most expensive layer on the ladder |

---

## CARD VZ-1 — D4: exact-cost split search driven by a greedy pre-parse

- **Vendor**: zopfli, `vendor/zopfli/src/zopfli/blocksplitter.c` (all below); pipeline in `vendor/zopfli/src/zopfli/deflate.c`.
- **file:line**:
  - `ZopfliBlockSplit` :275-320 — the split-point search parses with `ZopfliLZ77Greedy`, with the standing comment "Unintuitively, Using a simple LZ77 method here instead of ZopfliLZ77Optimal results in better blocks." (:294-296).
  - `ZopfliBlockSplitLZ77` :215-273 — iterative splitter: `FindMinimum(SplitCost, lstart+1, lend)` :244; accept iff `splitcost <= origcost` else mark `done[lstart] = 1` :251-255; then `FindLargestSplittableBlock` :258-262 re-enters the LARGEST unsplittable interval (:195-213 — largest-first makes a capped block budget spread evenly before it exhausts); floors: bail below 10 tokens :225 and when the largest interval is < 10 tokens :263-265; `maxblocks` cap from `blocksplittingmax` (15, util.c:34).
  - `FindMinimum` :43-96 — 9-point sectioning (`NUM 9` :60) narrowing with a `best > lastbest` inflation break :84; exhaustive linear scan when the range is < 1024 :45-57.
  - `SplitCost` :125-128 — cost of a candidate boundary = `EstimateCost(left) + EstimateCost(right)`; `EstimateCost` :108-111 = `ZopfliCalculateBlockSizeAutoType` → the REAL best-of-3 bit count including headers (deflate.c:610-621, fixed-cost eval skipped when the store has > 1000 tokens :615-616).
  - Pipeline placement (deflate.c `ZopfliDeflatePart` :811-906): split :845-850 on the greedy store → per split span run `ZopfliLZ77Optimal` (the 15-iteration squeeze) :854-869 → SECOND splitting attempt on the merged OPTIMAL store, keep the cheaper of the two whole trajectories :871-893 → emit per final span via `AddLZ77BlockAutoType` :895-901, whose `expensivefixed` re-parses a small or fixed-friendly block under fixed-tree costs (`lz77->size < 1000 || fixedcost <= dyncost * 1.1`, deflate.c:760).
- **Mechanism**: three separable mechanisms in one file. (i) Boundary search runs on a DIFFERENT (greedy, cheaper) parse than final emission — block boundaries are chosen where the greedy token stream's information changes, not where the optimal parse wants them. (ii) The split cost is the REAL bit cost (headers included), so the splitter directly prices the ~107 B header / table-refit trade instead of proxying it. (iii) Whole-trajectory re-split: after optimization, re-splitting the optimized token stream is tried as an independent candidate and the cheaper trajectory wins — split search is a fixed-point, not a pre-pass.
- **Cost on the ldx lattice**: per-split-search eval layer. Each candidate boundary costs two full block-cost builds (O(span) each), amortized outside position loops; on the search grid only 9 evals per narrowing step. The hot ladder instead pays the D2 observation layer (33.0M Ir, flat). Adopting D4 on L2-L9 bills per-eval, not per-position — the affordable shape ONLY if candidate splits are capped/recursive rather than exhaustive.
- **Named cell class**: hairlines ONLY — `minjs.min.js L5 +0.19%`, `data.sqlite L4 +0.13%`, `movie.mp4 L6 +0.0008%`. NOT the cadence band: the measured ceiling for re-optimizing boundary placement is **0.126-0.133%** (`examples/split_headroom.rs` / `examples/proposer_recall.rs`, plan §7) — an order below access.log's +1.07%. Also claims the access.log-L3 far-len3 cell for #364, not for a splitter.
- **Our state**: the crown engine already carries this exact mechanism, uncapped and ECT-class — `parse/ultra/blocksplit.rs`: greedy-parse split points (`block_split` :428-441), dynamic-only `estimate_cost` per the "crown-secondary variant A" A/B note (:17-26), largest-splittable loop with the 100-token floor (:246-262), and the recursive exact-cost `RecursiveSplitter` (:296-424) which `deflate.rs` runs BOTH cost-surface trajectories of (`auto_type` dyn-only vs best-of-3) and keeps the cheaper (doc :282-295). It is unreachable from the numeric ladder; nothing below L10 carries it.
- **Decisive fulcrum grid**:
  ```
  # zero-build receipt stage (local, no fulcrum):
  examples/split_headroom.rs   # placement ceiling AT the failing cells, not sil40 L9
  examples/proposer_recall.rs
  examples/blockspans          # span cadence/cv on data.sqlite L4, minjs L5, movie L6
  # box stage (only if the ceiling re-derives >= the cell deficit):
  CAMPAIGN_FULCRUM=/root/fulcrum/target/release/fulcrum \
  CAMPAIGN_CORPUS_ROOT=/root/l3-paretogate/corpus \
  CAMPAIGN_OUT_ROOT=/root/www/gzippy-bench/campaign \
  fulcrum try <ref> --levels 4,5,6 --threads 1,4
  fulcrum why "data.sqlite L4" --ours BIN --rival-cmd 'rival -{level} -c {input}' --corpus /root/l3-paretogate/corpus
  scripts/campaign/board-size.sh all   # promotions judge GATE only
  ```
- **Falsifier (in order)**: (1) re-derive the 0.126-0.133% ceiling at the FAILING cells themselves — the banked ceiling was measured on sil40/dickens at L9; do not generalize (hard stop). If ceiling < cell deficit, the lever dies without a build. (2) If live: a D4-cadence arm scoped to the hairline levels. (3) `won_cells_stay_won` + roundtrip + full census.
- **Death modes**: (a) **ceiling death** — re-derived placement ceiling < the hairline deficit at every named cell ⇒ the class is closed by receipt, card retires; (b) **tie-block risk** — L2-L9 ties libdeflate on 154-170 cells; any split-point change re-rolls ALL of them; if any tied cell re-opens harder than the hairlines close, clause 3 kills it; (c) **wall** — if the 9-point search bills more than the block_split 33.0M-budget at hot levels, clause 5 kills; the lever is then recorded crown-only (already shipped there) and closed at both coordinates per plan §8's stop rule.

## CARD VZ-2 — iterated squeeze with real-Huffman re-anchoring (the "parse" half)

- **Vendor**: zopfli, `vendor/zopfli/src/zopfli/squeeze.c`.
- **file:line**:
  - `ZopfliLZ77Optimal` :446-526 — the iteration driver: initial `ZopfliLZ77Greedy` :481, then per iteration: `LZ77OptimalRun` :489-491 (DP re-solve under `GetCostStat`), re-price with the REAL oracle `ZopfliCalculateBlockSize(..., 2)` :492, keep best store+stats :496-501, refresh stats from the actual emitted symbols :503-504, blend `1.0*current + 0.5*last` once randomization started :505-511, on a cost plateau after iteration 5 reset stats to best and randomize 1-in-3 freqs (MWC, fixed seed m_w=1/m_z=2 :84-101, :512-517). No early exit ever.
  - Cost model :146-157 — `GetCostStat` = entropy costs (`ZopfliCalculateEntropy`, tree.c:71-94: zero-count symbols priced `log2(sum)` — "the symbol will appear at least once anyway"), NOT code lengths.
  - DP :217-309 (`GetBestLengths`) — forward relaxation over every length edge 3..258 with sublen distances, `GetCostModelMinCost` precompute + skip `costs[j+k] <= mincostaddcostj` :287-293, literal edge :277-284, `ZOPFLI_SHORTCUT_LONG_REPETITIONS` skip for `same > 2*258` runs :251-271.
  - Emit with re-anchoring `FollowPath` :338-389 — walks the path and re-runs `FindLongestMatch(limit=length)` to recover the REAL distance for the chosen length (:366-373), i.e. the DP's assumed distances are re-verified against the matchfinder at emit time.
- **Mechanism**: the parse is not one pass — the cost model is *updated from the output of the previous parse* and the shortest path re-solved; the emit pass re-anchors against the real matchfinder. This is the "parse" half of "block BOUNDARIES or the parse" — and it is where zopfli's remaining 1-3% lives beyond the splitter.
- **Cost on the ldx lattice**: per-pass DP layer — the most expensive layer on the ladder. One extra full-parse pass ≈ doubles the parse wall at a hot level; the exact-Huffman receipt prices a similar per-block refinement at 10-14% wall for 0.001% (CLAUDE.md:117-126). On the crown engine this layer is already banked as the Squishy winner; on L2-L9 it is unpayable at full shape.
- **Named cell class**: hairlines only, and only their L5+ end — `minjs.min.js L5 +0.19%`, `weights.safetensors L7-L9 +0.00-0.02%`. Expected AMPLITUDE is the honest problem: at hot levels the near-optimal parse is already within ~0.1% of the squeezed optimum (the crown's own iterates move geomean by ~0.001-0.02% per doubling of work, compressor-architecture §3 Stage-D row).
- **Our state**: `parse/near_optimal.rs` runs libdeflate's single-pass DP with integer `costs.rs` BIT_COST; no re-anchor loop. The crown (`parse/ultra/squeeze.rs`) carries the full port PLUS multi-seed restarts (squeeze.rs:1099-1118) and the vectorized cost sweep (:516-706) — beyond what zopfli has.
- **Decisive fulcrum grid**:
  ```
  # amplitude pre-check FIRST (the lever lives or dies on amplitude, not mechanics):
  examples/dump_fixtures.rs / blockcensus on minjs @ L5 + weights @ L7-9:
  what fraction of a block's bits are token choices (re-parseable) vs headers?
  # box stage, scoped:
  fulcrum try <ref> --levels 5,7 --threads 1,4     # a 2-iteration re-anchor, capped
  fulcrum anatomy (paired valgrind) — the Ir bill may be counted once, never quoted as wall
  ```
- **Falsifier**: (1) amplitude probe above — if re-parseable bits are < 2x the failing margin, the lever cannot reach the cells even at infinite wall; close without building. (2) If live: 2-iteration re-anchor at L5/L7 behind `level.rs`, `ladder_is_monotone_t1` + `won_cells_stay_won` + census. (3) paired T1/T4 wall on the frozen box.
- **Death modes**: (a) **amplitude death** (probe shows token-choice bits < 2x margin at the named cells); (b) **wall death** (one extra parse pass fails clause 5 at L5's already-thin wall margin — the fixed +8.5 ms F-layer makes any per-pass addition maximally visible); (c) **per-label flip death** — re-anchoring is not monotone per file: a better parse on minjs can cost bytes on sil40-class files; census catches it, but a flip on any won GATE cell reverts.

## CARD VZ-3 — greedy pre-parse score: far-distance length penalty (`GetLengthScore`)

- **Vendor**: zopfli `vendor/zopfli/src/zopfli/lz77.c`; gzip analog `vendor/gzip/deflate.c`.
- **file:line**: `GetLengthScore` lz77.c:265-271 — `score = length - (distance > 1024 ? 1 : 0)`, with the upstream warning that it feeds (a) the len-3-far exclusion, (b) lazy tie-breaking (comparison `lengthscore > prevlengthscore + 1` at :583-590 — the +1 bias is deliberate), and (c) INDIRECTLY the block-split points and the first squeeze seed (:256-263). Greedy consumer loop `ZopfliLZ77Greedy` :544-630; `ZOPFLI_MAX_CHAIN_HITS 8192` early quit :527-530; the `same[]` run-jump inside `FindLongestMatch` :481-490. Gzip: `TOO_FAR 4096` :129-132 with the len-3 collapse at :711-717 (deflate() lazy path); `max_insert_length` skip in deflate_fast :632-651.
- **Mechanism**: one bit of length, one branch — long-distance short matches are demoted in the SCORING parse only. Not an output decision by itself; it shifts which tokens exist when the splitter and the first squeeze read the store, which is exactly how it moves block boundaries "in a rather unpredictable way" (lz77.c:260-261).
- **Cost on the ldx lattice**: per-node layer, ~zero wall (one comparison per accepted match). The cheapest card in the deck that can still move bytes.
- **Named cell class**: admission — the visible far-len3 cell (`access.log L3 vs libdeflate +0.42%`) is **#364's lever, already queued** (plan §2b + handoff §4.4); this card does NOT duplicate it. What #364 does not own: the cadence band's L5 cell could carry a far-match component via this scorer — and it all rides on the #364 verdict, per its own merge-order comment.
- **Our state**: #364 (`lever/ldx-len3`) ports the far-len3 machinery to the legacy arm — pending checklists in its body; the port side of `TOO_FAR`/pick-min semantics lives in `src/compress/ldx/` (greedy/lazy arms) with the 4096/8192 gates inherited from libdeflate (`:2573-2575`, `:2666-2668` legacy arms).
- **Decisive fulcrum grid**:
  ```
  fulcrum try pr-364-rebased --levels 3,6,7 --threads 1,4     # #364's own scoped grid first
  fulcrum why "access.log L5" --ours BIN --rival-cmd 'gzip -{level} -c {input}' --corpus /root/l3-paretogate/corpus
  # only on a #364 SHIP: extend the gate to --levels 5 and re-census
  ```
- **Falsifier**: #364's verdict governs the class; a second lever here may not be built while #364 is open (land-gated-win-first, plan §8.1).
- **Death modes**: (a) #364 lands and the gate is measured at ~0 on access.log L5's byte accounting (`fulcrum why` token diff shows no far-len3 mass) ⇒ card closed as mechanism-not-present-at-THIS-cell; (b) wall: the gate is ~free, so this card does not die on wall; it dies on (c) **no-amplitude**: +0.42% is split across boundary placement AND token choice; if `fulcrum why` attributes the cell mostly to the grid, the token-side lever is dead and the card folds into GZ-1/GZ-2's cadence cards.

## CARD VZ-4 — katajainen exact length limiting — VOID (closed class, receipt card)

- **Vendor**: zopfli, `vendor/zopfli/src/zopfli/katajainen.c`.
- **file:line** (so nobody re-reads it): `ZopfliLengthLimitedCodeLengths` :172-261; `BoundaryPM` :69-101 (recursive two-chain step; `2n-4` runs), `BoundaryPMFinal` :103-119; run count `2 * numsymbols - 4` :250-254; stable sort by `weight<<9 | count` :222-235; flat node pool `maxbits * 2 * numsymbols` :242; error `(1 << maxbits) < numsymbols` :202; degenerate 1/2-symbol cases :206-220; `ExtractBitLengths` :143-163.
- **Mechanism** (for the record): exact length-limited Huffman via boundary package-merge — provably optimal given the constraint; gzippy has had this ported since the crown campaign: `src/compress/deflate/huffman/optimal.rs:268` (`calculate_bit_lengths`) + the RLE-aware shaper `OptimizeHuffmanForRle`/`TryOptimizeHuffmanForRle` port (:419-466, from zopfli deflate.c:434-560), used by ultra + deflate64 only; hot levels use the libdeflate demotion heuristic (`huffman/fast.rs:279`).
- **Cost on the ldx lattice**: per-block, exact — the affordable shape — but the AMPLITUDE is measured at ~0.001%.
- **Named cell class**: none — this card exists to keep the class closed. The 0.01% the zero-headroom seam cells needed (CLAUDE.md:102-115) is 10x the released margin.
- **The binding receipt (cite, do not re-derive)**: CLAUDE.md:117-126 — the dual-candidate was built BOTH ways and measured: unconditional swap = a wash that OPENS cells; costed dual candidate = 49/49 smaller invariant at ~0.001% margin, +10-14% wall, flips sil40 L6 pass→fail. libdeflate's heuristic limiter is already within 0.001% of exact (`deflate_compress.c:1022-1090`). plan §7: "no Huffman-construction levers (exact package-merge dual-candidate closed at ~0.001% vs ~0.01% needed)". In-code FALSIFY notes were deleted wholesale 2026-08-01; the git history + these two lines ARE the receipt.
- **Decisive fulcrum grid**: none — do not spend one.
- **Falsifier**: none needed; reopening requires a NEW mechanism that projects ≥ 0.1% (the whole-hairline-class order), not a faster package-merge.
- **Death modes** (all three already fired, kept so the card dies honestly): (a) amplitude 0.001% « 0.01% needed; (b) 10-14% wall = clause-5 fail; (c) unconditional variant flips won cells (wash that opens).

## CARD GZ-1 — gzip D3: the 4096-symbol estimated-cost early flush

- **Vendor**: gzip, `vendor/gzip/trees.c` (zlib lineage inherits equivalents; zlib-ng lit budget 16384).
- **file:line**: `ct_tally` budget + early-flush block, trees.c:991-1006: trigger `level > 2 && (last_lit & 0xfff) == 0` (:992) — every exactly-4096th symbol; estimate `out_length = last_lit*8L + Σ_d dyn_dtree[dcode].Freq*(5L+extra_dbits[dcode])` :994-999 (an upper bound: data at 8 bits/symbol, distance table only, NO litlen tree term); flush iff `last_dist < last_lit/2 && out_length < in_length/2` :1004. Elsewise the block ends at the hard symbol budget `last_lit == LIT_BUFSIZE-1 || last_dist == DIST_BUFSIZE` :1006, `LIT_BUFSIZE 0x8000` :118-131. Note the `in_length` numerator is `strstart - block_start` (window positions, :995).
- **Mechanism**: a content-INDEPENDENT trigger (periodic 4096-symbol checkpoint) with content-derived GATES (match sparsity + a halve-or-better profit estimate). Two consequences: block spans stay near-periodic on match-dense files (the 32,767-symbol budget) and get flushed EARLY, at 4,096-multiples, wherever matches are sparse — which is how gzip's cadence is both cheap and adaptive with zero per-position observation cost.
- **Cost on the ldx lattice**: cold — one mask-compare per tally plus the check every 4,096; the bytes move through the BLOCK layer (header mass 107 B + span-fit tables), not through per-position work. The estimate itself reuses state the tally loop already updates (`dyn_dtree` freqs), so there is no counter sweep added.
- **Named cell class**: cadence band — `access.log L5 +1.07/+0.75`, `dd79_bin6 L2/L3 +0.93`. These are the gzip/pigz-family cells: the opponent runs NOTHING like D2 (vendor index D3 note "we have D2 instead, which subsumes it" is contradicted by this measured band — "subsumes" was true against libdeflate, untested against zlib-cadence).
- **Our state**: absent. Our hot arm carries D2 (`src/compress/deflate/block_split.rs` port + `src/compress/ldx/split.rs` transliteration with the blockspans finding quoted in its header). The D3 shape would be a new, level-scoped arm in `level.rs`'s split policy.
- **Decisive fulcrum grid**:
  ```
  # stage 0 — receipts, all local, zero build risk:
  examples/blockspans          # span cadence + cv on access.log/dd79_bin6 L2/L3/L5, ours vs gzip
  examples/blockcensus         # BTYPE + header mass on the same cells
  # stage 1 — cell accounting:
  fulcrum why "access.log L5"  --ours BIN --rival-cmd 'gzip -{level} -c {input}' --corpus /root/l3-paretogate/corpus
  fulcrum why "dd79_bin6 L3"   --ours BIN --rival-cmd 'gzip -{level} -c {input}' --corpus /root/l3-paretogate/corpus
  # stage 2 — lever, arm-gated to the cadence band via level.rs (NO content detector):
  fulcrum try <ref> --levels 2,3,5 --threads 1,4
  scripts/campaign/board-size.sh all
  ```
- **Falsifier** (in order): (1) blockspans on the two GATE band cells: if our spans are ALREADY ~34K-cadence there (the 8x/cv=0.373 finding is a text-file artifact), the mechanism story dies and nothing is built — that is the cheapest kill in the deck; (2) `fulcrum why` must show the deficit lives in the block grid (header + table-fit mass), not in token choice — otherwise this card dies and the cell belongs to VZ-3/parse cards; (3) the D3-flush arm scoped to L2/L3/L5, judged by `fulcrum try --threads 1,4`, then GATE census + `won_cells_stay_won`.
- **Death modes**: (a) **theory death**: the byte accounting shows no grid component in these cells' deficits (same block count, same boundaries; delta all in tokens) — lever AND its class closed for the cadence band; (b) **size death**: cadence arm closes none of the two band cells or opens any GATE cell it currently wins (sil40-class libdeflate-tie files sit at L2-L5 too — the tie block is 154-170 cells; care not courage); (c) **wall death**: D3's checkpoint is nearly free, so the real wall risk is the CHANGED SPLIT COUNT changing emit cost — paired T1/T4 wall must stay inside clause 5 on every cell the arm passes.

## CARD GZ-2 — gzip D1: the 32,768-symbol budget block termination

- **Vendor**: gzip, `vendor/gzip/trees.c`, `deflate.c` FLUSH_BLOCK plumbing.
- **file:line**: `LIT_BUFSIZE 0x8000` = 32,768 symbols (trees.c:118-131, with the design rationale list :132-151 — note reason 4, "fast adaptation... smaller buffer sizes transmit trees more frequently" — the trade is a stated design goal, not an accident); flush condition :1006; `flush_block` consumer :859-957 (its 3-way type choice :887-946 includes the whole-file stored rewrite on seekable outputs :902-913); deflate.c: `FLUSH_BLOCK` :586-588, tally path `ct_tally` :963-1006.
- **Mechanism**: block length as a pure RESOURCE BUDGET (symbol count), not a heuristic — zlib-family discipline. Severs coding from content detection: every ~32,767 symbols a new block/baseline, regardless of content. Against a periodic opponent (log cadence, sar tables, .sqlite page cycles), frequent re-anchoring of the code tables buys more table-fit than the extra ~107 B headers cost.
- **Cost on the ldx lattice**: per-block layer only. Changing the budget changes block COUNT; on sil40-class files at our D2 span (~8x) this would MULTIPLY header mass (0.634% at 913 blocks already) — the budget is only sane where the opponent came from the same budget.
- **Named cell class**: cadence band (same two cells as GZ-1; D1 and D3 are one joint lever — gzip's cadence is D1 budget + D3 early flush; either alone is not the opponent).
- **Our state**: D2 SAD + `SOFT_MAX_BLOCK_LENGTH 300000` / `SEQ_STORE_LENGTH 50000` budget backstops (`parse/mod.rs:65,71`, vendor index D1) — we already HAVE the budget half at 300K/50K, 9x looser than gzip's; the gzip-shape move is 32,768-symbols-at-L2-L5-scoped, which the literal reading of non-negotiable #3 permits (parameter table keyed on level, never on content).
- **Decisive fulcrum grid**: same three stages as GZ-1 (blockspans → fulcrum why → scoped `fulcrum try --levels 2,3,5`), plus the T>1 interaction leg:
  ```
  scripts/campaign/parity-census.sh          # budget change perturbs every chunk seam
  fulcrum trace critpath                     # emit/SF burst changes are thread-visible
  ```
- **Falsifier**: (1) GZ-1's stage 0/1 receipts first — D1 without the accounting is exactly the "seam-shrinking lever without headroom" anti-pattern CLAUDE.md:111-115 bans; (2) per-label bar at L2/L3/L5 vs gzip AND pigz AND libdeflate simultaneously (the budget must not lose to libdeflate to gain on gzip); (3) census + parity-census + roundtrip.
- **Death modes**: (a) **warning-shot death**: blockspans shows the cadence gap is a TEXT-file phenomenon only and absent on dd79_bin6 (binary) — one band cell unexplained ⇒ mechanism incomplete, do not ship a half-theory; (b) **tie-block death** (any libdeflate-tied cell re-opens — clause 3 absolute); (c) **seam death**: T>1 output moves (more blocks per chunk, seams re-land) — parity-census fails or the seam class re-opens; per plan §8 the lever then records closed-at-both-coordinates.

## CARD GZ-3 — gzip `configuration_table` depths + the good_match chain-halving

- **Vendor**: gzip, `vendor/gzip/deflate.c`.
- **file:line**: `configuration_table[10]` :242-254 — {good, lazy, nice, chain}: L4 {4,4,16,16}, L5 {8,16,32,32}, L6 {8,16,128,128}, L7 {8,32,128,256}, L8 {32,128,258,1024}, L9 {32,258,258,4096}; the ASYMMETRIC budget: `if (prev_length >= good_match) chain_length >>= 2` :404-407 — a cheap-running match REDUCES the search, so depth is spent only where it is not already paying; the lazy gate `prev_length < max_lazy_match` :700-702; the lazy loop :689-764; `deflate_fast` insert-skip for long matches :632-646; `nice_match` early-exit :482.
- **Mechanism**: depth is a BUDGET SPEND, spent only where the current match is not already good — at L5 gzip buys chain 32 for hopeless positions and 8 for matched ones, in one `if`. Our L5 is libdeflate's `Lazy(16,30)` flat (vendor-diff table, l3-vendor-diff.md). The handoff order stands: "dd79_bin6 L2/L3 + access.log L5 is the gzip-cadence band (L4/L5 depth/knob class); price the `good_match` extension to L4/L5 after the wall leg decides the pipeline" (handoff-2026-09-04.md §4.5).
- **Cost on the ldx lattice**: per-node (P) layer — measured ours 17.3% CHEAPER per node (fixed-vs-pernode), so a budget RE-shape spends a cheaper unit; the F-layer (+8.5 ms) is untouched. The wall bill is near-neutral; the risk is bytes.
- **Named cell class**: L4/L5 depth/knob class of the cadence band (`access.log L5 +1.07`, `dd79_bin6 L2/L3` adjacent). NOT the hairlines: they are at depth 16-65 where this knob's bytes are flat.
- **Our state**: `level.rs` L4/L5 = libdeflate's presets; the #363 `good_match` port exists only at L6/L7 ({128,65,8} / {256,130,32}) on the legacy arms — the L4/L5 extension is unbuilt and explicitly deferred until the wall leg lands it first.
- **Decisive fulcrum grid**:
  ```
  fulcrum try <ref> --levels 4,5,6,7 --threads 1,4     # after the #363 wall leg verdict
  ladder_is_monotone_t1 + won_cells_stay_won           # the Phase-3 gate pair (plan §5)
  fulcrum why "dd79_bin6 L3" --rival-cmd 'pigz -{level} -c {input}' --corpus /root/l3-paretogate/corpus
  ```
- **Falsifier**: (1) the L4/L5 arms must beat gzip AND stay ≤ libdeflate (per-label bar is per-rival); measured once at `fulcrum try` on the frozen box; (2) `ladder_is_monotone_t1` — L4 < L5 < L6 monotonicity must hold (the KNOWN_SAGS `(tabular,3)/(binary,3)` guard); (3) census: the 154-170-cell tie block at L2-L5 must not move (it is libdeflate-parity, and a `good_match`-shaped knob on the port breaks byte-identity everywhere by design — so this lever CANNOT be shipped anywhere libdeflate parity holds, which is the same admission GZ-2 makes).
- **Death modes**: (a) **parity death** (most likely): any L4/L5 depth change breaks the libdeflate byte-tie block — the lever is only viable if it wins BOTH the gzip cells AND net-new vs libdeflate, which the vendor-diff table says is impossible at identical params ⇒ the honest form is scoped to the two failing cells' levels with a full tie-block re-check, and dies if any tie opens; (b) **ladder death**: L4/L5 scalar changes re-open the KNOWN_SAGS (monotonicity constraint, plan §5); (c) **wall death**: shallow-good matches SHORTEN chains (fast) but shift the LAZY decision surface — T1 wall must not regress on tabular/binary class cells it currently passes.

## CARD RG-1 — ECT-shaped material in `vendor/rapidgzip`: none — honest load-bearing negative

- **Vendor**: rapidgzip (the DECODE-optimized fork), `vendor/rapidgzip/**`.
- **file:line receipts for the negative**: tree-wide grep for `LzFind|MatchFinder|matchFinder|lzFind` returns NOTHING; the only compressor delegation is `compressWithIsal` (`librapidarchive/src/rapidgzip/gzip/isal.hpp:589-606`), which drives ISA-L itself (igzip, hash-chain class — its own catalog, not this one); everything else is decode machinery (`GzipBlockFinder`, `huffman/`, `chunkdecoding/`, `MarkerReplacement.hpp`, `WindowMap.hpp`). rapidgzip is a parallel READER; its vendored `external/isa-l` is bundled for decode too.
- **Consequence for the deck**: "anything ECT-shaped here" has exactly one real reference — OUR OWN crown-engine matchfinder `src/compress/deflate/matchfinder/lzfind.rs`, the ECT/7-zip `Bt3Zip`-class port: CRC-table 3-byte hash (`:136` + table :63-78), cyclic tree update, FULL Pareto frontier, one `(len, dist)` per length `[3, 258]` ⇒ ≤ 512 packed `u16` pairs (`:180` doc, `:46`), used ONLY by ultra's squeeze DP driver. Its in-repo falsifier is the brute-force frontier reference + property test (`frontier_random` / `frontier_dna_4symbol` / `assert_frontier`, lzfind.rs :343-420) — any ECT-shape change to the matchfinder is falsified locally in seconds without the box.
- **Literal-run decisions, noted as instructed**: the vendored rapidgzip has NO encoder-side literal-run decision surface (no matcher, no greedy DP): its literal-run handling is decode-side — copy-loop batching inside the huffman/deflate readers and the marker-replacement stored-vs-decoded block choice (`MarkerReplacement.hpp`) — a decode artifact, and recommends nothing for the encoder. The encoder-shape analog of "literal-run decisions" the crown engine actually carries is zopfli's run machinery: the `same[]` run-jump inside the chain walk (lz77.c:481-490) and the DP-side `ZOPFLI_SHORTCUT_LONG_REPETITIONS` forced-258 sweep (squeeze.c:251-271) — both already ported (`parse/ultra/hash.rs`, `parse/ultra/squeeze.rs`).
- **Named cell class**: none directly. This card's function is to keep the campaign honest about WHERE the ECT mechanisms live: crown engine only; the numeric ladder has no ECT-shaped surface, and falsifying "rapidgzip holds an encoder technique" costs nothing.
- **Decisive fulcrum grid**: none. The only adjacent receipt is the crown-vs-ECT score line (3.205795 @ -F 80 > ect-10009, compressor-architecture.md:25) and CLAUDE.md STEP 3's 4/4.
- **Falsifier**: if anyone proposes an ECT-shaped lever "from vendored rapidgzip", require the file:line first; the grep above is the receipt that none exists.
- **Death modes**: (a) card is already in its terminal state — a negative; (b) it dies (upgrades to a mechanism card) only if the vendored repo ever swaps to an ECT-encoder lineage or a vendored ECT lands in `vendor/` — at which point THIS catalog must be re-read, not TRUSTED; (c) anyone citing rapidgzip as ECT precedent for the LADDER is committing a vault error — the crown engine, not this vendor, is the ECT analog.

---

## Cross-references and receipts indexed

- The named cells (plan §2b, measured 2026-09-04, artifact
  `/root/www/gzippy-bench/campaign/size-all-ee0c1d2c/census.json`): 30 failing
  of 1,320; the six classes quoted in the header; full ranked list in plan §2b.
- Instrument map: plan §7 table — `board-size.sh tune|all`, `fulcrum try`,
  `fulcrum why`, `fulcrum candidates`, parity-census.sh, examples/blockspans,
  blockcensus, split_headroom.rs, proposer_recall.rs, chunkgrid.rs.
- `fulcrum` grid syntax: CLAUDE.md:299-303 (`why <cell> --ours BIN
  --rival-cmd 'CMD -{level} -c {input}' --corpus F`; `try <ref> --threads
  1,4` defaults to T1 — always pass `--threads 1,4`). Box facts: handoff §5
  (solvency = root@10.0.2.240, corpus `/root/l3-paretogate/corpus`, fulcrum
  `/root/fulcrum/target/release/fulcrum`).
- Cost-layer receipts: fixed-vs-pernode.md (F=+8.5 ms; retraction discipline),
  l2-instruction-attribution.md (block_split 33.0M Ir flat; ms-vs-Ir rule),
  target-encoder-and-gap-analysis.md G27a (headers 107.2 B; misplacement ~240
  B/boundary), plan §7 (placement ceiling 0.126-0.133%), CLAUDE.md:117-126
  (Huffman-construction closure 0.001%), CLAUDE.md:131-133 (crown 4/4 vs
  zopfli).
- Vendor technique anchors: `docs/vendor-technique-index.md` D1 (:267-269), D3
  (:277-279), D4 (:281-283), E3 (:307-310), E5 (:317-318), E6 (:320-322), M11
  (:247-248), M13 (:203-207), P16/:985-502 rows for the squeeze cluster —
  this catalog's line numbers were re-verified against the physical checkouts
  and differ ONLY where noted inline (e.g. the gzip whole-file stored rewrite
  sits at trees.c:902-913 in this checkout).
- Standing rules that bind every card: valid gzip roundtrip (3 decoders);
  per-label bar; no env knobs, no content detection — level-keyed parameter
  tables only; wall outranks size; a closed cell stays closed; land gated work
  first; verify the binary you measured (CLAUDE.md non-negotiables + plan §8).
