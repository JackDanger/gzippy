# The gzippy Unified Compressor — Architecture

Status: DESIGN (2026-07-21). Owner: campaign supervisor. Basis: the full two-engine
inventory (technique catalog + duplication map, measured provenance for every piece).

## 1. What exists today, and why it must become one thing

gzippy carries two complete DEFLATE encoders plus fragments of a third:

- **Level engine** (`src/compress/deflate/`, ~4.8k LOC): libdeflate-faithful hc/bt
  matchfinders and greedy/lazy/near-optimal parses, igzip-derived fast path (L0/L1),
  and the SF-chain speed work (sequence emit, inline tables, copy elimination,
  prefetch). Byte-identical output to libdeflate at L2-12. Serves L0-12, T1 and T>1.
- **Crown engine** (`src/backends/zopfli_pure/`, ~4.7k LOC): zopfli port carrying the
  ECT-grade wins — LzFind BT4 full-Pareto matchfinder, multi-seed iterated squeeze
  with real-Huffman re-anchoring and ultra refinement, uncapped exact-cost recursive
  block splitter, 4-way RLE-aware table shaping, katajainen exact length-limiting.
  Holds the Squishy crown (3.205795 @ -F 80 > ect-10009). Reached ONLY via -F/-I/-J.
- Fragments: `compress/deflate64.rs` (CLI-unreachable third encoder), three gzip
  wrappers, two stored-block emitters, dead routing predicates, stale API docs.

Every technique in both engines is individually measurement-gated. The duplication
is pure history: the engines were built in separate campaigns. One compressor with
one policy table expresses all of it with less code and no whack-a-mole.

## 2. Target shape

```
src/compress/deflate/
  mod.rs            — entry points (oneshot, block, streaming w/ dict + is_last,
                      in-place padded, gzip wrap). ONE gzip wrapper (GzipHeaderInfo).
                      ONE stored-block emitter.
  level.rs          — THE single policy table: L0..L12 + Ultra(iterations) →
                      {matchfinder tier, parse tier, splitter tier, huffman tier,
                       cost model, knobs}. All tuning lives here. Nothing else
                      switches on level anywhere in the crate.
  bitstream.rs      — the branchless 64-bit word BitWriter (SF1-D grade), sole bit
                      emitter for every tier. Gains the MSB-first canonical-code
                      helper (add_huffman_bits) and cross-call bit-position
                      threading the master-block chain needs.
  tables.rs         — RFC constant tables (already shared).
  huffman/
    fast.rs         — libdeflate approximate length-limited builder (hot levels).
    optimal.rs      — katajainen exact length-limited + RLE-aware count shaping
                      (2-way search-time, 4-way final-emit). Ultra + anywhere a
                      final block can afford exactness.
    header.rs       — ONE dynamic-header (precode/RLE) builder + emitter.
  matchfinder/
    common.rs       — hash fns, lz_extend, rebase, prefetch (already shared).
    ht.rs           — chainless single-probe (igzip class). L0-1.
    hc.rs           — hash chains (libdeflate class). L2-9.
    bt.rs           — libdeflate BT (single-best). L10-12 (byte-identity keeper).
    lzfind.rs       — ECT BT4 full-Pareto frontier. Ultra (squeeze DP driver).
    (bt.rs and lzfind.rs are siblings behind one trait; converging them is a
     gated follow-up, not part of the structural move — see §5 Stage D.)
  parse/
    mod.rs          — Seq/Sink sequence representation + emit_block (SF1-D), the
                      shared emit substrate for fast/greedy/lazy/near-optimal.
    fast.rs greedy.rs lazy.rs near_optimal.rs — as today.
    ultra/          — the crown pipeline as the highest parse tier:
      lz77.rs       — LZ77Store token rep (DP/slice-optimized; distinct from Seq
                      by design — different access shape, both documented).
      squeeze.rs    — multi-seed iterated squeeze + real-Huffman re-anchor + ultra
                      post-loop (fix2/fix3), LzCache replay.
      blocksplit.rs — exact-cost greedy + recursive splitter (uncapped, 5MB
                      masters), refine/re-squeeze (single-trajectory twice pass).
      cost.rs       — f64 CostModel (Fixed/Stat) + real-length anchoring.
  block_split.rs    — streaming SAD heuristic (cheap tier, L2-12 inline splitting).
  costs.rs          — integer BIT_COST model (near-optimal DP).
```

`src/backends/zopfli_pure/` ceases to exist. `ZopfliGzEncoder` folds into the main
entry points; `-F/-I/-J` become knobs that select the Ultra tier through level.rs
(and `--ultra`/`-11+` routing gets ONE documented meaning). `deflate64.rs` stays a
separate format module (it is a different format, not a rival implementation), but
its matchfinder/huffman internals become thin instantiations of the shared modules
in a later cleanup.

## 3. Which duplicate wins, and why (from measured properties)

| Concern | Winner | Rationale (measured) |
|---|---|---|
| Bit emission | level engine's word BitWriter | crown's own doc: bit-emit speed is irrelevant to squeeze wall → adopting the fast writer is pure simplification; must preserve bp threading + MSB-first helper. Gate: crown outputs byte-identical. |
| Huffman (hot) | libdeflate approximate | carries L2-12 byte-identity. |
| Huffman (exact) | katajainen + RLE shaping | provably optimal + crown-gated; becomes available (not default) to other tiers. |
| Dynamic-header wire format | BOTH (`huffman::header` + ultra's `deflate`/`deflate_size`) | **residual duplication, recorded not merged (Stage C, 2026-07-21).** Both emit the identical RFC-1951 precode/RLE wire format (`compute_precode_items`/`build_dynamic_header`/`DynamicHeader::emit` in `huffman/header.rs` vs `build_rle`/`encode_tree_emit`/`add_dynamic_tree` in `parse/ultra/{deflate_size,deflate}.rs`), but their COST-ACCOUNTING shapes differ: the level engine always tries all 3 RLE flags and trims precode length by codeword length; ultra re-walks all 8 use-16/17/18 combinations per block via `build_rle`'s shared size/emit walk and threads `hlit`/`hdist`/`hclen` through histogram-driven counts. Force-merging would couple that cost-accounting difference across engines for a wire-format match that is already achieved independently — not attempted this stage. `huffman/header.rs`'s module doc carries the pointer; unmerge is intentional, not oversight. |
| Splitting (hot) | SAD streaming | O(1)/token, carries byte-identity L2-12. |
| Splitting (exact) | uncapped recursive exact-cost | crown-gated (+0.00116 geomean lever). |
| Matchfinder seed for squeeze | **legacy zopfli hash/lz77 chain finder (KEPT, measured 2026-07-21)** | Stage D part 2 built an hc-matchfinder-driven greedy replacement (`greedy_hc.rs`, L9 depth/nice-len) for both consumers (squeeze seed + block-splitter greedy pre-pass) and scored it: on a fresh own-binary F15 baseline of 3.205250 (19-file geomean, this box/corpus), the replacement scored 3.204923 (FAILS the ≥-baseline bar); the spliced F80 estimate (8 fresh + 11 `squishy_f80_local.jsonl` reference) scored 3.205685 vs the 3.205752 ect-10009-crossing bar (also FAILS). Both misses are small — inside the ~0.001 geomean noise this codebase has documented elsewhere for repeat runs — but the pre-registered rule carried no noise margin, so the revert trigger fired on both legs; wall was also not clearly faster (contaminated by one outlier file, see commit message). Per the pre-registered rule: **reverted in full** (`greedy_hc.rs` deleted, `hash.rs`/`cache.rs`/`lz77::find_longest_match`/`lz77_greedy` restored) — the legacy finder earns its LOC. Part 1 (the matchfinder tier module doc + the `LzMatch` vocabulary-type move) shipped independently; it does not depend on this outcome. Re-open trigger: a future measurement wants to retry with a deeper HC search or a genuine multi-run significance test (N≥7) to separate the ~0.02-0.1% miss from noise. |
| Token reps | BOTH (Seq + LZ77Store) | different access shapes serve different DPs; forcing one rep is elegance-by-deletion, not by design. Documented side by side. |
| Gzip wrapper / stored blocks | mod.rs single implementation | three wrappers → one; framing is format law, not strategy. |

## 4. Invariants (the gates every stage must pass)

1. **L2-12 byte-identity**: outputs at L2-12 (T1 and T>1, both 6MB corpora + 4
   ratio-corpus files) byte-identical before/after every stage.
2. **Crown score**: 19-file Squishy geomean at `-F 80` ≥ 3.205795, all roundtrips.
   (Byte-identity additionally asserted for stages meant to be inert, e.g. B.)
3. **Loss-map no-regress**: spot paired cells (igzip L1/L2, ldgz L6, pigz L6) on the
   box for any stage touching hot paths.
4. **Full suite green** incl. 3-oracle differentials + proptest + zopfli fixtures
   (fixtures re-pinned ONLY when a stage intentionally changes ultra output, with
   the score gate as the license).
5. One-branch discipline: each stage is one PR on feat/pure-rust-encoder.

## 5. Migration stages (each independently shippable, gated, revertable)

- **A. STRUCTURAL MOVE (no behavior)**: relocate zopfli_pure → parse/ultra/ +
  routing/level unification (-F/-I/-J through level.rs; delete use_zopfli dead code,
  ZopfliGzEncoder's private header writer, gzip.rs/zlib.rs test-only wrappers,
  blocksplittingmax dead field; fix stale lib.rs/cli.rs docs). Gate: byte-identity
  EVERYWHERE (L0-12 + ultra), tests green. Pure `git mv` + import surgery.
- **B. ONE BITSTREAM**: ultra emits through the word BitWriter. Gate: crown outputs
  byte-identical (emission is representation-independent), score unchanged.
- **C. ONE HUFFMAN MODULE (DONE 2026-07-21)**: `huffman/` is now a directory
  module — `mod.rs` (shared `HuffmanCode` + re-exports), `fast.rs` (libdeflate
  approximate builder, unchanged, still used by near-optimal + every hot
  level), `optimal.rs` (katajainen + the RLE-aware count shaping
  `optimize_huffman_for_rle`/`try_optimize_huffman_for_rle`/`ShapeDepth`,
  MOVED here from `parse::ultra::deflate_size` since they shape Huffman
  counts, not block-size math — `deflate_size` keeps calling them via the new
  path and stays the home of the true-cost helpers they call back into),
  `header.rs` (the level engine's ONE dynamic-header build+emit, moved from
  the old `huffman.rs`). `tree.rs` (zopfli `tree.c` port) is dissolved:
  `calculate_bit_lengths`/`lengths_to_symbols` (Huffman construction) joined
  `optimal.rs`; `calculate_entropy` (a cost-model proxy, single consumer)
  moved into `parse::ultra::squeeze`, its sole call site. Ultra's OWN
  dynamic-header emitter (`deflate_size::build_rle` + `deflate::encode_tree_emit`)
  is UNMERGED with `huffman::header` — see §3's "Dynamic-header wire format"
  row for why. Gate: byte-identity both engines (L0/L1/L2/L6/L9/L12 × T1/T4 ×
  {dickens, data.parquet} + `-F 15`/`-F 80` × 5-file ratio corpus, all
  sha256-equal vs pre-stage HEAD); full test suite green; clippy/fmt clean.
- **D. MATCHFINDER CONSOLIDATION (measured, riskiest) — DONE 2026-07-21, mixed
  outcome, both parts closed.**
  - **Part 1 (structural, byte-identity gate) — KEPT.** `matchfinder/mod.rs`
    carries a module-doc table of all four tiers (ht/hc/bt/lzfind), their
    consumers, output shapes, and position models, plus an explicit
    documented reason `ht` stays fused into `parse/fast.rs` rather than being
    extracted (codegen risk to the SF1-A/C prefetch/pipeline work — no
    measured need). The one piece of shared vocabulary two tiers'
    signatures already agreed on — `bt`'s per-position match LIST — moved
    to `matchfinder::common::LzMatch` (bt.rs re-exports it, so no call site
    changed); `hc` (single best match) and `lzfind` (packed `u16` frontier)
    stay their own shapes, documented as a deliberate non-unification, not
    an oversight. Gate: byte-identity (L0/L1/L2/L6/L9/L12 × T1/T4 × {dickens,
    data.parquet} + `-F 15`/`-F 80` × {dickens, data.parquet, markup.xml},
    all sha256-equal vs pre-stage HEAD) — 30/30 PASSED.
  - **Part 2 (measured, retire the legacy zopfli chain finder) — REVERTED.**
    Built `parse/ultra/greedy_hc.rs`: an `HcMatchfinder`-driven greedy walk
    (level engine's L9 depth=600/nice_len=258) replacing `lz77::lz77_greedy`
    at both call sites (squeeze's DP seed, the block splitter's greedy
    pre-pass), deleting `hash.rs`+`cache.rs`+`find_longest_match`+
    `lz77_greedy` (~700 LOC) in exchange for ~290 LOC of new driver+tests.
    Scored on the 19-file Squishy corpus against a FRESH own-binary
    before/after pair (not the stale 3.203126/3.205795 numbers this doc
    previously cited — those predate this session's measurement):
    F15 geomean 3.205250 (before) → 3.204923 (after, **FAILS** the
    ≥-baseline bar); spliced F80 geomean (8 fresh + 11
    `squishy_f80_local.jsonl` reference) 3.205685 vs the 3.205752
    ect-10009-crossing bar (**FAILS**). Both misses are small (~0.01-0.02%,
    comparable to the ~0.001 repeat-run geomean noise documented elsewhere
    in this doc) but the pre-registered rule carried no noise margin, so
    both legs of the AND-gate fired the revert trigger — per the
    pre-registered rule this REVERTS IN FULL: `greedy_hc.rs` deleted,
    `hash.rs`/`cache.rs`/`lz77::find_longest_match`/`lz77_greedy`/
    `BlockState`'s `LongestMatchCache` field all restored verbatim from
    pre-stage HEAD. "measured: legacy seed earns its LOC" — closed, not
    open. Re-open trigger: a multi-run (N≥7) significance test that
    actually separates this miss from noise, or a deeper/differently-tuned
    HC seed.
- **E. POLISH**: deflate64 internals on shared modules; docs/compressor-
  architecture.md (this file) lands in-repo; lib.rs API docs rewritten from the
  actual routing; LOC and dead-code audit re-run.

## 6. What this buys

- Every learned technique (libdeflate, igzip, zlib-ng-tested levers, zopfli, ECT,
  and the gzippy-original SF-chain + crown session wins) reachable from ONE policy
  table, individually selectable per level — the original "level.rs is the only
  tuning source" promise, now including the crown tier.
- ~1.5-2k LOC of duplicated framing/plumbing deleted; three gzip writers → one;
  four matchfinders → four *documented tiers behind one trait* (with a gated path
  to three).
- The routing inconsistencies (-11 vs -F, stale docs, dead predicates) resolved by
  construction: there is only one dispatcher and it reads level.rs.
