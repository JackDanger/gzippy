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

- **A. STRUCTURAL MOVE (no behavior) — DONE (`449d064a`, PR #140).** relocate zopfli_pure → parse/ultra/ +
  routing/level unification (-F/-I/-J through level.rs; delete use_zopfli dead code,
  ZopfliGzEncoder's private header writer, gzip.rs/zlib.rs test-only wrappers,
  blocksplittingmax dead field; fix stale lib.rs/cli.rs docs). Gate: byte-identity
  EVERYWHERE (L0-12 + ultra), tests green. Pure `git mv` + import surgery.
- **B. ONE BITSTREAM — DONE (`6797fcdb`, PR #141).** ultra emits through the word BitWriter. Gate: crown outputs
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
- **E. POLISH — DONE (2026-07-21).** deflate64 internals rebuilt on shared
  modules (bitstream, exact Huffman construction, dynamic-header wire format,
  stored-block framing — see §7 for exactly what moved and what stayed
  private and why); docs/compressor-architecture.md (this file) lands
  in-repo; main.rs `--help` / README.md corrected to describe the actual
  unified routing (no more "libdeflate for 1-6 / zlib-ng for 7-9" — that
  split predates this campaign and was never true post-Increment-7); LOC and
  dead-code audit re-run (5 genuinely-dead items deleted, the module-wide
  `#![allow(dead_code)]` in `compress::deflate` removed in favor of one
  narrow, documented allow). Full outcome in §7.

## 6. What this buys

- Every learned technique (libdeflate, igzip, zlib-ng-tested levers, zopfli, ECT,
  and the gzippy-original SF-chain + crown session wins) reachable from ONE policy
  table, individually selectable per level — the original "level.rs is the only
  tuning source" promise, now including the crown tier.
- Three gzip writers → one; four matchfinders → four *documented tiers behind
  one trait* (with a gated path to three); duplicated stored-block framing,
  dynamic-header emission, and exact-Huffman construction collapsed to one
  call site each (deflate64 now instantiates the SAME shared modules the main
  engine uses, rather than reimplementing them — §7).
- The routing inconsistencies (-11 vs -F, stale docs, dead predicates) resolved by
  construction: there is only one dispatcher and it reads level.rs.
- **Correction (Stage E, §7): the "~1.5-2k LOC deleted" estimate below this
  line in earlier drafts of this document did NOT hold under measurement.**
  Raw LOC across the unified surface is nearly FLAT end-to-end (+134 lines,
  measured — see §7): the framing/plumbing genuinely deleted was offset by
  the (much larger, and much more valuable) volume of documentation this
  campaign's convention requires for every "merged" AND every "deliberately
  NOT merged" decision, plus real new tests (Stage D's built-then-reverted
  `greedy_hc.rs`, this stage's deflate64 real-corpus differential). The win
  this campaign banks is NOT a LOC count — it is ONE location (not two split
  across `compress::deflate` and `backends::zopfli_pure`), ONE tuning surface
  (`level.rs`), and every remaining duplication EXPLICITLY DOCUMENTED with
  the reason it wasn't collapsed, instead of silently re-implemented.

## 7. Outcome (Stage E, 2026-07-21) — measured, not estimated

Every number below is a direct `wc -l` / `git diff --stat` measurement against
this repo at the stated commits — re-derivable by anyone, not a narrative
estimate. "Initial" = `3064c1b2` (the commit immediately before Stage A's
`docs: add compressor unification architecture` landed); "final" = this
stage's HEAD.

### LOC: initial two-engine layout vs final unified layout

| Location | Initial (`3064c1b2`) | Final (this HEAD) |
|---|---:|---:|
| `src/compress/**` (level engine + deflate64 + fragments) | 10,620 | 16,719 |
| `src/backends/zopfli_pure/**` (crown engine, separate tree) | 5,965 | 0 (moved) |
| **Total (two locations → one)** | **16,585** | **16,719 (+134, +0.8%)** |

Raw LOC is essentially FLAT — see the §6 correction above for why a "LOC
deleted" framing is the wrong scoreboard here. What DID change, precisely:

- **Two module trees → one.** `src/backends/zopfli_pure/` (5,965 LOC) no
  longer exists; every byte of it lives under `src/compress/deflate/parse/ultra/`
  now (Stage A, `449d064a`). There is no longer a "which tree has the crown
  engine" question.
- **Per-file breakdown of the current unified tree** (`src/compress/**`,
  16,719 LOC total):

  | Subtree | LOC | Role |
  |---|---:|---|
  | `deflate/{mod,bitstream,block_split,costs,level,tables}.rs` | 2,231 | entry points, THE policy table, shared bitstream/tables |
  | `deflate/huffman/{mod,fast,optimal,header}.rs` | 1,228 | ONE Huffman module tree (Stage C) — approximate (hot) + exact (katajainen) builders + the dynamic-header emitter |
  | `deflate/matchfinder/{mod,common,hc,bt,lzfind}.rs` | 2,319 | four documented tiers behind one vocabulary (`ht` stays fused into `parse/fast.rs`, not its own file — Stage D pt.1) |
  | `deflate/parse/{mod,fast,greedy,lazy,near_optimal}.rs` | 2,499 | hot-level parsers (levels 0-9) |
  | `deflate/parse/ultra/**` | 5,162 | the crown engine (relocated, Stage A; bitstream unified, Stage B) |
  | `deflate64.rs` | 534 | Deflate64/ZIP-method-9, now built on the shared bitstream/huffman/header modules (Stage E, this stage) |
  | `mod.rs`, `io.rs`, `parallel.rs`, `pipelined.rs`, `simple.rs`, `optimization.rs` | 2,746 | routing, gzip I/O, the C-FFI oracle backends (`ffi-oracle`-gated, off the production graph) |

### What was deleted/merged/documented-as-residual, by stage

- **Stage A** (`449d064a`, PR #140): `zopfli_pure` → `parse/ultra/` relocated
  whole; `use_zopfli` dead routing code, `ZopfliGzEncoder`'s private header
  writer, `gzip.rs`/`zlib.rs` test-only wrappers, and a dead
  `blocksplittingmax` field deleted; stale lib.rs/cli.rs docs fixed at the
  time.
- **Stage B** (`6797fcdb`, PR #141): ultra's own bit-at-a-time `BitWriter`
  deleted; every ultra emitter ported onto the shared word `BitWriter`
  (`bitstream.rs`). Two ultra-specific unit tests (a direct MSB-first check +
  a differential fuzz vs a from-scratch reference) were PORTED onto the
  shared writer's test module rather than dropped — they still exist, just
  relocated (`bitstream.rs`'s `add_huffman_bits_msb_first` /
  `add_huffman_bits_matches_bit_at_a_time_reference`). A real bug
  (`align_to_byte`'s `>> 64` UB on `bitcount` landing exactly on 64) was
  found and fixed during this port — documented in `bitstream.rs` with a
  regression test, not just fixed silently.
- **Stage C** (`1c6410a9`, PR #141): `huffman.rs` (level engine) +
  `parse::ultra::deflate_size`'s RLE-shaping functions + zopfli `tree.c`'s
  non-entropy half MERGED into one `huffman/` directory module
  (`fast`/`optimal`/`header`). `tree.rs` dissolved into `optimal.rs` +
  `squeeze.rs`. The dynamic-header WIRE FORMAT duplication (level engine's
  `huffman::header` vs ultra's `deflate`/`deflate_size` RLE builder) was
  identified, measured to be cost-accounting-incompatible, and RECORDED as
  intentional residual rather than force-merged (§3 table).
- **Stage D part 1** (`5d9d0d46`, PR #142, byte-identity KEPT): matchfinder
  tier module doc + the shared `LzMatch` vocabulary type extracted for the
  one tier-pair (`bt`) that already agreed on a match-list shape; `hc` and
  `lzfind` documented as deliberately NOT unified (single-best-match vs.
  packed-frontier are different consumption patterns, not oversight).
- **Stage D part 2** (measured, REVERTED in full): a from-scratch
  `greedy_hc.rs` (HC-matchfinder-driven greedy walk replacing the legacy
  zopfli hash/lz77 chain finder) was built, gated on the Squishy score bar,
  FAILED both legs (F15 and spliced F80 geomean), and was reverted verbatim
  per the pre-registered rule — `hash.rs`/`cache.rs`/`lz77::find_longest_match`
  restored. Recorded as a FALSIFY entry (§5 Stage D), not deleted from the
  record.
- **Stage E** (this stage, 2026-07-21):
  - `compress/deflate64.rs` (711 → 534 LOC net, despite ADDING ~45 lines of
    module doc and a new ~28-line real-corpus test — the implementation
    portion alone, excluding `#[cfg(test)]`, shrank from 605 to 398 lines,
    -34%): its private
    `BitWriter` (struct + 4 methods, ~65 LOC), `package_merge` (~65 LOC),
    `assign_codes` (~22 LOC), and the entire precode/RLE machinery (`ClSym`,
    `rle_lengths`, `cl_sym_index`, `CL_ORDER`, hlit/hdist/hclen computation,
    ~110 LOC combined) DELETED and replaced with direct calls into
    `compress::deflate::bitstream::BitWriter`,
    `huffman::optimal::{calculate_bit_lengths, lengths_to_symbols}` (the
    same package-merge/boundary-PM algorithm class deflate64 used to
    hand-roll), and `huffman::header::build_dynamic_header` (which turned out
    to be fully format-agnostic — it trims/pads litlen/offset alphabets
    itself, so it needed zero changes to accept Deflate64's 286/32-symbol
    arrays). The empty-input stored-block special case now calls the shared
    `emit_stored_block` (newly `pub(crate)`) instead of hardcoding
    `[0x01,0x00,0x00,0xFF,0xFF]`. The match-length extension loop was
    switched from a hand-rolled byte compare to the shared
    `matchfinder::common::lz_extend` word-at-a-time primitive (contract
    satisfied exactly by deflate64's own bounds).
  - What STAYED private, and why (documented in the module's own doc
    comment, not just here): the length/distance code TABLES
    (`LENGTH_BASE`/`LENGTH_EXTRA`/`DIST_BASE`/`DIST_EXTRA`) — Deflate64
    extends litlen symbol 285 to 16 extra bits (length up to 65538) and adds
    distance codes 30/31 for the 64 KiB window, genuinely different RFC
    tables from `compress::deflate::tables`'s 32 KiB set, not a dedup gap.
    The match finder itself — `matchfinder::hc::HcMatchfinder` hard-codes a
    32 KiB window via SIGNED `i16` chain positions
    (`WINDOW_SIZE = 1 << 15`, `MATCHFINDER_INITVAL = i16::MIN`); `i16`
    cannot address a 64 KiB window at all (max magnitude 32768 < 65536), so
    reuse would require widening every position field to `i32`/`u32` across
    a heavily-pinned, gated hot module for a CLI-unreachable format with no
    performance constituency — verified by reading the type, not assumed.
    `matchfinder::common::LzMatch` (the shared match-LIST vocabulary type)
    was also checked and rejected: it packs length/offset as `u16`, which
    cannot represent Deflate64's length-65538/offset-65536 range either way
    (confirmed: `u16::MAX == 65535`).
  - Correctness: deflate64's existing 18 tests (9 compress + 9 decompress,
    covering code 285, dist codes 30/31, multi-block, LCG fuzz sizes) all
    pass unchanged, PLUS one new test added this stage
    (`test_silesia_slice_roundtrip`, two real-text slices from
    `benchmark_data/silesia.tar`, exercising the rewritten Huffman/header
    path against a literal/match frequency distribution no synthetic
    generator in the existing suite produces). 19/19 green.
  - Dead-code audit: the module-wide `#![allow(dead_code)]` in
    `compress::deflate::mod.rs` (present since Increment 1, excused as
    "some substrate primitives are used only by later increments") was
    REMOVED — near-optimal/ultra landed in Stages A-D, so the excuse no
    longer held, and a `cargo build --release` with the allow stripped
    surfaced exactly 5 warnings, all genuinely dead in BOTH production and
    tests (zero references anywhere in the repo): `BitWriter::with_capacity`,
    `BitWriter::buffered_bits`, `HcMatchfinder::reset` (plus its now-orphaned
    `matchfinder_init` import), `tables::DEFLATE_MAX_NUM_SYMS`,
    `tables::DEFLATE_MAX_CODEWORD_LEN` — all deleted. One item,
    `level::max_passthrough_size` (a libdeflate port,
    `deflate_compress.c:3918`, formula-pinned by its own unit test but never
    wired into the near-optimal parse entry point), was KEPT with a narrow
    `#[allow(dead_code)]` and a doc note rather than deleted — wiring it in
    would change L10-12 output for tiny inputs (an algorithmic change, out
    of scope for a polish stage) but it is correct, cited, and tested, so
    deleting a real unexploited lever would be throwing away work for no
    reason. The module is now warning-clean WITHOUT a blanket suppression,
    so anything that goes dead in the future will be caught immediately.
  - Docs: `main.rs`'s `--help` "Compression levels" block described a
    pre-unification split ("1-6 libdeflate", "7-9 zlib-ng", "10,12 libdeflate
    ultra") that stopped being true when Increment 7 removed C-FFI from the
    compress routing graph — corrected to describe the one pure-Rust engine.
    `README.md`'s "One caveat" section claimed `threads > 1` at levels 0-5
    still emits gzippy's own "GZ" multi-block format by default — also
    stale (confirmed via `compress::io.rs`: the CLI entry point never
    constructs `ParallelGzEncoder`; it is compiled only under the
    `ffi-oracle` feature as a differential oracle) — removed, since every
    level/thread-count combination now produces standard single-member gzip
    and there is no longer a caveat to state.

### Residuals NOT touched by Stage E (honest, not silent)

- **`man/gzippy.1` and `man/gzippy-format.5`** still describe the
  pre-unification decode routing table (per-format ISA-L/libdeflate/zlib-ng
  dispatch) and compression backend split. This stage's gate was scoped to
  "lib.rs + README.md" (mission brief); the man pages are a real, separately-
  sized staleness surface (decode routing, not compress) and were left
  alone rather than half-fixed under time pressure. Flagged for a follow-up
  pass.
- **`--no-default-features` build is BROKEN independent of this stage.**
  Confirmed by temporarily reverting this stage's two touched files back to
  HEAD and rebuilding: the failure reproduces identically (`E0432`
  unresolved import `crate::decompress::parallel::sm_driver`, `E0433`
  missing `fd_vectored_write`/`chunk_data` in `parallel`, three `E0433`
  missing `Ordering` imports) — all in `src/decompress/parallel/{single_member,
  stored_split}.rs`, entirely outside `src/compress/**`. This is a
  structural feature-gating bug in the decode module, pre-existing at
  `e20b7736` (this stage's start commit), NOT introduced by anything in
  Stage E, and NOT a trivial one-line fix (it spans several missing
  symbols/imports suggesting a cfg-gating mismatch, not a single typo).
  Left unfixed rather than risking an under-scoped patch to an unrelated,
  gated decode module during a compress-only polish stage; the default
  build (what actually ships) is unaffected and fully green.
- **Dynamic-header wire-format duplication** (Stage C's residual, §3) and
  **the legacy zopfli matchfinder for squeeze** (Stage D part 2's FALSIFY
  entry) both stand as recorded, deliberate non-merges — re-checked during
  this audit and still accurate as written; not re-litigated.
