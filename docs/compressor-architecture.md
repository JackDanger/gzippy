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
| Splitting (hot) | SAD streaming | O(1)/token, carries byte-identity L2-12. |
| Splitting (exact) | uncapped recursive exact-cost | crown-gated (+0.00116 geomean lever). |
| Matchfinder seed for squeeze | hc (replaces zopfli legacy hash/lz77 chain finder) | legacy finder is demoted duty; hc is faster and maintained. GATED: crown score must hold ≥ 3.205795 (seed changes outputs — score gate, not byte gate). If it loses ratio, keep legacy finder — measurement decides. |
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
- **C. ONE HUFFMAN MODULE**: colocate fast/optimal builders + single header
  emitter; near-optimal keeps the approximate builder (byte-identity); ultra keeps
  katajainen. Gate: byte-identity both engines.
- **D. MATCHFINDER CONSOLIDATION (measured, riskiest)**: one trait over ht/hc/bt/
  lzfind; retire the zopfli legacy chain finder by seeding squeeze + the greedy
  splitter pre-pass from hc. Gate: crown score ≥ 3.205795 (this changes ultra
  bytes); if score drops, the legacy finder stays and the stage closes as
  "measured: legacy seed earns its LOC".
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
