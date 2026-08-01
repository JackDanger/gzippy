# `ldx` — exact pure-Rust port of libdeflate's compressor: STATUS

**Branch:** `port/libdeflate-exact`  **Source:** `vendor/libdeflate/lib/deflate_compress.c` (4,155 lines)

**GOAL (user, 2026-08-01):** "perfectly copy libdeflate's exact implementation in pure
rust, performing exactly the same and producing the same output."

Read `src/compress/ldx/mod.rs` first — it carries the porting rules and the
verification ladder. This file is only the checklist.

## Why the port exists

Our core deflate path is ~17,000 lines across 25 modules and is FOUR lineages
(igzip's `parse/fast.rs` IS our L1, zopfli's `parse/ultra/`, libdeflate's `hc`/`ht`,
our own dispatch + `lzfind`). libdeflate's is 4,155 lines in one file. We do not HAVE
their implementation — we have pieces of it plus two other vendors'.

**Measured 2026-08-01, and it narrows the job a lot:** the level->config map is
already a transliteration — at L0-L9 exactly TWO parameters differ (L0: they store,
we compress; L3: they Greedy, we Lazy — and ours wins 20/22 files). Every
`max_search_depth`/`nice_match_length` matches byte-for-byte. **So the parity gap is
CODE, not configuration.** A separate function-by-function audit found `hc.rs` and
`bt.rs` to be semantically EXACT transliterations with zero output-affecting
divergence, consistent with L2/L4-L9 being 154/154 byte ties.

**The T1 tie work list is 17 losses, not 44** (of 198 libdeflate@T1 cells: 154 ties,
27 wins, 17 losses). **15 of the 17 are L1.** So L1 — where we ship igzip's chainless
finder instead of their `ht_matchfinder` — is essentially the entire phase-1 gap.

## Done (6 commits, 17 tests passing)

| C | item | commit |
|---|---|---|
| `deflate_constants.h` | constants, verbatim | `4e35dc5c` |
| `:761-814` | `heapify_subtree` / `heapify_array` / `heap_sort` | `4e35dc5c` |
| `:848` | `sort_symbols` (counting sort + heapsort tail) | `bb0d4d47` |
| `:941` | `build_tree` (Van Leeuwen two-queue) | `65d60dd2` |
| `:1024` | `compute_length_counts` (heuristic length limiter) | `17a3297d` |
| `:1105/:1146`, `:1179` | `reverse_codeword`, `gen_codewords` | `71829415` |
| `:1320` | `deflate_make_huffman_code` | `475ae4d5` |

The Huffman construction chain is COMPLETE and verified: Kraft equality, prefix-free
+ complete codespace, exhaustive `reverse_codeword` check (2^16 x 16 cases) against
the C's own table variant, and all three degenerate `num_used_syms < 2` variants
pinned explicitly.

## Next, in dependency order

1. `:1455` `deflate_get_offset_slot`; `:1484` `deflate_compute_precode_items`;
   `:1572` `deflate_precompute_huffman_header`; `:1640`
   `deflate_compute_full_len_codewords`.
2. `:1708` **`deflate_flush_block`** — 334 lines, the big one. **First point where a
   real byte-level differential against libdeflate is possible.** Get here before
   porting any parser; it is the oracle everything downstream needs.
3. `:2094-2225` block-split stats. NOTE our `block_split.rs:192-200` computes the
   cutoff in `u64` where the C uses `u32` AND WRAPS — port the C's widths here.
4. `:2224-2270` sequence store (`deflate_begin_sequences` / `_choose_literal` /
   `_choose_match`); `:2272-2392` the min-match-len helpers.
5. Matchfinders: `ht_matchfinder.h` FIRST (it is the 15-cell L1 gap). **Note our
   `matchfinder/ht.rs` is NOT a port of it** — it adds a length-3 table the C
   explicitly refuses (`ht_matchfinder.h:38-40`) and imports `HT_MAX_LEN3_OFFSET`
   from a different C function. Read the binding FALSIFY at `parse/mod.rs:540` before
   touching this: both prior attempts are already recorded.
6. `:2394-2843` the four compressors: `_none` / `_fastest` / `_greedy` /
   `_lazy_generic`.
7. `:2845-3853` near-optimal (costs, `deflate_find_min_cost_path`).
8. `:3874` `libdeflate_alloc_compressor_ex` — the level->config map, ported verbatim.

## Standing cautions

* **Nothing routes until whole-stream sha256 matches libdeflate-gzip for that level.**
  `wc -c` never counts. Run `scripts/campaign/tie-guard.sh` before any change that
  alters T1 output — 154 tied cells have ZERO tolerance and clause 3 is absolute.
* **Do not "improve" the heuristic length limiter.** Binding FALSIFY at
  `src/compress/deflate/huffman/fast.rs:432`: built both ways, heuristic is within
  ~0.001% of exact package-merge, the swap OPENS cells and costs 10-14% wall.
* **Clippy will ask you to rewrite the C's expressions** (`i + 1 <= last_idx` ->
  `i < last_idx`). Suppress with a reason at the site; the file's contract is that it
  diffs against the C line by line.
* **Where the C ships two spellings of the same operation** (e.g. `reverse_codeword`),
  matching the operation is faithful — but PROVE it exhaustively, do not argue it.
