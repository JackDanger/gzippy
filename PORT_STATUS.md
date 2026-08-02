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

## ⭐⭐ RUNG 3 IS GREEN FOR EVERY NON-EXOTIC LEVEL — L0-L9, 23/23 EACH

```
$ scripts/campaign/ldx-differential.sh 0 1 2 3 4 5 6 7 8 9
L0: 23/23   L2: 23/23   L4: 23/23   L6: 23/23   L8: 23/23
L1: 23/23   L3: 23/23   L5: 23/23   L7: 23/23   L9: 23/23
```

Raw DEFLATE, sha256, all 22 corpus files plus the empty input, against libdeflate's
own `libdeflate_deflate_compress` linked from `vendor/libdeflate/build/libdeflate.a`.
`wc -c` never counts — two streams of equal length can differ in every bit.

**`ldx` is now a complete, exact, pure-Rust reimplementation of libdeflate's compressor
for every level a user reaches without asking for -10 or above.** Only the exotic
levels (10-12, near-optimal parsing) remain unported.

**This closes ZERO cells.** Nothing is routed; `src/compress/deflate` is untouched and
every shipped byte is unchanged. Routing is a PROMOTION-RULE question (clause 3 on 154
tied cells, clause 5 on the wall), and the binding record at
`src/compress/deflate/parse/mod.rs:540` still governs L1 specifically.

## Done (15 commits, 101 tests passing)

| C | item | file | commit |
|---|---|---|---|
| `deflate_constants.h` | constants, verbatim | `mod.rs` | `4e35dc5c` |
| `:761-814` | `heapify_subtree` / `heapify_array` / `heap_sort` | `heap.rs` | `4e35dc5c` |
| `:848` | `sort_symbols` (counting sort + heapsort tail) | `huffman.rs` | `bb0d4d47` |
| `:941` | `build_tree` (Van Leeuwen two-queue) | `huffman.rs` | `65d60dd2` |
| `:1024` | `compute_length_counts` (heuristic length limiter) | `huffman.rs` | `17a3297d` |
| `:1105/:1146`, `:1179` | `reverse_codeword`, `gen_codewords` | `huffman.rs` | `71829415` |
| `:1320` | `deflate_make_huffman_code` | `huffman.rs` | `475ae4d5` |
| `:238-320` | the six RFC 1951 tables | `tables.rs` | `9fabdda8` |
| `:1455` | `deflate_get_offset_slot` | `tables.rs` | `9fabdda8` |
| `:325-350` | `deflate_{codewords,lens,codes,freqs}` | `codes.rs` | `9fabdda8` |
| `:1416`, `:1433` | `deflate_make_huffman_codes`, `_init_static_codes` | `codes.rs` | `9fabdda8` |
| `:1484`, `:1572` | `deflate_compute_precode_items`, `_precompute_huffman_header` | `precode.rs` | `9fabdda8` |
| `:1640` | `deflate_compute_full_len_codewords` | `length.rs` | `9fabdda8` |
| `:667-750` | output bitstream, `ADD_BITS`, `FLUSH_BITS`, `CAN_BUFFER` | `bitstream.rs` | `21cac499` |
| `:354`, `:1662`, `:1708` | `deflate_sequence`, `WRITE_MATCH`, **`deflate_flush_block`** | `flush.rs` | `21cac499` |
| `:440-449`, `:2104-2222` | block split stats, `should_end_block` | `split.rs` | `3d503360` |
| `:2042`, `:2224-2270` | `deflate_finish_block`, the sequence store | `sequences.rs` | `3d503360` |
| `matchfinder_common.h` | `mf_pos_t`, rebase, `lz_hash`, `lz_extend` | `matchfinder_common.rs` | `5f5d0420` |
| `ht_matchfinder.h` | the whole header | `ht_matchfinder.rs` | `5f5d0420` |
| `:2381`, `:2452` | `choose_max_block_end`, **`deflate_compress_fastest`** | `compress_fastest.rs` | `5f5d0420` |
| `:2393`, `:4050` | `deflate_compress_none`, **`libdeflate_deflate_compress`** | `compress.rs` | `5f5d0420` |

Verified by: Kraft equality; prefix-free AND complete codespace; an exhaustive
`reverse_codeword` differential (2^16 x 16 cases) against the C's own table variant;
all three degenerate `num_used_syms < 2` variants; the offset-slot map checked over
the WHOLE range 1..=32768; precode items round-tripped through an independent RFC
1951 decoder; and **whole DEFLATE streams round-tripped through flate2** (literals,
matches at lengths 3..258, offsets 1..32768, stored-block splitting at 65535, a
short-buffer overflow, and two blocks sharing a partial byte).

**The cost model is self-checking and the check is PROVEN LIVE.** The C's closing
`ASSERT(8 * (out_next - os->next) + bitcount - os->bitcount == best_cost)` drives the
single output-buffer bounds check for a whole block. Perturbing `best_cost` by one
bit fails 6 of the 7 flush tests — verified by doing it, not by assuming the
assertion compiles in.

### Three things the port had to get exactly right, each pinned by a test

1. `deflate_get_offset_slot`'s `n = (256 - offset) >> 29` is an **unsigned wrap**.
   Signed, it gives -1 instead of 7 and every offset above 256 lands in the wrong slot.
2. `DeflateLens` must be `#[repr(C)]` with `offset` at byte 288, because
   `deflate_precompute_huffman_header` memmoves the offset lengths down and RLEs
   ACROSS the join — a zero run spanning the boundary is ONE precode item.
3. The precode's nonzero-run threshold is `>= 4`, not `>= 3`.

### Two known-and-pinned faithful oddities

* **The header restore leaves residue.** `deflate_precompute_huffman_header`'s second
  memmove does not erase the copy the first one made, so litlen lengths
  `[num_litlen_syms, +num_offset_syms)` hold stale offset lengths on return. The C
  does the same. Unobservable for two separately-checked reasons (those symbols have
  zero frequency by construction; the next block rewrites all 288 lengths). Pinned by
  `precompute_leaves_the_same_residue_as_c` so a future tidy-up fails a test.
* **The tie-break order is observable.** `MIN(dynamic, MIN(static, uncompressed))` then
  `if (best == uncompressed) ... if (best == static)` sends ties to uncompressed, then
  static, then dynamic. Reordering, or `<=` instead of `==`, is a different valid
  stream.

### Deliberate, behaviour-free divergences (all stated at the site)

* The C's `union o { precode; length; }` is two separate types — storage economy, not
  behaviour, and unioning it in Rust would need `unsafe` for nothing.
* `deflate_init_static_codes` takes its scratch frequency table as a parameter rather
  than aliasing the live one.
* `next`/`end` are indices into a `&mut [u8]`, not raw pointers.
* The live code table is a flag plus a clone, not a rebindable pointer — the borrow
  checker cannot see that writing `c.o_length` does not alias the table being read.

## Next

1. `bt_matchfinder.h` + `:2845-3853` near-optimal -> gates **L10-L12**. This also
   supplies `deflate_flush_block`'s missing `sequences == NULL` arm (`:1935`), which
   walks `optimum_nodes` — deliberately left out of the signature rather than stubbed.
   `LdxCompressor::new` refuses 10-12 today rather than guessing.
2. Then the ROUTING question, which is NOT a port question. It is clause 3 on 154 tied
   cells and clause 5 on the wall, per level, and it needs `fulcrum try --threads 1,4`
   rather than this differential.

**L3 is the one level where we deliberately diverge and we WIN.** Our shipping L3 is
LAZY(12,14) against libdeflate's GREEDY(12,14) — same knobs, different parser — and
ours is smaller on 20 of 22 files, median ~44 KB. `ldx` matching THEIR choice is phase 1
succeeding; our lazy L3 is a phase-2 win to re-layer. Do not "fix" the port to keep it.

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
