# Vendor technique index — every encoder technique in the compressors we compete with

This is a durable reference index of the optimizations, approaches, and algorithms used by
the DEFLATE/gzip encoders we compete with, grounded line-by-line in the vendored sources,
with an explicit verdict per technique on whether WE already do it. It is technique-oriented
and broader than `docs/vendor-structure-comparison.md`, which holds the level tables (§1),
block sizing (§2), hash geometry (§3), and the L2 operation-level cost diff (§4) — those are
cross-referenced here, not duplicated. Decompression is out of scope.

Every citation was taken from a numbered read of the vendored file. Paths are relative to
the repo root. "Ours" paths are `src/compress/...`.

## Vendored versions (verified)

| Vendor | Version | Evidence |
|---|---|---|
| libdeflate | 1.25 | `vendor/libdeflate/libdeflate.h:17` |
| zlib-ng | 2.3.90 (develop) | `vendor/zlib-ng/zlib-ng.h.in:51` |
| ISA-L (igzip) | 2.31.1 + local decode-side patches (branch `gzippy-stopping-points`) | `vendor/isa-l/configure.ac:6`; encoder is stock |
| pigz | 2.8 | `vendor/pigz/pigz.c:213` |
| GNU gzip | 1.6 (distrotech mirror) | `vendor/gzip/NEWS` (release 1.6, 2013-06-09) |
| zopfli | 1.0.3 (+9 commits) | `vendor/zopfli/CMakeLists.txt:58-60`, git `zopfli-1.0.3-9-gccf9f05` |
| isal-rs | 0.5.3 — pure FFI wrapper over ISA-L, no encoder logic of its own; only exposes levels 0/1/3 | `vendor/isal-rs/Cargo.toml:1-11`, `vendor/isal-rs/src/lib.rs:200-233` |
| rapidgzip | decoder only — out of scope | — |

Entry format: **Name** (vendor identifiers) · Who/where · Mechanism & when active ·
Parameters · **Ours:** YES/PARTIALLY/NO · Applicability (L0/L1 fast, L2-9 greedy/lazy,
L10-12 near-optimal, T>1).

---

# A. Parsing strategies

## P1. Greedy parse
- **Identifiers**: libdeflate `deflate_compress_greedy`; zlib-ng `deflate_fast`; gzip `deflate_fast`; igzip (all levels — igzip has NO other parse class).
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:2528-2602` (levels 2-4); `vendor/zlib-ng/deflate_fast.c:21-112` (level 2 only — see P9/P10); `vendor/gzip/deflate.c:596-671` (levels 1-3); `vendor/isa-l/igzip/igzip_base.c:61-85` / `igzip_icf_base.c:76-82` (all levels).
- **Mechanism**: one match search per position; accept the longest immediately, insert (or skip-insert) covered positions, jump.
- **Parameters**: acceptance min length — libdeflate: adaptive `min_len` (see P7) with len-3 gated at offset ≤ 4096 (`deflate_compress.c:2573-2575`); zlib-ng/igzip: hard minimum 4 (`WANT_MIN_MATCH` `vendor/zlib-ng/zutil.h:60`; `SHORTEST_MATCH 4` `vendor/isa-l/igzip/huff_codes.h:89`); gzip: 3.
- **Ours**: YES — `parse/greedy.rs:175-185` is the libdeflate shape exactly (min_len + `length > 3 || offset <= 4096`, `skip_bytes(length-1)`), same adaptive min_len (`parse/mod.rs:611-648`).
- **Applicability**: already our L2/L4.

## P2. Lazy one-ahead parse
- **Identifiers**: libdeflate `deflate_compress_lazy`; zlib-ng/gzip `deflate_slow`; zopfli `ZOPFLI_LAZY_MATCHING` (in its greedy pre-parse).
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:2604-2808` (levels 5-7); `vendor/zlib-ng/deflate_slow.c:18-151` (levels 7-9); `vendor/gzip/deflate.c` lazy loop (levels 4-9, table `deflate.c:242-254`); `vendor/zopfli/src/zopfli/lz77.c:578-613`.
- **Mechanism**: defer emitting a match one position; emit the previous match only if the next search doesn't beat it. zlib/gzip lineage compares raw lengths (`match_len <= s->prev_length`, `deflate_slow.c:84`) and gates the second search on `prev_length < max_lazy_match` (`deflate_slow.c:66`). libdeflate uses a cost heuristic (P4) instead. zopfli's pre-parse penalizes score by 1 when dist > 1024 (`GetLengthScore`, `lz77.c:265-271`).
- **Ours**: YES — `parse/lazy.rs:190-230`, libdeflate-shaped (P4 heuristic, half-depth P5).
- **Applicability**: our L5-7. Note vendor-structure-comparison §1: gzip/zlib-ng go lazy at L4, we (like libdeflate) stay greedy at L4 — the P4-violation cause.

## P3. Lazy2 (two-ahead deferral)
- **Identifiers**: libdeflate `deflate_compress_lazy2`. **Only libdeflate has this** among the vendors.
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:2604-2808` (one `forceinline` body monomorphized on a `lazy2` bool; wrappers :2815-2834), levels 8-9.
- **Mechanism**: after a deferral wins at pos+1, probe pos+2 at quarter depth; deferral threshold 6 instead of 2; on success emit two literals (`deflate_compress.c:2742-2763`). Cursor advance `cur_len - 3` when `cur_len > 3` (:2774-2782).
- **Ours**: YES — `parse/lazy.rs:243-247` (quarter depth `depth >> 2`, threshold 6), our L8-9.
- **Applicability**: already our L8-9.

## P4. Deferral cost heuristic — 4·Δlen + Δlog2(offset) > threshold
- **Who**: libdeflate only: `vendor/libdeflate/lib/deflate_compress.c:2722-2725` (`next_len >= cur_len && 4*(next_len-cur_len) + (bsr32(cur_offset) - bsr32(next_offset)) > 2`); threshold 6 for two-ahead (:2742-2755).
- **Mechanism**: trades 4 cost units per extra match byte against the log2 offset-bits difference — a fractional-bit proxy so a barely-longer but much-farther match doesn't win. Replaces zlib's raw length compare.
- **Ours**: YES — `parse/lazy.rs:22-33` (`better_match`, identical formula, thresholds 2/6); also reused by the L1 lazy-peek (`parse/fast.rs:1935-1942`).
- **Applicability**: L1 peek + L5-9. Already everywhere it applies.

## P5. Reduced-depth deferred searches (half/quarter)
- **Who**: libdeflate: 1-ahead at `max_search_depth >> 1` (`vendor/libdeflate/lib/deflate_compress.c:2712-2721`), 2-ahead at `>> 2` (:2742-2755). No other vendor does this (zlib-ng's lazy search runs full chain).
- **Ours**: YES — `parse/lazy.rs:221` (`depth >> 1`), `:243` (`depth >> 2`).

## P6. nice_len immediate-accept and skip
- **Who**: all. libdeflate `deflate_compress.c:2681-2692` (lazy takes `cur_len >= nice_len` at once, skips rest); zlib-ng `match_tpl.h:165-166`; gzip `deflate.c:479-482`; per-level nice values in vendor-structure-comparison §1.
- **Ours**: YES — `parse/lazy.rs:198`; matchfinder-side exits `matchfinder/hc.rs:447,583`.

## P7. Adaptive minimum match length from literal diversity
- **Identifiers**: libdeflate `choose_min_match_len` / `calculate_min_match_len` / `recalculate_min_match_len`. libdeflate only.
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:2299-2378`.
- **Mechanism**: counts distinct literals in the first 4096 bytes of a block; few distinct literals (binary-ish data) ⇒ raise min_len so cheap literals aren't displaced by expensive short matches. Re-calibrated mid-block in lazy at ~10000 bytes then geometrically (`deflate_compress.c:2624-2651`), using freq cutoff `literal_freq >> 10` (:2359-2378). Also feeds the near-optimal default costs.
- **Parameters**: table 0-5 used lits→9, 6-7→8, 8-9→7, 10-15→6, 16-44→5, 45-79→4, else 3 (:2299-2314); depth clamps <5→4, <10→5, <16→7 (:2318-2325); inputs < 512 bytes → 3 (:2340-2352).
- **Ours**: YES — `parse/mod.rs:611-648` (80-entry `MIN_LENS`, same clamps), recalibration `parse/mod.rs:651-659` + `parse/lazy.rs:168-176`.
- **Applicability**: L2-12 (already). Note this is content-ADAPTIVE parameterization inherited from libdeflate, distinct from the content-DETECTOR gates rule 3 bans.

## P8. Length-3 distance gates (TOO_FAR family)
- **Who**: gzip `TOO_FAR 4096` — a len-3 match farther than 4096 becomes a literal (`vendor/gzip/deflate.c:129-132, 711-717`). libdeflate: greedy gate 4096 (`deflate_compress.c:2573-2575`), lazy gate 8192 (`:2666-2668`). zlib-ng: **removed entirely** (no `TOO_FAR` anywhere; agent-verified grep). igzip: moot (min match 4). zopfli: score-1 penalty for dist > 1024 in the pre-parse (`lz77.c:265-271`).
- **Ours**: YES (libdeflate values) — greedy 4096 `parse/greedy.rs:175`; lazy 8192 `parse/lazy.rs:190`. L1 hash3 gate `L1_HASH3_MAX_DIST = 32768` i.e. no gate (`parse/fast.rs:924`).
- **Applicability**: parameter worth sweeping per level — three vendors chose three different answers (4096 / 8192 / none).

## P9. deflate_quick — static-trees-only single-probe L1 strategy
- **Identifiers**: zlib-ng `deflate_quick` (Intel-contributed). zlib-ng only.
- **Who**: `vendor/zlib-ng/deflate_quick.c:48-138` (provenance :5-11); routed at level 1 (`deflate.c:113`).
- **Mechanism**: emits **static-Huffman codes directly as symbols are found** — no symbol buffer, no histogram, no dynamic tree, no per-block flush decision. One static block can span the whole stream (`s->block_open` state, `deflate.h:152-155`; open/close `deflate_quick.c:31-46`). Single hash probe (`quick_insert_value` returns only the chain head, `insert_string_tpl.h:42-56`), min match 4, first-4-bytes compared as one u32 (`deflate_quick.c:99-101`), then functable `compare256` (:103). Flush only when the pending buffer nears full (:69-74).
- **Parameters**: hash = Knuth multiplicative 16-bit (H1); `DEFLATE_QUICK_LIT_MAX_BITS 9` tight output bound (`zutil.h:77-78`).
- **Ours**: NO — our L1 (`parse/fast.rs`) builds a dynamic Huffman table per 64 KiB block (`fast.rs:3276-3292`). We pay histogram + tree build + header per 64 KiB; quick pays a fixed ~9-bit/literal ratio cost and zero table work. Our L0 emitter is static-or-stored (`parse/mod.rs:781-827`) but L0 is stored-only in production (`deflate/mod.rs:184-196`).
- **Applicability**: L0/L1 fast path as a wall lever; the ratio loss vs our per-64K dynamic tables is the open question (O-2).

## P10. deflate_medium — match at n and n+1 with overlap fixup
- **Identifiers**: zlib-ng `deflate_medium` (Intel, Arjan van de Ven). zlib-ng only.
- **Who**: `vendor/zlib-ng/deflate_medium.c` (provenance :3-5); levels 3-6 (`deflate.c:123-126`).
- **Mechanism**: keeps the match at the position AFTER the current match (`next_match`) for the next iteration instead of re-searching (`deflate_medium.c:210-244`); `fizzle_matches` extends the next match leftward byte-at-a-time, shortening the current one, when the byte before next-match also matches (:128-176). `early_exit = level < 5` disables the n+1 probe for L3-4 (:184-185) — medium at L3/L4 is greedy with medium's insert policy. Insert interior iff `match_len <= 16 * max_insert_length` (:70).
- **Ours**: NO — nothing keeps the post-match search result across iterations; greedy re-searches from scratch after the jump.
- **Applicability**: L2-6 greedy band. A middle point between greedy and lazy that amortizes one search per match.

## P11. igzip L3 map-then-extend (ICF match map)
- **Identifiers**: igzip `gen_icf_map_h1`/`gen_icf_map_lh1` + `set_long_icf_fg` + `compress_icf_map_g`.
- **Who**: `vendor/isa-l/igzip/igzip_icf_body.c:252-330` (driver), :80-136 (map gen), :33-73 (fix-up greedy), :143-231 (greedy walk). AVX2/AVX512 lane versions `vendor/isa-l/igzip/igzip_gen_icf_map_lh1_06.asm`, `igzip_set_long_icf_fg_06.asm:202-256`.
- **Mechanism**: three decoupled sweeps per 4 KiB chunk (`MATCH_BUF_SIZE 4096`, `igzip_level_buf_structs.h:8`): (1) build a per-position token map with ONE hash probe and ONE 8-byte compare each — match lengths capped 4..8; (2) for every map length ≥ 8, run the full compare from +8 and re-stamp following positions with decreasing lengths of the same distance so a greedy chooser later sees the best forward extension anywhere; (3) walk the map greedily with a free 1-step lazy (peek pos+1) and pack literal pairs. This decouples the serial hash-probe dependency chain from length extension — which is what makes it vectorizable: the AVX512 map gen does 16 positions/iteration with `vpgatherdd`/`vpscatterdd` on the hash table itself and lane-rotate lazy compare (`igzip_gen_icf_map_lh1_06.asm:279-406`), tolerating intra-vector hash aliasing (no `vpconflictd` — later scatter wins, matches re-verified).
- **Ours**: NO — all our parsers are serial per-position probe-then-extend loops.
- **Applicability**: the only vendor design that vectorizes MATCH FINDING itself. Candidate for a distinct fast-path strategy; requires the ICF-style token intermediate (see O6). Our ICF-uniform-record flush redesign was FALSIFIED for the emit path (`parse/mod.rs:968-1465`) — that falsification was about emit cadence, not about map-style matching, so it does not bind this.

## P12. Two-positions-per-iteration software pipeline
- **Identifiers**: igzip L0 body loop; ours "SF2".
- **Who**: `vendor/isa-l/igzip/igzip_body.asm:246-330` — computes hash and hash2 for pos and pos+1, probes both, 8-byte xor-compares both candidates, and speculatively loads both literal Huffman codes so the no-match path writes two literals in one `write_bits`.
- **Ours**: PARTIALLY — L1 `fastloop_l1` issues both head reads before consuming either (SF2, `parse/fast.rs:2407-2492`) but does not speculatively pre-load literal codes or emit two literals per write.
- **Applicability**: L0/L1 fast. The speculative literal-code load half is the missing part.

## P13. Literal-run skip / acceleration on incompressible data
- **Who**: igzip L1/L2 asm only: after `SKIP_SIZE_BASE (2<<10)` bytes without a match, emit literals **8 per iteration with no hash lookups at all** (PSHUFB packs 8 literals into 4 dual-literal ICF tokens); skip amount grows from `SKIP_BASE 32` by `SKIP_RATE 2` past `SKIP_START 512` up to `MAX_SKIP_SIZE 128` per round; matches decrement the skip level (`vendor/isa-l/igzip/igzip_icf_body_h1_gr_bt.asm:54-59, 129-130, 234-236, 534-640`). The C base path has no skipping. libdeflate/zlib-ng/gzip: no equivalent (libdeflate relies on depth + nice_len only).
- **Ours**: PARTIALLY — L0-only `ACCEL` ramp: step `1 + (misses >> 0)` after 3 consecutive misses, cap 8, skipped positions untouched (`parse/fast.rs:1242-1262, 2376-2402`). Nothing at L1+ (`parse/lazy.rs`: no skip-ahead; agent-verified).
- **Applicability**: L1 fast and L2-9 — igzip's version is the standing explanation for its incompressible-data wall dominance; our random-1M cells are where this shows.

## P14. Early-exit on first non-improving candidate
- **Identifiers**: zlib-ng `EARLY_EXIT_TRIGGER_LEVEL`. zlib-ng only.
- **Who**: `vendor/zlib-ng/match_tpl.h:13, 117, 233-238` — for levels < 5, the chain walk stops outright on the first candidate that fails to improve best_len (rather than continuing to depth).
- **Mechanism**: cheap-level bet that the first (nearest) chain entry is the best; trades ratio for a shorter dependent-load chain.
- **Ours**: NO — our hc walk always runs to depth/nice_len/staleness.
- **Applicability**: L2-4 greedy. Cheap experiment: one branch in `matchfinder/hc.rs`.

## P15. Near-optimal parse (cached-matches min-cost path)
- **Identifiers**: libdeflate `deflate_compress_near_optimal`. libdeflate only (zopfli's is Z1, a different design).
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:3592-3849` (driver), :3327-3399 (`deflate_find_min_cost_path`), :3416-3530 (`deflate_optimize_and_flush_block`); levels 10-12.
- **Mechanism**: bt matchfinder records ALL matches per position into a cache (M14); backward DP over `optimum_nodes` picks min-cost path under a fractional-bit cost model; per pass, real Huffman codes are rebuilt from the chosen path and the EXACT block cost recomputed (`deflate_compute_true_cost` :2889-2921); iterate up to `max_optim_passes`, stop when improvement < `min_improvement_to_continue`, restore prior path if final pass regressed ≥ `min_bits_to_use_nonfinal_path` (:3518-3527). Extras: all-literals candidate block (:3433-3439), static-optimized candidate for blocks ≤ `max_len_to_optimize_static_block` (:3451-3466), cost carry-over blending between blocks keyed on observation similarity (`deflate_adjust_costs` :3207-3296), match-cache-rewind block splitting (:3782-3826), per-level params (passes 2/4/10, improve 32/16/1, nonfinal 32/16/1, static 0/1000/10000, `deflate_compress.c:3974-4004`).
- **Ours**: YES — `parse/near_optimal.rs` is a faithful port: cache 1,500,000 (`:53-57`), backward DP (`:328-457`), same per-level params (`level.rs:240-272`), all-literals + static candidates (`:347, :356-365`), only-literals feedback (`:560-564`). BIT_COST 16, NOSTAT 13/13/10 (`costs.rs:26-32`).
- **Applicability**: our L10-12 (Step 3); already present.

## P16. Zopfli squeeze — iterated entropy-driven re-parse
See section K (Z1-Z8). Listed here for parse-strategy completeness: forward DP over exact
per-symbol entropy costs, iterated with stat feedback — the only vendor parse that
re-derives the cost model from its own output.

---

# B. Match finding

## M1. Single-probe hash table (chainless)
- **Who**: igzip ALL levels — one 16-bit position per hash slot, overwrite on insert; accept iff `dist-1 < dist_mask` (`vendor/isa-l/igzip/igzip_base.c:61-67`, `igzip_icf_base.c:76-82`). **igzip has no chains anywhere, at any level** (agent-confirmed absence). zlib-ng deflate_quick likewise probes only the head (`deflate_quick.c:93`).
- **Parameters**: igzip table entries — L0/L1: 8K (`IGZIP_LVL0_HASH_SIZE`/`LVL1` = `8 * IGZIP_K`, `vendor/isa-l/include/igzip_lib.h:121-126`); L2/L3: 32K (:122-128). See vendor-structure-comparison §3 for the working-set argument (8K entries stays L1d-resident).
- **Ours**: YES for L0/L1 (`parse/fast.rs:1057-1065` head table, overwrite-on-collision) — but at `HASH_BITS = 16` = 64K entries, 8x igzip, plus a 32K hash3 side table (`fast.rs:882`). The comment justifying the widening is the instruction-count argument vendor-structure-comparison §3 refutes with the D1-miss measurement.
- **Applicability**: L0/L1. The 8K-vs-64K table size is the single clearest parameter difference against the vendor that beats us at L0/L1.

## M2. ht_matchfinder — 2-entry buckets inline in the hash table
- **Identifiers**: libdeflate `ht_matchfinder` (level 1 only).
- **Who**: `vendor/libdeflate/lib/ht_matchfinder.h:50-60` — `hash_tab[1<<15][2]` of s16; `HT_MATCHFINDER_HASH_ORDER 15`, `BUCKET_SIZE 2`, `MIN_MATCH_LEN 4`.
- **Mechanism**: probe both slots (2 candidates, still no chains); insert shifts slot0→slot1 and writes cur to slot0 (`ht_matchfinder.h:131-170`). Second slot only extended if its first 4 bytes AND the 4 bytes at `best_len - 3` match (:153-155). Single hash precomputed one ahead + `prefetchw` (:114-119). `skip_bytes` inserts skipped positions maintaining bucket order (:220-228).
- **Ours**: NO — L1 is single-probe (a BUCKET2 tune knob exists only behind the dead `l1-tune` feature, `fast.rs` tune block). Our second table (hash3) is a different-key table, not a second way.
- **Applicability**: L1. A 2-way 32K-slot table is a midpoint between igzip's 8K single-probe and our 64K+32K two-table layout — 128 KiB working set vs our ~192 KiB.

## M3. Classic hash chains (head[] + prev[])
- **Who**: gzip `vendor/gzip/deflate.c:292-295` (`INSERT_STRING`), 15-bit head; zlib-ng `match_tpl.h:15-18` (`GOTO_NEXT_CHAIN` via `prev[cur_match & wmask]`), 16-bit head table; zopfli `hash.c:100-114` (head 65536, prev chains, 15-bit hashval).
- **Parameters**: chain depth = per-level `max_chain` (tables in vendor-structure-comparison §1); zopfli `ZOPFLI_MAX_CHAIN_HITS 8192` (`util.h:84`).
- **Ours**: NO (this exact layout) — we use libdeflate's hc layout (M4) instead, which is the same idea with the prev[] table renamed `next_tab` and dual hash lengths.

## M4. hc_matchfinder — dual-length hashing (hash3 singleton + hash4 chains)
- **Identifiers**: libdeflate `hc_matchfinder`.
- **Who**: `vendor/libdeflate/lib/hc_matchfinder.h:112-131` — `hash3_tab[1<<15]` (singleton, 24-bit key), `hash4_tab[1<<16]` (chain heads), `next_tab[32768]`, all s16.
- **Mechanism & parameters**: hash3 probed once, only when `best_len < 4` (:241-256); hash4 chain walked with `depth_remaining` decrement (:193, 272), stale cutoff `node <= cur_pos - 32768` (:212), nice_len exits (:279-287). Specialized first-match loop (4-byte compare then `lz_extend`) vs longer-match loop (last-4 + first-4 prefilter, M9). Next position's both hashes precomputed + `prefetchw` (:234-239). Deliberate per-call-site inlining for branch prediction/registers (:76-98).
- **Ours**: YES — `matchfinder/hc.rs` is a port with identical geometry (`HC_HASH3_ORDER 15`, `HC_HASH4_ORDER 16`, `hc.rs:73-98`), same multiplier, same prefilter (`hc.rs:530-549`), same prefetch (`hc.rs:315-330`). Two FALSIFY records live in the file: chain-node prefetch (net loss, `hc.rs:388-403`) and prefilter operand hoisting (Dr up, `hc.rs:499-526`). The remaining gap vs libdeflate is register discipline, not structure — vendor-structure-comparison §4.

## M5. bt_matchfinder — binary tree with in-walk re-rooting
- **Identifiers**: libdeflate `bt_matchfinder` (levels 10-12).
- **Who**: `vendor/libdeflate/lib/bt_matchfinder.h:70-101` — `hash3_tab[1<<16][2]` (2-way LRU), `hash4_tab[1<<16]` tree roots, `child_tab[2 * 32768]`.
- **Mechanism**: one traversal both collects matches and re-roots the tree at the current position (`pending_lt_ptr`/`pending_gt_ptr` re-linking, :207-254); common-prefix restart `len = MIN(best_lt_len, best_gt_len)` (:244-253); records strictly increasing lengths only, max `nice_len - 2` matches; nice_len stop adopts matched node's children (:232-236); depth exhaustion severs the subtree (sets children to INITVAL, :256-260). Window slide is explicit in the driver every 32768 bytes (`deflate_compress.c:3601-3602, 3653-3659`).
- **Ours**: YES — `matchfinder/bt.rs:29-56` is a port (same orders, one contiguous ~512 KiB `Box<[i16]>`). Additionally ours has `matchfinder/lzfind.rs` — an ECT/7-zip `Bt3Zip` (CRC-table 3-byte hash, cyclic tree, full Pareto frontier up to 512 pairs) used only by ultra; no vendored counterpart (ECT itself is not vendored).

## M6. Position rebasing (i16 saturating subtract) vs. sliding rebuild
- **Who**: libdeflate: positions are s16 relative to a base; at `cur_pos == 32768` all tables rebase via branchless `0x8000 | (v & ~(v >> 15))` (`vendor/libdeflate/lib/matchfinder_common.h:148-149`; hc trigger `hc_matchfinder.h:205-209`), SIMD'd at compile time (S7). zlib-ng/gzip instead memcpy the window down 32K and fix every head/prev entry: zlib-ng `fill_window` `deflate.c:1183-1296` + dispatched SIMD `slide_hash` (`arch/generic/slide_hash_c.c:19-56`, AVX2 `slide_hash_avx2.c:20-46` `_mm*_subs_epu16`); gzip branchy scalar (`deflate.c:556-566`). igzip: neither — 16-bit positions wrap mod 64K and stale entries fail the `dist_mask` test by construction; reset fills entries with `total_in & 0xffff` so aliases read as ≈64K-far (`igzip.c:878-934`).
- **Ours**: YES (libdeflate scheme) — `matchfinder/common.rs:180-186` branchless rebase, `hc.rs:214-218`; streaming buffer slides via `copy_within` + O(1) state shift (`deflate/mod.rs:349-354`, `parse/mod.rs:482-498`). igzip's zero-cost wrap trick is noteworthy as the cheapest of the three (no rebase pass at all) — possible for our single-probe fast path.

## M7. Chain quartering on good-enough match (good_match)
- **Who**: gzip `deflate.c:404-407`; zlib-ng `match_tpl.h:76-77` (`if (best_len >= s->good_match) chain_length >>= 2`). Per-level `good` values in vendor-structure-comparison §1. libdeflate has no equivalent (depth is halved on lazy deferral instead, P5).
- **Ours**: NO — we have no good_match analogue; our depth budget is fixed per call.
- **Applicability**: L5-9. Alternative depth-shaping lever to libdeflate's; zlib-ng runs both this AND deeper base chains.

## M8. Endpoint prefilter before match extension
- **Who**: three generations of the same idea. gzip: 2-byte `scan_end1/scan_end` check (`deflate.c:451-454`). zlib-ng: width-adaptive — best_len < 4 → 2-byte, 4-7 → 4-byte, ≥8 → 8-byte compares of scan start AND at `best_len-1` adjusted so a hit still extends (`match_tpl.h:63-68, 131-152`); deliberately reads uninitialized lookahead, made safe by window over-zeroing (M12). libdeflate: `load_u32(matchptr + best_len - 3) == load_u32(in_next + best_len - 3) && load_u32(matchptr) == load_u32(in_next)` (`hc_matchfinder.h:300-304`).
- **Ours**: YES (libdeflate form) — `matchfinder/hc.rs:530-549`.

## M9. Word-at-a-time match extension (and SIMD compare)
- **Who**: libdeflate `lz_extend` — u64 XOR, 4x unrolled, bsf/bsr of the differing word (`matchfinder_common.h:178-222`). igzip `compare258` — 8-byte XOR + tzcnt in C (`huffman.h:260-314`), asm tiers 16B SSE `pcmpeqb/pmovmskb`, 32B AVX2, 64B AVX512 k-mask (`igzip_compare_types.asm:43,102,189,289`). zlib-ng `compare256` — dispatched SIMD: SSE2 16B (`compare256_sse2.c:16-69`), AVX2 2x32B (`compare256_avx2.c:19-44`), AVX512 masked 64B (`compare256_avx512.c:20-60`), NEON/Power/RVV (`functable.c:264-411`); generic 64-bit SWAR (`compare256_c.c:16-63`). gzip: byte-wise, 8 bytes per loop iteration unrolled (`deflate.c:467-472`). zopfli `GetMatch` size_t loads (`lz77.c:297-331`).
- **Ours**: PARTIALLY — scalar u64 XOR + trailing_zeros only (`matchfinder/common.rs:97-133`); **no SIMD compare anywhere in the level engine** (agent-verified). Caveat before borrowing: vendor-structure-comparison §4 shows our matchfinder gap is register pressure, and wall-delta ≠ instruction-delta.
- **Applicability**: L2-12; biggest effect where matches are long (nice_len 258 levels).

## M10. Hash-insert limiting inside matches
- **Who**: igzip `ISAL_LIMIT_HASH_UPDATE` — always on; after a match only positions +0,+1,+2 are inserted, not the match body (`vendor/isa-l/include/igzip_lib.h:119`, `igzip_base.c:73-85`). zlib-ng: insert interior only if `match_len <= max_insert_length` (= max_lazy) (`deflate_fast.c:79-87`), medium: `<= 16 *` that (`deflate_medium.c:70`); else re-hash only at the jump target. gzip: same idea via `max_insert_length` (`deflate.c:203-207`). libdeflate: the opposite — `skip_bytes` inserts EVERY skipped position (`hc_matchfinder.h:360-399`), buying ratio with insert cost.
- **Ours**: SPLIT — fast path YES: L0 inserts 2, L1 inserts 3 then jumps (`parse/fast.rs:1090, 1204, 1985-2000`). L2-9 NO: we follow libdeflate and insert everything (`parse/greedy.rs:177-185` → `skip_bytes`).
- **Applicability**: L2-9. A ratio-vs-wall knob three vendors set three ways; never swept on our shape.

## M11. Rolling-hash offset-chain search (fast_zlib) — L9 only
- **Identifiers**: zlib-ng `longest_match_roll` + `insert_string_roll` (Konstantin Nosov's fast_zlib). zlib-ng only.
- **Who**: `vendor/zlib-ng/match_tpl.h:83-115, 179-228` (attribution :6-8); selected for level ≥ 9 (`deflate_slow.c:25-31`, `deflate.c:1194-1197`).
- **Mechanism**: L9 switches the WHOLE hash to a 3-byte rolling hash (`(h<<5)^c`, 15-bit — H3) so len-3 matches are findable; the search re-hashes scan interior bytes to jump to the most distant relevant chain, and on each improvement re-scans prev[] across the match interior and probes head[] at len-2 to jump chains again — finding matches whose interior is hashed even when their head chain is exhausted.
- **Ours**: NO — our L9 is Lazy2 over the same hc finder as L5-8, depth 600.
- **Applicability**: L9 (max-compression cell). The one vendor structure aimed exactly at "-9 must be smallest".

## M12. Window over-initialization for speculative reads
- **Who**: zlib-ng `WIN_INIT = STD_MAX_MATCH` — window zeroed up to 258 bytes past valid data, tracked by `high_water`, so the prefilter's uninitialized reads are defined (`deflate.h:396`, `deflate.c:1261-1292`).
- **Ours**: DIFFERENT — we require the caller to pad the input buffer with 16 zero bytes (`INPLACE_TAIL_PAD = BUF_PAD = 16`, `deflate/mod.rs:93,143`) and clamp reads; matchfinders refuse to search within 5 bytes of end (`hc.rs:282-284, 635-637`).

## M13. Run-aware hashing (zopfli same[] + second hash)
- **Identifiers**: zopfli `ZOPFLI_HASH_SAME` / `ZOPFLI_HASH_SAME_HASH`.
- **Who**: `vendor/zopfli/src/zopfli/hash.c:116-136`; use `lz77.c:481-519`.
- **Mechanism**: `same[pos]` = incremental count of following identical bytes; second hash `val2 = ((same-3) & 255) ^ val` folds RUN LENGTH into the key. FindLongestMatch skips `min(same0, same1)` bytes before comparing (`lz77.c:481-490`) and switches to the val2 chain once `bestlength >= same[hpos]` (:509-519) — degenerate runs stop clogging the chain.
- **Ours**: NO in the level engine; ultra (zopfli port) has it (`parse/ultra/hash.rs`).
- **Applicability**: L8-9 and near-optimal — chain pollution by runs is exactly what depth budgets burn on.

## M14. Match caching
- **Who**: two designs. libdeflate near-optimal: all matches per position appended to `match_cache` (`MATCH_CACHE_LENGTH = 300000*5`, `deflate_compress.c:158`), block-end on overflow, headers encode count+literal (:3706-3709); cache walked backward for split rewind (:3782-3826). zopfli: per-block `length[]`+`dist[]` plus compressed sublen cache of `ZOPFLI_CACHE_LENGTH 8` 3-byte change-point triples per position — 28 bytes/input byte (`cache.c:33, 54-108`) — amortizing 15 re-parses.
- **Ours**: YES (libdeflate form) — `parse/near_optimal.rs:53-57` (1,500,000 + slop); ultra has the zopfli cache (`parse/ultra/cache.rs`).

## M15. Large-match emit loop (run fast path)
- **Identifiers**: igzip `LARGE_MATCH_MIN`.
- **Who**: `vendor/isa-l/igzip/igzip_body.asm:41-44, 561-735` — compare up to `MAX_EMIT_SIZE = 258*16` bytes; a match ≥ `LARGE_MATCH_MIN 264` enters an emit loop writing repeated 258-length codes with NO re-searching, then re-hashes only `4*LARGE_MATCH_HASH_REP` trailing positions.
- **Mechanism**: igzip's answer to zero pages / long runs — one compare covers up to 4128 bytes, then pure emission. Related: zlib-ng `deflate_rle` (strategy Z_RLE only) does distance-1 runs with a dedicated `compare256_rle` and no hash table at all (`deflate_rle.c:24-80`).
- **Ours**: NO — a 300 KiB zero block costs us ~1160 searches+skips of len-258 matches; igzip does ~73 compares. No RLE detection anywhere in the level engine (agent-verified).
- **Applicability**: L0-9. Clean, condition-triggered (match length), not content detection.

## M16. Next-position hash precompute + prefetchw
- **Who**: libdeflate — both hashes for pos+1 computed and both buckets `prefetchw`'d before searching pos (`hc_matchfinder.h:234-239`; ht :114-119; bt :170-178).
- **Ours**: YES — `matchfinder/hc.rs:315-330`; fast path uses a longer explicit pipeline instead (`PF_DIST = 4`, `parse/fast.rs:1074, 2295-2299`) — see C1.

---

# C. Hash functions

## H1. Multiplicative hashes — the multipliers side by side

| Vendor | Function | Constant | Shift/width | Key bytes | Cite |
|---|---|---|---|---|---|
| libdeflate | `lz_hash` | 0x1E35A7BD | `>> (32-bits)` | 4 (hash4), low-24 (hash3) | `vendor/libdeflate/lib/matchfinder_common.h:168-172` |
| zlib-ng L1-8 | `HASH_CALC` | 2654435761 (Knuth) | `>> 16`, mask 16 bits | 4 | `vendor/zlib-ng/insert_string_p.h:12-14` |
| igzip (non-SSE4.2 fallback) | `compute_hash` | 0xB2D06057, applied twice | `*k >> 16` twice | 4 | `vendor/isa-l/igzip/huffman.h:207-226` |
| ours | `lz_hash` | 0x1E35A7BD | `>> (32-bits)` | 4 / low-24 | `src/compress/deflate/matchfinder/common.rs:49-51` |

- **Ours**: YES — libdeflate's constant everywhere (fast head, hash3 via 24-bit key, hc, bt).

## H2. CRC32-instruction hashing
- **Who**: igzip primary hash when SSE4.2: `_mm_crc32_u32(0, data)` (`vendor/isa-l/igzip/huffman.h:207-213`); bare `crc32` in asm (`huffman.asm:243-249`); 4-at-a-time crc dictionary preload (`igzip_deflate_hash.asm:105-132`). zlib-ng: **removed** in 2.3.x (older zlib-ng had it; agent-verified absent).
- **Mechanism**: 1 instruction, 3-cycle latency, excellent mixing; frees a multiply port in the probe dependency chain.
- **Ours**: NO — multiplicative only.
- **Applicability**: L0/L1 fast path where the hash is on the critical path; both x86 SSE4.2 and aarch64 (`crc32cw`) have it. zlib-ng removing it is weak counter-evidence worth understanding before borrowing.

## H3. Rolling hash (h<<5)^c
- **Who**: gzip (`UPDATE_HASH`, `deflate.c:282`, 15-bit); zlib-ng level 9 only (`insert_string_p.h:28-35`, 15-bit, enables len-3 finds — M11); zopfli (`hash.c:96-98`, 15-bit).
- **Ours**: NO in the level engine; ultra's zopfli port has it.

## H4. SIMD multiply-add hash (vpmaddwd)
- **Who**: igzip L3 `compute_hash_mad` — two rounds of `PROD1*low16 + PROD2*high16`, `PROD1 0xFFFFE84B`, `PROD2 0xFFFF97B1` (`vendor/isa-l/igzip/huffman.h:228-245`); vectorized as two `vpmaddwd` over 16 lanes (`igzip_gen_icf_map_lh1_06.asm:206-207`).
- **Mechanism**: chosen because it vectorizes (crc32 doesn't); the price of P11's vector map generation.
- **Ours**: NO (no vectorized hashing).

## H5. Hash-mask shrink for tiny inputs
- **Who**: igzip — `if (hash_mask > 2*avail_in) hash_mask = (1 << bsr(avail_in)) - 1` (`vendor/isa-l/igzip/igzip.c:1402-1403`; streaming :1545-1547): a 100-byte input touches a 128-entry table, not 8K.
- **Ours**: NO — full-size tables regardless of input size (pooled, but reset cost scales with table size, `parse/fast.rs:2776-2795`).
- **Applicability**: L0/L1 small-file cells (our `small-256B`/`one-1B` style corpus rows).

---

# D. Block splitting and sizing

(Block size limits per vendor: vendor-structure-comparison §2. This section is the
*mechanisms*.)

## D1. Symbol-budget block termination
- **Who**: zlib lineage — block ends when the symbol buffer fills: zlib-ng `sym_buf` / `LIT_MEM` split layout, `lit_bufsize = 1 << (memLevel+6)` = 16384 default (`vendor/zlib-ng/deflate.c:289, 352-360`; tally-full check `deflate_p.h:64-115`); gzip `LIT_BUFSIZE 0x8000` = 32K symbols (`vendor/gzip/trees.c:118-126`, flush `trees.c:1006-1010`). libdeflate uses BOTH byte and sequence budgets (300000 bytes / 50000 seqs; FAST path 65535/8192 — `deflate_compress.c:66-108`).
- **Ours**: YES (libdeflate form) — `SOFT_MAX_BLOCK_LENGTH 300000` (`parse/mod.rs:65`), `SEQ_STORE_LENGTH 50000` (`parse/mod.rs:71, 594-607`), fast 65536/1 MiB blocks (`parse/fast.rs:1549, 1560`). The 3x symbol-budget difference vs zlib-ng was inherited untested (structure-comparison §2).

## D2. Observation-based adaptive split detector
- **Identifiers**: libdeflate `init_block_split_stats` / `do_end_block_check`. libdeflate only (zlib lineage has D3 instead; zopfli has D4).
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:2056-2218`; active levels 2-12 (not fastest).
- **Parameters**: `NUM_OBSERVATION_TYPES 10` = 8 literal classes (`((lit >> 5) & 0x6) | (lit & 1)`) + 2 match classes (len < 9 / ≥ 9) (:439-443, 2109-2126); check every 512 new observations once block ≥ `MIN_BLOCK_LENGTH 5000` with ≥ 5000 bytes remaining (:2199-2207); split when SAD of cross-multiplied expected/actual ≥ cutoff `num_new * 200/512 * num_obs`, plus `block_length/4096 * num_obs` length bias and a short-block penalty when < 10000 bytes and < 8192 items (:2141-2192).
- **Ours**: YES — `block_split.rs` is an exact port (types :57-66, cadence :18-21, test :104-130).

## D3. Periodic estimated-cost early flush
- **Who**: gzip — every 4096 symbols (`(last_lit & 0xfff) == 0`, level > 2), flush early if matches are sparse and estimated output < half the input (`vendor/gzip/trees.c:995-1005`). zlib-ng inherits the equivalent logic in its tally path.
- **Ours**: NO (we have D2 instead, which subsumes it).

## D4. Exact-cost block-split search (zopfli)
- **Who**: `vendor/zopfli/src/zopfli/blocksplitter.c` — `FindMinimum` 9-point sectioning search (`NUM 9`, :60; exhaustive when range < 1024, :45-57); "estimate" is the REAL deflate bit count `ZopfliCalculateBlockSizeAutoType` (:108-111 → `deflate.c:610-621`); repeatedly splits the LARGEST splittable block (:195-213) while `splitcost <= origcost` (:251-255), max `blocksplittingmax 15` blocks (`util.c:34`); splits on a GREEDY parse ("unintuitively... better blocks", :294-296); re-splits after optimization and keeps the cheaper of the two splittings (`deflate.c:871-893`).
- **Ours**: YES in ultra — uncapped recursive greedy-bisection, 9-point search (`FIND_MINIMUM_NUM 9`, `parse/ultra/blocksplit.rs:13`), dynamic-only split cost (:24-27), parallel candidate evaluation. NO on the numeric ladder (L10-12 use D2 + cache rewind).

## D5. Final-block absorption
- **Who**: libdeflate `choose_max_block_end` — if fewer than `soft_max + MIN_BLOCK_LENGTH` bytes remain, run the block to input end so no tiny final block exists (`vendor/libdeflate/lib/deflate_compress.c:2380-2387`).
- **Ours**: YES — `parse/mod.rs:575-581`.

## D6. Per-chunk Huffman adaptation via buffer-sized blocks (igzip)
- **Who**: igzip levels 1-3 — a block is (re)opened per level_buf fill; the ICF buffer's capacity IS the block size, so tables adapt per ~level_buf-sized chunk. Default buffer = LARGE (+256 KiB tokens) for all levels (`vendor/isa-l/include/igzip_lib.h:310-329`); flush `flush_icf_block` `igzip.c:200-228`.
- **Mechanism**: block sizing as a MEMORY parameter, not a heuristic — the user picks the buffer, ratio follows.
- **Ours**: NO (fixed budgets). Relevant as prior art for "block size is a legal tuning parameter" under rule 3's parameter-tuning carve-out.

---

# E. Huffman coding

## E1. Heap-based tree construction (zlib lineage, igzip)
- **Who**: zlib-ng `build_tree` heap + `pqdownheap`, depth[] tie-break (`vendor/zlib-ng/trees.c:151-269`); gzip same ancestry; igzip heap build in asm (`vendor/isa-l/igzip/proc_heap.asm:61`, macros `heap_macros.asm`; C fallback `proc_heap_base.c`), heap padded to ≥ 2 symbols (`huff_codes.c:749-762`).
- **Ours**: NO (we use E2). Equivalent output; different constant factor.

## E2. Counting-sort + in-place non-leaf tree construction
- **Identifiers**: libdeflate `deflate_make_huffman_code` (7-Zip HuffEnc.c-derived, comment `deflate_compress.c:1315-1316`).
- **Who**: `vendor/libdeflate/lib/deflate_compress.c:846-995, 1318-1396` — counting sort with heapsort tail for high frequencies (:846-906), entries pack `sym | freq << 10` (`NUM_SYMBOL_BITS 10`), two-queue merge without a heap (:939-995).
- **Ours**: YES — `huffman/fast.rs` full port (pack :20, sort :63-119, tree :213), thread-local sort scratch (:123-151).

## E3. Length limiting: demotion heuristic vs count-shuffle vs package-merge
- **Who**: three tiers. libdeflate: overlong depths demoted to deepest available shorter length — "not optimal... good enough" (`deflate_compress.c:1022-1090`, esp. 1073-1082). igzip `fix_code_lens`: classic count-shuffle rebalance (`huff_codes.c:857-924`), limits `MAX_HUFF_TREE_DEPTH 15`. zlib-ng `gen_bitlen` overflow fixup (`trees.c:281-366`). zopfli: EXACT boundary package-merge (Katajainen), two lookahead chains per list, `2n-4` BoundaryPM runs, flat node pool (`vendor/zopfli/src/zopfli/katajainen.c:143-254`); maxbits 15 (lit/dist), 7 (precode) (`deflate.c:546-578, 208`).
- **Ours**: BOTH — approximate demotion in `huffman/fast.rs:279` (all levels); exact package-merge in `huffman/optimal.rs:46` (ultra + deflate64 only).
- **Applicability**: running package-merge at L10-12 instead of the demotion heuristic is a size-only lever libdeflate itself declines; would need measurement.

## E4. 14-bit litlen cap enabling 4-literal batching
- **Who**: libdeflate `MAX_LITLEN_CODEWORD_LEN 14` (not 15) so 4 litlen codewords (56 bits) always fit one 64-bit flush (`vendor/libdeflate/lib/deflate_compress.c:113-117, 1968-1999`).
- **Ours**: YES — limits 14/15/7 (`tables.rs:26-32`), 4-literal groups (`parse/mod.rs:1040-1051`), const proof `can_buffer` (`parse/mod.rs:943-952`).

## E5. Precode / code-length RLE and its optimizations
- **Who**: all emit RFC1951 symbols 16/17/18. libdeflate: trailing trim + offset lens made contiguous with litlen lens so RLE runs CROSS the boundary (`deflate_compress.c:1575-1598`); permutation trim to min 4 explicit lens (:1617-1623). zopfli, two exclusives: `TryOptimizeHuffmanForRle` mutates the histogram to favor precode-RLE-able length patterns, keeping whichever total (tree+data, data priced with TRUE counts) is smaller (`deflate.c:434-560`); and all 8 combinations of use-16/17/18 are size-evaluated per header (`deflate.c:251-290`). igzip `rl_encode` (`huff_codes.c:1105-1212`).
- **Ours**: PARTIALLY — standard RLE with all three symbols (`huffman/header.rs:71-140`); ultra has the 8-combo search (`parse/ultra/huffman/header.rs` doc :9-17) and RLE count shaping (`huffman/optimal.rs:331`). The level engine has neither zopfli extra, same as libdeflate. Cross-boundary RLE runs: worth verifying ours does the contiguous-lens trick.

## E6. Stored/static/dynamic selection by exact cost
- **Who**: libdeflate computes exact whole-bit cost of all three per block up front, ties prefer uncompressed then static (`deflate_compress.c:1732-1808`). zlib-ng compares `opt_lenb`/`static_lenb`/`stored_len+4` (`trees.c:661-684`). igzip: dynamic-vs-static at table-build time (`huff_codes.c:1611-1648`), stored fallback per ICF block when predicted bits ≥ type0 size and raw input recoverable (`igzip.c:356-438`). gzip adds a whole-file rewrite: if the entire member so far ≥ stored size and output is seekable, rewrite as STORED member (`trees.c:899-912`).
- **Ours**: YES — exact three-way per block (`parse/mod.rs:673-768, 841-877`), stored wins ties then static. No whole-file rewrite (we don't need it; per-block stored bounds expansion).

## E7. Pre-canned trained Huffman tables + one-pass emit (igzip L0)
- **Identifiers**: igzip `hufftables_default` / `isal_deflate_pass`.
- **Who**: `vendor/isa-l/igzip/hufftables_c.c` (generated, 5422 lines; two variants keyed on `IGZIP_HIST_SIZE`, :45, :1849) — a trained dynamic-Huffman code with a precomputed 109-byte header, baked at build time; L0 emits final bits in ONE pass against it (`igzip.c:569-572`). Header replayed by memcpy with BFINAL toggled by xor of bit 0 (`igzip.c:1829-1835`). Training corpus: NOT stated anywhere in the tree (regenerable via `generate_custom_hufftables.c:29-46`). Public `isal_update_histogram` + `isal_create_hufftables` support the "semi-dynamic" segment pattern (`huff_codes.c:633-688, 1357-1529`).
- **Mechanism**: removes the entire second pass AND the table build from the hot path; ratio depends on corpus resemblance to training data.
- **Ours**: NO — nothing trained; our L1 builds real per-block tables, our L0 is stored-only.
- **Applicability**: L1 fast path (a tuned static-ish table beats RFC1951 fixed codes at identical speed). No env knob needed — a compiled-in table is a constant.

## E8. ICF two-pass semi-dynamic coding (igzip L1-3)
- **Identifiers**: igzip intermediate compression format, `struct deflate_icf`.
- **Who**: token format `vendor/isa-l/igzip/encode_df.h:9-32` — 32-bit: `lit_len:10` (257..512 = len 3..258 via `LEN_OFFSET 254`), `lit_dist:9` (30 = literal; **31..287 = a SECOND literal**, so one token carries two literals), `dist_extra:13`. Pass 1 tokenizes + accumulates `isal_mod_hist {d_hist[30]; ll_hist[513]}` (`igzip_lib.h:283-287`); block end builds an exact per-block Huffman code and pass 2 re-walks tokens emitting bits (`flush_icf_block` `igzip.c:200-228`). The 288-entry `dist_lit_table` union (`huff_codes.h:156-168`) is what lets the dual-literal token encode with one lookup; `expand_hufftables_icf` pre-splices length extra bits into `code_and_extra` so pass 2 has no length logic at all (`huff_codes.c:1531-1561`).
- **Mechanism**: histogram-exact tables (better than streamed heuristics) AND a vectorizable second pass (S5/O6) — the token array is fixed-width, so emission gathers.
- **Ours**: NO. Our seq store (`Sink.seqs`) is close in spirit (tokenize then emit) but records are not designed for vectorized emission; our AVX2 gather-emit and ICF-uniform-record experiments on the EMIT side were FALSIFIED (`parse/mod.rs:968-1465`). The remaining unexplored half is pass-1 vectorization (P11), which requires this format.

## E9. Bitbuf-safety code-length re-limit (56-bit unchecked writes)
- **Who**: igzip `are_hufftables_useable` — if the max lit+len+dist+extra bit chain exceeds `MAX_BITBUF_BIT_WRITE 56`, REGENERATE the code with `MAX_SAFE_LIT_CODE_LEN 13` / `MAX_SAFE_DIST_CODE_LEN 12` (`vendor/isa-l/igzip/huff_codes.c:1315-1408`, `huff_codes.h:63-64`) — guaranteeing the asm can emit a whole match in one unchecked write.
- **Ours**: EQUIVALENT BY CONSTRUCTION — our caps (14+5+15+13 = 47 bits ≤ 63) are static, proven by `can_buffer` (`parse/mod.rs:943-952`); igzip needs the dynamic check because its caps are 15/15.

## E10. Degenerate-code handling
- **Who**: libdeflate: < 2 used symbols ⇒ always emit 2 codewords of length 1 (zlib-compat; Windows Explorer rejects empty offset codes) (`deflate_compress.c:1342-1378`); zlib-ng forces ≥ 2 nonzero codes (`trees.c:216-224`); zopfli `PatchDistanceCodesForBuggyDecoders` (`deflate.c:74-99`).
- **Ours**: YES — `huffman/fast.rs:414-429` (< 2 used symbols ⇒ two length-1 codewords).

---

# F. Bit output

## F1. Wide bit buffer with branchless whole-word flush
- **Who**: libdeflate: `bitbuf_t` = machine word, `BITBUF_NBITS 63`, flush = ONE unaligned 8-byte store then `bitcount &= 7` — branchless, gated only by a precomputed `out_fast_end` (`deflate_compress.c:669-751, 1726-1727`). zlib-ng: 64-bit `bi_buf` (`deflate.h:60-62`), `send_bits` three-way branch flushing 8 bytes via `put_uint64` (`trees_emit.h:47-66`). igzip: `BitBuf2` — **unconditional** 8-byte store per `write_bits`, advance by whole bytes, safety from 8 bytes of slop (`bitbuf2.h:83-130, 51-57`). gzip: 16-bit `bi_buf` (`bits.c:81-92`) — the ancestor's main handicap.
- **Ours**: YES — 63-bit accumulator, `flush_word_unchecked` single unaligned store (`bitstream.rs:15, 164-185`), `reserve(block*2+16)` up front (`parse/mod.rs:1490`). Five emit-cadence redesigns already FALSIFIED in place (`parse/mod.rs:968-1465`).

## F2. Combined length+distance emission
- **Who**: zlib-ng `zng_emit_dist` packs lencode + len-extra + distcode + dist-extra into ONE ≤ 48-bit `send_bits` using merged `lbase_extra[]`/`dbase_extra[]` tables (`trees_emit.h:20-22, 116-156`; tables `trees_tbl.h:128, 139`). libdeflate: precomputed "full" length codewords (litlen code + extra bits concatenated per length 3-258), so a match is 3 ADD_BITS + ≤ 2 flushes (`deflate_compress.c:1638-1694`).
- **Ours**: YES (libdeflate form) — merged `codeword | nbits << 24` LUTs, full-length LUT, one-shift offset entries (`parse/mod.rs:889-936, 1506-1523`).

## F3. Deferred/unsafe writes inside a proven budget
- **Who**: igzip `write_bits_unsafe` — three deferred accumulates then one flush per token, legal by E9's ≤ 56-bit guarantee (`encode_df.c:24-33`). libdeflate: ONE output-bounds check per block; if the precomputed exact cost fits, no per-write checks at all (`deflate_compress.c:1811-1818`), cost asserted equal post-hoc (:2034).
- **Ours**: YES — `add_bits_raw` unchecked accumulate (`bitstream.rs:128-140`) under the `can_buffer` proof; block-level `reserve`.

## F4. Vectorized second-pass encode (gather-based)
- **Who**: igzip `encode_deflate_icf` AVX2/AVX512 — 8/16 ICF tokens per iteration, `vpgatherdd` from `lit_len_table`/`dist_lit_table`, vector code concatenation + length prefix-sums, ONE scalar bitbuf drain per group (`vendor/isa-l/igzip/encode_df_04.asm:178-243`, `encode_df_06.asm:94-101, 189`).
- **Ours**: NO — and our AVX2 gather-merge emit experiment was FALSIFIED (`parse/mod.rs:968-1027`). The falsification was measured WITHOUT the fixed-width ICF token layout that makes igzip's version work; re-attempt would have to change the seq store format first (E8), not just the emit loop.

---

# G. SIMD and CPU dispatch

## G1. Runtime dispatch mechanisms — three designs
- **zlib-ng functable**: struct of function pointers, lazily initialized via stubs or eagerly via `__attribute__((constructor))`, atomic stores + fence (`vendor/zlib-ng/functable.c:18-33, 440-499`). Dispatched on the COMPRESS path: `longest_match`, `longest_match_roll`, `compare256`, `slide_hash`, `crc32(_copy)`, `adler32(_copy)` (`functable.c:422-432`). ISA ladder: SSE2→SSSE3→SSE4.1/2→PCLMUL→AVX2+BMI2→AVX512{F,DQ,BW,VL}→VNNI→VPCLMUL; ARM armv6→NEON→CRC→PMULL+EOR3; Power, RISC-V, s390, LoongArch (`functable.c:110-417`).
- **ISA-L multibinary**: self-patching function pointers (`mbin_dispatch_init5/6`, `vendor/isa-l/include/multibinary.asm:201-259`); suffix tiers `_base`(C)/`01`(SSE4.2)/`02`(AVX)/`04`(AVX2)/`06`(AVX512); the WHOLE BODY of the encoder is dispatched, not helpers — e.g. `isal_deflate_body`, `icf_body_lvl1/2/3`, `encode_deflate_icf`, `gen_icf_map_lh1`, `set_long_icf_fg`, `isal_update_histogram` (`vendor/isa-l/igzip/igzip_multibinary.asm:84-132`). One asm source expanded 3x by `%rep` for arch variants (`igzip_body.asm:122-133, 782-788`).
- **libdeflate**: runtime dispatch ONLY for crc32/adler32 (static volatile fn-ptr trampoline, `crc32.c:235-253`); the compress path has ZERO runtime dispatch — SIMD only in compile-time-selected matchfinder init/rebase (G6).
- **Ours**: ALMOST NONE — no dispatch in L0-12; only ultra's cached `is_x86_feature_detected!("avx")` cost sweep (`parse/ultra/squeeze.rs:516-531`) and NEON equivalent. Prefetch hints are our only hot-path intrinsics (`matchfinder/common.rs:143-160`).
- **Applicability**: the vendor evidence CONTRADICTS "dispatch everything": the fastest scalar encoder (libdeflate) dispatches nothing on compress. What zlib-ng gains from dispatch is G2/G3; what igzip gains is whole-kernel asm. Dispatch is a delivery mechanism, not a technique — decide per-kernel.

## G2. SIMD match-length compare (compare256)
- See M9 for who/parameters. Dispatched in zlib-ng; tiered asm in igzip; ABSENT in libdeflate and in us.

## G3. SIMD slide_hash
- **Who**: zlib-ng — `_mm*_subs_epu16` over head (65536) + prev (wsize) on slide (`slide_hash_avx2.c:20-46`; many arches, `functable.c:252-415`).
- **Ours**: NOT APPLICABLE-ish — our rebase is the libdeflate branchless scalar (`matchfinder/common.rs:180-186`), auto-vectorizable; we never measured whether rustc actually vectorizes it (open question O-8).

## G4. Vectorized match-map generation with gather/scatter
- See P11 — `vpgatherdd`/`vpscatterdd` on the hash table itself, `vplzcntq` lengths, lane-rotate lazy (`igzip_gen_icf_map_lh1_06.asm:279-406`). The only vendor instance of SIMD match FINDING.

## G5. SIMD histogram
- **Who**: igzip `isal_update_histogram` asm 01/04 (`vendor/isa-l/igzip/igzip_update_histogram.asm:257`).
- **Ours**: NO — scalar freq counting in the Sink.

## G6. Compile-time SIMD matchfinder init/rebase
- **Who**: libdeflate — AVX2/SSE2/NEON bulk-fill and saturating-subtract for the s16 tables, selected by `#ifdef` not runtime (`vendor/libdeflate/lib/x86/matchfinder_impl.h:33-120`, `arm/matchfinder_impl.h:33-76`).
- **Ours**: PARTIALLY — same table layout, scalar (auto-vec candidate) rebase; alignment guarantees (`MATCHFINDER_MEM_ALIGNMENT 32`) not replicated explicitly.

## G7. Hand-written whole-kernel assembly
- **Who**: igzip only — body/finish/icf_body/encode/map kernels are asm on x86 AND aarch64 (`vendor/isa-l/igzip/aarch64/`). gzip has dead ASMV hooks with no shipped match.S (`vendor/gzip/deflate.c:229-233`).
- **Ours**: NO (two prefetch asm hints only). Our register-pressure findings (structure-comparison §4) are exactly the problem asm solves; Rust-side alternatives (fewer live locals, monomorphized loops) come first.

---

# H. Memory and cache technique

## C1. Prefetching — who prefetches what, at what distance

| Vendor | What | Distance | Cite |
|---|---|---|---|
| libdeflate | `prefetchw` next position's hash bucket(s) | 1 position | `hc_matchfinder.h:234-239`, `ht_matchfinder.h:119`, `bt_matchfinder.h:177-178` |
| igzip | implicit via 2-position pipeline; `FIX_CACHE_READ` option | — | `igzip_body.asm:246-330`, `options.asm:36-63` |
| zlib-ng, gzip, zopfli | none in the encoder | — | agent-verified absence |
| ours | `prefetch_write` head slot | 4 positions (`PF_DIST 4`) fast; 1 position hc | `parse/fast.rs:1074, 2295-2299`; `hc.rs:315-330` |

- FALSIFY on record: prefetching chain NODES lost (`hc.rs:388-403`). Note vendor-structure-comparison §3: our fast-path prefetch exists to paper over the 8x table widening; igzip needs none because its table fits L1d.

## C2. Single-allocation state carving and cacheline layout
- **Who**: zlib-ng — window, prev, head, pending_buf, state carved from ONE zalloc with 64-byte-aligned sub-buffers (`deflate.c:165-227`); state struct `ALIGNED_(64)` with fields grouped by cacheline (`deflate.h:138-314`). libdeflate: one aligned alloc per compressor, per-level struct sizing (`deflate_compress.c:3891-3906`). deflate_medium packs its two match structs to 16 bytes (`deflate_medium.c:179-181`).
- **Ours**: PARTIALLY — per-object boxing with thread-local pooling (hc ~256 KiB pooled `hc.rs:100-210`; head tables `fast.rs:2755-2795`; near-optimal one ~7 MB arena `near_optimal.rs:106-117`); no deliberate cacheline grouping of hot loop state (relevant to the §4 register/spill finding).

## C3. User-supplied working memory (igzip level_buf)
- **Who**: igzip — all working state is caller-provided `level_buf`, sizes graded MIN/SMALL/MEDIUM/LARGE/XL (`igzip_lib.h:294-329`); ICF token space is "whatever remains" of the buffer (`igzip.c:316-321`), making block size a memory knob (D6).
- **Ours**: NO (internal pooling); the graded-sizes idea maps onto our T>1 shared-memory-per-thread-count parameter tuning allowance.

## C4. Window over-zeroing (WIN_INIT) — see M12.

## C5. 16-bit position wrap with init-to-current trick — see M6 (igzip).

## C6. Stored-block direct copy bypassing all buffers
- **Who**: zlib-ng (madler's optimized `deflate_stored`) — stored blocks copied straight `next_in → next_out`, header length bytes patched after a dummy `zng_tr_stored_block`, checksum fused in `read_buf`, window replayed afterward only to keep a valid dictionary for level switches (`vendor/zlib-ng/deflate_stored.c:27-136`).
- **Ours**: YES — L0 is a true stored passthrough in 65535-byte sub-blocks with no window at all (`deflate/mod.rs:55, 637-676, 184-196`).

---

# I. Checksums

## I1. Implementations per vendor (compress-relevant)
- **libdeflate**: crc32 slice-by-8 default (`crc32.c:175-232`); x86 PCLMUL→VPCLMUL/AVX512 ladder (`x86/crc32_impl.h:51-153`); ARM crc-insn/pmull hybrids (`arm/crc32_impl.h`); adler32 SSE2→AVX512-VNNI, NEON-dotprot (`x86/adler32_impl.h:35-129`). Dispatch = volatile fn-ptr trampoline (`crc32.c:235-253`).
- **zlib-ng**: braid (generic), **Chorba** (arXiv 2412.16398; generic/SSE2/SSE4.1 — `arch/generic/crc32_chorba_c.c:14`, `functable.c:104-153`), PCLMUL/VPCLMUL, ARM crc+PMULL/EOR3, Power vpmsum, s390 VX; `crc32_small` shortcut for < 32-byte header bytes (`crc32.c:21-27`).
- **igzip**: `crc32_gzip_refl` PCLMUL by-8 / VPCLMUL by16 (`crc32_gzip_refl_by8.asm`, dispatch `crc/crc_multibinary.asm:222-223`); adler stored as `B|(A-1)` so zero-init serves both checksums (`igzip.c:118-132`).
- **pigz**: zlib's crc32 + `crc32_comb` copied in with the per-block combine shift PRECOMPUTED once per run (`g.shift = x2nmodp(g.block, 3)`, `pigz.c:1355-1397, 4343`).
- **Ours**: `crc32fast` crate v1.4 (dispatch inside the crate) — `Cargo.toml:318`.

## I2. Checksum/compression overlap
- **Who**: zlib-ng FUSES checksum with the window copy — `crc32_copy`/`adler32_copy` inside `read_buf` (`deflate_p.h:159-182`), every SIMD checksum has a `_copy` twin. igzip: separate SIMD pass after each compression pass over the consumed span (`igzip.c:134-148, 468-469`). libdeflate: one whole-input crc call AFTER compression (`gzip_compress.c:67-79`).
- **Ours**: SPLIT — streaming T1 interleaves per-chunk while hot (`deflate/mod.rs:399-415`; the separate sweep cost 29.3 ms/232 MiB, comment :571-574); whole-buffer T1 is a separate pass (`deflate/mod.rs:277, 304`); T>1 per-chunk + ordered combine (`pipelined.rs:442-485`).
- **Applicability**: whole-buffer T1 is the odd one out; fusing crc into the (single) input read is the zlib-ng pattern we haven't applied there.

## I3. CRC combining for parallel streams — see J5.

---

# J. Parallelism (pigz; ours)

## J1. Job pipeline with ordered write thread
- **Who**: pigz — reader appends `struct job` (seq, in-buf, dict, check lock) to a compress list (`pigz.c:1578-1591`); N compress threads pull; ONE write thread consumes a seq-sorted list (`:1911-1921, 1987-1991`); threads launched lazily up to `-p` (`:2227-2231`), parked between streams; yarn.c monitor primitives. Input pool `(procs<<1)+3` buffers bounds read-ahead (`:521, 1619`).
- **Ours**: YES (same shape) — workers claim chunks off an atomic counter + dedicated in-order writer, `std::thread::scope` (`scheduler.rs:104-296`).

## J2. Dictionary carryover between chunks
- **Who**: pigz — each job's zlib stream gets `deflateSetDictionary(last 32K of previous chunk)` (`DICT 32768`, `pigz.c:463, 1754-1772`); previous chunk kept alive as the next dict (`:2205-2219`). Cost: full hash rebuild per chunk inside zlib.
- **Ours**: YES — each chunk seeds the prior 32 KiB (`scheduler.rs:270-277`, `deflate/mod.rs:157-171`).

## J3. Chunk-boundary byte alignment WITHOUT sync-flush bytes
- **Who**: pigz — after `deflate(Z_BLOCK)`: if `bits & 1` → Z_SYNC_FLUSH (5-byte empty stored block); **else if `bits & 7` → repeated `deflatePrime(10, 2)`** — 10-bit EMPTY STATIC blocks (btype 01 + 7-bit EOB) injected until bit position ≡ 0 mod 8 (`pigz.c:1831-1846`). Max 4 bytes wasted, usually fewer, vs always-5 for sync flush. Single-thread path does the identical dance so threaded output is byte-identical to unthreaded (`:2432-2455`, comment :261-271).
- **Ours**: PARTIALLY — we byte-align every non-final chunk with the full Z_SYNC_FLUSH empty stored block (`deflate/mod.rs:109-130, 229-231`). Adopting the 10-bit-empty-static-block padding saves up to ~4 bytes per chunk seam (input-length-only chunk grid `pipelined.rs:77-103` — e.g. 8 MiB file = 16 seams) and pigz proves T-invariance is preservable.

## J4. Independent-chunk mode
- **Who**: pigz `-i` — `setdict=0`; every chunk ends sync marker + full-flush marker (9-byte `00 00 FF FF 00 00 00 FF FF` signature) so history never crosses and scanners can find block starts (`pigz.c:4517, 1836, 1852-1853`).
- **Ours**: NO (no user-facing equivalent; not a goal — noted for completeness).

## J5. Parallel CRC with combine
- **Who**: pigz — each thread computes its chunk CRC AFTER queuing the write job (overlapping the write), write thread combines with the precomputed per-block shift (`pigz.c:1923-1939, 2011, 1397`).
- **Ours**: YES — per-chunk `Hasher`, ordered `combine` after join (`pipelined.rs:442-485`). We have not precomputed the fixed-block-size combine operator (pigz's `x2nmodp` trick) — ours re-derives per combine inside crc32fast.

## J6. Rsyncable content-defined flush points
- **Who**: pigz — rolling hash `((hash<<1)^byte) & ((1<<12)-1)`, mean 4 KiB blocks (`RSYNCBITS 12`, `pigz.c:469-514, 2130-2166`); hit offsets shipped to workers as varints so flush points are identical at any thread count (`:2039-2068, 1792-1808`). gzip 1.6: NOT present (upstream added 1.7).
- **Ours**: YES (separate path) — `parallel.rs:613, 661`, `io.rs:237-254`.

## J7. Zopfli as pigz level 11
- **Who**: pigz bundles zopfli (Mar-2013 state), `-11` runs `ZopfliDeflatePart` per chunk with the 32K dictionary MEMCPY'd in front of the input (zopfli has no setDictionary) (`pigz.c:435-437, 1766-1784, 1867-1869`); `-I/-J/-O` map to iterations/maxsplits/oneblock (`:4478, 4260, 4492`).
- **Ours**: analogous crown path exists (ultra via `-F/-I/-J`, `compress/mod.rs:102-125`) but is not on the numeric ladder and not parallelized per-chunk. pigz's dict-prepend trick is the known recipe for parallel ultra.

---

# K. Zopfli-class techniques (levels 10+ reference)

## K1. Squeeze: forward DP over a statistical cost model
- **Who**: `vendor/zopfli/src/zopfli/squeeze.c:217-309` (`GetBestLengths` — forward DAG relaxation, float costs), `:317-336` traceback, `:338-389` `FollowPath` re-running match finding to recover distances (a second pass per iteration).
- **Ours**: YES (ultra) — `parse/ultra/squeeze.rs`, plus a vendored-nowhere extension: multi-seed restarts (greedy + literal-dominant + fixed-price seeds, `squeeze.rs:1099-1118`) and an AVX/NEON vectorized cost sweep (`squeeze.rs:516-706`).

## K2. Cost models and min-cost pruning
- **Who**: `GetCostFixed`/`GetCostStat` (`squeeze.c:125-157`); symbol costs are ENTROPY (`log2(sum)-log2(count)`, `tree.c:71-94`), not code lengths; `GetCostModelMinCost` precomputes the cheapest possible edge and the DP skips positions that can't improve (`squeeze.c:163-198, 287-293`).
- **Ours**: YES (ultra port).

## K3. Iteration with stat feedback and plateau randomization
- **Who**: `ZopfliLZ77Optimal` (`squeeze.c:446-526`): default 15 iterations (`util.c:31`); TRUE cost measured each round by the exact oracle (:492); after iteration 5 on a cost plateau, stats reset to best + `RandomizeStatFreqs` (1-in-3 chance a freq is replaced by another symbol's; MWC PRNG, fixed seed) (:84-101, 512-517); subsequent stats blended `1.0*current + 0.5*last` (:505-511). No early exit ever.
- **Ours**: YES (ultra) — MWC randomization `squeeze.rs:232-280`, iterations tunable (`tuning.rs:19-40`).

## K4. Longest-match cache with compressed sublen — see M14.

## K5. The exact-bits oracle used for EVERYTHING
- **Who**: `ZopfliCalculateBlockSize` (`deflate.c:584-621`) prices a candidate as REAL output bits (package-merge + RLE-trick + 8-combo tree encoding included) — used for iteration scoring, block splitting, AND the btype choice; `AddLZ77BlockAutoType` even re-parses the whole block under fixed-tree costs when `size < 1000 || fixedcost <= dyncost*1.1` (`deflate.c:747-800`).
- **Ours**: YES (ultra `deflate_size.rs`).

## K6. Long-repetition DP shortcut
- **Who**: `ZOPFLI_SHORTCUT_LONG_REPETITIONS` — inside the DP, when `same > 2*258`, 258 positions are cost-modeled as forced max-length matches with NO match search (`util.h:114`, `squeeze.c:251-271`).
- **Ours**: YES (ultra port). The same idea scaled down is M15 for the fast levels.

## K7. Master blocks
- **Who**: `ZOPFLI_MASTER_BLOCK_SIZE 1000000` bounds memory (`util.h:60`, `deflate.c:908-924`).
- **Ours**: ultra operates whole-buffer with its own budgeting.

## K8. Why zopfli is slow (for the record)
15 × (full DP + FollowPath re-search) (`squeeze.c:486-519`); exact-cost evaluation at every
probed split point, recursively (`blocksplitter.c:43-128`); 8 tree encodings per size query
(`deflate.c:282-287`); the split+optimize sequence re-evaluates the final stream
(`deflate.c:871-893`). Any Step-3 ladder placement must budget these four multipliers.

---

# L. Special-case fast paths

## L1. Tiny-input passthrough
- **Who**: libdeflate — inputs ≤ `55 - 4*level` bytes skip compression entirely and go to stored (`deflate_compress.c:3918, 4034-4036`).
- **Ours**: PARTIALLY — ported (`level.rs:291-294`) but **not wired into the routing path** (agent-verified). Free correctness-safe win on the 1B/256B corpus rows.

## L2. Repeated-char prefix detection with canned header (igzip)
- **Who**: stateless mode only — if the first 8 bytes are all-0x00 or all-0xFF, scan the run; runs ≥ `MIN_REPEAT_LEN 4096` (`repeated_char_result.h:51`) emit from a PRE-BAKED dynamic-Huffman header as 258-length matches with code-arithmetic tail fill (`igzip.c:614-742`).
- **Ours**: NO. Condition-triggered (prefix bytes), not corpus detection; adjacent to M15.

## L3. Guaranteed-bound stored rewrite (igzip stateless)
- **Who**: `isal_deflate_stateless` caps avail_out at the stored-size bound, attempts compression, and on overflow REWRITES the entire output as stored blocks (`igzip.c:1358-1481`, rewrite :1443-1468) — hard bound input + 5/64K + wrapper.
- **Ours**: per-block stored fallback gives the same bound incrementally (`parse/mod.rs:725-737`); `compress_bound` equivalent in `deflate/mod.rs:70-76`.

## L4. Empty input
- **Who**: libdeflate 5-byte stored block (`deflate_compress.c:2406-2414`); igzip 10-bit static empty block at trailer (`igzip.c:1994-2012`).
- **Ours**: YES — empty stored BFINAL block (`deflate/mod.rs:217-218`; T>1 `pipelined.rs:175-190`).

## L5. Streaming state machine (igzip) — noted, not indexed in depth
igzip's resumable `ZSTATE_*` machine with mirrored TMP states and a 16-byte tmp_out for
avail_out < 8 (`igzip_lib.h:178-206`, `igzip.c:392-394, 541-611`) is contract machinery for
its zlib-like API; our streaming model (chunk + resumable parsers,
`deflate/mod.rs:332-354`, `parse/mod.rs:482-529`) covers our CLI contract without it.

---

# Where we stand — techniques by verdict

**We already do (YES)**: P1-P8, P15, M4-M6, M8, M14, M16, H1, D1, D2, D5, E2, E4, E6, E10,
F1-F3, C1, C6, J1, J2, J5, J6, K1-K6 (ultra), L3-L4.
**Partial**: P12 (no speculative literal loads), P13 (L0 only), M9 (no SIMD), M10 (fast path
only), E5 (level engine lacks zopfli extras), G6 (unverified auto-vec), C2 (no cacheline
discipline), I2 (whole-buffer T1 not fused), J3 (5-byte sync flush, not 10-bit padding),
L1 (ported, unwired).
**We don't (NO)**: P9, P10, P11, P14, M1-at-igzip-size, M2, M7, M11, M13 (level engine),
M15, H2, H3-H5, D3, D6, E1, E7, E8, E9-style dynamic re-limit, F4, G1-style dispatch, G4,
G5, G7, C3, J4, L2.

# Highest-value candidates to borrow

Ranked by how clearly the source shows a vendor doing something we don't — not by imagined
speedup. No performance numbers; none have been measured on our shape.

1. **igzip L1-size hash table (M1)** — igzip runs L0/L1 on an 8K-entry single-probe table
   (`igzip_lib.h:121-126`) where we run 64K + a 32K side table; structure-comparison §3
   already ties our D1-miss rate and IPC collapse to the widening. The one candidate with a
   banked measurement pointing at it.
2. **igzip large-match emit loop (M15)** — a match ≥ 264 triggers emit-without-research up
   to 258·16 compare bytes (`igzip_body.asm:41-44, 561-735`); we re-search every 258 bytes
   of a run. Condition-triggered, no content detection, applies L0-9.
3. **igzip literal-run skip at L1/L2 (P13)** — hash-free 8-literals-per-iteration emission
   with a growing skip after 2 KiB of no matches (`igzip_icf_body_h1_gr_bt.asm:54-59`); our
   only equivalent is the L0 ACCEL ramp. Directly targets incompressible-data wall cells.
4. **pigz 10-bit empty-static-block chunk padding (J3)** — `deflatePrime(10, 2)` loops
   instead of a 5-byte sync flush per T>1 chunk seam (`pigz.c:1838-1845`); strictly smaller
   output at identical wall, and pigz proves T-invariance survives.
5. **zlib-ng deflate_quick static-stream L1 (P9)** — an entire L1 strategy with zero
   per-block table work (`deflate_quick.c:48-138`); we pay histogram + tree + header every
   64 KiB. The ratio give-back is the open question; igzip's trained table (E7) is the
   middle path.
6. **libdeflate tiny-input passthrough, wired (L1)** — already ported at `level.rs:291-294`;
   it is routing, not research.
7. **zlib-ng early-exit heuristic for L2-4 (P14)** — one branch (`match_tpl.h:233-238`)
   ending the chain walk on the first non-improving candidate at cheap levels.
8. **igzip crc32-instruction hash (H2)** — one instruction replacing multiply+shift on the
   probe critical path (`huffman.h:207-213`); tempered by zlib-ng having removed theirs.
9. **zlib-ng longest_match_roll for L9 (M11)** — rolling 3-byte hash + offset-chain interior
   search (`match_tpl.h:83-228`) is the only vendor structure aimed purely at the -9 size
   cell, where per-label size is our binding constraint.
10. **igzip ICF two-pass + vectorized encode (E8/P11/F4)** — the only vendor architecture
    that vectorizes matching and emission; large change, and our emit-side gather FALSIFY
    (`parse/mod.rs:968-1027`) was measured without the token format that makes it work, so
    it is re-attemptable but expensive.

# Open questions (source cannot settle; need measurement)

- **O-1**: 8K vs 64K L1 head table on OUR loop shape — does shrinking recover igzip's IPC,
  and what does it cost on text ratio (the stated reason we widened)? Also the M2 midpoint
  (32K×2-way).
- **O-2**: deflate_quick-style static emission at L1 — what is the per-label size give-back
  vs our per-64K dynamic tables on the real corpus, and does the wall win cover it? Same
  question for an igzip-style trained default table (E7) — and igzip's own training corpus
  is unrecorded, so we'd train our own.
- **O-3**: hash-insert policy at L2-9 (M10) — libdeflate inserts everything, igzip inserts
  3, zlib-ng gates on match length; never swept on our shape.
- **O-4**: SIMD compare256 (M9) — instruction-count math says yes, but our two register-
  pressure FALSIFYs and the wall≠instructions rule say measure first, especially at short
  average match lengths.
- **O-5**: J3 padding — exact bytes saved per corpus file at T>1 (seam count × ≤4 bytes),
  and confirmation the T>1 = T1-or-smaller invariant still closes.
- **O-6**: block sizing as a parameter (D1/D6) — libdeflate's 300K/50K vs zlib-ng's 16K
  symbols was inherited untested (structure-comparison §2); igzip shows block size can be a
  plain memory parameter.
- **O-7**: does len-3 gating (P8) at 4096/8192 beat zlib-ng's no-gate on our corpus, and is
  zopfli's dist>1024 score penalty better than either at high levels?
- **O-8**: does rustc auto-vectorize our scalar rebase (`matchfinder/common.rs:180-186`)
  and the fill loops libdeflate hand-vectorizes (G6)? Check the emitted asm before adding
  intrinsics.
- **O-9**: E5 cross-boundary precode RLE — verify whether our header builder lets RLE runs
  cross the litlen/offset boundary as libdeflate does; if not it is a few-bytes-per-block
  size leak at every level.
- **O-10**: M11 at L9 — fast_zlib's own claim is large depth-budget savings at equal size;
  vendored source proves the structure, not the payoff on silesia/our corpus.
