# vv — vendor-falsification catalog: igzip (ISA-L) + zlib-ng

Standing mechanism catalog, written 2026-09-05 from a line-by-line read of the vendored
sources. Every winning lever this campaign has had started as a vendor diff (3-for-3);
this file is the per-mechanism diff for the two hottest uint vendors that are NOT
libdeflate (libdeflate itself is already the port — `src/compress/ldx/`).

**Sources (verified pins, identical to this worktree's submodule pins):**
- ISA-L 2.31.1, branch `gzippy-stopping-points`, tip `55916f5c` — read from
  `/Users/jackdanger/www/gzippy/vendor/isa-l/` (the brain worktree's `vendor/isa-l/` is
  an empty submodule dir; the pinned commit is checked out in the parent repo and is
  the SAME commit — `git -C vendor/isa-l rev-parse HEAD` = `55916f5c…`).
- zlib-ng 2.3.90 develop, tip `da22434b` — `vendor/zlib-ng/` same deal.
- Cross-reference index: `docs/vendor-technique-index.md` (technique IDs P*/M*/H*/D*/E*/F*/G*/C*
  below are ITS section IDs — G3 = its §G.3 SIMD slide, G5 = §G.5 histogram, F4 = §F.4, D6 = §D.6, E8 = §E.8, O-8 = its
  open question 8). OUR-SIDE line numbers inherited from that index carry its merge
  state (the unmerged `perf/t1-output-cap` stack changes `parse/`, `deflate/mod.rs`,
  and the pick-min plumbing) — ours-side cites are ATTRIBUTIONS to re-verify at the
  tip you build on; the VENDOR-side cites are pinned and exact.
- Structure/parameters: `docs/vendor-structure-comparison.md` (§1 level tables, §2 block
  sizing, §3 hash geometry, §4-void per-component split).
- Board + named cells: `docs/plan-2026-09-one-encoder.md` §2b (30 failing of 1,320 @ ee0c1d2c).

## The cadence band (named cell classes, plan §2b — the ONLY legitimate targets)

| class | named cells | margin |
|---|---|---|
| `CADENCE-L23` | dd79_bin6 L2/L3 vs gzip/pigz | up to +0.93% |
| `CADENCE-L5` | access.log L5 vs gzip / pigz | +1.07% / +0.75%; minjs.min.js L5 +0.19/+0.16% |
| `LADDER-L4` | data.sqlite L4 vs gzip/pigz | +0.13% / +0.17%; KNOWN_SAGS (tabular,3)/(binary,3) |
| `LEN3-L3` | access.log L3 vs libdeflate | +0.42% (far-len3 machinery still legacy-arm; #364) |
| `HAIRLINE-L79` | weights.safetensors L7-L9 | +0.00-0.02% |
| `HAIRLINE-NOCOMP` | photo.jpg L1-L3 vs gzip +0.04%; movie.mp4 L6 vs libdeflate +0.0008% | hairlines |
| `WALL` | codegen wall vs libdeflate (T1 in-process 1.03-1.11x; Ir rows in `tests/fingerprints/ir_budget.tsv`) | axis, not a size cell |
| `T>1` | T>1 parity + wall | axis (Phase 1 state carry) |

Note the binding shape: at L2-L9 we are byte-tied to libdeflate on nearly every T1 cell —
so a size lever can only pay where we currently LOSE to gzip/pigz (the cadence band), and
any parse change must first clear `scripts/campaign/tie-guard.sh` (zero tie tolerance).
Wall levers operate on an axis where we are already decision-identical; size levers must be
adopted ONE level at a time through the #363/#364 exception-retirement template.

## The three death modes (taxonomy used by every card)

1. **TIE-FLIP** — the mechanism changes emitted bytes on cells we currently tie or win
   (99% of size cells are byte-ties vs libdeflate), re-opening closed cells. Guard:
   `scripts/campaign/tie-guard.sh <ref>` BEFORE the edit, `board-size.sh tune` after;
   two prior levers died exactly here (hash3 chaining 6 closed/12 flipped; zlib
   good_match 31 closed/17 flipped, data.csv L2 1.0000 → 1.0431).
2. **WALL-NO-SHIP** — clean size story, but `fulcrum try --threads 1,4` on the frozen box
   refuses: the T1 wall leg rejected it, or the T4 leg regressed where T1 passed. Receipt:
   L1 hash3 measured 15-50% self-tax (legacy codegen); the pipeline re-scheduling levers
   all died here.
3. **EMPTY-CLASS** — the mechanism cannot pay because the target class has no slack:
   header-construction class closed at ~0.001% vs ~0.01% needed
   (`src/compress/deflate/huffman/fast.rs:432` record); the L1 size class GONE after the
   pivot (plan §2b); or the mechanism's give-back exceeds the band's total margin at the
   level measured (hard stop #3: shallow+deep before believing any "not the cost").

For levers that touch bitstream/splicing, a fourth is always in force: **CORRUPTION**
(roundtrip sha through gzip/pigz/libdeflate at T1 AND T>1; `parity-census.sh` for T>1).

## Instruments (no hand-rolls; hard stop #6)

- `scripts/campaign/board-size.sh tune` — cheapest falsifier for ANY size claim (2 min, deterministic).
- `fulcrum try <ref> --threads 1,4 [--scope "levels=…;threads=…;corpus=a,b"] [--scope-sentinels N]`
  — the promotion adjudicator (5. wave-runner.sh queues them). `--scope` measures the
  declared sub-grid in full AND grades out-of-scope SENTINEL cells by clause 3 only
  (`fulcrum/src/promote.rs:1274-1381`); scope must be a STRICT sub-grid of the declared
  `--levels/--threads/--corpus` (a scope covering the ENTIRE declared grid is REFUSED),
  so declare a wider `--levels/--corpus` superset than you scope. `fulcrum try` defaults
  to `--threads 1` — a T>1 change judged without `--threads 1,4` reads NO-SHIP for the
  wrong reason.
- `fulcrum why <cell> --ours BIN --rival-cmd 'CMD -{level} -c {input}' --corpus F` — the
  vendor diff; position counts name the mechanism in one command.
- `fulcrum candidates <cell> --repo .`; `fulcrum profile …` / `ab ablate` / `trace critpath`;
  `fulcrum anatomy` (Ir deltas, never wall claims); `fulcrum board size|wall`.
- `examples/blockcensus`, `examples/blockspans` (per-block BTYPE/hdr structure);
  `examples/chunkgrid.rs` (counted chunk arithmetic); `examples/ldxloop.rs` +
  `ldx_divergence.rs` (codegen parity head-to-head); `scripts/campaign/profile-ldx.sh`.
- T1-output-changing levers: `scripts/campaign/tie-guard.sh <ref>` first (~2 min).

# A. ISA-L / igzip (levels 0-3, hand-written kernel architecture)

The machine to copy conceptually: every hot kernel is hand-written asm on x86 AND
aarch64 (`vendor/isa-l/igzip/aarch64/`), the WHOLE body is dispatched per ISA tier
(`igzip_multibinary.asm:84-132`), and the parse is decoupled from emission by a
fixed-width 32-bit token (ICF). Cards A-V01…A-V19.

---

### A-V01 · igzip's 8K single-probe L0/L1 hash table (the banked-working-set lever)
- **Vendor**: `vendor/isa-l/include/igzip_lib.h:121-126` (`IGZIP_LVL0_HASH_SIZE`/`LVL1` =
  `8*IGZIP_K` entries × u16 = 16 KiB); probe+accept `vendor/isa-l/igzip/igzip_base.c:61-67`,
  `igzip_icf_base.c:76-82`. No chains at ANY igzip level; accept iff `dist-1 < dist_mask`.
- **Mechanism**: one 16-bit slot per 15-bit-diff hash, overwrite-on-collision, table
  sized to STAY L1d-resident (16-32 KiB) instead of our 64K head + 32K hash3 (~192 KiB).
- **Port cost**: legacy arm = shrink `src/compress/deflate/parse/fast.rs` head table
  (one const + the `hash3` table drop); ldx arm = `ldx/ht_matchfinder.rs` / `ldx/compress_fastest.rs`
  table order 15 → 13 (~128 KiB → 32 KiB working set). One-line geometry; the falsifier fits in a day.
- **Serves**: `WALL` (L1 route is the one legacy-codegen level left — Ir 150.84/b text
  vs 67.03 on the stack, handoff §3.2); secondarily the L1 exception's exit case (Phase 2).
- **Instrument**: `fulcrum try <ref> --levels 1,2 --threads 1,4 --corpus <declared TUNE
  superset>` (full leg at levels 1,2 — no scope needed; the `--scope` tool is for
  narrowing a wider board); `fast_l1_ratio_multi_corpus` must stay green
  (port+config must stay `<=` pigz -1 on text — that property is WHY L1 is an exception).
- **Falsifier (cheapest first)**: 1) `l1_bakeoff` one-block test (`parse/mod.rs:2560+`,
  `cargo test --release l1_bakeoff -- --nocapture`) — deterministic size delta per file that
  answers "does the smaller table still find the matches" in seconds; 2) paired Ir via
  `fulcrum anatomy` on text+binary at L1 (D1-miss/IPC hypothesis from
  vendor-structure-comparison §3 says misses drop); 3) the scoped try.
- **Risk**: igzip wins here BECAUSE its whole L1 working set is ~64 KiB including tokens;
  in ldx the ht is per-block state alongside 256 KiB of other state — the win may be
  smaller. The standing reason we widened is a size argument on text (`parse/fast.rs`
  comment) — the ratchet test must show ratio-neutral OR accept a level-1-only config.
- **Death modes**: (TIE-FLIP) L1 ratchet files go worse (armexe.elf -3863 bound);
  (WALL-NO-SHIP) wall at T1 only if the port's own inline caches absorb the win —
  measure, never argue; (EMPTY-CLASS) the ldx-L1 board class is GONE post-census — this is
  a WALL lever now, re-derive necessity in that framing (handoff §4.7).

---

### A-V02 · igzip's zero-cost window slide: mod-64K wrap + init-to-current (no rebase pass)
- **Vendor**: `vendor/isa-l/igzip/igzip.c:878-934` (`reset_match_history` fills every slot
  with `total_in & 0xffff`), `igzip_base.c:61-67` + `igzip_icf_base.c:76-82`: positions are
  absolute-mod-64K; a stale entry reads as ~64K-far and the `dist_mask` accept test
  (`and dist, (D-1); neg` + range check, asm `igzip_icf_body_h1_gr_bt.asm:271-276`) rejects
  it by CONSTRUCTION. No saturating rebase, no window memcpy, no SIMD slide — the slide
  is a memset sized to `2*(hash_mask+1)`.
- **Mechanism**: replace value-space position deltas with wrap arithmetic so table
  maintenance is O(1) per 64 KiB instead of one full-table pass per slide.
- **Port cost**: applies ONLY to a single-probe layout — for ldx today that means
  `ldx/compress_fastest.rs` + `ldx/ht_matchfinder.rs` (s16 saturating-rebase
  `ldx/matchfinder_common.rs:59-90` is libdeflate-inherited); the hc arm's chain walk
  compares `node <= cur_pos - 32768` and would need the same wrap semantic PLUS the
  `dist-1 < dist_mask` accept gate. Medium: geometry, not intrinsics.
- **Serves**: `WALL` (slide cost is once per 32 KiB of input — small but it removes the
  per-slide pipeline burst that shows at short streams); blocks micro-conflicts with
  the streaming `slide_window` in current ldx matchfinders.
- **Instrument**: `fulcrum try --levels 1 --threads 1,4` scoped; Ir per slide via
  `fulcrum anatomy` on a 1 MB fixture (count `matchfinder_rebase` calls first: they
  happens `input/32768` times — count, never infer).
- **Falsifier**: 1) check the actual slide frequency on the corpus with an anatomy
  counter (if slides are <0.1% of Ir, class empty — stop); 2) `cargo asm` the rebase
  (see J-V38/O-8) — if rustc already vectorizes it, the alpha is already banked; 3) port.
- **Risk**: wrap semantics poison EVERY stored position on aliasing — igzip gets away
  with it because their accept test ALSO clamps distances; ours must gain the same clamp
  or it emits invalid offsets.
- **Death modes**: (TIE-FLIP) the accept-clamp change alters matches chosen → bytes;
  (WALL-NO-SHIP) at 32 KiB granularity the pass is amortized below measurement;
  (EMPTY-CLASS) if ldx's streaming layout (`slide_window`) never fires the rebase on
  long T1 encodes.

---

### A-V03 · D6: block cadence as a memory parameter (level_buf grading)
- **Vendor**: `vendor/isa-l/include/igzip_lib.h:294-329` — per-level REQ + token budget
  graded MIN/SMALL/MEDIUM/LARGE/EXTRA_LARGE (1K/16K/32K/64K/128K tokens × 4 B); default
  LARGE = 64K tokens ≈ 256 KiB of input per Huffman refresh. Driver: `igzip.c:306-324`
  (`init_new_icf_block` — the ICF buffer capacity IS the block size; tokens written
  `igzip_icf_body.c:143-231`, block closes via `icf_body_next_state :238-250` when
  `icf_buf_avail_out <= 0`), header+emit via `create_icf_block_hdr`/`flush_icf_block`
  (`igzip.c:356-438`, `:200-228`).
- **Mechanism**: block MILIEU is a MEMORY parameter, not a heuristic — the caller's
  level_buf size sets how often dynamic tables re-fit; ratio follows the fit.
- **Port cost**: ldx has a fixed 300 K byte / 50 K seq budget (`ldx/mod.rs:148-164`)
  inherited from libdeflate; making cadence a parameter touches `ldx/mod.rs` consts,
  `ldx/compress.rs` block accounting, and T>1's `pipelined.rs` chunk grid contract.
- **Serves**: `CADENCE-L5` (access.log L5's +1.07% is the gzip-cadence band: gzip
  rebuilds tables every ~16-32K symbols; gzip's cadence is ~34K symbols cv=0.023 vs our
  8x-longer spans — `examples/blockspans`) and `CADENCE-L23` (dd79_bin6).
- **Instrument**: `examples/blockcensus` first — count actual block sizes/hdr cadence on
  access.log + dd79_bin6 at the failing levels; then
  `fulcrum try --levels 2,5 --threads 1,4 --scope "levels=2,5;threads=1,4;corpus=access.log,dd79_bin6…"`.
- **Falsifier**: 1) blockcensus on the two cadence files: are our spans 8x longer, and
  does the header share differ? (deterministic, seconds); 2) sweep SOFT_MAX_BLOCK_LENGTH
  ∈ {64K, 150K, 300K} in a builder-only A/B (`board-size.sh tune`) — NO ship, pure
  measurement of the size sensitivity; 3) only if a direction pays on cadence files
  WITHOUT flipping tie cells elsewhere: the scoped try.
- **Risk**: ⚠ this is adjacent-but-contrary to the FIVE falsified chunk-grid shapes
  (CLAUDE.md §2: monotone-T1 size wins, not grid tuning, buy headroom) and to Phase 1's
  own conclusion that seam costs cannot be shrunk. However those falsifications were
  about T>1 seam TAX; V03 is about T1 table-refresh cadence on files where we lose to a
  vendor with a much SHORTER cadence. State that distinction in the commit, and re-run
  the falsifier rather than eating the old verdict without re-derivation.
- **Death modes**: (EMPTY-CLASS) measured span delta smaller than the header-byte delta;
  (TIE-FLIP) cadence change moves every tied cell's bytes — the census kills it instantly;
  (WALL-NO-SHIP) more blocks = more table builds = wall regression on the same cells.

---

### A-V04 · E8: the ICF two-pass semi-dynamic token store (the format that unlocks everything else)
- **Vendor**: `vendor/isa-l/igzip/encode_df.h:9-32` — 32-bit token `lit_len:10 (257..512
  = len 3-258 via LEN_OFFSET 254) | lit_dist:9 (30 = literal; 31-287 = SECOND literal) |
  dist_extra:13`; pass 1 tokenize+histogram (`igzip_icf_body.c`, `isal_mod_hist`
  `include/igzip_lib.h:283-287`); block end builds exact per-block Huffman from the token
  histogram (`create_hufftables_icf`, `huff_codes.c:1563-1652`); pass 2 re-walks tokens
  emitting bits (`flush_icf_block` `igzip.c:200-228`).
- **Mechanism**: tokenize once into a FIXED-WIDTH record; emission then needs no
  parsing decisions, no length logic (`expand_hufftables_icf` pre-splices length extra
  bits into the tables, `huff_codes.c:1531-1561`) and becomes vectorizable.
- **Port cost**: ldx's seq store is `{litrunlen_and_length: u32, offset: u16, offset_slot:
  u16}` with literal bytes read from input at flush (`ldx/sequences.rs:132-137`,
  `ldx/flush.rs:117-660`; the literal bytes are NOT in the store). A port replaces
  `DeflateSequence` wholesale and rewrites all three parser writers
  (`ldx/compress_{fastest,greedy,lazy}.rs`) + the store's terminator convention
  (half-open entry with length==0). This is the single biggest lattice move in the
  catalog: sequences (store), flush (emit), split (observation tie-ins) — 3 named modules.
- **Serves**: mostly `WALL` (emit was 27.4% over libdeflate at the last sound attribution,
  `docs/board/l2-component-map.md`) and it is the PREREQUISITE for V05/V06. No direct size
  delta claimed (histogram-exact tables we already have).
- **Instrument**: `fulcrum ab ablate` Ir diff of emit stage pre/post (`fulcrum anatomy`
  on 1 MB text at L2/L6), never wall quotes from it; decisive ship instrument stays
  `fulcrum try --threads 1,4`.
- **Falsifier**: 1) `ldx_divergence.rs` must stay zero (byte-identity is the
  construction); 2) Ir delta of the emit loop isolated via `fulcrum ab ablate`;
  3) the try.
- **Risk**: the emit-side redesigns were FALSIFIED FIVE TIMES in place
  (`parse/mod.rs:968-1465` receipts — deleted from code 2026-08-01, surviving in git
  history; see `docs/vendor-technique-index.md` F1/F3/F4 notes). This card exists to
  bound WHY those died: they were measured without the fixed-width layout. That is a
  re-attempt license, not a tabula rasa — the falsifiers above MUST run before a port.
- **Death modes**: (TIE-FLIP) token repack moves a litrun/length boundary → any byte diff
  kills at tie-guard; (EMPTY-CLASS) emit Ir is already cheaper than libdeflate's under
  the checked parse — re-pull `l2-component-map.md` numbers before building; correct order:
  confirm the debt moved post-#366 codegen; (WALL-NO-SHIP) the store rewrite costs more
  than the gather saves (dependent-load chain at token walk).

---

### A-V05 · F4: vectorized second-pass emit — vpgatherdd over the ICF tables
- **Vendor**: `vendor/isa-l/igzip/encode_df_04.asm:178-243` (AVX2: 8 lanes/iter — gather
  lit+dist codes AND lengths, `vpsllvd`/`vpxor` variable-shift merge into qwords, ONE
  scalar bitbuf drain per group via the vpblendd/split ladder); AVX512 `encode_df_06.asm:94-101,189`
  (16 lanes). Bound check: one per group (`cmp out_buf, end_ptr`).
- **Mechanism**: token emission as a gather from two 4-byte-entry code tables — the ONLY
  way the bit-wall (serial 63-bit accumulator) becomes an SIMD problem.
- **Port cost**: requires V04 first (format), plus a Rust emit kernel with real
  intrinsics (`ldx/flush.rs` + new `ldx/emit_simd.rs`), plus the E9-style budget
  regen (A-V16) to make group writes bounds-legal in Rust (no uninitialized reads —
  we use `Vec`/slices with real checks).
- **Serves**: `WALL` (emit tree walk + bit transput) — gated on V04 landing first.
- **Instrument/Gates**: same as V04.
- **Falsifier**: after V04, FIRST count the emit-loop Ir share (`fulcrum anatomy`) — if
  emit is <5% of the wall at the failing wall coordinates, stop before writing any
  kernel. Note: ours executes FEWER emit Ir than libdeflate post-port (555M→measure),
  and the wall-vs-instructions rule (hard stop #4) binds.
- **Risk**: the F4 receipt (index): "the falsification was measured WITHOUT the
  fixed-width ICF token layout; re-attempt would have to change the seq store format
  first (E8)". That is a re-open, not a green light to repeat the old experiment.
- **Death modes**: (EMPTY-CLASS) emit share measured small post-codegen-slices;
  (WALL-NO-SHIP) gather latency (vpgatherdd ~4-10 cyc/element serialized) exceeds the
  scalar loop's per-token 3 stores; (TIE-FLIP) n/a if byte-identical — but V04's format
  change MUST be byte-proven first or the two changes confound (campaign rule: compose
  before concluding — land V04 alone, bank, THEN V05).

---

### A-V06 · P11/G4: SIMD match-map generation (gather/scatter on the hash table itself)
- **Vendor**: `vendor/isa-l/igzip/igzip_gen_icf_map_lh1_06.asm:206-406` (AVX512: 16
  positions/iter — `vpmaddwd` hash over shifted lanes, `vpgatherdd` of heads +
  `vpscatterdd` overwrite (later lane wins; intra-vector aliasing tolerated because
  matches re-verified against input), `vplzcntq` lengths, lane-rotate LAZY compare);
  `igzip_set_long_icf_fg_06.asm:202-256` (re-stamp following positions with decreasing
  lengths of the same distance); base C `igzip_icf_body.c:80-136` (`gen_icf_map_h1_base`)
  + `:34-74` (`set_long_icf_fg_base`).
- **Mechanism**: three decoupled sweeps per 4 KiB chunk (`MATCH_BUF_SIZE 4096`,
  `igzip_level_buf_structs.h:8`): (1) vector map with ONE hash probe + ONE u64 xor
  each, lengths capped 4..8; (2) for every ≥8 map length, full-compare forward
  extension re-stamps; (3) greedy walk of the map w/ free 1-step lazy.
- **Port cost**: needs V04 token format + a new `ldx/matchmap.rs` (new module), and — the
  hard part — aarch64 NEON paths (we ship both arches; a NEON map kernel is a
  second kernel).
- **Serves**: `WALL` (the only vendor design that vectorizes MATCH FINDING itself, at
  L3-shaped configs). NOT the cadence size band: sizes stay libdeflate-tied at any
  ported level — this can only win wall.
- **Instrument**: `fulcrum try <ref> --levels 1,3 --threads 1,4 --corpus <declared
  superset>` scoped `--scope "levels=3;threads=1,4"`; `fulcrum profile` first to
  confirm the dependent-load chain is still the wall-blocker after V04 lands.
- **Falsifier**: the pipeline: 1) `examples/ldxloop.rs` kernel-only Ir/wall A/B of
  "map-then-choose" vs serial on synthetic 4 KiB blocks (kills the idea in 1 day if the
  map writes cost); 2) `fulcrum trace critpath` under T4 to prove no regression in
  starve/schedule; 3) scoped try.
- **Risk**: igzip-style aliasing tolerating scatter (later-lane wins) CHANGES which
  matches the map proposes vs our serial probe — output differs → only legal at a
  level routed OFF the tie (none today except L1). SIMD matchfinding lands easiest
  where tie-invariance is already waived: the legacy L1 arm, i.e. Phase 2 territory,
  not the cadence band.
- **Death modes**: (TIE-FLIP) map-vs-serial decision divergence on ANY tied level kills
  the census; (WALL-NO-SHIP) scatter/gather on 16-bit entries is 2 stores per byte;
  (CORRUPTION) alias tolerance becomes an invalid-gzip source if the re-verify drops
  one guard byte.

---

### A-V07 · dual-literal token packing + free lazy in the map walk
- **Vendor**: `vendor/isa-l/igzip/igzip_icf_body.c:143-231` (`compress_icf_map_g`: literal
  PAIRS packed into one token via `lit_len` + `lit_dist>=LIT_START`; match at n or n+1
  accepted with a 1-step lookahead peek at NO extra search cost, block cadence
  `:223-228` advances `block_end`); asm equivalent packs 8 literals/iter with PSHUFB
  (`igzip_icf_body_h1_gr_bt.asm:592-640`).
- **Mechanism**: two literals share one token so emission is 4 B/2 literals, and the
  greedy walk gets a free lazy decision at pos+1 without a second search.
- **Port cost**: subsumed by V04 (token format carries the pairing); the *decision*
  change (accept at n+1 when equal-length → prefer pos+1 start) is a parse change in
  `ldx/compress_lazy.rs` — event-level, measurable alone.
- **Serves**: `CADENCE-L5` (per-position decision cadence is exactly what gzip's parse
  does differently on token-structured logs), `LADDER-L4`.
- **Instrument**: `fulcrum why access.log…` — the position-count diff between our L5
  parse and gzip -5's names which side of this lives there; then
  `fulcrum try --levels 4,5 --threads 1,4`.
- **Falsifier**: 1) `fulcrum why` on the two cadence cells (position counts); 2)
  builder-level knob sweep (A/B parse at n+start bias — `board-size.sh tune`, size
  only); 3) tie-guard; 4) scoped try.
- **Risk**: our lazy already has a 1-step peek with a P4 heuristic (libdeflate's) — this
  mechanism overlaps; the measured virtue at igzip is the zero-cost PEEK inside an
  already-computed map, which we can't have without V04.
- **Death modes**: (TIE-FLIP) parse change → census; (WALL-NO-SHIP); (EMPTY-CLASS) if
  `fulcrum why` shows position counts already match gzip's on access.log — then the
  deficit is Huffman/block-cadence, pointing back at V03.

---

### A-V08 · G5: the histogram pass is a 2-position pipelined LZ77 re-run (not vectorized counting)
- **Vendor**: `vendor/isa-l/igzip/igzip_update_histogram.asm:256-546` (with `04 = AVX2`
  wrapper `igzip_update_histogram_04.asm`): a full single-probe match-finder over the
  block (hash/hash2 double compute, `compare250` AVX2 loop `:548-566`, tzcnt lengths)
  whose only OUTPUTS are u64 histogram buckets (`inc qword`, scalar) — SIMD lives in the
  pipeline (xdata shift via PSRLDQ `:320-353`) and the compares, NOT in the counting.
  Exposed publicly (`isal_update_histogram`, `huff_codes.c:634-688`) so a caller can
  histogram a whole stream, then `isal_create_hufftables` one code for the WHOLE stream
  (the semi-dynamic segment pattern).
- **Mechanism**: one cheap pass finds every match once and histograms it exactly; a
  second, cheaper pass can then encode against globally-gathered tables.
- **Port cost**: ours histograms inline per-symbol (`ldx/sequences.rs:69-77` — two
  anatomy_count! lines + `get_unchecked_mut` add). A two-pass option touches
  `sequences.rs` + a new histogram-prepass in `compress.rs`; correctness contract
  unchanged (same histogram, better cache/branch profile).
- **Serves**: the compression-frequency work is NOT on our wall/libdeflate diff today
  (`l2-component-map.md` is the atlas) — this card exists to KILL the idea of porting it
  until V04 says otherwise. Keep as a "do-not-port-yet" instrument card: if the
  semi-dynamic variant pattern is ever borrowed (two-pass encoder for streaming T>1
  chunk N with table from chunk N-1's histogram — Phase 1 adjacent), this is the vendor
  precedent to cite.
- **Instrument**: `fulcrum anatomy` histogram_update counter (already exists —
  `anatomy_count!(histogram_updates)` at `ldx/sequences.rs:70,110`) on a 1 MB corpus;
  if histogram Ir > 3% of any named cell's excess, re-open.
- **Falsifier**: the counters. One run.
- **Risk**: none as an instrument. The error mode it prevents: a session building
  "SIMD histogram" that never asked which cell it serves (the docs-commit-wearing-a-
  tool's-costume anti-receipt, CLAUDE.md "What IS forbidden").
- **Death modes**: (EMPTY-CLASS) — the pass costs 2-4% of Ir and the cards' walls are
  elsewhere; (TIE-FLIP) n/a if histogram-only; (WALL-NO-SHIP) a re-run pass DOUBLES the
  pass-1 wall by construction — it only pays against a saved second parse/emit, i.e.
  only inside V04.

---

### A-V09 · H2: crc32-instruction hash on the probe chain
- **Vendor**: `vendor/isa-l/igzip/huffman.h:207-226` (`compute_hash` = `_mm_crc32_u32`
  under SSE4_2, else 0xB2D06057 ×2; asm `huffman.asm:243-249`); dictionary preload
  4-at-a-time `igzip_deflate_hash.asm:105-132`.
- **Mechanism**: 1-instr/3-cyc hash on the dependent chain that feeds the head-table
  load; frees a multiplier port that `lz_hash` (mul+shift) occupies.
- **Port cost**: `ldx/matchfinder_common.rs:103-107` (`lz_hash`) + both matchfinder
  call sites — one `#[target_feature]` shim, falling back to the multiplier.
- **Serves**: `WALL`. Decided per-arch (aarch64 `crc32cw` has the same shape).
- **Instrument**: `examples/ldxloop.rs` head-probe micro A/B (Ir first, wall second,
  hand-rolled A/B allowed as FALSIFIER only — promotion stays `fulcrum try --levels 1,2 --threads 1,4`).
- **Falsifier**: Ir counter first; then x86+aarch64 paired wall (any single-arch claim
  has died here before — the hyperfine-on-M1 "3.4x wall win" reverted when x86 measured
  1.19x slower; CLAUDE.md structured finding #10).
- **Risk**: zlib-ng REMOVED their crc32 hash (2.3.x, agent-grepped absent) — weak
  counter-precedent that must be answered in the commit message before the lever opens.
- **Death modes**: (WALL-NO-SHIP) the multiply was never on the critical path in the
  port's codegen (LLVM already scheduled); (TIE-FLIP) n/a — same hash VALUES required
  or the diff invalid: adopt only if a CRC32-x-mix equals `lz_hash`'s bucket behavior,
  which it does NOT — so this is a config-table change (different hash ⇒ different
  matches ⇒ bytes) — reclassify: it MAY only land inside a waiver level. That kills it
  for the cadence band outright.

---

### A-V10 · H4: vpmaddwd-friendly hash for vector map generation
- **Vendor**: `vendor/isa-l/igzip/huffman.h:228-245` (`compute_hash_mad`: PROD1*low16 +
  PROD2*high16 twice — `vpmaddwd` lanes); asm pair `igzip_gen_icf_map_lh1_06.asm:206-207`.
- **Mechanism**: same trick as V09 but lane-wise: the price of V06's vectorization.
- **Port cost**: only inside V06 (choose constants sized for 15-bit masks).
- **Serves**: `WALL` (L3-ish arms).
- **Instrument**: with V06.
- **Falsifier**: with V06.
- **Risk**: constant choice changes the hash function ⇒ matches change ⇒ bytes change
  (same block as V09). It is a V06-internal detail, not an independent lever.
- **Death modes**: subsumed by V06's.

---

### A-V11 · P13: literal-run skip ramp (hash-free emission on incompressible data)
- **Vendor**: `vendor/isa-l/igzip/igzip_icf_body_h1_gr_bt.asm:54-59` (defines:
  `SKIP_SIZE_BASE 2048` bytes without a match to open skipping, `SKIP_BASE 32`,
  `SKIP_START 512`, `SKIP_RATE 2`, `MAX_SKIP_SIZE 128`), ramp machine `:534-640` —
  8 literals/iter via PSHUFB packing into 4 dual-literal tokens (`:592-640`), matches
  decrement the level (`:366, :458, :460`); the C base has no skipping.
- **Mechanism**: after a matchless run of 2 KiB, SKIP hash probes entirely and emit
  literals 8/iteration, ramping the skip width by round; every match deflates the level.
- **Port cost**: a bounded skip state in `ldx/compress_greedy.rs` (the only arm bound
  where igzip has it: L1/L2); constant-only, ~80 lines; legacy arm's L0 ACCEL
  (`parse/fast.rs:1242-1262, 2376-2402`) is the same idea in weaker form.
- **Serves**: `HAIRLINE-NOCOMP` (photo.jpg L1-L3 +0.04% — the binding rival is gzip,
  whose ACCEL is stronger; movie.mp4) and the random-1M WALL cells; NOT the
  cadence band (they are match-rich).
- **Instrument**: the legible scope is gzip-bound hairlines:
  `fulcrum try --levels 2,3,4 --threads 1,4 --corpus photo.jpg,movie.mp4,random-1M…`
  (hairlines are small margins — raise `--scope-sentinels` enough that a hairline
  regression is visible in the sentinel grading).
- **Falsifier**: 1) implement the ramp in the GREEDY printer with an env-var OFF switch
  is BANNED (non-negotiable 3) — parameter must be compile-time constant; measure
  instead as: builder flag → census; 2) skip must never shorten a block — it's
  monotone-in-emission-cadence, not in bytes; bytes only move where a skipped position
  WOULD have matched (that is the measured risk); 3) tie-guard THEN the try.
- **Risk**: it changes WHICH positions get hashed — deeper cells (L7-L9) lose matches
  the ramp skipped and can go WORSE on ratio; scope to the levels igzip scopes it
  (L1/L2 body) unless measured otherwise.
- **Death modes**: (TIE-FLIP) any cadence-band cell flips (random-1M-type cells are
  tied-tight); (WALL-NO-SHIP) our parsers already run at higher IPC than the legacy
  fast path, so the skip's win may not show; (EMPTY-CLASS) if the hairline gap is
  Huffman-structural, not probe-count (check `examples/blockcensus` first).

---

### A-V12 · M15: large-match emit loop (258-repetition fast path)
- **Vendor**: `vendor/isa-l/igzip/igzip_body.asm:41-44` + `:561-735` — compare up to
  `MAX_EMIT_SIZE`: 258×16 = 4128 bytes; a match ≥ `LARGE_MATCH_MIN 264` enters an
  emit-only loop writing 258-length codes with NO re-searches, then re-hashes only
  `4*LARGE_MATCH_HASH_REP` trailing positions; same in the ICF body
  (`h1_gr_bt.asm:719-789`: `or len_code2, dist_code; cmp m_out_buf…` emit loop); C
  counterpart zero. Related vendor: zlib-ng deflate_rle (strategy-only; B-V33).
- **Mechanism**: long runs pay ONE compare and then pure emission; ours re-searches
  every 258 bytes of a run (a 300 KiB zero block costs ~1160 searches per the atlas).
- **Port cost**: an emit-loop branch in `ldx/compress_greedy.rs` (+ `ldx/lazy` if
  eligible there): find-first-≥264 then loop `len-258` WITHOUT re-search — decode-legal,
  parse-equivalent for runs by construction. Careful: `adjust_max_and_nice_len` and
  window-edge arithmetic (`ldx/sequences.rs:145-150`) must agree with the loop's crazy
  long length arithmetic.
- **Serves**: `movie.mp4 L6 vs libdeflate +0.0008%` (HAIRLINE-NOCOMP; runs), and wall on
  all run-heavy rows; potentially the ONLY size lever on the near-tie cells: monotone?
  NO — the emitted length sequence over a 300-KiB run is identical (258,258,…
  remainder) either way; the difference is WHICH DISTANCE the extra matches take —
  monotone-safe only if the same distance repeats, which it does for real runs.
- **Instrument**: `fulcrum try <ref> --levels 6,9 --threads 1,4` scoped to
  `movie.mp4,weights.safetensors,…`; plus a builder-const probe that EMITS ONLY
  (guaranteed-equivalent bytes) to measure the wall alpha via `fulcrum anatomy`
  matches_emitted/loop counter diff.
- **Falsifier**: 1) count: `fulcrum anatomy` on movie.mp4 L6 — how many searches does
  the run cost today (name the counter, do not infer); 2) builder-flag byte-equality
  probe on the corpus (must be byte-IDENTICAL for run-only matches — if not, the loop's
  search policy differs → TIE-FLIP risk made explicit); 3) scoped try.
- **Risk**: libdeflate does NOT have this and we are tied to libdeflate bytes — the
  probe in (2) is the load-bearing step. The known false-negative risk: our parse's
  post-match skip inserts (`skip_bytes`) every interior position, whose distance can
  beat the run-distance on cyclic data — the loop must not change those decisions.
- **Death modes**: (TIE-FLIP) interior-insert distance policy diverges → bytes shift;
  (WALL-NO-SHIP) runs are rarer than the 4128-byte threshold implies on the corpus;
  (EMPTY-CLASS) `movie.mp4 L6` is +0.0008% — a byte-level lever of that size is likely
  below measurement resolution; treat the WALL leg as the real prize.

---

### A-V13 · P12: two-position pipeline + SPECULATIVE literal-code loads
- **Vendor**: `vendor/isa-l/igzip/igzip_body.asm:246-330` — body loop hashes pos and
  pos+1 (`compute_hash hash/hash2` over a shifted `xdata`), probes BOTH heads
  (`:262-276`), 8-byte xor-compares both candidates, and — the half we lack — loads
  BOTH literal Huffman codes speculatively so the no-match path writes TWO literals in
  one `write_bits` (`get_lit_code` loads at `:303,322; SHLX/or ADD_BITs pair :316-321`).
- **Mechanism**: hides the dependent-load chain of one position behind the other's, and
  halves literal-write instructions.
- **Port cost**: legacy L1 `fastloop_l1` already issues both head reads (`parse/fast.rs:2407-2492`,
  the "SF2" shape) but emits literals singly; ldx L1 route (`compress_fastest.rs`/ht)
  is strictly per-position. Port = fast.rs emit cadence (known risk: the five
  falsified emit-cadence redesigns) AND the ldx ht loop (untouched shape today).
- **Serves**: `WALL` (L1 legacy arm; `photo.jpg L1-L3` wall leg).
- **Instrument**: `fulcrum anatomy` literal-emit counter at L1 before work; scoped try
  `--levels 1 --threads 1,4`.
- **Falsifier**: 1) count literal-emit instruction share of the L1 Ir budget (if <10%,
  class empty); 2) the SF2 tune (`fast.rs` existing knob set) is UNROUTED — route test
  is cheap (`l1_bakeoff` gives the size delta in seconds, sub-second per block).
- **Risk**: a tuned/instrumented build is 1.17x slower than a vanilla one — never quote
  one against a rival (CLAUDE.md wall rule); and the 42-cell payer #227 teaches the
  zone this card sits in: emission cadence redesigns were the systematic
  violation-shaped work that shipped nothing from 31 branches.
- **Death modes**: (TIE-FLIP) n/a if byte-identical emit; realistic: (WALL-NO-SHIP) on
  the port codegen (LLVM already interleaves — check `ldxloop.rs` divergence first);
  (EMPTY-CLASS) L1 wall cells post-codegen-slices may already be within clause 5.

---

### A-V14 · E7: trained default Huffman tables, one-pass emit (igzip L0)
- **Vendor**: `vendor/isa-l/igzip/hufftables_c.c` (generated, trained dynamic table,
  109-byte prebuilt header `deflate_hdr`); emit path `igzip.c:569-572` → `isal_deflate_pass`
  emits FINAL bits in ONE pass against the canned tables; header replayed by memcpy
  with BFINAL toggled (`:1829-1835`); training corpus NOT recorded — regenerable via
  `generate_custom_hufftables.c:29-46`.
- **Mechanism**: deletes pass 2 AND table build; pays a fixed ratio (no adaptation).
- **Port cost**: a `const` table (`src/compress/ldx/tables.rs` sibling) + a one-pass
  emit variant in the L0/L1 arm; ours builds dynamic tables per block today
  (`parse/fast.rs:3276-3292` legacy; ldx emits via `make_huffman_codes` per block).
- **Serves**: `WALL` (L0/L1), `HAIRLINE-NOCOMP` (photo class at L1: header-per-block
  cost vanishes; sizes may go +0.04% worse-or-better — must measure).
- **Instrument**: `fulcrum try --levels 1 --threads 1,4` scoped;
  `fast_l1_ratio_multi_corpus` gate (pigz -1 text bound).
- **Falsifier**: train OURS on the TUNE corpus (`generate_custom_hufftables.c` shape;
  never igzip's — their corpus is unrecorded), builder A/B on `board-size.sh tune` L1
  rows: if the give-back on the cadence-band files exceeds the header savings,
  class empty; then the try.
- **Risk**: non-negotiable #2 (per-label) — L1 has a pigz-1 WINS-on-text property
  (`#347`) to preserve; a static table is a ratio bet on text.
- **Death modes**: (EMPTY-CLASS) header amortization at 64 KiB blocks is already ~0.17%
  recorded street — but the WIN here is wall (no per-block histogram/tree/header);
  the size give-back must stay under the band's margin; (TIE-FLIP) every L0/L1-tied
  cell by construction; (WALL-NO-SHIP) histogram+tree per 64 KiB is only ~1-2% of the
  L1 wall (l2-component atlas) — the alpha ceiling is small.

---

### A-V15 · L2: repeated-char prefix canned-header block (stateless)
- **Vendor**: `vendor/isa-l/igzip/repeated_char_result.h:51` (`MIN_REPEAT_LEN 4096`),
  `igzip.c:614-742` — first 8 bytes all-0x00/0xFF ⇒ scan the run; run ≥4096 emits from
  a pre-baked dynamic header as 258-codes with code-arithmetic tail fill; stateless
  path only (`isal_deflate_int_stateless:736-742`).
- **Mechanism**: zero-page/pure-run prefix has a KNOWN optimal block shape; bake it.
- **Port cost**: trivial condition in `ldx/compress.rs`'s level-0/entry checks +
  a `const` header; condition is input-triggered (not a content detector — passes
  non-negotiable #3 as parameterization-free, like igzip's own comment says).
- **Serves**: `HAIRLINE-NOCOMP` (zeros-heavy rows of movie.mp4/weights patterns),
  plus tiny-file WALL.
- **Instrument**: builder probe: count prefix-run frequency on the GATE corpus
  (`examples/chunkgrid.rs`-style one-liner); then scoped try.
- **Falsifier**: how often does the trigger fire on OUR corpus? If never, empty class —
  one grep-level script over the corpus, no cargo needed.
- **Risk**: our L0 is stored-only in ldx (`ldx/compress.rs:246-258` — the pure-stored
  passthrough test) — a zeros-heavy input is ALREADY cheap as stored; the dynamic
  header beats stored margin (259 B per 1 KiB) but loses to plain 258-code hash runs
  (tiny output). Measured claim needed on the near-tie cells only.
- **Death modes**: (EMPTY-CLASS) trigger never fires on the cadence corpus; (TIE-FLIP)
  changes L0 bytes (L0 is stored-or-static today — moving to dynamic-tokens changes
  every zero-run cell); (WALL-NO-SHIP) negligible.

---

### A-V16 · do-not-port: E9 `are_hufftables_useable` + `expand_hufftables_icf`
- **Vendor**: `vendor/isa-l/igzip/huff_codes.c:1316-1355` + `:1396-1408` (regen trees at
  MAX_SAFE_LIT 13 / MAX_SAFE_DIST 12 when lit+len+dist chain would exceed
  `MAX_BITBUF_BIT_WRITE 56`), `:1531-1561` (`expand_hufftables_icf` pre-splices length
  extra bits into codes 265-512 so pass 2 is pure lookups + `dist_codes[DIST_LEN]`
  zeroed trailer).
- **Mechanism**: dynamic reproducibility of a cause our STATIC caps have by
  construction (lits 14 + extra 5 + dist 15 + dist-extra 13 = 47 ≤ 63, proven
  `ldx/codes.rs`/`parse/mod.rs:943-952` const `can_buffer`).
- **Verdict**: card exists so nobody "ports" it as a metric-capture exercise. The
  only surviving idea is V04's: pre-splicing extra bits into emitted code tables is
  WHY igzip's pass 2 needs no length logic — if V04 lands, adopt the pre-splice in
  `ldx/flush.rs` tables (monotone Ir win, zero size effect).
- **Instrument**: n/a until V04.
- **Death modes**: (EMPTY-CLASS) by construction; (TIE-FLIP) n/a; (WALL-NO-SHIP) —.

---

### A-V17 · do-not-port: E1 heap-based tree construction (with ≥2-symbol padding)
- **Vendor**: `vendor/isa-l/igzip/proc_heap_base.c:34-86` + `heap_macros.asm` (branchless
  cmov heapify) + padding-to-2 rule `huff_codes.c:749-767` — zlib-lineage heap build
  with packed `freq<<16|sym` u64 entries, merged nodes written back into the SAME heap
  array downward.
- **Why we do not steal the structure**: ours is counting-sort + two-queue merge
  (`ldx/huffman.rs` port of libdeflate E2) — different constant factor, same output.
- **What IS worth one measurement**: the `heap_size < 2` padding rule (igzip pads the
  heap to 2 symbols so the tree always has ≥1 internal node) vs our
  `<2-symbols ⇒ two length-1 codewords` special case (`ldx/huffman.rs` ~:414-429). If
  `fulcrum why` ever shows a degenerate-block size gap, this is the reference shape;
  today the two-sym rule and igzip's emit identical byte-level behavior — no lever.
- **Instrument**: `fulcrum why` receipts only.
- **Death modes**: (EMPTY-CLASS) by construction today.

---

### A-V18 · G1 dispatch: multibinary per-kernel tiers — delivery mechanism, not a technique
- **Vendor**: `vendor/isa-l/igzip/igzip_multibinary.asm:84-132` — `mbin_dispatch_init5/6`
  self-patching pointers per kernel (body, icf_body per level, encode_df, gen_map,
  update_histogram, hashes, adler); the suffix tiers are `_base/01/02/04/06`.
  zlib-ng's counterpart: `functable.c:18-33, 100-436, 440-499` (eager atomic
  constructor + stub dispatch). Note the ASYMMETRY: libdeflate dispatches NOTHING on
  the compress path (zero runtime dispatch; helpers only) and is the fastest
  scalar encoder — the vendor evidence contradicts "dispatch everything".
- **Port cost / verdict**: our wins would be the individual kernels above (V01-V17),
  not the dispatch machinery. A `codspeed`-style `fulcrum compare` will show whether
  any 01→04 tier jump pays before any Rust dispatch plumbing: build igzip twice
  (`--disable-asm`-equivalent env or by patching `mbin_dispatch_init` to `_base`) and
  A/B on the frozen wall-to-size coordinate first — THAT is the falsifier.
- **Death modes**: (EMPTY-CLASS) the tier delta measured on this box is the ONLY ground
  truth; a name-only port gets reverted (3-for-3 rule).

---

### A-V19 · H5: hash-mask shrink for tiny inputs
- **Vendor**: `vendor/isa-l/igzip/igzip.c:1402-1403` (stateless) + `:1545-1547`
  (streaming): `if (hash_mask > 2*avail_in) hash_mask = (1 << bsr(avail_in)) - 1` —
  a 100-byte input touches a 128-entry table, not 8K; makes the table reset accurate
  to input size (reset cost scales with table size, `parse/fast.rs:2776-2795` legacy).
- **Mechanism**: working-set accuracy at the small end.
- **Port cost**: parameter clamp in `ldx/compress_fastest.rs::new`/`ht::init` +
  `ldx/matchfinder_common.rs` init; sizes only — smaller init loop, no steady-state cost.
- **Serves**: tiny-input WALL cells (one-1B, small-256B style rows — NOT in the cadence
  band; board wall shows short files dominated by fixed costs, so gains may register
  only on `fulcrum dropin` CLI-latency axes).
- **Instrument**: `fulcrum try --levels 0,1 --threads 1` subdivided on the small
  fixtures if they are declared; else `fulcrum dropin` (the CLI third axis).
- **Falsifier**: count the reset Ir on a 256B input (`fulcrum anatomy` — one number),
  then decide.
- **Death modes**: (EMPTY-CLASS) tiny-input wall is dominated by process+I/O, not the
  hash clear; (TIE-FLIP) real: a small mask changes which candidates probe → bytes
  change on small files — same tie-cage as V09/V10; (WALL-NO-SHIP) the reset loop is
  already vectorized-class memset work in both engines.

---

# B. zlib-ng (the strategy/dialect layer)

zlib-ng's contribution is not hardware asm (compare256 aside) but a richer LEVEL
STRUCTURE: four strategies (quick/fast/medium/slow), per-level 4-tuples, and a
cost-evaluating medium arm. Its level table sits at `vendor/zlib-ng/deflate.c:105-131`.
Cards B-V20…B-V36, plus joint cards J-V37…J-V40.

---

### B-V20 · P10: deflate_medium — carry the n+1 search across the emit step
- **Vendor**: `vendor/zlib-ng/deflate_medium.c:178-264` (strategy), levels 3-6
  (`deflate.c:123-126` — L3 {4,6,nice16,chain6}, L4 {4,12,32,24}, L5/L6 {8,16,32,32}/{8,16,128,128}).
  Mechanism: search pos n, then ALSO search pos+1 BEFORE emitting at n
  (`:227-244`: `next_match = find_best_match(...)` after advancing to
  `current_match.strstart + current_match.match_length`), then emit at n;
  the n+1 result is consumed next iteration when it starts where the cursor moves to
  (`:211-213`: `current_match = next_match` when `next_match.match_length > 0`).
  `early_exit = level < 5` (`:184-185`) disables the n+1 probe at L3-4 — so zlib-ng's
  L3 medium is single-probe (L3 config {4,6,16,6} in deflate.c:123!).
- **Mechanism**: one search per emitted symbol instead of two (lazy's shape) at a
  fraction of lazy's cost — the "why is zlib-ng L3-6 monotone AND cheap" answer.
- **Port cost (THE Phase-3 lever)**: a new rung in `ldx/compress_greedy.rs` (or extend
  `ldx/compress_lazy.rs`'s generic with a `medium` bool like `lazy2`); needs matching
  insert policy (`insert_match`'s interior rules `:47-91`, esp. the 16×
  `max_insert_length` bound `:70`). ~200 lines, one module.
- **Serves**: `LADDER-L4` (data.sqlite L4 +0.13/+0.17% and the KNOWN_SAGS
  (tabular,3)/(binary,3) L4<L3 inversion — plan §5 names zlib-ng's medium family as
  THE vendor-precedented candidate), `CADENCE-L5`.
- **Instrument**: `fulcrum try <ref> --levels 3,4,5 --threads 1,4 --scope "levels=4;threads=1,4;corpus=data.sqlite,tabular…,binary…"`
  as the serialization point; `ladder_is_monotone_t1` + `won_cells_stay_won` are the in-repo gates.
- **Falsifier**: 1) size leg on the KNOWN_SAG files only (deterministic); 2)
  `tie-guard.sh` — medium CHANGES bytes on levels that are tied today (L4-L6);
  expect partial-credit math BEFORE the try: if `board-size.sh tune` cannot show the
  `access.log L5` and `data.sqlite L4` rows shrinking, there is no cell to close; 3)
  the scoped try — and the wall leg is where medium must win, since ratio is tied.
- **Risk**: #363 is the same shape (good_match ported to L6/L7 for byte-identity);
  a medium port must COMPOSE with it or replace it — read #363's pending checklist
  before starting. Also the 2026-07-31 falsified "good_match rescues depth" record
  (structure-comparison): any medium/good lever must run its OWN size+wall gate, not
  ride along.
- **Death modes**: (TIE-FLIP) every L4-L6 byte-tied cell; (WALL-NO-SHIP) at the port's
  1.03-1.11x in-process budget, a second search per match swaps size for wall —
  check clause 5 at both T1 and T4; (EMPTY-CLASS) the L4 band may re-close via the
  pick-min-deletion census before this lever opens (plan §5's re-measurement first).

---

### B-V21 · P10b: fizzle_matches — leftward shift of the NEXT match (the anti-P4 mechanic)
- **Vendor**: `vendor/zlib-ng/deflate_medium.c:123-176` — when the byte BEFORE
  next-match's start also matches (`*match != *orig` quick exit `:138`), walk the next
  match left one byte at a time (`:147-167`) while shortening the current one; adopt
  only if `c.match_length <= 1` ended and `n.match_length != 2` (`:169-175`);
  `orgstart` guard prevents double-inserts.
- **Mechanism**: recoups P4's literal deficit (a shorter earlier match) by converting
  length into an extra literal exactly when the NEXT match starts mid-literal-run —
  the arithmetic libdeflate's `4Δlen + Δlog2(dist) > 2` heuristic approximates, but
  fizzle makes it a byte-level repair with no thresholds.
- **Port cost**: rides with V20 (it operates on the medium rung's `next_match`).
- **Serves**: `CADENCE-L5`, `LADDER-L4`.
- **Instrument**: same grid as V20; isolate ON/OFF with a two-leg build (fizzle-off =
  medium plain, ON = fizzle) — the delta IS the fizzle's economic value.
- **Falsifier**: per-file size census first (fizzle informs the same band as V20);
  the usual gates.
- **Risk**: byte-level repair interacts with `check_match` assumptions about
  `match_start`/`strstart` invariants (`zlib-ng/deflate_medium.c:123-127` precondition
  block) — 4 guard branches sit in a hot middle loop; port them or drop the UNLIKELY
  arms consciously, never silently.
- **Death modes**: (TIE-FLIP) same as V20; (WALL-NO-SHIP) two shifts per emit on
  match-dense text (the while loop is serial); (EMPTY-CLASS) if the quick-exit branch
  (`*match != *orig`) fires ~always, the mechanic is dead weight.

---

### B-V22 · medium's own L3/L4 table rows are CHEAPER than our libdeflate copies
- **Vendor**: `vendor/zlib-ng/deflate.c:105-131` — zlib-ng L3 with medium ON =
  {4,6,nice 16,chain 6}; L4 = {4,12,nice 32,chain 24} — versus our (libdeflate's)
  L3 Greedy/12/14 and L4 Greedy/16/30 (para copy `ldx/compress.rs:115-134`). The
  vendor-structure-comparison §1 gzip column was hiding zlib-ng's tighter numbers behind
  the NO_MEDIUM_STRATEGY ifdef — re-read the CONFIG header itself when quoting chains.
- **Mechanism**: a per-level parameter difference we inherited without testing (the
  chain-depth record: matching zlib-ng chain at L5-9 closed 84 cells pre-pivot; the
  mid-band is where their parameter choices DIVERGE from ours most (chain 6 vs 12)).
- **Port cost**: `ldx/compress.rs:115-127` mapping + `LevelParams` — a TABLE row
  change, no code.
- **Serves**: `LEN3-L3` (access.log L3 vs libdeflate +0.42% — today that lever is
  #364's far-len3 on the legacy arm; an ldx-side L3 config change is the
  re-derivation), `CADENCE-L23` (dd79_bin6 L2/L3).
- **Instrument**: `fulcrum try --levels 2,3 --threads 1,4 --scope "levels=3;threads=1,4;corpus=access.log…"`.
- **Falsifier**: builder-level config sweep on the two files (size only, seconds) →
  tie-guard → try. NOTE the falsified record: L2 retune as WALL lever died (Lazy/4/10
  +23.4% Ir at L2, structure-comparison FALSIFIED 2026-07-28) — the SAME class reopens
  legitimately at L3 as a SIZE lever with the cadence band on the board.
- **Risk**: L2 is byte-tied to libdeflate -2 EVERYWHERE (0 slack: a config move that
  gains bytes at L3 can also LOSE them — the sweep is per-level, never general).
- **Death modes**: (TIE-FLIP) all tied cells; (EMPTY-CLASS) L3 cadence gap is Huffman
  not parse (fulcrum why tells); (WALL-NO-SHIP) depth-lightening pays wall only on
  match-poor streams; scope the try.

---

### B-V23 · P14: EARLY_EXIT — stop the chain walk on the first non-improving candidate
- **Vendor**: `vendor/zlib-ng/match_tpl.h:13` (`EARLY_EXIT_TRIGGER_LEVEL 5`) +
  `:117,233-238` (`early_exit = s->level < 5`; on a failed extension `break`).
- **Mechanism**: for cheap levels bet the nearest chain entry is the best; trades
  ratio for a shorter dependent-load chain IN THE WALK itself.
- **Port cost**: one branch in `ldx/hc_matchfinder.rs` long-match walk (named module —
  the falsified matchfinder space: three prior levers aimed at this component and
  died; check `hc.rs` FALSIFY records (chain-node prefetch net-loss, prefilter-operand
  hoist) BEFORE touching this file's loop — they are code-adjacent).
- **Serves**: `WALL` (L2-L4 arm — the walk today runs to depth/nice as libdeflate does).
- **Instrument**: `fulcrum try --levels 2,3,4 --threads 1,4` scoped; `ldxloop.rs`
  prologue/loop instruction census first.
- **Falsifier**: (1) `examples/ldxloop.rs` chain-walk cycle attribution at L2/L3
  (cheap, local); (2) size leg MUST be measured — this CHANGES bytes (a break at
  failed-extension-vs-depth-exhaust changes picked matches); tie-guard;
  (3) the try.
- **Risk**: depth IS the cost at L5-L9 (84-cell record) — raising depth closed cells;
  early-exit cuts depth ADAPTIVELY: opposite-direction bet. NEVER generalize to L7+.
- **Death modes**: (TIE-FLIP) byte-changes on shallow cells; (WALL-NO-SHIP) the branch
  added per candidate is itself the new top line (LLVM codegen — the 3-for-3 of
  hand-scheduling); (EMPTY-CLASS) walker Ir share is already < the parse loop's
  (`l2-component-map`), so even a 30% walker win moves nothing.

---

### B-V24 · M7: good_match chain quartering — extend past #363's L6/L7 to L5/L8/L9
- **Vendor**: `vendor/zlib-ng/match_tpl.h:75-77` (`if (best_len >= s->good_match)
  chain_length >>= 2`); per-level `good` 4/4/4/8/8/8/32/32 from gzip lineage;
  libdeflate has NO equivalent (they halve on lazy deferral instead, P5).
- **Mechanism**: spend chain depth only when the current best is short.
- **Port cost**: `ldx/hc_matchfinder.rs` walk gets a `good` parameter threaded from
  `ldx/compress.rs` per level (#363 already did L6/L7 — read that PR's shape; extend
  is a parameter row + one branch).
- **Serves**: `HAIRLINE-L79` (weights L7-L9 hairlines — quartering at L8/L9 when the
  plate is already long saves wall on the SAME bytes IF the parse is insensitive).
  CAREFUL: this rides a falsified record. FALSIFIED AS COMPOSITION 2026-07-31
  (structure-comparison): depth+good_match measured TOGETHER lost worse on data.csv
  L9 (+458 B) and flipped 13 cells. Good_match ALONE was never judged with its own
  size gate — that is the re-open license: `fulcrum try` scoped `--levels 5,8,9`,
  size gates independent.
- **Instrument**: `board-size.sh tune` (size, free) → tie-guard → scoped try both coords.
- **Falsifier**: two-leg design: leg 1 = good-only at L5 (the plan-§5 candidate rung);
  leg 2 = good-only at L8/9 (wall only — if bytes change at all, it's dead there).
- **Risk**: the deepest falsified cohort lives here — two STRUCTURAL members died at
  the tie guard (hash3 chaining: 6 closed/12 flipped; zlib good_match depth+good:
  31 closed/17 flipped, data.csv L2 1.0000→1.0431), and the class re-sampled five times
  in one session was the codegen cousin. The 2026-07-31 receipt's WHY stands: flipped
  cells were byte-identical BECAUSE our config IS libdeflate's; any deviation pays.
  The only real prize is the WALL on cells that pass — and quartering changes bytes
  first.
- **Death modes**: (TIE-FLIP) overwhelmingly likely — pre-recorded; (EMPTY-CLASS)
  plus if `won_cells_stay_won` ledger re-verifies against the older 84-cell closure;
  (WALL-NO-SHIP) the walk it shortens may be the matchfinder we ALREADY win with.

---

### B-V25 · M11: longest_match_roll — the fast_zlib rolling-hash search for L9
- **Vendor**: `vendor/zlib-ng/match_tpl.h:83-115, 179-229` + insert_string_p.h:27-40
  (H3 rolling `(h<<5)^c`, 15-bit, `HASH_CALC_OFFSET STD_MIN_MATCH-1`); selected at
  level ≥ 9 (`deflate_slow.c:25-31`, deflate.c:1194-1197, fill_window 1240-1246).
- **Mechanism**: switch the WHOLE hash to 3-byte rolling so len-3 matches are
  findable; the search re-hashes scan interior bytes to JUMP CHAINS toward the most
  distant relevant position, and on every improvement re-scans prev[] across the match
  interior + probes head[] at len-2 — matches whose interior is hashed even when the
  head chain is exhausted.
- **Mechanism effect**: big DEPTH-BUDGET savings claimed at equal size (vendor claim;
  O-10) — the point at L9 is `weights L9 +0.00-0.02%` scale margins with per-label
  size binding.
- **Port cost**: second hashing mode in `ldx/hc_matchfinder.rs` + `ldx/matchfinder_common.rs`
  (roll insert); the walk becomes roll-aware; ~2 modules, medium size. The
  LOAD-BEARING fact: ldx has NO len-3 visibility today (hash3 exists only on the
  legacy L1 arm; ldx L9 is lazy2 over depth 600 — `ldx/compress.rs:132-133`), so the
  roll shape is a THIRD parser layer, not a tweak.
- **Serves**: `HAIRLINE-L79` (the deepest size cells) + L9 WALL.
- **Instrument**: `fulcrum try --levels 8,9 --threads 1,4 --corpus <declared superset
  incl. weights.safetensors>` scoped `--scope "levels=9;threads=1,4;corpus=weights…"`;
  Ir check via `fulcrum anatomy` per position.
- **Falsifier**: 1) measured depth statistics: log avg chain depth + hits per position
  on weights-style data for our L9 today (add a debug counter — instrument named by
  cell); 2) builder A/B size on the family; 3) the scoped try.
- **Risk**: zopfli's-hash-lineage; our L9 is `lazy2 (300,600)` per `ldx/compress.rs:132-133`
  — a third insertion ordering changes ALL decisions; byte-tie death at L9 for the
  600+ not-flipped cells.
- **Death modes**: (TIE-FLIP) the whole L9 tie cohort (biggest blast radius on the board);
  (EMPTY-CLASS) depth-600 is already sufficient on weights-like data (the hairline is
  0.00-0.02% — margin to construct around, not a search-depth deficit — check the
  `fulcrum why` position counts FIRST); (WALL-NO-SHIP) roll-hash inserts cost more than
  the chain budget saves on 32 KiB windows.

---

### B-V26 · P9: deflate_quick — static-trees L1 strategy (zero table work)
- **Vendor**: `vendor/zlib-ng/deflate_quick.c:48-138` (provenance :5-11); routed L1
  (`deflate.c:113`). Static-Huffman symbols emitted AS FOUND (no symbol buffer, no
  histogram, no dynamic tree); one static block spans the whole stream
  (`block_open`, `deflate.h:152-155`); single-probe hash (`quick_insert_value`
  `insert_string_tpl.h:42-56`); u32 first-4 compare then functable compare256;
  flush only when the pending buffer nears full (`:69-74`).
- **Mechanism**: no per-block table work at L1 (we pay histogram+tree+header per 64 KiB).
- **Port cost**: variant arm in `ldx/compress_fastest.rs` — it BYPASSES the seq store
  (V04 conflicts) and build tables; a "quick" rung exists in ldx only wholesale.
- **Serves**: `WALL` L1 (the Ir 150.84/b number), `HAIRLINE-NOCOMP` wall leg.
- **Instrument**: `fulcrum try --levels 1 --threads 1,4` + `fast_l1_ratio_multi_corpus`
  (pigz -1 text bound is the preserved property).
- **Falsifier**: builder A/B on `board-size.sh tune` L1 rows: the size give-back is
  the open question (O-2) — static tables pay a fixed ratio; on foto-like inputs it
  may LOSE bigger than the dynamic table would. Run the census FIRST (deterministic),
  then the scoped try.
- **Risk**: Phase-2 interplay (L1-in-ldx — plan §4); two competing L1 rewrites
  (hash3-in-port vs quick-arm) must not both start.
- **Death modes**: (EMPTY-CLASS) the L1 size class is GONE post-census (re-derive
  before building — handoff §4.7); (TIE-FLIP) all L1 ties vs libdeflate -1; (WALL)
  if our histogram+tree is <10% of the L1 Ir budget (anatomy says it is), the win is
  small.

---

### B-V27 · zlib-ng `configuration_table` as a FOUR-TUPLE GENERATOR (inverse: ours is FREE to change)
- **Vendor**: `vendor/zlib-ng/deflate.c:105-131` — the table IS the API; four numbers
  per level + strategy bound; `deflateTune` (deflate.c:639-648) exposes the values.
- **Mechanism**: none beyond B-V22's; the CARD is a governance note: our
  `ldx/compress.rs:115-134` table is a libdeflate copy and the charter says the map is
  ours to change ("never inherit a vendor's decisions", non-negotiable). The mid-band
  (L3-L5) is where the two tables disagree the MOST (chain 6/24/32 vs our 12/16/16).
- **Port cost**: level-table rows only.
- **Serves**: `CADENCE-L23`, `CADENCE-L5`, `LEN3-L3` — the whole cadence band is a
  4-tuple fit problem once the medium arm exists (V20).
- **Instrument**: the sweeps are builder-consts; the adjudicator grid is the same scoped try.
- **Death modes**: (TIE-FLIP) every tied cell (param changes bytes BY DESIGN — the
  band ONLY closes via one-level exception-retirement like #363/#364, never a
  sweeping map change); (WALL-NO-SHIP) cheap-ish; (EMPTY-CLASS) sweep hits a plateau
  (the L2 knob retune record says Lazy-at-L2 costs +23.4% Ir — parameter changes must
  re-run the STRATEGY census, hard stop #3).

---

### B-V28 · zlib-ng symbol-budget blocks (`lit_bufsize` 16384) vs our 300K/50K (O-6)
- **Vendor**: `vendor/zlib-ng/deflate.c:289, 352-360` + deflate_p.h:64-115 (tally-full
  check `sym_next == sym_end`); LIT_MEM split layout d_buf/l_buf 2-byte records
  (`deflate_p.h:64-101`) vs our 8-byte seq entries.
- **Mechanism**: block ends when the SYMBOL buffer fills ⇒ 3x smaller blocks than
  libdeflate's in symbols; gzip's `LIT_BUFSIZE` 0x8000 (trees.c:118-126) is its
  ancestor. Ours = libdeflate's two budgets (300000 bytes / 50000 seqs; fast 65535/8192,
  `ldx/mod.rs:148-164`).
- **Port cost**: consts (`ldx/mod.rs`) + `ldx/compress.rs` block accounting.
- **Serves**: `CADENCE-L5`/`CADENCE-L23` — the gzip-cadence band is a TABLE-REFRESH
  cadence band (see V03); the vendor bet (16K symbols) is the OTHER endpoint; igzip's
  default 64K tokens sits between. Same sweep as V03 — run ONE on the cadence files.
- **Instrument**: `examples/blockcensus` + `board-size.sh tune` sweep.
- **Falsifier**: the sweep records the size sensitivity; if smaller blocks are a wash
  on cadence cells and a regression on large blocks elsewhere, the class closes with
  its own numbers.
- **Death modes**: (TIE-FLIP) inevitably (bytes change); (EMPTY-CLASS) header share
  already tiny at our cadence (blockcensus says); (WALL-NO-SHIP) more blocks = more
  table builds (wall).

---

### B-V29 · M8: width-adaptive endpoint prefilter (2/4/8-byte by best_len)
- **Vendor**: `vendor/zlib-ng/match_tpl.h:63-68, 131-152` (`zng_memcmp_2/4/8` ladder:
  best_len < 4 → 2-byte, 4-7 → 4-byte, ≥8 → 8-byte, BOTH at scan_start AND
  scan_end=mbase_start+cur+offset); libdeflate/gzip single-tier variants (`hc_matchfinder.h:300-304`,
  gzip deflate.c:451-454); igzip has its own u64 form (huffman.h:260-314).
- **Mechanism**: cheaper chain-walking filter at SHORT best_len, where the 8-byte
  end-check loads (and their misses) dominate per-candidate cost.
- **Port cost**: `ldx/hc_matchfinder.rs` compare ladder branch — small, but this file
  has two in-loop FALSIFY records and the matchfinder is our CHEAPEST component
  (-2.2% vs libdeflate, `l2-component-map.md`); a candidate that beats THAT needs
  instruction evidence, not reasoning.
- **Serves**: `WALL` L2-L4 (short matches dominate on cadence-band cell types... which
  are size-failing, so the wall prize is indirect).
- **Instrument**: `examples/ldxloop.rs` + `fulcrum anatomy` at L2/L3; scoped try only
  if Ir moves.
- **Falsifier**: 1) count the walk's per-candidate loads today (`Dr` census); 2)
  builder A/B Ir; 3) try. The caveat from B-V35 reads across: at L9 we EXECUTE MORE
  matchfinder instructions than libdeflate and WIN the wall — Ir≠wall is the standing trap.
- **Death modes**: (WALL-NO-SHIP) the whole observed effect is register-pressure-driven
  and our walk wins already; (TIE-FLIP) none (byte-safe if the filter accepts exactly
  the same candidates — the width ladder MUST be provably accept-equivalent);
  (EMPTY-CLASS) walk Ir < 5% at target cells.

---

### B-V30 · M12/C4: WIN_INIT window over-zeroing (deliberate speculative reads made safe)
- **Vendor**: `vendor/zlib-ng/deflate.h:396` + `deflate.c:1261-1292` (`high_water`
  tracking, zero `WIN_INIT`=258 bytes past valid data so the prefilter's reads of
  unset lookahead are defined — marked "accessing uninitialized memory is deliberate"
  match_tpl.h:124-130).
- **Mechanism**: trade a tiny zeroing cost for the unclamped fast compare tails.
- **Ours is DIFFERENT**: caller-padded 16-byte tail (`parse::BUF_PAD`, applied at
  `deflate/mod.rs:160-165, 206-212`; `INPLACE_TAIL_PAD` alias :101-102) + clamped
  reads + matchfinders refuse within 5 bytes of end (`hc.rs:282-284`).
- **Port cost**: ldx-side window ownership (mmap/input lifetime) — heavier than it
  looks at the boundary (the 16-byte Z0 pad is a callers' contract —
  `deflate/mod.rs:160-165, 206-212`).
- **Serves**: mostly NEGATIVE space: we deliberately chose clamped reads after a real
  corruption (the lz_extend x86 corruption fix — CLAUDE.md status box). The card's
  value is the CONTRACT DIFF, not a port.
- **Instrument**: none — decision card.
- **Death modes**: (CORRUPTION) if anyone removes the clamps to mimic this; this card
  exists to pre-kill the "0-copy look-ahead" refactor.

---

### B-V31 · M10: hash-insert limiting — THREE vendor answers, never swept on our shape (O-3)
- **Vendor**: igzip `ISAL_LIMIT_HASH_UPDATE` — always on, insert only +0/+1/+2
  (`include/igzip_lib.h:119`, `igzip_base.c:73-85`); zlib-ng medium `<= 16 *
  max_insert_length` (`deflate_medium.c:70`), fast `<= max_insert_length`
  (`deflate_fast.c:79-87`); libdeflate (OURS): insert EVERYTHING (`skip_bytes`,
  `hc_matchfinder.h:360-399`) — the opposite.
- **Mechanism**: insert-cost vs match-quality knob on the jump after a match.
- **Port cost**: threeway threshold in `ldx/matchfinder_common.rs::skip_bytes` +
  `ldx/hc_matchfinder.rs::skip_bytes` (~:421).
- **Serves**: `WALL` (inserts/byte measured 0.276 vs libdeflate's 1.000 on our legacy
  arm — git-hist record — i.e. our shipped Insert densiity is ALREADY lower than
  libdeflate's somewhere else; the ldx port's number is unknown — count it first);
  `CADENCE` band indirectly (more inserts = better far matches = cadence repair).
- **Instrument**: `fulcrum anatomy` insert counter per cell (exists: `anatomy_count!(...)`
  machinery in sequences.rs) → then `fulcrum try --levels 2,4,5 --threads 1,4` scoped.
- **Falsifier**: count inserts/byte at L2 on text NOW (one anatomy run); if ours ≥
  libdeflate's 1.0 the lever has buy; if below, the sweep direction flips (RAISE
  inserts for cadence size).
- **Risk**: hard stop #3 — the insert-density record (0.276/1.000) was measured on the
  LEGACY matchfinder; post-port number may differ; never generalize.
- **Death modes**: (EMPTY-CLASS) insert Ir share below 3% of the loop; (TIE-FLIP)
  bytes change (only inside exception-retire path); (WALL-NO-SHIP) worse ratio forces
  deeper compensating search.

---

### B-V32 · zlib-ng stored arm (deflate_stored) + `matches` accounting — mostly a parity card
- **Vendor**: `vendor/zlib-ng/deflate_stored.c:27-136` — stored blocks copied straight
  `next_in → next_out`, header patched after a dummy `zng_tr_stored_block`,
  `read_buf` fuses the CHECKSUM; min_block adaptive to pending/window.
- **We already do**: L0 true stored passthrough in 65535-byte sub-blocks
  (`ldx/compress.rs` L0 arm + `deflate/mod.rs:63-64` `MAX_STORED_SUBBLOCK 65535`,
  `write_stored_subblock` :246-248).
- **The one live delta**: the crc fusion lives in `read_buf` for zlib-ng (B-V36),
  and zlib-ng's `s->matches` clamp for level-SWITCH hash validity is contract
  machinery we don't need (our levels are compiled-in, not runtime-switchable mid-stream).
- **Verdict**: no lever; keep the card to close the "their stored path is faster"
  comparison — `fulcrum why` on any stored-heavy cell (movie/photo L0/L1) names the
  real diff (usually our per-block 5-byte headers match theirs, so it will say
  position-counts match).

---

### B-V33 · zlib-ng `deflate_rle` (Z_RLE strategy) — a no-hash distance-1 parser
- **Vendor**: `vendor/zlib-ng/deflate_rle.c:24-80` + `compare256_rle`.
- **Mechanism**: pure run detection, distance-1 matches only, no hash table at all —
  chosen by STRATEGY, not level.
- **Ours**: no equivalent in the level engine (agent-verified in the index).
- **Serves**: `HAIRLINE-L79` (weights.safetensors: distance-1 runs are the meat of a
  safetensors zero-fill) — BUT as an opt-in STRATEGY it violates our per-label bar
  (a `-9` run must beat a level-9 rival, not an RLE rival) UNLESS adopted as a
  sub-strategy inside the level (e.g. run-aware hashing M13 as ECT does).
- **Instrument**: not promotable via try (strategy is not part of our CLI); the legal
  route is the run-aware-hashing idea (M13, zopfli `same[]` + second hash — index
  §M13), adopted INSIDE a level, not as a rival strategy.
- **Death modes**: (EMPTY-CLASS) inside the per-label rule; (TIE-FLIP) whole board.

---

### B-V34 · G3: SIMD slide_hash — the SHAPE to verify O-8 against
- **Vendor**: `vendor/zlib-ng/arch/x86/slide_hash_avx2.c:20-46` ─ `_mm256_subs_epu16`
  over head(65536)+prev(wsize); sse2 twin; NEON/VMX/LoongArch dispatches
  (functable.c:252-415); called from fill_window on slide (`deflate.c:1205-1218`).
- **Mechanism**: saturating-subtract halfword lanes = the same op as our branchless
  0x8000|(v&~(v>>15)) rebase on s16 (`ldx/matchfinder_common.rs:59-76`), which we run
  SCALAR. Whether rustc auto-vectorizes it is open question O-8 ("does rustc actually
  vectorize it?" — measure advisory).
- **Port cost**: NOT a layout swap (our rebase is per-32768-bytes and cheap); the MEASURE
  is first, and the aarch64 intrinsics fill for `matchfinder_init/rebase` is the cheap
  insurance (`libdeflate`'s own compile-time vector forms: `x86/matchfinder_impl.h:33-120`,
  `arm/matchfinder_impl.h:33-76` — libdeflate compiles choosing AVX2/SSE2/NEON by target).
- **Serves**: `WALL` (slides happen input/32768 times — if the scalar rebase is
  auto-vectorized, this is already optimal; if not, a NEON/SSE2 native init/rebase
  matches libdeflate's own compile-time dispatch, which the vendor index calls G6).
- **Instrument**: `cargo rustc --release -- --emit=asm` and read
  `matchfinder_rebase`'s codegen on BOTH arches (or godbolt-equivalent locally);
  then `fulcrum anatomy` slide counter if unvectorized.
- **Falsifier**: THE O-8 advisory: one asm read, one Ir delta — then decide.
- **Death modes**: (EMPTY-CLASS) rustc already emits `movdqu/pand/por`-shaped auto-vec
  (highly plausible for this loop shape); (WALL-NO-SHIP) slides < 0.5% of any named
  cell's Ir; (TIE-FLIP) none — byte-identical by construction (assert-identical
  outputs first; the saturating-rebase test in matchfinder_common.rs proves semantics).

---

### B-V35 · G2/compare256: the SIMD compare ladder
- **Vendor**: `vendor/zlib-ng/arch/x86/compare256_sse2.c:16-69`, `compare256_avx2.c:19-44`
  (2×32-byte unrolled), `compare256_avx512.c:20-60` (masked 64); NEON/Power/RVV
  (`functable.c:264-411`); generic SWAR `compare256_c.c:16-63`. igzip's tiers:
  `igzip_compare_types.asm:43 (u64), 102 (16B), 189 (32B), 289 (64B k-mask)`.
- **Mechanism**: match extension goes 8-at-a-time → 16/32/64-at-a-time.
- **Ours**: scalar u64 `lz_extend` + trailing_zeros (`ldx/matchfinder_common.rs:113-157`);
  there IS a precedent already in-tree: the NEON/SSE 16-byte match-EXTENSION landed
  with the perf/t1-output-cap slices (#366's SSE/NEON 16-byte match extension) —
  check `ldx/matchfinder_common.rs` for the shipped intrinsic shape before writing
  a new one (the SSE lz_extend polarity fix lives there too).
- **Serves**: `WALL` (long matches: nice 258 arms); tempered by the atlas: at L2/T1
  our matchfinder is CHEAPER than libdeflate's, and at L9 we run 26% MORE
  matchfinder instructions than theirs and WIN the wall cell by 12.3% — the two
  recorded falsifiers (register-pressure levers) died here. This card is
  "measure-only": `fulcrum profile` the wall band first.
- **Instrument**: `fulcrum profile` / `chainlat` on the wall band; `ldx_divergence.rs`.
- **Death modes**: (WALL-NO-SHIP) & (EMPTY-CLASS) as recorded; (TIE-FLIP) n/a when
  the extension returns the same length on both engines (intrinsic vs scalar must be
  assert-identical; the polarity bug already taught us that lesson).

---

### B-V36 · zlib-ng crc/adler `_copy` fusion in read_buf (I2)
- **Vendor**: `vendor/zlib-ng/deflate_p.h:159-182` + every SIMD checksum's `_copy`
  twin — checksum computed fused with the window copy (`crc32_copy`/`adler32_copy`).
  igzip: separate SIMD pass per consumed span (`igzip.c:134-148, 468-469`);
  libdeflate: one whole-input crc AFTER compression (`gzip_compress.c:67-79`).
- **Mechanism**: the input bytes are hot exactly once; fusing saves a second pass.
- **Ours**: SPLIT — streaming T1 interleaves per-chunk (`deflate/mod.rs:399-415`),
  whole-buffer T1 is a SEPARATE pass (`deflate/mod.rs:277, 304`; measured cost of the
  separate sweep: 29.3 ms / 232 MiB, comment `:571-574`); T>1 per-chunk + combine
  (`pipelined.rs:442-485`).
- **Port cost**: whole-buffer T1 route: fold crc32fast into the encode's input walk
  (feed the hasher from ldx's own reads) — infra-level, touch
  `src/compress/io.rs`/`deflate/mod.rs` plus the Hasher plumbing.
- **Serves**: `WALL` (all levels). The fused alpha ceiling is bounded by the
  checksum's own cost; count it, never infer: `fulcrum anatomy` may not have a
  checksum counter yet — if not, this is the first anatomy candidate.
- **Instrument**: `fulcrum try --levels 1,6,9 --threads 1,4` (checksum fusion touches
  every level) — full try, not scoped (behaviour is global).
- **Falsifier**: 1) count the separate pass's Ir (`fulcrum ab ablate` with the hasher
  stubbed vs full); 2) `board-size.sh tune` must be byte-neutral and MUST verify
  byte-neutrality (checksums do not affect deflate bytes — assert, don't assume); 3)
  the try.
- **Risk**: crc32fast's crate-dispatch vs our manual fusion (the crate's own SIMD
  path may already be near-optimal in-process — measure before replacing the pass).
- **Death modes**: (EMPTY-CLASS) if the anatomy shows the separate sweep ≈ the fused
  read cost (both must run over the input once either way); (WALL-NO-SHIP) fusion
  forces the hot compress loop to interleave a dependent chain;
  (TIE-FLIP) n/a (trailer-side only).

---

# J. Cross-vendor / falsifier-only cards

### J-V37 · P8: the TOO_FAR lineage sweep (len-3 distance gates) — three vendors, three answers
- **Vendor**: gzip `TOO_FAR 4096` (`vendor/gzip/deflate.c:129-132, 711-717`); libdeflate
  greedy 4096 (`deflate_compress.c:2573-2575`) lazy 8192 (`:2666-2668`); zlib-ng
  REMOVED it entirely (agent-grepped absent from the 2.3.90 tree); igzip moot (min
  match 4); zopfli: dist>1024 score penalty in the pre-parse (`lz77.c:265-271`).
- **Mechanism**: a len-3 match farther than the gate is worth less than two literals
  + a cheaper code.
- **Port cost**: `ldx/compress_{greedy,lazy}.rs` gate constants — our current:
  greedy 4096 / lazy 8192 (libdeflate's), plus the L1 hash3 max-dist 256-4096 family
  (legacy `parse/fast.rs:924`, `L1_HASH3_MAX_DIST=32768` shipped = no gate).
- **Serves**: `LEN3-L3` (access.log L3 vs libdeflate +0.42% — #364's far-len3 lever IS
  this family's ldx item; read #364 before any new attempt), `HAIRLINE-L79` at the
  lazy gate.
- **Instrument**: `fulcrum try --levels 3,5 --threads 1,4` scoped; builder-sweep the
  two constants on `board-size.sh tune` FIRST (deterministic, minutes).
- **Falsifier**: the sweep itself is the falsifier (three vendor values are ALL
  defensible — the corpus decides); the `won_cells_stay_won` ledger gates promotion.
- **Risk**: hard stop #3 — the gate interacts with `min_len` adaptation (P7) — bid
  sweep, not single knob.
- **Death modes**: (TIE-FLIP) every len-3-heavy tied cell; (EMPTY-CLASS) the gate only
  fires on >4096-apart repeated 3-grams (rare off log-class data); (WALL-NO-SHIP)
  weaker gates shorten
  matches but add matches elsewhere (the 2026-07-31 L2 Lazy record's per-file split
  shows the four text files that GET worse are exactly the access.log-shaped ones).

### J-V38 · O-8 measurement card: what rustc ACTUALLY does to our matchfinder scan/emit loops
- **Vendor (the ask)**: igzip G7 hand-asm whole kernels + libdeflate G6 compile-time
  SIMD init/rebase (cite both) are measured claims that hand-vectorizing these loops
  pays. O-8 asks the OPPOSITE question locally: is our scalar rebase/fill already
  auto-vectorized?
- **Method**: `cargo rustc -- --emit=asm` targeted at `matchfinder_init`,
  `matchfinder_rebase`, `lz_extend`, the ldx `heap sort block` — 4 symbols, read the
  instructions; x86_64 on solvency (cross-target not installed on the Mac) + aarch64
  locally. THEN: `fulcrum anatomy` Ir of the init/rebase share at T1 L2 (the slide
  event fires input/32768 iterations).
- **Serves**: the gate to V32 (SIMD slide) AND to adding any `matchfinder_impl`
  intrinsics. Named cell: any WALL leg of the cadence band.
- **Falsifier**: the asm read; publish to `docs/board/attack/atlas/` as a
  measurement-class finding when done (per seq-verdict playbook lane).
- **Death modes**: (EMPTY-CLASS) rustc vectorizes (record: stop chasing G6);
  (TIE-FLIP/WALL) n/a — instrument only.

### J-V39 · O-2 instrument card: what does a full-stream STATIC table cost TODAY (deflate_quick / E7 shared falsifier)
- **Ask**: ONE builder-flag static-tree L1 run on the TUNE census (bytes only) — decides
  P9 (V26) AND E7 (A-V14) simultaneously; if static-vs-dynamic at T1-L1 is within 0.1%
  on all cadence-band files, both cards reopen as WALL levers.
- **Method**: set the candidates' block tables to `StaticCodes` in a builder build
  (`board-size.sh tune --levels 1`), sha-verified; no production change.

### J-V40 · O-4: SIMD compare & the Ir-wall divergence floor (recorded STOP for V35)
- Records already banked: two register-pressure levers falsified (`hc.rs:388-403`,
  `:499-526`); matchfinder re-measured CHEAPER (-2.2%) with parse (+48.7%) and emit
  (+27.4%) as the real debt (`board/l2-component-map.md`); at L9 we run MORE
  matchfinder Ir than libdeflate and still win the wall.
- **Verdict**: no G2/M9 intrinsic additions until `fulcrum profile` shows the wall
  blocked INSIDE lz_extend on a NAMED wall cell. This card exists so a future session
  greps the stand instead of re-deriving it (a paste is not a decision).

---

## Roundup — how to read the catalog scores

Counting serves (using the cadence-table ids):
- `LADDER-L4`: V03, V20, V21, V22, V28, V31, J-V37.
- `CADENCE-L5`: V03, V04, V07, V20, V21, V28.
- `CADENCE-L23`: V03, V22, V28.
- `LEN3-L3`: V22, J-V37.
- `HAIRLINE-L79`: V12, V24, V25, V33.
- `HAIRLINE-NOCOMP`: V11, V14, V15, plus wall legs of V13 and V26.
- `WALL` axis: V01, V02, V05, V06, V09, V13, V23, V24, V26, V29, V32, V33, V35, V36, V38, V39.

Three envelopes wrap these:
1. **The cadence band is VENDOR-STRUCTURAL, not parameter-deep**: igzip's whole L1-3
   story (V03+V04+V07) is "rebuild exact tables on a SHORT cadence at token width" —
   that is the mechanism gzip's cadence exploit in us; nothing in band needs new
   hardware math, only new block cadence.
2. **The wall is already vendor-par**: after #366 the entire remaining gap is
   codegen (plan §6); every WALL card here is a codegen proposal, and the falsifier
   is always Ir/asm-first (`fulcrum anatomy`, `ldxloop`) before wall.
3. **In the tie cage**: 66/66 T1 ties at the tie-guard grades (L2/L6/L9, full corpus)
   means the size band cannot be entered by any parse change; only monotone-T1 wins
   buy headroom — the monotone subs are V12 (run-emit), V03/V28's block-cadence IF
   smaller blocks shrink (unproven), and the len-3 gate family (J-V37). Everything
   else in the size band enters through exception retirement one level at a time.

Standing records to grep before opening ANY of these (`grep -rnE 'FALSIF|REOPEN' src/`),
plus the retraction caveat on records themselves: FALSIFY comments are NOT binding
(retracted 2026-08-01 by the owner); re-measure when the question comes back, and scope
every record to its measured coordinate in the sentence itself (hard stop #3).
