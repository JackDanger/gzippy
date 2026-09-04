# PLAN 2026-09 — win at compression under "one encoder"

State-and-strategy doc written 2026-09-03 by the session taking over the campaign.
`CLAUDE.md` has the rules; this has the actual plan and the phase order. The old
`encoder-campaign-plan.md` is kept for its falsification records (§4) — everything
TOP of those records is superseded. The Aug-20 charter is superseded where it
contradicts this doc (§8 lists the deltas landed in the same PR).

---

## 0. What changed while the old plan slept (2026-08-21 → 09-02)

* **`ldx` is the encoder.** `src/compress/ldx/` is a per-decision Rust
  transliteration of `vendor/libdeflate/lib/deflate_compress.c`. Landed on main
  2026-08-23 via #357/#358/#359/#360; on main it runs at L0/L8/L9 (byte-identical).
  In-process, decision-identical: **1.03–1.11x of C libdeflate**, and it replaced
  an encoder that ran **1.6–4.1x slower than our own port** to defend 0.1–2.3%
  of size on a handful of files.
* **The unmerged 55-commit stack (`perf/t1-output-cap`) finishes the pivot:**
  L2/L4/L5 routed to the port, **pick-min deleted** (one encode per input),
  a **bit-splicing parallel writer** replacing sync-flush seams (the ~5 B/chunk
  framing floor is gone; fragments with stored blocks still byte-align via
  `ChunkMeta`), a thread-aware chunk grid (≈2 chunks/thread, cap 8 MiB L1-5 /
  2 MiB L6+, seams aligned to `SOFT_MAX_BLOCK_LENGTH`), an SSE `lz_extend`
  polarity fix (x86 corruption, landed 3 commits before tip), and NEON/SSE
  16-byte match extension.
* **`lever/ldx-good-match` (#363) + `lever/ldx-len3` (#364)** retire the L6/L7
  and L3 routing exceptions by making the port byte-identical to the legacy
  arms at those levels (`good_match` (128,65,8)/(256,130,32); far-len3 gate +
  224x sparse modulator). L1 stays the only exception (below).
* **Board before the pivot:** 200 failing of 1,320 measured cells
  (120bfa9c; libdeflate@T4 = 125 = 63%). After #227 (landed Aug 2): 85 failing
  on the TUNE census, of which 48 closed, residual 37 = **15 near-tie seam
  (≤0.1%) + 16 L1 (up to 4.58%) + 6 spread**. Those are the classes this plan
  inherits.

## 1. The strategy in one sentence

**Tie libdeflate byte-for-byte at T1 with a faster implementation, make T>1
emit the *same bytes* by carrying exact encoder state across chunks, and spend
the codegen headroom on the only remaining size lever (L1) and the wall.**

Per-label bar and every non-negotiable in `CLAUDE.md` still apply. Ties pass
`<=`-style comparisons, so byte-parity never loses a cell it could have won;
adopting-parse differences are adopted ONE level at a time through the
exception-retirement path (#363/#364 template).

## 2. The three live exceptions after the stack lands

| level | exception | why | exit condition |
|---|---|---|---|
| L1 | legacy igzip-derived fast path | beats pigz -1 on text where port is 1.038x (#347) | Phase 2 or owner decision |
| L4/L5 | *pending pick-min-deletion census* | KNOWN_SAGS (tabular,3)/(binary,3) re-opened L4<L3; L5 may have had pick-min-dependent wins | Phase 3 |
| L10-12 | our own `parse/ultra` engine | no libdeflate counterpart; Step 3 of the charter; unmatched by any numeric level | untouched |

## 2b. THE BOARD, MEASURED 2026-09-04 — 30 failing of 1,320, down from 200

**MEASURED on the frozen authority box (solvency, AMD Zen2, 4 rivals, 22 files,
L1-9, T1+T4, commit ee0c1d2c, binary sha 14ce0435..., roundtrip-verified,
artifact /root/www/gzippy-bench/campaign/size-all-ee0c1d2c/census.json):**

* **TUNE: 660 measured, ZERO bigger, zero roundtrip failures** (also banked as
  `size-tune-ee0c1d2c`). The pre-pivot residual (85 -> 37 after #227) is gone.
* **FULL BOARD (TUNE+GATE): 30 failing of 1,320 measured, down from 200 at
  120bfa9c. 170 cells closed by the pivot; 0 roundtrip failures anywhere.**
* **The L1 class (was 35 cells, worst +4.6%) and the libdeflate@T4 seam class
  (was 125 cells = 63% of the board) are BOTH GONE from the size board.**
* The 30 survivors (fulcrum board --size, ranked): `access.log L5 vs gzip
  +1.07%` / pigz +0.75%; `dd79_bin6 L2/L3 vs gzip/pigz` up to +0.93%;
  `minjs.min.js L5` +0.19/+0.16; `data.sqlite L4` +0.13/+0.17;
  `photo.jpg L1-L3 vs gzip` +0.04%; `weights.safetensors L7-L9`
  +0.00-0.02%; `access.log L3 vs libdeflate +0.42%` (the far-len3 machinery
  still on the legacy arm - #364's target); `movie.mp4 L6 vs libdeflate
  +0.0008%`.
* Concentration: L2-L5 gzip-cadence band plus dd79_bin6/access.log. Worst
  margin on the board is +1.07%. These are Phase 3's named targets.

NOTE: phase 2 (L1-in-ldx) targeted a class the census shows GONE - the
pre-census L1 idea aimed at the legacy igzip parser whose deficit the board
never saw again after the pivot. Re-derive necessity before building it;
the 30 named cells above are the priority order now. Phase 1 (state carry)
still has wall-axis value (T>1 codegen/tiering) but its SIZE premise is
superseded by the same census.

## 3. Phase 1 — exact state carry: T>1 becomes T1 bytes, the seam class dies by construction

**Mechanism.** Today each parallel chunk is compressed independently with a
dict window and an empty matchfinder; its block-split detector state (counts,
running histogram, bytes-in-block) also starts from zero. The chunk fragment
therefore has its own Huffman headers and its own block grid — the seam class.
Phase 1 seeds each chunk compressor with:

1. the 32 KiB window bytes (already done: dict seeding),
2. the **exact matchfinder table state** at its start position — deterministic
   to reconstruct by replaying the inserts of the preceding bytes (libdeflate's
   tables are pure functions of insert order; our port writes every position
   the T1 encoder wrote when given the same history),
3. the **block-split detector's counters and the in-flight block's histograms**,
   so the first block of chunk N is the continuation of the block chunk N-1
   ended mid-flight, using the combined histogram.

With all three, chunk N's emitted bytes are bit-identical to the T1 stream's
span for chunk N; the splicer's job reduces to byte-copying between chunks on
bit boundaries that cannot move. Output of T2/T4/T8/T16 == T1 == libdeflate,
on every file, every level routed to the port.

Why this is the right lever and not another grid shape: the five falsified grid
shapes (CLAUDE.md §2, `pipelined.rs`) established that seam cost cannot be
*shrunk* enough — the class needs **zero per-chunk reset**, which is what state
carry is. It is monotone by construction: output can only get smaller-or-equal
(it converges to the T1 bytes, which are already the shipped T1 bytes), so
clause 3 cannot trip, and the wall pays only the insert replay.

Cost model (to be counted, never inferred): replay = one extra sweep of inserts
over ~1 chunk of input per boundary ≈ 2x insert work; `fulcrum trace critpath`
+ paired wall decide whether it is invisible at T4 (slack measured 249–330%).
If the replay shows up in the wall, chunk caps (8/2 MiB) are the dilution dial,
and they were already measured at both endpoints.

**Falsifiers, in order:**

1. `scripts/campaign/board-size.sh tune` on a build with carry — the T4
   column must equal the T1 column byte-for-byte on every file (new
   `parity-census.sh` makes the same check a one-command gate for any
   build/ref).
2. Roundtrip unchanged (gzip/pigz/libdeflate decoders, sha256).
3. `fulcrum try <ref> --threads 1,4` on the frozen box — wall must not
   regress any T4 cell it previously passed (it will *win* most of them).
4. The replay's Ir delta counted once (`fulcrum anatomy` on an
   instrumented build, never quoted as a wall number).

**Expected closure:** all remaining `libdeflate@T4` seam-class cells
(= every cell that passes at T1 and fails only at T4). On the last measured
board that is the whole 125-cell libdeflate-T4 block; on the post-stack board
it is whatever the census in §2 measurement says is left. Either way it empties
the largest class on the board exactly once, by construction, without spending
a knob.

## 4. Phase 2 — L1 in the port: re-run the only measured size lever in the new codegen regime

The lever that nearly worked before (`ht_fast` + `hash3` at 256, the "one
vendor synthesis") died on wall **measured against legacy codegen** — a parser
that ran 1.6–4.1x our port. The regime changed; the wall verdict expired; the
size facts survive:

* fault line: 2-way buckets (libdeflate) + length-3 table (ours, threshold
  256 = measured interior optimum on 11 TUNE files; `gzip TOO_FAR` heritage);
* measured size: 4→8/11 passing L1 files at T1, the same at T4, **10 board
  cells closed, 0 opened** (documented in `encoder-campaign-plan.md` §B1 and
  `docs/board/l1-*.md`);
* `L1_HASH3_MAX_DIST=4096` (shipped) is monotone-worse than 256 (−642 B data.csv,
  −935 aozora, −1448 dickens, parked) — the value 256 comes from that sweep.

**Build it inside `ldx/compress_fastest.rs` path** (libdeflate's own L1 route is
`ht_matchfinder`-shaped — 2-entry buckets — so the delta is literally the 64 KiB
`hash3_tab` + the threshold + a `head3`-shaped check), not inside `parse/fast.rs`.
The port's ht implementation already has the SIMD `lz_extend`, the C-shaped walk
and 192 KiB working set; the length-3 table adds dependent-load pressure at every
HIT-able position, which is exactly what killed it last time — so the load
ordering pre-registration from the parked record applies (search first; blind
store on hit; load-then-store on miss).

**Order and gates:**

1. `fulcrum try --levels 1 --threads 1,4` on the frozen box. Particularly the
   T1 leg: last time the self-tax was measured at 15–50% *before* the port;
   if it is outside clause 5 even at T4, the L1 exception stays and the class
   is recorded closed on the wall at both coordinates — no re-sampling.
2. `fast_l1_ratio_multi_corpus` must stay green: the port+hash3 must remain
   `<=` pigz -1 on text (that cell is the reason L1 is an exception at all).
3. 256 was fitted on TUNE; promotion is judged on the GATE census only.

Expected closure: the 16 L1 cells of the post-#227 residual (43% of it),
potentially 10+ on the older 200-cell board.

## 5. Phase 3 — L4/L5 ladder and the gzip-cadence residuals

* **L4 ladder**: after the stack lands, resolve the two KNOWN_SAGS
  (`tabular`,3), (`binary`,3) — the L4-greedy < L3-lazy inversion that the
  pick-min deletion re-opened. The vendor-precedented candidate is
  zlib-ng's `deflate_medium` family (exists exactly to keep L3–6 monotone at a
  fraction of lazy's cost), i.e. an L4 rung inside the port, or extension of
  #363's `good_match` port to L4. Falsifier: `ladder_is_monotone_t1` +
  `won_cells_stay_won` + census.
* **L5 post-pick-min census.** If L5 lost won cells when pick-min went away
  (the aozora/minjs L5-band cells were won by the winning arm, not by a
  synthesized arm) and the zlib-style arm reproduces them at one encode, L5
  gets the #363 treatment (port the `good_match` rung). If even the single-arm
  cannot hold them, L5 stays an exception at T1 like L1 — same shape, no new
  rules.
* **igzip L1/L3 micro-cells (~6)**: only if Phases 1-2 leave them failing and
  a vendor diff names a mechanism; they have never warranted a lever by them-
  selves.

## 6. Phase 4 — the wall war (codegen-only, vendor-parity by construction)

The T1 wall class vs libdeflate is shallow levels on match-poor input
(31 of 51 pre-port losses at L1–3; worst 1.33–1.61x). After the stack, the
in-process gap is 1.03–1.11x with all work decision-identical — the entire
remaining gap is Rust-vs-C codegen. Names levers, in vendor-precedent order:

1. per-position prologue instruction count: fix the last known spill/hoist
   differences named by `fulcrum candidates` (the falsified space is the
   *chain walk* — the prologue has no falsify record);
2. if Ir/byte still loses after that: hand-shaped asm entry points for
   `longest_match`/`lz_extend` on x86_64+aarch64 (igzip precedent — their
   kernels are asm on both arches);
3. T>1 wall: after Phase 1 there are no seams left to schedule around — verify
   and bank with `fulcrum trace critpath` that no thread starves and that the
   splicer/writer is off the critical path; then leave T>1 alone.

## 7. The minimum instrument set (and nothing else gets built)

Already built — use, do not rebuild:

| question | command |
|---|---|
| did T4 shrink? cheapest falsifier | `scripts/campaign/board-size.sh tune` (then `all` for promotions) |
| is this ref shippable? | `fulcrum try <ref> --threads 1,4` (5. wave-runner.sh queues them) |
| verdict without rerun? | `fulcrum try --rescore <dir>` |
| what is the vendor diff? | `fulcrum why <cell>` / `fulcrum candidates <cell>` |
| CLI-drift third axis | `fulcrum dropin` |
| T>1 starvation/causation | `fulcrum trace …` |
| per-block BTYPE/hdr structure | `examples/blockcensus`, `examples/blockspans` (already exist) |
| block-placement ceiling | `examples/split_headroom.rs`, `examples/proposer_recall.rs` (measured 0.126–0.133%) |
| profiling a non-stripped build | `scripts/campaign/profile-ldx.sh` |
| codegen parity head-to-head | `examples/ldxloop.rs` + `ldx_divergence.rs` |

New in this plan (two files only):

* **`scripts/campaign/parity-census.sh`** — one question: *is every T>1 output
  byte-identical to T1 on the real 23-file corpus?* sha256 matrix, L0-9 x
  threads {1,2,4,8,16}, refuses an unidentified binary and a missing rival
  (same guards as `lib.sh`). This is the falsifier Phase 1 will be judged on
  and the regression gate the CI currently lacks (today's parity test covers
  4 tiny generated fixtures, where the grid is trivially single-chunk).
* **`examples/chunkgrid.rs`** — prints `pipelined_block_size(input, level,
  threads)` per fixture, so any arithmetic in this plan (fragments per file,
  expected replay cost) is a counted fact, not an inference.

Everything else is concluded. In particular: no per-chunk-grid sweeps (five
shapes falsified; the grid's remaining freedom is closed by Phase 1), no
Huffman-construction levers (exact package-merge dual-candidate closed at
~0.001% vs ~0.01% needed), no content detectors (non-negotiable #3 — the
`L1_HASH3_GATE` + `l1-tune` module deletions are queued as repo hygiene
commits in the same PR series that lands this plan).

## 8. Execution order, ownership, stop-rules

1. **Land the stack** (perf/t1-output-cap in reviewable slices, then #363,
   then #364). Each slice's gate: `cargo test --release` + `fulcrum try
   --threads 1,4` on the sliced ref + census zero-delta vs its own baseline
   where the slice is byte-identical.
2. **Bank the post-stack board** (`board-size.sh all`) — this sizes Phases 1
   and 2 against real numbers, replaces every hand-typed count above, and is
   the session's first deliverable once the slices merge.
3. **Phase 1** (state carry) — full falsifier list in §3. Stop rule: if the
   paired T4 wall regresses beyond clause 5 on cells it passed before, the
   lever is recorded closed-at-both-coordinates like L1/L4 before it; no
   re-sampling.
4. **Phase 2** (L1-in-ldx) — only its wall leg remains uncertain; everything
   else is measured history.
5. **Phase 3** mop-up; **Phase 4** is continuous, priority-lite, and must not
   block landed work (land-gated-win-first).

Charter deltas landed with this doc (see `CLAUDE.md` in this branch): STEP 1/2
rewritten around the one-encoder reality; the "seam closure needs headroom"
paragraph replaced by the state-carry formulation; toolbox `make falsified`
references replaced by the in-`src/` record search; memory pointer moved to
`docs/board/`; thread-parity-vs-non-negotiable contradiction resolved by
recording it as owner-directed (wall-for-size trade, #356 quote, 2026-08-22).
