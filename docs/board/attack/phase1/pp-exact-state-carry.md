# pp-exact-state-carry — Phase 1 implementation design (T>1 exact state carry)

Written 2026-09-05 from a full read of the brain worktree (`ee0c1d2c` lineage, docs tip
5900d17a). Theory of record: `docs/plan-2026-09-one-encoder.md` §3 ("Phase 1 — exact
state carry"). This file is the implementation-grade spec: every field, type, owner,
signature and falsifier a coder needs to build it in one turn. READ+WRITE-ONLY
planning: no cargo runs were performed here; every number below is quoted from
measured receipts or derived from the named source lines.

Status line the implementer must internalize first:

> **Carry means T>1 becomes the T1 streaming parse distributed across workers.**
> The T>1 chunk worker stops being a per-chunk fresh engine and becomes
> `parse_resumable` (the machinery `encode_deflate_stateful_to_writer`,
> `deflate/mod.rs:1466`, already exercises) fed by (a) a window replay that
> reproduces the matchfinder tables exactly, and (b) a 4-scalar handoff
> published by the preceding chunk's worker. Whole-file semantics are already
> proven by `tests/streaming_identity.rs` for the T1 streaming path; Phase 1
> re-uses those invariants and adds exactly two new mechanisms: the window
> sweep and the handoff cell.

---

## 0. Scope and the one design-fork ruling

**Levels in scope v1: 2..=7** — the levels whose T>1 chunk engine is
`parse::compress`'s `Strategy::Greedy | Lazy | Lazy2` over the hc matchfinder
(`level_uses_stateful_t4`, `parse/mod.rs:840`; hc tables,
`deflate/matchfinder/hc.rs:108`). Verified routing of `params_parallel`
(`level.rs:434`): L8/L9 return `params_parallel(11)` = `Strategy::NearOptimal`
(`level.rs:488-490`) — a different engine with its own big state, excluded from
v1 (§7); L1 keeps `fast_hash_update_inserts = 8` (`level.rs:615-617`) which
makes its insert set parse-dependent (Mode P, §2.3), excluded from v1; L0 is
pure stored, nothing to carry.

**The fork ruling (read before implementing).** Plan §3 envisions the seam as
"chunk N's fragment continues chunk N-1's block mid-flight, using the combined
histogram". Emission constraints force a sharper shape:

* A DEFLATE block's header (cost-model pick of dynamic/static/stored + the code
  tables) sits at the block's first bit, but is computed from the block's FULL
  histogram, which is known only when the split detector ends the block.
  Therefore the worker that owns the block's first byte must see the block's
  END before it emits anything (today `emit_block` already has this property —
  it runs after `run_block` returns; `parse/mod.rs:1294`).
* Fragments are spliced in slot index order (`infra/scheduler.rs:189-220`), so
  bits for a straddling block cannot be left to a later fragment without
  carrying Huffman codes across a fragment boundary — valid but pure cost.

**Ruling: each block is emitted WHOLLY by the fragment owning its first byte;
the owner parses forward past its grid boundary until the in-flight block's
split-detector end X′ (bounded by the `STREAM_BLOCK_LOOKAHEAD` margin) so it
holds the full (combined) histogram; chunk N's fragment then starts at a block
boundary.** The seam still dies by construction — every fragment is a bit-exact
slice of the T1 stream — but there is no mid-block fragment cut, `emit_block`
is unchanged, and `ChunkMeta` gains nothing (§4). The literal plan variant
(mid-block cut, codes handed off in `ChunkMeta`) is fully specified as Variant
B2 in §4.3 and rejected on cost, not on output (both produce T1's bytes). If
the owner wants fragments to mirror grid credit exactly, B2 is the build to
ask for; the state list below is written so only §4.3's extra fields change.

Deviation to acknowledge upstream: plan §3's sentence "the first block of chunk
N is the continuation of the block chunk N-1 ended mid-flight" is delivered as
"chunk N's first block is the T1-consecutive block after chunk N-1 completed
and emitted the straddler". Byte consequence identical; ownership differs.

**Precondition:** build on the stack's pick-min deletion (plan §8 step 1). In
the brain worktree today L1/L2/L4 T1 still run mmap pick-min
(`deflate/mod.rs:392-393`) and L5-L7 zlib pick-min — a carry T4 would converge
to the single-arm bytes, not those pick-min minima. On the stack, T1 is one
encode per input, so the parity target "T4 column == T1 column" is well-defined.

---

## 1. Today's chunk path — the facts (with line anchors)

```
pipelined.rs:574  compress_parallel_pipeline_pure
  └─ infra/scheduler.rs:126  compress_parallel(data, grid, T, writer, closure)
       ├─ writer thread (scheduler.rs:189-220): BitSplicer::splice_to per slot,
       │  in slot order, per ChunkMeta { pad_bits, needs_alignment } (bitstream.rs:333)
       └─ worker_loop_timed (scheduler.rs:292-345)
            claims slot j via AtomicUsize, slices block = input[S..S+len),
            dict = input[S-32768..S]                        (scheduler.rs:318-324)
       └─ closure (pipelined.rs:602-635):
            encode_deflate_splice_chunk_to_sink(block, dict, level, is_last,
                out, parallel=true, data_len)               (deflate/mod.rs:258)
              └─ deflate_into(parallel=true)                (deflate/mod.rs:595)
                   params = level::params_parallel(level)   (level.rs:434)
                   budget  = HeaderBudget::Generous
                   parse::compress(buf=[dict|data|pad], data_start, ...)  (parse/mod.rs:507)
                     └─ Strategy::Greedy → greedy::run      (greedy.rs:38)
                          ParseState::new() + dict seed via mf.skip_bytes (greedy.rs:52-60)
                          block loop: sink.begin() resets EVERYTHING        (parse/mod.rs:301)
                          emit_block → header from the block's own histogram (parse/mod.rs:1294)
            BitWriter::finish_unaligned → (bytes, pad_bits) (bitstream.rs:310)
            ChunkMeta { pad_bits, needs_alignment = STORED_BLOCK_EMITTED TLS }
            crc_parts[block_idx] = Hasher over the slot      (pipelined.rs:626-632)
```

Why the seam class exists, mechanically: every chunk starts (a) an empty
matchfinder with a 32 KiB dict seed whose stored positions are LOCAL
(`cur_pos = p_buf — in_base_local`), not the T1 values `p mod 32768`; (b) a
fresh block grid (the block-split detector's `should_end_block` gate
`ready_to_check_block` sees `in_end` = the chunk end, so the last ~5 KB of each
chunk can never split — a content decision the T1 stream does not make there);
(c) its own Huffman headers on its own block histograms. Fixtures where the
grid is trivially single-chunk see none of this — the shipped test coverage
stated exactly that (plan §7 note); the parity-census script (hardened,
1c0aefe6) is the real gate.

The T>1 knobs differ from T1 (`params_parallel`: depth ×4, `try_exact_huffman`,
`Generous` shaping, L4→Lazy, L8/L9→near-optimal) — these produce *different
bytes* from T1 today. Carry requires the chunk worker to run the T1 knobs
(`level::params(level)` + `HeaderBudget::Lean`); on the stack that is the
one-encoder collapse the plan asks for. Consequences listed in §6.4.

---

## 2. The exact state that must cross a chunk boundary

### 2.1 The replay theorem (what makes tables reconstructible)

Facts from `matchfinder/hc.rs` (identical shapes in `ldx/hc_matchfinder.rs`):

* Every stream position is inserted **exactly once, in increasing position
  order**, regardless of parse decisions: a taken match of length L at p inserts
  p (inside `longest_match`, before every early-out — `hc.rs:345-350`) plus
  p+1..p+L−1 via `skip_bytes` (`greedy.rs:248`, `lazy.rs:279/339`); a literal
  inserts p and the next token's probe inserts p+1. The lazy2 len-3 edge never
  double-inserts (the two lookahead probes ARE the inserts of p+1, p+2 —
  `lazy.rs:317-348`). Verified per-branch against `greedy.rs:231-356` and
  `lazy.rs:236-364`.
* The insert op is a pure function of the bytes at that position: `<hash3 =
  lz_hash(seq & 0xFFFFFF, 15)> → hash3_tab[h3] = cur_pos`,
  `next_tab[cur_pos] = hash4_tab[h4]`, `hash4_tab[h4] = cur_pos`
  (`hc.rs:673-675`; same order in `ldx/hc_matchfinder.rs:468-471`), advancing
  `next_hashes` to the hashes of seq(p+1) (`hc.rs:686-687`).
* `cur_pos = p − in_base` and `in_base` is always a multiple of 32,768
  (`parse/mod.rs:715-721`), so every stored entry value is `p mod 32768`; a
  slide fires exactly when the insert position is a window boundary
  (`hc.rs:664-668`), rebasing every prior entry by `matchfinder_rebase`
  (signed-saturating subtract of 32,768 — `ldx/matchfinder_common.rs:59-87`).
* Any entry inserted more than one window behind a query position has been
  rebased twice or more ⇒ saturated to `MATCHFINDER_INITVAL` (−32768) —
  bit-identical to "never inserted". Therefore the ENTIRE table state (all
  three arrays) at absolute position S is a function of the last ≤ 2 windows
  of inserts.

**Theorem used by the build.** For `s̄ = S mod 32768`, running the
`skip_bytes`-shaped insert loop over stream positions
`[S − s̄ − 32768, S)` (length `ℓ = s̄ + 32768 ∈ [32768, 65535]`, mean 49,152)
into a freshly-initialised matchfinder, starting with
`in_base_local = buf_pos(S − s̄ − 32768)` so `cur_pos` walks `0..ℓ` with exactly
one slide at the T1 slide point, leaves
`hash3_tab / hash4_tab / next_tab / in_base / next_hashes` **bit-identical to
the T1 whole-stream matchfinder at S**. Buffer requirement: the sweep's start
lies at most 64 KiB before the slot start — the head window (2.3).

The guard in `skip_bytes` (`if count + 5 > in_end − in_next`, `hc.rs:652`) must
not fire mid-sweep: the sweep is its own entry point (see `warm_inserts`,
§6.1) with the same loop body but an unconditional-run contract, documented
`T>1-only callers`.

For the ldx ht matchfinder (`ldx/ht_matchfinder.rs:56-88`): same theorem, one
flat `hash_tab: [i16; 65536]` (bucket 2), no separate `next_tab`, one
`next_hash: u32`; the bucket-2 slot-1 propagation is likewise position-pure at
every insert. Noted for the L1 follow-up; not touched in v1.

### 2.2 State crossing table — hc levels (L2..=7), v1 scope

| # | field | type | owner (writes) | consumer (reads) | transport |
|---|-------|------|----------------|------------------|-----------|
| 1 | `resume` | `usize` — absolute stream position after chunk N−1's last complete block | chunk N−1 worker (the `parse_resumable` return) | chunk N worker: parse start, sweep endpoint, `l3_sparse_split_latch` origin | `HandoffCell[j−1]` |
| 2 | `blocks_completed` | `u32` — running block counter across the whole file (only consumed for the one-shot L3 sparse latch at exactly `L3_OVER_SPLIT_LATCH_BLOCKS`) | chunk N−1 worker's latch state | chunk N worker's `run_resumable` loop | `HandoffCell[j−1]` |
| 3 | `split_hold_latched` | `bool` | chunk N−1 (`greedy.rs:112-119` / `lazy.rs:123-134`) | chunk N | `HandoffCell[j−1]` |
| 4 | `split_hold_decided` | `bool` | same source | chunk N | `HandoffCell[j−1]` |
| — | `hash3_tab: [i16; 32768]`, `hash4_tab: [i16; 65536]`, `next_tab: [i16; 32768]`, `in_base: usize`, `next_hashes: [u32; 2]` | matchfinder state | **NOT serialized** — rebuilt by `warm_inserts` over the head window (§2.1) from the shared input | chunk N worker, before its first token | input slice `[S − s̄ − 32768, S)` |
| — | `BlockSplitStats` (`block_split.rs:24-29`), `Sink::litlen_freqs/offset_freqs`, `litrun/nseqs/block_length` | in-flight block state | **NOT carried** — the owner worker completes the straddling block to X′, so no partial histogram ever crosses a fragment (§0 ruling) | — | — |

Notes on dropped candidates:

* `min_len` / `FarLen3Gate` recalc points are per-BLOCK
  (`greedy.rs:212-232`, `lazy.rs:212-233`) and derive from the current block's
  histogram; fresh block ⇒ fresh state. No carry.
* `SparseSplitGuardMul` is set per block from params (`greedy.rs:121`); only
  the latch booleans above persist across blocks.
* The fast strategy's cross-chunk state is `FastResume`
  (`fast.rs:3817-3838`: `head/head2/head3` u32 absolute-position tables,
  5 gate fields, `StoredCoalescer`) — it already exists for the T1 streaming
  path; L1 re-activation is the follow-up (§7), not v1.
* `input_total_len` is file-global (already passed down; `ParseState:729`).

Chunk 1 (`block_idx == 0`): no head window, no handoff; publishes
`HandoffCell[0] = { resume: 0, latches zeroed }` immediately so the chain is
never a wait on nothing.

### 2.3 What crosses on the OUTPUT side (splicer contract)

Nothing new. Fragments remain `(bytes, ChunkMeta { pad_bits, needs_alignment })`
and are consumed in slot order by the existing `BitSplicer`
(`bitstream.rs:395-482`); the CRC grid stays over slot boundaries
(`pipelined.rs:626-632`) and is untouched by which fragment covers which input
span. The per-chunk `resume` chain is worker-side state, not splicer state.

### 2.4 Window cost model — the two startup questions answered

**(a) Deterministic replay cost — counted, not inferred.** The sweep is the
`hc.rs:663-693` loop (shared body with `skip_bytes:643`). Per-position work,
counted from the loop body:

| op | count | note |
|---|---|---|
| `hash3_tab[h3] = cur_pos` store | 1 | L1 |
| `next_tab[cur] = hash4_tab[h4]` (load + store) | 1+1 | head read |
| `hash4_tab[h4] = cur_pos` store | 1 | L1 |
| `load_u32(buf, p)` unaligned | 1 | next seq |
| `lz_hash` ×2 (mul + shift each) | 2 mul, 2 shift | wrap-mult |
| `cur_pos += 1`, `remaining -= 1`, compare/branch | ~3 | loop |
| slide test (`cur_pos == 32768`) | 1 | cold |
| `prefetch_write` ×2 | 2 | hint |
| **total** | **≈ 13–16 uops, 4 mem ops** | |

**Sweep per chunk boundary = ℓ = s̄ + 32768 ≤ 65,535 positions ⇒ ≈ 0.4–0.9 M
cycles ≈ 100–250 µs at 3.5–4 GHz.** Dilution against the grid: 49,152 positions
is **1.5 %** of the 3.27 MB chunk a 26.2 MB file gets at T4 (dickens, 8
chunks), 9.4 % of a 512 KiB chunk, 0.6 % of an 8 MiB chunk. Against the branch's
measured wall slack of **249–330 % at T4** (plan §3, quoted verbatim) the sweep
is invisible *if* it is parallel (see (b)) — which the falsifier will confirm;
the exact number goes into `fulcrum anatomy` as claim #7, never as a wall
quote.

**Check against the plan's "≈2x insert work" budget:** the plan priced replay at
"one extra sweep of inserts over ~1 chunk of input per boundary ≈ 2x insert
work". The theorem (§2.1) proves the REQUIRED span is one window, not one
chunk: `Σ_chunks ℓ ≤ n_chunks × 64 KiB + input_len/2` — for a 1 GiB file at
512 KiB chunks that is ≤ 1.06 % of the plan's upper budget (≈ 33 MB of insert
sweeps vs ≈ 2 GB). Monotone cost reduction vs the plan's stated ceiling; kept
under the same falsifier (`replay_positions` anatomy count must show ≈ 49 Ki
per boundary, not ≈ chunk length).

**(b) Where replay happens — PER WORKER, at its chunk's start.** Ruling and
justification:

1. The sweep is a **pure function of the shared input** (§2.1): no dependency
   on the predecessor's run, no serialization, perfect T-way parallelism. A
   spare thread (or the writer thread) would serialize what parallelism gets
   free: one thread walking all boundaries turns
   `input_len/chunk × 150 µs` (~ms per big file) into a serial prologue on
   extra dependency edge — strictly worse shape, same total FLOPs.
2. The writer thread runs `BitSplicer::splice_to` on the only fully serial
   stage of the pipeline; the BlockSlot ring (`scheduler.rs:28-113`) plus its
   wait loop is the "reorder ring" whose slack (249–330 %) belongs to the
   WORKERS — the writer's per-slot path is wall-critical (its latency shows up
   1:1 in the pipeline). Parking the sweep on the writer would spend real
   slack currency; parking it in workers spends the idle quota that already
   exists.
3. The sweep overlaps with everything: a worker that renders chunk j spins on
   `HandoffCell[j−1]` (scalar ready flag, same shape as `mark_ready`), then
   sweeps its own head window (~150 µs), while all other workers compress.
   `CHUNKS_PER_THREAD = 2` keeps a straggler's chain latency ≤ half a chunk's
   share. The reorder ring's existing metrics (`total_wait_ns`,
   `scheduler.rs:202`) plus `trace_spans` ("write_wait", new
   "carry_wait" span) are the falsifier's instrument; no new thread roles.

Fallback (do not build unless the Ir falsifier shows the sweep stealing the
worker's L2 working set): a `replay` thread pre-warming `HandoffCell[j].tables`
(same 256 KiB tables, handed over as `Box` under a second ready flag). Recorded
here only so the branch is pre-named, not built.

---

## 3. The worker contract after carry

Per chunk (slot) j with grid slot `S_j = input[j·grid .. (j+1)·grid)`:

```
worker j:
  prev = handoff[j-1].acquire()            # scalar-ready spin, same shape as BlockSlot
  S      = start of MY slot                # grid arithmetic unchanged (pipelined_block_size)
  W      = prev.resume                     # stream position where MY parse starts
  head   = input[S-65536 .. S]             # extended dict window (scheduler change)
  tail   = input[slot_end .. min(len, slot_end + STREAM_BLOCK_LOOKAHEAD + BUF_PAD))
  # 1. seed tables (replay)
  q0 = W - (W mod 32768) - 32768           # clamp at 0
  warm_inserts(mf, buf=[head|block|tail|pad], q0 .. W)   # §2.1 theorem
  # 2. parse with T1 knobs, streaming-identical decisions
  in_end_e = min(slot_end + STREAM_BLOCK_LOOKAHEAD, input.len())
  resume' = parse_resumable_carry(buf, state(replayed), W,
               msg_end = slot_end, in_end = in_end_e,     # decisions see the lookahead
               role = j == last ? Final : Interior,
               input_mode = Bounded,                      # stops at last complete block
               budget = Lean, params = level::params(level),
               handoff_in = prev, handoff_out = mine)
  # 3. fragment = T1 bits for blocks with begin in [W, resume′)  (owned blocks incl.
  #    any straddler completed to X′ ≤ in_end_e − 5 KiB)        (§4)
  publish HandoffCell[j] = { resume: resume′, latches }
  meta = { pad_bits: bw.finish_unaligned(), needs_alignment: STORED_BLOCK_EMITTED }
```

Decision-identity argument (all measured in the T1 streaming path already —
`STREAM_BLOCK_LOOKAHEAD` doc, `parse/mod.rs:781-792`, pinned by
`tests/streaming_identity.rs`): for every block decided inside worker j's span,
`choose_max_block_end` (`parse/mod.rs:968`) sees
`in_end_e − block_begin ≥ 305,000` ⇒ returns `block_begin + 300,000` exactly as
T1; `adjust_max_and_nice_len` never clamps (`parse/mod.rs:978`); the split gate
`ready_to_check_block(bytes_in_block, in_end − in_next)`
(`block_split.rs:135`) is true iff T1's is (both see ≥ 5,000 remaining); the
Bounded guard stops the loop at the first block boundary where fewer than
305 KB remain ⇒ the returned `resume′` ≤ slot_end — fragments tile the stream
consecutively, and the final chunk (role=Final, Drain, in_end = file end) sets
BFINAL exactly as the last chunk of the T1 streaming pass does.

---

## 4. Bit-splice spec under state carry

### 4.1 What changes in `emit_block` / finish sequencing

**Nothing in the functions; everything in the wrapper.**

* `emit_block` (`parse/mod.rs:1294-1585`) is invoked with the SAME
  (whole-block histogram) inputs as today — the owner's carry-parse extends
  into the tail's data range before emission, so the stored/static/dynamic
  cost pick and the header are computed from the full block exactly as T1.
  No mid-block state leaves the owner.
* Block closing (`emit_stored_block` / BFINAL / `align_to_byte`,
  `deflate/mod.rs:1589`) is unchanged, including the straddler's BFINAL=0/1
  side: BFINAL is set by the worker whose role is Final (last slot), on the
  block that reaches input end — bit-identical to the T1 streaming close.
* The `STORED_BLOCK_EMITTED` tripwire (`deflate/mod.rs:320-343`) keeps its
  semantics: `needs_alignment` still flags fragments containing BTYPE=00 framing.
  Under carry this becomes RARER per fragment but identical in kind; the
  splicer's `emit_alignment_seam` (`bitstream.rs:488-503`) still fires only
  for an alignment-sensitive fragment arriving unaligned. A fragment that ends
  exactly on a stored sub-block boundary (the `StoredCoalescer` case is
  fast-only, out of v1) ends byte-aligned with `pad_bits = 0` and splices
  verbatim: no seam inside a carried stored span is possible in v1 because
  only the fast parser coalesces (out of scope), and hc's stored escape
  (`emit_block`'s stored arm, `parse/mod.rs:1520-1543`) emits closed blocks.

### 4.2 What `ChunkMeta` gains

**Nothing — in the adopted ruling.** The fragment-boundary state lives in the
worker-side `ChunkHandoff` (§2.2), never in the splicer contract. ChunkMeta
keeps `pad_bits: u8` + `needs_alignment: bool` exactly as shipped
(`bitstream.rs:333-352`).

### 4.3 Variant B2 — the plan's literal mid-block cut (fallback spec, not built)

If ownership-credit symmetry ("fragment content == grid slot") overrides the
cost arguments, the fragment cut moves inside the token stream and the
handoff grows:

* `ChunkMeta` gains
  `continuation: Option<Continuation<'static>>` where
  `struct Continuation { choice: BitChoice(Dynamic|Static),`
  `lit_lens: Box<[u8; 288]>, off_lens: Box<[u8; 32]>,`
  `seq_tail: Vec<Seq>, litrun_pending: u32, litrun_start: usize }`
  (`Seq` = `parse/mod.rs:120-127`; the code tables come from the full
  histogram computed by the header owner — same carry-parse requirement as
  the ruling).
* Sequencing: chunk N−1's `emit_sequences` walk stops at the last token fully
  inside its credit span, splitting a pending literal run at the cut; chunk N
  re-opens emission with the carried codes, emits the token tail to X′ with
  NO header, then proceeds to fresh blocks. A stored-priced straddling block
  cannot be split mid-sub-block at all: the cut is forced to a sub-block
  boundary and the span continuation is carried (fast-only concern).
* Cost: a new truncating emit path, ~0.4 KiB boxed state + up to ~400 KiB of
  `Seq` tail in the worst straddle, a second `HuffmanCode` handoff, and a
  mid-codeword fragment start (splicer shift path — already supported).
* Output: byte-identical to the ruling's shape and to T1. No size or wall
  upside was found in the walk; every delta above is machinery cost.

The ruling (§0) stands unless the owner asks for B2 explicitly; the code plan
below implements the ruling only.

---

## 5. Falsifiers

Verbatim from plan §3 (they are Phase 1's acceptance gate; none may be
re-derived or softened):

1. `scripts/campaign/board-size.sh tune` on a build with carry — the T4
   column must equal the T1 column byte-for-byte on every file (new
   `parity-census.sh` makes the same check a one-command gate for any
   build/ref).
2. Roundtrip unchanged (gzip/pigz/libdeflate decoders, sha256).
3. `fulcrum try <ref> --threads 1,4` on the frozen box — wall must not
   regress any T4 cell it previously passed (it will *win* most of them).
4. The replay's Ir delta counted once (`fulcrum anatomy` on an instrumented
   build, never quoted as a wall number).

New falsifiers implied by this build:

5. **Replay A/A determinism.** Property test in
   `deflate/matchfinder/hc.rs`'s test module following the existing `RefHc`
   reference-net pattern (`hc.rs:717-735`): drive a greedy parse over a
   seeded ≥ 1 MiB buffer, snapshot `hash3_tab/hash4_tab/next_tab/in_base/
   next_hashes` at absolute S; then a fresh matchfinder +
   `warm_inserts(q0..S)` must produce a byte-identical snapshot — for a
   lattice of S values (aligned + misaligned to the window, first/last/only
   chunk, S < 65,536). Run twice from different pool states to pin
   determinism across threads (no shared state beyond the input slice).
6. **Seam-confidence diff per chunk.** `examples/carry_diff.rs` (patterned
   on `examples/ldx_divergence.rs`'s `decompress::block_walker` diff, which
   reports the first divergent decision): walk the T4-carry stream and the T1
   stream, assert zero decision diffs, and attribute the first divergence (if
   any) to its grid chunk index by absolute position. One command per file,
   milliseconds, no box round-trip.
7. **Ir delta counted once** (sharpened #4): counters
   `carry_replay_positions` + anatomy-wall region `carry_replay_ns` fire ONLY
   inside `warm_inserts` (the sweep site), so the replay never double-counts
   into the parser's `hc_positions_skipped` or the wall regions, and the
   sweep-vs-chunk-work ratio is a single anatomy row. Gate: replay Ir ≤ 5 %
   of its chunk's Ir (upper bound from (a); the number is counted on the box,
   never inferred and never translated into a wall claim).
8. **The T4 wall must not regress — the three census arms**, in plan §7's
   vocabulary: (i) `board-size.sh tune` size+T4, (ii) `board-size.sh all`
   for promotion, (iii) hardened `parity-census.sh` byte-matrix — plus
   `fulcrum try <ref> --threads 1,4` (#3 above). Expectation to pre-bank: the
   carry worker runs T1 knobs (`params(level)`), so the depth-×4 parse
   (params_parallel) leaves the T4 wall at shallow levels — the wall moves
   DOWN both from that and from the sweep being sub-1 % — falsifier #3 wins
   rather than holds.

Stop rule (plan §8.3): if the paired T4 wall regresses beyond clause 5 on
cells it passed before, the lever is recorded closed-at-both-coordinates like
L1/L4 — no re-sampling.

---

## 6. Patch outline (file-by-file, coder-ready signatures)

### 6.1 `src/compress/deflate/matchfinder/hc.rs`

* Factor the store-loop out of `skip_bytes` (line 643) into
  ```
  #[inline(always)]
  fn insert_positions(&mut self, buf: &[u8], in_base: &mut usize,
      mut in_next: usize, count: usize, next_hashes: &mut [u32; 2])
  ```
  (the raw loop body exactly as shipped including slide-first, prefetch, and
  the closing `next_hashes` write; `skip_bytes` keeps its decline guard
  around it — zero behaviour change, `cargo test` byte-identical).
* Add, with `T-SCOPE: **T>1 ONLY**` marker and a doc comment citing §2.1:
  ```
  /// Window-sweep warm start for the T>1 carry pipeline
  /// (docs/board/attack/phase1/pp-exact-state-carry.md §2.1).
  /// phase1/pp-exact-state-carry.md §2.1). `sweep_from`/`sweep_to` are BUF
  /// positions; `in_base_seed` must place `cur_pos = 0` at the window
  /// boundary the sweep's single wrap crosses.
  pub fn warm_inserts(&mut self, buf: &[u8], in_base: &mut usize,
      sweep_from: usize, sweep_to: usize,
      next_hashes: &mut [u32; 2])
  ```
  implementation: `matchfinder_init`-fresh tables assumption + call
  `insert_positions(buf, in_base, sweep_from, sweep_to − sweep_from,
  next_hashes)` with the slide arithmetic active; emits
  `anatomy_count!(carry_replay_positions)` per position.
* Add the A/A determinism test (§5.6 falsifier).

### 6.2 `src/compress/deflate/parse/mod.rs`

* Hoist the three per-`run_resumable` locals (`blocks_completed`,
  `split_hold_latched`, `split_hold_decided` — `greedy.rs:99-101`,
  `lazy.rs:109-111`) into `ParseState` (fields listed in §2.2 with types and
  owners). `greedy::run_resumable` / `lazy::run_resumable` read/ write them
  from `state` instead of locals — pure code motion, output unchanged.
  `set file_start` semantics: `l3_sparse_split_latch` (mod:1015) must see
  STREAM-absolute bytes (`file_start_buf = from_buf − resume`) — one-line
  plumbing at the two call sites.
* Add (T>1-only module, `T-SCOPE` marker):
  ```
  pub(crate) struct ChunkHandoff {
      pub resume: usize,              // bail-in stream position for the next chunk
      pub blocks_completed: u32,
      pub split_hold_latched: bool,
      pub split_hold_decided: bool,
  }

  /// T>1 carry: seed `state` (fresh pooled matchfinder) via `hc::warm_inserts`
  /// over `buf[q0..W]`, apply `handoff_in` latches, run one resumable parse
  /// bounded by `STREAM_BLOCK_LOOKAHEAD`, return the position after the last
  /// complete block, and fill `handoff_out`.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn parse_chunk_carry(
      buf: &[u8],
      state: &mut ParseState,
      from: usize,            // buf position of the first coded byte (= W in buf coords)
      in_end: usize,          // min(slot_end + STREAM_BLOCK_LOOKAHEAD, buf work end)
      params: &LevelParams,   // level::params(level) — T1 knobs, NOT params_parallel
      statics: &StaticCodes,
      bw: &mut BitWriter,
      role: BlockRole,        // Interior or Final
      budget: HeaderBudget,   // Lean
      handoff_in: &ChunkHandoff,
      handoff_out: &mut ChunkHandoff,
  ) -> usize
  ```
  body = existing `parse_resumable` (`parse/mod.rs:855-961`) with the
  Greedy/Lazy arms reading latches from `state`; the sweep call precedes it
  inside the wrapper. No changes to `parse()`/`compress()` (T1 byte path).
* `pub(crate) fn carry_active(level: u32) -> bool` —
  `matches!(level::params(level).strategy, Strategy::Greedy | Strategy::Lazy |
  Strategy::Lazy2)` (i.e. levels 2..=7 as routed today), read once per run by
  the pipeline.

### 6.3 `src/compress/deflate/mod.rs`

* Add (KEEPING the existing splice fn untouched for L0/L1/L8/L9):
  ```
  /// T>1 carry chunk (docs/board/attack/phase1/pp-exact-state-carry.md).
  /// buf layout [head_window 64 KiB | data | lookahead pad | BUF_PAD];
  /// runs `parse_chunk_carry` on the T1 params (parallel knobs frozen out).
  pub fn encode_deflate_carry_chunk_to_sink(
      data: &[u8], head_window: &[u8], level: u32, is_last: bool,
      out: &mut Vec<u8>, input_total_len: usize,
      handoff_in: &mut ChunkHandoff, handoff_out: &mut ChunkHandoff,
  ) -> bitstream::ChunkMeta
  ```
  wraps `deflate_into_carry(bw, buf, data_start = 64 * 1024,
  in_end_e = …, level, is_last, parallel=true)` — the parallel path routes to
  `parse_chunk_carry` instead of `parse::compress`; returns `ChunkMeta` from
  the same `finish_unaligned` + `STORED_BLOCK_EMITTED` protocol as today.

### 6.4 `src/infra/scheduler.rs`

* `HandoffCell` (same single-writer protocol as `BlockSlot`):
  ```
  pub struct HandoffCell { ready: AtomicBool, inner: UnsafeCell<ChunkHandoff> }
  impl HandoffCell { pub fn publish(&self, h: ChunkHandoff); pub fn acquire(&self) -> ChunkHandoff; }
  ```
* `compress_parallel`'s closure signature extends by two refs and the head
  window (replacing the existing `dict` slice, which is its last-32 KiB
  prefix — strictly larger today):
  ```
  F: Fn(
      usize,              // block_idx
      &[u8],              // slot = input[S..E]
      &[u8],              // head window = input[S.saturating_sub(65536)..S]
      bool,               // is_last
      &mut Vec<u8>,       // fragment buffer (slot-owned, as today)
      HandoffView<'_>,    // { prev: &HandoffCell (None for j==0), mine: &HandoffCell }
  ) -> ChunkMeta + Sync
  ```
  Worker spins on `prev` before invoking the closure (bounded by the
  predecessor's parse; `trace_spans::record("carry_wait", …)` for the
  falsifier). The `ffi-oracle` users keep `compress_parallel_independent` and
  are cfg-unchanged.

### 6.5 `src/compress/pipelined.rs`

* The worker closure dispatches: `carry_active(level)` ⇒
  `encode_deflate_carry_chunk_to_sink(...)`; otherwise the existing
  `encode_deflate_splice_chunk_to_sink` arm (L0 stored, L1 chunked fast,
  L8/L9 chunked near-opt path) — untouched. CRC slots unchanged. The doc
  comment block above the call site updates its 'stronger parse' sentence:
  carry levels now run T1 knobs by construction (one encoder).
* Consequence ledger (keep in a comment, verified by the census):
  `params_parallel`'s depth ×4, `try_exact_huffman`, `Generous` shaping,
  L4→Lazy step and L1's `usize::MAX` insert policy go dead on levels 2..=7 at
  T>1 — the T4 output at those levels IS the shipped T1 stream. Post-carry
  `board-size.sh all` must show zero net cell flips (T4 column = T1 column);
  any T4-passing-cell whose T1 twin fails would surface here and must be
  adjudicated before promotion (expected: zero, per the §2b census where T1
  is the byte-parity truth at these levels).

### 6.6 Instruments and tests

* `examples/carry_diff.rs` (§5.6) + a `--per-chunk` ledger of fragment
  byte-spans so failing seams name their chunk, mirroring
  `examples/ldx_divergence.rs`'s output contract.
* `tests/carry_parity.rs`: byte-identity of `encode_*_carry` chunked vs
  whole-buffer T1 across inputs that cross every grid/lookahead boundary
  (128 KiB, 300,001, 305,000+1, 512 KiB±1, 1 MiB, multi-window, incompressible
  stored-escape stretches) × levels 2..=7 × handoff misalignment cases
  (`W` not a 32768 multiple).
* anatomy counters: `carry_replay_positions`, wall region
  `carry_replay_ns` (fires `n_chunks` times per encode exactly).

---

## 7. Out of v1 scope (named follow-ups, no code here)

* **L8/L9 near-optimal carry** — the chunk engine today is `params_parallel(11)`
  (`Strategy::NearOptimal`, `level.rs:488-490`); its handoff would carry the
  near-opt parser's own state (per-position chain, cost-model tables, its
  block splitter). The same window-replay theorem applies to its hc tables;
  the split/squeeze state is its own deliverable. Until then L8/L9 keep the
  two-candidate pick (chunked vs `whole_t1`, `pipelined.rs:302-331`), whose
  whole_t1 arm already bounds seam damage.
* **L1 fast carry (Mode P replay).** `fast_hash_update_inserts = 8`
  (`level.rs:607-614`) makes L1's insert set parse-dependent, so its replay is
  a mini-parse over the head window (the fast single-probe loop with emission
  suppressed), not a pure sweep; `FastResume` (`fast.rs:3817-3838`) is the
  carried struct and needs no new fields. Gated behind the L1 exception's
  Phase-2 re-derivation, not Phase 1.
* **L0 stored path** — no matchfinder, no histograms, no carry definition
  possible; it stays the single-shot stored pass (`pipelined.rs:332-371`).

## 8. Source receipts for this doc

All line numbers verified in this worktree at read time (2026-09-05):
`pipelined.rs:110,574,602-635`; `infra/scheduler.rs:28,126,189-220,292-345`;
`deflate/bitstream.rs:310,333-352,370-515,488-503`;
`deflate/mod.rs:258-318,320-351,595,871,874,1400-1466,1575-1619`;
`deflate/parse/mod.rs:83,86,120,185-211,301,315,413,507,713,792,840,855,968,
1015,1051,1114,1128,1185,1294`; `deflate/block_split.rs:18,21,24-148`;
`deflate/parse/greedy.rs:38,76,99-123,177,231-356`;
`deflate/parse/lazy.rs:87,109-134,183,236-364`;
`deflate/parse/fast.rs:1474,2086-2095,2353,2478,3817-3898,3931-4155`;
`deflate/matchfinder/hc.rs:97,108,262,345-350,643-703,717-735`;
`ldx/hc_matchfinder.rs:41,171,225-239,421-488`;
`ldx/ht_matchfinder.rs:56,122,164-211,231-285`;
`ldx/matchfinder_common.rs:5-102`; `deflate/level.rs:434,488-490,536-549,600,615-618`;
`deflate/encode_types.rs:20-78`; `examples/chunkgrid.rs`;
`examples/ldx_divergence.rs`; plan doc §2b/§3/§7/§8.
