# VAR_V Integration — Adversarial Disproof Verdict

Read-only, source-first. Repo branch `reimplement-isa-l`, HEAD `fa9fd73c` + the one
uncommitted +181-line change in `src/decompress/parallel/marker_inflate.rs`
(`git diff --stat` confirms the diff IS exactly the "VAR_V SPECULATIVE
SOFTWARE-PIPELINED FAST LOOP" block, lines 1454–1633).

---

## CLAIM 1 — BYTE-EXACTNESS — **UPHELD**

Every sub-attack was source-checked against the careful loop (`marker_inflate.rs:1635-1785`)
and the decode/back-ref primitives.

- **8-byte speculative store cannot straddle.** Store is at
  `ring8_fast.add(dst_phys)` writing `sym0 & 0x00FF_FFFF` as a u64
  (`:1514-1517`). Top-of-loop guard `dst_phys + FAST_OUT_SLOP <= U8_RING_SIZE`
  with `FAST_OUT_SLOP = 8 + 258 + 16 = 282` (`:1474`, `:1492-1493`). So
  `dst_phys + 8 <= U8_RING_SIZE - 274` — the 8-byte store never reaches the ring
  end. The packed mask is only 3 bytes (`0x00FF_FFFF`); the LUT packs at most 3
  literal lanes (bits 0..25, `lut_huffman.rs:63-65`, `LARGE_SYM_COUNT_MASK`
  `:76-78`), so 3 bytes covers every literal lane low-byte-first — identical
  ordering to the careful loop's per-lane `code & 0xFF` + `sym >>= 8`
  (`:1685`, `:1704`).
- **Word-copy back-ref overshoot bounded.** `emit_backref_ring_u8` non-overlap
  rounds the run up to a multiple of 8 (`rounded = (length+7)&!7`, `:2888`),
  max overshoot **7** bytes (the comment's "16" is conservative). The guard
  reserves 16. The routine *also* self-protects: `dst_round_fits`/`src_round_fits`
  (`:2889-2890`) fall to a `% U8_RING_SIZE` per-byte arm (`:2912-2918`) if the
  word copy would straddle, and the RLE/overlap arms mask every index
  (`:2922-2938`). Cannot corrupt undrained bytes: the store writes only at
  `pos..pos+8`, all `>= pos` (never-yet-emitted slots); back-ref source
  `dst - distance < dst` reads only committed bytes behind `pos`; the up-to-2
  speculative garbage bytes (a trailing length code's low bytes) sit *ahead* of
  the advanced `pos` and are overwritten by the back-ref dst (`:1608-1619`).
- **Literal-prefix unpack identical.** Fast loop counts leading literals
  (`lit_prefix`) and isolates the single trailing `code > 255` element
  (`:1518-1539`); `pos`/`emitted += lit_prefix` (`:1540-1541`). Trailing
  EOB → `at_end_of_block = true` + `Ok(emitted)` (`:1545-1551`), matching careful
  `:1709-1712`. Trailing length → `length = code - 254` (`:1558`) matching careful
  `:1716`. The inner `if remaining == 1 && code > 255` (`:1526`) is provably dead
  (to enter the branch with `remaining==1`, `code<=255` must hold) — behaviorally
  inert, not a divergence. NOTE #4 (`:1371-1376`) says multi-sym packets never
  end in EOB, so "literal+EOB" doesn't occur; both loops handle it identically
  anyway.
- **Fast→careful handoff has no desync.** `LutLitLenCode::decode`
  (`lut_huffman.rs:1021-1063`) only `peek`s — it never calls `bits.consume`.
  The fast loop consumes via `bits.consume(bit_count0)` *after* reading `pre`
  (`:1506`); every `break 'fast` / fall-through leaves a freshly-decoded,
  **un-consumed** `pre` (`:1488` preload, re-preloaded at `:1626`), so the bit
  cursor sits exactly before `pre`'s bits and the careful loop re-decodes the same
  symbol (`:1656`, then `bits.consume`). The `bit_count0 == 0` invalid-code case
  breaks *without* consuming (`:1501-1505`) and lets the careful loop raise the
  error — clean, no double-commit.
- **Error/EOB returns commit state.** Each early return sets
  `ring_pos`/`decoded_bytes += emitted`/`distance_to_last_marker_byte`
  (`:1546-1556`, `:1566-1606`), matching the `commit!` macro (`:1432-1439`).
- **Distance uses production `self.dist_hc`, NOT the bench's `LutDistCode`.**
  Confirmed `:1563`. Dist-extra read (`:1578-1595`) and the clean-mode
  `distance > self.decoded_bytes + emitted` check (`:1602`) are byte-identical to
  the careful loop (`:1739-1755`, `:1763`). (The bench instead uses
  `build_block_tables` → `LutDistCode` on a flat buffer, `engine_isolation.rs:436`.)
- **Slow-knob + marker gating.** Fast loop entered only when
  `!CONTAINS_MARKERS && slow_spin == 0 && !slow_yield` (`:1478`); the careful
  loop carries `slow_knob::inject` (`:1638`). So the causal-perturbation harness
  is preserved (slow-knob ⇒ careful path runs). `CONTAINS_MARKERS` const-folds the
  whole block away on the marker `<true>` path. The function is compiled on both
  arches (`pure_inflate_decode` OR `x86_64+isal`, `:1316`), so the SAME code backs
  the dual-arch DUAL-SHA gate.

Empirical DUAL-SHA (arm64-native + x86-isal, same sha via `path=ParallelSM`)
corroborates the structural read. **UPHELD.**

---

## CLAIM 2 — NO STRIPPED SHORTCUT — **UPHELD**

The integration kept every overhead the bench elided.

- **(a) Writes go through `% U8_RING_SIZE`:** `dst_phys = pos % U8_RING_SIZE`
  recomputed every iteration (`:1491`); store at that physical slot (`:1516`).
- **(b) Back-refs go through the existing wrap-safe `emit_backref_ring_u8`**
  (`:1616` → defn `:2876`), which masks every index and has explicit
  wrap-straddle / RLE / overlap arms — the SAME routine the careful clean tail
  uses (`:1778`).
- **(c) Resumable `n_max_to_decode` cap honored:** capped to
  `ring_modulus - MAX_RUN_LENGTH` (`:1402`) and the fast-loop guard references it
  (`emitted + FAST_OUT_SLOP < n_max_to_decode`, `:1492`).
- **(d) Drain + CRC untouched:** the fast loop lives entirely inside
  `read_internal_compressed_specialized`; the `read()` wrapper's drain
  (`:1116-1134`, `drain_to_output` / `drain_transition_narrow_u16` /
  `flip_repack_to_u8`) is unchanged. CRC/ISIZE verification is downstream and
  un-edited.
- **"Interior-only, wrap→careful" is legitimate, not a disguised shortcut.** The
  fast loop is a *strict superset* of work over the careful loop on the same
  bytes: it carries `% U8_RING_SIZE`, the production `self.dist_hc`, the real
  `emit_backref_ring_u8`, and the `n_max` cap, and *adds* a speculative store on
  top. Handing the wrap tail (last 282 ring bytes) and the input tail (last 8
  bytes) to the careful loop is *more* conservative at the boundary, the opposite
  of dropping overhead for speed. This is faithful, not stripped. **UPHELD.**

By contrast the bench (`engine_isolation.rs:375-464`) IS the forbidden shape — a
flat `Vec<u8>` with prepended window, no ring modulo, `LutDistCode`, `Block` used
as header-parser only. The integration reproduces none of that optimism.

---

## CLAIM 3 — WALL / BINDER — **UPHELD-WITH-CAVEATS**

**TIE verdict is sound.** T1 1.018x within a 7% spread; T8 run1 1.035x / run2
0.966x is a sign flip ⇒ Δ < inter-run spread ⇒ TIE by the project's own Rule 5.
decodeBlock 1.227s→1.248s is within noise. Sound.

**The strongest attack — "the fast loop almost never fires, so the TIE proves
nothing" — FAILS on the source.** Firing fraction estimate:
- Per-call cap makes `n_max_to_decode = usize::MAX → 130814` (`:1402`,
  caller `gzip_chunk.rs:1117`), so the `emitted + 282 < n_max` guard holds until
  the last ~282 emitted bytes of a 128 KB call.
- The `dst_phys + 282 <= U8_RING_SIZE` wrap guard gates the loop off only in the
  last 282 of 131072 ring bytes (0.22%).
- The `bits.pos + 8 < in_end` guard gates off only the final 8 input bytes.
- Decoded chunks are ~4 MiB (`chunk_data.rs:88` `split = 4*1024*1024`). Each chunk
  starts marker-mode and flips after ~32–64 KB of consecutive clean output
  (`read()` seam `:1116-1119`), so the clean `<false>` fast loop processes
  **~98% of decoded bytes per chunk**.

So the fast loop fires heavily. Given that, "decodeBlock unchanged ⇒ the
speculative-store gain was absorbed by the real ring/wrap/dist overheads" is the
*supported* reading — and is independently explained by source: the integration
re-adds, versus the 0.555×-bench, (1) per-iteration `% U8_RING_SIZE`, (2) the
slower reversed-bits `self.dist_hc` in place of the bench's `LutDistCode`, and
(3) `emit_backref_ring_u8`'s branchy arms. The bench's flat-buffer 0.555× was
never reproducible on the real ring, exactly as the leader concluded. The §3
"decode stops binding at 0.410s" projection is therefore correctly **FALSIFIED**.

**CAVEATS (why "with caveats," not clean UPHELD):**
1. **No fired-fraction / clean-self-time counter.** The leader *inferred* firing
   from decodeBlock-unchanged; he did not instrument it. The wall measures TIME,
   and "fast loop runs on 98% of BYTES" is not the same as "the clean path is 98%
   of decodeBlock TIME" — marker-mode bootstrap is ~1.7× slower per byte (u16
   width, backward marker scan), so its *time* share exceeds its *byte* share.
   The inference is structurally strong but unproven. A fired-fraction counter
   plus a clean-vs-marker self-time split would convert "probably absorbed" into
   "proven absorbed." This is owed before the TIE is recorded as a refutation of
   VAR_V as a *direction* (vs. on *this run*), per Rule 7.
2. **Production `reset(None, window_opt)` discards the supplied window**
   (`gzip_chunk.rs:1107`; the seed only fires when `output` is `Some`,
   `marker_inflate.rs:590`). So *every* chunk pays a full marker-mode bootstrap
   before the fast loop engages, even when a window is available. That is a real,
   measurable structural cost orthogonal to VAR_V and arguably a larger
   rapidgzip-parity lever than the inner speculative store.

**UPHELD-WITH-CAVEATS.**

---

## OVERALL VERDICT

The integration is **faithful** (Claim 2 clean), **byte-exact** (Claim 1 clean,
DUAL-SHA-backed), and the **wall TIE is real** (Claim 3 sound). VAR_V's bench
0.555× did not survive contact with the production ring — correctly diagnosed.
The correct (byte-identical) change is keep-and-layer-able even on a TIE
([[feedback_layer_dont_revert_whole_system]] / Rule 7), but it is not a wall
lever as integrated.

**Single most important thing the leader may have gotten wrong:** he conflated
"the fast loop runs on the majority of *bytes*" with "the measurement is
sensitive to the fast loop's speed" **without instrumenting either** the
fired-fraction *or* the clean-vs-marker self-time split. Source makes "absorbed"
the most likely reading, but the TIE is currently an *inference*, not a *proof* —
and the un-instrumented marker-mode bootstrap (every chunk, because the window
seed is discarded at `gzip_chunk.rs:1107`) is the more likely place the real wall
gap lives.
