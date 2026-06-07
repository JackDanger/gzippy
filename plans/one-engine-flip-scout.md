# One-Engine Flip-in-Place Scout (read-only)

Charts the path from gzippy's current TWO-engine clean tail to the faithful
ONE-engine flip-in-place (governing memory), and rigorously tests the user's
performance hypothesis. All claims cite `file:line`. NO builds/benches were run.

Frozen inputs (parent-supplied, Final Fulcrum T8 silesia):
- wall: gzippy 841ms vs rapidgzip 461ms = **1.83×**
- d_c (clean decode/chunk): 85.5 vs 44.1 = **1.94×**
- d_w (window-absent decode/chunk): 122.5 vs 66.7 = **1.84×**
- Artifacts: `/tmp/gzippy-parity-final/`, `/tmp/gzippy-locked-fulcrum-20260606-100735/`.

---

## PART A — THE HYPOTHESIS

**User's hypothesis:** "the one-engine flip-in-place (pure-Rust) will perform about
the SAME, in runtime, as rapidgzip's hand-off-to-ISA-L."

### A.0 The decisive structural fact the hypothesis rests on is FALSE

"rapidgzip's hand-off-to-ISA-L" is **not** a one-engine path — it is itself a
TWO-engine handoff, structurally identical in shape to gzippy's current path:

- `GzipChunk.hpp:521-524` — when ISA-L is compiled in
  (`#ifdef LIBRAPIDARCHIVE_WITH_ISAL`), at `cleanDataCount >= MAX_WINDOW_SIZE`
  rapidgzip does `return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(…)`
  — it ABANDONS `deflate::Block` and hands the clean tail to **ISA-L**. Two
  engines: `deflate::Block` (marker phase) → `IsalInflateWrapper` (ISA-L, clean
  tail). This is the SAME two-phase shape as gzippy.
- The TRUE one-engine flip-in-place is rapidgzip's **non-ISA-L build**: that whole
  `cleanDataCount` handoff block is `#ifdef`'d out (`GzipChunk.hpp:520-525`), so the
  SAME `block` (`GzipChunk.hpp:456`) stays in the `while(true)` loop and decodes the
  clean tail in place. The flip is `Block::setInitialWindow` flipping
  `m_containsMarkerBytes=false` (`deflate.hpp:1759`/`1785`), after which `Block::read`
  takes the u8 `result.data = lastBuffers(window,…)` branch (`deflate.hpp:1289-1292`)
  on the same `m_window`/`bitReader`.
- rapidgzip itself treats the non-ISA-L (pure-Block) clean tail as the **slow
  fallback** — see the `@todo` at `GzipChunk.hpp:432-440` lamenting that without
  ISA-L they cannot even use zlib-ng for the clean tail.

So the comparison the user proposes is: gzippy's pure-Rust one-engine flip (= the
vendor NON-ISA-L slow path) vs rapidgzip's ISA-L two-engine fast path. These are
different engines for the clean tail.

### A.1 Decompose d_c — and note the flip does not touch it

gzippy's d_c chunks are **window-seeded** (the window arrives from the `WindowMap`).
`decode_chunk_with_rapidgzip_impl` (`gzip_chunk.rs:585`) takes the
`initial_window.len() == MAX_WINDOW_SIZE` branch (`gzip_chunk.rs:598-630`) →
`finish_decode_chunk_with_inexact_offset` directly. **No marker phase, no flip.**
The flip-in-place change is on the window-ABSENT path only — it cannot move d_c.

d_c is therefore PURE clean-engine cost:
- (i) per-symbol decode: `IsalInflateWrapper` = `Inflate<Clean, Generic, Streaming>`
  (`inflate_wrapper.rs:154-159`) — pure-Rust, **Generic (BMI2 OFF)**. The `X86_64Bmi2`
  ArchProfile is a wired-out placeholder (`unified.rs:107`, `130`; only
  `Inflate<Clean, Generic, Streaming>` is implemented, `unified.rs:200-225`).
  rapidgzip's d_c = real ISA-L.
- (ii) handoff: ≈0 in d_c (window set once, no marker→clean transition).
- (iii) resumable-streaming tax: present, but rapidgzip's
  `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>` is the SAME outer/inner
  resumable streaming loop, so this tax is structurally MATCHED on both sides.

⇒ The d_c ratio **1.94× is essentially pure per-symbol generic-pure-Rust ÷ ISA-L**.

### A.2 Decompose d_w — what the flip actually removes

- rapidgzip: d_w − d_c = 66.7 − 44.1 = **+22.6ms** = its marker bootstrap +
  `getLastWindow` materialize + ISA-L re-init. The clean-tail PORTION of d_w runs at
  the ISA-L d_c rate.
- gzippy: d_w − d_c = 122.5 − 85.5 = **+37.0ms** = its marker bootstrap +
  `last_32kib_window_vec` materialize (`gzip_chunk.rs:767` → `chunk_data.rs:527-565`)
  + `IsalInflateWrapper::with_until_bits` alloc (`gzip_chunk.rs:379`) + `set_window`
  32 KiB copy (`gzip_chunk.rs:380`). Clean tail runs at the generic d_c rate.

The flip-in-place removes ONLY the gzippy-side handoff (ii)+(iii):
- delete `last_32kib_window_vec()` materialize (narrow+copy 32 KiB from
  `data_with_markers`),
- delete the fresh wrapper alloc + 32 KiB `set_window` copy,
- delete the re-seek/re-prime of a cold `Bits` at `end_bit`.

Magnitude bound: the gzippy-EXTRA marker+handoff vs rapidgzip is 37.0 − 22.6 =
**14.4ms total** across all d_w chunks, and the handoff is only a fraction of that
(the bulk is marker-mode per-symbol decode + backward-scan overhead, which the flip
does NOT remove). The handoff is one-time-per-window-absent-chunk
(materialize + memcpy + alloc) — μs × chunk count. Realistic removal: **a few ms**.

The flip does NOT change clean-tail engine class: it switches the clean tail from
`Inflate<Clean,Generic,Streaming>` (wrapper) to
`Block::read_internal_compressed_specialized::<false>` (`marker_inflate.rs:1191`,
drain via `push_clean_u8` `marker_inflate.rs:738-750`) — a DIFFERENT pure-Rust loop,
**still not ISA-L**, possibly faster OR slower than the wrapper.

### A.3 VERDICT — the PARENT is right; the user's hypothesis is FALSE as stated

The ~1.85× gap is **uniform** across d_c and d_w because the dominant cost on both
is the clean-tail per-symbol decode, and gzippy's clean tail is pure-Rust-Generic
while rapidgzip's is ISA-L. The one-engine flip keeps the clean tail pure-Rust; it
removes only the handoff. It cannot close a per-symbol-engine gap.

Steelman partial vindication for the user: the flip IS the faithful STRUCTURE (it is
exactly what vendor's non-ISA-L build does), and it DOES remove real handoff work.
But "same RUNTIME as rapidgzip's ISA-L" requires an ISA-L-class clean decoder, which
the flip does not provide. The user is likely conflating "rapidgzip's ISA-L handoff"
(a 2-engine fast path) with "one engine"; the genuine one-engine analog is vendor's
SLOW fallback.

### A.4 Falsifiable predictions

1. **d_c: UNCHANGED ≈ 85.5ms** post-flip. (Window-seeded chunks never flip; the
   flip path is window-absent only.) This is a sharper claim than the parent's
   "~81ms" — the flip literally does not execute on d_c chunks.
2. **d_w: drops by handoff-only**, 122.5 → ~118-121ms (a few ms) IF
   `Block::<false>` clean decode ≈ the wrapper. If `Block::<false>` is SLOWER than
   the wrapper, d_w is FLAT or REGRESSES (the two-engine design may exist precisely
   because the wrapper's clean loop is the more-tuned one). Clean tail stays pure-Rust.
3. **wall: 841 → ~825-835ms best case, still ≈1.8×. NOT parity.**
4. **Settling measurement (the one that decides):** a dev-oracle that swaps gzippy's
   clean decode (both the d_c wrapper AND the post-flip tail) to real ISA-L, kept
   off the production graph. Prediction: closes to ~1.0× ⇒ proves engine (i)
   dominates and the flip/handoff is noise. The production-legal twin of this test
   is wiring the `X86_64Bmi2` ArchProfile (`unified.rs:130`) and re-measuring d_c.
   If THAT closes the gap, the real lever is the BMI2 clean loop, not the flip.

**Bottom line:** the flip is correct-and-faithful and worth doing for fidelity (and
a few ms), but it is NOT the wall lever. The wall lever is an ISA-L-class pure-Rust
clean decoder (BMI2 path), orthogonal to the flip.

---

## PART B — THE COMPLETE PATH TO LIVE, BYTE-EXACT ONE-ENGINE FLIP

### B.1 Responsibilities of the current handoff and where the one-engine path covers them

Current handoff = `finish_decode_chunk_with_inexact_offset` (`gzip_chunk.rs:334`) →
`finish_decode_chunk_impl` (`gzip_chunk.rs:353-561`), invoked from the `FlipToClean`
arm (`gzip_chunk.rs:757-782`).

| # | Responsibility | Two-engine site | One-engine coverage | Gap? |
|---|---|---|---|---|
| a | exact end-offset / until-bit | `read_cap`/`set_coalesce_stop_hint` `gzip_chunk.rs:374-388` | marker loop stop `gzip_chunk.rs:1222` | **GAP** — missing FIXED_HUFFMAN exception + exact `==stop_hint` case (vendor `GzipChunk.hpp:545-549`) |
| b | stop-bit / EOB detection | `StoppingPoints` `gzip_chunk.rs:381-385` | `while !block.eob()` `:1237`; `is_last_block` `:1293` | covered (single-member); inter-member gzip-header re-read is out of scope (loop is single-member, `gzip_chunk.rs:977-979`) |
| c | back-refs into PRE-flip (marker) window | copies `last_32kib_window_vec` into fresh wrapper `gzip_chunk.rs:767,380` | Block ring already holds last 32 KiB, internally flipped clean (`set_initial_window` `marker_inflate.rs:640-716`, `emit_backref_ring`) | NO gap — strictly safer (no copy/narrow) |
| d | subchunk boundary emission | `append_block_boundary_at` inner loop `gzip_chunk.rs:429-499` | `note_block_boundary(end_bit, sink_len())` `:1302` → `UnifiedMarkerSink` `:701-703` → `apply_recorded_block_boundaries` `:707-717,750` | **GAP (offset base)** — must keep ONE decoded-offset convention across the flip; guarded by `UNSPLIT_BLOCKS_EMPLACED` trap (`tests/routing.rs:725`) |
| e | inexact-offset resolution (offset is a guess) | decode-until-first-boundary-≥-hint via coalesce `gzip_chunk.rs:468-497` | `next_block_offset >= stop_hint_bits` `:1222` + `finalize_with_deflate(end_bit)` `:1293-1300`/`chunk_data.rs:1380`; consumer reconciles via `matches_encoded_offset` `chunk_data.rs:466` | covered IFF gap (a) closed |
| f | CRC / size accounting | `crc32s.last_mut().update` + `data.commit` + `non_marker_count` `gzip_chunk.rs:511-520` | `push_clean_u8` → `append_clean` `chunk_data.rs:706-717` (CRC + non_marker + subchunk size) | NO gap |
| g | writev / output sink | both write `chunk.data` (`SegmentedU8`); consumer `fd_vectored_write` drains | identical | NO gap |

Net: f, c, g are already done; b is done for single-member; **the real gaps are (a)
stop-hint parity and (d) subchunk-offset-base consistency**. Plus the open
**performance question**: `Block::<false>` clean loop vs the wrapper's clean loop.

### B.2 Concrete change set (file:line)

1. **`gzip_chunk.rs:757-782` — `MarkerStep::FlipToClean` arm.** Keep the counter
   (`FLIP_TO_CLEAN_CHUNKS`) + trace; DELETE the `last_32kib_window_vec()` materialize
   (`:767`) and the `finish_decode_chunk_with_inexact_offset(...) + return Ok(chunk)`
   (`:773-781`). Replace with: fall through to the same handling as
   `MarkerStep::Continue` (i.e. `continue` the `loop`). The thread-local `Block`
   (`gzip_chunk.rs:1091-1093`) and `MarkerDecodeCtx` persist; `ctx.current_bit_offset`
   already equals `end_bit_offset` (`:1194`), `ctx.flipped=true` prevents re-trigger,
   and the Block is already internally clean (it pushed 32 KiB u8 via `push_clean_u8`
   before `clean_appended_len()>=MAX_WINDOW_SIZE` could fire `:1191`). The clean tail
   then decodes in place via `UnifiedMarkerSink.push_clean_u8` → `pending_clean` →
   `chunk.append_clean` (`gzip_chunk.rs:655-659,744-749`).

2. **`gzip_chunk.rs:1222` — marker loop stop condition.** Upgrade to vendor parity
   (`GzipChunk.hpp:545-549`):
   `(next_block_offset >= stop_hint_bits && !block.is_last_block() &&
   block.compression_type() != FixedHuffman) || next_block_offset == stop_hint_bits`.
   Requires exposing compression-type on the `BootstrapEngine` trait
   (`gzip_chunk.rs:1121-1151`): `Block::compression_type()` exists
   (`marker_inflate.rs:463`, `CompressionType::FixedHuffman` `:164`); `MarkerRing`
   (legacy `GZIPPY_MARKER_RING=1`) needs an equivalent accessor or that env path must
   be excluded from the new condition.

3. **Sink choice.** KEEP `UnifiedMarkerSink` for the whole decode (it already routes
   `push_clean_u8` and uses ONE `sink_len()`-based boundary convention `:669,1302`).
   This avoids the offset-base divergence in gap (d): the dead `CleanTailSink`
   (`gzip_chunk.rs:812-862`) uses a *clean-tail-relative* boundary offset
   (`:849-860`) — a SECOND convention that would desync subchunk sizes across the
   flip. DELETE `CleanTailSink` (`:806-862`) per the no-dead-code rule once the flip
   is wired through `UnifiedMarkerSink`.

4. **DO NOT delete** `finish_decode_chunk_impl` / `finish_decode_chunk_with_inexact_offset`
   — still the production clean engine for window-SEEDED (d_c) chunks
   (`gzip_chunk.rs:598-630`, `:318`) and `decode_chunk_with_inflate_wrapper`. Only the
   `FlipToClean` call site changes.

### B.3 Correctness traps + gating

- **Byte-exact gate:** silesia sha + `diverged=0` via `src/tests/three_oracle_diff.rs`
  (flate2 + libdeflate oracles) AND `scripts/measure.sh` sha-verify. Real-corpus
  differential ships in the same commit (memory: `feedback_real_corpus_test_with_lever`).
- **Wiring guards:** `tests::routing::test_single_member_routing_multithread` (T4,
  24 MiB, byte-perfect + adversarial chunks) and
  `test_single_member_parallel_not_slower_than_sequential`. Keep
  `UNSPLIT_BLOCKS_EMPLACED` deletion trap green (`tests/routing.rs:725`) — it locks
  subchunk emission (gap d).
- **Load-flaky:** `decompress::parallel` tests deadlock on a loaded box (memory
  `project_parallel_test_hang`). Run on a quiet box, single `cargo` invocation, wrap
  in `timeout`.
- **Marker-leak trap:** `Block::<false>` clean drain must never emit a value ≥256 —
  `debug_assert` at `marker_inflate.rs:744` and `chunk_data.rs:739`; in release a
  stray marker corrupts and the CRC check catches it (terminal Err, no silent
  fallback, CLAUDE rule 5).
- **Stop-hint mismatch trap:** if (a) is wrong, the resolved end disagrees with the
  block finder → chunk gap/overlap → consumer `matches_encoded_offset` rejects →
  error or re-decode cliff. Isolate (a) in its own phase.
- **`GZIPPY_MARKER_RING=1`** legacy path must keep compiling with the new trait
  method (add `compression_type` to its `BootstrapEngine` impl or gate it out).

### B.4 Phasing (each phase independently byte-exact-gated + measured)

- **Phase 0 (measure, no behavior change).** Use existing
  `BOOTSTRAP_POST_FLIP_U16_BYTES` (`gzip_chunk.rs:1284`) + a new counter for
  post-flip clean-tail bytes to bound how much of d_w is the post-flip tail. CRUCIAL:
  micro-measure `Block::read::<false>` clean loop vs `Inflate<Clean,Generic,Streaming>`
  on the SAME post-flip input. This decides whether Phase 2 helps or regresses (the
  biggest risk). If the wrapper wins, do NOT do the naive flip — instead make the
  handoff cheap (hand the Block's ring directly, skip the 32 KiB materialize) or port
  the wrapper's clean inner loop into `Block::<false>`.
- **Phase 1 (stop-condition parity).** Change `gzip_chunk.rs:1222` + add trait
  accessor. Behavior-neutral for the marker phase (the wrapper still owns the clean
  tail). Gate: full sha + routing tests. Independently shippable.
- **Phase 2 (the flip).** Change the `FlipToClean` arm (`gzip_chunk.rs:757-782`) to
  `continue`, keep `UnifiedMarkerSink`. Gate: silesia sha-exact + three_oracle +
  routing + UNSPLIT trap. Measure d_w/wall on neurotic.
- **Phase 3 (cleanup).** Delete dead `CleanTailSink` (`gzip_chunk.rs:806-862`). Keep
  `finish_decode_chunk_impl` (window-seeded path needs it).
- **Phase 4 (the REAL wall lever, separate program).** Wire the `X86_64Bmi2`
  ArchProfile (`unified.rs:130`, currently placeholder) for the clean decode — both
  the d_c wrapper and the post-flip `Block::<false>` tail — or port an ISA-L-class
  pure-Rust clean loop. This is what closes the ~1.85×; Phases 1-3 do not.

### B.5 Risks (ranked)

1. **`Block::<false>` clean tail slower than the wrapper ⇒ Phase 2 REGRESSES d_w.**
   The two-engine split may exist because the wrapper's clean loop is the tuned one.
   Mitigation: Phase 0 micro-bench gates Phase 2; fall back to cheap-handoff or
   inner-loop port.
2. **Subchunk offset-base divergence across the flip ⇒ seekable-index corruption**
   (silent if the index is unused by the bench). Mitigation: single `UnifiedMarkerSink`
   convention + `UNSPLIT_BLOCKS_EMPLACED` trap + a `decoded_size()` invariant assert.
3. **Stop-hint off-by-one vs block finder ⇒ chunk rejection/re-decode cliff.**
   Mitigation: Phase 1 isolation.
4. **OOM (host just OOM'd).** Serialize builds (no parallel `cargo`), prefer neurotic
   for benches, no parallel agents, wrap tests in `timeout`, `df -h` before builds.

### B.6 Existence-proof check

- **Vendor non-ISA-L IS a true flip-in-place-and-continue:** `Block::setInitialWindow`
  flips `m_containsMarkerBytes=false` in place on the same object
  (`deflate.hpp:1739-1785`, esp. `:1759`/`:1785`); `Block::read` then takes the u8
  `else { … result.data = lastBuffers(window,…) }` branch on the same
  `m_window`/`bitReader` (`deflate.hpp:1289-1292`); and `GzipChunk.hpp:520-525`'s
  clean-tail handoff is `#ifdef LIBRAPIDARCHIVE_WITH_ISAL`, so without ISA-L the same
  `block` (`GzipChunk.hpp:456`) stays in the `while(true)` loop decoding the clean
  tail. CONFIRMED.
- **gzippy's `Block` is a faithful structural analog:** `set_initial_window`
  (`marker_inflate.rs:640-716`, flips `contains_marker_bytes` `:714`),
  `read_internal_compressed_specialized::<false>` (`:1191`) = u8 clean path,
  `drain_to_output` routes clean u8 via `push_clean_u8` (`:738-750`) = the
  `deflate.hpp:1285-1292` `result.data` path, and the internal flip predicate
  (`marker_inflate.rs:1000-1005`) mirrors `deflate.hpp:1282-1284` (item G already
  matched). gzippy's `Block` IS the analog — it is merely NOT WIRED to continue: the
  `gzip_chunk.rs:757` `FlipToClean` handoff intercepts it. **The one-engine change is
  removing that interception.**

---

## Summary

- **Hypothesis ruling:** FALSE as stated; parent correct. The ~1.85× is the
  pure-Rust-Generic ÷ ISA-L per-symbol clean-decode ratio, uniform on d_c and d_w
  because the clean tail dominates both. The flip keeps the clean tail pure-Rust
  (just swaps wrapper → `Block::<false>` and removes the handoff) so it cannot close
  that gap; it buys a few ms of handoff at best (and risks a regression if
  `Block::<false>` < wrapper). The real wall lever is an ISA-L-class pure-Rust clean
  decoder (the unwired `X86_64Bmi2` profile), orthogonal to the flip.
- **Biggest path risk:** `Block::read::<false>` clean tail being SLOWER than the
  existing `Inflate<Clean,Generic,Streaming>` wrapper, which would make the flip
  regress d_w — gate with a Phase-0 micro-bench before committing the flip.
- **Phase count:** 5 (Phase 0 measure → 1 stop-parity → 2 flip → 3 cleanup → 4 the
  separate BMI2 wall lever).
