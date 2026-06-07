# Parity Divergence Map — gzippy parallel single-member ↔ rapidgzip

Read-only structural analysis. Goal = FAITHFUL TRANSLITERATION of rapidgzip's
runtime decode PATTERN, not perf levers. Each Fulcrum number below is a
BREADCRUMB pointing at a structural deviation located at file:line.

Sources read: Fulcrum report `/tmp/gzippy-locked-fulcrum-20260606-100735/fulcrum-report.txt`;
gzippy `marker_inflate.rs`, `gzip_chunk.rs`, `apply_window.rs`, `isal_huffman_pure.rs`,
`chunk_fetcher.rs`; vendor `gzip/deflate.hpp`, `chunkdecoding/GzipChunk.hpp`,
`huffman/HuffmanCodingISAL.hpp` (paths under
`vendor/rapidgzip/librapidarchive/src/rapidgzip/`).

## CORRECTION 2026-06-06 (orchestrator) — #A WAS INVERTED, IS ACTUALLY MATCHED

**The original #1 target (#A "two-engine clean-tail split is a divergence —
delete the handoff") IS WRONG, verified against vendor source on the
benchmarked LIBRAPIDARCHIVE_WITH_ISAL=ON build.** rapidgzip ALSO hands the
32 KiB-clean tail to a SECOND ISA-L engine:
`GzipChunk.hpp:520-525` — `if (cleanDataCount >= deflate::MAX_WINDOW_SIZE)
return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(...)`. The
in-place u16→u8 flip the map cited (`deflate.hpp:1289-1292`) is the NON-ISAL
fallback that the benchmark build does NOT execute. The SAME `#ifdef WITH_ISAL`
that selects `HuffmanCodingISAL` (which the map agrees is MATCHED, #E) also
selects this handoff — the map cannot accept one branch and reject the other.
Fulcrum corroborates: rapidgzip `worker.isal_stream_inflate` busy = 1321ms
(NON-ZERO). ⇒ gzippy's `FlipToClean → IsalInflateWrapper` handoff
(`gzip_chunk.rs:1191-1202` / `:773`) is a FAITHFUL PORT. Deleting it would
INTRODUCE divergence. **#A is re-marked MATCHED, citing `GzipChunk.hpp:432-526`.**
Section 1's #A row, sections 2-4 (which proposed deleting the handoff) are
SUPERSEDED by this correction and the clean-tail-engine-map verdict.

WHAT'S ACTUALLY LEFT (re-ranked): the wall-critical region is the clean-tail
ISA-L stream engine — gzippy `worker.isal_stream_inflate` busy 3039ms vs
rapidgzip 1321ms = 2.3×, +268ms wall-critical (the named T8 lever). The
window-absent marker bootstrap (`block_body`) is STARVED / wall-dead. The
Huffman inner loop (lit/len) is MATCHED. Genuine remaining divergences: #B
(distance coding, trivial swap), #C (window-map keying), #D (resolve per-link).

---

## Headline reframe (read this first — NOTE: its #A conclusion is REVERSED above)

The mandate pre-named d_w (window-absent decode, gzippy 124.7ms vs rapidgzip
67.9ms, 1.84×) as the primary target and asked *which HuffmanCoding variant*
gzippy picked for the marker decode. Answer after reading the code:

- **The lit/len coding is MATCHED.** Both use `HuffmanCodingISAL`
  (gzippy `isal_huffman_pure::IsalLitLenCodePure` ↔ vendor
  `deflate.hpp:175 using LiteralOrLengthHuffmanCoding = HuffmanCodingISAL`),
  TRIPLE-symbol packing and all. The marker decode is NOT slow because of a
  wrong Huffman class.
- **The decode is uniformly ~1.85× slow in BOTH modes** (d_c 86.2 vs 44.8 =
  1.92×; d_w 124.7 vs 67.9 = 1.84×). A wrong-marker-machinery hypothesis would
  inflate d_w only. A uniform slowdown points at the *decode architecture*, not
  marker bookkeeping.
- **The architecture divergence is the two-engine clean-tail split** — the
  literal "Divergence #2" already recorded in memory
  `project_faithful_unified_decoder_over_perf`. gzippy decodes the window-absent
  bootstrap with the pure-Rust `marker_inflate::Block` (span `worker.block_body`,
  1801ms) and then **hands the clean tail to a SECOND engine**, the ISA-L FFI
  streaming `IsalInflateWrapper` (span `worker.isal_stream_inflate`, 3039ms).
  rapidgzip stays in ONE `Block::read` that flips its window type u16→u8 in
  place on the same cursor (`worker.block_body` busy = **0** in the rapidgzip
  trace; everything is the single decode span).

So the #1 target shifts from "marker HuffmanCoding variant" to "**delete the
clean-tail engine handoff; continue the same `Block` in u8 mode**." This is also
exactly what the governing memory says to do first, regardless of speed.

---

## 1. Ranked divergence list (largest structural/wall impact first)

| # | gzippy file:line | vendor file:line | structural difference | verdict |
|---|---|---|---|---|
| A | `gzip_chunk.rs:1191-1202` (FlipToClean short-circuit) + `gzip_chunk.rs:773` (`finish_decode_chunk_with_inexact_offset`) + `gzip_chunk.rs:354-561` / `:379` (`IsalInflateWrapper`) | **`GzipChunk.hpp:432-526`** — `if (cleanDataCount >= deflate::MAX_WINDOW_SIZE) return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(...)` (`:520-525`) on the WITH_ISAL build (the `deflate.hpp:1289-1292` in-place flip is the NON-ISAL fallback the benchmark does NOT run) | vendor ALSO hands the 32 KiB-clean tail to a SECOND ISA-L engine on the benchmarked WITH_ISAL build; same `#ifdef WITH_ISAL` selects both `HuffmanCodingISAL` (#E MATCHED) and this handoff. Fulcrum: rapidgzip `isal_stream_inflate` busy = 1321ms (non-zero). gzippy's handoff is the SAME shape. | **MATCHED** (CORRECTED 2026-06-06; was inverted to DIVERGES). Residual is the engine's 2.3× busy gap — see clean-tail-engine-map. |
| B | `marker_inflate.rs:1199-1216` / `:1401-1404` (`isal_lut_dist_decode` → `isal_dist_pure`) | `deflate.hpp:336 using DistanceHuffmanCoding = HuffmanCodingReversedBitsCached` (ISA-L distance commented out at `:338`) | gzippy's PRODUCTION ISA-L path decodes distances with an ISA-L-derived table (`isal_huffman_pure` small table); vendor decodes distances with `HuffmanCodingReversedBitsCached`, having explicitly rejected ISA-L for distance. Note gzippy's *canonical fallback* (`marker_inflate.rs:1514`) already uses the matching `HuffmanCodingReversedBitsCached` — so the ported type exists; the production path just doesn't call it. | **DIVERGES** (fidelity; modest wall) |
| C | `chunk_fetcher.rs` window-map lookup at `run_decode_task` / `try_speculative_decode_candidate` (causal §2/§5 keys) | `GzipChunk.hpp:716-722` (rewrite chunk metadata encoded=seed, max=handoff) + `WindowMap` real-boundary keys | gzippy workers call `WindowMap::get(partition_seed)`; windows publish at the REAL boundary key, which a seed never equals → 97% of window-absent are KEY-MISMATCH. This is an architecture-layer keying divergence. It KEEPS the desired ~90% window-absent fraction (does not drift toward 31%), so it is faithful in *fraction* but divergent in *mechanism*. | **DIVERGES (mechanism)** / fraction MATCHED — out of inner-decode scope |
| D | `apply_window.rs:22-45` (`resolve_markers_u16`) dispatched on pool via `post_process.task` | `ChunkData.hpp:302 applyWindow` / `MarkerReplacement.hpp` | Placement MATCHED (both resolve on the worker pool — gzippy `post_process.apply_window` busy 866ms across 8 threads; rapidgzip 13415ms across 8 threads, both ~2.7–255ms wall-crit). But per-link L_resolve is 20.47ms vs 10.46ms (1.96×) and the data-model tax moves 1312 MiB vs 437 MiB fused-ideal. Likely a resolve-routine efficiency gap (`SegmentedU16::resolve_markers_u16` vs vendor's contiguous `MarkerReplacement`) and/or downstream of #A. | **DIVERGES (efficiency, secondary)**; structure MATCHED |
| E | `isal_huffman_pure.rs:116-121,243-738` (`IsalLitLenCodePure`, TRIPLE_SYM) | `HuffmanCodingISAL.hpp:71,95-173` | lit/len multi-symbol decode (short/long lookup, 3-symbol packing, length pre-expansion) is a faithful port. | **MATCHED** |
| F | `gzip_chunk.rs:309-561` (`finish_decode_chunk_impl` / `IsalInflateWrapper`) for KNOWN-window chunks | `GzipChunk.hpp:192-268 decodeChunkWithInflateWrapper` | known-window chunks use the inflate wrapper in BOTH tools — this use of a stream engine is faithful. (It is ONLY divergent when used to finish a *window-absent* chunk's clean tail — see #A.) | **MATCHED** |
| G | `marker_inflate.rs:979-985` (Block-internal flip on `distance_to_last_marker_byte`) | `deflate.hpp:1283-1287` | the Block's OWN u16→u8 flip predicate is a byte-for-byte port (`>= RING_SIZE` ‖ (`>= MAX_WINDOW_SIZE` ∧ `== decoded_bytes`)). The machinery to continue in u8 already exists (`marker_inflate.rs:1098-1102`, `drain_to_output` u8 path `:731-743`). | **MATCHED** (and it is what #A should hand control to) |

---

## 2. Two-column pattern map for the #1 target (#A)

The decode of a **window-absent** chunk.

```
VENDOR  decodeChunkWithRapidgzip (GzipChunk.hpp:414)        gzippy  decode_chunk_unified_marker (gzip_chunk.rs:724)
────────────────────────────────────────────────────       ──────────────────────────────────────────────────────────
block->setInitialWindow(initialWindow)   (:458)             marker_decode_step loop (gzip_chunk.rs:1184)
while (true)                              (:468)               loop {
  while (!block->eob())                   (:566)                 read_header(block,bits)        (gzip_chunk.rs:1207)
    block->read(bitReader, ...)                                  while !block.eob() {
      └─ Block::read (deflate.hpp:1192)                            read_body = block.read(...)  (gzip_chunk.rs:1238)
         if (m_containsMarkerBytes)        (deflate.hpp:1274)        └─ marker_inflate::Block::read (marker_inflate.rs:947)
            readInternal(.., m_window16)   ── u16 marker decode         read_internal_compressed_specialized::<true>  (:1099)
            if (distToLastMarker hits 32K) (deflate.hpp:1283)           internal flip contains_marker_bytes=false      (:984)
               setInitialWindow();         ◄── FLIP IN PLACE            drain push_clean_u8 → pending_clean            (:743)
               result.data = window (u8)                            }   ▲ Block CAN continue u8 — but driver intervenes
         else                              (deflate.hpp:1289)
            readInternal(.., window)       ── u8 CLEAN decode    ╔══════════════════════════════════════════════════════╗
            result.data = window           SAME ENGINE,          ║  ✗ DEVIATION POINT  gzip_chunk.rs:1191-1202          ║
                                           SAME cursor,          ║  if clean_appended_len() >= 32 KiB && !flipped {     ║
  ... loop continues to block end ...      SAME Huffman tables   ║      return MarkerStep::FlipToClean { end_bit }      ║
                                                                 ║  }                                                   ║
                                                                 ╚══════════════════════════════════════════════════════╝
                                                                   caller (gzip_chunk.rs:773):
                                                                   finish_decode_chunk_with_inexact_offset(...)
                                                                     └─ IsalInflateWrapper  (gzip_chunk.rs:379)  ◄── 2nd ENGINE
                                                                        ISA-L FFI stream, re-seed 32 KiB window,
                                                                        re-open inflate state, per-call read_stream loop
```

Circled deviation:
- **vendor circle:** `deflate.hpp:1289-1291` — the `else { readInternal(.., window) }`
  u8 branch is reached from the SAME `Block::read` inside the SAME
  `while(!block->eob())` loop (`GzipChunk.hpp:566`). No engine swap.
- **gzippy circle:** `gzip_chunk.rs:1191-1202` returns `FlipToClean` and
  `gzip_chunk.rs:773` calls into `IsalInflateWrapper` (`gzip_chunk.rs:379`).
  The pure-Rust `Block` has *already* flipped internally (`marker_inflate.rs:984`)
  and *could* produce the u8 tail itself (`marker_inflate.rs:1100-1102` +
  `:731-743`), but the driver short-circuits it out to the FFI stream engine.

---

## 3. Proposed faithful transliteration for #A (delete-divergent / create-match)

This is a **delete-divergent** change (the default per the bias guardrails).

1. **Delete** the driver-level flip short-circuit `gzip_chunk.rs:1191-1202`
   (`if output.clean_appended_len() >= MAX_WINDOW_SIZE && !ctx.flipped { return
   FlipToClean }`). The Block's OWN flip (`marker_inflate.rs:979-985`) is the
   faithful trigger and already fires (`flip_to_clean=29` in the trace).
2. **Delete** the `MarkerStep::FlipToClean` arm in `decode_chunk_unified_marker`
   (`gzip_chunk.rs:757-782`) that calls `finish_decode_chunk_with_inexact_offset`
   → `IsalInflateWrapper`. Let the `loop` (`gzip_chunk.rs:734`) keep calling
   `marker_decode_step`; after the internal flip the body runs
   `read_internal_compressed_specialized::<false>` (`marker_inflate.rs:1100`)
   and drains clean u8 straight into `chunk.data` via `push_clean_u8`
   (`UnifiedMarkerSink::push_clean_u8`, `gzip_chunk.rs:655` →
   `chunk.append_clean`, `gzip_chunk.rs:746`). This is the exact analog of vendor
   `deflate.hpp:1289-1291` `else { readInternal(.., window); result.data = window }`.
3. **Keep** `IsalInflateWrapper` ONLY for the genuinely-known-window path
   (`decode_chunk_with_inflate_wrapper`, `gzip_chunk.rs:309`) — that mirrors
   vendor `decodeChunkWithInflateWrapper` (`GzipChunk.hpp:192`) and is MATCHED (#F).
4. After #A lands, fold #B: swap the production distance decode to the already-
   ported `HuffmanCodingReversedBitsCached` (`marker_inflate.rs:1514` shows the
   call site pattern) to match `deflate.hpp:336`.

Cite for every step: vendor `deflate.hpp:1273-1292` + `GzipChunk.hpp:414-566`.
NO perf framing — this is correctness-of-shape. Per the governing memory, do it
even if the wall does not improve; reverting the clean tail to a 2nd engine "for
speed" IS the divergence.

---

## 4. Pre-registered FALSIFIER for #A

Capture a fresh locked-Fulcrum trace (same harness, T8, interleaved, sha-verified)
AFTER the transliteration. The divergence is **CLOSED** iff ALL hold:

1. **Span structure matches vendor:** `worker.isal_stream_inflate` busy drops to
   ~0 for window-absent chunks (it survives only for known-window chunks via #F);
   each window-absent chunk shows ONE decode span (`worker.block_body` / its
   rename), as in the rapidgzip trace (`block_body` ≠ 0, `isal_stream_inflate`
   not the dominant span).
2. **Byte-exact:** sha of decompressed output unchanged (1444635 / round-trips).
3. **Window-absent fraction stays faithful:** runtime window-absent ∈ [88%, 97%]
   (currently 90.5%); it MUST NOT drift toward the 31% static fraction.
4. **Clean-tail time converges toward vendor:** model d_c moves from 86.2ms
   toward rapidgzip's 44.8ms (target ≤ ~55ms). 

The transliteration is **FALSIFIED / incomplete** if, after removing the
handoff: two decode spans persist per window-absent chunk; OR sha diverges; OR
window-absent fraction collapses toward 31%. 

**Important split-test:** if span structure becomes single-engine AND sha holds
BUT d_c stays ~86ms (does not converge to ~45ms), then #A is structurally closed
but a SEPARATE inner-loop divergence in the pure-Rust `Block` u8 decode (ring
`% RING_SIZE` per-write, `emit_backref_ring`, distance coding #B) is the residual
— open a follow-up map on `read_internal_compressed_specialized::<false>`
(`marker_inflate.rs:1226`) vs vendor `readInternalCompressed`
(`deflate.hpp:1510-1666`). Do NOT re-introduce the FFI engine to recover d_c.

---

## 5. Honest reassessment of prior "matched" / "diverges" calls

- **Prior framing (Fulcrum §5 remediation):** "PRIMARY lever = KEY-MISMATCH
  window keying; STRUCTURAL = limit prefetch depth." → That is divergence **#C**
  and is real, but it is an *architecture/window-map* item and it KEEPS the good
  ~90% window-absent fraction. It is NOT the inner-decode centerpiece the d_w/d_c
  breadcrumbs point at. Re-ranked below #A.
- **Prior framing (Fulcrum model):** "LEVER = L_resolve / publish-chain binds."
  → #D. Structure (resolve-on-pool) is MATCHED; only per-link cost diverges.
  Treat as a secondary efficiency map, not a structural rewrite — consistent with
  the mandate's note that marker-resolution is "serial-in-both / faithful."
- **Confirmed MATCHED (against the memory's worry):** the lit/len HuffmanCoding
  variant (#E) and the Block-internal flip predicate (#G) ARE faithful ports —
  the worry was that gzippy picked a wrong marker Huffman class; it did not.
- **Confirmed DIVERGES (newly surfaced here, not previously ranked #1):** the
  distance coding class (#B) — production uses ISA-L distance where vendor uses
  `HuffmanCodingReversedBitsCached`. Small, but it is a genuine unlike-vendor
  choice and trivially fixable since the matching type is already ported.
- **The governing memory was right:** #A (two-engine clean tail) is the dominant
  structural divergence and the d_c/d_w uniform-1.85× breadcrumb corroborates it.
  **[CORRECTED 2026-06-06: #A is MATCHED, not divergent — see top-of-file
  correction. The dominant residual is the clean-tail ENGINE's 2.3× busy gap, not
  the existence of the handoff.]**

## #B distance-coding — confirmation (read-only)

**CONFIRMED.** The production ISA-L decode path decodes DISTANCE codes with an
ISA-L-derived LUT (`IsalDistCodePure`); vendor rapidgzip explicitly rejected
ISA-L for distance and uses `HuffmanCodingReversedBitsCached`. The matching type
is already ported (gzippy's canonical fallback uses it) but the production path
doesn't call it. The swap is small but **not** a literal one-line swap. Output
bytes unchanged.

1. **Vendor — CONFIRMED.** `deflate.hpp:333-338`: distance uses
   `HuffmanCodingReversedBitsCached` (`:336-337`); ISA-L distance commented out
   (`:338`) with benchmarked rationale that ISA-L is ~1-2% *slower* for the small
   distance table (`:333-334`).
2. **gzippy paths** (file is `src/decompress/parallel/marker_inflate.rs`):
   - **Production**: `read()` (`:947`) → `read_internal_compressed()` (`:960`) →
     on x86_64+isal/`pure_inflate_decode` (cfg `:1085-1092`) →
     `read_internal_compressed_specialized` → distance via
     `isal_lut_dist_decode(bits)` (`:1401-1405`) → `self.isal_dist_pure.decode(bits)`
     (`:1202`), field `isal_dist_pure: IsalDistCodePure` (`:349`; type
     `isal_huffman_pure.rs:1071`, decode `:1158` returns `Option<(sym,bit_count)>`).
     **DIVERGES.**
   - **Fallback** (`read_internal_compressed_canonical`, only NOT-(isal/pure) cfg
     `:1995-2009`): builds `HuffmanCodingReversedBitsCached<MAX_DISTANCE_SYMBOL_COUNT>`
     (`:1514-1516`), decodes at `:1580`. **Matches vendor** — compiled out of the
     production build.
3. **Swap NOT literally trivial** (matching type ported + `Bits: LsbBitReader`
   exists at `huffman_base.rs:149`, no new type needed), but decode contracts
   differ: `IsalDistCodePure::decode` is non-consuming returning `(sym,bits)` with
   caller `bits.consume(dbit)` (`:1405`); `HuffmanCodingReversedBitsCached::decode`
   (`huffman_reversed_bits_cached.rs:91`) consumes internally, returns only
   `Symbol`. Required: (1) add a `dist_hc` field + per-block
   `initialize_from_lengths(&dist_lens,false)` build (mirroring `:1516`) in place
   of `isal_lut_dist_rebuild` (`:1142-1145`); (2) at `:1401-1405` call
   `dist_hc.decode(bits)` and delete the `bits.consume(dbit)` line.
4. **Correctness — output unchanged (low risk):** both decode the same canonical
   distance Huffman from the same `dist_lens` into the same
   `DISTANCE_BASE`/`DISTANCE_EXTRA` (`:1406-1417` vs `:1584-1587`); difference is
   table representation/speed only. The cached variant already passes the corpus
   differential via the fallback. Residual risk is `rebuild_from` vs
   `initialize_from_lengths` edge parity — ship with the silesia differential.
