# RG-MARKER COMPLETING GATE — INDEPENDENT DISPROOF ADVISOR VERDICT (READ-ONLY)

Adversarial Opus disproof of DIS-19 (the marker-engine COMPLETING attribution +
CLAIM A reversal + CLAIM B lever). Every load-bearing vendor claim re-derived
first-hand from `vendor/rapidgzip/` source, not from result prose. Posture: break
the claim. HEAD per the ledger (d56cb0f5 / origin 7bf26096), branch reimplement-isa-l.

---

## ONE-LINE VERDICT

**CLAIM A (the bombshell reversal) is REFUTED by vendor source. rapidgzip flips
the CLEAN TAIL of every streaming chunk OUT to a SEPARATE real ISA-L call
(`isal_inflate`) at exactly 32 KiB of clean output — byte-for-byte the same
structure as gzippy's `flip_to_clean`. gzippy's flip-to-clean is a CONVERGENCE
with rapidgzip, NOT a divergence. DIS-19's own "shared igzip kernel rg 1.68e9"
IS that clean-tail flip and silently CONFIRMS the convergence its narrative
denies. The "0 samples for `decodeChunkWithInflateWrapper`" is a MISREAD of the
WRONG symbol (the top-level index/seek-only entry); the streaming clean tail runs
through the SIBLING `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`.
CLAIM B (de-frag the u16 output) is a FAITHFUL, architectural, correctly-bucketed
lever — but its WALL payoff is UNPROVEN (instruction-count is a hypothesis
generator, not a verdict; page-warmth already sized the u16-footprint axis as
wall-slack). Do NOT delete flip-to-clean; do NOT port the de-frag off the split —
size it with a causal perturbation first.**

---

## CLAIM A — "rg marker-decodes EVERY streaming chunk via Block::read with
HuffmanCodingISAL-inside, then resolves via applyWindow; gzippy's flip-to-clean to
a separate real ISA-L FFI is a DIVERGENCE." → **REFUTED (vendor source).**

Sub-facts checked first-hand:

**(a) Does Block::read use HuffmanCodingISAL as its per-symbol lit/len primitive in
marker mode? → YES, confirmed.**
- `deflate.hpp:174-175`: `#ifdef LIBRAPIDARCHIVE_WITH_ISAL` → `using
  LiteralOrLengthHuffmanCoding = HuffmanCodingISAL;`. The symboled 0.16.0 .so links
  the igzip kernels (campaign STEP-0), so this branch is live.
- `deflate.hpp:922` `LiteralOrLengthHuffmanCoding m_literalHC;` is the Block member;
  `readInternal` (`:1455/1462`) dispatches to `readInternalCompressed(..., m_literalHC)`.
  The Huffman decode is window-INDEPENDENT — marker-ness only changes how
  back-references are EMITTED (markers vs resolved), not which Huffman primitive
  runs. So the marker PREFIX of each window-absent chunk genuinely decodes lit/len
  symbols via `HuffmanCodingISAL::decode` (a per-symbol C++ LUT decode built by ISA-L's
  `make_inflate_huff_code_lit_len`, `HuffmanCodingISAL.hpp:70,95`). This sub-fact is SOUND.
- NOTE the template `Block<ENABLE_STATISTICS>` parameter is STATISTICS, not marker-mode.
  There is no `Block<false>` "marker monomorphization" — marker vs clean is the
  `Window::value_type` (uint16_t vs uint8_t via `m_window16`), not a type parameter.
  DIS-19's "Block<false>::read (marker mode)" phrasing is imprecise; harmless here.

**(b) Is the u8-direct path gated on index/seek only? → YES for the TOP-LEVEL entry,
but that is NOT the path rg uses for the clean tail.**
- `GzipChunk.hpp:671` `if ( initialWindow && untilOffsetIsExact )` → the ONLY caller
  of `decodeChunkWithInflateWrapper` (the function DIS-19 saw 0 samples for). That IS
  the resolved-window + exact-offset (random-access/index) case. Streaming
  speculative chunks fall through to `decodeChunkWithRapidgzip` (`:709/:719`). SOUND.

**(c) THE REFUTATION — rg flips the clean tail OUT to real ISA-L, exactly like gzippy:**
- Inside `decodeChunkWithRapidgzip`'s decode loop, `GzipChunk.hpp:576-577` accumulates
  `cleanDataCount += bufferViews.dataSize()` (clean, non-marker bytes), and
  **`GzipChunk.hpp:520-525`:**
  ```
  #ifdef LIBRAPIDARCHIVE_WITH_ISAL
      if ( cleanDataCount >= deflate::MAX_WINDOW_SIZE ) {
          return finishDecodeChunkWithInexactOffset<IsalInflateWrapper>(
              bitReader, untilOffset, result.getLastWindow( {} ), ... );
      }
  #endif
  ```
- `finishDecodeChunkWithInexactOffset` (`GzipChunk.hpp:282`) constructs an
  `IsalInflateWrapper` and drives `inflateWrapper.readStream(...)` →
  **`isal.hpp:302` `isal_inflate( &m_stream )`** — the REAL ISA-L igzip NASM block
  kernel (`igzip_decode_block_stateless_*`). This is a SEPARATE real-ISA-L call on the
  clean tail, dispatched from a SIBLING of `Block::read` (not from inside it).
- This is gzippy's `flip_to_clean` (FlipToClean at 32 KiB → `finish_decode_chunk_isal_oracle`
  → `decompress_deflate_from_bit_into`, DIS-12/13) BYTE-FOR-BYTE in structure: marker-decode
  the ≤32 KiB window-absent prefix with the per-symbol Huffman primitive, then RETURN into a
  real-ISA-L whole-block decoder for the clean continuation.

**Therefore gzippy's flip-to-clean is CONVERGENT with rapidgzip, not divergent.**
CLAIM A's load-bearing inference is false. What is TRUE and survives: (i) rg never
takes the *top-level* `decodeChunkWithInflateWrapper` in streaming (index-only); (ii)
rg uses `HuffmanCodingISAL` per-symbol for the marker PREFIX. CLAIM A erred by
reading "0 samples for `decodeChunkWithInflateWrapper`" as "rg never uses the
u8-direct igzip kernel," when rg reaches that kernel through a DIFFERENTLY-NAMED
function (`finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`).

**Two independent internal contradictions confirm the refutation:**
1. DIS-19's OWN split credits rg a "shared igzip kernel 1.68e9." If CLAIM A were
   true (rg never u8-direct-decodes streaming), that 1.68e9 could not exist. It IS the
   clean-tail flip to `isal_inflate`. DIS-19's NUMBERS support convergence; only its
   NARRATIVE denies it.
2. The campaign's OWN prior banked verdict (`marker-attribution-gate-verdict.md`
   CLAIM 4) already established `GzipChunk.hpp:521 cleanDataCount >= MAX_WINDOW_SIZE ↔
   gzippy FlipToClean`. DIS-19 CLAIM A is a REGRESSION that re-asserts a divergence the
   campaign already refuted.

Verdict: **REFUTED.**

---

## THE SPLIT (71/25/shared) — **mostly SOUND; one FIX-NEEDED footnote.**

- The 71/25/~0 partition is structurally coherent and the "shared igzip kernel ≈
  equal (gz 1.88e9 ≈ rg 1.68e9)" leg actively CORROBORATES convergence: both reach
  `isal_inflate` for the clean tail, both flip at 32 KiB, so the clean byte fraction
  matches → kernel instructions match. Good.
- **Double-count check (the prompt's Q2): NO double-count between `Block::read` and
  the kernel line.** `Block::read`'s inline tree contains `HuffmanCodingISAL::decode`
  (per-symbol C++ LUT) — NOT `isal_inflate`. The NASM block kernel is reached via
  `finishDecodeChunkWithInexactOffset` (a SIBLING of `Block::read`, called from the
  `decodeChunkWithRapidgzip` loop at `:522`, not a callee of `read`). So rg's 41.75%
  `Block::read` (marker prefix) and the 1.68e9 kernel (clean tail) are disjoint code.
- **FIX-NEEDED footnote:** verify the symbol behind "shared igzip kernel rg 1.68e9"
  is specifically `isal_inflate` / `igzip_decode_block_stateless_*` (NASM, clean tail),
  NOT `HuffmanCodingISAL::decode` (C++ LUT, which lives INSIDE `Block::read`'s
  marker-prefix tree). Both are "ISA-L code"; a too-coarse "any isal symbol" bucket
  would fold the marker-prefix LUT decode into the clean-tail kernel line and
  contaminate the 71/25 split. ±2-3% self-basis precision is fine for a 71/25 verdict
  but NOT for finer sub-bucketing.

Verdict: **SOUND (partition) / FIX-NEEDED (confirm the kernel-line symbol is
isal_inflate, not HuffmanCodingISAL::decode).**

---

## CLAIM B — "~1.70e9 is u16 output/backref fragmentation (SegmentedU16 push_slice
0.706e9 + emit_backref_ring u16-ring 0.993e9); rg fuses inlined flat
m_window16[pos++] writes near-free; largest faithful tractable lever." →
**SOUND as a faithful + architectural lever; FIX-NEEDED on the unproven WALL claim.**

Vendor existence proof (the prompt's Q3), first-hand:
- `deflate.hpp:926` `alignas(64) PreDecodedBuffer m_window16{...}` — ONE FLAT aligned
  16-bit buffer for the marker phase.
- `appendToWindow`/`appendToWindowUnsafe` (`deflate.hpp:1319-1322,1341-1343`): a flat
  indexed write `window[m_windowPosition++] = decodedSymbol` (+ a cheap
  marker-distance branch). This is literally CLAIM B's "flat m_window16[pos++]."
- `resolveBackreference` (`deflate.hpp:1374-1390`): the common case is a SINGLE
  `std::memcpy(&window[m_windowPosition], &window[offset], length*sizeof(u16))` in the
  same flat buffer (the `m_windowPosition + length < window.size()` fast path skips
  modulo entirely), plus a short marker-scan. Only overlapping/wrapping copies fall to
  per-symbol `appendToWindowUnsafe`.
- Width flip is IN-PLACE: `replaceMarkerBytes` over `m_window16` (`deflate.hpp:1765`),
  then narrows to the u8 window.

So rg's marker output is one flat aligned u16 buffer with memcpy back-refs and an
in-place u16→u8 narrowing. gzippy's `SegmentedU16::push_slice` (segmented buffer) +
`emit_backref_ring` (separate u16 ring with per-symbol modulo wrap) is the
DIVERGENCE. Flattening toward `m_window16` is:
- **FAITHFUL** — `m_window16` is the vendor existence proof. ✓
- **Aligned with the GOVERNING memory** (`project_faithful_unified_decoder`): "ONE
  MarkerRing that flips u16→u8 WIDTH in place." rg's `m_window16` + `replaceMarkerBytes`
  IS that one buffer. Flattening is NOT a new divergence — it is the faithful unified
  decoder the memory mandates; the SegmentedU16+separate-ring is the current SHORTCUT. ✓
- **Architectural** (3-fn fuse + buffer flatten), not a one-liner. ✓
- DIS-19's RE-bucketing of `emit_backref_ring` into the OUTPUT-convergeable bucket
  (vs the prior verdict's symbol-rate/asm bucket) is MORE correct: back-ref resolution
  is DATA MOVEMENT (rg = memcpy), not Huffman — so it is faithfully convergeable
  WITHOUT the user-gated asm. Point to DIS-19.

**FIX-NEEDED (the load-bearing caveat):** instruction-count is a HYPOTHESIS
GENERATOR, never a wall verdict (CLAUDE.md Measurement PROCESS; the in-order
pipeline wall is not predictable from per-symbol attribution). TIE-6 already sized
the u16-footprint/page-warmth axis as wall-SLACK (−12% fault ceiling, TIE wall). So a
1.70e9 (~27% of 6.25e9) instruction reduction does NOT establish a wall move. Before
any port, size the de-frag's WALL contribution with a causal perturbation (a
flat-buffer A/B or a `GZIPPY_SLOW` knob on push_slice/emit_backref_ring + a
frequency-neutral control, removal-oracle to set the ceiling). Per rule 7, reject it
only with a mechanism; per rules 1/8, bank it only with a perturbation.

Verdict: **SOUND (faithful/architectural/largest-non-asm sub-lever; matches the
governing memory) / FIX-NEEDED (wall payoff unproven — perturbation owed before port).**

---

## BAR-1 READ — "converging the marker engine alone is unlikely to clear isal T4
0.91→0.99 (VAR_VIII precedent)" + "delete flip-to-clean, unify ISA-L-inside-marker?"
→ **SOUND with refinement; the unify/delete idea is REFUTED.**

- Sound for the SYMBOL-RATE half (~1.61e9, read_internal_compressed): VAR_VIII full
  register-pinned asm reached only 0.667x ISA-L (closes ~20% of the LLVM→ISA-L gap),
  and LEV-1 caps even a perfect engine at T4 0.900x. The marker-prefix symbol rate is
  asm-bounded and the asm is user-gated. Sound.
- **Refinement:** the OUTPUT-fragmentation half (CLAIM B, ~1.70e9) is ORTHOGONAL to
  the asm question (data movement, not Huffman) and was NOT touched by VAR_VIII. Do
  NOT fold it into the asm-bounded pessimism. Unlike the marker COMPUTE (which OPEN-1 /
  structural-residual-sizing found is a SHARED term, ~0% net gzippy-excess vs rg), the
  output-fragmentation is plausibly gzippy-SPECIFIC excess rg does NOT pay (rg's flat
  memcpy vs gzippy's segmented ring) — so de-fragging would converge toward rg, not
  OVERSHOOT it. That makes it the one un-refuted faithful candidate; it is owed a
  perturbation, not a TIE-based dismissal (rule 7).
- **The "delete flip-to-clean / unify on ISA-L-inside-marker like rg" move is
  REFUTED by CLAIM A's refutation:** rg does NOT decode the clean tail inside the
  marker engine — it flips OUT to `isal_inflate` at 32 KiB exactly like gzippy
  (`GzipChunk.hpp:520-525`). Deleting gzippy's flip-to-clean would DIVERGE from rg and
  lose the real-ISA-L clean-tail kernel both tools rely on. **KEEP flip-to-clean.**

Verdict: **SOUND (engine-alone pessimism for the symbol-rate half) / the unify-delete
proposal is REFUTED (would diverge; flip-to-clean is the convergent design).**

---

## DOES THIS COMPLETE THE ATTRIBUTION HONESTLY + NEXT MOVE

NOT yet honestly complete: its HEADLINE (gzippy flip-to-clean = divergence) is FALSE
and reverses a correct banked belief on a perf misread. Corrected completion:

- gzippy is structurally CONVERGENT with rapidgzip on the marker path: both
  marker-decode the ≤32 KiB window-absent prefix with a per-symbol Huffman primitive
  (gz pure-Rust u16 vs rg `HuffmanCodingISAL`), both FLIP the clean tail out to real
  ISA-L (`isal_inflate`) at 32 KiB. The "shared igzip kernel ≈ equal" leg proves the
  clean tail is already matched.
- The genuine residual is PER-BYTE on the matched marker-prefix fraction (consistent
  with the prior verdict's STEP-0 33.62% ≈ 34.50%), splitting into (i) marker-prefix
  SYMBOL RATE (u16 pure-Rust vs HuffmanCodingISAL — user-gated asm, bounded ~0.667x by
  VAR_VIII) and (ii) marker-output FRAGMENTATION (SegmentedU16+ring vs flat m_window16
  — faithful, untested, the new candidate).

**Single highest-value next move:** a CAUSAL PERTURBATION / removal-oracle that SIZES
the WALL contribution of the u16 output-fragmentation (flat-buffer A/B reusing the
parity contamination bar: interleaved N≥7, sha-verified, isal_chunks asserted, frozen
guest, vs rg 0.16.0; or a GZIPPY_SLOW knob on push_slice/emit_backref_ring +
frequency-neutral control). NOT a blind de-frag port (banking off the instruction
split violates rules 1/8), and NOT the unify-on-rg rewrite (CLAIM A refuted — gzippy
already matches rg's flip-to-clean). If the perturbation moves the wall, the de-frag
is a faithful, governing-memory-aligned lever; if it ties, reject only with a
mechanism (rule 7), not "Δ < spread."

---

## PER-CLAIM SCORECARD

| # | claim | verdict |
|---|-------|---------|
| A | rg marker-decodes every streaming chunk ISA-L-inside; gzippy flip-to-clean is DIVERGENT | **REFUTED** — vendor GzipChunk.hpp:520-525 + finishDecodeChunkWithInexactOffset<IsalInflateWrapper> → isal.hpp:302 isal_inflate: rg flips clean tail to separate real ISA-L at 32 KiB exactly like gzippy. CONVERGENT. (Sub-facts: top-level u8-direct is index-only ✓; HuffmanCodingISAL is the marker-prefix per-symbol primitive ✓ — but the inference is false.) |
| Split | 71/25/shared partition; shared kernel ≈ equal; double-count? | **SOUND (partition; no Block::read↔kernel double-count — kernel is a sibling) / FIX-NEEDED (confirm the 1.68e9 line is isal_inflate, NOT HuffmanCodingISAL::decode inside Block::read)** |
| B | ~1.70e9 u16 output fragmentation = largest faithful tractable lever; flat m_window16 is the proof | **SOUND (faithful/architectural; matches the governing ONE-MarkerRing memory; emit_backref_ring correctly re-bucketed as data-movement) / FIX-NEEDED (wall payoff unproven — TIE-6 sized u16 footprint as slack; perturbation owed)** |
| BAR-1 | marker convergence alone won't clear 0.91→0.99; unify-delete flip-to-clean? | **SOUND for symbol-rate half (VAR_VIII 0.667x / LEV-1 0.900x ceiling); the output-frag half is orthogonal+untested (size it). UNIFY/DELETE flip-to-clean is REFUTED (would diverge).** |
| Complete? | does this complete the isal attribution + next move | **Not honestly complete (false headline). Corrected: convergent marker path; residual is per-byte symbol-rate (asm-gated) + output-frag (faithful, untested). Next move = perturbation sizing the de-frag wall, NOT a port and NOT a unify-rewrite.** |

No edits to src/. No orphan processes (read-only audit). This is the synchronous
Opus gate DIS-19 requested.
