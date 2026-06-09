# MARKER-ATTRIBUTION GATE — INDEPENDENT DISPROOF ADVISOR VERDICT (READ-ONLY)

Independent adversarial Opus disproof of DIS-18 (the LOCATE-+40% per-symbol
attribution) + reconciliation against OPEN-1, STEP-0, and the banked charter
numbers. HEAD d56cb0f5 / origin tip 7bf26096, branch reimplement-isa-l. Every
load-bearing claim re-derived first-hand from source (vendor + gzippy file:line),
not from result prose. Posture: break the claim.

---

## ONE-LINE VERDICT

**DIS-18's per-symbol ATTRIBUTION of gzippy's OWN instructions is SOUND. Its
FALSIFIER and LEVER #1 are REFUTED: they rest on a premise about rapidgzip's
behavior ("rg decodes window-present chunks u8-direct via ISA-L; markers only for a
SMALL bootstrap fraction") that is FALSE in vendor source AND contradicts the
banked STEP-0 (33.62% ≈ 34.50%) and the charter (rg 73.1M ≈ gzippy 73.0M replaced
markers). rg marker-decodes the SAME ~34.5% of bytes. The +1.54e9 excess is NOT a
FRACTION gap (the thing lever #1 chases); it is PER-BYTE / structural-width
efficiency on a MATCHED byte fraction — DIS-18's demoted lever #2, which is the
user-gated inner-loop question plus a faithfully-convergeable u16-width/scaffold
slice.**

---

## CLAIM 1 — ATTRIBUTION SOUND? → **SOUND for gzippy's own breakdown; the
FALSIFIER step is FIX-NEEDED (compares against a phantom rg baseline).**

Per-symbol self-share, audited first-hand:

- The marker symbols are GENUINELY the pure-Rust u16 path, not lumped with clean.
  `read_internal_compressed` / `emit_backref_ring` are `const CONTAINS_MARKERS:
  bool` generics (`marker_inflate.rs:1299-1307,3351`) → TWO monomorphizations.
  `<true>` = u16 marker ring; `<false>` = clean u8-direct. On the **gzippy-ISAL**
  binary the clean tail is REAL ISA-L FFI (DIS-12/DIS-13, source-verified:
  FlipToClean → `finish_decode_chunk_isal_oracle` → `decompress_deflate_from_bit_into`),
  so the `<false>` pure-Rust clean path does NOT execute in production. ⇒ the
  19.57% `read_internal_compressed` + 15.89% `emit_backref_ring` are the `<true>`
  marker monomorphization — marker-specific, NOT a clean-path mis-attribution. Good.
- `SegmentedU16::push_slice` 11.29% is the u16 marker-output write (corroborated
  independently by the page-warmth finding: push_slice = 44.5% of page-faults, the
  u16 buffer). Marker-specific. Good.
- "ISA-L ~30% shared with rg" is FAIR and the strongest leg: the STEP-0 Level-2
  disasm (ledger:407) proved gzippy and rg link the byte-identical AVX2/BMI2
  `igzip_decode_block_stateless_04` kernel. Same kernel, same clean bytes ⇒ ~equal
  ISA-L instructions. No objection.
- Double-count / inlining risk: LOW. The inline-tree is disjoint
  (decode_chunk_unified_marker → … → read_internal_compressed → emit_backref_ring →
  push_slice), self-share not inclusive. One residual risk worth a footnote: a
  symboled fat-LTO build can fold a `<true>`/`<false>` pair under one demangled
  name; here the `<false>` clean path is dead on isal so the lump is harmless.

**Where it breaks — the FALSIFIER ("located fns sum to ~the 1.54e9 excess"):** this
does NOT pass on its own terms. It "passes" only by ASSUMING rg's non-ISA-L work is
~rg-scaffold and **rg-marker ≈ 0** — i.e. by treating gzippy's whole 3.56e9 marker
engine as excess. The arithmetic: gz 6.252e9 = ISA-L 1.88e9 + marker 3.56e9 +
scaffold 0.6e9; rg 4.708e9 = ISA-L 1.88e9 + **rg-other 2.83e9**. The +1.54e9 excess
= gz-(marker+scaffold) − rg-(marker+scaffold). rg's 2.83e9 non-ISA-L INCLUDES rg's
own marker decode of its 34.5% bytes. DIS-18 never measured rg's marker-engine
instruction count — it ASSUMED it small (the explicit parenthetical "rg's 31.25%
replaced-marker symbols is over a SMALL chunk fraction"). That assumption is
refuted (Claim 4 + vendor). So the falsifier maps the excess to gzippy's marker
TOTAL when it should map it to the gz-minus-rg marker DIFFERENCE. **The number
3.56e9 is real; the inference "≈ the excess" is not.**

---

## CLAIM 2 — RECONCILE vs OPEN-1 ("placement is DEAD"): are boundaries-vs-windows
DIFFERENT levers, so OPEN-1 does not refute window-propagation? → **PARTIALLY YES
(different mechanisms), but window-propagation is refuted by a DIFFERENT proof
(DIS-6 + STEP-0), and OPEN-1's seed-all leg already tested it and showed it is an
uncounted cheat.**

The prompt's distinction is CORRECT and fair: OPEN-1's `placement-marker` leg
(`GZIPPY_SEED_NO_WINDOWS=1`) granted rg-grade BOUNDARIES while SUPPRESSING the
window-clean upgrade — it tested boundary-placement WITHOUT window-propagation, and
got −34ms (TIE-to-worse). That leg does NOT speak to window-propagation. So
"placement is dead" ≠ "window-propagation is dead." Granted.

BUT OPEN-1 ALSO ran the WITH-windows leg (seed-all, isal_chunks 14→17): 318ms =
1.55x rg. That IS window-propagation, and OPEN-1's own verdict already disposed of
it: (a) it is an UNCOUNTED p=1 pre-pass handing gzippy precomputed predecessor
windows — a masks-binder OVER-removal, explicitly flagged as an UPPER BOUND not a
reachable target; (b) the disentanglement isolated the seed-all win as the
engine-swap on the +3 prefix chunks, NOT a schedulable placement slice. So
window-propagation is NOT an untested new direction — OPEN-1 measured its ceiling
(1.55x) and showed the realization requires free precomputed windows.

The decisive independent refutation is **DIS-6** (DEAD, mechanism-backed):
consumer-confirmation prefetch CANNOT get the predecessor window to the worker
before it decodes — the ~1-chunk lead is too short; the decode is IN-FLIGHT-NOT-DONE
when the consumer arrives. The predecessor window only exists AFTER the predecessor
finishes; delivering it early = serializing the decode = destroying the parallelism
that is the entire point. So DIS-18 lever #1 ("get the predecessor window to the
worker before it decodes") is the DIS-6-refuted direction restated.

---

## CLAIM 3 — RECONCILE vs the OPEN-1 "seed-all = the gated native asm in disguise"
dismissal: a seeded chunk decodes via REAL ISA-L FFI (not native VAR_VIII asm), so
is getting more chunks to ISA-L FFI a FAITHFUL, isal-owner-turnable lever OPEN-1
WRONGLY fenced? → **The LABEL is imprecise (FIX-NEEDED) but the CONCLUSION (not a
faithful runtime-reachable lever) is CORRECT for a different reason.**

The prompt is RIGHT on the mechanism: on the gzippy-ISAL build a window-seeded
chunk decodes via REAL ISA-L FFI (DIS-12/13: `finish_decode_chunk_isal_oracle`),
NOT the gated native pure-Rust VAR_VIII asm. So OPEN-1's "the gated native asm in
disguise" wording is technically wrong — no native asm is involved; it is the FFI
the isal build already has. Score that correction to the prompt.

But "turnable" requires getting the window to the worker before decode, and that is
gated by a REAL constraint, not a labeling confusion:
1. The predecessor window is the predecessor's last 32 KiB of DECODED output. It
   does not exist until the predecessor decodes. Early delivery ⇒ serialize ⇒ kill
   parallelism (DIS-6; and CLAUDE.md "what doesn't work: speculative parallel …
   sequential re-decodes").
2. **Vendor proves rg does NOT do this either.** GzipChunk.hpp:671
   `if ( initialWindow && untilOffsetIsExact )` is the ONLY branch that takes the
   ISA-L u8-direct `decodeChunkWithInflateWrapper`. It requires BOTH a resolved
   window AND an exact offset — i.e. a chunk whose predecessor is already resolved.
   The speculative-parallel case (no resolved window, block-finder-guessed offset)
   falls through to `decodeChunkWithRapidgzip` (:705-710), the marker-producing
   engine. So rg ALSO marker-decodes its speculative chunks; it does not magically
   ISA-L-u8-direct them. Converging to rg means matching rg's MARKER decode, NOT
   eliminating it.

So: the seed-all win is real but unreachable at runtime (free precomputed windows),
and faithfully it CANNOT be turned beyond what gzippy already does. OPEN-1's verdict
stands; only its "native asm" phrasing needs the FFI correction.

---

## CLAIM 4 — THE CRUX: is the lever FRACTION (propagate windows, fewer marker
chunks) or PER-BYTE marker efficiency? → **PER-BYTE / structural. STEP-0's
33.62% ≈ 34.50% REFUTES "gzippy marker-decodes far more than rg." This is the
single most important correction to DIS-18.**

Source chain, first-hand:
- STEP-0 (converge-bootstrap-divergence.md:14-18): "gzippy 33.62% window-absent
  marker bytes vs rg 34.50% — gzippy marginally AHEAD … the flip-to-clean threshold
  is a byte-for-byte vendor port. So the 56ms is NOT 'more window-absent chunks.'"
- Charter, repeatedly: rg 73.1M ≈ gzippy 73.0M replaced markers; parallel-sm-model
  .md:49 "rg 31.25% … gzippy 31.97% — they MATCH."
- Vendor (Claim 3): rg's speculative chunks go through the marker engine; both
  gzippy and rg flip mid-chunk to ISA-L once 32 KiB of own output exists
  (GzipChunk.hpp:521 `cleanDataCount >= MAX_WINDOW_SIZE` ↔ gzippy FlipToClean). The
  marker PREFIX (≤32 KiB/chunk) is window-absent for BOTH.

⇒ The byte fractions are matched. If gzippy marker-decoded a fraction rg decodes
u8-direct (DIS-18's mechanism), gzippy's window-absent fraction would EXCEED rg's.
It does not — it is marginally LOWER. So DIS-18's mechanism is falsified by its own
campaign's banked budget. The +1.54e9 excess is the instruction DIFFERENCE on the
SAME ~34.5% bytes (plus scaffold/width) = **per-byte efficiency**, not fraction.

This SPLITS the 3.56e9 marker engine into two convergence classes:
- **Inner-loop symbol rate** (read_internal_compressed 19.57 + emit_backref_ring
  15.89 + HuffmanShortBitsCached 2.91 ≈ 38% ≈ 2.4e9): the pure-Rust u16 symbol
  decode vs rg's C++ marker inflate. This is the USER-GATED inner-loop asm
  question. NOTE the unresolved internal contradiction the gate must own: TIE-1
  ("marker loop FASTER than rg already") and CAMPAIGN-CHARTER:409 ("gzippy marker
  0.68x = FASTER") say gzippy's marker decode is NOT slower per byte, which would
  put the excess OUTSIDE the inner loop; CHARTER:517 says the opposite (rg ~2x
  faster). These are decodeBlock-SUM attributions (CLAUDE.md rule-8 phantom risk),
  not perf-stat. DIS-18 did not resolve this because it never counted rg's marker
  instructions.
- **u16 WIDTH + scaffold + resolve** (push_slice 11.29 + finalize_with_deflate 7.91
  + resolve_chunk_markers 3.52 ≈ 1.1-1.4e9): data-structure work, NOT the inner
  symbol loop. rg's counterpart is applyWindow narrowing (MarkerReplacement.hpp;
  DecodedData.hpp). This slice is potentially FAITHFULLY CONVERGEABLE without the
  gated asm — though page-warmth already sized the u16-footprint axis as wall-slack
  (−12% fault ceiling, TIE wall), so treat its WALL payoff as unproven.

---

## CLAIM 5 — THE VERDICT: real BAR-1 lever = faithful window-propagation, OR the
harder per-byte engine? → **NEITHER as DIS-18 frames it. Window-propagation is
refuted (Claims 2-4). The residual is per-byte engine + a u16-width/scaffold slice.
The honest next move is a MEASUREMENT, not a port.**

DIS-18's ranked CONVERGENCE TARGET #1 ("shrink the marker-decoded FRACTION via
window-propagation") should be **STRUCK** — it chases a fraction gap STEP-0 proved
absent, restates the DIS-6-dead direction, and is non-faithful (rg marker-decodes
the same fraction; vendor GzipChunk.hpp:671).

The genuinely-owed next deliverable, before ANY port or asm authorization:

> **Measure rapidgzip's window-absent (marker) decode INSTRUCTION COUNT per byte,
> apples-to-apples vs gzippy's, on the matched ~34.5% fraction.** DIS-18 counted
> gzippy's symbols and ASSUMED rg-marker ≈ 0 (the parenthetical "small chunk
> fraction"). Both the FALSIFIER (Claim 1) and the inner-loop-vs-width split
> (Claim 4) hinge on rg's marker instruction count, which is UNMEASURED. With rg's
> .so exposing `decode_huffman_code_block_stateless_*` (disasm turn) and the
> rapidgzip marker path being `decodeChunkWithRapidgzip`, a per-symbol perf-record
> on rg over the window-absent chunks is feasible and is the decisive
> discriminator between "gzippy's marker inner loop is heavier" (→ inner-loop asm,
> user-gated) and "gzippy's u16-width/scaffold is heavier" (→ faithful
> data-structure convergence, owner-turnable).

Until that runs, the BAR-1 low-T story remains: even a perfect engine = ISA-L loses
T1 0.899x / T4 0.900x (LEV-1, removal-oracle), so the engine alone is necessary-
not-sufficient; and the marker per-byte excess located by DIS-18 is real but its
SIZE-vs-rg and its inner-loop-vs-width split are unmeasured. Do not authorize a port
off DIS-18's fraction framing.

---

## PER-CLAIM SCORECARD

| # | claim | verdict |
|---|-------|---------|
| 1 | marker engine ~57% / 3.56e9 attribution; ISA-L ~30% shared | **SOUND** (own breakdown) / **FIX-NEEDED** (falsifier compares vs phantom rg-marker≈0) |
| 2 | OPEN-1 refutes window-propagation? boundaries vs windows distinct? | **distinct mechanisms (prompt correct)**, but window-propagation **REFUTED** independently by DIS-6 + seed-all-is-uncounted-cheat |
| 3 | seed-all = "gated native asm in disguise" wrong; ISA-L-FFI turnable? | label **FIX-NEEDED** (it is FFI not native asm), conclusion **SOUND** (not runtime-reachable; vendor GzipChunk.hpp:671 proves rg doesn't do it either) |
| 4 | FRACTION vs per-byte | **REFUTED for fraction; per-byte/width is the lever** (STEP-0 33.62%≈34.50% + vendor) — the crux correction |
| 5 | lever = window-propagation or harder per-byte engine | **window-propagation REFUTED; per-byte engine + width slice; next move is to MEASURE rg's marker instr/byte, not port** |

No edits to src/. No orphan processes (read-only audit). Owes nothing further; this
IS the synchronous Opus gate DIS-18 requested.
