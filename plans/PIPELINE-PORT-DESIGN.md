# PIPELINE-PORT-DESIGN — converging gz's marker machinery to rapidgzip

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz git:** 04ef912d
**Pairs with:** `plans/PIPELINE-TAX-LOCALIZE-2026-06-21.md` (the located gap).
**Scope:** macOS-aarch64 localization (tax is coordination/marker-bound, ~arch-
independent → the component RANKING transfers). x86/AMD owed for LAW.

## The located gap (from D1)
- **CAUSAL:** 91-97% of gz's parallel pipeline tax is the speculative window-absent
  **marker machinery** (NO_PREFETCH removal collapses it to the clean floor).
  Coordination/prefetch/block-find is ≤9% of the tax — NOT the lever.
- **ATTRIBUTION (self-time + rg --verbose):** within the marker machinery the
  **marker-DECODE loop dominates** (block_body 46-65% of work); `apply_window`
  (resolve+narrow) is secondary (5-19%); block-find minor.
- rg has the SAME component shape (decode-dominated). So the differential is
  per-symbol/per-pass MAGNITUDE in the marker path, not a missing/extra component.

## Component → rg counterpart → convergence status

| gz component | gz file | rg counterpart | rg file:line | status |
|---|---|---|---|---|
| marker DECODE loop | `marker_inflate.rs` `decode_marker_fast_loop` (via `read_internal_compressed_specialized<true>`) | `Block::readInternalCompressedMultiCached<CONTAINS_MARKERS=true>` | `gzip/deflate.hpp:1589-1666` | **PRIMARY TARGET** (heavy, mechanism not yet causally pinned) |
| mid-chunk flip-to-clean | `marker_inflate.rs` ring flip; `chunk_decode.rs` flip_to_clean | `m_containsMarkerBytes` flip (`distanceToLastMarkerByte + uncompressedSize >= MAX_WINDOW_SIZE`) | `gzip/deflate.hpp:1220-1259, 1744-1784` | PORTED (verify trigger parity — see §2) |
| used-window-symbol minimization | `used_window_symbols.rs` | `getUsedWindowSymbols` | `gzip/deflate.hpp:1846-1988` | PORTED (verify it actually fires on the prefetch path) |
| marker resolve + u16→u8 narrow | `apply_window.rs` / `replace_markers.rs` (`replace_markers_lut_narrow`) | `DecodedData::applyWindow` (in-place, buffer-reuse, zero-copy views) | `DecodedData.hpp:325-388` | **ALREADY CONVERGED** (in-place single pass; `GZIPPY_VERBOSE` `fused_lut`, `"in_place":true`) — NOT the target |
| block-find / speculation scan | `chunk_fetcher.rs` `with_sync_boundary_search` + `blockfinder_validation.rs` | `GzipBlockFinder` + `blockfinder/DynamicHuffman.hpp` | `GzipBlockFinder.hpp`, `blockfinder/` | minor (≤9% of tax); not now |

## §1 — PRIMARY TARGET: the marker-mode decode loop
**WHAT rg does:** `readInternalCompressedMultiCached<true>` (deflate.hpp:1589-1666) is
the SAME templated multi-symbol cached loop as the clean (`<false>`) variant — markers
differ only in how a back-reference into the unresolved window is emitted (a u16
marker value vs a u8 copy). rg's marker overhead over its own clean decode is only
1.2-1.8× (the rg `marker% → infl` table).

**WHAT gz does:** gz's `decode_marker_fast_loop` is also templated
(`read_internal_compressed_specialized<CONTAINS_MARKERS>`), BUT its marker overhead
over gz's own clean decode is 1.8-4.1× (the located tax). The marker-mode decode is
the dominant self-time component (46-65%).

**WHY it could be heavier (HYPOTHESES — NOT yet causally split):**
1. **per-marker decode cost:** the marker fast-loop may not retain the clean loop's
   full multi-symbol packed-write fast path (the `litpack`/2-3-literal packed-u64
   store, dist cache) — i.e. it may drop to the careful/per-symbol path more often
   under `CONTAINS_MARKERS=true`. (deflate.hpp:1589-1666 keeps the multi-cached loop
   for BOTH; gz must too.)
2. **symbol-count:** gz may emit markers for MORE symbols than rg (flip-to-clean fires
   later, or `getUsedWindowSymbols` doesn't prune on the prefetch path). The flip
   trigger (deflate.hpp:1220) is `distanceToLastMarkerByte + uncompressedSize >=
   MAX_WINDOW_SIZE` — gz must use the identical predicate and reset
   `distanceToLastMarkerByte` on exactly the same events.
3. **u16 width amortization:** if gz's marker back-ref copy (`emit_backref_ring` u16)
   is narrower-throughput than rg's, the 2-byte copy dominates on long-match corpora
   (matches nasa being worst).

**HOW to converge (byte-exact plan):**
- **STEP 0 (REQUIRED FIRST — build the discriminator):** the current decode-loop
  attribution is TIME-based + perturbed. Build a **marker-decode-ISOLATING oracle**
  (byte-transparent counter, no decode-path semantics change): instrument
  `decode_marker_fast_loop` to count, per chunk, (a) symbols emitted as markers vs
  clean, (b) careful-tail entries vs fast-loop iterations, (c) u16 back-ref copy bytes.
  Run gz and a `--verbose`-traced rg side-by-side on the 4 corpora. This causally
  splits hypothesis 1 vs 2 vs 3 BEFORE any code change (CLAUDE.md Gate-2: locate the
  mechanism, don't pre-judge). Pre-register: the winning hypothesis is the one whose
  per-chunk metric ratio (gz/rg) ≈ the per-chunk instruction tax ratio.
- **STEP 1:** port the SPECIFIC rg structure the discriminator implicates — e.g. if
  H1: lift the clean loop's multi-symbol packed path into the marker template
  (deflate.hpp:1589-1666 is ONE function over `CONTAINS_MARKERS`); if H2: align the
  flip predicate + `getUsedWindowSymbols` pruning (deflate.hpp:1220-1259, 1846-1988);
  if H3: widen the marker back-ref copy throughput.
- **GATE:** byte-exact (silesia+monorepo+nasa+squishy sha == gzip -d) +
  `mac_pipeline_components.sh` shows base4 instr drop with base1 unchanged + the
  per-chunk discriminator metric moves toward rg + N≥7 sig/spread ≥10× + replicate on
  x86/AMD before LAW.

## §2 — SECONDARY (verify, low cost): flip-to-clean trigger parity
gz's `clean_flipped_bytes` is 2.9% (monorepo) / 0.0% (nasa) per `GZIPPY_VERBOSE`. The
flip predicate is ported (deflate.hpp:1220) but its FIRING RATE on the production
prefetch path is unverified. If gz flips later than rg, more of each chunk decodes as
markers. CHEAP CHECK (no code change): add the marker/clean per-chunk counter from
STEP 0 and compare the flip offset distribution to a `--verbose` rg trace. Only port
if the discriminator shows gz flips materially later.

## §3 — apply_window: NO ACTION (already converged)
`apply_window.rs` + `replace_markers_lut_narrow` already resolve markers AND narrow
u16→u8 in a SINGLE in-place pass reusing the u16 buffer as u8 (chunk_data.rs:207-211
cites `DecodedData.hpp:325-388`; verbose `Post-process path: fused_lut`,
`"in_place":true`). gz's apply_window self-time (5-19%) ≈ rg's apply-window phase
(4-19%). Touching this is not the lever.

## Archive-type gate — NOT needed (one port closes all types)
The tax tracks the MARKER FRACTION (nasa 90% → infl 3.94 worst; squishy 32% → 1.80
mildest). But the heavy component (marker-decode loop) is the SAME path on every type
— a lighter marker loop reduces the tax PROPORTIONALLY to the marker fraction, so it
helps high-marker corpora MOST without any type-specific branch. No detectable-signal
gate (ratio/block-size/entropy) is required; the single decode-loop convergence covers
all types. (Revisit only if the discriminator shows a type-specific failure mode, e.g.
long-match corpora hitting the careful tail.)

## Owed before LAW
- STEP-0 marker-decode discriminator oracle (converts the decode-loop attribution to
  causal; pins H1/H2/H3).
- x86 (rg ISA-L) + AMD/Zen2 replication of the component ranking and any port win.
- Honesty carry-over (from D1): on aarch64 gz is already LIGHTER absolute than rg; the
  lever is the marker machinery's ABSOLUTE added instructions (6.9-14.3 instr/B), which
  is the arch-independent term and the one that closes the x86 ISA-L gap.
